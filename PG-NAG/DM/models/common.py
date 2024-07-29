# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, radius
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torch_sparse import coalesce
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np

from utils.chem import BOND_TYPES


class MeanReadout(nn.Module):

    def forward(self, data, input):
        output = scatter_mean(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output


class SumReadout(nn.Module):

    def forward(self, data, input):
        output = scatter_add(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output



class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

    def forward(self, input):
        """"""
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


def assemble_atom_pair_feature(node_attr, edge_index, edge_attr):
    h_row, h_col = node_attr[edge_index[0]], node_attr[edge_index[1]]
    h_pair = torch.cat([h_row*h_col, edge_attr], dim=-1)    # (E, 2H)
    return h_pair
    

def generate_symmetric_edge_noise(num_nodes_per_graph, edge_index, edge2graph, device):
    num_cum_nodes = num_nodes_per_graph.cumsum(0)  # (G, )
    node_offset = num_cum_nodes - num_nodes_per_graph  # (G, )
    edge_offset = node_offset[edge2graph]  # (E, )

    num_nodes_square = num_nodes_per_graph**2  # (G, )
    num_nodes_square_cumsum = num_nodes_square.cumsum(-1)  # (G, )
    edge_start = num_nodes_square_cumsum - num_nodes_square  # (G, )
    edge_start = edge_start[edge2graph]

    all_len = num_nodes_square_cumsum[-1]

    node_index = edge_index.t() - edge_offset.unsqueeze(-1)
    node_large = node_index.max(dim=-1)[0]
    node_small = node_index.min(dim=-1)[0]
    undirected_edge_id = node_large * (node_large + 1) + node_small + edge_start

    symm_noise = torch.zeros(size=[all_len.item()], device=device)
    symm_noise.normal_()
    d_noise = symm_noise[undirected_edge_id].unsqueeze(-1)  # (E, 1)
    return d_noise


def _extend_graph_order(num_nodes, edge_index, edge_type, order=3):

    def binarize(x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(adj, order):
        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                    binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

        for i in range(2, order+1):
            adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order+1):
            order_mat += (adj_mats[i] - adj_mats[i-1]) * i

        return order_mat

    num_types = len(BOND_TYPES)

    N = num_nodes
    adj = to_dense_adj(edge_index).squeeze(0)
    adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

    type_mat = to_dense_adj(edge_index, edge_attr=edge_type).squeeze(0)   # (N, N)
    type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
    assert (type_mat * type_highorder == 0).all()
    type_new = type_mat + type_highorder

    new_edge_index, new_edge_type = dense_to_sparse(type_new)
    _, edge_order = dense_to_sparse(adj_order)


    new_edge_index, new_edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N)

    return new_edge_index, new_edge_type
    

def _extend_to_radius_graph(pos, edge_index, edge_type, cutoff, batch, unspecified_type_number=0, is_sidechain=None):

    assert edge_type.dim() == 1
    N = pos.size(0)

    bgraph_adj = torch.sparse.LongTensor(
        edge_index, 
        edge_type, 
        torch.Size([N, N])
    )

    if is_sidechain is None:
        rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)    # (2, E_r)
    else:
        is_sidechain = is_sidechain.bool()
        dummy_index = torch.arange(pos.size(0), device=pos.device)
        sidechain_pos = pos[is_sidechain]
        sidechain_index = dummy_index[is_sidechain]
        sidechain_batch = batch[is_sidechain]

        assign_index = radius(x=pos, y=sidechain_pos, r=cutoff, batch_x=batch, batch_y=sidechain_batch)
        r_edge_index_x = assign_index[1]
        r_edge_index_y = assign_index[0]
        r_edge_index_y = sidechain_index[r_edge_index_y]

        rgraph_edge_index1 = torch.stack((r_edge_index_x, r_edge_index_y)) # (2, E)
        rgraph_edge_index2 = torch.stack((r_edge_index_y, r_edge_index_x)) # (2, E)
        rgraph_edge_index = torch.cat((rgraph_edge_index1, rgraph_edge_index2), dim=-1) # (2, 2E)
        # delete self loop
        rgraph_edge_index = rgraph_edge_index[:, (rgraph_edge_index[0] != rgraph_edge_index[1])]

    rgraph_adj = torch.sparse.LongTensor(
        rgraph_edge_index, 
        torch.ones(rgraph_edge_index.size(1)).long().to(pos.device) * unspecified_type_number,
        torch.Size([N, N])
    )

    composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)


    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()
    
    return new_edge_index, new_edge_type


def extend_graph_order_radius(edge_index,edge_type, batch, order=3, cutoff=10.0,
                              extend_order=True, extend_radius=True, is_sidechain=None):
    
    if extend_order:
        edge_index, edge_type = _extend_graph_order(
            num_nodes=num_nodes, 
            edge_index=edge_index, 
            edge_type=edge_type, order=order
        )


    if extend_radius:
        edge_index, edge_type = _extend_to_radius_graph(
            pos=pos, 
            edge_index=edge_index, 
            edge_type=edge_type, 
            cutoff=cutoff, 
            batch=batch,
            is_sidechain=is_sidechain

        )
    
    return edge_index,edge_type

def coarse_grain(pos, node_attr, subgraph_index, batch):
    cluster_pos = scatter_mean(pos, index=subgraph_index, dim=0)    # (num_clusters, 3)
    cluster_attr = scatter_add(node_attr, index=subgraph_index, dim=0)  # (num_clusters, H)
    cluster_batch, _ = scatter_max(batch, index=subgraph_index, dim=0) # (num_clusters, )

    return cluster_pos, cluster_attr, cluster_batch


def batch_to_natoms(batch):
    return scatter_add(torch.ones_like(batch), index=batch, dim=0)


def get_complete_graph(natoms):

    natoms_sqr = (natoms ** 2).long()
    num_atom_pairs = torch.sum(natoms_sqr)
    natoms_expand = torch.repeat_interleave(natoms, natoms_sqr)

    index_offset = torch.cumsum(natoms, dim=0) - natoms
    index_offset_expand = torch.repeat_interleave(index_offset, natoms_sqr)

    index_sqr_offset = torch.cumsum(natoms_sqr, dim=0) - natoms_sqr
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, natoms_sqr)

    atom_count_sqr = torch.arange(num_atom_pairs, device=num_atom_pairs.device) - index_sqr_offset

    index1 = (atom_count_sqr // natoms_expand).long() + index_offset_expand
    index2 = (atom_count_sqr % natoms_expand).long() + index_offset_expand
    edge_index = torch.cat([index1.view(1, -1), index2.view(1, -1)])
    mask = torch.logical_not(index1 == index2)
    edge_index = edge_index[:, mask]

    num_edges = natoms_sqr - natoms # Number of edges per graph

    return edge_index, num_edges

def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x

def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)

def cosine_beta_schedule(timesteps, s=0.008):

    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)


    alphas = np.sqrt(alphas)
    return alphas

def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    sample_onehot = F.one_hot(sample_index, 5)

    return sample_onehot

class DiscreteTransition(nn.Module):
    def __init__(self, noise_schedule, num_timesteps, s, num_classes, prior_probs=None):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_classes = 6

        if noise_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, s)
            print('cosine v alpha schedule applied!')
        else:
            raise NotImplementedError
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))
        if prior_probs is None:#先验概率分布
            uniform_probs = -np.log(6).repeat(5)[None, :]  # (1, num_classes)

            self.prior_probs = to_torch_const(uniform_probs)
        else:

            log_probs = np.log(prior_probs.clip(min=1e-30))

            self.prior_probs = to_torch_const(log_probs)


    # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)

        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        print("aa", log_vt_1+log_alpha_t)# 7
        print("bb", log_1_min_alpha_t+self.prior_probs)
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t + self.prior_probs
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0) = alpha_bar_t * log_v0 + (1 - alpha_bar_t) * (prior_prob)
        # log_v0: (N, num_classes)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        # print('1',log_v0)
        print('log_v0',log_v0)
        print('4', self.prior_probs)
        print('11', log_v0 + log_cumprod_alpha_t)
        print('22', log_1_min_cumprod_alpha +  self.prior_probs)
        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha + self.prior_probs
        )
        return log_probs

    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        print('log_qvt_v0',log_qvt_v0)
        sample_index = log_sample_categorical(log_qvt_v0)
        print('sample',sample_index)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        print('log_sample',log_sample)
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        print('log_v0',log_v0)
        # print('t_minus',t_minus_1)

        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t_minus_1, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v,t_minus_1,batch)
        prob = self.prior_probs[0][0]
        probs = prob.expand(1,5)
        log_qvt1_v0 = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha +probs
        )
        # log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        log_alpha_t = extract(self.log_alphas_v,t,batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v,t,batch)
        unnormed_logprobs = log_add_exp(
            log_vt + log_alpha_t,
            log_1_min_alpha_t + probs
        )
        unnormed_logprobs =  unnormed_logprobs + log_qvt1_v0
        # unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, fix_offset=True):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        if fix_offset:
            offset = torch.tensor([0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])
            self.num_gaussians = 20
        else:
            offset = torch.linspace(start, stop, num_gaussians)
            self.num_gaussians = num_gaussians
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def __repr__(self):
        return f'GaussianSmearing(start={self.start}, stop={self.stop}, num_gaussians={self.num_gaussians})'

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

def index_to_log_onehot(x, num_classes):
    log_x = torch.log(x.clamp(min=1e-30))
    return log_x

def find_index_after_sorting(size_all, size_p, size_l, sort_idx, device):
    # find protein/ligand index in ctx
    ligand_index_in_ctx = torch.zeros(size_all, device=device)
    ligand_index_in_ctx[size_p:size_p + size_l] = torch.arange(1, size_l + 1, device=device)
    ligand_index_in_ctx = torch.sort(ligand_index_in_ctx[sort_idx], stable=True).indices[-size_l:]
    ligand_index_in_ctx = ligand_index_in_ctx.to(device)

    protein_index_in_ctx = torch.zeros(size_all, device=device)
    protein_index_in_ctx[:size_p] = torch.arange(1, size_p + 1, device=device)
    protein_index_in_ctx = torch.sort(protein_index_in_ctx[sort_idx], stable=True).indices[-size_p:]
    protein_index_in_ctx = protein_index_in_ctx.to(device)
    return protein_index_in_ctx, ligand_index_in_ctx

def compose_context(h_arch, h_node, edge, edge_index, batch):

    batch_ctx = batch
    sort_idx = torch.sort(batch_ctx, stable=True).indices
    mask_node = torch.cat([
        torch.zeros([batch_arch.size(0)],device=batch_arch.device).bool(),
        torch.ones([batch_node.size(0)], device=batch_node.device).bool(),
    ], dim=0)[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_arch, h_node], dim=0)[sort_idx]
    arch_index_in_ctx, node_index_in_ctx = find_index_after_sorting(
        len(h_ctx), len(h_arch), len(h_node), sort_idx, batch_arch.device
    )
    return h_ctx, batch_ctx, mask_node, arch_index_in_ctx, node_index_in_ctx

