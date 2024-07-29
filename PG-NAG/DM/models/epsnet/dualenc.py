import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
import numpy as np
from numpy import pi as PI
from tqdm.auto import tqdm
import dgl
from dgl.nn import GraphConv
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.encoder import get_refine_net

from utils.chem import BOND_TYPES
from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, generate_symmetric_edge_noise, \
    extend_graph_order_radius, \
    to_torch_const, extract, DiscreteTransition, \
    ShiftedSoftplus, GaussianSmearing, index_to_log_onehot, compose_context, log_sample_categorical
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder
from ..geometry import get_distance, get_angle, get_dihedral, eq_transform
# from diffusion import get_timestep_embedding, get_beta_schedule
import pdb



def onehot(featurelist):
    features = {
        'input': [0, 1, 0, 0, 0],
        'maxpool3x3': [0, 0, 1, 0, 0],
        'conv3x3-bn-relu': [1, 0, 0, 0, 0],
        'conv1x1-bn-relu': [0, 0, 0, 0, 1],
        'output': [0, 0, 0, 1, 0],
        'null': [0, 0, 0, 0, 0]
    }
    feature_matrix = []
    for feature in featurelist:
        if feature in features:
            feature_matrix.append(features[feature])
        else:
            feature_matrix.append([0, 0, 0, 0, 0])
    feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32)
    return feature_matrix


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(meanl, logvar1, mean2, logvar2):
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) - (meanl - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb




def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


MAX_FEATURES = 6

class MlpPredictor(nn.Module):
    def __init__(self, num_node=5, hidden_dim=64, output_dim=5):
        super(MlpPredictor, self).__init__()
        self.fc1 = nn.Linear(5, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        print(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class Encoder(nn.Module):
    def __init__(self, num_node=6, hidden_dim=64, feature=50):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(num_node, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, feature)

    def forward(self, node_features, edge_index):
        # print(node_features.shape)
        # print(edge_index.shape)
        h = self.conv1(node_features, edge_index)
        h = torch.relu(h)
        h = self.conv2(h, edge_index)
        return h


class GinEncoder(nn.Module):
    def __init__(self, inter_len=64, feature_len=50):
        super(GinEncoder, self).__init__()
        self.conv1 = GraphConv(MAX_FEATURES, inter_len)
        self.conv2 = GraphConv(inter_len, feature_len)
        self.relu = nn.ReLU(True)

    def forward(self, g, input_data):
        h = self.conv1(g, input_data)
        h = self.relu(h)
        h = self.conv2(g, h)
        h = self.relu(h)
        return h


class DualEncoderEpsNetwork(nn.Module):

    def __init__(self, config, prior_node_type=None, prior_edge_type=None):
        super().__init__()
        self.config = config

        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        # self.edge_encoder_global = get_edge_encoder(config)
        # self.edge_encoder_local = get_edge_encoder(config)
        # self.hidden_dim = config.hidden_dim
        '''
        timestep embedding
        '''
        # self.temb = nn.Module()
        # self.temb.dense = nn.ModuleList([
        #     torch.nn.Linear(config.hidden_dim,
        #                     config.hidden_dim*4),
        #     torch.nn.Linear(config.hidden_dim*4,
        #                     config.hidden_dim*4),
        # ])
        # self.temb_proj = torch.nn.Linear(config.hidden_dim*4,
        #                                  config.hidden_dim)
        """
        The graph neural network that extracts node-wise features.
        """
        self.encoder_global = Encoder(num_node=6, hidden_dim=64, feature=50)
        # self.encoder_global = SchNetEncoder(
        #     hidden_channels=config.hidden_dim,
        #     num_filters=config.hidden_dim,
        #     num_interactions=config.num_convs,
        #     edge_channels=self.edge_encoder_global.out_channels,
        #     cutoff=config.cutoff,
        #     smooth=config.smooth_conv,
        # )
        # self.encoder_local = GINEncoder(
        #     hidden_dim=config.hidden_dim,
        #     num_convs=config.num_convs_local,
        # )
        self.encoder_local = GinEncoder(inter_len=config.inter_l, feature_len=config.feature_l)

        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs 
            gradients w.r.t. edge_length (out_dim = 1).
        """
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=config.mlp_act
        )

        self.grad_local_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=config.mlp_act
        )

        '''
        Incorporate parameters together
        '''
        self.model_global = nn.ModuleList([self.encoder_global, self.grad_global_dist_mlp])
        self.model_local = nn.ModuleList([self.encoder_local, self.grad_local_dist_mlp])

        self.model_type = config.type  # config.type  # 'diffusion'; 'dsm'
        if self.model_type == 'diffusion':
            # denoising diffusion
            ## betas
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            ## variances
            alphas = (1. - betas)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)
        self.num_timesteps = self.betas.size(0)
        self.sample_time_method = config.sample_time_method

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        # self.posterior_logvar = to_torch_const(np.log(np.maximum(posterior_variance, 1e-10)))
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))
        self.pos_score_coef = to_torch_const(betas / np.sqrt(alphas))

        # atom/bond(node/edge) type trainsition
        self.num_classes = 5
        self.num_edge_classes = 6

        self.node_type_trans = DiscreteTransition(
            config.v_beta_schedule, self.num_timesteps,
            s=config.v_beta_s, num_classes=self.num_classes, prior_probs=prior_node_type
        )
        self.edge_type_trans = DiscreteTransition(
            config.v_beta_schedule, self.num_timesteps,
            s=config.v_beta_s, num_classes=self.num_edge_classes, prior_probs=prior_edge_type
        )

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # model definition
        self.hidden_dim = config.hidden_dim

        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config)

        self.predictor = GCNnodeFeature(num_features=5, hidden_dim=64, output_dim=5)
        self.hoppredictor = MlpPredictor(num_node=5, hidden_dim=64, output_dim=5)

        # how to add prior node

        # 修改了atom emb
        self.node_emb = Encoder(num_node=7, hidden_dim=64, feature=50)

        # time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']

        # node type prediction
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        self.distance_expansion = GaussianSmearing(0., 5., num_gaussians=config.num_r_gaussian, fix_offset=False)
        # add prior bond ???
        edge_input_dim = self.hidden_dim
        self.bond_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, self.num_edge_classes),
        )


    def forward(self, config, node, edge, num_nodes, num_edges, time_step=None):
        batch = config.train.batch_size
        device = torch.device("cuda:" + self.config.gpu_id)
        edge = edge[0]
        edge = edge.float()
        edge = edge.to(device)
        init_node_v = node
        unique_elements = sorted(set(init_node_v))
        num_unique_elements = 5
        one_hot_matrix = np.zeros((len(init_node_v), num_unique_elements), dtype=int)
        for i, element in enumerate(init_node_v):
            index = unique_elements.index(element)
            one_hot_matrix[i, index] = 1
        # make node feature to 5
        init_node_v = one_hot_matrix.astype(np.float32)
        init_node_v = torch.tensor(init_node_v, dtype=torch.float32)
        print("init_node", init_node_v)
        init_node_v = init_node_v.to(device)
        init_hop_v = aggregate_1hop_embeddings(edge,init_node_v)
        #init_hop_v = edge
        print("init_hop_v",init_hop_v)

        if self.time_emb_dim > 0:
            input_node_feat = torch.cat(
                init_node_v,
                (time_step / self.num_timesteps)[node].unsqueeze(-1)
            )

        # print("edge",edge)
        # print("node",node)
        edge_index = torch.nonzero(edge, as_tuple=False).t()
        # print("index", edge_index)


        #h_arch = self.node_emb(init_node_v, edge_index)
        # print("h_arch", h_arch)

        # add prior node

        # h_all, batch_all, mask_node, arch_index_in_ctx, node_index_in_ctx = compose_context(
        #     h_arch=h_arch,
        #     node=init_node_v,
        #     edge=edge,
        #     edge_index=edge_index,
        #     batch=batch
        # )
        group_idx_all = None

        node_pred = self.predictor(init_node_v, edge_index)
        #node_pred = self.hoppredictor(init_node_v)
        hop_pred = self.hoppredictor(init_hop_v)
        preds = {
            'pred_node': node_pred,
            'pred_hop': hop_pred
        }
        print("pred", preds)

        return preds

    def get_loss_diffusion(self, config, edge, node, num_nodes, num_edges, input_data, time_step=None):
        device = torch.device("cuda:" + self.config.gpu_id)
        batch = config.train.batch_size

        # input_data = input_data.to(device)
        # node2graph = torch.zeros(input_data.shape)[0].to(device)
        num_graphs = 1000

        unique_elements = list(set(node))
        unique_elements.sort()
        num_unique_elements = len(unique_elements)
        # one_hot_matrix = np.zeros((len(node), 5), dtype=int)
        # for i, element in enumerate(node):
        #     index = unique_elements.index(element)
        #     one_hot_matrix[i, index] = 1
        # node_v = one_hot_matrix.astype(np.float32)
        # node_v = torch.from_numpy(node_v)
        node_v = onehot(node)
        # node_v = node
        node_v = node_v.to(device)

        num_unique_elements_t = len(unique_elements) + 2
        one_hot_matrix_t = np.zeros((len(node), 5), dtype=int)
        for i, element in enumerate(node):
            index = unique_elements.index(element)
            one_hot_matrix_t[i, index] = 1
        node_v_t = one_hot_matrix_t.astype(np.float32)
        node_v_t = torch.from_numpy(node_v_t)
        # node_v_t = node_v
        node_v_t = node_v_t.to(device)

        edge_v = edge[0]
        #edge_v = edge
        edge_v = edge_v.to(device)
        edge_v = edge_v.float()
        one_hop_graph = aggregate_1hop_embeddings(edge_v,node_v_t)# (7,1)
        #one_hop_graph = node_v_t

        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        # 2. perturb node, (and edge)
        # Vt = a * V0 + (1-a) / K

        log_node_v0 = index_to_log_onehot(node_v, self.num_classes + 2)
        #log_node_v0 = node_v
        node_v_perturbed, log_node_vt = self.node_type_trans.q_v_sample(
            log_node_v0, time_step, batch
        )

        # log_edge_v0 = edge_v
        # edge_v_perturbed, log_edge_vt = self.edge_type_trans.q_v_sample(
        #     log_edge_v0, time_step, batch
        # )
        log_hop_v0 = one_hop_graph
        hop_v_perturbed, log_hop_vt = self.edge_type_trans.q_v_sample(
            log_hop_v0, time_step, batch
        )

        # 3. forward-pass NN, feed perturbed node and edge
        preds = self.forward(
            config=config,
            node=node,
            edge=edge,
            num_nodes=num_nodes,
            num_edges=num_edges
        )


        pred_node_v = preds['pred_node']
        pred_hop_v = preds['pred_hop']


        # node type
        log_node_v_recon = F.log_softmax(pred_node_v, dim=-1)
        log_node_v0_t = index_to_log_onehot(node_v_t, self.num_classes)
        #log_node_v0_t = node_v_t
        log_v_model_prob = self.node_type_trans.q_v_posterior(
            log_node_v_recon, log_node_vt, time_step, batch
        )
        log_v_true_prob = self.node_type_trans.q_v_posterior(
            log_node_v0_t, log_node_vt, time_step, batch
        )

        # subgraph type   hop_v_perturbed, log_hop_vt
        log_hop_v_recon = F.log_softmax(pred_hop_v, dim=-1)
        log_hop_v0_t = one_hop_graph
        log_hop_model_prob = self.edge_type_trans.q_v_posterior(
            log_hop_v0_t, log_hop_vt, time_step, batch
        )
        log_hop_true_prob = self.edge_type_trans.q_v_posterior(
            log_hop_v0_t, log_hop_vt, time_step, batch
        )

        # compute kl
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_node_v0_t,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch)
        loss_node = torch.mean(kl_v)

        kl_hop = self.compute_v_Lt(log_v_model_prob=log_hop_model_prob, log_v0=log_hop_v0,
                                 log_v_true_prob=log_hop_true_prob, t=time_step, batch=batch)
        loss_hop = torch.mean(kl_hop)

        results = {
            'losses': {
                'node': loss_node,
                 'hop':loss_hop,
            },
            'pred_node': pred_node_v,
            'pred_hop':pred_hop_v
        }

        return results

    @torch.no_grad()
    def sample_diffusion(self,config, batch,node, edge, num_nodes, num_edges, num_steps=None):
        device = torch.device("cuda:" + self.config.gpu_id)
        if num_steps == None:
            num_steps = self.num_timesteps
        num_graphs = batch + 1

        v0_pred_node, vt_pred_node = [],[]
        b0_pred_hop, bt_pred_hop = [], []

        #time_sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=device)
            preds = self(
                config=config,
                node=node,
                edge=edge,
                num_nodes=num_nodes,
                num_edges=num_edges
            )
            v0_from_e = preds['pred_node']# 7
            hop0_from_e = preds['pred_hop']# 7


            node_v = onehot(node)
            #node_v = node
            node_v = node_v.to(device)

            log_node_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_node_v= index_to_log_onehot(node_v, self.num_classes )
            #log_node_v = torch.cat((log_node_v, log_node_v[:,-2:]),dim=1)
            #log_node_v = node_v
            print('++++++++++++++++++++')
            print(log_node_v_recon,log_node_v)
            log_model_prob = self.node_type_trans.q_v_posterior(
                log_node_v_recon,log_node_v, t, batch)
            ligand_v_next = log_sample_categorical(log_model_prob)

            v0_pred_node.append(log_node_v_recon.clone().cpu())
            vt_pred_node.append(log_model_prob.clone().cpu())

            log_hop_v_recon = F.log_softmax(hop0_from_e, dim=-1)
            unique_elements = list(set(node))
            unique_elements.sort()
            one_hot_matrix_t = np.zeros((len(node), 5), dtype=int)
            for i, element in enumerate(node):
                index = unique_elements.index(element)
                one_hot_matrix_t[i, index] = 1
            node_v_t = one_hot_matrix_t.astype(np.float32)
            node_v_t = torch.from_numpy(node_v_t)
            #node_v_t = node
            node_v_t = node_v_t.to(device)
            edge_v = edge[0]
            edge_v = edge_v.to(device)
            edge_v = edge_v.float()
            one_hop_graph = aggregate_1hop_embeddings(edge_v, node_v_t)  # (7,1)
            #one_hop_graph = node_v_t
            log_hop_v0_t = one_hop_graph
            _, log_hop_vt = self.edge_type_trans.q_v_sample(
                log_hop_v0_t, t, batch
            )
            log_hop_model_prob = self.edge_type_trans.q_v_posterior(
                log_hop_v0_t, log_hop_vt, t, batch
            )

            b0_pred_hop.append(log_hop_v_recon.clone().cpu())
            bt_pred_hop.append(log_hop_model_prob.clone().cpu())

        return {
            'v0_node': v0_pred_node,
            'vt_node': vt_pred_node,
            'v0_hop': b0_pred_hop,
            'vt_hop': bt_pred_hop
        }


    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def sample_time(self, num_graphs, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt

        else:
            raise ValueError

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        device = torch.device("cuda:" + self.config.gpu_id)
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        # loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch_t, dim=0)
        return kl_v


def is_bond(edge_type):
    return torch.logical_and(edge_type < len(BOND_TYPES), edge_type > 0)


def is_angle_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 1 - 1


def is_dihedral_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 2 - 1


def is_radius_edge(edge_type):
    return edge_type == 0


def is_local_edge(edge_type):
    return edge_type > 0


def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(is_sidechain[edge_index[0]], is_sidechain[edge_index[1]])


def regularize_bond_length(edge_type, edge_length, rng=5.0):
    mask = is_bond(edge_type).float().reshape(-1, 1)
    d = -torch.clamp(edge_length - rng, min=0.0, max=float('inf')) * mask
    return d


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom


class GCNnodeFeature(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64, output_dim=5):
        super(GCNnodeFeature, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 7-> 5
def aggregate_1hop_embeddings(adjacency_matrix, node_features):
    #adjacency_matrix_sparse = torch.sparse.FloatTensor(torch.nonzero(adjacency_matrix).t(),
                                                       # torch.ones(adjacency_matrix.nonzero().size(0)),
                                                       # torch.Size(adjacency_matrix.size()))

    neighbor_features = torch.sparse.mm(adjacency_matrix, node_features)
    print('nf', neighbor_features)

    neighbor_features_mean = torch.mean(neighbor_features, dim=1)
    neighbor_features_mean = neighbor_features_mean[:5]

    return neighbor_features_mean