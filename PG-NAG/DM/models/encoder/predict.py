import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_sum
import numpy as np
from dgl.nn import GraphConv



class refine_net(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, num_nodes, num_edge_features):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_edge_features = num_edge_features

        self.node_GNN = GraphConv(node_feature_dim, hidden_dim)
        self.edge_GNN = GraphConv(num_edge_features, hidden_dim)
        self.agg_gloabl = GraphConv(hidden_dim, hidden_dim)


    def forward(self, h, group_idx, batch):
        all_h = [h]

        x = self.node_GNN()





