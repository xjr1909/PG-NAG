from cProfile import label
from nasbench.lib import model_metrics_pb2
import igraph
import tensorflow as tf
import json
import base64
import numpy as np
import random
import pickle
import argparse

import utils
from utils import *
import torch
# from torch.utils.data import Dataset
from dgl.data import DGLDataset
import dgl
from scipy import sparse
import copy
import nasbench.api as api


NAS_BENCH_101 = ""
NAS_BENCH_201 = ""
MAX_NUMBER_201 = 15625
DARTS = ""
NASBENCH_101_dict_op = {"input": 0, "output": 1, "conv1x1-bn-relu": 2, "conv3x3-bn-relu": 3, "maxpool3x3": 4}
NASBENCH_201_dict_op= {"input": 0, "output": 1, "nor_conv_1x1": 2, "nor_conv_3x3":3, "avg_pool_3x3": 4}


def load_nasbench101_graphs(num_data, n_types=3, fmt="dgl", all=False, regurized=True, rand_seed=0,
                            graph_args=argparse.ArgumentParser().parse_known_args()[0]):
    # load NASBENCH format NNs to igraphs or tensors
    g_list = []

    max_n = 7
    i = 0
    acc_list = []
    for serialized_row in tf.compat.v1.python_io.tf_record_iterator(NAS_BENCH_101):
        # print(serialized_row)
        acc_l = []
        module_hash, epochs, raw_adjacency, raw_operations, raw_metrics = (
            json.loads(serialized_row.decode('utf-8')))
        dim = int(np.sqrt(len(raw_adjacency)))
        if (dim != 7):
            continue
        adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
        adjacency = np.reshape(adjacency, (dim, dim))
        operations = raw_operations.split(',')
        metrics = model_metrics_pb2.ModelMetrics.FromString(base64.b64decode(raw_metrics))
        final_evaluation = metrics.evaluation_data[2]
        y = final_evaluation.test_accuracy
        acc_l.append(y)
        if i % 3 == 2:
            # print(operations)
            mean_acc = np.mean(np.array(acc_l))
            acc_list.append(mean_acc)
            if (fmt == 'igraph'):
                g = decode_NASBENCH_to_igraph(adjacency, operations)
            elif (fmt == 'dgl'):
                # print(adjacency)
                # print(operations)
                g = decode_NASBENCH_to_dgl(adjacency, operations)

            g_list.append((g, mean_acc))
            acc_l = []
        i += 1
        if i > int(num_data * 3) and all == False:
            break
    if regurized:
        # mean_value = 0.902434
        # std_value = 0.058647
        acc_list = np.array(acc_list)
        mean_value = np.mean(acc_list)
        std_value = np.std(acc_list)
        g_list_copy = []
        for g, y in g_list:
            g_list_copy.append((g, (y - mean_value) / std_value))
        g_list = g_list_copy
        del g_list_copy
    graph_args.num_vertex_type = 5
    graph_args.max_n = max_n
    graph_args.START_TYPE = 0
    graph_args.END_TYPE = 1
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    random.Random(rand_seed).shuffle(g_list)
    #return list
    return g_list[:int(ng * 0.9)], g_list[int(ng * 0.9):], graph_args


def denormalize_nasbench101(x):
    mean_value = 0.902434
    std_value = 0.058647
    return x * std_value + mean_value


def decode_NASBENCH_to_dgl(adjacency, operations, dataset="101"):
    MAX_FEATURES_Darts = 6
    if dataset == "101":
        NASBENCH_dict_op = NASBENCH_101_dict_op
    # print(adjacency, operations)
    nonzero = sparse.coo_matrix(adjacency).nonzero()
    src = nonzero[0].tolist()
    dst = nonzero[1].tolist()
    src = torch.tensor(src)
    dst = torch.tensor(dst)
    g = dgl.graph((src, dst))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    num_verticles = adjacency.shape[0]
    g_features = np.zeros((num_verticles, MAX_FEATURES_Darts), dtype=float)
    # g_features[-1, :] = dict_feat['global']
    for i, op in enumerate(operations):
        g_features[i, NASBENCH_dict_op[op]] = 1
    # print(g_features)
    g.ndata['attr'] = torch.tensor(g_features, dtype=torch.float32)

    # g_edges = []
    # for scr, dst in zip(nonzero[0],nonzero[1]):
    #     edge_type = src*10+dst
    #     g_edges.append(edge_type)
    # print(g_edges)
    # g.edata['edge_type'] = torch.tensor(g_edges,dtype=torch.float32)
    return g


def decode_NASBENCH_to_igraph(adjacency, operations, dataset="101"):
    # convert NASBENCH adjacency matrix to ENAS format which is list of lists
    if dataset == "101":
        NASBENCH_dict_op = NASBENCH_101_dict_op
    g = igraph.Graph(directed=True)
    g.add_vertices(len(operations))
    for i, op in enumerate(operations):
        g.vs[i]['type'] = NASBENCH_dict_op[op]
    for i, node in enumerate(adjacency):
        for j, edge in enumerate(node):
            if (edge == 1):
                g.add_edge(i, j)
    return g











class NASBench101Dataset(DGLDataset):
    def __init__(self, NUM_EXAMPLES=30000, all=False):
        self.NUM_EXAMPLES = NUM_EXAMPLES
        self.all = all
        super().__init__(name='NASBench')

    def process(self):
        print("Building dataset NASBench_101...")
        # NAS_Bench_101
        self.graphs = []
        self.labels = []
        self.adjacency = []
        self.operation = []
        g_train, g_test, _ = load_nasbench101_graphs(self.NUM_EXAMPLES, all=self.all)
        #list = load_nasbench101_graphs(self.NUM_EXAMPLES, all=self.all)
        all_graphs = g_train + g_test
        for (g, y) in all_graphs:
            self.graphs.append(g)
            self.labels.append(y)
        # all_graphs = list
        # for (ad,op,acc) in all_graphs:
        #     self.graphs.append(ad,op)
        #     self.labels.append(acc)
        # Convert the label list to tensor for saving.
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        # self.features = np.array(self.features, dtype=object)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)




