import torch
import numpy as np
import pickle
import copy
from genotypes import *
import nasbench.api as api
from dgl.data import DGLDataset
import dgl
from scipy import sparse

DARTS = ''

def decode_DARTS_to_dgl(adjacency, operations):
    MAX_FEATURES_Darts = 6
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
        g_features[i, op] = 1
    g.ndata['attr'] = torch.tensor(g_features, dtype=torch.float32)
    return g

class ArchDarts:
    def __init__(self, arch):
        self.arch = arch

    @classmethod
    def random_arch(cls):
        # output a uniformly random architecture spec
        # from the DARTS repository
        # https://github.com/quark0/darts
        NUM_VERTICES = 4
        OPS = ['none',
               'sep_conv_3x3',
               'dil_conv_3x3',
               'sep_conv_5x5',
               'dil_conv_5x5',
               'max_pool_3x3',
               'avg_pool_3x3',
               'skip_connect'
               ]
        normal = []
        reduction = []
        for i in range(NUM_VERTICES):
            ops = np.random.choice(range(1, len(OPS)), NUM_VERTICES)

            # input nodes for conv
            nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
            # input nodes for reduce
            nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)

            normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
            reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])
        return (normal, reduction)

class DataSetDarts:
    def __init__(self, dataset_num=int(1e2), dataset=None):
        self.dataset = 'darts'
        self.INPUT_1 = 'c_k-2'  # num 0
        self.INPUT_2 = 'c_k-1'  # num 1
        self.BASIC_MATRIX = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],  # 5
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # 6
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # 7
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 8
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # 9
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # 10
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 11
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # 12
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 13
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 14
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        # a mapping between genotype and op_list
        self.mapping_intermediate_node_ops = [{'input_0': 1, 'input_1': 5},  # 2
                                              {'input_0': 2, 'input_1': 6, 2: 9},  # 3
                                              {'input_0': 3, 'input_1': 7, 2: 10, 3: 12},  # 4
                                              {'input_0': 4, 'input_1': 8, 2: 11, 3: 13, 4: 14}]  # 5
        self.op_integer = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: -1}
        if dataset is not None:
            self.dataset = dataset
            print('Generate DARTS dataset, the size is :{}'.format(dataset_num))
        else:
            if dataset_num > 0:
                self.dataset = self.generate_random_dataset(dataset_num)
                print('Generate DARTS dataset, the size is :{}'.format(dataset_num))

    def generate_random_dataset(self, num):
        """
        create a dataset of randomly sampled architectures where may exist duplicates
        """
        data = []
        while len(data) < num:
            archtuple = ArchDarts.random_arch()
            data.append(archtuple)
        return data

    def get_ops(self, cell_tuple):
        all_ops = []
        mapping = self.mapping_intermediate_node_ops
        # assign op list
        # initial ops are all zeros, i.e. all types are None
        ops = np.zeros(16, dtype='int8')
        # 'input' -2, 'output' -3
        input_output_integer = {'input': -2, 'output': -3}
        ops[0], ops[-1] = input_output_integer['input'], input_output_integer['output']
        for position, op in enumerate(cell_tuple):
            intermediate_node = position // 2
            prev_node = op[0]
            if prev_node == 0:
                prev_node = 'input_0'
            elif prev_node == 1:
                prev_node = 'input_1'

            # determine the position in the ops
            ops_position = mapping[intermediate_node][prev_node]
            op_type = op[1]
            ops[ops_position] = op_type
        print('op',ops)
        t = self.get_cell_tuple(ops)
        print('r',t)
        return ops

    def get_cell_tuple(self, ops):
        cell_tuple = []
        mapping = self.mapping_intermediate_node_ops
        for intermediate_node, node_map in enumerate(mapping):
            for prev_node, ops_position in node_map.items():
                if ops[ops_position] != 0:
                    if prev_node == 'input_0':
                        prev_node = 0
                    elif prev_node == 'input_1':
                        prev_node = 1
                    cell_tuple.append((prev_node, ops[ops_position]))
        return cell_tuple

    def delete_useless_nodes(self, cell_tuple):
        '''
        This function would not change the op integers (1-6)
        The skip connection is 7, the none is 0
        '''
        print('tuple',cell_tuple)
        ops = self.get_ops(cell_tuple)

        BASICMATRIX_LENGTH = len(self.BASIC_MATRIX)
        matrix = copy.deepcopy(self.BASIC_MATRIX)
        for i, op in enumerate(ops):
            if op == 7:  # skip connection
                m, n = [], []
                for m_index in range(BASICMATRIX_LENGTH):
                    ele = matrix[m_index][i]
                    if ele == 1:
                        # set element to 0
                        matrix[m_index][i] = 0
                        m.append(m_index)

                for n_index in range(BASICMATRIX_LENGTH):
                    ele = matrix[i][n_index]
                    if ele == 1:
                        # set element to 0
                        matrix[i][n_index] = 0
                        n.append(n_index)

                for m_index in m:
                    for n_index in n:
                        matrix[m_index][n_index] = 1

            elif op == 0:  # none op type
                for m_index in range(BASICMATRIX_LENGTH):
                    matrix[m_index][i] = 0
                for n_index in range(BASICMATRIX_LENGTH):
                    matrix[i][n_index] = 0

            # start pruning
        model_spec = api.ModelSpec(matrix=matrix, ops=list(ops))
        print('model',model_spec.matrix, model_spec.ops )
        return model_spec.matrix, model_spec.ops

    def reconstruct_cell_tuple(self,ops):
        '''
        Attempts to reconstruct the original cell tuple from the modified matrix and ops.
        This is a hypothetical function and may not perfectly reconstruct the original cell tuple
        if the original delete_useless_nodes function performed irreversible operations.
        '''

        cell_tuple = []

        # Iterate over each operation in ops
        for op_index, op in enumerate(ops):
            if op == 7:  # Skip connection
                # We need to find which nodes are connected in the matrix
                # and reconstruct the skip connection pattern.
                # This is a complex task and depends on the specifics of the matrix structure.
                # Here, we just assume we can find the connected nodes and add a placeholder.
                connected_nodes = [i for i, row in enumerate(ops[op_index]) if row == 1]
                cell_tuple.append((7, connected_nodes))
            elif op == 0:  # None op type
                # No connection for this node
                cell_tuple.append((0, []))
            else:
                # For other ops, we assume the matrix has a 1 in the corresponding position
                # and add a placeholder for the connection.
                cell_tuple.append((op, [op_index]))

        return cell_tuple

    def transfer_ops(self, ops):
        '''
        op_dict = {
                0: 'none',
                1: 'sep_conv_5x5',
                2: 'dil_conv_5x5',
                3: 'sep_conv_3x3',
                4: 'dil_conv_3x3',
                5: 'max_pool_3x3',
                6: 'avg_pool_3x3',
                7: 'skip_connect'
            }
        transfer_ops:
        op_dict = {
                0: 'input',
                1: 'output',
                2: 'conv_1x1',
                3: 'conv_3x3',
                4: 'pool',
                5: 'conv5x5',
            }
        '''
        trans_op = copy.deepcopy(ops)
        for index, op_value in enumerate(trans_op):
            if op_value == -2:
                trans_op[index] = 0
            elif op_value == -3:
                trans_op[index] = 1
            elif op_value == 3 or op_value == 4:
                trans_op[index] = 3
            elif op_value == 5 or op_value == 6:
                trans_op[index] = 4
            elif op_value == 1 or op_value == 2:
                trans_op[index] = 5
            else:
                raise ValueError("Error: unknown ops: %d" % (op_value))
        return trans_op

    def load_DARTS_graphs(self, transfer_ops=True, type="dgl"):
        g_list = []
        for index, tuple_arch in enumerate(self.dataset):
            norm_matrixes, norm_ops = self.delete_useless_nodes(tuple_arch[0])
            reduc_matrixes, reduc_ops = self.delete_useless_nodes(tuple_arch[1])
            if transfer_ops:
                norm_ops = self.transfer_ops(norm_ops)
                reduc_ops = self.transfer_ops(reduc_ops)
            norm_adj = np.array(norm_matrixes, dtype=np.int8)
            reduc_adj = np.array(reduc_matrixes, dtype=np.int8)
            if type == "igraph":
                norm_g = decode_DARTS_to_igraph(norm_adj, norm_ops)
                reduc_g = decode_DARTS_to_igraph(reduc_adj, reduc_ops)
            elif type == "dgl":
                norm_g = decode_DARTS_to_dgl(norm_adj, norm_ops)
                reduc_g = decode_DARTS_to_dgl(reduc_adj, reduc_ops)
            g_list.append(norm_g)
            g_list.append(reduc_g)
        return g_list

    def transfer_tuple_to_graph(self, tuple, transfer_ops=True, type='dgl'):
        matrixes, ops = self.delete_useless_nodes(tuple)
        if transfer_ops:
            ops = self.transfer_ops(ops)
        adj = np.array(matrixes, dtype=np.int8)
        # print(adj)
        # print("len:"+str(len(ops)))
        if type == "igraph":
            g = decode_DARTS_to_igraph(adj, ops)
        elif type == "dgl":
            g = decode_DARTS_to_dgl(adj, ops)

        return g





# def read_darts_dataset(regurized=True):
#     data = torch.load(DARTS)
#     dataset = data['dataset']
#
#     d = DataSetDarts(100000)
#     DARTS_geno = {}
#     acc = data['best_acc_list']
#     #print(acc)
#     acc_1 = []
#     for i in acc:
#         acc_1.append(i)
#     acc_np = np.array(acc_1)
#
#     top20_acc = acc_np.argsort()[-20:][::-1]  # 取排序后的最后20个，即前20高
#
#     mean = np.mean(acc_np[top20_acc])
#     std = np.std(acc_np[top20_acc])
#     #print(top20_acc)
#
#     for index in top20_acc:
#         norm_tuple, reduc_tuple = dataset[index]
#         acc_value = acc_1[index]
#         if regurized:
#             acc_value = (acc_value - mean) / std
#
#         DARTS_geno[norm_tuple,reduc_tuple]=acc_value
#
#         # normal_tup = []
#         # for (node, op) in norm_tuple:
#         #     normal_tup.append((PRIMITIVES[op], node))
#         # reduc_tup = []
#         # for (node, op) in reduc_tuple:
#         #     reduc_tup.append((PRIMITIVES[op], node))
#         # geno = Genotype(normal=normal_tup, normal_concat=[2, 3, 4, 5], reduce=reduc_tup, reduce_concat=[2, 3, 4, 5])
#         # DARTS_geno.append((geno,acc_value))
#
#     return DARTS_geno, mean, std

def read_darts_dataset(regurized=True):
    data = torch.load(DARTS)
    dataset = data['dataset']
    d = DataSetDarts(0)
    DARTS_norm_g_y = []
    DARTS_reduc_g_y = []
    acc = data['best_acc_list']
    acc_1 = []
    for i in acc:
        acc_1.append(i / 100.)
    acc_np = np.array(acc_1)
    mean = np.mean(acc_np)
    std = np.std(acc_np)

    for index, (norm_tuple, reduc_tuple) in enumerate(dataset):
        norm_tuple_g = d.transfer_tuple_to_graph(norm_tuple)
        reduc_tuple_g = d.transfer_tuple_to_graph(reduc_tuple)
        acc_value = acc_1[index]
        if regurized:
            acc_value = (acc_value - mean) / std
        DARTS_norm_g_y.append((norm_tuple_g, acc_value))
        DARTS_reduc_g_y.append((reduc_tuple_g, acc_value))
    return DARTS_norm_g_y, DARTS_reduc_g_y, mean, std


def replace_max_pool_with_none(genotype):
    new_genotypes = []
    normal = genotype.normal[:-1] + [('none', 1)]
    reduce = genotype.reduce[:-1] + [('none', 1)]
    new_gt = Genotype(normal=normal, reduce= reduce, normal_concat=[2, 3, 4, 5],reduce_concat=[2, 3, 4, 5])


    return new_gt


class DartsDataset(DGLDataset):
    def __init__(self, NUM_EXAMPLES_Darts=30000, labeled=False):
        self.labeled = labeled
        if labeled:
            self.NUM_EXAMPLES_Darts = 20
            print("Labeled data...")
        else:
            self.NUM_EXAMPLES_Darts = NUM_EXAMPLES_Darts
        super().__init__(name='Darts')

    def process(self):
        print("Building dataset Darts...")
        self.norm_graphs = []
        self.reduc_graphs = []
        self.norm_labels = []
        self.reduc_labels = []
        self.graphs = []
        if self.labeled:
            norm_DARTS_test, reduc_DARTS_test, mean, std = read_darts_dataset()
            for (g, y) in norm_DARTS_test:
                self.norm_graphs.append(g)
                self.norm_labels.append(y)
            self.norm_labels = torch.tensor(self.norm_labels, dtype=torch.float32)
            for (g, y) in reduc_DARTS_test:
                self.reduc_graphs.append(g)
                self.reduc_labels.append(y)
            self.reduc_labels = torch.tensor(self.reduc_labels, dtype=torch.float32)
        else:
            self.graphs = DataSetDarts(int(self.NUM_EXAMPLES_Darts / 2)).load_DARTS_graphs()

    def __getitem__(self, i):
        if self.labeled:
            return self.norm_graphs[i], self.reduc_graphs[i], self.norm_labels[i]
        else:
            return self.graphs[i]

    def __len__(self):
        if self.labeled:
            return len(self.norm_graphs)
        else:
            return len(self.graphs)



if __name__ == '__main__':
    #DARTS_geno, mean, std = read_darts_dataset()
    # op = torch.tensor([[1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 1],
    #     [0, 1, 0, 0, 0, 0]])
    # d = DataSetDarts()
    # tuple =
    # g = d.get_ops(tuple)
    # g = d.reconstruct_cell_tuple(op)
    # print(g)
    # DARTS_geno, mean, std = read_darts_dataset()
    # print(DARTS_geno)
    geno = DartsDataset(NUM_EXAMPLES_Darts=100, labeled=False)
    dict = {}
    for i in range(50):
        inter_dict = {}
        g = geno[i]
        adjancy = g.ndata['attr']
        node_features = g.ndata['attr']
        inter_dict['module_adjacency'] = adjancy
        inter_dict['module_operations'] = node_features
        dict[i]= inter_dict



    with open('best20_darts_g.pkl', 'wb') as file:
        pickle.dump(dict, file)
    # dictfile = open("best20_darts.pkl", 'rb')
    # geno_list = pickle.load(dictfile)
    # data = torch.load(DARTS)
    # dataset = data['dataset']
    # acc_list = data['best_acc_list']
    # #print(data)
    # acc_index = 1
    # for (geno, acc) in geno_list:
    #     #print(geno)
    #     print(geno.normal[-1])
    #     print(geno.reduce[-1])
    #     new_genotype = replace_max_pool_with_none(geno)
    #     normal_cell = new_genotype.normal
    #     reduce_cell = new_genotype.reduce
    #
    #     # print(normal_cell[-1])
    #     # print(reduce_cell[-1])
    #
    #     for index, (norm_tuple, reduc_tuple) in enumerate(dataset):
    #         normal_tup = []
    #         for (node, op) in norm_tuple:
    #             normal_tup.append((PRIMITIVES[op], node))
    #         reduce_tup = []
    #         for (node, op) in reduc_tuple:
    #             reduce_tup.append((PRIMITIVES[op], node))
    #         # print('normal',normal_tup)
    #         # print('reduce', reduce_tup)
    #
    #
    #         if normal_tup[:-1] == normal_cell[:-1] or reduce_tup[:-1] == reduce_cell[:-1]:
    #             acc_index = index
    #             #print(index)
    #         else:
    #             continue
    #
    #     #print(acc_index)
    #     acc_change = acc_list[acc_index]
    #     # print('acc',acc)
    #     # print('c',acc_change)
    #     marginal_contribution = acc - acc_change
    #     print(f"{marginal_contribution}")
    #     # shapley_value = marginal_contribution / (2 ** 8 - 1)
    #     # print(f"{shapley_value}")
    #     print('\n')
    #
    # dictfile.close()
