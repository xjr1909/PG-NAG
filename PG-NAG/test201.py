from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random

from absl import app
import pickle
import numpy as np
import copy
from nas_201_api import NASBench201API as API
from nasbench import api

api201 = API('', verbose=False)
NAS_BENCH_201 = ''

# basic matrix for nas_bench 201
BASIC_MATRIX = [[0, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0]]

MAX_NUMBER = 15625
NULL = 'null'
CONV1X1 = 'nor_conv_1x1'
CONV3X3 = 'nor_conv_3x3'
AP3X3 = 'avg_pool_3x3'

def calculate_accuracy(adjacency_matrix,operations):
    dictfile = open("201.pkl", 'rb')
    dict = pickle.load(dictfile)
    for key, value in dict.items():
        if value['fixed_metrics']['module_operations'] == operations:
            print(f"find: {key}, 'final_test_accuracy': {value['final_test_accuracy']}")
            return value['final_test_accuracy']
            break
    else:
        print("not find")


def remove_operation(adjacency_matrix, operation_list,operation_index):
    print('index',operation_index)
    new_adjacency_matrix = adjacency_matrix


    new_adjacency_matrix = np.delete(new_adjacency_matrix, operation_index, axis=0)
    new_adjacency_matrix = np.delete(new_adjacency_matrix, operation_index, axis=1)

    new_operation_list = operation_list[:operation_index]+ ['null'] +operation_list[operation_index+1:]



    return new_adjacency_matrix, new_operation_list



def calculate_shapley_value(adjacency_matrix, operation_list):
    original_accuracy = calculate_accuracy(adjacency_matrix,operation_list)
    shapley_values = {}


    for j in range(1,7):
        new_adjacency_matrix, new_operation_list = remove_operation(adjacency_matrix, operation_list, j)

        print(new_operation_list)
        modified_accuracy = calculate_accuracy(new_adjacency_matrix, new_operation_list)
        print(modified_accuracy)
        if original_accuracy is None or modified_accuracy is None:
            print("Error: One or both of the values is None.")
            return None
        marginal_contribution = original_accuracy - modified_accuracy
        num_nodes = len(operation_list)
        shapley_value = marginal_contribution / (2 ** num_nodes - 1)
        print('value',shapley_value)
        temp = str(operation_list[j])
        print(temp)
        shapley_values[temp] = shapley_value


    return shapley_values



def delete_useless_node(ops):
    # delete the skip connections nodes and the none nodes
    # output the pruned metrics
    # start to change matrix
    matrix = copy.deepcopy(BASIC_MATRIX)
    for i, op in enumerate(ops, start=1):
        m = []
        n = []

        if op == 'skip_connect':
            for m_index in range(8):
                ele = matrix[m_index][i]
                if ele == 1:
                    # set element to 0
                    matrix[m_index][i] = 0
                    m.append(m_index)

            for n_index in range(8):
                ele = matrix[i][n_index]
                if ele == 1:
                    # set element to 0
                    matrix[i][n_index] = 0
                    n.append(n_index)

            for m_index in m:
                for n_index in n:
                    matrix[m_index][n_index] = 1

        elif op == 'none':
            for m_index in range(8):
                matrix[m_index][i] = 0
            for n_index in range(8):
                matrix[i][n_index] = 0

    ops_copy = copy.deepcopy(ops)
    ops_copy.insert(0, 'input')
    ops_copy.append('output')

    # start pruning
    model_spec = api.ModelSpec(matrix=matrix, ops=ops_copy)
    return model_spec.matrix, model_spec.ops

def padding_zeros(matrix, op_list):
    assert len(op_list) == len(matrix)
    padding_matrix = matrix
    len_operations = len(op_list)
    if not len_operations == 8:
        for j in range(len_operations, 8):
            op_list.insert(j - 1, NULL)
        adjecent_matrix = copy.deepcopy(matrix)
        padding_matrix = np.insert(adjecent_matrix, len_operations - 1, np.zeros([8 - len_operations, len_operations]),
                                   axis=0)
        padding_matrix = np.insert(padding_matrix, [len_operations - 1], np.zeros([8, 8 - len_operations]), axis=1)

    return padding_matrix, op_list

def save_arch_str2op_list(save_arch_str):
    op_list = []
    save_arch_str_list = API.str2lists(save_arch_str)
    op_list.append(save_arch_str_list[0][0][0])
    op_list.append(save_arch_str_list[1][0][0])
    op_list.append(save_arch_str_list[1][1][0])
    op_list.append(save_arch_str_list[2][0][0])
    op_list.append(save_arch_str_list[2][1][0])
    op_list.append(save_arch_str_list[2][2][0])
    return op_list

def get_rank(data):
    data = list(data.values())
    sorted_list = sorted(data, key=lambda x: x['final_test_accuracy'], reverse=True)
    sorted_list_modified = [dict({'module_adjacency': item['fixed_metrics']['module_adjacency'],
                                  'module_operations': item['fixed_metrics']['module_operations']}) for item in sorted_list]
    top_20 = sorted_list_modified[:20]
    return top_20

def get_metrics_from_index_list(index_list, ordered_dic, metrics_num, dataset, upper_limit_time=MAX_NUMBER):
    metrics = {}
    times = 0
    total_time = 0
    for index in index_list:
        if times == metrics_num:
            break
        final_test_acc = ordered_dic[index][dataset]
        epoch12_time = ordered_dic[index]['cifar10_all_time']
        total_time += epoch12_time
        if total_time > upper_limit_time:
            break
        op_list = save_arch_str2op_list(ordered_dic[index]['arch_str'])
        pruned_matrix, pruned_op = delete_useless_node(op_list)
        if pruned_matrix is None:
            continue
        else:
            times += 1
        padding_matrix, padding_op = padding_zeros(pruned_matrix, pruned_op)

        metrics[index] = {'final_training_time': epoch12_time, 'final_test_accuracy': final_test_acc / 100}
        metrics[index]['fixed_metrics'] = {'module_adjacency': padding_matrix, 'module_operations': padding_op,
                                           'trainable_parameters': -1}
    return metrics










# If you are passing command line flags to modify the default config values, you
# must use app.run(main)
if __name__ == '__main__':
    #arch_info = get_data()
    with open(NAS_BENCH_201, 'rb') as file:
        ordered_dic = pickle.load(file)
    train_list = random.sample(list(range(0, MAX_NUMBER)), MAX_NUMBER)
    data = get_metrics_from_index_list(train_list, ordered_dic, MAX_NUMBER, 'cifar10_valid')
    # with open('201.pkl', 'wb') as file:
    #
    #     pickle.dump(data, file)
    top = get_rank(data)

    with open('random20_201.pkl', 'wb') as file:

        pickle.dump(top, file)
    # dictfile = open("best20_201.pkl", 'rb')
    # dict = pickle.load(dictfile)
    # dictfile.close()
    # # print(dict)
    # shapley_values = {}
    # # for key, value in dict.items():
    # for i in range(0, 20):
    #     adjacency_matrix = dict[i]['module_adjacency']
    #     print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    #     print(adjacency_matrix)
    #     operation_list = dict[i]['module_operations']
    #     print(operation_list)
    #
    #     num_nodes = len(operation_list)
    #
    #     shapley_values = calculate_shapley_value(adjacency_matrix, operation_list)
    #     print(f"{shapley_values}")
