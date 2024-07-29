from nasbench import api
import numpy as np
import random
import os
import pickle
from itertools import combinations

NASBENCH_TFRECORD = ""
NASBENCH_MAX_LEN = 423624

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NULL = 'null'


def get_feature(type):
    if type == 'input':
        feature = [1,0,0,0,0]
    elif type =='conv1x1-bn-relu':
        feature = [0,1,0,0,0]
    elif type == 'conv3x3-bn-relu':
        feature = [0,0,1,0,0]
    elif type == 'maxpool3x3':
        feature = [0,0,0,1,0]
    elif type =='output':
        feature = [0,0,0,0,1]
    return feature

def calculate_accuracy(adjacency_matrix,operations):
    model_spec = api.ModelSpec(
        matrix=adjacency_matrix,ops=operations
    )
    data = nasbench.query(model_spec)
    return data['test_accuracy']


def remove_operation(adjacency_matrix, operation_list,operation_index):

    new_adjacency_matrix = adjacency_matrix


    new_adjacency_matrix = np.delete(new_adjacency_matrix, operation_index, axis=0)
    new_adjacency_matrix = np.delete(new_adjacency_matrix, operation_index, axis=1)

    new_operation_list = operation_list[:operation_index]+operation_list[operation_index+1:]
    return new_adjacency_matrix, new_operation_list, operation_list[operation_index]


def marginal_contribution(adjacency_matrix, operation_list, calculate_accuracy):

    new_adjacency_matrix,new_operation_list = remove_operation(adjacency_matrix,operation_list)

    original_accuracy = calculate_accuracy(adjacency_matrix)
    modified_accuracy = calculate_accuracy(new_adjacency_matrix)

    return original_accuracy - modified_accuracy


def calculate_shapley_value(adjacency_matrix, operation_list):
    original_accuracy = calculate_accuracy(adjacency_matrix,operation_list)
    shapley_values = {}

    drop_conv3 = []
    drop_conv1 = []
    drop_pool = []


    for i in range(1,4):
        new_adjacency_matrix, new_operation_list, index = remove_operation(adjacency_matrix, operation_list, i)
        modified_accuracy = calculate_accuracy(new_adjacency_matrix, new_operation_list)

        marginal_contribution = original_accuracy - modified_accuracy
        num_nodes = len(operation_list)
        shapley_value = marginal_contribution / (2 ** num_nodes - 1)

        temp = str(operation_list[i])
        shapley_values[temp] = shapley_value
        if index == 'conv3x3-bn-relu':
            drop_conv3.append(marginal_contribution/original_accuracy)
        elif index == 'maxpool3x3':
            drop_pool.append(marginal_contribution/original_accuracy)
        elif index == 'conv1x1-bn-relu':
            drop_conv1.append(marginal_contribution/original_accuracy)
    return shapley_values, drop_conv1,drop_conv3,drop_pool

def padding_zero_in_matrix(important_metrics):
    for i in important_metrics:
        len_operations = len(important_metrics[i]['fixed_metrics']['module_operations'])
        if len_operations != 7:
            # if the operations is less than 7
            for j in range(len_operations, 7):
                important_metrics[i]['fixed_metrics']['module_operations'].insert(j - 1, 'null')
            # print(important_metrics[i]['fixed_metrics']['module_operations'])

            adjecent_matrix = important_metrics[i]['fixed_metrics']['module_adjacency']
            padding_matrix = np.insert(adjecent_matrix, len_operations - 1,
                                       np.zeros([7 - len_operations, len_operations]), axis=0)
            padding_matrix = np.insert(padding_matrix, [len_operations - 1], np.zeros([7, 7 - len_operations]), axis=1)
            important_metrics[i]['fixed_metrics']['module_adjacency'] = padding_matrix
    return important_metrics

if __name__ == '__main__':
    nasbench = api.NASBench(NASBENCH_TFRECORD)

    ori_dict = {}
    for key in nasbench.computed_statistics.keys():
        len_operations = len(nasbench.fixed_statistics[key]['module_operations'])
        if len_operations != 7:
            for j in range(len_operations, 7):
                nasbench.fixed_statistics[key]['module_operations'].insert(j - 1, 'null')
            adjecent_matrix = nasbench.fixed_statistics[key]['module_adjacency']
            padding_matrix = np.insert(adjecent_matrix, len_operations - 1,
                                       np.zeros([7 - len_operations, len_operations]), axis=0)
            padding_matrix = np.insert(padding_matrix, [len_operations - 1], np.zeros([7, 7 - len_operations]), axis=1)
            nasbench.fixed_statistics[key]['module_adjacency'] = padding_matrix


        ori_dict[key] = {'adjacency':nasbench.fixed_statistics[key]['module_adjacency'],
                     'operation':nasbench.fixed_statistics[key]['module_operations'],
                     'accuracy':(nasbench.computed_statistics[key][108][0]['final_test_accuracy'] +
                     nasbench.computed_statistics[key][108][1]['final_test_accuracy']
                     + nasbench.computed_statistics[key][108][2]['final_test_accuracy']) / 3}



    sorted_dict = sorted(ori_dict.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    sorted_dict = dict(sorted_dict)
    dict = {}
    num=0
    for key in sorted_dict:
        dict[key] = sorted_dict[key]
        num += 1
        if num == 20:
            break

    print(dict)
    with open('best20.pkl','wb') as file:
        pickle.dump(dict,file)
   #  dictfile = open("best20.pkl", 'rb')
   #  dict = pickle.load(dictfile)
   #  dictfile.close()
   # # print(dict)
   #  shapley_values = {}
   #  for key, value in dict.items():
   #      adjacency_matrix = value['adjacency']
   #      #print(adjacency_matrix)
   #      operation_list = value['operation']
   #     # print(operation_list)
   #
   #      num_nodes = len(operation_list)
   #
   #
   #      shapley_values, drop_conv1,drop_conv3,drop_pool= calculate_shapley_value(adjacency_matrix, operation_list)
   #      print('1x1', drop_conv1)
   #      print('3x3', drop_conv3)
   #      print('pool', drop_pool)
   #      #print(f"{shapley_values}")
