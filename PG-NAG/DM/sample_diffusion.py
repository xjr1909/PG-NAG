import argparse
import os
import pickle
import shutil
import time

import numpy as np
import torch
import yaml
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from glob import glob
from easydict import EasyDict

import utils.misc as misc

from models.common import log_sample_categorical
from models.epsnet.dualenc import DualEncoderEpsNetwork

class CustomDataset(Dataset):
    def __init__(self, pkl_file_path):
        with open(pkl_file_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key, value = list(self.data.items())[idx]
        return value

def change(tensor):
    max_indices = np.argmax(np.abs(tensor), axis=1)
    max_index_list = list(max_indices)

    index_to_string = {
        0: 'none',
        1: 'nor_conv_3x3',
        2: 'skip_connect',
        3: 'nor_conv_1x1',
        4: "avg_pool_3x3"
    }
    selected_indices = max_index_list
    formatted_string = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|"
    formatted_string = formatted_string.format(
        *[index_to_string[i] for i in selected_indices[:6]] +
         [selected_indices[:6]] +
         [index_to_string[selected_indices[6]]] if len(selected_indices) > 6 else None,
        selected_indices[6] if len(selected_indices) > 6 else None
    )
    return formatted_string

@torch.no_grad()
def sample_diffusion_decomp(
        model, data_loader, config, num_samples, prior_mode, batch_size=16,
        device='cuda:0', num_steps=None):
    all_pred_node, all_pred_edge = [], []
    all_pred_v0_node, all_pred_v0_hop = [], []
    all_pred_vt_node, all_pred_vt_hop = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0



    outfile = ''
    with open(outfile, 'w') as f:
        for dict in data_loader:
            dict['operation'] = [item[0] for item in dict['operation'] if item]

            if prior_mode == 'nasbench101':
                init_node = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'null', 'output']
                init_edge = torch.empty(7, 7)
                # init node v
                uniform_logits = torch.zeros(7, model.num_classes).to(device)
                init_node_v = log_sample_categorical(uniform_logits)

            if prior_mode == 'darts':  # trans to dgl
                init_node = ['none', 'sep_conv_3x3', 'dil_conv_3x3', 'sep_conv_5x5', 'dil_conv_5x5', 'max_pool_3x3',
                             'avg_pool_3x3', 'skip_connect']
                init_edge = torch.empty(6, 10)
                # init node v
                uniform_logits = torch.zeros(6, model.num_classes).to(device)
                init_node_v = log_sample_categorical(uniform_logits)

            if prior_mode == 'transnas101':
                init_node =  ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'zero', 'skip-connect', 'output']
                init_edge = torch.empty(5, 5)
                # init node v
                uniform_logits = torch.zeros(5, model.num_classes).to(device)
                init_node_v = log_sample_categorical(uniform_logits)

            r = model.sample_diffusion(
                config=config,
                batch=config.train.batch_size,
                node=dict['operation'],
                edge=dict['adjacency'],
                num_nodes=len(dict['operation']),
                num_edges=np.count_nonzero(np.array(dict['adjacency']))
            )

            node_v0, node_vt = r['v0_node'], r['vt_node']
            hop_v0, hop_vt = r['v0_hop'], r['vt_hop']
            all_pred_v0_node.append(node_v0)
            all_pred_vt_node.append(node_vt)
            all_pred_v0_hop.append(hop_v0)
            all_pred_vt_hop.append(hop_vt)

        all_pred_v0_node = all_pred_v0_node.append(all_pred_v0_node)
        all_pred_vt_node = all_pred_vt_node.append(all_pred_vt_node)
        all_pred_v0_hop = all_pred_v0_hop.append(all_pred_v0_hop)
        all_pred_vt_hop = all_pred_vt_hop.append(all_pred_vt_hop)

        all_pred_vt_node = change(all_pred_vt_node)
        all_pred_vt_hop = change(all_pred_vt_hop)



        print('all', all_pred_vt_hop,  all_pred_vt_node,file=f)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_sample', type=str, default='')
    parser.add_argument('--config_train', type=str, default='')
    parser.add_argument('--ori_data_path', type=str, default='')
    parser.add_argument('--outdir', type=str, default='./outputs_test')
    parser.add_argument('-i', '--data_id', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')

    resume_sample = os.path.isdir(args.config_sample)
    resume_train = os.path.isdir(args.config_train)
    if resume_sample:
        config_sample_path = glob(os.path.join(args.config_sample, '*.yml'))[0]
        resume_sample_from = args.config_sample
    else:
        config_sample_path = args.config_sample

    if resume_train:
        config_train_path = glob(os.path.join(args.config_train, '*.yml'))[0]
        resume_train_from = args.config_train
    else:
        config_train_path = args.config_train

    with open(config_sample_path, 'r') as f1:
        config_sample = EasyDict(yaml.safe_load(f1))
    with open(config_train_path, 'r') as f2:
        config_train = EasyDict(yaml.safe_load(f2))

    config_name_sample = os.path.basename(config_sample_path)[:os.path.basename(config_sample_path).rfind('.')]
    config_name_train = os.path.basename(config_train_path)[:os.path.basename(config_train_path).rfind('.')]
    misc.seed_all(config_train.train.seed)
    misc.seed_all(config_sample.sample.seed)

    # logging


    # config = misc.load_config(args.config)
    # config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    # misc.seed_all(config.sample.seed)

    with open(args.ori_data_path,'rb') as f:
        test_data = pickle.load(f)

    log_dir = os.path.join(args.outdir, '%s_%03d_%s' % (config_name_sample, args.data_id, os.path.basename(args.ori_data_path)[:-4]))
    os.makedirs(log_dir, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir)
    logger.info(args)
    logger.info(config_sample)
    shutil.copyfile(args.config_sample, os.path.join(log_dir, os.path.basename(args.config_sample)))

    # load checkpoint
    ckpt_path = config_sample.model.checkpoint
    ckpt = torch.load(ckpt_path, map_location=args.device)
    if 'train_config' in config_sample.model:
        logger.info(f"Load training config from: {config_sample.model['train_config']}")
        ckpt['config'] = misc.load_config(config_sample.model['train_config'])
    logger.info(f"Training Config: {ckpt['config']}")

    # transforms

    # prior_mode list

    # load model
    model = DualEncoderEpsNetwork(
        ckpt['config'].model
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=True)
    logger.info(f'Successfully load the model! {config_sample.model.checkpoint}')

    dataset = CustomDataset(args.ori_data_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    raw_results = sample_diffusion_decomp(
        model, data_loader,
        config = config_train,
        num_samples=config_sample.sample.num_samples,
        batch_size=args.batch_size,
        prior_mode='nasbench101'
    )

