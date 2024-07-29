import os
import shutil
import argparse

import numpy as np
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from dgl.dataloading import GraphDataLoader

from models.epsnet import get_model
from utils.datasets import ConformationDataset
from utils.transforms import *
from utils.misc import *
from utils.common import get_optimizer, get_scheduler
from dataset import *
from torch.utils.data import Dataset, DataLoader

NAS_BENCH_101 = ''
NASBENCH_101_dict_op = {"input": 0, "output": 1, "conv1x1-bn-relu": 2, "conv3x3-bn-relu": 3, "maxpool3x3": 4}

class CustomDataset(Dataset):
    def __init__(self, pkl_file_path):
        with open(pkl_file_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key, value = list(self.data.items())[idx]
        return value




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()


    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    seed_all(config.train.seed)




    # Logging
    if resume:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='resume')
        os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        shutil.copytree('./models', os.path.join(log_dir, 'models'))
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # Datasets and loaders
    # 101
    logger.info('Loading datasets...')
    # transforms = CountNodesPerGraph()#能够记录节点数目
    pkl_file_path = ""
    dataset = CustomDataset(pkl_file_path)
    data_loader = DataLoader(dataset, batch_size=config.train.batch_size)
    # train_set = NASBench101Dataset(config.dataset.dataset_num, all=config.dataset.all_101)
    # train_iterator = GraphDataLoader(train_set,batch_size=config.train.batch_size)


    # Model
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)

    # Optimizer
    optimizer_global = get_optimizer(config.train.optimizer, model.model_global)
    optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
    scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
    scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)
    start_iter = 1

    # Resume from checkpoint
    # if resume:
    #     ckpt_path, start_iter = get_checkpoint_path(os.path.join(resume_from, 'checkpoints'), it=args.resume_iter)
    #     logger.info('Resuming from: %s' % ckpt_path)
    #     logger.info('Iteration: %d' % start_iter)
    #     ckpt = torch.load(ckpt_path)
    #     model.load_state_dict(ckpt['model'])
    #     optimizer_global.load_state_dict(ckpt['optimizer_global'])
    #     optimizer_local.load_state_dict(ckpt['optimizer_local'])
    #     scheduler_global.load_state_dict(ckpt['scheduler_global'])
    #     scheduler_local.load_state_dict(ckpt['scheduler_local'])

    def train(it):
        model.train()
        optimizer_global.zero_grad()
        optimizer_local.zero_grad()

        for dict in data_loader:
            dict['operation'] = [item[0] for item in dict['operation'] if item]
            print("dict: ", dict)
            print("ad:", dict['adjacency'])
            print("op:", dict['operation'])
            print("num_nodes", len(dict['operation']))
            print("num_edges",np.count_nonzero(np.array(dict['adjacency'])))
            results = model.get_loss_diffusion(
                config = config,
                edge = dict['adjacency'],
                node =  dict['operation'],
                num_nodes=len(dict['operation']),
                num_edges=np.count_nonzero(np.array(dict['adjacency'])),
                input_data=dict
             )
            loss_dict = results['losses']
            loss_node = loss_dict['node']
            loss_hop = loss_dict['hop']
            loss = loss_hop + loss_node
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer_global.step()
            optimizer_local.step()

            logger.info(
                '[Train] Iter %05d | Loss %.2f | Grad %.2f | LR(Global) %.6f | LR(Local) %.6f' % (
                    it, loss.item(), orig_grad_norm,
                    optimizer_global.param_groups[0]['lr'], optimizer_local.param_groups[0]['lr'],
                ))
            writer.add_scalar('train/loss', loss, it)
            writer.add_scalar('train/lr_global', optimizer_global.param_groups[0]['lr'], it)
            writer.add_scalar('train/lr_local', optimizer_local.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad_norm', orig_grad_norm, it)
            writer.flush()



    try:
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:

                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer_global': optimizer_global.state_dict(),
                    'scheduler_global': scheduler_global.state_dict(),
                    'optimizer_local': optimizer_local.state_dict(),
                    'scheduler_local': scheduler_local.state_dict(),
                    'iteration': it,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')

