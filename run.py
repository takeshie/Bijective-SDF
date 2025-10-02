# -*- coding: utf-8 -*-

import time
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from models.dataset import DatasetNP
from models.gdo import MeshTrainer
from models.tet_init import tet_init
import argparse
from pyhocon import ConfigFactory
import os
from shutil import copyfile
import numpy as np
import trimesh
from models.utils import get_root_logger, print_log
import math
import mcubes
import warnings
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from scipy.spatial import cKDTree
import point_cloud_utils as pcu
from sklearn.neighbors import LocalOutlierFactor,NearestNeighbors 
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
warnings.filterwarnings("ignore")
from models.model_BSP import build_BSP


class Runner:
    def __init__(self, args, conf_path, mode='train'):
        self.device = torch.device(f'cuda:{args.gpu}')
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()
        self.conf = ConfigFactory.parse_string(conf_text)
        self.base_exp_dir=self.conf['general.base_exp_dir']+os.path.join(args.data_class,args.dataname)
        self.mesh_save_dir=os.path.join(self.base_exp_dir,'mesh')
        os.makedirs(self.base_exp_dir, exist_ok=True)
        os.makedirs(self.mesh_save_dir, exist_ok=True)
        
        self.dataset_np = DatasetNP(self.conf['dataset'],args.data_dir, args.dataname,self.conf['bsp_net'].get_int('preenc_npoints'))
        self.dataname = args.dataname
        self.iter_step = 0
        self.maxiter = self.conf.get_int('train.maxiter')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_learning_rate = self.conf.get_float('train.warm_up_learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.mode = mode
        self.grid_network = MeshTrainer(device=self.device)
        self.bsp_network = build_BSP(self.conf['bsp_net']).cuda()
        self.param_list = list(self.bsp_network.parameters())+list(tet_init.parameters())
        self.optimizer = torch.optim.Adam(self.param_list, lr=self.warm_up_learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.9)
        self.ChamferDisL1 = ChamferDistanceL1().cuda()
        self.ChamferDisL2 = ChamferDistanceL2().cuda()

    def train(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(os.path.join(self.base_exp_dir), f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        self.mse_loss = nn.MSELoss()
        res_step = self.maxiter - self.iter_step
        for iter_i in tqdm(range(res_step)):
            point_gt,shape_scale,shape_center = self.dataset_np.np_train_data()
            pts = point_gt.to(self.device)
            if(iter_i==0):
                self.grid_network.initialize_model(finetune_net=True)
            train_pts = self.bsp_network(pts)
            test_set = pts.detach()
            test_pts = self.bsp_network(test_set)
            test_pts = torch.cat([test_pts, pts],dim=0).float().to(self.device)
            self.grid_network.load_pts(test_pts)
            laplas,sdf,mesh_verts=self.grid_network.run_iteration(it=iter_i,data_name=self.dataname,save_dir=self.mesh_save_dir,shape_scale=shape_scale,shape_center=
            shape_center)
            loss_pt = self.ChamferDisL2(train_pts.unsqueeze(0), pts.unsqueeze(0))
            loss_sdf = self.ChamferDisL2(test_pts.unsqueeze(0), mesh_verts.unsqueeze(0))
            onsurf_loss=self.mse_loss(sdf, torch.zeros_like(sdf))
            loss = loss_sdf + 0.1*loss_pt + 1e-5*onsurf_loss +laplas
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if(iter_i==self.warm_up_end):
                self.optimizer.param_groups[0]['lr'] =self.learning_rate
            if(iter_i>self.warm_up_end):
                self.scheduler.step()
            self.iter_step += 1
            if self.iter_step % self.report_freq == 0:
                print_log('iter:{:8>d} cd_l1 = {} lr={}'.format(self.iter_step, loss_sdf, self.optimizer.param_groups[0]['lr']), logger=logger)

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataname', type=str, default='')
    parser.add_argument('--data_class', type=str, default='plane')
    parser.add_argument('--file_list', type=str, default='datalist')
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()
    seed=20
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(args.gpu)
    src='./outs/'+ args.data_class
    os.makedirs(src, exist_ok=True)
    datalist=[]
    with open('./data_list/{}.txt'.format(args.file_list), 'r') as file:
        for line in file:
            filename = line.strip()
            datalist.append(filename)
    print(datalist)
    for i in range(len(datalist)):
        args.dataname=datalist[i]
        runner = Runner(args, args.conf,args.mode)
        if args.mode == 'train':
            runner.train()
