
import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy.spatial import cKDTree
import trimesh

def process_data(data_dir, dataname,pt_num):
    if os.path.exists(os.path.join(data_dir, dataname) + '.ply'):
        pointcloud = trimesh.load(os.path.join(data_dir, dataname) + '.ply').vertices
        pointcloud = np.asarray(pointcloud).reshape(-1,3)
        print('ply')

    elif os.path.exists(os.path.join(data_dir, dataname) + '.txt'):
        pointcloud = np.loadtxt(os.path.join(data_dir, dataname) + '.txt')[:,:3]
        print('txt')
    elif os.path.exists(os.path.join(data_dir, dataname) + '.obj'):
        pointcloud = trimesh.load(os.path.join(data_dir, dataname) + '.obj').vertices
        pointcloud = np.asarray(pointcloud).reshape(-1,3)
        print('obj')
    else:
        print('Only support .xyz or .ply data. Please make adjust your data.')
        exit()
    if pointcloud.shape[0]<pt_num:
        point_idx = np.random.choice(pointcloud.shape[0], pt_num, replace = True)
    else:
        point_idx = np.random.choice(pointcloud.shape[0], pt_num, replace = False)
    pointcloud = pointcloud[point_idx,:]
    max_val = np.max(pointcloud)
    min_val = np.min(pointcloud)
    shape_scale = np.max([np.max(pointcloud[:,0])-np.min(pointcloud[:,0]),np.max(pointcloud[:,1])-np.min(pointcloud[:,1]),np.max(pointcloud[:,2])-np.min(pointcloud[:,2])])
    shape_center = [(np.max(pointcloud[:,0])+np.min(pointcloud[:,0]))/2, (np.max(pointcloud[:,1])+np.min(pointcloud[:,1]))/2, (np.max(pointcloud[:,2])+np.min(pointcloud[:,2]))/2]
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale
    return pointcloud,shape_scale,shape_center


class DatasetNP:
    def __init__(self, conf, data_dir,dataname,pt_num):
        super(DatasetNP, self).__init__()
        self.device = torch.device('cuda')
        self.conf = conf
        self.data_dir = data_dir
        self.pt_num = pt_num
        pointcloud,shape_scale,shape_center=process_data(self.data_dir, dataname,self.pt_num)
        self.point_gt = np.asarray(pointcloud).reshape(-1,3)
        self.shape_scale = shape_scale
        self.shape_center = shape_center
        self.point_gt = torch.from_numpy(self.point_gt).to(self.device).float()
    def np_train_data(self):
        return self.point_gt, self.shape_scale, self.shape_center
