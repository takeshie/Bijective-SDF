import torch
import tqdm
import numpy as np
import kaolin as kal
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import torch.nn.functional as F
import trimesh
from models.tet_init import tet_init

class MeshTrainer:
    def __init__(self, device='cuda:3', iter=100000, learning_rate=1e-5, voxel_grid_res=72):
        self.device = device
        self.iter = iter
        self.learning_rate = learning_rate
        self.voxel_grid_res = voxel_grid_res
        self.sdf_net = tet_init.to(self.device)
        self.ChamferDisL1 = ChamferDistanceL1().to(self.device)
        self.fc = kal.non_commercial.FlexiCubes(self.device)
        self.optimizer = torch.optim.Adam(self.sdf_net.parameters(), lr=self.learning_rate)
        self.x_nx3=None
        self.cube_fx8=None
        self.gt=None
        self.x_nx3, self.cube_fx8 = self.fc.construct_voxel_grid(self.voxel_grid_res)
        self.x_nx3 *= 1.5
        self.x_nx3.requires_grad=True
    def load_pts(self, points):
        self.gt = points
        vmin, vmax = self.gt.min(dim=0)[0], self.gt.max(dim=0)[0]
        scale = 1.8 / torch.max(vmax - vmin).item()
        self.gt = self.gt - (vmax + vmin) / 2
        self.gt = self.gt * scale
    def initialize_model(self,finetune_net=False,init_lr=1e-4,finetune_epoch=3000):
        self.sdf_net.pre_train_np(iter=finetune_epoch,finetune_net=finetune_net,lr=init_lr)
    def gradient(self, x, sdf):
        x.requires_grad_(True)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

    def laplace_regularizer_const(self, mesh_verts, mesh_faces):
        term = torch.zeros_like(mesh_verts)
        norm = torch.zeros_like(mesh_verts[..., 0:1])

        v0 = mesh_verts[mesh_faces[:, 0], :]
        v1 = mesh_verts[mesh_faces[:, 1], :]
        v2 = mesh_verts[mesh_faces[:, 2], :]

        term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
        term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
        term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

        two = torch.ones_like(v0) * 2.0
        norm.scatter_add_(0, mesh_faces[:, 0:1], two)
        norm.scatter_add_(0, mesh_faces[:, 1:2], two)
        norm.scatter_add_(0, mesh_faces[:, 2:3], two)

        term = term / torch.clamp(norm, min=1.0)

        return torch.mean(term**2)

    def predict_for_batch(self, tet_verts, batch_size):
        n = tet_verts.shape[0]
        all_sdf = []
        all_deforms = []
        for i in range(0, n, batch_size):
            batch_verts = tet_verts[i:i + batch_size]
            pred = self.sdf_net(batch_verts)
            sdf = pred[:, 0]
            verts_gradient = self.gradient(batch_verts, sdf).squeeze()
            grad_norm = F.normalize(verts_gradient, dim=1)
            deform = batch_verts - torch.tanh(grad_norm * sdf.unsqueeze(1))
            all_sdf.append(sdf)
            all_deforms.append(deform)
        all_sdf = torch.cat(all_sdf, dim=0)
        all_deforms = torch.cat(all_deforms, dim=0)
        return all_sdf, all_deforms

    def run_iteration(self, it, data_name,save_dir,shape_scale,shape_center,val_flag=False):
        sdf, grid_verts = self.predict_for_batch(tet_verts=self.x_nx3, batch_size=100000)
        vertices, faces, _ = self.fc(grid_verts, sdf, self.cube_fx8, self.voxel_grid_res, training=True)
        laplas= self.laplace_regularizer_const(vertices, faces)
        if it % 50 == 0:
            with torch.no_grad():
                vertices, faces, L_dev = self.fc(
                    grid_verts, sdf, self.cube_fx8, self.voxel_grid_res,training=False)
                res=vertices.detach().cpu().numpy()
                shape_center=np.asarray(shape_center)
                res=(vertices*shape_scale).detach().cpu().numpy()
                res+=shape_center
                mesh = trimesh.Trimesh(res, faces.detach().cpu().numpy())
                dir='{}/val_{}.ply'.format(save_dir,it)
                mesh.export('{}/val_{}.ply'.format(save_dir,it))
                print('saved at:{}'.format(dir))
        return laplas,sdf,vertices