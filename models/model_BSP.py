import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from .pointTransf_featureExtr import PointTransfFE
from .pointTransf_refinement import PointTransfRef
from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine
from models.transformer import (TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)
from utils.util import convert_spherical, convert_rectangular
from torch.distributions import MultivariateNormal
from scipy.spatial import KDTree
import torch.nn.functional as F

class ModelBSP(nn.Module):
  def __init__(
      self,
      pre_encoder,
      encoder,
      decoder,
      args
  ):
    super().__init__()
    self.pre_encoder = pre_encoder
    self.preenc_to_repr = GenericMLP(
      input_dim=args.enc_dim,
      hidden_dims=[args.enc_dim // 2],
      output_dim=3 + 3,
      norm_fn_name="bn1d",
      activation="relu",
      use_conv=True,
      output_use_activation=False,
      output_use_norm=False,
      output_use_bias=False,
    )
    self.points_lin1=nn.Linear(128, 128)
    self.points_lin2=nn.Linear(128, 128)
    self.points_lin3=nn.Linear(128, 128)
    self.points_lin4=nn.Linear(128, 128)
    self.points_lin5=nn.Linear(128, 3)
    self.relu = nn.ReLU()
    self.encoder = encoder
    if hasattr(self.encoder, "masking_radius"):
      hidden_dims = [args.enc_dim]
    else:
      hidden_dims = [args.enc_dim, args.enc_dim]
    self.encoder_to_decoder_projection = GenericMLP(
      input_dim=args.enc_dim,
      hidden_dims=hidden_dims,
      output_dim=args.dec_dim,
      norm_fn_name="bn1d",
      activation="relu",
      use_conv=True,
      output_use_activation=True,
      output_use_norm=True,
      output_use_bias=False,
    )
    self.pos_embedding = PositionEmbeddingCoordsSine(
      d_pos=args.dec_dim, pos_type='fourier', normalize=False
    )
    self.query_projection = GenericMLP(
      input_dim=args.enc_dim,
      hidden_dims=[args.dec_dim],
      output_dim=args.dec_dim,
      use_conv=True,
      output_use_activation=True,
      hidden_use_bias=True,
    )
    self.decoder = decoder
    self.build_mlp_heads(args.dec_dim)
    self.num_pts_sphere = args.preenc_npoints

  def build_mlp_heads(self, decoder_dim, mlp_dropout=0.0):
    mlp_func = partial(
      GenericMLP,
      norm_fn_name="bn1d",
      activation="relu",
      use_conv=True,
      hidden_dims=[decoder_dim // 2, decoder_dim // 4],
      dropout=mlp_dropout,
      input_dim=decoder_dim,
    )

    out_dim = 3
    upsampling_pc_head = mlp_func(output_dim=out_dim)

    mlp_heads = [
      ("upsampling_pc_head", upsampling_pc_head),
    ]
    self.mlp_heads = nn.ModuleDict(mlp_heads)

  def get_query_embeddings(self, parametr_xyz, point_cloud_dims):
    pos_embed = self.pos_embedding(parametr_xyz.contiguous(), input_range=point_cloud_dims)
    query_embed = self.query_projection(pos_embed)

    return query_embed

  def _break_up_pc(self, pc):
    xyz = pc[..., 0:3].contiguous()
    features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
    return xyz, features
  def normalize_points_to_unit_sphere(self,points):
    v_norm = torch.norm(points, dim=-1, keepdim=True)
    unit_sphere_points = points / v_norm
    return unit_sphere_points
  def knn_sample_points(self,points, k=10):
      uniform_points=np.loadtxt('./models/uniform_sphere.txt')
      tree = KDTree(uniform_points)
      distances, indices = tree.query(points, k=k)
      sampled_points = uniform_points[indices]
      return torch.tensor(sampled_points)
  def run_encoder(self, point_clouds):
    xyz, _ = self._break_up_pc(point_clouds)
    pre_enc_xyz, pre_enc_features, _ = self.pre_encoder(xyz)
    nptssphere = self.num_pts_sphere
    bsize = pre_enc_features.shape[0]
    inter_repr = self.preenc_to_repr(pre_enc_features).permute(0, 2, 1)
    coords = self.normalize_points_to_unit_sphere(normalize(inter_repr[..., :3], dim=-1))
    sampled_points = self.knn_sample_points(coords.detach().cpu().numpy(), k=10)
    sampled_points=sampled_points.requires_grad_(True)
    pre_enc_features = pre_enc_features.permute(2, 0, 1)
    return pre_enc_xyz, pre_enc_features, sampled_points

  def get_pts_predictions(self, point_features):
    point_features = point_features.permute(0, 2, 3, 1)
    num_layers, batch, channel, num_queries = (
      point_features.shape[0],
      point_features.shape[1],
      point_features.shape[2],
      point_features.shape[3],
    )
    point_features = point_features.reshape(num_layers * batch, channel, num_queries)
    pts_logits = self.mlp_heads["upsampling_pc_head"](point_features).transpose(1, 2)
    pts_logits = pts_logits.reshape(num_layers, batch, num_queries, -1)
    outputs = []
    res=[]
    for l in range(num_layers):
      points_prediction = {
        "points_logits": pts_logits[l, :, :, :3]
      }
      outputs.append(points_prediction)
      res.append(pts_logits[l, :, :, :3])
    return res

  def forward(self, inputs):
    point_clouds = inputs.reshape(1,-1,3)
    enc_xyz, enc_features, parametr = self.run_encoder(point_clouds)
    enc_features = self.encoder_to_decoder_projection(
      enc_features.permute(1, 2, 0)
    ).permute(2, 0, 1)
    min_range=np.min(point_clouds.detach().cpu().numpy(),axis=1).reshape(-1,3)
    max_range=np.max(point_clouds.detach().cpu().numpy(),axis=1).reshape(-1,3)
    min_range=torch.from_numpy(min_range).float().cuda()
    max_range=torch.from_numpy(max_range).float().cuda()
    point_cloud_dims = [
      min_range,
      max_range,
    ]
    query_embed = self.get_query_embeddings(parametr.squeeze(0).to(torch.float32),point_cloud_dims)
    enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)
    query_embed = query_embed.permute(0, 2, 1)
    enc_features=enc_features.repeat(1, query_embed.shape[1], 1)

    tgt = torch.zeros_like(query_embed)
    point_features = self.decoder(tgt, enc_features, query_pos=query_embed)[0]
    point_features=self.relu(self.points_lin1(point_features))
    point_features=self.relu(self.points_lin2(point_features))
    point_features=self.relu(self.points_lin3(point_features))
    point_features=self.relu(self.points_lin4(point_features))
    pts=self.points_lin5(point_features)
    return pts.reshape(-1,3)


def build_preencoder(args):
  if args.preencoder == 'pointTransformerFeatureExtr':
    preencoder = PointTransfFE(out_planes=args.enc_dim, nsample=args.preenc_nsample)
    return preencoder

def build_encoder(args):

  encoder_layer = TransformerEncoderLayer(
    d_model=args.enc_dim,
    nhead=args.enc_nhead,
    dim_feedforward=args.enc_ffn_dim,
    dropout=args.enc_dropout,
    activation=args.enc_activation,
  )
  encoder = TransformerEncoder(
    encoder_layer=encoder_layer, num_layers=args.enc_nlayers
  )
  return encoder


def build_decoder(args):

  decoder_layer = TransformerDecoderLayer(
    d_model=args.dec_dim,
    nhead=args.dec_nhead,
    dim_feedforward=args.dec_ffn_dim,
    dropout=args.dec_dropout,
  )
  decoder = TransformerDecoder(
    decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
  )
  return decoder

def build_BSP(args):
  pre_encoder = build_preencoder(args)
  encoder = build_encoder(args)
  decoder = build_decoder(args)
  model = ModelBSP(
    pre_encoder,
    encoder,
    decoder,
    args=args
  )
  return model
