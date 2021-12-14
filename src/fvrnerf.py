import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
import random

from torch.nn.modules import activation
from src.nerf import CommonNeRF
import src.refl as refl
import math

from src.utils import dir_to_elev_azim, fat_sigmoid, vec_to_spherical_coords

from .neural_blocks import (
  SkipConnMLP, UpdateOperator, FourierEncoder, PositionalEncoder, NNEncoder
)

class FVRNeRF(CommonNeRF):
  def __init__(
    self,
    out_features: int = 6, # real and imag
    intermediate_size: int = 32,
    device: torch.device = "cuda",
    **kwargs,
  ):
    # workaround refl init
    self.refl= lambda: None

    super().__init__(**kwargs, device=device)
    self.empty_latent = torch.zeros(1,1,1,1,0, device=device, dtype=torch.float)

    self.intermediate_size = intermediate_size
    self.latent_size = 0
    self.mlp = SkipConnMLP(
      in_size=3, out=intermediate_size*2,
      latent_size = self.latent_size,
      num_layers=5, hidden_size=256,
      enc=FourierEncoder(input_dims=3, device=device),
      skip=3,
      xavier_init=True,
      activations=[F.leaky_relu]*8,
    )

    # self.middle_fenc = FourierEncoder(input_dims=3, device=device)
    # self.middle_mlp = SkipConnMLP(
    #   in_size=intermediate_size*2, out=intermediate_size*2,
    #   # latent_size=self.middle_fenc.output_dims(),
    #   latent_size=3,
    #   num_layers=3, hidden_size=256,
    #   # enc=FourierEncoder(input_dims=6, device=device),
    #   skip=3,
    #   xavier_init=True,
    #   activations=[F.leaky_relu]*8,
    # )

    self.out_mlp = SkipConnMLP(
      in_size=intermediate_size, out=out_features,
      latent_size=3,
      num_layers=3, hidden_size=256,
      # enc=FourierEncoder(input_dims=intermediate_size, device=device),
      skip=3,
      xavier_init=True,
      activations= [F.leaky_relu]*8,
    )

    # self.pool = nn.MaxPool2d(2)

    # self.conv = nn.Sequential(
    #   nn.Conv2d(intermediate_size*2+3, intermediate_size*2, kernel_size=3),
    #   nn.Conv2d(intermediate_size*2, intermediate_size*2, kernel_size=3),
    #   nn.Conv2d(intermediate_size*2, intermediate_size*2, kernel_size=3)
    # )

    # self.upconv = nn.Sequential(
    #   nn.ConvTranspose2d(intermediate_size*2, intermediate_size*2, kernel_size=3),
    #   nn.ConvTranspose2d(intermediate_size*2, intermediate_size*2, kernel_size=3),
    #   nn.ConvTranspose2d(intermediate_size*2, intermediate_size*2, kernel_size=3)
    # )

    self.refl = refl.MultFVRView(
      out_features=6,
      latent_size=256,
      # latent_size=0
    )

    if isinstance(self.refl, refl.LightAndRefl): self.refl.refl.act = self.feat_act
    else: self.refl.act = self.feat_act
 
  # def set_sigmoid(self, _):
  #   pass

  def forward(self, samples):
    pts, r_d = samples.split([3,3], dim=-1)
    # r_d = dir_to_elev_azim(r_d)
    # pts = vec_to_spherical_coords(pts)
    pmin = pts.shape[1]//2 + 1
    pts_halved = pts[:, :pmin, :, :]
    rd_halved = r_d[:, :pmin, :, :]
    fout = self.mlp(pts_halved)
    # middle_latent = torch.cat([pts_halved, rd_halved], dim=-1)
    # middle_latent = rd_halved
    # middle_latent = self.middle_fenc(middle_latent)
    # fout = self.middle_mlp(sout, middle_latent)
    re, im = fout.split([self.intermediate_size, self.intermediate_size], dim=-1)
    fout = torch.complex(re, im)
    conj_fout = torch.conj(torch.flip(fout[:, :pmin - 1, :, :], dims=(1,2)))
    out = torch.cat([fout, conj_fout], dim=1)
    # out[:, pmin:, :, :] = torch.conj(torch.flip(out[:, :pmin - 1, :, :], dims=(1,2)))

    # circle_ratio = 1
    # ii, jj = torch.meshgrid(
    #   torch.arange(out.shape[1], device="cuda", dtype=torch.float) - out.shape[1]/2 + 0.5,
    #   torch.arange(out.shape[2], device="cuda", dtype=torch.float) - out.shape[2]/2 + 0.5,
    # )
    # positions = torch.stack([ii.transpose(-1, -2), jj.transpose(-1, -2)], dim=-1)
    # circle_mask = positions.norm(dim=-1) > circle_ratio * out.shape[1]//2
    # out[:, circle_mask, :] = 0

    # # [N, H, W, C] -> [N, C, H, W]
    # out = torch.cat([out.real, out.imag, r_d], dim=-1)
    # out = out.permute(0, 3, 1, 2)
    # out = self.conv(out)
    # out = self.upconv(out)
    # # [N, C, H, W] -> [N, H, W, C]
    # out = out.permute(0, 2, 3, 1)
    # re, im = out.split([self.intermediate_size, self.intermediate_size], dim=-1)
    # out = torch.complex(re, im)
    fft = torch.fft.ifftshift(out, dim=(1,2))
    img = torch.fft.ifftn(fft, dim=(1,2), s=(out.size()[1], out.size()[2]), norm="ortho")
    # view = dir_to_elev_azim(r_d)
    img = self.out_mlp(img.real, r_d)
    # img = img.permute(0, 3, 1, 2)
    # img = self.pool(img)
    # img = img.permute(0, 2, 3, 1)
    # img = fat_sigmoid(img)
    # TODO: maybe need to change norm so not dependent on size
    return (img, out) # + self.sky_color(view, self.weights)

def empty():
  return None

class LearnedFVR(CommonNeRF):
  def __init__(
    self,
    out_features: int = 3,
    intermediate_size: int = 32,
    device: torch.device = "cuda",
    **kwargs,
  ):
    # workaround refl init
    self.refl= empty 
    self.mlp = empty

    super().__init__(**kwargs, device=device)
    self.empty_latent = torch.zeros(1,1,1,1,0, device=device, dtype=torch.float)

    self.latent_size = 0

    self.refl = refl.View(
      out_features=0,
      latent_size=0,
    )

    self.resolution = 129
    self.intermediate_size = intermediate_size
    self.fourier_volume = nn.parameter.Parameter(torch.zeros(self.resolution, self.resolution, self.resolution, self.intermediate_size * 2, device=device))

    self.out_mlp = SkipConnMLP(
      in_size=intermediate_size, out=out_features,
      latent_size=3,
      num_layers=3, hidden_size=256,
      # enc=FourierEncoder(input_dims=intermediate_size, device=device),
      skip=3,
      xavier_init=True,
      activations= [F.leaky_relu]*8,
    )
    if isinstance(self.refl, refl.LightAndRefl): self.refl.refl.act = self.feat_act
    else: self.refl.act = self.feat_act
 
  # def set_sigmoid(self, _):
  #   pass
  def trilinear_interp_vec(self, v):
    v1 = v.ceil().to(torch.long)
    v0 = v.floor().to(torch.long)
    v1[v1 == v0] += 1

    x1 = v1[..., 0]
    y1 = v1[..., 1]
    z1 = v1[..., 2]
    x0 = v0[..., 0]
    y0 = v0[..., 1]
    z0 = v0[..., 2]
    vd = (v - v0) / (v1 - v0)
    xd = vd[..., 0]
    yd = vd[..., 1]
    zd = vd[..., 2]

    c000 = self.fourier_volume[z0, y0, x0]
    c100 = self.fourier_volume[z0, y0, x1]
    c010 = self.fourier_volume[z0, y1, x0]
    c001 = self.fourier_volume[z1, y0, x0]
    c110 = self.fourier_volume[z0, y1, x1]
    c011 = self.fourier_volume[z1, y1, x0]
    c101 = self.fourier_volume[z1, y0, x1]
    c111 = self.fourier_volume[z1, y1, x1]

    c00 = c000 * (1-xd[..., None]) + c100 * xd[..., None]
    c01 = c001 * (1-xd[..., None]) + c101 * xd[..., None]
    c10 = c010 * (1-xd[..., None]) + c110 * xd[..., None]
    c11 = c011 * (1-xd[..., None]) + c111 * xd[..., None]

    c0 = c00 * (1-yd[..., None]) + c10 * yd[..., None]
    c1 = c01 * (1-yd[..., None]) + c11 * yd[..., None]

    c = c0 * (1-zd[..., None]) + c1 * zd[..., None]
    re, im = c.split([self.intermediate_size, self.intermediate_size], dim=-1)
    return torch.complex(re, im)

  def forward(self, samples):
    size = 65
    maxdiam = math.ceil(size * math.sqrt(2))+2

    pts, r_d = samples.split([3,3], dim=-1)
    pts = pts*(self.resolution/(maxdiam + 3)) + self.resolution//2
    out = torch.zeros_like(pts, dtype=torch.complex64)

    out = self.trilinear_interp_vec(pts)

    circle_ratio = 0.5
    ii, jj = torch.meshgrid(
      torch.arange(out.shape[1], device="cuda", dtype=torch.float) - out.shape[1]/2 + 0.5,
      torch.arange(out.shape[2], device="cuda", dtype=torch.float) - out.shape[2]/2 + 0.5,
    )
    positions = torch.stack([ii.transpose(-1, -2), jj.transpose(-1, -2)], dim=-1)
    circle_mask = positions.norm(dim=-1) > circle_ratio * out.shape[1]//2
    out[:, circle_mask, :] = 0

    fft = torch.fft.ifftshift(out, dim=(1,2))
    img = torch.fft.ifftn(fft, dim=(1,2), s=(out.size()[1], out.size()[2]), norm="ortho")

    img = self.out_mlp(img.real, r_d)

    return (img, out) # + self.sky_color(view, self.weights)
