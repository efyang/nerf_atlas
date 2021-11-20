import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
import random

from torch.nn.modules import activation
from src.nerf import CommonNeRF
import src.refl as refl
import math

from src.utils import fat_sigmoid

from .neural_blocks import (
  SkipConnMLP, UpdateOperator, FourierEncoder, PositionalEncoder, NNEncoder
)

def csinact(x):
  return torch.complex(torch.sin(x.real), torch.sin(x.imag))

def rsin(x):
  return F.relu(torch.sin(x))

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

    self.latent_size = 0
    self.mlp = SkipConnMLP(
      # in_size=3, out=1 + intermediate_size,
      # in_size=3, out=intermediate_size,
      in_size=3, out=out_features+256,
      latent_size = self.latent_size,
      num_layers=5, hidden_size=256,
      # enc=FourierEncoder(input_dims=3, device=device),
      skip=3,
      siren_init=True,
      activations=[torch.sin]*5,
    )

    # self.mlp2 = SkipConnMLP(
      # in_size=3, out=1 + intermediate_size,
      # in_size=2*intermediate_size, out=out_features,
      # latent_size = self.latent_size,
      # num_layers=2, hidden_size=256,
      # enc=FourierEncoder(input_dims=2*intermediate_size, device=device),
    #   kaiming_init=True,
    # )
    # self.mlp = SkipConnMLP(
    #   in_size=3, out=out_features,
    #   latent_size = 0,
    #   num_layers=9, hidden_size=256,
    #   enc=FourierEncoder(input_dims=3, device=device),
    #   xavier_init=True,
    # )
    self.refl = refl.MultFVRView(
      out_features=out_features,
      latent_size=3+256,
      # latent_size=0
    )


    if isinstance(self.refl, refl.LightAndRefl): self.refl.refl.act = self.feat_act
    else: self.refl.act = self.feat_act
 
  # def set_sigmoid(self, _):
  #   pass

  def forward(self, samples):
    pts, r_d = samples.split([3,3], dim=-1)
    out = torch.zeros_like(pts, dtype=torch.complex64)
    pmin = pts.size()[1]//2 + 1
    pts = pts[:, :pmin, :, :]
    r_d = r_d[:, :pmin, :, :]
    # latent = self.curr_latent(pts.shape)
    latent = None
    view = r_d
    # hemisphere = torch.sign(view[:, :, :, [-1]])
    # signed_pts = torch.cat([pts, hemisphere], dim=-1)
    # first_out = self.mlp(pts)
    fout, first_latent = self.mlp(pts).split([6, 256], dim=-1)
    re, im = fout.split([3,3], dim=-1)
    coeff = torch.complex(re, im) * out.size()[1]
    # fourier = self.mlp(pts)
    # fout = fout * pts.size()[1]
    # foutfft = torch.fft.fft(fout, dim=-1)
    # fin2 = torch.cat([foutfft.real, foutfft.imag], dim=-1)
    # fourier = self.mlp2(fin2)
    fourier = self.refl(
      x = coeff, latent=torch.cat([pts, first_latent], dim=-1), view=view,
    )
    coeff = fourier

    # re, im = fourier.split([3,3], dim=-1)
    # coeff = torch.complex(re, im) * out.size()[1]
    # real inputs always have hermitian fft
    coefft = torch.flip(torch.conj(coeff), (1,2))
    # coeff[:, :pts.size()[1]//2 + 2, :, :] = coefft[:, :pts.size()[1]//2 + 2, :, :]
    out[:, :pmin, :, :] = coeff
    out[:, pmin - 1:, :, :] = coefft

    fft = torch.fft.ifftshift(out, dim=(1,2))
    img = torch.fft.ifftn(fft, dim=(1,2), s=(out.size()[1], out.size()[2]), norm="ortho")
    img = img.real
    img = fat_sigmoid(img)
    # img = (torch.sin(img)+1)/2
    # TODO: predict coefficients of some set of basis functions (fourier basis in this case)
    # to reduce the output solution space
    # TODO: maybe need to change norm so not dependent on size
    return (img, out) # + self.sky_color(view, self.weights)
