import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from src.nerf import CommonNeRF
import src.refl as refl

from .neural_blocks import (
  SkipConnMLP, UpdateOperator, FourierEncoder, PositionalEncoder, NNEncoder
)

class FVRNeRF(CommonNeRF):
  def __init__(
    self,
    out_features: int = 3,
    intermediate_size: int = 32,
    device: torch.device = "cuda",
    **kwargs,
  ):
    super().__init__(**kwargs, device=device)
    self.empty_latent = torch.zeros(1,1,1,1,0, device=device, dtype=torch.float)

    self.latent_size = 0
    # self.mlp = SkipConnMLP(
    #   in_size=3, out=1 + intermediate_size,
    #   latent_size = 0,
    #   num_layers=9, hidden_size=256,
    #   enc=FourierEncoder(input_dims=3, device=device),
    #   xavier_init=True,
    # )
    self.mlp = SkipConnMLP(
      in_size=3, out=out_features,
      latent_size = 0,
      num_layers=9, hidden_size=256,
      enc=FourierEncoder(input_dims=3, device=device),
      xavier_init=True,
    )
    self.refl = refl.View(
      out_features=out_features,
      # latent_size=self.latent_size+intermediate_size,
      latent_size=0
    )

  def forward(self, samples):
    pts, r_d = samples.split([3,3], dim=-1)
    # latent = self.curr_latent(pts.shape)
    latent = None
    #print(pts.shape)
    #print(latent.shape)
    #first_out = self.mlp(pts, latent if latent.shape[-1] != 0 else None)
    # first_out = self.mlp(pts)
    fourier = self.mlp(pts)
    # print("first_out max:", first_out.max().value, "min:", first_out.min())
    # intermediate = first_out[..., 1:]
    # view = r_d
    # fourier = self.refl(
    #   x=pts, view=view,
    #   latent=intermediate,
    # )
    #print(pts.shape)

    # print("fourier max:", fourier.max(), "min:", fourier.min())
    img = torch.fft.ifftn(fourier, dim=(1,2), s=(pts.size()[1], pts.size()[2]))
    # print("img max:", img.real.max(), "min:", img.real.min(), '\n')
    #print(img.real.shape)
    # TODO: maybe need to change norm so not dependent on size
    return img.real * pts.size()[1] * pts.size()[2] # + self.sky_color(view, self.weights)
