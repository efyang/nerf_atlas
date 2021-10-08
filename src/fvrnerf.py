import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .neural_blocks import (
  SkipConnMLP, UpdateOperator, FourierEncoder, PositionalEncoder, NNEncoder
)
from .utils import (
  dir_to_elev_azim, autograd, sample_random_hemisphere, laplace_cdf, load_sigmoid,
)

# sigmoids which shrink or expand the total range to prevent gradient vanishing,
# or prevent it from representing full density items.
# fat sigmoid has no vanishing gradient, but thin sigmoid leads to better outlines.
def fat_sigmoid(v, eps: float = 1e-3): return v.sigmoid() * (1+2*eps) - eps
def thin_sigmoid(v, eps: float = 1e-2): return fat_sigmoid(v, -eps)
def cyclic_sigmoid(v, eps:float=-1e-2,period:int=5):
  return ((v/period).sin()+1)/2 * (1+2*eps) - eps

class FVRNeRF(nn.Module):
  def __init__(
    self,
    out_features: int = 3,
    device: torch.device = "cuda",

    instance_latent_size: int = 0,
    per_pixel_latent_size: int = 0,
    per_point_latent_size: int = 0,

    sigmoid_kind: str = "thin",
    bg: str = "black",
  ):
    super().__init__()
    self.empty_latent = torch.zeros(1,1,1,1,0, device=device, dtype=torch.float)
    self.per_pixel_latent_size = per_pixel_latent_size
    self.per_pixel_latent = None

    self.instance_latent_size = instance_latent_size
    self.instance_latent = None

    self.per_pt_latent_size = per_point_latent_size
    self.per_pt_latent = None

    self.set_bg(bg)
    self.set_sigmoid(sigmoid_kind)

    self.latent_size = 32
    self.mlp = SkipConnMLP(
      in_size=3, out=1 + out_features,
      latent_size = self.latent_size,
      num_layers=9, hidden_size=256,
      enc=FourierEncoder(input_dims=3, device=device),
      xavier_init=True,
    )

  def set_sigmoid(self, kind="thin"): self.feat_act = load_sigmoid(kind)
  def set_per_pt_latent(self, latent):
    assert(latent.shape[-1] == self.per_pt_latent_size), \
      f"expected latent in [T, B, H, W, L={self.per_pixel_latent_size}], got {latent.shape}"
    assert(len(latent.shape) == 5), \
      f"expected latent in [T, B, H, W, L], got {latent.shape}"
    self.per_pt_latent = latent
  def set_per_pixel_latent(self, latent):
    assert(latent.shape[-1] == self.per_pixel_latent_size), \
      f"expected latent in [B, H, W, L={self.per_pixel_latent_size}], got {latent.shape}"
    assert(len(latent.shape) == 4), \
      f"expected latent in [B, H, W, L], got {latent.shape}"
    self.per_pixel_latent = latent
  def set_instance_latent(self, latent):
    assert(latent.shape[-1] == self.instance_latent_size), "expected latent in [B, L]"
    assert(len(latent.shape) == 2), "expected latent in [B, L]"
    self.instance_latent = latent

  # gets the current latent vector for this NeRF instance
  def curr_latent(self, pts_shape):# -> ["T", "B", "H", "W", "L_pp + L_inst"]:
    curr = self.empty_latent.expand(pts_shape[:-1] + (0,)) if self.per_pt_latent is None \
      else self.per_pt_latent

    if self.per_pixel_latent is not None:
      ppl = self.per_pixel_latent[None, ...].expand(pts_shape[:-1] + (-1,))
      curr = torch.cat([curr, ppl], dim=-1)

    if self.instance_latent is not None:
      il = self.instance_latent[None, :, None, None, :].expand_as(pts_shape[:-1] + (-1,))
      curr = torch.cat([curr, il], dim=-1)

    return curr

  def forward(self, pts, pts_center):
    pts = pts - pts_center
    latent = self.curr_latent(pts.shape)
    f_c = self.mlp(pts, latent)
    return f_c
