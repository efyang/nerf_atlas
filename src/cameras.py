import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .utils import dir_to_elev_azim, rotate_vector, rotation_matrix, vec_to_spherical_coords
from .neural_blocks import ( SkipConnMLP )
import random

# General Camera interface
@dataclass
class Camera(nn.Module):
  # samples from positions in [0,size] screen space to global
  def sample_positions(self, positions): raise NotImplementedError()

# A camera made specifically for generating rays from NeRF models
@dataclass
class NeRFCamera(Camera):
  cam_to_world:torch.tensor = None
  focal: float=None
  device:str ="cuda"
  near: float = None
  far: float = None

  def __len__(self): return self.cam_to_world.shape[0]

  @classmethod
  def identity(cls, batch_size: int, device="cuda"):
    c2w = torch.tensor([
      [1,0,0, 0],
      [0,1,0, 0],
      [0,0,1, 0],
    ], device=device).unsqueeze(0).expand(batch_size, 3, 4)
    return cls(cam_to_world=c2w, focal=0.5, device=device)

  # support indexing to get sub components of a camera
  def __getitem__(self, v):
    return NeRFCamera(cam_to_world=self.cam_to_world[v], focal=self.focal, device=self.device)

  def sample_positions(
    self,
    position_samples,
    size: int,
    with_noise=False,
  ):
    device=self.device
    u,v = position_samples.split([1,1], dim=-1)
    # u,v each in range [0, size]
    if with_noise:
      u = u + (torch.rand_like(u)-0.5)*with_noise
      v = v + (torch.rand_like(v)-0.5)*with_noise

    d = torch.stack([
        (u - size * 0.5) / self.focal,
        -(v - size * 0.5) / self.focal,
        -torch.ones_like(u),
    ], dim=-1)
    r_d = torch.sum(d[..., None, :] * self.cam_to_world[..., :3, :3], dim=-1)
    r_d = r_d.permute(2,0,1,3) # [H, W, B, 3] -> [B, H, W, 3]
    r_o = self.cam_to_world[..., :3, -1][:, None, None, :].expand_as(r_d)
    return torch.cat([r_o, r_d], dim=-1)

@dataclass
class FVRNeRFCamera(Camera):
  cam_to_world:torch.tensor = None
  ortho_scale: float=None
  device:str ="cuda"
  
  def __len__(self): return self.cam_to_world.shape[0]

  def __getitem__(self, v):
    return FVRNeRFCamera(cam_to_world=self.cam_to_world[v], ortho_scale=self.ortho_scale, device=self.device)

  def sample_positions(
    self,
    position_samples,
    size: int,
    with_noise=False,
  ):
    device=self.device
    u,v = position_samples.split([1,1], dim=-1)
    # u,v each in range [0, size]
    # if with_noise:
    #   u = u + ((torch.randn_like(u))*with_noise).clamp(-with_noise, with_noise)
    #   v = v + ((torch.randn_like(v))*with_noise).clamp(-with_noise, with_noise)

    s = torch.stack([
        (u - (size-1) * 0.5)/size, # * self.ortho_scale,
        -(v - (size-1) * 0.5)/size, # * self.ortho_scale,
        torch.zeros_like(u),
    ], dim=-1)

    # TODO: check that this actually make sense
    d = torch.stack([
        torch.zeros_like(u),
        torch.zeros_like(u),
        -torch.ones_like(u),
    ], dim=-1)

    r_d = torch.sum(d[..., None, :] * self.cam_to_world[..., :3, :3], dim=-1)
    slice_positions = torch.sum(s[..., None, :] * self.cam_to_world[..., :3, :3], dim=-1)
    # if with_noise:
      # a, b, c = (torch.randn(3, device=device)*with_noise).clamp(-with_noise, with_noise) * 3 * math.pi/180
      # slice_positions = torch.sum(slice_positions[..., None, :] * rotation_matrix(a,b,c), dim=-1)
      # a, b, c = (torch.randn(3, device=device)*with_noise).clamp(-with_noise, with_noise) * math.pi/180
      # r_d = torch.sum(r_d[..., None, :] * rotation_matrix(a,b,c), dim=-1)
    r_d = r_d.permute(2,0,1,3) # [H, W, B, 3] -> [B, H, W, 3]
    slice_positions = slice_positions.permute(2,0,1,3) # [H, W, B, 3] -> [B, H, W, 3]
    uvs = s.permute(2,0,1,3) # [H, W, B, 3] -> [B, H, W, 3]
    distances = uvs.norm(dim=-1)

    position_noise = None
    nonoise_positions = slice_positions
    if with_noise:
      # slice_positions += (torch.rand_like(slice_positions) - 0.5) * 0.1 *with_noise
      # slice_positions += (torch.randn_like(slice_positions) * with_noise * (distances[..., None])).clamp(-with_noise*3, with_noise*3)
      # distance_weighting = distances + 0.1
      # force everything to be jittered by adding 1e-2
      distance_weighting = distances + 1/size
      # position_noise = (torch.randn_like(slice_positions) * with_noise).clamp(-with_noise*3, with_noise*3)
      position_noise = (torch.randn_like(slice_positions) * with_noise * distance_weighting[..., None]).clamp(-with_noise*3, with_noise*3)
      slice_positions += position_noise
      # r_d = F.normalize(r_d + (torch.rand_like(r_d)-0.5)*0.05, dim=-1)
      # r_d += torch.randn_like(slice_positions).clamp(-with_noise*3, with_noise*3) * with_noise * 0.5
      # r_d = F.normalize(r_d, dim=-1)
    # return the slice through the origin parallel to image plane
    # slice_positions = vec_to_spherical_coords(slice_positions)
    return (torch.cat([slice_positions, r_d], dim=-1), position_noise, nonoise_positions)



def vec2skew(v):
  zero = torch.zeros(v.shape[:-1] + (1,), device=v.device, dtype=v.dtype)
  return torch.stack([
    torch.cat([zero, -v[..., 2:3], v[..., 1:2]], dim=-1),
    torch.cat([v[..., 2:3], zero, -v[..., 0:1]], dim=-1),
    torch.cat([-v[..., 1:2], v[..., 0:1], zero], dim=-1),
  ], dim=-2)

def exp(r):
  skew_r = vec2skew(r)
  norm_r = torch.linalg.norm(r).clamp(min=1e-6)
  eye = torch.eye(3, dtype=r.dtype, device=r.device)
  R = eye + \
    (norm_r.sin()/norm_r) * skew_r + \
    ((1 - norm_r.cos())/norm_r.square()) * (skew_r @ skew_r)
  return R

class OrthogonalCamera(Camera):
  def __init__(
    self,
    pos: torch.tensor,
    at: torch.tensor,
    up: torch.tensor,
    view_width:float=10.,
    view_height:float=None,
  ):
    super().__init__()
    if view_height is None: view_height = view_width
    self.pos = pos
    self.dir = dir = F.normalize(at - pos, dim=-1)
    self.right = r = view_width * F.normalize(torch.cross(dir, up), dim=-1)
    self.up = view_height * F.normalize(torch.cross(r, dir), dim=-1)
  def __len__(self): return self.pos.shape[0]
  def __getitem__(self, v):
    # FIXME to make this not use an odd part of the language? it should be fine
    cam = OrthogonalCamera.__new__()
    cam.pos = self.pos[v]
    cam.dir = self.dir[v]
    cam.right = self.up[v]
    cam.up = self.up[v]
    return cam
  def sample_positions(self, position_samples, size:int, with_noise=False):
    u, v = position_samples.split([1,1], dim=-1)
    u = 2 * ((u.squeeze(-1)/size) - 0.5)
    v = 2 * ((v.squeeze(-1)/size) - 0.5)
    r_o = self.pos[:, None, None, :] + \
      u[None, :, :, None] * self.right[:, None, None, :] +\
      v[None, :, :, None] * self.up[:, None, None, :]
    r_d = self.dir.expand_as(r_o)
    return torch.cat([r_o, r_d], dim=-1)

# The camera described in the NeRF-- paper
@dataclass
class NeRFMMCamera(Camera):
  # position
  t: torch.tensor = None
  # angle of rotation about axis
  r: torch.tensor = None
  # intrinsic focal positions
  focals: torch.tensor = None
  device:str ="cuda"

  def __len__(self): return self.t.shape[0]

  @classmethod
  def identity(cls, batch_size: int, device="cuda"):
    t = torch.zeros(batch_size, 3, dtype=torch.float, device=device, requires_grad=True)
    r = torch.zeros_like(t, requires_grad=True)
    # focals are for all the cameras and thus don't have batch dim
    focals = torch.tensor([0.7, 0.7], dtype=torch.float, device=device, requires_grad=True)
    return cls(t=t, r=r, focals=focals, device=device)

  def parameters(self): return [self.r, self.t, self.focals]

  def __getitem__(self, v):
    return NeRFMMCamera(t=self.t[v],r=self.r[v],focals=self.focals)

  def sample_positions(self, position_samples, size:int, with_noise=False):
    device=self.device
    u,v = position_samples.split(1, dim=-1)
    # u,v each in range [0, size]
    if with_noise:
      u = u + (torch.rand_like(u)-0.5)*with_noise
      v = v + (torch.rand_like(v)-0.5)*with_noise

    d = torch.stack(
      [
        (u - size * 0.5) / self.focals[..., 0],
        -(v - size * 0.5) / self.focals[..., 1],
        -torch.ones_like(u),
      ],
      dim=-1,
    )
    R = exp(self.r)
    r_d = torch.sum(d[..., None, :] * R, dim=-1)
    # normalize direction and exchange [W,H,B,3] -> [B,W,H,3]
    r_d = F.normalize(r_d, dim=-1).permute(2,0,1,3)
    r_o = self.t[:, None, None, :].expand_as(r_d)
    return torch.cat([r_o, r_d], dim=-1)

# learned time varying camera
class NeRFMMTimeCamera(Camera):
  def __init__(
    self,
    batch_size,
    # position
    translate: torch.tensor,
    # angle of rotation about axis
    rot: torch.tensor,
    # intrinsic focal positions
    focals: torch.tensor,
    delta_params: SkipConnMLP = None,
    device:str ="cuda"
  ):
    ...
    if delta_params is None:
      delta_params = SkipConnMLP(
        in_size=1, out=6,
        zero_init=True,
      )
    self.delta_params = delta_params
    self.focals = nn.Parameter(focals.requires_grad_())
    self.rot = nn.Parameter(translate.requires_grad_())
    self.translate = nn.Parameter(translate.requires_grad_())
  def __len__(self): return self.t.shape[0]
  def __getitem__(self, v):
    raise NotImplementedError()
    return NeRFMMTimeCamera(
      translate=self.translate[v], rot=self.rot[v], focal=self.focal,
      delta_params=self.delta_params, device=self.device
    )
  def sample_positions(
    self,
    position_samples,
    t,
    size:int,
    with_noise=False,
  ):
    raise NotImplementedError()
    ...

def lift(x,y,z,intrinsics, size):
    total_shape = x.shape
    fx = intrinsics[..., 0, 0, None].expand(total_shape)
    fy = intrinsics[..., 1, 1, None].expand(total_shape)
    cx = intrinsics[..., 0, 2, None].expand(total_shape) # size of image
    cy = intrinsics[..., 1, 2, None].expand(total_shape) # size of image
    sk = intrinsics[..., 0, 1, None].expand(total_shape)
    x = x.expand(total_shape)
    y = y.expand(total_shape)
    z = z.expand(total_shape)

    x_lift = (x - cx + cy*sk/fy - sk*y/fy) / fx * z
    y_lift = (y - cy) / fy * z

    # homogeneous
    return torch.stack([x_lift, y_lift, z, torch.ones_like(z)], dim=-1)

# A camera specifically for rendering DTU scenes as described in IDR.
@dataclass
class DTUCamera(Camera):
  pose: torch.Tensor = None
  intrinsic: torch.Tensor = None
  device: str = "cuda"
  def __len__(self): return self.pose.shape[0]
  def __getitem__(self, v):
    return DTUCamera(pose=self.pose[v], intrinsic=self.intrinsic[v], device=self.device)
  def sample_positions(
    self,
    position_samples,
    size:int=512,
    with_noise:bool=False,
  ):
    device = self.device
    pose = self.pose
    intrinsic = self.intrinsic
    # copied directly from https://github.com/lioryariv/idr/blob/main/code/utils/rend_util.py#L48
    if pose.shape[1] == 7: #In case of quaternion vector representation
      raise NotImplementedError()
    # In case of pose matrix representation
    else: r_o = pose[:, :3, 3]

    W, H, _ = position_samples.shape
    N = len(self)

    # 1600, 1200 is magical because it's the size of the original images given to us
    # In theory it would need to be changed if training on a different image set.
    normalize= torch.tensor([1600, 1200], device=device, dtype=torch.float)/size
    u,v = (position_samples * normalize)\
      .reshape(-1, 2)\
      .split([1,1], dim=-1)
    u = u.reshape(1, -1).expand(N, -1)
    v = v.reshape(1, -1).expand(N, -1)

    points = lift(u, v, torch.ones_like(u), intrinsics=intrinsic, size=size)

    world_coords = torch.bmm(pose, points.permute(0,2,1)).permute(0,2,1)[..., :3]

    r_o = r_o[:, None, :].expand_as(world_coords)
    r_d = F.normalize(world_coords - r_o, dim=-1)

    return torch.cat([r_o, r_d], dim=-1).reshape(N, W, H, 6)

