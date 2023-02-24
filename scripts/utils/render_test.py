#!/usr/bin/env python3

import os
import json
import time
import torch
import numpy as np
# import matplotlib.pyplot as plt
import functional_scenes
from functional_scenes.render import (render_scene,
                                      SimpleGraphics)
from functional_scenes.render.utils import from_voxels

from pytorch3d.structures import join_meshes_as_scene

# device = 'cpu'

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# src = '/spaths/datasets/vss_pilot/1/scene.json'
# with open(src, 'r') as f:
#     scene = json.load(f)

# graphics = SimpleGraphics((480, 720), device)
graphics = SimpleGraphics((240, 360), device)
graphics.set_camera({'position': [0., -20.0, -10.0]})

# const tile_height = 5.0
# const obstacle_height = 0.3 * tile_height

# Z (up,down), Y (forward, backward), X (left right)
# voxels = np.zeros((11, 32, 32), dtype = np.float32)
voxels = np.zeros((32, 32, 32), dtype = np.float32)

floor_voxels = voxels.copy()
# floor_voxels[5, :, :] = 1.
floor_voxels[0, :, :] = 1.

wall_voxels = voxels.copy()
# wall_voxels[5:, :, 0] = 1.
# wall_voxels[5:, -1, :] = 1.
wall_voxels[1:6, :, 0] = 1. # left wall
wall_voxels[1:6, -1, :] = 1. # back wall
wall_voxels[1:6, :, -1] = 1. # right wall

obs_voxels = voxels.copy()
# obs_voxels[5:8, 24:26, 14:16] = 1. # an obstacle
# obs_voxels[5:8, 15:18, 5:8] = 1. # an obstacle
obs_voxels[1:3, 24:26, 14:16] = 1. # an obstacle
obs_voxels[1:3, 15:18, 5:8] = 1. # an obstacle

# scene = {}
# scene['wall_voxels'] = wall_voxels
# scene['obstacle_voxels'] = obs_voxels
# scene['floor_voxels'] = floor_voxels
# # scene['voxel_dim'] = np.array([16, 96, 24]) * 0.5
# scene['voxel_dim'] = max(voxels.shape) * 0.5
vdim = max(voxels.shape) * 0.5

floor_mesh = from_voxels(floor_voxels, vdim, device)
wall_mesh = from_voxels(wall_voxels, vdim, device)
beg_ts = time.time()
obs_mesh = from_voxels(obs_voxels, vdim, device, color='blue')
end_ts = time.time()
print(end_ts - beg_ts)

scene_mesh = join_meshes_as_scene([floor_mesh, wall_mesh, obs_mesh])

beg_ts = time.time()
img = functional_scenes.render.render_mesh_pil(scene_mesh, graphics)
end_ts = time.time()
print(end_ts - beg_ts)
print(graphics.device)

img.save('/spaths/datasets/test.png')
