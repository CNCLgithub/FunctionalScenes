#!/usr/bin/env python3

import os
import json
import time
import torch
import numpy as np
# import matplotlib.pyplot as plt
import functional_scenes
from functional_scenes.render import render_scene, SimpleGraphics
from pytorch3d.io import save_obj

# device = 'cpu'

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

print(device)
# device = 'cuda:1'
src = '/spaths/datasets/train_ddp_vss_pilot/1/scene.json'
with open(src, 'r') as f:
    scene = json.load(f)

graphics = SimpleGraphics((480, 720), device)
# graphics = SimpleGraphics((240, 360), device)
graphics.set_from_scene(scene)


# Z (up,down), Y (forward, backward), X (left right)
voxels = np.zeros((11, 48, 32), dtype = np.float32)

floor_voxels = voxels.copy()
floor_voxels[5, :, :] = 1.

wall_voxels = voxels.copy()
wall_voxels[5:, :, 0] = 1.
wall_voxels[5:, -1, :] = 1.

obs_voxels = voxels.copy()
obs_voxels[5:8, 24:26, 14:16] = 1. # an obstacle
obs_voxels[5:8, 15:18, 5:8] = 1. # an obstacle

scene['wall_voxels'] = wall_voxels
scene['obstacle_voxels'] = obs_voxels
scene['floor_voxels'] = floor_voxels
# scene['voxel_dim'] = np.array([16, 96, 24]) * 0.5
scene['voxel_dim'] = max(voxels.shape) * 0.5


beg_ts = time.time()
img = functional_scenes.render_scene_pil(scene, graphics)
end_ts = time.time()
print(end_ts - beg_ts)

img.save('/spaths/datasets/test.png')
