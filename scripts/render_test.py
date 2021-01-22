#!/usr/bin/env python3

import os
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from PIL import Image
from functional_scenes.render import render_scene, SimpleGraphics
from pytorch3d.io import save_obj
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene

# device = 'cpu'

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

print(device)
# device = 'cuda:1'
src = '/renders/2e_1p_30s/1/scene.json'
with open(src, 'r') as f:
    scene = json.load(f)

# graphics = SimpleGraphics((480, 720), device)
graphics = SimpleGraphics((240, 360), device)
graphics.set_from_scene(scene)

img, mesh = render_scene(scene, graphics)
save_obj( '/renders/test.obj', mesh.verts_packed(), mesh.faces_packed(),)

print(img.shape)
plt.figure()
plt.imshow(img)
plt.grid("off");
plt.axis("off");
plt.savefig('/renders/test.png')
