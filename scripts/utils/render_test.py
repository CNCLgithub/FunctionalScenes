#!/usr/bin/env python3

import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
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
src = '/renders/2e_1p_30s_matchedc3/1/scene.json'
with open(src, 'r') as f:
    scene = json.load(f)

graphics = SimpleGraphics((480, 720), device)
# graphics = SimpleGraphics((240, 360), device)
graphics.set_from_scene(scene)

beg_ts = time.time()
img = functional_scenes.render_scene_pil(scene, graphics)
end_ts = time.time()
print(end_ts - beg_ts)

img.save('/renders/test.png')
