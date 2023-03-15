import time
import numpy as np
from PIL import Image
from . utils import create_cuboid, create_cuboids, scene_to_mesh
from abc import ABC, abstractmethod
from pytorch3d.structures import join_meshes_as_batch

class AbstractGraphics(ABC):

    @abstractmethod
    def set_from_scene(self):
        pass

    @abstractmethod
    def set_camera(self):
        pass

    @abstractmethod
    def set_lighting(self):
        pass

    @abstractmethod
    def render(self):
        pass


def render_mesh_pil(mesh, graphics:AbstractGraphics):
    r = graphics.render(mesh)
    r = r[0, ..., :3].cpu().numpy()
    r =  Image.fromarray((r * 255).astype(np.uint8),
                         mode = 'RGB')
    return r


def render_mesh_batch(meshes, graphics:AbstractGraphics):
    # beg_ts = time.time()
    # device = graphics.device
    n = len(meshes)
    mesh = join_meshes_as_batch(meshes)# REVIEW: add `.to(device)` ?
    # end_ts = time.time()
    # print('render_scene_batch mesh {}'.format(end_ts - beg_ts))
    # beg_ts = end_ts
    result = graphics.render(mesh)
    # end_ts = time.time()
    # print('render_scene_batch render {}'.format(end_ts - beg_ts))
    return result[:n, ..., :3].permute(0, 3, 1, 2)

def render_mesh_single(mesh, graphics:AbstractGraphics):
    result = graphics.render(mesh)
    return result[0:1, ..., :3].permute(0, 3, 1, 2).cpu().numpy()

def batch_render_and_stats(scenes, graphics:AbstractGraphics):
    imgs = render_mesh_batch(scenes, graphics)
    mu = imgs.mean(axis = 0).cpu().numpy()
    sd = imgs.std(axis = 0).cpu().numpy()
    return (mu, sd)
