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



def render_scene(scene, graphics:AbstractGraphics):
    # assuming graphics is already configured on template scene

    # Load Objects / Tiles
    # Setup floor
    # floor = create_floor(scene['floor'])
    # ceiling = create_ceiling(scenedict['ceiling'])
    # meshes = [floor, ceiling] + list(map(create_cuboid2, scene['objects']))

    mesh = scene_to_mesh(scene, graphics.device)
    result = graphics.render(mesh)
    return result[0, ..., :3].cpu().numpy(), mesh

def render_scene_pil(scene, graphics:AbstractGraphics):
    r, _ = render_scene(scene, graphics)
    r =  Image.fromarray((r * 255).astype(np.uint8),
                         mode = 'RGB')
    return r

def render_mesh_pil(mesh, graphics:AbstractGraphics):
    r = graphics.render(mesh)
    r = r[0, ..., :3].cpu().numpy()
    r =  Image.fromarray((r * 255).astype(np.uint8),
                         mode = 'RGB')
    return r


def render_scene_batch(scenes, graphics:AbstractGraphics):
    beg_ts = time.time()
    n = len(scenes)
    device = graphics.device
    # mesh = list(map(lambda x: scene_to_mesh(x, device), scenes))
    # mesh = join_meshes_as_batch(mesh)
    mesh = join_meshes_as_batch([scene_to_mesh(x, device) for x
                                 in scenes]).to(device)
    end_ts = time.time()
    # print('render_scene_batch mesh {}'.format(end_ts - beg_ts))
    beg_ts = end_ts
    result = graphics.render(mesh)
    end_ts = time.time()
    # print('render_scene_batch render {}'.format(end_ts - beg_ts))
    # result = result.cpu().numpy()
    # return (result[:n] * 255).astype(np.uint8)
    return result[:n, ..., :3].permute(0, 3, 1, 2)
