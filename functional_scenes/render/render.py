import numpy as np
from PIL import Image
from . utils import create_cuboid, create_cuboids
from abc import ABC, abstractmethod
from pytorch3d.structures import join_meshes_as_scene

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

    floor = create_cuboid(scene['floor']).to(graphics.device)
    mesh = create_cuboids(scene['objects']).to(graphics.device)
    mesh = join_meshes_as_scene([mesh, floor])
    result = graphics.render(mesh)
    return result[0, ..., :3].cpu().numpy(), mesh

def render_scene_pil(scene, graphics:AbstractGraphics):
    r, _ = render_scene(scene, graphics)
    r =  Image.fromarray((r * 255).astype(np.uint8),
                         mode = 'RGB')
    return r
