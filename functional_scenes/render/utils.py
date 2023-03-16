import time
import torch
import numpy as np
from pytorch3d.io import save_obj
from pytorch3d.ops import cubify
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import join_meshes_as_scene


# TODO: implement rotation
def create_cube(pos, dims, color, device):
    c_verts = torch.tensor(np.array(
        [
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
        ]),
        dtype=torch.float32,
        device = device,
    )
    # faces corresponding to a unit cube: 12x3
    c_faces = torch.tensor(np.array(
        [
            [0, 1, 2],
            [1, 3, 2],  # left face: 0, 1
            [2, 3, 6],
            [3, 7, 6],  # bottom face: 2, 3
            [0, 2, 6],
            [0, 6, 4],  # front face: 4, 5
            [0, 5, 1],
            [0, 4, 5],  # up face: 6, 7
            [6, 7, 5],
            [6, 5, 4],  # right face: 8, 9
            [1, 7, 3],
            [1, 5, 7],  # back face: 10, 11
        ]),
        dtype=torch.int64,
        device = device,
    )

    color = torch.tensor(color, device=device)
    verts_features = torch.tile(color, (1, 8, 1))
    textures = TexturesVertex(verts_features = verts_features)

    c_verts *= torch.tensor(dims, device=device)
    c_verts += torch.tensor(pos,  device=device)
    m = Meshes(verts = c_verts.unsqueeze(0),
               faces = c_faces.unsqueeze(0),
               textures = textures)
    return m

def pad_voxels(voxels):
    max_d = max(voxels.shape)
    padding = []
    for i in range(voxels.ndim):
        p = int(np.floor(0.5 * (max_d - voxels.shape[i])))
        padding.append((p, p))

    return np.pad(voxels, padding, 'constant')

def from_voxels(voxels, factor, device, color = ''):
    voxels = torch.from_numpy(voxels).to(device)[None]
    m = cubify(voxels, 0.5, align = 'corner')
    # scale vertices to correct dimensions
    m.scale_verts_(factor)
    n = m.verts_padded()
    vfs = torch.ones((1, n.shape[1], 4),
                     device = device,
                     dtype = torch.float32)
    if color == 'blue':
        vfs[:, :, :2] *= 0.0  # blue
    m.textures = TexturesVertex(
        verts_features = vfs
    )
    return m

def scene_to_mesh(scene, device):
    if 'voxel_dim' in scene:
        f = scene['voxel_dim']
        wall_voxels = pad_voxels(scene['wall_voxels'])
        wall_mesh = from_voxels(wall_voxels, f, device)
        obs_voxels = pad_voxels(scene['obstacle_voxels'])
        obs_mesh  = from_voxels(obs_voxels, f, device, color = 'blue')
        # print(obs_mesh.verts_packed())
        # floor_voxels = pad_voxels(scene['floor_voxels'])
        # floor_mesh  = from_voxels(floor_voxels, f, device)
        mesh = join_meshes_as_scene([
            wall_mesh,
            obs_mesh,
            # floor_mesh,
            # create_cuboid(scene['ceiling']).to(device),
            # create_cuboid(scene['floor']).to(device)
        ])
    else:
        floor = create_cuboid(scene['floor']).to(device)
        mesh = create_cuboids(scene['objects']).to(device)
        mesh = join_meshes_as_scene([mesh, floor])
    return mesh
