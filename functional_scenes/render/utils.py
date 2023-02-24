import time
import torch
import numpy as np
from pytorch3d.io import save_obj
from pytorch3d.ops import cubify
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import join_meshes_as_scene

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
)

# TODO: implement rotation
def create_cuboid_verts(dims, rotation, loc):
    # verts = quaternion_apply(rotation, cverts)
    verts = c_verts.clone().detach()
    verts *= torch.tensor(dims) # * 0.5
    verts += torch.tensor(loc)
    return verts



def create_cuboid(obj_data):
    beg_ts = time.time()
    dims  = torch.tensor(obj_data['dims'],
                         dtype=torch.float32)
    rotation = obj_data['orientation']
    loc = torch.tensor(obj_data['position'],
                       dtype=torch.float32)
    verts_features = torch.ones_like(c_verts)
    if obj_data['appearance'] == 'blue':
        verts_features[:, :2] *= 0.0  # blue

    textures = TexturesVertex(
        verts_features = verts_features[None])

    vs = c_verts * dims
    vs += loc
    vs = vs.unsqueeze(0)
    cf = c_faces.unsqueeze(0)
    m = Meshes(verts = vs, faces = cf,
               textures = textures)
    end_ts = time.time()
    # print('create_cuboid {}'.format(end_ts - beg_ts))
    return m

def create_cuboids(objects):
    mesh = list(map(create_cuboid, objects))
    mesh = join_meshes_as_scene(mesh)
    return mesh

OBSTACLE = 2

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
    verts_features = torch.ones_like(m.verts_padded())
    if color == 'blue':
        verts_features[:, :, :2] *= 0.0  # blue
    m.textures = TexturesVertex(
        verts_features = verts_features
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
