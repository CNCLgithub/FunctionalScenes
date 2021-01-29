import time
import torch
from pytorch3d.io import save_obj
from pytorch3d.ops import cubify
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import join_meshes_as_scene

c_verts = torch.tensor(
    [
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
    ],
    dtype=torch.float32,
)
# faces corresponding to a unit cube: 12x3
c_faces = torch.tensor(
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
    ],
    dtype=torch.int64,
)

# TODO: implement rotation
def create_cuboid_verts(dims, rotation, loc):
    # verts = quaternion_apply(rotation, cverts)
    verts = c_verts.clone().detach()
    verts *= torch.tensor(dims) * 0.5
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
    m = Meshes(verts = [vs], faces = [c_faces],
               textures =textures)
    end_ts = time.time()
    # print('create_cuboid {}'.format(end_ts - beg_ts))
    return m

def create_cuboids(objects):
    mesh = list(map(create_cuboid, objects))
    mesh = join_meshes_as_scene(mesh)
    return mesh

def from_voxels(voxels, color):
    device = torch.cuda.current_device()
    voxels = torch.from_numpy(voxels).to(device)
    voxels = voxels.unsqueeze(0)
    m = cubify(voxels, 0.5, align = 'center')
    vs = m.verts_packed() * torch.Tensor([11, 20, 3.0]).to(device)
    vs += torch.Tensor([0.0, 0.0, 3.5]).to(device)
    verts_features = torch.ones_like(vs)
    faces  = m.faces_packed()
    if color == 'blue':
        verts_features[:, :2] *= 0.0  # blue

    textures = TexturesVertex(
        verts_features = verts_features[None])
    return Meshes(verts = [vs], faces = [faces],
                  textures = textures)

def scene_to_mesh(scene, device):
    beg_ts = time.time()
    floor = create_cuboid(scene['floor']).to(device)
    if 'objects' in scene:
        mesh = create_cuboids(scene['objects']).to(device)
    else:
        walls = from_voxels(scene['walls'], '')
        furn = from_voxels(scene['furniture'], 'blue')
        mesh = join_meshes_as_scene([walls, furn])
    end_ts = time.time()
    # print('scene to mesh {}'.format(end_ts - beg_ts))
    mesh = join_meshes_as_scene([mesh, floor])
    # save_obj('/renders/test.obj', mesh.verts_packed(),
    #          mesh.faces_packed())
    return mesh
