import torch
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
    dims  = torch.tensor(obj_data['dims'], dtype=torch.float32)
    rotation = obj_data['orientation']
    loc = torch.tensor(obj_data['position'], dtype=torch.float32)
    verts_features = torch.ones_like(c_verts)
    if obj_data['appearance'] == 'blue':
        verts_features[:, :2] *= 0.0  # blue

    textures = TexturesVertex(
        verts_features = verts_features[None])
    vs = c_verts * dims
    vs += loc
    m = Meshes(verts = [vs], faces = [c_faces],textures =textures)
    return m

def create_cuboids(objects):
    mesh = list(map(create_cuboid, objects))
    mesh = join_meshes_as_scene(mesh)
    return mesh
