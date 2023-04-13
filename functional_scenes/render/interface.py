import numpy as np
import drjit as dr
import mitsuba as mi


variant = "cuda_ad_rgb" if "cuda_ad_rbg" in mi.variants() else "scalar_rgb"
mi.set_variant(variant)

from mitsuba import ScalarTransform4f as T


def initialize_scene(dimensions,
                     door,
                     res):

    camera_pos = [-0.55*dimensions[0], 0, 3.75]

    floor_dims = [dimensions[0], dimensions[1], 1]
    floor_t = T.scale(floor_dims)
    ceiling_t = T.translate([0, 0, dimensions[2]]).\
        scale(floor_dims)

    wall_dims = [dimensions[0], dimensions[2], 1]
    left_wall_pos = [0, 0.5*dimensions[1], 0.5*dimensions[2]]
    left_wall_t = T.translate(left_wall_pos).\
        rotate(axis=[1, 0, 0], angle = 90).\
        scale(wall_dims)
    right_wall_pos = [0, -0.5*dimensions[1], 0.5*dimensions[2]]
    right_wall_t = T.translate(right_wall_pos).\
        rotate(axis=[1, 0, 0], angle = -90).\
        scale(wall_dims)

    back_left_dims = [dimensions[2], 0.5*(0.5*dimensions[1] - door[0]), 1.0]
    back_left_hlf = 0.5*(0.5*dimensions[1] + door[0])
    back_left_t = T.translate([+0.5*dimensions[0],
                               back_left_hlf,
                               0.5*dimensions[2]]).\
                  rotate(axis=[0, 1, 0], angle = -90).\
                  scale(back_left_dims)

    back_right_dims = [dimensions[2], 0.5*(0.5*dimensions[1] - door[1]), 1.]
    back_right_hlf = -0.5*(0.5*dimensions[1] + door[1])
    back_right_t = T.translate([+0.5*dimensions[0],
                                back_right_hlf,
                                0.5*dimensions[2]]).\
                  rotate(axis=[0, 1, 0], angle = -90).\
                  scale(back_right_dims)

    d = {
        "type": "scene",
        "integrator": {
            'type': 'prbvolpath',
            'max_depth': 8,
        },
        # REVIEW: alternatives
        # 'sampler': {'type': 'independent',
        #             'sample_count': 64},
        'sampler': {'type': 'stratified',
                    'sample_count': 4},
        "sensor": {
            "type": "perspective",
            "near_clip": 0.01,
            "far_clip": 50.0,
            'fov': 61.9,
            'fov_axis': 'x',
            "to_world": T.look_at(origin=camera_pos,
                                  target=[0, 0, 2.5],
                                  up=[0, 0, 1]),
            "film": {
                "type": "hdrfilm",
                "width": res[1],
                "height": res[0],
                'pixel_format' : 'rgb',
                'component_format': 'float32',
                'rfilter': {'type': 'gaussian'},
            },
        },
        "light": {
            'type': 'spot',
            'to_world': T.look_at(
                origin=[0, 0, 0.9*dimensions[2]],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'cutoff_angle' : 120.,
            'intensity': {
                'type': 'spectrum',
                'value': 100.0,
            }

        },
        # Color used for room surfaces
        "gray" : {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.3, 0.3, 0.3]
            }
        },
        # scene geometry
        'floor' : {
            'type': 'rectangle',
            'to_world' : floor_t,
            'material': {
                'type': 'ref',
                'id'  : 'gray'
                }
        },
        'ceiling' : {
            'type': 'rectangle',
            'flip_normals' : True,
            'to_world' : ceiling_t,
            'material': {
                'type': 'ref',
                'id'  : 'gray'
                }
        },
        'left_wall' : {
            'type': 'rectangle',
            'to_world' : left_wall_t,
            'material': {
                'type': 'ref',
                'id'  : 'gray'
                }
        },
        'right_wall' : {
            'type': 'rectangle',
            'to_world' : right_wall_t,
            'material': {
                'type': 'ref',
                'id'  : 'gray'
                }
        },
        'back_left' : {
            'type': 'rectangle',
            'to_world' : back_left_t,
            'material': {
                'type': 'ref',
                'id'  : 'gray'
                }
        },
        'back_right' : {
            'type': 'rectangle',
            'to_world' : back_right_t,
            'material': {
                'type': 'ref',
                'id'  : 'gray'
                }
        },
    }
    return d


def spec_alpha_bsdf(rgb, alpha:float) -> dict:
    d = {'type': 'mask',
         'material': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': rgb
            }
         },
         'opacity': {
             'type': 'uniform',
             'value': alpha
         }
         }
    return d


def create_cube(pos, dims, alpha) -> dict:
    t = T.translate(pos).\
        scale(dims)
    d = {'type': 'cube',
         'to_world': t,
         'bsdf': spec_alpha_bsdf([0.2, 0.25, 0.7],
                                 alpha)
         }
    return d

def create_volume(dims):
    """
        create_volume(dims)

    Creates a `mi.VolumeGrid` object specification.
    The volume is located inside of a transparent cube.
    The volume's coordinates need to be aligned with its parent.
    """
    vdims = [1, dims[0], dims[1]]
    data = np.zeros((*vdims, 1))
    # data = np.full((*vdims, 1), 0.5)
    # # # top left quad
    # data[0, :16, :16] = 0.0
    t = T.translate([-0.5*dims[0],0.5*dims[1],0.0]).\
        rotate(axis=[0,0,1],angle=-90).\
        scale([dims[0], dims[1], 1.5])
    # Modify the scene dictionary
    d = {
        'type': 'cube',
        'interior': {
            'type': 'heterogeneous',
            'sigma_t': {
                'type': 'gridvolume',
                'grid': mi.VolumeGrid(data),
                'filter_type' : 'nearest',
                'to_world': t
            },
            'scale': 1.0,
            'albedo': {
                'type': 'rgb',
                'value': [0.2, 0.2, 0.7]
            },
            # 'sample_emitters' : False,
        },
        'to_world' : T.translate([0,0,0.75]).\
        scale([dims[0], dims[1], 1.5]),
        'bsdf': {'type': 'null'}
    }
    return d
