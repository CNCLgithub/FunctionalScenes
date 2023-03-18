import numpy as np
import drjit as dr
import mitsuba as mi


def initialize_scene(dimensions,
                     door,
                     res):

    camera_pos = [-0.55*dimensions[0], 0, 3.75]

    floor_dims = [dimensions[0], dimensions[1], 1]
    floor_t = mi.ScalarTransform4f.scale(floor_dims)
    ceiling_t = mi.ScalarTransform4f.translate([0, 0, dimensions[2]]).\
        scale(floor_dims)

    wall_dims = [dimensions[0], dimensions[2], 1]
    left_wall_pos = [0, 0.5*dimensions[1], 0.5*dimensions[2]]
    left_wall_t = mi.ScalarTransform4f.translate(left_wall_pos).\
        rotate(axis=[1, 0, 0], angle = 90).\
        scale(wall_dims)
    right_wall_pos = [0, -0.5*dimensions[1], 0.5*dimensions[2]]
    right_wall_t = mi.ScalarTransform4f.translate(right_wall_pos).\
        rotate(axis=[1, 0, 0], angle = -90).\
        scale(wall_dims)
    # print(f'{left_wall_pos=}')
    # print(f'{right_wall_pos=}')
    # print(f'{wall_dims=}')

    back_left_dims = [dimensions[2], 0.5*(0.5*dimensions[1] - door[0]), 1.0]
    back_left_hlf = 0.5*(0.5*dimensions[1] + door[0])
    # print(f'{back_left_dims=}')
    # print(f'{back_left_hlf=}')
    back_left_t = mi.ScalarTransform4f.translate([+0.5*dimensions[0],
                                                  back_left_hlf,
                                                  0.5*dimensions[2]]).\
                  rotate(axis=[0, 1, 0], angle = -90).\
                  scale(back_left_dims)

    back_right_dims = [dimensions[2], 0.5*(0.5*dimensions[1] - door[1]), 1.]
    back_right_hlf = -0.5*(0.5*dimensions[1] + door[1])
    # print(f"{back_right_dims=}")
    # print(f"{back_right_hlf=}")
    back_right_t = mi.ScalarTransform4f.translate([+0.5*dimensions[0],
                                                  back_right_hlf,
                                                  0.5*dimensions[2]]).\
                  rotate(axis=[0, 1, 0], angle = -90).\
                  scale(back_right_dims)

    d = {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            'max_depth': 4
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
            # 'focal_length': '25mm',
            'fov_axis': 'x',
            # REVIEW: necessary?
            # 'focus_distance': 1000,
            "to_world": mi.ScalarTransform4f.look_at(origin=camera_pos,
                                                     target=[0, 0, 3.5],
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
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 0, 0.9*dimensions[2]],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'cutoff_angle' : 120.,
            'intensity': {
                'type': 'spectrum',
                'value': 100.0,
            }


            # 'type': 'point',
            # 'position': [5.0, 0.0, 0.6*dimensions[2]],
            # 'intensity': {
            #     'type': 'uniform',
            #     'value': 100.0
            #     # 'type': 'spectrum',
            #     # 'value': 100.0,
            # }
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
    t = mi.ScalarTransform4f.translate(pos).scale(dims)
    d = {'type': 'cube',
         'to_world': t,
         'bsdf': spec_alpha_bsdf([0.2, 0.25, 0.7],
                                 alpha)
         }
    return d
