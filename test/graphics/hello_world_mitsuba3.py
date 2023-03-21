#!/usr/bin/env python3
import time
import mitsuba as mi

variants = ['cuda_rgb']
for v in variants:
    mi.set_variant(v)
    start_time = time.time()
    d = mi.cornell_box()
    d['integrator'] = {'type': 'direct',
                       'shading_samples': 4}

    # d['integrator'] = {'type': 'volpath',
    #                    'max_depth': 2}

    d['small-box']['bsdf'] = {
        'type': 'mask',
        # Base material: a two-sided textured diffuse BSDF
        'material': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.2, 0.25, 0.7]
            }
        },
        'opacity': {
            'type': 'uniform',
            'value': 0.5
        }
    }
    img = mi.render(mi.load_dict(d))
    end_time = time.time()
    print(f'Variant: {v}')
    print(f'\tDuration: {end_time-start_time}')
    mi.util.write_bitmap(f'/spaths/datasets/{v}-cbox.png', img)
# mi.util.write_bitmap("my_first_render.exr", image)
# img.write('/spaths/datasets/cbox.png')
# mi.Bitmap(img).write('/spaths/datasets/cbox.png')
{'type': 'scene',
 'integrator': {'type': 'path', 'max_depth': 8},
 'sensor': {'type': 'perspective', 'fov_axis': 'smaller', 'near_clip': 0.001, 'far_clip': 100.0,
            'focus_distance': 1000, 'fov': 39.3077, 'to_world': [[-1, 0, 0, 0],
                                                                 [0, 1, 0, 0],
                                                                 [0, 0, -1, 3.9],
                                                                 [0, 0, 0, 1]],
            'sampler': {'type': 'independent', 'sample_count': 64},
            'film': {'type': 'hdrfilm', 'width': 256, 'height': 256, 'rfilter': {'type': 'gaussian'},
                     'pixel_format': 'rgb', 'component_format': 'float32'}},
 'white': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.885809, 0.698859, 0.666422]}},
 'green': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.105421, 0.37798, 0.076425]}},
 'red': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.570068, 0.0430135,0.0443706]}},
 'light': {'type': 'rectangle', 'to_world': [[0.23, 0, 0, 0],
                                             [0, -8.30516e-09, -0.19, 0.99],
                                             [0, 0.19, -8.30516e-09, 0.01],
                                             [0, 0, 0, 1]],
           'bsdf': {'type': 'ref', 'id': 'white'},
           'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': [18.387, 13.9873, 6.75357]}}},
 'floor': {'type': 'rectangle', 'to_world': [[1, 0, 0, 0],
                                             [0, -4.37114e-08, 1, -1],
                                             [0, -1, -4.37114e-08, 0],
                                             [0, 0, 0, 1]],
           'bsdf': {'type': 'ref', 'id': 'white'}}, 'ceiling': {'type': 'rectangle', 'to_world': [[1, 0, 0, 0],
 [0, -4.37114e-08, -1, 1],
 [0, 1, -4.37114e-08, 0],
 [0, 0, 0, 1]], 'bsdf': {'type': 'ref', 'id': 'white'}}, 'back': {'type': 'rectangle', 'to_world': [[1, 0, 0, 0],
 [0, 1, 0, 0],
 [0, 0, 1, -1],
 [0, 0, 0, 1]], 'bsdf': {'type': 'ref', 'id': 'white'}}, 'green-wall': {'type': 'rectangle', 'to_world': [[-4.37114e-08, 0, -1, 1],
 [0, 1, 0, 0],
 [1, 0, -4.37114e-08, 0],
 [0, 0, 0, 1]], 'bsdf': {'type': 'ref', 'id': 'green'}}, 'red-wall': {'type': 'rectangle', 'to_world': [[-4.37114e-08, 0, 1, -1],
 [0, 1, 0, 0],
 [-1, 0, -4.37114e-08, 0],
 [0, 0, 0, 1]], 'bsdf': {'type': 'ref', 'id': 'red'}}, 'small-box': {'type': 'cube', 'to_world': [[0.286891, 0, -0.0877115, 0.335],
 [0, 0.3, 0, -0.7],
 [0.0877115, 0, 0.286891, 0.38],
 [0, 0, 0, 1]], 'bsdf': {'type': 'ref', 'id': 'white'}}, 'large-box': {'type': 'cube', 'to_world': [[0.28491, 0, 0.0939491, -0.33],
 [0, 0.61, 0, -0.4],
 [-0.0939491, 0, 0.28491, -0.28],
 [0, 0, 0, 1]], 'bsdf': {'type': 'ref', 'id': 'white'}}}
