#!/usr/bin/env python3
import time
import numpy as np
import drjit as dr
import mitsuba as mi
from pprint import pprint
mi.set_variant('cuda_ad_rgb')
from mitsuba import ScalarTransform4f as T
from functional_scenes.render.interface import (initialize_scene,
                                                create_cube)

def volume(dims):
    vdims = [1, dims[0], dims[1]]
    print(f'{vdims=}')
    data = np.zeros((*vdims, 1))
    data[0, :16, :16] = np.reshape(np.random.rand(16*16) > 0.5, (16,16,1))
    # data = np.full((*vdims, 1), 1.0)
    # # top left quad
    # data[0, :16, :16] = 0.0
    t = T.translate([-0.5*dims[0],0.5*dims[1],0]).\
        rotate(axis=[0,0,1],angle=-90).\
        scale([dims[0], dims[1], 1])
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
        'to_world' : T.scale([dims[0], dims[1], 1]),
        'bsdf': {'type': 'null'}
    }
    return d

def main():
    dimensions = [32, 32, 5]
    door = [10, -8]
    res = (480, 720)
    # res = (120, 180)
    d = initialize_scene(dimensions,
                         door,
                         res)
    d['object'] = volume(dimensions)
    scene = mi.load_dict(d)
    key = 'object.interior_medium.sigma_t.data'
    params = mi.traverse(scene)
    # params[key] = np.random.rand(*dimensions, 1)
    # params.update()
    # print(params)
    start_time = time.time()
    result = mi.render(scene, spp=16)
    end_time = time.time()
    print(f'Duration: {end_time - start_time}')
    img = mi.Bitmap(result).convert(srgb_gamma=True)
    mi.util.write_bitmap('/spaths/tests/mitsuba_volume.png',
                            img)



if __name__ == '__main__':
    main()
