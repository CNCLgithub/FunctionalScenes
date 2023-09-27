#!/usr/bin/env python3
import time
import numpy as np
import drjit as dr
import mitsuba as mi
from pprint import pprint
mi.set_variant('cuda_ad_rgb')
from mitsuba import ScalarTransform4f as T
from functional_scenes.render.interface import (initialize_scene,
                                                create_volume)

def main():
    mi.set_variant('cuda_ad_rgb')
    dimensions = [16, 16, 5]
    door = [10, -8]
    res = (128, 128)
    # res = (120, 180)
    d = initialize_scene(dimensions,
                         door,
                         res)
    d['object'] = create_volume(dimensions)
    scene = mi.load_dict(d)
    key = 'object.interior_medium.sigma_t.data'
    params = mi.traverse(scene)
    params[key] = np.random.rand(1, 16, 16, 1)
    params.update()
    # print(params)
    start_time = time.time()
    result = mi.render(scene, spp=24)
    end_time = time.time()
    print(f'Duration: {end_time - start_time}')
    img = mi.Bitmap(result).convert(srgb_gamma=True)
    mi.util.write_bitmap('/spaths/tests/mitsuba_volume.png',
                         img)



if __name__ == '__main__':
    main()
