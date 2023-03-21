#!/usr/bin/env python3
import time
import numpy as np
import drjit as dr
import mitsuba as mi
from pprint import pprint
from functional_scenes.render.interface import (initialize_scene,
                                                create_cube)
def initialize_grid(d:dict, dims, bounds):
    deltas = bounds / dims
    hlf = bounds * 0.5
    print(f'{hlf}=')
    print(f'{dims}=')
    print(f'{bounds}=')
    xs = np.linspace(-hlf[0], +hlf[0], num=dims[0])
    ys = np.linspace(-hlf[1], +hlf[1], num=dims[1])

    dims = [deltas[0], deltas[1], 0.75]
    alpha = 0.1
    c = 0
    for x in xs:
        for y in ys:
            pos = [x, y, 1.]
            d[f'cube_{c}'] = create_cube(pos, dims, alpha)
            c += 1
    return None

def main():
    mi.set_variant('cuda_ad_rgb')
    dimensions = [32, 32, 5]
    door = [10, -8]
    # res = (256, 256)
    res = (120, 180)

    # img = mi.render(scene)
    d = initialize_scene(dimensions,
                         door,
                         res)
    initialize_grid(d,
                    np.array(dimensions,
                             dtype = int),
                    np.array(dimensions,
                             dtype= float))
    scene = mi.load_dict(d)
    params = mi.traverse(scene)
    print(params)
    start_time = time.time()
    # params['cube.bsdf.opacity.value'] = 0.1 * (i + 1)
    # params.update()
    result = mi.render(scene, spp=16)
    end_time = time.time()
    print(f'Duration: {end_time - start_time}')
    # dr.flush_kernel_cache()
    # dr.flush_malloc_cache()

    img = mi.Bitmap(result).convert(srgb_gamma=True)
    mi.util.write_bitmap(f'/spaths/tests/scene_template.png',
                            img)

if __name__ == '__main__':
    main()
