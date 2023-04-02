#!/usr/bin/env python3
import time
import mitsuba as mi

variants = ['llvm_ad_rgb']
for v in variants:
    mi.set_variant(v)
    start_time = time.time()
    d = mi.cornell_box()
    # d['integrator'] = {'type': 'direct',
    #                    'shading_samples': 4}

    d['integrator'] = {'type': 'prbvolpath',
                       'max_depth': 4}

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
