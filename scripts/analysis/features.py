#!/usr/bin/env python3

import os
import json
import torch
import argparse
import numpy as np
from PIL import Image
from pandas import read_csv, DataFrame
from functional_scenes import (init_alexnet, init_alexnet_objects,
                               compare_features)
from functional_scenes.render import render_scene, SimpleGraphics

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(device)

features = {
    'features.0' : 'c1',
    'features.1' : 'r1',
    'features.3' : 'c2',
    'features.4' : 'r2',
    'features.6' : 'c3',
    'features.7' : 'r3',
    'features.8' : 'c4',
    'features.10'  : 'c5',
    'classifier.1' : 'fc1',
    'classifier.4' : 'fc2'
}

# places
model = init_alexnet('/datasets/alexnet_places365.pth.tar')
# objects
# model = init_alexnet_objects('pytorch/vision:v0.6.0')


src = '/renders/2e_1p_30s/1/scene.json'
with open(src, 'r') as f:
    template = json.load(f)

graphics = SimpleGraphics((480, 720), device)
# graphics = SimpleGraphics((240, 360), device)
graphics.set_from_scene(template)

def render_p3d(src):

    with open(src, 'r') as f:
        scene = json.load(f)
    img, _ = render_scene(scene, graphics)
    return Image.fromarray((img * 255).astype(np.uint8),
                           mode = 'RGB')


def feature_corr(r, renders):

    base = os.path.join(renders, '{0:d}.png'.format(r.id))
    img = os.path.join(renders, '{0:d}_{1!s}_{2!s}.png'.format(r.id,
                                                                r.furniture,
                                                                r.move))
    a = Image.open(base).convert('RGB')
    b = Image.open(img).convert('RGB')
    return compare_features(model, features, a, b)

def foo(src, exp: str):
    base = '{0:d}'.format(src.id)
    name = '{0:d}_{1!s}_{2!s}'.format(src.id, src.furniture,
                                      src.move)

    base_scene = '/renders/{0!s}/{1!s}/scene.json'.format(exp, base)
    row_scene = '/renders/{0!s}/{1!s}/scene.json'.format(exp, name)
    base_pd = render_p3d(base_scene)
    row_pd = render_p3d(row_scene)
    feats = compare_features(model, features, base_pd, row_pd)
    feats = {(k+'_3d'):v for (k,v) in feats.items()}

    base_blend = '/renders/{0!s}/{1!s}.png'.format(exp, base)
    base_blend = Image.open(base_blend).convert('RGB')
    row_blend = '/renders/{0!s}/{1!s}.png'.format(exp, name)
    row_blend = Image.open(row_blend).convert('RGB')

    feats.update(compare_features(model, features, base_blend, row_blend))
    return feats


def main():
    parser = argparse.ArgumentParser(
        description = 'Generates nn features for exp',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--exp', type = str,
                        help = "Which scene dataset to use",
                        default = '2e_1p_30s')
    args = parser.parse_args()
    df = read_csv("/scenes/{0!s}.csv".format(args.exp))
    results = DataFrame(columns = ['scene', 'furniture', 'move',
                                   *features.values()])

    df = df.assign(**df.apply(foo,
                       args = (args.exp,),
                       axis = 1,
                       result_type = 'expand')).drop(['d'], axis =1)
    print(df)

    out = '/experiments/' + args.exp
    os.path.isdir(out) or os.mkdir(out)
    df.to_csv(os.path.join(out, 'features.csv'))

if __name__ == '__main__':
    main()
