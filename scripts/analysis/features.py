#!/usr/bin/env python3

import os
import argparse
from pandas import read_csv, DataFrame
from functional_scenes import init_alexnet, init_alexnet_objects, compare_features


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
# model = init_alexnet('/datasets/alexnet_places365.pth.tar')
# objects
model = init_alexnet_objects('pytorch/vision:v0.6.0')

def foo(r, renders):
    base = os.path.join(renders, '{0:d}.png'.format(r.id))
    img = os.path.join(renders, '{0:d}_{1!s}_{2!s}.png'.format(r.id,
                                                                r.furniture,
                                                                r.move))
    return compare_features(model, features, base, img)

def main():
    parser = argparse.ArgumentParser(
        description = 'Generates nn features for exp',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--exp', type = str,
                        help = "Which scene dataset to use",
                        default = '2e_1p_30s')
    args = parser.parse_args()
    renders = '/renders/' + args.exp
    df = read_csv("/scenes/{0!s}.csv".format(args.exp))
    results = DataFrame(columns = ['scene', 'furniture', 'move',
                                   *features.values()])

    df = df.assign(**df.apply(foo,
                       args = (renders,),
                       axis = 1,
                       result_type = 'expand')).drop(['d'], axis =1)
    print(df)

    df.to_csv(os.path.join('/experiments', args.exp, 'features.csv'))

if __name__ == '__main__':
    main()
