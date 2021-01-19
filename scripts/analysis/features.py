#!/usr/bin/env python3

import os
import argparse
from pandas import read_csv, DataFrame
from functional_scenes import init_alexnet, compare_features


features = {
    'features.10'  : 'c5',
    'classifier.1' : 'fc1',
    'classifier.4' : 'fc2'
}
model = init_alexnet('/datasets/alexnet_places365.pth.tar')

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
    # for r in df.rows():
    #     results.append({
    #         'scene': r.id,
    #         'furniture': r.furniture,
    #         'move': r.move,
    #         **fs
    #     })
    print(df)

    df.to_csv(os.path.join('/experiments', args.exp, 'features.csv'))

if __name__ == '__main__':
    main()
