#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import pandas as pd

def main():

    parser = argparse.ArgumentParser(
        description = 'Generates condlist for experiment',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type = str,
                        help = "Which scene dataset to use",
                        default = 'vss_pilot')
    parser.add_argument('--render', type = str,
                        help = "Which render mode", choices = ['cycles'],
                        default = 'cycles')
    args = parser.parse_args()

    dataset = '/spaths/datasets/' + args.dataset
    renders = '{0!s}/render_{1!s}'.format(dataset, args.render)

    df = pd.read_csv(os.path.join(dataset, 'scenes.csv'))

    aa_movies = []
    ab_movies = []
    # groupby move
    for (_, r) in df.iterrows():
        # first create each `a->a` trial
        base = '{0:d}_{1:d}.png'.format(r.id, r.door)

        aa_movies.append([base, base, r.flip])
        # then proceed to make `a -> b` trials
        move = '{0:d}_{1:d}_{2:d}_{3!s}.png'.format(r.id, r.door,
                                                    r.furniture,
                                                    r.move)
        ab_movies.append([base, move, r.flip])

    # repeate aa trials to have a 50/50 split
    naa = len(aa_movies)
    nab = len(ab_movies)

    trials = [aa_movies + ab_movies]
    with open(os.path.join(dataset, 'condlist.json'), 'w') as f:
       json.dump(trials, f)


if __name__ == '__main__':
    main()
