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
                        default = 'pathcost')
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
        base = f'{r.scene}_{r.door}.png'
        aa_movies.append([base, base, r.flipx])
        # then proceed to make `a -> b` trials
        move = f'{r.scene}_{r.door}_shifted.png'
        ab_movies.append([base, move, r.flipx])

    # repeate aa trials to have a 50/50 split
    naa = len(aa_movies)
    nab = len(ab_movies)

    trials = [aa_movies + ab_movies]
    with open(os.path.join(dataset, 'condlist.json'), 'w') as f:
       json.dump(trials, f)


if __name__ == '__main__':
    main()
