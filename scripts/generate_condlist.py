#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import pandas as pd
from functional_scenes import blank, chain, still, concat, run_cmd

noise = '/project/src/blender/noise.jpeg'

# def stimuli(a, b, fps, im_dur, mk_dur, out):
#     """Creates a video of the pattern `a -> mask -< b`
#     """
#     src = '' # blank takes an empty argument
#     p1 = out + '_p1'
#     p2 = out + '_p2'
#     cmd = chain([still], [(im_dur,fps)], a, p1, 'a')
#     cmd += chain([still], [(im_dur,fps)], b, p2, 'b')
#     cmd += chain([still, concat, concat],
#                  [(mk_dur, fps), (p1+'.mp4', True), (p2+'.mp4', False)],
#                  noise, out, 'c')
#     cmd.append('rm ' + p1 + '.mp4')
#     cmd.append('rm ' + p2 + '.mp4')
#     run_cmd(cmd)


def stimuli(a, b, fps, im_dur, mk_dur, out):
    """Creates a video of the pattern `a -> mask -< b`
    """
    src = '' # blank takes an empty argument
    p1 = out + '_p1'
    p2 = out + '_p2'
    cmd = chain([still], [(im_dur,fps)], a, p1, 'a')
    cmd += chain([still], [(im_dur,fps)], b, p2, 'b')
    cmd += chain([blank, concat, concat],
                 [(mk_dur, fps), (p2+'.mp4', True), (p1+'.mp4', False)],
                 src, out, 'c')
    cmd.append('rm ' + p1 + '.mp4')
    cmd.append('rm ' + p2 + '.mp4')
    run_cmd(cmd)


def main():

    parser = argparse.ArgumentParser(
        description = 'Generates condlist for 1exit',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--scene', type = str,
                        help = "Which scene dataset to use",
                        default = '2e_1p_30s')
    parser.add_argument('--fps', type = int,
                        help = "FPS of resulting videos",
                        default = 60)
    parser.add_argument('--stim_dur', type = float,
                        help = 'duration of A or B in seconds',
                        default = 0.750)
    parser.add_argument('--mask_dur', type = float,
                        help = 'duration of mask in seconds',
                        default = 0.250)
    args = parser.parse_args()

    renders = '/renders/' + args.scene
    movies = '/movies/' + args.scene
    os.path.isdir(movies) or os.mkdir(movies)
    df = pd.read_csv('/scenes/' + args.scene + '.csv')
    bases = np.unique(df.id)

    # first create each `a->a` trial
    aa_movies = []
    for i in bases:
        src = os.path.join(renders, '{0:d}.png'.format(i))
        suffix = '{0:d}_{0:d}'.format(i)
        out = os.path.join(movies, suffix)
        stimuli(src, src, args.fps, args.stim_dur, args.mask_dur, out)
        aa_movies.append(suffix + '.mp4')

    # then proceed to make `a -> b` trials
    ab_movies = []
    for _, row in df.iterrows():
        base = os.path.join(renders, '{0:d}.png'.format(row.id))
        suffix = '{0:d}_{1:d}_{2!s}'.format(row.id, row.furniture, row.move)
        src = os.path.join(renders, suffix + '.png')
        out = os.path.join(movies, suffix)
        stimuli(base, src, args.fps, args.stim_dur, args.mask_dur, out)
        ab_movies.append(suffix + '.mp4')

    # repeate aa trials to have a 50/50 split
    naa = len(aa_movies)
    nab = len(ab_movies)
    reps = int(nab / naa)
    aa_movies = np.repeat(aa_movies, reps).tolist()

    trials = [aa_movies + ab_movies]
    with open(os.path.join(movies, args.scene + '.json'), 'w') as f:
       json.dump(trials, f)


if __name__ == '__main__':
    main()
