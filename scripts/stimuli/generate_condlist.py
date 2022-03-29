#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import pandas as pd
from functional_scenes import blank, chain, still, concat, vflip, run_cmd

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


def stimuli(a, b, fps, im_dur, mk_dur, out,
            flip = False):
    """Creates a video of the pattern `a -> mask -< b`
    """
    src = '' # blank takes an empty argument
    p1 = out + '_p1'
    p2 = out + '_p2'
    cmd = chain([still], [(im_dur,fps)], a, p1, 'a')
    cmd += chain([still], [(im_dur,fps)], b, p2, 'b')
    cmd += chain([blank, concat, concat, vflip],
                 [(mk_dur, fps), (p2+'.mp4', True), (p1+'.mp4', False),
                  (flip,)],
                 src, out, 'c')
    cmd.append('rm ' + p1 + '.mp4')
    cmd.append('rm ' + p2 + '.mp4')
    run_cmd(cmd)


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
    parser.add_argument('--fps', type = int,
                        help = "FPS of resulting videos",
                        default = 60)
    parser.add_argument('--stim_dur', type = float,
                        help = 'duration of A or B in seconds',
                        default = 0.750)
    parser.add_argument('--mask_dur', type = float,
                        help = 'duration of mask in seconds',
                        default = 0.750)
    args = parser.parse_args()

    dataset = '/spaths/datasets/' + args.dataset
    renders = dataset + '/render_cycles/' + args.scene
    renders = '{0!s}/render_{1!s}'.format(dataset, args.render)
    renders = '{0!s}/movies_{1!s}'.format(dataset, args.render)

    os.path.isdir(movies) or os.mkdir(movies)
    df = pd.read_csv(os.path.join(dataset, 'scenes.csv'))

    aa_movies = []
    ab_movies = []
    # groupby move
    for (_, r) in df.iterrows():
        # first create each `a->a` trial
        base = os.path.join(renders, '{0:d}_{1:d}.png'.format(r.id, r.door))
        base_suffix = '{0:d}_{0:d}_{1:d}'.format(r.id, r.door)
        base_out = os.path.join(movies, base_suffix)
        stimuli(base, base, args.fps, args.stim_dur, args.mask_dur, base_out,
                flip = r.flip)
        aa_movies.append(base_suffix + '.mp4')
        # then proceed to make `a -> b` trials
        move_suffix = '{0:d}_{1:d}_{2:d}_{3!s}'.format(r.id, r.door,
                                                  r.furniture,
                                                  r.move)
        move_src = os.path.join(renders, move_suffix + '.png')
        move_out = os.path.join(movies, move_suffix)
        stimuli(base, move_src, args.fps, args.stim_dur, args.mask_dur, move_out,
                flip = r.flip)
        ab_movies.append(move_suffix + '.mp4')

    # repeate aa trials to have a 50/50 split
    naa = len(aa_movies)
    nab = len(ab_movies)

    trials = [aa_movies + ab_movies]
    with open(os.path.join(movies, args.scene + '.json'), 'w') as f:
       json.dump(trials, f)


if __name__ == '__main__':
    main()
