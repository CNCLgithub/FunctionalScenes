#!/usr/bin/env python3

import pathlib
import numpy as np
import plotly.express as px

EXPNAME = 'ccn_2023_exp'
burn_in = 10
steps = 200

def render_scene(path):
    agg_path = f'{path}/aggregated.npz'
    agg = np.load(agg_path)
    print(agg['geo'].shape)

    pathlib.Path(f'{path}/geo').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/att').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/pmat').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{path}/img_mu').mkdir(parents=True, exist_ok=True)

    for i in range(steps):
        fig = px.imshow(np.rot90(agg['geo'][0, i]), color_continuous_scale="blues")
        fig.update(layout_coloraxis_showscale=False)
        fig.write_image(f'{path}/geo/{i}.png')

        fig = px.imshow(np.rot90(agg['att'][0, i]), color_continuous_scale="reds")
        fig.update(layout_coloraxis_showscale=False)
        fig.write_image(f'{path}/att/{i}.png')

        fig = px.imshow(np.rot90(agg['pmat'][0, i]), color_continuous_scale="greens")
        fig.update(layout_coloraxis_showscale=False)
        fig.write_image(f'{path}/pmat/{i}.png')

        fig = px.imshow(agg['img'][0, i])
        fig.update(layout_coloraxis_showscale=False)
        fig.write_image(f'{path}/img_mu/{i}.png')


def main():
    path = f'/spaths/experiments/{EXPNAME}'
    scene = f'{path}/7_1'
    render_scene(scene)
    scene = f'{path}/7_2'
    render_scene(scene)


if __name__ == '__main__':
    main()
