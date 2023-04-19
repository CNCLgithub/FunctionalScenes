#!/usr/bin/env python3

import pathlib
import numpy as np
import pandas as pd
import plotly.express as px

EXPNAME = 'ccn_2023_exp'
steps = 150
scene = 19

def render_scene(path):
    agg_path = f'{path}_aggregated.npz'
    agg = np.load(agg_path)

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
    df_path = f"/spaths/datasets/{EXPNAME}/scenes.csv"
    df = pd.read_csv(df_path)
    df = df.loc[map(lambda x: x == scene, df['id'])]
    path = f'/spaths/experiments/{EXPNAME}'
    for (ri, r) in df.iterrows():
        path = f'{path}/{r.id}_{r.door}'
        render_scene(path)
        path = f'/spaths/experiments/{EXPNAME}/{r.id}_{r.door}_furniture_{r.move}_aggregated.npz'
        render_scene(path)


if __name__ == '__main__':
    main()
