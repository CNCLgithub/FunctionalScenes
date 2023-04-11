#!/usr/bin/env python3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def downsample(a, n:int = 2):
    if n ==1 :
        return a
    b = a.shape[0]//n
    a_downsampled = a.reshape(-1, n, b, n).sum((-1, -3)) / (n*n)
    return a_downsampled


EXPNAME = 'ccn_2023_exp'
steps = 150
burn_in = 1
scale = 8

def main():
    scenes = [7, 8, 10, 22, 27, 30]
    titles = ['geo', 'att', 'pmat']


    df_path = f"/spaths/datasets/{EXPNAME}/scenes.csv"
    df = pd.read_csv(df_path)
    # df = df.loc[map(lambda x: x in scenes, df['id'])]

    row_count = 1
    fig = make_subplots(rows=2*len(df), cols=len(titles),
                        shared_xaxes=True,
                        shared_yaxes=True,
                        subplot_titles = titles)

    for (ri, r) in df.iterrows():
        fig.update_yaxes(title_text=f"{r.id}, {r.door}", row=row_count, col=1)

        data_path = f'/spaths/experiments/{EXPNAME}/{r.id}_{r.door}_aggregated.npz'
        data_1 = np.load(data_path)

        data_path = f'/spaths/experiments/{EXPNAME}/{r.id}_{r.door}_furniture_{r.move}_aggregated.npz'
        data_2 = np.load(data_path)

        geo_a = np.mean(data_1['geo'][:, burn_in:steps], axis = (0,1))
        geo_a = downsample(geo_a, n = scale)
        geo_hm =  go.Heatmap(z = geo_a.T, coloraxis="coloraxis1")
        fig.add_trace(geo_hm, row = row_count, col = 1)

        att_a = np.mean(data_1['att'][:, burn_in:steps], axis = (0,1))
        att_a = downsample(att_a, n = scale)
        att_hm =  go.Heatmap(z = att_a.T, coloraxis="coloraxis2")
        fig.add_trace(att_hm, row = row_count, col = 2)

        pmat_a = np.mean(data_1['pmat'][:, burn_in:steps], axis = (0,1))
        pmat_a = downsample(pmat_a, n = scale)
        pmat_hm =  go.Heatmap(z = pmat_a.T, coloraxis="coloraxis3")
        fig.add_trace(pmat_hm, row = row_count, col = 3)

        row_count += 1

        geo_b = np.mean(data_2['geo'][:, burn_in:steps], axis = (0,1))
        geo_b = downsample(geo_b, n = scale)
        geo_hm =  go.Heatmap(z = geo_b.T, coloraxis="coloraxis1")
        fig.add_trace(geo_hm, row = row_count, col = 1)

        att_b = np.mean(data_2['att'][:, burn_in:steps], axis = (0,1))
        att_b = downsample(att_b, n = scale)
        att_hm =  go.Heatmap(z = att_b.T, coloraxis="coloraxis2")
        fig.add_trace(att_hm, row = row_count, col = 2)

        pmat_b = np.mean(data_2['pmat'][:, burn_in:steps], axis = (0,1))
        pmat_b = downsample(pmat_b, n = scale)
        pmat_hm =  go.Heatmap(z = pmat_b.T, coloraxis="coloraxis3")
        fig.add_trace(pmat_hm, row = row_count, col = 3)

        row_count += 1

    fig.update_layout(
        height = 300 * len(df) * 2,
        width = 800,
        coloraxis1=dict(colorscale='blues'),
        coloraxis2=dict(colorscale='reds'),
        coloraxis3=dict(colorscale='greens'),
        showlegend=False
        # aspectratio = {'x' : 1, 'y' : 1},
    )
    fig.write_html(f'/spaths/experiments/{EXPNAME}_each_trial.html')

if __name__ == '__main__':
    main()
