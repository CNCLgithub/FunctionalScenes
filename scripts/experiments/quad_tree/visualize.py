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
burn_in = 1
scale = 4

def main():
    scenes = [7, 8, 10, 22, 27, 30]
    titles = ['geo', 'att', 'pmat']


    df_path = f"/spaths/datasets/{EXPNAME}/scenes.csv"
    df = pd.read_csv(df_path)
    # df = df.loc[map(lambda x: x in scenes, df['id'])]

    row_count = 1
    fig = make_subplots(rows=len(df), cols=len(titles),
                        # shared_xaxes=True,
                        # shared_yaxes=True,
                        subplot_titles = titles)

    for (ri, r) in df.iterrows():
        fig.update_yaxes(title_text=f"{r.id}, {r.door}", row=row_count, col=1)

        data_path = f'/spaths/experiments/{EXPNAME}/{r.id}_{r.door}_aggregated.npz'
        data_1 = np.load(data_path)

        data_path = f'/spaths/experiments/{EXPNAME}/{r.id}_{r.door}_furniture_{r.move}_aggregated.npz'
        data_2 = np.load(data_path)

        geo_a = np.mean(data_1['geo'][:, burn_in:], axis = (0,1))
        geo_a = downsample(geo_a, n = scale)


        geo_b = np.mean(data_2['geo'][:, burn_in:], axis = (0,1))
        geo_b = downsample(geo_b, n = scale)

        geo_a_init = np.mean(data_1['geo'][:, :1], axis = (0,1))
        geo_a_init = downsample(geo_a_init, n = scale)

        geo_b_init = np.mean(data_2['geo'][:, :1], axis = (0,1))
        geo_b_init = downsample(geo_b_init, n = scale)


        geo_diff = (geo_a - geo_b) - (geo_a_init - geo_b_init)
        geo_hm =  go.Heatmap(z = geo_diff.T, coloraxis="coloraxis1")
        fig.add_trace(geo_hm, row = row_count, col = 1)

        att_a = np.mean(data_1['att'][:, burn_in:], axis = (0,1))
        att_a = downsample(att_a, n = scale)
        att_b = np.mean(data_2['att'][:, burn_in:], axis = (0,1))
        att_b = downsample(att_b, n = scale)
        att_diff = att_a - att_b
        att_hm =  go.Heatmap(z = att_diff.T, coloraxis="coloraxis2")
        fig.add_trace(att_hm, row = row_count, col = 2)


        pmat_a = np.mean(data_1['pmat'][:, burn_in:], axis = (0,1))
        pmat_a = downsample(pmat_a, n = scale)
        pmat_b = np.mean(data_2['pmat'][:, burn_in:], axis = (0,1))
        pmat_b = downsample(pmat_b, n = scale)
        pmat_diff = pmat_a - pmat_b
        pmat_hm =  go.Heatmap(z = pmat_diff.T, coloraxis="coloraxis3")
        fig.add_trace(pmat_hm, row = row_count, col = 3)

        row_count += 1

    fig.update_layout(
        height = 325 * len(df),
        width = 900,
        coloraxis1=dict(colorscale='picnic'),
        coloraxis2=dict(colorscale='picnic'),
        coloraxis3=dict(colorscale='picnic'),
        showlegend=False
        # aspectratio = {'x' : 1, 'y' : 1},
    )
    fig.write_html(f'/spaths/experiments/{EXPNAME}_diff.html')

if __name__ == '__main__':
    main()
