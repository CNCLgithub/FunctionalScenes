#!/usr/bin/env python3

import scipy
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

def metric(model_data, human_data):
    diff_cor = np.zeros(24)
    for scene in range(24):
        model_diff = downsample(model_data[scene, 0] - model_data[scene, 1], n=scale).flatten()
        human_diff = downsample(human_data[scene,0]-human_data[scene,1],n=int(scale/2)).flatten()
        diff_cor[scene] = scipy.stats.pearsonr(model_diff, human_diff).statistic**2
    print(diff_cor)
    return np.mean(diff_cor)

def ordinary_bootstrap(model_data, human_data):
    n =  human_data.shape[0]
    inds = np.random.choice(n, n)
    sample = human_data[sample].mean(dim = 0)
    return metric(model_data, sample)

def main():
    scenes = list(range(1, 25))

    titles = ['geo_model', 'geo_human']


    df_path = f"/spaths/datasets/{EXPNAME}/scenes.csv"
    df = pd.read_csv(df_path)
    df = df.loc[map(lambda x: x in scenes, df['id'])]

    row_count = 1
    fig = make_subplots(rows=2*len(df), cols=len(titles),
                        shared_xaxes=True,
                        shared_yaxes=True,
                        subplot_titles = titles)
    # human_data = np.load(f'/spaths/experiments/{EXPNAME}_human_recon_750ms.npy')
    human_data = np.load(f'/spaths/experiments/{EXPNAME}_human_recon_1500ms.npy')
    avg_human_data = np.mean(human_data, axis = 0)
  
    model_data = np.zeros((24, 2, 32, 32))

    for (ri, r) in df.iterrows():
        fig.update_yaxes(title_text=f"{r.id}, {r.door}", row=row_count, col=1)

        data_path = f'/spaths/experiments/{EXPNAME}/{r.id}_{r.door}_aggregated.npz'
        data_1 = np.load(data_path)

        geo_a = np.mean(data_1['geo'][:, burn_in:steps], axis = (0,1))
        # store model esitmate for analysis
        model_data[r.id-1, r.door-1] = geo_a
        geo_a = downsample(geo_a, n = scale)
        geo_a = np.array(geo_a.T)
        geo_hm =  go.Heatmap(z = geo_a, coloraxis="coloraxis1")
        fig.add_trace(geo_hm, row = row_count, col = 1)

        geo_b = avg_human_data[r.id-1, r.door-1]
        geo_b = downsample(geo_b[::-1], n = int(scale / 2))
        geo_hm =  go.Heatmap(z = geo_b, coloraxis="coloraxis1")
        fig.add_trace(geo_hm, row = row_count, col = 2)

        row_count += 1

    diff_cor = metric(model_data, avg_human_data)
    print(f'{diff_cor=}')

    fig.update_layout(
        height = 300 * len(df) * 2,
        width = 600,
        coloraxis1=dict(colorscale='blues'),
        showlegend=False
        # aspectratio = {'x' : 1, 'y' : 1},
    )
    fig.write_html(f'/spaths/experiments/{EXPNAME}_quad_tree_recon.html')

if __name__ == '__main__':
    main()
