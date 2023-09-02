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


EXPNAME = 'pathcost_3.0'
scale = 1

def main():

    df_path = f"/spaths/datasets/{EXPNAME}/scenes.csv"
    df = pd.read_csv(df_path)
    # df = df.loc[map(lambda x: x in scenes, df['id'])]

    row_count = 1
    fig = make_subplots(rows=60, cols=2,
                        shared_xaxes=True,
                        shared_yaxes=True)

    path_file = '1.64_0.01_7'
    paths = np.load(f'/spaths/datasets/{EXPNAME}_path/{path_file}_noisy_paths.npy')

    for scene in range(30):
        for door in range(2):
            fig.update_yaxes(title_text=f"{scene+1}, {door+1}",
                             row=row_count, col=1)
            pmat_a = downsample(paths[scene, door, 0], n = scale)
            pmat_hm =  go.Heatmap(z = pmat_a.T, coloraxis="coloraxis3")
            fig.add_trace(pmat_hm, row = row_count, col = 1)

            pmat_b = downsample(paths[scene, door, 1], n = scale)
            pmat_hm =  go.Heatmap(z = pmat_b.T, coloraxis="coloraxis3")
            fig.add_trace(pmat_hm, row = row_count, col = 2)
            row_count += 1

    fig.update_layout(
        height = 300 * 30 * 2,
        width = 800,
        coloraxis3=dict(colorscale='greens'),
        showlegend=False
    )
    fig.write_html(f'/spaths/datasets/{EXPNAME}_path/{path_file}_noisy_paths.html')

if __name__ == '__main__':
    main()
