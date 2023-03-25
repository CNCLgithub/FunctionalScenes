#!/usr/bin/env python3

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# def make_animation(data, title:str, colorscale):
#     frames = []
#     for k in range(nb_frames):
#         d = go.Heatmap(z = vol[k],
#                        colorscale = colorscale,
#                        )
#         f = go.Frame(data = d,
#                      name = str(k))
#         frames.append(f)

#     fig = go.Figure(frames = frames)

#     def frame_args(duration):
#         return {
#                 "frame": {"duration": duration},
#                 "mode": "immediate",
#                 "fromcurrent": True,
#                 "transition": {"duration": duration, "easing": "linear"},
#             }

#     sliders = [
#                 {
#                     "pad": {"b": 10, "t": 60},
#                     "len": 0.9,
#                     "x": 0.1,
#                     "y": 0,
#                     "steps": [
#                         {
#                             "args": [[f.name], frame_args(0)],
#                             "label": str(k),
#                             "method": "animate",
#                         }
#                         for k, f in enumerate(fig.frames)
#                     ],
#                 }
#             ]
#     # Layout
#     fig.update_layout(
#             title=title,
#             width=600,
#             height=600,
#             scene=dict(
#                         zaxis=dict(range=[-0.1, 6.8], autorange=False),
#                         aspectratio=dict(x=1, y=1, z=1),
#                         ),
#             updatemenus = [
#                 {
#                     "buttons": [
#                         {
#                             "args": [None, frame_args(50)],
#                             "label": "&#9654;", # play symbol
#                             "method": "animate",
#                         },
#                         {
#                             "args": [[None], frame_args(0)],
#                             "label": "&#9724;", # pause symbol
#                             "method": "animate",
#                         },
#                     ],
#                     "direction": "left",
#                     "pad": {"r": 10, "t": 70},
#                     "type": "buttons",
#                     "x": 0.1,
#                     "y": 0,
#                 }
#             ],
#             sliders=sliders
#     )

# # Add data to be displayed before animation starts
# fig.add_trace(go.Surface(
#     z=6.7 * np.ones((r, c)),
#     surfacecolor=np.flipud(volume[67]),
#     colorscale='Gray',
#     cmin=0, cmax=200,
#     colorbar=dict(thickness=20, ticklen=4)
#     ))





EXPNAME = 'ccn_2023_exp'
burn_in = 10

def main():
    n = 30
    titles = []
    for i in range(n):
        titles.append(f'{i+1}_1')
        titles.append(f'{i+1}_2')
    fig = make_subplots(rows=n, cols=2,
                        shared_xaxes=True,
                        shared_yaxes=True,
                        subplot_titles = titles)

    for i in range(n):
        scene = f'{i+1}_1'
        path = f'/spaths/experiments/{EXPNAME}/{scene}'
        agg_path = f'{path}/aggregated.npz'
        agg = np.load(agg_path)
        geo = np.mean(agg['geo'][:, burn_in:], axis = (0,1))
        d1 =  go.Heatmap(z = geo.T,
                         colorscale = "blues",
                         )

        scene = f'{i+1}_2'
        path = f'/spaths/experiments/{EXPNAME}/{scene}'
        agg_path = f'{path}/aggregated.npz'
        agg = np.load(agg_path)
        geo = np.mean(agg['geo'][:, burn_in:], axis = (0,1))
        d2 =  go.Heatmap(z = geo.T,
                         colorscale = "blues",
                         )

        fig.add_trace(d1, row = i+1, col = 1)
        fig.add_trace(d2, row = i+1, col = 2)

    fig.update_layout(
        height = 325 * n,
        width = 600,
        # aspectratio = {'x' : 1, 'y' : 1},
    )
    fig.write_html(f'/spaths/experiments/{EXPNAME}_geo.html')

    fig = make_subplots(rows=n, cols=2,
                        shared_xaxes=True,
                        shared_yaxes=True,
                        subplot_titles = titles)


    for i in range(n):
        scene = f'{i+1}_1'
        path = f'/spaths/experiments/{EXPNAME}/{scene}'
        agg_path = f'{path}/aggregated.npz'
        agg = np.load(agg_path)
        att = np.mean(agg['att'][:, burn_in:], axis = (0,1))
        d1 =  go.Heatmap(z = att.T,
                         colorscale = "reds",
                         )

        scene = f'{i+1}_2'
        path = f'/spaths/experiments/{EXPNAME}/{scene}'
        agg_path = f'{path}/aggregated.npz'
        agg = np.load(agg_path)
        att = np.mean(agg['att'][:, burn_in:], axis = (0,1))
        d2 =  go.Heatmap(z = att.T,
                         colorscale = "reds",
                         )

        fig.add_trace(d1, row = i+1, col = 1)
        fig.add_trace(d2, row = i+1, col = 2)

    fig.update_layout(
        height = 325 * n,
        width = 600,
        # aspectratio = {'x' : 1, 'y' : 1},
    )
    fig.write_html(f'/spaths/experiments/{EXPNAME}_att.html')
    # pmat = np.mean(agg['pmat'][:, burn_in:], axis = (0,1))
    # pmat[pmat > 100] = 0
    # fig = go.Figure(data = go.Heatmap(z = pmat.T,
    #                                   colorscale = "hot",
    #                                   ))
    # fig.update_scenes(
    #     aspectratio = {'x' : 1, 'y' : 1},
    # )
    # fig.write_html(f'{path}/path.html')


    # att = np.mean(agg['att'][:, burn_in:], axis = (0,1))
    # fig = go.Figure(data = go.Heatmap(z = att.T,
    #                                   colorscale = "reds",
    #                                   ))
    # fig.update_scenes(
    #     aspectratio = {'x' : 1, 'y' : 1},
    # )
    # fig.write_html(f'{path}/att.html')


    # gran = np.mean(agg['gran'][:, burn_in:], axis = (0,1))
    # fig = go.Figure(data = go.Heatmap(z = gran.T,
    #                                   colorscale = "reds",
    #                                   ))
    # fig.update_scenes(
    #     aspectratio = {'x' : 1, 'y' : 1},
    # )
    # fig.write_html(f'{path}/gran.html')




if __name__ == '__main__':
    main()
