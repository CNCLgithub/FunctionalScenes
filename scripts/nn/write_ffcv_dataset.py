#!/usr/bin/env python

import os
import argparse
import numpy as np

from functional_scenes.og_proposal.dataset import OGVAEDataset, write_ffcv_data

def main():
    parser = argparse.ArgumentParser(
        description = 'Converts dataset to .beton format for FFCV',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('src', type = str,
                        help = 'Which scene dataset to use')
    parser.add_argument('--render', type = str,
                        help = 'Which render mode', choices = ['pytorch'],
                        default = 'pytorch')
    parser.add_argument('--num_workers', type = int,
                        help = 'Number of write workers',
                        default = 4)

    args = parser.parse_args()
    dpath = os.path.join('/spaths/datasets', args.src)
    d = OGVAEDataset(dpath)

    img_kwargs = dict()
    og_kwargs = dict(
        shape = d.manifest['og_shape'],
        dtype = np.dtype('uint8')
    )
    writer_kwargs = dict(
        num_workers = args.num_workers
    )
    path = dpath + '.beton'

    print(og_kwargs)
    write_ffcv_data(d, path, img_kwargs, og_kwargs, writer_kwargs)


# TODO: compute img stats
# from: https://www.binarystudy.com/2021/04/how-to-calculate-mean-standard-deviation-images-pytorch.html
#
# batch_size = 2

# loader = DataLoader(
#   image_data,
#   batch_size = batch_size,
#   num_workers=1)

# def batch_mean_and_sd(loader):

#     cnt = 0
#     fst_moment = torch.empty(3)
#     snd_moment = torch.empty(3)

#     for images, _ in loader:
#         b, c, h, w = images.shape
#         nb_pixels = b * h * w
#         sum_ = torch.sum(images, dim=[0, 2, 3])
#         sum_of_square = torch.sum(images ** 2,
#                                   dim=[0, 2, 3])
#         fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
#         snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
#         cnt += nb_pixels

#     mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
#     return mean,std

# mean, std = batch_mean_and_sd(loader)

if __name__ == '__main__':
    main()
