#!/usr/bin/env python

import os
import argparse
import torch
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
    og_kwargs = dict()
    writer_kwargs = dict(
        num_workers = args.num_workers
    )
    path = dpath + '.beton'
    write_ffcv_data(d, path, img_kwargs, og_kwargs, writer_kwargs)

if __name__ == '__main__':
    main()
