#!/usr/bin/env python3
import numpy as np
from image_shuffler import Shuffler

def main():
    dpath = '/spaths/datasets/path_block_2024-01-12/render_cycles'
    for i in range(5):
        img_path = f'{dpath}/{i+1}_1.png'
        img = Shuffler(img_path)
        img.shuffle(matrix=(15, 10))
        img.save()

if __name__ == '__main__':
    main()
