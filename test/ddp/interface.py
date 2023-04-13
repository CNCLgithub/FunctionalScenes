import torch
import numpy as np
import functional_scenes as fs

def main():

    config = '/project/scripts/nn/configs/og_decoder.yaml'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # PyCall will pass Julia arrays as Numpy arrays
    x = np.random.rand(1, 3, 256, 256)
    nn = fs.init_dd_state(config, device)
    y = fs.dd_state(nn, x, device)


if __name__ == '__main__':
    main()
