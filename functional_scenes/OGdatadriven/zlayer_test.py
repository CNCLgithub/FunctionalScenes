import torch
from model import BetaVAE_H,BetaVAE_OG
from utils import cuda
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from torch.autograd import Variable
from dataset import CustomImageFolder
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

# checkpoints file path
ckpt_dir = 'checkpoints/main'
ckpt_name = 'last'
ckpt_name_og = 'og'
file_path = os.path.join(ckpt_dir, ckpt_name)
file_path_og = os.path.join(ckpt_dir, ckpt_name_og)

# neural network details
use_cuda = torch.cuda.is_available()
z_dim = 20
nc = 3

# load checkpoints
checkpoint = torch.load(file_path)
checkpoint_og = torch.load(file_path_og)

nn_weights = checkpoint_og['model_states']['net']

# intialization of neural network
def init_dd_state(nn_weights,use_cuda):
   net = cuda(BetaVAE_H(z_dim, nc), use_cuda)
   OG_model = cuda(BetaVAE_OG(net.encoder,z_dim, nc),use_cuda)
   OG_model.decoder.load_state_dict(nn_weights)
   return OG_model

# "nn" is OG_model 
def dd_state(nn,img):
   img_tensor = torch.from_numpy(img)
   img_tensor = Variable(cuda(img_tensor, use_cuda))
   z, mu, logvar = nn(img_tensor)
   z = z.cpu().detach().numpy()
   return z

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),])
    dset = CustomImageFolder('output/datasets/test_occupancy_grid_data_driven_twodoors/', transform)
    loader = DataLoader(dset,
                       batch_size=1,
                       shuffle=True,
                       num_workers=1,
                       pin_memory=False,
                       drop_last=True)

    nn = init_dd_state(nn_weights,use_cuda)
    for x, path in loader:
        img = np.array(x) #img is np.array with dimension (1, 3, 64, 64)
        z_recon = dd_state(nn,img) #z_recon is np.array with dimension (1, 1, 40, 22)
        #print(z_recon.shape)
    print("Everything passed")
