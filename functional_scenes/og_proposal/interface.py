# native packages go first
import os

# external packages order by "generality"
import numpy as np

from PIL import Image

# try to sort related imports by length
import torch
from torchvision import transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

# place interal imports at the end
from . dataset import CustomImageFolder
from . model import BetaVAE_H, BetaVAE_OG

# neural network details
z_dim = 20
nc = 3

# intialization of neural network
def init_dd_state(enc_path:str, dec_path:str, device,
                  z_dim:int = 20, nc:int = 3):
   # first load encoder
   enc = BetaVAE_H(z_dim, nc).to(device)
   enc_weights = torch.load(enc_path)['model_states']['net']
   enc.encoder.load_state_dict(enc_weights)

   # then load decoder
   dec = BetaVAE_OG(enc.encoder,z_dim, nc).to(device)
   dec_weights = torch.load(dec_path)['model_states']['net']
   dec.decoder.load_state_dict(dec_weights)
   return dec

def dd_state(nn,img):
   """ dd_state

   Proposes an furniture occupancy grid given an model and an image

   Arguments:
       nn : Trained and initialized VAE
       img: Image matrix

   Returns:
       A 1xMxN matrix containting probability of occupancy for each
       cell in the room
   """
   device = nn.device()
   img_tensor = torch.from_numpy(img)
   img_tensor = Variable(img_tensor).to(device)
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
