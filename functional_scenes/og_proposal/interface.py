import os
import yaml
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder

# place interal imports at the end
# from . dataset import CustomImageFolder
from . model import BetaVAE, Decoder
from . tasks import OGVAE, OGDecoder

archs = {
    'BetaVAE' : BetaVAE,
    'Decoder' : Decoder
}

# intialization of neural network for forward execution
def init_dd_state(config_path:str, device):

   with open(config_path, 'r') as file:
      config = yaml.safe_load(file)
   # first load encoder
   decoder_arch = archs[config['model_params']['name']](**config['model_params'])
   vae_arch = archs[config['vae_params']['name']](**config['vae_params'])
   vae = OGVAE.load_from_checkpoint(config['vae_chkpt'],
                                    vae_model = vae_arch,
                                    params = config['exp_params'])
   decode_path = config['logging_params']['save_dir']
   decode_path = os.path.join(decode_path,
                              config['mode'] + '_' + config['model_params']['name'])
   decode_path = os.path.join(decode_path,
                              'version_6',
                              'checkpoints',
                              'last.ckpt')
   task = OGDecoder.load_from_checkpoint(decode_path,
                                          vae = vae,
                                          decoder = decoder_arch,
                                          params = config['exp_params'])
   task.eval()
   task.freeze()
   return task

def dd_state(nn, img, device):
   """ dd_state

   Proposes an furniture occupancy grid given an model and an image

   Arguments:
       nn : Trained and initialized VAE
       img: Image matrix

   Returns:
       A 1xMxN matrix containting probability of occupancy for each
       cell in the room
   """
   img_tensor = torch.Tensor(img).float()
   x = nn.forward(img_tensor)
   x = x.cpu().detach().numpy()
   return x
