import os
import yaml
import torch
import numpy as np

from . model import VAE, Decoder
from . tasks import SceneEmbedding, OGDecoder

# intialization of neural network for forward execution
def init_dd_state(config_path:str, device):

   with open(config_path, 'r') as file:
      config = yaml.safe_load(file)
   # first load encoder
   decoder = Decoder(**config['model_params'])
   vae = VAE(**config['vae_params'])
   encoder = SceneEmbedding.load_from_checkpoint(config['vae_chkpt'],
                                                 model = vae)
   decode_path = config['logging_params']['save_dir']
   decode_path = os.path.join(decode_path,
                              config['mode'])
   # TODO: generalize version
   decode_path = os.path.join(decode_path,
                              'version_0',
                              'checkpoints',
                              'last.ckpt')
   task = OGDecoder.load_from_checkpoint(decode_path,
                                         encoder = encoder,
                                         decoder = decoder)
   task.eval()
   task.freeze()
   return task

@torch.no_grad()
def dd_state(nn, x, device):
   """ dd_state

   Proposes an furniture occupancy grid given an model and an image

   Arguments:
       nn : Trained and initialized VAE
       x: Image matrix

   Returns:
       A 1xMxN matrix containting probability of occupancy for each
       cell in the room
   """
   x = torch.tensor(x, device = device)
   if x.dim == 3:
      # single image -> batchxCxHxW
      x = x.unsqueeze(0)
   x = nn.forward(x)
   x = x.cpu().detach().numpy()
   return x
