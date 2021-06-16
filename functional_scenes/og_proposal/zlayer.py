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

ckpt_dir = 'checkpoints/main'
ckpt_name = 'last'
ckpt_name_og = 'og'
use_cuda = torch.cuda.is_available()
z_dim = 20
nc = 3
net = cuda(BetaVAE_H(z_dim, nc), use_cuda)

#net.load_state_dict(checkpoint['model_states']['net'])
#BetaVAE_OG = cuda(BetaVAE_OG(net.encoder,z_dim, nc), self.use_cuda)

def dd_state(net,img):
   file_path = os.path.join(ckpt_dir, ckpt_name)
   checkpoint = torch.load(file_path)
   net.load_state_dict(checkpoint['model_states']['net'])

   OG_model = cuda(BetaVAE_OG(net.encoder,z_dim, nc),use_cuda)
   file_path_og = os.path.join(ckpt_dir, ckpt_name_og)
   checkpoint_og = torch.load(file_path_og)
   OG_model.decoder.load_state_dict(checkpoint_og['model_states']['net'])

   x = Variable(cuda(img, use_cuda))
   z, mu, logvar = OG_model(x)
   #z = net.encoder(x)
   z = z.cpu().detach().numpy()
   return z

if __name__ == "__main__":
    #path_to_file= 'output/datasets/occupancy_grid_data_driven_twodoors/'
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
    for x, path in loader:
        z = dd_state(net,x)
        #print(z.shape)
    #images1 = iter(loader).next()
   # im_frame = Image.open(path_to_file + '1/render.png')
    #np_frame = np.array(im_frame)
   # transform = transforms.Compose([
      #  transforms.Resize((64, 64)),
     #   transforms.ToTensor(),])
    #img = transform(im_frame)
    #dd_state(net,images1)
    print("Everything passed")
