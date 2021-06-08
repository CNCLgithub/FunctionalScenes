"""solver.py"""

import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
#import visdom
import json

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda
from model import BetaVAE_H, BetaVAE_B, BetaVAE_OG
from dataset import return_data

from torchvision import transforms

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss

def l2_distance(og,og_recon):
    batch_size = og.size(0)
    l2_dist = torch.dist(og, og_recon, p=2).div(batch_size)
    return l2_dist


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0

        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        if args.dataset.lower() == 'occupancy_grid_data_driven_twodoors':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'test_occupancy_grid_data_driven_twodoors':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        else:
            raise NotImplementedError

        if args.model == 'H':
            net = BetaVAE_H
        elif args.model == 'B':
            net = BetaVAE_B
        else:
            raise NotImplementedError('only support model H or B')

        # optimizer for rendering room image
        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))
        
        #parameter about visualization, skip for now
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None
        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port)

        # load checkpoint if any
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        # data loader
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)
        self.test_iter = 0
        self.test_og_iter = 0
        self.og_iter = 0
        self.gather = DataGather()
      
        self.ckpt_name_og = args.ckpt_name_og

        # whether to train occupancy grid or room 
        self.train = args.train
        if not self.train:
            self.BetaVAE_OG = cuda(BetaVAE_OG(self.net.encoder,self.z_dim, self.nc), self.use_cuda)
        #print(list(self.net.encoder.parameters()))

    def train_OG(self):
        out = False
        pbar = tqdm(total=self.max_iter)
        pbar.update(self.og_iter)
        file_path = os.path.join(self.ckpt_dir, self.ckpt_name)
        checkpoint = torch.load(file_path)
        self.net.load_state_dict(checkpoint['model_states']['net'])
        
        # load checkpoint for occupancy grid if any
        self.load_checkpoint_occupancygrid(self.ckpt_name_og)
        
        # train decoder parameters only
        #occupancy_grid_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.BetaVAE_OG.parameters()), lr=self.lr,
        #                            betas=(self.beta1, self.beta2))
        occupancy_grid_optimizer = optim.Adam(self.BetaVAE_OG.decoder.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))
        while not out:
            for x,path in self.data_loader:
                self.og_iter += 1
                pbar.update(1)
                x = Variable(cuda(x, self.use_cuda))
                z_recon, mu, logvar = self.BetaVAE_OG(x)
                og_true_batch = []
                for occupancy_grid_path in path:
                       with open(os.path.join(occupancy_grid_path.replace('/render.png',''),'og.json'),'r') as f:
                              og_true = torch.FloatTensor(json.load(f)).unsqueeze(0)
                              og_true_batch.append(og_true)
                           
                og_true_batch = torch.stack(og_true_batch,dim = 0) #tensor of dimension [batch_size, channel = 1, room_height, room_width]

                occupancy_grid_distance = l2_distance(og_true_batch.cuda(),z_recon)
                occupancy_grid_optimizer.zero_grad()
                occupancy_grid_distance.backward()
                occupancy_grid_optimizer.step()
              
                if self.og_iter%self.display_step == 0:
                    pbar.write('[{}] occupancy_grid_distance:{:.3f}'.format(
                        self.og_iter, occupancy_grid_distance.item()))

                if self.og_iter%self.save_step == 0:
                    #self.save_checkpoint_occupancygrid('og')
                    self.save_checkpoint_occupancygrid(str(self.ckpt_name_og))
                    pbar.write('Saved checkpoint(iter:{})'.format(self.og_iter))

                if self.og_iter >= self.max_iter:
                    out = True 
                    break


    def test_OG(self):
        file_path_og = os.path.join(self.ckpt_dir, self.ckpt_name_og)
        checkpoint_og = torch.load(file_path_og)
        self.load_checkpoint_occupancygrid(self.ckpt_name_og)
        #self.BetaVAE_OG.decoder.load_state_dict(checkpoint_og['model_states']['net'])

        for x,path in self.data_loader:
                self.test_og_iter += 1
                x = Variable(cuda(x, self.use_cuda))
                z_recon, mu, logvar = self.BetaVAE_OG(x)
                og_true_batch = []
                for occupancy_grid_path in path:
                       with open(os.path.join(occupancy_grid_path.replace('/render.png',''),'og.json'),'r') as f:
                              og_true = torch.FloatTensor(json.load(f)).unsqueeze(0)
                              og_true_batch.append(og_true)
                og_true_batch = torch.stack(og_true_batch,dim = 0)
                self.reconstructed_z(og_true_batch,z_recon)

    def test_vae(self):
        file_path = os.path.join(self.ckpt_dir, self.ckpt_name)
        checkpoint = torch.load(file_path)
        self.net.load_state_dict(checkpoint['model_states']['net'])
        #print(list(self.net.encoder.parameters()))
        print("checkpoint loaded for testing")

        for x,path in self.data_loader: 
            self.test_iter += 1
            x = Variable(cuda(x, self.use_cuda))
            x_recon, mu, logvar = self.net(x)
            self.gather.insert(images=x.data)
            self.gather.insert(images=F.sigmoid(x_recon).data)
            self.reconstructed_image()
            self.gather.flush()

    def train_vae(self):
        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        out = False
        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)

        while not out:
            for x,path in self.data_loader:
                self.global_iter += 1
                pbar.update(1)
                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar = self.net(x)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                if self.objective == 'H':
                    beta_vae_loss = recon_loss + self.beta*total_kld
                elif self.objective == 'B':
                    C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
                    beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                if self.viz_on and self.global_iter%self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,
                                       mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                       recon_loss=recon_loss.data, total_kld=total_kld.data,
                                       dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data)

                if self.global_iter%self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                        #self.global_iter, recon_loss.data[0], total_kld.data[0], mean_kld.data[0]))
                        self.global_iter, recon_loss.item(), total_kld.item(), mean_kld.item()))

                    var = logvar.exp().mean(0).data
                    var_str = ''
                    for j, var_j in enumerate(var):
                        var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                    pbar.write(var_str)

                    if self.objective == 'B':
                        pbar.write('C:{:.3f}'.format(C.data[0]))

                    if self.viz_on:
                        self.gather.insert(images=x.data)
                        self.gather.insert(images=F.sigmoid(x_recon).data)
                        self.viz_reconstruction()
                        self.viz_lines()
                        self.gather.flush()

                if self.global_iter%self.save_step == 0:
                    #self.save_checkpoint('last')
                    self.save_checkpoint(str(self.ckpt_name))
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))

                if self.global_iter%50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()

    def reconstructed_image(self):
        #self.net_mode(train=False)
        output_dir = os.path.join(self.output_dir, str(self.global_iter)+str(self.model)+"image_test_twodoors")
        os.makedirs(output_dir, exist_ok=True)
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True)
        x_recon = self.gather.data['images'][1][:100]
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()
        #images = x_recon.cpu()
        inv_transform = transforms.Resize((480, 720))
        images = inv_transform(images)
        save_image(tensor=images,
                      fp=os.path.join(output_dir, '{}.jpg'.format(self.test_iter)),
                               nrow=8, pad_value=1)
        #self.net_mode(train=True)

    def reconstructed_z(self,og_true_batch,z_recon):
        output_dir = os.path.join(self.output_dir, str(self.max_iter)+str(self.model)+"OG_test_twodoors")
        os.makedirs(output_dir, exist_ok=True)
        inv_transform = transforms.Resize((480, 720))
        og_true_image = make_grid(og_true_batch.cuda(), normalize=True)
        z_recon_image = make_grid(z_recon, normalize=True)
        images = torch.stack([og_true_image,z_recon_image],dim = 0).cpu()
        #images = z_recon_image.cpu()
        #images = og_true_image.cpu()
        images = inv_transform(images)
        save_image(tensor=images,
                      fp=os.path.join(output_dir, '{}.jpg'.format(self.test_og_iter)),
                               nrow=8, pad_value=1)


    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()


    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'kld':self.win_kld,
                      'mu':self.win_mu,
                      'var':self.win_var,}
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))


    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))


    def save_checkpoint_occupancygrid(self, filename, silent=True):
        model_states = {'net':self.BetaVAE_OG.decoder.state_dict(),}
        states = {'iter':self.og_iter,
                  'model_states':model_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint for occupancy grid '{}' (iter {})".format(file_path, self.og_iter))


    def load_checkpoint_occupancygrid(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.og_iter = checkpoint['iter']
            self.BetaVAE_OG.decoder.load_state_dict(checkpoint['model_states']['net'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.og_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))






