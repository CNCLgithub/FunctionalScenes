import os
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torchvision.utils as vutils
from torchvision.transforms.functional import (resize,
                                               rotate)

from . pytypes import *
from . model import VAE, Decoder

class SceneEmbedding(pl.LightningModule):
    """Task of embedding image space into z-space"""

    def __init__(self,
                 model: VAE,
                 beta: float = 1.0,
                 kld_weight: float = 1.0,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 sched_gamma: float = 0.9,
                 ) -> None:
        super(SceneEmbedding, self).__init__()
        self.model = model
        self.save_hyperparameters(ignore='model')

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def loss_function(self,
                      recons: Tensor,
                      x: Tensor,
                      mu: Tensor,
                      log_var: Tensor) -> dict:
        recons_loss = F.mse_loss(recons, x)
        kld_loss = torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
        kld_loss = torch.mean(-0.5 * kld_loss, dim = 0)
        # H-loss, see https://openreview.net/forum?id=Sy2fzU9gl
        loss = recons_loss + self.hparams.beta * \
            self.hparams.kld_weight * kld_loss
        return {'loss': loss, 'rec_loss':recons_loss, 'kld_loss':kld_loss}

    def training_step(self, batch, batch_idx):
        x = batch[0]
        mu, log_var, y = self.forward(x)
        l = self.loss_function(y, x, mu, log_var)
        self.log_dict(l)
        return l['loss']

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        mu, log_var, y = self.forward(x)
        l = self.loss_function(y, x, mu, log_var)
        vutils.save_image(y.data,
                          os.path.join(self.logger.log_dir ,
                                       "reconstructions",
                                       f"recons_{self.logger.name}_epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)
        self.log_dict({f"val_{key}": val.item()
                       for key, val in l.items()})
        self.sample_images()


    def sample_images(self):
        samples = self.model.sample(25,
                                    self.device)

        vutils.save_image(samples.cpu().data,
                        os.path.join(self.logger.log_dir ,
                                        "samples",
                                        f"{self.logger.name}_epoch_{self.current_epoch}.png"),
                        normalize=True,
                        nrow=12)

    def configure_optimizers(self):

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.hparams.lr,
                               weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma = self.hparams.sched_gamma)
        return [optimizer], [scheduler]

class OGDecoder(pl.LightningModule):
    """Task of decoding z-space to grid space"""

    def __init__(self,
                 encoder: SceneEmbedding,
                 decoder: Decoder,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 sched_gamma: float = 0.9,
                 ) -> None:
        super(OGDecoder, self).__init__()
        # REVIEW: is this the proper way to freeze the encoder?
        encoder.eval()
        encoder.freeze()
        self.encoder = encoder
        self.decoder = decoder
        self.save_hyperparameters(ignore=['encoder', 'decoder'])

    def forward(self, x: Tensor) -> Tensor:
        mu, log_var = self.encoder.model.encode(x)
        z = self.encoder.model.reparameterize(mu, log_var)
        return self.decoder(z)

    def loss_function(self, x: Tensor, y: Tensor):
        loss = F.mse_loss(x, y)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        x, og = batch
        pred_og = self.forward(x)
        train_loss = self.loss_function(pred_og, og)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        x, og = batch
        pred_og = self.forward(x)
        val_loss = self.loss_function(pred_og, og)
        og_pred_img = rotate(resize(pred_og.unsqueeze(1), 256), 90).data
        vutils.save_image(og_pred_img,
                          os.path.join(self.logger.log_dir ,
                                       "reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=False,
                          nrow=2)
        og_gt_img = rotate(resize(og.unsqueeze(1), 256), 90).data
        vutils.save_image(og_gt_img,
                          os.path.join(self.logger.log_dir ,
                                       "reconstructions",
                                       f"gt_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=False,
                          nrow=2)
        vutils.save_image(x.data,
                          os.path.join(self.logger.log_dir ,
                                       "reconstructions",
                                       f"input_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=2)
        self.log('val_loss', val_loss)
        self.sample_ogs()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def sample_ogs(self):
        samples = self.decoder.sample(25,
                                      self.device).unsqueeze(1)
        sdata = resize(samples, 256).cpu().data
        vutils.save_image(sdata ,
                        os.path.join(self.logger.log_dir ,
                                        "samples",
                                        f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                        normalize=False,
                        nrow=2)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.decoder.parameters(),
                               lr=self.hparams.lr,
                               weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma = self.hparams.sched_gamma)
        return [optimizer], [scheduler]
