import os
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from . pytypes import *
from . models import BaseVAE

class OGVAE(pl.LightningModule):
    """Task of embedding image space into z-space"""

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(OGVAE, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img = batch
        results = self.forward(real_img)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img = batch
        results = self.forward(real_img)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)


    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir ,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device)

            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir ,
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        if self.params['scheduler_gamma'] is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                         gamma = self.params['scheduler_gamma'])
            scheds.append(scheduler)

        return optims, scheds

class OGDecoder(pl.LightningModule):
    """Task of decoding z-space to grid space"""

    def __init__(self,
                 vae: BaseVAE,
                 decoder: Decoder,
                 params: dict) -> None:
        super(OGDecoder, self).__init__()

        self.vae = vae
        self.decoder = decoder
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.vae.encode(input)
        return self.decoder(mu)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, real_og = batch
        pred_og = self.forward(real_img)
        train_loss = self.model.loss_function(pred_og,
                                              real_og,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()},
                      sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, real_og = batch
        pred_og = self.forward(real_img)
        val_loss = self.model.loss_function(pred_og,
                                              real_og,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()},
                      sync_dist=True)


    def on_validation_end(self) -> None:
        self.sample_ogs()

    def sample_ogs(self):
        # Get sample reconstruction image
        test_input, test_og = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_og = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.forward(test_input)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir ,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)


    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.decoder.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        if self.params['scheduler_gamma'] is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                         gamma = self.params['scheduler_gamma'])
            scheds.append(scheduler)

        return optims, scheds
