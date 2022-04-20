import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from functional_scenes.og_proposal.model import BetaVAE, Decoder
from functional_scenes.og_proposal.tasks import OGVAE, OGDecoder
from functional_scenes.og_proposal.dataset import ogvae_loader, ogdecoder_loader

archs = {
    'BetaVAE' : BetaVAE,
    'Decoder' : Decoder
}

def main():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('config', type = str,
                        help =  'path to the config file')

    args = parser.parse_args()
    with open(f"/project/scripts/nn/configs/{args.config}.yaml", 'r') as file:
        config = yaml.safe_load(file)


    logger = CSVLogger(save_dir=config['logging_params']['save_dir'],
                       name=config['mode'] + '_' + config['model_params']['name'],)

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    if config['mode'] == 'og_vae':
        arch = archs[config['model_params']['name']](**config['model_params'])
        task = OGVAE(arch,  config['exp_params'])
        loader = ogvae_loader
    elif config['mode'] == 'og_decoder':
        decoder_arch = archs[config['model_params']['name']](**config['model_params'])
        vae_arch = archs[config['vae_params']['name']](**config['vae_params'])
        task = OGDecoder(vae_arch, decoder_arch,  config['exp_params'])
        loader = ogdecoder_loader
    else:
        # TODO
        # arch = models[config['model_params']['name']](**config['model_params'])
        # model = OGVAE(arch,  config['exp_params'])
        raise ValueError(f"mode {config['mode']} not recognized")

    
    runner = Trainer(logger=logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=2,
                                         dirpath =os.path.join(logger.log_dir , "checkpoints"),
                                         monitor= "val_loss",
                                         save_last=True),
                     ],
                     accelerator = 'auto',
                     # strategy=DDPPlugin(find_unused_parameters=False),
                     deterministic = True,
                     **config['trainer_params'])
    device = runner.device_ids[0]
    train_loader = loader(config['path_params']['train_path'],
                          device, 
                          **config['loader_params'])
    test_loader = loader(config['path_params']['test_path'],
                         device, 
                         **config['loader_params'])

    Path(f"{logger.log_dir}/samples").mkdir(exist_ok=True, parents=True)
    Path(f"{logger.log_dir}/reconstructions").mkdir(exist_ok=True, parents=True)
    print(f"======= Training {logger.name} =======")
    runner.fit(task, train_loader, test_loader)

if __name__ == '__main__':
    main()
