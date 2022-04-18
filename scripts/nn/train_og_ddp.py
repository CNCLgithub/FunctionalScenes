import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

def main():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('config',
                        metavar='FILE',
                        help =  'path to the config file')

    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)


    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                   name=config['model_params']['name'],)

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    arch = vae_models[config['model_params']['name']](**config['model_params'])
    model = OGVAE(arch,  config['exp_params'])

    runner = Trainer(logger=tb_logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=2,
                                         dirpath =os.path.join(tb_logger.log_dir , "checkpoints"),
                                         monitor= "val_loss",
                                         save_last=True),
                     ],
                     strategy=DDPPlugin(find_unused_parameters=False),
                     deterministic = True,
                     **config['trainer_params'])


    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, train_loader, test_loader)

if __name__ == '__init__':
    main()
