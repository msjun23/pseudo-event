import os
import yaml
import hydra
import random
import shutil
import numpy as np
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
import torch.distributed
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import models


def set_random_seed(seed):
    # Python
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Pytorch-Lightning
    pl.seed_everything(seed)
    
def test(cfg):
    torch.set_float32_matmul_precision(cfg.trainer.gpu_precision)
    
    # Initialize WandB Logger
    logger = WandbLogger(**cfg.logger.wandb)
    
    # Prepare model
    model = getattr(models, cfg.model.name)(cfg.model)
    # Load pretrained weights if specified
    if cfg.model.ckpt_path is not None:
        print(f'\n Load from {cfg.model.ckpt_path} \n')
        checkpoint = torch.load(cfg.model.ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('Please specify ckpt_path in model config file.')
        return
    
    # Prepare trainer
    trainer = pl.Trainer(**cfg.trainer.params, 
                         logger=logger,
                         )
    # trainer.fit(model)
    trainer.test(model)
    
@hydra.main(config_path="configs", config_name="default", version_base="1.2")
def main(cfg: OmegaConf):
    # Set seed
    set_random_seed(cfg.seed)
    
    # Save scripts
    source_folder = '/root/code/scripts'
    shutil.copytree(source_folder, 'scripts')
    
    # Test
    test(cfg)
    print('\n', '# Save dir: ', HydraConfig.get().run.dir, '\n')
    
if __name__=='__main__':
    main()