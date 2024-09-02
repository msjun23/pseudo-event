import os
import yaml
import hydra
import random
import shutil
import numpy as np
import torch.distributed as dist
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
import torch.distributed
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

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
    
def train(cfg):
    torch.set_float32_matmul_precision(cfg.trainer.gpu_precision)
    
    # Initialize WandB Logger
    logger = WandbLogger(**cfg.logger.wandb)
    
    # Prepare model
    model = getattr(models, cfg.model.name)(cfg.model)
    
    # Learning rate monitor
    # lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True, log_weight_decay=True)
    
    # Prepare trainer
    trainer = pl.Trainer(**cfg.trainer.params, 
                        #  callbacks=[lr_monitor], 
                         logger=logger,
                         )
    trainer.fit(model)
    # trainer.test(model)
    
@hydra.main(config_path="configs", config_name="default", version_base="1.2")
def main(cfg: OmegaConf):
    # Initialize the process group if using DDP
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        
    # Set seed
    set_random_seed(cfg.seed)
    
    # Save scripts only if this is the master process (rank 0) or DDP is not initialized
    if not dist.is_initialized() or dist.get_rank() == 0:
        source_folder = '/root/code/scripts'
        destination_folder = 'scripts'
        if not os.path.exists(destination_folder):
            shutil.copytree(source_folder, destination_folder)
            
    # Cleanup process group
    if dist.is_initialized():
        dist.destroy_process_group()
        
    # Train
    train(cfg)
    print('\n', '# Save dir: ', HydraConfig.get().run.dir, '\n')
    
if __name__=='__main__':
    main()