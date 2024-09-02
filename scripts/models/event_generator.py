import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from torch.utils.data import DataLoader

from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

from utils.rb_image import save_as_rb_img, return_as_rb_img
from utils.helper import pad_array_to_match
from models.stream_embedding import match_patches_to_vocab
from dataset.provider import DatasetProvider


class EventGenerator(pl.LightningModule):
    def __init__(self, cfg):
        super(EventGenerator, self).__init__()
        
        # Event stream to batched sequence input
        # self.stream_embedding = EventStreamEmbedding(**cfg.stream_embedding)
        
        # Mamba Language Model
        mamba_cfg = MambaConfig(**cfg.mamba)
        self.mamba_lm = MambaLMHeadModel(mamba_cfg)
        
        self.cfg_dataset = cfg.dataset
        self.vocab_path = cfg.vocab_path
        self.setup()
        
    def setup(self, stage=None):
        dataset_provider = DatasetProvider(**self.cfg_dataset.dataset)
        self.train_dataset = dataset_provider.get_train_dataset()
        self.test_dataset = dataset_provider.get_test_dataset()
        
        self.event_vocab = torch.load(self.vocab_path).to(self.device)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        # Mamba LM
        logits = self.mamba_lm(x).logits    # [Bhw L] -> [Bhw L V]
        
        # The predicted next step's event stream
        with torch.no_grad():
            # Convert logits into a probability distribution using the softmax function
            probabilities = F.softmax(logits, dim=-1)
            # Select the index of the token with the highest probability
            predicted_indices = torch.argmax(probabilities, dim=-1)
        return logits, predicted_indices
    
    def training_step(self, batch, batch_idx):
        seq_name = batch['sequence_name']
        file_index = batch['file_index']
        event = batch['event']['left']      # [B L C H W], B (batch size) must be '1'
        # gt = batch['disparity_gt']
        
        with torch.no_grad():
            event_b = []
            for lchw in event:
                event_l = []
                for chw in lchw:
                    event_l.append(match_patches_to_vocab(self.event_vocab, chw))
                event_l = torch.stack(event_l)  # [L hw]
                event_b.append(event_l)
            event_b = torch.stack(event_b)      # [B L hw]
            event_indices = rearrange(event_b, 'B L hw -> (B hw) L').to(event.device)   # [Bhw L]
        
        logits, pred_indices = self(event_indices)  # [Bhw L V], [Bhw L]
        '''
        input : a b c d e f g <- can be used as both input and self_gt
                 / / / / / /
        output: b c d e f g h <- 'h' is the event stream w/o gt at next timestep
        Calculate loss using above pairs
        '''
        logits = logits[:,:-1,:]
        self_gt = event_indices[:,1:]
        
        self._show_sequences(event, pred_indices)
        
        loss = self.criterion(rearrange(logits, 'Bhw L V -> (Bhw L) V'), rearrange(self_gt, 'Bhw L -> (Bhw L)'))
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Save checkpoint every end of step
        loss = outputs['loss'].item()
        checkpoint_path = f'checkpoints/step_{self.global_step}_loss_{loss:.4f}.ckpt'
        self.trainer.save_checkpoint(checkpoint_path)
        
    def on_train_epoch_end(self):
        self._save_model_checkpoint()
        
    def test_step(self, batch, batch_idx):
        seq_name = batch['sequence_name']
        file_index = batch['file_index']
        event = batch['event']['left']      # [B L C H W], B (batch size) must be '1'
        
        with torch.no_grad():
            event_b = []
            for lchw in event:
                event_l = []
                for chw in lchw:
                    event_l.append(match_patches_to_vocab(self.event_vocab, chw))
                event_l = torch.stack(event_l)  # [L hw]
                event_b.append(event_l)
            event_b = torch.stack(event_b)      # [B L hw]
            event_indices = rearrange(event_b, 'B L hw -> (B hw) L').to(event.device)   # [Bhw L]
        
        logits, pred_indices = self(event_indices)  # [Bhw L V], [Bhw L]
        '''
        input : a b c d e f g <- can be used as both input and self_gt
                 / / / / / /
        output: b c d e f g h <- 'h' is the event stream w/o gt at next timestep
        Calculate loss using above pairs
        '''
        
        self._show_sequences(event, pred_indices)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg_dataset.dataloader.train)
    
    def val_dataloader(self):
        raise NotImplementedError
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg_dataset.dataloader.test)
    
    '''
    Custom functions
    '''
    def _show_sequences(self, event, pred_indices):
        # event_input: [B L C H W]
        # pred_indices: [Bhw L]
        H, W = event.shape[3:]
        h, w = H//2, W//3   # event patch size: [c=2 h=2 w=3]
        
        # Set figure size (row: 2, column: num_iterations)
        num_iter = pred_indices.size(-1) - 1    # Except last timestep; no comparable input
        fig, axes = plt.subplots(2, num_iter, figsize=(num_iter * 2, 4))
        for l in range(num_iter):
            # save_as_rb_img(event[0,l,...], f'original_img_{l}.png')
            input_ev = return_as_rb_img(event[0,l+1,...])   # batch size must be 1, np, (H W 3)
            
            patches = self.event_vocab[pred_indices[:,l]]   # [Bhw 2 2 3], B=1
            patches = patches.view(h, w, 2, 2, 3)           # [h, w, 2, 2, 3]
            pred_ev = patches.permute(2, 0, 3, 1, 4).contiguous().view(2, h*2, w*3) # [2 H W]
            # save_as_rb_img(pred_ev, f'predicted_img_{l}.png')
            pred_ev = return_as_rb_img(pred_ev)             # np, (H W 3)
            
            if not input_ev.shape == pred_ev.shape:
                pred_ev = pad_array_to_match(input_ev, pred_ev)
                
            axes[0, l].set_title(f'timestep={l+1}, input', fontsize=10)  # Add title to the top row
            axes[0, l].imshow(input_ev)
            axes[0, l].axis('off')
            axes[1, l].set_title(f'predicted', fontsize=10)
            axes[1, l].imshow(pred_ev)
            axes[1, l].axis('off')
        # Save figure
        plt.tight_layout()
        plt.savefig('sequence_vis.png')
        plt.clf()
        plt.close()
    
    def _save_model_checkpoint(self):
        # Include loss info to file name
        loss_str = f"{self.trainer.callback_metrics['train_loss'].item():.4f}"
        checkpoint_path = f'checkpoints/epoch_{self.current_epoch}_loss_{loss_str}.ckpt'
        self.trainer.save_checkpoint(checkpoint_path)