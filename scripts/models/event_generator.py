import copy
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
from utils.helper import pad_array_to_match, pad_tensor_to_match
from models.stream_embedding import encode_patches_to_vocab
from dataset.provider import DatasetProvider
from dataset.sequential_provider import SequentialDatasetProvider


class EventGenerator(pl.LightningModule):
    def __init__(self, cfg):
        super(EventGenerator, self).__init__()
        
        # Event stream to batched sequence input
        # self.stream_embedding = EventStreamEmbedding(**cfg.stream_embedding)
        
        # Mamba Language Model
        mamba_cfg = MambaConfig(**cfg.mamba)
        self.mamba_lm = MambaLMHeadModel(mamba_cfg)
        
        self.cfg_dataset = cfg.dataset
        self.cfg_optim = cfg.optimizer
        self.vocab_path = cfg.vocab_path
        
        self.prev_pred = None
        self.fig_num = 0
        
    def setup(self, stage=None):
        # At training
        if stage == 'fit' or stage is None:
            dataset_provider = DatasetProvider(**self.cfg_dataset.dataset)
            self.train_dataset = dataset_provider.get_train_dataset()
            
            self.criterion = torch.nn.CrossEntropyLoss()
            
        # At inference
        if stage == 'test' or stage is None:
            seq_dataset_provider = SequentialDatasetProvider(**self.cfg_dataset.dataset)
            self.test_dataset = seq_dataset_provider.get_test_dataset()
            
        self.event_vocab = torch.load(self.vocab_path).to(self.device)
        
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
                    event_l.append(encode_patches_to_vocab(chw))
                event_l = torch.stack(event_l)  # [L hw]
                event_b.append(event_l)
            event_b = torch.stack(event_b)      # [B L hw]
            event_indices = rearrange(event_b, 'B L hw -> (B hw) L').to(event.device)   # [Bhw L]
        
        # logits, pred_indices = self(event_indices)  # [Bhw L V], [Bhw L]
        logits, pred_indices = self(event_indices[:,:-1])  # [Bhw L V], [Bhw L], not use last as input
        '''
        input : a b c d e f g <- can be used as both input and self_gt
                 / / / / / /
        output: b c d e f g h <- 'h' is the event stream w/o gt at next timestep
        Calculate loss using above pairs
        '''
        # logits = logits[:,:-1,:]
        self_gt = event_indices[:,1:]
        logits = rearrange(logits, 'Bhw L V -> (Bhw L) V')
        self_gt = rearrange(self_gt, 'Bhw L -> (Bhw L)')
        
        self._show_sequences(event, pred_indices)
        
        loss = self.criterion(logits, self_gt)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Save checkpoint every end of step
        loss = outputs['loss'].item()
        checkpoint_path = f'checkpoints/step_{self.global_step}_loss_{loss:.4f}.ckpt'
        self.trainer.save_checkpoint(checkpoint_path)
        
    def on_train_epoch_end(self):
        # Include loss info to file name
        loss = f"{self.trainer.callback_metrics['train_loss'].item():.4f}"
        checkpoint_path = f'checkpoints/epoch_{self.current_epoch}_loss_{loss}.ckpt'
        self.trainer.save_checkpoint(checkpoint_path)
        
    def test_step(self, batch, batch_idx):
        seq_name = batch['sequence_name']
        event = batch['event']      # [B L C H W], B (batch size) must be '1'
        real_event = copy.deepcopy(batch['event'])  # Just for visualization
        H, W = event.shape[3:]
        h, w = H//2, W//3   # event patch size: [c=2 h=2 w=3]
        
        # Update input event stream using previous prediction
        if self.prev_pred is not None:
            # Use prev_pred if there are not enough events
            for seq_idx in range(event.size(1)):
                if event[0,seq_idx].mean() < 0.1:   # Not enough events in sequence
                    event[0,seq_idx] = (event[0,seq_idx].bool() | self.prev_pred[0,seq_idx].bool()).float()
                    # event[0,seq_idx] = self.prev_pred[0,seq_idx]
            
        with torch.no_grad():
            event_b = []
            for lchw in event:
                event_l = []
                for chw in lchw:
                    event_l.append(encode_patches_to_vocab(chw))
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
        
        # Save output sequence as self.prev_pred, [B L C H W]
        pred_ev_list = []
        for l in range(pred_indices.size(-1)):
            patches = self.event_vocab[pred_indices[:,l]]   # [Bhw 2 2 3], B=1
            patches = patches.view(h, w, 2, 2, 3)           # [h w 2 2 3]
            pred_ev = patches.permute(2, 0, 3, 1, 4).contiguous().view(2, h*2, w*3) # [2 H W]
            
            if not event[0,l,...].shape == pred_ev.shape:   # [C H W]
                pred_ev = pad_tensor_to_match(event[0,l,...], pred_ev)
            assert event[0,l,...].shape == pred_ev.shape
            pred_ev_list.append(pred_ev)
        self.prev_pred = torch.stack(pred_ev_list, dim=0).unsqueeze(dim=0).float()  # [B L C H W]
        
        # self._show_sequences(event, pred_indices)
        self._show_sequences_3row(real_event, event, pred_indices, batch_idx)
        
    def configure_optimizers(self):
        step_cycle = self.trainer.max_steps
        warmup_steps = int(self.trainer.max_steps * 0.05)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.parameters(), **self.cfg_optim)
        
        # Warmup scheduler
        warmup_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer, 
                lr_lambda=lambda step: min(1.0, step / warmup_steps)    # Warmup for given steps
            ),
            'interval': 'step',
            'frequency': 1,
            'name': 'warmup_scheduler'
        }
        # Cosine Annealing scheduler
        cosine_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_cycle),   # T_max is the number of steps in one cycle
            'interval': 'step', # Update every step
            'frequency': 1,
            'name': 'cosine_scheduler'
        }
        
        return [optimizer], [warmup_scheduler, cosine_scheduler]
    
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
        # num_iter = pred_indices.size(-1) - 1    # Except last timestep; no comparable input
        num_iter = pred_indices.size(-1)        # length of (L - 1)
        fig, axes = plt.subplots(2, num_iter, figsize=(num_iter * 2, 4))
        for l in range(num_iter):
            input_ev = event[0,l+1,...]                 # [C H W]
            # save_as_rb_img(input_ev, f'original_img_{l}.png')
            input_ev_np = return_as_rb_img(input_ev)    # batch size must be 1, np, (H W 3)
            
            patches = self.event_vocab[pred_indices[:,l]]   # [Bhw 2 2 3], B=1
            patches = patches.view(h, w, 2, 2, 3)           # [h, w, 2, 2, 3]
            pred_ev = patches.permute(2, 0, 3, 1, 4).contiguous().view(2, h*2, w*3) # [C H W]
            # save_as_rb_img(pred_ev, f'predicted_img_{l}.png')
            pred_ev_np = return_as_rb_img(pred_ev)          # np, (H W 3)
            
            if not input_ev_np.shape == pred_ev_np.shape:
                pred_ev_np = pad_array_to_match(input_ev_np, pred_ev_np)
            assert input_ev_np.shape == pred_ev_np.shape
            
            pix_err = np.abs(input_ev_np/255 - pred_ev_np/255).sum() / (2*H*W) * 100.
            axes[0, l].set_title(f'timestep={l+1}, input', fontsize=10)  # Add title to the top row
            axes[0, l].imshow(input_ev_np)
            axes[0, l].axis('off')
            axes[1, l].set_title(f'predicted, pix_err: {pix_err:.2f}%', fontsize=10)
            axes[1, l].imshow(pred_ev_np)
            # axes[1, l].text(0.5, -0.1, f'Pix_err: {pix_err:.2f}%', ha='center', transform=axes[0, l].transAxes, fontsize=8)
            axes[1, l].axis('off')
        # Save figure
        plt.tight_layout()
        plt.savefig('sequence_vis.png')
        plt.clf()
        plt.close()
        
    def _show_sequences_3row(self, real_event, event, pred_indices, batch_idx=0):
        if self.fig_num % 2 == 0:
            # event_input: [B L C H W]
            # pred_indices: [Bhw L]
            H, W = event.shape[3:]
            h, w = H//2, W//3   # event patch size: [c=2 h=2 w=3]
            
            # Set figure size (row: 3, column: num_iterations)
            num_iter = pred_indices.size(-1) - 1    # Except fist timestep; no comparable output
            fig, axes = plt.subplots(3, num_iter, figsize=(num_iter * 2, 4))
            for l in range(num_iter):
                real = real_event[0,l+1,...]
                real_np = return_as_rb_img(real)    # batch size must be 1, np, (H W 3)
                
                input_ev = event[0,l+1,...]                 # [C H W]
                # save_as_rb_img(input_ev, f'original_img_{l}.png')
                input_ev_np = return_as_rb_img(input_ev)    # batch size must be 1, np, (H W 3)
                
                patches = self.event_vocab[pred_indices[:,l]]   # [Bhw 2 2 3], B=1
                patches = patches.view(h, w, 2, 2, 3)           # [h, w, 2, 2, 3]
                pred_ev = patches.permute(2, 0, 3, 1, 4).contiguous().view(2, h*2, w*3) # [C H W]
                # save_as_rb_img(pred_ev, f'predicted_img_{l}.png')
                pred_ev_np = return_as_rb_img(pred_ev)          # np, (H W 3)
                
                if not input_ev_np.shape == pred_ev_np.shape:
                    pred_ev_np = pad_array_to_match(input_ev_np, pred_ev_np)
                assert input_ev_np.shape == pred_ev_np.shape
                
                pix_err = np.abs(real_np/255 - pred_ev_np/255).sum() / (2*H*W) * 100.
                axes[0, l].set_title(f'timestep={l+1}, real event', fontsize=10)  # Add title to the top row
                axes[0, l].imshow(real_np)
                axes[0, l].axis('off')
                axes[1, l].set_title(f'input', fontsize=10)
                axes[1, l].imshow(input_ev_np)
                axes[1, l].axis('off')
                axes[2, l].set_title(f'predicted, pix_err: {pix_err:.2f}%', fontsize=10)
                axes[2, l].imshow(pred_ev_np)
                axes[2, l].axis('off')
            # Save figure
            plt.tight_layout()
            plt.savefig(f'sequence_vis_{batch_idx}.png')
            plt.clf()
            plt.close()
        self.fig_num += 1