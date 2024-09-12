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
        # # To use sparse ts for training
        # batch_event = batch['event']['left']      # [B L C H W], B (batch size) must be '1'
        # sparse_event_list = []
        # for ts in range(batch_event.size(1)):
        #     if ts % 10 == 0:
        #         sparse_event_list.append(batch_event[:,ts,...])
        # event = torch.stack(sparse_event_list, dim=1)
        # gt = batch['disparity_gt']
        
        with torch.no_grad():
            # Frame to indices
            B_hw_L = torch.stack([encode_patches_to_vocab(lchw.float()).T for lchw in event], dim=0)    # [hw L] * B
            event_indices = rearrange(B_hw_L, 'B hw L -> (B hw) L').to(event.device)    # [Bhw L]
        
        # logits, pred_indices = self(event_indices)  # [Bhw L V], [Bhw L]
        logits, pred_indices = self(event_indices[:,:-1])  # [Bhw L V], [Bhw L], not use last as input
        '''
        input : a b c d e f g   <- can be used as both input and self_gt
                 \ \ \ \ \ \    <- prediction pairs
        output:   b c d e f g h <- 'h' is the event stream w/o gt at next timestep
        Calculate loss using above pairs; b - g pairs
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
        if self.global_step % 1000 == 0:
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
        
        # # Update input event stream using previous prediction
        # if self.prev_pred is not None:
        #     # Use prev_pred if there are not enough events
        #     for seq_idx in range(event.size(1)):
        #         if event[0,seq_idx].mean() < 0.1:   # Not enough events in sequence
        #             event[0,seq_idx] = (event[0,seq_idx].bool() | self.prev_pred[0,seq_idx].bool()).float()
        #             # event[0,seq_idx] = self.prev_pred[0,seq_idx]
        
        # Define the 4 sub-tensors by slicing
        sub_events = [
            event[..., 0:H//2, 0:W//2], # Top-left
            event[..., 0:H//2, W//2:W], # Top-right
            event[..., H//2:H, 0:W//2], # Bottom-left
            event[..., H//2:H, W//2:W]  # Bottom-right
        ]
        # Loop through the sub-events
        sub_pred = []
        for i, sub_event in enumerate(sub_events):  # [B L C H W]
            sub_H, sub_W = sub_event.shape[3:]
            sub_h, sub_w = sub_H//2, sub_W//3   # event patch size: [c=2 h=2 w=3]
            
            # Frame to indices
            B_hw_L = torch.stack([encode_patches_to_vocab(lchw.float()).T for lchw in sub_event], dim=0)    # [hw L] * B
            event_indices = rearrange(B_hw_L, 'B hw L -> (B hw) L').to(sub_event.device)    # [Bhw L]
            
            logits, pred_indices = self(event_indices)  # [Bhw L V], [Bhw L]
            '''
            input : a b c d e f g   <- can be used as both input and self_gt
                    \ \ \ \ \ \    <- prediction pairs
            output:   b c d e f g h <- 'h' is the event stream w/o gt at next timestep
            Calculate loss using above pairs; b - g pairs
            '''
            
            # Save output sequence as self.prev_pred, [B L C H W]
            pred_ev_list = []
            for l in range(pred_indices.size(-1)):
                patches = self.event_vocab[pred_indices[:,l]]   # [Bhw 2 2 3], B=1
                patches = patches.view(sub_h, sub_w, 2, 2, 3)   # [h w 2 2 3]
                pred_ev = patches.permute(2, 0, 3, 1, 4).contiguous().view(2, sub_h*2, sub_w*3) # [2 H W]
                
                if not sub_event[0,l,...].shape == pred_ev.shape:   # [C H W]
                    pred_ev = pad_tensor_to_match(sub_event[0,l,...], pred_ev)
                assert sub_event[0,l,...].shape == pred_ev.shape
                pred_ev_list.append(pred_ev)
            sub_pred.append(torch.stack(pred_ev_list, dim=0).unsqueeze(dim=0).to(torch.uint8))  # [B L C H W]
            del logits, pred_indices    # To free GPU memory
            
        # Sub-tensors after processing
        top_left = sub_pred[0]      # [B, L, C, H//2, W//2]
        top_right = sub_pred[1]     # [B, L, C, H//2, W//2]
        bottom_left = sub_pred[2]   # [B, L, C, H//2, W//2]
        bottom_right = sub_pred[3]  # [B, L, C, H//2, W//2]
        # Recreate the top half and bottom half by concatenating horizontally (dim=-1 for W dimension)
        top_half = torch.cat((top_left, top_right), dim=-1)             # [B, L, C, H//2, W]
        bottom_half = torch.cat((bottom_left, bottom_right), dim=-1)    # [B, L, C, H//2, W]
        full_ev_pred = torch.cat((top_half, bottom_half), dim=-2)       # [B, L, C, H, W]
        self.prev_pred = full_ev_pred
        
        self._show_sequences_3row(real_event, event, full_ev_pred, batch_idx)
        
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
        ph, pw = 2, 3
        h, w = H//ph, W//pw   # event patch size: [c=2 ph pw]
        
        # Set figure size (row: 2, column: num_iterations)
        # num_iter = pred_indices.size(-1) - 1    # Except last timestep; no comparable input
        # num_iter = pred_indices.size(-1)        # length of (L - 1)
        num_iter = 10
        fig, axes = plt.subplots(2, num_iter, figsize=(num_iter * 2, 4))
        for l in range(num_iter):
            input_ev = event[0,l+1,...]                 # [C H W]
            # save_as_rb_img(input_ev, f'original_img_{l}.png')
            input_ev_np = return_as_rb_img(input_ev)    # batch size must be 1, np, (H W 3)
            
            patches = self.event_vocab[pred_indices[:,l]]   # [Bhw 2 2 3], B=1
            patches = patches.view(h, w, 2, ph, pw)           # [h, w, 2, ph, pw]
            pred_ev = patches.permute(2, 0, 3, 1, 4).contiguous().view(2, h*ph, w*pw) # [C H W]
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
            # To debug
            # Comment out below two lines when you use for real
            diff_w_prev = np.abs(input_ev_np/255 - return_as_rb_img(event[0,l,...])/255).sum() / (2*H*W) * 100
            axes[1, l].text(0.5, -0.1, f'diff_w_prev: {diff_w_prev:.2f}%', ha='center', transform=axes[0, l].transAxes, fontsize=8)
            axes[1, l].axis('off')
        # Save figure
        plt.tight_layout()
        plt.savefig('sequence_vis.png')
        plt.clf()
        plt.close()
        
    def _show_sequences_3row(self, real_event, event, pred_event, batch_idx=0):
        # real_event, event, pred_event: [B L C H W]
        H, W = event.shape[3:]
        ph, pw = 2, 3
        h, w = H//ph, W//pw   # event patch size: [c=2 ph pw]
        
        # Set figure size (row: 3, column: num_iterations)
        # num_iter = pred_indices.size(-1) - 1    # Except fist timestep; no comparable output
        # num_iter = event.size(1)
        num_iter = 10
        fig, axes = plt.subplots(3, num_iter, figsize=(num_iter * 2, 4))
        for l in range(num_iter):
            real = real_event[0,l,...]                  # [C H W]
            real_np = return_as_rb_img(real)            # batch size must be 1, np, (H W 3)
            
            input_ev = event[0,l,...]                   # [C H W]
            input_ev_np = return_as_rb_img(input_ev)    # batch size must be 1, np, (H W 3)
            
            pred_ev = pred_event[0,l,...]               # [C H W]
            pred_ev_np = return_as_rb_img(pred_ev)      # batch size must be 1, np, (H W 3)
            
            if not input_ev_np.shape == pred_ev_np.shape:
                pred_ev_np = pad_array_to_match(input_ev_np, pred_ev_np)
            assert input_ev_np.shape == pred_ev_np.shape
            
            pix_err = np.abs(real_np/255 - pred_ev_np/255).sum() / (2*H*W) * 100.
            axes[0, l].set_title(f'timestep={l}, real event', fontsize=10)  # Add title to the top row
            axes[0, l].imshow(real_np)
            axes[0, l].axis('off')
            axes[1, l].set_title(f'input', fontsize=10)
            axes[1, l].imshow(input_ev_np)
            axes[1, l].axis('off')
            axes[2, l].set_title(f'predicted ts={l+1}', fontsize=10)
            axes[2, l].set_title(f'predicted, pix_err: {pix_err:.2f}%', fontsize=10)
            axes[2, l].imshow(pred_ev_np)
            axes[2, l].axis('off')
        # Save figure
        plt.tight_layout()
        plt.savefig(f'sequence_vis_{batch_idx}.png')
        plt.clf()
        plt.close()