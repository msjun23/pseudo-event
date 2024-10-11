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
from dataset.provider import DatasetProvider
from dataset.sequential_provider import SequentialDatasetProvider
from models.ae import Event2VocabEncoder, Vocab2EventDecoder


class EventGenerator(pl.LightningModule):
    def __init__(self, cfg):
        super(EventGenerator, self).__init__()
        
        # Event and Vocab mapper
        self.event_to_vocab = Event2VocabEncoder(vocab_size=cfg.mamba.vocab_size)
        self.vocab_to_event = Vocab2EventDecoder()
        
        # Mamba Language Model
        mamba_cfg = MambaConfig(**cfg.mamba)
        self.mamba_lm = MambaLMHeadModel(mamba_cfg)
        
        self.cfg_dataset = cfg.dataset
        self.cfg_optim = cfg.optimizer
        
    def setup(self, stage=None):
        # At training
        if stage == 'fit' or stage is None:
            dataset_provider = DatasetProvider(**self.cfg_dataset.dataset)
            self.train_dataset = dataset_provider.get_train_dataset()
            
            self.ce_loss = nn.CrossEntropyLoss()
            self.l2_loss = nn.MSELoss()
            self.frame_loss_w = 100.
            
        # At inference
        if stage == 'test' or stage is None:
            seq_dataset_provider = SequentialDatasetProvider(**self.cfg_dataset.dataset)
            self.test_dataset = seq_dataset_provider.get_test_dataset()
                    
    def forward(self, x):
        # x: [B L C H W]
        # Event to Vocab; encoding
        x_vocab = self.event_to_vocab(x)            # [Bhw L]
        
        # Mamba LM
        logits = self.mamba_lm(x_vocab).logits      # [Bhw L] -> [Bhw L V]
        
        # The predicted next step's event vocab stream
        # with torch.no_grad():
        # Convert logits into a probability distribution using the softmax function
        probabilities = F.softmax(logits, dim=-1)
        # Select the index of the token with the highest probability
        predicted_indices = torch.argmax(probabilities, dim=-1)     # [Bhw L]
        
        # Vocab to Event; decoding
        predicted_frames = self.vocab_to_event(predicted_indices)       # [B L C H W]
        
        return x_vocab, logits, predicted_indices, predicted_frames
    
    def training_step(self, batch, batch_idx):
        seq_name = batch['sequence_name']
        file_index = batch['file_index']
        event = batch['event']['left']      # [B L C H W], B (batch size) must be '1'
        
        # [Bhw L] [Bhw L V]     [Bhw L]         [B L C H W]
        ev_vocab, logits, predicted_indices, predicted_frames = self(event)
            # input
        ev_vocab = ev_vocab[:,1:]
        gt_frame = event[:,1:,...]
            # output
        logits = logits[:,:-1,:]
        predicted_indices = predicted_indices[:,:-1]
        predicted_frames = predicted_frames[:,:-1,...]
        assert ev_vocab.shape[1] == gt_frame.shape[1] == logits.shape[1] == predicted_indices.shape[1] == predicted_frames.shape[1]
        '''
        input : a b c d e f g   <- can be used as both input and self_gt
                 \ \ \ \ \ \ \  <- prediction pairs
        output:   b c d e f g h <- prediction of next timestep
        Calculate loss using pairs in same timestep; b - g
        '''
        
        with torch.no_grad():
            self._show_sequences(event, predicted_frames)
        # # Get the current GPU memory usage
        # print(event.shape)
        # print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        # print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        # exit()
        
        # CE Loss using vocab
        ev_vocab = rearrange(ev_vocab, 'Bhw L -> (Bhw L)')
        logits = rearrange(logits, 'Bhw L V -> (Bhw L) V')
        _ce_loss = self.ce_loss(logits, ev_vocab)
        
        # L2 Loss using frame
        _l2_loss = self.l2_loss(predicted_frames, gt_frame)
        
        loss = _ce_loss + _l2_loss
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Save checkpoint every end of step
        if self.global_step % 1000 == 0:
            loss = outputs['loss'].item()
            checkpoint_path = f'checkpoints/step_{self.global_step}_loss_{loss:.4f}.ckpt'
            self.trainer.save_checkpoint(checkpoint_path)
        
    def test_step(self, batch, batch_idx):
        seq_name = batch['sequence_name']
        event = batch['event']      # [B L C H W], B (batch size) must be '1'
        
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
    def _show_sequences(self, ev_frame, pred_frame):
        # ev_frame: [B L C H W]
        # pred_frame: [B L C H W]
        B, L, C, H, W = ev_frame.shape
        
        # Set figure size (row: 2, column: num_iterations)
        num_iter = 10
        fig, axes = plt.subplots(2, num_iter, figsize=(num_iter * 2, 4))
        for l in range(num_iter):
            _l = l if l < 5 else l + 80
            input_ev = ev_frame[0,_l+1,...]              # [C H W]
            input_ev_np = return_as_rb_img(input_ev)    # batch size must be 1, np, (H W 3)
            
            pred_ev = pred_frame[0,_l,...]               # [C H W]
            pred_ev_np = return_as_rb_img(pred_ev)      # batch size must be 1, np, (H W 3)
            
            if not input_ev_np.shape == pred_ev_np.shape:
                pred_ev_np = pad_array_to_match(input_ev_np, pred_ev_np)
            assert input_ev_np.shape == pred_ev_np.shape
            
            pix_err = np.abs(input_ev_np/255 - pred_ev_np/255).sum() / (2*H*W) * 100.
            axes[0, l].set_title(f'timestep={_l+1}, input', fontsize=10)  # Add title to the top row
            axes[0, l].imshow(input_ev_np)
            axes[0, l].axis('off')
            axes[1, l].set_title(f'predicted, pix_err: {pix_err:.2f}%', fontsize=10)
            axes[1, l].imshow(pred_ev_np)
            # To debug
            # Comment out below two lines when you use for real
            diff_w_prev = np.abs(input_ev_np/255 - return_as_rb_img(ev_frame[0,_l,...])/255).sum() / (2*H*W) * 100
            axes[1, l].text(0.5, -0.1, f'diff_w_prev: {diff_w_prev:.2f}%', ha='center', transform=axes[0, l].transAxes, fontsize=8)
            axes[1, l].axis('off')
        # Save figure
        plt.tight_layout()
        plt.savefig('sequence_vis.png')
        plt.clf()
        plt.close()