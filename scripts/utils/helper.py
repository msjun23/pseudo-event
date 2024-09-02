import numpy as np
import torch

def pad_array_to_match(origin, pred):
    # (H W C) numpy input
    H, W, _ = origin.shape
    h, w, _ = pred.shape
    
    # Pad only when pred is smaller than origin
    if h < H or w < W:
        pad_h = H - h
        pad_w = W - w
        
        # Convert to torch tensor to apply padding
        pred_tensor = torch.tensor(pred)
        padding = (0, 0, 0, pad_w, 0, pad_h)
        padded_pred_tensor = torch.nn.functional.pad(pred_tensor.permute(2, 0, 1), padding, mode='constant', value=0)
        
        # Reshape to (H W 3)
        padded_pred = padded_pred_tensor.permute(1, 2, 0).numpy()
    else:
        padded_pred = pred
    
    return padded_pred