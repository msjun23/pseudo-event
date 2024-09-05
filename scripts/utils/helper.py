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
        padding = (0, pad_h, 0, pad_w, 0, 0)
        padded_pred_tensor = torch.nn.functional.pad(pred_tensor, padding, mode='constant', value=0)
        
        # Reshape to (H W 3)
        padded_pred = padded_pred_tensor.numpy()
    else:
        padded_pred = pred
    
    return padded_pred


def pad_tensor_to_match(origin, pred):
    # [C H W] torch tensor input
    _, H, W = origin.shape
    _, h, w = pred.shape
    
    # Pad only when pred is smaller than origin
    if h < H or w < W:
        pad_h = H - h
        pad_w = W - w
        
        # Apply padding
        padding = (0, pad_w, 0, pad_h)
        padded_pred = torch.nn.functional.pad(pred, padding, mode='constant', value=0)
    else:
        padded_pred = pred
    
    return padded_pred

a = np.random.randn(480,640,3)
b = np.random.randn(480,639,3)
b = pad_array_to_match(a,b)
print(a.shape, b.shape)

c = torch.randn([3,480,640])
d = torch.randn([3,480,639])
d = pad_tensor_to_match(c,d)
print(c.shape, d.shape)