import torch
import torch.nn.functional as F


def encode_patches_to_vocab(data):
    # L, C, H, W = data.shape
    # h = H // 2
    # w = W // 3
    ph, pw = 2, 3   # patch height, patch width; [2 2 3]
    
    # Apply sliding window (stride=2 in height, stride=3 in width)
    patches = F.unfold(data, kernel_size=(ph,pw), stride=(ph,pw)).permute(0, 2, 1)  # [L, hw, 2*ph*pw]
    
    # Convert patches to binary strings and then to decimal indices, using GPU
    # Cast the patches to int and shift the bits to construct the binary representation
    powers_of_two = 2 ** torch.arange(patches.shape[2] - 1, -1, -1, device=patches.device)
    indices_tensor = (patches.int() * powers_of_two).sum(dim=2) # Binary to decimal conversion
    
    return indices_tensor