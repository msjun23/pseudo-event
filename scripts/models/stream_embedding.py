import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


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


# # To implement convolutional operation to 5D tensor
# from spikingjelly.activation_based import layer

# class EventStreamEmbedding(nn.Module):
#     def __init__(self, in_channels=2, out_channels=2, kernel_size=3, stride=2, padding=1, step_mode='m'):
#         super(EventStreamEmbedding, self).__init__()
#         self.downsample_conv1 = layer.Conv2d(in_channels=in_channels, 
#                                              out_channels=out_channels, 
#                                              kernel_size=kernel_size, 
#                                              stride=stride, 
#                                              padding=padding, 
#                                              step_mode=step_mode)
#         self.downsample_conv2 = layer.Conv2d(in_channels=out_channels, 
#                                              out_channels=out_channels, 
#                                              kernel_size=kernel_size, 
#                                              stride=stride, 
#                                              padding=padding, 
#                                              step_mode=step_mode)
#         self.downsample_conv3 = layer.Conv2d(in_channels=out_channels, 
#                                              out_channels=out_channels, 
#                                              kernel_size=kernel_size, 
#                                              stride=stride, 
#                                              padding=padding, 
#                                              step_mode=step_mode)
#         self.downsample_conv4 = layer.Conv2d(in_channels=out_channels, 
#                                              out_channels=out_channels, 
#                                              kernel_size=kernel_size, 
#                                              stride=stride, 
#                                              padding=padding, 
#                                              step_mode=step_mode)
        
#         self.conv1x1 = nn.Conv1d(in_channels=out_channels, out_channels=1, kernel_size=1)
        
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.Conv3d)):
#                 nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
        
#     def forward(self, x):
#         # x: [B, L, C, H, W]
#         x = rearrange(x, 'B L C H W -> L B C H W')  # [L B C H W]
#         x = self.downsample_conv1(x)
#         x = self.downsample_conv2(x)
#         x = self.downsample_conv3(x)    
#         x = self.downsample_conv4(x)                # [L B C h w]
#         x = rearrange(x, 'L B C h w -> (B h w) C L')# [Bhw C L]
#         x = self.conv1x1(x).squeeze(dim=1)          # [Bhw L]
        
#         return x