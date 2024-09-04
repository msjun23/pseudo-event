import torch
import torch.nn as nn

from einops import rearrange


def encode_patches_to_vocab(data):
    # Patch size is [2 2 3], [c h w]
    # _, H, W = data.shape
    # h = H // 2
    # w = W // 3
    
    # Apply sliding window (stride=2 in height, stride=3 in width)
    patches = data.unfold(1, 2, 2).unfold(2, 3, 3)  # [2, h, w, 2, 3]
    patches = rearrange(patches, 'c h w p_h p_w -> (h w) c p_h p_w')    # [h*w, 2, 2, 3]
    
    # Match each patch to an index in event_vocab
    indices = []
    for patch in patches:
        # Convert the tensor to integer (binary)
        binary_str = ''.join([str(int(x.item())) for x in patch.view(-1)])
        # Convert binary string to decimal
        idx = int(binary_str, 2)
        indices.append(idx)
        
    # Convert indices to a tensor
    indices_tensor = torch.tensor(indices)  # length: h x w
    
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