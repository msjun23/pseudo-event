import torch
import itertools
from einops import rearrange

from scripts.utils.rb_image import save_as_rb_img

# Create all possible 2x2x3 binary tensor combinations
all_combinations = list(itertools.product([0, 1], repeat=12))

# Create and Initialize event_vocab
event_vocab = []

# Convert each combination to a 2x2x3 tensor and add it to event_vocab
for combination in all_combinations:
    tensor = torch.tensor(combination, dtype=torch.int).reshape(2, 2, 3)
    event_vocab.append(tensor)

# Save event_vocab as torch tensor
event_vocab = torch.stack(event_vocab)
torch.save(event_vocab, 'event_vocab.pt')


# Example usage
# def match_patches_to_vocab(event_vocab, data):
#     # Patch size is [2 2 3], [c h w]
#     _, H, W = data.shape
#     h = H // 2
#     w = W // 3
#     # Visualization to compare
#     save_as_rb_img(data, 'ori.png')
    
#     # Apply sliding window (stride=2 in height, stride=3 in width)
#     patches = data.unfold(1, 2, 2).unfold(2, 3, 3)  # [2, h, w, 2, 3]
#     patches = rearrange(patches, 'c h w p_h p_w -> (h w) c p_h p_w')    # [h*w, 2, 2, 3]
    
#     # Match each patch to an index in event_vocab
#     indices = []
#     for patch in patches:
#         # Find the index in event_vocab that matches the patch
#         matches = (event_vocab == patch).all(dim=1).all(dim=1).all(dim=1)
#         idx = matches.nonzero(as_tuple=True)[0].item()
#         indices.append(idx)
        
#     # Convert indices to a tensor
#     indices_tensor = torch.tensor(indices)  # length: h x w
    
#     # Visualization to compare
#     # Restore original image from patches
#     reconstructed_patches = event_vocab[indices_tensor]  # [h*w, 2, 2, 3]
#     reconstructed_patches = reconstructed_patches.view(h, w, 2, 2, 3)  # [h, w, 2, 2, 3]
#     reconstructed_tensor = reconstructed_patches.permute(2, 0, 3, 1, 4).contiguous().view(2, h*2, w*3)
#     save_as_rb_img(reconstructed_tensor, 'reconstructed_image.png')
    
#     return indices_tensor

# event_vocab = torch.load('event_vocab.pt')
# data_tensor = torch.randint(0, 2, (2, 480, 640), dtype=torch.int)
# result = match_patches_to_vocab(event_vocab, data_tensor)
# print(result.shape)