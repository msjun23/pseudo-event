import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from sklearn.cluster import KMeans

from spikingjelly.activation_based import layer, neuron, surrogate, functional

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# Encoder Block
class Event2VocabEncoder(nn.Module):
    def __init__(self, vocab_size=50277):
        super(Event2VocabEncoder, self).__init__()
        self.vocab_size = vocab_size
        
        self.conv = nn.Sequential(
            layer.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, step_mode='m'),
            layer.BatchNorm2d(32, step_mode='m'),
            nn.ReLU(),
            layer.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, step_mode='m'),
            layer.BatchNorm2d(64, step_mode='m'),
            nn.ReLU(),
            layer.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, step_mode='m'),
            layer.BatchNorm2d(128, step_mode='m'),
            # nn.ReLU(),
            layer.Conv2d(128, 1, kernel_size=3, stride=2, padding=1, step_mode='m'),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B L C H W]
        x_device = x.device # K-mean of scikit does not support GPU
        x = rearrange(x, 'B L C H W -> L B C H W')
        x = self.conv(x)  # Shape after conv layers: [L B c h w]
        x = rearrange(x, 'L B c h w -> (B c h w) L')    # [Bchw L]
        x = (self.vocab_size * x).to(torch.int64)
        # x = rearrange(x, 'L B c h w -> L (B h w) c')    # [L Bhw c]
        
        # # Feature space -> vocab label
        # x_vocabs = []
        # for l in range(x.shape[0]): # Iter for sequence length
        #     # K-mean Clustering
        #     # num of cluster == vocab_size
        #     kmeans = KMeans(n_clusters=self.vocab_size)
        #     # Input shape: [n_samples, n_features]
        #     kmeans.fit(x[l].detach().cpu().numpy())
        #     # Output shape: [n_samples]
        #     x_vocab = torch.tensor(kmeans.labels_, dtype=torch.int64)   # [Bhw]
        #     x_vocabs.append(x_vocab)
        # x_vocab_tensor = torch.stack(x_vocabs, dim=1).to(x_device)      # [Bhw, L]
        
        # return x_vocab_tensor   # Final shape: [Bhw L]
        return x

# Decoder Block
class Vocab2EventDecoder(nn.Module):
    def __init__(self, vocab_size=50277):
        super(Vocab2EventDecoder, self).__init__()
        self.vocab_size = vocab_size
        
        self.deconv = nn.Sequential(
            layer.ConvTranspose2d(1, 128, kernel_size=3, stride=2, padding=1, output_padding=1, step_mode='m'),
            layer.BatchNorm2d(128, step_mode='m'),
            nn.ReLU(),
            # neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m'),
            layer.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, step_mode='m'),
            layer.BatchNorm2d(64, step_mode='m'),
            nn.ReLU(),
            # neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m'),
            layer.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, step_mode='m'),
            layer.BatchNorm2d(32, step_mode='m'),
            nn.ReLU(),
            # neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m'),
            layer.ConvTranspose2d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=1, step_mode='m'),
            # nn.Sigmoid()  # Assuming input image is normalized between [0, 1]
            neuron.LIFNode(tau=1.1, v_threshold=1., v_reset=0., surrogate_function=surrogate.ATan(), step_mode='m'),
        )

    def forward(self, x):
        functional.reset_net(self.deconv)
        # x: [Bhw L]
        h = 20
        w = 30
        x = x.to(torch.float32) / self.vocab_size
        x = rearrange(x, '(B c h w) L -> L B c h w', B=1, c=1, h=h, w=w)    # [L B c h w]
        x = self.deconv(x)  # Shape after deconv layers: [L B C H W]
        x = rearrange(x, 'L B C H W -> B L C H W')
        return x  # Final shape: [B L C H W]