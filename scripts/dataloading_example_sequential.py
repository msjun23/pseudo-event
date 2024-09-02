from pathlib import Path
import matplotlib.pyplot as plt

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.sequential_provider import SequentialDatasetProvider
from dataset.utils.visualization import disp_img_to_rgb_img, show_disp_overlay, show_image


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsec_dir', default='/root/data/DSEC', help='Path to DSEC dataset directory')
    parser.add_argument('--save_event_repr', action='store_true', help='Visualize and save event representation as image file')
    parser.add_argument('--visualize', action='store_true', help='Visualize data')
    parser.add_argument('--overlay', action='store_true', help='If visualizing, overlay disparity and voxel grid image')
    args = parser.parse_args()

    visualize = args.visualize
    dsec_dir = Path(args.dsec_dir)
    assert dsec_dir.is_dir()

    dataset_provider = SequentialDatasetProvider(dsec_dir, 'on_off', 100, 10)
    train_dataset = dataset_provider.get_train_dataset()
    test_dataset = dataset_provider.get_test_dataset()

    batch_size = 1
    num_workers = 0
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False)
    with torch.no_grad():
        for data in tqdm(train_loader):
            seq_name = data['sequence_name'][0]
            # file_index = f"{data['file_index'][0]:06}"
            event = data['event']['left']
            # gt = data['disparity_gt']
            
            print(seq_name)
            print(event.shape)
            # print(gt.shape)
            
            # print(event.unique())
            
            # if args.save_event_repr:
            #     # Save directory
            #     save_dir = f'save/event_representation/{seq_name}/'
            #     os.makedirs(save_dir, exist_ok=True)
                
            #     # Separate even and odd channels
            #     event = event.squeeze(0)        # [B L C H W] -> [L C H W], (B=1)
            #     pos_channels = event[:, 0, :, :]    # [L H W]
            #     neg_channels = event[:, 1, :, :]    # [L H W]
                
            #     # Summing up positive channel data (red channel)
            #     red_channel = pos_channels.sum(dim=0)   # [H W]
            #     # Summing up negative channel data (blue channel)
            #     blue_channel = neg_channels.sum(dim=0)  # [H W]
                
            #     # Create RGB Image
            #     rgb_image = torch.zeros((480, 640, 3))
            #     rgb_image[:, :, 0] = red_channel  # Red channel
            #     rgb_image[:, :, 2] = blue_channel # Blue channel
                
            #     # Visualize and save as a file
            #     plt.imshow(rgb_image.numpy())
            #     plt.axis('off')
            #     plt.savefig(f'{save_dir}{file_index}.png', bbox_inches='tight', pad_inches=0)
            #     plt.close()