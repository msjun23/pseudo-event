from pathlib import Path

import torch
import torch.utils.data

from dataset.constant import DATA_SPLIT
from dataset.utils.sequence import Sequence


class DatasetProvider:
    def __init__(self, dataset_path: str, representation='on_off', delta_t_ms: int=50, num_bins=10, 
                 stereo=True, disp_gt=True, 
                 crop_height=480, crop_width=640, pad_height=480, pad_width=640):
        self.dataset_path = Path(dataset_path)
        assert self.dataset_path.is_dir(), str(dataset_path)
        
        self.representation = representation
        self.delta_t_ms = delta_t_ms
        self.num_bins = num_bins
        
        self.stereo = stereo
        self.disp_gt = disp_gt
        
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.pad_height = pad_height
        self.pad_width = pad_width
        
    def get_train_dataset(self):
        train_sequences = list()
        # for child in train_path.iterdir():
        for seq in DATA_SPLIT['train']:
            child = self.dataset_path / seq
            if not child.is_dir():
                continue
            train_sequences.append(Sequence(child, 'train', self.delta_t_ms, self.num_bins, 
                                            self.representation, 
                                            self.stereo, self.disp_gt, 
                                            self.crop_height, self.crop_width))
            
        self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)
        return self.train_dataset
    
    def get_val_dataset(self):
        # Implement this according to your needs.
        raise NotImplementedError
    
    def get_test_dataset(self):
        test_sequences = list()
        # for child in test_path.iterdir():
        for seq in DATA_SPLIT['test']:
            child = self.dataset_path / seq
            if not child.is_dir():
                continue
            test_sequences.append(Sequence(child, 'test', self.delta_t_ms, self.num_bins, 
                                           self.representation, 
                                           self.stereo, self.disp_gt))
            
        self.test_dataset = torch.utils.data.ConcatDataset(test_sequences)
        return self.test_dataset