from pathlib import Path

import torch
import torch.utils.data

from dataset.constant import DATA_SPLIT
from dataset.utils.sequence import Sequence


class DatasetProvider:
    def __init__(self, dataset_path: str, representation='on_off', delta_t_ms: int=50, num_bins=10, 
                 stereo=True, disp_gt=True, 
                 crop_height=480, crop_width=640, pad_height=480, pad_width=640):
        dataset_path = Path(dataset_path)
        assert dataset_path.is_dir(), str(dataset_path)
        
        train_sequences = list()
        # for child in train_path.iterdir():
        for seq in DATA_SPLIT['train']:
            child = dataset_path / seq
            if not child.is_dir():
                continue
            train_sequences.append(Sequence(child, 'train', delta_t_ms, num_bins, 
                                            representation, 
                                            stereo, disp_gt))
            
        test_sequences = list()
        # for child in test_path.iterdir():
        for seq in DATA_SPLIT['test']:
            child = dataset_path / seq
            if not child.is_dir():
                continue
            test_sequences.append(Sequence(child, 'test', delta_t_ms, num_bins, 
                                           representation, 
                                           stereo, disp_gt))
            
        self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)
        self.test_dataset = torch.utils.data.ConcatDataset(test_sequences)

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        # Implement this according to your needs.
        raise NotImplementedError

    def get_test_dataset(self):
        return self.test_dataset
