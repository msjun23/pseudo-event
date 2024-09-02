import h5py
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from dataset.constant import SEQ_DATA_SPLIT
from dataset.utils.representations import VoxelGrid, OnOffFrame
from dataset.utils.eventslicer import EventSlicer
from dataset.visualization.eventreader import EventReaderAbstract


class SequentialEventReader(EventReaderAbstract):
    def __init__(self, filepath: Path, dt_milliseconds: int, num_bins: int):
        super().__init__(filepath)
        self.event_slicer = EventSlicer(self.h5f)
        
        self.dt_us = int(dt_milliseconds * 1000)
        self.t_start_us = self.event_slicer.get_start_time_us()
        self.t_end_us = self.event_slicer.get_final_time_us()
        self.num_bins = num_bins
        
        self._length = ((self.t_end_us - self.t_start_us) - self.dt_us) // \
                        (self.dt_us // self.num_bins) + 1
        self.timestamps = []
        
    def __len__(self):
        return self._length
    
    def __next__(self):
        '''
        Full iteration should be done for initialization
        '''
        t_end_us = self.t_start_us + self.dt_us
        if t_end_us > self.t_end_us:
            raise StopIteration
        events = self.event_slicer.get_events(self.t_start_us, t_end_us)
        if events is None:
            raise StopIteration
        
        self.t_start_us = self.t_start_us + (self.dt_us // self.num_bins)
        self.timestamps.append(self.t_start_us)
        return events
    
    def __getitem__(self, idx):
        t_start_us = self.timestamps[idx]
        t_end_us = t_start_us + self.dt_us
        events = self.event_slicer.get_events(t_start_us, t_end_us)
        return events
    
    
class SequentialDataset(Dataset):
    def __init__(self, seq_path: Path, delta_t_ms: int=100, num_bins: int=10, 
                 representation: str='on_off'):
        assert seq_path.is_dir(), print(seq_path)
        
        self.height = 480
        self.width = 640
        self.seq_name = seq_path.stem
        
        # Set event representation
        if representation == 'voxel':
            self.event_repr = VoxelGrid(num_bins, self.height, self.width, normalize=True)
        elif representation == 'on_off':
            self.event_repr = OnOffFrame(num_bins, self.height, self.width)
        else:
            raise NotImplementedError
        
        # Load event and rectification map data
        ev_data_file = seq_path / 'events/left/events.h5'
        ev_rect_file = seq_path / 'events/left/rectify_map.h5'
        with h5py.File(str(ev_rect_file), 'r') as h5_rect:
            self.rectify_ev_map = h5_rect['rectify_map'][()]
            
        self.event_reader = SequentialEventReader(ev_data_file, delta_t_ms, num_bins)
        print(f'\t # {self.seq_name}, Sequential event sequene initializing...')
        for _ in tqdm(self.event_reader):
            pass
            
    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        assert self.rectify_ev_map.shape == (self.height, self.width, 2), self.rectify_ev_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return self.rectify_ev_map[y, x]
    
    def events_to_representation(self, x, y, p, t, device: str='cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return self.event_repr.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))
        
    def __len__(self):
        return len(self.event_reader)

    def __getitem__(self, idx):
        output = dict()
        output['sequence_name'] = self.seq_name
        
        events = self.event_reader[idx]
        p = events['p'] # np.ndarray
        x = events['x']
        y = events['y']
        t = events['t']
        
        xy_rect = self.rectify_events(x, y)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]
        output['event'] = self.events_to_representation(x_rect, y_rect, p, t)
        return output
    
    
class SequentialDatasetProvider:
    def __init__(self, dataset_path: str, representation='on_off', delta_t_ms: int=50, num_bins=10, 
                 stereo=True, disp_gt=True, 
                 crop_height=480, crop_width=640, pad_height=480, pad_width=640):
        self.dataset_path = Path(dataset_path)
        assert self.dataset_path.is_dir(), str(self.dataset_path)
        
        self.representation = representation
        self.delta_t_ms = delta_t_ms
        self.num_bins = num_bins
        
    def get_train_dataset(self):
        train_sequence = list()
        for seq in SEQ_DATA_SPLIT['train']:
            child = self.dataset_path / seq
            if not child.is_dir():
                continue
            train_sequence.append(SequentialDataset(child, self.delta_t_ms, self.num_bins, self.representation))
        return torch.utils.data.ConcatDataset(train_sequence)
    
    def get_val_dataset(self):
        # Implement this according to your needs.
        raise NotImplementedError
    
    def get_test_dataset(self):
        test_sequence = list()
        for seq in SEQ_DATA_SPLIT['test']:
            child = self.dataset_path / seq
            if not child.is_dir():
                continue
            test_sequence.append(SequentialDataset(child, self.delta_t_ms, self.num_bins, self.representation))
        return torch.utils.data.ConcatDataset(test_sequence)