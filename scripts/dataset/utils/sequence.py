from pathlib import Path
import weakref

import cv2
import csv
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.utils.representations import VoxelGrid, OnOffFrame
from dataset.utils.eventslicer import EventSlicer


class Sequence(Dataset):
    # NOTE: This is just an EXAMPLE class for convenience. Adapt it to your case.
    # In this example, we use the voxel grid representation.
    #
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. zurich_city_11_a)
    # ├── disparity
    # │   ├── event
    # │   │   ├── 000000.png
    # │   │   └── ...
    # │   └── timestamps.txt
    # └── events
    #     ├── left
    #     │   ├── events.h5
    #     │   └── rectify_map.h5
    #     └── right
    #         ├── events.h5
    #         └── rectify_map.h5

    def __init__(self, seq_path: Path, mode: str='train', delta_t_ms: int=50, num_bins: int=10, 
                 representation: str='on_off', 
                 stereo: bool=True, disp_gt: bool=True):
        assert num_bins >= 1
        assert delta_t_ms <= 100, 'adapt this code, if duration is higher than 100 ms'
        assert seq_path.is_dir(), print(seq_path)

        # NOTE: Adapt this code according to the present mode (e.g. train, val or test).
        self.mode = mode

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins
        
        # Save sequence name
        self.seq_name = seq_path.stem

        # Set event representation
        if representation == 'voxel':
            self.event_repr = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)
        elif representation == 'on_off':
            self.event_repr = OnOffFrame(self.num_bins, self.height, self.width)
        else:
            raise NotImplementedError

        if stereo:
            self.locations = ['left', 'right']
        else:
            self.locations = ['left']

        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000

        # load disparity timestamps
        self.disp_gt = disp_gt
        if mode in ['train', 'validaion']:
            disp_dir = seq_path / 'disparity'
            assert disp_dir.is_dir()
            self.timestamps = np.loadtxt(disp_dir / 'timestamps.txt', dtype='int64')

            # load disparity paths
            ev_disp_dir = disp_dir / 'event'
            assert ev_disp_dir.is_dir()
            disp_gt_pathstrings = list()
            for entry in ev_disp_dir.iterdir():
                assert str(entry.name).endswith('.png')
                disp_gt_pathstrings.append(str(entry))
            disp_gt_pathstrings.sort()
            self.disp_gt_pathstrings = disp_gt_pathstrings

            assert len(self.disp_gt_pathstrings) == self.timestamps.size

            # Remove first disparity path and corresponding timestamp.
            # This is necessary because we do not have events before the first disparity map.
            assert int(Path(self.disp_gt_pathstrings[0]).stem) == 0
            self.disp_gt_pathstrings.pop(0)
            self.timestamps = self.timestamps[1:]
        elif self.mode == 'test':
            # load test events timestamps
            test_csv_path = seq_path / Path(seq_path.stem+'.csv')
            csv_f  = open(test_csv_path, 'r')
            next(csv.reader(csv_f))
            self.timestamps = np.array([])
            self.test_file_index = np.array([])
            for l in csv.reader(csv_f):
                self.timestamps = np.append(self.timestamps, np.array([int(l[0])], dtype='int64'), axis=0)
                self.test_file_index = np.append(self.test_file_index, np.array([int(l[1])], dtype='int32'), axis=0)

        self.h5f = dict()
        self.rectify_ev_maps = dict()
        self.event_slicers = dict()

        ev_dir = seq_path / 'events'
        for location in self.locations:
            ev_dir_location = ev_dir / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = ev_dir_location / 'rectify_map.h5'

            h5f_location = h5py.File(str(ev_data_file), 'r')
            self.h5f[location] = h5f_location
            self.event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]


        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

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

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32')/256

    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()

    def __len__(self):
        if self.mode in ['train', 'validation']:
            return len(self.disp_gt_pathstrings)
        else:
            return self.timestamps.size

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def __getitem__(self, index):
        ts_end = self.timestamps[index]
        # ts_start should be fine (within the window as we removed the first disparity map)
        ts_start = ts_end - self.delta_t_us

        if self.mode in ['train', 'validation']:
            disp_gt_path = Path(self.disp_gt_pathstrings[index])
            file_index = int(disp_gt_path.stem)
            if self.disp_gt:
                output = {
                    'sequence_name': self.seq_name,
                    'disparity_gt': self.get_disparity_map(disp_gt_path),
                    'file_index': file_index,
                }
            else:
                output = {
                    'sequence_name': self.seq_name,
                    'file_index': file_index,
                }
        elif self.mode in ['test']:
            # Test set do not have gt
            file_index = int(self.test_file_index[index])
            output = {
                'sequence_name': self.seq_name,
                'file_index': file_index,
            }
        
        for location in self.locations:
            event_data = self.event_slicers[location].get_events(ts_start, ts_end)

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            xy_rect = self.rectify_events(x, y, location)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            event_representation = self.events_to_representation(x_rect, y_rect, p, t)
            if 'event' not in output:
                output['event'] = dict()
            output['event'][location] = event_representation

        return output