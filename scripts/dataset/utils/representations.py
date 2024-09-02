import torch


class EventRepresentation:
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool):
        self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.nb_channels = channels
        self.normalize = normalize

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(pol.device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = time
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()

            value = 2*pol-1

            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid

class OnOffFrame(EventRepresentation):
    def __init__(self, num_bins: int, height: int, width: int):
        self.on_off_frame = torch.zeros((num_bins, 2, height, width), dtype=torch.float, requires_grad=False)
        self.num_bins = num_bins
        
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        # NOTE: shape of (num_bins, 2, Height, Width)
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1
        
        L, C, H, W = self.on_off_frame.shape
        with torch.no_grad():
            self.on_off_frame = self.on_off_frame.to(pol.device)
            on_off_frame = self.on_off_frame.clone()
            
            t_norm = time
            t_norm = (L - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])
            
            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()
            
            for xlim in [x0, x0+1]:
                for ylim in [y0, y0+1]:
                    for tlim in [t0, t0+1]:
                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.num_bins)
                        
                        index = C * H * W * tlim.long() + \
                                H * W * pol.long() + \
                                W * ylim.long() + \
                                xlim.long()
                                
                        one_values = torch.ones_like(index, dtype=torch.float)
                        on_off_frame.put_(index[mask], one_values[mask], accumulate=False)
                        
        return on_off_frame