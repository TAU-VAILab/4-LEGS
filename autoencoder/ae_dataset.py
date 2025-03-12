from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from feature_extraction.pyramid_data_loader import PyramidEmbeddingDataloader

ENCODER_FEATURE_DIM=768

class AE_dataset(Dataset):
    def __init__(self, large_idx, sequence, feature_extraction_model, loader_step_size, features_dir, train=True):
        self.image_height = 360
        self.image_width = 640
        self. interpolators = []
        self.large_idx = large_idx
        self.step = loader_step_size

        if train:
            for t in range(self.step-1):
                cache_dir = Path(f"./data/{sequence}/{features_dir}/{feature_extraction_model}/timestep_{t+(self.large_idx*self.step)}")
                interpolator = PyramidEmbeddingDataloader(
                    image_list=None,
                    device="cuda",
                    cfg={
                        "tile_size_range": [0.15, 0.6],
                        "tile_size_res": 5,
                        "stride_scaler": 0.5,
                        "image_shape": [self.image_height,self.image_width],
                        "model_name": feature_extraction_model,
                    },
                    cache_path=cache_dir,
                    model=None,
                    )
                self.interpolators.append(interpolator)

        else:
            t= self.step *(self.large_idx+1) -1
            cache_dir = Path(f"./data/{sequence}/{features_dir}/{feature_extraction_model}/timestep_{t}")
            interpolator = PyramidEmbeddingDataloader(
                image_list=None,
                device="cuda",
                cfg={
                    "tile_size_range": [0.15, 0.6],
                    "tile_size_res": 5,
                    "stride_scaler": 0.5,
                    "image_shape": [self.image_height,self.image_width],
                    "model_name": feature_extraction_model,
                },
                cache_path=cache_dir,
                model=None,
                )
            self.interpolators.append(interpolator)
        self.num_cams = self.interpolators[0].data_dict[0].data.shape[0]
        self.num_images = self.num_cams * len(self.interpolators)

    def __getitem__(self, idx):
        t = idx // self.num_cams
        index = idx % self.num_cams
        batch = torch.stack(torch.meshgrid(torch.arange(self.image_height), torch.arange(self.image_width)), dim=-1).reshape(-1,2).cpu()
        batch = torch.cat([(torch.zeros(batch.shape[0]) + index).reshape(-1,1).long(), batch], dim =-1).cpu()
        full = torch.zeros((self.image_height*self.image_width, ENCODER_FEATURE_DIM)).cuda()
        for k in self.interpolators[t].data_dict.keys():
            full += self.interpolators[t].data_dict[k](batch)        
        full /= len(self.interpolators[t].data_dict.keys())
        return full.cpu()

    def __len__(self):
        return self.num_images