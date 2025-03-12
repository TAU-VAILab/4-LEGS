import json
import os
from pathlib import Path

import numpy as np
import torch
from feature_extraction.feature_data_loader import FeatureDataloader
from feature_extraction.patch_data_loader import PatchEmbeddingDataloader
from tqdm import tqdm


class PyramidEmbeddingDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model,
        image_list: torch.Tensor = None,
        cache_path: str = None,
        embed_size = None,
        multiview = True
    ):
        assert "tile_size_range" in cfg
        assert "tile_size_res" in cfg
        assert "stride_scaler" in cfg
        assert "image_shape" in cfg
        assert "model_name" in cfg

        self.tile_sizes = torch.linspace(*cfg["tile_size_range"], cfg["tile_size_res"]).to(device)
        self.strider_scaler_list = [self._stride_scaler(tr.item(), cfg["stride_scaler"]) for tr in self.tile_sizes]

        self.model = model
        self.embed_size = self.model.embedding_dim if self.model is not None else embed_size
        self.multiview = multiview
        self.data_dict = {}
        super().__init__(cfg, device, image_list, cache_path)

    def __call__(self, img_points, scale):
        if scale is None:
            return self._random_scales(img_points)
        else:
            return self._uniform_scales(img_points, scale)

    def _stride_scaler(self, tile_ratio, stride_scaler):
        return np.interp(tile_ratio, [0.05, 0.15], [1.0, stride_scaler])

    def load(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        if not cache_info_path.exists():
            raise FileNotFoundError
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            raise ValueError("Config mismatch")

        raise FileNotFoundError  

    def create(self, image_list):
        os.makedirs(self.cache_path, exist_ok=True)
        for i, tr in enumerate(self.tile_sizes):
            stride_scaler = self.strider_scaler_list[i]
            self.data_dict[i] = PatchEmbeddingDataloader(
                cfg={
                    "tile_ratio": tr.item(),
                    "stride_ratio": stride_scaler,
                    "image_shape": self.cfg["image_shape"],
                    "model_name": self.cfg["model_name"],
                },
                device=self.device,
                model=self.model,
                image_list=image_list,
                cache_path=Path(f"{self.cache_path}/level_{i}.npy"),
                embed_size=self.embed_size,
                multiview = self.multiview,
            )

    def save(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        with open(cache_info_path, "w") as f:
            f.write(json.dumps(self.cfg))
        pass

    def _random_scales(self, img_points):
        img_points = img_points.to(self.device)
        random_scale_bin = torch.randint(self.tile_sizes.shape[0] - 1, size=(img_points.shape[0],), device=self.device)
        random_scale_weight = torch.rand(img_points.shape[0], device=self.device)

        stepsize = (self.tile_sizes[1] - self.tile_sizes[0]) / (self.tile_sizes[-1] - self.tile_sizes[0])

        bottom_interp = torch.zeros((img_points.shape[0], self.embed_size), device=self.device)
        top_interp = torch.zeros((img_points.shape[0], self.embed_size), device=self.device)

        for i in range(len(self.tile_sizes) - 1):
            ids = img_points[random_scale_bin == i]
            bottom_interp[random_scale_bin == i] = self.data_dict[i](ids)
            top_interp[random_scale_bin == i] = self.data_dict[i + 1](ids)

        return (
            torch.lerp(bottom_interp, top_interp, random_scale_weight[..., None]),
            (random_scale_bin * stepsize + random_scale_weight * stepsize)[..., None],
        )

    def _uniform_scales(self, img_points, scale):
        scale_bin = torch.floor(
            (scale - self.tile_sizes[0]) / (self.tile_sizes[-1] - self.tile_sizes[0]) * (self.tile_sizes.shape[0] - 1)
        ).to(torch.int64)
        scale_weight = (scale - self.tile_sizes[scale_bin]) / (
            self.tile_sizes[scale_bin + 1] - self.tile_sizes[scale_bin]
        )
        interp_lst = torch.stack([interp(img_points) for interp in self.data_dict.values()])
        point_inds = torch.arange(img_points.shape[0])
        interp = torch.lerp(
            interp_lst[scale_bin, point_inds],
            interp_lst[scale_bin + 1, point_inds],
            torch.Tensor([scale_weight]).to(self.device)[..., None],
        )
        return interp / interp.norm(dim=-1, keepdim=True), scale