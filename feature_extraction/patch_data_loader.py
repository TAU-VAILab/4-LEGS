import json

import numpy as np
import torch
from feature_extraction.feature_data_loader import FeatureDataloader
from tqdm import tqdm

BATCH_SIZE = 20

class PatchEmbeddingDataloader(FeatureDataloader):
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
        assert "tile_ratio" in cfg
        assert "stride_ratio" in cfg
        assert "image_shape" in cfg
        assert "model_name" in cfg

        self.kernel_size = int(cfg["image_shape"][0] * cfg["tile_ratio"])
        self.stride = int(self.kernel_size * cfg["stride_ratio"])
        self.padding = self.kernel_size // 2
        self.center_x = (
            (self.kernel_size - 1) / 2
            - self.padding
            + self.stride
            * np.arange(
                np.floor((cfg["image_shape"][0] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
            )
        )
        self.center_y = (
            (self.kernel_size - 1) / 2
            - self.padding
            + self.stride
            * np.arange(
                np.floor((cfg["image_shape"][1] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
            )
        )
        self.center_x = torch.from_numpy(self.center_x)
        self.center_y = torch.from_numpy(self.center_y)
        self.start_x = self.center_x[0].float()
        self.start_y = self.center_y[0].float()

        self.model = model
        self.embed_size = self.model.embedding_dim if self.model is not None else embed_size
        self.multiview = multiview
        super().__init__(cfg, device, image_list, cache_path)

    def load(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        if not cache_info_path.exists():
            raise FileNotFoundError
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg and float('%.3f'%(cfg["tile_ratio"])) != float('%.3f'%(self.cfg["tile_ratio"])):
            raise ValueError("Config mismatch")
        self.data = torch.from_numpy(np.load(self.cache_path))

    def create(self, image_list):
        assert self.model is not None, "model must be provided to generate features"
        assert image_list is not None, "image_list must be provided to generate features"

        unfold_func = torch.nn.Unfold(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        ).to(self.device)

        img_embeds = []
        for idx, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):
            img_embeds.append(self._embed_clip_tiles(img, unfold_func))
        self.data = torch.from_numpy(np.stack(img_embeds))

    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y) (img_ind, row, col)
        # return: (B, 512)
        img_points = img_points.cpu()
        img_ind, img_points_x, img_points_y = img_points[:, 0], img_points[:, 1], img_points[:, 2]

        x_ind = torch.floor((img_points_x - (self.start_x)) / self.stride).long()
        y_ind = torch.floor((img_points_y - (self.start_y)) / self.stride).long()
        return self._interp_inds(img_ind, x_ind, y_ind, img_points_x, img_points_y)

    def _interp_inds(self, img_ind, x_ind, y_ind, img_points_x, img_points_y):
        img_ind = img_ind.to(self.data.device)  # self.data is on cpu to save gpu memory, hence this line
        topleft = self.data[img_ind, x_ind, y_ind].to(self.device)
        topright = self.data[img_ind, x_ind + 1, y_ind].to(self.device)
        botleft = self.data[img_ind, x_ind, y_ind + 1].to(self.device)
        botright = self.data[img_ind, x_ind + 1, y_ind + 1].to(self.device)

        x_stride = self.stride
        y_stride = self.stride
        right_w = ((img_points_x - (self.center_x[x_ind])) / x_stride).to(self.device).type(topleft.dtype)  
        top = torch.lerp(topleft, topright, right_w[:, None])
        bot = torch.lerp(botleft, botright, right_w[:, None])

        bot_w = ((img_points_y - (self.center_y[y_ind])) / y_stride).to(self.device).type(top.dtype) 
        return torch.lerp(top, bot, bot_w[:, None])

    def _embed_clip_tiles(self, image, unfold_func):
        aug_imgs = torch.stack([im for im in image])
        tiles = unfold_func(aug_imgs).permute(2, 0, 1).reshape(-1, aug_imgs.shape[0], 3, self.kernel_size, self.kernel_size).to("cuda")

        with torch.no_grad():
            torch.cuda.empty_cache()
            part_tiles = []
            batch_size = BATCH_SIZE
            steps = tiles.shape[0] // batch_size
            for i in range(steps):
                part_tiles.append(self.model.encode_video(tiles[i*batch_size:(i+1)*batch_size,...].cuda()))
            if steps*batch_size != tiles.shape[0]:
                part_tiles.append(self.model.encode_video(tiles[steps*batch_size:,...].cuda()))

            clip_embeds = torch.cat(part_tiles)

        clip_embeds = clip_embeds.reshape((self.center_x.shape[0], self.center_y.shape[0], -1))
        clip_embeds = torch.concat((clip_embeds, clip_embeds[:, [-1], :]), dim=1)
        clip_embeds = torch.concat((clip_embeds, clip_embeds[[-1], :, :]), dim=0)
        return clip_embeds.detach().cpu().numpy()