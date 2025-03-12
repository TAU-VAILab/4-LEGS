from argparse import ArgumentParser
import copy
import json
from pathlib import Path
import torch
from tqdm import tqdm
from feature_extraction.pyramid_data_loader import PyramidEmbeddingDataloader
from feature_extraction.viclip_encoder import VICLIPNetwork, VICLIPNetworkConfig
from torchvision import transforms as T
from PIL import Image
import os

transform = T.ToTensor()

FRAMES = 8

def create_init_viclip_features(sequence, timestep, encoder, args):
    md = json.load(open(f"./data/{sequence}/train_meta.json", 'r'))
    images = []
    for cam in range(len(md['fn'][timestep])):
        clip = []
        for i in range(-FRAMES//2,FRAMES//2, FRAMES//FRAMES):
            if timestep + i < 0 or timestep + i > len(md['fn']) - 1:
                    continue
            clip.append(transform(copy.deepcopy(Image.open(os.path.join(f"./data/{sequence}/ims", md['fn'][timestep+i][cam])))))
        while len(clip) < FRAMES:
            clip.append(clip[-1].clone().detach())       
        images.append(torch.stack(clip, dim=0))  
    images = torch.stack(images, dim=0)
    cache_dir = Path(f"./data/{sequence}/{args.features_dir}/viclip/timestep_{timestep}")
    torch.cuda.empty_cache()
    PyramidEmbeddingDataloader(
        image_list=images,
        device="cuda",
        cfg={
            "tile_size_range": [0.15, 0.6],
            "tile_size_res": 5,
            "stride_scaler": 0.5,
            "image_shape": list(images.shape[-2:]),
            "model_name": "viclip",
        },
        cache_path=cache_dir,
        model=encoder,
        )



if __name__ == "__main__":
    parser = ArgumentParser(description="Feature Extraction parameters")
    parser.add_argument("-s","--sequence", type=str, required=True)
    parser.add_argument("-f","--first_timestep", type=int, required=True)
    parser.add_argument("-l","--last_timestep", type=int, required=True)
    parser.add_argument("--features_dir", type=str, default="interpolators")
    args = parser.parse_args()
    encoder = VICLIPNetwork(VICLIPNetworkConfig)
    for i in tqdm(range(args.first_timestep,args.last_timestep), desc="TIMESTEPS"):
        torch.cuda.empty_cache()
        create_init_viclip_features(args.sequence, i,encoder, args)

