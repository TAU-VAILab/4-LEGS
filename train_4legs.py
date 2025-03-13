from argparse import ArgumentParser
import copy
import json
from pathlib import Path

import imageio
import numpy as np
import torch
from tqdm import tqdm

from attention.attention import LanguageFeatureAttention
from feature_extraction.pyramid_data_loader import PyramidEmbeddingDataloader
from torchvision import transforms as T
from PIL import Image
import os
from diff_gaussian_rasterization_feat import GaussianRasterizer as Renderer
from helpers import apply_colormap, l1_loss_v1, o3d_knn,  setup_camera
from autoencoder.autoencoder import Autoencoder
from train_d3dgs import get_batch
from feature_extraction.viclip_encoder import VICLIPNetwork, VICLIPNetworkConfig

LEGS_FEATURE_DIM = 128
ENCODER_FEATURE_DIM =768

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)    

def get_gt_features(sequence, w,h, segs, t, args):
    ckpt_path = f"./data/{sequence}/{args.autoencoder_dir}/best_ckpt.pth"
    checkpoint = torch.load(ckpt_path)
    ae = Autoencoder().cuda()
    ae.load_state_dict(checkpoint)
    ae.eval()

    cache_dir = Path(f"./data/{sequence}/{args.features_dir}/viclip/timestep_{t}")
    interpolator = PyramidEmbeddingDataloader(
        image_list=None,
        device="cuda",
        cfg={
            "tile_size_range": [0.15, 0.6],
            "tile_size_res": 5,
            "stride_scaler": 0.5,
            "image_shape": [h,w],
            "model_name": "viclip",
        },
        cache_path=cache_dir,
        model=None,
    )
    features = []
    for index in range(len(segs)):
        batch = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), dim=-1).reshape(-1,2).cpu()
        batch = torch.cat([(torch.zeros(batch.shape[0]) + index).reshape(-1,1).long(), batch], dim =-1).cpu()
        full = torch.zeros((h*w, ENCODER_FEATURE_DIM)).cuda()
        for k in interpolator.data_dict.keys():
            full += interpolator.data_dict[k](batch)
        full /= len(interpolator.data_dict.keys())
        with torch.no_grad():
            full = ae.encode(full).permute(1,0)
        features.append(full.reshape(LEGS_FEATURE_DIM,h,w).cpu())        
    return features

def get_dataset(t, md, seq, args):    
    segs = [np.array(copy.deepcopy(Image.open(f"./data/{seq}/seg/{ md['fn'][t][c].replace('.jpg', '.png')}"))).astype(np.float32) for c in range(len(md['fn'][t]))]
    segs = np.arange(1)
    features = get_gt_features(seq, md['w'], md['h'], segs, t, args)
    dataset = []
    for c in range(len(segs)):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        feat = features[c]
        dataset.append({'cam': cam, 'feat': feat, 'id': c})
        
    return dataset

def load_pre_scene_data(sequence,first_timestep,last_timestep, output_dir):
    params = dict(np.load(f"./{output_dir}/pretrained/{sequence}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    pretrained_scene_data= []
    for t in range(first_timestep,last_timestep):
        _, neighbor_indices = o3d_knn(params['means3D'][t].detach().cpu().numpy(), 20)
        indices = np.arange(len(neighbor_indices)).reshape(len(neighbor_indices), 1)
        neighbor_indices = np.hstack((indices, neighbor_indices)).astype(np.int32)
        pretrained_scene_data.append({
                'means3D': params['means3D'][t],
                'colors_precomp': params['rgb_colors'][t],
                'seg_precomp':params['seg_colors'],
                'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
                'opacities': torch.sigmoid(params['logit_opacities']),
                'scales': torch.exp(params['log_scales']),
                'means2D': torch.zeros_like(params['means3D'][0], device="cuda"),
                "neighbor_indices": torch.Tensor(neighbor_indices).long().cuda()
            })
        
    return pretrained_scene_data

def initialize_params_and_optimizer(pretrained_scene_data, args):
    attn = LanguageFeatureAttention(LEGS_FEATURE_DIM).cuda()
    params = {
        'legs_features': np.zeros((pretrained_scene_data["means3D"].shape[0], LEGS_FEATURE_DIM)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    lrs = {
        'legs_features':args.legs_lr,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    param_groups.append({"params":list(attn.parameters()), "name":"attn", "lr": args.attn_lr})
    return params, torch.optim.Adam(param_groups, lr=0.0, eps=1e-15), attn

def init_per_timestep(pretrained_scene_data):
    new_params = {'legs_features': np.zeros((pretrained_scene_data["means3D"].shape[0], LEGS_FEATURE_DIM)),}
    new_params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              new_params.items()}
    return new_params

def params2rendervar(pretrained_scene_data, feats):
    rendervar = {k:v for k,v in pretrained_scene_data.items() if k not in ["seg_precomp", "neighbor_indices"]}
    rendervar["legs_features"] = feats
    return rendervar

def get_loss(params,attn, curr_data, pretrained_scene_data):
    losses = {}
    torch.cuda.empty_cache()
    feats = attn(params["legs_features"],pretrained_scene_data["neighbor_indices"])   
    torch.cuda.empty_cache() 
    rendervar = params2rendervar(pretrained_scene_data, feats)
    _, _, _, feature_legs = Renderer(raster_settings=curr_data['cam'])(**rendervar)    
    gt_feat = curr_data['feat'].cuda()
    losses['legs_features'] = l1_loss_v1(feature_legs, gt_feat)
    loss_weights = {'legs_features': 1.0}
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    return loss

def params2save(params, neighbors):
    to_save = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}
    to_save["neighbor_indices"] = neighbors.detach().cpu().contiguous().numpy()
    return to_save

def save_output(output_params, seq, exp, t, attn, args):
    to_save = {}
    for k in output_params[0].keys():
        to_save[k] = np.stack([params[k] for params in output_params])
    np.savez(f"./{args.output_dir}/{exp}/{seq}/params_{t}", **to_save)
    torch.save(attn.state_dict(), f"./{args.output_dir}/{exp}/{seq}/attn_{t}.pth")

def run_timestep(t, md, sequence, pretrained_scene_data, progress_bar, args):
    output_params = []
    dataset = get_dataset(t, md, sequence, args)
    todo_dataset = []
    num_iter_per_timestep = args.num_iter_per_timestep
    params, optimizer, attn = initialize_params_and_optimizer(pretrained_scene_data, args)
    for i in range(num_iter_per_timestep):
        torch.cuda.empty_cache()
        curr_data = get_batch(todo_dataset, dataset)
        torch.cuda.empty_cache()
        loss =run_iter(params, attn, curr_data, pretrained_scene_data, optimizer)
    progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})    
    output_params.append(params2save(params, pretrained_scene_data["neighbor_indices"]))
    return output_params, attn

def run_iter(params, attn, curr_data, pretrained_scene_data, optimizer):
    loss = get_loss(params,attn,  curr_data, pretrained_scene_data)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss

def train_timestep(sequence, exp_name, md, t, pretrained_scene_data, progress_bar, args):   
    torch.cuda.empty_cache()
    output_params, attn =run_timestep(t, md, sequence, pretrained_scene_data, progress_bar, args) 
    progress_bar.update(1)
    save_output(output_params, sequence, exp_name, t, attn, args)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train args")
    parser.add_argument("-s","--sequence", type=str, required=True)
    parser.add_argument("-e", "--exp_name", type=str, required=True)
    parser.add_argument("-f","--first_timestep", type=int, required=True)
    parser.add_argument("-l","--last_timestep", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--num_iter_per_timestep", type=int, default=2000)
    parser.add_argument("--autoencoder_dir", type=str, default="ae")
    parser.add_argument("--features_dir", type=str, default="interpolators")
    parser.add_argument("--legs_lr", type=float, default=0.0025)
    parser.add_argument("--attn_lr", type=float, default=0.00016)
    args = parser.parse_args()
    exp_name = args.exp_name
    sequence = args.sequence
    first_timestep = args.first_timestep
    last_timestep = args.last_timestep
    torch.cuda.empty_cache()
    exp_dir = f"./{args.output_dir}/{exp_name}/{sequence}"
    os.makedirs(exp_dir, exist_ok=True)    
    md = json.load(open(f"./data/{sequence}/train_meta.json", 'r'))
    pretrained_scene_data = load_pre_scene_data(sequence, first_timestep, last_timestep, args.output_dir)
    torch.cuda.empty_cache()
    progress_bar = tqdm(range(first_timestep, last_timestep), desc=f"TIMESTEPS")
    for t in range(first_timestep, last_timestep):
        train_timestep(sequence, exp_name,md, t, pretrained_scene_data[t-first_timestep],progress_bar, args)
    progress_bar.close()

