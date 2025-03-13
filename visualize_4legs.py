from argparse import ArgumentParser
import json
import imageio
import numpy as np
import torch
from tqdm import tqdm
import os
from diff_gaussian_rasterization_feat import GaussianRasterizer as Renderer
from attention.attention import LanguageFeatureAttention
from helpers import apply_colormap, setup_camera
from autoencoder.autoencoder import Autoencoder
from feature_extraction.viclip_encoder import VICLIPNetwork, VICLIPNetworkConfig
from PIL import Image

LEGS_FEATURE_DIM = 128
ENCODER_FEATURE_DIM =768
w, h = 640, 360
near, far = 0.01, 100.0

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)    

REMOVE_BACKGROUND = False

def init_camera(y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k


def load_scene_data(seq, args):
    pre_params = dict(np.load(f"./{args.pre_output_dir}/{args.pre_exp_name}/{seq}/params.npz"))
    ckpt_path = f"./data/{sequence}/{args.autoencoder_dir}/best_ckpt.pth"
    checkpoint = torch.load(ckpt_path)
    decoder = Autoencoder().cuda()
    decoder.load_state_dict(checkpoint)
    decoder.eval()
    return pre_params, decoder

def render_prob_sim(w2c, k, data, decoder, vlm_encoder,t):
    md = json.load(open(f"./data/{sequence}/train_meta.json", 'r'))
    w, h, k, w2c = md['w'], md['h'], md['k'][t][0], md['w2c'][t][0]
    cam = setup_camera(w, h, k, w2c, near, far)
    render_data = {k:v for k,v in data.items() if k != "seg_precomp"}
    im, _, _,feature_legs = Renderer(raster_settings=cam)(**render_data)
    render_data["colors_precomp"] = data["seg_precomp"]
    seg, _, _,_ = Renderer(raster_settings=cam)(**render_data)
    seg = seg.permute(1,2,0)[:,:,0].unsqueeze(-1)
    seg[seg>0.5] = 1
    seg[seg<0.5] = 0
    fl = feature_legs.permute(1,2,0).reshape(-1,LEGS_FEATURE_DIM)
    feature_legs = decoder.decode(fl / fl.norm(dim=-1,keepdim=True))
    probs, _ = vlm_encoder.get_relevancy(feature_legs, 0, ret_both=True)
    probs = probs.reshape((im.shape[1], im.shape[2], -1)) 
    return probs.cpu(), seg.cpu(), im.permute(1,2,0).cpu()


def get_render_var(pre_params, legs_param_dir, t):
    legs_params = dict(np.load(f"{legs_param_dir}/params_{t}.npz"))
    ckpt_path = f"{legs_param_dir}/attn_{t}.pth"
    checkpoint = torch.load(ckpt_path)
    attn = LanguageFeatureAttention(LEGS_FEATURE_DIM).cuda()
    attn.load_state_dict(checkpoint)
    attn.eval()
    feats = attn(torch.tensor(legs_params["legs_features"][0]).cuda().float(), torch.tensor(legs_params["neighbor_indices"][0]).cuda().long())
    rendervar = {
        'means3D': torch.tensor(pre_params['means3D'][t]).cuda().float(),
        'colors_precomp': torch.tensor(pre_params['rgb_colors'][t]).cuda().float(),
        'seg_precomp':torch.tensor(pre_params['seg_colors']).cuda().float(),
        'rotations': torch.nn.functional.normalize(torch.tensor(pre_params['unnorm_rotations'][t]).cuda().float()),
        'opacities': torch.sigmoid(torch.tensor(pre_params['logit_opacities']).cuda().float()),
        'scales': torch.exp(torch.tensor(pre_params['log_scales']).cuda().float()),
        'means2D': torch.zeros_like(torch.tensor(pre_params['means3D'][0], device="cuda")).cuda().float(),
        "legs_features": feats,
    }
    if REMOVE_BACKGROUND:
        is_fg = pre_params['seg_colors'][:, 0] > 0.5
        rendervar = {k: v[is_fg] for k, v in rendervar.items()}
    return rendervar

def visualize(sequence, exp_name,prompt,res_dir, args):
    with torch.no_grad():
        pre_params, decoder = load_scene_data(sequence, args)
        legs_param_dir = f"./{args.trained_output_dir}/{exp_name}/{sequence}"
        vlm_encoder = VICLIPNetwork(VICLIPNetworkConfig)
        vlm_encoder.set_positives(prompt)
        num_timesteps = len(os.listdir(legs_param_dir)) // 2
        images = []
        probs, ims, segs = [],[],[]
        for t in tqdm(range(num_timesteps)):
            render_var = get_render_var(pre_params, legs_param_dir, t)            
            y_angle = 360*t / num_timesteps
            w2c, k = init_camera(y_angle)
            prob,seg, im = render_prob_sim(w2c, k, render_var, decoder, vlm_encoder, t)
            torch.cuda.empty_cache()
            probs.append(prob)
            ims.append(im)
            segs.append(seg)
        probs = torch.stack(probs, dim =0)
        probs = torch.clip(probs-0.5,0,1)
        probs_norm = probs - torch.min(probs)
        probs_norm = probs_norm / probs_norm.max() 
        for i in range(len(ims)):
            p =probs_norm[i]* segs[i]
            p = apply_colormap(p, normalize=False, shift=False, clip=False)
            composited =p*0.5 +ims[i] * 0.5
            images.append(to8b(composited))
        imageio.mimwrite(os.path.join(res_dir, "4legs_vid.mp4"), images, fps=args.fps)

if __name__ == "__main__":
    parser = ArgumentParser(description="Train args")
    parser.add_argument("-p", "--prompt", type=str, required=True)
    parser.add_argument("-s","--sequence", type=str, required=True)
    parser.add_argument("-e", "--exp_name", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pre_output_dir", type=str, default="output")
    parser.add_argument("--pre_exp_name", type=str, default="pretrained")
    parser.add_argument("--autoencoder_dir", type=str, default="ae")
    parser.add_argument("--trained_output_dir", type=str, default="output")
    args = parser.parse_args()
    exp_name = args.exp_name
    sequence = args.sequence
    prompt = args.prompt
    prompt_dir = "_".join(prompt.split(" "))
    res_dir = f"./{args.results_dir}/{exp_name}/{sequence}/{prompt_dir}"
    os.makedirs(res_dir, exist_ok=True)    
    print(prompt)
    prompt = [prompt]
    visualize(sequence, exp_name, prompt, res_dir,args)

