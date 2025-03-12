
from argparse import ArgumentParser
import torch
import numpy as np
import os
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera
import imageio

REMOVE_BACKGROUND = False  # False or True
w, h = 640, 360
near, far = 0.01, 100.0

def init_camera(y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k

def load_scene_data(seq, exp,args,  seg_as_col=False):
    params = dict(np.load(f"./{args.trained_output_dir}/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        if REMOVE_BACKGROUND:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg

def render(w2c, k, timestep_data):
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        im, _, depth, = Renderer(raster_settings=cam)(**timestep_data)
        return im, depth

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def visualize(seq, exp, args):
    scene_data, is_fg = load_scene_data(seq, exp, args)
    num_timesteps = len(scene_data)
    render_images = []
    for t in range(num_timesteps):
        y_angle = 360*t / num_timesteps
        w2c, k = init_camera(y_angle)
        im, depth = render(w2c, k, scene_data[t])
        render_images.append(to8b(im).transpose(1,2,0))
    imageio.mimwrite(f"./{args.results_dir}/{exp}/{seq}/video_rgb.mp4", render_images, fps=args.fps)

if __name__ == "__main__":
    parser = ArgumentParser(description="Train args")
    parser.add_argument("-s","--sequence", type=str, required=True)
    parser.add_argument("-e", "--exp_name", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--trained_output_dir", type=str, default="output")
    args = parser.parse_args()
    exp_name = args.exp_name
    sequence = args.sequence
    os.makedirs(f"./{args.results_dir}/{exp_name}/{sequence}", exist_ok=True)
    visualize(sequence, exp_name, args)
