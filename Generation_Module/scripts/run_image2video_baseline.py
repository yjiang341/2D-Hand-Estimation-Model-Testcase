"""
FH03 — Behavior-preserving execution of Image2Video baseline.

This script reproduces the exact logic of FoundHand/demos/Image2Video.ipynb
without any Jupyter kernel. The only differences from the notebook:
  - matplotlib backend is 'Agg' (no display); intermediate figures are saved.
  - plt.show() calls are replaced with plt.savefig() + plt.close().
  - Output is directed to a timestamped directory under Generation_Module/outputs/.
  - Checkpoint and data paths are resolved from the script's location.

Run from workspace root:
    python Generation_Module/scripts/run_image2video_baseline.py

Equivalently, run from Generation_Module/FoundHand/demos/ with --demo-cwd flag.
"""

import sys
import os
import os.path as osp
import argparse
import logging
import pickle
import time

# ── output routing ──────────────────────────────────────────────────────────
OUTPUT_DIR = osp.join(
    osp.dirname(__file__), "..", "outputs", "image2video_original"
)
OUTPUT_DIR = osp.abspath(OUTPUT_DIR)
FRAMES_DIR = osp.join(OUTPUT_DIR, "frames")
LOG_PATH = osp.join(OUTPUT_DIR, "execution.log")
os.makedirs(FRAMES_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── matplotlib Agg — MUST happen before any other matplotlib import ─────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── FoundHand source on path ────────────────────────────────────────────────
_SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
_FH_SRC = osp.abspath(osp.join(_SCRIPT_DIR, "..", "FoundHand"))
if _FH_SRC not in sys.path:
    sys.path.insert(0, _FH_SRC)

# ── remaining imports (same as notebook) ────────────────────────────────────
import torch
from dataclasses import dataclass
import numpy as np
import skimage.io as io
import cv2
import mediapipe as mp
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from models import vqvae
from models import vit
from diffusion import create_diffusion
from utils.utils import scale_keypoint, keypoint_heatmap, check_keypoints_validity
from utils.segment_hoi import init_sam, show_mask  # noqa: F401
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Notebook helpers (copied verbatim from Cell 0)
# ─────────────────────────────────────────────────────────────────────────────

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def unnormalize(x):
    return (((x + 1) / 2) * 255).astype(np.uint8)


def visualize_hand(ax, all_joints, img):
    connections = [
        ((0, 1), 'red'), ((1, 2), 'green'), ((2, 3), 'blue'), ((3, 4), 'purple'),
        ((0, 5), 'orange'), ((5, 6), 'pink'), ((6, 7), 'brown'), ((7, 8), 'cyan'),
        ((0, 9), 'yellow'), ((9, 10), 'magenta'), ((10, 11), 'lime'), ((11, 12), 'indigo'),
        ((0, 13), 'olive'), ((13, 14), 'teal'), ((14, 15), 'navy'), ((15, 16), 'gray'),
        ((0, 17), 'lavender'), ((17, 18), 'silver'), ((18, 19), 'maroon'), ((19, 20), 'fuchsia'),
    ]
    H, W, C = img.shape
    ax.imshow(img)
    for start_i in [0, 21]:
        joints = all_joints[start_i: start_i + 21]
        for connection, color in connections:
            joint1 = joints[connection[0]]
            joint2 = joints[connection[1]]
            ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], color=color)
    ax.set_xlim([0, W])
    ax.set_ylim([0, H])
    ax.grid(False)
    ax.set_axis_off()
    ax.invert_yaxis()


def draw_keypoint_trajectories(image_path, keypoints_list, image_size=(256, 256)):
    image = cv2.imread(image_path)
    num_keypoints = 42
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i / num_keypoints) for i in range(num_keypoints)]
    for kp_idx in range(num_keypoints):
        for t in range(1, len(keypoints_list)):
            pt1 = tuple(keypoints_list[t - 1][kp_idx].astype(int))
            pt2 = tuple(keypoints_list[t][kp_idx].astype(int))
            color = tuple(int(255 * c) for c in colors[kp_idx][:3][::-1])
            cv2.line(image, pt1, pt2, color, 1)
    return image


def make_ref_cond(img, keypts, hand_mask, device='cuda',
                  target_size=(256, 256), latent_size=(32, 32)):
    image_transform = Compose([
        ToTensor(),
        Resize(target_size),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    image = image_transform(img).to(device)
    kpts_valid = check_keypoints_validity(keypts, target_size)
    heatmaps = torch.tensor(
        keypoint_heatmap(scale_keypoint(keypts, target_size, latent_size),
                         latent_size, var=1.) * kpts_valid[:, None, None],
        dtype=torch.float, device=device,
    )[None, ...]
    mask = torch.tensor(
        cv2.resize(hand_mask.astype(int), dsize=latent_size,
                   interpolation=cv2.INTER_NEAREST),
        dtype=torch.float, device=device,
    ).unsqueeze(0)[None, ...]
    return image[None, ...], heatmaps, mask


def frames_to_video(frames, output_path, fps=30, resize=None, rgb2bgr=False):
    if len(frames) == 0:
        raise ValueError("No frames provided")
    height, width = frames[0].shape[:2]
    if resize is not None:
        width, height = resize
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for k_frame, frame in enumerate(frames):
        if len(frames[0].shape) == 2:
            frame = frame[..., None].repeat(3, axis=-1)
        if resize is not None:
            frame = cv2.resize(frame, (width, height))
        if rgb2bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HandDiffOpts:
    run_name: str = 'ViT_256_handmask_heatmap_nvs_b25_lr1e-5'
    sd_path: str = '/users/kchen157/scratch/weights/SD/sd-v1-4.ckpt'
    log_dir: str = '/users/kchen157/scratch/log'
    data_root: str = '/users/kchen157/data/users/kchen157/dataset/handdiff'
    image_size: tuple = (256, 256)
    latent_size: tuple = (32, 32)
    latent_dim: int = 4
    mask_bg: bool = False
    kpts_form: str = 'heatmap'
    n_keypoints: int = 42
    n_mask: int = 1
    noise_steps: int = 1000
    test_sampling_steps: int = 250
    ddim_steps: int = 100
    ddim_discretize: str = "uniform"
    ddim_eta: float = 0.
    beta_start: float = 8.5e-4
    beta_end: float = 0.012
    latent_scaling_factor: float = 0.18215
    cfg_pose: float = 5.
    cfg_appearance: float = 3.5
    batch_size: int = 25
    lr: float = 1e-5
    max_epochs: int = 500
    log_every_n_steps: int = 100
    limit_val_batches: int = 1
    n_gpu: int = 8
    num_nodes: int = 1
    precision: str = '16-mixed'
    profiler: str = 'simple'
    swa_epoch_start: int = 10
    swa_lrs: float = 1e-3
    num_workers: int = 10
    n_val_samples: int = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--idx", type=str, default="IMG_1087")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--right-hand-only", action="store_true")
    args = parser.parse_args()

    # Paths resolved relative to FoundHand source root (= demos/../)
    _demos_root = osp.join(_FH_SRC, "demos")
    model_weights_dir = osp.abspath(osp.join(_demos_root, "..", "weights"))
    data_root = osp.abspath(osp.join(_demos_root, "..", "test_data", "iphone_video"))

    model_path = osp.join(model_weights_dir, 'DINO_EMA_11M_b50_lr1e-5_epoch6_step320k.ckpt')
    vae_path = osp.join(model_weights_dir, 'vae-ft-mse-840000-ema-pruned.ckpt')
    sam_path = osp.join(model_weights_dir, 'sam_vit_h_4b8939.pth')

    log.info("=== Image2Video Baseline ===")
    log.info(f"FoundHand source : {_FH_SRC}")
    log.info(f"weights dir      : {model_weights_dir}")
    log.info(f"data root        : {data_root}")
    log.info(f"output dir       : {OUTPUT_DIR}")
    log.info(f"idx={args.idx}, start_frame={args.start_frame}, max_frames={args.max_frames}")

    opts = HandDiffOpts()

    # ── Cell 0: load models ───────────────────────────────────────────────
    log.info("Loading diffusion model...")
    t0 = time.time()
    diffusion = create_diffusion(str(opts.test_sampling_steps))
    model = vit.DiT_XL_2(
        input_size=opts.latent_size[0],
        latent_dim=opts.latent_dim,
        in_channels=opts.latent_dim + opts.n_keypoints + opts.n_mask,
        learn_sigma=True,
    ).cuda()
    ckpt_state_dict = torch.load(model_path)['ema_state_dict']
    missing_keys, extra_keys = model.load_state_dict(ckpt_state_dict, strict=False)
    model.eval()
    assert len(missing_keys) == 0, f"DiT missing keys: {missing_keys}"
    log.info(f"DiT loaded  — extra_keys={len(extra_keys)}  ({time.time()-t0:.1f}s)")

    vae_state_dict = torch.load(vae_path)['state_dict']
    autoencoder = vqvae.create_model(3, 3, opts.latent_dim).eval().requires_grad_(False).cuda()
    missing_keys, extra_keys = autoencoder.load_state_dict(vae_state_dict, strict=False)
    autoencoder.eval()
    assert len(missing_keys) == 0, f"VAE missing keys: {missing_keys}"
    log.info(f"VAE loaded  — extra_keys={len(extra_keys)}")

    log.info("Initializing SAM...")
    sam_predictor = init_sam(ckpt_path=sam_path)
    log.info("SAM ready.")

    # ── Cell 1: load data and visualise trajectories ──────────────────────
    idx = args.idx
    start_frame = args.start_frame
    max_frames = args.max_frames
    right_hand_only = args.right_hand_only

    image_file = osp.join(data_root, idx, f'{start_frame:04d}.jpg')
    path_file = osp.join(data_root, f'{idx}.pkl')

    with open(path_file, 'rb') as f:
        data = pickle.load(f)

    log.info(f"Sequence length available: {len(data[start_frame:start_frame+max_frames])}")
    traj_img = draw_keypoint_trajectories(image_file, data[start_frame:start_frame+max_frames])
    traj_path = osp.join(OUTPUT_DIR, "trajectory_vis.jpg")
    cv2.imwrite(traj_path, traj_img)
    log.info(f"Trajectory visualisation saved -> {traj_path}")

    # ── Cell 2: build target conditioning tensors ─────────────────────────
    with open(path_file, 'rb') as f:
        keypts_sequence = pickle.load(f)

    target_conds = []
    for keypts in keypts_sequence[start_frame:start_frame + max_frames]:
        kpts_valid = check_keypoints_validity(keypts, opts.image_size)
        if right_hand_only:
            kpts_valid[21:] *= 0
        target_heatmaps = torch.tensor(
            keypoint_heatmap(
                scale_keypoint(keypts, opts.image_size, opts.latent_size),
                opts.latent_size, var=1.) * kpts_valid[:, None, None],
            dtype=torch.float, device='cuda',
        )[None, ...]
        target_cond = torch.cat([
            target_heatmaps,
            torch.zeros((1, 1, opts.latent_size[0], opts.latent_size[1])).to(target_heatmaps),
        ], 1)
        target_conds.append(target_cond)
    log.info(f"Built {len(target_conds)} target conditioning tensors.")

    # ── Cell 3: reference image + SAM mask ───────────────────────────────
    ref_conds = []
    bootstrap_frames = [start_frame]
    for k, frame in enumerate(bootstrap_frames):
        image_file = osp.join(data_root, idx, f'{frame:04d}.jpg')
        img = io.imread(image_file)
        log.info(f"Reference image shape: {img.shape}")
        img = cv2.resize(img, opts.image_size, interpolation=cv2.INTER_AREA)
        keypts = keypts_sequence[frame]

        sam_predictor.set_image(img)
        if keypts[0].sum() != 0 and keypts[21].sum() != 0:
            input_point = np.array([keypts[0], keypts[21]])
            input_label = np.array([1, 1])
        elif keypts[0].sum() != 0:
            input_point = np.array(keypts[:1])
            input_label = np.array([1])
        elif keypts[21].sum() != 0:
            input_point = np.array(keypts[21:22])
            input_label = np.array([1])
        else:
            input_point = np.array(keypts[:1])
            input_label = np.array([1])

        masks, _, _ = sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        hand_mask = masks[0]
        masked_img = img * hand_mask[..., None] + 255 * (1 - hand_mask[..., None])

        fig, axs = plt.subplots(1, 2, figsize=(3 * 2, 3))
        visualize_hand(axs[0], keypts, img)
        visualize_hand(axs[1], keypts, masked_img.astype(np.uint8))
        plt.tight_layout()
        ref_fig_path = osp.join(OUTPUT_DIR, f"ref_frame_{frame:04d}.jpg")
        plt.savefig(ref_fig_path, dpi=72, bbox_inches='tight')
        plt.close()
        log.info(f"Reference frame visualisation saved -> {ref_fig_path}")

        image, heatmaps, mask = make_ref_cond(
            img, keypts, hand_mask, device='cuda',
            target_size=opts.image_size, latent_size=opts.latent_size,
        )
        latent = opts.latent_scaling_factor * autoencoder.encode(image).sample()
        if k == 0:
            ref_image = img
            ref_keypts = keypts
            src_ref_cond = torch.cat([latent, heatmaps, mask], 1)
        else:
            ref_conds.append(torch.cat([latent, heatmaps, mask], 1))

    log.info(f"ref_conds: {len(ref_conds)}")

    # ── Cell 4: diffusion sampling loop ───────────────────────────────────
    cfg_scale = args.cfg_scale
    last_N_frames = 1
    nvs = torch.zeros(1, dtype=torch.int, device='cuda')
    z = torch.randn(
        (1, opts.latent_dim, opts.latent_size[0], opts.latent_size[1]),
        device='cuda',
    )
    z = torch.cat([z, z], 0)

    temp_ref_conds = []
    video_frames = []
    t_gen_start = time.time()

    for k, target_cond in enumerate(target_conds):
        log.info(f"Sampling frame {k}/{min(max_frames, len(target_conds))}...")
        model_kwargs = dict(
            target_cond=torch.cat([target_cond, torch.zeros_like(target_cond)]),
            ref_cond=torch.cat([src_ref_cond, torch.zeros_like(src_ref_cond)]),
            nvs=torch.cat([nvs, 2 * torch.ones_like(nvs)]),
            cfg_scale=cfg_scale,
        )
        samples, _ = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs,
            ref_conds=[src_ref_cond] + ref_conds + temp_ref_conds,
            progress=False, device='cuda',
        ).chunk(2)

        sampled_images = autoencoder.decode(samples / opts.latent_scaling_factor)
        sampled_images = torch.clamp(sampled_images, min=-1., max=1.)
        sampled_images = unnormalize(sampled_images.permute(0, 2, 3, 1).cpu().numpy())
        sampled_image = sampled_images[0]
        video_frames.append(sampled_image)

        # Save individual frame
        frame_path = osp.join(FRAMES_DIR, f'{idx}_{k:04d}.jpg')
        io.imsave(frame_path, sampled_image)

        # Update autoregressive reference
        sam_predictor.set_image(sampled_image)
        masks, _, _ = sam_predictor.predict(
            point_coords=np.array([keypts_sequence[start_frame + k][0]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        hand_mask = masks[0]
        mask = torch.tensor(
            cv2.resize(masks[0].astype(int), dsize=opts.latent_size,
                       interpolation=cv2.INTER_NEAREST),
            dtype=torch.float, device='cuda',
        ).unsqueeze(0)[None, ...]
        if len(temp_ref_conds) >= last_N_frames and len(temp_ref_conds) > 0:
            temp_ref_conds.pop(0)
        temp_ref_conds.append(torch.cat([samples, target_cond], 1))

        # Save per-frame visualisation
        fig, axs = plt.subplots(1, 3, figsize=(6 * 3, 6))
        for i, vis_img in enumerate([ref_image, sampled_image]):
            axs[i].imshow(vis_img)
            axs[i].axis('off')
            axs[i].grid(False)
        visualize_hand(axs[2], keypts_sequence[start_frame + k], sampled_image)
        axs[2].imshow(
            cv2.resize(target_cond.cpu().numpy()[0, :42].sum(0),
                       opts.image_size, interpolation=cv2.INTER_AREA),
            cmap='hot', alpha=0.5,
        )
        plt.tight_layout()
        plt.title(f'{k}/{len(target_conds)}')
        vis_path = osp.join(FRAMES_DIR, f'{idx}_{k:04d}_vis.jpg')
        plt.savefig(vis_path, dpi=72, bbox_inches='tight')
        plt.close()

    elapsed = time.time() - t_gen_start
    log.info(f"Generation complete: {len(video_frames)} frames in {elapsed:.1f}s "
             f"({elapsed/max(1,len(video_frames)):.1f}s/frame)")

    # ── Save video ────────────────────────────────────────────────────────
    video_path = osp.join(OUTPUT_DIR, f'{idx}.avi')
    frames_to_video(video_frames, video_path, fps=20, rgb2bgr=True)
    log.info(f"Video saved -> {video_path}")
    log.info("=== FH03 COMPLETE ===")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        logging.getLogger(__name__).error("FATAL EXCEPTION:\n" + traceback.format_exc())
        raise
