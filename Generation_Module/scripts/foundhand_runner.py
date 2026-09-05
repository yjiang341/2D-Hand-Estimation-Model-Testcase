"""
FH04 — Reproducible FoundHand Image2Video inference runner.

This module separates model initialization, checkpoint loading, conditioning
construction, and DDIM sampling into explicit, independently testable functions.
It is the authoritative non-notebook entry point consumed by all downstream
tasks (FH05 MediaPipe adapter, FH08 temporal baseline, FH10 chunked streaming).

Public API
----------
load_models(weights_dir, device, sampling_steps)
    -> (model, autoencoder, sam_predictor, diffusion, opts)

build_target_cond(keypts, opts, device)
    -> Tensor [1, 43, 32, 32]   (42 heatmaps + 1 zero mask channel)

build_ref_cond(img_rgb_256, keypts, sam_predictor, autoencoder, opts, device)
    -> src_ref_cond Tensor [1, 47, 32, 32]  (4 VAE latent + 42 heatmaps + 1 SAM mask)

sample_frame(model, autoencoder, diffusion, target_cond, src_ref_cond,
             ref_conds, temp_ref_conds, opts, device, cfg_scale, z_seed)
    -> (sampled_image np.uint8 [H,W,3], samples Tensor [1,4,32,32])

update_autoreg_cond(samples, target_cond, sam_predictor, keypts,
                    temp_ref_conds, opts, device, last_N_frames)
    -> temp_ref_conds (updated in place, returns same list)

CLI
---
python foundhand_runner.py \\
    --weights-dir  PATH \\
    --data-root    PATH \\
    --idx          IMG_1087 \\
    --start-frame  0 \\
    --max-frames   50 \\
    --output-dir   PATH \\
    [--cfg-scale   2.5] \\
    [--sampling-steps 250] \\
    [--right-hand-only] \\
    [--device      cuda]
"""

from __future__ import annotations

import sys
import os
import os.path as osp
import argparse
import logging
import pickle
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ── FoundHand source on path ─────────────────────────────────────────────────
_THIS_DIR = osp.abspath(osp.dirname(__file__))
_FH_SRC = osp.abspath(osp.join(_THIS_DIR, "..", "FoundHand"))
if _FH_SRC not in sys.path:
    sys.path.insert(0, _FH_SRC)

import torch
import cv2
import skimage.io as skio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from models import vqvae, vit
from diffusion import create_diffusion
from utils.utils import scale_keypoint, keypoint_heatmap, check_keypoints_validity
from utils.segment_hoi import init_sam

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass (mirrors HandDiffOpts from the original notebook)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunnerConfig:
    """All hyper-parameters needed for inference, with explicit types."""
    image_size: Tuple[int, int] = (256, 256)
    latent_size: Tuple[int, int] = (32, 32)
    latent_dim: int = 4
    n_keypoints: int = 42
    n_mask: int = 1
    latent_scaling_factor: float = 0.18215
    cfg_pose: float = 5.0
    cfg_appearance: float = 3.5
    sampling_steps: int = 250
    ddim_eta: float = 0.0
    beta_start: float = 8.5e-4
    beta_end: float = 0.012
    noise_steps: int = 1000


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoundHandModels:
    model: torch.nn.Module
    autoencoder: torch.nn.Module
    sam_predictor: object
    diffusion: object
    cfg: RunnerConfig


def load_models(
    weights_dir: str,
    device: str = "cuda",
    sampling_steps: int = 250,
) -> FoundHandModels:
    """
    Load DiT, VAE, and SAM from checkpoints.

    Parameters
    ----------
    weights_dir : str
        Directory containing the three checkpoint files.
    device : str
        PyTorch device string, default "cuda".
    sampling_steps : int
        Number of DDIM steps, default 250.

    Returns
    -------
    FoundHandModels
        Named container with model, autoencoder, sam_predictor, diffusion, cfg.
    """
    cfg = RunnerConfig(sampling_steps=sampling_steps)

    model_path = osp.join(weights_dir, "DINO_EMA_11M_b50_lr1e-5_epoch6_step320k.ckpt")
    vae_path   = osp.join(weights_dir, "vae-ft-mse-840000-ema-pruned.ckpt")
    sam_path   = osp.join(weights_dir, "sam_vit_h_4b8939.pth")

    for p in (model_path, vae_path, sam_path):
        if not osp.exists(p):
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    log.info("Loading diffusion schedule (%d steps)...", sampling_steps)
    diffusion = create_diffusion(str(sampling_steps))

    log.info("Loading DiT model...")
    t0 = time.time()
    # Notebook pickled HandDiffOpts in the checkpoint; provide a stub so torch.load succeeds
    import __main__
    if not hasattr(__main__, "HandDiffOpts"):
        __main__.HandDiffOpts = type("HandDiffOpts", (), {})

    in_channels = cfg.latent_dim + cfg.n_keypoints + cfg.n_mask  # 4+42+1 = 47
    model = vit.DiT_XL_2(
        input_size=cfg.latent_size[0],
        latent_dim=cfg.latent_dim,
        in_channels=in_channels,
        learn_sigma=True,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    missing, extra = model.load_state_dict(ckpt["ema_state_dict"], strict=False)
    assert len(missing) == 0, f"DiT missing keys: {missing}"
    model.eval()
    log.info("DiT loaded — extra_keys=%d (%.1fs)", len(extra), time.time() - t0)

    log.info("Loading VAE...")
    vae_ckpt = torch.load(vae_path, map_location=device)
    autoencoder = (
        vqvae.create_model(3, 3, cfg.latent_dim)
        .eval()
        .requires_grad_(False)
        .to(device)
    )
    missing, extra = autoencoder.load_state_dict(vae_ckpt["state_dict"], strict=False)
    assert len(missing) == 0, f"VAE missing keys: {missing}"
    log.info("VAE loaded — extra_keys=%d", len(extra))

    log.info("Loading SAM ViT-H...")
    sam_predictor = init_sam(ckpt_path=sam_path)
    log.info("SAM ready.")

    return FoundHandModels(
        model=model,
        autoencoder=autoencoder,
        sam_predictor=sam_predictor,
        diffusion=diffusion,
        cfg=cfg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Conditioning construction
# ─────────────────────────────────────────────────────────────────────────────

def build_target_cond(
    keypts: np.ndarray,
    cfg: RunnerConfig,
    device: str = "cuda",
    right_hand_only: bool = False,
) -> torch.Tensor:
    """
    Convert one frame's [42, 2] keypoints into the target conditioning tensor.

    Parameters
    ----------
    keypts : np.ndarray [42, 2]
        Pixel-space keypoints in image_size coordinates.
        right-hand wrist = keypts[0], left-hand wrist = keypts[21].
    cfg : RunnerConfig
    device : str
    right_hand_only : bool
        Zero out left-hand keypoints when True.

    Returns
    -------
    Tensor [1, 43, 32, 32]  (42 heatmaps + 1 zero mask channel)
    """
    kpts_valid = check_keypoints_validity(keypts, cfg.image_size)
    if right_hand_only:
        kpts_valid[21:] *= 0
    heatmaps = torch.tensor(
        keypoint_heatmap(
            scale_keypoint(keypts, cfg.image_size, cfg.latent_size),
            cfg.latent_size,
            var=1.0,
        ) * kpts_valid[:, None, None],
        dtype=torch.float,
        device=device,
    )[None, ...]  # [1, 42, 32, 32]
    zero_mask = torch.zeros(
        (1, 1, cfg.latent_size[0], cfg.latent_size[1]),
        dtype=torch.float,
        device=device,
    )
    return torch.cat([heatmaps, zero_mask], dim=1)  # [1, 43, 32, 32]


def build_ref_cond(
    img_rgb_256: np.ndarray,
    keypts: np.ndarray,
    sam_predictor,
    autoencoder: torch.nn.Module,
    cfg: RunnerConfig,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode the reference image into the src_ref_cond tensor.

    Parameters
    ----------
    img_rgb_256 : np.ndarray [256, 256, 3]  uint8  RGB
        Reference frame already resized to image_size.
    keypts : np.ndarray [42, 2]
        Reference frame keypoints in pixel space.
    sam_predictor : SamPredictor
    autoencoder : nn.Module  (VAE encoder/decoder)
    cfg : RunnerConfig
    device : str

    Returns
    -------
    Tensor [1, 47, 32, 32]  (4 VAE latent + 42 heatmaps + 1 SAM mask)
    """
    # SAM hand mask
    sam_predictor.set_image(img_rgb_256)
    if keypts[0].sum() != 0 and keypts[21].sum() != 0:
        pts, lbls = np.array([keypts[0], keypts[21]]), np.array([1, 1])
    elif keypts[0].sum() != 0:
        pts, lbls = np.array(keypts[:1]), np.array([1])
    elif keypts[21].sum() != 0:
        pts, lbls = np.array(keypts[21:22]), np.array([1])
    else:
        pts, lbls = np.array(keypts[:1]), np.array([1])

    masks, _, _ = sam_predictor.predict(
        point_coords=pts, point_labels=lbls, multimask_output=False
    )
    hand_mask = masks[0]

    # VAE image latent
    image_transform = Compose([
        ToTensor(),
        Resize(cfg.image_size),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    image_t = image_transform(img_rgb_256).to(device)[None, ...]  # [1,3,256,256]

    with torch.no_grad():
        latent = cfg.latent_scaling_factor * autoencoder.encode(image_t).sample()  # [1,4,32,32]

    # Reference heatmaps
    kpts_valid = check_keypoints_validity(keypts, cfg.image_size)
    heatmaps = torch.tensor(
        keypoint_heatmap(
            scale_keypoint(keypts, cfg.image_size, cfg.latent_size),
            cfg.latent_size,
            var=1.0,
        ) * kpts_valid[:, None, None],
        dtype=torch.float,
        device=device,
    )[None, ...]  # [1,42,32,32]

    mask_t = torch.tensor(
        cv2.resize(
            hand_mask.astype(int),
            dsize=cfg.latent_size,
            interpolation=cv2.INTER_NEAREST,
        ),
        dtype=torch.float,
        device=device,
    ).unsqueeze(0)[None, ...]  # [1,1,32,32]

    return torch.cat([latent, heatmaps, mask_t], dim=1)  # [1,47,32,32]


# ─────────────────────────────────────────────────────────────────────────────
# Single-frame sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_frame(
    models: FoundHandModels,
    target_cond: torch.Tensor,
    src_ref_cond: torch.Tensor,
    ref_conds: list,
    temp_ref_conds: list,
    cfg_scale: float = 2.5,
    device: str = "cuda",
    z_seed: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Run one DDIM sampling pass.

    Parameters
    ----------
    models : FoundHandModels
    target_cond : Tensor [1, 43, 32, 32]
    src_ref_cond : Tensor [1, 47, 32, 32]
    ref_conds : list of Tensor [1, 47, 32, 32]   (static bootstrap refs)
    temp_ref_conds : list of Tensor              (autoregressive rolling refs)
    cfg_scale : float
    device : str
    z_seed : Tensor [1,4,32,32] or None — reused across frames for determinism

    Returns
    -------
    sampled_image : np.ndarray [256, 256, 3]  uint8  RGB
    samples : Tensor [1, 4, 32, 32]  (latent before decoding, for autoreg)
    """
    cfg = models.cfg
    if z_seed is None:
        z_seed = torch.randn(
            (1, cfg.latent_dim, cfg.latent_size[0], cfg.latent_size[1]),
            device=device,
        )
    z = torch.cat([z_seed, z_seed], dim=0)  # CFG-doubled batch

    nvs = torch.zeros(1, dtype=torch.int, device=device)
    model_kwargs = dict(
        target_cond=torch.cat([target_cond, torch.zeros_like(target_cond)]),
        ref_cond=torch.cat([src_ref_cond, torch.zeros_like(src_ref_cond)]),
        nvs=torch.cat([nvs, 2 * torch.ones_like(nvs)]),
        cfg_scale=cfg_scale,
    )

    with torch.no_grad():
        out = models.diffusion.p_sample_loop(
            models.model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            ref_conds=[src_ref_cond] + ref_conds + temp_ref_conds,
            progress=False,
            device=device,
        )
    samples, _ = out.chunk(2)

    decoded = models.autoencoder.decode(samples / cfg.latent_scaling_factor)
    decoded = torch.clamp(decoded, min=-1.0, max=1.0)
    sampled_image = (
        ((decoded.permute(0, 2, 3, 1).cpu().numpy()[0] + 1) / 2 * 255)
        .astype(np.uint8)
    )
    return sampled_image, samples


def update_autoreg_cond(
    samples: torch.Tensor,
    target_cond: torch.Tensor,
    sam_predictor,
    keypts: np.ndarray,
    temp_ref_conds: list,
    cfg: RunnerConfig,
    device: str = "cuda",
    last_N_frames: int = 1,
) -> list:
    """
    Update the rolling autoregressive reference conditioning list in place.

    Parameters
    ----------
    samples : Tensor [1,4,32,32]  — latent from sample_frame
    target_cond : Tensor [1,43,32,32]
    sam_predictor : SamPredictor
    keypts : np.ndarray [42,2]   — keypoints for the *current* frame
    temp_ref_conds : list        — rolling list mutated in place
    cfg : RunnerConfig
    device : str
    last_N_frames : int          — rolling window size

    Returns
    -------
    temp_ref_conds (same list, mutated)
    """
    new_entry = torch.cat([samples, target_cond], dim=1)  # [1,47,32,32]
    if len(temp_ref_conds) >= last_N_frames and len(temp_ref_conds) > 0:
        temp_ref_conds.pop(0)
    temp_ref_conds.append(new_entry)
    return temp_ref_conds


# ─────────────────────────────────────────────────────────────────────────────
# Video export
# ─────────────────────────────────────────────────────────────────────────────

def frames_to_video(
    frames: list,
    output_path: str,
    fps: int = 20,
    resize: Optional[Tuple[int, int]] = None,
    rgb2bgr: bool = True,
) -> None:
    """Write a list of RGB uint8 numpy frames to an AVI file."""
    if not frames:
        raise ValueError("No frames to write")
    height, width = frames[0].shape[:2]
    if resize is not None:
        width, height = resize
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for k, frame in enumerate(frames):
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if resize is not None:
            frame = cv2.resize(frame, (width, height))
        if rgb2bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        if k % 10 == 0:
            log.info("  Writing frame %d/%d", k, len(frames))
    out.release()
    log.info("Video saved -> %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = osp.join(output_dir, "runner.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FoundHand Image2Video runner (FH04)")
    parser.add_argument("--weights-dir",     required=True,  help="Directory with the three checkpoint files")
    parser.add_argument("--data-root",       required=True,  help="Directory containing <idx>/ frames and <idx>.pkl")
    parser.add_argument("--idx",             required=True,  help="Sequence identifier, e.g. IMG_1087")
    parser.add_argument("--start-frame",     type=int, default=0)
    parser.add_argument("--max-frames",      type=int, default=50)
    parser.add_argument("--output-dir",      required=True,  help="Where to write frames/ and video")
    parser.add_argument("--cfg-scale",       type=float, default=2.5)
    parser.add_argument("--sampling-steps",  type=int, default=250)
    parser.add_argument("--last-n-frames",   type=int, default=1, help="Autoregressive window size")
    parser.add_argument("--right-hand-only", action="store_true")
    parser.add_argument("--device",          default="cuda")
    args = parser.parse_args()

    _setup_logging(args.output_dir)
    frames_dir = osp.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    log.info("=== FoundHand Runner (FH04) ===")
    log.info("weights_dir    : %s", args.weights_dir)
    log.info("data_root      : %s", args.data_root)
    log.info("idx=%s, start_frame=%d, max_frames=%d", args.idx, args.start_frame, args.max_frames)
    log.info("output_dir     : %s", args.output_dir)

    # ── Load models ──────────────────────────────────────────────────────────
    models = load_models(args.weights_dir, device=args.device, sampling_steps=args.sampling_steps)
    cfg = models.cfg

    # ── Load sequence data ───────────────────────────────────────────────────
    pkl_path = osp.join(args.data_root, f"{args.idx}.pkl")
    with open(pkl_path, "rb") as f:
        keypts_sequence = pickle.load(f)
    seq = keypts_sequence[args.start_frame: args.start_frame + args.max_frames]
    log.info("Loaded %d frames from %s", len(seq), pkl_path)

    # ── Build all target conditioning tensors ────────────────────────────────
    target_conds = [
        build_target_cond(kp, cfg, device=args.device, right_hand_only=args.right_hand_only)
        for kp in seq
    ]
    log.info("Built %d target conditioning tensors", len(target_conds))

    # ── Reference frame ───────────────────────────────────────────────────────
    ref_frame_path = osp.join(args.data_root, args.idx, f"{args.start_frame:04d}.jpg")
    ref_img_raw = skio.imread(ref_frame_path)
    ref_img = cv2.resize(ref_img_raw, cfg.image_size, interpolation=cv2.INTER_AREA)
    ref_keypts = keypts_sequence[args.start_frame]

    log.info("Building reference conditioning (SAM)...")
    src_ref_cond = build_ref_cond(
        ref_img, ref_keypts, models.sam_predictor, models.autoencoder, cfg, device=args.device
    )
    log.info("Reference conditioning built — src_ref_cond shape: %s", tuple(src_ref_cond.shape))

    # ── Sampling loop ─────────────────────────────────────────────────────────
    ref_conds: list = []
    temp_ref_conds: list = []
    video_frames: list = []
    z_seed = torch.randn(
        (1, cfg.latent_dim, cfg.latent_size[0], cfg.latent_size[1]),
        device=args.device,
    )
    t_start = time.time()

    for k, target_cond in enumerate(target_conds):
        log.info("Sampling frame %d/%d...", k, len(target_conds))
        sampled_image, samples = sample_frame(
            models=models,
            target_cond=target_cond,
            src_ref_cond=src_ref_cond,
            ref_conds=ref_conds,
            temp_ref_conds=temp_ref_conds,
            cfg_scale=args.cfg_scale,
            device=args.device,
            z_seed=z_seed,
        )
        video_frames.append(sampled_image)

        # Save frame
        frame_path = osp.join(frames_dir, f"{args.idx}_{k:04d}.jpg")
        skio.imsave(frame_path, sampled_image)

        # Update autoregressive conditioning
        update_autoreg_cond(
            samples=samples,
            target_cond=target_cond,
            sam_predictor=models.sam_predictor,
            keypts=keypts_sequence[args.start_frame + k],
            temp_ref_conds=temp_ref_conds,
            cfg=cfg,
            device=args.device,
            last_N_frames=args.last_n_frames,
        )

    elapsed = time.time() - t_start
    log.info(
        "Generation complete: %d frames in %.1fs (%.1fs/frame)",
        len(video_frames), elapsed, elapsed / max(1, len(video_frames)),
    )

    # ── Write video ───────────────────────────────────────────────────────────
    video_path = osp.join(args.output_dir, f"{args.idx}.avi")
    frames_to_video(video_frames, video_path, fps=20, rgb2bgr=True)
    log.info("=== Runner complete ===")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        logging.getLogger(__name__).error("FATAL:\n%s", traceback.format_exc())
        raise
