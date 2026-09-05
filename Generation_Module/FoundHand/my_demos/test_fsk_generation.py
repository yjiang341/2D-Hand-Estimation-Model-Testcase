import os
import os.path as osp

import numpy as np
import torch
import cv2

from models import vqvae
from models import vit
from diffusion import create_diffusion

from utils.utils import (
    scale_keypoint,
    keypoint_heatmap,
    check_keypoints_validity,
)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = (256, 256)
LATENT_SIZE = (32, 32)
LATENT_DIM = 4

N_KEYPOINTS = 42
N_MASK = 1

TEST_SAMPLING_STEPS = 20
LATENT_SCALING_FACTOR = 0.18215

CFG_SCALE = 2.5

POSE_PATH = r"D:\Project\2D-Hand\Generation_Module\FoundHand\test_result\result_npz\v3_recovered_pose.npz"

WEIGHTS_DIR = r"..\..\weights"

MODEL_PATH = osp.join(
    WEIGHTS_DIR,
    "DINO_EMA_11M_b50_lr1e-5_epoch6_step320k.ckpt"
)

VAE_PATH = osp.join(
    WEIGHTS_DIR,
    "vae-ft-mse-840000-ema-pruned.ckpt"
)

OUTPUT_PATH = "test_fsk_generation.png"


# ============================================================
# Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Device")
print("=" * 60)
print(device)

if device != "cuda":
    raise RuntimeError(
        "FoundHand generation requires CUDA for this test."
    )


# ============================================================
# Load FSK pose
# ============================================================

print()
print("=" * 60)
print("Loading FSK pose")
print("=" * 60)

print("Loading:", POSE_PATH)

pose_data = np.load(POSE_PATH)

keypts_sequence = pose_data["keypoints"]
frame_ids = pose_data["frame_ids"]

print("Keypoints shape:", keypts_sequence.shape)
print("Frame IDs shape:", frame_ids.shape)


# ============================================================
# Detect coordinate format
# ============================================================

raw_min = float(keypts_sequence.min())
raw_max = float(keypts_sequence.max())

print()
print("Raw coordinate range:")
print("min:", raw_min)
print("max:", raw_max)

if raw_max <= 1.0 + 1e-5:
    coordinate_format = "normalized"

    keypts_sequence = (
        keypts_sequence.astype(np.float32)
        * np.array(IMAGE_SIZE, dtype=np.float32)
    )

elif raw_max <= max(IMAGE_SIZE) + 1e-5:
    coordinate_format = "pixel"

    keypts_sequence = keypts_sequence.astype(np.float32)

else:
    raise RuntimeError(
        f"Unexpected coordinate range: {raw_min} -> {raw_max}"
    )

print("Detected coordinate format:", coordinate_format)

print(
    "Pixel coordinate range:",
    keypts_sequence.min(),
    "->",
    keypts_sequence.max(),
)


# ============================================================
# Find first non-empty frame
# ============================================================

non_empty = []

for i in range(len(keypts_sequence)):
    if np.count_nonzero(keypts_sequence[i]) > 0:
        non_empty.append(i)

print()
print("Non-empty frames:", non_empty)

if len(non_empty) == 0:
    raise RuntimeError("No valid pose frames found.")

frame_index = non_empty[0]

keypts = keypts_sequence[frame_index]

print()
print("Using frame index:", frame_index)
print("Using frame ID:", frame_ids[frame_index])

print()
print("Selected keypoints:")
print(keypts)


# ============================================================
# Validate keypoints
# ============================================================

kpts_valid = check_keypoints_validity(
    keypts,
    IMAGE_SIZE,
)

print()
print("Valid keypoints:", int(kpts_valid.sum()), "/", N_KEYPOINTS)

if kpts_valid.sum() == 0:
    raise RuntimeError("No valid keypoints found.")


# ============================================================
# Convert to latent coordinates
# ============================================================

latent_keypts = scale_keypoint(
    keypts,
    IMAGE_SIZE,
    LATENT_SIZE,
)

print()
print("Latent keypoints shape:", latent_keypts.shape)

print(
    "Latent X range:",
    latent_keypts[:, 0].min(),
    "->",
    latent_keypts[:, 0].max(),
)

print(
    "Latent Y range:",
    latent_keypts[:, 1].min(),
    "->",
    latent_keypts[:, 1].max(),
)


# ============================================================
# Create heatmaps
# ============================================================

heatmaps_np = keypoint_heatmap(
    latent_keypts,
    LATENT_SIZE,
    var=1.0,
)

heatmaps_np *= kpts_valid[:, None, None]

print()
print("Heatmap shape:", heatmaps_np.shape)

print(
    "Heatmap range:",
    heatmaps_np.min(),
    "->",
    heatmaps_np.max(),
)


# ============================================================
# Convert heatmaps to torch
# ============================================================

target_heatmaps = torch.tensor(
    heatmaps_np,
    dtype=torch.float32,
    device=device,
).unsqueeze(0)

print()
print("Target heatmaps:", target_heatmaps.shape)


# ============================================================
# Add mask channel
# ============================================================

mask = torch.zeros(
    (1, 1, LATENT_SIZE[0], LATENT_SIZE[1]),
    dtype=torch.float32,
    device=device,
)

target_cond = torch.cat(
    [
        target_heatmaps,
        mask,
    ],
    dim=1,
)

print("Target condition:", target_cond.shape)


# ============================================================
# Load diffusion model
# ============================================================

print()
print("=" * 60)
print("Loading diffusion model")
print("=" * 60)

print("Model:", MODEL_PATH)

diffusion = create_diffusion(
    str(TEST_SAMPLING_STEPS)
)

model = vit.DiT_XL_2(
    input_size=LATENT_SIZE[0],
    latent_dim=LATENT_DIM,
    in_channels=LATENT_DIM + N_KEYPOINTS + N_MASK,
    learn_sigma=True,
).cuda()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cuda",
)

ckpt_state_dict = checkpoint["ema_state_dict"]

missing_keys, extra_keys = model.load_state_dict(
    ckpt_state_dict,
    strict=False,
)

print("Missing keys:", missing_keys)
print("Extra keys:", extra_keys)

if len(missing_keys) != 0:
    raise RuntimeError(
        f"Model has missing keys: {missing_keys}"
    )

model.eval()


# ============================================================
# Load VAE
# ============================================================

print()
print("=" * 60)
print("Loading VAE")
print("=" * 60)

print("VAE:", VAE_PATH)

vae_checkpoint = torch.load(
    VAE_PATH,
    map_location="cuda",
)

vae_state_dict = vae_checkpoint["state_dict"]

autoencoder = (
    vqvae.create_model(
        3,
        3,
        LATENT_DIM,
    )
    .eval()
    .requires_grad_(False)
    .cuda()
)

missing_keys, extra_keys = autoencoder.load_state_dict(
    vae_state_dict,
    strict=False,
)

print("Missing keys:", missing_keys)
print("Extra keys:", extra_keys)

if len(missing_keys) != 0:
    raise RuntimeError(
        f"VAE has missing keys: {missing_keys}"
    )

autoencoder.eval()


# ============================================================
# Prepare diffusion input
# ============================================================

print()
print("=" * 60)
print("Preparing diffusion input")
print("=" * 60)

z = torch.randn(
    (
        1,
        LATENT_DIM,
        LATENT_SIZE[0],
        LATENT_SIZE[1],
    ),
    device=device,
)

# CFG requires duplicated batch
z = torch.cat(
    [z, z],
    dim=0,
)

print("Noise shape:", z.shape)


# ============================================================
# CFG conditioning
# ============================================================

# Positive condition
cond = target_cond

# Unconditional condition
uncond = torch.zeros_like(target_cond)

model_kwargs = dict(
    target_cond=torch.cat(
        [
            cond,
            uncond,
        ],
        dim=0,
    ),

    ref_cond=torch.zeros(
        (
            2,
            LATENT_DIM + N_KEYPOINTS + N_MASK,
            LATENT_SIZE[0],
            LATENT_SIZE[1],
        ),
        device=device,
    ),

    nvs=torch.tensor(
        [0, 2],
        dtype=torch.int,
        device=device,
    ),

    cfg_scale=CFG_SCALE,
)


# ============================================================
# Sample
# ============================================================

print()
print("=" * 60)
print("Running diffusion")
print("=" * 60)

with torch.no_grad():

    samples, _ = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        ref_conds=[],
        progress=True,
        device=device,
    )

    # Keep first CFG output
    samples = samples[:1]


print()
print("Diffusion samples:", samples.shape)


# ============================================================
# Decode VAE
# ============================================================

print()
print("=" * 60)
print("Decoding VAE")
print("=" * 60)

with torch.no_grad():

    decoded = autoencoder.decode(
        samples / LATENT_SCALING_FACTOR
    )

    decoded = torch.clamp(
        decoded,
        min=-1.0,
        max=1.0,
    )

    image = (
        ((decoded + 1.0) / 2.0)
        * 255.0
    )

    image = image.permute(
        0,
        2,
        3,
        1,
    ).cpu().numpy().astype(np.uint8)

image = image[0]

print("Generated image shape:", image.shape)


# ============================================================
# Save
# ============================================================

cv2.imwrite(
    OUTPUT_PATH,
    cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    ),
)

print()
print("=" * 60)
print("SUCCESS")
print("=" * 60)

print("Saved:", osp.abspath(OUTPUT_PATH))