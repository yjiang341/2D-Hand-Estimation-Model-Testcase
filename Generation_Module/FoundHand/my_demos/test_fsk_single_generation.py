from __future__ import annotations

import gc
import os
import os.path as osp
import pickle
import random
import sys
from dataclasses import dataclass
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


# ============================================================
# Make FoundHand root importable
# ============================================================

CURRENT_DIR = osp.dirname(osp.abspath(__file__))
FOUNDHAND_ROOT = osp.dirname(CURRENT_DIR)

if FOUNDHAND_ROOT not in sys.path:
    sys.path.insert(0, FOUNDHAND_ROOT)

from models import vqvae
from models import vit
from diffusion import create_diffusion
from utils.utils import (
    check_keypoints_validity,
    keypoint_heatmap,
    scale_keypoint,
)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = (256, 256)
LATENT_SIZE = (32, 32)
LATENT_DIM = 4
NUM_KEYPOINTS = 42
NUM_MASK = 1

SAMPLING_STEPS = 100
LATENT_SCALING_FACTOR = 0.18215
CFG_SCALE = 2.5

# ------------------------------------------------------------
# Fast skeleton-only debug mode
# ------------------------------------------------------------
# True  -> stop after saving the debug skeleton figure
# False -> run full FoundHand generation
POSE_ONLY = False

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
SEED = 1234
REF_LATENT_SEED = SEED + 1
Z_SEED = SEED + 2
SAMPLING_NOISE_SEED = SEED + 3

# ------------------------------------------------------------
# Target mode
# ------------------------------------------------------------
# TARGET_MODE = "fsk_raw"
# TARGET_MODE = "fsk_retarget"   # old baseline
# TARGET_MODE = "fsk_kinematic"  # old absolute-direction transfer
# TARGET_MODE = "fsk_angle"      # new relative-angle retarget
TARGET_MODE = "fsk_retarget"

FSK_FRAME_ID = 8

# ------------------------------------------------------------
# Old baseline retarget settings
# ------------------------------------------------------------
BASELINE_MIRROR_X = False

# ------------------------------------------------------------
# Kinematic retarget settings
# ------------------------------------------------------------
KINEMATIC_REFLECTION_MODE = "never"   # "never" | "always" | "auto"
KINEMATIC_DIRECTION_BLEND = 1.0

# ------------------------------------------------------------
# Angle retarget settings
# ------------------------------------------------------------
# Based on your experiments, a fixed mirror around source wrist is
# currently more reliable than automatic reflection heuristics.
ANGLE_MIRROR_X = True

# Blend source finger articulation against reference articulation.
# 1.0 = pure source articulation.
# <1.0 = slightly pull toward reference anatomy for stability.
ANGLE_BASE_BLEND = 0.90
ANGLE_BEND_BLEND = 0.95

# Optional clamp (degrees) to prevent pathological bends.
ANGLE_MAX_BEND_DEG = 95.0

# ------------------------------------------------------------
# Reference mask settings
# ------------------------------------------------------------
# "sam" tries Segment Anything first, then falls back to convex hull.
# "hull" always uses a convex-hull mask built from keypoints.
MASK_MODE = "sam"
SAM_PATH = osp.join(
    FOUNDHAND_ROOT,
    "weights",
    "sam_vit_h_4b8939.pth",
)

# ------------------------------------------------------------
# Input paths
# ------------------------------------------------------------
POSE_PATH = r"D:\Project\2D-Hand\Generation_Module\FoundHand\test_result\result_npz\v3_recovered_pose.npz"

MODEL_PATH = osp.join(
    FOUNDHAND_ROOT,
    "weights",
    "DINO_EMA_11M_b50_lr1e-5_epoch6_step320k.ckpt",
)

VAE_PATH = osp.join(
    FOUNDHAND_ROOT,
    "weights",
    "vae-ft-mse-840000-ema-pruned.ckpt",
)

REF_DATA_ROOT = osp.join(
    FOUNDHAND_ROOT,
    "test_data",
    "iphone_video",
)

PREFERRED_REF_IDXS = [
    "IMG_1087",
    "IMG_1173",
]
PREFERRED_REF_FRAME = 6

# ------------------------------------------------------------
# Output paths
# ------------------------------------------------------------
OUTPUT_DIR = osp.join(FOUNDHAND_ROOT, "my_demos")

RUN_TAG = f"{TARGET_MODE}_{FSK_FRAME_ID}_{SAMPLING_STEPS}steps"
if POSE_ONLY:
    RUN_TAG += "_poseonly"

OUTPUT_PATH = osp.join(OUTPUT_DIR, f"{RUN_TAG}_generated.png")
DEBUG_PATH = osp.join(OUTPUT_DIR, f"{RUN_TAG}_debug.png")


# ============================================================
# Checkpoint compatibility class
# ============================================================

@dataclass
class HandDiffOpts:
    run_name: str = "ViT_256_handmask_heatmap_nvs_b25_lr1e-5"
    sd_path: str = "/users/kchen157/scratch/weights/SD/sd-v1-4.ckpt"
    log_dir: str = "/users/kchen157/scratch/log"
    data_root: str = "/users/kchen157/data/users/kchen157/dataset/handdiff"
    image_size: tuple = (256, 256)
    latent_size: tuple = (32, 32)
    latent_dim: int = 4
    mask_bg: bool = False
    kpts_form: str = "heatmap"
    n_keypoints: int = 42
    n_mask: int = 1
    noise_steps: int = 1000
    test_sampling_steps: int = 250
    ddim_steps: int = 100
    ddim_discretize: str = "uniform"
    ddim_eta: float = 0.0
    beta_start: float = 8.5e-4
    beta_end: float = 0.012
    latent_scaling_factor: float = 0.18215
    cfg_pose: float = 5.0
    cfg_appearance: float = 3.5
    batch_size: int = 25
    lr: float = 1e-5
    max_epochs: int = 500
    log_every_n_steps: int = 100
    limit_val_batches: int = 1
    n_gpu: int = 8
    num_nodes: int = 1
    precision: str = "16-mixed"
    profiler: str = "simple"
    swa_epoch_start: int = 10
    swa_lrs: float = 1e-3
    num_workers: int = 10
    n_val_samples: int = 4


# ============================================================
# Hand topology
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_CHAINS = [
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
]

THUMB_CHAIN = [1, 2, 3, 4]
PALM_MCP_INDICES = np.array([5, 9, 13, 17], dtype=np.int64)
PALM_ANCHOR_INDICES = np.array([0, 5, 9, 13, 17], dtype=np.int64)


# ============================================================
# Generic helpers
# ============================================================

def reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hand_presence(keypts: np.ndarray) -> tuple[bool, bool]:
    right = bool(np.any(keypts[:21] != 0))
    left = bool(np.any(keypts[21:42] != 0))
    return right, left


def describe_presence(keypts: np.ndarray) -> str:
    right, left = hand_presence(keypts)
    if right and left:
        return "RIGHT+LEFT"
    if right:
        return "RIGHT"
    if left:
        return "LEFT"
    return "NONE"


def normalize_vector(
    vector: np.ndarray,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        return vector / norm

    if fallback is None:
        raise RuntimeError("Cannot normalize zero-length vector.")

    fallback = np.asarray(fallback, dtype=np.float32)
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm <= 1e-8:
        raise RuntimeError("Both vector and fallback are degenerate.")

    return fallback / fallback_norm


def vector_angle(vector: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=np.float32)
    return float(np.arctan2(vector[1], vector[0]))


def wrap_angle(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def blend_angle(ref_angle: float, src_angle: float, blend: float) -> float:
    return wrap_angle(ref_angle + blend * wrap_angle(src_angle - ref_angle))


def unit_from_angle(angle_rad: float) -> np.ndarray:
    return np.asarray(
        [np.cos(angle_rad), np.sin(angle_rad)],
        dtype=np.float32,
    )


def rotate_vectors(vectors: np.ndarray, angle_rad: float) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    out = np.empty_like(vectors, dtype=np.float32)
    out[..., 0] = c * vectors[..., 0] - s * vectors[..., 1]
    out[..., 1] = s * vectors[..., 0] + c * vectors[..., 1]
    return out


def signed_cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def hand_has_points(hand: np.ndarray) -> bool:
    return bool(np.any(hand != 0))


def valid_joint_indices(hand: np.ndarray) -> np.ndarray:
    return np.where(np.any(hand != 0, axis=1))[0]


def palm_center(hand: np.ndarray) -> np.ndarray:
    return np.mean(hand[PALM_MCP_INDICES], axis=0).astype(np.float32)


def palm_axis(hand: np.ndarray) -> np.ndarray:
    wrist = hand[0]
    center = palm_center(hand)
    return normalize_vector(
        center - wrist,
        fallback=np.asarray([0.0, -1.0], dtype=np.float32),
    )


def palm_scale(hand: np.ndarray) -> float:
    wrist = hand[0]
    return float(
        np.mean(
            [
                np.linalg.norm(hand[idx] - wrist)
                for idx in PALM_MCP_INDICES
            ]
        )
    )


def hand_chirality_sign(hand: np.ndarray) -> float:
    wrist = hand[0]
    index_mcp = hand[5]
    pinky_mcp = hand[17]
    return signed_cross_2d(index_mcp - wrist, pinky_mcp - wrist)


def mirror_hand_x_about_wrist(hand: np.ndarray) -> np.ndarray:
    hand = np.asarray(hand, dtype=np.float32).copy()
    wrist_x = float(hand[0, 0])
    hand[:, 0] = 2.0 * wrist_x - hand[:, 0]
    return hand


def draw_hand(ax, all_keypts: np.ndarray, title: str, background=None):
    if background is not None:
        ax.imshow(background)

    for start in (0, 21):
        hand = all_keypts[start:start + 21]
        if not hand_has_points(hand):
            continue

        for a, b in HAND_CONNECTIONS:
            p1 = hand[a]
            p2 = hand[b]
            if np.all(p1 == 0) or np.all(p2 == 0):
                continue
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                linewidth=1.5,
            )

        valid = np.any(hand != 0, axis=1)
        ax.scatter(hand[valid, 0], hand[valid, 1], s=18)

        for i in np.where(valid)[0]:
            ax.text(
                hand[i, 0] + 2,
                hand[i, 1],
                str(start + i),
                fontsize=7,
            )

    ax.set_xlim(0, IMAGE_SIZE[0])
    ax.set_ylim(IMAGE_SIZE[1], 0)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(False)


# ============================================================
# FoundHand condition builders
# ============================================================

def make_pose_condition(
    keypts: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    keypts = np.asarray(keypts, dtype=np.float32)
    valid = check_keypoints_validity(keypts, IMAGE_SIZE)

    latent_keypts = scale_keypoint(
        keypts,
        IMAGE_SIZE,
        LATENT_SIZE,
    )

    heatmaps_np = keypoint_heatmap(
        latent_keypts,
        LATENT_SIZE,
        var=1.0,
    )
    heatmaps_np *= valid[:, None, None]

    heatmaps = torch.tensor(
        heatmaps_np,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    zero_mask = torch.zeros(
        (1, NUM_MASK, LATENT_SIZE[0], LATENT_SIZE[1]),
        dtype=torch.float32,
        device=device,
    )

    condition = torch.cat([heatmaps, zero_mask], dim=1)
    return condition


def make_ref_cond_inputs(
    img: np.ndarray,
    keypts: np.ndarray,
    hand_mask: np.ndarray,
    device: torch.device,
):
    image_transform = Compose(
        [
            ToTensor(),
            Resize(IMAGE_SIZE),
            Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                inplace=True,
            ),
        ]
    )

    image = image_transform(img).to(device)

    valid = check_keypoints_validity(keypts, IMAGE_SIZE)
    latent_keypts = scale_keypoint(
        keypts,
        IMAGE_SIZE,
        LATENT_SIZE,
    )

    heatmaps_np = keypoint_heatmap(
        latent_keypts,
        LATENT_SIZE,
        var=1.0,
    )
    heatmaps_np *= valid[:, None, None]

    heatmaps = torch.tensor(
        heatmaps_np,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    resized_mask = cv2.resize(
        hand_mask.astype(np.uint8),
        dsize=LATENT_SIZE,
        interpolation=cv2.INTER_NEAREST,
    )

    mask = torch.tensor(
        resized_mask,
        dtype=torch.float32,
        device=device,
    )[None, None, ...]

    return image.unsqueeze(0), heatmaps, mask


# ============================================================
# Reference-data helpers
# ============================================================

def load_reference_sequence(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_pose_from_sequence(sequence, frame: int) -> np.ndarray:
    if not (0 <= frame < len(sequence)):
        raise RuntimeError(
            f"Reference frame {frame} outside sequence length {len(sequence)}"
        )

    keypts = np.asarray(sequence[frame], dtype=np.float32)

    if keypts.shape != (42, 2):
        raise RuntimeError(
            f"Reference frame {frame}: expected (42,2), got {keypts.shape}"
        )

    return keypts


def resolve_reference():
    if not osp.isdir(REF_DATA_ROOT):
        raise FileNotFoundError(
            f"Reference data directory does not exist:\n{REF_DATA_ROOT}"
        )

    candidates = []
    for idx in PREFERRED_REF_IDXS:
        if idx not in candidates:
            candidates.append(idx)

    for pkl_path in sorted(glob(osp.join(REF_DATA_ROOT, "*.pkl"))):
        idx = osp.splitext(osp.basename(pkl_path))[0]
        if idx not in candidates:
            candidates.append(idx)

    for idx in candidates:
        pkl_path = osp.join(REF_DATA_ROOT, f"{idx}.pkl")
        image_dir = osp.join(REF_DATA_ROOT, idx)

        if not osp.isfile(pkl_path) or not osp.isdir(image_dir):
            continue

        try:
            sequence = load_reference_sequence(pkl_path)
        except Exception as exc:
            print(f"[reference] skipping {idx}: failed to read pkl: {exc}")
            continue

        available_frames = []
        for image_path in sorted(glob(osp.join(image_dir, "*.jpg"))):
            stem = osp.splitext(osp.basename(image_path))[0]
            if stem.isdigit():
                available_frames.append(int(stem))

        if not available_frames:
            continue

        frame_candidates = []
        if PREFERRED_REF_FRAME in available_frames:
            frame_candidates.append(PREFERRED_REF_FRAME)

        for frame in available_frames:
            if frame not in frame_candidates:
                frame_candidates.append(frame)

        for frame in frame_candidates:
            try:
                keypts = get_pose_from_sequence(sequence, frame)
            except Exception:
                continue

            if not np.any(keypts != 0):
                continue

            image_path = osp.join(image_dir, f"{frame:04d}.jpg")
            return idx, frame, image_path, pkl_path, sequence, keypts

    raise RuntimeError(
        f"Could not find a usable FoundHand reference pair under:\n{REF_DATA_ROOT}"
    )


# ============================================================
# Reference mask helpers
# ============================================================

def convex_hull_mask_from_keypoints(
    keypts: np.ndarray,
    pad: int = 12,
) -> np.ndarray:
    mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)
    valid = keypts[np.any(keypts != 0, axis=1)]

    if len(valid) < 3:
        return mask

    pts = np.round(valid).astype(np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 1)

    if pad > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * pad + 1, 2 * pad + 1),
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask.astype(np.uint8)


def sam_mask_from_keypoints(
    img: np.ndarray,
    keypts: np.ndarray,
) -> np.ndarray:
    try:
        from utils.segment_hoi import init_sam
    except Exception as exc:
        print(
            f"[mask] Could not import SAM helper ({exc}). "
            f"Falling back to convex hull mask."
        )
        return convex_hull_mask_from_keypoints(keypts)

    if not osp.isfile(SAM_PATH):
        print(
            f"[mask] SAM weight not found at {SAM_PATH}. "
            f"Falling back to convex hull mask."
        )
        return convex_hull_mask_from_keypoints(keypts)

    valid = keypts[np.any(keypts != 0, axis=1)]
    if len(valid) == 0:
        return np.zeros(IMAGE_SIZE, dtype=np.uint8)

    predictor = None

    try:
        predictor = init_sam(SAM_PATH)
        predictor.set_image(img)

        x0 = float(np.min(valid[:, 0]))
        y0 = float(np.min(valid[:, 1]))
        x1 = float(np.max(valid[:, 0]))
        y1 = float(np.max(valid[:, 1]))

        pad = 18.0
        box = np.array(
            [
                max(0.0, x0 - pad),
                max(0.0, y0 - pad),
                min(float(IMAGE_SIZE[0] - 1), x1 + pad),
                min(float(IMAGE_SIZE[1] - 1), y1 + pad),
            ],
            dtype=np.float32,
        )

        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=True,
        )

        best = int(np.argmax(scores))
        mask = masks[best].astype(np.uint8)

        if mask.sum() == 0:
            return convex_hull_mask_from_keypoints(keypts)

        return mask

    except Exception as exc:
        print(
            f"[mask] SAM prediction failed ({exc}). "
            f"Falling back to convex hull mask."
        )
        return convex_hull_mask_from_keypoints(keypts)

    finally:
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_reference_mask(
    img: np.ndarray,
    keypts: np.ndarray,
) -> np.ndarray:
    if MASK_MODE == "hull":
        return convex_hull_mask_from_keypoints(keypts)
    if MASK_MODE == "sam":
        return sam_mask_from_keypoints(img, keypts)
    raise ValueError(f"Unknown MASK_MODE={MASK_MODE!r}")


# ============================================================
# FSK coordinate conversion
# ============================================================

def convert_fsk_keypoints_to_pixels(
    keypoints: np.ndarray,
) -> np.ndarray:
    keypoints = np.asarray(keypoints, dtype=np.float32)

    raw_min = float(keypoints.min())
    raw_max = float(keypoints.max())

    print()
    print("Raw FSK coordinate range:", raw_min, "->", raw_max)

    if raw_max <= 1.001:
        print("Detected normalized [0,1] FSK coordinates.")
        result = keypoints.copy()
        result[..., 0] *= (IMAGE_SIZE[0] - 1)
        result[..., 1] *= (IMAGE_SIZE[1] - 1)
        return result

    print("Detected pixel FSK coordinates.")
    result = keypoints.copy()

    if result[..., 0].max() > 255.0 or result[..., 1].max() > 255.0:
        print("Detected old 0..256 pixel convention; rescaling to 0..255.")
        result[..., 0] *= ((IMAGE_SIZE[0] - 1) / IMAGE_SIZE[0])
        result[..., 1] *= ((IMAGE_SIZE[1] - 1) / IMAGE_SIZE[1])

    return result


# ============================================================
# Retarget methods
# ============================================================

def retarget_pose_to_reference_baseline(
    source_pose: np.ndarray,
    reference_pose: np.ndarray,
) -> np.ndarray:
    source_pose = np.asarray(source_pose, dtype=np.float32)
    reference_pose = np.asarray(reference_pose, dtype=np.float32)

    output = np.zeros((42, 2), dtype=np.float32)

    for start in (0, 21):
        side = "RIGHT" if start == 0 else "LEFT"
        source_hand = source_pose[start:start + 21].copy()
        reference_hand = reference_pose[start:start + 21].copy()

        if not hand_has_points(source_hand):
            continue

        if not hand_has_points(reference_hand):
            print(
                f"[baseline] WARNING: source {side} exists but reference {side} does not."
            )
            continue

        if BASELINE_MIRROR_X:
            source_hand = mirror_hand_x_about_wrist(source_hand)

        source_wrist = source_hand[0].copy()
        reference_wrist = reference_hand[0].copy()

        source_scale = float(
            np.linalg.norm(source_hand[9] - source_hand[0])
        )
        reference_scale = float(
            np.linalg.norm(reference_hand[9] - reference_hand[0])
        )

        if source_scale <= 1e-8 or reference_scale <= 1e-8:
            raise RuntimeError(f"[baseline] {side}: invalid hand scale.")

        ratio = reference_scale / source_scale

        output[start:start + 21] = (
            (source_hand - source_wrist) * ratio
            + reference_wrist
        )

        print(
            f"[baseline] {side}: "
            f"mirror_x={BASELINE_MIRROR_X}, "
            f"source_scale={source_scale:.2f}, "
            f"reference_scale={reference_scale:.2f}, "
            f"ratio={ratio:.3f}"
        )

    return output


def maybe_reflect_source_hand_for_kinematic(
    source_hand: np.ndarray,
    reference_hand: np.ndarray,
) -> tuple[np.ndarray, bool]:
    mode = KINEMATIC_REFLECTION_MODE.lower()
    reflected = False
    hand = source_hand.copy()

    if mode == "always":
        hand = mirror_hand_x_about_wrist(hand)
        reflected = True
    elif mode == "never":
        reflected = False
    elif mode == "auto":
        source_sign = hand_chirality_sign(hand)
        reference_sign = hand_chirality_sign(reference_hand)
        if source_sign * reference_sign < 0:
            hand = mirror_hand_x_about_wrist(hand)
            reflected = True
    else:
        raise ValueError(
            f"Unknown KINEMATIC_REFLECTION_MODE={KINEMATIC_REFLECTION_MODE!r}"
        )

    return hand, reflected


def retarget_pose_to_reference_kinematic(
    source_pose: np.ndarray,
    reference_pose: np.ndarray,
) -> np.ndarray:
    source_pose = np.asarray(source_pose, dtype=np.float32)
    reference_pose = np.asarray(reference_pose, dtype=np.float32)

    output = np.zeros((42, 2), dtype=np.float32)

    for start in (0, 21):
        side = "RIGHT" if start == 0 else "LEFT"
        source_hand = source_pose[start:start + 21].copy()
        reference_hand = reference_pose[start:start + 21].copy()

        if not hand_has_points(source_hand):
            continue

        if not hand_has_points(reference_hand):
            print(
                f"[kinematic] WARNING: source {side} exists but reference {side} does not."
            )
            continue

        source_hand, reflected = maybe_reflect_source_hand_for_kinematic(
            source_hand,
            reference_hand,
        )

        source_axis = palm_axis(source_hand)
        reference_axis = palm_axis(reference_hand)
        rot = wrap_angle(
            vector_angle(reference_axis) - vector_angle(source_axis)
        )

        source_centered = source_hand - source_hand[0]
        source_aligned = rotate_vectors(source_centered, rot) + reference_hand[0]

        new_hand = np.zeros((21, 2), dtype=np.float32)
        new_hand[0] = reference_hand[0]
        new_hand[PALM_MCP_INDICES] = reference_hand[PALM_MCP_INDICES]
        new_hand[1] = reference_hand[1]

        def build_chain(anchor_idx: int, chain: list[int]):
            parent = anchor_idx

            for child in chain:
                ref_len = float(
                    np.linalg.norm(reference_hand[child] - reference_hand[parent])
                )

                if ref_len <= 1e-8:
                    new_hand[child] = reference_hand[child]
                    parent = child
                    continue

                src_vec = source_aligned[child] - source_aligned[parent]
                ref_vec = reference_hand[child] - reference_hand[parent]

                src_dir = normalize_vector(src_vec, fallback=ref_vec)
                ref_dir = normalize_vector(
                    ref_vec,
                    fallback=np.asarray([0.0, -1.0], dtype=np.float32),
                )

                blended = normalize_vector(
                    KINEMATIC_DIRECTION_BLEND * src_dir
                    + (1.0 - KINEMATIC_DIRECTION_BLEND) * ref_dir,
                    fallback=ref_dir,
                )

                new_hand[child] = new_hand[parent] + ref_len * blended
                parent = child

        build_chain(1, THUMB_CHAIN[1:])
        for chain in FINGER_CHAINS:
            build_chain(chain[0], chain[1:])

        output[start:start + 21] = new_hand

        source_sign = hand_chirality_sign(source_hand)
        ref_sign = hand_chirality_sign(reference_hand)

        print(
            f"[kinematic] {side}: "
            f"reflection_mode={KINEMATIC_REFLECTION_MODE}, "
            f"reflected={reflected}, "
            f"rotation={np.degrees(rot):.2f} deg, "
            f"direction_blend={KINEMATIC_DIRECTION_BLEND:.2f}"
        )
        print(
            f"    chirality source={source_sign:.3f}, "
            f"reference={ref_sign:.3f}"
        )

    return output


def retarget_single_hand_angle(
    source_hand: np.ndarray,
    reference_hand: np.ndarray,
    side: str,
) -> np.ndarray:
    if ANGLE_MIRROR_X:
        source_hand = mirror_hand_x_about_wrist(source_hand)

    new_hand = np.zeros((21, 2), dtype=np.float32)

    # Preserve palm anatomy from reference.
    new_hand[PALM_ANCHOR_INDICES] = reference_hand[PALM_ANCHOR_INDICES]
    new_hand[1] = reference_hand[1]

    source_palm_ang = vector_angle(palm_axis(source_hand))
    ref_palm_ang = vector_angle(palm_axis(reference_hand))

    bend_limit = np.deg2rad(ANGLE_MAX_BEND_DEG)

    def bone_len(hand: np.ndarray, a: int, b: int) -> float:
        return float(np.linalg.norm(hand[b] - hand[a]))

    def rel_angles_for_chain(
        hand: np.ndarray,
        anchor: int,
        chain: list[int],
        palm_ang: float,
    ) -> tuple[list[float], list[float]]:
        pts = [anchor] + chain
        abs_angles = []
        rel_angles = []

        for i in range(len(pts) - 1):
            parent = pts[i]
            child = pts[i + 1]
            vec = hand[child] - hand[parent]
            ang = vector_angle(vec)
            abs_angles.append(ang)

        rel_angles.append(wrap_angle(abs_angles[0] - palm_ang))
        for i in range(1, len(abs_angles)):
            rel_angles.append(wrap_angle(abs_angles[i] - abs_angles[i - 1]))

        return abs_angles, rel_angles

    def reconstruct_chain(anchor: int, chain: list[int]):
        _, src_rel = rel_angles_for_chain(
            source_hand,
            anchor,
            chain,
            source_palm_ang,
        )

        _, ref_rel = rel_angles_for_chain(
            reference_hand,
            anchor,
            chain,
            ref_palm_ang,
        )

        new_rel0 = blend_angle(ref_rel[0], src_rel[0], ANGLE_BASE_BLEND)
        abs0 = ref_palm_ang + new_rel0
        abs_angles = [abs0]

        for j in range(1, len(src_rel)):
            bend = blend_angle(ref_rel[j], src_rel[j], ANGLE_BEND_BLEND)
            bend = float(np.clip(bend, -bend_limit, bend_limit))
            abs_angles.append(abs_angles[-1] + bend)

        parent = anchor
        for j, child in enumerate(chain):
            ref_length = bone_len(reference_hand, parent, child)

            if ref_length <= 1e-8:
                new_hand[child] = reference_hand[child]
            else:
                new_hand[child] = (
                    new_hand[parent]
                    + ref_length * unit_from_angle(abs_angles[j])
                )

            parent = child

        print(
            f"    [{side}] chain {anchor}->{chain[-1]} "
            f"ref_base_rel={np.degrees(ref_rel[0]):.1f} "
            f"src_base_rel={np.degrees(src_rel[0]):.1f} "
            f"new_base_rel={np.degrees(new_rel0):.1f}"
        )

    reconstruct_chain(1, THUMB_CHAIN[1:])
    for chain in FINGER_CHAINS:
        reconstruct_chain(chain[0], chain[1:])

    return new_hand


def retarget_pose_to_reference_angle(
    source_pose: np.ndarray,
    reference_pose: np.ndarray,
) -> np.ndarray:
    source_pose = np.asarray(source_pose, dtype=np.float32)
    reference_pose = np.asarray(reference_pose, dtype=np.float32)

    output = np.zeros((42, 2), dtype=np.float32)

    for start in (0, 21):
        side = "RIGHT" if start == 0 else "LEFT"
        source_hand = source_pose[start:start + 21].copy()
        reference_hand = reference_pose[start:start + 21].copy()

        if not hand_has_points(source_hand):
            continue

        if not hand_has_points(reference_hand):
            print(
                f"[angle] WARNING: source {side} exists but reference {side} does not."
            )
            continue

        output[start:start + 21] = retarget_single_hand_angle(
            source_hand,
            reference_hand,
            side,
        )

        print(
            f"[angle] {side}: "
            f"mirror_x={ANGLE_MIRROR_X}, "
            f"base_blend={ANGLE_BASE_BLEND:.2f}, "
            f"bend_blend={ANGLE_BEND_BLEND:.2f}, "
            f"bend_limit={ANGLE_MAX_BEND_DEG:.1f} deg"
        )

    return output


# ============================================================
# Figure output
# ============================================================

def save_debug_figure(
    ref_img: np.ndarray,
    ref_idx: str,
    ref_frame: int,
    ref_keypts: np.ndarray,
    target_title: str,
    target_keypts: np.ndarray,
    generated: np.ndarray | None = None,
):
    if generated is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax0, ax1, ax2 = axes
    else:
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        ax0, ax1, ax2, ax3 = axes

    ax0.imshow(ref_img)
    ax0.set_title(f"Reference\n{ref_idx} frame {ref_frame}")
    ax0.axis("off")

    draw_hand(ax1, ref_keypts, "Reference pose", background=ref_img)
    draw_hand(ax2, target_keypts, target_title)

    if generated is not None:
        ax3.imshow(generated)
        ax3.set_title(f"Generated\n{SAMPLING_STEPS} steps")
        ax3.axis("off")

    plt.tight_layout()
    plt.savefig(DEBUG_PATH, dpi=160)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("FoundHand FSK Single-Frame Reference-Conditioned Test")
    print("(Raw / Old Retarget / Kinematic / Angle)")
    print("=" * 70)
    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Device:", device)

    print()
    print("=" * 70)
    print("Checking files")
    print("=" * 70)

    required = [MODEL_PATH, VAE_PATH, POSE_PATH]
    if MASK_MODE == "sam":
        required.append(SAM_PATH)

    for path in required:
        print(path)
        if osp.exists(path):
            print("  OK")
        else:
            print("  MISSING")

    if not osp.isfile(POSE_PATH):
        raise FileNotFoundError(POSE_PATH)
    if not osp.isfile(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    if not osp.isfile(VAE_PATH):
        raise FileNotFoundError(VAE_PATH)

    print()
    print("=" * 70)
    print("Resolving reference data")
    print("=" * 70)

    ref_idx, ref_frame, ref_image_path, ref_pkl_path, ref_sequence, ref_keypts = resolve_reference()

    ref_img = io.imread(ref_image_path)
    if ref_img.ndim == 2:
        ref_img = np.stack([ref_img] * 3, axis=-1)
    ref_img = ref_img[..., :3]

    print("Reference sequence :", ref_idx)
    print("Reference frame    :", ref_frame)
    print("Reference image    :", ref_image_path)
    print("Reference pose pkl :", ref_pkl_path)
    print("Reference hands    :", describe_presence(ref_keypts))
    print("Reference image shape:", ref_img.shape)

    print()
    print("=" * 70)
    print("Loading FSK target pose")
    print("=" * 70)

    data = np.load(POSE_PATH)
    keypoints = data["keypoints"]
    frame_ids = data["frame_ids"]

    print("Keypoints shape:", keypoints.shape)
    print("Frame IDs shape:", frame_ids.shape)

    keypoints = convert_fsk_keypoints_to_pixels(keypoints)

    if FSK_FRAME_ID not in frame_ids.tolist():
        raise RuntimeError(
            f"FSK frame_id={FSK_FRAME_ID} not found in {frame_ids.tolist()}"
        )

    array_idx = int(np.where(frame_ids == FSK_FRAME_ID)[0][0])
    raw_target_keypts = keypoints[array_idx].astype(np.float32)
    raw_target_presence = describe_presence(raw_target_keypts)
    target_frame_id = int(frame_ids[array_idx])

    print()
    print("=" * 70)
    print("Selecting target pose")
    print("=" * 70)

    if TARGET_MODE == "fsk_raw":
        target_keypts = raw_target_keypts.copy()
        target_source = "FSK_RAW"

    elif TARGET_MODE == "fsk_retarget":
        target_keypts = retarget_pose_to_reference_baseline(
            raw_target_keypts,
            ref_keypts,
        )
        target_source = "FSK_RETARGET"

    elif TARGET_MODE == "fsk_kinematic":
        target_keypts = retarget_pose_to_reference_kinematic(
            raw_target_keypts,
            ref_keypts,
        )
        target_source = "FSK_KINEMATIC"

    elif TARGET_MODE == "fsk_angle":
        target_keypts = retarget_pose_to_reference_angle(
            raw_target_keypts,
            ref_keypts,
        )
        target_source = "FSK_ANGLE"

    else:
        raise ValueError(f"Unknown TARGET_MODE={TARGET_MODE!r}")

    valid_mask = check_keypoints_validity(target_keypts, IMAGE_SIZE)

    invalid_present_indices = [
        i
        for i in range(len(target_keypts))
        if np.any(target_keypts[i] != 0) and not bool(valid_mask[i])
    ]

    print("Raw FSK hands     :", raw_target_presence)
    print("Target source      :", target_source)
    print("Target frame       :", target_frame_id)
    print("Target hands       :", describe_presence(target_keypts))
    print("Reference hands    :", describe_presence(ref_keypts))
    print(
        "Valid target keypoints:",
        int(valid_mask.sum()),
        "/",
        len(valid_mask),
    )
    print("Invalid present keypoint indices:", invalid_present_indices)

    if not np.any(valid_mask):
        raise RuntimeError("Selected target contains no valid keypoints.")

    target_title = f"{target_source} target pose\nframe {target_frame_id}"

    if POSE_ONLY:
        save_debug_figure(
            ref_img=ref_img,
            ref_idx=ref_idx,
            ref_frame=ref_frame,
            ref_keypts=ref_keypts,
            target_title=target_title,
            target_keypts=target_keypts,
            generated=None,
        )

        print()
        print("=" * 70)
        print("POSE_ONLY SUCCESS")
        print("=" * 70)
        print("Saved diagnostic image:", DEBUG_PATH)
        return

    print()
    print("=" * 70)
    print("Building target condition")
    print("=" * 70)

    target_cond = make_pose_condition(target_keypts, device)
    print("target_cond:", target_cond.shape)

    print()
    print("=" * 70)
    print("Building reference mask")
    print("=" * 70)

    hand_mask = make_reference_mask(ref_img, ref_keypts)
    print("Reference mask shape:", hand_mask.shape)
    print("Reference mask pixels:", int(hand_mask.sum()))

    print()
    print("=" * 70)
    print("Loading diffusion")
    print("=" * 70)

    print("Sampling steps:", SAMPLING_STEPS)
    diffusion = create_diffusion(str(SAMPLING_STEPS))
    print("Diffusion ready.")

    print()
    print("=" * 70)
    print("Loading DiT")
    print("=" * 70)

    model = vit.DiT_XL_2(
        input_size=LATENT_SIZE[0],
        latent_dim=LATENT_DIM,
        in_channels=LATENT_DIM + NUM_KEYPOINTS + NUM_MASK,
        learn_sigma=True,
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    if "ema_state_dict" in checkpoint:
        state_dict = checkpoint["ema_state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        raise RuntimeError(
            "Checkpoint has no ema_state_dict or model_state_dict."
        )

    missing_keys, extra_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print("DiT missing keys:", missing_keys)
    print("DiT extra keys:", extra_keys)

    if missing_keys:
        raise RuntimeError(f"DiT has missing keys: {missing_keys}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("DiT ready.")

    del checkpoint
    del state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("Loading VAE")
    print("=" * 70)

    vae_checkpoint = torch.load(VAE_PATH, map_location=device)
    vae_state_dict = (
        vae_checkpoint["state_dict"]
        if "state_dict" in vae_checkpoint
        else vae_checkpoint
    )

    autoencoder = vqvae.create_model(3, 3, LATENT_DIM).to(device)
    vae_missing, vae_extra = autoencoder.load_state_dict(
        vae_state_dict,
        strict=False,
    )

    print("VAE missing keys:", vae_missing)
    print("VAE extra keys:", vae_extra)

    if vae_missing:
        raise RuntimeError(f"VAE has missing keys: {vae_missing}")

    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad_(False)

    print("VAE ready.")

    del vae_checkpoint
    del vae_state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("Building reference condition")
    print("=" * 70)

    ref_image_tensor, ref_heatmaps, ref_mask = make_ref_cond_inputs(
        ref_img,
        ref_keypts,
        hand_mask,
        device,
    )

    print("Reference image tensor:", ref_image_tensor.shape)
    print("Reference heatmaps:", ref_heatmaps.shape)
    print("Reference mask:", ref_mask.shape)

    reset_seed(REF_LATENT_SEED)
    with torch.inference_mode():
        ref_latent = LATENT_SCALING_FACTOR * autoencoder.encode(
            ref_image_tensor
        ).sample()

    print("Reference latent:", ref_latent.shape)

    src_ref_cond = torch.cat([ref_latent, ref_heatmaps, ref_mask], dim=1)
    print("src_ref_cond:", src_ref_cond.shape)

    print()
    print("=" * 70)
    print("Preparing diffusion input")
    print("=" * 70)

    reset_seed(Z_SEED)
    z = torch.randn(
        (1, LATENT_DIM, LATENT_SIZE[0], LATENT_SIZE[1]),
        dtype=torch.float32,
        device=device,
    )
    z_cfg = torch.cat([z, z], dim=0)

    print("z:", z.shape)
    print("z_cfg:", z_cfg.shape)

    target_cond_cfg = torch.cat(
        [target_cond, torch.zeros_like(target_cond)],
        dim=0,
    )
    ref_cond_cfg = torch.cat(
        [src_ref_cond, torch.zeros_like(src_ref_cond)],
        dim=0,
    )
    nvs = torch.tensor([0, 2], dtype=torch.int32, device=device)

    model_kwargs = dict(
        target_cond=target_cond_cfg,
        ref_cond=ref_cond_cfg,
        nvs=nvs,
        cfg_scale=CFG_SCALE,
    )

    print("target_cond_cfg:", target_cond_cfg.shape)
    print("ref_cond_cfg:", ref_cond_cfg.shape)
    print("nvs:", nvs)

    print()
    print("=" * 70)
    print("Running diffusion")
    print("=" * 70)

    print(f"Steps = {SAMPLING_STEPS}")
    print("Reference =", f"{ref_idx} frame {ref_frame}")
    print("Target source =", target_source)
    print("Target frame =", target_frame_id)

    reset_seed(SAMPLING_NOISE_SEED)
    with torch.inference_mode():
        samples_cfg = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z_cfg.shape,
            z_cfg,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            ref_conds=[src_ref_cond],
            progress=True,
            device=device,
        )

    samples, _ = samples_cfg.chunk(2, dim=0)

    print()
    print("Sampling complete.")
    print("samples:", samples.shape)

    print()
    print("=" * 70)
    print("Decoding image with VAE")
    print("=" * 70)

    with torch.inference_mode():
        decoded = autoencoder.decode(samples / LATENT_SCALING_FACTOR)

    decoded = torch.clamp(decoded, min=-1.0, max=1.0)
    image = ((decoded + 1.0) / 2.0 * 255.0)
    image = (
        image.permute(0, 2, 3, 1)
        .cpu()
        .numpy()
        .round()
        .clip(0, 255)
        .astype(np.uint8)
    )

    generated = image[0]

    io.imsave(OUTPUT_PATH, generated)

    save_debug_figure(
        ref_img=ref_img,
        ref_idx=ref_idx,
        ref_frame=ref_frame,
        ref_keypts=ref_keypts,
        target_title=target_title,
        target_keypts=target_keypts,
        generated=generated,
    )

    print("Generated image:", generated.shape)
    print("Saved generated image:", OUTPUT_PATH)
    print("Saved diagnostic image:", DEBUG_PATH)

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)
    print("Reference:", f"{ref_idx} frame {ref_frame}")
    print("Target source:", target_source)
    print("Target frame:", target_frame_id)
    print("Target hands:", describe_presence(target_keypts))
    print("Sampling steps:", SAMPLING_STEPS)


if __name__ == "__main__":
    main()