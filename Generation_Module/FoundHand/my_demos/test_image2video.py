from __future__ import annotations

import gc
import json
import os
import os.path as osp
import pickle
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from glob import glob

import cv2
import numpy as np
import skimage.io as io
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


# ============================================================
# FoundHand imports
# ============================================================

CURRENT_DIR = osp.dirname(osp.abspath(__file__))
FOUNDHAND_ROOT = osp.dirname(CURRENT_DIR)

if FOUNDHAND_ROOT not in sys.path:
    sys.path.insert(0, FOUNDHAND_ROOT)

from models import vqvae
from models import vit
from diffusion import create_diffusion
from utils.segment_hoi import init_sam
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

LATENT_SCALING_FACTOR = 0.18215
CFG_SCALE = 2.5

# 100 was the best practical single-frame baseline in our tests.
SAMPLING_STEPS = 100

# Current best FSK -> FoundHand adapter.
BASELINE_MIRROR_X = True

# ------------------------------------------------------------
# Which recovered FSK frames should be generated?
# ------------------------------------------------------------
#
# First temporal sanity test:
#   4, 5, 6, 7 were already verified independently.
#
FSK_FRAME_IDS = list(range(20))

# If you later want to use every usable frame in the NPZ,
# set this True. Frames with no compatible hand are skipped.
USE_ALL_USABLE_FRAMES = True

# ------------------------------------------------------------
# Pose interpolation
# ------------------------------------------------------------
#
# Keep 0 for the first video test.
# Example:
#   1 -> add one midpoint pose between each pair
#   3 -> add three in-between poses
#
INTERPOLATION_STEPS_BETWEEN = 0

# Output playback FPS. This does NOT change the diffusion workload.
VIDEO_FPS = 1.8

# ------------------------------------------------------------
# Temporal consistency controls
# ------------------------------------------------------------

# Reuse the exact same initial diffusion z for every frame.
# This is intentionally ON for the first temporal test because
# target pose changes while the appearance/noise basis stays fixed.
REUSE_INITIAL_Z = True

# Reuse the same diffusion RNG state at the start of each frame.
SAME_SAMPLING_NOISE_EACH_FRAME = True

# EXPERIMENTAL.
#
# False:
#   every target frame uses the same official reference condition only.
#   This is the safest first video baseline and matches the already
#   validated single-frame invocation.
#
# True:
#   previous generated-frame conditions are appended to ref_conds.
#   This is provided for temporal experiments, but because the exact
#   FoundHand notebook temporal-reference policy is not present in the
#   supplied source material, test False first.
#
USE_PREVIOUS_GENERATED_REFS = False

# How many generated frames to keep as extra temporal refs when the
# experimental option above is enabled.
MAX_PREVIOUS_GENERATED_REFS = 1

# ------------------------------------------------------------
# Randomness
# ------------------------------------------------------------

SEED = 1234
REF_LATENT_SEED = SEED + 1
INITIAL_Z_SEED = SEED + 2
SAMPLING_NOISE_SEED = SEED + 3

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

POSE_PATH = r"D:\Project\2D-Hand\Generation_Module\FoundHand\test_result\result_npz\v3_recovered_pose.npz"

# ------------------------------------------------------------
# Orientation-aware personal reference bank
# ------------------------------------------------------------
#
# First validate one side at a time. This avoids assuming an unverified
# FoundHand API for combining two independent appearance references in one
# diffusion sample.
#
# Change to "LEFT" after the RIGHT test passes.
TARGET_HAND = "RIGHT"

ORIENTATION_UNKNOWN = 0
ORIENTATION_PALM = 1
ORIENTATION_BACK = 2

OFFICIAL_REF_SEQUENCE = "IMG_1087"
OFFICIAL_REF_FRAME = 6

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

SAM_PATH = osp.join(
    FOUNDHAND_ROOT,
    "weights",
    "sam_vit_h_4b8939.pth",
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

RUN_ROOT = osp.join(
    FOUNDHAND_ROOT,
    "my_demos",
    "image2video_runs",
)


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
# Generic helpers
# ============================================================

def reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hand_presence(
    keypts: np.ndarray,
) -> tuple[bool, bool]:

    right = bool(
        np.any(keypts[:21] != 0)
    )

    left = bool(
        np.any(keypts[21:42] != 0)
    )

    return right, left


def describe_presence(
    keypts: np.ndarray,
) -> str:

    right, left = hand_presence(
        keypts
    )

    if right and left:
        return "RIGHT+LEFT"

    if right:
        return "RIGHT"

    if left:
        return "LEFT"

    return "NONE"


def convert_fsk_keypoints_to_pixels(
    keypoints: np.ndarray,
) -> np.ndarray:

    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    raw_min = float(
        keypoints.min()
    )

    raw_max = float(
        keypoints.max()
    )

    print(
        "Raw FSK coordinate range:",
        raw_min,
        "->",
        raw_max,
    )

    if raw_max <= 1.001:

        print(
            "Detected normalized [0,1] coordinates."
        )

        result = (
            keypoints.copy()
        )

        result[..., 0] *= (
            IMAGE_SIZE[0] - 1
        )

        result[..., 1] *= (
            IMAGE_SIZE[1] - 1
        )

        return result

    print(
        "Detected pixel coordinates."
    )

    result = (
        keypoints.copy()
    )

    if (
        result[..., 0].max() > 255.0
        or
        result[..., 1].max() > 255.0
    ):

        print(
            "Detected old 0..256 convention; "
            "rescaling to 0..255."
        )

        result[..., 0] *= (
            (IMAGE_SIZE[0] - 1)
            / IMAGE_SIZE[0]
        )

        result[..., 1] *= (
            (IMAGE_SIZE[1] - 1)
            / IMAGE_SIZE[1]
        )

    return result


# ============================================================
# Current best retarget: mirror + scale + translation
# ============================================================

def mirror_hand_x_about_wrist(
    hand: np.ndarray,
) -> np.ndarray:

    hand = np.asarray(
        hand,
        dtype=np.float32,
    ).copy()

    wrist_x = float(
        hand[0, 0]
    )

    hand[:, 0] = (
        2.0 * wrist_x
        - hand[:, 0]
    )

    return hand


def retarget_pose_to_reference(
    source_pose: np.ndarray,
    reference_pose: np.ndarray,
) -> np.ndarray:
    """
    Current best baseline established by the single-frame tests.

    FoundHand:
        slot 0 = RIGHT  (0:21)
        slot 1 = LEFT   (21:42)

    For each source hand that also exists in the reference:
        1. horizontal mirror about source wrist
        2. scale by wrist -> middle MCP (0 -> 9)
        3. translate wrist to reference wrist

    If the reference contains only RIGHT, any source LEFT hand is ignored.
    """

    source_pose = np.asarray(
        source_pose,
        dtype=np.float32,
    )

    reference_pose = np.asarray(
        reference_pose,
        dtype=np.float32,
    )

    output = np.zeros(
        (42, 2),
        dtype=np.float32,
    )

    for start in (
        0,
        21,
    ):

        side = (
            "RIGHT"
            if start == 0
            else "LEFT"
        )

        source_hand = (
            source_pose[
                start:start + 21
            ].copy()
        )

        reference_hand = (
            reference_pose[
                start:start + 21
            ].copy()
        )

        source_present = bool(
            np.any(
                source_hand != 0
            )
        )

        reference_present = bool(
            np.any(
                reference_hand != 0
            )
        )

        if not source_present:
            continue

        if not reference_present:

            print(
                f"[retarget] {side}: "
                f"source exists but reference "
                f"does not; dropping this hand."
            )

            continue

        if BASELINE_MIRROR_X:

            source_hand = (
                mirror_hand_x_about_wrist(
                    source_hand
                )
            )

        source_wrist = (
            source_hand[0].copy()
        )

        reference_wrist = (
            reference_hand[0].copy()
        )

        source_scale = float(
            np.linalg.norm(
                source_hand[9]
                - source_hand[0]
            )
        )

        reference_scale = float(
            np.linalg.norm(
                reference_hand[9]
                - reference_hand[0]
            )
        )

        if (
            source_scale <= 1e-8
            or
            reference_scale <= 1e-8
        ):

            raise RuntimeError(
                f"[retarget] {side}: "
                f"invalid wrist->middle-MCP scale."
            )

        ratio = (
            reference_scale
            / source_scale
        )

        output[
            start:start + 21
        ] = (
            (
                source_hand
                - source_wrist
            )
            * ratio
            + reference_wrist
        )

        print(
            f"[retarget] {side}: "
            f"mirror={BASELINE_MIRROR_X}, "
            f"ratio={ratio:.3f}"
        )

    return output


# ============================================================
# Reference loading
# ============================================================

def load_reference_sequence(
    pkl_path: str,
):

    with open(
        pkl_path,
        "rb",
    ) as f:

        return pickle.load(
            f
        )


def get_pose_from_sequence(
    sequence,
    frame: int,
) -> np.ndarray:

    if not (
        0
        <= frame
        < len(sequence)
    ):

        raise RuntimeError(
            f"Reference frame {frame} "
            f"is outside sequence length "
            f"{len(sequence)}."
        )

    keypts = np.asarray(
        sequence[frame],
        dtype=np.float32,
    )

    if keypts.shape != (
        42,
        2,
    ):

        raise RuntimeError(
            f"Expected reference pose "
            f"(42,2), got "
            f"{keypts.shape}"
        )

    return keypts


def resolve_reference():
    """
    Resolve one official FoundHand appearance/pose reference.
    """

    if not osp.isdir(
        REF_DATA_ROOT
    ):

        raise FileNotFoundError(
            f"Missing reference root:\n"
            f"{REF_DATA_ROOT}"
        )

    candidates = []

    for idx in (
        PREFERRED_REF_IDXS
    ):

        if idx not in candidates:

            candidates.append(
                idx
            )

    for pkl_path in sorted(
        glob(
            osp.join(
                REF_DATA_ROOT,
                "*.pkl",
            )
        )
    ):

        idx = osp.splitext(
            osp.basename(
                pkl_path
            )
        )[0]

        if idx not in candidates:

            candidates.append(
                idx
            )

    for idx in candidates:

        pkl_path = osp.join(
            REF_DATA_ROOT,
            f"{idx}.pkl",
        )

        image_dir = osp.join(
            REF_DATA_ROOT,
            idx,
        )

        if not osp.isfile(
            pkl_path
        ):

            continue

        if not osp.isdir(
            image_dir
        ):

            continue

        sequence = (
            load_reference_sequence(
                pkl_path
            )
        )

        available_frames = []

        for image_path in sorted(
            glob(
                osp.join(
                    image_dir,
                    "*.jpg",
                )
            )
        ):

            stem = osp.splitext(
                osp.basename(
                    image_path
                )
            )[0]

            if stem.isdigit():

                available_frames.append(
                    int(stem)
                )

        if not available_frames:

            continue

        frame_candidates = []

        if (
            PREFERRED_REF_FRAME
            in available_frames
        ):

            frame_candidates.append(
                PREFERRED_REF_FRAME
            )

        for frame in (
            available_frames
        ):

            if (
                frame
                not in frame_candidates
            ):

                frame_candidates.append(
                    frame
                )

        for frame in (
            frame_candidates
        ):

            keypts = (
                get_pose_from_sequence(
                    sequence,
                    frame,
                )
            )

            if not np.any(
                keypts != 0
            ):

                continue

            image_path = osp.join(
                image_dir,
                f"{frame:04d}.jpg",
            )

            return (
                idx,
                frame,
                image_path,
                pkl_path,
                sequence,
                keypts,
            )

    raise RuntimeError(
        "Could not resolve FoundHand reference."
    )



# ============================================================
# Exact official IMG_1087 reference
# ============================================================

def orientation_name(code: int) -> str:
    if int(code) == ORIENTATION_PALM:
        return "PALM"
    if int(code) == ORIENTATION_BACK:
        return "BACK"
    return "UNKNOWN"


def load_official_reference_exact(
    sequence_id: str,
    frame: int,
) -> dict:
    """
    Load one exact official FoundHand reference image + pose.

    Expected layout:
      test_data/iphone_video/IMG_1087.pkl
      test_data/iphone_video/IMG_1087/0006.jpg
    """

    pkl_path = osp.join(
        REF_DATA_ROOT,
        f"{sequence_id}.pkl",
    )

    image_path = osp.join(
        REF_DATA_ROOT,
        sequence_id,
        f"{frame:04d}.jpg",
    )

    if not osp.isfile(pkl_path):
        raise FileNotFoundError(pkl_path)

    if not osp.isfile(image_path):
        raise FileNotFoundError(image_path)

    sequence = load_reference_sequence(
        pkl_path
    )

    keypts = get_pose_from_sequence(
        sequence,
        frame,
    )

    if not np.any(keypts != 0):
        raise RuntimeError(
            f"Official reference {sequence_id} frame {frame} has no hand."
        )

    image = io.imread(
        image_path
    )

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    return {
        "sequence_id": sequence_id,
        "frame": int(frame),
        "image_path": image_path,
        "pose_path": pkl_path,
        "image": image,
        "keypoints": keypts,
    }


# ============================================================
# FoundHand conditions
# ============================================================

def image_to_model_tensor(
    img: np.ndarray,
    device: torch.device,
) -> torch.Tensor:

    transform = Compose([
        ToTensor(),
        Resize(
            IMAGE_SIZE
        ),
        Normalize(
            mean=[
                0.5,
                0.5,
                0.5,
            ],
            std=[
                0.5,
                0.5,
                0.5,
            ],
            inplace=True,
        ),
    ])

    return (
        transform(
            img
        )
        .to(
            device
        )
        .unsqueeze(
            0
        )
    )


def pose_heatmaps(
    keypts: np.ndarray,
    device: torch.device,
) -> torch.Tensor:

    keypts = np.asarray(
        keypts,
        dtype=np.float32,
    )

    valid = (
        check_keypoints_validity(
            keypts,
            IMAGE_SIZE,
        )
    )

    latent_keypts = (
        scale_keypoint(
            keypts,
            IMAGE_SIZE,
            LATENT_SIZE,
        )
    )

    heatmaps_np = (
        keypoint_heatmap(
            latent_keypts,
            LATENT_SIZE,
            var=1.0,
        )
    )

    heatmaps_np *= (
        valid[:, None, None]
    )

    return torch.tensor(
        heatmaps_np,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(
        0
    )


def pose_mask_from_keypoints(
    keypts: np.ndarray,
) -> np.ndarray:
    """
    Simple mask for a generated temporal reference.
    This does NOT replace the official SAM mask for the fixed source image.
    """

    mask = np.zeros(
        IMAGE_SIZE,
        dtype=np.uint8,
    )

    for start in (
        0,
        21,
    ):

        hand = keypts[
            start:start + 21
        ]

        valid = hand[
            np.any(
                hand != 0,
                axis=1,
            )
        ]

        if len(
            valid
        ) < 3:

            continue

        pts = np.round(
            valid
        ).astype(
            np.int32
        )

        hull = cv2.convexHull(
            pts
        )

        cv2.fillConvexPoly(
            mask,
            hull,
            1,
        )

    # A small dilation makes the mask less brittle.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            13,
            13,
        ),
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )

    return mask


def latent_mask_tensor(
    mask_256: np.ndarray,
    device: torch.device,
) -> torch.Tensor:

    resized = cv2.resize(
        mask_256.astype(
            np.uint8
        ),
        dsize=LATENT_SIZE,
        interpolation=(
            cv2.INTER_NEAREST
        ),
    )

    return torch.tensor(
        resized,
        dtype=torch.float32,
        device=device,
    )[None, None, ...]


def build_reference_condition(
    image_rgb: np.ndarray,
    keypts: np.ndarray,
    mask_256: np.ndarray,
    autoencoder,
    device: torch.device,
    latent_seed: int | None = None,
) -> torch.Tensor:
    """
    Return:
        [1, 47, 32, 32]
        = VAE latent(4) + pose heatmaps(42) + mask(1)
    """

    image_tensor = (
        image_to_model_tensor(
            image_rgb,
            device,
        )
    )

    heatmaps = (
        pose_heatmaps(
            keypts,
            device,
        )
    )

    mask_tensor = (
        latent_mask_tensor(
            mask_256,
            device,
        )
    )

    if latent_seed is not None:

        reset_seed(
            latent_seed
        )

    with torch.inference_mode():

        latent = (
            LATENT_SCALING_FACTOR
            * autoencoder.encode(
                image_tensor
            ).sample()
        )

    cond = torch.cat(
        [
            latent,
            heatmaps,
            mask_tensor,
        ],
        dim=1,
    )

    expected = (
        1,
        LATENT_DIM
        + NUM_KEYPOINTS
        + NUM_MASK,
        LATENT_SIZE[0],
        LATENT_SIZE[1],
    )

    if tuple(
        cond.shape
    ) != expected:

        raise RuntimeError(
            f"Bad reference condition shape: "
            f"{cond.shape}, expected {expected}"
        )

    return cond


def build_target_condition(
    keypts: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    [1,43,32,32] = pose(42) + zero mask(1)
    """

    heatmaps = (
        pose_heatmaps(
            keypts,
            device,
        )
    )

    zero_mask = torch.zeros(
        (
            1,
            1,
            LATENT_SIZE[0],
            LATENT_SIZE[1],
        ),
        dtype=torch.float32,
        device=device,
    )

    return torch.cat(
        [
            heatmaps,
            zero_mask,
        ],
        dim=1,
    )


# ============================================================
# SAM mask for the fixed reference image
# ============================================================

def make_sam_mask(
    ref_img: np.ndarray,
    ref_keypts: np.ndarray,
) -> np.ndarray:
    """
    SAM mask using the same wrist-point strategy as the previously
    successful fixed official-reference baseline.
    """

    predictor = init_sam(
        ckpt_path=SAM_PATH
    )

    predictor.set_image(
        ref_img
    )

    right_present, left_present = hand_presence(
        ref_keypts
    )

    input_points = []

    if right_present:
        input_points.append(
            ref_keypts[0]
        )

    if left_present:
        input_points.append(
            ref_keypts[21]
        )

    if not input_points:
        raise RuntimeError(
            "Reference pose has no hand."
        )

    input_point = np.asarray(
        input_points,
        dtype=np.float32,
    )

    input_label = np.ones(
        len(input_points),
        dtype=np.int32,
    )

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    hand_mask = masks[0].astype(
        np.uint8
    )

    del predictor
    del masks

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if hand_mask.sum() == 0:
        raise RuntimeError(
            "SAM produced an empty mask."
        )

    return hand_mask


# ============================================================
# Sequence preparation
# ============================================================

def interpolate_pose_pair(
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    num_between: int,
) -> list[np.ndarray]:
    """
    Return poses strictly BETWEEN a and b.
    Only intended after both poses have already been retargeted into
    the same FoundHand reference coordinate system.
    """

    output = []

    if num_between <= 0:

        return output

    for j in range(
        1,
        num_between + 1,
    ):

        alpha = (
            j
            / (
                num_between + 1
            )
        )

        pose = np.zeros(
            (42, 2),
            dtype=np.float32,
        )

        for start in (
            0,
            21,
        ):

            a = pose_a[
                start:start + 21
            ]

            b = pose_b[
                start:start + 21
            ]

            a_present = bool(
                np.any(
                    a != 0
                )
            )

            b_present = bool(
                np.any(
                    b != 0
                )
            )

            if (
                a_present
                and
                b_present
            ):

                pose[
                    start:start + 21
                ] = (
                    (
                        1.0
                        - alpha
                    )
                    * a
                    + alpha
                    * b
                )

            elif a_present:

                pose[
                    start:start + 21
                ] = a

            elif b_present:

                pose[
                    start:start + 21
                ] = b

        output.append(
            pose
        )

    return output


def prepare_target_sequence(
    keypoints_pixel: np.ndarray,
    frame_ids: np.ndarray,
    hand_present: np.ndarray,
    orientations: np.ndarray,
    orientation_valid: np.ndarray,
    canonical_reference_pose: np.ndarray,
) -> list[dict]:
    """
    Prepare an orientation-aware target sequence for ONE hand.

    Geometry is always retargeted into the PALM reference coordinate system.
    PALM/BACK only selects the appearance reference condition later. This
    prevents reference switching from moving/scaling the target hand.
    """

    if TARGET_HAND == "RIGHT":
        target_slot = 0
        target_start = 0
    elif TARGET_HAND == "LEFT":
        target_slot = 1
        target_start = 21
    else:
        raise ValueError(
            f"Unsupported TARGET_HAND={TARGET_HAND!r}"
        )

    id_to_index = {
        int(frame_id): i
        for i, frame_id in enumerate(frame_ids)
    }

    if USE_ALL_USABLE_FRAMES:
        requested = [
            int(x)
            for x in frame_ids
        ]
    else:
        requested = list(
            FSK_FRAME_IDS
        )

    base_items = []
    previous_valid_orientation = None

    for frame_id in requested:
        if frame_id not in id_to_index:
            print(
                f"[sequence] frame {frame_id} not found; skipping."
            )
            continue

        row = id_to_index[frame_id]

        if not bool(hand_present[row, target_slot]):
            print(
                f"[sequence] frame {frame_id}: "
                f"{TARGET_HAND} absent; skipping."
            )
            continue

        raw_pose = keypoints_pixel[row].copy()

        # Single-hand validation: explicitly remove the other hand.
        if target_start == 0:
            raw_pose[21:42] = 0
        else:
            raw_pose[0:21] = 0

        ori = int(orientations[row, target_slot])
        ori_is_valid = bool(orientation_valid[row, target_slot])

        if ori_is_valid and ori in (
            ORIENTATION_PALM,
            ORIENTATION_BACK,
        ):
            previous_valid_orientation = ori
        elif previous_valid_orientation is not None:
            print(
                f"[sequence] frame {frame_id}: "
                f"orientation UNKNOWN; holding previous "
                f"{orientation_name(previous_valid_orientation)}."
            )
            ori = previous_valid_orientation
        else:
            # A first-frame UNKNOWN should be rare with the stabilized sender.
            # PALM is the conservative bootstrap reference.
            print(
                f"[sequence] frame {frame_id}: "
                "orientation UNKNOWN with no history; defaulting to PALM."
            )
            ori = ORIENTATION_PALM
            previous_valid_orientation = ori

        retargeted = retarget_pose_to_reference(
            raw_pose,
            canonical_reference_pose,
        )

        valid = check_keypoints_validity(
            retargeted,
            IMAGE_SIZE,
        )

        if valid.sum() == 0:
            print(
                f"[sequence] frame {frame_id} has no compatible "
                f"retargeted {TARGET_HAND} hand; skipping."
            )
            continue

        base_items.append(
            {
                "source_frame_id": frame_id,
                "label": f"fsk_{frame_id}",
                "pose": retargeted,
                "orientation": int(ori),
                "orientation_name": orientation_name(ori),
            }
        )

    if not base_items:
        raise RuntimeError(
            "No usable orientation-aware FSK target frames."
        )

    if INTERPOLATION_STEPS_BETWEEN <= 0:
        return base_items

    sequence = []

    for i, current in enumerate(base_items):
        sequence.append(current)

        if i == len(base_items) - 1:
            continue

        nxt = base_items[i + 1]

        between = interpolate_pose_pair(
            current["pose"],
            nxt["pose"],
            INTERPOLATION_STEPS_BETWEEN,
        )

        for j, pose in enumerate(
            between,
            start=1,
        ):
            alpha = j / (
                INTERPOLATION_STEPS_BETWEEN + 1
            )

            # If the stable state changes across this interval, switch the
            # appearance reference at the halfway point.
            if current["orientation"] == nxt["orientation"]:
                ori = current["orientation"]
            else:
                ori = (
                    current["orientation"]
                    if alpha < 0.5
                    else nxt["orientation"]
                )

            sequence.append(
                {
                    "source_frame_id": None,
                    "label": (
                        f"interp_"
                        f"{current['source_frame_id']}_"
                        f"{nxt['source_frame_id']}_"
                        f"{j}"
                    ),
                    "pose": pose,
                    "orientation": int(ori),
                    "orientation_name": orientation_name(ori),
                }
            )

    return sequence


# ============================================================
# Model loading
# ============================================================

def load_models(
    device: torch.device,
):

    print()
    print("=" * 70)
    print("Loading diffusion")
    print("=" * 70)

    diffusion = (
        create_diffusion(
            str(
                SAMPLING_STEPS
            )
        )
    )

    print(
        "Diffusion ready."
    )

    print()
    print("=" * 70)
    print("Loading DiT")
    print("=" * 70)

    model = vit.DiT_XL_2(
        input_size=(
            LATENT_SIZE[0]
        ),
        latent_dim=LATENT_DIM,
        in_channels=(
            LATENT_DIM
            + NUM_KEYPOINTS
            + NUM_MASK
        ),
        learn_sigma=True,
    ).to(
        device
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    if (
        "ema_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "ema_state_dict"
            ]
        )

    elif (
        "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        raise RuntimeError(
            "Checkpoint has no "
            "ema_state_dict or model_state_dict."
        )

    missing, extra = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    print(
        "DiT missing keys:",
        missing,
    )

    print(
        "DiT extra keys:",
        extra,
    )

    if missing:

        raise RuntimeError(
            f"DiT missing keys: "
            f"{missing}"
        )

    model.eval()

    for parameter in (
        model.parameters()
    ):

        parameter.requires_grad_(
            False
        )

    del checkpoint
    del state_dict

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    print(
        "DiT ready."
    )

    print()
    print("=" * 70)
    print("Loading VAE")
    print("=" * 70)

    vae_checkpoint = (
        torch.load(
            VAE_PATH,
            map_location=device,
        )
    )

    if (
        "state_dict"
        in vae_checkpoint
    ):

        vae_state = (
            vae_checkpoint[
                "state_dict"
            ]
        )

    else:

        vae_state = (
            vae_checkpoint
        )

    autoencoder = (
        vqvae.create_model(
            3,
            3,
            LATENT_DIM,
        ).to(
            device
        )
    )

    missing, extra = (
        autoencoder.load_state_dict(
            vae_state,
            strict=False,
        )
    )

    print(
        "VAE missing keys:",
        missing,
    )

    print(
        "VAE extra keys:",
        extra,
    )

    if missing:

        raise RuntimeError(
            f"VAE missing keys: "
            f"{missing}"
        )

    autoencoder.eval()

    for parameter in (
        autoencoder.parameters()
    ):

        parameter.requires_grad_(
            False
        )

    del vae_checkpoint
    del vae_state

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    print(
        "VAE ready."
    )

    return (
        diffusion,
        model,
        autoencoder,
    )


# ============================================================
# One target-frame generation
# ============================================================

def generate_one_frame(
    *,
    target_pose: np.ndarray,
    frame_index: int,
    fixed_ref_cond: torch.Tensor,
    temporal_ref_conds: list[torch.Tensor],
    base_z: torch.Tensor,
    diffusion,
    model,
    autoencoder,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:

    target_cond = (
        build_target_condition(
            target_pose,
            device,
        )
    )

    target_cond_cfg = torch.cat(
        [
            target_cond,
            torch.zeros_like(
                target_cond
            ),
        ],
        dim=0,
    )

    fixed_ref_cond_cfg = (
        torch.cat(
            [
                fixed_ref_cond,
                torch.zeros_like(
                    fixed_ref_cond
                ),
            ],
            dim=0,
        )
    )

    nvs = torch.tensor(
        [
            0,
            2,
        ],
        dtype=torch.int,
        device=device,
    )

    model_kwargs = dict(
        target_cond=target_cond_cfg,
        ref_cond=fixed_ref_cond_cfg,
        nvs=nvs,
        cfg_scale=CFG_SCALE,
    )

    if REUSE_INITIAL_Z:

        z = (
            base_z.clone()
        )

    else:

        reset_seed(
            INITIAL_Z_SEED
            + frame_index
        )

        z = torch.randn(
            (
                1,
                LATENT_DIM,
                LATENT_SIZE[0],
                LATENT_SIZE[1],
            ),
            dtype=torch.float32,
            device=device,
        )

    z_cfg = torch.cat(
        [
            z,
            z,
        ],
        dim=0,
    )

    ref_conds = [
        fixed_ref_cond
    ]

    if (
        USE_PREVIOUS_GENERATED_REFS
        and temporal_ref_conds
    ):

        ref_conds.extend(
            temporal_ref_conds
        )

    if SAME_SAMPLING_NOISE_EACH_FRAME:

        reset_seed(
            SAMPLING_NOISE_SEED
        )

    else:

        reset_seed(
            SAMPLING_NOISE_SEED
            + frame_index
        )

    with torch.inference_mode():

        samples_cfg = (
            diffusion.p_sample_loop(
                model.forward_with_cfg,
                z_cfg.shape,
                z_cfg,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                ref_conds=ref_conds,
                progress=True,
                device=device,
            )
        )

    if (
        samples_cfg.ndim != 4
        or samples_cfg.shape[0] != 2
    ):

        raise RuntimeError(
            f"Unexpected diffusion output "
            f"shape: {samples_cfg.shape}"
        )

    samples, _ = (
        samples_cfg.chunk(
            2,
            dim=0,
        )
    )

    with torch.inference_mode():

        decoded = (
            autoencoder.decode(
                samples
                / LATENT_SCALING_FACTOR
            )
        )

    decoded = torch.clamp(
        decoded,
        min=-1.0,
        max=1.0,
    )

    image = (
        (decoded + 1.0)
        / 2.0
        * 255.0
    )

    image = (
        image
        .permute(
            0,
            2,
            3,
            1,
        )
        .cpu()
        .numpy()
        .round()
        .clip(
            0,
            255,
        )
        .astype(
            np.uint8
        )
    )

    generated = (
        image[0]
    )

    temporal_cond = None

    if USE_PREVIOUS_GENERATED_REFS:
        # Experimental only. Keep disabled for the first orientation-aware
        # test because previous generated references previously produced
        # mixed palm/back surface semantics.
        generated_mask = pose_mask_from_keypoints(
            target_pose
        )

        temporal_cond = build_reference_condition(
            generated,
            target_pose,
            generated_mask,
            autoencoder,
            device,
            latent_seed=(
                REF_LATENT_SEED
                + 1000
                + frame_index
            ),
        )

    return (
        generated,
        temporal_cond,
    )


# ============================================================
# Video writer
# ============================================================

def write_mp4(
    frames_rgb: list[np.ndarray],
    output_path: str,
    fps: float,
) -> None:

    if not frames_rgb:

        raise RuntimeError(
            "No generated frames to encode."
        )

    h, w = (
        frames_rgb[0].shape[
            :2
        ]
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        float(
            fps
        ),
        (
            w,
            h,
        ),
    )

    if not writer.isOpened():

        raise RuntimeError(
            "OpenCV could not open MP4 writer. "
            "Install an FFmpeg-enabled OpenCV build "
            "or encode the saved PNG frames manually."
        )

    try:

        for frame_rgb in (
            frames_rgb
        ):

            writer.write(
                cv2.cvtColor(
                    frame_rgb,
                    cv2.COLOR_RGB2BGR,
                )
            )

    finally:

        writer.release()


# ============================================================
# Main
# ============================================================

def main() -> None:

    reset_seed(
        SEED
    )

    torch.backends.cudnn.deterministic = (
        True
    )

    torch.backends.cudnn.benchmark = (
        False
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = osp.join(
        RUN_ROOT,
        timestamp,
    )

    frames_dir = osp.join(
        run_dir,
        "frames",
    )

    os.makedirs(
        frames_dir,
        exist_ok=True,
    )

    output_video = osp.join(
        run_dir,
        "test_image2video.mp4",
    )

    manifest_path = osp.join(
        run_dir,
        "run_info.json",
    )

    print("=" * 70)
    print("FoundHand FSK test_image2video - fixed IMG_1087 ablation")
    print("=" * 70)

    print(
        "Python:",
        sys.executable,
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        "Run directory:",
        run_dir,
    )

    print(
        "Temporal generated refs:",
        USE_PREVIOUS_GENERATED_REFS,
    )

    print(
        "Interpolation steps:",
        INTERPOLATION_STEPS_BETWEEN,
    )

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    for path in (
        POSE_PATH,
        MODEL_PATH,
        VAE_PATH,
        SAM_PATH,
    ):

        if not osp.exists(
            path
        ):

            raise FileNotFoundError(
                path
            )

    # --------------------------------------------------------
    # Fixed official reference: IMG_1087 frame 6
    # --------------------------------------------------------

    official_ref = load_official_reference_exact(
        OFFICIAL_REF_SEQUENCE,
        OFFICIAL_REF_FRAME,
    )

    ref_img = official_ref["image"]
    ref_keypts = official_ref["keypoints"]

    print()
    print("=" * 70)
    print("Fixed official FoundHand reference")
    print("=" * 70)
    print("Sequence:", OFFICIAL_REF_SEQUENCE)
    print("Frame:", OFFICIAL_REF_FRAME)
    print("Image:", official_ref["image_path"])
    print("Hands:", describe_presence(ref_keypts))
    print(
        "NOTE: recovered PALM/BACK is logged for analysis, "
        "but does NOT switch the appearance reference in this ablation."
    )

    # --------------------------------------------------------
    # Load FSK poses
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Loading FSK pose sequence")
    print("=" * 70)

    pose_data = np.load(
        POSE_PATH
    )

    keypoints = np.asarray(
        pose_data[
            "keypoints"
        ],
        dtype=np.float32,
    )

    frame_ids = np.asarray(
        pose_data[
            "frame_ids"
        ]
    )

    required_pose_fields = {
        "hand_present",
        "orientation",
        "orientation_valid",
    }
    missing_pose_fields = required_pose_fields.difference(
        pose_data.files
    )

    if missing_pose_fields:
        raise RuntimeError(
            "POSE_PATH is not a protocol-v3 recovered NPZ. "
            f"Missing: {sorted(missing_pose_fields)}"
        )

    hand_present = np.asarray(
        pose_data["hand_present"],
        dtype=np.uint8,
    )

    orientations = np.asarray(
        pose_data["orientation"],
        dtype=np.uint8,
    )

    orientation_valid = np.asarray(
        pose_data["orientation_valid"],
        dtype=np.uint8,
    )

    keypoints_pixel = convert_fsk_keypoints_to_pixels(
        keypoints
    )

    sequence = prepare_target_sequence(
        keypoints_pixel,
        frame_ids,
        hand_present,
        orientations,
        orientation_valid,
        ref_keypts,
    )

    print(
        "Target sequence length:",
        len(
            sequence
        ),
    )

    print(
        "Target labels:",
        [
            x["label"]
            for x in sequence
        ],
    )

    # --------------------------------------------------------
    # SAM mask once for the fixed official reference
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAM fixed official reference mask")
    print("=" * 70)

    ref_mask_256 = make_sam_mask(
        ref_img,
        ref_keypts,
    )

    print(
        "Mask pixels:",
        int(ref_mask_256.sum()),
    )

    # --------------------------------------------------------
    # Models once
    # --------------------------------------------------------

    (
        diffusion,
        model,
        autoencoder,
    ) = load_models(
        device
    )

    # --------------------------------------------------------
    # Fixed official reference condition once
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Building fixed official reference condition")
    print("=" * 70)

    fixed_ref_cond = build_reference_condition(
        ref_img,
        ref_keypts,
        ref_mask_256,
        autoencoder,
        device,
        latent_seed=REF_LATENT_SEED,
    )

    print(
        "fixed_ref_cond:",
        fixed_ref_cond.shape,
    )

    # --------------------------------------------------------
    # One fixed initial z
    # --------------------------------------------------------

    reset_seed(
        INITIAL_Z_SEED
    )

    base_z = torch.randn(
        (
            1,
            LATENT_DIM,
            LATENT_SIZE[0],
            LATENT_SIZE[1],
        ),
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------
    # Generate loop
    # --------------------------------------------------------

    generated_frames = []
    temporal_ref_conds = []
    frame_records = []

    print()
    print("=" * 70)
    print("Generating sequence")
    print("=" * 70)

    for index, item in enumerate(
        sequence
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"Output frame "
            f"{index + 1}/"
            f"{len(sequence)}"
        )

        print(
            "Label:",
            item["label"],
        )

        print(
            "Pose hands:",
            describe_presence(
                item["pose"]
            ),
        )

        print(
            "Recovered orientation:",
            item["orientation_name"],
        )

        active_ref_name = (
            f"{OFFICIAL_REF_SEQUENCE}_{OFFICIAL_REF_FRAME:04d}_FIXED"
        )

        active_ref_cond = fixed_ref_cond

        print(
            "Appearance reference:",
            active_ref_name,
        )

        generated, temporal_cond = (
            generate_one_frame(
                target_pose=item[
                    "pose"
                ],
                frame_index=index,
                fixed_ref_cond=(
                    active_ref_cond
                ),
                temporal_ref_conds=(
                    temporal_ref_conds
                ),
                base_z=base_z,
                diffusion=diffusion,
                model=model,
                autoencoder=autoencoder,
                device=device,
            )
        )

        frame_path = osp.join(
            frames_dir,
            f"frame_{index:04d}.png",
        )

        io.imsave(
            frame_path,
            generated,
        )

        generated_frames.append(
            generated
        )

        frame_records.append(
            {
                "output_index": index,
                "label": item[
                    "label"
                ],
                "source_frame_id": item[
                    "source_frame_id"
                ],
                "orientation": item[
                    "orientation_name"
                ],
                "appearance_reference": active_ref_name,
                "path": frame_path,
            }
        )

        if (
            USE_PREVIOUS_GENERATED_REFS
            and temporal_cond is not None
        ):

            temporal_ref_conds.append(
                temporal_cond
            )

            temporal_ref_conds = (
                temporal_ref_conds[
                    -MAX_PREVIOUS_GENERATED_REFS:
                ]
            )

        print(
            "Saved:",
            frame_path,
        )

    # --------------------------------------------------------
    # Encode MP4
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Encoding MP4")
    print("=" * 70)

    write_mp4(
        generated_frames,
        output_video,
        VIDEO_FPS,
    )

    print(
        "Video:",
        output_video,
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    run_info = {
        "pose_path": POSE_PATH,
        "reference_mode": "OFFICIAL_FIXED_ABLATION",
        "official_reference_sequence": OFFICIAL_REF_SEQUENCE,
        "official_reference_frame": int(
            OFFICIAL_REF_FRAME
        ),
        "official_reference_image": official_ref["image_path"],
        "official_reference_pose": official_ref["pose_path"],
        "target_hand": TARGET_HAND,
        "sampling_steps": (
            SAMPLING_STEPS
        ),
        "cfg_scale": CFG_SCALE,
        "baseline_mirror_x": (
            BASELINE_MIRROR_X
        ),
        "requested_fsk_frame_ids": (
            FSK_FRAME_IDS
        ),
        "use_all_usable_frames": (
            USE_ALL_USABLE_FRAMES
        ),
        "interpolation_steps_between": (
            INTERPOLATION_STEPS_BETWEEN
        ),
        "video_fps": VIDEO_FPS,
        "reuse_initial_z": (
            REUSE_INITIAL_Z
        ),
        "same_sampling_noise_each_frame": (
            SAME_SAMPLING_NOISE_EACH_FRAME
        ),
        "use_previous_generated_refs": (
            USE_PREVIOUS_GENERATED_REFS
        ),
        "max_previous_generated_refs": (
            MAX_PREVIOUS_GENERATED_REFS
        ),
        "frames": frame_records,
        "video_path": output_video,
    }

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            run_info,
            f,
            indent=2,
        )

    print(
        "Manifest:",
        manifest_path,
    )

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        "Generated frames:",
        len(
            generated_frames
        ),
    )

    print(
        "Video:",
        output_video,
    )

    print(
        "Run directory:",
        run_dir,
    )


if __name__ == "__main__":
    main()
