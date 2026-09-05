import os
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import (
    scale_keypoint,
    keypoint_heatmap,
    check_keypoints_validity,
)


POSE_PATH = r"D:\Project\2D-Hand\Generation_Module\FoundHand\test_result\result_npz\v3_recovered_pose.npz"

IMAGE_SIZE = (256, 256)
LATENT_SIZE = (32, 32)

print("=" * 60)
print("Visualizing FoundHand FSK Heatmaps")
print("=" * 60)

data = np.load(POSE_PATH)

keypoints = data["keypoints"]

print("Keypoints:", keypoints.shape)

# ---------------------------------------------------------
# Find first valid frame
# ---------------------------------------------------------

valid_frames = []

for i in range(len(keypoints)):
    n_valid = np.count_nonzero(keypoints[i].sum(axis=1))

    if n_valid > 0:
        valid_frames.append(i)

print("Valid frames:", valid_frames)

frame_idx = valid_frames[0]

print("Using frame:", frame_idx)

keypts = keypoints[frame_idx].astype(np.float32)

# ---------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------

print("Coordinate range:")
print("X:", keypts[:, 0].min(), "->", keypts[:, 0].max())
print("Y:", keypts[:, 1].min(), "->", keypts[:, 1].max())

# Your current recovered_pose is already pixel coordinates.
# Do NOT multiply by 256 again.

# ---------------------------------------------------------
# Validity
# ---------------------------------------------------------

valid = check_keypoints_validity(
    keypts,
    IMAGE_SIZE
)

print("Valid keypoints:", valid.sum(), "/", len(valid))

# ---------------------------------------------------------
# Convert to latent coordinates
# ---------------------------------------------------------

latent_keypts = scale_keypoint(
    keypts,
    IMAGE_SIZE,
    LATENT_SIZE
)

print("Latent coordinates:")
print(latent_keypts)

# ---------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------

heatmaps = keypoint_heatmap(
    latent_keypts,
    LATENT_SIZE,
    var=1.0
)

heatmaps = heatmaps * valid[:, None, None]

print("Heatmaps:", heatmaps.shape)

# ---------------------------------------------------------
# Save directory
# ---------------------------------------------------------

out_dir = "heatmap_debug"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# Individual keypoint heatmaps
# ---------------------------------------------------------

names = [
    "wrist",

    "thumb_CMC",
    "thumb_MCP",
    "thumb_IP",
    "thumb_TIP",

    "index_MCP",
    "index_PIP",
    "index_DIP",
    "index_TIP",

    "middle_MCP",
    "middle_PIP",
    "middle_DIP",
    "middle_TIP",

    "ring_MCP",
    "ring_PIP",
    "ring_DIP",
    "ring_TIP",

    "pinky_MCP",
    "pinky_PIP",
    "pinky_DIP",
    "pinky_TIP",

    "hand2_wrist",
    "hand2_1",
    "hand2_2",
    "hand2_3",
    "hand2_4",
    "hand2_5",
    "hand2_6",
    "hand2_7",
    "hand2_8",
    "hand2_9",
    "hand2_10",
    "hand2_11",
    "hand2_12",
    "hand2_13",
    "hand2_14",
    "hand2_15",
    "hand2_16",
    "hand2_17",
    "hand2_18",
    "hand2_19",
    "hand2_20",
]

for i in range(42):

    plt.figure(figsize=(5, 5))

    plt.imshow(heatmaps[i], cmap="hot")

    plt.title(
        f"Channel {i}: {names[i]}\n"
        f"valid={bool(valid[i])}"
    )

    plt.colorbar()

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        f"{i:02d}_{names[i]}.png"
    )

    plt.savefig(path, dpi=150)

    plt.close()

# ---------------------------------------------------------
# Finger groups
# ---------------------------------------------------------

groups = {
    "thumb": range(1, 5),
    "index": range(5, 9),
    "middle": range(9, 13),
    "ring": range(13, 17),
    "pinky": range(17, 21),
}

for group_name, indices in groups.items():

    plt.figure(figsize=(6, 6))

    for i in indices:

        plt.imshow(
            heatmaps[i],
            cmap="hot",
            alpha=0.8
        )

    plt.title(group_name)

    plt.axis("off")

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        f"group_{group_name}.png"
    )

    plt.savefig(path, dpi=150)

    plt.close()

# ---------------------------------------------------------
# Skeleton on 32x32 latent space
# ---------------------------------------------------------

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (0, 9), (9, 10), (10, 11), (11, 12),

    (0, 13), (13, 14), (14, 15), (15, 16),

    (0, 17), (17, 18), (18, 19), (19, 20),
]

plt.figure(figsize=(8, 8))

for a, b in connections:

    if valid[a] and valid[b]:

        plt.plot(
            [
                latent_keypts[a, 0],
                latent_keypts[b, 0]
            ],
            [
                latent_keypts[a, 1],
                latent_keypts[b, 1]
            ],
            "b-"
        )

for i in range(21):

    if valid[i]:

        plt.scatter(
            latent_keypts[i, 0],
            latent_keypts[i, 1]
        )

        plt.text(
            latent_keypts[i, 0] + 0.3,
            latent_keypts[i, 1] + 0.3,
            str(i),
            fontsize=8
        )

plt.xlim(0, 32)
plt.ylim(32, 0)

plt.grid()

plt.title("FSK skeleton in FoundHand latent coordinates")

plt.tight_layout()

plt.savefig(
    os.path.join(
        out_dir,
        "latent_skeleton.png"
    ),
    dpi=200
)

plt.close()

print()
print("=" * 60)
print("DONE")
print("=" * 60)
print("Output:", os.path.abspath(out_dir))