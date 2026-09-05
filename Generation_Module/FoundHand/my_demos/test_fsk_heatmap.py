import numpy as np
import matplotlib.pyplot as plt

from utils.utils import (
    scale_keypoint,
    keypoint_heatmap,
    check_keypoints_validity,
)


# ============================================================
# Configuration
# ============================================================

POSE_FILE = r"D:\Project\2D-Hand\Generation_Module\FoundHand\test_result\result_npz\v3_recovered_pose.npz"

IMAGE_SIZE = (256, 256)
LATENT_SIZE = (32, 32)


# ============================================================
# Load
# ============================================================

print("=" * 60)
print("Loading:", POSE_FILE)
print("=" * 60)

data = np.load(POSE_FILE)

keypoints = data["keypoints"]
frame_ids = data["frame_ids"]
timestamps = data["timestamps"]

print("Keypoints shape:", keypoints.shape)
print("Frame IDs shape:", frame_ids.shape)
print("Timestamps shape:", timestamps.shape)

assert keypoints.shape[1:] == (42, 2)


# ============================================================
# Determine coordinate format
# ============================================================

raw_min = keypoints.min()
raw_max = keypoints.max()

print()
print("Raw coordinate range:")
print("X/Y min:", raw_min)
print("X/Y max:", raw_max)


if raw_max <= 1.0:
    coordinate_format = "normalized"
elif raw_max <= 256.0:
    coordinate_format = "pixel"
else:
    raise RuntimeError(
        f"Unexpected coordinate range: {raw_min} -> {raw_max}"
    )


print("Detected coordinate format:", coordinate_format)


# ============================================================
# Convert to FoundHand pixel coordinates
# ============================================================

keypoints_pixel = keypoints.astype(np.float32).copy()

if coordinate_format == "normalized":
    keypoints_pixel[..., 0] *= IMAGE_SIZE[0]
    keypoints_pixel[..., 1] *= IMAGE_SIZE[1]

elif coordinate_format == "pixel":
    # Already in pixel coordinates.
    # Do NOT multiply by 256.
    pass


print()
print("FoundHand pixel coordinate range:")
print(
    "X:",
    keypoints_pixel[..., 0].min(),
    "->",
    keypoints_pixel[..., 0].max(),
)

print(
    "Y:",
    keypoints_pixel[..., 1].min(),
    "->",
    keypoints_pixel[..., 1].max(),
)


# ============================================================
# Find first frame containing keypoints
# ============================================================

print()
print("Searching for first non-empty frame...")

nonempty_frames = []

for i in range(len(keypoints_pixel)):
    count = np.count_nonzero(keypoints_pixel[i])
    if count > 0:
        nonempty_frames.append(i)

print("Non-empty frames:", nonempty_frames)

if len(nonempty_frames) == 0:
    raise RuntimeError("No keypoints found in recovered pose.")


test_frame = nonempty_frames[0]

print()
print("Using test frame index:", test_frame)
print("Using frame ID:", frame_ids[test_frame])


# ============================================================
# Get keypoints
# ============================================================

pts = keypoints_pixel[test_frame]

print()
print("Selected frame keypoints:")
print(pts)


# ============================================================
# Check validity
# ============================================================

valid = check_keypoints_validity(
    pts,
    IMAGE_SIZE,
)

print()
print("Valid keypoints:", int(valid.sum()), "/", len(valid))

print("Validity mask:")
print(valid)


# ============================================================
# Scale to latent space
# ============================================================

latent_pts = scale_keypoint(
    pts,
    IMAGE_SIZE,
    LATENT_SIZE,
)

print()
print("Latent keypoints shape:", latent_pts.shape)

print(
    "Latent X range:",
    latent_pts[:, 0].min(),
    "->",
    latent_pts[:, 0].max(),
)

print(
    "Latent Y range:",
    latent_pts[:, 1].min(),
    "->",
    latent_pts[:, 1].max(),
)


# ============================================================
# Generate heatmaps
# ============================================================

heatmaps = keypoint_heatmap(
    latent_pts,
    LATENT_SIZE,
    var=1.0,
)

print()
print("Heatmap shape:", heatmaps.shape)

assert heatmaps.shape == (42, 32, 32)


# ============================================================
# Apply validity mask
# ============================================================

heatmaps_masked = heatmaps * valid[:, None, None]

print(
    "Masked heatmap min:",
    heatmaps_masked.min()
)

print(
    "Masked heatmap max:",
    heatmaps_masked.max()
)


# ============================================================
# Save heatmap visualization
# ============================================================

# Sum all valid keypoint heatmaps into one image.
combined_heatmap = heatmaps_masked.sum(axis=0)

plt.figure(figsize=(6, 6))
plt.imshow(combined_heatmap)
plt.title(
    f"FSK Frame {frame_ids[test_frame]} "
    f"({int(valid.sum())} valid keypoints)"
)
plt.colorbar()
plt.axis("off")

heatmap_output = "test_fsk_heatmap.png"
plt.savefig(
    heatmap_output,
    bbox_inches="tight",
    dpi=150,
)

plt.close()

print()
print("Saved:", heatmap_output)


# ============================================================
# Save skeleton visualization
# ============================================================

plt.figure(figsize=(6, 6))

# First hand
if valid[:21].sum() > 0:

    hand = pts[:21]

    plt.plot(
        hand[:, 0],
        hand[:, 1],
        "o-",
        linewidth=2,
    )


# Second hand
if valid[21:].sum() > 0:

    hand = pts[21:]

    plt.plot(
        hand[:, 0],
        hand[:, 1],
        "o-",
        linewidth=2,
    )


plt.xlim(0, IMAGE_SIZE[0])
plt.ylim(IMAGE_SIZE[1], 0)

plt.title(
    f"Recovered FSK Skeleton - Frame {frame_ids[test_frame]}"
)

plt.grid(True)

skeleton_output = "test_fsk_skeleton.png"

plt.savefig(
    skeleton_output,
    bbox_inches="tight",
    dpi=150,
)

plt.close()

print("Saved:", skeleton_output)


# ============================================================
# Final
# ============================================================

print()
print("=" * 60)
print("SUCCESS")
print("=" * 60)

print("Coordinate format :", coordinate_format)
print("Test frame index   :", test_frame)
print("Frame ID           :", frame_ids[test_frame])
print("Valid keypoints    :", int(valid.sum()))
print("Heatmap shape      :", heatmaps.shape)
print("Heatmap output     :", heatmap_output)
print("Skeleton output    :", skeleton_output)