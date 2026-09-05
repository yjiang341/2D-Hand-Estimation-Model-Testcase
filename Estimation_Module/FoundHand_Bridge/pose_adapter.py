from __future__ import annotations

import numpy as np

from Estimation_Module.Pose_PacketUp.pose_codec import (
    quantized_hand_to_xy_pairs,
)
from Estimation_Module.Pose_PacketUp.pose_packet import PosePacket


def quantized_hand_to_foundhand_xy(
    hand,
    image_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Convert one QuantizedHand into FoundHand keypoints.

    Input:
        QuantizedHand
        coordinates are decoded to normalized [0, 1]

    Output:
        np.ndarray with shape (21, 2)
        coordinates are in image pixels.
    """

    xy = np.asarray(
        quantized_hand_to_xy_pairs(hand),
        dtype=np.float32,
    )

    width, height = image_size

    xy[:, 0] *= (width - 1)
    xy[:, 1] *= (height - 1)

    return xy


def pose_packet_to_foundhand_keypoints(
    packet: PosePacket,
    image_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Convert one PosePacket into FoundHand's 42x2 keypoint format.

    Output:
        shape = (42, 2)

        [0:21]  -> RIGHT hand
        [21:42] -> LEFT hand

    Missing hands remain zero.
    """

    keypts = np.zeros((42, 2), dtype=np.float32)

    if packet.hands[0] is not None:
        keypts[:21] = quantized_hand_to_foundhand_xy(
            packet.hands[0],
            image_size=image_size,
        )

    if packet.hands[1] is not None:
        keypts[21:42] = quantized_hand_to_foundhand_xy(
            packet.hands[1],
            image_size=image_size,
        )

    return keypts


def packets_to_keypoints(
    packets: list[PosePacket],
    image_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Convert a list of PosePackets into FoundHand keypoint sequence.

    Output:
        shape = (num_frames, 42, 2)
    """

    if not packets:
        return np.empty((0, 42, 2), dtype=np.float32)

    return np.stack(
        [
            pose_packet_to_foundhand_keypoints(
                packet,
                image_size=image_size,
            )
            for packet in packets
        ],
        axis=0,
    )