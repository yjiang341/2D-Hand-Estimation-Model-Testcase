from Pose_PacketUp.tests.test_pose_reconstruct import *


if __name__ == "__main__":
    test_decode_stream()
    test_sanity_max_step()
    test_ema_behavior()
    test_frames_to_numpy()
    print("All pose reconstruction tests passed.")
