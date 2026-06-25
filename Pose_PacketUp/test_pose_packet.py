from Pose_PacketUp.tests.test_pose_packet import *


if __name__ == "__main__":
    test_layout_constants()
    test_encode_size()
    test_header_fields()
    test_payload_slots()
    test_round_trip()
    test_quantization_boundaries()
    test_crc_corruption()
    test_structural_rejection()
    test_iter_decode_stream()
    print("All 9 test groups passed.")
