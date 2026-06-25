from FSK_Module.tests.test_fsk_receiver import *


if __name__ == "__main__":
    test_receiver_roundtrip_waveform()
    test_receiver_roundtrip_wav_file()
    test_receiver_corruption_drop()
    print("All FSK receiver tests passed.")
