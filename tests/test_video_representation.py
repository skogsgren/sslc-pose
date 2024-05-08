from pathlib import Path
import numpy as np
from src.A_extract_timestamps.extract import read_video_file


def test_video_representation():
    files = {
        "tests/signing_low.mp4": {"start_offset": 5, "end_offset": 96},
        "tests/highway_low.mp4": {"start_offset": 4, "end_offset": 117},
    }
    for test_file, offsets in files.items():
        test_arr = np.load(Path(test_file).with_suffix(".npy"))
        read_arr = read_video_file(
            test_file,
            start_offset=offsets["start_offset"],
            end_offset=offsets["end_offset"],
        )
        assert np.array_equal(test_arr, read_arr)
