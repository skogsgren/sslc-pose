from pathlib import Path
from src.A_extract_timestamps.extract import extract_timestamps, read_video_file


def test_extract_timestamps():
    for raw_file, subset_file in [
        ("tests/signing_raw.mp4", "tests/signing_low.mp4"),
        ("tests/highway_raw.mp4", "tests/highway_low.mp4"),
    ]:
        timestamps = extract_timestamps(
            Path(raw_file),
            read_video_file(raw_file),
            Path(subset_file),
            early_stopping_threshold=100,
            step=1,
            skip_factor=1,
            abs_intro_length=0,
            abs_outro_length=50,
        )
        assert timestamps["in_original"] == 1
