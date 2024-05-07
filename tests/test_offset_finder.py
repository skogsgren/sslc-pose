from src.A_extract_timestamps.extract import find_offsets
from pathlib import Path


def test_find_offsets():
    files = {
        "tests/signing_low.mp4": {"start_offset": 5, "end_offset": 96},
        "tests/highway_low.mp4": {"start_offset": 4, "end_offset": 117},
    }
    for file, offsets in files.items():
        calc_offsets = find_offsets(file, 0, 50, cache_dir=Path("tests/cache"))
        assert {file: calc_offsets} == {file: offsets}
