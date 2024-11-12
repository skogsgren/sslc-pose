__doc__ = """
double_pass_extract.py contains functions for extracting timestamps from a
videofile, as well as some functions for troubleshooting. with the exception of
export_first_last_frames, all the rest of the functions should be platform
independent since they rely on pathlib.

It does this by performing a double pass sliding window search, the initial
pass being very coarse-grained, and then using that minima as the center of the
next pass, where a fine-grained sliding window calculation is performed.

These functions are task-agnostic, and should be easy to change to other
datasets.
"""

from decord import VideoReader
from pathlib import Path
import numpy as np
import json
from datetime import timedelta
import matplotlib.pyplot as plt
from numba import njit

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import naming_function
import logging

logging.basicConfig(
    filename="runtime.log",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)

FIRST_PASS_ARRAY_SIZE = 50
FIRST_PASS_SKIP = 12

SECOND_PASS_ARRAY_SIZE = 250
SECOND_PASS_RANGE = 125


def read_video_file(
    vp: str, start_offset: int = 0, end_offset: int | None = None
) -> np.ndarray:
    """given path to video file returns arr of reduced res b/w numpy representations"""
    logging.info(f"readVideoFile: creating VideoReader with {vp}")
    vr: VideoReader = VideoReader(vp, width=160, height=90)  # assuming 16:9
    if end_offset is not None:
        desired_frames: list[int] = list(range(start_offset, end_offset))
    else:
        desired_frames: list[int] = list(range(start_offset, len(vr)))
    logging.info(f"readVideoFile: extracting desired_frames for {vp} using get_batch")
    extracted_frames: np.ndarray = vr.get_batch(desired_frames).asnumpy()
    logging.info(f"readVideoFile: turning all frames into monochrome for {vp}")
    res: np.ndarray = np.mean(extracted_frames, axis=-1).astype(
        np.uint8
    )  # mean turns image b/w
    return res


@njit
def distance(original_frame: np.ndarray, subset_frame: np.ndarray) -> np.floating:
    """calculates distance from one [array] image representation to another"""
    return np.mean(
        np.abs(original_frame.astype(np.int16) - subset_frame.astype(np.int16))
    )


def double_pass_timestamp_extraction(
    raw_file_path: Path,
    original_matrix: np.ndarray,
    clip_video_path: Path,
    cache_dir: Path = Path("./cache"),
    decision_boundary: float = 0.93,
    pred_dir: Path = Path("./pred"),
    early_stopping_threshold=35,
    n_comparison: int = 30,
) -> dict | None:
    """extract_timestamps exports json/plot of distance function as well as returns
    timestamps of a subset of a larger video"""

    cached_clip_file: Path = cache_dir.joinpath(
        Path(clip_video_path.name).with_suffix(".npy")
    )
    logging.info(f"checking if {cached_clip_file} exists...")
    if not cached_clip_file.exists():
        subset_matrix = read_video_file(str(clip_video_path))
        np.save(str(cached_clip_file), subset_matrix)
    else:
        logging.info(f"found existing matrix for {clip_video_path.name}")
        subset_matrix = np.load(str(cached_clip_file))

    logging.info(
        f"performing distance calculations for {raw_file_path.name}/{clip_video_path.name}"
    )
    # we need fps to be able to skip frames consistently to create denser array
    ofps = VideoReader(str(raw_file_path)).get_avg_fps()
    if ofps != 25.0:  # if applying to other datasets, change this accordingly
        logging.info(f"WARN: FPS is {ofps}")

    # there are some raw files that are very short but since a clip cannot be
    # part of a raw file if the clip is longer then the raw file, return if
    # that is the case
    if original_matrix.shape[0] - subset_matrix.shape[0] < 0:
        logging.info(f"{clip_video_path.name} longer than raw file. skipping...")
        return

    @njit
    def sliding_window_calc(
        window_ranges: np.ndarray, n_frames: int
    ) -> dict[int, np.floating] | None:
        """numba wrapper that loops over all sliding windows and returns a
        dictionary with k/v pairs: [frame nr => mean of the difference between
        b/w pixel values]. n_frames controls how large the comparison array
        ought to be"""
        distances: dict[int, np.floating] = {}
        i: int
        for i in window_ranges:
            sms = subset_matrix.shape[0]
            fpor = original_matrix[i : i + sms : sms / n_frames]
            fpsb = subset_matrix[:: sms / n_frames]
            distances[i] = distance(fpor, fpsb)
            if distances[i] > early_stopping_threshold:
                return None
        return distances

    sms = subset_matrix.shape[0]
    oms = original_matrix.shape[0]

    # first pass
    logging.info(f"first pass for {clip_video_path.name}")
    first_pass_window_ranges = np.arange(
        0, (original_matrix.shape[0] - subset_matrix.shape[0]), FIRST_PASS_SKIP
    )
    first_pass_distances = sliding_window_calc(
        first_pass_window_ranges,
        FIRST_PASS_ARRAY_SIZE if sms > FIRST_PASS_ARRAY_SIZE else sms,
    )
    first_pass_min = min(first_pass_distances, key=first_pass_distances.get)

    # second pass
    logging.info(f"second pass for {clip_video_path.name}")
    if (first_pass_min - SECOND_PASS_RANGE) < 0:
        second_pass_min = 0
    else:
        second_pass_min = first_pass_min - SECOND_PASS_RANGE
    if (first_pass_min + sms + SECOND_PASS_RANGE) > oms:
        second_pass_max = oms - sms
    else:
        second_pass_max = first_pass_min + SECOND_PASS_RANGE
    second_pass_window_ranges = np.arange(second_pass_min, second_pass_max)

    second_pass_distances: dict[int, np.floating] | None = sliding_window_calc(
        second_pass_window_ranges,
        SECOND_PASS_ARRAY_SIZE if sms > SECOND_PASS_ARRAY_SIZE else sms,
    )
    if second_pass_distances is None:
        logging.info(
            f"WARN:found no distances for {clip_video_path.name}"
            f"\t{second_pass_distances}"
        )
        return

    f_arr: list = sorted(list(second_pass_distances.values()))
    # in very short raw files an exception can be raised otherwise
    if len(f_arr) - 1 < n_comparison:
        logging.info(
            f"""
        WARN: length of f_arr ({len(f_arr)}) < n_comparison ({n_comparison})
        for {clip_video_path.name}
        """
        )
        return

    raw_pred_dir: Path = pred_dir / naming_function(raw_file_path)
    raw_pred_dir.mkdir(exist_ok=True)

    logging.info(f"exporting results for {clip_video_path.name} to {raw_pred_dir.name}")
    # determine membership on the basis of decision boundary
    in_original: int = 1 if (f_arr[n_comparison] - f_arr[0]) > decision_boundary else 0

    # always save plot to pred_dir for easier troubleshooting (at the cost of diskspace)
    plt.plot(list(second_pass_distances.values()))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"distance {clip_video_path.name} > {raw_file_path.name}")
    plt.savefig(raw_pred_dir / Path(clip_video_path.name).with_suffix(".png"))
    plt.close()

    sorted_idx = sorted(second_pass_distances, key=lambda x: second_pass_distances[x])
    start = sorted_idx[0] / ofps
    timestamps = {
        "start": str(timedelta(seconds=start)),
        "end": str(timedelta(seconds=start + (subset_matrix.shape[0] / ofps))),
    }

    export = {
        "filename": str(clip_video_path.absolute()),
        "raw_file": str(raw_file_path.absolute()),
        "in_original": in_original,
        "start_frame_n": sorted_idx[0],
        "median": np.median(f_arr),
        "lowest_fx": f_arr[np.argmin(f_arr)],
        "in_original_calc": f_arr[n_comparison] - f_arr[0],
        "timestamps": timestamps,
        "fx": list(second_pass_distances.values()),
    }

    with open(raw_pred_dir / clip_video_path.with_suffix(".json").name, "w") as f:
        json.dump(export, f)
    logging.info(f"finished exporting results for {clip_video_path.name}")

    return export
