"""
extract.py contains functions for extracting timestamps from a videofile, as well as some
functions for troubleshooting. with the exception of export_first_last_frames, all the
rest of the functions should be platform independent since they rely on pathlib.
"""

from decord import VideoReader
from pathlib import Path
import numpy as np
import json
from datetime import timedelta
import matplotlib.pyplot as plt
from PIL import Image
from numba import njit

import logging

logging.basicConfig(
    filename="runtime.log",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


def find_offsets(
    vp: str,
    abs_intro_length: int,
    abs_outro_length: int,
    cutoff_boundary: int = 30,
    cache_dir: Path | None = None,
) -> dict[str, int]:
    """given a video file and 'absolute intro/outro length' (in frames) finds the start
    of the actual video as the length of the fade in/out can differ"""
    logging.info(f"finding offsets for {vp}")
    vr: VideoReader = VideoReader(vp, width=160, height=90)
    logging.info(f"finding start offset for {vp}")
    start_offset: int = abs_intro_length
    i: int
    for i in range(len(vr)):
        if i <= abs_intro_length:
            continue
        frame: np.ndarray = np.mean(vr[i].asnumpy(), axis=-1, dtype=int)
        sum_of_color_values: int = np.mean(frame).astype(int)
        if sum_of_color_values > cutoff_boundary:
            start_offset = i
            if cache_dir is not None:
                im = Image.fromarray(vr[i].asnumpy())
                fn: str = f"{Path(vp).name}_first_frame.jpg"
                im.save(f"{cache_dir.joinpath(fn)}")
            break

    logging.info(f"finding end offset for {vp}")
    end_offset: int = -1
    for i in range((len(vr) - abs_outro_length), len(vr)):
        frame: np.ndarray = np.mean(vr[i].asnumpy(), axis=-1, dtype=int)
        sum_of_color_values: int = np.mean(frame).astype(int)
        if sum_of_color_values <= cutoff_boundary:
            end_offset = i
            if cache_dir is not None:
                im = Image.fromarray(vr[i].asnumpy())
                fn: str = f"{Path(vp).name}_last_frame.jpg"
                im.save(f"{cache_dir.joinpath(fn)}")
            break

    return {"start_offset": start_offset, "end_offset": end_offset}


def readVideoFile(
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


def get_pred_suffix(ovp: Path) -> str:
    # since there can be conflicts then use longer folder name and then taper
    # down to only raw file stem if it's too long. This should be enough for
    # most, if not all, conflicts that will realistically show up.
    parentparent: str = ovp.parent.parent.name
    parent: str = ovp.parent.name
    if len(parentparent + parent + ovp.stem) < 255:
        raw_file_pred_suffix: str = f"{parentparent}_{parent}_{ovp.stem}"
    elif len(parent + ovp.stem) < 255:
        raw_file_pred_suffix: str = f"{parent}_{ovp.stem}"
    else:
        raw_file_pred_suffix: str = f"{ovp.stem}"
    return raw_file_pred_suffix


@njit
def distance(original_frame: np.ndarray, subset_frame: np.ndarray) -> np.floating:
    """calculates distance from one [array] image representation to another"""
    return np.mean(
        np.abs(original_frame.astype(np.int16) - subset_frame.astype(np.int16))
    )


def extract_timestamps(
    ovp: Path,
    original_matrix: np.ndarray,
    svp: Path,
    cache_dir: Path = Path("./cache"),
    decision_boundary: float = 0.93,
    pred_dir: Path = Path("./pred"),
    abs_intro_length: int = 106,
    abs_outro_length: int = 500,
    early_stopping_threshold=35,
    step: int = 10,
    skip_factor: int = 2,
    n_comparison: int = 30,
) -> dict | None:
    """extract_timestamps exports json/plot of distance function as well as returns
    timestamps of a subset of a larger video"""

    cached_subset_file: Path = cache_dir.joinpath(Path(svp.name).with_suffix(".npy"))
    logging.info(f"checking if {cached_subset_file} exists...")
    if not cached_subset_file.exists():
        logging.info(f"{cached_subset_file} does not exist. creating...")

        logging.info(f"finding offsets for {svp.name}")
        sv_offsets: dict[str, int] = find_offsets(
            vp=str(svp),
            abs_intro_length=abs_intro_length,
            abs_outro_length=abs_outro_length,
            cache_dir=cache_dir,
        )
        cached_subset_file_offsets: Path = cached_subset_file.with_suffix(".json")
        with open(cached_subset_file_offsets, "w") as f:
            json.dump(sv_offsets, f)

        logging.info(f"creating matrix for {svp.name}")
        subset_matrix = readVideoFile(
            str(svp),
            start_offset=sv_offsets["start_offset"],
            end_offset=sv_offsets["end_offset"],
        )
        np.save(str(cached_subset_file), subset_matrix)
    else:
        logging.info(f"found existing matrix for {svp.name}")
        subset_matrix = np.load(str(cached_subset_file))

    logging.info(f"performing distance calculations for {ovp.name}/{svp.name}")
    # we need fps to be able to skip frames consistently to create denser array
    ofps = int(VideoReader(str(ovp)).get_avg_fps())
    # there are some raw files that are very short but since a clip cannot be
    # part of a raw file if the clip is longer then the raw file, return if
    # that is the case
    if original_matrix.shape[0] - subset_matrix.shape[0] < 0:
        logging.info(f"{svp.name} longer than raw file. skipping...")
        return

    @njit
    def sliding_window_calc(window_ranges: np.ndarray)->dict[int, np.floating]|None:
        """numba wrapper that loops over all sliding windows and returns a
        dictionary with k/v pairs: [frame nr => mean of the difference between
        b/w pixel values]"""
        distances: dict[int, np.floating] = {}
        i: int
        for i in window_ranges:
            distances[i] = distance(
                original_matrix[i : i + subset_matrix.shape[0] : ofps * skip_factor],
                subset_matrix[:: ofps * skip_factor],
            )
            if distances[i] > early_stopping_threshold:
                return None
        return distances

    window_ranges: np.ndarray = np.array(
        list(range(0, (original_matrix.shape[0] - subset_matrix.shape[0]), step))
    )
    d: dict[int, np.floating] | None = sliding_window_calc(window_ranges)
    if d is None:
        return

    f_arr: list = sorted(list(d.values()))
    # in very short raw files an exception can be raised otherwise
    if len(f_arr) - 1 < n_comparison:
        logging.warning(f"""
        WARN: length of f_arr ({len(f_arr)}) < n_comparison ({n_comparison})
        for {svp.name}
        """)
        return

    raw_pred_dir: Path = pred_dir.joinpath(get_pred_suffix(ovp))
    raw_pred_dir.mkdir(exist_ok=True)

    # determine membership on the basis of decision boundary
    in_original: int = (
        1 if (f_arr[n_comparison] - f_arr[0]) > decision_boundary else 0
    )

    # always save plot to pred_dir for easier troubleshooting (at the cost of diskspace)
    plt.plot(list(d.values()))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"distance {svp.name} > {ovp.name}")
    plt.savefig(raw_pred_dir.joinpath(Path(svp.name).with_suffix(".png")))
    plt.close()

    sorted_idx = sorted(d, key=lambda x: d[x])
    start = sorted_idx[0] / ofps
    timestamps = {
        "start": str(timedelta(seconds=start)),
        "end": str(timedelta(seconds=start + (subset_matrix.shape[0] / ofps))),
    }

    export = {
        "filename": str(svp.absolute()),
        "in_original": in_original,
        "start_frame_n": sorted_idx[0],
        "median": np.median(f_arr),
        "lowest_fx": f_arr[np.argmin(f_arr)],
        "in_original_calc": f_arr[n_comparison] - f_arr[0],
        "timestamps": timestamps,
        "fx": list(d.values()),
    }

    with open(raw_pred_dir.joinpath(svp.with_suffix(".json").name), "w") as f:
        json.dump(export, f)

    return export
