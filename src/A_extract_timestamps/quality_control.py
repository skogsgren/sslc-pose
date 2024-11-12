__doc__ = """
This file contains functions which make it easier to quality control the
results of timestamp extraction.

Specifically, it does the following:

1) concatenates the best result (given the entries.json file) extracted from
the raw file with the clip, in the following two ways:
    a) side-by-side; e.g. proposed timestamp + clip
    b) diff; e.g. the difference between the two in (a).
2) the difference between the proposed timestamp and the clip for each frame
3) a plot of (2)

NOTE: most of these function have some degree of hardcoded SSLC element to them
(especially references to the speaker_id field in SSLC files), though the
primary functionality of the functions are task-agnostic.
"""

from pathlib import Path
import json
from datetime import datetime, timedelta
import numpy as np
import random
from numba import njit
import matplotlib.pyplot as plt
import cv2
import argparse

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import naming_function
from double_pass_extract import read_video_file


def get_ts(seconds):
    time_delta = timedelta(seconds=seconds)
    time = datetime(1, 1, 1) + time_delta
    timestamp = time.strftime("%H:%M:%S.%f")
    return timestamp


def get_timestamps(clip: Path, raw_file: Path, pred_dir: Path) -> tuple[str, str]:
    with open(
        pred_dir / naming_function(raw_file) / clip.with_suffix(".json").name, "r"
    ) as f:
        timestamps = json.load(f)["timestamps"]
    return (timestamps["start"], timestamps["end"])


def get_start_frame(clip: Path, raw_file: Path, pred_dir: Path) -> int:
    with open(
        pred_dir / naming_function(raw_file) / clip.with_suffix(".json").name, "r"
    ) as f:
        return json.load(f)["start_frame_n"]


def export_frames_as_video(out: str, fps: int, frames: np.ndarray):
    height: int = frames.shape[1]
    width: int = frames.shape[2]
    vw = cv2.VideoWriter(
        out, cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height), isColor=False
    )
    for frame in frames:
        vw.write(frame)
    vw.release()


def quality_control(
    cfg_json: str,
    QCRUN: Path,
    k: int,
    speaker_id: int | None = None,
    debug_clip: str | None = None,
    no_diff: bool = False,
):
    with open(cfg_json, "r") as f:
        cfg = json.load(f)
    with open(Path(cfg["pred_dir"]) / "entries.json", "r") as f:
        entries = json.load(f)

    entry_data: list[dict] = list(entries.values())
    if speaker_id:
        sample = [entries[speaker_id]]
    elif len(entry_data) < k:
        sample = entry_data
    else:
        sample = random.choices(entry_data, k=k)

    sample_data: list = []
    for i in sample:
        sample_data += i["matched_clips"]
        sample_data += i["unmatched_clips"]

    if debug_clip:
        sample_data = [x for x in sample_data if Path(x[1]).stem == debug_clip]

    @njit
    def distance_calc(orig_range: np.ndarray, clip_range: np.ndarray) -> np.ndarray:
        calc = np.abs(orig_range.astype(np.int16) - clip_range.astype(np.int16))
        return calc

    for score, clip, raw_file in [
        (round(float(clip[0]), 2), Path(clip[1]), Path(clip[2])) for clip in sample_data
    ]:
        print(score, clip, raw_file)
        print("reading clip matrix")
        clip_matrix = read_video_file(str(clip))
        print("reading raw matrix")
        start_frame = get_start_frame(clip, raw_file, Path(cfg["pred_dir"]))
        end_frame = start_frame + clip_matrix.shape[0]
        raw_matrix = read_video_file(
            str(raw_file), start_offset=start_frame, end_offset=end_frame
        )

        print("concatenating raw_matrix and clip_matrix")
        if raw_matrix.shape != clip_matrix.shape:
            print(
                f"raw_matrix ({raw_matrix.shape}) is not the same shape as clip_matrix ({clip_matrix.shape})"
            )
            continue
        concat_frames = np.concatenate((clip_matrix, raw_matrix), axis=-1)
        print("concat_frames_shape: ", concat_frames.shape)

        print("exporting as video")
        export_frames_as_video(
            str(QCRUN / (clip.stem + f"_CV_side_by_side_{score}.avi")),
            25,
            concat_frames,
        )

        if no_diff:
            print("--no_diff; skipping calculating diff...")
            continue

        print("calculating debug distance calculations")
        distances = distance_calc(raw_matrix, clip_matrix)
        d_mean = np.mean(distances)

        initial_scores = np.mean(distances, axis=(1, 2))
        mask = initial_scores > d_mean
        initial_scores[mask] = initial_scores[mask] ** 2
        debug_score = np.mean(initial_scores)

        export_frames_as_video(
            str(QCRUN / (clip.stem + f"_CV_diff_{score}.avi")),
            25,
            (((distances / 255) ** 2 * 255) * 3).astype("uint8"),
        )

        plt.plot([np.mean(x) for x in distances])
        plt.xlabel("frame in range")
        plt.ylabel("f(x)")
        plt.title(f"distances for {clip.name} frame by frame")
        plt.savefig(QCRUN / (clip.stem + f"_D_{score}.png"))
        plt.close()
        with open(QCRUN / (clip.stem + f"_D_{score}.json"), "w") as f:
            json.dump(
                {
                    "clip_file": str(clip),
                    "raw_file": str(raw_file),
                    "debug_score": float(debug_score),
                    "scores": list(initial_scores),
                    "distances": [np.mean(x) for x in distances],
                },
                f,
            )


def create_quality_control_dir(base_dir: Path) -> Path:
    """creates a quality control dir given base directory, returns Path pointer"""
    if not base_dir.exists():
        base_dir.mkdir()
    quality_control_dir = base_dir / datetime.now().strftime("%Y%m%d%H%M")
    quality_control_dir.mkdir(exist_ok=True)
    return quality_control_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", required=True, help="path to config json")
    parser.add_argument(
        "--speaker_id",
        required=False,
        default=None,
        type=str,
        help="only quality check one speaker id",
    )
    parser.add_argument(
        "--debug_clip",
        default=None,
        type=str,
        help="used to debug one clip inside speaker_id (only use with --speaker_id flag)",
    )
    parser.add_argument(
        "-k", required=False, default=5, type=int, help="size of quality check sample"
    )
    parser.add_argument("--no_diff", action="store_true", default=False)
    args = parser.parse_args()

    quality_control(
        cfg_json=args.cfg,
        k=args.k,
        QCRUN=create_quality_control_dir(Path("./quality_check")),
        speaker_id=args.speaker_id,
        debug_clip=args.debug_clip,
        no_diff=args.no_diff,
    )
