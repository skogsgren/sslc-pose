from collections import Counter
import numpy as np
from numba import njit
from pathlib import Path
import json
from tqdm import tqdm
import random
import argparse

from double_pass_extract import distance
from utils import naming_function, timestamp_to_seconds, seconds_to_timestamp
from quality_control import export_frames_as_video, create_quality_control_dir


def timestamp_wrapper(
    pred_dir: Path, raw_video_file: Path, clip_video_file: Path
) -> list[int]:
    """given a prediction directory of inferred timestamps (see extract.py)
    returns a list of frames, where index 0 is the start point and index 1 is
    the end point for that timestamp."""
    with open(
        (
            pred_dir
            / naming_function(raw_video_file)
            / clip_video_file.with_suffix(".json").name
        ),
        "r",
    ) as f:
        return [
            int(timestamp_to_seconds(v) * 25)
            for _, v in json.load(f)["timestamps"].items()
        ]


@njit
def beam_search(
    original_matrix: np.ndarray, clip_matrix: np.ndarray, k: int = 2, b: float = 1.0
):
    """perform beam_search for best path fit for the minimum distance between
    original_matrix and clip_matrix with the possibility of duplicate frames
    given k beam width. b is a bias that is multiplied to the duplicated
    distance calculation since they're less likely to occur, but can still have
    similar distance calculations (for example when the subject remains still
    in frame).  note that original_matrix is assumed to be in sync in regards
    to start/end point"""
    sequences: list[tuple[np.floating, list[int], np.int32]] = [
        (np.float64(0.0), [0], np.int32(0))
    ]
    for i in range(1, clip_matrix.shape[0]):
        candidates: list[tuple[np.floating, list[int], np.int32]] = []
        for sequence in sequences:
            prev_idx = sequence[1][-1]

            norm_dist = distance(clip_matrix[i], original_matrix[prev_idx + 1])
            candidates.append(
                (sequence[0] + norm_dist, sequence[1] + [prev_idx + 1], sequence[2] + 1)
            )

            # don't allow triple duplicate frames in a row
            if len(sequence[1]) >= 2 and prev_idx == sequence[1][-2]:
                continue
            dupl_dist = distance(clip_matrix[i], original_matrix[prev_idx]) * b
            candidates.append(
                (sequence[0] + dupl_dist, sequence[1] + [prev_idx], sequence[2])
            )
        sequences = sorted(candidates)[:k]
    return sorted(sequences)


def beam_search_wrapper(
    raw_video_file: Path,
    clip_video_file: Path,
    cfg_json: Path,
    n: int = 2000,
    k: int = 2,
    b: float = 1.0,
):
    """wrapper function to perform 'two-tailed' beam_search given an index,
    meaning that it performs beam search for the lowest distance for a raw
    video file and a video clip +- n, where n is a number of frames. For
    example, if n = 10 and the lowest start frame for a raw video file is i=50,
    then the function performs beam search for the range {i+-n} and returns a
    dict with the total sum of the beam_search for that beginning frame. the
    function returns a dict {frame_offset: distance_sum}"""
    with open(cfg_json, "r") as f:
        cfg = json.load(f)
        PRED_DIR = Path(cfg["pred_dir"])
        CACHE_DIR = Path(cfg["cache_dir"])
    timestamps = timestamp_wrapper(PRED_DIR, raw_video_file, clip_video_file)

    # slice only the needed frames in raw video file to save memory
    raw_matrix = np.load(CACHE_DIR / (naming_function(raw_video_file) + ".npy"))[
        timestamps[0] - n : timestamps[1] + n :
    ]
    clip_matrix = np.load(CACHE_DIR / clip_video_file.with_suffix(".npy").name)

    beam_search_dict: dict[int, list] = {}
    for i in tqdm(list(range(n * 2))):  # multiply by two since it's two-tailed
        # same goes for why n is subtracted, I want it to begin to the left
        beam_search_dict[i - n] = beam_search(
            # this logic is confusing since i begins from 0, but index 0 for
            # the raw video file matrix in this case is timestamp_start - n,
            # i.e. n frames before the inferred best match
            raw_matrix[i : clip_matrix.shape[0] - i],
            clip_matrix,
            k=k,
            b=b,
        )

    for i, sequences in beam_search_dict.items():
        best = sequences[0]
        frame_counter = Counter(best[1])
        print(
            i,
            " duplicate frames: ",
            len([x for x, y in frame_counter.items() if y == 2]),
            "; distance: ",
            best[0],
        )

    min_distance = sorted([(v[0][0], i, v[0][2]) for i, v in beam_search_dict.items()])[
        0
    ]
    print("min distance: ", min_distance[0], min_distance[1])

    return (beam_search_dict, min_distance[1])


def get_new_timestamps(
    frame_timestamps: tuple[int, int], frame_offset: int, n_dupl_frames: int
) -> tuple[str, str]:
    """given a tuple of timestamps (in absolute frames) and an offset, apply
    that offset and convert the frames to an ffmpeg compatible timestamp"""
    FPS = 25
    start = frame_timestamps[0]
    end = frame_timestamps[1]
    return (
        seconds_to_timestamp((start + frame_offset) / FPS),
        seconds_to_timestamp((end + frame_offset - n_dupl_frames) / FPS),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="using mapping json perform beam search for duplicate frames"
    )
    parser.add_argument(
        "-c",
        "--cfg",
        type=str,
        help="config json file used during initial search",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--samples",
        type=int,
        help="instead of the whole file, take SAMPLES samples from the entries.json file in pred_dir",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--mappings",
        type=str,
        help="path to mappings json with same structure as below (for debug purposes)",
    )
    args = parser.parse_args()

    QCRUN = create_quality_control_dir(Path("quality_control"))
    CFGP = Path(args.cfg)
    with open(CFGP, "r") as f:
        CFG = json.load(f)
        PRED_DIR = Path(CFG["pred_dir"])
        CACHE_DIR = Path(CFG["cache_dir"])

    with open(PRED_DIR / "entries.json", "r") as f:
        entries = json.load(f)
    matched_clips = {}
    unmatched_clips = {}
    for _, clips in entries.items():
        matched_clips.update({x[1]: (x[2], x[0]) for x in clips["matched_clips"]})
        unmatched_clips.update({x[1]: (x[2], x[0]) for x in clips["unmatched_clips"]})
    mappings = {}
    if args.samples:
        for k in random.sample(list(matched_clips.keys()), args.samples):
            mappings[k] = matched_clips[k]
        for k in random.sample(list(unmatched_clips.keys()), args.samples):
            mappings[k] = unmatched_clips[k]
    elif args.mappings:
        with open(args.mappings, "r") as f:
            mappings = json.load(f)
    else:
        for k, v in matched_clips.items():
            mappings[k] = v
        for k, v in unmatched_clips.items():
            mappings[k] = v

    for clip, raw_file, score in [
        (Path(x), Path(y[0]), y[1]) for x, y in mappings.items()
    ]:
        res, min_off = beam_search_wrapper(raw_file, clip, CFGP, n=16, b=1.25, k=15)
        with open(QCRUN / f"{clip.name}_{round(score, 2)}.json", "w") as f:
            json.dump(res, f)

        timestamps = timestamp_wrapper(PRED_DIR, raw_file, clip)
        start = timestamps[0]
        end = timestamps[1]

        raw_matrix = np.load(CACHE_DIR / (naming_function(raw_file) + ".npy"))
        clip_matrix = np.load(CACHE_DIR / clip.with_suffix(".npy").name)

        dupl_fr = [x for x, y in Counter(res[min_off][0][1]).items() if y == 2]

        cutout_raw_matrix = raw_matrix[start + min_off : end + min_off - len(dupl_fr)]
        cutout_clip_matrix = np.delete(clip_matrix, dupl_fr, axis=0)

        try:
            concat_fr = np.concatenate((cutout_clip_matrix, cutout_raw_matrix), axis=-1)
            export_frames_as_video(
                out=str(QCRUN / clip.with_suffix(".mp4").name), fps=25, frames=concat_fr
            )
            print(f"finished processing {clip.name} ({round(score, 2)})")
        except ValueError:
            print(
                "input/output arr dimension mismatch for ",
                clip.name,
                " with shape ",
                cutout_clip_matrix.shape,
                " and ",
                raw_file.name,
                " with shape ",
                cutout_raw_matrix.shape,
            )
            ap = np.concatenate(
                (cutout_raw_matrix, np.zeros((1, 90, 160), dtype="uint8")), axis=0
            )
            try:
                concat_fr = np.concatenate(
                    (
                        cutout_clip_matrix,
                        ap,
                    ),
                    axis=-1,
                )
            except ValueError:
                print("error again, skipping...")
                continue
            export_frames_as_video(
                out=str(QCRUN / clip.with_suffix(".mp4").name), fps=25, frames=concat_fr
            )
            print("padding...")
