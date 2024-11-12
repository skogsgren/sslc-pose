__doc__ = """
This file contains functions that, given a folder of inputs and a desired
output folder, uses ffmpeg to cut the intro/outro of each respective file. This
leaves only the actual content of the video file.

NOTE: this currently contains hardcoded elements of SSLC in its implementation,
however, not that many. That's because SSLC video clips intro first contains a
preamble (with licensing information and such) and then a varying amount of
fade from black.
"""

from decord import VideoReader
import numpy as np
import argparse
from pathlib import Path
import json
import subprocess
from datetime import timedelta


def find_offsets(
    vp: str,
    abs_intro_length: int,
    abs_outro_length: int,
    cutoff_boundary: float = 0.9,
) -> dict[str, int]:
    """given a video file and 'absolute intro/outro length' (in frames) finds
    the start of the actual video as the length of the fade in/out can
    differ"""
    vr: VideoReader = VideoReader(vp, width=160, height=90)
    assert vr.get_avg_fps() == 25.0
    N: int = 250
    MID: int = len(vr) // 2
    mid_mean: np.floating = (
        vr.get_batch(
            np.arange((abs_intro_length + MID - N), (abs_intro_length + MID + N))
        )
        .asnumpy()
        .mean()
    )

    start_offset: int = abs_intro_length
    i: int
    for i in range(abs_intro_length, len(vr)):
        frame: np.ndarray = np.mean(vr[i].asnumpy(), axis=-1, dtype=int)
        calc: np.floating = frame.mean() / mid_mean
        if calc >= cutoff_boundary:
            start_offset = i
            break

    end_offset: int = -1
    for i in range((len(vr) - abs_outro_length), len(vr)):
        frame: np.ndarray = np.mean(vr[i].asnumpy(), axis=-1, dtype=int)
        calc: np.floating = frame.mean() / mid_mean
        if calc <= cutoff_boundary:
            end_offset = i - 1
            break

    return {"start_offset": start_offset, "end_offset": end_offset}


def frame_to_timestamp(frame: int, fps: float = 25.0) -> str:
    return str(timedelta(seconds=(frame / fps)))


def wrapper(inp: Path, out: Path, c: float, intro_len: int, outro_len: int):
    if not out.exists():
        out.mkdir(parents=True)
    for f in sorted([x for x in inp.iterdir() if x.suffix == ".mp4"]):
        offsets: dict = find_offsets(
            str(f),
            abs_intro_length=intro_len,
            abs_outro_length=outro_len,
            cutoff_boundary=c,
        )
        print("==", f.name, offsets)

        start_timestamp = frame_to_timestamp(offsets["start_offset"])
        end_timestamp = frame_to_timestamp(offsets["end_offset"])
        offsets["start_timestamp"] = start_timestamp
        offsets["end_timestamp"] = end_timestamp

        with open(out / (f.stem + ".json"), "w") as fo:
            json.dump(offsets, fo)

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-ss",
                start_timestamp,
                "-to",
                end_timestamp,
                "-i",
                f,
                "-an",
                "-fps_mode",
                "passthrough",
                out / f.name,
            ],
            check=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        help="directory of input files",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="output directory",
    )
    parser.add_argument(
        "-c",
        type=float,
        default=0.9,
        help="cutoff boundary for how much of the frame can be black ",
    )
    parser.add_argument(
        "--intro_len",
        type=int,
        default=109,
        help="absolute frame length of intro before it goes black",
    )
    parser.add_argument(
        "--outro_len",
        type=int,
        default=250,
        help="absolute frame length of how long to check before end",
    )
    args = parser.parse_args()

    wrapper(
        Path(args.input_dir),
        Path(args.output_dir),
        c=args.c,
        intro_len=args.intro_len,
        outro_len=args.outro_len,
    )
