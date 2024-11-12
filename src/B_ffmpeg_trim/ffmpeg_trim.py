__doc__ = """
acts as a callwrapper around ffmpeg provided an entries file to cut out
timestamps from raw footage.

some links regarding exact cutting in ffmpeg and quality during reencoding:
    https://superuser.com/a/459488
    https://superuser.com/a/677580
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import tempfile
import subprocess
import json
from utils import naming_function
from collections import defaultdict
import argparse

from pprint import pprint

FPS: int = 25


def trim_wrapper(
    pred_dir: Path,
    matches: dict,
    export_dir: Path = Path(tempfile.gettempdir()),
    overwrite: bool = False,
    cropped_input_mappings: dict = {},
) -> None:
    """wrapper that handles the calls to the trim function"""
    raw_file: str
    matching_clips: list[str]
    for raw_file, matching_clips in matches.items():
        raw_pred_dir: Path = pred_dir.joinpath(naming_function(Path(raw_file)))
        if cropped_input_mappings:
            raw_file = cropped_input_mappings[raw_file]
        for clip in matching_clips:
            clip_json_path = raw_pred_dir.joinpath(Path(clip).with_suffix(".json").name)
            if not clip_json_path.exists():
                print(f"ERROR: {clip_json_path} not found")
                continue
            with open(clip_json_path, "r") as f:
                clip_json: dict = json.load(f)
            export_path: Path = export_dir.joinpath(Path(clip).with_suffix(".mp4").name)
            if export_path.exists():
                continue

            err: subprocess.CompletedProcess = trim(
                input_path=raw_file,
                output_path=str(export_path),
                start=clip_json["timestamps"]["start"],
                end=clip_json["timestamps"]["end"],
                overwrite=overwrite,
            )
            if err.returncode != 0:
                print("ERR during trimming:")
                print(err.stdout)


def trim(
    input_path: str, output_path, start: str, end: str, overwrite: bool
) -> subprocess.CompletedProcess:
    """
    calls ffmpeg to cut exactly using provided args. must be reencoded since
    https://superuser.com/questions/459313/how-to-cut-at-exact-frames-using-ffmpeg
    """
    print(f"== trimming {input_path} to {output_path}")
    err: subprocess.CompletedProcess = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            start,
            "-to",
            end,
            "-i",
            input_path,
            "-vcodec",
            "libx264",
            "-an",  # remove audio track
            # "-crf",
            # "18",
            "-preset",
            "ultrafast",
            "-filter:v",
            # apply deinterlacing; set FPS
            f"bwdif=mode=send_field:parity=auto:deint=all,fps={FPS}",
            "-fps_mode",  # ensure no duplicate frames are added
            "passthrough",
            "-y" if overwrite else "-n",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    return err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ffmpeg wrapper to trim according to entries.json file"
    )
    parser.add_argument(
        "--entries",
        help="path to entries file",
    )
    parser.add_argument(
        "--output_dir",
        help="path to desired output folder",
    )
    parser.add_argument(
        "--cropped_input_mapping",
        help="path to json which maps cropped files to originals",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="boolean flag to overwrite existing files and folders in output",
    )
    args = parser.parse_args()

    with open(args.entries, "r") as f:
        entries: dict = json.load(f)
        pred_dir: Path = Path(args.entries).parent

    cropped_input_mappings: dict = {}
    if args.cropped_input_mapping:
        with open(args.cropped_input_mapping) as f:
            cropped_input_mappings: dict = json.load(f)

    matches = defaultdict(list)
    for _, entry in entries.items():
        for clip in entry["matched_clips"]:
            matches[clip[2]].append(clip[1])

    export_dir: Path = Path(args.output_dir)
    if export_dir.exists() and not args.overwrite:
        print("FATAL: export dir already exists and overwrite wasn't specified")
        exit(1)
    export_dir.mkdir(exist_ok=True, parents=True)

    trim_wrapper(
        pred_dir=pred_dir,
        matches=matches,
        export_dir=export_dir,
        overwrite=args.overwrite,
        cropped_input_mappings=cropped_input_mappings,
    )
