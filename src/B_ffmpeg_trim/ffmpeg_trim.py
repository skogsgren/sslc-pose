"""
acts as a callwrapper around ffmpeg provided the configuration in cfg.json

some links regarding exact cutting in ffmpeg and quality during reencoding:
    https://superuser.com/a/459488
    https://superuser.com/a/677580
"""


from pathlib import Path
import tempfile
import subprocess
import json
import logging
from utils import parse_pred_name
import sys

logging.basicConfig(
    filename="runtime.log",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


def trim_wrapper(
    pred_dir: Path, matches: dict, export_dir: Path = Path(tempfile.gettempdir())
) -> None:
    """ wrapper that handles the calls to the trim function """
    if not export_dir.exists():
        export_dir.mkdir()
    if not export_dir.is_dir():
        print("FATAL: export_dir exists and is not a directory.")
        sys.exit(1)
    raw_file: str
    matching_clips: list[str]
    for raw_file, matching_clips in matches.items():
        raw_pred_dir: Path = pred_dir.joinpath(parse_pred_name(Path(raw_file)))
        for clip in matching_clips:
            clip_json_path = raw_pred_dir.joinpath(Path(clip).with_suffix(".json").name)
            if not clip_json_path.exists():
                logging.error(f"ERROR: {clip_json_path} not found")
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
                    end=clip_json["timestamps"]["end"]
            )
            if err.returncode != 0:
                logging.error(err.stderr)

def trim(
    input_path: str, output_path, start: str, end: str
) -> subprocess.CompletedProcess:
    """
    calls ffmpeg to cut exactly using provided args. must be reencoded since
    https://superuser.com/questions/459313/how-to-cut-at-exact-frames-using-ffmpeg
    """
    err: subprocess.CompletedProcess = subprocess.run(
        [
            "ffmpeg",
            "-ss",
            start,
            "-to",
            end,
            "-i",
            input_path,
            "-vcodec",
            "libx264",
            "-an", # remove audio track
            "-crf",
            "18",
            "-preset",
            "slow",
            "-filter:v",
            "bwdif=mode=send_field:parity=auto:deint=all", # apply deinterlacing
            "-y",
            output_path,
        ]
    )
    return err


if __name__ == "__main__":
    with open("cfg.json", "r") as f:
        cfg: dict = json.load(f)
    with open(cfg["results_json"], "r") as f:
        matches: dict = json.load(f)
    trim_wrapper(Path(cfg["pred_dir"]), matches, Path(cfg["export_dir"]))
