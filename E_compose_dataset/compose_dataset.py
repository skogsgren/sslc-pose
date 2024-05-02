import json
from pathlib import Path
from dataclasses import dataclass, asdict
from decord import VideoReader
from datetime import timedelta, datetime
from pprint import pprint
from tqdm import tqdm
import re
import pandas as pd

import sys


@dataclass
class Annotation:
    start: str
    end: str
    token: str


@dataclass
class FrameAnn:
    start: int
    end: int
    token: str

@dataclass
class TokenData:
    token: str
    start: int
    end: int
    keypoints: list[list[float]]

def time_to_frame(timestamp: str, fps: int) -> int:
    """given a timestamp and fps converts that timestamp to a frame number"""
    timefmt: str = "%H:%M:%S.%f"
    if len(timestamp.split(".")) == 1:
        timefmt = timefmt[:-3]
    delta_obj: timedelta = datetime.strptime(timestamp, timefmt) - datetime.strptime(
        "0:00:00.000000", "%H:%M:%S.%f"
    )
    return int(fps * delta_obj.total_seconds())


def compose_dataset(
    eaf_annotations: dict[str, list[Annotation]], pose_path: Path, subset_path: Path, out_dir: Path
) -> None:
    output_time: str = datetime.now().strftime('%y%m%d%H%M') + "-"

    corr_filename: list[str] = []
    total_tokens: list[str] = []
    total_keypoints: list[list[list[float]]] = []
    total_frame_start: list[int] = []
    total_frame_end: list[int] = []

    for filename, annotations in tqdm(eaf_annotations.items()):
        fps: int = VideoReader(str(subset_path / filename)).get_avg_fps()

        with open(pose_path / Path(filename).with_suffix(".json"), "r") as f:
            pose_inferences: dict[int, list[float]] = {
                x["frame_id"]: x["instances"][0]["keypoints"] for x in json.load(f)
            }

        clip_tokens: list[TokenData] = []
        for annotation in annotations:
            # NOTE: ignoring those with invalid timestamp. In my testing this seems
            # to depend on the fact that some files are annotated when the subset
            # file itself is black (i.e. somebody has annotated the raw file and
            # then the person compressing the original has made the intro longer
            # than it should.
            if re.search(r"day", annotation.start) or re.search(r"day", annotation.end):
                continue
            frame_start: int = time_to_frame(annotation.start, fps)
            frame_end: int = time_to_frame(annotation.end, fps)
            keypoints: list[list[float]] = [pose_inferences[i] for i in range(frame_start, frame_end + 1)]
            clip_tokens.append(TokenData(
                annotation.token,
                frame_start,
                frame_end,
                keypoints
            ))

            corr_filename.append(filename)
            total_tokens.append(annotation.token)
            total_keypoints.append(keypoints)
            total_frame_start.append(frame_start)
            total_frame_end.append(frame_end)

        output_dir: Path = out_dir / (output_time + "sslc-pose-dataset")
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        with open(output_dir / Path(filename).with_suffix(".json"), "w") as f:
            json.dump([asdict(x) for x in clip_tokens], f)

    full_df: pd.DataFrame = pd.DataFrame({
        "video_file": corr_filename,
        "token": total_tokens,
        "keypoints": total_keypoints,
        "frame_start": total_frame_start,
        "frame_end": total_frame_end,
    })
    full_df.to_parquet(out_dir / (output_time + "sslc-pose-dataset.parquet"))



if __name__ == "__main__":
    with open("./cfg.json", "r") as f:
        cfg: dict = json.load(f)

    with open(cfg["eaf_annotations"], "r") as f:
        raw_ann: dict[str, list[dict[str, str]]] = json.load(f)
        ann: dict[str, list[Annotation]] = {}
        for k, v in raw_ann.items():
            ann[k] = [Annotation(x["start"], x["end"], x["annotation"]) for x in v]
    compose_dataset(
        ann,
        Path(cfg["pose_inference_path"]),
        Path(cfg["subset_path"]),
        Path(cfg["output_dir"])
    )
