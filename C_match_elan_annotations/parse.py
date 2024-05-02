"""
exports a folder of elan .eaf files to a json file using the configuration in
cfg.json.

this needs a json file of offsets to work, generate by using export_offsets.py
to generate one using the information from ../A_extract_timestamps/

USAGE:
    python3 parse.py
"""

from pympi.Elan import Eaf
from pathlib import Path
import json
from decord import VideoReader
from datetime import timedelta, datetime
from tqdm import tqdm


def to_milliseconds(clip_path: str, start_offset: int) -> int:
    # TODO: implement function that given a path to a clip and start offset in
    # number of frames, converts that offset to milliseconds by taking into
    # account the FPS of the clip.
    fps: int = VideoReader(clip_path).get_avg_fps()
    return int(start_offset / fps * 1000)


def parse_wrapper(
    eaf_dir: Path,
    pred_dir: Path,
    matches: dict,
    offsets: dict,
) -> dict:
    annotations: dict = {}

    # TODO: create a dict {clip_file_name -> eaf_file_name}
    eaf_dict: dict[str, Path] = {}
    for f in eaf_dir.iterdir():
        if f.suffix != ".eaf":
            continue
        parser: Eaf = Eaf(f)
        for i in parser.media_descriptors:
            eaf_dict[Path(i["RELATIVE_MEDIA_URL"]).name] = f

    clips: list[str]
    for _, clips in tqdm(matches.items()):
        c: str
        for c in clips:
            clip: Path = Path(c)
            if eaf_dict.get(clip.name, None) is None:
                continue
            parsed_eaf: list | None = parse(
                eaf_dict[clip.name],
                clip,
                to_milliseconds(str(clip), offsets[clip.name]["start_offset"]),
            )
            if parsed_eaf is None:
                print(f"ERROR: couldn't find {clip.name} in EAF folder")
            annotations[clip.name] = parsed_eaf

    return annotations


def parse(eaf_file: Path, clip_file: Path, start_offset: int) -> list | None:
    annotations: list[dict[str, str | int]] = []
    parser: Eaf = Eaf(eaf_file)

    media_files: dict[str, int] = {
        Path(parser.media_descriptors[i]["RELATIVE_MEDIA_URL"]).name: i + 1
        for i, _ in enumerate(parser.media_descriptors)
    }
    if clip_file.name not in media_files:
        return None
    speaker_id: int = media_files[clip_file.name]

    annotation: tuple[str, str, str, None]
    """
        guide to indices:
            0: timestamp id of the beginning of the annotation
            1: timestamp id of the end of the annotation
            2: the actual annotation content
            3: unknown (is always set to None?)
    """
    for _, annotation in parser.tiers[f"Glosa_DH S{speaker_id}"][0].items():
        annotations.append(
            {
                "start": str(
                    timedelta(
                        seconds=(
                            parser.timeslots[annotation[0]] - start_offset
                        )
                        / 1000.0
                    )
                ),
                "end": str(
                    timedelta(
                        seconds=(
                            parser.timeslots[annotation[1]] - start_offset
                        )
                        / 1000.0
                    )
                ),
                "annotation": annotation[2],
            }
        )
    return annotations


if __name__ == "__main__":
    with open("cfg.json", "r") as f:
        cfg: dict = json.load(f)
    with open(cfg["matches"], "r") as f:
        matches: dict = json.load(f)
    with open(cfg["offsets"], "r") as f:
        offsets: dict = json.load(f)
    annotations: dict = parse_wrapper(
        eaf_dir=Path(cfg["eaf_dir"]),
        pred_dir=Path(cfg["pred_dir"]),
        matches=matches,
        offsets=offsets,
    )

    export_dir: Path = Path(cfg["export_dir"])
    if export_dir.exists():
        assert export_dir.is_dir()
    else:
        export_dir.mkdir()
    timestamp: str = datetime.now().strftime("%y%m%d-%H%M") + "-"
    with open(
        export_dir / (timestamp + "eaf_annotation_export.json"), "w"
    ) as f:
        json.dump(annotations, f)
