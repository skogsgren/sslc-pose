__doc__ = """
parses eaf files to json files, adjusting the offset according to the offset in
the matches.
"""

from pympi.Elan import Eaf
from pathlib import Path
import json
from datetime import timedelta
import argparse

FPS: int = 25


def to_milliseconds(start_offset: int) -> int:
    """reads a clip and converts a offset to milliseconds using read fps"""
    return int(start_offset / FPS * 1000)


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
    if not parser.tiers.get(f"Glosa_DH S{speaker_id}"):
        if speaker_id == 3:
            print(
                f"{clip_file.name}; speaker_id=3, len(glosa_DH) == 2; trying 'Glosa_DH S2' instead..."
            )
            speaker_id = 2
        else:
            print(
                f"KEY_ERROR for {clip_file.name}; check eaf 'Glosa_DH S{speaker_id}' field"
            )
            print(f"\tPossible media files: {media_files}")
            return None
    ignore_counter = 0
    for _, annotation in parser.tiers[f"Glosa_DH S{speaker_id}"][0].items():
        start = parser.timeslots[annotation[0]] - start_offset
        end = parser.timeslots[annotation[1]] - start_offset
        if start < 0:
            ignore_counter += 1
            continue
        annotations.append(
            {
                "start": str(timedelta(seconds=start / 1000.0)),
                "end": str(timedelta(seconds=end / 1000.0)),
                "annotation": annotation[2],
            }
        )
    if ignore_counter != 0:
        print(
            f"WARN: ignored {ignore_counter} annotations before {clip_file} start"
            f"(EAF={eaf_file})"
        )
    return annotations


def parse_eaf_dir(eaf_dir: Path) -> dict[str, Path]:
    """create a dict {clip_filename -> eaf_filepath}"""
    eaf_dict: dict[str, Path] = {}
    for f in eaf_dir.iterdir():
        if f.suffix != ".eaf":
            continue
        parser: Eaf = Eaf(f)
        for i in parser.media_descriptors:
            eaf_dict[Path(i["RELATIVE_MEDIA_URL"]).name] = f
    return eaf_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="matches and exports elan annotations according to provided clip offsets (i.e. how long intro/outro is )"
    )
    parser.add_argument(
        "--eaf_dir",
        required=True,
        help="path to directory with eaf files",
    )
    parser.add_argument(
        "--clip_dir",
        required=True,
        help="path to directory with clips",
    )
    parser.add_argument(
        "--entries",
        required=True,
        help="path to entries json",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="path to where annotations should be copied to",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        required=False,
        help="overwrite output directory",
    )
    parser.add_argument(
        "--non_matching_clips",
        action="store_true",
        required=False,
        default=False,
        help="also align annotations for non matching clips according to entries",
    )
    args = parser.parse_args()

    output_dir: Path = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        print(f"ERR: {output_dir} exists and overwrite is not specified.")
        exit(1)
    if output_dir.is_file():
        print(f"ERR: {output_dir} exists and is a file.")
        exit(1)
    output_dir.mkdir(exist_ok=True, parents=True)

    # we want an iterable datastructure that contains tuples of (clip, original)
    with open(args.entries, "r") as f:
        entries: dict = json.load(f)
    clips = []
    for _, data in entries.items():
        clips += [(Path(x[1]), x[2]) for x in data["matched_clips"]]
        if args.non_matching_clips:
            clips += [(Path(x[1]), x[2]) for x in data["unmatched_clips"]]

    eaf_map: dict[str, Path] = parse_eaf_dir(Path(args.eaf_dir))

    for clip, raw_file in clips:
        if not eaf_map.get(clip.name):
            print(f"== {clip} not found in {args.eaf_dir}. skipping...")
            continue

        # we have to define our start offset (i.e. how much to subtract from annotations)
        if clip.with_suffix(".json").exists():
            with open(clip.with_suffix(".json")) as f:
                crop_data = json.load(f)
            start_offset: int = to_milliseconds(int(crop_data["start_offset"]))
        else:
            print(f"== {clip.with_suffix('.json')} not found. using offset of 0...")
            start_offset: int = 0

        annotations: list[dict] | None = parse(
            eaf_file=eaf_map[clip.name],
            clip_file=clip,
            start_offset=start_offset,
        )

        if not annotations:
            print(
                f"ERR: something went wrong during extraction of annotations for {clip.name}"
            )
        else:
            with open(output_dir / clip.with_suffix(".json").name, "w") as f:
                json.dump(annotations, f)
