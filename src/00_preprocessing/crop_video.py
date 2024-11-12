__doc__ = """
This file contains functions which crops video files according to the mean
aspect ratio of files suspected to be in them.

These functions are highly SSLC specific seeing as the SSLC clips have a
speaker id, and if one does a quick pass through the data indiscriminately
first, then one can compile a pretty good guess on which files contain which
clips. Running this file, then, crops the original video files to the average
aspect ratio (since it can differ) of those guesses, which in testing provides
much better and robust results.

"""

import subprocess
import argparse
from pathlib import Path
from collections import Counter
import json


def crop_video(video: Path, output_dir: Path, aspect_ratio: str):
    # since some files can have non-square pixels
    _, _, resolution = get_aspect_ratio(str(video))
    h, w = resolution
    ah, aw = tuple(aspect_ratio.split("*"))
    calc = int(h) / int(ah) * int(aw)
    if (int(w) - calc < 5) and (int(w) - calc > -5):
        aspect_ratio = "1"
    print(f"== calc={round(int(w) - calc, 2)}; crop=ih/{aspect_ratio}:ih")
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-i",
        str(video),
        "-vcodec",
        "h264",
        "-an",
        "-fps_mode",
        "passthrough",
        "-preset",
        "ultrafast",
        "-vf",
        f"crop=ih/{aspect_ratio}:ih",
        str(output_dir / (video.stem + ".mp4")),
    ]
    subprocess.run(args=args)


def get_aspect_ratio(input_file: str) -> tuple[str, str, tuple[str, str]]:
    """given input file path return (DAR, SAR, (h, w))"""
    args = [
        "ffprobe",
        "-i",
        input_file,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=display_aspect_ratio,width,height,sample_aspect_ratio,frame_aspect_ratio",
        "-of",
        "json=c=1",
    ]
    res = subprocess.run(args=args, capture_output=True, text=True, check=True)
    res = json.loads(res.stdout)["streams"][0]
    dar = res["display_aspect_ratio"]
    sar = res["sample_aspect_ratio"]
    w = res["width"]
    h = res["height"]
    return (dar, sar, (h, w))


def format_aspect_ratio(ar: str) -> str:
    """given output from ffmpeg, e.g. h:w return arithmetic string w*h"""
    w = ar.split(":")[0]
    h = ar.split(":")[1]
    return f"{h}*{w}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_files",
        help="list of files you want cropped (separated by space)",
        nargs="+",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="base name for directory where files should be exported (defaults to {parent of raw files}/crop)",
    )
    parser.add_argument(
        "-c",
        "--clip",
        type=str,
        help="path to clip to determine aspect ratio",
    )
    parser.add_argument(
        "--mapping_json",
        type=str,
        help="crop files using mapping json, do both export and create new json (filename={mapping_json}_crop.json)",
    )
    args = parser.parse_args()

    if args.mapping_json:
        new_cfg: dict = {}
        new_mappings: dict = {}
        with open(args.mapping_json, "r") as f:
            original_mappings = json.load(f)
        # we save the mappings in a dictionary {new_file}->{original_file}
        crop_mappings: dict[str, str] = {}
        for si in original_mappings:
            new_raw_files: list[str] = []

            # get most common aspect ratio for all clips
            aspect_ratio = Counter(
                [
                    format_aspect_ratio(get_aspect_ratio(x)[0])
                    for x in original_mappings[si]["subset_video_files"]
                ]
            )
            aspect_ratio = aspect_ratio.most_common(1)[0][0]

            if not args.output_dir:
                output_dir = Path(original_mappings[si]["raw_dir"]) / "crop"
            else:
                output_dir = Path(args.output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            if not output_dir.is_dir():
                raise FileExistsError(
                    f"{output_dir} already exists and is not a directory"
                )

            for raw_file in [Path(x) for x in original_mappings[si]["raw_files"]]:
                crop_video(raw_file, output_dir, aspect_ratio)
                cropped_filename: str = str(
                    (output_dir / (raw_file.stem + ".mp4")).absolute()
                )
                new_raw_files.append(cropped_filename)
                crop_mappings[cropped_filename] = str(raw_file.absolute())
            new_cfg[si] = {
                "raw_dir": original_mappings[si]["raw_dir"],
                "subset_video_files": original_mappings[si]["subset_video_files"],
                "raw_files": new_raw_files,
            }

        mjp: Path = Path(args.mapping_json)
        with open(mjp.parent.absolute() / (mjp.stem + "_crop.json"), "w") as f:
            json.dump(new_cfg, f)

        with open(mjp.parent.absolute() / (mjp.stem + "_crop_mappings.json"), "w") as f:
            json.dump(crop_mappings, f)

    else:
        aspect_ratio = format_aspect_ratio(get_aspect_ratio(args.clip)[0])
        output_dir: Path = Path(args.output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        if not output_dir.is_dir():
            raise FileExistsError(f"{output_dir} already exists and is not a directory")

        for file in [Path(x) for x in args.input_files]:
            crop_video(file, output_dir, aspect_ratio)
