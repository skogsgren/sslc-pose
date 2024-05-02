"""
Automatic extraction does not always work. A lot of the tricks used to speed up
(mostly early stopping and frame skip) can sometimes make it so that the
algorithm misses to match up certain clips with their respective raw video
file. Precision is after all more important, but that isn't to say that recall
isn't. This file contains functions to aid in semi-manually increasing recall.

USAGE:

First call attrib_statistics using e.g.

```python
from check_speaker_id import get_attribution_statistics
from pprint import pprint

attrib_statistics: list[AttribEntry] = [x for x in get_attribution_statistics()]
pprint(attrib_statistics)
```

Check through the list and look if you spot some cases where you have some
matched files (i.e. more than one/two) so that you can be sure that the raw
files _actually_ contain the unmatched clips. Then you can use the recheck_files function.

Let's assume that the first index in the list of AttribEntries looks like this:

```
AttribEntry(speaker_id=42,
raw_files=['/hd/sts-raw/110322/428_0049.mov',
           '/hd/sts-raw/110322/428_0050.mov',
matched_clips=['/hd/sts-kor/SSLC02_413_S042_b.mp4',
               '/hd/sts-kor/SSLC01_400_S042_b.mp4',
               '/hd/sts-kor/SSLC01_401_S042_b.mp4',
               '/hd/sts-kor/SSLC01_402_S042_b.mp4',
               '/hd/sts-kor/SSLC01_403_S042_b.mp4',
               '/hd/sts-kor/SSLC01_404_S042_b.mp4',
               '/hd/sts-kor/SSLC01_405_S042_b.mp4',
               '/hd/sts-kor/SSLC01_406_S042_b.mp4',
               '/hd/sts-kor/SSLC01_407_S042_b.mp4',
               '/hd/sts-kor/SSLC01_408_S042_b.mp4'],
unmatched_clips=['/hd/sts-kor/SSLC02_409_S042_b.mp4',
                 '/hd/sts-kor/SSLC02_410_S042_b.mp4',
                 '/hd/sts-kor/SSLC02_411_S042_b.mp4',
                 '/hd/sts-kor/SSLC02_412_S042_b.mp4'],
)
```

You can then use recheck_files like so:

```python
from check_speaker_id import get_attribution_statistics, recheck_files

attrib_statistics: list[AttribEntry] = [x for x in get_attribution_statistics()]
recheck_files(attrib_statistics(
    attrib_statistics[0].raw_files,
    attrib_statistics[0].unmatched_clips
))
```

There may be cases where the files don't have any raw_files or matched_clips.
In those cases use the get_top_candidate_for_speaker_id function to print the
top candidates of raw files for that speaker id.

"""

from collections import defaultdict
from pathlib import Path
import json

from decord import VideoReader
from extract import extract_timestamps, get_pred_suffix, find_offsets
from main import update_raw_json
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from datetime import timedelta, datetime

from pprint import pprint


def esi(fn: str) -> int:
    """extracts speaker id from path object"""
    return int(Path(fn).name[-8:-6])


def format_cached_raw_file(fn: str, cache_dir: Path = Path("./cache")) -> Path:
    """since two conventions were used, format accordingly if one exists or the other"""
    base_name: Path = Path(cache_dir.joinpath(Path(fn).with_suffix(".npy").name))
    if base_name.exists():
        return base_name
    return cache_dir.joinpath(get_pred_suffix(Path(fn))).with_suffix(".npy")


def recheck_files(
    raw_video_files: list[str],
    unmatched_subset_video_files: list[str],
    cfg_json: Path = Path("./cfg.json"),
) -> None:
    """wrapper function that checks for timestamps for clips it knows
    reasonably should be in the raw files provided.

    you provide the following:
        raw_video_files:
            a list of strings (filepaths) to raw video files that you suspect
            the clips in unmatched_subset_video_files to be in.
        unmatched_subset_video_files:
            a list of strings (filepaths) to subset video clips that are not
            yet matched to a raw video file.
        cfg_json:
            path to a json file containing configuration (see extract.py for
            more information)
    """
    with open(cfg_json, "r") as f:
        cfg: dict = json.load(f)
    with open(Path(cfg["pred_dir"]).joinpath("results.json"), "r") as f:
        prev_res: dict = json.load(f)

    for raw_file in tqdm(raw_video_files):
        if prev_res.get("raw_file"):
            matches: list[str] = [x for x in prev_res[raw_file]]
        else:
            matches: list[str] = []
        original_matrix = np.load(format_cached_raw_file(raw_file))
        for subset_file in tqdm(unmatched_subset_video_files):
            if not Path(subset_file).exists():
                print(f"ERROR: {subset_file} does not exist, skipping...")
                continue
            timestamps: dict | None = extract_timestamps(
                Path(raw_file),
                original_matrix,
                Path(subset_file),
                Path(cfg["cache_dir"]),
                cfg["match_decision_boundary"],
                Path(cfg["pred_dir"]),
                cfg["abs_intro_length"],
                cfg["abs_outro_length"],
                9999,  # increase early stop threshold to make it not greedy
                1,  # skip no frames of raw file to make sure
                cfg["skip_factor"],
                cfg["n_comparison"],
            )

            if timestamps is None:
                continue
            if not timestamps["in_original"]:
                continue
            matches.append(timestamps["filename"])
            unmatched_subset_video_files = [
                x for x in unmatched_subset_video_files if x != subset_file
            ]
        update_raw_json(Path(cfg["pred_dir"]), Path(raw_file), matches)


@dataclass
class AttribEntry:
    """stores relevant attribution information"""

    speaker_id: int
    raw_files: list[str]
    matched_clips: list[str]
    unmatched_clips: list[str]


def get_attribution_statistics() -> list[AttribEntry]:
    """creates a list of statistics for video clip attribution"""
    with open("./cfg.json", "r") as f:
        cfg: dict = json.load(f)
        speaker_ids: set[int] = set([esi(x) for x in cfg["subset_video_files"]])
    with open(Path(cfg["pred_dir"]).joinpath("results.json"), "r") as f:
        res: dict = json.load(f)

    all_matched_clips: list[str] = []
    for k, v in res.items():
        for clip in v:
            if clip == ["damaged"]:
                continue
            all_matched_clips.append(clip)

    attrib_entries: list[AttribEntry] = []
    i: int
    for i in speaker_ids:
        raw_files: list[str] = []
        for k, v in res.items():
            if v == ["damaged"]:
                continue
            if i in [esi(x) for x in v]:
                raw_files.append(k)

        sid_clips: list[str] = [x for x in cfg["subset_video_files"] if esi(x) == i]
        attrib_entries.append(
            AttribEntry(
                speaker_id=i,
                raw_files=raw_files,
                matched_clips=[x for x in sid_clips if x in all_matched_clips],
                unmatched_clips=[x for x in sid_clips if x not in all_matched_clips],
            )
        )
    return attrib_entries


def get_top_candidates(vn: Path) -> list[tuple[str, float]]:
    """given a video clip name, returns the top candidates for that video clip"""
    with open("./cfg.json", "r") as f:
        cfg: dict = json.load(f)
    res: dict[str, float] = {}
    for i in Path(cfg["pred_dir"]).iterdir():
        if not i.is_dir():
            continue
        tested_clips: list[str] = [x.name for x in i.iterdir() if x.suffix == ".json"]
        vnj: str = vn.with_suffix(".json").name
        if vnj not in tested_clips:
            continue
        with open(i.joinpath(vnj), "r") as f:
            res[i.name] = json.load(f)["in_original_calc"]
    return list(reversed(sorted([(n, v) for n, v in res.items()], key=lambda x: x[1])))


def get_top_candidate_for_speaker_id(
    sid: int, clips: list[Path]
) -> list[tuple[str, float]]:
    """given a speaker id, gets the top candidates of raw files for that speaker id"""
    res: defaultdict[str, float] = defaultdict(float)
    for clip in clips:
        clip_res = get_top_candidates(clip)
        for k, v in clip_res:
            res[k] += v
    return list(reversed(sorted([(n, v) for n, v in res.items()], key=lambda x: x[1])))


def will_it_sandwich(filling: str) -> str | bool:
    """finds out if an unmatched clip can occur in between the two closest
    matching clip indices"""
    attrib_statistics: dict[int, AttribEntry] = {
        entry.speaker_id: entry for entry in get_attribution_statistics()
    }
    speaker_id: int = esi(filling)

    # append the unmatched clip to make it easier to get the closest indices
    attrib_statistics[speaker_id].matched_clips.append(filling)
    sorted_clips: list[str] = sorted(
        attrib_statistics[speaker_id].matched_clips, key=lambda x: int(x.split("_")[-3])
    )

    ci: int = sorted_clips.index(filling)

    if ci == 0:
        return f"{filling} occurs before any other clips for that speaker id"
    elif ci + 1 == len(sorted_clips):
        return f"{filling} occurs last of any other clips for that speaker id"

    upper_bun: str = sorted_clips[sorted_clips.index(filling) - 1]
    bottom_bun: str = sorted_clips[sorted_clips.index(filling) + 1]

    with open("./cfg.json", "r") as f:
        cfg: dict = json.load(f)
    with open(Path(cfg["pred_dir"]) / "results.json", "r") as f:
        results: dict = json.load(f)

    def gt(path: Path) -> dict:
        with open(path, "r") as f:
            return json.load(f)["timestamps"]

    for raw_file in attrib_statistics[speaker_id].raw_files:
        upper_bun_end: str | None = None
        bottom_bun_start: str | None = None
        base_path: Path = Path(cfg["pred_dir"]) / get_pred_suffix(Path(raw_file))
        if upper_bun in results[raw_file]:
            upper_bun_end = gt(base_path / Path(upper_bun).with_suffix(".json").name)[
                "end"
            ]
        if bottom_bun in results[raw_file]:
            bottom_bun_start = gt(
                base_path / Path(bottom_bun).with_suffix(".json").name
            )["start"]

        if upper_bun_end and bottom_bun_start:

            def dt_format_helper(string: str) -> datetime:
                try:
                    return datetime.strptime(string, "%H:%M:%S.%f")
                except ValueError:
                    return datetime.strptime(string, "%H:%M:%S")

            upper_bun_dt: datetime = dt_format_helper(upper_bun_end)
            bottom_bun_dt: datetime = dt_format_helper(bottom_bun_start)

            fps_filling: int = VideoReader(filling).get_avg_fps()
            offsets_filling: dict[str, int] = find_offsets(
                filling,
                abs_intro_length=cfg["abs_intro_length"],
                abs_outro_length=cfg["abs_outro_length"],
            )
            seconds_filling: float = (
                offsets_filling["end_offset"] - offsets_filling["start_offset"]
            ) / fps_filling
            filling_dt: timedelta = timedelta(seconds=seconds_filling)

            if filling_dt <= (bottom_bun_dt - upper_bun_dt):
                return True
            else:
                return False
    return "span not found in any raw files"


if __name__ == "__main__":
    attrib_statistics: list[AttribEntry] = get_attribution_statistics()
    pprint(attrib_statistics)
