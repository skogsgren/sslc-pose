__doc__ = """
This file contains functions to help creating a entries.json file. This file
creates a readable overview of match progress using a configuration file (to
get variables like match_boundary etc).

NOTE: these functions are currently SSLC hardcoded and many of the functions
would have to be changed in order to be used in another dataset.
"""


import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd
import argparse


def esi(fn: str) -> int:
    """extracts speaker id from path object"""
    return int(re.findall(r"\d+", Path(fn).name)[-1])


def vidx(fn: str) -> int:
    """extracts video id from path object"""
    return int("".join(re.findall(r"\d+", Path(fn).name)[:2]))


def get_top_candidates(clip: str, pred_dir: Path):
    """given a clip name returns the top candidates for all raw files in pred_dir"""
    candidates = pred_dir.rglob(clip)
    candidate_scores = []
    for candidate in candidates:
        if "debug" in candidate.stem:
            continue
        with open(candidate, "r") as f:
            candidate_data = json.load(f)
        candidate_scores.append(
            (
                candidate_data["in_original_calc"],
                candidate_data["filename"],
                candidate_data["raw_file"],
                candidate_data["timestamps"],
            )
        )
    return sorted(candidate_scores, reverse=True)


def recursive_generate_result(pred_dir: Path) -> dict[str, list[str]]:
    """generates a result dictionary with all files currently matched"""
    all_clips = pred_dir.rglob("SSLC*.json")
    clips: list[str] = list({x.name for x in all_clips})
    result_dict: defaultdict = defaultdict(list)
    print("generating result...")
    for clip in tqdm(clips):
        candidate_scores = get_top_candidates(clip, pred_dir)
        if not candidate_scores:
            continue
        result_dict[candidate_scores[0][2]].append(candidate_scores[0][1])
    return dict(sorted(result_dict.items()))


def generate_entries(pred_dir: Path, cfg_path: str) -> dict:
    """generates entry dict which group results according to speaker id"""
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    all_clips = [x for x in pred_dir.rglob("SSLC*.json") if "debug" not in x.stem]
    clips: list[str] = list({x.name for x in all_clips})
    entries: dict = {}
    for speaker_id in {esi(x) for x in clips}:
        entries[speaker_id] = {"matched_clips": [], "unmatched_clips": []}
    print("generating entries...")
    for clip in tqdm(clips):
        top_candidates = get_top_candidates(clip, pred_dir)
        if not top_candidates:
            entries[esi(clip)]["unmatched_clips"].append((0, clip, "N/A"))
        if top_candidates[0][0] > cfg["match_decision_boundary"]:
            entries[esi(clip)]["matched_clips"].append(top_candidates[0])
        else:
            entries[esi(clip)]["unmatched_clips"].append(top_candidates[0])

        for m in ["unmatched_clips", "matched_clips"]:
            entries[esi(clip)][m] = sorted(
                entries[esi(clip)][m], key=lambda x: vidx(x[1])
            )
    return entries


def get_speaker_id_folder_mapping(pred_dir: Path) -> dict:
    """given a pred_dir, calculates the most likely parent folder for every speaker id"""
    assert (pred_dir / "entries.json").exists()
    with open(pred_dir / "entries.json", "r") as f:
        entries = json.load(f)
    with open("cfg.json", "r") as f:
        all_subset_clips = json.load(f)["subset_video_files"]
    speaker_id_clips: defaultdict = defaultdict(list)
    for clip in all_subset_clips:
        speaker_id = esi(Path(clip).stem)
        speaker_id_clips[speaker_id].append(clip)

    most_likely_dict: dict[int, dict] = {}
    for speaker_id, entry in entries.items():
        entry_clips = entry["unmatched_clips"] + entry["matched_clips"]
        most_likely = Counter([Path(x[2]).parent.parent for x in entry_clips])
        most_likely_dict[int(speaker_id)] = {
            "raw_dir": str(most_likely.most_common()[0][0]),
            "subset_video_files": speaker_id_clips[int(speaker_id)],
        }
    return dict(sorted(most_likely_dict.items()))


def recursive_get_raw_files(attrib_json: Path) -> dict:
    """given attributions (i.e. mappings from speaker ids to raw_file parent
    folders), recursively searches that directory for raw_files"""
    with open(attrib_json, "r") as f:
        attributions: dict = json.load(f)
    for speaker_id, entry in attributions.items():
        rp = Path(entry["raw_dir"])
        candidates = list(rp.rglob("*.mov")) + list(rp.rglob("*.mp4"))
        attributions[speaker_id]["raw_files"] = [str(x) for x in candidates]
    return {int(k): v for k, v in sorted(attributions.items())}


def correct_entries_using_annotated_csv(entries: dict, csv_path: Path) -> dict:
    """corrects the entries dict according to annotated csv file"""
    df: pd.DataFrame = pd.read_csv(csv_path, comment="#")
    mutated_entries: dict = {}
    for speaker_id, entry in entries.items():
        mutated_entries[speaker_id] = {
            "matched_clips": [],
            "unmatched_clips": [],
        }

        entry_clips = entry["unmatched_clips"] + entry["matched_clips"]
        for clip in entry_clips:
            clip_path = Path(clip[1])
            label = df.loc[df["FILENAME"] == clip_path.name]["LABEL"]
            if int(label) == 0:
                mutated_entries[speaker_id]["matched_clips"].append(clip)
            else:
                mutated_entries[speaker_id]["unmatched_clips"].append(clip)

        for m in ["unmatched_clips", "matched_clips"]:
            mutated_entries[speaker_id][m] = sorted(
                mutated_entries[speaker_id][m], key=lambda x: vidx(x[1])
            )
    return mutated_entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", default="cfg.json")
    parser.add_argument("--annotated_corrections")
    parser.add_argument("dir")
    args = parser.parse_args()
    out = Path(args.dir)
    entries = generate_entries(out, cfg_path=args.cfg)
    if args.annotated_corrections:
        entries = correct_entries_using_annotated_csv(
            entries=entries,
            csv_path=Path(args.annotated_corrections),
        )
    with open(out / "entries.json", "w") as f:
        json.dump(entries, f)
