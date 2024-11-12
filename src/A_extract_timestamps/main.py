__doc__ = """
main.py is a CLI wrapper around double_pass_extract.py in order to extract
timestamps using config present in ./cfg.json (by default, though the main
function accepts another path)
"""

import json
from pathlib import Path

import decord
from double_pass_extract import double_pass_timestamp_extraction, read_video_file
import concurrent.futures
from tqdm import tqdm
import numpy as np
import argparse

import logging

from utils import naming_function

logging.basicConfig(
    filename="runtime.log",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


def main(cfgp: str = "./cfg.json") -> None:
    """given the path to a cfg json attempts to match every subset file
    specified in that configuration with each specified raw file, given the
    settings specified in the json."""
    logging.info("STARTING RUN")
    with open(cfgp, "r") as f:
        cfg: dict = json.load(f)
    cache_dir: Path = Path(cfg["cache_dir"])
    pred_dir: Path = Path(cfg["pred_dir"])
    if not cache_dir.exists():
        cache_dir.mkdir()
    if not pred_dir.exists():
        pred_dir.mkdir()

    with open(cfg["speaker_video_mappings"], "r") as f:
        speaker_video_mappings: dict = {
            int(sid): mappings for sid, mappings in json.load(f).items()
        }
    if pred_dir.joinpath("progress.json").exists():
        logging.info("resuming previous run...")
        with open(pred_dir / "progress.json", "r") as f:
            already_checked_ids = json.load(f)
        for speaker_id in already_checked_ids:
            if speaker_id in list(speaker_video_mappings.keys()):
                del speaker_video_mappings[speaker_id]
    for speaker_id, video_mappings in tqdm(speaker_video_mappings.items()):
        for i, raw_file in enumerate(sorted(video_mappings["raw_files"])):
            logging.info(
                f"{i+1}/{len(video_mappings['raw_files'])}"
                " "
                f"PROCESSING {Path(raw_file).name} for ID={speaker_id}"
            )
            cached_raw_file: Path = cache_dir.joinpath(
                naming_function(Path(raw_file)) + ".npy"
            )
            if not cached_raw_file.exists():
                logging.info(f"{cached_raw_file} not found.")
                logging.info(f"creating matrix for {raw_file}")
                try:
                    original_matrix: np.ndarray = read_video_file(raw_file)
                    np.save(str(cached_raw_file), original_matrix)
                except decord.DECORDError:
                    logging.error(f"{raw_file} file is damaged. ignoring...")
                    continue
            else:
                logging.info(f"found existing matrix for {raw_file}")
                original_matrix: np.ndarray = np.load(cached_raw_file)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=cfg["workers"] if cfg.get("workers") else None
            ) as executor:
                timestamps: list = []
                for clip in video_mappings["subset_video_files"]:
                    timestamps.append(
                        executor.submit(
                            double_pass_timestamp_extraction,
                            Path(raw_file),
                            original_matrix,
                            Path(clip),
                            cache_dir,
                            cfg["match_decision_boundary"],
                            pred_dir,
                            cfg["early_stopping_threshold"],
                            cfg["n_comparison"],
                        )
                    )

        if not (pred_dir / "progress.json").exists():
            with open(pred_dir / "progress.json", "w") as f:
                json.dump([], f)
        with open(pred_dir / "progress.json", "r") as f:
            progress = json.load(f)
        progress.append(speaker_id)
        with open(pred_dir / "progress.json", "w") as f:
            json.dump(progress, f)

    logging.info("FINISHED RUN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cfg",
        type=str,
        default="./cfg.json",
        help="path to config json (default=./cfg.json)",
    )
    args = parser.parse_args()
    main(args.cfg)
