"""
main.py is a CLI wrapper around extract.py in order to extract timestamps using
config present in ./cfg.json (by default, though the main function accepts another path)
"""

import json
from pathlib import Path
from pprint import pprint

import decord
from extract import extract_timestamps, readVideoFile
import concurrent.futures
from tqdm import tqdm
import numpy as np
import gc

import logging

logging.basicConfig(
    filename="runtime.log",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


def update_raw_json(pred_dir: Path, rfp: Path, matches: list[str]) -> None:
    """helper function to update the result json"""
    logging.info(f"writing matches for {rfp} to {pred_dir}/results.json")
    if pred_dir.joinpath("results.json").exists():
        with open(pred_dir.joinpath("results.json"), "r") as f:
            jsr: dict = json.load(f)
    else:
        jsr: dict = {}
    logging.info(f"setting k/v pairs in raw.json for {rfp}")
    jsr[str(rfp.absolute())] = matches

    with open(pred_dir.joinpath("results.json"), "w") as f:
        json.dump(jsr, f)


def main(cfgp: str = "./cfg.json") -> None:
    """given the path to a cfg json attempts to match every subset file
    specified in that configuration with each specified raw file, given the
    settings specified in the json."""
    logging.info("STARTING RUN")
    with open(cfgp, "r") as f:
        cfg: dict = json.load(f)
    cache_dir: Path = Path(cfg["cache_dir"])
    pred_dir: Path = Path(cfg["pred_dir"])
    raw_video_files: list[str] = cfg["raw_video_files"]
    subset_video_files: list[str] = cfg["subset_video_files"]

    if not cache_dir.exists():
        cache_dir.mkdir()
    if not pred_dir.exists():
        pred_dir.mkdir()

    if pred_dir.joinpath("results.json").exists():
        logging.info("found previous results.json file; resuming previous run...")
        with open(pred_dir.joinpath("results.json"), "r") as f:
            tmp: dict[str, list[str]] = json.load(f)
        raw_video_files = [x for x in raw_video_files if x not in list(tmp.keys())]
        i: list[str]
        for i in tmp.values():
            subset_video_files = [x for x in subset_video_files if x not in i]

    raw_file: str
    for raw_file in tqdm(raw_video_files):
        matches: list[str] = []

        cached_raw_file: Path = cache_dir.joinpath(
            Path(Path(raw_file).name).with_suffix(".npy")
        )
        logging.info(f"checking if {cached_raw_file} exists...")
        if not cached_raw_file.exists():
            logging.info(f"{cached_raw_file} not found.")
            logging.info(f"creating matrix for {raw_file}")
            try:
                original_matrix: np.ndarray = readVideoFile(raw_file)
                np.save(str(cached_raw_file), original_matrix)
            except decord.DECORDError:
                logging.error(
                    f"{raw_file} file cannot be processed (possibly damaged). ignoring..."
                )
                with open(pred_dir.joinpath("results.json"), "r") as f:
                    res: dict = json.load(f)
                res[raw_file] = ["damaged"]
                with open(pred_dir.joinpath("results.json"), "w") as f:
                    json.dump(res, f)
                continue
        else:
            logging.info(f"found existing matrix for {raw_file}")
            original_matrix: np.ndarray = np.load(cached_raw_file)

        with tqdm(total=len(subset_video_files)) as pbar:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=cfg["workers"]
                if cfg.get("workers", None) is not None
                else None
            ) as executor:
                timestamps: list = []
                for subset_file in subset_video_files:
                    timestamps.append(
                        executor.submit(
                            extract_timestamps,
                            Path(raw_file),
                            original_matrix,
                            Path(subset_file),
                            cache_dir,
                            cfg["match_decision_boundary"],
                            pred_dir,
                            cfg["abs_intro_length"],
                            cfg["abs_outro_length"],
                            cfg["early_stopping_threshold"],
                            cfg["step"],
                            cfg["skip_factor"],
                            cfg["n_comparison"],
                        )
                    )

                for f in concurrent.futures.as_completed(timestamps):
                    if f.result() is None:
                        pbar.update(1)
                        logging.info("skipping...")
                        continue
                    t: dict = f.result()
                    if t["in_original"] == 1:
                        subset_video_files = [
                            x for x in subset_video_files if x != t["filename"]
                        ]
                        matches.append(t["filename"])
                        pbar.update(1)

        update_raw_json(rfp=Path(raw_file), matches=matches, pred_dir=pred_dir)
        # trying to get to the bottom of memory leak problematics
        del original_matrix
        gc.collect()
    logging.info("FINISHED RUN")


if __name__ == "__main__":
    main()
