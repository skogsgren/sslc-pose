__doc__ = """
    subprocess wrapper around the inference script in MMPOSE. It includes
    parallel capabilities, seeing that the inference script only takes up 600MB
    of GPU memory. configuration is done in a separate json file, detailed
    below.

    == JSON SPECIFICATION ==
    workers: number of parallel workers
    script_path: path to inferencer_demo.py in MMPOSE (demo/inferencer_demo in
        their github repo)
    model_name: name of MMPOSE model
    input_dir: path to clips where pose inference is to be performed
    pred_dir: path to directory where pose predictions are to be placed

"""

import concurrent.futures
import subprocess
import json
import logging
import os
import sys
from pathlib import Path
import argparse

logging.basicConfig(
    filename="runtime.log",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


def cmd_wrapper(args: list[str]) -> None:
    """runs command provided list of arguments. logs error if it occurs"""
    err: subprocess.CompletedProcess = subprocess.run(
        args, check=False, capture_output=True
    )
    if err.returncode != 0:
        logging.error(
            err.stdout.decode("utf-8").strip(),
            err.stderr.decode("utf-8").strip(),
        )


def main(cfg_path: Path) -> None:
    logging.info("STARTING RUN")
    logging.info("opening config")
    with open(cfg_path, "r") as f:
        cfg: dict = json.load(f)
    logging.info("creating output directory if it doesn't exist")
    pred_dir: Path = Path(cfg["pred_dir"])
    if not pred_dir.exists():
        pred_dir.mkdir()
    if not pred_dir.is_dir():
        print("FATAL: pred_dir exists and is not a directory. exiting...")
        sys.exit(1)
    logging.info("checking output folder contents (if continuing)")
    pred_dir_contents: list[str] = []
    for i in Path(cfg["pred_dir"]).iterdir():
        pred_dir_contents.append(i.name)
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg["workers"]) as executor:
        for i in Path(cfg["input_dir"]).iterdir():
            if i.suffix != ".mp4":
                continue
            if i.with_suffix(".json").name in pred_dir_contents:
                continue
            executor.submit(
                cmd_wrapper,
                [
                    "python3",
                    os.path.expandvars(cfg["script_path"]),
                    str(i.absolute()),
                    "--pose2d",
                    cfg["model_name"],
                    "--pred-out-dir",
                    os.path.expandvars(cfg["pred_dir"]),
                ],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="multithreaded wrapper around mmpose inference script"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default="./cfg.json",
        help="path to config json file",
    )
    args = parser.parse_args()
    main(Path(args.config))
