import concurrent.futures
import subprocess
import json
import logging
import os
import sys
from pathlib import Path

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
        logging.error(err.stderr.decode("utf-8").strip())


def main() -> None:
    with open("cfg.json", "r") as f:
        cfg: dict = json.load(f)
    pred_dir: Path = Path(cfg["pred_dir"])
    if not pred_dir.exists():
        pred_dir.mkdir()
    if not pred_dir.is_dir():
        print("FATAL: pred_dir exists and is not a directory. exiting...")
        sys.exit(1)
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
    main()
