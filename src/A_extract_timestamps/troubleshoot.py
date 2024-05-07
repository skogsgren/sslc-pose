from extract import get_pred_suffix
from check_speaker_id import get_top_candidates
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
import json


def get_wrongly_assigned_matches(cfg_path: Path = Path("./cfg.json")):
    with open(cfg_path, "r") as f:
        cfg: dict = json.load(f)
    with open(Path(cfg["pred_dir"]) / "results.json", "r") as f:
        res: dict[str, list[str]] = json.load(f)

    def correctly_assigned(clip_path: str, raw_video_path: str) -> bool:
        pred_path: Path = Path(cfg["pred_dir"])
        clip_json_base: str = Path(clip_path).with_suffix(".json").name
        raw_suffix: str = get_pred_suffix(Path(raw_video_path))
        json_path: Path = pred_path / raw_suffix / clip_json_base
        if not json_path.exists():
            print(f"ERROR: {json_path} does not exist")
            return False
        with open(json_path, "r") as f:
            return True if json.load(f)["in_original"] == 1 else False

    wrongly_assigned_dict: dict[str, list[str]] = {}
    for raw_video_path, matches in tqdm(res.items()):
        wrongly_assigned = [
            x for x in matches if not correctly_assigned(x, raw_video_path)
        ]
        wrongly_assigned_dict[raw_video_path] = wrongly_assigned
    return wrongly_assigned_dict


if __name__ == "__main__":
    wrm: dict[str, list[str]] = get_wrongly_assigned_matches()
    pprint(wrm)

    for raw_video_file, matches in wrm.items():
        pprint({
            raw_video_file: [
                (x, get_top_candidates(Path(x))[0]) for x in matches
                if get_top_candidates(Path(x))[0][1] < 0.33
            ]
        })
