from pathlib import Path

def parse_pred_name(f: Path) -> str:
    parentparent: str = f.parent.parent.name
    parent: str = f.parent.name
    if len(parentparent + parent + f.stem) < 255:
        suff: str = f"{parentparent}_{parent}_{f.stem}"
    elif len(parent + f.stem) < 255:
        suff: str = f"{parent}_{f.stem}"
    else:
        suff: str = f"{f.stem}"
    return suff
