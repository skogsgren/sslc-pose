__doc__ = """
This file contains small util functions which are utilized in more than one
python file in more than one directory.
"""

from pathlib import Path
from datetime import datetime, timedelta
import json


def naming_function(path: Path) -> str:
    """given a path object returns a str that can be used as a file/dir name
    which can then later be used to extract folder information"""
    DIV = "+"
    parentparent = path.parent.parent.name
    parent = path.parent.name
    return parentparent + DIV + parent + DIV + path.stem


def timestamp_to_seconds(timestamp: str) -> float:
    """given a %H:%M:%S.%f timestamp, return total seconds of that timestamp"""
    try:
        dt = datetime.strptime(timestamp, "%H:%M:%S.%f")
    except ValueError:
        dt = datetime.strptime(timestamp, "%H:%M:%S")
    hr = dt.hour
    mn = dt.minute
    s = dt.second
    ms = dt.microsecond
    return hr * 3600 + mn * 60 + s + ms / 10e5


def seconds_to_timestamp(seconds: float) -> str:
    """given int of n seconds convert to %H:%M:%S.%f timestamp"""
    return str(timedelta(seconds=seconds))


def entries_to_mappings(entries_json: Path) -> dict:
    with open(entries_json, "r") as f:
        entries = json.load(f)
    mappings: dict = {}
    for _, matches in entries.items():
        for label in ["matched_clips", "unmatched_clips"]:
            for clip in matches[label]:
                mappings[clip[1]] = clip[2]
    return mappings
