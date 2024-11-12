# SSLC Pose Dataset Pipeline

Original Frame | Overlaid Keypoints
:-------------:|:------------------:
![](orig.jpg)  | ![](keyp.jpg)

Contains code for adding and packaging the **S**wedish **S**ign **L**anguage
**C**orpus along with pose estimation information.

An overview of the current pipeline:

### `00_preprocessing`

Contains files to aid in timestamp extraction later, depending on the input
data. Specifically, there are two files with specific purposes:

1. `remove_intro.py`: trims the intro/outro of a set of video files.

2. `crop_video.py`: crops raw video files to the most common aspect ratio for a
   group of clips. This is necessary since the timestamp search is very
   affected by large differences in aspect ratio.

These may, or may not, apply - depending on the data.

### `A_extract_timestamps`

Search for most likely attribution, and alignment, for a set of clips given
possible [raw footage] candidates. In practice, a JSON file is defined
beforehand with candidates for a group of clips. A double-pass sliding-window
calculation is then performed for each respective combination in that JSON
file.

### `B_ffmpeg_trim`

Trims raw footage to the alignment data given by the timestamp extraction
searches.

### `C_match_elan_annotations`

Extracts ELAN annotations for SSLC files, and aligns them to potential offsets,
and exports it to a JSON file to allow easier programmatic access to those
annotations when combining the pose estimation information with the
annotations.

### `D_mmpose_inferences`

Given a folder of video-files perform MMPOSE inferences and export those to
JSON format in some export directory.

These MMPOSE inferences can be performed more efficiently in paralell if that
is desired and if the host capability is there. For some reason MMPOSE does not
support parallelization during inference in their provided script, so each
inference has to be performed separately. During testing each inference took
about ~600MB of RAM and about 50% of processing capability for a Nvidia Titan
X. Adjust number of workers as needed.

# Installation

An example pipeline is provided in `./pipeline_test/pipeline.sh`.

Prerequisites, using `venv`:

```
python3 -m venv ./venv
source venv/bin/activate

pip3 install -r requirements.py
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmpose>=1.1.0"
```

Then adjust the configs according to the data. 

The first important config file is `timestamp_cfg.json`. The structure of that
file looks like this:

```
{
    "workers": NUMBER_OF_PARALLEL_WORKERS,
    "n_comparison":  NUMBER_OF_FRAMES_ABOVE_MINIMUM_DISTANCE_TO_COMPARE_TO,
    "match_decision_boundary": BOUNDARY_OF_MATCH,
    "early_stopping_threshold": VALUE_TO_EARLY_STOP_AT,
    "cache_dir": "path/to/cache/numpy/arrays",
    "pred_dir": "path/to/store/predictions/and/calculations",
    "speaker_video_mappings": "path/to/speaker/mappings/json"
}
```

The last option detailing the path to the mappings file, which is, arguably,
the most important. In the sample pipeline it is located at
`timestamp_speaker_mappings.json`. The structure of that file is as follows:

```
{
    ID: {
        "raw_dir": "path/to/raw/footage/dir",
        "subset_video_files": [
            "path/to/potential/candidate",
            "path/to/potential/candidate",
            ...
            ]
        "raw_files": [
            "path/to/raw/footage/candidate",
            "path/to/raw/footage/candidate",
            ...
        ]
    },
}
```

For the pose estimation there is the example `pose_cfg.json`. It has the
following structure:

```
{
    "workers": NUM_WORKERS,
    "script_path": "path/to/mmpose/inferencer_demo.py",
    "model_name": MODEL_NAME,
    "input_dir": "path/to/input/files",
    "pred_dir": "path/to/output/directory",
}
```

Where `inferencer_demo.py` can be found here:
[https://github.com/open-mmlab/mmpose](https://github.com/open-mmlab/mmpose)

I recommend going through `pipeline.sh` to see how each respective component is
called, and instead of running them all in one go, run them one by one ---
checking the results as you go. Especially the timestamp extraction is tricky,
and will require some manual intervention.
