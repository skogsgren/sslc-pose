# SSLC Pose Dataset Pipeline

Original Frame | Overlaid Keypoints
:-------------:|:------------------:
![](orig.jpg)  | ![](keyp.jpg)

Contains code for adding and packaging the **S**wedish **S**ign **L**anguage
**C**orpus along with pose estimation information.

Current pipeline is as follows:

### `A_extract_timestamps`

Given a folder of raw files, find out the timestamps of a folder of potential
subset video files (i.e. clips). The location of the clips are provided through
a JSON file that each RAW file must have, however, this is easily changed
because of the modularity in the function if the folder structure would require
it. One reason why I thought it necessary is because I thought that it could be
possible to have different clip intro/outro lengths (i.e. the number of frames
for the preamble with GDPR information, as well as the number of frames when it
fades to black) depending on the raw video file, however if this is not needed
one could just as well have a global JSON file that declares all of these
values for every raw file.

Given a folder of raw video files and a folder of potential subset video files,
find out which clips belong to which raw video files, and where. This algorithm
uses `NJIT` alongside `numpy` to perform efficient frame-by-frame calculations
in order to achieve that (see folder `README` for more information).

### `B_ffmpeg_trim`

Given a raw video file, timestamps [and clip filename], trims the raw video
file according to timestamps and exports that to a provided path.

### `C_match_elan_annotations`

Extracts ELAN annotations for SSLC files and exports it to a JSON file to allow
easier programmatic access to those annotations when combining the pose
estimation information with the annotations.

### `D_mmpose_inferences`

Given a folder of video-files perform MMPOSE inferences and export those to
JSON format in some export directory. A few things have to be specified in a
JSON file (cfg.json in this directory):

1. SCRIPT_PATH: the full path to where the 'inferencer_demo.py' (i.e. the
   CLI for running MMPOSE inferences) is located. For example, if you have
   MMPOSE cloned to your home directory, then "$HOME/demo/inferencer_demo.py"
   should do the trick.
2. MODEL_SPEC: MMPOSE has several models so the specific model desired must be
   specified.
3. PRED_DIR: where should the JSON files that are exported from MMPOSE be
   stored?

These MMPOSE inferences can be performed more efficiently in paralell if that
is desired and if the host capability is there. For some reason MMPOSE does not
support parallelization during inference in their provided script, so each
inference has to be performed separately. During testing each inference took
about ~600MB of RAM and about 50% of processing capability for a Nvidia Titan
X. Adjust number of workers as needed.

### `E_compose_dataset`

Put every thing together to form a cohesive dataset that can then be used
directly for research purposes.
