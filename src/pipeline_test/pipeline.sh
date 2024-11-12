#!/bin/bash

set -e

# uncomment the lines below if you want to rerun cleanly

# rm -rf ./runtime.log
# rm -rf ./cache/
# rm -rf ./pred/
# rm -rf ./data/crop/
# rm -rf ./out/
# rm -rf ./quality_check/
# rm -rf ./timestamp_speaker_mappings_crop_mappings.json
# rm -rf ./timestamp_speaker_mappings_crop.json

echo "trimming intro from clips"
python3 ../00_preprocessing/remove_intro.py \
    --input_dir ./data/clip/ \
    --output_dir ./data/crop/clip \
    --intro_len 106 \
    --outro_len 250 \
    -c 0.9

echo "cropping raw footage to aspect ratio of clip"
python3 ../00_preprocessing/crop_video.py \
    --mapping_json ./timestamp_speaker_mappings.json \
    --output_dir ./data/crop/raw

echo "perform double pass timestamp search"
python3 ../A_extract_timestamps/main.py \
    --cfg ./timestamp_cfg.json

echo "generating result/entries json"
python3 ../A_extract_timestamps/create_results_json.py \
    --cfg ./timestamp_cfg.json \
    ./pred

echo "creating side-by-side and diff for quality control"
# we set k to 999 since we want to quality control for every clip
python3 ../A_extract_timestamps/quality_control.py \
    --cfg ./timestamp_cfg.json \
    -k 999

echo "trimming original according to matched clips"
python3 ../B_ffmpeg_trim/ffmpeg_trim.py \
    --entries ./pred/entries.json \
    --cropped_input_mapping ./timestamp_speaker_mappings_crop_mappings.json \
    --output ./out/sslc \

echo "matching trimmed originals with ELAN annotations"
python3 ../C_align_annotations/parse.py \
    --eaf_dir data/eaf/ \
    --clip_dir out/sslc/ \
    --entries pred/entries.json \
    --output_dir out/eaf/

echo "performing keypoint inferences of trimmed originals"
python3 ../D_mmpose_inferences/parallell_inferences.py \
    --config ./pose_cfg.json
