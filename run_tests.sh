#!/bin/bash

DIRECTORY="test_venv"
if [ ! -d "$DIRECTORY" ]; then
    python3 -m venv $DIRECTORY
fi
source "${DIRECTORY}/bin/activate"

pip3 install -r ./requirements.txt
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmpose>=1.1.0"

python3 -m pytest

deactivate
