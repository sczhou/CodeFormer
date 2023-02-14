#!/bin/bash

python basicsr/setup.py develop
conda install -c conda-forge dlib
python scripts/download_pretrained_models.py facelib
python scripts/download_pretrained_models.py dlib
python scripts/download_pretrained_models.py CodeFormer
