#!/bin/bash

gdown "1qXhuKemh8ekA_fK04TlKkYyPIhfD9-cn"
unzip Data.zip
wget "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -P ./ckpt
git lfs install
git clone https://huggingface.co/ShilongLiu/GroundingDINO ./ckpt/GroundingDINO

git-lfs clone https://huggingface.co/windwhinny/chilloutmix.git experiments/pretrained_models/chilloutmix