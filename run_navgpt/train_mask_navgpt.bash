#!/bin/bash
name=VLNBERT-train-mask-navgpt-025-24vp

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train_mask.py \
      --cfg configs/navgpt/train_mask_navgpt.yaml \
      --target_cfg configs/mapgpt.yaml
