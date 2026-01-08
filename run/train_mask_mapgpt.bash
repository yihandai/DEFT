#!/bin/bash
name=VLNBERT-train-mask-mapgpt

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train_mask.py \
      --cfg configs/train_mask_mapgpt.yaml \
      --target_cfg configs/mapgpt.yaml
