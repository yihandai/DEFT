#!/bin/bash
name=VLNBERT-train-mask-navgpt-ablation

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train_mask.py \
      --cfg configs/navgpt/train_mask_navgpt_ablation.yaml \
      --target_cfg configs/mapgpt.yaml
