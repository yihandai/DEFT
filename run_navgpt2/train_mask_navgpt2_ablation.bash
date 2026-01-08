#!/bin/bash
name=VLNBERT-train-mask-navgpt2-ablation

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train_mask.py \
      --cfg configs/navgpt2/train_mask_navgpt2_ablation.yaml \
      --target_cfg configs/mapgpt.yaml
