#!/bin/bash
name=VLNBERT-train-mask-navgpt-025-2-24vp

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH
export PYTHONPATH=$(pwd)/NavGPT:$PYTHONPATH

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train_mask.py \
      --cfg configs/navgpt/train_mask_navgpt_2.yaml \
      --target_cfg configs/navgpt/navgpt.yaml
