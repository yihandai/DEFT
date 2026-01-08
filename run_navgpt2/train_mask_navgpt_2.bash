#!/bin/bash
name=VLNBERT-train-mask-navgpt2-1

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH
export PYTHONPATH=$(pwd)/NavGPT_2:$PYTHONPATH

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train_mask.py \
      --cfg configs/navgpt2/train_mask_navgpt_2.yaml \
      --target_cfg configs/navgpt2/navgpt2.yaml
