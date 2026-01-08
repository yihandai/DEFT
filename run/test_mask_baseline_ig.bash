#!/bin/bash
name=VLNBERT-test-baseline-mapgpt-ig

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train_mask.py \
      --cfg configs/test_mask_mapgpt_baseline_ig.yaml \
      --target_cfg configs/mapgpt.yaml
