#!/bin/bash
name=VLNBERT-test-baseline-navgpt-ig

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH
export PYTHONPATH=$(pwd)/NavGPT:$PYTHONPATH

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train_mask.py \
      --cfg configs/navgpt/test_mask_navgpt_baseline_ig_phase_inference.yaml \
      --target_cfg configs/navgpt/navgpt.yaml
