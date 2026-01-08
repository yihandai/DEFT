#!/bin/bash
name=VLNBERT-test-baseline-navgpt2-ig

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH
export PYTHONPATH=$(pwd)/NavGPT:$PYTHONPATH
export PYTHONPATH=$(pwd)/NavGPT_2:$PYTHONPATH

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train_mask.py \
      --cfg configs/navgpt2/test_mask_navgpt2_baseline_ig_phase_heatmap.yaml \
      --target_cfg configs/navgpt2/navgpt2.yaml
