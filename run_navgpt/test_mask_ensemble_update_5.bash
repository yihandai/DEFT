#!/bin/bash
name=VLNBERT-test-navgpt-ensemble

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH
export PYTHONPATH=$(pwd)/NavGPT:$PYTHONPATH

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train_mask.py \
      --cfg configs/navgpt/test_feature_navgpt_ensemble_phase_update_5.yaml \
      --target_cfg configs/navgpt/navgpt.yaml
