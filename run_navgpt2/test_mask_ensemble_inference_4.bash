#!/bin/bash
name=VLNBERT-test-navgpt2-ensemble

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH
export PYTHONPATH=$(pwd)/NavGPT_2:$PYTHONPATH

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train_mask.py \
      --cfg configs/navgpt2/test_feature_navgpt2_ensemble_phase_inference_4.yaml \
      --target_cfg configs/navgpt2/navgpt2.yaml
