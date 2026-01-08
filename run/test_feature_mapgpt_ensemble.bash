#!/bin/bash
name=VLNBERT-train-feature-mapgpt-ensemble

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train_mask.py \
      --cfg configs/test_feature_mapgpt_ensemble.yaml \
      --target_cfg configs/mapgpt.yaml
