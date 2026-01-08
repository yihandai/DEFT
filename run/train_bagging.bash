#!/bin/bash

# Train with bagging (Bootstrap Aggregating)
# This script trains multiple agents using bootstrap sampling and combines their predictions

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH

name=VLNBERT-train-Bagging

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train.py \
    --cfg configs/bagging.yaml 
