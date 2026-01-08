#!/bin/bash

# Train with bagging (Bootstrap Aggregating)
# This script trains multiple agents using bootstrap sampling and combines their predictions

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH
export PYTHONPATH=$(pwd)/NavGPT:$PYTHONPATH

name=VLNBERT-train-Bagging_NavGPT

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train.py \
    --cfg configs/navgpt/bagging_navgpt.yaml 
