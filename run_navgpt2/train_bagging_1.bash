#!/bin/bash

# Train with bagging (Bootstrap Aggregating)
# This script trains multiple agents using bootstrap sampling and combines their predictions

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH
export PYTHONPATH=$(pwd)/NavGPT_2:$PYTHONPATH

name=VLNBERT-train-Bagging_NavGPT2

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train.py \
    --cfg configs/navgpt2/bagging_navgpt2_1.yaml 
