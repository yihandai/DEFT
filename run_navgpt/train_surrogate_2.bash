#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH

name=VLNBERT-train-Surrogate_NavGPT_val_24vp

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train.py \
    --cfg configs/navgpt/surrogate_navgpt.yaml 
