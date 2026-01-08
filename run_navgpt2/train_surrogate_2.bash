#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH
export PYTHONPATH=$(pwd)/NavGPT_2:$PYTHONPATH

name=VLNBERT-train-Surrogate_NavGPT2

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train.py \
    --cfg configs/navgpt2/surrogate_navgpt2.yaml 
