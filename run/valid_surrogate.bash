#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/MapGPT:$PYTHONPATH

name=VLNBERT-valid-Surrogate

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train.py \
    --cfg configs/valid_surrogate.yaml 
