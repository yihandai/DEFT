#!/bin/bash
python NavGPT.py --llm_model_name gpt-4o \
    --output_dir ../datasets/R2R/exprs/gpt-4-val-unseen \
    --val_env_name R2R_val_unseen_instr \
    --batch_size 4 \
    --iters 2
