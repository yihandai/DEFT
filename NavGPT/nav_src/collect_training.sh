#!/bin/bash
python NavGPT.py --llm_model_name gpt-4o \
    --output_dir ../datasets/R2R/exprs/training_data \
    --val_env_name r2r_subset_instr_level_10percent \
