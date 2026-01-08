#!/bin/bash
python NavGPT.py --llm_model_name gpt-4o \
    --output_dir ../datasets/R2R/exprs/val72_navgpt_2 \
    --val_env_name r2r_subset_instr_level_val72_navgpt \
    --valid_file ../datasets/R2R/exprs/val72_navgpt_2/logs/runtime_val72_navgpt_recollect.json \
