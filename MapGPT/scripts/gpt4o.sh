DATA_ROOT=./datasets
outdir=${DATA_ROOT}/exprs_map/test/

flag="--root_dir ${DATA_ROOT}
      --img_root Observation_RGB
      --split MapGPT_72_scenes_processed
      --start 0
      --output_dir ${outdir}
      --max_action_len 15
      --save_pred
      --stop_after 3
      --llm gpt-4o
      --response_format json
      --max_tokens 1000
      "

python3 main_gpt.py $flag
