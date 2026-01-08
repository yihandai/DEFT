DATA_ROOT=./datasets
outdir=${DATA_ROOT}/exprs_map/test/
# pred_file=${outdir}/preds.json
pred_file=vln-bert/data/R2R_val72.json
flag="--root_dir ${DATA_ROOT}
      --img_root Observation_RGB
      --split MapGPT_72_scenes_processed
      --start 100
      --end 216  # the number of cases to be tested
      --output_dir ${outdir}
      --max_action_len 15
      --save_pred
      --stop_after 3
      --llm gpt-4o-2024-05-13
      --response_format json
      --max_tokens 1000
      --pred_file ${pred_file}
      "

python3 check_metrics.py $flag
