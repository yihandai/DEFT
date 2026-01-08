import os
import json
import time
from collections import defaultdict

from vln.data_utils import construct_instrs
from vln.env import R2RNavBatch
from vln.parser import parse_args

from utils.data import set_random_seed
from utils.logger import write_to_record_file

from vln.gpt_agent import GPTNavAgent


def build_dataset(args, rank=0, is_test=True):
    dataset_class = R2RNavBatch
    split = args.split
    val_envs = {}

    if "processed" in split:
        with open(os.path.join(args.anno_dir, split + ".json"), "r") as f:
            val_instr_data = json.load(f)

        if args.end is None:
            args.end = len(val_instr_data)
        val_instr_data = val_instr_data[args.start : args.end]
        print(
            f"------------------ Evaluate {args.start}-{args.end} in {split} ------------------"
        )

    else:
        val_instr_data = construct_instrs(
            args.anno_dir,
            args.dataset,
            [split],
            tokenizer=args.tokenizer,
            max_instr_len=args.max_instr_len,
            is_test=is_test,
        )

    val_env = dataset_class(
        val_instr_data,
        args.connectivity_dir,
        batch_size=args.batch_size,
        seed=args.seed + rank,
        name=split,
        args=args,
    )  # evaluation using all objects
    val_envs[split] = val_env

    return val_envs


def valid(args, val_envs, rank=0):
    pred = load_file(args.pred_file)
    for env_name, env in val_envs.items():
        score_summary, _ = env.eval_metrics(pred, args.dataset)
        loss_str = "Env name: %s" % env_name
        for metric, val in score_summary.items():
            loss_str += ", %s: %.2f" % (metric, val)
        print(loss_str)


def load_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def parse_json(json_data):
    preds = []
    for item in json_data:
        instr_id = item["instr_id"]
        traj = item["path"]
        preds.append({"instr_id": instr_id, "trajectory": traj})
    return preds


if __name__ == "__main__":
    args = parse_args()
    val_envs = build_dataset(args)
    valid(args, val_envs)
