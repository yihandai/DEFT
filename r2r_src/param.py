import argparse
import os
import torch
import yaml

import sys

from configs.defaults import Param


def load_config(path_to_config=None):
    param = Param(yaml_config=path_to_config)
    cfg = param.args

    cfg.description = cfg.name
    cfg.IMAGENET_FEATURES = "img_features/ResNet-152-imagenet.tsv"
    cfg.log_dir = "snap/%s" % cfg.name
    return cfg


# param = Param()
# args = param.args

# args.description = args.name
# args.IMAGENET_FEATURES = "img_features/ResNet-152-imagenet.tsv"
# args.log_dir = "snap/%s" % args.name

# parse args from command line
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default=None)
parser.add_argument("--target_cfg", type=str, default=None)
command_args = parser.parse_args()

cfg_path = command_args.cfg
target_cfg_path = command_args.target_cfg

# explainable model configs
args = load_config(path_to_config=cfg_path)

# target model configs
if args.target_agent == "MapGPT":
    from MapGPT.vln.parser import parse_args

    target_args = parse_args(target_cfg_path)
elif args.target_agent == "NavGPT":
    from NavGPT.nav_src.parser import parse_args

    target_args = parse_args(target_cfg_path)
elif args.target_agent == "NavGPT2":
    from NavGPT_2.map_nav_src.r2r.parser import parse_args

    target_args = parse_args(target_cfg_path)
else:
    raise ValueError(f"Unknown target agent: {args.target_agent}")

# set the batch size to the same as the source model
target_args.batch_size = args.batchSize

if args.feature_level_baseline == "hsic":
    target_args.load_patch_feature = False
if args.update_inference == "inference":
    target_args.load_patch_feature = False

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
DEBUG_FILE = open(os.path.join("snap", args.name, "debug.log"), "w")
