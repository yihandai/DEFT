import argparse
import os
import yaml


def parse_args(file=None):
    parser = argparse.ArgumentParser(description="")

    # datasets
    parser.add_argument("--root_dir", type=str, default="../datasets")
    parser.add_argument("--dataset", type=str, default="r2r", choices=["r2r", "r4r"])
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../datasets/R2R/exprs/gpt-3.5-turbo",
        help="experiment id",
    )
    # parser.add_argument('--output_dir', type=str, default='../datasets/R2R/exprs/LlaMA-2-13b-test', help='experiment id')
    parser.add_argument("--seed", type=int, default=0)

    # Agent
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="temperature for llm"
    )
    parser.add_argument(
        "--llm_model_name", type=str, default="gpt-3.5-turbo", help="llm model name"
    )
    # parser.add_argument('--llm_model_name', type=str, default='gpt-4', help='llm model name')
    # parser.add_argument('--llm_model_name', type=str, default='LlaMA-2-13b', help='llm model name')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=10)

    # General config
    # parser.add_argument(
    #     "--iters", type=int, default=10, help="number of iterations to run"
    # )
    parser.add_argument(
        "--iters", type=int, default=None, help="number of iterations to run"
    )
    parser.add_argument(
        "--max_scratchpad_length",
        type=int,
        default=1000,
        help="max number of steps in an episode",
    )
    parser.add_argument("--test", action="store_true", default=False)
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_0')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_1')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_2')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_3')
    # parser.add_argument('--val_env_name', type=str, default='R2R_val_unseen_instr_4')
    parser.add_argument("--val_env_name", type=str, default="R2R_val_unseen_instr")

    parser.add_argument("--load_instruction", action="store_true", default=True)
    parser.add_argument("--load_action_plan", action="store_true", default=True)

    parser.add_argument("--use_relative_angle", action="store_true", default=True)
    parser.add_argument("--use_history_chain", action="store_true", default=False)
    parser.add_argument("--use_tool_chain", action="store_true", default=False)
    parser.add_argument("--use_navigable", action="store_true", default=False)
    parser.add_argument("--use_single_action", action="store_true", default=True)

    parser.add_argument("--detailed_output", action="store_true", default=True)

    # parser.add_argument('--valid_file', type=str, default='../datasets/R2R/exprs/4-R2R_val_unseen_instr/4-R2R_val_unseen_instr.json', help='valid file name')
    parser.add_argument("--valid_file", type=str, default=None, help="valid file name")

    def load_args_from_yaml(parser, filename):
        # 1. Start with default values from argparse
        args = parser.parse_args([])

        # 2. Load YAML overrides
        with open(filename, "r") as f:
            loaded_args = yaml.safe_load(f)

        # 3. Override defaults with values from YAML
        for key, value in loaded_args.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                raise ValueError(f"Unknown argument '{key}' in YAML config")
        return args

    def load_args_from_yaml(parser, filename):
        # 1. Start with default values from argparse
        args = parser.parse_args([])

        # 2. Load YAML overrides
        with open(filename, "r") as f:
            loaded_args = yaml.safe_load(f)

        # 3. Override defaults with proper type conversion
        for key, value in loaded_args.items():
            if hasattr(args, key):
                action = next((a for a in parser._actions if a.dest == key), None)
                if action is None:
                    raise ValueError(f"Unknown argument '{key}' in YAML config")

                # Apply the action's type conversion if defined
                if action.type:
                    value = action.type(value)

                setattr(args, key, value)
            else:
                raise ValueError(f"Unknown argument '{key}' in YAML config")
        return args

    if file is None:
        args, _ = parser.parse_known_args()
    else:
        args = load_args_from_yaml(parser, file)

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    # Setup input paths
    args.obs_dir = os.path.join(ROOTDIR, "R2R", "observations_list_summarized")
    args.obs_summary_dir = os.path.join(ROOTDIR, "R2R", "observations_summarized")
    args.obj_dir = os.path.join(ROOTDIR, "R2R", "objects_list")
    # NOTE: add for perturbation
    args.obs_list_dir = os.path.join(ROOTDIR, "R2R", "observations_list")

    args.connectivity_dir = os.path.join(ROOTDIR, "R2R", "connectivity")
    args.scan_data_dir = os.path.join(ROOTDIR, "Matterport3D", "v1_unzip_scans")

    args.anno_dir = os.path.join(ROOTDIR, "R2R", "annotations")
    args.navigable_dir = os.path.join(ROOTDIR, "R2R", "navigable")

    # Build paths
    args.log_dir = os.path.join(args.output_dir, "logs")
    args.pred_dir = os.path.join(args.output_dir, "preds")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    return args
