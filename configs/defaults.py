import argparse
import os
import torch
import yaml


class Param:
    def __init__(self, yaml_config=None):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument(
            "--test_only", type=int, default=0, help="fast mode for testing"
        )

        self.parser.add_argument(
            "--iters", type=int, default=300000, help="training iterations"
        )
        self.parser.add_argument(
            "--name", type=str, default="default", help="experiment id"
        )
        self.parser.add_argument(
            "--vlnbert", type=str, default="oscar", help="oscar or prevalent"
        )
        self.parser.add_argument("--train", type=str, default="listener")
        self.parser.add_argument("--description", type=str, default="no description\n")

        # Bagging parameters
        self.parser.add_argument(
            "--bagging_agents",
            type=int,
            default=None,
            help="number of agents to train in bagging",
        )
        self.parser.add_argument(
            "--bootstrap_ratio",
            type=float,
            default=0.8,
            help="ratio of data to sample for each bootstrap",
        )
        # self.parser.add_argument(
        #     "--parallel_bagging",
        #     # action="store_true",
        #     type=bool,
        #     default=False,
        #     help="enable parallel training for bagging agents (default: False)",
        # )
        # self.parser.add_argument(
        #     "--max_workers",
        #     type=int,
        #     default=None,
        #     help="maximum number of parallel workers for bagging (default: None = auto-detect based on CPU/GPU count)",
        # )

        # Data preparation
        self.parser.add_argument(
            "--maxInput", type=int, default=80, help="max input instruction"
        )
        self.parser.add_argument(
            "--maxAction", type=int, default=15, help="Max Action sequence"
        )
        self.parser.add_argument("--batchSize", type=int, default=8)
        self.parser.add_argument("--ignoreid", type=int, default=-100)
        self.parser.add_argument("--feature_size", type=int, default=2048)
        self.parser.add_argument(
            "--loadOptim", action="store_const", default=False, const=True
        )

        # Load the model from
        self.parser.add_argument(
            "--load", default=None, help="path of the trained model"
        )
        # load mask wt
        self.parser.add_argument(
            "--loadmask", default=None, help="path of the trained model"
        )

        # Augmented Paths from
        self.parser.add_argument("--aug", default=None)

        # Listener Model Config
        self.parser.add_argument(
            "--zeroInit",
            dest="zero_init",
            action="store_const",
            default=False,
            const=True,
        )
        self.parser.add_argument(
            "--mlWeight", dest="ml_weight", type=float, default=0.20
        )
        self.parser.add_argument(
            "--teacherWeight", dest="teacher_weight", type=float, default=1.0
        )
        self.parser.add_argument("--features", type=str, default="places365")

        # Dropout Param
        self.parser.add_argument("--dropout", type=float, default=0.5)
        self.parser.add_argument("--featdropout", type=float, default=0.3)

        # Submision configuration
        self.parser.add_argument("--submit", type=int, default=0)

        # Training Configurations
        self.parser.add_argument("--optim", type=str, default="rms")  # rms, adam
        self.parser.add_argument(
            "--lr", type=float, default=0.00001, help="the learning rate"
        )
        self.parser.add_argument(
            "--decay", dest="weight_decay", type=float, default=0.0
        )
        self.parser.add_argument(
            "--feedback",
            type=str,
            default="sample",
            help="How to choose next position, one of ``teacher``, ``sample`` and ``argmax``",
        )
        self.parser.add_argument(
            "--teacher",
            type=str,
            default="final",
            help="How to get supervision. one of ``next`` and ``final`` ",
        )
        self.parser.add_argument("--epsilon", type=float, default=0.1)

        # Model hyper params:
        self.parser.add_argument(
            "--angleFeatSize", dest="angle_feat_size", type=int, default=4
        )

        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument(
            "--normalize",
            dest="normalize_loss",
            default="total",
            type=str,
            help="batch or total",
        )

        # GAIL
        self.parser.add_argument("--clip_param", default=0.2, type=float)
        self.parser.add_argument(
            "--actor_critic_update_num",
            type=int,
            default=10,
            help="update number of actor-critic (default: 10)",
        )
        self.parser.add_argument(
            "--discrim_update_num",
            type=int,
            default=2,
            help="update number of discriminator (default: 2)",
        )
        self.parser.add_argument("--gail_iteration", default=40000, type=float)
        self.parser.add_argument("--GAIL", action="store_true", default=False)
        self.parser.add_argument("--gail_sample_num", default=10, type=int)
        self.parser.add_argument("--update_interval", default=2, type=int)
        self.parser.add_argument(
            "--suspend_accu_exp",
            type=float,
            default=0.8,
            help="accuracy for suspending discriminator about expert data (default: 0.8)",
        )
        self.parser.add_argument(
            "--suspend_accu_gen",
            type=float,
            default=0.8,
            help="accuracy for suspending discriminator about generated data (default: 0.8)",
        )
        self.parser.add_argument(
            "--lamda",
            type=float,
            default=0.98,
            help="GAE hyper-parameter (default: 0.98)",
        )

        # train surrogate model
        self.parser.add_argument("--training_set_custom", type=str, default=None)
        self.parser.add_argument("--val_set_custom", type=str, default=None)
        self.parser.add_argument("--expert_policy", type=str, default="spl")

        # target agent
        self.parser.add_argument("--target_agent", type=str, default="MapGPT")
        self.parser.add_argument("--val_set_custom_surr", type=str, default=None)

        # ablation pretrained model
        self.parser.add_argument("--ablation_pretrained", type=bool, default=False)

        # feature level baseline test
        self.parser.add_argument("--feature_level_baseline", type=str, default=None)

        # Panoramic view configuration
        self.parser.add_argument(
            "--panoramic_horizontal_views",
            type=int,
            default=12,
            help="Number of horizontal views in panoramic image (default: 12 for 3x12, use 8 for 3x8)",
        )
        self.parser.add_argument(
            "--vfov",
            type=float,
            default=60,
            help="Vertical field of view in degrees (default: 60, use 45 for NavGPT surrogate model)",
        )

        self.parser.add_argument(
            "--update_inference",
            type=str,
            default="update",
            help="update or inference",
        )

        # for bagging training, start agent id
        self.parser.add_argument("--start_agent_id", type=int, default=0)

        # for test timestep-level agent
        self.parser.add_argument(
            "--timelevelbaseline",
            type=str,
            default="ours",
            help='["ours","gradient","ablation","value-based","random"]',
        )

        # self.args = self.parser.parse_args()
        assert yaml_config is not None, "yaml_config can't be None"
        self.load_args_from_yaml(yaml_config)

        if self.args.optim == "rms":
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == "adam":
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == "adamW":
            print("Optimizer: Using AdamW")
            self.args.optimizer = torch.optim.AdamW
        elif self.args.optim == "sgd":
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False

    # def load_args_from_yaml(self, filename):
    #     with open(filename, "r") as f:
    #         loaded_args = yaml.safe_load(f)
    #     self.args = argparse.Namespace(**loaded_args)
    # def load_args_from_yaml(self, filename):
    #     # 1. Start with default values from argparse
    #     self.args = self.parser.parse_args([])

    #     # 2. Load YAML overrides
    #     with open(filename, "r") as f:
    #         loaded_args = yaml.safe_load(f)

    #     # 3. Override defaults with values from YAML
    #     for key, value in loaded_args.items():
    #         if hasattr(self.args, key):
    #             setattr(self.args, key, value)
    #         else:
    #             raise ValueError(f"Unknown argument '{key}' in YAML config")
    def load_args_from_yaml(self, filename):
        # 1. Start with default values from argparse
        self.args = self.parser.parse_args([])

        # 2. Load YAML overrides
        with open(filename, "r") as f:
            loaded_args = yaml.safe_load(f)

        # 3. Override defaults with proper type conversion
        for key, value in loaded_args.items():
            if hasattr(self.args, key):
                action = next((a for a in self.parser._actions if a.dest == key), None)
                if action is None:
                    raise ValueError(f"Unknown argument '{key}' in YAML config")

                # Apply the action's type conversion if defined
                if action.type:
                    value = action.type(value)

                setattr(self.args, key, value)
            else:
                raise ValueError(f"Unknown argument '{key}' in YAML config")
