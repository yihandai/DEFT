import torch

import os
import time
import json
import random
import numpy as np
from collections import defaultdict

import argparse
from r2r_src.vln_utils import (
    read_vocab,
    write_vocab,
    build_vocab,
    padding_idx,
    timeSince,
    read_img_features,
    print_progress,
)
import r2r_src.vln_utils as vln_utils
from env import R2RBatch
from agent import Seq2SeqAgent
from agent_feature import FeatureAgent
from agent_feature_ensemble import FeatureAgentEnsemble
from eval import Evaluation
from param import args, target_args

from agent_mask import MaskAgent
import warnings

warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter

from vlnbert.vlnbert_init import get_tokenizer

# target agents modules
from MapGPT.vln.env import R2RNavBatch as R2RNavBatch_MapGPT

if args.target_agent == "MapGPT":
    from MapGPT.vln.data_utils import construct_instrs
elif args.target_agent == "NavGPT":
    from NavGPT.nav_src.env import R2RNavBatch as R2RNavBatch_NavGPT
    from agent_mask_navgpt import MaskAgent_NavGPT
    from NavGPT.nav_src.data_utils import construct_instrs
elif args.target_agent == "NavGPT2":
    from NavGPT_2.map_nav_src.r2r.data_utils import construct_instrs

    # target agents modules navgpt
    from agent_mask_navgpt2 import MaskAgent_NavGPT2

log_dir = "snap/%s" % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

IMAGENET_FEATURES = "img_features/ResNet-152-imagenet.tsv"
PLACE365_FEATURES = "img_features/ResNet-152-places365.tsv"
PLACE365_FEATURES_24VP = "img_features/ResNet-152-places365_24vp.tsv"

if args.features == "imagenet":
    features = IMAGENET_FEATURES
elif args.features == "places365":
    features = PLACE365_FEATURES
elif args.features == "places365_24vp":
    features = PLACE365_FEATURES_24VP

feedback_method = args.feedback  # teacher or sample

print(args)
print("")

# metric_1 = "nDTW"
# metric_2 = "CLS"
metric_1 = "success_rate"
metric_2 = "spl"

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"

""" train the listener """


def train(
    train_env,
    tok,
    n_iters,
    log_every=2000,
    val_envs={},
    evaluator_train=None,
    target_train_env=None,
    target_eval_env=None,
):
    writer = SummaryWriter(log_dir=log_dir)
    # listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    if args.target_agent == "MapGPT":
        listner = MaskAgent(train_env, "", tok, args.maxAction, args_target=target_args)
    elif args.target_agent == "NavGPT":
        listner = MaskAgent_NavGPT(
            train_env, "", tok, args.maxAction, args_target=target_args
        )
    elif args.target_agent == "NavGPT2":
        listner = MaskAgent_NavGPT2(
            train_env, "", tok, args.maxAction, args_target=target_args
        )
    else:
        assert False

    start_iter = 0
    if args.load is not None:
        if args.aug is None:
            listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}".format(args.load))
        else:
            listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}".format(args.load))
    print("\nListener training starts, start iteration: %s" % str(start_iter))

    if args.loadmask is not None:
        start_iter = listner.load_mask(os.path.join(args.loadmask))
        print(
            "Loaded the mask model at iter {} from {}".format(start_iter, args.loadmask)
        )

    for idx in range(start_iter, start_iter + n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters - idx)
        iter = idx + interval

        # Train for log_every interval
        listner.env = train_env
        # listner.target_agent.env = train_env
        listner.target_agent.env = target_train_env

        if args.train == "mask":
            listner.train(
                interval, feedback=feedback_method, mask=True
            )  # Train interval iters

        # Log the training stats to tensorboard
        total = max(sum(listner.logs["total"]), 1)
        length = max(len(listner.logs["critic_loss"]), 1)
        critic_loss = sum(listner.logs["critic_loss"]) / total
        RL_loss = sum(listner.logs["RL_loss"]) / max(len(listner.logs["RL_loss"]), 1)
        IL_loss = sum(listner.logs["IL_loss"]) / max(len(listner.logs["IL_loss"]), 1)
        entropy = sum(listner.logs["entropy"]) / total
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/RL_loss", RL_loss, idx)
        writer.add_scalar("loss/IL_loss", IL_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        # print("total_actions", total, ", max_length", length)

        # Run validation
        print("Iter {}   IL loss {}   RL loss {}".format(iter, IL_loss, RL_loss))


def valid(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print(
        "Loaded the listener model at iter %d from %s"
        % (agent.load(args.load), args.load)
    )

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback="argmax", iters=iters)
        result = agent.get_results()

        if env_name != "":
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric, val in score_summary.items():
                loss_str += ", %s: %.4f" % (metric, val)
            print(loss_str)

        if args.submit:
            json.dump(
                result,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), "w"),
                sort_keys=True,
                indent=4,
                separators=(",", ": "),
            )


def valid_mask(train_env, tok, val_envs={}, eval_env=None, target_eval_env=None):
    if args.target_agent == "MapGPT":
        agent = MaskAgent(train_env, "", tok, args.maxAction, args_target=target_args)
    elif args.target_agent == "NavGPT":
        agent = MaskAgent_NavGPT(
            train_env, "", tok, args.maxAction, args_target=target_args
        )
    elif args.target_agent == "NavGPT2":
        agent = MaskAgent_NavGPT2(
            train_env, "", tok, args.maxAction, args_target=target_args
        )

    print(
        "Loaded the listener model at iter %d from %s"
        % (agent.load(args.load), args.load)
    )
    print(
        "Loaded the mask model at iter %d from %s"
        % (agent.load_mask(args.loadmask), args.loadmask)
    )
    if target_eval_env is not None:
        agent.logs = defaultdict(list)
        agent.env = eval_env
        agent.target_agent.env = target_eval_env  # environment for target agent

        iters = None
        fidelity_score = agent.test_mask(iters=iters)
        print("fidelity_score: {}".format(fidelity_score))
    else:
        for env_name, (env, evaluator) in val_envs.items():
            agent.logs = defaultdict(list)
            agent.env = env

            iters = None
            fidelity_score = agent.test_mask(iters=iters)
            print("fidelity_score: {}".format(fidelity_score))


def valid_feature(train_env, tok, val_envs={}, eval_env=None, target_eval_env=None):
    if args.train == "feature":
        if args.target_agent == "MapGPT":
            agent = FeatureAgent(
                train_env,
                "",
                tok,
                args.maxAction,
                args_target=target_args,
                # , "", tok, args.maxAction, args_target=target_args
            )
        elif args.target_agent == "NavGPT":
            from agent_feature_navgpt import FeatureAgent_NavGPT

            agent = FeatureAgent_NavGPT(
                train_env,
                "",
                tok,
                args.maxAction,
                args_target=target_args,
            )
        elif args.target_agent == "NavGPT2":
            from agent_feature_navgpt2 import FeatureAgent_NavGPT2

            agent = FeatureAgent_NavGPT2(
                train_env,
                "",
                tok,
                args.maxAction,
                args_target=target_args,
            )
        print(
            "Loaded the listener model at iter %d from %s"
            % (agent.load(args.load), args.load)
        )
        print(
            "Loaded the mask model at iter %d from %s"
            % (agent.load_mask(args.loadmask), args.loadmask)
        )

    elif args.train == "feature_bagging":
        if args.target_agent == "MapGPT":
            agent = FeatureAgentEnsemble(
                train_env, "", tok, args.maxAction, args_target=target_args
            )
        elif args.target_agent == "NavGPT":
            # from agent_feature_navgpt import FeatureAgent_NavGPT
            from agent_feature_ensemble_navgpt import FeatureAgentEnsemble_NavGPT

            agent = FeatureAgentEnsemble_NavGPT(
                train_env,
                "",
                tok,
                args.maxAction,
                args_target=target_args,
            )
            # elif args.target_agent == "NavGPT2":
            #     agent = FeatureAgent_NavGPT2(
            #         train_env, "", tok, args.maxAction, args_target=target_args
            #     )
        elif args.target_agent == "NavGPT2":
            from agent_feature_ensemble_navgpt2 import FeatureAgentEnsemble_NavGPT2

            agent = FeatureAgentEnsemble_NavGPT2(
                train_env,
                "",
                tok,
                args.maxAction,
                args_target=target_args,
            )
    if target_eval_env is not None:
        agent.logs = defaultdict(list)
        agent.env = eval_env
        agent.target_agent.env = target_eval_env  # environment for target agent

        iters = None
        agent.test(iters=iters)


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(0)
    np.random.seed(0)


def train_val(
    test_only=False,
    train_env_custom=None,
    val_env_custom=None,
    val_env_custom_surr=None,
):
    """Train on the training set, and validate on seen and unseen splits."""
    setup()
    tok = get_tokenizer(args)

    rank = 0

    feat_dict = read_img_features(features, test_only=test_only)
    if val_env_custom is not None:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = [val_env_custom]
    elif test_only:
        featurized_scans = None
        val_env_names = ["val_train_seen"]
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = ["val_train_seen", "val_seen", "val_unseen"]
    if train_env_custom is not None:
        train_env = R2RBatch(
            feat_dict,
            batch_size=args.batchSize,
            splits=[train_env_custom],
            tokenizer=tok,
        )
        if args.target_agent == "MapGPT":
            # 对于 train_env_custom，instructions已经被分开了
            with open(
                os.path.join(target_args.anno_dir, train_env_custom + "./json"), "r"
            ) as f:
                train_instr_data = json.load(f)
            if target_args.end is None:
                target_args.end = len(train_instr_data)
            train_instr_data = train_instr_data[target_args.start : target_args.end]
            target_train_env = R2RNavBatch_MapGPT(
                train_instr_data,
                target_args.connectivity_dir,
                batch_size=target_args.batch_size,
                seed=target_args.seed + rank,
                name=train_env_custom,
                args=target_args,
            )  # evaluation using all objects
        elif args.target_agent == "NavGPT":
            from NavGPT.nav_src.utils.data import ImageObservationsDB

            feat_db = ImageObservationsDB(
                target_args.obs_dir,
                target_args.obs_summary_dir,
                target_args.obj_dir,
                target_args.obs_list_dir,
            )

            train_instr_data = construct_instrs(
                target_args.anno_dir, target_args.dataset, [train_env_custom]
            )
            target_train_env = R2RNavBatch_NavGPT(
                feat_db,
                train_instr_data,
                target_args.connectivity_dir,
                target_args.navigable_dir,
                batch_size=target_args.batch_size,
                seed=target_args.seed,
                name=train_env_custom,
            )  # evaluation using all objects
        elif args.target_agent == "NavGPT2":
            from NavGPT_2.map_nav_src.utils.data import ImageFeaturesDB
            from NavGPT_2.map_nav_src.r2r.env import R2RNavBatch

            # from NavGPT_2.map_nav_src.r2r.data_utils import construct_instrs

            feat_db = ImageFeaturesDB(
                target_args.img_ft_file, target_args.image_feat_size
            )
            train_instr_data = construct_instrs(
                target_args.anno_dir,
                target_args.dataset,
                [train_env_custom],
                tokenizer=target_args.tokenizer,
                max_instr_len=target_args.max_instr_len,
                is_test=False,
            )
            target_train_env = R2RNavBatch(
                feat_db,
                train_instr_data,
                target_args.connectivity_dir,
                target_args.candidate_file_dir,
                batch_size=target_args.batch_size,
                seed=target_args.seed + rank,
                sel_data_idxs=None,
                name=train_env_custom,
            )
    else:
        train_env = R2RBatch(
            feat_dict, batch_size=args.batchSize, splits=["train"], tokenizer=tok
        )
        if args.target_agent == "MapGPT":
            train_instr_data = construct_instrs(
                target_args.anno_dir,
                target_args.dataset,
                ["train"],
                tokenizer=target_args.tokenizer,
                max_instr_len=target_args.max_instr_len,
                is_test=False,
            )

            target_train_env = R2RNavBatch_MapGPT(
                train_instr_data,
                target_args.connectivity_dir,
                batch_size=target_args.batch_size,
                seed=target_args.seed + rank,
                name="train",
                args=target_args,
            )  # evaluation using all objects
        elif args.target_agent == "NavGPT":
            from NavGPT.nav_src.utils.data import ImageObservationsDB

            # from NavGPT.nav_src.data_utils import construct_instrs

            feat_db = ImageObservationsDB(
                target_args.obs_dir,
                target_args.obs_summary_dir,
                target_args.obj_dir,
                target_args.obs_list_dir,
            )

            train_instr_data = construct_instrs(
                target_args.anno_dir,
                target_args.dataset,
                ["train"],
            )
            target_train_env = R2RNavBatch_NavGPT(
                feat_db,
                train_instr_data,
                target_args.connectivity_dir,
                target_args.navigable_dir,
                batch_size=target_args.batch_size,
                seed=target_args.seed,
                name="train",
            )
        elif args.target_agent == "NavGPT2":
            from NavGPT_2.map_nav_src.utils.data import ImageFeaturesDB
            from NavGPT_2.map_nav_src.r2r.env import R2RNavBatch

            # from NavGPT_2.map_nav_src.r2r.data_utils import construct_instrs

            feat_db = ImageFeaturesDB(
                target_args.img_ft_file, target_args.image_feat_size
            )
            train_instr_data = construct_instrs(
                target_args.anno_dir,
                target_args.dataset,
                ["train"],
                tokenizer=target_args.tokenizer,
                max_instr_len=target_args.max_instr_len,
                is_test=False,
            )
            target_train_env = R2RNavBatch(
                feat_db,
                train_instr_data,
                target_args.connectivity_dir,
                target_args.candidate_file_dir,
                batch_size=target_args.batch_size,
                seed=target_args.seed + rank,
                sel_data_idxs=None,
                name="train",
            )

    from collections import OrderedDict

    if args.submit:
        val_env_names.append("test")
    else:
        pass

    val_envs = OrderedDict(
        (
            (
                split,
                (
                    R2RBatch(
                        feat_dict,
                        batch_size=args.batchSize,
                        splits=[split],
                        tokenizer=tok,
                    ),
                    Evaluation([split], featurized_scans, tok),
                ),
            )
            for split in val_env_names
        )
    )
    if args.target_agent == "MapGPT":
        # val72 中的instr就是已经被分开的
        with open(
            os.path.join(target_args.anno_dir, val_env_custom + ".json"), "r"
        ) as f:
            val_instr_data = json.load(f)
        if target_args.end is None:
            target_args.end = len(val_instr_data)
        val_instr_data = val_instr_data[target_args.start : target_args.end]
        rank = 0
        # env for MapGPT
        target_val_env = R2RNavBatch_MapGPT(
            val_instr_data,
            target_args.connectivity_dir,
            batch_size=target_args.batch_size,
            seed=target_args.seed + rank,
            name=val_env_custom,
            args=target_args,
        )  # evaluation using all objects
        # env for Xagent
    elif args.target_agent == "NavGPT":
        from NavGPT.nav_src.utils.data import ImageObservationsDB

        # from NavGPT.nav_src.data_utils import construct_instrs

        feat_db = ImageObservationsDB(
            target_args.obs_dir,
            target_args.obs_summary_dir,
            target_args.obj_dir,
            target_args.obs_list_dir,
        )

        val_instr_data = construct_instrs(
            target_args.anno_dir,
            target_args.dataset,
            [val_env_custom],
        )
        target_val_env = R2RNavBatch_NavGPT(
            feat_db,
            val_instr_data,
            target_args.connectivity_dir,
            target_args.navigable_dir,
            batch_size=target_args.batch_size,
            seed=target_args.seed,
            name=val_env_custom,
        )
    elif args.target_agent == "NavGPT2":
        from NavGPT_2.map_nav_src.utils.data import ImageFeaturesDB
        from NavGPT_2.map_nav_src.r2r.env import R2RNavBatch

        # from NavGPT_2.map_nav_src.r2r.data_utils import construct_instrs

        feat_db = ImageFeaturesDB(target_args.img_ft_file, target_args.image_feat_size)
        val_instr_data = construct_instrs(
            target_args.anno_dir,
            target_args.dataset,
            [val_env_custom],
            tokenizer=target_args.tokenizer,
            max_instr_len=target_args.max_instr_len,
            is_test=False,
        )
        target_val_env = R2RNavBatch(
            feat_db,
            val_instr_data,
            target_args.connectivity_dir,
            target_args.candidate_file_dir,
            batch_size=target_args.batch_size,
            seed=target_args.seed + rank,
            sel_data_idxs=None,
            name=val_env_custom,
        )

    if val_env_custom_surr is not None:
        target_val_env_surr = R2RBatch(
            feat_dict,
            batch_size=args.batchSize,
            splits=[val_env_custom_surr],
            tokenizer=tok,
        )

    # training env evaluation
    if train_env_custom is not None:
        evaluation_train = Evaluation([train_env_custom], featurized_scans, tok)

    if args.train == "mask":
        train(
            train_env,
            tok,
            args.iters,
            log_every=int(1e6),
            val_envs=val_envs,
            target_train_env=target_train_env,
            target_eval_env=target_val_env,
        )
    elif args.train == "testmask":
        valid_mask(
            train_env,
            tok,
            val_envs=val_envs,
            eval_env=target_val_env_surr,
            target_eval_env=target_val_env,
        )
    elif args.train == "feature" or args.train == "feature_bagging":
        filter_list = []
        with open("filter_list.txt", "r") as f:
            for line in f:
                if line.strip() == "":
                    continue
                filter_list.append(int(line.strip()))
        print("filter_list: ", filter_list)
        from collections import OrderedDict

        env_list = [train_env, val_envs, target_val_env_surr, target_val_env]
        do_filter = False
        if do_filter:
            for env in env_list:
                if env is not None and isinstance(env, OrderedDict):
                    for key, value in env.items():
                        # print("env name: ", value[0].name)
                        filter_env(value[0], filter_list, None)
                else:
                    filter_env(env, filter_list, None)
                # print("env name: ", env.name)
        # exit()

        valid_feature(
            train_env,
            tok,
            val_envs=val_envs,
            eval_env=target_val_env_surr,
            target_eval_env=target_val_env,
        )
        # valid_mask(
        #     train_env, tok, val_envs=val_envs, target_eval_env=target_val_env_surr
        # )

    else:
        assert False


def filter_env(env, filter_list, start_end):
    if start_end is not None:
        filter_list = filter_list[start_end[0] : start_end[1]]
    else:
        filter_list = filter_list
    # env.data is a list, so use a list comprehension to select items by index
    env.data = [env.data[i] for i in filter_list]
    # for e in env.data:
    #     print(e["instr_id"])
    # exit()
    return env


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    if (
        args.train == "mask"
        or args.train == "testmask"
        or args.train == "feature"
        or args.train == "feature_bagging"
    ):
        train_val(
            test_only=args.test_only,
            val_env_custom=args.val_set_custom,
            val_env_custom_surr=args.val_set_custom_surr,
            train_env_custom=args.training_set_custom,
        )
    else:
        assert False

    # add val_env_diy to args
