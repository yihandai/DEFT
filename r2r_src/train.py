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
from eval import Evaluation
from param import args

import warnings

warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter

from vlnbert.vlnbert_init import get_tokenizer


def bootstrap_sample(data_size, sample_ratio=0.8):
    """
    Create bootstrap sample indices for bagging.

    Args:
        data_size: Total size of the dataset
        sample_ratio: Fraction of data to sample (with replacement)
        random_seed: Random seed for reproducibility

    Returns:
        List of indices for the bootstrap sample
    """
    # if random_seed is not None:
    #     np.random.seed(random_seed)

    sample_size = int(data_size * sample_ratio)
    # Bootstrap sampling with replacement
    indices = np.random.choice(data_size, size=sample_size, replace=True)
    return indices.tolist()


class BootstrapR2RBatch(R2RBatch):
    """R2RBatch with custom bootstrap data"""

    def __init__(self, base_env, bootstrap_data, agent_id):
        # Copy all attributes from base environment
        self.env = base_env.env
        self.feature_size = base_env.feature_size
        self.data = bootstrap_data
        self.tok = base_env.tok
        self.name = f"bootstrap_{agent_id}"
        self.scans = base_env.scans
        self.splits = base_env.splits
        self.seed = base_env.seed + agent_id
        self.ix = 0
        self.batch_size = base_env.batch_size
        self._load_nav_graphs()
        self.angle_feature = base_env.angle_feature
        self.sim = base_env.sim
        self.buffered_state_dict = {}
        self.gt_trajs = self._get_gt_trajs(self.data)
        self.fake_data = self.data
        self.data_dict = {x["instr_id"]: x for x in self.data}

        print(f"BootstrapR2RBatch {agent_id} loaded with {len(self.data)} instructions")


def create_bootstrap_datasets(train_env, n_agents, bootstrap_ratio=0.8):
    """
    Create multiple bootstrap datasets for bagging.

    Args:
        train_env: Training environment
        n_agents: Number of agents to train
        bootstrap_ratio: Ratio of data to sample for each bootstrap

    Returns:
        List of bootstrap datasets
    """
    # Get the total number of training samples
    total_samples = len(train_env.data)
    bootstrap_datasets = []

    for i in range(n_agents):
        # Create bootstrap sample indices
        bootstrap_indices = bootstrap_sample(total_samples, bootstrap_ratio)

        # Create a new environment with bootstrap sampled data
        bootstrap_data = [train_env.data[idx] for idx in bootstrap_indices]

        # Create new environment with bootstrap data
        bootstrap_env = BootstrapR2RBatch(train_env, bootstrap_data, i)

        bootstrap_datasets.append(bootstrap_env)
        print(
            f"Created bootstrap dataset {i+1}/{n_agents} with {len(bootstrap_data)} samples"
        )

    return bootstrap_datasets


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

metric_1 = "nDTW"
metric_2 = "CLS"

""" train the listener """


def handle_evaluation_and_saving(
    args, loss_str, best_val, idx, iter, listner, start, n_iters, agent_id=None
):
    """
    Handle evaluation results and save models based on performance.

    Args:
        args: Command line arguments
        loss_str: String containing evaluation results
        best_val: Dictionary tracking best validation scores
        idx: Current iteration index
        iter: Current iteration number
        listner: The agent/listener model
        start: Training start time
        n_iters: Total number of iterations
        agent_id: ID of the agent (for ensemble training)
    """
    # Check if any validation environment achieved a new best score
    for env_name in best_val:
        if best_val[env_name]["update"]:
            best_val[env_name]["state"] = loss_str
            best_val[env_name]["update"] = False
            if agent_id is None:
                listner.save(
                    idx,
                    os.path.join("snap", args.name, "state_dict", f"best_{env_name}"),
                )
            else:
                listner.save(
                    idx,
                    os.path.join(
                        "snap",
                        args.name,
                        "state_dict",
                        f"agent_{agent_id}",
                        f"best_{env_name}",
                    ),
                )
            print(f"Save best val for {env_name} for agent {agent_id}")
        else:
            if agent_id is None:
                listner.save(
                    idx, os.path.join("snap", args.name, "state_dict", "latest_dict")
                )
            else:
                listner.save(
                    idx,
                    os.path.join(
                        "snap",
                        args.name,
                        "state_dict",
                        f"agent_{agent_id}",
                        "latest_dict",
                    ),
                )

    # Log results to file
    record_file = open("./logs/" + args.name + ".txt", "a")
    record_file.write(loss_str + "\n")
    record_file.close()

    # # Print periodic best results summary
    # if iter % 1000 == 0:
    #     print("BEST RESULT TILL NOW")
    #     for env_name in best_val:
    #         print(env_name, best_val[env_name]["state"])

    #         record_file = open("./logs/" + args.name + ".txt", "a")
    #         record_file.write(
    #             "BEST RESULT TILL NOW: "
    #             + env_name
    #             + " | "
    #             + best_val[env_name]["state"]
    #             + "\n"
    #         )
    #         record_file.close()


def evaluate_model(
    listner, val_envs, writer, best_val, iter_num, start_time, metric_1, metric_2
):
    """
    Evaluate the model on validation environments and update best validation scores.

    Args:
        listner: The agent/listener to evaluate
        val_envs: Dictionary of validation environments
        writer: TensorBoard writer
        best_val: Dictionary tracking best validation scores
        iter_num: Current iteration number
        start_time: Training start time
        metric_1: Primary metric for evaluation
        metric_2: Secondary metric for evaluation

    Returns:
        loss_str: String containing evaluation results
    """
    loss_str = "iter {}".format(iter_num)
    for env_name, (env, evaluator) in val_envs.items():
        listner.env = env

        # Get validation distance from goal under test evaluation conditions
        listner.test(use_dropout=False, feedback="argmax", iters=None)
        result = listner.get_results()
        score_summary, _ = evaluator.score(result)
        loss_str += ", %s " % env_name
        for metric, val in score_summary.items():
            if metric in [metric_1]:
                writer.add_scalar("ndtw/%s" % env_name, val, iter_num)
                if env_name in best_val:
                    if val > best_val[env_name][metric_1]:
                        best_val[env_name][metric_1] = val
                        best_val[env_name]["update"] = True
                    elif (val == best_val[env_name][metric_1]) and (
                        score_summary[metric_2] > best_val[env_name][metric_2]
                    ):
                        best_val[env_name][metric_1] = val
                        best_val[env_name]["update"] = True
            loss_str += ", %s: %.4f" % (metric, val)

    print("Evaluation metrics:")
    print(
        (
            "%s (%d %d%%) %s"
            % (
                timeSince(
                    start_time, 0.000001 if iter_num == 0 else float(iter_num) / 1000
                ),
                iter_num,
                0 if iter_num == 0 else float(iter_num) / 1000 * 100,
                loss_str,
            )
        )
    )

    return loss_str


def train(
    train_env,
    tok,
    n_iters,
    log_every=2000,
    val_envs={},
    aug_env=None,
    evaluator_train=None,
):
    writer = SummaryWriter(log_dir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    record_file = open("./logs/" + args.name + ".txt", "a")
    record_file.write(str(args) + "\n\n")
    record_file.close()

    # pretrained_wt = "snap/VLNBERT-PREVALENT-final/state_dict/pretrained"
    # print(
    #     "Loaded the listener model at iter %d from %s"
    #     % (listner.load(pretrained_wt), pretrained_wt)
    # )
    start_iter = 0
    if args.load is not None:
        start_iter = listner.load(os.path.join(args.load))
        print("\nLOAD the model from {}, iteration ".format(args.load, start_iter))

    start = time.time()
    print("\nListener training starts, start iteration: %s" % str(start_iter))

    # best_val = {"val_unseen": {"spl": 0.0, "sr": 0.0, "state": "", "update": False}}
    best_val = {
        "val_unseen": {"spl": 0.0, "sr": 0.0, "state": "", "update": False},
        "val72": {"update": False, metric_1: 0.0, metric_2: 0.0},
        "val72_navgpt": {"update": False, metric_1: 0.0, metric_2: 0.0},
        "val72_navgpt2": {"update": False, metric_1: 0.0, metric_2: 0.0},
    }
    print("Starting evaluation before training")
    # eval before training
    evaluate_model(listner, val_envs, writer, best_val, 0, start, metric_1, metric_2)
    print("Starting training loop")
    # training loop
    for idx in range(start_iter, start_iter + n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters - idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            if args.train == "surrogate":
                listner.train(
                    interval, feedback=feedback_method
                )  # Train interval iters
            elif args.train == "testmask":
                listner.train(
                    interval, feedback=feedback_method, mask=True
                )  # Train interval iters
            elif args.train == "surrogate_gail":
                listner.train_gail(interval)
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                args.ml_weight = 0.2
                listner.train(1, feedback=feedback_method)

                # Train with Augmented data
                listner.env = aug_env
                args.ml_weight = 0.2
                listner.train(1, feedback=feedback_method)

                print_progress(
                    jdx,
                    jdx_length,
                    prefix="Progress:",
                    suffix="Complete",
                    bar_length=50,
                )

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

        if args.train == "surrogate_gail":
            DISC_loss = sum(listner.logs["DISC_LOSS"]) / max(
                len(listner.logs["DISC_LOSS"]), 1
            )
            AC_loss = sum(listner.logs["AC_LOSS"]) / max(
                len(listner.logs["AC_LOSS"]), 1
            )
            print(
                "Iter {}   Disc loss {}   AC loss {}".format(iter, DISC_loss, AC_loss)
            )
        else:
            # Run validation
            print("Iter {}   IL loss {}   RL loss {}".format(iter, IL_loss, RL_loss))

        # training metrics
        if evaluator_train is not None:
            loss_str = "iter {}".format(iter)
            listner.test(use_dropout=False, feedback="argmax", iters=None)
            result = listner.get_results()
            score_summary, _ = evaluator_train.score(result)
            # loss_str += ", %s " % env_name
            loss_str += ", training"
            for metric, val in score_summary.items():
                loss_str += ", %s: %.4f" % (metric, val)
            print("Training metrics:")
            print(
                (
                    "%s (%d %d%%) %s"
                    % (
                        timeSince(start, 0.000001),
                        0,
                        0,
                        loss_str,
                    )
                )
            )

        # Use evaluate_model to handle validation/recording logic
        loss_str = evaluate_model(
            listner, val_envs, writer, best_val, idx, start, metric_1, metric_2
        )

        handle_evaluation_and_saving(
            args, loss_str, best_val, idx, iter, listner, start, n_iters
        )

        # if iter % 1000 == 0:
        #     print("BEST RESULT TILL NOW")
        #     for env_name in best_val:
        #         print(env_name, best_val[env_name]["state"])

        #         record_file = open("./logs/" + args.name + ".txt", "a")
        #         record_file.write(
        #             "BEST RESULT TILL NOW: "
        #             + env_name
        #             + " | "
        #             + best_val[env_name]["state"]
        #             + "\n"
        #         )
        #         record_file.close()

        listner.save(idx, os.path.join("snap", args.name, "state_dict", "latest_dict"))


def train_bagging(
    train_env,
    tok,
    n_iters,
    log_every=2000,
    val_envs={},
    n_agents=5,
    bootstrap_ratio=0.8,
    evaluator_train=None,
):
    """
    Train multiple agents using bootstrap aggregating (bagging).

    Args:
        train_env: Training environment
        tok: Tokenizer
        n_iters: Number of training iterations
        log_every: Logging frequency
        val_envs: Validation environments
        n_agents: Number of agents to train
        bootstrap_ratio: Ratio of data to sample for each bootstrap
        evaluator_train: Training evaluator
    """
    print(f"Starting bagging training with {n_agents} agents")

    # Create bootstrap datasets
    bootstrap_datasets = create_bootstrap_datasets(train_env, n_agents, bootstrap_ratio)

    # Train each agent on its bootstrap dataset
    trained_agents = []

    # create a log directory for the ensemble
    ensemble_log_dir = f"./logs/ensemble"
    if not os.path.exists(ensemble_log_dir):
        os.makedirs(ensemble_log_dir)
    start_agent_id = args.start_agent_id
    end_agent_id = start_agent_id + 1
    # for agent_id in range(n_agents):
    for agent_id in range(start_agent_id, end_agent_id):
        print(f"\n=== Training Agent {agent_id + 1}/{n_agents} ===")

        # # Create agent-specific log directory
        # agent_log_dir = f"snap/{args.name}_agent_{agent_id}"
        # if not os.path.exists(agent_log_dir):
        #     os.makedirs(agent_log_dir)

        agent_log_dir = os.path.join("snap", args.name)
        os.makedirs(agent_log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=agent_log_dir)
        agent = Seq2SeqAgent(bootstrap_datasets[agent_id], "", tok, args.maxAction)

        # Log args to agent-specific log
        record_file = open(f"{ensemble_log_dir}/agent_{agent_id}.txt", "a")
        record_file.write(str(args) + f"\nAgent {agent_id}\n\n")
        record_file.close()

        # Load pretrained weights if specified
        start_iter = 0
        if args.load is not None:
            if args.aug is None:
                start_iter = agent.load(os.path.join(args.load))
                print(f"\nLOAD the model from {args.load}, iteration {start_iter}")
            else:
                load_iter = agent.load(os.path.join(args.load))
                print(f"\nLOAD the model from {args.load}, iteration {load_iter}")

        start = time.time()
        print(f"\nAgent {agent_id + 1} training starts, start iteration: {start_iter}")

        # Best validation tracking for this agent
        best_val = {
            "val_unseen": {"spl": 0.0, "sr": 0.0, "state": "", "update": False},
            "val72": {"update": False, metric_1: 0.0, metric_2: 0.0},
            "val72_navgpt": {"update": False, metric_1: 0.0, metric_2: 0.0},
            "val72_navgpt2": {"update": False, metric_1: 0.0, metric_2: 0.0},
        }

        # --- Evaluate before training starts ---
        print(
            f"Evaluating agent {agent_id + 1} before training begins (pre-training evaluation)"
        )
        pre_train_loss_str = evaluate_model(
            agent, val_envs, writer, best_val, start_iter, start, metric_1, metric_2
        )

        # Training loop for this agent
        for idx in range(start_iter, start_iter + n_iters, log_every):
            agent.logs = defaultdict(list)
            interval = min(log_every, n_iters - idx)
            iter = idx + interval

            # Train for log_every interval
            agent.env = bootstrap_datasets[agent_id]
            # if args.train == "surrogate":
            if args.train == "bagging":
                agent.train(interval, feedback=feedback_method)
            elif args.train == "testmask":
                agent.train(interval, feedback=feedback_method, mask=True)
            elif args.train == "surrogate_gail":
                agent.train_gail(interval)

            # Log training stats
            total = max(sum(agent.logs["total"]), 1)
            length = max(len(agent.logs["critic_loss"]), 1)
            critic_loss = sum(agent.logs["critic_loss"]) / total
            RL_loss = sum(agent.logs["RL_loss"]) / max(len(agent.logs["RL_loss"]), 1)
            IL_loss = sum(agent.logs["IL_loss"]) / max(len(agent.logs["IL_loss"]), 1)
            entropy = sum(agent.logs["entropy"]) / total

            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/RL_loss", RL_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)

            if args.train == "surrogate_gail":
                DISC_loss = sum(agent.logs["DISC_LOSS"]) / max(
                    len(agent.logs["DISC_LOSS"]), 1
                )
                AC_loss = sum(agent.logs["AC_LOSS"]) / max(
                    len(agent.logs["AC_LOSS"]), 1
                )
                print(
                    f"Agent {agent_id + 1} Iter {iter}   Disc loss {DISC_loss:.4f}   AC loss {AC_loss:.4f}"
                )
            else:
                print(
                    f"Agent {agent_id + 1} Iter {iter}   IL loss {IL_loss:.4f}   RL loss {RL_loss:.4f}"
                )

            # Use evaluate_model and handle_evaluation_and_saving for validation and saving
            loss_str = evaluate_model(
                agent, val_envs, writer, best_val, idx, start, metric_1, metric_2
            )
            handle_evaluation_and_saving(
                args, loss_str, best_val, idx, iter, agent, start, n_iters, agent_id
            )

        # Save final model for this agent
        agent.save(
            idx,
            os.path.join(
                agent_log_dir, "state_dict", f"agent_{agent_id}", "latest_dict"
            ),
        )

        trained_agents.append(agent)
        print(f"Completed training agent {agent_id + 1}/{n_agents}")

    print(f"\nCompleted training all {n_agents} agents")
    return trained_agents


class EnsembleAgent:
    """Ensemble agent that combines predictions from multiple trained agents"""

    def __init__(self, trained_agents):
        self.agents = trained_agents
        self.n_agents = len(trained_agents)
        print(f"Created ensemble with {self.n_agents} agents")

    def test(self, use_dropout=False, feedback="argmax", iters=None):
        """
        Test the ensemble by running all agents and combining their predictions.
        For now, we'll use majority voting for action selection.
        """
        # Run each agent and collect their results
        all_results = []
        for i, agent in enumerate(self.agents):
            print(f"Running agent {i+1}/{self.n_agents}")
            agent.test(use_dropout=use_dropout, feedback=feedback, iters=iters)
            result = agent.get_results()
            all_results.append(result)

        # Combine results using majority voting
        self.ensemble_results = self._combine_results(all_results)
        return self.ensemble_results

    def _combine_results(self, all_results):
        """
        Combine results from multiple agents using majority voting.
        This is a simplified approach - in practice, you might want more sophisticated ensemble methods.
        """
        if not all_results:
            return []

        # For now, return the results from the first agent
        # In a more sophisticated implementation, you would:
        # 1. Collect all trajectories for each instruction
        # 2. Apply majority voting or other ensemble methods
        # 3. Return the combined trajectory

        print("Using first agent's results for ensemble (simplified implementation)")
        return all_results[0]

    def get_results(self):
        """Get the ensemble results"""
        return self.ensemble_results if hasattr(self, "ensemble_results") else []


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


def valid_mask(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print(
        "Loaded the listener model at iter %d from %s"
        % (agent.load(args.load), args.load)
    )
    print(
        "Loaded the mask model at iter %d from %s"
        % (agent.load_mask(args.loadmask), args.loadmask)
    )

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        expected_reward_preservation, fidelity_score = agent.test_mask(iters=iters)
        print(
            "expected_reward_preservation: {}, fidelity_score: {}".format(
                expected_reward_preservation, fidelity_score
            )
        )
        # result = agent.get_results()

        # if env_name != "":
        #     score_summary, _ = evaluator.score(result)
        #     loss_str = "Env name: %s" % env_name
        #     for metric, val in score_summary.items():
        #         loss_str += ", %s: %.4f" % (metric, val)
        #     print(loss_str)

        # if args.submit:
        #     json.dump(
        #         result,
        #         open(os.path.join(log_dir, "submit_%s.json" % env_name), "w"),
        #         sort_keys=True,
        #         indent=4,
        #         separators=(",", ": "),
        #     )


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(0)
    np.random.seed(0)


def train_val(test_only=False, train_env_custom=None, val_env_custom=None):
    """Train on the training set, and validate on seen and unseen splits."""
    setup()
    tok = get_tokenizer(args)

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
    else:
        train_env = R2RBatch(
            feat_dict, batch_size=args.batchSize, splits=["train"], tokenizer=tok
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

    # training env evaluation
    evaluation_train = Evaluation([train_env_custom], featurized_scans, tok)

    if args.train == "listener":
        train(train_env, tok, args.iters, log_every=int(1e6), val_envs=val_envs)
    elif args.train == "validlistener":
        valid(train_env, tok, val_envs=val_envs)
    elif args.train == "testmask":
        valid_mask(train_env, tok, val_envs=val_envs)

    elif args.train == "surrogate" or args.train == "surrogate_gail":
        train(
            train_env,
            tok,
            args.iters,
            log_every=int(1e2),
            val_envs=val_envs,
            evaluator_train=evaluation_train,
        )
    elif args.train == "bagging":
        # Train multiple agents using bagging
        trained_agents = train_bagging(
            train_env,
            tok,
            args.iters,
            log_every=int(1e2),
            val_envs=val_envs,
            n_agents=args.bagging_agents,
            bootstrap_ratio=args.bootstrap_ratio,
            evaluator_train=evaluation_train,
        )

        # Create ensemble agent
        ensemble_agent = EnsembleAgent(trained_agents)

        # Test ensemble on validation sets
        print("\n=== Testing Ensemble Agent ===")
        for env_name, (env, evaluator) in val_envs.items():
            print(f"\nTesting on {env_name}")
            ensemble_agent.test(use_dropout=False, feedback="argmax", iters=None)
            result = ensemble_agent.get_results()
            score_summary, _ = evaluator.score(result)

            loss_str = f"Ensemble {env_name}"
            for metric, val in score_summary.items():
                loss_str += f", {metric}: {val:.4f}"
            print(loss_str)

            # Save ensemble results
            if args.submit:
                json.dump(
                    result,
                    open(
                        os.path.join(log_dir, f"ensemble_submit_{env_name}.json"), "w"
                    ),
                    sort_keys=True,
                    indent=4,
                    separators=(",", ": "),
                )
    else:
        assert False


def train_val_augment(test_only=False):
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    tok_bert = get_tokenizer(args)

    # Load the env img features
    feat_dict = read_img_features(features, test_only=test_only)

    if test_only:
        featurized_scans = None
        val_env_names = ["val_train_seen"]
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = ["val_train_seen", "val_seen", "val_unseen"]

    # Load the augmentation data
    aug_path = args.aug
    # Create the training environment
    train_env = R2RBatch(
        feat_dict, batch_size=args.batchSize, splits=["train"], tokenizer=tok_bert
    )
    aug_env = R2RBatch(
        feat_dict,
        batch_size=args.batchSize,
        splits=[aug_path],
        tokenizer=tok_bert,
        name="aug",
    )

    # Setup the validation data
    val_envs = {
        split: (
            R2RBatch(
                feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok_bert
            ),
            Evaluation([split], featurized_scans, tok_bert),
        )
        for split in val_env_names
    }

    # Start training
    train(train_env, tok_bert, args.iters, val_envs=val_envs, aug_env=aug_env)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    if args.train in ["listener", "validlistener"]:
        train_val(test_only=args.test_only)
    elif args.train == "auglistener":
        train_val_augment(test_only=args.test_only)
    elif args.train == "testmask":
        train_val(test_only=args.test_only)
    elif args.train == "surrogate" or args.train == "surrogate_gail":
        train_val(
            test_only=args.test_only,
            train_env_custom=args.training_set_custom,
            val_env_custom=args.val_set_custom,
        )
    elif args.train == "bagging":
        train_val(
            test_only=args.test_only,
            train_env_custom=args.training_set_custom,
            val_env_custom=args.val_set_custom,
        )
    else:
        assert False

    # add val_env_diy to args
