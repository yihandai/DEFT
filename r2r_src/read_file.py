import os
import pickle
import random
import numpy as np

from r2r_src.test_util import select_critical_steps, compute_fidelity_score
from r2r_src.agent import Seq2SeqAgent
from r2r_src.env import R2RBatch
from r2r_src.vln_utils import read_img_features
from r2r_src.vlnbert.vlnbert_init import get_tokenizer

# args_name = "VLNBERT-train-mask-mapgpt"
# features = "img_features/ResNet-152-places365.tsv"
# ENV_NAME = "MapGPT_72_scenes_processed"

args_name = "VLNBERT-test-mask-navgpt2"
# features = "img_features/ResNet-152-places365_24vp.tsv"
features = "img_features/ResNet-152-places365.tsv"
ENV_NAME = "MapGPT_72_scenes_processed"

# Cache for agent and env to avoid repeated loading
_cached_env = None
_cached_tok = None
_cached_agent = None


def save_intermediate(dir_name, file_name, test_dict):
    the_dir = os.path.join("snap", args_name, "intermediate_test", dir_name)
    os.makedirs(the_dir, exist_ok=True)
    with open(os.path.join(the_dir, file_name + ".pkl"), "wb") as f:
        pickle.dump(
            test_dict,
            f,
        )


def load_intermediate(dir_name, file_name):
    the_dir = os.path.join("snap", args_name, "intermediate_test", dir_name)
    with open(os.path.join(the_dir, file_name + ".pkl"), "rb") as f:
        data = pickle.load(f)
    return data


# rewrite following code to function and choose "ours" acorrding to args_name
def evaluate_fidelity_score(dir_name=None):
    """
    Loads intermediate results and computes the fidelity score.
    If dir_name is not provided, chooses 'ours' according to args_name.
    """

    # rewards_baseline = load_intermediate(
    #     dir_name,
    #     "baseline",
    # )["rewards"]

    trajs_baseline = load_intermediate(
        dir_name,
        "baseline",
    )["trajs"]

    agent = load_agent()

    rewards_baseline, dis_rewards_baseline = collect_rewards(agent, trajs_baseline)
    mask_probs_baseline = load_intermediate(
        dir_name,
        "baseline",
    )["mask_probs"]

    counts_baseline = load_intermediate(
        dir_name,
        "baseline",
    )["counts"]

    # rewards_replay = load_intermediate(
    #     dir_name,
    #     "replay",
    # )["rewards"]
    trajs_replay = load_intermediate(
        dir_name,
        "replay",
    )["trajs"]
    rewards_replay, dis_rewards_replay = collect_rewards(agent, trajs_replay)

    critical_steps_starts, critical_steps_ends = select_critical_steps(
        mask_probs_baseline, counts_baseline, random_zone=True
    )

    # for i in range(100):
    #     print(critical_steps_starts[i], critical_steps_ends[i])

    fidelity_score_baseline = compute_fidelity_score(
        critical_steps_starts,
        critical_steps_ends,
        counts_baseline,
        # rewards_replay,
        # rewards_baseline,
        dis_rewards_replay,
        dis_rewards_baseline,
    )
    return fidelity_score_baseline


def load_env(env_name):
    global _cached_env, _cached_tok

    # Return cached env and tok if already loaded
    if _cached_env is not None and _cached_tok is not None:
        return _cached_env, _cached_tok

    feat_dict = read_img_features(features, test_only=0)

    # Create a pseudo args object with attribute vlnbert set to "prevalent"
    class PseudoArgs:
        pass

    args = PseudoArgs()
    args.vlnbert = "prevalent"

    tok = get_tokenizer(args)

    target_val_env_surr = R2RBatch(
        feat_dict,
        batch_size=1,
        splits=[env_name],
        tokenizer=tok,
    )

    # Cache the env and tok
    _cached_env = target_val_env_surr
    _cached_tok = tok

    return _cached_env, _cached_tok


def load_agent():
    global _cached_agent

    # Return cached agent if already loaded
    if _cached_agent is not None:
        return _cached_agent

    env, tok = load_env(ENV_NAME)
    agent = Seq2SeqAgent(env, "", tok, 15)

    # Cache the agent
    _cached_agent = agent

    return _cached_agent


def collect_rewards(agent, trajs):
    rewards = []
    dis_rewards = []
    for i, traj in enumerate(trajs):
        reward, dis_reward = psudo_rollout(agent, traj)
        rewards.append(reward)
        dis_rewards.append(dis_reward)
    return rewards, dis_rewards


def psudo_rollout(agent, traj):
    # get instr_id -> data dict
    data_dict = agent.env.data_dict

    batch_size = 1
    instr_id = traj["instr_id"]
    path = traj["path"]

    # init obs
    scanId = data_dict[instr_id]["scan"]
    # print("ScanId", scanId)
    viewpoint, heading, _ = path[0]
    obs = agent.env.set_scan_viewpoint_heading(
        {
            "scanIds": [scanId],
            "viewpointIds": [viewpoint],
            "headings": [heading],
            "instr_ids": [instr_id],
        }
    )

    # Init the reward shaping
    last_dist = np.zeros(batch_size, np.float32)
    last_ndtw = np.zeros(batch_size, np.float32)

    total_reward, total_discounted_reward = 0, 0

    for i, ob in enumerate(obs):  # The init distance from the view point to the target
        last_dist[i] = ob["distance"]
        path_act = [vp[0] for vp in traj["path"][0:1]]
        last_ndtw[i] = agent.ndtw_criterion[ob["scan"]](
            path_act, ob["gt_path"], metric="ndtw"
        )

    for t, (viewpoint, heading, elevation) in enumerate(path[1:]):
        scanId = data_dict[instr_id]["scan"]
        obs = agent.env.set_scan_viewpoint_heading(
            {
                "scanIds": [scanId],
                "viewpointIds": [viewpoint],
                "headings": [heading],
                "instr_ids": [instr_id],
            }
        )

        # Calculate the mask and reward
        dist = np.zeros(batch_size, np.float32)
        ndtw_score = np.zeros(batch_size, np.float32)
        reward = np.zeros(batch_size, np.float32)
        mask = np.ones(batch_size, np.float32)
        for i, ob in enumerate(obs):
            dist[i] = ob["distance"]
            path_act = [vp[0] for vp in traj["path"][: t + 2]]
            ndtw_score[i] = agent.ndtw_criterion[ob["scan"]](
                path_act, ob["gt_path"], metric="ndtw"
            )

            # action_idx = cpu_a_t[i]
            # Target reward
            # if action_idx == -1:  # If the action now is end
            if len(path) - 2 == t:  # If the action now is end
                if dist[i] < 3.0:  # Correct
                    reward[i] = 2.0 + ndtw_score[i] * 2.0
                else:  # Incorrect
                    reward[i] = -2.0
            else:  # The action is not end
                # Path fidelity rewards (distance & nDTW)
                reward[i] = -(dist[i] - last_dist[i])
                ndtw_reward = ndtw_score[i] - last_ndtw[i]
                if reward[i] > 0.0:  # Quantification
                    reward[i] = 1.0 + ndtw_reward
                elif reward[i] < 0.0:
                    reward[i] = -1.0 + ndtw_reward
                # else:
                #     raise NameError("The action doesn't change the move")
                # Miss the target penalty
                if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                    reward[i] -= (1.0 - last_dist[i]) * 2.0
        # rewards.append(reward)
        # masks.append(mask)
        last_dist[:] = dist
        last_ndtw[:] = ndtw_score

        total_reward += reward[0]
        total_discounted_reward += np.power(agent.GAE, t) * reward[0]
    # print("total_reward", total_reward)
    # print("total_discounted_reward", total_discounted_reward)
    return total_reward, total_discounted_reward


def compute_RRD(random_file, value_file):
    """
    Chen J, Wang Y, Wang J, et al. Understanding Individual Agent Importance in Multi-Agent System via Counterfactual Reasoning[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2025, 39(15): 15785-15794.
    """
    # collect discounted rewards from random_file and value_file
    # compute the RRD
    epsilon = 1e-8
    random_trajs = load_intermediate(random_file, "replay")["trajs"]
    value_trajs = load_intermediate(value_file, "replay")["trajs"]
    baseline_trajs = load_intermediate(value_file, "baseline")["trajs"]
    agent = load_agent()
    random_rewards, random_dis_rewards = collect_rewards(agent, random_trajs)
    value_rewards, value_dis_rewards = collect_rewards(agent, value_trajs)
    baseline_rewards, baseline_dis_rewards = collect_rewards(agent, baseline_trajs)

    # RRD = np.mean(random_dis_rewards) - np.mean(value_dis_rewards)
    # RRD = |R_v - R_b| / |R_r - R_b|
    # fix list to np
    value_dis_rewards = np.array(value_dis_rewards)
    random_dis_rewards = np.array(random_dis_rewards)
    baseline_dis_rewards = np.array(baseline_dis_rewards)
    # print("value_dis_rewards[:4]", value_dis_rewards[:4])
    # print("baseline_dis_rewards[:4]", baseline_dis_rewards[:4])
    # print("random_dis_rewards[:4]", random_dis_rewards[:4])
    RRD_list = np.abs(value_dis_rewards - baseline_dis_rewards) + epsilon / np.abs(
        random_dis_rewards - baseline_dis_rewards + epsilon
    )
    RRD = np.mean(RRD_list)

    # change the abs outer function to log
    # To avoid log(0), add a small epsilon to the absolute differences
    value_diff = np.abs(value_dis_rewards - baseline_dis_rewards) + epsilon
    random_diff = np.abs(random_dis_rewards - baseline_dis_rewards) + epsilon
    log_RRD = np.mean(np.log(value_diff)) / np.mean(np.log(random_diff))

    return RRD, log_RRD, RRD_list


def normalized_length(value_file):
    mask_probs_baseline = load_intermediate(
        value_file,
        "baseline",
    )["mask_probs"]
    counts_baseline = load_intermediate(
        value_file,
        "baseline",
    )["counts"]
    critical_steps_starts, critical_steps_ends = select_critical_steps(
        mask_probs_baseline, counts_baseline, random_zone=False
    )
    # to np
    critical_steps_ends = np.array(critical_steps_ends)
    critical_steps_starts = np.array(critical_steps_starts)
    critical_length = np.mean(
        (critical_steps_ends - critical_steps_starts + 1) / counts_baseline
    )
    log_critical_length = np.mean(
        np.log(critical_steps_ends - critical_steps_starts + 1) / counts_baseline
    )
    return critical_length, log_critical_length


def normalized_mask_len_by_random(random_file, value_file):
    mask_probs_random = load_intermediate(random_file, "baseline")["mask_probs"]
    mask_probs_value = load_intermediate(value_file, "baseline")["mask_probs"]
    counts_baseline = load_intermediate(value_file, "baseline")["counts"]
    critical_steps_starts, critical_steps_ends = select_critical_steps(
        mask_probs_value, counts_baseline, random_zone=False
    )
    critical_steps_starts_random, critical_steps_ends_random = select_critical_steps(
        mask_probs_random, counts_baseline, random_zone=False
    )
    critical_steps_ends_random = np.array(critical_steps_ends_random)
    critical_steps_starts_random = np.array(critical_steps_starts_random)
    critical_steps_ends = np.array(critical_steps_ends)
    critical_steps_starts = np.array(critical_steps_starts)
    critical_ratio_list = (critical_steps_ends - critical_steps_starts + 1) / (
        critical_steps_ends_random - critical_steps_starts_random + 1
    )
    critical_ratio = np.mean(critical_ratio_list)
    # write critical_steps_ends - critical_steps_starts + 1 and critical_steps_ends_random - critical_steps_starts_random + 1 to csv
    # Save both arrays as columns in one csv file named value_file + 'len.csv'
    to_save = np.column_stack(
        [
            (critical_steps_ends - critical_steps_starts + 1).astype(int),
            (critical_steps_ends_random - critical_steps_starts_random + 1).astype(int),
        ]
    )
    csv_dir = "./csv"
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    np.savetxt(
        os.path.join(csv_dir, value_file + "len.csv"),
        to_save,
        fmt="%d",
        delimiter=",",
        header="value_mask_length,random_mask_length",
        comments="",
    )
    return critical_ratio, critical_ratio_list


if __name__ == "__main__":
    np.random.seed(42)
    # fidelity_score_baseline = evaluate_fidelity_score(dir_name="random2")
    # print(fidelity_score_baseline)
    # value_file = "value-based2"
    # value_file = "gradient_merge"
    # value_file = "gradient2"
    # value_files = ["value-based", "gradient_merge", "random_nopadding", "baseline"]
    # value_files = ["value-based", "statemask"]
    # value_files = ["gradient_merge"]
    # value_files = ["ablation"]
    # value_files = ["value-based3"]
    # value_files = ["ours4000_42", "ablation", "value-based5", "gradient"]
    # value_files = ["ours4110", "ours4100"]
    # value_files = ["ours4160"]
    # value_files = ["value-based4000_30p", "ours4000_30p"]
    value_files = ["value-based500"]
    for value_file in value_files:
        print(f"--------{value_file}--------")
        RRD, log_RRD, RRD_list = compute_RRD(
            # random_file="random_nopadding", value_file=value_file
            # random_file="random_30p",
            random_file="random",
            value_file=value_file,
        )
        print("########################")
        print("RRD", RRD)
        # print("log_RRD", log_RRD)
        critical_length, log_critical_length = normalized_length(value_file=value_file)
        # print("log_critical_length", log_critical_length)
        print("critical_length", critical_length)

        # compute fidelity score
        fidelity_score = RRD - critical_length
        print("RRD - critical_length", fidelity_score)
        # fidelity_score_log = log_RRD - log_critical_length
        # print("fidelity_score_log", fidelity_score_log)
        print("RRD/critical_length", RRD / critical_length)
        print("log(RRD/critical_length)", np.log(RRD / critical_length))

        # compute normalized mask len by random
        critical_ratio, critical_ratio_list = normalized_mask_len_by_random(
            # random_file="random_nopadding", value_file=value_file
            random_file="random",
            value_file=value_file,
        )
        print("!critical_ratio", critical_ratio)
        print("!RRD - critical_ratio", RRD - critical_ratio)
        print("!RRD/critical_ratio", RRD / critical_ratio)
        print("!log(RRD/critical_ratio)", np.log(RRD / critical_ratio))

        # compute fidelity score with outer mean
        print("########################")
        print("outer mean")
        print("RRD - critical_ratio", np.mean(RRD_list - critical_ratio_list))
        print("RRD/critical_ratio", np.mean(RRD_list / critical_ratio_list))
        print(
            "log(RRD/critical_ratio)", np.mean(np.log(RRD_list / critical_ratio_list))
        )
