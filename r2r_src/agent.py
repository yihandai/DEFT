# R2R-EnvDrop, 2019, haotan@cs.unc.edu
# Modified in Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

import json
import os
import sys
import numpy as np
import random
import math
import time
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
import r2r_src.vln_utils as vln_utils
from r2r_src.vln_utils import padding_idx, print_progress
import model_OSCAR, model_PREVALENT
import param
from param import args
from collections import defaultdict, deque
from test_util import select_critical_steps, compute_fidelity_score
from train_gail import train_discrim, train_actor_critic, train_actor_critic_v2
from gail_utils import get_reward
from eval_utils import cal_dtw

# NAVGPT_STEP_EVAL_DIR = "snap/VLNBERT-test-mask-navgpt-step"


class BaseAgent(object):
    """Base class for an R2R agent to generate and save trajectories."""

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = []  # For learning agents

    def write_results(self):
        output = [{"instr_id": k, "trajectory": v} for k, v in self.results.items()]
        with open(self.results_path, "w") as f:
            json.dump(output, f)

    def get_results(self):
        output = [{"instr_id": k, "trajectory": v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        """Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]"""
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name + "Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(
            shuffle=(iters is not None)
        )  # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj["instr_id"]] = traj["path"]
        else:  # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj["instr_id"] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj["instr_id"]] = traj["path"]
                if looped:
                    break


class Seq2SeqAgent(BaseAgent):
    """An agent based on an LSTM seq2seq model with attention."""

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
        "left": ([0], [-1], [0]),  # left
        "right": ([0], [1], [0]),  # right
        "up": ([0], [0], [1]),  # up
        "down": ([0], [0], [-1]),  # down
        "forward": ([1], [0], [0]),  # forward
        "<end>": ([0], [0], [0]),  # <end>
        "<start>": ([0], [0], [0]),  # <start>
        "<ignore>": ([0], [0], [0]),  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size
        self.ERROR_MARGIN = 3.0
        self.GAE = 0.99

        # Models
        if args.vlnbert == "oscar":
            self.vln_bert = model_OSCAR.VLNBERT(
                feature_size=self.feature_size + args.angle_feat_size
            ).cuda()
            self.critic = model_OSCAR.Critic().cuda()
        elif args.vlnbert == "prevalent":
            self.vln_bert = model_PREVALENT.VLNBERT(
                feature_size=self.feature_size + args.angle_feat_size
            ).cuda()
            if args.ablation_pretrained:
                self.vln_bert_noneupdate = model_PREVALENT.VLNBERT(
                    feature_size=self.feature_size + args.angle_feat_size
                ).cuda()
                self.vln_bert_noneupdate.eval()
                self.vln_bert_noneupdate_optimizer = args.optimizer(
                    self.vln_bert_noneupdate.parameters(), lr=args.lr
                )
            # if args.train != "surrogate" :
            if args.train == "mask":
                use_noneupdate = True
            else:
                use_noneupdate = False
            if use_noneupdate:
                self.vln_bert_noneupdate = model_PREVALENT.VLNBERT(
                    feature_size=self.feature_size + args.angle_feat_size
                ).cuda()
                self.vln_bert_noneupdate.eval()
                self.vln_bert_noneupdate_optimizer = args.optimizer(
                    self.vln_bert_noneupdate.parameters(), lr=args.lr
                )

            self.critic = model_PREVALENT.Critic().cuda()
            self.critic4mask = model_PREVALENT.Critic().cuda()
            self.critical_head = model_PREVALENT.CriticalHead().cuda()
            if args.GAIL:
                self.discrim = model_PREVALENT.Discriminator(
                    input_dim=768 + 2176
                ).cuda()

        if args.GAIL:
            self.models = (
                self.vln_bert,
                self.critic,
                self.critic4mask,
                self.critical_head,
                self.discrim,
            )
        else:
            self.models = (
                self.vln_bert,
                self.critic,
                self.critic4mask,
                self.critical_head,
            )

        # Optimizers
        self.vln_bert_optimizer = args.optimizer(self.vln_bert.parameters(), lr=args.lr)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Optimizers for mask training
        self.critical_head_optimizer = args.optimizer(
            self.critical_head.parameters(), lr=args.lr
        )
        self.critic_optimizer4mask = args.optimizer(
            self.critic4mask.parameters(), lr=args.lr
        )

        self.optimizers4mask = (
            self.critical_head_optimizer,
            self.critic_optimizer4mask,
        )
        if args.GAIL:  # Optimizers for GAIL
            self.optimizer4discrim = args.optimizer(
                self.discrim.parameters(), lr=args.lr
            )

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=args.ignoreid, size_average=False
        )
        self.ndtw_criterion = vln_utils.ndtw_initialize()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _sort_batch(self, obs):
        seq_tensor = np.array([ob["instr_encoding"] for ob in obs], dtype=np.float32)
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)  # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = sorted_tensor != padding_idx

        token_type_ids = torch.zeros_like(mask)

        return (
            Variable(sorted_tensor, requires_grad=False).long().cuda(),
            mask.long().cuda(),
            token_type_ids.long().cuda(),
            list(seq_lengths),
            list(perm_idx),
        )

    def DAgger_action(self, obs, vpids, ended, visited_masks=None, traj=None):
        """get psuedo action"""
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = args.ignoreid
            else:
                if ob["viewpoint"] == ob["gt_path"][-1]:
                    a[i] = len(ob["candidate"])  # Stop if arrived
                else:
                    scan = ob["scan"]
                    cur_vp = ob["viewpoint"]
                    min_idx, min_dist = args.ignoreid, float("inf")
                    for j, vpid in enumerate(vpids[i]):
                        # if j > 0 and (visited_masks is None):
                        if visited_masks is None:
                            if args.expert_policy == "ndtw":
                                path_vps = [[vp[0]] for vp in traj[i]["path"]]
                                dist = -cal_dtw(
                                    self.env.distances[scan],
                                    sum(path_vps, [])
                                    + self.env.paths[scan][ob["viewpoint"]][
                                        vpid["viewpointId"]
                                    ][1:],
                                    ob["gt_path"],
                                    threshold=3.0,
                                )["nDTW"]
                            elif args.expert_policy == "spl":
                                # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                                dist = (
                                    self.env.distances[scan][vpid["viewpointId"]][
                                        ob["gt_path"][-1]
                                    ]
                                    + self.env.distances[scan][cur_vp][
                                        vpid["viewpointId"]
                                    ]
                                )
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == args.ignoreid:
                        print("scan %s: all vps are searched" % (scan))
        return torch.from_numpy(a).cuda()

    def _feature_variable(self, obs):
        """Extract precomputed features into variable."""
        features = np.empty(
            (len(obs), args.views, self.feature_size + args.angle_feat_size),
            dtype=np.float32,
        )
        for i, ob in enumerate(obs):
            features[i, :, :] = ob["feature"]  # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob["candidate"]) + 1 for ob in obs]  # +1 is for the end
        candidate_feat = np.zeros(
            (len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size),
            dtype=np.float32,
        )
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob["candidate"]):
                candidate_feat[i, j, :] = cc["feature"]

        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = vln_utils.angle_feature(ob["heading"], ob["elevation"])
        input_a_t = torch.from_numpy(input_a_t).cuda()
        # f_t = self._feature_variable(obs)      # Pano image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, candidate_feat, candidate_leng

    def _teacher_action_baseline(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = args.ignoreid
            else:
                index_vp_next = None
                for index_vp, vp in enumerate(ob["gt_path"]):
                    # print("{} {}".format(vp, ob["viewpoint"]))
                    if vp == ob["viewpoint"] and index_vp not in self.instr_buffer[i]:
                        self.instr_buffer[i].append(index_vp)
                        index_vp_next = (
                            index_vp + 1
                            if index_vp != len(ob["gt_path"]) - 1
                            else len(ob["gt_path"]) - 1
                        )
                        break
                assert index_vp_next is not None, "the viewpoint is not in the path!"
                vp_next = ob["gt_path"][index_vp_next]
                for k, candidate in enumerate(ob["candidate"]):
                    if candidate["viewpointId"] == vp_next:  # Next view point
                        a[i] = k
                        break
                else:  # Stop here
                    assert (
                        vp_next == ob["viewpoint"]
                    )  # The teacher action should be "STAY HERE"
                    a[i] = len(ob["candidate"])
        return torch.from_numpy(a).cuda()

    def _teacher_action_baseline_navgpt(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            print("gt", ob["gt_path"])
            print("viewpoint", ob["viewpoint"])
            if ended[i]:  # Just ignore this index
                a[i] = args.ignoreid
            else:
                index_vp_next = None
                for index_vp, vp in enumerate(ob["gt_path"]):
                    # print("{} {}".format(vp, ob["viewpoint"]))
                    if vp == ob["viewpoint"] and index_vp not in self.instr_buffer[i]:
                        self.instr_buffer[i].append(index_vp)
                        index_vp_next = (
                            index_vp + 1
                            if index_vp != len(ob["gt_path"]) - 1
                            else len(ob["gt_path"]) - 1
                        )
                        break
                assert index_vp_next is not None, "the viewpoint is not in the path!"
                vp_next = ob["gt_path"][index_vp_next]
                candidate_dict = ob.get("candidate", {})
                candidate_list = list(candidate_dict.keys())
                # for k, candidate in enumerate(ob["candidate"]):
                for k, candidate in enumerate(candidate_list):
                    # if candidate["viewpointId"] == vp_next:  # Next view point
                    print("candidate", candidate)
                    if candidate == vp_next:  # Next view point
                        # NOTE NOTE NOTE
                        a[i] = k + 1
                        break
                else:  # Stop here
                    assert (
                        vp_next == ob["viewpoint"]
                    )  # The teacher action should be "STAY HERE"
                    # a[i] = len(ob["candidate"])
                    a[i] = 0
        return torch.from_numpy(a).cuda()

    def _teacher_action_baseline_navgpt2(self, obs, ended, vpids):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = args.ignoreid
            else:
                index_vp_next = None
                for index_vp, vp in enumerate(ob["gt_path"]):
                    # print("{} {}".format(vp, ob["viewpoint"]))
                    if vp == ob["viewpoint"] and index_vp not in self.instr_buffer[i]:
                        self.instr_buffer[i].append(index_vp)
                        index_vp_next = (
                            index_vp + 1
                            if index_vp != len(ob["gt_path"]) - 1
                            else len(ob["gt_path"]) - 1
                        )
                        break
                assert index_vp_next is not None, "the viewpoint is not in the path!"
                vp_next = ob["gt_path"][index_vp_next]
                for k, vpid in enumerate(vpids[i]):
                    if vpid == vp_next:
                        a[i] = k
                        break
                else:
                    assert (
                        vp_next == ob["viewpoint"]
                    )  # The teacher action should be "STAY HERE"
                    a[i] = 0
        return torch.from_numpy(a).cuda()

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob["candidate"]):
                    if candidate["viewpointId"] == ob["teacher"]:  # Next view point
                        a[i] = k
                        break
                else:  # Stop here
                    assert (
                        ob["teacher"] == ob["viewpoint"]
                    )  # The teacher action should be "STAY HERE"
                    a[i] = len(ob["candidate"])
        return torch.from_numpy(a).cuda()

    def generate_pseudo_action(self, logit, candidate_mask, mode="sample"):
        logit.masked_fill_(candidate_mask, -float("inf"))
        if mode == "sample":
            probs = F.softmax(logit, 1)  # sampling an action from model
            c = torch.distributions.Categorical(probs)
            a_t = c.sample().detach()
        elif mode == "argmax":
            _, a_t = logit.max(1)  # student forcing - argmax
            a_t = a_t.detach()
        return a_t

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        navgpt_mode = args.panoramic_horizontal_views == 8
        if navgpt_mode:  # navgpt mode
            for i, idx in enumerate(perm_idx):
                action = a_t[i]
                if action != -1:  # -1 is the <stop> action
                    select_candidate = perm_obs[i]["candidate"][action]
                    normalized_heading = select_candidate["normalized_heading"]
                    normalized_elevation = select_candidate["normalized_elevation"]
                    state = self.env.env.sims[idx].getState()[0]
                    current_heading = state.heading
                    rel_heading = normalized_heading - current_heading
                    self.env.env.sims[idx].newEpisode(
                        [select_candidate["scanId"]],
                        [select_candidate["viewpointId"]],
                        [normalized_heading],
                        [normalized_elevation],
                    )

                    # Update trajectory after action (same as non-navgpt mode)
                    state_after = self.env.env.sims[idx].getState()[0]
                    if traj is not None:
                        traj[i]["path"].append(
                            (
                                state_after.location.viewpointId,
                                state_after.heading,
                                state_after.elevation,
                            )
                        )
        else:  # non-navgpt mode

            def take_action(i, idx, name):
                if type(name) is int:  # Go to the next view
                    self.env.env.sims[idx].makeAction([name], [0], [0])
                else:  # Adjust
                    self.env.env.sims[idx].makeAction(*self.env_actions[name])

            if perm_idx is None:
                perm_idx = range(len(perm_obs))

            for i, idx in enumerate(perm_idx):
                action = a_t[i]
                if action != -1:  # -1 is the <stop> action
                    select_candidate = perm_obs[i]["candidate"][action]
                    src_point = perm_obs[i]["viewIndex"]
                    trg_point = select_candidate["pointId"]
                    num_horizontal_views = args.panoramic_horizontal_views
                    src_level = (
                        src_point
                    ) // num_horizontal_views  # The point idx started from 0
                    trg_level = (trg_point) // num_horizontal_views
                    while src_level < trg_level:  # Tune up
                        take_action(i, idx, "up")
                        src_level += 1
                    while src_level > trg_level:  # Tune down
                        take_action(i, idx, "down")
                        src_level -= 1
                    while (
                        self.env.env.sims[idx].getState()[0].viewIndex != trg_point
                    ):  # Turn right until the target
                        take_action(i, idx, "right")
                    assert (
                        select_candidate["viewpointId"]
                        == self.env.env.sims[idx]
                        .getState()[0]
                        .navigableLocations[select_candidate["idx"]]
                        .viewpointId
                    )
                    take_action(i, idx, select_candidate["idx"])

                    state = self.env.env.sims[idx].getState()[0]
                    if traj is not None:
                        traj[i]["path"].append(
                            (state.location.viewpointId, state.heading, state.elevation)
                        )

    def if_succeed(self, obs, traj):
        # batch_size = states["batch_size"]
        # obs = states["obs"]
        eta = []
        for i in range(len(obs)):
            scan = obs[i]["scan"]
            instr_id = obs[i]["instr_id"]
            scan, gt_traj = self.env.gt_trajs[instr_id]
            path = traj[i]["path"]
            final_position = path[-1][0]  # the first of [view_id, angle, vofv]
            nav_error = self.env.distances[scan][final_position][gt_traj[-1]]
            success = float(nav_error < self.ERROR_MARGIN)
            eta.append(success)
        return eta

    def rollout_mask(self, train_ml=None, train_rl=True, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        train_rl = True

        if reset:  # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        # Language input
        sentence, language_attention_mask, token_type_ids, seq_lengths, perm_idx = (
            self._sort_batch(obs)
        )
        perm_obs = obs[perm_idx]

        """ Language BERT """
        language_inputs = {
            "mode": "language",
            "sentence": sentence,
            "attention_mask": language_attention_mask,
            "lang_mask": language_attention_mask,
            "token_type_ids": token_type_ids,
        }
        if args.vlnbert == "oscar":
            language_features = self.vln_bert(**language_inputs)
        elif args.vlnbert == "prevalent":
            h_t, language_features = self.vln_bert(**language_inputs)

        # Record starting point
        traj = [
            {
                "instr_id": ob["instr_id"],
                "path": [(ob["viewpoint"], ob["heading"], ob["elevation"])],
            }
            for ob in perm_obs
        ]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(
            perm_obs
        ):  # The init distance from the view point to the target
            last_dist[i] = ob["distance"]
            path_act = [vp[0] for vp in traj[i]["path"]]
            last_ndtw[i] = self.ndtw_criterion[ob["scan"]](
                path_act, ob["gt_path"], metric="ndtw"
            )

        # Initialization the tracking state
        ended = np.array(
            [False] * batch_size
        )  # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        num_masks = []
        ml_loss = 0.0

        for t in range(self.episode_len):

            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            # the first [CLS] token, initialized by the language BERT, serves
            # as the agent's state passing through time steps
            if (t >= 1) or (args.vlnbert == "prevalent"):
                language_features = torch.cat(
                    (h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1
                )

            visual_temp_mask = (vln_utils.length2mask(candidate_leng) == 0).long()
            visual_attention_mask = torch.cat(
                (language_attention_mask, visual_temp_mask), dim=-1
            )

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            """ Visual BERT """
            visual_inputs = {
                "mode": "visual",
                "sentence": language_features,
                "attention_mask": visual_attention_mask,
                "lang_mask": language_attention_mask,
                "vis_mask": visual_temp_mask,
                "token_type_ids": token_type_ids,
                "action_feats": input_a_t,
                # 'pano_feats':         f_t,
                "cand_feats": candidate_feat,
            }
            h_t, logit = self.vln_bert(**visual_inputs)
            hidden_states.append(h_t)

            # # Mask outputs where agent can't move forward
            # # Here the logit is [b, max_candidate]
            candidate_mask = vln_utils.length2mask(candidate_leng)
            # logit.masked_fill_(candidate_mask, -float("inf"))

            # generate action with original policy
            B_action_copy = (
                self.generate_pseudo_action(logit, candidate_mask, mode="sample")
                .cpu()
                .numpy()
            )
            # generate mask
            critical_logits = self.critical_head(h_t)
            critical_probs = F.softmax(
                critical_logits, 1
            )  # sampling an action from model
            critical_c = torch.distributions.Categorical(critical_probs)
            critical_a_t = critical_c.sample().detach()
            # count the number of masks
            mask_action = 1 - critical_a_t
            num_mask = torch.sum(mask_action[~ended])
            num_masks.append(num_mask)
            policy_log_probs.append(critical_c.log_prob(critical_a_t))

            self.logs["entropy"].append(critical_c.entropy().sum().item())  # For log
            entropys.append(critical_c.entropy())  # For optimization

            # determine the real action
            mask_action_copy = critical_a_t.cpu().numpy()
            real_action = []
            for i in range(batch_size):
                # mask_action_copy[i] = 1
                if mask_action_copy[i] == 1:
                    real_action.append(B_action_copy[i])
                else:
                    # real_action.append(np.random.choice(len(B_action_options[i])))
                    n = candidate_leng[i]
                    if n == 0:
                        # handle the case without options (adjust according to actual needs)
                        real_action.append(-1)
                    elif n == 1:
                        # when there is only 1 option, select directly
                        real_action.append(0)
                    else:
                        # generate a random index that is not equal to B_action_copy[i]
                        while True:
                            idx = np.random.choice(n)
                            if idx != B_action_copy[i]:
                                real_action.append(idx)
                                break

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            # cpu_a_t = a_t.cpu().numpy()
            cpu_a_t = real_action
            for i, next_id in enumerate(cpu_a_t):
                if (
                    next_id == (candidate_leng[i] - 1)
                    or next_id == args.ignoreid
                    or ended[i]
                ):  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]  # Perm the obs for the resu

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(perm_obs):
                    dist[i] = ob["distance"]
                    path_act = [vp[0] for vp in traj[i]["path"]]
                    ndtw_score[i] = self.ndtw_criterion[ob["scan"]](
                        path_act, ob["gt_path"], metric="ndtw"
                    )
                    ndtw_score = last_ndtw = np.zeros(batch_size, np.float32)
                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:  # If the action now is end
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
                            else:
                                # Action didn't change distance - give a small penalty
                                reward[i] = -0.1  # Small penalty for not moving
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                    # reward += 0.1 * num_mask.cpu().numpy() # add the mask to the reward, the more masks, the better
                reward += (
                    1.5 * mask_action.cpu().numpy()
                )  # add the mask to the reward, the more masks, the better
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break
        # end for

        if train_rl:
            # Last action in A2C
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            language_features = torch.cat(
                (h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1
            )

            visual_temp_mask = (vln_utils.length2mask(candidate_leng) == 0).long()
            visual_attention_mask = torch.cat(
                (language_attention_mask, visual_temp_mask), dim=-1
            )

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            """ Visual BERT """
            visual_inputs = {
                "mode": "visual",
                "sentence": language_features,
                "attention_mask": visual_attention_mask,
                "lang_mask": language_attention_mask,
                "vis_mask": visual_temp_mask,
                "token_type_ids": token_type_ids,
                "action_feats": input_a_t,
                # 'pano_feats':         f_t,
                "cand_feats": candidate_feat,
            }
            last_h_, _ = self.vln_bert(**visual_inputs)

            rl_loss = 0.0

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic4mask(
                last_h_
            ).detach()  # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(
                batch_size, np.float32
            )  # The inital reward is zero
            for i in range(batch_size):
                if not ended[
                    i
                ]:  # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length - 1, -1, -1):
                discount_reward = (
                    discount_reward * args.gamma + rewards[t]
                )  # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                v_ = self.critic4mask(hidden_states[t])
                a_ = (r_ - v_).detach()

                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss
                if self.feedback == "sample":
                    rl_loss += (-0.01 * entropys[t] * mask_).sum()
                self.logs["critic_loss"].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs["total"].append(total)

            # Normalize the loss function
            if args.normalize_loss == "total":
                rl_loss /= total
            elif args.normalize_loss == "batch":
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == "none"

            self.loss += rl_loss
            self.logs["RL_loss"].append(rl_loss.item())

        # if train_ml is not None:
        #     self.loss += ml_loss * train_ml / batch_size
        #     self.logs["IL_loss"].append((ml_loss * train_ml / batch_size).item())
        print("total reward", self.if_succeed(perm_obs, traj))
        if (
            type(self.loss) is int
        ):  # For safety, it will be activated if no losses are added
            self.losses.append(0.0)
        else:
            self.losses.append(
                self.loss.item() / self.episode_len
            )  # This argument is useless.

        return traj

    def rollout(self, train_ml=None, train_rl=True, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        # if self.feedback == "teacher" or self.feedback == "argmax":
        #     train_rl = False

        if reset:  # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        self.instr_buffer = [
            [] for _ in range(batch_size)
        ]  # to identify which vp's index has been used

        # Language input
        sentence, language_attention_mask, token_type_ids, seq_lengths, perm_idx = (
            self._sort_batch(obs)
        )
        perm_obs = obs[perm_idx]

        """ Language BERT """
        language_inputs = {
            "mode": "language",
            "sentence": sentence,
            "attention_mask": language_attention_mask,
            "lang_mask": language_attention_mask,
            "token_type_ids": token_type_ids,
        }
        if args.vlnbert == "oscar":
            language_features = self.vln_bert(**language_inputs)
        elif args.vlnbert == "prevalent":
            h_t, language_features = self.vln_bert(**language_inputs)

        # Record starting point
        traj = [
            {
                "instr_id": ob["instr_id"],
                "path": [(ob["viewpoint"], ob["heading"], ob["elevation"])],
            }
            for ob in perm_obs
        ]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(
            perm_obs
        ):  # The init distance from the view point to the target
            last_dist[i] = ob["distance"]
            path_act = [vp[0] for vp in traj[i]["path"]]
            last_ndtw[i] = self.ndtw_criterion[ob["scan"]](
                path_act, ob["gt_path"], metric="ndtw"
            )

        # Initialization the tracking state
        ended = np.array(
            [False] * batch_size
        )  # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.0

        for t in range(self.episode_len):
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
            # the first [CLS] token, initialized by the language BERT, serves
            # as the agent's state passing through time steps
            if (t >= 1) or (args.vlnbert == "prevalent"):
                language_features = torch.cat(
                    (h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1
                )

            visual_temp_mask = (vln_utils.length2mask(candidate_leng) == 0).long()
            visual_attention_mask = torch.cat(
                (language_attention_mask, visual_temp_mask), dim=-1
            )

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            """ Visual BERT """
            visual_inputs = {
                "mode": "visual",
                "sentence": language_features,
                "attention_mask": visual_attention_mask,
                "lang_mask": language_attention_mask,
                "vis_mask": visual_temp_mask,
                "token_type_ids": token_type_ids,
                "action_feats": input_a_t,
                # 'pano_feats':         f_t,
                "cand_feats": candidate_feat,
            }
            h_t, logit = self.vln_bert(**visual_inputs)
            hidden_states.append(h_t)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = vln_utils.length2mask(candidate_leng)
            logit.masked_fill_(candidate_mask, -float("inf"))

            # Supervised training
            if self.feedback == "teacher":
                # if self.feedback == "teacher" or self.feedback == "argmax":
                # target = self._teacher_action_baseline(perm_obs, ended)
                target = self._teacher_action(perm_obs, ended)
            elif self.feedback == "sample" or self.feedback == "argmax":
                # elif self.feedback == "sample":
                target = self.DAgger_action(
                    perm_obs,
                    # vpids,
                    [perm_obs[i]["candidate"] for i in range(len(perm_obs))],
                    ended,
                    visited_masks=None,
                    traj=traj,
                )
            ml_loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == "teacher":
                a_t = target  # teacher forcing
            elif self.feedback == "argmax":
                _, a_t = logit.max(1)  # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)  # Calculate the log_prob here
                policy_log_probs.append(
                    log_probs.gather(1, a_t.unsqueeze(1))
                )  # Gather the log_prob for each batch
            elif self.feedback == "sample":
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs["entropy"].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit("Invalid feedback option")
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if (
                    next_id == (candidate_leng[i] - 1)
                    or next_id == args.ignoreid
                    or ended[i]
                ):  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]  # Perm the obs for the resu

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(perm_obs):
                    dist[i] = ob["distance"]
                    path_act = [vp[0] for vp in traj[i]["path"]]
                    ndtw_score[i] = self.ndtw_criterion[ob["scan"]](
                        path_act, ob["gt_path"], metric="ndtw"
                    )

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:  # If the action now is end
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
                            else:
                                # Action didn't change distance - give a small penalty
                                reward[i] = -0.1  # Small penalty for not moving
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            # Last action in A2C
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            language_features = torch.cat(
                (h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1
            )

            visual_temp_mask = (vln_utils.length2mask(candidate_leng) == 0).long()
            visual_attention_mask = torch.cat(
                (language_attention_mask, visual_temp_mask), dim=-1
            )

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            """ Visual BERT """
            visual_inputs = {
                "mode": "visual",
                "sentence": language_features,
                "attention_mask": visual_attention_mask,
                "lang_mask": language_attention_mask,
                "vis_mask": visual_temp_mask,
                "token_type_ids": token_type_ids,
                "action_feats": input_a_t,
                # 'pano_feats':         f_t,
                "cand_feats": candidate_feat,
            }
            last_h_, _ = self.vln_bert(**visual_inputs)

            rl_loss = 0.0

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(
                last_h_
            ).detach()  # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(
                batch_size, np.float32
            )  # The inital reward is zero
            for i in range(batch_size):
                if not ended[
                    i
                ]:  # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length - 1, -1, -1):
                discount_reward = (
                    discount_reward * args.gamma + rewards[t]
                )  # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss
                if self.feedback == "sample":
                    rl_loss += (-0.01 * entropys[t] * mask_).sum()
                self.logs["critic_loss"].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs["total"].append(total)

            # Normalize the loss function
            if args.normalize_loss == "total":
                rl_loss /= total
            elif args.normalize_loss == "batch":
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == "none"

            self.loss += rl_loss
            self.logs["RL_loss"].append(rl_loss.item())

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs["IL_loss"].append((ml_loss * train_ml / batch_size).item())

        if (
            type(self.loss) is int
        ):  # For safety, it will be activated if no losses are added
            self.losses.append(0.0)
        else:
            self.losses.append(
                self.loss.item() / self.episode_len
            )  # This argument is useless.
        # print(self.loss)
        return traj

    def test(self, use_dropout=False, feedback="argmax", allow_cheat=False, iters=None):
        """Evaluate once on each instruction in the current environment"""
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        super(Seq2SeqAgent, self).test(iters)

    def zero_grad(self):
        self.loss = 0.0
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback="teacher", **kwargs):
        if feedback == "teacher":
            self.feedback = "teacher"
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == "sample":
            self.feedback = "teacher"
            self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
            self.feedback = "sample"
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.0)

        self.vln_bert_optimizer.step()
        self.critic_optimizer.step()

    def train(self, n_iters, feedback="teacher", mask=False, **kwargs):
        """Train for a given number of iterations"""
        # 自动发现求导的错误
        # torch.autograd.set_detect_anomaly(True)
        self.feedback = feedback
        if not mask:
            self.vln_bert.train()
            self.critic.train()
        elif mask:
            self.vln_bert.eval()
            # self.vln_bert.train()
            # self.vln_bert_noneupdate.eval()
            self.critic4mask.train()
            self.critical_head.train()
        self.losses = []
        for iter in range(1, n_iters + 1):
            # print("new iteration")
            if not mask:
                self.vln_bert_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
            elif mask:
                self.critic_optimizer4mask.zero_grad()
                self.critical_head_optimizer.zero_grad()

            self.loss = 0
            if mask:
                print(iter)
                self.rollout_mask(iter=iter)
            else:
                if feedback == "teacher":
                    self.feedback = "teacher"
                    self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
                elif feedback == "sample":  # agents in IL and RL separately
                    if args.ml_weight != 0:
                        self.feedback = "teacher"
                        self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
                    self.feedback = "sample"
                    self.rollout(train_ml=None, train_rl=True, **kwargs)
                elif feedback == "surrogate":
                    # BC
                    self.feedback = "teacher"
                    self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
                    # dagger
                    self.feedback = "sample"
                    self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
                    # RL
                    self.feedback = "sample"
                    self.rollout(train_ml=None, train_rl=True, **kwargs)
                else:
                    assert False
            # with torch.autograd.detect_anomaly():
            self.loss.backward()
            if not mask:
                torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.0)

                self.vln_bert_optimizer.step()
                self.critic_optimizer.step()
            else:
                torch.nn.utils.clip_grad_norm(self.critical_head.parameters(), 40.0)

                self.critical_head_optimizer.step()
                self.critic_optimizer4mask.step()
            if iter % 10 == 0:
                if mask:
                    self.save_mask(
                        iter,
                        os.path.join(
                            # "snap", args.name, "state_dict_mask", "LAST_iter%d" % (iter)
                            "snap",
                            args.name,
                            # "ablation_nomask",
                            "state_dict",
                            "LAST_iter%d" % (iter),
                        ),
                    )
            if args.aug is None:
                print_progress(
                    iter,
                    n_iters + 1,
                    prefix="Progress:",
                    suffix="Complete",
                    bar_length=50,
                )

    def save(self, epoch, path):
        """Snapshot models"""
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}

        def create_state(name, model, optimizer):
            states[name] = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

        all_tuple = [
            ("vln_bert", self.vln_bert, self.vln_bert_optimizer),
            ("critic", self.critic, self.critic_optimizer),
        ]
        if args.train == "surrogate_gail":
            all_tuple = [
                ("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                ("critic", self.critic, self.critic_optimizer),
                ("discrim", self.discrim, self.optimizer4discrim),
            ]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def save_mask(self, epoch, path):
        """Snapshot models"""
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}

        def create_state(name, model, optimizer):
            states[name] = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

        all_tuple = [
            ("critical_head", self.critical_head, self.critical_head_optimizer),
            ("critic4mask", self.critic4mask, self.critic_optimizer4mask),
        ]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        """Loads parameters (but not training state)"""
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            if name not in states.keys():
                print(
                    "Warning: There is no module {} in the loaded checkpoint!".format(
                        name
                    )
                )
                return
            load_keys = set(states[name]["state_dict"].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]["state_dict"])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]["optimizer"])

        all_tuple = [
            ("vln_bert", self.vln_bert, self.vln_bert_optimizer),
            ("critic", self.critic, self.critic_optimizer),
        ]
        if args.ablation_pretrained:
            all_tuple = [
                # don't initialize vln_bert
                (
                    "vln_bert",
                    self.vln_bert_noneupdate,
                    self.vln_bert_noneupdate_optimizer,
                ),
                ("critic", self.critic, self.critic_optimizer),
            ]
        if args.train == "surrogate_gail":
            all_tuple = [
                ("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                ("critic", self.critic, self.critic_optimizer),
                ("discrim", self.discrim, self.optimizer4discrim),
            ]
        for param in all_tuple:
            recover_state(*param)
        # copy vln_bert to vln_bert_noneupdate
        if not args.ablation_pretrained and args.train == "mask":
            self.vln_bert_noneupdate.load_state_dict(self.vln_bert.state_dict())
            self.vln_bert_noneupdate_optimizer.load_state_dict(
                self.vln_bert_optimizer.state_dict()
            )
        return states["vln_bert"]["epoch"] - 1

    def load_mask(self, path):
        """Loads parameters (but not training state)"""
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]["state_dict"].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]["state_dict"])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]["optimizer"])

        all_tuple = [
            # ("vln_bert", self.vln_bert, self.vln_bert_optimizer),
            # ("critic", self.critic, self.critic_optimizer),
            ("critical_head", self.critical_head, self.critical_head_optimizer),
            ("critic4mask", self.critic4mask, self.critic_optimizer4mask),
        ]
        for param in all_tuple:
            recover_state(*param)
        return states["critical_head"]["epoch"] - 1

    def test_mask(self, basline=None, iters=None, **kwargs):
        self.vln_bert.eval()
        self.critic.eval()
        self.critical_head.eval()
        self.critic4mask.eval()

        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj["instr_id"]] = traj
        else:  # do a full round
            # ---------------test without mask agent's influence-----------
            self.a = 0
            self.b = 0
            tmp_rewards_baseline = []
            tmp_counts_baseline = []
            tmp_disc_rewards_baseline = []
            actions_baseline = []
            tmp_mask_probs_baseline = []
            num_mask_baseline = []
            mask_pos_baseline = []
            traj_baseline = []
            looped = False
            self.env.reset_epoch(
                shuffle=(iters is not None)
            )  # If iters is not none, shuffle the env batch
            self.results = {}
            print("--------- baseline -------------")
            while True:
                (
                    traj,
                    total_reward,
                    total_discounted_reward,
                    count,
                    num_mask,
                    mask_pos,
                    action_seq,
                    mask_probs,
                ) = self.rollout_mask_test(test_model="baseline", threshod=0.5)
                tmp_rewards_baseline.append(total_reward)
                tmp_disc_rewards_baseline.append(total_discounted_reward)
                tmp_counts_baseline.append(count)
                actions_baseline.append(action_seq)
                tmp_mask_probs_baseline.append(mask_probs)
                num_mask_baseline.append(num_mask)
                mask_pos_baseline.append(mask_pos)
                traj_baseline.append(traj)
                if traj["instr_id"] in self.results:
                    looped = True
                else:
                    self.results[traj["instr_id"]] = traj
                if looped:
                    break

            self.save_intermediate(
                args.timelevelbaseline,
                "baseline",
                {
                    "trajs": traj_baseline,
                    "rewards": tmp_rewards_baseline,
                    "disc_rewards": tmp_disc_rewards_baseline,
                    "counts": tmp_counts_baseline,
                    "actions": actions_baseline,
                    "mask_probs": tmp_mask_probs_baseline,
                    "num_masks": num_mask_baseline,
                    "mask_pos": mask_pos_baseline,
                },
            )
            # NOTE----------------------------------
            critical_steps_starts, critical_steps_ends = select_critical_steps(
                tmp_mask_probs_baseline, tmp_counts_baseline, random_zone=False
            )
            for i in range(10):
                print(critical_steps_starts[i], critical_steps_ends[i])
            # ----------------------------- replay ------------------------
            looped = False
            self.env.reset_epoch(shuffle=(iters is not None))
            self.results = {}
            traj_index = 0
            tmp_rewards_replay = []
            tmp_counts_replay = []
            tmp_disc_rewards_replay = []
            actions_replay = []
            tmp_mask_probs_replay = []
            num_mask_replay = []
            mask_pos_replay = []
            traj_replay = []
            self.a = self.b = 0
            print("--------- replay -------------")
            while True:
                replay_info = {
                    "critical_steps_starts": critical_steps_starts[traj_index],
                    "critical_steps_ends": critical_steps_ends[traj_index],
                    "recorded_actions": actions_baseline[traj_index],
                }
                (
                    traj,
                    total_reward,
                    total_discounted_reward,
                    count,
                    num_mask,
                    mask_pos,
                    action_seq,
                    mask_probs,
                ) = self.rollout_mask_test(
                    test_model="replay",
                    replay_info=replay_info,
                    threshod=0.5,
                )
                tmp_rewards_replay.append(total_reward)
                tmp_disc_rewards_replay.append(total_discounted_reward)
                tmp_counts_replay.append(count)
                actions_replay.append(action_seq)
                tmp_mask_probs_replay.append(mask_probs)
                num_mask_replay.append(num_mask)
                mask_pos_replay.append(mask_pos)
                traj_replay.append(traj)
                traj_index += 1
                if traj["instr_id"] in self.results:
                    looped = True
                else:
                    self.results[traj["instr_id"]] = traj
                if looped:
                    break

            self.save_intermediate(
                args.timelevelbaseline,
                "replay",
                {
                    "trajs": traj_replay,
                    "rewards": tmp_rewards_replay,
                    "disc_rewards": tmp_disc_rewards_replay,
                    "counts": tmp_counts_replay,
                    "actions": actions_replay,
                    "mask_probs": tmp_mask_probs_replay,
                    "num_masks": num_mask_replay,
                    "mask_pos": mask_pos_replay,
                },
            )
            fidelity_score = compute_fidelity_score(
                critical_steps_starts,
                critical_steps_ends,
                tmp_counts_baseline,
                tmp_disc_rewards_replay,
                tmp_disc_rewards_baseline,
            )
            return fidelity_score

    def rollout_mask_test(
        self,
        B_agent,
        test_model="mask",
        threshod=None,
        save_rand_prob=False,
        replay_info=None,
        reset=True,
    ):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        train_rl = True
        if test_model == "replay":
            critical_steps_start = replay_info["critical_steps_starts"]
            critical_steps_end = replay_info["critical_steps_ends"]
            recorded_actions = replay_info["recorded_actions"]
        if reset:
            obs = np.array(self.env.reset_test())
        else:
            obs = self.env._get_obs()

        batch_size = len(obs)

        total_reward, total_discounted_reward = 0, 0
        # count = 0
        num_mask_total = 0
        num_action_total = 0
        mask_pos = []
        action_seq = []
        mask_probs = []
        rewards = []

        # Language input
        sentence, language_attention_mask, token_type_ids, seq_lengths, perm_idx = (
            self._sort_batch(obs)
        )
        perm_obs = obs[perm_idx]

        """ Language BERT """
        language_inputs = {
            "mode": "language",
            "sentence": sentence,
            "attention_mask": language_attention_mask,
            "lang_mask": language_attention_mask,
            "token_type_ids": token_type_ids,
        }
        if args.vlnbert == "oscar":
            language_features = self.vln_bert(**language_inputs)
        elif args.vlnbert == "prevalent":
            h_t, language_features = self.vln_bert(**language_inputs)

        # Record starting point
        traj = [
            {
                "instr_id": ob["instr_id"],
                "path": [(ob["viewpoint"], ob["heading"], ob["elevation"])],
            }
            for ob in perm_obs
        ]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(
            perm_obs
        ):  # The init distance from the view point to the target
            last_dist[i] = ob["distance"]
            path_act = [vp[0] for vp in traj[i]["path"]]
            last_ndtw[i] = self.ndtw_criterion[ob["scan"]](
                path_act, ob["gt_path"], metric="ndtw"
            )

        # Initialization the tracking state
        ended = np.array(
            [False] * batch_size
        )  # Indices match permuation of the model, not env

        # Init the logs
        hidden_states = []
        policy_log_probs = []

        for t in range(self.episode_len):

            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            # the first [CLS] token, initialized by the language BERT, serves
            # as the agent's state passing through time steps
            if (t >= 1) or (args.vlnbert == "prevalent"):
                language_features = torch.cat(
                    (h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1
                )

            visual_temp_mask = (vln_utils.length2mask(candidate_leng) == 0).long()
            visual_attention_mask = torch.cat(
                (language_attention_mask, visual_temp_mask), dim=-1
            )

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            """ Visual BERT """
            visual_inputs = {
                "mode": "visual",
                "sentence": language_features,
                "attention_mask": visual_attention_mask,
                "lang_mask": language_attention_mask,
                "vis_mask": visual_temp_mask,
                "token_type_ids": token_type_ids,
                "action_feats": input_a_t,
                # 'pano_feats':         f_t,
                "cand_feats": candidate_feat,
            }
            h_t, logit = self.vln_bert(**visual_inputs)
            hidden_states.append(h_t)

            # # Mask outputs where agent can't move forward
            # # Here the logit is [b, max_candidate]
            candidate_mask = vln_utils.length2mask(candidate_leng)
            # logit.masked_fill_(candidate_mask, -float("inf"))

            # generate action with original policy
            do_inference = True
            if test_model == "replay" and t <= critical_steps_end:
                do_inference = False
            B_action_copy = (
                self.generate_pseudo_action(logit, candidate_mask, mode="argmax")
                .cpu()
                .numpy()
            )
            # B_action_options = state_t["nav_inputs"]["vp_cand_vpids"]

            # generate mask
            critical_logits = self.critical_head(h_t).unsqueeze(0)  # NOTE
            critical_probs = F.softmax(
                critical_logits, 1
            )  # sampling an action from model
            critical_c = torch.distributions.Categorical(critical_probs)
            critical_a_t = critical_c.sample().detach()
            mask_probs.append(critical_c.probs.detach().cpu().numpy()[0])
            rand_f = np.random.rand()
            if save_rand_prob:
                mask_probs[t][1] = rand_f

            # count the number of masks
            mask_action = 1 - critical_a_t
            num_mask = torch.sum(mask_action[~ended])
            num_action_total += torch.sum(torch.ones_like(mask_action)[~ended])
            num_mask_total += num_mask

            # determine the real action
            mask_action_copy = critical_a_t.cpu().numpy()
            real_action = []
            for i in range(batch_size):
                if test_model == "baseline":
                    mask_action_copy[i] = 1
                elif (
                    test_model == "replay"
                    and critical_steps_start <= t <= critical_steps_end
                ):  # generate action randomly
                    mask_action_copy[i] = 0
                elif (
                    test_model == "replay" and t > critical_steps_end
                ):  # follow the original policy
                    mask_action_copy[i] = 1

                if test_model == "replay" and t < critical_steps_start:
                    real_action.append(recorded_actions[t])
                elif mask_action_copy[i] == 1:
                    real_action.append(B_action_copy[i])
                else:
                    n = candidate_leng[i]
                    if n == 0:
                        # handle the case without options (adjust according to actual needs)
                        real_action.append(-1)
                    elif n == 1:
                        # when there is only 1 option, select directly
                        real_action.append(0)
                    else:
                        # generate a random index that is not equal to B_action_copy[i]
                        while True:
                            idx = np.random.choice(n)
                            if idx != B_action_copy[i]:
                                real_action.append(idx)
                                break
            action_seq.append(real_action[0])
            mask_pos.append(t)

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = real_action
            for i, next_id in enumerate(cpu_a_t):
                if (
                    next_id == (candidate_leng[i] - 1)
                    or next_id == args.ignoreid
                    or ended[i]
                ):  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]  # Perm the obs for the resu

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(perm_obs):
                    dist[i] = ob["distance"]
                    path_act = [vp[0] for vp in traj[i]["path"]]
                    ndtw_score[i] = self.ndtw_criterion[ob["scan"]](
                        path_act, ob["gt_path"], metric="ndtw"
                    )

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:  # If the action now is end
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
                            else:
                                # Action didn't change distance - give a small penalty
                                reward[i] = -0.1  # Small penalty for not moving
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score
            total_reward += reward[0]
            total_discounted_reward += np.power(self.GAE, t) * reward[0]
            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break
        # end for

        self.a += num_action_total
        self.b += num_mask_total
        return (
            traj[i],
            total_reward,
            self.if_succeed(perm_obs, traj)[0],
            t + 1,
            num_mask_total,
            mask_pos,
            action_seq,
            mask_probs,
        )

    def save_intermediate(self, dir_name, file_name, test_dict):
        the_dir = os.path.join("snap", args.name, "intermediate_test", dir_name)
        os.makedirs(the_dir, exist_ok=True)
        with open(os.path.join(the_dir, file_name + ".pkl"), "wb") as f:
            pickle.dump(
                test_dict,
                f,
            )

    def load_intermediate(self, dir_name, file_name):
        the_dir = os.path.join("snap", args.name, "intermediate_test", dir_name)
        with open(os.path.join(the_dir, file_name + ".pkl"), "rb") as f:
            data = pickle.load(f)
        return data
