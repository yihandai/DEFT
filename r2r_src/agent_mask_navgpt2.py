from typing import Tuple
from agent_mask import MaskAgent
from param import args
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import json
import time
import sys
import os

# Add NavGPT-2 path to sys.path
navgpt2_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "NavGPT_2", "map_nav_src"
)
if navgpt2_path not in sys.path:
    sys.path.insert(0, navgpt2_path)

import r2r_src.vln_utils as vln_utils

from NavGPT_2.map_nav_src.r2r.agent import GMapNavAgent
from NavGPT_2.map_nav_src.r2r.env import R2RNavBatch
from NavGPT_2.map_nav_src.utils.data import ImageFeaturesDB
from NavGPT_2.map_nav_src.models.graph_utils import GraphMap
from NavGPT_2.map_nav_src.models.ops import pad_tensors_wgrad

open = False
if open:
    CRITICAL_PROBS_GT_DIR = "./critical_probs_gt"
else:
    CRITICAL_PROBS_REPRODUCE_DIR = "./critical_probs_reproduce"


def NavGPT2_genAction(
    agent: GMapNavAgent,
    obs,
    gmaps,
    instructions,
    t,
    ended=None,
    feedback="argmax",
    nav_inputs=None,
) -> Tuple[np.ndarray, list, dict]:
    """
    Generate one action step from NavGPT-2 agent.

    Args:
        agent: NavGPT-2 agent (GMapNavAgent instance)
        obs: List of observations (batch)
        gmaps: List of GraphMap objects (maintained across time steps)
        instructions: List of instruction strings
        t: Time step
        ended: Array indicating which episodes have ended
        feedback: Action selection mode ('argmax' or 'sample')

    Returns:
        a_t: action indices (0 = stop, 1+ = action index in nav_vpids)
        nav_vpids_list: list of nav_vpids for each sample
        nav_inputs_dict: dict containing nav_inputs including 'no_vp_left' for stop determination
    """
    batch_size = len(obs)
    a_t = np.zeros(batch_size, dtype=np.int32)
    nav_vpids_list = []

    # Use agent's rollout logic to get actions
    # We'll call the agent's internal methods to get action predictions
    with torch.no_grad():

        # Get local feature variables (using existing gmaps and instructions)
        # Note: We don't update graph maps here - they should be updated after getting new obs
        # from the previous step. The graph maps represent the state before the current action.
        local_inputs = agent._local_feature_variable(obs, gmaps, instructions)

        # Forward NavGPT thoughts
        local_outputs = agent.NavGPT("thought", local_inputs)
        view_embeds = local_outputs["view_embeds"]
        instruct_text_embeds = local_outputs["instruct_text_embeds"]
        instruct_text_masks = local_outputs["instruct_text_masks"]
        local_inputs["text_embeds"] = instruct_text_embeds
        local_inputs["text_masks"] = instruct_text_masks
        local_inputs["view_llm_fts"] = view_embeds

        # Split loc_fts (pad_tensors_wgrad is imported at the top level)
        if pad_tensors_wgrad is None:
            raise ImportError(
                "pad_tensors_wgrad not available. Please ensure NavGPT-2 is properly set up."
            )

        split_loc_fts = torch.split(
            local_inputs["loc_fts"], local_inputs["view_lens"].tolist(), 0
        )
        local_inputs["loc_fts"] = pad_tensors_wgrad(split_loc_fts)

        # Get panorama embeddings
        pano_embeds, pano_masks = agent.NavGPT("panorama", local_inputs)

        # Update graph node embeddings (this modifies gmaps in place)
        avg_pano_embeds = torch.sum(
            pano_embeds * pano_masks.unsqueeze(2), 1
        ) / torch.sum(pano_masks, 1, keepdim=True)

        for i, gmap in enumerate(gmaps):
            if ended is None or not ended[i] and gmap is not None:
                i_vp = obs[i]["viewpoint"]
                gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                for j, i_cand_vp in enumerate(local_inputs["cand_vpids"][i]):
                    if not gmap.graph.visited(i_cand_vp):
                        gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

        # Navigation policy
        nav_inputs = agent._nav_gmap_variable(obs, gmaps)
        nav_inputs.update(local_inputs)

        # Add [stop] token
        nav_vp_inputs = agent._nav_vp_variable(
            pano_embeds, local_inputs["cand_vpids"], local_inputs["view_lens"]
        )
        nav_inputs.update(nav_vp_inputs)

        # Forward action prediction
        nav_outs = agent.NavGPT("action", nav_inputs)

        # Get logits based on fusion mode
        if agent.args.fusion == "local":
            nav_logits = nav_outs["local_logits"]
            nav_vpids = nav_inputs["vp_cand_vpids"]
        elif agent.args.fusion == "global":
            nav_logits = nav_outs["global_logits"]
            nav_vpids = nav_inputs["gmap_vpids"]
        else:  # fused
            nav_logits = nav_outs["fused_logits"]
            nav_vpids = nav_inputs["gmap_vpids"]

        nav_probs = torch.softmax(nav_logits, 1)

        # Update graph node stop scores (like NavGPT-2 does)
        for i, gmap in enumerate(gmaps):
            if ended is None or not ended[i]:
                if gmap is not None:
                    i_vp = obs[i]["viewpoint"]
                    gmap.node_stop_scores[i_vp] = {
                        "stop": nav_probs[i, 0].data.item(),
                    }

        # Select actions
        for i in range(batch_size):
            if ended is not None and ended[i]:
                a_t[i] = 0
                nav_vpids_list.append([None])
                continue

            nav_vpids_list.append(nav_vpids[i])

            if feedback == "argmax":
                _, a_t[i] = nav_logits[i].max(0)
                a_t[i] = a_t[i].item()
            elif feedback == "sample":
                c = torch.distributions.Categorical(nav_probs[i])
                a_t[i] = c.sample().item()
            else:
                _, a_t[i] = nav_logits[i].max(0)
                a_t[i] = a_t[i].item()

    # Return nav_inputs for stop determination (includes 'no_vp_left')
    nav_inputs_dict = {
        "no_vp_left": nav_inputs.get("no_vp_left", [False] * batch_size),
        "nav_vpids": nav_vpids,
    }

    return a_t, nav_vpids_list, nav_inputs_dict


class MaskAgent_NavGPT2(MaskAgent):
    def __init__(self, env, results_path, tok, episode_len=20, args_target=None):
        super(MaskAgent_NavGPT2, self).__init__(env, results_path, tok, episode_len)
        rank = 0

        if args_target is None:
            raise ValueError("args_target must be provided for NavGPT-2 agent")

        # Initialize NavGPT-2 agent
        # Note: self.target_agent.scanvp_cands is initialized to {} in _build_model()
        self.target_agent = GMapNavAgent(args_target, env, rank=rank)

        # Load checkpoint if specified
        if hasattr(args_target, "resume_file") and args_target.resume_file is not None:
            self.target_agent.load(args_target.resume_file)

    def rollout_mask(self, train_ml=None, train_rl=True, reset=True, iter=0):
        """
        Main rollout function that routes to NavGPT-2 specific implementation.
        """
        return self.rollout_mask_navgpt2(
            train_ml=train_ml, train_rl=train_rl, reset=reset
        )

    def rollout_mask_navgpt2(self, train_ml=None, train_rl=True, reset=True):
        """
        Rollout with mask logic for NavGPT-2.

        :param train_ml: The weight to train with maximum likelihood
        :param train_rl: whether use RL in training
        :param reset: Reset the environment
        :return: trajectory
        """
        # NOTE: NavGPT-2 中 0 代表 stop
        # NOTE: RecVLN 中 len(candidate) - 1 代表 stop
        train_rl = True
        mask_weight = 1
        # mask_weight = 0
        if reset:
            obs = np.array(self.env.reset())
            # Synchronize target agent environment
            # target_obs = self._sync_target_env(obs)
            target_obs = np.array(
                self.target_agent.env.set_scan_viewpoint_heading(
                    self.env.get_scan_viewpoint_heading()
                )
            )
        else:
            obs = np.array(self.env._get_obs())
            # target_obs = self._sync_target_env(obs)
            target_obs = np.array(
                self.target_agent.env.set_scan_viewpoint_heading(
                    self.env.get_scan_viewpoint_heading()
                )
            )

        batch_size = len(obs)

        # Language input for VLN-BERT
        sentence, language_attention_mask, token_type_ids, seq_lengths, perm_idx = (
            self._sort_batch(obs)
        )
        perm_obs = obs[perm_idx]
        target_perm_obs = target_obs[perm_idx]

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

        # Init reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):
            last_dist[i] = ob["distance"]
            path_act = [vp[0] for vp in traj[i]["path"]]
            last_ndtw[i] = self.ndtw_criterion[ob["scan"]](
                path_act, ob["gt_path"], metric="ndtw"
            )

        # Initialization tracking state
        ended = np.array([False] * batch_size)

        # Init logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        num_masks = []
        ml_loss = 0.0

        # Target agent (NavGPT-2) initialization
        target_traj = [
            {
                "instr_id": ob["instr_id"],
                "path": [[ob["viewpoint"]]],
                "details": {},
                "a_t": {},
            }
            for ob in target_perm_obs
        ]
        target_ended = np.array([False] * batch_size)
        target_just_ended = np.array([False] * batch_size)

        # Initialize NavGPT-2 state variables (similar to NavGPT-2's rollout)
        # GraphMap is imported at the top level
        if GraphMap is None:
            raise ImportError(
                "GraphMap not available. Please ensure NavGPT-2 is properly set up."
            )

        # Initialize graph maps at the start of rollout
        target_gmaps = [GraphMap(ob["viewpoint"]) for ob in target_perm_obs]
        for i, ob in enumerate(target_perm_obs):
            target_gmaps[i].update_graph(ob)

        # Initialize instructions (extract once at the start)
        target_instructions = [ob["instruction"] for ob in target_perm_obs]

        # Update scanvp_cands (NavGPT-2 agent's internal state)
        self.target_agent._update_scanvp_cands(target_perm_obs)

        # Main loop
        for t in range(self.episode_len):
            print("t: ", t)
            print(
                "viewpoint: ",
                [target_perm_obs[i]["viewpoint"] for i in range(batch_size)],
            )
            # Update graph step ids (like NavGPT-2 does)
            for i, gmap in enumerate(target_gmaps):
                if not target_ended[i]:
                    gmap.node_step_ids[target_perm_obs[i]["viewpoint"]] = t + 1

            # Generate target agent (NavGPT-2) action (pass existing gmaps and instructions)
            target_action, target_nav_vpids, nav_inputs_dict = NavGPT2_genAction(
                self.target_agent,
                target_perm_obs,
                target_gmaps,
                target_instructions,
                t,
                ended=target_ended,
                feedback="argmax",
            )
            # target_action 是对应 NavGPT-2 的 nav_vpids 的 index
            # 其中 0 代表 stop

            # Get input features for VLN-BERT
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            # Generate mask action using VLN-BERT
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
                "cand_feats": candidate_feat,
            }
            h_t, logit = self.vln_bert(**visual_inputs)
            hidden_states.append(h_t)

            # Generate mask using critical_head
            critical_logits = self.critical_head(h_t)
            critical_probs = F.softmax(critical_logits, 1)
            critical_c = torch.distributions.Categorical(critical_probs)
            critical_a_t = critical_c.sample().detach()

            # Statistics
            mask_action = 1 - critical_a_t
            num_mask = torch.sum(mask_action[~ended])
            num_masks.append(num_mask)
            policy_log_probs.append(critical_c.log_prob(critical_a_t))

            self.logs["entropy"].append(critical_c.entropy().sum().item())
            entropys.append(critical_c.entropy())

            # Determine real action based on mask
            mask_action_copy = critical_a_t.cpu().numpy()
            real_action = []
            for i in range(batch_size):
                print("candidate_leng: ", candidate_leng)
                print("len(target_nav_vpids[i]): ", len(target_nav_vpids[i]))
                if mask_action_copy[i] == 1:
                    real_action.append(target_action[i])
                else:
                    # Use random action (mask = 0 means non-critical step)
                    # Select from intersection of NavGPT-2 and RecVLN action spaces
                    nav_vpids_i = (
                        target_nav_vpids[i] if len(target_nav_vpids) > i else []
                    )
                    random_action = self._random_action_from_intersection(
                        perm_obs[i],
                        nav_vpids_i,
                        candidate_leng[i],
                        exclude_action=target_action[i],
                        return_recvln_index=False,
                    )
                    real_action.append(random_action)

            # Update target agent trajectory
            for i in range(batch_size):
                # target_traj[i]["a_t"][t] = target_action[i]
                target_traj[i]["a_t"][t] = real_action[i]

            # Determine stop actions for NavGPT-2 (like NavGPT-2 does)
            # target_a_t_stop = [a_t_i == 0 for a_t_i in target_action]
            target_a_t_stop = [a_t_i == 0 for a_t_i in real_action]

            # Prepare environment action for NavGPT-2 (like NavGPT-2 does)
            target_cpu_a_t = []
            no_vp_left = nav_inputs_dict.get("no_vp_left", [False] * batch_size)
            for i in range(batch_size):
                if (
                    target_a_t_stop[i]
                    or target_ended[i]
                    or no_vp_left[i]
                    or (t == self.episode_len - 1)
                ):
                    target_cpu_a_t.append(None)  # Stop action for NavGPT-2
                    target_just_ended[i] = True
                else:
                    # Get viewpoint ID from nav_vpids
                    # if target_action[i] < len(target_nav_vpids[i]):
                    #     target_cpu_a_t.append(target_nav_vpids[i][target_action[i]])
                    if real_action[i] < len(target_nav_vpids[i]):
                        target_vp = target_nav_vpids[i][real_action[i]]
                        # Check if action is the same as current viewpoint (should be treated as stop)
                        if (
                            target_vp is None
                            or target_vp == target_perm_obs[i]["viewpoint"]
                        ):
                            target_cpu_a_t.append(None)
                            target_just_ended[i] = True
                        else:
                            target_cpu_a_t.append(target_vp)
                    else:
                        target_cpu_a_t.append(None)

            # Make action in NavGPT-2 environment (using maintained gmaps)
            self._make_navgpt2_action(
                target_cpu_a_t, target_perm_obs, target_traj, target_gmaps, perm_idx
            )

            # Handle stop node selection for just_ended episodes (like NavGPT-2 does)
            for i in range(batch_size):
                if (not target_ended[i]) and target_just_ended[i]:
                    stop_node, stop_score = None, {"stop": -float("inf")}
                    for k, v in target_gmaps[i].node_stop_scores.items():
                        if v["stop"] > stop_score["stop"]:
                            stop_score = v
                            stop_node = k
                    if (
                        stop_node is not None
                        and target_perm_obs[i]["viewpoint"] != stop_node
                    ):
                        target_traj[i]["path"].append(
                            target_gmaps[i].graph.path(
                                target_perm_obs[i]["viewpoint"], stop_node
                            )
                        )

            # Get new observations after action from NavGPT-2 environment
            # This is critical: we need to get obs from NavGPT-2's environment after the action
            target_obs_new = self.target_agent.env._get_obs()
            target_perm_obs = np.array(target_obs_new)[perm_idx]

            # Update scanvp_cands with new observations (like NavGPT-2 does at each step)
            # This must be done after getting new obs from NavGPT-2 environment
            self.target_agent._update_scanvp_cands(target_perm_obs)

            # Update graph maps with new observations
            for i, ob in enumerate(target_perm_obs):
                if not target_ended[i]:
                    target_gmaps[i].update_graph(ob)

            # Convert actions and handle direct moves for alignment
            real_action_surr = []
            need_direct_move = [False] * batch_size
            direct_move_targets = [None] * batch_size

            for i in range(batch_size):
                action_result = self._convert_navgpt2_to_recvln_action(
                    real_action[i],
                    target_nav_vpids[i] if len(target_nav_vpids) > i else [],
                    perm_obs[i],
                    candidate_leng[i],
                    return_target_info=True,
                )
                if isinstance(action_result, tuple):
                    action_idx, target_vpid = action_result
                    if action_idx == -2:  # Need direct move
                        need_direct_move[i] = True
                        direct_move_targets[i] = target_vpid
                        real_action_surr.append(-1)  # Will be handled separately
                    else:
                        real_action_surr.append(action_idx)
                else:
                    real_action_surr.append(action_result)

            # Handle direct moves for alignment
            has_direct_move = any(need_direct_move) and not all(ended)
            if has_direct_move:
                # Get NavGPT-2 environment state
                # navgpt2_location = self.target_agent.env.get_scan_viewpoint_heading()
                navgpt2_location = {
                    "scanIds": [x["scan"] for x in target_obs_new],
                    "viewpointIds": [x["viewpoint"] for x in target_obs_new],
                    "headings": [x["heading"] for x in target_obs_new],
                    "instr_ids": [x["instr_id"] for x in target_obs_new],
                    "batch": [None for x in target_obs_new],
                }
                print("navgpt2_location: ", navgpt2_location["scanIds"])
                print("navgpt2_location: ", navgpt2_location["viewpointIds"])
                # Create location tuple for VLN-BERT environment with updated viewpoints
                for i in range(batch_size):
                    if need_direct_move[i] and not ended[i]:
                        target_vpid = direct_move_targets[i]
                        if target_vpid is not None:
                            # Update viewpoint ID to target
                            # vlnbert_location["viewpointIds"][i] = target_vpid
                            print(
                                f"Directly moving to viewpoint {target_vpid} for alignment"
                            )

                # Set VLN-BERT environment to NavGPT-2's position
                obs_after_move = self.env.set_scan_viewpoint_heading(navgpt2_location)
                obs = np.array(obs_after_move)
                perm_obs = obs[perm_idx]

                # Update trajectory
                for i in range(batch_size):
                    if need_direct_move[i] and not ended[i] and traj is not None:
                        ob = perm_obs[i]
                        traj[i]["path"].append(
                            (ob["viewpoint"], ob["heading"], ob["elevation"])
                        )

            # Prepare environment action for RecVLN
            # Only process actions for samples that didn't need direct move
            else:
                cpu_a_t = np.array(real_action_surr)
                for i, next_id in enumerate(cpu_a_t):
                    if (
                        next_id == (candidate_leng[i] - 1)
                        or next_id == args.ignoreid
                        # or next_id == -2  # Direct move needed, treat as stop
                        or ended[i]
                    ):
                        cpu_a_t[i] = -1  # Change stop action to -1

                # Make action in RecVLN environment (only for samples that didn't need direct move)
                # make_equiv_action will skip -1 actions, but we still need to call it for other samples
                # Only call make_equiv_action if there are samples that didn't need direct move
                self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)

                # Only update obs if we didn't already update it from direct move
                obs = np.array(self.env._get_obs())
                perm_obs = obs[perm_idx]

            if train_rl:
                # Calculate reward
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
                        if action_idx == -1:  # Stop action
                            if dist[i] < 3.0:  # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:  # Incorrect
                                reward[i] = -2.0
                        else:  # Moving action
                            reward[i] = -(dist[i] - last_dist[i])
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:
                                reward[i] = 1.0 + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward
                            # else:
                            #     raise NameError("The action doesn't change the move")
                            # Miss target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0

                reward += (
                    mask_weight * mask_action.cpu().numpy()
                )  # 把掩码添加到奖励中，掩码越多越好
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            # Update finished actions
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))
            target_ended[:] = np.logical_or(
                target_ended, (np.array([x is None for x in target_cpu_a_t]))
            )

            # Early exit if all ended
            if ended.all():
                break

        # Calculate RL loss if needed
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
            visual_inputs = {
                "mode": "visual",
                "sentence": language_features,
                "attention_mask": visual_attention_mask,
                "lang_mask": language_attention_mask,
                "vis_mask": visual_temp_mask,
                "token_type_ids": token_type_ids,
                "action_feats": input_a_t,
                "cand_feats": candidate_feat,
            }
            last_h_, _ = self.vln_bert(**visual_inputs)

            rl_loss = 0.0

            # A2C
            last_value__ = self.critic4mask(last_h_).detach()
            discount_reward = np.zeros(batch_size, np.float32)
            for i in range(batch_size):
                if not ended[i]:
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length - 1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[t]
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                v_ = self.critic4mask(hidden_states[t])
                a_ = (r_ - v_).detach()

                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5
                if self.feedback == "sample":
                    rl_loss += (-0.01 * entropys[t] * mask_).sum()
                self.logs["critic_loss"].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs["total"].append(total)

            # Normalize loss
            if args.normalize_loss == "total":
                rl_loss /= total
            elif args.normalize_loss == "batch":
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == "none"

            self.loss += rl_loss
            self.logs["RL_loss"].append(rl_loss.item())

        print("total reward", self.if_succeed(perm_obs, traj))
        if type(self.loss) is int:
            self.losses.append(0.0)
        else:
            self.losses.append(self.loss.item() / self.episode_len)

        return traj

    def _convert_navgpt2_to_recvln_action(
        self,
        navgpt2_action,
        nav_vpids,
        recvln_ob,
        candidate_leng,
        return_target_info=False,
    ):
        """
        Convert NavGPT-2 action to RecVLN action space.

        Args:
            navgpt2_action: Action index in NavGPT-2 (0 = stop, 1+ = action index)
            nav_vpids: List of viewpoint IDs from NavGPT-2 (includes [None] for stop)
            recvln_ob: RecVLN observation
            candidate_leng: Number of candidates in RecVLN
            return_target_info: If True, return tuple (action_index, target_vpid) for direct move

        Returns:
            If return_target_info=False:
                recvln_action: Action index in RecVLN (0 to candidate_leng-2 for actions, candidate_leng-1 for stop, -2 for direct move needed)
            If return_target_info=True:
                (recvln_action, target_vpid): Tuple with action index and target viewpoint ID
        """
        if navgpt2_action == 0:
            # Stop action
            if return_target_info:
                return (candidate_leng - 1, None)
            return candidate_leng - 1

        # Get viewpoint ID from NavGPT-2
        print("navgpt2_action: ", navgpt2_action)
        print("len(nav_vpids): ", len(nav_vpids))
        print("nav_vpids: ", nav_vpids)
        # Check bounds: navgpt2_action must be non-negative and within nav_vpids range
        # Also handle ignoreid case (typically -100)
        if navgpt2_action == args.ignoreid or navgpt2_action < 0:
            # Invalid action (ignoreid or negative), return stop
            print(
                f"Invalid action (ignoreid or negative): {navgpt2_action}, return stop"
            )
            if return_target_info:
                return (candidate_leng - 1, None)
            return candidate_leng - 1
        elif navgpt2_action < len(nav_vpids):
            target_vpid = nav_vpids[navgpt2_action]
            if target_vpid is None:
                if return_target_info:
                    return (candidate_leng - 1, None)
                return candidate_leng - 1  # stop

            # Find corresponding candidate index in RecVLN
            recvln_candidates = recvln_ob.get("candidate", [])
            candidate_list = [x["viewpointId"] for x in recvln_candidates]

            # Find viewpoint ID in candidates
            for idx, cand in enumerate(candidate_list):
                if cand == target_vpid:
                    if return_target_info:
                        return (idx, target_vpid)
                    return idx

            # If not found, return special value for direct move
            print(f"didn't find action {target_vpid} in candidates, will move directly")
            if return_target_info:
                return (-2, target_vpid)  # -2 indicates need direct move
            return -2  # Special value indicating need for direct move
        else:
            # Invalid action, return stop
            print("Invalid action, return stop")
            if return_target_info:
                return (candidate_leng - 1, None)
            return candidate_leng - 1

    def _convert_recvln_to_navgpt2_action(
        self, recvln_action, nav_vpids, recvln_ob, candidate_leng
    ):
        """
        Convert RecVLN (VLN-BERT) action to NavGPT-2 action space.

        Args:
            recvln_action: Action index in RecVLN (0 to candidate_leng-2 for actions, candidate_leng-1 for stop)
            nav_vpids: List of viewpoint IDs from NavGPT-2 (includes [None] for stop)
            recvln_ob: RecVLN observation
            candidate_leng: Number of candidates in RecVLN

        Returns:
            navgpt2_action: Action index in NavGPT-2 (0 = stop, 1+ = action index in nav_vpids)
        """
        # Check if stop action in RecVLN
        if recvln_action == candidate_leng - 1:
            # Stop action
            return 0

        # Get viewpoint ID from RecVLN candidates
        recvln_candidates = recvln_ob.get("candidate", [])
        if recvln_action >= len(recvln_candidates):
            # Invalid action, return stop
            print("Invalid recvln_action index, return stop")
            return 0

        target_vpid = recvln_candidates[recvln_action]["viewpointId"]

        # Find corresponding index in nav_vpids
        # Note: nav_vpids[0] is None (stop), so valid actions start from index 1
        for idx in range(1, len(nav_vpids)):
            if nav_vpids[idx] == target_vpid:
                return idx

        # If not found in nav_vpids, return stop
        print(f"Viewpoint ID {target_vpid} not found in nav_vpids, return stop")
        print("nav_vpids: ", nav_vpids)
        return 0

    def _random_action_from_intersection(
        self,
        recvln_ob,
        nav_vpids,
        candidate_leng,
        exclude_action=None,
        return_recvln_index=True,
    ):
        """
        Randomly select an action from the intersection of NavGPT-2 and RecVLN action spaces.

        Args:
            recvln_ob: RecVLN observation
            nav_vpids: List of viewpoint IDs from NavGPT-2 (includes [None] for stop)
            candidate_leng: Number of candidates in RecVLN
            exclude_action: Action index to exclude from selection (in RecVLN space if return_recvln_index=True, else in NavGPT-2 space)
            return_recvln_index: If True, return RecVLN action index; if False, return NavGPT-2 action index

        Returns:
            action_index: Action index in the specified space (RecVLN or NavGPT-2)
                         - RecVLN: 0 to candidate_leng-2 for actions, candidate_leng-1 for stop
                         - NavGPT-2: 0 for stop, 1+ for action index in nav_vpids
        """
        n = candidate_leng
        if n == 0:
            if return_recvln_index:
                return candidate_leng - 1  # stop in RecVLN
            else:
                return 0  # stop in NavGPT-2
        elif n == 1:
            if return_recvln_index:
                return candidate_leng - 1  # stop in RecVLN
            else:
                return 0  # stop in NavGPT-2

        # Find intersection of action spaces
        # Get RecVLN candidate viewpoint IDs
        recvln_candidates = recvln_ob.get("candidate", [])
        recvln_vpids = [cand["viewpointId"] for cand in recvln_candidates]

        # Get NavGPT-2 nav_vpids (excluding stop at index 0)
        navgpt2_vpids = set()
        if nav_vpids:
            # Skip index 0 (stop), valid actions start from index 1
            for idx in range(1, len(nav_vpids)):
                if nav_vpids[idx] is not None:
                    navgpt2_vpids.add(nav_vpids[idx])

        # Find intersection: viewpoint IDs in both spaces
        intersection_vpids = navgpt2_vpids.intersection(set(recvln_vpids))

        # Get RecVLN action indices for intersection
        intersection_recvln_actions = []
        for idx, vpid in enumerate(recvln_vpids):
            if vpid in intersection_vpids:
                intersection_recvln_actions.append(idx)

        # If intersection is empty, return stop
        if len(intersection_recvln_actions) == 0:
            if return_recvln_index:
                return candidate_leng - 1  # stop in RecVLN
            else:
                return 0  # stop in NavGPT-2

        # Filter out excluded action from intersection
        if exclude_action is not None:
            if return_recvln_index:
                # exclude_action is in RecVLN space
                available_recvln_actions = [
                    idx for idx in intersection_recvln_actions if idx != exclude_action
                ]
            else:
                # exclude_action is in NavGPT-2 space, need to convert to RecVLN first
                exclude_vpid = (
                    nav_vpids[exclude_action]
                    if exclude_action < len(nav_vpids)
                    else None
                )
                available_recvln_actions = [
                    idx
                    for idx in intersection_recvln_actions
                    if recvln_vpids[idx] != exclude_vpid
                ]
        else:
            available_recvln_actions = intersection_recvln_actions

        if len(available_recvln_actions) == 0:
            # All intersection actions are excluded, return stop
            if return_recvln_index:
                return candidate_leng - 1  # stop in RecVLN
            else:
                return 0  # stop in NavGPT-2

        # Random action from intersection
        selected_recvln_idx = np.random.choice(available_recvln_actions)

        if return_recvln_index:
            return selected_recvln_idx
        else:
            # Convert to NavGPT-2 action index
            selected_vpid = recvln_vpids[selected_recvln_idx]
            for idx in range(1, len(nav_vpids)):
                if nav_vpids[idx] == selected_vpid:
                    return idx
            # If not found, return stop
            return 0

    def _make_navgpt2_action(
        self, cpu_a_t, target_perm_obs, target_traj, gmaps, perm_idx
    ):
        """
        Make action in NavGPT-2 environment.

        Args:
            cpu_a_t: List of viewpoint IDs or None for stop action
            target_perm_obs: Target observations (NavGPT-2 format)
            target_traj: Target trajectories
            gmaps: List of GraphMap objects (maintained across time steps)
            perm_idx: Permutation indices
        """
        # Call NavGPT-2's make_equiv_action with maintained gmaps
        # This updates the environment state
        self.target_agent.make_equiv_action(
            cpu_a_t, gmaps, target_perm_obs, target_traj, perm_idx=perm_idx
        )

    def rollout_mask_test(
        self,
        test_model="mask",
        threshod=None,
        save_rand_prob=False,
        replay_info=None,
        reset=True,
    ):
        """
        Main test routing function for NavGPT-2.
        """
        if (
            "ours" in args.timelevelbaseline
            or "random" in args.timelevelbaseline
            or "ablation" in args.timelevelbaseline
        ):
            return self.rollout_mask_test_navgpt2(
                test_model=test_model,
                threshod=threshod,
                save_rand_prob=save_rand_prob,
                replay_info=replay_info,
                reset=reset,
            )
        elif "gradient" in args.timelevelbaseline:
            return self.rollout_mask_test_navgpt2_gradient(
                test_model=test_model,
                threshod=threshod,
                save_rand_prob=save_rand_prob,
                replay_info=replay_info,
                reset=reset,
            )
        elif "value-based" in args.timelevelbaseline:
            return self.rollout_mask_test_navgpt2_value_based(
                test_model=test_model,
                threshod=threshod,
                save_rand_prob=save_rand_prob,
                replay_info=replay_info,
                reset=reset,
            )

    def rollout_mask_test_navgpt2(
        self,
        test_model="mask",
        threshod=None,
        save_rand_prob=False,
        replay_info=None,
        reset=True,
    ):
        """
        Test rollout with mask logic for NavGPT-2.

        :param test_model: Test mode ("mask", "baseline", "replay", "random_baseline")
        :param threshod: Threshold for random_baseline mode
        :param save_rand_prob: Whether to save random probabilities
        :param replay_info: Information for replay mode
        :param reset: Reset the environment
        :return: trajectory, total_reward, success, steps, num_mask_total, mask_pos, action_seq, mask_probs
        """
        train_rl = True
        if test_model == "replay":
            critical_steps_start = replay_info["critical_steps_starts"]
            critical_steps_end = replay_info["critical_steps_ends"]
            recorded_actions = replay_info["recorded_actions"]

        if reset:
            obs = np.array(self.env.reset_test())
            target_obs = np.array(
                self.target_agent.env.set_scan_viewpoint_heading(
                    self.env.get_scan_viewpoint_heading()
                )
            )
        else:
            obs = np.array(self.env._get_obs())
            target_obs = np.array(
                self.target_agent.env.set_scan_viewpoint_heading(
                    self.env.get_scan_viewpoint_heading()
                )
            )

        batch_size = len(obs)
        self.instr_buffer = [[] for _ in range(batch_size)]

        total_reward, total_discounted_reward = 0, 0
        num_mask_total = 0
        num_action_total = 0
        mask_pos = []
        action_seq = []
        mask_probs = []
        rewards = []
        _vp_critical_probs = {}
        # Language input
        sentence, language_attention_mask, token_type_ids, seq_lengths, perm_idx = (
            self._sort_batch(obs)
        )
        perm_obs = obs[perm_idx]
        target_perm_obs = target_obs[perm_idx]

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
        for i, ob in enumerate(perm_obs):
            last_dist[i] = ob["distance"]
            path_act = [vp[0] for vp in traj[i]["path"]]
            last_ndtw[i] = self.ndtw_criterion[ob["scan"]](
                path_act, ob["gt_path"], metric="ndtw"
            )

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        hidden_states = []

        # Target agent (NavGPT-2) initialization
        target_traj = [
            {
                "instr_id": ob["instr_id"],
                "path": [[ob["viewpoint"]]],
                "details": {},
                "a_t": {},
            }
            for ob in target_perm_obs
        ]
        target_ended = np.array([False] * batch_size)
        target_just_ended = np.array([False] * batch_size)

        # Initialize NavGPT-2 state variables
        if GraphMap is None:
            raise ImportError(
                "GraphMap not available. Please ensure NavGPT-2 is properly set up."
            )

        target_gmaps = [GraphMap(ob["viewpoint"]) for ob in target_perm_obs]
        for i, ob in enumerate(target_perm_obs):
            target_gmaps[i].update_graph(ob)

        target_instructions = [ob["instruction"] for ob in target_perm_obs]
        self.target_agent._update_scanvp_cands(target_perm_obs)
        _instr_id = target_perm_obs[0]["instr_id"]

        # Main loop
        for t in range(self.episode_len):
            _current_vp = target_perm_obs[0]["viewpoint"]
            # Update graph step ids
            for i, gmap in enumerate(target_gmaps):
                if not target_ended[i]:
                    gmap.node_step_ids[target_perm_obs[i]["viewpoint"]] = t + 1

            # Get input features for VLN-BERT
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            # Generate mask action using VLN-BERT
            if (t >= 1) or (args.vlnbert == "prevalent"):
                language_features = torch.cat(
                    (h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1
                )

            visual_temp_mask = (vln_utils.length2mask(candidate_leng) == 0).long()
            visual_attention_mask = torch.cat(
                (language_attention_mask, visual_temp_mask), dim=-1
            )

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            visual_inputs = {
                "mode": "visual",
                "sentence": language_features,
                "attention_mask": visual_attention_mask,
                "lang_mask": language_attention_mask,
                "vis_mask": visual_temp_mask,
                "token_type_ids": token_type_ids,
                "action_feats": input_a_t,
                "cand_feats": candidate_feat,
            }
            h_t, logit = self.vln_bert(**visual_inputs)
            hidden_states.append(h_t)

            # Generate mask using critical_head
            critical_logits = self.critical_head(h_t).unsqueeze(0)
            critical_probs = F.softmax(critical_logits, 1)
            critical_c = torch.distributions.Categorical(critical_probs)
            critical_a_t = critical_c.sample().detach()
            mask_probs.append(critical_c.probs.detach().cpu().numpy()[0])
            rand_f = np.random.rand()
            if save_rand_prob:
                mask_probs[t][1] = rand_f

            _critical_probs = critical_probs.clone().detach().cpu().numpy()[0]
            _vp_critical_probs[t] = {
                "viewpoint": _current_vp,
                "critical_probs": _critical_probs,
            }

            # Statistics
            mask_action = 1 - critical_a_t
            num_mask = torch.sum(mask_action[~ended])
            num_action_total += torch.sum(torch.ones_like(mask_action)[~ended])
            num_mask_total += num_mask

            # Generate target agent action
            if test_model == "baseline" or (
                test_model == "replay" and t < critical_steps_start
            ):
                do_inference_ = False
            else:
                do_inference_ = True

            target_action, target_nav_vpids, nav_inputs_dict = NavGPT2_genAction(
                self.target_agent,
                target_perm_obs,
                target_gmaps,
                target_instructions,
                t,
                ended=target_ended,
                feedback="argmax",
            )

            if self.target_agent is not None and do_inference_:
                pass
            elif test_model == "baseline":
                target_action = self._teacher_action_baseline_navgpt2(
                    target_perm_obs, target_ended, target_nav_vpids
                )
                target_action = target_action.cpu().numpy()
            # else:
            #     target_action = np.zeros(batch_size, dtype=np.int32)

            # Determine real action based on mask
            mask_action_copy = critical_a_t.cpu().numpy()
            real_action = []
            for i in range(batch_size):
                if test_model == "baseline":
                    mask_action_copy[i] = 1
                elif (
                    test_model == "replay"
                    and critical_steps_start <= t <= critical_steps_end
                ):
                    mask_action_copy[i] = 0
                elif test_model == "replay" and t > critical_steps_end:
                    mask_action_copy[i] = 1
                elif test_model == "random_baseline":
                    if rand_f < threshod:
                        mask_action_copy[i] = 1
                    else:
                        mask_action_copy[i] = 0

                # Determine final action
                if test_model == "replay" and t < critical_steps_start:
                    real_action.append(recorded_actions[t])
                elif mask_action_copy[i] == 1:
                    real_action.append(target_action[i])
                else:
                    # Use random action from intersection of NavGPT-2 and RecVLN action spaces
                    nav_vpids_i = (
                        target_nav_vpids[i] if len(target_nav_vpids) > i else []
                    )
                    random_action = self._random_action_from_intersection(
                        perm_obs[i],
                        nav_vpids_i,
                        candidate_leng[i],
                        exclude_action=target_action[i],
                        return_recvln_index=False,
                    )
                    real_action.append(random_action)

            action_seq.append(real_action[0])
            mask_pos.append(t)

            # Update target agent trajectory
            for i in range(batch_size):
                target_traj[i]["a_t"][t] = real_action[i]

            # Determine stop actions for NavGPT-2
            target_a_t_stop = [a_t_i == 0 for a_t_i in real_action]

            # Prepare environment action for NavGPT-2
            target_cpu_a_t = []
            no_vp_left = nav_inputs_dict.get("no_vp_left", [False] * batch_size)
            for i in range(batch_size):
                if (
                    target_a_t_stop[i]
                    or target_ended[i]
                    or no_vp_left[i]
                    or (t == self.episode_len - 1)
                ):
                    target_cpu_a_t.append(None)
                    target_just_ended[i] = True
                else:
                    if (
                        real_action[i] < len(target_nav_vpids[i])
                        if len(target_nav_vpids) > i
                        else False
                    ):
                        target_vp = target_nav_vpids[i][real_action[i]]
                        # Check if action is the same as current viewpoint (should be treated as stop)
                        if (
                            target_vp is None
                            or target_vp == target_perm_obs[i]["viewpoint"]
                        ):
                            target_cpu_a_t.append(None)
                            target_just_ended[i] = True
                        else:
                            target_cpu_a_t.append(target_vp)
                    else:
                        target_cpu_a_t.append(None)

            # Make action in NavGPT-2 environment
            self._make_navgpt2_action(
                target_cpu_a_t, target_perm_obs, target_traj, target_gmaps, perm_idx
            )

            # Handle stop node selection
            for i in range(batch_size):
                if (not target_ended[i]) and target_just_ended[i]:
                    stop_node, stop_score = None, {"stop": -float("inf")}
                    for k, v in target_gmaps[i].node_stop_scores.items():
                        if v["stop"] > stop_score["stop"]:
                            stop_score = v
                            stop_node = k
                    if (
                        stop_node is not None
                        and target_perm_obs[i]["viewpoint"] != stop_node
                    ):
                        target_traj[i]["path"].append(
                            target_gmaps[i].graph.path(
                                target_perm_obs[i]["viewpoint"], stop_node
                            )
                        )

            # Get new observations
            target_obs_new = self.target_agent.env._get_obs()
            target_perm_obs = np.array(target_obs_new)[perm_idx]
            self.target_agent._update_scanvp_cands(target_perm_obs)

            # Update graph maps
            for i, ob in enumerate(target_perm_obs):
                if not target_ended[i]:
                    target_gmaps[i].update_graph(ob)

            # Convert actions for RecVLN
            real_action_surr = []
            need_direct_move = [False] * batch_size
            direct_move_targets = [None] * batch_size

            for i in range(batch_size):
                print("target_nav_vpids[i]: ", target_nav_vpids[i])
                action_result = self._convert_navgpt2_to_recvln_action(
                    real_action[i],
                    target_nav_vpids[i] if len(target_nav_vpids) > i else [],
                    perm_obs[i],
                    candidate_leng[i],
                    return_target_info=True,
                )
                if isinstance(action_result, tuple):
                    action_idx, target_vpid = action_result
                    if action_idx == -2:  # Need direct move
                        need_direct_move[i] = True
                        direct_move_targets[i] = target_vpid
                        real_action_surr.append(-1)  # Will be handled separately
                    else:
                        real_action_surr.append(action_idx)
                else:
                    real_action_surr.append(action_result)

            # Handle direct moves for alignment
            has_direct_move = any(need_direct_move) and not all(ended)
            if has_direct_move:
                # Get NavGPT-2 environment state
                # navgpt2_location = self.target_agent.env.get_scan_viewpoint_heading()
                navgpt2_location = {
                    "scanIds": [x["scan"] for x in target_obs_new],
                    "viewpointIds": [x["viewpoint"] for x in target_obs_new],
                    "headings": [x["heading"] for x in target_obs_new],
                    "instr_ids": [x["instr_id"] for x in target_obs_new],
                    "batch": [None for x in target_obs_new],
                }
                print("navgpt2_location: ", navgpt2_location["scanIds"])
                print("navgpt2_location: ", navgpt2_location["viewpointIds"])
                # Create location tuple for VLN-BERT environment with updated viewpoints
                for i in range(batch_size):
                    if need_direct_move[i] and not ended[i]:
                        target_vpid = direct_move_targets[i]
                        if target_vpid is not None:
                            # Update viewpoint ID to target
                            # vlnbert_location["viewpointIds"][i] = target_vpid
                            print(
                                f"Directly moving to viewpoint {target_vpid} for alignment"
                            )

                # Set VLN-BERT environment to NavGPT-2's position
                obs_after_move = self.env.set_scan_viewpoint_heading(navgpt2_location)
                obs = np.array(obs_after_move)
                perm_obs = obs[perm_idx]

                # Update trajectory
                for i in range(batch_size):
                    if need_direct_move[i] and not ended[i] and traj is not None:
                        ob = perm_obs[i]
                        traj[i]["path"].append(
                            (ob["viewpoint"], ob["heading"], ob["elevation"])
                        )

            # Prepare environment action for RecVLN
            # Only process actions for samples that didn't need direct move
            else:
                cpu_a_t = np.array(real_action_surr)
                for i, next_id in enumerate(cpu_a_t):
                    if (
                        next_id == (candidate_leng[i] - 1)
                        or next_id == args.ignoreid
                        # or next_id == -2  # Direct move needed, treat as stop
                        or ended[i]
                    ):
                        cpu_a_t[i] = -1  # Change stop action to -1

                # Make action in RecVLN environment (only for samples that didn't need direct move)
                # make_equiv_action will skip -1 actions, but we still need to call it for other samples
                # Only call make_equiv_action if there are samples that didn't need direct move
                self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)

                # Only update obs if we didn't already update it from direct move
                obs = np.array(self.env._get_obs())
                perm_obs = obs[perm_idx]

            # Calculate reward (always calculate for test, even if not training)
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
                    if action_idx == -1:
                        if dist[i] < 3.0:
                            reward[i] = 2.0 + ndtw_score[i] * 2.0
                        else:
                            reward[i] = -2.0
                    else:
                        reward[i] = -(dist[i] - last_dist[i])
                        ndtw_reward = ndtw_score[i] - last_ndtw[i]
                        if reward[i] > 0.0:
                            reward[i] = 1.0 + ndtw_reward
                        elif reward[i] < 0.0:
                            reward[i] = -1.0 + ndtw_reward
                        # else:
                        #     raise NameError("The action doesn't change the move")
                        if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                            reward[i] -= (1.0 - last_dist[i]) * 2.0
            last_dist[:] = dist
            last_ndtw[:] = ndtw_score
            total_reward += reward[0]
            total_discounted_reward += np.power(self.GAE, t) * reward[0]

            # Update finished actions
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))
            target_ended[:] = np.logical_or(
                target_ended, (np.array([x is None for x in target_cpu_a_t]))
            )

            if ended.all():
                break

        print("total reward", self.if_succeed(perm_obs, traj))

        self.a += num_action_total
        self.b += num_mask_total
        print("count", t + 1)

        if not os.path.exists(
            os.path.join(CRITICAL_PROBS_REPRODUCE_DIR, args.timelevelbaseline)
        ):
            os.makedirs(
                os.path.join(CRITICAL_PROBS_REPRODUCE_DIR, args.timelevelbaseline)
            )
        return (
            traj[0],
            total_reward,
            self.if_succeed(perm_obs, traj)[0],
            t + 1,
            num_mask_total,
            mask_pos,
            action_seq,
            mask_probs,
        )
