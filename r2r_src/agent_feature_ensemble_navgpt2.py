from typing import Tuple
from collections import Counter

from agent_mask import MaskAgent
from agent_mask_navgpt2 import NavGPT2_genAction
from agent_feature_navgpt2 import (
    FeatureAgent_NavGPT2,
    NavGPT2_genAction_v2,
    NavGPT2_genAction,
)
from param import args
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import json
import time
import cv2
from PIL import Image
import os
import sys
import r2r_src.vln_utils as vln_utils

# Add NavGPT-2 path to sys.path
navgpt2_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "NavGPT_2", "map_nav_src"
)
if navgpt2_path not in sys.path:
    sys.path.insert(0, navgpt2_path)

from vlnbert.IG_utils import Exp
from vlnbert.XRAI import XRAI, extract_object_masks_yolo
from vlnbert.feature_level_eval import CausalMetric, NpImage
from r2r_src.vlnbert.smdl.submodular_cub_v2_pytorch import (
    CubSubModularExplanationV2,
)

# Import NavGPT-2 agent and utilities
from NavGPT_2.map_nav_src.r2r.agent import GMapNavAgent
from NavGPT_2.map_nav_src.r2r.env import R2RNavBatch
from NavGPT_2.map_nav_src.utils.data import ImageFeaturesDB
from NavGPT_2.map_nav_src.models.graph_utils import GraphMap
from NavGPT_2.map_nav_src.models.ops import pad_tensors_wgrad


class FeatureAgentEnsemble_NavGPT2(FeatureAgent_NavGPT2):
    """
    Ensemble version of FeatureAgent_NavGPT2 that:
    1. Generates feature-level explanations (saliency maps) for 5 agents
    2. Performs soft_vote ensemble on the 5 saliency maps
    3. Evaluates the explanatory effectiveness
    """

    def __init__(self, env, results_path, tok, episode_len=20, args_target=None):
        super(FeatureAgentEnsemble_NavGPT2, self).__init__(
            env, results_path, tok, episode_len, args_target=args_target
        )

        # Initialize agent ID list (5 agents by default)
        # self.agents_id_list = np.arange(args.bagging_agents)
        self.agents_id_list = np.arange(1, args.bagging_agents + 1)
        print(f"Agents ID list: {self.agents_id_list}")

        # Set ensemble mode to soft_vote
        self.ensemble_mode = "soft_vote"

        # Update version for ensemble
        self.VERSION = "v1_ensemble"

        # Update directory paths for ensemble
        # Segmentation map location (shared across agents)
        self.segmentation_map_dir = os.path.join(
            "snap",
            "VLNBERT-train-feature-navgpt2-ensemble",
            "segmentation_map",
        )
        if not os.path.exists(self.segmentation_map_dir):
            os.makedirs(self.segmentation_map_dir)

        # Saliency map location
        saliency_map_dir = os.path.join(
            "snap", args.name + self.VERSION, "saliency_map_pixel"
        )
        if not os.path.exists(saliency_map_dir):
            os.makedirs(saliency_map_dir)
        self.saliency_map_dir = saliency_map_dir

        # Causal metric location
        if self.ensemble_mode == "soft_vote":
            causal_metric_dir = os.path.join(
                "snap",
                args.name + self.VERSION,
                # "causal_metric_pixel" + "_update_seed6",
                "causal_metric_pixel" + f"_{args.bagging_agents}_25del",
            )
        elif self.ensemble_mode == "hard_vote":
            causal_metric_dir = os.path.join(
                "snap", args.name + self.VERSION, "causal_metric_pixel_hard"
            )
        else:
            raise ValueError(f"Unknown ensemble mode: {self.ensemble_mode}")

        if not os.path.exists(causal_metric_dir):
            os.makedirs(causal_metric_dir)
        self.causal_metric_dir = causal_metric_dir

    def rollout_mask_test_navgpt2_feature_per_agent(
        self,
        test_model="IG",
        mode="ins",
        reset=True,
        agent_id=0,
        expand_patch=False,
    ):
        """
        Phase 1: Generate and save saliency maps for a specific agent.

        :param test_model: Test model type
        :param mode: Mode for perturbation
        :param reset: Reset the environment
        :param agent_id: ID of the agent (0-4)
        :param expand_patch: Whether to expand patches
        :return: trajectory
        """
        if reset:
            obs = np.array(self.env.reset_test())
            target_obs = np.array(
                self.target_agent.env.set_scan_viewpoint_heading(
                    self.env.get_scan_viewpoint_heading()
                )
            )
        else:
            obs = np.array(self.env.reset_to_starting_point())
            target_obs = np.array(
                self.target_agent.env.set_scan_viewpoint_heading(
                    self.env.get_scan_viewpoint_heading()
                )
            )

        batch_size = len(obs)
        self.instr_buffer = [[] for _ in range(batch_size)]

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

        # Initialization the tracking state
        ended = np.array([False] * batch_size)

        # Target agent initialization
        if self.target_agent is not None:
            target_traj = [
                {
                    "instr_id": ob["instr_id"],
                    "path": [[ob["viewpoint"]]],
                    "details": {},
                    "a_t": {},
                }
                for ob in target_perm_obs
            ]
            print(target_traj[0]["instr_id"])
            target_ended = np.array([False] * batch_size)
            target_just_ended = np.array([False] * batch_size)

            if GraphMap is None:
                raise ImportError(
                    "GraphMap not available. Please ensure NavGPT-2 is properly set up."
                )

            target_gmaps = [GraphMap(ob["viewpoint"]) for ob in target_perm_obs]
            for i, ob in enumerate(target_perm_obs):
                target_gmaps[i].update_graph(ob)

            target_instructions = [ob["instruction"] for ob in target_perm_obs]
            self.target_agent._update_scanvp_cands(target_perm_obs)
        else:
            print("cannot find target agent")
            exit(0)

        for t in range(self.episode_len):
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

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

            # Generate saliency maps based on test_model
            if test_model in ["IG", "temporal", "IG_temporal"]:
                images, attribution, candidata_list = (
                    self.exp.compute_integrated_gradients(
                        perm_obs,
                        t,
                        h_t,
                        language_features,
                        language_inputs,
                        language_attention_mask,
                        token_type_ids,
                        mode=test_model,
                    )
                )
            else:
                print(f"test_model {test_model} not supported")
                exit(0)

            instr_id = perm_obs[0]["instr_id"]
            XRAI_test = XRAI()

            if expand_patch:
                if not os.path.exists(
                    os.path.join(self.segmentation_map_dir, f"{instr_id}", f"{t}.npy")
                ):
                    object_seg = extract_object_masks_yolo(
                        [
                            Image.fromarray(
                                cv2.cvtColor(x, cv2.COLOR_BGR2RGB), mode="RGB"
                            )
                            for x in images[0, candidata_list[0]]
                        ]
                    )
                    if not os.path.exists(
                        os.path.join(self.segmentation_map_dir, f"{instr_id}")
                    ):
                        os.makedirs(
                            os.path.join(self.segmentation_map_dir, f"{instr_id}")
                        )
                    np.save(
                        os.path.join(
                            self.segmentation_map_dir, f"{instr_id}", f"{t}.npy"
                        ),
                        np.array([x.cpu().numpy() for x in object_seg]),
                    )
                else:
                    object_seg = np.load(
                        os.path.join(
                            self.segmentation_map_dir, f"{instr_id}", f"{t}.npy"
                        )
                    )
                    object_seg = [torch.from_numpy(x) for x in object_seg]
                attr_map, attr_rank = XRAI_test.GetMaskWithDetails(
                    images[0, candidata_list[0]],
                    object_seg,
                    attribution[0][candidata_list[0]],
                    candidata_idx=candidata_list[0],
                    obs=perm_obs[0],
                )
            else:
                attr_map, attr_rank = XRAI_test.getMaskPixel(
                    images[0, candidata_list[0]],
                    attribution[0][candidata_list[0]],
                    candidata_idx=candidata_list[0],
                    obs=perm_obs[0],
                )

            # Save saliency maps with agent_id suffix
            if not os.path.exists(
                os.path.join(self.saliency_map_dir, f"{instr_id}", f"{t}", f"attr_map")
            ):
                os.makedirs(
                    os.path.join(
                        self.saliency_map_dir, f"{instr_id}", f"{t}", f"attr_map"
                    )
                )
            if not os.path.exists(
                os.path.join(self.saliency_map_dir, f"{instr_id}", f"{t}", f"attr_rank")
            ):
                os.makedirs(
                    os.path.join(
                        self.saliency_map_dir, f"{instr_id}", f"{t}", f"attr_rank"
                    )
                )
            np.save(
                os.path.join(
                    self.saliency_map_dir,
                    f"{instr_id}",
                    f"{t}",
                    f"attr_map",
                    f"{agent_id}.npy",
                ),
                attr_map.cpu().numpy(),
            )
            np.save(
                os.path.join(
                    self.saliency_map_dir,
                    f"{instr_id}",
                    f"{t}",
                    f"attr_rank",
                    f"{agent_id}.npy",
                ),
                attr_rank.cpu().numpy(),
            )

            # Get target action
            _, target_nav_vpids, nav_inputs_dict = NavGPT2_genAction(
                self.target_agent,
                target_perm_obs,
                target_gmaps,
                target_instructions,
                t,
                ended=target_ended,
                feedback="argmax",
            )

            target_action = self._teacher_action_baseline_navgpt2(
                target_perm_obs, target_ended, target_nav_vpids
            )
            target_action = target_action.cpu().numpy()
            # Convert to RecVLN action space
            target_action_surr = []
            need_direct_move = [False] * batch_size
            direct_move_targets = [None] * batch_size

            for i in range(batch_size):
                action_result = self._convert_navgpt2_to_recvln_action(
                    target_action[i],
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
                        target_action_surr.append(-1)  # Will be handled separately
                    else:
                        target_action_surr.append(action_idx)
                else:
                    target_action_surr.append(action_result)
            target_action_surr = np.array(target_action_surr)

            # Update target agent trajectory
            for i in range(batch_size):
                target_traj[i]["a_t"][t] = target_action[i]

            target_a_t_stop = [a_t_i == 0 for a_t_i in target_action]

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
                        target_action[i] < len(target_nav_vpids[i])
                        if len(target_nav_vpids) > i
                        else False
                    ):
                        target_vp = target_nav_vpids[i][target_action[i]]
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

            # Handle direct moves for alignment
            has_direct_move = any(need_direct_move) and not all(ended)
            if has_direct_move:
                # Get NavGPT-2 environment state
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

                # Set cpu_a_t for direct move case (all actions are -1 since already moved)
                cpu_a_t = np.array(target_action_surr)
                for i in range(batch_size):
                    if need_direct_move[i]:
                        cpu_a_t[i] = -1  # Already handled by direct move

            # Prepare environment action for RecVLN
            # Only process actions for samples that didn't need direct move
            else:
                cpu_a_t = np.array(target_action_surr)
                for i, next_id in enumerate(cpu_a_t):
                    if (
                        next_id == (candidate_leng[i] - 1)
                        or next_id == args.ignoreid
                        or ended[i]
                    ):
                        cpu_a_t[i] = -1  # Change stop action to -1

                # Make action in RecVLN environment (only for samples that didn't need direct move)
                self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)

                # Only update obs if we didn't already update it from direct move
                obs = np.array(self.env._get_obs())
                perm_obs = obs[perm_idx]

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))
            target_ended[:] = np.logical_or(
                target_ended, (np.array([x is None for x in target_cpu_a_t]))
            )

            if ended.all():
                break

        return traj[0]

    def ensemble_saliency_map(self, mode="soft_vote"):
        """
        Phase 2: Ensemble the saliency maps from different agents with the specified mode.

        Args:
            mode: "soft_vote" (weighted average, default) or "hard_vote" (Borda rank vote)
        Returns:
            dict: { (instr_id, t): (ensemble_map, ensemble_rank) }
        """

        def normalize_map(m):
            m = np.asarray(m, dtype=np.float32)
            if np.ptp(m) == 0:
                return np.zeros_like(m)
            return (m - m.min()) / (m.max() - m.min())

        def soft_vote(attr_map_list, weights=None):
            # Normalize and stack to [N_models, ...]
            maps = [normalize_map(m) for m in attr_map_list]
            maps = np.stack(maps, axis=0)  # shape: [N_models, ...]
            attr_shape = maps.shape[1:]  # always output to original spatial dims
            n_models = maps.shape[0]

            if weights is None:
                weights = np.ones(n_models) / n_models
            else:
                weights = np.array(weights)
                weights = weights / np.sum(weights)
            # Weighted average across model dimension (axis=0), keep shape [H,W] or [...]
            ensemble_map = np.tensordot(weights, maps, axes=(0, 0))  # shape: attr_shape
            # Compute rank for each pixel: each position stores its own importance rank
            flat = ensemble_map.reshape(-1)
            sorted_indices = np.argsort(-flat)  # Indices sorted by value (descending)
            ensemble_rank = np.empty_like(sorted_indices)
            ensemble_rank[sorted_indices] = np.arange(
                len(flat)
            )  # Assign rank to each position
            ensemble_rank = ensemble_rank.reshape(attr_shape)
            return ensemble_map, ensemble_rank

        def hard_vote(attr_rank_list):
            # attr_rank_list: list of [H, W] or [N, H, W]
            attr_shape = attr_rank_list[0].shape
            ranks = np.stack(attr_rank_list, axis=0)  # shape: [N_models, ...]
            # Flatten per model for voting
            ranks_flat = ranks.reshape(ranks.shape[0], -1)
            n_models, n_patches = ranks_flat.shape
            borda_scores = (n_patches - ranks_flat).sum(axis=0)
            # For visualization: normalize to [0, 1]
            ensemble_map = borda_scores / (
                borda_scores.max() if borda_scores.max() > 0 else 1
            )
            ensemble_map = ensemble_map.reshape(attr_shape)
            # Compute rank for each pixel: each position stores its own importance rank
            flat = ensemble_map.reshape(-1)
            sorted_indices = np.argsort(-flat)  # Indices sorted by value (descending)
            ensemble_rank = np.empty_like(sorted_indices)
            ensemble_rank[sorted_indices] = np.arange(
                len(flat)
            )  # Assign rank to each position
            ensemble_rank = ensemble_rank.reshape(attr_shape)
            return ensemble_map, ensemble_rank

        results = {}
        for instr_id in os.listdir(self.saliency_map_dir):
            instr_dir = os.path.join(self.saliency_map_dir, instr_id)
            if not os.path.isdir(instr_dir):
                continue
            for t in os.listdir(instr_dir):
                t_dir = os.path.join(instr_dir, t)
                if not os.path.isdir(t_dir):
                    continue
                attr_map_list = []
                attr_rank_list = []
                for agent_id in self.agents_id_list:
                    attr_map_path = os.path.join(
                        self.saliency_map_dir,
                        f"{instr_id}",
                        f"{t}",
                        "attr_map",
                        f"{agent_id}.npy",
                    )
                    attr_rank_path = os.path.join(
                        self.saliency_map_dir,
                        f"{instr_id}",
                        f"{t}",
                        "attr_rank",
                        f"{agent_id}.npy",
                    )
                    if not os.path.exists(attr_map_path) or not os.path.exists(
                        attr_rank_path
                    ):
                        print(
                            f"Warning: Missing saliency map for agent {agent_id}, instr_id {instr_id}, t {t}"
                        )
                        continue
                    saliency_map = np.load(attr_map_path)
                    attr_rank = np.load(attr_rank_path)
                    attr_map_list.append(saliency_map)
                    attr_rank_list.append(attr_rank)

                if len(attr_map_list) == 0:
                    print(
                        f"Warning: No saliency maps found for instr_id {instr_id}, t {t}"
                    )
                    continue

                if mode in ["soft_vote", "average"]:
                    ensemble_map, ensemble_rank = soft_vote(attr_map_list)
                elif mode in ["hard_vote", "vote"]:
                    ensemble_map, ensemble_rank = hard_vote(attr_rank_list)
                else:
                    raise ValueError("Unknown ensemble mode: {}".format(mode))

                # Save ensemble map and ensemble rank to local disk
                # file_name = (
                #     "ensemble_hard.npy" if mode == "hard_vote" else "ensemble.npy"
                # )
                file_name = "ensemble" + f"_{args.bagging_agents}" + ".npy"
                np.save(
                    os.path.join(
                        self.saliency_map_dir,
                        f"{instr_id}",
                        f"{t}",
                        "attr_map",
                        file_name,
                    ),
                    ensemble_map,
                )
                np.save(
                    os.path.join(
                        self.saliency_map_dir,
                        f"{instr_id}",
                        f"{t}",
                        "attr_rank",
                        file_name,
                    ),
                    ensemble_rank,
                )
                results[(instr_id, t)] = (ensemble_map, ensemble_rank)

        return results

    def rollout_mask_test_navgpt2_feature_ensemble(
        self, test_model="IG", mode="ins", reset=True, perturb_ratio=0.25
    ):
        """
        Phase 3: Load ensemble saliency maps and evaluate causal metrics for NavGPT-2.

        :param test_model: Test model type
        :param mode: Mode for perturbation ("ins" or "del")
        :param reset: Reset the environment
        :param perturb_ratio: Perturbation ratio
        :return: trajectory
        """
        if reset:
            obs = np.array(self.env.reset_test())
            target_obs = np.array(
                self.target_agent.env.set_scan_viewpoint_heading(
                    self.env.get_scan_viewpoint_heading()
                )
            )
        else:
            obs = np.array(self.env.reset_to_starting_point())
            target_obs = np.array(
                self.target_agent.env.set_scan_viewpoint_heading(
                    self.env.get_scan_viewpoint_heading()
                )
            )

        batch_size = len(obs)
        self.instr_buffer = [[] for _ in range(batch_size)]

        # Language input
        sentence, language_attention_mask, token_type_ids, seq_lengths, perm_idx = (
            self._sort_batch(obs)
        )
        perm_obs = obs[perm_idx]
        target_perm_obs = target_obs[perm_idx]

        # Record starting point
        traj = [
            {
                "instr_id": ob["instr_id"],
                "path": [(ob["viewpoint"], ob["heading"], ob["elevation"])],
            }
            for ob in perm_obs
        ]

        ended = np.array([False] * batch_size)

        # Target agent initialization
        if self.target_agent is not None:
            target_traj = [
                {
                    "instr_id": ob["instr_id"],
                    "path": [[ob["viewpoint"]]],
                    "details": {},
                    "a_t": {},
                }
                for ob in target_perm_obs
            ]
            print(target_traj[0]["instr_id"])
            target_ended = np.array([False] * batch_size)
            target_just_ended = np.array([False] * batch_size)

            if GraphMap is None:
                raise ImportError(
                    "GraphMap not available. Please ensure NavGPT-2 is properly set up."
                )

            target_gmaps = [GraphMap(ob["viewpoint"]) for ob in target_perm_obs]
            for i, ob in enumerate(target_perm_obs):
                target_gmaps[i].update_graph(ob)

            target_instructions = [ob["instruction"] for ob in target_perm_obs]
            self.target_agent._update_scanvp_cands(target_perm_obs)
        else:
            print("cannot find target agent")
            exit(0)

        for t in range(self.episode_len):
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            images, candidata_list = self.exp.get_images_and_candidata_list(perm_obs)

            instr_id = perm_obs[0]["instr_id"]

            # Load the ensemble saliency map and rank
            file_name = "ensemble" + f"_{args.bagging_agents}" + ".npy"
            attr_map = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    f"{instr_id}",
                    f"{t}",
                    "attr_map",
                    file_name,
                )
            )
            attr_rank = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    f"{instr_id}",
                    f"{t}",
                    "attr_rank",
                    file_name,
                )
            )

            # Get target action
            _, target_nav_vpids, nav_inputs_dict = NavGPT2_genAction(
                self.target_agent,
                target_perm_obs,
                target_gmaps,
                target_instructions,
                t,
                ended=target_ended,
                feedback="argmax",
            )

            # Use teacher action baseline to get target_action
            target_action = self._teacher_action_baseline_navgpt2(
                target_perm_obs, target_ended, target_nav_vpids
            )
            target_action = target_action.cpu().numpy()

            # Convert to RecVLN action space
            target_action_surr = []
            need_direct_move = [False] * batch_size
            direct_move_targets = [None] * batch_size

            for i in range(batch_size):
                action_result = self._convert_navgpt2_to_recvln_action(
                    target_action[i],
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
                        target_action_surr.append(-1)  # Will be handled separately
                    else:
                        target_action_surr.append(action_idx)
                else:
                    target_action_surr.append(action_result)
            target_action_surr = np.array(target_action_surr)

            params = (
                self.target_agent,
                target_perm_obs,
                target_gmaps,
                target_instructions,
                t,
                target_ended,
            )

            cls_idx = target_action[0]

            # self.causual.average_drop2(
            self.causual.average_drop_navgpt2(
                img=images[0],
                mask=attr_map,
                mask_rank=attr_rank,
                cls_idx=cls_idx,
                params=params,
                mode=mode,
                mask_perc=perturb_ratio,
                candidate_idx=candidata_list[0],
                causal_metric_dir=self.causal_metric_dir,
            )

            # Update target agent trajectory
            for i in range(batch_size):
                target_traj[i]["a_t"][t] = target_action[i]

            target_a_t_stop = [a_t_i == 0 for a_t_i in target_action]

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
                        target_action[i] < len(target_nav_vpids[i])
                        if len(target_nav_vpids) > i
                        else False
                    ):
                        target_vp = target_nav_vpids[i][target_action[i]]
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

            # Handle direct moves for alignment
            has_direct_move = any(need_direct_move) and not all(ended)
            if has_direct_move:
                # Get NavGPT-2 environment state
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

                # Set cpu_a_t for direct move case (all actions are -1 since already moved)
                cpu_a_t = np.array(target_action_surr)
                for i in range(batch_size):
                    if need_direct_move[i]:
                        cpu_a_t[i] = -1  # Already handled by direct move

            # Prepare environment action for RecVLN
            # Only process actions for samples that didn't need direct move
            else:
                cpu_a_t = np.array(target_action_surr)
                for i, next_id in enumerate(cpu_a_t):
                    if (
                        next_id == (candidate_leng[i] - 1)
                        or next_id == args.ignoreid
                        or ended[i]
                    ):
                        cpu_a_t[i] = -1  # Change stop action to -1

                # Make action in RecVLN environment (only for samples that didn't need direct move)
                self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)

                # Only update obs if we didn't already update it from direct move
                obs = np.array(self.env._get_obs())
                perm_obs = obs[perm_idx]

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))
            target_ended[:] = np.logical_or(
                target_ended, (np.array([x is None for x in target_cpu_a_t]))
            )

            if ended.all():
                break

        return traj[0]

    def rollout_mask_test_navgpt2_feature_ensemble_draw(
        self, test_model="IG", mode="ins", reset=True, perturb_ratio=0.25
    ):
        """
        Phase 3: Load ensemble saliency maps and evaluate causal metrics for NavGPT-2.

        :param test_model: Test model type
        :param mode: Mode for perturbation ("ins" or "del")
        :param reset: Reset the environment
        :param perturb_ratio: Perturbation ratio
        :return: trajectory
        """
        if reset:
            obs = np.array(self.env.reset_test())
            target_obs = np.array(
                self.target_agent.env.set_scan_viewpoint_heading(
                    self.env.get_scan_viewpoint_heading()
                )
            )
        else:
            obs = np.array(self.env.reset_to_starting_point())
            target_obs = np.array(
                self.target_agent.env.set_scan_viewpoint_heading(
                    self.env.get_scan_viewpoint_heading()
                )
            )

        batch_size = len(obs)
        self.instr_buffer = [[] for _ in range(batch_size)]

        # Language input
        sentence, language_attention_mask, token_type_ids, seq_lengths, perm_idx = (
            self._sort_batch(obs)
        )
        perm_obs = obs[perm_idx]
        target_perm_obs = target_obs[perm_idx]

        # Record starting point
        traj = [
            {
                "instr_id": ob["instr_id"],
                "path": [(ob["viewpoint"], ob["heading"], ob["elevation"])],
            }
            for ob in perm_obs
        ]

        ended = np.array([False] * batch_size)
        instr_id = perm_obs[0]["instr_id"]
        if instr_id not in ["48_1", "5876_1"]:
            return traj[0]
        # Target agent initialization
        if self.target_agent is not None:
            target_traj = [
                {
                    "instr_id": ob["instr_id"],
                    "path": [[ob["viewpoint"]]],
                    "details": {},
                    "a_t": {},
                }
                for ob in target_perm_obs
            ]
            print(target_traj[0]["instr_id"])
            target_ended = np.array([False] * batch_size)
            target_just_ended = np.array([False] * batch_size)

            if GraphMap is None:
                raise ImportError(
                    "GraphMap not available. Please ensure NavGPT-2 is properly set up."
                )

            target_gmaps = [GraphMap(ob["viewpoint"]) for ob in target_perm_obs]
            for i, ob in enumerate(target_perm_obs):
                target_gmaps[i].update_graph(ob)

            target_instructions = [ob["instruction"] for ob in target_perm_obs]
            self.target_agent._update_scanvp_cands(target_perm_obs)
        else:
            print("cannot find target agent")
            exit(0)

        for t in range(self.episode_len):
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            images, candidata_list = self.exp.get_images_and_candidata_list(perm_obs)

            instr_id = perm_obs[0]["instr_id"]

            # Load the ensemble saliency map and rank
            # file_name = (
            #     "ensemble_hard.npy"
            #     if self.ensemble_mode == "hard_vote"
            #     else "ensemble.npy"
            # )
            file_name = "ensemble" + f"_{args.bagging_agents}" + ".npy"
            attr_map = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    f"{instr_id}",
                    f"{t}",
                    "attr_map",
                    file_name,
                )
            )
            attr_rank = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    f"{instr_id}",
                    f"{t}",
                    "attr_rank",
                    file_name,
                )
            )

            # Overlay heatmap on panoramic images instead of running inference
            # Set output directory for overlaid images
            heatmap_overlay_dir = os.path.join(
                "snap", args.name + self.VERSION, "heatmap_overlay"
            )
            if not os.path.exists(heatmap_overlay_dir):
                os.makedirs(heatmap_overlay_dir)

            # Overlay heatmap on images
            saved_paths = self.causual.overlay_heatmap_on_panoramic_images(
                images=images,
                attr_map=attr_map,
                candidate_idx=candidata_list[0],
                instr_id=instr_id,
                t=t,
                output_dir=heatmap_overlay_dir,
                alpha=0.5,
                colormap=cv2.COLORMAP_JET,
            )
            print(f"Saved {len(saved_paths)} overlaid images for {instr_id} at t={t}")

            # Get target action for trajectory tracking (but don't run inference)
            _, target_nav_vpids, nav_inputs_dict = NavGPT2_genAction(
                self.target_agent,
                target_perm_obs,
                target_gmaps,
                target_instructions,
                t,
                ended=target_ended,
                feedback="argmax",
            )

            # Use teacher action baseline to get target_action
            target_action = self._teacher_action_baseline_navgpt2(
                target_perm_obs, target_ended, target_nav_vpids
            )
            target_action = target_action.cpu().numpy()

            # Convert to RecVLN action space
            target_action_surr = []
            need_direct_move = [False] * batch_size
            direct_move_targets = [None] * batch_size

            for i in range(batch_size):
                action_result = self._convert_navgpt2_to_recvln_action(
                    target_action[i],
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
                        target_action_surr.append(-1)  # Will be handled separately
                    else:
                        target_action_surr.append(action_idx)
                else:
                    target_action_surr.append(action_result)
            target_action_surr = np.array(target_action_surr)

            # Update target agent trajectory
            for i in range(batch_size):
                target_traj[i]["a_t"][t] = target_action[i]

            target_a_t_stop = [a_t_i == 0 for a_t_i in target_action]

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
                        target_action[i] < len(target_nav_vpids[i])
                        if len(target_nav_vpids) > i
                        else False
                    ):
                        target_vp = target_nav_vpids[i][target_action[i]]
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

            # Handle direct moves for alignment
            has_direct_move = any(need_direct_move) and not all(ended)
            if has_direct_move:
                # Get NavGPT-2 environment state
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

                # Set cpu_a_t for direct move case (all actions are -1 since already moved)
                cpu_a_t = np.array(target_action_surr)
                for i in range(batch_size):
                    if need_direct_move[i]:
                        cpu_a_t[i] = -1  # Already handled by direct move

            # Prepare environment action for RecVLN
            # Only process actions for samples that didn't need direct move
            else:
                cpu_a_t = np.array(target_action_surr)
                for i, next_id in enumerate(cpu_a_t):
                    if (
                        next_id == (candidate_leng[i] - 1)
                        or next_id == args.ignoreid
                        or ended[i]
                    ):
                        cpu_a_t[i] = -1  # Change stop action to -1

                # Make action in RecVLN environment (only for samples that didn't need direct move)
                self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)

                # Only update obs if we didn't already update it from direct move
                obs = np.array(self.env._get_obs())
                perm_obs = obs[perm_idx]

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))
            target_ended[:] = np.logical_or(
                target_ended, (np.array([x is None for x in target_cpu_a_t]))
            )

            if ended.all():
                break

        return traj[0]

    def test(self, iters=None, **kwargs):
        """
        Main test method that coordinates three phases:
        1. Generate saliency maps for each of the 5 agents
        2. Ensemble the saliency maps using soft_vote
        3. Evaluate the explanatory effectiveness
        """
        test_model = args.feature_level_baseline
        assert test_model is not None, "test_model cannot be None"

        phase1 = False  # Generate saliency maps for each agent
        phase2 = False  # Ensemble saliency maps
        phase3 = False  # Evaluate ensemble model
        mu = False  # Compute muFidelity
        if args.update_inference == "heatmap":
            phase1 = True
            phase2 = True
        elif args.update_inference == "inference":
            # phase2 = True
            phase3 = True
            mu = True
        # Phase 1: Generate saliency maps for each agent
        if phase1:
            self.env.reset_epoch(shuffle=(iters is not None))
            self.losses = []
            self.results = {}
            looped = False
            self.loss = 0

            if iters is not None:
                for agent_id in self.agents_id_list:
                    print(f"Phase 1: Generating saliency maps for agent {agent_id}")
                    # Load checkpoint for agent i
                    if hasattr(args, "load") and args.load is not None:
                        agent_checkpoint_path = os.path.join(
                            args.load, f"agent_{agent_id}", "best_val72"
                        )
                        if os.path.exists(agent_checkpoint_path):
                            self.load(agent_checkpoint_path)
                    for i in range(iters):
                        traj = self.rollout_mask_test_navgpt2_feature_per_agent(
                            test_model=test_model,
                            reset=True,
                            agent_id=agent_id,
                        )
                        self.loss = 0
                        self.results[traj["instr_id"]] = traj["path"]
            else:
                for agent_id in self.agents_id_list:
                    print(f"Phase 1: Generating saliency maps for agent {agent_id}")
                    self.env.reset_epoch(shuffle=False)
                    self.results = {}
                    looped = False
                    num = 0
                    # Load checkpoint for agent i
                    if hasattr(args, "load") and args.load is not None:
                        agent_checkpoint_path = os.path.join(
                            args.load, f"agent_{agent_id}", "best_val72"
                        )
                        if os.path.exists(agent_checkpoint_path):
                            self.load(agent_checkpoint_path)
                    while True:
                        traj = self.rollout_mask_test_navgpt2_feature_per_agent(
                            test_model=test_model,
                            reset=True,
                            agent_id=agent_id,
                        )
                        num += 1
                        if num > 5:
                            break
                        if traj["instr_id"] in self.results:
                            looped = True
                        else:
                            self.results[traj["instr_id"]] = traj["path"]
                        if looped:
                            break

        # Phase 2: Ensemble the saliency maps
        if phase2:
            print("Phase 2: Ensemble saliency maps using soft_vote")
            self.ensemble_saliency_map(mode=self.ensemble_mode)

        # Phase 3: Evaluate ensemble model
        if phase3:
            print("Phase 3: Evaluating ensemble model")
            self.env.reset_epoch(shuffle=(iters is not None))
            self.losses = []
            self.results = {}
            looped = False
            self.loss = 0

            if iters is not None:
                assert False, "iters is not None"
            else:
                while True:
                    print("new rollout")
                    # traj = self.rollout_mask_test_navgpt2_feature_ensemble(
                    #     test_model=test_model,
                    #     mode="ins",
                    #     reset=True,
                    #     perturb_ratio=0.25,
                    # )
                    # traj = self.rollout_mask_test_navgpt2_feature_ensemble(
                    #     test_model=test_model,
                    #     mode="del",
                    #     # reset=False,
                    #     reset=True,
                    #     perturb_ratio=0.25,
                    # )
                    traj = self.rollout_mask_test_navgpt2_feature_ensemble(
                        test_model=test_model,
                        mode="ins",
                        # reset=False,
                        reset=True,
                        perturb_ratio=0.5,
                    )
                    # traj = self.rollout_mask_test_navgpt2_feature_ensemble(
                    #     test_model=test_model,
                    #     mode="del",
                    #     reset=False,
                    #     perturb_ratio=0.5,
                    # )
                    traj = self.rollout_mask_test_navgpt2_feature_ensemble(
                        test_model=test_model,
                        mode="ins",
                        reset=False,
                        # reset=True,
                        perturb_ratio=0.75,
                    )
                    # traj = self.rollout_mask_test_navgpt2_feature_ensemble(
                    #     test_model=test_model,
                    #     mode="del",
                    #     reset=False,
                    #     perturb_ratio=0.75,
                    # )
                    if traj["instr_id"] in self.results:
                        looped = True
                    else:
                        self.results[traj["instr_id"]] = traj["path"]
                    if looped:
                        break

        # Compute muFidelity
        if mu:
            print("Computing muFidelity...")
            muFidelity = self.causual.compute_muFidelity(self.causal_metric_dir)
            print("muFidelity", muFidelity)
