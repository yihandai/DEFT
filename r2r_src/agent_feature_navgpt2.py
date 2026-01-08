from typing import Tuple
from collections import Counter

from agent_mask import MaskAgent
from agent_mask_navgpt2 import NavGPT2_genAction
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

CRITICAL_PROBS_GT_DIR = "./critical_probs_gt"


def NavGPT2_genAction_v2(
    agent: GMapNavAgent,
    obs,
    gmaps,
    instructions,
    t,
    ended=None,
    feedback="argmax",
    nav_inputs=None,
    new_imgs=None,
    candidata_list=None,
    instr_id=None,
    perc=None,
    mode=None,
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

        # Note: We skip updating graph node embeddings here to keep this as pure inference
        # The gmaps should be updated separately if needed

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

        # Note: We skip updating graph node stop scores here to keep this as pure inference
        # The gmaps should be updated separately if needed

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


class FeatureAgent_NavGPT2(MaskAgent):
    def __init__(self, env, results_path, tok, episode_len=20, args_target=None):
        super(FeatureAgent_NavGPT2, self).__init__(
            env, results_path, tok, episode_len, args_target=args_target
        )
        rank = 0

        self.vln_bert.eval()
        self.critic.eval()
        self.critical_head.eval()
        self.critic4mask.eval()

        if args.feature_level_baseline == "smdl":
            self.exp = CubSubModularExplanationV2(self.vln_bert, self.critical_head)
        else:
            self.exp = Exp(self.vln_bert, self.critical_head)

        if args_target is None:
            raise ValueError("args_target must be provided for NavGPT-2 agent")

        # Initialize NavGPT-2 agent
        self.target_agent = GMapNavAgent(args_target, env, rank=rank)

        # Load checkpoint if specified
        if hasattr(args_target, "resume_file") and args_target.resume_file is not None:
            self.target_agent.load(args_target.resume_file)

        self.causual = CausalMetric(
            call_fn=NavGPT2_genAction_v2,
            substrate_fn=np.zeros_like,
            H=480,
            W=640,
            target="NavGPT2",
        )

        self.VERSION = "v1"

        # Segmentation map location
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
        causal_metric_dir = os.path.join(
            "snap",
            args.name + self.VERSION,
            "causal_metric_pixel" + "_update_replication_1",
        )
        if not os.path.exists(causal_metric_dir):
            os.makedirs(causal_metric_dir)
        self.causal_metric_dir = causal_metric_dir

    def test(self, iters=None, **kwargs):
        test_model = args.feature_level_baseline
        assert test_model is not None, "test_model cannot be None"
        phase2 = False
        phase3 = False
        phase_merge = False
        original_image = False
        mu = False
        if args.update_inference == "heatmap":
            phase2 = True
            phase3 = True
        elif args.update_inference == "inference":
            phase3 = True
            mu = True

        if phase2:
            self.env.reset_epoch(shuffle=(iters is not None))
            self.losses = []
            self.results = {}
            looped = False
            self.loss = 0
            self.strt = False
            count = 5
            if iters is not None:
                for i in range(iters):
                    for traj in self.rollout_mask_test_navgpt2_feature_phase2(
                        test_model=test_model
                    ):
                        self.loss = 0
                        self.results[traj["instr_id"]] = traj["path"]
            else:
                lets_start = False
                while True:
                    traj = self.rollout_mask_test_navgpt2_feature_phase2(
                        test_model=test_model,
                        reset=True,
                    )
                    count += 1
                    if count > 5:
                        break
                    if traj["instr_id"] in self.results:
                        looped = True
                    else:
                        self.results[traj["instr_id"]] = traj["path"]
                    if looped:
                        break

        if phase_merge and args.feature_level_baseline == "IG_temporal":
            self.merge_IG_temporal_saliency_map(mode="soft_vote")

        if phase3:
            self.env.reset_epoch(shuffle=(iters is not None))
            self.losses = []
            self.results = {}
            looped = False
            self.loss = 0

            while True:
                # traj = self.rollout_mask_test_navgpt2_feature_phase3(
                #     test_model=test_model,
                #     mode="ins",
                #     reset=True,
                #     perturb_ratio=0.25,
                # )
                traj = self.rollout_mask_test_navgpt2_feature_phase3(
                    test_model=test_model,
                    mode="del",
                    # reset=False,
                    reset=True,
                    perturb_ratio=0.25,
                )
                # traj = self.rollout_mask_test_navgpt2_feature_phase3(
                #     test_model=test_model,
                #     mode="ins",
                #     reset=False,
                #     perturb_ratio=0.5,
                # )
                # traj = self.rollout_mask_test_navgpt2_feature_phase3(
                #     test_model=test_model,
                #     mode="del",
                #     reset=False,
                #     perturb_ratio=0.5,
                # )
                # traj = self.rollout_mask_test_navgpt2_feature_phase3(
                #     test_model=test_model,
                #     mode="ins",
                #     reset=False,
                #     perturb_ratio=0.75,
                # )
                # traj = self.rollout_mask_test_navgpt2_feature_phase3(
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

        if mu:
            muFidelity = self.causual.compute_muFidelity(self.causal_metric_dir)
            print("muFidelity", muFidelity)

        if original_image:
            self.env.reset_epoch(shuffle=(iters is not None))
            self.losses = []
            self.results = {}
            looped = False
            self.loss = 0

            while True:
                traj = self.rollout_mask_test_navgpt2_feature_phase3(
                    test_model=test_model,
                    mode="ins",
                    reset=True,
                    perturb_ratio=1.0,
                )
                if traj["instr_id"] in self.results:
                    looped = True
                else:
                    self.results[traj["instr_id"]] = traj["path"]
                if looped:
                    break

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
        self.target_agent.make_equiv_action(
            cpu_a_t, gmaps, target_perm_obs, target_traj, perm_idx=perm_idx
        )

    def rollout_mask_test_navgpt2_feature_phase2(
        self,
        test_model="IG",
        mode="ins",
        reset=True,
        expand_patch=False,
    ):
        """
        Phase 2: Generate and save saliency maps for NavGPT-2.

        :param test_model: Test model type
        :param mode: Mode for perturbation
        :param reset: Reset the environment
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

        # if traj[0]["instr_id"] == "2539_0":
        #     self.strt = True
        # if not self.strt:
        #     return traj[0]
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
            print("----target_traj[0]['instr_id']------", target_traj[0]["instr_id"])
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

        _vp_critical_probs = {}
        _instr_id = perm_obs[0]["instr_id"]
        for t in range(self.episode_len):
            print("t: ", t)
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
            elif test_model in ["guided_IG"]:
                images, attribution, candidata_list = self.exp.get_guided_ig(
                    perm_obs,
                    t,
                    h_t,
                    language_features,
                    language_inputs,
                    language_attention_mask,
                    token_type_ids,
                )
            elif test_model in ["smdl"]:
                images, attribution, candidata_list = self.exp.exp(
                    perm_obs,
                    t,
                    h_t,
                    language_features=language_features,
                    language_inputs=language_inputs,
                    language_attention_mask=language_attention_mask,
                    token_type_ids=token_type_ids,
                )
            elif test_model in ["random"]:
                images, attribution, candidata_list = self.exp.compute_random_salency(
                    perm_obs,
                    t,
                    h_t,
                    language_features=language_features,
                    language_inputs=language_inputs,
                    language_attention_mask=language_attention_mask,
                    token_type_ids=token_type_ids,
                )
            elif test_model in ["fg_cam"]:
                images, attribution, candidata_list = self.exp.compute_FG_CAM(
                    perm_obs,
                    t,
                    h_t,
                    language_features=language_features,
                    language_inputs=language_inputs,
                    language_attention_mask=language_attention_mask,
                    token_type_ids=token_type_ids,
                )
            elif test_model in ["hsic"]:
                print("current time (hour, minute, second): ", time.localtime())
                images, attribution, candidata_list = (
                    self.exp.compute_hsic_attribution_navgpt2(
                        target_perm_obs,  # Use target_perm_obs (NavGPT2 obs) instead of perm_obs (VLN-BERT obs)
                        t,
                        target_agent=self.target_agent,
                        gmaps=target_gmaps,
                        instructions=target_instructions,
                        navgpt2_gen_action_fn=NavGPT2_genAction_v2,  # Pass function as parameter
                        grid_size=getattr(args, "hsic_grid_size", 2),
                        nb_design=getattr(args, "hsic_nb_design", 10),
                        perturbation_function=getattr(
                            args, "hsic_perturbation", "inpainting"
                        ),
                        batch_size=getattr(args, "hsic_batch_size", 32),
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

            # Save saliency maps
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
                    f"attr_map.npy",
                ),
                attr_map.cpu().numpy(),
            )
            np.save(
                os.path.join(
                    self.saliency_map_dir,
                    f"{instr_id}",
                    f"{t}",
                    f"attr_rank.npy",
                ),
                attr_rank.cpu().numpy(),
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

                # # Set cpu_a_t for direct move case (all actions are -1 since already moved)
                # cpu_a_t = np.array(target_action_surr)
                # for i in range(batch_size):
                #     if need_direct_move[i]:
                #         cpu_a_t[i] = -1  # Already handled by direct move

            # Prepare environment action for RecVLN
            # Only process actions for samples that didn't need direct move
            else:
                cpu_a_t = np.array(target_action_surr)
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

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))
            target_ended[:] = np.logical_or(
                target_ended, (np.array([x is None for x in target_cpu_a_t]))
            )

            if ended.all():
                break

        return traj[0]

    def rollout_mask_test_navgpt2_feature_phase3(
        self, test_model="IG", mode="ins", reset=True, perturb_ratio=0.25
    ):
        """
        Phase 3: Load saliency maps and evaluate causal metrics for NavGPT-2.

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

            # Load the saliency map and rank
            attr_map = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    f"{instr_id}",
                    f"{t}",
                    "attr_map.npy",
                )
            )
            attr_rank = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    f"{instr_id}",
                    f"{t}",
                    "attr_rank.npy",
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

            # cls_idx = target_action_surr[0] if len(target_action_surr) > 0 else 0
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

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))
            target_ended[:] = np.logical_or(
                target_ended, (np.array([x is None for x in target_cpu_a_t]))
            )

            if ended.all():
                break

        return traj[0]

    def release_vlnbert_memory(self):
        """
        Release VLN-BERT model memory to free up GPU memory for NavGPT-2.
        This should be called before using phase3_2 functions.
        """
        if hasattr(self, "vln_bert") and self.vln_bert is not None:
            del self.vln_bert
            self.vln_bert = None
        if hasattr(self, "critic") and self.critic is not None:
            del self.critic
            self.critic = None
        if hasattr(self, "critical_head") and self.critical_head is not None:
            del self.critical_head
            self.critical_head = None
        if hasattr(self, "critic4mask") and self.critic4mask is not None:
            del self.critic4mask
            self.critic4mask = None
        if hasattr(self, "exp") and self.exp is not None:
            # Only release bert-related parts, keep sim for image loading
            if hasattr(self.exp, "bert"):
                del self.exp.bert
                self.exp.bert = None
            if hasattr(self.exp, "critical_head"):
                del self.exp.critical_head
                self.exp.critical_head = None
        # Force garbage collection and clear CUDA cache
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("VLN-BERT models released from memory")

    def rollout_mask_test_navgpt2_feature_phase3_2(
        self, test_model="IG", mode="ins", reset=True, perturb_ratio=0.25
    ):
        """
        Phase 3 (Memory-Optimized): Load saliency maps and evaluate causal metrics for NavGPT-2.
        This version does NOT use VLN-BERT, saving GPU memory for NavGPT-2.

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

        # Language input (only for sorting, not for VLN-BERT)
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
            # Get candidate_leng directly from observations (no VLN-BERT needed)
            candidate_leng = [
                len(ob["candidate"]) + 1 for ob in perm_obs
            ]  # +1 is for the end

            # Get images and candidate list (only uses sim, not VLN-BERT)
            images, candidata_list = self.exp.get_images_and_candidata_list(perm_obs)

            instr_id = perm_obs[0]["instr_id"]

            # Load the saliency map and rank
            attr_map = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    f"{instr_id}",
                    f"{t}",
                    "attr_map.npy",
                )
            )
            attr_rank = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    f"{instr_id}",
                    f"{t}",
                    "attr_rank.npy",
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

            # cls_idx = target_action_surr[0] if len(target_action_surr) > 0 else 0
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

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))
            target_ended[:] = np.logical_or(
                target_ended, (np.array([x is None for x in target_cpu_a_t]))
            )

            if ended.all():
                break

        return traj[0]

    def merge_IG_temporal_saliency_map(self, mode="soft_vote"):
        """
        Merge the saliency maps from IG and temporal models with the specified mode.

        Args:
            mode: "soft_vote" or "hard_vote"
        Returns:
            dict: { (instr_id, t): (merged_map, merged_rank) }
        """

        def normalize_map(m):
            m = np.asarray(m, dtype=np.float32)
            if np.ptp(m) == 0:
                return np.zeros_like(m)
            return (m - m.min()) / (m.max() - m.min())

        def soft_vote(attr_map_list, weights=None):
            maps = [normalize_map(m) for m in attr_map_list]
            maps = np.stack(maps, axis=0)
            attr_shape = maps.shape[1:]
            n_models = maps.shape[0]

            if weights is None:
                weights = np.ones(n_models) / n_models
            else:
                weights = np.array(weights)
                weights = weights / np.sum(weights)

            ensemble_map = np.tensordot(weights, maps, axes=(0, 0))
            flat = ensemble_map.reshape(-1)
            sorted_indices = np.argsort(-flat)
            ensemble_rank = np.empty_like(sorted_indices)
            ensemble_rank[sorted_indices] = np.arange(len(flat))
            ensemble_rank = ensemble_rank.reshape(attr_shape)
            return ensemble_map, ensemble_rank

        def hard_vote(attr_rank_list):
            attr_shape = attr_rank_list[0].shape
            ranks = np.stack(attr_rank_list, axis=0)
            ranks_flat = ranks.reshape(ranks.shape[0], -1)
            n_models, n_patches = ranks_flat.shape
            borda_scores = (n_patches - ranks_flat).sum(axis=0)
            ensemble_map = borda_scores / (
                borda_scores.max() if borda_scores.max() > 0 else 1
            )
            ensemble_map = ensemble_map.reshape(attr_shape)
            flat = ensemble_map.reshape(-1)
            sorted_indices = np.argsort(-flat)
            ensemble_rank = np.empty_like(sorted_indices)
            ensemble_rank[sorted_indices] = np.arange(len(flat))
            ensemble_rank = ensemble_rank.reshape(attr_shape)
            return ensemble_map, ensemble_rank

        results = {}
        dir_names = [
            "VLNBERT-test-baseline-navgpt2-ig",
            "VLNBERT-test-baseline-navgpt2-temporal",
        ]
        saliency_map_dir_IG = os.path.join("snap", dir_names[0], "saliency_map_pixel")
        saliency_map_dir_temporal = os.path.join(
            "snap", dir_names[1], "saliency_map_pixel"
        )
        saliency_map_dir_IG_temporal = self.saliency_map_dir

        for instr_id in os.listdir(saliency_map_dir_IG):
            instr_dir = os.path.join(saliency_map_dir_IG, instr_id)
            for t in os.listdir(instr_dir):
                attr_map_list = []
                attr_rank_list = []

                for dir_name in dir_names:
                    saliency_map_dir = os.path.join(
                        "snap", dir_name, "saliency_map_pixel", instr_id, t
                    )
                    attr_map_path = os.path.join(saliency_map_dir, "attr_map.npy")
                    attr_rank_path = os.path.join(saliency_map_dir, "attr_rank.npy")
                    saliency_map = np.load(attr_map_path)
                    attr_rank = np.load(attr_rank_path)
                    attr_map_list.append(saliency_map)
                    attr_rank_list.append(attr_rank)

                if mode in ["soft_vote", "average"]:
                    ensemble_map, ensemble_rank = soft_vote(attr_map_list)
                elif mode in ["hard_vote", "vote"]:
                    ensemble_map, ensemble_rank = hard_vote(attr_rank_list)
                else:
                    raise ValueError("Unknown ensemble mode: {}".format(mode))

                # Save ensemble map and rank
                if not os.path.exists(saliency_map_dir_IG_temporal):
                    os.makedirs(saliency_map_dir_IG_temporal)
                if not os.path.exists(
                    os.path.join(saliency_map_dir_IG_temporal, instr_id)
                ):
                    os.makedirs(os.path.join(saliency_map_dir_IG_temporal, instr_id))
                if not os.path.exists(
                    os.path.join(saliency_map_dir_IG_temporal, instr_id, t)
                ):
                    os.makedirs(os.path.join(saliency_map_dir_IG_temporal, instr_id, t))
                np.save(
                    os.path.join(
                        saliency_map_dir_IG_temporal, instr_id, t, "attr_map.npy"
                    ),
                    ensemble_map,
                )
                np.save(
                    os.path.join(
                        saliency_map_dir_IG_temporal, instr_id, t, "attr_rank.npy"
                    ),
                    ensemble_rank,
                )
                results[(instr_id, t)] = (ensemble_map, ensemble_rank)

        return results
