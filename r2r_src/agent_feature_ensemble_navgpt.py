from typing import Tuple
from collections import Counter

# from agent import Seq2SeqAgent
from agent_mask import MaskAgent
from param import args, target_args
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import json
import time
import cv2
from PIL import Image
import os
import json
import r2r_src.vln_utils as vln_utils

from NavGPT.nav_src.agent import NavAgent
from agent_mask_navgpt import NavGPT_genAction
from agent_feature_navgpt import (
    NavGPT_genAction_v2,
    collect_nav_info,
    get_navgpt_viewpoint_id_from_file,
    FeatureAgent_NavGPT,
)
from vlnbert.IG_utils import Exp
from vlnbert.XRAI import XRAI, extract_object_masks_yolo
from vlnbert.feature_level_eval import CausalMetric, NpImage
from r2r_src.vlnbert.smdl.submodular_cub_v2_pytorch import (
    CubSubModularExplanationV2,
)

try:
    from langchain.agents.agent import AgentAction
except ImportError:
    # Fallback for different langchain versions
    try:
        from langchain.schema import AgentAction
    except ImportError:
        AgentAction = None  # Will handle None case in code


nav_inputs = []


class FeatureAgentEnsemble_NavGPT(FeatureAgent_NavGPT):
    def __init__(self, env, results_path, tok, episode_len=20, args_target=None):
        super(FeatureAgentEnsemble_NavGPT, self).__init__(
            env, results_path, tok, episode_len, args_target=args_target
        )
        self.agents_id_list = np.arange(args.bagging_agents)
        print(f"Agents ID list: {self.agents_id_list}")

        self.VERSION = "v3"
        ensemble_mode = "soft_vote"
        self.ensemble_mode = ensemble_mode

        # saliency map location
        saliency_map_dir = os.path.join(
            "snap", args.name + self.VERSION, "saliency_map_pixel"
        )
        if not os.path.exists(saliency_map_dir):
            os.makedirs(saliency_map_dir)
        self.saliency_map_dir = saliency_map_dir

        # causal metric location
        if self.ensemble_mode == "soft_vote":
            causal_metric_dir = os.path.join(
                # "snap", args.name + self.VERSION, "causal_metric_pixel"
                "snap",
                args.name + self.VERSION,
                "causal_metric_pixel_3",
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

        # used for updating the description for observation
        description_update_dir = os.path.join(
            "snap", args.name + self.VERSION, "description_update"
        )
        if not os.path.exists(description_update_dir):
            os.makedirs(description_update_dir)
        self.description_update_dir = description_update_dir

    # ensemble pipeline
    # - get segmentation map
    # - save the segmentation map
    # - for each agent
    # ----- rollout the trajectory
    # ----- save the saliency map
    # - merge the saliency maps
    # - calculate f(x) using LLM
    # - get the causal metric
    # - return the causal metric
    def rollout_mask_test_navgpt_feature_per_agent(
        self,
        test_model="IG",
        mode="ins",
        reset=True,
        agent_id=0,
        expand_patch=False,
    ):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment
        :param agent_id:    ID of the agent for saving saliency maps

        :return:
        """
        if reset:
            obs = np.array(self.env.reset_test())
            target_obs = np.array(
                self.target_agent.env.set_scan_viewpoint_heading(
                    self.env.get_scan_viewpoint_heading()
                )
            )
        else:
            # obs = np.array(self.env._get_obs())
            obs = np.array(self.env.reset_to_starting_point())
            target_obs = np.array(
                self.target_agent.env.set_scan_viewpoint_heading(
                    self.env.get_scan_viewpoint_heading()
                )
            )

        batch_size = len(obs)
        self.instr_buffer = [
            [] for _ in range(batch_size)
        ]  # to identify which vp's index has been used

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
        ended = np.array(
            [False] * batch_size
        )  # Indices match permuation of the model, not env

        # baseline agent init --------------------------
        if self.target_agent is not None:
            # Initialize NavGPT agent
            self.target_agent.init_trajecotry(target_perm_obs)
            # Load the instruction for NavGPT
            instructions = [ob["instruction"] for ob in target_perm_obs]
            if self.target_agent.config.load_instruction:
                action_plans = instructions
            elif self.target_agent.config.load_action_plan:
                action_plans = [ob["action_plan"] for ob in target_perm_obs]
            else:
                action_plans = []
                for instruction in instructions:
                    action_plan = self.target_agent.plan_chain.run(
                        instruction=instruction
                    )
                    action_plans.append(action_plan)
            # Set action plan for first observation (batch_size=1 assumed)
            if len(target_perm_obs) > 0:
                self.target_agent.cur_action_plan = action_plans[0]

            # Initialize accumulated_intermediate_steps for NavGPT agent context
            if not hasattr(self.target_agent, "_accumulated_intermediate_steps"):
                self.target_agent._accumulated_intermediate_steps = []

            # Initialize last_observation for tool_chain mode
            if self.target_agent.config.use_tool_chain:
                if not hasattr(self.target_agent, "_last_observation"):
                    self.target_agent._last_observation = None

            target_traj = [
                {
                    "instr_id": ob["instr_id"],
                    "path": [[ob["viewpoint"]]],
                    "details": {},
                    "a_t": {},
                    "uncertainty": {},
                    "probs": {},
                }
                for ob in target_perm_obs
            ]
            print(target_traj[0]["instr_id"])
            # Initialization the tracking state
            target_ended = np.array([False] * batch_size)
            target_just_ended = np.array([False] * batch_size)

            previous_angle = [
                {"heading": ob["heading"], "elevation": ob["elevation"]}
                for ob in target_perm_obs
            ]
        else:
            print("cannot find target agent")
            exit(0)

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
            # the only thing that i need is h_t for the next t
            h_t, logit = self.vln_bert(**visual_inputs)

            # do integrated gradients and generate attribution maps
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
            else:
                print(f"test_model {test_model} not supported")
                exit(0)

            instr_id = perm_obs[0]["instr_id"]
            XRAI_test = XRAI()

            # get the saliency map and rank
            attr_map, attr_rank = XRAI_test.getMaskPixel(
                images[0, candidata_list[0]],
                attribution[0][candidata_list[0]],
                candidata_idx=candidata_list[0],
                obs=perm_obs[0],
            )

            # shape of attr_map is [len(candidate), H, W]
            # save attr_map and attr_rank with agent_id
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
            # Get action from file for NavGPT
            # Read viewpoint_id from file
            viewpoint_id = get_navgpt_viewpoint_id_from_file(
                perm_obs[0]["instr_id"], t, target_perm_obs[0].get("candidate", {})
            )
            print("viewpoint_id", viewpoint_id)
            candidate_list_surr = perm_obs[0].get("candidate", [])
            if viewpoint_id and viewpoint_id in [
                x["viewpointId"] for x in candidate_list_surr
            ]:
                target_action_surr = [
                    [x["viewpointId"] for x in candidate_list_surr].index(viewpoint_id)
                ]
                print("target_action_surr", target_action_surr)
            else:
                target_action_surr = [len(candidate_list_surr)]  # Stop action

            candidata_dict = target_perm_obs[0].get("candidate", {})
            if viewpoint_id and viewpoint_id in candidata_dict.keys():
                target_action = [list(candidata_dict.keys()).index(viewpoint_id) + 1]
                print("target_action", target_action)
            else:
                target_action = [0]  # Stop action
            print("candidata_dict", candidata_dict.keys())
            print(
                "candidate_list_surr", [x["viewpointId"] for x in candidate_list_surr]
            )

            # 确定真实动作
            # NOTE: NavGPT 的 real action 里面 0 是停止，
            # NOTE: 在 RecVLN 里面，candidate_len[i] - 1 是停止
            # ############### get new obs###########################
            # for target agent---------------------------
            if self.target_agent is not None:
                for i in range(batch_size):
                    target_traj[i]["a_t"][t] = target_action[i]

                # Determine stop actions
                target_a_t_stop = [a_t_i == 0 for a_t_i in target_action]

                # Prepare environment action
                # NavGPT uses viewpoint IDs
                target_cpu_a_t = []
                for i in range(batch_size):
                    if target_a_t_stop[i] or target_ended[i]:
                        target_cpu_a_t.append(None)  # Stop action for NavGPT
                        target_just_ended[i] = True
                    else:
                        target_cpu_a_t.append(viewpoint_id)

                # Execute action in NavGPT's environment
                # NOTE: We read action from file, so we execute it directly
                for i, vp_id in enumerate(target_cpu_a_t):
                    if vp_id is not None:
                        # Execute the action
                        _, new_obs = self.target_agent.make_equiv_action([vp_id])

                        # Update history for NavGPT (similar to rollout_mask_navgpt)
                        if self.target_agent.config.use_history_chain:
                            new_feature = new_obs.get("obs", "")
                            new_feature_sum = new_obs.get("obs_summary", "")
                            if hasattr(self.target_agent, "history_chain"):
                                history = self.target_agent.history_chain.run(
                                    observation=new_feature_sum,
                                    history=(
                                        self.target_agent.agent_executor.agent.history[
                                            -1
                                        ]
                                        if len(
                                            self.target_agent.agent_executor.agent.history
                                        )
                                        > 0
                                        else ""
                                    ),
                                    previous_action="Moved to viewpoint",
                                )
                            else:
                                history = self.target_agent.get_history(
                                    new_obs, "Moved to viewpoint"
                                )
                        else:
                            history = self.target_agent.get_history(
                                new_obs, "Moved to viewpoint"
                            )

                        # Update agent_executor's history
                        if (
                            hasattr(self.target_agent, "agent_executor")
                            and hasattr(self.target_agent.agent_executor, "agent")
                            and hasattr(
                                self.target_agent.agent_executor.agent, "history"
                            )
                        ):
                            self.target_agent.agent_executor.agent.history.append(
                                history
                            )

                        # Record detail in trajectory
                        if len(self.target_agent.traj) > 0:
                            detail = {
                                "viewpointID": vp_id,
                                "turned_angle": "Moved to viewpoint",
                                "feature": new_obs.get("obs", ""),
                                "history": history,
                            }
                            if "details" not in self.target_agent.traj[0]:
                                self.target_agent.traj[0]["details"] = []
                            self.target_agent.traj[0]["details"].append(detail)

                        target_obs = np.array(self.target_agent.env._get_obs())
                        target_perm_obs = target_obs[perm_idx]
                        break  # batch_size=1 assumed
                else:
                    # All actions are stop
                    target_obs = np.array(self.target_agent.env._get_obs())
                target_perm_obs = target_obs[perm_idx]

                previous_angle = [
                    {"heading": ob["heading"], "elevation": ob["elevation"]}
                    for ob in target_perm_obs
                ]

            cpu_a_t = np.array(target_action_surr)
            # print("cpu_a_t", cpu_a_t)
            for i, next_id in enumerate(cpu_a_t):
                if (
                    next_id == (candidate_leng[i] - 1)
                    or next_id == args.ignoreid
                    or ended[i]
                ):  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            # print("cpu_a_t", cpu_a_t)
            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]  # Perm the obs for the resu

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break
        # end for

        return traj[0]

    def rollout_mask_test_navgpt_feature_phase_update_obs(
        self, test_model="IG", mode="ins", reset=True, perturb_ratio=0.25
    ):
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
        self.instr_buffer = [
            [] for _ in range(batch_size)
        ]  # to identify which vp's index has been used

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

        # Initialization the tracking state
        ended = np.array(
            [False] * batch_size
        )  # Indices match permuation of the model, not env

        # baseline agent init --------------------------
        if self.target_agent is not None:
            # Initialize NavGPT agent
            self.target_agent.init_trajecotry(target_perm_obs)
            # Load the instruction for NavGPT
            instructions = [ob["instruction"] for ob in target_perm_obs]
            if self.target_agent.config.load_instruction:
                action_plans = instructions
            elif self.target_agent.config.load_action_plan:
                action_plans = [ob["action_plan"] for ob in target_perm_obs]
            else:
                action_plans = []
                for instruction in instructions:
                    action_plan = self.target_agent.plan_chain.run(
                        instruction=instruction
                    )
                    action_plans.append(action_plan)
            # Set action plan for first observation (batch_size=1 assumed)
            if len(target_perm_obs) > 0:
                self.target_agent.cur_action_plan = action_plans[0]

            # Initialize accumulated_intermediate_steps for NavGPT agent context
            if not hasattr(self.target_agent, "_accumulated_intermediate_steps"):
                self.target_agent._accumulated_intermediate_steps = []

            # Initialize last_observation for tool_chain mode
            if self.target_agent.config.use_tool_chain:
                if not hasattr(self.target_agent, "_last_observation"):
                    self.target_agent._last_observation = None

            target_traj = [
                {
                    "instr_id": ob["instr_id"],
                    "path": [[ob["viewpoint"]]],
                    "details": {},
                    "a_t": {},
                    "uncertainty": {},
                    "probs": {},
                }
                for ob in target_perm_obs
            ]
            # print(target_traj[0]["instr_id"])
            # Initialization the tracking state
            target_ended = np.array([False] * batch_size)
            target_just_ended = np.array([False] * batch_size)

            previous_angle = [
                {"heading": ob["heading"], "elevation": ob["elevation"]}
                for ob in target_perm_obs
            ]
        else:
            print("cannot find target agent")
            exit(0)

        for t in range(self.episode_len):
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            images, candidata_list = self.exp.get_images_and_candidata_list(perm_obs)

            instr_id = perm_obs[0]["instr_id"]

            # load the ensemble saliency map and rank
            # agent_id = (
            #     "ensemble_hard.npy"
            #     if self.ensemble_mode == "hard_vote"
            #     else "ensemble.npy"
            # )
            agent_id = "ensemble" + f"_{args.bagging_agents}" + ".npy"
            attr_map = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    f"{instr_id}",
                    f"{t}",
                    "attr_map",
                    agent_id,
                )
            )
            attr_rank = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    f"{instr_id}",
                    f"{t}",
                    "attr_rank",
                    agent_id,
                )
            )

            viewpoint_id = get_navgpt_viewpoint_id_from_file(
                perm_obs[0]["instr_id"], t, target_perm_obs[0].get("candidate", {})
            )

            # print("viewpoint_id", viewpoint_id)
            candidate_list_surr = perm_obs[0].get("candidate", [])
            if viewpoint_id and viewpoint_id in [
                x["viewpointId"] for x in candidate_list_surr
            ]:
                target_action_surr = [
                    [x["viewpointId"] for x in candidate_list_surr].index(viewpoint_id)
                ]
                # print("target_action_surr", target_action_surr)
            else:
                target_action_surr = [len(candidate_list_surr)]  # Stop action

            candidata_dict = target_perm_obs[0].get("candidate", {})
            if viewpoint_id and viewpoint_id in candidata_dict.keys():
                target_action = [list(candidata_dict.keys()).index(viewpoint_id) + 1]
                # print("target_action", target_action)
            else:
                target_action = [0]  # Stop action
            # print("candidata_dict", candidata_dict.keys())
            # print(
            #     "candidate_list_surr", [x["viewpointId"] for x in candidate_list_surr]
            # )

            # NOTE: don't need for generating text, keep for simple
            params = (
                self.target_agent,
                target_perm_obs,
                t,
                previous_angle,
                True,  # NOTE: do_inference - may need to be False if reading from file
                target_ended,
                perm_obs,
            )
            description_update = self.causual.average_drop_navgpt_gentext(
                img=images[0],
                mask=attr_map,
                mask_rank=attr_rank,
                # cls_idx=target_action[0],
                cls_idx=None,
                params=params,
                mode=mode,
                mask_perc=perturb_ratio,
                # topK=5,
                candidate_idx=candidata_list[0],
                causal_metric_dir=self.causal_metric_dir,
            )

            # save the description update to file
            # construct by {instr_id}/{t}/{mask_perc}/{ins or del}/description_update.json
            description_update_dir = os.path.join(
                self.description_update_dir,
                f"{instr_id}",
                f"{t}",
                f"{perturb_ratio}",
                mode,
            )
            if not os.path.exists(description_update_dir):
                os.makedirs(description_update_dir)
            description_file_name = (
                "description_update" + f"_{args.bagging_agents}" + ".json"
            )
            description_update_file = os.path.join(
                description_update_dir, description_file_name
            )
            with open(description_update_file, "w") as f:
                json.dump(description_update, f, indent=4)

            # get the real action
            # for target agent---------------------------
            if self.target_agent is not None:
                for i in range(batch_size):
                    target_traj[i]["a_t"][t] = target_action[i]

                # Determine stop actions
                target_a_t_stop = [a_t_i == 0 for a_t_i in target_action]

                # Prepare environment action
                # NavGPT uses viewpoint IDs
                target_cpu_a_t = []
                for i in range(batch_size):
                    if target_a_t_stop[i] or target_ended[i]:
                        target_cpu_a_t.append(None)  # Stop action for NavGPT
                        target_just_ended[i] = True
                    else:
                        target_cpu_a_t.append(viewpoint_id)

                # Execute action in NavGPT's environment
                # NOTE: We read action from file, so we execute it directly
                for i, vp_id in enumerate(target_cpu_a_t):
                    if vp_id is not None:
                        # Execute the action
                        _, new_obs = self.target_agent.make_equiv_action([vp_id])

                        # Update history for NavGPT (similar to rollout_mask_navgpt)
                        if self.target_agent.config.use_history_chain:
                            new_feature = new_obs.get("obs", "")
                            new_feature_sum = new_obs.get("obs_summary", "")
                            if hasattr(self.target_agent, "history_chain"):
                                history = self.target_agent.history_chain.run(
                                    observation=new_feature_sum,
                                    history=(
                                        self.target_agent.agent_executor.agent.history[
                                            -1
                                        ]
                                        if len(
                                            self.target_agent.agent_executor.agent.history
                                        )
                                        > 0
                                        else ""
                                    ),
                                    previous_action="Moved to viewpoint",
                                )
                            else:
                                history = self.target_agent.get_history(
                                    new_obs, "Moved to viewpoint"
                                )
                        else:
                            history = self.target_agent.get_history(
                                new_obs, "Moved to viewpoint"
                            )

                        # Update agent_executor's history
                        if (
                            hasattr(self.target_agent, "agent_executor")
                            and hasattr(self.target_agent.agent_executor, "agent")
                            and hasattr(
                                self.target_agent.agent_executor.agent, "history"
                            )
                        ):
                            self.target_agent.agent_executor.agent.history.append(
                                history
                            )

                        # Record detail in trajectory
                        if len(self.target_agent.traj) > 0:
                            detail = {
                                "viewpointID": vp_id,
                                "turned_angle": "Moved to viewpoint",
                                "feature": new_obs.get("obs", ""),
                                "history": history,
                            }
                            if "details" not in self.target_agent.traj[0]:
                                self.target_agent.traj[0]["details"] = []
                            self.target_agent.traj[0]["details"].append(detail)

                        target_obs = np.array(self.target_agent.env._get_obs())
                        target_perm_obs = target_obs[perm_idx]
                        break  # batch_size=1 assumed
                else:
                    # All actions are stop
                    target_obs = np.array(self.target_agent.env._get_obs())
                target_perm_obs = target_obs[perm_idx]

                previous_angle = [
                    {"heading": ob["heading"], "elevation": ob["elevation"]}
                    for ob in target_perm_obs
                ]

            # ############### end of get new obs###########################
            cpu_a_t = np.array(target_action_surr)
            # print("cpu_a_t", cpu_a_t)
            for i, next_id in enumerate(cpu_a_t):
                if (
                    next_id == (candidate_leng[i] - 1)
                    or next_id == args.ignoreid
                    or ended[i]
                ):  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            # print("cpu_a_t", cpu_a_t)
            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]  # Perm the obs for the resu

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break
        # end for

        return traj[0]

    def rollout_mask_test_navgpt_feature_ensemble(
        self, test_model="IG", mode="ins", reset=True, perturb_ratio=0.25
    ):
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
        self.instr_buffer = [
            [] for _ in range(batch_size)
        ]  # to identify which vp's index has been used

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

        # Initialization the tracking state
        ended = np.array(
            [False] * batch_size
        )  # Indices match permuation of the model, not env

        # baseline agent init --------------------------
        if self.target_agent is not None:
            # Initialize NavGPT agent
            self.target_agent.init_trajecotry(target_perm_obs)
            # Load the instruction for NavGPT
            instructions = [ob["instruction"] for ob in target_perm_obs]
            if self.target_agent.config.load_instruction:
                action_plans = instructions
            elif self.target_agent.config.load_action_plan:
                action_plans = [ob["action_plan"] for ob in target_perm_obs]
            else:
                action_plans = []
                for instruction in instructions:
                    action_plan = self.target_agent.plan_chain.run(
                        instruction=instruction
                    )
                    action_plans.append(action_plan)
            # Set action plan for first observation (batch_size=1 assumed)
            if len(target_perm_obs) > 0:
                self.target_agent.cur_action_plan = action_plans[0]

            # Initialize accumulated_intermediate_steps for NavGPT agent context
            if not hasattr(self.target_agent, "_accumulated_intermediate_steps"):
                self.target_agent._accumulated_intermediate_steps = []

            # Initialize last_observation for tool_chain mode
            if self.target_agent.config.use_tool_chain:
                if not hasattr(self.target_agent, "_last_observation"):
                    self.target_agent._last_observation = None

            target_traj = [
                {
                    "instr_id": ob["instr_id"],
                    "path": [[ob["viewpoint"]]],
                    "details": {},
                    "a_t": {},
                    "uncertainty": {},
                    "probs": {},
                }
                for ob in target_perm_obs
            ]
            # print(target_traj[0]["instr_id"])
            # Initialization the tracking state
            target_ended = np.array([False] * batch_size)
            target_just_ended = np.array([False] * batch_size)

            previous_angle = [
                {"heading": ob["heading"], "elevation": ob["elevation"]}
                for ob in target_perm_obs
            ]
        else:
            print("cannot find target agent")
            exit(0)

        for t in range(self.episode_len):
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            images, candidata_list = self.exp.get_images_and_candidata_list(perm_obs)

            instr_id = perm_obs[0]["instr_id"]

            # Load the ensemble saliency map and rank
            # agent_id = (
            #     "ensemble_hard.npy"
            #     if self.ensemble_mode == "hard_vote"
            #     else "ensemble.npy"
            # )
            agent_id = "ensemble" + f"_{args.bagging_agents}" + ".npy"
            attr_map = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    f"{instr_id}",
                    f"{t}",
                    "attr_map",
                    agent_id,
                )
            )
            attr_rank = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    f"{instr_id}",
                    f"{t}",
                    "attr_rank",
                    agent_id,
                )
            )

            viewpoint_id = get_navgpt_viewpoint_id_from_file(
                perm_obs[0]["instr_id"], t, target_perm_obs[0].get("candidate", {})
            )

            # print("viewpoint_id", viewpoint_id)
            candidate_list_surr = perm_obs[0].get("candidate", [])
            if viewpoint_id and viewpoint_id in [
                x["viewpointId"] for x in candidate_list_surr
            ]:
                target_action_surr = [
                    [x["viewpointId"] for x in candidate_list_surr].index(viewpoint_id)
                ]
                # print("target_action_surr", target_action_surr)
            else:
                target_action_surr = [len(candidate_list_surr)]  # Stop action

            candidata_dict = target_perm_obs[0].get("candidate", {})
            if viewpoint_id and viewpoint_id in candidata_dict.keys():
                target_action = [list(candidata_dict.keys()).index(viewpoint_id) + 1]
                # print("target_action", target_action)
            else:
                target_action = [0]  # Stop action

            # NOTE: don't need for generating text, keep for simple
            params = (
                self.target_agent,
                target_perm_obs,
                t,
                previous_angle,
                True,  # NOTE: do_inference - may need to be False if reading from file
                target_ended,
                perm_obs,
            )
            self.causual.average_drop_navgpt_inference(
                img=images[0],
                mask=attr_map,
                mask_rank=attr_rank,
                # cls_idx=target_action[0],
                cls_idx=viewpoint_id,
                params=params,
                mode=mode,
                mask_perc=perturb_ratio,
                # topK=5,
                candidate_idx=candidata_list[0],
                causal_metric_dir=self.causal_metric_dir,
                description_update_dir=self.description_update_dir,
            )

            # get the real action
            # for target agent---------------------------
            if self.target_agent is not None:
                for i in range(batch_size):
                    target_traj[i]["a_t"][t] = target_action[i]

                # Determine stop actions
                target_a_t_stop = [a_t_i == 0 for a_t_i in target_action]

                # Prepare environment action
                # NavGPT uses viewpoint IDs
                target_cpu_a_t = []
                for i in range(batch_size):
                    if target_a_t_stop[i] or target_ended[i]:
                        target_cpu_a_t.append(None)  # Stop action for NavGPT
                        target_just_ended[i] = True
                    else:
                        target_cpu_a_t.append(viewpoint_id)

                # Execute action in NavGPT's environment
                # NOTE: We read action from file, so we execute it directly
                for i, vp_id in enumerate(target_cpu_a_t):
                    if vp_id is not None:
                        # Execute the action
                        _, new_obs = self.target_agent.make_equiv_action([vp_id])

                        # Update history for NavGPT (similar to rollout_mask_navgpt)
                        if self.target_agent.config.use_history_chain:
                            new_feature = new_obs.get("obs", "")
                            new_feature_sum = new_obs.get("obs_summary", "")
                            if hasattr(self.target_agent, "history_chain"):
                                history = self.target_agent.history_chain.run(
                                    observation=new_feature_sum,
                                    history=(
                                        self.target_agent.agent_executor.agent.history[
                                            -1
                                        ]
                                        if len(
                                            self.target_agent.agent_executor.agent.history
                                        )
                                        > 0
                                        else ""
                                    ),
                                    previous_action="Moved to viewpoint",
                                )
                            else:
                                history = self.target_agent.get_history(
                                    new_obs, "Moved to viewpoint"
                                )
                        else:
                            history = self.target_agent.get_history(
                                new_obs, "Moved to viewpoint"
                            )

                        # Update agent_executor's history
                        if (
                            hasattr(self.target_agent, "agent_executor")
                            and hasattr(self.target_agent.agent_executor, "agent")
                            and hasattr(
                                self.target_agent.agent_executor.agent, "history"
                            )
                        ):
                            self.target_agent.agent_executor.agent.history.append(
                                history
                            )

                        # Record detail in trajectory
                        if len(self.target_agent.traj) > 0:
                            detail = {
                                "viewpointID": vp_id,
                                "turned_angle": "Moved to viewpoint",
                                "feature": new_obs.get("obs", ""),
                                "history": history,
                            }
                            if "details" not in self.target_agent.traj[0]:
                                self.target_agent.traj[0]["details"] = []
                            self.target_agent.traj[0]["details"].append(detail)

                        target_obs = np.array(self.target_agent.env._get_obs())
                        target_perm_obs = target_obs[perm_idx]
                        break  # batch_size=1 assumed
                else:
                    # All actions are stop
                    target_obs = np.array(self.target_agent.env._get_obs())
                target_perm_obs = target_obs[perm_idx]

                previous_angle = [
                    {"heading": ob["heading"], "elevation": ob["elevation"]}
                    for ob in target_perm_obs
                ]

            # ############### end of get new obs###########################
            cpu_a_t = np.array(target_action_surr)
            # print("cpu_a_t", cpu_a_t)
            for i, next_id in enumerate(cpu_a_t):
                if (
                    next_id == (candidate_leng[i] - 1)
                    or next_id == args.ignoreid
                    or ended[i]
                ):  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            # print("cpu_a_t", cpu_a_t)
            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]  # Perm the obs for the resu

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break
        # end for

        return traj[0]

    def ensemble_saliency_map(self, mode="soft_vote"):
        """
        Ensemble the saliency maps from different agents with the specified mode.
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
            for t in os.listdir(instr_dir):
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
                    saliency_map = np.load(attr_map_path)
                    attr_rank = np.load(attr_rank_path)

                    attr_map_list.append(saliency_map)
                    attr_rank_list.append(attr_rank)
                # print("shape of attr_map", attr_map_list[0].shape)
                # print("shape of attr_rank", attr_rank_list[0].shape)
                # exit()
                if mode in ["soft_vote", "average"]:
                    ensemble_map, ensemble_rank = soft_vote(attr_map_list)
                elif mode in ["hard_vote", "vote"]:
                    ensemble_map, ensemble_rank = hard_vote(attr_rank_list)
                else:
                    raise ValueError("Unknown ensemble mode: {}".format(mode))
                # save ensemble map and ensemble rank to local disk
                file_name = (
                    "ensemble_hard.npy" if mode == "hard_vote" else "ensemble.npy"
                )
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

    def test(self, iters=None, **kwargs):
        # test_model = args.feature_level_baseline
        # assert test_model is not None, "test_model cannot be None"
        phase1 = False  # rollout per agent
        phase2 = False  # ensemble saliency maps
        phase3 = False  # test with ensemble
        phase_update_obs = False  # update observation using ensemble saliency maps
        if args.update_inference == "heatmap":
            phase1 = True
        elif args.update_inference == "update":
            phase2 = True
            phase_update_obs = True
        elif args.update_inference == "inference":
            phase3 = True
        mu = True
        self.env.reset_epoch(iters is not None)
        self.results = {}
        looped = False
        while True:
            print("new rollout")
            traj = self.rollout_navgpt_inference_repeat()
            if traj["instr_id"] in self.results:
                looped = True
            else:
                self.results[traj["instr_id"]] = traj["path"]
            if looped:
                break
        exit(0)

        if phase1:
            self.env.reset_epoch(
                shuffle=(iters is not None)
            )  # If iters is not none, shuffle the env batch
            self.losses = []
            self.results = {}
            # We rely on env showing the entire batch before repeating anything
            # phase 1: rollout the trajectory for each agent---------------------------------
            # rollout the trajectory for each agent
            # collect segenment map and saliency map
            # save to local disk
            if iters is not None:
                for agent_id in self.agents_id_list:
                    for i in range(iters):
                        self.rollout_mask_test_navgpt_feature_per_agent(
                            # test_model=test_model,
                            reset=True,
                            agent_id=agent_id,
                        )
            else:
                for agent_id in self.agents_id_list:
                    print(f"Rollout the trajectory for agent {agent_id}")
                    self.env.reset_epoch(shuffle=False)
                    self.results = {}
                    looped = False
                    # load checkpoint for agent i
                    self.load(
                        os.path.join(
                            args.load, f"agent_{agent_id}", "best_val72_navgpt"
                        )
                    )
                    while True:
                        traj = self.rollout_mask_test_navgpt_feature_per_agent(
                            # test_model=test_model,
                            reset=True,
                            agent_id=agent_id,
                        )
                        if traj["instr_id"] in self.results:
                            looped = True
                        else:
                            self.results[traj["instr_id"]] = traj["path"]
                        if looped:
                            break

        if phase2:
            # phase 2: ensemble the saliency map---------------------------------
            self.ensemble_saliency_map(mode=self.ensemble_mode)

        if phase_update_obs:
            self.env.reset_epoch(
                shuffle=(iters is not None)
            )  # If iters is not none, shuffle the env batch
            self.losses = []
            self.results = {}
            # We rely on env showing the entire batch before repeating anything
            looped = False
            self.loss = 0
            while True:
                print("new rollout")
                # we use 4 groups sample
                traj = self.rollout_mask_test_navgpt_feature_phase_update_obs(
                    # test_model=test_model,
                    mode="ins",
                    reset=True,
                    perturb_ratio=0.25,
                )
                traj = self.rollout_mask_test_navgpt_feature_phase_update_obs(
                    # test_model=test_model,
                    mode="del",
                    reset=False,
                    perturb_ratio=0.25,
                )
                traj = self.rollout_mask_test_navgpt_feature_phase_update_obs(
                    # test_model=test_model,
                    mode="ins",
                    reset=False,
                    perturb_ratio=0.5,
                )
                traj = self.rollout_mask_test_navgpt_feature_phase_update_obs(
                    # test_model=test_model,
                    mode="del",
                    reset=False,
                    perturb_ratio=0.5,
                )
                traj = self.rollout_mask_test_navgpt_feature_phase_update_obs(
                    # test_model=test_model,
                    mode="ins",
                    reset=False,
                    perturb_ratio=0.75,
                )
                traj = self.rollout_mask_test_navgpt_feature_phase_update_obs(
                    # test_model=test_model,
                    mode="del",
                    reset=False,
                    perturb_ratio=0.75,
                )
                if traj["instr_id"] in self.results:
                    looped = True
                else:
                    self.results[traj["instr_id"]] = traj["path"]
                if looped:
                    break

        if phase3:
            # phase 3: rollout the trajectory for the ensemble model---------------------------------
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
                    traj = self.rollout_mask_test_navgpt_feature_ensemble(
                        # test_model=test_model,
                        mode="ins",
                        reset=True,
                        perturb_ratio=0.25,
                    )
                    self.loss = 0
                    self.results[traj["instr_id"]] = traj["path"]
            else:  # Do a full round
                lets_start = False
                while True:
                    # # we use 4 groups sample
                    traj = self.rollout_mask_test_navgpt_feature_ensemble(
                        # test_model=test_model,
                        mode="ins",
                        reset=True,
                        perturb_ratio=0.25,
                    )
                    traj = self.rollout_mask_test_navgpt_feature_ensemble(
                        mode="del",
                        reset=False,
                        # reset=True,
                        perturb_ratio=0.25,
                    )
                    traj = self.rollout_mask_test_navgpt_feature_ensemble(
                        mode="ins",
                        reset=False,
                        perturb_ratio=0.5,
                    )
                    traj = self.rollout_mask_test_navgpt_feature_ensemble(
                        mode="del",
                        reset=False,
                        perturb_ratio=0.5,
                    )
                    traj = self.rollout_mask_test_navgpt_feature_ensemble(
                        mode="ins",
                        reset=False,
                        perturb_ratio=0.75,
                    )
                    traj = self.rollout_mask_test_navgpt_feature_ensemble(
                        mode="del",
                        reset=False,
                        perturb_ratio=0.75,
                    )
                    if traj["instr_id"] in self.results:
                        looped = True
                    else:
                        self.results[traj["instr_id"]] = traj["path"]
                    if looped:
                        break

        if mu:
            muFidelity = self.causual.compute_muFidelity(self.causal_metric_dir)
            print("muFidelity", muFidelity)

    def rollout_navgpt_inference_repeat(self, reset=False, num_repeats=5):
        """
        重复调用rollout进行实际导航推理，统计每个时间步t处选择最多的action。

        按照get_navgpt_viewpoint_id_from_file的标准vp进行实际导航（从文件读取viewpoint_id执行），
        但在每个时间步进行实际的LLM推理（使用NavGPT_genAction_v2），记录推理出的action。
        统计5次rollout中每个t处推理出的action，找出选择最多的。

        Args:
            reset: 是否重置环境
            num_repeats: 重复rollout的次数，默认5次

        Returns:
            dict: 统计结果，格式为 {t: {"most_frequent_action": viewpoint_id, "action_counts": {...}}}
        """
        from collections import Counter, defaultdict

        # 存储每次rollout中每个时间步的推理action选择
        all_inference_actions = defaultdict(list)  # {t: [action1, action2, ...]}

        for repeat_idx in range(num_repeats):
            print(f"\n========== Repeat {repeat_idx + 1}/{num_repeats} ==========")

            # 重置环境
            if reset or repeat_idx == 0:
                obs = np.array(self.env.reset_test())
                target_obs = np.array(
                    self.target_agent.env.set_scan_viewpoint_heading(
                        self.env.get_scan_viewpoint_heading()
                    )
                )
            else:
                # 恢复到起始点
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

            # Initialization the tracking state
            ended = np.array([False] * batch_size)

            # 清零并初始化NavGPT agent状态
            if self.target_agent is not None:
                # 重置agent状态
                self.target_agent.init_trajecotry(target_perm_obs)

                # 清零accumulated_intermediate_steps
                if not hasattr(self.target_agent, "_accumulated_intermediate_steps"):
                    self.target_agent._accumulated_intermediate_steps = []
                else:
                    self.target_agent._accumulated_intermediate_steps = []

                # 清零last_observation
                if self.target_agent.config.use_tool_chain:
                    if not hasattr(self.target_agent, "_last_observation"):
                        self.target_agent._last_observation = None
                    else:
                        self.target_agent._last_observation = None

                # 清零agent_executor的history和intermediate_steps
                if (
                    hasattr(self.target_agent, "agent_executor")
                    and hasattr(self.target_agent.agent_executor, "agent")
                    and hasattr(self.target_agent.agent_executor.agent, "history")
                ):
                    self.target_agent.agent_executor.agent.history = []

                if hasattr(self.target_agent, "agent_executor") and hasattr(
                    self.target_agent.agent_executor, "intermediate_steps"
                ):
                    self.target_agent.agent_executor.intermediate_steps = []

                # Load the instruction for NavGPT
                instructions = [ob["instruction"] for ob in target_perm_obs]
                if self.target_agent.config.load_instruction:
                    action_plans = instructions
                elif self.target_agent.config.load_action_plan:
                    action_plans = [ob["action_plan"] for ob in target_perm_obs]
                else:
                    action_plans = []
                    for instruction in instructions:
                        action_plan = self.target_agent.plan_chain.run(
                            instruction=instruction
                        )
                        action_plans.append(action_plan)

                # Set action plan for first observation (batch_size=1 assumed)
                if len(target_perm_obs) > 0:
                    self.target_agent.cur_action_plan = action_plans[0]

                previous_angle = [
                    {"heading": ob["heading"], "elevation": ob["elevation"]}
                    for ob in target_perm_obs
                ]
            else:
                print("cannot find target agent")
                exit(0)

            # 进行rollout，按照文件中的标准vp导航，但记录推理结果
            for t in range(self.episode_len):
                input_a_t, candidate_feat, candidate_leng = self.get_input_feat(
                    perm_obs
                )

                # 使用NavGPT_genAction_v2进行实际推理（do_inference=True）
                # 注意：这里进行推理但不执行推理结果，而是执行文件中的标准vp
                a_t_inference, cand_nums, nav_inputs, viewpoint_ids_inference = (
                    NavGPT_genAction_v2(
                        agent=self.target_agent,
                        obs=target_perm_obs,
                        t=t,
                        previous_angle=previous_angle,
                        do_inference=True,  # 进行实际推理
                        ended=ended,
                        description_update=None,
                    )
                )

                # 获取推理得到的viewpoint_id（用于统计）
                # 直接从viewpoint_ids_inference获取，如果为空则从a_t_inference转换
                inference_viewpoint_id = None
                if len(viewpoint_ids_inference) > 0:
                    inference_viewpoint_id = viewpoint_ids_inference[0]
                else:
                    # Fallback: 从a_t_inference转换
                    candidate_dict = target_perm_obs[0].get("candidate", {})
                    if len(candidate_dict) > 0:
                        if a_t_inference[0] == 0:
                            inference_viewpoint_id = None  # Stop action
                        elif 1 <= a_t_inference[0] <= len(candidate_dict):
                            candidate_list = list(candidate_dict.keys())
                            inference_viewpoint_id = candidate_list[
                                a_t_inference[0] - 1
                            ]

                # 记录推理出的action（用于统计）
                all_inference_actions[t].append(inference_viewpoint_id)
                print(
                    f"  t={t}: inference_viewpoint_id={inference_viewpoint_id}, a_t_inference={a_t_inference[0]}"
                )

                # 从文件读取标准vp（用于实际导航）
                viewpoint_id_from_file = get_navgpt_viewpoint_id_from_file(
                    perm_obs[0]["instr_id"], t, target_perm_obs[0].get("candidate", {})
                )
                print(
                    f"  t={t}: viewpoint_id_from_file={viewpoint_id_from_file} (used for navigation)"
                )

                # 执行action（按照文件中的标准vp，而不是推理结果）
                if viewpoint_id_from_file is not None:
                    # 执行action
                    _, new_obs = self.target_agent.make_equiv_action(
                        [viewpoint_id_from_file]
                    )

                    # 更新history
                    if self.target_agent.config.use_history_chain:
                        new_feature = new_obs.get("obs", "")
                        new_feature_sum = new_obs.get("obs_summary", "")
                        if hasattr(self.target_agent, "history_chain"):
                            history = self.target_agent.history_chain.run(
                                observation=new_feature_sum,
                                history=(
                                    self.target_agent.agent_executor.agent.history[-1]
                                    if len(
                                        self.target_agent.agent_executor.agent.history
                                    )
                                    > 0
                                    else ""
                                ),
                                previous_action="Moved to viewpoint",
                            )
                        else:
                            history = self.target_agent.get_history(
                                new_obs, "Moved to viewpoint"
                            )
                    else:
                        history = self.target_agent.get_history(
                            new_obs, "Moved to viewpoint"
                        )

                    # 更新agent_executor的history
                    if (
                        hasattr(self.target_agent, "agent_executor")
                        and hasattr(self.target_agent.agent_executor, "agent")
                        and hasattr(self.target_agent.agent_executor.agent, "history")
                    ):
                        self.target_agent.agent_executor.agent.history.append(history)

                    # 更新observation
                    target_obs = np.array(self.target_agent.env._get_obs())
                    target_perm_obs = target_obs[perm_idx]
                else:
                    # Stop action
                    ended[0] = True
                    break

                # 更新surrogate agent的observation（用于下一次get_input_feat）
                candidate_list_surr = perm_obs[0].get("candidate", [])
                if viewpoint_id_from_file and viewpoint_id_from_file in [
                    x["viewpointId"] for x in candidate_list_surr
                ]:
                    target_action_surr = [
                        [x["viewpointId"] for x in candidate_list_surr].index(
                            viewpoint_id_from_file
                        )
                    ]
                else:
                    target_action_surr = [len(candidate_list_surr)]  # Stop action

                cpu_a_t = np.array(target_action_surr)
                for i, next_id in enumerate(cpu_a_t):
                    if (
                        next_id == (candidate_leng[i] - 1)
                        or next_id == args.ignoreid
                        or ended[i]
                    ):
                        cpu_a_t[i] = -1

                # 执行surrogate agent的action
                self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
                obs = np.array(self.env._get_obs())
                perm_obs = obs[perm_idx]

                # 更新previous_angle
                previous_angle = [
                    {"heading": ob["heading"], "elevation": ob["elevation"]}
                    for ob in target_perm_obs
                ]

                # 更新ended状态
                ended[:] = np.logical_or(ended, (cpu_a_t == -1))

                # Early exit if all ended
                if ended.all():
                    break

            print(
                f"Repeat {repeat_idx + 1} completed. Trajectory length: {len([k for k in all_inference_actions.keys() if all_inference_actions[k]])}"
            )

        # 统计每个时间步t处选择最多的推理action
        statistics = {}
        for t in sorted(all_inference_actions.keys()):
            actions_at_t = all_inference_actions[t]
            action_counts = Counter(actions_at_t)
            most_common = action_counts.most_common(1)

            if most_common:
                most_frequent_action, count = most_common[0]
                statistics[t] = {
                    "most_frequent_action": most_frequent_action,
                    "count": count,
                    "total": len(actions_at_t),
                    "action_counts": dict(action_counts),
                }
            else:
                statistics[t] = {
                    "most_frequent_action": None,
                    "count": 0,
                    "total": len(actions_at_t),
                    "action_counts": {},
                }

        # 打印统计结果
        print("\n" + "=" * 60)
        print("Statistics: Most frequent inference action at each time step t")
        print("=" * 60)
        for t in sorted(statistics.keys()):
            stat = statistics[t]
            print(f"t={t}:")
            print(
                f"  Most frequent: {stat['most_frequent_action']} (appeared {stat['count']}/{stat['total']} times)"
            )
            print(f"  All actions: {stat['action_counts']}")

        # return statistics
        return traj[0]
