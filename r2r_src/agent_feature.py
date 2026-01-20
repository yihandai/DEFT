from typing import Tuple
from collections import Counter

# from agent import Seq2SeqAgent
from agent_mask import MaskAgent
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
import json
import r2r_src.vln_utils as vln_utils

if args.target_agent == "MapGPT":
    from MapGPT.vln.gpt_agent import GPTNavAgent
    from MapGPT.GPT.one_stage_prompt_manager import OneStagePromptManager
    from MapGPT.GPT.api import gpt_infer
    from vlnbert.IG_utils import Exp
    from vlnbert.XRAI import XRAI, extract_object_masks_yolo
    from vlnbert.feature_level_eval import CausalMetric, NpImage
    from r2r_src.vlnbert.smdl.submodular_cub_v2_pytorch import (
        CubSubModularExplanationV2,
    )
# else:
#     from NavGPT.nav_src.agent import NavAgent

nav_inputs = []


def MapGPT_genAction(
    agent,
    obs,
    t,
    previous_angle,
    do_inference=True,
    ended=None,
    new_imgs=None,
    candidata_list=None,
    max_retries=3,
    retry_delay=1.0,  # seconds
    instr_id=None,
    perc=None,
    mode=None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    a_t = np.zeros(len(obs), dtype=np.int32)
    cand_nums = np.zeros(len(obs), dtype=np.int32)

    global nav_inputs
    nav_inputs = []

    np_image_manager = NpImage()
    for i, ob in enumerate(obs):
        cand_inputs = agent.prompt_managers[i].make_action_prompt(
            [ob], [previous_angle[i]]
        )
        if agent.args.response_format == "str":
            nav_input = agent.prompt_managers[i].make_r2r_prompts(
                cand_inputs=cand_inputs, obs=[ob], t=t
            )
        elif agent.args.response_format == "json":
            nav_input = agent.prompt_managers[i].make_r2r_json_prompts(
                cand_inputs=cand_inputs, obs=[ob], t=t
            )
        else:
            raise NotImplementedError

        image_list = agent.prompt_managers[i].node_imgs[0]
        environment_prompts = nav_input["prompts"][0]
        # print("-------------------- Environment Prompts --------------------")
        # print(environment_prompts)

        if new_imgs is not None and candidata_list is not None:
            image_list = np_image_manager.save_np2file(
                new_imgs[i],
                ob,
                candidata_list[i],
                instr_id=instr_id,
                t=t,
                perc=perc,
                mode=mode,
            )
            # print(image_list)

        if (
            agent.args.llm == "gpt-4o-2024-05-13" or agent.args.llm == "gpt-4o"
        ) and agent.args.response_format == "json":
            if len(image_list) > 20 or not do_inference or ended[i]:
                # GPT-4o currently does not support queries with more than 20 images
                a_t[i] = 0
                cand_nums[i] = len(nav_input["action_options"])
                nav_inputs.append(nav_input)
                # print("Exceed image limit and stop!")
            else:
                if do_inference:
                    for attempt in range(max_retries):
                        try:
                            nav_output, tokens = gpt_infer(
                                nav_input["task_description"],
                                environment_prompts,
                                image_list,
                                agent.args.llm,
                                agent.args.max_tokens,
                                response_format={"type": "json_object"},
                            )
                            json_output = json.loads(nav_output)  # may fail
                            break  # success → exit loop
                        except json.JSONDecodeError as e:
                            print(
                                f"[Retry {attempt+1}/{max_retries}] JSON parsing failed: {e}"
                            )
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)  # wait before retry
                            else:
                                raise RuntimeError(
                                    "Failed to parse valid JSON after retries"
                                )
                    a_t_i, action_t = agent.prompt_managers[i].parse_json_action(
                        json_output, nav_input["only_options"], t
                    )
                    agent.prompt_managers[i].parse_json_planning(json_output)
                    # print("-------------------- Output --------------------")
                    # print(nav_output)
                else:
                    a_t_i = [0]
                a_t[i] = a_t_i[0]
                cand_nums[i] = len(nav_input["action_options"])
                nav_inputs.append(nav_input)
                # probs_t = agent.prompt_manager.parse_probs(probs, action_t)
                # uncertainty_t = compute_entropy(np.array(probs_t)) # return 0 if probs_t is empty []

        else:
            raise NotImplementedError
        # if new_imgs is not None and candidata_list is not None:
        #     np_image_manager.delete_images(image_list)
    return a_t, cand_nums, nav_inputs


def read_nav_info():
    # with open(os.path.join("MapGPT", "nav_30", "all_nav_outputs.json"), "r") as f:
    with open(os.path.join("MapGPT", "nav_30", "all_nav_outputs_new.json"), "r") as f:
        info = json.load(f)
    return info


def collect_nav_info(instr_id, t):
    info = read_nav_info()
    t = str(t)
    nav_info = {
        "nav_input_json": info[instr_id][t]["nav_input_json"],
        "nav_output_json": info[instr_id][t]["nav_output_json"],
        "img_list": info[instr_id][t]["img_list"],
        "vp": info[instr_id][t]["vp"],
        "a_t_list": info[instr_id][t]["a_t_list"],
    }
    return nav_info


def update_image_list(image_list, image_list_update, nav_input):
    only_actions = nav_input["only_actions"][0]
    image_list = [
        os.path.join("./MapGPT", image) if image is not None else None
        for image in image_list
    ]

    # parser number x from `to Image x`
    for i, only_action in enumerate(only_actions):
        assert "to Image" in only_action, "only_action must contain `to Image`"
        number = int(only_action.split("to Image")[1].strip())
        image_list[number] = image_list_update[i]

    return image_list


def MapGPT_genAction_v2(
    agent,
    obs,
    t,
    previous_angle,
    do_inference=True,
    ended=None,
    new_imgs=None,
    candidata_list=None,
    max_retries=3,
    retry_delay=1.0,  # seconds
    instr_id=None,
    perc=None,
    mode=None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    a_t = np.zeros(len(obs), dtype=np.int32)
    cand_nums = np.zeros(len(obs), dtype=np.int32)

    global nav_inputs

    # nav_inputs = []
    print("t: ", t)
    np_image_manager = NpImage()
    for i, ob in enumerate(obs):
        nav_info = collect_nav_info(ob["instr_id"], t)
        nav_input_json = nav_info["nav_input_json"]
        nav_input = nav_input_json
        # print("nav_input: ", nav_input)
        environment_prompts = nav_input["prompts"][0]
        image_list = nav_info["img_list"]
        # print("image_list: ", image_list)

        if new_imgs is not None and candidata_list is not None and do_inference:
            image_list_update = np_image_manager.save_np2file(
                new_imgs[i],
                ob,
                candidata_list[i],
                instr_id=instr_id,
                t=t,
                perc=perc,
                mode=mode,
            )
            # print(image_list)
            # print("image_list update: ", image_list_update)
            image_list = update_image_list(image_list, image_list_update, nav_input)
            # print("image_list after update: ", image_list)
        if (
            agent.args.llm == "gpt-4o-2024-05-13" or agent.args.llm == "gpt-4o"
        ) and agent.args.response_format == "json":
            if len(image_list) > 20 or not do_inference or ended[i]:
                # GPT-4o currently does not support queries with more than 20 images
                a_t[i] = 0
                cand_nums[i] = len(nav_input["action_options"])
                nav_inputs.append(nav_input)
                # print("Exceed image limit and stop!")
            else:
                if do_inference:
                    for attempt in range(max_retries):
                        try:
                            nav_output, tokens = gpt_infer(
                                nav_input["task_description"],
                                environment_prompts,
                                image_list,
                                agent.args.llm,
                                agent.args.max_tokens,
                                response_format={"type": "json_object"},
                            )
                            json_output = json.loads(nav_output)  # may fail
                            break  # success → exit loop
                        except (json.JSONDecodeError, TypeError) as e:
                            if isinstance(e, TypeError):
                                print(
                                    f"[Retry {attempt+1}/{max_retries}] TypeError: {e}"
                                )
                            else:
                                print(
                                    f"[Retry {attempt+1}/{max_retries}] JSON parsing failed: {e}"
                                )
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)  # wait before retry
                            else:
                                raise RuntimeError(
                                    "Failed to parse valid JSON after retries"
                                )
                    a_t_i, action_t = agent.prompt_managers[i].parse_json_action(
                        json_output, nav_input["only_options"], t
                    )
                    # agent.prompt_managers[i].parse_json_planning(json_output)
                else:
                    a_t_i = [0]
                a_t[i] = a_t_i[0]
                cand_nums[i] = len(nav_input["action_options"])
                nav_inputs.append(nav_input)
                # probs_t = agent.prompt_manager.parse_probs(probs, action_t)
                # uncertainty_t = compute_entropy(np.array(probs_t)) # return 0 if probs_t is empty []

        else:
            raise NotImplementedError
        # if new_imgs is not None and candidata_list is not None:
        #     np_image_manager.delete_images(image_list)
    return a_t, cand_nums, nav_inputs


class FeatureAgent(MaskAgent):
    def __init__(self, env, results_path, tok, episode_len=20, args_target=None):
        super(FeatureAgent, self).__init__(
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

        if args.target_agent == "MapGPT":
            self.target_agent = GPTNavAgent(args_target, env, rank=rank)
            self.target_agent.prompt_managers = [
                OneStagePromptManager(args_target) for i in range(args.batchSize)
            ]
            self.causual = CausalMetric(
                call_fn=MapGPT_genAction_v2,
                substrate_fn=np.zeros_like,
                H=480,
                W=640,
                target="MapGPT",
            )
            # saving locations
            self.segmentation_map_dir = os.path.join(
                "snap",
                "VLNBERT-train-feature-mapgpt-ensemble",
                "segmentation_map",
            )
            if not os.path.exists(self.segmentation_map_dir):
                os.makedirs(self.segmentation_map_dir)
            # saliency map location
            saliency_map_dir = os.path.join("snap", args.name, "saliency_map_pixel")
            if not os.path.exists(saliency_map_dir):
                os.makedirs(saliency_map_dir)
            self.saliency_map_dir = saliency_map_dir
            # causal metric location
            causal_metric_dir = os.path.join("snap", args.name, "causal_metric_pixel")
            if not os.path.exists(causal_metric_dir):
                os.makedirs(causal_metric_dir)
            self.causal_metric_dir = causal_metric_dir

    def test(self, iters=None, **kwargs):
        test_model = args.feature_level_baseline
        assert test_model is not None, "test_model cannot be None"
        phase2 = True
        phase3 = False
        phase_merge = False
        original_image = False
        mu = False

        if phase2:
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
                    for traj in self.rollout_mask_test_mapgpt_feature(
                        test_model=test_model
                    ):
                        self.loss = 0
                        self.results[traj["instr_id"]] = traj["path"]
            else:  # Do a full round
                lets_start = False
                while True:
                    # for i in range(test_num):
                    if args.target_agent == "MapGPT":
                        traj = self.rollout_mask_test_mapgpt_feature_phase2(
                            test_model=test_model,
                            reset=True,
                        )
                    if traj["instr_id"] in self.results:
                        looped = True
                    else:
                        self.results[traj["instr_id"]] = traj["path"]
                    if looped:
                        break

        if phase_merge and args.feature_level_baseline == "IG_temporal":
            self.merge_IG_temporal_saliency_map(mode="soft_vote")

        if phase3:
            self.env.reset_epoch(
                shuffle=(iters is not None)
            )  # If iters is not none, shuffle the env batch
            self.losses = []
            self.results = {}
            # We rely on env showing the entire batch before repeating anything
            looped = False
            self.loss = 0
            # count_i = 0
            # self.find_6992_0 = False
            while True:
                # for i in range(test_num):
                if args.target_agent == "MapGPT":
                    # we use 4 groups sample
                    traj = self.rollout_mask_test_mapgpt_feature_phase3(
                        # test_model="IG_temporal",
                        test_model=test_model,
                        mode="ins",
                        reset=True,
                        perturb_ratio=0.25,
                    )
                    traj = self.rollout_mask_test_mapgpt_feature_phase3(
                        # test_model="IG_temporal",
                        test_model=test_model,
                        mode="del",
                        reset=False,
                        perturb_ratio=0.25,
                    )
                    # count_i += 1
                    # if count_i > 5:
                    #     exit(0)
                    traj = self.rollout_mask_test_mapgpt_feature_phase3(
                        # test_model="IG_temporal",
                        test_model=test_model,
                        mode="ins",
                        reset=False,
                        perturb_ratio=0.5,
                    )
                    traj = self.rollout_mask_test_mapgpt_feature_phase3(
                        # test_model="IG_temporal",
                        test_model=test_model,
                        mode="del",
                        reset=False,
                        perturb_ratio=0.5,
                    )
                    traj = self.rollout_mask_test_mapgpt_feature_phase3(
                        # test_model="IG_temporal",
                        test_model=test_model,
                        mode="ins",
                        reset=False,
                        perturb_ratio=0.75,
                    )
                    traj = self.rollout_mask_test_mapgpt_feature_phase3(
                        # test_model="IG_temporal",
                        test_model=test_model,
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

        if original_image:
            self.env.reset_epoch(
                shuffle=(iters is not None)
            )  # If iters is not none, shuffle the env batch
            self.losses = []
            self.results = {}
            # We rely on env showing the entire batch before repeating anything
            looped = False
            self.loss = 0
            # count_i = 0
            while True:
                if args.target_agent == "MapGPT":
                    # we use 4 groups sample
                    traj = self.rollout_mask_test_mapgpt_feature_phase3(
                        # test_model="IG_temporal",
                        test_model=test_model,
                        # mode="del",
                        mode="ins",
                        reset=True,
                        perturb_ratio=1.0,
                    )
                    # traj = self.rollout_mask_test_mapgpt_feature_phase3(
                    #     # test_model="IG_temporal",
                    #     test_model=test_model,
                    #     mode="del",
                    #     reset=False,
                    #     perturb_ratio=0.0,
                    # )
                if traj["instr_id"] in self.results:
                    looped = True
                else:
                    self.results[traj["instr_id"]] = traj["path"]
                if looped:
                    break

    def rollout_mask_test_mapgpt_feature(
        self,
        test_model="IG",
        mode="ins",
        reset=True,
        perturb_ratio=0.25,  # perturb the saliency map by this ratio
    ):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

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
            for i, _ in enumerate(target_perm_obs):
                self.target_agent.prompt_managers[i].history = [""]
                self.target_agent.prompt_managers[i].nodes_list = [[]]
                self.target_agent.prompt_managers[i].node_imgs = [[]]
                self.target_agent.prompt_managers[i].graph = [{}]
                self.target_agent.prompt_managers[i].trajectory = [[]]
                self.target_agent.prompt_managers[i].planning = [
                    ["Navigation has just started, with no planning yet."]
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

            # if self.target_agent is not None:
            #     _, _, target_nav_inputs = MapGPT_genAction(
            #         self.target_agent,
            #         target_perm_obs,
            #         t,
            #         previous_angle,
            #         do_inference=False,
            #         ended=target_ended,
            #     )
            # do integrated gradients and generate attribution maps
            images, attribution, candidata_list = self.exp.compute_integrated_gradients(
                perm_obs,
                t,
                h_t,
                language_features,
                language_inputs,
                language_attention_mask,
                token_type_ids,
                mode=test_model,
            )

            if test_model in ["IG", "temporal", "IG_temporal"]:
                object_seg = extract_object_masks_yolo(
                    [
                        Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), mode="RGB")
                        for x in images[0, candidata_list[0]]
                    ]
                )

                XRAI_test = XRAI()
                attr_map, attr_rank = XRAI_test.GetMaskWithDetails(
                    images[0, candidata_list[0]],
                    object_seg,
                    attribution[0][candidata_list[0]],
                    candidata_idx=candidata_list[0],
                    obs=perm_obs[0],
                )
                # shape of attr_map is [len(candidate), H, W]

            target_action_surr = self._teacher_action_baseline(
                target_perm_obs, target_ended
            )
            target_action_surr = target_action_surr.cpu().numpy()
            target_action = self.action_space_adaptor(
                "RecVLN", "MapGPT", target_action_surr, candidate_leng
            )

            params = (
                self.target_agent,
                target_perm_obs,
                t,
                previous_angle,
                True,
                target_ended,
            )
            self.causual.average_drop2(
                img=images[0],
                mask=attr_map,
                mask_rank=attr_rank,
                # cls_idx=target_action[0],
                cls_idx=self.get_cls(perm_obs, t)[0],
                params=params,
                mode=mode,
                mask_perc=perturb_ratio,
                # topK=5,
                candidate_idx=candidata_list[0],
                causal_metric_dir=self.causal_metric_dir,
            )

            # 确定真实动作
            # NOTE: MapGPT 的 real action 里面 0 是停止，
            # NOTE: 在 RecVLN 里面，candidate_len[i] - 1 是停止
            # ############### get new obs###########################
            # for target agent---------------------------
            if self.target_agent is not None:
                for i in range(batch_size):
                    target_traj[i]["a_t"][t] = target_action[i]

                # Determine stop actions
                target_a_t_stop = [a_t_i == 0 for a_t_i in target_action]

                # Prepare environment action
                target_cpu_a_t = []
                for i in range(batch_size):
                    if target_a_t_stop[i] or target_ended[i]:
                        target_cpu_a_t.append(-1)
                        target_just_ended[i] = True
                    else:
                        target_cpu_a_t.append(target_action[i] - 1)

                self.target_agent.make_equiv_action(
                    target_cpu_a_t, target_perm_obs, target_traj, perm_idx
                )
                target_obs = np.array(self.target_agent.env._get_obs())
                target_perm_obs = target_obs[perm_idx]

                previous_angle = [
                    {"heading": ob["heading"], "elevation": ob["elevation"]}
                    for ob in target_perm_obs
                ]

                # # we only implement batch_size=1
                # if real_action[0] == 0:
                #     break

                # get global nav inputs
                target_nav_inputs = nav_inputs
                for i in range(len(target_perm_obs)):
                    self.target_agent.prompt_managers[i].make_history(
                        [target_action[i]], target_nav_inputs[i], t
                    )
                    self.target_agent.prompt_managers[i].modify_planning(
                        np.array([target_perm_obs[i]])
                    )

            # ############### end of get new obs###########################
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            # cpu_a_t = a_t.cpu().numpy()
            # cpu_a_t = real_action

            # real_action_surr = self.action_space_adaptor(
            #     "MapGPT", "RecVLN", real_action, candidate_leng
            # )
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

    def get_cls(self, obs, t):
        t = str(t)
        gts = []
        for ob in obs:
            instr_id = ob["instr_id"]
            nav_info = collect_nav_info(instr_id, t)
            # nav_output = nav_info["nav_output_json"]
            # gt_action = nav_output[0]
            a_t_list = nav_info["a_t_list"]
            # count the most frequent a_t and index in the list
            a_t_count = Counter(a_t_list)
            # 找到a_t_count中value最大的对应的key，其在a_t_list中的index
            most_frequent_a_t = a_t_count.most_common(1)[0][0]
            a_t_index = a_t_list.index(most_frequent_a_t)
            a_t = [a_t_list[a_t_index]]
            gts.append(a_t[0])
        return gts

    def collect_mapgpt_action_perstep(
        self,
        reset=True,
    ):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

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
            obs = np.array(self.env._get_obs())
            target_obs = np.array(
                self.target_agent.env.set_scan_viewpoint_heading(
                    self.env.get_scan_viewpoint_heading()
                )
            )
        traj_dict = {}
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

        # baseline agent init --------------------------
        if self.target_agent is not None:
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
            for i, _ in enumerate(target_perm_obs):
                self.target_agent.prompt_managers[i].history = [""]
                self.target_agent.prompt_managers[i].nodes_list = [[]]
                self.target_agent.prompt_managers[i].node_imgs = [[]]
                self.target_agent.prompt_managers[i].graph = [{}]
                self.target_agent.prompt_managers[i].trajectory = [[]]
                self.target_agent.prompt_managers[i].planning = [
                    ["Navigation has just started, with no planning yet."]
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

            if self.target_agent is not None:
                real_action, _, target_nav_inputs = MapGPT_genAction(
                    self.target_agent,
                    target_perm_obs,
                    t,
                    previous_angle,
                    do_inference=True,
                    ended=target_ended,
                )
            # do integrated gradients and generate attribution maps
            # convert to python int to avoid np.int32 error
            traj_dict[perm_obs[0]["viewpoint"]] = int(real_action[0])
            target_action_surr = self._teacher_action_baseline(
                target_perm_obs, target_ended
            )
            target_action_surr = target_action_surr.cpu().numpy()
            target_action = self.action_space_adaptor(
                "RecVLN", "MapGPT", target_action_surr, candidate_leng
            )

            # 确定真实动作
            # NOTE: MapGPT 的 real action 里面 0 是停止，
            # NOTE: 在 RecVLN 里面，candidate_len[i] - 1 是停止
            # ############### get new obs###########################
            # for target agent---------------------------
            if self.target_agent is not None:
                for i in range(batch_size):
                    target_traj[i]["a_t"][t] = target_action[i]

                # Determine stop actions
                target_a_t_stop = [a_t_i == 0 for a_t_i in target_action]

                # Prepare environment action
                target_cpu_a_t = []
                for i in range(batch_size):
                    if target_a_t_stop[i] or target_ended[i]:
                        target_cpu_a_t.append(-1)
                        target_just_ended[i] = True
                    else:
                        target_cpu_a_t.append(target_action[i] - 1)

                self.target_agent.make_equiv_action(
                    target_cpu_a_t, target_perm_obs, target_traj, perm_idx
                )
                target_obs = np.array(self.target_agent.env._get_obs())
                target_perm_obs = target_obs[perm_idx]

                previous_angle = [
                    {"heading": ob["heading"], "elevation": ob["elevation"]}
                    for ob in target_perm_obs
                ]

                # # we only implement batch_size=1
                # if real_action[0] == 0:
                #     break
                # get target_nav_inputs in the target agent
                for i in range(len(target_perm_obs)):
                    self.target_agent.prompt_managers[i].make_history(
                        [target_action[i]], target_nav_inputs[i], t
                    )
                    self.target_agent.prompt_managers[i].modify_planning(
                        np.array([target_perm_obs[i]])
                    )

            # ############### end of get new obs###########################
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            # cpu_a_t = a_t.cpu().numpy()
            # cpu_a_t = real_action

            # real_action_surr = self.action_space_adaptor(
            #     "MapGPT", "RecVLN", real_action, candidate_leng
            # )
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

        the_dir = os.path.join("tmp_traj")
        os.makedirs(the_dir, exist_ok=True)
        with open(os.path.join(the_dir, perm_obs[0]["instr_id"] + ".json"), "w") as f:
            json.dump(traj_dict, f)
        return traj[0]

    def get_gt(self, gt_file="mapgpt_feature.json"):
        with open(gt_file, "r") as f:
            data = json.load(f)
        return data

    def merge_json_files(self, input_dir="tmp_traj", output_file="mapgpt_feature.json"):
        merged = {}

        for fname in os.listdir(input_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(input_dir, fname)
                with open(fpath, "r") as f:
                    data = json.load(f)
                key = os.path.splitext(fname)[0]  # filename without .json
                merged[key] = data

        # dump to new json file
        with open(output_file, "w") as f:
            json.dump(merged, f, indent=4)

        return merged

    def rollout_mask_test_mapgpt_feature_phase2(
        self,
        test_model="IG",
        mode="ins",
        reset=True,
        expand_patch=False,
    ):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

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
            for i, _ in enumerate(target_perm_obs):
                self.target_agent.prompt_managers[i].history = [""]
                self.target_agent.prompt_managers[i].nodes_list = [[]]
                self.target_agent.prompt_managers[i].node_imgs = [[]]
                self.target_agent.prompt_managers[i].graph = [{}]
                self.target_agent.prompt_managers[i].trajectory = [[]]
                self.target_agent.prompt_managers[i].planning = [
                    ["Navigation has just started, with no planning yet."]
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
                    )  # torch.Tensor
                    # save the segmentation map by npy
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
                    # object_seg = torch.from_numpy(object_seg).to(images.device)
                    # object_seg = torch.from_numpy(object_seg)
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
            # shape of attr_map is [len(candidate), H, W]
            # save attr_map and attr_rank
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
            target_action_surr = self._teacher_action_baseline(
                target_perm_obs, target_ended
            )
            target_action_surr = target_action_surr.cpu().numpy()
            target_action = self.action_space_adaptor(
                "RecVLN", "MapGPT", target_action_surr, candidate_leng
            )

            # determine the real action
            if self.target_agent is not None:
                for i in range(batch_size):
                    target_traj[i]["a_t"][t] = target_action[i]

                # Determine stop actions
                target_a_t_stop = [a_t_i == 0 for a_t_i in target_action]

                # Prepare environment action
                target_cpu_a_t = []
                for i in range(batch_size):
                    if target_a_t_stop[i] or target_ended[i]:
                        target_cpu_a_t.append(-1)
                        target_just_ended[i] = True
                    else:
                        target_cpu_a_t.append(target_action[i] - 1)

                self.target_agent.make_equiv_action(
                    target_cpu_a_t, target_perm_obs, target_traj, perm_idx
                )
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

    def rollout_mask_test_mapgpt_feature_phase3(
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
            for i, _ in enumerate(target_perm_obs):
                self.target_agent.prompt_managers[i].history = [""]
                self.target_agent.prompt_managers[i].nodes_list = [[]]
                self.target_agent.prompt_managers[i].node_imgs = [[]]
                self.target_agent.prompt_managers[i].graph = [{}]
                self.target_agent.prompt_managers[i].trajectory = [[]]
                self.target_agent.prompt_managers[i].planning = [
                    ["Navigation has just started, with no planning yet."]
                ]
        else:
            print("cannot find target agent")
            exit(0)

        for t in range(self.episode_len):
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            images, candidata_list = self.exp.get_images_and_candidata_list(perm_obs)

            instr_id = perm_obs[0]["instr_id"]

            # load the saliency map and rank
            attr_map = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    # f"{scanId}",
                    f"{instr_id}",
                    f"{t}",
                    # f"{viewpointId}",
                    "attr_map.npy",
                )
            )
            attr_rank = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    # f"{scanId}",
                    # f"{viewpointId}",
                    f"{instr_id}",
                    f"{t}",
                    "attr_rank.npy",
                )
            )
            target_action_surr = self._teacher_action_baseline(
                target_perm_obs, target_ended
            )
            target_action_surr = target_action_surr.cpu().numpy()
            target_action = self.action_space_adaptor(
                "RecVLN", "MapGPT", target_action_surr, candidate_leng
            )

            params = (
                self.target_agent,
                target_perm_obs,
                t,
                previous_angle,
                True,
                # False,  # NOTE: delete later
                target_ended,
            )
            self.causual.average_drop2(
                img=images[0],
                mask=attr_map,
                mask_rank=attr_rank,
                # cls_idx=target_action[0],
                cls_idx=self.get_cls(perm_obs, t)[0],
                params=params,
                mode=mode,
                mask_perc=perturb_ratio,
                # topK=5,
                candidate_idx=candidata_list[0],
                causal_metric_dir=self.causal_metric_dir,
            )

            # 确定真实动作
            # NOTE: MapGPT 的 real action 里面 0 是停止，
            # NOTE: 在 RecVLN 里面，candidate_len[i] - 1 是停止
            # ############### get new obs###########################
            # for target agent---------------------------
            if self.target_agent is not None:
                for i in range(batch_size):
                    target_traj[i]["a_t"][t] = target_action[i]

                # Determine stop actions
                target_a_t_stop = [a_t_i == 0 for a_t_i in target_action]

                # Prepare environment action
                target_cpu_a_t = []
                for i in range(batch_size):
                    if target_a_t_stop[i] or target_ended[i]:
                        target_cpu_a_t.append(-1)
                        target_just_ended[i] = True
                    else:
                        target_cpu_a_t.append(target_action[i] - 1)

                self.target_agent.make_equiv_action(
                    target_cpu_a_t, target_perm_obs, target_traj, perm_idx
                )
                target_obs = np.array(self.target_agent.env._get_obs())
                target_perm_obs = target_obs[perm_idx]

                previous_angle = [
                    {"heading": ob["heading"], "elevation": ob["elevation"]}
                    for ob in target_perm_obs
                ]

                # # we only implement batch_size=1
                # if real_action[0] == 0:
                #     break

                # get global nav inputs
                # target_nav_inputs = nav_inputs
                for i in range(len(target_perm_obs)):
                    ob = target_perm_obs[i]
                    nav_info = collect_nav_info(ob["instr_id"], t)
                    nav_output = nav_info["nav_output_json"]
                    target_nav_input = nav_info["nav_input_json"]

                    self.target_agent.prompt_managers[i].make_history(
                        [target_action[i]], target_nav_input, t
                    )
                    self.target_agent.prompt_managers[i].parse_json_planning(nav_output)

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

    def rollout_mask_test_mapgpt_feature_muFidelity(
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
            for i, _ in enumerate(target_perm_obs):
                self.target_agent.prompt_managers[i].history = [""]
                self.target_agent.prompt_managers[i].nodes_list = [[]]
                self.target_agent.prompt_managers[i].node_imgs = [[]]
                self.target_agent.prompt_managers[i].graph = [{}]
                self.target_agent.prompt_managers[i].trajectory = [[]]
                self.target_agent.prompt_managers[i].planning = [
                    ["Navigation has just started, with no planning yet."]
                ]
        else:
            print("cannot find target agent")
            exit(0)

        for t in range(self.episode_len):
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            images, candidata_list = self.exp.get_images_and_candidata_list(perm_obs)

            # if test_model in ["IG", "temporal", "IG_temporal", "guided_IG"]:
            # scanId = perm_obs[0]["scanId"]
            # viewpointId = perm_obs[0]["viewpoint"]

            instr_id = perm_obs[0]["instr_id"]

            # load the saliency map and rank
            attr_map = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    # f"{scanId}",
                    f"{instr_id}",
                    f"{t}",
                    # f"{viewpointId}",
                    "attr_map.npy",
                )
            )  # [valid_pano, H, W]
            attr_rank = np.load(
                os.path.join(
                    self.saliency_map_dir,
                    # f"{scanId}",
                    # f"{viewpointId}",
                    f"{instr_id}",
                    f"{t}",
                    "attr_rank.npy",
                )
            )  # [valid_pano, H, W]
            target_action_surr = self._teacher_action_baseline(
                target_perm_obs, target_ended
            )
            target_action_surr = target_action_surr.cpu().numpy()
            target_action = self.action_space_adaptor(
                "RecVLN", "MapGPT", target_action_surr, candidate_leng
            )

            params = (
                self.target_agent,
                target_perm_obs,
                t,
                previous_angle,
                True,
                # False,
                target_ended,
            )
            self.causual.average_drop2(
                # self.causual.average_drop_navgpt2(
                img=images[0],
                mask=attr_map,
                mask_rank=attr_rank,
                # cls_idx=target_action[0],
                cls_idx=self.get_cls(perm_obs)[0],
                params=params,
                mode=mode,
                mask_perc=perturb_ratio,
                # topK=5,
                candidate_idx=candidata_list[0],
                causal_metric_dir=self.causal_metric_dir,
            )

            # 确定真实动作
            # NOTE: MapGPT 的 real action 里面 0 是停止，
            # NOTE: 在 RecVLN 里面，candidate_len[i] - 1 是停止
            # ############### get new obs###########################
            # for target agent---------------------------
            if self.target_agent is not None:
                for i in range(batch_size):
                    target_traj[i]["a_t"][t] = target_action[i]

                # Determine stop actions
                target_a_t_stop = [a_t_i == 0 for a_t_i in target_action]

                # Prepare environment action
                target_cpu_a_t = []
                for i in range(batch_size):
                    if target_a_t_stop[i] or target_ended[i]:
                        target_cpu_a_t.append(-1)
                        target_just_ended[i] = True
                    else:
                        target_cpu_a_t.append(target_action[i] - 1)

                self.target_agent.make_equiv_action(
                    target_cpu_a_t, target_perm_obs, target_traj, perm_idx
                )
                target_obs = np.array(self.target_agent.env._get_obs())
                target_perm_obs = target_obs[perm_idx]

                previous_angle = [
                    {"heading": ob["heading"], "elevation": ob["elevation"]}
                    for ob in target_perm_obs
                ]

                # # we only implement batch_size=1
                # if real_action[0] == 0:
                #     break

                # get global nav inputs
                target_nav_inputs = nav_inputs
                for i in range(len(target_perm_obs)):
                    self.target_agent.prompt_managers[i].make_history(
                        [target_action[i]], target_nav_inputs[i], t
                    )
                    self.target_agent.prompt_managers[i].modify_planning(
                        np.array([target_perm_obs[i]])
                    )

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

    def merge_IG_temporal_saliency_map(self, mode="soft_vote"):
        """
        Merge the saliency maps from IG and temporal models with the specified mode.
        Args:
            mode: "IG" (IG model) or "temporal" (temporal model)
        Returns:
            dict: { (instr_id, t): (merged_map, merged_rank) }
        """
        use_critical = True
        critical_file = "./scripts/temporal_data.json"
        if use_critical:
            with open(critical_file, "r") as f:
                temporal_data = json.load(f)
        else:
            temporal_data = None

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
        dir_names = [
            "VLNBERT-test-baseline-mapgpt-ig",
            "VLNBERT-test-baseline-mapgpt-temporal",
        ]
        saliency_map_dir_IG = os.path.join("snap", dir_names[0], "saliency_map_pixel")
        saliency_map_dir_temporal = os.path.join(
            "snap", dir_names[1], "saliency_map_pixel"
        )
        # saliency_map_dir_IG_temporal = os.path.join(
        #     "snap", "VLNBERT-test-baseline-mapgpt-ig-temporal", "saliency_map_pixel"
        # )
        saliency_map_dir_IG_temporal = self.saliency_map_dir
        for instr_id in os.listdir(saliency_map_dir_IG):
            instr_dir = os.path.join(saliency_map_dir_IG, instr_id)
            for t in os.listdir(instr_dir):
                attr_map_list = []
                attr_rank_list = []
                # for agent_id in self.agents_id_list:
                for dir_name in dir_names:
                    saliency_map_dir = os.path.join(
                        "snap", dir_name, "saliency_map_pixel", instr_id, t
                    )
                    attr_map_path = os.path.join(
                        saliency_map_dir,
                        "attr_map.npy",
                    )
                    attr_rank_path = os.path.join(
                        saliency_map_dir,
                        "attr_rank.npy",
                    )
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
                # save ensemble map and ensemble rank to local disk
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
