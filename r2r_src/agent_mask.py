from typing import Tuple
from agent import Seq2SeqAgent
from param import args
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import json
import time
import cv2
from PIL import Image

import r2r_src.vln_utils as vln_utils

if args.target_agent == "MapGPT":
    from MapGPT.vln.gpt_agent import GPTNavAgent
    from MapGPT.GPT.one_stage_prompt_manager import OneStagePromptManager
    from MapGPT.GPT.api import gpt_infer
    from vlnbert.IG_utils import Exp
    from vlnbert.XRAI import XRAI, extract_object_masks_yolo
# else:
#     from NavGPT.nav_src.agent import NavAgent


def MapGPT_genAction(
    agent,
    obs,
    t,
    previous_angle,
    do_inference=True,
    ended=None,
    max_retries=3,
    retry_delay=1.0,  # seconds
) -> Tuple[np.ndarray, np.ndarray, list]:
    # if t == agent.args.max_action_len:
    #     break
    a_t = np.zeros(len(obs), dtype=np.int32)
    cand_nums = np.zeros(len(obs), dtype=np.int32)
    nav_inputs = []
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
    return a_t, cand_nums, nav_inputs


class MaskAgent(Seq2SeqAgent):
    def __init__(self, env, results_path, tok, episode_len=20, args_target=None):
        super(MaskAgent, self).__init__(env, results_path, tok, episode_len)
        rank = 0
        if args.target_agent == "MapGPT":
            self.target_agent = GPTNavAgent(args_target, env, rank=rank)
            self.target_agent.prompt_managers = [
                OneStagePromptManager(args_target) for i in range(args.batchSize)
            ]

    def rollout_mask(self, train_ml=None, train_rl=True, reset=True, iter=0):
        if args.target_agent == "MapGPT":
            return self.rollout_mask_surrogate()
            # return self.rollout_mask_mapgpt(train_ml=None, train_rl=True, reset=True)
        else:
            print("cannot find matched target agent.")
            assert False

    def rollout_mask_mapgpt(self, train_ml=None, train_rl=True, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        # 对于 MapGPT 来说，导航 viewpoint 中选择 `0` 代表stop
        # 对于 RecVLNBert来说，viewpoint选择`len(candidate)`代表stop
        # if self.feedback == "teacher" or self.feedback == "argmax":
        #     train_rl = False
        train_rl = True

        if reset:  # Reset env
            obs = np.array(self.env.reset())
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

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        num_masks = []
        ml_loss = 0.0

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
            # print(target_traj[0]["instr_id"])
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
        # --------------------------
        for t in range(self.episode_len):
            # generate target agent action
            if self.target_agent is not None:
                target_action, target_options, target_nav_inputs = MapGPT_genAction(
                    self.target_agent,
                    target_perm_obs,
                    t,
                    previous_angle,
                    do_inference=True,
                    ended=target_ended,
                )

            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            # genearte mask action
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

            # # # Mask outputs where agent can't move forward
            # # # Here the logit is [b, max_candidate]
            # candidate_mask = utils.length2mask(candidate_leng)
            # # logit.masked_fill_(candidate_mask, -float("inf"))

            # # 用原策略生成action
            # B_action_copy = (
            #     self.generate_pseudo_action(logit, candidate_mask, mode="sample")
            #     .cpu()
            #     .numpy()
            # )

            # 生成mask
            critical_logits = self.critical_head(h_t)
            critical_probs = F.softmax(
                critical_logits, 1
            )  # sampling an action from model
            critical_c = torch.distributions.Categorical(critical_probs)
            critical_a_t = critical_c.sample().detach()
            # 统计掩码个数
            mask_action = 1 - critical_a_t
            num_mask = torch.sum(mask_action[~ended])
            num_masks.append(num_mask)
            policy_log_probs.append(critical_c.log_prob(critical_a_t))

            self.logs["entropy"].append(critical_c.entropy().sum().item())  # For log
            entropys.append(critical_c.entropy())  # For optimization

            # 确定真实动作
            mask_action_copy = critical_a_t.cpu().numpy()
            real_action = []
            for i in range(batch_size):
                # mask_action_copy[i] = 1
                if mask_action_copy[i] == 1:
                    real_action.append(target_action[i])
                else:
                    # real_action.append(np.random.choice(len(B_action_options[i])))
                    n = candidate_leng[i]
                    if n == 0:
                        # 处理无选项情况（根据实际需求调整）
                        real_action.append(-1)
                    elif n == 1:
                        # 只有1个选项时直接选择
                        real_action.append(0)
                    else:
                        # 生成不等于B_action_copy[i]的随机索引
                        while True:
                            idx = np.random.choice(n)
                            if idx != target_action[i]:
                                real_action.append(idx)
                                break
            # NOTE: MapGPT 的 real action 里面 0 是停止，
            # NOTE: 在 RecVLN 里面，candidate_len[i] - 1 是停止
            # ############### get new obs###########################
            # for target agent---------------------------
            if self.target_agent is not None:
                for i in range(batch_size):
                    target_traj[i]["a_t"][t] = real_action[i]

                # Determine stop actions
                target_a_t_stop = [a_t_i == 0 for a_t_i in real_action]

                # Prepare environment action
                target_cpu_a_t = []
                for i in range(batch_size):
                    if target_a_t_stop[i] or target_ended[i]:
                        target_cpu_a_t.append(-1)
                        target_just_ended[i] = True
                    else:
                        target_cpu_a_t.append(real_action[i] - 1)
                target_cpu_a_t = np.array(target_cpu_a_t)
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

                for i in range(len(target_perm_obs)):
                    self.target_agent.prompt_managers[i].make_history(
                        [real_action[i]], target_nav_inputs[i], t
                    )
                    self.target_agent.prompt_managers[i].modify_planning(
                        np.array([target_perm_obs[i]])
                    )

            # ############### end of get new obs###########################
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            # cpu_a_t = a_t.cpu().numpy()
            # 调换一下action的index
            real_action_surr = self.action_space_adaptor(
                "MapGPT", "RecVLN", real_action, candidate_leng
            )
            cpu_a_t = np.array(real_action_surr)
            for i, next_id in enumerate(cpu_a_t):
                if (
                    next_id == (candidate_leng[i] - 1)
                    or next_id == args.ignoreid
                    or ended[i]
                ):  # The last action is <end> (args.ignoreid 只在 teacher_action中起作用)
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
                    # ndtw_score = last_ndtw = np.zeros(batch_size, np.float32)
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
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                # reward += (
                #     # 0.5 * mask_action.cpu().numpy()
                # )  # 把掩码添加到奖励中，掩码越多越好
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))
            target_ended[:] = np.logical_or(target_ended, (target_cpu_a_t == -1))
            # print("ended", ended)
            # print("target_ended", target_ended)

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
                # rl_loss += -1e-3 * num_masks[t]  # 限制掩码的数量，越多越好
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

    def rollout_mask_surrogate(self, train_ml=None, train_rl=True, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        train_rl = True
        mask_weight = 0.25
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
            language_features_noneupdate = self.vln_bert_noneupdate(**language_inputs)
        elif args.vlnbert == "prevalent":
            h_t, language_features = self.vln_bert(**language_inputs)
            h_t_noneupdate, language_features_noneupdate = self.vln_bert_noneupdate(
                **language_inputs
            )

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
            # Maintain separate language_features for vln_bert and vln_bert_noneupdate
            if (t >= 1) or (args.vlnbert == "prevalent"):
                language_features = torch.cat(
                    (h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1
                )
                language_features_noneupdate = torch.cat(
                    (
                        h_t_noneupdate.unsqueeze(1),
                        language_features_noneupdate[:, 1:, :],
                    ),
                    dim=1,
                )

            visual_temp_mask = (vln_utils.length2mask(candidate_leng) == 0).long()
            visual_attention_mask = torch.cat(
                (language_attention_mask, visual_temp_mask), dim=-1
            )

            # Prepare identical input for both vln_bert and vln_bert_noneupdate
            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            self.vln_bert_noneupdate.vln_bert.config.directions = max(candidate_leng)
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
            visual_inputs_noneupdate = {
                "mode": "visual",
                "sentence": language_features_noneupdate,
                "attention_mask": visual_attention_mask,
                "lang_mask": language_attention_mask,
                "vis_mask": visual_temp_mask,
                "token_type_ids": token_type_ids,
                "action_feats": input_a_t,
                "cand_feats": candidate_feat,
            }

            # 1. Use self.vln_bert to generate timestep importance
            h_t, logit = self.vln_bert(**visual_inputs)
            hidden_states.append(h_t)

            # 2. Use self.vln_bert_noneupdate to generate next action logits
            h_t_noneupdate, logit_noneupdate = self.vln_bert_noneupdate(
                **visual_inputs_noneupdate
            )

            # # Mask outputs where agent can't move forward
            candidate_mask = vln_utils.length2mask(candidate_leng)
            # logit.masked_fill_(candidate_mask, -float("inf"))

            # 用原策略生成action
            B_action_copy = (
                self.generate_pseudo_action(
                    logit_noneupdate, candidate_mask, mode="sample"
                )
                .cpu()
                .numpy()
            )

            # 生成mask -- still use h_t from vln_bert for importance
            critical_logits = self.critical_head(h_t)
            critical_probs = F.softmax(
                critical_logits, 1
            )  # sampling an action from model
            critical_c = torch.distributions.Categorical(critical_probs)
            critical_a_t = critical_c.sample().detach()
            # 统计掩码个数
            mask_action = 1 - critical_a_t
            num_mask = torch.sum(mask_action[~ended])
            num_masks.append(num_mask)
            policy_log_probs.append(critical_c.log_prob(critical_a_t))

            self.logs["entropy"].append(critical_c.entropy().sum().item())  # For log
            entropys.append(critical_c.entropy())  # For optimization

            # 确定真实动作
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
                        # 处理无选项情况（根据实际需求调整）
                        real_action.append(-1)
                    elif n == 1:
                        # 只有1个选项时直接选择
                        real_action.append(0)
                    else:
                        # 生成不等于B_action_copy[i]的随机索引
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
                    # ndtw_score = last_ndtw = np.zeros(batch_size, np.float32)
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
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                    # reward += 0.1 * num_mask.cpu().numpy() # 把掩码添加到奖励中，掩码越多越好
                reward += (
                    mask_weight * mask_action.cpu().numpy()
                )  # 把掩码添加到奖励中，掩码越多越好
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

            # Prepare same visual inputs for both vln_bert and vln_bert_noneupdate
            language_features = torch.cat(
                (h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1
            )
            language_features_noneupdate = torch.cat(
                (h_t_noneupdate.unsqueeze(1), language_features_noneupdate[:, 1:, :]),
                dim=1,
            )

            visual_temp_mask = (vln_utils.length2mask(candidate_leng) == 0).long()
            visual_attention_mask = torch.cat(
                (language_attention_mask, visual_temp_mask), dim=-1
            )

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            self.vln_bert_noneupdate.vln_bert.config.directions = max(candidate_leng)
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
            visual_inputs_noneupdate = {
                "mode": "visual",
                "sentence": language_features_noneupdate,
                "attention_mask": visual_attention_mask,
                "lang_mask": language_attention_mask,
                "vis_mask": visual_temp_mask,
                "token_type_ids": token_type_ids,
                "action_feats": input_a_t,
                "cand_feats": candidate_feat,
            }
            last_h_, _ = self.vln_bert(**visual_inputs)
            last_h_noneupdate, _ = self.vln_bert_noneupdate(**visual_inputs_noneupdate)

            rl_loss = 0.0

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic4mask(last_h_).detach()
            discount_reward = np.zeros(batch_size, np.float32)
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
                # rl_loss += -1e-3 * num_masks[t]  # 限制掩码的数量，越多越好
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

    def action_space_adaptor(self, from_, to_, action, action_space):
        new_action = []
        if from_ == "MapGPT" and to_ == "RecVLN":
            # mapgpt 中0是停止，RecVLN中最后一个元素是停止
            for i, a in enumerate(action):
                if a == 0:  # stop
                    new_action.append(action_space[i] - 1)
                else:
                    new_action.append(a - 1)
        elif from_ == "RecVLN" and to_ == "MapGPT":
            for i, a in enumerate(action):
                if a == action_space[i] - 1:
                    new_action.append(0)
                else:
                    new_action.append(a + 1)
        else:
            print("wrong adaptor mode")
            exit(0)
        return new_action

    def rollout_mask_test(
        self,
        test_model="mask",
        threshod=None,
        save_rand_prob=False,
        replay_info=None,
        reset=True,
    ):
        if args.target_agent == "MapGPT":
            return self.rollout_mask_test_mapgpt(
                test_model=test_model,
                threshod=threshod,
                save_rand_prob=save_rand_prob,
                replay_info=replay_info,
                reset=reset,
            )

            # return self.rollout_mask_test_surrogate(
            #     test_model=test_model,
            #     threshod=threshod,
            #     save_rand_prob=save_rand_prob,
            #     replay_info=replay_info,
            #     reset=reset,
            # )

            # return self.rollout_mask_test_mapgpt_gradient(
            #     test_model=test_model,
            #     threshod=threshod,
            #     save_rand_prob=save_rand_prob,
            #     replay_info=replay_info,
            #     reset=reset,
            # )

            # return self.rollout_mask_test_mapgpt_value_based(
            #     test_model=test_model,
            #     threshod=threshod,
            #     save_rand_prob=save_rand_prob,
            #     replay_info=replay_info,
            #     reset=reset,
            # )

    def rollout_mask_test_mapgpt(
        self,
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
        self.instr_buffer = [
            [] for _ in range(batch_size)
        ]  # to identify which vp's index has been used

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

        # Init the logs
        # rewards = []
        hidden_states = []
        policy_log_probs = []
        # masks = []
        # entropys = []
        # ml_loss = 0.0

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
            h_t, logit = self.vln_bert(**visual_inputs)
            hidden_states.append(h_t)

            # # Mask outputs where agent can't move forward
            # # Here the logit is [b, max_candidate]
            # candidate_mask = vln_utils.length2mask(candidate_leng)
            # logit.masked_fill_(candidate_mask, -float("inf"))

            # # 用原策略生成action
            # do_inference = True
            # if test_model == "replay" and t <= critical_steps_end:
            #     do_inference = False
            # B_action_copy = (
            #     self.generate_pseudo_action(logit, candidate_mask, mode="argmax")
            #     .cpu()
            #     .numpy()
            # )

            # B_action_options = state_t["nav_inputs"]["vp_cand_vpids"]

            # 生成mask
            critical_logits = self.critical_head(h_t).unsqueeze(0)  # NOTE
            # print(critical_logits)
            # print(critical_logits.shape)
            critical_probs = F.softmax(
                critical_logits, 1
            )  # sampling an action from model
            critical_c = torch.distributions.Categorical(critical_probs)
            critical_a_t = critical_c.sample().detach()
            mask_probs.append(critical_c.probs.detach().cpu().numpy()[0])
            rand_f = np.random.rand()
            if save_rand_prob:
                mask_probs[t][1] = rand_f

            # if test_model == "replay":
            #     if rand_f < threshod:
            #         critical_a_t = 1
            #     else:
            #         critical_a_t = 0

            # 统计掩码个数
            mask_action = 1 - critical_a_t
            num_mask = torch.sum(mask_action[~ended])
            num_action_total += torch.sum(torch.ones_like(mask_action)[~ended])
            num_mask_total += num_mask
            # policy_log_probs.append(critical_c.log_prob(critical_a_t))

            # self.logs["entropy"].append(critical_c.entropy().sum().item())  # For log
            # entropys.append(critical_c.entropy())  # For optimization

            # generate target agent action
            if test_model == "baseline" or (
                test_model == "replay" and t < critical_steps_start
            ):
                do_inference_ = False
            else:
                do_inference_ = True

            if self.target_agent is not None:
                target_action, target_options, target_nav_inputs = MapGPT_genAction(
                    self.target_agent,
                    target_perm_obs,
                    t,
                    previous_angle,
                    do_inference=do_inference_,
                    ended=target_ended,
                )
            if test_model == "baseline":
                target_action_surr = self._teacher_action_baseline(
                    target_perm_obs, target_ended
                )
                target_action_surr = target_action_surr.cpu().numpy()
                target_action = self.action_space_adaptor(
                    "RecVLN", "MapGPT", target_action_surr, candidate_leng
                )

            # 确定真实动作
            mask_action_copy = critical_a_t.cpu().numpy()
            real_action = []
            for i in range(batch_size):
                # modify the mask
                # mask_action_copy[i] = 1
                if test_model == "baseline":
                    mask_action_copy[i] = 1
                elif (
                    test_model == "replay"
                    and critical_steps_start <= t <= critical_steps_end
                ):  # do random choice
                    mask_action_copy[i] = 0
                elif (
                    test_model == "replay" and t > critical_steps_end
                ):  # follow ori policy
                    mask_action_copy[i] = 1
                elif test_model == "random_baseline":
                    if rand_f < threshod:  # critical
                        mask_action_copy[i] = 1
                    else:
                        mask_action_copy[i] = 0
                # elif (
                #     test_model == "random_baseline"
                #     or (test_model == "replay" and threshod is not None)
                # ):
                #     # prob = np.random.rand()
                #     if rand_f < threshod:  # critical
                #         mask_action_copy[i] = 1
                #     else:
                #         mask_action_copy[i] = 0
                #     # mask_probs[t][1] = prob

                # determine the final action based on pi and pi_mask
                if test_model == "replay" and t < critical_steps_start:
                    real_action.append(recorded_actions[t])
                elif mask_action_copy[i] == 1:
                    real_action.append(target_action[i])
                else:
                    # real_action.append(np.random.choice(len(B_action_options[i])))
                    n = candidate_leng[i]
                    if n == 0:
                        # 处理无选项情况（根据实际需求调整）
                        real_action.append(-1)
                    elif n == 1:
                        # 只有1个选项时直接选择
                        real_action.append(0)
                    else:
                        # 生成不等于B_action_copy[i]的随机索引
                        while True:
                            idx = np.random.choice(n)
                            if idx != target_action[i]:
                                real_action.append(idx)
                                break
            action_seq.append(real_action[0])
            mask_pos.append(t)

            # NOTE: MapGPT 的 real action 里面 0 是停止，
            # NOTE: 在 RecVLN 里面，candidate_len[i] - 1 是停止
            # ############### get new obs###########################
            # for target agent---------------------------
            if self.target_agent is not None:
                for i in range(batch_size):
                    target_traj[i]["a_t"][t] = real_action[i]

                # Determine stop actions
                target_a_t_stop = [a_t_i == 0 for a_t_i in real_action]

                # Prepare environment action
                target_cpu_a_t = []
                for i in range(batch_size):
                    if target_a_t_stop[i] or target_ended[i]:
                        target_cpu_a_t.append(-1)
                        target_just_ended[i] = True
                    else:
                        target_cpu_a_t.append(real_action[i] - 1)

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

                for i in range(len(target_perm_obs)):
                    self.target_agent.prompt_managers[i].make_history(
                        [real_action[i]], target_nav_inputs[i], t
                    )
                    self.target_agent.prompt_managers[i].modify_planning(
                        np.array([target_perm_obs[i]])
                    )

            # ############### end of get new obs###########################
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            # cpu_a_t = a_t.cpu().numpy()
            # cpu_a_t = real_action

            real_action_surr = self.action_space_adaptor(
                "MapGPT", "RecVLN", real_action, candidate_leng
            )
            cpu_a_t = np.array(real_action_surr)
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
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                # rewards.append(reward)
                # masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score
            total_reward += reward[0]
            total_discounted_reward += np.power(self.GAE, t) * reward[0]
            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))
            target_ended[:] = np.logical_or(target_ended, (target_cpu_a_t == -1))
            # print(ended)

            # Early exit if all ended
            if ended.all():
                break
        # end for

        print("total reward", self.if_succeed(perm_obs, traj))

        # if (
        #     type(self.loss) is int
        # ):  # For safety, it will be activated if no losses are added
        #     self.losses.append(0.0)
        # else:
        #     self.losses.append(
        #         self.loss.item() / self.episode_len
        #     )  # This argument is useless.
        self.a += num_action_total
        self.b += num_mask_total
        print("count", t + 1)
        return (
            traj[i],
            total_reward,
            # total_discounted_reward,
            self.if_succeed(perm_obs, traj)[0],
            t + 1,
            num_mask_total,
            mask_pos,
            action_seq,
            mask_probs,
        )

    def rollout_mask_test_mapgpt_value_based(
        self,
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
        self.instr_buffer = [
            [] for _ in range(batch_size)
        ]  # to identify which vp's index has been used

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

        # Init the logs
        # rewards = []
        hidden_states = []
        policy_log_probs = []
        # masks = []
        # entropys = []
        # ml_loss = 0.0

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
            h_t, logit = self.vln_bert(**visual_inputs)
            hidden_states.append(h_t)

            # get critic(x) to represent the importance
            importance_t = self.critic(h_t).unsqueeze(0).detach()
            # print("shape", importance_t)    # [B]?
            # NOTE: get shape of importance to determine if add `unsqueeze(0)`

            # mask_probs.append(importance_t)
            mask_probs.append(torch.cat([importance_t, importance_t]).cpu())

            # critical_a_t = torch.zeros(batch_size, dtype=torch.int32)
            # for i in range(batch_size):
            #     if importance_t < threshod:
            #         critical_a_t[i] = 1
            critical_a_t = (importance_t > threshod).int()

            # 统计掩码个数
            mask_action = 1 - critical_a_t
            num_mask = torch.sum(mask_action[~ended])
            num_action_total += torch.sum(torch.ones_like(mask_action)[~ended])
            num_mask_total += num_mask
            # policy_log_probs.append(critical_c.log_prob(critical_a_t))

            # self.logs["entropy"].append(critical_c.entropy().sum().item())  # For log
            # entropys.append(critical_c.entropy())  # For optimization

            # generate target agent action
            if test_model == "baseline" or (
                test_model == "replay" and t < critical_steps_start
            ):
                do_inference_ = False
            else:
                do_inference_ = True

            if self.target_agent is not None:
                target_action, target_options, target_nav_inputs = MapGPT_genAction(
                    self.target_agent,
                    target_perm_obs,
                    t,
                    previous_angle,
                    do_inference=do_inference_,
                    ended=target_ended,
                )
            if test_model == "baseline":
                target_action_surr = self._teacher_action_baseline(
                    target_perm_obs, target_ended
                )
                target_action_surr = target_action_surr.cpu().numpy()
                target_action = self.action_space_adaptor(
                    "RecVLN", "MapGPT", target_action_surr, candidate_leng
                )

            # 确定真实动作
            mask_action_copy = critical_a_t.cpu().numpy()
            real_action = []
            for i in range(batch_size):
                # modify the mask
                # mask_action_copy[i] = 1
                if test_model == "baseline":
                    mask_action_copy[i] = 1
                elif (
                    test_model == "replay"
                    and critical_steps_start <= t <= critical_steps_end
                ):  # do random choice
                    mask_action_copy[i] = 0
                elif (
                    test_model == "replay" and t > critical_steps_end
                ):  # follow ori policy
                    mask_action_copy[i] = 1

                # determine the final action based on pi and pi_mask
                if test_model == "replay" and t < critical_steps_start:
                    real_action.append(recorded_actions[t])
                elif mask_action_copy[i] == 1:
                    real_action.append(target_action[i])
                else:
                    # real_action.append(np.random.choice(len(B_action_options[i])))
                    n = candidate_leng[i]
                    if n == 0:
                        # 处理无选项情况（根据实际需求调整）
                        real_action.append(-1)
                    elif n == 1:
                        # 只有1个选项时直接选择
                        real_action.append(0)
                    else:
                        # 生成不等于B_action_copy[i]的随机索引
                        while True:
                            idx = np.random.choice(n)
                            if idx != target_action[i]:
                                real_action.append(idx)
                                break
            action_seq.append(real_action[0])
            mask_pos.append(t)

            # NOTE: MapGPT 的 real action 里面 0 是停止，
            # NOTE: 在 RecVLN 里面，candidate_len[i] - 1 是停止
            # ############### get new obs###########################
            # for target agent---------------------------
            if self.target_agent is not None:
                for i in range(batch_size):
                    target_traj[i]["a_t"][t] = real_action[i]

                # Determine stop actions
                target_a_t_stop = [a_t_i == 0 for a_t_i in real_action]

                # Prepare environment action
                target_cpu_a_t = []
                for i in range(batch_size):
                    if target_a_t_stop[i] or target_ended[i]:
                        target_cpu_a_t.append(-1)
                        target_just_ended[i] = True
                    else:
                        target_cpu_a_t.append(real_action[i] - 1)
                target_cpu_a_t = np.array(target_cpu_a_t)
                self.target_agent.make_equiv_action(
                    target_cpu_a_t, target_perm_obs, target_traj, perm_idx
                )
                target_obs = np.array(self.target_agent.env._get_obs())
                target_perm_obs = target_obs[perm_idx]

                previous_angle = [
                    {"heading": ob["heading"], "elevation": ob["elevation"]}
                    for ob in target_perm_obs
                ]

                for i in range(len(target_perm_obs)):
                    self.target_agent.prompt_managers[i].make_history(
                        [real_action[i]], target_nav_inputs[i], t
                    )
                    self.target_agent.prompt_managers[i].modify_planning(
                        np.array([target_perm_obs[i]])
                    )

            # ############### end of get new obs###########################
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            # cpu_a_t = a_t.cpu().numpy()
            # cpu_a_t = real_action

            real_action_surr = self.action_space_adaptor(
                "MapGPT", "RecVLN", real_action, candidate_leng
            )
            cpu_a_t = np.array(real_action_surr)
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
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                # rewards.append(reward)
                # masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score
            total_reward += reward[0]
            total_discounted_reward += np.power(self.GAE, t) * reward[0]
            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))
            target_ended[:] = np.logical_or(target_ended, (target_cpu_a_t == -1))
            # print(ended)

            # Early exit if all ended
            if ended.all():
                break
        # end for

        print("total reward", self.if_succeed(perm_obs, traj))

        self.a += num_action_total
        self.b += num_mask_total
        return (
            traj[i],
            total_reward,
            # total_discounted_reward,
            self.if_succeed(perm_obs, traj)[0],
            t + 1,
            num_mask_total,
            mask_pos,
            action_seq,
            mask_probs,
        )

    def rollout_mask_test_mapgpt_gradient(
        self,
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
        self.instr_buffer = [
            [] for _ in range(batch_size)
        ]  # to identify which vp's index has been used

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

        # Init the logs
        # rewards = []
        hidden_states = []
        policy_log_probs = []
        # masks = []
        # entropys = []
        # ml_loss = 0.0

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
            candidate_feat = candidate_feat.requires_grad_(True)
            input_a_t = input_a_t.requires_grad_(True)
            language_features = language_features.requires_grad_(True)
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
            max_value, a_t_ = logit.max(1)
            # logit.max(1) returns both the max value and its index.
            # we should gradient its max value

            # get critic(x) to represent the importance
            # importance_t = self.critic(h_t).unsqueeze(0)
            _grad_candidate_feat = self.compute_gradient(candidate_feat, max_value)
            _grad_language_features = self.compute_gradient(
                language_features, max_value
            )
            _grad_input_a_t = self.compute_gradient(input_a_t, max_value)

            def mean_over_features(x: torch.Tensor):
                # Take mean over all dims except batch
                return x.view(x.shape[0], -1).mean(dim=1)  # shape [B]

            grad_candidate_feat = mean_over_features(_grad_candidate_feat)  # [B]
            grad_language_features = mean_over_features(_grad_language_features)  # [B]
            grad_input_a_t = mean_over_features(_grad_input_a_t)  # [B]

            importance_t = (
                torch.stack(
                    [grad_candidate_feat, grad_language_features, grad_input_a_t],
                    dim=1,
                    # [grad_candidate_feat],
                    # dim=1,
                )
                .sum(dim=-1)
                .abs()
            )  # shape [B, 3] -> [B]
            # print(importance_t)
            mask_probs.append(torch.cat([importance_t, importance_t]))

            # critical_a_t = torch.zeros(batch_size, dtype=torch.int32)
            # for i in range(batch_size):
            #     if importance_t < threshod:
            #         critical_a_t[i] = 1
            critical_a_t = (importance_t > threshod).int()

            # 统计掩码个数
            mask_action = 1 - critical_a_t
            num_mask = torch.sum(mask_action[~ended])
            num_action_total += torch.sum(torch.ones_like(mask_action)[~ended])
            num_mask_total += num_mask
            # policy_log_probs.append(critical_c.log_prob(critical_a_t))

            # self.logs["entropy"].append(critical_c.entropy().sum().item())  # For log
            # entropys.append(critical_c.entropy())  # For optimization

            # generate target agent action
            if test_model == "baseline" or (
                test_model == "replay" and t < critical_steps_start
            ):
                do_inference_ = False
            else:
                do_inference_ = True

            if self.target_agent is not None:
                target_action, target_options, target_nav_inputs = MapGPT_genAction(
                    self.target_agent,
                    target_perm_obs,
                    t,
                    previous_angle,
                    do_inference=do_inference_,
                    ended=target_ended,
                )
            if test_model == "baseline":
                target_action_surr = self._teacher_action_baseline(
                    target_perm_obs, target_ended
                )
                target_action_surr = target_action_surr.cpu().numpy()
                target_action = self.action_space_adaptor(
                    "RecVLN", "MapGPT", target_action_surr, candidate_leng
                )

            # 确定真实动作
            mask_action_copy = critical_a_t.cpu().numpy()
            real_action = []
            for i in range(batch_size):
                # modify the mask
                # mask_action_copy[i] = 1
                if test_model == "baseline":
                    mask_action_copy[i] = 1
                elif (
                    test_model == "replay"
                    and critical_steps_start <= t <= critical_steps_end
                ):  # do random choice
                    mask_action_copy[i] = 0
                elif (
                    test_model == "replay" and t > critical_steps_end
                ):  # follow ori policy
                    mask_action_copy[i] = 1

                # determine the final action based on pi and pi_mask
                if test_model == "replay" and t < critical_steps_start:
                    real_action.append(recorded_actions[t])
                elif mask_action_copy[i] == 1:
                    real_action.append(target_action[i])
                else:
                    # real_action.append(np.random.choice(len(B_action_options[i])))
                    n = candidate_leng[i]
                    if n == 0:
                        # 处理无选项情况（根据实际需求调整）
                        real_action.append(-1)
                    elif n == 1:
                        # 只有1个选项时直接选择
                        real_action.append(0)
                    else:
                        # 生成不等于B_action_copy[i]的随机索引
                        while True:
                            idx = np.random.choice(n)
                            if idx != target_action[i]:
                                real_action.append(idx)
                                break
            action_seq.append(real_action[0])
            mask_pos.append(t)

            # NOTE: MapGPT 的 real action 里面 0 是停止，
            # NOTE: 在 RecVLN 里面，candidate_len[i] - 1 是停止
            # ############### get new obs###########################
            # for target agent---------------------------
            if self.target_agent is not None:
                for i in range(batch_size):
                    target_traj[i]["a_t"][t] = real_action[i]

                # Determine stop actions
                target_a_t_stop = [a_t_i == 0 for a_t_i in real_action]

                # Prepare environment action
                target_cpu_a_t = []
                for i in range(batch_size):
                    if target_a_t_stop[i] or target_ended[i]:
                        target_cpu_a_t.append(-1)
                        target_just_ended[i] = True
                    else:
                        target_cpu_a_t.append(real_action[i] - 1)
                target_cpu_a_t = np.array(target_cpu_a_t)
                self.target_agent.make_equiv_action(
                    target_cpu_a_t, target_perm_obs, target_traj, perm_idx
                )
                target_obs = np.array(self.target_agent.env._get_obs())
                target_perm_obs = target_obs[perm_idx]

                previous_angle = [
                    {"heading": ob["heading"], "elevation": ob["elevation"]}
                    for ob in target_perm_obs
                ]

                for i in range(len(target_perm_obs)):
                    self.target_agent.prompt_managers[i].make_history(
                        [real_action[i]], target_nav_inputs[i], t
                    )
                    self.target_agent.prompt_managers[i].modify_planning(
                        np.array([target_perm_obs[i]])
                    )

            # ############### end of get new obs###########################
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            # cpu_a_t = a_t.cpu().numpy()
            # cpu_a_t = real_action

            real_action_surr = self.action_space_adaptor(
                "MapGPT", "RecVLN", real_action, candidate_leng
            )
            cpu_a_t = np.array(real_action_surr)
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
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                # rewards.append(reward)
                # masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score
            total_reward += reward[0]
            total_discounted_reward += np.power(self.GAE, t) * reward[0]
            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))
            target_ended[:] = np.logical_or(target_ended, (target_cpu_a_t == -1))
            # print(ended)

            # Early exit if all ended
            if ended.all():
                break
        # end for

        print("total reward", self.if_succeed(perm_obs, traj))

        self.a += num_action_total
        self.b += num_mask_total
        return (
            traj[i],
            total_reward,
            # total_discounted_reward,
            self.if_succeed(perm_obs, traj)[0],
            t + 1,
            num_mask_total,
            mask_pos,
            action_seq,
            mask_probs,
        )

    def rollout_mask_test_surrogate(
        self,
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
        # rewards = []
        hidden_states = []
        policy_log_probs = []
        # masks = []
        # entropys = []
        # ml_loss = 0.0

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

            # 用原策略生成action
            do_inference = True
            if test_model == "replay" and t <= critical_steps_end:
                do_inference = False
            B_action_copy = (
                self.generate_pseudo_action(logit, candidate_mask, mode="argmax")
                .cpu()
                .numpy()
            )
            # B_action_options = state_t["nav_inputs"]["vp_cand_vpids"]

            # 生成mask
            critical_logits = self.critical_head(h_t).unsqueeze(0)  # NOTE
            # print(critical_logits)
            # print(critical_logits.shape)
            critical_probs = F.softmax(
                critical_logits, 1
            )  # sampling an action from model
            critical_c = torch.distributions.Categorical(critical_probs)
            critical_a_t = critical_c.sample().detach()
            mask_probs.append(critical_c.probs.detach().cpu().numpy()[0])
            rand_f = np.random.rand()
            if save_rand_prob:
                mask_probs[t][1] = rand_f
            # if test_model == "replay":
            #     if rand_f < threshod:
            #         critical_a_t = 1
            #     else:
            #         critical_a_t = 0

            # 统计掩码个数
            mask_action = 1 - critical_a_t
            num_mask = torch.sum(mask_action[~ended])
            num_action_total += torch.sum(torch.ones_like(mask_action)[~ended])
            num_mask_total += num_mask
            # policy_log_probs.append(critical_c.log_prob(critical_a_t))

            # self.logs["entropy"].append(critical_c.entropy().sum().item())  # For log
            # entropys.append(critical_c.entropy())  # For optimization

            # 确定真实动作
            mask_action_copy = critical_a_t.cpu().numpy()
            real_action = []
            for i in range(batch_size):
                # mask_action_copy[i] = 1
                if test_model == "baseline":
                    mask_action_copy[i] = 1
                elif (
                    test_model == "replay"
                    and critical_steps_start <= t <= critical_steps_end
                ):  # do random choice
                    mask_action_copy[i] = 0
                elif (
                    test_model == "replay" and t > critical_steps_end
                ):  # follow ori policy
                    mask_action_copy[i] = 1
                elif test_model == "random_baseline":
                    if rand_f < threshod:  # critical
                        mask_action_copy[i] = 1
                    else:
                        mask_action_copy[i] = 0
                # elif (
                #     test_model == "random_baseline"
                #     or (test_model == "replay" and threshod is not None)
                # ):
                #     # prob = np.random.rand()
                #     if rand_f < threshod:  # critical
                #         mask_action_copy[i] = 1
                #     else:
                #         mask_action_copy[i] = 0
                #     # mask_probs[t][1] = prob

                if test_model == "replay" and t < critical_steps_start:
                    real_action.append(recorded_actions[t])
                elif mask_action_copy[i] == 1:
                    real_action.append(B_action_copy[i])
                else:
                    # real_action.append(np.random.choice(len(B_action_options[i])))
                    n = candidate_leng[i]
                    if n == 0:
                        # 处理无选项情况（根据实际需求调整）
                        real_action.append(-1)
                    elif n == 1:
                        # 只有1个选项时直接选择
                        real_action.append(0)
                    else:
                        # 生成不等于B_action_copy[i]的随机索引
                        while True:
                            idx = np.random.choice(n)
                            if idx != B_action_copy[i]:
                                real_action.append(idx)
                                break
            action_seq.append(real_action[0])
            mask_pos.append(t)

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
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                # rewards.append(reward)
                # masks.append(mask)
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
            # total_discounted_reward,
            self.if_succeed(perm_obs, traj)[0],
            t + 1,
            num_mask_total,
            mask_pos,
            action_seq,
            mask_probs,
        )

    def compute_gradient(
        self, input_: torch.tensor, output_: torch.tensor
    ) -> torch.tensor:
        grad = torch.autograd.grad(
            output_, input_, retain_graph=True, grad_outputs=torch.ones_like(output_)
        )[0]

        return grad.cpu()
