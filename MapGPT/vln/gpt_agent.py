import sys
import numpy as np
from collections import defaultdict
from GPT.one_stage_prompt_manager import OneStagePromptManager
from .agent_base import BaseAgent
from GPT.api import gpt_infer, gpt_infer_with_probs
import json
from utils.utils import compute_entropy
from collections import Counter
import os


class GPTNavAgent(BaseAgent):
    env_actions = {
        "left": (0, -1, 0),  # left
        "right": (0, 1, 0),  # right
        "up": (0, 0, 1),  # up
        "down": (0, 0, -1),  # down
        "forward": (1, 0, 0),  # forward
        "<end>": (0, 0, 0),  # <end>
        "<start>": (0, 0, 0),  # <start>
        "<ignore>": (0, 0, 0),  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0):
        super().__init__(env)
        self.args = args

        self._build_prompt_manager()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _build_prompt_manager(self):
        self.prompt_manager = OneStagePromptManager(self.args)
        print("Model version:", self.args.llm)

    def make_equiv_action(self, a_t, obs, traj=None, perm_idx=None):
        if perm_idx is None:
            perm_idx = range(len(obs))

        def take_action(i, name):
            idx = perm_idx[i]
            if type(name) is int:  # Go to the next viewpoint
                self.env.env.sims[idx].makeAction([name], [0], [0])
            else:  # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])

        for i, ob in enumerate(obs):
            action = a_t[i]
            idx = perm_idx[i]
            if action != -1:  # -1 is the <stop> action
                # print("cand", ob["candidate"])
                # print("action", action)
                select_candidate = ob["candidate"][action]
                src_point = ob["viewIndex"]
                trg_point = select_candidate["pointId"]
                src_level = (src_point) // 12  # The point idx started from 0
                trg_level = (trg_point) // 12
                while src_level < trg_level:  # Tune up
                    take_action(i, "up")
                    src_level += 1
                while src_level > trg_level:  # Tune down
                    take_action(i, "down")
                    src_level -= 1
                while (
                    self.env.env.sims[idx].getState()[0].viewIndex != trg_point
                ):  # Turn right until the target
                    take_action(i, "right")
                assert (
                    select_candidate["viewpointId"]
                    == self.env.env.sims[idx]
                    .getState()[0]
                    .navigableLocations[select_candidate["idx"]]
                    .viewpointId
                )
                take_action(
                    i, select_candidate["idx"]
                )  # j+1: idx for navigable location

                state = self.env.env.sims[idx].getState()[0]
                if traj is not None:
                    traj[i]["path"].append([state.location.viewpointId])

    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        batch_size = len(obs)

        # Record the navigation path
        traj = [
            {
                "instr_id": ob["instr_id"],
                "path": [[ob["viewpoint"]]],
                "details": {},
                "a_t": {},
                "uncertainty": {},
                "probs": {},
            }
            for ob in obs
        ]

        if traj[0]["instr_id"] in self.results:
            return [None]
        print(traj[0]["instr_id"])
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        previous_angle = [
            {"heading": ob["heading"], "elevation": ob["elevation"]} for ob in obs
        ]

        self.prompt_manager.history = ["" for _ in range(self.args.batch_size)]
        self.prompt_manager.nodes_list = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.node_imgs = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.graph = [{} for _ in range(self.args.batch_size)]
        self.prompt_manager.trajectory = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.planning = [
            ["Navigation has just started, with no planning yet."]
            for _ in range(self.args.batch_size)
        ]

        for t in range(self.args.max_action_len):
            if t == self.args.max_action_len:
                break

            cand_inputs = self.prompt_manager.make_action_prompt(obs, previous_angle)
            if self.args.response_format == "str":
                nav_input = self.prompt_manager.make_r2r_prompts(
                    cand_inputs=cand_inputs, obs=obs, t=t
                )
            elif self.args.response_format == "json":
                nav_input = self.prompt_manager.make_r2r_json_prompts(
                    cand_inputs=cand_inputs, obs=obs, t=t
                )
            else:
                raise NotImplemented

            image_list = self.prompt_manager.node_imgs[0]
            environment_prompts = nav_input["prompts"][0]
            print("-------------------- Environment Prompts --------------------")
            print(environment_prompts)

            if (
                self.args.llm == "gpt-4-vision-preview"
                and self.args.response_format == "str"
            ):
                # GPT-4V only supports string mode output
                nav_output, tokens = gpt_infer(
                    nav_input["task_description"],
                    environment_prompts,
                    image_list,
                    self.args.llm,
                    self.args.max_tokens,
                )
                print("-------------------- Output --------------------")
                print(nav_output)
                nav_output = [nav_output]
                a_t = self.prompt_manager.parse_action(
                    nav_output=nav_output,
                    only_options_batch=nav_input["only_options"],
                    t=t,
                )
                self.prompt_manager.parse_planning(nav_output=nav_output)

            elif (
                self.args.llm == "gpt-4o-2024-05-13"
                and self.args.response_format == "json"
            ):
                if len(image_list) > 20:
                    # GPT-4o currently does not support queries with more than 20 images
                    a_t = [0]
                    print("Exceed image limit and stop!")
                else:
                    # nav_output, tokens, probs = gpt_infer_with_probs(
                    #     nav_input["task_description"],
                    #     environment_prompts,
                    #     image_list,
                    #     self.args.llm,
                    #     self.args.max_tokens,
                    #     response_format={"type": "json_object"},
                    # )
                    nav_output, tokens = gpt_infer(
                        nav_input["task_description"],
                        environment_prompts,
                        image_list,
                        self.args.llm,
                        self.args.max_tokens,
                        response_format={"type": "json_object"},
                    )
                    json_output = json.loads(nav_output)
                    a_t, action_t = self.prompt_manager.parse_json_action(
                        json_output, nav_input["only_options"], t
                    )
                    # probs_t = self.prompt_manager.parse_probs(probs, action_t)
                    self.prompt_manager.parse_json_planning(json_output)
                    # uncertainty_t = compute_entropy(
                    #     np.array(probs_t)
                    # )  # return 0 if probs_t is empty []
                    print("-------------------- Output --------------------")
                    print(nav_output)

            else:
                raise NotImplemented

            for i in range(batch_size):
                traj[i]["a_t"][t] = a_t[i]
                # traj[i]["probs"][t] = probs_t
                # traj[i]["uncertainty"][t] = uncertainty_t

            # Determine stop actions
            a_t_stop = [a_t_i == 0 for a_t_i in a_t]

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i]:
                    cpu_a_t.append(-1)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(a_t[i] - 1)

            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = self.env._get_obs()

            previous_angle = [
                {"heading": ob["heading"], "elevation": ob["elevation"]} for ob in obs
            ]

            # we only implement batch_size=1
            if a_t[0] == 0:
                break

            self.prompt_manager.make_history(a_t, nav_input, t)

        return traj

    def rollout_test_og(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        batch_size = len(obs)

        # Record the navigation path
        traj = [
            {
                "instr_id": ob["instr_id"],
                "path": [[ob["viewpoint"]]],
                "details": {},
                "a_t": {},
                "uncertainty": {},
                "probs": {},
            }
            for ob in obs
        ]

        if traj[0]["instr_id"] in self.results:
            return [None]
        print(traj[0]["instr_id"])
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        previous_angle = [
            {"heading": ob["heading"], "elevation": ob["elevation"]} for ob in obs
        ]

        self.prompt_manager.history = ["" for _ in range(self.args.batch_size)]
        self.prompt_manager.nodes_list = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.node_imgs = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.graph = [{} for _ in range(self.args.batch_size)]
        self.prompt_manager.trajectory = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.planning = [
            ["Navigation has just started, with no planning yet."]
            for _ in range(self.args.batch_size)
        ]

        for t in range(self.args.max_action_len):
            if t == self.args.max_action_len:
                break

            cand_inputs = self.prompt_manager.make_action_prompt(obs, previous_angle)
            if self.args.response_format == "str":
                nav_input = self.prompt_manager.make_r2r_prompts(
                    cand_inputs=cand_inputs, obs=obs, t=t
                )
            elif self.args.response_format == "json":
                nav_input = self.prompt_manager.make_r2r_json_prompts(
                    cand_inputs=cand_inputs, obs=obs, t=t
                )
            else:
                raise NotImplemented

            image_list = self.prompt_manager.node_imgs[0]
            environment_prompts = nav_input["prompts"][0]
            print("-------------------- Environment Prompts --------------------")
            print(environment_prompts)

            if (
                self.args.llm == "gpt-4-vision-preview"
                and self.args.response_format == "str"
            ):
                # GPT-4V only supports string mode output
                nav_output, tokens = gpt_infer(
                    nav_input["task_description"],
                    environment_prompts,
                    image_list,
                    self.args.llm,
                    self.args.max_tokens,
                )
                print("-------------------- Output --------------------")
                print(nav_output)
                nav_output = [nav_output]
                a_t = self.prompt_manager.parse_action(
                    nav_output=nav_output,
                    only_options_batch=nav_input["only_options"],
                    t=t,
                )
                self.prompt_manager.parse_planning(nav_output=nav_output)

            elif (
                # self.args.llm == "gpt-4o-2024-05-13"
                self.args.llm == "gpt-4o"
                and self.args.response_format == "json"
            ):
                if len(image_list) > 20:
                    # GPT-4o currently does not support queries with more than 20 images
                    a_t = [0]
                    print("Exceed image limit and stop!")
                else:
                    # nav_output, tokens, probs = gpt_infer_with_probs(
                    #     nav_input["task_description"],
                    #     environment_prompts,
                    #     image_list,
                    #     self.args.llm,
                    #     self.args.max_tokens,
                    #     response_format={"type": "json_object"},
                    # )
                    a_t_list = []  # store the a_t for each attempt
                    nav_output_list = []
                    repeat_num = 1
                    for ii in range(repeat_num):
                        nav_output, tokens = gpt_infer(
                            nav_input["task_description"],
                            environment_prompts,
                            image_list,
                            self.args.llm,
                            self.args.max_tokens,
                            response_format={"type": "json_object"},
                        )
                        print(f"--------{ii}--------")
                        print(nav_output)
                        nav_output_list.append(nav_output)

                    for ii in range(repeat_num):
                        nav_output = nav_output_list[ii]
                        json_output = json.loads(nav_output)
                        # self.prompt_manager.parse_json_planning(json_output)
                        a_t_tmp, action_t_tmp = self.prompt_manager.parse_json_action(
                            json_output, nav_input["only_options"], t
                        )
                        print(f"a_t_tmp {a_t_tmp} action_t_tmp {action_t_tmp}")
                        a_t_list.append(a_t_tmp[0])

                    # count the most frequent a_t and index in the list
                    a_t_count = Counter(a_t_list)
                    # 找到a_t_count中value最大的对应的key，其在a_t_list中的index
                    most_frequent_a_t = a_t_count.most_common(1)[0][0]
                    a_t_index = a_t_list.index(most_frequent_a_t)
                    a_t = [a_t_list[a_t_index]]
                    print(f"most frequent a_t {a_t} with index {a_t_index}")
                    nav_output = nav_output_list[a_t_index]
                    print("------picked nav_output-------")
                    print(nav_output)

                    json_output = json.loads(nav_output)

                    a_t, action_t = self.prompt_manager.parse_json_action(
                        json_output, nav_input["only_options"], t
                    )
                    # probs_t = self.prompt_manager.parse_probs(probs, action_t)
                    self.prompt_manager.parse_json_planning(json_output)
                    # uncertainty_t = compute_entropy(
                    #     np.array(probs_t)
                    # )  # return 0 if probs_t is empty []
                    print("-------------------- Output --------------------")
                    print(nav_output)
                    vp = obs[0]["viewpoint"]
                    self.save_nav_output(
                        nav_input,
                        nav_output,
                        image_list,
                        traj[0]["instr_id"],
                        vp,
                        t,
                        a_t_list,
                    )
                    print(f"saved nav_output for {traj[0]['instr_id']} at time {t}")
            else:
                raise NotImplemented

            for i in range(batch_size):
                traj[i]["a_t"][t] = a_t[i]
                # traj[i]["probs"][t] = probs_t
                # traj[i]["uncertainty"][t] = uncertainty_t

            # Determine stop actions
            a_t_stop = [a_t_i == 0 for a_t_i in a_t]

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i]:
                    cpu_a_t.append(-1)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(a_t[i] - 1)

            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = self.env._get_obs()

            previous_angle = [
                {"heading": ob["heading"], "elevation": ob["elevation"]} for ob in obs
            ]

            # we only implement batch_size=1
            if a_t[0] == 0:
                break

            self.prompt_manager.make_history(a_t, nav_input, t)

        return traj

    def save_nav_output(
        self, nav_input_json, nav_output_json, img_list, instr_id, vp, t, a_t_list
    ):
        # Save all nav_output_json in one JSON file with incremental updates
        root_dir = "./nav_30"
        output_file = os.path.join(root_dir, "all_nav_outputs.json")

        # Create directory if it doesn't exist
        os.makedirs(root_dir, exist_ok=True)

        # Load existing data if file exists
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    all_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                # If file is corrupted or empty, start fresh
                all_data = {}
        else:
            all_data = {}

        # Initialize instr_id entry if it doesn't exist
        if instr_id not in all_data:
            all_data[instr_id] = {}

        # Convert string keys to integers if needed (JSON loads numeric keys as strings)
        # This ensures we can use integer keys in Python
        if isinstance(all_data[instr_id], dict):
            converted_dict = {}
            for key, value in all_data[instr_id].items():
                # Try to convert numeric string keys to integers
                try:
                    int_key = int(key)
                    converted_dict[int_key] = value
                except (ValueError, TypeError):
                    converted_dict[key] = value
            all_data[instr_id] = converted_dict

        # Update the entry for this time step (as integer key)
        all_data[instr_id][int(t)] = {
            "nav_input_json": nav_input_json,
            "nav_output_json": nav_output_json,
            "a_t_list": a_t_list,
            "img_list": img_list,
            "vp": vp,
        }

        # Write back to file (incremental update)
        # Note: JSON will serialize integer keys as strings (standard JSON behavior)
        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=2, sort_keys=True)
