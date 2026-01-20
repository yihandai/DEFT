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

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception,
    retry_any,
)

import r2r_src.vln_utils as vln_utils

if args.target_agent == "MapGPT":
    from MapGPT.vln.gpt_agent import GPTNavAgent
    from MapGPT.GPT.one_stage_prompt_manager import OneStagePromptManager
    from MapGPT.GPT.api import gpt_infer
from vlnbert.IG_utils import Exp
from vlnbert.XRAI import XRAI, extract_object_masks_yolo

from NavGPT.nav_src.agent import NavAgent
from agent_mask import MaskAgent, MapGPT_genAction

try:
    from langchain.agents.agent import AgentAction
except ImportError:
    # Fallback for different langchain versions
    try:
        from langchain.schema import AgentAction
    except ImportError:
        AgentAction = None  # Will handle None case in code


def _generate_single_action_with_retry(
    agent: NavAgent,
    ob,
    t,
    candidate_dict,
    env_state_before,
    history_before,
    original_max_iter,
    accumulated_intermediate_steps=None,
    last_observation=None,
    init_observation_stored=None,
):
    """
    Internal function to generate a single action with retry logic.
    This function strictly follows rollout2 logic from NavGPT.

    Args:
        accumulated_intermediate_steps: List of accumulated intermediate steps for context.
                                       If None or empty, will not inject intermediate_steps (for first step).
        last_observation: Last observation from previous step (for tool_chain mode).
                         If None, will use empty string (for first step).
        init_observation_stored: Stored initial observation for tool_chain mode (remains constant).
                                 If None, will be set from current observation (first step only).

    Returns:
        viewpoint_id: The viewpoint ID string or None for stop action
        new_accumulated_intermediate_steps: Updated intermediate_steps after this call (for context maintenance)
        new_last_observation: Updated last_observation for next step (for tool_chain mode)
        new_init_observation_stored: Updated stored init_observation (for tool_chain mode, remains constant after first step)
    """
    # Save environment state before executing agent_executor
    agent.env.set_scan_viewpoint_heading(env_state_before)

    # Set max_iterations to 1 for step-by-step execution (like rollout2)
    agent.agent_executor.max_iterations = 1

    # Manually inject accumulated_intermediate_steps into agent_executor (exactly like rollout2)
    # This ensures the agent has access to previous steps
    accumulated_intermediate_steps = (
        accumulated_intermediate_steps.copy()
        if accumulated_intermediate_steps is not None
        else []
    )
    if hasattr(agent.agent_executor, "intermediate_steps"):
        agent.agent_executor.intermediate_steps = accumulated_intermediate_steps.copy()

    # Prepare input_dict based on use_tool_chain mode (exactly like rollout2)
    if agent.config.use_tool_chain:
        # For tool_chain mode, init_observation remains constant after first step
        if init_observation_stored is None:
            # First step: get init_observation from current observation
            cur_obs = agent.env._get_obs()[0]
            init_observation_stored = cur_obs.get("obs_summary", "")

        # observation field: empty for first step, last_observation for subsequent steps
        observation_for_input = last_observation if last_observation is not None else ""

        input_dict = {
            "action_plan": (
                agent.cur_action_plan if hasattr(agent, "cur_action_plan") else ""
            ),
            "init_observation": init_observation_stored,  # Constant after first step
            "observation": observation_for_input,  # Empty for first step, last_observation for subsequent steps
        }
    else:
        # For non-tool_chain mode, get current observation from environment (like rollout2)
        cur_obs = agent.env._get_obs()[0]
        feature = cur_obs.get("obs", "")
        navigable = candidate_dict
        heading = np.rad2deg(cur_obs.get("heading", 0))
        elevation = np.rad2deg(cur_obs.get("elevation", 0))
        objects = cur_obs.get("objects", {})
        orientation = f"\nheading: {heading:.2f}, elevation: {elevation:.2f}"

        if agent.config.use_relative_angle:
            feature = agent.modify_heading_angles(heading, feature, navigable, objects)
        if agent.config.use_navigable:
            navigable_str = agent.get_navigable_str(heading, elevation, navigable)
        else:
            navigable_str = ""

        # Build init_observation (exactly like rollout2)
        if agent.config.use_relative_angle:
            if agent.config.use_navigable:
                init_observation = f"\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable_str}"
            else:
                init_observation = f"\n\tCurrent Viewpoint:\n{feature}"
        else:
            if agent.config.use_navigable:
                init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable_str}"
            else:
                init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"

        input_dict = {
            "action_plan": (
                agent.cur_action_plan if hasattr(agent, "cur_action_plan") else ""
            ),
            "init_observation": init_observation,
        }

    # Ensure history is properly initialized before calling agent_executor (exactly like rollout2)
    # In rollout2, history is initialized in init_trajecotry, which should have been called before this function
    # history[0] is used by get_full_inputs when intermediate_steps exist
    # history[1], history[2], etc. are used by _construct_scratchpad for action_maker actions
    # In rollout2, history is maintained by action_maker tool (appends after each action)
    # So history length should be: 1 (initial from init_trajecotry) + number of action_maker executions
    if (
        hasattr(agent, "agent_executor")
        and hasattr(agent.agent_executor, "agent")
        and hasattr(agent.agent_executor.agent, "history")
    ):
        # Initialize history if None or empty (should not happen if init_trajecotry was called)
        # But we add this as a safety check
        if agent.agent_executor.agent.history is None:
            agent.agent_executor.agent.history = []
            # If history is None, we need to initialize it (fallback if init_trajecotry wasn't called)
            init_observation = input_dict.get("init_observation", "")
            agent.agent_executor.agent.history.append(init_observation)

        # Ensure history has enough entries for _construct_scratchpad
        # _construct_scratchpad uses history[nav_step] where nav_step starts at 1
        # and increments for each MAKE_ACTION_TOOL_NAME action that's not the last one
        # In rollout2, history is maintained by action_maker tool, so it should already have enough entries
        # But if accumulated_intermediate_steps has action_maker actions, we need to ensure history is long enough
        if len(accumulated_intermediate_steps) > 0:
            try:
                from NavGPT.nav_src.prompt.planner_prompt import MAKE_ACTION_TOOL_NAME
            except ImportError:
                MAKE_ACTION_TOOL_NAME = "action_maker"

            # Count action_maker actions that are not the last one (exactly like _construct_scratchpad logic)
            action_maker_count = 0
            for i, (action, _) in enumerate(accumulated_intermediate_steps):
                if (
                    i < len(accumulated_intermediate_steps) - 1
                    and hasattr(action, "tool")
                    and action.tool == MAKE_ACTION_TOOL_NAME
                ):
                    action_maker_count += 1

            # history[0] is init_observation (from init_trajecotry), history[1], history[2], ... are for nav_step 1, 2, ...
            # We need at least (action_maker_count + 1) entries total
            required_history_length = action_maker_count + 1
            while len(agent.agent_executor.agent.history) < required_history_length:
                # Pad with the last history entry as fallback
                # This prevents IndexError, though ideally these should be proper history entries
                # In rollout2, these would be set by action_maker tool, but we pad to prevent crashes
                fallback_entry = (
                    agent.agent_executor.agent.history[-1]
                    if len(agent.agent_executor.agent.history) > 0
                    else input_dict.get("init_observation", "")
                )
                agent.agent_executor.agent.history.append(fallback_entry)

    # Also pass manual_intermediate_steps in input_dict as a fallback (exactly like rollout2)
    # This allows VLNAgent.get_full_inputs to use it if AgentExecutor resets intermediate_steps
    input_dict_with_history = input_dict.copy()
    if len(accumulated_intermediate_steps) > 0:
        input_dict_with_history["manual_intermediate_steps"] = (
            accumulated_intermediate_steps.copy()
        )

    # Call agent_executor (exactly like rollout2, without retry decorator here - retry is handled by decorator)
    output = agent.agent_executor(input_dict_with_history)

    # Extract information from this step (exactly like rollout2)
    intermediate_steps = output.get("intermediate_steps", [])

    # Update accumulated_intermediate_steps with new steps from this iteration (exactly like rollout2)
    if len(intermediate_steps) > len(accumulated_intermediate_steps):
        # New steps were added, update accumulated list
        accumulated_intermediate_steps = intermediate_steps.copy()
    elif len(intermediate_steps) > 0:
        # Check if there are any new steps not in accumulated list
        for step in intermediate_steps:
            if step not in accumulated_intermediate_steps:
                accumulated_intermediate_steps.append(step)

    # Also update agent_executor's internal state after the call (exactly like rollout2)
    if hasattr(agent.agent_executor, "intermediate_steps"):
        agent.agent_executor.intermediate_steps = accumulated_intermediate_steps.copy()

    # Check if orchestrator decided to stop (Final Answer) - exactly like rollout2
    output_text = output.get("output", "")
    viewpoint_id = None
    if "Finished!" in output_text or "Final Answer" in output_text:
        # Navigation completed
        viewpoint_id = None  # Stop action
    elif len(intermediate_steps) > 0:
        # Get the last action to determine next step (exactly like rollout2)
        last_action, last_observation_from_step = intermediate_steps[-1]
        if last_action.tool == "action_maker":
            viewpoint_id = last_action.tool_input.strip('"').strip("'").strip()
        elif last_action.tool == "back_tracer":
            viewpoint_id = last_action.tool_input.strip('"').strip("'").strip()
        else:
            viewpoint_id = None

    # Extract last_observation for tool_chain mode (exactly like rollout2)
    # This is needed for the next step's input_dict
    new_last_observation = None
    if agent.config.use_tool_chain and len(intermediate_steps) > 0:
        # Get the last observation from intermediate_steps (from action_maker)
        last_action, last_obs = intermediate_steps[-1]
        new_last_observation = last_obs

    # Note: Do NOT restore environment state or history here
    # The tool (_make_action) has already executed the action and updated history
    # This matches the behavior in rollout2: tool executes action, we don't undo it
    # In rollout_mask_navgpt, if real_action == target_action, we don't need to execute again
    # If real_action != target_action (masked), we'll remove target_action's history/intermediate_steps
    # and execute the masked action

    return (
        viewpoint_id,
        accumulated_intermediate_steps,
        new_last_observation,
        init_observation_stored,
    )


# Retry decorator for API calls and parsing errors
# Retries on any exception (API errors, parsing errors, etc.)
@retry(
    stop=stop_after_attempt(3),  # Retry up to 3 times
    wait=wait_exponential(
        multiplier=1, min=1, max=10
    ),  # Exponential backoff: 1s, 2s, 4s...
    retry=retry_if_exception(
        lambda e: True
    ),  # Retry on any exception (predicate always returns True)
    reraise=True,  # Re-raise the exception after all retries fail
)
def _generate_single_action_with_retry_decorated(
    agent: NavAgent,
    ob,
    t,
    candidate_dict,
    env_state_before,
    history_before,
    original_max_iter,
    accumulated_intermediate_steps=None,
    last_observation=None,
    init_observation_stored=None,
):
    """Wrapper function with retry decorator."""
    return _generate_single_action_with_retry(
        agent,
        ob,
        t,
        candidate_dict,
        env_state_before,
        history_before,
        original_max_iter,
        accumulated_intermediate_steps,
        last_observation,
        init_observation_stored,
    )


def NavGPT_genAction(
    agent: NavAgent,
    obs,
    t,
    previous_angle,
    do_inference=True,
    ended=None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Generate one action step from NavGPT agent.
    Similar to MapGPT_genAction but for NavGPT which uses viewpoint IDs.

    NavGPT uses agent_executor which runs the entire navigation in one call.
    For step-by-step execution, we call agent_executor with max_iterations=1
    and extract the action from intermediate_steps.

    This function includes retry logic for API calls and output parsing errors.

    Returns:
        a_t: action indices (0 = stop, 1+ = action index, where index is 1-based)
        cand_nums: number of candidates
        nav_inputs: navigation inputs (empty list for NavGPT)
    """
    a_t = np.zeros(len(obs), dtype=np.int32)
    cand_nums = np.zeros(len(obs), dtype=np.int32)
    nav_inputs = []
    viewpoint_ids = []
    for i, ob in enumerate(obs):
        if ended is not None and ended[i]:
            a_t[i] = 0  # stop action
            cand_nums[i] = len(ob.get("candidate", {}))
            nav_inputs.append({})
            continue

        if not do_inference:
            a_t[i] = 0  # stop action
            cand_nums[i] = len(ob.get("candidate", {}))
            nav_inputs.append({})
            continue

        # Get candidate viewpoints
        candidate_dict = ob.get("candidate", {})
        cand_nums[i] = len(candidate_dict)

        if len(candidate_dict) == 0:
            a_t[i] = 0  # stop if no candidates
            nav_inputs.append({})
            continue

        try:
            # Save environment state before executing agent_executor
            env_state_before = agent.env.get_scan_viewpoint_heading()
            # Also save the last history entry (if exists) to restore later
            history_before = None
            if (
                hasattr(agent, "agent_executor")
                and hasattr(agent.agent_executor, "agent")
                and hasattr(agent.agent_executor.agent, "history")
                and len(agent.agent_executor.agent.history) > 0
            ):
                history_before = agent.agent_executor.agent.history[-1]

            original_max_iter = agent.agent_executor.max_iterations

            # Get accumulated_intermediate_steps from agent if available
            # This maintains context across steps (similar to rollout2)
            accumulated_intermediate_steps = None
            if hasattr(agent, "_accumulated_intermediate_steps"):
                accumulated_intermediate_steps = (
                    agent._accumulated_intermediate_steps.copy()
                )
            elif hasattr(agent, "agent_executor") and hasattr(
                agent.agent_executor, "intermediate_steps"
            ):
                accumulated_intermediate_steps = (
                    agent.agent_executor.intermediate_steps.copy()
                    if agent.agent_executor.intermediate_steps
                    else []
                )

            # Get last_observation from agent if available (for tool_chain mode)
            # This maintains observation context across steps (exactly like rollout2)
            last_observation = None
            if hasattr(agent, "_last_observation"):
                last_observation = agent._last_observation

            # Get init_observation_stored from agent if available (for tool_chain mode)
            # This remains constant after first step (exactly like rollout2)
            init_observation_stored = None
            if hasattr(agent, "_init_observation_stored"):
                init_observation_stored = agent._init_observation_stored

            # Call the retry-enabled function
            (
                viewpoint_id,
                new_accumulated_intermediate_steps,
                new_last_observation,
                new_init_observation_stored,
            ) = _generate_single_action_with_retry_decorated(
                agent,
                ob,
                t,
                candidate_dict,
                env_state_before,
                history_before,
                original_max_iter,
                accumulated_intermediate_steps,
                last_observation,
                init_observation_stored,
            )

            # Update agent's intermediate_steps for next iteration (maintain context)
            # Store in agent for persistence across calls (exactly like rollout2)
            if not hasattr(agent, "_accumulated_intermediate_steps"):
                agent._accumulated_intermediate_steps = []
            agent._accumulated_intermediate_steps = (
                new_accumulated_intermediate_steps.copy()
            )

            # Update agent's last_observation for next iteration (for tool_chain mode)
            # Store in agent for persistence across calls (exactly like rollout2)
            if agent.config.use_tool_chain:
                agent._last_observation = new_last_observation
                # Store init_observation_stored (remains constant after first step)
                if new_init_observation_stored is not None:
                    agent._init_observation_stored = new_init_observation_stored

            # Also update agent_executor's intermediate_steps
            if hasattr(agent, "agent_executor") and hasattr(
                agent.agent_executor, "intermediate_steps"
            ):
                agent.agent_executor.intermediate_steps = (
                    new_accumulated_intermediate_steps.copy()
                )

                # Always restore original max_iterations
                agent.agent_executor.max_iterations = original_max_iter

            # Convert viewpoint ID to action index
            # Note: NavGPT returns viewpoint IDs (strings), we need to convert to 1-based indices
            # where 0 = stop, 1+ = action index (matching MapGPT's format)
            if viewpoint_id and viewpoint_id in candidate_dict:
                # Action index is 1-based (1 = first candidate, 0 = stop)
                candidate_list = list(candidate_dict.keys())
                a_t[i] = candidate_list.index(viewpoint_id) + 1
                viewpoint_ids.append(viewpoint_id)
            else:
                # Stop action (Final Answer, invalid viewpoint, or error)
                a_t[i] = 0
                viewpoint_ids.append(None)

            nav_inputs.append({})

        except Exception as e:
            print(f"Error in NavGPT_genAction at step {t} (after retries): {e}")
            import traceback

            traceback.print_exc()
            a_t[i] = 0  # stop on error
            cand_nums[i] = len(candidate_dict)
            nav_inputs.append({})
            # Restore max_iterations and environment state in case of error
            try:
                if hasattr(agent, "agent_executor") and hasattr(
                    agent.agent_executor, "max_iterations"
                ):
                    agent.agent_executor.max_iterations = original_max_iter
                    # Try to restore environment state
                    if "env_state_before" in locals():
                        agent.env.set_scan_viewpoint_heading(env_state_before)
            except Exception:
                pass

    return a_t, cand_nums, nav_inputs, viewpoint_ids


class MaskAgent_NavGPT(MaskAgent):
    def __init__(self, env, results_path, tok, episode_len=20, args_target=None):
        super(MaskAgent, self).__init__(env, results_path, tok, episode_len)
        rank = 0
        if args.target_agent == "MapGPT":
            self.target_agent = GPTNavAgent(args_target, env, rank=rank)
            self.target_agent.prompt_managers = [
                OneStagePromptManager(args_target) for i in range(args.batchSize)
            ]
        elif args.target_agent == "NavGPT":
            # NavGPT requires a config Namespace object
            # You need to provide the appropriate config here
            # For now, using args_target as config (may need adjustment)
            if args_target is None:
                raise ValueError("args_target must be provided for NavGPT agent")
            self.target_agent = NavAgent(env, args_target)

    def rollout_mask(self, train_ml=None, train_rl=True, reset=True, iter=0):
        if args.target_agent == "NavGPT":
            # return self.rollout_mask_surrogate()
            return self.rollout_mask_navgpt(train_ml=None, train_rl=True, reset=True)
        else:
            print("cannot find matched target agent.")
            assert False

    def rollout_mask_navgpt(self, train_ml=None, train_rl=True, reset=True):
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
            # ------NOTE-----------
            # Initialize the trajectory
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
            # ---------------------
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

            # Reset accumulated_intermediate_steps for new episode (exactly like rollout2)
            # In rollout2, accumulated_intermediate_steps is reset to [] at the start of each episode
            # We need to reset it here to ensure each episode starts fresh
            self.target_agent._accumulated_intermediate_steps = []

            # Reset last_observation for tool_chain mode (exactly like rollout2)
            # In rollout2, last_observation is reset to None/empty at the start of each episode
            if self.target_agent.config.use_tool_chain:
                self.target_agent._last_observation = (
                    None  # First step uses empty string
                )
                # Reset init_observation_stored (exactly like rollout2)
                # This will be set on first step
                self.target_agent._init_observation_stored = None

            # Also reset agent_executor's intermediate_steps (exactly like rollout2)
            # In rollout2, intermediate_steps is reset to [] at the start of each episode
            if hasattr(self.target_agent, "agent_executor") and hasattr(
                self.target_agent.agent_executor, "intermediate_steps"
            ):
                self.target_agent.agent_executor.intermediate_steps = []
        else:
            print("cannot find target agent")
            exit(0)
        # --------------------------
        for t in range(self.episode_len):
            # generate target agent action
            if self.target_agent is not None:
                (
                    target_action,
                    target_options,
                    target_nav_inputs,
                    target_viewpoint_ids,
                ) = NavGPT_genAction(
                    self.target_agent,
                    target_perm_obs,
                    t,
                    previous_angle,
                    do_inference=True,
                    # do_inference=False,
                    ended=target_ended,
                )
                # print("target_action", target_action)
                # exit(0)
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
            critical_logits = self.critical_head(h_t).unsqueeze(0)
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
                # NOTE: delete later
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

            real_viewpoint_ids = []

            # get real viewpoint ids
            for i in range(batch_size):
                # 0 is stop
                candidate_dict = target_perm_obs[i].get("candidate", {})
                candidate_list = list(candidate_dict.keys())
                if real_action[i] > 0 and real_action[i] <= len(candidate_list):
                    real_viewpoint_ids.append(candidate_list[real_action[i] - 1])
                else:
                    real_viewpoint_ids.append(None)
            print("vp", target_viewpoint_ids)
            print("real_viewpoint_ids", real_viewpoint_ids)
            # NOTE: MapGPT/NavGPT 的 real action 里面 0 是停止，
            # NOTE: 在 RecVLN 里面，candidate_len[i] - 1 是停止
            # ############### get new obs###########################
            # for target agent---------------------------
            if self.target_agent is not None:
                for i in range(batch_size):
                    target_traj[i]["a_t"][t] = real_action[i]

                # Determine stop actions
                target_a_t_stop = [a_t_i == 0 for a_t_i in real_action]

                # Prepare environment action
                # IMPORTANT: NavGPT_genAction now saves and restores environment state
                # So agent.env was NOT modified by agent_executor
                # We need to execute real_action here

                target_cpu_a_t = []
                for i in range(batch_size):
                    if target_a_t_stop[i] or target_ended[i]:
                        target_cpu_a_t.append(None)  # Stop action for NavGPT
                        target_just_ended[i] = True
                    else:
                        # # Convert real_action index to viewpoint ID
                        # candidate_dict = target_perm_obs[i].get("candidate", {})
                        # candidate_list = list(candidate_dict.keys())
                        # if real_action[i] > 0 and real_action[i] <= len(candidate_list):
                        #     viewpoint_id = candidate_list[real_action[i] - 1]
                        #     target_cpu_a_t.append(viewpoint_id)
                        # else:
                        #     target_cpu_a_t.append(None)
                        #     target_just_ended[i] = True
                        target_cpu_a_t.append(real_viewpoint_ids[i])

                # Execute real_action in NavGPT's environment
                # NavGPT_genAction only extracted the decision, didn't execute it
                for i, vp_id in enumerate(target_cpu_a_t):
                    if vp_id is not None:
                        # Check if real_action differs from target_action (mask was applied)
                        action_was_masked = real_action[i] != target_action[i]

                        # If action was masked, we need to update intermediate_steps
                        # because intermediate_steps contains target_action's step, but we executed random action
                        if action_was_masked and AgentAction is not None:
                            # Remove the last intermediate_step (target_action's step) if it exists
                            if (
                                hasattr(
                                    self.target_agent, "_accumulated_intermediate_steps"
                                )
                                and len(
                                    self.target_agent._accumulated_intermediate_steps
                                )
                                > 0
                            ):
                                # Remove the last step (target_action's step)
                                self.target_agent._accumulated_intermediate_steps = (
                                    self.target_agent._accumulated_intermediate_steps[
                                        :-1
                                    ]
                                )

                            # Also update agent_executor's intermediate_steps
                            if (
                                hasattr(self.target_agent, "agent_executor")
                                and hasattr(
                                    self.target_agent.agent_executor,
                                    "intermediate_steps",
                                )
                                and self.target_agent.agent_executor.intermediate_steps
                                is not None
                                and len(
                                    self.target_agent.agent_executor.intermediate_steps
                                )
                                > 0
                            ):
                                self.target_agent.agent_executor.intermediate_steps = (
                                    self.target_agent.agent_executor.intermediate_steps[
                                        :-1
                                    ]
                                )

                            # Also need to remove the last history entry if it was added by target_action
                            if (
                                hasattr(self.target_agent, "agent_executor")
                                and hasattr(self.target_agent.agent_executor, "agent")
                                and hasattr(
                                    self.target_agent.agent_executor.agent, "history"
                                )
                                and len(self.target_agent.agent_executor.agent.history)
                                > 0
                            ):
                                # Remove the last history entry (target_action's history)
                                self.target_agent.agent_executor.agent.history = (
                                    self.target_agent.agent_executor.agent.history[:-1]
                                )

                        # Execute the action and update history
                        # IMPORTANT: If action was NOT masked, agent_executor already executed it via _make_action tool
                        # So we only need to execute and update state if action WAS masked
                        if action_was_masked:
                            # Execute the masked action
                            # IMPORTANT: make_equiv_action returns (turned_angle, new_obs)
                            # where turned_angle is a detailed angle description string like:
                            # "Turn heading direction 45.2 degrees from left 10.0 to right 35.2."
                            # This should be used for history update, not a fixed "Moved to viewpoint"
                            turned_angle, new_obs = self.target_agent.make_equiv_action(
                                [vp_id]
                            )

                            # Prepare observation string for intermediate_steps
                            if AgentAction is not None:
                                new_feature = new_obs.get("obs", "")
                                new_navigable = new_obs.get("candidate", {})
                                new_heading = np.rad2deg(new_obs.get("heading", 0))
                                new_elevation = np.rad2deg(new_obs.get("elevation", 0))
                                new_objects = new_obs.get("objects", {})

                                if self.target_agent.config.use_relative_angle:
                                    new_feature = (
                                        self.target_agent.modify_heading_angles(
                                            new_heading,
                                            new_feature,
                                            new_navigable,
                                            new_objects,
                                        )
                                    )
                                if self.target_agent.config.use_navigable:
                                    new_navigable_str = (
                                        self.target_agent.get_navigable_str(
                                            new_heading, new_elevation, new_navigable
                                        )
                                    )
                                else:
                                    new_navigable_str = ""

                                # Create observation string (similar to action_maker's return)
                                if self.target_agent.config.use_tool_chain:
                                    observation_str = f"\n\tAction_maker Thought:\n(Executed masked action)\n\tAction_maker Action:\nMoved to viewpoint\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable_str}"
                                elif self.target_agent.config.use_relative_angle:
                                    if self.target_agent.config.use_navigable:
                                        observation_str = f"\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable_str}"
                                    else:
                                        observation_str = (
                                            f"\n\tCurrent Viewpoint:\n{new_feature}"
                                        )
                                else:
                                    new_orientation = f"\nheading: {new_heading:.2f}, elevation: {new_elevation:.2f}"
                                    if self.target_agent.config.use_navigable:
                                        observation_str = f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable_str}"
                                    else:
                                        observation_str = f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}"

                                # Create AgentAction for intermediate_steps
                                actual_action = AgentAction(
                                    tool="action_maker",
                                    tool_input=f'"{vp_id}"',
                                    log=f'Thought: Executing masked action (random choice instead of target action)\nAction: action_maker\nAction Input: "{vp_id}"',
                                )
                                # Add to intermediate_steps
                                if not hasattr(
                                    self.target_agent, "_accumulated_intermediate_steps"
                                ):
                                    self.target_agent._accumulated_intermediate_steps = (
                                        []
                                    )
                                self.target_agent._accumulated_intermediate_steps.append(
                                    (actual_action, observation_str)
                                )

                                # Also update agent_executor's intermediate_steps
                                if hasattr(
                                    self.target_agent, "agent_executor"
                                ) and hasattr(
                                    self.target_agent.agent_executor,
                                    "intermediate_steps",
                                ):
                                    if (
                                        self.target_agent.agent_executor.intermediate_steps
                                        is None
                                    ):
                                        self.target_agent.agent_executor.intermediate_steps = (
                                            []
                                        )
                                    self.target_agent.agent_executor.intermediate_steps.append(
                                        (actual_action, observation_str)
                                    )

                                # Update last_observation for tool_chain mode (for next step's input_dict)
                                if self.target_agent.config.use_tool_chain:
                                    self.target_agent._last_observation = (
                                        observation_str
                                    )

                            # Update history for NavGPT (similar to what action_maker does)
                            # Use turned_angle (detailed angle description) instead of fixed "Moved to viewpoint"
                            # This matches the behavior in _make_action tool and collect_nav_info_rollout2
                            if self.target_agent.config.use_history_chain:
                                # Use history_chain to update history
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
                                        previous_action=turned_angle,  # Use turned_angle instead of "Moved to viewpoint"
                                    )
                                else:
                                    history = self.target_agent.get_history(
                                        new_obs,
                                        turned_angle,  # Use turned_angle instead of "Moved to viewpoint"
                                    )
                            else:
                                history = self.target_agent.get_history(
                                    new_obs,
                                    turned_angle,  # Use turned_angle instead of "Moved to viewpoint"
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
                                    "turned_angle": turned_angle,  # Use turned_angle instead of "Moved to viewpoint"
                                    "feature": new_obs.get("obs", ""),
                                    "history": history,
                                }
                                if "details" not in self.target_agent.traj[0]:
                                    self.target_agent.traj[0]["details"] = []
                                self.target_agent.traj[0]["details"].append(detail)
                        else:
                            # Action was NOT masked: agent_executor already executed it via _make_action tool
                            # History and intermediate_steps are already updated by agent_executor
                            # We just need to update last_observation for tool_chain mode (if needed)
                            if self.target_agent.config.use_tool_chain:
                                # Get the last observation from intermediate_steps (already updated by agent_executor)
                                if (
                                    hasattr(
                                        self.target_agent,
                                        "_accumulated_intermediate_steps",
                                    )
                                    and len(
                                        self.target_agent._accumulated_intermediate_steps
                                    )
                                    > 0
                                ):
                                    last_action, last_obs = (
                                        self.target_agent._accumulated_intermediate_steps[
                                            -1
                                        ]
                                    )
                                    self.target_agent._last_observation = last_obs

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

                # # we only implement batch_size=1
                # if real_action[0] == 0:
                #     break

                # NavGPT handles history internally (already updated above)

            # ############### end of get new obs###########################
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            # cpu_a_t = a_t.cpu().numpy()
            # 调换一下action的index
            # NavGPT uses same action format as MapGPT (0=stop, 1+=action index)
            # real_action_surr = self.action_space_adaptor(
            #     "MapGPT", "RecVLN", real_action, candidate_leng
            # )
            # get real_action_surr from real_viewpoint_ids
            real_action_surr = []
            for i in range(batch_size):
                candidate_list_surr = [
                    x["viewpointId"] for x in perm_obs[i].get("candidate", [])
                ]
                if real_viewpoint_ids[i] is not None:
                    real_action_surr.append(
                        candidate_list_surr.index(real_viewpoint_ids[i])
                    )
                else:
                    real_action_surr.append(len(candidate_list_surr))

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
            # For NavGPT, target_cpu_a_t contains viewpoint IDs or None
            target_ended[:] = np.logical_or(
                target_ended, np.array([vp_id is None for vp_id in target_cpu_a_t])
            )
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
            last_value__ = (
                self.critic4mask(last_h_).detach().unsqueeze(0)
            )  # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(
                batch_size, np.float32
            )  # The inital reward is zero
            for i in range(batch_size):
                if not ended[
                    i
                ]:  # If the action is not ended, use the value function as the last reward
                    # Handle 0-dim tensor case (when batch_size=1)
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
                v_ = self.critic4mask(hidden_states[t]).unsqueeze(0)
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
                    0.25 * mask_action.cpu().numpy()
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
                    # Handle 0-dim tensor case (when batch_size=1)
                    if last_value__[i].dim() == 0:
                        discount_reward[i] = last_value__[i].item()
                    else:
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
            # in MapGPT, 0 means stop, in RecVLN, the last element means stop
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
        elif args.target_agent == "NavGPT":
            return self.rollout_mask_test_navgpt_value_based(
                test_model=test_model,
                threshod=threshod,
                save_rand_prob=save_rand_prob,
                replay_info=replay_info,
                reset=reset,
            )
        assert args.target_agent not in [
            "NavGPT",
            "MapGPT",
        ], "Only NavGPT and MapGPT are supported"

    def rollout_mask_test_navgpt(
        self,
        test_model="mask",
        threshod=None,
        save_rand_prob=False,
        replay_info=None,
        reset=True,
    ):
        """
        Test version of rollout_mask_navgpt for NavGPT agent.
        Based on rollout_mask_navgpt but adapted for testing scenarios.
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
        hidden_states = []
        policy_log_probs = []

        # baseline agent init --------------------------
        if self.target_agent is not None:
            # ------NOTE-----------
            # Initialize the trajectory
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
            # ---------------------
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

            # Reset accumulated_intermediate_steps for new episode (exactly like rollout2)
            # In rollout2, accumulated_intermediate_steps is reset to [] at the start of each episode
            # We need to reset it here to ensure each episode starts fresh
            self.target_agent._accumulated_intermediate_steps = []

            # Reset last_observation for tool_chain mode (exactly like rollout2)
            # In rollout2, last_observation is reset to None/empty at the start of each episode
            if self.target_agent.config.use_tool_chain:
                self.target_agent._last_observation = (
                    None  # First step uses empty string
                )
                # Reset init_observation_stored (exactly like rollout2)
                # This will be set on first step
                self.target_agent._init_observation_stored = None

            # Also reset agent_executor's intermediate_steps (exactly like rollout2)
            # In rollout2, intermediate_steps is reset to [] at the start of each episode
            if hasattr(self.target_agent, "agent_executor") and hasattr(
                self.target_agent.agent_executor, "intermediate_steps"
            ):
                self.target_agent.agent_executor.intermediate_steps = []
        else:
            print("cannot find target agent")
            exit(0)
        # --------------------------
        for t in range(self.episode_len):
            # generate target agent action
            if test_model == "baseline" or (
                test_model == "replay" and t < critical_steps_start
            ):
                do_inference_ = False
            else:
                do_inference_ = True

            if self.target_agent is not None:
                (
                    target_action,
                    target_options,
                    target_nav_inputs,
                    target_viewpoint_ids,
                ) = NavGPT_genAction(
                    self.target_agent,
                    target_perm_obs,
                    t,
                    previous_angle,
                    do_inference=do_inference_,
                    ended=target_ended,
                )
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
            # TODO
            if test_model == "baseline":
                # target_action_surr = self._teacher_action_baseline_navgpt(
                #     target_perm_obs, target_ended
                # )
                target_action = self._teacher_action_baseline_navgpt(
                    target_perm_obs, target_ended
                )
                # # Generate target_action from target_viewpoint_ids
                # # Convert viewpoint IDs to action indices (1-based, 0 = stop)
                # target_action = np.zeros(batch_size, dtype=np.int32)
                # for i in range(batch_size):
                #     candidate_dict = target_perm_obs[i].get("candidate", {})
                #     candidate_list = list(candidate_dict.keys())
                #     if (
                #         target_viewpoint_ids[i]
                #         and target_viewpoint_ids[i] in candidate_list
                #     ):
                #         # Action index is 1-based (1 = first candidate, 0 = stop)
                #         target_action[i] = (
                #             candidate_list.index(target_viewpoint_ids[i]) + 1
                #         )
                #     else:
                #         # Stop action (None or invalid viewpoint)
                #         target_action[i] = 0

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

            # 生成mask
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

            # 统计掩码个数
            mask_action = 1 - critical_a_t
            num_mask = torch.sum(mask_action[~ended])
            num_action_total += torch.sum(torch.ones_like(mask_action)[~ended])
            num_mask_total += num_mask

            # 确定真实动作
            mask_action_copy = critical_a_t.cpu().numpy()
            real_action = []
            for i in range(batch_size):
                # modify the mask
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
                        # 生成不等于target_action[i]的随机索引
                        while True:
                            idx = np.random.choice(n)
                            if idx != target_action[i]:
                                real_action.append(idx)
                                break
            action_seq.append(real_action[0])
            mask_pos.append(t)

            # get real viewpoint ids
            real_viewpoint_ids = []
            for i in range(batch_size):
                # 0 is stop
                candidate_dict = target_perm_obs[i].get("candidate", {})
                candidate_list = list(candidate_dict.keys())
                if real_action[i] > 0 and real_action[i] <= len(candidate_list):
                    real_viewpoint_ids.append(candidate_list[real_action[i] - 1])
                    print(candidate_list[real_action[i] - 1])
                else:
                    real_viewpoint_ids.append(None)

            # NOTE: MapGPT/NavGPT 的 real action 里面 0 是停止，
            # NOTE: 在 RecVLN 里面，candidate_len[i] - 1 是停止
            # ############### get new obs###########################
            # for target agent---------------------------
            if self.target_agent is not None:
                for i in range(batch_size):
                    target_traj[i]["a_t"][t] = real_action[i]

                # Determine stop actions
                target_a_t_stop = [a_t_i == 0 for a_t_i in real_action]

                # Prepare environment action
                # IMPORTANT: NavGPT_genAction now saves and restores environment state
                # So agent.env was NOT modified by agent_executor
                # We need to execute real_action here

                target_cpu_a_t = []
                for i in range(batch_size):
                    if target_a_t_stop[i] or target_ended[i]:
                        target_cpu_a_t.append(None)  # Stop action for NavGPT
                        target_just_ended[i] = True
                    else:
                        # # Convert real_action index to viewpoint ID
                        # candidate_dict = target_perm_obs[i].get("candidate", {})
                        # candidate_list = list(candidate_dict.keys())
                        # if real_action[i] > 0 and real_action[i] <= len(candidate_list):
                        #     viewpoint_id = candidate_list[real_action[i] - 1]
                        #     target_cpu_a_t.append(viewpoint_id)
                        # else:
                        #     target_cpu_a_t.append(None)
                        #     target_just_ended[i] = True
                        target_cpu_a_t.append(real_viewpoint_ids[i])

                # Execute real_action in NavGPT's environment
                # NavGPT_genAction only extracted the decision, didn't execute it
                for i, vp_id in enumerate(target_cpu_a_t):
                    if vp_id is not None:
                        # Check if real_action differs from target_action (mask was applied)
                        action_was_masked = real_action[i] != target_action[i]

                        # If action was masked, we need to update intermediate_steps
                        # because intermediate_steps contains target_action's step, but we executed random action
                        if action_was_masked and AgentAction is not None:
                            # Remove the last intermediate_step (target_action's step) if it exists
                            if (
                                hasattr(
                                    self.target_agent, "_accumulated_intermediate_steps"
                                )
                                and len(
                                    self.target_agent._accumulated_intermediate_steps
                                )
                                > 0
                            ):
                                # Remove the last step (target_action's step)
                                self.target_agent._accumulated_intermediate_steps = (
                                    self.target_agent._accumulated_intermediate_steps[
                                        :-1
                                    ]
                                )

                            # Also update agent_executor's intermediate_steps
                            if (
                                hasattr(self.target_agent, "agent_executor")
                                and hasattr(
                                    self.target_agent.agent_executor,
                                    "intermediate_steps",
                                )
                                and self.target_agent.agent_executor.intermediate_steps
                                is not None
                                and len(
                                    self.target_agent.agent_executor.intermediate_steps
                                )
                                > 0
                            ):
                                self.target_agent.agent_executor.intermediate_steps = (
                                    self.target_agent.agent_executor.intermediate_steps[
                                        :-1
                                    ]
                                )

                            # Also need to remove the last history entry if it was added by target_action
                            if (
                                hasattr(self.target_agent, "agent_executor")
                                and hasattr(self.target_agent.agent_executor, "agent")
                                and hasattr(
                                    self.target_agent.agent_executor.agent, "history"
                                )
                                and len(self.target_agent.agent_executor.agent.history)
                                > 0
                            ):
                                # Remove the last history entry (target_action's history)
                                self.target_agent.agent_executor.agent.history = (
                                    self.target_agent.agent_executor.agent.history[:-1]
                                )

                        # Execute the action and update history
                        # IMPORTANT: If action was NOT masked, agent_executor already executed it via _make_action tool
                        # So we only need to execute and update state if action WAS masked
                        if action_was_masked or do_inference_ is False:
                            # Execute the masked action
                            # IMPORTANT: make_equiv_action returns (turned_angle, new_obs)
                            # where turned_angle is a detailed angle description string like:
                            # "Turn heading direction 45.2 degrees from left 10.0 to right 35.2."
                            # This should be used for history update, not a fixed "Moved to viewpoint"
                            turned_angle, new_obs = self.target_agent.make_equiv_action(
                                [vp_id]
                            )

                            # Prepare observation string for intermediate_steps
                            if AgentAction is not None:
                                new_feature = new_obs.get("obs", "")
                                new_navigable = new_obs.get("candidate", {})
                                new_heading = np.rad2deg(new_obs.get("heading", 0))
                                new_elevation = np.rad2deg(new_obs.get("elevation", 0))
                                new_objects = new_obs.get("objects", {})

                                if self.target_agent.config.use_relative_angle:
                                    new_feature = (
                                        self.target_agent.modify_heading_angles(
                                            new_heading,
                                            new_feature,
                                            new_navigable,
                                            new_objects,
                                        )
                                    )
                                if self.target_agent.config.use_navigable:
                                    new_navigable_str = (
                                        self.target_agent.get_navigable_str(
                                            new_heading, new_elevation, new_navigable
                                        )
                                    )
                                else:
                                    new_navigable_str = ""

                                # Create observation string (similar to action_maker's return)
                                if self.target_agent.config.use_tool_chain:
                                    observation_str = f"\n\tAction_maker Thought:\n(Executed masked action)\n\tAction_maker Action:\nMoved to viewpoint\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable_str}"
                                elif self.target_agent.config.use_relative_angle:
                                    if self.target_agent.config.use_navigable:
                                        observation_str = f"\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable_str}"
                                    else:
                                        observation_str = (
                                            f"\n\tCurrent Viewpoint:\n{new_feature}"
                                        )
                                else:
                                    new_orientation = f"\nheading: {new_heading:.2f}, elevation: {new_elevation:.2f}"
                                    if self.target_agent.config.use_navigable:
                                        observation_str = f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable_str}"
                                    else:
                                        observation_str = f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}"

                                # Create AgentAction for intermediate_steps
                                actual_action = AgentAction(
                                    tool="action_maker",
                                    tool_input=f'"{vp_id}"',
                                    log=f'Thought: Executing masked action (random choice instead of target action)\nAction: action_maker\nAction Input: "{vp_id}"',
                                )
                                # Add to intermediate_steps
                                if not hasattr(
                                    self.target_agent, "_accumulated_intermediate_steps"
                                ):
                                    self.target_agent._accumulated_intermediate_steps = (
                                        []
                                    )
                                self.target_agent._accumulated_intermediate_steps.append(
                                    (actual_action, observation_str)
                                )

                                # Also update agent_executor's intermediate_steps
                                if hasattr(
                                    self.target_agent, "agent_executor"
                                ) and hasattr(
                                    self.target_agent.agent_executor,
                                    "intermediate_steps",
                                ):
                                    if (
                                        self.target_agent.agent_executor.intermediate_steps
                                        is None
                                    ):
                                        self.target_agent.agent_executor.intermediate_steps = (
                                            []
                                        )
                                    self.target_agent.agent_executor.intermediate_steps.append(
                                        (actual_action, observation_str)
                                    )

                                # Update last_observation for tool_chain mode (for next step's input_dict)
                                if self.target_agent.config.use_tool_chain:
                                    self.target_agent._last_observation = (
                                        observation_str
                                    )

                            # Update history for NavGPT (similar to what action_maker does)
                            # Use turned_angle (detailed angle description) instead of fixed "Moved to viewpoint"
                            # This matches the behavior in _make_action tool and collect_nav_info_rollout2
                            if self.target_agent.config.use_history_chain:
                                # Use history_chain to update history
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
                                        previous_action=turned_angle,  # Use turned_angle instead of "Moved to viewpoint"
                                    )
                                else:
                                    history = self.target_agent.get_history(
                                        new_obs,
                                        turned_angle,  # Use turned_angle instead of "Moved to viewpoint"
                                    )
                            else:
                                history = self.target_agent.get_history(
                                    new_obs,
                                    turned_angle,  # Use turned_angle instead of "Moved to viewpoint"
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
                                    "turned_angle": turned_angle,  # Use turned_angle instead of "Moved to viewpoint"
                                    "feature": new_obs.get("obs", ""),
                                    "history": history,
                                }
                                if "details" not in self.target_agent.traj[0]:
                                    self.target_agent.traj[0]["details"] = []
                                self.target_agent.traj[0]["details"].append(detail)
                        else:
                            # Action was NOT masked: agent_executor already executed it via _make_action tool
                            # History and intermediate_steps are already updated by agent_executor
                            # We just need to update last_observation for tool_chain mode (if needed)
                            if self.target_agent.config.use_tool_chain:
                                # Get the last observation from intermediate_steps (already updated by agent_executor)
                                if (
                                    hasattr(
                                        self.target_agent,
                                        "_accumulated_intermediate_steps",
                                    )
                                    and len(
                                        self.target_agent._accumulated_intermediate_steps
                                    )
                                    > 0
                                ):
                                    last_action, last_obs = (
                                        self.target_agent._accumulated_intermediate_steps[
                                            -1
                                        ]
                                    )
                                    self.target_agent._last_observation = last_obs

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

                # NavGPT handles history internally (already updated above)

            # ############### end of get new obs###########################
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            # NavGPT uses same action format as MapGPT (0=stop, 1+=action index)
            # real_action_surr = self.action_space_adaptor(
            #     "MapGPT", "RecVLN", real_action, candidate_leng
            # )

            real_action_surr = []
            for i in range(batch_size):
                candidate_list_surr = [
                    x["viewpointId"] for x in perm_obs[i].get("candidate", [])
                ]
                if real_viewpoint_ids[i] is not None:
                    real_action_surr.append(
                        candidate_list_surr.index(real_viewpoint_ids[i])
                    )
                else:
                    real_action_surr.append(len(candidate_list_surr))

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
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score
            total_reward += reward[0]
            total_discounted_reward += np.power(self.GAE, t) * reward[0]
            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))
            # For NavGPT, target_cpu_a_t contains viewpoint IDs or None
            target_ended[:] = np.logical_or(
                target_ended, np.array([vp_id is None for vp_id in target_cpu_a_t])
            )

            # Early exit if all ended
            if ended.all():
                break
        # end for

        print("total reward", self.if_succeed(perm_obs, traj))

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

    def compute_gradient(
        self, input_: torch.tensor, output_: torch.tensor
    ) -> torch.tensor:
        grad = torch.autograd.grad(
            output_, input_, retain_graph=True, grad_outputs=torch.ones_like(output_)
        )[0]

        return grad.cpu()
