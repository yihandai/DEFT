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


def read_nav_info():
    with open(
        os.path.join("NavGPT/nav_src/NavGPT/nav_24vp", "all_nav_outputs.json"), "r"
    ) as f:
        info = json.load(f)
    return info


def collect_nav_info(instr_id, t):
    """
    Collect navigation info from file for step t.

    Returns a dictionary containing:
    - nav_input: dict with LLM input components:
        - action_plan: the action plan string (stored once at t=0)
        - init_observation: initial observation string (for non-tool_chain mode)
        - observation: previous step observation (for tool_chain mode)
        - intermediate_steps: list of (action, observation) tuples for context maintenance
    - viewpoint_id: the viewpoint ID string executed (or None for stop)
    """
    info = read_nav_info()
    t = str(t)

    # Get base info
    step_info = info[instr_id][t]

    nav_info = {
        "nav_input": step_info.get("nav_input", {}),  # LLM input components
        "viewpoint_id": step_info.get("viewpoint_id", None),  # Action output
        # "a_t_list": step_info.get("a_t_list", []),  # For backward compatibility
    }

    return nav_info


def get_navgpt_viewpoint_id_from_file(instr_id, t, candidate_dict):
    """
    Get NavGPT viewpoint_id from file for step t.
    Returns the viewpoint_id string or None for stop action.

    NOTE: The nav_info file should contain:
    - nav_input: dict with LLM input components (action_plan, init_observation, observation)
    - viewpoint_id: the viewpoint ID string executed (or None for stop)
    """
    nav_info = collect_nav_info(instr_id, t)

    # Get viewpoint_id directly
    viewpoint_id = nav_info.get("viewpoint_id")
    if viewpoint_id is not None:
        # If it's None (stop action), return None
        # If it's a string, check if it's valid
        if viewpoint_id == "":
            return None
        elif isinstance(viewpoint_id, str) and viewpoint_id not in candidate_dict:
            # Invalid viewpoint_id, return None
            return None
        else:
            return viewpoint_id if viewpoint_id else None

    # # Fallback: try to get from a_t_list (for backward compatibility)
    # a_t_list = nav_info.get("a_t_list", [])
    # if not a_t_list:
    #     return None  # Stop action

    # # Get the most frequent action (similar to get_cls)
    # a_t_count = Counter(a_t_list)
    # most_frequent_a_t = a_t_count.most_common(1)[0][0]

    # # If most_frequent_a_t is already a viewpoint_id string, return it
    # if isinstance(most_frequent_a_t, str) and most_frequent_a_t in candidate_dict:
    #     return most_frequent_a_t

    # # If it's an index (int), convert to viewpoint_id
    # if isinstance(most_frequent_a_t, int):
    #     candidate_list = list(candidate_dict.keys())
    #     if most_frequent_a_t == 0:
    #         return None  # Stop action
    #     elif 1 <= most_frequent_a_t <= len(candidate_list):
    #         return candidate_list[most_frequent_a_t - 1]

    # # If we still don't have a valid viewpoint_id, return None
    # return None


def NavGPT_genAction_v2(
    agent: NavAgent,
    obs,
    t,
    previous_angle,
    do_inference=True,
    ended=None,
    description_update=None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Generate one action step from NavGPT agent by restoring state from saved file and performing LLM inference.

    Similar to NavGPT_genAction, but all state information is read from file instead of from agent.
    This function does NOT execute the action, only performs inference and returns the action.

    This function:
    1. Reads navigation info from file for time step t
    2. Temporarily restores agent state (intermediate_steps, action_plan, last_observation) from file
    3. Prepares LLM input using restored state and current observation (which may be perturbed)
    4. Performs LLM inference to get new action
    5. Restores environment state (does NOT execute action)
    6. Restores agent's original state (does NOT persist changes)
    7. Returns the new action output

    NOTE: This function does NOT update agent's internal state (_accumulated_intermediate_steps,
    _last_observation, etc.) because each call is independent and restores state from file.
    All information is read from file, not from agent's current state.

    Args:
        agent: NavGPT agent (NavAgent instance)
        obs: List of observations (batch) - may contain perturbed images
        t: Time step to restore to
        previous_angle: Previous angles (not used, kept for compatibility)
        do_inference: Whether to do inference (should be True for this function)
        ended: Array indicating which episodes have ended

    Returns:
        a_t: action indices (0 = stop, 1+ = action index, where index is 1-based)
        cand_nums: number of candidates
        nav_inputs: navigation inputs used for LLM inference
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
            viewpoint_ids.append(None)
            continue

        if not do_inference:
            a_t[i] = 0  # stop action
            cand_nums[i] = len(ob.get("candidate", {}))
            nav_inputs.append({})
            viewpoint_ids.append(None)
            continue

        # Get candidate viewpoints
        candidate_dict = ob.get("candidate", {})
        cand_nums[i] = len(candidate_dict)

        if len(candidate_dict) == 0:
            a_t[i] = 0  # stop if no candidates
            nav_inputs.append({})
            viewpoint_ids.append(None)
            continue

        try:
            instr_id = ob["instr_id"]

            # Save environment state before executing agent_executor (will restore after)
            env_state_before = agent.env.get_scan_viewpoint_heading()
            # Save original max_iterations
            original_max_iter = agent.agent_executor.max_iterations

            # Read navigation info from file for time step t
            nav_info = collect_nav_info(instr_id, t)
            nav_input = nav_info.get("nav_input", {})

            # Save agent's original state (will restore after inference)
            saved_cur_action_plan = getattr(agent, "cur_action_plan", None)
            saved_accumulated_intermediate_steps = getattr(
                agent, "_accumulated_intermediate_steps", None
            )
            saved_agent_executor_intermediate_steps = None
            if hasattr(agent, "agent_executor") and hasattr(
                agent.agent_executor, "intermediate_steps"
            ):
                saved_agent_executor_intermediate_steps = (
                    agent.agent_executor.intermediate_steps
                )
            saved_last_observation = getattr(agent, "_last_observation", None)
            # Save agent's history state (CRITICAL for proper scratchpad reconstruction)
            saved_agent_history = None
            if (
                hasattr(agent, "agent_executor")
                and hasattr(agent.agent_executor, "agent")
                and hasattr(agent.agent_executor.agent, "history")
            ):
                saved_agent_history = agent.agent_executor.agent.history.copy()

            # Temporarily restore agent state from saved data (for this inference only)
            # 1. Restore action_plan
            action_plan = nav_input.get("action_plan", "")
            if action_plan:
                agent.cur_action_plan = action_plan

            # 2. Restore intermediate_steps (deserialize from JSON)
            # IMPORTANT: The saved intermediate_steps contains steps from 0 to t (including step t's inference result).
            # To reconstruct the LLM input at step t, we need intermediate_steps from 0 to t-1 (the input to step t).
            # So we need to remove the last element (step t's result) to get the correct input.
            # However, if intermediate_steps_data has only one element (t=0), we keep it as is
            # (for step 0, there are no previous steps, so intermediate_steps should be empty).
            serialized_intermediate_steps = nav_input.get("intermediate_steps", [])
            restored_intermediate_steps = []

            # Remove the last element only if there are multiple elements
            if len(serialized_intermediate_steps) > 1:
                # Remove the last element (current step's inference result) to get the input to current step
                steps_to_use = serialized_intermediate_steps[:-1]
            else:
                # For t=0 or empty, intermediate_steps should be empty (no previous steps)
                steps_to_use = []

            if AgentAction is not None and len(steps_to_use) > 0:
                for step_dict in steps_to_use:
                    # Reconstruct AgentAction object from dictionary
                    if isinstance(step_dict, dict) and "tool" in step_dict:
                        tool = step_dict.get("tool", "")
                        tool_input = step_dict.get("tool_input", "")
                        log = step_dict.get("log", "")
                        observation = step_dict.get("observation", "")

                        # Create AgentAction object
                        agent_action = AgentAction(
                            tool=tool, tool_input=tool_input, log=log
                        )
                        restored_intermediate_steps.append((agent_action, observation))

            # Temporarily set agent's _accumulated_intermediate_steps (will not persist)
            if not hasattr(agent, "_accumulated_intermediate_steps"):
                agent._accumulated_intermediate_steps = []
            agent._accumulated_intermediate_steps = restored_intermediate_steps.copy()

            # Temporarily set agent_executor's intermediate_steps
            if hasattr(agent, "agent_executor") and hasattr(
                agent.agent_executor, "intermediate_steps"
            ):
                agent.agent_executor.intermediate_steps = (
                    restored_intermediate_steps.copy()
                )

            # 3. Restore history (CRITICAL for proper agent_scratchpad reconstruction)
            # VLNAgent._construct_scratchpad uses history[nav_step] instead of observation
            # for action_maker tools, and get_full_inputs uses history[0] as init_observation
            # IMPORTANT: The saved history contains steps from 0 to t (including step t's execution result).
            # To reconstruct the LLM input at step t, we need history from 0 to t-1 (the input to step t).
            # So we need to remove the last element (step t's execution result) to get the correct input.
            # However, if history_list has only one element (t=0, only initial observation), we keep it as is.
            history_list = nav_input.get("history", [])
            if (
                hasattr(agent, "agent_executor")
                and hasattr(agent.agent_executor, "agent")
                and hasattr(agent.agent_executor.agent, "history")
                and len(history_list) > 0
            ):
                # Remove the last element only if there are multiple elements
                # (for t=0, history_list has only history[0], which is the initial observation, so we keep it)
                if len(history_list) > 1:
                    # Remove the last element (current step's execution result) to get the input to current step
                    restored_history = history_list[:-1]
                else:
                    # For t=0, history_list has only history[0] (initial observation), keep it as is
                    restored_history = history_list.copy()
                agent.agent_executor.agent.history = restored_history

            # Set max_iterations to 1 for step-by-step execution
            agent.agent_executor.max_iterations = 1

            # Restore environment state before preparing observation
            agent.env.set_scan_viewpoint_heading(env_state_before)

            # Prepare input using restored action_plan and init_observation from file
            # IMPORTANT: According to VLNAgent.get_full_inputs logic, if intermediate_steps exist,
            # we should use history[0] instead of saved init_observation
            saved_init_observation = nav_input.get("init_observation", "")
            # Use restored_history (which excludes the last element) for init_observation
            restored_history = (
                agent.agent_executor.agent.history
                if hasattr(agent.agent_executor.agent, "history")
                else []
            )
            if len(restored_intermediate_steps) > 0 and len(restored_history) > 0:
                # Use history[0] as init_observation when intermediate_steps exist
                init_observation = restored_history[0]
            else:
                # Use saved init_observation when no intermediate_steps
                init_observation = saved_init_observation

            # NOTE: -------------------update observation-------------------
            if description_update is not None and isinstance(description_update, dict):
                # Get current observation to access obs_list and objects
                cur_obs = (
                    agent.env._get_obs()[0] if hasattr(agent.env, "_get_obs") else ob
                )
                obs_list = cur_obs.get("obs_list", [])  # 8 observations (indices 0-7)
                objects_dict = cur_obs.get("objects", [])  # objects dict

                # Apply updates to obs_list and objects
                # description_update is a dict: {target_cand_idx: {"description": ..., "objects": ...}}
                for candidate_idx, update_info in description_update.items():
                    if not isinstance(update_info, dict):
                        continue

                    # candidate_idx is the key (target_cand_idx, 1-8)
                    # Convert candidate_idx (1-8) to obs_list index (0-7)
                    try:
                        candidate_idx_int = int(candidate_idx)
                    except (ValueError, TypeError):
                        continue

                    obs_list_idx = candidate_idx_int

                    # Update obs_list if index is valid
                    if 0 <= obs_list_idx < len(obs_list):
                        updated_desc = update_info.get("description")
                        if updated_desc:
                            # obs_list[obs_list_idx] is a string like "down: ...\nmiddle: ...\ntop: ..."
                            # The updated description is already merged, so we can use it directly
                            obs_list[obs_list_idx] = updated_desc

                    # Update objects dict
                    updated_objects = update_info.get("objects")
                    if updated_objects and isinstance(objects_dict, dict):
                        # objects_dict structure: {viewpoint_id: {object_name: {heading, distance}}}
                        # We need to update objects for the candidate view corresponding to candidate_idx
                        # But objects_dict is keyed by viewpoint_id, not candidate_idx
                        # So we need to find the viewpoint_id for this candidate_idx
                        # For now, we'll update the objects if they're provided
                        if isinstance(updated_objects, dict):
                            # Merge updated objects into objects_dict
                            # This is a simplified approach - may need refinement based on actual structure
                            objects_dict[candidate_idx_int] = updated_objects

                # Store updated obs_list and objects back to observation
                # We'll use these when building init_observation or updating history
                updated_obs_list = obs_list
                updated_objects_dict = objects_dict
            else:
                # No description_update, use original obs_list and objects
                cur_obs = (
                    agent.env._get_obs()[0] if hasattr(agent.env, "_get_obs") else ob
                )
                updated_obs_list = cur_obs.get("obs_list", [])
                updated_objects_dict = cur_obs.get("objects", [])

            # Build updated observation using updated_obs_list and updated_objects_dict
            # This will be used to update both init_observation and history
            cur_obs_for_observation = (
                agent.env._get_obs()[0] if hasattr(agent.env, "_get_obs") else ob
            )
            heading = np.rad2deg(cur_obs_for_observation.get("heading", 0))
            elevation = np.rad2deg(cur_obs_for_observation.get("elevation", 0))
            navigable = candidate_dict

            # Build updated feature using modify_heading_angles
            if agent.config.use_relative_angle:
                updated_feature = agent.modify_heading_angles(
                    heading, updated_obs_list, navigable, updated_objects_dict
                )
            else:
                # If not using relative_angle, just use the current view's observation
                current_target_cand_idx = int((heading - 22.5) // 45)  # 0-7
                updated_feature = (
                    updated_obs_list[current_target_cand_idx]
                    if 0 <= current_target_cand_idx < len(updated_obs_list)
                    else cur_obs_for_observation.get("obs", "")
                )

            # Build updated observation string
            if agent.config.use_navigable:
                navigable_str = agent.get_navigable_str(heading, elevation, navigable)
            else:
                navigable_str = ""

            orientation = f"\nheading: {heading:.2f}, elevation: {elevation:.2f}"
            if agent.config.use_relative_angle:
                if agent.config.use_navigable:
                    updated_observation_str = f"\n\tCurrent Viewpoint:\n{updated_feature}\n\tNavigable Viewpoints:\n{navigable_str}"
                else:
                    updated_observation_str = (
                        f"\n\tCurrent Viewpoint:\n{updated_feature}"
                    )
            else:
                if agent.config.use_navigable:
                    updated_observation_str = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{updated_feature}\n\tNavigable Viewpoints:\n{navigable_str}"
                else:
                    updated_observation_str = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{updated_feature}"

            # Update init_observation and history
            # IMPORTANT: For t=0, we update init_observation (or history[0])
            # For t != 0, we need to update history[t] (current step's observation)
            is_t0 = len(restored_intermediate_steps) == 0

            if is_t0:
                # For t=0, update init_observation (or history[0])
                init_observation = updated_observation_str
                # Also update history[0] if it exists
                if (
                    hasattr(agent.agent_executor, "agent")
                    and hasattr(agent.agent_executor.agent, "history")
                    and len(agent.agent_executor.agent.history) > 0
                ):
                    agent.agent_executor.agent.history[0] = updated_observation_str
            else:
                # For t != 0, update history[t] (current step's observation)
                # history[t] is the last element in history (before we removed it)
                if (
                    hasattr(agent.agent_executor, "agent")
                    and hasattr(agent.agent_executor.agent, "history")
                    and len(agent.agent_executor.agent.history) > 0
                ):
                    # Update the last element (current step's observation)
                    agent.agent_executor.agent.history[-1] = updated_observation_str
                # Also update init_observation if it's being used
                if init_observation:
                    init_observation = updated_observation_str

            # If init_observation is still not set (should not happen if file is correct)
            if not init_observation:
                init_observation = updated_observation_str

            input_dict = {
                "action_plan": (
                    agent.cur_action_plan if hasattr(agent, "cur_action_plan") else ""
                ),
                "init_observation": init_observation,
            }

            # Also pass manual_intermediate_steps in input_dict as a fallback
            input_dict_with_history = input_dict.copy()
            if (
                restored_intermediate_steps is not None
                and len(restored_intermediate_steps) > 0
            ):
                input_dict_with_history["manual_intermediate_steps"] = (
                    restored_intermediate_steps.copy()
                )

            # Call agent_executor to perform LLM inference
            # Add error handling for API errors (KeyError: 'content', etc.)
            viewpoint_id = None
            output = None
            try:
                output = agent.agent_executor(input_dict_with_history)
            except KeyError as e:
                # Handle KeyError: 'content' and other KeyError exceptions
                error_msg = str(e)
                print(
                    f"KeyError in NavGPT_genAction_v2 at step {t} for obs {i}: {error_msg}"
                )
                if "'content'" in error_msg:
                    print("  API response missing 'content' field. This may indicate:")
                    print("  - API returned an error response")
                    print("  - API response format changed")
                    print("  - Network/connection issue")
                # Set viewpoint_id to None (stop action) on error
                viewpoint_id = None
            except Exception as api_error:
                # Handle other API-related errors (network, timeout, etc.)
                print(
                    f"API error in NavGPT_genAction_v2 at step {t} for obs {i}: {api_error}"
                )
                import traceback

                traceback.print_exc()
                # Set viewpoint_id to None (stop action) on error
                viewpoint_id = None

            # Extract viewpoint ID from the result (only if output was successfully obtained)
            if output is not None:
                # Check if orchestrator decided to stop (Final Answer)
                output_text = output.get("output", "")
                if "Finished!" in output_text or "Final Answer" in output_text:
                    viewpoint_id = None  # Stop action
                else:
                    # Extract action from intermediate_steps
                    intermediate_steps = output.get("intermediate_steps", [])
                    if len(intermediate_steps) > 0:
                        # Get the last action (most recent tool call)
                        last_action, observation = intermediate_steps[-1]
                        if last_action.tool == "action_maker":
                            viewpoint_id = (
                                last_action.tool_input.strip('"').strip("'").strip()
                            )
                        elif last_action.tool == "back_tracer":
                            viewpoint_id = (
                                last_action.tool_input.strip('"').strip("'").strip()
                            )
                        else:
                            viewpoint_id = None
                    else:
                        viewpoint_id = None
            # If output is None (due to error), viewpoint_id is already set to None above

            # Restore environment state after extracting viewpoint_id (do NOT execute action)
            agent.env.set_scan_viewpoint_heading(env_state_before)

            # Restore original max_iterations
            agent.agent_executor.max_iterations = original_max_iter

            # Restore agent's original state (do NOT persist changes from this inference)
            if saved_cur_action_plan is not None:
                agent.cur_action_plan = saved_cur_action_plan
            elif hasattr(agent, "cur_action_plan"):
                # If it didn't exist before, remove it
                delattr(agent, "cur_action_plan")

            if saved_accumulated_intermediate_steps is not None:
                agent._accumulated_intermediate_steps = (
                    saved_accumulated_intermediate_steps
                )
            elif hasattr(agent, "_accumulated_intermediate_steps"):
                agent._accumulated_intermediate_steps = []

            if saved_agent_executor_intermediate_steps is not None:
                agent.agent_executor.intermediate_steps = (
                    saved_agent_executor_intermediate_steps
                )
            elif hasattr(agent, "agent_executor") and hasattr(
                agent.agent_executor, "intermediate_steps"
            ):
                agent.agent_executor.intermediate_steps = []

            # Restore agent's history (CRITICAL)
            if saved_agent_history is not None:
                if (
                    hasattr(agent, "agent_executor")
                    and hasattr(agent.agent_executor, "agent")
                    and hasattr(agent.agent_executor.agent, "history")
                ):
                    agent.agent_executor.agent.history = saved_agent_history

            if saved_last_observation is not None:
                agent._last_observation = saved_last_observation
            elif hasattr(agent, "_last_observation"):
                # Remove it if it didn't exist before
                delattr(agent, "_last_observation")

            # Convert viewpoint_id to action index
            # Note: NavGPT returns viewpoint IDs (strings), we need to convert to 1-based indices
            # where 0 = stop, 1+ = action index (matching MapGPT's format)
            if viewpoint_id and viewpoint_id in candidate_dict:
                # Action index is 1-based (1 = first candidate, 0 = stop)
                candidate_list = list(candidate_dict.keys())
                a_t[i] = candidate_list.index(viewpoint_id) + 1
                viewpoint_ids.append(viewpoint_id)
            else:
                # Stop action (None, empty string, or invalid viewpoint)
                a_t[i] = 0
                viewpoint_ids.append(None)

            nav_inputs.append(input_dict)

        except Exception as e:
            print(f"Error in NavGPT_genAction_v2 at step {t} for obs {i}: {e}")
            import traceback

            traceback.print_exc()
            a_t[i] = 0  # stop on error
            cand_nums[i] = len(candidate_dict)
            nav_inputs.append({})
            # Restore max_iterations, environment state, and agent state in case of error
            try:
                if hasattr(agent, "agent_executor") and hasattr(
                    agent.agent_executor, "max_iterations"
                ):
                    agent.agent_executor.max_iterations = original_max_iter
                    # Try to restore environment state
                    if "env_state_before" in locals():
                        agent.env.set_scan_viewpoint_heading(env_state_before)
                # Restore agent state
                if (
                    "saved_cur_action_plan" in locals()
                    and saved_cur_action_plan is not None
                ):
                    agent.cur_action_plan = saved_cur_action_plan
                if (
                    "saved_accumulated_intermediate_steps" in locals()
                    and saved_accumulated_intermediate_steps is not None
                ):
                    agent._accumulated_intermediate_steps = (
                        saved_accumulated_intermediate_steps
                    )
                if (
                    "saved_agent_executor_intermediate_steps" in locals()
                    and saved_agent_executor_intermediate_steps is not None
                ):
                    agent.agent_executor.intermediate_steps = (
                        saved_agent_executor_intermediate_steps
                    )
                # Restore agent's history
                if (
                    "saved_agent_history" in locals()
                    and saved_agent_history is not None
                ):
                    if (
                        hasattr(agent, "agent_executor")
                        and hasattr(agent.agent_executor, "agent")
                        and hasattr(agent.agent_executor.agent, "history")
                    ):
                        agent.agent_executor.agent.history = saved_agent_history
                if (
                    "saved_last_observation" in locals()
                    and saved_last_observation is not None
                ):
                    agent._last_observation = saved_last_observation
            except Exception:
                pass

    return a_t, cand_nums, nav_inputs, viewpoint_ids


# collect_navgpt_nav_info_rollout2 has been moved to NavGPT/nav_src/agent.py as NavAgent.collect_nav_info_rollout2
# This is a wrapper function for backward compatibility
def collect_navgpt_nav_info_rollout2(
    agent: NavAgent, env=None, reset=True, output_file=None, max_steps=20
):
    """
    Wrapper function for NavAgent.collect_nav_info_rollout2.

    This function is kept for backward compatibility.
    The actual implementation is now in NavGPT/nav_src/agent.py as NavAgent.collect_nav_info_rollout2.

    Args:
        agent: NavGPT agent (NavAgent instance)
        env: Environment instance (optional, defaults to agent.env)
        reset: Whether to reset the environment
        output_file: Path to output JSON file (default: MapGPT/nav_30/all_nav_outputs.json)
        max_steps: Maximum number of steps to execute (default: 20)

    Returns:
        Dictionary mapping instr_id -> {t -> nav_info}
    """
    return agent.collect_nav_info_rollout2(
        env=env, reset=reset, output_file=output_file, max_steps=max_steps
    )


class FeatureAgent_NavGPT(MaskAgent):
    def __init__(self, env, results_path, tok, episode_len=20, args_target=None):
        super(FeatureAgent_NavGPT, self).__init__(
            env, results_path, tok, episode_len, args_target=args_target
        )
        rank = 0

        self.vln_bert.eval()
        self.critic.eval()
        self.critical_head.eval()
        self.critic4mask.eval()
        self.target_agent = NavAgent(env, target_args)
        if args.feature_level_baseline == "smdl":
            self.exp = CubSubModularExplanationV2(self.vln_bert, self.critical_head)
        else:
            self.exp = Exp(self.vln_bert, self.critical_head)

        self.causual = CausalMetric(
            call_fn=NavGPT_genAction_v2,
            substrate_fn=np.zeros_like,
            H=480,
            W=640,
            target="NavGPT",
        )

        # self.groud_truth = self.get_gt()
        self.VERSION = "v1"
        # saliency map location
        saliency_map_dir = os.path.join(
            "snap", args.name + self.VERSION, "saliency_map_pixel"
        )
        if not os.path.exists(saliency_map_dir):
            os.makedirs(saliency_map_dir)
        self.saliency_map_dir = saliency_map_dir
        # causal metric location
        causal_metric_dir = os.path.join(
            "snap", args.name + self.VERSION, "causal_metric_pixel_2_del"
        )
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

    def test(self, iters=None, **kwargs):
        test_model = args.feature_level_baseline
        assert test_model is not None, "test_model cannot be None"
        phase2 = False
        phase_update_obs = False
        phase_inference = False
        phase_merge = False
        mu = True
        # test_num = 1
        if args.update_inference == "update":
            phase2 = True
            phase_update_obs = True
        elif args.update_inference == "inference":
            phase_inference = True

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
                    for traj in self.rollout_mask_test_navgpt_feature_phase2(
                        test_model=test_model
                    ):
                        self.loss = 0
                        self.results[traj["instr_id"]] = traj["path"]
            else:  # Do a full round
                lets_start = False
                while True:
                    # for i in range(test_num):
                    traj = self.rollout_mask_test_navgpt_feature_phase2(
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
                # we use 4 groups sample
                traj = self.rollout_mask_test_navgpt_feature_phase_update_obs(
                    test_model=test_model,
                    mode="ins",
                    reset=True,
                    perturb_ratio=0.25,
                )
                traj = self.rollout_mask_test_navgpt_feature_phase_update_obs(
                    test_model=test_model,
                    mode="del",
                    reset=False,
                    perturb_ratio=0.25,
                )
                traj = self.rollout_mask_test_navgpt_feature_phase_update_obs(
                    test_model=test_model,
                    mode="ins",
                    reset=False,
                    perturb_ratio=0.5,
                )
                traj = self.rollout_mask_test_navgpt_feature_phase_update_obs(
                    test_model=test_model,
                    mode="del",
                    reset=False,
                    perturb_ratio=0.5,
                )
                traj = self.rollout_mask_test_navgpt_feature_phase_update_obs(
                    test_model=test_model,
                    mode="ins",
                    reset=False,
                    perturb_ratio=0.75,
                )
                traj = self.rollout_mask_test_navgpt_feature_phase_update_obs(
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

        if phase_inference:
            self.env.reset_epoch(
                shuffle=(iters is not None)
            )  # If iters is not none, shuffle the env batch
            self.losses = []
            self.results = {}
            # We rely on env showing the entire batch before repeating anything
            looped = False
            self.loss = 0

            # self.strt = False
            while True:
                # traj = self.rollout_mask_test_navgpt_feature_phase_inference(
                #     test_model=test_model,
                #     mode="ins",
                #     reset=True,
                #     perturb_ratio=0.25,
                # )
                traj = self.rollout_mask_test_navgpt_feature_phase_inference(
                    test_model=test_model,
                    mode="del",
                    # reset=False,
                    reset=True,
                    perturb_ratio=0.25,
                )
                # traj = self.rollout_mask_test_navgpt_feature_phase_inference(
                #     test_model=test_model,
                #     mode="ins",
                #     reset=False,
                #     perturb_ratio=0.5,
                # )
                # traj = self.rollout_mask_test_navgpt_feature_phase_inference(
                #     test_model=test_model,
                #     mode="del",
                #     reset=False,
                #     perturb_ratio=0.5,
                # )
                # traj = self.rollout_mask_test_navgpt_feature_phase_inference(
                #     test_model=test_model,
                #     mode="ins",
                #     reset=False,
                #     perturb_ratio=0.75,
                # )
                # traj = self.rollout_mask_test_navgpt_feature_phase_inference(
                #     test_model=test_model,
                #     mode="del",
                #     reset=False,
                #     perturb_ratio=0.75,
                # )
                # exit(0)
                if traj["instr_id"] in self.results:
                    looped = True
                else:
                    self.results[traj["instr_id"]] = traj["path"]
                if looped:
                    break

        if mu:
            muFidelity = self.causual.compute_muFidelity(self.causal_metric_dir)
            print("muFidelity", muFidelity)

        # if original_image:
        #     self.env.reset_epoch(
        #         shuffle=(iters is not None)
        #     )  # If iters is not none, shuffle the env batch
        #     self.losses = []
        #     self.results = {}
        #     # We rely on env showing the entire batch before repeating anything
        #     looped = False
        #     self.loss = 0
        #     # count_i = 0
        #     while True:
        #         # we use 4 groups sample
        #         traj = self.rollout_mask_test_navgpt_feature_phase3(
        #             # test_model="IG_temporal",
        #             test_model=test_model,
        #             # mode="del",
        #             mode="ins",
        #             reset=True,
        #             perturb_ratio=1.0,
        #         )
        #         # traj = self.rollout_mask_test_navgpt_feature_phase3(
        #         #     # test_model="IG_temporal",
        #         #     test_model=test_model,
        #         #     mode="del",
        #         #     reset=False,
        #         #     perturb_ratio=0.0,
        #         # )
        #         if traj["instr_id"] in self.results:
        #             looped = True
        #         else:
        #             self.results[traj["instr_id"]] = traj["path"]
        #         if looped:
        #             break

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

    def get_gt(self, gt_file="navgpt_feature.json"):
        with open(gt_file, "r") as f:
            data = json.load(f)
        return data

    def merge_json_files(self, input_dir="tmp_traj", output_file="navgpt_feature.json"):
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

    def rollout_mask_test_navgpt_feature_phase2(
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

            # if test_model in ["IG", "temporal", "IG_temporal", "guided_IG"]:
            # scanId = perm_obs[0]["scanId"]
            # viewpointId = perm_obs[0]["viewpointId"]

            # scanId = perm_obs[0]["scan"]
            # viewpointId = perm_obs[0]["viewpoint"]
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
                    # f"{scanId}",
                    # f"{viewpointId}",
                    f"{instr_id}",
                    f"{t}",
                    f"attr_map.npy",
                ),
                attr_map.cpu().numpy(),
            )
            np.save(
                os.path.join(
                    self.saliency_map_dir,
                    # f"{scanId}",
                    # f"{viewpointId}",
                    f"{instr_id}",
                    f"{t}",
                    f"attr_rank.npy",
                ),
                attr_rank.cpu().numpy(),
            )
            # Get action from file for NavGPT
            # Read viewpoint_id from file
            viewpoint_id = get_navgpt_viewpoint_id_from_file(
                perm_obs[0]["instr_id"], t, target_perm_obs[0].get("candidate", {})
            )
            # # Convert viewpoint_id to action index for compatibility
            # candidate_dict = target_perm_obs[0].get("candidate", {})
            # if viewpoint_id and viewpoint_id in candidate_dict:
            #     candidate_list = list(candidate_dict.keys())
            #     target_action = [candidate_list.index(viewpoint_id) + 1]
            # else:
            #     target_action = [0]  # Stop action
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
            # # Convert to RecVLN format for action_space_adaptor
            # target_action_surr = self.action_space_adaptor(
            #     "MapGPT", "RecVLN", target_action, candidate_leng
            # )

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

            # load the saliency map and rank
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

            description_update_file = os.path.join(
                description_update_dir, "description_update.json"
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

    def rollout_mask_test_navgpt_feature_phase_inference(
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
        instr_id = obs[0]["instr_id"]
        print("-----instr_id------", instr_id)
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

        # if instr_id == "6992_0":
        #     self.strt = True
        # if not self.strt:
        #     return traj[0]

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

            # load the saliency map and rank
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

                    # # Flatten for hard vote if needed
                    # if len(saliency_map.shape) > 1:
                    #     attr_map_list.append(saliency_map)
                    #     attr_rank_list.append(attr_rank.flatten())
                    # else:
                    #     attr_map_list.append(saliency_map)
                    #     attr_rank_list.append(attr_rank)
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
