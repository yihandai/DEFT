"""Agent that interacts with Matterport3D simulator via a hierarchical planning approach."""

import json
import yaml
import re
import warnings
import os
import numpy as np
from typing import (
    Any,
    Callable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Dict,
    Union,
)

# Support both package and non-package imports
try:
    from .env import R2RNavBatch
    from .agent_base import BaseAgent
except ImportError:
    # Fallback for non-package imports (when called from outside the package)
    from env import R2RNavBatch
    from agent_base import BaseAgent
from argparse import Namespace

from langchain import HuggingFacePipeline
from langchain.agents.agent import AgentExecutor, AgentAction, AgentOutputParser
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.tools import Tool
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    BaseOutputParser,
    OutputParserException,
)
from langchain.base_language import BaseLanguageModel

from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS

# Support both package and non-package imports
try:
    from .prompt.planner_prompt import (
        ACTION_PROMPT,
        HISTORY_PROMPT,
        PLANNER_PROMPT,
        BACK_TRACE_PROMPT,
        MAKE_ACTION_TOOL_NAME,
        MAKE_ACTION_TOOL_DESCRIPTION,
        BACK_TRACE_TOOL_NAME,
        BACK_TRACE_TOOL_DESCRIPTION,
        VLN_ORCHESTRATOR_PROMPT,
        VLN_GPT4_PROMPT,
        VLN_GPT35_PROMPT,
    )
except ImportError:
    # Fallback for non-package imports (when called from outside the package)
    from prompt.planner_prompt import (
        ACTION_PROMPT,
        HISTORY_PROMPT,
        PLANNER_PROMPT,
        BACK_TRACE_PROMPT,
        MAKE_ACTION_TOOL_NAME,
        MAKE_ACTION_TOOL_DESCRIPTION,
        BACK_TRACE_TOOL_NAME,
        BACK_TRACE_TOOL_DESCRIPTION,
        VLN_ORCHESTRATOR_PROMPT,
        VLN_GPT4_PROMPT,
        VLN_GPT35_PROMPT,
    )
import time

FINAL_ANSWER_ACTION = "Final Answer:"
EXCEPTION_TOOL_NAME = "_Exception"
MAX_SCRATCHPAD_LENGTH = 7000

MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' after 'Thought:"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)


class NavGPTOutputParser(AgentOutputParser):
    """MRKL Output parser for the chat agent."""

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*\"?([a-fA-F0-9]{32})\"?"
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise OutputParserException(
                    f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                )
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            # ensure if its a well formed SQL query we don't remove any trailing " chars
            if tool_input.startswith("SELECT ") is False:
                tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")

    @property
    def _type(self) -> str:
        return "mrkl-NavGPT"


class VLNAgent(ZeroShotAgent):

    history: Optional[List[str]] = None

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[str, List[BaseMessage]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        nav_step = 1
        for i, (action, observation) in enumerate(intermediate_steps):
            thoughts += action.log
            if (i == len(intermediate_steps) - 1) or (
                action.tool != MAKE_ACTION_TOOL_NAME
            ):
                thoughts += (
                    f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
                )
            else:
                thoughts += f"\n{self.observation_prefix}{self.history[nav_step]}\n{self.llm_prefix}"
                nav_step += 1
        return thoughts

    # def get_full_inputs(
    #     self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    # ) -> Dict[str, Any]:
    #     """Create the full inputs for the LLMChain from intermediate steps."""
    #     thoughts = self._construct_scratchpad(intermediate_steps)[
    #         -MAX_SCRATCHPAD_LENGTH:
    #     ]
    #     new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
    #     if len(intermediate_steps) == 0:
    #         full_inputs = {**kwargs, **new_inputs}
    #     else:
    #         kwargs["init_observation"] = self.history[0]
    #         full_inputs = {**kwargs, **new_inputs}
    #     return full_inputs
    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        # Allow manual override of intermediate_steps from kwargs (for rollout2)
        if "manual_intermediate_steps" in kwargs and len(
            kwargs["manual_intermediate_steps"]
        ) > len(intermediate_steps):
            intermediate_steps = kwargs["manual_intermediate_steps"]
            del kwargs[
                "manual_intermediate_steps"
            ]  # Remove from kwargs to avoid passing to LLM

        thoughts = self._construct_scratchpad(intermediate_steps)[
            -MAX_SCRATCHPAD_LENGTH:
        ]
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        if len(intermediate_steps) == 0:
            full_inputs = {**kwargs, **new_inputs}
        else:
            kwargs["init_observation"] = self.history[0]
            full_inputs = {**kwargs, **new_inputs}
        return full_inputs


class NavAgent(BaseAgent):
    def __init__(self, env: R2RNavBatch, config: Namespace):
        """
        Initialize the LLM Navigation Agent.

        Args:
            env: The Matterport3D environment.
            config: The configuration.
        """
        super().__init__(env)
        self.config = config

        if config.llm_model_name.split("-")[0] == "gpt":
            self.llm = OpenAI(
                temperature=config.temperature,
                model_name=config.llm_model_name,
                seed=42,
            )
        # if config.llm_model_name.split("-")[0] == "gpt":
        #     from LLMs.Custom_Openai import Custom_Openai

        #     self.llm = Custom_Openai(
        #         model=config.llm_model_name,
        #         max_tokens=10000,
        #         response_format="str",
        #     )
        elif config.llm_model_name == "llama-2-13b":
            from .LLMs.Langchain_llama import Custom_Llama

            ckpt_dir = "LLMs/llama/llama-2-13b"
            tokenizer_path = "LLMs/llama/tokenizer.model"
            self.llm = Custom_Llama.from_model_id(
                temperature=config.temperature,
                ckpt_dir=ckpt_dir,
                tokenizer_path=tokenizer_path,
                max_seq_len=8000,
                max_gen_len=500,
                max_batch_size=1,
            )
        # elif config.llm_model_name == 'Vicuna-v1.5-13b':
        #     from LLMs.Langchain_Vicuna import Custom_Vicuna
        #     self.llm = Custom_Vicuna.from_config(
        #         config = config,
        #     )
        # elif config.llm_model_name == 'FlanT5XXL':
        #     from LLMs.Langchain_FlanT5 import Custom_FlanT5
        #     self.llm = Custom_FlanT5.from_config(
        #         config = config,
        #     )
        # elif config.llm_model_name == 'Emu-14B':
        #     from LLMs.Langchain_Emu import Custom_Emu
        #     self.llm = Custom_Emu.from_config(
        #         config = config,
        #     )
        # else:
        #     from LLMs.Langchain_InstructBLIP import Custom_NavGPT_InstructBLIP
        #     self.llm = Custom_NavGPT.from_config(
        #         config = config,
        #     )

        self.output_parser = NavGPTOutputParser()
        self.agent_executor = self.create_vln_agent()

        plan_prompt = PromptTemplate(
            template=PLANNER_PROMPT,
            input_variables=["instruction"],
        )
        self.plan_chain = LLMChain(llm=self.llm, prompt=plan_prompt)

    def parse_action(self, llm_output: str) -> Tuple[str, str]:
        regex = r"(.*?)Final Answer:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

        thought = match.group(1).strip()
        action = match.group(2).strip(" ").strip('"').strip("'")

        return thought, action

    def get_his_viewpoints(self) -> str:
        """Return the history of visited viewpoints for back tracing."""
        his_viewpoints = ""
        # The last vp is not included in the history
        for i, detail in enumerate(self.traj[0]["details"][:-1]):
            viewpointID = detail["viewpointID"]
            viewpoint_ob = detail["feature"]
            his_viewpoints += (
                f"Step {i+1}. Viewpoint ID '{viewpointID}':\n {viewpoint_ob}\n\n"
            )
        return his_viewpoints

    def get_history(self, obs: dict, angle: str) -> str:
        """Return the history of actions taken."""
        history = f'{angle}\nCurrent viewpoint "{obs["viewpoint"]}": Scene from the viewpoint is a {obs["obs_summary"]}'
        return history

    def get_navigable_str(
        self, cur_heading: float, cur_elevation: float, navigable: dict
    ) -> str:
        """Return the navigable viewpoints as a string."""
        navigable_str = ""

        for vp, items in navigable.items():
            heading = np.rad2deg(items["heading"])
            elevation = np.rad2deg(items["elevation"])
            distance = items["distance"]
            rel_heading = heading - cur_heading
            rel_elevation = elevation - cur_elevation

            if self.config.use_relative_angle:
                navigable_str += f"'{vp}':\nheading: {rel_heading:.2f}, elevation: {rel_elevation:.2f}, distance: {distance:.2f}\n"
            else:
                navigable_str += f"'{vp}':\nheading: {heading:.2f}, elevation: {elevation:.2f}, distance: {distance:.2f}\n"

        return navigable_str

    def modify_heading_angles(
        self, heading_angle, observation_list, candidate_dict, object_list
    ):
        # Function to normalize an angle to the range of -180 to 180
        def normalize_angle(angle):
            while angle > 180:
                angle -= 360
            while angle <= -180:
                angle += 360
            return angle

        def angle_to_left_right(angle):
            return f"left {-angle:.2f}" if angle < 0 else f"right {angle:.2f}"

        # Define the directions
        directions = [
            "Front",
            "Front Right",
            "Right",
            "Rear Right",
            "Rear",
            "Rear Left",
            "Left",
            "Front Left",
        ]

        # Calculate the range of heading angles belonging to each direction
        range_idx = int((heading_angle - 22.5) // 45) + 1
        obs_idx = [(i + range_idx) % 8 for i in range(8)]

        # Initialize a dictionary to store the candidate viewpoints for each direction
        candidate_range = {}
        if not self.config.use_navigable:
            for viewpoint_id, viewpoint_data in candidate_dict.items():
                viewpoint_heading = np.rad2deg(viewpoint_data["heading"])
                vp_range_idx = int((viewpoint_heading - 22.5) // 45) + 1
                rel_viewpoint_heading = viewpoint_heading - heading_angle
                rel_viewpoint_heading = normalize_angle(rel_viewpoint_heading)
                rel_viewpoint_heading = angle_to_left_right(rel_viewpoint_heading)
                vp_description = (
                    rel_viewpoint_heading + f', {viewpoint_data["distance"]:.2f}m'
                )
                # rel_range_idx = (vp_range_idx - range_idx) % 8
                candidate_range.setdefault(vp_range_idx, {}).update(
                    {viewpoint_id: vp_description}
                )

        # Calculate the relative angle ranges based on the heading angle
        angle_ranges = [
            (angle - 22.5 - heading_angle, angle + 22.5 - heading_angle)
            for angle in range(0, 360, 45)
        ]

        # Initialize an empty list to store the formatted strings
        formatted_strings = []

        # Iterate through the directions, angle ranges, and observation strings
        for direction, idx in zip(directions, obs_idx):
            # Calculate the relative angles and normalize them
            rel_angle1 = normalize_angle(angle_ranges[idx][0])
            rel_angle2 = normalize_angle(angle_ranges[idx][1])

            # Convert the angles to "left n" or "right n"
            left_right1 = angle_to_left_right(rel_angle1)
            left_right2 = angle_to_left_right(rel_angle2)

            # Create the formatted string
            formatted_string = f"{direction}, range ({left_right1} to {left_right2}): \n'{observation_list[idx]}'"

            # Add the objects to the formatted string
            object_dict = {}
            if len(object_list[idx]) > 0:
                object = object_list[idx]
                for obj, obj_data in object.items():
                    rel_obj_heading = obj_data["heading"] - heading_angle
                    rel_obj_heading = normalize_angle(rel_obj_heading)
                    rel_obj_heading = angle_to_left_right(rel_obj_heading)
                    object_dict[obj] = f'{rel_obj_heading}, {obj_data["distance"]:.2f}m'
                formatted_string += f"\n{direction} Objects in 3m: {object_dict}"
            else:
                formatted_string += f"\n{direction} Objects in 3m: None"

            # Add the candidate viewpoints to the formatted string
            if candidate_range.get(idx):
                formatted_string += (
                    f"\n{direction} Navigable Viewpoints:{candidate_range[idx]}"
                )
            else:
                formatted_string += f"\n{direction} Navigable Viewpoints: None"

            # Add the formatted string to the list
            formatted_strings.append(formatted_string)

        # Join the formatted strings into a single output string
        output_string = "\n".join(formatted_strings)

        return output_string

    def init_trajecotry(self, obs: List[dict]):
        """Initialize the trajectory with the given observation."""
        # Record the navigation path
        self.traj = [
            {
                "instr_id": ob["instr_id"],
                "path": [[ob["viewpoint"]]],
                "details": [],
            }
            for ob in obs
        ]
        # Record the history of actions taken
        self.agent_executor.agent.history = [
            f'Navigation start, no actions taken yet.\nCurrent viewpoint "{obs[0]["viewpoint"]}": Scene from the viewpoint is a {obs[0]["obs_summary"]}'
        ]

    def _create_make_action_tool(
        self,
        llm: BaseLanguageModel,
    ) -> Tool:
        """Create a tool to make single action prediction in MP3D.

        The tool is invoked with the simulation environment and records the
        action taken by the agent.
        The tool interacts with the environment to obtain the current observation,
        uses the LLM to predict the next action, and to summarize the previous trajectory
        into history.
        """

        action_prompt = PromptTemplate(
            template=ACTION_PROMPT,
            input_variables=[
                "action_plan",
                "observation",
                "history",
                "navigable_viewpoints",
            ],
        )
        history_prompt = PromptTemplate(
            template=HISTORY_PROMPT,
            input_variables=["history", "previous_action", "observation"],
        )
        self.action_chain = LLMChain(llm=llm, prompt=action_prompt)
        self.history_chain = LLMChain(llm=llm, prompt=history_prompt)

        def _make_action(*args, **kwargs) -> str:
            """Make single step action in MatterSim."""
            # Get current observation
            cur_obs = self.env._get_obs()[0]

            # Get current feature
            feature = cur_obs["obs"]
            heading = np.rad2deg(cur_obs["heading"])
            elevation = np.rad2deg(cur_obs["elevation"])
            objects = cur_obs["objects"]
            orientation = f"\nheading: {heading:.2f}, elevation: {elevation:.2f}"
            navigable = cur_obs["candidate"]
            if self.config.use_relative_angle:
                feature = self.modify_heading_angles(
                    heading, feature, navigable, objects
                )
            if self.config.use_navigable:
                navigable = self.get_navigable_str(heading, elevation, navigable)

            if self.config.use_tool_chain:
                # Get current action plan
                action_plan = self.cur_action_plan
                # Single step action
                LLM_action_output = self.action_chain.run(
                    action_plan=action_plan,
                    observation=feature,
                    history=self.agent_executor.agent.history[-1],
                    navigable_viewpoints=navigable,
                )
                # Parse LLM output, action is the next viewpoint ID
                thought, action = self.parse_action(LLM_action_output)
            else:
                action = args[0].strip(" ").strip('"').strip("'")

            # Make the action in Simulator
            if action not in self.env.env.sims[0].navigable_dict.keys():
                # Update history
                history = f'ViewpointID "{action}" is not valid, no action taken for the agent.'
                self.agent_executor.agent.history.append(history)
                if self.config.use_navigable:
                    return f"\nViewpointID '{action}' is not valid, agent not moved. DO NOT fabricate nonexistent IDs. The navigable viewpoints you can choose from current viewpoints are: {[key for key in navigable.keys()]}.\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                else:
                    return f"\nViewpointID '{action}' is not valid, agent not moved. DO NOT fabricate nonexistent IDs. The navigable viewpoints you can choose from current viewpoints are: {[key for key in navigable.keys()]}.\n\tCurrent Viewpoint:\n{feature}"
            else:
                turned_angle, new_obs = self.make_equiv_action([action])

            # Update the current feature
            new_feature = new_obs["obs"]
            new_feature_sum = new_obs["obs_summary"]
            new_navigable = new_obs["candidate"]
            new_objects = new_obs["objects"]
            new_heading = np.rad2deg(new_obs["heading"])
            new_elevation = np.rad2deg(new_obs["elevation"])
            if self.config.use_relative_angle:
                new_feature = self.modify_heading_angles(
                    new_heading, new_feature, new_navigable, new_objects
                )
            new_orientation = (
                f"\nheading: {new_heading:.2f}, elevation: {new_elevation:.2f}"
            )
            if self.config.use_navigable:
                new_navigable = self.get_navigable_str(
                    new_heading, new_elevation, new_navigable
                )

            # Update history
            if self.config.use_history_chain:
                history = self.history_chain.run(
                    observation=new_feature_sum,
                    history=self.agent_executor.agent.history[-1],
                    previous_action=turned_angle,
                )
            else:
                history = self.get_history(new_obs, turned_angle)

            self.agent_executor.agent.history.append(history)
            # Record single step detail
            if self.config.use_tool_chain:
                detail = {
                    "viewpointID": action,
                    "turned_angle": turned_angle,
                    "acion_maker_thought": thought,
                    "feature": new_feature,
                    "history": self.agent_executor.agent.history[-1],
                }
            else:
                detail = {
                    "viewpointID": action,
                    "turned_angle": turned_angle,
                    "feature": new_feature,
                    "history": self.agent_executor.agent.history[-1],
                }
            self.traj[0]["details"].append(detail)
            # Return LLM chain output as the observation of tool
            if self.config.use_tool_chain:
                return f"\n\tAction_maker Thought:\n{thought}\n\tAction_maker Action:\n{turned_angle}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
            elif self.config.use_relative_angle:
                if self.config.use_navigable:
                    return f"\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f'\nCurrent Viewpoint "{action}":\n{new_feature}'
            else:
                if self.config.use_navigable:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}"

        return Tool(
            name=MAKE_ACTION_TOOL_NAME,
            func=_make_action,
            description=MAKE_ACTION_TOOL_DESCRIPTION,
        )

    def _create_back_trace_tool(
        self,
        llm: BaseLanguageModel,
    ) -> Tool:
        """Create a tool to back trace during navigation.

        The tool is invoked with the history of navigation trajectory.
        Using the LLM to find a viewpoint on the trajectory to back trace to.
        """
        prompt = PromptTemplate(
            template=BACK_TRACE_PROMPT,
            input_variables=["action_plan", "history", "observation"],
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        def _back_trace(*args, **kwargs) -> str:
            """Back trace the action plan."""
            cur_obs = self.env._get_obs()[0]

            # Get current feature
            feature = cur_obs["obs"]
            navigable = cur_obs["candidate"]
            objects = cur_obs["objects"]
            heading = np.rad2deg(cur_obs["heading"])
            elevation = np.rad2deg(cur_obs["elevation"])
            orientation = f"\nheading: {heading:.2f}, elevation: {elevation:.2f}"
            if self.config.use_relative_angle:
                feature = self.modify_heading_angles(
                    heading, feature, navigable, objects
                )
            if self.config.use_navigable:
                navigable = self.get_navigable_str(heading, elevation, navigable)

            if self.config.use_tool_chain:
                # Get current action plan
                action_plan = self.cur_action_plan
                # Get all previous viewpoints observation
                previous_vp = self.get_his_viewpoints()
                # Back trace
                LLM_output = chain.run(
                    action_plan=action_plan,
                    observation=previous_vp,
                    history=self.agent_executor.agent.history[-1],
                )
                # Parse LLM output, action is the next viewpoint ID
                thought, action = self.parse_action(LLM_output)
            else:
                action = args[0].strip(" ").strip('"').strip("'")

            # Make the action in Simulator
            if action not in self.env.env.sims[0].navigable_dict.keys():
                if self.config.use_navigable:
                    return f"\nViewpointID '{action}' is not valid. DO NOT fabricate nonexistent IDs.\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                else:
                    return f"\nViewpointID '{action}' is not valid. DO NOT fabricate nonexistent IDs.\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"
            else:
                _, new_obs = self.make_equiv_action([action])

            # Update the current feature
            new_feature = new_obs["obs"]
            new_navigable = new_obs["candidate"]
            new_objects = new_obs["objects"]
            new_heading = np.rad2deg(new_obs["heading"])
            new_elevation = np.rad2deg(new_obs["elevation"])
            new_orientation = (
                f"\nheading: {new_heading:.2f}, elevation: {new_elevation:.2f}"
            )
            if self.config.use_relative_angle:
                new_feature = self.modify_heading_angles(
                    new_heading, new_feature, new_navigable, new_objects
                )
            if self.config.use_navigable:
                new_navigable = self.get_navigable_str(
                    new_heading, new_elevation, new_navigable
                )

            # Update history
            history = self.get_history(
                new_obs, "Seems going in a wrong way, back trace to a previous point."
            )
            self.agent_executor.agent.history.append(history)
            # Record single step detail
            if self.config.use_tool_chain:
                return f"\tBack_tracer Thought:\n{thought}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
            elif self.config.use_relative_angle:
                if self.config.use_navigable:
                    return f"\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f"\nCurrent Viewpoint:{action}\n{new_feature}"
            else:
                if self.config.use_navigable:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}"

        return Tool(
            name=BACK_TRACE_TOOL_NAME,
            func=_back_trace,
            description=BACK_TRACE_TOOL_DESCRIPTION,
        )

    def create_vln_agent(
        self,
    ) -> AgentExecutor:
        """Instantiate API planner and controller for a given trajectory.

        We use a top-level "orchestrator" agent to invoke the planner and controller,
        rather than a top-level planner
        that invokes a controller with its plan. This is to keep the planner simple.
        """

        self.action_maker = self._create_make_action_tool(self.llm)
        self.back_tracer = self._create_back_trace_tool(self.llm)

        tools = [self.action_maker, self.back_tracer]

        if self.config.use_tool_chain:
            prompt = PromptTemplate(
                template=VLN_ORCHESTRATOR_PROMPT,
                input_variables=[
                    "action_plan",
                    "init_observation",
                    "observation",
                    "agent_scratchpad",
                ],
                partial_variables={
                    "tool_names": ", ".join([tool.name for tool in tools]),
                    "tool_descriptions": "\n".join(
                        [f"{tool.name}: {tool.description}" for tool in tools]
                    ),
                },
            )
        elif self.config.use_single_action:
            tools = [self.action_maker]
            prompt = PromptTemplate(
                template=(
                    VLN_GPT4_PROMPT
                    if self.config.llm_model_name == "gpt-4"
                    else VLN_GPT35_PROMPT
                ),
                input_variables=["action_plan", "init_observation", "agent_scratchpad"],
                partial_variables={
                    "tool_names": ", ".join([tool.name for tool in tools]),
                    "tool_descriptions": "\n".join(
                        [f"{tool.name}: {tool.description}" for tool in tools]
                    ),
                },
            )
        else:
            prompt = PromptTemplate(
                template=VLN_ORCHESTRATOR_PROMPT,
                input_variables=["action_plan", "init_observation", "agent_scratchpad"],
                partial_variables={
                    "tool_names": ", ".join([tool.name for tool in tools]),
                    "tool_descriptions": "\n".join(
                        [f"{tool.name}: {tool.description}" for tool in tools]
                    ),
                },
            )
        agent = VLNAgent(
            llm_chain=LLMChain(llm=self.llm, prompt=prompt),
            allowed_tools=[tool.name for tool in tools],
            output_parser=self.output_parser,
        )
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            # verbose=True,
            verbose=False,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            max_iterations=self.config.max_iterations,
        )

    def make_equiv_action(self, actions: List[str]) -> str:
        """
        Interface between Panoramic view and Egocentric view
        Take in the next viewpoint ID and move the agent to that viewpoint
        return the turned angle and new observation
        """

        def normalize_angle(angle):
            while angle > 180:
                angle -= 360
            while angle <= -180:
                angle += 360
            return angle

        def angle_to_left_right(angle):
            return f"left {-angle:.2f}" if angle < 0 else f"right {angle:.2f}"

        # Get current agent facing angle
        cur_obs = self.env._get_obs()[0]
        cur_heading = np.rad2deg(cur_obs["heading"])
        # Make the action
        new_obs = self.env.step(actions)[0]
        new_heading = np.rad2deg(new_obs["heading"])
        # Record the trajectory
        self.traj[0]["path"].append(
            self.env.env.sims[0].gmap.bfs_shortest_path(
                cur_obs["viewpoint"], actions[0]
            )[1:]
        )
        # Calculate the turned angle
        turned_angle = new_heading - cur_heading
        # Generate action description
        cur_heading = angle_to_left_right(normalize_angle(cur_heading))
        new_heading = angle_to_left_right(normalize_angle(new_heading))
        action_description = f"Turn heading direction {turned_angle:.2f} degrees from {cur_heading} to {new_heading}."
        return action_description, new_obs

    def rollout(self, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        # Initialize the trajectory
        self.init_trajecotry(obs)

        # Load the instruction
        instructions = [ob["instruction"] for ob in obs]
        if self.config.load_instruction:
            action_plans = instructions
        elif self.config.load_action_plan:
            action_plans = [ob["action_plan"] for ob in obs]
        else:
            action_plans = []
            for instruction in instructions:
                action_plan = self.plan_chain.run(instruction=instruction)
                action_plans.append(action_plan)

        for i, init_ob in enumerate(obs):
            print("--------------------------------")
            instr_id = init_ob["instr_id"]
            print(f"instr_id: {instr_id}")
            self.cur_action_plan = action_plans[i]
            # Take the first action
            if self.config.use_tool_chain:
                first_obs = self.action_maker("")
                input = {
                    "action_plan": self.cur_action_plan,
                    "init_observation": init_ob["obs_summary"],
                    "observation": first_obs,
                }
            else:
                # Get current feature
                feature = init_ob["obs"]
                navigable = init_ob["candidate"]
                objects = init_ob["objects"]
                heading = np.rad2deg(init_ob["heading"])
                elevation = np.rad2deg(init_ob["elevation"])
                orientation = f"\nheading: {heading:.2f}, elevation: {elevation:.2f}"
                if self.config.use_relative_angle:
                    feature = self.modify_heading_angles(
                        heading, feature, navigable, objects
                    )
                if self.config.use_navigable:
                    navigable = self.get_navigable_str(heading, elevation, navigable)

                if self.config.use_relative_angle:
                    if self.config.use_navigable:
                        init_observation = f"\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                    else:
                        init_observation = f"\n\tCurrent Viewpoint:\n{feature}"
                else:
                    if self.config.use_navigable:
                        init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                    else:
                        init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"

                input = {
                    "action_plan": self.cur_action_plan,
                    "init_observation": init_observation,
                }
            print("init input: ", input)
            output = self.agent_executor(input)

            self.traj[i]["llm_output"] = output["output"]
            self.traj[i]["action_plan"] = output["action_plan"]
            # extract agent's thought from llm output
            intermediate_steps = output["intermediate_steps"]
            self.traj[i]["llm_thought"] = []
            self.traj[i]["llm_observation"] = []
            for action, observation in intermediate_steps:
                thought = action.log
                self.traj[i]["llm_thought"].append(thought)
                self.traj[i]["llm_observation"].append(observation)

        return self.traj

    def rollout2(self, reset=True, max_steps=10, max_retries=3, retry_delay=1.0):
        """
        Step-by-step rollout using max_iterations=1 with agent_executor.
        This allows for step-by-step execution similar to training loops.

        Differences from rollout():
        - rollout(): Calls agent_executor once, which executes the entire navigation
        - rollout2(): Calls agent_executor with max_iterations=1 repeatedly, one step at a time

        This is useful for:
        - Verifying step-by-step behavior
        - Training loops that need to control each step
        - Debugging and comparison with rollout()

        Args:
            reset: Whether to reset the environment
            max_steps: Maximum number of steps to execute (safety limit)
            max_retries: Maximum number of retry attempts for agent_executor calls (default: 3)
            retry_delay: Delay in seconds between retry attempts (default: 1.0)

        Returns:
            self.traj: Trajectory dictionary
        """
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        # Initialize the trajectory
        self.init_trajecotry(obs)

        # Load the instruction
        instructions = [ob["instruction"] for ob in obs]
        if self.config.load_instruction:
            action_plans = instructions
        elif self.config.load_action_plan:
            action_plans = [ob["action_plan"] for ob in obs]
        else:
            action_plans = []
            for instruction in instructions:
                action_plan = self.plan_chain.run(instruction=instruction)
                action_plans.append(action_plan)

        for i, init_ob in enumerate(obs):
            print("--------------------------------")
            instr_id = init_ob["instr_id"]
            print(f"instr_id: {instr_id}")
            self.cur_action_plan = action_plans[i]

            # Save original max_iterations
            original_max_iter = self.agent_executor.max_iterations

            # Set max_iterations to 1 for step-by-step execution
            self.agent_executor.max_iterations = 1

            # Prepare initial input based on use_tool_chain mode
            if self.config.use_tool_chain:
                # For tool_chain mode, prepare input similar to rollout()
                # But we don't call action_maker('') first - we let agent_executor handle it
                init_observation = init_ob["obs_summary"]
                input_dict = {
                    "action_plan": self.cur_action_plan,
                    "init_observation": init_observation,
                    "observation": "",  # Empty for first step
                }
            else:
                # For non-tool_chain mode, prepare full observation string
                feature = init_ob["obs"]
                navigable = init_ob["candidate"]
                objects = init_ob["objects"]
                heading = np.rad2deg(init_ob["heading"])
                elevation = np.rad2deg(init_ob["elevation"])
                orientation = f"\nheading: {heading:.2f}, elevation: {elevation:.2f}"

                if self.config.use_relative_angle:
                    feature = self.modify_heading_angles(
                        heading, feature, navigable, objects
                    )
                if self.config.use_navigable:
                    navigable = self.get_navigable_str(heading, elevation, navigable)

                if self.config.use_relative_angle:
                    if self.config.use_navigable:
                        init_observation = f"\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                    else:
                        init_observation = f"\n\tCurrent Viewpoint:\n{feature}"
                else:
                    if self.config.use_navigable:
                        init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                    else:
                        init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"

                input_dict = {
                    "action_plan": self.cur_action_plan,
                    "init_observation": init_observation,
                }

            # Step-by-step execution loop
            step_count = 0
            all_thoughts = []
            all_observations = []
            final_output = None
            accumulated_intermediate_steps = []  # Manually maintain intermediate steps

            while step_count < max_steps:
                try:

                    # Manually inject accumulated intermediate_steps into agent_executor
                    # This ensures the agent has access to previous steps
                    # We need to set it before calling, and LangChain's AgentExecutor
                    # should use these steps when calling agent.get_full_inputs()
                    if hasattr(self.agent_executor, "intermediate_steps"):
                        self.agent_executor.intermediate_steps = (
                            accumulated_intermediate_steps.copy()
                        )

                    # Also pass manual_intermediate_steps in input_dict as a fallback
                    # This allows VLNAgent.get_full_inputs to use it if AgentExecutor resets intermediate_steps
                    input_dict_with_history = input_dict.copy()
                    if len(accumulated_intermediate_steps) > 0:
                        input_dict_with_history["manual_intermediate_steps"] = (
                            accumulated_intermediate_steps.copy()
                        )

                    # Call agent_executor with max_iterations=1 with retry logic
                    # This will execute one step: orchestrator decides -> tool executes -> returns
                    output = None
                    last_exception = None
                    for attempt in range(max_retries):
                        try:
                            output = self.agent_executor(input_dict_with_history)
                            break  # Success - exit retry loop
                        except (KeyError, ValueError) as e:
                            # Handle API response format errors
                            # KeyError often indicates malformed API response (missing expected fields like 'content')
                            # ValueError can indicate invalid response format
                            last_exception = e
                            error_msg = str(e)
                            error_type = type(e).__name__

                            # Check if it's likely an API response error
                            is_api_response_error = (
                                "'content'" in error_msg
                                or "content" in error_msg.lower()
                                or "choices" in error_msg.lower()
                                or "message" in error_msg.lower()
                                or isinstance(
                                    e, KeyError
                                )  # KeyError from API response parsing
                            )

                            if is_api_response_error and attempt < max_retries - 1:
                                print(
                                    f"[Retry {attempt+1}/{max_retries}] API response error ({error_type}) at step {step_count}: {error_msg}"
                                )
                                print(
                                    "This may indicate a malformed API response. Retrying..."
                                )
                                time.sleep(retry_delay)
                                # Continue loop to retry
                            elif is_api_response_error:
                                # Last attempt failed
                                print(
                                    f"[Failed after {max_retries} attempts] API response error ({error_type}) at step {step_count}: {error_msg}"
                                )
                                # Break out of retry loop, output will be None
                                break
                            else:
                                # Other KeyError/ValueError not related to API - re-raise immediately
                                raise
                        except OutputParserException as e:
                            # Parsing errors: handle_parsing_errors=True should handle these,
                            # but if they bubble up, retry by calling agent_executor again
                            # The LLM will receive the error message and can correct itself
                            last_exception = e
                            error_msg = str(e)
                            if attempt < max_retries - 1:
                                print(
                                    f"[Retry {attempt+1}/{max_retries}] Parsing error at step {step_count}: {error_msg}"
                                )
                                print("Retrying with error message sent to LLM...")
                                time.sleep(retry_delay)
                                # Continue loop to retry
                            else:
                                # Last attempt failed - break instead of raise to handle gracefully
                                print(
                                    f"[Failed after {max_retries} attempts] Parsing error at step {step_count}: {error_msg}"
                                )
                                break
                        except Exception as e:
                            last_exception = e
                            error_msg = str(e)
                            # Check if it's a network/API error that should be retried
                            is_retryable = (
                                "timeout" in error_msg.lower()
                                or "connection" in error_msg.lower()
                                or "network" in error_msg.lower()
                                or "rate limit" in error_msg.lower()
                                or "api" in error_msg.lower()
                                or isinstance(e, (ConnectionError, TimeoutError))
                            )

                            if is_retryable and attempt < max_retries - 1:
                                print(
                                    f"[Retry {attempt+1}/{max_retries}] Error calling agent_executor at step {step_count}: {error_msg}"
                                )
                                time.sleep(retry_delay)
                            else:
                                # Not retryable or last attempt - break to handle gracefully
                                print(
                                    f"[Failed after {max_retries} attempts] Error calling agent_executor at step {step_count}: {error_msg}"
                                )
                                break

                    # Handle case where output is None (all retries failed)
                    if output is None:
                        print(
                            f"Warning: Failed to get output from agent_executor after {max_retries} attempts at step {step_count}"
                        )
                        if last_exception:
                            print(f"Last exception: {last_exception}")
                        # Set output to empty dict to prevent AttributeError later
                        output = {"output": "", "intermediate_steps": []}
                        # Break out of the step loop
                        break

                    # print("######## output: ")
                    # print(output)
                    # print("######## end of output")
                    # print("output: ", output)

                    # Extract information from this step
                    # Note: output['intermediate_steps'] contains ALL steps from this call,
                    # but since max_iterations=1, it should only contain the new step
                    intermediate_steps = output.get("intermediate_steps", [])

                    # Update accumulated intermediate_steps with new steps from this iteration
                    # The intermediate_steps from output should contain all steps including previous ones
                    # if AgentExecutor properly maintains state, but we'll merge them to be safe
                    if len(intermediate_steps) > len(accumulated_intermediate_steps):
                        # New steps were added, update accumulated list
                        accumulated_intermediate_steps = intermediate_steps.copy()
                    elif len(intermediate_steps) > 0:
                        # Check if there are any new steps not in accumulated list
                        for step in intermediate_steps:
                            if step not in accumulated_intermediate_steps:
                                accumulated_intermediate_steps.append(step)

                    # Also update agent_executor's internal state after the call
                    if hasattr(self.agent_executor, "intermediate_steps"):
                        self.agent_executor.intermediate_steps = (
                            accumulated_intermediate_steps.copy()
                        )

                    # Check if orchestrator decided to stop (Final Answer)
                    output_text = output.get("output", "")
                    if "Finished!" in output_text or "Final Answer" in output_text:
                        # Navigation completed
                        final_output = output_text
                        # Record the final step
                        if len(intermediate_steps) > 0:
                            for action, observation in intermediate_steps:
                                all_thoughts.append(action.log)
                                all_observations.append(observation)
                        break

                    # Process intermediate steps
                    if len(intermediate_steps) > 0:
                        for action, observation in intermediate_steps:
                            all_thoughts.append(action.log)
                            all_observations.append(observation)

                        # Get the last action to determine next step
                        last_action, last_observation = intermediate_steps[-1]

                        # Prepare input for next step
                        # For tool_chain mode, use the observation from action_maker
                        if self.config.use_tool_chain:
                            # The observation from action_maker contains the new state
                            input_dict = {
                                "action_plan": self.cur_action_plan,
                                "init_observation": init_observation,
                                "observation": last_observation,  # Use observation from action_maker
                            }
                        else:
                            # For non-tool_chain, agent_executor manages state through history
                            # Get current observation from environment
                            cur_obs = self.env._get_obs()[0]
                            feature = cur_obs["obs"]
                            navigable = cur_obs["candidate"]
                            objects = cur_obs["objects"]
                            heading = np.rad2deg(cur_obs["heading"])
                            elevation = np.rad2deg(cur_obs["elevation"])
                            orientation = (
                                f"\nheading: {heading:.2f}, elevation: {elevation:.2f}"
                            )

                            if self.config.use_relative_angle:
                                feature = self.modify_heading_angles(
                                    heading, feature, navigable, objects
                                )
                            if self.config.use_navigable:
                                navigable = self.get_navigable_str(
                                    heading, elevation, navigable
                                )

                            if self.config.use_relative_angle:
                                if self.config.use_navigable:
                                    init_observation = f"\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                                else:
                                    init_observation = (
                                        f"\n\tCurrent Viewpoint:\n{feature}"
                                    )
                            else:
                                if self.config.use_navigable:
                                    init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                                else:
                                    init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"

                            input_dict = {
                                "action_plan": self.cur_action_plan,
                                "init_observation": init_observation,
                            }
                    else:
                        # No intermediate steps - should not happen, but break to avoid infinite loop
                        break

                    step_count += 1

                except Exception as e:
                    print(f"Error in rollout2 at step {step_count}: {e}")
                    import traceback

                    traceback.print_exc()
                    break

            # Restore original max_iterations
            self.agent_executor.max_iterations = original_max_iter

            # Store results in trajectory
            self.traj[i]["llm_output"] = (
                final_output if final_output else output.get("output", "")
            )
            self.traj[i]["action_plan"] = self.cur_action_plan
            self.traj[i]["llm_thought"] = all_thoughts
            self.traj[i]["llm_observation"] = all_observations

            if step_count >= max_steps:
                print(
                    f"Warning: rollout2 reached max_steps ({max_steps}) for trajectory {i}"
                )

        return self.traj

    def rollout2_with_file(
        self,
        reset=True,
        max_steps=10,
        max_retries=3,
        retry_delay=1.0,
        output_file=None,
    ):
        """
        Step-by-step rollout using max_iterations=1 with agent_executor, with data collection.
        Based on rollout2, but collects navigation information and saves to JSON file.

        This function executes NavGPT rollout step-by-step and collects:
        1. LLM input components (separated by type for easy perturbation):
           - action_plan: the action plan string (stored once at t=0)
           - init_observation: initial observation string (for non-tool_chain mode)
           - observation: previous step observation (for tool_chain mode)
           - intermediate_steps: list of (action, observation) tuples for context maintenance
        2. Action output:
           - viewpoint_id: the viewpoint ID string executed (or None for stop)

        Args:
            reset: Whether to reset the environment
            max_steps: Maximum number of steps to execute (safety limit)
            max_retries: Maximum number of retry attempts for agent_executor calls (default: 3)
            retry_delay: Delay in seconds between retry attempts (default: 1.0)
            output_file: Path to output JSON file (default: NavGPT/nav_24vp/all_nav_outputs.json)

        Returns:
            self.traj: Trajectory dictionary
        """
        import time

        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        # Initialize the trajectory
        self.init_trajecotry(obs)

        # Load the instruction
        instructions = [ob["instruction"] for ob in obs]
        if self.config.load_instruction:
            action_plans = instructions
        elif self.config.load_action_plan:
            action_plans = [ob["action_plan"] for ob in obs]
        else:
            action_plans = []
            for instruction in instructions:
                action_plan = self.plan_chain.run(instruction=instruction)
                action_plans.append(action_plan)

        # Dictionary to store all navigation info: {instr_id: {t: nav_info}}
        all_nav_info = {}

        # Save original max_iterations
        original_max_iter = self.agent_executor.max_iterations

        # Set max_iterations to 1 for step-by-step execution
        self.agent_executor.max_iterations = 1

        for i, init_ob in enumerate(obs):
            print("--------------------------------")
            instr_id = init_ob["instr_id"]
            print(f"instr_id: {instr_id}")
            self.cur_action_plan = action_plans[i]

            # Initialize dict for this instruction
            if instr_id not in all_nav_info:
                all_nav_info[instr_id] = {}

            # Store action plan (stored once at t=0)
            stored_action_plan = action_plans[i]

            # Store initial observation for tool_chain mode (remains constant for this trajectory)
            stored_init_observation = None
            if self.config.use_tool_chain:
                stored_init_observation = init_ob.get("obs_summary", "")

            # Prepare initial input based on use_tool_chain mode
            if self.config.use_tool_chain:
                # For tool_chain mode, prepare input similar to rollout()
                init_observation = init_ob["obs_summary"]
                input_dict = {
                    "action_plan": self.cur_action_plan,
                    "init_observation": init_observation,
                    "observation": "",  # Empty for first step
                }
            else:
                # For non-tool_chain mode, prepare full observation string
                feature = init_ob["obs"]
                navigable = init_ob["candidate"]
                objects = init_ob["objects"]
                heading = np.rad2deg(init_ob["heading"])
                elevation = np.rad2deg(init_ob["elevation"])
                orientation = f"\nheading: {heading:.2f}, elevation: {elevation:.2f}"

                if self.config.use_relative_angle:
                    feature = self.modify_heading_angles(
                        heading, feature, navigable, objects
                    )
                if self.config.use_navigable:
                    navigable = self.get_navigable_str(heading, elevation, navigable)

                if self.config.use_relative_angle:
                    if self.config.use_navigable:
                        init_observation = f"\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                    else:
                        init_observation = f"\n\tCurrent Viewpoint:\n{feature}"
                else:
                    if self.config.use_navigable:
                        init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                    else:
                        init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"

                input_dict = {
                    "action_plan": self.cur_action_plan,
                    "init_observation": init_observation,
                }
                stored_init_observation = init_observation

            # Step-by-step execution loop
            step_count = 0
            all_thoughts = []
            all_observations = []
            final_output = None
            accumulated_intermediate_steps = []  # Manually maintain intermediate steps

            while step_count < max_steps:
                try:
                    # Manually inject accumulated_intermediate_steps into agent_executor
                    if hasattr(self.agent_executor, "intermediate_steps"):
                        self.agent_executor.intermediate_steps = (
                            accumulated_intermediate_steps.copy()
                        )

                    # Also pass manual_intermediate_steps in input_dict as a fallback
                    input_dict_with_history = input_dict.copy()
                    if len(accumulated_intermediate_steps) > 0:
                        input_dict_with_history["manual_intermediate_steps"] = (
                            accumulated_intermediate_steps.copy()
                        )

                    # Call agent_executor with max_iterations=1 with retry logic
                    output = None
                    last_exception = None
                    for attempt in range(max_retries):
                        try:
                            output = self.agent_executor(input_dict_with_history)
                            break  # Success - exit retry loop
                        except (KeyError, ValueError) as e:
                            last_exception = e
                            error_msg = str(e)
                            error_type = type(e).__name__

                            is_api_response_error = (
                                "'content'" in error_msg
                                or "content" in error_msg.lower()
                                or "choices" in error_msg.lower()
                                or "message" in error_msg.lower()
                                or isinstance(e, KeyError)
                            )

                            if is_api_response_error and attempt < max_retries - 1:
                                print(
                                    f"[Retry {attempt+1}/{max_retries}] API response error ({error_type}) at step {step_count}: {error_msg}"
                                )
                                time.sleep(retry_delay)
                            elif is_api_response_error:
                                print(
                                    f"[Failed after {max_retries} attempts] API response error ({error_type}) at step {step_count}: {error_msg}"
                                )
                                break
                            else:
                                raise
                        except OutputParserException as e:
                            last_exception = e
                            error_msg = str(e)
                            if attempt < max_retries - 1:
                                print(
                                    f"[Retry {attempt+1}/{max_retries}] Parsing error at step {step_count}: {error_msg}"
                                )
                                time.sleep(retry_delay)
                            else:
                                print(
                                    f"[Failed after {max_retries} attempts] Parsing error at step {step_count}: {error_msg}"
                                )
                                break
                        except Exception as e:
                            last_exception = e
                            error_msg = str(e)
                            is_retryable = (
                                "timeout" in error_msg.lower()
                                or "connection" in error_msg.lower()
                                or "network" in error_msg.lower()
                                or "rate limit" in error_msg.lower()
                                or "api" in error_msg.lower()
                                or isinstance(e, (ConnectionError, TimeoutError))
                            )

                            if is_retryable and attempt < max_retries - 1:
                                print(
                                    f"[Retry {attempt+1}/{max_retries}] Error calling agent_executor at step {step_count}: {error_msg}"
                                )
                                time.sleep(retry_delay)
                            else:
                                print(
                                    f"[Failed after {max_retries} attempts] Error calling agent_executor at step {step_count}: {error_msg}"
                                )
                                break

                    # Handle case where output is None (all retries failed)
                    if output is None:
                        print(
                            f"Warning: Failed to get output from agent_executor after {max_retries} attempts at step {step_count}"
                        )
                        if last_exception:
                            print(f"Last exception: {last_exception}")
                        output = {"output": "", "intermediate_steps": []}
                        break

                    # Extract information from this step
                    intermediate_steps = output.get("intermediate_steps", [])

                    # Update accumulated intermediate_steps with new steps from this iteration
                    if len(intermediate_steps) > len(accumulated_intermediate_steps):
                        accumulated_intermediate_steps = intermediate_steps.copy()
                    elif len(intermediate_steps) > 0:
                        for step in intermediate_steps:
                            if step not in accumulated_intermediate_steps:
                                accumulated_intermediate_steps.append(step)

                    # Also update agent_executor's internal state after the call
                    if hasattr(self.agent_executor, "intermediate_steps"):
                        self.agent_executor.intermediate_steps = (
                            accumulated_intermediate_steps.copy()
                        )

                    # Serialize intermediate_steps for JSON storage
                    serialized_intermediate_steps = []
                    for action, obs_str in accumulated_intermediate_steps:
                        if AgentAction is not None and isinstance(action, AgentAction):
                            serialized_intermediate_steps.append(
                                {
                                    "tool": action.tool,
                                    "tool_input": action.tool_input,
                                    "log": action.log,
                                    "observation": obs_str,
                                }
                            )
                        else:
                            try:
                                serialized_intermediate_steps.append(
                                    {
                                        "tool": getattr(action, "tool", ""),
                                        "tool_input": getattr(action, "tool_input", ""),
                                        "log": getattr(action, "log", ""),
                                        "observation": obs_str,
                                    }
                                )
                            except:
                                serialized_intermediate_steps.append(
                                    {
                                        "action": str(action),
                                        "observation": str(obs_str),
                                    }
                                )

                    # Extract viewpoint_id from output
                    viewpoint_id = None
                    output_text = output.get("output", "")

                    if "Finished!" in output_text or "Final Answer" in output_text:
                        viewpoint_id = None  # Stop action
                        final_output = output_text
                    elif len(intermediate_steps) > 0:
                        last_action, last_observation = intermediate_steps[-1]
                        if last_action.tool == "action_maker":
                            viewpoint_id = (
                                last_action.tool_input.strip('"').strip("'").strip()
                            )
                        elif last_action.tool == "back_tracer":
                            viewpoint_id = (
                                last_action.tool_input.strip('"').strip("'").strip()
                            )

                    # Convert action index for backward compatibility
                    candidate_dict = init_ob.get("candidate", {})
                    candidate_list = list(candidate_dict.keys())
                    a_t_index = 0
                    if viewpoint_id and viewpoint_id in candidate_dict:
                        a_t_index = candidate_list.index(viewpoint_id) + 1
                    elif viewpoint_id is None:
                        a_t_index = 0  # Stop action

                    # Get the actual observation used in this step
                    if self.config.use_tool_chain:
                        observation_to_save = input_dict.get("observation", "")
                    else:
                        observation_to_save = ""

                    # Save complete history list for full LLM input reconstruction
                    history_list = []
                    if (
                        hasattr(self, "agent_executor")
                        and hasattr(self.agent_executor, "agent")
                        and hasattr(self.agent_executor.agent, "history")
                    ):
                        history_list = list(self.agent_executor.agent.history)

                    # Collect nav_info for this step
                    nav_info = {
                        "nav_input": {
                            "action_plan": stored_action_plan,
                            "init_observation": input_dict.get("init_observation", ""),
                            "observation": observation_to_save,
                            "intermediate_steps": serialized_intermediate_steps,
                            "history": history_list,
                        },
                        "viewpoint_id": viewpoint_id,
                        "a_t_list": [int(a_t_index)],
                    }

                    all_nav_info[instr_id][str(step_count)] = nav_info

                    # Check if orchestrator decided to stop (Final Answer)
                    if "Finished!" in output_text or "Final Answer" in output_text:
                        # Navigation completed
                        final_output = output_text
                        # Record the final step
                        if len(intermediate_steps) > 0:
                            for action, observation in intermediate_steps:
                                all_thoughts.append(action.log)
                                all_observations.append(observation)
                        break

                    # Process intermediate steps
                    if len(intermediate_steps) > 0:
                        for action, observation in intermediate_steps:
                            all_thoughts.append(action.log)
                            all_observations.append(observation)

                        # Get the last action to determine next step
                        last_action, last_observation = intermediate_steps[-1]

                        # Prepare input for next step
                        if self.config.use_tool_chain:
                            input_dict = {
                                "action_plan": self.cur_action_plan,
                                "init_observation": stored_init_observation,
                                "observation": last_observation,  # Use observation from action_maker
                            }
                        else:
                            # For non-tool_chain, agent_executor manages state through history
                            # Get current observation from environment
                            cur_obs = self.env._get_obs()[0]
                            feature = cur_obs["obs"]
                            navigable = cur_obs["candidate"]
                            objects = cur_obs["objects"]
                            heading = np.rad2deg(cur_obs["heading"])
                            elevation = np.rad2deg(cur_obs["elevation"])
                            orientation = (
                                f"\nheading: {heading:.2f}, elevation: {elevation:.2f}"
                            )

                            if self.config.use_relative_angle:
                                feature = self.modify_heading_angles(
                                    heading, feature, navigable, objects
                                )
                            if self.config.use_navigable:
                                navigable = self.get_navigable_str(
                                    heading, elevation, navigable
                                )

                            if self.config.use_relative_angle:
                                if self.config.use_navigable:
                                    init_observation = f"\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                                else:
                                    init_observation = (
                                        f"\n\tCurrent Viewpoint:\n{feature}"
                                    )
                            else:
                                if self.config.use_navigable:
                                    init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                                else:
                                    init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"

                            input_dict = {
                                "action_plan": self.cur_action_plan,
                                "init_observation": init_observation,
                            }
                    else:
                        # No intermediate steps - should not happen, but break to avoid infinite loop
                        break

                    step_count += 1

                except Exception as e:
                    print(f"Error in rollout2_with_file at step {step_count}: {e}")
                    import traceback

                    traceback.print_exc()
                    break

            # Restore original max_iterations
            self.agent_executor.max_iterations = original_max_iter

            # Store results in trajectory
            self.traj[i]["llm_output"] = (
                final_output if final_output else output.get("output", "")
            )
            self.traj[i]["action_plan"] = self.cur_action_plan
            self.traj[i]["llm_thought"] = all_thoughts
            self.traj[i]["llm_observation"] = all_observations

            if step_count >= max_steps:
                print(
                    f"Warning: rollout2_with_file reached max_steps ({max_steps}) for trajectory {i}"
                )

        # Restore original max_iterations (in case of early exit)
        self.agent_executor.max_iterations = original_max_iter

        # Save to file
        if output_file is None:
            output_file = os.path.join("NavGPT", "nav_24vp", "all_nav_outputs.json")

        # Create directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Load existing data if file exists
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                existing_data = json.load(f)
            # Merge with new data
            for instr_id, step_data in all_nav_info.items():
                if instr_id not in existing_data:
                    existing_data[instr_id] = {}
                existing_data[instr_id].update(step_data)
            all_nav_info = existing_data

        # Save to file
        with open(output_file, "w") as f:
            json.dump(all_nav_info, f, indent=2)

        print(f"Saved NavGPT nav info to {output_file}")

        return self.traj

    def collect_nav_info_rollout2(
        self,
        env=None,
        reset=True,
        output_file=None,
        max_steps=10,
        max_retries=3,
        retry_delay=1.0,
    ):
        """
        Collect NavGPT navigation information for each step, similar to NavGPT's rollout2.

        This function executes NavGPT rollout step-by-step and collects:
        1. LLM input components (separated by type for easy perturbation):
           - action_plan: the action plan string (stored once at t=0)
           - init_observation: initial observation string (for non-tool_chain mode)
           - observation: previous step observation (for tool_chain mode)
           - intermediate_steps: list of (action, observation) tuples for context maintenance
             (needed to fully reproduce LLM input, especially in tool_chain mode)
        2. Action output:
           - viewpoint_id: the viewpoint ID string executed (or None for stop)

        Args:
            env: Environment instance (optional, defaults to self.env)
            reset: Whether to reset the environment
            output_file: Path to output JSON file (default: NavGPT/nav_24vp/all_nav_outputs.json)
            max_steps: Maximum number of steps to execute (default: 20)
            max_retries: Maximum number of retry attempts for agent_executor calls (default: 3)
            retry_delay: Delay in seconds between retry attempts (default: 1.0)

        Returns:
            Dictionary mapping instr_id -> {t -> nav_info}
        """
        # Use self.env if env is not provided
        if env is None:
            env = self.env

        if reset:
            obs = env.reset()
            # Sync self.env with env
            self.env.set_scan_viewpoint_heading(env.get_scan_viewpoint_heading())
        else:
            obs = env._get_obs()
            # Sync self.env with env
            self.env.set_scan_viewpoint_heading(env.get_scan_viewpoint_heading())

        batch_size = len(obs)

        # Initialize the trajectory
        self.init_trajecotry(obs)

        # Load the instruction for NavGPT
        instructions = [ob["instruction"] for ob in obs]
        if self.config.load_instruction:
            action_plans = instructions
        elif self.config.load_action_plan:
            action_plans = [ob["action_plan"] for ob in obs]
        else:
            action_plans = []
            for instruction in instructions:
                action_plan = self.plan_chain.run(instruction=instruction)
                action_plans.append(action_plan)

        # Dictionary to store all navigation info: {instr_id: {t: nav_info}}
        all_nav_info = {}

        # Track trajectories
        target_traj = [
            {
                "instr_id": ob["instr_id"],
                "path": [[ob["viewpoint"]]],
            }
            for ob in obs
        ]

        # Initialization the tracking state
        target_ended = np.array([False] * batch_size)
        target_just_ended = np.array([False] * batch_size)

        previous_angle = [
            {"heading": ob["heading"], "elevation": ob["elevation"]} for ob in obs
        ]

        # Store action plan for each instruction (stored once at t=0)
        stored_action_plans = {}

        # Store initial observation for each instruction (for tool_chain mode)
        # Similar to rollout2, init_observation should remain constant for each trajectory
        stored_init_observations = {}

        # Store input_dict for each trajectory (will be updated after each step, like rollout2)
        input_dict_dict = {}

        # Initialize accumulated_intermediate_steps for NavGPT agent context
        # Use a dictionary to store intermediate_steps for each instr_id (for batch processing)
        accumulated_intermediate_steps_dict = {}

        # Initialize last_observation for tool_chain mode (per batch item)
        # Use a dictionary to store last_observation for each instr_id
        if self.config.use_tool_chain:
            if not hasattr(self, "_last_observation_dict"):
                self._last_observation_dict = {}

        # Track all_thoughts and all_observations for each trajectory (like rollout2)
        all_thoughts_dict = {}
        all_observations_dict = {}
        final_output_dict = {}
        last_output_dict = {}  # Track last output_text for each trajectory
        cur_action_plan_dict = {}

        # Save original max_iterations
        original_max_iter = self.agent_executor.max_iterations

        # Set max_iterations to 1 for step-by-step execution
        self.agent_executor.max_iterations = 1

        for t in range(max_steps):
            # Collect data for each observation in the batch
            for i, ob in enumerate(obs):
                if target_ended[i]:
                    continue

                instr_id = ob["instr_id"]

                # Initialize dict for this instruction if needed
                if instr_id not in all_nav_info:
                    all_nav_info[instr_id] = {}

                # Get action plan and initial observation (stored once at t=0)
                if t == 0:
                    stored_action_plans[instr_id] = action_plans[i]
                    # Store initial observation for tool_chain mode (remains constant for this trajectory)
                    if self.config.use_tool_chain:
                        stored_init_observations[instr_id] = ob.get("obs_summary", "")
                        # Initialize last_observation for this trajectory (empty string for first step, like rollout2)
                        self._last_observation_dict[instr_id] = ""
                    # Initialize accumulated_intermediate_steps for this trajectory
                    accumulated_intermediate_steps_dict[instr_id] = []
                    # Initialize all_thoughts and all_observations for this trajectory (like rollout2)
                    all_thoughts_dict[instr_id] = []
                    all_observations_dict[instr_id] = []
                    final_output_dict[instr_id] = None
                    cur_action_plan_dict[instr_id] = action_plans[i]

                # Prepare input_dict for agent_executor (similar to rollout2)
                # Use stored input_dict if available (updated from previous step), otherwise prepare initial one
                if instr_id in input_dict_dict:
                    # Use input_dict from previous step (updated after action execution, like rollout2)
                    input_dict = input_dict_dict[instr_id]
                else:
                    # First step: prepare initial input_dict (like rollout2)
                    if self.config.use_tool_chain:
                        # For tool_chain mode, prepare input similar to rollout2
                        init_observation = stored_init_observations.get(instr_id, "")
                        input_dict = {
                            "action_plan": stored_action_plans.get(instr_id, ""),
                            "init_observation": init_observation,
                            "observation": "",  # Empty for first step (like rollout2)
                        }
                    else:
                        # For non-tool_chain mode, prepare full observation string (like rollout2)
                        feature = ob["obs"]
                        navigable = ob["candidate"]
                        objects = ob["objects"]
                        heading = np.rad2deg(ob["heading"])
                        elevation = np.rad2deg(ob["elevation"])
                        orientation = (
                            f"\nheading: {heading:.2f}, elevation: {elevation:.2f}"
                        )

                        if self.config.use_relative_angle:
                            feature = self.modify_heading_angles(
                                heading, feature, navigable, objects
                            )
                        if self.config.use_navigable:
                            navigable = self.get_navigable_str(
                                heading, elevation, navigable
                            )

                        if self.config.use_relative_angle:
                            if self.config.use_navigable:
                                init_observation = f"\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                            else:
                                init_observation = f"\n\tCurrent Viewpoint:\n{feature}"
                        else:
                            if self.config.use_navigable:
                                init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                            else:
                                init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"

                        input_dict = {
                            "action_plan": stored_action_plans.get(instr_id, ""),
                            "init_observation": init_observation,
                        }

                # Also pass manual_intermediate_steps in input_dict as a fallback
                input_dict_with_history = input_dict.copy()
                accumulated_steps = accumulated_intermediate_steps_dict.get(
                    instr_id, []
                )
                if len(accumulated_steps) > 0:
                    input_dict_with_history["manual_intermediate_steps"] = (
                        accumulated_steps.copy()
                    )

                # Manually inject accumulated_intermediate_steps into agent_executor
                if hasattr(self.agent_executor, "intermediate_steps"):
                    self.agent_executor.intermediate_steps = accumulated_steps.copy()

                # Call agent_executor once (no voting)
                # Call agent_executor with retry logic
                output = None
                last_exception = None
                for attempt in range(max_retries):
                    try:
                        output = self.agent_executor(input_dict_with_history)
                        break  # Success - exit retry loop
                    except (KeyError, ValueError) as e:
                        # Handle API response format errors
                        # KeyError often indicates malformed API response (missing expected fields like 'content')
                        # ValueError can indicate invalid response format
                        last_exception = e
                        error_msg = str(e)
                        error_type = type(e).__name__

                        # Check if it's likely an API response error
                        is_api_response_error = (
                            "'content'" in error_msg
                            or "content" in error_msg.lower()
                            or "choices" in error_msg.lower()
                            or "message" in error_msg.lower()
                            or isinstance(
                                e, KeyError
                            )  # KeyError from API response parsing
                        )

                        if is_api_response_error and attempt < max_retries - 1:
                            print(
                                f"[Retry {attempt+1}/{max_retries}] API response error ({error_type}) at step {t}, obs {i}: {error_msg}"
                            )
                            print(
                                "This may indicate a malformed API response. Retrying..."
                            )
                            time.sleep(retry_delay)
                            # Continue loop to retry
                        elif is_api_response_error:
                            # Last attempt failed
                            print(
                                f"[Failed after {max_retries} attempts] API response error ({error_type}) at step {t}, obs {i}: {error_msg}"
                            )
                            # Break out of retry loop, output will be None
                            break
                        else:
                            # Other KeyError/ValueError not related to API - re-raise immediately
                            raise
                    except OutputParserException as e:
                        # Parsing errors: handle_parsing_errors=True should handle these,
                        # but if they bubble up, retry by calling agent_executor again
                        # The LLM will receive the error message and can correct itself
                        last_exception = e
                        error_msg = str(e)
                        if attempt < max_retries - 1:
                            print(
                                f"[Retry {attempt+1}/{max_retries}] Parsing error at step {t}, obs {i}: {error_msg}"
                            )
                            print("Retrying with error message sent to LLM...")
                            time.sleep(retry_delay)
                            # Continue loop to retry
                        else:
                            # Last attempt failed - break instead of raise to handle gracefully
                            print(
                                f"[Failed after {max_retries} attempts] Parsing error at step {t}, obs {i}: {error_msg}"
                            )
                            break
                    except Exception as e:
                        last_exception = e
                        error_msg = str(e)
                        # Check if it's a network/API error that should be retried
                        is_retryable = (
                            "timeout" in error_msg.lower()
                            or "connection" in error_msg.lower()
                            or "network" in error_msg.lower()
                            or "rate limit" in error_msg.lower()
                            or "api" in error_msg.lower()
                            or isinstance(e, (ConnectionError, TimeoutError))
                        )

                        if is_retryable and attempt < max_retries - 1:
                            print(
                                f"[Retry {attempt+1}/{max_retries}] Error calling agent_executor at step {t}, obs {i}: {error_msg}"
                            )
                            time.sleep(retry_delay)
                        else:
                            # Not retryable or last attempt - break to handle gracefully
                            print(
                                f"[Failed after {max_retries} attempts] Error calling agent_executor at step {t}, obs {i}: {error_msg}"
                            )
                            break

                # Handle case where output is None (all retries failed)
                if output is None:
                    print(
                        f"Warning: Failed to get output from agent_executor after {max_retries} attempts at step {t}, obs {i}"
                    )
                    if last_exception:
                        print(f"Last exception: {last_exception}")
                    # Set output to empty dict to prevent AttributeError later
                    output = {"output": "", "intermediate_steps": []}
                    # Mark this trajectory as ended
                    target_ended[i] = True
                    # Store error info
                    serialized_intermediate_steps = []
                    accumulated_steps = accumulated_intermediate_steps_dict.get(
                        instr_id, []
                    )
                    for action, obs_str in accumulated_steps:
                        if AgentAction is not None and isinstance(action, AgentAction):
                            serialized_intermediate_steps.append(
                                {
                                    "tool": action.tool,
                                    "tool_input": action.tool_input,
                                    "log": action.log,
                                    "observation": obs_str,
                                }
                            )
                        else:
                            try:
                                serialized_intermediate_steps.append(
                                    {
                                        "tool": getattr(action, "tool", ""),
                                        "tool_input": getattr(action, "tool_input", ""),
                                        "log": getattr(action, "log", ""),
                                        "observation": obs_str,
                                    }
                                )
                            except:
                                serialized_intermediate_steps.append(
                                    {
                                        "action": str(action),
                                        "observation": str(obs_str),
                                    }
                                )
                    # Save history even on error
                    history_list = []
                    if (
                        hasattr(self, "agent_executor")
                        and hasattr(self.agent_executor, "agent")
                        and hasattr(self.agent_executor.agent, "history")
                    ):
                        history_list = list(self.agent_executor.agent.history)

                    nav_info = {
                        "nav_input": {
                            "action_plan": stored_action_plans.get(instr_id, ""),
                            "init_observation": input_dict.get("init_observation", ""),
                            "observation": input_dict.get("observation", ""),
                            "intermediate_steps": serialized_intermediate_steps,  # For context maintenance
                            "history": history_list,  # Complete history list for full LLM input reconstruction
                        },
                        "viewpoint_id": None,  # Stop action on error
                        "a_t_list": [0],  # For backward compatibility
                    }
                    all_nav_info[instr_id][str(t)] = nav_info
                    continue  # Skip to next observation in batch

                # Extract viewpoint_id from output
                viewpoint_id = None
                intermediate_steps = output.get("intermediate_steps", [])
                output_text = output.get("output", "")

                if "Finished!" in output_text or "Final Answer" in output_text:
                    viewpoint_id = None  # Stop action
                elif len(intermediate_steps) > 0:
                    last_action, observation = intermediate_steps[-1]
                    if last_action.tool == "action_maker":
                        viewpoint_id = (
                            last_action.tool_input.strip('"').strip("'").strip()
                        )
                    elif last_action.tool == "back_tracer":
                        viewpoint_id = (
                            last_action.tool_input.strip('"').strip("'").strip()
                        )

                # Handle case where output is empty
                if output.get("output") == "" and len(intermediate_steps) == 0:
                    print(f"Warning: Empty output at step {t}, obs {i}")
                    # Set output to empty dict to prevent AttributeError later
                    output = {"output": "", "intermediate_steps": []}
                    # Mark this trajectory as ended
                    target_ended[i] = True
                    # Store error info
                    serialized_intermediate_steps = []
                    accumulated_steps = accumulated_intermediate_steps_dict.get(
                        instr_id, []
                    )
                    for action, obs_str in accumulated_steps:
                        if AgentAction is not None and isinstance(action, AgentAction):
                            serialized_intermediate_steps.append(
                                {
                                    "tool": action.tool,
                                    "tool_input": action.tool_input,
                                    "log": action.log,
                                    "observation": obs_str,
                                }
                            )
                        else:
                            try:
                                serialized_intermediate_steps.append(
                                    {
                                        "tool": getattr(action, "tool", ""),
                                        "tool_input": getattr(action, "tool_input", ""),
                                        "log": getattr(action, "log", ""),
                                        "observation": obs_str,
                                    }
                                )
                            except:
                                serialized_intermediate_steps.append(
                                    {
                                        "action": str(action),
                                        "observation": str(obs_str),
                                    }
                                )
                    # Save history even on error
                    history_list = []
                    if (
                        hasattr(self, "agent_executor")
                        and hasattr(self.agent_executor, "agent")
                        and hasattr(self.agent_executor.agent, "history")
                    ):
                        history_list = list(self.agent_executor.agent.history)

                    nav_info = {
                        "nav_input": {
                            "action_plan": stored_action_plans.get(instr_id, ""),
                            "init_observation": input_dict.get("init_observation", ""),
                            "observation": input_dict.get("observation", ""),
                            "intermediate_steps": serialized_intermediate_steps,  # For context maintenance
                            "history": history_list,  # Complete history list for full LLM input reconstruction
                        },
                        "viewpoint_id": None,  # Stop action on error
                        "a_t_list": [0],  # For backward compatibility
                    }
                    all_nav_info[instr_id][str(t)] = nav_info
                    continue  # Skip to next observation in batch

                # Process successful output
                # Note: viewpoint_id is already extracted from output above
                try:
                    # Extract intermediate_steps from output for context maintenance
                    intermediate_steps = output.get("intermediate_steps", [])

                    # Collect thoughts and observations for this trajectory (like rollout2)
                    for action, observation in intermediate_steps:
                        all_thoughts_dict[instr_id].append(action.log)
                        all_observations_dict[instr_id].append(observation)

                    # Check if orchestrator decided to stop (Final Answer)
                    output_text = output.get("output", "")
                    # Track last output_text for each trajectory
                    last_output_dict[instr_id] = output_text
                    if viewpoint_id is None:  # Stop action
                        final_output_dict[instr_id] = output_text

                    # Update accumulated_intermediate_steps for this trajectory
                    accumulated_steps = accumulated_intermediate_steps_dict.get(
                        instr_id, []
                    )
                    if len(intermediate_steps) > len(accumulated_steps):
                        accumulated_intermediate_steps_dict[instr_id] = (
                            intermediate_steps.copy()
                        )
                    elif len(intermediate_steps) > 0:
                        # Check if there are any new steps not in accumulated list
                        for step in intermediate_steps:
                            if step not in accumulated_steps:
                                accumulated_steps.append(step)
                        accumulated_intermediate_steps_dict[instr_id] = (
                            accumulated_steps
                        )

                    # Also update agent_executor's internal state after the call
                    if hasattr(self.agent_executor, "intermediate_steps"):
                        self.agent_executor.intermediate_steps = (
                            accumulated_intermediate_steps_dict[instr_id].copy()
                        )

                    # Prepare input_dict for next step (like rollout2 updates it in while loop)
                    # Only update if we have intermediate_steps and trajectory is not ending
                    if len(intermediate_steps) > 0:
                        # Get the last action to determine next step
                        last_action, last_observation = intermediate_steps[-1]

                        # Update input_dict for next step (like rollout2)
                        # For tool_chain mode, update now (observation comes from intermediate_steps)
                        if self.config.use_tool_chain:
                            # The observation from action_maker contains the new state
                            init_observation = stored_init_observations.get(
                                instr_id, ""
                            )
                            input_dict_dict[instr_id] = {
                                "action_plan": stored_action_plans.get(instr_id, ""),
                                "init_observation": init_observation,
                                "observation": last_observation,  # Use observation from action_maker
                            }
                        # For non-tool_chain mode, will be updated after make_equiv_action (below)

                    # Serialize intermediate_steps for JSON storage
                    # Convert AgentAction objects to dictionaries
                    serialized_intermediate_steps = []
                    accumulated_steps = accumulated_intermediate_steps_dict.get(
                        instr_id, []
                    )
                    for action, obs_str in accumulated_steps:
                        if AgentAction is not None and isinstance(action, AgentAction):
                            serialized_intermediate_steps.append(
                                {
                                    "tool": action.tool,
                                    "tool_input": action.tool_input,
                                    "log": action.log,
                                    "observation": obs_str,
                                }
                            )
                        else:
                            # Fallback: try to serialize as dict
                            try:
                                serialized_intermediate_steps.append(
                                    {
                                        "tool": getattr(action, "tool", ""),
                                        "tool_input": getattr(action, "tool_input", ""),
                                        "log": getattr(action, "log", ""),
                                        "observation": obs_str,
                                    }
                                )
                            except:
                                serialized_intermediate_steps.append(
                                    {
                                        "action": str(action),
                                        "observation": str(obs_str),
                                    }
                                )

                    # Convert action index for backward compatibility
                    candidate_dict = ob.get("candidate", {})
                    candidate_list = list(candidate_dict.keys())
                    a_t_index = 0
                    if viewpoint_id and viewpoint_id in candidate_dict:
                        a_t_index = candidate_list.index(viewpoint_id) + 1
                    elif viewpoint_id is None:
                        a_t_index = 0  # Stop action

                    # Collect nav_info for this step
                    # Save LLM input components (including intermediate_steps for context) and action output
                    # Get the actual observation used in this step
                    # For tool_chain mode, observation should be from input_dict (previous step's last_observation)
                    # For non-tool_chain mode, observation is not part of the input (only init_observation is used)
                    if self.config.use_tool_chain:
                        # For tool_chain mode, observation comes from input_dict (previous step's last_observation)
                        observation_to_save = input_dict.get("observation", "")
                        # For t=0, observation should be empty string (correct)
                        # For t>0, observation should be from previous step's last_observation (stored in input_dict_dict)
                    else:
                        # For non-tool_chain mode, observation is not used in input_dict
                        observation_to_save = ""

                    # Save complete history list for full LLM input reconstruction
                    # This is critical because VLNAgent._construct_scratchpad uses history[nav_step]
                    # instead of observation for action_maker tools, and get_full_inputs uses history[0]
                    # as init_observation when intermediate_steps exist
                    history_list = []
                    if (
                        hasattr(self, "agent_executor")
                        and hasattr(self.agent_executor, "agent")
                        and hasattr(self.agent_executor.agent, "history")
                    ):
                        history_list = list(self.agent_executor.agent.history)

                    nav_info = {
                        "nav_input": {
                            "action_plan": stored_action_plans.get(instr_id, ""),
                            "init_observation": input_dict.get("init_observation", ""),
                            "observation": observation_to_save,  # For tool_chain: previous step's last_observation; for non-tool_chain: empty
                            "intermediate_steps": serialized_intermediate_steps,  # For context maintenance
                            "history": history_list,  # Complete history list for full LLM input reconstruction
                        },
                        "viewpoint_id": viewpoint_id,  # Action output
                        "a_t_list": [int(a_t_index)],  # For backward compatibility
                    }

                    all_nav_info[instr_id][str(t)] = nav_info

                    # Prepare input for next step (like rollout2)
                    # Note: action was already executed by the tool (_make_action or _back_trace)
                    # so we don't need to execute it again, just prepare input_dict for next step
                    if viewpoint_id is None:
                        # Stop action - don't update input_dict_dict for next step
                        target_ended[i] = True
                        target_just_ended[i] = True
                    else:
                        # Update input_dict for next step (like rollout2)
                        # For tool_chain mode, use the observation from action_maker
                        if self.config.use_tool_chain:
                            # The observation from action_maker contains the new state
                            init_observation = stored_init_observations.get(
                                instr_id, ""
                            )
                            if len(intermediate_steps) > 0:
                                last_action, last_observation = intermediate_steps[-1]
                                input_dict_dict[instr_id] = {
                                    "action_plan": stored_action_plans.get(
                                        instr_id, ""
                                    ),
                                    "init_observation": init_observation,
                                    "observation": last_observation,  # Use observation from action_maker
                                }
                        else:
                            # For non-tool_chain, agent_executor manages state through history
                            # Get current observation from environment (like rollout2)
                            cur_obs = self.env._get_obs()[i]
                            feature = cur_obs["obs"]
                            navigable = cur_obs["candidate"]
                            objects = cur_obs["objects"]
                            heading = np.rad2deg(cur_obs["heading"])
                            elevation = np.rad2deg(cur_obs["elevation"])
                            orientation = (
                                f"\nheading: {heading:.2f}, elevation: {elevation:.2f}"
                            )

                            if self.config.use_relative_angle:
                                feature = self.modify_heading_angles(
                                    heading, feature, navigable, objects
                                )
                            if self.config.use_navigable:
                                navigable = self.get_navigable_str(
                                    heading, elevation, navigable
                                )

                            if self.config.use_relative_angle:
                                if self.config.use_navigable:
                                    init_observation = f"\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                                else:
                                    init_observation = (
                                        f"\n\tCurrent Viewpoint:\n{feature}"
                                    )
                            else:
                                if self.config.use_navigable:
                                    init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                                else:
                                    init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"

                            input_dict_dict[instr_id] = {
                                "action_plan": stored_action_plans.get(instr_id, ""),
                                "init_observation": init_observation,
                            }

                except Exception as e:
                    # Handle exceptions during output processing (after successful retry)
                    # This catches errors during action execution, history update, etc.
                    print(
                        f"Error in collect_nav_info_rollout2 at step {t}, obs {i}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    # Store error info
                    # Serialize intermediate_steps even on error
                    serialized_intermediate_steps = []
                    accumulated_steps = accumulated_intermediate_steps_dict.get(
                        instr_id, []
                    )
                    for action, obs_str in accumulated_steps:
                        if AgentAction is not None and isinstance(action, AgentAction):
                            serialized_intermediate_steps.append(
                                {
                                    "tool": action.tool,
                                    "tool_input": action.tool_input,
                                    "log": action.log,
                                    "observation": obs_str,
                                }
                            )
                        else:
                            try:
                                serialized_intermediate_steps.append(
                                    {
                                        "tool": getattr(action, "tool", ""),
                                        "tool_input": getattr(action, "tool_input", ""),
                                        "log": getattr(action, "log", ""),
                                        "observation": obs_str,
                                    }
                                )
                            except:
                                serialized_intermediate_steps.append(
                                    {
                                        "action": str(action),
                                        "observation": str(obs_str),
                                    }
                                )
                    # Save history even on error
                    history_list = []
                    if (
                        hasattr(self, "agent_executor")
                        and hasattr(self.agent_executor, "agent")
                        and hasattr(self.agent_executor.agent, "history")
                    ):
                        history_list = list(self.agent_executor.agent.history)

                    nav_info = {
                        "nav_input": {
                            "action_plan": stored_action_plans.get(instr_id, ""),
                            "init_observation": input_dict.get("init_observation", ""),
                            "observation": input_dict.get("observation", ""),
                            "intermediate_steps": serialized_intermediate_steps,  # For context maintenance
                            "history": history_list,  # Complete history list for full LLM input reconstruction
                        },
                        "viewpoint_id": None,  # Stop action on error
                        "a_t_list": [0],  # For backward compatibility
                    }
                    all_nav_info[instr_id][str(t)] = nav_info
                    target_ended[i] = True

            # Update ended state
            target_ended[:] = np.logical_or(target_ended, target_just_ended)

            # Update observations for next step (after processing all items in batch)
            # Use self.env since make_equiv_action updates self.env
            obs = self.env._get_obs()
            # Sync env with self.env
            env.set_scan_viewpoint_heading(self.env.get_scan_viewpoint_heading())
            for i in range(len(obs)):
                if i < len(previous_angle):
                    previous_angle[i] = {
                        "heading": obs[i].get("heading", 0),
                        "elevation": obs[i].get("elevation", 0),
                    }

            # Early exit if all ended
            if target_ended.all():
                break

        # Restore original max_iterations
        self.agent_executor.max_iterations = original_max_iter

        # Save to file
        if output_file is None:
            output_file = os.path.join(
                "NavGPT", "nav_24vp", "all_nav_outputs_update.json"
            )

        # Create directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Load existing data if file exists
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                existing_data = json.load(f)
            # Merge with new data
            for instr_id, step_data in all_nav_info.items():
                if instr_id not in existing_data:
                    existing_data[instr_id] = {}
                existing_data[instr_id].update(step_data)
            all_nav_info = existing_data

        # Save to file
        with open(output_file, "w") as f:
            json.dump(all_nav_info, f, indent=2)

        print(f"Saved NavGPT nav info to {output_file}")

        # Store results in trajectory (like rollout2)
        # Fill self.traj with collected information for each trajectory
        for i, traj_item in enumerate(self.traj):
            instr_id = traj_item["instr_id"]
            if instr_id in all_thoughts_dict:
                # Fill trajectory with collected information
                # Use final_output if available, otherwise use last output_text (like rollout2)
                self.traj[i]["llm_output"] = (
                    final_output_dict.get(instr_id)
                    if instr_id in final_output_dict
                    else last_output_dict.get(instr_id, "")
                )
                self.traj[i]["action_plan"] = cur_action_plan_dict.get(instr_id, "")
                self.traj[i]["llm_thought"] = all_thoughts_dict.get(instr_id, [])
                self.traj[i]["llm_observation"] = all_observations_dict.get(
                    instr_id, []
                )

        return self.traj

    def reconstruct_llm_input_from_json(
        self,
        nav_info: dict,
        config: Optional[Namespace] = None,
    ) -> Dict[str, Any]:
        """
        Reconstruct LLM input from saved nav_info JSON data.

        This function reconstructs the exact LLM input that was used at a specific step,
        allowing you to replay the LLM inference with the same input.

        Args:
            nav_info: Dictionary containing nav_input from JSON file, with keys:
                - action_plan: str
                - init_observation: str
                - observation: str (for tool_chain mode)
                - intermediate_steps: list of dicts with keys: tool, tool_input, log, observation
                - history: list of str (complete history list)
            config: Optional config object. If None, uses self.config.

        Returns:
            Dictionary with keys:
                - action_plan: str
                - init_observation: str (may be replaced by history[0] if intermediate_steps exist)
                - observation: str (for tool_chain mode)
                - agent_scratchpad: str (reconstructed from intermediate_steps and history)
                - intermediate_steps: list of (AgentAction, str) tuples (reconstructed)
        """
        if config is None:
            config = self.config

        nav_input = nav_info.get("nav_input", {})
        action_plan = nav_input.get("action_plan", "")
        init_observation = nav_input.get("init_observation", "")
        observation = nav_input.get("observation", "")
        intermediate_steps_data = nav_input.get("intermediate_steps", [])
        history_list = nav_input.get("history", [])

        # IMPORTANT: The saved intermediate_steps contains steps from 0 to t (including step t's inference result).
        # To reconstruct the LLM input at step t, we need intermediate_steps from 0 to t-1 (the input to step t).
        # So we need to remove the last element (step t's result) to get the correct input.
        # However, if intermediate_steps_data is empty or has only one element, we keep it as is
        # (for step 0, there are no previous steps, so intermediate_steps should be empty).
        if len(intermediate_steps_data) > 1:
            # Remove the last element (current step's inference result) to get the input to current step
            intermediate_steps_data = intermediate_steps_data[:-1]

        # Reconstruct intermediate_steps as (AgentAction, str) tuples
        intermediate_steps = []
        for step_data in intermediate_steps_data:
            if isinstance(step_data, dict):
                # Create AgentAction object
                action = AgentAction(
                    tool=step_data.get("tool", ""),
                    tool_input=step_data.get("tool_input", ""),
                    log=step_data.get("log", ""),
                )
                obs_str = step_data.get("observation", "")
                intermediate_steps.append((action, obs_str))

        # Reconstruct agent_scratchpad following VLNAgent._construct_scratchpad logic
        # We need observation_prefix and llm_prefix from ZeroShotAgent
        # Default values (can be overridden if needed)
        observation_prefix = "Observation: "
        llm_prefix = "Thought: "

        thoughts = ""
        nav_step = 1
        for i, (action, obs_str) in enumerate(intermediate_steps):
            thoughts += action.log
            if (i == len(intermediate_steps) - 1) or (
                action.tool != MAKE_ACTION_TOOL_NAME
            ):
                # Use observation directly
                thoughts += f"\n{observation_prefix}{obs_str}\n{llm_prefix}"
            else:
                # Use history[nav_step] instead of observation for action_maker tools
                if nav_step < len(history_list):
                    thoughts += (
                        f"\n{observation_prefix}{history_list[nav_step]}\n{llm_prefix}"
                    )
                else:
                    # Fallback to observation if history is missing
                    thoughts += f"\n{observation_prefix}{obs_str}\n{llm_prefix}"
                nav_step += 1

        # Truncate to MAX_SCRATCHPAD_LENGTH
        agent_scratchpad = thoughts[-MAX_SCRATCHPAD_LENGTH:]

        # Determine init_observation following VLNAgent.get_full_inputs logic
        # If intermediate_steps exist, use history[0] instead of saved init_observation
        if len(intermediate_steps) > 0 and len(history_list) > 0:
            actual_init_observation = history_list[0]
        else:
            actual_init_observation = init_observation

        return {
            "action_plan": action_plan,
            "init_observation": actual_init_observation,
            "observation": observation,
            "agent_scratchpad": agent_scratchpad,
            "intermediate_steps": intermediate_steps,
        }

    def infer_from_reconstructed_input(
        self,
        reconstructed_input: Dict[str, Any],
        config: Optional[Namespace] = None,
    ) -> Dict[str, Any]:
        """
        Perform LLM inference using reconstructed input.

        This function takes the reconstructed LLM input and performs inference,
        returning the LLM output.

        Args:
            reconstructed_input: Dictionary from reconstruct_llm_input_from_json()
            config: Optional config object. If None, uses self.config.

        Returns:
            Dictionary with LLM output, similar to agent_executor output:
                - output: str (LLM response)
                - intermediate_steps: list (if any new steps were generated)
        """
        if config is None:
            config = self.config

        # Prepare input for agent_executor
        input_dict = {
            "action_plan": reconstructed_input["action_plan"],
            "init_observation": reconstructed_input["init_observation"],
        }

        # Add observation if in tool_chain mode
        if config.use_tool_chain:
            input_dict["observation"] = reconstructed_input["observation"]

        # Save original state
        original_history = None
        if (
            hasattr(self, "agent_executor")
            and hasattr(self.agent_executor, "agent")
            and hasattr(self.agent_executor.agent, "history")
        ):
            original_history = self.agent_executor.agent.history.copy()

        # Temporarily set intermediate_steps in agent_executor
        original_intermediate_steps = None
        if hasattr(self.agent_executor, "intermediate_steps"):
            original_intermediate_steps = self.agent_executor.intermediate_steps.copy()

        # Set intermediate_steps
        self.agent_executor.intermediate_steps = reconstructed_input[
            "intermediate_steps"
        ]

        # Also set history in agent for proper scratchpad construction
        # History should be passed separately or accessed from temp attribute
        history_list = []
        if hasattr(self, "_temp_nav_info"):
            nav_input = self._temp_nav_info.get("nav_input", {})
            history_list = nav_input.get("history", [])

        if (
            hasattr(self, "agent_executor")
            and hasattr(self.agent_executor, "agent")
            and hasattr(self.agent_executor.agent, "history")
            and len(history_list) > 0
        ):
            self.agent_executor.agent.history = history_list.copy()

        # Call agent_executor with max_iterations=1 to get one step of inference
        original_max_iter = self.agent_executor.max_iterations
        self.agent_executor.max_iterations = 1

        try:
            # Perform inference
            output = self.agent_executor(input_dict)

            return output
        finally:
            # Restore original state
            self.agent_executor.max_iterations = original_max_iter
            if original_intermediate_steps is not None:
                self.agent_executor.intermediate_steps = original_intermediate_steps
            if original_history is not None:
                self.agent_executor.agent.history = original_history

    def restore_and_infer_from_json(
        self,
        json_file: str,
        instr_id: str,
        step: int,
        config: Optional[Namespace] = None,
    ) -> Dict[str, Any]:
        """
        Convenience function to restore LLM input from JSON and perform inference.

        Args:
            json_file: Path to JSON file containing nav_info
            instr_id: Instruction ID to restore
            step: Step number (as string or int)
            config: Optional config object. If None, uses self.config.

        Returns:
            Dictionary with:
                - reconstructed_input: Dict from reconstruct_llm_input_from_json()
                - llm_output: Dict from infer_from_reconstructed_input()
        """
        import json

        # Load JSON file
        with open(json_file, "r") as f:
            all_nav_info = json.load(f)

        # Get nav_info for specific instruction and step
        step_str = str(step)
        if instr_id not in all_nav_info:
            raise ValueError(f"Instruction ID {instr_id} not found in JSON file")
        if step_str not in all_nav_info[instr_id]:
            raise ValueError(
                f"Step {step} not found for instruction {instr_id} in JSON file"
            )

        nav_info = all_nav_info[instr_id][step_str]

        # Reconstruct input
        reconstructed_input = self.reconstruct_llm_input_from_json(nav_info, config)

        # Store nav_info for infer_from_reconstructed_input to access history
        # We'll pass it through a temporary attribute
        self._temp_nav_info = nav_info

        try:
            # Perform inference
            llm_output = self.infer_from_reconstructed_input(
                reconstructed_input, config
            )

            return {
                "reconstructed_input": reconstructed_input,
                "llm_output": llm_output,
            }
        finally:
            # Clean up
            if hasattr(self, "_temp_nav_info"):
                delattr(self, "_temp_nav_info")
