from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import Field

# from LLMs.llama.llama import Llama

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from openai import OpenAI
import base64


generation_key = "YOUR_API_KEY"


class Custom_Openai(LLM):
    model: Any  #: :meta private:
    max_tokens: int = 1000
    response_format: str = "str"

    client: OpenAI = Field(
        default_factory=lambda: OpenAI(
            api_key=generation_key,
            base_url="https://api.chatanywhere.tech/v1",
        )
    )
    # self.client = OpenAI(
    #     api_key=generation_key,
    #     base_url="https://api.chatanywhere.tech/v1",
    # )
    # def __init__(
    #     self, model: str, max_tokens: int = 1000, response_format: str = "str"
    # ):
    #     self.model = model
    #     self.max_tokens = max_tokens
    #     self.response_format = response_format
    #     self.client = OpenAI(
    #         api_key=generation_key,
    #         base_url="https://api.chatanywhere.tech/v1",
    #     )

    """Key word arguments passed to the model."""
    # ckpt_dir: str
    # tokenizer_path: str
    # temperature: float = 0.6
    # top_p: float = 0.9
    # max_seq_len: int = 128
    # max_gen_len: int = 64
    # max_batch_size: int = 4

    @property
    def _llm_type(self) -> str:
        return "custom_openai"

    # @classmethod
    # def from_model_id(
    #     cls,
    #     ckpt_dir: str,
    #     tokenizer_path: str,
    #     temperature: float = 0.6,
    #     top_p: float = 0.9,
    #     max_seq_len: int = 128,
    #     max_gen_len: int = 64,
    #     max_batch_size: int = 4,
    #     **kwargs: Any,
    # ) -> LLM:
    #     """Construct the pipeline object from model_id and task."""

    #     model = Llama.build(
    #         ckpt_dir=ckpt_dir,
    #         tokenizer_path=tokenizer_path,
    #         max_seq_len=max_seq_len,
    #         max_batch_size=max_batch_size,
    #     )

    #     return cls(
    #         model=model,
    #         ckpt_dir=ckpt_dir,
    #         tokenizer_path=tokenizer_path,
    #         # set as default
    #         temperature=0.6,
    #         top_p=top_p,
    #         max_seq_len=max_seq_len,
    #         max_gen_len=max_gen_len,
    #         max_batch_size=max_batch_size,
    #         **kwargs,
    #     )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        response_format: Optional[str] = None,
    ) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")

        # result = self.model.text_completion(
        #     [prompt],
        #     max_gen_len=self.max_gen_len,
        #     temperature=self.temperature,
        #     top_p=self.top_p,
        # )
        # return result[0]["generation"]

        # print(prompt)
        # print(len(prompt))
        user_content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": user_content}]
        print("messages: ", messages)

        if response_format:
            chat_message = self.completion_with_backoff(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=self.max_tokens,
                response_format=self.response_format,
            )
        else:
            chat_message = self.completion_with_backoff(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=self.max_tokens,
            )

        return chat_message.choices[0].message.content

    # @property
    # def _identifying_params(self) -> Mapping[str, Any]:
    #     """Get the identifying parameters."""
    #     return {
    #         "ckpt_dir": self.ckpt_dir,
    #         "tokenizer_path": self.tokenizer_path,
    #         "temperature": self.temperature,
    #         "top_p": self.top_p,
    #         "max_seq_len": self.max_seq_len,
    #         "max_gen_len": self.max_gen_len,
    #         "max_batch_size": self.max_batch_size,
    #     }
