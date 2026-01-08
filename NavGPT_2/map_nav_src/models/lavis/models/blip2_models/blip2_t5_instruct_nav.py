"""
Copyright (c) 2023, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import string
import random
import copy
import os

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.utils import is_url
from lavis.common.registry import registry
from lavis.common.dist_utils import download_cached_file
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


DEFAULT_IMG_TOKEN = "[IMG]"
DEFAULT_IMG_END_TOKEN = "[/IMG]"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_INSTRUCTION_TOKEN = "[INST]"
DEFAULT_INSTRUCTION_END_TOKEN = "[/INST]"


@registry.register_model("blip2_t5_instruct_nav")
class Blip2T5InstructNav(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - flant5xl
        - flant5xxl
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5_instruct_nav", "flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xl": "configs/models/blip2/blip2_instruct_nav_flant5xl.yaml",
        "flant5xxl": "configs/models/blip2/blip2_instruct_nav_flant5xxl.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=1024,
        max_output_txt_len=1024,
        apply_lemmatizer=False,
        num_few_shot_examples=0,
        few_shot_prob=0,
        qformer_text_input=True,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        # Resolve t5_model path: support relative paths from NavGPT-2 root
        # Also expand ~ to home directory if present
        import os

        if t5_model.startswith("~"):
            t5_model = os.path.expanduser(t5_model)
        t5_model_path = self._resolve_t5_model_path(t5_model)

        # Load T5 tokenizer with error handling
        try:
            self.t5_tokenizer = T5TokenizerFast.from_pretrained(
                t5_model_path, truncation_side="left"
            )
        except OSError as e:
            import os

            # Check if there's a local directory with the same name
            if os.path.exists(t5_model_path) and os.path.isdir(t5_model_path):
                # Try loading from local directory
                try:
                    self.t5_tokenizer = T5TokenizerFast.from_pretrained(
                        t5_model_path, truncation_side="left", local_files_only=True
                    )
                except Exception as e2:
                    raise OSError(
                        f"Failed to load tokenizer from '{t5_model_path}'. "
                        f"Please ensure the model is downloaded or specify a valid path. "
                        f"Original error: {e}. Local load error: {e2}"
                    )
            else:
                # Try with cache_dir from environment
                cache_dir = (
                    os.environ.get("HF_HOME")
                    or os.environ.get("TRANSFORMERS_CACHE")
                    or os.path.expanduser("~/.cache/huggingface")
                )
                try:
                    self.t5_tokenizer = T5TokenizerFast.from_pretrained(
                        t5_model_path, truncation_side="left", cache_dir=cache_dir
                    )
                except Exception as e2:
                    raise OSError(
                        f"Failed to load tokenizer for '{t5_model_path}'. "
                        f"Please ensure you have internet connection to download from HuggingFace, "
                        f"or download the model manually and specify the local path. "
                        f"Original error: {e}. Cache dir error: {e2}"
                    )

        # add special tokens
        num_new_tokens = self.t5_tokenizer.add_special_tokens(
            {
                "bos_token": "[DEC]",
                "additional_special_tokens": [
                    DEFAULT_IMG_TOKEN,
                    DEFAULT_IMG_END_TOKEN,
                    DEFAULT_IMAGE_TOKEN,
                    DEFAULT_INSTRUCTION_TOKEN,
                    DEFAULT_INSTRUCTION_END_TOKEN,
                ],
            }
        )

        try:
            self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(
                t5_model_path, truncation_side="right"
            )
        except OSError as e:
            import os

            # Check if there's a local directory with the same name
            if os.path.exists(t5_model_path) and os.path.isdir(t5_model_path):
                try:
                    self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(
                        t5_model_path, truncation_side="right", local_files_only=True
                    )
                except Exception as e2:
                    raise OSError(
                        f"Failed to load output tokenizer from '{t5_model_path}'. "
                        f"Please ensure the model is downloaded or specify a valid path. "
                        f"Original error: {e}. Local load error: {e2}"
                    )
            else:
                cache_dir = (
                    os.environ.get("HF_HOME")
                    or os.environ.get("TRANSFORMERS_CACHE")
                    or os.path.expanduser("~/.cache/huggingface")
                )
                try:
                    self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(
                        t5_model_path, truncation_side="right", cache_dir=cache_dir
                    )
                except Exception as e2:
                    raise OSError(
                        f"Failed to load output tokenizer for '{t5_model_path}'. "
                        f"Please ensure you have internet connection to download from HuggingFace, "
                        f"or download the model manually and specify the local path. "
                        f"Original error: {e}. Cache dir error: {e2}"
                    )

        # Load T5 config and model with error handling
        try:
            t5_config = T5Config.from_pretrained(t5_model_path)
        except OSError as e:
            import os

            if os.path.exists(t5_model_path) and os.path.isdir(t5_model_path):
                try:
                    t5_config = T5Config.from_pretrained(
                        t5_model_path, local_files_only=True
                    )
                except Exception as e2:
                    raise OSError(
                        f"Failed to load T5 config from '{t5_model_path}'. "
                        f"Original error: {e}. Local load error: {e2}"
                    )
            else:
                cache_dir = (
                    os.environ.get("HF_HOME")
                    or os.environ.get("TRANSFORMERS_CACHE")
                    or os.path.expanduser("~/.cache/huggingface")
                )
                try:
                    t5_config = T5Config.from_pretrained(
                        t5_model_path, cache_dir=cache_dir
                    )
                except Exception as e2:
                    raise OSError(
                        f"Failed to load T5 config for '{t5_model_path}'. "
                        f"Original error: {e}. Cache dir error: {e2}"
                    )

        t5_config.dense_act_fn = "gelu"

        # Check if model directory exists and what file format it uses
        import os

        use_safetensors = True
        if os.path.exists(t5_model_path) and os.path.isdir(t5_model_path):
            # Check if directory contains .bin files instead of .safetensors
            has_bin = any(
                f.endswith(".bin")
                for f in os.listdir(t5_model_path)
                if os.path.isfile(os.path.join(t5_model_path, f))
            )
            has_safetensors = any(
                f.endswith(".safetensors")
                for f in os.listdir(t5_model_path)
                if os.path.isfile(os.path.join(t5_model_path, f))
            )
            if has_bin and not has_safetensors:
                use_safetensors = False

        try:
            # Try loading with appropriate format
            if use_safetensors:
                self.t5_model = T5ForConditionalGeneration.from_pretrained(
                    t5_model_path, config=t5_config
                )
            else:
                # Use .bin files
                self.t5_model = T5ForConditionalGeneration.from_pretrained(
                    t5_model_path, config=t5_config, use_safetensors=False
                )
        except OSError as e:
            import os

            if os.path.exists(t5_model_path) and os.path.isdir(t5_model_path):
                # Try with opposite format if first attempt failed
                try:
                    if use_safetensors:
                        # First attempt used safetensors, try with .bin
                        self.t5_model = T5ForConditionalGeneration.from_pretrained(
                            t5_model_path,
                            config=t5_config,
                            local_files_only=True,
                            use_safetensors=False,
                        )
                    else:
                        # First attempt used .bin, try with safetensors
                        self.t5_model = T5ForConditionalGeneration.from_pretrained(
                            t5_model_path, config=t5_config, local_files_only=True
                        )
                except Exception as e2:
                    # If both formats fail, try without specifying format (let transformers decide)
                    try:
                        self.t5_model = T5ForConditionalGeneration.from_pretrained(
                            t5_model_path, config=t5_config, local_files_only=True
                        )
                    except Exception as e3:
                        raise OSError(
                            f"Failed to load T5 model from '{t5_model_path}'. "
                            f"Tried both .safetensors and .bin formats. "
                            f"Original error: {e}. "
                            f"Alternative format error: {e2}. "
                            f"Fallback error: {e3}. "
                            f"Please ensure the model directory contains valid model files."
                        )
            else:
                cache_dir = (
                    os.environ.get("HF_HOME")
                    or os.environ.get("TRANSFORMERS_CACHE")
                    or os.path.expanduser("~/.cache/huggingface")
                )
                try:
                    self.t5_model = T5ForConditionalGeneration.from_pretrained(
                        t5_model_path, config=t5_config, cache_dir=cache_dir
                    )
                except Exception as e2:
                    raise OSError(
                        f"Failed to load T5 model for '{t5_model_path}'. "
                        f"Original error: {e}. Cache dir error: {e2}"
                    )

        # resize token embeddings
        self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))
        if num_new_tokens > 0:
            input_embeddings = self.t5_model.get_input_embeddings().weight.data
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            input_embeddings[-num_new_tokens:] = input_embeddings_avg

        # Get supported dtype (bfloat16 if supported, otherwise float16)
        supported_dtype = self._get_supported_dtype(torch.bfloat16)

        # Check T5 model weights for NaN/Inf before conversion
        nan_count = 0
        inf_count = 0
        for name, param in self.t5_model.named_parameters():
            if torch.isnan(param.data).any():
                nan_count += torch.isnan(param.data).sum().item()
            if torch.isinf(param.data).any():
                inf_count += torch.isinf(param.data).sum().item()
        if nan_count > 0 or inf_count > 0:
            print(
                f"[WARNING] T5 model has NaN/Inf in weights before dtype conversion: NaN={nan_count}, Inf={inf_count}"
            )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            # Convert to supported dtype, but preserve values carefully to avoid NaN
            # If converting from bfloat16 to float16, values might overflow
            if param.data.dtype != supported_dtype:
                # Check for potential overflow when converting bfloat16 to float16
                if (
                    param.data.dtype == torch.bfloat16
                    and supported_dtype == torch.float16
                ):
                    # Clamp extreme values to prevent overflow
                    max_val = torch.finfo(torch.float16).max
                    min_val = torch.finfo(torch.float16).min
                    param_float = param.data.float()
                    # Check for NaN/Inf before clamping
                    if torch.isnan(param_float).any() or torch.isinf(param_float).any():
                        print(
                            f"[WARNING] NaN/Inf in param {name} before conversion: NaN={torch.isnan(param_float).sum().item()}, Inf={torch.isinf(param_float).sum().item()}"
                        )
                    param_float = torch.clamp(param_float, min_val, max_val)
                    param.data = param_float.to(supported_dtype)
                else:
                    param.data = param.data.to(supported_dtype)

            # Final check for NaN/Inf after conversion (only report, do not fix)
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                print(
                    f"[ERROR] NaN/Inf in param {name} after dtype conversion: NaN={torch.isnan(param.data).sum().item()}, Inf={torch.isinf(param.data).sum().item()}"
                )

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.num_few_shot_examples = num_few_shot_examples
        self.few_shot_prob = few_shot_prob

        self.qformer_text_input = qformer_text_input

    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')

        image = samples["images"]
        qformer_text_inputs = samples["qformer_text_input"]
        with self.maybe_autocast():
            # image_embeds = self.ln_vision(self.visual_encoder(image))
            image_embeds = self.ln_vision(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                qformer_text_inputs,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                image.device
            )
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_t5 = self.t5_proj(
            query_output.last_hidden_state[:, : query_tokens.size(1), :]
        )

        fs_embeds, fs_atts = None, None
        if self.few_shot_prob > 0 and "few_shot_samples" in samples.keys():
            fs_embeds, fs_atts = self.prepare_few_shot_embeds(
                samples["few_shot_samples"]
            )

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            output_tokens = self.t5_output_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = input_tokens.attention_mask

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            # Get image tokens
            self.image_token_id = self.t5_tokenizer.convert_tokens_to_ids(["<image>"])[
                0
            ]
            all_image_indices = (input_tokens.input_ids == self.image_token_id).to(
                inputs_t5.device
            )

            assert (
                input_tokens.input_ids[all_image_indices].shape[0]
                == inputs_t5.shape[0] * inputs_t5.shape[1]
            ), f"image tokens in input ids {input_tokens.input_ids[input_tokens.input_ids == self.image_token_id].shape[0]} != the number of image tokens {inputs_t5.shape[0]}*{inputs_t5.shape[1]}"
            assert (
                inputs_t5.shape[-1] == inputs_embeds.shape[-1]
            ), f"{inputs_t5.shape[-1]} != {inputs_embeds.shape[-1]}"

            inputs_t5 = inputs_t5.reshape(-1, inputs_t5.shape[-1]).to(
                inputs_embeds.dtype
            )
            inputs_embeds[all_image_indices] = inputs_t5

            if fs_embeds is not None:
                inputs_embeds = torch.cat([fs_embeds, inputs_embeds], dim=1)
                encoder_atts = torch.cat([fs_atts, encoder_atts], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            return {"loss": loss}

    def prepare_few_shot_embeds(self, samples):
        this_n_fs = random.choices(
            list(range(self.num_few_shot_examples + 1)),
            weights=[1 - self.few_shot_prob]
            + [self.few_shot_prob / self.num_few_shot_examples]
            * self.num_few_shot_examples,
        )[0]

        if this_n_fs == 0:
            return None, None

        images = []
        text_input = []
        for sample in samples:
            for n in range(this_n_fs):
                images.append(sample["image"][n])
                text_input.append(sample["text_input"][n])
        images = torch.stack(images, dim=0)

        image = images

        # with self.maybe_autocast():
        #     image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                image.device
            )
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_t5 = self.t5_proj(
            query_output.last_hidden_state[:, : query_tokens.size(1), :]
        )
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        if this_n_fs > 1:
            encoder_atts = encoder_atts.reshape(
                encoder_atts.size(0) // this_n_fs, encoder_atts.size(1) * this_n_fs
            )
            inputs_embeds = inputs_embeds.reshape(
                inputs_embeds.size(0) // this_n_fs,
                inputs_embeds.size(1) * this_n_fs,
                inputs_embeds.size(2),
            )

        return inputs_embeds, encoder_atts

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["images"]
        qformer_text_inputs = samples["qformer_text_input"]
        # with self.maybe_autocast():
        #     image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                qformer_text_inputs,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                image.device
            )
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_t5 = self.t5_proj(
            query_output.last_hidden_state[:, : query_tokens.size(1), :]
        )

        fs_embeds, fs_atts = None, None
        if self.few_shot_prob > 0 and "few_shot_samples" in samples.keys():
            fs_embeds, fs_atts = self.prepare_few_shot_embeds(
                samples["few_shot_samples"]
            )

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = input_tokens.attention_mask

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            # Get image tokens
            self.image_token_id = self.t5_tokenizer.convert_tokens_to_ids(["<image>"])[
                0
            ]
            all_image_indices = (input_tokens.input_ids == self.image_token_id).to(
                inputs_t5.device
            )

            assert (
                input_tokens.input_ids[all_image_indices].shape[0]
                == inputs_t5.shape[0] * inputs_t5.shape[1]
            ), f"image tokens in input ids {input_tokens.input_ids[input_tokens.input_ids == self.image_token_id].shape[0]} != the number of image tokens {inputs_t5.shape[0]}*{inputs_t5.shape[1]}"
            assert (
                inputs_t5.shape[-1] == inputs_embeds.shape[-1]
            ), f"{inputs_t5.shape[-1]} != {inputs_embeds.shape[-1]}"

            inputs_t5 = inputs_t5.reshape(-1, inputs_t5.shape[-1]).to(
                inputs_embeds.dtype
            )
            inputs_embeds[all_image_indices] = inputs_t5

            if fs_embeds is not None:
                inputs_embeds = torch.cat([fs_embeds, inputs_embeds], dim=1)
                encoder_atts = torch.cat([fs_atts, encoder_atts], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs,
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if "ocr_tokens" in samples:
                    text_input = [
                        prompt.format(
                            ", ".join(samples["ocr_tokens"][i][:30]),
                            samples["text_input"][i],
                        )
                        for i in range(len(samples["text_input"]))
                    ]
                elif "choices" in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [
                            f"({string.ascii_lowercase[j]}) {ch}"
                            for j, ch in enumerate(samples["choices"][i])
                        ]
                        this_choices = " ".join(this_choices)
                        text_input.append(
                            prompt.format(samples["text_input"][i], this_choices)
                        )
            else:
                text_input = [
                    prompt.format(question) for question in samples["text_input"]
                ]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty,
        )

        if self._apply_lemmatizer or (
            "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]
        ):
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if "context" in samples.keys():
                    this_sample["context"] = [samples["context"][i]]

                if "history" in samples.keys():
                    this_sample["history"] = [samples["history"][i]]

                if "caption" in samples.keys():
                    this_sample["caption"] = [samples["caption"][i]]

                this_result = self._predict_class(
                    this_sample, candidates[i], n_segments
                )
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - prompt: the instruction
            candidates:
                (list): A list of candidate class names;
            n_segments:
                (int): Split the candidates into n_segments and predict one by one. This is useful when the number of candidates is too large.
        Returns:
            output_class: predicted class index
        """

        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert (
                len(prompt) == bs
            ), "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [
                    prompt[i].format(*samples["text_input"][i])
                    for i in range(len(prompt))
                ]
            else:
                prompt = [
                    prompt[i].format(samples["text_input"][i])
                    for i in range(len(prompt))
                ]

        # scienceqa
        if "context" in samples.keys() and samples["context"] != "":
            prompt = [
                f'context: {samples["context"][i]}. {prompt[i]}'
                for i in range(len(prompt))
            ]

        # visual dialog
        if "history" in samples.keys() and samples["history"][0] != "":
            prompt = [
                f'dialog history: {samples["history"][i]}\n{prompt[i]}'
                for i in range(len(prompt))
            ]

        if "caption" in samples.keys() and samples["caption"][0] != "":
            prompt = [
                f'This image has the caption "{samples["caption"][i]}". {prompt[i]}'
                for i in range(len(prompt))
            ]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                image.device
            )
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_t5, atts_t5 = [], []
            for j in range(image.size(2)):
                this_frame = image[:, :, j, :, :]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(
                        frame_embeds.size()[:-1], dtype=torch.long
                    ).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_t5 = self.t5_proj(
                    frame_query_output.last_hidden_state[:, : query_tokens.size(1), :]
                )
                frame_atts_t5 = torch.ones(
                    frame_inputs_t5.size()[:-1], dtype=torch.long
                ).to(image.device)
                inputs_t5.append(frame_inputs_t5)
                atts_t5.append(frame_atts_t5)
            inputs_t5 = torch.cat(inputs_t5, dim=1)
            atts_t5 = torch.cat(atts_t5, dim=1)
        else:
            # with self.maybe_autocast():
            #     image_embeds = self.ln_vision(self.visual_encoder(image))
            image_embeds = image
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_t5 = self.t5_proj(
                query_output.last_hidden_state[:, : query_tokens.size(1), :]
            )
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(
                image.device
            )

        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)
        output_tokens = self.t5_tokenizer(
            candidates, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        n_cands = len(candidates)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            encoder_outputs = self.t5_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
            )

            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                # this_encoder_outputs = copy.deepcopy(encoder_outputs)
                this_encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0].clone(),
                )

                this_encoder_outputs["last_hidden_state"] = this_encoder_outputs[
                    0
                ].repeat_interleave(seg_len, dim=0)
                this_encoder_atts = encoder_atts.repeat_interleave(seg_len, dim=0)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len
                this_output_tokens_ids = output_tokens.input_ids[start_i:end_i].repeat(
                    bs, 1
                )
                this_output_tokens_atts = output_tokens.attention_mask[
                    start_i:end_i
                ].repeat(bs, 1)

                this_targets = this_output_tokens_ids.masked_fill(
                    this_output_tokens_ids == self.t5_tokenizer.pad_token_id, -100
                )

                outputs = self.t5_model(
                    encoder_outputs=this_encoder_outputs,
                    attention_mask=this_encoder_atts,
                    decoder_attention_mask=this_output_tokens_atts,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )
                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)

            # encoder_outputs['last_hidden_state'] = encoder_outputs[0].repeat_interleave(n_cands, dim=0)
            # encoder_atts = encoder_atts.repeat_interleave(n_cands, dim=0)
            # output_tokens.input_ids = output_tokens.input_ids.repeat(bs, 1)
            # output_tokens.attention_mask = output_tokens.attention_mask.repeat(bs, 1)

            # # compute the LM loss for each candidate (sum logprob across all tokens) and select the highest
            # targets = output_tokens.input_ids.masked_fill(output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100)

            # outputs = self.t5_model(
            #     encoder_outputs=encoder_outputs,
            #     attention_mask=encoder_atts,
            #     decoder_attention_mask=output_tokens.attention_mask,
            #     return_dict=True,
            #     labels=targets,
            #     reduction="none",
            # )
            # loss = outputs.loss

            # loss = loss.reshape(bs, n_cands)
            # output_class_ranks = torch.argsort(loss, dim=-1) # (bs, num_candidates)

        return output_class_ranks

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 1024)
        max_output_txt_len = cfg.get("max_output_txt_len", 1024)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        num_few_shot_examples = cfg.get("num_few_shot_examples", 0)
        few_shot_prob = cfg.get("few_shot_prob", 0.0)

        qformer_text_input = cfg.get("qformer_text_input", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_prob=few_shot_prob,
            qformer_text_input=qformer_text_input,
        )

        # if qformer_text_input:
        #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
        #     model.load_from_pretrained(
        #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
        #     )

        model.load_checkpoint_from_config(cfg)

        return model

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if url_or_filename.split(".")[-1] == "bin":
            state_dict = {}
            for name, param in checkpoint.items():
                if "query_tokens" in name or "Qformer" in name:
                    state_dict[name] = param
        else:
            state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    def _resolve_t5_model_path(self, t5_model):
        """
        Resolve T5 model path. Supports:
        1. HuggingFace model IDs (e.g., "google/flan-t5-xl")
        2. Absolute paths
        3. Relative paths from NavGPT-2 root directory

        Args:
            t5_model: Model identifier or path

        Returns:
            Resolved model path
        """
        import os

        # If it's a HuggingFace model ID (contains '/'), check if it's a path first
        if "/" in t5_model:
            # Check if it's an absolute path
            if os.path.isabs(t5_model):
                return t5_model

            # Check if it exists as a relative path from current directory
            if os.path.exists(t5_model) and os.path.isdir(t5_model):
                return os.path.abspath(t5_model)

            # Try relative to NavGPT-2 root (repo_root)
            try:
                from lavis.common.registry import registry

                repo_root = registry.get_path("repo_root")
                navgpt2_root = os.path.abspath(os.path.join(repo_root, "..", ".."))
                potential_path = os.path.join(navgpt2_root, t5_model)
                if os.path.exists(potential_path) and os.path.isdir(potential_path):
                    return os.path.abspath(potential_path)
            except:
                pass

            # If not found as local path, assume it's a HuggingFace model ID
            return t5_model
        else:
            # Single word - likely HuggingFace model ID
            return t5_model
