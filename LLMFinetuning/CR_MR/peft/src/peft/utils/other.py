# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import inspect
import warnings

# needed for prefix-tuning of bloom model
def bloom_model_postprocess_past_key_value(past_key_values):
    past_key_values = torch.cat(past_key_values)
    total_layers, batch_size, num_attention_heads, num_virtual_tokens, head_dim = past_key_values.shape
    keys = past_key_values[: total_layers // 2]
    keys = keys.transpose(2, 3).reshape(
        total_layers // 2, batch_size * num_attention_heads, head_dim, num_virtual_tokens
    )
    values = past_key_values[total_layers // 2 :]
    values = values.reshape(total_layers // 2, batch_size * num_attention_heads, num_virtual_tokens, head_dim)

    return tuple(zip(keys, values))

def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
    r"""
    Note this method only works for `transformers` models.

    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32 4- Freezing the base model layers to ensure they are not updated during training


    Args:
        model (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
        use_gradient_checkpointing (`bool`, *optional*, defaults to `True`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        gradient_checkpointing_kwargs (`dict`, *optional*, defaults to `None`):
            Keyword arguments to pass to the gradient checkpointing function, please refer to the documentation of
            `torch.utils.checkpoint.checkpoint` for more details about the arguments that you can pass to that method.
            Note this is only available in the latest transformers versions (> 4.34.1).
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    is_aqlm_quantized = getattr(model, "quantization_method", None) == "aqlm"
    is_eetq_quantized = getattr(model, "quantization_method", None) == "eetq"
    is_torchao_quantized = getattr(model, "quantization_method", None) == "torchao"
    is_hqq_quantized = getattr(model, "quantization_method", None) == "hqq" or getattr(model, "hqq_quantized", False)

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    if (
        not is_gptq_quantized
        and not is_aqlm_quantized
        and not is_eetq_quantized
        and not is_hqq_quantized
        and not is_torchao_quantized
    ):
        # cast all non INT8 parameters to fp32
        for param in model.parameters():
            if (
                (param.dtype == torch.float16) or (param.dtype == torch.bfloat16)
            ) and param.__class__.__name__ != "Params4bit":
                param.data = param.data.to(torch.float32)

    if (
        loaded_in_kbit
        or is_gptq_quantized
        or is_aqlm_quantized
        or is_eetq_quantized
        or is_hqq_quantized
        or is_torchao_quantized
    ) and use_gradient_checkpointing:
        # When having `use_reentrant=False` + gradient_checkpointing, there is no need for this hack
        if "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]:
            # For backward compatibility
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # To support older transformers versions, check if the model supports gradient_checkpointing_kwargs
        _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
            inspect.signature(model.gradient_checkpointing_enable).parameters
        )

        if not _supports_gc_kwargs and len(gradient_checkpointing_kwargs) > 0:
            warnings.warn(
                "gradient_checkpointing_kwargs is not supported in this version of transformers. The passed kwargs will be ignored."
                " if you want to use that feature, please upgrade to the latest version of transformers.",
                FutureWarning,
            )

        gc_enable_kwargs = (
            {} if not _supports_gc_kwargs else {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs}
        )

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable(**gc_enable_kwargs)
    return model

def prepare_model_for_int8_training(
    model, output_embedding_layer_name="lm_head", use_gradient_checkpointing=True, layer_norm_names=["layer_norm"]
):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

        if loaded_in_8bit:
            # cast layer norm in fp32 for stability for 8bit models
            if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
                param.data = param.data.to(torch.float32)

    if loaded_in_8bit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):
            r"""
            Manually cast to the expected dtype of the lm_head as sometimes there is a final layer norm that is casted
            in fp32

            """

            def forward(self, x):
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    return model


TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING = {
    "bloom": bloom_model_postprocess_past_key_value,
}


# copied from transformers.models.bart.modeling_bart
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`): input ids
        pad_token_id (`int`): The id of the `padding` token.
        decoder_start_token_id (`int`): The id of the `start` token.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _set_trainable(model):
    if model.modules_to_save is not None:
        for name, param in model.named_parameters():
            if any(module_name in name for module_name in model.modules_to_save):
                param.requires_grad = True


def fsdp_auto_wrap_policy(model):
    import functools
    import os

    from accelerate import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    from ..tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(
            PrefixEncoder,
            PromptEncoder,
            PromptEmbedding,
            FullyShardedDataParallelPlugin.get_module_class_from_name(
                model, os.environ.get("FSDP_TRANSFORMER_CLS_TO_WRAP", "")
            ),
        ),
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight
