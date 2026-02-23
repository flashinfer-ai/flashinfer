import logging
from typing import Callable, Tuple, Union

import torch
from flashinfer.parallel_attention.utils import convert_qkv_layout, convert_output_layout

logger = logging.getLogger(__name__)

try:
    import flash_attn_interface
except ImportError:
    flash_attn_interface = None
    logger.warning("FlashAttn3 is not installed.")


class AttentionOpManager:
    _attn_registry = {}

    @classmethod
    def op_type(cls):
        return "attention"

    @classmethod
    def set_attn_config(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"'{cls.__name__}' has no attribute '{key}'")

    @classmethod
    def register_attn(cls, attn_type):
        def decorator(attn_class):
            # Register the attention class
            cls._attn_registry[attn_type] = attn_class
            return attn_class

        return decorator

    @classmethod
    def get_impl(cls, name=None):
        if name is None:
            name = cls.attn_type
        attn_class = cls._attn_registry.get(name)
        if attn_class is None:
            raise ValueError(f"Attention function {name} not found in registry")
        return attn_class()  # Create and return an instance

    @classmethod
    def get_registered_types(cls):
        return list(cls._attn_registry.keys())




@AttentionOpManager.register_attn("flash-attn3")
class FlashAttn3:
    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        tensor_layout="HND",
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=0,
        max_seqlen_k=0,
    ):

        if flash_attn_interface is None:
            raise ImportError("FlashAttn3 is not installed")

        logger.debug("Using FlashAttn3")
        if attn_mask is not None:
            raise NotImplementedError("Attn mask not supported for FlashAttn3")
        if dropout_p != 0.0:
            raise NotImplementedError("Dropout is not supported for FlashAttn3")
        if tensor_layout not in ["HND", "NHD"]:
            raise NotImplementedError("Tensor layout not supported for FlashAttn3")

        if tensor_layout == "HND":
            query, key, value = convert_qkv_layout(query, key, value, src_layout="HND", dst_layout="NHD")

        # FA3 only supports float16 and bfloat16
        origin_dtype = query.dtype
        if query.dtype not in [torch.float16, torch.bfloat16]:
            query = query.to(torch.float16)
            key = key.to(torch.float16)
            value = value.to(torch.float16)

        if cu_seqlens_q is None:
            query = torch.unsqueeze(query, dim=0)
            key = torch.unsqueeze(key, dim=0)
            value = torch.unsqueeze(value, dim=0)
            output = flash_attn_interface.flash_attn_func(
                q=query,
                k=key,
                v=value,
                softmax_scale=scale,
                causal=is_causal,
                qv=None,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                window_size=(-1, -1),
                attention_chunk=0,
                softcap=0.0,
                num_splits=1,
                pack_gqa=None,
                deterministic=False,
                sm_margin=0,
                return_attn_probs=return_lse,
            )

            if isinstance(output, tuple):
                lse = torch.squeeze(output[1], dim=0)
                output = torch.squeeze(output[0], dim=0)
                output = (output, lse)
            else:
                output = torch.squeeze(output, dim=0)

        else:
            output = flash_attn_interface.flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                seqused_q=None,
                seqused_k=None,
                softmax_scale=scale,
                causal=is_causal,
                qv=None,
                q_descale=None, k_descale=None, v_descale=None,
                window_size=(-1, -1),
                attention_chunk=0,
                softcap=0.0,
                num_splits=1,
                pack_gqa=None,
                deterministic=False,
                sm_margin=0,
                return_attn_probs=return_lse,
            )

        lse = None
        if isinstance(output, tuple):
            lse = output[1]
            output = output[0]

        if tensor_layout == "HND":
            output = convert_output_layout(output, src_layout="NHD", dst_layout="HND")

        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)
 
        if return_lse:
            assert lse is not None, "lse is not returned by FlashAttn3"
            return output, lse
        else:
            return output
