import logging

import math
import torch

from .utils import (
    convert_output_layout,
    convert_qkv_layout,
)

logger = logging.getLogger(__name__)

try:
    import flash_attn_interface
except ImportError:
    flash_attn_interface = None

from flashinfer.prefill import fmha_varlen


class AttentionOpManager:
    _attn_registry: dict[str, type] = {}

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
        is_causal=False,
        return_lse=False,
        tensor_layout="HND",
        cur_rank_cu_seqlens_q=None,
        cur_rank_cu_seqlens_k=None,
        cur_rank_max_seqlen_q=0,
        cur_rank_max_seqlen_k=0,
        **kwargs,
    ):
        if flash_attn_interface is None:
            raise ImportError("FlashAttn3 is not installed")

        if tensor_layout not in ["HND", "NHD"]:
            raise NotImplementedError("Tensor layout not supported for FlashAttn3")

        if tensor_layout == "HND":
            query, key, value = convert_qkv_layout(
                query, key, value, src_layout="HND", dst_layout="NHD"
            )

        if attn_mask is not None:
            raise NotImplementedError("FlashAttn3 does not support attn_mask yet")

        # FA3 only supports float16 and bfloat16
        origin_dtype = query.dtype
        if query.dtype not in [torch.float16, torch.bfloat16]:
            query = query.to(torch.float16)
            key = key.to(torch.float16)
            value = value.to(torch.float16)

        if cur_rank_cu_seqlens_q is None:
            query = torch.unsqueeze(query, dim=0)
            key = torch.unsqueeze(key, dim=0)
            value = torch.unsqueeze(value, dim=0)
            output = flash_attn_interface.flash_attn_func(
                q=query,
                k=key,
                v=value,
                softmax_scale=None,
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
                cu_seqlens_q=cur_rank_cu_seqlens_q,
                cu_seqlens_k=cur_rank_cu_seqlens_k,
                max_seqlen_q=cur_rank_max_seqlen_q,
                max_seqlen_k=cur_rank_max_seqlen_k,
                seqused_q=None,
                seqused_k=None,
                softmax_scale=None,
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

        lse = None
        if isinstance(output, tuple):
            lse = output[1]
            output = output[0]

        if tensor_layout == "HND":
            output = convert_output_layout(output, src_layout="NHD", dst_layout="HND")

        if tensor_layout == "NHD" and lse is not None:
            lse = lse.permute(1, 0)

        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)

        if return_lse:
            assert lse is not None, "lse is not returned by FlashAttn3"
            return output, lse
        else:
            return output


@AttentionOpManager.register_attn("cutlass")
class CutlassFmha:
    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        is_causal=False,
        return_lse=False,
        tensor_layout="HND",
        cur_rank_cu_seqlens_q=None,
        cur_rank_cu_seqlens_k=None,
        cur_rank_max_seqlen_q=0,
        cur_rank_max_seqlen_k=0,
        **kwargs,
    ):
        if tensor_layout not in ["HND", "NHD"]:
            raise NotImplementedError("Tensor layout not supported for CutlassFmha")

        if tensor_layout == "HND":
            query, key, value = convert_qkv_layout(
                query, key, value, src_layout="HND", dst_layout="NHD"
            )

        if attn_mask is not None:
            raise NotImplementedError("CutlassFmha does not support attn_mask yet")

        # CutlassFmha only supports float16 and bfloat16
        origin_dtype = query.dtype
        if query.dtype not in [torch.float16, torch.bfloat16]:
            query = query.to(torch.float16)
            key = key.to(torch.float16)
            value = value.to(torch.float16)

        if cur_rank_cu_seqlens_q is None:
            qo_segment_offsets = torch.tensor(
                [0, query.shape[0]], device=query.device, dtype=torch.int32
            )
            kv_segment_offsets = torch.tensor(
                [0, key.shape[0]], device=key.device, dtype=torch.int32
            )
            max_qo_len = query.shape[0]
        else:
            qo_segment_offsets = cur_rank_cu_seqlens_q
            kv_segment_offsets = cur_rank_cu_seqlens_k
            max_qo_len = cur_rank_max_seqlen_q

        output = fmha_varlen(
            query,
            key,
            value,
            qo_segment_offsets=qo_segment_offsets,
            kv_segment_offsets=kv_segment_offsets,
            max_qo_len=max_qo_len,
            causal=is_causal,
            sm_scale=1.0 / math.sqrt(query.size(-1)),
            return_lse=return_lse,
        )

        lse = None
        if isinstance(output, tuple):
            lse = output[1]
            output = output[0]

        if tensor_layout == "HND":
            output = convert_output_layout(output, src_layout="NHD", dst_layout="HND")
            if lse is not None:
                lse = lse.permute(1, 0)

        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)

        if return_lse:
            assert lse is not None, "lse is not returned by cutlass fmha"
            return output, lse
        else:
            return output
