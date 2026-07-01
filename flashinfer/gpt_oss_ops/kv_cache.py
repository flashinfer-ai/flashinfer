"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.jit import gen_gpt_oss_reshape_cache_fp8_module
from flashinfer.trace.templates.gpt_oss import gpt_oss_reshape_cache_fp8_trace
from flashinfer.utils import register_custom_op, register_fake_op


@functools.cache
def get_gpt_oss_reshape_cache_fp8_module():
    return gen_gpt_oss_reshape_cache_fp8_module().build_and_load()


@register_custom_op(
    "flashinfer::gpt_oss_reshape_and_cache_fp8",
    mutates_args=("key_cache", "value_cache"),
)
def _reshape_and_cache_fp8_kernel(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    get_gpt_oss_reshape_cache_fp8_module().reshape_and_cache_fp8(
        key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale
    )


@register_fake_op("flashinfer::gpt_oss_reshape_and_cache_fp8")
def _fake_reshape_and_cache_fp8_kernel(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    pass


@flashinfer_api(trace=gpt_oss_reshape_cache_fp8_trace)
def reshape_and_cache_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    r"""Reshape BF16 key/value tensors and scatter them into an FP8 paged KV cache.

    This operation covers the paged-cache update shape used by GPT-OSS decode
    paths: ``key`` and ``value`` have shape ``[num_tokens, num_heads, 64]`` and
    the cache tensors have shape ``[num_blocks, 16, num_heads, 64]``. The cache
    layout must store the token dimension contiguously within each page, i.e.
    ``cache.stride(1)`` is 64. Negative entries in ``slot_mapping`` are skipped.

    This is a specialized SM100+ FP8 E4M3 cache-update helper. Use
    :func:`flashinfer.append_paged_kv_cache` or a framework's generic cache
    update path for other page layouts, head dimensions, or cache dtypes.

    Parameters
    ----------
    key : torch.Tensor
        BF16 key tensor, shape ``[num_tokens, num_heads, 64]``.
    value : torch.Tensor
        BF16 value tensor, shape ``[num_tokens, num_heads, 64]``.
    key_cache : torch.Tensor
        FP8 E4M3 key cache storage, shape ``[num_blocks, 16, num_heads, 64]``.
        ``torch.uint8`` storage is also accepted.
    value_cache : torch.Tensor
        FP8 E4M3 value cache storage, shape ``[num_blocks, 16, num_heads, 64]``.
        ``torch.uint8`` storage is also accepted.
    slot_mapping : torch.Tensor
        Token-to-cache-slot mapping, shape ``[num_tokens]`` and dtype
        ``torch.int64``.
    k_scale : torch.Tensor
        Scalar FP32 key scale.
    v_scale : torch.Tensor
        Scalar FP32 value scale.
    """
    _reshape_and_cache_fp8_kernel(
        key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale
    )
