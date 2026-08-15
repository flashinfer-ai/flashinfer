"""Cake FMHA routing and validation for DCP speculative decode.

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

from __future__ import annotations

import math
from typing import Optional

import torch

from .utils import (
    _check_workspace_buffer_alignment,
    check_shape_dtype_device,
    get_compute_capability,
    get_device_sm_count,
)

_BLOCK_N = 128
_HEAD_DIM = 128
_BF16_PAGE_SIZE = 16
_FP8_PAGE_SIZE = 64
_MAX_NUM_SPLIT = 16
_MIN_SPLIT_LOCAL_BLOCKS = 16
_TARGET_PAIRS_PER_SPLIT = 3
_RETAIN_KV_L2_MAX_BLOCKS = 9
_FP8_MAX_NUM_SPLIT = 4
_FP8_MIN_SPLIT_LOCAL_BLOCKS = 4
_FP8_RETAIN_KV_L2_MAX_BLOCKS = 18
_BF16_SUPPORTED_Q_LENS = (1, 2, 3, 4, 5, 6, 8)
_FP8_SUPPORTED_Q_LENS = (1, 2, 3, 4, 5, 6, 8)
_SUPPORTED_CP_WORLDS = (1, 2, 4, 8)


def get_dcp_spec_workspace_size_bytes(
    batch_size: int,
    q_len_per_req: int,
    num_qo_heads: int,
    num_split: int = _MAX_NUM_SPLIT,
) -> int:
    """Bytes for Cake FMHA Split-KV BF16 partial-O and FP32 partial-LSE scratch."""

    if min(batch_size, q_len_per_req, num_qo_heads) <= 0:
        raise ValueError("batch_size, q_len_per_req, and num_qo_heads must be positive")
    if not 2 <= num_split <= _MAX_NUM_SPLIT:
        raise ValueError(f"num_split must be in [2, {_MAX_NUM_SPLIT}]")
    partial_rows = batch_size * q_len_per_req * num_qo_heads * num_split
    return partial_rows * (_HEAD_DIM * 2 + 4)


def get_dcp_spec_counter_bytes(
    batch_size: int,
    q_len_per_req: int,
    num_kv_heads: int,
) -> int:
    """Bytes for the v4 completion tickets, zeroed once then self-reset."""

    if min(batch_size, q_len_per_req, num_kv_heads) <= 0:
        raise ValueError("batch_size, q_len_per_req, and num_kv_heads must be positive")
    return batch_size * q_len_per_req * num_kv_heads * 4


def _split_workspace_views(
    *,
    workspace_buffer: torch.Tensor,
    completion_buffer: Optional[torch.Tensor],
    device: torch.device,
    batch_size: int,
    q_len_per_req: int,
    num_qo_heads: int,
    num_kv_heads: int,
    num_split: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bind caller-owned Split-KV scratch without allocating during launch."""

    required_counter = get_dcp_spec_counter_bytes(
        batch_size, q_len_per_req, num_kv_heads
    )
    if completion_buffer is None:
        raise ValueError(
            "multi_ctas_kv_counter_buffer is required for the DCP Split-KV route; "
            f"pass a zero-initialized reusable CUDA buffer with at least {required_counter} bytes"
        )
    if completion_buffer.device != device or not completion_buffer.is_contiguous():
        raise ValueError(
            "multi_ctas_kv_counter_buffer must be contiguous and on the query device"
        )
    _check_workspace_buffer_alignment(completion_buffer, "multi_ctas_kv_counter_buffer")
    completion_u8 = completion_buffer.view(torch.uint8).reshape(-1)
    if completion_u8.numel() < required_counter:
        raise ValueError(
            "multi_ctas_kv_counter_buffer is too small for DCP Split-KV: "
            f"got {completion_u8.numel()} bytes, need {required_counter}"
        )
    split_completion = completion_u8[:required_counter].view(torch.int32)

    if workspace_buffer.device != device or not workspace_buffer.is_contiguous():
        raise ValueError("workspace_buffer must be contiguous and on the query device")
    _check_workspace_buffer_alignment(workspace_buffer, "workspace_buffer")
    required_workspace = get_dcp_spec_workspace_size_bytes(
        batch_size,
        q_len_per_req,
        num_qo_heads,
        num_split,
    )
    workspace_u8 = workspace_buffer.view(torch.uint8).reshape(-1)
    if workspace_u8.numel() < required_workspace:
        raise ValueError(
            "workspace_buffer is too small for DCP Split-KV: "
            f"got {workspace_u8.numel()} bytes, need {required_workspace}"
        )
    partial_rows = batch_size * q_len_per_req * num_qo_heads * num_split
    partial_o_bytes = partial_rows * _HEAD_DIM * 2
    partial_lse_bytes = partial_rows * 4
    partial_o = workspace_u8[:partial_o_bytes].view(torch.bfloat16)
    partial_lse = workspace_u8[
        partial_o_bytes : partial_o_bytes + partial_lse_bytes
    ].view(torch.float32)
    return partial_o, partial_lse, split_completion


def _select_num_split(
    *,
    logical_tiles: int,
    sm_count: int,
    local_blocks: int,
) -> int:
    if local_blocks < _MIN_SPLIT_LOCAL_BLOCKS:
        return 1
    total_pairs = (local_blocks + 1) // 2
    work_cap = (total_pairs + _TARGET_PAIRS_PER_SPLIT - 1) // _TARGET_PAIRS_PER_SPLIT
    num_split = min(
        _MAX_NUM_SPLIT,
        total_pairs,
        work_cap,
        sm_count // logical_tiles,
    )
    return num_split if num_split >= 2 else 1


def _select_fp8_num_split(
    *,
    logical_tiles: int,
    sm_count: int,
    local_blocks: int,
    cp_world: int,
) -> int:
    """Fill one SM wave while retaining two FP8 K/V block pairs per CTA."""

    if logical_tiles >= sm_count or local_blocks < _FP8_MIN_SPLIT_LOCAL_BLOCKS:
        return 1
    total_pairs = (local_blocks + 1) // 2
    work_cap = (total_pairs + 1) // 2
    max_num_split = 3 if cp_world > 1 else _FP8_MAX_NUM_SPLIT
    num_split = min(
        max_num_split,
        total_pairs,
        work_cap,
        sm_count // logical_tiles,
    )
    return num_split if num_split >= 2 else 1


def _is_cuda_version_at_least(version: str) -> bool:
    from .jit.cpp_ext import is_cuda_version_at_least

    return is_cuda_version_at_least(version)


def _select_target(device: torch.device) -> str:
    capability = get_compute_capability(device)
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError(
            "DCP speculative FMHA requires compute capability 10.0 "
            f"(B200/GB200) or 10.3 (B300/GB300), got {capability[0]}.{capability[1]}"
        )
    # Base and add-on sources use the same exact product architecture names.
    if capability == (10, 0):
        if _is_cuda_version_at_least("12.8"):
            return "sm100a"
        raise RuntimeError(
            "DCP speculative FMHA on compute capability 10.0 requires CUDA "
            "12.8 or newer"
        )
    if _is_cuda_version_at_least("12.9"):
        return "sm103a"
    if capability == (10, 3):
        raise RuntimeError(
            "DCP speculative FMHA on compute capability 10.3 requires CUDA 12.9 "
            "or newer for the sm_103a exact target"
        )
    raise AssertionError(f"unreachable DCP target capability: {capability}")


def _validate_core_inputs(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    causal_seqlens_kv_global: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    batch_size: int,
    q_len_per_req: int,
    cp_world: int,
    cp_rank: int,
) -> tuple[int, int, str, int]:
    supported_q_lens: tuple[int, ...]
    if query.dtype != torch.bfloat16:
        raise TypeError("DCP speculative FMHA requires a BF16 query tensor")
    if k_cache.dtype != v_cache.dtype:
        raise TypeError("DCP speculative FMHA key and value dtypes must match")
    if k_cache.dtype == torch.bfloat16:
        profile = "bf16_p16"
        page_size_required = _BF16_PAGE_SIZE
        supported_q_lens = _BF16_SUPPORTED_Q_LENS
    elif k_cache.dtype == torch.float8_e4m3fn:
        profile = "fp8_p64"
        page_size_required = _FP8_PAGE_SIZE
        supported_q_lens = _FP8_SUPPORTED_Q_LENS
    else:
        raise TypeError(
            "DCP speculative FMHA requires BF16 or float8_e4m3fn key/value tensors"
        )
    if query.ndim != 3:
        raise ValueError(
            "DCP speculative FMHA query must have shape "
            "[batch_size * q_len_per_req, num_qo_heads, 128]"
        )
    num_tokens, num_qo_heads, head_dim = query.shape
    if num_tokens != batch_size * q_len_per_req or head_dim != _HEAD_DIM:
        raise ValueError(
            "query shape does not match the DCP specialization: "
            f"got {tuple(query.shape)}, expected tokens={batch_size * q_len_per_req} "
            f"and head_dim={_HEAD_DIM}"
        )
    if q_len_per_req not in supported_q_lens:
        raise ValueError(
            f"q_len_per_req must be one of {supported_q_lens} for the {profile} profile"
        )
    if cp_world not in _SUPPORTED_CP_WORLDS:
        raise ValueError(f"cp_world must be one of {_SUPPORTED_CP_WORLDS}")
    if not 0 <= cp_rank < cp_world:
        raise ValueError(f"cp_rank must be in [0, {cp_world}), got {cp_rank}")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "DCP speculative FMHA HND caches must have shape "
            "[num_pages, num_kv_heads, page_size, 128]"
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError("key and value cache shapes must match")
    _, num_kv_heads, page_size, kv_head_dim = k_cache.shape
    if page_size != page_size_required or kv_head_dim != _HEAD_DIM:
        raise ValueError(
            f"DCP speculative FMHA {profile} requires HND "
            f"page_size={page_size_required} and head_dim=128, "
            f"got page_size={page_size}, head_dim={kv_head_dim}"
        )
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    group_ratio = num_qo_heads // num_kv_heads
    if not 1 <= group_ratio <= 8:
        raise ValueError(f"DCP head group ratio must be in [1, 8], got {group_ratio}")

    device = query.device
    for name, tensor in (
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("block_tables", block_tables),
        ("seq_lens", seq_lens),
        ("causal_seqlens_kv_global", causal_seqlens_kv_global),
        ("out", out),
        ("lse", lse),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on the same CUDA device as query")
    check_shape_dtype_device(out, query.shape, torch.bfloat16, device, "out")
    check_shape_dtype_device(
        lse,
        (num_tokens, num_qo_heads),
        torch.float32,
        device,
        "lse",
    )
    check_shape_dtype_device(
        causal_seqlens_kv_global,
        (batch_size,),
        torch.int32,
        device,
        "causal_seqlens_kv_global",
    )
    if block_tables.ndim != 2 or not block_tables.is_contiguous():
        raise ValueError("block_tables must be a contiguous 2D tensor")
    if block_tables.dtype != torch.int32 or block_tables.shape[0] != batch_size:
        raise ValueError(
            "block_tables must be int32 with shape [batch_size, max_pages_per_seq]"
        )
    if block_tables.shape[1] <= 0:
        raise ValueError("block_tables must contain at least one physical page slot")
    if (
        seq_lens.dtype != torch.int32
        or seq_lens.ndim != 1
        or seq_lens.shape[0] != batch_size
        or not seq_lens.is_contiguous()
    ):
        raise ValueError(
            "seq_lens must be a contiguous int32 tensor with shape [batch_size]"
        )
    if not causal_seqlens_kv_global.is_contiguous():
        raise ValueError("causal_seqlens_kv_global must be contiguous")
    return num_qo_heads, num_kv_heads, profile, page_size_required


def run_dcp_spec_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    causal_seqlens_kv_global: torch.Tensor,
    max_local_seq_len: int,
    bmm1_scale: float,
    bmm2_scale: float,
    cp_world: int,
    cp_rank: int,
    q_len_per_req: int,
    out: torch.Tensor,
    lse: torch.Tensor,
    completion_buffer: Optional[torch.Tensor],
) -> None:
    """Run one rank-local Cake FMHA DCP speculative specialization."""

    if q_len_per_req <= 0 or query.shape[0] % q_len_per_req != 0:
        raise ValueError("query token count must be divisible by q_len_per_req")
    batch_size = query.shape[0] // q_len_per_req
    num_qo_heads, num_kv_heads, profile, page_size = _validate_core_inputs(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        causal_seqlens_kv_global,
        out,
        lse,
        batch_size=batch_size,
        q_len_per_req=q_len_per_req,
        cp_world=cp_world,
        cp_rank=cp_rank,
    )
    if max_local_seq_len < 0:
        raise ValueError(
            f"max_local_seq_len must be nonnegative, got {max_local_seq_len}"
        )
    local_capacity = block_tables.shape[1] * page_size
    if max_local_seq_len > local_capacity:
        raise ValueError(
            "max_local_seq_len exceeds the rank-local page-table capacity: "
            f"got {max_local_seq_len}, capacity={local_capacity}"
        )

    if not math.isfinite(float(bmm1_scale)) or not math.isfinite(float(bmm2_scale)):
        raise ValueError(
            "DCP speculative FMHA bmm1_scale and bmm2_scale must be finite"
        )

    sm_count = get_device_sm_count(query.device)
    logical_tiles = batch_size * q_len_per_req * num_kv_heads
    target = _select_target(query.device)
    softmax_scale_log2 = float(bmm1_scale) / math.log(2.0)
    max_pages_per_seq = block_tables.shape[1]

    if profile == "fp8_p64":
        from .jit.dcp import load_dcp_spec_fp8_module

        local_blocks = max(1, (max_local_seq_len + _BLOCK_N - 1) // _BLOCK_N)
        num_split = _select_fp8_num_split(
            logical_tiles=logical_tiles,
            sm_count=sm_count,
            local_blocks=local_blocks,
            cp_world=cp_world,
        )
        retain_kv_l2 = int(
            cp_world > 1 and local_blocks <= _FP8_RETAIN_KV_L2_MAX_BLOCKS
        )
        module = load_dcp_spec_fp8_module(
            target,
            batch_size,
            q_len_per_req,
            num_qo_heads,
            num_kv_heads,
            cp_world,
            num_split,
            retain_kv_l2,
        )
        if num_split == 1:
            partial_o = out
            partial_lse = lse
            split_completion = seq_lens
        else:
            partial_o, partial_lse, split_completion = _split_workspace_views(
                workspace_buffer=workspace_buffer,
                completion_buffer=completion_buffer,
                device=query.device,
                batch_size=batch_size,
                q_len_per_req=q_len_per_req,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                num_split=num_split,
            )
        grid = min(sm_count, logical_tiles * num_split)
        module.run(
            query,
            k_cache.view(torch.uint8),
            v_cache.view(torch.uint8),
            partial_o,
            partial_lse,
            out,
            lse,
            split_completion,
            block_tables,
            seq_lens,
            causal_seqlens_kv_global,
            max_pages_per_seq,
            max_local_seq_len,
            softmax_scale_log2,
            float(bmm2_scale),
            cp_rank,
            num_qo_heads,
            num_kv_heads,
            batch_size,
            grid,
            1,
            1,
        )
        return

    if float(bmm2_scale) != 1.0:
        raise ValueError("the BF16/page16 DCP profile requires bmm2_scale=1.0")

    local_blocks = max(1, (max_local_seq_len + _BLOCK_N - 1) // _BLOCK_N)
    num_split = _select_num_split(
        logical_tiles=logical_tiles,
        sm_count=sm_count,
        local_blocks=local_blocks,
    )
    from .jit.dcp import load_dcp_spec_module

    if num_split == 1:
        retain_kv_l2 = int(local_blocks <= _RETAIN_KV_L2_MAX_BLOCKS)
        module = load_dcp_spec_module(
            "v1",
            target,
            batch_size,
            q_len_per_req,
            num_qo_heads,
            num_kv_heads,
            cp_world,
            retain_kv_l2,
        )
        grid = min(sm_count, logical_tiles)
        module.run(
            query,
            k_cache,
            v_cache,
            out,
            lse,
            block_tables,
            causal_seqlens_kv_global,
            max_pages_per_seq,
            max_local_seq_len,
            softmax_scale_log2,
            cp_rank,
            num_qo_heads,
            num_kv_heads,
            batch_size,
            grid,
            1,
            1,
        )
        return

    partial_o, partial_lse, split_completion = _split_workspace_views(
        workspace_buffer=workspace_buffer,
        completion_buffer=completion_buffer,
        device=query.device,
        batch_size=batch_size,
        q_len_per_req=q_len_per_req,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        num_split=num_split,
    )

    module = load_dcp_spec_module(
        "v4",
        target,
        batch_size,
        q_len_per_req,
        num_qo_heads,
        num_kv_heads,
        cp_world,
        num_split,
    )
    grid = min(sm_count, logical_tiles * num_split)
    module.run(
        query,
        k_cache,
        v_cache,
        partial_o,
        partial_lse,
        out,
        lse,
        split_completion,
        block_tables,
        causal_seqlens_kv_global,
        max_pages_per_seq,
        max_local_seq_len,
        softmax_scale_log2,
        cp_rank,
        num_qo_heads,
        num_kv_heads,
        batch_size,
        grid,
        1,
        1,
    )


__all__ = [
    "get_dcp_spec_counter_bytes",
    "get_dcp_spec_workspace_size_bytes",
]
