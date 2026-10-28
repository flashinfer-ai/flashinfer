# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DSA (DeepSeek Sparse Attention) backend of the PrimTS sparse MLA wrapper.

GLM-5.3 / DeepSeek-V3.2 decode: every query token attends to the top-k rows
of an E4M3 ``[.., 576]`` (512 latent + 64 RoPE) KV pool that its indexer
selected.  :class:`~flashinfer.attention.prims_ts.mla_decode.BatchSparseMLADecodePagedTSWrapper`
routes ``plan(kv_lora_rank=512, qk_rope_head_dim=64)`` here; the indices /
lengths / output / LSE contract is the wrapper's.  The kernel is the
``throughput_latency_1cta`` keeps-MMA-AB q64 schedule in its token-sparse
configuration: one CTA per (token, 64 heads), Gather4 row loads, a per-tile
validity bitmask (``-1`` may appear anywhere), and below one wave of CTAs a
split of each token's top-k list (and of the V columns) merged by the 1CTA
GMEM reducer.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

from .kernels.mla_decode.helpers.math import (
    canonicalize_softmax_rescale_threshold_log2,
)
from .kernels.mla_decode.throughput_latency_1cta.sparse_mla_d576_launch import (
    HEAD_DIM,
    V_DIM,
    compile_sparse_mla_d576_kernel,
    sparse_mla_d576_workspace_bytes,
    launch_sparse_mla_d576_kernel,
)

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
_SUPPORTED_HEADS = (64, 128)

# Split-KV: below this many CTAs (query tokens x 64-head groups) most SMs would idle,
# so every token's top-k list is split into S contiguous tile ranges (one CTA each)
# and a standalone reducer merges the partials (``_select_fp8_split``, which
# follows trtllm-gen's MLA generation rule).
SPLIT_KV_MIN_CTAS = 96
SPLIT_KV_CHOICES = (16, 8, 4, 2)


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _v_head_dim_split(total_ctas: int, sm_count: int) -> int:
    """V head-dim split of a split launch of ``total_ctas`` CTAs: 4 / 2 while the
    grid fills at most a quarter / half of the SMs, else 1."""
    if total_ctas * 4 <= sm_count:
        return 4
    if total_ctas * 2 <= sm_count:
        return 2
    return 1


def _select_fp8_split(
    num_ctas: int, sparse_mla_top_k: int, sm_count: int
) -> tuple[int, int]:
    """Return ``(split_kv, head_dim_split)`` for an E4M3 launch of ``num_ctas`` CTAs.

    trtllm-gen's MLA generation rule: split the top-k list so every CTA keeps
    at least two 128-key tiles within one wave (a top-k of one tile pair:
    one tile, if the V split then applies), then, while fewer than half the
    SMs have work, split the 512 V columns over 4 (or 2) CTAs.  Each V
    slice repeats the QK / softmax of its tiles, but a one- or two-tile CTA is
    bound by its fixed latency (Q load, first gather, the single QK-softmax-PV
    chain, the partial-O store), which the slice cuts.  B200, K=2048: B1 is
    8 x 4 = 32 CTAs instead of 16 x 1.
    """
    if num_ctas >= SPLIT_KV_MIN_CTAS:
        return 1, 1
    pairs = -(-sparse_mla_top_k // 256)
    limit = min(pairs, sm_count // num_ctas)
    split = next((s for s in SPLIT_KV_CHOICES if s <= limit), 1)
    if split == 1 and sparse_mla_top_k > 128 and num_ctas * 4 <= sm_count:
        # A single tile pair (top-k <= 256): one tile per CTA, so the V split
        # below can apply.  B200, top-k 256: B1 9.77 -> 7.52 us (trtllm-gen
        # 8.13), B32 9.78 -> 9.35; without the V split (B64) one-tile CTAs
        # are slower (10.6 -> 12.0), hence the quarter-wave bound.
        split = 2
    if split == 1:
        return 1, 1
    return split, _v_head_dim_split(num_ctas * split, sm_count)


def check_plan(
    *,
    num_heads: int,
    max_topk: int,
    max_extra_topk: int,
    q_data_type: torch.dtype,
    has_sinks: bool,
    output_quant_block_size: Optional[int],
) -> None:
    """Reject wrapper options the DSA schedule does not implement."""
    if q_data_type != torch.float8_e4m3fn:
        raise ValueError("D576 sparse MLA (qk_rope_head_dim=64) requires E4M3 Q / KV")
    if num_heads not in _SUPPORTED_HEADS:
        raise ValueError(
            "D576 sparse MLA supports 64 or 128 query heads (64-row tiles), "
            f"got {num_heads}"
        )
    if max_extra_topk:
        raise ValueError("D576 sparse MLA attends to one KV pool (max_extra_topk=0)")
    if has_sinks:
        raise ValueError("D576 sparse MLA does not support attention sinks")
    if output_quant_block_size is not None:
        raise ValueError("D576 sparse MLA does not support output quantization")


def resolve_split(device_index: int, max_rows: int, num_heads: int, max_topk: int):
    """``(split_kv, head_dim_split)`` for a plan of ``max_rows`` query tokens.

    Fixed at plan time, so every run of the plan (and a CUDA graph captured
    from it) uses one compiled kernel.
    """
    return _select_fp8_split(
        max_rows * (num_heads // 64), max_topk, _sm_count(device_index)
    )


def workspace_bytes(max_rows: int, num_heads: int, split_kv: int) -> int:
    """BF16 partial O / FP32 LSE scratch of a split plan (0 without split-KV)."""
    return sparse_mla_d576_workspace_bytes(max_rows, num_heads, split_kv)


def compile_plan(
    device_index: int,
    *,
    num_heads: int,
    max_topk: int,
    softmax_rescale_threshold_log2: Optional[float],
    split_kv: int,
    head_dim_split: int,
):
    """Compile (or load from the PrimTS disk cache) the kernel of one plan."""
    threshold = canonicalize_softmax_rescale_threshold_log2(
        0.0
        if softmax_rescale_threshold_log2 is None
        else softmax_rescale_threshold_log2,
        qkv_dtype="e4m3",
    )
    return compile_sparse_mla_d576_kernel(
        device_index,
        num_heads=num_heads,
        # The kernel uses the capacity only through its 128-key tile count:
        # capacities in one tile bucket share a kernel.
        sparse_mla_top_k=-(-max_topk // 128) * 128,
        softmax_rescale_threshold_log2=threshold,
        split_kv=split_kv,
        head_dim_split=head_dim_split,
    )


def kv_rows(cache: torch.Tensor, *, device, kv_layout: str, name: str) -> torch.Tensor:
    """Flatten an E4M3 ``[N, 576]`` / ``[pages, page_size, 576]`` / one-KV-head
    4-D cache to the ``[rows, 576]`` view the Gather4 maps address."""
    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if cache.ndim == 4:
        axis = 1 if kv_layout == "HND" else 2
        if cache.shape[axis] != 1:
            raise ValueError(f"{name} requires one KV head")
        cache = cache.squeeze(axis)
    if cache.ndim not in (2, 3) or cache.shape[-1] != HEAD_DIM or cache.numel() == 0:
        raise ValueError(
            f"{name} must have shape [N,{HEAD_DIM}] or [pages,page_size,{HEAD_DIM}]"
        )
    if cache.device != device or cache.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{name} must be an E4M3 tensor on the planned device")
    if not cache.is_contiguous() or cache.data_ptr() % 16:
        raise ValueError(
            f"{name} must be contiguous and start at a 16-byte-aligned address"
        )
    rows = cache.view(-1, HEAD_DIM)
    if rows.shape[0] >= 0x7FFFFFFF:
        raise ValueError(f"{name} exceeds the int32 row range")
    return rows


def launch(
    compiled,
    *,
    query: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    output_scale: float,
    workspace: Optional[torch.Tensor],
    split_kv: int,
) -> None:
    """Launch one planned DSA kernel over ``[T, H, 576]`` queries."""
    launch_sparse_mla_d576_kernel(
        compiled,
        query=query,
        kv_pool=kv,
        block_tables=indices,
        top_k_lens=lengths,
        out=out,
        lse=lse,
        bmm1_scale=softmax_scale,
        bmm2_scale=output_scale,
        workspace=workspace,
        split_kv=split_kv,
    )


__all__ = ["KV_LORA_RANK", "QK_ROPE_HEAD_DIM", "V_DIM"]
