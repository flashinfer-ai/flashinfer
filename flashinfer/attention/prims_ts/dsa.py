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

"""Task-scheduled DeepSeek Sparse Attention (DSA) MLA decode over token-level routing.

GLM-5.3 / DeepSeek-V3.2 decode: every packed query token attends to the
``sparse_mla_top_k`` physical KV rows its indexer selected.  The argument
names follow ``trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k > 0)``:
``block_tables`` carries flat token-row indices (``page_id * page_size +
offset`` of the caller's paged cache) with ``-1`` marking an empty slot
anywhere in the row.

:func:`prims_ts_dsa_sparse_mla_fp8` reads an E4M3 ``[.., 576]`` cache with a
1CTA kernel (64 query rows = the 64 heads of one token per CTA, one CTA per
token).  It requires SM100 / SM103.  The kernel trusts the routing values it
is given: the active ``block_tables`` entries (the first
``sparse_mla_top_k_lens[t]`` of row ``t``) must be ``-1`` or in ``[0, N)`` for an
``N``-row pool and ``sparse_mla_top_k_lens`` values must lie in
``[0, sparse_mla_top_k]``; ``FLASHINFER_VALIDATE_INPUTS=1`` adds
device-synchronizing checks of both (not under CUDA-graph capture).  ``query``,
``out`` and ``lse`` must start at 16-byte-aligned addresses.
"""

from __future__ import annotations

import functools
import math
import os
from typing import Optional

import torch

from flashinfer.api_logging import flashinfer_experimental_api
from flashinfer.trace.templates.attention import prims_ts_dsa_sparse_mla_fp8_trace

from .kernels.mla_decode.helpers.math import (
    canonicalize_softmax_rescale_threshold_log2,
)
from .kernels.mla_decode.throughput_latency_1cta.token_sparse_launch import (
    HEAD_DIM as _HEAD_DIM,
    V_DIM as _V_DIM,
    compile_dsa_sparse_kernel,
    dsa_split_workspace_bytes,
    launch_dsa_sparse_kernel,
)

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


def _split_workspace(
    device: torch.device, total_q: int, num_heads: int, split_kv: int
) -> Optional[torch.Tensor]:
    """Partial O / LSE scratch for a split launch (``None`` without split-KV)."""
    if split_kv <= 1:
        return None
    return torch.empty(
        dsa_split_workspace_bytes(total_q, num_heads, split_kv),
        dtype=torch.int8,
        device=device,
    )


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _require_on_device(device: torch.device, **tensors: Optional[torch.Tensor]) -> None:
    for name, tensor in tensors.items():
        _require(
            tensor is None or tensor.device == device,
            f"{name} must be on {device} (the query's device), got {tensor.device if tensor is not None else None}",
        )


def _validate_common_inputs(
    query: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    cum_seq_lens_q: Optional[torch.Tensor],
    *,
    sparse_mla_top_k: int,
    sparse_mla_top_k_lens: Optional[torch.Tensor],
    bmm1_scale: float,
    bmm2_scale: float,
    skip_corr_threshold: float,
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Check the query / routing / output arguments; return ``(top_k_lens, out, lse)``."""

    _require(
        query.is_cuda and query.dtype == torch.float8_e4m3fn and query.ndim == 3,
        "query must be a CUDA E4M3 tensor of shape [T, H, 576]",
    )
    total_q, num_heads, head_dim = query.shape
    _require(
        head_dim == _HEAD_DIM and num_heads in _SUPPORTED_HEADS,
        f"query must be [T, 64|128, {_HEAD_DIM}], got {tuple(query.shape)}",
    )
    _require(query.is_contiguous(), "query must be contiguous")
    _require(
        isinstance(sparse_mla_top_k, int)
        and sparse_mla_top_k >= 4
        and sparse_mla_top_k % 4 == 0,
        "sparse_mla_top_k must be a positive multiple of 4",
    )
    _require(
        block_tables.is_cuda
        and block_tables.dtype == torch.int32
        and block_tables.ndim == 2
        and tuple(block_tables.shape) == (total_q, sparse_mla_top_k)
        and block_tables.is_contiguous(),
        "block_tables must be contiguous INT32 [T, sparse_mla_top_k]",
    )
    device = query.device
    for name, tensor in (("seq_lens", seq_lens), ("cum_seq_lens_q", cum_seq_lens_q)):
        if tensor is not None:
            _require(
                tensor.is_cuda and tensor.dtype == torch.int32 and tensor.ndim == 1,
                f"{name} must be a 1-D CUDA INT32 tensor when given",
            )
    if sparse_mla_top_k_lens is None:
        top_k_lens = torch.full(
            (total_q,), sparse_mla_top_k, dtype=torch.int32, device=device
        )
    else:
        _require(
            sparse_mla_top_k_lens.is_cuda
            and sparse_mla_top_k_lens.dtype == torch.int32
            and tuple(sparse_mla_top_k_lens.shape) == (total_q,)
            and sparse_mla_top_k_lens.is_contiguous(),
            "sparse_mla_top_k_lens must be contiguous INT32 [T]",
        )
        top_k_lens = sparse_mla_top_k_lens
    for scale, name in (
        (bmm1_scale, "bmm1_scale"),
        (bmm2_scale, "bmm2_scale"),
        (skip_corr_threshold, "skip_corr_threshold"),
    ):
        _require(
            isinstance(scale, (int, float))
            and not isinstance(scale, bool)
            and math.isfinite(float(scale)),
            f"{name} must be a finite number",
        )
    # E4M3 P: 1.75 * 2**8 == 448 is the largest row-max shift the frozen
    # maximum may lag behind.
    _require(
        0.0 <= float(skip_corr_threshold) <= 8.0,
        "skip_corr_threshold must be in [0, 8] (log2 units)",
    )
    if out is None:
        out = torch.empty(
            (total_q, num_heads, _V_DIM), dtype=torch.bfloat16, device=device
        )
    else:
        _require(
            out.is_cuda
            and out.dtype == torch.bfloat16
            and tuple(out.shape) == (total_q, num_heads, _V_DIM)
            and out.is_contiguous(),
            "out must be contiguous BF16 [T, H, 512]",
        )
    if lse is None:
        lse = torch.empty((total_q, num_heads), dtype=torch.float32, device=device)
    else:
        _require(
            lse.is_cuda
            and lse.dtype == torch.float32
            and tuple(lse.shape) == (total_q, num_heads)
            and lse.is_contiguous(),
            "lse must be contiguous FP32 [T, H]",
        )
    # Q is loaded by TMA and O / LSE (and the split partials) are stored with
    # 16-byte vector accesses.
    for name, tensor in (("query", query), ("out", out), ("lse", lse)):
        _require(
            tensor.data_ptr() % 16 == 0,
            f"{name} must start at a 16-byte-aligned address",
        )
    _require_on_device(
        device,
        block_tables=block_tables,
        seq_lens=seq_lens,
        cum_seq_lens_q=cum_seq_lens_q,
        sparse_mla_top_k_lens=sparse_mla_top_k_lens,
        out=out,
        lse=lse,
    )
    return top_k_lens, out, lse


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.lru_cache(maxsize=None)
def _require_supported_device(device_index: int) -> None:
    """Raise once per device if it is not SM100 or SM103 (cached per device)."""
    capability = torch.cuda.get_device_capability(device_index)
    if capability not in ((10, 0), (10, 3)):
        raise NotImplementedError(
            f"DSA sparse MLA TS requires SM100 or SM103, got "
            f"SM{capability[0]}{capability[1]}"
        )


def _validate_inputs_enabled() -> bool:
    """Return whether ``FLASHINFER_VALIDATE_INPUTS`` asks for device-synchronizing value checks."""
    if os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") in ("", "0"):
        return False
    # The checks read routing values on the host; like the MLA wrapper, skip
    # them during CUDA graph capture (validate eagerly before capturing).
    return not torch.cuda.is_current_stream_capturing()


def _validate_route_values(
    block_tables: torch.Tensor,
    top_k_lens: torch.Tensor,
    *,
    pool_rows: int,
    sparse_mla_top_k: int,
) -> None:
    """Check the values the kernel trusts (``FLASHINFER_VALIDATE_INPUTS=1`` only; synchronizes).

    Only the active prefix of every ``block_tables`` row (``top_k_lens``) is
    checked: the kernel never reads the slots past it.
    """
    if top_k_lens.numel():
        lo, hi = int(top_k_lens.min()), int(top_k_lens.max())
        _require(
            lo >= 0 and hi <= sparse_mla_top_k,
            f"sparse_mla_top_k_lens must be in [0, {sparse_mla_top_k}], got [{lo}, {hi}]",
        )
    if block_tables.numel():
        active = (
            torch.arange(block_tables.shape[1], device=block_tables.device)[None, :]
            < top_k_lens[:, None]
        )
        routes = block_tables[active]
        if routes.numel():
            lo, hi = int(routes.min()), int(routes.max())
            _require(
                lo >= -1 and hi < pool_rows,
                f"block_tables entries must be -1 or in [0, {pool_rows}), got [{lo}, {hi}]",
            )


@flashinfer_experimental_api(trace=prims_ts_dsa_sparse_mla_fp8_trace)
def prims_ts_dsa_sparse_mla_fp8(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    cum_seq_lens_q: Optional[torch.Tensor],
    *,
    max_q_len: int = 1,
    sparse_mla_top_k: int,
    sparse_mla_top_k_lens: Optional[torch.Tensor] = None,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    skip_corr_threshold: float = 0.0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""DSA sparse MLA decode with an E4M3 ``[.., 576]`` KV cache (PrimTS 1CTA, 64 rows per CTA).

    Parameters
    ----------
    query : ``[T, H, 576]`` E4M3 packed queries (H in {64, 128}; 128 heads run as
        two 64-row tiles of the same token).
    kv_cache : E4M3 ``[num_pages, page_size, 576]`` / ``[num_pages, 1, page_size, 576]``
        / ``[N, 576]`` contiguous cache, addressed as flat rows.
    block_tables : INT32 ``[T, sparse_mla_top_k]`` flat row indices: ``-1`` = empty
        slot (zero-filled), every other active entry must address a pool row;
        slots past ``sparse_mla_top_k_lens[t]`` are never read.
    seq_lens, cum_seq_lens_q : accepted for interface parity with trtllm-gen; the
        kernel does not read them (causality is encoded by the indexer).
    max_q_len : accepted for interface parity; the kernel is independent of it.
    sparse_mla_top_k : routing slots per query token (positive multiple of 4).
    sparse_mla_top_k_lens : optional INT32 ``[T]`` active prefix per token, values in
        ``[0, sparse_mla_top_k]``; ``None`` scans the full width.
    bmm1_scale, bmm2_scale : fused QK / PV scales (``q_scale * k_scale * sm_scale``,
        ``v_scale * o_scale``).
    skip_corr_threshold : skip-correction threshold in log2 units, in ``[0, 8]``;
        ``0.0`` rescales the running sum exactly.  The value is compiled into
        the kernel (``softmax_rescale_threshold_log2``): each distinct
        threshold is its own specialization, and E4M3 P uses the scale
        ``448 * 2**-threshold``.
    out, lse : optional caller-owned BF16 ``[T, H, 512]`` / FP32 ``[T, H]`` (base-2 LSE).

    Returns
    -------
    ``out`` (BF16 ``[T, H, 512]``).  Fully masked tokens produce zeros and an LSE of ``-inf``.
    """

    del max_q_len
    top_k_lens, out, lse = _validate_common_inputs(
        query,
        block_tables,
        seq_lens,
        cum_seq_lens_q,
        sparse_mla_top_k=sparse_mla_top_k,
        sparse_mla_top_k_lens=sparse_mla_top_k_lens,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        skip_corr_threshold=skip_corr_threshold,
        out=out,
        lse=lse,
    )
    _require(
        kv_cache.is_cuda
        and kv_cache.dtype == torch.float8_e4m3fn
        and kv_cache.is_contiguous()
        and kv_cache.shape[-1] == _HEAD_DIM,
        f"kv_cache must be a contiguous CUDA E4M3 tensor with last dim {_HEAD_DIM}",
    )
    _require(
        kv_cache.data_ptr() % 16 == 0,
        "kv_cache must start at a 16-byte-aligned address",
    )
    _require_on_device(query.device, kv_cache=kv_cache)
    kv_pool = kv_cache.view(-1, _HEAD_DIM)
    device_index = _device_index(query.device)
    _require_supported_device(device_index)
    if query.shape[0] == 0:
        return out
    if _validate_inputs_enabled():
        _validate_route_values(
            block_tables,
            top_k_lens,
            pool_rows=kv_pool.shape[0],
            sparse_mla_top_k=sparse_mla_top_k,
        )
    num_heads = query.shape[1]
    num_ctas = query.shape[0] * (num_heads // 64)
    split_kv, head_dim_split = _select_fp8_split(
        num_ctas, sparse_mla_top_k, _sm_count(device_index)
    )
    compiled = compile_dsa_sparse_kernel(
        device_index,
        num_heads=num_heads,
        # Every top-k up to one 128-key tile runs the same one-tile kernel.
        sparse_mla_top_k=max(sparse_mla_top_k, 128),
        # Compiled into the kernel (one specialization per threshold).
        softmax_rescale_threshold_log2=canonicalize_softmax_rescale_threshold_log2(
            skip_corr_threshold, qkv_dtype="e4m3"
        ),
        split_kv=split_kv,
        head_dim_split=head_dim_split,
    )
    launch_dsa_sparse_kernel(
        compiled,
        query=query,
        kv_pool=kv_pool,
        block_tables=block_tables,
        top_k_lens=top_k_lens,
        out=out,
        lse=lse,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        workspace=_split_workspace(query.device, query.shape[0], num_heads, split_kv),
        split_kv=split_kv,
    )
    return out


__all__ = ["prims_ts_dsa_sparse_mla_fp8"]
