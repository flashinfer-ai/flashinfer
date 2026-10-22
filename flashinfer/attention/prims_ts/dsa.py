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

Two KV-cache formats share one 1CTA kernel (64 query rows = the 64 heads of one
token per CTA, one CTA per token):

* :func:`prims_ts_dsa_sparse_mla_fp8` reads an E4M3 ``[.., 576]`` cache.
* :func:`prims_ts_dsa_sparse_mla_nvfp4` reads the TRT-LLM NVFP4 MLA cache
  (``quantizeAndWriteMlaFp4``: 320 data bytes + 40 E4M3 block-scale bytes per
  token row).  QK runs natively on the E2M1 rows (``tcgen05.mma.kind::mxf4nvf4``
  with Q quantized to NVFP4 on chip); V is dequantized on chip to E4M3 for the
  FP8 PV MMA.  Needs CUDA 13.4+ / nvidia-cutlass-dsl 4.8+ (the PTX ISA 9.4
  packed E2M1 x E4M3 multiply of the V dequantization).

Both entries require SM100 / SM103.  The kernel trusts the routing values it
is given: ``block_tables`` entries must be ``-1`` or in ``[0, N)`` for an
``N``-row pool and ``sparse_mla_top_k_lens`` values must lie in
``[0, sparse_mla_top_k]``; ``FLASHINFER_VALIDATE_INPUTS=1`` adds
device-synchronizing checks of both (not under CUDA-graph capture).
"""

from __future__ import annotations

import functools
import math
import os
from typing import Optional

import torch

from flashinfer.api_logging import flashinfer_experimental_api
from flashinfer.trace.templates.attention import (
    prims_ts_dsa_sparse_mla_fp8_trace,
    prims_ts_dsa_sparse_mla_nvfp4_trace,
)

from .kernels.mla_decode.throughput_latency_1cta.token_sparse_launch import (
    HEAD_DIM as _HEAD_DIM,
    V_DIM as _V_DIM,
    compile_dsa_sparse_kernel,
    launch_dsa_sparse_kernel,
)

_SUPPORTED_HEADS = (64, 128)
_NVFP4_RESIDUAL_DIM = 64


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


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
    """Check the arguments both entry points share; return ``(top_k_lens, out, lse)``."""

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
    # maximum may lag behind (see ``max_skip_corr_threshold``).
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
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FLASHINFER_VALIDATE_INPUTS reads routing values on the host and "
            "cannot run during CUDA graph capture; validate the inputs eagerly "
            "before capture or unset the variable"
        )
    return True


def _validate_route_values(
    block_tables: torch.Tensor,
    top_k_lens: torch.Tensor,
    *,
    pool_rows: int,
    sparse_mla_top_k: int,
) -> None:
    """Check the values the kernel trusts (``FLASHINFER_VALIDATE_INPUTS=1`` only; synchronizes)."""
    if block_tables.numel():
        lo, hi = int(block_tables.min()), int(block_tables.max())
        _require(
            lo >= -1 and hi < pool_rows,
            f"block_tables entries must be -1 or in [0, {pool_rows}), got [{lo}, {hi}]",
        )
    if top_k_lens.numel():
        lo, hi = int(top_k_lens.min()), int(top_k_lens.max())
        _require(
            lo >= 0 and hi <= sparse_mla_top_k,
            f"sparse_mla_top_k_lens must be in [0, {sparse_mla_top_k}], got [{lo}, {hi}]",
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
        slot (zero-filled), every other entry must address a pool row.
    seq_lens, cum_seq_lens_q : accepted for interface parity with trtllm-gen; the
        kernel does not read them (causality is encoded by the indexer).
    max_q_len : accepted for interface parity; the kernel is independent of it.
    sparse_mla_top_k : routing slots per query token (positive multiple of 4).
    sparse_mla_top_k_lens : optional INT32 ``[T]`` active prefix per token, values in
        ``[0, sparse_mla_top_k]``; ``None`` scans the full width.
    bmm1_scale, bmm2_scale : fused QK / PV scales (``q_scale * k_scale * sm_scale``,
        ``v_scale * o_scale``).
    skip_corr_threshold : skip-correction threshold in log2 units, in ``[0, 8]``;
        ``0.0`` rescales the running sum exactly.
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
    kv_pool = kv_cache.view(-1, _HEAD_DIM)
    device_index = _device_index(query.device)
    _require_supported_device(device_index)
    if _validate_inputs_enabled():
        _validate_route_values(
            block_tables,
            top_k_lens,
            pool_rows=kv_pool.shape[0],
            sparse_mla_top_k=sparse_mla_top_k,
        )
    num_heads = query.shape[1]
    compiled = compile_dsa_sparse_kernel(
        device_index,
        num_heads=num_heads,
        sparse_mla_top_k=sparse_mla_top_k,
        enable_skip_correction=skip_corr_threshold > 0.0,
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
        skip_corr_threshold=skip_corr_threshold,
    )
    return out


_DUMMY_POOL: dict[int, torch.Tensor] = {}


def _dummy_fp8_pool(device: torch.device, device_index: int) -> torch.Tensor:
    """One small E4M3 pool for the dense-cache TMA descriptors of the shared kernel ABI.

    The NVFP4 schedule gathers every byte from the data / scale pools and never
    touches these descriptors, but the compiled kernel still binds them.
    """
    pool = _DUMMY_POOL.get(device_index)
    if pool is None:
        pool = torch.zeros((64, _HEAD_DIM), dtype=torch.float8_e4m3fn, device=device)
        _DUMMY_POOL[device_index] = pool
    return pool


@flashinfer_experimental_api(trace=prims_ts_dsa_sparse_mla_nvfp4_trace)
def prims_ts_dsa_sparse_mla_nvfp4(
    query: torch.Tensor,
    kv_data_pool: torch.Tensor,
    kv_scale_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    cum_seq_lens_q: Optional[torch.Tensor],
    *,
    max_q_len: int = 1,
    sparse_mla_top_k: int,
    sparse_mla_top_k_lens: Optional[torch.Tensor] = None,
    residual_dim: int = _NVFP4_RESIDUAL_DIM,
    kv_global_dequant_scale: float,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    skip_corr_threshold: float = 0.0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""DSA sparse MLA decode over the TRT-LLM NVFP4 MLA KV cache (PrimTS 1CTA, 64 rows per CTA).

    ``kv_data_pool`` is the uint8 data pool with ``(576 + residual_dim) // 2`` bytes
    per token row (``[N, 320]`` for ``residual_dim=64``) and ``kv_scale_pool`` the
    E4M3 block-scale pool with ``(576 + residual_dim) // 16`` bytes per row
    (``[N, 40]``), both as TRT-LLM's ``quantizeAndWriteMlaFp4`` writes them (paged
    pools are passed flat: row = page_id * page_size + offset).  The last
    ``residual_dim`` dims (the RoPE part) are stored as main + residual pairs.

    ``kv_global_dequant_scale`` is TRT-LLM's ``amax / (448 * 6)``.  QK multiplies the
    raw E2M1 rows by their E4M3 block scales inside the MMA, so the whole global
    factor is folded into the QK scale.  For V the power-of-two part of the factor
    is folded into the E4M3 block scales during the on-chip dequantization (an
    exponent shift, exact unless the scaled block scale leaves the E4M3 normal
    range) and the residual mantissa ``m`` in ``[1, 2)`` into the PV scale, so V is
    rounded to E4M3 before ``m`` is applied where TRT-LLM's gather kernel rounds
    after it (relative L2 difference ~3e-2 on random data for a non-power-of-two
    factor, none when the factor is a power of two).  ``bmm1_scale`` /
    ``bmm2_scale`` therefore carry only the query / softmax / output factors.
    Everything else follows :func:`prims_ts_dsa_sparse_mla_fp8`.
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
        residual_dim == _NVFP4_RESIDUAL_DIM,
        f"residual_dim={_NVFP4_RESIDUAL_DIM} (the TRT-LLM default) is the only supported layout",
    )
    data_row = (_HEAD_DIM + residual_dim) // 2
    scale_row = (_HEAD_DIM + residual_dim) // 16
    _require(
        kv_data_pool.is_cuda
        and kv_data_pool.dtype == torch.uint8
        and kv_data_pool.is_contiguous()
        and kv_data_pool.shape[-1] == data_row,
        f"kv_data_pool must be a contiguous CUDA uint8 tensor with {data_row}-byte rows",
    )
    _require(
        kv_scale_pool.is_cuda
        and kv_scale_pool.dtype in (torch.uint8, torch.float8_e4m3fn)
        and kv_scale_pool.is_contiguous()
        and kv_scale_pool.shape[-1] == scale_row,
        f"kv_scale_pool must be a contiguous CUDA E4M3/uint8 tensor with {scale_row}-byte rows",
    )
    data_pool = kv_data_pool.view(-1, data_row)
    scale_pool = kv_scale_pool.view(torch.uint8).view(-1, scale_row)
    _require(
        data_pool.shape[0] == scale_pool.shape[0],
        "kv_data_pool and kv_scale_pool must have the same row count",
    )
    _require(
        data_pool.data_ptr() % 16 == 0 and scale_pool.data_ptr() % 16 == 0,
        "kv_data_pool and kv_scale_pool must start at 16-byte-aligned addresses",
    )
    _require(
        isinstance(kv_global_dequant_scale, (int, float))
        and not isinstance(kv_global_dequant_scale, bool)
        and math.isfinite(float(kv_global_dequant_scale))
        and kv_global_dequant_scale > 0,
        "kv_global_dequant_scale must be a positive finite number",
    )
    device = query.device
    device_index = _device_index(device)
    _require_supported_device(device_index)
    if _validate_inputs_enabled():
        _validate_route_values(
            block_tables,
            top_k_lens,
            pool_rows=data_pool.shape[0],
            sparse_mla_top_k=sparse_mla_top_k,
        )
    num_heads = query.shape[1]
    compiled = compile_dsa_sparse_kernel(
        device_index,
        num_heads=num_heads,
        sparse_mla_top_k=sparse_mla_top_k,
        enable_skip_correction=skip_corr_threshold > 0.0,
        nvfp4=True,
        residual_dim=residual_dim,
    )
    # s = m * 2**e with m in [1, 2): the exponent folds exactly into the E4M3
    # block scales of the V dequantization; m goes through the PV scale.  QK
    # multiplies the raw block scales inside the mxf4nvf4 MMA, so it takes the
    # whole factor.
    mantissa, exponent = math.frexp(float(kv_global_dequant_scale))
    pow2_scale = math.ldexp(1.0, exponent - 1)
    mantissa_scale = 2.0 * mantissa
    launch_dsa_sparse_kernel(
        compiled,
        query=query,
        kv_pool=_dummy_fp8_pool(device, device_index),
        block_tables=block_tables,
        top_k_lens=top_k_lens,
        out=out,
        lse=lse,
        bmm1_scale=bmm1_scale * float(kv_global_dequant_scale),
        bmm2_scale=bmm2_scale * mantissa_scale,
        skip_corr_threshold=skip_corr_threshold,
        nvfp4_data_pool=data_pool,
        nvfp4_scale_pool=scale_pool,
        nvfp4_dequant_scale=pow2_scale,
    )
    return out


__all__ = ["prims_ts_dsa_sparse_mla_fp8", "prims_ts_dsa_sparse_mla_nvfp4"]
