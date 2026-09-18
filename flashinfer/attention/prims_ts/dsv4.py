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

"""Task-scheduled DeepSeek V4 sparse MLA attention over physical routing.

Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) share
these kernels.  Cache append/compression, candidate scoring, and page-table
lowering happen upstream; the inputs here are the physical routing tensors.

One M128 work tile is one packed query token times 128 heads, so a launch has
``B * max_seq_len_q`` tiles.  Within one resident wave of 2CTA clusters the
kernel runs on a static grid; beyond it the persistent CLC kernel steals tiles
dynamically.  Both variants produce identical results.

CUDA graphs: capture performs no host work.  Run the same KV pool tensors once
eagerly at the ``B * max_seq_len_q`` you will capture so the Gather4 descriptor
pair and the scheduler variant are cached; a cache miss inside capture raises.

This module is the public adapter: argument validation, the routing contract,
compile-cache lookup, scheduler selection, and the launch call.  Descriptor
encoding, the compile specification, and the positional kernel ABI live in
``kernels/mla_decode/throughput_2cta/token_sparse_launch.py``.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch

from flashinfer.api_logging import flashinfer_experimental_api
from flashinfer.trace.templates.attention import (
    prims_ts_dsv4_sparse_mla_rope_quant_trace,
    prims_ts_dsv4_sparse_mla_rope_quant_ue8m0_trace,
    prims_ts_dsv4_sparse_mla_trace,
)

from .decode import _validate_positive_int
from .kernels.mla_decode.throughput_2cta.token_sparse_launch import (
    HEADS as _HEADS,
    HEAD_DIM as _HEAD_DIM,
    SWA_TOPK as _SWA_TOPK,
    compile_token_sparse_kernel,
    launch_token_sparse_kernel,
    max_active_clusters_2cta as _dsv4_max_active_clusters,
    rope_quant_output_shapes,
)


# The compile cache is inspected under this name by tests and local tooling.
_get_compiled_dsv4_sparse_mla = compile_token_sparse_kernel

_DUMMY_LSE_CACHE: dict[int, torch.Tensor] = {}


@functools.lru_cache(maxsize=None)
def _require_supported_device(device_index: int) -> None:
    """Raise once per device if it is not SM100 or SM103 (cached per device)."""
    capability = torch.cuda.get_device_capability(device_index)
    if capability not in ((10, 0), (10, 3)):
        raise NotImplementedError(
            f"DSV4 sparse MLA TS requires SM100 or SM103, got "
            f"SM{capability[0]}{capability[1]}"
        )


_validate_skip_corr_threshold = None


def _validate_threshold(threshold: float, *, bmm1_scale: float) -> float:
    """Range-check the skip-correction threshold (kernel helper imported lazily)."""
    global _validate_skip_corr_threshold
    if _validate_skip_corr_threshold is None:
        from .kernels.mla_decode.helpers.math import validate_skip_corr_threshold

        _validate_skip_corr_threshold = validate_skip_corr_threshold
    return _validate_skip_corr_threshold(
        threshold, qkv_dtype="e4m3", bmm1_scale=bmm1_scale
    )


def _dummy_lse(device: torch.device, device_index: int, total_q: int) -> torch.Tensor:
    """Return a never-accessed ``[T, 128]`` FP32 view bound to the LSE ABI slot when ``stores_lse=False``."""

    cached = _DUMMY_LSE_CACHE.get(device_index)
    if cached is None or cached.shape[0] < total_q:
        cached = torch.empty((total_q, _HEADS), device=device, dtype=torch.float32)
        _DUMMY_LSE_CACHE[device_index] = cached
    return cached[:total_q]


def _require_cuda_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    dtype: torch.dtype,
    ndim: int,
    align: int = 16,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got rank {tensor.ndim}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % align:
        raise ValueError(f"{name} must be {align}-byte aligned")


def _require_int32_cuda_vector(
    tensor: torch.Tensor, name: str, *, align: int = 16
) -> None:
    _require_cuda_tensor(tensor, name, dtype=torch.int32, ndim=1, align=align)


def _require_output(
    tensor: Optional[torch.Tensor],
    name: str,
    *,
    device: torch.device,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate ``name`` when omitted, otherwise check the caller-owned buffer."""

    if tensor is None:
        return torch.empty(shape, device=device, dtype=dtype)
    _require_cuda_tensor(tensor, name, dtype=dtype, ndim=len(shape))
    if tensor.device != device or tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return tensor


def _dsv4_uses_persistent_scheduler(
    batch_size: int, max_seq_len_q: int, max_active_clusters: int
) -> bool:
    """Choose CLC persistent scheduling only beyond one resident wave.

    Scheduled work is ``B * max_seq_len_q`` 2CTA tiles (padding included).
    Within the resident cluster capacity every tile launches directly on the
    static grid; beyond it the persistent kernel steals tiles dynamically.
    """

    return batch_size * max_seq_len_q > max_active_clusters


def _value_validation_requested() -> bool:
    """Return whether ``FLASHINFER_VALIDATE_INPUTS`` asks for device-synchronizing value checks."""

    if os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") in ("", "0"):
        return False
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FLASHINFER_VALIDATE_INPUTS reads metadata values on the host and "
            "cannot run during CUDA graph capture; validate the inputs eagerly "
            "before capture or unset the variable"
        )
    return True


def _validate_optional_values(
    *,
    sparse_topk_lens: torch.Tensor,
    sparse_capacity: int,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    sparse_indices: torch.Tensor,
    swa_rows: int,
    compressed_rows: int,
    max_seq_len_q: int,
) -> None:
    """Run device-synchronizing metadata and route-bound checks when ``FLASHINFER_VALIDATE_INPUTS`` is set."""

    if not _value_validation_requested():
        return
    if (
        int(cum_seq_lens_q[0]) != 0
        or int(cum_seq_lens_q[-1]) != sparse_topk_lens.numel()
    ):
        raise ValueError("cum_seq_lens_q must start at zero and end at T")
    query_lens = cum_seq_lens_q[1:] - cum_seq_lens_q[:-1]
    if torch.any(query_lens <= 0):
        raise ValueError("every packed request must contain at least one Q row")
    if int(query_lens.max()) > max_seq_len_q:
        # The grid has B * max_seq_len_q work tiles; rows past the bound would
        # never be scheduled and their output would stay unwritten.
        raise ValueError(
            "max_seq_len_q must be at least the longest packed request "
            f"({int(query_lens.max())}), got {max_seq_len_q}"
        )
    if torch.any(seq_lens < query_lens):
        raise ValueError("seq_lens must be raw lengths after this forward")
    if torch.any(sparse_topk_lens < _SWA_TOPK) or torch.any(
        sparse_topk_lens > sparse_capacity
    ):
        raise ValueError("sparse_topk_lens values must be in [128, Kmax]")
    swa_slots = sparse_indices[:, :_SWA_TOPK]
    if torch.any((swa_slots < -1) | (swa_slots >= swa_rows)):
        raise ValueError(
            "sparse_indices SWA slots must be -1 or a sliding_window_kv_pool row"
        )
    compressed_slots = sparse_indices[:, _SWA_TOPK:]
    reachable = (
        torch.arange(_SWA_TOPK, sparse_capacity, device=sparse_indices.device)[None, :]
        < sparse_topk_lens[:, None]
    )
    if torch.any(
        reachable & ((compressed_slots < 0) | (compressed_slots >= compressed_rows))
    ):
        raise ValueError(
            "sparse_indices compressed slots inside the scan width must index "
            "compressed_kv_pool rows"
        )


@dataclass(frozen=True)
class _CommonInputs:
    """Host-validated launch facts shared by the three public DSV4 entry points."""

    device: torch.device
    device_index: int
    total_q: int
    batch_size: int
    sparse_capacity: int
    max_seq_len_q: int
    skip_corr_threshold: float
    enable_skip_correction: bool
    is_persistent: bool


def _validate_common_inputs(
    *,
    query: torch.Tensor,
    sliding_window_kv_pool: torch.Tensor,
    compressed_kv_pool: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    max_seq_len_q: int,
    bmm1_scale: float,
    bmm2_scale: float,
    skip_corr_threshold: float,
) -> _CommonInputs:
    """Host-side shape/dtype/device validation shared by the BF16 and RoPE/FP8 entry points."""

    _require_cuda_tensor(query, "query", dtype=torch.float8_e4m3fn, ndim=3)
    if tuple(query.shape[1:]) != (_HEADS, _HEAD_DIM):
        raise ValueError(
            f"query must have shape [T, {_HEADS}, {_HEAD_DIM}], got {tuple(query.shape)}"
        )
    total_q = int(query.shape[0])
    if total_q <= 0:
        raise ValueError("query must contain at least one packed Q row")
    max_seq_len_q = _validate_positive_int(max_seq_len_q, "max_seq_len_q")
    device = query.device
    _require_supported_device(device.index if device.index is not None else 0)

    for pool, name in (
        (sliding_window_kv_pool, "sliding_window_kv_pool"),
        (compressed_kv_pool, "compressed_kv_pool"),
    ):
        _require_cuda_tensor(pool, name, dtype=torch.float8_e4m3fn, ndim=2)
        if pool.shape[0] <= 0 or pool.shape[1] != _HEAD_DIM:
            raise ValueError(f"{name} must have shape [N>0, {_HEAD_DIM}]")
        if pool.device != device:
            raise ValueError(f"{name} must be on {device}, got {pool.device}")

    _require_cuda_tensor(sparse_indices, "sparse_indices", dtype=torch.int32, ndim=2)
    if sparse_indices.device != device or sparse_indices.shape[0] != total_q:
        raise ValueError("sparse_indices must have shape [T, Kmax] on query.device")
    sparse_capacity = int(sparse_indices.shape[1])
    # Selector rows are read as 16-byte int32x4 vectors, so Kmax must be a
    # multiple of 4; a partial final K128 tile (e.g. Kmax=192) is allowed.
    if sparse_capacity < _SWA_TOPK or sparse_capacity % 4:
        raise ValueError("Kmax must be at least 128 and divisible by 4")

    _require_int32_cuda_vector(sparse_topk_lens, "sparse_topk_lens")
    if sparse_topk_lens.device != device or sparse_topk_lens.shape != (total_q,):
        raise ValueError("sparse_topk_lens must have shape [T] on query.device")
    _require_int32_cuda_vector(seq_lens, "seq_lens")
    # The compiled spec declares ``assumed_align=4`` for the packed offsets,
    # so indptr views such as ``qo_indptr[1:]`` are legal here.
    _require_int32_cuda_vector(cum_seq_lens_q, "cum_seq_lens_q", align=4)
    if seq_lens.device != device or cum_seq_lens_q.device != device:
        raise ValueError("sequence metadata must be on query.device")
    batch_size = int(seq_lens.numel())
    if batch_size <= 0 or cum_seq_lens_q.shape != (batch_size + 1,):
        raise ValueError("sequence metadata must have shapes [B] and [B + 1]")
    _validate_optional_values(
        sparse_topk_lens=sparse_topk_lens,
        sparse_capacity=sparse_capacity,
        seq_lens=seq_lens,
        cum_seq_lens_q=cum_seq_lens_q,
        sparse_indices=sparse_indices,
        swa_rows=int(sliding_window_kv_pool.shape[0]),
        compressed_rows=int(compressed_kv_pool.shape[0]),
        max_seq_len_q=max_seq_len_q,
    )
    for scale, name in ((bmm1_scale, "bmm1_scale"), (bmm2_scale, "bmm2_scale")):
        if not isinstance(scale, (float, int)) or not math.isfinite(float(scale)):
            raise ValueError(f"{name} must be a finite Python number")
    skip_corr_threshold = _validate_threshold(
        skip_corr_threshold, bmm1_scale=float(bmm1_scale)
    )
    enable_skip_correction = skip_corr_threshold > 0.0
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    is_persistent = _dsv4_uses_persistent_scheduler(
        batch_size, max_seq_len_q, _dsv4_max_active_clusters(device_index)
    )
    return _CommonInputs(
        device=device,
        device_index=device_index,
        total_q=total_q,
        batch_size=batch_size,
        sparse_capacity=sparse_capacity,
        max_seq_len_q=max_seq_len_q,
        skip_corr_threshold=skip_corr_threshold,
        enable_skip_correction=enable_skip_correction,
        is_persistent=is_persistent,
    )


def _resolve_lse(
    lse: Optional[torch.Tensor], common: _CommonInputs
) -> tuple[torch.Tensor, bool]:
    """Return ``(lse_buffer, stores_lse)``; the ABI always binds an LSE tensor."""

    if lse is None:
        # The store itself is compiled away, so bind a never-accessed buffer.
        return _dummy_lse(common.device, common.device_index, common.total_q), False
    expected = (common.total_q, _HEADS)
    _require_cuda_tensor(lse, "lse", dtype=torch.float32, ndim=2)
    if lse.device != common.device or tuple(lse.shape) != expected:
        raise ValueError(f"lse must have shape {expected}")
    return lse, True


@flashinfer_experimental_api(trace=prims_ts_dsv4_sparse_mla_trace)
def prims_ts_dsv4_sparse_mla(
    query: torch.Tensor,
    sliding_window_kv_pool: torch.Tensor,
    compressed_kv_pool: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    *,
    max_seq_len_q: int,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    skip_corr_threshold: float = 8.0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Run DeepSeek V4 sparse MLA attention over physical routing metadata.

    Compressed Sparse Attention (CSA, normally R=4) and Heavily Compressed
    Attention (HCA, normally R=128) share this kernel; they differ only in the
    routing metadata.  KV append/compression, candidate selection, and
    page-table lowering must happen before this call.

    Parameters
    ----------
    query : torch.Tensor
        Packed E4M3 queries ``[T, 128, 512]``.
    sliding_window_kv_pool, compressed_kv_pool : torch.Tensor
        Independent physical E4M3 row pools ``[N, 512]`` (page size one).
    sparse_indices : torch.Tensor
        INT32 physical routing ``[T, Kmax]``. Slots ``[0, 128)`` address the
        sliding-window pool, the rest address the compressed pool. Inactive
        sliding-window slots may be ``-1``; slots beyond the scan width are
        never read. ``Kmax >= 128`` and ``Kmax % 4 == 0``.
    sparse_topk_lens : torch.Tensor
        INT32 ``[T]`` active scan width per query token.
    seq_lens : torch.Tensor
        INT32 ``[B]`` raw post-append KV lengths.
    cum_seq_lens_q : torch.Tensor
        INT32 ``[B + 1]`` cumulative packed query offsets.
    max_seq_len_q : int
        Runtime bound on the request-local query length; not part of the JIT
        key. Must cover the longest packed request
        (checked by ``FLASHINFER_VALIDATE_INPUTS=1``).
    bmm1_scale, bmm2_scale : float
        QK and value/output scaling factors.
    skip_corr_threshold : float
        Skip-correction threshold in log2 units. A positive value selects the
        skip-correction specialization: the running row max is frozen while a
        K tile raises it by at most this amount and P is scaled by 1.75. The
        default ``8.0`` is also the E4M3 bound (``1.75 * 2**8 == 448``);
        ``0.0`` keeps the exact rescale.
    out : torch.Tensor, optional
        Caller-owned BF16 output ``[T, 128, 512]``.
    lse : torch.Tensor, optional
        FP32 ``[T, 128]`` log2-sum-exp buffer. When omitted, the LSE store is
        compiled away.

    Returns
    -------
    torch.Tensor
        The BF16 attention output ``[T, 128, 512]``.
    """

    common = _validate_common_inputs(
        query=query,
        sliding_window_kv_pool=sliding_window_kv_pool,
        compressed_kv_pool=compressed_kv_pool,
        sparse_indices=sparse_indices,
        sparse_topk_lens=sparse_topk_lens,
        seq_lens=seq_lens,
        cum_seq_lens_q=cum_seq_lens_q,
        max_seq_len_q=max_seq_len_q,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        skip_corr_threshold=skip_corr_threshold,
    )
    out = _require_output(
        out,
        "out",
        device=common.device,
        shape=(common.total_q, _HEADS, _HEAD_DIM),
        dtype=torch.bfloat16,
    )
    lse_buffer, stores_lse = _resolve_lse(lse, common)
    compiled = compile_token_sparse_kernel(
        common.device_index,
        enable_skip_correction=common.enable_skip_correction,
        fuses_inv_rope_fp8_quant=False,
        stores_lse=stores_lse,
        is_persistent=common.is_persistent,
        uses_ue8m0_scale_o=False,
    )
    launch_token_sparse_kernel(
        compiled,
        query=query,
        sliding_window_kv_pool=sliding_window_kv_pool,
        compressed_kv_pool=compressed_kv_pool,
        sparse_indices=sparse_indices,
        sparse_topk_lens=sparse_topk_lens,
        seq_lens=seq_lens,
        cum_seq_lens_q=cum_seq_lens_q,
        out=out,
        lse=lse_buffer,
        max_seq_len_q=common.max_seq_len_q,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        skip_corr_threshold=common.skip_corr_threshold,
    )
    return out


def _rope_quant_impl(
    query: torch.Tensor,
    sliding_window_kv_pool: torch.Tensor,
    compressed_kv_pool: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    inv_rope_cos_sin_cache: torch.Tensor,
    *,
    max_seq_len_q: int,
    bmm1_scale: float,
    bmm2_scale: float,
    skip_corr_threshold: float,
    out: Optional[torch.Tensor],
    out_scale: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    uses_ue8m0: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared implementation of both fused RoPE/FP8 quant entry points."""

    common = _validate_common_inputs(
        query=query,
        sliding_window_kv_pool=sliding_window_kv_pool,
        compressed_kv_pool=compressed_kv_pool,
        sparse_indices=sparse_indices,
        sparse_topk_lens=sparse_topk_lens,
        seq_lens=seq_lens,
        cum_seq_lens_q=cum_seq_lens_q,
        max_seq_len_q=max_seq_len_q,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        skip_corr_threshold=skip_corr_threshold,
    )
    out_shape, out_scale_shape, out_scale_dtype = rope_quant_output_shapes(
        common.total_q, ue8m0=uses_ue8m0
    )
    _require_cuda_tensor(
        inv_rope_cos_sin_cache,
        "inv_rope_cos_sin_cache",
        dtype=torch.float32,
        ndim=2,
    )
    if (
        inv_rope_cos_sin_cache.device != common.device
        or inv_rope_cos_sin_cache.shape[0] <= 0
        or inv_rope_cos_sin_cache.shape[1] != 64
    ):
        raise ValueError(
            "inv_rope_cos_sin_cache must have shape [max_position>0, 64] "
            "on query.device"
        )
    if inv_rope_cos_sin_cache.data_ptr() % 16:
        # The kernel reads each 256-byte cache row with 128-bit loads.
        raise ValueError("inv_rope_cos_sin_cache must be 16-byte aligned")
    if _value_validation_requested():
        if int(seq_lens.max()) > inv_rope_cos_sin_cache.shape[0]:
            raise ValueError("inv_rope_cos_sin_cache does not cover raw KV positions")

    out = _require_output(
        out, "out", device=common.device, shape=out_shape, dtype=torch.float8_e4m3fn
    )
    out_scale = _require_output(
        out_scale,
        "out_scale",
        device=common.device,
        shape=out_scale_shape,
        dtype=out_scale_dtype,
    )
    lse_buffer, stores_lse = _resolve_lse(lse, common)
    compiled = compile_token_sparse_kernel(
        common.device_index,
        enable_skip_correction=common.enable_skip_correction,
        fuses_inv_rope_fp8_quant=True,
        stores_lse=stores_lse,
        is_persistent=common.is_persistent,
        uses_ue8m0_scale_o=uses_ue8m0,
    )
    launch_token_sparse_kernel(
        compiled,
        query=query,
        sliding_window_kv_pool=sliding_window_kv_pool,
        compressed_kv_pool=compressed_kv_pool,
        sparse_indices=sparse_indices,
        sparse_topk_lens=sparse_topk_lens,
        seq_lens=seq_lens,
        cum_seq_lens_q=cum_seq_lens_q,
        out=out,
        lse=lse_buffer,
        max_seq_len_q=common.max_seq_len_q,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        skip_corr_threshold=common.skip_corr_threshold,
        inv_rope_cos_sin_cache=inv_rope_cos_sin_cache,
        out_scale=out_scale,
    )
    return out, out_scale


@flashinfer_experimental_api(trace=prims_ts_dsv4_sparse_mla_rope_quant_trace)
def prims_ts_dsv4_sparse_mla_rope_quant(
    query: torch.Tensor,
    sliding_window_kv_pool: torch.Tensor,
    compressed_kv_pool: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    inv_rope_cos_sin_cache: torch.Tensor,
    *,
    max_seq_len_q: int,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    skip_corr_threshold: float = 8.0,
    out: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Run DeepSeek V4 sparse MLA with fused inverse-RoPE and FP8 quantization.

    Attention and routing follow :func:`prims_ts_dsv4_sparse_mla`; only the
    epilogue differs. Inverse non-NeoX RoPE is applied to logical D
    ``[448:512]``, then every D128 block is quantized to E4M3 with one FP32
    scale ``amax / 448``. The RoPE position of request ``b``, local query
    ``q`` is ``seq_lens[b] - query_len[b] + q``.

    Parameters
    ----------
    inv_rope_cos_sin_cache : torch.Tensor
        FP32 ``[max_position, 64]`` rows ``[cos(32), sin(32)]``, 16-byte
        aligned, covering every RoPE position.
    out : torch.Tensor, optional
        Caller-owned E4M3 output in grouped physical layout ``[16, T, 8, 512]``.
    out_scale : torch.Tensor, optional
        Caller-owned FP32 ``[16, 32, pad4(T)]`` block scales.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(out, out_scale)``. Columns past ``T`` of ``out_scale`` are not written.
    """

    return _rope_quant_impl(
        query,
        sliding_window_kv_pool,
        compressed_kv_pool,
        sparse_indices,
        sparse_topk_lens,
        seq_lens,
        cum_seq_lens_q,
        inv_rope_cos_sin_cache,
        max_seq_len_q=max_seq_len_q,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        skip_corr_threshold=skip_corr_threshold,
        out=out,
        out_scale=out_scale,
        lse=lse,
        uses_ue8m0=False,
    )


@flashinfer_experimental_api(trace=prims_ts_dsv4_sparse_mla_rope_quant_ue8m0_trace)
def prims_ts_dsv4_sparse_mla_rope_quant_ue8m0(
    query: torch.Tensor,
    sliding_window_kv_pool: torch.Tensor,
    compressed_kv_pool: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    inv_rope_cos_sin_cache: torch.Tensor,
    *,
    max_seq_len_q: int,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    skip_corr_threshold: float = 8.0,
    out: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Run DeepSeek V4 sparse MLA with fused inverse-RoPE/FP8 quant and UE8M0 scales.

    Same as :func:`prims_ts_dsv4_sparse_mla_rope_quant` except for the scale
    format.

    Parameters
    ----------
    out_scale : torch.Tensor, optional
        Caller-owned INT32 ``[16, 8, pad4(T)]``. Word ``[g, h, t]`` packs the
        four D128-block scale bytes of head ``8 * g + h``, block 0 in the
        least significant byte; each byte is the biased exponent of
        ``exp2(ceil(log2(max(amax, 1e-10) / 448)))``, i.e.
        ``scale = 2 ** (byte - 127)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(out, out_scale)``. ``out_scale.permute(2, 0, 1)[:T]`` is the
        ``[T, 16, 8]`` view consumed by UE8M0 block-scaled GEMMs.
    """

    return _rope_quant_impl(
        query,
        sliding_window_kv_pool,
        compressed_kv_pool,
        sparse_indices,
        sparse_topk_lens,
        seq_lens,
        cum_seq_lens_q,
        inv_rope_cos_sin_cache,
        max_seq_len_q=max_seq_len_q,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        skip_corr_threshold=skip_corr_threshold,
        out=out,
        out_scale=out_scale,
        lse=lse,
        uses_ue8m0=True,
    )


__all__ = [
    "prims_ts_dsv4_sparse_mla",
    "prims_ts_dsv4_sparse_mla_rope_quant",
    "prims_ts_dsv4_sparse_mla_rope_quant_ue8m0",
]
