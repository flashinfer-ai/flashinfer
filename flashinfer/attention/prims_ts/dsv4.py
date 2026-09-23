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
"""

from __future__ import annotations

import ctypes
import functools
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch

from flashinfer.api_logging import flashinfer_experimental_api
from flashinfer.trace.templates.attention import (
    prims_ts_dsv4_sparse_mla_rope_quant_trace,
    prims_ts_dsv4_sparse_mla_rope_quant_ue8m0_trace,
    prims_ts_dsv4_sparse_mla_trace,
)


_HEADS = 128
_HEAD_DIM = 512
_SWA_TOPK = 128
_TILE_K = 128
# The fused RoPE/FP8 epilogue receives O as one flat [T * 128 * 512] E4M3 tensor
# whose extent is a DSL Int32 dynamic shape; T * 65536 must stay below 2**31.
_MAX_ROPE_QUANT_TOTAL_Q = (2**31 - 1) // (_HEADS * _HEAD_DIM)
_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 2"
_RAW_GATHER4_DESCRIPTOR_BYTES = 128
# Encoded descriptor pairs are keyed by pool identity.  Serving reuses a
# handful of KV pools, so a small LRU bounds device memory when pools are
# reallocated.
_RAW_GATHER4_DESCRIPTOR_PAIR_CACHE: OrderedDict[
    tuple[tuple[int, int, int], tuple[int, int, int]], torch.Tensor
] = OrderedDict()
_RAW_GATHER4_DESCRIPTOR_PAIR_CACHE_CAPACITY = 16


def _encode_raw_gather4_descriptor_host(pool: torch.Tensor) -> torch.Tensor:
    """Encode the token-sparse ``[512, INT_MAX]`` Gather4 map (one page slot per transaction) on the host."""

    from cuda.bindings import driver as cuda_drv

    err, tensor_map = cuda_drv.cuTensorMapEncodeTiled(
        cuda_drv.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        pool.data_ptr(),
        [
            cuda_drv.cuuint64_t(_HEAD_DIM),
            cuda_drv.cuuint64_t((1 << 31) - 1),
        ],
        [cuda_drv.cuuint64_t(_HEAD_DIM)],
        [cuda_drv.cuuint32_t(128), cuda_drv.cuuint32_t(1)],
        [cuda_drv.cuuint32_t(1), cuda_drv.cuuint32_t(1)],
        cuda_drv.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda_drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda_drv.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        cuda_drv.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if int(err) != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled for DSV4 Gather4 failed: {err}")

    return torch.tensor(
        list(ctypes.string_at(tensor_map.getPtr(), _RAW_GATHER4_DESCRIPTOR_BYTES)),
        dtype=torch.uint8,
    )


def _raw_gather4_descriptor_pair(
    sliding_window_kv_pool: torch.Tensor, compressed_kv_pool: torch.Tensor
) -> torch.Tensor:
    """Pack raw Gather4 maps in the existing static-split workspace ABI."""

    swa_key = (
        sliding_window_kv_pool.device.index or 0,
        sliding_window_kv_pool.data_ptr(),
        sliding_window_kv_pool.numel(),
    )
    compressed_key = (
        compressed_kv_pool.device.index or 0,
        compressed_kv_pool.data_ptr(),
        compressed_kv_pool.numel(),
    )
    key = (swa_key, compressed_key)
    cached = _RAW_GATHER4_DESCRIPTOR_PAIR_CACHE.get(key)
    if cached is not None:
        _RAW_GATHER4_DESCRIPTOR_PAIR_CACHE.move_to_end(key)
        return cached
    if torch.cuda.is_current_stream_capturing():
        # The miss path below performs a pageable H2D copy and may synchronize
        # the device on eviction; neither is legal under stream capture.
        raise RuntimeError(
            "DSV4 Gather4 descriptors for this KV pool pair were not built before "
            "CUDA graph capture; run the same pool tensors once eagerly first"
        )

    # Assemble the immutable 256-byte pair on the host and publish it with one
    # H2D copy; mutating a device-resident tensor map would need a
    # ``fence.proxy.tensormap`` before its first TMA use.
    host_descriptor_pair = torch.cat(
        (
            _encode_raw_gather4_descriptor_host(sliding_window_kv_pool),
            _encode_raw_gather4_descriptor_host(compressed_kv_pool),
        )
    )
    descriptor_pair = host_descriptor_pair.to(
        device=sliding_window_kv_pool.device, non_blocking=False
    )
    if descriptor_pair.data_ptr() % 64:
        raise RuntimeError("DSV4 Gather4 descriptor pair must be 64-byte aligned")
    if (
        len(_RAW_GATHER4_DESCRIPTOR_PAIR_CACHE)
        >= _RAW_GATHER4_DESCRIPTOR_PAIR_CACHE_CAPACITY
    ):
        # An evicted pair may still be read by an in-flight launch; drain the
        # device before releasing it.  Eviction only happens when a pool is
        # reallocated, so this synchronization is off the steady-state path.
        torch.cuda.synchronize(sliding_window_kv_pool.device)
        _RAW_GATHER4_DESCRIPTOR_PAIR_CACHE.popitem(last=False)
    _RAW_GATHER4_DESCRIPTOR_PAIR_CACHE[key] = descriptor_pair
    return descriptor_pair


_DUMMY_LSE_CACHE: dict[int, torch.Tensor] = {}


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


@functools.lru_cache(maxsize=None)
def _dsv4_max_active_clusters(device_index: int) -> int:
    """Return the resident 2CTA cluster count used for scheduler selection."""

    import cutlass.utils as cutlass_utils
    from cuda.bindings import driver as cuda_drv

    with torch.cuda.device(device_index):
        stream = cuda_drv.CUstream(torch.cuda.current_stream(device_index).cuda_stream)
        return int(
            cutlass_utils.HardwareInfo(device_index).get_max_active_clusters(2, stream)
        )


def _dsv4_uses_persistent_scheduler(
    batch_size: int, max_seq_len_q: int, max_active_clusters: int
) -> bool:
    """Choose CLC persistent scheduling only beyond one resident wave.

    Scheduled work is ``B * max_seq_len_q`` 2CTA tiles (padding included).
    Within the resident cluster capacity every tile launches directly on the
    static grid; beyond it the persistent kernel steals tiles dynamically.
    """

    return batch_size * max_seq_len_q > max_active_clusters


@functools.lru_cache(maxsize=None)
def _get_compiled_dsv4_sparse_mla(
    device_index: int,
    enable_skip_correction: bool,
    fuses_inv_rope_fp8_quant: bool,
    stores_lse: bool,
    is_persistent: bool = True,
    uses_ue8m0_scale_o: bool = False,
):
    """Compile one DSV4 sparse MLA launch specialization.

    The JIT key is exactly the argument list; packed ``B/T/maxQ/Kmax`` are
    runtime fields and enter it only through ``is_persistent``.  A server
    whose ``B * max_seq_len_q`` crosses the resident-wave boundary therefore
    owns at most two variants and should warm both before graph capture.
    """

    import cutlass
    import cutlass.cute as cute

    from .kernels.mla_decode.throughput_2cta.kernel import MlaDecodeTs

    max_active_clusters = _dsv4_max_active_clusters(device_index)

    kernel = MlaDecodeTs(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=(128, 128),
        mma_pv_tiler_mn=(128, 256),
        max_active_clusters=max_active_clusters,
        page_size=1,
        is_persistent=is_persistent,
        is_var_seq=True,
        is_var_split_kv=False,
        static_split_kv=1,
        static_seq_len_k=None,
        qkv_dtype="e4m3",
        out_dtype="e4m3" if fuses_inv_rope_fp8_quant else "bf16",
        rope_dim=0,
        num_heads=_HEADS,
        # Hq=M=128 fixes the group ratio at one.  Runtime maxQ/B arrive
        # through the metadata/scalar ABI and only steer the scheduler; these
        # placeholders just satisfy the constructor checks.
        seq_len_q=1,
        batch_size=1,
        mask_type="causal",
        is_dynamic_token_sparse=True,
        sparse_swa_topk=_SWA_TOPK,
        enable_skip_correction=enable_skip_correction,
        dsv4_fuses_inv_rope_fp8_quant=fuses_inv_rope_fp8_quant,
        dsv4_uses_ue8m0_scale_o=uses_ue8m0_scale_o,
        stores_lse=stores_lse,
    )
    compressed_rows = cute.sym_int()
    swa_rows = cute.sym_int()
    runtime_batch = cute.sym_int()
    runtime_num_q_offsets = cute.sym_int()
    runtime_total_q = cute.sym_int()
    runtime_sparse_capacity = cute.sym_int()
    q_fake = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (_HEADS, _HEAD_DIM, runtime_total_q),
        stride=(_HEAD_DIM, 1, _HEADS * _HEAD_DIM),
        assumed_align=16,
    )
    compressed_cache_fake = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (1, _HEAD_DIM, compressed_rows),
        stride=(_HEAD_DIM, 1, _HEAD_DIM),
        assumed_align=16,
    )
    swa_cache_fake = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (1, _HEAD_DIM, swa_rows),
        stride=(_HEAD_DIM, 1, _HEAD_DIM),
        assumed_align=16,
    )
    physical_indices_fake = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (runtime_sparse_capacity, runtime_total_q),
        stride=(1, runtime_sparse_capacity),
        assumed_align=16,
    )
    if fuses_inv_rope_fp8_quant:
        # The physical output is [H/8, T, 8, D]; the epilogue owns the
        # non-affine head mapping, so the ABI sees one flat element span.
        out_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN,
            (runtime_total_q * _HEADS * _HEAD_DIM,),
            stride_order=(0,),
            assumed_align=16,
        )
        runtime_cos_sin_elts = cute.sym_int()
        runtime_scale_elts = cute.sym_int()
        inv_rope_cos_sin_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (runtime_cos_sin_elts,),
            stride_order=(0,),
            assumed_align=16,
        )
        # FP32 scales are one value per (head, D128 block); UE8M0 scales are
        # one INT32 word per head packing the four block exponent bytes.
        dsv4_o_scale_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32 if uses_ue8m0_scale_o else cutlass.Float32,
            (runtime_scale_elts,),
            stride_order=(0,),
            assumed_align=16,
        )
    else:
        out_fake = cute.runtime.make_fake_tensor(
            cutlass.BFloat16,
            (_HEADS, _HEAD_DIM, runtime_total_q),
            stride=(_HEAD_DIM, 1, _HEADS * _HEAD_DIM),
            assumed_align=16,
        )
        inv_rope_cos_sin_fake = None
        dsv4_o_scale_fake = None
    lse_fake = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (_HEADS, runtime_total_q),
        stride=(1, _HEADS),
        assumed_align=16,
    )
    raw_seq_lens_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (runtime_batch,), stride_order=(0,), assumed_align=16
    )
    sparse_lens_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (runtime_total_q,), stride_order=(0,), assumed_align=16
    )
    cu_seqlens_q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (runtime_num_q_offsets,),
        stride_order=(0,),
        assumed_align=4,
    )
    raw_tma_descriptor_pair_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (2 * _RAW_GATHER4_DESCRIPTOR_BYTES,),
        stride_order=(0,),
        assumed_align=64,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compile_specializations: tuple[object, ...] = (cute.FrontendNext,)
    with torch.cuda.device(device_index):
        return cute.compile[compile_specializations](
            kernel,
            q_fake,
            q_fake,
            compressed_cache_fake,
            compressed_cache_fake,
            physical_indices_fake,
            out_fake,
            lse_fake,
            None,
            raw_tma_descriptor_pair_fake,
            cutlass.Int32(1),
            raw_seq_lens_fake,
            sparse_lens_fake,
            cu_seqlens_q_fake,
            None,
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            swa_cache_fake,
            cutlass.Int32(1),
            cutlass.Float32(8.0 if enable_skip_correction else 0.0),
            inv_rope_cos_sin_fake,
            dsv4_o_scale_fake,
            stream_fake,
            options=_COMPILE_OPTIONS,
        )


def _validate_optional_values(
    *,
    sparse_topk_lens: torch.Tensor,
    sparse_capacity: int,
    seq_lens_kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_idx_kv: torch.Tensor,
    swa_rows: int,
    compressed_rows: int,
    max_seq_len_q: int,
) -> None:
    """Run device-synchronizing metadata and route-bound checks when ``FLASHINFER_VALIDATE_INPUTS`` is set."""

    if os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") in ("", "0"):
        return
    if int(cu_seqlens_q[0]) != 0 or int(cu_seqlens_q[-1]) != sparse_topk_lens.numel():
        raise ValueError("ptr_cum_seq_lens_q must start at zero and end at T")
    query_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    if torch.any(query_lens <= 0):
        raise ValueError("every packed request must contain at least one Q row")
    if int(query_lens.max()) > max_seq_len_q:
        # The grid has B * max_seq_len_q work tiles; rows past the bound would
        # never be scheduled and their output would stay unwritten.
        raise ValueError(
            "max_seq_len_q must be at least the longest packed request "
            f"({int(query_lens.max())}), got {max_seq_len_q}"
        )
    if torch.any(seq_lens_kv < query_lens):
        raise ValueError("ptr_seq_lens_kv must be raw lengths after this forward")
    if torch.any(sparse_topk_lens < _SWA_TOPK) or torch.any(
        sparse_topk_lens > sparse_capacity
    ):
        raise ValueError("ptr_sparse_mla_topk_lens values must be in [128, Kmax]")
    swa_slots = page_idx_kv[:, :_SWA_TOPK]
    if torch.any((swa_slots < -1) | (swa_slots >= swa_rows)):
        raise ValueError(
            "ptr_page_idx_kv SWA slots must be -1 or a sliding_window_kv_pool row"
        )
    compressed_slots = page_idx_kv[:, _SWA_TOPK:]
    reachable = (
        torch.arange(_SWA_TOPK, sparse_capacity, device=page_idx_kv.device)[None, :]
        < sparse_topk_lens[:, None]
    )
    if torch.any(
        reachable & ((compressed_slots < 0) | (compressed_slots >= compressed_rows))
    ):
        raise ValueError(
            "ptr_page_idx_kv compressed slots inside the scan width must index "
            "compressed_kv_pool rows"
        )


@dataclass(frozen=True)
class _CommonInputs:
    """Host-validated launch facts shared by both public DSV4 entry points."""

    device: torch.device
    device_index: int
    total_q: int
    batch_size: int
    sparse_capacity: int
    skip_corr_threshold: float
    enable_skip_correction: bool
    is_persistent: bool


def _validate_common_inputs(
    *,
    query: torch.Tensor,
    compressed_kv_pool: torch.Tensor,
    sliding_window_kv_pool: torch.Tensor,
    ptr_page_idx_kv: torch.Tensor,
    ptr_sparse_mla_topk_lens: torch.Tensor,
    ptr_seq_lens_kv: torch.Tensor,
    ptr_cum_seq_lens_q: torch.Tensor,
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
    if not isinstance(max_seq_len_q, int) or max_seq_len_q <= 0:
        raise ValueError("max_seq_len_q must be a positive Python int")
    device = query.device
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((10, 0), (10, 3)):
        raise NotImplementedError(
            f"DSV4 sparse MLA TS requires SM100 or SM103, got "
            f"SM{capability[0]}{capability[1]}"
        )

    for pool, name in (
        (compressed_kv_pool, "compressed_kv_pool"),
        (sliding_window_kv_pool, "sliding_window_kv_pool"),
    ):
        _require_cuda_tensor(pool, name, dtype=torch.float8_e4m3fn, ndim=2)
        if pool.shape[0] <= 0 or pool.shape[1] != _HEAD_DIM:
            raise ValueError(f"{name} must have shape [N>0, {_HEAD_DIM}]")
        if pool.device != device:
            raise ValueError(f"{name} must be on {device}, got {pool.device}")

    _require_cuda_tensor(ptr_page_idx_kv, "ptr_page_idx_kv", dtype=torch.int32, ndim=2)
    if ptr_page_idx_kv.device != device or ptr_page_idx_kv.shape[0] != total_q:
        raise ValueError("ptr_page_idx_kv must have shape [T, Kmax] on query.device")
    sparse_capacity = int(ptr_page_idx_kv.shape[1])
    # Selector rows are read as 16-byte int32x4 vectors, so Kmax must be a
    # multiple of 4; a partial final K128 tile (e.g. Kmax=192) is allowed.
    if sparse_capacity < _SWA_TOPK or sparse_capacity % 4:
        raise ValueError("Kmax must be at least 128 and divisible by 4")

    _require_int32_cuda_vector(ptr_sparse_mla_topk_lens, "ptr_sparse_mla_topk_lens")
    if ptr_sparse_mla_topk_lens.device != device or ptr_sparse_mla_topk_lens.shape != (
        total_q,
    ):
        raise ValueError("ptr_sparse_mla_topk_lens must have shape [T] on query.device")
    _require_int32_cuda_vector(ptr_seq_lens_kv, "ptr_seq_lens_kv")
    # The compiled spec declares ``assumed_align=4`` for the packed offsets,
    # so indptr views such as ``qo_indptr[1:]`` are legal here.
    _require_int32_cuda_vector(ptr_cum_seq_lens_q, "ptr_cum_seq_lens_q", align=4)
    if ptr_seq_lens_kv.device != device or ptr_cum_seq_lens_q.device != device:
        raise ValueError("sequence metadata must be on query.device")
    batch_size = int(ptr_seq_lens_kv.numel())
    if batch_size <= 0 or ptr_cum_seq_lens_q.shape != (batch_size + 1,):
        raise ValueError("sequence metadata must have shapes [B] and [B + 1]")
    _validate_optional_values(
        sparse_topk_lens=ptr_sparse_mla_topk_lens,
        sparse_capacity=sparse_capacity,
        seq_lens_kv=ptr_seq_lens_kv,
        cu_seqlens_q=ptr_cum_seq_lens_q,
        page_idx_kv=ptr_page_idx_kv,
        swa_rows=int(sliding_window_kv_pool.shape[0]),
        compressed_rows=int(compressed_kv_pool.shape[0]),
        max_seq_len_q=max_seq_len_q,
    )
    for scale, name in ((bmm1_scale, "bmm1_scale"), (bmm2_scale, "bmm2_scale")):
        if not isinstance(scale, (float, int)) or not math.isfinite(float(scale)):
            raise ValueError(f"{name} must be a finite Python number")
    from .kernels.mla_decode.throughput_2cta.config import (
        validate_skip_corr_threshold,
    )

    skip_corr_threshold = validate_skip_corr_threshold(
        skip_corr_threshold, qkv_dtype="e4m3", bmm1_scale=float(bmm1_scale)
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
        skip_corr_threshold=skip_corr_threshold,
        enable_skip_correction=enable_skip_correction,
        is_persistent=is_persistent,
    )


@flashinfer_experimental_api(trace=prims_ts_dsv4_sparse_mla_trace)
def prims_ts_dsv4_sparse_mla(
    query: torch.Tensor,
    compressed_kv_pool: torch.Tensor,
    sliding_window_kv_pool: torch.Tensor,
    ptr_page_idx_kv: torch.Tensor,
    ptr_sparse_mla_topk_lens: torch.Tensor,
    ptr_seq_lens_kv: torch.Tensor,
    ptr_cum_seq_lens_q: torch.Tensor,
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
    compressed_kv_pool, sliding_window_kv_pool : torch.Tensor
        Independent physical E4M3 row pools ``[N, 512]`` (page size one).
    ptr_page_idx_kv : torch.Tensor
        INT32 physical routing ``[T, Kmax]``. Slots ``[0, 128)`` address the
        sliding-window pool, the rest address the compressed pool. Inactive
        sliding-window slots may be ``-1``; slots beyond the scan width are
        never read. ``Kmax >= 128`` and ``Kmax % 4 == 0``.
    ptr_sparse_mla_topk_lens : torch.Tensor
        INT32 ``[T]`` active scan width per query token.
    ptr_seq_lens_kv : torch.Tensor
        INT32 ``[B]`` raw post-append KV lengths.
    ptr_cum_seq_lens_q : torch.Tensor
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
        compressed_kv_pool=compressed_kv_pool,
        sliding_window_kv_pool=sliding_window_kv_pool,
        ptr_page_idx_kv=ptr_page_idx_kv,
        ptr_sparse_mla_topk_lens=ptr_sparse_mla_topk_lens,
        ptr_seq_lens_kv=ptr_seq_lens_kv,
        ptr_cum_seq_lens_q=ptr_cum_seq_lens_q,
        max_seq_len_q=max_seq_len_q,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        skip_corr_threshold=skip_corr_threshold,
    )
    device, device_index, total_q = common.device, common.device_index, common.total_q
    skip_corr_threshold = common.skip_corr_threshold
    enable_skip_correction = common.enable_skip_correction
    is_persistent = common.is_persistent
    expected_out_shape = (total_q, _HEADS, _HEAD_DIM)
    if out is None:
        out = torch.empty(expected_out_shape, device=device, dtype=torch.bfloat16)
    else:
        _require_cuda_tensor(out, "out", dtype=torch.bfloat16, ndim=3)
        if out.device != device or tuple(out.shape) != expected_out_shape:
            raise ValueError(f"out must have shape {expected_out_shape}")
    expected_lse_shape = (total_q, _HEADS)
    stores_lse = lse is not None
    if lse is None:
        # The ABI still binds an LSE tensor; the store itself is compiled away.
        lse_kernel = _dummy_lse(device, device_index, total_q)
    else:
        _require_cuda_tensor(lse, "lse", dtype=torch.float32, ndim=2)
        if lse.device != device or tuple(lse.shape) != expected_lse_shape:
            raise ValueError(f"lse must have shape {expected_lse_shape}")
        lse_kernel = lse

    compiled = _get_compiled_dsv4_sparse_mla(
        device_index,
        enable_skip_correction,
        fuses_inv_rope_fp8_quant=False,
        stores_lse=stores_lse,
        is_persistent=is_persistent,
    )
    q_kernel = query.permute(1, 2, 0)
    compressed_kernel = compressed_kv_pool.view(-1, 1, _HEAD_DIM).permute(1, 2, 0)
    swa_kernel = sliding_window_kv_pool.view(-1, 1, _HEAD_DIM).permute(1, 2, 0)
    raw_tma_descriptor_pair = _raw_gather4_descriptor_pair(
        sliding_window_kv_pool, compressed_kv_pool
    )
    compiled(
        q_kernel,
        q_kernel,
        compressed_kernel,
        compressed_kernel,
        ptr_page_idx_kv.transpose(0, 1),
        out.permute(1, 2, 0),
        lse_kernel.transpose(0, 1),
        None,
        raw_tma_descriptor_pair,
        1,
        ptr_seq_lens_kv,
        ptr_sparse_mla_topk_lens,
        ptr_cum_seq_lens_q,
        None,
        float(bmm1_scale),
        float(bmm2_scale),
        swa_kernel,
        max_seq_len_q,
        float(skip_corr_threshold),
        None,
        None,
    )
    return out


def _launch_dsv4_rope_quant(
    query: torch.Tensor,
    compressed_kv_pool: torch.Tensor,
    sliding_window_kv_pool: torch.Tensor,
    ptr_page_idx_kv: torch.Tensor,
    ptr_sparse_mla_topk_lens: torch.Tensor,
    ptr_seq_lens_kv: torch.Tensor,
    ptr_cum_seq_lens_q: torch.Tensor,
    inv_rope_cos_sin_cache: torch.Tensor,
    *,
    max_seq_len_q: int,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    skip_corr_threshold: float = 8.0,
    out: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    uses_ue8m0_scale_o: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared launcher for both fused RoPE/FP8 quant scale formats."""

    common = _validate_common_inputs(
        query=query,
        compressed_kv_pool=compressed_kv_pool,
        sliding_window_kv_pool=sliding_window_kv_pool,
        ptr_page_idx_kv=ptr_page_idx_kv,
        ptr_sparse_mla_topk_lens=ptr_sparse_mla_topk_lens,
        ptr_seq_lens_kv=ptr_seq_lens_kv,
        ptr_cum_seq_lens_q=ptr_cum_seq_lens_q,
        max_seq_len_q=max_seq_len_q,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        skip_corr_threshold=skip_corr_threshold,
    )
    device, device_index, total_q = common.device, common.device_index, common.total_q
    skip_corr_threshold = common.skip_corr_threshold
    enable_skip_correction = common.enable_skip_correction
    is_persistent = common.is_persistent
    if total_q > _MAX_ROPE_QUANT_TOTAL_Q:
        raise ValueError(
            "the fused RoPE/FP8 output is addressed as a flat T*128*512 Int32 "
            f"extent; total packed queries must be <= {_MAX_ROPE_QUANT_TOTAL_Q}, "
            f"got {total_q}"
        )
    _require_cuda_tensor(
        inv_rope_cos_sin_cache,
        "inv_rope_cos_sin_cache",
        dtype=torch.float32,
        ndim=2,
    )
    if (
        inv_rope_cos_sin_cache.device != device
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
    if os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") not in ("", "0"):
        if int(ptr_seq_lens_kv.max()) > inv_rope_cos_sin_cache.shape[0]:
            raise ValueError("inv_rope_cos_sin_cache does not cover raw KV positions")

    expected_out_shape = (_HEADS // 8, total_q, 8, _HEAD_DIM)
    if out is None:
        out = torch.empty(expected_out_shape, device=device, dtype=torch.float8_e4m3fn)
    else:
        _require_cuda_tensor(out, "out", dtype=torch.float8_e4m3fn, ndim=4)
        if out.device != device or tuple(out.shape) != expected_out_shape:
            raise ValueError(f"out must have shape {expected_out_shape}")
    scale_buf_m = (total_q + 3) // 4 * 4
    if uses_ue8m0_scale_o:
        # One INT32 word per (group, head, padded token) packing four
        # D128-block exponent bytes.
        expected_scale_shape = (_HEADS // 8, 8, scale_buf_m)
        scale_dtype = torch.int32
    else:
        expected_scale_shape = (_HEADS // 8, 8 * 4, scale_buf_m)
        scale_dtype = torch.float32
    if out_scale is None:
        out_scale = torch.empty(expected_scale_shape, device=device, dtype=scale_dtype)
    else:
        _require_cuda_tensor(out_scale, "out_scale", dtype=scale_dtype, ndim=3)
        if out_scale.device != device or tuple(out_scale.shape) != expected_scale_shape:
            raise ValueError(f"out_scale must have shape {expected_scale_shape}")

    expected_lse_shape = (total_q, _HEADS)
    stores_lse = lse is not None
    if lse is None:
        lse_kernel = _dummy_lse(device, device_index, total_q)
    else:
        _require_cuda_tensor(lse, "lse", dtype=torch.float32, ndim=2)
        if lse.device != device or tuple(lse.shape) != expected_lse_shape:
            raise ValueError(f"lse must have shape {expected_lse_shape}")
        lse_kernel = lse

    compiled = _get_compiled_dsv4_sparse_mla(
        device_index,
        enable_skip_correction,
        fuses_inv_rope_fp8_quant=True,
        stores_lse=stores_lse,
        is_persistent=is_persistent,
        uses_ue8m0_scale_o=uses_ue8m0_scale_o,
    )
    q_kernel = query.permute(1, 2, 0)
    compressed_kernel = compressed_kv_pool.view(-1, 1, _HEAD_DIM).permute(1, 2, 0)
    swa_kernel = sliding_window_kv_pool.view(-1, 1, _HEAD_DIM).permute(1, 2, 0)
    raw_tma_descriptor_pair = _raw_gather4_descriptor_pair(
        sliding_window_kv_pool, compressed_kv_pool
    )
    compiled(
        q_kernel,
        q_kernel,
        compressed_kernel,
        compressed_kernel,
        ptr_page_idx_kv.transpose(0, 1),
        out.view(-1),
        lse_kernel.transpose(0, 1),
        None,
        raw_tma_descriptor_pair,
        1,
        ptr_seq_lens_kv,
        ptr_sparse_mla_topk_lens,
        ptr_cum_seq_lens_q,
        None,
        float(bmm1_scale),
        float(bmm2_scale),
        swa_kernel,
        max_seq_len_q,
        float(skip_corr_threshold),
        inv_rope_cos_sin_cache.view(-1),
        out_scale.view(-1),
    )
    return out, out_scale


@flashinfer_experimental_api(trace=prims_ts_dsv4_sparse_mla_rope_quant_trace)
def prims_ts_dsv4_sparse_mla_rope_quant(
    query: torch.Tensor,
    compressed_kv_pool: torch.Tensor,
    sliding_window_kv_pool: torch.Tensor,
    ptr_page_idx_kv: torch.Tensor,
    ptr_sparse_mla_topk_lens: torch.Tensor,
    ptr_seq_lens_kv: torch.Tensor,
    ptr_cum_seq_lens_q: torch.Tensor,
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
    ``q`` is ``ptr_seq_lens_kv[b] - query_len[b] + q``.

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

    return _launch_dsv4_rope_quant(
        query,
        compressed_kv_pool,
        sliding_window_kv_pool,
        ptr_page_idx_kv,
        ptr_sparse_mla_topk_lens,
        ptr_seq_lens_kv,
        ptr_cum_seq_lens_q,
        inv_rope_cos_sin_cache,
        max_seq_len_q=max_seq_len_q,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        skip_corr_threshold=skip_corr_threshold,
        out=out,
        out_scale=out_scale,
        lse=lse,
        uses_ue8m0_scale_o=False,
    )


@flashinfer_experimental_api(trace=prims_ts_dsv4_sparse_mla_rope_quant_ue8m0_trace)
def prims_ts_dsv4_sparse_mla_rope_quant_ue8m0(
    query: torch.Tensor,
    compressed_kv_pool: torch.Tensor,
    sliding_window_kv_pool: torch.Tensor,
    ptr_page_idx_kv: torch.Tensor,
    ptr_sparse_mla_topk_lens: torch.Tensor,
    ptr_seq_lens_kv: torch.Tensor,
    ptr_cum_seq_lens_q: torch.Tensor,
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

    return _launch_dsv4_rope_quant(
        query,
        compressed_kv_pool,
        sliding_window_kv_pool,
        ptr_page_idx_kv,
        ptr_sparse_mla_topk_lens,
        ptr_seq_lens_kv,
        ptr_cum_seq_lens_q,
        inv_rope_cos_sin_cache,
        max_seq_len_q=max_seq_len_q,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        skip_corr_threshold=skip_corr_threshold,
        out=out,
        out_scale=out_scale,
        lse=lse,
        uses_ue8m0_scale_o=True,
    )


__all__ = [
    "prims_ts_dsv4_sparse_mla",
    "prims_ts_dsv4_sparse_mla_rope_quant",
    "prims_ts_dsv4_sparse_mla_rope_quant_ue8m0",
]
