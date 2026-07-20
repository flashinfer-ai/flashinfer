# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/indexer/contiguous_kernel.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""CuTeDSL contiguous logits kernel for the non-paged NSA contract."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Boolean, Float32, Int32, Uint32
from cutlass.base_dsl.compiler import OptLevel, PtxasOptions
from cutlass._mlir.dialects import llvm
from cutlass.cute.core import make_swizzle
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import Int64, T

from flashinfer.experimental.sm12x.attention._shared.cute import copy as cute_copy
from flashinfer.experimental.sm12x.attention._shared.cute import (
    pipeline as cute_pipeline,
)
from flashinfer.experimental.sm12x.attention._shared.cute import ops as attention_ops
from flashinfer.experimental.sm12x._lib.compiler import (
    KernelCompileSpec,
    launch as sm12x_launch,
)
from flashinfer.experimental.sm12x._lib.intrinsics import (
    frag_layout_swizzle_16b_to_8b,
    get_ptr_as_int64,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x4_left_half_b16,
    ldmatrix_m8n8x4_right_half_b16,
    mxfp8_mma_m16n8k32_f32_e4m3,
    shared_ptr_to_u32,
    st_global_v2_f32,
    st_global_v4_f32,
)
from flashinfer.experimental.sm12x._lib.utils import current_cuda_stream
from .kernel import IndexerScoreMode


_INDEX_HEAD_DIM = 128
_FP8_ROW_U32 = _INDEX_HEAD_DIM // 4
_FP8_ROW_VECS = _INDEX_HEAD_DIM // 16
_BLOCK_Q = 32
_BLOCK_K = 64
_WARP_THREADS = 32
_WARPS_Q = _BLOCK_Q // 16
_WARPS_K = _BLOCK_K // 16
_WARPS_PER_CTA = _WARPS_Q * _WARPS_K
_THREADS_PER_CTA = _WARPS_PER_CTA * _WARP_THREADS
_MAX_Q_HEADS = 64
_CONTIGUOUS_TMA_DESC_CACHE_SIZE = 32

_PREFILL_BLOCK_Q = 32  # Same Q tile size as decode
_PREFILL_BLOCK_K = 256  # 4x K tile — fewer CTAs, more work per CTA
_PREFILL_WARPS_Q = _PREFILL_BLOCK_Q // 16  # 2
_PREFILL_WARPS_K = 4  # 4 K-warps, each covering 32 K-rows via num_mma_kv=2
_PREFILL_WARPS_PER_CTA = _PREFILL_WARPS_Q * _PREFILL_WARPS_K  # 8
_PREFILL_THREADS_PER_CTA = _PREFILL_WARPS_PER_CTA * _WARP_THREADS  # 256
_PREFILL_NUM_MMA_Q = 1  # same as decode
_PREFILL_NUM_MMA_KV = 4  # each K-warp covers 64 K-rows (4x16)

_PREFILL_Q_STAGE_ROWS = _PREFILL_BLOCK_Q  # 32 Q rows per tile
_PREFILL_Q_STAGE_COLS = _INDEX_HEAD_DIM  # 128 bytes per Q row
_PREFILL_WEIGHT_COLS = _MAX_Q_HEADS  # 64 heads per Q row
_PREFILL_Q_HEADS_BATCH = (
    8  # Number of Q heads staged in smem at once (fits in 102KB SMEM).
)
_PREFILL_Q_STAGE_BYTES = _PREFILL_Q_STAGE_ROWS * _PREFILL_Q_STAGE_COLS
_PREFILL_WEIGHT_BYTES = _PREFILL_BLOCK_Q * _PREFILL_WEIGHT_COLS * 4  # 8KB

_PREFILL512_BLOCK_Q = _PREFILL_BLOCK_Q
_PREFILL512_BLOCK_K = 512
_PREFILL512_WARPS_Q = _PREFILL512_BLOCK_Q // 16
_PREFILL512_WARPS_K = 4
_PREFILL512_WARPS_PER_CTA = _PREFILL512_WARPS_Q * _PREFILL512_WARPS_K
_PREFILL512_THREADS_PER_CTA = _PREFILL512_WARPS_PER_CTA * _WARP_THREADS
_PREFILL512_NUM_MMA_Q = 1
_PREFILL512_NUM_MMA_KV = 8
_PREFILL512_Q_HEADS_BATCH = 7  # Exp29: BF16 weights free 4KB smem, 10 batches vs 11
_PREFILL512_H32_Q_HEADS_BATCH = 7
_PREFILL512_H32_WEIGHT_COLS = 32

_NSA_CONTIGUOUS_PREFILL_THRESHOLD_ENV = (
    "FLASHINFER_EXP_SM12X_NSA_CONTIGUOUS_PREFILL_THRESHOLD"
)
_NSA_CONTIGUOUS_PREFILL_BLOCK_K_ENV = (
    "FLASHINFER_EXP_SM12X_NSA_CONTIGUOUS_PREFILL_BLOCK_K"
)
_PREFILL512_MIN_Q_ROWS = 1024
_PREFILL512_MIN_K_ROWS = 4096
_PREFILL512_SUPPORTED_NUM_HEADS = (32, 64)
_CONTIGUOUS_TMA_DESC_WORDS = 16


def _assume_contiguous_k_tma_source_aligned(t: cute.Tensor) -> cute.Tensor:
    divby = 128 // t.element_type.width
    strides = []
    for dim, stride in enumerate(t.stride):
        if dim == 1 or isinstance(stride, int):
            strides.append(stride)
        else:
            strides.append(cute.assume(stride, divby=divby))
    return cute.make_tensor(
        t.iterator, cute.make_layout(t.shape, stride=tuple(strides))
    )


def _make_contiguous_k_tma_source(k_quant_bytes: cute.Tensor) -> cute.Tensor:
    return _assume_contiguous_k_tma_source_aligned(k_quant_bytes)


def _make_contiguous_k_tma_smem_layout(block_k: int) -> cute.Layout:
    return cute.make_composed_layout(
        make_swizzle(3, 4, 3),
        0,
        cute.make_layout(
            (block_k, _INDEX_HEAD_DIM),
            stride=(_INDEX_HEAD_DIM, 1),
        ),
    )


def _make_contiguous_k_tma_smem_stage_layout(block_k: int) -> cute.Layout:
    return cute.tile_to_shape(
        _make_contiguous_k_tma_smem_layout(block_k),
        (block_k, _INDEX_HEAD_DIM, 1),
        (0, 1, 2),
    )


@lru_cache(maxsize=16)
def _dummy_contiguous_k_tma_desc_ptrs(device_index: int) -> torch.Tensor:
    return torch.zeros(
        (1,), dtype=torch.int64, device=torch.device("cuda", device_index)
    )


def _raise_binding_extras(api_name: str, extras: list[str]) -> None:
    raise ValueError(
        f"{api_name} binding owns runtime tensors and kernel options; "
        f"do not also pass {', '.join(extras)}"
    )


def _require_bound_arg(value, *, api_name: str, name: str):
    if value is None:
        raise TypeError(f"{api_name} requires {name} or binding")
    return value


@dataclass(frozen=True, kw_only=True)
class IndexerContiguousLogitsKernelBinding:
    q_fp8: torch.Tensor
    weights: torch.Tensor
    k_quant: torch.Tensor
    k_scale: torch.Tensor
    k_start: torch.Tensor
    k_end: torch.Tensor
    preinitialize_invalid_logits: bool = True
    tile_logits: torch.Tensor | None = None
    tile_k_offset: int = 0
    tile_num_k_tiles: int | None = None
    prefill_block_k: int | None = None
    score_mode: int = IndexerScoreMode.NSA_RELU_SUM
    q_u32: torch.Tensor | None = None
    q_bytes: torch.Tensor | None = None
    weights_kernel: torch.Tensor | None = None
    k_quant_bytes: torch.Tensor | None = None
    k_scale_kernel: torch.Tensor | None = None
    k_start_kernel: torch.Tensor | None = None
    k_end_kernel: torch.Tensor | None = None
    out_kernel: torch.Tensor | None = None
    out_view: torch.Tensor | None = None
    k_tma_desc_ptrs: torch.Tensor | None = None
    k_tma_prefill_desc_ptrs: torch.Tensor | None = None

    def run(self) -> torch.Tensor:
        return run_contiguous_logits_kernel(binding=self)


def build_indexer_contiguous_logits_kernel_binding(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    preinitialize_invalid_logits: bool = True,
    tile_logits: torch.Tensor | None = None,
    tile_k_offset: int = 0,
    tile_num_k_tiles: int | None = None,
    prefill_block_k: int | None = None,
    score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
    q_u32: torch.Tensor | None = None,
    q_bytes: torch.Tensor | None = None,
    weights_kernel: torch.Tensor | None = None,
    k_quant_bytes: torch.Tensor | None = None,
    k_scale_kernel: torch.Tensor | None = None,
    k_start_kernel: torch.Tensor | None = None,
    k_end_kernel: torch.Tensor | None = None,
    out_kernel: torch.Tensor | None = None,
    out_view: torch.Tensor | None = None,
    k_tma_desc_ptrs: torch.Tensor | None = None,
    k_tma_prefill_desc_ptrs: torch.Tensor | None = None,
) -> IndexerContiguousLogitsKernelBinding:
    return IndexerContiguousLogitsKernelBinding(
        q_fp8=q_fp8,
        weights=weights,
        k_quant=k_quant,
        k_scale=k_scale,
        k_start=k_start,
        k_end=k_end,
        preinitialize_invalid_logits=bool(preinitialize_invalid_logits),
        tile_logits=tile_logits,
        tile_k_offset=int(tile_k_offset),
        tile_num_k_tiles=None if tile_num_k_tiles is None else int(tile_num_k_tiles),
        prefill_block_k=None if prefill_block_k is None else int(prefill_block_k),
        score_mode=int(score_mode),
        q_u32=q_u32,
        q_bytes=q_bytes,
        weights_kernel=weights_kernel,
        k_quant_bytes=k_quant_bytes,
        k_scale_kernel=k_scale_kernel,
        k_start_kernel=k_start_kernel,
        k_end_kernel=k_end_kernel,
        out_kernel=out_kernel,
        out_view=out_view,
        k_tma_desc_ptrs=k_tma_desc_ptrs,
        k_tma_prefill_desc_ptrs=k_tma_prefill_desc_ptrs,
    )


def _to_kernel_tensor(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int = 16,
) -> cutlass.cute.Tensor:
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    leading_dim = next(
        (idx for idx, stride in enumerate(tensor.stride()) if stride == 1), None
    )
    if leading_dim is not None and tensor.ndim >= 1:
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


def _tensor_meta_key(
    tensor: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...], str, tuple[str, int | None]]:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


def _contiguous_logits_compile_facts(
    *,
    variant: str,
    tiled_output: bool,
    score_mode: int,
    q_u32: torch.Tensor,
    weights: torch.Tensor,
    k_quant_bytes: torch.Tensor,
    k_scale: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    logits_out: torch.Tensor,
    tile_logits: torch.Tensor,
    block_k: int,
    block_score_output: bool = False,
    block_scores: torch.Tensor | None = None,
    q_heads_batch: int | None = None,
) -> tuple[object, ...]:
    facts: list[object] = [
        ("variant", variant),
        ("tiled_output", bool(tiled_output)),
        ("score_mode", int(score_mode)),
        ("head_dim", _INDEX_HEAD_DIM),
        ("q_heads", int(q_u32.shape[1])),
        ("q_u32_cols", int(q_u32.shape[2])),
        ("q_u32_stride_head", int(q_u32.stride(1))),
        ("q_u32_stride_col", int(q_u32.stride(2))),
        ("weights_stride_q", int(weights.stride(0))),
        ("weights_stride_head", int(weights.stride(1))),
        ("k_quant_stride_row", int(k_quant_bytes.stride(0))),
        ("k_quant_stride_col", int(k_quant_bytes.stride(1))),
        ("k_scale_stride", int(k_scale.stride(0))),
        ("k_start_stride", int(k_start.stride(0))),
        ("k_end_stride", int(k_end.stride(0))),
        ("block_k", int(block_k)),
        ("block_score_output", bool(block_score_output)),
    ]
    if q_heads_batch is not None:
        facts.append(("q_heads_batch", int(q_heads_batch)))
    if tiled_output:
        tile_logits_flat = tile_logits.reshape(-1)
        facts.append(("tile_logits_stride", int(tile_logits_flat.stride(0))))
    else:
        facts.extend(
            [
                ("logits_stride_q", int(logits_out.stride(0))),
                ("logits_stride_k", int(logits_out.stride(1))),
            ]
        )
    if block_scores is not None:
        facts.extend(
            [
                ("block_scores_heads", int(block_scores.shape[0])),
                ("block_scores_blocks", int(block_scores.shape[2])),
                ("block_scores_stride_head", int(block_scores.stride(0))),
                ("block_scores_stride_q", int(block_scores.stride(1))),
                ("block_scores_stride_block", int(block_scores.stride(2))),
            ]
        )
    return tuple(facts)


def _pad_kv_rows(
    *,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    pad_block_k: int = _BLOCK_K,
) -> tuple[torch.Tensor, torch.Tensor]:
    padded_rows = ((k_quant.shape[0] + pad_block_k - 1) // pad_block_k) * pad_block_k
    k_quant = k_quant.contiguous()
    k_scale = k_scale.contiguous()
    if padded_rows == k_quant.shape[0]:
        return k_quant, k_scale

    k_quant_padded = torch.empty(
        (padded_rows, _INDEX_HEAD_DIM),
        dtype=k_quant.dtype,
        device=k_quant.device,
    )
    k_scale_padded = torch.empty(
        (padded_rows,),
        dtype=k_scale.dtype,
        device=k_scale.device,
    )
    k_quant_padded[: k_quant.shape[0]].copy_(k_quant)
    k_quant_padded[k_quant.shape[0] :].zero_()
    k_scale_padded[: k_scale.shape[0]].copy_(k_scale)
    k_scale_padded[k_scale.shape[0] :].zero_()
    return k_quant_padded, k_scale_padded


def _view_last_dim_as_u32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.uint8:
        raise ValueError(f"expected uint8 tensor, got {tensor.dtype}")
    if tensor.stride(-1) != 1:
        raise ValueError(f"expected contiguous last dim, got stride={tensor.stride()}")
    if tensor.shape[-1] % 4 != 0:
        raise ValueError(f"last dim must be divisible by 4, got {tensor.shape[-1]}")
    out_shape = (*tensor.shape[:-1], tensor.shape[-1] // 4)
    return tensor.view(torch.uint32).view(out_shape)


@lru_cache(maxsize=1)
def get_sparse_nsa_contiguous_shared_storage_cls():
    class SharedStorage:
        pass

    SharedStorage.__annotations__ = {
        "mbar_ptr_k": cute.struct.MemRange[cutlass.Int64, 1],
        "tile_live": cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, 1],
            16,
        ],
        "k_perm": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, _BLOCK_K * _INDEX_HEAD_DIM],
            1024,
        ],
        "scales": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _BLOCK_K],
            16,
        ],
        "k_start": cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, _BLOCK_Q],
            16,
        ],
        "k_end": cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, _BLOCK_Q],
            16,
        ],
    }

    return cute.struct(SharedStorage)


@lru_cache(maxsize=1)
def get_sparse_nsa_contiguous_prefill_shared_storage_cls():
    class SharedStorage:
        pass

    SharedStorage.__annotations__ = {
        "mbar_ptr_k": cute.struct.MemRange[cutlass.Int64, 1],
        "tile_live": cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, 1],
            16,
        ],
        "k_perm": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, _PREFILL_BLOCK_K * _INDEX_HEAD_DIM],
            1024,
        ],
        "q_bytes_smem": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Uint8,
                _PREFILL_Q_STAGE_ROWS * _PREFILL_Q_STAGE_COLS * _PREFILL_Q_HEADS_BATCH,
            ],
            1024,
        ],
        "w_smem": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _PREFILL_BLOCK_Q * _MAX_Q_HEADS],
            1024,
        ],
        "scales": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _PREFILL_BLOCK_K],
            16,
        ],
        "k_start": cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, _PREFILL_BLOCK_Q],
            16,
        ],
        "k_end": cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, _PREFILL_BLOCK_Q],
            16,
        ],
        "block_partial": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _PREFILL_WARPS_K * _PREFILL_BLOCK_Q],
            16,
        ],
    }

    return cute.struct(SharedStorage)


@cute.jit
def _reduce_quad_max(value: Float32) -> Float32:
    value = attention_ops.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=1))
    value = attention_ops.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=2))
    return value


def _encode_contiguous_k_tma_descriptor_into(
    k_quant_bytes: torch.Tensor,
    desc: torch.Tensor,
    desc_ptrs: torch.Tensor,
    *,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_quant_bytes.ndim != 2 or k_quant_bytes.shape[1] != _INDEX_HEAD_DIM:
        raise ValueError(
            f"k_quant_bytes must have shape (rows, {_INDEX_HEAD_DIM}), "
            f"got {tuple(k_quant_bytes.shape)}"
        )
    if k_quant_bytes.dtype != torch.uint8:
        raise TypeError(
            f"k_quant_bytes must have dtype torch.uint8, got {k_quant_bytes.dtype}"
        )
    if desc.shape != (_CONTIGUOUS_TMA_DESC_WORDS,) or desc.dtype != torch.uint64:
        raise ValueError(
            "desc must be a uint64 tensor with shape "
            f"({_CONTIGUOUS_TMA_DESC_WORDS},), got {tuple(desc.shape)} "
            f"and {desc.dtype}"
        )
    if desc_ptrs.shape != (1,) or desc_ptrs.dtype != torch.int64:
        raise ValueError(
            "desc_ptrs must be an int64 tensor with shape (1,), got "
            f"{tuple(desc_ptrs.shape)} and {desc_ptrs.dtype}"
        )
    if desc.device != k_quant_bytes.device or desc_ptrs.device != k_quant_bytes.device:
        raise ValueError("descriptor tensors must be on the K tensor device")

    U64 = cuda.cuuint64_t
    U32 = cuda.cuuint32_t
    row_bytes = int(k_quant_bytes.stride(0)) * k_quant_bytes.element_size()
    base_ptr = int(k_quant_bytes.data_ptr())
    total_rows = int(k_quant_bytes.shape[0])

    result, tensor_map = cuda.cuTensorMapEncodeTiled(
        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        base_ptr,
        [U64(_INDEX_HEAD_DIM), U64(total_rows)],
        [U64(row_bytes)],
        [U32(_INDEX_HEAD_DIM), U32(int(block_k))],
        [U32(1), U32(1)],
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {result}")

    desc.copy_(
        torch.tensor(
            [int(word) for word in tensor_map.opaque],
            dtype=torch.uint64,
        ),
        non_blocking=True,
    )
    desc_ptrs.copy_(
        torch.tensor([int(desc.data_ptr())], dtype=torch.int64),
        non_blocking=True,
    )
    return desc, desc_ptrs


def _encode_contiguous_k_tma_descriptor(
    k_quant_bytes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_quant_bytes.ndim != 2 or k_quant_bytes.shape[1] != _INDEX_HEAD_DIM:
        raise ValueError(
            f"k_quant_bytes must have shape (rows, {_INDEX_HEAD_DIM}), got {tuple(k_quant_bytes.shape)}"
        )
    if k_quant_bytes.dtype != torch.uint8:
        raise TypeError(
            f"k_quant_bytes must have dtype torch.uint8, got {k_quant_bytes.dtype}"
        )

    # Phase 3: Always use SWIZZLE_128B — hardware XOR pattern matches _permuted_offset_128b
    U64 = cuda.cuuint64_t
    U32 = cuda.cuuint32_t
    row_bytes = int(k_quant_bytes.stride(0)) * k_quant_bytes.element_size()
    base_ptr = int(k_quant_bytes.data_ptr())
    total_rows = int(k_quant_bytes.shape[0])

    result, tensor_map = cuda.cuTensorMapEncodeTiled(
        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        base_ptr,
        [U64(_INDEX_HEAD_DIM), U64(total_rows)],
        [U64(row_bytes)],
        [U32(_INDEX_HEAD_DIM), U32(_BLOCK_K)],
        [U32(1), U32(1)],
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {result}")

    desc = torch.tensor(
        [int(word) for word in tensor_map.opaque],
        dtype=torch.uint64,
        device=k_quant_bytes.device,
    )
    desc_ptrs = torch.tensor(
        [int(desc.data_ptr())], dtype=torch.int64, device=k_quant_bytes.device
    )
    return desc, desc_ptrs


def _get_cached_contiguous_k_tma_descriptor(
    k_quant_bytes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        int(k_quant_bytes.data_ptr()),
        tuple(k_quant_bytes.shape),
        tuple(k_quant_bytes.stride()),
        str(k_quant_bytes.dtype),
        (k_quant_bytes.device.type, k_quant_bytes.device.index),
    )
    cache = getattr(_get_cached_contiguous_k_tma_descriptor, "_cache", None)
    if cache is None:
        cache = OrderedDict()
        setattr(_get_cached_contiguous_k_tma_descriptor, "_cache", cache)
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached
    desc = _encode_contiguous_k_tma_descriptor(k_quant_bytes)
    cache[key] = desc
    if len(cache) > _CONTIGUOUS_TMA_DESC_CACHE_SIZE:
        cache.popitem(last=False)
    return desc


def _encode_contiguous_k_tma_descriptor_prefill(
    k_quant_bytes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TMA descriptor for prefill kernel with _PREFILL_BLOCK_K=256 tile."""
    if k_quant_bytes.ndim != 2 or k_quant_bytes.shape[1] != _INDEX_HEAD_DIM:
        raise ValueError(
            f"k_quant_bytes must have shape (rows, {_INDEX_HEAD_DIM}), got {tuple(k_quant_bytes.shape)}"
        )
    if k_quant_bytes.dtype != torch.uint8:
        raise TypeError(
            f"k_quant_bytes must be dtype torch.uint8, got {k_quant_bytes.dtype}"
        )

    U64 = cuda.cuuint64_t
    U32 = cuda.cuuint32_t
    row_bytes = int(k_quant_bytes.stride(0)) * k_quant_bytes.element_size()
    base_ptr = int(k_quant_bytes.data_ptr())
    total_rows = int(k_quant_bytes.shape[0])

    result, tensor_map = cuda.cuTensorMapEncodeTiled(
        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        base_ptr,
        [U64(_INDEX_HEAD_DIM), U64(total_rows)],
        [U64(row_bytes)],
        [U32(_INDEX_HEAD_DIM), U32(_PREFILL_BLOCK_K)],
        [U32(1), U32(1)],
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuTensorMapEncodeTiled (prefill) failed: {result}")

    desc = torch.tensor(
        [int(word) for word in tensor_map.opaque],
        dtype=torch.uint64,
        device=k_quant_bytes.device,
    )
    desc_ptrs = torch.tensor(
        [int(desc.data_ptr())], dtype=torch.int64, device=k_quant_bytes.device
    )
    return desc, desc_ptrs


def _get_cached_contiguous_k_tma_descriptor_prefill(
    k_quant_bytes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        int(k_quant_bytes.data_ptr()),
        tuple(k_quant_bytes.shape),
        tuple(k_quant_bytes.stride()),
        str(k_quant_bytes.dtype),
        (k_quant_bytes.device.type, k_quant_bytes.device.index),
        "prefill",
    )
    cache = getattr(_get_cached_contiguous_k_tma_descriptor_prefill, "_cache", None)
    if cache is None:
        cache = OrderedDict()
        setattr(_get_cached_contiguous_k_tma_descriptor_prefill, "_cache", cache)
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached
    desc = _encode_contiguous_k_tma_descriptor_prefill(k_quant_bytes)
    cache[key] = desc
    if len(cache) > _CONTIGUOUS_TMA_DESC_CACHE_SIZE:
        cache.popitem(last=False)
    return desc


def _encode_contiguous_k_tma_descriptor_prefill512(
    k_quant_bytes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TMA descriptor for the BK=512 prefill kernel's 256-row load subtiles."""
    if k_quant_bytes.ndim != 2 or k_quant_bytes.shape[1] != _INDEX_HEAD_DIM:
        raise ValueError(
            f"k_quant_bytes must have shape (rows, {_INDEX_HEAD_DIM}), got {tuple(k_quant_bytes.shape)}"
        )
    if k_quant_bytes.dtype != torch.uint8:
        raise TypeError(
            f"k_quant_bytes must be dtype torch.uint8, got {k_quant_bytes.dtype}"
        )

    U64 = cuda.cuuint64_t
    U32 = cuda.cuuint32_t
    row_bytes = int(k_quant_bytes.stride(0)) * k_quant_bytes.element_size()
    base_ptr = int(k_quant_bytes.data_ptr())
    total_rows = int(k_quant_bytes.shape[0])

    result, tensor_map = cuda.cuTensorMapEncodeTiled(
        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        base_ptr,
        [U64(_INDEX_HEAD_DIM), U64(total_rows)],
        [U64(row_bytes)],
        [U32(_INDEX_HEAD_DIM), U32(_PREFILL_BLOCK_K)],
        [U32(1), U32(1)],
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuTensorMapEncodeTiled (prefill512) failed: {result}")

    desc = torch.tensor(
        [int(word) for word in tensor_map.opaque],
        dtype=torch.uint64,
        device=k_quant_bytes.device,
    )
    desc_ptrs = torch.tensor(
        [int(desc.data_ptr())], dtype=torch.int64, device=k_quant_bytes.device
    )
    return desc, desc_ptrs


def _get_cached_contiguous_k_tma_descriptor_prefill512(
    k_quant_bytes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        int(k_quant_bytes.data_ptr()),
        tuple(k_quant_bytes.shape),
        tuple(k_quant_bytes.stride()),
        str(k_quant_bytes.dtype),
        (k_quant_bytes.device.type, k_quant_bytes.device.index),
        "prefill512",
    )
    cache = getattr(_get_cached_contiguous_k_tma_descriptor_prefill512, "_cache", None)
    if cache is None:
        cache = OrderedDict()
        setattr(_get_cached_contiguous_k_tma_descriptor_prefill512, "_cache", cache)
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached
    desc = _encode_contiguous_k_tma_descriptor_prefill512(k_quant_bytes)
    cache[key] = desc
    if len(cache) > _CONTIGUOUS_TMA_DESC_CACHE_SIZE:
        cache.popitem(last=False)
    return desc


@cute.jit
def _issue_contiguous_k_tma_copy(
    load_tma,
    producer_state,
    mbar_ptr,
    expected_bytes,
    k_tile_idx,
):
    full_mbar_ptr = mbar_ptr + producer_state.index
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(full_mbar_ptr, expected_bytes)
    load_tma(
        src_idx=k_tile_idx, dst_idx=producer_state.index, tma_bar_ptr=full_mbar_ptr
    )


@cute.jit
def _issue_contiguous_k_tma_copy_pair(
    load_tma0,
    load_tma1,
    producer_state,
    mbar_ptr,
    expected_bytes,
    k_tile_idx0,
    k_tile_idx1,
):
    full_mbar_ptr = mbar_ptr + producer_state.index
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(full_mbar_ptr, expected_bytes)
    load_tma0(
        src_idx=k_tile_idx0, dst_idx=producer_state.index, tma_bar_ptr=full_mbar_ptr
    )
    load_tma1(
        src_idx=k_tile_idx1, dst_idx=producer_state.index, tma_bar_ptr=full_mbar_ptr
    )


@cute.jit
def _permuted_offset_128b(row_idx, vec_idx, row_stride_128b):
    return row_idx * row_stride_128b + (vec_idx ^ (row_idx % 8))


@cute.jit
def _smem_addr_from_b128_offset(base_addr: Int32, offset_128b):
    return base_addr + Int32(offset_128b * 16)


@cute.jit
def _advance_offset_by_row_128b(offset_128b, step_size, row_stride_128b):
    return offset_128b + step_size * row_stride_128b


@cute.jit
def _advance_offset_by_column_128b_2(offset_128b, step_idx):
    xor_term = Int32(0x2) + (Int32(0x4) if step_idx % 2 == 1 else Int32(0))
    extra = Int32(8) if step_idx % 4 == 3 else Int32(0)
    return (offset_128b ^ xor_term) + extra


@cute.jit
def _zero_score_frag(score_frag: cute.Tensor) -> None:
    for reg_id in cutlass.range_constexpr(8):
        score_frag[0, 0, reg_id] = Float32(0.0)


@cute.jit
def _pack_q_mxfp8_reg_global(
    q_u32: cute.Tensor,
    head_idx: Int32,
    abs_row: Int32,
    col_pair_base: Int32,
    valid_q_rows: Int32,
) -> Uint32:
    """Pack 4 FP8 bytes from global q_u32 into one MMA register word.

    Original smem path accessed s_q_bytes[row, col_pair_base+{0,1,8,9}].
    Global view is u32, so we read two u32 values and extract the same
    4 bytes via shifts.  Uses cutlass.select_ to avoid early return
    (not allowed in CuTe JIT functions).
    """
    u32_idx_lo = col_pair_base // Int32(4)
    u32_idx_hi = u32_idx_lo + Int32(2)
    byte_shift = (col_pair_base % Int32(4)) * Int32(8)
    lo = (
        Uint32(q_u32[abs_row, head_idx, u32_idx_lo])
        if abs_row < valid_q_rows
        else Uint32(0)
    )
    hi = (
        Uint32(q_u32[abs_row, head_idx, u32_idx_hi])
        if abs_row < valid_q_rows
        else Uint32(0)
    )
    lo_half = (lo >> byte_shift) & Uint32(0xFFFF)
    hi_half = ((hi >> byte_shift) & Uint32(0xFFFF)) << Int32(16)
    return lo_half | hi_half


@cute.jit
def _literal_qk_mma_into_sfrag_mxfp8_raw(
    s_frag: cute.Tensor,
    q_u32: cute.Tensor,
    head_idx: Int32,
    q_tile_base: Int32,
    valid_q_rows: Int32,
    k_base_addr: Int32,
    lane,
    warp_q_idx,
    warp_kv_idx,
    row_base,
    num_mma_kv,
    num_mma_d_qk,
    upcast_stride_k,
):
    # This helper is decode-only: its sole call site maps one 16-row Q MMA
    # tile per warp.  Keeping that invariant structural lets the two Q-row
    # pointers stay materialized across the unrolled K dimension without
    # silently aliasing a future second Q tile.
    num_mma_q = 1
    unit_scale = Uint32(0x7F7F7F7F)
    group_id = lane // Int32(4)
    thread_id_in_group = lane % Int32(4)
    row_base_q = warp_q_idx * Int32(16)
    abs_row_0 = q_tile_base + row_base_q + group_id
    abs_row_8 = abs_row_0 + Int32(8)
    q_row_0 = cute.make_tensor(
        q_u32.iterator + cute.crd2idx((abs_row_0, head_idx, Int32(0)), q_u32.layout),
        cute.make_layout((_FP8_ROW_U32,), stride=(1,)),
    )
    q_row_8 = cute.make_tensor(
        q_u32.iterator + cute.crd2idx((abs_row_8, head_idx, Int32(0)), q_u32.layout),
        cute.make_layout((_FP8_ROW_U32,), stride=(1,)),
    )
    k_offset = _permuted_offset_128b(
        row_base
        + warp_kv_idx * num_mma_kv * Int32(16)
        + Int32(8) * (lane // Int32(16))
        + lane % Int32(8),
        (lane % Int32(16)) // Int32(8),
        upcast_stride_k,
    )
    for mma_pair in cutlass.range_constexpr(num_mma_d_qk // 2):
        q_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        # RAW fragment order on BOTH operands: each Q fragment register is one
        # aligned u32 read (bytes {4t..4t+3} of the k-half), so the byte
        # gather + shift chain and the K-side 16b->8b swizzles both vanish.
        u32_lo = Int32(mma_pair * 8) + thread_id_in_group
        u32_hi = u32_lo + Int32(4)
        for mma_q in cutlass.range_constexpr(num_mma_q):
            q_regs[mma_q, 0] = (
                Uint32(q_row_0[u32_lo]) if abs_row_0 < valid_q_rows else Uint32(0)
            )
            q_regs[mma_q, 1] = (
                Uint32(q_row_8[u32_lo]) if abs_row_8 < valid_q_rows else Uint32(0)
            )
            q_regs[mma_q, 2] = (
                Uint32(q_row_0[u32_hi]) if abs_row_0 < valid_q_rows else Uint32(0)
            )
            q_regs[mma_q, 3] = (
                Uint32(q_row_8[u32_hi]) if abs_row_8 < valid_q_rows else Uint32(0)
            )

        k_offset_cur = k_offset
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            b0_k0, b1_k0 = ldmatrix_m8n8x4_left_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            b0_k1, b1_k1 = ldmatrix_m8n8x4_right_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            k_offset_cur = _advance_offset_by_row_128b(
                k_offset_cur, Int32(16), upcast_stride_k
            )

            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    s_frag[mma_q, mma_kv, 0],
                    s_frag[mma_q, mma_kv, 1],
                    s_frag[mma_q, mma_kv, 2],
                    s_frag[mma_q, mma_kv, 3],
                    q_regs[mma_q, 0],
                    q_regs[mma_q, 1],
                    q_regs[mma_q, 2],
                    q_regs[mma_q, 3],
                    b0_k0,
                    b0_k1,
                    unit_scale,
                    unit_scale,
                )
                d4, d5, d6, d7 = mxfp8_mma_m16n8k32_f32_e4m3(
                    s_frag[mma_q, mma_kv, 4],
                    s_frag[mma_q, mma_kv, 5],
                    s_frag[mma_q, mma_kv, 6],
                    s_frag[mma_q, mma_kv, 7],
                    q_regs[mma_q, 0],
                    q_regs[mma_q, 1],
                    q_regs[mma_q, 2],
                    q_regs[mma_q, 3],
                    b1_k0,
                    b1_k1,
                    unit_scale,
                    unit_scale,
                )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7

        k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_pair) - Int32(
            num_mma_kv * Int32(16) * upcast_stride_k
        )


class SparseNSAContiguousLogitsKernel:
    """Ragged logits kernel with Q and weights read from global memory.

    Phase 1+2+3: Q/weights from global, warp-ballot liveness, TMA SWIZZLE_128B (no k_linear/repack).
    """

    def __init__(
        self,
        *,
        tiled_output: bool = False,
        score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
    ):
        self._tiled_output = tiled_output
        self._score_mode = int(score_mode)

    @cute.jit
    def __call__(
        self,
        q_u32: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        k_scales: cute.Tensor,
        k_start: cute.Tensor,
        k_end: cute.Tensor,
        logits_out: cute.Tensor,
        tile_logits: cute.Tensor,
        valid_q_rows: Int32,
        valid_k_rows: Int32,
        k_tile_offset: Int32,
        output_num_k_tiles: Int32,
        write_invalid_logits: Int32,
        stream: cuda.CUstream,
    ):
        k_tma_source = _make_contiguous_k_tma_source(k_quant_bytes)
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_tma_source,
            _make_contiguous_k_tma_smem_layout(_BLOCK_K),
            (_BLOCK_K, _INDEX_HEAD_DIM),
            1,
        )
        self.kernel(
            q_u32,
            weights,
            k_quant_bytes,
            tma_tensor_k,
            k_tma_desc_ptrs,
            k_scales,
            k_start,
            k_end,
            logits_out,
            tile_logits,
            valid_q_rows,
            valid_k_rows,
            k_tile_offset,
            output_num_k_tiles,
            write_invalid_logits,
            tma_atom_k,
        ).launch(
            grid=(
                (valid_q_rows + _BLOCK_Q - 1) // _BLOCK_Q,
                output_num_k_tiles,
                1,
            ),
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_u32: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_tensor: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        k_scales: cute.Tensor,
        k_start: cute.Tensor,
        k_end: cute.Tensor,
        logits_out: cute.Tensor,
        tile_logits: cute.Tensor,
        valid_q_rows: Int32,
        valid_k_rows: Int32,
        k_tile_offset: Int32,
        output_num_k_tiles: Int32,
        write_invalid_logits: Int32,
        tma_atom_k: cute.CopyAtom,
    ):
        tx, _, _ = cute.arch.thread_idx()
        q_tile_idx, local_k_tile_idx, _ = cute.arch.block_idx()
        k_tile_idx = local_k_tile_idx + Int32(k_tile_offset)
        lane = tx % Int32(_WARP_THREADS)
        warp_idx = tx // Int32(_WARP_THREADS)
        warp_q_idx = warp_idx // Int32(_WARPS_K)
        warp_k_idx = warp_idx - warp_q_idx * Int32(_WARPS_K)

        q_tile_base = q_tile_idx * Int32(_BLOCK_Q)
        k_tile_base = k_tile_idx * Int32(_BLOCK_K)
        valid_q_rows = Int32(valid_q_rows)
        k_total_rows = Int32(valid_k_rows)
        num_heads = Int32(q_u32.shape[1])

        smem = cutlass.utils.SmemAllocator()

        SharedStorage = get_sparse_nsa_contiguous_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        mbar_ptr_k = storage.mbar_ptr_k.data_ptr()
        k_perm_base_addr = shared_ptr_to_u32(storage.k_perm.data_ptr())
        s_k_perm_bytes = storage.k_perm.get_tensor(
            cute.make_layout((_BLOCK_K * _INDEX_HEAD_DIM,), stride=(1,))
        )
        s_k_tma_stage = cute.make_tensor(
            s_k_perm_bytes.iterator,
            _make_contiguous_k_tma_smem_stage_layout(_BLOCK_K),
        )
        load_k_tma, _, _ = cute_copy.tma_get_copy_fn(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.local_tile(k_tma_tensor, (_BLOCK_K, _INDEX_HEAD_DIM), (None, 0)),
            s_k_tma_stage,
        )
        s_scales = storage.scales.get_tensor(cute.make_layout((_BLOCK_K,), stride=(1,)))
        s_k_start = storage.k_start.get_tensor(
            cute.make_layout((_BLOCK_Q,), stride=(1,))
        )
        s_k_end = storage.k_end.get_tensor(cute.make_layout((_BLOCK_Q,), stride=(1,)))
        s_tile_live = storage.tile_live.get_tensor(cute.make_layout((1,), stride=(1,)))

        if tx == 0:
            cute.arch.mbarrier_init(mbar_ptr_k, Int32(1))
        row_linear = tx
        while row_linear < Int32(_BLOCK_Q):
            q_row = q_tile_base + row_linear
            s_k_start[row_linear] = (
                Int32(k_start[q_row]) if q_row < valid_q_rows else Int32(0)
            )
            s_k_end[row_linear] = (
                Int32(k_end[q_row]) if q_row < valid_q_rows else Int32(0)
            )
            row_linear += Int32(_THREADS_PER_CTA)
        cute.arch.sync_threads()

        # Phase 2: Parallel liveness via warp ballot
        # Warp 0 threads (tx 0-31) each check one Q-row; ballot
        # combines all 32 predicates in one hardware instruction.
        if warp_idx == Int32(0):
            tile_k_end = k_tile_base + Int32(_BLOCK_K)
            if tile_k_end > k_total_rows:
                tile_k_end = k_total_rows
            row_start = Int32(s_k_start[tx])
            row_end = Int32(s_k_end[tx])
            row_live = (
                (row_end > k_tile_base)
                & (row_start < tile_k_end)
                & (k_tile_base < tile_k_end)
            )
            ballot = cute.arch.vote_ballot_sync(Boolean(row_live))
            if tx == Int32(0):
                s_tile_live[Int32(0)] = ballot
        cute.arch.sync_threads()
        if s_tile_live[Int32(0)] != Int32(0):
            producer_state = cute_pipeline.PipelineStateSimple(1, Int32(0))
            consumer_state = cute_pipeline.PipelineStateSimple(1, Int32(0))
            if warp_idx == Int32(0):
                cpasync.prefetch_descriptor(tma_atom_k)
            if warp_idx == Int32(0):
                _issue_contiguous_k_tma_copy(
                    load_k_tma,
                    producer_state,
                    mbar_ptr_k,
                    Int32(_BLOCK_K * _INDEX_HEAD_DIM),
                    k_tile_idx,
                )
            cute.arch.mbarrier_wait(
                mbar_ptr_k + consumer_state.index,
                phase=consumer_state.phase,
            )
            cute.arch.sync_threads()

            scale_linear = tx
            while scale_linear < Int32(_BLOCK_K):
                s_scales[scale_linear] = Float32(k_scales[k_tile_base + scale_linear])
                scale_linear += Int32(_THREADS_PER_CTA)
            cute.arch.sync_threads()

            frag_layout = cute.make_layout((1, 1, 8), stride=(8, 8, 1))
            acc_frag = cute.make_rmem_tensor(frag_layout, Float32)
            _zero_score_frag(acc_frag)

            head_idx = Int32(0)
            while head_idx < num_heads:
                score_frag = cute.make_rmem_tensor(frag_layout, Float32)
                _zero_score_frag(score_frag)
                _literal_qk_mma_into_sfrag_mxfp8_raw(
                    score_frag,
                    q_u32,
                    head_idx,
                    q_tile_base,
                    valid_q_rows,
                    k_perm_base_addr,
                    lane,
                    warp_q_idx,
                    warp_k_idx,
                    Int32(0),
                    Int32(1),
                    Int32(_INDEX_HEAD_DIM // 16),
                    Int32(_FP8_ROW_VECS),
                )
                lane_group = lane // Int32(4)
                q_local_0 = warp_q_idx * Int32(16) + lane_group
                q_local_8 = q_local_0 + Int32(8)
                w_val_0 = (
                    Float32(weights[q_tile_base + q_local_0, head_idx])
                    if q_tile_base + q_local_0 < valid_q_rows
                    else Float32(0.0)
                )
                w_val_8 = (
                    Float32(weights[q_tile_base + q_local_8, head_idx])
                    if q_tile_base + q_local_8 < valid_q_rows
                    else Float32(0.0)
                )
                for reg_id in cutlass.range_constexpr(8):
                    row_slot = (reg_id % 4) // 2
                    q_local = warp_q_idx * Int32(16) + lane_group + Int32(8 * row_slot)
                    if q_local < Int32(_BLOCK_Q):
                        w_val = w_val_0 if row_slot == 0 else w_val_8
                        if cutlass.const_expr(
                            self._score_mode == IndexerScoreMode.MSA_BILINEAR
                        ):
                            acc_frag[0, 0, reg_id] = Float32(
                                acc_frag[0, 0, reg_id]
                                + score_frag[0, 0, reg_id] * w_val
                            )
                        else:
                            acc_frag[0, 0, reg_id] = Float32(
                                acc_frag[0, 0, reg_id]
                                + attention_ops.fmax(
                                    score_frag[0, 0, reg_id], Float32(0.0)
                                )
                                * w_val
                            )
                head_idx += Int32(1)

            lane_group = lane // Int32(4)
            lane_pair_base = Int32(2) * (lane % Int32(4))
            for reg_id in cutlass.range_constexpr(8):
                row_slot = (reg_id % 4) // 2
                q_local = warp_q_idx * Int32(16) + lane_group + Int32(8 * row_slot)
                k_local = (
                    warp_k_idx * Int32(16)
                    + lane_pair_base
                    + Int32(8 * (reg_id // 4))
                    + Int32(reg_id % 2)
                )
                q_row = q_tile_base + q_local
                k_row = k_tile_base + k_local
                if (
                    q_local < Int32(_BLOCK_Q)
                    and k_local < Int32(_BLOCK_K)
                    and q_row < valid_q_rows
                    and k_row < k_total_rows
                ):
                    row_start = Int32(s_k_start[q_local])
                    row_end = Int32(s_k_end[q_local])
                    if k_row >= row_start and k_row < row_end:
                        logits_out[q_row, k_row] = Float32(
                            acc_frag[0, 0, reg_id] * s_scales[k_local]
                        )
                    elif write_invalid_logits != Int32(0):
                        logits_out[q_row, k_row] = Float32(-Float32.inf)
        else:
            if write_invalid_logits != Int32(0):
                lane_group = lane // Int32(4)
                lane_pair_base = Int32(2) * (lane % Int32(4))
                for reg_id in cutlass.range_constexpr(8):
                    row_slot = (reg_id % 4) // 2
                    q_local = warp_q_idx * Int32(16) + lane_group + Int32(8 * row_slot)
                    k_local = (
                        warp_k_idx * Int32(16)
                        + lane_pair_base
                        + Int32(8 * (reg_id // 4))
                        + Int32(reg_id % 2)
                    )
                    q_row = q_tile_base + q_local
                    k_row = k_tile_base + k_local
                    if (
                        q_local < Int32(_BLOCK_Q)
                        and k_local < Int32(_BLOCK_K)
                        and q_row < valid_q_rows
                        and k_row < k_total_rows
                    ):
                        logits_out[q_row, k_row] = Float32(-Float32.inf)


def cp_async_128b_pred(
    smem_addr: Int32, gmem_addr: Int64, predicate: Int32, *, loc=None, ip=None
):
    """Async 16B global→shared copy, predicated. Both addresses must be 16B-aligned."""
    llvm.inline_asm(
        None,
        [
            Int32(predicate).ir_value(loc=loc, ip=ip),
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, $0, 0;\n"
        " @p cp.async.cg.shared.global.L2::128B [$1], [$2], 16;\n"
        "}",
        "r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def ld_shared_u32(smem_addr: Int32, *, loc=None, ip=None) -> Uint32:
    """Load 32 bits from shared memory."""
    result = llvm.inline_asm(
        T.i32(),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.u32 {$0}, [$1];",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


def ld_shared_f32(smem_addr: Int32, *, loc=None, ip=None) -> Float32:
    """Load 32 bits from shared memory as Float32."""
    result = llvm.inline_asm(
        T.f32(),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.f32 {$0}, [$1];",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(result)


def st_shared_v4_f32(
    smem_addr: Int32,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
):
    """Store 128 bits (4 x f32) to shared memory. smem_addr is a u32 shared-memory address."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def get_sparse_nsa_prefill512_shared_storage_cls(q_heads_batch: int):
    q_heads_batch = int(q_heads_batch)

    class SharedStorage:
        pass

    SharedStorage.__annotations__ = {
        "mbar_ptr_k": cute.struct.MemRange[cutlass.Int64, 1],
        "tile_live": cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, 1],
            16,
        ],
        "tile_full": cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, 1],
            16,
        ],
        "k_start": cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, _PREFILL512_BLOCK_Q],
            16,
        ],
        "k_end": cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, _PREFILL512_BLOCK_Q],
            16,
        ],
        "scales": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _PREFILL512_BLOCK_K],
            16,
        ],
        "k_perm": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, _PREFILL512_BLOCK_K * _INDEX_HEAD_DIM],
            1024,
        ],
        "q_bytes_smem": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Uint8,
                _PREFILL_Q_STAGE_ROWS * _PREFILL_Q_STAGE_COLS * q_heads_batch,
            ],
            1024,
        ],
    }

    return cute.struct(SharedStorage)


@cute.jit
def _pack_q_mxfp8_reg_smem_ptr(
    q_smem_base: Int32,
    row_local: Int32,
    col_pair_base: Int32,
) -> Uint32:
    """Pack 4 FP8 bytes from swizzled smem into one MMA register word.

    Reads bytes at [row_local, col_pair_base+0], [row_local, col_pair_base+1],
    [row_local, col_pair_base+8], [row_local, col_pair_base+9].
    Uses _permuted_offset_128b to match the swizzled write layout.
    """
    vec_idx = col_pair_base // Int32(16)
    word_pair = (col_pair_base // Int32(4)) % Int32(4)
    byte_shift = (col_pair_base % Int32(4)) * Int32(8)
    q_rs = Int32(_PREFILL_Q_STAGE_COLS // 16)
    addr = q_smem_base + _permuted_offset_128b(row_local, vec_idx, q_rs) * Int32(16)
    v0, v1, v2, v3 = ld_shared_v4_u32(addr)
    lo = v0 if word_pair == Int32(0) else v1
    hi = v2 if word_pair == Int32(0) else v3
    lo_half = (lo >> byte_shift) & Uint32(0xFFFF)
    hi_half = ((hi >> byte_shift) & Uint32(0xFFFF)) << Int32(16)
    return lo_half | hi_half


@cute.jit
def _prefill_qk_mma_from_smem_q(
    s_frag: cute.Tensor,
    q_smem_base: Int32,
    k_base_addr: Int32,
    lane,
    warp_q_idx,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_qk,
    upcast_stride_k,
) -> None:
    """QK MMA using Q from smem (via raw pointer) instead of global memory."""
    unit_scale = Uint32(0x7F7F7F7F)
    k_offset = _permuted_offset_128b(
        row_base
        + warp_kv_idx * num_mma_kv * Int32(16)
        + Int32(8) * (lane // Int32(16))
        + lane % Int32(8),
        (lane % Int32(16)) // Int32(8),
        upcast_stride_k,
    )
    q_rs = Int32(_PREFILL_Q_STAGE_COLS // 16)
    for mma_pair in cutlass.range_constexpr(num_mma_d_qk // 2):
        q_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        # RAW fragment order on BOTH operands (see the paged stream scorer):
        # Q comes straight off ldmatrix.x4 over the permuted Q stage (one
        # instruction per 16-row tile instead of four 16B-load+extract packs),
        # and the K halves skip the 16b->8b fragment swizzles entirely.
        q_row_in_tile = lane & Int32(15)
        q_half = lane >> Int32(4)
        for mma_q in cutlass.range_constexpr(num_mma_q):
            row_base_q = warp_q_idx * Int32(16) + mma_q * Int32(16)
            q_addr = q_smem_base + _permuted_offset_128b(
                row_base_q + q_row_in_tile,
                Int32(2 * mma_pair) + q_half,
                q_rs,
            ) * Int32(16)
            qa0, qa1, qa2, qa3 = ldmatrix_m8n8x4_b16(q_addr)
            q_regs[mma_q, 0] = qa0
            q_regs[mma_q, 1] = qa1
            q_regs[mma_q, 2] = qa2
            q_regs[mma_q, 3] = qa3

        k_offset_cur = k_offset
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            b0_k0, b1_k0 = ldmatrix_m8n8x4_left_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            b0_k1, b1_k1 = ldmatrix_m8n8x4_right_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            k_offset_cur = _advance_offset_by_row_128b(
                k_offset_cur, Int32(16), upcast_stride_k
            )

            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    s_frag[mma_q, mma_kv, 0],
                    s_frag[mma_q, mma_kv, 1],
                    s_frag[mma_q, mma_kv, 2],
                    s_frag[mma_q, mma_kv, 3],
                    q_regs[mma_q, 0],
                    q_regs[mma_q, 1],
                    q_regs[mma_q, 2],
                    q_regs[mma_q, 3],
                    b0_k0,
                    b0_k1,
                    unit_scale,
                    unit_scale,
                )
                d4, d5, d6, d7 = mxfp8_mma_m16n8k32_f32_e4m3(
                    s_frag[mma_q, mma_kv, 4],
                    s_frag[mma_q, mma_kv, 5],
                    s_frag[mma_q, mma_kv, 6],
                    s_frag[mma_q, mma_kv, 7],
                    q_regs[mma_q, 0],
                    q_regs[mma_q, 1],
                    q_regs[mma_q, 2],
                    q_regs[mma_q, 3],
                    b1_k0,
                    b1_k1,
                    unit_scale,
                    unit_scale,
                )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7

        k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_pair) - Int32(
            num_mma_kv * Int32(16) * upcast_stride_k
        )


class SparseNSAContiguousLogitsPrefillKernel:
    """Prefill-specialized contiguous logits kernel with _PREFILL_BLOCK_K=256.

    Uses 2 Q-warps x 4 K-warps = 256 threads. Each K-warp covers 64 K-rows
    (num_mma_kv=4), so the CTA processes 256 K rows per tile. This quarters the
    K-CTA count versus the decode tile, reducing redundant Q reads from global
    memory. Q is staged into shared memory per head to avoid redundant scattered
    global reads.
    """

    def __init__(
        self,
        *,
        tiled_output: bool = False,
        score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
        block_score_output: bool = False,
    ):
        self._tiled_output = tiled_output
        self._score_mode = int(score_mode)
        self._block_score_output = bool(block_score_output)

    @cute.jit
    def __call__(
        self,
        q_u32: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        k_scales: cute.Tensor,
        k_start: cute.Tensor,
        k_end: cute.Tensor,
        logits_out: cute.Tensor,
        tile_logits: cute.Tensor,
        block_scores: cute.Tensor,
        valid_q_rows: Int32,
        valid_k_rows: Int32,
        num_blocks_out: Int32,
        k_tile_offset: Int32,
        output_num_k_tiles: Int32,
        write_invalid_logits: Int32,
        stream: cuda.CUstream,
    ):
        k_tma_source = _make_contiguous_k_tma_source(k_quant_bytes)
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_tma_source,
            _make_contiguous_k_tma_smem_layout(_PREFILL_BLOCK_K),
            (_PREFILL_BLOCK_K, _INDEX_HEAD_DIM),
            1,
        )
        self.kernel(
            q_u32,
            weights,
            k_quant_bytes,
            tma_tensor_k,
            k_tma_desc_ptrs,
            k_scales,
            k_start,
            k_end,
            logits_out,
            tile_logits,
            block_scores,
            valid_q_rows,
            valid_k_rows,
            num_blocks_out,
            k_tile_offset,
            output_num_k_tiles,
            write_invalid_logits,
            tma_atom_k,
        ).launch(
            grid=(
                (valid_q_rows + _PREFILL_BLOCK_Q - 1) // _PREFILL_BLOCK_Q,
                output_num_k_tiles,
                1,
            ),
            block=[_PREFILL_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_u32: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_tensor: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        k_scales: cute.Tensor,
        k_start: cute.Tensor,
        k_end: cute.Tensor,
        logits_out: cute.Tensor,
        tile_logits: cute.Tensor,
        block_scores: cute.Tensor,
        valid_q_rows: Int32,
        valid_k_rows: Int32,
        num_blocks_out: Int32,
        k_tile_offset: Int32,
        output_num_k_tiles: Int32,
        write_invalid_logits: Int32,
        tma_atom_k: cute.CopyAtom,
    ):
        tx, _, _ = cute.arch.thread_idx()
        q_tile_idx, local_k_tile_idx, _ = cute.arch.block_idx()
        k_tile_idx = local_k_tile_idx + Int32(k_tile_offset)
        lane = tx % Int32(_WARP_THREADS)
        warp_idx = tx // Int32(_WARP_THREADS)
        warp_k_idx = warp_idx % Int32(_PREFILL_WARPS_K)
        warp_q_idx = warp_idx // Int32(_PREFILL_WARPS_K)

        q_tile_base = q_tile_idx * Int32(_PREFILL_BLOCK_Q)
        k_tile_base = k_tile_idx * Int32(_PREFILL_BLOCK_K)
        valid_q_rows = Int32(valid_q_rows)
        k_total_rows = Int32(valid_k_rows)
        TILED_OUTPUT = cutlass.const_expr(self._tiled_output)
        BLOCK_SCORE_OUTPUT = cutlass.const_expr(self._block_score_output)
        num_heads = Int32(q_u32.shape[1])

        smem = cutlass.utils.SmemAllocator()

        SharedStorage = get_sparse_nsa_contiguous_prefill_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        mbar_ptr_k = storage.mbar_ptr_k.data_ptr()
        k_perm_base_addr = shared_ptr_to_u32(storage.k_perm.data_ptr())
        s_k_perm_bytes = storage.k_perm.get_tensor(
            cute.make_layout((_PREFILL_BLOCK_K * _INDEX_HEAD_DIM,), stride=(1,))
        )
        s_k_tma_stage = cute.make_tensor(
            s_k_perm_bytes.iterator,
            _make_contiguous_k_tma_smem_stage_layout(_PREFILL_BLOCK_K),
        )
        load_k_tma, _, _ = cute_copy.tma_get_copy_fn(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.local_tile(
                k_tma_tensor, (_PREFILL_BLOCK_K, _INDEX_HEAD_DIM), (None, 0)
            ),
            s_k_tma_stage,
        )
        s_scales = storage.scales.get_tensor(
            cute.make_layout((_PREFILL_BLOCK_K,), stride=(1,))
        )
        s_k_start = storage.k_start.get_tensor(
            cute.make_layout((_PREFILL_BLOCK_Q,), stride=(1,))
        )
        s_k_end = storage.k_end.get_tensor(
            cute.make_layout((_PREFILL_BLOCK_Q,), stride=(1,))
        )
        s_tile_live = storage.tile_live.get_tensor(cute.make_layout((1,), stride=(1,)))
        s_block_partial = storage.block_partial.get_tensor(
            cute.make_layout(
                (_PREFILL_WARPS_K, _PREFILL_BLOCK_Q),
                stride=(_PREFILL_BLOCK_Q, 1),
            )
        )

        if tx == 0:
            cute.arch.mbarrier_init(mbar_ptr_k, Int32(1))
        row_linear = tx
        while row_linear < Int32(_PREFILL_BLOCK_Q):
            q_row = q_tile_base + row_linear
            s_k_start[row_linear] = (
                Int32(k_start[q_row]) if q_row < valid_q_rows else Int32(0)
            )
            s_k_end[row_linear] = (
                Int32(k_end[q_row]) if q_row < valid_q_rows else Int32(0)
            )
            row_linear += Int32(_PREFILL_THREADS_PER_CTA)
        cute.arch.sync_threads()

        # Single ballot liveness check (32 Q-rows, same as decode)
        if warp_idx == Int32(0):
            tile_k_end = k_tile_base + Int32(_PREFILL_BLOCK_K)
            if tile_k_end > k_total_rows:
                tile_k_end = k_total_rows
            row_start = Int32(s_k_start[tx])
            row_end = Int32(s_k_end[tx])
            row_live = (
                (row_end > k_tile_base)
                & (row_start < tile_k_end)
                & (k_tile_base < tile_k_end)
            )
            ballot = cute.arch.vote_ballot_sync(Boolean(row_live))
            if tx == Int32(0):
                s_tile_live[Int32(0)] = ballot
        cute.arch.sync_threads()

        if s_tile_live[Int32(0)] != Int32(0):
            # q_bytes_smem now batches _PREFILL_Q_HEADS_BATCH Q heads (32KB for 8),
            # w_smem sits after all Q buffers.
            q_smem_base = k_perm_base_addr + Int32(_PREFILL_BLOCK_K * _INDEX_HEAD_DIM)
            w_smem_base = q_smem_base + Int32(
                _PREFILL_Q_STAGE_BYTES * _PREFILL_Q_HEADS_BATCH
            )
            # TMA load K-tile (256 rows x 128 bytes = 32 KB)
            producer_state = cute_pipeline.PipelineStateSimple(1, Int32(0))
            consumer_state = cute_pipeline.PipelineStateSimple(1, Int32(0))
            if warp_idx == Int32(0):
                cpasync.prefetch_descriptor(tma_atom_k)
            if warp_idx == Int32(0):
                _issue_contiguous_k_tma_copy(
                    load_k_tma,
                    producer_state,
                    mbar_ptr_k,
                    Int32(_PREFILL_BLOCK_K * _INDEX_HEAD_DIM),
                    k_tile_idx,
                )
            cute.arch.mbarrier_wait(
                mbar_ptr_k + consumer_state.index,
                phase=consumer_state.phase,
            )

            # Overlap: issue weight loads before TMA sync (no K dependency)
            w_linear = tx * Int32(4)
            while w_linear < Int32(_PREFILL_BLOCK_Q * _MAX_Q_HEADS):
                w_q_local = w_linear // num_heads
                w_head = w_linear % num_heads
                w_q_row = q_tile_base + w_q_local
                w0 = (
                    Float32(weights[w_q_row, w_head])
                    if (w_q_row < valid_q_rows and w_head < num_heads)
                    else Float32(0.0)
                )
                w1 = (
                    Float32(weights[w_q_row, w_head + Int32(1)])
                    if (w_q_row < valid_q_rows and w_head + Int32(1) < num_heads)
                    else Float32(0.0)
                )
                w2 = (
                    Float32(weights[w_q_row, w_head + Int32(2)])
                    if (w_q_row < valid_q_rows and w_head + Int32(2) < num_heads)
                    else Float32(0.0)
                )
                w3 = (
                    Float32(weights[w_q_row, w_head + Int32(3)])
                    if (w_q_row < valid_q_rows and w_head + Int32(3) < num_heads)
                    else Float32(0.0)
                )
                st_shared_v4_f32(
                    w_smem_base + w_linear * Int32(4),
                    w0,
                    w1,
                    w2,
                    w3,
                )
                w_linear += Int32(_PREFILL_THREADS_PER_CTA * 4)
            cute.arch.sync_threads()

            # Load scales (256 values) after TMA arrival
            scale_linear = tx
            while scale_linear < Int32(_PREFILL_BLOCK_K):
                s_scales[scale_linear] = Float32(k_scales[k_tile_base + scale_linear])
                scale_linear += Int32(_PREFILL_THREADS_PER_CTA)
            cute.arch.sync_threads()

            # Accumulator
            acc_layout = cute.make_layout(
                (1, _PREFILL_NUM_MMA_KV, 8), stride=(16, 8, 1)
            )
            acc_frag = cute.make_rmem_tensor(acc_layout, Float32)
            for mma_kv in cutlass.range_constexpr(_PREFILL_NUM_MMA_KV):
                for reg_id in cutlass.range_constexpr(8):
                    acc_frag[Int32(0), mma_kv, reg_id] = Float32(0.0)

            # Precompute weight smem offsets (only 2 distinct q_local values per thread).
            lane_group_pre = lane // Int32(4)
            q_local_rs0 = warp_q_idx * Int32(16) + lane_group_pre
            q_local_rs1 = q_local_rs0 + Int32(8)
            w_off_rs0 = q_local_rs0 * num_heads * Int32(4)
            w_off_rs1 = q_local_rs1 * num_heads * Int32(4)

            # Batched Q staging: stage _PREFILL_Q_HEADS_BATCH heads into smem at once,
            # then compute QK for all of them before moving to the next batch.
            threads_per_row = Int32(8)
            row_local = tx // threads_per_row
            col_group = tx % threads_per_row
            u32_col_base = col_group * Int32(4)
            q_row = q_tile_base + row_local
            in_bounds_pred = (
                Int32(1)
                if (row_local < Int32(_PREFILL_Q_STAGE_ROWS) and q_row < valid_q_rows)
                else Int32(0)
            )
            q_smem_stride = Int32(_PREFILL_Q_STAGE_BYTES)
            q_rs = Int32(_PREFILL_Q_STAGE_COLS // 16)
            thread_offset = _permuted_offset_128b(row_local, col_group, q_rs) * Int32(
                16
            )
            q_row_for_async = (
                q_row if row_local < Int32(_PREFILL_Q_STAGE_ROWS) else Int32(0)
            )

            batch_base_head = Int32(0)
            while batch_base_head < num_heads:
                # Stage batch of heads into contiguous smem regions.
                for block_offset in cutlass.range_constexpr(_PREFILL_Q_HEADS_BATCH):
                    head_in_batch = Int32(block_offset)
                    global_head = batch_base_head + head_in_batch
                    if global_head < num_heads:
                        gmem_u32_offset = (
                            q_row_for_async * num_heads + global_head
                        ) * Int32(_INDEX_HEAD_DIM // 4) + u32_col_base
                        cp_async_128b_pred(
                            q_smem_base + head_in_batch * q_smem_stride + thread_offset,
                            get_ptr_as_int64(q_u32, gmem_u32_offset),
                            in_bounds_pred,
                        )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                # Compute QK for each head in the batch (reads from stable smem).
                for block_offset in cutlass.range_constexpr(_PREFILL_Q_HEADS_BATCH):
                    head_idx = batch_base_head + Int32(block_offset)
                    if head_idx < num_heads:
                        curr_mma_base = (
                            q_smem_base + Int32(block_offset) * q_smem_stride
                        )

                        score_frag = cute.make_rmem_tensor(acc_layout, Float32)
                        for mma_kv in cutlass.range_constexpr(_PREFILL_NUM_MMA_KV):
                            for reg_id in cutlass.range_constexpr(8):
                                score_frag[Int32(0), mma_kv, reg_id] = Float32(0.0)
                        _prefill_qk_mma_from_smem_q(
                            score_frag,
                            curr_mma_base,
                            k_perm_base_addr,
                            lane,
                            warp_q_idx,
                            warp_k_idx,
                            Int32(0),
                            Int32(_PREFILL_NUM_MMA_Q),
                            Int32(_PREFILL_NUM_MMA_KV),
                            Int32(_INDEX_HEAD_DIM // 16),
                            Int32(_FP8_ROW_VECS),
                        )
                        # Accumulate per K-sub-tile (q_local always in [0,31], no bounds check needed)
                        lane_group = lane // Int32(4)
                        w_rs0 = ld_shared_f32(
                            w_smem_base + w_off_rs0 + head_idx * Int32(4)
                        )
                        w_rs1 = ld_shared_f32(
                            w_smem_base + w_off_rs1 + head_idx * Int32(4)
                        )
                        if BLOCK_SCORE_OUTPUT:
                            lane_pair_base = Int32(2) * (lane % Int32(4))
                            local_max_rs0 = Float32(-Float32.inf)
                            local_max_rs1 = Float32(-Float32.inf)
                            for mma_kv in cutlass.range_constexpr(_PREFILL_NUM_MMA_KV):
                                for reg_id in cutlass.range_constexpr(8):
                                    row_slot = (reg_id % 4) // 2
                                    q_local = (
                                        warp_q_idx * Int32(16)
                                        + lane_group
                                        + Int32(8 * row_slot)
                                    )
                                    k_local = (
                                        warp_k_idx * Int32(_PREFILL_NUM_MMA_KV * 16)
                                        + mma_kv * Int32(16)
                                        + lane_pair_base
                                        + Int32(8 * (reg_id // 4))
                                        + Int32(reg_id % 2)
                                    )
                                    q_row = q_tile_base + q_local
                                    k_row = k_tile_base + k_local
                                    row_start = Int32(s_k_start[q_local])
                                    row_end = Int32(s_k_end[q_local])
                                    if (
                                        q_row < valid_q_rows
                                        and k_row < k_total_rows
                                        and k_row >= row_start
                                        and k_row < row_end
                                    ):
                                        w_val = w_rs0 if row_slot == Int32(0) else w_rs1
                                        score = Float32(
                                            score_frag[Int32(0), mma_kv, reg_id]
                                            * w_val
                                            * s_scales[k_local]
                                        )
                                        if row_slot == Int32(0):
                                            local_max_rs0 = attention_ops.fmax(
                                                local_max_rs0,
                                                score,
                                            )
                                        else:
                                            local_max_rs1 = attention_ops.fmax(
                                                local_max_rs1,
                                                score,
                                            )
                            local_max_rs0 = _reduce_quad_max(local_max_rs0)
                            local_max_rs1 = _reduce_quad_max(local_max_rs1)
                            if lane % Int32(4) == Int32(0):
                                s_block_partial[warp_k_idx, q_local_rs0] = local_max_rs0
                                s_block_partial[warp_k_idx, q_local_rs1] = local_max_rs1
                            cute.arch.sync_threads()
                            if tx < Int32(64):
                                q_local_out = tx % Int32(_PREFILL_BLOCK_Q)
                                block_local = tx // Int32(_PREFILL_BLOCK_Q)
                                block_idx = k_tile_idx * Int32(2) + block_local
                                q_row_out = q_tile_base + q_local_out
                                v = attention_ops.fmax(
                                    s_block_partial[
                                        block_local * Int32(2), q_local_out
                                    ],
                                    s_block_partial[
                                        block_local * Int32(2) + Int32(1), q_local_out
                                    ],
                                )
                                if (
                                    q_row_out < valid_q_rows
                                    and block_idx < num_blocks_out
                                ):
                                    block_scores[head_idx, q_row_out, block_idx] = v
                            cute.arch.sync_threads()
                        else:
                            for mma_kv in cutlass.range_constexpr(_PREFILL_NUM_MMA_KV):
                                for reg_id in cutlass.range_constexpr(8):
                                    row_slot = (reg_id % 4) // 2
                                    q_local = (
                                        warp_q_idx * Int32(16)
                                        + lane_group
                                        + Int32(8 * row_slot)
                                    )
                                    w_val = w_rs0 if row_slot == Int32(0) else w_rs1
                                    if cutlass.const_expr(
                                        self._score_mode
                                        == IndexerScoreMode.MSA_BILINEAR
                                    ):
                                        acc_frag[Int32(0), mma_kv, reg_id] = Float32(
                                            acc_frag[Int32(0), mma_kv, reg_id]
                                            + score_frag[Int32(0), mma_kv, reg_id]
                                            * w_val
                                        )
                                    else:
                                        acc_frag[Int32(0), mma_kv, reg_id] = Float32(
                                            acc_frag[Int32(0), mma_kv, reg_id]
                                            + attention_ops.fmax(
                                                score_frag[Int32(0), mma_kv, reg_id],
                                                Float32(0.0),
                                            )
                                            * w_val
                                        )
                # The next batch reuses the same Q shared-memory slots.
                cute.arch.sync_threads()
                batch_base_head += Int32(_PREFILL_Q_HEADS_BATCH)

            # Write back — each K-warp covers NUM_MMA_KV * 16 K-rows
            lane_group = lane // Int32(4)
            lane_pair_base = Int32(2) * (lane % Int32(4))
            if BLOCK_SCORE_OUTPUT:
                pass
            elif TILED_OUTPUT:
                for mma_kv in cutlass.range_constexpr(_PREFILL_NUM_MMA_KV):
                    for reg_id in cutlass.range_constexpr(8):
                        row_slot = (reg_id % 4) // 2
                        q_local = (
                            warp_q_idx * Int32(16) + lane_group + Int32(8 * row_slot)
                        )
                        k_local = (
                            warp_k_idx * Int32(_PREFILL_NUM_MMA_KV * 16)
                            + mma_kv * Int32(16)
                            + lane_pair_base
                            + Int32(8 * (reg_id // 4))
                            + Int32(reg_id % 2)
                        )
                        if (
                            q_local < Int32(_PREFILL_BLOCK_Q)
                            and k_local < Int32(_PREFILL_BLOCK_K)
                            and q_tile_base + q_local < valid_q_rows
                        ):
                            _tile_id = (
                                q_tile_idx * Int32(output_num_k_tiles)
                                + local_k_tile_idx
                            )
                            _offset_in_tile = (
                                q_local * Int32(_PREFILL_BLOCK_K) + k_local
                            )
                            _flat_offset = (
                                _tile_id * Int32(_PREFILL_BLOCK_Q * _PREFILL_BLOCK_K)
                                + _offset_in_tile
                            )
                            tile_logits[_flat_offset] = Float32(
                                acc_frag[Int32(0), mma_kv, reg_id] * s_scales[k_local]
                            )
            else:
                for mma_kv in cutlass.range_constexpr(_PREFILL_NUM_MMA_KV):
                    for reg_id in cutlass.range_constexpr(8):
                        row_slot = (reg_id % 4) // 2
                        q_local = (
                            warp_q_idx * Int32(16) + lane_group + Int32(8 * row_slot)
                        )
                        k_local = (
                            warp_k_idx * Int32(_PREFILL_NUM_MMA_KV * 16)
                            + mma_kv * Int32(16)
                            + lane_pair_base
                            + Int32(8 * (reg_id // 4))
                            + Int32(reg_id % 2)
                        )
                        q_row = q_tile_base + q_local
                        k_row = k_tile_base + k_local
                        if (
                            q_local < Int32(_PREFILL_BLOCK_Q)
                            and k_local < Int32(_PREFILL_BLOCK_K)
                            and q_row < valid_q_rows
                            and k_row < k_total_rows
                        ):
                            row_start = Int32(s_k_start[q_local])
                            row_end = Int32(s_k_end[q_local])
                            if k_row >= row_start and k_row < row_end:
                                logits_out[q_row, k_row] = Float32(
                                    acc_frag[Int32(0), mma_kv, reg_id]
                                    * s_scales[k_local]
                                )
                            elif write_invalid_logits != Int32(0):
                                logits_out[q_row, k_row] = Float32(-Float32.inf)
        else:
            if BLOCK_SCORE_OUTPUT:
                pass  # dead tile; host -inf prefill covers block scores
            elif TILED_OUTPUT:
                pass  # dead tile; topk filters with k_start/lengths
            elif write_invalid_logits != Int32(0):
                lane_group = lane // Int32(4)
                lane_pair_base = Int32(2) * (lane % Int32(4))
                for mma_kv in cutlass.range_constexpr(_PREFILL_NUM_MMA_KV):
                    for reg_id in cutlass.range_constexpr(8):
                        row_slot = (reg_id % 4) // 2
                        q_local = (
                            warp_q_idx * Int32(16) + lane_group + Int32(8 * row_slot)
                        )
                        k_local = (
                            warp_k_idx * Int32(_PREFILL_NUM_MMA_KV * 16)
                            + mma_kv * Int32(16)
                            + lane_pair_base
                            + Int32(8 * (reg_id // 4))
                            + Int32(reg_id % 2)
                        )
                        q_row = q_tile_base + q_local
                        k_row = k_tile_base + k_local
                        if (
                            q_local < Int32(_PREFILL_BLOCK_Q)
                            and k_local < Int32(_PREFILL_BLOCK_K)
                            and q_row < valid_q_rows
                            and k_row < k_total_rows
                        ):
                            logits_out[q_row, k_row] = Float32(-Float32.inf)


class SparseNSAContiguousLogitsPrefillKernelBlockScores(
    SparseNSAContiguousLogitsPrefillKernel
):
    """Distinct compile identity for the MSA block-score specialization.

    The implementation and ABI intentionally remain inherited from the
    prefill logits kernel.  A concrete type prevents two semantically distinct
    compile specs from collapsing onto one generated CUDA entry-point symbol,
    which is required for exact per-launch resource accounting.
    """


class SparseNSAContiguousLogitsPrefill512Kernel:
    """Experimental prefill scorer with _PREFILL512_BLOCK_K=512.

    This halves the K-CTA count versus the BK=256 prefill scorer. To keep
    shared-memory use under the launch limit, it batches a small group of Q
    heads in shared memory.
    """

    def __init__(
        self,
        *,
        tiled_output: bool = False,
        q_heads_batch: int = _PREFILL512_Q_HEADS_BATCH,
        score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
    ):
        self._tiled_output = tiled_output
        self._q_heads_batch = int(q_heads_batch)
        self._score_mode = int(score_mode)

    @cute.jit
    def __call__(
        self,
        q_u32: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        k_scales: cute.Tensor,
        k_start: cute.Tensor,
        k_end: cute.Tensor,
        logits_out: cute.Tensor,
        tile_logits: cute.Tensor,
        valid_q_rows: Int32,
        valid_k_rows: Int32,
        k_tile_offset: Int32,
        output_num_k_tiles: Int32,
        write_invalid_logits: Int32,
        stream: cuda.CUstream,
    ):
        k_tma_source = _make_contiguous_k_tma_source(k_quant_bytes)
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_tma_source,
            _make_contiguous_k_tma_smem_layout(_PREFILL_BLOCK_K),
            (_PREFILL_BLOCK_K, _INDEX_HEAD_DIM),
            1,
        )
        self.kernel(
            q_u32,
            weights,
            k_quant_bytes,
            tma_tensor_k,
            k_tma_desc_ptrs,
            k_scales,
            k_start,
            k_end,
            logits_out,
            tile_logits,
            valid_q_rows,
            valid_k_rows,
            k_tile_offset,
            output_num_k_tiles,
            write_invalid_logits,
            tma_atom_k,
        ).launch(
            grid=(
                (valid_q_rows + _PREFILL512_BLOCK_Q - 1) // _PREFILL512_BLOCK_Q,
                output_num_k_tiles,
                1,
            ),
            block=[_PREFILL512_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_u32: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_tensor: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        k_scales: cute.Tensor,
        k_start: cute.Tensor,
        k_end: cute.Tensor,
        logits_out: cute.Tensor,
        tile_logits: cute.Tensor,
        valid_q_rows: Int32,
        valid_k_rows: Int32,
        k_tile_offset: Int32,
        output_num_k_tiles: Int32,
        write_invalid_logits: Int32,
        tma_atom_k: cute.CopyAtom,
    ):
        tx, _, _ = cute.arch.thread_idx()
        q_tile_idx, local_k_tile_idx, _ = cute.arch.block_idx()
        k_tile_idx = local_k_tile_idx + Int32(k_tile_offset)
        lane = tx % Int32(_WARP_THREADS)
        warp_idx = tx // Int32(_WARP_THREADS)
        warp_k_idx = warp_idx % Int32(_PREFILL512_WARPS_K)
        warp_q_idx = warp_idx // Int32(_PREFILL512_WARPS_K)

        q_tile_base = q_tile_idx * Int32(_PREFILL512_BLOCK_Q)
        k_tile_base = k_tile_idx * Int32(_PREFILL512_BLOCK_K)
        valid_q_rows = Int32(valid_q_rows)
        k_total_rows = Int32(valid_k_rows)
        TILED_OUTPUT = cutlass.const_expr(self._tiled_output)
        Q_HEADS_BATCH = cutlass.const_expr(self._q_heads_batch)
        num_heads = Int32(q_u32.shape[1])

        smem = cutlass.utils.SmemAllocator()

        SharedStorage = get_sparse_nsa_prefill512_shared_storage_cls(
            self._q_heads_batch,
        )
        storage = smem.allocate(SharedStorage)
        mbar_ptr_k = storage.mbar_ptr_k.data_ptr()
        k_perm_base_addr = shared_ptr_to_u32(storage.k_perm.data_ptr())
        s_k_perm_bytes = storage.k_perm.get_tensor(
            cute.make_layout((_PREFILL512_BLOCK_K * _INDEX_HEAD_DIM,), stride=(1,))
        )
        s_k_tma_stage0 = cute.make_tensor(
            s_k_perm_bytes.iterator,
            _make_contiguous_k_tma_smem_stage_layout(_PREFILL_BLOCK_K),
        )
        s_k_tma_stage1 = cute.make_tensor(
            s_k_perm_bytes.iterator + Int32(_PREFILL_BLOCK_K * _INDEX_HEAD_DIM),
            _make_contiguous_k_tma_smem_stage_layout(_PREFILL_BLOCK_K),
        )
        g_k_tma_tile = cute.local_tile(
            k_tma_tensor,
            (_PREFILL_BLOCK_K, _INDEX_HEAD_DIM),
            (None, 0),
        )
        load_k_tma0, _, _ = cute_copy.tma_get_copy_fn(
            tma_atom_k,
            0,
            cute.make_layout(1),
            g_k_tma_tile,
            s_k_tma_stage0,
        )
        load_k_tma1, _, _ = cute_copy.tma_get_copy_fn(
            tma_atom_k,
            0,
            cute.make_layout(1),
            g_k_tma_tile,
            s_k_tma_stage1,
        )
        s_scales = storage.scales.get_tensor(
            cute.make_layout((_PREFILL512_BLOCK_K,), stride=(1,))
        )
        s_k_start = storage.k_start.get_tensor(
            cute.make_layout((_PREFILL512_BLOCK_Q,), stride=(1,))
        )
        s_k_end = storage.k_end.get_tensor(
            cute.make_layout((_PREFILL512_BLOCK_Q,), stride=(1,))
        )
        s_tile_live = storage.tile_live.get_tensor(cute.make_layout((1,), stride=(1,)))
        s_tile_full = storage.tile_full.get_tensor(cute.make_layout((1,), stride=(1,)))

        if tx == 0:
            cute.arch.mbarrier_init(mbar_ptr_k, Int32(1))
        row_linear = tx
        while row_linear < Int32(_PREFILL512_BLOCK_Q):
            q_row = q_tile_base + row_linear
            s_k_start[row_linear] = (
                Int32(k_start[q_row]) if q_row < valid_q_rows else Int32(0)
            )
            s_k_end[row_linear] = (
                Int32(k_end[q_row]) if q_row < valid_q_rows else Int32(0)
            )
            row_linear += Int32(_PREFILL512_THREADS_PER_CTA)
        cute.arch.sync_threads()

        if warp_idx == Int32(0):
            tile_k_end = k_tile_base + Int32(_PREFILL512_BLOCK_K)
            if tile_k_end > k_total_rows:
                tile_k_end = k_total_rows
            row_start = Int32(s_k_start[tx])
            row_end = Int32(s_k_end[tx])
            physical_tile_full = tile_k_end == k_tile_base + Int32(_PREFILL512_BLOCK_K)
            row_live = (
                (row_end > k_tile_base)
                & (row_start < tile_k_end)
                & (k_tile_base < tile_k_end)
            )
            row_full = (
                (q_tile_base + tx < valid_q_rows)
                & physical_tile_full
                & (row_start <= k_tile_base)
                & (row_end >= tile_k_end)
            )
            ballot = cute.arch.vote_ballot_sync(Boolean(row_live))
            full_ballot = cute.arch.vote_ballot_sync(Boolean(row_full))
            if tx == Int32(0):
                s_tile_live[Int32(0)] = ballot
                s_tile_full[Int32(0)] = (
                    Int32(1) if full_ballot == Int32(-1) else Int32(0)
                )
        cute.arch.sync_threads()

        if s_tile_live[Int32(0)] != Int32(0):
            q_smem_base = k_perm_base_addr + Int32(
                _PREFILL512_BLOCK_K * _INDEX_HEAD_DIM
            )
            producer_state = cute_pipeline.PipelineStateSimple(1, Int32(0))
            consumer_state = cute_pipeline.PipelineStateSimple(1, Int32(0))
            if warp_idx == Int32(0):
                cpasync.prefetch_descriptor(tma_atom_k)
            if warp_idx == Int32(0):
                k_subtile_idx = k_tile_idx * Int32(2)
                _issue_contiguous_k_tma_copy_pair(
                    load_k_tma0,
                    load_k_tma1,
                    producer_state,
                    mbar_ptr_k,
                    Int32(_PREFILL512_BLOCK_K * _INDEX_HEAD_DIM),
                    k_subtile_idx,
                    k_subtile_idx + Int32(1),
                )
            # Exp40: weight staging removed — weights read from gmem directly (hidden by QK MMA latency).
            # Only stage scales while the K TMA copy is in flight.
            scale_linear = tx
            while scale_linear < Int32(_PREFILL512_BLOCK_K):
                s_scales[scale_linear] = Float32(k_scales[k_tile_base + scale_linear])
                scale_linear += Int32(_PREFILL512_THREADS_PER_CTA)

            acc_layout = cute.make_layout(
                (1, _PREFILL512_NUM_MMA_KV, 8), stride=(16, 8, 1)
            )
            acc_frag = cute.make_rmem_tensor(acc_layout, Float32)
            for mma_kv in cutlass.range_constexpr(_PREFILL512_NUM_MMA_KV):
                for reg_id in cutlass.range_constexpr(8):
                    acc_frag[Int32(0), mma_kv, reg_id] = Float32(0.0)

            # Exp38: compute Q staging variables and issue first Q-head batch
            # cp_async BEFORE mbarrier_wait — Q loads from gmem, independent of K TMA.
            # This overlaps first-batch Q staging latency with K TMA tail + weight/scale loads.
            threads_per_row = Int32(8)
            row_local = tx // threads_per_row
            col_group = tx % threads_per_row
            u32_col_base = col_group * Int32(4)
            q_row = q_tile_base + row_local
            in_bounds_pred = (
                Int32(1)
                if (row_local < Int32(_PREFILL_Q_STAGE_ROWS) and q_row < valid_q_rows)
                else Int32(0)
            )
            q_smem_stride = Int32(_PREFILL_Q_STAGE_BYTES)
            q_rs = Int32(_PREFILL_Q_STAGE_COLS // 16)
            thread_offset = _permuted_offset_128b(row_local, col_group, q_rs) * Int32(
                16
            )
            q_row_for_async = (
                q_row if row_local < Int32(_PREFILL_Q_STAGE_ROWS) else Int32(0)
            )

            batch_base_head = Int32(0)
            for block_offset in cutlass.range_constexpr(Q_HEADS_BATCH):
                head_in_batch = Int32(block_offset)
                global_head = batch_base_head + head_in_batch
                if global_head < num_heads:
                    gmem_u32_offset = (
                        q_row_for_async * num_heads + global_head
                    ) * Int32(_INDEX_HEAD_DIM // 4) + u32_col_base
                    cp_async_128b_pred(
                        q_smem_base + head_in_batch * q_smem_stride + thread_offset,
                        get_ptr_as_int64(q_u32, gmem_u32_offset),
                        in_bounds_pred,
                    )
            cute.arch.cp_async_commit_group()

            cute.arch.mbarrier_wait(
                mbar_ptr_k + consumer_state.index,
                phase=consumer_state.phase,
            )
            cute.arch.sync_threads()

            lane_group_pre = lane // Int32(4)
            q_local_rs0 = warp_q_idx * Int32(16) + lane_group_pre
            q_local_rs1 = q_local_rs0 + Int32(8)
            # Pipelined Q-head batch loop: overlap staging of batch N+1 with compute of batch N
            # Prologue: first Q-head batch already staged above before mbarrier_wait

            while batch_base_head < num_heads:
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                for block_offset in cutlass.range_constexpr(Q_HEADS_BATCH):
                    head_idx = batch_base_head + Int32(block_offset)
                    if head_idx < num_heads:
                        curr_mma_base = (
                            q_smem_base + Int32(block_offset) * q_smem_stride
                        )

                        score_frag = cute.make_rmem_tensor(acc_layout, Float32)
                        for mma_kv in cutlass.range_constexpr(_PREFILL512_NUM_MMA_KV):
                            for reg_id in cutlass.range_constexpr(8):
                                score_frag[Int32(0), mma_kv, reg_id] = Float32(0.0)
                        _prefill_qk_mma_from_smem_q(
                            score_frag,
                            curr_mma_base,
                            k_perm_base_addr,
                            lane,
                            warp_q_idx,
                            warp_k_idx,
                            Int32(0),
                            Int32(_PREFILL512_NUM_MMA_Q),
                            Int32(_PREFILL512_NUM_MMA_KV),
                            Int32(_INDEX_HEAD_DIM // 16),
                            Int32(_FP8_ROW_VECS),
                        )
                        lane_group = lane // Int32(4)
                        q_row_rs0 = q_tile_base + q_local_rs0
                        q_row_rs1 = q_tile_base + q_local_rs1
                        w_rs0 = (
                            Float32(weights[q_row_rs0, head_idx])
                            if q_row_rs0 < valid_q_rows
                            else Float32(0.0)
                        )
                        w_rs1 = (
                            Float32(weights[q_row_rs1, head_idx])
                            if q_row_rs1 < valid_q_rows
                            else Float32(0.0)
                        )
                        for mma_kv in cutlass.range_constexpr(_PREFILL512_NUM_MMA_KV):
                            for reg_id in cutlass.range_constexpr(8):
                                row_slot = (reg_id % 4) // 2
                                w_val = w_rs0 if row_slot == Int32(0) else w_rs1
                                if cutlass.const_expr(
                                    self._score_mode == IndexerScoreMode.MSA_BILINEAR
                                ):
                                    acc_frag[Int32(0), mma_kv, reg_id] = Float32(
                                        acc_frag[Int32(0), mma_kv, reg_id]
                                        + score_frag[Int32(0), mma_kv, reg_id] * w_val
                                    )
                                else:
                                    acc_frag[Int32(0), mma_kv, reg_id] = Float32(
                                        acc_frag[Int32(0), mma_kv, reg_id]
                                        + attention_ops.fmax(
                                            score_frag[Int32(0), mma_kv, reg_id],
                                            Float32(0.0),
                                        )
                                        * w_val
                                    )
                cute.arch.sync_threads()
                batch_base_head += Int32(Q_HEADS_BATCH)

                # Issue Q staging for next batch (overlapped with current sync overhead)
                if batch_base_head < num_heads:
                    for block_offset in cutlass.range_constexpr(Q_HEADS_BATCH):
                        head_in_batch = Int32(block_offset)
                        global_head = batch_base_head + head_in_batch
                        if global_head < num_heads:
                            gmem_u32_offset = (
                                q_row_for_async * num_heads + global_head
                            ) * Int32(_INDEX_HEAD_DIM // 4) + u32_col_base
                            cp_async_128b_pred(
                                q_smem_base
                                + head_in_batch * q_smem_stride
                                + thread_offset,
                                get_ptr_as_int64(q_u32, gmem_u32_offset),
                                in_bounds_pred,
                            )
                    cute.arch.cp_async_commit_group()

            lane_group = lane // Int32(4)
            lane_pair_base = Int32(2) * (lane % Int32(4))
            if TILED_OUTPUT:
                for mma_kv in cutlass.range_constexpr(_PREFILL512_NUM_MMA_KV):
                    for reg_id in cutlass.range_constexpr(8):
                        row_slot = (reg_id % 4) // 2
                        q_local = (
                            warp_q_idx * Int32(16) + lane_group + Int32(8 * row_slot)
                        )
                        k_local = (
                            warp_k_idx * Int32(_PREFILL512_NUM_MMA_KV * 16)
                            + mma_kv * Int32(16)
                            + lane_pair_base
                            + Int32(8 * (reg_id // 4))
                            + Int32(reg_id % 2)
                        )
                        if (
                            q_local < Int32(_PREFILL512_BLOCK_Q)
                            and k_local < Int32(_PREFILL512_BLOCK_K)
                            and q_tile_base + q_local < valid_q_rows
                        ):
                            _tile_id = (
                                q_tile_idx * Int32(output_num_k_tiles)
                                + local_k_tile_idx
                            )
                            _offset_in_tile = (
                                q_local * Int32(_PREFILL512_BLOCK_K) + k_local
                            )
                            _flat_offset = (
                                _tile_id
                                * Int32(_PREFILL512_BLOCK_Q * _PREFILL512_BLOCK_K)
                                + _offset_in_tile
                            )
                            tile_logits[_flat_offset] = Float32(
                                acc_frag[Int32(0), mma_kv, reg_id] * s_scales[k_local]
                            )
            else:
                if s_tile_full[Int32(0)] != Int32(0):
                    q_full_row0 = q_tile_base + warp_q_idx * Int32(16) + lane_group
                    q_full_row1 = q_full_row0 + Int32(8)
                    out_base0 = q_full_row0 * k_total_rows + k_tile_base
                    out_base1 = q_full_row1 * k_total_rows + k_tile_base
                    for mma_kv in cutlass.range_constexpr(_PREFILL512_NUM_MMA_KV):
                        k_pair0 = (
                            warp_k_idx * Int32(_PREFILL512_NUM_MMA_KV * 16)
                            + mma_kv * Int32(16)
                            + lane_pair_base
                        )
                        k_pair1 = k_pair0 + Int32(8)
                        scale0 = s_scales[k_pair0]
                        scale1 = s_scales[k_pair0 + Int32(1)]
                        scale8 = s_scales[k_pair1]
                        scale9 = s_scales[k_pair1 + Int32(1)]
                        st_global_v2_f32(
                            get_ptr_as_int64(logits_out, out_base0 + k_pair0),
                            Float32(acc_frag[Int32(0), mma_kv, 0] * scale0),
                            Float32(acc_frag[Int32(0), mma_kv, 1] * scale1),
                        )
                        st_global_v2_f32(
                            get_ptr_as_int64(logits_out, out_base1 + k_pair0),
                            Float32(acc_frag[Int32(0), mma_kv, 2] * scale0),
                            Float32(acc_frag[Int32(0), mma_kv, 3] * scale1),
                        )
                        st_global_v2_f32(
                            get_ptr_as_int64(logits_out, out_base0 + k_pair1),
                            Float32(acc_frag[Int32(0), mma_kv, 4] * scale8),
                            Float32(acc_frag[Int32(0), mma_kv, 5] * scale9),
                        )
                        st_global_v2_f32(
                            get_ptr_as_int64(logits_out, out_base1 + k_pair1),
                            Float32(acc_frag[Int32(0), mma_kv, 6] * scale8),
                            Float32(acc_frag[Int32(0), mma_kv, 7] * scale9),
                        )
                else:
                    for mma_kv in cutlass.range_constexpr(_PREFILL512_NUM_MMA_KV):
                        for reg_id in cutlass.range_constexpr(8):
                            row_slot = (reg_id % 4) // 2
                            q_local = (
                                warp_q_idx * Int32(16)
                                + lane_group
                                + Int32(8 * row_slot)
                            )
                            k_local = (
                                warp_k_idx * Int32(_PREFILL512_NUM_MMA_KV * 16)
                                + mma_kv * Int32(16)
                                + lane_pair_base
                                + Int32(8 * (reg_id // 4))
                                + Int32(reg_id % 2)
                            )
                            q_row = q_tile_base + q_local
                            k_row = k_tile_base + k_local
                            if (
                                q_local < Int32(_PREFILL512_BLOCK_Q)
                                and k_local < Int32(_PREFILL512_BLOCK_K)
                                and q_row < valid_q_rows
                                and k_row < k_total_rows
                            ):
                                row_start = Int32(s_k_start[q_local])
                                row_end = Int32(s_k_end[q_local])
                                if k_row >= row_start and k_row < row_end:
                                    logits_out[q_row, k_row] = Float32(
                                        acc_frag[Int32(0), mma_kv, reg_id]
                                        * s_scales[k_local]
                                    )
                                elif write_invalid_logits != Int32(0):
                                    logits_out[q_row, k_row] = Float32(-Float32.inf)
        else:
            if TILED_OUTPUT:
                pass  # dead tile; topk kernel uses k_start/k_end to filter
            else:
                if write_invalid_logits != Int32(0):
                    lane_group = lane // Int32(4)
                    lane_pair_base = Int32(2) * (lane % Int32(4))
                    if (
                        q_tile_base + Int32(_PREFILL512_BLOCK_Q) <= valid_q_rows
                        and k_tile_base + Int32(_PREFILL512_BLOCK_K) <= k_total_rows
                    ):
                        neg_inf = Float32(-Float32.inf)
                        for fill_iter in cutlass.range_constexpr(
                            (_PREFILL512_BLOCK_Q * _PREFILL512_BLOCK_K)
                            // (_PREFILL512_THREADS_PER_CTA * 4)
                        ):
                            fill_linear = tx * Int32(4) + Int32(
                                fill_iter * _PREFILL512_THREADS_PER_CTA * 4
                            )
                            q_local = fill_linear // Int32(_PREFILL512_BLOCK_K)
                            k_local = fill_linear - q_local * Int32(_PREFILL512_BLOCK_K)
                            out_base = (
                                (q_tile_base + q_local) * k_total_rows
                                + k_tile_base
                                + k_local
                            )
                            st_global_v4_f32(
                                get_ptr_as_int64(logits_out, out_base),
                                neg_inf,
                                neg_inf,
                                neg_inf,
                                neg_inf,
                            )
                    else:
                        for mma_kv in cutlass.range_constexpr(_PREFILL512_NUM_MMA_KV):
                            for reg_id in cutlass.range_constexpr(8):
                                row_slot = (reg_id % 4) // 2
                                q_local = (
                                    warp_q_idx * Int32(16)
                                    + lane_group
                                    + Int32(8 * row_slot)
                                )
                                k_local = (
                                    warp_k_idx * Int32(_PREFILL512_NUM_MMA_KV * 16)
                                    + mma_kv * Int32(16)
                                    + lane_pair_base
                                    + Int32(8 * (reg_id // 4))
                                    + Int32(reg_id % 2)
                                )
                                q_row = q_tile_base + q_local
                                k_row = k_tile_base + k_local
                                if (
                                    q_local < Int32(_PREFILL512_BLOCK_Q)
                                    and k_local < Int32(_PREFILL512_BLOCK_K)
                                    and q_row < valid_q_rows
                                    and k_row < k_total_rows
                                ):
                                    logits_out[q_row, k_row] = Float32(-Float32.inf)


@lru_cache(maxsize=16)
def _build_sparse_nsa_contiguous_prefill_kernel(
    *,
    tiled_output: bool = False,
    score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
    block_score_output: bool = False,
) -> SparseNSAContiguousLogitsPrefillKernel:
    kernel_type = (
        SparseNSAContiguousLogitsPrefillKernelBlockScores
        if block_score_output
        else SparseNSAContiguousLogitsPrefillKernel
    )
    return kernel_type(
        tiled_output=tiled_output,
        score_mode=score_mode,
        block_score_output=block_score_output,
    )


@lru_cache(maxsize=16)
def _build_sparse_nsa_contiguous_prefill512_kernel(
    *,
    tiled_output: bool = False,
    num_heads: int = 0,
    score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
) -> SparseNSAContiguousLogitsPrefill512Kernel:
    if int(num_heads) == _PREFILL512_H32_WEIGHT_COLS:
        return SparseNSAContiguousLogitsPrefill512Kernel(
            tiled_output=tiled_output,
            q_heads_batch=_PREFILL512_H32_Q_HEADS_BATCH,
            score_mode=score_mode,
        )
    return SparseNSAContiguousLogitsPrefill512Kernel(
        tiled_output=tiled_output,
        score_mode=score_mode,
    )


@lru_cache(maxsize=16)
def _build_sparse_nsa_contiguous_kernel(
    *,
    tiled_output: bool = False,
    score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
) -> SparseNSAContiguousLogitsKernel:
    return SparseNSAContiguousLogitsKernel(
        tiled_output=tiled_output,
        score_mode=score_mode,
    )


def _use_sparse_nsa_contiguous_prefill(valid_q_rows: int) -> bool:
    prefill_env = os.environ.get(_NSA_CONTIGUOUS_PREFILL_THRESHOLD_ENV, None)
    if prefill_env is not None:
        return int(prefill_env) > 0
    return valid_q_rows >= 256


def _prefill512_unsupported_reasons(
    *,
    valid_q_rows: int,
    k_rows: int,
    num_heads: int,
) -> list[str]:
    reasons: list[str] = []
    if valid_q_rows < _PREFILL512_MIN_Q_ROWS:
        reasons.append(f"valid_q_rows={valid_q_rows} < {_PREFILL512_MIN_Q_ROWS}")
    if k_rows < _PREFILL512_MIN_K_ROWS:
        reasons.append(f"k_rows={k_rows} < {_PREFILL512_MIN_K_ROWS}")
    if num_heads not in _PREFILL512_SUPPORTED_NUM_HEADS:
        reasons.append(
            f"num_heads={num_heads} not in {_PREFILL512_SUPPORTED_NUM_HEADS}"
        )
    return reasons


def resolve_contiguous_prefill_block_k(
    *,
    valid_q_rows: int,
    k_rows: int,
    num_heads: int,
) -> int | None:
    """Resolve the contiguous scorer path for the current shape.

    Returns None for the decode scorer, 256 for the existing prefill tile, or
    512 for the larger experimental prefill tile.
    """

    valid_q_rows = int(valid_q_rows)
    k_rows = int(k_rows)
    num_heads = int(num_heads)
    if valid_q_rows <= 0 or k_rows <= 0:
        return None
    if not _use_sparse_nsa_contiguous_prefill(valid_q_rows):
        return None

    block_k_env = (
        os.environ.get(_NSA_CONTIGUOUS_PREFILL_BLOCK_K_ENV, "auto").strip().lower()
    )
    if block_k_env in ("", "auto"):
        return (
            _PREFILL512_BLOCK_K
            if not _prefill512_unsupported_reasons(
                valid_q_rows=valid_q_rows,
                k_rows=k_rows,
                num_heads=num_heads,
            )
            else _PREFILL_BLOCK_K
        )
    if block_k_env == str(_PREFILL_BLOCK_K):
        return _PREFILL_BLOCK_K
    if block_k_env == str(_PREFILL512_BLOCK_K):
        reasons = _prefill512_unsupported_reasons(
            valid_q_rows=valid_q_rows,
            k_rows=k_rows,
            num_heads=num_heads,
        )
        if reasons:
            raise ValueError(
                f"{_NSA_CONTIGUOUS_PREFILL_BLOCK_K_ENV}=512 is unsupported for this shape: "
                + "; ".join(reasons)
            )
        return _PREFILL512_BLOCK_K
    raise ValueError(
        f"{_NSA_CONTIGUOUS_PREFILL_BLOCK_K_ENV} must be auto, 256, or 512; got {block_k_env!r}"
    )


def supports_contiguous_logits_kernel(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
) -> bool:
    if q_fp8.device.type != "cuda":
        return False
    if q_fp8.ndim != 3 or q_fp8.shape[2] != _INDEX_HEAD_DIM:
        return False
    if q_fp8.shape[1] > _MAX_Q_HEADS:
        return False
    if weights.ndim != 2 or weights.shape != q_fp8.shape[:2]:
        return False
    if k_quant.ndim != 2 or k_quant.shape[1] != _INDEX_HEAD_DIM:
        return False
    if k_scale.ndim != 1 or k_scale.shape[0] != k_quant.shape[0]:
        return False
    if k_start.ndim != 1 or k_end.ndim != 1 or k_start.shape != k_end.shape:
        return False
    if k_start.shape[0] > q_fp8.shape[0]:
        return False
    if q_fp8.dtype != torch.float8_e4m3fn:
        return False
    if weights.dtype != torch.float32:
        return False
    if k_quant.dtype != torch.float8_e4m3fn:
        return False
    if k_scale.dtype != torch.float32:
        return False
    if k_start.dtype != torch.int32 or k_end.dtype != torch.int32:
        return False
    if not (
        q_fp8.device
        == weights.device
        == k_quant.device
        == k_scale.device
        == k_start.device
        == k_end.device
    ):
        return False
    return True


def run_contiguous_logits_kernel(
    *,
    q_fp8: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    k_quant: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    k_start: torch.Tensor | None = None,
    k_end: torch.Tensor | None = None,
    preinitialize_invalid_logits: bool | None = None,
    tile_logits: torch.Tensor | None = None,
    tile_k_offset: int | None = None,
    tile_num_k_tiles: int | None = None,
    prefill_block_k: int | None = None,
    score_mode: int | None = None,
    binding: IndexerContiguousLogitsKernelBinding | None = None,
) -> torch.Tensor:
    staged_binding = binding
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("q_fp8", q_fp8),
                ("weights", weights),
                ("k_quant", k_quant),
                ("k_scale", k_scale),
                ("k_start", k_start),
                ("k_end", k_end),
                ("preinitialize_invalid_logits", preinitialize_invalid_logits),
                ("tile_logits", tile_logits),
                ("tile_k_offset", tile_k_offset),
                ("tile_num_k_tiles", tile_num_k_tiles),
                ("prefill_block_k", prefill_block_k),
                ("score_mode", score_mode),
            )
            if value is not None
        ]
        if extras:
            _raise_binding_extras("run_contiguous_logits_kernel", extras)
        q_fp8 = binding.q_fp8
        weights = binding.weights
        k_quant = binding.k_quant
        k_scale = binding.k_scale
        k_start = binding.k_start
        k_end = binding.k_end
        preinitialize_invalid_logits = binding.preinitialize_invalid_logits
        tile_logits = binding.tile_logits
        tile_k_offset = binding.tile_k_offset
        tile_num_k_tiles = binding.tile_num_k_tiles
        prefill_block_k = binding.prefill_block_k
        score_mode = binding.score_mode

    q_fp8 = _require_bound_arg(
        q_fp8, api_name="run_contiguous_logits_kernel", name="q_fp8"
    )
    weights = _require_bound_arg(
        weights, api_name="run_contiguous_logits_kernel", name="weights"
    )
    k_quant = _require_bound_arg(
        k_quant, api_name="run_contiguous_logits_kernel", name="k_quant"
    )
    k_scale = _require_bound_arg(
        k_scale, api_name="run_contiguous_logits_kernel", name="k_scale"
    )
    k_start = _require_bound_arg(
        k_start, api_name="run_contiguous_logits_kernel", name="k_start"
    )
    k_end = _require_bound_arg(
        k_end, api_name="run_contiguous_logits_kernel", name="k_end"
    )
    preinitialize_invalid_logits = (
        True
        if preinitialize_invalid_logits is None
        else bool(preinitialize_invalid_logits)
    )
    tile_k_offset = 0 if tile_k_offset is None else int(tile_k_offset)
    score_mode = (
        IndexerScoreMode.NSA_RELU_SUM if score_mode is None else int(score_mode)
    )
    if score_mode not in (IndexerScoreMode.NSA_RELU_SUM, IndexerScoreMode.MSA_BILINEAR):
        raise ValueError(f"unsupported indexer score_mode {score_mode}")

    if not supports_contiguous_logits_kernel(
        q_fp8=q_fp8,
        weights=weights,
        k_quant=k_quant,
        k_scale=k_scale,
        k_start=k_start,
        k_end=k_end,
    ):
        raise ValueError(
            "sparse NSA contiguous logits kernel only supports the exact CUDA FP8 contract"
        )

    q_rows_total = int(q_fp8.shape[0])
    valid_q_rows = int(k_start.shape[0])
    k_rows = int(k_quant.shape[0])
    if not preinitialize_invalid_logits and valid_q_rows != q_rows_total:
        raise ValueError(
            "preinitialize_invalid_logits=False requires all q rows to be valid; "
            f"got q_rows={q_rows_total} and valid_q_rows={valid_q_rows}"
        )

    # Dispatch: use prefill kernels for large q_len, decode kernel otherwise.
    # BK=512 is deliberately limited to target-ish long-prefill shapes; other
    # shapes keep the validated BK=256 prefill tile.
    if prefill_block_k is None:
        _prefill_block_k = resolve_contiguous_prefill_block_k(
            valid_q_rows=valid_q_rows,
            k_rows=k_rows,
            num_heads=int(q_fp8.shape[1]),
        )
    else:
        _prefill_block_k = int(prefill_block_k)
        if _prefill_block_k not in (_PREFILL_BLOCK_K, _PREFILL512_BLOCK_K):
            raise ValueError(
                "prefill_block_k must be "
                f"{_PREFILL_BLOCK_K} or {_PREFILL512_BLOCK_K}, "
                f"got {_prefill_block_k}"
            )
    if tile_logits is not None and _prefill_block_k is None:
        # Tiled output is only produced by the prefill scorer. The threshold
        # resolver may prefer the decode scorer for small q batches, but callers
        # that pass tile_logits are explicitly requesting the tiled layout.
        _prefill_block_k = _PREFILL_BLOCK_K
    _use_prefill = _prefill_block_k is not None
    _tiled_output = tile_logits is not None
    _grid_block_k = _prefill_block_k if _use_prefill else _BLOCK_K
    full_num_k_tiles = max(1, (k_rows + _grid_block_k - 1) // _grid_block_k)
    output_num_k_tiles = full_num_k_tiles
    if tile_num_k_tiles is not None:
        output_num_k_tiles = int(tile_num_k_tiles)
    tile_k_offset = int(tile_k_offset)
    if tile_k_offset < 0:
        raise ValueError(f"tile_k_offset must be non-negative, got {tile_k_offset}")
    if output_num_k_tiles <= 0:
        raise ValueError(f"tile_num_k_tiles must be positive, got {output_num_k_tiles}")
    if tile_k_offset + output_num_k_tiles > full_num_k_tiles:
        raise ValueError(
            "tile K range exceeds source K tiles: "
            f"offset={tile_k_offset}, tiles={output_num_k_tiles}, full={full_num_k_tiles}"
        )
    if (tile_k_offset != 0 or tile_num_k_tiles is not None) and not (
        _tiled_output and _use_prefill
    ):
        raise ValueError(
            "tile_k_offset/tile_num_k_tiles are only supported for tiled prefill output"
        )

    if _tiled_output and _use_prefill:
        _block_q_prefill = (
            _PREFILL512_BLOCK_Q
            if _prefill_block_k == _PREFILL512_BLOCK_K
            else _PREFILL_BLOCK_Q
        )
        num_q_tiles = (valid_q_rows + _block_q_prefill - 1) // _block_q_prefill
        if tile_logits is None:
            # 2D layout: (num_tiles, tile_size) where tile_size = block_q * block_k
            # 1D flat layout: total elements = num_tiles * tile_size
            # flat_offset = (q_tile_idx * num_k_tiles + k_tile_idx) * tile_size + q_local * block_k + k_local
            num_tiles = num_q_tiles * output_num_k_tiles
            tile_size = _block_q_prefill * _prefill_block_k
            tile_logits = torch.empty(
                (num_tiles * tile_size,),
                dtype=torch.float32,
                device=q_fp8.device,
            )
        else:
            expected_elements = (
                num_q_tiles * output_num_k_tiles * _block_q_prefill * _prefill_block_k
            )
            if int(tile_logits.numel()) < expected_elements:
                raise ValueError(
                    f"tile_logits has {int(tile_logits.numel())} elements, expected at least "
                    f"{expected_elements} for {num_q_tiles}x{output_num_k_tiles} tiles"
                )
        # Store metadata for run_tiled_topk
        tile_logits._sm12x_num_q_tiles = num_q_tiles
        tile_logits._sm12x_num_k_tiles = output_num_k_tiles
        tile_logits._sm12x_block_q = _block_q_prefill
        tile_logits._sm12x_block_k = _prefill_block_k
        preinitialize_invalid_logits = True

    if staged_binding is not None and staged_binding.q_u32 is not None:
        required_staged = {
            "q_bytes": staged_binding.q_bytes,
            "weights_kernel": staged_binding.weights_kernel,
            "k_quant_bytes": staged_binding.k_quant_bytes,
            "k_scale_kernel": staged_binding.k_scale_kernel,
            "k_start_kernel": staged_binding.k_start_kernel,
            "k_end_kernel": staged_binding.k_end_kernel,
            "out_kernel": staged_binding.out_kernel,
            "out_view": staged_binding.out_view,
        }
        missing = [name for name, value in required_staged.items() if value is None]
        if missing:
            raise ValueError(
                "staged indexer contiguous binding is missing " + ", ".join(missing)
            )
        q_u32 = staged_binding.q_u32
        q_bytes_kernel = staged_binding.q_bytes
        weights_kernel = staged_binding.weights_kernel
        k_quant_bytes = staged_binding.k_quant_bytes
        k_scale_kernel = staged_binding.k_scale_kernel
        k_start_kernel = staged_binding.k_start_kernel
        k_end_kernel = staged_binding.k_end_kernel
        out_kernel = staged_binding.out_kernel
        out_view = staged_binding.out_view
    else:
        if _tiled_output and _use_prefill:
            # In tiled mode, no need for the full scatter matrix
            out_kernel = torch.empty((1, 1), dtype=torch.float32, device=q_fp8.device)
        elif preinitialize_invalid_logits:
            out_kernel = torch.full(
                (q_rows_total, k_rows),
                float("-inf"),
                dtype=torch.float32,
                device=q_fp8.device,
            )
        else:
            out_kernel = torch.empty(
                (q_rows_total, k_rows),
                dtype=torch.float32,
                device=q_fp8.device,
            )
        out_view = out_kernel
        if valid_q_rows == 0 or k_rows == 0:
            return out_view

        q_bytes = q_fp8.contiguous().view(torch.uint8)
        q_bytes_kernel = q_bytes
        _pad_k = _prefill_block_k if _use_prefill else _BLOCK_K
        k_quant_padded, k_scale_padded = _pad_kv_rows(
            k_quant=k_quant, k_scale=k_scale, pad_block_k=_pad_k
        )
        k_quant_bytes = k_quant_padded.contiguous().view(torch.uint8)
        q_u32 = _view_last_dim_as_u32(q_bytes)
        weights_kernel = weights.contiguous()
        k_scale_kernel = k_scale_padded.contiguous()
        k_start_kernel = k_start.contiguous()
        k_end_kernel = k_end.contiguous()

    if valid_q_rows == 0 or k_rows == 0:
        return out_view

    device_index = q_fp8.device.index or 0
    k_tma_desc_ptrs = (
        staged_binding.k_tma_prefill_desc_ptrs
        if _use_prefill
        and staged_binding is not None
        and staged_binding.k_tma_prefill_desc_ptrs is not None
        else (
            staged_binding.k_tma_desc_ptrs
            if staged_binding is not None and staged_binding.k_tma_desc_ptrs is not None
            else _dummy_contiguous_k_tma_desc_ptrs(device_index)
        )
    )

    if _prefill_block_k == _PREFILL512_BLOCK_K:
        kernel = _build_sparse_nsa_contiguous_prefill512_kernel(
            tiled_output=_tiled_output,
            num_heads=int(q_fp8.shape[1]),
            score_mode=score_mode,
        )
    elif _use_prefill:
        kernel = _build_sparse_nsa_contiguous_prefill_kernel(
            tiled_output=_tiled_output,
            score_mode=score_mode,
            block_score_output=False,
        )
    else:
        kernel = _build_sparse_nsa_contiguous_kernel(
            tiled_output=_tiled_output,
            score_mode=score_mode,
        )

    if tile_logits is not None and _tiled_output:
        tile_logits_kernel = tile_logits
    elif staged_binding is not None:
        # TILED_OUTPUT is a compile-time false branch for this launch, so the
        # kernel cannot dereference tile_logits.  Preserve the historical 1x1
        # argument layout with a storage-aliasing view instead of allocating a
        # throwaway CUDA tensor on every staged replay.
        tile_logits_kernel = out_kernel.as_strided((1, 1), (1, 1))
    else:
        tile_logits_kernel = torch.empty(
            (1, 1), dtype=torch.float32, device=q_fp8.device
        )
    if staged_binding is not None:
        # run_contiguous_logits_kernel always compiles BLOCK_SCORE_OUTPUT=False;
        # this argument is therefore constexpr-dead.  Keep its rank/strides
        # stable while aliasing caller-owned output storage for graph replay.
        block_scores_kernel = out_kernel.as_strided((1, 1, 1), (1, 1, 1))
    else:
        block_scores_kernel = torch.empty(
            (1, 1, 1), dtype=torch.float32, device=q_fp8.device
        )

    if _use_prefill and _prefill_block_k == _PREFILL_BLOCK_K:
        write_invalid_logits = 0 if preinitialize_invalid_logits else 1
        args = (
            _to_kernel_tensor(q_u32, cutlass.Uint32),
            _to_kernel_tensor(weights_kernel, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(k_quant_bytes, cutlass.Uint8),
            _to_kernel_tensor(k_tma_desc_ptrs, cutlass.Int64, assumed_align=8),
            _to_kernel_tensor(k_scale_kernel, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(k_start_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(k_end_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(out_kernel, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(tile_logits_kernel, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(block_scores_kernel, cutlass.Float32, assumed_align=4),
            valid_q_rows,
            k_rows,
            Int32(0),
            Int32(tile_k_offset),
            Int32(output_num_k_tiles),
            write_invalid_logits,
            current_cuda_stream(),
        )
    elif _use_prefill:
        write_invalid_logits = 0 if preinitialize_invalid_logits else 1
        args = (
            _to_kernel_tensor(q_u32, cutlass.Uint32),
            _to_kernel_tensor(weights_kernel, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(k_quant_bytes, cutlass.Uint8),
            _to_kernel_tensor(k_tma_desc_ptrs, cutlass.Int64, assumed_align=8),
            _to_kernel_tensor(k_scale_kernel, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(k_start_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(k_end_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(out_kernel, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(tile_logits_kernel, cutlass.Float32, assumed_align=4),
            valid_q_rows,
            k_rows,
            Int32(tile_k_offset),
            Int32(output_num_k_tiles),
            write_invalid_logits,
            current_cuda_stream(),
        )
    else:
        write_invalid_logits = 0 if preinitialize_invalid_logits else 1
        args = (
            _to_kernel_tensor(q_u32, cutlass.Uint32),
            _to_kernel_tensor(weights_kernel, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(k_quant_bytes, cutlass.Uint8),
            _to_kernel_tensor(k_tma_desc_ptrs, cutlass.Int64, assumed_align=8),
            _to_kernel_tensor(k_scale_kernel, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(k_start_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(k_end_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(out_kernel, cutlass.Float32, assumed_align=4),
            _to_kernel_tensor(tile_logits_kernel, cutlass.Float32, assumed_align=4),
            valid_q_rows,
            k_rows,
            Int32(tile_k_offset),
            Int32(output_num_k_tiles),
            write_invalid_logits,
            current_cuda_stream(),
        )
    if _use_prefill:
        _prefill_cache_variant = (
            "prefill512_h32"
            if _prefill_block_k == _PREFILL512_BLOCK_K
            and int(q_fp8.shape[1]) == _PREFILL512_H32_WEIGHT_COLS
            else (
                "prefill512" if _prefill_block_k == _PREFILL512_BLOCK_K else "prefill"
            )
        )
        q_heads_batch = (
            (
                _PREFILL512_H32_Q_HEADS_BATCH
                if _prefill_cache_variant == "prefill512_h32"
                else _PREFILL512_Q_HEADS_BATCH
            )
            if _prefill_block_k == _PREFILL512_BLOCK_K
            else None
        )
        facts = _contiguous_logits_compile_facts(
            variant=_prefill_cache_variant,
            tiled_output=_tiled_output,
            score_mode=score_mode,
            q_u32=q_u32,
            weights=weights_kernel,
            k_quant_bytes=k_quant_bytes,
            k_scale=k_scale_kernel,
            k_start=k_start_kernel,
            k_end=k_end_kernel,
            logits_out=out_kernel,
            tile_logits=tile_logits_kernel,
            block_k=_prefill_block_k,
            block_score_output=False,
            q_heads_batch=q_heads_batch,
        )
    else:
        facts = _contiguous_logits_compile_facts(
            variant="decode",
            tiled_output=_tiled_output,
            score_mode=score_mode,
            q_u32=q_u32,
            weights=weights_kernel,
            k_quant_bytes=k_quant_bytes,
            k_scale=k_scale_kernel,
            k_start=k_start_kernel,
            k_end=k_end_kernel,
            logits_out=out_kernel,
            tile_logits=tile_logits_kernel,
            block_k=_BLOCK_K,
            block_score_output=False,
        )
    compile_spec = KernelCompileSpec.from_facts(
        "attention.indexer.contiguous_logits",
        3,
        *facts,
    )
    sm12x_launch(
        kernel,
        compile_spec=compile_spec,
        compile_args=args,
        runtime_args=args,
        compile_kwargs=(
            {
                "dsl_compile_options": (
                    OptLevel(2),
                    PtxasOptions("--register-usage-level=4"),
                )
            }
            if not _use_prefill
            else None
        ),
    )
    if _tiled_output and _use_prefill:
        return tile_logits
    return out_view


def run_contiguous_block_scores_kernel(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    block_scores: torch.Tensor | None = None,
    num_blocks_out: int | None = None,
    q_u32: torch.Tensor | None = None,
    q_bytes: torch.Tensor | None = None,
    weights_kernel: torch.Tensor | None = None,
    k_quant_bytes: torch.Tensor | None = None,
    k_scale_kernel: torch.Tensor | None = None,
    k_start_kernel: torch.Tensor | None = None,
    k_end_kernel: torch.Tensor | None = None,
    out_kernel: torch.Tensor | None = None,
    tile_logits_kernel: torch.Tensor | None = None,
    k_tma_prefill_desc_ptrs: torch.Tensor | None = None,
) -> torch.Tensor:
    if not supports_contiguous_logits_kernel(
        q_fp8=q_fp8,
        weights=weights,
        k_quant=k_quant,
        k_scale=k_scale,
        k_start=k_start,
        k_end=k_end,
    ):
        raise ValueError(
            "MSA contiguous block-score kernel only supports the production CUDA FP8 contract"
        )
    if int(q_fp8.shape[1]) > 8:
        raise ValueError(
            "MSA contiguous block-score kernel supports at most 8 index heads"
        )
    if weights.dtype != torch.float32:
        raise ValueError(f"weights must have dtype torch.float32, got {weights.dtype}")
    q_rows_total = int(q_fp8.shape[0])
    valid_q_rows = int(k_start.shape[0])
    k_rows = int(k_quant.shape[0])
    num_heads = int(q_fp8.shape[1])
    inferred_blocks = (k_rows + _PREFILL_BLOCK_K // 2 - 1) // (_PREFILL_BLOCK_K // 2)
    if num_blocks_out is None:
        num_blocks_out = inferred_blocks
    num_blocks_out = int(num_blocks_out)
    if num_blocks_out < inferred_blocks:
        raise ValueError(
            f"num_blocks_out {num_blocks_out} is smaller than required {inferred_blocks}"
        )
    if block_scores is None:
        block_scores = torch.full(
            (num_heads, q_rows_total, num_blocks_out),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp8.device,
        )
    else:
        if block_scores.dtype != torch.float32:
            raise ValueError(
                f"block_scores must have dtype torch.float32, got {block_scores.dtype}"
            )
        if block_scores.device != q_fp8.device:
            raise ValueError("block_scores device must match q_fp8")
        if (
            block_scores.ndim != 3
            or int(block_scores.shape[0]) < num_heads
            or int(block_scores.shape[1]) < q_rows_total
            or int(block_scores.shape[2]) < num_blocks_out
        ):
            raise ValueError(
                "block_scores must have shape at least "
                f"({num_heads}, {q_rows_total}, {num_blocks_out}), got "
                f"{tuple(block_scores.shape)}"
            )
        block_scores = block_scores[:num_heads, :q_rows_total, :num_blocks_out]
        block_scores.fill_(float("-inf"))
    if valid_q_rows == 0 or k_rows == 0:
        return block_scores

    if q_bytes is None:
        q_bytes = q_fp8.contiguous().view(torch.uint8)
    if q_u32 is None:
        q_u32 = _view_last_dim_as_u32(q_bytes)
    if weights_kernel is None:
        weights_kernel = weights.contiguous()
    if k_quant_bytes is None or k_scale_kernel is None:
        k_quant_padded, k_scale_padded = _pad_kv_rows(
            k_quant=k_quant,
            k_scale=k_scale,
            pad_block_k=_PREFILL_BLOCK_K,
        )
        k_quant_bytes = k_quant_padded.contiguous().view(torch.uint8)
        k_scale_kernel = k_scale_padded.contiguous()
    if k_start_kernel is None:
        k_start_kernel = k_start.contiguous()
    if k_end_kernel is None:
        k_end_kernel = k_end.contiguous()
    if out_kernel is None:
        out_kernel = torch.empty((1, 1), dtype=torch.float32, device=q_fp8.device)
    if tile_logits_kernel is None:
        tile_logits_kernel = torch.empty(
            (1, 1), dtype=torch.float32, device=q_fp8.device
        )
    if k_tma_prefill_desc_ptrs is None:
        device_index = q_fp8.device.index or 0
        k_tma_prefill_desc_ptrs = _dummy_contiguous_k_tma_desc_ptrs(device_index)

    output_num_k_tiles = max(1, (k_rows + _PREFILL_BLOCK_K - 1) // _PREFILL_BLOCK_K)
    kernel = _build_sparse_nsa_contiguous_prefill_kernel(
        tiled_output=False,
        score_mode=IndexerScoreMode.MSA_BILINEAR,
        block_score_output=True,
    )
    args = (
        _to_kernel_tensor(q_u32, cutlass.Uint32),
        _to_kernel_tensor(weights_kernel, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_quant_bytes, cutlass.Uint8),
        _to_kernel_tensor(k_tma_prefill_desc_ptrs, cutlass.Int64, assumed_align=8),
        _to_kernel_tensor(k_scale_kernel, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_start_kernel, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(k_end_kernel, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(out_kernel, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(tile_logits_kernel, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(block_scores, cutlass.Float32, assumed_align=4),
        valid_q_rows,
        k_rows,
        Int32(num_blocks_out),
        Int32(0),
        Int32(output_num_k_tiles),
        Int32(0),
        current_cuda_stream(),
    )
    facts = _contiguous_logits_compile_facts(
        variant="prefill_msa_blockmax",
        tiled_output=False,
        score_mode=IndexerScoreMode.MSA_BILINEAR,
        q_u32=q_u32,
        weights=weights_kernel,
        k_quant_bytes=k_quant_bytes,
        k_scale=k_scale_kernel,
        k_start=k_start_kernel,
        k_end=k_end_kernel,
        logits_out=out_kernel,
        tile_logits=tile_logits_kernel,
        block_k=_PREFILL_BLOCK_K,
        block_score_output=True,
        block_scores=block_scores,
    )
    compile_spec = KernelCompileSpec.from_facts(
        "attention.indexer.contiguous_block_scores",
        2,
        *facts,
    )
    sm12x_launch(
        kernel,
        compile_spec=compile_spec,
        compile_args=args,
        runtime_args=args,
    )
    return block_scores
