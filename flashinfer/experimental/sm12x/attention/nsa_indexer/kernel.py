# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/indexer/kernel.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""CuTeDSL paged decode score kernel for the paged NSA contract."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import Float32, Int32, Uint32
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from flashinfer.experimental.sm12x.attention._shared.cute import copy as cute_copy
from flashinfer.experimental.sm12x.attention._shared.cute import (
    pipeline as cute_pipeline,
)
from flashinfer.experimental.sm12x.attention._shared.cute import ops as attention_ops
from flashinfer.experimental.sm12x._lib.compiler import (
    KernelCompileSpec,
    launch as sm12x_launch,
    tensor_compile_fact,
)
from flashinfer.experimental.sm12x._lib.intrinsics import get_sm_version
from flashinfer.experimental.sm12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    frag_layout_swizzle_16b_to_8b,
    get_ptr_as_int64,
    ld_global_v4_u32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x2_b16,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x4_left_half_b16,
    ldmatrix_m8n8x4_right_half_b16,
    mxfp8_mma_m16n8k32_f32_e4m3,
    shared_ptr_to_u32,
    st_shared_v4_u32,
)
from cutlass import Int64
from flashinfer.experimental.sm12x._lib.utils import current_cuda_stream


_INDEX_HEAD_DIM = 128
_PAGE_SIZE = 64
_SCALE_BYTES = 4
_WARP_THREADS = 32
_PAGED_WARPS_PER_CTA = 4
_PAGED_THREADS_PER_CTA = _WARP_THREADS * _PAGED_WARPS_PER_CTA
_PAGED_TOKENS_PER_GROUP = 8
PAGED_MQA_LOGITS_SCHEDULE_PAGES_PER_SPLIT = 4
_SCHEDULE_MIN_PAGES = 1024
_ENABLE_MULTI_ROW_SCHEDULE = True
_SCHEDULE_SINGLE_ROW_PARALLEL_CTAS = 4
_SCHEDULE_MULTI_ROW_PARALLEL_CTAS = 4
_SCHEDULE_MULTI_ROW_MAX_Q_ROWS = 8
_MAX_SUPPORTED_Q_HEADS = 64
_PAGED_TILED_BLOCK_Q = 32
_PAGED_TILED_BLOCK_K = 512
_PAGED_Q_HEAD_TILE = 16
_BLACKWELL_TINY_STRIDED_TMA_MAX_BACKING_BYTES = 128 * 1024


class IndexerScoreMode:
    NSA_RELU_SUM = 0
    MSA_BILINEAR = 1


class IndexerOutputMode:
    TOKEN_LOGITS = 0
    PAGE_HEAD_MAX = 1


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
class IndexerPagedLogitsKernelBinding:
    q_fp8: torch.Tensor
    weights: torch.Tensor
    index_k_cache: torch.Tensor
    real_page_table: torch.Tensor
    seqlens_per_query: torch.Tensor
    schedule_metadata: torch.Tensor | None = None
    active_width: torch.Tensor | None = None
    page_size: int = _PAGE_SIZE
    preinitialize_invalid_logits: bool = True
    score_mode: int = IndexerScoreMode.NSA_RELU_SUM
    output_mode: int = IndexerOutputMode.TOKEN_LOGITS
    page_scores: torch.Tensor | None = None

    def run(self) -> torch.Tensor:
        return run_paged_logits_kernel(binding=self)


@dataclass(frozen=True, kw_only=True)
class IndexerPagedTiledLogitsKernelBinding:
    q_fp8: torch.Tensor
    weights: torch.Tensor
    index_k_cache: torch.Tensor
    real_page_table: torch.Tensor
    seqlens_per_query: torch.Tensor
    active_width: torch.Tensor
    tile_logits: torch.Tensor
    page_size: int = _PAGE_SIZE
    tile_block_q: int = _PAGED_TILED_BLOCK_Q
    tile_block_k: int = _PAGED_TILED_BLOCK_K
    preinitialize_tile_logits: bool = True

    def run(self) -> torch.Tensor:
        return run_paged_tiled_logits_kernel(binding=self)


@dataclass(frozen=True, kw_only=True)
class IndexerPagedSupertileLogitsKernelBinding:
    q_fp8: torch.Tensor
    weights: torch.Tensor
    index_k_cache: torch.Tensor
    real_page_table: torch.Tensor
    seqlens_per_query: torch.Tensor
    active_width: torch.Tensor
    tile_logits: torch.Tensor
    source_page_offset: int
    output_width_tokens: int
    page_size: int = _PAGE_SIZE
    tile_block_q: int = _PAGED_TILED_BLOCK_Q
    tile_block_k: int = _PAGED_TILED_BLOCK_K
    preinitialize_tile_logits: bool = True

    def run(self) -> torch.Tensor:
        return run_paged_supertile_logits_kernel(binding=self)


def build_indexer_paged_logits_kernel_binding(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    schedule_metadata: torch.Tensor | None = None,
    active_width: torch.Tensor | None = None,
    page_size: int = _PAGE_SIZE,
    preinitialize_invalid_logits: bool = True,
    score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
    output_mode: int = IndexerOutputMode.TOKEN_LOGITS,
    page_scores: torch.Tensor | None = None,
) -> IndexerPagedLogitsKernelBinding:
    return IndexerPagedLogitsKernelBinding(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        schedule_metadata=schedule_metadata,
        active_width=active_width,
        page_size=int(page_size),
        preinitialize_invalid_logits=bool(preinitialize_invalid_logits),
        score_mode=int(score_mode),
        output_mode=int(output_mode),
        page_scores=page_scores,
    )


def build_indexer_paged_tiled_logits_kernel_binding(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    active_width: torch.Tensor,
    tile_logits: torch.Tensor,
    page_size: int = _PAGE_SIZE,
    tile_block_q: int = _PAGED_TILED_BLOCK_Q,
    tile_block_k: int = _PAGED_TILED_BLOCK_K,
    preinitialize_tile_logits: bool = True,
) -> IndexerPagedTiledLogitsKernelBinding:
    return IndexerPagedTiledLogitsKernelBinding(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        active_width=active_width,
        tile_logits=tile_logits,
        page_size=int(page_size),
        tile_block_q=int(tile_block_q),
        tile_block_k=int(tile_block_k),
        preinitialize_tile_logits=bool(preinitialize_tile_logits),
    )


def build_indexer_paged_supertile_logits_kernel_binding(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    active_width: torch.Tensor,
    tile_logits: torch.Tensor,
    source_page_offset: int,
    output_width_tokens: int,
    page_size: int = _PAGE_SIZE,
    tile_block_q: int = _PAGED_TILED_BLOCK_Q,
    tile_block_k: int = _PAGED_TILED_BLOCK_K,
    preinitialize_tile_logits: bool = True,
) -> IndexerPagedSupertileLogitsKernelBinding:
    return IndexerPagedSupertileLogitsKernelBinding(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        active_width=active_width,
        tile_logits=tile_logits,
        source_page_offset=int(source_page_offset),
        output_width_tokens=int(output_width_tokens),
        page_size=int(page_size),
        tile_block_q=int(tile_block_q),
        tile_block_k=int(tile_block_k),
        preinitialize_tile_logits=bool(preinitialize_tile_logits),
    )


def _num_q_head_tiles(num_heads: int) -> int:
    return max((num_heads + _PAGED_Q_HEAD_TILE - 1) // _PAGED_Q_HEAD_TILE, 1)


def _page_splits_for_num_heads(num_heads: int) -> int:
    num_q_head_tiles = _num_q_head_tiles(num_heads)
    token_groups = _PAGED_WARPS_PER_CTA // num_q_head_tiles
    tokens_per_work = _PAGED_TOKENS_PER_GROUP * token_groups
    return _PAGE_SIZE // tokens_per_work


@lru_cache(maxsize=16)
def _paged_indexer_shared_storage_cls(
    padded_q_heads: int,
    tokens_per_work: int,
    num_q_head_tiles: int,
):
    class SharedStorage:
        pass

    def k_page_storage():
        return cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, _PAGE_SIZE * _INDEX_HEAD_DIM],
            1024,
        ]

    annotations = {
        "mbar_ptr_k": cute.struct.MemRange[cutlass.Int64, 1],
        "q_bytes": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, int(padded_q_heads) * _INDEX_HEAD_DIM],
            16,
        ],
        "weights": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, int(padded_q_heads)],
            16,
        ],
    }
    annotations["k_page"] = k_page_storage()
    annotations.update(
        {
            "k_page_perm": k_page_storage(),
            "scales": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, _PAGE_SIZE],
                16,
            ],
            "partial_logits": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    max(
                        int(tokens_per_work) * int(num_q_head_tiles),
                        _PAGED_WARPS_PER_CTA * _PAGED_Q_HEAD_TILE,
                    ),
                ],
                16,
            ],
        }
    )
    SharedStorage.__annotations__ = annotations
    return cute.struct(SharedStorage)


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
    if leading_dim is not None and tensor.ndim >= 2:
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


def _is_row_shared_i32_matrix(tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim == 2
        and tensor.dtype == torch.int32
        and int(tensor.stride(0)) == 0
        and int(tensor.stride(1)) == 1
    )


def _tensor_meta_key(
    tensor: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...], str, tuple[str, int | None]]:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


def _tensor_compile_key(
    name: str,
    tensor: torch.Tensor,
    *,
    dynamic_dims: tuple[int, ...] = (),
) -> tuple[object, ...]:
    return tensor_compile_fact(name, tensor, dynamic_dims=dynamic_dims)


def _assume_paged_k_tma_source_aligned(t: cute.Tensor):
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


def _make_paged_index_k_tma_source(
    k_quant_bytes: cutlass.cute.Tensor,
) -> cutlass.cute.Tensor:
    t_pages = cute.make_tensor(
        k_quant_bytes.iterator, cute.select(k_quant_bytes.layout, mode=[1, 2, 0])
    )
    return _assume_paged_k_tma_source_aligned(t_pages)


@lru_cache(maxsize=16)
def _default_sparse_nsa_persistent_ctas(device_index: int) -> int:
    num_sms = int(torch.cuda.get_device_properties(device_index).multi_processor_count)
    return max(num_sms * 4, 1)


def _resolve_sparse_nsa_persistent_ctas(
    *,
    device_index: int,
    q_rows: int,
) -> int:
    persistent_ctas = _default_sparse_nsa_persistent_ctas(device_index)
    if q_rows >= 4:
        persistent_ctas = max(persistent_ctas // 2, 1)
    return persistent_ctas


def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
    storage = (
        tensor.untyped_storage()
        if hasattr(tensor, "untyped_storage")
        else tensor.storage()
    )
    if hasattr(storage, "nbytes"):
        return int(storage.nbytes())
    return int(storage.size()) * tensor.element_size()


def _is_dense_non_overlapping(tensor: torch.Tensor) -> bool:
    expected_stride = 1
    strides_and_sizes = []
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        if int(size) <= 1:
            continue
        strides_and_sizes.append((abs(int(stride)), int(size)))
    for stride, size in sorted(strides_and_sizes):
        if stride != expected_stride:
            return False
        expected_stride *= size
    return True


def _needs_paged_index_k_scalar_load(
    index_k_cache: torch.Tensor,
    k_quant_bytes: torch.Tensor,
) -> bool:
    # Real serving caches are large enough for the normal TMA path. Tiny
    # strided caches appear in fake/warmup runs and should still be valid, so
    # load those pages cooperatively instead of depending on tiny strided TMA
    # descriptor behavior.
    if get_sm_version(index_k_cache.device) // 10 != 12:
        return False
    if (
        _tensor_storage_nbytes(index_k_cache)
        >= _BLACKWELL_TINY_STRIDED_TMA_MAX_BACKING_BYTES
    ):
        return False
    return not _is_dense_non_overlapping(k_quant_bytes)


@lru_cache(maxsize=16)
def _dummy_paged_index_k_tma_desc_ptrs(device_index: int) -> torch.Tensor:
    return torch.zeros(
        (1,), dtype=torch.int64, device=torch.device("cuda", device_index)
    )


@lru_cache(maxsize=32)
def _cached_int32_scalar(value: int, device_index: int) -> torch.Tensor:
    return torch.tensor(
        [value], dtype=torch.int32, device=torch.device("cuda", device_index)
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
def _pack_q_mxfp8_reg(
    s_q_bytes: cute.Tensor,
    row: Int32,
    col_pair_base: Int32,
) -> Uint32:
    b0 = Uint32(s_q_bytes[row, col_pair_base + Int32(0)])
    b1 = Uint32(s_q_bytes[row, col_pair_base + Int32(1)])
    b2 = Uint32(s_q_bytes[row, col_pair_base + Int32(8)])
    b3 = Uint32(s_q_bytes[row, col_pair_base + Int32(9)])
    return b0 | (b1 << Int32(8)) | (b2 << Int32(16)) | (b3 << Int32(24))


@cute.jit
def _repack_k_page_to_permuted(
    k_linear_base_addr: Int32,
    k_perm_base_addr: Int32,
    lane_linear: Int32,
):
    linear = lane_linear
    total = Int32(_PAGE_SIZE * (_INDEX_HEAD_DIM // 16))
    while linear < total:
        row = linear // Int32(_INDEX_HEAD_DIM // 16)
        vec_idx = linear - row * Int32(_INDEX_HEAD_DIM // 16)
        src_addr = k_linear_base_addr + Int32(row * _INDEX_HEAD_DIM + vec_idx * 16)
        dst_addr = _smem_addr_from_b128_offset(
            k_perm_base_addr,
            _permuted_offset_128b(row, vec_idx, Int32(_INDEX_HEAD_DIM // 16)),
        )
        v0, v1, v2, v3 = ld_shared_v4_u32(src_addr)
        st_shared_v4_u32(dst_addr, v0, v1, v2, v3)
        linear += Int32(_PAGED_THREADS_PER_CTA)


@cute.jit
def _load_index_k_page_scalar(
    k_quant_bytes: cute.Tensor,
    page_id: Int32,
    s_k_page_stage: cute.Tensor,
    lane_linear: Int32,
):
    linear = lane_linear
    total = Int32(_PAGE_SIZE * _INDEX_HEAD_DIM)
    while linear < total:
        row = linear // Int32(_INDEX_HEAD_DIM)
        col = linear - row * Int32(_INDEX_HEAD_DIM)
        s_k_page_stage[row, col, Int32(0)] = k_quant_bytes[page_id, row, col]
        linear += Int32(_PAGED_THREADS_PER_CTA)


@cute.jit
def _reduce_column_pair_sum(value: Float32) -> Float32:
    value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=4))
    value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=8))
    value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=16))
    return value


@cute.jit
def _reduce_quad_max(value: Float32) -> Float32:
    value = attention_ops.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=1))
    value = attention_ops.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=2))
    return value


@cute.jit
def _compute_mxfp8_tile_partials(
    s_q_bytes: cute.Tensor,
    s_w: cute.Tensor,
    num_heads: Int32,
    k_perm_base_addr: Int32,
    token_base: Int32,
    head_tile_base: Int32,
    lane: Int32,
    s_partial_logits: cute.Tensor,
    partial_row_base: Int32,
    head_tile_slot: Int32,
    score_mode: cutlass.Constexpr[int] = IndexerScoreMode.NSA_RELU_SUM,
):
    group_id = lane // Int32(4)
    thread_id_in_group = lane % Int32(4)
    col_pair_base = thread_id_in_group * Int32(2)
    q0_acc = Float32(0.0)
    q1_acc = Float32(0.0)
    q2_acc = Float32(0.0)
    q3_acc = Float32(0.0)
    k_offset = _permuted_offset_128b(
        token_base + Int32(8) * (lane // Int32(16)) + lane % Int32(8),
        (lane % Int32(16)) // Int32(8),
        Int32(_INDEX_HEAD_DIM // 16),
    )
    for mma_pair in cutlass.range_constexpr(_INDEX_HEAD_DIM // 32):
        pair_base = Int32(mma_pair * 32) + col_pair_base
        q0 = _pack_q_mxfp8_reg(s_q_bytes, head_tile_base + group_id, pair_base)
        q1 = _pack_q_mxfp8_reg(
            s_q_bytes, head_tile_base + group_id + Int32(8), pair_base
        )
        q2 = _pack_q_mxfp8_reg(
            s_q_bytes, head_tile_base + group_id, pair_base + Int32(16)
        )
        q3 = _pack_q_mxfp8_reg(
            s_q_bytes,
            head_tile_base + group_id + Int32(8),
            pair_base + Int32(16),
        )
        b0_k0, _ = ldmatrix_m8n8x4_left_half_b16(
            _smem_addr_from_b128_offset(k_perm_base_addr, k_offset)
        )
        b0_k1, _ = ldmatrix_m8n8x4_right_half_b16(
            _smem_addr_from_b128_offset(k_perm_base_addr, k_offset)
        )
        b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
        b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1)
        k_offset_cur = _advance_offset_by_row_128b(
            k_offset,
            Int32(16),
            Int32(_INDEX_HEAD_DIM // 16),
        )
        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            q0_acc,
            q1_acc,
            q2_acc,
            q3_acc,
            q0,
            q1,
            q2,
            q3,
            b0_k0,
            b0_k1,
            Uint32(0x7F7F7F7F),
            Uint32(0x7F7F7F7F),
        )
        q0_acc = d0
        q1_acc = d1
        q2_acc = d2
        q3_acc = d3
        k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_pair) - Int32(
            16 * (_INDEX_HEAD_DIM // 16)
        )

    head0 = head_tile_base + group_id
    head1 = head0 + Int32(8)
    w0 = Float32(0.0)
    w1 = Float32(0.0)
    if head0 < num_heads:
        w0 = Float32(s_w[head0])
    if head1 < num_heads:
        w1 = Float32(s_w[head1])
    col0 = col_pair_base
    col1 = col_pair_base + Int32(1)
    if cutlass.const_expr(score_mode == IndexerScoreMode.MSA_BILINEAR):
        partial0 = Float32(q0_acc * w0)
        partial0 = Float32(partial0 + q2_acc * w1)
        partial1 = Float32(q1_acc * w0)
        partial1 = Float32(partial1 + q3_acc * w1)
    else:
        partial0 = Float32(attention_ops.fmax(q0_acc, Float32(0.0)) * w0)
        partial0 = Float32(partial0 + attention_ops.fmax(q2_acc, Float32(0.0)) * w1)
        partial1 = Float32(attention_ops.fmax(q1_acc, Float32(0.0)) * w0)
        partial1 = Float32(partial1 + attention_ops.fmax(q3_acc, Float32(0.0)) * w1)
    partial0 = _reduce_column_pair_sum(partial0)
    partial1 = _reduce_column_pair_sum(partial1)
    if group_id == Int32(0):
        s_partial_logits[partial_row_base + col0, head_tile_slot] = partial0
        s_partial_logits[partial_row_base + col1, head_tile_slot] = partial1


@cute.jit
def _compute_mxfp8_tile_head_token_max(
    s_q_bytes: cute.Tensor,
    s_w: cute.Tensor,
    num_heads: Int32,
    k_perm_base_addr: Int32,
    token_base: Int32,
    head_tile_base: Int32,
    lane: Int32,
    s_scale: cute.Tensor,
    valid_slots: Int32,
    s_page_partials: cute.Tensor,
    token_group: Int32,
):
    group_id = lane // Int32(4)
    thread_id_in_group = lane % Int32(4)
    col_pair_base = thread_id_in_group * Int32(2)
    q0_acc = Float32(0.0)
    q1_acc = Float32(0.0)
    q2_acc = Float32(0.0)
    q3_acc = Float32(0.0)
    k_offset = _permuted_offset_128b(
        token_base + Int32(8) * (lane // Int32(16)) + lane % Int32(8),
        (lane % Int32(16)) // Int32(8),
        Int32(_INDEX_HEAD_DIM // 16),
    )
    for mma_pair in cutlass.range_constexpr(_INDEX_HEAD_DIM // 32):
        pair_base = Int32(mma_pair * 32) + col_pair_base
        q0 = _pack_q_mxfp8_reg(s_q_bytes, head_tile_base + group_id, pair_base)
        q1 = _pack_q_mxfp8_reg(
            s_q_bytes, head_tile_base + group_id + Int32(8), pair_base
        )
        q2 = _pack_q_mxfp8_reg(
            s_q_bytes, head_tile_base + group_id, pair_base + Int32(16)
        )
        q3 = _pack_q_mxfp8_reg(
            s_q_bytes,
            head_tile_base + group_id + Int32(8),
            pair_base + Int32(16),
        )
        b0_k0, _ = ldmatrix_m8n8x4_left_half_b16(
            _smem_addr_from_b128_offset(k_perm_base_addr, k_offset)
        )
        b0_k1, _ = ldmatrix_m8n8x4_right_half_b16(
            _smem_addr_from_b128_offset(k_perm_base_addr, k_offset)
        )
        b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
        b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1)
        k_offset_cur = _advance_offset_by_row_128b(
            k_offset,
            Int32(16),
            Int32(_INDEX_HEAD_DIM // 16),
        )
        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            q0_acc,
            q1_acc,
            q2_acc,
            q3_acc,
            q0,
            q1,
            q2,
            q3,
            b0_k0,
            b0_k1,
            Uint32(0x7F7F7F7F),
            Uint32(0x7F7F7F7F),
        )
        q0_acc = d0
        q1_acc = d1
        q2_acc = d2
        q3_acc = d3
        k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_pair) - Int32(
            16 * (_INDEX_HEAD_DIM // 16)
        )

    head0 = head_tile_base + group_id
    head1 = head0 + Int32(8)
    w0 = Float32(0.0)
    w1 = Float32(0.0)
    if head0 < num_heads:
        w0 = Float32(s_w[head0])
    if head1 < num_heads:
        w1 = Float32(s_w[head1])

    col0 = token_base + col_pair_base
    col1 = col0 + Int32(1)
    max0 = Float32(-Float32.inf)
    max1 = Float32(-Float32.inf)
    if (head0 < num_heads) and (col0 < valid_slots):
        max0 = attention_ops.fmax(max0, Float32(q0_acc * w0 * s_scale[col0]))
    if (head0 < num_heads) and (col1 < valid_slots):
        max0 = attention_ops.fmax(max0, Float32(q1_acc * w0 * s_scale[col1]))
    if (head1 < num_heads) and (col0 < valid_slots):
        max1 = attention_ops.fmax(max1, Float32(q2_acc * w1 * s_scale[col0]))
    if (head1 < num_heads) and (col1 < valid_slots):
        max1 = attention_ops.fmax(max1, Float32(q3_acc * w1 * s_scale[col1]))

    max0 = _reduce_quad_max(max0)
    max1 = _reduce_quad_max(max1)
    if thread_id_in_group == Int32(0):
        if head0 < num_heads:
            s_page_partials[token_group, head0] = attention_ops.fmax(
                s_page_partials[token_group, head0],
                max0,
            )
        if head1 < num_heads:
            s_page_partials[token_group, head1] = attention_ops.fmax(
                s_page_partials[token_group, head1],
                max1,
            )


@cute.jit
def _issue_index_k_tma_copy(
    load_tma,
    producer_state,
    mbar_ptr,
    expected_bytes,
    page_id,
):
    full_mbar_ptr = mbar_ptr + producer_state.index
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(full_mbar_ptr, expected_bytes)
    load_tma(src_idx=page_id, dst_idx=producer_state.index, tma_bar_ptr=full_mbar_ptr)


class SparseNSAPagedLogitsKernel:
    """One CTA reuses a query row across a small tile of paged candidate positions."""

    def __init__(
        self,
        persistent_ctas: int,
        num_heads_static: int,
        *,
        tiled_output: bool = False,
        tile_block_q: int = _PAGED_TILED_BLOCK_Q,
        tile_block_k: int = _PAGED_TILED_BLOCK_K,
        score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
    ):
        self.persistent_ctas = int(persistent_ctas)
        self.num_heads_static = int(num_heads_static)
        self.tiled_output = bool(tiled_output)
        self.tile_block_q = int(tile_block_q)
        self.tile_block_k = int(tile_block_k)
        self.score_mode = int(score_mode)
        if self.tiled_output and self.tile_block_k != _PAGED_TILED_BLOCK_K:
            raise ValueError(
                f"paged tiled logits currently require block_k={_PAGED_TILED_BLOCK_K}, "
                f"got {self.tile_block_k}"
            )
        self.num_q_head_tiles = _num_q_head_tiles(self.num_heads_static)
        if self.num_q_head_tiles not in (1, 2, 4):
            raise ValueError(
                f"paged logits kernel only supports 1/2/4 head tiles, got {self.num_q_head_tiles}"
            )
        self.padded_q_heads = self.num_q_head_tiles * _PAGED_Q_HEAD_TILE
        self.token_groups = _PAGED_WARPS_PER_CTA // self.num_q_head_tiles
        self.tokens_per_work = _PAGED_TOKENS_PER_GROUP * self.token_groups
        self.page_splits = _PAGE_SIZE // self.tokens_per_work

    def _get_shared_storage_cls(self):
        return _paged_indexer_shared_storage_cls(
            self.padded_q_heads,
            self.tokens_per_work,
            self.num_q_head_tiles,
        )

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        use_scalar_k_load: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        active_width: cute.Tensor,
        logits_out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        k_tma_source = _make_paged_index_k_tma_source(k_quant_bytes)
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_tma_source,
            cute.make_layout(
                (_PAGE_SIZE, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1)
            ),
            (_PAGE_SIZE, _INDEX_HEAD_DIM),
            1,
        )
        self.kernel(
            q_bytes,
            weights,
            k_quant_bytes,
            tma_tensor_k,
            k_tma_desc_ptrs,
            use_scalar_k_load,
            k_scales,
            real_page_table,
            seqlens_per_query,
            active_width,
            Int32(0),
            Int32(0),
            logits_out,
            tma_atom_k,
        ).launch(
            grid=(
                q_bytes.shape[0],
                self.persistent_ctas,
                1,
            ),
            block=[_PAGED_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_tensor: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        use_scalar_k_load: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        active_width: cute.Tensor,
        source_page_offset: Int32,
        output_width_tokens: Int32,
        logits_out: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
    ):
        tx, _, _ = cute.arch.thread_idx()
        q_idx, cta_idx, _ = cute.arch.block_idx()
        lane = tx % Int32(_WARP_THREADS)
        warp_idx = tx // Int32(_WARP_THREADS)

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._get_shared_storage_cls()

        source_width_tokens = Int32(real_page_table.shape[1]) * Int32(_PAGE_SIZE)
        source_offset_pages = source_page_offset
        if source_offset_pages < Int32(0):
            source_offset_pages = Int32(0)
        source_offset_tokens = source_offset_pages * Int32(_PAGE_SIZE)
        remaining_width_tokens = source_width_tokens - source_offset_tokens
        if remaining_width_tokens < Int32(0):
            remaining_width_tokens = Int32(0)

        width_tokens = output_width_tokens
        if width_tokens <= Int32(0):
            width_tokens = remaining_width_tokens
        valid_width_tokens = width_tokens
        if valid_width_tokens > remaining_width_tokens:
            valid_width_tokens = remaining_width_tokens

        live_width = Int32(active_width[Int32(0)])
        if live_width > source_width_tokens:
            live_width = source_width_tokens
        live_width = live_width - source_offset_tokens
        if live_width < Int32(0):
            live_width = Int32(0)
        if live_width > valid_width_tokens:
            live_width = valid_width_tokens

        seq_len = Int32(seqlens_per_query[q_idx]) - source_offset_tokens
        if seq_len < Int32(0):
            seq_len = Int32(0)
        if seq_len > live_width:
            seq_len = live_width
        total_work = (seq_len + Int32(_PAGE_SIZE - 1)) // Int32(_PAGE_SIZE)

        storage = smem.allocate(SharedStorage)
        mbar_ptr_k = storage.mbar_ptr_k.data_ptr()
        s_q = storage.q_bytes.get_tensor(
            cute.make_layout(
                (self.padded_q_heads, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1)
            )
        )
        s_w = storage.weights.get_tensor(
            cute.make_layout((self.padded_q_heads,), stride=(1,))
        )
        k_page_base_addr = shared_ptr_to_u32(storage.k_page.data_ptr())
        k_page_perm_base_addr = shared_ptr_to_u32(storage.k_page_perm.data_ptr())
        s_k_page_stage = storage.k_page.get_tensor(
            cute.make_layout(
                (_PAGE_SIZE, _INDEX_HEAD_DIM, 1),
                stride=(_INDEX_HEAD_DIM, 1, _PAGE_SIZE * _INDEX_HEAD_DIM),
            )
        )
        s_scale = storage.scales.get_tensor(
            cute.make_layout((_PAGE_SIZE,), stride=(1,))
        )
        s_partial_logits = storage.partial_logits.get_tensor(
            cute.make_layout(
                (self.tokens_per_work, self.num_q_head_tiles),
                stride=(self.num_q_head_tiles, 1),
            )
        )
        load_k_tma, _, _ = cute_copy.tma_get_copy_fn(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.local_tile(k_tma_tensor, (_PAGE_SIZE, _INDEX_HEAD_DIM), (0, 0, None)),
            s_k_page_stage,
        )
        use_scalar_k_load_flag = Int32(use_scalar_k_load[Int32(0)])
        if cta_idx < total_work:
            if tx == 0:
                cute.arch.mbarrier_init(mbar_ptr_k, Int32(1))
            if (warp_idx == Int32(0)) & (use_scalar_k_load_flag == Int32(0)):
                cpasync.prefetch_descriptor(tma_atom_k)

            num_heads = Int32(self.num_heads_static)
            q_linear = tx
            total_q_bytes = Int32(self.padded_q_heads * _INDEX_HEAD_DIM)
            while q_linear < total_q_bytes:
                head_idx = q_linear // Int32(_INDEX_HEAD_DIM)
                col_idx = q_linear - head_idx * Int32(_INDEX_HEAD_DIM)
                s_q[head_idx, col_idx] = (
                    q_bytes[q_idx, head_idx, col_idx]
                    if head_idx < num_heads
                    else cutlass.Uint8(0)
                )
                q_linear += Int32(_PAGED_THREADS_PER_CTA)

            w_linear = tx
            while w_linear < Int32(self.padded_q_heads):
                s_w[w_linear] = (
                    Float32(weights[q_idx, w_linear])
                    if w_linear < num_heads
                    else Float32(0.0)
                )
                w_linear += Int32(_PAGED_THREADS_PER_CTA)
            cute.arch.sync_threads()

            producer_state = cute_pipeline.PipelineStateSimple(1, Int32(0))
            consumer_state = cute_pipeline.PipelineStateSimple(1, Int32(0))
            head_tile_slot = warp_idx % Int32(self.num_q_head_tiles)
            token_group = warp_idx // Int32(self.num_q_head_tiles)
            work_idx = cta_idx
            while work_idx < total_work:
                page_col = work_idx
                source_page_col = source_offset_pages + page_col
                page_base = page_col * Int32(_PAGE_SIZE)
                if (page_base < seq_len) & (
                    source_page_col < Int32(real_page_table.shape[1])
                ):
                    page_id = Int32(real_page_table[q_idx, source_page_col])
                    if page_id >= Int32(0):
                        if use_scalar_k_load_flag != Int32(0):
                            _load_index_k_page_scalar(
                                k_quant_bytes,
                                page_id,
                                s_k_page_stage,
                                tx,
                            )
                        else:
                            if warp_idx == Int32(0):
                                _issue_index_k_tma_copy(
                                    load_k_tma,
                                    producer_state,
                                    mbar_ptr_k,
                                    Int32(_PAGE_SIZE * _INDEX_HEAD_DIM),
                                    page_id,
                                )
                        scale_idx = tx
                        while scale_idx < Int32(_PAGE_SIZE):
                            s_scale[scale_idx] = Float32(k_scales[page_id, scale_idx])
                            scale_idx += Int32(_PAGED_THREADS_PER_CTA)
                        if use_scalar_k_load_flag == Int32(0):
                            cute.arch.mbarrier_wait(
                                mbar_ptr_k + consumer_state.index,
                                phase=consumer_state.phase,
                            )
                        cute.arch.sync_threads()
                        _repack_k_page_to_permuted(
                            k_page_base_addr, k_page_perm_base_addr, tx
                        )
                        cute.arch.sync_threads()

                        valid_slots = seq_len - page_base
                        if valid_slots > Int32(_PAGE_SIZE):
                            valid_slots = Int32(_PAGE_SIZE)
                        split_idx = Int32(0)
                        while split_idx < Int32(self.page_splits):
                            token_base = split_idx * Int32(
                                self.tokens_per_work
                            ) + token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                            zero_idx = tx
                            while zero_idx < Int32(
                                self.tokens_per_work * self.num_q_head_tiles
                            ):
                                token_idx = zero_idx // Int32(self.num_q_head_tiles)
                                head_tile_idx = zero_idx - token_idx * Int32(
                                    self.num_q_head_tiles
                                )
                                s_partial_logits[token_idx, head_tile_idx] = Float32(
                                    0.0
                                )
                                zero_idx += Int32(_PAGED_THREADS_PER_CTA)
                            cute.arch.sync_threads()
                            if token_base < valid_slots:
                                head_tile_base = head_tile_slot * Int32(
                                    _PAGED_Q_HEAD_TILE
                                )
                                _compute_mxfp8_tile_partials(
                                    s_q,
                                    s_w,
                                    num_heads,
                                    k_page_perm_base_addr,
                                    token_base,
                                    head_tile_base,
                                    lane,
                                    s_partial_logits,
                                    token_group * Int32(_PAGED_TOKENS_PER_GROUP),
                                    head_tile_slot,
                                    self.score_mode,
                                )
                            cute.arch.sync_threads()
                            if (head_tile_slot == Int32(0)) & (
                                lane < Int32(_PAGED_TOKENS_PER_GROUP)
                            ):
                                slot_idx = token_base + lane
                                if slot_idx < valid_slots:
                                    logit = Float32(0.0)
                                    head_tile_idx = Int32(0)
                                    partial_row = (
                                        token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                                        + lane
                                    )
                                    while head_tile_idx < Int32(self.num_q_head_tiles):
                                        logit = Float32(
                                            logit
                                            + s_partial_logits[
                                                partial_row, head_tile_idx
                                            ]
                                        )
                                        head_tile_idx += Int32(1)
                                    value = Float32(logit * s_scale[slot_idx])
                                    if cutlass.const_expr(self.tiled_output):
                                        tile_block_q = Int32(self.tile_block_q)
                                        tile_block_k = Int32(self.tile_block_k)
                                        tile_size = Int32(
                                            self.tile_block_q * self.tile_block_k
                                        )
                                        num_k_tiles = (
                                            width_tokens + tile_block_k - Int32(1)
                                        ) // tile_block_k
                                        q_tile_idx = q_idx // tile_block_q
                                        q_local = q_idx - q_tile_idx * tile_block_q
                                        k_tile_idx = page_base // tile_block_k
                                        k_local = (
                                            page_base
                                            - k_tile_idx * tile_block_k
                                            + slot_idx
                                        )
                                        flat_offset = (
                                            q_tile_idx * num_k_tiles * tile_size
                                            + k_tile_idx * tile_size
                                            + q_local * tile_block_k
                                            + k_local
                                        )
                                        logits_out[flat_offset] = value
                                    else:
                                        logits_out[q_idx, page_base + slot_idx] = value
                            cute.arch.sync_threads()
                            split_idx += Int32(1)
                        producer_state.advance()
                        consumer_state.advance()
                        cute.arch.sync_threads()

                work_idx += Int32(self.persistent_ctas)


class SparseNSAPagedSupertileLogitsKernel(SparseNSAPagedLogitsKernel):
    """Paged supertile adapter over a full paged table.

    The shared paged NSA scorer keeps its historical public ABI. This wrapper
    exposes the extra page offset and width scalars only to paged supertile
    callers.
    """

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        use_scalar_k_load: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        active_width: cute.Tensor,
        source_page_offset: Int32,
        output_width_tokens: Int32,
        logits_out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        k_tma_source = _make_paged_index_k_tma_source(k_quant_bytes)
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_tma_source,
            cute.make_layout(
                (_PAGE_SIZE, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1)
            ),
            (_PAGE_SIZE, _INDEX_HEAD_DIM),
            1,
        )
        self.kernel(
            q_bytes,
            weights,
            k_quant_bytes,
            tma_tensor_k,
            k_tma_desc_ptrs,
            use_scalar_k_load,
            k_scales,
            real_page_table,
            seqlens_per_query,
            active_width,
            source_page_offset,
            output_width_tokens,
            logits_out,
            tma_atom_k,
        ).launch(
            grid=(
                q_bytes.shape[0],
                self.persistent_ctas,
                1,
            ),
            block=[_PAGED_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )


def _should_use_schedule_kernel(
    *,
    q_rows: int,
    max_pages: int,
) -> bool:
    return q_rows > 0 and max_pages >= _SCHEDULE_MIN_PAGES


def _should_use_schedule_single_row_kernel(
    *,
    q_rows: int,
    max_pages: int,
) -> bool:
    return q_rows == 1 and _should_use_schedule_kernel(
        q_rows=q_rows, max_pages=max_pages
    )


def _should_use_schedule_multi_row_kernel(
    *,
    q_rows: int,
    max_pages: int,
) -> bool:
    return (
        _ENABLE_MULTI_ROW_SCHEDULE
        and q_rows > 1
        and q_rows <= _SCHEDULE_MULTI_ROW_MAX_Q_ROWS
        and _should_use_schedule_kernel(q_rows=q_rows, max_pages=max_pages)
    )


class SparseNSAScheduledSingleRowLogitsKernel:
    """Schedule-driven scorer for the long single-row decode case."""

    def __init__(
        self,
        parallel_ctas: int,
        num_heads_static: int,
        score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
        output_mode: int = IndexerOutputMode.TOKEN_LOGITS,
    ):
        self.parallel_ctas = int(parallel_ctas)
        self.num_heads_static = int(num_heads_static)
        self.score_mode = int(score_mode)
        self.output_mode = int(output_mode)
        self.num_q_head_tiles = _num_q_head_tiles(self.num_heads_static)
        if self.num_q_head_tiles not in (1, 2, 4):
            raise ValueError(
                f"paged logits kernel only supports 1/2/4 head tiles, got {self.num_q_head_tiles}"
            )
        if (
            self.output_mode == IndexerOutputMode.PAGE_HEAD_MAX
            and self.num_heads_static > _PAGED_Q_HEAD_TILE
        ):
            raise ValueError("scheduled page-head-max output supports at most 16 heads")
        self.padded_q_heads = self.num_q_head_tiles * _PAGED_Q_HEAD_TILE
        self.token_groups = _PAGED_WARPS_PER_CTA // self.num_q_head_tiles
        self.tokens_per_work = _PAGED_TOKENS_PER_GROUP * self.token_groups
        self.page_splits = _PAGE_SIZE // self.tokens_per_work
        self.schedule_pages_per_split = int(PAGED_MQA_LOGITS_SCHEDULE_PAGES_PER_SPLIT)

    def _get_shared_storage_cls(self):
        return _paged_indexer_shared_storage_cls(
            self.padded_q_heads,
            self.tokens_per_work,
            self.num_q_head_tiles,
        )

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        use_scalar_k_load: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        schedule_metadata: cute.Tensor,
        active_width: cute.Tensor,
        logits_out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        k_tma_source = _make_paged_index_k_tma_source(k_quant_bytes)
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_tma_source,
            cute.make_layout(
                (_PAGE_SIZE, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1)
            ),
            (_PAGE_SIZE, _INDEX_HEAD_DIM),
            1,
        )
        self.kernel(
            q_bytes,
            weights,
            k_quant_bytes,
            tma_tensor_k,
            k_tma_desc_ptrs,
            use_scalar_k_load,
            k_scales,
            real_page_table,
            seqlens_per_query,
            schedule_metadata,
            active_width,
            logits_out,
            tma_atom_k,
        ).launch(
            grid=(
                schedule_metadata.shape[0] - 1,
                self.parallel_ctas,
                1,
            ),
            block=[_PAGED_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_tensor: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        use_scalar_k_load: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        schedule_metadata: cute.Tensor,
        active_width: cute.Tensor,
        logits_out: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
    ):
        tx, _, _ = cute.arch.thread_idx()
        interval_idx, cta_lane_idx, _ = cute.arch.block_idx()
        lane = tx % Int32(_WARP_THREADS)
        warp_idx = tx // Int32(_WARP_THREADS)

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._get_shared_storage_cls()

        width_tokens = Int32(real_page_table.shape[1]) * Int32(_PAGE_SIZE)
        live_width = Int32(active_width[Int32(0)])
        if live_width > width_tokens:
            live_width = width_tokens
        seq_len = Int32(seqlens_per_query[Int32(0)])
        if seq_len > live_width:
            seq_len = live_width
        live_pages = (seq_len + Int32(_PAGE_SIZE - 1)) // Int32(_PAGE_SIZE)

        start_q_idx = Int32(schedule_metadata[interval_idx, Int32(0)])
        start_split_idx = Int32(schedule_metadata[interval_idx, Int32(1)])
        end_q_idx = Int32(schedule_metadata[interval_idx + Int32(1), Int32(0)])
        end_split_idx = Int32(schedule_metadata[interval_idx + Int32(1), Int32(1)])

        interval_page_start = start_split_idx * Int32(self.schedule_pages_per_split)
        interval_page_end = live_pages
        if end_q_idx == Int32(0):
            interval_page_end = end_split_idx * Int32(self.schedule_pages_per_split)
            if interval_page_end > live_pages:
                interval_page_end = live_pages

        storage = smem.allocate(SharedStorage)
        mbar_ptr_k = storage.mbar_ptr_k.data_ptr()
        s_q = storage.q_bytes.get_tensor(
            cute.make_layout(
                (self.padded_q_heads, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1)
            )
        )
        s_w = storage.weights.get_tensor(
            cute.make_layout((self.padded_q_heads,), stride=(1,))
        )
        k_page_base_addr = shared_ptr_to_u32(storage.k_page.data_ptr())
        k_page_perm_base_addr = shared_ptr_to_u32(storage.k_page_perm.data_ptr())
        s_k_page_stage = storage.k_page.get_tensor(
            cute.make_layout(
                (_PAGE_SIZE, _INDEX_HEAD_DIM, 1),
                stride=(_INDEX_HEAD_DIM, 1, _PAGE_SIZE * _INDEX_HEAD_DIM),
            )
        )
        s_scale = storage.scales.get_tensor(
            cute.make_layout((_PAGE_SIZE,), stride=(1,))
        )
        s_partial_logits = storage.partial_logits.get_tensor(
            cute.make_layout(
                (self.tokens_per_work, self.num_q_head_tiles),
                stride=(self.num_q_head_tiles, 1),
            )
        )
        s_page_partials = storage.partial_logits.get_tensor(
            cute.make_layout(
                (_PAGED_WARPS_PER_CTA, _PAGED_Q_HEAD_TILE),
                stride=(_PAGED_Q_HEAD_TILE, 1),
            )
        )
        load_k_tma, _, _ = cute_copy.tma_get_copy_fn(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.local_tile(k_tma_tensor, (_PAGE_SIZE, _INDEX_HEAD_DIM), (0, 0, None)),
            s_k_page_stage,
        )
        use_scalar_k_load_flag = Int32(use_scalar_k_load[Int32(0)])
        PAGE_HEAD_MAX = cutlass.const_expr(
            self.output_mode == IndexerOutputMode.PAGE_HEAD_MAX
        )

        if (
            (start_q_idx == Int32(0))
            & (interval_page_start < interval_page_end)
            & (cta_lane_idx < Int32(self.parallel_ctas))
        ):
            if tx == 0:
                cute.arch.mbarrier_init(mbar_ptr_k, Int32(1))
            if (warp_idx == Int32(0)) & (use_scalar_k_load_flag == Int32(0)):
                cpasync.prefetch_descriptor(tma_atom_k)

            num_heads = Int32(self.num_heads_static)
            q_linear = tx
            total_q_bytes = Int32(self.padded_q_heads * _INDEX_HEAD_DIM)
            while q_linear < total_q_bytes:
                head_idx = q_linear // Int32(_INDEX_HEAD_DIM)
                col_idx = q_linear - head_idx * Int32(_INDEX_HEAD_DIM)
                s_q[head_idx, col_idx] = (
                    q_bytes[Int32(0), head_idx, col_idx]
                    if head_idx < num_heads
                    else cutlass.Uint8(0)
                )
                q_linear += Int32(_PAGED_THREADS_PER_CTA)

            w_linear = tx
            while w_linear < Int32(self.padded_q_heads):
                s_w[w_linear] = (
                    Float32(weights[Int32(0), w_linear])
                    if w_linear < num_heads
                    else Float32(0.0)
                )
                w_linear += Int32(_PAGED_THREADS_PER_CTA)
            cute.arch.sync_threads()

            producer_state = cute_pipeline.PipelineStateSimple(1, Int32(0))
            consumer_state = cute_pipeline.PipelineStateSimple(1, Int32(0))
            head_tile_slot = warp_idx % Int32(self.num_q_head_tiles)
            token_group = warp_idx // Int32(self.num_q_head_tiles)
            page_col = interval_page_start + cta_lane_idx
            while page_col < interval_page_end:
                page_base = page_col * Int32(_PAGE_SIZE)
                if page_base < seq_len:
                    page_id = Int32(real_page_table[Int32(0), page_col])
                    if page_id >= Int32(0):
                        if use_scalar_k_load_flag != Int32(0):
                            _load_index_k_page_scalar(
                                k_quant_bytes,
                                page_id,
                                s_k_page_stage,
                                tx,
                            )
                        else:
                            if warp_idx == Int32(0):
                                _issue_index_k_tma_copy(
                                    load_k_tma,
                                    producer_state,
                                    mbar_ptr_k,
                                    Int32(_PAGE_SIZE * _INDEX_HEAD_DIM),
                                    page_id,
                                )
                        scale_idx = tx
                        while scale_idx < Int32(_PAGE_SIZE):
                            s_scale[scale_idx] = Float32(k_scales[page_id, scale_idx])
                            scale_idx += Int32(_PAGED_THREADS_PER_CTA)
                        if use_scalar_k_load_flag == Int32(0):
                            cute.arch.mbarrier_wait(
                                mbar_ptr_k + consumer_state.index,
                                phase=consumer_state.phase,
                            )
                        cute.arch.sync_threads()
                        _repack_k_page_to_permuted(
                            k_page_base_addr, k_page_perm_base_addr, tx
                        )
                        cute.arch.sync_threads()

                        valid_slots = seq_len - page_base
                        if valid_slots > Int32(_PAGE_SIZE):
                            valid_slots = Int32(_PAGE_SIZE)
                        zero_idx = Int32(0)
                        if cutlass.const_expr(PAGE_HEAD_MAX):
                            zero_idx = tx
                            while zero_idx < Int32(
                                _PAGED_WARPS_PER_CTA * _PAGED_Q_HEAD_TILE
                            ):
                                group_idx = zero_idx // Int32(_PAGED_Q_HEAD_TILE)
                                head_idx = zero_idx - group_idx * Int32(
                                    _PAGED_Q_HEAD_TILE
                                )
                                s_page_partials[group_idx, head_idx] = Float32(
                                    -Float32.inf
                                )
                                zero_idx += Int32(_PAGED_THREADS_PER_CTA)
                            cute.arch.sync_threads()
                        split_idx = Int32(0)
                        while split_idx < Int32(self.page_splits):
                            token_base = split_idx * Int32(
                                self.tokens_per_work
                            ) + token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                            if cutlass.const_expr(not PAGE_HEAD_MAX):
                                zero_idx = tx
                                while zero_idx < Int32(
                                    self.tokens_per_work * self.num_q_head_tiles
                                ):
                                    token_idx = zero_idx // Int32(self.num_q_head_tiles)
                                    head_tile_idx = zero_idx - token_idx * Int32(
                                        self.num_q_head_tiles
                                    )
                                    s_partial_logits[token_idx, head_tile_idx] = (
                                        Float32(0.0)
                                    )
                                    zero_idx += Int32(_PAGED_THREADS_PER_CTA)
                                cute.arch.sync_threads()
                            if token_base < valid_slots:
                                head_tile_base = head_tile_slot * Int32(
                                    _PAGED_Q_HEAD_TILE
                                )
                                if cutlass.const_expr(PAGE_HEAD_MAX):
                                    _compute_mxfp8_tile_head_token_max(
                                        s_q,
                                        s_w,
                                        num_heads,
                                        k_page_perm_base_addr,
                                        token_base,
                                        head_tile_base,
                                        lane,
                                        s_scale,
                                        valid_slots,
                                        s_page_partials,
                                        token_group,
                                    )
                                else:
                                    _compute_mxfp8_tile_partials(
                                        s_q,
                                        s_w,
                                        num_heads,
                                        k_page_perm_base_addr,
                                        token_base,
                                        head_tile_base,
                                        lane,
                                        s_partial_logits,
                                        token_group * Int32(_PAGED_TOKENS_PER_GROUP),
                                        head_tile_slot,
                                        self.score_mode,
                                    )
                            cute.arch.sync_threads()
                            if cutlass.const_expr(not PAGE_HEAD_MAX):
                                if (head_tile_slot == Int32(0)) & (
                                    lane < Int32(_PAGED_TOKENS_PER_GROUP)
                                ):
                                    slot_idx = token_base + lane
                                    if slot_idx < valid_slots:
                                        logit = Float32(0.0)
                                        head_tile_idx = Int32(0)
                                        partial_row = (
                                            token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                                            + lane
                                        )
                                        while head_tile_idx < Int32(
                                            self.num_q_head_tiles
                                        ):
                                            logit = Float32(
                                                logit
                                                + s_partial_logits[
                                                    partial_row, head_tile_idx
                                                ]
                                            )
                                            head_tile_idx += Int32(1)
                                        logits_out[Int32(0), page_base + slot_idx] = (
                                            Float32(logit * s_scale[slot_idx])
                                        )
                            cute.arch.sync_threads()
                            split_idx += Int32(1)
                        if cutlass.const_expr(PAGE_HEAD_MAX):
                            if tx < Int32(_PAGED_Q_HEAD_TILE):
                                head_idx = tx
                                if head_idx < num_heads:
                                    page_max = Float32(-Float32.inf)
                                    group_idx = Int32(0)
                                    while group_idx < Int32(_PAGED_WARPS_PER_CTA):
                                        page_max = attention_ops.fmax(
                                            page_max,
                                            s_page_partials[group_idx, head_idx],
                                        )
                                        group_idx += Int32(1)
                                    logits_out[head_idx, Int32(0), page_col] = page_max
                            cute.arch.sync_threads()
                        producer_state.advance()
                        consumer_state.advance()
                        cute.arch.sync_threads()

                page_col += Int32(self.parallel_ctas)


class SparseNSAScheduledMultiRowLogitsKernel:
    """Schedule-driven scorer for long multi-row decode."""

    def __init__(
        self,
        parallel_ctas: int,
        num_heads_static: int,
        score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
        output_mode: int = IndexerOutputMode.TOKEN_LOGITS,
    ):
        self.parallel_ctas = int(parallel_ctas)
        self.num_heads_static = int(num_heads_static)
        self.score_mode = int(score_mode)
        self.output_mode = int(output_mode)
        self.num_q_head_tiles = _num_q_head_tiles(self.num_heads_static)
        if self.num_q_head_tiles not in (1, 2, 4):
            raise ValueError(
                f"paged logits kernel only supports 1/2/4 head tiles, got {self.num_q_head_tiles}"
            )
        if (
            self.output_mode == IndexerOutputMode.PAGE_HEAD_MAX
            and self.num_heads_static > _PAGED_Q_HEAD_TILE
        ):
            raise ValueError("scheduled page-head-max output supports at most 16 heads")
        self.padded_q_heads = self.num_q_head_tiles * _PAGED_Q_HEAD_TILE
        self.token_groups = _PAGED_WARPS_PER_CTA // self.num_q_head_tiles
        self.tokens_per_work = _PAGED_TOKENS_PER_GROUP * self.token_groups
        self.page_splits = _PAGE_SIZE // self.tokens_per_work
        self.schedule_pages_per_split = int(PAGED_MQA_LOGITS_SCHEDULE_PAGES_PER_SPLIT)

    def _get_shared_storage_cls(self):
        return _paged_indexer_shared_storage_cls(
            self.padded_q_heads,
            self.tokens_per_work,
            self.num_q_head_tiles,
        )

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        use_scalar_k_load: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        schedule_metadata: cute.Tensor,
        active_width: cute.Tensor,
        logits_out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        k_tma_source = _make_paged_index_k_tma_source(k_quant_bytes)
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_tma_source,
            cute.make_layout(
                (_PAGE_SIZE, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1)
            ),
            (_PAGE_SIZE, _INDEX_HEAD_DIM),
            1,
        )
        self.kernel(
            q_bytes,
            weights,
            k_quant_bytes,
            tma_tensor_k,
            k_tma_desc_ptrs,
            use_scalar_k_load,
            k_scales,
            real_page_table,
            seqlens_per_query,
            schedule_metadata,
            active_width,
            logits_out,
            tma_atom_k,
        ).launch(
            grid=(
                schedule_metadata.shape[0] - 1,
                self.parallel_ctas,
                1,
            ),
            block=[_PAGED_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_tensor: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        use_scalar_k_load: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        schedule_metadata: cute.Tensor,
        active_width: cute.Tensor,
        logits_out: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
    ):
        tx, _, _ = cute.arch.thread_idx()
        interval_idx, cta_lane_idx, _ = cute.arch.block_idx()
        lane = tx % Int32(_WARP_THREADS)
        warp_idx = tx // Int32(_WARP_THREADS)

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._get_shared_storage_cls()

        width_tokens = Int32(real_page_table.shape[1]) * Int32(_PAGE_SIZE)
        live_width = Int32(active_width[Int32(0)])
        if live_width > width_tokens:
            live_width = width_tokens
        q_rows = Int32(seqlens_per_query.shape[0])
        total_schedule_intervals = Int32(schedule_metadata.shape[0] - 1)
        start_q_idx = Int32(schedule_metadata[interval_idx, Int32(0)])
        start_split_idx = Int32(schedule_metadata[interval_idx, Int32(1)])
        end_q_idx = Int32(schedule_metadata[interval_idx + Int32(1), Int32(0)])
        end_split_idx = Int32(schedule_metadata[interval_idx + Int32(1), Int32(1)])

        storage = smem.allocate(SharedStorage)
        mbar_ptr_k = storage.mbar_ptr_k.data_ptr()
        s_q = storage.q_bytes.get_tensor(
            cute.make_layout(
                (self.padded_q_heads, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1)
            )
        )
        s_w = storage.weights.get_tensor(
            cute.make_layout((self.padded_q_heads,), stride=(1,))
        )
        k_page_base_addr = shared_ptr_to_u32(storage.k_page.data_ptr())
        k_page_perm_base_addr = shared_ptr_to_u32(storage.k_page_perm.data_ptr())
        s_k_page_stage = storage.k_page.get_tensor(
            cute.make_layout(
                (_PAGE_SIZE, _INDEX_HEAD_DIM, 1),
                stride=(_INDEX_HEAD_DIM, 1, _PAGE_SIZE * _INDEX_HEAD_DIM),
            )
        )
        s_scale = storage.scales.get_tensor(
            cute.make_layout((_PAGE_SIZE,), stride=(1,))
        )
        s_partial_logits = storage.partial_logits.get_tensor(
            cute.make_layout(
                (self.tokens_per_work, self.num_q_head_tiles),
                stride=(self.num_q_head_tiles, 1),
            )
        )
        s_page_partials = storage.partial_logits.get_tensor(
            cute.make_layout(
                (_PAGED_WARPS_PER_CTA, _PAGED_Q_HEAD_TILE),
                stride=(_PAGED_Q_HEAD_TILE, 1),
            )
        )
        load_k_tma, _, _ = cute_copy.tma_get_copy_fn(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.local_tile(k_tma_tensor, (_PAGE_SIZE, _INDEX_HEAD_DIM), (0, 0, None)),
            s_k_page_stage,
        )
        use_scalar_k_load_flag = Int32(use_scalar_k_load[Int32(0)])
        PAGE_HEAD_MAX = cutlass.const_expr(
            self.output_mode == IndexerOutputMode.PAGE_HEAD_MAX
        )

        if (interval_idx < total_schedule_intervals) & (
            cta_lane_idx < Int32(self.parallel_ctas)
        ):
            if tx == 0:
                cute.arch.mbarrier_init(mbar_ptr_k, Int32(1))
            if (warp_idx == Int32(0)) & (use_scalar_k_load_flag == Int32(0)):
                cpasync.prefetch_descriptor(tma_atom_k)

            num_heads = Int32(self.num_heads_static)
            producer_state = cute_pipeline.PipelineStateSimple(1, Int32(0))
            consumer_state = cute_pipeline.PipelineStateSimple(1, Int32(0))
            head_tile_slot = warp_idx % Int32(self.num_q_head_tiles)
            token_group = warp_idx // Int32(self.num_q_head_tiles)
            current_q_idx = start_q_idx
            current_split_idx = start_split_idx

            while (current_q_idx < q_rows) & (
                (current_q_idx < end_q_idx)
                | ((current_q_idx == end_q_idx) & (end_split_idx > Int32(0)))
            ):
                seq_len = Int32(seqlens_per_query[current_q_idx])
                if seq_len > live_width:
                    seq_len = live_width
                live_pages = (seq_len + Int32(_PAGE_SIZE - 1)) // Int32(_PAGE_SIZE)
                row_total_splits = (
                    live_pages + Int32(self.schedule_pages_per_split - 1)
                ) // Int32(self.schedule_pages_per_split)
                row_split_start = (
                    current_split_idx if current_q_idx == start_q_idx else Int32(0)
                )
                row_split_end = row_total_splits
                if current_q_idx == end_q_idx:
                    row_split_end = end_split_idx
                    if row_split_end > row_total_splits:
                        row_split_end = row_total_splits

                if row_split_start < row_split_end:
                    q_linear = tx
                    total_q_bytes = Int32(self.padded_q_heads * _INDEX_HEAD_DIM)
                    while q_linear < total_q_bytes:
                        head_idx = q_linear // Int32(_INDEX_HEAD_DIM)
                        col_idx = q_linear - head_idx * Int32(_INDEX_HEAD_DIM)
                        s_q[head_idx, col_idx] = (
                            q_bytes[current_q_idx, head_idx, col_idx]
                            if head_idx < num_heads
                            else cutlass.Uint8(0)
                        )
                        q_linear += Int32(_PAGED_THREADS_PER_CTA)

                    w_linear = tx
                    while w_linear < Int32(self.padded_q_heads):
                        s_w[w_linear] = (
                            Float32(weights[current_q_idx, w_linear])
                            if w_linear < num_heads
                            else Float32(0.0)
                        )
                        w_linear += Int32(_PAGED_THREADS_PER_CTA)
                    cute.arch.sync_threads()

                    row_page_end = row_split_end * Int32(self.schedule_pages_per_split)
                    if row_page_end > live_pages:
                        row_page_end = live_pages
                    page_col = (
                        row_split_start * Int32(self.schedule_pages_per_split)
                        + cta_lane_idx
                    )
                    while page_col < row_page_end:
                        page_base = page_col * Int32(_PAGE_SIZE)
                        if page_base < seq_len:
                            page_id = Int32(real_page_table[current_q_idx, page_col])
                            if page_id >= Int32(0):
                                if use_scalar_k_load_flag != Int32(0):
                                    _load_index_k_page_scalar(
                                        k_quant_bytes,
                                        page_id,
                                        s_k_page_stage,
                                        tx,
                                    )
                                else:
                                    if warp_idx == Int32(0):
                                        _issue_index_k_tma_copy(
                                            load_k_tma,
                                            producer_state,
                                            mbar_ptr_k,
                                            Int32(_PAGE_SIZE * _INDEX_HEAD_DIM),
                                            page_id,
                                        )
                                scale_idx = tx
                                while scale_idx < Int32(_PAGE_SIZE):
                                    s_scale[scale_idx] = Float32(
                                        k_scales[page_id, scale_idx]
                                    )
                                    scale_idx += Int32(_PAGED_THREADS_PER_CTA)
                                if use_scalar_k_load_flag == Int32(0):
                                    cute.arch.mbarrier_wait(
                                        mbar_ptr_k + consumer_state.index,
                                        phase=consumer_state.phase,
                                    )
                                cute.arch.sync_threads()
                                _repack_k_page_to_permuted(
                                    k_page_base_addr, k_page_perm_base_addr, tx
                                )
                                cute.arch.sync_threads()

                                valid_slots = seq_len - page_base
                                if valid_slots > Int32(_PAGE_SIZE):
                                    valid_slots = Int32(_PAGE_SIZE)
                                zero_idx = Int32(0)
                                if cutlass.const_expr(PAGE_HEAD_MAX):
                                    zero_idx = tx
                                    while zero_idx < Int32(
                                        _PAGED_WARPS_PER_CTA * _PAGED_Q_HEAD_TILE
                                    ):
                                        group_idx = zero_idx // Int32(
                                            _PAGED_Q_HEAD_TILE
                                        )
                                        head_idx = zero_idx - group_idx * Int32(
                                            _PAGED_Q_HEAD_TILE
                                        )
                                        s_page_partials[group_idx, head_idx] = Float32(
                                            -Float32.inf
                                        )
                                        zero_idx += Int32(_PAGED_THREADS_PER_CTA)
                                    cute.arch.sync_threads()
                                split_idx = Int32(0)
                                while split_idx < Int32(self.page_splits):
                                    token_base = split_idx * Int32(
                                        self.tokens_per_work
                                    ) + token_group * Int32(_PAGED_TOKENS_PER_GROUP)
                                    if cutlass.const_expr(not PAGE_HEAD_MAX):
                                        zero_idx = tx
                                        while zero_idx < Int32(
                                            self.tokens_per_work * self.num_q_head_tiles
                                        ):
                                            token_idx = zero_idx // Int32(
                                                self.num_q_head_tiles
                                            )
                                            head_tile_idx = (
                                                zero_idx
                                                - token_idx
                                                * Int32(self.num_q_head_tiles)
                                            )
                                            s_partial_logits[
                                                token_idx, head_tile_idx
                                            ] = Float32(0.0)
                                            zero_idx += Int32(_PAGED_THREADS_PER_CTA)
                                        cute.arch.sync_threads()
                                    if token_base < valid_slots:
                                        head_tile_base = head_tile_slot * Int32(
                                            _PAGED_Q_HEAD_TILE
                                        )
                                        if cutlass.const_expr(PAGE_HEAD_MAX):
                                            _compute_mxfp8_tile_head_token_max(
                                                s_q,
                                                s_w,
                                                num_heads,
                                                k_page_perm_base_addr,
                                                token_base,
                                                head_tile_base,
                                                lane,
                                                s_scale,
                                                valid_slots,
                                                s_page_partials,
                                                token_group,
                                            )
                                        else:
                                            _compute_mxfp8_tile_partials(
                                                s_q,
                                                s_w,
                                                num_heads,
                                                k_page_perm_base_addr,
                                                token_base,
                                                head_tile_base,
                                                lane,
                                                s_partial_logits,
                                                token_group
                                                * Int32(_PAGED_TOKENS_PER_GROUP),
                                                head_tile_slot,
                                                self.score_mode,
                                            )
                                    cute.arch.sync_threads()
                                    if cutlass.const_expr(not PAGE_HEAD_MAX):
                                        if head_tile_slot == Int32(0) and lane < Int32(
                                            _PAGED_TOKENS_PER_GROUP
                                        ):
                                            slot_idx = token_base + lane
                                            if slot_idx < valid_slots:
                                                logit = Float32(0.0)
                                                head_tile_idx = Int32(0)
                                                partial_row = (
                                                    token_group
                                                    * Int32(_PAGED_TOKENS_PER_GROUP)
                                                    + lane
                                                )
                                                while head_tile_idx < Int32(
                                                    self.num_q_head_tiles
                                                ):
                                                    logit = Float32(
                                                        logit
                                                        + s_partial_logits[
                                                            partial_row, head_tile_idx
                                                        ]
                                                    )
                                                    head_tile_idx += Int32(1)
                                                logits_out[
                                                    current_q_idx, page_base + slot_idx
                                                ] = Float32(logit * s_scale[slot_idx])
                                    cute.arch.sync_threads()
                                    split_idx += Int32(1)
                                if cutlass.const_expr(PAGE_HEAD_MAX):
                                    if tx < Int32(_PAGED_Q_HEAD_TILE):
                                        head_idx = tx
                                        if head_idx < num_heads:
                                            page_max = Float32(-Float32.inf)
                                            group_idx = Int32(0)
                                            while group_idx < Int32(
                                                _PAGED_WARPS_PER_CTA
                                            ):
                                                page_max = attention_ops.fmax(
                                                    page_max,
                                                    s_page_partials[
                                                        group_idx, head_idx
                                                    ],
                                                )
                                                group_idx += Int32(1)
                                            logits_out[
                                                head_idx, current_q_idx, page_col
                                            ] = page_max
                                    cute.arch.sync_threads()
                                producer_state.advance()
                                consumer_state.advance()
                                cute.arch.sync_threads()

                        page_col += Int32(self.parallel_ctas)

                current_q_idx += Int32(1)
                current_split_idx = Int32(0)


@lru_cache(maxsize=32)
def _build_sparse_nsa_paged_kernel(
    persistent_ctas: int,
    num_heads_static: int,
    score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
) -> SparseNSAPagedLogitsKernel:
    return SparseNSAPagedLogitsKernel(
        persistent_ctas, num_heads_static, score_mode=score_mode
    )


@lru_cache(maxsize=32)
def _build_sparse_nsa_paged_tiled_kernel(
    persistent_ctas: int,
    num_heads_static: int,
    tile_block_q: int,
    tile_block_k: int,
    score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
) -> SparseNSAPagedLogitsKernel:
    return SparseNSAPagedLogitsKernel(
        persistent_ctas,
        num_heads_static,
        tiled_output=True,
        tile_block_q=tile_block_q,
        tile_block_k=tile_block_k,
        score_mode=score_mode,
    )


@lru_cache(maxsize=32)
def _build_sparse_nsa_paged_supertile_kernel(
    persistent_ctas: int,
    num_heads_static: int,
    tile_block_q: int,
    tile_block_k: int,
    score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
) -> SparseNSAPagedSupertileLogitsKernel:
    return SparseNSAPagedSupertileLogitsKernel(
        persistent_ctas,
        num_heads_static,
        tiled_output=True,
        tile_block_q=tile_block_q,
        tile_block_k=tile_block_k,
        score_mode=score_mode,
    )


@lru_cache(maxsize=16)
def _build_sparse_nsa_schedule_single_row_kernel(
    parallel_ctas: int,
    num_heads_static: int,
    score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
    output_mode: int = IndexerOutputMode.TOKEN_LOGITS,
) -> SparseNSAScheduledSingleRowLogitsKernel:
    return SparseNSAScheduledSingleRowLogitsKernel(
        parallel_ctas,
        num_heads_static,
        score_mode,
        output_mode,
    )


@lru_cache(maxsize=16)
def _build_sparse_nsa_schedule_multi_row_kernel(
    parallel_ctas: int,
    num_heads_static: int,
    score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
    output_mode: int = IndexerOutputMode.TOKEN_LOGITS,
) -> SparseNSAScheduledMultiRowLogitsKernel:
    return SparseNSAScheduledMultiRowLogitsKernel(
        parallel_ctas,
        num_heads_static,
        score_mode,
        output_mode,
    )


def _split_index_k_cache_runtime_views(
    index_k_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if index_k_cache.stride(-1) != 1:
        raise ValueError(
            f"index_k_cache must have a contiguous last dimension, got stride={index_k_cache.stride()}"
        )
    num_pages = index_k_cache.shape[0]
    data_bytes = _PAGE_SIZE * _INDEX_HEAD_DIM
    k_quant_bytes = index_k_cache[:, :data_bytes].view(
        num_pages, _PAGE_SIZE, _INDEX_HEAD_DIM
    )
    k_scales = (
        index_k_cache[:, data_bytes : data_bytes + _PAGE_SIZE * _SCALE_BYTES]
        .view(num_pages, _PAGE_SIZE, _SCALE_BYTES)
        .view(torch.float32)
        .squeeze(-1)
    )
    return k_quant_bytes, k_scales


def clear_indexer_kernel_cache() -> None:
    _build_sparse_nsa_paged_kernel.cache_clear()
    _build_sparse_nsa_paged_tiled_kernel.cache_clear()
    _build_sparse_nsa_paged_supertile_kernel.cache_clear()
    _build_sparse_nsa_schedule_single_row_kernel.cache_clear()
    _build_sparse_nsa_schedule_multi_row_kernel.cache_clear()


def supports_paged_logits_kernel(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    page_size: int,
) -> bool:
    if page_size != _PAGE_SIZE:
        return False
    if q_fp8.device.type != "cuda":
        return False
    if not (
        weights.device
        == index_k_cache.device
        == real_page_table.device
        == seqlens_per_query.device
        == q_fp8.device
    ):
        return False
    if q_fp8.ndim != 3 or q_fp8.shape[2] != _INDEX_HEAD_DIM:
        return False
    if q_fp8.shape[1] > _MAX_SUPPORTED_Q_HEADS:
        return False
    if _num_q_head_tiles(q_fp8.shape[1]) not in (1, 2, 4):
        return False
    if weights.ndim != 2 or weights.shape != q_fp8.shape[:2]:
        return False
    if real_page_table.ndim != 2:
        return False
    if seqlens_per_query.ndim != 1 or seqlens_per_query.shape[0] != q_fp8.shape[0]:
        return False
    if index_k_cache.ndim != 2 or index_k_cache.shape[1] != _PAGE_SIZE * (
        _INDEX_HEAD_DIM + _SCALE_BYTES
    ):
        return False
    if q_fp8.dtype != torch.float8_e4m3fn:
        return False
    if weights.dtype != torch.float32:
        return False
    if index_k_cache.dtype != torch.uint8:
        return False
    if index_k_cache.stride(-1) != 1:
        return False
    if real_page_table.dtype != torch.int32 or seqlens_per_query.dtype != torch.int32:
        return False
    return True


def run_paged_logits_kernel(
    *,
    q_fp8: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    index_k_cache: torch.Tensor | None = None,
    real_page_table: torch.Tensor | None = None,
    seqlens_per_query: torch.Tensor | None = None,
    schedule_metadata: torch.Tensor | None = None,
    active_width: torch.Tensor | None = None,
    page_size: int | None = None,
    preinitialize_invalid_logits: bool | None = None,
    score_mode: int | None = None,
    output_mode: int | None = None,
    page_scores: torch.Tensor | None = None,
    binding: IndexerPagedLogitsKernelBinding | None = None,
) -> torch.Tensor:
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("q_fp8", q_fp8),
                ("weights", weights),
                ("index_k_cache", index_k_cache),
                ("real_page_table", real_page_table),
                ("seqlens_per_query", seqlens_per_query),
                ("schedule_metadata", schedule_metadata),
                ("active_width", active_width),
                ("page_size", page_size),
                ("preinitialize_invalid_logits", preinitialize_invalid_logits),
                ("score_mode", score_mode),
                ("output_mode", output_mode),
                ("page_scores", page_scores),
            )
            if value is not None
        ]
        if extras:
            _raise_binding_extras("run_paged_logits_kernel", extras)
        q_fp8 = binding.q_fp8
        weights = binding.weights
        index_k_cache = binding.index_k_cache
        real_page_table = binding.real_page_table
        seqlens_per_query = binding.seqlens_per_query
        schedule_metadata = binding.schedule_metadata
        active_width = binding.active_width
        page_size = binding.page_size
        preinitialize_invalid_logits = binding.preinitialize_invalid_logits
        score_mode = binding.score_mode
        output_mode = binding.output_mode
        page_scores = binding.page_scores

    q_fp8 = _require_bound_arg(q_fp8, api_name="run_paged_logits_kernel", name="q_fp8")
    weights = _require_bound_arg(
        weights, api_name="run_paged_logits_kernel", name="weights"
    )
    index_k_cache = _require_bound_arg(
        index_k_cache,
        api_name="run_paged_logits_kernel",
        name="index_k_cache",
    )
    real_page_table = _require_bound_arg(
        real_page_table,
        api_name="run_paged_logits_kernel",
        name="real_page_table",
    )
    seqlens_per_query = _require_bound_arg(
        seqlens_per_query,
        api_name="run_paged_logits_kernel",
        name="seqlens_per_query",
    )
    page_size = _PAGE_SIZE if page_size is None else int(page_size)
    preinitialize_invalid_logits = (
        True
        if preinitialize_invalid_logits is None
        else bool(preinitialize_invalid_logits)
    )
    score_mode = (
        IndexerScoreMode.NSA_RELU_SUM if score_mode is None else int(score_mode)
    )
    if score_mode not in (IndexerScoreMode.NSA_RELU_SUM, IndexerScoreMode.MSA_BILINEAR):
        raise ValueError(f"unsupported indexer score_mode {score_mode}")
    output_mode = (
        IndexerOutputMode.TOKEN_LOGITS if output_mode is None else int(output_mode)
    )
    if output_mode not in (
        IndexerOutputMode.TOKEN_LOGITS,
        IndexerOutputMode.PAGE_HEAD_MAX,
    ):
        raise ValueError(f"unsupported indexer output_mode {output_mode}")
    if output_mode == IndexerOutputMode.PAGE_HEAD_MAX:
        if score_mode != IndexerScoreMode.MSA_BILINEAR:
            raise ValueError("PAGE_HEAD_MAX output requires MSA_BILINEAR score mode")
        if int(q_fp8.shape[1]) > _PAGED_Q_HEAD_TILE:
            raise ValueError("PAGE_HEAD_MAX output supports at most 16 heads")

    if not supports_paged_logits_kernel(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        page_size=page_size,
    ):
        raise ValueError(
            "sparse NSA paged logits kernel only supports the exact CUDA page_size=64 FP8 contract"
        )

    rows = q_fp8.shape[0]
    width_tokens = real_page_table.shape[1] * page_size
    if rows == 0 or width_tokens == 0:
        if output_mode == IndexerOutputMode.PAGE_HEAD_MAX:
            max_pages = int(real_page_table.shape[1])
            even_pages = max_pages if max_pages % 2 == 0 else max_pages + 1
            return torch.empty(
                (int(q_fp8.shape[1]), rows, even_pages),
                dtype=torch.float32,
                device=q_fp8.device,
            )
        return torch.empty(
            (rows, width_tokens), dtype=torch.float32, device=q_fp8.device
        )
    if active_width is None:
        active_width = torch.tensor(
            [width_tokens], dtype=torch.int32, device=q_fp8.device
        )
    if active_width.shape != (1,):
        raise ValueError(
            f"active_width must have shape (1,), got {tuple(active_width.shape)}"
        )
    if active_width.dtype != torch.int32:
        raise ValueError(
            f"active_width must have dtype torch.int32, got {active_width.dtype}"
        )
    if active_width.device != q_fp8.device:
        raise ValueError(
            f"active_width device {active_width.device} does not match q_fp8 device {q_fp8.device}"
        )

    k_quant_bytes, k_scales = _split_index_k_cache_runtime_views(index_k_cache)
    use_scalar_k_load = _needs_paged_index_k_scalar_load(
        index_k_cache,
        k_quant_bytes,
    )
    device_index = q_fp8.device.index or 0
    k_tma_desc_ptrs = _dummy_paged_index_k_tma_desc_ptrs(device_index)
    use_scalar_k_load_tensor = _cached_int32_scalar(
        int(use_scalar_k_load),
        device_index,
    )
    q_bytes = q_fp8.contiguous().view(torch.uint8)
    weights_kernel = weights.contiguous()
    real_page_table_kernel = real_page_table.contiguous()
    seqlens_per_query_kernel = seqlens_per_query.contiguous()
    active_width_kernel = active_width.contiguous()
    schedule_metadata_kernel = None
    max_pages = int(real_page_table.shape[1])
    if output_mode == IndexerOutputMode.PAGE_HEAD_MAX:
        even_pages = max_pages if max_pages % 2 == 0 else max_pages + 1
        if page_scores is None:
            logits = torch.full(
                (int(q_fp8.shape[1]), rows, even_pages),
                float("-inf"),
                dtype=torch.float32,
                device=q_fp8.device,
            )
        else:
            if page_scores.dtype != torch.float32:
                raise ValueError(
                    f"page_scores must have dtype torch.float32, got {page_scores.dtype}"
                )
            if page_scores.device != q_fp8.device:
                raise ValueError("page_scores device must match q_fp8")
            if (
                page_scores.ndim != 3
                or int(page_scores.shape[0]) < int(q_fp8.shape[1])
                or int(page_scores.shape[1]) < rows
                or int(page_scores.shape[2]) < even_pages
            ):
                raise ValueError(
                    "page_scores must have shape at least "
                    f"({int(q_fp8.shape[1])}, {rows}, {even_pages}), got "
                    f"{tuple(page_scores.shape)}"
                )
            logits = page_scores[: int(q_fp8.shape[1]), :rows, :even_pages]
            if preinitialize_invalid_logits:
                logits.fill_(float("-inf"))
        logits_view = logits
    else:
        if page_scores is not None:
            raise ValueError("page_scores is only valid with PAGE_HEAD_MAX output")
        if preinitialize_invalid_logits:
            logits = torch.full(
                (rows, width_tokens),
                float("-inf"),
                dtype=torch.float32,
                device=q_fp8.device,
            )
        else:
            logits = torch.empty(
                (rows, width_tokens), dtype=torch.float32, device=q_fp8.device
            )
        logits_view = logits
    common_args = (
        _to_kernel_tensor(q_bytes, cutlass.Uint8),
        _to_kernel_tensor(weights_kernel, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_quant_bytes, cutlass.Uint8),
        _to_kernel_tensor(k_tma_desc_ptrs, cutlass.Int64, assumed_align=8),
        _to_kernel_tensor(use_scalar_k_load_tensor, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(k_scales, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(real_page_table_kernel, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(seqlens_per_query_kernel, cutlass.Int32, assumed_align=4),
    )
    common_cache_key = (
        q_fp8.shape[1],
        score_mode,
        output_mode,
        _tensor_compile_key(
            "q_bytes",
            q_bytes,
            dynamic_dims=(0,),
        ),
        _tensor_compile_key(
            "weights",
            weights_kernel,
            dynamic_dims=(0,),
        ),
        _tensor_meta_key(k_quant_bytes),
        _tensor_meta_key(k_tma_desc_ptrs),
        _tensor_meta_key(use_scalar_k_load_tensor),
        _tensor_meta_key(k_scales),
        _tensor_compile_key(
            "real_page_table",
            real_page_table_kernel,
            dynamic_dims=(0,),
        ),
        _tensor_compile_key(
            "seqlens_per_query",
            seqlens_per_query_kernel,
            dynamic_dims=(0,),
        ),
        _tensor_meta_key(active_width_kernel),
        (
            _tensor_compile_key(
                "page_scores",
                logits,
                dynamic_dims=(1,),
            )
            if output_mode == IndexerOutputMode.PAGE_HEAD_MAX
            else _tensor_compile_key(
                "logits",
                logits,
                dynamic_dims=(0,),
            )
        ),
    )
    if schedule_metadata is not None and schedule_metadata_kernel is None:
        schedule_metadata_kernel = (
            schedule_metadata
            if schedule_metadata.is_contiguous()
            else schedule_metadata.contiguous()
        )
    if _should_use_schedule_single_row_kernel(q_rows=rows, max_pages=max_pages):
        if schedule_metadata is None:
            raise ValueError(
                "schedule_metadata is required for the scheduled single-row decode path"
            )
        kernel = _build_sparse_nsa_schedule_single_row_kernel(
            _SCHEDULE_SINGLE_ROW_PARALLEL_CTAS,
            q_fp8.shape[1],
            score_mode,
            output_mode,
        )
        args = (
            *common_args,
            _to_kernel_tensor(schedule_metadata_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(active_width_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(logits, cutlass.Float32, assumed_align=4),
            current_cuda_stream(),
        )
        cache_key = (
            "schedule_single_row",
            _SCHEDULE_SINGLE_ROW_PARALLEL_CTAS,
            *common_cache_key,
            _tensor_compile_key(
                "schedule_metadata",
                schedule_metadata_kernel,
                dynamic_dims=(0,),
            ),
        )
    elif _should_use_schedule_multi_row_kernel(q_rows=rows, max_pages=max_pages):
        if schedule_metadata is None:
            raise ValueError(
                "schedule_metadata is required for the scheduled multi-row decode path"
            )
        kernel = _build_sparse_nsa_schedule_multi_row_kernel(
            _SCHEDULE_MULTI_ROW_PARALLEL_CTAS,
            q_fp8.shape[1],
            score_mode,
            output_mode,
        )
        args = (
            *common_args,
            _to_kernel_tensor(schedule_metadata_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(active_width_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(logits, cutlass.Float32, assumed_align=4),
            current_cuda_stream(),
        )
        cache_key = (
            "schedule_multi_row",
            _SCHEDULE_MULTI_ROW_PARALLEL_CTAS,
            *common_cache_key,
            _tensor_compile_key(
                "schedule_metadata",
                schedule_metadata_kernel,
                dynamic_dims=(0,),
            ),
        )
    else:
        if output_mode == IndexerOutputMode.PAGE_HEAD_MAX:
            raise NotImplementedError(
                "PAGE_HEAD_MAX output is only implemented for scheduled paged decode routes"
            )
        persistent_ctas = _resolve_sparse_nsa_persistent_ctas(
            device_index=device_index,
            q_rows=rows,
        )
        kernel = _build_sparse_nsa_paged_kernel(
            persistent_ctas,
            q_fp8.shape[1],
            score_mode,
        )
        args = (
            *common_args,
            _to_kernel_tensor(active_width_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(logits, cutlass.Float32, assumed_align=4),
            current_cuda_stream(),
        )
        cache_key = (
            "persistent",
            persistent_ctas,
            *common_cache_key,
        )
    compile_spec = KernelCompileSpec.from_key(
        "attention.indexer.paged_logits",
        2,
        cache_key,
    )
    sm12x_launch(
        kernel,
        compile_spec=compile_spec,
        compile_args=args,
        runtime_args=args,
    )
    return logits_view


def run_paged_tiled_logits_kernel(
    *,
    q_fp8: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    index_k_cache: torch.Tensor | None = None,
    real_page_table: torch.Tensor | None = None,
    seqlens_per_query: torch.Tensor | None = None,
    active_width: torch.Tensor | None = None,
    tile_logits: torch.Tensor | None = None,
    page_size: int | None = None,
    tile_block_q: int | None = None,
    tile_block_k: int | None = None,
    preinitialize_tile_logits: bool | None = None,
    binding: IndexerPagedTiledLogitsKernelBinding | None = None,
) -> torch.Tensor:
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("q_fp8", q_fp8),
                ("weights", weights),
                ("index_k_cache", index_k_cache),
                ("real_page_table", real_page_table),
                ("seqlens_per_query", seqlens_per_query),
                ("active_width", active_width),
                ("tile_logits", tile_logits),
                ("page_size", page_size),
                ("tile_block_q", tile_block_q),
                ("tile_block_k", tile_block_k),
                ("preinitialize_tile_logits", preinitialize_tile_logits),
            )
            if value is not None
        ]
        if extras:
            _raise_binding_extras("run_paged_tiled_logits_kernel", extras)
        q_fp8 = binding.q_fp8
        weights = binding.weights
        index_k_cache = binding.index_k_cache
        real_page_table = binding.real_page_table
        seqlens_per_query = binding.seqlens_per_query
        active_width = binding.active_width
        tile_logits = binding.tile_logits
        page_size = binding.page_size
        tile_block_q = binding.tile_block_q
        tile_block_k = binding.tile_block_k
        preinitialize_tile_logits = binding.preinitialize_tile_logits

    q_fp8 = _require_bound_arg(
        q_fp8, api_name="run_paged_tiled_logits_kernel", name="q_fp8"
    )
    weights = _require_bound_arg(
        weights, api_name="run_paged_tiled_logits_kernel", name="weights"
    )
    index_k_cache = _require_bound_arg(
        index_k_cache,
        api_name="run_paged_tiled_logits_kernel",
        name="index_k_cache",
    )
    real_page_table = _require_bound_arg(
        real_page_table,
        api_name="run_paged_tiled_logits_kernel",
        name="real_page_table",
    )
    seqlens_per_query = _require_bound_arg(
        seqlens_per_query,
        api_name="run_paged_tiled_logits_kernel",
        name="seqlens_per_query",
    )
    active_width = _require_bound_arg(
        active_width,
        api_name="run_paged_tiled_logits_kernel",
        name="active_width",
    )
    tile_logits = _require_bound_arg(
        tile_logits,
        api_name="run_paged_tiled_logits_kernel",
        name="tile_logits",
    )
    page_size = _PAGE_SIZE if page_size is None else int(page_size)
    tile_block_q = _PAGED_TILED_BLOCK_Q if tile_block_q is None else int(tile_block_q)
    tile_block_k = _PAGED_TILED_BLOCK_K if tile_block_k is None else int(tile_block_k)
    preinitialize_tile_logits = (
        True if preinitialize_tile_logits is None else bool(preinitialize_tile_logits)
    )

    return _run_paged_tiled_logits_kernel_common(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        active_width=active_width,
        tile_logits=tile_logits,
        page_size=page_size,
        tile_block_q=tile_block_q,
        tile_block_k=tile_block_k,
        preinitialize_tile_logits=preinitialize_tile_logits,
        source_page_offset=0,
        output_width_tokens=None,
        supertile=False,
    )


def run_paged_supertile_logits_kernel(
    *,
    q_fp8: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    index_k_cache: torch.Tensor | None = None,
    real_page_table: torch.Tensor | None = None,
    seqlens_per_query: torch.Tensor | None = None,
    active_width: torch.Tensor | None = None,
    tile_logits: torch.Tensor | None = None,
    source_page_offset: int | None = None,
    output_width_tokens: int | None = None,
    page_size: int | None = None,
    tile_block_q: int | None = None,
    tile_block_k: int | None = None,
    preinitialize_tile_logits: bool | None = None,
    binding: IndexerPagedSupertileLogitsKernelBinding | None = None,
) -> torch.Tensor:
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("q_fp8", q_fp8),
                ("weights", weights),
                ("index_k_cache", index_k_cache),
                ("real_page_table", real_page_table),
                ("seqlens_per_query", seqlens_per_query),
                ("active_width", active_width),
                ("tile_logits", tile_logits),
                ("source_page_offset", source_page_offset),
                ("output_width_tokens", output_width_tokens),
                ("page_size", page_size),
                ("tile_block_q", tile_block_q),
                ("tile_block_k", tile_block_k),
                ("preinitialize_tile_logits", preinitialize_tile_logits),
            )
            if value is not None
        ]
        if extras:
            _raise_binding_extras("run_paged_supertile_logits_kernel", extras)
        q_fp8 = binding.q_fp8
        weights = binding.weights
        index_k_cache = binding.index_k_cache
        real_page_table = binding.real_page_table
        seqlens_per_query = binding.seqlens_per_query
        active_width = binding.active_width
        tile_logits = binding.tile_logits
        source_page_offset = binding.source_page_offset
        output_width_tokens = binding.output_width_tokens
        page_size = binding.page_size
        tile_block_q = binding.tile_block_q
        tile_block_k = binding.tile_block_k
        preinitialize_tile_logits = binding.preinitialize_tile_logits

    q_fp8 = _require_bound_arg(
        q_fp8,
        api_name="run_paged_supertile_logits_kernel",
        name="q_fp8",
    )
    weights = _require_bound_arg(
        weights,
        api_name="run_paged_supertile_logits_kernel",
        name="weights",
    )
    index_k_cache = _require_bound_arg(
        index_k_cache,
        api_name="run_paged_supertile_logits_kernel",
        name="index_k_cache",
    )
    real_page_table = _require_bound_arg(
        real_page_table,
        api_name="run_paged_supertile_logits_kernel",
        name="real_page_table",
    )
    seqlens_per_query = _require_bound_arg(
        seqlens_per_query,
        api_name="run_paged_supertile_logits_kernel",
        name="seqlens_per_query",
    )
    active_width = _require_bound_arg(
        active_width,
        api_name="run_paged_supertile_logits_kernel",
        name="active_width",
    )
    tile_logits = _require_bound_arg(
        tile_logits,
        api_name="run_paged_supertile_logits_kernel",
        name="tile_logits",
    )
    source_page_offset = _require_bound_arg(
        source_page_offset,
        api_name="run_paged_supertile_logits_kernel",
        name="source_page_offset",
    )
    output_width_tokens = _require_bound_arg(
        output_width_tokens,
        api_name="run_paged_supertile_logits_kernel",
        name="output_width_tokens",
    )
    page_size = _PAGE_SIZE if page_size is None else int(page_size)
    tile_block_q = _PAGED_TILED_BLOCK_Q if tile_block_q is None else int(tile_block_q)
    tile_block_k = _PAGED_TILED_BLOCK_K if tile_block_k is None else int(tile_block_k)
    preinitialize_tile_logits = (
        True if preinitialize_tile_logits is None else bool(preinitialize_tile_logits)
    )
    return _run_paged_tiled_logits_kernel_common(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        active_width=active_width,
        tile_logits=tile_logits,
        page_size=page_size,
        tile_block_q=tile_block_q,
        tile_block_k=tile_block_k,
        preinitialize_tile_logits=preinitialize_tile_logits,
        source_page_offset=source_page_offset,
        output_width_tokens=output_width_tokens,
        supertile=True,
    )


def _run_paged_tiled_logits_kernel_common(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    active_width: torch.Tensor,
    tile_logits: torch.Tensor,
    page_size: int,
    tile_block_q: int,
    tile_block_k: int,
    preinitialize_tile_logits: bool,
    source_page_offset: int = 0,
    output_width_tokens: int | None = None,
    supertile: bool,
) -> torch.Tensor:
    if page_size != _PAGE_SIZE:
        raise ValueError(
            f"paged tiled logits kernel requires page_size={_PAGE_SIZE}, got {page_size}"
        )
    if int(tile_block_k) != _PAGED_TILED_BLOCK_K:
        raise ValueError(
            f"paged tiled logits kernel requires tile_block_k={_PAGED_TILED_BLOCK_K}, got {tile_block_k}"
        )
    if not supports_paged_logits_kernel(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        page_size=page_size,
    ):
        raise ValueError(
            "sparse NSA paged tiled logits kernel only supports the exact CUDA page_size=64 FP8 contract"
        )
    if active_width.shape != (1,):
        raise ValueError(
            f"active_width must have shape (1,), got {tuple(active_width.shape)}"
        )
    if active_width.dtype != torch.int32:
        raise ValueError(
            f"active_width must have dtype torch.int32, got {active_width.dtype}"
        )
    if active_width.device != q_fp8.device:
        raise ValueError(
            f"active_width device {active_width.device} does not match q_fp8 device {q_fp8.device}"
        )
    if tile_logits.dtype != torch.float32 or tile_logits.device != q_fp8.device:
        raise ValueError(
            "tile_logits must be a CUDA torch.float32 tensor on the q_fp8 device"
        )
    if not tile_logits.is_contiguous():
        raise ValueError("tile_logits must be contiguous")

    rows = int(q_fp8.shape[0])
    source_width_tokens = int(real_page_table.shape[1]) * int(page_size)
    source_page_offset = int(source_page_offset)
    if source_page_offset < 0:
        raise ValueError(
            f"source_page_offset must be non-negative, got {source_page_offset}"
        )
    if source_page_offset > int(real_page_table.shape[1]):
        raise ValueError(
            "source_page_offset exceeds real_page_table width: "
            f"offset={source_page_offset}, width={int(real_page_table.shape[1])}"
        )
    if supertile:
        if output_width_tokens is None:
            raise ValueError("paged supertile logits require output_width_tokens")
        width_tokens = int(output_width_tokens)
    else:
        width_tokens = source_width_tokens
    if rows == 0 or width_tokens == 0:
        return tile_logits
    if width_tokens < 0:
        raise ValueError(
            f"output_width_tokens must be non-negative, got {width_tokens}"
        )
    if width_tokens % int(tile_block_k) != 0:
        raise ValueError(
            f"paged tiled logits width {width_tokens} must be divisible by tile_block_k={tile_block_k}"
        )
    num_q_tiles = (rows + int(tile_block_q) - 1) // int(tile_block_q)
    num_k_tiles = width_tokens // int(tile_block_k)
    required_elements = (
        num_q_tiles * num_k_tiles * int(tile_block_q) * int(tile_block_k)
    )
    if int(tile_logits.numel()) < required_elements:
        raise ValueError(
            f"tile_logits has {int(tile_logits.numel())} elements, expected at least {required_elements}"
        )

    k_quant_bytes, k_scales = _split_index_k_cache_runtime_views(index_k_cache)
    use_scalar_k_load = _needs_paged_index_k_scalar_load(
        index_k_cache,
        k_quant_bytes,
    )
    device_index = q_fp8.device.index or 0
    k_tma_desc_ptrs = _dummy_paged_index_k_tma_desc_ptrs(device_index)
    use_scalar_k_load_tensor = _cached_int32_scalar(
        int(use_scalar_k_load),
        device_index,
    )
    if not q_fp8.is_contiguous():
        raise ValueError("paged tiled logits requires contiguous q_fp8")
    if not weights.is_contiguous():
        raise ValueError("paged tiled logits requires contiguous weights")
    if not real_page_table.is_contiguous() and not _is_row_shared_i32_matrix(
        real_page_table
    ):
        raise ValueError("paged tiled logits requires contiguous real_page_table")
    if not seqlens_per_query.is_contiguous():
        raise ValueError("paged tiled logits requires contiguous seqlens_per_query")
    if not active_width.is_contiguous():
        raise ValueError("paged tiled logits requires contiguous active_width")
    q_bytes = q_fp8.view(torch.uint8)
    weights_kernel = weights
    real_page_table_kernel = real_page_table
    seqlens_per_query_kernel = seqlens_per_query
    active_width_kernel = active_width
    logits = tile_logits
    logits_view = tile_logits[:required_elements]
    if preinitialize_tile_logits:
        logits_view.fill_(float("-inf"))

    common_args = (
        _to_kernel_tensor(q_bytes, cutlass.Uint8),
        _to_kernel_tensor(weights_kernel, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_quant_bytes, cutlass.Uint8),
        _to_kernel_tensor(k_tma_desc_ptrs, cutlass.Int64, assumed_align=8),
        _to_kernel_tensor(use_scalar_k_load_tensor, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(k_scales, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(real_page_table_kernel, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(seqlens_per_query_kernel, cutlass.Int32, assumed_align=4),
    )
    common_cache_key = (
        q_fp8.shape[1],
        _tensor_compile_key(
            "q_bytes",
            q_bytes,
            dynamic_dims=(0,),
        ),
        _tensor_compile_key(
            "weights",
            weights_kernel,
            dynamic_dims=(0,),
        ),
        _tensor_meta_key(k_quant_bytes),
        _tensor_meta_key(k_tma_desc_ptrs),
        _tensor_meta_key(use_scalar_k_load_tensor),
        _tensor_meta_key(k_scales),
        _tensor_compile_key(
            "real_page_table",
            real_page_table_kernel,
            dynamic_dims=(0,),
        ),
        _tensor_compile_key(
            "seqlens_per_query",
            seqlens_per_query_kernel,
            dynamic_dims=(0,),
        ),
        _tensor_meta_key(active_width_kernel),
        _tensor_compile_key(
            "tile_logits",
            logits,
            dynamic_dims=(0,),
        ),
    )
    persistent_ctas = _resolve_sparse_nsa_persistent_ctas(
        device_index=device_index,
        q_rows=rows,
    )
    if supertile and _env_indexer_stream_scorer_enabled():
        stream_ctas = _resolve_stream_scorer_ctas(
            device_index=device_index,
            q_rows=rows,
        )
        kernel = _build_sparse_nsa_paged_stream_supertile_kernel(
            stream_ctas,
            q_fp8.shape[1],
            int(tile_block_q),
            int(tile_block_k),
            int(k_quant_bytes.stride(0)),
            int(k_scales.stride(0)),
        )
        args = (
            *common_args,
            _to_kernel_tensor(active_width_kernel, cutlass.Int32, assumed_align=4),
            Int32(source_page_offset),
            Int32(width_tokens),
            _to_kernel_tensor(logits, cutlass.Float32, assumed_align=4),
            current_cuda_stream(),
        )
        cache_key = (
            "stream_supertile",
            stream_ctas,
            int(tile_block_q),
            int(tile_block_k),
            int(k_quant_bytes.stride(0)),
            int(k_scales.stride(0)),
            *common_cache_key,
        )
    elif supertile:
        kernel = _build_sparse_nsa_paged_supertile_kernel(
            persistent_ctas,
            q_fp8.shape[1],
            int(tile_block_q),
            int(tile_block_k),
        )
        args = (
            *common_args,
            _to_kernel_tensor(active_width_kernel, cutlass.Int32, assumed_align=4),
            Int32(source_page_offset),
            Int32(width_tokens),
            _to_kernel_tensor(logits, cutlass.Float32, assumed_align=4),
            current_cuda_stream(),
        )
        cache_key = (
            "persistent_supertile",
            persistent_ctas,
            int(tile_block_q),
            int(tile_block_k),
            *common_cache_key,
        )
    else:
        kernel = _build_sparse_nsa_paged_tiled_kernel(
            persistent_ctas,
            q_fp8.shape[1],
            int(tile_block_q),
            int(tile_block_k),
        )
        args = (
            *common_args,
            _to_kernel_tensor(active_width_kernel, cutlass.Int32, assumed_align=4),
            _to_kernel_tensor(logits, cutlass.Float32, assumed_align=4),
            current_cuda_stream(),
        )
        cache_key = (
            "persistent_tiled",
            persistent_ctas,
            int(tile_block_q),
            int(tile_block_k),
            *common_cache_key,
        )
    compile_spec = KernelCompileSpec.from_key(
        "attention.indexer.paged_tiled_logits",
        2,
        cache_key,
    )
    sm12x_launch(
        kernel,
        compile_spec=compile_spec,
        compile_args=args,
        runtime_args=args,
    )
    logits_view._sm12x_num_q_tiles = num_q_tiles
    logits_view._sm12x_num_k_tiles = num_k_tiles
    logits_view._sm12x_block_q = int(tile_block_q)
    logits_view._sm12x_block_k = int(tile_block_k)
    return logits_view


# ═════════════════════════════════════════════════════════════════════════════
# Streamed paged scorer (v2): persistent right-sized grid + multi-stage
# cp.async K ring loaded straight into the ldmatrix-permuted layout.
#
# Replaces, for the production tiled route, the historical per-row 4xSM grid
# whose single-stage TMA pipeline (issue -> wait -> compute per 8.4KB page)
# and SMEM->SMEM repack left DRAM at ~15% of SOL while saturating L1TEX. The
# ring keeps _STREAM_KV_STAGES-1 page fetches in flight (DeepGEMM
# sm120_fp8_paged_mqa_logits idiom) and the 32-warp CTA covers a full
# 64-token page in ONE MMA pass (the fused kernel's proven score shape), so
# the steady state is wait_group -> MMA -> reduce with three barriers per
# page instead of ~24.
# ═════════════════════════════════════════════════════════════════════════════
_STREAM_WARPS_PER_CTA = 16
_STREAM_THREADS_PER_CTA = _WARP_THREADS * _STREAM_WARPS_PER_CTA
_STREAM_KV_STAGES = 8
# Four warps cooperate on one 64-token page (16 tokens per warp, DeepGEMM's
# A=KV geometry); a 512-thread CTA therefore consumes four pages per
# iteration, each quad ping-ponging two ring stages independently.
_STREAM_WARPS_PER_PAGE = 4
_STREAM_PAGES_PER_ITER = _STREAM_WARPS_PER_CTA // _STREAM_WARPS_PER_PAGE
_STREAM_TOKENS_PER_WARP = _PAGE_SIZE // _STREAM_WARPS_PER_PAGE
_INDEXER_STREAM_SCORER_ENV = "FLASHINFER_EXP_SM12X_INDEXER_STREAM_SCORER"


def _env_indexer_stream_scorer_enabled() -> bool:
    raw = os.environ.get(_INDEXER_STREAM_SCORER_ENV)
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "on", "yes"}


def _resolve_stream_scorer_ctas(*, device_index: int, q_rows: int) -> int:
    """One wave of 1024-thread CTAs (1 CTA/SM) split evenly across rows."""
    num_sms = int(torch.cuda.get_device_properties(device_index).multi_processor_count)
    rows = max(1, int(q_rows))
    return max(1, (num_sms + rows - 1) // rows)


@lru_cache(maxsize=None)
def _paged_stream_indexer_shared_storage_cls(
    padded_q_heads: int,
    tokens_per_work: int,
    num_q_head_tiles: int,
):
    class SharedStorage:
        pass

    SharedStorage.__annotations__ = {
        "q_bytes": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, int(padded_q_heads) * _INDEX_HEAD_DIM],
            16,
        ],
        "weights": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, int(padded_q_heads)],
            16,
        ],
        # _STREAM_KV_STAGES ldmatrix-permuted K pages (cp.async ring).
        "k_perm_ring": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Uint8, _STREAM_KV_STAGES * _PAGE_SIZE * _INDEX_HEAD_DIM
            ],
            1024,
        ],
        "scales_ring": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _STREAM_KV_STAGES * _PAGE_SIZE],
            16,
        ],
    }
    return cute.struct(SharedStorage)


@cute.jit
def _stage_q_permuted(
    q_bytes: cute.Tensor,  # (rows, heads, 128) uint8
    q_idx: Int32,
    num_heads: Int32,
    q_perm_base_addr: Int32,
    tx: Int32,
    q_row_stride_bytes: Int64,
    *,
    padded_q_heads: cutlass.Constexpr,
    stage_threads: cutlass.Constexpr = _STREAM_THREADS_PER_CTA,
):
    """Stage Q into per-head-tile XOR-permuted 16B granules (ldmatrix layout).

    Head-tile t occupies a 2KB block (16 head rows x 8 vec16), swizzled with
    the same `_permuted_offset_128b` pattern as the K pages, so the score core
    can ldmatrix.x4 the Q operand instead of gathering 64 bytes per fragment.
    Heads past num_heads stage zeros (same padding contract as the linear
    staging).
    """
    linear = tx
    total = Int32(int(padded_q_heads) * (_INDEX_HEAD_DIM // 16))
    while linear < total:
        head_idx = linear // Int32(_INDEX_HEAD_DIM // 16)
        vec_idx = linear - head_idx * Int32(_INDEX_HEAD_DIM // 16)
        v0 = Uint32(0)
        v1 = Uint32(0)
        v2 = Uint32(0)
        v3 = Uint32(0)
        if head_idx < num_heads:
            v0, v1, v2, v3 = ld_global_v4_u32(
                get_ptr_as_int64(
                    q_bytes,
                    Int64(q_idx) * q_row_stride_bytes
                    + Int64(head_idx) * Int64(_INDEX_HEAD_DIM)
                    + Int64(vec_idx) * Int64(16),
                )
            )
        tile_idx = head_idx // Int32(_PAGED_Q_HEAD_TILE)
        row_in_tile = head_idx - tile_idx * Int32(_PAGED_Q_HEAD_TILE)
        tile_base = q_perm_base_addr + tile_idx * Int32(
            _PAGED_Q_HEAD_TILE * _INDEX_HEAD_DIM
        )
        dst_addr = _smem_addr_from_b128_offset(
            tile_base,
            _permuted_offset_128b(row_in_tile, vec_idx, Int32(_INDEX_HEAD_DIM // 16)),
        )
        st_shared_v4_u32(dst_addr, v0, v1, v2, v3)
        linear += Int32(stage_threads)


@cute.jit
def _compute_mxfp8_tile_partials_qldm(
    q_perm_base_addr: Int32,
    s_w: cute.Tensor,
    num_heads: Int32,
    k_perm_base_addr: Int32,
    token_base: Int32,
    head_tile_base: Int32,
    head_tile_slot: Int32,
    lane: Int32,
    s_partial_logits: cute.Tensor,
    partial_row_base: Int32,
    partial_col: Int32,
    score_mode: cutlass.Constexpr[int] = IndexerScoreMode.NSA_RELU_SUM,
):
    """Score core with BOTH operands ldmatrix-loaded from permuted smem.

    Byte-identical math to `_compute_mxfp8_tile_partials`: the manual
    `_pack_q_mxfp8_reg` byte gather (64 single-byte smem loads + shifts per
    warp per page) is replaced by one ldmatrix.x4 + the standard 16b->8b
    fragment swizzle per 32-dim step, which reproduces exactly the
    {2t, 2t+1, 2t+8, 2t+9} per-lane byte pattern the MMA expects.
    """
    group_id = lane // Int32(4)
    thread_id_in_group = lane % Int32(4)
    q0_acc = Float32(0.0)
    q1_acc = Float32(0.0)
    q2_acc = Float32(0.0)
    q3_acc = Float32(0.0)
    q_tile_base = q_perm_base_addr + head_tile_slot * Int32(
        _PAGED_Q_HEAD_TILE * _INDEX_HEAD_DIM
    )
    q_row = lane & Int32(15)
    q_half = lane >> Int32(4)
    k_offset = _permuted_offset_128b(
        token_base + Int32(8) * (lane // Int32(16)) + lane % Int32(8),
        (lane % Int32(16)) // Int32(8),
        Int32(_INDEX_HEAD_DIM // 16),
    )
    for mma_pair in cutlass.range_constexpr(_INDEX_HEAD_DIM // 32):
        q_vec = Int32(2 * mma_pair) + q_half
        q_addr = _smem_addr_from_b128_offset(
            q_tile_base,
            _permuted_offset_128b(q_row, q_vec, Int32(_INDEX_HEAD_DIM // 16)),
        )
        # RAW b16 ldmatrix order on BOTH operands: the fp8 MMA's K reduction is
        # permutation-invariant as long as A and B agree, so the historical
        # 16b->8b fragment swizzles (2 shuffles + 2 byte-perms each, x6 per
        # pair) are unnecessary. DeepGEMM's sm120 fp8_mma uses the same raw
        # convention.
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(q_addr)
        # One x4 yields both K halves for this warp's 8 tokens as matrices
        # 0 and 1 (lanes 0-7 / 8-15 address token rows 0-7 at halves 0/1);
        # the historical left+right pair issued TWO x4 loads and kept one
        # matrix from each.
        b0_k0, b0_k1, _bk2, _bk3 = ldmatrix_m8n8x4_b16(
            _smem_addr_from_b128_offset(k_perm_base_addr, k_offset)
        )
        k_offset_cur = _advance_offset_by_row_128b(
            k_offset,
            Int32(16),
            Int32(_INDEX_HEAD_DIM // 16),
        )
        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
            q0_acc,
            q1_acc,
            q2_acc,
            q3_acc,
            a0,
            a1,
            a2,
            a3,
            b0_k0,
            b0_k1,
            Uint32(0x7F7F7F7F),
            Uint32(0x7F7F7F7F),
        )
        q0_acc = d0
        q1_acc = d1
        q2_acc = d2
        q3_acc = d3
        k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_pair) - Int32(
            16 * (_INDEX_HEAD_DIM // 16)
        )

    head0 = head_tile_base + group_id
    head1 = head0 + Int32(8)
    w0 = Float32(0.0)
    w1 = Float32(0.0)
    if head0 < num_heads:
        w0 = Float32(s_w[head0])
    if head1 < num_heads:
        w1 = Float32(s_w[head1])
    col0 = thread_id_in_group * Int32(2)
    col1 = col0 + Int32(1)
    if cutlass.const_expr(score_mode == IndexerScoreMode.MSA_BILINEAR):
        partial0 = Float32(q0_acc * w0)
        partial0 = Float32(partial0 + q2_acc * w1)
        partial1 = Float32(q1_acc * w0)
        partial1 = Float32(partial1 + q3_acc * w1)
    else:
        partial0 = Float32(attention_ops.fmax(q0_acc, Float32(0.0)) * w0)
        partial0 = Float32(partial0 + attention_ops.fmax(q2_acc, Float32(0.0)) * w1)
        partial1 = Float32(attention_ops.fmax(q1_acc, Float32(0.0)) * w0)
        partial1 = Float32(partial1 + attention_ops.fmax(q3_acc, Float32(0.0)) * w1)
    partial0 = _reduce_column_pair_sum(partial0)
    partial1 = _reduce_column_pair_sum(partial1)
    if group_id == Int32(0):
        s_partial_logits[partial_row_base + col0, partial_col] = partial0
        s_partial_logits[partial_row_base + col1, partial_col] = partial1


@cute.jit
def _stream_issue_k_page_cp_async(
    k_quant_bytes: cute.Tensor,  # (pages, 64, 128) uint8 view of the packed cache
    k_scales: cute.Tensor,  # (pages, 64) float32 view of the packed cache
    page_id: Int32,
    k_ring_base_addr: Int32,
    scales_ring_base_addr: Int32,
    stage: Int32,
    tx: Int32,
    *,
    k_quant_page_stride: cutlass.Constexpr,  # dim-0 BYTE stride (8448 packed)
    k_scales_row_stride: cutlass.Constexpr,  # dim-0 ELEMENT stride (2112 packed)
    issue_threads: cutlass.Constexpr = _STREAM_THREADS_PER_CTA,
):
    """Async global->shared page fetch straight into the permuted layout.

    16B `cp.async.cg` granules use the same XOR-swizzled destination as the
    fused kernel's synchronous `_load_permute_k_page_g2s`, so the MMA-side
    addressing is unchanged; the copy is simply asynchronous and ring-staged.
    The caller commits the group.
    """
    stage_k_base = k_ring_base_addr + stage * Int32(_PAGE_SIZE * _INDEX_HEAD_DIM)
    page_elem_base = Int64(page_id) * Int64(k_quant_page_stride)
    linear = tx
    total = Int32(_PAGE_SIZE * (_INDEX_HEAD_DIM // 16))
    while linear < total:
        row = linear // Int32(_INDEX_HEAD_DIM // 16)
        vec_idx = linear - row * Int32(_INDEX_HEAD_DIM // 16)
        g_addr = get_ptr_as_int64(
            k_quant_bytes,
            page_elem_base
            + Int64(row) * Int64(_INDEX_HEAD_DIM)
            + Int64(vec_idx) * Int64(16),
        )
        dst_addr = _smem_addr_from_b128_offset(
            stage_k_base,
            _permuted_offset_128b(row, vec_idx, Int32(_INDEX_HEAD_DIM // 16)),
        )
        cp_async4_shared_global(dst_addr, g_addr)
        linear += Int32(issue_threads)
    if tx < Int32(_PAGE_SIZE):
        s_addr = scales_ring_base_addr + (stage * Int32(_PAGE_SIZE) + tx) * Int32(4)
        g_addr = get_ptr_as_int64(
            k_scales,
            Int64(page_id) * Int64(k_scales_row_stride) + Int64(tx),
        )
        cp_async_u32_shared_global(s_addr, g_addr)


class SparseNSAPagedStreamLogitsKernel:
    """Persistent streamed paged scorer (tiled-output supertile contract)."""

    def __init__(
        self,
        persistent_ctas: int,
        num_heads_static: int,
        *,
        tile_block_q: int = _PAGED_TILED_BLOCK_Q,
        tile_block_k: int = _PAGED_TILED_BLOCK_K,
        score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
        k_quant_page_stride: int,
        k_scales_row_stride: int,
    ):
        self.persistent_ctas = int(persistent_ctas)
        self.num_heads_static = int(num_heads_static)
        self.tile_block_q = int(tile_block_q)
        self.tile_block_k = int(tile_block_k)
        self.score_mode = int(score_mode)
        self.k_quant_page_stride = int(k_quant_page_stride)
        self.k_scales_row_stride = int(k_scales_row_stride)
        if self.tile_block_k != _PAGED_TILED_BLOCK_K:
            raise ValueError(
                f"stream paged logits require block_k={_PAGED_TILED_BLOCK_K}, "
                f"got {self.tile_block_k}"
            )
        self.num_q_head_tiles = _num_q_head_tiles(self.num_heads_static)
        if self.num_q_head_tiles not in (1, 2, 4):
            raise ValueError(
                "stream paged logits kernel only supports 1/2/4 head tiles, "
                f"got {self.num_q_head_tiles}"
            )
        self.padded_q_heads = self.num_q_head_tiles * _PAGED_Q_HEAD_TILE
        # DeepGEMM A=KV geometry: a warp owns 16 tokens and loops every 8-head
        # B tile in registers, so a warp quad covers one 64-token page and the
        # CTA consumes _STREAM_PAGES_PER_ITER pages per iteration.
        self.tokens_per_work = _PAGE_SIZE
        self.num_b_head_tiles = (self.num_heads_static + 7) // 8

    def _get_shared_storage_cls(self):
        return _paged_stream_indexer_shared_storage_cls(
            self.padded_q_heads,
            self.tokens_per_work,
            self.num_q_head_tiles,
        )

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,  # unused (ABI parity, kept)
        use_scalar_k_load: cute.Tensor,  # unused (single cp.async path)
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        active_width: cute.Tensor,
        source_page_offset: Int32,
        output_width_tokens: Int32,
        logits_out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_bytes,
            weights,
            k_quant_bytes,
            k_scales,
            real_page_table,
            seqlens_per_query,
            active_width,
            source_page_offset,
            output_width_tokens,
            logits_out,
        ).launch(
            grid=(q_bytes.shape[0], self.persistent_ctas, 1),
            block=[_STREAM_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_scales: cute.Tensor,
        real_page_table: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        active_width: cute.Tensor,
        source_page_offset: Int32,
        output_width_tokens: Int32,
        logits_out: cute.Tensor,
    ):
        tx, _, _ = cute.arch.thread_idx()
        q_idx, cta_idx, _ = cute.arch.block_idx()
        lane = tx % Int32(_WARP_THREADS)
        warp_idx = tx // Int32(_WARP_THREADS)

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._get_shared_storage_cls()

        # ── live width / seqlen clamps (identical to the historical scorer) ──
        source_width_tokens = Int32(real_page_table.shape[1]) * Int32(_PAGE_SIZE)
        source_offset_pages = source_page_offset
        if source_offset_pages < Int32(0):
            source_offset_pages = Int32(0)
        source_offset_tokens = source_offset_pages * Int32(_PAGE_SIZE)
        remaining_width_tokens = source_width_tokens - source_offset_tokens
        if remaining_width_tokens < Int32(0):
            remaining_width_tokens = Int32(0)

        width_tokens = output_width_tokens
        if width_tokens <= Int32(0):
            width_tokens = remaining_width_tokens
        valid_width_tokens = width_tokens
        if valid_width_tokens > remaining_width_tokens:
            valid_width_tokens = remaining_width_tokens

        live_width = Int32(active_width[Int32(0)])
        if live_width > source_width_tokens:
            live_width = source_width_tokens
        live_width = live_width - source_offset_tokens
        if live_width < Int32(0):
            live_width = Int32(0)
        if live_width > valid_width_tokens:
            live_width = valid_width_tokens

        seq_len = Int32(seqlens_per_query[q_idx]) - source_offset_tokens
        if seq_len < Int32(0):
            seq_len = Int32(0)
        if seq_len > live_width:
            seq_len = live_width
        total_work = (seq_len + Int32(_PAGE_SIZE - 1)) // Int32(_PAGE_SIZE)

        storage = smem.allocate(SharedStorage)
        s_w = storage.weights.get_tensor(
            cute.make_layout((self.padded_q_heads,), stride=(1,))
        )
        k_ring_base_addr = shared_ptr_to_u32(storage.k_perm_ring.data_ptr())
        scales_ring_base_addr = shared_ptr_to_u32(storage.scales_ring.data_ptr())
        q_perm_base_addr = shared_ptr_to_u32(storage.q_bytes.data_ptr())
        s_scales_ring = storage.scales_ring.get_tensor(
            cute.make_layout((_STREAM_KV_STAGES * _PAGE_SIZE,), stride=(1,))
        )

        if cta_idx < total_work:
            num_heads = Int32(self.num_heads_static)
            _stage_q_permuted(
                q_bytes,
                q_idx,
                num_heads,
                q_perm_base_addr,
                tx,
                Int64(q_bytes.shape[1]) * Int64(_INDEX_HEAD_DIM),
                padded_q_heads=self.padded_q_heads,
            )
            w_linear = tx
            while w_linear < Int32(self.padded_q_heads):
                s_w[w_linear] = (
                    Float32(weights[q_idx, w_linear])
                    if w_linear < num_heads
                    else Float32(0.0)
                )
                w_linear += Int32(_STREAM_THREADS_PER_CTA)

            # ── v3 geometry: warp quad per page, warp per 16 tokens ──
            quad = warp_idx // Int32(_STREAM_WARPS_PER_PAGE)
            warp_in_quad = warp_idx % Int32(_STREAM_WARPS_PER_PAGE)
            tx_in_quad = tx % Int32(_WARP_THREADS * _STREAM_WARPS_PER_PAGE)
            token_base_w = warp_in_quad * Int32(_STREAM_TOKENS_PER_WARP)
            g = lane // Int32(4)
            t = lane % Int32(4)

            ctas_stride = Int32(self.persistent_ctas)
            # This CTA's page count and iteration count (uniform across CTA).
            n_pages_cta = (
                total_work - Int32(cta_idx) + ctas_stride - Int32(1)
            ) // ctas_stride
            n_iters = (n_pages_cta + Int32(_STREAM_PAGES_PER_ITER - 1)) // Int32(
                _STREAM_PAGES_PER_ITER
            )

            # Q staging must be visible before B-fragment loads.
            cute.arch.sync_threads()

            # ── prologue: each quad prefetches its first page (stage=quad) ──
            j0 = quad
            if j0 < n_pages_cta:
                work0 = Int32(cta_idx) + j0 * ctas_stride
                src_col0 = source_offset_pages + work0
                if (work0 * Int32(_PAGE_SIZE) < seq_len) & (
                    src_col0 < Int32(real_page_table.shape[1])
                ):
                    page_id0 = Int32(real_page_table[q_idx, src_col0])
                    if page_id0 >= Int32(0):
                        _stream_issue_k_page_cp_async(
                            k_quant_bytes,
                            k_scales,
                            page_id0,
                            k_ring_base_addr,
                            scales_ring_base_addr,
                            quad,
                            tx_in_quad,
                            k_quant_page_stride=self.k_quant_page_stride,
                            k_scales_row_stride=self.k_scales_row_stride,
                            issue_threads=_WARP_THREADS * _STREAM_WARPS_PER_PAGE,
                        )
            cute.arch.cp_async_commit_group()

            it = Int32(0)
            while it < n_iters:
                j = it * Int32(_STREAM_PAGES_PER_ITER) + quad
                # ping-pong stage for this quad: quad or quad + PAGES_PER_ITER
                stage = quad + (it % Int32(2)) * Int32(_STREAM_PAGES_PER_ITER)
                next_stage = quad + ((it + Int32(1)) % Int32(2)) * Int32(
                    _STREAM_PAGES_PER_ITER
                )

                # Issue the quad's NEXT page into the other stage, then wait
                # for the current one (1 group lookahead per quad).
                j_next = j + Int32(_STREAM_PAGES_PER_ITER)
                if j_next < n_pages_cta:
                    work_n = Int32(cta_idx) + j_next * ctas_stride
                    src_col_n = source_offset_pages + work_n
                    if (work_n * Int32(_PAGE_SIZE) < seq_len) & (
                        src_col_n < Int32(real_page_table.shape[1])
                    ):
                        page_id_n = Int32(real_page_table[q_idx, src_col_n])
                        if page_id_n >= Int32(0):
                            _stream_issue_k_page_cp_async(
                                k_quant_bytes,
                                k_scales,
                                page_id_n,
                                k_ring_base_addr,
                                scales_ring_base_addr,
                                next_stage,
                                tx_in_quad,
                                k_quant_page_stride=self.k_quant_page_stride,
                                k_scales_row_stride=self.k_scales_row_stride,
                                issue_threads=_WARP_THREADS * _STREAM_WARPS_PER_PAGE,
                            )
                cute.arch.cp_async_commit_group()
                # Current page landed (own-thread groups: <=1 pending).
                cute.arch.cp_async_wait_group(1)
                cute.arch.sync_threads()

                consume = Int32(0)
                work = Int32(cta_idx) + j * ctas_stride
                page_base = work * Int32(_PAGE_SIZE)
                if j < n_pages_cta:
                    src_col = source_offset_pages + work
                    if (page_base < seq_len) & (
                        src_col < Int32(real_page_table.shape[1])
                    ):
                        if Int32(real_page_table[q_idx, src_col]) >= Int32(0):
                            consume = Int32(1)
                if consume != Int32(0):
                    valid_slots = seq_len - page_base
                    if valid_slots > Int32(_PAGE_SIZE):
                        valid_slots = Int32(_PAGE_SIZE)
                    stage_k_base = k_ring_base_addr + stage * Int32(
                        _PAGE_SIZE * _INDEX_HEAD_DIM
                    )

                    # Hoist the warp's A (K-token) fragments: 16 tokens x 128
                    # dims as 4 k-step x4 fragments, reused across all B head
                    # tiles.
                    a_row = token_base_w + (lane & Int32(15))
                    a_half = lane >> Int32(4)
                    ak = cute.make_rmem_tensor(16, Uint32)
                    for kk in cutlass.range_constexpr(4):
                        a_addr = _smem_addr_from_b128_offset(
                            stage_k_base,
                            _permuted_offset_128b(
                                a_row,
                                Int32(2 * kk) + a_half,
                                Int32(_INDEX_HEAD_DIM // 16),
                            ),
                        )
                        aa0, aa1, aa2, aa3 = ldmatrix_m8n8x4_b16(a_addr)
                        ak[kk * 4 + 0] = aa0
                        ak[kk * 4 + 1] = aa1
                        ak[kk * 4 + 2] = aa2
                        ak[kk * 4 + 3] = aa3

                    partial0 = Float32(0.0)
                    partial1 = Float32(0.0)
                    b_row = lane & Int32(7)
                    b_half = (lane >> Int32(3)) & Int32(1)
                    for nt in cutlass.range_constexpr(2 * self.num_q_head_tiles):
                        if cutlass.const_expr(True):
                            d0 = Float32(0.0)
                            d1 = Float32(0.0)
                            d2 = Float32(0.0)
                            d3 = Float32(0.0)
                            tile16 = nt // 2
                            row_in_tile = Int32((nt % 2) * 8) + b_row
                            q_tile_base = q_perm_base_addr + Int32(
                                tile16 * _PAGED_Q_HEAD_TILE * _INDEX_HEAD_DIM
                            )
                            for kk in cutlass.range_constexpr(4):
                                b_addr = _smem_addr_from_b128_offset(
                                    q_tile_base,
                                    _permuted_offset_128b(
                                        row_in_tile,
                                        Int32(2 * kk) + b_half,
                                        Int32(_INDEX_HEAD_DIM // 16),
                                    ),
                                )
                                b0, b1 = ldmatrix_m8n8x2_b16(b_addr)
                                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                                    d0,
                                    d1,
                                    d2,
                                    d3,
                                    ak[kk * 4 + 0],
                                    ak[kk * 4 + 1],
                                    ak[kk * 4 + 2],
                                    ak[kk * 4 + 3],
                                    b0,
                                    b1,
                                    Uint32(0x7F7F7F7F),
                                    Uint32(0x7F7F7F7F),
                                )
                            h0 = Int32(nt * 8) + t * Int32(2)
                            h1 = h0 + Int32(1)
                            w0 = Float32(0.0)
                            w1 = Float32(0.0)
                            if h0 < num_heads:
                                w0 = Float32(s_w[h0])
                            if h1 < num_heads:
                                w1 = Float32(s_w[h1])
                            if cutlass.const_expr(
                                self.score_mode == IndexerScoreMode.MSA_BILINEAR
                            ):
                                partial0 = Float32(partial0 + d0 * w0 + d1 * w1)
                                partial1 = Float32(partial1 + d2 * w0 + d3 * w1)
                            else:
                                partial0 = Float32(
                                    partial0
                                    + attention_ops.fmax(d0, Float32(0.0)) * w0
                                    + attention_ops.fmax(d1, Float32(0.0)) * w1
                                )
                                partial1 = Float32(
                                    partial1
                                    + attention_ops.fmax(d2, Float32(0.0)) * w0
                                    + attention_ops.fmax(d3, Float32(0.0)) * w1
                                )

                    # Quad-lane (t) reduction: sum head-column pairs.
                    for off in cutlass.range_constexpr(2):
                        partial0 = partial0 + cute.arch.shuffle_sync_bfly(
                            partial0, offset=1 << off
                        )
                        partial1 = partial1 + cute.arch.shuffle_sync_bfly(
                            partial1, offset=1 << off
                        )

                    if t == Int32(0):
                        tile_block_q = Int32(self.tile_block_q)
                        tile_block_k = Int32(self.tile_block_k)
                        tile_size = Int32(self.tile_block_q * self.tile_block_k)
                        num_k_tiles = (
                            width_tokens + tile_block_k - Int32(1)
                        ) // tile_block_k
                        q_tile_idx = q_idx // tile_block_q
                        q_local = q_idx - q_tile_idx * tile_block_q
                        k_tile_idx = page_base // tile_block_k
                        row_flat_base = (
                            q_tile_idx * num_k_tiles * tile_size
                            + k_tile_idx * tile_size
                            + q_local * tile_block_k
                            + page_base
                            - k_tile_idx * tile_block_k
                        )
                        slot0 = token_base_w + g
                        if slot0 < valid_slots:
                            logits_out[row_flat_base + slot0] = Float32(
                                partial0
                                * s_scales_ring[stage * Int32(_PAGE_SIZE) + slot0]
                            )
                        slot1 = slot0 + Int32(8)
                        if slot1 < valid_slots:
                            logits_out[row_flat_base + slot1] = Float32(
                                partial1
                                * s_scales_ring[stage * Int32(_PAGE_SIZE) + slot1]
                            )
                # Ring safety: all quad threads finished reading this stage
                # before the NEXT iteration's issue targets it again (the
                # next issue writes next_stage of it+1 == stage of it).
                cute.arch.sync_threads()
                it += Int32(1)


@lru_cache(maxsize=None)
def _build_sparse_nsa_paged_stream_supertile_kernel(
    persistent_ctas: int,
    num_heads_static: int,
    tile_block_q: int,
    tile_block_k: int,
    k_quant_page_stride: int,
    k_scales_row_stride: int,
    score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
) -> SparseNSAPagedStreamLogitsKernel:
    return SparseNSAPagedStreamLogitsKernel(
        persistent_ctas,
        num_heads_static,
        tile_block_q=tile_block_q,
        tile_block_k=tile_block_k,
        score_mode=score_mode,
        k_quant_page_stride=k_quant_page_stride,
        k_scales_row_stride=k_scales_row_stride,
    )
