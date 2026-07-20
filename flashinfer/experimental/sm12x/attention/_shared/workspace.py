# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/workspace.py @ c368a837 (2026-07-14) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Workspace state shared by sparse attention and indexer execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

_MLA_PACKED_DIM = 656
_INDEX_HEAD_DIM = 128
_INDEXER_BLOCK_K = 64
_INDEXER_PREFILL_BLOCK_K = 256
_INDEXER_TILE_BLOCK_Q = 32
_PAGED_INDEXER_TILE_BLOCK_K = 512
_ARENA_ALIGN_BYTES = 1024
_MHC_MULT = 4
_MHC_PARTIALS = 25
_MHC_DEFAULT_SPLIT_K = 64
_WO_MXFP8_SCALE_VEC_SIZE = 32
_WO_MXFP8_SCALE_ROW_TILE = 128
_WO_MXFP8_SCALE_K_TILE = 4
_SPLIT_CHUNK_LADDER = (8, 16, 32, 64, 128, 256, 512, 1024)
_SPLIT_MAX_CHUNKS = 256
_SPLIT_MAX_WIDTH = _SPLIT_CHUNK_LADDER[-1] * _SPLIT_MAX_CHUNKS


class SM12XIndexerTopKPositionBufferUnavailable(RuntimeError):
    """Raised when an arena does not reserve reusable top-k merge positions."""


@dataclass(frozen=True)
class SparseMLASplitDecodeConfig:
    chunk_size: int
    num_chunks: int


@dataclass(frozen=True)
class _PagedIndexerTiledTopKPlan:
    topk: int
    block_q: int
    block_k: int
    q_rows: int
    num_k_tiles: int


@dataclass(frozen=True)
class _PagedIndexerTiledScorerPlan:
    block_q: int
    block_k: int
    q_rows: int
    width_tokens: int
    source_page_width: int


def _canonical_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _shape_only_cuda_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a tiny CUDA tensor whose shape/stride/dtype/device are stable.

    Used as a phantom in host-launcher cache keys so that varying batch sizes
    do not trigger CUTLASS recompilation.  The tensor is never read by kernels.
    """
    base = torch.empty(1, dtype=dtype, device=device)
    return base.as_strided(shape, (0,) * len(shape))


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((int(value) + alignment - 1) // alignment) * alignment


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def default_sparse_mla_split_decode_config_for_width(
    width: int,
    *,
    max_chunks: int = _SPLIT_MAX_CHUNKS,
) -> SparseMLASplitDecodeConfig | None:
    if width <= _SPLIT_CHUNK_LADDER[0] or width > _SPLIT_MAX_WIDTH:
        return None

    max_chunks = max(1, min(int(max_chunks), _SPLIT_MAX_CHUNKS))
    for chunk_size in _SPLIT_CHUNK_LADDER:
        num_chunks = _ceil_div(width, chunk_size)
        if num_chunks <= max_chunks:
            return SparseMLASplitDecodeConfig(
                chunk_size=chunk_size,
                num_chunks=num_chunks,
            )
    return None


def forced_sparse_mla_split_decode_config_for_width(
    width: int,
    *,
    max_chunks: int = _SPLIT_MAX_CHUNKS,
) -> SparseMLASplitDecodeConfig | None:
    if width <= 0 or width > _SPLIT_MAX_WIDTH:
        return None

    max_chunks = max(1, min(int(max_chunks), _SPLIT_MAX_CHUNKS))
    for chunk_size in _SPLIT_CHUNK_LADDER:
        num_chunks = _ceil_div(width, chunk_size)
        if num_chunks <= max_chunks:
            return SparseMLASplitDecodeConfig(
                chunk_size=chunk_size,
                num_chunks=num_chunks,
            )
    return None


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _resolve_contiguous_topk_supertile_k(value: int) -> int:
    value = int(value)
    if value <= 0:
        return 0
    return _align_up(value, _INDEXER_BLOCK_K)


def _resolve_paged_topk_supertile_k(value: int) -> int:
    value = int(value)
    if value <= 0:
        return 0
    return _align_up(value, _PAGED_INDEXER_TILE_BLOCK_K)


def _resolve_paged_indexer_persistent_ctas(
    *,
    device: torch.device,
    q_rows: int,
) -> int:
    if device.type != "cuda":
        return 1
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    persistent_ctas = max(num_sms * 4, 1)
    if int(q_rows) >= 4:
        persistent_ctas = max(persistent_ctas // 2, 1)
    return persistent_ctas


def _shape_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _materialize_arena_view(
    arena: torch.Tensor,
    *,
    offset_bytes: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    offset_bytes = _align_up(
        offset_bytes, max(_ARENA_ALIGN_BYTES, _dtype_nbytes(dtype))
    )
    nbytes = _shape_numel(shape) * _dtype_nbytes(dtype)
    view_bytes = arena.narrow(0, offset_bytes, nbytes)
    typed_view = view_bytes.view(dtype).view(shape)
    return typed_view, offset_bytes + nbytes


def _materialize_arena_strided_view(
    arena: torch.Tensor,
    *,
    offset_bytes: int,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    offset_bytes = _align_up(
        offset_bytes, max(_ARENA_ALIGN_BYTES, _dtype_nbytes(dtype))
    )
    nbytes = _shape_numel(shape) * _dtype_nbytes(dtype)
    view_bytes = arena.narrow(0, offset_bytes, nbytes)
    typed_storage = view_bytes.view(dtype)
    return typed_storage.as_strided(shape, stride), offset_bytes + nbytes


def _split_tmp_output_stride(
    *,
    max_total_q: int,
    num_q_heads: int,
    max_chunks_per_row: int,
    v_head_dim: int,
) -> tuple[int, int, int, int]:
    del max_chunks_per_row
    row_stride = int(num_q_heads) * int(v_head_dim)
    head_stride = int(v_head_dim)
    chunk_stride = int(max_total_q) * int(num_q_heads) * int(v_head_dim)
    return (row_stride, head_stride, chunk_stride, 1)


def _allocate_split_tmp_output(
    *,
    max_total_q: int,
    num_q_heads: int,
    max_chunks_per_row: int,
    v_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    shape = (
        int(max_total_q),
        int(num_q_heads),
        int(max_chunks_per_row),
        int(v_head_dim),
    )
    storage = torch.empty(_shape_numel(shape), dtype=dtype, device=device)
    return storage.as_strided(
        shape,
        _split_tmp_output_stride(
            max_total_q=max_total_q,
            num_q_heads=num_q_heads,
            max_chunks_per_row=max_chunks_per_row,
            v_head_dim=v_head_dim,
        ),
    )


def _split_output_buffer_from_tmp(tmp_output: torch.Tensor) -> torch.Tensor:
    if tmp_output.ndim != 4:
        raise ValueError(f"tmp_output must be rank 4, got {tmp_output.ndim}")
    output = tmp_output[:, :, 0, :]
    if not output.is_contiguous():
        raise RuntimeError(
            "split MLA tmp_output layout must make chunk 0 a contiguous output buffer"
        )
    return output


def _encode_indexer_k_tma_descriptor(
    k_quant_bytes: torch.Tensor,
    *,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a stable TMA descriptor for the fixed indexer K workspace."""
    if k_quant_bytes.ndim != 2 or k_quant_bytes.shape[1] != _INDEX_HEAD_DIM:
        raise ValueError(
            f"k_quant_bytes must have shape (rows, {_INDEX_HEAD_DIM}), got {tuple(k_quant_bytes.shape)}"
        )
    if k_quant_bytes.dtype != torch.uint8:
        raise TypeError(
            f"k_quant_bytes must be dtype torch.uint8, got {k_quant_bytes.dtype}"
        )

    import cuda.bindings.driver as cuda

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
        raise RuntimeError(
            f"cuTensorMapEncodeTiled failed for indexer K workspace: {result}"
        )

    desc = torch.tensor(
        [int(word) for word in tensor_map.opaque],
        dtype=torch.uint64,
        device=k_quant_bytes.device,
    )
    desc_ptrs = torch.tensor(
        [int(desc.data_ptr())], dtype=torch.int64, device=k_quant_bytes.device
    )
    return desc, desc_ptrs


SM12XWorkspaceMode = Literal["decode", "extend", "verify", "draft_extend"]


@dataclass(frozen=True, kw_only=True)
class SM12XAttentionArenaCaps:
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    indexer_num_q_heads: int
    head_dim: int
    max_v_head_dim: int
    topk: int
    max_page_table_width: int
    extend_max_total_q: int
    extend_max_batch: int
    extend_max_kv_rows: int
    paged_max_q_rows: int
    paged_max_batch: int
    indexer_topk: int | None = None
    indexer_max_k_rows: int | None = None
    mla_max_total_q: int | None = None
    mla_max_q_chunks: int | None = None
    page_size: int = 64
    padded_heads: int = 128
    max_chunks_per_row: int = 64
    reserve_extend_indexer_logits: bool = True
    reserve_paged_indexer_logits: bool = True
    reserve_compressed_mla_staging: bool = False
    reserve_mhc: bool = False
    mhc_max_tokens: int = 0
    mhc_hidden_size: int = 0
    mhc_split_k: int = _MHC_DEFAULT_SPLIT_K
    reserve_wo_projection: bool = False
    wo_max_tokens: int = 0
    wo_groups: int = 0
    wo_group_width: int = 0
    wo_rank: int = 0
    wo_hidden_size: int = 0
    extend_indexer_tile_logits_k_rows: int = 0
    paged_indexer_logits_q_rows: int = 0
    paged_indexer_logits_k_rows: int = 0
    paged_indexer_tile_logits_k_rows: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _canonical_device(self.device))
        object.__setattr__(self, "num_q_heads", max(int(self.num_q_heads), 1))
        object.__setattr__(
            self,
            "indexer_num_q_heads",
            max(int(self.indexer_num_q_heads), 1),
        )
        object.__setattr__(self, "head_dim", max(int(self.head_dim), 1))
        object.__setattr__(self, "max_v_head_dim", max(int(self.max_v_head_dim), 1))
        object.__setattr__(self, "topk", max(int(self.topk), 1))
        indexer_topk = self.topk if self.indexer_topk is None else self.indexer_topk
        object.__setattr__(self, "indexer_topk", max(int(indexer_topk), 1))
        object.__setattr__(
            self,
            "max_page_table_width",
            max(int(self.max_page_table_width), 1),
        )
        object.__setattr__(
            self,
            "extend_max_total_q",
            max(int(self.extend_max_total_q), 1),
        )
        object.__setattr__(
            self,
            "extend_max_batch",
            max(int(self.extend_max_batch), 1),
        )
        object.__setattr__(
            self,
            "extend_max_kv_rows",
            max(int(self.extend_max_kv_rows), 0),
        )
        indexer_max_k_rows = (
            int(self.extend_max_kv_rows)
            if self.indexer_max_k_rows is None
            else int(self.indexer_max_k_rows)
        )
        object.__setattr__(
            self,
            "indexer_max_k_rows",
            max(indexer_max_k_rows, 0),
        )
        object.__setattr__(
            self,
            "paged_max_q_rows",
            max(int(self.paged_max_q_rows), 1),
        )
        object.__setattr__(
            self,
            "paged_max_batch",
            max(int(self.paged_max_batch), 1),
        )
        if self.mla_max_total_q is None:
            mla_max_total_q = max(
                int(self.extend_max_total_q),
                int(self.paged_max_q_rows),
                1,
            )
        else:
            mla_max_total_q = max(int(self.mla_max_total_q), 1)
        object.__setattr__(self, "mla_max_total_q", mla_max_total_q)
        if self.mla_max_q_chunks is not None:
            object.__setattr__(
                self,
                "mla_max_q_chunks",
                max(int(self.mla_max_q_chunks), 1),
            )
        object.__setattr__(self, "page_size", max(int(self.page_size), 1))
        object.__setattr__(self, "padded_heads", max(int(self.padded_heads), 1))
        object.__setattr__(
            self,
            "max_chunks_per_row",
            max(int(self.max_chunks_per_row), 1),
        )
        object.__setattr__(
            self,
            "reserve_paged_indexer_logits",
            bool(self.reserve_paged_indexer_logits),
        )
        object.__setattr__(
            self,
            "reserve_compressed_mla_staging",
            bool(self.reserve_compressed_mla_staging),
        )
        object.__setattr__(self, "reserve_mhc", bool(self.reserve_mhc))
        object.__setattr__(self, "mhc_max_tokens", max(int(self.mhc_max_tokens), 0))
        object.__setattr__(self, "mhc_hidden_size", max(int(self.mhc_hidden_size), 0))
        object.__setattr__(self, "mhc_split_k", max(int(self.mhc_split_k), 1))
        if self.reserve_mhc and (
            int(self.mhc_max_tokens) <= 0 or int(self.mhc_hidden_size) <= 0
        ):
            raise ValueError(
                "reserve_mhc requires positive mhc_max_tokens and mhc_hidden_size"
            )
        object.__setattr__(
            self, "reserve_wo_projection", bool(self.reserve_wo_projection)
        )
        object.__setattr__(self, "wo_max_tokens", max(int(self.wo_max_tokens), 0))
        object.__setattr__(self, "wo_groups", max(int(self.wo_groups), 0))
        object.__setattr__(self, "wo_group_width", max(int(self.wo_group_width), 0))
        object.__setattr__(self, "wo_rank", max(int(self.wo_rank), 0))
        object.__setattr__(self, "wo_hidden_size", max(int(self.wo_hidden_size), 0))
        if self.reserve_wo_projection:
            if (
                int(self.wo_max_tokens) <= 0
                or int(self.wo_groups) <= 0
                or int(self.wo_group_width) <= 0
                or int(self.wo_rank) <= 0
                or int(self.wo_hidden_size) <= 0
            ):
                raise ValueError(
                    "reserve_wo_projection requires positive wo_max_tokens, "
                    "wo_groups, wo_group_width, wo_rank, and wo_hidden_size"
                )
            _check_wo_mxfp8_k(int(self.wo_group_width))
            _check_wo_mxfp8_k(int(self.wo_rank) * int(self.wo_groups))
        object.__setattr__(
            self,
            "extend_indexer_tile_logits_k_rows",
            _resolve_contiguous_topk_supertile_k(
                self.extend_indexer_tile_logits_k_rows
            ),
        )
        paged_indexer_logits_q_rows = int(self.paged_indexer_logits_q_rows)
        if paged_indexer_logits_q_rows <= 0:
            paged_indexer_logits_q_rows = int(self.paged_max_q_rows)
        if paged_indexer_logits_q_rows > int(self.paged_max_q_rows):
            raise ValueError(
                "paged_indexer_logits_q_rows "
                f"{paged_indexer_logits_q_rows} exceeds paged_max_q_rows "
                f"{self.paged_max_q_rows}"
            )
        object.__setattr__(
            self,
            "paged_indexer_logits_q_rows",
            max(paged_indexer_logits_q_rows, 1),
        )
        object.__setattr__(
            self,
            "paged_indexer_logits_k_rows",
            _resolve_contiguous_topk_supertile_k(self.paged_indexer_logits_k_rows),
        )
        object.__setattr__(
            self,
            "paged_indexer_tile_logits_k_rows",
            _resolve_paged_topk_supertile_k(self.paged_indexer_tile_logits_k_rows),
        )


@dataclass(frozen=True, kw_only=True)
class SM12XAttentionWorkspaceContract:
    mode: SM12XWorkspaceMode
    max_total_q: int
    max_batch: int
    max_paged_q_rows: int
    max_kv_rows: int
    v_head_dim: int
    indexer_num_q_heads: int
    max_page_table_width: int
    topk: int | None = None
    max_chunks_per_row: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_total_q", max(int(self.max_total_q), 1))
        object.__setattr__(self, "max_batch", max(int(self.max_batch), 1))
        object.__setattr__(
            self,
            "max_paged_q_rows",
            max(int(self.max_paged_q_rows), 1),
        )
        object.__setattr__(self, "max_kv_rows", max(int(self.max_kv_rows), 0))
        object.__setattr__(self, "v_head_dim", max(int(self.v_head_dim), 1))
        object.__setattr__(
            self,
            "indexer_num_q_heads",
            max(int(self.indexer_num_q_heads), 1),
        )
        object.__setattr__(
            self,
            "max_page_table_width",
            max(int(self.max_page_table_width), 1),
        )
        if self.topk is not None:
            object.__setattr__(self, "topk", max(int(self.topk), 1))
        if self.max_chunks_per_row is not None:
            object.__setattr__(
                self,
                "max_chunks_per_row",
                max(int(self.max_chunks_per_row), 1),
            )


@dataclass(frozen=True, kw_only=True)
class _SM12XWOProjectionArenaLayout:
    nbytes: int = 0
    x_q_values_offset_bytes: int = 0
    x_q_scale_rows_offset_bytes: int = 0
    x_q_scale_mma_offset_bytes: int = 0
    tmp_offset_bytes: int = 0
    tmp_q_values_offset_bytes: int = 0
    tmp_q_scale_rows_offset_bytes: int = 0
    tmp_q_scale_mma_offset_bytes: int = 0
    output_offset_bytes: int = 0


def _check_wo_mxfp8_k(k: int) -> None:
    if int(k) <= 0 or int(k) % 128 != 0:
        raise ValueError(
            f"WO MXFP8 dense-GEMM K must be a positive multiple of 128, got {k}"
        )


def _wo_mxfp8_scale_physical_shape(
    *,
    m: int,
    k: int,
    num_groups: int,
) -> tuple[int, int, int, int, int, int]:
    sf_k = int(k) // _WO_MXFP8_SCALE_VEC_SIZE
    return (
        int(num_groups),
        _ceil_div(int(m), _WO_MXFP8_SCALE_ROW_TILE),
        _ceil_div(sf_k, _WO_MXFP8_SCALE_K_TILE),
        32,
        4,
        4,
    )


def _layout_wo_projection(
    *,
    offset_bytes: int,
    tokens: int,
    groups: int,
    group_width: int,
    rank: int,
    hidden: int,
) -> _SM12XWOProjectionArenaLayout:
    tokens = max(int(tokens), 1)
    groups = int(groups)
    group_width = int(group_width)
    rank = int(rank)
    hidden = int(hidden)
    if groups <= 0 or group_width <= 0 or rank <= 0 or hidden <= 0:
        raise ValueError(
            "WO projection arena requires positive groups, group_width, rank, and hidden"
        )
    _check_wo_mxfp8_k(group_width)
    _check_wo_mxfp8_k(rank * groups)

    start = int(offset_bytes)
    cursor = _align_up(start, _ARENA_ALIGN_BYTES)

    x_q_values_offset_bytes = cursor
    cursor += tokens * group_width * groups * _dtype_nbytes(torch.float8_e4m3fn)
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    x_q_scale_rows_offset_bytes = cursor
    cursor += (
        groups
        * tokens
        * (group_width // _WO_MXFP8_SCALE_VEC_SIZE)
        * _dtype_nbytes(torch.float8_e8m0fnu)
    )
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    x_q_scale_mma_offset_bytes = cursor
    cursor += _shape_numel(
        _wo_mxfp8_scale_physical_shape(
            m=tokens,
            k=group_width,
            num_groups=groups,
        )
    ) * _dtype_nbytes(torch.uint8)
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    tmp_offset_bytes = cursor
    cursor += tokens * rank * groups * _dtype_nbytes(torch.bfloat16)
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    tmp_q_width = rank * groups
    tmp_q_values_offset_bytes = cursor
    cursor += tokens * tmp_q_width * _dtype_nbytes(torch.float8_e4m3fn)
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    tmp_q_scale_rows_offset_bytes = cursor
    cursor += (
        tokens
        * (tmp_q_width // _WO_MXFP8_SCALE_VEC_SIZE)
        * _dtype_nbytes(torch.float8_e8m0fnu)
    )
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    tmp_q_scale_mma_offset_bytes = cursor
    cursor += _shape_numel(
        _wo_mxfp8_scale_physical_shape(
            m=tokens,
            k=tmp_q_width,
            num_groups=1,
        )
    ) * _dtype_nbytes(torch.uint8)
    cursor = _align_up(cursor, _ARENA_ALIGN_BYTES)

    output_offset_bytes = cursor
    cursor += tokens * hidden * _dtype_nbytes(torch.bfloat16)

    return _SM12XWOProjectionArenaLayout(
        nbytes=max(0, int(cursor) - start),
        x_q_values_offset_bytes=x_q_values_offset_bytes,
        x_q_scale_rows_offset_bytes=x_q_scale_rows_offset_bytes,
        x_q_scale_mma_offset_bytes=x_q_scale_mma_offset_bytes,
        tmp_offset_bytes=tmp_offset_bytes,
        tmp_q_values_offset_bytes=tmp_q_values_offset_bytes,
        tmp_q_scale_rows_offset_bytes=tmp_q_scale_rows_offset_bytes,
        tmp_q_scale_mma_offset_bytes=tmp_q_scale_mma_offset_bytes,
        output_offset_bytes=output_offset_bytes,
    )


@dataclass(frozen=True, kw_only=True)
class _SM12XAttentionArenaLayout:
    arena_nbytes: int
    mla_phase_nbytes: int
    indexer_phase_nbytes: int
    indexer_k_rows: int
    mla_tmp_q_chunks: int
    paged_logits_q_rows: int
    paged_logits_width_tokens: int
    paged_tile_logits_width_tokens: int
    ragged_kv_nbytes: int
    output_buffer_nbytes: int
    final_lse_nbytes: int
    compressed_q_stage_nbytes: int
    compressed_index_stage_nbytes: int
    compressed_page_table_stage_nbytes: int
    compressed_lengths_stage_nbytes: int
    indexer_logits_nbytes: int
    indexer_contiguous_logits_nbytes: int
    indexer_contiguous_tile_logits_nbytes: int
    indexer_contiguous_topk_indices_nbytes: int
    indexer_contiguous_topk_values_nbytes: int
    indexer_contiguous_topk_scratch_indices_nbytes: int
    indexer_contiguous_topk_scratch_values_nbytes: int
    indexer_contiguous_topk_position_nbytes: int
    indexer_contiguous_candidate_values_nbytes: int
    indexer_contiguous_candidate_indices_nbytes: int
    indexer_contiguous_lengths_nbytes: int
    indexer_contiguous_mapped_indices_nbytes: int
    indexer_paged_logits_nbytes: int
    mhc_nbytes: int
    mhc_partials_nbytes: int
    mhc_y_nbytes: int
    mhc_post_nbytes: int
    mhc_comb_nbytes: int
    mhc_out_nbytes: int
    wo_projection_layout: _SM12XWOProjectionArenaLayout
    ragged_kv_offset_bytes: int
    tmp_output_offset_bytes: int
    tmp_lse_offset_bytes: int
    output_buffer_offset_bytes: int
    final_lse_offset_bytes: int
    compressed_q_stage_offset_bytes: int
    compressed_swa_indices_stage_offset_bytes: int
    compressed_swa_lengths_stage_offset_bytes: int
    compressed_indexed_indices_stage_offset_bytes: int
    compressed_indexed_lengths_stage_offset_bytes: int
    compressed_indexed_page_table_stage_offset_bytes: int
    indexer_k_quant_offset_bytes: int
    indexer_k_scale_offset_bytes: int
    indexer_contiguous_logits_offset_bytes: int
    indexer_contiguous_tile_logits_offset_bytes: int
    indexer_contiguous_topk_indices_offset_bytes: int
    indexer_contiguous_topk_values_offset_bytes: int
    indexer_contiguous_topk_scratch_indices_offset_bytes: int
    indexer_contiguous_topk_scratch_values_offset_bytes: int
    indexer_contiguous_topk_position_offset_bytes: int
    indexer_contiguous_candidate_values_offset_bytes: int
    indexer_contiguous_candidate_indices_offset_bytes: int
    indexer_contiguous_lengths_offset_bytes: int
    indexer_contiguous_mapped_indices_offset_bytes: int
    indexer_paged_logits_offset_bytes: int
    mhc_partials_offset_bytes: int
    mhc_y_offset_bytes: int
    mhc_post_offset_bytes: int
    mhc_comb_offset_bytes: int
    mhc_out_offset_bytes: int


@dataclass(kw_only=True)
class SM12XAttentionArena:
    caps: SM12XAttentionArenaCaps
    shared_arena: torch.Tensor
    shared_arena_nbytes: int
    mla_phase_nbytes: int
    indexer_phase_nbytes: int
    indexer_k_rows: int
    mla_tmp_q_chunks: int
    paged_logits_q_rows: int
    paged_logits_width_tokens: int
    paged_tile_logits_width_tokens: int
    ragged_kv_nbytes: int
    output_buffer_nbytes: int
    final_lse_nbytes: int
    compressed_q_stage_nbytes: int
    compressed_index_stage_nbytes: int
    compressed_page_table_stage_nbytes: int
    compressed_lengths_stage_nbytes: int
    indexer_logits_nbytes: int
    indexer_contiguous_logits_nbytes: int
    indexer_contiguous_tile_logits_nbytes: int
    indexer_contiguous_topk_indices_nbytes: int
    indexer_contiguous_topk_values_nbytes: int
    indexer_contiguous_topk_scratch_indices_nbytes: int
    indexer_contiguous_topk_scratch_values_nbytes: int
    indexer_contiguous_topk_position_nbytes: int
    indexer_contiguous_candidate_values_nbytes: int
    indexer_contiguous_candidate_indices_nbytes: int
    indexer_contiguous_lengths_nbytes: int
    indexer_contiguous_mapped_indices_nbytes: int
    indexer_paged_logits_nbytes: int
    mhc_nbytes: int
    mhc_partials_nbytes: int
    mhc_y_nbytes: int
    mhc_post_nbytes: int
    mhc_comb_nbytes: int
    mhc_out_nbytes: int
    wo_projection_layout: _SM12XWOProjectionArenaLayout
    ragged_kv_offset_bytes: int
    tmp_output_offset_bytes: int
    tmp_lse_offset_bytes: int
    output_buffer_offset_bytes: int
    final_lse_offset_bytes: int
    compressed_q_stage_offset_bytes: int
    compressed_swa_indices_stage_offset_bytes: int
    compressed_swa_lengths_stage_offset_bytes: int
    compressed_indexed_indices_stage_offset_bytes: int
    compressed_indexed_lengths_stage_offset_bytes: int
    compressed_indexed_page_table_stage_offset_bytes: int
    indexer_k_quant_offset_bytes: int
    indexer_k_scale_offset_bytes: int
    indexer_contiguous_logits_offset_bytes: int
    indexer_contiguous_tile_logits_offset_bytes: int
    indexer_contiguous_topk_indices_offset_bytes: int
    indexer_contiguous_topk_values_offset_bytes: int
    indexer_contiguous_topk_scratch_indices_offset_bytes: int
    indexer_contiguous_topk_scratch_values_offset_bytes: int
    indexer_contiguous_topk_position_offset_bytes: int
    indexer_contiguous_candidate_values_offset_bytes: int
    indexer_contiguous_candidate_indices_offset_bytes: int
    indexer_contiguous_lengths_offset_bytes: int
    indexer_contiguous_mapped_indices_offset_bytes: int
    indexer_paged_logits_offset_bytes: int
    mhc_partials_offset_bytes: int
    mhc_y_offset_bytes: int
    mhc_post_offset_bytes: int
    mhc_comb_offset_bytes: int
    mhc_out_offset_bytes: int

    @classmethod
    def _layout(cls, caps: SM12XAttentionArenaCaps) -> _SM12XAttentionArenaLayout:
        indexer_q_rows = max(
            int(caps.extend_max_total_q), int(caps.paged_max_q_rows), 1
        )
        mla_max_total_q = max(int(caps.mla_max_total_q or indexer_q_rows), 1)
        max_paged_q_rows = max(int(caps.paged_max_q_rows), 1)
        paged_logits_q_rows = max(int(caps.paged_indexer_logits_q_rows), 1)
        max_kv_rows = max(int(caps.extend_max_kv_rows), 1)
        raw_indexer_k_rows = max(int(caps.indexer_max_k_rows or 0), 0)
        indexer_k_rows = (
            0
            if raw_indexer_k_rows == 0
            else _align_up(raw_indexer_k_rows, _INDEXER_BLOCK_K)
        )
        default_mla_tmp_q_chunks = mla_max_total_q * int(caps.max_chunks_per_row)
        mla_tmp_q_chunks = (
            default_mla_tmp_q_chunks
            if caps.mla_max_q_chunks is None
            else int(caps.mla_max_q_chunks)
        )
        mla_tmp_q_chunks = max(int(mla_tmp_q_chunks), 1)
        indexer_topk = int(caps.indexer_topk)
        paged_width_tokens = max(
            int(caps.max_page_table_width) * int(caps.page_size),
            1,
        )
        paged_logits_width_tokens = paged_width_tokens
        if int(caps.paged_indexer_logits_k_rows) > 0:
            paged_logits_width_tokens = min(
                paged_width_tokens,
                int(caps.paged_indexer_logits_k_rows),
            )
        paged_tile_logits_width_tokens = 0
        if int(caps.paged_indexer_tile_logits_k_rows) > 0:
            paged_tile_logits_width_tokens = min(
                paged_width_tokens,
                int(caps.paged_indexer_tile_logits_k_rows),
            )

        mla_offset = 0
        mla_offset = _align_up(mla_offset, _ARENA_ALIGN_BYTES)
        ragged_kv_offset_bytes = mla_offset
        mla_offset += max_kv_rows * _MLA_PACKED_DIM * _dtype_nbytes(caps.kv_dtype)
        mla_offset = _align_up(mla_offset, _ARENA_ALIGN_BYTES)

        tmp_output_offset_bytes = mla_offset
        mla_offset += (
            mla_tmp_q_chunks
            * int(caps.num_q_heads)
            * int(caps.max_v_head_dim)
            * _dtype_nbytes(caps.dtype)
        )
        mla_offset = _align_up(mla_offset, _ARENA_ALIGN_BYTES)
        tmp_lse_offset_bytes = mla_offset
        mla_offset += (
            mla_tmp_q_chunks * int(caps.num_q_heads) * _dtype_nbytes(torch.float32)
        )
        mla_offset = _align_up(mla_offset, _ARENA_ALIGN_BYTES)
        output_buffer_offset_bytes = tmp_output_offset_bytes
        output_buffer_nbytes = 0
        final_lse_offset_bytes = mla_offset
        final_lse_nbytes = (
            mla_max_total_q * int(caps.num_q_heads) * _dtype_nbytes(torch.float32)
        )
        mla_offset += final_lse_nbytes
        mla_offset = _align_up(mla_offset, _ARENA_ALIGN_BYTES)

        compressed_q_stage_offset_bytes = mla_offset
        compressed_q_stage_nbytes = 0
        compressed_index_stage_nbytes = 0
        compressed_page_table_stage_nbytes = 0
        compressed_lengths_stage_nbytes = 0
        compressed_swa_indices_stage_offset_bytes = (
            compressed_swa_lengths_stage_offset_bytes
        ) = 0
        compressed_indexed_indices_stage_offset_bytes = (
            compressed_indexed_lengths_stage_offset_bytes
        ) = 0
        compressed_indexed_page_table_stage_offset_bytes = 0
        if caps.reserve_compressed_mla_staging:
            compressed_q_stage_nbytes = (
                mla_max_total_q
                * int(caps.num_q_heads)
                * int(caps.head_dim)
                * _dtype_nbytes(caps.dtype)
            )
            mla_offset += compressed_q_stage_nbytes
            mla_offset = _align_up(mla_offset, _ARENA_ALIGN_BYTES)
            compressed_swa_indices_stage_offset_bytes = mla_offset
            compressed_index_stage_nbytes = (
                mla_max_total_q * int(caps.topk) * _dtype_nbytes(torch.int32)
            )
            mla_offset += compressed_index_stage_nbytes
            mla_offset = _align_up(mla_offset, _ARENA_ALIGN_BYTES)
            compressed_swa_lengths_stage_offset_bytes = mla_offset
            compressed_lengths_stage_nbytes = mla_max_total_q * _dtype_nbytes(
                torch.int32
            )
            mla_offset += compressed_lengths_stage_nbytes
            mla_offset = _align_up(mla_offset, _ARENA_ALIGN_BYTES)
            compressed_indexed_indices_stage_offset_bytes = mla_offset
            mla_offset += compressed_index_stage_nbytes
            mla_offset = _align_up(mla_offset, _ARENA_ALIGN_BYTES)
            compressed_indexed_lengths_stage_offset_bytes = mla_offset
            mla_offset += compressed_lengths_stage_nbytes
            mla_offset = _align_up(mla_offset, _ARENA_ALIGN_BYTES)
            compressed_indexed_page_table_stage_offset_bytes = mla_offset
            compressed_page_table_stage_nbytes = (
                mla_max_total_q
                * int(caps.max_page_table_width)
                * _dtype_nbytes(torch.int32)
            )
            mla_offset += compressed_page_table_stage_nbytes
            mla_offset = _align_up(mla_offset, _ARENA_ALIGN_BYTES)
        mla_phase_nbytes = int(mla_offset)

        extend_offset = 0
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_k_quant_offset_bytes = extend_offset
        extend_offset += indexer_k_rows * _INDEX_HEAD_DIM
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_k_scale_offset_bytes = extend_offset
        extend_offset += indexer_k_rows * _dtype_nbytes(torch.float32)
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_contiguous_logits_offset_bytes = extend_offset
        if caps.reserve_extend_indexer_logits:
            contiguous_logits_nbytes = (
                int(caps.extend_max_total_q)
                * indexer_k_rows
                * _dtype_nbytes(torch.float32)
            )
        else:
            contiguous_logits_nbytes = 0
        extend_offset += contiguous_logits_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_contiguous_tile_logits_offset_bytes = extend_offset
        contiguous_tile_logits_k_rows = min(
            indexer_k_rows,
            _resolve_contiguous_topk_supertile_k(
                caps.extend_indexer_tile_logits_k_rows
            ),
        )
        paged_tile_logits_k_rows = int(paged_tile_logits_width_tokens)
        contiguous_tile_logits_q_rows = _align_up(
            int(caps.extend_max_total_q),
            _INDEXER_TILE_BLOCK_Q,
        )
        paged_tile_logits_q_rows = _align_up(
            max_paged_q_rows,
            _INDEXER_TILE_BLOCK_Q,
        )
        contiguous_tile_logits_nbytes = 0
        if contiguous_tile_logits_k_rows:
            contiguous_tile_logits_nbytes = max(
                contiguous_tile_logits_nbytes,
                contiguous_tile_logits_q_rows
                * contiguous_tile_logits_k_rows
                * _dtype_nbytes(torch.float32),
            )
            extend_candidate_chunks = (
                indexer_k_rows + contiguous_tile_logits_k_rows - 1
            ) // contiguous_tile_logits_k_rows
        else:
            extend_candidate_chunks = 0
        if paged_tile_logits_k_rows:
            contiguous_tile_logits_nbytes = max(
                contiguous_tile_logits_nbytes,
                paged_tile_logits_q_rows
                * paged_tile_logits_k_rows
                * _dtype_nbytes(torch.float32),
            )
        extend_offset += contiguous_tile_logits_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_contiguous_topk_indices_offset_bytes = extend_offset
        contiguous_topk_indices_nbytes = (
            indexer_q_rows * indexer_topk * _dtype_nbytes(torch.int32)
        )
        extend_offset += contiguous_topk_indices_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_contiguous_topk_values_offset_bytes = extend_offset
        contiguous_topk_values_nbytes = (
            indexer_q_rows * indexer_topk * _dtype_nbytes(torch.float32)
        )
        extend_offset += contiguous_topk_values_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_contiguous_topk_scratch_indices_offset_bytes = extend_offset
        contiguous_topk_scratch_indices_nbytes = (
            indexer_q_rows * indexer_topk * _dtype_nbytes(torch.int32)
        )
        extend_offset += contiguous_topk_scratch_indices_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_contiguous_topk_scratch_values_offset_bytes = extend_offset
        contiguous_topk_scratch_values_nbytes = (
            indexer_q_rows * indexer_topk * _dtype_nbytes(torch.float32)
        )
        extend_offset += contiguous_topk_scratch_values_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        paged_candidate_chunks = (
            (paged_width_tokens + paged_logits_width_tokens - 1)
            // paged_logits_width_tokens
            if caps.reserve_paged_indexer_logits and paged_logits_width_tokens > 0
            else 0
        )
        if paged_tile_logits_k_rows:
            paged_tile_candidate_chunks = (
                paged_width_tokens + paged_tile_logits_k_rows - 1
            ) // paged_tile_logits_k_rows
            paged_candidate_chunks = max(
                paged_candidate_chunks,
                paged_tile_candidate_chunks,
            )
        candidate_chunks = max(extend_candidate_chunks, paged_candidate_chunks)
        # Streaming-fold carry double-buffer: when more than one supertile chunk is
        # possible the fold needs exactly two (M, topk) carry halves to ping-pong;
        # otherwise no carry buffer is needed. This replaces the old map+merge slab
        # (candidate_chunks halves) and its merge_positions int64 buffer.
        fold_carry_chunks = 2 if int(candidate_chunks) > 1 else 0
        # merge_positions is gone with the merge step; keep the field at zero bytes.
        indexer_contiguous_topk_position_offset_bytes = extend_offset
        contiguous_topk_position_nbytes = 0
        indexer_contiguous_candidate_values_offset_bytes = extend_offset
        extend_candidate_values_nbytes = (
            int(fold_carry_chunks)
            * indexer_q_rows
            * indexer_topk
            * _dtype_nbytes(torch.float32)
        )
        extend_offset += extend_candidate_values_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_contiguous_candidate_indices_offset_bytes = extend_offset
        extend_candidate_indices_nbytes = (
            int(fold_carry_chunks)
            * indexer_q_rows
            * indexer_topk
            * _dtype_nbytes(torch.int32)
        )
        extend_offset += extend_candidate_indices_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_contiguous_lengths_offset_bytes = extend_offset
        contiguous_lengths_nbytes = indexer_q_rows * _dtype_nbytes(torch.int32)
        extend_offset += contiguous_lengths_nbytes
        extend_offset = _align_up(extend_offset, _ARENA_ALIGN_BYTES)
        indexer_contiguous_mapped_indices_offset_bytes = extend_offset
        extend_mapped_indices_nbytes = (
            indexer_q_rows * indexer_topk * _dtype_nbytes(torch.int32)
        )
        extend_offset += extend_mapped_indices_nbytes

        paged_offset = 0
        paged_offset = _align_up(paged_offset, _ARENA_ALIGN_BYTES)
        indexer_paged_logits_offset_bytes = paged_offset
        paged_logits_nbytes = 0
        if caps.reserve_paged_indexer_logits:
            paged_logits_nbytes = (
                paged_logits_q_rows
                * paged_logits_width_tokens
                * _dtype_nbytes(torch.float32)
            )
        paged_offset += paged_logits_nbytes

        indexer_phase_nbytes = int(max(extend_offset, paged_offset))
        attention_phase_nbytes = max(mla_phase_nbytes, indexer_phase_nbytes, 1)
        aux_offset = attention_phase_nbytes
        mhc_partials_offset_bytes = mhc_y_offset_bytes = 0
        mhc_post_offset_bytes = mhc_comb_offset_bytes = mhc_out_offset_bytes = 0
        mhc_partials_nbytes = mhc_y_nbytes = mhc_post_nbytes = 0
        mhc_comb_nbytes = mhc_out_nbytes = 0
        if caps.reserve_mhc:
            aux_offset = _align_up(aux_offset, _ARENA_ALIGN_BYTES)
            mhc_partials_offset_bytes = aux_offset
            mhc_partials_nbytes = (
                int(caps.mhc_max_tokens)
                * int(caps.mhc_split_k)
                * _MHC_PARTIALS
                * _dtype_nbytes(torch.float32)
            )
            aux_offset += mhc_partials_nbytes
            aux_offset = _align_up(aux_offset, _ARENA_ALIGN_BYTES)
            mhc_y_offset_bytes = aux_offset
            mhc_y_nbytes = (
                int(caps.mhc_max_tokens)
                * int(caps.mhc_hidden_size)
                * _dtype_nbytes(caps.dtype)
            )
            aux_offset += mhc_y_nbytes
            aux_offset = _align_up(aux_offset, _ARENA_ALIGN_BYTES)
            mhc_post_offset_bytes = aux_offset
            mhc_post_nbytes = (
                int(caps.mhc_max_tokens) * _MHC_MULT * _dtype_nbytes(torch.float32)
            )
            aux_offset += mhc_post_nbytes
            aux_offset = _align_up(aux_offset, _ARENA_ALIGN_BYTES)
            mhc_comb_offset_bytes = aux_offset
            mhc_comb_nbytes = (
                int(caps.mhc_max_tokens)
                * _MHC_MULT
                * _MHC_MULT
                * _dtype_nbytes(torch.float32)
            )
            aux_offset += mhc_comb_nbytes
            aux_offset = _align_up(aux_offset, _ARENA_ALIGN_BYTES)
            mhc_out_offset_bytes = aux_offset
            mhc_out_nbytes = (
                int(caps.mhc_max_tokens)
                * _MHC_MULT
                * int(caps.mhc_hidden_size)
                * _dtype_nbytes(caps.dtype)
            )
            aux_offset += mhc_out_nbytes
        mhc_nbytes = max(0, int(aux_offset) - int(attention_phase_nbytes))

        wo_projection_layout = _SM12XWOProjectionArenaLayout()
        if caps.reserve_wo_projection:
            wo_projection_layout = _layout_wo_projection(
                offset_bytes=aux_offset,
                tokens=int(caps.wo_max_tokens),
                groups=int(caps.wo_groups),
                group_width=int(caps.wo_group_width),
                rank=int(caps.wo_rank),
                hidden=int(caps.wo_hidden_size),
            )
            aux_offset += wo_projection_layout.nbytes
        arena_nbytes = max(attention_phase_nbytes, int(aux_offset), 1)
        ragged_kv_nbytes = max_kv_rows * _MLA_PACKED_DIM * _dtype_nbytes(caps.kv_dtype)
        return _SM12XAttentionArenaLayout(
            arena_nbytes=int(arena_nbytes),
            mla_phase_nbytes=mla_phase_nbytes,
            indexer_phase_nbytes=indexer_phase_nbytes,
            indexer_k_rows=int(indexer_k_rows),
            mla_tmp_q_chunks=int(mla_tmp_q_chunks),
            paged_logits_q_rows=int(paged_logits_q_rows),
            paged_logits_width_tokens=int(paged_logits_width_tokens),
            paged_tile_logits_width_tokens=int(paged_tile_logits_width_tokens),
            ragged_kv_nbytes=ragged_kv_nbytes,
            output_buffer_nbytes=output_buffer_nbytes,
            final_lse_nbytes=final_lse_nbytes,
            compressed_q_stage_nbytes=compressed_q_stage_nbytes,
            compressed_index_stage_nbytes=compressed_index_stage_nbytes,
            compressed_page_table_stage_nbytes=compressed_page_table_stage_nbytes,
            compressed_lengths_stage_nbytes=compressed_lengths_stage_nbytes,
            indexer_logits_nbytes=max(
                contiguous_logits_nbytes,
                contiguous_tile_logits_nbytes,
                contiguous_topk_indices_nbytes,
                contiguous_topk_values_nbytes,
                contiguous_topk_scratch_indices_nbytes,
                contiguous_topk_scratch_values_nbytes,
                contiguous_topk_position_nbytes,
                extend_candidate_values_nbytes,
                extend_candidate_indices_nbytes,
                contiguous_lengths_nbytes,
                extend_mapped_indices_nbytes,
                paged_logits_nbytes,
            ),
            indexer_contiguous_logits_nbytes=contiguous_logits_nbytes,
            indexer_contiguous_tile_logits_nbytes=contiguous_tile_logits_nbytes,
            indexer_contiguous_topk_indices_nbytes=contiguous_topk_indices_nbytes,
            indexer_contiguous_topk_values_nbytes=contiguous_topk_values_nbytes,
            indexer_contiguous_topk_scratch_indices_nbytes=contiguous_topk_scratch_indices_nbytes,
            indexer_contiguous_topk_scratch_values_nbytes=contiguous_topk_scratch_values_nbytes,
            indexer_contiguous_topk_position_nbytes=contiguous_topk_position_nbytes,
            indexer_contiguous_candidate_values_nbytes=extend_candidate_values_nbytes,
            indexer_contiguous_candidate_indices_nbytes=extend_candidate_indices_nbytes,
            indexer_contiguous_lengths_nbytes=contiguous_lengths_nbytes,
            indexer_contiguous_mapped_indices_nbytes=extend_mapped_indices_nbytes,
            indexer_paged_logits_nbytes=paged_logits_nbytes,
            mhc_nbytes=mhc_nbytes,
            mhc_partials_nbytes=mhc_partials_nbytes,
            mhc_y_nbytes=mhc_y_nbytes,
            mhc_post_nbytes=mhc_post_nbytes,
            mhc_comb_nbytes=mhc_comb_nbytes,
            mhc_out_nbytes=mhc_out_nbytes,
            wo_projection_layout=wo_projection_layout,
            ragged_kv_offset_bytes=ragged_kv_offset_bytes,
            tmp_output_offset_bytes=tmp_output_offset_bytes,
            tmp_lse_offset_bytes=tmp_lse_offset_bytes,
            output_buffer_offset_bytes=output_buffer_offset_bytes,
            final_lse_offset_bytes=final_lse_offset_bytes,
            compressed_q_stage_offset_bytes=compressed_q_stage_offset_bytes,
            compressed_swa_indices_stage_offset_bytes=compressed_swa_indices_stage_offset_bytes,
            compressed_swa_lengths_stage_offset_bytes=compressed_swa_lengths_stage_offset_bytes,
            compressed_indexed_indices_stage_offset_bytes=compressed_indexed_indices_stage_offset_bytes,
            compressed_indexed_lengths_stage_offset_bytes=compressed_indexed_lengths_stage_offset_bytes,
            compressed_indexed_page_table_stage_offset_bytes=compressed_indexed_page_table_stage_offset_bytes,
            indexer_k_quant_offset_bytes=indexer_k_quant_offset_bytes,
            indexer_k_scale_offset_bytes=indexer_k_scale_offset_bytes,
            indexer_contiguous_logits_offset_bytes=indexer_contiguous_logits_offset_bytes,
            indexer_contiguous_tile_logits_offset_bytes=indexer_contiguous_tile_logits_offset_bytes,
            indexer_contiguous_topk_indices_offset_bytes=indexer_contiguous_topk_indices_offset_bytes,
            indexer_contiguous_topk_values_offset_bytes=indexer_contiguous_topk_values_offset_bytes,
            indexer_contiguous_topk_scratch_indices_offset_bytes=indexer_contiguous_topk_scratch_indices_offset_bytes,
            indexer_contiguous_topk_scratch_values_offset_bytes=indexer_contiguous_topk_scratch_values_offset_bytes,
            indexer_contiguous_topk_position_offset_bytes=indexer_contiguous_topk_position_offset_bytes,
            indexer_contiguous_candidate_values_offset_bytes=indexer_contiguous_candidate_values_offset_bytes,
            indexer_contiguous_candidate_indices_offset_bytes=indexer_contiguous_candidate_indices_offset_bytes,
            indexer_contiguous_lengths_offset_bytes=indexer_contiguous_lengths_offset_bytes,
            indexer_contiguous_mapped_indices_offset_bytes=indexer_contiguous_mapped_indices_offset_bytes,
            indexer_paged_logits_offset_bytes=indexer_paged_logits_offset_bytes,
            mhc_partials_offset_bytes=mhc_partials_offset_bytes,
            mhc_y_offset_bytes=mhc_y_offset_bytes,
            mhc_post_offset_bytes=mhc_post_offset_bytes,
            mhc_comb_offset_bytes=mhc_comb_offset_bytes,
            mhc_out_offset_bytes=mhc_out_offset_bytes,
        )

    @classmethod
    def _build(
        cls,
        caps: SM12XAttentionArenaCaps,
        *,
        shared_arena: torch.Tensor | None,
        storage: str,
    ) -> "SM12XAttentionArena":
        layout = cls._layout(caps)
        if shared_arena is None:
            shared_arena = torch.empty(
                (layout.arena_nbytes,),
                dtype=torch.uint8,
                device=caps.device,
            )
        elif shared_arena.dtype != torch.uint8:
            raise TypeError(
                f"shared_arena must have dtype torch.uint8, got {shared_arena.dtype}"
            )
        elif shared_arena.device != caps.device:
            raise ValueError(
                f"shared_arena device {shared_arena.device} does not match caps device {caps.device}"
            )
        elif shared_arena.numel() < layout.arena_nbytes:
            raise ValueError(
                f"shared_arena has {shared_arena.numel()} bytes, but attention arena requires {layout.arena_nbytes}"
            )
        arena = cls(
            caps=caps,
            shared_arena=shared_arena,
            shared_arena_nbytes=layout.arena_nbytes,
            mla_phase_nbytes=layout.mla_phase_nbytes,
            indexer_phase_nbytes=layout.indexer_phase_nbytes,
            indexer_k_rows=layout.indexer_k_rows,
            mla_tmp_q_chunks=layout.mla_tmp_q_chunks,
            paged_logits_q_rows=layout.paged_logits_q_rows,
            paged_logits_width_tokens=layout.paged_logits_width_tokens,
            paged_tile_logits_width_tokens=layout.paged_tile_logits_width_tokens,
            ragged_kv_nbytes=layout.ragged_kv_nbytes,
            output_buffer_nbytes=layout.output_buffer_nbytes,
            final_lse_nbytes=layout.final_lse_nbytes,
            compressed_q_stage_nbytes=layout.compressed_q_stage_nbytes,
            compressed_index_stage_nbytes=layout.compressed_index_stage_nbytes,
            compressed_page_table_stage_nbytes=layout.compressed_page_table_stage_nbytes,
            compressed_lengths_stage_nbytes=layout.compressed_lengths_stage_nbytes,
            indexer_logits_nbytes=layout.indexer_logits_nbytes,
            indexer_contiguous_logits_nbytes=layout.indexer_contiguous_logits_nbytes,
            indexer_contiguous_tile_logits_nbytes=layout.indexer_contiguous_tile_logits_nbytes,
            indexer_contiguous_topk_indices_nbytes=layout.indexer_contiguous_topk_indices_nbytes,
            indexer_contiguous_topk_values_nbytes=layout.indexer_contiguous_topk_values_nbytes,
            indexer_contiguous_topk_scratch_indices_nbytes=layout.indexer_contiguous_topk_scratch_indices_nbytes,
            indexer_contiguous_topk_scratch_values_nbytes=layout.indexer_contiguous_topk_scratch_values_nbytes,
            indexer_contiguous_topk_position_nbytes=layout.indexer_contiguous_topk_position_nbytes,
            indexer_contiguous_candidate_values_nbytes=layout.indexer_contiguous_candidate_values_nbytes,
            indexer_contiguous_candidate_indices_nbytes=layout.indexer_contiguous_candidate_indices_nbytes,
            indexer_contiguous_lengths_nbytes=layout.indexer_contiguous_lengths_nbytes,
            indexer_contiguous_mapped_indices_nbytes=layout.indexer_contiguous_mapped_indices_nbytes,
            indexer_paged_logits_nbytes=layout.indexer_paged_logits_nbytes,
            mhc_nbytes=layout.mhc_nbytes,
            mhc_partials_nbytes=layout.mhc_partials_nbytes,
            mhc_y_nbytes=layout.mhc_y_nbytes,
            mhc_post_nbytes=layout.mhc_post_nbytes,
            mhc_comb_nbytes=layout.mhc_comb_nbytes,
            mhc_out_nbytes=layout.mhc_out_nbytes,
            wo_projection_layout=layout.wo_projection_layout,
            ragged_kv_offset_bytes=layout.ragged_kv_offset_bytes,
            tmp_output_offset_bytes=layout.tmp_output_offset_bytes,
            tmp_lse_offset_bytes=layout.tmp_lse_offset_bytes,
            output_buffer_offset_bytes=layout.output_buffer_offset_bytes,
            final_lse_offset_bytes=layout.final_lse_offset_bytes,
            compressed_q_stage_offset_bytes=layout.compressed_q_stage_offset_bytes,
            compressed_swa_indices_stage_offset_bytes=layout.compressed_swa_indices_stage_offset_bytes,
            compressed_swa_lengths_stage_offset_bytes=layout.compressed_swa_lengths_stage_offset_bytes,
            compressed_indexed_indices_stage_offset_bytes=layout.compressed_indexed_indices_stage_offset_bytes,
            compressed_indexed_lengths_stage_offset_bytes=layout.compressed_indexed_lengths_stage_offset_bytes,
            compressed_indexed_page_table_stage_offset_bytes=layout.compressed_indexed_page_table_stage_offset_bytes,
            indexer_k_quant_offset_bytes=layout.indexer_k_quant_offset_bytes,
            indexer_k_scale_offset_bytes=layout.indexer_k_scale_offset_bytes,
            indexer_contiguous_logits_offset_bytes=layout.indexer_contiguous_logits_offset_bytes,
            indexer_contiguous_tile_logits_offset_bytes=layout.indexer_contiguous_tile_logits_offset_bytes,
            indexer_contiguous_topk_indices_offset_bytes=layout.indexer_contiguous_topk_indices_offset_bytes,
            indexer_contiguous_topk_values_offset_bytes=layout.indexer_contiguous_topk_values_offset_bytes,
            indexer_contiguous_topk_scratch_indices_offset_bytes=layout.indexer_contiguous_topk_scratch_indices_offset_bytes,
            indexer_contiguous_topk_scratch_values_offset_bytes=layout.indexer_contiguous_topk_scratch_values_offset_bytes,
            indexer_contiguous_topk_position_offset_bytes=layout.indexer_contiguous_topk_position_offset_bytes,
            indexer_contiguous_candidate_values_offset_bytes=layout.indexer_contiguous_candidate_values_offset_bytes,
            indexer_contiguous_candidate_indices_offset_bytes=layout.indexer_contiguous_candidate_indices_offset_bytes,
            indexer_contiguous_lengths_offset_bytes=layout.indexer_contiguous_lengths_offset_bytes,
            indexer_contiguous_mapped_indices_offset_bytes=layout.indexer_contiguous_mapped_indices_offset_bytes,
            indexer_paged_logits_offset_bytes=layout.indexer_paged_logits_offset_bytes,
            mhc_partials_offset_bytes=layout.mhc_partials_offset_bytes,
            mhc_y_offset_bytes=layout.mhc_y_offset_bytes,
            mhc_post_offset_bytes=layout.mhc_post_offset_bytes,
            mhc_comb_offset_bytes=layout.mhc_comb_offset_bytes,
            mhc_out_offset_bytes=layout.mhc_out_offset_bytes,
        )
        return arena

    @classmethod
    def allocate(cls, caps: SM12XAttentionArenaCaps) -> "SM12XAttentionArena":
        return cls._build(caps, shared_arena=None, storage="standalone")

    @classmethod
    def from_shared_arena(
        cls,
        caps: SM12XAttentionArenaCaps,
        shared_arena: torch.Tensor,
    ) -> "SM12XAttentionArena":
        """Materialize an attention arena over caller-owned uint8 storage."""
        return cls._build(caps, shared_arena=shared_arena, storage="shared")

    @classmethod
    def required_nbytes(cls, caps: SM12XAttentionArenaCaps) -> int:
        """Return the backing-store byte requirement without retaining storage."""
        return cls._layout(caps).arena_nbytes

    def make_mhc_workspace(self):
        if not self.caps.reserve_mhc or self.mhc_nbytes <= 0:
            raise RuntimeError(
                "attention arena was allocated without mHC workspace capacity"
            )
        from flashinfer.experimental.sm12x.norm.mhc._impl import MHCWorkspace

        max_tokens = int(self.caps.mhc_max_tokens)
        hidden_size = int(self.caps.mhc_hidden_size)
        split_k = int(self.caps.mhc_split_k)
        partials, _ = _materialize_arena_view(
            self.shared_arena,
            offset_bytes=self.mhc_partials_offset_bytes,
            shape=(max_tokens, split_k, _MHC_PARTIALS),
            dtype=torch.float32,
        )
        y, _ = _materialize_arena_view(
            self.shared_arena,
            offset_bytes=self.mhc_y_offset_bytes,
            shape=(max_tokens, hidden_size),
            dtype=self.caps.dtype,
        )
        post, _ = _materialize_arena_view(
            self.shared_arena,
            offset_bytes=self.mhc_post_offset_bytes,
            shape=(max_tokens, _MHC_MULT),
            dtype=torch.float32,
        )
        comb, _ = _materialize_arena_view(
            self.shared_arena,
            offset_bytes=self.mhc_comb_offset_bytes,
            shape=(max_tokens, _MHC_MULT, _MHC_MULT),
            dtype=torch.float32,
        )
        out, _ = _materialize_arena_view(
            self.shared_arena,
            offset_bytes=self.mhc_out_offset_bytes,
            shape=(max_tokens, _MHC_MULT, hidden_size),
            dtype=self.caps.dtype,
        )
        return MHCWorkspace(
            partials=partials,
            y=y,
            post=post,
            comb=comb,
            out=out,
            split_k=split_k,
        )

    def make_wo_projection_workspace(self, tokens: int | None = None):
        if not self.caps.reserve_wo_projection or self.wo_projection_layout.nbytes <= 0:
            raise RuntimeError(
                "attention arena was allocated without WO projection workspace capacity"
            )
        from flashinfer.experimental.sm12x.gemm._shared.wo_mxfp8 import (
            MXFP8Rows,
            WOProjectionWorkspace,
        )

        max_tokens = int(self.caps.wo_max_tokens)
        tokens = max_tokens if tokens is None else int(tokens)
        if tokens <= 0 or tokens > max_tokens:
            raise ValueError(
                f"WO projection tokens={tokens} exceeds arena capacity {max_tokens}"
            )

        groups = int(self.caps.wo_groups)
        group_width = int(self.caps.wo_group_width)
        rank = int(self.caps.wo_rank)
        hidden = int(self.caps.wo_hidden_size)
        layout = _layout_wo_projection(
            offset_bytes=self.wo_projection_layout.x_q_values_offset_bytes,
            tokens=tokens,
            groups=groups,
            group_width=group_width,
            rank=rank,
            hidden=hidden,
        )
        if layout.nbytes > self.wo_projection_layout.nbytes:
            raise RuntimeError(
                "WO projection workspace layout exceeds reserved arena capacity: "
                f"requested={layout.nbytes}, reserved={self.wo_projection_layout.nbytes}"
            )

        def mxfp8_rows(
            *,
            values_offset_bytes: int,
            scale_rows_offset_bytes: int,
            scale_mma_offset_bytes: int,
            m: int,
            k: int,
            num_groups: int,
        ) -> MXFP8Rows:
            if num_groups == 1:
                values, _ = _materialize_arena_view(
                    self.shared_arena,
                    offset_bytes=values_offset_bytes,
                    shape=(m, k),
                    dtype=torch.float8_e4m3fn,
                )
            else:
                values, _ = _materialize_arena_strided_view(
                    self.shared_arena,
                    offset_bytes=values_offset_bytes,
                    shape=(m, k, num_groups),
                    stride=(k, 1, m * k),
                    dtype=torch.float8_e4m3fn,
                )
            scale_rows, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=scale_rows_offset_bytes,
                shape=(num_groups, m, k // _WO_MXFP8_SCALE_VEC_SIZE),
                dtype=torch.float8_e8m0fnu,
            )
            scale_physical_u8, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=scale_mma_offset_bytes,
                shape=_wo_mxfp8_scale_physical_shape(
                    m=m,
                    k=k,
                    num_groups=num_groups,
                ),
                dtype=torch.uint8,
            )
            if m % _WO_MXFP8_SCALE_ROW_TILE:
                scale_physical_u8.fill_(127)
            scale_mma = scale_physical_u8.view(torch.float8_e8m0fnu).permute(
                3,
                4,
                1,
                5,
                2,
                0,
            )
            return MXFP8Rows(
                values=values,
                scale_rows=scale_rows,
                scale_mma=scale_mma,
            )

        x_q = mxfp8_rows(
            values_offset_bytes=layout.x_q_values_offset_bytes,
            scale_rows_offset_bytes=layout.x_q_scale_rows_offset_bytes,
            scale_mma_offset_bytes=layout.x_q_scale_mma_offset_bytes,
            m=tokens,
            k=group_width,
            num_groups=groups,
        )
        tmp, _ = _materialize_arena_strided_view(
            self.shared_arena,
            offset_bytes=layout.tmp_offset_bytes,
            shape=(tokens, rank, groups),
            stride=(rank, 1, tokens * rank),
            dtype=torch.bfloat16,
        )
        tmp_q = mxfp8_rows(
            values_offset_bytes=layout.tmp_q_values_offset_bytes,
            scale_rows_offset_bytes=layout.tmp_q_scale_rows_offset_bytes,
            scale_mma_offset_bytes=layout.tmp_q_scale_mma_offset_bytes,
            m=tokens,
            k=rank * groups,
            num_groups=1,
        )
        output, _ = _materialize_arena_view(
            self.shared_arena,
            offset_bytes=layout.output_offset_bytes,
            shape=(tokens, hidden, 1),
            dtype=torch.bfloat16,
        )
        return WOProjectionWorkspace(
            x_q=x_q,
            tmp=tmp,
            tmp_q=tmp_q,
            output=output,
        )

    def _make_workspace_views(
        self,
        contract: SM12XAttentionWorkspaceContract,
        *,
        use_cuda_graph: bool = False,
    ) -> "SM12XAttentionWorkspace":
        workspace_topk = (
            int(contract.topk) if contract.topk is not None else int(self.caps.topk)
        )
        workspace_indexer_topk = min(workspace_topk, int(self.caps.indexer_topk))
        if contract.v_head_dim > self.caps.max_v_head_dim:
            raise ValueError(
                f"workspace v_head_dim {contract.v_head_dim} exceeds arena max_v_head_dim {self.caps.max_v_head_dim}"
            )
        if (
            contract.max_total_q > self.caps.extend_max_total_q
            and contract.max_total_q > self.caps.paged_max_q_rows
        ):
            raise ValueError(
                f"workspace max_total_q {contract.max_total_q} exceeds arena capacities "
                f"(extend={self.caps.extend_max_total_q}, paged={self.caps.paged_max_q_rows})"
            )
        if contract.max_batch > max(
            self.caps.extend_max_batch, self.caps.paged_max_batch
        ):
            raise ValueError(
                f"workspace max_batch {contract.max_batch} exceeds arena capacities "
                f"(extend={self.caps.extend_max_batch}, paged={self.caps.paged_max_batch})"
            )
        if contract.max_paged_q_rows > self.caps.paged_max_q_rows:
            raise ValueError(
                f"workspace max_paged_q_rows {contract.max_paged_q_rows} exceeds arena paged_max_q_rows {self.caps.paged_max_q_rows}"
            )
        if contract.max_kv_rows > self.caps.extend_max_kv_rows:
            raise ValueError(
                f"workspace max_kv_rows {contract.max_kv_rows} exceeds arena extend_max_kv_rows {self.caps.extend_max_kv_rows}"
            )
        if contract.indexer_num_q_heads > self.caps.indexer_num_q_heads:
            raise ValueError(
                "workspace indexer_num_q_heads "
                f"{contract.indexer_num_q_heads} exceeds arena indexer_num_q_heads "
                f"{self.caps.indexer_num_q_heads}"
            )
        if contract.max_page_table_width > self.caps.max_page_table_width:
            raise ValueError(
                "workspace max_page_table_width "
                f"{contract.max_page_table_width} exceeds arena max_page_table_width "
                f"{self.caps.max_page_table_width}"
            )
        if workspace_topk > int(self.caps.topk):
            raise ValueError(
                f"workspace topk {workspace_topk} exceeds arena topk {self.caps.topk}"
            )
        if (
            contract.max_kv_rows > 0 or workspace_topk > 1
        ) and contract.max_total_q > int(self.caps.mla_max_total_q):
            raise ValueError(
                f"workspace MLA max_total_q {contract.max_total_q} exceeds arena mla_max_total_q {self.caps.mla_max_total_q}"
            )
        workspace_max_chunks_per_row = (
            int(contract.max_chunks_per_row)
            if contract.max_chunks_per_row is not None
            else int(self.caps.max_chunks_per_row)
        )
        if workspace_max_chunks_per_row > int(self.caps.max_chunks_per_row):
            raise ValueError(
                "workspace max_chunks_per_row "
                f"{workspace_max_chunks_per_row} exceeds arena max_chunks_per_row "
                f"{self.caps.max_chunks_per_row}"
            )
        workspace_q_chunks = int(contract.max_total_q) * workspace_max_chunks_per_row
        if workspace_q_chunks > int(self.mla_tmp_q_chunks):
            raise ValueError(
                "workspace MLA split scratch "
                f"{workspace_q_chunks} q-chunks exceeds arena capacity "
                f"{self.mla_tmp_q_chunks}"
            )
        workspace = SM12XAttentionWorkspace(
            arena=self,
            contract=contract,
            mode=contract.mode,
            device=self.caps.device,
            dtype=self.caps.dtype,
            kv_dtype=self.caps.kv_dtype,
            num_q_heads=self.caps.num_q_heads,
            indexer_num_q_heads=contract.indexer_num_q_heads,
            head_dim=self.caps.head_dim,
            v_head_dim=contract.v_head_dim,
            topk=workspace_topk,
            indexer_topk=workspace_indexer_topk,
            max_page_table_width=contract.max_page_table_width,
            max_total_q=contract.max_total_q,
            max_batch=contract.max_batch,
            max_paged_q_rows=contract.max_paged_q_rows,
            max_kv_rows=contract.max_kv_rows,
            page_size=self.caps.page_size,
            padded_heads=self.caps.padded_heads,
            use_cuda_graph=use_cuda_graph,
            fixed_capacity=True,
            max_chunks_per_row=workspace_max_chunks_per_row,
            shared_arena=self.shared_arena,
            shared_arena_nbytes=self.shared_arena_nbytes,
            mla_phase_nbytes=self.mla_phase_nbytes,
            indexer_phase_nbytes=self.indexer_phase_nbytes,
            indexer_k_rows=self.indexer_k_rows,
            paged_logits_q_rows=self.paged_logits_q_rows,
            paged_logits_width_tokens=self.paged_logits_width_tokens,
            paged_tile_logits_width_tokens=self.paged_tile_logits_width_tokens,
            ragged_kv_nbytes=self.ragged_kv_nbytes,
            compressed_q_stage_nbytes=self.compressed_q_stage_nbytes,
            compressed_index_stage_nbytes=self.compressed_index_stage_nbytes,
            compressed_page_table_stage_nbytes=self.compressed_page_table_stage_nbytes,
            compressed_lengths_stage_nbytes=self.compressed_lengths_stage_nbytes,
            indexer_logits_nbytes=self.indexer_logits_nbytes,
            indexer_contiguous_logits_nbytes=self.indexer_contiguous_logits_nbytes,
            indexer_contiguous_tile_logits_nbytes=self.indexer_contiguous_tile_logits_nbytes,
            indexer_contiguous_topk_indices_nbytes=self.indexer_contiguous_topk_indices_nbytes,
            indexer_contiguous_topk_values_nbytes=self.indexer_contiguous_topk_values_nbytes,
            indexer_contiguous_topk_scratch_indices_nbytes=self.indexer_contiguous_topk_scratch_indices_nbytes,
            indexer_contiguous_topk_scratch_values_nbytes=self.indexer_contiguous_topk_scratch_values_nbytes,
            indexer_contiguous_topk_position_nbytes=self.indexer_contiguous_topk_position_nbytes,
            indexer_contiguous_candidate_values_nbytes=self.indexer_contiguous_candidate_values_nbytes,
            indexer_contiguous_candidate_indices_nbytes=self.indexer_contiguous_candidate_indices_nbytes,
            indexer_contiguous_lengths_nbytes=self.indexer_contiguous_lengths_nbytes,
            indexer_contiguous_mapped_indices_nbytes=self.indexer_contiguous_mapped_indices_nbytes,
            indexer_paged_logits_nbytes=self.indexer_paged_logits_nbytes,
        )
        workspace._allocate_fixed_capacity_views()
        workspace._initialize_split_chunk_config_if_needed()
        workspace._allocate_contract_phantoms()
        if use_cuda_graph:
            workspace._allocate_paged_indexer_runtime_metadata()
        # Reserve the fused-indexer decode merge scratch now, at construction (before any
        # lock/capture), so every architecture-specific fused route fits without a live
        # allocation.
        workspace._reserve_fused_indexer_scratch()
        return workspace

    def make_workspace(
        self,
        contract: SM12XAttentionWorkspaceContract,
        *,
        use_cuda_graph: bool = False,
    ) -> "SM12XAttentionWorkspace":
        return self._make_workspace_views(
            contract,
            use_cuda_graph=use_cuda_graph,
        )


@dataclass(kw_only=True)
class SM12XAttentionWorkspace:
    arena: SM12XAttentionArena | None = None
    contract: SM12XAttentionWorkspaceContract | None = None
    mode: SM12XWorkspaceMode
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    indexer_num_q_heads: int = 0
    head_dim: int
    v_head_dim: int
    topk: int
    indexer_topk: int = 0
    max_page_table_width: int = 1
    max_total_q: int
    max_batch: int
    max_paged_q_rows: int = 0
    max_kv_rows: int = 0
    page_size: int = 64
    padded_heads: int = 128
    use_cuda_graph: bool = False
    fixed_capacity: bool = False
    max_chunks_per_row: int = 64
    paged_indexer_real_page_table_runtime: torch.Tensor | None = None
    paged_indexer_seqlens_per_query_runtime: torch.Tensor | None = None
    paged_indexer_active_width_runtime: torch.Tensor | None = None
    paged_indexer_active_width_cap: torch.Tensor | None = None
    paged_indexer_schedule_metadata_runtime: torch.Tensor | None = None
    tmp_output: torch.Tensor | None = None
    tmp_lse: torch.Tensor | None = None
    output_buffer: torch.Tensor | None = None
    final_lse: torch.Tensor | None = None
    ragged_kv_cache: torch.Tensor | None = None
    kv_chunk_size_ptr: torch.Tensor | None = None
    num_chunks_ptr: torch.Tensor | None = None
    sm_scale_tensor: torch.Tensor | None = None
    sm_scale_value: float | None = None
    kv_chunk_size_value: int | None = None
    num_chunks_value: int | None = None
    shared_arena: torch.Tensor | None = None
    shared_arena_nbytes: int = 0
    mla_phase_nbytes: int = 0
    indexer_phase_nbytes: int = 0
    indexer_k_rows: int = 0
    paged_logits_q_rows: int = 0
    paged_logits_width_tokens: int = 0
    paged_tile_logits_width_tokens: int = 0
    ragged_kv_nbytes: int = 0
    output_buffer_nbytes: int = 0
    final_lse_nbytes: int = 0
    compressed_q_stage_nbytes: int = 0
    compressed_index_stage_nbytes: int = 0
    compressed_page_table_stage_nbytes: int = 0
    compressed_lengths_stage_nbytes: int = 0
    indexer_logits_nbytes: int = 0
    indexer_contiguous_logits_nbytes: int = 0
    indexer_contiguous_tile_logits_nbytes: int = 0
    indexer_contiguous_topk_indices_nbytes: int = 0
    indexer_contiguous_topk_values_nbytes: int = 0
    indexer_contiguous_topk_scratch_indices_nbytes: int = 0
    indexer_contiguous_topk_scratch_values_nbytes: int = 0
    indexer_contiguous_topk_position_nbytes: int = 0
    indexer_contiguous_candidate_values_nbytes: int = 0
    indexer_contiguous_candidate_indices_nbytes: int = 0
    indexer_contiguous_lengths_nbytes: int = 0
    indexer_contiguous_mapped_indices_nbytes: int = 0
    indexer_paged_logits_nbytes: int = 0
    indexer_k_quant_bytes: torch.Tensor | None = None
    indexer_k_scales: torch.Tensor | None = None
    indexer_k_tma_desc: torch.Tensor | None = None
    indexer_k_tma_desc_ptrs: torch.Tensor | None = None
    indexer_k_tma_prefill_desc: torch.Tensor | None = None
    indexer_k_tma_prefill_desc_ptrs: torch.Tensor | None = None
    indexer_contiguous_logits: torch.Tensor | None = None
    indexer_contiguous_tile_logits: torch.Tensor | None = None
    indexer_contiguous_topk_indices: torch.Tensor | None = None
    indexer_contiguous_topk_values: torch.Tensor | None = None
    indexer_contiguous_topk_scratch_indices: torch.Tensor | None = None
    indexer_contiguous_topk_scratch_values: torch.Tensor | None = None
    indexer_contiguous_topk_positions: torch.Tensor | None = None
    indexer_contiguous_candidate_values: torch.Tensor | None = None
    indexer_contiguous_candidate_indices: torch.Tensor | None = None
    indexer_contiguous_lengths: torch.Tensor | None = None
    indexer_contiguous_mapped_indices: torch.Tensor | None = None
    indexer_paged_logits: torch.Tensor | None = None
    compressed_mla_q_stage: torch.Tensor | None = None
    compressed_mla_swa_indices_stage: torch.Tensor | None = None
    compressed_mla_swa_lengths_stage: torch.Tensor | None = None
    compressed_mla_indexed_indices_stage: torch.Tensor | None = None
    compressed_mla_indexed_lengths_stage: torch.Tensor | None = None
    compressed_mla_indexed_page_table_stage: torch.Tensor | None = None
    # Phantom tensors for stable host-launcher cache keys (fixed_capacity only).
    _contract_q: torch.Tensor | None = None
    _contract_kv_rows: torch.Tensor | None = None
    _contract_kv_scales: torch.Tensor | None = None
    _contract_page_table: torch.Tensor | None = None
    _contract_indexer_cache_seqlens: torch.Tensor | None = None
    _contract_output: torch.Tensor | None = None
    _contract_tmp_output: torch.Tensor | None = None
    _contract_tmp_lse: torch.Tensor | None = None
    # Fused-indexer cross-CTA merge scratch (pack_v, pack_i, state). Reserved EAGERLY at
    # construction at a FIXED, seq-/batch-independent architecture-policy capacity so a
    # live fused decode never allocates after lock_workspace()/graph capture.
    _sm12x_fused_indexer_scratch: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    ) = None
    _indexer_contiguous_tiled_topk_prewarmed: bool = False
    _paged_indexer_tiled_topk_prewarmed: bool = False
    _paged_indexer_tiled_topk_plan: _PagedIndexerTiledTopKPlan | None = None
    _paged_indexer_tiled_scorer_prewarmed: bool = False
    _paged_indexer_tiled_scorer_plan: _PagedIndexerTiledScorerPlan | None = None

    def __post_init__(self) -> None:
        self.device = _canonical_device(self.device)
        self.num_q_heads = int(self.num_q_heads)
        self.indexer_num_q_heads = int(self.indexer_num_q_heads) or int(
            self.num_q_heads
        )
        self.indexer_topk = int(self.indexer_topk) or int(self.topk)
        self.max_page_table_width = max(int(self.max_page_table_width), 1)
        self.max_paged_q_rows = max(int(self.max_paged_q_rows), 1)
        self.max_chunks_per_row = max(int(self.max_chunks_per_row), 1)

    def runtime_metadata_nbytes(self) -> int:
        if not self.use_cuda_graph:
            return 0
        num_sms = 1
        if self.device.type == "cuda":
            num_sms = torch.cuda.get_device_properties(
                self.device
            ).multi_processor_count
        return (
            int(self.max_paged_q_rows)
            * int(self.max_page_table_width)
            * _dtype_nbytes(torch.int32)
            + int(self.max_paged_q_rows) * _dtype_nbytes(torch.int32)
            + _dtype_nbytes(torch.int32)
            + (int(num_sms) + 1) * 2 * _dtype_nbytes(torch.int32)
        )

    def standalone_scratch_nbytes(self) -> int:
        if self.fixed_capacity:
            return 0
        return (
            int(self.max_total_q)
            * int(self.num_q_heads)
            * int(self.max_chunks_per_row)
            * int(self.v_head_dim)
            * _dtype_nbytes(self.dtype)
            + int(self.max_total_q)
            * int(self.num_q_heads)
            * int(self.max_chunks_per_row)
            * _dtype_nbytes(torch.float32)
            + 2 * _dtype_nbytes(torch.int32)
        )

    @classmethod
    def for_contract(
        cls,
        *,
        mode: Literal["decode", "extend", "verify", "draft_extend"],
        device: torch.device | str,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        num_q_heads: int,
        indexer_num_q_heads: int | None = None,
        head_dim: int,
        v_head_dim: int,
        topk: int,
        max_page_table_width: int | None = None,
        max_total_q: int,
        max_batch: int,
        max_paged_q_rows: int | None = None,
        max_kv_rows: int | None = None,
        indexer_max_k_rows: int | None = None,
        reserve_paged_indexer_logits: bool = True,
        paged_indexer_logits_q_rows: int = 0,
        paged_indexer_logits_k_rows: int = 0,
        paged_indexer_tile_logits_k_rows: int = 0,
        page_size: int = 64,
        use_cuda_graph: bool = False,
        padded_heads: int = 128,
        max_chunks_per_row: int = 64,
    ) -> SM12XAttentionWorkspace:
        device = _canonical_device(device)
        if indexer_num_q_heads is None:
            indexer_num_q_heads = num_q_heads
        if max_page_table_width is None:
            max_page_table_width = topk
        if max_paged_q_rows is None:
            max_paged_q_rows = max_batch
        workspace = cls(
            mode=mode,
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            indexer_num_q_heads=indexer_num_q_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            topk=topk,
            max_page_table_width=max_page_table_width,
            max_total_q=int(max_total_q),
            max_batch=int(max_batch),
            max_paged_q_rows=int(max_paged_q_rows),
            max_kv_rows=max(0, int(max_kv_rows)) if max_kv_rows is not None else 0,
            page_size=page_size,
            padded_heads=padded_heads,
            use_cuda_graph=use_cuda_graph,
            max_chunks_per_row=max_chunks_per_row,
        )
        workspace._allocate_split_buffers()
        if use_cuda_graph:
            workspace._allocate_paged_indexer_runtime_metadata()
        return workspace

    @classmethod
    def for_fixed_capacity(
        cls,
        *,
        mode: Literal["decode", "extend", "verify", "draft_extend"],
        device: torch.device | str,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        num_q_heads: int,
        indexer_num_q_heads: int | None = None,
        head_dim: int,
        v_head_dim: int,
        topk: int,
        max_page_table_width: int | None = None,
        max_total_q: int,
        max_batch: int,
        max_paged_q_rows: int | None = None,
        max_kv_rows: int | None = None,
        indexer_max_k_rows: int | None = None,
        page_size: int = 64,
        use_cuda_graph: bool = False,
        padded_heads: int = 128,
        reserve_paged_indexer_logits: bool = True,
        paged_indexer_logits_q_rows: int = 0,
        paged_indexer_logits_k_rows: int = 0,
        paged_indexer_tile_logits_k_rows: int = 0,
        max_chunks_per_row: int = 64,
        reserve_compressed_mla_staging: bool = False,
    ) -> SM12XAttentionWorkspace:
        device = _canonical_device(device)
        if indexer_num_q_heads is None:
            indexer_num_q_heads = num_q_heads
        topk = int(topk)
        if max_page_table_width is None:
            max_page_table_width = topk
        max_page_table_width = max(int(max_page_table_width), 1)
        if max_paged_q_rows is None:
            max_paged_q_rows = max_batch
        max_paged_q_rows = max(int(max_paged_q_rows), 1)
        caps = SM12XAttentionArenaCaps(
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            indexer_num_q_heads=indexer_num_q_heads,
            head_dim=head_dim,
            max_v_head_dim=v_head_dim,
            topk=topk,
            max_page_table_width=max_page_table_width,
            extend_max_total_q=max_total_q,
            extend_max_batch=max_batch,
            extend_max_kv_rows=max(0, int(max_kv_rows))
            if max_kv_rows is not None
            else 0,
            indexer_max_k_rows=(
                None if indexer_max_k_rows is None else max(0, int(indexer_max_k_rows))
            ),
            paged_max_q_rows=max_paged_q_rows,
            paged_max_batch=max_batch,
            page_size=page_size,
            padded_heads=padded_heads,
            max_chunks_per_row=max_chunks_per_row,
            reserve_paged_indexer_logits=reserve_paged_indexer_logits,
            reserve_compressed_mla_staging=reserve_compressed_mla_staging,
            paged_indexer_logits_q_rows=int(paged_indexer_logits_q_rows),
            paged_indexer_logits_k_rows=int(paged_indexer_logits_k_rows),
            paged_indexer_tile_logits_k_rows=int(paged_indexer_tile_logits_k_rows),
        )
        arena = SM12XAttentionArena.allocate(caps)
        contract = SM12XAttentionWorkspaceContract(
            mode=mode,
            max_total_q=max_total_q,
            max_batch=max_batch,
            max_paged_q_rows=max_paged_q_rows,
            max_kv_rows=max(0, int(max_kv_rows)) if max_kv_rows is not None else 0,
            v_head_dim=v_head_dim,
            indexer_num_q_heads=indexer_num_q_heads,
            max_page_table_width=max_page_table_width,
            topk=topk,
            max_chunks_per_row=max_chunks_per_row,
        )
        return arena.make_workspace(contract, use_cuda_graph=use_cuda_graph)

    def _reserve_fused_indexer_scratch(self) -> None:
        """Eagerly allocate the fused-indexer cross-CTA merge scratch at a FIXED capacity.

        Sized once at construction from the workspace contract -- pack = num_sms * topk
        (seq- AND batch-independent: the merge candidate count is capped by per-CTA top-k
        trimming), while state rows follow the architecture-specific routing policy. This
        is allocated BEFORE lock_workspace()/graph capture. Idempotent; no-op off CUDA."""
        if self._sm12x_fused_indexer_scratch is not None:
            return
        if self.device is None or self.device.type != "cuda":
            return
        topk = int(self.indexer_topk)
        if topk <= 0:
            return
        from flashinfer.experimental.sm12x.attention.nsa_indexer.fused_indexer import (
            fused_indexer_scratch_max_rows,
            fused_indexer_scratch_capacity,
        )

        props = torch.cuda.get_device_properties(self.device)
        num_sms = props.multi_processor_count
        max_rows = fused_indexer_scratch_max_rows(
            topk=topk,
            num_heads=int(self.indexer_num_q_heads),
            compute_capability=(int(props.major), int(props.minor)),
        )
        pack_elems, state_words = fused_indexer_scratch_capacity(
            max_rows, topk, num_sms
        )
        self._sm12x_fused_indexer_scratch = (
            torch.empty((pack_elems,), dtype=torch.float32, device=self.device),
            torch.empty((pack_elems,), dtype=torch.int32, device=self.device),
            torch.zeros((state_words,), dtype=torch.int32, device=self.device),
        )

    def get_fused_indexer_scratch(
        self, *, topk: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the eagerly-reserved fused merge scratch (pack_v, pack_i, state).

        Reserves on first call as a fallback, but the normal construction path reserves it
        up front, so in serving this never allocates after the workspace is locked."""
        self._reserve_fused_indexer_scratch()
        cache = self._sm12x_fused_indexer_scratch
        if cache is None:
            raise RuntimeError(
                "fused indexer scratch unavailable (non-CUDA workspace or topk<=0)"
            )
        from flashinfer.experimental.sm12x.attention.nsa_indexer.fused_indexer import (
            fused_indexer_scratch_max_rows,
            fused_indexer_scratch_capacity,
        )

        props = torch.cuda.get_device_properties(self.device)
        num_sms = props.multi_processor_count
        max_rows = fused_indexer_scratch_max_rows(
            topk=int(topk),
            num_heads=int(self.indexer_num_q_heads),
            compute_capability=(int(props.major), int(props.minor)),
        )
        need_pack, need_state = fused_indexer_scratch_capacity(
            max_rows, int(topk), num_sms
        )
        if cache[0].numel() < need_pack or cache[2].numel() < need_state:
            raise RuntimeError(
                f"fused indexer scratch reserved for topk<={self.topk} but call needs "
                f"topk={topk}: pack have={cache[0].numel()} need={need_pack}"
            )
        return cache

    def _allocate_paged_indexer_runtime_metadata(self) -> None:
        if self.paged_indexer_real_page_table_runtime is None:
            self.paged_indexer_real_page_table_runtime = torch.empty(
                (self.max_paged_q_rows, self.max_page_table_width),
                dtype=torch.int32,
                device=self.device,
            )
        if self.paged_indexer_seqlens_per_query_runtime is None:
            self.paged_indexer_seqlens_per_query_runtime = torch.empty(
                (self.max_paged_q_rows,),
                dtype=torch.int32,
                device=self.device,
            )
        if self.paged_indexer_active_width_runtime is None:
            self.paged_indexer_active_width_runtime = torch.empty(
                (1,),
                dtype=torch.int32,
                device=self.device,
            )
        if self.paged_indexer_active_width_cap is None:
            width_cap = max(
                int(self.paged_logits_width_tokens),
                int(self.max_page_table_width) * int(self.page_size),
                1,
            )
            self.paged_indexer_active_width_cap = torch.full(
                (1,),
                int(width_cap),
                dtype=torch.int32,
                device=self.device,
            )
        if self.paged_indexer_schedule_metadata_runtime is None:
            num_sms = 1
            if self.device.type == "cuda":
                num_sms = torch.cuda.get_device_properties(
                    self.device
                ).multi_processor_count
            self.paged_indexer_schedule_metadata_runtime = torch.empty(
                (int(num_sms) + 1, 2),
                dtype=torch.int32,
                device=self.device,
            )

    def _allocate_fixed_capacity_views(self) -> None:
        if self.arena is None:
            raise RuntimeError(
                "_allocate_fixed_capacity_views requires an arena-backed workspace"
            )
        max_total_q = max(int(self.max_total_q), 1)
        max_paged_q_rows = max(int(self.max_paged_q_rows), 1)
        indexer_q_rows = max(max_total_q, max_paged_q_rows)
        max_kv_rows = max(int(self.max_kv_rows), 1)
        indexer_k_rows = (
            int(self.arena.indexer_k_rows)
            if self.arena is not None
            else (
                0
                if int(self.max_kv_rows) <= 0
                else _align_up(max_kv_rows, _INDEXER_BLOCK_K)
            )
        )
        paged_width_tokens = (
            max(int(self.arena.paged_logits_width_tokens), 1)
            if self.arena is not None and int(self.arena.paged_logits_width_tokens) > 0
            else max(int(self.max_page_table_width) * int(self.page_size), 1)
        )
        self.shared_arena = self.arena.shared_arena
        self.shared_arena_nbytes = self.arena.shared_arena_nbytes
        self.mla_phase_nbytes = self.arena.mla_phase_nbytes
        self.indexer_phase_nbytes = self.arena.indexer_phase_nbytes
        self.ragged_kv_nbytes = self.arena.ragged_kv_nbytes
        self.output_buffer_nbytes = self.arena.output_buffer_nbytes
        self.final_lse_nbytes = self.arena.final_lse_nbytes
        self.compressed_q_stage_nbytes = self.arena.compressed_q_stage_nbytes
        self.compressed_index_stage_nbytes = self.arena.compressed_index_stage_nbytes
        self.compressed_page_table_stage_nbytes = (
            self.arena.compressed_page_table_stage_nbytes
        )
        self.compressed_lengths_stage_nbytes = (
            self.arena.compressed_lengths_stage_nbytes
        )
        self.paged_logits_q_rows = self.arena.paged_logits_q_rows
        self.indexer_contiguous_logits_nbytes = (
            self.arena.indexer_contiguous_logits_nbytes
        )
        self.indexer_contiguous_tile_logits_nbytes = (
            self.arena.indexer_contiguous_tile_logits_nbytes
        )
        self.indexer_contiguous_topk_indices_nbytes = (
            self.arena.indexer_contiguous_topk_indices_nbytes
        )
        self.indexer_contiguous_topk_values_nbytes = (
            self.arena.indexer_contiguous_topk_values_nbytes
        )
        self.indexer_contiguous_topk_scratch_indices_nbytes = (
            self.arena.indexer_contiguous_topk_scratch_indices_nbytes
        )
        self.indexer_contiguous_topk_scratch_values_nbytes = (
            self.arena.indexer_contiguous_topk_scratch_values_nbytes
        )
        self.indexer_contiguous_topk_position_nbytes = (
            self.arena.indexer_contiguous_topk_position_nbytes
        )
        self.indexer_contiguous_candidate_values_nbytes = (
            self.arena.indexer_contiguous_candidate_values_nbytes
        )
        self.indexer_contiguous_candidate_indices_nbytes = (
            self.arena.indexer_contiguous_candidate_indices_nbytes
        )
        self.indexer_contiguous_lengths_nbytes = (
            self.arena.indexer_contiguous_lengths_nbytes
        )
        self.indexer_contiguous_mapped_indices_nbytes = (
            self.arena.indexer_contiguous_mapped_indices_nbytes
        )
        self.indexer_paged_logits_nbytes = self.arena.indexer_paged_logits_nbytes
        self.indexer_logits_nbytes = self.arena.indexer_logits_nbytes

        assert self.shared_arena is not None
        self.ragged_kv_cache, mla_offset = _materialize_arena_view(
            self.shared_arena,
            offset_bytes=self.arena.ragged_kv_offset_bytes,
            shape=(max_kv_rows, 1, _MLA_PACKED_DIM),
            dtype=self.kv_dtype,
        )
        self.tmp_output, mla_offset = _materialize_arena_strided_view(
            self.shared_arena,
            offset_bytes=self.arena.tmp_output_offset_bytes,
            shape=(
                max_total_q,
                int(self.num_q_heads),
                int(self.max_chunks_per_row),
                int(self.v_head_dim),
            ),
            stride=_split_tmp_output_stride(
                max_total_q=max_total_q,
                num_q_heads=int(self.num_q_heads),
                max_chunks_per_row=int(self.max_chunks_per_row),
                v_head_dim=int(self.v_head_dim),
            ),
            dtype=self.dtype,
        )
        self.tmp_lse, _ = _materialize_arena_view(
            self.shared_arena,
            offset_bytes=self.arena.tmp_lse_offset_bytes,
            shape=(max_total_q, int(self.num_q_heads), int(self.max_chunks_per_row)),
            dtype=torch.float32,
        )
        self.output_buffer = _split_output_buffer_from_tmp(self.tmp_output)
        self.final_lse, _ = _materialize_arena_view(
            self.shared_arena,
            offset_bytes=self.arena.final_lse_offset_bytes,
            shape=(max_total_q, int(self.num_q_heads)),
            dtype=torch.float32,
        )
        if self.compressed_q_stage_nbytes:
            self.compressed_mla_q_stage, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.compressed_q_stage_offset_bytes,
                shape=(max_total_q, int(self.num_q_heads), int(self.head_dim)),
                dtype=self.dtype,
            )
            self.compressed_mla_swa_indices_stage, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.compressed_swa_indices_stage_offset_bytes,
                shape=(max_total_q, int(self.topk)),
                dtype=torch.int32,
            )
            self.compressed_mla_swa_lengths_stage, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.compressed_swa_lengths_stage_offset_bytes,
                shape=(max_total_q,),
                dtype=torch.int32,
            )
            self.compressed_mla_indexed_indices_stage, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.compressed_indexed_indices_stage_offset_bytes,
                shape=(max_total_q, int(self.topk)),
                dtype=torch.int32,
            )
            self.compressed_mla_indexed_lengths_stage, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.compressed_indexed_lengths_stage_offset_bytes,
                shape=(max_total_q,),
                dtype=torch.int32,
            )
            self.compressed_mla_indexed_page_table_stage, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.compressed_indexed_page_table_stage_offset_bytes,
                shape=(max_total_q, int(self.max_page_table_width)),
                dtype=torch.int32,
            )

        if indexer_k_rows > 0:
            self.indexer_k_quant_bytes, extend_offset = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_k_quant_offset_bytes,
                shape=(indexer_k_rows, _INDEX_HEAD_DIM),
                dtype=torch.uint8,
            )
            self.indexer_k_scales, extend_offset = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_k_scale_offset_bytes,
                shape=(indexer_k_rows,),
                dtype=torch.float32,
            )
            (
                self.indexer_k_tma_desc,
                self.indexer_k_tma_desc_ptrs,
            ) = _encode_indexer_k_tma_descriptor(
                self.indexer_k_quant_bytes,
                block_k=_INDEXER_BLOCK_K,
            )
            (
                self.indexer_k_tma_prefill_desc,
                self.indexer_k_tma_prefill_desc_ptrs,
            ) = _encode_indexer_k_tma_descriptor(
                self.indexer_k_quant_bytes,
                block_k=_INDEXER_PREFILL_BLOCK_K,
            )
        else:
            self.indexer_k_quant_bytes = None
            self.indexer_k_scales = None
            self.indexer_k_tma_desc = None
            self.indexer_k_tma_desc_ptrs = None
            self.indexer_k_tma_prefill_desc = None
            self.indexer_k_tma_prefill_desc_ptrs = None
        if self.indexer_contiguous_logits_nbytes:
            self.indexer_contiguous_logits, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_contiguous_logits_offset_bytes,
                shape=(max_total_q * indexer_k_rows,),
                dtype=torch.float32,
            )
        else:
            self.indexer_contiguous_logits = None
        if self.indexer_contiguous_tile_logits_nbytes:
            self.indexer_contiguous_tile_logits, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_contiguous_tile_logits_offset_bytes,
                shape=(
                    self.indexer_contiguous_tile_logits_nbytes
                    // _dtype_nbytes(torch.float32),
                ),
                dtype=torch.float32,
            )
        else:
            self.indexer_contiguous_tile_logits = None
        if self.indexer_contiguous_topk_indices_nbytes:
            self.indexer_contiguous_topk_indices, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_contiguous_topk_indices_offset_bytes,
                shape=(indexer_q_rows, int(self.indexer_topk)),
                dtype=torch.int32,
            )
        else:
            self.indexer_contiguous_topk_indices = None
        if self.indexer_contiguous_topk_values_nbytes:
            self.indexer_contiguous_topk_values, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_contiguous_topk_values_offset_bytes,
                shape=(indexer_q_rows, int(self.indexer_topk)),
                dtype=torch.float32,
            )
        else:
            self.indexer_contiguous_topk_values = None
        if self.indexer_contiguous_topk_scratch_indices_nbytes:
            self.indexer_contiguous_topk_scratch_indices, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_contiguous_topk_scratch_indices_offset_bytes,
                shape=(indexer_q_rows, int(self.indexer_topk)),
                dtype=torch.int32,
            )
        else:
            self.indexer_contiguous_topk_scratch_indices = None
        if self.indexer_contiguous_topk_scratch_values_nbytes:
            self.indexer_contiguous_topk_scratch_values, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_contiguous_topk_scratch_values_offset_bytes,
                shape=(indexer_q_rows, int(self.indexer_topk)),
                dtype=torch.float32,
            )
        else:
            self.indexer_contiguous_topk_scratch_values = None
        if self.indexer_contiguous_topk_position_nbytes:
            self.indexer_contiguous_topk_positions, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_contiguous_topk_position_offset_bytes,
                shape=(indexer_q_rows, int(self.indexer_topk)),
                dtype=torch.int64,
            )
        else:
            self.indexer_contiguous_topk_positions = None
        if self.indexer_contiguous_candidate_values_nbytes:
            candidate_chunks = self.indexer_contiguous_candidate_values_nbytes // (
                indexer_q_rows * int(self.indexer_topk) * _dtype_nbytes(torch.float32)
            )
            self.indexer_contiguous_candidate_values, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_contiguous_candidate_values_offset_bytes,
                shape=(candidate_chunks, indexer_q_rows, int(self.indexer_topk)),
                dtype=torch.float32,
            )
        else:
            self.indexer_contiguous_candidate_values = None
        if self.indexer_contiguous_candidate_indices_nbytes:
            candidate_chunks = self.indexer_contiguous_candidate_indices_nbytes // (
                indexer_q_rows * int(self.indexer_topk) * _dtype_nbytes(torch.int32)
            )
            self.indexer_contiguous_candidate_indices, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_contiguous_candidate_indices_offset_bytes,
                shape=(candidate_chunks, indexer_q_rows, int(self.indexer_topk)),
                dtype=torch.int32,
            )
        else:
            self.indexer_contiguous_candidate_indices = None
        if self.indexer_contiguous_lengths_nbytes:
            self.indexer_contiguous_lengths, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_contiguous_lengths_offset_bytes,
                shape=(indexer_q_rows,),
                dtype=torch.int32,
            )
        else:
            self.indexer_contiguous_lengths = None
        if self.indexer_contiguous_mapped_indices_nbytes:
            self.indexer_contiguous_mapped_indices, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_contiguous_mapped_indices_offset_bytes,
                shape=(indexer_q_rows, int(self.indexer_topk)),
                dtype=torch.int32,
            )
        else:
            self.indexer_contiguous_mapped_indices = None
        if self.indexer_paged_logits_nbytes and max_paged_q_rows <= int(
            self.paged_logits_q_rows
        ):
            self.indexer_paged_logits, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.indexer_paged_logits_offset_bytes,
                shape=(max_paged_q_rows * paged_width_tokens,),
                dtype=torch.float32,
            )
        else:
            self.indexer_paged_logits = None

    def _allocate_split_buffers(self) -> None:
        if self.mode not in ("decode", "extend", "verify", "draft_extend"):
            return
        if self.fixed_capacity:
            if self.shared_arena is None:
                self._allocate_fixed_capacity_views()
        elif self.tmp_output is None:
            self.tmp_output = _allocate_split_tmp_output(
                max_total_q=self.max_total_q,
                num_q_heads=self.num_q_heads,
                max_chunks_per_row=self.max_chunks_per_row,
                v_head_dim=self.v_head_dim,
                dtype=self.dtype,
                device=self.device,
            )
        if self.tmp_lse is None:
            self.tmp_lse = torch.empty(
                (self.max_total_q, self.num_q_heads, self.max_chunks_per_row),
                dtype=torch.float32,
                device=self.device,
            )
        if self.output_buffer is None:
            if self.tmp_output is None:
                raise RuntimeError("workspace is missing split MLA output scratch")
            self.output_buffer = _split_output_buffer_from_tmp(self.tmp_output)
        if self.final_lse is None:
            self.final_lse = torch.empty(
                (self.max_total_q, self.num_q_heads),
                dtype=torch.float32,
                device=self.device,
            )
        if self.kv_chunk_size_ptr is None:
            self.kv_chunk_size_ptr = torch.empty(
                (1,), dtype=torch.int32, device=self.device
            )
            self.kv_chunk_size_value = None
        if self.num_chunks_ptr is None:
            self.num_chunks_ptr = torch.empty(
                (1,), dtype=torch.int32, device=self.device
            )
            self.num_chunks_value = None

    def _initialize_split_chunk_config_if_needed(self) -> None:
        self._allocate_split_buffers()
        if not (self.fixed_capacity or self.use_cuda_graph):
            return
        if self.kv_chunk_size_value is not None and self.num_chunks_value is not None:
            return
        split_cfg = default_sparse_mla_split_decode_config_for_width(
            int(self.topk),
            max_chunks=self.max_chunks_per_row,
        )
        if split_cfg is None:
            return
        assert self.kv_chunk_size_ptr is not None
        assert self.num_chunks_ptr is not None
        self.kv_chunk_size_ptr.fill_(int(split_cfg.chunk_size))
        self.num_chunks_ptr.fill_(int(split_cfg.num_chunks))
        self.kv_chunk_size_value = int(split_cfg.chunk_size)
        self.num_chunks_value = int(split_cfg.num_chunks)

    def set_split_chunk_config(self, *, kv_chunk_size: int, num_chunks: int) -> None:
        if num_chunks <= 0 or num_chunks > self.max_chunks_per_row:
            raise ValueError(
                f"num_chunks must be in [1, {self.max_chunks_per_row}], got {num_chunks}"
            )
        if kv_chunk_size <= 0:
            raise ValueError(f"kv_chunk_size must be positive, got {kv_chunk_size}")
        self._allocate_split_buffers()
        assert self.kv_chunk_size_ptr is not None
        assert self.num_chunks_ptr is not None
        if self.kv_chunk_size_value != int(kv_chunk_size):
            self.kv_chunk_size_ptr.fill_(int(kv_chunk_size))
            self.kv_chunk_size_value = int(kv_chunk_size)
        if self.num_chunks_value != int(num_chunks):
            self.num_chunks_ptr.fill_(int(num_chunks))
            self.num_chunks_value = int(num_chunks)

    def set_decode_chunk_config(self, *, kv_chunk_size: int, num_chunks: int) -> None:
        self.set_split_chunk_config(kv_chunk_size=kv_chunk_size, num_chunks=num_chunks)

    def gather_ragged_kv_rows(
        self,
        *,
        kv_cache: torch.Tensor,
        row_ids: torch.Tensor,
    ) -> torch.Tensor:
        if kv_cache.ndim != 3:
            raise ValueError(f"kv_cache must be rank-3, got {tuple(kv_cache.shape)}")
        if row_ids.ndim != 1:
            raise ValueError(f"row_ids must be rank-1, got {tuple(row_ids.shape)}")
        if kv_cache.device != self.device:
            raise ValueError(
                f"kv_cache device {kv_cache.device} does not match workspace device {self.device}"
            )
        if row_ids.device != self.device:
            raise ValueError(
                f"row_ids device {row_ids.device} does not match workspace device {self.device}"
            )
        if kv_cache.dtype != self.kv_dtype:
            raise ValueError(
                f"kv_cache dtype {kv_cache.dtype} does not match workspace kv_dtype {self.kv_dtype}"
            )

        row_count = int(row_ids.shape[0])
        capacity = max(int(self.max_kv_rows), row_count, 1)
        expected_row_shape = tuple(int(dim) for dim in kv_cache.shape[1:])
        buffer = self.ragged_kv_cache
        if (
            buffer is None
            or buffer.device != self.device
            or buffer.dtype != kv_cache.dtype
            or tuple(int(dim) for dim in buffer.shape[1:]) != expected_row_shape
            or buffer.shape[0] < capacity
        ):
            if self.fixed_capacity and buffer is not None:
                raise ValueError(
                    f"row_count {row_count} exceeds fixed-capacity ragged KV workspace {buffer.shape[0]}"
                )
            buffer = torch.empty(
                (capacity, *expected_row_shape),
                dtype=kv_cache.dtype,
                device=self.device,
            )
            self.ragged_kv_cache = buffer
            self.max_kv_rows = capacity
            self._refresh_ragged_kv_contracts()
        elif self._contract_kv_rows is None or self._contract_kv_scales is None:
            self._refresh_ragged_kv_contracts()

        assert buffer is not None
        if row_count != 0:
            kv_bytes = kv_cache.view(torch.uint8)
            gathered_bytes = buffer[:row_count].view(torch.uint8)
            torch.index_select(kv_bytes, 0, row_ids.to(torch.long), out=gathered_bytes)
        # Return the full-capacity scratch buffer so launcher cache keys follow
        # workspace capacity instead of the live ragged row count for this prefill.
        return buffer

    def bind_compressed_mla(
        self,
        *,
        q: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lengths: torch.Tensor,
        indexed_indices: torch.Tensor | None = None,
        indexed_lengths: torch.Tensor | None = None,
        indexed_page_table: torch.Tensor | None = None,
    ):
        from flashinfer.experimental.sm12x.attention.compressed_mla._scratch import (
            build_compressed_mla_binding,
        )

        return build_compressed_mla_binding(
            workspace=self,
            q=q,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
            indexed_page_table=indexed_page_table,
        )

    def bind_sparse_mla(
        self,
        *,
        q: torch.Tensor,
        selected_indices: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        nsa_cache_seqlens_int32: torch.Tensor,
    ):
        from flashinfer.experimental.sm12x.attention.sparse_mla._scratch import (
            build_sparse_mla_binding,
        )

        return build_sparse_mla_binding(
            scratch=self,
            q=q,
            selected_indices=selected_indices,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        )

    def contract_kv_tensors_for(
        self,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return stable KV phantoms only for the ragged scratch allocation.

        Extend/verify share a workspace in SGLang. After a ragged prefill allocates
        `ragged_kv_cache`, later paged launches must not reuse those KV phantoms or
        they can collide with a launcher compiled for a different KV layout.
        """
        buffer = self.ragged_kv_cache
        if buffer is None:
            return None, None
        if kv_cache.device != buffer.device or kv_cache.dtype != buffer.dtype:
            return None, None
        if kv_cache.ndim != buffer.ndim:
            return None, None
        if kv_cache.data_ptr() != buffer.data_ptr():
            return None, None
        if tuple(int(dim) for dim in kv_cache.shape[1:]) != tuple(
            int(dim) for dim in buffer.shape[1:]
        ):
            return None, None
        return self._contract_kv_rows, self._contract_kv_scales

    def _refresh_ragged_kv_contracts(self) -> None:
        if self.ragged_kv_cache is None:
            self._contract_kv_rows = None
            self._contract_kv_scales = None
            return

        from flashinfer.experimental.sm12x.attention._shared.mla.packed import (
            _extract_packed_kv_runtime_views,
        )

        kv_rows_u32, kv_scales = _extract_packed_kv_runtime_views(self.ragged_kv_cache)
        self._contract_kv_rows = _shape_only_cuda_tensor(
            tuple(int(dim) for dim in kv_rows_u32.shape),
            dtype=kv_rows_u32.dtype,
            device=self.device,
        )
        self._contract_kv_scales = _shape_only_cuda_tensor(
            tuple(int(dim) for dim in kv_scales.shape),
            dtype=kv_scales.dtype,
            device=self.device,
        )

    def _allocate_contract_phantoms(self) -> None:
        """Create zero-stride phantom tensors at max capacity for stable cache keys."""
        # q is viewed as uint32 in the kernel: (max_total_q, num_q_heads, head_dim // 4).
        self._contract_q = _shape_only_cuda_tensor(
            (self.max_total_q, self.num_q_heads, self.head_dim // 4),
            dtype=torch.uint32,
            device=self.device,
        )
        self._contract_page_table = _shape_only_cuda_tensor(
            (self.max_total_q, self.topk),
            dtype=torch.int32,
            device=self.device,
        )
        self._contract_indexer_cache_seqlens = _shape_only_cuda_tensor(
            (self.max_total_q,),
            dtype=torch.int32,
            device=self.device,
        )
        self._contract_output = _shape_only_cuda_tensor(
            (self.max_total_q, self.num_q_heads, self.v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        if self.tmp_output is not None and self.tmp_lse is not None:
            self._contract_tmp_output = _shape_only_cuda_tensor(
                (
                    self.max_total_q,
                    self.num_q_heads,
                    self.max_chunks_per_row,
                    self.v_head_dim,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            self._contract_tmp_lse = _shape_only_cuda_tensor(
                (self.max_total_q, self.num_q_heads, self.max_chunks_per_row),
                dtype=torch.float32,
                device=self.device,
            )
        if self.ragged_kv_cache is not None:
            self._refresh_ragged_kv_contracts()
