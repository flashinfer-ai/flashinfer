# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/indexer/paged.py @ 77bd50eb (2026-07-01) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Paged sparse-indexer integration surface.

This module adapts the generic indexer scorer/top-k kernels to the DSV4/GLM
paged index-cache layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from flashinfer.experimental.sm12x.attention.nsa_indexer._impl import (
    build_paged_mqa_schedule_metadata,
    uses_paged_mqa_schedule,
)
from flashinfer.experimental.sm12x.attention.nsa_indexer.contiguous_kernel import (
    run_contiguous_logits_kernel,
)
from flashinfer.experimental.sm12x.attention.nsa_indexer.kernel import (
    _env_indexer_stream_scorer_enabled,
    _split_index_k_cache_runtime_views,
    run_paged_supertile_logits_kernel,
)
from flashinfer.experimental.sm12x.attention.nsa_indexer.tiled_topk import (
    run_row_topk,
    run_tiled_topk,
)

# Two-level fold: target slice width for level-1 pseudo-row parallelism and the
# cap that keeps the candidate buffers capacity-independent (~topk*8B*cap per
# row) at very long contexts.
_TWO_LEVEL_SLICE_TOKENS = 16384
_TWO_LEVEL_MAX_SLICES = 32
from flashinfer.experimental.sm12x.attention.nsa_indexer.reference import (
    pack_index_k_cache_reference,
    paged_decode_logits_reference,
    unpack_index_k_cache_reference,
)


INDEX_HEAD_DIM = 128
PAGED_INDEX_PAGE_SIZE = 64
_PAGED_INDEX_SUPERTILE_K_ENV = "FLASHINFER_EXP_SM12X_PAGED_INDEX_SUPERTILE_K"
_PAGED_INDEX_SUPERTILE_K_DEFAULT = 32768
_PAGED_INDEX_TILE_BLOCK_Q = 32
_PAGED_INDEX_TILE_BLOCK_K = 512
_PAGED_INDEX_CACHE_DATA_BYTES = PAGED_INDEX_PAGE_SIZE * INDEX_HEAD_DIM


@triton.jit(
    do_not_specialize=[
        "q_rows",
        "page_table_width",
        "source_page_offset",
        "cache_row_stride_bytes",
    ]
)
def _gather_shared_paged_supertile_kernel(
    index_k_cache,
    real_page_table,
    seqlens_per_query,
    k_quant_out,
    k_scale_bytes_out,
    k_start_out,
    k_end_out,
    q_rows,
    page_table_width,
    source_page_offset,
    supertile_tokens: tl.constexpr,
    block_tokens: tl.constexpr,
    page_size: tl.constexpr,
    index_head_dim: tl.constexpr,
    cache_row_stride_bytes,
    cache_data_bytes: tl.constexpr,
):
    pid = tl.program_id(0)
    token_offsets = pid * block_tokens + tl.arange(0, block_tokens)
    token_mask = token_offsets < supertile_tokens

    page_cols = source_page_offset + token_offsets // page_size
    slot_offsets = token_offsets % page_size
    page_ids = tl.load(
        real_page_table + page_cols,
        mask=token_mask & (page_cols < page_table_width),
        other=-1,
    )
    valid_tokens = token_mask & (page_ids >= 0)
    # Packed multi-group allocations can span more than 4 GiB. Keep address
    # math 64-bit even though the logical page IDs themselves are int32.
    page_byte_offsets = page_ids.to(tl.int64) * cache_row_stride_bytes

    byte_offsets = tl.arange(0, index_head_dim)
    k_bytes = tl.load(
        index_k_cache
        + page_byte_offsets[:, None]
        + slot_offsets[:, None] * index_head_dim
        + byte_offsets[None, :],
        mask=valid_tokens[:, None],
        other=0,
    )
    tl.store(
        k_quant_out + token_offsets[:, None] * index_head_dim + byte_offsets[None, :],
        k_bytes,
        mask=token_mask[:, None],
    )

    scale_byte_offsets = tl.arange(0, 4)
    scale_bytes = tl.load(
        index_k_cache
        + page_byte_offsets[:, None]
        + cache_data_bytes
        + slot_offsets[:, None] * 4
        + scale_byte_offsets[None, :],
        mask=valid_tokens[:, None],
        other=0,
    )
    tl.store(
        k_scale_bytes_out + token_offsets[:, None] * 4 + scale_byte_offsets[None, :],
        scale_bytes,
        mask=token_mask[:, None],
    )

    row_offsets = token_offsets
    row_mask = row_offsets < q_rows
    row_lengths = tl.load(seqlens_per_query + row_offsets, mask=row_mask, other=0)
    local_ends = row_lengths - source_page_offset * page_size
    local_ends = tl.minimum(tl.maximum(local_ends, 0), supertile_tokens)
    tl.store(
        k_start_out + row_offsets, tl.zeros((block_tokens,), tl.int32), mask=row_mask
    )
    tl.store(k_end_out + row_offsets, local_ends, mask=row_mask)


@dataclass(frozen=True)
class IndexerPagedMetadata:
    """Metadata for paged FP8 MQA indexer logits.

    ``expected_num_q_heads`` is optional for the generic path, but integrations
    should set it to the exact indexer-head count they pass to sm12x. Replicated
    selector paths such as paged use the full model-global selector-head count on
    every attention TP rank.
    """

    real_page_table: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    schedule_metadata: torch.Tensor | None = None
    expected_num_q_heads: int | None = None
    shared_page_table: bool = False


def resolve_replicated_num_q_heads(
    *,
    global_num_q_heads: int,
    tensor_parallel_size: int | None = None,
) -> int:
    """Return the replicated query/index head count used on every TP rank."""

    global_num_q_heads = int(global_num_q_heads)
    if global_num_q_heads <= 0:
        raise ValueError(
            f"global_num_q_heads must be positive, got {global_num_q_heads}"
        )
    if tensor_parallel_size is not None and int(tensor_parallel_size) <= 0:
        raise ValueError(
            f"tensor_parallel_size must be positive, got {int(tensor_parallel_size)}"
        )
    return global_num_q_heads


def resolve_local_num_q_heads(
    *,
    global_num_q_heads: int,
    tensor_parallel_size: int,
) -> int:
    """Return a TP-local head count for legacy sharded-indexer callers."""

    global_num_q_heads = int(global_num_q_heads)
    tensor_parallel_size = int(tensor_parallel_size)
    if global_num_q_heads <= 0:
        raise ValueError(
            f"global_num_q_heads must be positive, got {global_num_q_heads}"
        )
    if tensor_parallel_size <= 0:
        raise ValueError(
            f"tensor_parallel_size must be positive, got {tensor_parallel_size}"
        )
    if global_num_q_heads % tensor_parallel_size != 0:
        raise ValueError(
            f"global_num_q_heads={global_num_q_heads} is not divisible by "
            f"tensor_parallel_size={tensor_parallel_size}"
        )
    return global_num_q_heads // tensor_parallel_size


def _is_cuda_graph_capture_active(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _validate_i32_contiguous(
    tensor: torch.Tensor,
    *,
    name: str,
    ndim: int,
) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank-{ndim}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _is_row_shared_i32_matrix(tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim == 2
        and tensor.dtype == torch.int32
        and int(tensor.stride(0)) == 0
        and int(tensor.stride(1)) == 1
    )


def _validate_raw_page_lengths(
    *,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    page_size: int,
) -> None:
    """Reject positive lengths whose active page-table entries are missing."""

    if _is_cuda_graph_capture_active(real_page_table.device):
        raise RuntimeError(
            "paged-index metadata prep must run outside CUDA graph capture"
        )
    if (
        real_page_table.device.type == "cuda"
        and os.getenv("FLASHINFER_EXP_SM12X_VALIDATE_PAGED_INDEXER_CUDA_VALUES", "0")
        != "1"
    ):
        return
    if cache_seqlens_int32.numel() == 0:
        return
    if torch.any(cache_seqlens_int32 < 0).item():
        raise ValueError("cache_seqlens_int32 must be non-negative")

    max_width_tokens = int(real_page_table.shape[1]) * int(page_size)
    if torch.any(cache_seqlens_int32 > max_width_tokens).item():
        max_len = int(cache_seqlens_int32.max().item())
        raise ValueError(
            f"cache_seqlens_int32 contains length {max_len}, but page-table capacity "
            f"is {max_width_tokens} tokens"
        )

    required_pages = torch.div(
        cache_seqlens_int32.to(torch.int64) + int(page_size) - 1,
        int(page_size),
        rounding_mode="floor",
    )
    if real_page_table.shape[1] == 0:
        return
    cols = torch.arange(
        int(real_page_table.shape[1]),
        dtype=torch.int64,
        device=real_page_table.device,
    ).unsqueeze(0)
    active_page_mask = cols < required_pages.unsqueeze(1)
    if torch.any(active_page_mask & (real_page_table.to(torch.int64) < 0)).item():
        raise ValueError(
            "cache_seqlens_int32 marks page-table slots active, but real_page_table "
            "contains -1 in those slots; pass raw unclamped paged-index lengths"
        )


def _validate_schedule_metadata(
    schedule_metadata: torch.Tensor,
    *,
    device: torch.device,
) -> None:
    _validate_i32_contiguous(
        schedule_metadata,
        name="schedule_metadata",
        ndim=2,
    )
    if schedule_metadata.shape[1] != 2:
        raise ValueError(
            "schedule_metadata must have trailing dimension 2, got "
            f"{tuple(schedule_metadata.shape)}"
        )
    if schedule_metadata.device != device:
        raise ValueError(
            "schedule_metadata device "
            f"{schedule_metadata.device} does not match real_page_table device {device}"
        )


def prepare_paged_indexer_metadata(
    *,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    page_size: int = PAGED_INDEX_PAGE_SIZE,
    expected_num_q_heads: int | None = None,
    schedule_metadata: torch.Tensor | None = None,
    schedule_out: torch.Tensor | None = None,
    schedule_num_sms: int | None = None,
    build_schedule: bool | None = None,
    validate_raw_lengths: bool = True,
    shared_page_table: bool = False,
) -> IndexerPagedMetadata:
    """Validate and optionally build metadata for paged indexer logits.

    ``cache_seqlens_int32`` must be the raw token length for the paged indexer
    layout. Do not pass attention-kernel clamp-to-1 lengths here.
    """

    page_size = int(page_size)
    if page_size != PAGED_INDEX_PAGE_SIZE:
        raise ValueError(
            f"paged indexer currently supports page_size={PAGED_INDEX_PAGE_SIZE}, "
            f"got {page_size}"
        )
    if bool(shared_page_table) and _is_row_shared_i32_matrix(real_page_table):
        pass
    else:
        _validate_i32_contiguous(real_page_table, name="real_page_table", ndim=2)
    _validate_i32_contiguous(cache_seqlens_int32, name="cache_seqlens_int32", ndim=1)
    if real_page_table.shape[0] != cache_seqlens_int32.shape[0]:
        raise ValueError(
            f"real_page_table rows {real_page_table.shape[0]} do not match "
            f"cache_seqlens_int32 rows {cache_seqlens_int32.shape[0]}"
        )
    if real_page_table.device != cache_seqlens_int32.device:
        raise ValueError(
            f"real_page_table device {real_page_table.device} does not match "
            f"cache_seqlens_int32 device {cache_seqlens_int32.device}"
        )
    if expected_num_q_heads is not None:
        expected_num_q_heads = int(expected_num_q_heads)
        if expected_num_q_heads <= 0:
            raise ValueError(
                f"expected_num_q_heads must be positive, got {expected_num_q_heads}"
            )
    if validate_raw_lengths:
        _validate_raw_page_lengths(
            real_page_table=real_page_table,
            cache_seqlens_int32=cache_seqlens_int32,
            page_size=page_size,
        )

    if build_schedule is None:
        build_schedule = uses_paged_mqa_schedule(
            q_rows=int(real_page_table.shape[0]),
            max_pages=int(real_page_table.shape[1]),
        )
    if build_schedule:
        if schedule_metadata is not None and schedule_out is not None:
            raise ValueError("pass only one of schedule_metadata or schedule_out")
        if schedule_metadata is None:
            if _is_cuda_graph_capture_active(real_page_table.device):
                raise RuntimeError(
                    "paged-indexer schedule metadata must be built before CUDA graph capture"
                )
            schedule_metadata = build_paged_mqa_schedule_metadata(
                cache_seqlens_int32,
                page_size,
                schedule_num_sms,
                out=schedule_out,
            )
        else:
            _validate_schedule_metadata(
                schedule_metadata,
                device=real_page_table.device,
            )
    elif schedule_metadata is not None:
        _validate_schedule_metadata(
            schedule_metadata,
            device=real_page_table.device,
        )
    elif schedule_out is not None:
        raise ValueError("schedule_out was provided, but build_schedule is false")

    return IndexerPagedMetadata(
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens_int32,
        schedule_metadata=schedule_metadata,
        expected_num_q_heads=expected_num_q_heads,
        shared_page_table=bool(shared_page_table),
    )


def _resolve_binding_metadata(
    *,
    binding,
    metadata: IndexerPagedMetadata | None,
    active_width_override: torch.Tensor | None = None,
) -> tuple[IndexerPagedMetadata, object, torch.Tensor | None]:
    if binding is None:
        raise TypeError("paged indexer launch requires a plan binding")

    binding_scratch = getattr(binding, "scratch", None)
    if binding_scratch is None:
        raise TypeError("paged indexer binding is missing scratch")
    if metadata is not None:
        raise ValueError("pass either metadata or binding, not both")
    metadata = IndexerPagedMetadata(
        real_page_table=getattr(binding, "real_page_table"),
        cache_seqlens_int32=getattr(binding, "cache_seqlens_int32"),
        schedule_metadata=getattr(binding, "schedule_metadata", None),
        expected_num_q_heads=getattr(binding, "expected_num_q_heads", None),
        shared_page_table=bool(getattr(binding, "shared_page_table", False)),
    )
    if active_width_override is None:
        active_width_override = getattr(binding, "active_width", None)
    return metadata, binding_scratch, active_width_override


def _prepare_shared_paged_supertile(
    *,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    scratch,
    q_rows: int,
    page_table_width: int,
    page_begin: int,
    supertile_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if index_k_cache.ndim != 2 or index_k_cache.dtype != torch.uint8:
        raise ValueError(
            "shared paged-index supertile gather requires uint8 index_k_cache with "
            f"rank 2, got shape={tuple(index_k_cache.shape)} dtype={index_k_cache.dtype}"
        )
    if int(index_k_cache.stride(1)) != 1:
        raise ValueError(
            "shared paged-index supertile gather requires unit inner stride, "
            f"got stride={tuple(index_k_cache.stride())}"
        )
    expected_width = PAGED_INDEX_PAGE_SIZE * (INDEX_HEAD_DIM + 4)
    if int(index_k_cache.shape[1]) != expected_width:
        raise ValueError(
            f"index_k_cache width must be {expected_width}, got {int(index_k_cache.shape[1])}"
        )

    k_quant_bytes, k_scale_bytes = scratch.get_indexer_gather_outputs(
        row_count=supertile_tokens,
    )
    k_start = scratch.get_indexer_contiguous_lengths(row_count=q_rows)
    get_end = getattr(scratch, "get_paged_indexer_runtime_lengths", None)
    if get_end is None:
        raise RuntimeError("paged indexer scratch is missing runtime length scratch")
    k_end = get_end(row_count=q_rows)

    block_tokens = 128
    grid_elems = max(int(supertile_tokens), int(q_rows))
    grid = (triton.cdiv(grid_elems, block_tokens),)
    _gather_shared_paged_supertile_kernel[grid](
        index_k_cache,
        real_page_table,
        seqlens_per_query,
        k_quant_bytes,
        k_scale_bytes,
        k_start,
        k_end,
        q_rows,
        int(page_table_width),
        int(page_begin),
        int(supertile_tokens),
        block_tokens,
        PAGED_INDEX_PAGE_SIZE,
        INDEX_HEAD_DIM,
        int(index_k_cache.stride(0)),
        _PAGED_INDEX_CACHE_DATA_BYTES,
        num_warps=4,
    )

    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError(
            "torch.float8_e4m3fn is required for shared paged-index scoring"
        )
    return (
        k_quant_bytes.view(fp8_dtype),
        k_scale_bytes.view(torch.float32).view(-1),
        k_start,
        k_end,
    )


def _validate_q_head_contract(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    metadata: IndexerPagedMetadata,
    expected_num_q_heads: int | None,
    allow_partial_rows: bool,
) -> int:
    if q_fp8.ndim != 3:
        raise ValueError(f"q_fp8 must be rank-3, got {tuple(q_fp8.shape)}")
    if q_fp8.shape[2] != INDEX_HEAD_DIM:
        raise ValueError(
            f"q_fp8 head_dim must be {INDEX_HEAD_DIM}, got {q_fp8.shape[2]}"
        )
    if expected_num_q_heads is not None and metadata.expected_num_q_heads is not None:
        if int(expected_num_q_heads) != int(metadata.expected_num_q_heads):
            raise ValueError(
                "expected_num_q_heads argument does not match metadata "
                f"({expected_num_q_heads} vs {metadata.expected_num_q_heads})"
            )
    expected_heads = (
        int(expected_num_q_heads)
        if expected_num_q_heads is not None
        else metadata.expected_num_q_heads
    )
    if expected_heads is not None and q_fp8.shape[1] != int(expected_heads):
        raise ValueError(
            f"q_fp8 must use expected indexer head count {int(expected_heads)}, got "
            f"{q_fp8.shape[1]}"
        )
    if weights.ndim == 3:
        if weights.shape[2] != 1:
            raise ValueError(
                f"weights rank-3 input must have trailing dimension 1, got {tuple(weights.shape)}"
            )
        weight_shape = (weights.shape[0], weights.shape[1])
    elif weights.ndim == 2:
        weight_shape = tuple(weights.shape)
    else:
        raise ValueError(
            f"weights must be rank-2 or rank-3, got {tuple(weights.shape)}"
        )
    if weight_shape != (q_fp8.shape[0], q_fp8.shape[1]):
        raise ValueError(
            f"weights must have shape {(q_fp8.shape[0], q_fp8.shape[1])}, got "
            f"{tuple(weights.shape)}"
        )
    metadata_rows = int(metadata.real_page_table.shape[0])
    if allow_partial_rows:
        if metadata_rows > q_fp8.shape[0]:
            raise ValueError(
                f"metadata rows {metadata_rows} exceed q rows {q_fp8.shape[0]}"
            )
    elif metadata_rows != q_fp8.shape[0]:
        raise ValueError(
            f"metadata rows {metadata_rows} must match q rows {q_fp8.shape[0]}"
        )
    return int(expected_heads) if expected_heads is not None else int(q_fp8.shape[1])


def _weights_as_2d(weights: torch.Tensor) -> torch.Tensor:
    if weights.ndim == 3:
        return weights.squeeze(-1)
    return weights


def _resolve_supertile_k(supertile_k: int | None, *, page_size: int) -> int:
    if supertile_k is None:
        raw = os.environ.get(_PAGED_INDEX_SUPERTILE_K_ENV)
        if raw is None:
            supertile_k = _PAGED_INDEX_SUPERTILE_K_DEFAULT
        else:
            try:
                supertile_k = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{_PAGED_INDEX_SUPERTILE_K_ENV} must be an integer, got {raw!r}"
                ) from exc
    alignment = _PAGED_INDEX_TILE_BLOCK_K
    if alignment % int(page_size) != 0:
        raise ValueError(
            f"internal paged supertile alignment {alignment} must be divisible by page_size={page_size}"
        )
    supertile_k = max(int(supertile_k), alignment)
    return ((supertile_k + alignment - 1) // alignment) * alignment


def index_topk_fp8(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    metadata: IndexerPagedMetadata | None = None,
    binding=None,
    page_size: int = PAGED_INDEX_PAGE_SIZE,
    topk: int | None = None,
    expected_num_q_heads: int | None = None,
    out_indices: torch.Tensor | None = None,
    out_scores: torch.Tensor | None = None,
    supertile_k: int | None = None,
) -> torch.Tensor:
    """Indexer top-k selection over the paged FP8 index cache.

    Phase-agnostic and model-agnostic: serves DSV4 and GLM because the byte
    layout the kernel reads is identical, and spans both the few-row decode
    regime and the large-row shared-prefill regime (q_rows>=1024 via
    run_contiguous_logits_kernel). Routes the few-row decode case onto the fused
    score+select kernel (no external top-k blob) and otherwise scores into a
    tiled-logits supertile and folds the shared `run_tiled_topk` selector.
    Returns top-k indices per query row. When ``out_scores`` is provided, it is
    filled with the float32 top-k scores corresponding position-for-position to
    the returned indices. A binding with ``output_physical_slots=True`` makes the
    producer emit flat physical cache slots directly; no post-selection remap is
    performed or supported by this entrypoint.
    """

    metadata, scratch, binding_active_width = _resolve_binding_metadata(
        binding=binding,
        metadata=metadata,
    )
    page_size = int(page_size)
    if page_size != PAGED_INDEX_PAGE_SIZE:
        raise ValueError(
            f"paged indexer currently supports page_size={PAGED_INDEX_PAGE_SIZE}, "
            f"got {page_size}"
        )
    _validate_q_head_contract(
        q_fp8=q_fp8,
        weights=weights,
        metadata=metadata,
        expected_num_q_heads=expected_num_q_heads,
        allow_partial_rows=False,
    )
    weights = _weights_as_2d(weights)
    if q_fp8.device.type != "cuda":
        raise NotImplementedError("paged index supertile top-k requires CUDA")
    if metadata.real_page_table.device != q_fp8.device:
        raise ValueError("real_page_table must be on the same device as q_fp8")
    if not metadata.real_page_table.is_contiguous() and not (
        metadata.shared_page_table
        and _is_row_shared_i32_matrix(metadata.real_page_table)
    ):
        raise ValueError("metadata.real_page_table must be contiguous")

    q_rows = int(q_fp8.shape[0])
    if topk is None:
        if out_indices is not None:
            topk = int(out_indices.shape[1])
        elif out_scores is not None:
            topk = int(out_scores.shape[1])
        else:
            topk = int(getattr(scratch, "topk", 0))
            if topk <= 0:
                raise RuntimeError(
                    "paged index supertile top-k requires topk, out_indices, "
                    "out_scores, or a bound scratch with topk capacity"
                )
    topk = int(topk)
    if out_indices is not None:
        if out_indices.shape != (q_rows, topk):
            raise ValueError(
                f"out_indices must have shape {(q_rows, topk)}, got "
                f"{tuple(out_indices.shape)}"
            )
        if out_indices.dtype != torch.int32 or not out_indices.is_contiguous():
            raise ValueError("out_indices must be contiguous torch.int32")
    if out_scores is not None:
        if out_scores.shape != (q_rows, topk):
            raise ValueError(
                f"out_scores must have shape {(q_rows, topk)}, got "
                f"{tuple(out_scores.shape)}"
            )
        if out_scores.dtype != torch.float32 or not out_scores.is_contiguous():
            raise ValueError("out_scores must be contiguous torch.float32")
        if out_scores.device != q_fp8.device:
            raise ValueError("out_scores device must match q_fp8")

    page_table_width = int(metadata.real_page_table.shape[1])

    route = str(getattr(binding, "route", getattr(scratch, "route", "paged_tiled")))
    output_physical_slots = bool(getattr(binding, "output_physical_slots", False))
    # Fused score+top-k route: single launch, no logits blob. Route selection is
    # owned by the scratch plan; launch only carries it out.
    from flashinfer.experimental.sm12x.attention.nsa_indexer.fused_indexer import (
        run_fused_paged_indexer,
    )

    indexer_heads = int(q_fp8.shape[1])
    if route == "paged_fused":
        if bool(metadata.shared_page_table):
            raise RuntimeError(
                "fused paged indexer route cannot consume a shared page table"
            )
        cache = scratch.get_fused_indexer_scratch(topk=topk)
        quant, scales = _split_index_k_cache_runtime_views(index_k_cache)
        idx, _ = run_fused_paged_indexer(
            q_bytes=q_fp8.view(torch.uint8),
            weights=weights,
            k_quant_bytes=quant,
            k_scales=scales,
            real_page_table=metadata.real_page_table,
            seqlens=metadata.cache_seqlens_int32,
            num_heads=indexer_heads,
            topk=topk,
            out_indices=out_indices,
            out_values=out_scores,
            pack_values=cache[0],
            pack_indices=cache[1],
            merge_state=cache[2],
            merge_state_preinitialized=bool(
                getattr(scratch, "fused_indexer_merge_state_preinitialized", False)
            ),
            output_physical_slots=output_physical_slots,
        )
        return idx

    if supertile_k is None:
        supertile_k = getattr(binding, "supertile_k", None)
    if supertile_k is None:
        supertile_k = getattr(scratch, "paged_tile_logits_k_rows", None)
    supertile_tokens = _resolve_supertile_k(supertile_k, page_size=page_size)
    supertile_pages = max(1, supertile_tokens // page_size)
    num_chunks = max(1, (page_table_width + supertile_pages - 1) // supertile_pages)
    if int(getattr(scratch, "max_page_table_width", 0)) < page_table_width:
        raise RuntimeError(
            "paged index supertile top-k scratch page-table capacity is too small: "
            f"need={page_table_width}, have={getattr(scratch, 'max_page_table_width', None)}"
        )
    use_shared_prefill_scorer = route == "packed_contiguous"
    topk_block_k = (
        int(
            getattr(binding, "prefill_block_k", 0)
            or getattr(scratch, "prefill_block_k", 0)
        )
        if use_shared_prefill_scorer
        else _PAGED_INDEX_TILE_BLOCK_K
    )
    if topk_block_k <= 0:
        raise RuntimeError(
            "packed-contiguous paged indexer binding is missing prefill_block_k"
        )
    if supertile_tokens % topk_block_k != 0:
        raise RuntimeError(
            "paged indexer supertile width must be divisible by route block_k: "
            f"supertile_tokens={supertile_tokens}, block_k={topk_block_k}"
        )
    supertile_k_tiles = supertile_tokens // topk_block_k
    if route not in ("paged_tiled", "packed_contiguous"):
        raise RuntimeError(f"unsupported paged indexer route {route!r}")
    tile_logits = scratch.get_indexer_contiguous_tile_logits()
    if tile_logits is None:
        raise RuntimeError(
            "paged index supertile top-k scratch is missing tiled logits"
        )

    scratch_values, scratch_raw_indices = scratch.get_indexer_contiguous_topk_buffers(
        row_count=q_rows,
    )
    final_values = out_scores if out_scores is not None else scratch_values[:, :topk]
    final_raw_indices = (
        out_indices if out_indices is not None else scratch_raw_indices[:, :topk]
    )
    if final_values.shape != (q_rows, topk) or final_raw_indices.shape != (
        q_rows,
        topk,
    ):
        raise ValueError(
            f"paged indexer scratch buffers are smaller than requested paged top-k {topk}"
        )
    if final_values.dtype != torch.float32 or final_values.device != q_fp8.device:
        raise ValueError(
            "paged indexer top-k values must be a CUDA torch.float32 tensor "
            "on the q_fp8 device"
        )
    if (
        final_raw_indices.dtype != torch.int32
        or final_raw_indices.device != q_fp8.device
    ):
        raise ValueError(
            "out_indices must be a CUDA torch.int32 tensor on the q_fp8 device"
        )
    if not final_values.is_contiguous() or not final_raw_indices.is_contiguous():
        raise ValueError("paged indexer top-k buffers must be contiguous")
    # Streaming-fold carry double-buffer (2, M, topk): chunk j reads the running top-k
    # written by chunk j-1 and writes the next half; the final chunk writes the user
    # Two-level cross-chunk fold: each chunk's slices are top-k'd by parallel
    # pseudo-row CTAs straight into a shared candidate buffer (bounded by the
    # slice cap, NOT the context capacity); one linear pass folds them after
    # the last chunk. The per-chunk tile-logits scratch stays at the supertile
    # ceiling. Physical-slot output keeps the legacy carry chain (the fold
    # gather emits logical indices).
    width_tokens = page_table_width * page_size
    two_level_slices: list[tuple[int, int]] = []
    total_slices = 0
    fold_values = None
    fold_indices = None
    fold_lengths = None
    if not output_physical_slots and width_tokens >= 2 * _TWO_LEVEL_SLICE_TOKENS:
        slice_tokens = _TWO_LEVEL_SLICE_TOKENS
        min_slice = -(-width_tokens // _TWO_LEVEL_MAX_SLICES)
        if min_slice > slice_tokens:
            slice_tokens = -(-min_slice // page_size) * page_size
        base = 0
        for c in range(num_chunks):
            c_pages = (
                min((c + 1) * supertile_pages, page_table_width) - c * supertile_pages
            )
            c_tokens = c_pages * page_size
            splits_c = max(1, -(-c_tokens // slice_tokens))
            two_level_slices.append((splits_c, base))
            base += splits_c
        total_slices = base
        fold_values = torch.empty(
            (q_rows * total_slices, topk), dtype=torch.float32, device=q_fp8.device
        )
        fold_indices = torch.empty(
            (q_rows * total_slices, topk), dtype=torch.int32, device=q_fp8.device
        )
        fold_lengths = torch.full(
            (q_rows,), total_slices * topk, dtype=torch.int32, device=q_fp8.device
        )
    # output. Only needed when there is more than one supertile chunk.
    carry_buf_values = None
    carry_buf_indices = None
    if num_chunks > 1 and not two_level_slices:
        carry_buf_values, carry_buf_indices = (
            scratch.get_indexer_contiguous_candidate_buffers()
        )
        if carry_buf_values.shape[0] < 2 or carry_buf_indices.shape[0] < 2:
            raise RuntimeError(
                "paged indexer scratch carry buffers need a first dim of at least 2: "
                f"have={carry_buf_values.shape[0]}"
            )
        carry_buf_values = carry_buf_values[:2, :q_rows, :topk]
        carry_buf_indices = carry_buf_indices[:2, :q_rows, :topk]
        if (
            carry_buf_values.dtype != torch.float32
            or carry_buf_values.device != q_fp8.device
        ):
            raise ValueError(
                "paged indexer carry values must be a CUDA torch.float32 "
                "tensor on the q_fp8 device"
            )
        if (
            carry_buf_indices.dtype != torch.int32
            or carry_buf_indices.device != q_fp8.device
        ):
            raise ValueError(
                "paged indexer carry indices must be a CUDA torch.int32 "
                "tensor on the q_fp8 device"
            )
        if (
            not carry_buf_values.is_contiguous()
            or not carry_buf_indices.is_contiguous()
        ):
            raise ValueError("paged indexer carry buffers must be contiguous")

    active_width = (
        binding_active_width
        if binding_active_width is not None
        else scratch.get_paged_indexer_active_width_cap()
    )
    page_table_for_kernel = metadata.real_page_table
    lengths_for_kernel = metadata.cache_seqlens_int32
    if use_shared_prefill_scorer:
        if not bool(metadata.shared_page_table):
            raise RuntimeError(
                "packed-contiguous paged indexer route requires a shared page table"
            )
        indexer_q_capacity = max(
            int(getattr(scratch, "max_total_q", 0)),
            int(getattr(scratch, "max_paged_q_rows", 0)),
        )
        if indexer_q_capacity < q_rows:
            raise RuntimeError(
                "packed-contiguous paged-index route scratch capacity is too small: "
                f"q_rows={q_rows}, capacity={indexer_q_capacity}"
            )

    for chunk_idx in range(num_chunks):
        page_begin = chunk_idx * supertile_pages
        page_end = min(page_begin + supertile_pages, page_table_width)
        chunk_pages = page_end - page_begin
        chunk_width_tokens = chunk_pages * page_size
        chunk_start_token = page_begin * page_size
        if not _env_indexer_stream_scorer_enabled() and uses_paged_mqa_schedule(
            q_rows=q_rows, max_pages=chunk_pages
        ):
            # The streamed scorer schedules its own persistent grid over the
            # whole supertile; only the legacy tiled scorer needs the
            # unscheduled-tile contract.
            raise RuntimeError(
                "paged supertile top-k requires an unscheduled paged scorer tile; "
                f"reduce {_PAGED_INDEX_SUPERTILE_K_ENV} below "
                f"{chunk_width_tokens} tokens"
            )

        if use_shared_prefill_scorer:
            k_quant, k_scale, k_start, k_end = _prepare_shared_paged_supertile(
                index_k_cache=index_k_cache,
                real_page_table=page_table_for_kernel,
                seqlens_per_query=lengths_for_kernel,
                scratch=scratch,
                q_rows=q_rows,
                page_table_width=page_table_width,
                page_begin=page_begin,
                supertile_tokens=supertile_tokens,
            )
            logits = run_contiguous_logits_kernel(
                q_fp8=q_fp8,
                weights=weights,
                k_quant=k_quant,
                k_scale=k_scale,
                k_start=k_start,
                k_end=k_end,
                tile_logits=tile_logits,
                tile_k_offset=0,
                tile_num_k_tiles=supertile_k_tiles,
                prefill_block_k=topk_block_k,
            )
            # The shared scorer consumes local supertile K bounds, but tiled
            # top-k clips against global raw-token offsets below.
            topk_lengths = lengths_for_kernel
        else:
            logits = run_paged_supertile_logits_kernel(
                q_fp8=q_fp8,
                weights=weights,
                index_k_cache=index_k_cache,
                real_page_table=page_table_for_kernel,
                seqlens_per_query=lengths_for_kernel,
                active_width=active_width,
                tile_logits=tile_logits,
                source_page_offset=page_begin,
                output_width_tokens=supertile_tokens,
                page_size=page_size,
                tile_block_q=_PAGED_INDEX_TILE_BLOCK_Q,
                tile_block_k=_PAGED_INDEX_TILE_BLOCK_K,
                preinitialize_tile_logits=False,
            )
            topk_lengths = lengths_for_kernel
        if not logits.is_contiguous():
            raise RuntimeError(
                "paged supertile scorer returned non-contiguous tiled logits"
            )

        is_first = chunk_idx == 0
        is_last = chunk_idx == num_chunks - 1
        if carry_buf_values is not None:
            carry_values = carry_buf_values[(chunk_idx - 1) % 2]
            carry_indices = carry_buf_indices[(chunk_idx - 1) % 2]
            out_values = final_values if is_last else carry_buf_values[chunk_idx % 2]
            out_indices = (
                final_raw_indices if is_last else carry_buf_indices[chunk_idx % 2]
            )
        else:
            # Single chunk: is_first folds nothing and writes straight to the output.
            carry_values = None
            carry_indices = None
            out_values = final_values
            out_indices = final_raw_indices
        if two_level_slices:
            chunk_splits, chunk_slice_base = two_level_slices[chunk_idx]
            split_extent = -(-chunk_width_tokens // chunk_splits)
            split_extent = -(-split_extent // topk_block_k) * topk_block_k
            run_tiled_topk(
                tile_logits=tile_logits,
                k_start=None,
                lengths=topk_lengths,
                topk=topk,
                block_q=_PAGED_INDEX_TILE_BLOCK_Q,
                block_k=topk_block_k,
                output_values=fold_values,
                output_indices=fold_indices,
                num_k_tiles=supertile_k_tiles,
                input_index_offset=chunk_start_token,
                input_extent=split_extent,
                output_index_offset=chunk_start_token,
                zero_row_start=True,
                is_first=True,
                extent_splits=chunk_splits,
                output_row_stride=total_slices,
                output_row_base=chunk_slice_base,
            )
            if is_last:
                run_row_topk(
                    row_logits=fold_values.view(q_rows, total_slices * topk),
                    lengths=fold_lengths,
                    topk=topk,
                    output_values=final_values,
                    output_indices=final_raw_indices,
                    output_gather_table=fold_indices.view(q_rows, total_slices * topk),
                )
        else:
            run_tiled_topk(
                tile_logits=tile_logits,
                k_start=None,
                lengths=topk_lengths,
                topk=topk,
                block_q=_PAGED_INDEX_TILE_BLOCK_Q,
                block_k=topk_block_k,
                output_values=out_values,
                output_indices=out_indices,
                num_k_tiles=supertile_k_tiles,
                input_index_offset=chunk_start_token,
                input_extent=chunk_width_tokens,
                output_index_offset=chunk_start_token,
                zero_row_start=True,
                carry_values=carry_values,
                carry_indices=carry_indices,
                is_first=is_first,
                output_page_table=(
                    metadata.real_page_table
                    if is_last and output_physical_slots
                    else None
                ),
                output_page_size=page_size,
            )

    return final_raw_indices


pack_paged_index_k_cache_reference = pack_index_k_cache_reference
unpack_paged_index_k_cache_reference = unpack_index_k_cache_reference
paged_index_logits_reference = paged_decode_logits_reference

from .scratch import (
    SM12XIndexerPagedBinding,
    SM12XIndexerPagedScratch,
    SM12XIndexerPagedScratchCaps,
    SM12XIndexerPagedScratchPlan,
    plan_indexer_paged_scratch,
)


__all__ = [
    "SM12XIndexerPagedBinding",
    "SM12XIndexerPagedScratch",
    "SM12XIndexerPagedScratchCaps",
    "SM12XIndexerPagedScratchPlan",
    "INDEX_HEAD_DIM",
    "PAGED_INDEX_PAGE_SIZE",
    "IndexerPagedMetadata",
    "pack_paged_index_k_cache_reference",
    "paged_index_logits_reference",
    "index_topk_fp8",
    "prepare_paged_indexer_metadata",
    "plan_indexer_paged_scratch",
    "resolve_local_num_q_heads",
    "resolve_replicated_num_q_heads",
    "unpack_paged_index_k_cache_reference",
]
