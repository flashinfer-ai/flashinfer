# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/indexer/scratch.py @ c368a837 (2026-07-14) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Caller-owned scratch plans for indexer paths."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from flashinfer.experimental.sm12x.attention.nsa_indexer._impl import (
    IndexerContiguousMetadata,
    IndexerPagedDecodeMetadata,
)
from flashinfer.experimental.sm12x.attention.nsa_indexer.msa_reference import (
    MSA_BLOCK_TOKENS,
)
from flashinfer.experimental.sm12x._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    dtype_nbytes,
    materialize_scratch_view,
)
from flashinfer.experimental.sm12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)

_PAGED_INDEX_SUPERTILE_K_ENV = "FLASHINFER_EXP_SM12X_PAGED_INDEX_SUPERTILE_K"
_PAGED_INDEX_SUPERTILE_K_DEFAULT = 32768
_PAGED_INDEX_TILE_BLOCK_Q = 32
_PAGED_INDEX_TILE_BLOCK_K = 512
_PAGED_INDEX_HEAD_DIM = 128
_INDEXER_CONTIGUOUS_BLOCK_Q = 32
_INDEXER_CONTIGUOUS_PREFILL_BLOCK_K = 256
_INDEXER_CONTIGUOUS_DECODE_BLOCK_K = 64
_INDEXER_CONTIGUOUS_HEAD_DIM = 128
_INDEXER_CONTIGUOUS_SCALE_BYTES = 4
_INDEXER_CONTIGUOUS_TMA_DESC_WORDS = 16
INDEXER_PAGED_ROUTE_AUTO = "auto"
INDEXER_PAGED_ROUTE_FUSED = "paged_fused"
INDEXER_PAGED_ROUTE_TILED = "paged_tiled"
INDEXER_PAGED_ROUTE_PACKED_CONTIGUOUS = "packed_contiguous"
INDEXER_SOURCE_LAYOUT_PAGED = "paged"
INDEXER_SOURCE_LAYOUT_CONTIGUOUS = "contiguous"
_INDEXER_PAGED_ROUTES = frozenset(
    {
        INDEXER_PAGED_ROUTE_AUTO,
        INDEXER_PAGED_ROUTE_FUSED,
        INDEXER_PAGED_ROUTE_TILED,
        INDEXER_PAGED_ROUTE_PACKED_CONTIGUOUS,
    }
)
_INDEXER_SOURCE_LAYOUTS = frozenset(
    {
        INDEXER_SOURCE_LAYOUT_PAGED,
        INDEXER_SOURCE_LAYOUT_CONTIGUOUS,
    }
)


class SM12XIndexerTopKPositionBufferUnavailable(RuntimeError):
    """Raised when scratch does not reserve reusable top-k merge positions."""


@dataclass(frozen=True, kw_only=True)
class SM12XIndexerScratchCaps:
    """Production-facing capacity inputs for indexer scratch planning.

    `source_layout` describes the caller's K-cache contract, not the scratch
    implementation. For paged sources, row/page capacities are measured in
    indexer K-cache rows. C4 callers should therefore pass one K row per four
    model-context tokens.
    """

    device: torch.device | str
    source_layout: str
    num_q_heads: int
    max_q_rows: int
    topk: int
    mode: str = "decode"
    max_k_rows: int | None = None
    max_page_table_width: int | None = None
    page_size: int = 64
    supertile_k: int = 0
    shared_page_table: bool = False
    dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype = torch.uint8
    k_dtype: torch.dtype = torch.float8_e4m3fn
    max_batch: int | None = None
    reserve_paged_logits: bool = False
    paged_logits_k_rows: int = 0
    route: str = INDEXER_PAGED_ROUTE_AUTO
    prefill_block_k: int = _INDEXER_CONTIGUOUS_PREFILL_BLOCK_K
    score_mode: str = "nsa"
    num_idx_heads: int = 1

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)

        source_layout = str(self.source_layout)
        if source_layout not in _INDEXER_SOURCE_LAYOUTS:
            raise ValueError(
                "indexer source_layout must be one of "
                f"{sorted(_INDEXER_SOURCE_LAYOUTS)}, got {source_layout!r}"
            )
        object.__setattr__(self, "source_layout", source_layout)

        mode = str(self.mode)
        if mode not in ("decode", "prefill"):
            raise ValueError(
                f"indexer scratch mode must be decode or prefill, got {mode!r}"
            )
        object.__setattr__(self, "mode", mode)

        route = str(self.route)
        if route not in _INDEXER_PAGED_ROUTES:
            raise ValueError(
                "indexer route must be one of "
                f"{sorted(_INDEXER_PAGED_ROUTES)}, got {route!r}"
            )
        object.__setattr__(self, "route", route)
        score_mode = str(self.score_mode).lower()
        if score_mode not in ("nsa", "msa"):
            raise ValueError(
                f"indexer scratch score_mode must be nsa or msa, got {score_mode!r}"
            )
        object.__setattr__(self, "score_mode", score_mode)

        object.__setattr__(self, "num_q_heads", max(int(self.num_q_heads), 1))
        object.__setattr__(self, "num_idx_heads", max(int(self.num_idx_heads), 1))
        object.__setattr__(self, "max_q_rows", max(int(self.max_q_rows), 1))
        object.__setattr__(self, "topk", max(int(self.topk), 1))
        object.__setattr__(self, "page_size", max(int(self.page_size), 1))
        object.__setattr__(self, "supertile_k", max(int(self.supertile_k), 0))
        object.__setattr__(self, "shared_page_table", bool(self.shared_page_table))
        object.__setattr__(
            self, "reserve_paged_logits", bool(self.reserve_paged_logits)
        )
        object.__setattr__(
            self, "paged_logits_k_rows", max(int(self.paged_logits_k_rows), 0)
        )
        object.__setattr__(self, "prefill_block_k", max(int(self.prefill_block_k), 1))
        max_batch = self.max_q_rows if self.max_batch is None else self.max_batch
        object.__setattr__(self, "max_batch", max(int(max_batch), 1))

        max_k_rows = None if self.max_k_rows is None else max(int(self.max_k_rows), 1)
        max_page_table_width = (
            None
            if self.max_page_table_width is None
            else max(int(self.max_page_table_width), 1)
        )
        if source_layout == INDEXER_SOURCE_LAYOUT_PAGED:
            if max_page_table_width is None:
                if max_k_rows is None:
                    raise ValueError(
                        "paged indexer scratch planning requires max_page_table_width "
                        "or max_k_rows"
                    )
                max_page_table_width = (max_k_rows + int(self.page_size) - 1) // int(
                    self.page_size
                )
            object.__setattr__(self, "max_page_table_width", max_page_table_width)
            object.__setattr__(self, "max_k_rows", max_k_rows)
        else:
            if max_k_rows is None:
                raise ValueError(
                    "contiguous indexer scratch planning requires max_k_rows"
                )
            object.__setattr__(self, "max_k_rows", max_k_rows)
            object.__setattr__(self, "max_page_table_width", max_page_table_width)


@dataclass(frozen=True, kw_only=True)
class SM12XIndexerPagedScratchCaps:
    """Capacity inputs for the paged indexer scratch planner.

    `max_page_table_width` and `paged_tile_logits_k_rows` are measured in
    indexer K-cache rows/pages, not original model-context tokens. For compressed
    sources such as C4, callers must pass the compressed K-row capacity.
    """

    device: torch.device | str
    num_q_heads: int
    max_q_rows: int
    max_page_table_width: int
    topk: int
    dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype = torch.uint8
    max_batch: int | None = None
    page_size: int = 64
    max_k_rows: int = 0
    reserve_paged_logits: bool = True
    paged_logits_k_rows: int = 0
    paged_tile_logits_k_rows: int = 0
    mode: str = "decode"
    shared_page_table: bool = False
    route: str = INDEXER_PAGED_ROUTE_AUTO
    score_mode: str = "nsa"
    num_idx_heads: int = 1

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "num_q_heads", max(int(self.num_q_heads), 1))
        object.__setattr__(self, "num_idx_heads", max(int(self.num_idx_heads), 1))
        object.__setattr__(self, "max_q_rows", max(int(self.max_q_rows), 1))
        object.__setattr__(
            self,
            "max_page_table_width",
            max(int(self.max_page_table_width), 1),
        )
        object.__setattr__(self, "topk", max(int(self.topk), 1))
        max_batch = self.max_q_rows if self.max_batch is None else self.max_batch
        object.__setattr__(self, "max_batch", max(int(max_batch), 1))
        object.__setattr__(self, "page_size", max(int(self.page_size), 1))
        object.__setattr__(self, "max_k_rows", max(int(self.max_k_rows), 0))
        object.__setattr__(
            self, "reserve_paged_logits", bool(self.reserve_paged_logits)
        )
        object.__setattr__(
            self, "paged_logits_k_rows", max(int(self.paged_logits_k_rows), 0)
        )
        object.__setattr__(
            self,
            "paged_tile_logits_k_rows",
            max(int(self.paged_tile_logits_k_rows), 0),
        )
        mode = str(self.mode)
        if mode not in ("decode", "prefill"):
            raise ValueError(
                f"indexer paged scratch mode must be decode or prefill, got {mode!r}"
            )
        route = str(self.route)
        if route not in _INDEXER_PAGED_ROUTES:
            raise ValueError(
                "indexer paged scratch route must be one of "
                f"{sorted(_INDEXER_PAGED_ROUTES)}, got {route!r}"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "shared_page_table", bool(self.shared_page_table))
        object.__setattr__(self, "route", route)
        score_mode = str(self.score_mode).lower()
        if score_mode not in ("nsa", "msa"):
            raise ValueError(
                f"indexer paged scratch score_mode must be nsa or msa, got {score_mode!r}"
            )
        object.__setattr__(self, "score_mode", score_mode)


@dataclass(frozen=True, kw_only=True)
class SM12XIndexerContiguousScratchCaps:
    device: torch.device | str
    num_q_heads: int
    max_q_rows: int
    max_k_rows: int
    topk: int
    k_dtype: torch.dtype = torch.float8_e4m3fn
    supertile_k: int = 32768
    prefill_block_k: int = _INDEXER_CONTIGUOUS_PREFILL_BLOCK_K
    score_mode: str = "nsa"
    num_idx_heads: int = 1

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "num_q_heads", max(int(self.num_q_heads), 1))
        object.__setattr__(self, "num_idx_heads", max(int(self.num_idx_heads), 1))
        object.__setattr__(self, "max_q_rows", max(int(self.max_q_rows), 1))
        max_k_rows = max(int(self.max_k_rows), 1)
        max_k_rows = (
            (max_k_rows + _INDEXER_CONTIGUOUS_DECODE_BLOCK_K - 1)
            // _INDEXER_CONTIGUOUS_DECODE_BLOCK_K
        ) * _INDEXER_CONTIGUOUS_DECODE_BLOCK_K
        object.__setattr__(self, "max_k_rows", max_k_rows)
        object.__setattr__(self, "topk", max(int(self.topk), 1))
        supertile_k = max(int(self.supertile_k), int(self.prefill_block_k), 1)
        prefill_block_k = int(self.prefill_block_k)
        if prefill_block_k != _INDEXER_CONTIGUOUS_PREFILL_BLOCK_K:
            raise ValueError(
                "indexer contiguous scratch currently supports "
                f"prefill_block_k={_INDEXER_CONTIGUOUS_PREFILL_BLOCK_K}, got "
                f"{prefill_block_k}"
            )
        supertile_k = (
            (supertile_k + prefill_block_k - 1) // prefill_block_k
        ) * prefill_block_k
        object.__setattr__(self, "supertile_k", supertile_k)
        object.__setattr__(self, "prefill_block_k", prefill_block_k)
        score_mode = str(self.score_mode).lower()
        if score_mode not in ("nsa", "msa"):
            raise ValueError(
                f"indexer contiguous scratch score_mode must be nsa or msa, got {score_mode!r}"
            )
        object.__setattr__(self, "score_mode", score_mode)


@dataclass(frozen=True, kw_only=True)
class _SM12XIndexerPagedScratchLayout:
    nbytes: int
    supertile_tokens: int
    max_chunks: int
    route: str
    prefill_block_k: int | None
    tile_logits_elements: int
    gather_k_rows: int
    fused_pack_elements: int
    fused_state_words: int
    gather_k_quant_offset_bytes: int
    gather_k_scale_offset_bytes: int
    contiguous_lengths_offset_bytes: int
    runtime_lengths_offset_bytes: int
    tile_logits_offset_bytes: int
    topk_values_offset_bytes: int
    topk_indices_offset_bytes: int
    candidate_values_offset_bytes: int
    candidate_indices_offset_bytes: int
    merge_positions_offset_bytes: int
    active_width_offset_bytes: int
    fused_pack_values_offset_bytes: int
    fused_pack_indices_offset_bytes: int
    fused_merge_state_offset_bytes: int
    msa_page_scores_offset_bytes: int
    msa_block_scores_offset_bytes: int
    msa_q2k_indices_offset_bytes: int
    msa_topk_score_scratch_offset_bytes: int
    msa_topk_values_offset_bytes: int
    msa_topk_indices_offset_bytes: int
    msa_sort_values_offset_bytes: int
    msa_sort_indices_offset_bytes: int
    msa_expanded_page_table_offset_bytes: int
    msa_expanded_seqlens_offset_bytes: int


@dataclass(frozen=True, kw_only=True)
class _SM12XIndexerContiguousScratchLayout:
    nbytes: int
    max_k_rows: int
    max_chunk_tiles: int
    tile_logits_elements: int
    k_quant_offset_bytes: int
    k_scale_offset_bytes: int
    dummy_logits_offset_bytes: int
    tile_logits_offset_bytes: int
    lengths_offset_bytes: int
    topk_values_offset_bytes: int
    topk_indices_offset_bytes: int
    candidate_values_offset_bytes: int
    candidate_indices_offset_bytes: int
    metadata_k_start_offset_bytes: int
    metadata_k_end_offset_bytes: int
    k_tma_desc_offset_bytes: int
    k_tma_desc_ptrs_offset_bytes: int
    k_tma_prefill_desc_offset_bytes: int
    k_tma_prefill_desc_ptrs_offset_bytes: int
    msa_block_scores_offset_bytes: int
    msa_q2k_indices_offset_bytes: int
    msa_topk_score_scratch_offset_bytes: int
    msa_topk_values_offset_bytes: int
    msa_topk_indices_offset_bytes: int
    msa_sort_values_offset_bytes: int
    msa_sort_indices_offset_bytes: int


@dataclass(kw_only=True)
class SM12XIndexerPagedScratch:
    """Paged indexer scratch views over caller-owned storage."""

    shared_scratch: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    topk: int
    max_page_table_width: int
    max_total_q: int
    max_paged_q_rows: int
    max_batch: int
    page_size: int
    paged_tile_logits_k_rows: int
    max_chunks: int
    route: str
    shared_page_table: bool = False
    prefill_block_k: int | None = None
    fixed_capacity: bool = True
    use_cuda_graph: bool = False
    indexer_k_quant_bytes: torch.Tensor | None = None
    indexer_k_scales_bytes: torch.Tensor | None = None
    indexer_contiguous_lengths: torch.Tensor | None = None
    paged_indexer_runtime_lengths: torch.Tensor | None = None
    indexer_contiguous_tile_logits: torch.Tensor | None = None
    indexer_contiguous_topk_values: torch.Tensor | None = None
    indexer_contiguous_topk_indices: torch.Tensor | None = None
    indexer_contiguous_candidate_values: torch.Tensor | None = None
    indexer_contiguous_candidate_indices: torch.Tensor | None = None
    indexer_contiguous_topk_positions: torch.Tensor | None = None
    paged_indexer_active_width_cap: torch.Tensor | None = None
    fused_indexer_pack_values: torch.Tensor | None = None
    fused_indexer_pack_indices: torch.Tensor | None = None
    fused_indexer_merge_state: torch.Tensor | None = None
    fused_indexer_merge_state_preinitialized: bool = False
    msa_page_scores: torch.Tensor | None = None
    msa_block_scores: torch.Tensor | None = None
    msa_q2k_indices: torch.Tensor | None = None
    msa_topk_score_scratch: torch.Tensor | None = None
    msa_topk_values: torch.Tensor | None = None
    msa_topk_indices: torch.Tensor | None = None
    msa_sort_values: torch.Tensor | None = None
    msa_sort_indices: torch.Tensor | None = None
    msa_expanded_page_table: torch.Tensor | None = None
    msa_expanded_seqlens: torch.Tensor | None = None

    def bind(
        self,
        *,
        real_page_table: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        active_width: torch.Tensor | None = None,
        schedule_metadata: torch.Tensor | None = None,
        expected_num_q_heads: int | None = None,
        shared_page_table: bool | None = None,
        output_physical_slots: bool = False,
    ) -> "SM12XIndexerPagedBinding":
        if shared_page_table is None:
            shared_page_table = bool(self.shared_page_table)
        return build_indexer_paged_binding(
            scratch=self,
            real_page_table=real_page_table,
            cache_seqlens_int32=cache_seqlens_int32,
            active_width=active_width,
            schedule_metadata=schedule_metadata,
            expected_num_q_heads=expected_num_q_heads,
            shared_page_table=shared_page_table,
            output_physical_slots=output_physical_slots,
        )

    def bind_msa(
        self,
        *,
        real_page_table: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        active_width: torch.Tensor | None = None,
        schedule_metadata: torch.Tensor | None = None,
        topk: int | None = None,
    ) -> "SM12XIndexerMSAPagedBinding":
        return build_indexer_msa_paged_binding(
            scratch=self,
            real_page_table=real_page_table,
            cache_seqlens_int32=cache_seqlens_int32,
            active_width=active_width,
            schedule_metadata=schedule_metadata,
            topk=topk,
        )

    def get_indexer_contiguous_tile_logits(self) -> torch.Tensor:
        if self.indexer_contiguous_tile_logits is None:
            raise RuntimeError("paged indexer scratch is missing tiled logits")
        return self.indexer_contiguous_tile_logits

    def get_indexer_contiguous_topk_buffers(
        self,
        *,
        row_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.indexer_contiguous_topk_values is None
            or self.indexer_contiguous_topk_indices is None
        ):
            raise RuntimeError("paged indexer scratch is missing top-k buffers")
        row_count = int(row_count)
        if row_count < 0:
            raise ValueError(f"row_count must be non-negative, got {row_count}")
        if row_count > int(self.indexer_contiguous_topk_indices.shape[0]):
            raise ValueError(
                "row_count "
                f"{row_count} exceeds paged indexer scratch top-k capacity "
                f"{int(self.indexer_contiguous_topk_indices.shape[0])}"
            )
        return (
            self.indexer_contiguous_topk_values[:row_count],
            self.indexer_contiguous_topk_indices[:row_count],
        )

    def get_indexer_contiguous_candidate_buffers(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.indexer_contiguous_candidate_values is None
            or self.indexer_contiguous_candidate_indices is None
        ):
            raise RuntimeError("paged indexer scratch is missing candidate buffers")
        return (
            self.indexer_contiguous_candidate_values,
            self.indexer_contiguous_candidate_indices,
        )

    def get_indexer_contiguous_topk_position_buffer(
        self, *, row_count: int
    ) -> torch.Tensor:
        if self.indexer_contiguous_topk_positions is None:
            raise SM12XIndexerTopKPositionBufferUnavailable(
                "paged indexer scratch is missing top-k position buffer"
            )
        row_count = int(row_count)
        if row_count < 0:
            raise ValueError(f"row_count must be non-negative, got {row_count}")
        if row_count > int(self.indexer_contiguous_topk_positions.shape[0]):
            raise ValueError(
                "row_count "
                f"{row_count} exceeds paged indexer scratch position capacity "
                f"{int(self.indexer_contiguous_topk_positions.shape[0])}"
            )
        return self.indexer_contiguous_topk_positions[:row_count]

    def get_paged_indexer_active_width_cap(self) -> torch.Tensor:
        if self.paged_indexer_active_width_cap is None:
            raise RuntimeError("paged indexer scratch is missing active-width cap")
        return self.paged_indexer_active_width_cap

    def get_fused_indexer_scratch(
        self,
        *,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topk = int(topk)
        if topk <= 0 or topk > int(self.topk):
            raise ValueError(
                f"fused indexer topk must be in [1, {int(self.topk)}], got {topk}"
            )
        if (
            self.fused_indexer_pack_values is None
            or self.fused_indexer_pack_indices is None
            or self.fused_indexer_merge_state is None
        ):
            raise RuntimeError("paged indexer scratch is missing fused buffers")
        return (
            self.fused_indexer_pack_values,
            self.fused_indexer_pack_indices,
            self.fused_indexer_merge_state,
        )

    def get_indexer_gather_outputs(
        self,
        *,
        row_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.indexer_k_quant_bytes is None or self.indexer_k_scales_bytes is None:
            raise RuntimeError(
                "paged indexer scratch is missing packed-contiguous gather buffers"
            )
        row_count = int(row_count)
        if row_count < 0:
            raise ValueError(f"row_count must be non-negative, got {row_count}")
        if row_count > int(self.indexer_k_quant_bytes.shape[0]):
            raise ValueError(
                f"row_count {row_count} exceeds paged indexer gather capacity "
                f"{int(self.indexer_k_quant_bytes.shape[0])}"
            )
        return (
            self.indexer_k_quant_bytes[:row_count],
            self.indexer_k_scales_bytes[:row_count],
        )

    def get_indexer_contiguous_lengths(self, *, row_count: int) -> torch.Tensor:
        if self.indexer_contiguous_lengths is None:
            raise RuntimeError(
                "paged indexer scratch is missing packed-contiguous k_start"
            )
        row_count = int(row_count)
        if row_count < 0:
            raise ValueError(f"row_count must be non-negative, got {row_count}")
        if row_count > int(self.indexer_contiguous_lengths.shape[0]):
            raise ValueError(
                f"row_count {row_count} exceeds paged indexer metadata capacity "
                f"{int(self.indexer_contiguous_lengths.shape[0])}"
            )
        return self.indexer_contiguous_lengths[:row_count]

    def get_paged_indexer_runtime_lengths(self, *, row_count: int) -> torch.Tensor:
        if self.paged_indexer_runtime_lengths is None:
            raise RuntimeError(
                "paged indexer scratch is missing packed-contiguous k_end"
            )
        row_count = int(row_count)
        if row_count < 0:
            raise ValueError(f"row_count must be non-negative, got {row_count}")
        if row_count > int(self.paged_indexer_runtime_lengths.shape[0]):
            raise ValueError(
                f"row_count {row_count} exceeds paged indexer metadata capacity "
                f"{int(self.paged_indexer_runtime_lengths.shape[0])}"
            )
        return self.paged_indexer_runtime_lengths[:row_count]


@dataclass(kw_only=True)
class SM12XIndexerContiguousScratch:
    """Contiguous-indexer scratch views over caller-owned storage."""

    shared_scratch: torch.Tensor
    device: torch.device
    num_q_heads: int
    max_q_rows: int
    max_k_rows: int
    topk: int
    supertile_k: int
    prefill_block_k: int
    k_quant: torch.Tensor
    k_scale_bytes: torch.Tensor
    k_scale: torch.Tensor
    dummy_logits: torch.Tensor
    tile_logits: torch.Tensor
    lengths: torch.Tensor
    topk_values: torch.Tensor
    topk_indices: torch.Tensor
    candidate_values: torch.Tensor
    candidate_indices: torch.Tensor
    metadata_k_start: torch.Tensor
    metadata_k_end: torch.Tensor
    k_tma_desc_ptrs: torch.Tensor
    k_tma_prefill_desc_ptrs: torch.Tensor
    msa_block_scores: torch.Tensor | None = None
    msa_q2k_indices: torch.Tensor | None = None
    msa_topk_score_scratch: torch.Tensor | None = None
    msa_topk_values: torch.Tensor | None = None
    msa_topk_indices: torch.Tensor | None = None
    msa_sort_values: torch.Tensor | None = None
    msa_sort_indices: torch.Tensor | None = None

    def prepare_k_padding(self, *, k_rows: int) -> None:
        k_rows = int(k_rows)
        if k_rows < 0:
            raise ValueError(f"k_rows must be non-negative, got {k_rows}")
        if k_rows > int(self.max_k_rows):
            raise ValueError(
                f"k_rows {k_rows} exceed contiguous scratch capacity {self.max_k_rows}"
            )
        padded_rows = (
            (max(k_rows, 1) + int(self.prefill_block_k) - 1)
            // int(self.prefill_block_k)
        ) * int(self.prefill_block_k)
        padded_rows = min(padded_rows, int(self.max_k_rows))
        if padded_rows > k_rows:
            self.k_quant[k_rows:padded_rows].zero_()
            self.k_scale_bytes[k_rows:padded_rows].zero_()

    def bind_msa(
        self,
        *,
        k_start: torch.Tensor,
        k_end: torch.Tensor,
        topk: int | None = None,
    ) -> "SM12XIndexerMSAContiguousBinding":
        return build_indexer_msa_contiguous_binding(
            scratch=self,
            k_start=k_start,
            k_end=k_end,
            topk=topk,
        )


@dataclass(frozen=True, kw_only=True)
class SM12XIndexerPagedBinding:
    scratch: object
    metadata: IndexerPagedDecodeMetadata
    real_page_table: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    active_width: torch.Tensor
    schedule_metadata: torch.Tensor | None = None
    expected_num_q_heads: int | None = None
    shared_page_table: bool = False
    output_physical_slots: bool = False
    route: str = INDEXER_PAGED_ROUTE_TILED
    supertile_k: int | None = None
    prefill_block_k: int | None = None


@dataclass(frozen=True, kw_only=True)
class SM12XIndexerContiguousBinding:
    scratch: object
    metadata: IndexerContiguousMetadata
    topk: int | None = None
    tile_logits: torch.Tensor | None = None
    lengths: torch.Tensor | None = None
    output_values: torch.Tensor | None = None
    output_indices: torch.Tensor | None = None
    candidate_values: torch.Tensor | None = None
    candidate_indices: torch.Tensor | None = None
    merge_positions: torch.Tensor | None = None
    prefill_block_k: int | None = None
    supertile_k: int | None = None
    strict: bool = False


@dataclass(frozen=True, kw_only=True)
class SM12XIndexerMSAPagedBinding:
    scratch: object
    metadata: IndexerPagedDecodeMetadata
    real_page_table: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    active_width: torch.Tensor
    schedule_metadata: torch.Tensor | None = None
    page_scores: torch.Tensor | None = None
    block_scores: torch.Tensor | None = None
    q2k_indices: torch.Tensor | None = None
    topk_score_scratch: torch.Tensor | None = None
    topk_values: torch.Tensor | None = None
    topk_indices: torch.Tensor | None = None
    sort_values: torch.Tensor | None = None
    sort_indices: torch.Tensor | None = None
    expanded_page_table: torch.Tensor | None = None
    expanded_seqlens: torch.Tensor | None = None
    topk: int | None = None
    num_idx_heads: int | None = None
    strict: bool = True


@dataclass(frozen=True, kw_only=True)
class SM12XIndexerMSAContiguousBinding:
    scratch: object
    metadata: IndexerContiguousMetadata
    block_scores: torch.Tensor | None = None
    q2k_indices: torch.Tensor | None = None
    topk_score_scratch: torch.Tensor | None = None
    topk_values: torch.Tensor | None = None
    topk_indices: torch.Tensor | None = None
    sort_values: torch.Tensor | None = None
    sort_indices: torch.Tensor | None = None
    topk: int | None = None
    num_idx_heads: int | None = None
    strict: bool = True


def _resolve_indexer_paged_supertile_tokens(
    raw_tokens: int,
    *,
    capacity_tokens: int | None = None,
) -> int:
    use_capacity_default = False
    if int(raw_tokens) <= 0:
        raw = os.environ.get(_PAGED_INDEX_SUPERTILE_K_ENV)
        if raw is None:
            raw_tokens = _PAGED_INDEX_SUPERTILE_K_DEFAULT
            use_capacity_default = True
        else:
            try:
                raw_tokens = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{_PAGED_INDEX_SUPERTILE_K_ENV} must be an integer, got {raw!r}"
                ) from exc
    tokens = max(int(raw_tokens), _PAGED_INDEX_TILE_BLOCK_K)
    tokens = (
        (tokens + _PAGED_INDEX_TILE_BLOCK_K - 1) // _PAGED_INDEX_TILE_BLOCK_K
    ) * _PAGED_INDEX_TILE_BLOCK_K
    # The default is a chunk-size ceiling, not a request to reserve beyond the
    # caller's fixed cache capacity. Keeping the explicit argument and env knob
    # authoritative preserves tuning/debug behavior while making the normal
    # vLLM plan proportional to its configured page-table width.
    if use_capacity_default and capacity_tokens is not None:
        capacity = max(int(capacity_tokens), 1)
        capacity = (
            (capacity + _PAGED_INDEX_TILE_BLOCK_K - 1) // _PAGED_INDEX_TILE_BLOCK_K
        ) * _PAGED_INDEX_TILE_BLOCK_K
        tokens = min(tokens, max(capacity, _PAGED_INDEX_TILE_BLOCK_K))
    return tokens


def _resolve_indexer_paged_route(
    caps: SM12XIndexerPagedScratchCaps,
    *,
    supertile_tokens: int,
) -> tuple[str, int | None]:
    route = str(caps.route)
    prefill_block_k: int | None = None
    device = torch.device(caps.device)
    compute_capability: tuple[int, int] | None = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        compute_capability = (int(props.major), int(props.minor))
    if route == INDEXER_PAGED_ROUTE_AUTO:
        if bool(caps.shared_page_table) or str(caps.mode) == "prefill":
            route = INDEXER_PAGED_ROUTE_PACKED_CONTIGUOUS
        else:
            route = INDEXER_PAGED_ROUTE_TILED
            if os.getenv("FLASHINFER_EXP_SM12X_FUSED_INDEXER", "1") != "0":
                from flashinfer.experimental.sm12x.attention.nsa_indexer.fused_indexer import (
                    resolve_fused_indexer_path,
                )
                from flashinfer.experimental.sm12x.attention.nsa_indexer.kernel import (
                    _num_q_head_tiles,
                )

                width = int(caps.max_page_table_width) * int(caps.page_size)
                if resolve_fused_indexer_path(
                    topk=int(caps.topk),
                    num_rows=int(caps.max_q_rows),
                    width=int(width),
                    num_heads=int(caps.num_q_heads),
                    compute_capability=compute_capability,
                ) and _num_q_head_tiles(int(caps.num_q_heads)) in (1, 2, 4):
                    route = INDEXER_PAGED_ROUTE_FUSED
    if route == INDEXER_PAGED_ROUTE_PACKED_CONTIGUOUS:
        from flashinfer.experimental.sm12x.attention.nsa_indexer.contiguous_kernel import (
            resolve_contiguous_prefill_block_k,
        )

        prefill_block_k = resolve_contiguous_prefill_block_k(
            valid_q_rows=int(caps.max_q_rows),
            k_rows=int(supertile_tokens),
            num_heads=int(caps.num_q_heads),
        )
        if prefill_block_k is None:
            prefill_block_k = _INDEXER_CONTIGUOUS_PREFILL_BLOCK_K
        if int(supertile_tokens) % int(prefill_block_k) != 0:
            raise ValueError(
                "packed-contiguous paged indexer route requires supertile_tokens "
                f"divisible by prefill_block_k, got supertile_tokens={supertile_tokens}, "
                f"prefill_block_k={prefill_block_k}"
            )
    elif route == INDEXER_PAGED_ROUTE_FUSED:
        if bool(caps.shared_page_table) or str(caps.mode) == "prefill":
            raise ValueError("fused paged indexer route is decode-only")
        from flashinfer.experimental.sm12x.attention.nsa_indexer.fused_indexer import (
            resolve_fused_indexer_path,
        )
        from flashinfer.experimental.sm12x.attention.nsa_indexer.kernel import (
            _num_q_head_tiles,
        )

        width = int(caps.max_page_table_width) * int(caps.page_size)
        if not (
            os.getenv("FLASHINFER_EXP_SM12X_FUSED_INDEXER", "1") != "0"
            and resolve_fused_indexer_path(
                topk=int(caps.topk),
                num_rows=int(caps.max_q_rows),
                width=int(width),
                num_heads=int(caps.num_q_heads),
                compute_capability=compute_capability,
            )
            and _num_q_head_tiles(int(caps.num_q_heads)) in (1, 2, 4)
        ):
            raise ValueError(
                "fused paged indexer route is not available for the planned caps: "
                f"max_q_rows={int(caps.max_q_rows)}, topk={int(caps.topk)}, "
                f"width={width}, num_q_heads={int(caps.num_q_heads)}"
            )
    elif route != INDEXER_PAGED_ROUTE_TILED:
        raise ValueError(f"unsupported indexer paged route {route!r}")
    return route, prefill_block_k


def _indexer_paged_scratch_layout(
    caps: SM12XIndexerPagedScratchCaps,
) -> _SM12XIndexerPagedScratchLayout:
    max_q_rows = max(int(caps.max_q_rows), 1)
    page_size = max(int(caps.page_size), 1)
    supertile_tokens = _resolve_indexer_paged_supertile_tokens(
        int(caps.paged_tile_logits_k_rows),
        capacity_tokens=int(caps.max_page_table_width) * page_size,
    )
    if supertile_tokens % page_size != 0:
        raise ValueError(
            "paged indexer supertile width must be divisible by page_size, "
            f"got supertile_tokens={supertile_tokens}, page_size={page_size}"
        )
    route, prefill_block_k = _resolve_indexer_paged_route(
        caps,
        supertile_tokens=supertile_tokens,
    )
    supertile_pages = max(1, supertile_tokens // page_size)
    max_chunks = max(
        1,
        (int(caps.max_page_table_width) + supertile_pages - 1) // supertile_pages,
    )
    num_q_tiles = (
        max_q_rows + _PAGED_INDEX_TILE_BLOCK_Q - 1
    ) // _PAGED_INDEX_TILE_BLOCK_Q
    num_k_tiles = supertile_tokens // _PAGED_INDEX_TILE_BLOCK_K
    tile_logits_elements = max(
        1,
        num_q_tiles
        * num_k_tiles
        * _PAGED_INDEX_TILE_BLOCK_Q
        * _PAGED_INDEX_TILE_BLOCK_K,
    )
    device = torch.device(caps.device)
    if device.type == "cuda":
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    else:
        num_sms = 1
    fused_pack_elements = 0
    fused_state_words = 0
    if route == INDEXER_PAGED_ROUTE_FUSED:
        from flashinfer.experimental.sm12x.attention.nsa_indexer.fused_indexer import (
            fused_indexer_scratch_capacity,
        )

        fused_pack_elements, fused_state_words = fused_indexer_scratch_capacity(
            max_q_rows,
            int(caps.topk),
            int(num_sms),
        )
    gather_k_rows = (
        int(supertile_tokens) if route == INDEXER_PAGED_ROUTE_PACKED_CONTIGUOUS else 0
    )

    cursor = 0
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
    gather_k_quant_offset_bytes = cursor
    cursor += gather_k_rows * _PAGED_INDEX_HEAD_DIM * dtype_nbytes(torch.uint8)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    gather_k_scale_offset_bytes = cursor
    cursor += (
        gather_k_rows * _INDEXER_CONTIGUOUS_SCALE_BYTES * dtype_nbytes(torch.uint8)
    )
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    contiguous_lengths_offset_bytes = cursor
    cursor += max_q_rows * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    runtime_lengths_offset_bytes = cursor
    cursor += max_q_rows * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
    tile_logits_offset_bytes = cursor
    cursor += tile_logits_elements * dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    topk_values_offset_bytes = cursor
    cursor += max_q_rows * int(caps.topk) * dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    topk_indices_offset_bytes = cursor
    cursor += max_q_rows * int(caps.topk) * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    # Streaming-fold carry double-buffer: two (M, topk) halves ping-pong across
    # supertile chunks (replaces the old max_chunks-deep candidate slab and the
    # merge_positions buffer, both of which the merge step required).
    fold_carry_chunks = 2 if int(max_chunks) > 1 else 0

    candidate_values_offset_bytes = cursor
    cursor += (
        fold_carry_chunks * max_q_rows * int(caps.topk) * dtype_nbytes(torch.float32)
    )
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    candidate_indices_offset_bytes = cursor
    cursor += (
        fold_carry_chunks * max_q_rows * int(caps.topk) * dtype_nbytes(torch.int32)
    )
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    # merge_positions is gone with the merge step; keep the offset for layout
    # compatibility but reserve no bytes.
    merge_positions_offset_bytes = cursor

    active_width_offset_bytes = cursor
    cursor += dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    fused_pack_values_offset_bytes = cursor
    cursor += fused_pack_elements * dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    fused_pack_indices_offset_bytes = cursor
    cursor += fused_pack_elements * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    fused_merge_state_offset_bytes = cursor
    cursor += fused_state_words * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_enabled = str(caps.score_mode) == "msa"
    num_idx_heads = max(int(caps.num_idx_heads), 1)
    max_pages_even = int(caps.max_page_table_width)
    if max_pages_even % 2 != 0:
        max_pages_even += 1
    max_blocks = max(
        1,
        (int(caps.max_page_table_width) * page_size + MSA_BLOCK_TOKENS - 1)
        // MSA_BLOCK_TOKENS,
    )
    msa_page_score_elements = (
        num_idx_heads * max_q_rows * max_pages_even if msa_enabled else 0
    )
    msa_block_score_elements = (
        num_idx_heads * max_q_rows * max_blocks if msa_enabled else 0
    )
    msa_q2k_elements = num_idx_heads * max_q_rows * int(caps.topk) if msa_enabled else 0
    msa_expanded_rows = num_idx_heads * max_q_rows if msa_enabled else 0

    msa_page_scores_offset_bytes = cursor
    cursor += msa_page_score_elements * dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_block_scores_offset_bytes = cursor
    cursor += msa_block_score_elements * dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_q2k_indices_offset_bytes = cursor
    cursor += msa_q2k_elements * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_topk_score_scratch_offset_bytes = cursor
    cursor += msa_block_score_elements * dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_topk_values_offset_bytes = cursor
    cursor += msa_q2k_elements * dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_topk_indices_offset_bytes = cursor
    cursor += msa_q2k_elements * dtype_nbytes(torch.int64)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_sort_values_offset_bytes = cursor
    cursor += msa_q2k_elements * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_sort_indices_offset_bytes = cursor
    cursor += msa_q2k_elements * dtype_nbytes(torch.int64)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_expanded_page_table_offset_bytes = cursor
    cursor += (
        msa_expanded_rows * int(caps.max_page_table_width) * dtype_nbytes(torch.int32)
    )
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_expanded_seqlens_offset_bytes = cursor
    cursor += msa_expanded_rows * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    return _SM12XIndexerPagedScratchLayout(
        nbytes=max(int(cursor), SCRATCH_ALIGN_BYTES),
        supertile_tokens=supertile_tokens,
        max_chunks=max_chunks,
        route=route,
        prefill_block_k=prefill_block_k,
        tile_logits_elements=tile_logits_elements,
        gather_k_rows=gather_k_rows,
        fused_pack_elements=fused_pack_elements,
        fused_state_words=fused_state_words,
        gather_k_quant_offset_bytes=gather_k_quant_offset_bytes,
        gather_k_scale_offset_bytes=gather_k_scale_offset_bytes,
        contiguous_lengths_offset_bytes=contiguous_lengths_offset_bytes,
        runtime_lengths_offset_bytes=runtime_lengths_offset_bytes,
        tile_logits_offset_bytes=tile_logits_offset_bytes,
        topk_values_offset_bytes=topk_values_offset_bytes,
        topk_indices_offset_bytes=topk_indices_offset_bytes,
        candidate_values_offset_bytes=candidate_values_offset_bytes,
        candidate_indices_offset_bytes=candidate_indices_offset_bytes,
        merge_positions_offset_bytes=merge_positions_offset_bytes,
        active_width_offset_bytes=active_width_offset_bytes,
        fused_pack_values_offset_bytes=fused_pack_values_offset_bytes,
        fused_pack_indices_offset_bytes=fused_pack_indices_offset_bytes,
        fused_merge_state_offset_bytes=fused_merge_state_offset_bytes,
        msa_page_scores_offset_bytes=msa_page_scores_offset_bytes,
        msa_block_scores_offset_bytes=msa_block_scores_offset_bytes,
        msa_q2k_indices_offset_bytes=msa_q2k_indices_offset_bytes,
        msa_topk_score_scratch_offset_bytes=msa_topk_score_scratch_offset_bytes,
        msa_topk_values_offset_bytes=msa_topk_values_offset_bytes,
        msa_topk_indices_offset_bytes=msa_topk_indices_offset_bytes,
        msa_sort_values_offset_bytes=msa_sort_values_offset_bytes,
        msa_sort_indices_offset_bytes=msa_sort_indices_offset_bytes,
        msa_expanded_page_table_offset_bytes=msa_expanded_page_table_offset_bytes,
        msa_expanded_seqlens_offset_bytes=msa_expanded_seqlens_offset_bytes,
    )


def _indexer_contiguous_scratch_layout(
    caps: SM12XIndexerContiguousScratchCaps,
) -> _SM12XIndexerContiguousScratchLayout:
    max_q_rows = max(int(caps.max_q_rows), 1)
    max_k_rows = max(int(caps.max_k_rows), 1)
    topk = max(int(caps.topk), 1)
    prefill_block_k = int(caps.prefill_block_k)
    supertile_tiles = max(1, int(caps.supertile_k) // prefill_block_k)
    num_q_tiles = (
        max_q_rows + _INDEXER_CONTIGUOUS_BLOCK_Q - 1
    ) // _INDEXER_CONTIGUOUS_BLOCK_Q
    num_k_tiles = (max_k_rows + prefill_block_k - 1) // prefill_block_k
    max_chunk_tiles = min(supertile_tiles, num_k_tiles)
    tile_logits_elements = max(
        1,
        num_q_tiles * max_chunk_tiles * _INDEXER_CONTIGUOUS_BLOCK_Q * prefill_block_k,
    )

    cursor = 0
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
    k_quant_offset_bytes = cursor
    cursor += max_k_rows * _INDEXER_CONTIGUOUS_HEAD_DIM * dtype_nbytes(caps.k_dtype)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    k_scale_offset_bytes = cursor
    cursor += max_k_rows * _INDEXER_CONTIGUOUS_SCALE_BYTES * dtype_nbytes(torch.uint8)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    dummy_logits_offset_bytes = cursor
    cursor += dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    tile_logits_offset_bytes = cursor
    cursor += tile_logits_elements * dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    lengths_offset_bytes = cursor
    cursor += max_q_rows * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    topk_values_offset_bytes = cursor
    cursor += max_q_rows * topk * dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    topk_indices_offset_bytes = cursor
    cursor += max_q_rows * topk * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    candidate_values_offset_bytes = cursor
    cursor += 2 * max_q_rows * topk * dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    candidate_indices_offset_bytes = cursor
    cursor += 2 * max_q_rows * topk * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    metadata_k_start_offset_bytes = cursor
    cursor += max_q_rows * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    metadata_k_end_offset_bytes = cursor
    cursor += max_q_rows * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    k_tma_desc_offset_bytes = cursor
    cursor += _INDEXER_CONTIGUOUS_TMA_DESC_WORDS * dtype_nbytes(torch.uint64)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    k_tma_desc_ptrs_offset_bytes = cursor
    cursor += dtype_nbytes(torch.int64)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    k_tma_prefill_desc_offset_bytes = cursor
    cursor += _INDEXER_CONTIGUOUS_TMA_DESC_WORDS * dtype_nbytes(torch.uint64)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    k_tma_prefill_desc_ptrs_offset_bytes = cursor
    cursor += dtype_nbytes(torch.int64)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_enabled = str(caps.score_mode) == "msa"
    num_idx_heads = max(int(caps.num_idx_heads), 1)
    max_blocks = max(1, (max_k_rows + MSA_BLOCK_TOKENS - 1) // MSA_BLOCK_TOKENS)
    msa_block_score_elements = (
        num_idx_heads * max_q_rows * max_blocks if msa_enabled else 0
    )
    msa_q2k_elements = num_idx_heads * max_q_rows * topk if msa_enabled else 0

    msa_block_scores_offset_bytes = cursor
    cursor += msa_block_score_elements * dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_q2k_indices_offset_bytes = cursor
    cursor += msa_q2k_elements * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_topk_score_scratch_offset_bytes = cursor
    cursor += msa_block_score_elements * dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_topk_values_offset_bytes = cursor
    cursor += msa_q2k_elements * dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_topk_indices_offset_bytes = cursor
    cursor += msa_q2k_elements * dtype_nbytes(torch.int64)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_sort_values_offset_bytes = cursor
    cursor += msa_q2k_elements * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    msa_sort_indices_offset_bytes = cursor
    cursor += msa_q2k_elements * dtype_nbytes(torch.int64)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    return _SM12XIndexerContiguousScratchLayout(
        nbytes=max(int(cursor), SCRATCH_ALIGN_BYTES),
        max_k_rows=max_k_rows,
        max_chunk_tiles=max_chunk_tiles,
        tile_logits_elements=tile_logits_elements,
        k_quant_offset_bytes=k_quant_offset_bytes,
        k_scale_offset_bytes=k_scale_offset_bytes,
        dummy_logits_offset_bytes=dummy_logits_offset_bytes,
        tile_logits_offset_bytes=tile_logits_offset_bytes,
        lengths_offset_bytes=lengths_offset_bytes,
        topk_values_offset_bytes=topk_values_offset_bytes,
        topk_indices_offset_bytes=topk_indices_offset_bytes,
        candidate_values_offset_bytes=candidate_values_offset_bytes,
        candidate_indices_offset_bytes=candidate_indices_offset_bytes,
        metadata_k_start_offset_bytes=metadata_k_start_offset_bytes,
        metadata_k_end_offset_bytes=metadata_k_end_offset_bytes,
        k_tma_desc_offset_bytes=k_tma_desc_offset_bytes,
        k_tma_desc_ptrs_offset_bytes=k_tma_desc_ptrs_offset_bytes,
        k_tma_prefill_desc_offset_bytes=k_tma_prefill_desc_offset_bytes,
        k_tma_prefill_desc_ptrs_offset_bytes=k_tma_prefill_desc_ptrs_offset_bytes,
        msa_block_scores_offset_bytes=msa_block_scores_offset_bytes,
        msa_q2k_indices_offset_bytes=msa_q2k_indices_offset_bytes,
        msa_topk_score_scratch_offset_bytes=msa_topk_score_scratch_offset_bytes,
        msa_topk_values_offset_bytes=msa_topk_values_offset_bytes,
        msa_topk_indices_offset_bytes=msa_topk_indices_offset_bytes,
        msa_sort_values_offset_bytes=msa_sort_values_offset_bytes,
        msa_sort_indices_offset_bytes=msa_sort_indices_offset_bytes,
    )


def _materialize_indexer_paged_scratch(
    caps: SM12XIndexerPagedScratchCaps,
    scratch_storage: torch.Tensor,
    layout: _SM12XIndexerPagedScratchLayout,
) -> SM12XIndexerPagedScratch:
    max_q_rows = max(int(caps.max_q_rows), 1)
    topk = max(int(caps.topk), 1)
    gather_k_rows = int(layout.gather_k_rows)
    if gather_k_rows > 0:
        indexer_k_quant_bytes, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.gather_k_quant_offset_bytes,
            shape=(gather_k_rows, _PAGED_INDEX_HEAD_DIM),
            dtype=torch.uint8,
        )
        indexer_k_scales_bytes, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.gather_k_scale_offset_bytes,
            shape=(gather_k_rows, _INDEXER_CONTIGUOUS_SCALE_BYTES),
            dtype=torch.uint8,
        )
    else:
        indexer_k_quant_bytes = None
        indexer_k_scales_bytes = None
    contiguous_lengths, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.contiguous_lengths_offset_bytes,
        shape=(max_q_rows,),
        dtype=torch.int32,
    )
    runtime_lengths, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.runtime_lengths_offset_bytes,
        shape=(max_q_rows,),
        dtype=torch.int32,
    )
    tile_logits, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.tile_logits_offset_bytes,
        shape=(int(layout.tile_logits_elements),),
        dtype=torch.float32,
    )
    topk_values, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.topk_values_offset_bytes,
        shape=(max_q_rows, topk),
        dtype=torch.float32,
    )
    topk_indices, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.topk_indices_offset_bytes,
        shape=(max_q_rows, topk),
        dtype=torch.int32,
    )
    # Streaming-fold carry double-buffer (two halves) when chunking is possible;
    # otherwise no carry buffer. merge_positions is gone with the merge step.
    fold_carry_chunks = 2 if int(layout.max_chunks) > 1 else 0
    if fold_carry_chunks:
        candidate_values, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.candidate_values_offset_bytes,
            shape=(fold_carry_chunks, max_q_rows, topk),
            dtype=torch.float32,
        )
        candidate_indices, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.candidate_indices_offset_bytes,
            shape=(fold_carry_chunks, max_q_rows, topk),
            dtype=torch.int32,
        )
    else:
        candidate_values = None
        candidate_indices = None
    merge_positions = None
    active_width_cap, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.active_width_offset_bytes,
        shape=(1,),
        dtype=torch.int32,
    )
    width_cap = max(
        int(caps.max_page_table_width) * int(caps.page_size),
        int(layout.supertile_tokens),
        1,
    )
    active_width_cap.fill_(int(width_cap))
    if int(layout.fused_pack_elements) > 0 and int(layout.fused_state_words) > 0:
        fused_pack_values, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.fused_pack_values_offset_bytes,
            shape=(int(layout.fused_pack_elements),),
            dtype=torch.float32,
        )
        fused_pack_indices, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.fused_pack_indices_offset_bytes,
            shape=(int(layout.fused_pack_elements),),
            dtype=torch.int32,
        )
        fused_merge_state, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.fused_merge_state_offset_bytes,
            shape=(int(layout.fused_state_words),),
            dtype=torch.int32,
        )
        # One-time initialization at workspace bind/materialization.  Both fused
        # merge strategies restore their cross-launch counters before returning,
        # so serving graph replays do not need to capture a state memset.
        fused_merge_state.zero_()
    else:
        fused_pack_values = None
        fused_pack_indices = None
        fused_merge_state = None

    if str(caps.score_mode) == "msa":
        num_idx_heads = max(int(caps.num_idx_heads), 1)
        max_pages_even = int(caps.max_page_table_width)
        if max_pages_even % 2 != 0:
            max_pages_even += 1
        max_blocks = max(
            1,
            (
                int(caps.max_page_table_width) * int(caps.page_size)
                + MSA_BLOCK_TOKENS
                - 1
            )
            // MSA_BLOCK_TOKENS,
        )
        msa_page_scores, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_page_scores_offset_bytes,
            shape=(num_idx_heads, max_q_rows, max_pages_even),
            dtype=torch.float32,
        )
        msa_block_scores, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_block_scores_offset_bytes,
            shape=(num_idx_heads, max_q_rows, max_blocks),
            dtype=torch.float32,
        )
        msa_q2k_indices, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_q2k_indices_offset_bytes,
            shape=(num_idx_heads, max_q_rows, topk),
            dtype=torch.int32,
        )
        msa_topk_score_scratch, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_topk_score_scratch_offset_bytes,
            shape=(num_idx_heads, max_q_rows, max_blocks),
            dtype=torch.float32,
        )
        msa_topk_values, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_topk_values_offset_bytes,
            shape=(num_idx_heads, max_q_rows, topk),
            dtype=torch.float32,
        )
        msa_topk_indices, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_topk_indices_offset_bytes,
            shape=(num_idx_heads, max_q_rows, topk),
            dtype=torch.int64,
        )
        msa_sort_values, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_sort_values_offset_bytes,
            shape=(num_idx_heads, max_q_rows, topk),
            dtype=torch.int32,
        )
        msa_sort_indices, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_sort_indices_offset_bytes,
            shape=(num_idx_heads, max_q_rows, topk),
            dtype=torch.int64,
        )
        msa_expanded_page_table, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_expanded_page_table_offset_bytes,
            shape=(num_idx_heads * max_q_rows, int(caps.max_page_table_width)),
            dtype=torch.int32,
        )
        msa_expanded_seqlens, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_expanded_seqlens_offset_bytes,
            shape=(num_idx_heads * max_q_rows,),
            dtype=torch.int32,
        )
    else:
        msa_page_scores = None
        msa_block_scores = None
        msa_q2k_indices = None
        msa_topk_score_scratch = None
        msa_topk_values = None
        msa_topk_indices = None
        msa_sort_values = None
        msa_sort_indices = None
        msa_expanded_page_table = None
        msa_expanded_seqlens = None

    scratch = SM12XIndexerPagedScratch(
        shared_scratch=scratch_storage,
        device=caps.device,
        dtype=caps.dtype,
        kv_dtype=caps.kv_dtype,
        num_q_heads=caps.num_q_heads,
        topk=caps.topk,
        max_page_table_width=caps.max_page_table_width,
        max_total_q=caps.max_q_rows,
        max_paged_q_rows=caps.max_q_rows,
        max_batch=caps.max_batch,
        page_size=caps.page_size,
        paged_tile_logits_k_rows=layout.supertile_tokens,
        max_chunks=layout.max_chunks,
        route=layout.route,
        shared_page_table=bool(caps.shared_page_table),
        prefill_block_k=layout.prefill_block_k,
        indexer_k_quant_bytes=indexer_k_quant_bytes,
        indexer_k_scales_bytes=indexer_k_scales_bytes,
        indexer_contiguous_lengths=contiguous_lengths,
        paged_indexer_runtime_lengths=runtime_lengths,
        indexer_contiguous_tile_logits=tile_logits,
        indexer_contiguous_topk_values=topk_values,
        indexer_contiguous_topk_indices=topk_indices,
        indexer_contiguous_candidate_values=candidate_values,
        indexer_contiguous_candidate_indices=candidate_indices,
        indexer_contiguous_topk_positions=merge_positions,
        paged_indexer_active_width_cap=active_width_cap,
        fused_indexer_pack_values=fused_pack_values,
        fused_indexer_pack_indices=fused_pack_indices,
        fused_indexer_merge_state=fused_merge_state,
        fused_indexer_merge_state_preinitialized=fused_merge_state is not None,
        msa_page_scores=msa_page_scores,
        msa_block_scores=msa_block_scores,
        msa_q2k_indices=msa_q2k_indices,
        msa_topk_score_scratch=msa_topk_score_scratch,
        msa_topk_values=msa_topk_values,
        msa_topk_indices=msa_topk_indices,
        msa_sort_values=msa_sort_values,
        msa_sort_indices=msa_sort_indices,
        msa_expanded_page_table=msa_expanded_page_table,
        msa_expanded_seqlens=msa_expanded_seqlens,
    )
    return scratch


def _materialize_indexer_contiguous_scratch(
    caps: SM12XIndexerContiguousScratchCaps,
    scratch_storage: torch.Tensor,
    layout: _SM12XIndexerContiguousScratchLayout,
) -> SM12XIndexerContiguousScratch:
    max_q_rows = max(int(caps.max_q_rows), 1)
    max_k_rows = max(int(layout.max_k_rows), 1)
    topk = max(int(caps.topk), 1)
    k_quant, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.k_quant_offset_bytes,
        shape=(max_k_rows, _INDEXER_CONTIGUOUS_HEAD_DIM),
        dtype=caps.k_dtype,
    )
    k_scale_bytes, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.k_scale_offset_bytes,
        shape=(max_k_rows, _INDEXER_CONTIGUOUS_SCALE_BYTES),
        dtype=torch.uint8,
    )
    k_scale = k_scale_bytes.view(torch.float32).flatten()
    dummy_logits, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.dummy_logits_offset_bytes,
        shape=(1, 1),
        dtype=torch.float32,
    )
    tile_logits, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.tile_logits_offset_bytes,
        shape=(int(layout.tile_logits_elements),),
        dtype=torch.float32,
    )
    lengths, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.lengths_offset_bytes,
        shape=(max_q_rows,),
        dtype=torch.int32,
    )
    topk_values, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.topk_values_offset_bytes,
        shape=(max_q_rows, topk),
        dtype=torch.float32,
    )
    topk_indices, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.topk_indices_offset_bytes,
        shape=(max_q_rows, topk),
        dtype=torch.int32,
    )
    candidate_values, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.candidate_values_offset_bytes,
        shape=(2, max_q_rows, topk),
        dtype=torch.float32,
    )
    candidate_indices, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.candidate_indices_offset_bytes,
        shape=(2, max_q_rows, topk),
        dtype=torch.int32,
    )
    metadata_k_start, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.metadata_k_start_offset_bytes,
        shape=(max_q_rows,),
        dtype=torch.int32,
    )
    metadata_k_end, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.metadata_k_end_offset_bytes,
        shape=(max_q_rows,),
        dtype=torch.int32,
    )
    k_tma_desc, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.k_tma_desc_offset_bytes,
        shape=(_INDEXER_CONTIGUOUS_TMA_DESC_WORDS,),
        dtype=torch.uint64,
    )
    k_tma_desc_ptrs, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.k_tma_desc_ptrs_offset_bytes,
        shape=(1,),
        dtype=torch.int64,
    )
    k_tma_prefill_desc, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.k_tma_prefill_desc_offset_bytes,
        shape=(_INDEXER_CONTIGUOUS_TMA_DESC_WORDS,),
        dtype=torch.uint64,
    )
    k_tma_prefill_desc_ptrs, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.k_tma_prefill_desc_ptrs_offset_bytes,
        shape=(1,),
        dtype=torch.int64,
    )
    k_quant_bytes = k_quant.view(torch.uint8)
    if k_quant.device.type == "cuda":
        from flashinfer.experimental.sm12x.attention.nsa_indexer.contiguous_kernel import (
            _encode_contiguous_k_tma_descriptor_into,
        )

        _encode_contiguous_k_tma_descriptor_into(
            k_quant_bytes,
            k_tma_desc,
            k_tma_desc_ptrs,
            block_k=_INDEXER_CONTIGUOUS_DECODE_BLOCK_K,
        )
        _encode_contiguous_k_tma_descriptor_into(
            k_quant_bytes,
            k_tma_prefill_desc,
            k_tma_prefill_desc_ptrs,
            block_k=_INDEXER_CONTIGUOUS_PREFILL_BLOCK_K,
        )
    else:
        k_tma_desc_ptrs.fill_(int(k_tma_desc.data_ptr()))
        k_tma_prefill_desc_ptrs.fill_(int(k_tma_prefill_desc.data_ptr()))

    if str(caps.score_mode) == "msa":
        num_idx_heads = max(int(caps.num_idx_heads), 1)
        max_blocks = max(1, (max_k_rows + MSA_BLOCK_TOKENS - 1) // MSA_BLOCK_TOKENS)
        msa_block_scores, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_block_scores_offset_bytes,
            shape=(num_idx_heads, max_q_rows, max_blocks),
            dtype=torch.float32,
        )
        msa_q2k_indices, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_q2k_indices_offset_bytes,
            shape=(num_idx_heads, max_q_rows, topk),
            dtype=torch.int32,
        )
        msa_topk_score_scratch, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_topk_score_scratch_offset_bytes,
            shape=(num_idx_heads, max_q_rows, max_blocks),
            dtype=torch.float32,
        )
        msa_topk_values, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_topk_values_offset_bytes,
            shape=(num_idx_heads, max_q_rows, topk),
            dtype=torch.float32,
        )
        msa_topk_indices, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_topk_indices_offset_bytes,
            shape=(num_idx_heads, max_q_rows, topk),
            dtype=torch.int64,
        )
        msa_sort_values, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_sort_values_offset_bytes,
            shape=(num_idx_heads, max_q_rows, topk),
            dtype=torch.int32,
        )
        msa_sort_indices, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_sort_indices_offset_bytes,
            shape=(num_idx_heads, max_q_rows, topk),
            dtype=torch.int64,
        )
    else:
        msa_block_scores = None
        msa_q2k_indices = None
        msa_topk_score_scratch = None
        msa_topk_values = None
        msa_topk_indices = None
        msa_sort_values = None
        msa_sort_indices = None

    return SM12XIndexerContiguousScratch(
        shared_scratch=scratch_storage,
        device=caps.device,
        num_q_heads=caps.num_q_heads,
        max_q_rows=max_q_rows,
        max_k_rows=max_k_rows,
        topk=topk,
        supertile_k=caps.supertile_k,
        prefill_block_k=caps.prefill_block_k,
        k_quant=k_quant,
        k_scale_bytes=k_scale_bytes,
        k_scale=k_scale,
        dummy_logits=dummy_logits,
        tile_logits=tile_logits,
        lengths=lengths,
        topk_values=topk_values,
        topk_indices=topk_indices,
        candidate_values=candidate_values,
        candidate_indices=candidate_indices,
        metadata_k_start=metadata_k_start,
        metadata_k_end=metadata_k_end,
        k_tma_desc_ptrs=k_tma_desc_ptrs,
        k_tma_prefill_desc_ptrs=k_tma_prefill_desc_ptrs,
        msa_block_scores=msa_block_scores,
        msa_q2k_indices=msa_q2k_indices,
        msa_topk_score_scratch=msa_topk_score_scratch,
        msa_topk_values=msa_topk_values,
        msa_topk_indices=msa_topk_indices,
        msa_sort_values=msa_sort_values,
        msa_sort_indices=msa_sort_indices,
    )


def _validate_device(
    tensor: torch.Tensor,
    *,
    scratch: object | None = None,
    name: str,
) -> None:
    if scratch is None:
        raise TypeError("_validate_device requires scratch")
    if tensor.device != scratch.device:
        raise ValueError(
            f"{name} device {tensor.device} does not match resource device {scratch.device}"
        )


def _is_row_shared_i32_matrix(tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim == 2 and int(tensor.stride(0)) == 0 and int(tensor.stride(1)) == 1
    )


def _validate_i32_contiguous(
    tensor: torch.Tensor,
    *,
    scratch: object | None = None,
    name: str,
    ndim: int,
) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank-{ndim}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    _validate_device(tensor, scratch=scratch, name=name)


def build_indexer_paged_binding(
    *,
    scratch: object,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    active_width: torch.Tensor | None = None,
    schedule_metadata: torch.Tensor | None = None,
    expected_num_q_heads: int | None = None,
    shared_page_table: bool = False,
    output_physical_slots: bool = False,
) -> SM12XIndexerPagedBinding:
    if scratch is None:
        raise TypeError("build_indexer_paged_binding requires scratch")

    if bool(shared_page_table) and _is_row_shared_i32_matrix(real_page_table):
        if real_page_table.dtype != torch.int32:
            raise ValueError(
                f"real_page_table must have dtype torch.int32, got {real_page_table.dtype}"
            )
        _validate_device(real_page_table, scratch=scratch, name="real_page_table")
    else:
        _validate_i32_contiguous(
            real_page_table,
            scratch=scratch,
            name="real_page_table",
            ndim=2,
        )
    _validate_i32_contiguous(
        cache_seqlens_int32,
        scratch=scratch,
        name="cache_seqlens_int32",
        ndim=1,
    )
    if active_width is None:
        active_width = scratch.get_paged_indexer_active_width_cap()
    _validate_i32_contiguous(
        active_width,
        scratch=scratch,
        name="active_width",
        ndim=1,
    )
    if active_width.shape != (1,):
        raise ValueError(
            f"active_width must have shape (1,), got {tuple(active_width.shape)}"
        )
    if int(real_page_table.shape[0]) != int(cache_seqlens_int32.shape[0]):
        raise ValueError(
            f"real_page_table rows {int(real_page_table.shape[0])} do not match "
            f"cache_seqlens_int32 rows {int(cache_seqlens_int32.shape[0])}"
        )
    if int(real_page_table.shape[0]) > int(scratch.max_paged_q_rows):
        raise ValueError(
            f"real_page_table rows {int(real_page_table.shape[0])} exceed paged indexer capacity "
            f"{scratch.max_paged_q_rows}"
        )
    if int(real_page_table.shape[1]) > int(scratch.max_page_table_width):
        raise ValueError(
            f"real_page_table width {int(real_page_table.shape[1])} exceeds paged indexer capacity "
            f"{scratch.max_page_table_width}"
        )
    if bool(shared_page_table) != bool(
        getattr(scratch, "shared_page_table", bool(shared_page_table))
    ):
        raise ValueError(
            "shared_page_table does not match the paged indexer scratch plan: "
            f"launch={bool(shared_page_table)}, plan={bool(getattr(scratch, 'shared_page_table'))}"
        )
    if schedule_metadata is not None:
        _validate_i32_contiguous(
            schedule_metadata,
            scratch=scratch,
            name="schedule_metadata",
            ndim=2,
        )
        if int(schedule_metadata.shape[1]) != 2:
            raise ValueError(
                f"schedule_metadata must have shape (num_sms + 1, 2), got {tuple(schedule_metadata.shape)}"
            )
    if expected_num_q_heads is not None:
        expected_num_q_heads = int(expected_num_q_heads)
        if expected_num_q_heads <= 0:
            raise ValueError(
                f"expected_num_q_heads must be positive, got {expected_num_q_heads}"
            )
    return SM12XIndexerPagedBinding(
        scratch=scratch,
        metadata=IndexerPagedDecodeMetadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=cache_seqlens_int32,
            paged_mqa_schedule_metadata=schedule_metadata,
        ),
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens_int32,
        active_width=active_width,
        schedule_metadata=schedule_metadata,
        expected_num_q_heads=expected_num_q_heads,
        shared_page_table=bool(shared_page_table),
        output_physical_slots=bool(output_physical_slots),
        route=str(getattr(scratch, "route", INDEXER_PAGED_ROUTE_TILED)),
        supertile_k=int(getattr(scratch, "paged_tile_logits_k_rows", 0)) or None,
        prefill_block_k=getattr(scratch, "prefill_block_k", None),
    )


def build_indexer_contiguous_binding(
    *,
    scratch: object,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    gather_rows: int | None = None,
    topk: int | None = None,
    include_topk_buffers: bool = True,
    include_candidate_buffers: bool = True,
    include_lengths: bool = True,
    include_merge_positions: bool = True,
    strict: bool = False,
) -> SM12XIndexerContiguousBinding:
    if scratch is None:
        raise TypeError("build_indexer_contiguous_binding requires scratch")

    _validate_i32_contiguous(
        k_start,
        scratch=scratch,
        name="k_start",
        ndim=1,
    )
    _validate_i32_contiguous(
        k_end,
        scratch=scratch,
        name="k_end",
        ndim=1,
    )
    if k_start.shape != k_end.shape:
        raise ValueError(
            f"k_start and k_end must have the same shape, got "
            f"{tuple(k_start.shape)} and {tuple(k_end.shape)}"
        )
    row_count = int(k_start.shape[0])
    max_rows = int(
        getattr(scratch, "max_q_rows", getattr(scratch, "max_total_q", row_count))
    )
    if row_count > max_rows:
        raise ValueError(
            f"k_start rows {row_count} exceed indexer contiguous capacity {max_rows}"
        )
    if gather_rows is not None:
        gather_rows = int(gather_rows)
        max_k_rows = int(getattr(scratch, "max_k_rows", gather_rows))
        if gather_rows < 0:
            raise ValueError(f"gather_rows must be non-negative, got {gather_rows}")
        if gather_rows > max_k_rows:
            raise ValueError(
                f"gather_rows {gather_rows} exceed indexer contiguous K capacity "
                f"{max_k_rows}"
            )

    if topk is None:
        topk = int(getattr(scratch, "topk", getattr(scratch, "indexer_topk", 0)))
    topk = int(topk)
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    tile_logits = getattr(scratch, "tile_logits", None)
    if tile_logits is None and hasattr(scratch, "get_indexer_contiguous_tile_logits"):
        tile_logits = scratch.get_indexer_contiguous_tile_logits()
    if tile_logits is None:
        tile_logits = getattr(scratch, "indexer_contiguous_tile_logits", None)

    output_values = None
    output_indices = None
    if include_topk_buffers:
        if hasattr(scratch, "topk_values") and hasattr(scratch, "topk_indices"):
            output_values = scratch.topk_values[:row_count, :topk]
            output_indices = scratch.topk_indices[:row_count, :topk]
        else:
            output_values, output_indices = scratch.get_indexer_contiguous_topk_buffers(
                row_count=row_count
            )
            output_values = output_values[:, :topk]
            output_indices = output_indices[:, :topk]

    candidate_values = None
    candidate_indices = None
    if include_candidate_buffers:
        if hasattr(scratch, "candidate_values") and hasattr(
            scratch, "candidate_indices"
        ):
            candidate_values = scratch.candidate_values[:, :row_count, :topk]
            candidate_indices = scratch.candidate_indices[:, :row_count, :topk]
        else:
            candidate_values, candidate_indices = (
                scratch.get_indexer_contiguous_candidate_buffers()
            )
            candidate_values = candidate_values[:, :row_count, :topk]
            candidate_indices = candidate_indices[:, :row_count, :topk]

    lengths = None
    if include_lengths:
        if hasattr(scratch, "lengths"):
            lengths = scratch.lengths[:row_count]
        else:
            lengths = scratch.get_indexer_contiguous_lengths(row_count=row_count)

    merge_positions = None
    if include_merge_positions and not strict:
        try:
            merge_positions = scratch.get_indexer_contiguous_topk_position_buffer(
                row_count=row_count
            )[:, :topk]
        except SM12XIndexerTopKPositionBufferUnavailable:
            merge_positions = None
        except AttributeError:
            merge_positions = getattr(
                scratch, "indexer_contiguous_topk_positions", None
            )
            if merge_positions is not None:
                merge_positions = merge_positions[:row_count, :topk]

    return SM12XIndexerContiguousBinding(
        scratch=scratch,
        metadata=IndexerContiguousMetadata(k_start=k_start, k_end=k_end),
        topk=topk,
        tile_logits=tile_logits,
        lengths=lengths,
        output_values=output_values,
        output_indices=output_indices,
        candidate_values=candidate_values,
        candidate_indices=candidate_indices,
        merge_positions=merge_positions,
        prefill_block_k=getattr(scratch, "prefill_block_k", None),
        supertile_k=getattr(scratch, "supertile_k", None),
        strict=bool(strict),
    )


def _require_msa_scratch_tensor(scratch: object, name: str) -> torch.Tensor:
    tensor = getattr(scratch, name, None)
    if tensor is None:
        raise RuntimeError(
            "MSA indexer binding requires scratch planned with score_mode='msa'; "
            f"missing {name}"
        )
    return tensor


def build_indexer_msa_paged_binding(
    *,
    scratch: object,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    active_width: torch.Tensor | None = None,
    schedule_metadata: torch.Tensor | None = None,
    topk: int | None = None,
) -> SM12XIndexerMSAPagedBinding:
    if scratch is None:
        raise TypeError("build_indexer_msa_paged_binding requires scratch")
    _validate_i32_contiguous(
        real_page_table,
        scratch=scratch,
        name="real_page_table",
        ndim=2,
    )
    _validate_i32_contiguous(
        cache_seqlens_int32,
        scratch=scratch,
        name="cache_seqlens_int32",
        ndim=1,
    )
    if int(real_page_table.shape[0]) != int(cache_seqlens_int32.shape[0]):
        raise ValueError("real_page_table rows must match cache_seqlens_int32")
    if active_width is None:
        active_width = scratch.get_paged_indexer_active_width_cap()
    _validate_i32_contiguous(active_width, scratch=scratch, name="active_width", ndim=1)
    if active_width.shape != (1,):
        raise ValueError(
            f"active_width must have shape (1,), got {tuple(active_width.shape)}"
        )
    if schedule_metadata is not None:
        _validate_i32_contiguous(
            schedule_metadata,
            scratch=scratch,
            name="schedule_metadata",
            ndim=2,
        )
        if int(schedule_metadata.shape[1]) != 2:
            raise ValueError("schedule_metadata must have trailing dimension 2")
    if int(real_page_table.shape[0]) > int(
        getattr(scratch, "max_total_q", real_page_table.shape[0])
    ):
        raise ValueError("real_page_table rows exceed MSA paged scratch capacity")
    if int(real_page_table.shape[1]) > int(
        getattr(scratch, "max_page_table_width", real_page_table.shape[1])
    ):
        raise ValueError("real_page_table width exceeds MSA paged scratch capacity")

    block_scores = _require_msa_scratch_tensor(scratch, "msa_block_scores")
    q2k_indices = _require_msa_scratch_tensor(scratch, "msa_q2k_indices")
    page_scores = _require_msa_scratch_tensor(scratch, "msa_page_scores")
    topk = int(getattr(scratch, "topk", q2k_indices.shape[2]) if topk is None else topk)
    if topk <= 0 or topk > int(q2k_indices.shape[2]):
        raise ValueError(
            f"topk must be in [1, {int(q2k_indices.shape[2])}], got {topk}"
        )
    num_idx_heads = int(block_scores.shape[0])
    return SM12XIndexerMSAPagedBinding(
        scratch=scratch,
        metadata=IndexerPagedDecodeMetadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=cache_seqlens_int32,
            paged_mqa_schedule_metadata=schedule_metadata,
        ),
        real_page_table=real_page_table,
        cache_seqlens_int32=cache_seqlens_int32,
        active_width=active_width,
        schedule_metadata=schedule_metadata,
        page_scores=page_scores,
        block_scores=block_scores,
        q2k_indices=q2k_indices,
        topk_score_scratch=_require_msa_scratch_tensor(
            scratch, "msa_topk_score_scratch"
        ),
        topk_values=_require_msa_scratch_tensor(scratch, "msa_topk_values"),
        topk_indices=_require_msa_scratch_tensor(scratch, "msa_topk_indices"),
        sort_values=_require_msa_scratch_tensor(scratch, "msa_sort_values"),
        sort_indices=_require_msa_scratch_tensor(scratch, "msa_sort_indices"),
        expanded_page_table=_require_msa_scratch_tensor(
            scratch, "msa_expanded_page_table"
        ),
        expanded_seqlens=_require_msa_scratch_tensor(scratch, "msa_expanded_seqlens"),
        topk=topk,
        num_idx_heads=num_idx_heads,
        strict=True,
    )


def build_indexer_msa_contiguous_binding(
    *,
    scratch: object,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    topk: int | None = None,
) -> SM12XIndexerMSAContiguousBinding:
    if scratch is None:
        raise TypeError("build_indexer_msa_contiguous_binding requires scratch")
    _validate_i32_contiguous(k_start, scratch=scratch, name="k_start", ndim=1)
    _validate_i32_contiguous(k_end, scratch=scratch, name="k_end", ndim=1)
    if k_start.shape != k_end.shape:
        raise ValueError("k_start and k_end must have matching shapes")
    if int(k_start.shape[0]) > int(getattr(scratch, "max_q_rows", k_start.shape[0])):
        raise ValueError("k_start rows exceed MSA contiguous scratch capacity")
    block_scores = _require_msa_scratch_tensor(scratch, "msa_block_scores")
    q2k_indices = _require_msa_scratch_tensor(scratch, "msa_q2k_indices")
    topk = int(getattr(scratch, "topk", q2k_indices.shape[2]) if topk is None else topk)
    if topk <= 0 or topk > int(q2k_indices.shape[2]):
        raise ValueError(
            f"topk must be in [1, {int(q2k_indices.shape[2])}], got {topk}"
        )
    return SM12XIndexerMSAContiguousBinding(
        scratch=scratch,
        metadata=IndexerContiguousMetadata(k_start=k_start, k_end=k_end),
        block_scores=block_scores,
        q2k_indices=q2k_indices,
        topk_score_scratch=_require_msa_scratch_tensor(
            scratch, "msa_topk_score_scratch"
        ),
        topk_values=_require_msa_scratch_tensor(scratch, "msa_topk_values"),
        topk_indices=_require_msa_scratch_tensor(scratch, "msa_topk_indices"),
        sort_values=_require_msa_scratch_tensor(scratch, "msa_sort_values"),
        sort_indices=_require_msa_scratch_tensor(scratch, "msa_sort_indices"),
        topk=topk,
        num_idx_heads=int(block_scores.shape[0]),
        strict=True,
    )


@dataclass(frozen=True)
class SM12XIndexerPagedScratchPlan:
    caps: SM12XIndexerPagedScratchCaps
    layout: _SM12XIndexerPagedScratchLayout
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        real_page_table: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        active_width: torch.Tensor | None = None,
        schedule_metadata: torch.Tensor | None = None,
        expected_num_q_heads: int | None = None,
        shared_page_table: bool | None = None,
        output_physical_slots: bool = False,
    ) -> SM12XIndexerPagedBinding:
        scratch_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="paged indexer",
        )
        scratch_views = _materialize_indexer_paged_scratch(
            self.caps,
            scratch_storage,
            self.layout,
        )
        if shared_page_table is None:
            shared_page_table = bool(self.caps.shared_page_table)
        return build_indexer_paged_binding(
            scratch=scratch_views,
            real_page_table=real_page_table,
            cache_seqlens_int32=cache_seqlens_int32,
            active_width=active_width,
            schedule_metadata=schedule_metadata,
            expected_num_q_heads=expected_num_q_heads,
            shared_page_table=shared_page_table,
            output_physical_slots=output_physical_slots,
        )

    def bind_msa(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        real_page_table: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        active_width: torch.Tensor | None = None,
        schedule_metadata: torch.Tensor | None = None,
        topk: int | None = None,
    ) -> SM12XIndexerMSAPagedBinding:
        scratch_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="paged MSA indexer",
        )
        scratch_views = _materialize_indexer_paged_scratch(
            self.caps,
            scratch_storage,
            self.layout,
        )
        return build_indexer_msa_paged_binding(
            scratch=scratch_views,
            real_page_table=real_page_table,
            cache_seqlens_int32=cache_seqlens_int32,
            active_width=active_width,
            schedule_metadata=schedule_metadata,
            topk=topk,
        )


@dataclass(frozen=True)
class SM12XIndexerContiguousScratchPlan:
    caps: SM12XIndexerContiguousScratchCaps
    layout: _SM12XIndexerContiguousScratchLayout
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        k_start: torch.Tensor | None = None,
        k_end: torch.Tensor | None = None,
        gather_rows: int | None = None,
        topk: int | None = None,
    ) -> SM12XIndexerContiguousBinding:
        scratch_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="indexer contiguous",
        )
        scratch_views = _materialize_indexer_contiguous_scratch(
            self.caps,
            scratch_storage,
            self.layout,
        )
        if k_start is None:
            k_start = scratch_views.metadata_k_start
        if k_end is None:
            k_end = scratch_views.metadata_k_end
        return build_indexer_contiguous_binding(
            scratch=scratch_views,
            k_start=k_start,
            k_end=k_end,
            gather_rows=gather_rows,
            topk=self.caps.topk if topk is None else topk,
            include_topk_buffers=True,
            include_candidate_buffers=True,
            include_lengths=True,
            include_merge_positions=False,
            strict=True,
        )

    def bind_msa(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        k_start: torch.Tensor | None = None,
        k_end: torch.Tensor | None = None,
        topk: int | None = None,
    ) -> SM12XIndexerMSAContiguousBinding:
        scratch_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="indexer contiguous MSA",
        )
        scratch_views = _materialize_indexer_contiguous_scratch(
            self.caps,
            scratch_storage,
            self.layout,
        )
        if k_start is None:
            k_start = scratch_views.metadata_k_start
        if k_end is None:
            k_end = scratch_views.metadata_k_end
        return build_indexer_msa_contiguous_binding(
            scratch=scratch_views,
            k_start=k_start,
            k_end=k_end,
            topk=self.caps.topk if topk is None else topk,
        )


def plan_indexer_paged_scratch(
    caps: SM12XIndexerPagedScratchCaps,
) -> SM12XIndexerPagedScratchPlan:
    layout = _indexer_paged_scratch_layout(caps)
    return SM12XIndexerPagedScratchPlan(
        caps=caps,
        layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "paged_indexer.scratch",
                nbytes=int(layout.nbytes),
                device=caps.device,
            ),
        ),
    )


def plan_indexer_contiguous_scratch(
    caps: SM12XIndexerContiguousScratchCaps,
) -> SM12XIndexerContiguousScratchPlan:
    layout = _indexer_contiguous_scratch_layout(caps)
    return SM12XIndexerContiguousScratchPlan(
        caps=caps,
        layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "indexer_contiguous.arena",
                nbytes=int(layout.nbytes),
                device=caps.device,
            ),
        ),
    )


@dataclass(frozen=True)
class SM12XIndexerScratchPlan:
    caps: SM12XIndexerScratchCaps
    inner: SM12XIndexerPagedScratchPlan | SM12XIndexerContiguousScratchPlan

    @property
    def layout(self):
        return self.inner.layout

    @property
    def source_layout(self) -> str:
        return self.caps.source_layout

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self.inner.scratch_specs()

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return self.inner.shapes_and_dtypes()

    def bind(self, **kwargs):
        return self.inner.bind(**kwargs)

    def bind_msa(self, **kwargs):
        return self.inner.bind_msa(**kwargs)


def plan_indexer_scratch(
    caps: SM12XIndexerScratchCaps,
) -> SM12XIndexerScratchPlan:
    if caps.source_layout == INDEXER_SOURCE_LAYOUT_PAGED:
        assert caps.max_page_table_width is not None
        inner = plan_indexer_paged_scratch(
            SM12XIndexerPagedScratchCaps(
                device=caps.device,
                num_q_heads=caps.num_q_heads,
                max_q_rows=caps.max_q_rows,
                max_page_table_width=caps.max_page_table_width,
                topk=caps.topk,
                dtype=caps.dtype,
                kv_dtype=caps.kv_dtype,
                max_batch=caps.max_batch,
                page_size=caps.page_size,
                max_k_rows=0 if caps.max_k_rows is None else caps.max_k_rows,
                reserve_paged_logits=caps.reserve_paged_logits,
                paged_logits_k_rows=caps.paged_logits_k_rows,
                paged_tile_logits_k_rows=caps.supertile_k,
                mode=caps.mode,
                shared_page_table=caps.shared_page_table,
                route=caps.route,
                score_mode=caps.score_mode,
                num_idx_heads=caps.num_idx_heads,
            )
        )
    elif caps.source_layout == INDEXER_SOURCE_LAYOUT_CONTIGUOUS:
        assert caps.max_k_rows is not None
        inner = plan_indexer_contiguous_scratch(
            SM12XIndexerContiguousScratchCaps(
                device=caps.device,
                num_q_heads=caps.num_q_heads,
                max_q_rows=caps.max_q_rows,
                max_k_rows=caps.max_k_rows,
                topk=caps.topk,
                k_dtype=caps.k_dtype,
                supertile_k=(
                    caps.supertile_k if caps.supertile_k > 0 else caps.max_k_rows
                ),
                prefill_block_k=caps.prefill_block_k,
                score_mode=caps.score_mode,
                num_idx_heads=caps.num_idx_heads,
            )
        )
    else:
        raise ValueError(f"unsupported indexer source_layout {caps.source_layout!r}")
    return SM12XIndexerScratchPlan(caps=caps, inner=inner)


__all__ = [
    "ScratchBufferSpec",
    "SM12XIndexerScratchCaps",
    "SM12XIndexerScratchPlan",
    "SM12XIndexerPagedBinding",
    "SM12XIndexerPagedScratch",
    "SM12XIndexerPagedScratchCaps",
    "SM12XIndexerPagedScratchPlan",
    "SM12XIndexerContiguousBinding",
    "SM12XIndexerMSAPagedBinding",
    "SM12XIndexerMSAContiguousBinding",
    "SM12XIndexerContiguousScratch",
    "SM12XIndexerContiguousScratchCaps",
    "SM12XIndexerContiguousScratchPlan",
    "INDEXER_SOURCE_LAYOUT_CONTIGUOUS",
    "INDEXER_SOURCE_LAYOUT_PAGED",
    "build_indexer_paged_binding",
    "build_indexer_contiguous_binding",
    "build_indexer_msa_paged_binding",
    "build_indexer_msa_contiguous_binding",
    "plan_indexer_scratch",
    "plan_indexer_paged_scratch",
    "plan_indexer_contiguous_scratch",
]
