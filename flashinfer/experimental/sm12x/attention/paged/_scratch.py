# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/integration/paged_attention_scratch.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Caller-owned scratch plans for the primary paged-attention backend.

Eager PLAN -> BIND -> KERNEL. The scratch plan owns only shape/capacity policy
and tiny shape-only planning tensors. Each bind maps caller-owned uint8 scratch
into plain tensor views, prepares/copies metadata into those views, and returns
a binding consumed by ``flashinfer.experimental.sm12x.attention.paged._forward.paged_attention_forward``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import os
from typing import Literal

import torch
from torch.profiler import record_function

from flashinfer.experimental.sm12x.attention.paged.planner import (
    PagedPlan,
    PagedPlanBudget,
    build_decode_chunk_pages_lut,
    create_paged_plan,
    decode_graph_max_chunks_per_request_budget,
    infer_paged_mode,
    resolve_decode_graph_ctas_per_sm,
)
from flashinfer.experimental.sm12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)
from flashinfer.experimental.sm12x._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    dtype_nbytes,
    materialize_scratch_view,
    shape_numel,
)


def _paged_lse_storage_shape(total_q: int, num_q_heads: int) -> tuple[int, int]:
    return (num_q_heads, total_q)


def _copy_int_metadata(
    values: tuple[int, ...] | tuple[bool, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=device)


def _canonical_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _page_table_width_for_scratch(caps: "SM12XPagedAttentionScratchCaps") -> int:
    width = int(caps.max_page_table_width)
    if (
        caps.mode == "decode"
        and caps.use_cuda_graph
        and caps.copy_runtime_metadata
        and caps.kv_dtype == torch.float8_e4m3fn
        and width >= 16_384
        and width & (width - 1) == 0
    ):
        return width + 1
    return width


def _shape_only_cuda_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"shape must be positive in every dimension, got {shape}")
    base = torch.empty(1, dtype=dtype, device=device)
    return base.as_strided(shape, (0,) * len(shape))


def _infer_mode_from_host_total(
    cu_seqlens_q: torch.Tensor,
    active_total_q: int,
) -> Literal["decode", "extend"]:
    batch = max(int(cu_seqlens_q.shape[0]) - 1, 0)
    return "decode" if batch > 0 and int(active_total_q) == batch else "extend"


@dataclass(frozen=True, kw_only=True)
class SM12XPagedAttentionScratchCaps:
    device: torch.device | str
    mode: Literal["decode", "extend", "verify"]
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    page_size: int
    max_total_q: int
    max_batch: int
    max_page_table_width: int
    max_work_items: int
    max_partial_rows: int
    num_cache_pages: int
    use_cuda_graph: bool = False
    copy_runtime_metadata: bool = True
    msa_block_sparse: bool = False
    msa_union_tile: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "msa_block_sparse", bool(self.msa_block_sparse))
        if (
            self.msa_block_sparse
            and os.environ.get("FLASHINFER_EXP_SM12X_PAGED_MSA") == "0"
        ):
            raise RuntimeError(
                "FLASHINFER_EXP_SM12X_PAGED_MSA=0 disables MSA block-sparse paged attention"
            )
        if self.msa_union_tile is None:
            msa_union_tile = (
                self.msa_block_sparse
                and self.mode == "extend"
                and os.environ.get("FLASHINFER_EXP_SM12X_PAGED_MSA_UNION_PREFILL", "1")
                != "0"
            )
        else:
            msa_union_tile = bool(self.msa_union_tile)
        if msa_union_tile and not (self.msa_block_sparse and self.mode == "extend"):
            raise ValueError(
                "msa_union_tile requires mode='extend' and msa_block_sparse=True"
            )
        object.__setattr__(self, "msa_union_tile", msa_union_tile)
        object.__setattr__(self, "device", _canonical_device(self.device))
        object.__setattr__(self, "num_q_heads", max(int(self.num_q_heads), 1))
        object.__setattr__(self, "num_kv_heads", max(int(self.num_kv_heads), 1))
        object.__setattr__(self, "head_dim_qk", max(int(self.head_dim_qk), 1))
        object.__setattr__(self, "head_dim_vo", max(int(self.head_dim_vo), 1))
        object.__setattr__(self, "page_size", max(int(self.page_size), 1))
        object.__setattr__(self, "max_total_q", max(int(self.max_total_q), 1))
        object.__setattr__(self, "max_batch", max(int(self.max_batch), 1))
        object.__setattr__(
            self,
            "max_page_table_width",
            max(int(self.max_page_table_width), 1),
        )
        object.__setattr__(self, "max_work_items", max(int(self.max_work_items), 0))
        object.__setattr__(self, "max_partial_rows", max(int(self.max_partial_rows), 0))
        object.__setattr__(self, "num_cache_pages", max(int(self.num_cache_pages), 1))
        object.__setattr__(self, "use_cuda_graph", bool(self.use_cuda_graph))
        object.__setattr__(
            self, "copy_runtime_metadata", bool(self.copy_runtime_metadata)
        )
        object.__setattr__(
            self,
            "max_page_table_width",
            _page_table_width_for_scratch(self),
        )


@dataclass(frozen=True)
class _SM12XPagedAttentionScratchLayout:
    nbytes: int
    request_indices_offset_bytes: int
    qo_tile_indices_offset_bytes: int
    kv_tile_indices_offset_bytes: int
    block_valid_mask_offset_bytes: int
    page_table_offset_bytes: int | None
    cache_seqlens_offset_bytes: int | None
    cu_seqlens_q_offset_bytes: int | None
    merge_indptr_offset_bytes: int
    o_indptr_offset_bytes: int
    kv_chunk_size_ptr_offset_bytes: int
    kv_window_start_tokens_offset_bytes: int
    total_num_rows_ptr_offset_bytes: int
    lse_offset_bytes: int
    tmp_output_offset_bytes: int
    tmp_lse_offset_bytes: int
    msa_union_blocks_offset_bytes: int | None
    msa_union_masks_offset_bytes: int | None
    msa_union_counts_offset_bytes: int | None


def _paged_attention_scratch_layout(
    caps: SM12XPagedAttentionScratchCaps,
) -> _SM12XPagedAttentionScratchLayout:
    offset = 0

    def reserve(shape: tuple[int, ...], dtype: torch.dtype) -> int:
        nonlocal offset
        offset = align_up(offset, max(SCRATCH_ALIGN_BYTES, dtype_nbytes(dtype)))
        current = offset
        offset += shape_numel(shape) * dtype_nbytes(dtype)
        return current

    request_indices_offset_bytes = reserve((caps.max_work_items,), torch.int32)
    qo_tile_indices_offset_bytes = reserve((caps.max_work_items,), torch.int32)
    kv_tile_indices_offset_bytes = reserve((caps.max_work_items,), torch.int32)
    block_valid_mask_offset_bytes = reserve((caps.max_work_items,), torch.int32)
    if caps.copy_runtime_metadata:
        page_table_offset_bytes = reserve(
            (caps.max_batch, caps.max_page_table_width),
            torch.int32,
        )
        cache_seqlens_offset_bytes = reserve((caps.max_batch,), torch.int32)
        cu_seqlens_q_offset_bytes = reserve((caps.max_batch + 1,), torch.int32)
    else:
        page_table_offset_bytes = None
        cache_seqlens_offset_bytes = None
        cu_seqlens_q_offset_bytes = None
    merge_indptr_offset_bytes = reserve((caps.max_total_q + 1,), torch.int32)
    o_indptr_offset_bytes = reserve((caps.max_batch + 1,), torch.int32)
    kv_chunk_size_ptr_offset_bytes = reserve((1,), torch.int32)
    kv_window_start_tokens_offset_bytes = reserve((caps.max_batch,), torch.int32)
    total_num_rows_ptr_offset_bytes = reserve((1,), torch.int32)
    lse_offset_bytes = reserve(
        _paged_lse_storage_shape(caps.max_total_q, caps.num_q_heads),
        torch.float32,
    )
    tmp_output_offset_bytes = reserve(
        (caps.max_partial_rows, caps.num_q_heads, caps.head_dim_vo),
        caps.dtype,
    )
    tmp_lse_offset_bytes = reserve(
        (caps.max_partial_rows, caps.num_q_heads),
        torch.float32,
    )
    msa_union_blocks_offset_bytes = None
    msa_union_masks_offset_bytes = None
    msa_union_counts_offset_bytes = None
    if caps.msa_union_tile:
        msa_union_blocks_offset_bytes = reserve(
            (caps.max_work_items, caps.num_kv_heads, 128),
            torch.int32,
        )
        msa_union_masks_offset_bytes = reserve(
            (caps.max_work_items, caps.num_kv_heads, 128),
            torch.int32,
        )
        msa_union_counts_offset_bytes = reserve(
            (caps.max_work_items, caps.num_kv_heads),
            torch.int32,
        )

    return _SM12XPagedAttentionScratchLayout(
        nbytes=max(int(offset), SCRATCH_ALIGN_BYTES),
        request_indices_offset_bytes=request_indices_offset_bytes,
        qo_tile_indices_offset_bytes=qo_tile_indices_offset_bytes,
        kv_tile_indices_offset_bytes=kv_tile_indices_offset_bytes,
        block_valid_mask_offset_bytes=block_valid_mask_offset_bytes,
        page_table_offset_bytes=page_table_offset_bytes,
        cache_seqlens_offset_bytes=cache_seqlens_offset_bytes,
        cu_seqlens_q_offset_bytes=cu_seqlens_q_offset_bytes,
        merge_indptr_offset_bytes=merge_indptr_offset_bytes,
        o_indptr_offset_bytes=o_indptr_offset_bytes,
        kv_chunk_size_ptr_offset_bytes=kv_chunk_size_ptr_offset_bytes,
        kv_window_start_tokens_offset_bytes=kv_window_start_tokens_offset_bytes,
        total_num_rows_ptr_offset_bytes=total_num_rows_ptr_offset_bytes,
        lse_offset_bytes=lse_offset_bytes,
        tmp_output_offset_bytes=tmp_output_offset_bytes,
        tmp_lse_offset_bytes=tmp_lse_offset_bytes,
        msa_union_blocks_offset_bytes=msa_union_blocks_offset_bytes,
        msa_union_masks_offset_bytes=msa_union_masks_offset_bytes,
        msa_union_counts_offset_bytes=msa_union_counts_offset_bytes,
    )


@dataclass(frozen=True)
class _SM12XPagedPlanMetadataCache:
    request_indices: torch.Tensor
    qo_tile_indices: torch.Tensor
    kv_tile_indices: torch.Tensor
    merge_indptr: torch.Tensor
    o_indptr: torch.Tensor
    block_valid_mask: torch.Tensor
    kv_window_start_tokens: torch.Tensor
    kv_chunk_size: int
    total_q: int


def _make_plan_metadata_cache(
    plan: PagedPlan,
    *,
    device: torch.device,
) -> _SM12XPagedPlanMetadataCache:
    return _SM12XPagedPlanMetadataCache(
        request_indices=_copy_int_metadata(plan.request_indices, device=device),
        qo_tile_indices=_copy_int_metadata(plan.qo_tile_indices, device=device),
        kv_tile_indices=_copy_int_metadata(plan.kv_tile_indices, device=device),
        merge_indptr=_copy_int_metadata(plan.merge_indptr, device=device),
        o_indptr=_copy_int_metadata(plan.o_indptr, device=device),
        block_valid_mask=_copy_int_metadata(plan.block_valid_mask, device=device),
        kv_window_start_tokens=_copy_int_metadata(
            plan.kv_window_start_tokens,
            device=device,
        ),
        kv_chunk_size=int(plan.kv_chunk_size),
        total_q=int(plan.total_q),
    )


@dataclass(kw_only=True)
class SM12XPagedAttentionScratch:
    """Paged-attention scratch views over caller-owned storage."""

    shared_scratch: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    mode: Literal["decode", "extend", "verify"]
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    page_size: int
    max_total_q: int
    max_batch: int
    max_page_table_width: int
    max_work_items: int
    max_partial_rows: int
    num_cache_pages: int
    use_cuda_graph: bool = False
    copy_runtime_metadata: bool = True
    fixed_capacity: bool = True
    request_indices: torch.Tensor | None = None
    qo_tile_indices: torch.Tensor | None = None
    kv_tile_indices: torch.Tensor | None = None
    merge_indptr: torch.Tensor | None = None
    o_indptr: torch.Tensor | None = None
    kv_chunk_size_ptr: torch.Tensor | None = None
    kv_window_start_tokens: torch.Tensor | None = None
    total_num_rows_ptr: torch.Tensor | None = None
    block_valid_mask: torch.Tensor | None = None
    page_table: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    lse: torch.Tensor | None = None
    tmp_output: torch.Tensor | None = None
    tmp_lse: torch.Tensor | None = None
    msa_union_blocks: torch.Tensor | None = None
    msa_union_masks: torch.Tensor | None = None
    msa_union_counts: torch.Tensor | None = None
    q2k_indices: torch.Tensor | None = None
    _plan_q: torch.Tensor | None = None
    _plan_output: torch.Tensor | None = None
    _plan_k_cache: torch.Tensor | None = None
    _plan_v_cache: torch.Tensor | None = None
    _plan: PagedPlan | None = None
    _plan_metadata_cache: _SM12XPagedPlanMetadataCache | None = None
    _planner_budget: PagedPlanBudget | None = None
    _decode_graph_chunk_pages_lut: torch.Tensor | None = None
    _decode_graph_max_chunks_per_req: int | None = None
    _use_regular_decode_graph_replay: bool = False
    _decode_graph_metadata_captured_in_graph: bool = False
    _prefill_graph_max_q_tiles_per_req: int | None = None
    _prefill_graph_max_chunks_per_q_tile: int | None = None
    _prefill_graph_max_q_rows_per_req: int | None = None
    msa_block_sparse: bool = False
    msa_union_tile: bool = False
    _q2k_indices_data_ptr: int | None = None
    _live_plane_tma_desc_cache: dict[
        tuple[int, int, tuple[int, ...], tuple[int, ...], int, int],
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor],
    ] = field(default_factory=dict)

    @property
    def prepared(self) -> bool:
        return self._plan is not None

    @property
    def plan(self) -> PagedPlan:
        if self._plan is None:
            raise RuntimeError("paged scratch has not been prepared")
        return self._plan

    @property
    def active_total_q(self) -> int:
        return self.plan.total_q

    @property
    def total_q_capacity(self) -> int:
        if self._plan_q is None:
            raise RuntimeError("paged scratch planning contract is not initialized")
        return int(self._plan_q.shape[0])

    @property
    def planner_budget(self) -> PagedPlanBudget | None:
        return self._planner_budget

    def current_lse_view(self) -> torch.Tensor:
        if self.lse is None:
            raise RuntimeError("paged scratch has not been prepared")
        return self.lse[:, : self.active_total_q].transpose(0, 1)

    def bind(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        page_table: torch.Tensor | None = None,
        cache_seqlens: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        fixed_split_size: int | None = None,
        disable_split_kv: bool = False,
        window_left: int = -1,
        active_total_q: int | None = None,
        q2k_indices: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
        attention_sink_bias: torch.Tensor | None = None,
        relative_attention_bias: torch.Tensor | None = None,
    ) -> "SM12XPagedAttentionBinding":
        return build_paged_attention_binding(
            scratch=self,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
            window_left=window_left,
            active_total_q=active_total_q,
            q2k_indices=q2k_indices,
            k_descale=k_descale,
            v_descale=v_descale,
            attention_sink_bias=attention_sink_bias,
            relative_attention_bias=relative_attention_bias,
        )

    @torch._dynamo.disable
    def prepare(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        *,
        fixed_split_size: int | None = None,
        disable_split_kv: bool = False,
        window_left: int = -1,
        active_total_q: int | None = None,
    ) -> "SM12XPagedAttentionScratch":
        with record_function(f"paged_scratch.prepare.{self.mode}"):
            if window_left < -1:
                raise ValueError(
                    "window_left must be -1 for full attention or a "
                    "non-negative token count"
                )
            if self.use_cuda_graph and torch.cuda.is_current_stream_capturing():
                if self._plan is None:
                    raise RuntimeError(
                        "graph-mode paged scratch must be prepared before "
                        "CUDA graph capture"
                    )
                if int(window_left) != int(self._plan.window_left):
                    raise ValueError(
                        "captured paged attention graph was prepared with "
                        f"window_left={self._plan.window_left}, got "
                        f"window_left={int(window_left)}"
                    )
                if self._plan_metadata_cache is None:
                    raise RuntimeError(
                        "graph-mode paged scratch is missing cached plan metadata"
                    )
                self._bind_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
                self._copy_cached_plan_metadata(self._plan_metadata_cache)
                if (
                    self.mode == "decode"
                    and self._decode_graph_chunk_pages_lut is not None
                ):
                    self.update_decode_graph_replay_metadata_from_runtime_cache_seqlens()
                    self._decode_graph_metadata_captured_in_graph = True
                return self

            if (
                self.use_cuda_graph
                and self.mode == "decode"
                and self._decode_graph_chunk_pages_lut is not None
                and self._plan is not None
            ):
                if int(window_left) != int(self._plan.window_left):
                    raise ValueError(
                        "decode graph replay scratch was prepared with "
                        f"window_left={self._plan.window_left}, got "
                        f"window_left={int(window_left)}"
                    )
                self._bind_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
                if self._plan_metadata_cache is None:
                    raise RuntimeError(
                        "graph-mode paged scratch is missing cached plan metadata"
                    )
                self._copy_cached_plan_metadata(self._plan_metadata_cache)
                if not self._decode_graph_metadata_captured_in_graph:
                    self.update_decode_graph_replay_metadata_from_runtime_cache_seqlens()
                    if torch.cuda.is_current_stream_capturing():
                        self._decode_graph_metadata_captured_in_graph = True
                return self

            if active_total_q is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "paged attention scratch prepare() requires active_total_q "
                        "to be supplied before CUDA graph capture; inferring it from "
                        "a device cu_seqlens_q would require an illegal host sync"
                    )
                inferred_mode = infer_paged_mode(cu_seqlens_q)
                active_total_q = int(cu_seqlens_q[-1].item())
            else:
                active_total_q = int(active_total_q)
                inferred_mode = _infer_mode_from_host_total(
                    cu_seqlens_q,
                    active_total_q,
                )
            if (
                inferred_mode != self.mode
                and not (self.mode == "extend" and inferred_mode == "decode")
                and not (self.mode == "verify" and inferred_mode == "extend")
            ):
                raise ValueError(
                    f"scratch mode {self.mode} does not match prepared mode "
                    f"{inferred_mode}"
                )
            self._ensure_plan_contract(active_total_q)
            assert self._plan_q is not None
            assert self._plan_k_cache is not None
            assert self._plan_v_cache is not None
            if active_total_q <= 0 or active_total_q > int(self._plan_q.shape[0]):
                raise ValueError(
                    f"cu_seqlens_q implies total_q={active_total_q}, but "
                    f"scratch capacity is {int(self._plan_q.shape[0])}"
                )

            plan = create_paged_plan(
                self._plan_q[:active_total_q],
                self._plan_k_cache,
                self._plan_v_cache,
                page_table,
                cache_seqlens,
                cu_seqlens_q,
                mode=self.mode,
                fixed_split_size=(
                    -1 if fixed_split_size is None else int(fixed_split_size)
                ),
                disable_split_kv=disable_split_kv,
                window_left=int(window_left),
                enable_cuda_graph=self.use_cuda_graph,
                graph_chunk_policy=self.use_cuda_graph,
                plan_budget=(
                    self.planner_budget
                    if self.mode in ("extend", "verify") and not self.use_cuda_graph
                    else None
                ),
                msa_block_sparse=self.msa_block_sparse,
                msa_union_tile=self.msa_union_tile,
            )
            self._ensure_capacity(plan)
            self._bind_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
            self._plan_metadata_cache = self._copy_plan_metadata(plan)
            self._plan = plan
            return self

    def _ensure_plan_contract(self, active_total_q: int) -> None:
        if (
            self._plan_q is None
            or self._plan_k_cache is None
            or self._plan_v_cache is None
        ):
            raise RuntimeError("paged scratch planning contract is not initialized")
        if active_total_q <= int(self._plan_q.shape[0]):
            return
        raise ValueError(
            "fixed-capacity paged scratch exceeded; construct a larger scratch "
            f"plan for total_q={active_total_q}"
        )

    def _ensure_capacity(self, plan: PagedPlan) -> None:
        work_items_needed = int(plan.new_batch_size)
        block_valid_needed = int(plan.padded_batch_size)
        total_q_needed = int(plan.total_q)
        batch_needed = int(plan.page_table_shape[0])
        page_table_width_needed = int(plan.page_table_shape[1])
        partial_rows_needed = int(plan.total_num_partial_rows) if plan.split_kv else 0

        if work_items_needed > self.max_work_items:
            raise ValueError(
                f"paged plan needs {work_items_needed} work items, scratch "
                f"capacity is {self.max_work_items}"
            )
        if block_valid_needed > self.max_work_items:
            raise ValueError(
                f"paged plan needs {block_valid_needed} block-valid entries, "
                f"scratch capacity is {self.max_work_items}"
            )
        if total_q_needed > self.max_total_q:
            raise ValueError(
                f"paged plan total_q={total_q_needed} exceeds scratch "
                f"capacity {self.max_total_q}"
            )
        if batch_needed > self.max_batch:
            raise ValueError(
                f"paged plan batch={batch_needed} exceeds scratch capacity "
                f"{self.max_batch}"
            )
        if page_table_width_needed > self.max_page_table_width:
            raise ValueError(
                "paged plan page table width="
                f"{page_table_width_needed} exceeds scratch capacity "
                f"{self.max_page_table_width}"
            )
        if partial_rows_needed > self.max_partial_rows:
            raise ValueError(
                f"paged plan needs {partial_rows_needed} partial rows, "
                f"scratch capacity is {self.max_partial_rows}"
            )

    def _bind_runtime_metadata(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> None:
        if self.copy_runtime_metadata:
            self._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
            return
        self._validate_runtime_metadata_reference(
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
        )
        self.page_table = page_table
        self.cache_seqlens = cache_seqlens
        self.cu_seqlens_q = cu_seqlens_q

    def _validate_runtime_metadata_reference(
        self,
        *,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> None:
        if (
            page_table.device != self.device
            or cache_seqlens.device != self.device
            or cu_seqlens_q.device != self.device
        ):
            raise ValueError("paged scratch metadata references must stay on device")
        if (
            page_table.dtype != torch.int32
            or cache_seqlens.dtype != torch.int32
            or cu_seqlens_q.dtype != torch.int32
        ):
            raise TypeError("paged scratch metadata references must be int32 tensors")
        if not (
            page_table.is_contiguous()
            and cache_seqlens.is_contiguous()
            and cu_seqlens_q.is_contiguous()
        ):
            raise ValueError("paged scratch metadata references must be contiguous")
        if page_table.ndim != 2:
            raise ValueError("page_table must be rank-2")
        if cache_seqlens.ndim != 1 or cu_seqlens_q.ndim != 1:
            raise ValueError("cache_seqlens and cu_seqlens_q must be rank-1")
        if int(page_table.shape[0]) > self.max_batch:
            raise ValueError(
                f"page_table batch={int(page_table.shape[0])} exceeds scratch "
                f"capacity {self.max_batch}"
            )
        if int(page_table.shape[1]) > self.max_page_table_width:
            raise ValueError(
                f"page_table width={int(page_table.shape[1])} exceeds scratch "
                f"capacity {self.max_page_table_width}"
            )
        if int(cache_seqlens.shape[0]) > self.max_batch:
            raise ValueError(
                f"cache_seqlens batch={int(cache_seqlens.shape[0])} exceeds "
                f"scratch capacity {self.max_batch}"
            )
        if int(cu_seqlens_q.shape[0]) > self.max_batch + 1:
            raise ValueError(
                f"cu_seqlens_q length={int(cu_seqlens_q.shape[0])} exceeds "
                f"scratch capacity {self.max_batch + 1}"
            )

    def _copy_runtime_metadata(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> None:
        assert self.page_table is not None
        assert self.cache_seqlens is not None
        assert self.cu_seqlens_q is not None

        if (
            page_table.dtype == torch.int32
            and cache_seqlens.dtype == torch.int32
            and cu_seqlens_q.dtype == torch.int32
            and int(self.page_table.data_ptr()) == int(page_table.data_ptr())
            and int(self.cache_seqlens.data_ptr()) == int(cache_seqlens.data_ptr())
            and int(self.cu_seqlens_q.data_ptr()) == int(cu_seqlens_q.data_ptr())
        ):
            return

        page_table_i32 = (
            page_table
            if page_table.dtype == torch.int32
            else page_table.to(torch.int32)
        )
        cache_seqlens_i32 = (
            cache_seqlens
            if cache_seqlens.dtype == torch.int32
            else cache_seqlens.to(torch.int32)
        )
        cu_seqlens_q_i32 = (
            cu_seqlens_q
            if cu_seqlens_q.dtype == torch.int32
            else cu_seqlens_q.to(torch.int32)
        )

        self.page_table[: page_table_i32.shape[0], : page_table_i32.shape[1]].copy_(
            page_table_i32
        )
        self.cache_seqlens[: cache_seqlens_i32.shape[0]].copy_(cache_seqlens_i32)
        self.cu_seqlens_q[: cu_seqlens_q_i32.shape[0]].copy_(cu_seqlens_q_i32)

    def _copy_cached_plan_metadata(
        self,
        cache: _SM12XPagedPlanMetadataCache,
    ) -> None:
        assert self.request_indices is not None
        assert self.qo_tile_indices is not None
        assert self.kv_tile_indices is not None
        assert self.merge_indptr is not None
        assert self.o_indptr is not None
        assert self.kv_chunk_size_ptr is not None
        assert self.kv_window_start_tokens is not None
        assert self.total_num_rows_ptr is not None
        assert self.block_valid_mask is not None

        if self._decode_graph_chunk_pages_lut is None:
            self._use_regular_decode_graph_replay = False
        self._prefill_graph_max_q_tiles_per_req = None
        self._prefill_graph_max_chunks_per_q_tile = None
        self._prefill_graph_max_q_rows_per_req = None

        self.request_indices.zero_()
        self.qo_tile_indices.zero_()
        self.kv_tile_indices.zero_()
        self.block_valid_mask.zero_()
        self.kv_window_start_tokens.zero_()
        self.request_indices[: cache.request_indices.shape[0]].copy_(
            cache.request_indices
        )
        self.qo_tile_indices[: cache.qo_tile_indices.shape[0]].copy_(
            cache.qo_tile_indices
        )
        self.kv_tile_indices[: cache.kv_tile_indices.shape[0]].copy_(
            cache.kv_tile_indices
        )
        self.merge_indptr[: cache.merge_indptr.shape[0]].copy_(cache.merge_indptr)
        self.o_indptr[: cache.o_indptr.shape[0]].copy_(cache.o_indptr)
        self.block_valid_mask[: cache.block_valid_mask.shape[0]].copy_(
            cache.block_valid_mask
        )
        self.kv_window_start_tokens[: cache.kv_window_start_tokens.shape[0]].copy_(
            cache.kv_window_start_tokens
        )
        self.kv_chunk_size_ptr.fill_(int(cache.kv_chunk_size))
        self.total_num_rows_ptr.fill_(int(cache.total_q))

    def _copy_plan_metadata(
        self,
        plan: PagedPlan,
    ) -> _SM12XPagedPlanMetadataCache:
        cache = _make_plan_metadata_cache(plan, device=self.device)
        self._copy_cached_plan_metadata(cache)
        return cache

    def _validate_decode_graph_replay_capacity(self, *, batch: int) -> None:
        if self._decode_graph_max_chunks_per_req is None:
            raise RuntimeError("decode graph replay policy has not been prepared")
        if self.request_indices is None:
            raise RuntimeError("decode graph scratch is missing request indices")
        if batch <= 0:
            raise ValueError("decode graph replay requires bs > 0")
        work_items_capacity = int(self.request_indices.shape[0])
        if work_items_capacity % batch != 0:
            raise RuntimeError(
                "decode graph scratch request_indices shape is incompatible "
                "with the batch bucket"
            )
        max_chunks_per_req = work_items_capacity // batch
        if max_chunks_per_req <= 0:
            raise RuntimeError(
                "decode graph scratch must allocate at least one chunk per request"
            )
        if self._decode_graph_max_chunks_per_req > max_chunks_per_req:
            raise RuntimeError(
                "decode graph scratch capacity is too small for the current "
                "chunking policy"
            )

    def _window_page_span_from_plan(self, plan: PagedPlan) -> int:
        window_left = int(plan.window_left)
        if window_left < 0:
            return 0
        return max(
            (window_left + self.page_size + self.page_size - 1) // self.page_size,
            1,
        )

    def update_decode_graph_replay_metadata_from_runtime_cache_seqlens(
        self,
    ) -> "SM12XPagedAttentionScratch":
        if not self.use_cuda_graph:
            raise RuntimeError(
                "update_decode_graph_replay_metadata_from_runtime_cache_seqlens "
                "is only valid for graph-mode paged scratch"
            )
        if self.mode != "decode":
            raise RuntimeError(
                "update_decode_graph_replay_metadata_from_runtime_cache_seqlens "
                "is only valid for decode scratch"
            )
        if self._decode_graph_chunk_pages_lut is None:
            raise RuntimeError("decode graph replay policy has not been prepared")
        if self.cache_seqlens is None:
            raise RuntimeError("decode graph scratch is missing cache_seqlens")
        if self.request_indices is None:
            raise RuntimeError("decode graph scratch is missing request indices")
        if self.qo_tile_indices is None or self.kv_tile_indices is None:
            raise RuntimeError("decode graph scratch is missing tile indices")
        if self.merge_indptr is None or self.o_indptr is None:
            raise RuntimeError("decode graph scratch is missing indptr buffers")
        if self.kv_chunk_size_ptr is None:
            raise RuntimeError("decode graph scratch is missing kv_chunk_size_ptr")
        if self.total_num_rows_ptr is None:
            raise RuntimeError("decode graph scratch is missing total_num_rows_ptr")
        if self.cu_seqlens_q is None:
            raise RuntimeError("decode graph scratch is missing cu_seqlens_q")
        if self.block_valid_mask is None:
            raise RuntimeError("decode graph scratch is missing block_valid_mask")
        if self.kv_window_start_tokens is None:
            raise RuntimeError("decode graph scratch is missing kv_window_start_tokens")
        if self._plan is None:
            raise RuntimeError("decode graph scratch has not been prepared")

        uses_compact_work_metadata = self._plan.split_kv and (
            getattr(self._plan, "msa_block_sparse", False)
            or not self._use_regular_decode_graph_replay
        )
        if uses_compact_work_metadata:
            self._validate_decode_graph_replay_capacity(
                batch=int(self.cache_seqlens.shape[0])
            )
        window_page_span = self._window_page_span_from_plan(self._plan)
        if not self._plan.split_kv:
            if int(self._plan.window_left) < 0:
                return self
            from flashinfer.experimental.sm12x.attention.paged.graph_replay import (
                update_decode_graph_window_start_tokens,
            )

            update_decode_graph_window_start_tokens(
                cache_seqlens=self.cache_seqlens,
                kv_window_start_tokens=self.kv_window_start_tokens,
                page_size=self.page_size,
                window_left=int(self._plan.window_left),
            )
        elif getattr(self._plan, "msa_block_sparse", False):
            from flashinfer.experimental.sm12x.attention.paged.graph_replay import (
                update_msa_decode_graph_chunk_metadata,
            )

            update_msa_decode_graph_chunk_metadata(
                cache_seqlens=self.cache_seqlens,
                request_indices=self.request_indices,
                qo_tile_indices=self.qo_tile_indices,
                kv_tile_indices=self.kv_tile_indices,
                merge_indptr=self.merge_indptr,
                o_indptr=self.o_indptr,
                block_valid_mask=self.block_valid_mask,
                kv_chunk_size_ptr=self.kv_chunk_size_ptr,
                kv_window_start_tokens=self.kv_window_start_tokens,
                kv_chunk_size=int(self._plan.kv_chunk_size),
                page_size=self.page_size,
            )
            batch = int(self.cache_seqlens.shape[0])
            if int(self.cu_seqlens_q.shape[0]) < batch + 1:
                raise RuntimeError(
                    "decode graph cu_seqlens_q is smaller than the graph batch"
                )
            self.total_num_rows_ptr[:1].copy_(self.cu_seqlens_q[batch : batch + 1])
        elif self._use_regular_decode_graph_replay:
            from flashinfer.experimental.sm12x.attention.paged.graph_replay import (
                update_regular_decode_graph_chunk_metadata_from_lut,
            )

            update_regular_decode_graph_chunk_metadata_from_lut(
                cache_seqlens=self.cache_seqlens,
                merge_indptr=self.merge_indptr,
                o_indptr=self.o_indptr,
                kv_chunk_size_ptr=self.kv_chunk_size_ptr,
                kv_window_start_tokens=self.kv_window_start_tokens,
                decode_chunk_pages_lut=self._decode_graph_chunk_pages_lut,
                page_size=self.page_size,
                window_page_span=window_page_span,
                window_left=int(self._plan.window_left),
            )
        else:
            from flashinfer.experimental.sm12x.attention.paged.graph_replay import (
                update_decode_graph_chunk_metadata,
            )

            update_decode_graph_chunk_metadata(
                cache_seqlens=self.cache_seqlens,
                request_indices=self.request_indices,
                qo_tile_indices=self.qo_tile_indices,
                kv_tile_indices=self.kv_tile_indices,
                merge_indptr=self.merge_indptr,
                o_indptr=self.o_indptr,
                block_valid_mask=self.block_valid_mask,
                kv_chunk_size_ptr=self.kv_chunk_size_ptr,
                kv_window_start_tokens=self.kv_window_start_tokens,
                decode_chunk_pages_lut=self._decode_graph_chunk_pages_lut,
                page_size=self.page_size,
                window_page_span=window_page_span,
                window_left=int(self._plan.window_left),
            )
        if (
            not getattr(self._plan, "msa_block_sparse", False)
            and not torch.cuda.is_current_stream_capturing()
        ):
            self.total_num_rows_ptr[0] = int(self._plan.total_q)
        return self

    def _validate_static_shapes(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> None:
        if (
            q.device != self.device
            or k_cache.device != self.device
            or v_cache.device != self.device
        ):
            raise ValueError("paged scratch inputs must stay on the scratch device")
        if q.dtype != self.dtype:
            raise TypeError(
                f"paged scratch expects q dtype {self.dtype}, got {q.dtype}"
            )
        if k_cache.dtype != self.kv_dtype or v_cache.dtype != self.kv_dtype:
            raise TypeError(
                "paged scratch expects kv dtype "
                f"{self.kv_dtype}, got {k_cache.dtype}/{v_cache.dtype}"
            )
        if tuple(q.shape[1:]) != (self.num_q_heads, self.head_dim_qk):
            raise ValueError(
                "q shape does not match the scratch contract: expected "
                f"(*, {self.num_q_heads}, {self.head_dim_qk}), got {tuple(q.shape)}"
            )
        if (
            int(k_cache.shape[1]) != self.page_size
            or int(v_cache.shape[1]) != self.page_size
        ):
            raise ValueError(f"paged scratch expects page_size={self.page_size}")
        if (
            int(k_cache.shape[2]) != self.num_kv_heads
            or int(v_cache.shape[2]) != self.num_kv_heads
        ):
            raise ValueError("kv head count does not match the scratch contract")
        if int(k_cache.shape[3]) != self.head_dim_qk:
            raise ValueError("k_cache head_dim does not match the scratch contract")
        if int(v_cache.shape[3]) != self.head_dim_vo:
            raise ValueError("v_cache head_dim does not match the scratch contract")

    def _validate_q2k_indices_reference(
        self,
        q2k_indices: torch.Tensor | None,
    ) -> None:
        if not self.msa_block_sparse:
            if q2k_indices is not None:
                raise ValueError(
                    "q2k_indices can only be bound when scratch caps set msa_block_sparse=True"
                )
            self.q2k_indices = None
            return
        if os.environ.get("FLASHINFER_EXP_SM12X_PAGED_MSA") == "0":
            raise RuntimeError(
                "FLASHINFER_EXP_SM12X_PAGED_MSA=0 disables MSA block-sparse paged attention"
            )
        if q2k_indices is None:
            raise ValueError("MSA block-sparse paged attention requires q2k_indices")
        if q2k_indices.device != self.device:
            raise ValueError("q2k_indices must stay on the scratch device")
        if q2k_indices.dtype != torch.int32:
            raise TypeError("q2k_indices must be a torch.int32 tensor")
        if q2k_indices.ndim != 3:
            raise ValueError(
                "q2k_indices must have shape [kv_heads, total_q_capacity, 16]"
            )
        if not q2k_indices.is_contiguous():
            raise ValueError("q2k_indices must be contiguous")
        if (
            int(q2k_indices.shape[0]) != self.num_kv_heads
            or int(q2k_indices.shape[2]) != 16
        ):
            raise ValueError(
                "q2k_indices must have shape "
                f"({self.num_kv_heads}, >=total_q_capacity, 16), got {tuple(q2k_indices.shape)}"
            )
        if int(q2k_indices.shape[1]) < self.total_q_capacity:
            raise ValueError(
                "q2k_indices total_q_capacity is smaller than scratch total_q_capacity"
            )
        data_ptr = int(q2k_indices.data_ptr())
        if self._q2k_indices_data_ptr is None:
            self._q2k_indices_data_ptr = data_ptr
        elif self._q2k_indices_data_ptr != data_ptr:
            raise ValueError(
                "MSA q2k_indices data_ptr changed after prepare; graph-safe bindings "
                "must rewrite the captured buffer in place"
            )
        self.q2k_indices = q2k_indices


@dataclass(frozen=True, kw_only=True)
class SM12XPagedAttentionBinding:
    scratch: object
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    output: torch.Tensor
    q2k_indices: torch.Tensor | None = None
    k_descale: torch.Tensor | None = None
    v_descale: torch.Tensor | None = None
    attention_sink_bias: torch.Tensor | None = None
    relative_attention_bias: torch.Tensor | None = None

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.experimental.sm12x.attention.paged._forward import (
            paged_attention_forward,
        )

        return paged_attention_forward(binding=self)


def _validate_optional_tensor_device(
    tensor: torch.Tensor | None,
    *,
    scratch: object,
    name: str,
) -> None:
    if tensor is not None and tensor.device != scratch.device:
        raise ValueError(
            f"{name} device {tensor.device} does not match scratch device "
            f"{scratch.device}"
        )


def _validate_output(
    output: torch.Tensor,
    *,
    scratch: object,
) -> None:
    if output.device != scratch.device:
        raise ValueError(
            f"output device {output.device} does not match scratch device "
            f"{scratch.device}"
        )
    if output.dtype != scratch.dtype:
        raise TypeError(f"output must have dtype {scratch.dtype}, got {output.dtype}")
    if output.ndim != 3:
        raise ValueError(
            f"output must be rank-3 [total_q, heads, head_dim], got "
            f"{tuple(output.shape)}"
        )
    if tuple(output.shape[1:]) != (scratch.num_q_heads, scratch.head_dim_vo):
        raise ValueError(
            "output shape does not match the scratch contract: expected "
            f"(*, {scratch.num_q_heads}, {scratch.head_dim_vo}), got "
            f"{tuple(output.shape)}"
        )


def _metadata_tuple(
    *,
    page_table: torch.Tensor | None,
    cache_seqlens: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    values = (page_table, cache_seqlens, cu_seqlens_q)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "page_table, cache_seqlens, and cu_seqlens_q must be provided together"
        )
    return page_table, cache_seqlens, cu_seqlens_q


def build_paged_attention_binding(
    *,
    scratch: object,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output: torch.Tensor,
    page_table: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    fixed_split_size: int | None = None,
    disable_split_kv: bool = False,
    window_left: int = -1,
    active_total_q: int | None = None,
    q2k_indices: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    attention_sink_bias: torch.Tensor | None = None,
    relative_attention_bias: torch.Tensor | None = None,
) -> SM12XPagedAttentionBinding:
    scratch._validate_static_shapes(q, k_cache, v_cache)
    _validate_output(output, scratch=scratch)
    _validate_optional_tensor_device(k_descale, scratch=scratch, name="k_descale")
    _validate_optional_tensor_device(v_descale, scratch=scratch, name="v_descale")
    _validate_optional_tensor_device(
        attention_sink_bias,
        scratch=scratch,
        name="attention_sink_bias",
    )
    _validate_optional_tensor_device(
        relative_attention_bias,
        scratch=scratch,
        name="relative_attention_bias",
    )

    metadata = _metadata_tuple(
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    if metadata is not None:
        scratch.prepare(
            metadata[0],
            metadata[1],
            metadata[2],
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
            window_left=window_left,
            active_total_q=active_total_q,
        )
    elif not scratch.prepared:
        raise RuntimeError("paged attention binding requires prepared scratch metadata")
    scratch._validate_q2k_indices_reference(q2k_indices)

    return SM12XPagedAttentionBinding(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        q2k_indices=q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
        attention_sink_bias=attention_sink_bias,
        relative_attention_bias=relative_attention_bias,
    )


def _materialize_paged_attention_scratch(
    caps: SM12XPagedAttentionScratchCaps,
    scratch_storage: torch.Tensor,
    layout: _SM12XPagedAttentionScratchLayout,
    *,
    plan: PagedPlan | None,
    plan_q: torch.Tensor,
    plan_output: torch.Tensor,
    plan_k_cache: torch.Tensor,
    plan_v_cache: torch.Tensor,
    planner_budget: PagedPlanBudget,
    plan_metadata_cache: _SM12XPagedPlanMetadataCache | None,
    decode_graph_chunk_pages_lut: torch.Tensor | None,
    decode_graph_max_chunks_per_req: int | None,
    use_regular_decode_graph_replay: bool,
) -> SM12XPagedAttentionScratch:
    request_indices, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.request_indices_offset_bytes,
        shape=(caps.max_work_items,),
        dtype=torch.int32,
    )
    qo_tile_indices, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.qo_tile_indices_offset_bytes,
        shape=(caps.max_work_items,),
        dtype=torch.int32,
    )
    kv_tile_indices, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.kv_tile_indices_offset_bytes,
        shape=(caps.max_work_items,),
        dtype=torch.int32,
    )
    block_valid_mask, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.block_valid_mask_offset_bytes,
        shape=(caps.max_work_items,),
        dtype=torch.int32,
    )
    page_table = None
    if layout.page_table_offset_bytes is not None:
        page_table, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.page_table_offset_bytes,
            shape=(caps.max_batch, caps.max_page_table_width),
            dtype=torch.int32,
        )
    cache_seqlens = None
    if layout.cache_seqlens_offset_bytes is not None:
        cache_seqlens, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.cache_seqlens_offset_bytes,
            shape=(caps.max_batch,),
            dtype=torch.int32,
        )
    cu_seqlens_q = None
    if layout.cu_seqlens_q_offset_bytes is not None:
        cu_seqlens_q, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.cu_seqlens_q_offset_bytes,
            shape=(caps.max_batch + 1,),
            dtype=torch.int32,
        )
    merge_indptr, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.merge_indptr_offset_bytes,
        shape=(caps.max_total_q + 1,),
        dtype=torch.int32,
    )
    o_indptr, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.o_indptr_offset_bytes,
        shape=(caps.max_batch + 1,),
        dtype=torch.int32,
    )
    kv_chunk_size_ptr, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.kv_chunk_size_ptr_offset_bytes,
        shape=(1,),
        dtype=torch.int32,
    )
    kv_window_start_tokens, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.kv_window_start_tokens_offset_bytes,
        shape=(caps.max_batch,),
        dtype=torch.int32,
    )
    total_num_rows_ptr, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.total_num_rows_ptr_offset_bytes,
        shape=(1,),
        dtype=torch.int32,
    )
    lse, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.lse_offset_bytes,
        shape=_paged_lse_storage_shape(caps.max_total_q, caps.num_q_heads),
        dtype=torch.float32,
    )
    tmp_output = None
    tmp_lse = None
    if caps.max_partial_rows > 0:
        tmp_output, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.tmp_output_offset_bytes,
            shape=(caps.max_partial_rows, caps.num_q_heads, caps.head_dim_vo),
            dtype=caps.dtype,
        )
        tmp_lse, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.tmp_lse_offset_bytes,
            shape=(caps.max_partial_rows, caps.num_q_heads),
            dtype=torch.float32,
        )
    msa_union_blocks = None
    msa_union_masks = None
    msa_union_counts = None
    if layout.msa_union_blocks_offset_bytes is not None:
        msa_union_blocks, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_union_blocks_offset_bytes,
            shape=(caps.max_work_items, caps.num_kv_heads, 128),
            dtype=torch.int32,
        )
    if layout.msa_union_masks_offset_bytes is not None:
        msa_union_masks, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_union_masks_offset_bytes,
            shape=(caps.max_work_items, caps.num_kv_heads, 128),
            dtype=torch.int32,
        )
    if layout.msa_union_counts_offset_bytes is not None:
        msa_union_counts, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.msa_union_counts_offset_bytes,
            shape=(caps.max_work_items, caps.num_kv_heads),
            dtype=torch.int32,
        )

    return SM12XPagedAttentionScratch(
        shared_scratch=scratch_storage,
        device=caps.device,
        dtype=caps.dtype,
        kv_dtype=caps.kv_dtype,
        mode=caps.mode,
        num_q_heads=caps.num_q_heads,
        num_kv_heads=caps.num_kv_heads,
        head_dim_qk=caps.head_dim_qk,
        head_dim_vo=caps.head_dim_vo,
        page_size=caps.page_size,
        max_total_q=caps.max_total_q,
        max_batch=caps.max_batch,
        max_page_table_width=caps.max_page_table_width,
        max_work_items=caps.max_work_items,
        max_partial_rows=caps.max_partial_rows,
        num_cache_pages=caps.num_cache_pages,
        use_cuda_graph=caps.use_cuda_graph,
        copy_runtime_metadata=caps.copy_runtime_metadata,
        request_indices=request_indices,
        qo_tile_indices=qo_tile_indices,
        kv_tile_indices=kv_tile_indices,
        block_valid_mask=block_valid_mask,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        merge_indptr=merge_indptr,
        o_indptr=o_indptr,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        kv_window_start_tokens=kv_window_start_tokens,
        total_num_rows_ptr=total_num_rows_ptr,
        lse=lse,
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        msa_union_blocks=msa_union_blocks,
        msa_union_masks=msa_union_masks,
        msa_union_counts=msa_union_counts,
        msa_block_sparse=caps.msa_block_sparse,
        msa_union_tile=bool(caps.msa_union_tile),
        _plan_q=plan_q,
        _plan_output=plan_output,
        _plan_k_cache=plan_k_cache,
        _plan_v_cache=plan_v_cache,
        _plan=plan,
        _plan_metadata_cache=plan_metadata_cache,
        _planner_budget=planner_budget,
        _decode_graph_chunk_pages_lut=decode_graph_chunk_pages_lut,
        _decode_graph_max_chunks_per_req=decode_graph_max_chunks_per_req,
        _use_regular_decode_graph_replay=use_regular_decode_graph_replay,
    )


@dataclass
class SM12XPagedAttentionScratchPlan:
    caps: SM12XPagedAttentionScratchCaps
    layout: _SM12XPagedAttentionScratchLayout
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    _plan_q: torch.Tensor
    _plan_output: torch.Tensor
    _plan_k_cache: torch.Tensor
    _plan_v_cache: torch.Tensor
    _planner_budget: PagedPlanBudget
    _plan: PagedPlan | None = None
    _plan_metadata_cache: _SM12XPagedPlanMetadataCache | None = None
    _decode_graph_chunk_pages_lut: torch.Tensor | None = None
    _decode_graph_max_chunks_per_req: int | None = None
    _use_regular_decode_graph_replay: bool = False
    _q2k_indices_data_ptr: int | None = None

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    @property
    def prepared(self) -> bool:
        return self._plan is not None

    @property
    def plan(self) -> PagedPlan:
        if self._plan is None:
            raise RuntimeError("paged scratch plan has not been prepared")
        return self._plan

    @property
    def total_q_capacity(self) -> int:
        return int(self._plan_q.shape[0])

    @staticmethod
    def _plan_has_regular_decode_graph_grid(plan: PagedPlan) -> bool:
        batch = int(plan.page_table_shape[0])
        if (
            batch <= 0
            or plan.mode != "decode"
            or not plan.enable_cuda_graph
            or not plan.split_kv
        ):
            return False
        if int(plan.total_q) != batch:
            return False
        work_items = len(plan.request_indices)
        if work_items <= 0 or work_items % batch != 0:
            return False
        max_chunks_per_req = work_items // batch
        for req_idx in range(batch):
            base = req_idx * max_chunks_per_req
            for chunk_idx in range(max_chunks_per_req):
                work_idx = base + chunk_idx
                if plan.request_indices[work_idx] != req_idx:
                    return False
                if plan.qo_tile_indices[work_idx] != 0:
                    return False
                if plan.kv_tile_indices[work_idx] != chunk_idx:
                    return False
        return True

    def _ensure_capacity(self, plan: PagedPlan) -> None:
        work_items_needed = int(plan.new_batch_size)
        block_valid_needed = int(plan.padded_batch_size)
        total_q_needed = int(plan.total_q)
        batch_needed = int(plan.page_table_shape[0])
        page_table_width_needed = int(plan.page_table_shape[1])
        partial_rows_needed = int(plan.total_num_partial_rows) if plan.split_kv else 0
        if work_items_needed > self.caps.max_work_items:
            raise ValueError(
                f"paged plan needs {work_items_needed} work items, scratch "
                f"capacity is {self.caps.max_work_items}"
            )
        if block_valid_needed > self.caps.max_work_items:
            raise ValueError(
                f"paged plan needs {block_valid_needed} block-valid entries, "
                f"scratch capacity is {self.caps.max_work_items}"
            )
        if total_q_needed > self.caps.max_total_q:
            raise ValueError(
                f"paged plan total_q={total_q_needed} exceeds scratch "
                f"capacity {self.caps.max_total_q}"
            )
        if batch_needed > self.caps.max_batch:
            raise ValueError(
                f"paged plan batch={batch_needed} exceeds scratch capacity "
                f"{self.caps.max_batch}"
            )
        if page_table_width_needed > self.caps.max_page_table_width:
            raise ValueError(
                f"paged plan page table width={page_table_width_needed} exceeds "
                f"scratch capacity {self.caps.max_page_table_width}"
            )
        if partial_rows_needed > self.caps.max_partial_rows:
            raise ValueError(
                f"paged plan needs {partial_rows_needed} partial rows, scratch "
                f"capacity is {self.caps.max_partial_rows}"
            )

    @staticmethod
    def _build_capacity_cu_seqlens_q(
        *,
        batch: int,
        total_q_capacity: int,
        device: torch.device,
    ) -> torch.Tensor:
        del total_q_capacity
        return torch.arange(0, batch + 1, dtype=torch.int32, device=device)

    def prepare_graph_replay_state(
        self,
        *,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        active_total_q: int | None = None,
        fixed_split_size: int = -1,
        disable_split_kv: bool = False,
        window_left: int = -1,
    ) -> "SM12XPagedAttentionScratchPlan":
        """Prepare non-decode graph metadata before CUDA capture.

        A scratch plan is rematerialized on every bind.  Installing the plan
        and its device metadata cache on this caller-owned object ensures that
        the first bind performed during capture starts from prepared state.
        """
        if not self.caps.use_cuda_graph:
            raise RuntimeError(
                "prepare_graph_replay_state is only valid for graph-mode "
                "paged scratch plans"
            )
        if self.caps.mode == "decode":
            raise RuntimeError("decode plans require prepare_decode_graph_replay_state")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "prepare_graph_replay_state must be called before CUDA graph capture"
            )
        if window_left < -1:
            raise ValueError(
                "window_left must be -1 for full attention or a non-negative token count"
            )
        if active_total_q is None:
            active_total_q = int(cu_seqlens_q[-1].item())
        else:
            active_total_q = int(active_total_q)
        inferred_mode = _infer_mode_from_host_total(cu_seqlens_q, active_total_q)
        if inferred_mode != self.caps.mode and not (
            self.caps.mode == "verify" and inferred_mode == "extend"
        ):
            raise ValueError(
                f"scratch mode {self.caps.mode} does not match prepared mode "
                f"{inferred_mode}"
            )
        if active_total_q <= 0 or active_total_q > int(self._plan_q.shape[0]):
            raise ValueError(
                f"active_total_q={active_total_q} exceeds scratch capacity "
                f"{int(self._plan_q.shape[0])}"
            )

        plan = create_paged_plan(
            self._plan_q[:active_total_q],
            self._plan_k_cache,
            self._plan_v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            mode=self.caps.mode,
            fixed_split_size=int(fixed_split_size),
            disable_split_kv=bool(disable_split_kv),
            window_left=int(window_left),
            enable_cuda_graph=True,
            graph_chunk_policy=True,
            plan_budget=None,
            msa_block_sparse=self.caps.msa_block_sparse,
            msa_union_tile=bool(self.caps.msa_union_tile),
        )
        self._ensure_capacity(plan)
        self._plan = plan
        self._plan_metadata_cache = _make_plan_metadata_cache(
            plan,
            device=self.caps.device,
        )
        return self

    def prepare_decode_graph_replay_state(
        self,
        *,
        batch: int,
        max_page_table_width: int,
        total_q_capacity: int | None = None,
        max_cache_page_count: int | None = None,
        fixed_split_size: int = -1,
        window_left: int = -1,
        force_split_kv: bool = False,
    ) -> "SM12XPagedAttentionScratchPlan":
        if not self.caps.use_cuda_graph:
            raise RuntimeError(
                "prepare_decode_graph_replay_state is only valid for graph-mode "
                "paged scratch plans"
            )
        if self.caps.mode != "decode":
            raise RuntimeError(
                "prepare_decode_graph_replay_state is only valid for decode plans"
            )
        if total_q_capacity is None:
            total_q_capacity = int(self.caps.max_total_q)
        else:
            total_q_capacity = int(total_q_capacity)
        if max_cache_page_count is None:
            max_cache_page_count = int(max_page_table_width)
        else:
            max_cache_page_count = int(max_cache_page_count)
        if batch <= 0:
            raise ValueError("batch must be positive")
        if total_q_capacity <= 0:
            raise ValueError("total_q_capacity must be positive")
        if max_page_table_width <= 0:
            raise ValueError("max_page_table_width must be positive")
        if max_cache_page_count <= 0:
            raise ValueError("max_cache_page_count must be positive")
        if window_left < -1:
            raise ValueError(
                "window_left must be -1 for full attention or a non-negative token count"
            )

        if self.caps.msa_block_sparse:
            if window_left != -1:
                raise ValueError(
                    "MSA block-sparse decode graph replay does not support window_left/SWA"
                )
            if self.caps.page_size not in (64, 128):
                raise ValueError(
                    "MSA block-sparse decode graph replay requires page_size=64 or page_size=128"
                )
            if self.caps.head_dim_qk != 128 or self.caps.head_dim_vo != 128:
                raise ValueError(
                    "MSA block-sparse decode graph replay requires head_dim_qk=head_dim_vo=128"
                )
            if self.caps.num_q_heads // self.caps.num_kv_heads != 16:
                raise ValueError(
                    "MSA block-sparse decode graph replay requires gqa_group_size=16"
                )
            max_page_ids = torch.arange(
                max_page_table_width, dtype=torch.int32, device=self.caps.device
            )
            max_page_table = (
                (max_page_ids % self.caps.num_cache_pages)
                .unsqueeze(0)
                .expand(batch, -1)
                .contiguous()
            )
            max_cache_seqlens = torch.full(
                (batch,),
                int(max_cache_page_count) * self.caps.page_size,
                dtype=torch.int32,
                device=self.caps.device,
            )
            max_cu_seqlens_q = self._build_capacity_cu_seqlens_q(
                batch=batch,
                total_q_capacity=total_q_capacity,
                device=self.caps.device,
            )
            plan = create_paged_plan(
                self._plan_q[:batch],
                self._plan_k_cache,
                self._plan_v_cache,
                max_page_table,
                max_cache_seqlens,
                max_cu_seqlens_q,
                mode="decode",
                fixed_split_size=int(fixed_split_size),
                force_split_kv=True,
                window_left=-1,
                enable_cuda_graph=True,
                graph_chunk_policy=False,
                max_batch_size_if_split=batch * 32,
                msa_block_sparse=True,
                plan_budget=None,
            )
            self._ensure_capacity(plan)
            self._plan = plan
            self._plan_metadata_cache = _make_plan_metadata_cache(
                plan, device=self.caps.device
            )
            self._decode_graph_chunk_pages_lut = torch.empty(
                (1,), dtype=torch.int32, device=self.caps.device
            )
            self._decode_graph_max_chunks_per_req = max(
                int(plan.o_indptr[idx + 1] - plan.o_indptr[idx]) for idx in range(batch)
            )
            self._use_regular_decode_graph_replay = False
            return self

        from flashinfer.experimental.sm12x.attention.paged.graph_replay import (
            make_decode_chunk_pages_lut_tensor,
            summarize_decode_chunk_pages_lut,
        )

        max_effective_kv_pages = int(max_cache_page_count)
        if window_left >= 0:
            max_effective_kv_pages = min(
                max_effective_kv_pages,
                max(
                    1,
                    (int(window_left) + self.caps.page_size + self.caps.page_size - 1)
                    // self.caps.page_size,
                ),
            )
        graph_ctas_per_sm = resolve_decode_graph_ctas_per_sm(
            kv_dtype=self.caps.kv_dtype,
            batch=batch,
            page_size=self.caps.page_size,
            head_dim_qk=self.caps.head_dim_qk,
            head_dim_vo=self.caps.head_dim_vo,
            gqa_group_size=self.caps.num_q_heads // self.caps.num_kv_heads,
        )
        max_chunks_per_req_budget = decode_graph_max_chunks_per_request_budget(
            device=self.caps.device,
            num_kv_heads=self.caps.num_kv_heads,
            batch=batch,
            graph_ctas_per_sm=graph_ctas_per_sm,
        )
        decode_chunk_pages_lut = build_decode_chunk_pages_lut(
            q_dtype=self.caps.dtype,
            kv_dtype=self.caps.kv_dtype,
            batch=batch,
            page_size=self.caps.page_size,
            head_dim_qk=self.caps.head_dim_qk,
            head_dim_vo=self.caps.head_dim_vo,
            gqa_group_size=self.caps.num_q_heads // self.caps.num_kv_heads,
            max_effective_kv_pages=max_effective_kv_pages,
            max_chunks_per_req=max_chunks_per_req_budget,
        )
        worst_page_count, max_chunks_per_req = summarize_decode_chunk_pages_lut(
            decode_chunk_pages_lut
        )
        capacity_cache_seqlen = int(worst_page_count) * self.caps.page_size
        if window_left >= 0:
            capacity_cache_seqlen = int(max_cache_page_count) * self.caps.page_size - 1
        max_cache_seqlens = torch.full(
            (batch,),
            int(capacity_cache_seqlen),
            dtype=torch.int32,
            device=self.caps.device,
        )
        max_page_ids = torch.arange(
            max_page_table_width, dtype=torch.int32, device=self.caps.device
        )
        max_page_table = (
            (max_page_ids % self.caps.num_cache_pages)
            .unsqueeze(0)
            .expand(batch, -1)
            .contiguous()
        )
        max_cu_seqlens_q = self._build_capacity_cu_seqlens_q(
            batch=batch,
            total_q_capacity=total_q_capacity,
            device=self.caps.device,
        )
        plan = create_paged_plan(
            self._plan_q[:batch],
            self._plan_k_cache,
            self._plan_v_cache,
            max_page_table,
            max_cache_seqlens,
            max_cu_seqlens_q,
            mode="decode",
            fixed_split_size=-1,
            disable_split_kv=not force_split_kv,
            force_split_kv=force_split_kv,
            window_left=int(window_left),
            enable_cuda_graph=True,
            graph_chunk_policy=True,
            plan_budget=None,
        )
        self._ensure_capacity(plan)
        self._plan = plan
        self._plan_metadata_cache = _make_plan_metadata_cache(
            plan, device=self.caps.device
        )
        self._decode_graph_chunk_pages_lut = make_decode_chunk_pages_lut_tensor(
            decode_chunk_pages_lut,
            device=self.caps.device,
        )
        self._decode_graph_max_chunks_per_req = int(max_chunks_per_req)
        self._use_regular_decode_graph_replay = int(plan.gqa_group_size) <= int(
            plan.cta_tile_q
        ) and self._plan_has_regular_decode_graph_grid(plan)
        return self

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        fixed_split_size: int | None = None,
        disable_split_kv: bool = False,
        window_left: int = -1,
        active_total_q: int | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
        attention_sink_bias: torch.Tensor | None = None,
        relative_attention_bias: torch.Tensor | None = None,
        q2k_indices: torch.Tensor | None = None,
    ) -> SM12XPagedAttentionBinding:
        scratch_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="paged attention",
        )
        scratch_views = _materialize_paged_attention_scratch(
            self.caps,
            scratch_storage,
            self.layout,
            plan=self._plan,
            plan_q=self._plan_q,
            plan_output=self._plan_output,
            plan_k_cache=self._plan_k_cache,
            plan_v_cache=self._plan_v_cache,
            planner_budget=self._planner_budget,
            plan_metadata_cache=self._plan_metadata_cache,
            decode_graph_chunk_pages_lut=self._decode_graph_chunk_pages_lut,
            decode_graph_max_chunks_per_req=self._decode_graph_max_chunks_per_req,
            use_regular_decode_graph_replay=self._use_regular_decode_graph_replay,
        )
        scratch_views._q2k_indices_data_ptr = self._q2k_indices_data_ptr
        binding = build_paged_attention_binding(
            scratch=scratch_views,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
            window_left=window_left,
            active_total_q=active_total_q,
            k_descale=k_descale,
            v_descale=v_descale,
            attention_sink_bias=attention_sink_bias,
            relative_attention_bias=relative_attention_bias,
            q2k_indices=q2k_indices,
        )
        self._q2k_indices_data_ptr = scratch_views._q2k_indices_data_ptr
        return binding


def plan_paged_attention_scratch(
    caps: SM12XPagedAttentionScratchCaps,
) -> SM12XPagedAttentionScratchPlan:
    layout = _paged_attention_scratch_layout(caps)
    plan_q = _shape_only_cuda_tensor(
        (caps.max_total_q, caps.num_q_heads, caps.head_dim_qk),
        dtype=caps.dtype,
        device=caps.device,
    )
    plan_output = _shape_only_cuda_tensor(
        (caps.max_total_q, caps.num_q_heads, caps.head_dim_vo),
        dtype=caps.dtype,
        device=caps.device,
    )
    plan_k_cache = _shape_only_cuda_tensor(
        (caps.num_cache_pages, caps.page_size, caps.num_kv_heads, caps.head_dim_qk),
        dtype=caps.kv_dtype,
        device=caps.device,
    )
    plan_v_cache = _shape_only_cuda_tensor(
        (caps.num_cache_pages, caps.page_size, caps.num_kv_heads, caps.head_dim_vo),
        dtype=caps.kv_dtype,
        device=caps.device,
    )
    planner_budget = PagedPlanBudget(
        max_total_q=caps.max_total_q,
        max_batch=caps.max_batch,
        max_page_table_width=caps.max_page_table_width,
        max_work_items=caps.max_work_items,
        max_partial_rows=caps.max_partial_rows,
    )
    return SM12XPagedAttentionScratchPlan(
        caps=caps,
        layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "paged_attention.scratch",
                nbytes=int(layout.nbytes),
                device=caps.device,
            ),
        ),
        _plan_q=plan_q,
        _plan_output=plan_output,
        _plan_k_cache=plan_k_cache,
        _plan_v_cache=plan_v_cache,
        _planner_budget=planner_budget,
    )


__all__ = [
    "SM12XPagedAttentionBinding",
    "SM12XPagedAttentionScratch",
    "SM12XPagedAttentionScratchCaps",
    "SM12XPagedAttentionScratchPlan",
    "build_paged_attention_binding",
    "plan_paged_attention_scratch",
]
