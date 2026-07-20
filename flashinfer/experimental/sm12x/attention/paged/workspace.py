# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/paged/workspace.py @ 3044f545 (2026-07-17) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Dynamic workspace state for the primary paged-attention backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch.profiler import record_function

from .planner import (
    PagedPlan,
    PagedPlanBudget,
    build_decode_chunk_pages_lut,
    create_paged_plan,
    decode_graph_max_chunks_per_request_budget,
    infer_paged_mode,
    resolve_decode_graph_ctas_per_sm,
)

_ARENA_ALIGN_BYTES = 1024


def _paged_lse_storage_shape(total_q: int, num_q_heads: int) -> tuple[int, int]:
    return (num_q_heads, total_q)


def _copy_int_metadata(
    values: tuple[int, ...], *, device: torch.device
) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=device)


def _canonical_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((int(value) + alignment - 1) // alignment) * alignment


def _infer_mode_from_host_total(
    cu_seqlens_q: torch.Tensor, active_total_q: int
) -> Literal["decode", "extend"]:
    batch = max(int(cu_seqlens_q.shape[0]) - 1, 0)
    return "decode" if batch > 0 and int(active_total_q) == batch else "extend"


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


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
    if nbytes == 0:
        return arena.narrow(0, 0, 0).view(dtype).view(shape), offset_bytes
    view_bytes = arena.narrow(0, offset_bytes, nbytes)
    typed_view = view_bytes.view(dtype).view(shape)
    return typed_view, offset_bytes + nbytes


def _shape_only_cuda_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a tiny CUDA tensor view with the requested logical shape.

    The paged planner only inspects shape, dtype, and device for the
    contract-side KV tensors; it never reads their data. Represent the
    contract with a single-element zero-stride view so graph workspaces do
    not allocate full shadow copies of the KV cache.
    """

    if any(dim <= 0 for dim in shape):
        raise ValueError(f"shape must be positive in every dimension, got {shape}")
    base = torch.empty(1, dtype=dtype, device=device)
    return base.as_strided(shape, (0,) * len(shape))


@dataclass(frozen=True, kw_only=True)
class PagedAttentionArenaCaps:
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    max_head_dim_vo: int
    page_size: int
    max_total_q: int
    max_batch: int
    max_page_table_width: int
    max_work_items: int
    max_partial_rows: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _canonical_device(self.device))
        object.__setattr__(self, "num_q_heads", max(int(self.num_q_heads), 1))
        object.__setattr__(self, "num_kv_heads", max(int(self.num_kv_heads), 1))
        object.__setattr__(self, "head_dim_qk", max(int(self.head_dim_qk), 1))
        object.__setattr__(self, "max_head_dim_vo", max(int(self.max_head_dim_vo), 1))
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


@dataclass(frozen=True, kw_only=True)
class PagedAttentionWorkspaceContract:
    mode: Literal["decode", "extend", "verify"]
    max_total_q: int
    max_batch: int
    max_page_table_width: int
    max_work_items: int
    max_partial_rows: int
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    num_cache_pages: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_total_q", max(int(self.max_total_q), 1))
        object.__setattr__(self, "max_batch", max(int(self.max_batch), 1))
        object.__setattr__(
            self,
            "max_page_table_width",
            max(int(self.max_page_table_width), 1),
        )
        object.__setattr__(self, "max_work_items", max(int(self.max_work_items), 0))
        object.__setattr__(self, "max_partial_rows", max(int(self.max_partial_rows), 0))
        object.__setattr__(self, "num_q_heads", max(int(self.num_q_heads), 1))
        object.__setattr__(self, "num_kv_heads", max(int(self.num_kv_heads), 1))
        object.__setattr__(self, "head_dim_qk", max(int(self.head_dim_qk), 1))
        object.__setattr__(self, "head_dim_vo", max(int(self.head_dim_vo), 1))
        object.__setattr__(self, "num_cache_pages", max(int(self.num_cache_pages), 1))


@dataclass(frozen=True, kw_only=True)
class _PagedAttentionArenaLayout:
    arena_nbytes: int
    request_indices_offset_bytes: int
    qo_tile_indices_offset_bytes: int
    kv_tile_indices_offset_bytes: int
    block_valid_mask_offset_bytes: int
    page_table_offset_bytes: int
    cache_seqlens_offset_bytes: int
    cu_seqlens_q_offset_bytes: int
    merge_indptr_offset_bytes: int
    o_indptr_offset_bytes: int
    kv_chunk_size_ptr_offset_bytes: int
    kv_window_start_tokens_offset_bytes: int
    total_num_rows_ptr_offset_bytes: int
    lse_offset_bytes: int
    tmp_output_offset_bytes: int
    tmp_lse_offset_bytes: int


@dataclass(kw_only=True)
class PagedAttentionArena:
    caps: PagedAttentionArenaCaps
    shared_arena: torch.Tensor
    shared_arena_nbytes: int
    request_indices_offset_bytes: int
    qo_tile_indices_offset_bytes: int
    kv_tile_indices_offset_bytes: int
    block_valid_mask_offset_bytes: int
    page_table_offset_bytes: int
    cache_seqlens_offset_bytes: int
    cu_seqlens_q_offset_bytes: int
    merge_indptr_offset_bytes: int
    o_indptr_offset_bytes: int
    kv_chunk_size_ptr_offset_bytes: int
    kv_window_start_tokens_offset_bytes: int
    total_num_rows_ptr_offset_bytes: int
    lse_offset_bytes: int
    tmp_output_offset_bytes: int
    tmp_lse_offset_bytes: int

    @classmethod
    def _layout(cls, caps: PagedAttentionArenaCaps) -> _PagedAttentionArenaLayout:
        offset = 0

        def reserve(shape: tuple[int, ...], dtype: torch.dtype) -> int:
            nonlocal offset
            offset = _align_up(offset, max(_ARENA_ALIGN_BYTES, _dtype_nbytes(dtype)))
            current = offset
            offset += _shape_numel(shape) * _dtype_nbytes(dtype)
            return current

        request_indices_offset_bytes = reserve((caps.max_work_items,), torch.int32)
        qo_tile_indices_offset_bytes = reserve((caps.max_work_items,), torch.int32)
        kv_tile_indices_offset_bytes = reserve((caps.max_work_items,), torch.int32)
        block_valid_mask_offset_bytes = reserve((caps.max_work_items,), torch.int32)
        page_table_offset_bytes = reserve(
            (caps.max_batch, caps.max_page_table_width),
            torch.int32,
        )
        cache_seqlens_offset_bytes = reserve((caps.max_batch,), torch.int32)
        cu_seqlens_q_offset_bytes = reserve((caps.max_batch + 1,), torch.int32)
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
            (caps.max_partial_rows, caps.num_q_heads, caps.max_head_dim_vo),
            caps.dtype,
        )
        tmp_lse_offset_bytes = reserve(
            (caps.max_partial_rows, caps.num_q_heads),
            torch.float32,
        )
        return _PagedAttentionArenaLayout(
            arena_nbytes=max(int(offset), 1),
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
        )

    @classmethod
    def _build(
        cls,
        caps: PagedAttentionArenaCaps,
        *,
        shared_arena: torch.Tensor | None,
    ) -> "PagedAttentionArena":
        layout = cls._layout(caps)
        if shared_arena is None:
            shared_arena = torch.empty(
                layout.arena_nbytes,
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
                f"shared_arena has {shared_arena.numel()} bytes, but paged attention arena requires {layout.arena_nbytes}"
            )
        return cls(
            caps=caps,
            shared_arena=shared_arena,
            shared_arena_nbytes=layout.arena_nbytes,
            request_indices_offset_bytes=layout.request_indices_offset_bytes,
            qo_tile_indices_offset_bytes=layout.qo_tile_indices_offset_bytes,
            kv_tile_indices_offset_bytes=layout.kv_tile_indices_offset_bytes,
            block_valid_mask_offset_bytes=layout.block_valid_mask_offset_bytes,
            page_table_offset_bytes=layout.page_table_offset_bytes,
            cache_seqlens_offset_bytes=layout.cache_seqlens_offset_bytes,
            cu_seqlens_q_offset_bytes=layout.cu_seqlens_q_offset_bytes,
            merge_indptr_offset_bytes=layout.merge_indptr_offset_bytes,
            o_indptr_offset_bytes=layout.o_indptr_offset_bytes,
            kv_chunk_size_ptr_offset_bytes=layout.kv_chunk_size_ptr_offset_bytes,
            kv_window_start_tokens_offset_bytes=layout.kv_window_start_tokens_offset_bytes,
            total_num_rows_ptr_offset_bytes=layout.total_num_rows_ptr_offset_bytes,
            lse_offset_bytes=layout.lse_offset_bytes,
            tmp_output_offset_bytes=layout.tmp_output_offset_bytes,
            tmp_lse_offset_bytes=layout.tmp_lse_offset_bytes,
        )

    @classmethod
    def allocate(cls, caps: PagedAttentionArenaCaps) -> "PagedAttentionArena":
        return cls._build(caps, shared_arena=None)

    @classmethod
    def from_shared_arena(
        cls,
        caps: PagedAttentionArenaCaps,
        shared_arena: torch.Tensor,
    ) -> "PagedAttentionArena":
        return cls._build(caps, shared_arena=shared_arena)

    @classmethod
    def required_nbytes(cls, caps: PagedAttentionArenaCaps) -> int:
        return cls._layout(caps).arena_nbytes

    def _make_workspace_views(
        self,
        contract: PagedAttentionWorkspaceContract,
        *,
        use_cuda_graph: bool = False,
    ) -> "PagedAttentionWorkspace":
        if contract.max_total_q > self.caps.max_total_q:
            raise ValueError(
                f"workspace max_total_q {contract.max_total_q} exceeds arena max_total_q {self.caps.max_total_q}"
            )
        if contract.max_batch > self.caps.max_batch:
            raise ValueError(
                f"workspace max_batch {contract.max_batch} exceeds arena max_batch {self.caps.max_batch}"
            )
        if contract.max_page_table_width > self.caps.max_page_table_width:
            raise ValueError(
                "workspace max_page_table_width "
                f"{contract.max_page_table_width} exceeds arena max_page_table_width {self.caps.max_page_table_width}"
            )
        if contract.max_work_items > self.caps.max_work_items:
            raise ValueError(
                f"workspace max_work_items {contract.max_work_items} exceeds arena max_work_items {self.caps.max_work_items}"
            )
        if contract.max_partial_rows > self.caps.max_partial_rows:
            raise ValueError(
                "workspace max_partial_rows "
                f"{contract.max_partial_rows} exceeds arena max_partial_rows {self.caps.max_partial_rows}"
            )
        if contract.num_q_heads > self.caps.num_q_heads:
            raise ValueError(
                f"workspace num_q_heads {contract.num_q_heads} exceeds arena num_q_heads {self.caps.num_q_heads}"
            )
        if contract.num_kv_heads > self.caps.num_kv_heads:
            raise ValueError(
                f"workspace num_kv_heads {contract.num_kv_heads} exceeds arena num_kv_heads {self.caps.num_kv_heads}"
            )
        if contract.head_dim_qk > self.caps.head_dim_qk:
            raise ValueError(
                f"workspace head_dim_qk {contract.head_dim_qk} exceeds arena head_dim_qk {self.caps.head_dim_qk}"
            )
        if contract.head_dim_vo > self.caps.max_head_dim_vo:
            raise ValueError(
                f"workspace head_dim_vo {contract.head_dim_vo} exceeds arena max_head_dim_vo {self.caps.max_head_dim_vo}"
            )

        plan_q = _shape_only_cuda_tensor(
            (contract.max_total_q, contract.num_q_heads, contract.head_dim_qk),
            dtype=self.caps.dtype,
            device=self.caps.device,
        )
        plan_output = _shape_only_cuda_tensor(
            (contract.max_total_q, contract.num_q_heads, contract.head_dim_vo),
            dtype=self.caps.dtype,
            device=self.caps.device,
        )
        plan_k_cache = _shape_only_cuda_tensor(
            (
                contract.num_cache_pages,
                self.caps.page_size,
                contract.num_kv_heads,
                contract.head_dim_qk,
            ),
            dtype=self.caps.kv_dtype,
            device=self.caps.device,
        )
        plan_v_cache = _shape_only_cuda_tensor(
            (
                contract.num_cache_pages,
                self.caps.page_size,
                contract.num_kv_heads,
                contract.head_dim_vo,
            ),
            dtype=self.caps.kv_dtype,
            device=self.caps.device,
        )
        workspace = PagedAttentionWorkspace(
            arena=self,
            contract=contract,
            mode=contract.mode,
            device=self.caps.device,
            dtype=self.caps.dtype,
            kv_dtype=self.caps.kv_dtype,
            num_q_heads=contract.num_q_heads,
            num_kv_heads=contract.num_kv_heads,
            head_dim_qk=contract.head_dim_qk,
            head_dim_vo=contract.head_dim_vo,
            page_size=self.caps.page_size,
            use_cuda_graph=use_cuda_graph,
            fixed_capacity=True,
            _plan_q=plan_q,
            _plan_output=plan_output,
            _plan_k_cache=plan_k_cache,
            _plan_v_cache=plan_v_cache,
            _planner_budget=PagedPlanBudget(
                max_total_q=contract.max_total_q,
                max_batch=contract.max_batch,
                max_page_table_width=contract.max_page_table_width,
                max_work_items=contract.max_work_items,
                max_partial_rows=contract.max_partial_rows,
            ),
        )
        workspace._allocate_runtime_buffers(
            work_items_capacity=contract.max_work_items,
            block_valid_capacity=contract.max_work_items,
            total_q_capacity=contract.max_total_q,
            batch_capacity=contract.max_batch,
            page_table_width_capacity=contract.max_page_table_width,
            partial_rows_capacity=contract.max_partial_rows,
        )
        return workspace

    def make_workspace(
        self,
        contract: PagedAttentionWorkspaceContract,
        *,
        use_cuda_graph: bool = False,
    ) -> "PagedAttentionWorkspace":
        return self._make_workspace_views(
            contract,
            use_cuda_graph=use_cuda_graph,
        )


@dataclass(kw_only=True)
class PagedAttentionWorkspace:
    arena: PagedAttentionArena | None = None
    contract: PagedAttentionWorkspaceContract | None = None
    mode: Literal["decode", "extend", "verify"]
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    page_size: int = 64
    use_cuda_graph: bool = False
    fixed_capacity: bool = False
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
    _plan_q: torch.Tensor | None = None
    _plan_output: torch.Tensor | None = None
    _plan_k_cache: torch.Tensor | None = None
    _plan_v_cache: torch.Tensor | None = None
    _plan: PagedPlan | None = None
    _planner_budget: PagedPlanBudget | None = None
    shared_arena: torch.Tensor | None = None
    shared_arena_nbytes: int = 0

    # Pre-compiled kernel state (set by compile_paged_kernels in api.py).
    _compiled_forward: object | None = None
    _compiled_merge: object | None = None
    _compiled_forward_kernel: object | None = None
    _compiled_key: tuple | None = None
    _decode_graph_chunk_pages_lut: torch.Tensor | None = None
    _decode_graph_max_chunks_per_req: int | None = None
    _use_regular_decode_graph_replay: bool = False
    _decode_graph_metadata_captured_in_graph: bool = False
    _prefill_graph_max_q_tiles_per_req: int | None = None
    _prefill_graph_max_chunks_per_q_tile: int | None = None
    _prefill_graph_max_q_rows_per_req: int | None = None
    _live_plane_tma_desc_cache: dict[
        tuple[int, int, tuple[int, ...], tuple[int, ...], int, int],
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor],
    ] = field(default_factory=dict)

    @staticmethod
    def eager_extend_work_items_capacity(
        *,
        max_total_q: int,
        num_q_heads: int,
        num_kv_heads: int,
    ) -> int:
        if max_total_q <= 0:
            raise ValueError("max_total_q must be positive")
        if num_q_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("head counts must be positive")
        gqa_group_size = num_q_heads // num_kv_heads
        return max((int(max_total_q) * int(gqa_group_size) + 15) // 16, 1)

    @classmethod
    def for_contract(
        cls,
        *,
        mode: Literal["decode", "extend", "verify"],
        device: torch.device | str,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        max_total_q: int,
        num_cache_pages: int,
        use_cuda_graph: bool = False,
    ) -> PagedAttentionWorkspace:
        device = _canonical_device(device)
        if max_total_q <= 0:
            raise ValueError("max_total_q must be positive")
        if num_cache_pages <= 0:
            raise ValueError("num_cache_pages must be positive")
        plan_q = _shape_only_cuda_tensor(
            (max_total_q, num_q_heads, head_dim_qk),
            dtype=dtype,
            device=device,
        )
        plan_output = _shape_only_cuda_tensor(
            (max_total_q, num_q_heads, head_dim_vo),
            dtype=dtype,
            device=device,
        )
        plan_k_cache = _shape_only_cuda_tensor(
            (num_cache_pages, page_size, num_kv_heads, head_dim_qk),
            dtype=kv_dtype,
            device=device,
        )
        plan_v_cache = _shape_only_cuda_tensor(
            (num_cache_pages, page_size, num_kv_heads, head_dim_vo),
            dtype=kv_dtype,
            device=device,
        )
        return cls(
            mode=mode,
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            page_size=page_size,
            use_cuda_graph=use_cuda_graph,
            _plan_q=plan_q,
            _plan_output=plan_output,
            _plan_k_cache=plan_k_cache,
            _plan_v_cache=plan_v_cache,
        )

    @classmethod
    def for_fixed_capacity(
        cls,
        *,
        mode: Literal["decode", "extend", "verify"],
        device: torch.device | str,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        max_total_q: int,
        max_batch: int,
        max_page_table_width: int,
        max_work_items: int,
        max_partial_rows: int,
        num_cache_pages: int,
        use_cuda_graph: bool = False,
    ) -> PagedAttentionWorkspace:
        device = _canonical_device(device)
        caps = PagedAttentionArenaCaps(
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            max_head_dim_vo=head_dim_vo,
            page_size=page_size,
            max_total_q=max_total_q,
            max_batch=max_batch,
            max_page_table_width=max_page_table_width,
            max_work_items=max_work_items,
            max_partial_rows=max_partial_rows,
        )
        arena = PagedAttentionArena.allocate(caps)
        contract = PagedAttentionWorkspaceContract(
            mode=mode,
            max_total_q=int(max_total_q),
            max_batch=int(max_batch),
            max_page_table_width=int(max_page_table_width),
            max_work_items=int(max_work_items),
            max_partial_rows=int(max_partial_rows),
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            num_cache_pages=num_cache_pages,
        )
        return arena.make_workspace(contract, use_cuda_graph=use_cuda_graph)

    @classmethod
    def for_eager_extend_capacity(
        cls,
        *,
        mode: Literal["extend", "verify"] = "extend",
        device: torch.device | str,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        max_total_q: int,
        max_batch: int,
        max_page_table_width: int,
        num_cache_pages: int,
        use_cuda_graph: bool = False,
    ) -> PagedAttentionWorkspace:
        return cls.for_fixed_capacity(
            mode=mode,
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            page_size=page_size,
            max_total_q=max_total_q,
            max_batch=max_batch,
            max_page_table_width=max_page_table_width,
            max_work_items=cls.eager_extend_work_items_capacity(
                max_total_q=max_total_q,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
            ),
            max_partial_rows=0,
            num_cache_pages=num_cache_pages,
            use_cuda_graph=use_cuda_graph,
        )

    @classmethod
    def for_tensors(
        cls,
        *,
        mode: Literal["decode", "extend", "verify"],
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        use_cuda_graph: bool = False,
    ) -> PagedAttentionWorkspace:
        if q.ndim != 3:
            raise ValueError(
                f"q must have shape [total_q, q_heads, head_dim], got {tuple(q.shape)}"
            )
        if k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError("k_cache and v_cache must be rank-4 paged tensors")
        return cls.for_contract(
            mode=mode,
            device=q.device,
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=int(q.shape[1]),
            num_kv_heads=int(k_cache.shape[2]),
            head_dim_qk=int(q.shape[2]),
            head_dim_vo=int(v_cache.shape[3]),
            page_size=int(k_cache.shape[1]),
            max_total_q=int(q.shape[0]),
            num_cache_pages=int(k_cache.shape[0]),
            use_cuda_graph=use_cuda_graph,
        )

    @property
    def prepared(self) -> bool:
        return self._plan is not None

    @property
    def plan(self) -> PagedPlan:
        if self._plan is None:
            raise RuntimeError("paged workspace has not been prepared")
        return self._plan

    @property
    def active_total_q(self) -> int:
        return self.plan.total_q

    @property
    def active_batch(self) -> int:
        return self.plan.page_table_shape[0]

    @property
    def total_q_capacity(self) -> int:
        if self._plan_q is None:
            raise RuntimeError("paged workspace planning contract is not initialized")
        return int(self._plan_q.shape[0])

    @property
    def planner_budget(self) -> PagedPlanBudget | None:
        return self._planner_budget

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
    ) -> PagedAttentionWorkspace:
        with record_function(f"paged_workspace.prepare.{self.mode}"):
            if window_left < -1:
                raise ValueError(
                    "window_left must be -1 for full attention or a non-negative token count"
                )
            if self.use_cuda_graph and torch.cuda.is_current_stream_capturing():
                if self._plan is None:
                    raise RuntimeError(
                        "graph-mode paged workspace must be prepared before CUDA graph capture"
                    )
                if int(window_left) != int(self._plan.window_left):
                    raise ValueError(
                        f"captured paged attention graph was prepared with window_left={self._plan.window_left}, "
                        f"got window_left={int(window_left)}"
                    )
                with record_function("paged_workspace.copy_runtime_metadata.capture"):
                    self._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
                return self

            if (
                self.use_cuda_graph
                and self.mode == "decode"
                and self._decode_graph_chunk_pages_lut is not None
                and self._plan is not None
            ):
                if int(window_left) != int(self._plan.window_left):
                    raise ValueError(
                        f"decode graph replay workspace was prepared with window_left={self._plan.window_left}, "
                        f"got window_left={int(window_left)}"
                    )
                with record_function("paged_workspace.copy_runtime_metadata"):
                    self._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
                if not self._decode_graph_metadata_captured_in_graph:
                    with record_function(
                        "paged_workspace.update_decode_graph_replay_metadata"
                    ):
                        self.update_decode_graph_replay_metadata_from_runtime_cache_seqlens()
                    if torch.cuda.is_current_stream_capturing():
                        self._decode_graph_metadata_captured_in_graph = True
                return self

            if active_total_q is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "paged workspace prepare() requires active_total_q to be "
                        "supplied before CUDA graph capture; inferring it from a "
                        "device cu_seqlens_q would require an illegal host sync"
                    )
                with record_function("paged_workspace.infer_mode"):
                    inferred_mode = infer_paged_mode(cu_seqlens_q)
                active_total_q = int(cu_seqlens_q[-1].item())
            else:
                active_total_q = int(active_total_q)
                inferred_mode = _infer_mode_from_host_total(
                    cu_seqlens_q, active_total_q
                )
            if (
                inferred_mode != self.mode
                and not (self.mode == "extend" and inferred_mode == "decode")
                and not (self.mode == "verify" and inferred_mode == "extend")
            ):
                raise ValueError(
                    f"workspace mode {self.mode} does not match prepared mode {inferred_mode}"
                )
            with record_function("paged_workspace.ensure_plan_contract"):
                self._ensure_plan_contract(active_total_q)
            assert self._plan_q is not None
            assert self._plan_k_cache is not None
            assert self._plan_v_cache is not None
            if active_total_q <= 0 or active_total_q > int(self._plan_q.shape[0]):
                raise ValueError(
                    f"cu_seqlens_q implies total_q={active_total_q}, but workspace capacity is {int(self._plan_q.shape[0])}"
                )

            with record_function("paged_workspace.create_paged_plan"):
                plan = create_paged_plan(
                    self._plan_q[:active_total_q],
                    self._plan_k_cache,
                    self._plan_v_cache,
                    page_table,
                    cache_seqlens,
                    cu_seqlens_q,
                    mode=self.mode,
                    fixed_split_size=-1
                    if fixed_split_size is None
                    else int(fixed_split_size),
                    disable_split_kv=disable_split_kv,
                    window_left=int(window_left),
                    enable_cuda_graph=self.use_cuda_graph,
                    graph_chunk_policy=self.use_cuda_graph,
                    plan_budget=(
                        self.planner_budget
                        if self.mode in ("extend", "verify") and not self.use_cuda_graph
                        else None
                    ),
                )
            with record_function("paged_workspace.ensure_capacity"):
                self._ensure_capacity(plan)
            with record_function("paged_workspace.copy_runtime_metadata"):
                self._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
            with record_function("paged_workspace.copy_plan_metadata"):
                self._copy_plan_metadata(plan)
            self._plan = plan
            return self

    def update_decode_graph_replay_metadata_from_runtime_cache_seqlens(
        self,
    ) -> PagedAttentionWorkspace:
        if not self.use_cuda_graph:
            raise RuntimeError(
                "update_decode_graph_replay_metadata_from_runtime_cache_seqlens "
                "is only valid for graph-mode workspaces"
            )
        if self.mode != "decode":
            raise RuntimeError(
                "update_decode_graph_replay_metadata_from_runtime_cache_seqlens "
                "is only valid for decode workspaces"
            )
        if self._decode_graph_chunk_pages_lut is None:
            raise RuntimeError("decode graph replay policy has not been prepared")
        if self.cache_seqlens is None:
            raise RuntimeError("decode graph workspace is missing cache_seqlens")
        if self.request_indices is None:
            raise RuntimeError("decode graph workspace is missing request indices")
        if self.qo_tile_indices is None or self.kv_tile_indices is None:
            raise RuntimeError("decode graph workspace is missing tile indices")
        if self.merge_indptr is None or self.o_indptr is None:
            raise RuntimeError("decode graph workspace is missing indptr buffers")
        if self.kv_chunk_size_ptr is None:
            raise RuntimeError("decode graph workspace is missing kv_chunk_size_ptr")
        if self.total_num_rows_ptr is None:
            raise RuntimeError("decode graph workspace is missing total_num_rows_ptr")
        if self.block_valid_mask is None:
            raise RuntimeError("decode graph workspace is missing block_valid_mask")
        if self.kv_window_start_tokens is None:
            raise RuntimeError(
                "decode graph workspace is missing kv_window_start_tokens"
            )
        if self._plan is None:
            raise RuntimeError("decode graph workspace has not been prepared")

        batch = int(self.cache_seqlens.shape[0])
        self._validate_decode_graph_replay_capacity(batch=batch)
        window_page_span = self._window_page_span_from_plan(self._plan)

        if not self._plan.split_kv:
            from .graph_replay import update_regular_decode_graph_chunk_metadata

            update_regular_decode_graph_chunk_metadata(
                cache_seqlens=self.cache_seqlens,
                merge_indptr=self.merge_indptr,
                o_indptr=self.o_indptr,
                kv_chunk_size_ptr=self.kv_chunk_size_ptr,
                kv_chunk_size=int(self._plan.kv_chunk_size),
                kv_window_start_tokens=self.kv_window_start_tokens,
                max_chunks_per_req=1,
                page_size=self.page_size,
                window_page_span=window_page_span,
                window_left=int(self._plan.window_left),
            )
        elif self._use_regular_decode_graph_replay:
            from .graph_replay import (
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
            from .graph_replay import update_decode_graph_chunk_metadata

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
        return self

    def update_prefill_graph_replay_metadata(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        *,
        window_left: int = -1,
    ) -> PagedAttentionWorkspace:
        if not self.use_cuda_graph:
            raise RuntimeError(
                "update_prefill_graph_replay_metadata is only valid for graph-mode workspaces"
            )
        if self.mode not in ("extend", "verify"):
            raise RuntimeError(
                "update_prefill_graph_replay_metadata is only valid for extend/verify workspaces"
            )
        if self._plan is None:
            raise RuntimeError("prefill graph workspace has not been prepared")
        if int(window_left) != int(self._plan.window_left):
            raise ValueError(
                f"prefill graph replay workspace was prepared with window_left={self._plan.window_left}, "
                f"got window_left={int(window_left)}"
            )
        if self.page_table is None:
            raise RuntimeError("prefill graph workspace is missing page_table")
        if self.cache_seqlens is None:
            raise RuntimeError("prefill graph workspace is missing cache_seqlens")
        if self.cu_seqlens_q is None:
            raise RuntimeError("prefill graph workspace is missing cu_seqlens_q")

        with record_function("paged_workspace.copy_runtime_metadata"):
            self._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
        with record_function("paged_workspace.update_prefill_graph_replay_metadata"):
            self._update_prefill_graph_replay_metadata_from_runtime()
        return self

    def prepare_for_cuda_graph_replay(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        *,
        fixed_split_size: int | None = None,
        disable_split_kv: bool = False,
        active_total_q: int | None = None,
    ) -> PagedAttentionWorkspace:
        if not self.use_cuda_graph:
            raise RuntimeError(
                "prepare_for_cuda_graph_replay is only valid for graph-mode workspaces"
            )
        return self.prepare(
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
            active_total_q=active_total_q,
        )

    def bind_cuda_graph_runtime_metadata(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> PagedAttentionWorkspace:
        if not self.use_cuda_graph:
            raise RuntimeError(
                "bind_cuda_graph_runtime_metadata is only valid for graph-mode workspaces"
            )
        # Decode graph replay already selected the workspace by mode, so avoid
        # re-reading cu_seqlens_q back to the CPU just to rediscover q_len=1.
        inferred_mode = (
            self.mode if self.mode == "decode" else infer_paged_mode(cu_seqlens_q)
        )
        if (
            inferred_mode != self.mode
            and not (self.mode == "extend" and inferred_mode == "decode")
            and not (self.mode == "verify" and inferred_mode == "extend")
        ):
            raise ValueError(
                f"workspace mode {self.mode} does not match bound mode {inferred_mode}"
            )
        if (
            page_table.device != self.device
            or cache_seqlens.device != self.device
            or cu_seqlens_q.device != self.device
        ):
            raise ValueError("bound graph metadata must stay on the workspace device")
        self.page_table = page_table
        self.cache_seqlens = cache_seqlens
        self.cu_seqlens_q = cu_seqlens_q
        return self

    def prepare_decode_graph_replay_state(
        self,
        *,
        batch: int,
        max_page_table_width: int,
        total_q_capacity: int | None = None,
        max_cache_page_count: int | None = None,
        window_left: int = -1,
    ) -> PagedAttentionWorkspace:
        if not self.use_cuda_graph:
            raise RuntimeError(
                "prepare_decode_graph_replay_state is only valid for graph-mode workspaces"
            )
        if self.mode != "decode":
            raise RuntimeError(
                "prepare_decode_graph_replay_state is only valid for decode workspaces"
            )
        if total_q_capacity is None:
            total_q_capacity = (
                int(self._plan_q.shape[0]) if self._plan_q is not None else int(batch)
            )
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

        from .graph_replay import (
            make_decode_chunk_pages_lut_tensor,
            summarize_decode_chunk_pages_lut,
        )

        max_effective_kv_pages = int(max_cache_page_count)
        if window_left >= 0:
            max_effective_kv_pages = min(
                max_effective_kv_pages,
                max(
                    1,
                    (int(window_left) + self.page_size + self.page_size - 1)
                    // self.page_size,
                ),
            )
        gqa_group_size = self.num_q_heads // self.num_kv_heads
        graph_ctas_per_sm = resolve_decode_graph_ctas_per_sm(
            kv_dtype=self.kv_dtype,
            batch=batch,
            page_size=self.page_size,
            head_dim_qk=self.head_dim_qk,
            head_dim_vo=self.head_dim_vo,
            gqa_group_size=gqa_group_size,
        )
        max_chunks_per_req_budget = decode_graph_max_chunks_per_request_budget(
            device=self.device,
            num_kv_heads=self.num_kv_heads,
            batch=batch,
            graph_ctas_per_sm=graph_ctas_per_sm,
        )
        decode_chunk_pages_lut = build_decode_chunk_pages_lut(
            q_dtype=self.dtype,
            kv_dtype=self.kv_dtype,
            batch=batch,
            page_size=self.page_size,
            head_dim_qk=self.head_dim_qk,
            head_dim_vo=self.head_dim_vo,
            gqa_group_size=gqa_group_size,
            max_effective_kv_pages=max_effective_kv_pages,
            max_chunks_per_req=max_chunks_per_req_budget,
        )
        worst_page_count, max_chunks_per_req = summarize_decode_chunk_pages_lut(
            decode_chunk_pages_lut
        )
        self._decode_graph_chunk_pages_lut = make_decode_chunk_pages_lut_tensor(
            decode_chunk_pages_lut,
            device=self.device,
        )
        self._decode_graph_max_chunks_per_req = int(max_chunks_per_req)
        self._use_regular_decode_graph_replay = False
        self._decode_graph_metadata_captured_in_graph = False
        capacity_cache_seqlen = worst_page_count * self.page_size
        if window_left >= 0:
            capacity_cache_seqlen = max_cache_page_count * self.page_size - 1
        self.prepare_for_capacity(
            batch=batch,
            total_q_capacity=total_q_capacity,
            max_page_table_width=max_page_table_width,
            max_cache_seqlen=capacity_cache_seqlen,
            window_left=window_left,
        )
        self._use_regular_decode_graph_replay = (
            self._plan is not None
            and int(self._plan.gqa_group_size) <= int(self._plan.cta_tile_q)
            and self._plan_has_regular_decode_graph_grid(self._plan)
        )
        self._validate_decode_graph_replay_capacity(batch=batch)
        return self

    def prepare_prefill_graph_replay_state(
        self,
        *,
        batch: int,
        total_q_capacity: int,
        max_page_table_width: int,
        max_cache_seqlen: int,
        cu_seqlens_q: torch.Tensor,
        window_left: int = -1,
    ) -> PagedAttentionWorkspace:
        if not self.use_cuda_graph:
            raise RuntimeError(
                "prepare_prefill_graph_replay_state is only valid for graph-mode workspaces"
            )
        if self.mode not in ("extend", "verify"):
            raise RuntimeError(
                "prepare_prefill_graph_replay_state is only valid for extend/verify workspaces"
            )
        if batch <= 0:
            raise ValueError("batch must be positive")
        if total_q_capacity <= 0:
            raise ValueError("total_q_capacity must be positive")
        if max_page_table_width <= 0:
            raise ValueError("max_page_table_width must be positive")
        if max_cache_seqlen <= 0:
            raise ValueError("max_cache_seqlen must be positive")
        if window_left < -1:
            raise ValueError(
                "window_left must be -1 for full attention or a non-negative token count"
            )
        if tuple(cu_seqlens_q.shape) != (int(batch) + 1,):
            raise ValueError("cu_seqlens_q shape must match the graph batch")

        max_cache_seqlens = torch.full(
            (batch,),
            int(max_cache_seqlen),
            dtype=torch.int32,
            device=self.device,
        )
        num_cache_pages = (
            int(self._plan_k_cache.shape[0]) if self._plan_k_cache is not None else 0
        )
        if num_cache_pages <= 0:
            raise RuntimeError("paged workspace planning contract is not initialized")
        max_page_ids = torch.arange(
            max_page_table_width, dtype=torch.int32, device=self.device
        )
        max_page_table = (
            (max_page_ids % num_cache_pages).unsqueeze(0).expand(batch, -1).contiguous()
        )
        self.prepare(
            max_page_table,
            max_cache_seqlens,
            cu_seqlens_q,
            window_left=window_left,
            active_total_q=total_q_capacity,
        )
        self._cache_prefill_graph_replay_shape_from_plan()
        return self

    def prepare_for_capacity(
        self,
        *,
        batch: int,
        total_q_capacity: int,
        max_page_table_width: int,
        max_cache_seqlen: int,
        window_left: int = -1,
    ) -> PagedAttentionWorkspace:
        if batch <= 0:
            raise ValueError("batch must be positive")
        if total_q_capacity <= 0:
            raise ValueError("total_q_capacity must be positive")
        if max_page_table_width <= 0:
            raise ValueError("max_page_table_width must be positive")
        if max_cache_seqlen <= 0:
            raise ValueError("max_cache_seqlen must be positive")

        max_cache_seqlens = torch.full(
            (batch,),
            int(max_cache_seqlen),
            dtype=torch.int32,
            device=self.device,
        )
        num_cache_pages = (
            int(self._plan_k_cache.shape[0]) if self._plan_k_cache is not None else 0
        )
        if num_cache_pages <= 0:
            raise RuntimeError("paged workspace planning contract is not initialized")
        max_page_ids = torch.arange(
            max_page_table_width, dtype=torch.int32, device=self.device
        )
        max_page_table = (
            (max_page_ids % num_cache_pages).unsqueeze(0).expand(batch, -1).contiguous()
        )
        max_cu_seqlens_q = self._build_capacity_cu_seqlens_q(
            batch=batch,
            total_q_capacity=total_q_capacity,
        )
        return self.prepare(
            max_page_table,
            max_cache_seqlens,
            max_cu_seqlens_q,
            window_left=window_left,
            active_total_q=batch if self.mode == "decode" else total_q_capacity,
        )

    def _build_capacity_cu_seqlens_q(
        self,
        *,
        batch: int,
        total_q_capacity: int,
    ) -> torch.Tensor:
        if self.mode == "decode":
            return torch.arange(0, batch + 1, dtype=torch.int32, device=self.device)
        if total_q_capacity < batch:
            raise ValueError(
                f"extend graph bucket requires total_q_capacity >= batch, got {total_q_capacity} < {batch}"
            )
        q_lens = torch.ones(batch, dtype=torch.int32, device=self.device)
        q_lens[-1] = int(total_q_capacity - (batch - 1))
        cu_seqlens_q = torch.zeros(batch + 1, dtype=torch.int32, device=self.device)
        torch.cumsum(q_lens, dim=0, out=cu_seqlens_q[1:])
        return cu_seqlens_q

    def _ensure_plan_contract(self, active_total_q: int) -> None:
        if (
            self._plan_q is None
            or self._plan_k_cache is None
            or self._plan_v_cache is None
        ):
            raise RuntimeError("paged workspace planning contract is not initialized")
        if active_total_q <= int(self._plan_q.shape[0]):
            return
        if self.use_cuda_graph:
            raise ValueError(
                "graph-mode paged workspace capacity exceeded; construct a larger workspace or capture a larger graph bucket"
            )
        if self.fixed_capacity:
            raise ValueError(
                "fixed-capacity paged workspace exceeded; construct a larger eager extend workspace"
            )
        self._plan_q = _shape_only_cuda_tensor(
            (active_total_q, self.num_q_heads, self.head_dim_qk),
            dtype=self.dtype,
            device=self.device,
        )

    def update_decode_graph_replay_metadata(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> PagedAttentionWorkspace:
        if not self.use_cuda_graph:
            raise RuntimeError(
                "update_decode_graph_replay_metadata is only valid for graph-mode workspaces"
            )
        if self.mode != "decode":
            raise RuntimeError(
                "update_decode_graph_replay_metadata is only valid for decode workspaces"
            )
        if self._decode_graph_chunk_pages_lut is None:
            raise RuntimeError("decode graph replay policy has not been prepared")
        if self.page_table is None:
            raise RuntimeError("decode graph workspace is missing page_table")
        if self.cache_seqlens is None:
            raise RuntimeError("decode graph workspace is missing cache_seqlens")
        if self.cu_seqlens_q is None:
            raise RuntimeError("decode graph workspace is missing cu_seqlens_q")
        if self.request_indices is None:
            raise RuntimeError("decode graph workspace is missing request indices")
        if self.qo_tile_indices is None or self.kv_tile_indices is None:
            raise RuntimeError("decode graph workspace is missing tile indices")
        if self.block_valid_mask is None:
            raise RuntimeError("decode graph workspace is missing block_valid_mask")
        if self.merge_indptr is None or self.o_indptr is None:
            raise RuntimeError("decode graph workspace is missing indptr buffers")
        if self.kv_chunk_size_ptr is None:
            raise RuntimeError("decode graph workspace is missing kv_chunk_size_ptr")
        if self.total_num_rows_ptr is None:
            raise RuntimeError("decode graph workspace is missing total_num_rows_ptr")
        if self.kv_window_start_tokens is None:
            raise RuntimeError(
                "decode graph workspace is missing kv_window_start_tokens"
            )
        if self._plan is None:
            raise RuntimeError("decode graph workspace has not been prepared")

        batch = int(self.cache_seqlens.shape[0])
        self._validate_decode_graph_replay_capacity(batch=batch)
        window_page_span = self._window_page_span_from_plan(self._plan)

        if not self._plan.split_kv:
            from .graph_replay import (
                _DECODE_BLOCK_PAGES,
                build_decode_graph_page_table_full_triton,
                update_regular_decode_graph_chunk_metadata,
            )

            page_blocks = (
                int(self.page_table.shape[1]) + _DECODE_BLOCK_PAGES - 1
            ) // _DECODE_BLOCK_PAGES
            build_decode_graph_page_table_full_triton[(batch, page_blocks)](
                req_to_token,
                req_pool_indices,
                self.page_table,
                req_to_token.stride(0),
                req_to_token.numel(),
                self.page_table.stride(0),
                PAGE_SIZE=self.page_size,
                MAX_PAGES=int(self.page_table.shape[1]),
                BLOCK_PAGES=_DECODE_BLOCK_PAGES,
            )
            update_regular_decode_graph_chunk_metadata(
                cache_seqlens=self.cache_seqlens,
                merge_indptr=self.merge_indptr,
                o_indptr=self.o_indptr,
                kv_chunk_size_ptr=self.kv_chunk_size_ptr,
                kv_chunk_size=int(self._plan.kv_chunk_size),
                kv_window_start_tokens=self.kv_window_start_tokens,
                max_chunks_per_req=1,
                page_size=self.page_size,
                window_page_span=window_page_span,
                window_left=int(self._plan.window_left),
            )
        elif self._use_regular_decode_graph_replay:
            from .graph_replay import update_regular_decode_graph_replay_metadata

            update_regular_decode_graph_replay_metadata(
                req_to_token=req_to_token,
                req_pool_indices=req_pool_indices,
                page_table=self.page_table,
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
            from .graph_replay import update_decode_graph_replay_metadata

            update_decode_graph_replay_metadata(
                req_to_token=req_to_token,
                req_pool_indices=req_pool_indices,
                page_table=self.page_table,
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
        return self

    def bind_paged_attention(
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
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
        attention_sink_bias: torch.Tensor | None = None,
        relative_attention_bias: torch.Tensor | None = None,
    ):
        from flashinfer.experimental.sm12x.attention.paged._scratch import (
            build_paged_attention_binding,
        )

        return build_paged_attention_binding(
            workspace=self,
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
        )

    def _validate_q2k_indices_reference(self, q2k_indices: torch.Tensor | None) -> None:
        if q2k_indices is not None:
            raise ValueError(
                "q2k_indices are only supported by MSA block-sparse scratch bindings"
            )

    @torch._dynamo.disable
    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        output: torch.Tensor,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
        attention_sink_bias: torch.Tensor | None = None,
        relative_attention_bias: torch.Tensor | None = None,
        prepare_decode_graph_metadata: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from ._forward import paged_attention_forward
        from flashinfer.experimental.sm12x.attention.paged._scratch import (
            SM12XPagedAttentionBinding,
        )

        if prepare_decode_graph_metadata is None:
            prepare_decode_graph_metadata = (
                self.use_cuda_graph
                and self.mode == "decode"
                and self._decode_graph_chunk_pages_lut is not None
                and self._plan is not None
                and torch.cuda.is_current_stream_capturing()
            )
        if prepare_decode_graph_metadata:
            if not self.use_cuda_graph:
                raise RuntimeError(
                    "prepare_decode_graph_metadata requires a graph-mode workspace"
                )
            if self.mode != "decode":
                raise RuntimeError(
                    "prepare_decode_graph_metadata is only valid for decode workspaces"
                )
            if self._decode_graph_chunk_pages_lut is None or self._plan is None:
                raise RuntimeError(
                    "prepare_decode_graph_metadata requires a decode replay-state workspace"
                )
            with record_function(
                "paged_workspace.capture_decode_graph_replay_metadata"
            ):
                self.update_decode_graph_replay_metadata_from_runtime_cache_seqlens()
            if torch.cuda.is_current_stream_capturing():
                self._decode_graph_metadata_captured_in_graph = True

        out, lse = paged_attention_forward(
            binding=SM12XPagedAttentionBinding(
                scratch=self,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                output=output,
                k_descale=k_descale,
                v_descale=v_descale,
                attention_sink_bias=attention_sink_bias,
                relative_attention_bias=relative_attention_bias,
            )
        )
        return out, lse

    def current_lse_view(self) -> torch.Tensor:
        if self.lse is None:
            raise RuntimeError("workspace has not been prepared")
        return self.lse[:, : self.active_total_q].transpose(0, 1)

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
            raise ValueError("workspace inputs must stay on the workspace device")
        if q.dtype != self.dtype:
            raise TypeError(f"workspace expects q dtype {self.dtype}, got {q.dtype}")
        if k_cache.dtype != self.kv_dtype or v_cache.dtype != self.kv_dtype:
            raise TypeError(
                f"workspace expects kv dtype {self.kv_dtype}, got {k_cache.dtype}/{v_cache.dtype}"
            )
        if tuple(q.shape[1:]) != (self.num_q_heads, self.head_dim_qk):
            raise ValueError(
                "q shape does not match the workspace contract: "
                f"expected (*, {self.num_q_heads}, {self.head_dim_qk}), got {tuple(q.shape)}"
            )
        if (
            int(k_cache.shape[1]) != self.page_size
            or int(v_cache.shape[1]) != self.page_size
        ):
            raise ValueError(f"workspace expects page_size={self.page_size}")
        if (
            int(k_cache.shape[2]) != self.num_kv_heads
            or int(v_cache.shape[2]) != self.num_kv_heads
        ):
            raise ValueError("kv head count does not match the workspace contract")
        if int(k_cache.shape[3]) != self.head_dim_qk:
            raise ValueError("k_cache head_dim does not match the workspace contract")
        if int(v_cache.shape[3]) != self.head_dim_vo:
            raise ValueError("v_cache head_dim does not match the workspace contract")

    def _ensure_capacity(self, plan: PagedPlan) -> None:
        work_items_needed = int(plan.new_batch_size)
        block_valid_needed = int(plan.padded_batch_size)
        total_q_needed = int(plan.total_q)
        batch_needed = int(plan.page_table_shape[0])
        page_table_width_needed = int(plan.page_table_shape[1])
        partial_rows_needed = int(plan.total_num_partial_rows) if plan.split_kv else 0

        work_items_capacity = (
            0 if self.request_indices is None else int(self.request_indices.shape[0])
        )
        block_valid_capacity = (
            0 if self.block_valid_mask is None else int(self.block_valid_mask.shape[0])
        )
        total_q_capacity = 0 if self.lse is None else int(self.lse.shape[1])
        batch_capacity = 0 if self.o_indptr is None else int(self.o_indptr.shape[0] - 1)
        page_table_width_capacity = (
            0 if self.page_table is None else int(self.page_table.shape[1])
        )
        partial_rows_capacity = (
            0 if self.tmp_output is None else int(self.tmp_output.shape[0])
        )

        needs_growth = (
            work_items_needed > work_items_capacity
            or block_valid_needed > block_valid_capacity
            or total_q_needed > total_q_capacity
            or batch_needed > batch_capacity
            or page_table_width_needed > page_table_width_capacity
            or partial_rows_needed > partial_rows_capacity
        )
        if not needs_growth:
            return
        if self.use_cuda_graph and self.request_indices is not None:
            raise ValueError(
                "graph-mode paged workspace capacity exceeded; "
                f"needed work_items={work_items_needed}, block_valid={block_valid_needed}, "
                f"total_q={total_q_needed}, batch={batch_needed}, "
                f"page_table_width={page_table_width_needed}, partial_rows={partial_rows_needed}; "
                f"capacity work_items={work_items_capacity}, block_valid={block_valid_capacity}, "
                f"total_q={total_q_capacity}, batch={batch_capacity}, "
                f"page_table_width={page_table_width_capacity}, partial_rows={partial_rows_capacity}; "
                "construct a larger workspace or capture a larger graph bucket"
            )
        if self.fixed_capacity and self.request_indices is not None:
            raise ValueError(
                "fixed-capacity paged workspace exceeded; construct a larger eager extend workspace"
            )

        work_items_capacity = max(work_items_capacity, work_items_needed)
        block_valid_capacity = max(block_valid_capacity, block_valid_needed)
        total_q_capacity = max(total_q_capacity, total_q_needed)
        batch_capacity = max(batch_capacity, batch_needed)
        page_table_width_capacity = max(
            page_table_width_capacity, page_table_width_needed
        )
        partial_rows_capacity = max(partial_rows_capacity, partial_rows_needed)

        self._allocate_runtime_buffers(
            work_items_capacity=work_items_capacity,
            block_valid_capacity=block_valid_capacity,
            total_q_capacity=total_q_capacity,
            batch_capacity=batch_capacity,
            page_table_width_capacity=page_table_width_capacity,
            partial_rows_capacity=partial_rows_capacity,
        )

    def _allocate_runtime_buffers(
        self,
        *,
        work_items_capacity: int,
        block_valid_capacity: int,
        total_q_capacity: int,
        batch_capacity: int,
        page_table_width_capacity: int,
        partial_rows_capacity: int,
    ) -> None:
        if self.arena is not None:
            if work_items_capacity > self.arena.caps.max_work_items:
                raise ValueError("paged attention arena work-item capacity exceeded")
            if block_valid_capacity > self.arena.caps.max_work_items:
                raise ValueError("paged attention arena block-valid capacity exceeded")
            if total_q_capacity > self.arena.caps.max_total_q:
                raise ValueError("paged attention arena total-q capacity exceeded")
            if batch_capacity > self.arena.caps.max_batch:
                raise ValueError("paged attention arena batch capacity exceeded")
            if page_table_width_capacity > self.arena.caps.max_page_table_width:
                raise ValueError(
                    "paged attention arena page-table width capacity exceeded"
                )
            if partial_rows_capacity > self.arena.caps.max_partial_rows:
                raise ValueError("paged attention arena partial-row capacity exceeded")

            self.shared_arena = self.arena.shared_arena
            self.shared_arena_nbytes = self.arena.shared_arena_nbytes
            assert self.shared_arena is not None
            self.request_indices, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.request_indices_offset_bytes,
                shape=(work_items_capacity,),
                dtype=torch.int32,
            )
            self.qo_tile_indices, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.qo_tile_indices_offset_bytes,
                shape=(work_items_capacity,),
                dtype=torch.int32,
            )
            self.kv_tile_indices, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.kv_tile_indices_offset_bytes,
                shape=(work_items_capacity,),
                dtype=torch.int32,
            )
            self.block_valid_mask, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.block_valid_mask_offset_bytes,
                shape=(block_valid_capacity,),
                dtype=torch.int32,
            )
            self.page_table, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.page_table_offset_bytes,
                shape=(batch_capacity, page_table_width_capacity),
                dtype=torch.int32,
            )
            self.cache_seqlens, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.cache_seqlens_offset_bytes,
                shape=(batch_capacity,),
                dtype=torch.int32,
            )
            self.cu_seqlens_q, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.cu_seqlens_q_offset_bytes,
                shape=(batch_capacity + 1,),
                dtype=torch.int32,
            )
            self.merge_indptr, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.merge_indptr_offset_bytes,
                shape=(total_q_capacity + 1,),
                dtype=torch.int32,
            )
            self.o_indptr, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.o_indptr_offset_bytes,
                shape=(batch_capacity + 1,),
                dtype=torch.int32,
            )
            self.kv_chunk_size_ptr, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.kv_chunk_size_ptr_offset_bytes,
                shape=(1,),
                dtype=torch.int32,
            )
            self.kv_window_start_tokens, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.kv_window_start_tokens_offset_bytes,
                shape=(batch_capacity,),
                dtype=torch.int32,
            )
            self.total_num_rows_ptr, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.total_num_rows_ptr_offset_bytes,
                shape=(1,),
                dtype=torch.int32,
            )
            self.lse, _ = _materialize_arena_view(
                self.shared_arena,
                offset_bytes=self.arena.lse_offset_bytes,
                shape=_paged_lse_storage_shape(total_q_capacity, self.num_q_heads),
                dtype=torch.float32,
            )
            if partial_rows_capacity > 0:
                self.tmp_output, _ = _materialize_arena_view(
                    self.shared_arena,
                    offset_bytes=self.arena.tmp_output_offset_bytes,
                    shape=(partial_rows_capacity, self.num_q_heads, self.head_dim_vo),
                    dtype=self.dtype,
                )
                self.tmp_lse, _ = _materialize_arena_view(
                    self.shared_arena,
                    offset_bytes=self.arena.tmp_lse_offset_bytes,
                    shape=(partial_rows_capacity, self.num_q_heads),
                    dtype=torch.float32,
                )
            else:
                self.tmp_output = None
                self.tmp_lse = None
            return

        self.request_indices = torch.empty(
            work_items_capacity, dtype=torch.int32, device=self.device
        )
        self.qo_tile_indices = torch.empty(
            work_items_capacity, dtype=torch.int32, device=self.device
        )
        self.kv_tile_indices = torch.empty(
            work_items_capacity, dtype=torch.int32, device=self.device
        )
        self.block_valid_mask = torch.empty(
            block_valid_capacity, dtype=torch.int32, device=self.device
        )
        self.page_table = torch.empty(
            (batch_capacity, page_table_width_capacity),
            dtype=torch.int32,
            device=self.device,
        )
        self.cache_seqlens = torch.empty(
            batch_capacity, dtype=torch.int32, device=self.device
        )
        self.cu_seqlens_q = torch.empty(
            batch_capacity + 1, dtype=torch.int32, device=self.device
        )
        self.merge_indptr = torch.empty(
            total_q_capacity + 1, dtype=torch.int32, device=self.device
        )
        self.o_indptr = torch.empty(
            batch_capacity + 1, dtype=torch.int32, device=self.device
        )
        self.kv_chunk_size_ptr = torch.empty(1, dtype=torch.int32, device=self.device)
        self.kv_window_start_tokens = torch.empty(
            batch_capacity, dtype=torch.int32, device=self.device
        )
        self.total_num_rows_ptr = torch.empty(1, dtype=torch.int32, device=self.device)
        self.lse = torch.empty(
            _paged_lse_storage_shape(total_q_capacity, self.num_q_heads),
            dtype=torch.float32,
            device=self.device,
        )
        if partial_rows_capacity > 0:
            self.tmp_output = torch.empty(
                (partial_rows_capacity, self.num_q_heads, self.head_dim_vo),
                dtype=self.dtype,
                device=self.device,
            )
            self.tmp_lse = torch.empty(
                (partial_rows_capacity, self.num_q_heads),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.tmp_output = None
            self.tmp_lse = None

    def _cache_prefill_graph_replay_shape_from_plan(self) -> None:
        if self._plan is None:
            raise RuntimeError("prefill graph replay plan has not been prepared")
        if self.mode not in ("extend", "verify") or not self.use_cuda_graph:
            raise RuntimeError(
                "prefill graph replay shape is only valid for graph extend/verify"
            )
        active_work_items = [
            (request_idx, q_tile_idx, kv_tile_idx)
            for request_idx, q_tile_idx, kv_tile_idx, valid in zip(
                self._plan.request_indices,
                self._plan.qo_tile_indices,
                self._plan.kv_tile_indices,
                self._plan.block_valid_mask,
                strict=False,
            )
            if valid
        ]
        if not active_work_items:
            raise RuntimeError("prefill graph replay plan contains no active work")
        max_q_tiles_per_req = (
            max(q_tile_idx for _, q_tile_idx, _ in active_work_items) + 1
        )
        max_chunks_per_q_tile = (
            max(kv_tile_idx for _, _, kv_tile_idx in active_work_items) + 1
        )
        max_q_rows_per_req = max(
            1,
            (
                int(max_q_tiles_per_req) * int(self._plan.cta_tile_q)
                + int(self._plan.gqa_group_size)
                - 1
            )
            // int(self._plan.gqa_group_size),
        )
        self._prefill_graph_max_q_tiles_per_req = int(max_q_tiles_per_req)
        self._prefill_graph_max_chunks_per_q_tile = int(max_chunks_per_q_tile)
        self._prefill_graph_max_q_rows_per_req = int(max_q_rows_per_req)

    def _update_prefill_graph_replay_metadata_from_runtime(self) -> None:
        if self._plan is None:
            raise RuntimeError("prefill graph workspace has not been prepared")
        if self.cache_seqlens is None:
            raise RuntimeError("prefill graph workspace is missing cache_seqlens")
        if self.cu_seqlens_q is None:
            raise RuntimeError("prefill graph workspace is missing cu_seqlens_q")
        if self.request_indices is None:
            raise RuntimeError("prefill graph workspace is missing request indices")
        if self.qo_tile_indices is None or self.kv_tile_indices is None:
            raise RuntimeError("prefill graph workspace is missing tile indices")
        if self.merge_indptr is None or self.o_indptr is None:
            raise RuntimeError("prefill graph workspace is missing indptr buffers")
        if self.kv_chunk_size_ptr is None:
            raise RuntimeError("prefill graph workspace is missing kv_chunk_size_ptr")
        if self.total_num_rows_ptr is None:
            raise RuntimeError("prefill graph workspace is missing total_num_rows_ptr")
        if self.block_valid_mask is None:
            raise RuntimeError("prefill graph workspace is missing block_valid_mask")
        if self.kv_window_start_tokens is None:
            raise RuntimeError(
                "prefill graph workspace is missing kv_window_start_tokens"
            )
        if (
            self._prefill_graph_max_q_tiles_per_req is None
            or self._prefill_graph_max_chunks_per_q_tile is None
            or self._prefill_graph_max_q_rows_per_req is None
        ):
            self._cache_prefill_graph_replay_shape_from_plan()

        from .graph_replay import update_prefill_graph_chunk_metadata

        update_prefill_graph_chunk_metadata(
            cache_seqlens=self.cache_seqlens,
            cu_seqlens_q=self.cu_seqlens_q,
            request_indices=self.request_indices,
            qo_tile_indices=self.qo_tile_indices,
            kv_tile_indices=self.kv_tile_indices,
            merge_indptr=self.merge_indptr,
            o_indptr=self.o_indptr,
            block_valid_mask=self.block_valid_mask,
            kv_chunk_size_ptr=self.kv_chunk_size_ptr,
            kv_window_start_tokens=self.kv_window_start_tokens,
            total_num_rows_ptr=self.total_num_rows_ptr,
            batch=int(self._plan.page_table_shape[0]),
            max_q_tiles_per_req=int(self._prefill_graph_max_q_tiles_per_req),
            max_chunks_per_q_tile=int(self._prefill_graph_max_chunks_per_q_tile),
            max_q_rows_per_req=int(self._prefill_graph_max_q_rows_per_req),
            cta_tile_q=int(self._plan.cta_tile_q),
            gqa_group_size=int(self._plan.gqa_group_size),
            page_size=int(self.page_size),
            split_kv=bool(self._plan.split_kv),
            window_left=int(self._plan.window_left),
        )

    def _validate_decode_graph_replay_capacity(
        self,
        *,
        batch: int,
    ) -> None:
        if self._decode_graph_max_chunks_per_req is None:
            raise RuntimeError("decode graph replay policy has not been prepared")
        if self.request_indices is None:
            raise RuntimeError("decode graph workspace is missing request indices")
        if batch <= 0:
            raise ValueError("decode graph replay requires bs > 0")
        work_items_capacity = int(self.request_indices.shape[0])
        if work_items_capacity % batch != 0:
            raise RuntimeError(
                "decode graph workspace request_indices shape is incompatible with the batch bucket"
            )
        max_chunks_per_req = work_items_capacity // batch
        if max_chunks_per_req <= 0:
            raise RuntimeError(
                "decode graph workspace must allocate at least one chunk per request"
            )
        if self._decode_graph_max_chunks_per_req > max_chunks_per_req:
            raise RuntimeError(
                "decode graph workspace capacity is too small for the current chunking policy"
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

        with record_function("paged_workspace.runtime_cast_metadata"):
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

        with record_function("paged_workspace.runtime_copy_page_table"):
            self.page_table[: page_table_i32.shape[0], : page_table_i32.shape[1]].copy_(
                page_table_i32
            )
        with record_function("paged_workspace.runtime_copy_cache_seqlens"):
            self.cache_seqlens[: cache_seqlens_i32.shape[0]].copy_(cache_seqlens_i32)
        with record_function("paged_workspace.runtime_copy_cu_seqlens_q"):
            self.cu_seqlens_q[: cu_seqlens_q_i32.shape[0]].copy_(cu_seqlens_q_i32)

    def _copy_regular_decode_graph_plan_metadata(self, plan: PagedPlan) -> None:
        assert self.request_indices is not None
        assert self.merge_indptr is not None
        assert self.o_indptr is not None
        assert self.kv_chunk_size_ptr is not None
        assert self.kv_window_start_tokens is not None
        assert self.total_num_rows_ptr is not None
        assert self.cache_seqlens is not None

        batch = int(plan.page_table_shape[0])
        if batch <= 0:
            raise RuntimeError("regular decode graph replay requires bs > 0")
        work_items_capacity = int(self.request_indices.shape[0])
        if work_items_capacity % batch != 0:
            raise RuntimeError(
                "decode graph workspace request_indices shape is incompatible with the batch bucket"
            )
        capture_max_chunks_per_req = work_items_capacity // batch
        current_work_items = len(plan.request_indices)
        if current_work_items % batch != 0:
            raise RuntimeError(
                "decode graph replay plan work-items are incompatible with the batch bucket"
            )
        current_max_chunks_per_req = current_work_items // batch
        if current_max_chunks_per_req > capture_max_chunks_per_req:
            raise RuntimeError(
                "decode graph replay plan exceeds the captured fixed-grid capacity"
            )

        from .graph_replay import update_regular_decode_graph_chunk_metadata

        update_regular_decode_graph_chunk_metadata(
            cache_seqlens=self.cache_seqlens,
            merge_indptr=self.merge_indptr,
            o_indptr=self.o_indptr,
            kv_chunk_size_ptr=self.kv_chunk_size_ptr,
            kv_chunk_size=int(plan.kv_chunk_size),
            kv_window_start_tokens=self.kv_window_start_tokens,
            max_chunks_per_req=capture_max_chunks_per_req,
            page_size=self.page_size,
            window_page_span=self._window_page_span_from_plan(plan),
            window_left=int(plan.window_left),
        )
        self.total_num_rows_ptr[0] = int(plan.total_q)

    def _window_page_span_from_plan(self, plan: PagedPlan) -> int:
        window_left = int(plan.window_left)
        if window_left < 0:
            return 0
        return max(
            (window_left + self.page_size + self.page_size - 1) // self.page_size,
            1,
        )

    def _copy_plan_metadata(self, plan: PagedPlan) -> None:
        assert self.request_indices is not None
        assert self.qo_tile_indices is not None
        assert self.kv_tile_indices is not None
        assert self.merge_indptr is not None
        assert self.o_indptr is not None
        assert self.kv_chunk_size_ptr is not None
        assert self.kv_window_start_tokens is not None
        assert self.total_num_rows_ptr is not None
        assert self.block_valid_mask is not None

        self._use_regular_decode_graph_replay = False
        self._prefill_graph_max_q_tiles_per_req = None
        self._prefill_graph_max_chunks_per_q_tile = None
        self._prefill_graph_max_q_rows_per_req = None

        with record_function("paged_workspace.plan_metadata_to_device"):
            request_indices = _copy_int_metadata(
                plan.request_indices, device=self.device
            )
            qo_tile_indices = _copy_int_metadata(
                plan.qo_tile_indices, device=self.device
            )
            kv_tile_indices = _copy_int_metadata(
                plan.kv_tile_indices, device=self.device
            )
            merge_indptr = _copy_int_metadata(plan.merge_indptr, device=self.device)
            o_indptr = _copy_int_metadata(plan.o_indptr, device=self.device)
            block_valid_mask = torch.tensor(
                plan.block_valid_mask, dtype=torch.int32, device=self.device
            )
            kv_window_start_tokens = _copy_int_metadata(
                plan.kv_window_start_tokens, device=self.device
            )

        with record_function("paged_workspace.plan_metadata_zero_buffers"):
            self.request_indices.zero_()
            self.qo_tile_indices.zero_()
            self.kv_tile_indices.zero_()
            self.block_valid_mask.zero_()
            self.kv_window_start_tokens.zero_()
        with record_function("paged_workspace.plan_metadata_copy_buffers"):
            self.request_indices[: request_indices.shape[0]].copy_(request_indices)
            self.qo_tile_indices[: qo_tile_indices.shape[0]].copy_(qo_tile_indices)
            self.kv_tile_indices[: kv_tile_indices.shape[0]].copy_(kv_tile_indices)
            self.merge_indptr[: merge_indptr.shape[0]].copy_(merge_indptr)
            self.o_indptr[: o_indptr.shape[0]].copy_(o_indptr)
            self.block_valid_mask[: block_valid_mask.shape[0]].copy_(block_valid_mask)
            self.kv_window_start_tokens[: kv_window_start_tokens.shape[0]].copy_(
                kv_window_start_tokens
            )
        with record_function("paged_workspace.plan_metadata_scalar_updates"):
            self.kv_chunk_size_ptr[0] = int(plan.kv_chunk_size)
            self.total_num_rows_ptr[0] = int(plan.total_q)
