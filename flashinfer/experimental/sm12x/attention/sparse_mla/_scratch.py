# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/integration/sparse_mla_scratch.py @ 17428af5 (2026-07-01) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Caller-owned scratch plans for sparse MLA paths.

Eager PLAN -> BIND -> KERNEL, never a workspace/arena. bind() maps the
caller-owned scratch tensor into per-spec kernel-argument VIEWS and returns a
plain SM12XSparseMLAScratch views container (mirroring SM12XCompressedMLAScratch).
It never constructs a SM12XAttentionWorkspace / arena, allocates, or init-writes.
The unified SM120 sparse-MLA decode/extend kernels duck-type the workspace
(tmp_output/tmp_lse/output_buffer/final_lse/num_chunks_ptr/kv_chunk_size_ptr/
set_split_chunk_config/...), so the views container is a drop-in -- no kernel
signature change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from flashinfer.experimental.sm12x.attention._shared.workspace import (
    _split_output_buffer_from_tmp,
    _split_tmp_output_stride,
)
from flashinfer.experimental.sm12x._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    dtype_nbytes,
    materialize_scratch_strided_view,
    materialize_scratch_view,
)
from flashinfer.experimental.sm12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)


@dataclass(frozen=True, kw_only=True)
class SM12XSparseMLAScratchCaps:
    device: torch.device | str
    num_q_heads: int
    max_q_rows: int
    max_width: int
    dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype = torch.bfloat16
    head_dim: int = 576
    v_head_dim: int = 512
    mode: Literal["decode", "extend", "verify", "draft_extend"] = "decode"
    max_batch: int | None = None
    max_kv_rows: int = 0
    max_page_table_width: int | None = None
    max_chunks_per_row: int = 64
    max_q_chunks: int | None = None
    page_size: int = 64

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "num_q_heads", max(int(self.num_q_heads), 1))
        object.__setattr__(self, "max_q_rows", max(int(self.max_q_rows), 1))
        object.__setattr__(self, "max_width", max(int(self.max_width), 1))
        object.__setattr__(self, "head_dim", max(int(self.head_dim), 1))
        object.__setattr__(self, "v_head_dim", max(int(self.v_head_dim), 1))
        max_batch = self.max_q_rows if self.max_batch is None else self.max_batch
        object.__setattr__(self, "max_batch", max(int(max_batch), 1))
        object.__setattr__(self, "max_kv_rows", max(int(self.max_kv_rows), 0))
        max_page_table_width = (
            self.max_width
            if self.max_page_table_width is None
            else self.max_page_table_width
        )
        object.__setattr__(
            self,
            "max_page_table_width",
            max(int(max_page_table_width), 1),
        )
        object.__setattr__(
            self,
            "max_chunks_per_row",
            max(int(self.max_chunks_per_row), 1),
        )
        if self.max_q_chunks is not None:
            object.__setattr__(self, "max_q_chunks", max(int(self.max_q_chunks), 1))
        object.__setattr__(self, "page_size", max(int(self.page_size), 1))


@dataclass(kw_only=True)
class SM12XSparseMLAScratch:
    """Component-owned sparse-MLA scratch VIEWS over caller-owned storage.

    Exposes exactly the attributes the unified SM120 sparse-MLA decode/extend
    kernels duck-type off the (former) workspace. NEVER a SM12XAttentionWorkspace.
    """

    shared_scratch: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    head_dim: int
    v_head_dim: int
    topk: int
    max_total_q: int
    max_batch: int
    max_chunks_per_row: int
    page_size: int
    mode: str = "decode"
    fixed_capacity: bool = True
    use_cuda_graph: bool = False
    tmp_output: torch.Tensor | None = None
    tmp_lse: torch.Tensor | None = None
    output_buffer: torch.Tensor | None = None
    final_lse: torch.Tensor | None = None
    kv_chunk_size_ptr: torch.Tensor | None = None
    num_chunks_ptr: torch.Tensor | None = None
    sm_scale_tensor: torch.Tensor | None = None
    kv_chunk_size_value: int | None = None
    num_chunks_value: int | None = None
    sm_scale_value: float | None = None

    def set_split_chunk_config(self, *, kv_chunk_size: int, num_chunks: int) -> None:
        if num_chunks <= 0 or num_chunks > self.max_chunks_per_row:
            raise ValueError(
                f"num_chunks must be in [1, {self.max_chunks_per_row}], got {num_chunks}"
            )
        if kv_chunk_size <= 0:
            raise ValueError(f"kv_chunk_size must be positive, got {kv_chunk_size}")
        if self.kv_chunk_size_ptr is None or self.num_chunks_ptr is None:
            raise RuntimeError("sparse MLA scratch is missing split-control tensors")
        if self.kv_chunk_size_value != int(kv_chunk_size):
            self.kv_chunk_size_ptr.fill_(int(kv_chunk_size))
            self.kv_chunk_size_value = int(kv_chunk_size)
        if self.num_chunks_value != int(num_chunks):
            self.num_chunks_ptr.fill_(int(num_chunks))
            self.num_chunks_value = int(num_chunks)

    def bind(
        self,
        *,
        q: torch.Tensor,
        selected_indices: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        nsa_cache_seqlens_int32: torch.Tensor,
    ) -> "SM12XSparseMLABinding":
        return build_sparse_mla_binding(
            scratch=self,
            q=q,
            selected_indices=selected_indices,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        )


@dataclass(frozen=True, kw_only=True)
class SM12XSparseMLABinding:
    scratch: object
    q: torch.Tensor
    selected_indices: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    nsa_cache_seqlens_int32: torch.Tensor


def _validate_device(
    tensor: torch.Tensor,
    *,
    scratch: object,
    name: str,
) -> None:
    if tensor.device != scratch.device:
        raise ValueError(
            f"{name} device {tensor.device} does not match scratch device {scratch.device}"
        )


def _validate_q(q: torch.Tensor, *, scratch: object) -> torch.Tensor:
    if q.ndim != 3:
        raise ValueError(f"q must be rank-3, got {tuple(q.shape)}")
    if q.dtype != scratch.dtype:
        raise TypeError(f"q must have dtype {scratch.dtype}, got {q.dtype}")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    _validate_device(q, scratch=scratch, name="q")
    if int(q.shape[0]) > int(scratch.max_total_q):
        raise ValueError(
            f"q rows {int(q.shape[0])} exceed scratch capacity {scratch.max_total_q}"
        )
    if int(q.shape[1]) != int(scratch.num_q_heads):
        raise ValueError(
            f"q heads {int(q.shape[1])} do not match scratch heads {scratch.num_q_heads}"
        )
    if int(q.shape[2]) != int(scratch.head_dim):
        raise ValueError(
            f"q head_dim {int(q.shape[2])} does not match scratch head_dim {scratch.head_dim}"
        )
    return q.detach()


def _validate_selected_indices(
    selected_indices: torch.Tensor,
    *,
    scratch: object,
    rows: int,
) -> torch.Tensor:
    if selected_indices.ndim != 2:
        raise ValueError(
            f"selected_indices must be rank-2, got {tuple(selected_indices.shape)}"
        )
    if selected_indices.dtype != torch.int32:
        raise TypeError(
            f"selected_indices must have dtype torch.int32, got {selected_indices.dtype}"
        )
    if not selected_indices.is_contiguous():
        raise ValueError("selected_indices must be contiguous")
    _validate_device(selected_indices, scratch=scratch, name="selected_indices")
    if int(selected_indices.shape[0]) != int(rows):
        raise ValueError(
            f"selected_indices rows {int(selected_indices.shape[0])} do not match q rows {rows}"
        )
    if int(selected_indices.shape[1]) > int(scratch.topk):
        raise ValueError(
            f"selected_indices width {int(selected_indices.shape[1])} exceeds scratch topk {scratch.topk}"
        )
    return selected_indices


def _validate_i32_vector(
    tensor: torch.Tensor,
    *,
    scratch: object,
    name: str,
    max_rows: int | None = None,
    rows: int | None = None,
) -> torch.Tensor:
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    _validate_device(tensor, scratch=scratch, name=name)
    if rows is not None and int(tensor.shape[0]) != int(rows):
        raise ValueError(
            f"{name} rows {int(tensor.shape[0])} do not match q rows {rows}"
        )
    if max_rows is not None and int(tensor.shape[0]) > int(max_rows):
        raise ValueError(
            f"{name} rows {int(tensor.shape[0])} exceed capacity {max_rows}"
        )
    return tensor


def build_sparse_mla_binding(
    *,
    scratch: object,
    q: torch.Tensor,
    selected_indices: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    nsa_cache_seqlens_int32: torch.Tensor,
) -> SM12XSparseMLABinding:
    q = _validate_q(q, scratch=scratch)
    rows = int(q.shape[0])
    selected_indices = _validate_selected_indices(
        selected_indices,
        scratch=scratch,
        rows=rows,
    )
    cache_seqlens_int32 = _validate_i32_vector(
        cache_seqlens_int32,
        scratch=scratch,
        name="cache_seqlens_int32",
        max_rows=scratch.max_batch,
    )
    nsa_cache_seqlens_int32 = _validate_i32_vector(
        nsa_cache_seqlens_int32,
        scratch=scratch,
        name="nsa_cache_seqlens_int32",
        rows=rows,
    )
    return SM12XSparseMLABinding(
        scratch=scratch,
        q=q,
        selected_indices=selected_indices,
        cache_seqlens_int32=cache_seqlens_int32,
        nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
    )


@dataclass(frozen=True)
class _SM12XSparseMLAScratchLayout:
    nbytes: int
    split: bool
    output_offset_bytes: int
    tmp_output_offset_bytes: int
    tmp_lse_offset_bytes: int
    final_lse_offset_bytes: int
    kv_chunk_size_offset_bytes: int
    num_chunks_offset_bytes: int
    sm_scale_offset_bytes: int


def _sparse_mla_scratch_layout(
    caps: SM12XSparseMLAScratchCaps,
) -> _SM12XSparseMLAScratchLayout:
    max_total_q = max(int(caps.max_q_rows), 1)
    num_q_heads = int(caps.num_q_heads)
    v_head_dim = int(caps.v_head_dim)
    max_chunks_per_row = max(int(caps.max_chunks_per_row), 1)
    # Only the split-K DECODE path needs tmp_output/tmp_lse/final_lse + split
    # control. The single-pass prefill (extend/verify/draft_extend) only writes
    # output_buffer, so it gets a standalone output view -- no multi-chunk scratch.
    split = caps.mode == "decode"

    cursor = 0
    tmp_output_offset_bytes = 0
    output_offset_bytes = 0
    tmp_lse_offset_bytes = 0
    final_lse_offset_bytes = 0
    kv_chunk_size_offset_bytes = 0
    num_chunks_offset_bytes = 0
    if split:
        cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
        tmp_output_offset_bytes = cursor
        # output_buffer aliases tmp_output[:, :, 0, :] (chunk-major stride), so no
        # separate output allocation is needed for decode.
        output_offset_bytes = cursor
        cursor += (
            max_total_q
            * max_chunks_per_row
            * num_q_heads
            * v_head_dim
            * dtype_nbytes(caps.dtype)
        )
        cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
        tmp_lse_offset_bytes = cursor
        cursor += (
            max_total_q * max_chunks_per_row * num_q_heads * dtype_nbytes(torch.float32)
        )
        cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
        final_lse_offset_bytes = cursor
        cursor += max_total_q * num_q_heads * dtype_nbytes(torch.float32)
        cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
        kv_chunk_size_offset_bytes = cursor
        cursor += dtype_nbytes(torch.int32)
        cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
        num_chunks_offset_bytes = cursor
        cursor += dtype_nbytes(torch.int32)
        cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
    else:
        cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
        output_offset_bytes = cursor
        cursor += max_total_q * num_q_heads * v_head_dim * dtype_nbytes(caps.dtype)
        cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    sm_scale_offset_bytes = cursor
    cursor += dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    return _SM12XSparseMLAScratchLayout(
        nbytes=max(int(cursor), SCRATCH_ALIGN_BYTES),
        split=split,
        output_offset_bytes=output_offset_bytes,
        tmp_output_offset_bytes=tmp_output_offset_bytes,
        tmp_lse_offset_bytes=tmp_lse_offset_bytes,
        final_lse_offset_bytes=final_lse_offset_bytes,
        kv_chunk_size_offset_bytes=kv_chunk_size_offset_bytes,
        num_chunks_offset_bytes=num_chunks_offset_bytes,
        sm_scale_offset_bytes=sm_scale_offset_bytes,
    )


def _materialize_sparse_mla_scratch(
    caps: SM12XSparseMLAScratchCaps,
    scratch_storage: torch.Tensor,
    layout: _SM12XSparseMLAScratchLayout,
) -> SM12XSparseMLAScratch:
    max_total_q = max(int(caps.max_q_rows), 1)
    num_q_heads = int(caps.num_q_heads)
    v_head_dim = int(caps.v_head_dim)
    max_chunks_per_row = max(int(caps.max_chunks_per_row), 1)

    tmp_output = None
    tmp_lse = None
    final_lse = None
    kv_chunk_size_ptr = None
    num_chunks_ptr = None
    if layout.split:
        tmp_output, _ = materialize_scratch_strided_view(
            scratch_storage,
            offset_bytes=layout.tmp_output_offset_bytes,
            shape=(max_total_q, num_q_heads, max_chunks_per_row, v_head_dim),
            stride=_split_tmp_output_stride(
                max_total_q=max_total_q,
                num_q_heads=num_q_heads,
                max_chunks_per_row=max_chunks_per_row,
                v_head_dim=v_head_dim,
            ),
            dtype=caps.dtype,
        )
        output_buffer = _split_output_buffer_from_tmp(tmp_output)
        tmp_lse, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.tmp_lse_offset_bytes,
            shape=(max_total_q, num_q_heads, max_chunks_per_row),
            dtype=torch.float32,
        )
        final_lse, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.final_lse_offset_bytes,
            shape=(max_total_q, num_q_heads),
            dtype=torch.float32,
        )
        kv_chunk_size_ptr, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.kv_chunk_size_offset_bytes,
            shape=(1,),
            dtype=torch.int32,
        )
        num_chunks_ptr, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.num_chunks_offset_bytes,
            shape=(1,),
            dtype=torch.int32,
        )
    else:
        output_buffer, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.output_offset_bytes,
            shape=(max_total_q, num_q_heads, v_head_dim),
            dtype=caps.dtype,
        )

    sm_scale_tensor, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.sm_scale_offset_bytes,
        shape=(1,),
        dtype=torch.float32,
    )

    scratch = SM12XSparseMLAScratch(
        shared_scratch=scratch_storage,
        device=caps.device,
        dtype=caps.dtype,
        kv_dtype=caps.kv_dtype,
        num_q_heads=num_q_heads,
        head_dim=caps.head_dim,
        v_head_dim=v_head_dim,
        topk=caps.max_width,
        max_total_q=caps.max_q_rows,
        max_batch=caps.max_batch,
        max_chunks_per_row=max_chunks_per_row,
        page_size=caps.page_size,
        mode=caps.mode,
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        output_buffer=output_buffer,
        final_lse=final_lse,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        num_chunks_ptr=num_chunks_ptr,
        sm_scale_tensor=sm_scale_tensor,
    )
    return scratch


@dataclass(frozen=True)
class SM12XSparseMLAScratchPlan:
    caps: SM12XSparseMLAScratchCaps
    layout: _SM12XSparseMLAScratchLayout
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        q: torch.Tensor,
        selected_indices: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        nsa_cache_seqlens_int32: torch.Tensor,
    ) -> SM12XSparseMLABinding:
        scratch_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="sparse MLA",
        )
        scratch_views = _materialize_sparse_mla_scratch(
            self.caps,
            scratch_storage,
            self.layout,
        )
        return build_sparse_mla_binding(
            scratch=scratch_views,
            q=q,
            selected_indices=selected_indices,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        )


def plan_sparse_mla_scratch(
    caps: SM12XSparseMLAScratchCaps,
) -> SM12XSparseMLAScratchPlan:
    layout = _sparse_mla_scratch_layout(caps)
    return SM12XSparseMLAScratchPlan(
        caps=caps,
        layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "sparse_mla.scratch",
                nbytes=int(layout.nbytes),
                device=caps.device,
            ),
        ),
    )


__all__ = [
    "SM12XSparseMLABinding",
    "SM12XSparseMLAScratch",
    "SM12XSparseMLAScratchCaps",
    "SM12XSparseMLAScratchPlan",
    "build_sparse_mla_binding",
    "plan_sparse_mla_scratch",
]
