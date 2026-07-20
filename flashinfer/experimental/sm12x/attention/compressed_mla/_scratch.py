# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/integration/compressed_scratch.py @ 149d6bb2 (2026-06-12) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Caller-owned scratch plans for compressed MLA paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from flashinfer.experimental.sm12x.attention._shared.mla.compressed_config import (
    compressed_mla_split_config_for_contract,
)
from flashinfer.experimental.sm12x.attention._shared.mla.compressed_reference import (
    COMPRESSED_MLA_HEAD_DIM,
)
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
class SM12XCompressedMLAScratchCaps:
    device: torch.device | str
    num_q_heads: int
    max_q_rows: int
    max_width: int
    max_page_table_width: int | None = None
    dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype = torch.uint8
    head_dim: int = COMPRESSED_MLA_HEAD_DIM
    v_head_dim: int = COMPRESSED_MLA_HEAD_DIM
    max_batch: int | None = None
    max_kv_rows: int = 0
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
        max_page_table_width = (
            self.max_width
            if self.max_page_table_width is None
            else self.max_page_table_width
        )
        object.__setattr__(
            self, "max_page_table_width", max(int(max_page_table_width), 1)
        )
        object.__setattr__(self, "head_dim", max(int(self.head_dim), 1))
        object.__setattr__(self, "v_head_dim", max(int(self.v_head_dim), 1))
        max_batch = self.max_q_rows if self.max_batch is None else self.max_batch
        object.__setattr__(self, "max_batch", max(int(max_batch), 1))
        object.__setattr__(self, "max_kv_rows", max(int(self.max_kv_rows), 0))
        object.__setattr__(
            self, "max_chunks_per_row", max(int(self.max_chunks_per_row), 1)
        )
        if self.max_q_chunks is not None:
            object.__setattr__(self, "max_q_chunks", max(int(self.max_q_chunks), 1))
        object.__setattr__(self, "page_size", max(int(self.page_size), 1))


@dataclass(frozen=True, kw_only=True)
class _SM12XCompressedMLAScratchLayout:
    nbytes: int
    max_q_chunks: int
    tmp_output_offset_bytes: int
    tmp_lse_offset_bytes: int
    final_lse_offset_bytes: int
    kv_chunk_size_offset_bytes: int
    num_chunks_offset_bytes: int
    sm_scale_offset_bytes: int


@dataclass(kw_only=True)
class SM12XCompressedMLAScratch:
    """Component-owned compressed MLA scratch views over caller-owned storage."""

    shared_scratch: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    head_dim: int
    v_head_dim: int
    topk: int
    max_page_table_width: int
    max_total_q: int
    max_batch: int
    max_kv_rows: int
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
    _contract_q: torch.Tensor | None = None
    _contract_page_table: torch.Tensor | None = None
    _contract_indexer_cache_seqlens: torch.Tensor | None = None
    _contract_output: torch.Tensor | None = None
    _contract_tmp_output: torch.Tensor | None = None
    _contract_tmp_lse: torch.Tensor | None = None

    def set_split_chunk_config(self, *, kv_chunk_size: int, num_chunks: int) -> None:
        if num_chunks <= 0 or num_chunks > self.max_chunks_per_row:
            raise ValueError(
                f"num_chunks must be in [1, {self.max_chunks_per_row}], got {num_chunks}"
            )
        if kv_chunk_size <= 0:
            raise ValueError(f"kv_chunk_size must be positive, got {kv_chunk_size}")
        if self.kv_chunk_size_ptr is None or self.num_chunks_ptr is None:
            raise RuntimeError(
                "compressed MLA scratch is missing split-control tensors"
            )
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
        swa_indices: torch.Tensor,
        swa_lengths: torch.Tensor,
        indexed_indices: torch.Tensor | None = None,
        indexed_lengths: torch.Tensor | None = None,
        indexed_page_table: torch.Tensor | None = None,
    ) -> "SM12XCompressedMLABinding":
        return build_compressed_mla_binding(
            scratch=self,
            q=q,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
            indexed_page_table=indexed_page_table,
        )


@dataclass(frozen=True, kw_only=True)
class SM12XCompressedMLABinding:
    scratch: object
    q: torch.Tensor
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor
    indexed_indices: torch.Tensor | None = None
    indexed_lengths: torch.Tensor | None = None
    indexed_page_table: torch.Tensor | None = None


def _compressed_mla_scratch_layout(
    caps: SM12XCompressedMLAScratchCaps,
) -> _SM12XCompressedMLAScratchLayout:
    max_total_q = max(int(caps.max_q_rows), 1)
    max_chunks_per_row = max(int(caps.max_chunks_per_row), 1)
    default_q_chunks = max_total_q * max_chunks_per_row
    max_q_chunks = (
        default_q_chunks
        if caps.max_q_chunks is None
        else max(int(caps.max_q_chunks), default_q_chunks)
    )

    cursor = 0
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
    tmp_output_offset_bytes = cursor
    cursor += (
        max_q_chunks
        * int(caps.num_q_heads)
        * int(caps.v_head_dim)
        * dtype_nbytes(caps.dtype)
    )
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    tmp_lse_offset_bytes = cursor
    cursor += max_q_chunks * int(caps.num_q_heads) * dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    final_lse_offset_bytes = cursor
    cursor += max_total_q * int(caps.num_q_heads) * dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    kv_chunk_size_offset_bytes = cursor
    cursor += dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    num_chunks_offset_bytes = cursor
    cursor += dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    sm_scale_offset_bytes = cursor
    cursor += dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    return _SM12XCompressedMLAScratchLayout(
        nbytes=max(int(cursor), SCRATCH_ALIGN_BYTES),
        max_q_chunks=max_q_chunks,
        tmp_output_offset_bytes=tmp_output_offset_bytes,
        tmp_lse_offset_bytes=tmp_lse_offset_bytes,
        final_lse_offset_bytes=final_lse_offset_bytes,
        kv_chunk_size_offset_bytes=kv_chunk_size_offset_bytes,
        num_chunks_offset_bytes=num_chunks_offset_bytes,
        sm_scale_offset_bytes=sm_scale_offset_bytes,
    )


def _shape_only_scratch_tensor(
    scratch: torch.Tensor,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    base = scratch.narrow(0, 0, dtype_nbytes(dtype)).view(dtype)
    return base.as_strided(shape, (0,) * len(shape))


def _install_compressed_mla_contract_phantoms(
    scratch: SM12XCompressedMLAScratch,
) -> None:
    storage = scratch.shared_scratch
    scratch._contract_q = _shape_only_scratch_tensor(
        storage,
        (
            int(scratch.max_total_q),
            int(scratch.num_q_heads),
            int(scratch.head_dim) // 4,
        ),
        dtype=torch.uint32,
    )
    scratch._contract_page_table = _shape_only_scratch_tensor(
        storage,
        (int(scratch.max_total_q), int(scratch.topk)),
        dtype=torch.int32,
    )
    scratch._contract_indexer_cache_seqlens = _shape_only_scratch_tensor(
        storage,
        (int(scratch.max_total_q),),
        dtype=torch.int32,
    )
    scratch._contract_output = _shape_only_scratch_tensor(
        storage,
        (
            int(scratch.max_total_q),
            int(scratch.num_q_heads),
            int(scratch.v_head_dim),
        ),
        dtype=scratch.dtype,
    )
    scratch._contract_tmp_output = _shape_only_scratch_tensor(
        storage,
        (
            int(scratch.max_total_q),
            int(scratch.num_q_heads),
            int(scratch.max_chunks_per_row),
            int(scratch.v_head_dim),
        ),
        dtype=scratch.dtype,
    )
    scratch._contract_tmp_lse = _shape_only_scratch_tensor(
        storage,
        (
            int(scratch.max_total_q),
            int(scratch.num_q_heads),
            int(scratch.max_chunks_per_row),
        ),
        dtype=torch.float32,
    )


def _materialize_compressed_mla_scratch(
    caps: SM12XCompressedMLAScratchCaps,
    scratch_storage: torch.Tensor,
    layout: _SM12XCompressedMLAScratchLayout,
) -> SM12XCompressedMLAScratch:
    max_total_q = max(int(caps.max_q_rows), 1)
    tmp_output, _ = materialize_scratch_strided_view(
        scratch_storage,
        offset_bytes=layout.tmp_output_offset_bytes,
        shape=(
            max_total_q,
            int(caps.num_q_heads),
            int(caps.max_chunks_per_row),
            int(caps.v_head_dim),
        ),
        stride=_split_tmp_output_stride(
            max_total_q=max_total_q,
            num_q_heads=int(caps.num_q_heads),
            max_chunks_per_row=int(caps.max_chunks_per_row),
            v_head_dim=int(caps.v_head_dim),
        ),
        dtype=caps.dtype,
    )
    tmp_lse, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.tmp_lse_offset_bytes,
        shape=(max_total_q, int(caps.num_q_heads), int(caps.max_chunks_per_row)),
        dtype=torch.float32,
    )
    final_lse, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.final_lse_offset_bytes,
        shape=(max_total_q, int(caps.num_q_heads)),
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
    sm_scale_tensor, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.sm_scale_offset_bytes,
        shape=(1,),
        dtype=torch.float32,
    )
    scratch = SM12XCompressedMLAScratch(
        shared_scratch=scratch_storage,
        device=caps.device,
        dtype=caps.dtype,
        kv_dtype=caps.kv_dtype,
        num_q_heads=caps.num_q_heads,
        head_dim=caps.head_dim,
        v_head_dim=caps.v_head_dim,
        topk=caps.max_width,
        max_page_table_width=caps.max_page_table_width,
        max_total_q=caps.max_q_rows,
        max_batch=caps.max_batch,
        max_kv_rows=caps.max_kv_rows,
        max_chunks_per_row=caps.max_chunks_per_row,
        page_size=caps.page_size,
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        output_buffer=_split_output_buffer_from_tmp(tmp_output),
        final_lse=final_lse,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        num_chunks_ptr=num_chunks_ptr,
        sm_scale_tensor=sm_scale_tensor,
    )
    _install_compressed_mla_contract_phantoms(scratch)
    split_cfg = compressed_mla_split_config_for_contract(
        rows=caps.max_q_rows,
        width=caps.max_width,
        max_chunks=caps.max_chunks_per_row,
    )
    scratch.set_split_chunk_config(
        kv_chunk_size=split_cfg.chunk_size,
        num_chunks=split_cfg.num_chunks,
    )
    return scratch


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
            f"{name} device {tensor.device} does not match scratch device {scratch.device}"
        )


def _normalize_q(q: torch.Tensor, *, scratch: object) -> torch.Tensor:
    if q.ndim == 4 and q.shape[1] == 1:
        q = q[:, 0]
    if q.ndim != 3:
        raise ValueError(
            f"q must be rank-3 or [rows, 1, heads, dim], got {tuple(q.shape)}"
        )
    if int(q.shape[1]) != int(scratch.num_q_heads):
        raise ValueError(
            f"q heads {int(q.shape[1])} do not match scratch heads {scratch.num_q_heads}"
        )
    if int(q.shape[2]) != COMPRESSED_MLA_HEAD_DIM:
        raise ValueError(
            f"q head_dim must be {COMPRESSED_MLA_HEAD_DIM}, got {int(q.shape[2])}"
        )
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must have dtype torch.bfloat16, got {q.dtype}")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    _validate_device(q, scratch=scratch, name="q")
    if int(q.shape[0]) > int(scratch.max_total_q):
        raise ValueError(
            f"q rows {int(q.shape[0])} exceed scratch capacity {scratch.max_total_q}"
        )
    return q.detach()


def _is_row_shared_i32_matrix(tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim == 2 and int(tensor.stride(0)) == 0 and int(tensor.stride(1)) == 1
    )


def _normalize_i32_matrix(
    tensor: torch.Tensor,
    *,
    scratch: object,
    rows: int,
    name: str,
    allow_row_shared: bool = False,
) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 2:
        raise ValueError(
            f"{name} must be rank-2 or [rows, 1, width], got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if not tensor.is_contiguous() and not (
        allow_row_shared and _is_row_shared_i32_matrix(tensor)
    ):
        raise ValueError(f"{name} must be contiguous")
    _validate_device(tensor, scratch=scratch, name=name)
    if int(tensor.shape[0]) != int(rows):
        raise ValueError(
            f"{name} rows {int(tensor.shape[0])} do not match q rows {rows}"
        )
    return tensor


def _validate_i32_vector(
    tensor: torch.Tensor, *, scratch: object, rows: int, name: str
) -> torch.Tensor:
    if tensor.shape != (int(rows),):
        raise ValueError(f"{name} must have shape ({rows},), got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    _validate_device(tensor, scratch=scratch, name=name)
    return tensor


def build_compressed_mla_binding(
    *,
    scratch: object,
    q: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    indexed_indices: torch.Tensor | None = None,
    indexed_lengths: torch.Tensor | None = None,
    indexed_page_table: torch.Tensor | None = None,
) -> SM12XCompressedMLABinding:
    q = _normalize_q(q, scratch=scratch)
    rows = int(q.shape[0])
    swa_indices = _normalize_i32_matrix(
        swa_indices,
        scratch=scratch,
        rows=rows,
        name="swa_indices",
    )
    if int(swa_indices.shape[1]) > int(scratch.topk):
        raise ValueError(
            f"swa_indices width {int(swa_indices.shape[1])} exceeds scratch topk {scratch.topk}"
        )
    swa_lengths = _validate_i32_vector(
        swa_lengths,
        scratch=scratch,
        rows=rows,
        name="swa_lengths",
    )
    if (indexed_indices is None) != (indexed_lengths is None):
        raise ValueError(
            "indexed_indices and indexed_lengths must be provided together"
        )
    indexed_width = 0
    if indexed_indices is not None:
        indexed_indices = _normalize_i32_matrix(
            indexed_indices,
            scratch=scratch,
            rows=rows,
            name="indexed_indices",
        )
        indexed_width = int(indexed_indices.shape[1])
        indexed_lengths = _validate_i32_vector(
            indexed_lengths,  # type: ignore[arg-type]
            scratch=scratch,
            rows=rows,
            name="indexed_lengths",
        )
    if indexed_page_table is not None:
        indexed_page_table = _normalize_i32_matrix(
            indexed_page_table,
            scratch=scratch,
            rows=rows,
            name="indexed_page_table",
            allow_row_shared=True,
        )
        if int(indexed_page_table.shape[1]) > int(scratch.max_page_table_width):
            raise ValueError(
                "indexed_page_table width "
                f"{int(indexed_page_table.shape[1])} exceeds scratch capacity {scratch.max_page_table_width}"
            )
    total_width = int(swa_indices.shape[1]) + indexed_width
    if total_width > int(scratch.topk):
        raise ValueError(
            f"compressed MLA width {total_width} exceeds scratch topk {scratch.topk}"
        )
    return SM12XCompressedMLABinding(
        scratch=scratch,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        indexed_page_table=indexed_page_table,
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


@dataclass(frozen=True)
class SM12XCompressedMLAScratchPlan:
    caps: SM12XCompressedMLAScratchCaps
    layout: _SM12XCompressedMLAScratchLayout
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
        swa_indices: torch.Tensor,
        swa_lengths: torch.Tensor,
        indexed_indices: torch.Tensor | None = None,
        indexed_lengths: torch.Tensor | None = None,
        indexed_page_table: torch.Tensor | None = None,
    ) -> SM12XCompressedMLABinding:
        scratch_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="compressed MLA",
        )
        scratch_views = _materialize_compressed_mla_scratch(
            self.caps,
            scratch_storage,
            self.layout,
        )
        return build_compressed_mla_binding(
            scratch=scratch_views,
            q=q,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
            indexed_page_table=indexed_page_table,
        )


def plan_compressed_mla_scratch(
    caps: SM12XCompressedMLAScratchCaps,
) -> SM12XCompressedMLAScratchPlan:
    layout = _compressed_mla_scratch_layout(caps)
    return SM12XCompressedMLAScratchPlan(
        caps=caps,
        layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "compressed_mla.scratch",
                nbytes=int(layout.nbytes),
                device=caps.device,
            ),
        ),
    )


__all__ = [
    "ScratchBufferSpec",
    "SM12XCompressedMLABinding",
    "SM12XCompressedMLAScratch",
    "SM12XCompressedMLAScratchCaps",
    "SM12XCompressedMLAScratchPlan",
    "build_compressed_mla_binding",
    "plan_compressed_mla_scratch",
]
