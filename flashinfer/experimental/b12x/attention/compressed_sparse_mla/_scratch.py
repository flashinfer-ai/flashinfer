"""Caller-owned scratch plans for compressed MLA paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from b12x.preparation import Plan

from b12x.attention._shared.mla.compressed_config import (
    compressed_sparse_mla_split_config_for_contract,
)
from b12x.attention._shared.mla.compressed_reference import (
    COMPRESSED_SPARSE_MLA_HEAD_DIM,
)
from b12x.attention._shared.workspace import (
    _split_output_buffer_from_tmp,
    _split_tmp_output_stride,
)
from b12x._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    dtype_nbytes,
    materialize_scratch_strided_view,
    materialize_scratch_view,
)
from b12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)


@dataclass(frozen=True, kw_only=True)
class B12XCompressedSparseMLAScratchCaps:
    device: torch.device | str
    num_q_heads: int
    max_q_rows: int
    max_width: int
    max_page_table_width: int | None = None
    dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype = torch.uint8
    head_dim: int = COMPRESSED_SPARSE_MLA_HEAD_DIM
    v_head_dim: int = COMPRESSED_SPARSE_MLA_HEAD_DIM
    max_batch: int | None = None
    max_kv_rows: int = 0
    max_chunks_per_row: int | None = None
    max_q_chunks: int | None = None
    decode_row_capacity: int | None = None
    page_size: int = 64
    layout: str = "compressed_dsv4"
    cache_format: Literal["deepseek_v4", "deepseek_v41"] = "deepseek_v4"
    mode: Literal["decode", "extend", "verify", "draft_extend"] = "decode"
    swa_width: int | None = None
    indexed_width: int | None = None
    swa_page_size: int | None = None
    indexed_page_size: int | None = None
    shared_width_capacity: bool | None = field(default=None, repr=False)
    use_cuda_graph: bool = False

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "num_q_heads", max(int(self.num_q_heads), 1))
        object.__setattr__(self, "max_q_rows", max(int(self.max_q_rows), 1))
        object.__setattr__(self, "max_width", max(int(self.max_width), 1))
        if self.layout != "compressed_dsv4":
            raise ValueError(f"unsupported compressed sparse MLA layout {self.layout!r}")
        if self.cache_format not in ("deepseek_v4", "deepseek_v41"):
            raise ValueError(f"unsupported compressed MLA cache_format {self.cache_format!r}")
        if self.mode not in ("decode", "extend", "verify", "draft_extend"):
            raise ValueError(f"unsupported compressed sparse MLA mode {self.mode!r}")
        legacy_shared_width = (
            self.swa_width is None and self.indexed_width is None
            if self.shared_width_capacity is None
            else bool(self.shared_width_capacity)
        )
        if legacy_shared_width:
            if self.swa_width is None and self.indexed_width is None:
                swa_width = indexed_width = self.max_width
            else:
                swa_width = int(self.swa_width)
                indexed_width = int(self.indexed_width)
                if swa_width != self.max_width or indexed_width != self.max_width:
                    raise ValueError(
                        "shared width capacity must equal max_width for both routes"
                    )
        elif self.swa_width is None:
            indexed_width = int(self.indexed_width)
            swa_width = self.max_width - indexed_width
        elif self.indexed_width is None:
            swa_width = int(self.swa_width)
            indexed_width = self.max_width - swa_width
        else:
            swa_width = int(self.swa_width)
            indexed_width = int(self.indexed_width)
        if swa_width < 0 or indexed_width < 0:
            raise ValueError("swa_width and indexed_width must be non-negative")
        if not legacy_shared_width and swa_width + indexed_width != self.max_width:
            raise ValueError(
                "swa_width + indexed_width must equal max_width"
            )
        object.__setattr__(self, "swa_width", swa_width)
        object.__setattr__(self, "indexed_width", indexed_width)
        object.__setattr__(self, "shared_width_capacity", legacy_shared_width)
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
        if self.max_chunks_per_row is not None:
            object.__setattr__(
                self,
                "max_chunks_per_row",
                max(int(self.max_chunks_per_row), 1),
            )
        if self.max_q_chunks is not None:
            object.__setattr__(self, "max_q_chunks", max(int(self.max_q_chunks), 1))
        if self.decode_row_capacity is not None:
            decode_row_capacity = int(self.decode_row_capacity)
            if decode_row_capacity <= 0:
                raise ValueError(
                    f"decode_row_capacity must be positive, got {decode_row_capacity}"
                )
            object.__setattr__(self, "decode_row_capacity", decode_row_capacity)
        swa_page_size = (
            self.page_size if self.swa_page_size is None else self.swa_page_size
        )
        swa_page_size = max(int(swa_page_size), 1)
        indexed_page_size = (
            swa_page_size
            if self.indexed_page_size is None
            else max(int(self.indexed_page_size), 1)
        )
        object.__setattr__(self, "page_size", swa_page_size)
        object.__setattr__(self, "swa_page_size", swa_page_size)
        object.__setattr__(self, "indexed_page_size", indexed_page_size)
        object.__setattr__(self, "use_cuda_graph", bool(self.use_cuda_graph))


@dataclass(frozen=True, kw_only=True)
class _B12XCompressedSparseMLAScratchLayout:
    nbytes: int
    max_q_chunks: int
    tmp_output_offset_bytes: int
    tmp_lse_offset_bytes: int
    final_lse_offset_bytes: int
    kv_chunk_size_offset_bytes: int
    num_chunks_offset_bytes: int
    sm_scale_offset_bytes: int
    mapped_indices_offset_bytes: int


@dataclass(kw_only=True)
class B12XCompressedSparseMLAScratch:
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
    max_swa_width: int
    max_indexed_width: int
    indexed_page_size: int
    layout: str
    mode: str = "decode"
    cache_format: str = "deepseek_v4"
    execution_config: object | None = None
    fixed_capacity: bool = True
    use_cuda_graph: bool = False
    tmp_output: torch.Tensor | None = None
    tmp_lse: torch.Tensor | None = None
    output_buffer: torch.Tensor | None = None
    final_lse: torch.Tensor | None = None
    kv_chunk_size_ptr: torch.Tensor | None = None
    num_chunks_ptr: torch.Tensor | None = None
    sm_scale_tensor: torch.Tensor | None = None
    mapped_indices: torch.Tensor | None = None
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
    ) -> "B12XCompressedSparseMLABinding":
        return build_compressed_sparse_mla_binding(
            scratch=self,
            q=q,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
            indexed_page_table=indexed_page_table,
        )


@dataclass(frozen=True, kw_only=True)
class B12XCompressedSparseMLABinding:
    """Caller-owned dynamic inputs bound to one prepared compressed MLA state."""

    scratch: object
    q: torch.Tensor
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor
    indexed_indices: torch.Tensor | None = None
    indexed_lengths: torch.Tensor | None = None
    indexed_page_table: torch.Tensor | None = None
    plan: Plan | None = None

def _compressed_sparse_mla_scratch_layout(
    caps: B12XCompressedSparseMLAScratchCaps,
) -> _B12XCompressedSparseMLAScratchLayout:
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

    mapped_indices_offset_bytes = cursor
    cursor += max_total_q * int(caps.indexed_width) * dtype_nbytes(torch.int32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    return _B12XCompressedSparseMLAScratchLayout(
        nbytes=max(int(cursor), SCRATCH_ALIGN_BYTES),
        max_q_chunks=max_q_chunks,
        tmp_output_offset_bytes=tmp_output_offset_bytes,
        tmp_lse_offset_bytes=tmp_lse_offset_bytes,
        final_lse_offset_bytes=final_lse_offset_bytes,
        kv_chunk_size_offset_bytes=kv_chunk_size_offset_bytes,
        num_chunks_offset_bytes=num_chunks_offset_bytes,
        sm_scale_offset_bytes=sm_scale_offset_bytes,
        mapped_indices_offset_bytes=mapped_indices_offset_bytes,
    )


def _shape_only_scratch_tensor(
    scratch: torch.Tensor,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    base = scratch.narrow(0, 0, dtype_nbytes(dtype)).view(dtype)
    return base.as_strided(shape, (0,) * len(shape))


def _install_compressed_sparse_mla_contract_phantoms(
    scratch: B12XCompressedSparseMLAScratch,
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


def _materialize_compressed_sparse_mla_scratch(
    caps: B12XCompressedSparseMLAScratchCaps,
    scratch_storage: torch.Tensor,
    layout: _B12XCompressedSparseMLAScratchLayout,
    execution_config: object | None,
) -> B12XCompressedSparseMLAScratch:
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
    mapped_indices, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.mapped_indices_offset_bytes,
        shape=(max_total_q, int(caps.indexed_width)),
        dtype=torch.int32,
    )
    sm_scale_tensor, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.sm_scale_offset_bytes,
        shape=(1,),
        dtype=torch.float32,
    )
    scratch = B12XCompressedSparseMLAScratch(
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
        max_swa_width=caps.swa_width,
        max_indexed_width=caps.indexed_width,
        cache_format=caps.cache_format,
        execution_config=execution_config,
        mapped_indices=mapped_indices,
        indexed_page_size=caps.indexed_page_size,
        layout=caps.layout,
        mode=caps.mode,
        use_cuda_graph=caps.use_cuda_graph,
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        output_buffer=_split_output_buffer_from_tmp(tmp_output),
        final_lse=final_lse,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        num_chunks_ptr=num_chunks_ptr,
        sm_scale_tensor=sm_scale_tensor,
    )
    _install_compressed_sparse_mla_contract_phantoms(scratch)
    split_cfg = compressed_sparse_mla_split_config_for_contract(
        rows=caps.max_q_rows,
        width=caps.max_width,
        max_chunks=caps.max_chunks_per_row,
        decode_row_capacity=caps.decode_row_capacity,
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
    if int(q.shape[2]) != COMPRESSED_SPARSE_MLA_HEAD_DIM:
        raise ValueError(
            f"q head_dim must be {COMPRESSED_SPARSE_MLA_HEAD_DIM}, got {int(q.shape[2])}"
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


def build_compressed_sparse_mla_binding(
    *,
    scratch: object,
    q: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    indexed_indices: torch.Tensor | None = None,
    indexed_lengths: torch.Tensor | None = None,
    indexed_page_table: torch.Tensor | None = None,
) -> B12XCompressedSparseMLABinding:
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
    max_swa_width = int(getattr(scratch, "max_swa_width", scratch.topk))
    if int(swa_indices.shape[1]) > max_swa_width:
        raise ValueError(
            f"swa_indices width {int(swa_indices.shape[1])} exceeds "
            f"planned SWA width {max_swa_width}"
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
        max_indexed_width = int(
            getattr(scratch, "max_indexed_width", scratch.topk)
        )
        if indexed_width > max_indexed_width:
            raise ValueError(
                f"indexed_indices width {indexed_width} exceeds planned "
                f"indexed width {max_indexed_width}"
            )
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
    return B12XCompressedSparseMLABinding(
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
class B12XCompressedSparseMLAScratchPlan:
    caps: B12XCompressedSparseMLAScratchCaps
    layout: _B12XCompressedSparseMLAScratchLayout
    execution_config: object | None
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
    ) -> B12XCompressedSparseMLABinding:
        scratch_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="compressed MLA",
        )
        scratch_views = _materialize_compressed_sparse_mla_scratch(
            self.caps,
            scratch_storage,
            self.layout,
            self.execution_config,
        )
        return build_compressed_sparse_mla_binding(
            scratch=scratch_views,
            q=q,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
            indexed_page_table=indexed_page_table,
        )


def plan_compressed_sparse_mla_scratch(
    caps: B12XCompressedSparseMLAScratchCaps,
    *,
    execution_config: object | None = None,
) -> B12XCompressedSparseMLAScratchPlan:
    if caps.max_chunks_per_row is None:
        caps = replace(
            caps,
            max_chunks_per_row=getattr(execution_config, "max_chunks_per_row", 64),
        )
    elif (
        execution_config is not None
        and caps.max_chunks_per_row != getattr(execution_config, "max_chunks_per_row", None)
    ):
        raise ValueError("caps.max_chunks_per_row must match execution_config")
    layout = _compressed_sparse_mla_scratch_layout(caps)
    return B12XCompressedSparseMLAScratchPlan(
        caps=caps,
        layout=layout,
        execution_config=execution_config,
        _scratch_specs=(
            scratch_buffer_spec(
                "compressed_sparse_mla.scratch",
                nbytes=int(layout.nbytes),
                device=caps.device,
            ),
        ),
    )


__all__ = [
    "ScratchBufferSpec",
    "B12XCompressedSparseMLABinding",
    "B12XCompressedSparseMLAScratch",
    "B12XCompressedSparseMLAScratchCaps",
    "B12XCompressedSparseMLAScratchPlan",
    "build_compressed_sparse_mla_binding",
    "plan_compressed_sparse_mla_scratch",
]
