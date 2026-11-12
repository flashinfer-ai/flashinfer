"""Caller-owned scratch plans for sparse MLA paths.

Eager PLAN -> BIND -> KERNEL, never a workspace/arena. bind() maps the
caller-owned scratch tensor into per-spec kernel-argument VIEWS and returns a
plain B12XSparseMLAScratch views container (mirroring B12XCompressedSparseMLAScratch).
It never constructs a B12XAttentionWorkspace / arena, allocates, or init-writes.
The unified SM120 sparse-MLA decode/extend kernels duck-type the workspace
(tmp_output/tmp_lse/output_buffer/final_lse/num_chunks_ptr/kv_chunk_size_ptr/
set_split_chunk_config/...), so the views container is a drop-in -- no kernel
signature change.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch

from b12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)
from b12x._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    dtype_nbytes,
    materialize_scratch_strided_view,
    materialize_scratch_view,
)
from b12x.attention._shared.mla.traits import (
    ModelType,
    UnifiedMLATraits,
    infer_model_type,
    make_unified_traits,
)
from b12x.attention._shared.workspace import (
    _split_output_buffer_from_tmp,
    _split_tmp_output_stride,
)


@dataclass(frozen=True, kw_only=True)
class B12XSparseMLAScratchCaps:
    device: torch.device | str
    num_q_heads: int
    max_q_rows: int
    max_width: int
    softmax_scale: float
    dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype = torch.bfloat16
    head_dim: int = 576
    v_head_dim: int = 512
    model_type: int | None = None
    scale_format: int | None = None
    cache_record_bytes: int | None = None
    fp8_rope: bool | None = None
    latent_scale_per_token: bool = False
    cache_traits: UnifiedMLATraits | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    mode: Literal["decode", "extend", "verify", "draft_extend"] = "decode"
    max_batch: int | None = None
    max_kv_rows: int = 0
    max_page_table_width: int | None = None
    max_chunks_per_row: int = 64
    max_q_chunks: int | None = None
    page_size: int = 64
    head_major_output: bool = False
    latent_scale: float = 1.0
    return_lse: bool = False
    lse_scale: Literal["base2", "natural"] = "base2"
    has_attention_sink: bool = False

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "num_q_heads", max(int(self.num_q_heads), 1))
        object.__setattr__(self, "max_q_rows", max(int(self.max_q_rows), 1))
        object.__setattr__(self, "max_width", max(int(self.max_width), 1))
        softmax_scale = float(self.softmax_scale)
        if not math.isfinite(softmax_scale) or softmax_scale <= 0.0:
            raise ValueError(
                f"softmax_scale must be finite and positive, got {softmax_scale}"
            )
        object.__setattr__(self, "softmax_scale", softmax_scale)
        object.__setattr__(self, "head_dim", max(int(self.head_dim), 1))
        object.__setattr__(self, "v_head_dim", max(int(self.v_head_dim), 1))
        requested_model_type = self.model_type
        if requested_model_type is not None:
            requested_model_type = int(requested_model_type)
            if requested_model_type == int(ModelType.GLM_NEXT):
                if int(self.head_dim) != 512 or int(self.v_head_dim) != 512:
                    raise ValueError(
                        "ModelType.GLM_NEXT requires head_dim=512 and "
                        f"v_head_dim=512; got {self.head_dim} and {self.v_head_dim}"
                    )
        model_type, compute_mode, default_scale_format = infer_model_type(
            int(self.head_dim),
            self.kv_dtype,
            model_type=requested_model_type,
        )
        scale_format = (
            int(default_scale_format)
            if self.scale_format is None
            else int(self.scale_format)
        )
        traits = make_unified_traits(
            model_type,
            compute_mode,
            scale_format,
            fp8_rope=self.fp8_rope,
            latent_scale_per_token=bool(self.latent_scale_per_token),
        )
        record_bytes = int(traits.kv_gmem_stride)
        if (
            self.cache_record_bytes is not None
            and int(self.cache_record_bytes) != record_bytes
        ):
            raise ValueError(
                "sparse MLA cache_record_bytes does not match its planned "
                f"recipe: got {int(self.cache_record_bytes)}, expected {record_bytes}"
            )
        object.__setattr__(self, "model_type", int(traits.model_type))
        object.__setattr__(self, "scale_format", int(traits.scale_format))
        object.__setattr__(self, "cache_record_bytes", record_bytes)
        object.__setattr__(self, "fp8_rope", bool(traits.fp8_rope))
        object.__setattr__(
            self,
            "latent_scale_per_token",
            bool(traits.latent_scale_per_token),
        )
        object.__setattr__(self, "cache_traits", traits)
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
        latent_scale = float(self.latent_scale)
        if not math.isfinite(latent_scale) or latent_scale <= 0.0:
            raise ValueError(
                f"latent_scale must be finite and positive, got {latent_scale}"
            )
        object.__setattr__(self, "latent_scale", latent_scale)
        object.__setattr__(self, "return_lse", bool(self.return_lse))
        if self.lse_scale not in ("base2", "natural"):
            raise ValueError(
                f"lse_scale must be 'base2' or 'natural', got {self.lse_scale!r}"
            )
        object.__setattr__(self, "has_attention_sink", bool(self.has_attention_sink))


@dataclass(kw_only=True)
class B12XSparseMLAScratch:
    """Component-owned sparse-MLA scratch VIEWS over caller-owned storage.

    Exposes exactly the attributes the unified SM120 sparse-MLA decode/extend
    kernels duck-type off the (former) workspace. NEVER a B12XAttentionWorkspace.
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
    head_major_output: bool = False
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
    model_type: int | None = None
    scale_format: int | None = None
    cache_record_bytes: int | None = None
    fp8_rope: bool = False
    latent_scale_per_token: bool = False
    cache_traits: UnifiedMLATraits | None = None

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
        kv_cache: torch.Tensor | None = None,
    ) -> "B12XSparseMLABinding":
        return build_sparse_mla_binding(
            scratch=self,
            q=q,
            selected_indices=selected_indices,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
            kv_cache=kv_cache,
        )


@dataclass(frozen=True, kw_only=True)
class B12XSparseMLABinding:
    scratch: object
    q: torch.Tensor
    selected_indices: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    nsa_cache_seqlens_int32: torch.Tensor
    model_type: int | None = None
    scale_format: int | None = None
    cache_record_bytes: int | None = None
    fp8_rope: bool | None = None
    latent_scale_per_token: bool | None = None
    cache_traits: UnifiedMLATraits | None = None
    kv_cache: torch.Tensor | None = None


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
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
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
    selected_width = int(selected_indices.shape[1])
    planned_width = int(scratch.topk)
    if selected_width > planned_width:
        raise ValueError(
            f"selected_indices width {selected_width} exceeds scratch topk {planned_width}"
        )
    model_type = getattr(scratch, "model_type", None)
    if model_type in (ModelType.GLM_NSA, ModelType.GLM_NEXT) and (
        selected_width != planned_width
    ):
        raise ValueError(
            "GLM sparse MLA selected_indices width must match the planned "
            f"top-k width {planned_width}, got {selected_width}; carry the live "
            "valid count in nsa_cache_seqlens_int32"
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
    kv_cache: torch.Tensor | None = None,
) -> B12XSparseMLABinding:
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
    if kv_cache is not None:
        if kv_cache.ndim != 3:
            raise ValueError(f"kv_cache must be rank-3, got {tuple(kv_cache.shape)}")
        _validate_device(kv_cache, scratch=scratch, name="kv_cache")
        if kv_cache.dtype != scratch.kv_dtype:
            raise TypeError(
                f"kv_cache must have dtype {scratch.kv_dtype}, got {kv_cache.dtype}"
            )
        if int(kv_cache.shape[-1]) != int(scratch.cache_record_bytes):
            raise ValueError(
                "kv_cache record width does not match the sparse MLA plan: "
                f"got {int(kv_cache.shape[-1])}, expected "
                f"{int(scratch.cache_record_bytes)}"
            )
        # Contiguous caches are flat record storage; their first two dimensions
        # may be reshaped without changing the address contract. A strided cache
        # encodes physical page boundaries and must match the planned page size.
        if not kv_cache.is_contiguous() and int(kv_cache.shape[1]) != int(
            scratch.page_size
        ):
            raise ValueError(
                "kv_cache page size does not match the sparse MLA plan: "
                f"got {int(kv_cache.shape[1])}, expected {int(scratch.page_size)}"
            )
    return B12XSparseMLABinding(
        scratch=scratch,
        q=q,
        selected_indices=selected_indices,
        cache_seqlens_int32=cache_seqlens_int32,
        nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        model_type=getattr(scratch, "model_type", None),
        scale_format=getattr(scratch, "scale_format", None),
        cache_record_bytes=getattr(scratch, "cache_record_bytes", None),
        fp8_rope=getattr(scratch, "fp8_rope", None),
        latent_scale_per_token=getattr(scratch, "latent_scale_per_token", None),
        cache_traits=getattr(scratch, "cache_traits", None),
        kv_cache=kv_cache,
    )


@dataclass(frozen=True)
class _B12XSparseMLAScratchLayout:
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
    caps: B12XSparseMLAScratchCaps,
) -> _B12XSparseMLAScratchLayout:
    max_total_q = max(int(caps.max_q_rows), 1)
    num_q_heads = int(caps.num_q_heads)
    v_head_dim = int(caps.v_head_dim)
    max_chunks_per_row = max(int(caps.max_chunks_per_row), 1)
    # Only the split-K DECODE path needs tmp_output/tmp_lse + split control.
    # Every mode owns output and final-LSE views so prefill never allocates its
    # LSE output during capture.
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
        final_lse_offset_bytes = cursor
        cursor += max_total_q * num_q_heads * dtype_nbytes(torch.float32)
        cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    sm_scale_offset_bytes = cursor
    cursor += dtype_nbytes(torch.float32)
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)

    return _B12XSparseMLAScratchLayout(
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
    caps: B12XSparseMLAScratchCaps,
    scratch_storage: torch.Tensor,
    layout: _B12XSparseMLAScratchLayout,
) -> B12XSparseMLAScratch:
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
                head_major_output=caps.head_major_output,
            ),
            dtype=caps.dtype,
        )
        output_buffer = _split_output_buffer_from_tmp(
            tmp_output,
            head_major_output=caps.head_major_output,
        )
        tmp_lse, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.tmp_lse_offset_bytes,
            shape=(max_total_q, num_q_heads, max_chunks_per_row),
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
        if caps.head_major_output:
            output_buffer, _ = materialize_scratch_strided_view(
                scratch_storage,
                offset_bytes=layout.output_offset_bytes,
                shape=(max_total_q, num_q_heads, v_head_dim),
                stride=(v_head_dim, max_total_q * v_head_dim, 1),
                dtype=caps.dtype,
            )
        else:
            output_buffer, _ = materialize_scratch_view(
                scratch_storage,
                offset_bytes=layout.output_offset_bytes,
                shape=(max_total_q, num_q_heads, v_head_dim),
                dtype=caps.dtype,
            )

    final_lse, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.final_lse_offset_bytes,
        shape=(max_total_q, num_q_heads),
        dtype=torch.float32,
    )

    sm_scale_tensor, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.sm_scale_offset_bytes,
        shape=(1,),
        dtype=torch.float32,
    )

    scratch = B12XSparseMLAScratch(
        shared_scratch=scratch_storage,
        device=caps.device,
        dtype=caps.dtype,
        kv_dtype=caps.kv_dtype,
        num_q_heads=num_q_heads,
        head_dim=caps.head_dim,
        v_head_dim=v_head_dim,
        model_type=caps.model_type,
        scale_format=caps.scale_format,
        cache_record_bytes=caps.cache_record_bytes,
        fp8_rope=bool(caps.fp8_rope),
        latent_scale_per_token=bool(caps.latent_scale_per_token),
        cache_traits=caps.cache_traits,
        topk=caps.max_width,
        max_total_q=caps.max_q_rows,
        max_batch=caps.max_batch,
        max_chunks_per_row=max_chunks_per_row,
        page_size=caps.page_size,
        mode=caps.mode,
        head_major_output=caps.head_major_output,
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
class B12XSparseMLAScratchPlan:
    caps: B12XSparseMLAScratchCaps
    layout: _B12XSparseMLAScratchLayout
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    # Materialized layout only; declarations live in _preparation.

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
        kv_cache: torch.Tensor | None = None,
    ) -> B12XSparseMLABinding:
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
            kv_cache=kv_cache,
        )


def plan_sparse_mla_scratch(
    caps: B12XSparseMLAScratchCaps,
) -> B12XSparseMLAScratchPlan:
    """Build the caller-owned scratch layout for an already selected route."""
    if not isinstance(caps, B12XSparseMLAScratchCaps):
        raise TypeError("caps must be B12XSparseMLAScratchCaps")
    layout = _sparse_mla_scratch_layout(caps)
    return B12XSparseMLAScratchPlan(
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
    "B12XSparseMLABinding",
    "B12XSparseMLAScratch",
    "B12XSparseMLAScratchCaps",
    "B12XSparseMLAScratchPlan",
    "build_sparse_mla_binding",
    "plan_sparse_mla_scratch",
]
