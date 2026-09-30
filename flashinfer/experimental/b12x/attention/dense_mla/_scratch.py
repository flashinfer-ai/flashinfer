"""Caller-owned plan/bind storage for dense K3 MLA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    materialize_scratch_view,
)

from .._shared import static_fp8_quant
from .planner import Budget, choose_num_splits
from ._layout import make_smem_layout
from ._reference import (
    K3_ABSORBED_DIM,
    K3_SM_SCALE,
    K3_VALUE_DIM,
)

_FP8 = torch.float8_e4m3fn
_MAX_Q_ROWS = 65_536
_MAX_CACHE_TOKENS = 1_048_576


def _canonical_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _query_tile(caps: Caps) -> int:
    if caps.mode == "decode" or caps.max_batch != 1 or caps.window_size is not None:
        return 1
    if caps.kv_dtype == _FP8 and caps.max_total_q >= 3:
        return 4
    return 2


def _max_attended_tokens(caps: Caps) -> int:
    if caps.window_size is None:
        return caps.max_cache_tokens
    # A local interval can begin and end in partial physical pages.
    return min(caps.max_cache_tokens, caps.window_size + caps.page_size - 1)


@dataclass(frozen=True, kw_only=True)
class Caps:
    device: torch.device | str
    mode: Literal["decode", "extend", "verify"]
    kv_dtype: torch.dtype
    num_q_heads: int
    page_size: int
    max_total_q: int
    max_batch: int
    max_cache_tokens: int
    max_page_table_width: int
    num_cache_pages: int
    dtype: torch.dtype = torch.bfloat16
    q_dtype: torch.dtype | None = None
    head_dim: int = K3_ABSORBED_DIM
    v_head_dim: int = K3_VALUE_DIM
    physical_record_width: int = K3_ABSORBED_DIM
    window_size: int | None = None
    use_cuda_graph: bool = False
    budget: Budget | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _canonical_device(self.device))
        if self.mode not in ("decode", "extend", "verify"):
            raise ValueError(f"unsupported dense MLA mode {self.mode!r}")
        if self.dtype != torch.bfloat16:
            raise TypeError("dense MLA output must be torch.bfloat16")
        if self.kv_dtype not in (torch.bfloat16, _FP8):
            raise TypeError("dense MLA KV must be BF16 or E4M3")
        q_dtype = self.kv_dtype if self.q_dtype is None else self.q_dtype
        if q_dtype not in (torch.bfloat16, _FP8):
            raise TypeError("dense MLA query must be BF16 or E4M3")
        if q_dtype != self.kv_dtype and not (
            q_dtype == torch.bfloat16 and self.kv_dtype == _FP8
        ):
            raise TypeError(
                "dense MLA supports matching query/cache dtypes or a BF16 "
                "query with an E4M3 cache"
            )
        object.__setattr__(self, "q_dtype", q_dtype)
        geometry = (int(self.head_dim), int(self.v_head_dim))
        if geometry not in ((K3_ABSORBED_DIM, K3_VALUE_DIM), (1088, 1024)):
            raise ValueError(
                "dense MLA supports logical (QK, V) widths (576, 512) and (1088, 1024)"
            )
        if int(self.physical_record_width) < int(self.head_dim):
            raise ValueError("physical_record_width must cover the logical key record")
        if self.window_size is not None and int(self.window_size) <= 0:
            raise ValueError("window_size must be positive or None")
        heads = int(self.num_q_heads)
        if heads <= 0:
            raise ValueError("num_q_heads must be positive")
        page_size = int(self.page_size)
        if page_size <= 0 or (page_size != 1 and page_size % 16):
            raise ValueError("page_size must be 1 or a positive multiple of 16")
        max_total_q = int(self.max_total_q)
        if not 1 <= max_total_q <= _MAX_Q_ROWS:
            raise ValueError(f"max_total_q must be in [1, {_MAX_Q_ROWS}]")
        max_batch = int(self.max_batch)
        if not 1 <= max_batch <= max_total_q:
            raise ValueError("max_batch must be in [1, max_total_q]")
        max_cache_tokens = int(self.max_cache_tokens)
        if not 1 <= max_cache_tokens <= _MAX_CACHE_TOKENS:
            raise ValueError(f"max_cache_tokens must be in [1, {_MAX_CACHE_TOKENS}]")
        min_width = (max_cache_tokens + page_size - 1) // page_size
        if int(self.max_page_table_width) < min_width:
            raise ValueError(
                "max_page_table_width does not cover max_cache_tokens: "
                f"need {min_width}, got {self.max_page_table_width}"
            )
        if int(self.num_cache_pages) <= 0:
            raise ValueError("num_cache_pages must be positive")
        if self.budget is not None and not isinstance(self.budget, Budget):
            raise TypeError("budget must be dense_mla.Budget or None")
        for name in (
            "num_q_heads",
            "page_size",
            "max_total_q",
            "max_batch",
            "max_cache_tokens",
            "max_page_table_width",
            "num_cache_pages",
            "head_dim",
            "v_head_dim",
            "physical_record_width",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "use_cuda_graph", bool(self.use_cuda_graph))
        if self.window_size is not None:
            object.__setattr__(self, "window_size", int(self.window_size))


@dataclass(frozen=True)
class _ScratchLayout:
    nbytes: int
    num_splits: int
    chunks_per_split: int
    partial_output_offset_bytes: int | None
    partial_lse_offset_bytes: int | None
    final_lse_offset_bytes: int
    quantized_q_offset_bytes: int | None


def _dense_mla_scratch_layout(
    caps: Caps,
) -> _ScratchLayout:
    query_tile = _query_tile(caps)
    max_attended_tokens = _max_attended_tokens(caps)
    if caps.device.type == "cuda":
        properties = torch.cuda.get_device_properties(caps.device)
        sm_count = int(properties.multi_processor_count)
        shared_layout = make_smem_layout(
            query_tile=query_tile,
            fp8=caps.kv_dtype == _FP8,
            qk_dim=caps.head_dim,
        )
        shared_limit = int(properties.shared_memory_per_block_optin)
        if shared_layout.total_bytes > shared_limit:
            raise ValueError(
                "dense MLA shared-memory specialization exceeds the device "
                f"limit: needs {shared_layout.total_bytes} bytes, device "
                f"permits {shared_limit}"
            )
    else:
        sm_count = 1
    num_splits = choose_num_splits(
        max_cache_tokens=max_attended_tokens,
        max_total_q=caps.max_total_q,
        num_q_heads=caps.num_q_heads,
        query_tile=query_tile,
        sm_count=sm_count,
        budget=caps.budget,
    )
    num_chunks = (max_attended_tokens + 63) // 64
    chunks_per_split = (num_chunks + num_splits - 1) // num_splits
    partial_rows = caps.max_total_q * num_splits
    if (
        caps.budget is not None
        and caps.budget.max_partial_rows is not None
        and num_splits > 1
        and partial_rows > int(caps.budget.max_partial_rows)
    ):
        raise ValueError(
            f"dense MLA needs {partial_rows} partial rows, budget permits "
            f"{caps.budget.max_partial_rows}"
        )

    cursor = 0
    partial_output_offset_bytes = None
    partial_lse_offset_bytes = None
    if num_splits > 1:
        cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
        partial_output_offset_bytes = cursor
        cursor += (
            caps.max_total_q
            * caps.num_q_heads
            * num_splits
            * caps.v_head_dim
            * dtype_nbytes(torch.bfloat16)
        )
        cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
        partial_lse_offset_bytes = cursor
        cursor += (
            caps.max_total_q
            * caps.num_q_heads
            * num_splits
            * dtype_nbytes(torch.float32)
        )

    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
    final_lse_offset_bytes = cursor
    cursor += caps.max_total_q * caps.num_q_heads * dtype_nbytes(torch.float32)
    quantized_q_offset_bytes = None
    if caps.q_dtype != caps.kv_dtype:
        cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
        quantized_q_offset_bytes = cursor
        cursor += (
            caps.max_total_q * caps.num_q_heads * caps.head_dim * dtype_nbytes(_FP8)
        )
    cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
    return _ScratchLayout(
        nbytes=max(cursor, SCRATCH_ALIGN_BYTES),
        num_splits=num_splits,
        chunks_per_split=chunks_per_split,
        partial_output_offset_bytes=partial_output_offset_bytes,
        partial_lse_offset_bytes=partial_lse_offset_bytes,
        final_lse_offset_bytes=final_lse_offset_bytes,
        quantized_q_offset_bytes=quantized_q_offset_bytes,
    )


@dataclass(kw_only=True)
class Scratch:
    shared_scratch: torch.Tensor
    device: torch.device
    mode: str
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    page_size: int
    max_total_q: int
    max_batch: int
    max_cache_tokens: int
    max_page_table_width: int
    num_cache_pages: int
    physical_record_width: int
    head_dim: int
    v_head_dim: int
    window_size: int | None
    num_splits: int
    chunks_per_split: int
    query_tile: int
    use_cuda_graph: bool
    partial_output: torch.Tensor | None
    partial_lse: torch.Tensor | None
    final_lse: torch.Tensor
    quantized_q: torch.Tensor | None


@dataclass(frozen=True, kw_only=True)
class Binding:
    scratch: Scratch
    q: torch.Tensor
    kv_cache: torch.Tensor
    output: torch.Tensor
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    kv_scale: torch.Tensor | None
    q_scale: torch.Tensor | None
    sm_scale: float
    active_splits: int
    query_quant: static_fp8_quant.Binding | None


def _storage_bounds(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.numel() == 0:
        return
    lo = hi = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        extent = (int(size) - 1) * int(stride)
        if extent < 0:
            lo += extent
        else:
            hi += extent
    storage_elems = tensor.untyped_storage().nbytes() // tensor.element_size()
    if lo < 0 or hi >= storage_elems:
        raise ValueError(
            f"{name} view is out of storage bounds: shape={tuple(tensor.shape)}, "
            f"stride={tuple(tensor.stride())}, storage_offset={tensor.storage_offset()}"
        )


def _require_16_byte_records(
    tensor: torch.Tensor,
    *,
    name: str,
    row_stride: int,
) -> None:
    if int(tensor.data_ptr()) % 16:
        raise ValueError(f"{name} base pointer must be 16-byte aligned")
    if int(row_stride) * tensor.element_size() % 16:
        raise ValueError(f"{name} row stride must be 16-byte aligned")


def _validate_scalar(
    value: torch.Tensor | None,
    *,
    name: str,
    device: torch.device,
    required: bool,
) -> torch.Tensor | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    if value.dtype != torch.float32 or value.numel() != 1:
        raise ValueError(f"{name} must be a scalar float32 tensor")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}")
    return value.detach()


def _rank3_cache(
    cache: torch.Tensor,
    *,
    scratch: Scratch,
) -> torch.Tensor:
    if cache.ndim == 4:
        if int(cache.shape[2]) != 1:
            raise ValueError("rank-4 dense MLA cache must have one KV head")
        cache = cache[:, :, 0, :]
    if cache.ndim != 3:
        raise ValueError(
            "kv_cache must be [pages,page_size,physical_record_width] or "
            "[pages,page_size,1,physical_record_width]"
        )
    if cache.dtype != scratch.kv_dtype:
        raise TypeError(
            f"kv_cache dtype {cache.dtype} does not match plan {scratch.kv_dtype}"
        )
    if cache.device != scratch.device:
        raise ValueError("kv_cache device does not match dense MLA plan")
    if tuple(cache.shape[1:]) != (
        scratch.page_size,
        scratch.physical_record_width,
    ):
        raise ValueError(
            "kv_cache inner shape must be "
            f"({scratch.page_size}, {scratch.physical_record_width}), got "
            f"{tuple(cache.shape[1:])}"
        )
    if int(cache.shape[0]) > scratch.num_cache_pages:
        raise ValueError("kv_cache page count exceeds planned capacity")
    if (
        int(cache.stride(2)) != 1
        or int(cache.stride(1)) < scratch.physical_record_width
    ):
        raise ValueError(
            "kv_cache requires contiguous elements and a token stride covering "
            "physical_record_width"
        )
    if int(cache.stride(0)) < scratch.page_size * int(cache.stride(1)):
        raise ValueError("kv_cache page stride is smaller than one page payload")
    _require_16_byte_records(
        cache,
        name="kv_cache",
        row_stride=int(cache.stride(0)),
    )
    _storage_bounds(cache, name="kv_cache")
    return cache.detach()


def _validate_binding(
    *,
    scratch: Scratch,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_scale: torch.Tensor | None,
    q_scale: torch.Tensor | None,
    sm_scale: float,
    active_splits: int,
) -> Binding:
    if q.ndim != 3 or tuple(q.shape[1:]) != (
        scratch.num_q_heads,
        scratch.head_dim,
    ):
        raise ValueError(
            "q must have shape "
            f"[total_q,{scratch.num_q_heads},{scratch.head_dim}], got {tuple(q.shape)}"
        )
    if q.dtype != scratch.q_dtype:
        raise TypeError(
            f"q dtype {q.dtype} does not match planned query dtype {scratch.q_dtype}"
        )
    if q.device != scratch.device:
        raise ValueError("q device does not match dense MLA plan")
    if not 1 <= int(q.shape[0]) <= scratch.max_total_q:
        raise ValueError("q rows exceed planned capacity")
    if int(q.stride(2)) != 1 or int(q.stride(1)) != scratch.head_dim:
        raise ValueError("q requires contiguous absorbed head records")
    if int(q.stride(0)) < scratch.num_q_heads * scratch.head_dim:
        raise ValueError("q row stride overlaps absorbed head records")
    _require_16_byte_records(
        q,
        name="q",
        row_stride=int(q.stride(0)),
    )
    _storage_bounds(q, name="q")

    cache = _rank3_cache(kv_cache, scratch=scratch)
    if output.ndim != 3 or tuple(output.shape) != (
        int(q.shape[0]),
        scratch.num_q_heads,
        scratch.v_head_dim,
    ):
        raise ValueError(
            "output must have shape "
            f"{(int(q.shape[0]), scratch.num_q_heads, scratch.v_head_dim)}, "
            f"got {tuple(output.shape)}"
        )
    if output.dtype != torch.bfloat16 or output.device != scratch.device:
        raise TypeError("output must be BF16 on the plan device")
    if int(output.stride(2)) != 1 or int(output.stride(1)) != scratch.v_head_dim:
        raise ValueError("output requires contiguous latent head records")
    if int(output.stride(0)) < scratch.num_q_heads * scratch.v_head_dim:
        raise ValueError("output row stride overlaps latent head records")
    _require_16_byte_records(
        output,
        name="output",
        row_stride=int(output.stride(0)),
    )
    _storage_bounds(output, name="output")

    batch = int(cache_seqlens.shape[0]) if cache_seqlens.ndim == 1 else -1
    if not 1 <= batch <= scratch.max_batch:
        raise ValueError("cache_seqlens batch exceeds planned capacity")
    for name, tensor, shape in (
        ("cache_seqlens", cache_seqlens, (batch,)),
        ("cu_seqlens_q", cu_seqlens_q, (batch + 1,)),
    ):
        if tensor.dtype != torch.int32 or tuple(tensor.shape) != shape:
            raise TypeError(f"{name} must be contiguous int32 with shape {shape}")
        if tensor.device != scratch.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the plan device")
    if (
        page_table.ndim != 2
        or int(page_table.shape[0]) != batch
        or int(page_table.shape[1]) > scratch.max_page_table_width
    ):
        raise ValueError(
            "page_table must have shape [batch, width<=max_page_table_width]"
        )
    if page_table.dtype != torch.int32:
        raise TypeError("page_table must be torch.int32")
    if page_table.device != scratch.device or not page_table.is_contiguous():
        raise ValueError("page_table must be contiguous on the plan device")

    kv_scale = _validate_scalar(
        kv_scale,
        name="kv_scale",
        device=scratch.device,
        required=cache.dtype == _FP8,
    )
    q_scale = _validate_scalar(
        q_scale,
        name="q_scale",
        device=scratch.device,
        required=cache.dtype == _FP8,
    )
    if cache.dtype != _FP8 and (kv_scale is not None or q_scale is not None):
        raise ValueError("BF16 dense MLA does not accept quantization scales")
    query_quant = None
    native_q = q
    if q.dtype != cache.dtype:
        if scratch.quantized_q is None:
            raise RuntimeError("dense MLA query quantization scratch is missing")
        if q_scale is None:
            raise RuntimeError("dense MLA query quantization scale is missing")
        native_q = scratch.quantized_q[: int(q.shape[0])]
        query_quant = static_fp8_quant.bind(
            source=q,
            output=native_q,
            scale=q_scale,
            max_numel=(scratch.max_total_q * scratch.num_q_heads * scratch.head_dim),
        )
    sm_scale = float(sm_scale)
    if not (sm_scale > 0.0):
        raise ValueError("sm_scale must be positive")
    active_splits = int(active_splits)
    if not 1 <= active_splits <= scratch.num_splits:
        raise ValueError(
            "active_splits must be in the planned range "
            f"[1, {scratch.num_splits}], got {active_splits}"
        )
    return Binding(
        scratch=scratch,
        q=native_q.detach(),
        kv_cache=cache,
        output=output,
        page_table=page_table.detach(),
        cache_seqlens=cache_seqlens.detach(),
        cu_seqlens_q=cu_seqlens_q.detach(),
        kv_scale=kv_scale,
        q_scale=q_scale,
        sm_scale=sm_scale,
        active_splits=active_splits,
        query_quant=query_quant,
    )


def _materialize(
    caps: Caps,
    scratch_storage: torch.Tensor,
    layout: _ScratchLayout,
) -> Scratch:
    partial_output = None
    partial_lse = None
    if layout.num_splits > 1:
        assert layout.partial_output_offset_bytes is not None
        assert layout.partial_lse_offset_bytes is not None
        partial_output, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.partial_output_offset_bytes,
            shape=(
                caps.max_total_q,
                caps.num_q_heads,
                layout.num_splits,
                caps.v_head_dim,
            ),
            dtype=torch.bfloat16,
        )
        partial_lse, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.partial_lse_offset_bytes,
            shape=(caps.max_total_q, caps.num_q_heads, layout.num_splits),
            dtype=torch.float32,
        )
    final_lse, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=layout.final_lse_offset_bytes,
        shape=(caps.max_total_q, caps.num_q_heads),
        dtype=torch.float32,
    )
    quantized_q = None
    if layout.quantized_q_offset_bytes is not None:
        quantized_q, _ = materialize_scratch_view(
            scratch_storage,
            offset_bytes=layout.quantized_q_offset_bytes,
            shape=(caps.max_total_q, caps.num_q_heads, caps.head_dim),
            dtype=_FP8,
        )
    query_tile = _query_tile(caps)
    return Scratch(
        shared_scratch=scratch_storage,
        device=caps.device,
        mode=caps.mode,
        q_dtype=caps.q_dtype,
        kv_dtype=caps.kv_dtype,
        num_q_heads=caps.num_q_heads,
        page_size=caps.page_size,
        max_total_q=caps.max_total_q,
        max_batch=caps.max_batch,
        max_cache_tokens=caps.max_cache_tokens,
        max_page_table_width=caps.max_page_table_width,
        num_cache_pages=caps.num_cache_pages,
        physical_record_width=caps.physical_record_width,
        head_dim=caps.head_dim,
        v_head_dim=caps.v_head_dim,
        window_size=caps.window_size,
        num_splits=layout.num_splits,
        chunks_per_split=layout.chunks_per_split,
        query_tile=query_tile,
        use_cuda_graph=caps.use_cuda_graph,
        partial_output=partial_output,
        partial_lse=partial_lse,
        final_lse=final_lse,
        quantized_q=quantized_q,
    )


@dataclass(frozen=True)
class Plan:
    caps: Caps
    layout: _ScratchLayout
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    @property
    def num_splits(self) -> int:
        return int(self.layout.num_splits)

    @property
    def query_tile(self) -> int:
        return _query_tile(self.caps)

    @property
    def chunks_per_split(self) -> int:
        return int(self.layout.chunks_per_split)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        kv_scale: torch.Tensor | None = None,
        q_scale: torch.Tensor | None = None,
        sm_scale: float = K3_SM_SCALE,
        active_splits: int | None = None,
    ) -> Binding:
        storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="dense MLA",
        )
        scratch_views = _materialize(self.caps, storage, self.layout)
        return _validate_binding(
            scratch=scratch_views,
            q=q,
            kv_cache=kv_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            kv_scale=kv_scale,
            q_scale=q_scale,
            sm_scale=sm_scale,
            active_splits=(
                self.num_splits if active_splits is None else int(active_splits)
            ),
        )


def plan_dense_mla_scratch(
    caps: Caps,
) -> Plan:
    layout = _dense_mla_scratch_layout(caps)
    return Plan(
        caps=caps,
        layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "dense_mla.scratch",
                nbytes=layout.nbytes,
                device=caps.device,
            ),
        ),
    )


__all__ = [
    "Binding",
    "Caps",
    "Plan",
    "Scratch",
    "plan_dense_mla_scratch",
]
