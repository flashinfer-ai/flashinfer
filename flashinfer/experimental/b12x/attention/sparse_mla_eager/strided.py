"""Planned strided-record sparse-MLA API."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..._lib.gating import default_is_supported
from ..._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor
from ..._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    materialize_scratch_view,
)
from .._shared import static_fp8_quant
from ..dense_mla._kernel import (
    clear_dense_mla_kernel_caches,
    compile_dense_mla,
    run_dense_mla,
)
from ..dense_mla._scratch import Binding as _NativeBinding
from ..dense_mla._scratch import Caps as _NativeCaps
from ..dense_mla._scratch import Plan as _NativePlan
from ..dense_mla._scratch import Scratch
from ..dense_mla._scratch import plan_dense_mla_scratch
from ..dense_mla.planner import Budget
from . import _paged_index_remap

FP8 = torch.float8_e4m3fn
BLOCK_SIZE = 64
TOTAL_HEADS = 128
QK_DIM = 576
VALUE_DIM = 512
PHYSICAL_RECORD_WIDTH = 1088
TOPK = 2048
SM_SCALE = 1.0 / math.sqrt(192)


@dataclass(frozen=True, kw_only=True)
class Caps:
    device: torch.device | str
    num_q_heads: int
    tp_size: int
    max_q_rows: int
    num_cache_blocks: int
    max_physical_records: int | None = None
    block_size: int = BLOCK_SIZE
    topk: int = TOPK
    kv_dtype: torch.dtype = FP8
    use_cuda_graph: bool = False
    budget: Budget | None = None

    def __post_init__(self) -> None:
        if int(self.tp_size) != 8:
            raise ValueError("strided sparse MLA is qualified only for TP8")
        if int(self.num_q_heads) != TOTAL_HEADS // int(self.tp_size):
            raise ValueError("num_q_heads must equal total_heads / tp_size")
        if int(self.block_size) != BLOCK_SIZE:
            raise ValueError(f"strided sparse MLA block_size must be {BLOCK_SIZE}")
        if int(self.topk) != TOPK:
            raise ValueError(f"strided sparse MLA topk must be {TOPK}")
        if self.kv_dtype != FP8:
            raise TypeError("strided sparse MLA requires E4M3 KV cache")
        if int(self.max_q_rows) <= 0:
            raise ValueError("max_q_rows must be positive")
        if int(self.num_cache_blocks) <= 0:
            raise ValueError("num_cache_blocks must be positive")
        if int(self.num_cache_blocks) > torch.iinfo(torch.int32).max // BLOCK_SIZE:
            raise ValueError("num_cache_blocks exceeds the physical-slot int32 range")
        max_physical_records = (
            int(self.num_cache_blocks) * BLOCK_SIZE
            if self.max_physical_records is None
            else int(self.max_physical_records)
        )
        if not int(self.num_cache_blocks) * BLOCK_SIZE <= max_physical_records:
            raise ValueError(
                "max_physical_records must cover all contiguous physical slots"
            )
        if max_physical_records > torch.iinfo(torch.int32).max:
            raise ValueError("max_physical_records exceeds the index int32 range")
        object.__setattr__(self, "max_physical_records", max_physical_records)


@dataclass(frozen=True)
class Binding:
    native: _NativeBinding
    query_quant: static_fp8_quant.Binding
    index_remap: _paged_index_remap.Binding | None
    kv_cache: torch.Tensor
    selected_indices: torch.Tensor
    selected_counts: torch.Tensor


@dataclass(frozen=True)
class Plan:
    caps: Caps
    native: _NativePlan
    query_offset_bytes: int
    physical_indices_offset_bytes: int
    selected_counts_offset_bytes: int
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self):
        return self._scratch_specs

    def shapes_and_dtypes(self):
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    @property
    def num_splits(self) -> int:
        return self.native.num_splits


def plan(caps: Caps) -> Plan:
    native_caps = _NativeCaps(
        device=caps.device,
        mode="decode",
        kv_dtype=FP8,
        num_q_heads=int(caps.num_q_heads),
        page_size=1,
        max_total_q=int(caps.max_q_rows),
        max_batch=int(caps.max_q_rows),
        max_cache_tokens=TOPK,
        max_page_table_width=TOPK,
        num_cache_pages=int(caps.max_physical_records),
        head_dim=QK_DIM,
        v_head_dim=VALUE_DIM,
        physical_record_width=PHYSICAL_RECORD_WIDTH,
        use_cuda_graph=bool(caps.use_cuda_graph),
        budget=caps.budget,
    )
    native = plan_dense_mla_scratch(native_caps)
    native_spec = native.scratch_specs()[0]
    query_offset_bytes = align_up(native_spec.nbytes, SCRATCH_ALIGN_BYTES)
    query_nbytes = int(caps.max_q_rows) * int(caps.num_q_heads) * QK_DIM
    physical_indices_offset_bytes = align_up(
        query_offset_bytes + query_nbytes,
        SCRATCH_ALIGN_BYTES,
    )
    physical_indices_nbytes = int(caps.max_q_rows) * TOPK * torch.int32.itemsize
    selected_counts_offset_bytes = align_up(
        physical_indices_offset_bytes + physical_indices_nbytes,
        SCRATCH_ALIGN_BYTES,
    )
    selected_counts_nbytes = int(caps.max_q_rows) * torch.int32.itemsize
    total_nbytes = align_up(
        selected_counts_offset_bytes + selected_counts_nbytes,
        SCRATCH_ALIGN_BYTES,
    )
    return Plan(
        caps=caps,
        native=native,
        query_offset_bytes=query_offset_bytes,
        physical_indices_offset_bytes=physical_indices_offset_bytes,
        selected_counts_offset_bytes=selected_counts_offset_bytes,
        _scratch_specs=(
            scratch_buffer_spec(
                "sparse_mla.strided.scratch",
                nbytes=total_nbytes,
                device=native_spec.device,
            ),
        ),
    )


def _physical_record_view(
    plan: Plan,
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    expected_record_shape = (BLOCK_SIZE, PHYSICAL_RECORD_WIDTH)
    if (
        kv_cache.dtype != FP8
        or kv_cache.ndim != 3
        or tuple(kv_cache.shape[1:]) != expected_record_shape
        or not 0 < int(kv_cache.shape[0]) <= int(plan.caps.num_cache_blocks)
    ):
        raise ValueError(
            "kv_cache must be E4M3 with shape "
            f"[1..{plan.caps.num_cache_blocks}, {BLOCK_SIZE}, "
            f"{PHYSICAL_RECORD_WIDTH}], got "
            f"dtype={kv_cache.dtype}, shape={tuple(kv_cache.shape)}"
        )
    block_stride, token_stride, element_stride = map(int, kv_cache.stride())
    if (
        element_stride != 1
        or token_stride < PHYSICAL_RECORD_WIDTH
        or block_stride < BLOCK_SIZE * token_stride
        or token_stride % PHYSICAL_RECORD_WIDTH
        or block_stride % PHYSICAL_RECORD_WIDTH
    ):
        raise ValueError(
            "kv_cache must use non-overlapping, whole 1088-element physical "
            f"record strides, got stride={tuple(kv_cache.stride())}"
        )
    block_stride_records = block_stride // PHYSICAL_RECORD_WIDTH
    token_stride_records = token_stride // PHYSICAL_RECORD_WIDTH
    physical_records = (
        (int(kv_cache.shape[0]) - 1) * block_stride_records
        + (BLOCK_SIZE - 1) * token_stride_records
        + 1
    )
    if physical_records > int(plan.caps.max_physical_records):
        raise ValueError(
            "kv_cache physical record span exceeds planned capacity: "
            f"need {physical_records}, planned {plan.caps.max_physical_records}"
        )
    flat_cache = torch.as_strided(
        kv_cache,
        size=(physical_records, 1, PHYSICAL_RECORD_WIDTH),
        stride=(PHYSICAL_RECORD_WIDTH, PHYSICAL_RECORD_WIDTH, 1),
    )
    return flat_cache, block_stride_records, token_stride_records


def _bind_native(
    plan: Plan,
    *,
    scratch_storage: torch.Tensor,
    q: torch.Tensor,
    flat_cache: torch.Tensor,
    original_cache: torch.Tensor,
    output: torch.Tensor,
    record_indices: torch.Tensor,
    selected_counts: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_scale: torch.Tensor,
    q_scale: torch.Tensor,
    active_splits: int | None = None,
) -> Binding:
    if q.dtype != torch.bfloat16:
        raise TypeError("strided sparse MLA absorbed query must be BF16")
    rows = int(q.shape[0])
    if record_indices.dtype != torch.int32 or tuple(record_indices.shape) != (
        rows,
        TOPK,
    ):
        raise ValueError(f"record_indices must be int32 with shape ({rows}, {TOPK})")
    if selected_counts.dtype != torch.int32 or tuple(selected_counts.shape) != (rows,):
        raise ValueError(f"selected_counts must be int32 with shape ({rows},)")
    if tuple(cu_seqlens_q.shape) != (rows + 1,):
        raise ValueError(f"cu_seqlens_q must have shape ({rows + 1},)")
    q_fp8, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.query_offset_bytes,
        shape=tuple(q.shape),
        dtype=FP8,
    )
    query_quant = static_fp8_quant.bind(
        source=q,
        output=q_fp8,
        scale=q_scale,
        max_numel=int(plan.caps.max_q_rows) * int(plan.caps.num_q_heads) * QK_DIM,
    )
    native = plan.native.bind(
        scratch=scratch_storage,
        q=q_fp8,
        kv_cache=flat_cache,
        output=output,
        page_table=record_indices,
        cache_seqlens=selected_counts,
        cu_seqlens_q=cu_seqlens_q,
        kv_scale=kv_scale,
        q_scale=q_scale,
        sm_scale=SM_SCALE,
        active_splits=active_splits,
    )
    return Binding(
        native=native,
        query_quant=query_quant,
        index_remap=None,
        kv_cache=original_cache,
        selected_indices=record_indices,
        selected_counts=selected_counts,
    )


def _index_scratch(
    plan: Plan,
    scratch_storage: torch.Tensor,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    physical_indices, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.physical_indices_offset_bytes,
        shape=(rows, TOPK),
        dtype=torch.int32,
    )
    selected_counts, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.selected_counts_offset_bytes,
        shape=(rows,),
        dtype=torch.int32,
    )
    return physical_indices, selected_counts


def bind(
    plan: Plan,
    *,
    scratch,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_counts: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_scale: torch.Tensor,
    q_scale: torch.Tensor,
    active_splits: int | None = None,
) -> Binding:
    scratch_storage = scratch_tensor(
        scratch,
        plan.scratch_specs(),
        owner="strided sparse MLA",
    )
    rows = int(q.shape[0])
    record_indices, remapped_counts = _index_scratch(plan, scratch_storage, rows)
    flat_cache, block_stride_records, token_stride_records = _physical_record_view(
        plan, kv_cache
    )
    binding = _bind_native(
        plan,
        scratch_storage=scratch_storage,
        q=q,
        flat_cache=flat_cache,
        original_cache=kv_cache,
        output=output,
        record_indices=record_indices,
        selected_counts=remapped_counts,
        cu_seqlens_q=cu_seqlens_q,
        kv_scale=kv_scale,
        q_scale=q_scale,
        active_splits=active_splits,
    )
    remap = _paged_index_remap.bind_physical_slots(
        physical_slots=selected_indices,
        input_counts=selected_counts,
        physical_indices=record_indices,
        selected_counts=remapped_counts,
        max_q_rows=int(plan.caps.max_q_rows),
        num_cache_blocks=int(kv_cache.shape[0]),
        block_stride_records=block_stride_records,
        token_stride_records=token_stride_records,
    )
    return Binding(
        native=binding.native,
        query_quant=binding.query_quant,
        index_remap=remap,
        kv_cache=binding.kv_cache,
        selected_indices=binding.selected_indices,
        selected_counts=binding.selected_counts,
    )


def bind_indexed(
    plan: Plan,
    *,
    scratch,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    logical_indices: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_scale: torch.Tensor,
    q_scale: torch.Tensor,
    active_splits: int | None = None,
) -> Binding:
    scratch_storage = scratch_tensor(
        scratch,
        plan.scratch_specs(),
        owner="strided sparse MLA",
    )
    rows = int(q.shape[0])
    physical_indices, selected_counts = _index_scratch(plan, scratch_storage, rows)
    flat_cache, block_stride_records, token_stride_records = _physical_record_view(
        plan, kv_cache
    )
    binding = _bind_native(
        plan,
        scratch_storage=scratch_storage,
        q=q,
        flat_cache=flat_cache,
        original_cache=kv_cache,
        output=output,
        record_indices=physical_indices,
        selected_counts=selected_counts,
        cu_seqlens_q=cu_seqlens_q,
        kv_scale=kv_scale,
        q_scale=q_scale,
        active_splits=active_splits,
    )
    remap = _paged_index_remap.bind(
        request_ids=request_ids,
        block_table=block_table,
        logical_indices=logical_indices,
        physical_indices=physical_indices,
        selected_counts=selected_counts,
        max_q_rows=int(plan.caps.max_q_rows),
        num_cache_blocks=int(kv_cache.shape[0]),
        block_stride_records=block_stride_records,
        token_stride_records=token_stride_records,
    )
    return Binding(
        native=binding.native,
        query_quant=binding.query_quant,
        index_remap=remap,
        kv_cache=binding.kv_cache,
        selected_indices=binding.selected_indices,
        selected_counts=binding.selected_counts,
    )


def compile(*, binding: Binding) -> None:
    if binding.index_remap is not None:
        _paged_index_remap.compile(binding=binding.index_remap)
    static_fp8_quant.compile(binding=binding.query_quant)
    compile_dense_mla(binding=binding.native)


def run_decode(*, binding: Binding) -> tuple[torch.Tensor, torch.Tensor]:
    if binding.index_remap is not None:
        _paged_index_remap.run(binding=binding.index_remap)
    static_fp8_quant.run(binding=binding.query_quant)
    return run_dense_mla(binding=binding.native)


def run_extend(*, binding: Binding) -> tuple[torch.Tensor, torch.Tensor]:
    if binding.index_remap is not None:
        _paged_index_remap.run(binding=binding.index_remap)
    static_fp8_quant.run(binding=binding.query_quant)
    return run_dense_mla(binding=binding.native)


def reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_counts: torch.Tensor,
    *,
    kv_scale: torch.Tensor | float,
    q_scale: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    def scalar(value: torch.Tensor | float) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    count_host = [int(value) for value in selected_counts.detach().cpu().tolist()]
    q_mul = scalar(q_scale)
    q_f32 = (q.float() / q_mul).to(FP8).float() * q_mul
    kv_mul = scalar(kv_scale)
    output = torch.empty(
        q.shape[0], q.shape[1], VALUE_DIM, dtype=torch.float32, device=q.device
    )
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    for row, count in enumerate(count_host):
        ids = selected_indices[row, :count].to(torch.long)
        blocks = torch.div(ids, BLOCK_SIZE, rounding_mode="floor")
        tokens = ids.remainder(BLOCK_SIZE)
        records = kv_cache[blocks, tokens, :QK_DIM].float() * kv_mul
        logits = torch.einsum("hd,kd->hk", q_f32[row], records) * SM_SCALE
        probability = torch.softmax(logits, dim=-1)
        output[row] = torch.einsum("hk,kd->hd", probability, records[:, :VALUE_DIM])
        lse[row] = torch.logsumexp(logits, dim=-1)
    return output.to(torch.bfloat16), lse


def is_supported(device=None) -> bool:
    if not default_is_supported(device, requires=()):
        return False
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)
    return tuple(torch.cuda.get_device_capability(device)) in ((12, 0), (12, 1))


def clear_caches() -> None:
    _paged_index_remap.clear_caches()
    static_fp8_quant.clear_caches()
    clear_dense_mla_kernel_caches()


__all__ = [
    "Binding",
    "Budget",
    "Caps",
    "Plan",
    "Scratch",
    "bind",
    "bind_indexed",
    "clear_caches",
    "compile",
    "is_supported",
    "plan",
    "reference",
    "run_decode",
    "run_extend",
]
