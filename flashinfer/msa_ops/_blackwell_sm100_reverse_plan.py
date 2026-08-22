"""Host-only planners for exact Blackwell sparse-prefill routes.

The exported CUDA kernels consume a six-int reverse worklist and a packed
K-to-Q CSR.  This module builds those tensors with standard Python and Torch;
it intentionally has no dependency on the source-generation environment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

_BLOCK_M = 128
_BLOCK_KV = 128
_SCHEDULE_FIELDS = 6
_QSPLIT_Q_MASK = 0x00FF_FFFF
_QSPLIT_SLOT_SHIFT = 24
_QSPLIT_SLOT_MASK = 0xFF


@dataclass(frozen=True)
class ReversePrefillShape:
    """Logical dimensions that determine one reverse schedule."""

    route: str
    batch_size: int
    seqlen_q: int
    seqlen_kv: int
    num_q_heads: int
    num_kv_heads: int
    topk: int

    @property
    def total_q(self) -> int:
        return self.batch_size * self.seqlen_q

    @property
    def qhead_per_kv(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @property
    def q_tokens_per_group(self) -> int:
        return _BLOCK_M // self.qhead_per_kv


@dataclass(frozen=True)
class ReverseWorkItem:
    """One row fragment in the six-field scheduler ABI."""

    head_kv: int
    row_linear: int
    q_begin: int
    q_count: int
    batch_idx: int
    kv_block_idx: int


@dataclass(frozen=True)
class ReverseScheduleGeometry:
    """Resolved schedule dimensions for one concrete selection tensor."""

    route: str
    sm_count: int
    total_rows: int
    row_head_count: int
    edge_count: int
    target_q_per_cta: int
    schedule_capacity: int
    work_count: int
    split_row_count: int
    max_row_count: int
    packed_m128_groups: int
    packed_q_utilization: float
    work_items: tuple[ReverseWorkItem, ...]


_FP8_TOPK8_SHAPE = ReversePrefillShape(
    route="fp8_topk8_qagg_pdl",
    batch_size=3,
    seqlen_q=1024,
    seqlen_kv=8192,
    num_q_heads=32,
    num_kv_heads=2,
    topk=8,
)
_BF16_TOPK4_SHAPE = ReversePrefillShape(
    route="bf16_paged_topk4_qload4",
    batch_size=3,
    seqlen_q=4096,
    seqlen_kv=8192,
    num_q_heads=8,
    num_kv_heads=2,
    topk=4,
)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def balanced_target_q_per_cta(
    shape: ReversePrefillShape,
    *,
    sm_count: int,
) -> int:
    """Resolve the exact work-item target used by the exported routes."""

    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    if shape.num_q_heads % shape.num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    total_refs_upper = shape.total_q * shape.topk * shape.num_kv_heads
    total_groups_upper = _ceil_div(max(total_refs_upper, 1), shape.q_tokens_per_group)
    desired_cap = 296 if shape.route == _FP8_TOPK8_SHAPE.route else 256
    desired_work_items = min(max(sm_count * 2, 1), desired_cap)
    target_groups_per_cta = min(
        512,
        max(1, _ceil_div(total_groups_upper, desired_work_items)),
    )
    occupancy_target = target_groups_per_cta * shape.q_tokens_per_group
    sink_balance_cap = max(shape.q_tokens_per_group, shape.topk * _BLOCK_KV * 2)
    target = min(max(occupancy_target, shape.q_tokens_per_group), sink_balance_cap)
    return _ceil_div(target, shape.q_tokens_per_group) * shape.q_tokens_per_group


def _validate_cpu_q2k(shape: ReversePrefillShape, q2k: torch.Tensor) -> None:
    expected = (shape.num_kv_heads, shape.total_q, shape.topk)
    if not isinstance(q2k, torch.Tensor):
        raise TypeError("q2k_indices must be a torch tensor")
    if q2k.device.type != "cpu":
        raise ValueError("reverse schedule construction requires a CPU snapshot")
    if q2k.dtype != torch.int32 or not q2k.is_contiguous():
        raise ValueError("q2k_indices must be contiguous int32")
    if tuple(int(value) for value in q2k.shape) != expected:
        raise ValueError(f"q2k_indices must have shape {expected}")


def analyze_reverse_schedule(
    shape: ReversePrefillShape,
    *,
    sm_count: int,
    q2k: torch.Tensor,
) -> ReverseScheduleGeometry:
    """Resolve row fanout and the logical six-field worklist."""

    _validate_cpu_q2k(shape, q2k)
    rows_per_batch = _ceil_div(shape.seqlen_kv, _BLOCK_KV)
    total_rows = shape.batch_size * rows_per_batch
    row_counts = [[0 for _ in range(total_rows)] for _ in range(shape.num_kv_heads)]
    for head in range(shape.num_kv_heads):
        for q_abs in range(shape.total_q):
            batch_idx = q_abs // shape.seqlen_q
            for slot in range(shape.topk):
                kv_block_idx = int(q2k[head, q_abs, slot])
                if 0 <= kv_block_idx < rows_per_batch:
                    row_linear = kv_block_idx * shape.batch_size + batch_idx
                    row_counts[head][row_linear] += 1

    target_q_per_cta = balanced_target_q_per_cta(shape, sm_count=sm_count)
    work_items: list[ReverseWorkItem] = []
    split_row_count = 0
    max_row_count = 0
    packed_m128_groups = 0
    edge_count = 0
    for head, head_counts in enumerate(row_counts):
        for row_linear, row_count in enumerate(head_counts):
            edge_count += row_count
            max_row_count = max(max_row_count, row_count)
            packed_m128_groups += _ceil_div(row_count, shape.q_tokens_per_group)
            chunks = _ceil_div(row_count, target_q_per_cta)
            if chunks > 1:
                split_row_count += 1
            for chunk in range(chunks):
                q_begin = chunk * target_q_per_cta
                work_items.append(
                    ReverseWorkItem(
                        head_kv=head,
                        row_linear=row_linear,
                        q_begin=q_begin,
                        q_count=min(target_q_per_cta, row_count - q_begin),
                        batch_idx=row_linear % shape.batch_size,
                        kv_block_idx=row_linear // shape.batch_size,
                    )
                )

    row_head_count = total_rows * shape.num_kv_heads
    refs_upper = shape.total_q * shape.topk * shape.num_kv_heads
    schedule_capacity = max(
        1,
        row_head_count + _ceil_div(max(refs_upper, 1), target_q_per_cta),
    )
    packed_q_capacity = packed_m128_groups * shape.q_tokens_per_group
    return ReverseScheduleGeometry(
        route=shape.route,
        sm_count=sm_count,
        total_rows=total_rows,
        row_head_count=row_head_count,
        edge_count=edge_count,
        target_q_per_cta=target_q_per_cta,
        schedule_capacity=schedule_capacity,
        work_count=len(work_items),
        split_row_count=split_row_count,
        max_row_count=max_row_count,
        packed_m128_groups=packed_m128_groups,
        packed_q_utilization=edge_count / packed_q_capacity,
        work_items=tuple(work_items),
    )


def build_reverse_schedule_inputs(
    shape: ReversePrefillShape,
    *,
    sm_count: int,
    q2k: torch.Tensor,
) -> tuple[ReverseScheduleGeometry, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build CPU scheduler metadata, row pointers, and packed query edges."""

    _validate_cpu_q2k(shape, q2k)
    geometry = analyze_reverse_schedule(shape, sm_count=sm_count, q2k=q2k)
    rows: list[list[list[int]]] = [
        [[] for _ in range(geometry.total_rows)] for _ in range(shape.num_kv_heads)
    ]
    rows_per_batch = _ceil_div(shape.seqlen_kv, _BLOCK_KV)
    for head in range(shape.num_kv_heads):
        for q_abs in range(shape.total_q):
            batch_idx = q_abs // shape.seqlen_q
            local_q = q_abs - batch_idx * shape.seqlen_q
            for slot in range(shape.topk):
                kv_block_idx = int(q2k[head, q_abs, slot])
                if 0 <= kv_block_idx < rows_per_batch:
                    row_linear = kv_block_idx * shape.batch_size + batch_idx
                    rows[head][row_linear].append(
                        local_q | ((slot & _QSPLIT_SLOT_MASK) << _QSPLIT_SLOT_SHIFT)
                    )

    nnz_per_head = shape.total_q * shape.topk
    row_ptr = torch.zeros(
        (shape.num_kv_heads, geometry.total_rows + 1), dtype=torch.int32
    )
    qsplit = torch.full((shape.num_kv_heads, nnz_per_head), -1, dtype=torch.int32)
    for head, head_rows in enumerate(rows):
        cursor = 0
        for row_linear, payload in enumerate(head_rows):
            row_ptr[head, row_linear] = cursor
            if payload:
                qsplit[head, cursor : cursor + len(payload)] = torch.tensor(
                    payload, dtype=torch.int32
                )
            cursor += len(payload)
        row_ptr[head, geometry.total_rows] = cursor
        if cursor != nnz_per_head:
            raise ValueError(
                f"{shape.route}: expected {nnz_per_head} edges/head, got {cursor}"
            )

    metadata = torch.zeros(
        (geometry.schedule_capacity, _SCHEDULE_FIELDS), dtype=torch.int32
    )
    for work_idx, item in enumerate(geometry.work_items):
        metadata[work_idx] = torch.tensor(
            (
                item.head_kv,
                item.row_linear,
                item.q_begin,
                item.q_count,
                item.batch_idx,
                item.kv_block_idx,
            ),
            dtype=torch.int32,
        )
    return (
        geometry,
        metadata.contiguous(),
        row_ptr.contiguous(),
        qsplit.contiguous(),
    )


def _batch_major_work_order(
    geometry: ReverseScheduleGeometry,
    metadata: torch.Tensor,
) -> tuple[ReverseScheduleGeometry, torch.Tensor]:
    """Stably order active work by batch while preserving the zero tail."""

    work_count = geometry.work_count
    permutation = tuple(
        sorted(
            range(work_count),
            key=lambda index: (geometry.work_items[index].batch_idx, index),
        )
    )
    if sorted(permutation) != list(range(work_count)):
        raise RuntimeError("reverse work order is not a permutation")
    ordered_items = tuple(geometry.work_items[index] for index in permutation)
    ordered = metadata.clone()
    ordered[:work_count] = metadata[list(permutation)]
    if ordered[work_count:].numel() and bool(
        torch.any(ordered[work_count:] != 0).item()
    ):
        raise RuntimeError("inactive scheduler metadata must remain zero")
    return replace(geometry, work_items=ordered_items), ordered.contiguous()


def _build_qagg_cohorts(
    shape: ReversePrefillShape,
    geometry: ReverseScheduleGeometry,
    row_ptr: torch.Tensor,
    qsplit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build and prove the exact producer dependency set for each cohort."""

    contributors = torch.full(
        (shape.total_q, shape.num_kv_heads, shape.topk),
        -1,
        dtype=torch.int32,
    )
    counts = torch.zeros((shape.total_q, shape.num_kv_heads), dtype=torch.int32)
    latest = torch.full_like(counts, -1)
    for work_idx, item in enumerate(geometry.work_items):
        row_begin = int(row_ptr[item.head_kv, item.row_linear]) + item.q_begin
        payload = qsplit[item.head_kv, row_begin : row_begin + item.q_count]
        if payload.numel() != item.q_count or bool(torch.any(payload < 0).item()):
            raise RuntimeError("reverse worklist does not cover its CSR payload")
        q_abs = (payload & _QSPLIT_Q_MASK) + item.batch_idx * shape.seqlen_q
        for query in q_abs.tolist():
            slot = int(counts[query, item.head_kv])
            if slot >= shape.topk:
                raise RuntimeError("query cohort has too many contributors")
            contributors[query, item.head_kv, slot] = work_idx
            counts[query, item.head_kv] += 1
            latest[query, item.head_kv] = max(
                int(latest[query, item.head_kv]), work_idx
            )
    if not bool(torch.all(counts == shape.topk).item()):
        raise RuntimeError("qagg reducer requires eight live slots per cohort")
    sorted_ids = torch.sort(contributors, dim=2).values
    if bool(torch.any(sorted_ids < 0).item()) or bool(
        torch.any(sorted_ids[:, :, 1:] == sorted_ids[:, :, :-1]).item()
    ):
        raise RuntimeError("qagg contributor proof is incomplete")
    observed = torch.unique(contributors, sorted=True)
    expected = torch.arange(geometry.work_count, dtype=torch.int32)
    if not torch.equal(observed, expected):
        raise RuntimeError("qagg contributors do not cover every producer")
    ready = latest.min(dim=1).values
    q_order = torch.argsort(ready, stable=True).to(torch.int32).contiguous()
    if not torch.equal(
        torch.sort(q_order).values,
        torch.arange(shape.total_q, dtype=torch.int32),
    ):
        raise RuntimeError("qagg query order is not a permutation")
    return q_order, contributors.contiguous()


def build_fp8_topk8_qagg_plan(
    q2k_cpu: torch.Tensor,
    *,
    sm_count: int,
) -> dict[str, Any]:
    """Build the CPU half of the exact TopK8 qagg plan."""

    shape = _FP8_TOPK8_SHAPE
    geometry, metadata, row_ptr, qsplit = build_reverse_schedule_inputs(
        shape, sm_count=sm_count, q2k=q2k_cpu
    )
    if geometry.schedule_capacity != 677 or geometry.work_count != 384:
        raise RuntimeError(
            "TopK8 qagg schedule geometry must be capacity/work 677/384; "
            f"got {geometry.schedule_capacity}/{geometry.work_count}"
        )
    geometry, metadata = _batch_major_work_order(geometry, metadata)
    split_counts = (
        (q2k_cpu >= 0).sum(dim=2, dtype=torch.int32).transpose(0, 1).contiguous()
    )
    if not bool(torch.all(split_counts == shape.topk).item()):
        raise RuntimeError("TopK8 qagg reducer requires eight live slots per row")
    q_order, contributors = _build_qagg_cohorts(shape, geometry, row_ptr, qsplit)
    return {
        "geometry": geometry,
        "scheduler_metadata": metadata,
        "k2q_row_ptr": row_ptr,
        "k2q_qsplit_indices": qsplit,
        "split_counts": split_counts,
        "q_order": q_order,
        "contributor_work_ids": contributors,
    }


def build_bf16_paged_topk4_plan(
    q2k_cpu: torch.Tensor,
    *,
    sm_count: int,
) -> dict[str, Any]:
    """Build the CPU half of the exact paged TopK4 qload4 plan."""

    shape = _BF16_TOPK4_SHAPE
    geometry, metadata, row_ptr, qsplit = build_reverse_schedule_inputs(
        shape, sm_count=sm_count, q2k=q2k_cpu
    )
    if (
        geometry.schedule_capacity != 640
        or geometry.work_count != 389
        or geometry.target_q_per_cta != 384
    ):
        raise RuntimeError(
            "paged TopK4 schedule geometry must be capacity/work/target "
            "640/389/384; got "
            f"{geometry.schedule_capacity}/{geometry.work_count}/"
            f"{geometry.target_q_per_cta}"
        )
    active = metadata[: geometry.work_count]
    group_counts = [
        _ceil_div(int(value), shape.q_tokens_per_group)
        for value in active[:, 3].tolist()
    ]
    if not group_counts or min(group_counts) < 1 or max(group_counts) > 21:
        raise RuntimeError("paged TopK4 group counts must be in [1, 21]")
    order = sorted(range(len(group_counts)), key=group_counts.__getitem__, reverse=True)
    metadata = torch.cat(
        (
            active.index_select(0, torch.tensor(order, dtype=torch.long)),
            metadata[geometry.work_count :],
        ),
        dim=0,
    ).contiguous()
    counts_by_group = [0] * 22
    for index in order:
        counts_by_group[group_counts[index]] += 1
    running = 0
    segment_ends: list[int] = []
    for group_count in range(21, 1, -1):
        running += counts_by_group[group_count]
        segment_ends.append(running)
    if running + counts_by_group[1] != geometry.work_count:
        raise RuntimeError("paged TopK4 segment ends do not cover all work")
    split_counts = (
        (q2k_cpu >= 0).sum(dim=2, dtype=torch.int32).transpose(0, 1).contiguous()
    )
    if not bool(torch.all(split_counts == shape.topk).item()):
        raise RuntimeError("paged TopK4 reducer requires four live slots per row")
    return {
        "geometry": geometry,
        "scheduler_metadata": metadata,
        "k2q_row_ptr": row_ptr,
        "k2q_qsplit_indices": qsplit,
        "split_counts": split_counts,
        "group_segment_ends": tuple(segment_ends),
    }


def _binding_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        id(tensor),
        int(tensor.data_ptr()),
        int(getattr(tensor, "_version", 0)),
        str(tensor.device),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        str(tensor.dtype),
    )


def _uploaded_plan_signature(
    route: str,
    *,
    sm_count: int,
    stream_id: int,
    owners: tuple[torch.Tensor, ...],
) -> tuple[Any, ...]:
    return (
        route,
        int(sm_count),
        int(stream_id),
        tuple(_binding_signature(tensor) for tensor in owners),
    )


def _cached_uploaded_plan(
    state: dict[str, Any],
    signature: tuple[Any, ...],
    owners: tuple[torch.Tensor, ...],
) -> bool:
    cached_owners = state.get("owners")
    return bool(
        state.get("signature") == signature
        and isinstance(cached_owners, tuple)
        and len(cached_owners) == len(owners)
        and all(cached_owners[index] is tensor for index, tensor in enumerate(owners))
    )


def _require_cuda_i32(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if (
        tensor.device.type != "cuda"
        or tensor.dtype != torch.int32
        or not tensor.is_contiguous()
        or tuple(int(value) for value in tensor.shape) != shape
    ):
        raise ValueError(f"{name} must be contiguous CUDA int32 with shape {shape}")


def _require_exact_values(
    tensor: torch.Tensor, *, name: str, expected: tuple[int, ...]
) -> None:
    actual = tuple(
        int(value)
        for value in tensor.detach().to(device="cpu", dtype=torch.int32).tolist()
    )
    if actual != expected:
        raise ValueError(f"{name} must equal {expected}, got {actual}")


def _prepare_uploaded_plan(
    *,
    route: str,
    q2k_indices: torch.Tensor,
    owners: tuple[torch.Tensor, ...],
    sm_count: int,
    stream_id: int,
    state: dict[str, Any],
    build,
) -> dict[str, Any]:
    signature = _uploaded_plan_signature(
        route,
        sm_count=sm_count,
        stream_id=stream_id,
        owners=owners,
    )
    if _cached_uploaded_plan(state, signature, owners):
        return state
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"{route} plan must be prepared before CUDA graph capture")
    q2k_cpu = q2k_indices.detach().to(device="cpu", dtype=torch.int32).contiguous()
    built = build(q2k_cpu, sm_count=sm_count)
    state.clear()
    state.update(
        {
            "signature": signature,
            "owners": owners,
            **{
                name: value.to(device=q2k_indices.device)
                if isinstance(value, torch.Tensor)
                else value
                for name, value in built.items()
            },
        }
    )
    return state


def prepare_fp8_topk8_qagg_plan(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    sm_count: int,
    stream_id: int,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Prepare and upload the exact TopK8 qagg plan."""

    if not isinstance(state, dict):
        raise TypeError("state must be a dict")
    _require_cuda_i32(
        q2k_indices,
        name="q2k_indices",
        shape=(2, 3072, 8),
    )
    _require_cuda_i32(cu_seqlens_q, name="cu_seqlens_q", shape=(4,))
    _require_cuda_i32(cu_seqlens_k, name="cu_seqlens_k", shape=(4,))
    owners = (q2k_indices, cu_seqlens_q, cu_seqlens_k)
    signature = _uploaded_plan_signature(
        _FP8_TOPK8_SHAPE.route,
        sm_count=sm_count,
        stream_id=stream_id,
        owners=owners,
    )
    if _cached_uploaded_plan(state, signature, owners):
        return state
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "fp8_topk8_qagg_pdl plan must be prepared before CUDA graph capture"
        )
    _require_exact_values(
        cu_seqlens_q,
        name="cu_seqlens_q",
        expected=(0, 1024, 2048, 3072),
    )
    _require_exact_values(
        cu_seqlens_k,
        name="cu_seqlens_k",
        expected=(0, 8192, 16384, 24576),
    )
    return _prepare_uploaded_plan(
        route=_FP8_TOPK8_SHAPE.route,
        q2k_indices=q2k_indices,
        owners=owners,
        sm_count=sm_count,
        stream_id=stream_id,
        state=state,
        build=build_fp8_topk8_qagg_plan,
    )


def prepare_bf16_paged_topk4_plan(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    *,
    sm_count: int,
    stream_id: int,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Prepare and upload the exact paged TopK4 qload4 plan."""

    if not isinstance(state, dict):
        raise TypeError("state must be a dict")
    _require_cuda_i32(
        q2k_indices,
        name="q2k_indices",
        shape=(2, 12288, 4),
    )
    _require_cuda_i32(cu_seqlens_q, name="cu_seqlens_q", shape=(4,))
    _require_cuda_i32(cu_seqlens_k, name="cu_seqlens_k", shape=(4,))
    _require_cuda_i32(page_table, name="page_table", shape=(3, 64))
    _require_cuda_i32(seqused_k, name="seqused_k", shape=(3,))
    owners = (
        q2k_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        page_table,
        seqused_k,
    )
    signature = _uploaded_plan_signature(
        _BF16_TOPK4_SHAPE.route,
        sm_count=sm_count,
        stream_id=stream_id,
        owners=owners,
    )
    if _cached_uploaded_plan(state, signature, owners):
        return state
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "bf16_paged_topk4_qload4 plan must be prepared before CUDA graph capture"
        )
    _require_exact_values(
        cu_seqlens_q,
        name="cu_seqlens_q",
        expected=(0, 4096, 8192, 12288),
    )
    _require_exact_values(
        cu_seqlens_k,
        name="cu_seqlens_k",
        expected=(0, 8192, 16384, 24576),
    )
    _require_exact_values(
        seqused_k,
        name="seqused_k",
        expected=(8192, 8192, 8192),
    )
    return _prepare_uploaded_plan(
        route=_BF16_TOPK4_SHAPE.route,
        q2k_indices=q2k_indices,
        owners=owners,
        sm_count=sm_count,
        stream_id=stream_id,
        state=state,
        build=build_bf16_paged_topk4_plan,
    )


__all__ = [
    "ReversePrefillShape",
    "ReverseScheduleGeometry",
    "ReverseWorkItem",
    "analyze_reverse_schedule",
    "balanced_target_q_per_cta",
    "build_bf16_paged_topk4_plan",
    "build_fp8_topk8_qagg_plan",
    "build_reverse_schedule_inputs",
    "prepare_bf16_paged_topk4_plan",
    "prepare_fp8_topk8_qagg_plan",
]
