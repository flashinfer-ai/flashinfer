"""Caller-owned state, workspace, and bindings for delta-rule prefill."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from b12x._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor
from b12x._lib.scratch_layout import SCRATCH_ALIGN_BYTES, align_up, dtype_nbytes, materialize_scratch_view
from b12x.preparation import Plan
from ..tensors import overlaps, positive, require_paged_recurrent_state, require_row_contiguous, require_tensor
from .workspace import V_SPLIT_CHOICES, WorkspaceRecord, tiles_capacity

HEAD_DIM = 128


class PrefillCaps(Protocol):
    device: torch.device
    max_tokens: int
    max_seqs: int
    max_state_slots: int
    heads: int
    key_heads: int
    state_dtype: torch.dtype
    qk_l2norm: bool
    checkpoint_export: bool
    null_state_index: int | None
    is_gdn: bool
    op_name: str

    @property
    def tiles_capacity(self) -> int: ...


@dataclass(frozen=True)
class Layout:
    """Fixed launch geometry and caller-allocated scratch layout for one Caps.

    Chunk tiles are ordered in bands (local tile index major, sequence rank
    minor, longest sequences first) and processed in windows of
    ``window_tiles`` consecutive positions, so every live sequence advances
    in every window. The prepare kernel writes each window into one of two
    workspace ring slots and the recurrence kernel of a window consumes that
    slot while the next window is being prepared. A sequence that continues
    across a window boundary keeps its running state in its final state slot,
    which must therefore not be null. ``max_windows`` is the launch count
    that covers the full capacity.
    """

    caps: PrefillCaps
    v_split: int
    k_split: int
    stages: int
    window_tiles: int
    max_windows: int
    offsets: Mapping[str, int]
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    workspace_windows: int = 2
    max_sequence_tiles: int = 0

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    @property
    def recurrence_rows(self) -> int:
        """Grid rows of one recurrence launch: sequences a window can intersect."""
        return min(self.caps.max_seqs, self.window_tiles)

    def launched_windows(self, max_live_tokens: int | None, max_live_seqs: int | None) -> int:
        """Windows to launch for a run bounded by the given live counts."""
        if max_live_tokens is None and max_live_seqs is None:
            return self.max_windows
        tokens = self.caps.max_tokens if max_live_tokens is None else int(max_live_tokens)
        seqs = self.caps.max_seqs if max_live_seqs is None else int(max_live_seqs)
        if tokens < 0 or tokens > self.caps.max_tokens:
            raise ValueError(f"max_live_tokens={tokens} exceeds capacity {self.caps.max_tokens}")
        if seqs < 0 or seqs > self.caps.max_seqs:
            raise ValueError(f"max_live_seqs={seqs} exceeds capacity {self.caps.max_seqs}")
        tiles = tiles_capacity(tokens, seqs)
        return max(1, min(self.max_windows, -(-tiles // self.window_tiles)))

    def output_shape(self, tokens: int | None = None) -> tuple[int, int, int]:
        live_tokens = self.caps.max_tokens if tokens is None else int(tokens)
        if live_tokens < 0 or live_tokens > self.caps.max_tokens:
            raise ValueError(f"tokens={live_tokens} exceeds capacity {self.caps.max_tokens}")
        return (live_tokens, self.caps.heads, HEAD_DIM)



@dataclass(frozen=True)
class Binding:
    """Caller-owned tensors and scratch views for one prefill invocation."""

    _state: Layout
    scratch: torch.Tensor
    band_base: torch.Tensor
    sorted_seq: torch.Tensor
    rank_of: torch.Tensor
    pos_seq: torch.Tensor
    pos_local: torch.Tensor
    window_table: torch.Tensor
    ready_flags: torch.Tensor
    ws: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    raw_g: torch.Tensor
    raw_beta: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    recurrent_state: torch.Tensor
    cu_seqlens: torch.Tensor
    initial_state_indices: torch.Tensor
    final_state_indices: torch.Tensor
    checkpoint_state_indices: torch.Tensor
    checkpoint_offsets: torch.Tensor
    num_seqs: torch.Tensor
    num_tokens: torch.Tensor
    output: torch.Tensor
    token_capacity: int
    seq_capacity: int
    plan: Plan | None = None


def materialize_layout(
    caps: PrefillCaps,
    *,
    layout_type: type[Layout],
    v_split: int,
    k_split: int,
    stages: int,
    window_tiles: int,
    workspace_windows: int = 2,
    max_sequence_tiles: int = 0,
) -> Layout:
    if v_split not in V_SPLIT_CHOICES:
        raise ValueError(f"v_split must be one of {V_SPLIT_CHOICES}, got {v_split}")
    tiles = caps.tiles_capacity
    heads = caps.heads
    window_tiles = max(1, min(int(window_tiles), tiles))
    max_windows = -(-tiles // window_tiles)
    if workspace_windows not in (0, 1, 2):
        raise ValueError("workspace_windows must be zero, one, or two")
    if workspace_windows == 1 and max_windows != 1:
        raise ValueError("one workspace window requires a single-window plan")
    ring_records = workspace_windows * window_tiles * heads
    regions = (
        ("band_base", tiles + 2, torch.int32),
        ("sorted_seq", caps.max_seqs, torch.int32),
        ("rank_of", caps.max_seqs, torch.int32),
        ("pos_seq", tiles, torch.int32),
        ("pos_local", tiles, torch.int32),
        ("window_table", 2 * max_windows, torch.int32),
        ("ready_flags", ring_records, torch.int32),
        ("ws", ring_records * WorkspaceRecord.BYTES, torch.uint8),
    )
    offsets: dict[str, int] = {}
    cursor = 0
    for name, elements, dtype in regions:
        cursor = align_up(cursor, SCRATCH_ALIGN_BYTES)
        offsets[name] = cursor
        cursor += elements * dtype_nbytes(dtype)
    spec = scratch_buffer_spec(caps.op_name, nbytes=cursor, device=caps.device)
    return layout_type(
        caps=caps,
        v_split=int(v_split),
        k_split=int(k_split),
        stages=int(stages),
        window_tiles=window_tiles,
        max_windows=max_windows,
        offsets=offsets,
        _scratch_specs=(spec,),
        workspace_windows=workspace_windows,
        max_sequence_tiles=max_sequence_tiles,
    )


def _record_view(
    storage: torch.Tensor, plan: Layout, name: str, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    view, _ = materialize_scratch_view(
        storage, offset_bytes=plan.offsets[name], shape=shape, dtype=dtype
    )
    return view


def bind_tensors(
    plan: Layout,
    *,
    binding_type: type[Binding],
    _plan: Plan | None = None,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state_indices: torch.Tensor,
    final_state_indices: torch.Tensor,
    checkpoint_state_indices: torch.Tensor,
    checkpoint_offsets: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    output: torch.Tensor,
) -> Binding:
    """Bind live tensors to a plan without allocating or launching work.

    Live capacities come from the bound tensors: ``q.shape[0]`` tokens and
    ``cu_seqlens.numel() - 1`` sequences, each at most the planned capacity.
    """
    if not isinstance(plan, Layout):
        raise TypeError("binding requires a delta-rule prefill layout")
    caps = plan.caps
    device = caps.device
    heads = caps.heads
    tiles = caps.tiles_capacity
    token_capacity = positive("q token capacity", q.shape[0]) if q.dim() == 3 else 0
    if token_capacity > caps.max_tokens:
        raise ValueError(f"token capacity {token_capacity} exceeds planned {caps.max_tokens}")
    seq_capacity = int(cu_seqlens.numel()) - 1
    if seq_capacity < 1 or seq_capacity > caps.max_seqs:
        raise ValueError(
            f"cu_seqlens must hold 2..{caps.max_seqs + 1} entries, got {cu_seqlens.numel()}"
        )
    row_shape = (token_capacity, heads, HEAD_DIM)
    for name, tensor in (("q", q), ("k", k)):
        require_row_contiguous(name, tensor, shape=(token_capacity, caps.key_heads, HEAD_DIM), device=device, dtypes=(torch.bfloat16,))
    require_row_contiguous("v", v, shape=row_shape, device=device, dtypes=(torch.bfloat16,))
    decay_shape = (token_capacity, heads) if caps.is_gdn else row_shape
    require_row_contiguous("a" if caps.is_gdn else "raw_g", raw_g, shape=decay_shape, device=device, dtypes=(torch.bfloat16,))
    require_tensor(
        "raw_beta", raw_beta, shape=(token_capacity, heads), device=device,
        dtypes=(torch.bfloat16,), contiguous=False,
    )
    if any(s <= 0 for s in raw_beta.stride()):
        raise ValueError("raw_beta must have positive strides")
    require_tensor("A_log", A_log, shape=(heads,), device=device, dtypes=(torch.bfloat16, torch.float32))
    require_tensor(
        "dt_bias", dt_bias, shape=(heads,) if caps.is_gdn else (heads, HEAD_DIM), device=device,
        dtypes=(torch.bfloat16, torch.float32),
    )
    require_paged_recurrent_state(
        recurrent_state,
        shape=(caps.max_state_slots, heads, HEAD_DIM, HEAD_DIM),
        device=device,
        dtype=caps.state_dtype,
    )
    require_tensor("cu_seqlens", cu_seqlens, shape=(seq_capacity + 1,), device=device, dtypes=(torch.int32,))
    index_dtypes = (torch.int32, torch.int64)
    for name, tensor in (
        ("initial_state_indices", initial_state_indices),
        ("checkpoint_state_indices", checkpoint_state_indices),
    ):
        require_tensor(name, tensor, shape=(seq_capacity,), device=device, dtypes=index_dtypes)
    require_tensor(
        "final_state_indices",
        final_state_indices,
        shape=(seq_capacity,),
        device=device,
        dtypes=index_dtypes,
        contiguous=False,
    )
    if final_state_indices.stride(0) <= 0:
        raise ValueError("final_state_indices must have a positive stride")
    if not (initial_state_indices.dtype == final_state_indices.dtype == checkpoint_state_indices.dtype):
        raise TypeError("state index tensors must share one dtype")
    require_tensor(
        "checkpoint_offsets", checkpoint_offsets, shape=(seq_capacity,), device=device, dtypes=(torch.int32,)
    )
    for name, tensor in (("num_seqs", num_seqs), ("num_tokens", num_tokens)):
        require_tensor(name, tensor, shape=(1,), device=device, dtypes=(torch.int32,))
    require_row_contiguous("output", output, shape=row_shape, device=device, dtypes=(torch.bfloat16,))

    storage = scratch_tensor(scratch, plan.scratch_specs(), owner=caps.op_name)
    ring = plan.workspace_windows * plan.window_tiles
    views = {
        "band_base": _record_view(storage, plan, "band_base", (tiles + 2,), torch.int32),
        "sorted_seq": _record_view(storage, plan, "sorted_seq", (caps.max_seqs,), torch.int32),
        "rank_of": _record_view(storage, plan, "rank_of", (caps.max_seqs,), torch.int32),
        "pos_seq": _record_view(storage, plan, "pos_seq", (tiles,), torch.int32),
        "pos_local": _record_view(storage, plan, "pos_local", (tiles,), torch.int32),
        "window_table": _record_view(storage, plan, "window_table", (plan.max_windows, 2), torch.int32),
        "ready_flags": _record_view(storage, plan, "ready_flags", (ring, heads), torch.int32),
        "ws": _record_view(storage, plan, "ws", (ring, heads, WorkspaceRecord.BYTES), torch.uint8),
    }
    mutable = {"scratch": storage, "recurrent_state": recurrent_state, "output": output}
    read_only = {
        "q": q, "k": k, "v": v, "raw_g": raw_g, "raw_beta": raw_beta, "A_log": A_log,
        "dt_bias": dt_bias, "cu_seqlens": cu_seqlens,
        "initial_state_indices": initial_state_indices,
        "final_state_indices": final_state_indices,
        "checkpoint_state_indices": checkpoint_state_indices,
        "checkpoint_offsets": checkpoint_offsets, "num_seqs": num_seqs, "num_tokens": num_tokens,
    }
    names = list(mutable)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            if overlaps(mutable[names[left]], mutable[names[right]]):
                raise ValueError(f"{names[left]} and {names[right]} must not overlap")
    for name, tensor in mutable.items():
        for other, candidate in read_only.items():
            if overlaps(tensor, candidate):
                raise ValueError(f"{name} must not overlap read-only tensor {other}")
    return binding_type(
        _state=plan,
        plan=_plan,
        scratch=storage,
        **views,
        q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta, A_log=A_log, dt_bias=dt_bias,
        recurrent_state=recurrent_state, cu_seqlens=cu_seqlens,
        initial_state_indices=initial_state_indices,
        final_state_indices=final_state_indices,
        checkpoint_state_indices=checkpoint_state_indices,
        checkpoint_offsets=checkpoint_offsets, num_seqs=num_seqs, num_tokens=num_tokens,
        output=output, token_capacity=token_capacity, seq_capacity=seq_capacity,
    )
