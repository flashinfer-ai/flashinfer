"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from collections import OrderedDict
import math
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import tvm_ffi

from ..api_logging import flashinfer_api
from ..jit.cake_minimax_h3_sm120_quant_varlen_attention import (
    ATTENTION,
    ATTENTION_LEAD,
    ATTENTION_TAIL,
    KV_STATS,
    PREPROCESS_HEAD_MAJOR,
    QUANTIZE,
)
from ..jit.cake_program import CakeProgram
from ..utils import register_custom_op, register_fake_op

MINIMAX_H3_HEAD_DIM = 128
MINIMAX_H3_NUM_HEADS = 56
# 128 query rows per unit, 128 keys per K/V tile (absolute-token aligned K blocks).
_BLOCK_M = 128
_BLOCK_N = 128
_MAX_HEADS = 1 << 15
_MAX_SEGMENT_TILES = 1 << 16
# The kernels index the [tokens padded to _BLOCK_N, heads, 128] operands, the [segments, heads, 128]
# statistics and the [query tiles, heads, _PARTIAL_FLOATS] statistics partials with 32-bit offsets.
_MAX_ELEMENTS = (1 << 31) - 1
# Fixed per-unit overhead (Q TMA, pipeline fill, epilogue) in K/V-tile units for the LPT placement.
_UNIT_OVERHEAD_TILES = 2
_PARTIAL_FLOATS = 2 * MINIMAX_H3_HEAD_DIM
# The attention kernel exponentiates in the log2 domain: the launch takes ``softmax_scale * log2(e)``,
# log2(e) spelled as the double the scale is multiplied by before the binding rounds it to float.
_LOG2E = 1.4426950408889634
# Grid rule (``minimax_h3_sm120_quant_varlen_attention_launch_grids``): grid-stride statistics and
# quantizer launches of at most 4 CTAs per SM, one persistent attention CTA per SM.
_PREP_CTAS_PER_SM = 4
# The overlapped form of a plan inside the overlap band (``_overlap_selected``) on a part whose SM
# count has an overlap width P (``minimax_h3_sm120_quant_varlen_attention_overlap_grids``), in launch
# order: the head-major pre-processing producers (a plain launch; one CTA of two workers on each SM it
# takes), the lead attention launch over the SMs the producers leave (a programmatic dependent of the
# producers) and the tail attention launch over P CTAs (a programmatic dependent of the lead).  The
# three share the head-control words ``hctr`` (4 words per head, then 8): each head's operands are
# published by a ready flag the attention CTAs wait on before their loads, the tail CTAs end after
# every lead CTA and the producers' completion, and the words restart within every complete call.  A
# CTA waits only on CTAs of the launches before it in its call, which have all been resident by then,
# so a call completes whatever else runs on the device.  Each producer item and attention unit
# computes what the sequential launches compute for it, so the output is bitwise the sequential
# form's.
_OVERLAP_WIDTH = {170: 32, 188: 32}
# The overlap width P of a plan also follows its attention-weighted mean segment length (the sum of
# the squared segment lengths over the token count: the attention work per pre-processed token).  A
# plan whose mean is at most ``_OVERLAP_LIGHT_MAX_SEGMENT`` tokens (per SM count) is bound by its
# producers and takes the width ``_OVERLAP_WIDTH_LIGHT``; a plan of more than
# ``_OVERLAP_LONG_MIN_PADDED`` padded tokens whose mean is at least ``_OVERLAP_LONG_MIN_SEGMENT``
# tokens takes ``_OVERLAP_WIDTH_LONG``; the rest keep ``_OVERLAP_WIDTH``.
_OVERLAP_WIDTH_LIGHT = {170: 64, 188: 64}
_OVERLAP_LIGHT_MAX_SEGMENT = {170: 1792, 188: 2304}
_OVERLAP_WIDTH_LONG = {170: 64, 188: 48}
_OVERLAP_LONG_MIN_PADDED = 16384
_OVERLAP_LONG_MIN_SEGMENT = 4096
_OVERLAP_PRODUCER_WORKERS = 2
_HCTR_WORDS_PER_HEAD = 4
_HCTR_TAIL_WORDS = 8
# The overlap band: 56 heads and an attention work W = sum(len^2) over the segments of 2^23 to
# 24576^2 (below it the pre-processing is too short to hide, above it the attention dwarfs it).
_OVERLAP_HEADS = MINIMAX_H3_NUM_HEADS
_OVERLAP_MIN_WORK = 1 << 23
_OVERLAP_MAX_WORK = 24576 * 24576
_PROGRAMS = (
    KV_STATS,
    QUANTIZE,
    ATTENTION,
    PREPROCESS_HEAD_MAJOR,
    ATTENTION_LEAD,
    ATTENTION_TAIL,
)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def normalize_cu_seqlens(
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    cu_seqlens_host: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """Validate ``cu_seqlens`` and return its host copy as a tuple of ints.

    When ``cu_seqlens`` is a CUDA tensor and no ``cu_seqlens_host`` is given, the bounds are
    copied to the host once (a synchronizing ``tolist``); pass ``cu_seqlens_host`` to avoid it.
    """

    if isinstance(cu_seqlens, torch.Tensor):
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
            raise ValueError("cu_seqlens must be a one-dimensional int32 tensor")
        if cu_seqlens_host is None:
            cu_seqlens_host = cu_seqlens.tolist()
        elif len(cu_seqlens_host) != int(cu_seqlens.shape[0]):
            raise ValueError("cu_seqlens_host must have the same length as cu_seqlens")
    elif cu_seqlens_host is None:
        cu_seqlens_host = cu_seqlens
    bounds = tuple(int(v) for v in cu_seqlens_host)
    if len(bounds) < 2 or bounds[0] != 0:
        raise ValueError("cu_seqlens must have at least two entries and start at zero")
    if any(b < a for a, b in zip(bounds, bounds[1:], strict=False)):
        raise ValueError("cu_seqlens must be non-decreasing")
    return bounds


def assign_unit_slots(unit_costs: List[int], num_ctas: int) -> List[int]:
    """Longest-processing-time-first placement of units into persistent-grid slots.

    Slot ``k * G + i`` is the ``k``-th unit of CTA ``i`` (``G = min(num_ctas, units)``).  The
    partial tail round receives the cheapest units; within every full round the CTAs that also
    own a tail unit receive that round's cheapest units.  Equal costs keep enumeration order so
    a segment's units stay head-major (the resident CTAs share one head's K/V through L2).
    """

    total = len(unit_costs)
    if total == 0:
        return []
    G = min(num_ctas, total)
    order = sorted(range(total), key=lambda u: -unit_costs[u])
    tail = total % G
    slots = [0] * total
    full_units = order[: total - tail]
    for k in range(len(full_units) // G):
        chunk = full_units[k * G : (k + 1) * G]
        for i in range(G):
            slots[k * G + i] = chunk[G - 1 - i]
    for i, unit in enumerate(order[total - tail :] if tail else []):
        slots[(total // G) * G + i] = unit
    return slots


def _overlap_selected(num_heads: int, work: int, num_sms: int) -> bool:
    """Whether a plan of ``num_heads`` heads and attention work ``work`` (the sum of the squared
    segment lengths) runs the overlapped form on a device of ``num_sms`` SMs."""
    return (
        int(num_sms) in _OVERLAP_WIDTH
        and num_heads == _OVERLAP_HEADS
        and _OVERLAP_MIN_WORK <= work <= _OVERLAP_MAX_WORK
    )


class MiniMaxH3VarlenPlan:
    """Segment tables and persistent-grid unit table for one ``(cu_seqlens, heads, device)``; a plan
    of the overlap band also holds the overlapped form's head-major unit table."""

    def __init__(
        self,
        bounds: Tuple[int, ...],
        num_heads: int,
        device: torch.device,
        num_ctas: int,
    ):
        self.bounds = bounds
        self.key = (bounds, num_heads, device.index, int(num_ctas))
        self.num_heads = num_heads
        self.total_tokens = bounds[-1]
        padded_tokens = max(1, _ceil_div(self.total_tokens, _BLOCK_N)) * _BLOCK_N
        if padded_tokens * num_heads * MINIMAX_H3_HEAD_DIM > _MAX_ELEMENTS:
            raise ValueError(
                f"tokens (padded to {_BLOCK_N}) * heads * {MINIMAX_H3_HEAD_DIM} must fit the "
                f"kernels' 32-bit offsets, got tokens={self.total_tokens}, heads={num_heads}"
            )
        self.num_segments = len(bounds) - 1
        begins = [bounds[s] for s in range(self.num_segments)]
        lens = [bounds[s + 1] - bounds[s] for s in range(self.num_segments)]
        num_tiles = sum(_ceil_div(length, _BLOCK_M) for length in lens)
        if (
            max(self.num_segments * MINIMAX_H3_HEAD_DIM, num_tiles * _PARTIAL_FLOATS)
            * num_heads
            > _MAX_ELEMENTS
        ):
            raise ValueError(
                f"segments * heads * {MINIMAX_H3_HEAD_DIM} and query tiles * heads * "
                f"{_PARTIAL_FLOATS} must fit the statistics kernels' 32-bit offsets, got "
                f"segments={self.num_segments}, tiles={num_tiles}, heads={num_heads}"
            )
        tile_table: List[int] = []
        seg_tile_begin: List[int] = [0]
        units: List[Tuple[int, int, int]] = []
        costs: List[int] = []
        for seg, (begin, length) in enumerate(zip(begins, lens, strict=True)):
            q_tiles = _ceil_div(length, _BLOCK_M)
            if q_tiles >= _MAX_SEGMENT_TILES:
                raise ValueError(
                    f"a segment needs fewer than {_MAX_SEGMENT_TILES} query tiles"
                )
            for t in range(q_tiles):
                tile_table.extend((seg, t))
            seg_tile_begin.append(seg_tile_begin[-1] + q_tiles)
            if length > 0:
                cost = (
                    (begin + length - 1) // _BLOCK_N
                    - begin // _BLOCK_N
                    + 1
                    + _UNIT_OVERHEAD_TILES
                )
                for head in range(num_heads):
                    for t in range(q_tiles):
                        units.append((seg, head, t))
                        costs.append(cost)
        self.num_tiles = seg_tile_begin[-1]
        self.num_units = len(units)
        self.grid = max(1, min(int(num_ctas), max(self.num_units, 1)))
        unit_table: List[int] = []
        for unit in assign_unit_slots(costs, self.grid):
            seg, head, t = units[unit]
            unit_table.extend((seg, (head << 16) | t))
        self.work = sum(length * length for length in lens)
        self.overlap = _overlap_selected(num_heads, self.work, int(num_ctas))
        # The overlapped form's units come from one counter in this order: head by head (the order the
        # producers publish the heads), each head's units longest first (equal costs keep segment and
        # tile order).
        unit_table_head_major: List[int] = []
        if self.overlap:
            per_head = sorted(
                (
                    (cost, seg, t)
                    for (seg, head, t), cost in zip(units, costs, strict=True)
                    if head == 0
                ),
                key=lambda unit: -unit[0],
            )
            for head in range(num_heads):
                for _, seg, t in per_head:
                    unit_table_head_major.extend((seg, (head << 16) | t))
        self.num_kblocks = max(1, _ceil_div(self.total_tokens, _BLOCK_N))
        self.padded_tokens = self.num_kblocks * _BLOCK_N
        i32 = lambda values: torch.tensor(values, dtype=torch.int32, device=device)  # noqa: E731
        self.cu_seqlens = i32(list(bounds))
        self.seg_begin = i32(begins or [0])
        self.seg_len = i32(lens or [0])
        self.seg_tile_begin = i32(seg_tile_begin)
        self.tile_table = i32(tile_table or [0, 0])
        self.unit_table = i32(unit_table or [0, 0])
        self.unit_table_head_major = i32(unit_table_head_major or [0, 0])


# Most recently used segment plans (device tables) and workspace sets, one set per workspace owner
# (``_owner``).  Both are bounded: the oldest entry is evicted once the limit is reached (a workspace set as a
# whole), so varying segment layouts cannot grow device memory without bound.
_MAX_PLANS = 256
_MAX_WORKSPACE_SETS = 8
_PLANS: "OrderedDict[Tuple, MiniMaxH3VarlenPlan]" = OrderedDict()
_WORKSPACES: "OrderedDict[Tuple[int, ...], Dict[str, torch.Tensor]]" = OrderedDict()
_STREAM_WORKSPACES: "OrderedDict[Tuple[int, ...], _StreamWorkspace]" = OrderedDict()


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _plan(
    bounds: Tuple[int, ...], num_heads: int, device: torch.device
) -> MiniMaxH3VarlenPlan:
    return _plan_at(bounds, num_heads, _device_index(device))


def _plan_at(
    bounds: Tuple[int, ...], num_heads: int, index: int
) -> MiniMaxH3VarlenPlan:
    num_ctas = _sm_count(index)
    key = (bounds, num_heads, index, num_ctas)
    plan = _PLANS.get(key)
    if plan is None:
        while len(_PLANS) >= _MAX_PLANS:
            _PLANS.popitem(last=False)
        plan = MiniMaxH3VarlenPlan(
            bounds, num_heads, torch.device("cuda", index), num_ctas
        )
        _PLANS[key] = plan
    else:
        _PLANS.move_to_end(key)
    return plan


class _Launch:
    """One plan's workspace slices and launch facts, built once per workspace set (``_owner``): ``args``
    are the custom op's arguments from ``cu_seqlens`` through ``tail_grid``, the tensors among them
    already converted to tvm-ffi tensors, so a call hands the bindings no DLPack exchange but Q, K, V
    and the output's.  The overlapped form's grids are zero for a plan outside the overlap band."""

    __slots__ = ("args",)
    args: Tuple[Any, ...]

    def __init__(
        self, plan: MiniMaxH3VarlenPlan, workspace: "_StreamWorkspace", index: int
    ):
        tokens, heads, hd = plan.total_tokens, plan.num_heads, MINIMAX_H3_HEAD_DIM
        # The bindings check the tensors handed in and the attention binding encodes its tensor maps
        # from their shapes, so every buffer is sliced to the plan's exact size.  A plan with a unit has
        # a query tile, so the statistics launch always has an item.
        n_rows8 = tokens * heads * hd
        n_vt8 = heads * hd * plan.padded_tokens
        n_rows = tokens * heads
        n_kblocks = heads * plan.num_kblocks
        n_stats = plan.num_segments * heads * hd
        n_partials = plan.num_tiles * heads * _PARTIAL_FLOATS
        n_pairs = plan.num_segments * heads
        n_hctr = _HCTR_WORDS_PER_HEAD * heads + _HCTR_TAIL_WORDS
        buffer = workspace.buffer
        q8 = buffer("q8", n_rows8, torch.uint8)[:n_rows8]
        k8 = buffer("k8", n_rows8, torch.uint8)[:n_rows8]
        vt8 = buffer("vt8", n_vt8, torch.uint8)[:n_vt8]
        q_scale = buffer("q_scale", n_rows, torch.float32)[:n_rows]
        k_scale = buffer("k_scale", n_kblocks, torch.float32)[:n_kblocks]
        v_scale = buffer("v_scale", n_stats, torch.float32)[:n_stats]
        mean_k = buffer("mean_k", n_stats, torch.float32)[:n_stats]
        partials = buffer("partials", n_partials, torch.float32)[:n_partials]
        counters = buffer("counters", n_pairs, torch.uint32, zero=True)[:n_pairs]
        # the head-control words start zeroed and every complete overlapped call leaves them zero
        hctr = buffer("hctr", n_hctr, torch.uint32, zero=True)[:n_hctr]
        stats_grid, quantize_grid, attention_grid = (
            minimax_h3_sm120_quant_varlen_attention_launch_grids(
                _sm_count(index),
                num_heads=heads,
                num_tiles=plan.num_tiles,
                num_kblocks=plan.num_kblocks,
                num_units=plan.num_units,
            )
        )
        overlap_grids = (0, 0, 0)
        if plan.overlap:
            grids = minimax_h3_sm120_quant_varlen_attention_overlap_grids(
                _sm_count(index),
                num_units=plan.num_units,
                total_tokens=plan.total_tokens,
                work=plan.work,
                padded_tokens=plan.padded_tokens,
            )
            assert grids is not None, "an overlap plan has an overlap width"
            overlap_grids = grids
        tensors = (
            plan.cu_seqlens,
            plan.seg_begin,
            plan.seg_len,
            plan.seg_tile_begin,
            plan.tile_table,
            plan.unit_table,
            q8.view(torch.uint32),
            k8.view(torch.uint32),
            vt8.view(torch.uint32),
            q8.view(tokens, heads, hd),
            k8.view(tokens, heads, hd),
            vt8.view(heads, hd, plan.padded_tokens),
            q_scale,
            k_scale,
            v_scale,
            mean_k,
            partials,
            counters,
            hctr,
            plan.unit_table_head_major,
        )
        self.args = (
            *(tvm_ffi.from_dlpack(t) for t in tensors),
            plan.num_segments,
            plan.num_tiles,
            plan.num_units,
            plan.num_kblocks,
            plan.padded_tokens,
            stats_grid,
            quantize_grid,
            attention_grid,
            *overlap_grids,
        )


class _StreamWorkspace:
    """The grow-only buffers of one workspace owner (``_owner``) and the launches sliced from them, per plan.

    The quantized operands, the scales and the statistics partials are rewritten every call, so the
    buffers are shared between the calls of one owner, which are ordered; concurrent streams and
    distinct graph captures get their own set (no cross-stream reuse races).  The arrival counters of
    the statistics launch start zeroed and the kernel resets them after every complete call, so they
    too are per owner (interleaved arrivals would misidentify the last tile of a (segment, head)).
    Created during a graph capture, a buffer comes from the graph's memory pool, and a zeroed one is
    zeroed by a memset node of the graph at every replay.
    """

    __slots__ = ("device", "buffers", "launches")

    def __init__(self, index: int):
        self.device = torch.device("cuda", index)
        self.buffers: Dict[str, torch.Tensor] = {}
        self.launches: "OrderedDict[Tuple, _Launch]" = OrderedDict()

    def buffer(
        self, name: str, numel: int, dtype: torch.dtype, zero: bool = False
    ) -> torch.Tensor:
        buffer = self.buffers.get(name)
        if buffer is None or buffer.numel() < numel:
            make = torch.zeros if zero else torch.empty
            buffer = make(max(numel, 16), dtype=dtype, device=self.device)
            self.buffers[name] = buffer
            # the cached launches slice the buffer this one replaces
            self.launches.clear()
        return buffer

    def launch(self, plan: MiniMaxH3VarlenPlan, index: int) -> _Launch:
        launch = self.launches.get(plan.key)
        if launch is None:
            launch = _Launch(plan, self, index)
            while len(self.launches) >= _MAX_PLANS:
                self.launches.popitem(last=False)
            self.launches[plan.key] = launch
        return launch


def _owner_entry(
    entries: "OrderedDict[Tuple[int, ...], Any]",
    owner: Tuple[int, ...],
    make: Callable[[], Any],
) -> Any:
    """The entry of workspace owner ``owner`` in the bounded map ``entries``, made by ``make()`` when missing
    (the least recently used entries are evicted first, whole), and now the most recently used one.  Only
    calls of one owner touch its entry, under one stream lock; a call of another owner on another host
    thread may evict it, so the entry is moved by pop and re-insert rather than ``move_to_end``."""
    entry = entries.get(owner)
    if entry is None:
        while len(entries) >= _MAX_WORKSPACE_SETS:
            entries.popitem(last=False)
        entry = entries[owner] = make()
    elif next(reversed(entries)) != owner:
        entries[owner] = entries.pop(owner, entry)
    return entry


def _workspace(
    owner: Tuple[int, ...], name: str, numel: int, dtype: torch.dtype
) -> torch.Tensor:
    """Grow-only buffer ``name`` of the workspace set of ``owner`` (``_owner``; device ``owner[0]``).

    The quantized operands and scales are rewritten every call, so buffers are shared between the calls
    of one owner, which are ordered; concurrent streams and distinct graph captures get their own set
    (no cross-stream reuse races).  Created during a graph capture, a buffer comes from the graph's
    memory pool.
    """

    buffers = _owner_entry(_WORKSPACES, owner, dict)
    buffer = buffers.get(name)
    if buffer is None or buffer.numel() < numel:
        buffer = torch.empty(
            max(numel, 16), dtype=dtype, device=torch.device("cuda", owner[0])
        )
        buffers[name] = buffer
    return buffer


def _stream_workspace(owner: Tuple[int, ...]) -> _StreamWorkspace:
    """The ``_StreamWorkspace`` of ``owner`` (``_owner``; device ``owner[0]``)."""
    return _owner_entry(_STREAM_WORKSPACES, owner, lambda: _StreamWorkspace(owner[0]))


_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _current_stream(index: int) -> int:
    """The raw handle of the current torch stream of device ``index`` (a CUDA-graph capture stream is
    the current stream as well)."""
    if _raw_stream is not None:
        return _raw_stream(index)
    return torch.cuda.current_stream(index).cuda_stream


def _capture_id(stream: int) -> int:
    """CUDA's id of the capture the raw stream ``stream`` takes part in, 0 when it is not capturing.

    Two captures are distinct even on the same stream, and one capture can be joined by several
    streams (forked branches); the id identifies the capture itself.
    """

    from cuda.bindings import runtime as cudart

    info = cudart.cudaStreamGetCaptureInfo(stream)
    if info[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaStreamGetCaptureInfo failed: {info[0]!r}")
    if info[1] != cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
        return 0
    return int(info[2])


def _capturing(index: int, stream: int) -> bool:
    """Whether the raw stream ``stream`` of device ``index`` takes part in a CUDA-graph capture.

    ``torch.cuda.is_current_stream_capturing`` reads the current stream of the current device, which is
    ``stream`` only when ``index`` is the current device; a call whose tensors live on another device asks
    CUDA about ``stream`` itself.
    """
    if index == torch.cuda.current_device():
        return torch.cuda.is_current_stream_capturing()
    return _capture_id(stream) != 0


def _owner(index: int, stream: int) -> Tuple[int, ...]:
    """The workspace owner of a call on device ``index`` and raw stream ``stream``: ``(index, stream)`` for
    an eager call, ``(index, stream, capture id)`` for a call captured into a CUDA graph.

    Every workspace buffer a call reads or writes, its zero-start control words (arrival counters,
    grid-barrier and head-control words) and its cached launches belong to the workspace set of its
    owner.  Calls of one owner are ordered: eager calls on one stream by the stream, captured calls by
    the stream order of their capture (replays of one graph are ordered too), and the per-(device,
    stream) host lock (``_stream_lock``) keeps the launches of each call contiguous on its stream.  Calls
    of different owners share no buffer: graphs captured one after another on one stream, a graph
    replayed beside eager calls on its capture stream, forked branches of one capture and calls on
    other streams never see each other's words.
    """
    # Concurrency: safe on any number of streams, host threads, CUDA graphs and MPS clients. Exec graphs
    # instantiated more than once from one captured graph (or cloned as child graphs) share this op's
    # workspace and must not run concurrently.
    if _capturing(index, stream):
        capture = _capture_id(stream)
        if capture:
            return (index, stream, capture)
    return (index, stream)


_STREAM_LOCKS: Dict[Tuple[int, int], threading.Lock] = {}
_STREAM_LOCKS_LOCK = threading.Lock()


def _stream_lock(index: int, stream: int) -> threading.Lock:
    """The host lock of (device ``index``, raw stream ``stream``), held by a call from its workspace lookup
    through its last launch: the launches of one call stay contiguous on the stream even when several host
    threads call on it (interleaved launches of two calls would share the owner's control words)."""
    key = (index, stream)
    lock = _STREAM_LOCKS.get(key)
    if lock is None:
        with _STREAM_LOCKS_LOCK:
            lock = _STREAM_LOCKS.setdefault(key, threading.Lock())
    return lock


@functools.cache
def _ffi_device(index: int):
    return tvm_ffi.device(f"cuda:{index}")


def workspace_bytes(tokens: int, num_heads: int, num_segments: int) -> int:
    """Bytes of one workspace set (``_owner``) the operator needs for ``tokens`` packed rows."""

    padded = max(1, _ceil_div(tokens, _BLOCK_N)) * _BLOCK_N
    q8 = k8 = tokens * num_heads * MINIMAX_H3_HEAD_DIM
    vt8 = num_heads * MINIMAX_H3_HEAD_DIM * padded
    scales = (
        tokens * num_heads
        + num_heads * (padded // _BLOCK_N)
        + 2 * num_segments * num_heads * MINIMAX_H3_HEAD_DIM
    )
    partials = _ceil_div(tokens, _BLOCK_M) * num_heads * _PARTIAL_FLOATS
    counters = num_segments * num_heads
    hctr = _HCTR_WORDS_PER_HEAD * num_heads + _HCTR_TAIL_WORDS
    return q8 + k8 + vt8 + 4 * (scales + partials + counters + hctr)


def _softmax_scale_log2(softmax_scale: float) -> float:
    """The log2-domain scale the attention launch takes: ``softmax_scale * log2(e)``."""
    return float(softmax_scale) * _LOG2E


@functools.cache
def _fp8_target(index: int) -> str:
    """The exact target (``"sm_120a"``) every program of the operator ships for device ``index``."""
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(index))
    target = KV_STATS.target_for((capability[0], capability[1]))
    if target is None or any(target not in program.targets for program in _PROGRAMS):
        raise RuntimeError(
            "minimax_h3_sm120_varlen_attention_fp8 requires exact compute capability 12.0 "
            f"(SM120), got {capability[0]}.{capability[1]}"
        )
    return target


def minimax_h3_sm120_quant_varlen_attention_launch_grids(
    num_sms: int,
    *,
    num_heads: int,
    num_tiles: Optional[int] = None,
    num_kblocks: Optional[int] = None,
    num_units: Optional[int] = None,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """The ``(statistics, quantizer, attention)`` launch grids on a device of ``num_sms`` SMs:
    ``max(1, min(n * num_heads, 4 * num_sms))`` grid-stride CTAs over the (query tile, head) and
    (128-key block, head) items of the first two launches, and one persistent attention CTA per SM
    up to the unit count, ``max(1, min(num_sms, max(num_units, 1)))`` (the plan's own grid, the CTA
    count its unit table is laid out for).  A component whose fact is ``None`` is ``None``."""
    sms, heads = int(num_sms), int(num_heads)
    if sms <= 0 or heads <= 0:
        raise ValueError(
            f"num_sms and num_heads must be positive, got {num_sms}, {num_heads}"
        )
    cap = _PREP_CTAS_PER_SM * sms

    def prep(items: Optional[int]) -> Optional[int]:
        return None if items is None else max(1, min(int(items) * heads, cap))

    attention = None if num_units is None else max(1, min(sms, max(int(num_units), 1)))
    return prep(num_tiles), prep(num_kblocks), attention


def minimax_h3_sm120_quant_varlen_attention_overlap_grids(
    num_sms: int,
    *,
    num_units: int,
    total_tokens: Optional[int] = None,
    work: Optional[int] = None,
    padded_tokens: Optional[int] = None,
) -> Optional[Tuple[int, int, int]]:
    """The overlapped form's ``(lead, producers, tail)`` grids on a device of ``num_sms`` SMs, or
    ``None`` when that SM count has no overlap width ``P``: the lead attention over
    ``max(1, min(num_sms - P, num_units))`` CTAs (one per SM), the producers over
    ``min(2 * P, num_sms - lead)`` CTAs (one CTA of two workers on each SM the lead leaves) and the
    tail attention over ``P`` CTAs.  The tuple is in grid order; the launch order is the producers,
    the lead, the tail.  ``P`` is 32 on 170 and 188 SMs; with the plan's ``total_tokens`` and
    ``work`` (the sum of the squared segment lengths) it is 64 when ``work`` is at most 1792 (170
    SMs) or 2304 (188 SMs) times ``total_tokens``, else, with ``padded_tokens`` above 16384 and
    ``work`` at least 4096 times ``total_tokens``, 64 on 170 SMs and 48 on 188 SMs."""
    sms = int(num_sms)
    if sms <= 0:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    width = _OVERLAP_WIDTH.get(sms)
    if width is None:
        return None
    if (
        total_tokens is not None
        and work is not None
        and int(total_tokens) > 0
        and int(work) <= _OVERLAP_LIGHT_MAX_SEGMENT.get(sms, 0) * int(total_tokens)
    ):
        width = _OVERLAP_WIDTH_LIGHT.get(sms, width)
    elif (
        total_tokens is not None
        and work is not None
        and padded_tokens is not None
        and int(padded_tokens) > _OVERLAP_LONG_MIN_PADDED
        and int(work) >= _OVERLAP_LONG_MIN_SEGMENT * int(total_tokens)
    ):
        width = _OVERLAP_WIDTH_LONG.get(sms, width)
    lead = max(1, min(sms - width, int(num_units)))
    return lead, min(_OVERLAP_PRODUCER_WORKERS * width, sms - lead), width


def minimax_h3_sm120_quant_varlen_attention_programs(
    plan: MiniMaxH3VarlenPlan,
) -> Tuple[CakeProgram, ...]:
    """The programs a call of ``plan`` launches, in launch order: ``(PREPROCESS_HEAD_MAJOR,
    ATTENTION_LEAD, ATTENTION_TAIL)`` for a plan of the overlap band (56 heads, an attention work
    ``sum(len^2)`` of 2^23 to 24576^2, a part of 170 or 188 SMs), else ``(KV_STATS, QUANTIZE,
    ATTENTION)``."""
    if plan.overlap:
        return PREPROCESS_HEAD_MAJOR, ATTENTION_LEAD, ATTENTION_TAIL
    return KV_STATS, QUANTIZE, ATTENTION


@register_custom_op(
    "flashinfer::minimax_h3_sm120_varlen_attention_fp8",
    mutates_args=(
        "out",
        "q8_words",
        "k8_words",
        "vt8_words",
        "q_scale",
        "k_scale",
        "v_scale",
        "mean_k",
        "partials",
        "counters",
        "hctr",
    ),
)
def _minimax_h3_sm120_varlen_attention_fp8_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens: Any,
    seg_begin: Any,
    seg_len: Any,
    seg_tile_begin: Any,
    tile_table: Any,
    unit_table: Any,
    q8_words: Any,
    k8_words: Any,
    vt8_words: Any,
    q8_rows: Any,
    k8_rows: Any,
    vt8_rows: Any,
    q_scale: Any,
    k_scale: Any,
    v_scale: Any,
    mean_k: Any,
    partials: Any,
    counters: Any,
    hctr: Any,
    unit_table_head_major: Any,
    num_segments: int,
    num_tiles: int,
    num_units: int,
    num_kblocks: int,
    padded_tokens: int,
    stats_grid: int,
    quantize_grid: int,
    attention_grid: int,
    lead_grid: int,
    producer_grid: int,
    tail_grid: int,
    softmax_scale_log2: float,
    stream: int,
    target: str,
) -> None:
    # Three launches on ``stream`` (the current torch stream): the K / V statistics (per-tile partials,
    # the last tile of each (segment, head) folding them into its K means and V scales) -> E4M3 Q8 /
    # centred K8 / transposed V8 -> attention; or, for a plan of the overlap band (``tail_grid`` > 0),
    # the overlapped form's three launches (``_launch_overlapped``).  The bindings take the host
    # arguments in the kernels' signature order followed by the grid triple; the attention bindings
    # encode their three tensor maps from the byte tensors handed in their descriptor positions
    # (``*_rows``, views of ``*_words``).
    tokens, heads = int(q.shape[0]), int(q.shape[1])
    with tvm_ffi.use_raw_stream(_ffi_device(q.device.index), stream):
        if tail_grid > 0:
            _launch_overlapped(
                target,
                q.device.index,
                stream,
                (
                    q,
                    k,
                    v,
                    tile_table,
                    seg_begin,
                    seg_len,
                    seg_tile_begin,
                    cu_seqlens,
                    partials,
                    counters,
                    mean_k,
                    v_scale,
                    q8_words,
                    k8_words,
                    vt8_words,
                    q_scale,
                    k_scale,
                    hctr,
                ),
                (num_tiles, tokens, padded_tokens, num_segments, heads, num_kblocks),
                (
                    q8_rows,
                    k8_rows,
                    vt8_rows,
                    out,
                    q_scale,
                    k_scale,
                    v_scale,
                    seg_begin,
                    seg_len,
                    unit_table_head_major,
                ),
                (num_units, heads, num_kblocks, softmax_scale_log2, hctr),
                (lead_grid, producer_grid, tail_grid),
            )
            return
        stats, quantize, attention = _entries(target)
        stats(
            k,
            v,
            tile_table,
            seg_begin,
            seg_len,
            seg_tile_begin,
            partials,
            counters,
            mean_k,
            v_scale,
            num_tiles,
            heads,
            stats_grid,
            1,
            1,
        )
        quantize(
            q,
            k,
            v,
            cu_seqlens,
            mean_k,
            v_scale,
            q8_words,
            k8_words,
            vt8_words,
            q_scale,
            k_scale,
            tokens,
            padded_tokens,
            num_segments,
            heads,
            num_kblocks,
            quantize_grid,
            1,
            1,
        )
        attention(
            q8_rows,
            k8_rows,
            vt8_rows,
            out,
            q_scale,
            k_scale,
            v_scale,
            seg_begin,
            seg_len,
            unit_table,
            num_units,
            heads,
            num_kblocks,
            softmax_scale_log2,
            attention_grid,
            1,
            1,
        )


@register_fake_op("flashinfer::minimax_h3_sm120_varlen_attention_fp8")
def _minimax_h3_sm120_varlen_attention_fp8_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens: Any,
    seg_begin: Any,
    seg_len: Any,
    seg_tile_begin: Any,
    tile_table: Any,
    unit_table: Any,
    q8_words: Any,
    k8_words: Any,
    vt8_words: Any,
    q8_rows: Any,
    k8_rows: Any,
    vt8_rows: Any,
    q_scale: Any,
    k_scale: Any,
    v_scale: Any,
    mean_k: Any,
    partials: Any,
    counters: Any,
    hctr: Any,
    unit_table_head_major: Any,
    num_segments: int,
    num_tiles: int,
    num_units: int,
    num_kblocks: int,
    padded_tokens: int,
    stats_grid: int,
    quantize_grid: int,
    attention_grid: int,
    lead_grid: int,
    producer_grid: int,
    tail_grid: int,
    softmax_scale_log2: float,
    stream: int,
    target: str,
) -> None:
    pass


@functools.cache
def _entries(target: str) -> Tuple[Any, Any, Any]:
    """The ``run`` entries of the statistics, quantizer and attention programs on ``target``."""
    return KV_STATS.entry(target), QUANTIZE.entry(target), ATTENTION.entry(target)


@functools.cache
def _overlap_entries(target: str) -> Tuple[Any, Any, Any]:
    """The ``run`` entries of the overlapped form's producers, lead and tail attention on ``target``."""
    return (
        PREPROCESS_HEAD_MAJOR.entry(target),
        ATTENTION_LEAD.entry(target),
        ATTENTION_TAIL.entry(target),
    )


# (device, target) pairs whose overlapped programs ran once (``_launch_overlapped``).
_OVERLAP_LOADED: set = set()


def _launch_overlapped(
    target: str,
    index: int,
    stream: int,
    producer_tensors: Tuple[Any, ...],
    producer_facts: Tuple[int, ...],
    attention_tensors: Tuple[Any, ...],
    attention_facts: Tuple[Any, ...],
    grids: Tuple[int, int, int],
) -> None:
    """The overlapped form's launches on the current tvm-ffi stream, in order: the producers (a plain
    launch: it starts after every earlier launch of the stream has completed), the lead attention (a
    programmatic dependent of the producers) and the tail attention (a programmatic dependent of the
    lead).  ``producer_tensors`` / ``producer_facts`` are the producers' arguments through ``hctr`` /
    after it, ``attention_tensors`` / ``attention_facts`` the attention's through ``unit_table`` / from
    ``total_units`` through ``hctr``; ``grids`` is ``(lead, producers, tail)``.  The caller holds the
    stream's host lock (``_stream_lock``), so the three launches are contiguous on the stream.

    The first overlapped call on a device whose raw stream ``stream`` takes part in no graph capture
    (``_capturing``) first runs the three programs once with no work, in the same order on the call's
    own head-control words (read as one head's, left zero), so their modules load before the call's
    own launches.  This only keeps a module load out of the first call's overlap: a CTA of the overlap
    waits only on CTAs of the earlier launches of its call, so a lazy module load delays a call and
    never blocks it.
    """
    producers, lead, tail = _overlap_entries(target)
    if (index, target) not in _OVERLAP_LOADED and not _capturing(index, stream):
        # one head, no statistics tile, no 128-key block, one segment, no unit
        tokens, padded_tokens = producer_facts[1], producer_facts[2]
        producers(*producer_tensors, 0, tokens, padded_tokens, 1, 1, 0, 1, 1, 1)
        softmax_scale_log2, hctr = attention_facts[3], attention_facts[4]
        for attention in (lead, tail):
            attention(
                *attention_tensors, 0, 1, 1, softmax_scale_log2, hctr, 1, 1, 1, 1, 1
            )
        _OVERLAP_LOADED.add((index, target))
    lead_grid, producer_grid, tail_grid = grids
    producers(*producer_tensors, *producer_facts, producer_grid, 1, 1)
    lead(*attention_tensors, *attention_facts, lead_grid, tail_grid, lead_grid, 1, 1)
    tail(*attention_tensors, *attention_facts, lead_grid, tail_grid, tail_grid, 1, 1)


def _check_thd(name: str, tensor: torch.Tensor, tokens: int, heads: int) -> None:
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16, got {tensor.dtype}")
    if tensor.ndim != 3 or tuple(tensor.shape) != (tokens, heads, MINIMAX_H3_HEAD_DIM):
        raise ValueError(
            f"{name} must have shape [{tokens}, {heads}, {MINIMAX_H3_HEAD_DIM}], got {tuple(tensor.shape)}"
        )
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA tensor")


@flashinfer_api
def minimax_h3_sm120_varlen_attention_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    r"""FP8 (E4M3) non-causal packed-varlen self-attention for MiniMax-H3 on SM120 (GB202).

    Computes, for every segment ``[a, b)`` of ``cu_seqlens`` and every head ``h``::

        out[a:b, h] = softmax(q[a:b, h] @ k[a:b, h]^T * softmax_scale) @ v[a:b, h]

    with FP8 E4M3 tensor-core operands and an FP32 softmax: Q is quantized per token, K per
    128-key block after subtracting its segment's per-channel mean (an exact softmax shift),
    the probabilities are stored as E4M3 with a 2^8 exponent bias and V per (segment, channel).
    Both QK^T and PV run E4M3 ``mma.sync`` m16n8k32 MMAs in the block-scaled ``kind::mxf8f6f4`` form
    with unit scales (bitwise the dense form; GB202 GeForce boards issue the dense FP32-accumulating
    form at half rate); the output is rounded to BF16 once.
    Three runtime-variable generated programs, launched in order on the current stream (the K / V
    statistics with their per-segment fold, the quantizer, the attention), serve any ``tokens``
    and any segment lengths (including empty segments and lengths below one tile); the grids are
    ``minimax_h3_sm120_quant_varlen_attention_launch_grids``.  A plan of 56 heads whose attention
    work (the sum of the squared segment lengths) lies in [2^23, 24576^2] runs, on a part of 170 or
    188 SMs, the overlapped form instead (``minimax_h3_sm120_quant_varlen_attention_programs``,
    ``minimax_h3_sm120_quant_varlen_attention_overlap_grids``): the head-major pre-processing
    producers start first and the attention starts each head as soon as its operands are written,
    with the same output bit for bit.  The quantized operands
    live in a grow-only workspace per (device, stream), and per capture for a call captured into a
    CUDA graph (``workspace_bytes``); the most recent 8 workspace sets and 256 segment plans are
    kept.  The programs ship for ``sm_120a`` alone: a call with a non-empty segment on a device of
    another compute capability than 12.0 raises ``RuntimeError``.

    Concurrency: safe on any number of streams, host threads, CUDA graphs and MPS clients. Exec graphs
    instantiated more than once from one captured graph (or cloned as child graphs) share this op's
    workspace and must not run concurrently.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Contiguous ``bfloat16`` CUDA tensors of shape ``[tokens, heads, 128]`` (packed THD
        layout; MiniMax-H3 uses 56 heads).
    cu_seqlens : torch.Tensor
        ``int32`` tensor of shape ``[segments + 1]`` on the same device with ``cu_seqlens[0] = 0``,
        non-decreasing entries and ``cu_seqlens[-1] = tokens``.
    out : Optional[torch.Tensor]
        Optional pre-allocated output of the same shape/dtype as ``q``; allocated when omitted.
    cu_seqlens_host : Optional[Sequence[int]]
        Host copy of ``cu_seqlens`` (avoids a synchronizing device-to-host copy).  The
        segment plan is cached per ``(cu_seqlens, heads, device)`` (most recent 256).
    softmax_scale : Optional[float]
        Softmax scale; defaults to ``1 / sqrt(128)``.

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[tokens, heads, 128]`` attention output.
    """

    if q.ndim != 3:
        raise ValueError(f"q must be [tokens, heads, 128], got shape {tuple(q.shape)}")
    tokens, heads = int(q.shape[0]), int(q.shape[1])
    if not 1 <= heads < _MAX_HEADS:
        raise ValueError(f"heads must lie in [1, {_MAX_HEADS}), got {heads}")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _check_thd(name, tensor, tokens, heads)
    if not cu_seqlens.is_cuda or cu_seqlens.device != q.device:
        raise ValueError(
            "cu_seqlens must be an int32 CUDA tensor on the same device as q"
        )
    bounds = normalize_cu_seqlens(cu_seqlens, cu_seqlens_host)
    if bounds[-1] != tokens:
        raise ValueError(f"cu_seqlens[-1] = {bounds[-1]} must equal tokens = {tokens}")
    if out is None:
        out = torch.empty_like(q)
    else:
        _check_thd("out", out, tokens, heads)
        if out.device != q.device:
            raise ValueError("out must be on the same device as q")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(MINIMAX_H3_HEAD_DIM)
    index = _device_index(q.device)
    plan = _plan_at(bounds, heads, index)
    if plan.num_units == 0:
        return out  # every segment is empty: nothing to write
    target = _fp8_target(index)
    stream = _current_stream(index)
    # the stream's host lock from the workspace lookup through the last launch (``_stream_lock``)
    with _stream_lock(index, stream):
        _minimax_h3_sm120_varlen_attention_fp8_impl(
            q,
            k,
            v,
            out,
            *_stream_workspace(_owner(index, stream)).launch(plan, index).args,
            _softmax_scale_log2(softmax_scale),
            stream,
            target,
        )
    return out


__all__ = [
    "MINIMAX_H3_HEAD_DIM",
    "MINIMAX_H3_NUM_HEADS",
    "MiniMaxH3VarlenPlan",
    "assign_unit_slots",
    "minimax_h3_sm120_quant_varlen_attention_launch_grids",
    "minimax_h3_sm120_quant_varlen_attention_overlap_grids",
    "minimax_h3_sm120_quant_varlen_attention_programs",
    "minimax_h3_sm120_varlen_attention_fp8",
    "normalize_cu_seqlens",
    "workspace_bytes",
]
