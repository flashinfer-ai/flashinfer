# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in exact sequence split of the Cake GDN prefill on SM100, SM103 and
SM107.

``chunk_gated_delta_rule(..., backend="cake_gdn", use_affine_split=True)``
cuts a sequence of ``L`` tokens into ``P`` chunk-aligned segments and runs
them in one wave instead of one walk:

* pass 1 (``PASS1``) walks segment 0 from the initial state into the output
  and, for every later segment ``s``, walks it once from zeros (its local
  output ``o_loc`` and state ``B_s``) and once from the identity with
  ``v = 0`` (its transfer ``c`` and ``M_s``);
* pass 2 (``PASS2_W*``) composes the carried states
  ``h_in(s + 1) = h_in(s) M_s + B_s`` and corrects every later segment's
  output, ``O = o_loc + h_in(s) c``, rounded to BF16 once, and writes the
  final state.

The result is the same recurrence in another summation order: it is not
bitwise equal to the unsplit route when ``P > 1``.

Segment rule (both kernels read the live ``cu_seqlens``): with ``start =
clamp(cu[n], 0, T)``, ``L = clamp(cu[n + 1], start, T) - start`` and
``nchunks = ceil(L / 64)``, ``P = 1`` when ``L == 0`` and
``clamp(nchunks // c_min, 1, p_cap)`` otherwise; segment ``s`` holds the
tokens ``[start + 64 * floor(s * nchunks / P), start + min(64 * floor((s +
1) * nchunks / P), L))``.  ``p_cap`` and ``c_min`` are plan constants of the
device (its target and SM count) and the value-head count only
(:func:`plan_constants`), so a sequence's bits do not depend on the other
sequences of a batch.

A call of ``N`` sequences launches with the cap ``p_launch``, the segment
count of its longest possible sequence (:func:`plan_split`): every live
sequence is at most that long, so the clamp gives it the same ``P`` and the
same segments as at ``p_cap``, and only the slots of the segment-state
workspace move.  ``p_launch = 1`` runs pass 1 alone, every sequence in one
segment, over no split workspace.

A plan entry owns its workspaces, allocated and zeroed once: the launch path
allocates nothing, reads nothing back from the device and issues no memset
(pass 2 leaves its flags and ticket at zero after every launch), so a warm
entry is CUDA-graph capturable.  Eager entries live in a bounded LRU keyed by
the geometry and the stream; a capture moves the warm entry of its key and
stream out of the LRU into a pinned list, so the captured graph keeps its
workspaces for the life of the process and eager calls never share them.
Replays of one graph must not overlap each other.
"""

from __future__ import annotations

import collections
import functools
from typing import Any, NamedTuple, Optional

import torch
import tvm_ffi

from ...jit import cake_gdn_affine_split as _programs

__all__ = [
    "C_MIN",
    "MAX_PASS1_TILES",
    "MAX_TOKENS",
    "P_MAX",
    "SPLIT_MIN_TOKENS",
    "SplitPlan",
    "TARGETS",
    "chunk_gated_delta_rule_gdn_affine_split_sm100",
    "clear_affine_split_plans",
    "pass1_grid",
    "pass2_grid",
    "plan_constants",
    "plan_split",
    "segment_count",
    "tile_cap",
]

#: The targets the programs ship for.
TARGETS = ("sm_100a", "sm_103a", "sm_107a")
_HEAD_SIZE = 128
_CHUNK = 64
#: Bytes of the pass-1 device tensor-map scratch per CTA.
_TMAP_BYTES = 512
#: The largest segment count per target and value-head count, measured on
#: B200 (sm_100a), GB300 (sm_103a) and VR200 (sm_107a). One pass-1 wave
#: bounds it too (:func:`tile_cap`).
P_MAX: dict[str, dict[int, int]] = {
    "sm_100a": {4: 8, 8: 5, 16: 2, 32: 1},
    "sm_103a": {4: 8, 8: 5, 16: 2, 32: 1},
    "sm_107a": {4: 8, 8: 6, 16: 3, 32: 2},
}
#: The fewest chunks per segment per target and value-head count.
C_MIN: dict[str, dict[int, int]] = {
    "sm_100a": {4: 8, 8: 8, 16: 16, 32: 16},
    "sm_103a": {4: 8, 8: 8, 16: 16, 32: 16},
    "sm_107a": {4: 8, 8: 8, 16: 12, 32: 32},
}
#: No sequence of at most this many tokens is split, on any target and at any
#: value-head count: a consequence of :data:`C_MIN`, which the device rule
#: applies to every live length (15 chunks at 8 chunks per segment).
SPLIT_MIN_TOKENS = _CHUNK * (
    2 * min(min(table.values()) for table in C_MIN.values()) - 1
)
#: The pass-1 kernel's limits on its tiles and tokens.
MAX_PASS1_TILES = 131072
MAX_TOKENS = 1 << 20
#: The pass-2 program name of each value-head count (its compose slice width).
_PASS2_NAMES: dict[int, str] = {
    4: "PASS2_W4",
    8: "PASS2_W8",
    16: "PASS2_W16",
    32: "PASS2_W16",
}
_SLICE_WIDTH: dict[int, int] = {4: 4, 8: 8, 16: 16, 32: 16}
_EAGER_LIMIT = 8

_PASS1_ARGS = (
    "Q",
    "K",
    "V",
    "O",
    "gate",
    "beta",
    "cu_seqlens",
    "state_indices",
    "initial_state",
    "output_state",
    "checkpoint_state",
    "cu_checkpoints",
    "tensormap_workspace",
    "initial_state_stride_slot",
    "output_state_stride_slot",
    "checkpoint_every_n_tokens",
    "scale",
    "num_seqs",
    "total_tokens",
    "num_q_heads",
    "num_v_heads",
    "total_tiles",
    "o_loc",
    "c",
    "bm",
    "p_cap",
    "c_min",
    "grid_x",
    "grid_y",
    "grid_z",
)
_PASS2_ARGS = (
    "o_loc",
    "c_rows",
    "c_tma",
    "bm",
    "hpub",
    "hpub_tma",
    "output",
    "output_state",
    "flags",
    "ticket",
    "cu_seqlens",
    "output_state_stride_slot",
    "total_tokens",
    "num_seqs",
    "num_heads",
    "p_cap",
    "c_min",
    "grid_x",
    "grid_y",
    "grid_z",
)


def tile_cap(num_heads: int, sm_count: int) -> int:
    """The largest segment count whose pass-1 tiles of one sequence fit one
    wave: ``num_heads * (4 * P - 2) <= sm_count``, at least 1. A batch may
    have more tiles than CTAs."""

    return max(1, (sm_count // num_heads + 2) // 4)


def plan_constants(
    num_heads: int,
    sm_count: int,
    split_count: Optional[int] = None,
    *,
    target: str,
) -> tuple[int, int]:
    """``(p_cap, c_min)``: the planner's constants of ``target``, or with
    ``split_count`` the forced cap ``min(split_count, tile_cap)`` at one chunk
    per segment."""

    cap = tile_cap(num_heads, sm_count)
    if split_count is not None:
        return min(int(split_count), cap), 1
    return min(cap, P_MAX[target][num_heads]), C_MIN[target][num_heads]


def segment_count(length: int, p_cap: int, c_min: int) -> int:
    """The segment count of a sequence of ``length`` tokens."""

    if length <= 0:
        return 1
    return max(1, min(p_cap, -(-length // _CHUNK) // c_min))


class SplitPlan(NamedTuple):
    p_cap: int
    c_min: int
    #: The segment count of the longest possible sequence, the cap the call
    #: launches with (``p_launch``): 1 means every sequence of the call runs
    #: in one segment, pass 1 alone.
    p_bound: int


def plan_split(
    *,
    num_heads: int,
    max_seqlen: int,
    sm_count: int,
    target: str,
    split_count: Optional[int] = None,
) -> SplitPlan:
    """The plan constants of ``target`` and the segment count of a sequence of
    ``max_seqlen`` tokens, an upper bound on every live sequence's (the
    device rule is monotone in the length).

    Without ``split_count``, ``max_seqlen <= SPLIT_MIN_TOKENS`` gives
    ``p_bound = 1``."""

    p_cap, c_min = plan_constants(num_heads, sm_count, split_count, target=target)
    return SplitPlan(p_cap, c_min, segment_count(max_seqlen, p_cap, c_min))


def pass1_grid(
    num_seqs: int, num_heads: int, p_cap: int, sm_count: int
) -> tuple[int, int]:
    """``(total_tiles, grid_x)`` of pass 1: segment 0 has ``2 * H`` tiles,
    every later segment ``4 * H``; at most ``sm_count`` CTAs, so a CTA runs
    several tiles when there are more tiles than CTAs."""

    tiles = num_seqs * num_heads * (4 * p_cap - 2)
    return tiles, min(tiles, sm_count)


def pass2_grid(
    num_seqs: int,
    num_heads: int,
    total_tokens: int,
    p_cap: int,
    slice_width: int,
    sm_count: int,
) -> int:
    """Pass 2's grid: every compose ticket, then correction CTAs up to two
    CTAs per SM (at least 32), never more than the work bound."""

    compose = num_seqs * num_heads * _HEAD_SIZE // slice_width
    bound = compose + 2 * num_heads * (-(-total_tokens // 128) + num_seqs * p_cap)
    grid = min(bound, compose + max(2 * sm_count - compose, 32))
    # Progress needs a correction CTA past the compose tickets.
    assert grid > compose
    return grid


def _pass2_program(num_heads: int) -> Any:
    """The pass-2 program of a value-head count."""

    return getattr(_programs, _PASS2_NAMES[num_heads])


def _check_args(program: Any, expected: tuple[str, ...]) -> None:
    if tuple(program.args) != expected:
        raise RuntimeError(
            f"the generated {program.name} binding takes {list(program.args)}; "
            f"the host spells {list(expected)}"
        )


@functools.cache
def _entries(device_index: int, pass2: Any) -> tuple[Any, Optional[Any]]:
    """The loaded pass-1 and pass-2 entries of one device (``pass2`` None:
    pass 1 alone)."""

    _check_args(_programs.PASS1, _PASS1_ARGS)
    capability = torch.cuda.get_device_capability(device_index)
    target = _programs.PASS1.target_for(capability)
    if target not in TARGETS:
        raise RuntimeError(
            f"the Cake GDN affine-split programs ship no build for compute "
            f"capability {capability[0]}.{capability[1]}"
        )
    first = _programs.PASS1.entry(target)
    if pass2 is None:
        return first, None
    _check_args(pass2, _PASS2_ARGS)
    return first, pass2.entry(target)


class _Entry:
    """One plan entry: launch constants and the workspaces it owns."""

    def __init__(
        self,
        *,
        device: torch.device,
        sm_count: int,
        num_seqs: int,
        total_tokens: int,
        num_q_heads: int,
        num_heads: int,
        p_cap: int,
        c_min: int,
        has_initial_state: bool,
        output_final_state: bool,
        gates_present: tuple[bool, bool],
        cu_int64: bool,
    ) -> None:
        n, t, h = num_seqs, total_tokens, num_heads
        self.num_seqs, self.total_tokens = n, t
        self.num_q_heads, self.num_heads = num_q_heads, h
        self.p_cap, self.c_min = p_cap, c_min
        self.tiles1, self.grid1 = pass1_grid(n, h, p_cap, sm_count)
        # p_cap = 1: every sequence in one segment, which writes no split
        # workspace, so pass 1 takes placeholders and pass 2 does not run.
        self.run_pass2 = p_cap >= 2
        pass2 = _pass2_program(h) if self.run_pass2 else None
        self.grid2 = (
            pass2_grid(n, h, t, p_cap, _SLICE_WIDTH[h], sm_count)
            if self.run_pass2
            else 0
        )
        self.pass1, self.pass2 = _entries(int(device.index), pass2)
        f32, bf16, i32 = torch.float32, torch.bfloat16, torch.int32
        d = _HEAD_SIZE
        rows, slots = (t, n * p_cap) if self.run_pass2 else (0, 0)
        self.o_loc = torch.zeros(max(rows * h * d, 1), dtype=f32, device=device)
        self.c = torch.zeros(max(rows * h * d, 1), dtype=bf16, device=device)
        self.bm = torch.zeros(max(slots * 2 * h * d * d, 1), dtype=f32, device=device)
        if self.run_pass2:
            self.o_loc = self.o_loc.view(t, h, d)
            self.c = self.c.view(t, h, d)
            self.bm = self.bm.view(n, 2, p_cap, h, d, d)
        self.hpub = torch.zeros(
            n * max(p_cap - 1, 1) * h * 2 if self.run_pass2 else 1,
            d,
            d,
            dtype=bf16,
            device=device,
        )
        self.flags = torch.zeros(
            n if self.run_pass2 else 1, p_cap, h, 2, dtype=torch.uint32, device=device
        )
        self.ticket = torch.zeros(4, dtype=torch.uint32, device=device)
        self.tensormap_workspace = torch.zeros(
            self.grid1 * _TMAP_BYTES, dtype=torch.uint8, device=device
        )
        self.empty_i32 = torch.zeros(1, dtype=i32, device=device)
        self.empty_state = torch.zeros(1, dtype=f32, device=device)
        self.zero_state = (
            None
            if has_initial_state
            else torch.zeros(n, h, d, d, dtype=f32, device=device)
        )
        self.scratch_state = (
            None
            if output_final_state
            else torch.empty(n, h, d, d, dtype=f32, device=device)
        )
        self.unit_gates = (
            None if all(gates_present) else torch.ones(t, h, dtype=f32, device=device)
        )
        self.cu32 = torch.zeros(n + 1, dtype=i32, device=device) if cu_int64 else None
        self.warm = False

    def launch(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        scale: float,
        cu_seqlens: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        output: torch.Tensor,
        output_state: Optional[torch.Tensor],
    ) -> None:
        cu = cu_seqlens
        if self.cu32 is not None:
            self.cu32.copy_(cu_seqlens)
            cu = self.cu32
        h0 = initial_state if initial_state is not None else self.zero_state
        state = output_state if output_state is not None else self.scratch_state
        assert h0 is not None and state is not None
        gate = g if g is not None else self.unit_gates
        update = beta if beta is not None else self.unit_gates
        n, t, h = self.num_seqs, self.total_tokens, self.num_heads
        with tvm_ffi.use_torch_stream():
            self.pass1(
                q,
                k,
                v,
                output,
                gate,
                update,
                cu,
                self.empty_i32,
                h0,
                state,
                self.empty_state,
                self.empty_i32,
                self.tensormap_workspace,
                int(h0.stride(0)),
                int(state.stride(0)),
                0,
                float(scale),
                n,
                t,
                self.num_q_heads,
                h,
                self.tiles1,
                self.o_loc,
                self.c,
                self.bm,
                self.p_cap,
                self.c_min,
                self.grid1,
                1,
                1,
            )
            if self.pass2 is not None:
                self.pass2(
                    self.o_loc,
                    self.c,
                    self.c,
                    self.bm,
                    self.hpub,
                    self.hpub,
                    output,
                    state,
                    self.flags,
                    self.ticket,
                    cu,
                    int(state.stride(0)),
                    t,
                    n,
                    h,
                    self.p_cap,
                    self.c_min,
                    self.grid2,
                    1,
                    1,
                )
        self.warm = True


_EAGER: "collections.OrderedDict[tuple[Any, ...], _Entry]" = collections.OrderedDict()
#: The entries captured graphs own, by capture id and plan key.
_PINNED: dict[tuple[int, tuple[Any, ...]], _Entry] = {}


def clear_affine_split_plans() -> None:
    """Drop every plan entry, the pinned entries of captured graphs included.
    Unsafe while a graph captured over an entry may still replay."""

    _EAGER.clear()
    _PINNED.clear()


def _capture_id(stream: int) -> int:
    """CUDA's id of the capture the raw stream ``stream`` takes part in, 0
    when it is not capturing. Two captures are distinct even on one stream."""

    from cuda.bindings import runtime as cudart

    info = cudart.cudaStreamGetCaptureInfo(stream)
    if info[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaStreamGetCaptureInfo failed: {info[0]!r}")
    if info[1] != cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
        return 0
    return int(info[2])


def _entry_for(key: tuple[Any, ...], stream: int, make: Any) -> _Entry:
    """The plan entry of ``key`` on the raw stream ``stream``.

    A capture takes the warmed eager entry and owns it for the life of the
    process; a later call of the same key in the same capture reuses it.
    The key holds the stream, so those calls are stream-ordered, as repeated
    eager calls are."""

    capture = _capture_id(stream)
    if capture:
        pinned = _PINNED.get((capture, key))
        if pinned is not None:
            return pinned
        entry = _EAGER.get(key)
        if entry is None or not entry.warm:
            raise RuntimeError(
                "affine-split plan must be warmed before CUDA graph capture: "
                "run the same call once eagerly on this stream first"
            )
        del _EAGER[key]
        _PINNED[(capture, key)] = entry
        return entry
    entry = _EAGER.get(key)
    if entry is not None:
        _EAGER.move_to_end(key)
        return entry
    entry = make()
    _EAGER[key] = entry
    while len(_EAGER) > _EAGER_LIMIT:
        _EAGER.popitem(last=False)
    return entry


def chunk_gated_delta_rule_gdn_affine_split_sm100(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    output: torch.Tensor,
    output_state: Optional[torch.Tensor],
    plan: SplitPlan,
    sm_count: int,
) -> None:
    """Launch pass 1 and pass 2 of a validated call at the cap
    ``plan.p_bound`` (pass 1 alone when it is 1). ``output_state`` is the
    caller's final-state destination or ``None``."""

    device = q.device
    device_index = int(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    num_seqs = int(cu_seqlens.numel()) - 1
    total_tokens, num_q_heads = int(q.shape[0]), int(q.shape[1])
    num_heads = int(v.shape[1])
    gates_present = (g is not None, beta is not None)
    stream = torch.cuda.current_stream(device_index).cuda_stream
    key = (
        device_index,
        sm_count,
        num_seqs,
        total_tokens,
        num_q_heads,
        int(k.shape[1]),
        num_heads,
        initial_state is not None,
        output_final_state,
        gates_present,
        cu_seqlens.dtype,
        plan.p_bound,
        plan.c_min,
        int(stream),
    )

    def make() -> _Entry:
        return _Entry(
            device=torch.device("cuda", device_index),
            sm_count=sm_count,
            num_seqs=num_seqs,
            total_tokens=total_tokens,
            num_q_heads=num_q_heads,
            num_heads=num_heads,
            p_cap=plan.p_bound,
            c_min=plan.c_min,
            has_initial_state=initial_state is not None,
            output_final_state=output_final_state,
            gates_present=gates_present,
            cu_int64=cu_seqlens.dtype == torch.int64,
        )

    entry = _entry_for(key, int(stream), make)
    entry.launch(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output=output,
        output_state=output_state if output_final_state else None,
    )
