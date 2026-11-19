# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The per-pool state of the KDA class and its two placements.

An fp32 state pool ``[S, HV, 128, 128]`` passed as ``initial_state`` holds, per slot, the state at the sequence's last
16-token boundary. The class keeps next to it, keyed by the pool tensor, what the pool cannot hold: per (slot, head) the
rows of the open chunk that later rows of the chunk depend on (the decayed-inverse keys in bf16 and fp32, the ``rhs`` and
bf16 ``update`` rows, the rows of ``L``, the running log2 decay prefix), a bf16 copy of the boundary state, the open-row
count and a stamp of the state's first fp32 values that tells a state the caller replaced (a new request in the slot)
from one the class wrote. Two placements compute the class's term:

* the step kernel (:meth:`ExactPool.step`): one launch appends one token per slot and returns its row; the 16th row of a
  chunk commits the next boundary state. Count and stamp are read on the device, so a step needs no host metadata and
  is one node of a captured CUDA graph;
* the chunked prefill on its pinned route (``api._pinned``) for the rows that form whole chunks from a boundary.

``api._append`` appends several tokens per sequence with both: the step kernel up to the next boundary, the prefill
kernel over the whole chunks, the step kernel for the tail. :meth:`ExactPool.span` is the step kernel looping over
several rows per slot in one launch.

A multi-token verify (:meth:`ExactPool.verify`) is that span behind a snapshot of what it overwrites (the count and the
decay prefix; the boundary state and the cached rows a completed chunk replaces, only when the span crosses a boundary)
and a copy of its token rows; :meth:`ExactPool.accept` takes a slot that keeps fewer rows than were run back to the
snapshot and appends the kept rows again through the same span. The two kernels of
``include/flashinfer/kda/batch_invariant_verify.cuh`` move bytes and counts only, so every row is still the step
kernel's and the class's fingerprint sources are untouched.
"""

from __future__ import annotations

import functools
import math
import weakref
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..support import unsupported
from .spec import CHUNK, CLASS, GATE_LOG2_E, HEAD_DIM, STAMP

_IO = torch.bfloat16
_Key = Tuple[int, int, Tuple[int, ...]]
_POOLS: Dict[_Key, Tuple["ExactPool", Any]] = {}
#: Side states an overlapping extent took over from: kept with the buffer, since a captured graph may still write them.
_RETIRED: Dict[_Key, "ExactPool"] = {}


@functools.cache
def _module():
    from ...jit.kda_batch_invariant import gen_kda_batch_invariant_module

    return gen_kda_batch_invariant_module().build_and_load()


@functools.cache
def _verify_module():
    from ...jit.kda_batch_invariant_verify import gen_kda_batch_invariant_verify_module

    return gen_kda_batch_invariant_verify_module().build_and_load()


def _release(storage: int) -> None:
    for table in (_POOLS, _RETIRED):
        for key in [k for k in table if k[0] == storage]:
            del table[key]


def pool_key(pool: torch.Tensor) -> _Key:
    """The identity of a pool: its storage, its offset in it and its shape, so every alias of the same extent of one
    buffer (a full view, a ``detach()``) is the same pool."""
    return (pool.untyped_storage()._cdata, pool.storage_offset(), tuple(pool.shape))


def get_pool(pool: torch.Tensor) -> "ExactPool":
    """The class's state next to ``pool``: created on first use, released when the pool's storage is freed."""
    key = pool_key(pool)
    entry = _POOLS.get(key)
    if entry is not None and not entry[1].expired():
        return entry[0]
    if torch.cuda.is_current_stream_capturing():
        raise unsupported(
            CLASS,
            "a pool first used inside a CUDA graph capture",
            "one eager call on the pool before the capture",
        )
    from torch.multiprocessing.reductions import StorageWeakRef

    for dead in {k[0] for k, (_, ref) in _POOLS.items() if ref.expired()}:
        _release(dead)
    if pool.data_ptr() % 16:
        raise unsupported(
            CLASS,
            "a state pool that does not start at a 16-byte boundary",
            "a pool whose first element is 16-byte aligned (the kernels load 16 bytes at a time)",
        )
    first, last = key[1], key[1] + pool.numel()
    for other in list(_POOLS):
        storage, offset, shape = other
        if storage != key[0] or offset >= last or first >= offset + math.prod(shape):
            continue
        # another extent of a tracked buffer starts with empty rows: fine only if the tracked one has none open
        if bool((_POOLS[other][0].cnt > 0).any()):
            raise unsupported(
                CLASS,
                "a state pool that overlaps a pool of another shape or offset on the same buffer while that pool has open chunks",
                "the same pool extent on every call (its open rows are kept per extent)",
            )
        _RETIRED[other] = _POOLS.pop(other)[0]
    ep = _RETIRED.pop(key, None)
    if ep is None:
        ep = ExactPool(pool)
    else:
        ep.cnt.fill_(-1)
        if ep.record is not None:
            ep.record.meta.zero_()
    storage = pool.untyped_storage()
    if not any(k[0] == key[0] for k in (*_POOLS, *_RETIRED)):
        weakref.finalize(storage, _release, key[0])
    _POOLS[key] = (ep, StorageWeakRef(storage))
    return ep


class ExactPool:
    def __init__(self, pool: torch.Tensor) -> None:
        S, HV = int(pool.shape[0]), int(pool.shape[1])
        dev = pool.device
        self.slots, self.H, self.device = S, HV, dev

        def zeros(*shape, dtype=_IO):
            return torch.zeros(shape, dtype=dtype, device=dev)

        # a negative count: never begun, so every (slot, head) builds its bf16 copy at its first step
        self.cnt = torch.full((S, HV), -1, dtype=torch.int32, device=dev)
        self.stamp = zeros(S, HV, STAMP, dtype=torch.float32)
        self.state_bf16 = zeros(S, HV, HEAD_DIM, HEAD_DIM)
        self.prefix = zeros(S, HV, HEAD_DIM, dtype=torch.float32)
        self.k_inv_rows = zeros(S, HV, CHUNK, HEAD_DIM)
        self.k_inv_f32_rows = zeros(S, HV, CHUNK, HEAD_DIM, dtype=torch.float32)
        self.rhs_rows = zeros(S, HV, CHUNK, HEAD_DIM)
        self.update_rows = zeros(S, HV, CHUNK, HEAD_DIM)
        self.l_rows = zeros(S, HV, CHUNK, CHUNK)
        #: The verify record (snapshot and token rows), allocated by the first verify on the pool.
        self.record: Optional[VerifyRecord] = None
        self._outgrown: List[VerifyRecord] = []

    def mutable(self) -> Tuple[torch.Tensor, ...]:
        """The tensors a step may change besides the pool and the cached rows of the row it writes."""
        return (self.cnt, self.stamp, self.state_bf16, self.prefix)

    def step(
        self,
        pool: torch.Tensor,
        slot_ids: torch.Tensor,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        g_t: torch.Tensor,
        beta_t: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float,
        scale: float,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """One token per active slot (``slot_ids`` int32 ``[B]``; token inputs ``[B, HV, 128]`` / ``[B, HV]`` bf16,
        contiguous); writes bf16 ``[B, HV, 128]`` into ``out`` and returns it. One launch."""
        _module().kda_batch_invariant_decode_step(
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            slot_ids,
            out,
            pool,
            self.state_bf16,
            self.cnt,
            self.prefix,
            self.k_inv_rows,
            self.k_inv_f32_rows,
            self.rhs_rows,
            self.update_rows,
            self.l_rows,
            self.stamp,
            A_log,
            dt_bias.reshape(self.H, HEAD_DIM),
            float(scale),
            float(lower_bound) * GATE_LOG2_E * 0.5,
        )
        return out

    def begin(self, pool: torch.Tensor, slots: torch.Tensor) -> None:
        """The pool states of ``slots`` (int32) are taken as chunk boundaries: open rows dropped, stamp and bf16 copy
        refreshed. One launch."""
        if slots.numel():
            _module().kda_batch_invariant_begin(
                slots, pool, self.state_bf16, self.cnt, self.stamp
            )
            if self.record is not None:
                # a slot taken as a boundary has no verify left to settle
                self.record.meta.index_fill_(0, slots.long(), 0)

    def counts(self, pool: torch.Tensor, slot_list: Sequence[int]) -> List[int]:
        """The open rows of each slot (one device read). A slot whose state the caller replaced is begun first, as
        the step kernel would; heads of one slot that disagree are refused."""
        idx = torch.as_tensor(list(slot_list), dtype=torch.int64, device=self.device)
        cur = pool[:, :, 0, :STAMP].index_select(0, idx).view(torch.int32)
        old = self.stamp.index_select(0, idx).view(torch.int32)
        fresh = self.cnt.index_select(0, idx) < 0
        moved = ((cur != old).flatten(1).any(1) | fresh.any(1)).tolist()
        self.begin(
            pool,
            torch.tensor(
                [s for s, m in zip(slot_list, moved, strict=True) if m],
                dtype=torch.int32,
                device=self.device,
            ),
        )
        cells = self.cnt.index_select(0, idx) & (CHUNK - 1)
        if bool((cells != cells[:, :1]).any()):
            raise unsupported(
                CLASS,
                "a slot whose heads hold different open-row counts",
                "one sequence per slot, every head stepped together",
            )
        return cells[:, 0].tolist()

    def span(
        self,
        pool: torch.Tensor,
        slot_ids: torch.Tensor,
        packed: Tuple[torch.Tensor, ...],
        out: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float,
        scale: float,
        *,
        starts: Optional[torch.Tensor] = None,
        lens: Optional[torch.Tensor] = None,
        stride: int = 0,
    ) -> None:
        """Sequence ``i`` appends packed rows ``[starts[i], starts[i] + lens[i])`` (int32, on the device; ``stride``
        rows from row ``i * stride`` when ``stride`` is given instead) to slot ``slot_ids[i]`` and writes their rows
        into ``out`` at the same positions. One launch: each block loops over its rows and commits every chunk it
        completes."""
        q, k, v, g, beta = packed
        if not stride and (starts is None or lens is None):
            raise ValueError("span needs starts and lens, or a stride")
        # with a stride the kernel ignores starts and lens: any int32 tensor stands in for them
        _module().kda_batch_invariant_decode_span(
            q,
            k,
            v,
            g,
            beta,
            slot_ids,
            slot_ids if stride else starts,
            slot_ids if stride else lens,
            int(stride),
            out,
            pool,
            self.state_bf16,
            self.cnt,
            self.prefix,
            self.k_inv_rows,
            self.k_inv_f32_rows,
            self.rhs_rows,
            self.update_rows,
            self.l_rows,
            self.stamp,
            A_log,
            dt_bias.reshape(self.H, HEAD_DIM),
            float(scale),
            float(lower_bound) * GATE_LOG2_E * 0.5,
        )

    def _side(self, pool: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return (
            pool,
            self.cnt,
            self.prefix,
            self.k_inv_rows,
            self.k_inv_f32_rows,
            self.rhs_rows,
            self.update_rows,
            self.l_rows,
            self.stamp,
        )

    def verify_record(self, tokens: int) -> "VerifyRecord":
        """The pool's verify record with room for ``tokens`` rows per slot (grown outside a graph capture only)."""
        rec = self.record
        if rec is not None and rec.cap >= tokens:
            return rec
        if torch.cuda.is_current_stream_capturing():
            raise unsupported(
                CLASS,
                f"a verify of {tokens} tokens first run inside a CUDA graph capture",
                "one eager verify of at least as many tokens on the pool before the capture",
            )
        new = VerifyRecord(self.slots, self.H, tokens, self.device)
        if rec is not None:
            # the pending verifies stay settleable; a graph captured on the smaller record may still write it
            new.adopt(rec)
            self._outgrown.append(rec)
        self.record = new
        return new

    def verify(
        self,
        pool: torch.Tensor,
        slot_ids: torch.Tensor,
        packed: Tuple[torch.Tensor, ...],
        out: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float,
        scale: float,
        tokens: int,
    ) -> None:
        """``tokens`` packed rows per sequence appended to its slot behind a snapshot; their rows go to ``out``. Two
        launches, nothing read back."""
        rec = self.verify_record(tokens)
        _verify_module().kda_batch_invariant_verify_snapshot(
            *packed, slot_ids, *self._side(pool), *rec.snapshot(), *rec.tokens()
        )
        self.span(
            pool,
            slot_ids,
            packed,
            out,
            A_log,
            dt_bias,
            lower_bound,
            scale,
            stride=tokens,
        )

    def accept(
        self,
        pool: torch.Tensor,
        slot_ids: torch.Tensor,
        accepted: Optional[torch.Tensor],
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float,
        scale: float,
    ) -> None:
        """Settle each slot's verify with ``accepted[i]`` of its rows kept (``None``: all of them): a slot that keeps
        fewer goes back to its snapshot and appends the kept rows again. Two launches, nothing read back; a no-op
        on a pool that never ran a verify."""
        rec = self.record
        if rec is None:
            return
        spans = torch.empty(
            (2, slot_ids.numel()), dtype=torch.int32, device=self.device
        )
        _verify_module().kda_batch_invariant_verify_accept(
            slot_ids,
            slot_ids if accepted is None else accepted,
            int(accepted is not None),
            pool,
            self.state_bf16,
            *self._side(pool)[1:],
            *rec.snapshot(),
            spans[0],
            spans[1],
            rec.cap,
        )
        self.span(
            pool,
            slot_ids,
            rec.tokens(),
            rec.rows,
            A_log,
            dt_bias,
            lower_bound,
            scale,
            starts=spans[0],
            lens=spans[1],
        )


class VerifyRecord:
    """Per slot: what the last verify's span overwrote and its token rows (``cap`` rows of room)."""

    def __init__(self, S: int, HV: int, cap: int, dev: torch.device) -> None:
        self.cap = cap

        def zeros(*shape, dtype=_IO):
            return torch.zeros(shape, dtype=dtype, device=dev)

        # per (slot, head): rows pending, the count after them, the count before them, the cached rows saved
        self.meta = zeros(S, HV, 4, dtype=torch.int32)
        self.prefix = zeros(S, HV, HEAD_DIM, dtype=torch.float32)
        self.state = zeros(S, HV, HEAD_DIM, HEAD_DIM, dtype=torch.float32)
        self.k_inv_rows = zeros(S, HV, CHUNK, HEAD_DIM)
        self.k_inv_f32_rows = zeros(S, HV, CHUNK, HEAD_DIM, dtype=torch.float32)
        self.rhs_rows = zeros(S, HV, CHUNK, HEAD_DIM)
        self.update_rows = zeros(S, HV, CHUNK, HEAD_DIM)
        self.l_rows = zeros(S, HV, CHUNK, CHUNK)
        self.q = zeros(S * cap, HV, HEAD_DIM)
        self.k = zeros(S * cap, HV, HEAD_DIM)
        self.v = zeros(S * cap, HV, HEAD_DIM)
        self.g = zeros(S * cap, HV, HEAD_DIM)
        self.beta = zeros(S * cap, HV)
        #: Where the rows of a replay go: they repeat rows the verify already returned.
        self.rows = zeros(S * cap, HV, HEAD_DIM)

    def snapshot(self) -> Tuple[torch.Tensor, ...]:
        return (
            self.meta,
            self.prefix,
            self.state,
            self.k_inv_rows,
            self.k_inv_f32_rows,
            self.rhs_rows,
            self.update_rows,
            self.l_rows,
        )

    def tokens(self) -> Tuple[torch.Tensor, ...]:
        return (self.q, self.k, self.v, self.g, self.beta)

    def adopt(self, old: "VerifyRecord") -> None:
        for mine, theirs in zip(self.snapshot(), old.snapshot(), strict=True):
            mine.copy_(theirs)
        S = self.meta.shape[0]
        for mine, theirs in zip(self.tokens(), old.tokens(), strict=True):
            mine.view(S, self.cap, *mine.shape[1:])[:, : old.cap].copy_(
                theirs.view(S, old.cap, *theirs.shape[1:])
            )
