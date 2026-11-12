# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The per-pool state of the GDN class and its placements.

An fp32 state pool ``[S, HV, 128, 128]`` passed as ``initial_state`` holds, per slot, the state at the sequence's last
64-token boundary. The class keeps next to it, keyed by the pool's storage, what the pool cannot hold: the rows of the
open chunk's solve that the step kernel caches per (slot, v-head) (the normalized keys, the log2 gates and betas, the M
and X rows of the blocked inverse, the VKS and NV rows), a bf16 copy of the boundary state and the open-row count. Two
kernels compute the class's term:

* the step kernel: one launch appends one token per slot and returns its row (:meth:`GdnPool.step`), or, in its span
  form, a run of tokens per slot (:meth:`GdnPool.span`); the 64th row of a chunk commits the next boundary state. It
  reads its count on the device, so neither form needs host metadata;
* the pinned prefill for whole chunks from a boundary (:meth:`GdnPool.extend_whole_chunks`, :meth:`GdnPool.extend`):
  one call in place that also returns the rows of the open tails and stores each sequence's last boundary state as a
  checkpoint. The tails' raw rows are kept next to the pool (one launch) and their cached rows built by one span
  launch before the slot's next step (:meth:`GdnPool.rebuild`); until then the slot's count cell is negative and a
  step launched without that rebuild (a captured graph) returns NaN rows for it.
"""

from __future__ import annotations

import functools
import weakref
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .spec import CHUNK, HEAD_DIM

_IO = torch.bfloat16
#: Count-cell values of the step kernel: bit 8 asks for a refresh of the bf16 state copy, negative marks unbuilt rows.
STALE = 256
PENDING = -1
_POOLS: Dict[Tuple[int, int, int, torch.device], "GdnPool"] = {}


@functools.cache
def _module():
    from ...jit.gdn_batch_invariant import gen_gdn_batch_invariant_module

    return gen_gdn_batch_invariant_module().build_and_load()


def pool_key(pool: torch.Tensor) -> Tuple[int, int, int, torch.device]:
    """The pool's identity: its storage, where it starts in it and its head count (any view of it is the same pool)."""
    return (
        pool.untyped_storage().data_ptr(),
        pool.storage_offset(),
        int(pool.shape[1]),
        pool.device,
    )


def get_pool(pool: torch.Tensor, num_q_heads: int) -> "GdnPool":
    """The class's state next to ``pool`` (created on first use for every slot the pool's storage can hold from the
    pool's offset, released when that storage is freed, so any view of a long-lived buffer keeps its state)."""
    key = pool_key(pool)
    gp = _POOLS.get(key)
    if gp is None:
        storage = pool.untyped_storage()
        slot_bytes = 4 * int(pool.shape[1]) * HEAD_DIM * HEAD_DIM
        capacity = (storage.nbytes() - 4 * pool.storage_offset()) // slot_bytes
        gp = GdnPool(max(capacity, int(pool.shape[0])), pool, num_q_heads)
        _POOLS[key] = gp
        weakref.finalize(storage, _POOLS.pop, key, None)
    return gp


def _pinned(
    rows,
    cu,
    cu_cpu,
    pool,
    read_slots,
    write_slots,
    scale,
    output=None,
    checkpoints=None,
):
    """The class's term: the chunked prefill without context parallelism, from the pool's fp32 states.
    ``checkpoints = (rows, starts)`` stores 64-token boundary states in the pool rows listed (-1 skips one)."""
    from ...gdn_prefill import chunk_gated_delta_rule

    q, k, v, g, beta = rows
    extra = {}
    if checkpoints is not None:
        extra = dict(
            state_checkpoints=pool,
            checkpoint_every_n_tokens=CHUNK,
            checkpoint_state_indices=checkpoints[0],
            checkpoint_cu_starts=checkpoints[1],
        )
    out, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=pool,
        output_state=pool,
        output_final_state=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        state_indices=read_slots,
        output_state_indices=write_slots,
        output=output,
        use_qk_l2norm_in_kernel=True,
        backend="flashinfer",
        use_cp=False,
        gate_domain="log",
        **extra,
    )
    return out


class GdnPool:
    def __init__(self, slots: int, pool: torch.Tensor, num_q_heads: int) -> None:
        S, HV, HQ = int(slots), int(pool.shape[1]), int(num_q_heads)
        dev = pool.device
        self.slots, self.HV, self.HQ, self.device = S, HV, HQ, dev

        def zeros(*shape, dtype=_IO):
            return torch.zeros(shape, dtype=dtype, device=dev)

        self.kn = zeros(S, CHUNK, HQ, HEAD_DIM)
        self.state_bf16 = zeros(S, HV, HEAD_DIM, HEAD_DIM)
        self.cnt = torch.full((S, HV), STALE, dtype=torch.int32, device=dev)
        self.ring_l = zeros(S, HV, CHUNK, dtype=torch.float32)
        self.ring_b = zeros(S, HV, CHUNK, dtype=torch.float32)
        self.m_rows = zeros(S, HV, CHUNK, CHUNK)
        self.x_rows = zeros(S, HV, CHUNK, CHUNK)
        self.vks_rows = zeros(S, HV, CHUNK, HEAD_DIM)
        self.nv_rows = zeros(S, HV, CHUNK, HEAD_DIM)
        #: the ``prefix_lens`` of a call whose sequences all start at a chunk boundary
        self.zero_prefix = zeros(S, dtype=torch.int32)
        #: slot -> open rows a prefill left for the next step to build (their raw rows are in ``_hist``)
        self.pending: Dict[int, int] = {}
        self._hist: Optional[Tuple[torch.Tensor, ...]] = None
        self._no_out = torch.empty((0, HV, HEAD_DIM), dtype=_IO, device=dev)

    def _side(self):
        return (
            self.state_bf16,
            self.cnt,
            self.ring_l,
            self.ring_b,
            self.kn,
            self.m_rows,
            self.x_rows,
            self.vks_rows,
            self.nv_rows,
        )

    def _idx(self, values, dtype=torch.int32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values), dtype=dtype, device=self.device)

    def stash(self, slots, starts, ends, slot_list, tails, packed, pool) -> None:
        """Keep rows ``[starts[i], ends[i])`` of ``packed``, the open tail a prefill left in ``slots[i]``, for
        :meth:`rebuild`; a slot without a tail is marked for a refresh of its bf16 state copy."""
        if self._hist is None:
            rows = self.slots * CHUNK
            self._hist = tuple(
                torch.zeros((rows, *x.shape[1:]), dtype=x.dtype, device=self.device)
                for x in packed
            )
        _module().gdn_batch_invariant_stash(
            *packed, slots, starts, ends, pool, self.cnt, *self._hist
        )
        for slot, n in zip(slot_list, tails, strict=True):
            if n:
                self.pending[slot] = n
            else:
                self.pending.pop(slot, None)

    def rebuild(self, pool: torch.Tensor, slot_list: Sequence[int]) -> None:
        """Build the cached rows of the listed slots' pending tails: one span launch over their kept rows."""
        work = [
            (s, self.pending.pop(s))
            for s in dict.fromkeys(slot_list)
            if s in self.pending
        ]
        if not work:
            return
        m = self._idx(
            [
                [s for s, _ in work],
                [s * CHUNK for s, _ in work],
                [s * CHUNK + n for s, n in work],
            ]
        )
        self.span(
            pool,
            m[0],
            m[1],
            m[2],
            self.zero_prefix[: len(work)],
            self._hist,
            1.0,
            self._no_out,
        )

    def step(self, pool, slots, q, k, v, g, beta, scale, out) -> None:
        """One token per slot: ``out[i]`` is the row of ``slots[i]``'s sequence at its position."""
        if self.pending:
            self.rebuild(pool, slots.tolist())
        _module().gdn_batch_invariant_decode_step(
            q, k, v, g, beta, slots, out, pool, *self._side(), scale
        )

    def span(self, pool, slots, starts, ends, prefix, packed, scale, out) -> None:
        """Rows ``[starts[i], ends[i])`` of ``packed`` appended to ``slots[i]`` in one launch (int32 device tensors).
        ``prefix[i] % 64`` must be the slot's open-row count, or 0 to start from the slot's state as a chunk boundary;
        the rows of a slot that holds another count are NaN and the slot is left as it was. An empty run at a boundary
        only marks the bf16 state copy for a refresh."""
        q, k, v, g, beta = packed
        _module().gdn_batch_invariant_decode_span(
            q,
            k,
            v,
            g,
            beta,
            slots,
            starts,
            ends,
            prefix,
            out,
            pool,
            *self._side(),
            scale,
        )

    def extend_whole_chunks(self, pool, slots, offs, cu, packed, scale, out) -> bool:
        """A prefill of whole chunks from the slots' states: the caller's tensors as they are, nothing read back before
        the kernel and the bookkeeping after it. False (nothing done) when a sequence leaves an open tail."""
        if any((b - a) % CHUNK for a, b in zip(offs, offs[1:], strict=False)):
            return False
        if cu is None:
            cu = self._idx(offs)
        _pinned(packed, cu, list(offs), pool, slots, slots, scale, output=out)
        if self.pending:
            for s in slots.tolist():
                self.pending.pop(s, None)
        self.cnt.index_fill_(0, slots.long(), STALE)
        return True

    def extend(
        self,
        pool: torch.Tensor,
        slots: torch.Tensor,
        slot_list: Sequence[int],
        counts: Sequence[int],
        offs: Sequence[int],
        cu: Optional[torch.Tensor],
        packed: Tuple[torch.Tensor, ...],
        scale,
        out: torch.Tensor,
    ) -> None:
        """Append tokens ``[offs[i], offs[i + 1])`` of ``packed`` to slot ``slot_list[i]`` (``slots`` and the int32
        offsets ``cu`` on the device), which holds ``counts[i]`` open rows; a slot at 0 starts from its state as a chunk
        boundary. ``out`` receives the rows. At least one sequence brings a whole chunk."""
        N = len(slot_list)
        if not any(counts):
            # one prefill call in place, each sequence's last boundary state stored as a checkpoint; the rows after
            # it are kept for the step kernel to build from
            rows, starts, tail_starts = [], [0], []
            for i, slot in enumerate(slot_list):
                chunks = (offs[i + 1] - offs[i]) // CHUNK
                rows += [-1] * (chunks - 1) + [slot] * (chunks > 0)
                starts.append(len(rows))
                tail_starts.append(offs[i] + chunks * CHUNK)
            meta = self._idx(
                np.concatenate([rows, starts, [-1] * N, tail_starts, offs[1:], offs])
            )
            cuts = np.cumsum([0, len(rows), N + 1, N, N, N, N + 1])
            ck_rows, ck_starts, no_write, t0, t1, cu_up = (
                meta[a:b] for a, b in zip(cuts, cuts[1:], strict=False)
            )
            _pinned(
                packed,
                cu if cu is not None else cu_up,
                list(offs),
                pool,
                slots,
                no_write,
                scale,
                output=out,
                checkpoints=(ck_rows, ck_starts),
            )
            tails = [offs[i + 1] - t for i, t in enumerate(tail_starts)]
            self.stash(slots, t0, t1, slot_list, tails, packed, pool)
            return
        # open chunks first (to their boundary, or the whole piece), then whole chunks, then open tails
        self.rebuild(pool, [s for s, c in zip(slot_list, counts, strict=True) if c])
        for s in slot_list:
            self.pending.pop(s, None)
        pre, body, post = [], [], []
        for i, slot in enumerate(slot_list):
            c, start, end = counts[i], offs[i], offs[i + 1]
            if c:
                cut = min(end, start + CHUNK - c)
                pre.append((slot, start, cut, c))
                if cut - start < CHUNK - c:
                    continue
                start = cut
            whole = (end - start) // CHUNK * CHUNK
            if whole:
                body.append((slot, start, whole))
            if whole or end > start:
                post.append((slot, start + whole, end, 0))
        for runs in (pre, None, post):
            if runs is None:
                if body:
                    self._pinned_gathered(pool, body, packed, scale, out)
            elif runs:
                m = self._idx(np.asarray(runs).T.copy())
                self.span(pool, m[0], m[1], m[2], m[3], packed, scale, out)

    def _pinned_gathered(self, pool, segs, packed, scale, out) -> None:
        """One pinned call over whole-chunk pieces ``(slot, first row, rows)`` that do not tile the packed rows."""
        cu = [0]
        for _, _, n in segs:
            cu.append(cu[-1] + n)
        slots = self._idx([s for s, _, _ in segs])
        src = self._idx(
            np.concatenate([st + np.arange(n) for _, st, n in segs]), torch.int64
        )
        rows = tuple(x.index_select(0, src) for x in packed)
        o = _pinned(rows, self._idx(cu), cu, pool, slots, slots, scale)
        out.index_copy_(0, src, o)
