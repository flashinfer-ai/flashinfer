# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The per-pool state of the GDN class and its three placements.

An fp32 state pool ``[S, HV, 128, 128]`` passed as ``initial_state`` holds, per slot, the state at the sequence's last
64-token boundary. The class keeps next to it, keyed by the pool tensor, what the pool cannot hold: the raw rows of the
open chunk (at most 63 per slot), the rows of the chunk's solve that the step kernel caches per (slot, v-head) (the
normalized keys, the log2 gates and betas, the M and X rows of the blocked inverse, the VKS and NV rows), a bf16 copy
of the boundary state and the open-row count. Three placements compute the class's term:

* the step kernel (:meth:`GdnPool.step`): one launch appends one token per slot and returns its row; the 64th row of
  a chunk commits the next boundary state. It reads its count on the device, so a decode step needs no host metadata;
* the step kernel once per token (:meth:`GdnPool.extend` for short calls);
* the pinned prefill over the open chunk plus the new tokens (:meth:`GdnPool.extend` otherwise): the whole chunks in
  one call that writes the boundary state, the open tail in a second call from it. The tail's cached rows are rebuilt
  by the step kernel before the slot's next step (``pending``); until then the slot's count cell is negative and a
  step launched without that rebuild (a captured graph) returns NaN rows for it.
"""

from __future__ import annotations

import functools
import weakref
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .spec import CHUNK, HEAD_DIM, extend_split, uses_step_kernel

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


def _pinned(rows, cu, cu_cpu, pool, read_slots, write_slots, scale, output=None):
    """The class's term: the chunked prefill without context parallelism, from the pool's fp32 states."""
    from ...gdn_prefill import chunk_gated_delta_rule

    q, k, v, g, beta = rows
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
    )
    return out


class GdnPool:
    def __init__(self, slots: int, pool: torch.Tensor, num_q_heads: int) -> None:
        S, HV, HQ = int(slots), int(pool.shape[1]), int(num_q_heads)
        dev = pool.device
        self.slots, self.HV, self.HQ, self.device = S, HV, HQ, dev

        def zeros(*shape, dtype=_IO):
            return torch.zeros(shape, dtype=dtype, device=dev)

        self.hist_q = zeros(S, CHUNK, HQ, HEAD_DIM)
        self.hist_k = zeros(S, CHUNK, HQ, HEAD_DIM)
        self.hist_v = zeros(S, CHUNK, HV, HEAD_DIM)
        self.hist_g = zeros(S, CHUNK, HV, dtype=torch.float32)
        self.hist_beta = zeros(S, CHUNK, HV, dtype=torch.float32)
        self.kn = zeros(S, CHUNK, HQ, HEAD_DIM)
        self.state_bf16 = zeros(S, HV, HEAD_DIM, HEAD_DIM)
        self.cnt = torch.full((S, HV), STALE, dtype=torch.int32, device=dev)
        self.ring_l = zeros(S, HV, CHUNK, dtype=torch.float32)
        self.ring_b = zeros(S, HV, CHUNK, dtype=torch.float32)
        self.m_rows = zeros(S, HV, CHUNK, CHUNK)
        self.x_rows = zeros(S, HV, CHUNK, CHUNK)
        self.vks_rows = zeros(S, HV, CHUNK, HEAD_DIM)
        self.nv_rows = zeros(S, HV, CHUNK, HEAD_DIM)
        #: slot -> (rows the step kernel has built, open rows): tails a prefill left for the next step to rebuild
        self.pending: Dict[int, Tuple[int, int]] = {}

    @property
    def _hist(self):
        S = self.slots
        return (
            self.hist_q.view(S * CHUNK, self.HQ, HEAD_DIM),
            self.hist_k.view(S * CHUNK, self.HQ, HEAD_DIM),
            self.hist_v.view(S * CHUNK, self.HV, HEAD_DIM),
            self.hist_g.view(S * CHUNK, self.HV),
            self.hist_beta.view(S * CHUNK, self.HV),
        )

    def _idx(self, values, dtype=torch.int64) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values), dtype=dtype, device=self.device)

    def _launch(self, pool, rows, slots, out, scale) -> None:
        q, k, v, g, beta = rows
        _module().gdn_batch_invariant_decode_step(
            q,
            k,
            v,
            g,
            beta,
            slots,
            out,
            pool,
            self.state_bf16,
            self.cnt,
            self.ring_l,
            self.ring_b,
            self.kn,
            self.m_rows,
            self.x_rows,
            self.vks_rows,
            self.nv_rows,
            self.hist_q,
            self.hist_k,
            self.hist_v,
            self.hist_g,
            self.hist_beta,
            scale,
        )

    def _set_counts(self, slot_list: Sequence[int], values: Sequence[int]) -> None:
        if len(slot_list) == 0:
            return
        cells = self._idx(values, torch.int32)[:, None].expand(-1, self.HV)
        self.cnt.index_copy_(0, self._idx(slot_list), cells.contiguous())

    def counts(self, slot_list: Sequence[int]) -> List[int]:
        """The open rows of each slot (one device read for the slots without a pending tail)."""
        live = [self.pending[s][1] if s in self.pending else -1 for s in slot_list]
        ask = [i for i, n in enumerate(live) if n < 0]
        if ask:
            cells = self.cnt[:, 0].index_select(
                0, self._idx([slot_list[i] for i in ask])
            )
            for i, c in zip(ask, cells.tolist(), strict=True):
                live[i] = int(c) & (CHUNK - 1)
        return live

    def reset(self, slot_list: Sequence[int]) -> None:
        """The slots' states are taken as chunk boundaries: open rows dropped, the bf16 copy refreshed by the next step."""
        for s in slot_list:
            self.pending.pop(s, None)
        self._set_counts(slot_list, [STALE] * len(slot_list))

    def catch_up(self, pool: torch.Tensor, slot_list: Sequence[int], scale) -> None:
        """Rebuild the cached rows of the listed slots' pending tails: the step kernel over their raw rows."""
        work = [
            (s, *self.pending[s]) for s in dict.fromkeys(slot_list) if s in self.pending
        ]
        if not work:
            return
        self._set_counts(
            [s for s, _, _ in work], [b if b > 0 else STALE for _, b, _ in work]
        )
        hist = self._hist
        scratch = torch.empty(
            (len(work), self.HV, HEAD_DIM), dtype=_IO, device=self.device
        )
        for j in range(max(live - built for _, built, live in work)):
            act = [(s, built + j) for s, built, live in work if built + j < live]
            rows = self._idx([s * CHUNK + r for s, r in act])
            self._launch(
                pool,
                tuple(h.index_select(0, rows) for h in hist),
                self._idx([s for s, _ in act], torch.int32),
                scratch[: len(act)],
                scale,
            )
        for s, _, _ in work:
            del self.pending[s]

    def step(self, pool, slots, q, k, v, g, beta, scale, out) -> None:
        """One token per slot: ``out[i]`` is the row of ``slots[i]``'s sequence at its position."""
        if self.pending:
            self.catch_up(pool, slots.tolist(), scale)
        self._launch(pool, (q, k, v, g, beta), slots, out, scale)

    def extend_whole_chunks(self, pool, slots, offs, cu, packed, scale, out) -> bool:
        """A prefill of whole chunks from the slots' states: the caller's tensors as they are, nothing read back before
        the kernel and the bookkeeping after it. False (nothing done) when a sequence leaves an open tail."""
        lens = [b - a for a, b in zip(offs, offs[1:], strict=False)]
        if uses_step_kernel(max(lens)) or any(n % CHUNK for n in lens):
            return False
        if cu is None:
            cu = self._idx(offs, torch.int32)
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
        boundary. ``out`` receives the rows."""
        lens = [offs[i + 1] - offs[i] for i in range(len(slot_list))]
        if uses_step_kernel(max(lens)):
            self.reset([s for s, c in zip(slot_list, counts, strict=True) if c == 0])
            self._extend_steps(pool, slots, slot_list, lens, offs, packed, scale, out)
        else:
            self._extend_prefill(
                pool, slot_list, counts, lens, offs, packed, scale, out
            )

    def _extend_steps(
        self, pool, slots, slot_list, lens, offs, packed, scale, out
    ) -> None:
        self.catch_up(pool, slot_list, scale)
        N, T = len(lens), lens[0]
        if all(n == T for n in lens):
            if T == 1:
                self._launch(pool, packed, slots, out, scale)
                return
            # token-major copies: launch j takes token j of every sequence as one contiguous batch
            rows = tuple(
                x.view(N, T, *x.shape[1:]).transpose(0, 1).contiguous() for x in packed
            )
            o = torch.empty((T, N, self.HV, HEAD_DIM), dtype=_IO, device=self.device)
            for j in range(T):
                self._launch(pool, tuple(r[j] for r in rows), slots, o[j], scale)
            out.view(N, T, self.HV, HEAD_DIM).copy_(o.transpose(0, 1))
            return
        for j in range(max(lens)):
            act = [i for i, n in enumerate(lens) if j < n]
            rows = self._idx([offs[i] + j for i in act])
            o = torch.empty(
                (len(act), self.HV, HEAD_DIM), dtype=_IO, device=self.device
            )
            self._launch(
                pool,
                tuple(x.index_select(0, rows) for x in packed),
                self._idx([slot_list[i] for i in act], torch.int32),
                o,
                scale,
            )
            out.index_copy_(0, rows, o)

    def _run_pinned(self, pool, segs, packed, write: bool, scale, out) -> None:
        """One pinned call over ``segs``: per sequence ``(slot, open rows replayed, first new row, new rows)``."""
        cu = [0]
        for _, n_hist, _, n_new in segs:
            cu.append(cu[-1] + n_hist + n_new)
        slots = self._idx([s for s, _, _, _ in segs], torch.int32)
        write_slots = slots if write else torch.full_like(slots, -1)
        cu_dev = self._idx(cu, torch.int32)
        has_hist = any(n_hist for _, n_hist, _, _ in segs)
        contiguous = not has_hist and all(
            a[2] + a[3] == b[2] for a, b in zip(segs, segs[1:], strict=False)
        )
        if contiguous:
            a, b = segs[0][2], segs[-1][2] + segs[-1][3]
            rows = tuple(x[a:b] for x in packed)
            _pinned(rows, cu_dev, cu, pool, slots, write_slots, scale, output=out[a:b])
            return
        if not has_hist and len(segs) <= 32:
            rows = tuple(
                torch.cat([x[st : st + n] for _, _, st, n in segs]) for x in packed
            )
            o = _pinned(rows, cu_dev, cu, pool, slots, write_slots, scale)
            for (_, _, st, n), c in zip(segs, cu, strict=False):
                out[st : st + n].copy_(o[c : c + n])
            return
        hist_src, hist_dst, new_src, new_dst = [], [], [], []
        for (slot, n_hist, start, n_new), base in zip(segs, cu, strict=False):
            hist_src.append(slot * CHUNK + np.arange(n_hist))
            hist_dst.append(base + np.arange(n_hist))
            new_src.append(start + np.arange(n_new))
            new_dst.append(base + n_hist + np.arange(n_new))
        src, dst = (
            self._idx(np.concatenate(new_src)),
            self._idx(np.concatenate(new_dst)),
        )
        if has_hist:
            rows = tuple(
                torch.empty((cu[-1], *x.shape[1:]), dtype=x.dtype, device=x.device)
                for x in packed
            )
            for r, x in zip(rows, packed, strict=True):
                r.index_copy_(0, dst, x.index_select(0, src))
            hs, hd = (
                self._idx(np.concatenate(hist_src)),
                self._idx(np.concatenate(hist_dst)),
            )
            for r, h in zip(rows, self._hist, strict=True):
                r.index_copy_(0, hd, h.index_select(0, hs))
            o = _pinned(rows, cu_dev, cu, pool, slots, write_slots, scale)
            out.index_copy_(0, src, o.index_select(0, dst))
        else:
            rows = tuple(x.index_select(0, src) for x in packed)
            o = _pinned(rows, cu_dev, cu, pool, slots, write_slots, scale)
            out.index_copy_(0, src, o)

    def _extend_prefill(
        self, pool, slot_list, counts, lens, offs, packed, scale, out
    ) -> None:
        # three pinned calls at most: open chunks to their boundary, whole chunks, open tails
        heads, bodies, tails = [], [], []
        hist_src, hist_dst, cells = [], [], []
        for i, slot in enumerate(slot_list):
            c, n, start = counts[i], lens[i], offs[i]
            built = self.pending.get(slot, (c, c))[0] if c else 0
            if c and c + n >= CHUNK:
                heads.append((slot, c, start, CHUNK - c))
                start, n, c, built = start + CHUNK - c, n - (CHUNK - c), 0, 0
            whole, tail = extend_split(c, n)
            if whole:
                bodies.append((slot, 0, start, whole))
                start, n, built = start + whole, n - whole, 0
            if tail:
                tails.append((slot, c, start, n))
                hist_src.append(start + np.arange(n))
                hist_dst.append(slot * CHUNK + c + np.arange(n))
                self.pending[slot] = (built, tail)
            else:
                self.pending.pop(slot, None)
            cells.append(PENDING if tail else STALE)
        for segs, write in ((heads, True), (bodies, True), (tails, False)):
            if segs:
                self._run_pinned(pool, segs, packed, write, scale, out)
        if tails:
            src, dst = (
                self._idx(np.concatenate(hist_src)),
                self._idx(np.concatenate(hist_dst)),
            )
            for h, x in zip(self._hist, packed, strict=True):
                h.index_copy_(0, dst, x.index_select(0, src))
        self._set_counts(slot_list, cells)
