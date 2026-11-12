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
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..support import unsupported
from .spec import CHUNK, CLASS, GATE_LOG2_E, HEAD_DIM, STAMP

_IO = torch.bfloat16
_POOLS: Dict[Tuple[int, int, Tuple[int, ...]], Tuple["ExactPool", Any]] = {}


@functools.cache
def _module():
    from ...jit.kda_batch_invariant import gen_kda_batch_invariant_module

    return gen_kda_batch_invariant_module().build_and_load()


def pool_key(pool: torch.Tensor) -> Tuple[int, int, Tuple[int, ...]]:
    """The identity of a pool: its storage, its offset in it and its shape, so every alias of one buffer (a view, a
    ``detach()``) is the same pool."""
    return (pool.untyped_storage()._cdata, pool.storage_offset(), tuple(pool.shape))


def get_pool(pool: torch.Tensor) -> "ExactPool":
    """The class's state next to ``pool``: created on first use, kept while the pool's storage is alive."""
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

    for dead in [k for k, (_, ref) in _POOLS.items() if ref.expired()]:
        del _POOLS[dead]
    ep = ExactPool(pool)
    _POOLS[key] = (ep, StorageWeakRef(pool.untyped_storage()))
    return ep


class ExactPool:
    def __init__(self, pool: torch.Tensor) -> None:
        S, HV = int(pool.shape[0]), int(pool.shape[1])
        dev = pool.device
        self.slots, self.H, self.device = S, HV, dev

        def zeros(*shape, dtype=_IO):
            return torch.zeros(shape, dtype=dtype, device=dev)

        self.cnt = zeros(S, HV, dtype=torch.int32)
        # NaN matches no state: every (slot, head) builds its bf16 copy at its first step
        self.stamp = torch.full((S, HV, STAMP), float("nan"), device=dev)
        self.state_bf16 = zeros(S, HV, HEAD_DIM, HEAD_DIM)
        self.prefix = zeros(S, HV, HEAD_DIM, dtype=torch.float32)
        self.k_inv_rows = zeros(S, HV, CHUNK, HEAD_DIM)
        self.k_inv_f32_rows = zeros(S, HV, CHUNK, HEAD_DIM, dtype=torch.float32)
        self.rhs_rows = zeros(S, HV, CHUNK, HEAD_DIM)
        self.update_rows = zeros(S, HV, CHUNK, HEAD_DIM)
        self.l_rows = zeros(S, HV, CHUNK, CHUNK)

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

    def counts(self, pool: torch.Tensor, slot_list: Sequence[int]) -> List[int]:
        """The open rows of each slot (one device read). A slot whose state the caller replaced is begun first, as
        the step kernel would; heads of one slot that disagree are refused."""
        idx = torch.as_tensor(list(slot_list), dtype=torch.int64, device=self.device)
        cur = pool[:, :, 0, :STAMP].index_select(0, idx).view(torch.int32)
        old = self.stamp.index_select(0, idx).view(torch.int32)
        moved = (cur != old).flatten(1).any(1).tolist()
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
        none = slot_ids if stride else None
        _module().kda_batch_invariant_decode_span(
            q,
            k,
            v,
            g,
            beta,
            slot_ids,
            none if stride else starts,
            none if stride else lens,
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
