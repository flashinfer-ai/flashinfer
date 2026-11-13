# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The per-pool state of the KDA class ``kda.bf16.s16`` and its two kernels.

A bf16 state pool ``[S, HV, 128, 128]`` passed as ``initial_state`` holds, per slot, the state at the sequence's last
16-token boundary. The class keeps next to it, keyed by the pool tensor, the open chunk of every (slot, head): the
decayed-inverse key rows (bf16), the ``u`` rows and the log2-decay prefix rows (fp32), the open-row count, and a stamp
of the state's first values that tells a state the caller replaced from one the class wrote.

* :meth:`ClassSPool.step`: one launch appends one token per slot and returns its row; the 16th row of a chunk folds
  the chunk into the boundary state. Count and stamp are read on the device, so a step needs no host metadata and is
  one node of a captured CUDA graph.
* :meth:`ClassSPool.prefill`: one launch appends any number of tokens per slot, folding every chunk it completes and
  leaving the tail as the slot's open rows.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Tuple

import torch

from ..support import unsupported
from .spec import CHUNK, CLASS, HEAD_DIM

_IO = torch.bfloat16
_POOLS: Dict[int, Tuple["ClassSPool", Any, Tuple[int, Tuple[int, ...]]]] = {}


@functools.cache
def _module():
    from ...jit.kda_batch_invariant_class_s import (
        gen_kda_batch_invariant_class_s_module,
    )

    return gen_kda_batch_invariant_class_s_module().build_and_load()


def get_pool(pool: torch.Tensor) -> "ClassSPool":
    """The class's state next to ``pool``: created on first use, kept while the pool's storage is alive. A pool is
    identified by its storage, so a view of all of it or a ``detach()`` is the same pool; another offset or shape of
    a storage the class already holds is refused."""
    key = pool.untyped_storage()._cdata
    view = (pool.storage_offset(), tuple(pool.shape))
    if len(_POOLS) > 1:
        for dead in [k for k, (_, ref, _) in _POOLS.items() if ref.expired()]:
            del _POOLS[dead]
    entry = _POOLS.get(key)
    if entry is not None and not entry[1].expired():
        if entry[2] != view:
            raise unsupported(
                CLASS,
                "another slice or shape of a state pool the class already keeps rows for",
                "the pool tensor itself (or a view of all of it) in every call",
            )
        return entry[0]
    if torch.cuda.is_current_stream_capturing():
        raise unsupported(
            CLASS,
            "a pool first used inside a CUDA graph capture",
            "one eager call on the pool before the capture",
        )
    from torch.multiprocessing.reductions import StorageWeakRef

    sp = ClassSPool(pool)
    _POOLS[key] = (sp, StorageWeakRef(pool.untyped_storage()), view)
    return sp


class ClassSPool:
    def __init__(self, pool: torch.Tensor) -> None:
        S, HV = int(pool.shape[0]), int(pool.shape[1])
        dev = pool.device
        self.slots, self.H, self.device = S, HV, dev
        self.key_rows = torch.zeros(S, HV, CHUNK, HEAD_DIM, dtype=_IO, device=dev)
        self.u_rows = torch.zeros(S, HV, CHUNK, HEAD_DIM, device=dev)
        self.prefix_rows = torch.zeros(S, HV, CHUNK, HEAD_DIM, device=dev)
        self.cnt = torch.zeros(S, HV, dtype=torch.int32, device=dev)
        self.stamp = torch.zeros(S, HV, 2, dtype=torch.int64, device=dev)
        self._cu: Dict[Tuple[int, int], torch.Tensor] = {}
        self._ids: Dict[int, torch.Tensor] = {}

    def _side(self) -> Tuple[torch.Tensor, ...]:
        return (self.key_rows, self.u_rows, self.prefix_rows, self.cnt, self.stamp)

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
        """One token per slot (``slot_ids`` int32 ``[B]``; token inputs ``[B, HV, 128]`` / ``[B, HV]`` bf16,
        contiguous); writes bf16 ``[B, HV, 128]`` into ``out`` and returns it. One launch."""
        _module().kda_batch_invariant_class_s_decode_step(
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            A_log,
            dt_bias,
            pool,
            slot_ids,
            out,
            *self._side(),
            float(scale),
            float(lower_bound),
        )
        return out

    def prefill(
        self,
        pool: torch.Tensor,
        slot_ids: torch.Tensor,
        packed: Tuple[torch.Tensor, ...],
        cu: torch.Tensor,
        out: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float,
        scale: float,
        *,
        fresh: bool,
    ) -> torch.Tensor:
        """Sequence ``i`` appends packed rows ``[cu[i], cu[i + 1])`` (``cu`` int32 ``[B + 1]`` on the device) to slot
        ``slot_ids[i]`` and writes their rows into ``out``. ``fresh`` takes every slot's state as a chunk boundary (its
        open rows are dropped); otherwise the rows continue the slot's open chunk. One launch."""
        q, k, v, g, beta = packed
        _module().kda_batch_invariant_class_s_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            pool,
            slot_ids,
            cu,
            out,
            *self._side(),
            float(scale),
            float(lower_bound),
            bool(fresh),
        )
        return out

    def uniform_cu(self, sequences: int, tokens: int) -> torch.Tensor:
        """``arange(sequences + 1) * tokens`` on the device; kept per shape unless made during a graph capture."""
        key = (sequences, tokens)
        cu = self._cu.get(key)
        if cu is None:
            cu = torch.arange(sequences + 1, dtype=torch.int32, device=self.device)
            cu = cu * tokens
            if not torch.cuda.is_current_stream_capturing():
                self._cu[key] = cu
        return cu

    def default_slots(self, sequences: int) -> torch.Tensor:
        """``arange(sequences)`` on the device; kept per size unless made during a graph capture."""
        ids = self._ids.get(sequences)
        if ids is None:
            ids = torch.arange(sequences, dtype=torch.int32, device=self.device)
            if not torch.cuda.is_current_stream_capturing():
                self._ids[sequences] = ids
        return ids

    def open_chunk(
        self, slot: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The open rows of ``slot``: their count and copies of the key, ``u`` and prefix rows ``[HV, count, 128]``
        (one device read; for tests and debugging)."""
        n = int(self.cnt[slot, 0])
        return (
            n,
            self.key_rows[slot, :, :n].clone(),
            self.u_rows[slot, :, :n].clone(),
            self.prefix_rows[slot, :, :n].clone(),
        )

    def reset(self, slots: torch.Tensor) -> None:
        """Drop the open rows of ``slots`` (int32): their pool states are chunk boundaries again."""
        if slots.numel():
            self.cnt[slots.long()] = 0
