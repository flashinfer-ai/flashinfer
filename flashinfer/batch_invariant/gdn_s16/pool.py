# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The per-pool state of the GDN class ``gdn.bf16.s16`` and its launches.

A bf16 state pool ``[S, HV, 128, 128]`` passed as ``initial_state`` holds, per slot, the state at the sequence's last
16-token boundary. The class keeps next to it, keyed by the pool's storage and offset, the open chunk of every slot:
the normalized keys (bf16), the updates ``u`` (fp32), the log-decay prefix (fp32) and the open-row count. Every kernel
reads the count on the device and writes the slot in place, so no call reads anything on the host:

* :meth:`GdnS16Pool.decode`: one token per slot, one CTA per (slot, v-head);
* :meth:`GdnS16Pool.prefill`: a run of tokens per slot, one CTA per (slot, v-head, V-tile), looping over chunks; the
  short-call kernel when no sequence can bring ``CHAIN_MIN_TOKENS`` tokens, the long-run kernel otherwise.
"""

from __future__ import annotations

import functools
import weakref
from typing import Dict, Tuple

import torch

from ..support import unsupported
from .spec import CHUNK, CLASS, HEAD_DIM

#: Calls whose longest sequence may bring this many tokens run the long-run prefill kernel (a placement).
CHAIN_MIN_TOKENS = 2 * CHUNK
_POOLS: Dict[Tuple[int, int, int, torch.device], "GdnS16Pool"] = {}


@functools.cache
def _module():
    from ...jit.gdn_batch_invariant_class_s import (
        gen_gdn_batch_invariant_class_s_module,
    )

    return gen_gdn_batch_invariant_class_s_module().build_and_load()


def pool_key(pool: torch.Tensor) -> Tuple[int, int, int, torch.device]:
    """The pool's identity: its storage, where it starts in it and its head count (a view from the same offset is the
    same pool; a view from another offset is another pool with its own open chunks)."""
    return (
        pool.untyped_storage().data_ptr(),
        pool.storage_offset(),
        int(pool.shape[1]),
        pool.device,
    )


def get_pool(pool: torch.Tensor, num_q_heads: int) -> "GdnS16Pool":
    """The class's state next to ``pool`` (created on first use for every slot the pool's storage can hold from the
    pool's offset, released when that storage is freed). Not created inside a CUDA graph capture: a replay would
    zero it again."""
    key = pool_key(pool)
    gp = _POOLS.get(key)
    if gp is None:
        if torch.cuda.is_current_stream_capturing():
            raise unsupported(
                CLASS,
                "a CUDA graph capture as the first call on a state pool",
                "one eager call on the pool before the capture (it creates the pool's side state)",
            )
        storage = pool.untyped_storage()
        slot_bytes = 2 * int(pool.shape[1]) * HEAD_DIM * HEAD_DIM
        capacity = (storage.nbytes() - 2 * pool.storage_offset()) // slot_bytes
        gp = GdnS16Pool(max(capacity, int(pool.shape[0])), pool, num_q_heads)
        _POOLS[key] = gp
        weakref.finalize(storage, _POOLS.pop, key, None)
    return gp


class GdnS16Pool:
    def __init__(self, slots: int, pool: torch.Tensor, num_q_heads: int) -> None:
        dev = pool.device
        self.slots, self.HQ, self.HV = slots, num_q_heads, int(pool.shape[1])

        def zeros(*shape, dtype):
            return torch.zeros(*shape, dtype=dtype, device=dev)

        # two banks: a call that crosses a chunk boundary writes the other one
        self.hist_k = zeros(slots, 2, self.HQ, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
        self.hist_u = zeros(slots, 2, self.HV, CHUNK, HEAD_DIM, dtype=torch.float32)
        self.hist_l = zeros(slots, 2, self.HV, CHUNK, dtype=torch.float32)
        self.cnt = zeros(slots, self.HV, 8, dtype=torch.int32)
        self.no_prefix = torch.empty(0, dtype=torch.int32, device=dev)
        self.sm_count = torch.cuda.get_device_properties(dev).multi_processor_count
        self._arange: Dict[int, torch.Tensor] = {}
        self.module = _module()

    def arange(self, n: int) -> torch.Tensor:
        cu = self._arange.get(n)
        if cu is None:
            cu = torch.arange(n, dtype=torch.int32, device=self.cnt.device)
            self._arange[n] = cu
        return cu

    def num_tiles(self, num_seqs: int) -> int:
        """V-tiles per (sequence, v-head) of a prefill launch: a placement, the bits do not depend on it."""
        ctas = num_seqs * self.HV
        return 4 if 4 * ctas <= self.sm_count else 2 if 2 * ctas <= self.sm_count else 1

    def _side(self):
        return self.hist_k, self.hist_u, self.hist_l, self.cnt

    def decode(self, pool, slots, rows, A_log, dt_bias, scale, out) -> None:
        # a placement: with at most two blocks per SM the slot-independent work is issued first
        early = int(slots.numel()) * self.HV <= 2 * self.sm_count
        self.module.gdn_batch_invariant_class_s_decode(
            *rows, A_log, dt_bias, slots, out, pool, *self._side(), scale, int(early)
        )

    def prefill(
        self, pool, slots, cu, prefix, rows, A_log, dt_bias, scale, out, max_tokens
    ) -> None:
        """``prefix`` None: every sequence starts from its slot's state as a chunk boundary."""
        if max_tokens >= CHAIN_MIN_TOKENS:
            self.module.gdn_batch_invariant_class_s_prefill_chain(
                *rows,
                A_log,
                dt_bias,
                slots,
                cu,
                self.no_prefix if prefix is None else prefix,
                out,
                pool,
                *self._side(),
                scale,
                min(self.num_tiles(int(slots.numel())), 4),
                int(prefix is None),
            )
            return
        self.module.gdn_batch_invariant_class_s_prefill(
            *rows,
            A_log,
            dt_bias,
            slots,
            cu,
            self.no_prefix if prefix is None else prefix,
            out,
            pool,
            *self._side(),
            scale,
            self.num_tiles(int(slots.numel())),
            max_tokens,
            int(prefix is None),
        )
