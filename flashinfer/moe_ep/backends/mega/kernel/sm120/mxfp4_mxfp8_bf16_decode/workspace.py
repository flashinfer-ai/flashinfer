# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workspace of the SM120 decode MegaMoE backend.

One symmetric buffer per rank holds the staged inputs, the expert pools, the
intermediate activations, the combine partials, the flags and the counters; the
BF16 output is a separate local tensor. With the ``"auto"`` / ``"nic"``
transports the buffer is an NCCL window and the workspace also owns the NCCL
device communicator the kernel's GIN puts use.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

# The kernel's counter block (sm120_mxfp4_mega_moe.cuh kCtrInts: per-call counters, per-expert and
# per-tile counts, GIN signal totals; error word at 3) and its slot in the layout below.
COUNTER_INTS = 1592
ERROR_INDEX = 3
_CTR_SLOT = 14
# GIN contexts (queue-pair sets): CTAs spread their puts over them; on RTX 6000D EP8 four beat one
# (1.8 vs 2.5 ms at 128 tokens per rank) and sixteen added signal-poll cost at small batches.
GIN_CONTEXTS = 4


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def pool_row_bytes(hidden: int) -> int:
    """An expert-pool row: the MXFP8 token, its UE8M0 scales and the source's (token * 8 + k), padded to 16 bytes."""
    return hidden + _align(hidden // 32 + 4, 16)


def buffer_layout(
    ranks: int,
    experts: int,
    capacity: int,
    topk: int,
    hidden: int,
    intermediate: int,
    gin: bool = False,
) -> list[int]:
    """Byte offsets of the symmetric buffer, in the order the binding reads them:
    [total, x, x_sf, topk_idx, topk_weights, pool, recv_cnt, recv_flag, comb, comb_flag,
    route, route_flag, inter, inter_s, ctr, stage_row, inbox, stage_comb, comb_gin, slotmap,
    stage_slot, cidx]. Every region sits at the same offset on every rank; pool / recv_* /
    route* are written by the other ranks, comb by the expert ranks, the rest only locally.
    The last seven regions (empty without NIC pairs) serve GIN, which sends a token once per
    rank: the token row staged for and received from GIN ranks, the partials staged for and
    received from GIN ranks, the slot tables received from and staged for GIN ranks (both
    packed in the expert rank's (local expert, slot) order), and the packed row of each
    (token, k)."""
    epr = experts // ranks
    rows = (
        epr * ranks * capacity
    )  # an expert pool slot per (local expert, source rank, token)
    row_bytes = pool_row_bytes(hidden)
    sizes = (
        capacity * hidden,
        capacity * (hidden // 32),
        capacity * topk * 8,
        capacity * topk * 4,
        rows * row_bytes,
        ranks * epr * 4,
        ranks * 4,
        capacity * topk * hidden * 2,
        ranks * 4,
        ranks * (1 + capacity * topk) * 4,
        ranks * 4,
        rows * intermediate,
        rows * (intermediate // 128) * 4,
        COUNTER_INTS * 4,
        capacity * row_bytes if gin else 0,
        ranks * capacity * row_bytes if gin else 0,
        ranks * capacity * topk * hidden * 2 if gin else 0,
        ranks * capacity * topk * hidden * 2 if gin else 0,
        ranks * capacity * topk * 4 if gin else 0,
        ranks * capacity * topk * 4 if gin else 0,
        capacity * topk * 4 if gin else 0,
    )
    offsets, cursor = [], 0
    for size in sizes:
        offsets.append(cursor)
        cursor = _align(cursor + size, 256)
    return [cursor, *offsets]


@functools.cache
def kernel_module(fast_math: bool, gin_type: int | None = None) -> Any:
    """The kernel's JIT module: P2P only, or with GIN pairs for one NCCL GIN backend."""
    from .......jit.megamoe import (
        gen_megamoe_sm120_mxfp4_gin_module,
        gen_megamoe_sm120_mxfp4_module,
    )

    if gin_type is None:
        return gen_megamoe_sm120_mxfp4_module(fast_math).build_and_load()
    return gen_megamoe_sm120_mxfp4_gin_module(fast_math, gin_type).build_and_load()


@functools.cache
def _gin_session_module() -> Any:
    from .......jit.megamoe import gen_megamoe_sm120_gin_session_module

    return gen_megamoe_sm120_gin_session_module().build_and_load()


class _DeviceBytes:
    """CUDA array interface over memory the workspace owns (the NCCL window), for torch.as_tensor."""

    def __init__(self, pointer: int, num_bytes: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (num_bytes,),
            "typestr": "|i1",
            "data": (pointer, False),
            "strides": None,
            "version": 3,
        }


class DecodeWorkspace:
    """Collective allocation (and rendezvous) over the EP group; must precede CUDA graph capture."""

    def __init__(
        self,
        *,
        group: Any,
        rank: int,
        world_size: int,
        num_experts: int,
        capacity: int,
        top_k: int,
        hidden: int,
        intermediate: int,
        transport: str,
        fast_math: bool,
    ) -> None:
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self.num_experts = num_experts
        self.capacity = capacity
        self.top_k = top_k
        self.hidden = hidden
        self.intermediate = intermediate
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.staged_rows = 0
        self.compile_tokens_per_rank: int | None = None
        self.gin: list[
            int
        ] = []  # [device ncclDevComm*, ncclWindow_t, GIN rank mask, contexts]
        self._gin_type: int | None = None
        self._gin_session: tuple[Any, int] | None = None
        self._symm_handle: Any = None
        self.offsets = buffer_layout(
            world_size,
            num_experts,
            capacity,
            top_k,
            hidden,
            intermediate,
            gin=transport != "p2p",
        )
        num_bytes = self.offsets[0]
        if world_size == 1:
            self.buffer = torch.zeros(num_bytes, dtype=torch.int8, device=self.device)
            self.peer_ptrs = [self.buffer.data_ptr()]
        elif transport == "p2p":
            import torch.distributed._symmetric_memory as symm_mem

            self.buffer = symm_mem.empty(
                num_bytes, dtype=torch.int8, device=self.device
            )
            self._symm_handle = symm_mem.rendezvous(self.buffer, group=group)
            self.buffer.zero_()
            self.peer_ptrs = [int(p) for p in self._symm_handle.buffer_ptrs]
        else:
            self._init_nccl_window(num_bytes, transport)
        self.module = kernel_module(fast_math, self._gin_type if self.gin else None)
        if world_size > 1:
            torch.distributed.barrier(group=group)
        torch.cuda.synchronize(self.device)
        self.x = self._view(1, (capacity, hidden), torch.uint8)
        self.x_sf = self._view(2, (capacity, hidden // 32), torch.uint8)
        self.topk_idx = self._view(3, (capacity, top_k), torch.int64)
        self.topk_weights = self._view(4, (capacity, top_k), torch.float32)
        self.output = torch.zeros(
            capacity, hidden, dtype=torch.bfloat16, device=self.device
        )
        self.destroyed = False

    def _init_nccl_window(self, num_bytes: int, transport: str) -> None:
        """The buffer as one NCCL symmetric window on the group's communicator: load/store peer
        pointers where NCCL maps a pair, NCCL GIN for the rest ("auto") or for every other rank
        ("nic")."""
        group, rank, ranks = self.group, self.rank, self.world_size
        torch.distributed.barrier(
            group=group
        )  # ProcessGroupNCCL creates its communicator lazily
        backend = group._get_backend(torch.device("cuda"))
        if not hasattr(backend, "_comm_ptr"):
            raise RuntimeError(
                "the NIC transport needs a ProcessGroupNCCL exposing _comm_ptr"
            )
        session = _gin_session_module()
        handle = int(
            session.megamoe_sm120_gin_session_create(
                int(backend._comm_ptr()), rank, ranks, num_bytes, GIN_CONTEXTS
            )
        )
        self._gin_session = (session, handle)
        base, win, dev_comm, gin_type, _, contexts = (
            int(v) for v in session.megamoe_sm120_gin_session_info(handle)
        )
        peers = [int(p) for p in session.megamoe_sm120_gin_session_peers(handle)]
        mask = 0
        for w in range(ranks):
            if w != rank and (transport == "nic" or peers[w] == 0):
                mask |= 1 << w
        # the kernel sends and receives over the same transport per pair: the masks must agree
        masks: list[Any] = [None] * ranks
        torch.distributed.all_gather_object(masks, mask, group=group)
        for w in range(ranks):
            if (mask >> w & 1) != (masks[w] >> rank & 1):
                raise RuntimeError(
                    f"ranks {rank} and {w} disagree on the NIC transport between them "
                    "(asymmetric NCCL load/store mapping)"
                )
        if mask and dev_comm == 0:
            raise RuntimeError(
                f"ranks {[w for w in range(ranks) if mask >> w & 1]} need the NIC transport, "
                "but the NCCL communicator offers no GIN"
            )
        self.peer_ptrs = [0 if mask >> w & 1 else peers[w] for w in range(ranks)]
        self.gin = [dev_comm, win, mask, contexts] if mask else []
        self._gin_type = gin_type
        self.buffer = torch.as_tensor(_DeviceBytes(base, num_bytes), device="cuda")

    def _view(
        self, slot: int, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        itemsize = torch.empty((), dtype=dtype).element_size()
        numel = 1
        for s in shape:
            numel *= s
        offset = self.offsets[slot]
        return self.buffer[offset : offset + numel * itemsize].view(dtype).view(*shape)

    def launch(
        self,
        y: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        num_tokens: int,
        swiglu_limit: float,
        tokens_per_item: int | None,
    ) -> None:
        self.module.megamoe_sm120_mxfp4_fused(
            y,
            self.buffer,
            self.peer_ptrs,
            self.rank,
            w13,
            w2,
            self.offsets,
            [
                self.num_experts,
                self.top_k,
                self.hidden,
                self.intermediate,
                self.capacity,
            ],
            num_tokens,
            float(swiglu_limit),
            None,
            None,
            None,
            self.gin,
            tokens_per_item or 0,
        )

    def error(self) -> int:
        """The error word of the last call, 0 when every spin wait completed. A wait that times out
        (a rank that never entered the call) records its code here and the kernel exits instead of
        hanging; each call clears it on entry. Synchronizes with the device."""
        ctr = self._view(_CTR_SLOT, (COUNTER_INTS,), torch.int32)
        return int(ctr[ERROR_INDEX].item())

    def destroy(self) -> None:
        """Release the buffer (collective with the NIC transport, which deregisters the NCCL window)."""
        if self.destroyed:
            return
        self.destroyed = True
        torch.cuda.synchronize(self.device)
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None
        self.buffer = None
        self._symm_handle = None
        session, self._gin_session = self._gin_session, None
        if session is not None:
            session[0].megamoe_sm120_gin_session_close(session[1])


__all__ = ["DecodeWorkspace", "buffer_layout", "kernel_module", "pool_row_bytes"]
