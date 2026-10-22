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

"""Shared pieces of the LL column all-gathers fused into a row consumer.

Each rank multicasts its ``[M, H/TP]`` column shard with the shared-only LL
publisher; the consumer kernel reads every global column from its owner rank.
"""

from __future__ import annotations

from typing import Any

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..cute_dsl_primitives import (
    VEC_BF16,
    fragment_has_negative_zero,
    load_global_u32x4,
    load_volatile_u32,
    map_shared_to_peer,
    store_global_u32,
    store_lamport_sentinel_u32x4,
    store_shared_cluster_f32,
)
from ..runtime import to_cute, to_cute_dynamic
from .device_kernels import ACTIVE_STAGE, LAMPORT_GENERATIONS, NEXT_STAGE
from .protocol import compile_shared_only_publish, create_ll_state


@cute.jit
def cluster_row_sum(
    value: Float32,
    slots: cute.Pointer,
    warps: cutlass.Constexpr[int],
    cluster_size: cutlass.Constexpr[int],
    cluster_rank: Int32,
    warp: Int32,
    lane: Int32,
) -> Float32:
    """Sum one value per thread over the cluster; every thread gets the total.

    Each warp scatters its sum straight into ``slots[cluster_rank, warp]`` of
    every peer CTA, so one cluster barrier replaces the CTA-level stage.

    The caller must complete one cluster barrier before the first call: a
    remote shared-memory store to a peer CTA that has not started executing
    faults with "cluster target block not present". Kernels arrive (relaxed)
    at entry and wait just before the first call so the barrier overlaps the
    parameter loads and the mailbox poll.
    """
    for offset in (16, 8, 4, 2, 1):
        value = value + cute.arch.shuffle_sync_bfly(value, offset=offset)
    if lane < cluster_size:
        remote_slot = map_shared_to_peer(slots + (cluster_rank * warps + warp), lane)
        store_shared_cluster_f32(remote_slot, value)
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    total = Float32(0.0)
    for slot in cutlass.range_constexpr(cluster_size * warps):
        total = total + cute.arch.load((slots + slot).llvm_ptr, Float32)
    return total


@cute.jit
def poll_owned_fragment(
    mailbox: cute.Tensor,
    stage_state: cute.Tensor,
    packed: cute.Tensor,
    token: Int32,
    fragment: Int32,
    tp: cutlass.Constexpr[int],
    capacity_m: cutlass.Constexpr[int],
    hidden_local: cutlass.Constexpr[int],
    enable_pdl: cutlass.Constexpr[bool],
) -> None:
    """Wait for this token's 8 global columns ``fragment`` from their owner.

    Stores the BF16x8 fragment into ``packed`` and re-arms the mailbox slot;
    one thread per launch advances the stage. Releases PDL dependents.
    """
    local_fragments = hidden_local // VEC_BF16
    active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
    # Each global column has exactly one owner; read it from that slot only.
    source_rank = fragment // local_fragments
    local_fragment = fragment % local_fragments
    source_element = (
        (Int64(active_stage) * tp + Int64(source_rank)) * capacity_m + Int64(token)
    ) * hidden_local + Int64(local_fragment) * VEC_BF16
    source_pointer = cute.make_ptr(
        BFloat16,
        (mailbox.iterator + source_element).llvm_ptr,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    # Always true; a runtime value keeps the poll a device loop.
    dirty = fragment >= Int32(0)
    while dirty:
        value = load_global_u32x4(source_pointer, volatile=True)
        dirty = fragment_has_negative_zero(value)
        packed.store(value)

    if token == 0 and fragment == 0:
        store_global_u32(
            stage_state.iterator + NEXT_STAGE,
            (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
        )
    if cutlass.const_expr(enable_pdl):
        cute.arch.griddepcontrol_launch_dependents()
    store_lamport_sentinel_u32x4(Int64(source_pointer.toint()))


class LLColumnGather:
    """Mailbox, stage state and publisher of one LL column all-gather.

    Construction is collective over ``group``. Subclasses compile their
    consumer in :meth:`_compile`; calls on all ranks must use the same ``m``
    and must not overlap one another.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        tp_size: int,
        rank: int,
        capacity_m: int,
        group: dist.ProcessGroup,
        publish_threads: int,
        publish_release_before_store: bool,
        enable_pdl: bool,
    ) -> None:
        if hidden_size % (tp_size * VEC_BF16):
            raise ValueError("hidden_size must split into 8-element rank shards")
        self.hidden_size = hidden_size
        self.hidden_local = hidden_size // tp_size
        self.tp_size = tp_size
        self.capacity_m = capacity_m
        self.state = create_ll_state(
            group, tp_size, rank, capacity_m, self.hidden_local
        )
        self._mailbox = to_cute(self.state.contribution_mailbox.tensor.flatten(), 16)
        self._mailbox_multicast = Int64(
            self.state.contribution_mailbox.multicast_address
        )
        self._stage_state = to_cute(self.state.stage_state, 4)
        self._publisher = compile_shared_only_publish(
            hidden=self.hidden_local,
            tp=tp_size,
            rank=rank,
            capacity_m=capacity_m,
            elements_per_thread=VEC_BF16,
            threads=publish_threads,
            release_before_store=publish_release_before_store,
            enable_pdl=enable_pdl,
        )
        self._collectives: dict[Any, Any] = {}
        torch.cuda.synchronize()
        dist.barrier(group)

    def _compile(self, constants: Any) -> Any:
        raise NotImplementedError

    def compiled(self, constants: Any) -> Any:
        """The consumer for one constant set; compile it before graph capture."""
        collective = self._collectives.get(constants)
        if collective is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"Compile {type(self).__name__} before CUDA graph capture"
                )
            collective = self._collectives[constants] = self._compile(constants)
        return collective

    def _publish(self, shard: torch.Tensor, m: Int32, stream: Any) -> None:
        self._publisher(
            to_cute_dynamic(shard.flatten(), 16, divisibility=self.hidden_local),
            self._stage_state,
            self._mailbox_multicast,
            None,
            m,
            stream,
        )
