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
The exchange goes through this op's own LL mailbox, or through a caller's
:class:`PushPlane`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..cute_dsl_primitives import (
    NEGATIVE_ZERO_BF16_PAIR,
    VEC_BF16,
    fragment_has_negative_zero,
    fragment_has_sentinel,
    load_global_u32x4,
    load_volatile_u32,
    map_shared_to_peer,
    red_add_relaxed_gpu_u64,
    sanitize_negative_zero_u32x4,
    store_global_u32,
    store_lamport_sentinel_u32x4,
    store_shared_cluster_f32,
)
from ..runtime import to_cute, to_cute_dynamic
from .device_kernels import (
    ACTIVE_STAGE,
    LAMPORT_GENERATIONS,
    NEXT_STAGE,
    push_plane_phase,
)
from ..symmetric_buffer import SymmetricBuffer
from .protocol import (
    LLProtocolState,
    compile_shared_only_publish,
    create_ll_state,
)


@dataclass(frozen=True, slots=True)
class PushPlane:
    """A caller-owned two-phase Lamport buffer to gather through.

    ``buffer`` holds two halves of ``half_bytes``; the half of the current
    phase holds a dense ``[tp, m, H/tp]`` BF16 block for each call. Zero is
    the "not arrived" sentinel per ``atom_bytes``: with 4 (SGLang's
    ``LamportTrait`` default and FlashInfer's peer-store GEMM epilogue) a zero
    32-bit word, stored by producers as ``0x00008000``; with 2 a zero BF16,
    stored as -0. Consumers re-arm what they read with zeros.
    ``epoch`` is the rank-local 64-bit phase word: the phase is bit 0 of its
    high half, read after the PDL wait. Each consumer cluster (one per token)
    adds to it once after its re-arms, so the phase flips with the last one.
    Start it at zero on every rank and never reset it while the plane is in use.
    Every plane user runs on one stream, in the same order on every rank.
    """

    buffer: torch.Tensor  # this rank's mapping of the plane, >= 2 * half_bytes
    multicast_address: int  # multicast VA of the plane's base
    epoch: torch.Tensor  # [1] int64 (8 bytes), rank-local
    half_bytes: int
    atom_bytes: int = 2  # sentinel granularity: 2 = BF16, 4 = 32-bit word

    def ll_state(self) -> LLProtocolState:
        """The plane as the LL kernels' mailbox and stage state (not owned)."""
        plane = self.buffer.view(torch.uint8).flatten()[: 2 * self.half_bytes]
        return LLProtocolState(
            contribution_mailbox=SymmetricBuffer(
                plane.view(torch.bfloat16),
                _handle=None,
                multicast_address=self.multicast_address,
            ),
            stage_state=self.epoch.view(torch.int32),
        )


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
    lamport_2slot: cutlass.Constexpr[bool] = False,
    half_elements: Int64 = 0,
    lamport_atom_bytes: cutlass.Constexpr[int] = 2,
) -> None:
    """Wait for this token's 8 global columns ``fragment`` from their owner.

    Stores the BF16x8 fragment into ``packed`` and re-arms the mailbox slot;
    one thread per launch advances the stage. Releases PDL dependents.
    With ``lamport_2slot``, ``mailbox`` and ``stage_state`` are a
    :class:`PushPlane`'s buffer and epoch, ``half_elements`` its slot size,
    and the grid's x extent is ``m``; the caller commits the epoch with
    :func:`commit_push_plane_epoch`.
    """
    local_fragments = hidden_local // VEC_BF16
    # Each global column has exactly one owner; read it from that slot only.
    source_rank = fragment // local_fragments
    local_fragment = fragment % local_fragments
    if cutlass.const_expr(lamport_2slot):
        phase = push_plane_phase(stage_state)
        m, _, _ = cute.arch.grid_dim()
        # half[phase]: [tp, m, hidden_local]
        source_element = (
            Int64(phase) * Int64(half_elements)
            + (Int64(source_rank) * m + Int64(token)) * hidden_local
            + Int64(local_fragment) * VEC_BF16
        )
    else:
        active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
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
        if cutlass.const_expr(lamport_2slot):
            dirty = fragment_has_sentinel(value, 0, lamport_atom_bytes)
        elif cutlass.const_expr(lamport_atom_bytes == 4):
            dirty = fragment_has_sentinel(
                value, NEGATIVE_ZERO_BF16_PAIR, lamport_atom_bytes
            )
        else:
            dirty = fragment_has_negative_zero(value)
        packed.store(value)

    sentinel_pair = NEGATIVE_ZERO_BF16_PAIR
    if cutlass.const_expr(lamport_2slot):
        # The producer stored a zero atom with -0.
        packed.store(sanitize_negative_zero_u32x4(packed))
        sentinel_pair = 0
    else:
        if token == 0 and fragment == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
    if cutlass.const_expr(enable_pdl):
        cute.arch.griddepcontrol_launch_dependents()
    store_lamport_sentinel_u32x4(Int64(source_pointer.toint()), sentinel_pair)


@cute.jit
def commit_push_plane_epoch(
    epoch: cute.Tensor, participant: Int32, participants: Int32
) -> None:
    """Count one participant into the push plane's next phase.

    Call once per participant, from one thread, after all of its re-arms.
    Participant 0 adds ``2**32 - (participants - 1)`` and the others 1, so
    the phase (high half) flips with the last one.
    """
    arrival = Int64(1)
    if participant == 0:
        arrival = Int64(1 << 32) - Int64(participants - 1)
    red_add_relaxed_gpu_u64(Int64(epoch.iterator.toint()), arrival)


class LLColumnGather:
    """Mailbox, stage state and publisher of one LL column all-gather.

    Construction is collective over ``group``. Subclasses compile their
    consumer in :meth:`_compile`; calls on all ranks must use the same ``m``
    and must not overlap one another. With ``push_plane`` the op allocates no
    mailbox and exchanges through the caller's plane instead.
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
        push_plane: PushPlane | None = None,
    ) -> None:
        if hidden_size % (tp_size * VEC_BF16):
            raise ValueError("hidden_size must split into 8-element rank shards")
        self.hidden_size = hidden_size
        self.hidden_local = hidden_size // tp_size
        self.tp_size = tp_size
        self.capacity_m = capacity_m
        self.push_plane = push_plane
        if push_plane is not None and (
            dist.get_world_size(group) != tp_size or dist.get_rank(group) != rank
        ):
            # create_ll_state checks these in the mailbox mode.
            raise ValueError("ProcessGroup size or rank does not match tp_size or rank")
        self.state = (
            create_ll_state(group, tp_size, rank, capacity_m, self.hidden_local)
            if push_plane is None
            else push_plane.ll_state()
        )
        self._mailbox = to_cute(self.state.contribution_mailbox.tensor.flatten(), 16)
        self._mailbox_multicast = Int64(
            self.state.contribution_mailbox.multicast_address
        )
        self._stage_state = to_cute(self.state.stage_state, 4)
        self._lamport_2slot = push_plane is not None
        self._lamport_atom_bytes = 2 if push_plane is None else push_plane.atom_bytes
        self._half_elements = Int64(
            0 if push_plane is None else push_plane.half_bytes // 2
        )
        self._publisher = compile_shared_only_publish(
            hidden=self.hidden_local,
            tp=tp_size,
            rank=rank,
            capacity_m=capacity_m,
            elements_per_thread=VEC_BF16,
            threads=publish_threads,
            release_before_store=publish_release_before_store,
            enable_pdl=enable_pdl,
            lamport_2slot=self._lamport_2slot,
            lamport_atom_bytes=self._lamport_atom_bytes,
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
            self._half_elements,
            stream,
        )
