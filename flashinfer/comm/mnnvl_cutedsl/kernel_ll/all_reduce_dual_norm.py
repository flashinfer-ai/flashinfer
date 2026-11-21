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

"""LL all-reduce consumed by a gated residual and the next RMSNorm.

Two-shot: every rank sends each column slice of its ``[M, H]`` BF16 partial
to the slice's owner, which sums it in FP32 in rank order and multicasts the
BF16 sum. One-shot: every rank multicasts its whole partial into slot
``[rank]`` of every rank, and each rank sums the ``tp`` copies in rank order;
a producer GEMM may already have stored them (``pushed``). Each token's CTAs
then run the boundary on the reduced row (see
:func:`all_reduce_gated_residual_next_rmsnorm`). The exchange is this op's
three-generation LL mailbox or a caller's :class:`PushPlane`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass.cute.runtime import make_fake_compact_tensor

from ..cute_dsl_primitives import (
    VEC_BF16,
    WARP_SIZE,
    bf16x8_to_packed_u32x4,
    fragment_has_negative_zero,
    fragment_has_sentinel,
    load_bf16x8_as_f32,
    load_f32x8,
    load_global_u32x4_address,
    load_volatile_u32,
    mark_zero_u32x4,
    packed_u32x4_to_bf16x8,
    sanitize_negative_zero_u32x4,
    stmc_bf16x8,
    store_f32x8,
    store_global_u32,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
)
from ....batch_invariant.fp_interface.cute import CuteF32 as F
from ....cute_dsl.fp4_common import fmax_f32
from ..runtime import (
    current_cu_stream,
    make_fake_dynamic_compact_tensor,
    to_cute,
    to_cute_dynamic,
)
from ..symmetric_buffer import SymmetricBuffer
from .column_gather import PushPlane, commit_push_plane_epoch
from .device_kernels import LAMPORT_GENERATIONS, push_plane_phase

MAILBOX, PLANE = "mailbox", "plane"
# Sizes the mailbox's one-shot slots: one-shot pays off while M * tp is at most this.
ONE_SHOT_MAX_TOKENS_X_TP = 128


@dataclass(frozen=True, slots=True)
class LLAllReduceDualNormTuning:
    """CTAs per token (a cluster) and threads per CTA; ``8 * threads *
    cluster_size`` must divide the hidden size."""

    cluster_size: int
    threads: int


def default_all_reduce_dual_norm_tuning(
    hidden: int, num_tokens: int
) -> LLAllReduceDualNormTuning:
    """More CTAs per token for fewer tokens; one CTA per token from 65 tokens."""
    cluster_size = next(
        (c for m, c in ((8, 16), (16, 8), (32, 4), (64, 2)) if num_tokens <= m), 1
    )
    while cluster_size > 1:
        threads = min(256, hidden // (cluster_size * VEC_BF16)) // WARP_SIZE * WARP_SIZE
        if threads >= WARP_SIZE and hidden % (cluster_size * threads * VEC_BF16) == 0:
            return LLAllReduceDualNormTuning(cluster_size, threads)
        cluster_size //= 2
    choices = (256, 512, 1024, 128) if num_tokens < 192 else (512, 256, 128)
    for threads in choices:
        if hidden % (threads * VEC_BF16) == 0:
            return LLAllReduceDualNormTuning(1, threads)
    raise ValueError("hidden_size must be a multiple of 1024")


@dataclass(frozen=True, slots=True)
class _DualNormConstants:
    post_eps: float
    next_eps: float
    has_post_weight: bool
    post_weight_is_input_scale: bool
    has_next_weight: bool
    gate: str  # "none", "scalar" ([1] alpha / beta) or "channel" ([H])
    has_router: bool
    pushed: bool  # requires one_shot
    one_shot: bool
    tuning: LLAllReduceDualNormTuning


class _Slots(NamedTuple):
    """One call's exchange state (device values)."""

    local: Int64  # this rank's scatter mailbox or plane
    gather: Int64  # this rank's gather mailbox or plane
    multicast: Int64  # multicast address of the gather mailbox or plane
    sync: cute.Tensor  # per-token generations, or the plane epoch
    phase: Int64  # byte offset of the plane's phase half (mailbox: 0)
    slot_bytes: Int64  # plane bytes per source rank
    m: Int32


class _AllReduceDualNormDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        rank: int,
        capacity_m: int,
        one_shot_rows: int,
        constants: _DualNormConstants,
        exchange: str,
        atom_bytes: int,
        enable_pdl: bool,
    ) -> None:
        cluster_size, threads = constants.tuning.cluster_size, constants.tuning.threads
        if threads % WARP_SIZE or threads < cluster_size:
            raise ValueError("threads must be a warp multiple of at least cluster_size")
        if cluster_size not in (1, 2, 4, 8, 16):
            raise ValueError("cluster_size must be 1, 2, 4, 8 or 16")
        if hidden % (VEC_BF16 * threads * cluster_size) or hidden % (VEC_BF16 * tp):
            raise ValueError(
                "hidden must split into 8-element fragments per thread and rank"
            )
        self.hidden, self.tp, self.rank, self.capacity_m = hidden, tp, rank, capacity_m
        self.one_shot_rows = one_shot_rows
        # Mailbox: the one-shot slots follow the [3, capacity_m, H] gather rows.
        self.broadcast_offset = LAMPORT_GENERATIONS * capacity_m * hidden * 2
        self.constants, self.exchange, self.atom_bytes = constants, exchange, atom_bytes
        self.enable_pdl = enable_pdl
        self.cluster_size, self.num_threads = cluster_size, threads
        self.warps = threads // WARP_SIZE
        self.fragments = hidden // VEC_BF16
        self.shard_fragments = self.fragments // tp
        self.num_row_threads = threads * cluster_size
        self.packs = self.fragments // self.num_row_threads
        self.shard_trips = -(-self.shard_fragments // self.num_row_threads)

    def smem_size_in_bytes(self) -> int:
        if self.cluster_size == 1:
            return 2 * self.warps * 4  # one (warps) area per statistic
        # (C,4) peer sums, an mbarrier, (4,warps) warp sums
        return self.cluster_size * 16 + 8 + 4 * self.warps * 4

    @cute.jit
    def __call__(
        self,
        partial: cute.Tensor,
        residual: cute.Tensor,
        alpha: cute.Tensor | None,
        beta: cute.Tensor | None,
        post_weight: cute.Tensor | None,
        next_weight: cute.Tensor | None,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        router_output: cute.Tensor | None,
        buffer: cute.Tensor,
        peer_buffers: cute.Tensor | None,
        gather_buffer: cute.Tensor,
        multicast_address: Int64,
        sync: cute.Tensor,
        slot_bytes: Int64,
        m: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ) -> None:
        # A cluster per token, or ``ctas`` persistent CTAs over the tokens.
        kernel = self.kernel(
            partial,
            residual,
            alpha,
            beta,
            post_weight,
            next_weight,
            residual_output,
            norm_output,
            router_output,
            buffer,
            peer_buffers,
            gather_buffer,
            multicast_address,
            sync,
            slot_bytes,
            m,
        )
        if cutlass.const_expr(self.cluster_size == 1):
            kernel.launch(
                grid=(ctas, 1, 1),
                block=(self.num_threads, 1, 1),
                smem=self.smem_size_in_bytes(),
                stream=stream,
                use_pdl=self.enable_pdl,
            )
        else:
            kernel.launch(
                grid=(m, self.cluster_size, 1),
                block=(self.num_threads, 1, 1),
                cluster=(1, self.cluster_size, 1),
                smem=self.smem_size_in_bytes(),
                stream=stream,
                use_pdl=self.enable_pdl,
            )

    # -------------------------------------------------------------------------
    # Exchange
    # -------------------------------------------------------------------------
    @cute.jit
    def _sanitize_sentinel(self, packed):
        """Producer side: no Lamport atom of ``packed`` reads as "not arrived"."""
        if cutlass.const_expr(self.exchange == MAILBOX):
            return sanitize_negative_zero_u32x4(packed)
        return mark_zero_u32x4(packed, self.atom_bytes)

    @cute.jit
    def _has_sentinel(self, packed):
        """Consumer side: some atom of ``packed`` has not arrived yet."""
        if cutlass.const_expr(self.exchange == MAILBOX):
            return fragment_has_negative_zero(packed)
        return fragment_has_sentinel(packed, 0, self.atom_bytes)

    @cute.jit
    def _store_sentinel(self, address: Int64) -> None:
        """Re-arm a read slot for the next exchange."""
        if cutlass.const_expr(self.exchange == MAILBOX):
            store_lamport_sentinel_u32x4(address)
        else:
            store_lamport_sentinel_u32x4(address, 0)

    @cute.jit
    def _poll(self, values: cute.Tensor, addresses: cute.Tensor) -> None:
        """Spin until every fragment at ``addresses`` has arrived; all loads of
        a round are in flight before any is tested."""
        count = cute.size(addresses)
        dirty = cutlass.Boolean(True)
        while dirty:
            dirty = False
            for i in cutlass.range_constexpr(count):
                pointer = cute.make_ptr(
                    Uint32, addresses[i], cute.AddressSpace.gmem, assumed_align=16
                )
                packed = cute.TensorSSA(
                    cute.arch.load(
                        pointer,
                        ir.VectorType.get([4], Uint32.mlir_type),
                        sem="relaxed",
                        scope="gpu",
                    ),
                    4,
                    Uint32,
                )
                dirty = dirty | self._has_sentinel(packed)
                for word in cutlass.range_constexpr(4):
                    values[i, word] = packed[word]

    @cute.jit
    def _sum_bf16(self, values: cute.Tensor, first: int):
        """BF16 of the FP32 rank-order sum of rows ``first .. first + tp``."""
        total = None
        for source in cutlass.range_constexpr(self.tp):
            term = packed_u32x4_to_bf16x8(values[first + source, None].load()).to(
                Float32
            )
            total = term if source == 0 else total + term
        return bf16x8_to_packed_u32x4(total.to(BFloat16))

    @cute.jit
    def _scatter_area(self, s: _Slots, token: Int32) -> Int64:
        """Offset of this call's scatter slots: the token's generation or the phase."""
        if cutlass.const_expr(self.exchange == MAILBOX):
            stage_bytes = self.tp * self.capacity_m * self.shard_fragments * 16
            return Int64(load_volatile_u32(s.sync.iterator + token)) * stage_bytes
        return s.phase

    @cute.jit
    def _scatter_slot(self, s: _Slots, area: Int64, source, token, local) -> Int64:
        """Mailbox ``[tp, capacity_m, H / tp]`` or plane ``[tp][m, H / tp]``."""
        if cutlass.const_expr(self.exchange == MAILBOX):
            row = Int64(source) * self.capacity_m + token
            return area + (row * self.shard_fragments + local) * 16
        row = Int64(token) * self.shard_fragments + local
        return area + Int64(source) * s.slot_bytes + row * 16

    @cute.jit
    def _gather_slot(self, s: _Slots, base: Int64, owner, token, local) -> Int64:
        """Mailbox ``[3, capacity_m, H]``, or plane ``[phase][owner]`` after the
        ``[m, H / tp]`` scatter rows."""
        if cutlass.const_expr(self.exchange == MAILBOX):
            stage = Int64(load_volatile_u32(s.sync.iterator + token))
            row = stage * self.capacity_m + token
            column = (Int64(owner) * self.shard_fragments + local) * VEC_BF16
            return base + (row * self.hidden + column) * 2
        row = (Int64(s.m) + token) * self.shard_fragments + local
        return base + s.phase + Int64(owner) * s.slot_bytes + row * 16

    @cute.jit
    def _broadcast_area(self, s: _Slots, token: Int32) -> Int64:
        """Offset of this call's one-shot slots: the token's generation or the phase."""
        if cutlass.const_expr(self.exchange == MAILBOX):
            source_bytes = self.one_shot_rows * self.hidden * 2
            stage = Int64(load_volatile_u32(s.sync.iterator + token))
            return self.broadcast_offset + stage * (self.tp * source_bytes)
        return s.phase

    @cute.jit
    def _broadcast_slot(self, s: _Slots, area: Int64, source, token, fragment):
        """Mailbox ``[3, tp, one_shot_rows, H]`` or plane ``[phase][source]``
        ``[m, H]`` (the peer-store GEMM's ``broadcast`` layout)."""
        row = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
        if cutlass.const_expr(self.exchange == MAILBOX):
            source_bytes = self.one_shot_rows * self.hidden * 2
            return area + source * source_bytes + row * 2
        return area + source * s.slot_bytes + row * 2

    @cute.jit
    def _row_address(self, tensor: cute.Tensor, token, fragment) -> Int64:
        element = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
        return Int64((tensor.iterator + element).toint())

    @cute.jit
    def _scatter(self, s: _Slots, row_tidx, partial, peer_buffers, token) -> None:
        """Scatter: this rank's partial fragments to their owners."""
        shard = self.shard_fragments
        area = self._scatter_area(s, token)
        for p in cutlass.range_constexpr(self.packs):
            fragment = row_tidx + p * self.num_row_threads
            owner = fragment // shard
            if owner != Int32(self.rank):
                packed = self._sanitize_sentinel(
                    load_global_u32x4_address(
                        self._row_address(partial, token, fragment)
                    )
                )
                peer = cute.arch.load((peer_buffers.iterator + owner).llvm_ptr, Int64)
                slot = self._scatter_slot(
                    s, peer + area, self.rank, token, fragment % shard
                )
                store_global_u32x4(slot, packed)

    @cute.jit
    def _broadcast(self, s: _Slots, row_tidx, partial, token) -> None:
        """One-shot: this rank's partial row into slot ``[rank]`` of every rank."""
        area = s.multicast + self._broadcast_area(s, token)
        for p in cutlass.range_constexpr(self.packs):
            fragment = row_tidx + p * self.num_row_threads
            packed = self._sanitize_sentinel(
                load_global_u32x4_address(self._row_address(partial, token, fragment))
            )
            slot = self._broadcast_slot(s, area, self.rank, token, fragment)
            stmc_bf16x8(slot, packed, word_atoms=self.atom_bytes == 4)

    @cute.jit
    def _reduce(self, s: _Slots, row_tidx, partial, token, branch: cute.Tensor) -> None:
        """Two-shot: sum this rank's slice and multicast it. One-shot: sum every
        rank's row into ``branch``."""
        if cutlass.const_expr(self.constants.one_shot):
            self._reduce_rows(s, row_tidx, token, branch)
        else:
            self._reduce_slice(s, row_tidx, partial, token)

    @cute.jit
    def _reduce_slice(self, s: _Slots, row_tidx, partial, token) -> None:
        """Sum this rank's slice of ``token`` in rank order; multicast the BF16 sum."""
        tp, rank = self.tp, self.rank
        area = s.local + self._scatter_area(s, token)
        for trip in cutlass.range_constexpr(self.shard_trips):
            local = row_tidx + trip * self.num_row_threads
            if local < self.shard_fragments:
                fragment = rank * self.shard_fragments + local
                # Peers from their slots; this rank's own fragment from the partial.
                peers = cute.make_rmem_tensor(cute.make_layout((tp - 1,)), Int64)
                for i in cutlass.range_constexpr(tp - 1):
                    source = i + (1 if i >= rank else 0)
                    peers[i] = self._scatter_slot(s, area, source, token, local)
                own = load_global_u32x4_address(
                    self._row_address(partial, token, fragment)
                )
                if cutlass.const_expr(self.exchange == MAILBOX):
                    own = sanitize_negative_zero_u32x4(own)
                peer_values = cute.make_rmem_tensor(
                    cute.make_layout((tp - 1, 4), stride=(4, 1)), Uint32
                )
                self._poll(peer_values, peers)
                values = cute.make_rmem_tensor(
                    cute.make_layout((tp, 4), stride=(4, 1)), Uint32
                )
                for source in cutlass.range_constexpr(tp):
                    if cutlass.const_expr(source == rank):
                        values[source, None].store(own)
                    else:
                        i = source - (1 if source > rank else 0)
                        values[source, None].store(peer_values[i, None].load())
                reduced = self._sanitize_sentinel(self._sum_bf16(values, 0))
                slot = self._gather_slot(s, s.multicast, rank, token, local)
                stmc_bf16x8(slot, reduced, word_atoms=self.atom_bytes == 4)
                for i in cutlass.range_constexpr(tp - 1):
                    self._store_sentinel(peers[i])

    @cute.jit
    def _receive(self, addresses: cute.Tensor, values: cute.Tensor) -> None:
        """Poll ``addresses`` into ``values``, reconverge the warp, re-arm the slots."""
        # This thread's own multicast stores (one-shot rows, two-shot slice sums) are
        # read back through the unicast alias.
        cute.arch.fence_proxy("alias")
        self._poll(values, addresses)
        # The warp must reconverge before the norm's shuffles: lanes leave the spin
        # at different rounds.
        cute.arch.sync_warp()
        for i in cutlass.range_constexpr(cute.size(addresses)):
            self._store_sentinel(addresses[i])

    @cute.jit
    def _reduce_rows(self, s: _Slots, row_tidx, token, branch: cute.Tensor) -> None:
        """One-shot: wait for every rank's row of ``token``, sum it in rank order into
        ``branch`` (P,4):(4,1)."""
        tp = self.tp
        base = s.gather if self.exchange == MAILBOX else s.local
        area = base + self._broadcast_area(s, token)
        addresses = cute.make_rmem_tensor(cute.make_layout((self.packs * tp,)), Int64)
        for p in cutlass.range_constexpr(self.packs):
            fragment = row_tidx + p * self.num_row_threads
            for source in cutlass.range_constexpr(tp):
                addresses[p * tp + source] = self._broadcast_slot(
                    s, area, source, token, fragment
                )
        rows = cute.make_rmem_tensor(
            cute.make_layout((self.packs * tp, 4), stride=(4, 1)), Uint32
        )
        self._receive(addresses, rows)
        for p in cutlass.range_constexpr(self.packs):
            branch[p, None].store(self._sum_bf16(rows, p * tp))

    @cute.jit
    def _gather(self, s: _Slots, row_tidx, token, branch: cute.Tensor) -> None:
        """Two-shot: wait for the owners' reduced slices of ``token`` into ``branch``
        (P,4):(4,1)."""
        addresses = cute.make_rmem_tensor(cute.make_layout((self.packs,)), Int64)
        for p in cutlass.range_constexpr(self.packs):
            fragment = row_tidx + p * self.num_row_threads
            owner = fragment // self.shard_fragments
            local = fragment % self.shard_fragments
            addresses[p] = self._gather_slot(s, s.gather, owner, token, local)
        self._receive(addresses, branch)

    @cute.jit
    def _advance(self, sync, participant, participants, token) -> None:
        """Mailbox: ``token``'s next generation. Plane: one epoch commit."""
        if cutlass.const_expr(self.exchange == MAILBOX):
            stage = Uint32(load_volatile_u32(sync.iterator + token))
            store_global_u32(
                sync.iterator + token, (stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS)
            )
        else:
            commit_push_plane_epoch(sync, participant, participants)

    # -------------------------------------------------------------------------
    # Boundary
    # -------------------------------------------------------------------------
    @cute.jit
    def _warp_sum(self, value: Float32, width: cutlass.Constexpr[int]) -> Float32:
        for offset in tuple(width >> k for k in range(1, width.bit_length())):
            value = value + cute.arch.shuffle_sync_bfly(value, offset=offset)
        return value

    @cute.jit
    def _cta_sum(self, value: Float32, slots: cute.Pointer) -> Float32:
        """Warp butterfly, then one warp over the per-warp sums."""
        value = self._warp_sum(value, WARP_SIZE)
        if cutlass.const_expr(self.warps > 1):
            lane = cute.arch.lane_idx()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            with cute.arch.elect_one():
                cute.arch.store((slots + warp).llvm_ptr, value)
            cute.arch.barrier()
            value = Float32(0.0)
            if lane < self.warps:
                value = cute.arch.load((slots + lane).llvm_ptr, Float32)
            value = self._warp_sum(value, WARP_SIZE)
        return value

    @cute.jit
    def _normed_bf16(self, values, gamma, rstd: Float32, p: int):
        """``bf16(values[p] * rstd [* gamma[p]])`` as FP32 (the post-norm output)."""
        normed = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
        for e in cutlass.range_constexpr(VEC_BF16):
            normed[e] = F.mul(values[p, e], rstd)
            if cutlass.const_expr(
                self.constants.has_post_weight
                and not self.constants.post_weight_is_input_scale
            ):
                normed[e] = F.mul(normed[e], gamma[p, e])
        return normed.load().to(BFloat16).to(Float32)

    @cute.jit
    def _load_residual(self, row_tidx, residual, token, stream_values) -> None:
        for p in cutlass.range_constexpr(self.packs):
            element = (
                Int64(token) * self.hidden
                + Int64(row_tidx + p * self.num_row_threads) * VEC_BF16
            )
            stream_values[p, None].store(load_f32x8(residual, element))

    @cute.jit
    def _boundary(
        self,
        row_tidx,
        cluster_rank,
        token,
        branch,
        stream_values,
        params,
        outputs,
        smem,
    ) -> None:
        """Post-norm, gated residual and next RMSNorm of ``token``'s reduced row.
        ``reduction``: two (warps) areas, or (4,warps):(warps,1) with clusters."""
        post_gamma, next_gamma, alpha_values, beta_values = params
        residual_output, norm_output, router_output = outputs
        reduction, partials, mbarrier = smem
        constants = self.constants
        packs = self.packs
        input_scale = constants.has_post_weight and constants.post_weight_is_input_scale
        post_scale = (
            constants.has_post_weight and not constants.post_weight_is_input_scale
        )
        layout = cute.make_layout((packs, VEC_BF16), stride=(VEC_BF16, 1))
        values = cute.make_rmem_tensor(layout, Float32)
        updated = cute.make_rmem_tensor(layout, Float32)
        post_sum = Float32(0.0)
        moments = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        moments.fill(Float32(0.0))
        # 1. Branch statistic; with clusters, four moments: b^2, (alpha s)^2,
        # (alpha s)(beta b'), (beta b')^2.
        for p in cutlass.range_constexpr(packs):
            values[p, None].store(
                packed_u32x4_to_bf16x8(branch[p, None].load()).to(Float32)
            )
            if cutlass.const_expr(input_scale):
                scaled = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
                for e in cutlass.range_constexpr(VEC_BF16):
                    scaled[e] = F.mul(values[p, e], post_gamma[p, e])
                values[p, None].store(scaled.load().to(BFloat16).to(Float32))
            for e in cutlass.range_constexpr(VEC_BF16):
                if cutlass.const_expr(self.cluster_size == 1):
                    post_sum = F.fma(values[p, e], values[p, e], post_sum)
                else:
                    moment = values[p, e]
                    if cutlass.const_expr(post_scale):
                        moment = F.mul(moment, post_gamma[p, e])
                    gated = F.mul(alpha_values[p, e], stream_values[p, e])
                    scaled = F.mul(beta_values[p, e], moment)
                    updated[p, e] = gated
                    moments[0] = F.fma(values[p, e], values[p, e], moments[0])
                    moments[1] = F.fma(gated, gated, moments[1])
                    moments[2] = F.fma(gated, scaled, moments[2])
                    moments[3] = F.fma(scaled, scaled, moments[3])
        inv_hidden = Float32(1.0 / self.hidden)
        if cutlass.const_expr(self.cluster_size == 1):
            post_sum = self._cta_sum(post_sum, reduction)
            post_rstd = cute.math.rsqrt(
                F.fma(post_sum, inv_hidden, Float32(constants.post_eps)), approx=True
            )
            # 2. Gated residual and its statistic (second area: no WAR barrier).
            next_sum = Float32(0.0)
            for p in cutlass.range_constexpr(packs):
                normed = self._normed_bf16(values, post_gamma, post_rstd, p)
                for e in cutlass.range_constexpr(VEC_BF16):
                    if cutlass.const_expr(constants.gate == "none"):
                        updated[p, e] = F.add(stream_values[p, e], normed[e])
                    else:
                        updated[p, e] = F.fma(
                            alpha_values[p, e],
                            stream_values[p, e],
                            F.mul(beta_values[p, e], normed[e]),
                        )
                    next_sum = F.fma(updated[p, e], updated[p, e], next_sum)
            next_sum = self._cta_sum(next_sum, reduction + self.warps)
            next_rstd = cute.math.rsqrt(
                F.fma(next_sum, inv_hidden, Float32(constants.next_eps)), approx=True
            )
        else:
            # 2. CTA sums of the four moments behind one barrier.
            tidx, _, _ = cute.arch.thread_idx()
            lane = cute.arch.lane_idx()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            for k in cutlass.range_constexpr(4):
                moments[k] = self._warp_sum(moments[k], WARP_SIZE)
                with cute.arch.elect_one():
                    cute.arch.store(
                        (reduction + k * self.warps + warp).llvm_ptr, moments[k]
                    )
            cute.arch.barrier()
            for k in cutlass.range_constexpr(4):
                value = Float32(0.0)
                if lane < self.warps:
                    value = cute.arch.load(
                        (reduction + k * self.warps + lane).llvm_ptr, Float32
                    )
                moments[k] = self._warp_sum(value, WARP_SIZE)
            # 3. Send the CTA sums to every peer CTA, sum the peers' sums.
            cute.arch.cluster_wait()
            if tidx < self.cluster_size and tidx != cluster_rank:
                words = cute.recast_tensor(moments, Int32)  # bit view
                cute.arch.store_async_dsmem(
                    (partials + cluster_rank * 4).align(16),
                    (words[0], words[1], words[2], words[3]),
                    mbarrier,
                    tidx,
                )
            while not cute.arch.mbarrier_try_wait(mbarrier, 0):  # spin, no sleep
                pass
            slot = tidx % self.cluster_size
            for k in cutlass.range_constexpr(4):
                peer_sum = Float32(0.0)
                if slot != cluster_rank:
                    peer_sum = cute.arch.load(
                        (partials + slot * 4 + k).llvm_ptr, Float32
                    )
                moments[k] = moments[k] + self._warp_sum(peer_sum, self.cluster_size)
            post_rstd = cute.math.rsqrt(
                F.fma(moments[0], inv_hidden, Float32(constants.post_eps)), approx=True
            )
            next_sum = F.fma(
                post_rstd,
                F.fma(post_rstd, moments[3], F.mul(Float32(2.0), moments[2])),
                moments[1],
            )
            next_sum = fmax_f32(next_sum, Float32(0.0))
            next_rstd = cute.math.rsqrt(
                F.fma(next_sum, inv_hidden, Float32(constants.next_eps)), approx=True
            )
            # 4. Gated residual from the reduced statistics.
            for p in cutlass.range_constexpr(packs):
                normed = self._normed_bf16(values, post_gamma, post_rstd, p)
                for e in cutlass.range_constexpr(VEC_BF16):
                    updated[p, e] = F.fma(beta_values[p, e], normed[e], updated[p, e])

        # Outputs: FP32 residual, BF16 norm, optional FP32 router copy.
        for p in cutlass.range_constexpr(packs):
            column = (
                Int64(token) * self.hidden
                + Int64(row_tidx + p * self.num_row_threads) * VEC_BF16
            )
            normalized = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            for e in cutlass.range_constexpr(VEC_BF16):
                normalized[e] = F.mul(updated[p, e], next_rstd)
                if cutlass.const_expr(constants.has_next_weight):
                    normalized[e] = F.mul(normalized[e], next_gamma[p, e])
            store_f32x8(residual_output, column, updated[p, None].load())
            store_global_u32x4(
                Int64((norm_output.iterator + column).toint()),
                bf16x8_to_packed_u32x4(normalized.load().to(BFloat16)),
            )
            if cutlass.const_expr(constants.has_router):
                store_f32x8(router_output, column, normalized.load())

    # -------------------------------------------------------------------------
    # Kernel
    # -------------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        partial: cute.Tensor,
        residual: cute.Tensor,
        alpha: cute.Tensor | None,
        beta: cute.Tensor | None,
        post_weight: cute.Tensor | None,
        next_weight: cute.Tensor | None,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        router_output: cute.Tensor | None,
        buffer: cute.Tensor,
        peer_buffers: cute.Tensor | None,
        gather_buffer: cute.Tensor,
        multicast_address: Int64,
        sync: cute.Tensor,
        slot_bytes: Int64,
        m: Int32,
    ) -> None:
        # ---------------------------------------------------------------------
        # Special registers
        # ---------------------------------------------------------------------
        constants = self.constants
        cluster_size = self.cluster_size
        packs = self.packs
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        cluster_rank = Int32(0)
        if cutlass.const_expr(cluster_size > 1):
            cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # Pack p of this thread is fragment row_tidx + p * num_row_threads of a token.
        row_tidx = cluster_rank * self.num_threads + tidx

        # ---------------------------------------------------------------------
        # Shared storage and cluster mbarrier
        # ---------------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        partials = mbarrier = None
        if cutlass.const_expr(cluster_size == 1):
            reduction = smem.allocate_array(Float32, 2 * self.warps)
        else:
            partials = smem.allocate_array(Float32, cluster_size * 4)  # (C,4):(4,1)
            mbarrier = smem.allocate_array(Int64, 1)
            reduction = smem.allocate_array(Float32, 4 * self.warps)
            # Expect (C - 1) peer moment quads on the mbarrier.
            if warp == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(mbarrier, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbarrier, (cluster_size - 1) * 16
                    )
                    cute.arch.mbarrier_init_fence()
            # The matching wait must come before the first remote store.
            cute.arch.cluster_arrive_relaxed()

        # ---------------------------------------------------------------------
        # Load weights (before the PDL wait)
        # ---------------------------------------------------------------------
        # Each is (P,8):(8,1).
        layout = cute.make_layout((packs, VEC_BF16), stride=(VEC_BF16, 1))
        post_gamma = cute.make_rmem_tensor(layout, Float32)
        next_gamma = cute.make_rmem_tensor(layout, Float32)
        alpha_values = cute.make_rmem_tensor(layout, Float32)
        beta_values = cute.make_rmem_tensor(layout, Float32)
        alpha_values.fill(Float32(1.0))
        beta_values.fill(Float32(1.0))
        for p in cutlass.range_constexpr(packs):
            column = Int64(row_tidx + p * self.num_row_threads) * VEC_BF16
            if cutlass.const_expr(constants.has_post_weight):
                post_gamma[p, None].store(load_bf16x8_as_f32(post_weight, column))
            if cutlass.const_expr(constants.has_next_weight):
                next_gamma[p, None].store(load_bf16x8_as_f32(next_weight, column))
            if cutlass.const_expr(constants.gate == "channel"):
                alpha_values[p, None].store(load_f32x8(alpha, column))
                beta_values[p, None].store(load_f32x8(beta, column))
            elif cutlass.const_expr(constants.gate == "scalar"):
                for e in cutlass.range_constexpr(VEC_BF16):
                    alpha_values[p, e] = alpha[0]
                    beta_values[p, e] = beta[0]
        branch = cute.make_rmem_tensor(
            cute.make_layout((packs, 4), stride=(4, 1)), Uint32
        )
        stream_values = cute.make_rmem_tensor(layout, Float32)
        params = (post_gamma, next_gamma, alpha_values, beta_values)
        outputs = (residual_output, norm_output, router_output)
        shared = (reduction, partials, mbarrier)

        # ---------------------------------------------------------------------
        # PDL wait and exchange slots
        # ---------------------------------------------------------------------
        # Wait for the upstream partial (or plane) and residual.
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        phase = Int64(0)
        if cutlass.const_expr(self.exchange != MAILBOX):
            phase = Int64(push_plane_phase(sync)) * self.tp * slot_bytes
        s = _Slots(
            Int64(buffer.iterator.toint()),
            Int64(gather_buffer.iterator.toint()),
            multicast_address,
            sync,
            phase,
            slot_bytes,
            m,
        )

        # ---------------------------------------------------------------------
        # Exchange and boundary
        # ---------------------------------------------------------------------
        if cutlass.const_expr(cluster_size > 1):
            # One cluster per token.
            token = block
            self._load_residual(row_tidx, residual, token, stream_values)

            # =================================================================
            # All-reduce (the next kernel may launch once every input is read)
            # =================================================================
            if cutlass.const_expr(not constants.one_shot):
                # Two-shot: scatter slices, reduce this rank's slice, gather the sums.
                self._scatter(s, row_tidx, partial, peer_buffers, token)
                self._reduce(s, row_tidx, partial, token, branch)
                if cutlass.const_expr(self.enable_pdl):
                    cute.arch.griddepcontrol_launch_dependents()
                self._gather(s, row_tidx, token, branch)
            elif cutlass.const_expr(constants.pushed):
                # One-shot, pushed: the producer already stored every rank's row.
                if cutlass.const_expr(self.enable_pdl):
                    cute.arch.griddepcontrol_launch_dependents()
                self._reduce(s, row_tidx, partial, token, branch)
            else:
                # One-shot, published: broadcast this rank's row, sum every rank's row.
                self._broadcast(s, row_tidx, partial, token)
                if cutlass.const_expr(self.enable_pdl):
                    cute.arch.griddepcontrol_launch_dependents()
                self._reduce(s, row_tidx, partial, token, branch)

            # =================================================================
            # Boundary
            # =================================================================
            self._boundary(
                row_tidx,
                cluster_rank,
                token,
                branch,
                stream_values,
                params,
                outputs,
                shared,
            )
            # Re-arm only after the cluster reduction: every CTA has read the
            # generation.
            if cluster_rank == 0 and warp == 0:
                with cute.arch.elect_one():
                    self._advance(sync, token, m, token)
        else:
            # Persistent CTAs over tokens block + k * ctas.

            # =================================================================
            # All-reduce, send side (then the next kernel may launch)
            # =================================================================
            if cutlass.const_expr(not constants.one_shot):
                # Two-shot: scatter every token's slices; reduce the first token's slice.
                for token in cutlass.range(block, m, ctas):
                    self._scatter(s, row_tidx, partial, peer_buffers, token)
                self._reduce(s, row_tidx, partial, block, branch)
            elif cutlass.const_expr(not constants.pushed):
                # One-shot, published: broadcast every token's row.
                for token in cutlass.range(block, m, ctas):
                    self._broadcast(s, row_tidx, partial, token)
            # One-shot, pushed: the producer already stored every rank's row.
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()

            for token in cutlass.range(block, m, ctas):
                self._load_residual(row_tidx, residual, token, stream_values)
                # =============================================================
                # All-reduce, receive side
                # =============================================================
                if cutlass.const_expr(not constants.one_shot):
                    # Two-shot: reduce the next token's slice before gathering this one.
                    if token + ctas < m:
                        self._reduce(s, row_tidx, partial, token + ctas, branch)
                    self._gather(s, row_tidx, token, branch)
                else:
                    # One-shot: sum every rank's row.
                    self._reduce(s, row_tidx, partial, token, branch)
                # =============================================================
                # Boundary
                # =============================================================
                self._boundary(
                    row_tidx,
                    cluster_rank,
                    token,
                    branch,
                    stream_values,
                    params,
                    outputs,
                    shared,
                )
                if cutlass.const_expr(self.exchange == MAILBOX):
                    # Re-arm only after the norm's barriers: every thread has read
                    # the generation.
                    if cutlass.const_expr(self.warps == 1):
                        cute.arch.barrier()
                    if warp == 0:
                        with cute.arch.elect_one():
                            self._advance(sync, token, m, token)
            if cutlass.const_expr(self.exchange != MAILBOX):
                if warp == 0:
                    with cute.arch.elect_one():
                        self._advance(sync, block, ctas, block)  # one commit per CTA


class LLAllReduceDualNorm:
    """Mailbox (or push plane) and compiled kernels of one TP group.

    Construction is collective over ``group``. Calls on all ranks must use the
    same ``m`` and must not overlap one another.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        tp_size: int,
        rank: int,
        capacity_m: int,
        group: dist.ProcessGroup,
        enable_pdl: bool,
        push_plane: PushPlane | None = None,
    ) -> None:
        if hidden_size % (tp_size * VEC_BF16):
            raise ValueError("hidden_size must split into 8-element rank shards")
        if dist.get_world_size(group) != tp_size or dist.get_rank(group) != rank:
            raise ValueError("ProcessGroup size or rank does not match tp_size or rank")
        self.hidden_size, self.tp_size, self.rank = hidden_size, tp_size, rank
        self.capacity_m, self.enable_pdl = capacity_m, enable_pdl
        self.push_plane = push_plane
        device = torch.device("cuda", torch.cuda.current_device())
        # Mailbox one-shot holds at most min(capacity_m, 128 // tp) tokens
        # (the plane's slot bounds its own).
        self.one_shot_rows = min(
            capacity_m, max(1, ONE_SHOT_MAX_TOKENS_X_TP // tp_size)
        )
        if push_plane is None:
            # [3, tp, capacity_m, H / tp] scatter mailbox; [3, capacity_m, H] gather
            # then [3, tp, one_shot_rows, H] one-shot mailboxes, multicast.
            shard = (LAMPORT_GENERATIONS, tp_size, capacity_m, hidden_size // tp_size)
            self._scatter = SymmetricBuffer.allocate(
                shard, torch.bfloat16, device, group, materialize_peer_addresses=True
            )
            rows = capacity_m + tp_size * self.one_shot_rows
            self._gather = SymmetricBuffer.allocate(
                (LAMPORT_GENERATIONS * rows, hidden_size),
                torch.bfloat16,
                device,
                group,
                require_multicast=True,
            )
            buffer = self._scatter.tensor.flatten()
            gather = self._gather.tensor.flatten()
            for mailbox in (buffer, gather):
                mailbox.view(torch.int16).fill_(-32768)
            peers = self._scatter.peer_addresses
            multicast = self._gather.multicast_address
            # One Lamport generation per token.
            sync = torch.zeros((capacity_m,), dtype=torch.int32, device=device)
            slot_bytes = 0
        else:
            plane = push_plane.buffer.view(torch.uint8).flatten()
            buffer = gather = plane[: 2 * push_plane.half_bytes].view(torch.bfloat16)
            peers = push_plane.peer_addresses
            multicast = push_plane.multicast_address
            sync = push_plane.epoch.view(torch.int32)
            slot_bytes = push_plane.half_bytes // tp_size
        self._buffer_elements, self._gather_elements = buffer.numel(), gather.numel()
        self._sync_words = sync.numel()
        self._buffer, self._gather_buffer = to_cute(buffer, 16), to_cute(gather, 16)
        self._peer_buffers = to_cute(peers, 8) if peers is not None else None
        self._multicast = Int64(multicast)
        self._sync = to_cute(sync, 4)
        self._sync_tensor = sync  # keeps the generations alive
        self._slot_bytes = Int64(slot_bytes)
        self._sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        self._compiled: dict[_DualNormConstants, Any] = {}
        torch.cuda.synchronize()
        dist.barrier(group)

    def compiled(self, constants: _DualNormConstants) -> Any:
        """The kernel for one constant set; compile it before graph capture."""
        kernel = self._compiled.get(constants)
        if kernel is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Compile LLAllReduceDualNorm before CUDA graph capture"
                )
            kernel = self._compiled[constants] = self._compile(constants)
        return kernel

    def _compile(self, constants: _DualNormConstants) -> Any:
        hidden = self.hidden_size
        device_kernel = _AllReduceDualNormDeviceKernel(
            hidden=hidden,
            tp=self.tp_size,
            rank=self.rank,
            capacity_m=self.capacity_m,
            one_shot_rows=self.one_shot_rows,
            constants=constants,
            exchange=MAILBOX if self.push_plane is None else PLANE,
            atom_bytes=2 if self.push_plane is None else self.push_plane.atom_bytes,
            enable_pdl=self.enable_pdl,
        )
        rows_bf16 = make_fake_dynamic_compact_tensor(
            BFloat16, alignment=16, divisibility=hidden
        )
        rows_f32 = make_fake_dynamic_compact_tensor(
            Float32, alignment=32, divisibility=hidden
        )
        gate_len = 1 if constants.gate == "scalar" else hidden
        gate = make_fake_compact_tensor(
            Float32, (gate_len,), assumed_align=32 if gate_len > 1 else 4
        )
        gate = gate if constants.gate != "none" else None
        weight = make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16)
        flat = lambda n: make_fake_compact_tensor(BFloat16, (n,), assumed_align=16)  # noqa: E731
        return cute.compile(
            device_kernel,
            rows_bf16,
            rows_f32,
            gate,
            gate,
            weight if constants.has_post_weight else None,
            weight if constants.has_next_weight else None,
            rows_f32,
            rows_bf16,
            rows_f32 if constants.has_router else None,
            flat(self._buffer_elements),
            make_fake_compact_tensor(Int64, (self.tp_size,), assumed_align=8)
            if not constants.one_shot
            else None,
            flat(self._gather_elements),
            Int64(0),
            make_fake_compact_tensor(Int32, (self._sync_words,), assumed_align=4),
            Int64(0),
            Int32(self.capacity_m),
            Int32(self.capacity_m),
            current_cu_stream(),
        )

    def __call__(
        self,
        partial: torch.Tensor | None,
        residual: torch.Tensor,
        residual_output: torch.Tensor,
        norm_output: torch.Tensor,
        router_output: torch.Tensor | None,
        *,
        constants: _DualNormConstants,
        alpha: torch.Tensor | None,
        beta: torch.Tensor | None,
        post_weight: torch.Tensor | None,
        next_weight: torch.Tensor | None,
    ) -> None:
        """``partial`` None: the rows are already in the push plane (consumer only)."""
        m = residual.shape[0]
        # Persistent CTAs must all be resident: at most one per SM.
        ctas = m if constants.tuning.cluster_size > 1 else min(m, self._sm_count)
        rows = lambda t, a: to_cute_dynamic(
            t.flatten(), a, divisibility=self.hidden_size
        )  # noqa: E731
        vector = lambda t, a: to_cute(t, a) if t is not None else None  # noqa: E731
        gate_align = 32 if constants.gate == "channel" else 4
        self.compiled(constants)(
            rows(partial if partial is not None else norm_output, 16),
            rows(residual, 32),
            vector(alpha, gate_align),
            vector(beta, gate_align),
            vector(post_weight, 16),
            vector(next_weight, 16),
            rows(residual_output, 32),
            rows(norm_output, 16),
            rows(router_output, 32) if router_output is not None else None,
            self._buffer,
            self._peer_buffers if not constants.one_shot else None,
            self._gather_buffer,
            self._multicast,
            self._sync,
            self._slot_bytes,
            Int32(m),
            Int32(ctas),
            current_cu_stream(),
        )
