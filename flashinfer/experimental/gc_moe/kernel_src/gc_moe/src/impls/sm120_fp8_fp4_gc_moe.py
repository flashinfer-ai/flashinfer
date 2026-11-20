"""SM120 CuTe dispatch, swap-AB routed/shared compute, and combine kernels.

Role1/role3 follow the SM90 structure over the packed-UE8M0 SM120
workspace. Role2 runs the block-scaled mma.sync mainloop: weights stream
through a 4D TensorMap with the FP4->8bit SMEM unpack, activations through
a 2D TensorMap, and the packed UE8M0 scale words stage with cp.async into
the SFA/SFB SMEM layouts.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import (Boolean, Float32, Float4E2M1FN, Float8E4M3FN,
                     Float8E8M0FNU, Int8, Int32, Int64, Uint8, Uint16, Uint32,
                     Uint64)
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from ..common.math import (
    abs_f32,
    bits_float,
    fast_exp_neg_f32,
    round_bf16_f32,
    f32_to_bf16_bits,
    f32x2_to_bf16x2_bits,
    f32x2_to_e4m3x2_bits,
    ue8m0_fp8_inv,
    ue8m0_fp8_scale_exp,
)
from ..mma.sm120 import (
    MMA_N,
    _arch_mma_m16n8k32_mxfp8,
    issue_m64n8k32_mxfp8_packed_sfb,
    lds_b32_raw_ordered,
    load_packed_fp4_fragment,
    make_sm120_ldmatrix_atom,
    make_swapab_m64n8k128_tiled_mma,
    shift_fp4_fragment_for_mxf8f6f4,
)
from ..ptx.ld_st import (
    EVICT_FIRST,
    EVICT_LAST,
    atom_acq_rel_add_u32,
    atom_add_u32,
    atom_release_gpu_add_u32,
    atom_shared_add_u32,
    copy_global_16,
    ld_acquire_gpu_u32,
    ld_acquire_gpu_u64,
    ld_acquire_sys_u32_sm90 as ld_acquire_sys_u32,
    ld_global_16,
    ld_shared_u32x4,
    load_bf16_8,
    st_global_16,
    st_global_u32x4,
    stmatrix_bf16x8_trans,
    st_release_u32,
    store_bf16_8,
)
from ..ptx.tma import (
    bulk_g2s_1d,
    bulk_s2g_1d,
    copy_arrive,
    copy_async_4,
    copy_async_16,
    copy_commit,
    copy_wait,
    global_async_proxy_fence,
    prefetch_g2s_4d,
    prefetch_tensormap,
    tile_g2s_2d,
    tile_g2s_4d,
)
from ..ptx.utils import (
    fence_gpu,
    fence_sys,
    sync_aligned,
    sync_math,
    sync_tile,
)


# Swap-AB tile: 64 weight rows, an adaptive token width, and K128.
MMA_TILE_M, MMA_TILE_K = 64, 128
MMA_SF_STAGE_BYTES = 128 * MMA_TILE_K // 32
RELAY_CHUNK_BYTES = 16 * 1024
RELAY_SMEM_BYTES = 2 * (RELAY_CHUNK_BYTES + 8)
_ROW_BYTES = 128
_CELL_BYTES = 64 * _ROW_BYTES


def shared_gemm_stage_bytes(tile_n):
    alignment = 1024
    size = 8192 + 512 + tile_n * 132
    return (size + alignment - 1) // alignment * alignment


def make_data_smem_layout(tile_mnk, *, dtype, is_a, num_stages=1):
    """SM120 TMA/ldmatrix-compatible SW128 (M,K)/(N,K) staged SMEM layout."""
    smem_shape = (tile_mnk[0] if is_a else tile_mnk[1], tile_mnk[2])
    layout = utils.LayoutEnum.ROW_MAJOR
    smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(layout, dtype, smem_shape[1]), dtype)
    return cute.tile_to_shape(
        smem_layout_atom, cute.append(smem_shape, num_stages), order=(0, 1, 2))


@dsl_user_op
def add_f32_pair(a, b, x, y, *, loc=None, ip=None):
    pair = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, x, y)],
        '{ .reg .b64 a,b,s; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; '
        'add.rn.f32x2 s,a,b; mov.b64 {$0,$1},s; }',
        '=f,=f,f,f,f,f', has_side_effects=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return tuple(Float32(llvm.extractvalue(T.f32(), pair, [i], loc=loc, ip=ip))
                 for i in range(2))

@dsl_user_op
def check_smem_budget(used: Uint32, budget: Uint32, *, loc=None, ip=None):
    """Fail before touching the handoff if a compiler changes its footprint."""
    llvm.inline_asm(
        None, [used.ir_value(loc=loc, ip=ip), budget.ir_value(loc=loc, ip=ip)],
        '{ .reg .pred overflow; setp.gt.u32 overflow, $0, $1; @overflow trap; }',
        'r,r', has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def handoff_load_shared_ordered(ptr, *, loc=None, ip=None):
    values = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4), [ptr.toint().ir_value()],
        'ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];',
        '=r,=r,=r,=r,r,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), values, [i], loc=loc, ip=ip))
                 for i in range(4))


@dsl_user_op
def ballot_leading_ones(mask: Int32, *, loc=None, ip=None) -> Uint32:
    """Run length from a warp ballot mask: the count of leading set bits,
    32 when every lane voted yes (fns.b32's not-found sentinel 0x80000000
    folds to the window bound through the unsigned min)."""
    result = llvm.inline_asm(
        T.i32(), [mask.ir_value(loc=loc, ip=ip)],
        '{ .reg .b32 x, n; not.b32 x, $1; fns.b32 n, x, 0, 1; '
        'min.u32 n, n, 32; mov.s32 $0, n; }',
        '=r,r', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(result)


@cute.jit
def wait_count(ptr: cute.Pointer, target: Uint32):
    value = ld_acquire_gpu_u32(ptr)
    while value < target:
        value = ld_acquire_gpu_u32(ptr)


@dsl_user_op
def dispatch_cache_claim(ptr, *, loc=None, ip=None):
    value = llvm.inline_asm(
        T.i64(), [ptr.toint().ir_value()],
        'atom.acq_rel.gpu.global.cas.b64 $0, [$1], 0, 1;',
        '=l,l,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return Uint64(value)


@dsl_user_op
def dispatch_cache_publish(ptr, address: Uint64, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [ptr.toint().ir_value(), address.ir_value()],
        'st.release.gpu.global.u64 [$0], $1;', 'l,l,~{memory}',
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@cute.jit
def cell(base: Uint64, offset: Uint64, dtype: cutlass.Constexpr):
    return cute.make_tensor(cute.make_ptr(dtype, base + offset, cute.AddressSpace.gmem,
                                         assumed_align=1), cute.make_layout((1,)))


@cute.jit
def wait_epoch(ptr: cute.Pointer, target: Uint32, system: cutlass.Constexpr):
    # Same-order all-rank calls and the tail rendezvous prevent skipping a
    # still-needed epoch. Equality also remains valid at uint32 wraparound.
    if cutlass.const_expr(system):
        value = ld_acquire_sys_u32(ptr)
    else:
        value = ld_acquire_gpu_u32(ptr)
    while value != target:
        if cutlass.const_expr(system):
            value = ld_acquire_sys_u32(ptr)
        else:
            value = ld_acquire_gpu_u32(ptr)


class SM120Dispatch:
    """Scan routes, pull peer activations and publish expert readiness."""

    def _configure_dispatch(self, schedule):
        self.mirror_metadata = schedule.mirror_metadata
        self.pull_tma = self.role == 1 and schedule.pull_engine
        self.pull_warps = self.threads // 32
        self.pull_rb = 8 if self.local < 48 else 1
        self.dispatch_cache = self.pull_tma and schedule.dispatch_cache
        if self.dispatch_cache:
            self.pull_rb = 4
        if self.batched_returns and not self.mirror_metadata:
            self.pull_rb = 2

    @cute.jit
    def _field(self, base: Uint64, name: cutlass.Constexpr, dtype: cutlass.Constexpr):
        return cute.make_ptr(dtype, base + Uint64(getattr(self, 'off_' + name)),
                             cute.AddressSpace.gmem, assumed_align=16)

    @cute.jit
    def _dispatch_grid(self, head: cute.Pointer, epoch: Uint32,
                       counter: cutlass.Constexpr, signal: cutlass.Constexpr,
                       publish_source: cutlass.Constexpr,
                       publish_slot: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(publish_source):
            fence_sys()
        else:
            fence_gpu()
        cute.arch.sync_threads()
        if tid == 0:
            arrived = atom_acq_rel_add_u32(head + counter, Uint32(1))
            if arrived == self.grid - 1:
                if cutlass.const_expr(publish_source):
                    st_release_u32(head + publish_slot, epoch, True)
                st_release_u32(head + signal, epoch, False)
            wait_epoch(head + signal, epoch, False)
        cute.arch.sync_threads()

    @cute.jit
    def _poll_peers(self, base: Uint64, peers: cute.Pointer, head: cute.Pointer,
                    epoch: Uint32, remote_slot: cutlass.Constexpr,
                    gate_slot: cutlass.Constexpr):
        """Wait until every peer publishes `remote_slot` for this epoch.

        One block polls the peers over PCIe and opens a local gate, so the
        other blocks spin on local memory instead of re-issuing hundreds of
        sys-scope acquire loads (the same shape the direct scan uses).
        """
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        if block == 0:
            if tid < self.ranks:
                wait_epoch(self._field(Uint64(peer_table[tid]), 'header', Uint32)
                           + remote_slot, epoch, True)
            cute.arch.sync_threads()
            if tid == 0:
                st_release_u32(head + gate_slot, epoch, False)
        else:
            if tid == 0:
                wait_epoch(head + gate_slot, epoch, False)
        cute.arch.sync_threads()

    @cute.jit
    def _pull_row(self, src: Uint64, dst: Uint64, lane: Uint32):
        """Copy one activation row from a peer (lane-load fallback).

        A fused load/store pair stalls a full PCIe latency per chunk, so a
        batch keeps several reads in flight at a register cost of four
        words per chunk.
        """
        if cutlass.const_expr(self.hidden % 512 == 0):
            chunks = self.hidden // 512
            for base_chunk in cutlass.range_constexpr(0, chunks, 4):
                batch = []
                for chunk in cutlass.range_constexpr(4):
                    index = base_chunk + chunk
                    if cutlass.const_expr(index < chunks):
                        offset = Uint64(index * 512) + Uint64(lane * 16)
                        batch.append(ld_global_16(src + offset))
                for chunk in cutlass.range_constexpr(4):
                    index = base_chunk + chunk
                    if cutlass.const_expr(index < chunks):
                        offset = Uint64(index * 512) + Uint64(lane * 16)
                        value_0, value_1, value_2, value_3 = batch[chunk]
                        st_global_16(dst + offset, value_0, value_1,
                                     value_2, value_3)
        else:
            for k in cutlass.range(lane, self.hidden // 16, 32):
                copy_global_16(dst + Uint64(k) * 16, src + Uint64(k) * 16)

    @cute.jit
    def _pull_issue(self, src: Uint64, lane: Uint32, warp: Uint32,
                    pull: cute.Tensor, bar: cute.Pointer, slot: Uint32):
        """Post one engine bulk copy of a peer row into the warp's slot.

        The copy's complete_tx flips the slot mbarrier's phase, so the warp
        can issue its next row into the other slot before draining this
        one: two engine copies stay in flight per warp.
        """
        slot_ptr = cute.recast_ptr(pull.iterator, dtype=Uint8) \
            + warp * Uint32(2 * self.hidden) + slot * Uint32(self.hidden)
        if lane == 0:
            src_ptr = cute.make_ptr(
                Uint8, src, cute.AddressSpace.gmem, assumed_align=16)
            bulk_g2s_1d(slot_ptr, src_ptr, Uint32(self.hidden),
                        bar + warp * Uint32(2) + slot)
            cute.arch.mbarrier_arrive_and_expect_tx(
                bar + warp * Uint32(2) + slot, Uint32(self.hidden))

    @cute.jit
    def _pull_cached_issue(self, src: Uint64, lane: Uint32, warp: Uint32,
                           pull: cute.Tensor, bar: cute.Pointer, slot: Uint32,
                           entry: Uint64, wait: cutlass.Constexpr = False):
        # Entries are empty (0), in flight (1), or a published local row address.
        state = Uint32(2)
        if lane == 0:
            ptr = cute.make_ptr(Uint64, entry, cute.AddressSpace.gmem, assumed_align=8)
            cached = ld_acquire_gpu_u64(ptr)
            if cutlass.const_expr(wait):
                while cached == Uint64(1):
                    cached = ld_acquire_gpu_u64(ptr)
            elif cached == Uint64(0):
                cached = dispatch_cache_claim(ptr)
            if cached <= Uint64(1):
                state = Uint32(cached)
            else:
                src = cached
                global_async_proxy_fence()
        state = cute.arch.shuffle_sync(state, 0)
        if state != Uint32(1):
            self._pull_issue(src, lane, warp, pull, bar, slot)
        return state

    @cute.jit
    def _pull_drain(self, dst: Uint64, lane: Uint32, warp: Uint32,
                    pull: cute.Tensor, bar: cute.Pointer, slot: Uint32,
                    phase: Uint32, cache_entry: Uint64 = 0):
        """Wait for the warp's older in-flight row and store it locally.

        Each slot owns its barrier (one arrival per use), so a barrier never
        carries two rows in one phase even when the engine queue backs up.
        Row k is the (k//2+1)-th use of slot k&1, so it completes that
        barrier's (k//2)&1 phase. The drain's vector stores are local
        writes, keeping the engine dedicated to peer reads.
        """
        cute.arch.mbarrier_wait(bar + warp * Uint32(2) + slot, phase)
        base_words = warp * Uint32(self.hidden // 2) \
            + slot * Uint32(self.hidden // 4)
        for chunk in cutlass.range_constexpr(self.hidden // 512):
            word = base_words + Uint32(chunk * 128 + lane * 4)
            value_0 = pull[word]
            value_1 = pull[word + Uint32(1)]
            value_2 = pull[word + Uint32(2)]
            value_3 = pull[word + Uint32(3)]
            st_global_16(dst + Uint64(chunk * 512 + lane * 16), value_0,
                         value_1, value_2, value_3)
        if cutlass.const_expr(self.dispatch_cache):
            if cache_entry != Uint64(0):
                cute.arch.sync_warp()
                if lane == 0:
                    dispatch_cache_publish(cute.make_ptr(Uint64, cache_entry,
                        cute.AddressSpace.gmem, assumed_align=8), dst)

    @cute.jit
    def _publish_expert(self, base: Uint64, stats: cute.Pointer,
                        slot: Uint32, total: Uint32,
                        head: cute.Pointer, epoch: Uint32):
        """Per-expert command publish (ordered sweep, batched pull).

        The group rendezvous waited for every CTA's share of eight experts
        before publishing any of them, so role2's claims idled behind the
        whole group's fabric stream. Publishing an expert the instant its
        last row lands overlaps role2's W TMA and mainloop with the
        remaining pull instead. Visibility: every pulling warp issues a
        release-add on expert_arrive[slot] after its drain (sync_warp
        first, so all lanes' row stores are ordered); the completing
        thread's acq-rel add acquires the whole release sequence, and its
        expert_ready/active_ready releases carry the rows to role2's
        claim-side acquires.
        """
        count_out = cell(base, Uint64(self.off_expert_count) + Uint64(slot) * 4, Uint32)
        count_out[0] = total
        if cutlass.const_expr(self.has_stats):
            recv_stats = cute.make_tensor(stats, cute.make_layout((self.local,)))
            recv_stats[slot] += total
        st_release_u32(self._field(base, 'expert_ready', Uint32) + Uint32(slot), epoch, False)
        if total > Uint32(0):
            active_slot = atom_add_u32(head + 7, Uint32(1))
            active_out = cell(base, Uint64(self.off_active_expert) + Uint64(active_slot) * 4, Uint32)
            active_out[0] = slot
            st_release_u32(self._field(base, 'active_ready', Uint32) + active_slot, epoch, False)
        # Every expert arrives once, including those with zero rows. The
        # last arrival acquires the other publishers before exposing the
        # final queue size to claimers through dispatch_done.
        finished = atom_acq_rel_add_u32(head + 8, Uint32(1))
        if finished + Uint32(1) >= self.local:
            st_release_u32(head + 9, epoch, False)

    @cute.jit
    def _dispatch_route_index(self, base, expert, source, slot):
        if cutlass.const_expr(self.batched_returns):
            entry = Uint64(expert) * self.ranks + Uint64(source)
            pos = Uint32(cell(base, Uint64(self.off_pos_base) + entry * 4, Uint32)[0]) + Uint32(slot)
            routes = self._field(base, 'batched_routes_in', Uint32)
            route = Uint32(routes[source * (self.capacity * self.topk) + pos])
        else:
            offset = ((Uint64(expert) * self.ranks + source) * self.capacity + slot) * 4
            route = Uint32(cell(base, Uint64(self.off_source_index) + offset, Uint32)[0])
        return route


    @cute.jit
    def _input_metadata(self, base, remote, source):
        if cutlass.const_expr(self.mirror_metadata):
            sf = base + Uint64(self.off_batched_scales_in) + Uint64(source) * (self.capacity * (self.hidden // 128) * 4)
            weight = base + Uint64(self.off_batched_weights_in) + Uint64(source) * (self.capacity * self.topk * 4)
        else:
            sf = remote + Uint64(self.off_x_sf)
            weight = remote + Uint64(self.off_topk_weights)
        return sf, weight

    @cute.jit
    def _push_batched_routes(self, base, peer_table, rank, epoch, tokens):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        if block == 0:
            lane = Uint32(tid % 32)
            for peer_idx in cutlass.range(tid // 32, self.ranks, self.threads // 32):
                peer = Uint32(peer_idx)
                start = peer * self.local
                end = start + Uint32(self.local - 1)
                bases = self._field(base, 'expert_base', Uint32)
                counts = self._field(base, 'send_count', Uint32)
                start_pos = Uint32(bases[start])
                end_pos = Uint32(bases[end]) + cute.math.min(Uint32(counts[end]), Uint32(self.capacity))
                remote = Uint64(peer_table[peer])
                if end_pos > start_pos:
                    low = start_pos // Uint32(4)
                    high = (end_pos + Uint32(3)) // Uint32(4)
                    remote_start = (remote + Uint64(self.off_batched_routes_in)
                                    + Uint64(rank) * (self.capacity * self.topk * 4))
                    for i in cutlass.range(low + lane, high, 32):
                        offset = Uint64(i) * 16
                        copy_global_16(remote_start + offset,
                                       base + Uint64(self.off_batched_routes_out) + offset)
                if cutlass.const_expr(self.mirror_metadata):
                    # Stage the scale and route-weight vectors with
                    # the indices. Publish the peer epoch after all stores.
                    scales = self.hidden // 128
                    sf_base = (remote + Uint64(self.off_batched_scales_in)
                               + Uint64(rank) * (self.capacity * scales * 4))
                    weight_base = (remote + Uint64(self.off_batched_weights_in)
                                   + Uint64(rank) * (self.capacity * self.topk * 4))
                    for vec in cutlass.range(lane, (tokens * scales + 3) // 4, 32):
                        offset = Uint64(vec) * 16
                        copy_global_16(sf_base + offset, base + Uint64(self.off_x_sf) + offset)
                    for vec in cutlass.range(lane, (tokens * self.topk + 3) // 4, 32):
                        offset = Uint64(vec) * 16
                        copy_global_16(weight_base + offset, base + Uint64(self.off_topk_weights) + offset)
            fence_sys()
            cute.arch.sync_threads()
            if tid == 0:
                st_release_u32(self._field(base, 'header', Uint32) + 1, epoch, True)


    @cute.jit
    def _dispatch(self, base: Uint64, peers: cute.Pointer, rank: Uint32,
                  tokens: Uint32, stats: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        head = self._field(base, 'header', Uint32)
        epoch = ld_acquire_gpu_u32(head)
        if block == 0 and tid == 0:
            token_out = cell(base, Uint64(8), Uint32)
            token_out[0] = tokens
        for e in cutlass.range(block * self.threads + tid, self.ranks * self.local,
                               self.grid * self.threads):
            out = cell(base, Uint64(self.off_send_count) + Uint64(e) * 4, Uint32)
            out[0] = Uint32(0)
        for e in cutlass.range(block * self.threads + tid, self.local,
                               self.grid * self.threads):
            out = cell(base, Uint64(self.off_expert_arrive) + Uint64(e) * 4, Uint32)
            out[0] = Uint32(0)
            if cutlass.const_expr(self.relay):
                done = self._field(base, 'relay_done', Uint32)
                done[e] = Uint32(0)
        if cutlass.const_expr(self.dispatch_cache):
            for i in cutlass.range(block * self.threads + tid, self.ranks * self.capacity,
                                   self.grid * self.threads):
                cache_entry = cell(base, Uint64(self.off_handoff) + Uint64(i) * 8, Uint64)
                cache_entry[0] = Uint64(0)
        # Four launch-bounded resident CTAs per communication SM make this
        # local grid rendezvous safe even when the compute partition is busy.
        self._dispatch_grid(head, epoch, 11, 12, False, 1)
        for route in cutlass.range(block * self.threads + tid, tokens * self.topk,
                                   self.grid * self.threads):
            expert_id = cell(base, Uint64(self.off_topk_idx) + Uint64(route) * 8, Int64)[0]
            if expert_id >= 0 and expert_id < self.ranks * self.local:
                expert_u = Uint32(expert_id)
                slot = atom_add_u32(self._field(base, 'send_count', Uint32) + expert_u, Uint32(1))
                if cutlass.const_expr(not self.batched_returns):
                    if slot < self.capacity:
                        destination = Uint64(peer_table[expert_u // self.local])
                        address = Uint64(self.off_source_index) + ((Uint64(expert_u % self.local) * self.ranks + rank) * self.capacity + slot) * 4
                        route_out = cell(destination, address, Uint32)
                        route_out[0] = Uint32(route)
                # The atomic slot is the only record of which arena position
                # this route owns; park it locally so the position pass below
                # (after the per-expert bases exist) can publish pos_table.
                slot_out = cell(base, Uint64(self.off_slot_table) + Uint64(route) * 4, Uint32)
                slot_out[0] = slot
        # Position passes: per-expert arena bases (expert-id prefix over this
        # rank's own send_count), then each route's position = base + slot.
        # Rows of one (expert, source) hold consecutive positions, so the
        # role2 scatter's source runs land as contiguous 128B lines.
        self._dispatch_grid(head, epoch, 20, 21, False, 1)
        for e in cutlass.range(block * self.threads + tid, self.ranks * self.local,
                               self.grid * self.threads):
            running = Uint32(0)
            for e2 in cutlass.range(e):
                count = cell(base, Uint64(self.off_send_count) + Uint64(e2) * 4, Uint32)[0]
                running += cute.math.min(Uint32(count), Uint32(self.capacity))
            base_out = cell(base, Uint64(self.off_expert_base) + Uint64(e) * 4, Uint32)
            base_out[0] = running
        self._dispatch_grid(head, epoch, 22, 23, False, 1)
        for route in cutlass.range(block * self.threads + tid, tokens * self.topk,
                                   self.grid * self.threads):
            expert_id = cell(base, Uint64(self.off_topk_idx) + Uint64(route) * 8, Int64)[0]
            if expert_id >= 0 and expert_id < self.ranks * self.local:
                slot = Uint32(cell(base, Uint64(self.off_slot_table) + Uint64(route) * 4, Uint32)[0])
                # Repeated experts can exhaust a source row. Mark rejected
                # routes for finalize without publishing a destination row.
                pos = Uint32(self.capacity * self.topk)
                if slot < self.capacity:
                    pos = Uint32(cell(base, Uint64(self.off_expert_base) + Uint64(expert_id) * 4, Uint32)[0]) + slot
                    if cutlass.const_expr(self.batched_returns):
                        outgoing = self._field(base, 'batched_routes_out', Uint32)
                        outgoing[pos] = Uint32(route)
                pos_out = cell(base, Uint64(self.off_pos_table) + Uint64(route) * 4, Uint32)
                pos_out[0] = pos
        if cutlass.const_expr(self.batched_returns):
            self._dispatch_grid(head, epoch, 13, 14, False, 1)
            self._push_batched_routes(base, peer_table, rank, epoch, tokens)
        else:
            self._dispatch_grid(head, epoch, 13, 14, True, 1)
        # One block polls the peers over PCIe, the rest wait on the local
        # gate instead of re-polling remote memory from every block.
        self._poll_peers(base, peers, head, epoch, 1, 15)
        alloc = cutlass.utils.SmemAllocator()
        counts = alloc.allocate_tensor(Uint32, cute.make_layout((self.ranks,)), 4)
        prefix = alloc.allocate_tensor(Uint32, cute.make_layout((self.ranks + 1,)), 4)
        total_shared = alloc.allocate_tensor(Uint32, cute.make_layout((1,)), 4)
        # Ordered-sweep group scratch: per-group expert row totals and the
        # flattened (expert, row) offsets across the group.
        group_total = alloc.allocate_tensor(Uint32, cute.make_layout((8,)), 4)
        group_prefix = alloc.allocate_tensor(Uint32, cute.make_layout((9,)), 4)
        if cutlass.const_expr(self.batched_returns):
            group_rotate = alloc.allocate_tensor(Uint32, cute.make_layout((8,)), 4)
        # Engine-pull staging: two x-row slots per warp, each with its own
        # mbarrier (one arrival per use, so a barrier never carries two
        # rows in one phase when the engine queue backs up).
        if cutlass.const_expr(self.pull_tma):
            pull_raw = alloc.allocate(self.pull_warps * 2 * self.hidden, 128)
            pull = cute.make_tensor(
                cute.recast_ptr(pull_raw, dtype=Uint32),
                cute.make_layout((self.pull_warps * self.hidden // 2,)))
            pull_bar = cute.recast_ptr(
                alloc.allocate(8 * self.pull_warps * 2, 8), dtype=Uint64)
            # One arrival per use: the copy's complete_tx satisfies the phase.
            if tid < self.pull_warps * 2:
                cute.arch.mbarrier_init(pull_bar + tid, 1)
            cute.arch.mbarrier_init_fence()
        # Stage the whole (expert, source) count matrix once: block 0
        # crosses PCIe for it in a single parallel round and opens a local
        # gate, so neither expert sweep below re-reads peer send_count.
        if block == 0:
            for idx in cutlass.range(tid, self.local * self.ranks, self.threads):
                expert = Uint32(idx) // self.ranks
                source = Uint32(idx) % self.ranks
                remote = Uint64(peer_table[source])
                source_count = self._field(remote, 'send_count', Uint32)
                value = ld_acquire_sys_u32(source_count + rank * self.local + expert)
                table_out = cell(base, Uint64(self.off_counts) + Uint64(idx) * 4, Uint32)
                table_out[0] = cute.math.min(value, Uint32(self.capacity))
                # Same round also stages each source's arena base for our
                # local expert: role2's scatter dest = base + slot, and the
                # base lives in the source's expert-id numbering.
                source_base = self._field(remote, 'expert_base', Uint32)
                bvalue = ld_acquire_sys_u32(source_base + rank * self.local + expert)
                btable_out = cell(base, Uint64(self.off_pos_base) + Uint64(idx) * 4, Uint32)
                btable_out[0] = bvalue
            fence_gpu()
            cute.arch.sync_threads()
            if tid == 0:
                st_release_u32(head + 24, epoch, False)
        if tid == 0:
            wait_epoch(head + 24, epoch, False)
        cute.arch.sync_threads()
        # Streaming dispatch: at decode batch sizes every expert's rows
        # cross PCIe at a shared ~25GB/s whether they are pulled or
        # pushed, so the lever is overlap, not the transport direction.
        # Role2's graph node has no edge back to role1, so it launches
        # alongside the sweep and consumes experts through the command
        # queue. The expert-ordered sweep lands expert 0 within one round
        # trip and publishes its command immediately, letting role2
        # compute under the remaining fabric stream instead of idling
        # behind the whole transfer. Small batches keep the flat
        # one-CTA-per-expert sweep: the transfer fits a single round trip
        # there, and the sweep order would only serialize it.
        if cutlass.const_expr(self.batched_returns):
            ordered = True
        else:
            ordered = tokens * self.topk >= 16 * self.local
        if ordered:
            # The effective PCIe round trip under an all-rank transfer is
            # ~18us, so throughput is in-flight bytes over that latency
            # until the ~30GB/s fabric cap: a single expert's rows (a few
            # hundred KB) drain at a fraction of the cap. The sweep groups
            # experts behind one barrier epoch instead, keeping hundreds of
            # rows in flight (the flat sweep's depth) while still landing
            # the first group's commands within one round trip of the scan,
            # so role2 computes under the remaining stream.
            my_warp = Uint32(block * (self.threads // 32) + warp)
            total_warps = self.grid * (self.threads // 32)
            # Warp's engine-pull pipeline state: row_k tracks slot/phase
            # (slot = k&1, use parity = (k//2)&1), and the older row stays
            # pending until the next iteration's issue has opened a second
            # in-flight copy.
            row_k = Uint32(0)
            pending_dst = Uint64(0)
            pending_k = Uint32(0)
            pending_valid = Uint32(0)
            # Per-expert publish state (batched pull): the expert slot of
            # the row whose drain is still pending, so its completion
            # atomic can fire once the drain's stores land.
            pending_expert = Uint32(0)
            pending_gslot = Uint32(0)
            pending_cache_entry = Uint64(0)
            for group_start in cutlass.range(0, self.local, 8):
                group_size = self.local - group_start
                if group_size > 8:
                    group_size = 8
                if tid < 8:
                    expert = group_start + tid
                    if expert < self.local:
                        running = Uint32(0)
                        rotated = Uint32(0)
                        for source in cutlass.range_constexpr(self.ranks):
                            rows = cell(base, Uint64(self.off_counts) + (Uint64(expert) * self.ranks + source) * 4, Uint32)[0]
                            running += rows
                            if cutlass.const_expr(self.batched_returns):
                                if Uint32(source) < rank:
                                    rotated += rows
                        if cutlass.const_expr(self.batched_returns):
                            group_rotate[tid] = rotated
                        group_total[tid] = running
                cute.arch.sync_threads()
                if tid == 0:
                    group_prefix[0] = Uint32(0)
                    for slot in cutlass.range_constexpr(1, 9):
                        slot_rows = group_total[slot - 1]
                        if slot > group_size:
                            slot_rows = Uint32(0)
                        group_prefix[slot] = group_prefix[slot - 1] + slot_rows
                cute.arch.sync_threads()
                group_rows = group_prefix[8]
                if cutlass.const_expr(self.pull_tma):
                    # Zero-row experts have no data to wait for: publish
                    # their commands at prefix time (count 0, no active
                    # slot) so the per-expert stream only gates experts
                    # that actually carry rows.
                    if block == 0 and tid == 0:
                        for expert in cutlass.range(0, group_size):
                            if group_total[expert] == Uint32(0):
                                slot = Uint32(group_start) + Uint32(expert)
                                self._publish_expert(base, stats, slot,
                                                     Uint32(0), head, epoch)
                # One warp owns one flattened (expert, row) pair at a time;
                # _pull_row keeps a batch of chunk loads in flight per warp,
                # and the group's rows span every warp, matching the flat
                # sweep's in-flight depth.
                if cutlass.const_expr(self.pull_tma and self.pull_rb > 1):
                    # Batched form: k indexes this warp's rows as
                    # row_flat = my_warp + k * total_warps. Phase A/B
                    # resolves the batch's routes (local reads) and issues
                    # every row's scale-factor and route-weight reads as
                    # one burst, so only the first consuming store pays a
                    # PCIe round trip; phase C runs the per-row pipeline
                    # (engine copy issue, pending drain, stores).
                    k_total = Uint32(0)
                    if group_rows > my_warp:
                        k_total = (group_rows - Uint32(1) - my_warp) \
                            // Uint32(total_warps) + Uint32(1)
                    for k0 in cutlass.range(0, k_total, self.pull_rb):
                        srcs, dsts, routes = [], [], []
                        experts_b, rows_b, sources_b, slots_b = [], [], [], []
                        sf0s, sf1s, ws = [], [], []
                        for b in cutlass.range_constexpr(self.pull_rb):
                            row_flat = my_warp \
                                + (Uint32(k0) + Uint32(b)) * Uint32(total_warps)
                            expert = Uint32(group_start)
                            row = Uint32(0)
                            source = Uint32(0)
                            source_slot = Uint32(0)
                            route = Uint32(0)
                            src = Uint64(0)
                            dst = Uint64(0)
                            sf0 = Uint32(0)
                            sf1 = Uint32(0)
                            w = Float32(0)
                            if row_flat < group_rows:
                                slot = Uint32(0)
                                for candidate in cutlass.range_constexpr(1, 9):
                                    if row_flat >= group_prefix[candidate]:
                                        slot = Uint32(candidate)
                                expert = Uint32(group_start) + slot
                                row = Uint32(row_flat) - group_prefix[slot]
                                if cutlass.const_expr(self.batched_returns):
                                    row += Uint32(group_rotate[slot])
                                    if row >= Uint32(group_total[slot]):
                                        row -= Uint32(group_total[slot])
                                running = Uint32(0)
                                source = Uint32(0)
                                lower = Uint32(0)
                                for candidate in cutlass.range_constexpr(0, self.ranks):
                                    count = cell(base, Uint64(self.off_counts) + (Uint64(expert) * self.ranks + candidate) * 4, Uint32)[0]
                                    if row >= running:
                                        source = Uint32(candidate)
                                        lower = running
                                    running += count
                                source_slot = row - lower
                                route = self._dispatch_route_index(base, expert, source, source_slot)
                                source_token = route // self.topk
                                remote = Uint64(peer_table[source])
                                sf_base, weight_base = self._input_metadata(base, remote, source)
                                src = remote + Uint64(self.off_x) + Uint64(source_token) * self.hidden
                                dst = base + Uint64(self.off_l1_acts) + (Uint64(expert) * self.rows + row) * self.hidden
                                if cutlass.const_expr(self.hidden // 128 > 32):
                                    if lane < Uint32(self.hidden // 128 - 32):
                                        sf1 = Uint32(cell(sf_base, (Uint64(source_token) * (self.hidden // 128) + lane + Uint32(32)) * 4, Uint32)[0])
                                if lane < Uint32(self.hidden // 128):
                                    sf0 = Uint32(cell(sf_base, (Uint64(source_token) * (self.hidden // 128) + lane) * 4, Uint32)[0])
                                if lane == 0:
                                    w = cell(weight_base, Uint64(route) * 4, Float32)[0]
                            srcs.append(src)
                            dsts.append(dst)
                            routes.append(route)
                            experts_b.append(expert)
                            rows_b.append(row)
                            sources_b.append(source)
                            slots_b.append(source_slot)
                            sf0s.append(sf0)
                            sf1s.append(sf1)
                            ws.append(w)
                        for b in cutlass.range_constexpr(self.pull_rb):
                            row_flat = my_warp \
                                + (Uint32(k0) + Uint32(b)) * Uint32(total_warps)
                            if row_flat < group_rows:
                                expert = experts_b[b]
                                row = rows_b[b]
                                source = sources_b[b]
                                source_slot = slots_b[b]
                                route = routes[b]
                                source_token, route_slot = route // self.topk, route % self.topk
                                meta_base = Uint64(self.off_route_meta) + Uint64(expert) * self.rows * 16
                                if lane < 3:
                                    rank_out = cell(base, meta_base + Uint64(row * 4) * 4, Uint32)
                                    rank_out[0] = source
                                    token_out = cell(base, meta_base + Uint64(row * 4 + 1) * 4, Uint32)
                                    token_out[0] = source_token
                                    slot_out = cell(base, meta_base + Uint64(row * 4 + 2) * 4, Uint32)
                                    slot_out[0] = route_slot
                                elif lane == 3:
                                    arena_base = cell(base, Uint64(self.off_pos_base) + (Uint64(expert) * self.ranks + source) * 4, Uint32)[0]
                                    pos_out = cell(base, meta_base + Uint64(row * 4 + 3) * 4, Uint32)
                                    pos_out[0] = arena_base + source_slot
                                cache_entry = Uint64(0)
                                cache_state = Uint32(2)
                                if cutlass.const_expr(self.dispatch_cache):
                                    cache_index = source * Uint32(self.capacity) + source_token
                                    cache_entry = base + Uint64(self.off_handoff) + Uint64(cache_index) * 8
                                    cache_state = self._pull_cached_issue(srcs[b], lane, warp, pull,
                                        pull_bar, row_k & Uint32(1), cache_entry)
                                else:
                                    self._pull_issue(srcs[b], lane, warp, pull, pull_bar, row_k & Uint32(1))
                                if pending_valid == 1:
                                    self._pull_drain(
                                        pending_dst, lane, warp, pull, pull_bar,
                                        pending_k & Uint32(1),
                                        (pending_k // Uint32(2)) & Uint32(1),
                                        pending_cache_entry)
                                    # Per-expert publish: the drained
                                    # row's completion atomic; the warp
                                    # that lands an expert's last row
                                    # publishes its command here
                                    # instead of at the group
                                    # rendezvous, so role2's claims
                                    # track the pull stream at expert
                                    # granularity.
                                    cute.arch.sync_warp()
                                    if lane == 0:
                                        arrived = atom_acq_rel_add_u32(
                                            self._field(base, 'expert_arrive', Uint32) + pending_expert, Uint32(1))
                                        total = group_total[pending_gslot]
                                        if arrived + Uint32(1) == total:
                                            self._publish_expert(
                                                base, stats, pending_expert,
                                                total, head, epoch)
                                if cutlass.const_expr(self.dispatch_cache):
                                    # A warp publishes its own outstanding row before
                                    # waiting on another warp, so cache owners cannot
                                    # form a cycle. Every dispatch CTA is resident.
                                    if cache_state == Uint32(1):
                                        self._pull_cached_issue(srcs[b], lane, warp, pull,
                                            pull_bar, row_k & Uint32(1), cache_entry, wait=True)
                                    pending_cache_entry = Uint64(0)
                                    if cache_state == Uint32(0):
                                        pending_cache_entry = cache_entry
                                if lane == 0:
                                    weight_out = cell(base, Uint64(self.off_route_weight) + Uint64(expert) * self.rows * 4 + Uint64(row) * 4, Float32)
                                    weight_out[0] = ws[b]
                                if cutlass.const_expr(self.hidden // 128 > 32):
                                    if lane < Uint32(self.hidden // 128 - 32):
                                        sf_out = cell(base, Uint64(self.off_l1_acts_sf) + ((Uint64(expert) * (self.hidden // 128) + lane + Uint32(32)) * self.rows + row) * 4, Uint32)
                                        sf_out[0] = sf1s[b]
                                if lane < Uint32(self.hidden // 128):
                                    sf_out = cell(base, Uint64(self.off_l1_acts_sf) + ((Uint64(expert) * (self.hidden // 128) + lane) * self.rows + row) * 4, Uint32)
                                    sf_out[0] = sf0s[b]
                                pending_dst = dsts[b]
                                pending_k = row_k
                                pending_valid = Uint32(1)
                                pending_expert = expert
                                pending_gslot = expert - Uint32(group_start)
                                row_k = row_k + Uint32(1)
                else:
                  for row_flat in cutlass.range(my_warp,
                                                group_rows, total_warps):
                    slot = Uint32(0)
                    for candidate in cutlass.range_constexpr(1, 9):
                        if row_flat >= group_prefix[candidate]:
                            slot = Uint32(candidate)
                    expert = Uint32(group_start) + slot
                    row = Uint32(row_flat) - group_prefix[slot]
                    if cutlass.const_expr(self.batched_returns):
                        row += Uint32(group_rotate[slot])
                        if row >= Uint32(group_total[slot]):
                            row -= Uint32(group_total[slot])
                    # Source scan straight off the local counts table: the
                    # running sum gives each source's row window.
                    running = Uint32(0)
                    source = Uint32(0)
                    lower = Uint32(0)
                    for candidate in cutlass.range_constexpr(0, self.ranks):
                        count = cell(base, Uint64(self.off_counts) + (Uint64(expert) * self.ranks + candidate) * 4, Uint32)[0]
                        if row >= running:
                            source = Uint32(candidate)
                            lower = running
                        running += count
                    source_slot = row - lower
                    route = self._dispatch_route_index(base, expert, source, source_slot)
                    source_token, route_slot = route // self.topk, route % self.topk
                    meta_base = Uint64(self.off_route_meta) + Uint64(expert) * self.rows * 16
                    if lane < 3:
                        rank_out = cell(base, meta_base + Uint64(row * 4) * 4, Uint32)
                        rank_out[0] = source
                        token_out = cell(base, meta_base + Uint64(row * 4 + 1) * 4, Uint32)
                        token_out[0] = source_token
                        slot_out = cell(base, meta_base + Uint64(row * 4 + 2) * 4, Uint32)
                        slot_out[0] = route_slot
                    elif lane == 3:
                        # Combine position: staged source base + slot within
                        # the (expert, source) window. Consecutive rows of a
                        # source run hold consecutive positions, so the L2
                        # scatter pushes contiguous lines.
                        arena_base = cell(base, Uint64(self.off_pos_base) + (Uint64(expert) * self.ranks + source) * 4, Uint32)[0]
                        pos_out = cell(base, meta_base + Uint64(row * 4 + 3) * 4, Uint32)
                        pos_out[0] = arena_base + source_slot
                    remote = Uint64(peer_table[source])
                    sf_base, weight_base = self._input_metadata(base, remote, source)
                    src = remote + Uint64(self.off_x) + Uint64(source_token) * self.hidden
                    dst = base + Uint64(self.off_l1_acts) + (Uint64(expert) * self.rows + row) * self.hidden
                    sf0 = Uint32(0)
                    sf1 = Uint32(0)
                    if cutlass.const_expr(self.pull_tma):
                        # Post the row's copy first, then the 4-byte scale
                        # reads: their PCIe latency rides under the engine
                        # copy this warp already has in flight.
                        self._pull_issue(src, lane, warp, pull, pull_bar,
                                         row_k & Uint32(1))
                        if cutlass.const_expr(self.hidden // 128 > 32):
                            if lane < Uint32(self.hidden // 128 - 32):
                                sf1 = Uint32(cell(sf_base, (Uint64(source_token) * (self.hidden // 128) + lane + Uint32(32)) * 4, Uint32)[0])
                        if lane < Uint32(self.hidden // 128):
                            sf0 = Uint32(cell(sf_base, (Uint64(source_token) * (self.hidden // 128) + lane) * 4, Uint32)[0])
                    else:
                        self._pull_row(src, dst, lane)
                        for sf_k in cutlass.range(lane, self.hidden // 128, 32):
                            sf_value = cell(sf_base, (Uint64(source_token) * (self.hidden // 128) + Uint32(sf_k)) * 4, Uint32)[0]
                            sf_out = cell(base, Uint64(self.off_l1_acts_sf) + ((Uint64(expert) * (self.hidden // 128) + Uint32(sf_k)) * self.rows + row) * 4, Uint32)
                            sf_out[0] = sf_value
                    if lane == 0:
                        weight_out = cell(base, Uint64(self.off_route_weight) + Uint64(expert) * self.rows * 4 + Uint64(row) * 4, Float32)
                        weight_out[0] = cell(weight_base, Uint64(route) * 4, Float32)[0]
                    if cutlass.const_expr(self.pull_tma):
                        if pending_valid == 1:
                            self._pull_drain(
                                pending_dst, lane, warp, pull, pull_bar,
                                pending_k & Uint32(1),
                                (pending_k // Uint32(2)) & Uint32(1))
                            # Per-expert publish from the legacy row
                            # loop: same completion atomic as the
                            # batched form, so role2's claims track
                            # expert completion even without the
                            # batched route resolution.
                            cute.arch.sync_warp()
                            if lane == 0:
                                arrived = atom_acq_rel_add_u32(
                                    self._field(base, 'expert_arrive', Uint32) + pending_expert, Uint32(1))
                                total = group_total[pending_gslot]
                                if arrived + Uint32(1) == total:
                                    self._publish_expert(
                                        base, stats, pending_expert,
                                        total, head, epoch)
                        if cutlass.const_expr(self.hidden // 128 > 32):
                            if lane < Uint32(self.hidden // 128 - 32):
                                sf_out = cell(base, Uint64(self.off_l1_acts_sf) + ((Uint64(expert) * (self.hidden // 128) + lane + Uint32(32)) * self.rows + row) * 4, Uint32)
                                sf_out[0] = sf1
                        if lane < Uint32(self.hidden // 128):
                            sf_out = cell(base, Uint64(self.off_l1_acts_sf) + ((Uint64(expert) * (self.hidden // 128) + lane) * self.rows + row) * 4, Uint32)
                            sf_out[0] = sf0
                        pending_dst = dst
                        pending_k = row_k
                        pending_valid = Uint32(1)
                        # Stash the row's expert slot for the legacy-loop
                        # completion atomic (dead when the publish runs
                        # from the batched loop or the group rendezvous).
                        pending_expert = expert
                        pending_gslot = expert - Uint32(group_start)
                        row_k = row_k + Uint32(1)
                if cutlass.const_expr(self.pull_tma):
                    if pending_valid == 1:
                        self._pull_drain(
                            pending_dst, lane, warp, pull, pull_bar,
                            pending_k & Uint32(1),
                            (pending_k // Uint32(2)) & Uint32(1),
                            pending_cache_entry)
                        # Same completion atomic for the warp's last
                        # pending row of the group.
                        cute.arch.sync_warp()
                        if lane == 0:
                            arrived = atom_acq_rel_add_u32(
                                self._field(base, 'expert_arrive', Uint32) + pending_expert, Uint32(1))
                            total = group_total[pending_gslot]
                            if arrived + Uint32(1) == total:
                                self._publish_expert(
                                    base, stats, pending_expert, total,
                                    head, epoch)
                    pending_valid = Uint32(0)
                cute.arch.sync_threads()
                if tid == 0:
                    # The last CTA to finish its share of the group
                    # publishes every command in it; its acquire pairs the
                    # other CTAs' release arrivals, so their pulled rows
                    # are visible before role2 reads them. Skipped whole
                    # under the per-expert publish (batched pull): the
                    # group-indexed arrival counter would collide with the
                    # per-expert row counters on the shared expert_arrive
                    # slots (group g's counter IS slot g).
                    if cutlass.const_expr(not (self.pull_tma)):
                        arrived = atom_acq_rel_add_u32(self._field(base, 'expert_arrive', Uint32) + Uint32(group_start // 8), Uint32(1))
                        if arrived == self.grid - 1:
                            fence_gpu()
                            for expert in cutlass.range(0, group_size):
                                slot = group_start + expert
                                total = group_total[expert]
                                count_out = cell(base, Uint64(self.off_expert_count) + Uint64(slot) * 4, Uint32)
                                count_out[0] = total
                                if cutlass.const_expr(self.has_stats):
                                    recv_stats = cute.make_tensor(stats, cute.make_layout((self.local,)))
                                    recv_stats[slot] += total
                                st_release_u32(self._field(base, 'expert_ready', Uint32) + Uint32(slot), epoch, False)
                                if total > 0:
                                    active_slot = atom_add_u32(head + 7, Uint32(1))
                                    active_out = cell(base, Uint64(self.off_active_expert) + Uint64(active_slot) * 4, Uint32)
                                    active_out[0] = Uint32(slot)
                                    st_release_u32(self._field(base, 'active_ready', Uint32) + active_slot, epoch, False)
                            # atom_add returns the pre-add value: the counter only
                            # reads local after this group's group_size lands.
                            finished = atom_acq_rel_add_u32(head + 8, Uint32(group_size))
                            if finished + Uint32(group_size) >= self.local:
                                st_release_u32(head + 9, epoch, False)
                cute.arch.sync_threads()
        else:
            # Persistent per-warp use counter: the slot mbarrier's phase
            # parity advances across experts, so row_k must never reset
            # mid-sweep. A per-expert reset lets a wait return before the
            # copy lands, and the next expert's arrive then lands in the
            # still-armed phase, wedging the barrier's tx count for good.
            row_k = Uint32(0)
            pending_dst = Uint64(0)
            pending_k = Uint32(0)
            pending_valid = Uint32(0)
            for expert in cutlass.range(block, self.local, self.grid):
                if tid < self.ranks:
                    counts[tid] = cell(base, Uint64(self.off_counts) + (Uint64(expert) * self.ranks + tid) * 4, Uint32)[0]
                cute.arch.sync_threads()
                if tid == 0:
                    running_count = Uint32(0)
                    prefix[0] = Uint32(0)
                    for source in cutlass.range_constexpr(self.ranks):
                        running_count += counts[source]
                        prefix[source + 1] = running_count
                    total_shared[0] = running_count
                cute.arch.sync_threads()
                total = total_shared[0]
                meta_base = Uint64(self.off_route_meta) + Uint64(expert) * self.rows * 16
                weight_base = Uint64(self.off_route_weight) + Uint64(expert) * self.rows * 4
                for row in cutlass.range(tid, total, self.threads):
                    source = Uint32(0)
                    for candidate in cutlass.range_constexpr(1, self.ranks):
                        if row >= prefix[candidate]:
                            source = Uint32(candidate)
                    source_slot = Uint32(row) - prefix[source]
                    route = self._dispatch_route_index(base, expert, source, source_slot)
                    source_token, route_slot = route // self.topk, route % self.topk
                    rank_out = cell(base, meta_base + Uint64(row * 4) * 4, Uint32)
                    rank_out[0] = source
                    token_out = cell(base, meta_base + Uint64(row * 4 + 1) * 4, Uint32)
                    token_out[0] = source_token
                    slot_out = cell(base, meta_base + Uint64(row * 4 + 2) * 4, Uint32)
                    slot_out[0] = route_slot
                    arena_base = cell(base, Uint64(self.off_pos_base) + (Uint64(expert) * self.ranks + source) * 4, Uint32)[0]
                    pos_out = cell(base, meta_base + Uint64(row * 4 + 3) * 4, Uint32)
                    pos_out[0] = arena_base + source_slot
                    weight_out = cell(base, weight_base + Uint64(row) * 4, Float32)
                    weight_out[0] = cell(Uint64(peer_table[source]), Uint64(self.off_topk_weights) + Uint64(route) * 4, Float32)[0]
                cute.arch.sync_threads()
                for pull_row in cutlass.range(
                        warp,
                        total, self.threads // 32):
                    source_rank = cell(base, meta_base + Uint64(pull_row * 4) * 4, Uint32)[0]
                    token = cell(base, meta_base + Uint64(pull_row * 4 + 1) * 4, Uint32)[0]
                    remote = Uint64(peer_table[source_rank])
                    src = remote + Uint64(self.off_x) + Uint64(token) * self.hidden
                    dst = base + Uint64(self.off_l1_acts) + (Uint64(expert) * self.rows + Uint32(pull_row)) * self.hidden
                    sf0 = Uint32(0)
                    sf1 = Uint32(0)
                    if cutlass.const_expr(self.pull_tma):
                        self._pull_issue(src, lane, warp, pull, pull_bar,
                                         row_k & Uint32(1))
                        if cutlass.const_expr(self.hidden // 128 > 32):
                            if lane < Uint32(self.hidden // 128 - 32):
                                sf1 = Uint32(cell(remote, Uint64(self.off_x_sf) + (Uint64(token) * (self.hidden // 128) + lane + Uint32(32)) * 4, Uint32)[0])
                        if lane < Uint32(self.hidden // 128):
                            sf0 = Uint32(cell(remote, Uint64(self.off_x_sf) + (Uint64(token) * (self.hidden // 128) + lane) * 4, Uint32)[0])
                        if pending_valid == 1:
                            self._pull_drain(
                                pending_dst, lane, warp, pull, pull_bar,
                                pending_k & Uint32(1),
                                (pending_k // Uint32(2)) & Uint32(1))
                        if cutlass.const_expr(self.hidden // 128 > 32):
                            if lane < Uint32(self.hidden // 128 - 32):
                                sf_out = cell(base, Uint64(self.off_l1_acts_sf) + ((Uint64(expert) * (self.hidden // 128) + lane + Uint32(32)) * self.rows + Uint32(pull_row)) * 4, Uint32)
                                sf_out[0] = sf1
                        if lane < Uint32(self.hidden // 128):
                            sf_out = cell(base, Uint64(self.off_l1_acts_sf) + ((Uint64(expert) * (self.hidden // 128) + lane) * self.rows + Uint32(pull_row)) * 4, Uint32)
                            sf_out[0] = sf0
                        pending_dst = dst
                        pending_k = row_k
                        pending_valid = Uint32(1)
                        row_k = row_k + Uint32(1)
                    else:
                        self._pull_row(src, dst, lane)
                        for sf_k in cutlass.range(lane, self.hidden // 128, 32):
                            sf_value = cell(remote, Uint64(self.off_x_sf) + (Uint64(token) * (self.hidden // 128) + Uint32(sf_k)) * 4, Uint32)[0]
                            sf_out = cell(base, Uint64(self.off_l1_acts_sf) + ((Uint64(expert) * (self.hidden // 128) + Uint32(sf_k)) * self.rows + Uint32(pull_row)) * 4, Uint32)
                            sf_out[0] = sf_value
                if cutlass.const_expr(self.pull_tma):
                    if pending_valid == 1:
                        self._pull_drain(
                            pending_dst, lane, warp, pull, pull_bar,
                            pending_k & Uint32(1),
                            (pending_k // Uint32(2)) & Uint32(1))
                    pending_valid = Uint32(0)
                fence_gpu()
                cute.arch.sync_threads()
                if tid == 0:
                    count_out = cell(base, Uint64(self.off_expert_count) + Uint64(expert) * 4, Uint32)
                    count_out[0] = total
                    if cutlass.const_expr(self.has_stats):
                        recv_stats = cute.make_tensor(stats, cute.make_layout((self.local,)))
                        recv_stats[expert] += total
                    st_release_u32(self._field(base, 'expert_ready', Uint32) + Uint32(expert), epoch, False)
                    if total > 0:
                        active_slot = atom_add_u32(head + 7, Uint32(1))
                        active_out = cell(base, Uint64(self.off_active_expert) + Uint64(active_slot) * 4, Uint32)
                        active_out[0] = Uint32(expert)
                        st_release_u32(self._field(base, 'active_ready', Uint32) + active_slot, epoch, False)
                    finished = atom_acq_rel_add_u32(head + 8, Uint32(1))
                    if finished == self.local - 1:
                        st_release_u32(head + 9, epoch, False)
                cute.arch.sync_threads()

    @cute.jit
    def _dispatch_direct(self, base: Uint64, peers: cute.Pointer, rank: Uint32,
                         tokens: Uint32, stats: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        expert, _, _ = cute.arch.block_idx()
        peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        head = self._field(base, 'header', Uint32)
        epoch = ld_acquire_gpu_u32(head)
        if expert == 0 and tid == 0:
            scan_out1 = cell(base, Uint64(8), Uint32)
            scan_out1[0] = tokens
            st_release_u32(head + 1, epoch, True)
        alloc = cutlass.utils.SmemAllocator()
        counts = alloc.allocate_tensor(Uint32, cute.make_layout((self.ranks,)), 4)
        prefix = alloc.allocate_tensor(Uint32, cute.make_layout((self.ranks + 1,)), 4)
        route_count = alloc.allocate_tensor(Uint32, cute.make_layout((1,)), 4)
        # Cross-rank rendezvous: one block polls the eight peers over PCIe and
        # publishes a local gate plus the staged route counts. Letting every
        # block re-poll all remotes keeps hundreds of sys-scope acquire loads
        # in flight per rank, and that traffic delays the very peer flag
        # writes the pollers are waiting on.
        if expert == 0:
            if tid < self.ranks:
                remote = Uint64(peer_table[tid])
                wait_epoch(self._field(remote, 'header', Uint32) + 1, epoch, True)
                counts[tid] = cell(remote, Uint64(8), Uint32)[0] * self.topk
                staged_count = cell(base, Uint64(self.off_counts) + Uint64(tid) * 4, Uint32)
                staged_count[0] = counts[tid]
            fence_gpu()
            cute.arch.sync_threads()
            if tid == 0:
                st_release_u32(head + 15, epoch, False)
        else:
            if tid == 0:
                wait_epoch(head + 15, epoch, False)
            cute.arch.sync_threads()
            if tid < self.ranks:
                staged_count = cell(base, Uint64(self.off_counts) + Uint64(tid) * 4, Uint32)
                counts[tid] = staged_count[0]
        cute.arch.sync_threads()
        if tid == 0:
            running_count = Uint32(0)
            prefix[0] = Uint32(0)
            for prefix_source in cutlass.range_constexpr(self.ranks):
                running_count += counts[prefix_source]
                prefix[prefix_source + 1] = running_count
        cute.arch.sync_threads()
        # The gate depends on block 0. Bound the grid to resident CTAs and
        # let each CTA publish the remaining experts assigned to it.
        for local_expert in cutlass.range(expert, self.local, self.grid):
            self._dispatch_direct_expert(base, peer_table, head, epoch, rank,
                                         Uint32(local_expert), prefix, route_count, stats)

    @cute.jit
    def _dispatch_direct_expert(self, base, peer_table, head, epoch, rank,
                                expert, prefix, route_count, stats):
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        if tid == 0:
            route_count[0] = Uint32(0)
        cute.arch.sync_threads()
        meta_base = Uint64(self.off_route_meta) + Uint64(expert) * self.rows * 16
        weight_base = Uint64(self.off_route_weight) + Uint64(expert) * self.rows * 4
        for candidate_route in cutlass.range(tid, prefix[self.ranks], self.threads):
            source = Uint32(0)
            for candidate in cutlass.range_constexpr(1, self.ranks):
                if candidate_route >= prefix[candidate]:
                    source = Uint32(candidate)
            source_route = Uint32(candidate_route) - prefix[source]
            remote = Uint64(peer_table[source])
            expert_id = cell(remote, Uint64(self.off_topk_idx) + Uint64(source_route) * 8, Int64)[0]
            if expert_id == Int64(rank * self.local + Uint32(expert)):
                row = atom_shared_add_u32(route_count.iterator, Uint32(1))
                scan_out3 = cell(base, meta_base + Uint64(row * 4) * 4, Uint32)
                scan_out3[0] = source
                scan_out4 = cell(base, meta_base + Uint64(row * 4 + 1) * 4, Uint32)
                scan_out4[0] = source_route // self.topk
                scan_out5 = cell(base, meta_base + Uint64(row * 4 + 2) * 4, Uint32)
                scan_out5[0] = source_route % self.topk
                # Direct mode (T<=2): no source-side scan ran, so the arena
                # position falls back to the route index itself. The combine
                # traffic at this tier is tiny; contiguity is irrelevant.
                scan_out5b = cell(base, meta_base + Uint64(row * 4 + 3) * 4, Uint32)
                scan_out5b[0] = source_route
                scan_out6 = cell(base, weight_base + Uint64(row) * 4, Float32)
                scan_out6[0] = cell(remote, Uint64(self.off_topk_weights) + Uint64(source_route) * 4, Float32)[0]
        cute.arch.sync_threads()
        total = route_count[0]
        # A warp retains one source/token pair while copying an entire
        # route, avoiding metadata reloads for every sixteen data bytes.
        for pull_row in cutlass.range(warp, total, self.threads // 32):
            source_rank = cell(base, meta_base + Uint64(pull_row * 4) * 4, Uint32)[0]
            token = cell(base, meta_base + Uint64(pull_row * 4 + 1) * 4, Uint32)[0]
            remote = Uint64(peer_table[source_rank])
            src = remote + Uint64(self.off_x) + Uint64(token) * self.hidden
            dst = base + Uint64(self.off_l1_acts) + (Uint64(expert) * self.rows + Uint32(pull_row)) * self.hidden
            self._pull_row(src, dst, lane)
            # Packed-UE8M0 activation scale words pass through verbatim.
            for sf_k in cutlass.range(lane, self.hidden // 128, 32):
                sf_value = cell(remote, Uint64(self.off_x_sf) + (Uint64(token) * (self.hidden // 128) + Uint32(sf_k)) * 4, Uint32)[0]
                sf_out = cell(base, Uint64(self.off_l1_acts_sf) + ((Uint64(expert) * (self.hidden // 128) + Uint32(sf_k)) * self.rows + Uint32(pull_row)) * 4, Uint32)
                sf_out[0] = sf_value
        fence_gpu()
        cute.arch.sync_threads()
        if tid == 0:
            count_out = cell(base, Uint64(self.off_expert_count) + Uint64(expert) * 4, Uint32)
            count_out[0] = total
            if cutlass.const_expr(self.has_stats):
                recv_stats = cute.make_tensor(stats, cute.make_layout((self.local,)))
                recv_stats[expert] += total
            st_release_u32(self._field(base, 'expert_ready', Uint32) + Uint32(expert), epoch, False)
            if total > 0:
                active_slot = atom_add_u32(head + 7, Uint32(1))
                active_out = cell(base, Uint64(self.off_active_expert) + Uint64(active_slot) * 4, Uint32)
                active_out[0] = Uint32(expert)
                st_release_u32(self._field(base, 'active_ready', Uint32) + active_slot, epoch, False)
            finished = atom_acq_rel_add_u32(head + 8, Uint32(1))
            if finished == self.local - 1:
                st_release_u32(head + 9, epoch, False)
        cute.arch.sync_threads()


class SM120GCMoERoleKernel(SM120Dispatch):
    def __init__(self, config, layout, role, schedule, has_stats, *,
                 launch_dependents=False, shared_layout=None):
        self.role = role
        self.launch_dependents = launch_dependents
        self.batched_returns = schedule.batched_returns
        self.relay = schedule.relay_ctas > 0
        self.relay_ctas = schedule.relay_ctas if role == 2 else 0
        self.direct_route_scan = 0 < config.bound <= 2
        self.ranks, self.local = config.ranks, config.experts // config.ranks
        self.capacity, self.topk = config.capacity, config.topk
        self.hidden, self.intermediate = config.hidden, config.intermediate
        self.rows = config.ranks * config.capacity
        self.fast_math, self.has_stats = config.fast_math, has_stats
        for name, spec in layout.views.items():
            setattr(self, 'off_' + name, spec.offset)
        self.grid, self.threads = schedule.grids[role - 1], schedule.threads[role - 1]
        self.smem = schedule.smem[role - 1]
        self.reg_cap = schedule.reg_cap if role == 2 else 0
        self.shared_experts = config.shared_experts if role == 3 else 0
        self.shared_fc1_flags = (layout.views['shared_fc1_done'].shape[0]
                                 if self.shared_experts else 0)
        self.shared_tile_n = schedule.shared_tile_n if role == 2 else 0
        self.shared_layout = shared_layout if self.shared_tile_n else None
        self.shared_intermediate = config.shared_experts * config.intermediate
        self.tokens = config.tokens if self.shared_tile_n else 0
        self._l2_dedicated = schedule.l2_dedicated if role == 2 else 1
        self.tile_n = tile_n = schedule.tile_n
        self.mma_stages = schedule.mma_stages
        self.gmem_handoff = schedule.gmem_handoff
        self.compact_w = schedule.compact_w
        self.shared_chunks = schedule.shared_chunks
        self.bf16_utility = schedule.bf16_handoff
        self.handoff_rows = 32 if self.shared_chunks else tile_n
        self.bf16_bulk = self.bf16_utility and (self.shared_chunks or not self.gmem_handoff)
        self.combine_slots = schedule.combine_slots
        self.handoff_stride = 32768 if tile_n > 64 else 16384
        if self.gmem_handoff and not self.shared_chunks and role == 2:
            if layout.views['handoff'].nbytes < self.grid * self.handoff_stride:
                raise ValueError('workspace is too small for the global handoff grid')
        self._configure_dispatch(schedule)
        self.mma_tile = (MMA_TILE_M, tile_n, MMA_TILE_K)
        self.a_stage_bytes = MMA_TILE_M * MMA_TILE_K // (2 if self.compact_w else 1)
        self.b_stage_bytes = tile_n * 128
        self.n_groups = tile_n // MMA_N
        self.epilogue_pairs = tile_n // 16
        # Warp-specialized pipeline barriers: `full[s]` completes when a
        # stage's TMA + cp.async scale staging lands (the issuing leader's
        # expect-tx arrival plus one cp.async-tracked arrival per producer
        # lane); `empty[s]` completes when each of the four consumer
        # warps retires the stage. Phase 0 of `empty` is pre-completed at
        # init so the producer's first round issues without waiting.
        self.full_arrivals = 1 + 32
        # Each FC1/FC2 task covers one 64-weight-row tile.
        self.l1_ntiles = 2 * self.intermediate // 64
        self.l2_ntiles = self.hidden // 64
        # The combine arena is tiled into 64-position by 64-column BF16 cells.
        self.colblocks = self.hidden // 64
        self.cleanup_grids = min(self.grid, (self.local + 127) // 128)
        self.finalize_flags = layout.views['combine_block_done'].shape[0]
        if role == 3 and self.grid > self.finalize_flags:
            raise ValueError('combine grid exceeds the SM120 completion-flag capacity')

    @cute.jit
    def _routed_swiglu(self, gate: Float32, up: Float32, clamp: Float32):
        # Match MegaMoE: round the FC1 outputs and clamp bound to BF16.
        bound = round_bf16_f32(clamp)
        gate = cute.math.min(round_bf16_f32(gate), bound)
        up = cute.math.min(cute.math.max(round_bf16_f32(up), -bound), bound)
        if cutlass.const_expr(self.fast_math):
            # Cake uses approximate exp2 followed by a rounded FP32 reciprocal.
            denom = Float32(1) + fast_exp_neg_f32(gate)
            sigmoid = cute.math.div(Float32(1), denom, fastmath=False, ftz=False,
                                    rounding=cute.math.RoundingMode.NEAREST_EVEN)
            gate = gate * sigmoid
        else:
            gate = cute.math.div(gate, Float32(1) + cute.math.exp(-gate, fastmath=False),
                                 fastmath=False, ftz=False,
                                 rounding=cute.math.RoundingMode.NEAREST_EVEN)
        return gate * up

    @cute.jit
    def _claim_command(self, base: Uint64, command: cute.Tensor,
                       epoch: Uint32, is_l1: Boolean):
        """Claim one (expert, ntile) task; command = [valid, expert, ntile, switch].

        command[3] marks an L1 ticket claimed inside the low-watermark tail of
        the L1 queue: the CTA retires that task and migrates to L2 claims, so
        combine pushes begin overlapping the L1 tail instead of bunching into
        a PCIe-saturating L2 wave.
        """
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        head = self._field(base, 'header', Uint32)
        active_ptr = self._field(base, 'active_ready', Uint32)
        expert_ptr = self._field(base, 'expert_ready', Uint32)
        l1_ptr = self._field(base, 'l1_done', Uint32)
        if tid == 0:
            task, active_slot, ntile = Uint32(0), Uint32(0), Uint32(0)
            switch = Uint32(0)
            if is_l1:
                task = atom_add_u32(head + 3, Uint32(1))
                active_slot, ntile = task // Uint32(self.l1_ntiles), task % Uint32(self.l1_ntiles)
            else:
                task = atom_add_u32(head + 10, Uint32(1))
                active_slot, ntile = task // Uint32(self.l2_ntiles), task % Uint32(self.l2_ntiles)
            active_count = ld_acquire_gpu_u32(head + 7)
            if active_slot >= active_count:
                # A ticket outside the provisional queue can still name work.
                # Load the count after acquiring completion before retiring.
                dispatch_done = ld_acquire_gpu_u32(head + 9)
                active_count = ld_acquire_gpu_u32(head + 7)
                while active_slot >= active_count and dispatch_done != epoch:
                    dispatch_done = ld_acquire_gpu_u32(head + 9)
                    active_count = ld_acquire_gpu_u32(head + 7)
            if cutlass.const_expr(self._l2_dedicated <= 1 and self.grid > 1):
                # Only a subset can migrate on the provisional watermark.
                # The remaining resident CTAs drain L1 as dispatch appends
                # experts, even if the early migrators are waiting on L2.
                if is_l1 and active_slot < active_count and block % 3 == 0:
                    total = active_count * Uint32(self.l1_ntiles)
                    if total > Uint32(64) and total - task <= total // Uint32(2):
                        switch = Uint32(1)
            command[0] = Uint32(0)
            if active_slot < active_count:
                wait_epoch(active_ptr + active_slot, epoch, False)
                expert = cell(base, Uint64(self.off_active_expert) + Uint64(active_slot) * 4, Uint32)[0]
                wait_epoch(expert_ptr + expert, epoch, False)
                if not is_l1:
                    wait_count(l1_ptr + expert, Uint32(self.l1_ntiles))
                command[0] = Uint32(1)
                command[1] = Uint32(expert)
                command[2] = Uint32(ntile)
            command[3] = switch

    @cute.jit
    def _producer_w_stage(self, base: Uint64, w1_map: cute.Pointer,
                          w2_map: cute.Pointer, w_smem: cute.Pointer,
                          sfa_smem: cute.Pointer, full_bar: cute.Pointer,
                          empty_bar: cute.Pointer, expert: Uint32,
                          ntile: Uint32, kt: Uint32, load_idx: Uint32,
                          kt_count: Uint32, n_tiles: Uint32, is_l1, w_sf_base: Uint64):
        """Producer-warp issue of one K128 W stage: drain `empty`, TMA W, SFA.

        Runs on the dedicated TMA warp (all 32 lanes; lane 0 is the leader).
        ``load_idx`` is the GLOBAL W issue index: it selects the decoupled
        W-ring slot and both barrier phases, so tasks whose kt_count is not
        a multiple of the ring depth stay aligned. The empty barrier's
        phase 0 is pre-completed at init, so the first stages issue without
        waiting; every later round blocks until each consumer warp has
        retired the slot. The leader arrives with the W transaction bytes;
        the other lanes stage the packed SFA words (each mma group's rows)
        and arrive through cp.async tracking.
        """
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        stage = load_idx % Uint32(self.mma_stages)
        w_off = Uint32(stage) * Uint32(self.a_stage_bytes)
        sf_off = Uint32(stage) * Uint32(MMA_SF_STAGE_BYTES)
        stage_full = full_bar + stage
        cute.arch.mbarrier_wait(empty_bar + stage,
                                (load_idx // Uint32(self.mma_stages)) & Uint32(1))
        if lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(stage_full, Uint32(4096))
            w_dst = w_smem + w_off
            if is_l1:
                tile_g2s_4d(w_dst, w1_map, stage_full, Uint32(0), ntile * 64,
                            kt, expert, Uint64(EVICT_FIRST))
            else:
                tile_g2s_4d(w_dst, w2_map, stage_full, Uint32(0), ntile * 64,
                            kt, expert, Uint64(EVICT_FIRST))
            if cutlass.const_expr(self.mma_stages >= 4):
                dist = Uint32(self.mma_stages)
                if kt + dist < kt_count:
                    if is_l1:
                        prefetch_g2s_4d(w1_map, Uint32(0), ntile * 64, kt + dist, expert)
                    else:
                        prefetch_g2s_4d(w2_map, Uint32(0), ntile * 64, kt + dist, expert)
        for m in cutlass.range_constexpr(2):
            row = lane + Uint32(32 * m)
            sfa_dst = sfa_smem + sf_off + Uint32((row % 32) * 16 + (row // 32) * 4)
            permuted = (ntile // 2) * 128 + (row % 32) * 4 + (ntile % 2) * 2 + row // 32
            sf_word = (Uint64(expert) * kt_count + Uint64(kt)) * n_tiles * 64 + Uint64(permuted)
            w_sf_address = w_sf_base + sf_word * 4
            copy_async_4(sfa_dst, cute.make_ptr(
                Uint8, w_sf_address, cute.AddressSpace.gmem, assumed_align=4), Uint32(4))
        # cp.async.mbarrier.arrive tracks completion: no wait_group needed,
        # the full barrier only completes once the SF words land.
        copy_commit()
        copy_arrive(stage_full)

    @cute.jit
    def _producer_x_stage(self, base: Uint64, x1_map: cute.Pointer,
                          x2_map: cute.Pointer, x_smem: cute.Pointer,
                          sfb_smem: cute.Pointer, full_bar: cute.Pointer,
                          empty_bar: cute.Pointer, expert: Uint32,
                          start: Uint32, kt: Uint32, load_idx: Uint32,
                          kt_count: Uint32, valid: Uint32, is_l1,
                          sf_offset: Uint64):
        """Producer-warp issue of one K128 X stage: drain `empty`, TMA X, SFB.

        The X ring keeps the mma_stages depth: X is small and L2-resident
        (EVICT_LAST, re-read across the expert's tasks), so its TMA latency
        hides under the consumers' mma time at two slots. ``load_idx`` is
        the GLOBAL X issue index, mirroring _producer_w_stage.
        """
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        stage = load_idx % Uint32(self.mma_stages)
        x_off = Uint32(stage) * Uint32(self.b_stage_bytes)
        sf_off = Uint32(stage) * Uint32(MMA_SF_STAGE_BYTES)
        stage_full = full_bar + stage
        cute.arch.mbarrier_wait(empty_bar + stage,
                                (load_idx // Uint32(self.mma_stages)) & Uint32(1))
        if lane == 0:
            tx = Uint32(0)
            tx = tx + Uint32(self.tile_n * 128)
            cute.arch.mbarrier_arrive_and_expect_tx(stage_full, tx)
            if is_l1:
                tile_g2s_2d(x_smem + x_off, x1_map, stage_full, kt * Uint32(128),
                            expert * Uint32(self.rows) + start, Uint64(EVICT_LAST))
            else:
                tile_g2s_2d(x_smem + x_off, x2_map, stage_full, kt * Uint32(128),
                            expert * Uint32(self.rows) + start, Uint64(EVICT_LAST))
        for m in cutlass.range_constexpr(3 if self.tile_n > 64 else 2):
            row = lane + Uint32(32 * m)
            sfb_dst = sfb_smem + sf_off + Uint32((row % 32) * 16 + (row // 32) * 4)
            x_sf_size = Uint32(0)
            if Uint32(row) < valid:
                x_sf_size = Uint32(4)
            x_sf_address = base + sf_offset + ((Uint64(expert) * kt_count + Uint64(kt)) * self.rows + Uint64(start) + Uint64(row)) * 4
            copy_async_4(sfb_dst, cute.make_ptr(
                Uint8, x_sf_address, cute.AddressSpace.gmem, assumed_align=4), x_sf_size)
        copy_commit()
        copy_arrive(stage_full)

    @cute.jit
    def _consumer_k_loop(self, n_groups: cutlass.Constexpr,
                         read_total, kt_count, lane,
                         tiled_mma, accs, sSFB, operands):
        """Consume a whole K loop with a fixed number of live N8 groups."""
        (tCsA_copy_view, tCsB_copy_view, tCsSFA_copy_view,
         smem_tiled_copy_A, smem_tiled_copy_B,
         tCrA_copy_view, tCrB_copy_view, tCrA, tCrB,
         full_w_bar, full_x_bar, empty_w_bar, empty_x_bar, packed_a) = operands
        rSFB = [Int32(0) for _ in range(n_groups)]
        for kt in cutlass.range(0, kt_count):
            # Both rings derive their slot and parity from the global read index.
            x_stage = read_total % Uint32(self.mma_stages)
            x_phase = (read_total // Uint32(self.mma_stages)) & Uint32(1)
            cute.arch.mbarrier_wait(full_x_bar + x_stage, x_phase)
            w_stage = read_total % Uint32(self.mma_stages)
            w_phase = (read_total // Uint32(self.mma_stages)) & Uint32(1)
            cute.arch.mbarrier_wait(full_w_bar + w_stage, w_phase)
            read_total = read_total + Uint32(1)
            packed_stage = cute.make_tensor(
                packed_a.iterator + w_stage * Uint32(self.a_stage_bytes),
                cute.make_layout((64, 64), stride=(64, 1)))
            tCsA_p = tCsA_copy_view[None, None, None, w_stage]
            tCsB_p = tCsB_copy_view[None, None, None, x_stage]
            sfa_src = tCsSFA_copy_view[None, None, None, w_stage]
            # First k-tile copies issue before the k_inner loop;
            # the loop only prefetches k_inner + 1.
            # One u32 per (n-group, K128): the staged SFB
            # layout packs this lane's four K32 scale bytes
            # into a single word, so the QMMA's byte-id
            # selector replaces the scalar SFB fragment
            # copies and their per-mma byte extraction.
            # Row mapping matches the producer's staging:
            # n-group ng covers token rows ng*8..ng*8+7,
            # lane//4 picks the lane's row inside the N8.
            for ng in cutlass.range_constexpr(n_groups):
                row = Uint32(ng * MMA_N) + lane // Uint32(4)
                rSFB[ng] = lds_b32_raw_ordered(
                    sSFB.iterator + Uint32(x_stage) * Uint32(MMA_SF_STAGE_BYTES)
                    + Uint32((row % Uint32(32)) * 16
                             + (row // Uint32(32)) * 4))
            # Packed SFA: the staged layout packs this
            # lane's four K32 scale bytes into one word
            # starting exactly at the copy view's base
            # (k_inner stride is 1 byte, stage and
            # m-group shifts are already folded into
            # sfa_src.iterator), so one u32 load plus
            # the QMMA's byte_id_a selector replaces the
            # four SFA fragment copies -- the
            # cvt/prmt/mov chains that dominate the
            # inner-loop instruction count.
            rSFA = lds_b32_raw_ordered(sfa_src.iterator)
            if cutlass.const_expr(self.compact_w):
                load_packed_fp4_fragment(tCrA[None, None, 0], packed_stage, 0)
            else:
                cute.copy(smem_tiled_copy_A, tCsA_p[None, None, 0],
                          tCrA_copy_view[None, None, 0])
                shift_fp4_fragment_for_mxf8f6f4(tCrA[None, None, 0])
            cute.copy(smem_tiled_copy_B, tCsB_p[None, None, 0],
                      tCrB_copy_view[None, None, 0])
            for k_inner in cutlass.range_constexpr(4):
                if k_inner + 1 < 4:
                    if cutlass.const_expr(self.compact_w):
                        load_packed_fp4_fragment(tCrA[None, None, k_inner + 1],
                                                 packed_stage, k_inner + 1)
                    else:
                        cute.copy(smem_tiled_copy_A,
                                  tCsA_p[None, None, k_inner + 1],
                                  tCrA_copy_view[None, None, k_inner + 1])
                        shift_fp4_fragment_for_mxf8f6f4(tCrA[None, None, k_inner + 1])
                    cute.copy(smem_tiled_copy_B,
                              tCsB_p[None, None, k_inner + 1],
                              tCrB_copy_view[None, None, k_inner + 1])
                for ng in cutlass.range_constexpr(n_groups):
                    issue_m64n8k32_mxfp8_packed_sfb(
                        tiled_mma, accs[ng], tCrA, tCrB, rSFA, rSFB[ng],
                        n_group=ng, active_n_groups=n_groups,
                        k_inner=k_inner, a_dtype=Float4E2M1FN,
                        b_dtype=Float8E4M3FN, sf_dtype=Float8E8M0FNU)
            # Retire this stage in both rings so the
            # producer's next round may reissue the slots
            # (one arrival per consumer warp per ring).
            cute.arch.sync_warp()
            if cutlass.const_expr(self.compact_w):
                cute.arch.fence_view_async_shared()
            if lane == 0:
                cute.arch.mbarrier_arrive(empty_w_bar + w_stage)
                cute.arch.mbarrier_arrive(empty_x_bar + x_stage)
        return read_total

    @cute.jit
    def _compute_mma(self, base: Uint64, peers: cute.Pointer, w1: cute.Pointer,
                     s1: cute.Pointer, w2: cute.Pointer, s2: cute.Pointer,
                     w1_map: cute.Pointer, x1_map: cute.Pointer,
                     w2_map: cute.Pointer, x2_map: cute.Pointer, clamp: Float32,
                     tiled_mma: cute.TiledMma, a_smem_layout: cute.ComposedLayout,
                     b_smem_layout: cute.ComposedLayout,
                     sfa_smem_layout: cute.Layout, sfb_smem_layout: cute.Layout):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = tid % Uint32(32)
        alloc = cutlass.utils.SmemAllocator()
        storage = alloc.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout)
        command = alloc.allocate_tensor(Uint32, cute.make_layout((5,)), 4)
        # Separate W and X pipeline barriers: `fullW[s]` completes when the W TMA plus the SFA staging for
        # W-ring slot s lands, `fullX[s]` when the X TMA plus the SFB staging
        # for X-ring slot s lands; `emptyW/emptyX[s]` complete when every
        # consumer warp retires the slot. Phases derive from the global
        # read/issue counters, so tasks never re-initialize barriers. Empty
        # phase 0 is pre-completed below so the producer's first round
        # issues without waiting.
        bar = cute.recast_ptr(
            alloc.allocate((8 * (2 * self.mma_stages + 2 * self.mma_stages
                                  + 2 + 4 * self.combine_slots)
                            if self.gmem_handoff else
                            16 * (2 * self.mma_stages + 2 * self.mma_stages
                                  + 3 + 3 * self.combine_slots)), 8),
            dtype=Uint64)
        full_w_bar = bar
        full_x_bar = bar + Int32(self.mma_stages)
        empty_w_bar = full_x_bar + Int32(self.mma_stages)
        empty_x_bar = empty_w_bar + Int32(self.mma_stages)
        # Free, full, and empty each have one barrier per slot. With two
        # slots the phase flips every second round for the selected slot,
        # allowing the math and utility warps to work on different tiles.
        free_bar = empty_x_bar + Int32(self.mma_stages + 2)
        hand_full = free_bar + Int32(self.combine_slots)
        hand_empty = free_bar + Int32(2 * self.combine_slots)
        combine_stage = alloc.allocate(
            16 if self.gmem_handoff and not self.shared_chunks
            else 2 * self.combine_slots * self.handoff_rows * 64, 16)
        stage16 = cute.make_tensor(
            cute.recast_ptr(combine_stage, dtype=Uint16),
            cute.make_layout((self.combine_slots * self.handoff_rows * 64,)))
        stage32 = cute.make_tensor(
            cute.recast_ptr(combine_stage, dtype=Uint32),
            cute.make_layout((self.combine_slots * self.handoff_rows * 32,)))
        if cutlass.const_expr(self.gmem_handoff and not self.shared_chunks):
            hand_base = base + Uint64(self.off_handoff) + Uint64(block) * self.handoff_stride
            stage16 = cute.make_tensor(cute.make_ptr(
                Uint16, hand_base, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((self.combine_slots * self.tile_n * 64,)))
            stage32 = cute.make_tensor(cute.recast_ptr(stage16.iterator, dtype=Uint32),
                cute.make_layout((self.combine_slots * self.handoff_rows * 32,)))
        # L1 handoff view over the same 8KB: tile_n tokens x 32
        # swiglu'd f32 features, plus a five-word header (expert,
        # start, ntile, valid, task-last) the utility warp reads after
        # the full barrier. valid == 0 doubles as the walk terminator.
        hand32 = cute.make_tensor(
            cute.recast_ptr(stage32.iterator, dtype=Float32),
            cute.make_layout((self.combine_slots * self.handoff_rows * 32,)))
        hand_meta = alloc.allocate_tensor(
            Uint32, cute.make_layout((6 * self.combine_slots,)), 4)
        # Prefetch metadata during MMA. The 2*slots ring keeps upcoming
        # tiles from overwriting metadata while the utility warp reads it.
        pf_slots = 2 * self.combine_slots
        src_pf = cute.make_tensor(
            cute.recast_ptr(alloc.allocate(4 * pf_slots * self.tile_n, 4),
                            dtype=Uint32),
            cute.make_layout((pf_slots * self.tile_n,)))
        pos_pf = cute.make_tensor(
            cute.recast_ptr(alloc.allocate(4 * pf_slots * self.tile_n, 4),
                            dtype=Uint32),
            cute.make_layout((pf_slots * self.tile_n,)))
        return_dests = command
        if cutlass.const_expr(self.batched_returns and not self.relay):
            return_dests = alloc.allocate_tensor(Uint64, cute.make_layout((self.handoff_rows,)), 8)
        allocated_smem_bytes = alloc.calculate_total_usage()
        if tid == 0:
            if cutlass.const_expr(self.gmem_handoff):
                check_smem_budget(Uint32(allocated_smem_bytes), Uint32(self.smem))
            for stage in cutlass.range_constexpr(self.mma_stages):
                cute.arch.mbarrier_init(full_w_bar + Int32(stage), self.full_arrivals)
                cute.arch.mbarrier_init(empty_w_bar + Int32(stage), 4)
            for stage in cutlass.range_constexpr(self.mma_stages):
                cute.arch.mbarrier_init(full_x_bar + Int32(stage), self.full_arrivals)
                cute.arch.mbarrier_init(empty_x_bar + Int32(stage), 4)
            # The utility warp advances each slot's free barrier for
            # both L1 requantization and L2 returns.
            for z in cutlass.range_constexpr(self.combine_slots):
                cute.arch.mbarrier_init(free_bar + Int32(z), 1)
            for z in cutlass.range_constexpr(self.combine_slots):
                cute.arch.mbarrier_init(hand_full + Int32(z),
                                        4)
                cute.arch.mbarrier_init(hand_empty + Int32(z),
                                        1)
            prefetch_tensormap(w1_map)
            prefetch_tensormap(x1_map)
            prefetch_tensormap(w2_map)
            prefetch_tensormap(x2_map)
        cute.arch.mbarrier_init_fence()
        # Pre-complete every empty barrier's phase 0: threads 0..3
        # each arrive once, matching the consumer-warp retirement count.
        if tid < Uint32(4):
            for stage in cutlass.range_constexpr(self.mma_stages):
                cute.arch.mbarrier_arrive(empty_w_bar + Int32(stage))
            for stage in cutlass.range_constexpr(self.mma_stages):
                cute.arch.mbarrier_arrive(empty_x_bar + Int32(stage))
        # Pre-complete each slot's handoff-empty phase 0: the first
        # dump on every slot issues without waiting (thread z arrives
        # slot z's barrier).
        if tid < Uint32(self.combine_slots):
            cute.arch.mbarrier_arrive(hand_empty + Int32(tid))
        cute.arch.sync_threads()
        head = self._field(base, 'header', Uint32)
        epoch = ld_acquire_gpu_u32(head)

        # Warps 0–3 consume MMA stages, warp 4 produces TMA stages,
        # and warp 5 handles requantization and returns.
        producer = (tid - Uint32(128)) < Uint32(32)
        utility = tid >= Uint32(128 + 32)
        lane = tid % 32
        mma_tidx = tid % 128
        thr_mma = tiled_mma.get_slice(mma_tidx)
        fragment_a = sA
        if cutlass.const_expr(self.compact_w):
            # Only the nominal expanded shape is needed to allocate the
            # MMA registers. Compact loads use the real sA storage below.
            nominal = make_data_smem_layout(self.mma_tile, dtype=Int8, is_a=True,
                                            num_stages=self.mma_stages)
            fragment_a = cute.make_tensor(sA.iterator, nominal.outer)
        tCsA = thr_mma.partition_A(fragment_a)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        atom_copy_ldmatrix_A = make_sm120_ldmatrix_atom(
            Float4E2M1FN, transpose=False, mixed_mode=True)
        atom_copy_ldmatrix_B = make_sm120_ldmatrix_atom(
            Float8E4M3FN, transpose=False, mixed_mode=False)
        smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)
        thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(mma_tidx)
        thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(mma_tidx)
        tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(fragment_a)
        tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
        tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
        tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

        atom_copy_scale = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float8E8M0FNU)
        smem_tiled_copy_SFA = cute.make_tiled_copy(
            atom_copy_scale,
            sm100_utils.get_layoutSFA_TV(tiled_mma),
            (cute.size(tiled_mma.permutation_mnk[0]),
             cute.size(tiled_mma.permutation_mnk[2])))
        tCsSFA_copy_view = smem_tiled_copy_SFA.get_slice(mma_tidx).partition_S(sSFA)

        n_groups = self.n_groups
        accs = [cute.make_rmem_tensor(
                    tiled_mma.partition_shape_C((MMA_TILE_M, MMA_N)), Float32)
                for _ in range(n_groups)]
        consumer_operands = (
            tCsA_copy_view, tCsB_copy_view, tCsSFA_copy_view,
            smem_tiled_copy_A, smem_tiled_copy_B,
            tCrA_copy_view, tCrB_copy_view, tCrA, tCrB,
            full_w_bar, full_x_bar, empty_w_bar, empty_x_bar, sA)

        w_smem = cute.recast_ptr(sA.iterator, dtype=Uint8)
        x_smem = cute.recast_ptr(sB.iterator, dtype=Uint8)
        sfa_smem = cute.recast_ptr(sSFA.iterator, dtype=Uint8)
        sfb_smem = cute.recast_ptr(sSFB.iterator, dtype=Uint8)

        # Dispatch publishes only nonempty experts. Exhaust L1 claims before
        # any L2 wait: every dependency is in flight, except for the
        # dedicated-L2 CTAs (their claim spin handles the growing queue).
        is_l1 = True
        if cutlass.const_expr(self._l2_dedicated > 1):
            if block % self._l2_dedicated == 0:
                is_l1 = False
        running = True
        # Global K-tile indices: the consumer's read and the producer's
        # issue counters advance once per k-tile over identical loop trips,
        # so every barrier's phase parity follows from them and never drifts
        # across tasks.
        read_total = Uint32(0)
        w_total = Uint32(0)
        tile_seq = Uint32(0)
        # Consumer-side handoff round counter (drives the empty parity and
        # the terminator round).
        hand_seq = Uint32(0)
        if utility:
            # Drain the handoff ring until the claim group posts its terminator.
            self._utility_walk_mma(base, peers, hand32, stage32, stage16,
                                   hand_meta, hand_full, hand_empty,
                                   free_bar, src_pf, pos_pf, return_dests)
            running = False
        while running:
            self._claim_command(base, command, epoch, is_l1)
            self._walk_sync()
            if command[0] == 0:
                if is_l1:
                    is_l1 = False
                else:
                    running = False
            else:
                expert = Uint32(command[1])
                ntile = Uint32(command[2])
                count = cell(base, Uint64(self.off_expert_count) + Uint64(expert) * 4, Uint32)[0]
                # Layer-selected operands (L2 defaults, L1 overrides).
                n_tiles = Uint32(self.l2_ntiles)
                kt_count = Uint32(self.intermediate // 128)
                w_sf_base = Uint64(s2.toint())
                sf_offset = Uint64(self.off_l2_acts_sf)
                if is_l1:
                    n_tiles = Uint32(self.l1_ntiles)
                    kt_count = Uint32(self.hidden // 128)
                    w_sf_base = Uint64(s1.toint())
                    sf_offset = Uint64(self.off_l1_acts_sf)
                if producer:
                    if cutlass.const_expr(self.compact_w):
                        # Claim-group synchronization imports the dispatch /
                        # L1 ready acquire; make those generic writes visible
                        # to the following TMA global reads as in Cake.
                        global_async_proxy_fence()
                for start in cutlass.range(0, count, self.tile_n):
                    valid = cute.math.min(Uint32(self.tile_n), count - Uint32(start))
                    if producer:
                        for kt in cutlass.range(0, kt_count):
                            self._producer_w_stage(base, w1_map, w2_map, w_smem,
                                                   sfa_smem, full_w_bar, empty_w_bar,
                                                   expert, ntile, Uint32(kt), w_total,
                                                   kt_count, n_tiles, is_l1, w_sf_base)
                            self._producer_x_stage(base, x1_map, x2_map, x_smem,
                                                   sfb_smem, full_x_bar, empty_x_bar,
                                                   expert, Uint32(start), Uint32(kt),
                                                   w_total, kt_count, valid, is_l1, sf_offset)
                            w_total = w_total + Uint32(1)
                    else:
                        # Overlap route metadata loads with MMA. The utility
                        # warp reads the prefetched positions during BF16 return.
                        if cutlass.const_expr(self.bf16_utility and not self.shared_chunks):
                            if tid < Uint32(self.tile_n) and Uint32(tid) < valid:
                                tile_token = (Uint64(expert) * self.rows
                                              + Uint32(start) + Uint64(tid))
                                meta = Uint64(self.off_route_meta) + tile_token * 16
                                pf = ((hand_seq
                                       & Uint32(2 * self.combine_slots - 1))
                                      * Uint32(self.tile_n)
                                      + Uint32(tid))
                                src_pf[pf] = cell(
                                    base, meta, Uint32)[0]
                                pos_pf[pf] = cell(
                                    base, meta + 12, Uint32)[0]
                        for ng in cutlass.range_constexpr(n_groups):
                            accs[ng].fill(Float32(0))
                        read_total = self._consumer_k_loop(
                            n_groups, read_total, kt_count, lane,
                            tiled_mma, accs, sSFB, consumer_operands)
                    if is_l1:
                        if tid < Uint32(128):
                            if cutlass.const_expr(self.shared_chunks):
                                for chunk in cutlass.range_constexpr((self.tile_n + 31) // 32):
                                    if Uint32(chunk * 32) < valid:
                                        chunk_valid = cute.math.min(
                                            Uint32(32), valid - Uint32(chunk * 32))
                                        self._l1_dump_mma(base, expert, ntile,
                                            Uint32(start) + Uint32(chunk * 32),
                                            chunk_valid, count, accs, hand32,
                                            hand_meta, hand_full, hand_empty,
                                            clamp, hand_seq, chunk)
                                        hand_seq = hand_seq + Uint32(1)
                            else:
                                self._l1_dump_mma(base, expert, ntile,
                                                  Uint32(start), valid, count,
                                                  accs, hand32, hand_meta,
                                                  hand_full, hand_empty,
                                                  clamp, hand_seq)
                                hand_seq = hand_seq + Uint32(1)
                    else:
                        if cutlass.const_expr(self.bf16_utility):
                            # Hand BF16 rows off to the utility warp for return.
                            if cutlass.const_expr(self.shared_chunks):
                                if tid < Uint32(128):
                                    for chunk in cutlass.range_constexpr((self.tile_n + 31) // 32):
                                        if Uint32(chunk * 32) < valid:
                                            chunk_start = Uint32(start) + Uint32(chunk * 32)
                                            chunk_valid = cute.math.min(
                                                Uint32(32), valid - Uint32(chunk * 32))
                                            if tid < chunk_valid:
                                                tile_token = Uint64(expert) * self.rows \
                                                    + Uint64(chunk_start) + Uint64(tid)
                                                meta = Uint64(self.off_route_meta) + tile_token * 16
                                                pf = (hand_seq & Uint32(2 * self.combine_slots - 1)) \
                                                    * Uint32(self.tile_n) + Uint32(tid)
                                                src_pf[pf] = cell(base, meta, Uint32)[0]
                                                pos_pf[pf] = cell(base, meta + 12, Uint32)[0]
                                            self._l2_dump_mma(base, expert, ntile,
                                                chunk_start, chunk_valid, accs, stage16,
                                                hand_meta, hand_full, hand_empty, free_bar,
                                                hand_seq, hand_seq, chunk)
                                            hand_seq = hand_seq + Uint32(1)
                            else:
                                self._l2_dump_mma(base, expert, ntile,
                                                 Uint32(start), valid, accs,
                                                 stage16, hand_meta,
                                                 hand_full, hand_empty,
                                                 free_bar,
                                                 tile_seq, hand_seq)
                                hand_seq = hand_seq + Uint32(1)
                        else:
                            self._push_bf16_tile(base, peers, expert, ntile, Uint32(start), valid, accs)
                    # Every start tile -- L1 or L2 -- advances the free_bar
                    # phase the next section-end wait keys on.
                    tile_seq = tile_seq + Uint32(1)
                # The utility warp owns the L1 fence and l1_done
                # release now (one release per task, on its last start
                # tile). One claim-group rendezvous keeps the command
                # tensor read below clear of the next claim's rewrite.
                self._walk_sync()
                # Low-watermark L1 ticket: retire into L2 claims so the
                # combine pushes overlap the L1 tail. The command tensor is
                # smem-read past the claim sync, so every thread (and the
                # producer warp) migrates on the same task boundary.
                if is_l1 and command[3] != 0:
                    is_l1 = False
        # Terminator round for the utility walk: the empty wait orders
        # the metadata rewrite behind the utility warp's last real
        # consumption, the valid word doubles as its exit flag, and
        # the CTA-wide exit syncs below hold the claim group until
        # the final epilogue's stores and l1_done release have landed.
        if not utility:
            # The terminator is sequence number hand_seq on the ring:
            # its slot's parity and the same slot-indexed waits the
            # real rounds use (depth one reduces to the old form).
            sl = hand_seq & Uint32(self.combine_slots - 1)
            tph = (hand_seq >> Uint32(self.combine_slots - 1)) & Uint32(1)
            if tid < Uint32(128):
                cute.arch.mbarrier_wait(hand_empty + sl, tph)
                cute.arch.mbarrier_wait(free_bar + sl, tph)
            if tid == 0:
                ms = hand_seq & Uint32(self.combine_slots - 1)
                hand_meta[ms * Uint32(6) + Uint32(3)] = Uint32(0)
            if tid < Uint32(128) and lane == 0:
                cute.arch.mbarrier_arrive(hand_full + sl)
        cute.arch.cp_async_bulk_wait_group(0)
        fence_sys()

    @cute.jit
    def _walk_sync(self):
        """Synchronize the producer and math warps, excluding the utility warp."""
        sync_aligned(128 + 32, 1)

    @cute.jit
    def _l1_dump_mma(self, base: Uint64, expert: Uint32, ntile: Uint32,
                     start: Uint32, valid: Uint32, count: Uint32, accs,
                     hand32: cute.Tensor, hand_meta: cute.Tensor,
                     full_bar: cute.Pointer, empty_bar: cute.Pointer,
                     clamp: Float32, hand_seq: Uint32, chunk: cutlass.Constexpr = -1):
        """Consumer half of the L1 handoff: swiglu, dump, arrive.

        The math warps keep only the cheap arithmetic (the exact swiglu
        they ran before, so the values are bit-identical) and park the
        tile's outputs in the combine region; route weights, the amax
        rendezvous, requant, the scattered stores and the l1_done release
        all move to the utility warp, so the consumers roll straight into
        the next task's k-loop instead of draining the pipeline at every
        epilogue boundary.
        """
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        warp = (tid // 32) % 4
        lane_g = Uint32(lane // 4)
        lane_t = Uint32(lane % 4)
        # Pair p spans tokens 16p..16p+15: accs[2p] the first eight, accs[2p+1]
        # the next eight. Slot r of a pair is token 16p + (8 if r>=2 else 0) +
        # 2*lane_t + r%2; c0/c1 hold the gate row, c2/c3 the up row eight rows
        # below (the same feature under SWAP_AB_INTERLEAVE=8).
        pairs = min(2, self.epilogue_pairs - 2 * chunk) if chunk >= 0 else self.epilogue_pairs
        acc_offset = 4 * chunk if chunk >= 0 else 0
        values = cute.make_rmem_tensor((4 * pairs,), Float32)
        tokens = cute.make_rmem_tensor((4 * pairs,), Uint32)
        for p in cutlass.range_constexpr(pairs):
            acc0, acc1 = accs[acc_offset + 2 * p], accs[acc_offset + 2 * p + 1]
            for r in cutlass.range_constexpr(4):
                j = 4 * p + r
                token = Uint32(16 * p + (8 if r >= 2 else 0)
                               + 2 * lane_t + (r % 2))
                tokens[j] = token
                gate_reg = acc0[r] if r < 2 else acc1[r - 2]
                up_reg = acc0[r + 2] if r < 2 else acc1[r]
                values[j] = self._routed_swiglu(gate_reg, up_reg, clamp)
        # The empty wait gates the slot's reuse: dump s lands in slot
        # s & (slots-1), whose previous round is s-slots, and the slot's
        # parity ph = (s >> (slots-1)) & 1 maps to exactly that round
        # through the init pre-arrivals. At depth one this is the old
        # (slot 0, phase s&1) expression.
        sl = hand_seq & Uint32(self.combine_slots - 1)
        hph = (hand_seq >> Uint32(self.combine_slots - 1)) & Uint32(1)
        cute.arch.mbarrier_wait(empty_bar + sl, hph)
        for j in cutlass.range_constexpr(4 * pairs):
            hand32[sl * Uint32(self.handoff_rows * 32)
                   + self._handoff_word_offset(tokens[j], Uint32(8 * warp) + lane_g)] = values[j]
        if tid == 0:
            m = (hand_seq & Uint32(self.combine_slots - 1)) * Uint32(6)
            hand_meta[m + Uint32(0)] = expert
            hand_meta[m + Uint32(1)] = start
            hand_meta[m + Uint32(2)] = ntile
            hand_meta[m + Uint32(3)] = valid
            last = Uint32(0)
            last_end = start + Uint32(self.tile_n)
            if cutlass.const_expr(self.shared_chunks):
                last_end = start + valid
            if last_end >= count:
                last = Uint32(1)
            hand_meta[m + Uint32(4)] = last
            hand_meta[m + Uint32(5)] = Uint32(1)
        # The leader's release must cover every lane's stores, including
        # short-tail dumps that can outrun the utility warp's previous round.
        cute.arch.sync_warp()
        if lane == 0:
            cute.arch.mbarrier_arrive(full_bar + sl)

    @cute.jit
    def _l2_dump_mma(self, base: Uint64, expert: Uint32, ntile: Uint32,
                     start: Uint32, valid: Uint32, accs,
                     stage16: cute.Tensor, hand_meta: cute.Tensor,
                     full_bar: cute.Pointer, empty_bar: cute.Pointer,
                     free_bar: cute.Pointer, tile_seq: Uint32,
                     hand_seq: Uint32, chunk: cutlass.Constexpr = -1):
        """Round MMA fragments to BF16 and publish the staged rows for return."""
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        warp = (tid // 32) % 4
        lane_g = Uint32(lane // 4)
        lane_t = Uint32(lane % 4)
        producer = tid >= Uint32(128)
        # Slot/phase of the dump's sequence: the two waits key on the same
        # round's retirement either way (tile_seq == hand_seq numerically --
        # both count every start tile), so one z/ph pair serves both.
        z = tile_seq & Uint32(self.combine_slots - 1)
        ph = (tile_seq >> Uint32(self.combine_slots - 1)) & Uint32(1)
        if not producer:
            # The slot's previous round -- pack, push engine reads and
            # all -- has retired: entry arrivals plus one per round keep
            # the slot's phase s-slots complete exactly then.
            cute.arch.mbarrier_wait(free_bar + z, ph)
            # The utility warp's previous round on this slot (its reads
            # and its pack writes) must retire before this dump lands.
            cute.arch.mbarrier_wait(empty_bar + z, ph)
        sl = hand_seq & Uint32(self.combine_slots - 1)
        # Pair p spans tokens 16p..16p+15: accs[2p] the first eight,
        # accs[2p+1] the next eight; c0/c1 sit on feature 16*lane_g,
        # c2/c3 eight below.
        pairs = min(2, self.epilogue_pairs - 2 * chunk) if chunk >= 0 else self.epilogue_pairs
        acc_offset = 4 * chunk if chunk >= 0 else 0
        if cutlass.const_expr(self.bf16_bulk):
            if not producer:
                for p in cutlass.range_constexpr(pairs):
                    lo, hi = accs[acc_offset + 2 * p], accs[acc_offset + 2 * p + 1]
                    row = Uint32(16 * p) + Uint32(lane % 8) + Uint32(lane // 16) * 8
                    col = Uint32(16 * warp) + Uint32((lane // 8) % 2) * 8
                    dst = stage16.iterator + sl * Uint32(self.handoff_rows * 64) + row * 64 + col
                    stmatrix_bf16x8_trans(dst,
                        f32x2_to_bf16x2_bits(lo[0], lo[1]),
                        f32x2_to_bf16x2_bits(lo[2], lo[3]),
                        f32x2_to_bf16x2_bits(hi[0], hi[1]),
                        f32x2_to_bf16x2_bits(hi[2], hi[3]))
                cute.arch.fence_view_async_shared()
        else:
            for p in cutlass.range_constexpr(pairs):
                acc_pair = (accs[acc_offset + 2 * p], accs[acc_offset + 2 * p + 1])
                for r in cutlass.range_constexpr(4):
                    token = Uint32(16 * p + (8 if r >= 2 else 0)
                                   + 2 * lane_t + (r % 2))
                    if token < valid and not producer:
                        acc = acc_pair[0] if r < 2 else acc_pair[1]
                        for f in cutlass.range_constexpr(2):
                            local_col = Uint32(16 * warp) + lane_g + Uint32(8 * f)
                            if cutlass.const_expr(self.shared_chunks):
                                local_col = local_col ^ ((token % Uint32(8)) * Uint32(8))
                            stage16[sl * Uint32(self.handoff_rows * 64)
                                    + token * Uint32(64)
                                    + local_col] = f32_to_bf16_bits(
                                acc[2 * f + (r % 2)])
        if tid == 0:
            m = (hand_seq & Uint32(self.combine_slots - 1)) * Uint32(6)
            hand_meta[m + Uint32(0)] = expert
            hand_meta[m + Uint32(1)] = start
            hand_meta[m + Uint32(2)] = ntile
            hand_meta[m + Uint32(3)] = valid
            last = Uint32(0)
            if cutlass.const_expr(self.relay):
                count = self._field(base, 'expert_count', Uint32)[expert]
                last = Uint32(start + valid >= count)
            hand_meta[m + Uint32(4)] = last
            hand_meta[m + Uint32(5)] = Uint32(0)
        if not producer:
            # Warp rendezvous before the arrival: each leader's release
            # must cover its own warp's dump stores.
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(full_bar + sl)

    @cute.jit
    def _utility_round_mma(self, base: Uint64, expert: Uint32, ntile: Uint32,
                           start: Uint32, token: Uint32, live, weight: Float32,
                           values):
        """One 32-token chunk of the utility round: route-weight multiply,
        amax fold, E4M3 requant and the scattered L2-arena stores."""
        amax = Float32(0)
        for f in cutlass.range_constexpr(32):
            values[f] = values[f] * weight
            amax = cute.math.max(amax, abs_f32(values[f]))
        exponent = ue8m0_fp8_scale_exp(amax)
        inverse = ue8m0_fp8_inv(exponent)
        if live:
            col = ntile * Uint32(32)
            tile_token = (Uint64(expert) * self.rows
                          + Uint64(start) + token)
            row_base = (base + Uint64(self.off_l2_acts)
                        + tile_token * self.intermediate + Uint64(col))
            # Store the token's 32 E4M3 values in two 16-byte vectors.
            words = [Uint32(0) for _ in range(8)]
            for f in cutlass.range_constexpr(8):
                q0 = f32x2_to_e4m3x2_bits(
                    values[4 * f] * inverse, values[4 * f + 1] * inverse)
                q1 = f32x2_to_e4m3x2_bits(
                    values[4 * f + 2] * inverse, values[4 * f + 3] * inverse)
                words[f] = Uint32(q0) | (Uint32(q1) << Uint32(16))
            st_global_u32x4(cute.make_ptr(
                Uint8, row_base, cute.AddressSpace.gmem,
                assumed_align=16),
                words[0], words[1], words[2], words[3])
            st_global_u32x4(cute.make_ptr(
                Uint8, row_base + Uint64(16), cute.AddressSpace.gmem,
                assumed_align=16),
                words[4], words[5], words[6], words[7])
            sf_out = cell(
                base, Uint64(self.off_l2_acts_sf)
                + ((Uint64(expert) * (self.intermediate // 128)
                    + ntile // 4) * self.rows
                   + Uint64(start) + token) * 4
                + ntile % 4, Uint8)
            sf_out[0] = Uint8(exponent)

    @cute.jit
    def _handoff_word_offset(self, row, column):
        if cutlass.const_expr(self.shared_chunks):
            column = column ^ ((row % Uint32(8)) * Uint32(4))
        return row * Uint32(32) + column

    @cute.jit
    def _handoff_load_u32x4(self, ptr):
        if cutlass.const_expr(self.gmem_handoff and not self.shared_chunks):
            return ld_global_16(Uint64(ptr.toint()))
        elif cutlass.const_expr(self.shared_chunks):
            return handoff_load_shared_ordered(ptr)
        else:
            return ld_shared_u32x4(ptr)

    @cute.jit
    def _utility_bf16_bulk(self, peers: cute.Pointer, ntile: Uint32,
                           valid: Uint32, stage: cute.Pointer,
                           src_pf: cute.Tensor, pos_pf: cute.Tensor, park: Uint32):
        lane = cute.arch.thread_idx()[0] % Uint32(32)
        peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        for offset in cutlass.range(0, valid, 32):
            row = Uint32(offset) + lane
            source = Uint32(0)
            pos = Uint32(0)
            live = row < valid
            if live:
                source = Uint32(src_pf[park + row])
                pos = Uint32(pos_pf[park + row])
            prev_source = cute.arch.shuffle_sync_up(source, 1, mask_and_clamp=0)
            prev_pos = cute.arch.shuffle_sync_up(pos, 1, mask_and_clamp=0)
            first = ((lane == 0) | (source != prev_source)
                     | (pos != prev_pos + 1) | ((pos & Uint32(63)) == 0))
            boundaries = Uint32(cute.arch.vote_ballot_sync(first | ~live))
            if first & live:
                later = boundaries & (Uint32(0xFFFFFFFE) << lane)
                count = ballot_leading_ones(Int32(~later)) - lane
                cell_offset = ((Uint64(pos) // 64) * self.colblocks + Uint64(ntile)) * _CELL_BYTES
                dst = Uint64(peer_table[source]) + Uint64(self.off_combine) + cell_offset
                dst = dst + (Uint64(pos) % 64) * _ROW_BYTES
                bulk_s2g_1d(cute.make_ptr(Uint8, dst, cute.AddressSpace.gmem, assumed_align=16),
                            cute.recast_ptr(stage + row * 32, dtype=Uint8), count * _ROW_BYTES)
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def _utility_relay_bf16(self, base, ntile, valid, stage, src_pf, pos_pf, park):
        """Stage exact BF16 packets in row order for the resident return copiers."""
        lane = cute.arch.thread_idx()[0] % Uint32(32)
        row_lane = lane & Uint32(7)
        leader = lane - row_lane
        for offset in cutlass.range(0, valid, 4):
            row = Uint32(offset) + lane // Uint32(8)
            source, pos = Uint32(0), Uint32(0)
            if row_lane == 0 and row < valid:
                source = Uint32(src_pf[park + row])
                pos = Uint32(pos_pf[park + row])
            source = cute.arch.shuffle_sync(source, leader)
            pos = cute.arch.shuffle_sync(pos, leader)
            if row < valid:
                words = ld_shared_u32x4(stage + row * 32 + row_lane * 4)
                row_index = Uint64(source) * self.capacity * self.topk + Uint64(pos)
                output = (base + Uint64(self.off_relay_output)
                          + (row_index * self.colblocks + Uint64(ntile)) * _ROW_BYTES)
                st_global_16(output + Uint64(row_lane) * 16, *words)

    @cute.jit
    def _relay_returns(self, base, peers):
        """Resident copier CTAs copy each expert's completed ranges by source.

        These CTAs replace compute CTAs within the checked residency budget.
        Their shared-memory slots alias the mutually exclusive MMA workspace.
        """
        tid = cute.arch.thread_idx()[0]
        block = cute.arch.block_idx()[0]
        data = cute.arch.get_dyn_smem(Uint8, alignment=128)
        bars = cute.recast_ptr(data + 2 * RELAY_CHUNK_BYTES, dtype=Uint64)
        if tid == 0:
            cute.arch.mbarrier_init(bars, 1)
            cute.arch.mbarrier_init(bars + 1, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        head = self._field(base, 'header', Uint32)
        epoch = ld_acquire_gpu_u32(head)
        sequence = Uint32(0)
        table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        if tid == 0:
            for source_idx in cutlass.range(block, self.ranks, self.relay_ctas):
                source = Uint32(source_idx)
                remote = Uint64(table[source])
                for ticket in cutlass.range(self.local):
                    # Follow FC2's active queue so a later expert cannot hold up
                    # already completed output. Re-read its size after completion.
                    active_count = ld_acquire_gpu_u32(head + 7)
                    dispatch_done = ld_acquire_gpu_u32(head + 9)
                    while Uint32(ticket) >= active_count and dispatch_done != epoch:
                        dispatch_done = ld_acquire_gpu_u32(head + 9)
                        active_count = ld_acquire_gpu_u32(head + 7)
                    active_count = ld_acquire_gpu_u32(head + 7)
                    count, expert = Uint32(0), Uint32(0)
                    if Uint32(ticket) < active_count:
                        wait_epoch(self._field(base, 'active_ready', Uint32) + ticket, epoch, False)
                        expert = Uint32(self._field(base, 'active_expert', Uint32)[ticket])
                        count = Uint32(self._field(base, 'counts', Uint32)[expert * self.ranks + source])
                    if count > Uint32(0):
                        wait_count(self._field(base, 'relay_done', Uint32) + expert,
                                   Uint32(self.l2_ntiles))
                        pos = self._field(base, 'pos_base', Uint32)[expert * self.ranks + source]
                        row_bytes = self.colblocks * _ROW_BYTES
                        local_base = (base + Uint64(self.off_relay_output)
                                      + (Uint64(source) * self.capacity * self.topk
                                         + Uint64(pos)) * row_bytes)
                        remote_base = (remote + Uint64(self.off_combine)
                                       + Uint64(pos) * row_bytes)
                        length = count * Uint32(self.colblocks * _ROW_BYTES)
                        global_async_proxy_fence()
                        for offset in cutlass.range(0, length, RELAY_CHUNK_BYTES):
                            slot = sequence & Uint32(1)
                            phase = (sequence // Uint32(2)) & Uint32(1)
                            size = cute.math.min(Uint32(RELAY_CHUNK_BYTES), length - Uint32(offset))
                            # Retire the older send before overwriting its source.
                            cute.arch.cp_async_bulk_wait_group(1, read=True)
                            staged = cute.recast_ptr(data, dtype=Uint8) + slot * Uint32(RELAY_CHUNK_BYTES)
                            incoming = cute.make_ptr(Uint8, local_base + Uint64(offset),
                                                     cute.AddressSpace.gmem, assumed_align=16)
                            bulk_g2s_1d(staged, incoming, size, bars + slot)
                            cute.arch.mbarrier_arrive_and_expect_tx(bars + slot, size)
                            cute.arch.mbarrier_wait(bars + slot, phase)
                            cute.arch.fence_view_async_shared()
                            outgoing = cute.make_ptr(Uint8, remote_base + Uint64(offset),
                                                     cute.AddressSpace.gmem, assumed_align=16)
                            bulk_s2g_1d(outgoing, staged, size)
                            cute.arch.cp_async_bulk_commit_group()
                            sequence += Uint32(1)
            cute.arch.cp_async_bulk_wait_group(0)
            fence_sys()
        cute.arch.sync_threads()

    @cute.jit
    def _utility_batched_bf16(self, peers: cute.Pointer, ntile: Uint32,
                               valid: Uint32, stage: cute.Pointer,
                               src_pf: cute.Tensor, pos_pf: cute.Tensor,
                               dests: cute.Tensor, park: Uint32):
        """Batch contiguous BF16 rows and keep one bulk copy group in flight."""
        lane = cute.arch.thread_idx()[0] % Uint32(32)
        row_lane = lane & Uint32(7)
        peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        for offset in cutlass.range(0, valid, 4):
            row = Uint32(offset) + lane // Uint32(8)
            if row_lane == 0 and row < valid:
                source = Uint32(src_pf[park + row])
                pos = Uint32(pos_pf[park + row])
                cell_id = (Uint64(pos) // 64) * self.colblocks + Uint64(ntile)
                dests[row] = (Uint64(peer_table[source]) + Uint64(self.off_combine)
                              + (cell_id * 64 + Uint64(pos) % 64) * _ROW_BYTES)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        for offset in cutlass.range(0, valid, 32):
            row = Uint32(offset) + lane
            dst = Uint64(0)
            if row < valid:
                dst = Uint64(dests[row])
            previous = (Uint64(cute.arch.shuffle_sync_up(Uint32(dst), 1, mask_and_clamp=0))
                        | (Uint64(cute.arch.shuffle_sync_up(Uint32(dst >> 32), 1,
                                                          mask_and_clamp=0)) << 32))
            first = ((lane == 0) | (dst == 0) | (previous == 0) | (dst != previous + _ROW_BYTES))
            boundaries = Uint32(cute.arch.vote_ballot_sync(first))
            if first & (dst != 0):
                later = boundaries & (Uint32(0xFFFFFFFE) << lane)
                count = ballot_leading_ones(Int32(~later)) - lane
                bulk_s2g_1d(cute.make_ptr(Uint8, dst, cute.AddressSpace.gmem, assumed_align=16),
                            cute.recast_ptr(stage + row * 32, dtype=Uint8), count * _ROW_BYTES)
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(1, read=True)


    @cute.jit
    def _utility_bf16_round(self, peers: cute.Pointer, ntile: Uint32,
                            valid: Uint32, stage32: cute.Tensor,
                            free_bar: cute.Pointer, empty_bar: cute.Pointer,
                            src_pf: cute.Tensor, pos_pf: cute.Tensor, seq: Uint32):
        """Post four unmodified BF16 rows per warp iteration."""
        lane = cute.arch.thread_idx()[0] % Uint32(32)
        row_lane = lane & Uint32(7)
        row_leader = lane - row_lane
        peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        slot = seq & Uint32(self.combine_slots - 1)
        stage = stage32.iterator + slot * Uint32(self.handoff_rows * 32)
        park = (seq & Uint32(2 * self.combine_slots - 1)) * Uint32(self.tile_n)
        if cutlass.const_expr(self.bf16_bulk):
            self._utility_bf16_bulk(peers, ntile, valid, stage, src_pf, pos_pf, park)
        else:
            for offset in cutlass.range(0, valid, 4):
                row = Uint32(offset) + lane // Uint32(8)
                source = Uint32(0)
                pos = Uint32(0)
                if row_lane == 0 and row < valid:
                    source = Uint32(src_pf[park + row])
                    pos = Uint32(pos_pf[park + row])
                source = cute.arch.shuffle_sync(source, row_leader)
                pos = cute.arch.shuffle_sync(pos, row_leader)
                if row < valid:
                    column = row_lane * Uint32(4)
                    words = self._handoff_load_u32x4(
                        stage + self._handoff_word_offset(row, column))
                    cell_offset = ((Uint64(pos) // 64) * self.colblocks + Uint64(ntile)) * _CELL_BYTES
                    dst = (Uint64(peer_table[source]) + Uint64(self.off_combine)
                           + cell_offset + (Uint64(pos) % 64) * _ROW_BYTES + Uint64(column) * 4)
                    st_global_16(dst, words[0], words[1], words[2], words[3])
        cute.arch.sync_warp()
        if lane == 0:
            cute.arch.mbarrier_arrive(free_bar)
            cute.arch.mbarrier_arrive(empty_bar)

    @cute.jit
    def _utility_walk_mma(self, base: Uint64, peers: cute.Pointer,
                          hand32: cute.Tensor, stage32: cute.Tensor,
                          stage16: cute.Tensor, hand_meta: cute.Tensor,
                          full_bar: cute.Pointer, empty_bar: cute.Pointer,
                          free_bar: cute.Pointer, src_pf: cute.Tensor,
                          pos_pf: cute.Tensor, return_dests: cute.Tensor):
        """Drain L1 requantization and BF16 L2 returns in ring order.

        Each slot starts empty. The utility warp releases the slot only after
        requantization or the asynchronous peer copy has consumed its data.
        """
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % Uint32(32)
        if lane == 0:
            # Entry completes each slot's phase 0 ("round -1" per slot):
            # the stage starts empty, so the first dump on every
            # slot's wait passes immediately. One arrival per slot -- at
            # depth two each slot owns its barrier, since a shared one
            # would leave the even-round consumer two phases behind the
            # parity wait's one-phase window.
            for z in cutlass.range_constexpr(self.combine_slots):
                cute.arch.mbarrier_arrive(free_bar + Int32(z))
        seq = Uint32(0)
        running = True
        pending = Boolean(False)
        while running:
            z = seq & Uint32(self.combine_slots - 1)
            ph = (seq >> Uint32(self.combine_slots - 1)) & Uint32(1)
            cute.arch.mbarrier_wait(full_bar + z, ph)
            m = z * Uint32(6)
            expert = hand_meta[m + Uint32(0)]
            start = hand_meta[m + Uint32(1)]
            ntile = hand_meta[m + Uint32(2)]
            valid = hand_meta[m + Uint32(3)]
            last = hand_meta[m + Uint32(4)]
            kind = hand_meta[m + Uint32(5)]
            if cutlass.const_expr(self.batched_returns):
                if pending and ((valid == Uint32(0)) | (kind != Uint32(0))):
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    cute.arch.sync_warp()
                    if lane == 0:
                        cute.arch.mbarrier_arrive(free_bar + (z ^ Uint32(1)))
                        cute.arch.mbarrier_arrive(empty_bar + (z ^ Uint32(1)))
                    pending = Boolean(False)
            if valid == Uint32(0):
                running = False
            elif cutlass.const_expr(self.bf16_utility) and kind == Uint32(0):
                if cutlass.const_expr(self.batched_returns):
                    stage = stage32.iterator + z * Uint32(self.handoff_rows * 32)
                    park = (seq & Uint32(2 * self.combine_slots - 1)) * Uint32(self.tile_n)
                    if cutlass.const_expr(self.relay):
                        self._utility_relay_bf16(base, ntile, valid, stage, src_pf, pos_pf, park)
                        if last != Uint32(0):
                            fence_gpu()
                            cute.arch.sync_warp()
                            if lane == 0:
                                atom_release_gpu_add_u32(self._field(base, 'relay_done', Uint32) + expert, Uint32(1))
                    else:
                        self._utility_batched_bf16(peers, ntile, valid, stage, src_pf, pos_pf, return_dests, park)
                    cute.arch.sync_warp()
                    if cutlass.const_expr(self.relay):
                        if lane == 0:
                            cute.arch.mbarrier_arrive(free_bar + z)
                            cute.arch.mbarrier_arrive(empty_bar + z)
                    else:
                        if pending and lane == 0:
                            cute.arch.mbarrier_arrive(free_bar + (z ^ Uint32(1)))
                            cute.arch.mbarrier_arrive(empty_bar + (z ^ Uint32(1)))
                        pending = Boolean(True)
                else:
                    self._utility_bf16_round(peers, ntile, valid, stage32,
                                             free_bar + z, empty_bar + z,
                                             src_pf, pos_pf, seq)
            else:
                # Transposed round: lane is the TOKEN axis and the 32
                # features iterate inside the lane, so the per-token
                # amax folds with plain fmax -- the token-serial form
                # ran a six-shuffle dependent chain per token, which
                # outran the consumers' mainloop and stalled the ring.
                # Route weights load one per lane and the requant packs
                # feature pairs into 16-bit stores. The bytes match the
                # consumer epilogue exactly: same f32 values, same
                # multiply-then-max set, same requant operands.
                # Keep the handoff owned through requantization so
                # short next tiles cannot reuse its values or metadata
                # until every operation in this round has finished.
                token = lane
                live = token < valid
                weight = Float32(0)
                if live:
                    weight = cell(
                        base, Uint64(self.off_route_weight)
                        + (Uint64(expert) * self.rows
                           + Uint64(start) + token) * 4, Float32)[0]
                values = cute.make_rmem_tensor((32,), Float32)
                # Vector smem fetch, 8 x 16B loads instead of 32 scalar
                # words: the scalar form's lane-major addresses all
                # landed in one bank (token*32 mod 32 = 0 -- a 32-way
                # conflict per read instruction, most of a round's
                # latency), while the 16B chunks issue as 8-transaction
                # groups. The 16B alignment holds: the region base is
                # 16-aligned and each token's row is 32 words.
                hand32w = cute.recast_ptr(hand32.iterator, dtype=Uint32)
                # Round seq reads slot seq & (slots-1) -- the same slot
                # its dump landed in.
                hand32w = hand32w + Uint64(
                    (seq & Uint32(self.combine_slots - 1))
                    * Uint32(self.handoff_rows * 32))
                # Dead lanes (token past the tile) never read: the
                # narrow tier's handoff region is only tile_n rows
                # deep, and reading past it leaves the CTA's shared
                # window -- an out-of-bounds shared load the early
                # empty arrival then forces to completion.
                if token < Uint32(self.handoff_rows):
                    for w in cutlass.range_constexpr(8):
                        w0, w1, w2, w3 = self._handoff_load_u32x4(
                            hand32w + Uint64(self._handoff_word_offset(token, Uint32(4 * w))))
                        values[4 * w] = bits_float(w0)
                        values[4 * w + 1] = bits_float(w1)
                        values[4 * w + 2] = bits_float(w2)
                        values[4 * w + 3] = bits_float(w3)
                if cutlass.const_expr(self.handoff_rows > 32):
                    token_b = Uint32(32) + lane
                    live_b = token_b < valid
                    weight_b = Float32(0)
                    if live_b:
                        weight_b = cell(
                            base, Uint64(self.off_route_weight)
                            + (Uint64(expert) * self.rows
                               + Uint64(start) + token_b) * 4, Float32)[0]
                    values_b = cute.make_rmem_tensor((32,), Float32)
                    if token_b < Uint32(self.handoff_rows):
                        for w in cutlass.range_constexpr(8):
                            w0, w1, w2, w3 = self._handoff_load_u32x4(
                                hand32w + Uint64(self._handoff_word_offset(token_b, Uint32(4 * w))))
                            values_b[4 * w] = bits_float(w0)
                            values_b[4 * w + 1] = bits_float(w1)
                            values_b[4 * w + 2] = bits_float(w2)
                            values_b[4 * w + 3] = bits_float(w3)
                self._utility_round_mma(base, expert, ntile, start,
                                        token, live, weight, values)
                if cutlass.const_expr(self.handoff_rows > 32):
                    self._utility_round_mma(base, expert, ntile,
                                            start, token_b, live_b,
                                            weight_b, values_b)
                if last != Uint32(0):
                    # One release per (expert, ntile) task: the task's
                    # writes (this warp's stores) must be GPU-visible
                    # before the counter flip other CTAs' L2 claims spin
                    # on. Join the lanes after their fences before the
                    # leader publishes completion.
                    fence_gpu()
                    cute.arch.sync_warp()
                    if lane == 0:
                        atom_release_gpu_add_u32(
                            self._field(base, 'l1_done', Uint32) + expert,
                            Uint32(1))
                if lane == 0:
                    cute.arch.mbarrier_arrive(free_bar + z)
                # The round no longer reads either values or metadata.
                # Only now may a short next tile overwrite their alias.
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(empty_bar + z)
            # Both L1 and L2 rounds advance the ring parity.
            seq = seq + Uint32(1)

    @cute.jit
    def _push_bf16_tile(self, base: Uint64, peers: cute.Pointer, expert: Uint32,
                        ntile: Uint32, start: Uint32, valid: Uint32, accs):
        """Post BF16 directly from the MMA fragments."""
        tid, _, _ = cute.arch.thread_idx()
        if tid < Uint32(128):
            lane = Uint32(tid % 32)
            lane_g, lane_t = lane // Uint32(4), lane % Uint32(4)
            tile = ntile
            warp = Uint32(tid // 32 % 4)
            peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
            for pair in cutlass.range_constexpr(self.epilogue_pairs):
                for r in cutlass.range_constexpr(4):
                    token = Uint32(16 * pair + (8 if r >= 2 else 0) + r % 2) + 2 * lane_t
                    acc = accs[2 * pair + r // 2]
                    post = lane_g == Uint32(0) and token < valid
                    dst = Uint64(0)
                    if post:
                        meta = Uint64(self.off_route_meta) + (
                            Uint64(expert) * self.rows + Uint64(start) + token) * 16
                        source = cell(base, meta, Uint32)[0]
                        pos = Uint64(cell(base, meta + 12, Uint32)[0])
                        cell_offset = ((pos // 64) * self.colblocks + Uint64(tile)) * _CELL_BYTES
                        dst = (Uint64(peer_table[source]) + Uint64(self.off_combine)
                               + cell_offset + (pos % 64) * _ROW_BYTES + Uint64(warp) * 32)
                    for f in cutlass.range_constexpr(2):
                        bits = Uint32(f32_to_bf16_bits(acc[2 * f + r % 2]))
                        # Each group owns one feature at a fixed token lane.
                        # Form four BF16 pairs and gather them for one 16B post.
                        pair_bits = bits | (cute.arch.shuffle_sync_bfly(bits, 4) << Uint32(16))
                        w0 = cute.arch.shuffle_sync(pair_bits, lane_t)
                        w1 = cute.arch.shuffle_sync(pair_bits, lane_t + Uint32(8))
                        w2 = cute.arch.shuffle_sync(pair_bits, lane_t + Uint32(16))
                        w3 = cute.arch.shuffle_sync(pair_bits, lane_t + Uint32(24))
                        if post:
                            st_global_16(dst + Uint64(16 * f), w0, w1, w2, w3)

    @cute.jit
    def _combine_bf16_vectors(self, base: Uint64, y: cute.Pointer,
                              tokens: Uint32, index_start: Uint32):
        """One thread per 16 output bytes, with the original FP32 slot order."""
        vectors_per_token = self.hidden // 8
        for index in cutlass.range(index_start, tokens * vectors_per_token, self.grid * 256):
            token = index // vectors_per_token
            col = index % vectors_per_token
            cb, vector = col // 8, col % 8
            total = cute.make_rmem_tensor((8,), Float32)
            total.fill(Float32(0))
            for slot in cutlass.range_constexpr(self.topk):
                route = Uint64(token * self.topk + slot)
                expert = cell(base, Uint64(self.off_topk_idx) + route * 8, Int64)[0]
                if expert >= 0 and expert < self.ranks * self.local:
                    if cutlass.const_expr(self.direct_route_scan):
                        pos = Uint32(route)
                    else:
                        pos = Uint32(cell(base, Uint64(self.off_pos_table) + route * 4, Uint32)[0])
                    if pos < self.capacity * self.topk:
                        cell_offset = Uint64(self.off_combine) + (
                            Uint64(pos // 64) * self.colblocks + Uint64(cb)) * _CELL_BYTES
                        row = Uint64(pos % 64)
                        values = load_bf16_8(base + cell_offset + row * _ROW_BYTES + Uint64(vector) * 16)
                        for i in cutlass.range_constexpr(8):
                            total[i] = total[i] + values[i]
                    else:
                        total.fill(Float32(float('nan')))
            byte = Uint64(index) * 16
            if cutlass.const_expr(self.shared_experts):
                values = load_bf16_8(base + Uint64(self.off_shared_y) + byte)
                for i in cutlass.range_constexpr(8):
                    total[i] = total[i] + values[i]
            store_bf16_8(Uint64(y.toint()) + byte, tuple(total[i] for i in range(8)))

    @cute.jit
    def _combine_batched_returns(self, base, y, tokens, head, epoch):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        alloc = cutlass.utils.SmemAllocator()
        route_pos = alloc.allocate_tensor(Int32, cute.make_layout((self.topk,)), 4)
        vectors = self.hidden // 8
        for token in cutlass.range(block, tokens, self.grid):
            if tid < self.topk:
                route = Uint64(token * self.topk + tid)
                expert = cell(base, Uint64(self.off_topk_idx) + route * 8, Int64)[0]
                pos = Int32(-1)
                if expert >= 0 and expert < self.ranks * self.local:
                    pos = Int32(cell(base, Uint64(self.off_pos_table) + route * 4, Uint32)[0])
                route_pos[tid] = pos
            if tid == 0:
                wait_epoch(head + 19, epoch, False)
            cute.arch.sync_threads()
            for vector in cutlass.range(tid, vectors, self.threads):
                cb, within = vector // 8, vector % 8
                total = cute.make_rmem_tensor((8,), Float32)
                total.fill(Float32(0))
                for slot in cutlass.range_constexpr(self.topk):
                    pos = Int32(route_pos[slot])
                    if pos == Int32(self.capacity * self.topk):
                        total.fill(Float32(float('nan')))
                    elif pos >= Int32(0):
                        if cutlass.const_expr(self.relay):
                            row = Uint64(pos) * self.colblocks + Uint64(cb)
                        else:
                            cell_id = Uint64(pos // 64) * self.colblocks + Uint64(cb)
                            row = cell_id * 64 + Uint64(pos % 64)
                        start = Uint64(self.off_combine) + row * _ROW_BYTES
                        values = load_bf16_8(base + start + Uint64(within) * 16)
                        for i in cutlass.range_constexpr(0, 8, 2):
                            lo, hi = add_f32_pair(total[i], total[i + 1], values[i], values[i + 1])
                            total[i], total[i + 1] = lo, hi
                byte = (Uint64(token) * vectors + Uint64(vector)) * 16
                if cutlass.const_expr(self.shared_experts):
                    shared = load_bf16_8(base + Uint64(self.off_shared_y) + byte)
                    for i in cutlass.range_constexpr(0, 8, 2):
                        lo, hi = add_f32_pair(total[i], total[i + 1], shared[i], shared[i + 1])
                        total[i], total[i + 1] = lo, hi
                store_bf16_8(Uint64(y.toint()) + byte, tuple(total[i] for i in range(8)))
            if cutlass.const_expr(self.grid < self.capacity):
                cute.arch.sync_threads()

    @cute.jit
    def _clear_shared_fc1(self, base, block, tid):
        if cutlass.const_expr(self.shared_fc1_flags > 0):
            if block == 0:
                done = self._field(base, 'shared_fc1_done', Uint32)
                for tile in cutlass.range(tid, self.shared_fc1_flags, self.threads):
                    done[tile] = Uint32(0)

    @cute.jit
    def _finalize_batched_returns(self, base, peers, y, rank, tokens):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        self._clear_shared_fc1(base, block, tid)
        head = self._field(base, 'header', Uint32)
        epoch = ld_acquire_gpu_u32(head)
        table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        if block == 0 and tid < self.ranks:
            peer_ready = self._field(Uint64(table[tid]), 'compute_ready', Uint32)
            st_release_u32(peer_ready + rank, epoch, True)
        active = self._field(base, 'active_ready', Uint32)
        l1_done = self._field(base, 'l1_done', Uint32)
        for expert in cutlass.range(block * self.threads + tid, self.local, self.grid * self.threads):
            active[expert] = epoch
            l1_done[expert] = Uint32(0)
        if block == 0 and tid == 0:
            for index in cutlass.range_constexpr(8):
                head[(3, 7, 8, 10, 11, 13, 20, 22)[index]] = Uint32(0)
        if block == 0 and tid < 32:
            if tid < self.ranks:
                wait_epoch(self._field(base, 'compute_ready', Uint32) + tid, epoch, True)
            cute.arch.sync_warp()
            if tid == 0:
                st_release_u32(head + 19, epoch, False)
        self._combine_batched_returns(base, y, tokens, head, epoch)
        # Keep completion flags safe when the workspace switches back to
        # the ordinary finalize at epoch wraparound.
        if epoch == Uint32(0xFFFFFFFF):
            flags = self._field(base, 'combine_block_done', Uint32)
            for flag in cutlass.range(block * self.threads + tid, self.finalize_flags,
                                      self.grid * self.threads):
                flags[flag] = epoch
        cute.arch.sync_threads()
        if tid == 0:
            # CUDA Graph completion makes the output visible to its callers.
            # The last CTA retires the epoch after every local CTA has arrived.
            done = atom_acq_rel_add_u32(head + 4, Uint32(1))
            if done + Uint32(1) == Uint32(self.grid):
                head[4] = Uint32(0)
                head[18] = Uint32(0)
                st_release_u32(head, epoch + Uint32(1), False)

    @cute.jit
    def _combine(self, base: Uint64, peers: cute.Pointer, y: cute.Pointer, rank: Uint32, tokens: Uint32):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        self._clear_shared_fc1(base, block, tid)
        head = self._field(base, 'header', Uint32)
        epoch = ld_acquire_gpu_u32(head)
        table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        cute.arch.sync_threads()
        if tid < 128:
            # Local Role1/2 graph predecessors have retired. These fields
            # have ONLY local readers; cleanup may overlap peer scatter wait
            # and complete FP32 slot combine in the other 256 threads.
            if block == 0 and tid < self.ranks:
                # Push completion to each peer; consumers acquire local flags.
                st_release_u32(self._field(Uint64(table[tid]), 'compute_ready', Uint32) + rank, epoch, True)
            for expert in cutlass.range(block * 128 + tid, self.local, self.grid * 128):
                flag_out = cell(base, Uint64(self.off_active_ready) + Uint64(expert) * 4, Uint32)
                flag_out[0] = epoch
                done_out = cell(base, Uint64(self.off_l1_done) + Uint64(expert) * 4, Uint32)
                done_out[0] = Uint32(0)
            if block == 0 and tid == 0:
                for c in cutlass.range_constexpr(8):
                    counter_out = cell(base, Uint64((3, 7, 8, 10, 11, 13, 20, 22)[c] * 4), Uint32)
                    counter_out[0] = Uint32(0)
            # Retain peer-readable send_count/source_index through this call.
            # The all-peer compute_ready rendezvous below retires their old readers
            # before any rank may begin its next dispatch initialization.
            fence_gpu()
            if tid == 0:
                if cutlass.const_expr(self.cleanup_grids == 1):
                    if block == 0:
                        st_release_u32(head + 19, epoch, False)
                else:
                    if block < self.cleanup_grids:
                        arrived = atom_acq_rel_add_u32(head + 18, Uint32(1))
                        if arrived == self.cleanup_grids - 1:
                            st_release_u32(head + 19, epoch, False)
                    if block == 0:
                        wait_epoch(head + 19, epoch, False)
        else:
            math_tid = tid - 128
            if math_tid < self.ranks:
                wait_epoch(self._field(base, 'compute_ready', Uint32) + math_tid, epoch, True)
            # Only the first `ranks` math threads wait on peers; this barrier
            # extends their acquire to every combine reader in the block.
            sync_math()
            if cutlass.const_expr(self.bf16_utility):
                self._combine_bf16_vectors(base, y, tokens, Uint32(block * 256 + math_tid))
            else:
                # Each thread combines 32 columns of one token. A full
                # 64-column cell would exceed the register budget.
                ncells = self.hidden // 64
                for index in cutlass.range(block * 256 + math_tid, tokens * 2 * ncells, self.grid * 256):
                    token, rem = index // (2 * ncells), index % (2 * ncells)
                    cb, half = rem // 2, rem % 2
                    total = [Float32(0) for _ in range(32)]
                    for slot in cutlass.range_constexpr(self.topk):
                        expert = cell(base, Uint64(self.off_topk_idx) + Uint64(token * self.topk + slot) * 8, Int64)[0]
                        if expert >= 0 and expert < self.ranks * self.local:
                            if cutlass.const_expr(self.direct_route_scan):
                                # Direct mode parked no pos table: the route index
                                # itself is the arena position.
                                pos = Uint32(token * self.topk + slot)
                            else:
                                pos = cell(base, Uint64(self.off_pos_table)
                                           + Uint64(token * self.topk + slot) * 4, Uint32)[0]
                            if pos < self.capacity * self.topk:
                                cellb = Uint64(self.off_combine) + (
                                    (Uint64(pos) // 64 * (self.hidden // 64) + Uint64(cb)) * _CELL_BYTES)
                                raw = base + cellb + (Uint64(pos) % 64) * _ROW_BYTES + Uint64(half) * 64
                                for k in cutlass.range_constexpr(4):
                                    values8 = load_bf16_8(raw + Uint64(16 * k))
                                    for i in cutlass.range_constexpr(8):
                                        total[8 * k + i] = total[8 * k + i] + values8[i]
                            else:
                                for i in cutlass.range_constexpr(32):
                                    total[i] = Float32(float('nan'))
                    y_slot = Uint64(y.toint()) + (
                        Uint64(token) * self.hidden + Uint64(cb) * 64
                        + Uint64(half) * 32) * 2
                    if cutlass.const_expr(self.shared_experts):
                        shared_slot = base + Uint64(self.off_shared_y) + (
                            Uint64(token) * self.hidden + Uint64(cb) * 64
                            + Uint64(half) * 32) * 2
                        for shared_chunk in cutlass.range_constexpr(4):
                            shared_values = load_bf16_8(shared_slot + Uint64(shared_chunk * 16))
                            for shared_column in cutlass.range_constexpr(8):
                                shared_index = shared_chunk * 8 + shared_column
                                total[shared_index] = total[shared_index] + shared_values[shared_column]
                    for c in cutlass.range_constexpr(2):
                        store_bf16_8(y_slot + Uint64(c * 32),
                                    tuple(total[c * 16 + i] for i in range(8)))
                        store_bf16_8(y_slot + Uint64(c * 32 + 16),
                                    tuple(total[c * 16 + 8 + i] for i in range(8)))
            fence_sys()
            # Retire every combine store in the block before thread 0
            # publishes this block's completion flag.
            sync_math()
            if math_tid == 0:
                st_release_u32(self._field(base, 'combine_block_done', Uint32) + block, epoch, False)
            if block == 0:
                for other_block in cutlass.range(math_tid, self.grid, 256):
                    wait_epoch(self._field(base, 'combine_block_done', Uint32) + other_block, epoch, False)
                sync_math()
                # Refresh EVERY unused flag after all old readers retire.
                # This prevents stale zero from aliasing epoch FFFFFFFF->0,
                # including when successive calls use different grid sizes.
                for flag in cutlass.range(math_tid, self.finalize_flags, 256):
                    flag_out = cell(base, Uint64(self.off_combine_block_done) + Uint64(flag) * 4, Uint32)
                    flag_out[0] = epoch
                fence_gpu()
        # Retire every LOCAL combine/cleanup group before the successor.
        # A next-call scatter cannot overwrite a peer's old combine slots:
        # both dispatch paths await every peer's next input_ready before
        # publishing any expert_ready, so that peer has finished this call.
        # The earlier all-peer compute_ready wait also retires all old source readers.
        # This removes a redundant all-rank exit wait, including for T0.
        sync_tile()
        if block == 0 and tid == 0:
            final_out = cell(base, Uint64(16), Uint32)
            final_out[0] = Uint32(0)
            cleanup_out = cell(base, Uint64(18 * 4), Uint32)
            cleanup_out[0] = Uint32(0)
            if cutlass.const_expr(self.direct_route_scan):
                # Keep streaming-only tags one generation behind even if
                # direct calls span the entire epoch counter range.
                for flag in cutlass.range_constexpr(5):
                    stream_tag = cell(base, Uint64((12, 14, 21, 23, 24)[flag] * 4), Uint32)
                    stream_tag[0] = epoch
            st_release_u32(head, epoch + Uint32(1), False)

    @cute.jit
    def _compute_shared(self, base, s1, s2, w1_map, w2_map, xsf, clamp):
        tid = cute.arch.thread_idx()[0]
        block = cute.arch.block_idx()[0] - self.relay_ctas
        if tid < 128:
            slab = cute.arch.get_dyn_smem(Uint8, alignment=1024)
            x = self._field(base, 'x', Uint8)
            l2 = self._field(base, 'shared_l2_acts', Uint8)
            l2_sf = self._field(base, 'shared_l2_acts_sf', Uint8)
            output = self._field(base, 'shared_y', Uint8)
            done = self._field(base, 'shared_fc1_done', Uint32)
            workers = self.grid - self.relay_ctas
            for tile in cutlass.range(block, self.shared_fc1.tiles, workers):
                self.shared_fc1.run_tile(slab, Uint32(tile), s1, x, xsf,
                                         l2, l2_sf, w1_map, clamp)
                sync_shared_gemm()
                if tid == 0:
                    atom_acq_rel_add_u32(done + tile // self.shared_fc1.output_tiles, Uint32(1))
            # Every resident worker publishes its FC1 tiles before waiting for FC2.
            for tile in cutlass.range(block, self.shared_fc2.tiles, workers):
                if tid == 0:
                    wait_count(done + tile // self.shared_fc2.output_tiles,
                               Uint32(self.shared_fc1.output_tiles))
                sync_shared_gemm()
                global_async_proxy_fence()
                self.shared_fc2.run_tile(slab, Uint32(tile), s2, l2, l2_sf,
                                         output, l2_sf, w2_map, clamp)
                sync_shared_gemm()
        # The shared and routed GEMMs reuse the same dynamic shared memory.
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(self, workspace: cute.Pointer, peers: cute.Pointer, y: cute.Pointer,
               w1: cute.Pointer, s1: cute.Pointer, w2: cute.Pointer,
               s2: cute.Pointer, w1_map: cute.Pointer, x1_map: cute.Pointer,
               w2_map: cute.Pointer, x2_map: cute.Pointer, stats: cute.Pointer,
               shared_s1: cute.Pointer, shared_s2: cute.Pointer,
               shared_w1_map: cute.Pointer, shared_w2_map: cute.Pointer,
               shared_xsf: cute.Pointer,
               rank: Uint32, tokens: Uint32, clamp: Float32,
               a_smem_layout: cute.ComposedLayout,
               b_smem_layout: cute.ComposedLayout, sfa_smem_layout: cute.Layout,
               sfb_smem_layout: cute.Layout):
        base = Uint64(workspace.toint())
        if cutlass.const_expr(self.role == 1):
            if cutlass.const_expr(self.launch_dependents):
                if cute.arch.thread_idx()[0] == 0:
                    cute.arch.griddepcontrol_launch_dependents()
            if cutlass.const_expr(self.direct_route_scan):
                self._dispatch_direct(base, peers, rank, tokens, stats)
            else:
                self._dispatch(base, peers, rank, tokens, stats)
        elif cutlass.const_expr(self.role == 2):
            compute = Boolean(True)
            if cutlass.const_expr(self.relay):
                block, _, _ = cute.arch.block_idx()
                compute = block >= Uint32(self.relay_ctas)
                if not compute:
                    self._relay_returns(base, peers)
            if compute:
                if cutlass.const_expr(self.shared_tile_n > 0):
                    self._compute_shared(base, shared_s1, shared_s2,
                                         shared_w1_map, shared_w2_map, shared_xsf, clamp)
                # Build the tiler here to keep it out of the pointer-only launch ABI.
                tiled_mma = make_swapab_m64n8k128_tiled_mma(
                    a_dtype=Float4E2M1FN, b_dtype=Float8E4M3FN,
                    acc_dtype=Float32, sf_dtype=Float8E8M0FNU)
                self._compute_mma(base, peers, w1, s1, w2, s2, w1_map, x1_map,
                                  w2_map, x2_map, clamp, tiled_mma, a_smem_layout,
                                  b_smem_layout, sfa_smem_layout, sfb_smem_layout)
        elif cutlass.const_expr(self.batched_returns):
            self._finalize_batched_returns(base, peers, y, rank, tokens)
        else:
            self._combine(base, peers, y, rank, tokens)

    @cute.jit
    def __call__(self, workspace: cute.Pointer, peers: cute.Pointer, y: cute.Pointer,
                 w1: cute.Pointer, s1: cute.Pointer, w2: cute.Pointer,
                 s2: cute.Pointer, w1_map: cute.Pointer, x1_map: cute.Pointer,
                 w2_map: cute.Pointer, x2_map: cute.Pointer, stats: cute.Pointer,
                 shared_s1: cute.Pointer, shared_s2: cute.Pointer,
                 shared_w1_map: cute.Pointer, shared_w2_map: cute.Pointer,
                 shared_xsf: cute.Pointer,
                 rank: Uint32, tokens: Uint32, clamp: Float32):
        tiled_mma = make_swapab_m64n8k128_tiled_mma(
            a_dtype=Float4E2M1FN, b_dtype=Float8E4M3FN, acc_dtype=Float32,
            sf_dtype=Float8E8M0FNU)
        a_smem_layout_staged = make_data_smem_layout(
            (self.mma_tile[0], self.mma_tile[1],
             self.mma_tile[2] // (2 if self.compact_w else 1)), dtype=Int8, is_a=True,
            num_stages=self.mma_stages)
        b_smem_layout_staged = make_data_smem_layout(
            self.mma_tile, dtype=Float8E4M3FN, is_a=False,
            num_stages=self.mma_stages)
        # Vendored contract: SFA tiler M padded to 128, SFB tiler N padded
        # to 128; one stage per K128 tile.
        sfa_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfa(
            tiled_mma, (128, self.mma_tile[1], self.mma_tile[2]), 32,
            self.mma_stages)
        sfb_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfb(
            tiled_mma, (self.mma_tile[0], 128, self.mma_tile[2]), 32,
            self.mma_stages)

        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[Int8, cute.cosize(a_smem_layout_staged)], 128]
            sB: cute.struct.Align[
                cute.struct.MemRange[Float8E4M3FN, cute.cosize(b_smem_layout_staged)], 128]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[Float8E8M0FNU, cute.cosize(sfa_smem_layout_staged)], 128]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[Float8E8M0FNU, cute.cosize(sfb_smem_layout_staged)], 128]
        self.shared_storage = SharedStorage

        if cutlass.const_expr(self.shared_tile_n > 0):
            k128, block_m, input_sf_rows = self.shared_layout
            self.shared_fc1 = SM120SharedExpertGemm(
                self.hidden, self.shared_intermediate, self.tokens,
                first=True, tile_n=self.shared_tile_n, fast_math=self.fast_math,
                k128_weights=k128[0],
                input_block_m=block_m, input_sf_rows=input_sf_rows)
            self.shared_fc2 = SM120SharedExpertGemm(
                self.hidden, self.shared_intermediate, self.tokens,
                first=False, tile_n=self.shared_tile_n,
                k128_weights=k128[1])
        self.kernel(workspace, peers, y, w1, s1, w2, s2, w1_map, x1_map, w2_map,
                    x2_map, stats, shared_s1, shared_s2,
                    shared_w1_map, shared_w2_map, shared_xsf, rank, tokens, clamp,
                    a_smem_layout_staged, b_smem_layout_staged,
                    sfa_smem_layout_staged, sfb_smem_layout_staged).launch(
            grid=(self.grid, 1, 1), block=(self.threads, 1, 1),
            max_number_threads=(self.threads, 1, 1),
            min_blocks_per_mp=1 if self.role == 2 else 4)


@cute.jit
def shared_cell(ptr: cute.Pointer, offset: Uint32, dtype: cutlass.Constexpr):
    return cute.make_tensor(cute.recast_ptr(ptr + offset, dtype=dtype),
                            cute.make_layout((1,)))


@dsl_user_op
def sync_shared_gemm(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'barrier.sync 2, 128;', '~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT,
                    loc=loc, ip=ip)


@dsl_user_op
def invalidate_shared_gemm_barrier(bar: cute.Pointer, *, loc=None, ip=None):
    llvm.inline_asm(None, [bar.toint(loc=loc, ip=ip).ir_value()],
                    'mbarrier.inval.shared::cta.b64 [$0];', 'r,~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT,
                    loc=loc, ip=ip)


class SM120SharedExpertGemm:
    """Shared FC1 and FC2, including the BF16 SwiGLU and MXFP8 epilogue."""

    def __init__(self, hidden, intermediate, tokens, *, first, tile_n=16,
                 fast_math=True, k128_weights=False,
                 input_block_m=16, input_sf_rows=0):
        self.hidden, self.intermediate = hidden, intermediate
        self.tokens, self.first, self.tile_n = tokens, first, tile_n
        self.fast_math = fast_math
        self.k128_weights = k128_weights
        self.input_block_m = input_block_m
        self.input_sf_rows = input_sf_rows
        self.k = hidden if first else intermediate
        self.rows = 2 * intermediate if first else hidden
        self.groups = tile_n // 8
        # Packed SF loads cover all 128 rows of one SM100 permutation block.
        self.x_offset = 8192 + 512
        self.sf_offset = self.x_offset + tile_n * 128
        self.stage_bytes = shared_gemm_stage_bytes(tile_n)
        self.output_tiles = self.rows // 64
        self.tiles = self.output_tiles * ((tokens + tile_n - 1) // tile_n)

    @cute.jit
    def _stage(self, slab, bar, scales, x, xsf, weight_map, start, valid, ntile, kb):
        tid = cute.arch.thread_idx()[0]
        if tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(bar, self.x_offset)
            tile_g2s_2d(slab, weight_map, bar, Uint32(0),
                        kb * self.rows + ntile * 64, Uint64(EVICT_FIRST))
            sf_col = kb // 4 if cutlass.const_expr(self.k128_weights) else kb
            sf_word = Uint64(sf_col) * self.rows + Uint64(ntile // 2) * 128
            bulk_g2s_1d(slab + 8192, scales + sf_word * 4, Uint32(512), bar)
        for j in cutlass.range(tid, self.tile_n * 8, 128):
            row, col = Uint32(j // 8), Uint32(j % 8) * 16
            size = Uint32(16) if row < valid else Uint32(0)
            src = x + (Uint64(start) + Uint64(row)) * self.k + Uint64(kb) * 128 + Uint64(col)
            copy_async_16(slab + self.x_offset + row * 128
                           + (col ^ ((row & 7) * 16)), src, size)
        if tid < self.tile_n:
            size = Uint32(4) if tid < valid else Uint32(0)
            token = start + Uint32(tid)
            if cutlass.const_expr(self.first):
                within = token % self.input_block_m
                row = (token // self.input_block_m) * ((self.input_block_m + 127) // 128 * 128)
                row += (within // 128) * 128 + (within % 32) * 4 + (within % 128) // 32
                src = xsf + (Uint64(kb) * self.input_sf_rows + Uint64(row)) * 4
            else:
                src = xsf + (Uint64(token) * (self.k // 128) + Uint64(kb)) * 4
            copy_async_4(slab + self.sf_offset + tid * 4, src, size)
        copy_commit()

    @cute.jit
    def run_tile(self, slab: cute.Pointer, tile: Uint32,
                 scales: cute.Pointer, x: cute.Pointer, xsf: cute.Pointer,
                 output: cute.Pointer, outsf: cute.Pointer,
                 weight_map: cute.Pointer, clamp: Float32):
        tid = cute.arch.thread_idx()[0]
        lane, warp = Uint32(tid % 32), Uint32(tid // 32)
        ntile = tile % self.output_tiles
        start = tile // self.output_tiles * self.tile_n
        valid = cute.math.min(Uint32(self.tokens) - start, Uint32(self.tile_n))
        bars = cute.recast_ptr(slab + 2 * self.stage_bytes, dtype=Uint64)
        if tid < 2:
            cute.arch.mbarrier_init(bars + tid, 1)
        cute.arch.mbarrier_init_fence()
        sync_shared_gemm()
        total = cute.make_rmem_tensor((self.groups, 4), Float32)
        total.fill(Float32(0))
        a = cute.make_rmem_tensor((4,), Uint32)
        b = cute.make_rmem_tensor((2,), Uint32)
        self._stage(slab, bars, scales, x, xsf, weight_map, start, valid, ntile, Uint32(0))
        for kb in cutlass.range(self.k // 128):
            current = slab + Uint32(kb % 2) * self.stage_bytes
            copy_wait()
            if tid == 0:
                cute.arch.mbarrier_wait(bars + Uint32(kb % 2), Uint32((kb // 2) % 2))
            sync_shared_gemm()
            if kb + 1 < self.k // 128:
                nxt = Uint32((kb + 1) % 2)
                self._stage(slab + nxt * self.stage_bytes, bars + nxt,
                            scales, x, xsf, weight_map, start, valid, ntile, Uint32(kb + 1))
            row = warp * 16 + lane // 4
            for ki in cutlass.range_constexpr(4):
                column = Uint32(ki * 32) + (lane % 4) * 4
                wcol0 = column ^ ((row & 7) * 16)
                wcol1 = (column + 16) ^ ((row & 7) * 16)
                a[0] = lds_b32_raw_ordered(current + row * 128 + wcol0)
                a[1] = lds_b32_raw_ordered(current + (row + 8) * 128 + wcol0)
                a[2] = lds_b32_raw_ordered(current + row * 128 + wcol1)
                a[3] = lds_b32_raw_ordered(current + (row + 8) * 128 + wcol1)
                sf_row = row + (lane & 1) * 8
                sf_row = (sf_row % 32) * 4 + (ntile % 2) * 2 + sf_row // 32
                weight_word = lds_b32_raw_ordered(current + 8192 + sf_row * 4)
                sf_byte = kb % 4 if cutlass.const_expr(self.k128_weights) else ki
                weight_exponent = Uint8((weight_word >> (sf_byte * 8)) & 255)
                for ng in cutlass.range_constexpr(self.groups):
                    token = Uint32(ng * 8) + lane // 4
                    xp = current + self.x_offset + token * 128
                    b[0] = lds_b32_raw_ordered(xp + (column ^ ((token & 7) * 16)))
                    b[1] = lds_b32_raw_ordered(xp + ((column + 16) ^ ((token & 7) * 16)))
                    sf = lds_b32_raw_ordered(current + self.sf_offset + token * 4)
                    exponent = Uint8((sf >> (ki * 8)) & 255)
                    _arch_mma_m16n8k32_mxfp8(
                        total[ng, None], a, b, weight_exponent, exponent,
                        a_dtype=Float8E4M3FN, b_dtype=Float8E4M3FN,
                        acc_dtype=Float32, sf_dtype=Float8E8M0FNU)
            cute.arch.fence_view_async_shared()
            sync_shared_gemm()
        if tid < 2:
            invalidate_shared_gemm_barrier(bars + tid)
        # Reuse retired stage storage for the output tile.
        for ng in cutlass.range_constexpr(self.groups):
            token = Uint32(ng * 8) + (lane % 4) * 2
            row = warp * 16 + lane // 4
            for r in cutlass.range_constexpr(2):
                for t in cutlass.range_constexpr(2):
                    dst = shared_cell(slab, ((token + t) * 64 + row + r * 8) * 4, Float32)
                    dst[0] = total[ng, r * 2 + t]
        sync_shared_gemm()
        if cutlass.const_expr(self.first):
            bound = round_bf16_f32(clamp)
            for ep_token in cutlass.range(warp, valid, 4):
                row = (lane // 8) * 16 + lane % 8
                gate = shared_cell(slab, (Uint32(ep_token) * 64 + row) * 4, Float32)[0]
                up = shared_cell(slab, (Uint32(ep_token) * 64 + row + 8) * 4, Float32)[0]
                gate = bits_float(Uint32(f32_to_bf16_bits(gate)) << 16)
                up = bits_float(Uint32(f32_to_bf16_bits(up)) << 16)
                gate = cute.math.min(gate, bound)
                up = cute.math.min(cute.math.max(up, -bound), bound)
                denom = Float32(1) + cute.math.exp(-gate, fastmath=self.fast_math)
                if cutlass.const_expr(self.fast_math):
                    gate = gate * cute.math.rcp(denom)
                else:
                    gate = cute.math.div(gate, denom, fastmath=False, ftz=False,
                                         rounding=cute.math.RoundingMode.NEAREST_EVEN)
                value = gate * up
                amax = abs_f32(value)
                for shift in cutlass.range_constexpr(5):
                    amax = cute.math.max(amax, cute.arch.shuffle_sync_bfly(amax, 16 >> shift))
                ep_exponent = ue8m0_fp8_scale_exp(amax)
                inverse = ue8m0_fp8_inv(ep_exponent)
                packed = f32x2_to_e4m3x2_bits(value * inverse, Float32(0))
                ep_dst = shared_cell(output, (start + Uint32(ep_token)) * self.intermediate + ntile * 32 + lane, Uint8)
                ep_dst[0] = Uint8(packed)
                if lane == 0:
                    ep_sf = shared_cell(outsf, (start + Uint32(ep_token)) * (self.intermediate // 32) + ntile, Uint8)
                    ep_sf[0] = Uint8(ep_exponent)
        else:
            for j in cutlass.range(tid, valid * 64, 128):
                value = shared_cell(slab, Uint32(j) * 4, Float32)[0]
                y_dst = shared_cell(output, ((start + Uint32(j // 64)) * self.hidden + ntile * 64 + Uint32(j % 64)) * 2, cutlass.Uint16)
                y_dst[0] = f32_to_bf16_bits(value)
