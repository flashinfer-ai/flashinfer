"""SM90 CuTe dispatch, register-A WGMMA compute and combine roles."""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int64, Uint8, Uint32, Uint64

from ..common.math import (
    abs_f32,
    bits_float,
    f32x2_to_e4m3x2_bits,
    round_bf16_f32,
    preweight_routed_sf,
    ue8m0_fp8_inv,
    ue8m0_fp8_scale_exp,
)
from ..mma.sm90 import (
    canonical_offset,
    decode_fragment_nib4,
    smem_descriptor,
    wgmma_rs_issue,
    wgmma_rs_k128,
    wgmma_rs_k32_pair,
    wgmma_rs_k32_pair_issue,
    wgmma_rs_wait,
)
from ..ptx.ld_st import (
    EVICT_FIRST,
    EVICT_NORMAL,
    atom_acq_rel_add_u32,
    atom_add_u32,
    atom_release_gpu_add_u32,
    atom_shared_add_u32,
    copy_global_16,
    copy_global_2x16,
    ld_acquire_gpu_u32,
    ld_acquire_sys_u32_sm90 as ld_acquire_sys_u32,
    ld_shared_u32x4_sm90 as load_shared_u32x4,
    load_bf16_8,
    load_scale_pair,
    load_shared_4,
    load_shared_8,
    st_release_u32,
    store_bf16_8,
)
from ..ptx.tma import (
    bulk_g2s_1d,
    copy_arrive,
    copy_async_4,
    copy_async_16,
    copy_commit,
    copy_wait,
    tma_store_fence,
)
from ..ptx.utils import (
    fence_gpu,
    fence_sys,
    sync_loader,
    sync_math,
    sync_tile,
    sync_activation,
    sync_weight,
)
from ..scheduler.sm90_gc_moe import claim_sm90_task
from ..layout.sm90_gc_moe import get_role_shared_layout


@cute.jit
def cell(base: Uint64, offset: Uint64, dtype: cutlass.Constexpr):
    return cute.make_tensor(cute.make_ptr(dtype, base + offset, cute.AddressSpace.gmem,
                                         assumed_align=1), cute.make_layout((1,)))


@cute.jit
def shared_cell(slab: cute.Pointer, offset: Uint32, dtype: cutlass.Constexpr):
    return cute.make_tensor(cute.recast_ptr(slab + offset, dtype=dtype), cute.make_layout((1,)))


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


@cute.jit
def wait_count(ptr: cute.Pointer, target: Uint32):
    value = ld_acquire_gpu_u32(ptr)
    while value < target:
        value = ld_acquire_gpu_u32(ptr)


class SM90GCMoERoleKernel:
    def __init__(self, config, layout, role, schedule, has_stats, routed_n_factor=1,
                 routed_stage_count=0, routed_k_pack=1):
        self.role = role
        self.direct_route_scan = 0 < config.bound <= 2
        self.ranks, self.local = config.ranks, config.experts // config.ranks
        self.capacity, self.topk = config.capacity, config.topk
        self.hidden, self.intermediate = config.hidden, config.intermediate
        self.shared_experts = config.shared_experts
        self.shared_scale_prefetch = role in (4, 5) and schedule.shared_scale_prefetch
        self.shared_phase_start = 1 if role == 5 else 0
        self.shared_phase_end = 1 if role == 4 and schedule.split_shared else 2
        if role == 5 and not schedule.split_shared:
            raise ValueError('shared L2 role requires the split shared schedule')
        self.shared_cache_hint = (EVICT_NORMAL if role in (4, 5) and schedule.shared_cache_normal
                                  else EVICT_FIRST)
        self.selected_combine = role == 3 and schedule.selected_combine
        self.combine_vector = (16 if role == 3 and schedule.policy == 'pipelined'
                               and self.hidden % 512 == 0 else 8)
        if self.selected_combine and (self.hidden % 256 or not 1 <= self.topk <= 32):
            raise ValueError('selected combine requires warp-uniform rows and topk in [1, 32]')
        self.shared_i = config.intermediate * config.shared_experts
        # Internal SF keeps RN(originalSF *64), optionally followed by exact
        # FP32 secondary multiplication in its original global writer.
        # Quantization inverse and public source x_sf retain original SF.
        self.rows = config.ranks * config.capacity
        self.fast_math, self.has_stats = config.fast_math, has_stats
        self.warp_specialized = ((role == 2 and schedule.threads[1] == 384)
                                 or role in (4, 5))
        self.max_m = schedule.tile_m
        # The existing common schedule qualifies BOTH receiver and consumer.
        self.preweighted_routed_sf = (role in (1, 2) and self.max_m in (16, 24, 32, 64)
                                      and schedule.policy == 'pipelined'
                                      and schedule.threads[1] == 384
                                      and schedule.routed_scale_preweighted)
        # Both roles use the same existing public schedule qualification.
        self.early_routed_count = self.preweighted_routed_sf
        # Copy eligibility is independent of the internal SF/count format.
        # Complete 1024-byte warp stripes use pairs; original single copies
        # cover the exact remainder for every legal hidden dimension.
        self.paired_routed_copy = role == 1 and schedule.policy == 'pipelined'
        if routed_n_factor not in (1, 2) or (routed_n_factor == 2 and not (
                role == 2 and not self.warp_specialized and self.max_m == 8
                and self.hidden % 512 == 0 and self.intermediate % 256 == 0)):
            raise ValueError('invalid SM90 routed N factor for the role geometry')
        self.routed_n_factor = routed_n_factor
        # Packing two K128 records into one stage is a scheduling change only:
        # both records are already adjacent weight records in global memory.
        if routed_k_pack not in (1, 2):
            raise ValueError('invalid SM90 routed K packing')
        if routed_k_pack == 2 and (self.hidden % 256 or self.intermediate % 256):
            # Each tile must split into whole K256 stages.
            raise ValueError('SM90 routed K packing requires even K128 counts')
        if routed_k_pack == 2 and (self.routed_n_factor != 1 or routed_stage_count != 5):
            raise ValueError('SM90 routed K packing requires N256 and a five-stage override')
        self.routed_k_pack = routed_k_pack
        if routed_stage_count:
            qualified = ((routed_stage_count == 6 and role == 2 and self.max_m == 64
                          and self.preweighted_routed_sf and self.early_routed_count)
                         or (routed_stage_count == 5 and role == 2
                             and self.max_m == 8 and self.warp_specialized
                             and routed_k_pack == 2))
            if not qualified:
                raise ValueError('stage override requires a qualified routed role')
        shared_layout = get_role_shared_layout(self.max_m, self.warp_specialized,
                                               self.routed_n_factor, routed_stage_count,
                                               self.routed_k_pack,
                                               policy=schedule.policy)
        for name in shared_layout.__dataclass_fields__:
            setattr(self, name, getattr(shared_layout, name))
        for name, spec in layout.views.items():
            setattr(self, 'off_' + name, spec.offset)
        resource_role = 1 if role in (4, 5) else role - 1
        self.grid, self.threads = schedule.grids[resource_role], schedule.threads[resource_role]
        self.smem = schedule.smem[resource_role]
        if self.routed_k_pack == 2 and (
                self.full_offset + 1024 + self.epilogue_rows * 1024 + 256 > self.smem):
            raise ValueError('SM90 routed K packing requires complete stages, control space and epilogue')
        if self.routed_n_factor == 2 and 2 * self.stage_bytes + 128 > self.smem:
            raise ValueError('SM90 routed N512 requires two complete stages and control space')
        if role in (4, 5):
            shared_stride = (36864 + self.max_m * 144 + 1023) // 1024 * 1024
            assert 2 * shared_stride + 128 <= self.smem
        self.cleanup_grids = min(self.grid, (self.local + 127) // 128)
        self.finalize_flags = layout.views['combine_block_done'].shape[0]
        if role == 3 and self.grid > self.finalize_flags:
            raise ValueError('combine grid exceeds the SM90 completion-flag capacity')

    @cute.jit
    def _field(self, base: Uint64, name: cutlass.Constexpr, dtype: cutlass.Constexpr):
        return cute.make_ptr(dtype, base + Uint64(getattr(self, 'off_' + name)),
                             cute.AddressSpace.gmem, assumed_align=16)

    @cute.jit
    def _dispatch_grid(self, head: cute.Pointer, epoch: Uint32,
                         counter: cutlass.Constexpr, signal: cutlass.Constexpr,
                         publish_source: cutlass.Constexpr):
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
                    st_release_u32(head + 1, epoch, True)
                st_release_u32(head + signal, epoch, False)
            wait_epoch(head + signal, epoch, False)
        cute.arch.sync_threads()

    @cute.jit
    def _dispatch(self, base: Uint64, peers: cute.Pointer, rank: Uint32,
                   tokens: Uint32, stats: cute.Pointer, secondary: cute.Pointer):
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
        # Four launch-bounded resident CTAs per communication SM make this
        # local grid rendezvous safe even when the compute partition is busy.
        self._dispatch_grid(head, epoch, 11, 12, False)
        for route in cutlass.range(block * self.threads + tid, tokens * self.topk,
                                   self.grid * self.threads):
            expert_id = cell(base, Uint64(self.off_topk_idx) + Uint64(route) * 8, Int64)[0]
            if expert_id >= 0 and expert_id < self.ranks * self.local:
                expert_u = Uint32(expert_id)
                slot = atom_add_u32(self._field(base, 'send_count', Uint32) + expert_u, Uint32(1))
                destination = Uint64(peer_table[expert_u // self.local])
                address = Uint64(self.off_source_index) + ((Uint64(expert_u % self.local) * self.ranks + rank) * self.capacity + slot) * 4
                route_out = cell(destination, address, Uint32)
                route_out[0] = Uint32(route)
        self._dispatch_grid(head, epoch, 13, 14, True)
        if tid < self.ranks:
            wait_epoch(self._field(Uint64(peer_table[tid]), 'header', Uint32) + 1, epoch, True)
        cute.arch.sync_threads()
        alloc = cutlass.utils.SmemAllocator()
        counts = alloc.allocate_tensor(Uint32, cute.make_layout((self.ranks,)), 4)
        prefix = alloc.allocate_tensor(Uint32, cute.make_layout((self.ranks + 1,)), 4)
        total_shared = alloc.allocate_tensor(Uint32, cute.make_layout((1,)), 4)
        for expert in cutlass.range(block, self.local, self.grid):
            if tid < self.ranks:
                source_base = Uint64(peer_table[tid])
                source_count = self._field(source_base, 'send_count', Uint32)
                counts[tid] = ld_acquire_sys_u32(source_count + rank * self.local + Uint32(expert))
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
            meta_base = Uint64(self.off_route_meta) + Uint64(expert) * self.rows * 12
            weight_base = Uint64(self.off_route_weight) + Uint64(expert) * self.rows * 4
            for row in cutlass.range(tid, total, self.threads):
                source = Uint32(0)
                for candidate in cutlass.range_constexpr(1, self.ranks):
                    if row >= prefix[candidate]:
                        source = Uint32(candidate)
                source_slot = Uint32(row) - prefix[source]
                route = cell(base, Uint64(self.off_source_index) + ((Uint64(expert) * self.ranks + source) * self.capacity + source_slot) * 4, Uint32)[0]
                source_token, route_slot = route // self.topk, route % self.topk
                rank_out = cell(base, meta_base + Uint64(row * 3) * 4, Uint32)
                rank_out[0] = source
                token_out = cell(base, meta_base + Uint64(row * 3 + 1) * 4, Uint32)
                token_out[0] = source_token
                slot_out = cell(base, meta_base + Uint64(row * 3 + 2) * 4, Uint32)
                slot_out[0] = route_slot
                weight_out = cell(base, weight_base + Uint64(row) * 4, Float32)
                weight_out[0] = cell(Uint64(peer_table[source]), Uint64(self.off_topk_weights) + Uint64(route) * 4, Float32)[0]
            if cutlass.const_expr(self.early_routed_count):
                # Publish ALL route metadata writers before the queue release.
                # Count readiness is separate from X/SF input readiness.
                if tid < total:
                    fence_gpu()
                cute.arch.sync_threads()
                if tid == 0:
                    early_count_out = cell(base, Uint64(self.off_expert_count) + Uint64(expert) * 4, Uint32)
                    early_count_out[0] = total
                    if total > 0:
                        active_slot = atom_add_u32(head + 7, Uint32(1))
                        early_active_out = cell(base, Uint64(self.off_active_expert) + Uint64(active_slot) * 4, Uint32)
                        early_active_out[0] = Uint32(expert)
                        st_release_u32(self._field(base, 'active_ready', Uint32) + active_slot, epoch, False)
            cute.arch.sync_threads()
            # A warp retains one source/token pair while copying an entire
            # route, avoiding metadata reloads for every sixteen data bytes.
            for pull_row in cutlass.range(warp, total, self.threads // 32):
                source_rank = cell(base, meta_base + Uint64(pull_row * 3) * 4, Uint32)[0]
                token = cell(base, meta_base + Uint64(pull_row * 3 + 1) * 4, Uint32)[0]
                remote = Uint64(peer_table[source_rank])
                src = remote + Uint64(self.off_x) + Uint64(token) * self.hidden
                dst = base + Uint64(self.off_l1_acts) + (Uint64(expert) * self.rows + Uint32(pull_row)) * self.hidden
                if cutlass.const_expr(self.paired_routed_copy):
                    for k in cutlass.range(lane, (self.hidden // 1024) * 64, 64):
                        copy_global_2x16(dst + Uint64(k) * 16, src + Uint64(k) * 16)
                    for k in cutlass.range((self.hidden // 1024) * 64 + lane,
                                           self.hidden // 16, 32):
                        copy_global_16(dst + Uint64(k) * 16, src + Uint64(k) * 16)
                else:
                    for k in cutlass.range(lane, self.hidden // 16, 32):
                        copy_global_16(dst + Uint64(k) * 16, src + Uint64(k) * 16)
                for sf_k in cutlass.range(lane, self.hidden // 128, 32):
                    sf_value = cell(remote, Uint64(self.off_x_sf) + (Uint64(token) * (self.hidden // 128) + Uint32(sf_k)) * 4, Float32)[0]
                    sf_out = cell(base, Uint64(self.off_l1_acts_sf) + ((Uint64(expert) * (self.hidden // 128) + Uint32(sf_k)) * self.rows + Uint32(pull_row)) * 4, Float32)
                    if cutlass.const_expr(self.preweighted_routed_sf):
                        secondary_value = cute.make_tensor(secondary, cute.make_layout((self.local,)))[Uint32(expert)]
                        sf_out[0] = preweight_routed_sf(sf_value, secondary_value)
                    else:
                        sf_out[0] = sf_value * Float32(64)
            fence_gpu()
            cute.arch.sync_threads()
            if tid == 0:
                if cutlass.const_expr(not self.early_routed_count):
                    count_out = cell(base, Uint64(self.off_expert_count) + Uint64(expert) * 4, Uint32)
                    count_out[0] = total
                if cutlass.const_expr(self.has_stats):
                    recv_stats = cute.make_tensor(stats, cute.make_layout((self.local,)))
                    recv_stats[expert] += total
                st_release_u32(self._field(base, 'expert_ready', Uint32) + Uint32(expert), epoch, False)
                if cutlass.const_expr(not self.early_routed_count):
                    if total > 0:
                        active_slot = atom_add_u32(head + 7, Uint32(1))
                        active_out = cell(base, Uint64(self.off_active_expert) + Uint64(active_slot) * 4, Uint32)
                        active_out[0] = Uint32(expert)
                        st_release_u32(self._field(base, 'active_ready', Uint32) + active_slot, epoch, False)
                finished = atom_release_gpu_add_u32(head + 8, Uint32(1))
                if finished == self.local - 1:
                    st_release_u32(head + 9, epoch, False)
            cute.arch.sync_threads()


    @cute.jit
    def _dispatch_direct(self, base: Uint64, peers: cute.Pointer, rank: Uint32,
                          tokens: Uint32, stats: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        expert, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
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
        if tid < self.ranks:
            remote = Uint64(peer_table[tid])
            wait_epoch(self._field(remote, 'header', Uint32) + 1, epoch, True)
            counts[tid] = cell(remote, Uint64(8), Uint32)[0] * self.topk
        cute.arch.sync_threads()
        if tid == 0:
            running_count = Uint32(0)
            prefix[0] = Uint32(0)
            for prefix_source in cutlass.range_constexpr(self.ranks):
                running_count += counts[prefix_source]
                prefix[prefix_source + 1] = running_count
            route_count[0] = Uint32(0)
        cute.arch.sync_threads()
        meta_base = Uint64(self.off_route_meta) + Uint64(expert) * self.rows * 12
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
                scan_out3 = cell(base, meta_base + Uint64(row * 3) * 4, Uint32)
                scan_out3[0] = source
                scan_out4 = cell(base, meta_base + Uint64(row * 3 + 1) * 4, Uint32)
                scan_out4[0] = source_route // self.topk
                scan_out5 = cell(base, meta_base + Uint64(row * 3 + 2) * 4, Uint32)
                scan_out5[0] = source_route % self.topk
                scan_out6 = cell(base, weight_base + Uint64(row) * 4, Float32)
                scan_out6[0] = cell(remote, Uint64(self.off_topk_weights) + Uint64(source_route) * 4, Float32)[0]
        cute.arch.sync_threads()
        total = route_count[0]
        # A warp retains one source/token pair while copying an entire
        # route, avoiding metadata reloads for every sixteen data bytes.
        for pull_row in cutlass.range(warp, total, self.threads // 32):
            source_rank = cell(base, meta_base + Uint64(pull_row * 3) * 4, Uint32)[0]
            token = cell(base, meta_base + Uint64(pull_row * 3 + 1) * 4, Uint32)[0]
            remote = Uint64(peer_table[source_rank])
            src = remote + Uint64(self.off_x) + Uint64(token) * self.hidden
            dst = base + Uint64(self.off_l1_acts) + (Uint64(expert) * self.rows + Uint32(pull_row)) * self.hidden
            if cutlass.const_expr(self.paired_routed_copy):
                for k in cutlass.range(lane, (self.hidden // 1024) * 64, 64):
                    copy_global_2x16(dst + Uint64(k) * 16, src + Uint64(k) * 16)
                for k in cutlass.range((self.hidden // 1024) * 64 + lane,
                                       self.hidden // 16, 32):
                    copy_global_16(dst + Uint64(k) * 16, src + Uint64(k) * 16)
            else:
                for k in cutlass.range(lane, self.hidden // 16, 32):
                    copy_global_16(dst + Uint64(k) * 16, src + Uint64(k) * 16)
            for sf_k in cutlass.range(lane, self.hidden // 128, 32):
                sf_value = cell(remote, Uint64(self.off_x_sf) + (Uint64(token) * (self.hidden // 128) + Uint32(sf_k)) * 4, Float32)[0]
                sf_out = cell(base, Uint64(self.off_l1_acts_sf) + ((Uint64(expert) * (self.hidden // 128) + Uint32(sf_k)) * self.rows + Uint32(pull_row)) * 4, Float32)
                # Direct M8 stores SF * 64; secondary scaling remains in compute.
                sf_out[0] = sf_value * Float32(64)
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
            finished = atom_release_gpu_add_u32(head + 8, Uint32(1))
            if finished == self.local - 1:
                st_release_u32(head + 9, epoch, False)
        cute.arch.sync_threads()



    @cute.jit
    def _stage(self, base: Uint64, slab: cute.Pointer, expert: Uint32,
               ntile: Uint32, start: Uint32, valid: Uint32, kb: Uint32,
               stage: Uint32, weights: cute.Pointer, offsets: cute.Pointer,
               n: cutlass.Constexpr, l1: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        k_size = self.hidden if l1 else self.intermediate
        n_size = 2 * self.intermediate if l1 else self.hidden
        input_offset = self.off_l1_acts if l1 else self.off_l2_acts
        stage_base = stage * self.stage_bytes
        if tid == 0:
            bar = cute.recast_ptr(slab + self.smem - 128 + stage * 8, dtype=Uint64)
            cute.arch.mbarrier_arrive_and_expect_tx(bar, 16896 * self.routed_n_factor)
            for sub in cutlass.range_constexpr(self.routed_n_factor):
                index = (Uint64(expert) * (n_size // 256) + ntile * self.routed_n_factor + sub) * (k_size // 128) + kb
                if cutlass.const_expr(self.routed_n_factor == 1):
                    bulk_g2s_1d(slab + stage_base, weights + index * 16896, Uint32(16896), bar)
                else:
                    # N512 stages both record payloads before their exponent blocks.
                    bulk_g2s_1d(slab + stage_base + sub * 16384, weights + index * 16896, Uint32(16384), bar)
                    bulk_g2s_1d(slab + stage_base + 16384 * self.routed_n_factor + sub * 512, offsets + index * 16896, Uint32(512), bar)
        for j in cutlass.range(tid, n * 8, 256):
            row, k = Uint32(j // 8), Uint32((j % 8) * 16)
            size = Uint32(0)
            if row < valid:
                size = Uint32(16)
            address = base + Uint64(input_offset) + ((Uint64(expert) * self.rows + start + row) * k_size + kb * 128 + k)
            src = cute.make_ptr(Uint8, address, cute.AddressSpace.gmem, assumed_align=16)
            copy_async_16(slab + stage_base + self.routed_x_offset + canonical_offset(row, k), src, size)
        if tid < n:
            sf_size = Uint32(0)
            if tid < valid:
                sf_size = Uint32(4)
            sf_offset = self.off_l1_acts_sf if l1 else self.off_l2_acts_sf
            sf_address = base + Uint64(sf_offset) + ((Uint64(expert) * (k_size // 128) + kb) * self.rows + start + tid) * 4
            sf_src = cute.make_ptr(Uint8, sf_address, cute.AddressSpace.gmem, assumed_align=4)
            copy_async_4(slab + stage_base + self.sf_stage_offset + tid * 4, sf_src, sf_size)
        copy_commit()

    @cute.jit
    def _tile_sync(self, base: Uint64, peers: cute.Pointer, slab: cute.Pointer,
               expert: Uint32, ntile: Uint32, start: Uint32, valid: Uint32,
               weights: cute.Pointer, offsets: cute.Pointer, secondary: cute.Pointer,
               clamp: Float32, n: cutlass.Constexpr, l1: cutlass.Constexpr):
        k_size = self.hidden if l1 else self.intermediate
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, (tid // 32) % 4
        wg = tid // 128
        if tid == 0:
            cute.arch.mbarrier_init(cute.recast_ptr(slab + self.smem - 128, dtype=Uint64), 1)
            cute.arch.mbarrier_init(cute.recast_ptr(slab + self.smem - 120, dtype=Uint64), 1)
        cute.arch.sync_threads()
        cute.arch.mbarrier_init_fence()
        final = cute.make_rmem_tensor((2 * self.routed_n_factor, n // 2), Float32)
        final.fill(Float32(0))
        secondary_value = cute.make_tensor(secondary, cute.make_layout((self.local,)))[expert]
        self._stage(base, slab, expert, ntile, start, valid, Uint32(0), Uint32(0), weights, offsets, n, l1)
        for kb in cutlass.range(k_size // 128):
            stage = Uint32(kb % 2)
            if tid == 0:
                cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.smem - 128 + stage * 8, dtype=Uint64), Uint32((kb // 2) % 2))
            copy_wait()
            cute.arch.sync_threads()
            tma_store_fence()
            if kb + 1 < k_size // 128:
                self._stage(base, slab, expert, ntile, start, valid, Uint32(kb + 1), stage ^ Uint32(1), weights, offsets, n, l1)
            scales = cute.make_rmem_tensor((n // 4,), Float32)
            for group in cutlass.range_constexpr(n // 8):
                token = Uint32(group * 8 + 2 * (lane % 4))
                raw_sf = load_scale_pair(slab + stage * self.stage_bytes + Uint32(self.sf_stage_offset) + token * 4)
                scale0 = raw_sf[0] * secondary_value
                scale1 = raw_sf[1] * secondary_value
                scales[group * 2], scales[group * 2 + 1] = scale0, scale1
            for h in cutlass.range_constexpr(2 * self.routed_n_factor):
                weight_half = Uint32(wg * 2 * self.routed_n_factor + h)
                row = Uint32(warp * 16 + lane // 4)
                word_ptr = slab + stage * self.stage_bytes + weight_half * 4096 + Uint32(warp * 1024 + lane * 16)
                scale_ptr = slab + stage * self.stage_bytes + Uint32(16384 * self.routed_n_factor) + weight_half * 128 + Uint32(warp * 32 + (lane // 4) * 4)
                frag = decode_fragment_nib4(word_ptr, scale_ptr)
                part = wgmma_rs_k128(frag, smem_descriptor(slab + stage * self.stage_bytes + self.routed_x_offset), n)
                for group in cutlass.range_constexpr(n // 8):
                    final[h, group * 4] = cute.math.fma(part[group * 4], scales[group * 2], final[h, group * 4])
                    final[h, group * 4 + 1] = cute.math.fma(part[group * 4 + 1], scales[group * 2 + 1], final[h, group * 4 + 1])
                    final[h, group * 4 + 2] = cute.math.fma(part[group * 4 + 2], scales[group * 2], final[h, group * 4 + 2])
                    final[h, group * 4 + 3] = cute.math.fma(part[group * 4 + 3], scales[group * 2 + 1], final[h, group * 4 + 3])
        cute.arch.sync_threads()
        for h in cutlass.range_constexpr(2 * self.routed_n_factor):
            output_half = Uint32(wg * 2 * self.routed_n_factor + h)
            for group in cutlass.range_constexpr(n // 8):
                token = Uint32(group * 8 + 2 * (lane % 4))
                row = Uint32(warp * 16 + lane // 4)
                skew = Uint32((token % 8) // 2 * 8)
                # L1 needs row XOR only; L2 also permutes vector chunks.
                if cutlass.const_expr(l1):
                    shared_cell(slab, (token * (256 * self.routed_n_factor) + ((output_half * 64 + row) ^ skew)) * 4, Float32)[0] = final[h, group * 4]
                    shared_cell(slab, ((token + 1) * (256 * self.routed_n_factor) + ((output_half * 64 + row) ^ skew)) * 4, Float32)[0] = final[h, group * 4 + 1]
                    shared_cell(slab, (token * (256 * self.routed_n_factor) + ((output_half * 64 + row + 8) ^ skew)) * 4, Float32)[0] = final[h, group * 4 + 2]
                    shared_cell(slab, ((token + 1) * (256 * self.routed_n_factor) + ((output_half * 64 + row + 8) ^ skew)) * 4, Float32)[0] = final[h, group * 4 + 3]
                else:
                    shared_cell(slab, (token * (256 * self.routed_n_factor) + ((output_half * 64 + row) ^ skew ^ (((output_half * 64 + row) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4]
                    shared_cell(slab, ((token + 1) * (256 * self.routed_n_factor) + ((output_half * 64 + row) ^ skew ^ (((output_half * 64 + row) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4 + 1]
                    shared_cell(slab, (token * (256 * self.routed_n_factor) + ((output_half * 64 + row + 8) ^ skew ^ (((output_half * 64 + row + 8) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4 + 2]
                    shared_cell(slab, ((token + 1) * (256 * self.routed_n_factor) + ((output_half * 64 + row + 8) ^ skew ^ (((output_half * 64 + row + 8) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4 + 3]
        cute.arch.sync_threads()
        if cutlass.const_expr(l1):
            for ep_token in cutlass.range(tid // 32, valid, 8):
                for sub in cutlass.range_constexpr(self.routed_n_factor):
                    values = cute.make_rmem_tensor((4,), Float32)
                    amax = Float32(0)
                    weight = cell(base, Uint64(self.off_route_weight) + (Uint64(expert) * self.rows + start + ep_token) * 4, Float32)[0]
                    skew = Uint32((ep_token % 8) // 2 * 8)
                    raw = load_shared_8(slab + ep_token * (1024 * self.routed_n_factor) + sub * 1024 + ((lane * 4) ^ skew) * 4, True)
                    for ep_j in cutlass.range_constexpr(4):
                        gate = cute.math.min(raw[ep_j], clamp)
                        up = cute.math.min(cute.math.max(raw[ep_j + 4], -clamp), clamp)
                        sigmoid = cute.math.rcp(Float32(1) + cute.math.exp(-gate, fastmath=self.fast_math))
                        values[ep_j] = gate * sigmoid * up * weight
                        amax = cute.math.max(amax, abs_f32(values[ep_j]))
                    for shift in cutlass.range_constexpr(5):
                        amax = cute.math.max(amax, cute.arch.shuffle_sync_bfly(amax, 16 >> shift))
                    scale = cute.math.max(amax, Float32(1e-10)) * Float32(1.0 / 448.0)
                    inverse = cute.math.rcp(scale)
                    if lane == 0:
                        activation_scale_out = cell(base, Uint64(self.off_l2_acts_sf) + ((Uint64(expert) * (self.intermediate // 128) + ntile * self.routed_n_factor + sub) * self.rows + start + ep_token) * 4, Float32)
                        activation_scale_out[0] = scale * Float32(64)
                    low = f32x2_to_e4m3x2_bits(values[0] * inverse, values[1] * inverse)
                    high = f32x2_to_e4m3x2_bits(values[2] * inverse, values[3] * inverse)
                    activation_out = cell(base, Uint64(self.off_l2_acts) + (Uint64(expert) * self.rows + start + ep_token) * self.intermediate + ntile * (128 * self.routed_n_factor) + sub * 128 + lane * 4, Uint32)
                    activation_out[0] = Uint32(low) | (Uint32(high) << 16)
        else:
            peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
            for ep_j in cutlass.range(tid, valid * (32 * self.routed_n_factor), 256):
                ep_token, ep_col = ep_j // (32 * self.routed_n_factor), (ep_j % (32 * self.routed_n_factor)) * 8
                meta = Uint64(self.off_route_meta) + (Uint64(expert) * self.rows + start + ep_token) * 12
                source = cell(base, meta, Uint32)[0]
                source_token = cell(base, meta + 4, Uint32)[0]
                slot = cell(base, meta + 8, Uint32)[0]
                remote = Uint64(peer_table[source])
                skew = Uint32((ep_token % 8) // 2 * 8)
                # DeepGEMM-style column-bit swizzle: address each LDS128
                # independently; the logical eight values and STG128 stay intact.
                values_lo = load_shared_4(slab + ep_token * (1024 * self.routed_n_factor) + (ep_col ^ skew ^ ((ep_col & 32) >> 3)) * 4)
                values_hi = load_shared_4(slab + ep_token * (1024 * self.routed_n_factor) + ((ep_col + 4) ^ skew ^ (((ep_col + 4) & 32) >> 3)) * 4)
                values8 = (*values_lo, *values_hi)
                destination = remote + Uint64(self.off_combine) + ((Uint64(source_token) * self.topk + slot) * self.hidden + ntile * (256 * self.routed_n_factor) + ep_col) * 2
                store_bf16_8(destination, values8)
        cute.arch.sync_threads()

    @cute.jit
    def _stage_weights(self, slab: cute.Pointer, expert: Uint32, ntile: Uint32,
                        kb: Uint32, stage: Uint32, weights: cute.Pointer,
                        l1: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        if tid == (0 if self.max_m == 64 else 64):
            k_size = self.hidden if l1 else self.intermediate
            n_size = 2 * self.intermediate if l1 else self.hidden
            stage_base = stage * self.stage_bytes
            full = cute.recast_ptr(slab + self.full_offset + stage * 8, dtype=Uint64)
            if cutlass.const_expr(self.routed_k_pack == 2):
                # One bulk request carries both adjacent K128 records: the
                # second record's payload and exponents stay contiguous.
                cute.arch.mbarrier_arrive_and_expect_tx(full, 33792)
                index = ((Uint64(expert) * (n_size // 256) + ntile) * (k_size // 128)
                         + Uint64(kb) * 2)
                bulk_g2s_1d(slab + stage_base, weights + index * 16896,
                            Uint32(33792), full)
            else:
                cute.arch.mbarrier_arrive_and_expect_tx(full, 16896)
                index = (Uint64(expert) * (n_size // 256) + ntile) * (k_size // 128) + kb
                bulk_g2s_1d(slab + stage_base, weights + index * 16896, Uint32(16896), full)

    @cute.jit
    def _stage_activation(self, base: Uint64, slab: cute.Pointer, expert: Uint32,
                           start: Uint32, valid: Uint32, kb: Uint32,
                           stage: Uint32, n: Uint32, l1: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        k_size = self.hidden if l1 else self.intermediate
        input_offset = self.off_l1_acts if l1 else self.off_l2_acts
        stage_base = stage * self.stage_bytes
        full = cute.recast_ptr(slab + self.full_offset + stage * 8, dtype=Uint64)
        if cutlass.const_expr(self.max_m == 64):
            # Fill the reserved 8 KB activation tile in two 4 KB row halves,
            # each with its own commit group; rows beyond valid are zero-filled.
            for half in cutlass.range_constexpr(2):
                for step in cutlass.range_constexpr(2):
                    j = tid + Uint32(half * 256 + step * 128)
                    row, k = Uint32(j // 8), Uint32((j % 8) * 16)
                    size = Uint32(0)
                    if row < valid:
                        size = Uint32(16)
                    address = base + Uint64(input_offset) + ((Uint64(expert) * self.rows + start + row) * k_size + kb * 128 + k)
                    src = cute.make_ptr(Uint8, address, cute.AddressSpace.gmem, assumed_align=16)
                    copy_async_16(slab + stage_base + self.routed_x_offset + canonical_offset(row, k), src, size)
                if cutlass.const_expr(half == 0):
                    if tid < n:
                        sf_size = Uint32(0)
                        if tid < valid:
                            sf_size = Uint32(4)
                        sf_offset = self.off_l1_acts_sf if l1 else self.off_l2_acts_sf
                        sf_address = base + Uint64(sf_offset) + ((Uint64(expert) * (k_size // 128) + kb) * self.rows + start + tid) * 4
                        sf_src = cute.make_ptr(Uint8, sf_address, cute.AddressSpace.gmem, assumed_align=4)
                        copy_async_4(slab + stage_base + self.sf_stage_offset + tid * 4, sf_src, sf_size)
                copy_commit()
                copy_arrive(full)
        else:
            if cutlass.const_expr(self.routed_k_pack == 2):
                # Both K128 halves share one commit group and one arrival, so
                # the stage barrier still completes exactly once per stage.
                for sub in cutlass.range_constexpr(2):
                    for j in cutlass.range(tid, n * 8, 64):
                        row, k = Uint32(j // 8), Uint32((j % 8) * 16)
                        size = Uint32(0)
                        if row < valid:
                            size = Uint32(16)
                        address = base + Uint64(input_offset) + ((Uint64(expert) * self.rows + start + row) * k_size + (kb * Uint32(2) + Uint32(sub)) * 128 + k)
                        src = cute.make_ptr(Uint8, address, cute.AddressSpace.gmem, assumed_align=16)
                        copy_async_16(slab + stage_base + self.routed_x_offset + sub * (self.max_m * 128) + canonical_offset(row, k), src, size)
                    if tid < n:
                        sf_size = Uint32(0)
                        if tid < valid:
                            sf_size = Uint32(4)
                        sf_offset = self.off_l1_acts_sf if l1 else self.off_l2_acts_sf
                        sf_address = base + Uint64(sf_offset) + ((Uint64(expert) * (k_size // 128) + kb * Uint32(2) + Uint32(sub)) * self.rows + start + tid) * 4
                        sf_src = cute.make_ptr(Uint8, sf_address, cute.AddressSpace.gmem, assumed_align=4)
                        copy_async_4(slab + stage_base + self.sf_stage_offset + sub * (self.max_m * 4) + tid * 4, sf_src, sf_size)
                copy_commit()
                copy_arrive(full)
            else:
                for j in cutlass.range(tid, n * 8, 64):
                    row, k = Uint32(j // 8), Uint32((j % 8) * 16)
                    size = Uint32(0)
                    if row < valid:
                        size = Uint32(16)
                    address = base + Uint64(input_offset) + ((Uint64(expert) * self.rows + start + row) * k_size + kb * 128 + k)
                    src = cute.make_ptr(Uint8, address, cute.AddressSpace.gmem, assumed_align=16)
                    copy_async_16(slab + stage_base + self.routed_x_offset + canonical_offset(row, k), src, size)
                if tid < n:
                    sf_size = Uint32(0)
                    if tid < valid:
                        sf_size = Uint32(4)
                    sf_offset = self.off_l1_acts_sf if l1 else self.off_l2_acts_sf
                    sf_address = base + Uint64(sf_offset) + ((Uint64(expert) * (k_size // 128) + kb) * self.rows + start + tid) * 4
                    sf_src = cute.make_ptr(Uint8, sf_address, cute.AddressSpace.gmem, assumed_align=4)
                    copy_async_4(slab + stage_base + self.sf_stage_offset + tid * 4, sf_src, sf_size)
                copy_commit()
                copy_arrive(full)

    @cute.jit
    def _bucket_tile(self, base: Uint64, peers: cute.Pointer, slab: cute.Pointer,
                      expert: Uint32, ntile: Uint32, start: Uint32, valid: Uint32,
                      weights: cute.Pointer, offsets: cute.Pointer, secondary: cute.Pointer,
                      clamp: Float32, l1: cutlass.Constexpr):
        if cutlass.const_expr(self.max_m == 8):
            self._tile_sync(base, peers, slab, expert, ntile, start, valid,
                       weights, offsets, secondary, clamp, 8, l1)
        else:
            if valid <= 8:
                self._tile_sync(base, peers, slab, expert, ntile, start, valid,
                           weights, offsets, secondary, clamp, 8, l1)
            else:
                if cutlass.const_expr(self.max_m == 16):
                    self._tile_sync(base, peers, slab, expert, ntile, start, valid,
                               weights, offsets, secondary, clamp, 16, l1)
                elif cutlass.const_expr(self.max_m == 24):
                    if valid <= 8:
                        self._tile_sync(base, peers, slab, expert, ntile, start, valid,
                                   weights, offsets, secondary, clamp, 8, l1)
                    elif valid <= 16:
                        self._tile_sync(base, peers, slab, expert, ntile, start, valid,
                                   weights, offsets, secondary, clamp, 16, l1)
                    else:
                        self._tile_sync(base, peers, slab, expert, ntile, start, valid,
                                   weights, offsets, secondary, clamp, 24, l1)
                else:
                    if valid <= 16:
                        self._tile_sync(base, peers, slab, expert, ntile, start, valid,
                                   weights, offsets, secondary, clamp, 16, l1)
                    elif valid <= 32:
                        self._tile_sync(base, peers, slab, expert, ntile, start, valid,
                                   weights, offsets, secondary, clamp, 32, l1)
                    else:
                        self._tile_sync(base, peers, slab, expert, ntile, start, valid,
                                   weights, offsets, secondary, clamp, 64, l1)

    @cute.jit
    def _claim_command(self, base: Uint64, command: cute.Tensor,
                         epoch: Uint32, is_l1: Boolean):
        tid, _, _ = cute.arch.thread_idx()
        head = self._field(base, 'header', Uint32)
        active_ptr = self._field(base, 'active_ready', Uint32)
        expert_ptr = self._field(base, 'expert_ready', Uint32)
        l1_ptr = self._field(base, 'l1_done', Uint32)
        if tid == 0:
            task, active_slot, ntile = Uint32(0), Uint32(0), Uint32(0)
            if is_l1:
                task = claim_sm90_task(head + 3)
                active_slot, ntile = task // Uint32(self.intermediate // (128 * self.routed_n_factor)), task % Uint32(self.intermediate // (128 * self.routed_n_factor))
            else:
                task = claim_sm90_task(head + 10)
                active_slot, ntile = task // Uint32(self.hidden // (256 * self.routed_n_factor)), task % Uint32(self.hidden // (256 * self.routed_n_factor))
            active_count = ld_acquire_gpu_u32(head + 7)
            if is_l1:
                dispatch_done = ld_acquire_gpu_u32(head + 9)
                while active_slot >= active_count and dispatch_done != epoch:
                    active_count = ld_acquire_gpu_u32(head + 7)
                    dispatch_done = ld_acquire_gpu_u32(head + 9)
                active_count = ld_acquire_gpu_u32(head + 7)
            command[0] = Uint32(0)
            if active_slot < active_count:
                wait_epoch(active_ptr + active_slot, epoch, False)
                expert = cell(base, Uint64(self.off_active_expert) + Uint64(active_slot) * 4, Uint32)[0]
                if cutlass.const_expr(self.early_routed_count):
                    # L1 metadata/count visibility comes from active_ready,
                    # then loader sync and task-record release/acquire.
                    if not is_l1:
                        wait_epoch(expert_ptr + expert, epoch, False)
                else:
                    wait_epoch(expert_ptr + expert, epoch, False)
                if not is_l1:
                    wait_count(l1_ptr + expert, Uint32(self.intermediate // 128))
                command[0] = Uint32(1)
                command[1] = Uint32(expert)
                command[2] = Uint32(ntile)

    @cute.jit
    def _publish_task(self, slab: cute.Pointer, task_iter: Uint32,
                       phase: Uint32, expert: Uint32, ntile: Uint32,
                       start: Uint32, valid: Uint32, last: Uint32):
        tid, _, _ = cute.arch.thread_idx()
        slot = task_iter % 2
        if tid == 0 and task_iter >= 2:
            cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.task_empty_offset + slot * 8, dtype=Uint64), Uint32((task_iter // 2 - 1) % 2))
        sync_loader()
        if tid == 0:
            record = slab + self.task_record_offset + slot * 32
            shared_cell(record, Uint32(0), Uint32)[0] = phase
            shared_cell(record, Uint32(4), Uint32)[0] = expert
            shared_cell(record, Uint32(8), Uint32)[0] = ntile
            shared_cell(record, Uint32(12), Uint32)[0] = start
            shared_cell(record, Uint32(16), Uint32)[0] = valid
            shared_cell(record, Uint32(20), Uint32)[0] = last
            cute.arch.mbarrier_arrive(cute.recast_ptr(slab + self.task_full_offset + slot * 8, dtype=Uint64))
        sync_loader()

    @cute.jit
    def _load_persistent_tile(self, base: Uint64, slab: cute.Pointer,
                               data_iter: Uint32, expert: Uint32, ntile: Uint32,
                               start: Uint32, valid: Uint32, weights: cute.Pointer,
                               epoch: Uint32, l1: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        k_size = self.hidden if l1 else self.intermediate
        load_m = Uint32(64)
        if valid <= 8:
            load_m = Uint32(8)
        elif valid <= 16:
            load_m = Uint32(16)
        elif valid <= 24 and self.max_m == 24:
            load_m = Uint32(24)
        elif valid <= 32:
            load_m = Uint32(32)
        elif valid <= 40:
            load_m = Uint32(40)
        elif valid <= 48:
            load_m = Uint32(48)
        if cutlass.const_expr(self.max_m == 64):
            for kb in cutlass.range(k_size // 128):
                iteration = data_iter + Uint32(kb)
                stage = iteration % self.stages
                if tid == 0 and iteration >= self.stages:
                    cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.empty_offset + stage * 8, dtype=Uint64), Uint32((iteration // self.stages - 1) % 2))
                sync_loader()
                self._stage_weights(slab, expert, ntile, Uint32(kb), stage, weights, l1)
                if cutlass.const_expr(self.early_routed_count and l1):
                    # M64 keeps one128thread loader. Only first W can overlap
                    # input publication; no whole independent W-stream claim.
                    if kb == 0:
                        if tid == 0:
                            wait_epoch(self._field(base, 'expert_ready', Uint32) + expert, epoch, False)
                        sync_loader()
                self._stage_activation(base, slab, expert, start, valid, Uint32(kb), stage, load_m, l1)
        else:
            if cutlass.const_expr(self.routed_k_pack == 2):
                # One stage per K256 block: the ring advances by one slot per
                # pair of K128 records (per-tile increments stay even).
                blocks = k_size // 256
                iter_base = data_iter // 2
            else:
                blocks = k_size // 128
                iter_base = data_iter
            if tid < 64:
                if cutlass.const_expr(self.early_routed_count and l1):
                    if tid == 0:
                        wait_epoch(self._field(base, 'expert_ready', Uint32) + expert, epoch, False)
                    sync_activation()
                for kb in cutlass.range(blocks):
                    iteration = iter_base + Uint32(kb)
                    stage = iteration % self.stages
                    if tid == 0 and iteration >= self.stages:
                        cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.empty_offset + stage * 8, dtype=Uint64), Uint32((iteration // self.stages - 1) % 2))
                    sync_activation()
                    self._stage_activation(base, slab, expert, start, valid, Uint32(kb), stage, load_m, l1)
            else:
                for weight_kb in cutlass.range(blocks):
                    weight_iter = iter_base + Uint32(weight_kb)
                    weight_stage = weight_iter % self.stages
                    if tid == 64 and weight_iter >= self.stages:
                        cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.empty_offset + weight_stage * 8, dtype=Uint64), Uint32((weight_iter // self.stages - 1) % 2))
                    sync_weight()
                    self._stage_weights(slab, expert, ntile, Uint32(weight_kb), weight_stage, weights, l1)
        sync_loader()
        return data_iter + Uint32(k_size // 128)

    @cute.jit
    def _produce_tasks(self, base: Uint64, slab: cute.Pointer, epoch: Uint32,
                        w1: cute.Pointer, w2: cute.Pointer):
        command = cute.make_tensor(cute.recast_ptr(slab + self.command_offset, dtype=Uint32), cute.make_layout((3,)))
        is_l1, running = True, True
        task_iter, data_iter = Uint32(0), Uint32(0)
        while running:
            self._claim_command(base, command, epoch, is_l1)
            sync_loader()
            if command[0] == 0:
                if is_l1:
                    is_l1 = False
                else:
                    running = False
            else:
                expert = Uint32(command[1])
                ntile = Uint32(command[2])
                count = cell(base, Uint64(self.off_expert_count) + Uint64(expert) * 4, Uint32)[0]
                for start in cutlass.range(0, count, self.max_m):
                    valid = cute.math.min(Uint32(self.max_m), count - Uint32(start))
                    last = Uint32(0)
                    if Uint32(start) + valid == count:
                        last = Uint32(1)
                    phase = Uint32(1 if is_l1 else 2)
                    self._publish_task(slab, task_iter, phase, expert, ntile,
                                       Uint32(start), valid, last)
                    task_iter += Uint32(1)
                    if is_l1:
                        data_iter = self._load_persistent_tile(base, slab, data_iter,
                                      expert, ntile, Uint32(start), valid, w1, epoch, True)
                    else:
                        data_iter = self._load_persistent_tile(base, slab, data_iter,
                                      expert, ntile, Uint32(start), valid, w2, epoch, False)
        self._publish_task(slab, task_iter, Uint32(0), Uint32(0), Uint32(0),
                           Uint32(0), Uint32(0), Uint32(0))

    @cute.jit
    def _math_persistent_tile(self, base: Uint64, peers: cute.Pointer, slab: cute.Pointer,
                               data_iter: Uint32, expert: Uint32, ntile: Uint32,
                               start: Uint32, valid: Uint32, secondary: cute.Pointer, output_secondary: cute.Pointer,
                               clamp: Float32, n: cutlass.Constexpr, l1: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        k_size = self.hidden if l1 else self.intermediate
        epi_slab = slab + (self.full_offset + 1024)
        math_tid = tid - 128
        lane, warp = math_tid % 32, (math_tid // 32) % 4
        wg = math_tid // 128
        final = cute.make_rmem_tensor((2, n // 2), Float32)
        final.fill(Float32(0))
        secondary_value = Float32(1)
        if cutlass.const_expr(not self.preweighted_routed_sf):
            secondary_value = cute.make_tensor(secondary, cute.make_layout((self.local,)))[expert]
        row = Uint32(warp * 16 + lane // 4)
        if cutlass.const_expr(self.routed_k_pack == 2):
            # One stage holds two adjacent K128 records: the slot advances on
            # even K128 indices, the second record drains it. The full wait and
            # the empty release follow exactly those two positions.
            first_iter = data_iter // 2
            first_slot = Uint32(first_iter % self.stages)
            if lane == 0:
                cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.full_offset + first_slot * 8, dtype=Uint64), Uint32((first_iter // self.stages) % 2))
            cute.arch.sync_warp()
            tma_store_fence()
            initial_word = slab + first_slot * self.stage_bytes + Uint32(wg * 8192 + warp * 1024 + lane * 16)
            initial_scale = slab + first_slot * self.stage_bytes + Uint32(16384 + wg * 256 + warp * 32 + (lane // 4) * 4)
            frag0 = decode_fragment_nib4(initial_word, initial_scale)
            for kb in cutlass.range(k_size // 128):
                iteration = data_iter + Uint32(kb)
                slot = Uint32(iteration // 2 % self.stages)
                half = Uint32(kb % 2)
                stage_base = slab + slot * self.stage_bytes
                descriptor = smem_descriptor(stage_base + self.routed_x_offset + half * Uint32(self.max_m * 128))
                raw0 = wgmma_rs_issue(frag0, descriptor, n)
                word_ptr = stage_base + half * Uint32(16896) + Uint32(wg * 8192 + warp * 1024 + lane * 16)
                scale_ptr = stage_base + half * Uint32(16896) + Uint32(16384 + wg * 256 + warp * 32 + (lane // 4) * 4)
                frag1 = decode_fragment_nib4(word_ptr + 4096, scale_ptr + 128)
                scales = cute.make_rmem_tensor((n // 4,), Float32)
                for group in cutlass.range_constexpr(n // 8):
                    token = Uint32(group * 8 + 2 * (lane % 4))
                    raw_sf = load_scale_pair(stage_base + Uint32(self.sf_stage_offset) + half * Uint32(self.max_m * 4) + token * 4)
                    if cutlass.const_expr(self.preweighted_routed_sf):
                        scale0, scale1 = raw_sf[0], raw_sf[1]
                    else:
                        scale0 = raw_sf[0] * secondary_value
                        scale1 = raw_sf[1] * secondary_value
                    scales[group * 2], scales[group * 2 + 1] = scale0, scale1
                part0 = wgmma_rs_wait(raw0)
                for group in cutlass.range_constexpr(n // 8):
                    final[0, group * 4] = cute.math.fma(part0[group * 4], scales[group * 2], final[0, group * 4])
                    final[0, group * 4 + 1] = cute.math.fma(part0[group * 4 + 1], scales[group * 2 + 1], final[0, group * 4 + 1])
                    final[0, group * 4 + 2] = cute.math.fma(part0[group * 4 + 2], scales[group * 2], final[0, group * 4 + 2])
                    final[0, group * 4 + 3] = cute.math.fma(part0[group * 4 + 3], scales[group * 2 + 1], final[0, group * 4 + 3])
                raw1 = wgmma_rs_issue(frag1, descriptor, n)
                if kb + 1 < k_size // 128:
                    next_half = Uint32((kb + 1) % 2)
                    next_iter = iteration + Uint32(1)
                    next_slot = Uint32(next_iter // 2 % self.stages)
                    if next_half == 0:
                        # First record of the next stage: its copies finish once.
                        if lane == 0:
                            cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.full_offset + next_slot * 8, dtype=Uint64), Uint32((next_iter // 2 // self.stages) % 2))
                        cute.arch.sync_warp()
                        # Activation uses non-bulk cp.async (generic proxy); publish
                        # each completed stage before WGMMA reads it in the async proxy.
                        tma_store_fence()
                    next_word = slab + next_slot * self.stage_bytes + next_half * Uint32(16896) + Uint32(wg * 8192 + warp * 1024 + lane * 16)
                    next_scale = slab + next_slot * self.stage_bytes + next_half * Uint32(16896) + Uint32(16384 + wg * 256 + warp * 32 + (lane // 4) * 4)
                    frag0 = decode_fragment_nib4(next_word, next_scale)
                part1 = wgmma_rs_wait(raw1)
                for group in cutlass.range_constexpr(n // 8):
                    final[1, group * 4] = cute.math.fma(part1[group * 4], scales[group * 2], final[1, group * 4])
                    final[1, group * 4 + 1] = cute.math.fma(part1[group * 4 + 1], scales[group * 2 + 1], final[1, group * 4 + 1])
                    final[1, group * 4 + 2] = cute.math.fma(part1[group * 4 + 2], scales[group * 2], final[1, group * 4 + 2])
                    final[1, group * 4 + 3] = cute.math.fma(part1[group * 4 + 3], scales[group * 2 + 1], final[1, group * 4 + 3])
                if half == 1:
                    # Second record drained the whole stage.
                    tma_store_fence()
                    cute.arch.sync_warp()
                    if lane == 0:
                        cute.arch.mbarrier_arrive(cute.recast_ptr(slab + self.empty_offset + slot * 8, dtype=Uint64))
        else:
            first_stage = Uint32(data_iter % self.stages)
            if lane == 0:
                cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.full_offset + first_stage * 8, dtype=Uint64), Uint32((data_iter // self.stages) % 2))
            cute.arch.sync_warp()
            tma_store_fence()
            initial_word = slab + first_stage * self.stage_bytes + Uint32(wg * 8192 + warp * 1024 + lane * 16)
            initial_scale = slab + first_stage * self.stage_bytes + Uint32(16384 + wg * 256 + warp * 32 + (lane // 4) * 4)
            frag0 = decode_fragment_nib4(initial_word, initial_scale)
            for kb in cutlass.range(k_size // 128):
                iteration = data_iter + Uint32(kb)
                stage = Uint32(iteration % self.stages)
                descriptor = smem_descriptor(slab + stage * self.stage_bytes + self.routed_x_offset)
                raw0 = wgmma_rs_issue(frag0, descriptor, n)
                word_ptr = slab + stage * self.stage_bytes + Uint32(wg * 8192 + warp * 1024 + lane * 16)
                scale_ptr = slab + stage * self.stage_bytes + Uint32(16384 + wg * 256 + warp * 32 + (lane // 4) * 4)
                frag1 = decode_fragment_nib4(word_ptr + 4096, scale_ptr + 128)
                scales = cute.make_rmem_tensor((n // 4,), Float32)
                for group in cutlass.range_constexpr(n // 8):
                    token = Uint32(group * 8 + 2 * (lane % 4))
                    raw_sf = load_scale_pair(slab + stage * self.stage_bytes + Uint32(self.sf_stage_offset) + token * 4)
                    if cutlass.const_expr(self.preweighted_routed_sf):
                        scale0, scale1 = raw_sf[0], raw_sf[1]
                    else:
                        scale0 = raw_sf[0] * secondary_value
                        scale1 = raw_sf[1] * secondary_value
                    scales[group * 2], scales[group * 2 + 1] = scale0, scale1
                part0 = wgmma_rs_wait(raw0)
                for group in cutlass.range_constexpr(n // 8):
                    final[0, group * 4] = cute.math.fma(part0[group * 4], scales[group * 2], final[0, group * 4])
                    final[0, group * 4 + 1] = cute.math.fma(part0[group * 4 + 1], scales[group * 2 + 1], final[0, group * 4 + 1])
                    final[0, group * 4 + 2] = cute.math.fma(part0[group * 4 + 2], scales[group * 2], final[0, group * 4 + 2])
                    final[0, group * 4 + 3] = cute.math.fma(part0[group * 4 + 3], scales[group * 2 + 1], final[0, group * 4 + 3])
                raw1 = wgmma_rs_issue(frag1, descriptor, n)
                if kb + 1 < k_size // 128:
                    next_stage = Uint32((iteration + 1) % self.stages)
                    if lane == 0:
                        cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.full_offset + next_stage * 8, dtype=Uint64), Uint32(((iteration + 1) // self.stages) % 2))
                    cute.arch.sync_warp()
                    # Activation uses non-bulk cp.async (generic proxy); publish
                    # each completed stage before WGMMA reads it in the async proxy.
                    tma_store_fence()
                    next_word = slab + next_stage * self.stage_bytes + Uint32(wg * 8192 + warp * 1024 + lane * 16)
                    next_scale = slab + next_stage * self.stage_bytes + Uint32(16384 + wg * 256 + warp * 32 + (lane // 4) * 4)
                    frag0 = decode_fragment_nib4(next_word, next_scale)
                part1 = wgmma_rs_wait(raw1)
                for group in cutlass.range_constexpr(n // 8):
                    final[1, group * 4] = cute.math.fma(part1[group * 4], scales[group * 2], final[1, group * 4])
                    final[1, group * 4 + 1] = cute.math.fma(part1[group * 4 + 1], scales[group * 2 + 1], final[1, group * 4 + 1])
                    final[1, group * 4 + 2] = cute.math.fma(part1[group * 4 + 2], scales[group * 2], final[1, group * 4 + 2])
                    final[1, group * 4 + 3] = cute.math.fma(part1[group * 4 + 3], scales[group * 2 + 1], final[1, group * 4 + 3])
                tma_store_fence()
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(cute.recast_ptr(slab + self.empty_offset + stage * 8, dtype=Uint64))
        # Tiles no wider than the slab keep the original epilogue verbatim;
        # only a wider tile is drained in several batches.
        if cutlass.const_expr(n <= self.epilogue_rows):
            sync_math()
            for h in cutlass.range_constexpr(2):
                output_half = Uint32(wg * 2 + h)
                for group in cutlass.range_constexpr(n // 8):
                    token = Uint32(group * 8 + 2 * (lane % 4))
                    row = Uint32(warp * 16 + lane // 4)
                    # XOR token-pair bits into columns, as in swizzled staging.
                    skew = Uint32((token % 8) // 2 * 8)
                    # L1 needs row XOR only; L2 also permutes vector chunks.
                    if cutlass.const_expr(l1):
                        shared_cell(epi_slab, (token * 256 + ((output_half * 64 + row) ^ skew)) * 4, Float32)[0] = final[h, group * 4]
                        shared_cell(epi_slab, ((token + 1) * 256 + ((output_half * 64 + row) ^ skew)) * 4, Float32)[0] = final[h, group * 4 + 1]
                        shared_cell(epi_slab, (token * 256 + ((output_half * 64 + row + 8) ^ skew)) * 4, Float32)[0] = final[h, group * 4 + 2]
                        shared_cell(epi_slab, ((token + 1) * 256 + ((output_half * 64 + row + 8) ^ skew)) * 4, Float32)[0] = final[h, group * 4 + 3]
                    else:
                        shared_cell(epi_slab, (token * 256 + ((output_half * 64 + row) ^ skew ^ (((output_half * 64 + row) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4]
                        shared_cell(epi_slab, ((token + 1) * 256 + ((output_half * 64 + row) ^ skew ^ (((output_half * 64 + row) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4 + 1]
                        shared_cell(epi_slab, (token * 256 + ((output_half * 64 + row + 8) ^ skew ^ (((output_half * 64 + row + 8) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4 + 2]
                        shared_cell(epi_slab, ((token + 1) * 256 + ((output_half * 64 + row + 8) ^ skew ^ (((output_half * 64 + row + 8) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4 + 3]
            sync_math()
            if cutlass.const_expr(l1):
                output_secondary_value = Float32(1)
                if cutlass.const_expr(self.preweighted_routed_sf):
                    output_secondary_value = cute.make_tensor(output_secondary, cute.make_layout((self.local,)))[expert]
                for ep_token in cutlass.range(math_tid // 32, valid, 8):
                    values = cute.make_rmem_tensor((4,), Float32)
                    amax = Float32(0)
                    weight = cell(base, Uint64(self.off_route_weight) + (Uint64(expert) * self.rows + start + ep_token) * 4, Float32)[0]
                    skew = Uint32((ep_token % 8) // 2 * 8)
                    raw = load_shared_8(epi_slab + ep_token * 1024 + ((lane * 4) ^ skew) * 4, True)
                    for ep_j in cutlass.range_constexpr(4):
                        gate = cute.math.min(raw[ep_j], clamp)
                        up = cute.math.min(cute.math.max(raw[ep_j + 4], -clamp), clamp)
                        sigmoid = cute.math.rcp(Float32(1) + cute.math.exp(-gate, fastmath=self.fast_math))
                        values[ep_j] = gate * sigmoid * up * weight
                        amax = cute.math.max(amax, abs_f32(values[ep_j]))
                    for shift in cutlass.range_constexpr(5):
                        amax = cute.math.max(amax, cute.arch.shuffle_sync_bfly(amax, 16 >> shift))
                    scale = cute.math.max(amax, Float32(1e-10)) * Float32(1.0 / 448.0)
                    inverse = cute.math.rcp(scale)
                    if lane == 0:
                        activation_scale_out = cell(base, Uint64(self.off_l2_acts_sf) + ((Uint64(expert) * (self.intermediate // 128) + ntile) * self.rows + start + ep_token) * 4, Float32)
                        if cutlass.const_expr(self.preweighted_routed_sf):
                            activation_scale_out[0] = preweight_routed_sf(scale, output_secondary_value)
                        else:
                            activation_scale_out[0] = scale * Float32(64)
                    low = f32x2_to_e4m3x2_bits(values[0] * inverse, values[1] * inverse)
                    high = f32x2_to_e4m3x2_bits(values[2] * inverse, values[3] * inverse)
                    activation_out = cell(base, Uint64(self.off_l2_acts) + (Uint64(expert) * self.rows + start + ep_token) * self.intermediate + ntile * 128 + lane * 4, Uint32)
                    activation_out[0] = Uint32(low) | (Uint32(high) << 16)
            else:
                peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
                for ep_j in cutlass.range(math_tid, valid * 32, 256):
                    ep_token, ep_col = ep_j // 32, (ep_j % 32) * 8
                    meta = Uint64(self.off_route_meta) + (Uint64(expert) * self.rows + start + ep_token) * 12
                    source = cell(base, meta, Uint32)[0]
                    source_token = cell(base, meta + 4, Uint32)[0]
                    slot = cell(base, meta + 8, Uint32)[0]
                    remote = Uint64(peer_table[source])
                    skew = Uint32((ep_token % 8) // 2 * 8)
                    # DeepGEMM-style column-bit swizzle: address each LDS128
                    # independently; the logical eight values and STG128 stay intact.
                    values_lo = load_shared_4(epi_slab + ep_token * 1024 + (ep_col ^ skew ^ ((ep_col & 32) >> 3)) * 4)
                    values_hi = load_shared_4(epi_slab + ep_token * 1024 + ((ep_col + 4) ^ skew ^ (((ep_col + 4) & 32) >> 3)) * 4)
                    values8 = (*values_lo, *values_hi)
                    destination = remote + Uint64(self.off_combine) + ((Uint64(source_token) * self.topk + slot) * self.hidden + ntile * 256 + ep_col) * 2
                    store_bf16_8(destination, values8)
            sync_math()
        else:
            # M32 writes and drains the epilogue slab in two 16-row batches so a
            # smaller slab can coexist with a fourth weight stage. Every row keeps
            # its values, amax reduction and store order, so results are identical.
            epi_batches = n // self.epilogue_rows
            groups_per_batch = (n // 8) // epi_batches
            for batch in cutlass.range_constexpr(epi_batches):
                batch_row = Uint32(batch * self.epilogue_rows)
                # Rows beyond `valid` stay untouched; the unsigned difference
                # wraps to a huge value when this batch is empty.
                batch_valid = valid - batch_row
                if batch_valid > Uint32(n):
                    batch_valid = Uint32(0)
                batch_valid = cute.math.min(batch_valid, Uint32(self.epilogue_rows))
                sync_math()
                for h in cutlass.range_constexpr(2):
                    output_half = Uint32(wg * 2 + h)
                    for batch_group in cutlass.range_constexpr(groups_per_batch):
                        group = batch * groups_per_batch + batch_group
                        token = Uint32(batch_group * 8 + 2 * (lane % 4))
                        row = Uint32(warp * 16 + lane // 4)
                        # XOR token-pair bits into columns, as in swizzled staging.
                        skew = Uint32((token % 8) // 2 * 8)
                        # L1 needs row XOR only; L2 also permutes vector chunks.
                        if cutlass.const_expr(l1):
                            shared_cell(epi_slab, (token * 256 + ((output_half * 64 + row) ^ skew)) * 4, Float32)[0] = final[h, group * 4]
                            shared_cell(epi_slab, ((token + 1) * 256 + ((output_half * 64 + row) ^ skew)) * 4, Float32)[0] = final[h, group * 4 + 1]
                            shared_cell(epi_slab, (token * 256 + ((output_half * 64 + row + 8) ^ skew)) * 4, Float32)[0] = final[h, group * 4 + 2]
                            shared_cell(epi_slab, ((token + 1) * 256 + ((output_half * 64 + row + 8) ^ skew)) * 4, Float32)[0] = final[h, group * 4 + 3]
                        else:
                            shared_cell(epi_slab, (token * 256 + ((output_half * 64 + row) ^ skew ^ (((output_half * 64 + row) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4]
                            shared_cell(epi_slab, ((token + 1) * 256 + ((output_half * 64 + row) ^ skew ^ (((output_half * 64 + row) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4 + 1]
                            shared_cell(epi_slab, (token * 256 + ((output_half * 64 + row + 8) ^ skew ^ (((output_half * 64 + row + 8) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4 + 2]
                            shared_cell(epi_slab, ((token + 1) * 256 + ((output_half * 64 + row + 8) ^ skew ^ (((output_half * 64 + row + 8) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4 + 3]
                sync_math()
                if cutlass.const_expr(l1):
                    output_secondary_value = Float32(1)
                    if cutlass.const_expr(self.preweighted_routed_sf):
                        output_secondary_value = cute.make_tensor(output_secondary, cute.make_layout((self.local,)))[expert]
                    for ep_local in cutlass.range(math_tid // 32, batch_valid, 8):
                        ep_token = batch_row + ep_local
                        values = cute.make_rmem_tensor((4,), Float32)
                        amax = Float32(0)
                        weight = cell(base, Uint64(self.off_route_weight) + (Uint64(expert) * self.rows + start + ep_token) * 4, Float32)[0]
                        skew = Uint32((ep_token % 8) // 2 * 8)
                        raw = load_shared_8(epi_slab + ep_local * 1024 + ((lane * 4) ^ skew) * 4, True)
                        for ep_j in cutlass.range_constexpr(4):
                            gate = cute.math.min(raw[ep_j], clamp)
                            up = cute.math.min(cute.math.max(raw[ep_j + 4], -clamp), clamp)
                            sigmoid = cute.math.rcp(Float32(1) + cute.math.exp(-gate, fastmath=self.fast_math))
                            values[ep_j] = gate * sigmoid * up * weight
                            amax = cute.math.max(amax, abs_f32(values[ep_j]))
                        for shift in cutlass.range_constexpr(5):
                            amax = cute.math.max(amax, cute.arch.shuffle_sync_bfly(amax, 16 >> shift))
                        scale = cute.math.max(amax, Float32(1e-10)) * Float32(1.0 / 448.0)
                        inverse = cute.math.rcp(scale)
                        if lane == 0:
                            activation_scale_out = cell(base, Uint64(self.off_l2_acts_sf) + ((Uint64(expert) * (self.intermediate // 128) + ntile) * self.rows + start + ep_token) * 4, Float32)
                            if cutlass.const_expr(self.preweighted_routed_sf):
                                activation_scale_out[0] = preweight_routed_sf(scale, output_secondary_value)
                            else:
                                activation_scale_out[0] = scale * Float32(64)
                        low = f32x2_to_e4m3x2_bits(values[0] * inverse, values[1] * inverse)
                        high = f32x2_to_e4m3x2_bits(values[2] * inverse, values[3] * inverse)
                        activation_out = cell(base, Uint64(self.off_l2_acts) + (Uint64(expert) * self.rows + start + ep_token) * self.intermediate + ntile * 128 + lane * 4, Uint32)
                        activation_out[0] = Uint32(low) | (Uint32(high) << 16)
                else:
                    peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
                    for ep_j in cutlass.range(math_tid, batch_valid * 32, 256):
                        ep_local, ep_col = ep_j // 32, (ep_j % 32) * 8
                        ep_token = batch_row + ep_local
                        meta = Uint64(self.off_route_meta) + (Uint64(expert) * self.rows + start + ep_token) * 12
                        source = cell(base, meta, Uint32)[0]
                        source_token = cell(base, meta + 4, Uint32)[0]
                        slot = cell(base, meta + 8, Uint32)[0]
                        remote = Uint64(peer_table[source])
                        skew = Uint32((ep_token % 8) // 2 * 8)
                        # DeepGEMM-style column-bit swizzle: address each LDS128
                        # independently; the logical eight values and STG128 stay intact.
                        values_lo = load_shared_4(epi_slab + ep_local * 1024 + (ep_col ^ skew ^ ((ep_col & 32) >> 3)) * 4)
                        values_hi = load_shared_4(epi_slab + ep_local * 1024 + ((ep_col + 4) ^ skew ^ (((ep_col + 4) & 32) >> 3)) * 4)
                        values8 = (*values_lo, *values_hi)
                        destination = remote + Uint64(self.off_combine) + ((Uint64(source_token) * self.topk + slot) * self.hidden + ntile * 256 + ep_col) * 2
                        store_bf16_8(destination, values8)
                sync_math()
        return data_iter + Uint32(k_size // 128)

    @cute.jit
    def _consume_tasks(self, base: Uint64, peers: cute.Pointer, slab: cute.Pointer,
                        sec1: cute.Pointer, sec2: cute.Pointer, clamp: Float32):
        tid, _, _ = cute.arch.thread_idx()
        math_tid, lane = tid - 128, tid % 32
        task_iter, data_iter = Uint32(0), Uint32(0)
        running = True
        while running:
            slot = task_iter % 2
            if lane == 0:
                cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.task_full_offset + slot * 8, dtype=Uint64), Uint32((task_iter // 2) % 2))
            cute.arch.sync_warp()
            record = slab + self.task_record_offset + slot * 32
            phase = Uint32(shared_cell(record, Uint32(0), Uint32)[0])
            expert = Uint32(shared_cell(record, Uint32(4), Uint32)[0])
            ntile = Uint32(shared_cell(record, Uint32(8), Uint32)[0])
            start = Uint32(shared_cell(record, Uint32(12), Uint32)[0])
            valid = Uint32(shared_cell(record, Uint32(16), Uint32)[0])
            last = Uint32(shared_cell(record, Uint32(20), Uint32)[0])
            sync_math()
            if math_tid == 0:
                cute.arch.mbarrier_arrive(cute.recast_ptr(slab + self.task_empty_offset + slot * 8, dtype=Uint64))
            task_iter += Uint32(1)
            if phase == 0:
                running = False
            else:
                if phase == 1:
                    data_iter = self._select_persistent_tile(base, peers, slab, data_iter,
                                     expert, ntile, start, valid, sec1, sec2, clamp, True)
                else:
                    data_iter = self._select_persistent_tile(base, peers, slab, data_iter,
                                     expert, ntile, start, valid, sec2, sec2, clamp, False)
                fence_gpu()
                sync_math()
                if math_tid == 0 and phase == 1 and last != 0:
                    atom_release_gpu_add_u32(self._field(base, 'l1_done', Uint32) + expert, Uint32(1))

    @cute.jit
    def _select_persistent_tile(self, base: Uint64, peers: cute.Pointer, slab: cute.Pointer,
                                 data_iter: Uint32, expert: Uint32, ntile: Uint32,
                                 start: Uint32, valid: Uint32, secondary: cute.Pointer, output_secondary: cute.Pointer,
                                 clamp: Float32, l1: cutlass.Constexpr):
        result = Uint32(data_iter)
        if cutlass.const_expr(self.max_m == 8):
            result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, output_secondary, clamp, 8, l1)
        elif cutlass.const_expr(self.max_m == 16):
            if valid <= 8:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, output_secondary, clamp, 8, l1)
            else:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, output_secondary, clamp, 16, l1)
        elif cutlass.const_expr(self.max_m == 24):
            if valid <= 8:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, output_secondary, clamp, 8, l1)
            elif valid <= 16:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, output_secondary, clamp, 16, l1)
            else:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, output_secondary, clamp, 24, l1)
        elif cutlass.const_expr(self.max_m == 32):
            if valid <= 8:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, output_secondary, clamp, 8, l1)
            elif valid <= 16:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, output_secondary, clamp, 16, l1)
            else:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, output_secondary, clamp, 32, l1)
        else:
            if valid <= 8:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, output_secondary, clamp, 8, l1)
            elif valid <= 16:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, output_secondary, clamp, 16, l1)
            elif valid <= 32:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, output_secondary, clamp, 32, l1)
            elif valid <= 40:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, output_secondary, clamp, 40, l1)
            elif valid <= 48:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, output_secondary, clamp, 48, l1)
            else:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, output_secondary, clamp, 64, l1)
        return result

    @cute.jit
    def _compute_persistent(self, base: Uint64, peers: cute.Pointer, w1: cute.Pointer,
                             sec1: cute.Pointer, w2: cute.Pointer,
                             sec2: cute.Pointer, clamp: Float32):
        tid, _, _ = cute.arch.thread_idx()
        alloc = cutlass.utils.SmemAllocator()
        slab = cute.recast_ptr(alloc.allocate(self.smem, 1024), dtype=Uint8)
        if tid < self.stages:
            cute.arch.mbarrier_init(cute.recast_ptr(slab + self.full_offset + tid * 8, dtype=Uint64), 257 if self.max_m == 64 else 65)
            cute.arch.mbarrier_init(cute.recast_ptr(slab + self.empty_offset + tid * 8, dtype=Uint64), 8)
        if tid < 2:
            cute.arch.mbarrier_init(cute.recast_ptr(slab + self.task_full_offset + tid * 8, dtype=Uint64), 1)
            cute.arch.mbarrier_init(cute.recast_ptr(slab + self.task_empty_offset + tid * 8, dtype=Uint64), 1)
        cute.arch.sync_threads()
        cute.arch.mbarrier_init_fence()
        epoch = ld_acquire_gpu_u32(self._field(base, 'header', Uint32))
        if tid < 128:
            cute.arch.setmaxregister_decrease(32 if self.max_m <= 32 else 64)
            self._produce_tasks(base, slab, epoch, w1, w2)
            fence_sys()
            sync_tile()
        else:
            cute.arch.setmaxregister_increase(104 if self.max_m <= 32 else 216)
            self._consume_tasks(base, peers, slab, sec1, sec2, clamp)
            fence_sys()
            sync_tile()

    @cute.jit
    def _compute_sync(self, base: Uint64, peers: cute.Pointer, w1: cute.Pointer,
                  s1: cute.Pointer, sec1: cute.Pointer, w2: cute.Pointer,
                  s2: cute.Pointer, sec2: cute.Pointer, clamp: Float32):
        tid, _, _ = cute.arch.thread_idx()
        alloc = cutlass.utils.SmemAllocator()
        slab = cute.recast_ptr(alloc.allocate(self.smem, 1024), dtype=Uint8)
        command = cute.make_tensor(cute.recast_ptr(slab + self.smem - 96, dtype=Uint32), cute.make_layout((3,)))
        head = self._field(base, 'header', Uint32)
        epoch = ld_acquire_gpu_u32(head)
        # Dispatch publishes only nonempty experts. Reserved queue entries
        # have an epoch flag so a consumer never reads an unpublished expert.
        # Exhaust L1 claims before any L2 wait: every dependency is in flight.
        is_l1 = True
        running = True
        while running:
            self._claim_command(base, command, epoch, is_l1)
            cute.arch.sync_threads()
            if command[0] == 0:
                if is_l1:
                    is_l1 = False
                else:
                    running = False
            else:
                expert = Uint32(command[1])
                ntile = Uint32(command[2])
                count = cell(base, Uint64(self.off_expert_count) + Uint64(expert) * 4, Uint32)[0]
                for start in cutlass.range(0, count, self.max_m):
                    valid = cute.math.min(Uint32(self.max_m), count - Uint32(start))
                    if is_l1:
                        self._bucket_tile(base, peers, slab, expert, ntile, Uint32(start), valid,
                                          w1, s1, sec1, clamp, True)
                    else:
                        self._bucket_tile(base, peers, slab, expert, ntile, Uint32(start), valid,
                                          w2, s2, sec2, clamp, False)
                fence_gpu()
                cute.arch.sync_threads()
                if is_l1 and tid == 0:
                    atom_release_gpu_add_u32(self._field(base, 'l1_done', Uint32) + expert, Uint32(self.routed_n_factor))
                cute.arch.sync_threads()
        fence_sys()

    @cute.jit
    def _compute(self, base: Uint64, peers: cute.Pointer, w1: cute.Pointer,
                  s1: cute.Pointer, sec1: cute.Pointer, w2: cute.Pointer,
                  s2: cute.Pointer, sec2: cute.Pointer, clamp: Float32):
        if cutlass.const_expr(self.warp_specialized):
            self._compute_persistent(base, peers, w1, sec1, w2, sec2, clamp)
        else:
            self._compute_sync(base, peers, w1, s1, sec1, w2, s2, sec2, clamp)

    @cute.jit
    def _shared_stage(self, base: Uint64, slab: cute.Pointer, weights: cute.Pointer,
                      scales: cute.Pointer, ntile: Uint32, start: Uint32,
                      valid: Uint32, kb: Uint32, bar: cute.Pointer,
                      n: cutlass.Constexpr, l1: cutlass.Constexpr):
        """Issue exact W/SF TMA and X/SF cp.async copies, without waiting."""
        tid, _, _ = cute.arch.thread_idx()
        k_size = self.hidden if l1 else self.shared_i
        input_offset = self.off_shared_l1_acts if l1 else self.off_shared_l2_acts
        sf_offset = self.off_shared_l1_acts_sf if l1 else self.off_shared_l2_acts_sf
        activation_offset = 36864
        sf_stage = activation_offset + n * 128
        if tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(bar, 36864)
            tile = Uint64(ntile) * (k_size // 128) + kb
            bulk_g2s_1d(slab, weights + tile * 32768, Uint32(32768), bar, cache_hint=self.shared_cache_hint)
            bulk_g2s_1d(slab + 32768, cute.recast_ptr(scales, dtype=Uint8) + tile * 4096, Uint32(4096), bar, cache_hint=self.shared_cache_hint)
        for j in cutlass.range(tid, n * 8, self.threads):
            row, col = Uint32(j // 8), Uint32(j % 8) * 16
            size = Uint32(16) if row < valid else Uint32(0)
            address = base + Uint64(input_offset) + Uint64(start + row) * k_size + kb * 128 + col
            copy_async_16(slab + activation_offset + canonical_offset(row, col),
                          cute.make_ptr(Uint8, address, cute.AddressSpace.gmem, assumed_align=16), size)
        for j in cutlass.range(tid, n * 4, self.threads):
            row, kg = Uint32(j % n), Uint32(j // n)
            size = Uint32(4) if row < valid else Uint32(0)
            address = base + Uint64(sf_offset) + (Uint64(start + row) * (k_size // 32) + kb * 4 + kg) * 4
            copy_async_4(slab + sf_stage + Uint32(j) * 4,
                         cute.make_ptr(Uint8, address, cute.AddressSpace.gmem, assumed_align=4), size)
        copy_commit()

    @cute.jit
    def _shared_tile(self, base: Uint64, slab: cute.Pointer, weights: cute.Pointer,
                     scales: cute.Pointer, ntile: Uint32, start: Uint32,
                     valid: Uint32, clamp: Float32, n: cutlass.Constexpr,
                     l1: cutlass.Constexpr):
        """Native dense FP8 SE tile in the shared compute role, with separate row/K32 scales.

        A K128 stage has 32768B FP8 W, 4096B FP32 W SF, n*128B X,
        and n*16B X SF. Persistent CTAs prefetch the next of two stages
        during current-stage math in the independent shared role.
        The end-of-stage CTA barrier retires all readers before slot reuse;
        each slot has its own mbarrier, with parity advancing on every reuse.
        """
        tid, _, _ = cute.arch.thread_idx()
        math_tid = tid
        lane, warp = math_tid % 32, (math_tid // 32) % 4
        wg = math_tid // 128
        k_size = self.hidden if l1 else self.shared_i
        activation_offset = 36864
        sf_stage = activation_offset + n * 128
        bar = cute.recast_ptr(slab + self.smem - 128, dtype=Uint64)
        # Each shared-role launch reserves both dense stages independently
        # of the routed geometry and its shared-memory budget.
        stage_stride = (36864 + n * 144 + 1023) // 1024 * 1024
        if tid < 2:
            cute.arch.mbarrier_init(bar + tid, 1)
        cute.arch.sync_threads()
        cute.arch.mbarrier_init_fence()
        final = cute.make_rmem_tensor((2, n // 2), Float32)
        final.fill(Float32(0))
        self._shared_stage(base, slab, weights, scales, ntile, start, valid,
                           Uint32(0), bar, n, l1)
        for kb in cutlass.range(k_size // 128):
            tile_slab = slab + Uint32(kb % 2) * stage_stride
            copy_wait()
            if tid == 0:
                cute.arch.mbarrier_wait(bar + Uint32(kb % 2), Uint32((kb // 2) % 2))
            cute.arch.sync_threads()
            tma_store_fence()
            if kb + 1 < k_size // 128:
                next_slot = Uint32((kb + 1) % 2)
                self._shared_stage(base, slab + next_slot * stage_stride,
                                   weights, scales, ntile, start, valid,
                                   Uint32(kb + 1), bar + next_slot, n, l1)
            for h in cutlass.range_constexpr(2):
                half = Uint32(wg * 2 + h)
                row = Uint32(warp * 16 + lane // 4)
                for pair in cutlass.range_constexpr(2):
                    fragment0 = load_shared_u32x4(tile_slab + ((half * 4 + pair * 2) * 128 + Uint32(math_tid % 128)) * 16)
                    fragment1 = load_shared_u32x4(tile_slab + ((half * 4 + pair * 2 + 1) * 128 + Uint32(math_tid % 128)) * 16)
                    descriptor = smem_descriptor(tile_slab + activation_offset) + Uint64(pair * 4)
                    if cutlass.const_expr(self.shared_scale_prefetch):
                        raw_parts = wgmma_rs_k32_pair_issue(
                            fragment0, fragment1, descriptor, n)
                        weight_sf = cute.make_rmem_tensor((2, 2), Float32)
                        input_sf = cute.make_rmem_tensor((2, n // 4), Float32)
                        for sf_bank in cutlass.range_constexpr(2):
                            sf_kg = pair * 2 + sf_bank
                            weight_sf[sf_bank, 0] = shared_cell(tile_slab, Uint32(32768) + (half * 256 + row * 4 + sf_kg) * 4, Float32)[0]
                            weight_sf[sf_bank, 1] = shared_cell(tile_slab, Uint32(32768) + (half * 256 + (row + 8) * 4 + sf_kg) * 4, Float32)[0]
                            for sf_group in cutlass.range_constexpr(n // 8):
                                sf_token = Uint32(sf_group * 8 + 2 * (lane % 4))
                                sf_pair = load_scale_pair(tile_slab + sf_stage + Uint32(sf_kg * n) * 4 + sf_token * 4)
                                input_sf[sf_bank, sf_group * 2] = sf_pair[0]
                                input_sf[sf_bank, sf_group * 2 + 1] = sf_pair[1]
                        parts = wgmma_rs_wait(raw_parts)
                    else:
                        parts = wgmma_rs_k32_pair(
                            fragment0, fragment1, descriptor, n)
                    for bank in cutlass.range_constexpr(2):
                        kg = pair * 2 + bank
                        part = tuple(parts[bank * (n // 2) + i] for i in range(n // 2))
                        if cutlass.const_expr(self.shared_scale_prefetch):
                            sw0, sw1 = weight_sf[bank, 0], weight_sf[bank, 1]
                        else:
                            sw0 = shared_cell(tile_slab, Uint32(32768) + (half * 256 + row * 4 + kg) * 4, Float32)[0]
                            sw1 = shared_cell(tile_slab, Uint32(32768) + (half * 256 + (row + 8) * 4 + kg) * 4, Float32)[0]
                        for group in cutlass.range_constexpr(n // 8):
                            token = Uint32(group * 8 + 2 * (lane % 4))
                            if cutlass.const_expr(self.shared_scale_prefetch):
                                sa = (input_sf[bank, group * 2], input_sf[bank, group * 2 + 1])
                            else:
                                sa = load_scale_pair(tile_slab + sf_stage + Uint32(kg * n) * 4 + token * 4)
                            final[h, group * 4] = cute.math.fma(part[group * 4], sw0 * sa[0], final[h, group * 4])
                            final[h, group * 4 + 1] = cute.math.fma(part[group * 4 + 1], sw0 * sa[1], final[h, group * 4 + 1])
                            final[h, group * 4 + 2] = cute.math.fma(part[group * 4 + 2], sw1 * sa[0], final[h, group * 4 + 2])
                            final[h, group * 4 + 3] = cute.math.fma(part[group * 4 + 3], sw1 * sa[1], final[h, group * 4 + 3])
            cute.arch.sync_threads()
        for h in cutlass.range_constexpr(2):
            half = Uint32(wg * 2 + h)
            row = Uint32(warp * 16 + lane // 4)
            for group in cutlass.range_constexpr(n // 8):
                token = Uint32(group * 8 + 2 * (lane % 4))
                skew = Uint32((token % 8) // 2 * 8)
                for r in cutlass.range_constexpr(2):
                    for t in cutlass.range_constexpr(2):
                        col = half * 64 + row + r * 8
                        if cutlass.const_expr(l1):
                            ep_smem_offset = (token + t) * 1024 + (col ^ skew) * 4
                        else:
                            ep_smem_offset = (token + t) * 1024 + (col ^ skew ^ ((col & 32) >> 3)) * 4
                        shared_cell(slab, ep_smem_offset, Float32)[0] = final[h, group * 4 + r * 2 + t]
        cute.arch.sync_threads()
        if cutlass.const_expr(l1):
            for ep_token in cutlass.range(math_tid // 32, valid, 8):
                skew = Uint32((ep_token % 8) // 2 * 8)
                raw = load_shared_8(slab + ep_token * 1024 + ((lane * 4) ^ skew) * 4, True)
                values = cute.make_rmem_tensor((4,), Float32)
                amax = Float32(0)
                for j in cutlass.range_constexpr(4):
                    # Match standalone DeepGEMM BF16 GEMM1 output rounding.
                    gate = cute.math.min(round_bf16_f32(raw[j]), clamp)
                    up = cute.math.min(cute.math.max(round_bf16_f32(raw[j + 4]), -clamp), clamp)
                    values[j] = gate * cute.math.rcp(Float32(1) + cute.math.exp(-gate, fastmath=self.fast_math)) * up
                    amax = cute.math.max(amax, abs_f32(values[j]))
                for shift in cutlass.range_constexpr(3):
                    amax = cute.math.max(amax, cute.arch.shuffle_sync_bfly(amax, 4 >> shift))
                exponent = ue8m0_fp8_scale_exp(amax)
                inverse = ue8m0_fp8_inv(exponent)
                if lane % 8 == 0:
                    sf = cell(base, Uint64(self.off_shared_l2_acts_sf) + (Uint64(start + ep_token) * (self.shared_i // 32) + ntile * 4 + lane // 8) * 4, Float32)
                    sf[0] = bits_float(exponent << 23)
                lo = f32x2_to_e4m3x2_bits(values[0] * inverse, values[1] * inverse)
                hi = f32x2_to_e4m3x2_bits(values[2] * inverse, values[3] * inverse)
                out = cell(base, Uint64(self.off_shared_l2_acts) + Uint64(start + ep_token) * self.shared_i + ntile * 128 + lane * 4, Uint32)
                out[0] = Uint32(lo) | (Uint32(hi) << 16)
        else:
            for j in cutlass.range(math_tid, valid * 32, 256):
                token, col = Uint32(j // 32), Uint32(j % 32) * 8
                skew = Uint32((token % 8) // 2 * 8)
                lo = load_shared_4(slab + token * 1024 + (col ^ skew ^ ((col & 32) >> 3)) * 4)
                hi = load_shared_4(slab + token * 1024 + ((col + 4) ^ skew ^ (((col + 4) & 32) >> 3)) * 4)
                ep_global_address = base + Uint64(self.off_shared_y) + (Uint64(start + token) * self.hidden + ntile * 256 + col) * 2
                store_bf16_8(ep_global_address, (*lo, *hi))
        cute.arch.sync_threads()

    @cute.jit
    def _shared_select(self, base: Uint64, slab: cute.Pointer, weights: cute.Pointer,
                       scales: cute.Pointer, ntile: Uint32, start: Uint32,
                       valid: Uint32, clamp: Float32, l1: cutlass.Constexpr):
        if cutlass.const_expr(self.max_m == 8):
            self._shared_tile(base, slab, weights, scales, ntile, start, valid, clamp, 8, l1)
        elif cutlass.const_expr(self.max_m == 16):
            self._shared_tile(base, slab, weights, scales, ntile, start, valid, clamp, 16, l1)
        elif cutlass.const_expr(self.max_m == 32):
            self._shared_tile(base, slab, weights, scales, ntile, start, valid, clamp, 32, l1)
        else:
            self._shared_tile(base, slab, weights, scales, ntile, start, valid, clamp, 64, l1)

    @cute.jit
    def _compute_shared(self, base: Uint64, slab: cute.Pointer, tokens: Uint32, w1: cute.Pointer,
                        s1: cute.Pointer, w2: cute.Pointer, s2: cute.Pointer,
                        clamp: Float32):
        # Combined CTAs exhaust L1 claims before L2; split L2 starts after all
        # L1 CTAs have begun. L1 never waits for L2, so producers can advance.
        # Shared phases use local-token queues; T0 claims no shared tasks.
        tid, _, _ = cute.arch.thread_idx()
        command = cute.make_tensor(cute.recast_ptr(slab + self.smem - 96, dtype=Uint32), cute.make_layout((1,)))
        head = self._field(base, 'header', Uint32)
        blocks = (tokens + self.max_m - 1) // self.max_m
        for phase in cutlass.range_constexpr(self.shared_phase_start, self.shared_phase_end):
            tiles = self.shared_i // 128 if phase == 0 else self.hidden // 256
            running = True
            while running:
                if tid == 0:
                    command[0] = claim_sm90_task(head + 20 + phase)
                cute.arch.sync_threads()
                task = Uint32(command[0])
                if task >= blocks * tiles:
                    running = False
                else:
                    block, ntile = task // tiles, task % tiles
                    start = block * self.max_m
                    valid = cute.math.min(Uint32(self.max_m), tokens - start)
                    if cutlass.const_expr(phase == 0):
                        self._shared_select(base, slab, w1, s1, ntile, start, valid, clamp, True)
                        fence_gpu()
                        cute.arch.sync_threads()
                        if tid == 0:
                            atom_release_gpu_add_u32(self._field(base, 'shared_l1_done', Uint32) + block, Uint32(1))
                    else:
                        if tid == 0:
                            wait_count(self._field(base, 'shared_l1_done', Uint32) + block, Uint32(self.shared_i // 128))
                        cute.arch.sync_threads()
                        self._shared_select(base, slab, w2, s2, ntile, start, valid, clamp, False)
                cute.arch.sync_threads()
        fence_sys()

    @cute.jit
    def _combine(self, base: Uint64, peers: cute.Pointer, y: cute.Pointer, rank: Uint32, tokens: Uint32):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        head = self._field(base, 'header', Uint32)
        epoch = ld_acquire_gpu_u32(head)
        table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        cute.arch.sync_threads()
        if tid < 128:
            # All local dispatch/routed/shared predecessors have retired. These fields
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
                for c in cutlass.range_constexpr(8 if self.shared_experts > 0 else 6):
                    counter_out = cell(base, Uint64((3, 7, 8, 10, 11, 13, 20, 21)[c] * 4), Uint32)
                    counter_out[0] = Uint32(0)
            if cutlass.const_expr(self.shared_experts > 0):
                for block_m in cutlass.range(block * 128 + tid, (self.capacity + 7) // 8, self.grid * 128):
                    out = cell(base, Uint64(self.off_shared_l1_done) + Uint64(block_m) * 4, Uint32)
                    out[0] = Uint32(0)
            # Retain peer-readable send_count/source_index through this call.
            # The all-peer compute_ready rendezvous below retires their old readers
            # before any rank may begin its next dispatch initialization.
            fence_gpu()
            sync_loader()
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
            sync_loader()
        else:
            math_tid = tid - 128
            if cutlass.const_expr(not self.selected_combine):
                if math_tid < self.ranks:
                    wait_epoch(self._field(base, 'compute_ready', Uint32) + math_tid, epoch, True)
                sync_math()
            for index in cutlass.range(block * 256 + math_tid, tokens * (self.hidden // self.combine_vector), self.grid * 256):
                token = index // (self.hidden // self.combine_vector)
                col = (index % (self.hidden // self.combine_vector)) * self.combine_vector
                if cutlass.const_expr(self.selected_combine):
                    # The row-vector count and both strides are multiples of 32.
                    # Every participating warp is complete and has one token.
                    lane = math_tid % 32
                    valid_owner = Boolean(False)
                    owner = Uint32(0)
                    if lane < self.topk:
                        routed_expert = cell(base, Uint64(self.off_topk_idx) + Uint64(token * self.topk + lane) * 8, Int64)[0]
                        valid_owner = routed_expert >= 0 and routed_expert < Int64(self.ranks * self.local)
                        if valid_owner:
                            owner = Uint32(routed_expert) // Uint32(self.local)
                    peer_ready = not valid_owner
                    while cute.arch.vote_ballot_sync(peer_ready) != Uint32(0xffffffff):
                        if valid_owner and not peer_ready:
                            peer_ready = ld_acquire_sys_u32(self._field(base, 'compute_ready', Uint32) + owner) == epoch
                    cute.arch.sync_warp()
                total = cute.make_rmem_tensor((self.combine_vector,), Float32)
                total.fill(Float32(0))
                for slot in cutlass.range_constexpr(self.topk):
                    expert = cell(base, Uint64(self.off_topk_idx) + Uint64(token * self.topk + slot) * 8, Int64)[0]
                    if expert >= 0:
                        address = base + Uint64(self.off_combine) + (Uint64(token * self.topk + slot) * self.hidden + col) * 2
                        for part in cutlass.range_constexpr(self.combine_vector // 8):
                            values8 = load_bf16_8(address + Uint64(part * 16))
                            for i in cutlass.range_constexpr(8):
                                total[part * 8 + i] += values8[i]
                if cutlass.const_expr(self.shared_experts > 0):
                    for part in cutlass.range_constexpr(self.combine_vector // 8):
                        shared = load_bf16_8(base + Uint64(self.off_shared_y) + Uint64(index) * (self.combine_vector * 2) + Uint64(part * 16))
                        for i in cutlass.range_constexpr(8):
                            total[part * 8 + i] = round_bf16_f32(total[part * 8 + i]) + shared[i]
                for part in cutlass.range_constexpr(self.combine_vector // 8):
                    store_bf16_8(Uint64(y.toint()) + Uint64(index) * (self.combine_vector * 2) + Uint64(part * 16),
                                 tuple(total[part * 8 + i] for i in range(8)))
            fence_sys()
            sync_math()
            if math_tid == 0:
                st_release_u32(self._field(base, 'combine_block_done', Uint32) + block, epoch, False)
            if block == 0:
                for other_block in cutlass.range(math_tid, self.grid, 256):
                    wait_epoch(self._field(base, 'combine_block_done', Uint32) + other_block, epoch, False)
                sync_math()
                if cutlass.const_expr(self.selected_combine):
                    # The selected waits protect output loads. This full join
                    # also retires peer readers of source_index/count/x,
                    # including T0 and all-invalid routes, before epoch advance.
                    if math_tid < self.ranks:
                        wait_epoch(self._field(base, 'compute_ready', Uint32) + math_tid, epoch, True)
                    sync_math()
                # Refresh EVERY unused flag after all old readers retire.
                # This prevents stale zero from aliasing epoch FFFFFFFF->0,
                # including when successive calls use different grid sizes.
                for flag in cutlass.range(math_tid, self.finalize_flags, 256):
                    flag_out = cell(base, Uint64(self.off_combine_block_done) + Uint64(flag) * 4, Uint32)
                    flag_out[0] = epoch
                fence_gpu()
                sync_math()
        # Retire every LOCAL combine/cleanup group before the successor.
        # A next-call scatter cannot overwrite a peer's old combine slots:
        # both dispatch paths await every peer's next input_ready before
        # publishing any expert_ready, so that peer has finished this call.
        # The all-peer compute_ready join (entry or tail) retires old source readers.
        # This removes a redundant all-rank exit wait, including for T0.
        sync_tile()
        if block == 0 and tid == 0:
            final_out = cell(base, Uint64(16), Uint32)
            final_out[0] = Uint32(0)
            cleanup_out = cell(base, Uint64(18 * 4), Uint32)
            cleanup_out[0] = Uint32(0)
            st_release_u32(head, epoch + Uint32(1), False)

    @cute.kernel
    def kernel(self, workspace: cute.Pointer, peers: cute.Pointer, y: cute.Pointer,
                w1: cute.Pointer, s1: cute.Pointer, sec1: cute.Pointer,
                w2: cute.Pointer, s2: cute.Pointer, sec2: cute.Pointer,
                stats: cute.Pointer, sw1: cute.Pointer, ss1: cute.Pointer,
                sw2: cute.Pointer, ss2: cute.Pointer, rank: Uint32, tokens: Uint32, clamp: Float32):
        base = Uint64(workspace.toint())
        if cutlass.const_expr(self.role == 1):
            if cutlass.const_expr(self.direct_route_scan):
                self._dispatch_direct(base, peers, rank, tokens, stats)
            else:
                self._dispatch(base, peers, rank, tokens, stats, sec1)
        elif cutlass.const_expr(self.role == 2):
            self._compute(base, peers, w1, s1, sec1, w2, s2, sec2, clamp)
        elif cutlass.const_expr(self.role in (4, 5)):
            alloc = cutlass.utils.SmemAllocator()
            slab = cute.recast_ptr(alloc.allocate(self.smem, 1024), dtype=Uint8)
            self._compute_shared(base, slab, tokens, sw1, ss1, sw2, ss2, clamp)
        else:
            self._combine(base, peers, y, rank, tokens)

    @cute.jit
    def __call__(self, workspace: cute.Pointer, peers: cute.Pointer, y: cute.Pointer,
                  w1: cute.Pointer, s1: cute.Pointer, sec1: cute.Pointer,
                  w2: cute.Pointer, s2: cute.Pointer, sec2: cute.Pointer,
                  stats: cute.Pointer, sw1: cute.Pointer, ss1: cute.Pointer,
                  sw2: cute.Pointer, ss2: cute.Pointer, rank: Uint32, tokens: Uint32, clamp: Float32):
        self.kernel(workspace, peers, y, w1, s1, sec1, w2, s2, sec2,
                    stats, sw1, ss1, sw2, ss2, rank, tokens, clamp).launch(
            grid=(self.grid, 1, 1), block=(self.threads, 1, 1),
            max_number_threads=(self.threads, 1, 1),
            min_blocks_per_mp=(4 if self.role == 3 else ((8 if self.threads == 128 else 4) if self.role == 1 else ((3 if self.max_m == 8 else 2) if self.role in (2, 4, 5) and not self.warp_specialized else (2 if self.role in (2, 4, 5) and self.max_m <= 32 else 1)))))
