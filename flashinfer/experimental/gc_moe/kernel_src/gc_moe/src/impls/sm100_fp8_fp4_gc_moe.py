"""CuTe DSL source for three persistent GC MoE roles and four worker phases."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import math
from cutlass import Float32, Int32, Uint8, Uint16, Uint32, Uint64

from ..comm.gc_barrier import (
    DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS,
    grid_sync,
    wait_sys_u64_eq,
)
from ..common.math import (
    bf16x2_high_f32_bits,
    bf16x2_max,
    bf16x2_min,
    bits_float,
    f32_to_bf16_bits,
    f32x2_to_bf16x2_bits,
    fast_exp_neg_f32,
    fast_rcp,
    abs_f32,
    f32x2_to_e4m3x2_bits,
    fmax_f32,
    pack_bf16_pair,
    ue8m0_fp8_inv,
    ue8m0_fp8_scale_exp,
    warp_group4_max,
)
from ..common.tma_copy import load_mixed_k128_tma, use_small_tile_path
from ..layout.gc_moe import (
    ACCUM_BARRIER_STAGES,
    COMBINE_WARPS,
    FINALIZE_THREADS,
    GCBufferLayout,
    MAX_RANKS,
    SIGNAL_OFFSETS,
)
from ..layout.sym_buffer import mapped_peer_base
from ..mma.sm100 import (
    MixedMmaTile,
    blockscaled_instr_desc_hi,
    issue_mixed_k128,
    kmajor_smem_desc,
    kmajor_smem_desc_from_start,
    select_weight_format,
)
from ..ptx.ld_st import (
    EVICT_FIRST,
    EVICT_NORMAL,
    atom_add_u64,
    atom_shared_add_u32,
    atom_sys_add_u64,
    bulk_zero_shared,
    ld_acquire_gpu_u32,
    ld_acquire_gpu_u64,
    ld_acquire_sys_u64,
    ld_global_u32x3,
    ld_global_u32x4,
    ld_global_u64,
    ld_shared_u32x4,
    ld_volatile_u32,
    ld_volatile_u64,
    red_gpu_add_s32,
    red_gpu_add_u32,
    red_release_gpu_add_u32,
    red_release_gpu_xor_u64,
    st_global_u32x4,
    st_global_u64,
    st_release_sys_u64,
    st_shared_u32x4,
    stmatrix_bf16x8_trans,
    stmatrix_fp8x4_trans_late,
)
from ..ptx.tcgen05 import (
    tcgen05_after_thread_sync,
    tcgen05_before_thread_sync,
    tmem_free_2cta,
    tmem_load_16dp256b4,
)
from ..ptx.tma import (
    bulk_g2s_1d,
    bulk_s2g_1d,
    generic_tensormap_ptr,
    prefetch_tensormap,
    tile_s2g_2d,
    tma_store_arrive,
    tma_store_fence,
    tma_store_wait_0,
    tma_store_wait_1,
)
from ..ptx.utils import (
    accumulate_bf16_into_f32,
    assume_u32_lt,
    device_trap,
    elect_one_sync,
    fence_acq_rel_sys,
    read_clock64,
    sync_aligned,
    sync_unaligned,
    undefined_u32,
    warp_nth_set_bit,
    warpgroup_reg_alloc,
    warpgroup_reg_dealloc,
)
from ..scheduler.gc_moe import (
    GCBlockPhase,
    TASK_EMPTY_CONSUMERS,
    advance_expert,
    release_task_info,
    schedule_tasks,
)


# A fixed small-decode floor, independent of the local expert count: 64 CTAs
# provide up to 512 combine warps before clamping to the compute SM budget.
SMALL_DECODE_FINALIZE_SMS = 64


@cute.jit
def _swiglu_pair(gate0_accum: Float32, gate1_accum: Float32,
                up0_accum: Float32, up1_accum: Float32,
                weight0: Float32, weight1: Float32,
                clamp_value: Float32, alpha: Float32,
                clamp_enabled: cutlass.Constexpr,
                oai_enabled: cutlass.Constexpr,
                fast_math: cutlass.Constexpr):
    """Reference W1 float2 SwiGLU, preserving the BF16x2 and FP32 order."""
    gate_bits = f32x2_to_bf16x2_bits(gate0_accum, gate1_accum)
    up_bits = f32x2_to_bf16x2_bits(up0_accum, up1_accum)
    if cutlass.const_expr(clamp_enabled):
        bound = Uint32(f32_to_bf16_bits(clamp_value))
        bound_pair = bound | (bound << Uint32(16))
        gate_bits = bf16x2_min(gate_bits, bound_pair)
        up_bits = bf16x2_max(up_bits, bound_pair ^ Uint32(0x80008000))
        up_bits = bf16x2_min(up_bits, bound_pair)
    gate0 = bits_float((gate_bits & Uint32(0xFFFF)) << Uint32(16))
    gate1 = bits_float(bf16x2_high_f32_bits(gate_bits))
    sigmoid0, sigmoid1 = gate0, gate1
    if cutlass.const_expr(oai_enabled):
        sigmoid0, sigmoid1 = cute.arch.mul_packed_f32x2(
            (gate0, gate1), (alpha, alpha), rnd='rn', ftz=False)
    if cutlass.const_expr(fast_math):
        exp0 = fast_exp_neg_f32(sigmoid0)
        exp1 = fast_exp_neg_f32(sigmoid1)
    else:
        exp0 = cute.math.exp(-sigmoid0, fastmath=False)
        exp1 = cute.math.exp(-sigmoid1, fastmath=False)
    denom0, denom1 = cute.arch.add_packed_f32x2(
        (Float32(1), Float32(1)), (exp0, exp1), rnd='rn', ftz=False)
    if cutlass.const_expr(fast_math):
        gate0, gate1 = cute.arch.mul_packed_f32x2(
            (gate0, gate1), (fast_rcp(denom0), fast_rcp(denom1)), rnd='rn', ftz=False)
    else:
        gate0 = cute.math.div(
            gate0, denom0, fastmath=False, ftz=False,
            rounding=cute.math.RoundingMode.NEAREST_EVEN)
        gate1 = cute.math.div(
            gate1, denom1, fastmath=False, ftz=False,
            rounding=cute.math.RoundingMode.NEAREST_EVEN)
    up0 = bits_float((up_bits & Uint32(0xFFFF)) << Uint32(16))
    up1 = bits_float(bf16x2_high_f32_bits(up_bits))
    if cutlass.const_expr(oai_enabled):
        up0, up1 = cute.arch.add_packed_f32x2(
            (up0, up1), (Float32(1), Float32(1)), rnd='rn', ftz=False)
    activation0, activation1 = cute.arch.mul_packed_f32x2(
        (gate0, gate1), (up0, up1), rnd='rn', ftz=False)
    return cute.arch.mul_packed_f32x2(
        (activation0, activation1), (weight0, weight1), rnd='rn', ftz=False)


@cute.jit
def sync_epilogue_warpgroup(warpgroup: Uint32) -> None:
    """Reference WG-local barrier ID 3 + warpgroup, with 128 callers."""
    sync_aligned(128, Uint32(3) + warpgroup)


@cute.jit
def _load_w2_atom(raw_values: cute.Tensor, buffer: cutlass.Constexpr,
                  atom: cutlass.Constexpr, address: Uint32) -> None:
    """Keep both 16dp halves of one 8-row W2 atom in registers."""
    first = tmem_load_16dp256b4(address)
    second = tmem_load_16dp256b4(address | Uint32(0x00100000))
    for register in cutlass.range_constexpr(4):
        raw_values[buffer, atom, register] = first[register]
        raw_values[buffer, atom, register + 4] = second[register]


@cute.jit
def _release_w2_tmem(accum_empty: cute.Pointer, stage: Uint32) -> None:
    tcgen05_before_thread_sync()
    cute.arch.mbarrier_arrive(accum_empty + stage, peer_cta_rank_in_cluster=0)


@cute.jit
def _load_w1_atom(raw_values: cute.Tensor, buffer: cutlass.Constexpr,
                  atom: cutlass.Constexpr, address: Uint32) -> None:
    """Source-contract 16dp W1 atom: gate/up for two post channels."""
    first = tmem_load_16dp256b4(address)
    second = tmem_load_16dp256b4(address | Uint32(0x00100000))
    for register in cutlass.range_constexpr(4):
        raw_values[buffer, atom, register] = first[register]
        raw_values[buffer, atom, register + 4] = second[register]


@cute.jit
def _release_w1_tmem(accum_empty: cute.Pointer, stage: Uint32) -> None:
    tcgen05_before_thread_sync()
    cute.arch.mbarrier_arrive(accum_empty + stage, peer_cta_rank_in_cluster=0)


@cute.jit
def _w1_pack_fp8x4(a: Float32, b: Float32,
                    c: Float32, d: Float32) -> Uint32:
    """Pack reference channel0/token0, channel0/token1, channel1/token0/1."""
    ab = f32x2_to_e4m3x2_bits(a, b)
    cd = f32x2_to_e4m3x2_bits(c, d)
    return Uint32(ab) | (Uint32(cd) << Uint32(16))


@cute.jit
def _load_w2_source_metadata(base: cute.Pointer, row_in_block: Uint32,
                              valid_m: Uint32, pool_token: Uint32,
                              rank_idx: Uint32, is_shared,
                              topk: cutlass.Constexpr,
                              has_shared: cutlass.Constexpr):
    """Read valid rows with fresh undefined values for inactive rows."""
    src_rank = undefined_u32()
    src_token = undefined_u32()
    topk_slot = undefined_u32()
    if row_in_block < valid_m:
        if cutlass.const_expr(has_shared):
            src_rank = rank_idx
            src_token = pool_token
            topk_slot = Uint32(topk)
            if not is_shared:
                src_rank, src_token, topk_slot = ld_global_u32x3(
                    base + pool_token * Uint32(3))
        else:
            src_rank, src_token, topk_slot = ld_global_u32x3(
                base + pool_token * Uint32(3))
    return src_rank, src_token, topk_slot


@cute.jit
def _initialize_private_combine_bar(bar: cute.Pointer) -> None:
    """Publish one warp-owned barrier before that warp's first use."""
    if elect_one_sync():
        cute.arch.mbarrier_init(bar, 1)
        cute.arch.mbarrier_init_fence()
    cute.arch.sync_warp()


class GCMoERoleKernel:
    """Keep the role on the Python object so CuTe erases other role paths."""

    def __init__(self, role: int, dispatch_sms: int = 1, compute_sms: int = 2,
                 block_m: int = 16, hidden: int = 256,
                 intermediate: int = 128, num_tokens: int = 16,
                 num_ranks: int = 1, local_experts: int = 1,
                 shared_experts: int = 0, capacity: int = 1920,
                 shared_weight_gran_k: int = 32,
                 finalize_output: bool = False,
                 ring_tokens: int = 1920,
                 sf_ring_tokens: int = 30720,
                 topk: int = 1,
                 activation_clamp: float = math.inf,
                 fast_math: bool = True,
                 has_stats: bool = False,
                 nvlink_timeout_secs: int = DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS,
                 *, role_config, combine_geometry: tuple[int, int, int] | None = None,
                 routed_weight_linger: bool = False):
        if role_config.role != role or role_config.block_m != block_m:
            raise ValueError('kernel role/block M differs from its host configuration')
        math_stages = role_config.num_stages
        if role not in (1, 2, 3):
            raise ValueError('role must be dispatch, compute or finalize')
        if local_experts < 1 or local_experts > 512 or \
                num_ranks * local_experts > 2048 or topk < 1 or \
                topk + bool(shared_experts) > 32:
            raise NotImplementedError('device scheduler requires 1..512 local experts, <=2048 global experts, topk>=1 and topk+shared slot<=32')
        if num_ranks < 1 or num_ranks > MAX_RANKS or not 0 <= num_tokens <= capacity:
            raise NotImplementedError('dispatch routing requires 1..72 ranks within capacity')
        if math_stages < 2 or math_stages > 32:
            raise ValueError('GC math pipeline needs 2..32 TMA stages')
        if not 0 < nvlink_timeout_secs <= 2_147_483_647:
            raise ValueError('NVLink barrier timeout must fit positive INT_MAX seconds')
        self.role = role
        self.dispatch_sms = dispatch_sms
        self.compute_sms = compute_sms
        self.block_m = block_m
        self.num_tokens = num_tokens
        self.hidden = hidden
        self.intermediate = intermediate
        # Base grid; only role3 with finalize_output overrides it below.
        self.finalize_sms = (min(compute_sms, SMALL_DECODE_FINALIZE_SMS)
                             if num_tokens <= 128 else compute_sms)
        self.num_ranks = num_ranks
        self.local_experts = local_experts
        self.topk = topk
        self.has_activation_clamp = not math.isinf(activation_clamp)
        self.activation_clamp = activation_clamp if self.has_activation_clamp else 0.0
        self.fast_math = fast_math
        self.has_stats = has_stats
        self.shared_experts = shared_experts
        self.use_small_tile_path = use_small_tile_path(shared_experts, block_m)
        self.capacity = capacity
        self.finalize_output = finalize_output
        self.shared_weight_gran_k = shared_weight_gran_k
        # Use FIRST for shared weights/SFs with larger decode tiles and
        # at most one routed block per expert on average. Keep BM16 NORMAL.
        self.shared_weight_cache_hint = (
            EVICT_FIRST if block_m >= 32 and
            num_tokens * topk <= block_m * local_experts
            else EVICT_NORMAL)
        self.math_stages = math_stages
        self.nvlink_timeout_secs = nvlink_timeout_secs
        self.routed_weight_linger = routed_weight_linger
        # Block configuration and the shared C/D storage union.
        self.store_m_l1 = role_config.store_m_l1
        self.store_m_l2 = role_config.store_m_l2
        self.cd_l1_bytes = 2 * 2 * self.store_m_l1 * 64
        self.cd_l2_bytes = 2 * self.store_m_l2 * 128 * 2
        self.cd_bytes = max(self.cd_l1_bytes, self.cd_l2_bytes)
        # float2[kNumEpilogueWarps][STORE_BLOCK_M_L1 / 2].
        self.amax_float_count = 8 * self.store_m_l1
        self.non_epi_registers = 40 if local_experts <= 64 else 72
        self.epi_registers = 208 if local_experts <= 64 else 168
        tile = MixedMmaTile(block_m, hidden, intermediate)
        self.l1_clusters = tile.l1_clusters
        self.l2_clusters = tile.l2_clusters
        self.l1_full_mask = (1 << tile.l1_n_blocks) - 1
        self.tmem_cols = tile.tmem_cols
        self.accum_stages = tile.epilogue_stages
        if self.accum_stages > ACCUM_BARRIER_STAGES:
            raise ValueError('epilogue stages exceed the reserved accumulator barriers')
        self.sf_block_m = role_config.sf_block_m
        self.sfa_tmem_cols = tile.sfa_tmem_cols
        self.dispatch_warps = role_config.routing_warps
        dispatch_registers = 48 if local_experts <= 64 else 96
        role_registers = (
            self.dispatch_warps * 32 * dispatch_registers if role == 1 else
            128 * self.non_epi_registers + 256 * self.epi_registers if role == 2 else
            128 * dispatch_registers + (COMBINE_WARPS * 32) * self.epi_registers)
        if role_registers > 64512:
            raise ValueError('role register allocation exceeds the SM budget')
        self.pull_bytes = role_config.pull_bytes
        self.pull_chunks = hidden // self.pull_bytes
        if role == 3 and finalize_output:
            if combine_geometry is None:
                raise ValueError('finalize requires host combine geometry')
            (self.combine_chunks, self.combine_chunk_bytes,
             self.combine_uint4_per_lane) = combine_geometry
            # Size fused output work independently of the expert-cleanup grid.
            self.finalize_sms = min(
                compute_sms,
                max(SMALL_DECODE_FINALIZE_SMS if num_tokens <= 128 else
                    min(compute_sms, local_experts + 1),
                    (num_tokens * self.combine_chunks + (COMBINE_WARPS - 1)) //
                    COMBINE_WARPS))
        self.shared_layout = role_config.shared_layout
        layout = GCBufferLayout(num_ranks, num_ranks * local_experts, capacity, topk,
                                hidden, intermediate, ring_tokens, sf_ring_tokens,
                                shared_experts)
        self.topk_offset_words = layout.views['topk_idx'].offset // 4
        self.input_x_words = layout.views['x'].offset // 4
        self.input_sf_words = layout.views['x_sf'].offset // 4
        self.input_weight_words = layout.views['topk_weights'].offset // 4
        self.l1_x_words = layout.views['l1_acts'].offset // 4
        self.l1_sf_words = layout.views['l1_acts_sf'].offset // 4
        self.l1_weight_words = layout.views['l1_topk_weights'].offset // 4
        self.l2_x_words = layout.views['l2_acts'].offset // 4
        self.l2_sf_words = layout.views['l2_acts_sf'].offset // 4
        self.shared_l2_x_words = layout.views['shared_l2_acts'].offset // 4
        self.shared_l2_sf_words = layout.views['shared_l2_acts_sf'].offset // 4
        self.shared_sf_tokens = ((capacity + 7) // 8) * 128
        self.combine_words = layout.views['combine_tokens'].offset // 4
        self.metadata_words = layout.workspace_offsets['token_src_metadata'] // 4
        self.ring_tokens = layout.ring_tokens
        self.sf_ring_tokens = layout.sf_ring_tokens
        self.ring_blocks = layout.ring_tokens // block_m
        sf_block_m = self.sf_block_m
        sf_row_max = max((t & ~127) + (t & 31) * 4 + ((t >> 5) & 3)
                         for t in range(block_m))
        sf_even_row_max = max((t & ~127) + (t & 31) * 4 + ((t >> 5) & 3)
                              for t in range(0, block_m, 2))
        last_ring_sf_row = (self.ring_blocks - 1) * sf_block_m
        l1_sf_local_max = ((hidden // 128 - 1) * self.sf_ring_tokens +
                           last_ring_sf_row + sf_row_max)
        routed_w1_sf_local_max = 4 * (
            (intermediate // 128 - 1) * self.sf_ring_tokens +
            last_ring_sf_row + sf_even_row_max) + 19
        self.wide_l1_sf = (l1_sf_local_max > 0xffffffff or
                           self.l1_sf_words + l1_sf_local_max > 0xffffffff)
        self.wide_routed_w1_sf = routed_w1_sf_local_max > 0xffffffff
        self.wide_shared_w1_sf = False
        if shared_experts > 0:
            last_shared_sf_row = ((capacity + block_m - 1) // block_m - 1) * sf_block_m
            shared_w1_sf_local_max = 4 * (
                (shared_experts * intermediate // 128 - 1) *
                self.shared_sf_tokens + last_shared_sf_row + sf_even_row_max) + 19
            self.wide_shared_w1_sf = shared_w1_sf_local_max > 0xffffffff

    @cute.jit
    def _initialize_shared_storage(self, slab: cute.Pointer,
                                    physical_warp: Uint32, lane: Uint32) -> None:
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        if physical_warp == 0:
            if elect_one_sync():
                if cutlass.const_expr(self.role == 1):
                    bulk_zero_shared(
                        slab + self.shared_layout.expert_counts,
                        Uint32(((self.num_ranks * self.local_experts * 4 + 1023) //
                                1024) * 1024))
        elif physical_warp == 1:
            if cutlass.const_expr(self.role == 1):
                bars = cute.recast_ptr(slab + self.shared_layout.dispatch_bars, dtype=Uint64)
                for index in cutlass.range(
                        Int32(lane), Int32(self.dispatch_warps), Int32(32), unroll_full=True):
                    cute.arch.mbarrier_init(bars + index, 1)
            cute.arch.mbarrier_init_fence()
        elif physical_warp == 2:
            # CuTe 4.8.1's single-thread-per-CTA docstring is conservative
            # for disjoint barriers; the wrapper does not elect a lane.
            # Each object below has exactly one initializer. Every such
            # lane executes the init fence before the cluster rendezvous
            # publishes initialization to consumers. The constructor's
            # math_stages <= 32 bound keeps all stages within one warp.
            if cutlass.const_expr(self.role == 2):
                cta_rank, _, _ = cute.arch.block_in_cluster_idx()
                data_full = cute.recast_ptr(slab + self.shared_layout.data_full, dtype=Uint64)
                data_empty = cute.recast_ptr(slab + self.shared_layout.data_empty, dtype=Uint64)
                if lane < self.math_stages:
                    if cta_rank == 0:
                        cute.arch.mbarrier_init(data_full + lane, 4)
                    cute.arch.mbarrier_init(data_empty + lane, 1)
                accum_full = cute.recast_ptr(slab + self.shared_layout.accum_full, dtype=Uint64)
                accum_empty = cute.recast_ptr(slab + self.shared_layout.accum_empty, dtype=Uint64)
                if lane < self.accum_stages:
                    cute.arch.mbarrier_init(accum_full + lane, 1)
                    if cta_rank == 0:
                        cute.arch.mbarrier_init(accum_empty + lane, TASK_EMPTY_CONSUMERS)
                full = cute.recast_ptr(slab + self.shared_layout.task_full, dtype=Uint64)
                empty = cute.recast_ptr(slab + self.shared_layout.task_empty, dtype=Uint64)
                if lane < 2:
                    cute.arch.mbarrier_init(full + lane, 1)
                    if cta_rank == 0:
                        cute.arch.mbarrier_init(empty + lane, TASK_EMPTY_CONSUMERS)
            elif cutlass.const_expr(self.role == 3):
                combine = cute.recast_ptr(slab + self.shared_layout.combine_bars, dtype=Uint64)
                if lane < COMBINE_WARPS:
                    cute.arch.mbarrier_init(combine + lane, 1)
            cute.arch.mbarrier_init_fence()
        elif physical_warp == 3:
            if cutlass.const_expr(self.role == 2):
                tptr = cute.recast_ptr(slab + self.shared_layout.tmem_ptr, dtype=Uint32)
                cute.arch.alloc_tmem(self.tmem_cols, tptr, is_two_cta=True)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        cute.arch.griddepcontrol_wait()

    @cute.jit
    def _role3_dispatch(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                        rank_idx: Uint32, tid: Uint32,
                        block: Uint32, recv_ptr: cute.Pointer, stats: cute.Tensor,
                        ready_epoch: Uint64) -> None:
        # Publish completed peer FC2 stores even with no routed tokens.
        # Epochs distinguish late old publications without resetting ready.
        if block == 0 and tid < self.num_ranks:
            dst = cute.make_ptr(
                Uint64, mapped_peer_base(peer_ptrs, Uint32(tid), self.num_ranks) +
                Uint64(SIGNAL_OFFSETS['combine_ready_grid_idx']) +
                Uint64(rank_idx * 8), cute.AddressSpace.gmem,
                assumed_align=8)
            st_release_sys_u64(dst, ready_epoch)
        # Dual-buffered receive sums: this replay consumes the generation
        # selected by the persistent parity word; the bump below flips it
        # for the next replay, and role 1 of the replay that retires this
        # generation clears it at entry.
        gen = Uint32(ready_epoch - Uint64(1)) & Uint32(1)
        recv_ptr = recv_ptr + gen * Uint32(self.local_experts * 2)
        # Role1 acquired every expert sum, including zero-count experts,
        # before publishing this summary. Normal graph completion dependencies
        # also retire every role2 ring access before this footer executes.
        pool_blocks = signals[SIGNAL_OFFSETS['routed_pool_blocks'] // 4]
        # All CTAs use the immutable role1 epoch, so block0 may flip parity
        # without a grid-wide cache-completion barrier. The next dispatch
        # clears this bank only after the entire local finalize grid ends.
        # That clear precedes count publication; all-rank sum acquires in
        # the intervening call order it before any peer can publish into
        # the same bank again. Zero-count experts must join too.
        if block == Uint32(0) and tid == Uint32(0):
            red_gpu_add_u32(
                signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4,
                Uint32(1))
        warpgroup_reg_dealloc(48 if self.local_experts <= 64 else 96)

        if block == 0:
            send_off = SIGNAL_OFFSETS['expert_send_count'] // 4
            for expert_iter in cutlass.range(
                    Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                    128, unroll_full=True):
                expert_idx = Uint32(expert_iter)
                st_global_u64(signals.iterator + send_off + expert_idx * Uint32(2),
                              Uint64(0))
            if tid // Uint32(32) == Uint32(0):
                if elect_one_sync():
                    for locality in cutlass.range(2, unroll_full=True):
                        signals[SIGNAL_OFFSETS['l1_task_count'] // 4 + locality] = Uint32(0)
                        signals[SIGNAL_OFFSETS['l2_task_count'] // 4 + locality] = Uint32(0)
                        signals[SIGNAL_OFFSETS['shared_l1_task_count'] // 4 + locality] = Uint32(0)
                        signals[SIGNAL_OFFSETS['shared_l2_task_count'] // 4 + locality] = Uint32(0)
            cute.arch.sync_warp()
            shared_block = Uint32(tid)
            while shared_block < Uint32((self.capacity + 7) // 8):
                signals[SIGNAL_OFFSETS['shared_l2_full_count'] // 4 +
                        shared_block] = Uint32(0)
                shared_block += Uint32(128)
            cute.arch.sync_warp()
        if cutlass.const_expr(self.has_stats):
            local_expert = block * Uint32(128) + tid
            while local_expert < Uint32(self.local_experts):
                received = Uint32(ld_global_u64(recv_ptr + local_expert * Uint32(2)))
                red_gpu_add_s32(stats.iterator + local_expert, Int32(received))
                local_expert += Uint32(self.finalize_sms * 128)

        # The padded expert ranges concatenate to [0,P). Modulo R, their
        # used slots are exactly [0,min(P,R)); clear each of those slots once.
        ring_block = block * Uint32(128) + tid
        while ring_block < min(pool_blocks, Uint32(self.ring_blocks)):
            signals[SIGNAL_OFFSETS['l1_full_count'] // 4 + ring_block] = Uint32(0)
            signals[SIGNAL_OFFSETS['l1_empty_count'] // 4 + ring_block] = Uint32(0)
            st_global_u64(
                signals.iterator + SIGNAL_OFFSETS['l2_full_mask'] // 4 +
                ring_block * Uint32(2), Uint64(0))
            signals[SIGNAL_OFFSETS['l2_empty_count'] // 4 + ring_block] = Uint32(0)
            ring_block += Uint32(self.finalize_sms * 128)

        # The former tail NVLink barrier (tag 2) is gone: the receive
        # signals are dual-buffered by recv_parity, so peers' next publish
        # targets the alternate generation and ordering is carried by the
        # all-expert sum acquires in the intervening call. Selected-peer
        # ready acquires protect each source's single-buffer inputs/indices.

    @cute.jit
    def _role3_combine(self, signals: cute.Tensor, rank_idx: Uint32,
                       num_tokens: Uint32, y_out: cute.Tensor,
                       block, physical_warp, lane,
                       combine_smem: cute.Pointer,
                       combine_bars: cute.Pointer, ready_epoch: Uint64) -> None:
        combine_chunks = self.combine_chunks
        chunk_bytes = self.combine_chunk_bytes
        u4_per_lane = self.combine_uint4_per_lane
        warpgroup_reg_alloc(self.epi_registers)
        epi_warp = Uint32(physical_warp - 4)
        ready = signals.iterator + SIGNAL_OFFSETS['combine_ready_grid_idx'] // 4
        if cutlass.const_expr(not self.finalize_output):
            # No token reduction exists to carry selected-peer acquires.
            peer = epi_warp
            while peer < Uint32(self.num_ranks):
                wait_sys_u64_eq(ready + peer * 2, ready_epoch, rank_idx, peer)
                peer += Uint32(COMBINE_WARPS)
            sync_aligned(COMBINE_WARPS * 32, 2)
        if cutlass.const_expr(self.finalize_output):
            # One warp owns a token chunk. Every routed slot of the chunk is
            # fetched into its own plane under a single barrier so the
            # reduction exposes one TMA latency instead of one per slot. The
            # output plane aliases slot 0 and is retired before the next
            # chunk overwrites it.
            combine = cute.recast_ptr(
                signals.iterator, dtype=Uint8) + \
                Uint64(self.combine_words) * Uint64(4)
            output = cute.recast_ptr(y_out.iterator, dtype=Uint8)
            routes = cute.recast_ptr(
                signals.iterator + self.topk_offset_words, dtype=Uint64)
            combine_slots = self.topk + (1 if self.shared_experts > 0 else 0)
            # Keep the historical reduction order: shared slot, then routes.
            slot_of = ([self.topk] if self.shared_experts > 0 else []) + \
                list(range(self.topk))
            store_smem = combine_smem + Uint32(epi_warp * chunk_bytes)
            store_words = cute.recast_ptr(store_smem, dtype=Uint32)
            bar = combine_bars + epi_warp
            phase = Uint32(0)
            token_chunk = epi_warp * Uint32(self.finalize_sms) + block
            if cutlass.const_expr(self.shared_experts > 0):
                if token_chunk < num_tokens * Uint32(combine_chunks):
                    _initialize_private_combine_bar(bar)
            while token_chunk < num_tokens * Uint32(combine_chunks):
                token = token_chunk // Uint32(combine_chunks)
                chunk = token_chunk % Uint32(combine_chunks)
                chunk_offset = chunk * Uint32(chunk_bytes)
                lane_valid = False
                routed_peer = Uint32(0)
                if lane < self.topk:
                    routed_expert = routes[token * Uint32(self.topk) + lane]
                    lane_valid = routed_expert < \
                        Uint64(self.num_ranks * self.local_experts)
                    if lane_valid:
                        routed_peer = Uint32(routed_expert) // Uint32(self.local_experts)
                # All lanes vote; invalid routes never form a peer pointer.
                # Duplicate owners may acquire the same publication twice.
                peer_ready = not lane_valid
                wait_start = read_clock64()
                while cute.arch.vote_ballot_sync(peer_ready) != Uint32(0xffffffff):
                    if lane_valid and not peer_ready:
                        peer_ready = ld_acquire_sys_u64(
                            ready + routed_peer * Uint32(2)) == ready_epoch
                    if Uint64(read_clock64() - wait_start) >= Uint64(60 * 2000000000):
                        if not peer_ready:
                            cute.printf('GC token readiness timeout rank={} peer={}',
                                        rank_idx, routed_peer)
                            device_trap()
                # Transfer selected-lane acquires to the elected TMA loader.
                cute.arch.sync_warp()
                if cutlass.const_expr(self.shared_experts > 0):
                    if lane == self.topk:
                        lane_valid = True
                mask = Uint32(cute.arch.vote_ballot_sync(lane_valid))
                # Retire the previous chunk's output before any fetch plane
                # reuses the aliased slot-0 region, also when this chunk has
                # no valid route and only writes zeros. The wait is per-thread
                # and only the elected lane issued the store, so converge the
                # warp before any lane touches the plane.
                tma_store_wait_0()
                cute.arch.sync_warp()
                if mask != Uint32(0):
                    if elect_one_sync():
                        expect = Uint32(0)
                        for g in cutlass.range_constexpr(combine_slots):
                            if ((mask >> Uint32(slot_of[g])) & Uint32(1)) != Uint32(0):
                                expect += Uint32(chunk_bytes)
                        cute.arch.mbarrier_arrive_and_expect_tx(bar, expect)
                        for g in cutlass.range_constexpr(combine_slots):
                            if ((mask >> Uint32(slot_of[g])) & Uint32(1)) != Uint32(0):
                                src = combine + \
                                    (Uint64(slot_of[g]) * Uint64(self.capacity) +
                                     Uint64(token)) * Uint64(self.hidden * 2) + \
                                    Uint64(chunk_offset)
                                bulk_g2s_1d(
                                    combine_smem +
                                    (Uint32(slot_of[g]) * Uint32(COMBINE_WARPS) + epi_warp) *
                                    Uint32(chunk_bytes), src,
                                    Uint32(chunk_bytes), bar)
                    cute.arch.sync_warp()
                    cute.arch.mbarrier_wait(bar, phase)
                    phase ^= Uint32(1)
                reduced = cute.make_rmem_tensor((u4_per_lane * 8,), Float32)
                for index in cutlass.range_constexpr(u4_per_lane * 8):
                    reduced[index] = Float32(0)
                for g in cutlass.range_constexpr(combine_slots):
                    if ((mask >> Uint32(slot_of[g])) & Uint32(1)) != Uint32(0):
                        load_words = cute.recast_ptr(
                            combine_smem + (Uint32(slot_of[g]) * Uint32(COMBINE_WARPS) + epi_warp) *
                            Uint32(chunk_bytes), dtype=Uint32)
                        for j in cutlass.range_constexpr(u4_per_lane):
                            words = ld_shared_u32x4(
                                load_words + (j * 32 + lane) * 4)
                            for q in cutlass.range_constexpr(4):
                                packed = words[q]
                                reduced[j * 8 + q * 2] = \
                                    accumulate_bf16_into_f32(
                                        reduced[j * 8 + q * 2],
                                        Uint16(packed & Uint32(0xffff)))
                                reduced[j * 8 + q * 2 + 1] = \
                                    accumulate_bf16_into_f32(
                                        reduced[j * 8 + q * 2 + 1],
                                        Uint16(packed >> Uint32(16)))
                cute.arch.fence_view_async_shared()

                for j in cutlass.range_constexpr(u4_per_lane):
                    packed_words = cute.make_rmem_tensor((4,), Uint32)
                    for q in cutlass.range_constexpr(4):
                        packed_words[q] = pack_bf16_pair(
                            reduced[j * 8 + q * 2],
                            reduced[j * 8 + q * 2 + 1])
                    st_shared_u32x4(
                        store_words + (j * 32 + lane) * 4,
                        packed_words[0], packed_words[1],
                        packed_words[2], packed_words[3])
                cute.arch.sync_warp()
                if elect_one_sync():
                    tma_store_fence()
                    dst = output + Uint64(token) * \
                        Uint64(self.hidden * 2) + Uint64(chunk_offset)
                    bulk_s2g_1d(dst, store_smem, Uint32(chunk_bytes))
                    tma_store_arrive()
                cute.arch.sync_warp()
                token_chunk += Uint32(self.finalize_sms * COMBINE_WARPS)
            # A CTA exit does not retire a committed bulk async group, so
            # drain the last chunk's store before the shared source is released.
            tma_store_wait_0()


    @cute.kernel
    def kernel(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
               rank_idx: Uint32, num_tokens: Uint32,
               tensor_maps=None, y_out=None, stats=None,
               activation_clamp=None, activation_alpha=None,
               tensor_map_1=None, tensor_map_2=None, tensor_map_3=None,
               tensor_map_4=None, tensor_map_5=None, tensor_map_6=None,
               tensor_map_7=None, tensor_map_8=None, tensor_map_9=None,
               tensor_map_10=None, tensor_map_11=None, tensor_map_12=None,
               tensor_map_13=None, tensor_map_14=None, tensor_map_15=None,
               tensor_map_16=None, tensor_map_17=None,
               tensor_map_18=None, tensor_map_19=None,
               tensor_map_20=None, tensor_map_21=None):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        physical_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tid % 32

        if cutlass.const_expr(self.role == 1):
            # First count local routes, reserve a distinct source-index span,
            # then publish the receive count only after writing that span.
            # Route each global expert to its owner rank and local expert.
            # Each local expert has an independent receive count and a padded
            # pool-block prefix; ring slots remain shared by all experts.
            dispatch_smem = cutlass.utils.SmemAllocator()
            slab = cute.recast_ptr(
                dispatch_smem.allocate(self.shared_layout.field_end, 1024), dtype=Uint8)
            expert_token_count = cute.recast_ptr(
                slab + self.shared_layout.expert_counts, dtype=Uint32)
            pull = cute.recast_ptr(slab + self.shared_layout.pull, dtype=Uint32)
            pull_bar = cute.recast_ptr(slab + self.shared_layout.dispatch_bars, dtype=Uint64)
            self._initialize_shared_storage(slab, Uint32(physical_warp), Uint32(lane))
            warpgroup_reg_dealloc(48 if self.local_experts <= 64 else 96)
            # Dual-buffered receive signals: replay N publishes and reads
            # generation parity & 1; role 3 flips the persistent parity
            # word after consuming each replay, and the replay that retires
            # a generation clears it here, so no cross-rank tail barrier is
            # needed.  Role 3 reads the epoch published below instead of a
            # grid barrier.
            parity = ld_volatile_u32(
                signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4)
            retired = (parity + Uint32(1)) & Uint32(1)
            gen = parity & Uint32(1)
            if block == Uint32(0):
                if tid == Uint32(0):
                    # The normal role1 -> role3 graph completion dependency
                    # makes this plain store visible; role3 never polls it
                    # concurrently with this dispatch.
                    signals[SIGNAL_OFFSETS['role3_epoch'] // 4] = parity + Uint32(1)
                retired_sum = signals.iterator + \
                    SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                    retired * Uint32(self.local_experts * 2)
                for retired_iter in cutlass.range(
                        Uint32(tid), Uint32(self.local_experts),
                        self.dispatch_warps * 32, unroll_full=True):
                    st_global_u64(
                        retired_sum + Uint32(retired_iter) * Uint32(2), Uint64(0))
                retired_recv = signals.iterator + \
                    SIGNAL_OFFSETS['expert_recv_count'] // 4 + \
                    retired * Uint32(self.num_ranks * self.local_experts * 2)
                for retired_iter in cutlass.range(
                        Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                        self.dispatch_warps * 32, unroll_full=True):
                    st_global_u64(
                        retired_recv + Uint32(retired_iter) * Uint32(2), Uint64(0))
            recv_ptr = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                gen * Uint32(self.local_experts * 2)
            send_ptr = signals.iterator + SIGNAL_OFFSETS['expert_send_count'] // 4
            routes = cute.make_tensor(
                cute.recast_ptr(signals.iterator + self.topk_offset_words, dtype=Uint64),
                cute.make_layout((self.capacity * self.topk,)))
            source_idx = cute.make_tensor(
                cute.recast_ptr(signals.iterator + SIGNAL_OFFSETS['size'] // 4,
                                dtype=Uint32),
                cute.make_layout((self.num_ranks * self.local_experts * self.capacity,)))
            # Reference read_topk_idx: adjacent lanes own adjacent topk
            # entries, so one warp processes floor(32 / topk) tokens per
            # iteration. Lanes past that whole-token group stay inactive.
            tokens_per_warp = 32 // self.topk
            active_route_lanes = tokens_per_warp * self.topk
            route_base = Uint32((block * self.dispatch_warps + physical_warp) *
                                tokens_per_warp)
            route_stride = self.dispatch_sms * self.dispatch_warps * tokens_per_warp
            for route_iter in cutlass.range(
                    route_base, num_tokens, route_stride, unroll_full=True):
                route_base = Uint32(route_iter)
                route_token = route_base + Uint32(lane // self.topk)
                token_topk = route_base * Uint32(self.topk) + Uint32(lane)
                if lane < active_route_lanes and route_token < num_tokens:
                    expert = routes[token_topk]
                    if expert < Uint64(self.num_ranks * self.local_experts):
                        atom_shared_add_u32(expert_token_count + Uint32(expert),
                                            Uint32(1))
                cute.arch.sync_warp()
            cute.arch.sync_threads()
            for expert_iter in cutlass.range(
                    Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                    self.dispatch_warps * 32, unroll_full=True):
                expert_idx = Uint32(expert_iter)
                old_send = atom_add_u64(
                    send_ptr + expert_idx * Uint32(2),
                    (Uint64(1) << 32) | Uint64(expert_token_count[expert_idx]))
                expert_token_count[expert_idx] = Uint32(old_send)
            cute.arch.sync_threads()
            route_base = Uint32((block * self.dispatch_warps + physical_warp) *
                                tokens_per_warp)
            for route_iter in cutlass.range(
                    route_base, num_tokens, route_stride, unroll_full=True):
                route_base = Uint32(route_iter)
                route_token = route_base + Uint32(lane // self.topk)
                token_topk = route_base * Uint32(self.topk) + Uint32(lane)
                if lane < active_route_lanes and route_token < num_tokens:
                    expert = routes[token_topk]
                    if expert < Uint64(self.num_ranks * self.local_experts):
                        dst_rank = Uint32(expert) // Uint32(self.local_experts)
                        dst_local = Uint32(expert) % Uint32(self.local_experts)
                        slot = atom_shared_add_u32(
                            expert_token_count + Uint32(expert), Uint32(1))
                        dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
                        dst_index = cute.make_ptr(
                            Uint32, dst_base + Uint64(SIGNAL_OFFSETS['size']) +
                            Uint64(((dst_local * Uint32(self.num_ranks) + rank_idx) *
                                    Uint32(self.capacity) +
                                    slot) *
                                   Uint32(4)), cute.AddressSpace.gmem)
                        dst_index[0] = token_topk
                cute.arch.sync_warp()
            grid_sync(signals, Uint32(block), Uint32(tid), self.dispatch_sms, 0)
            if block == 0:
                for expert_iter in cutlass.range(
                        Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                        self.dispatch_warps * 32, unroll_full=True):
                    expert_idx = Uint32(expert_iter)
                    dst_rank = expert_idx // Uint32(self.local_experts)
                    dst_local = expert_idx % Uint32(self.local_experts)
                    dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
                    status = ld_global_u64(
                        signals.iterator + SIGNAL_OFFSETS['expert_send_count'] // 4 +
                        expert_idx * Uint32(2))
                    remote_count = cute.make_ptr(
                        Uint64, dst_base + Uint64(SIGNAL_OFFSETS['expert_recv_count']) +
                        Uint64(gen * Uint32(self.num_ranks * self.local_experts * 8)) +
                        Uint64((rank_idx * Uint32(self.local_experts) + dst_local) *
                               Uint32(8)), cute.AddressSpace.gmem,
                        assumed_align=8)
                    st_global_u64(remote_count, status & Uint64(0xffffffff))
                # The grid acquire carries all route writes. Transfer
                # them, plus every per-rank count store,
                # to every publisher before the system release fence.
                cute.arch.sync_threads()
                fence_acq_rel_sys()
                for expert_iter in cutlass.range(
                        Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                        self.dispatch_warps * 32, unroll_full=True):
                    expert_idx = Uint32(expert_iter)
                    dst_rank = expert_idx // Uint32(self.local_experts)
                    dst_local = expert_idx % Uint32(self.local_experts)
                    dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
                    status = ld_global_u64(
                        signals.iterator + SIGNAL_OFFSETS['expert_send_count'] // 4 +
                        expert_idx * Uint32(2))
                    remote_sum = cute.make_ptr(
                        Uint64, dst_base + Uint64(SIGNAL_OFFSETS['expert_recv_count_sum']) +
                        Uint64(gen * Uint32(self.local_experts * 8)) +
                        Uint64(dst_local * Uint32(8)),
                        cute.AddressSpace.gmem, assumed_align=8)
                    atom_sys_add_u64(remote_sum, status)
            cute.arch.sync_threads()
            # Each per-expert acquire below imports every rank's route/count
            # stores, replacing the dispatch rank barrier. Include zero-token
            # experts: this join orders prior combine reads before new FC2
            # stores, and generation N's clear before N+2 can publish.
            # GCMoEScheduler::fetch_expert_recv_count: each lane caches
            # expert wave*32+lane, including zero for padded lanes.
            expert_counts = cute.make_rmem_tensor(
                ((self.local_experts + 31) // 32,), Uint32)
            for wave in cutlass.range_constexpr(
                    (self.local_experts + 31) // 32):
                expert_idx = Uint32(wave * 32) + lane
                observed = Uint64(0)
                if expert_idx < Uint32(self.local_experts):
                    wait_start = read_clock64()
                    while Uint32(observed >> 32) != Uint32(
                            self.dispatch_sms * self.num_ranks):
                        observed = ld_volatile_u64(
                            recv_ptr + expert_idx * Uint32(2))
                        if Uint64(read_clock64() - wait_start) >= Uint64(
                                self.nvlink_timeout_secs * 2000000000):
                            if block == Uint32(0):
                                cute.printf(
                                    'GC recv-count timeout role=1 rank={} expert={} gen={} observed_hi={} expected={}',
                                    rank_idx, expert_idx, gen, Uint32(observed >> 32),
                                    self.dispatch_sms * self.num_ranks)
                            device_trap()
                    observed = ld_acquire_sys_u64(
                        recv_ptr + expert_idx * Uint32(2))
                expert_counts[wave] = Uint32(observed)
            cute.arch.sync_warp()

            if block == Uint32(0) and physical_warp == Uint32(0):
                own_pool_blocks = Uint32(0)
                for wave in cutlass.range_constexpr(
                        (self.local_experts + 31) // 32):
                    own_pool_blocks += (expert_counts[wave] +
                                        Uint32(self.block_m - 1)) // Uint32(self.block_m)
                total_pool_blocks = cute.arch.warp_redux_sync(own_pool_blocks, 'add')
                if lane == Uint32(0):
                    signals[SIGNAL_OFFSETS['routed_pool_blocks'] // 4] = total_pool_blocks

            # One global warp cursor traverses concatenated, unpadded expert
            # tokens; the pool-block prefix advances by each expert's padded
            # M-block count. Only the TMA and final L1 release elect a lane.
            token_idx = Uint32(block) * Uint32(self.dispatch_warps) + Uint32(physical_warp)
            current_expert = Int32(-1)
            expert_start = Uint32(0)
            expert_end = Uint32(0)
            pool_block_offset = Uint32(0)
            pull_phase = Uint32(0)
            rank_waves = (self.num_ranks + 31) // 32
            current_expert, expert_start, expert_end, pool_block_offset = \
                advance_expert(
                    expert_counts, token_idx, current_expert, expert_start,
                    expert_end, pool_block_offset, lane, self.local_experts,
                    self.block_m)
            while current_expert < Int32(self.local_experts):
                received = expert_end - expert_start
                pull_token = token_idx - expert_start
                pool_block = pool_block_offset + pull_token // self.block_m
                pool_token = pool_block * Uint32(self.block_m) + pull_token % Uint32(self.block_m)
                ring_block = pool_block % self.ring_blocks
                generation = pool_block // self.ring_blocks

                # Each lane owns one source rank in each 32-rank wave.
                # Reduce across its register waves before the warp reduction.
                remaining = cute.make_rmem_tensor((rank_waves,), Uint32)
                # Reload per token to isolate the immutable source count from
                # the destructively peeled loop-carried remainder.
                for wave in cutlass.range_constexpr(rank_waves):
                    source_rank = Uint32(wave * 32) + lane
                    remaining[wave] = Uint32(0)
                    if source_rank < Uint32(self.num_ranks):
                        count_ptr = signals.iterator + \
                            SIGNAL_OFFSETS['expert_recv_count'] // 4 + \
                            gen * Uint32(self.num_ranks * self.local_experts * 2) + \
                            (source_rank * Uint32(self.local_experts) +
                             Uint32(current_expert)) * Uint32(2)
                        remaining[wave] = Uint32(
                            ld_global_u64(count_ptr))
                src_rank = Uint32(0)
                src_slot = Uint32(0)
                if cutlass.const_expr(self.num_ranks <= 8):
                    # Concatenate source-rank intervals without changing the
                    # source-index payload or its output metadata encoding.
                    prefix = Uint32(0)
                    for source_rank in cutlass.range_constexpr(self.num_ranks):
                        count = cute.arch.shuffle_sync(remaining[0], source_rank)
                        if pull_token >= prefix and pull_token < prefix + count:
                            src_rank = Uint32(source_rank)
                            src_slot = pull_token - prefix
                        prefix += count
                else:
                    peel_slot = Uint32(pull_token)
                    peel_offset = Uint32(0)
                    selected = False
                    while not selected:
                        lane_active = Uint32(0)
                        lane_minimum = Uint32(0xffffffff)
                        for wave in cutlass.range_constexpr(rank_waves):
                            count = remaining[wave]
                            lane_active += Uint32(1) if count > Uint32(0) else Uint32(0)
                            if count > Uint32(0):
                                lane_minimum = min(lane_minimum, count)
                        active = cute.arch.warp_redux_sync(lane_active, 'add')
                        minimum = cute.arch.warp_redux_sync(lane_minimum, 'min')
                        round_tokens = active * minimum
                        if peel_slot < round_tokens:
                            position = peel_slot % active
                            seen = Uint32(0)
                            for wave in cutlass.range_constexpr(rank_waves):
                                mask = Uint32(cute.arch.vote_ballot_sync(
                                    remaining[wave] > Uint32(0)))
                                wave_active = Uint32(cute.arch.popc(mask))
                                if position >= seen and position < seen + wave_active:
                                    src_rank = Uint32(wave * 32) + warp_nth_set_bit(
                                        mask, position - seen + Uint32(1))
                                seen += wave_active
                            src_slot = peel_offset + peel_slot // active
                            selected = True
                        else:
                            peel_slot -= round_tokens
                            peel_offset += minimum
                            for wave in cutlass.range_constexpr(rank_waves):
                                remaining[wave] -= min(remaining[wave], minimum)
                token_topk = source_idx[(Uint32(current_expert) * Uint32(self.num_ranks) +
                                         src_rank) * Uint32(self.capacity) + src_slot]
                src_token = token_topk // Uint32(self.topk)
                topk_slot = token_topk % Uint32(self.topk)
                if generation > 0:
                    empty_ptr = signals.iterator + \
                        SIGNAL_OFFSETS['l1_empty_count'] // 4 + ring_block
                    target = generation * (2 * self.intermediate // 128)
                    while ld_acquire_gpu_u32(empty_ptr) < target:
                        pass
                src_base = mapped_peer_base(peer_ptrs, src_rank, self.num_ranks)
                src_x = cute.make_ptr(
                    Uint32, src_base + Uint64(self.input_x_words * 4) +
                    Uint64(src_token) * Uint64(self.hidden),
                    cute.AddressSpace.gmem, assumed_align=16)
                dst_x = cute.recast_ptr(
                    cute.recast_ptr(signals.iterator, dtype=Uint8) +
                    Uint64(self.l1_x_words) * Uint64(4) +
                    Uint64(pool_token % self.ring_tokens) * Uint64(self.hidden),
                    dtype=Uint32)
                warp_pull = pull + physical_warp * (self.pull_bytes // 4)
                warp_bar = pull_bar + physical_warp
                if elect_one_sync():
                    for chunk in cutlass.range_constexpr(self.pull_chunks):
                        byte_words = chunk * (self.pull_bytes // 4)
                        bulk_g2s_1d(warp_pull, src_x + byte_words,
                                     Uint32(self.pull_bytes), warp_bar)
                        cute.arch.mbarrier_arrive_and_expect_tx(warp_bar, self.pull_bytes)
                        if chunk + 1 < self.pull_chunks:
                            cute.arch.mbarrier_wait(warp_bar, pull_phase)
                            pull_phase ^= Uint32(1)
                            bulk_s2g_1d(dst_x + byte_words, warp_pull,
                                         Uint32(self.pull_bytes))
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0)
                cute.arch.sync_warp()

                sf_row = ring_block * ((self.block_m + 127) // 128 * 128) + \
                    ((pull_token % self.block_m) & ~127) + \
                    ((pull_token % self.block_m) & 31) * 4 + \
                    (((pull_token % self.block_m) >> 5) & 3)
                remote_sf_base = src_base + Uint64(self.input_sf_words * 4) + \
                    Uint64(src_token) * Uint64(self.hidden // 128 * 4)
                remote_weight = cute.make_ptr(
                    Uint32, src_base + Uint64(self.input_weight_words * 4) +
                    Uint64(token_topk) * Uint64(4), cute.AddressSpace.gmem)
                weight = remote_weight[0]
                if cutlass.const_expr((self.hidden // 128) % 4 == 0):
                    sf_vectors = self.hidden // 512
                    for q_round in cutlass.range(
                            (sf_vectors + 31) // 32, unroll_full=True):
                        q = q_round * 32 + lane
                        if q < sf_vectors:
                            remote_vec = cute.make_ptr(
                                Uint32, remote_sf_base + Uint64(q * Uint32(16)),
                                cute.AddressSpace.gmem, assumed_align=16)
                            values = ld_global_u32x4(remote_vec)
                            for component in cutlass.range_constexpr(4):
                                sf_col = q * 4 + component
                                if cutlass.const_expr(self.wide_l1_sf):
                                    l1_sf_bytes = cute.recast_ptr(
                                        signals.iterator, dtype=Uint8) + \
                                        Uint64(self.l1_sf_words) * Uint64(4)
                                    sf_word = Uint64(sf_col) * \
                                        Uint64(self.sf_ring_tokens) + Uint64(sf_row)
                                    cute.recast_ptr(
                                        l1_sf_bytes + sf_word * Uint64(4),
                                        dtype=Uint32)[0] = values[component]
                                else:
                                    signals[self.l1_sf_words +
                                            sf_col * self.sf_ring_tokens + sf_row] = \
                                        values[component]
                else:
                    for sf_round in cutlass.range(
                            (self.hidden // 128 + 31) // 32, unroll_full=True):
                        sf_col = sf_round * 32 + lane
                        if sf_col < self.hidden // 128:
                            remote_word = cute.make_ptr(
                                Uint32, remote_sf_base + Uint64(sf_col * Uint32(4)),
                                cute.AddressSpace.gmem, assumed_align=4)
                            if cutlass.const_expr(self.wide_l1_sf):
                                l1_sf_bytes = cute.recast_ptr(
                                    signals.iterator, dtype=Uint8) + \
                                    Uint64(self.l1_sf_words) * Uint64(4)
                                sf_word = Uint64(sf_col) * \
                                    Uint64(self.sf_ring_tokens) + Uint64(sf_row)
                                cute.recast_ptr(
                                    l1_sf_bytes + sf_word * Uint64(4),
                                    dtype=Uint32)[0] = remote_word[0]
                            else:
                                signals[self.l1_sf_words +
                                        sf_col * self.sf_ring_tokens + sf_row] = \
                                    remote_word[0]
                cute.arch.sync_warp()
                if elect_one_sync():
                    signals[self.l1_weight_words + pool_token % self.ring_tokens] = weight
                    signals[self.metadata_words + pool_token * 3] = src_rank
                    signals[self.metadata_words + pool_token * 3 + 1] = src_token
                    signals[self.metadata_words + pool_token * 3 + 2] = topk_slot
                    # The last remote G2S overlapped weight/SF work. Drain
                    # it, store the chunk, then publish W1 after S2G finishes.
                    cute.arch.mbarrier_wait(warp_bar, pull_phase)
                    pull_phase ^= Uint32(1)
                    final_byte_words = (self.pull_chunks - 1) * (self.pull_bytes // 4)
                    bulk_s2g_1d(dst_x + final_byte_words, warp_pull,
                                 Uint32(self.pull_bytes))
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)
                    is_last_token = token_idx == expert_end - Uint32(1)
                    full_increment = Uint32(self.block_m) - (pull_token % self.block_m) \
                        if is_last_token else Uint32(1)
                    red_release_gpu_add_u32(
                        signals.iterator + SIGNAL_OFFSETS['l1_full_count'] // 4 + ring_block,
                        Uint32(full_increment))
                cute.arch.sync_warp()
                token_idx += Uint32(self.dispatch_sms * self.dispatch_warps)
                current_expert, expert_start, expert_end, pool_block_offset = \
                    advance_expert(
                        expert_counts, token_idx, current_expert, expert_start,
                        expert_end, pool_block_offset, lane, self.local_experts,
                        self.block_m)
        elif cutlass.const_expr(self.role == 2):
            if physical_warp == 0:
                # Each descriptor has its own aligned by-value kernel
                # parameter, in the configuration's map order.
                prefetch_tensormap(tensor_maps.iterator)
                prefetch_tensormap(tensor_map_1.iterator)
                prefetch_tensormap(tensor_map_2.iterator)
                prefetch_tensormap(tensor_map_3.iterator)
                prefetch_tensormap(tensor_map_4.iterator)
                prefetch_tensormap(tensor_map_5.iterator)
                prefetch_tensormap(tensor_map_6.iterator)
                prefetch_tensormap(tensor_map_7.iterator)
                prefetch_tensormap(tensor_map_8.iterator)
                prefetch_tensormap(tensor_map_9.iterator)
                prefetch_tensormap(tensor_map_10.iterator)
                prefetch_tensormap(tensor_map_11.iterator)
                prefetch_tensormap(tensor_map_12.iterator)
                prefetch_tensormap(tensor_map_13.iterator)
                prefetch_tensormap(tensor_map_14.iterator)
                prefetch_tensormap(tensor_map_15.iterator)
                prefetch_tensormap(tensor_map_16.iterator)
                prefetch_tensormap(tensor_map_17.iterator)
                if cutlass.const_expr(self.use_small_tile_path):
                    prefetch_tensormap(tensor_map_18.iterator)
                    prefetch_tensormap(tensor_map_19.iterator)
                    prefetch_tensormap(tensor_map_20.iterator)
                    prefetch_tensormap(tensor_map_21.iterator)
            smem = cutlass.utils.SmemAllocator()
            slab = cute.recast_ptr(
                smem.allocate(self.shared_layout.field_end, 1024), dtype=Uint8)
            full = cute.recast_ptr(slab + self.shared_layout.task_full, dtype=Uint64)
            empty = cute.recast_ptr(slab + self.shared_layout.task_empty, dtype=Uint64)
            task = cute.recast_ptr(slab + self.shared_layout.tasks, dtype=Uint32)
            cd_smem = slab + self.shared_layout.cd
            data_a = slab + self.shared_layout.data_a
            data_b = slab + self.shared_layout.data_b
            data_sfa = cute.recast_ptr(slab + self.shared_layout.data_sfa, dtype=Uint32)
            data_sfb = cute.recast_ptr(slab + self.shared_layout.data_sfb, dtype=Uint32)
            epi_amax = cute.recast_ptr(slab + self.shared_layout.amax, dtype=Float32)
            data_full = cute.recast_ptr(slab + self.shared_layout.data_full, dtype=Uint64)
            data_empty = cute.recast_ptr(slab + self.shared_layout.data_empty, dtype=Uint64)
            accum_full = cute.recast_ptr(slab + self.shared_layout.accum_full, dtype=Uint64)
            accum_empty = cute.recast_ptr(slab + self.shared_layout.accum_empty, dtype=Uint64)
            tptr = cute.recast_ptr(slab + self.shared_layout.tmem_ptr, dtype=Uint32)
            cta_rank, _, _ = cute.arch.block_in_cluster_idx()
            self._initialize_shared_storage(slab, Uint32(physical_warp), Uint32(lane))
            # Warpgroup-uniform deallocation/allocation. The first WG
            # runs load/MMA/scheduler; the other two run epilogue.
            if physical_warp < 4:
                warpgroup_reg_dealloc(self.non_epi_registers)
            else:
                warpgroup_reg_alloc(self.epi_registers)
            iteration = Uint32(0)
            # Dual-buffered receive sums: poll the generation selected by
            # the persistent parity word (volatile read, fresh from L2).
            gen = ld_volatile_u32(
                signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4) & Uint32(1)
            recv_ptr = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                gen * Uint32(self.local_experts * 2)
            l1_count_ptr = signals.iterator + SIGNAL_OFFSETS['l1_task_count'] // 4
            l2_count_ptr = signals.iterator + SIGNAL_OFFSETS['l2_task_count'] // 4
            shared_l1_count_ptr = signals.iterator + SIGNAL_OFFSETS['shared_l1_task_count'] // 4
            shared_l2_count_ptr = signals.iterator + SIGNAL_OFFSETS['shared_l2_task_count'] // 4
            expert_counts = cute.make_rmem_tensor(
                ((self.local_experts + 31) // 32,), Uint32)
            if cta_rank == 0 and physical_warp == 3:
                schedule_tasks(
                    task, full, empty, recv_ptr, l1_count_ptr, l2_count_ptr,
                    shared_l1_count_ptr, shared_l2_count_ptr, expert_counts,
                    iteration, Int32(lane), Int32(block), num_tokens,
                    self.block_m,
                    self.compute_sms,
                    self.dispatch_sms,
                    self.hidden,
                    self.intermediate,
                    self.l1_clusters,
                    self.l2_clusters,
                    self.local_experts,
                    self.num_ranks,
                    self.shared_experts,
                )
            # The same two-slot task stream drives both TMA producer warps
            # and the leader MMA warp. All three consume every valid phase.
            if physical_warp == 0:
                task_words = cute.make_tensor(task, cute.make_layout((16,)))
                task_iter = Uint32(0)
                data_stage = Uint32(0)
                data_phase = Uint32(0)
                running = True
                while running:
                    task_stage = task_iter % Uint32(2)
                    cute.arch.mbarrier_wait(full + task_stage,
                                            (task_iter // Uint32(2)) % Uint32(2))
                    task_phase = task_words[task_stage * 8]
                    running = task_phase != Uint32(GCBlockPhase.None_)
                    if running:
                        pool_block = Uint32(task_words[task_stage * 8 + 4])
                        valid_m = Uint32(task_words[task_stage * 8 + 5])
                        shape_k = Uint32(task_words[task_stage * 8 + 7])
                        n_cluster = Uint32(task_words[task_stage * 8 + 3])
                        local_expert = Uint32(task_words[task_stage * 8 + 1])
                        is_shared = Uint32(0)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_shared = Uint32(1) if task_phase >= \
                                Uint32(GCBlockPhase.SharedLinear1) else Uint32(0)
                        block_idx = pool_block % Uint32(self.ring_blocks)
                        if cutlass.const_expr(self.shared_experts > 0):
                            if is_shared != Uint32(0):
                                block_idx = pool_block
                        activation_map = tensor_maps.iterator
                        activation_sf_map = tensor_map_1.iterator
                        weight_map = tensor_map_2.iterator
                        weight_sf_map = tensor_map_3.iterator
                        if task_phase == Uint32(GCBlockPhase.Linear2):
                            activation_map = tensor_map_5.iterator
                            activation_sf_map = tensor_map_6.iterator
                            weight_map = tensor_map_7.iterator
                            weight_sf_map = tensor_map_8.iterator
                        if cutlass.const_expr(self.shared_experts > 0):
                            if task_phase == Uint32(GCBlockPhase.SharedLinear1):
                                activation_map = tensor_map_9.iterator
                                activation_sf_map = tensor_map_10.iterator
                                weight_map = tensor_map_11.iterator
                                weight_sf_map = tensor_map_12.iterator
                            elif task_phase == Uint32(GCBlockPhase.SharedLinear2):
                                activation_map = tensor_map_14.iterator
                                activation_sf_map = tensor_map_15.iterator
                                weight_map = tensor_map_16.iterator
                                weight_sf_map = tensor_map_17.iterator
                        # The MMA B operand consumes N/2 rows from each CTA.
                        # Keep the original stage pitch and full shared/fallback maps.
                        activation_rows = Uint32(self.block_m // 2)
                        if cutlass.const_expr(self.use_small_tile_path):
                            if is_shared == Uint32(0) and valid_m <= Uint32(32):
                                activation_rows = Uint32(16)
                                activation_map = tensor_map_18.iterator
                                if task_phase == Uint32(GCBlockPhase.Linear2):
                                    activation_map = tensor_map_19.iterator
                        if cutlass.const_expr(self.use_small_tile_path and self.block_m > 48):
                            if is_shared == Uint32(0) and valid_m > Uint32(32) and valid_m <= Uint32(48):
                                activation_rows = Uint32(24)
                                activation_map = tensor_map_20.iterator
                                if task_phase == Uint32(GCBlockPhase.Linear2):
                                    activation_map = tensor_map_21.iterator
                        # Keep this pair generic for the entire K loop.
                        # The conversion is full-warp, before elected TMA.
                        activation_map = generic_tensormap_ptr(activation_map)
                        activation_sf_map = generic_tensormap_ptr(activation_sf_map)
                        if task_phase == Uint32(GCBlockPhase.Linear1):
                            l1_full_ptr = signals.iterator + \
                                SIGNAL_OFFSETS['l1_full_count'] // 4 + block_idx
                            required = self.block_m * \
                                (pool_block // Uint32(self.ring_blocks) + Uint32(1))
                            while ld_acquire_gpu_u32(l1_full_ptr) != required:
                                pass
                        if cutlass.const_expr(self.shared_experts > 0):
                            if task_phase == Uint32(GCBlockPhase.SharedLinear2):
                                shared_full_ptr = signals.iterator + \
                                    SIGNAL_OFFSETS['shared_l2_full_count'] // 4 + block_idx
                                while ld_acquire_gpu_u32(shared_full_ptr) != \
                                        self.l1_clusters * self.shared_experts * 2:
                                    pass
                        # GCL2KBlockDependency belongs to this task, so
                        # the observed XOR-normalized mask survives all
                        # K128 waits and is reset for the next task.
                        mask_ptr = signals.iterator + \
                            SIGNAL_OFFSETS['l2_full_mask'] // 4 + \
                            (pool_block % Uint32(self.ring_blocks)) * 2
                        generation = pool_block // Uint32(self.ring_blocks)
                        expected = Uint64(0) if (generation & Uint32(1)) != Uint32(0) \
                            else Uint64(self.l1_full_mask)
                        pending = Uint64(self.l1_full_mask)
                        k_block = Uint32(0)
                        while k_block < shape_k // Uint32(128):
                            if task_phase == Uint32(GCBlockPhase.Linear2):
                                required_mask = Uint64(3) << (k_block * Uint32(2))
                                while (pending & required_mask) != Uint64(0):
                                    pending = ld_acquire_gpu_u64(mask_ptr) ^ expected
                            load_mixed_k128_tma(
                                activation_map, activation_sf_map,
                                weight_map, weight_sf_map,
                                data_a + data_stage * (self.block_m // 2 * 128),
                                data_sfa + data_stage * self.sf_block_m,
                                data_b + data_stage * (128 * 128),
                                data_sfb + data_stage * 128,
                                data_full + data_stage, data_empty + data_stage,
                                k_block, data_phase, block_idx,
                                n_cluster * Uint32(2) + cta_rank,
                                local_expert, valid_m, activation_rows, shape_k,
                                is_shared, cta_rank, 0,
                                self.block_m, self.sf_block_m,
                                self.shared_weight_gran_k,
                                self.shared_experts > 0,
                                self.shared_weight_cache_hint,
                                self.routed_weight_linger)
                            data_stage = Uint32(0) if data_stage == \
                                Uint32(self.math_stages - 1) else data_stage + Uint32(1)
                            data_phase ^= Uint32(data_stage == Uint32(0))
                            k_block += Uint32(1)
                    task_iter += Uint32(1)
            elif physical_warp == 1:
                task_words = cute.make_tensor(task, cute.make_layout((16,)))
                task_iter = Uint32(0)
                data_stage = Uint32(0)
                data_phase = Uint32(0)
                running = True
                while running:
                    task_stage = task_iter % Uint32(2)
                    cute.arch.mbarrier_wait(full + task_stage,
                                            (task_iter // Uint32(2)) % Uint32(2))
                    task_phase = task_words[task_stage * 8]
                    running = task_phase != Uint32(GCBlockPhase.None_)
                    if running:
                        pool_block = Uint32(task_words[task_stage * 8 + 4])
                        valid_m = Uint32(task_words[task_stage * 8 + 5])
                        shape_k = Uint32(task_words[task_stage * 8 + 7])
                        n_cluster = Uint32(task_words[task_stage * 8 + 3])
                        local_expert = Uint32(task_words[task_stage * 8 + 1])
                        is_shared = Uint32(0)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_shared = Uint32(1) if task_phase >= \
                                Uint32(GCBlockPhase.SharedLinear1) else Uint32(0)
                        block_idx = pool_block % Uint32(self.ring_blocks)
                        if cutlass.const_expr(self.shared_experts > 0):
                            if is_shared != Uint32(0):
                                block_idx = pool_block
                        activation_map = tensor_maps.iterator
                        activation_sf_map = tensor_map_1.iterator
                        weight_map = tensor_map_2.iterator
                        weight_sf_map = tensor_map_3.iterator
                        if task_phase == Uint32(GCBlockPhase.Linear2):
                            activation_map = tensor_map_5.iterator
                            activation_sf_map = tensor_map_6.iterator
                            weight_map = tensor_map_7.iterator
                            weight_sf_map = tensor_map_8.iterator
                        if cutlass.const_expr(self.shared_experts > 0):
                            if task_phase == Uint32(GCBlockPhase.SharedLinear1):
                                activation_map = tensor_map_9.iterator
                                activation_sf_map = tensor_map_10.iterator
                                weight_map = tensor_map_11.iterator
                                weight_sf_map = tensor_map_12.iterator
                            elif task_phase == Uint32(GCBlockPhase.SharedLinear2):
                                activation_map = tensor_map_14.iterator
                                activation_sf_map = tensor_map_15.iterator
                                weight_map = tensor_map_16.iterator
                                weight_sf_map = tensor_map_17.iterator
                        weight_map = generic_tensormap_ptr(weight_map)
                        weight_sf_map = generic_tensormap_ptr(weight_sf_map)
                        k_block = Uint32(0)
                        while k_block < shape_k // Uint32(128):
                            # The weight loader ignores the activation geometry.
                            load_mixed_k128_tma(
                                activation_map, activation_sf_map,
                                weight_map, weight_sf_map,
                                data_a + data_stage * (self.block_m // 2 * 128),
                                data_sfa + data_stage * self.sf_block_m,
                                data_b + data_stage * (128 * 128),
                                data_sfb + data_stage * 128,
                                data_full + data_stage, data_empty + data_stage,
                                k_block, data_phase, block_idx,
                                n_cluster * Uint32(2) + cta_rank,
                                local_expert, valid_m, Uint32(0), shape_k,
                                is_shared, cta_rank, 1,
                                self.block_m, self.sf_block_m,
                                self.shared_weight_gran_k,
                                self.shared_experts > 0,
                                self.shared_weight_cache_hint,
                                self.routed_weight_linger)
                            data_stage = Uint32(0) if data_stage == \
                                Uint32(self.math_stages - 1) else data_stage + Uint32(1)
                            data_phase ^= Uint32(data_stage == Uint32(0))
                            k_block += Uint32(1)
                    task_iter += Uint32(1)
            if cta_rank == 0 and physical_warp == 2:
                task_words = cute.make_tensor(task, cute.make_layout((16,)))
                # Create the two swapped K-major data descriptor bases
                # once for the persistent leader MMA warp.
                weight_desc_base = kmajor_smem_desc(
                    data_b, Uint32(1024), True, Uint32(0))
                activation_desc_base = kmajor_smem_desc(
                    data_a, Uint32(1024), True, Uint32(0))
                data_desc_hi = (weight_desc_base >> Uint64(32)) << Uint64(32)
                sf_desc_hi = kmajor_smem_desc_from_start(
                    Uint32(0), Uint32(128), False)
                task_iter = Uint32(0)
                data_stage = Uint32(0)
                data_phase = Uint32(0)
                running = True
                while running:
                    task_stage = task_iter % Uint32(2)
                    cute.arch.mbarrier_wait(full + task_stage,
                                            (task_iter // Uint32(2)) % Uint32(2))
                    task_phase = task_words[task_stage * 8]
                    running = task_phase != Uint32(GCBlockPhase.None_)
                    if running:
                        valid_m = Uint32(task_words[task_stage * 8 + 5])
                        shape_k = Uint32(task_words[task_stage * 8 + 7])
                        is_shared = Uint32(0)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_shared = Uint32(1) if task_phase >= \
                                Uint32(GCBlockPhase.SharedLinear1) else Uint32(0)
                        num_k_blocks = shape_k // Uint32(128)
                        weight_format = Uint32(5)
                        if cutlass.const_expr(self.shared_experts > 0):
                            weight_format = select_weight_format(is_shared)
                        aligned_n = (valid_m + Uint32(15)) & Uint32(~15)
                        instr_desc_base = blockscaled_instr_desc_hi(
                            aligned_n, weight_format, Uint32(0), Uint32(0))
                        accum_stage = task_iter % Uint32(self.accum_stages)
                        accum_phase = (task_iter // Uint32(self.accum_stages)) % Uint32(2)
                        cute.arch.mbarrier_wait(accum_empty + accum_stage,
                                                accum_phase ^ Uint32(1))
                        tcgen05_after_thread_sync()
                        # Unroll K128 MMA stages by the configured group size.
                        for k_iter in cutlass.range(
                                Int32(0), Int32(num_k_blocks),
                                Int32(1), unroll=4):
                            k_block = Uint32(k_iter)
                            cute.arch.mbarrier_wait(data_full + data_stage,
                                                    data_phase)
                            # Keep descriptor layout bits fixed and
                            # advance only the 16-byte SMEM address field.
                            weight_desc_start = Uint32(
                                Uint32(weight_desc_base) + data_stage * Uint32(1024))
                            activation_desc_start = Uint32(
                                Uint32(activation_desc_base) + data_stage *
                                Uint32((self.block_m // 2 * 128) // 16))
                            issue_mixed_k128(
                                weight_desc_start, activation_desc_start,
                                data_sfb, data_sfa, data_stage,
                                data_desc_hi, sf_desc_hi,
                                accum_stage * Uint32(self.block_m),
                                Uint32(self.accum_stages * self.block_m +
                                       self.sfa_tmem_cols),
                                Uint32(self.accum_stages * self.block_m),
                                data_empty + data_stage, accum_full + accum_stage,
                                instr_desc_base, k_block,
                                Uint32(k_block == num_k_blocks - Uint32(1)),
                                is_shared, self.sf_block_m,
                                self.shared_weight_gran_k,
                                self.shared_experts > 0,
                                self.use_small_tile_path)
                            data_stage = Uint32(0) if data_stage == \
                                Uint32(self.math_stages - 1) else data_stage + Uint32(1)
                            data_phase ^= Uint32(data_stage == Uint32(0))
                    task_iter += Uint32(1)
                # The terminal None task increments task_iter once. Drain
                # the final valid accumulator stage before leaving the
                # producer warp, as the reference does after its loop.
                if task_iter > Uint32(1):
                    last_valid_iter = task_iter - Uint32(2)
                    last_stage = last_valid_iter % Uint32(self.accum_stages)
                    last_phase = (last_valid_iter // Uint32(self.accum_stages)) % Uint32(2)
                    cute.arch.mbarrier_wait(accum_empty + last_stage,
                                            last_phase)
            if physical_warp >= 4:
                if tptr[0] != Uint32(0):
                    device_trap()
                task_words = cute.make_tensor(task, cute.make_layout((16,)))
                iteration = Uint32(0)
                running = True
                while running:
                    stage = iteration % 2
                    generation_phase = (iteration // 2) % 2
                    cute.arch.mbarrier_wait(full + stage, generation_phase)
                    task_phase = task_words[stage * 8]
                    n_cluster = Uint32(task_words[stage * 8 + 3])
                    pool_block = Uint32(task_words[stage * 8 + 4])
                    valid_m = Uint32(task_words[stage * 8 + 5])
                    if task_phase != Uint32(GCBlockPhase.None_):
                        accum_stage = iteration % Uint32(self.accum_stages)
                        accum_phase = (iteration // Uint32(self.accum_stages)) % Uint32(2)
                        cute.arch.mbarrier_wait(accum_full + accum_stage,
                                                accum_phase)
                        tcgen05_after_thread_sync()
                        release_task_info(empty, Int32(stage))
                        # Match the reference's lane-0 exchange so valid-M
                        # controls the epilogue uniformly within each warp.
                        valid_m = cute.arch.shuffle_sync(valid_m, 0)
                        if physical_warp == 4 and \
                                task_phase == Uint32(GCBlockPhase.Linear2):
                            if elect_one_sync():
                                # The final MMA commit completed every K128
                                # input read. The BF16 peer scatter below uses
                                # TMEM, so L1 may now reuse the input ring.
                                ring_block = pool_block % Uint32(self.ring_blocks)
                                red_gpu_add_u32(
                                    signals.iterator +
                                    SIGNAL_OFFSETS['l2_empty_count'] // 4 + ring_block,
                                    Uint32(1))
                        if task_phase == Uint32(GCBlockPhase.Linear2):
                            # The reference joins every epilogue warp after
                            # the elected routed L2 ring release.
                            cute.arch.sync_warp()
                        is_l1 = task_phase == Uint32(GCBlockPhase.Linear1)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_l1 = is_l1 or task_phase == Uint32(GCBlockPhase.SharedLinear1)
                        if is_l1:
                            # W1 epilogue: the 16dp register mapping below
                            # predicates stores with this task's valid rows.
                            is_shared_l1 = Uint32(0)
                            if cutlass.const_expr(self.shared_experts > 0):
                                is_shared_l1 = Uint32(1) if task_phase == \
                                    Uint32(GCBlockPhase.SharedLinear1) else Uint32(0)
                            ring_block = pool_block % Uint32(self.ring_blocks)
                            generation = pool_block // Uint32(self.ring_blocks)
                            if not is_shared_l1:
                                l2_empty_ptr = signals.iterator + \
                                    SIGNAL_OFFSETS['l2_empty_count'] // 4 + ring_block
                                while ld_acquire_gpu_u32(l2_empty_ptr) != \
                                        generation * Uint32(self.l2_clusters * 2):
                                    pass
                            output_block = ring_block
                            if is_shared_l1:
                                output_block = pool_block
                            route_weights = cute.recast_ptr(
                                signals.iterator + self.l1_weight_words,
                                dtype=Float32)
                            epi_warp = Uint32(physical_warp - 4)
                            warpgroup = epi_warp // Uint32(4)
                            warp_in_wg = epi_warp % Uint32(4)
                            wg_start_m = warpgroup * Uint32(self.block_m // 2)
                            valid_wg_rows = Uint32(0)
                            if valid_m > wg_start_m:
                                valid_wg_rows = valid_m - wg_start_m
                                if valid_wg_rows > Uint32(self.block_m // 2):
                                    valid_wg_rows = Uint32(self.block_m // 2)
                            valid_stores = (valid_wg_rows + Uint32(self.store_m_l1 - 1)) // \
                                Uint32(self.store_m_l1)
                            n_block = n_cluster * Uint32(2) + cta_rank
                            cd_wg = cd_smem + warpgroup * Uint32(
                                2 * self.store_m_l1 * 64)
                            raw_values = cute.make_rmem_tensor(
                                (2 if self.block_m == 240 else 1,
                                 self.store_m_l1 // 8, 8), Float32)
                            stored_cached_weight = Float32(1)
                            clamp_value = activation_clamp
                            if cutlass.const_expr(activation_clamp is None):
                                # Preserve the old direct diagnostic call
                                # while production supplies the FP32 ABI arg.
                                clamp_value = Float32(self.activation_clamp)
                            if valid_stores == Uint32(0):
                                _release_w1_tmem(accum_empty, accum_stage)
                            if cutlass.const_expr(self.block_m == 240):
                                if valid_stores > Uint32(0):
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l1 // 8):
                                        base = accum_stage * Uint32(self.block_m) + \
                                            wg_start_m + Uint32(atom * 8)
                                        _load_w1_atom(raw_values, 0, atom, base)
                                    if valid_stores == Uint32(1):
                                        cute.arch.fence_view_async_tmem_load()
                                        _release_w1_tmem(accum_empty, accum_stage)
                            for store in cutlass.range_constexpr(
                                    (self.block_m // 2) // self.store_m_l1):
                                if Uint32(store) < valid_stores:
                                    if cutlass.const_expr(self.block_m != 240):
                                        for atom in cutlass.range_constexpr(
                                                self.store_m_l1 // 8):
                                            base = accum_stage * Uint32(self.block_m) + \
                                                wg_start_m + Uint32(store * self.store_m_l1 + atom * 8)
                                            _load_w1_atom(raw_values, 0, atom, base)
                                            if atom == self.store_m_l1 // 8 - 1 and \
                                                    Uint32(store + 1) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w1_tmem(accum_empty, accum_stage)
                                    # BM240 overlaps the next register
                                    # block's loads with this block's math.
                                    cute.arch.fence_view_async_tmem_load()
                                    if cutlass.const_expr(self.block_m == 240):
                                        if Uint32(store + 1) < valid_stores:
                                            for atom in cutlass.range_constexpr(
                                                    self.store_m_l1 // 8):
                                                next_base = accum_stage * Uint32(self.block_m) + \
                                                    wg_start_m + Uint32(
                                                        (store + 1) * self.store_m_l1 + atom * 8)
                                                _load_w1_atom(
                                                    raw_values, (store + 1) % 2,
                                                    atom, next_base)
                                            if Uint32(store + 2) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w1_tmem(accum_empty, accum_stage)
                                    raw_buffer = store % 2 if self.block_m == 240 else 0
                                    activation_values = cute.make_rmem_tensor(
                                        (self.store_m_l1 // 8, 4), Float32)
                                    reduced_amax = cute.make_rmem_tensor(
                                        (self.store_m_l1 // 8, 2), Float32)
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l1 // 8):
                                        j = store * (self.store_m_l1 // 8) + atom
                                        if not is_shared_l1 and j % 4 == 0:
                                            if self.block_m // 2 % 32 == 0 or \
                                                    Uint32(j * 8 + lane) < \
                                                    Uint32(self.block_m // 2):
                                                weight_token = ring_block * \
                                                    Uint32(self.block_m) + wg_start_m + \
                                                    Uint32(j * 8 + lane)
                                                stored_cached_weight = \
                                                    route_weights[weight_token]
                                        source_lane = Uint32((j * 8) % 32) + \
                                            (Uint32(lane) % Uint32(4)) * Uint32(2)
                                        weight0 = cute.arch.shuffle_sync(
                                            stored_cached_weight, source_lane)
                                        weight1 = cute.arch.shuffle_sync(
                                            stored_cached_weight, source_lane + Uint32(1))
                                        value0, value1 = _swiglu_pair(
                                            raw_values[raw_buffer, atom, 0],
                                            raw_values[raw_buffer, atom, 1],
                                            raw_values[raw_buffer, atom, 2],
                                            raw_values[raw_buffer, atom, 3],
                                            weight0, weight1, clamp_value,
                                            Float32(1), self.has_activation_clamp,
                                            False, self.fast_math)
                                        value2, value3 = _swiglu_pair(
                                            raw_values[raw_buffer, atom, 4],
                                            raw_values[raw_buffer, atom, 5],
                                            raw_values[raw_buffer, atom, 6],
                                            raw_values[raw_buffer, atom, 7],
                                            weight0, weight1, clamp_value,
                                            Float32(1), self.has_activation_clamp,
                                            False, self.fast_math)
                                        activation_values[atom, 0] = value0
                                        activation_values[atom, 1] = value1
                                        activation_values[atom, 2] = value2
                                        activation_values[atom, 3] = value3
                                        amax0 = warp_group4_max(
                                            fmax_f32(
                                                fmax_f32(Float32(0), abs_f32(value0)),
                                                abs_f32(value2)))
                                        amax1 = warp_group4_max(
                                            fmax_f32(
                                                fmax_f32(Float32(0), abs_f32(value1)),
                                                abs_f32(value3)))
                                        reduced_amax[atom, 0] = amax0
                                        reduced_amax[atom, 1] = amax1
                                        if lane < 4:
                                            amax_offset = epi_warp * \
                                                Uint32(self.store_m_l1) + \
                                                Uint32(atom * 8 + lane * 2)
                                            epi_amax[amax_offset] = amax0
                                            epi_amax[amax_offset + 1] = amax1
                                        cute.arch.sync_warp()
                                    tma_store_wait_1()
                                    sync_epilogue_warpgroup(warpgroup)
                                    tma_stage = store % 2
                                    cd_stage = cd_wg + Uint32(
                                        tma_stage * self.store_m_l1 * 64)
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l1 // 8):
                                        partner_offset = (epi_warp ^ Uint32(1)) * \
                                            Uint32(self.store_m_l1) + \
                                            Uint32(atom * 8) + \
                                            (Uint32(lane) % Uint32(4)) * Uint32(2)
                                        sf_amax0 = fmax_f32(
                                            reduced_amax[atom, 0],
                                            epi_amax[partner_offset])
                                        sf_amax1 = fmax_f32(
                                            reduced_amax[atom, 1],
                                            epi_amax[partner_offset + 1])
                                        sf_exp0 = ue8m0_fp8_scale_exp(sf_amax0)
                                        sf_exp1 = ue8m0_fp8_scale_exp(sf_amax1)
                                        inv0 = ue8m0_fp8_inv(sf_exp0)
                                        inv1 = ue8m0_fp8_inv(sf_exp1)
                                        upper0, upper1 = cute.arch.mul_packed_f32x2(
                                            (activation_values[atom, 0],
                                             activation_values[atom, 1]), (inv0, inv1), rnd='rn', ftz=False)
                                        lower0, lower1 = cute.arch.mul_packed_f32x2(
                                            (activation_values[atom, 2],
                                             activation_values[atom, 3]), (inv0, inv1), rnd='rn', ftz=False)
                                        packed = _w1_pack_fp8x4(
                                            upper0, upper1, lower0, lower1)
                                        row = Uint32(lane)
                                        smem_offset = Uint32(atom * 8 * 64) + \
                                            row * Uint32(64) + \
                                            (warp_in_wg ^ (row // Uint32(2))) * Uint32(16)
                                        stmatrix_fp8x4_trans_late(
                                            cd_stage, smem_offset, packed)
                                        if warp_in_wg % Uint32(2) == Uint32(0) and lane < 4:
                                            output_sf_words = Uint64(self.l2_sf_words)
                                            sf_stride = Uint32(self.sf_ring_tokens)
                                            if is_shared_l1:
                                                output_sf_words = Uint64(
                                                    self.shared_l2_sf_words)
                                                sf_stride = Uint32(
                                                    self.shared_sf_tokens)
                                            l2_sf_bytes = cute.recast_ptr(
                                                signals.iterator, dtype=Uint8) + \
                                                output_sf_words * Uint64(4)
                                            k_sf = n_block * Uint32(2) + \
                                                warp_in_wg // Uint32(2)
                                            token_base = wg_start_m + Uint32(
                                                store * self.store_m_l1 + atom * 8)
                                            assume_u32_lt(
                                                token_base, Uint32(self.block_m))
                                            sf_token = output_block * Uint32(self.sf_block_m) + \
                                                (token_base & Uint32(~127)) + \
                                                (token_base & Uint32(31)) * Uint32(4) + \
                                                ((token_base >> Uint32(5)) & Uint32(3)) + \
                                                Uint32(lane * 8)
                                            if cutlass.const_expr(
                                                    self.wide_routed_w1_sf or
                                                    self.wide_shared_w1_sf):
                                                sf_addr = Uint64(k_sf // Uint32(4)) * \
                                                    (Uint64(sf_stride) * Uint64(4)) + \
                                                    Uint64(sf_token) * Uint64(4) + \
                                                    Uint64(k_sf % Uint32(4))
                                                l2_sf_bytes[sf_addr] = Uint8(sf_exp0)
                                                l2_sf_bytes[sf_addr + Uint64(16)] = \
                                                    Uint8(sf_exp1)
                                            else:
                                                sf_addr = (k_sf // Uint32(4)) * \
                                                    (sf_stride * Uint32(4)) + \
                                                    sf_token * Uint32(4) + k_sf % Uint32(4)
                                                l2_sf_bytes[sf_addr] = Uint8(sf_exp0)
                                                l2_sf_bytes[sf_addr + Uint32(16)] = \
                                                    Uint8(sf_exp1)
                                        cute.arch.sync_warp()
                                    sync_epilogue_warpgroup(warpgroup)
                                    if warp_in_wg == Uint32(0):
                                        with cute.arch.elect_one():
                                            output_map = tensor_map_4.iterator
                                            if is_shared_l1:
                                                output_map = tensor_map_13.iterator
                                            tma_store_fence()
                                            tile_s2g_2d(
                                                output_map, cd_stage,
                                                n_block * Uint32(64),
                                                output_block * Uint32(self.block_m) +
                                                wg_start_m + Uint32(store * self.store_m_l1))
                                            tma_store_arrive()
                                    cute.arch.sync_warp()
                            tma_store_wait_0()
                        else:
                            # Source-contract W2 vector epilogue. The
                            # 16dp register-to-value bijection still
                            # requires validM>8 device verification.
                            is_shared_l2 = Uint32(0)
                            if cutlass.const_expr(self.shared_experts > 0):
                                is_shared_l2 = Uint32(
                                    task_phase == Uint32(GCBlockPhase.SharedLinear2))
                            epi_warp = Uint32(physical_warp - 4)
                            warpgroup = epi_warp // Uint32(4)
                            warp_in_wg = epi_warp % Uint32(4)
                            wg_start_m = warpgroup * Uint32(self.block_m // 2)
                            valid_wg_rows = Uint32(0)
                            if valid_m > wg_start_m:
                                valid_wg_rows = valid_m - wg_start_m
                                if valid_wg_rows > Uint32(self.block_m // 2):
                                    valid_wg_rows = Uint32(self.block_m // 2)
                            valid_stores = (valid_wg_rows + Uint32(self.store_m_l2 - 1)) // \
                                Uint32(self.store_m_l2)
                            n_block = n_cluster * Uint32(2) + cta_rank
                            cd_wg = cd_smem + warpgroup * Uint32(
                                self.store_m_l2 * 128 * 2)
                            raw_values = cute.make_rmem_tensor(
                                (2 if self.block_m == 240 else 1,
                                 self.store_m_l2 // 8, 8), Float32)
                            if valid_stores == Uint32(0):
                                _release_w2_tmem(accum_empty, accum_stage)
                            if cutlass.const_expr(self.block_m == 240):
                                if valid_stores > Uint32(0):
                                    base = accum_stage * Uint32(self.block_m) + \
                                        wg_start_m
                                    _load_w2_atom(raw_values, 0, 0, base)
                                    if valid_stores == Uint32(1):
                                        cute.arch.fence_view_async_tmem_load()
                                        _release_w2_tmem(accum_empty, accum_stage)
                            for store in cutlass.range_constexpr(
                                    (self.block_m // 2) // self.store_m_l2):
                                if Uint32(store) < valid_stores:
                                    if cutlass.const_expr(self.block_m != 240):
                                        for atom in cutlass.range_constexpr(
                                                self.store_m_l2 // 8):
                                            base = accum_stage * Uint32(self.block_m) + \
                                                wg_start_m + Uint32(store * self.store_m_l2 + atom * 8)
                                            _load_w2_atom(raw_values, 0, atom, base)
                                            if atom == self.store_m_l2 // 8 - 1 and \
                                                    Uint32(store + 1) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w2_tmem(accum_empty, accum_stage)
                                    # Wait for the current atom before
                                    # issuing the BM240 next-store prefetch.
                                    cute.arch.fence_view_async_tmem_load()
                                    if cutlass.const_expr(self.block_m == 240):
                                        if Uint32(store + 1) < valid_stores:
                                            next_base = accum_stage * Uint32(self.block_m) + \
                                                wg_start_m + Uint32((store + 1) * 8)
                                            _load_w2_atom(
                                                raw_values, (store + 1) % 2,
                                                0, next_base)
                                            if Uint32(store + 2) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w2_tmem(accum_empty, accum_stage)
                                    # One metadata record for each lane's
                                    # row in every 8-row atom.
                                    src_metadata = ()
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l2 // 8):
                                        token_in_block = wg_start_m + \
                                            Uint32(store * self.store_m_l2 + atom * 8) + \
                                            warp_in_wg * Uint32(2) + Uint32(lane // 16)
                                        pool_token = pool_block * Uint32(self.block_m) + \
                                            token_in_block
                                        record = _load_w2_source_metadata(
                                            signals.iterator + self.metadata_words,
                                            token_in_block, valid_m, pool_token,
                                            rank_idx, is_shared_l2, self.topk,
                                            self.shared_experts > 0)
                                        src_metadata += (record,)
                                    cute.arch.sync_warp()
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l2 // 8):
                                        if atom == 0 and store > 0:
                                            sync_epilogue_warpgroup(warpgroup)
                                        row = Uint32(lane % 8)
                                        col = (epi_warp % Uint32(2)) * Uint32(4) + \
                                            Uint32(lane // 8)
                                        smem_store_offset = (warp_in_wg // Uint32(2)) * \
                                            Uint32(self.store_m_l2 * 128) + \
                                            Uint32(atom * 8 * 128) + row * Uint32(128) + \
                                            (col ^ row) * Uint32(16)
                                        buffer = store % 2 if self.block_m == 240 else 0
                                        stmatrix_bf16x8_trans(
                                            cd_wg + smem_store_offset,
                                            pack_bf16_pair(raw_values[buffer, atom, 0],
                                                            raw_values[buffer, atom, 1]),
                                            pack_bf16_pair(raw_values[buffer, atom, 2],
                                                            raw_values[buffer, atom, 3]),
                                            pack_bf16_pair(raw_values[buffer, atom, 4],
                                                            raw_values[buffer, atom, 5]),
                                            pack_bf16_pair(raw_values[buffer, atom, 6],
                                                            raw_values[buffer, atom, 7]))
                                    sync_epilogue_warpgroup(warpgroup)
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l2 // 8):
                                        row_in_atom = warp_in_wg * Uint32(2) + \
                                            Uint32(lane // 16)
                                        row_in_store = Uint32(atom * 8) + row_in_atom
                                        token_in_block = wg_start_m + \
                                            Uint32(store * self.store_m_l2) + row_in_store
                                        if token_in_block < valid_m:
                                            bank_group = Uint32(lane % 8)
                                            half = (Uint32(lane) % Uint32(16)) // Uint32(8)
                                            smem_load_offset = half * \
                                                Uint32(self.store_m_l2 * 128) + \
                                                row_in_store * Uint32(128) + \
                                                (bank_group ^ row_in_atom) * Uint32(16)
                                            words = ld_shared_u32x4(
                                                cd_wg + smem_load_offset)
                                            token_bytes = Uint64(self.hidden * 2)
                                            slot_bytes = Uint64(self.capacity) * token_bytes
                                            channel_bytes = n_block * Uint32(256) + \
                                                (Uint32(lane) % Uint32(16)) * Uint32(16)
                                            dst_bytes = (Uint64(src_metadata[atom][2]) * slot_bytes +
                                                         Uint64(src_metadata[atom][1]) * token_bytes +
                                                         Uint64(channel_bytes))
                                            dst = cute.make_ptr(
                                                Uint32, mapped_peer_base(
                                                    peer_ptrs, src_metadata[atom][0],
                                                    self.num_ranks) +
                                                Uint64(self.combine_words * 4) + \
                                                dst_bytes,
                                                cute.AddressSpace.gmem,
                                                assumed_align=16)
                                            st_global_u32x4(
                                                dst, words[0], words[1],
                                                words[2], words[3])
                        sync_aligned(256, 2)
                    is_w1_release = task_phase == Uint32(GCBlockPhase.Linear1)
                    if cutlass.const_expr(self.shared_experts > 0):
                        is_w1_release = is_w1_release or \
                            task_phase == Uint32(GCBlockPhase.SharedLinear1)
                    if is_w1_release:
                        if physical_warp == 4:
                            if elect_one_sync():
                                ring_block = pool_block % self.ring_blocks
                                n_block = n_cluster * 2 + cta_rank
                                if task_phase == Uint32(GCBlockPhase.Linear1):
                                    red_release_gpu_xor_u64(
                                        signals.iterator + SIGNAL_OFFSETS['l2_full_mask'] // 4 +
                                        ring_block * 2,
                                        Uint64(1) << n_block)
                                    red_gpu_add_u32(
                                        signals.iterator + SIGNAL_OFFSETS['l1_empty_count'] // 4 +
                                        ring_block, Uint32(1))
                                else:
                                    if cutlass.const_expr(self.shared_experts > 0):
                                        if task_phase == Uint32(GCBlockPhase.SharedLinear1):
                                            red_release_gpu_add_u32(
                                                signals.iterator +
                                                SIGNAL_OFFSETS['shared_l2_full_count'] // 4 +
                                                pool_block, Uint32(1))
                        cute.arch.sync_warp()
                    running = task_phase != Uint32(GCBlockPhase.None_)
                    iteration = Uint32(iteration + Uint32(1))
                # The same full epilogue warp in both CTAs retires TMEM.
                if physical_warp == 4:
                    tmem_free_2cta(tptr[0], Uint32(self.tmem_cols))
        else:
            # Both graph workers precede role 3. Fused finalization uses
            # warp-owned barriers; legacy paths retain the whole-CTA join.
            combine_smem = cute.recast_ptr(signals.iterator, dtype=Uint8)
            combine_bars = cute.recast_ptr(signals.iterator, dtype=Uint64)
            if cutlass.const_expr(self.finalize_output):
                # One fetch plane per combine slot; the output plane aliases
                # slot 0. Both live inside the role3 reusable prefix.
                combine_alloc = cutlass.utils.SmemAllocator()
                combine_smem = cute.recast_ptr(
                    combine_alloc.allocate(self.shared_layout.field_end, 1024), dtype=Uint8)
                combine_bars = cute.recast_ptr(
                    combine_smem + self.shared_layout.combine_bars, dtype=Uint64)
                if cutlass.const_expr(self.shared_experts > 0):
                    cute.arch.griddepcontrol_wait()
                else:
                    self._initialize_shared_storage(
                        combine_smem, Uint32(physical_warp), Uint32(lane))
            # Role 1's epoch remains immutable throughout this role-3 launch.
            # Each warp caches it independently; the footer advances recv_parity.
            ready_epoch = Uint64(ld_volatile_u32(
                signals.iterator + SIGNAL_OFFSETS['role3_epoch'] // 4))
            if cutlass.const_expr(not (self.finalize_output and self.shared_experts > 0)):
                sync_unaligned(FINALIZE_THREADS, 1)
            recv_off = SIGNAL_OFFSETS['expert_recv_count_sum'] // 4
            recv_ptr = signals.iterator + recv_off
            if physical_warp < 4:
                self._role3_dispatch(signals, peer_ptrs, rank_idx,
                                     Uint32(tid), Uint32(block),
                                     recv_ptr, stats, ready_epoch)
            else:
                self._role3_combine(signals, rank_idx, num_tokens, y_out,
                                    block, physical_warp, lane,
                                    combine_smem, combine_bars, ready_epoch)

    @cute.jit
    def __call__(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                 rank_idx: Uint32, num_tokens: Uint32,
                 tensor_maps=None, y_out=None, stats=None,
                 activation_clamp=None, activation_alpha=None,
                 tensor_map_1=None, tensor_map_2=None, tensor_map_3=None,
                 tensor_map_4=None, tensor_map_5=None, tensor_map_6=None,
                 tensor_map_7=None, tensor_map_8=None, tensor_map_9=None,
                 tensor_map_10=None, tensor_map_11=None, tensor_map_12=None,
                 tensor_map_13=None, tensor_map_14=None, tensor_map_15=None,
                 tensor_map_16=None, tensor_map_17=None,
                 tensor_map_18=None, tensor_map_19=None,
                 tensor_map_20=None, tensor_map_21=None):
        if cutlass.const_expr(self.role == 1):
            if cutlass.const_expr(activation_clamp is None and activation_alpha is None):
                self.kernel(signals, peer_ptrs, rank_idx, num_tokens).launch(
                    grid=(self.dispatch_sms, 1, 1),
                    block=(self.dispatch_warps * 32, 1, 1),
                    max_number_threads=(self.dispatch_warps * 32, 1, 1),
                    min_blocks_per_mp=1)
            else:
                self.kernel(signals, peer_ptrs, rank_idx, num_tokens,
                            None, None, None,
                            activation_clamp, activation_alpha).launch(
                    grid=(self.dispatch_sms, 1, 1),
                    block=(self.dispatch_warps * 32, 1, 1),
                    max_number_threads=(self.dispatch_warps * 32, 1, 1),
                    min_blocks_per_mp=1)
        elif cutlass.const_expr(self.role == 2):
            if cutlass.const_expr(activation_clamp is None and activation_alpha is None):
                self.kernel(signals, peer_ptrs, rank_idx, num_tokens,
                            tensor_maps, None, None, None, None,
                            tensor_map_1, tensor_map_2, tensor_map_3,
                            tensor_map_4, tensor_map_5, tensor_map_6,
                            tensor_map_7, tensor_map_8, tensor_map_9,
                            tensor_map_10, tensor_map_11, tensor_map_12,
                            tensor_map_13, tensor_map_14, tensor_map_15,
                            tensor_map_16, tensor_map_17,
                            tensor_map_18, tensor_map_19,
                            tensor_map_20, tensor_map_21).launch(
                    grid=(self.compute_sms, 1, 1), block=(384, 1, 1),
                    max_number_threads=(384, 1, 1),
                    cluster=(2, 1, 1), min_blocks_per_mp=1)
            else:
                self.kernel(signals, peer_ptrs, rank_idx, num_tokens, tensor_maps,
                            None, None,
                            activation_clamp, activation_alpha,
                            tensor_map_1, tensor_map_2, tensor_map_3,
                            tensor_map_4, tensor_map_5, tensor_map_6,
                            tensor_map_7, tensor_map_8, tensor_map_9,
                            tensor_map_10, tensor_map_11, tensor_map_12,
                            tensor_map_13, tensor_map_14, tensor_map_15,
                            tensor_map_16, tensor_map_17,
                            tensor_map_18, tensor_map_19,
                            tensor_map_20, tensor_map_21).launch(
                    grid=(self.compute_sms, 1, 1), block=(384, 1, 1),
                    max_number_threads=(384, 1, 1),
                    cluster=(2, 1, 1), min_blocks_per_mp=1)
        else:
            if cutlass.const_expr(self.finalize_output):
                if cutlass.const_expr(activation_clamp is None and activation_alpha is None):
                    self.kernel(signals, peer_ptrs, rank_idx, num_tokens, tensor_maps,
                                y_out, stats).launch(
                        grid=(self.finalize_sms, 1, 1),
                        block=(FINALIZE_THREADS, 1, 1),
                        max_number_threads=(FINALIZE_THREADS, 1, 1), min_blocks_per_mp=1)
                else:
                    self.kernel(signals, peer_ptrs, rank_idx, num_tokens, tensor_maps,
                                y_out, stats,
                                activation_clamp, activation_alpha).launch(
                        grid=(self.finalize_sms, 1, 1),
                        block=(FINALIZE_THREADS, 1, 1),
                        max_number_threads=(FINALIZE_THREADS, 1, 1), min_blocks_per_mp=1)
            else:
                if cutlass.const_expr(activation_clamp is None and activation_alpha is None):
                    self.kernel(signals, peer_ptrs, rank_idx, num_tokens).launch(
                        grid=(self.finalize_sms, 1, 1),
                        block=(FINALIZE_THREADS, 1, 1),
                        max_number_threads=(FINALIZE_THREADS, 1, 1), min_blocks_per_mp=1)
                else:
                    self.kernel(signals, peer_ptrs, rank_idx, num_tokens,
                                None, None, None,
                                activation_clamp, activation_alpha).launch(
                        grid=(self.finalize_sms, 1, 1),
                        block=(FINALIZE_THREADS, 1, 1),
                        max_number_threads=(FINALIZE_THREADS, 1, 1), min_blocks_per_mp=1)
