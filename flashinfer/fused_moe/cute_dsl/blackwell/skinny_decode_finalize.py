"""Native W4A8 token-N8 decode GEMM2/finalize. Minimum SM103; dispatch targets B300.

Private SM103 decode dispatch preserves token-major intermediate MXFP8/SF,
caller routing/output/workspace, BF16 contribution rounding and bulk scatter.
Six operand/SF rings reuse the qualified skinny gather schedule.

Tail stream-K (selected T4/T16 walks only): every full round of the static
persistent walk is unchanged; the items of the last partial round are cut along
K into K512-epoch pieces spread over the otherwise idle CTAs. A piece runs the
unchanged mainloop on its K sub-range with a fresh accumulator and performs the
unchanged linear-scale BF16 bulk reduce-add into the cleared output, so partial
sums add without a partial workspace, counter or fixup. Rows of a split item
receive up to six BF16 reduce-adds instead of one. Every decision derives from
the device routing arrays the scheduler already reads (Graph-replay and
concurrent-stream safe). Single-round launches are not split. Each CTA emits
its first walk position (block_id, never at or beyond the cut) before the
valid-tile scan, so the scan overlaps the consumers' first item.
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05.helpers import smem_descriptor_to_int
from .utils import (
    UnalignedNamedBarrier,
    blk_reduce_bf16,
    tcgen05_fence_after_thread_sync,
    tcgen05_fence_before_thread_sync,
)


@dsl_user_op
def finalize_instruction_selector(
    mma, ta0, tb0, sk: cutlass.Constexpr, *, loc=None, ip=None
):
    """Encode actual typed slot-zero origins; no SF data load or TMEM access."""
    op = mma.op
    # Named fields of UtcmmaDescriptorBlock: dense, K32, K-major and the
    # unused/unique-SFA bits are zero. No packed magic descriptor literal.
    a_format_e2m1, b_format_e4m3, scale_format_ue8m0 = 5, 0, 1
    a_format_shift, b_format_shift = 7, 10
    negate_a_shift, negate_b_shift = 13, 14
    n_dim_shift, scale_format_shift, m_dim_shift = 17, 23, 27
    base = cutlass.Uint32(
        (a_format_e2m1 << a_format_shift)
        | (b_format_e4m3 << b_format_shift)
        | ((op.shape_mnk[1] >> 3) << n_dim_shift)
        | (scale_format_ue8m0 << scale_format_shift)
        | ((op.shape_mnk[0] >> 7) << m_dim_shift)
    )
    base = base | (
        cutlass.Uint32(cutlass.Boolean(mma.get(tcgen05.Field.NEGATE_A, loc=loc, ip=ip)))
        << negate_a_shift
    )
    base = base | (
        cutlass.Uint32(cutlass.Boolean(mma.get(tcgen05.Field.NEGATE_B, loc=loc, ip=ip)))
        << negate_b_shift
    )
    assert sk in (0, 1, 2, 3)
    a = cutlass.Uint32(ta0[(None, None, 0)].iterator.toint(loc=loc, ip=ip))
    b = cutlass.Uint32(tb0[(None, None, 0)].iterator.toint(loc=loc, ip=ip))
    a = a + cutlass.Uint32(sk << 30)
    b = b + cutlass.Uint32(sk << 30)
    return (
        base
        | ((a >> 1) & cutlass.Uint32(0x60000000))
        | ((b >> 26) & cutlass.Uint32(0x30))
    )


@dsl_user_op
def issue_finalize_epoch(
    mma, acc, fa, fb, ta0, tb0, a_stage, b_stage, d0, d1, d2, d3, *, loc=None, ip=None
):
    """Sixteen ordered K32 MMAs for the exact existing M128/N8/K512 epoch.

    SmemDesc iterators come from the original CuTe fragments, not a new
    hand-written shared layout. Integer descriptor fields follow the primary
    UtcmmaDescriptorBlock/make_utcmma_desc_block factory, with its M>>7 rule.
    This helper intentionally supports only this source's exact atom/layout.
    """
    op = mma.op
    assert isinstance(op, tcgen05.MmaMXF8F6F4Op)
    assert tuple(op.shape_mnk) == (128, 8, 32)
    assert op.cta_group == tcgen05.CtaGroup.ONE
    assert op.a_src == tcgen05.OperandSource.SMEM
    assert op.a_dtype is cutlass.Float4E2M1FN
    assert op.b_dtype is cutlass.Float8E4M3FN
    assert op.acc_dtype is cutlass.Float32
    assert op.sf_dtype is cutlass.Float8E8M0FNU and op.sf_vec_size == 32
    assert op.a_major_mode.name == op.b_major_mode.name == "K"

    accumulate = cutlass.Uint32(
        cutlass.Boolean(mma.get(tcgen05.Field.ACCUMULATE, loc=loc, ip=ip))
    )
    # Actual CuTe-derived epoch origins include each independent ring slot.
    a = fa[(None, None, 0, a_stage)]
    b = fb[(None, None, 0, b_stage)]
    assert cute.size(a) == cute.size(b) == 1
    arguments = [
        cutlass.Uint32(acc.iterator.toint(loc=loc, ip=ip)),
        accumulate,
        d0,
        smem_descriptor_to_int(a.iterator, loc=loc, ip=ip),
        smem_descriptor_to_int(b.iterator, loc=loc, ip=ip),
        cutlass.Uint32(ta0[(None, None, 0)].iterator.toint(loc=loc, ip=ip)),
        cutlass.Uint32(tb0[(None, None, 0)].iterator.toint(loc=loc, ip=ip)),
        d1,
        d2,
        d3,
    ]
    constraints = ["r", "r", "r", "l", "l", "r", "r", "r", "r", "r"]

    # Fixed-layout offsets are proved against all sixteen original operands.
    # Low-word addition preserves the original high word: the largest start
    # field is 0x6bc6, below bit15 and the leading field starting at bit16.
    # Reuse temporaries only after their preceding MMA has sampled operands.
    asm = [
        "{ .reg .pred elected, accum_first, accum_next; ",
        ".reg .b32 sf_a, sf_b; ",
        ".reg .b64 a_desc, b_desc; ",
        ".reg .b32 a_lo0, a_hi, b_lo0, b_hi, a_lo, b_lo; ",
        "elect.sync _|elected, -1; @!elected bra EPOCH_DONE; ",
        "mov.b64 {a_lo0, a_hi}, $3; mov.b64 {b_lo0, b_hi}, $4; ",
        "setp.ne.u32 accum_first, $1, 0; setp.eq.u32 accum_next, 0, 0; ",
    ]
    for issue in range(16):
        kc, sk = issue // 4, issue % 4
        a_delta, b_delta = 1024 * kc + 2 * sk, 64 * kc + 2 * sk
        sfa_delta, sfb_delta = 4 * kc + (sk << 30), 2 * kc + (sk << 30)
        predicate = "accum_first" if issue == 0 else "accum_next"
        asm.append(
            f"add.u32 a_lo, a_lo0, {a_delta}; mov.b64 a_desc, {{a_lo, a_hi}}; "
            f"add.u32 b_lo, b_lo0, {b_delta}; mov.b64 b_desc, {{b_lo, b_hi}}; "
            f"add.u32 sf_a, $5, {sfa_delta}; add.u32 sf_b, $6, {sfb_delta}; "
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32."
            f"collector::a::discard [$0], a_desc, b_desc, ${2 if sk == 0 else 6 + sk}, "
            f"[sf_a], [sf_b], {predicate}; "
        )
    asm.append("EPOCH_DONE: }")
    llvm.inline_asm(
        None,
        [value.ir_value(loc=loc, ip=ip) for value in arguments],
        "".join(asm),
        ",".join(constraints + ["~{memory}"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class SkinnyDecodeFinalizeKernel:
    def __init__(
        self,
        *,
        enable_t16_const_scheduler=False,
        enable_t16_slot_planes=False,
        enable_t4_const_scheduler=False,
        enable_tail_stream_k=None,
    ):
        self.enable_skinny_finalize = True
        self.enable_contiguous_b_tma = True
        self.b_tma_bytes = 4096
        self.enable_intermediate_sfb_tma = True
        self.sfb_tma_bytes = 512
        self.enable_narrow_a = False
        self.enable_sparse_narrow_a = False
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tile = (128, 8, 512)
        self.scale_tile = (128, 8, 128)
        self.enable_t16_const_scheduler = enable_t16_const_scheduler
        self.enable_t16_slot_planes = enable_t16_slot_planes
        self.enable_t4_const_scheduler = enable_t4_const_scheduler
        # Tail stream-K follows the existing selected T4/T16 walks unless the
        # caller pins it; the generic (T8) walk keeps the five-word records.
        if enable_tail_stream_k is None:
            enable_tail_stream_k = bool(
                enable_t4_const_scheduler or enable_t16_slot_planes
            )
        self.enable_tail_stream_k = bool(enable_tail_stream_k)
        assert (
            not self.enable_tail_stream_k
            or enable_t4_const_scheduler
            or enable_t16_slot_planes
        )
        self.record_words = 7 if self.enable_tail_stream_k else 5
        self.work_words = 2 * self.record_words
        self.cache_key_marker = (
            "t4_t16_g2_tail_stream_k" if self.enable_tail_stream_k else None
        )
        self.num_ab_stage, self.num_acc_stage, self.num_c_stage = 3, 2, 1
        self.num_tmem_alloc_cols, self.threads_wo_sched = 128, 480
        self.threads_per_cta = 512
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=512)
        self.epilog_sync_barrier = UnalignedNamedBarrier(barrier_id=2, num_threads=128)
        self.tma_bytes = 32768
        self.num_tma_load_bytes = self.tma_bytes

    @cute.jit
    def scale_layout(self, rows, k, groups):
        # Exact selected finalize six-dimensional physical UE8M0 layout.
        return cute.make_ordered_layout(
            (32, 4, rows // 128, 4, k // 128, groups), order=(2, 1, 4, 0, 3, 5)
        )

    @cute.jit
    def epilogue_partition(self, acc_mn, tid):
        tc = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition.x8), cutlass.Float32
            ),
            acc_mn,
        )
        thr = tc.get_slice(tid)
        coords = thr.partition_D(cute.make_identity_tensor((128, 8)))
        return tc, thr.partition_S(acc_mn), coords

    @cute.jit
    def output_shared_layout(self):
        return cute.make_layout((8, 128), stride=(128, 1))

    @cute.jit
    def scale_copy(self, shared, tensor):
        shared_compact = cute.filter_zeros(shared)
        tensor_compact = cute.filter_zeros(tensor)
        atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        copy = tcgen05.make_s2t_copy(atom, tensor_compact)
        part = copy.get_slice(0)
        source = tcgen05.get_s2t_smem_desc_tensor(
            copy, part.partition_S(shared_compact)
        )
        return copy, source, part.partition_D(tensor_compact)

    @cute.jit
    def weight_scale_tma(self, pointer, features, k, experts):
        # Raw UE8M0 bytes, paired as Int16 without conversion. Four K128
        # slices form exactly the old contiguous 2048-byte warp12 transfer.
        source = cute.make_tensor(
            cute.recast_ptr(pointer, dtype=cutlass.Int16),
            cute.make_layout(
                (256, k // 128, features // 128, experts),
                stride=(1, 256, 256 * (k // 128), 256 * (k // 128) * (features // 128)),
            ),
        )
        destination = cute.make_layout((256, 4, 1, 1), stride=(1, 256, 1024, 1024))
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            source,
            destination,
            (256, 4, 1, 1),
        )

    @cute.jit
    def weight_scale_partition(self, atom, source, shared):
        destination = cute.make_tensor(
            cute.recast_ptr(shared.iterator, dtype=cutlass.Int16),
            cute.make_layout((256, 4, 1, 1, 3), stride=(1, 256, 1024, 1024, 1024)),
        )
        tiles = cute.local_tile(source, (256, 4, 1, 1), (None, None, None, None))
        return cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(destination, 0, 4),
            cute.group_modes(tiles, 0, 4),
        )

    @cute.jit
    def clamped_b_tma(self, pointer, k, swizzle):
        # Existing native intermediate bytes; 4D coordinates clamp only the
        # selected N8 sub-tile without reading another expert's padded rows.
        source = cute.make_tensor(
            pointer,
            cute.make_layout((k, 8, 1 << 31, 1 << 31), stride=(1, k, (1 << 35) - k, k)),
        )
        shared = cute.make_composed_layout(swizzle, 0, cute.make_layout((128, 8, 1, 1)))
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            source,
            shared,
            (128, 8, 1, 1),
        )

    @cute.jit
    def clamped_b_partition(self, atom, mapped, tma_shared, row_base, limit):
        valid = cutlass.min(
            cutlass.Int32(8), cutlass.max(cutlass.Int32(0), limit - row_base)
        )
        pad = 8 - valid
        source = cute.domain_offset(
            (0, pad, 1 << 30, row_base - pad + (1 << 30)), mapped
        )
        tiles = cute.local_tile(source, (128, 8, 1, 1), (None, 0, 0, 0))
        return cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(tma_shared, 0, 4),
            cute.group_modes(tiles, 0, 4),
        )

    @cute.jit
    def copy_clamped_b(self, atom, shared_copy, global_copy, stage, kt, barrier):
        for quarter in cutlass.range_constexpr(4):
            cute.copy(
                atom,
                global_copy[(None, kt * 4 + quarter)],
                shared_copy[(None, stage * 4 + quarter)],
                tma_bar_ptr=barrier,
            )

    @cute.jit
    def intermediate_scale_tma(self, pointer, rows, k):
        # Native R128c4 bytes: (r%32)*16 + ((r%128)//32)*4 +
        # (r//128)*512*Q + g + q*512. Overfetch all four row groups.
        source = cute.make_tensor(
            cute.recast_ptr(pointer, dtype=cutlass.Uint8),
            cute.make_layout(
                (16, 32, k // 128, rows // 128), stride=(1, 16, 512, 512 * (k // 128))
            ),
        )
        destination = cute.make_layout((16, 8, 4, 1))
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            source,
            destination,
            (16, 8, 4, 1),
        )

    @cute.jit
    def intermediate_scale_partition(self, atom, mapped, shared):
        destination = cute.make_tensor(
            shared.iterator, cute.make_layout((16, 8, 4, 1, 3))
        )
        tiles = cute.local_tile(mapped, (16, 8, 4, 1), (None, None, None, None))
        return cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(destination, 0, 4),
            cute.group_modes(tiles, 0, 4),
        )

    @cute.jit
    def copy_intermediate_scale(
        self, atom, shared_copy, global_copy, stage, row_base, kt, barrier
    ):
        cute.copy(
            atom,
            global_copy[(None, 0, (row_base % 32) // 8, kt, row_base // 128)],
            shared_copy[(None, stage)],
            tma_bar_ptr=barrier,
        )

    @cute.jit
    def intermediate_scale_word(self, words, stage, row_base, limit, kc, lane):
        word = cutlass.Uint32(0)
        if lane < 8:
            word = cutlass.Uint32(0x7F7F7F7F)
            if row_base + lane < limit:
                h = (row_base % 128) // 32
                word = words[stage * 128 + kc * 32 + lane * 4 + h].to(cutlass.Uint32)
        return word

    @cute.jit
    def read_work(self, mp, state, records, work):
        mp.consumer_wait(state)
        for i in cutlass.range_constexpr(self.record_words):
            work[i] = records[(i, state.index)]
        cute.arch.fence_proxy("async.shared", space="cta")
        mp.consumer_release(state)

    @cute.jit
    def k_range(self, work, k):
        """K512 epoch range [begin, end) of the current record; whole K otherwise."""
        if cutlass.const_expr(self.enable_tail_stream_k):
            return work[5], work[6]
        return 0, k // 512

    @cute.jit
    def scan_valid_tiles(self, route_limit, active, slot, target, lane):
        """Count route tiles whose 8-row slot is valid and find the tile that
        holds valid ordinal `target` (-1 when absent). Warp-cooperative over
        the same device routing words the walk reads; every lane ballots."""
        count = cutlass.Int32(0)
        found = cutlass.Int32(-1)
        base = cutlass.Int32(0)
        inclusive = cutlass.Uint32(0xFFFFFFFF) >> (31 - lane)
        while base < active:
            tile = base + lane
            valid = cutlass.Boolean(False)
            if tile < active:
                valid = tile * 128 + slot * 8 < route_limit[tile]
            mask = cute.arch.vote_ballot_sync(valid)
            below = cute.arch.popc(mask & (inclusive >> 1))
            hit = cute.arch.vote_ballot_sync(valid & (count + below == target))
            if hit != 0:
                found = base + cute.arch.popc(hit - 1)
            count += cute.arch.popc(mask)
            base += 32
        return count, found

    @cute.jit
    def locate_valid_tile(
        self, route_limit, active, planes: cutlass.Constexpr, count0, ordinal, lane
    ):
        """(slot, tile) of valid pair ordinal `ordinal` in walk order: slot-0
        tiles first, then slot-1 tiles when two planes are walked."""
        slot = cutlass.Int32(0)
        target = ordinal
        if cutlass.const_expr(planes == 2):
            if ordinal >= count0:
                slot = cutlass.Int32(1)
                target = ordinal - count0
        _, tile = self.scan_valid_tiles(route_limit, active, slot, target, lane)
        return slot, tile

    @cute.jit
    def tail_split(
        self,
        route_limit,
        active,
        planes: cutlass.Constexpr,
        grid_x,
        block_id,
        epochs,
        lane,
    ):
        """Tail cut of one launch. Returns the first walk position that is not a
        full item (positions are feature-fastest, plane-major), the number of
        full items, the slot-0 valid tile count and this CTA's unit range
        [begin, end) over the tail's K512 epochs (end == begin: nothing split).
        Splitting requires at least one full round and a shorter tail round."""
        count0, _ = self.scan_valid_tiles(
            route_limit, active, 0, cutlass.Int32(-1), lane
        )
        count1 = cutlass.Int32(0)
        if cutlass.const_expr(planes == 2):
            count1, _ = self.scan_valid_tiles(
                route_limit, active, 1, cutlass.Int32(-1), lane
            )
        items = (count0 + count1) * 56
        span = active * 56
        cut = planes * span
        full_items = items
        begin = cutlass.Int32(0)
        end = cutlass.Int32(0)
        if items > grid_x:
            rounds = cute.ceil_div(items, grid_x)
            full = (rounds - 1) * grid_x
            units = (items - full) * epochs
            per_cta = cute.ceil_div(units, grid_x)
            if per_cta < epochs:
                full_items = full
                pair = full // 56
                feature = full - pair * 56
                slot, tile = self.locate_valid_tile(
                    route_limit, active, planes, count0, pair, lane
                )
                cut = slot * span + tile * 56 + feature
                begin = cutlass.min(block_id * per_cta, units)
                end = cutlass.min(begin + per_cta, units)
        return cut, full_items, count0, begin, end

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        permuted_idx_to_expanded_idx_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        token_final_scales_ptr: cute.Pointer,
        a_per_token_scale_ptr: Optional[cute.Pointer],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,
        num_tokens: cutlass.Int64,
        top_k: cutlass.Int64,
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        weights = cute.make_tensor(
            b_ptr, cute.make_layout((n, k, l), stride=(k, 1, n * k))
        )
        sfa_tma, sfa_ma = self.weight_scale_tma(b_sf_ptr, n, k, l)
        x = cute.make_tensor(a_ptr, cute.make_layout((m, k), stride=(k, 1)))
        sfb_tma, sfb_ma = self.intermediate_scale_tma(a_sf_ptr, m, k)
        route_expert = cute.make_tensor(
            tile_idx_to_group_idx_ptr, cute.make_layout(m // 128)
        )
        route_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, cute.make_layout(m // 128)
        )
        routes = cute.make_tensor(permuted_idx_to_expanded_idx_ptr, cute.make_layout(m))
        active_tiles = cute.make_tensor(num_non_exiting_tiles_ptr, cute.make_layout(1))
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout(l))
        route_weights = cute.make_tensor(
            token_final_scales_ptr,
            cute.make_layout((num_tokens, top_k), stride=(top_k, 1)),
        )
        out = cute.make_tensor(c_ptr, cute.make_layout((num_tokens, n), stride=(n, 1)))
        mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            32,
            self.cta_group,
            self.tile[:2],
        )
        al = sm100_utils.make_smem_layout_a(mma, self.tile, cutlass.Int8, 3)
        bl = sm100_utils.make_smem_layout_b(mma, self.tile, cutlass.Float8E4M3FN, 3)
        b_tma, b_mapped = self.clamped_b_tma(a_ptr, k, bl.inner)
        sal = blockscaled_utils.make_smem_layout_sfa(mma, self.scale_tile, 32, 12)
        sbl = blockscaled_utils.make_smem_layout_sfb(mma, self.scale_tile, 32, 4)
        cluster = cute.tiled_divide(cute.make_layout((1, 1, 1, 1)), (mma.thr_id.shape,))
        tma, ma = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            weights,
            cute.slice_(al, (None, None, None, 0)),
            self.tile,
            mma,
            cluster.shape,
            internal_type=cutlass.Int8,
        )

        @cute.struct
        class Storage:
            barriers: cute.struct.MemRange[cutlass.Int64, 44]
            tmem_address: cutlass.Int32
            a: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, cute.cosize(al.outer)], 1024
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(bl.outer)], 1024
            ]
            sa: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, 6144], 128]
            sb: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 1536], 128]
            work: cute.struct.MemRange[cutlass.Int32, self.work_words]
            output_rows: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, 1024], 128
            ]

        self.shared_storage = Storage
        self.kernel(
            mma,
            tma,
            ma,
            sfa_tma,
            sfa_ma,
            b_tma,
            b_mapped,
            x,
            sfb_tma,
            sfb_ma,
            routes,
            route_expert,
            route_limit,
            active_tiles,
            alpha,
            route_weights,
            out,
            al,
            bl,
            sal,
            sbl,
        ).launch(
            grid=(max_active_clusters, 1, 1),
            block=(512, 1, 1),
            smem=Storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mma,
        tma,
        ma,
        sfa_tma,
        sfa_ma,
        b_tma,
        b_mapped,
        x,
        sfb_tma,
        sfb_ma,
        routes,
        route_expert,
        route_limit,
        active_tiles,
        alpha,
        route_weights,
        out,
        al,
        bl,
        sal,
        sbl,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block_id, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tid % 32
        k = cute.size(x, mode=[1])
        features = cute.size(out, mode=[1])
        feature_tiles = features // 128
        token_subtiles = cute.ceil_div(cute.size(out, mode=[0]), 8)
        top_k = cute.size(route_weights, mode=[1])
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        bars = storage.barriers.data_ptr()
        ap = pipeline.PipelineTmaUmma.create(
            num_stages=3,
            barrier_storage=bars,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tma_bytes,
            cta_layout_vmnk=None,
            defer_sync=True,
        )
        bp = pipeline.PipelineTmaUmma.create(
            num_stages=3,
            barrier_storage=bars + 6,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.b_tma_bytes,
            cta_layout_vmnk=None,
            defer_sync=True,
        )
        sap = pipeline.PipelineTmaUmma.create(
            num_stages=3,
            barrier_storage=bars + 12,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=2048,
            cta_layout_vmnk=None,
            defer_sync=True,
        )
        sbp = pipeline.PipelineTmaAsync.create(
            num_stages=3,
            barrier_storage=bars + 18,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            tx_count=self.sfb_tma_bytes,
            cta_layout_vmnk=None,
            # Single CTA: logical signaling index zero makes every reader
            # signal; preserve all 128 thread arrivals, not four warp arrivals.
            tidx=cutlass.Int32(0),
            enable_multicast_signaling=False,
            defer_sync=True,
        )
        tap = pipeline.PipelineAsyncUmma.create(
            num_stages=3,
            barrier_storage=bars + 24,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            cta_layout_vmnk=None,
            defer_sync=True,
        )
        tbp = pipeline.PipelineAsyncUmma.create(
            num_stages=3,
            barrier_storage=bars + 30,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            cta_layout_vmnk=None,
            defer_sync=True,
        )
        cp = pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            barrier_storage=bars + 36,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            cta_layout_vmnk=None,
            defer_sync=True,
        )
        mp = pipeline.PipelineAsync.create(
            num_stages=2,
            barrier_storage=bars + 40,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 480),
            defer_sync=True,
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        sa = storage.a.get_tensor(al.outer, swizzle=al.inner)
        sb = storage.b.get_tensor(bl.outer, swizzle=bl.inner)
        b_tma_shared = storage.b.get_tensor(
            cute.make_layout((128, 8, 1, 1, 12)), swizzle=bl.inner
        )
        sfa = storage.sa.get_tensor(sal)
        sfa_copy, gfa_copy = self.weight_scale_partition(sfa_tma, sfa_ma, sfa)
        raw_sfb = storage.sb.get_tensor(cute.make_layout(1536))
        sfb_words = cute.make_tensor(
            cute.recast_ptr(raw_sfb.iterator, dtype=cutlass.Uint32),
            cute.make_layout(384),
        )
        sfb_copy, gfb_copy = self.intermediate_scale_partition(sfb_tma, sfb_ma, raw_sfb)
        tmem = utils.TmemAllocator(
            storage.tmem_address.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
        )
        tmem.allocate(128)
        tmem.wait_for_alloc()
        acc_ptr = tmem.retrieve_ptr(cutlass.Float32)
        acc_layout = mma.make_fragment_C(mma.partition_shape_C(self.tile[:2])).layout
        sat = blockscaled_utils.make_tmem_layout_sfa(
            mma, self.scale_tile, 32, cute.slice_(sal, (None, None, None, 0))
        )
        sbt = blockscaled_utils.make_tmem_layout_sfb(
            mma, self.scale_tile, 32, cute.slice_(sbl, (None, None, None, 0))
        )
        ga = mma.get_slice(0).partition_A(
            cute.local_tile(ma, (128, 512), (None, None, None))
        )
        sa_copy, ga_copy = cpasync.tma_partition(
            tma,
            0,
            cute.make_layout(1),
            cute.group_modes(sa, 0, 3),
            cute.group_modes(ga, 0, 3),
        )
        fa, fb = mma.make_fragment_A(sa), mma.make_fragment_B(sb)
        records = storage.work.get_tensor(
            cute.make_layout((self.record_words, 2), stride=(1, self.record_words))
        )
        work = cute.make_rmem_tensor((self.record_words,), cutlass.Int32)
        output_rows = storage.output_rows.get_tensor(self.output_shared_layout())

        # Warp15 schedules valid virtual N8 work from existing device routing.
        if warp == 15:
            ms = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
            if cutlass.const_expr(self.enable_tail_stream_k):
                epochs = (k // 512).to(cutlass.Int32)
                planes = 1 if self.enable_t4_const_scheduler else 2
            if cutlass.const_expr(self.enable_t4_const_scheduler):
                total = active_tiles[0] * 56
                linear_work = cutlass.Int32(block_id)
                if cutlass.const_expr(self.enable_tail_stream_k):
                    # Peeled first walk position: block_id < grid_x is never at or
                    # beyond the cut (a split implies cut >= (rounds - 1) * grid_x
                    # >= grid_x; without a split there is no cut), so its record is
                    # emitted before the valid-tile scan and the consumers' first
                    # item overlaps the scan. Textual copy of the loop body below.
                    if linear_work < total:
                        feature = linear_work % 56
                        slot = cutlass.Int32(0)
                        route_tile = linear_work // 56
                        limit = route_limit[route_tile]
                        row_base = route_tile * 128 + slot * 8
                        if row_base < limit:
                            mp.producer_acquire(ms)
                            expert = route_expert[route_tile]
                            with cute.arch.elect_one():
                                records[(0, ms.index)] = feature.to(cutlass.Int32)
                                records[(1, ms.index)] = route_tile.to(cutlass.Int32)
                                records[(2, ms.index)] = slot.to(cutlass.Int32)
                                records[(3, ms.index)] = expert
                                records[(4, ms.index)] = limit
                                if cutlass.const_expr(self.enable_tail_stream_k):
                                    records[(5, ms.index)] = cutlass.Int32(0)
                                    records[(6, ms.index)] = epochs
                            cute.arch.fence_proxy("async.shared", space="cta")
                            mp.producer_commit(ms)
                            ms.advance()
                        linear_work += grid_x
                    # Positions at or beyond the cut are tail pieces, not full items.
                    cut, full_items, count0, unit_begin, unit_end = self.tail_split(
                        route_limit,
                        active_tiles[0],
                        planes,
                        grid_x,
                        block_id,
                        epochs,
                        lane,
                    )
                    total = cut
                while linear_work < total:
                    feature = linear_work % 56
                    slot = cutlass.Int32(0)
                    route_tile = linear_work // 56
                    limit = route_limit[route_tile]
                    row_base = route_tile * 128 + slot * 8
                    if row_base < limit:
                        mp.producer_acquire(ms)
                        expert = route_expert[route_tile]
                        with cute.arch.elect_one():
                            records[(0, ms.index)] = feature.to(cutlass.Int32)
                            records[(1, ms.index)] = route_tile.to(cutlass.Int32)
                            records[(2, ms.index)] = slot.to(cutlass.Int32)
                            records[(3, ms.index)] = expert
                            records[(4, ms.index)] = limit
                            if cutlass.const_expr(self.enable_tail_stream_k):
                                records[(5, ms.index)] = cutlass.Int32(0)
                                records[(6, ms.index)] = epochs
                        cute.arch.fence_proxy("async.shared", space="cta")
                        mp.producer_commit(ms)
                        ms.advance()
                    linear_work += grid_x
            elif cutlass.const_expr(self.enable_t16_slot_planes):
                span = active_tiles[0] * 56
                linear_work = cutlass.Int32(block_id)
                if cutlass.const_expr(self.enable_tail_stream_k):
                    # Peeled first plane-0 position (slot 0): block_id < grid_x is
                    # never at or beyond the cut (a split implies cut >= (rounds - 1)
                    # * grid_x >= grid_x; without a split there is no cut), so the
                    # cut test is omitted and the record is emitted before the
                    # valid-tile scan. Textual copy of the plane-0 loop body below
                    # with slot = 0.
                    if linear_work < span:
                        feature = linear_work % 56
                        route_tile = linear_work // 56
                        limit = route_limit[route_tile]
                        row_base = route_tile * 128
                        if row_base < limit:
                            mp.producer_acquire(ms)
                            expert = route_expert[route_tile]
                            with cute.arch.elect_one():
                                records[(0, ms.index)] = feature.to(cutlass.Int32)
                                records[(1, ms.index)] = route_tile.to(cutlass.Int32)
                                records[(2, ms.index)] = cutlass.Int32(0)
                                records[(3, ms.index)] = expert
                                records[(4, ms.index)] = limit
                                if cutlass.const_expr(self.enable_tail_stream_k):
                                    records[(5, ms.index)] = cutlass.Int32(0)
                                    records[(6, ms.index)] = epochs
                            cute.arch.fence_proxy("async.shared", space="cta")
                            mp.producer_commit(ms)
                            ms.advance()
                        linear_work += grid_x
                    cut, full_items, count0, unit_begin, unit_end = self.tail_split(
                        route_limit,
                        active_tiles[0],
                        planes,
                        grid_x,
                        block_id,
                        epochs,
                        lane,
                    )
                # Carry each CTA's global ordinal across the two slot planes.
                # Resetting here would put both plane remainders on the same CTAs.
                for slot in cutlass.range_constexpr(2):
                    while linear_work < span:
                        feature = linear_work % 56
                        route_tile = linear_work // 56
                        limit = route_limit[route_tile]
                        if cutlass.const_expr(self.enable_tail_stream_k):
                            # Positions at or beyond the cut are tail pieces, not
                            # full items; an empty limit fails the validity test.
                            if slot * span + linear_work >= cut:
                                limit = cutlass.Int32(0)
                        row_base = route_tile * 128 + slot * 8
                        if row_base < limit:
                            mp.producer_acquire(ms)
                            expert = route_expert[route_tile]
                            with cute.arch.elect_one():
                                records[(0, ms.index)] = feature.to(cutlass.Int32)
                                records[(1, ms.index)] = route_tile.to(cutlass.Int32)
                                records[(2, ms.index)] = cutlass.Int32(slot)
                                records[(3, ms.index)] = expert
                                records[(4, ms.index)] = limit
                                if cutlass.const_expr(self.enable_tail_stream_k):
                                    records[(5, ms.index)] = cutlass.Int32(0)
                                    records[(6, ms.index)] = epochs
                            cute.arch.fence_proxy("async.shared", space="cta")
                            mp.producer_commit(ms)
                            ms.advance()
                        linear_work += grid_x
                    linear_work -= span
            else:
                if cutlass.const_expr(self.enable_t16_const_scheduler):
                    total = active_tiles[0] * 112
                    linear_work = cutlass.Int32(block_id)
                else:
                    total = active_tiles[0] * feature_tiles * token_subtiles
                    linear_work = cutlass.Int64(block_id)
                while linear_work < total:
                    # Feature tiles vary fastest, so an even CTA stride does not
                    # pin half the CTAs to empty slot1 on sparse T9..16 routing.
                    if cutlass.const_expr(self.enable_t16_const_scheduler):
                        feature = linear_work % 56
                        slot = (linear_work // 56) % 2
                        route_tile = linear_work // 112
                    else:
                        feature = linear_work % feature_tiles
                        slot = (linear_work // feature_tiles) % token_subtiles
                        route_tile = linear_work // (token_subtiles * feature_tiles)
                    limit = route_limit[route_tile]
                    row_base = route_tile * 128 + slot * 8
                    if row_base < limit:
                        mp.producer_acquire(ms)
                        expert = route_expert[route_tile]
                        with cute.arch.elect_one():
                            records[(0, ms.index)] = feature.to(cutlass.Int32)
                            records[(1, ms.index)] = route_tile.to(cutlass.Int32)
                            records[(2, ms.index)] = slot.to(cutlass.Int32)
                            records[(3, ms.index)] = expert
                            records[(4, ms.index)] = limit
                        cute.arch.fence_proxy("async.shared", space="cta")
                        mp.producer_commit(ms)
                        ms.advance()
                    linear_work += grid_x
            if cutlass.const_expr(self.enable_tail_stream_k):
                # Tail pieces: this CTA's contiguous K512 units of the tail items,
                # in walk order; a unit range crosses at most one item boundary.
                unit = unit_begin
                while unit < unit_end:
                    item = unit // epochs
                    k_begin = unit - item * epochs
                    k_end = cutlass.min(epochs, k_begin + (unit_end - unit))
                    ordinal = full_items + item
                    pair = ordinal // 56
                    feature = ordinal - pair * 56
                    slot, route_tile = self.locate_valid_tile(
                        route_limit, active_tiles[0], planes, count0, pair, lane
                    )
                    mp.producer_acquire(ms)
                    expert = route_expert[route_tile]
                    limit = route_limit[route_tile]
                    with cute.arch.elect_one():
                        records[(0, ms.index)] = feature
                        records[(1, ms.index)] = route_tile
                        records[(2, ms.index)] = slot
                        records[(3, ms.index)] = expert
                        records[(4, ms.index)] = limit
                        records[(5, ms.index)] = k_begin
                        records[(6, ms.index)] = k_end
                    cute.arch.fence_proxy("async.shared", space="cta")
                    mp.producer_commit(ms)
                    ms.advance()
                    unit = unit + (k_end - k_begin)
            mp.producer_acquire(ms)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.record_words):
                    records[(i, ms.index)] = cutlass.Int32(-1)
            cute.arch.fence_proxy("async.shared", space="cta")
            mp.producer_commit(ms)
            ms.advance()
            mp.producer_tail(ms)

        # Warp 11: only native packed-weight TMA, one K512 transaction/stage.
        if warp == 11:
            a_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 3
            )
            meta_a = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            self.read_work(mp, meta_a, records, work)
            meta_a.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 8, work[4]
                kb, ke = self.k_range(work, k)
                for kt in cutlass.range(kb, ke, unroll=1):
                    ap.producer_acquire(a_state)
                    cute.copy(
                        tma,
                        ga_copy[(None, bm, kt, expert)],
                        sa_copy[(None, a_state.index)],
                        tma_bar_ptr=ap.producer_get_barrier(a_state),
                    )
                    a_state.advance()
                self.read_work(mp, meta_a, records, work)
                meta_a.advance()
            ap.producer_tail(a_state)

        # Warp8: four ordinary N8/K128 TMA copies into each native K512 slot.
        if warp == 8:
            b_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 3
            )
            meta_b = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            self.read_work(mp, meta_b, records, work)
            meta_b.advance()
            while work[0] >= 0:
                row_base, limit = work[1] * 128 + work[2] * 8, work[4]
                b_shared_copy, b_global_copy = self.clamped_b_partition(
                    b_tma, b_mapped, b_tma_shared, row_base, limit
                )
                kb, ke = self.k_range(work, k)
                for kt in cutlass.range(kb, ke, unroll=1):
                    bp.producer_acquire(b_state)
                    self.copy_clamped_b(
                        b_tma,
                        b_shared_copy,
                        b_global_copy,
                        b_state.index,
                        kt,
                        bp.producer_get_barrier(b_state),
                    )
                    b_state.advance()
                self.read_work(mp, meta_b, records, work)
                meta_b.advance()
            bp.producer_tail(b_state)

        # Warp9 keeps its 32 metadata consumers, including the sentinel.
        # It no longer touches the B data ring or contributes B arrivals.
        if warp == 9:
            meta_b = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            self.read_work(mp, meta_b, records, work)
            meta_b.advance()
            while work[0] >= 0:
                self.read_work(mp, meta_b, records, work)
                meta_b.advance()

        # Warp 12: one 2048-byte TMA into the unchanged expanded SFA ring.
        # Warp13 releases its shared lifetime only after CP reads complete.
        if warp == 12:
            sa_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 3
            )
            meta_sa = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            self.read_work(mp, meta_sa, records, work)
            meta_sa.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 8, work[4]
                kb, ke = self.k_range(work, k)
                for kt in cutlass.range(kb, ke, unroll=1):
                    sap.producer_acquire(sa_prod)
                    # kt is a K512 tile coordinate; local_tile maps it to
                    # the raw source K128 coordinate 4*kt.
                    cute.copy(
                        sfa_tma,
                        gfa_copy[(None, 0, kt, bm, expert)],
                        sfa_copy[(None, sa_prod.index)],
                        tma_bar_ptr=sap.producer_get_barrier(sa_prod),
                    )
                    sa_prod.advance()
                self.read_work(mp, meta_sa, records, work)
                meta_sa.advance()
            sap.producer_tail(sa_prod)

        # Warp 10: one 512-byte SFB TMA; publishers select 128 useful bytes.
        if warp == 10:
            sb_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 3
            )
            meta_sb = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            self.read_work(mp, meta_sb, records, work)
            meta_sb.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 8, work[4]
                kb, ke = self.k_range(work, k)
                for kt in cutlass.range(kb, ke, unroll=1):
                    sbp.producer_acquire(sb_prod)
                    self.copy_intermediate_scale(
                        sfb_tma,
                        sfb_copy,
                        gfb_copy,
                        sb_prod.index,
                        row_base,
                        kt,
                        sbp.producer_get_barrier(sb_prod),
                    )
                    sb_prod.advance()
                self.read_work(mp, meta_sb, records, work)
                meta_sb.advance()
            sbp.producer_tail(sb_prod)

        # Warp 13: four K128 CPs into this stage's independent 16-column
        # TMEM SFA slot. CP-to-MMA is the PTX pipelined cross-thread pattern.
        if warp == 13:
            sa_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 3
            )
            ta_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 3
            )
            meta_ta = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            self.read_work(mp, meta_ta, records, work)
            meta_ta.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 8, work[4]
                kb, ke = self.k_range(work, k)
                for kt in cutlass.range(kb, ke, unroll=1):  # noqa: B007
                    sap.consumer_wait(sa_cons)
                    tap.producer_acquire(ta_prod)
                    tcgen05_fence_after_thread_sync()
                    for kc in cutlass.range_constexpr(4):
                        ta_stage = cute.make_tensor(
                            cute.recast_ptr(
                                acc_ptr + 16 + ta_prod.index * 16 + kc * 4,
                                dtype=self.sf_dtype,
                            ),
                            sat,
                        )
                        cpa, cpa_src, cpa_dst = self.scale_copy(sfa, ta_stage)
                        cute.copy(
                            cpa,
                            cpa_src[(None, None, None, None, sa_cons.index * 4 + kc)],
                            cpa_dst,
                        )
                    tcgen05_fence_before_thread_sync()
                    tap.producer_commit(ta_prod)
                    # This method emits an elected tcgen05.commit: shared SFA
                    # cannot be overwritten until the preceding CP reads finish.
                    sap.consumer_release(sa_cons)
                    sa_cons.advance()
                    ta_prod.advance()
                self.read_work(mp, meta_ta, records, work)
                meta_ta.advance()
            tap.producer_tail(ta_prod)

        # Warps 4..7: four-warp compact scale-factor publication.
        if warp >= 4 and warp < 8:
            sb_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 3
            )
            tb_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 3
            )
            copy_tid = (warp - 4) * 32 + lane
            meta_tb = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            self.read_work(mp, meta_tb, records, work)
            meta_tb.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 8, work[4]
                kb, ke = self.k_range(work, k)
                for kt in cutlass.range(kb, ke, unroll=1):  # noqa: B007
                    sbp.consumer_wait(sb_cons)
                    tbp.producer_acquire(tb_prod)
                    tcgen05_fence_after_thread_sync()
                    for kc in cutlass.range_constexpr(4):
                        word_tensor = cute.make_tensor(
                            cute.recast_ptr(
                                acc_ptr + 64 + tb_prod.index * 8 + kc * 2,
                                dtype=cutlass.Uint32,
                            ),
                            cute.make_layout((128, 1), stride=(65536, 1)),
                        )
                        st = tcgen05.make_tmem_copy(
                            cute.make_copy_atom(
                                tcgen05.St32x32bOp(tcgen05.Repetition.x1),
                                cutlass.Uint32,
                            ),
                            word_tensor,
                        )
                        thread_st = st.get_slice(copy_tid)
                        coords = thread_st.partition_S(
                            cute.make_identity_tensor((128, 1))
                        )
                        registers = cute.make_rmem_tensor(coords.shape, cutlass.Uint32)
                        assert cute.size(registers) == 1
                        word = self.intermediate_scale_word(
                            sfb_words, sb_cons.index, row_base, limit, kc, lane
                        )
                        registers.fill(word)
                        cute.copy(st, registers, thread_st.partition_D(word_tensor))
                    cute.arch.fence_view_async_tmem_store()
                    tcgen05_fence_before_thread_sync()
                    tbp.producer_commit(tb_prod)
                    # Every one of 128 readers has finished shared reads before
                    # the ordinary empty event; each also completed its own ST.
                    sbp.consumer_release(sb_cons)
                    sb_cons.advance()
                    tb_prod.advance()
                self.read_work(mp, meta_tb, records, work)
                meta_tb.advance()
            tbp.producer_tail(tb_prod)

        # Warp 14: obtains four independent ready tokens, issues precisely
        # sixteen ordered K32 MMAs, releases each lifetime by completion.
        if warp == 14:
            ac = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 3)
            bc = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 3)
            tac = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 3)
            tbc = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 3)
            meta_mma = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            acc_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 2
            )
            # Address encodings are invariant across all independent ring slots.
            # Use the actual allocated TMEM base and original typed SF layouts.
            selector_a0 = cute.make_tensor(
                cute.recast_ptr(acc_ptr + 16, dtype=self.sf_dtype), sat
            )
            selector_b0 = cute.make_tensor(
                cute.recast_ptr(acc_ptr + 64, dtype=self.sf_dtype), sbt
            )
            d0 = finalize_instruction_selector(mma, selector_a0, selector_b0, 0)
            d1 = finalize_instruction_selector(mma, selector_a0, selector_b0, 1)
            d2 = finalize_instruction_selector(mma, selector_a0, selector_b0, 2)
            d3 = finalize_instruction_selector(mma, selector_a0, selector_b0, 3)
            self.read_work(mp, meta_mma, records, work)
            meta_mma.advance()
            while work[0] >= 0:
                cp.producer_acquire(acc_prod)
                acc = cute.make_tensor(acc_ptr + acc_prod.index * 8, acc_layout)
                mma.set(tcgen05.Field.ACCUMULATE, False)
                a_ready = cutlass.Boolean(0)
                if cutlass.const_expr(self.enable_t4_const_scheduler):
                    a_ready = ap.consumer_try_wait(ac)
                kb, ke = self.k_range(work, k)
                for kt in cutlass.range(kb, ke, unroll=1):
                    if cutlass.const_expr(self.enable_t4_const_scheduler):
                        ap.consumer_wait(ac, a_ready)
                    else:
                        ap.consumer_wait(ac)
                    bp.consumer_wait(bc)
                    tap.consumer_wait(tac)
                    tbp.consumer_wait(tbc)
                    tcgen05_fence_after_thread_sync()
                    ta0 = cute.make_tensor(
                        cute.recast_ptr(
                            acc_ptr + 16 + tac.index * 16, dtype=self.sf_dtype
                        ),
                        sat,
                    )
                    tb0 = cute.make_tensor(
                        cute.recast_ptr(
                            acc_ptr + 64 + tbc.index * 8, dtype=self.sf_dtype
                        ),
                        sbt,
                    )
                    if cutlass.const_expr(self.enable_t4_const_scheduler):
                        next_a_ready = a_ready
                        if kt + 1 < ke:
                            next_ac = ac.clone()
                            next_ac.advance()
                            next_a_ready = ap.consumer_try_wait(next_ac)
                    issue_finalize_epoch(
                        mma, acc, fa, fb, ta0, tb0, ac.index, bc.index, d0, d1, d2, d3
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                    ap.consumer_release(ac)
                    bp.consumer_release(bc)
                    tap.consumer_release(tac)
                    tbp.consumer_release(tbc)
                    ac.advance()
                    bc.advance()
                    tac.advance()
                    tbc.advance()
                    if cutlass.const_expr(self.enable_t4_const_scheduler):
                        a_ready = next_a_ready
                cp.producer_commit(acc_prod)
                acc_prod.advance()
                self.read_work(mp, meta_mma, records, work)
                meta_mma.advance()
            cp.producer_tail(acc_prod)

        # Four-warp TMEM load, token-major BF16 bridge, then existing bulk add.
        if warp < 4:
            meta_epi = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            acc_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            self.read_work(mp, meta_epi, records, work)
            meta_epi.advance()
            while work[0] >= 0:
                row_base, limit = work[1] * 128 + work[2] * 8, work[4]
                token = cutlass.Int32(0)
                combined_scale = cutlass.Float32(0.0)
                # All warps initialize every lane before uniform shuffles.
                if lane < 8:
                    if row_base + lane < limit:
                        expanded = routes[row_base + lane]
                        token = (expanded // top_k).to(cutlass.Int32)
                        choice = expanded % top_k
                        combined_scale = cutlass.Float32(
                            alpha[work[3]] * route_weights[(token, choice)]
                        )
                cp.consumer_wait(acc_cons)
                tcgen05_fence_after_thread_sync()
                acc_out = cute.make_tensor(acc_ptr + acc_cons.index * 8, acc_layout)
                acc_mn = acc_out[((None, None), 0, 0)]
                tc, source, coords = self.epilogue_partition(acc_mn, tid)
                values = cute.make_rmem_tensor(coords.shape, cutlass.Float32)
                cute.copy(tc, source, values)
                cute.arch.fence_view_async_tmem_load()
                tcgen05_fence_before_thread_sync()
                for vi in cutlass.range_constexpr(cute.size(values)):
                    channel, slot = coords[vi]
                    scale = cute.arch.shuffle_sync(combined_scale, cutlass.Int32(slot))
                    if row_base + slot < limit:
                        output_rows[(slot, channel)] = (scale * values[vi]).to(
                            cutlass.BFloat16
                        )
                cute.arch.fence_proxy("async.shared", space="cta")
                self.epilog_sync_barrier.arrive_and_wait()
                cp.consumer_release(acc_cons)
                acc_cons.advance()
                # Threads0..7 own one contiguous 256-byte BF16 row each.
                if tid < 8:
                    if row_base + tid < limit:
                        destination = cute.domain_offset((token, work[0] * 128), out)
                        blk_reduce_bf16(
                            destination, output_rows[tid, None], cutlass.Int32(256)
                        )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                self.epilog_sync_barrier.arrive_and_wait()
                self.read_work(mp, meta_epi, records, work)
                meta_epi.advance()

        # Every producer tail and final output completes before storage is freed.
        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(acc_ptr)
