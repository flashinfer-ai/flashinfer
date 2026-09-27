# Copyright (c) 2026 KDA Team
# SPDX-License-Identifier: MIT
# Adapted from humanfia/kda-for-kda-release; see licenses/LICENSE.kda-for-kda.

# ruff: noqa: F841, SIM102
"""Persistent M128, BT32 Kimi Delta Attention for SM100/SM103.

Each CTA streams a longest-processing-time schedule of (sequence, head)
chains through one continuous five-stage preparation pipeline. The 128x128
recurrent state stays in FP32 tensor memory. Split chains exchange complete
FP32 state through a producer/consumer handoff; no warmup approximation is
used by the FlashInfer host path.

The 1024 threads are partitioned into compute (warps 0-3), epilogue (4-7),
MMA/load (8-11), and five four-warp preparation instances (12-31). Preparation
fuses the bounded K3 gate, FP32 cumulative log decay, exact Q/K L2 norms,
decorated Q/K, tensor-core Gram products and a hierarchical triangular solve.
Compute performs the recurrent updates with BF16 MMA operands and FP32
accumulators; epilogue stores BF16 outputs and FP32 terminal state.

Tensor-memory columns: input/residual 0-63, master state 64-191, output
192-223, U 224-255, and five preparation Gram windows 256-415. The allocation
is rounded to 512 columns. Shared stage storage is reused only after its
smem-free barrier: gate scan, decorations, Gram/solve, restored operands,
then V staging. FP32 gate-prefix storage is retained on every route.
"""

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import cute, pipeline, utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.arch import nvvm_wrappers
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def _bulk_prefetch_l2(addr, size, *, loc=None, ip=None):
    gmem_addr = llvm.addrspacecast(
        llvm.PointerType.get(cutlass.AddressSpace.gmem.value),
        addr.to_llvm_ptr(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    nvvm_wrappers.nvvm.cp_async_bulk_prefetch(gmem_addr, size, loc=loc, ip=ip)


@dsl_user_op
def _load_state_f32x8(src, dst, *, loc=None, ip=None):
    """Load eight adjacent FP32 state values without allocating in L1."""
    addr_i64 = llvm.ptrtoint(
        T.i64(), src.iterator.to_llvm_ptr(loc=loc, ip=ip), loc=loc, ip=ip
    )
    result_type = llvm.StructType.get_literal([T.f32()] * 8)
    values = llvm.inline_asm(
        result_type,
        [addr_i64],
        "ld.global.L1::no_allocate.v8.f32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];",
        "=f,=f,=f,=f,=f,=f,=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    for index in range(8):
        value = llvm.extractvalue(T.f32(), values, [index], loc=loc, ip=ip)
        dst[index] = cutlass.Float32(value)


@dsl_user_op
def _store_final_f32x8(dst, values, *, loc=None, ip=None):
    """Store eight adjacent FP32 final-state values without allocating in L1."""
    addr = dst.iterator
    addr_i64 = llvm.ptrtoint(T.i64(), addr.to_llvm_ptr(loc=loc, ip=ip), loc=loc, ip=ip)
    operands = [addr_i64]
    operands.extend(values[i].ir_value(loc=loc, ip=ip) for i in range(8))
    llvm.inline_asm(
        None,
        operands,
        "st.global.L1::no_allocate.v8.f32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};",
        "l,f,f,f,f,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


D = 128
C = 32
BM = 128
NFT = 192
STAGES = 5
THREADS = 1024
# 256 cols hold the recurrent map (INP/vst/master/OUT/U); the gram
# windows (one [64,32] f32 window per prep stage) sit at GRAM_COL+.
# TMEM allocation must be a power of two, so the alloc is the full 512.
TMEM_COLS = 512
GRAM_COL = 256
STAGE_BYTES = 40960
STAGE_ELTS = STAGE_BYTES // 2
STAGE_F32 = STAGE_BYTES // 4
OFF_QD = 0
OFF_KD = 8192
OFF_FT = 16384
OFF_GCS = 24576
OFF_INV = OFF_GCS + 6144
OFF_GT = OFF_GCS + 8192
OFF_RF = OFF_GCS + 8704
OFF_BT = OFF_GCS + 9344
OFF_V = OFF_GCS + 8192

MB_QK = 0
MB_GRAW = 5
MB_QKRAW = 10
MB_VFULL = 15
MB_VFREE = 20
MB_SFREE = 25
MB_RFREE = 30
MB_OOUT = 40
MB_U2ACC = 50
MB_OFIN = 55
MB_TAB = 45
MB_INP = 35
MB_FIN = 60
MB_OEMPTY = 65
MB_BRAW = 66
MB_GRAM = 71


def _exp2f(x):
    return cute.math.exp2(x, fastmath=True)


def _tanhf(x):
    return cute.math.tanh(x, fastmath=True)


@cute.struct
class PkdStorage:
    mbar: cute.struct.MemRange[cutlass.Int64, 76]
    tmem_holding: cutlass.Int32


@cute.kernel
def _pkd(
    mma_1: cute.TiledMma,
    mma_3: cute.TiledMma,
    mma_4a: cute.TiledMma,
    mma_4b: cute.TiledMma,
    mma_g: cute.TiledMma,
    tma_q: cute.CopyAtom,
    mQ: cute.Tensor,
    tma_k: cute.CopyAtom,
    mK: cute.Tensor,
    tma_g: cute.CopyAtom,
    mG: cute.Tensor,
    tma_b: cute.CopyAtom,
    mB: cute.Tensor,
    tma_v: cute.CopyAtom,
    mV: cute.Tensor,
    tma_o: cute.CopyAtom,
    mO: cute.Tensor,
    tma_e: cute.CopyAtom,
    mE: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    out_raw: cute.Tensor,
    state0: cute.Tensor,
    stateT: cute.Tensor,
    cu: cute.Tensor,
    soff: cute.Tensor,
    schain: cute.Tensor,
    spt0: cute.Tensor,
    sptn: cute.Tensor,
    ssrc: cute.Tensor,
    sdst: cute.Tensor,
    midstate: cute.Tensor,
    mflags: cute.Tensor,
    expt: cute.Tensor,
    tprobe: cute.Tensor,
    failure_addr: cutlass.Int64,
    have_state: cutlass.Int32,
    do_export: cutlass.Int32,
    export_seq: cutlass.Int32,
    nc2: cutlass.Int32,
    fepoch: cutlass.Int32,
    scale: cutlass.Float32,
    lb2: cutlass.Float32,
    lay_qd: cute.ComposedLayout,
    lay_inv: cute.ComposedLayout,
    lay_ft: cute.ComposedLayout,
    lay_v: cute.ComposedLayout,
    H_: cutlass.Constexpr[int],
    TPROBE_: cutlass.Constexpr[int] = 0,
    GATE2_: cutlass.Constexpr[int] = 0,
    FINAL_: cutlass.Constexpr[int] = 1,
    SPLIT_: cutlass.Constexpr[int] = 1,
    EXPORT_: cutlass.Constexpr[int] = 1,
    STATE_: cutlass.Constexpr[int] = 0,
    BETA_TMA_: cutlass.Constexpr[int] = 0,
    BF16_DV_: cutlass.Constexpr[int] = 0,
    BETA_BF16_: cutlass.Constexpr[int] = 0,
    QK_ROWPAIR_: cutlass.Constexpr[int] = 0,
    RCP_DECOR_: cutlass.Constexpr[int] = 1,
    RF_HOIST_: cutlass.Constexpr[int] = 1,
    RESTORE_TAIL_: cutlass.Constexpr[int] = 0,
    GRAM_W9_: cutlass.Constexpr[int] = 0,
    REG_MODE_: cutlass.Constexpr[int] = 0,
    FULL_CHUNKS_: cutlass.Constexpr[int] = 0,
    LOWER_BOUND_M5_: cutlass.Constexpr[int] = 0,
    WARM_SPLIT_: cutlass.Constexpr[int] = 0,
    PREP_PEEL_: cutlass.Constexpr[int] = 0,
    GATE_ROLL_: cutlass.Constexpr[int] = 0,
):
    tidx, _, _ = cute.arch.thread_idx()
    slot, _, _ = cute.arch.block_idx()
    # persistent multi-chain: this CTA owns chains schain[k0:k1] (LPT)
    k0 = cute.arch.make_warp_uniform(soff[slot])
    k1 = cute.arch.make_warp_uniform(soff[slot + 1])
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx & 31

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(PkdStorage)
    arena = smem.allocate_tensor(
        cutlass.Int8, cute.make_layout((STAGES * STAGE_BYTES,)), 1024
    )
    s_out = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((2 * C * D,)), 1024)
    s_gt5 = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((D, STAGES), stride=(1, D)), 16
    )
    # rf row stride padded D+1 -> 136 (16B-aligned rows/stages so the
    # restore's vec8 rf slices autovectorize to LDS.128 pairs)
    s_rf5 = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((D + 1, STAGES), stride=(1, 136)), 16
    )
    s_bt5 = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((C, STAGES), stride=(1, C)), 16
    )
    if cutlass.const_expr(BETA_TMA_ == 1):
        s_br5 = smem.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((C, 8, STAGES), stride=(8, 1, C * 8)),
            128,
        )
    else:
        s_br5 = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((1,)), 16)
    # Trailing mirror keeps every pre-existing shared address unchanged.
    s_btb5 = smem.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((C, STAGES), stride=(1, C)), 16
    )

    ar0 = arena.iterator

    # ---- multi-stage canonical operand views (stage stride = arena) ----
    # qd/kd interleave as one canonical [64,128] block (qd rows 0-31,
    # kd rows 32-63 at +2048 elts; k-blocks 4096 apart) so the prep gram
    # runs as a single m64n32k128 SS-A tcgen05 MMA over qd||kd.
    p_qd = cute.recast_ptr(ar0 + OFF_QD, lay_qd.inner, dtype=cutlass.BFloat16)

    p_kd = p_qd + 2048

    p_inv = cute.recast_ptr(ar0 + OFF_INV, lay_inv.inner, dtype=cutlass.BFloat16)
    t_binv = cute.make_tensor(
        p_inv,
        cute.make_layout(
            ((32, (8, 2)), 1, 2, STAGES), stride=((8, (1, 256)), 0, 512, STAGE_ELTS)
        ),
    )
    p_ft = cute.recast_ptr(ar0 + OFF_FT, lay_ft.inner, dtype=cutlass.BFloat16)
    t_bfta = cute.make_tensor(
        p_ft,
        cute.make_layout(
            (((64, 2), 16), 1, 2, STAGES), stride=(((1, 2048), 64), 0, 1024, STAGE_ELTS)
        ),
    )
    t_bftb = cute.make_tensor(
        p_ft + 4096,
        cute.make_layout(
            ((32, 16), 1, 2, STAGES), stride=((1, 64), 0, 1024, STAGE_ELTS)
        ),
    )
    t_bki = cute.make_tensor(
        p_ft,
        cute.make_layout(
            ((32, 16), 1, (4, 2), STAGES), stride=((64, 1), 0, (16, 2048), STAGE_ELTS)
        ),
    )
    b_ki = mma_g.make_fragment_B(t_bki)

    t_bqd = cute.make_tensor(
        p_qd,
        cute.make_layout(
            ((32, 16), 1, (4, 2), STAGES), stride=((64, 1), 0, (16, 4096), STAGE_ELTS)
        ),
    )
    t_bkd = cute.make_tensor(
        p_kd,
        cute.make_layout(
            ((32, 16), 1, (4, 2), STAGES), stride=((64, 1), 0, (16, 4096), STAGE_ELTS)
        ),
    )
    b_qd = mma_1.make_fragment_B(t_bqd)
    b_kd = mma_1.make_fragment_B(t_bkd)
    # gram operands: A = interleaved qd||kd [64,128], B = ki (ft slot)
    t_aqk = cute.make_tensor(
        p_qd,
        cute.make_layout(
            ((2 * C, 16), 1, (4, 2), STAGES),
            stride=((64, 1), 0, (16, 4096), STAGE_ELTS),
        ),
    )
    a_qk = mma_g.make_fragment_A(t_aqk)
    b_inv = mma_3.make_fragment_B(t_binv)
    b_fta = mma_4a.make_fragment_B(t_bfta)
    b_ftb = mma_4b.make_fragment_B(t_bftb)

    # logical fill views (pre-swizzle affine over swizzled pointers);
    # vec8-sliceable forms: (row, 8elt, (seg8, blk), stage)
    v_qd8 = cute.make_tensor(
        p_qd,
        cute.make_layout((C, 8, 8, 2, STAGES), stride=(64, 1, 8, 4096, STAGE_ELTS)),
    )
    v_kd8 = cute.make_tensor(
        p_kd,
        cute.make_layout((C, 8, 8, 2, STAGES), stride=(64, 1, 8, 4096, STAGE_ELTS)),
    )
    v_ki8 = cute.make_tensor(
        p_ft,
        cute.make_layout((C, 8, 8, 2, STAGES), stride=(64, 1, 8, 2048, STAGE_ELTS)),
    )
    # Gram halves share one swizzled [64,32] image: G^T occupies rows
    # 0..31 and L occupies the otherwise-unused rows 32..63.  Besides
    # reclaiming 2 KiB/stage, this matches the tcgen05 result's native
    # logical tile and enables a single cooperative matrix-store copy.
    p_gram_s = p_ft + 4096
    t_sgram = cute.make_tensor(
        p_gram_s,
        cute.make_layout(
            ((2 * C, C), 1, 1, STAGES), stride=((1, 64), 0, 0, STAGE_ELTS)
        ),
    )
    v_gram = cute.make_tensor(
        p_gram_s, cute.make_layout((2 * C, C, STAGES), stride=(1, 64, STAGE_ELTS))
    )
    v_lw_diag = cute.make_tensor(
        p_gram_s + C,
        cute.make_layout((4, 8, 8, STAGES), stride=(520, 1, 64, STAGE_ELTS)),
    )
    v_inv = cute.make_tensor(
        p_inv, cute.make_layout((C, 8, 4, STAGES), stride=(8, 1, 256, STAGE_ELTS))
    )
    # raw g stages in the qd byte-positions of the interleaved block
    # (plain within each 4 KiB half; the kd positions hold raw k)
    v_graw = cute.make_tensor(
        cute.recast_ptr(ar0 + OFF_QD, dtype=cutlass.BFloat16),
        cute.make_layout((C, (64, 2), STAGES), stride=(64, (1, 4096), STAGE_ELTS)),
    )
    v_graw8 = cute.make_tensor(
        cute.recast_ptr(ar0 + OFF_QD, dtype=cutlass.BFloat16),
        cute.make_layout((C, 8, (8, 2), STAGES), stride=(64, 1, (8, 4096), STAGE_ELTS)),
    )
    v_graw2 = cute.make_tensor(
        cute.recast_ptr(ar0 + OFF_QD, dtype=cutlass.BFloat16),
        cute.make_layout(
            (C, 2, (32, 2), STAGES), stride=(64, 1, (2, 4096), STAGE_ELTS)
        ),
    )
    p_gcs = cute.recast_ptr(ar0 + OFF_GCS, dtype=cutlass.Float32)
    gcs_stage = STAGE_F32
    v_gcs = cute.make_tensor(
        p_gcs, cute.make_layout((C, D, STAGES), stride=(D, 1, gcs_stage))
    )
    v_gcs8 = cute.make_tensor(
        p_gcs, cute.make_layout((C, 8, 16, STAGES), stride=(D, 1, 8, gcs_stage))
    )
    v_gcs2 = cute.make_tensor(
        p_gcs, cute.make_layout((C, 2, 64, STAGES), stride=(D, 1, 2, gcs_stage))
    )
    v_gt = s_gt5
    v_rf = s_rf5
    v_rf8 = cute.make_tensor(
        s_rf5.iterator, cute.make_layout((16, 8, STAGES), stride=(8, 1, 136))
    )
    v_bt = s_bt5
    v_btb = s_btb5
    v_braw = s_br5
    # pair views for LDS.64 table reads (fragment quads share col pairs)
    v_gt8 = cute.make_tensor(
        s_gt5.iterator, cute.make_layout((D // 8, 8, STAGES), stride=(8, 1, D))
    )
    v_bt8 = cute.make_tensor(
        s_bt5.iterator, cute.make_layout((C // 8, 8, STAGES), stride=(8, 1, C))
    )
    v_btb8 = cute.make_tensor(
        s_btb5.iterator, cute.make_layout((C // 8, 8, STAGES), stride=(8, 1, C))
    )
    # Fixed chains use an MN-major A-operand staging layout for wide
    # residual loads.  Varlen chains retain the measured-faster raw layout.
    p_v_wide = cute.recast_ptr(ar0 + OFF_V, lay_v.inner, dtype=cutlass.BFloat16)
    p_v_raw = cute.recast_ptr(ar0 + OFF_V, dtype=cutlass.BFloat16)
    v_v = cute.make_tensor(
        p_v_raw, cute.make_layout((C, D, STAGES), stride=(D, 1, STAGE_ELTS))
    )
    p_o = cute.recast_ptr(s_out.iterator, lay_qd.inner, dtype=cutlass.BFloat16)
    v_o = cute.make_tensor(
        p_o, cute.make_layout((C, 64, 2, 2), stride=(64, 1, 2048, C * D))
    )

    # ---- mbarrier init ----
    mb = storage.mbar.data_ptr()
    if warp == 0:
        with cute.arch.elect_one():
            for i in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(mb + MB_QK + i, 1)
                cute.arch.mbarrier_init(mb + MB_GRAW + i, 1)
                cute.arch.mbarrier_init(mb + MB_QKRAW + i, 1)
                cute.arch.mbarrier_init(mb + MB_VFULL + i, 1)
                cute.arch.mbarrier_init(mb + MB_VFREE + i, 4)
                cute.arch.mbarrier_init(mb + MB_SFREE + i, 1)
                cute.arch.mbarrier_init(mb + MB_RFREE + i, 1)
                cute.arch.mbarrier_init(mb + MB_OOUT + i, 1)
                cute.arch.mbarrier_init(mb + MB_U2ACC + i, 1)
                cute.arch.mbarrier_init(mb + MB_OFIN + i, 1)
                cute.arch.mbarrier_init(mb + MB_TAB + i, 1)
                cute.arch.mbarrier_init(mb + MB_FIN + i, 1)
                cute.arch.mbarrier_init(mb + MB_BRAW + i, 1)
                cute.arch.mbarrier_init(mb + MB_GRAM + i, 1)
            cute.arch.mbarrier_init(mb + MB_INP, 8)
            cute.arch.mbarrier_init(mb + MB_OEMPTY, 4)
        cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    alloc_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=alloc_bar)
    tmem.allocate(TMEM_COLS)
    tmem.wait_for_alloc()
    tmem.relinquish_alloc_permit()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Int32)

    # ---- TMEM tensors ----
    thr1 = mma_1.get_slice(0)
    thr3 = mma_3.get_slice(0)
    thr4a = mma_4a.get_slice(0)
    thr4b = mma_4b.get_slice(0)
    inp_base = thr1.make_fragment_A(mma_1.partition_shape_A((BM, D)))
    t_inp = cute.make_tensor(
        cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16), inp_base.layout
    )
    ri3_base = thr3.make_fragment_A(mma_3.partition_shape_A((BM, C)))
    # dv resid parks in the U window (f32 cols 224-239): U is dead after
    # the dv leg's T2R, and MMA4's junk pad overwrites it only after MMA3
    # consumed it (in-order pipe).  Keeping it out of INP cols 0-63 lets
    # the U-first MMA1 split commit MB_OOUT while the OUT group still
    # reads INP (the dv STTM no longer races that read).
    t_ri3 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 224, dtype=cutlass.BFloat16), ri3_base.layout
    )
    ri4_base = thr4a.make_fragment_A(mma_4a.partition_shape_A((BM, C)))
    t_ri4 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16), ri4_base.layout
    )
    acc32_base = thr1.make_fragment_C(mma_1.partition_shape_C((BM, C)))
    t_vst = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 32, dtype=cutlass.Float32), acc32_base.layout
    )
    t_out = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 192, dtype=cutlass.Float32), acc32_base.layout
    )
    t_u = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 224, dtype=cutlass.Float32), acc32_base.layout
    )

    d4a_base = thr4a.make_fragment_C(mma_4a.partition_shape_C((BM, D)))
    t_d4a = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 64, dtype=cutlass.Float32), d4a_base.layout
    )
    d4b_base = thr4b.make_fragment_C(mma_4b.partition_shape_C((BM, C)))
    t_d4b = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 192, dtype=cutlass.Float32), d4b_base.layout
    )
    mst_base = acc32_base  # [128,32] column-window template
    # bf16 window template for INP stores ([128,32] bf16 = 16 words)
    w32b_base = thr1.make_fragment_A(mma_1.partition_shape_A((BM, 32)))

    ld32_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32
    )
    st32_atom = cute.make_copy_atom(
        tcgen05.St32x32bOp(tcgen05.Repetition.x32), cutlass.Float32
    )
    stwb_atom = cute.make_copy_atom(
        tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.BFloat16
    )
    mst0 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 64, dtype=cutlass.Float32), mst_base.layout
    )
    mst_ld = tcgen05.make_tmem_copy(ld32_atom, mst0)
    mst_st = tcgen05.make_tmem_copy(st32_atom, mst0)
    u_ld = mst_ld
    twb0 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16), w32b_base.layout
    )
    w16b_st = tcgen05.make_tmem_copy(stwb_atom, twb0)
    u2f = cute.make_tensor(
        t_u.iterator, cute.make_layout(((BM, 1), (C, 1)), stride=((65536, 0), (1, 0)))
    )
    u2b = cute.make_tensor(
        twb0.iterator,
        cute.make_layout(((BM, 1), (16, 2)), stride=((131072, 0), (1, 16))),
    )
    u_wide_ld = tcgen05.make_tmem_copy(ld32_atom, u2f)
    u_wide_st = tcgen05.make_tmem_copy(stwb_atom, u2b)
    v_vt = cute.make_tensor(p_v_wide, lay_v.outer)
    state_inp_bar = pipeline.NamedBarrier(barrier_id=3, num_threads=288)
    u_inp_bar = pipeline.NamedBarrier(barrier_id=4, num_threads=160)
    u2_inp_bar = pipeline.NamedBarrier(barrier_id=5, num_threads=160)
    u2_inp_lo_bar = pipeline.NamedBarrier(barrier_id=12, num_threads=160)
    # half-window ([128,16] channels) bf16 INP store template for wg2
    w8b_base = thr1.make_fragment_A(mma_1.partition_shape_A((BM, 16)))
    stwb8_atom = cute.make_copy_atom(
        tcgen05.St32x32bOp(tcgen05.Repetition.x8), cutlass.BFloat16
    )
    twb8 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16), w8b_base.layout
    )
    w8b_st = tcgen05.make_tmem_copy(stwb8_atom, twb8)

    # =====================================================================
    # COMPUTE warpgroup (warps 0-3)
    # =====================================================================
    if warp < 4:
        if cutlass.const_expr(REG_MODE_ == 1):
            cute.arch.warpgroup_reg_alloc(160)
        else:
            cute.arch.warpgroup_reg_alloc(152)
        mst_thr = mst_ld.get_slice(tidx)
        mst_sthr = mst_st.get_slice(tidx)
        mst_id = mst_thr.partition_D(
            thr1.partition_C(cute.make_identity_tensor((BM, C)))
        )
        r_mst = cute.make_rmem_tensor(mst_id.shape, cutlass.Float32)
        uw_ldthr = u_wide_ld.get_slice(tidx)
        uw_uid = uw_ldthr.partition_D(cute.make_identity_tensor((BM, C)))
        uw_v = uw_ldthr.partition_D(v_vt[None, 0, None, None])
        r_u_wide = cute.make_tensor(r_mst.iterator, cute.make_layout(uw_uid.shape))
        r_u_old = r_mst
        r_tab = cute.make_rmem_tensor(mst_id.shape, cutlass.Float32)
        r_gt = r_tab
        r_bt_f_wide = cute.make_tensor(r_tab.iterator, cute.make_layout(uw_uid.shape))
        r_bt_f_old = r_tab
        # bf16 INP staging fragment stored via bf16-typed St32x32b.x16
        # (no rmem pointer recast: O3 miscompiles aliased register views)
        w16b_thr = w16b_st.get_slice(tidx)
        w16b_id = w16b_thr.partition_S(
            thr1.partition_A(cute.make_identity_tensor((BM, 32)))
        )
        rb32f = cute.make_rmem_tensor(w16b_id.shape, cutlass.BFloat16)
        r_mst2 = cute.make_rmem_tensor(mst_id.shape, cutlass.Float32)
        uw_sthr = u_wide_st.get_slice(tidx)
        uw_sid = uw_sthr.partition_S(cute.make_identity_tensor((BM, C)))
        rb_wide = cute.make_tensor(rb32f.iterator, cute.make_layout(uw_sid.shape))
        # Beta is dead as soon as the residual is formed, so its BF16
        # prefetch can live in the residual destination fragment itself.
        # This avoids introducing another register frame.
        r_bt_b_wide = rb_wide
        r_bt_b_old = cute.make_tensor(rb32f.iterator, cute.make_layout(mst_id.shape))
        r_vv_old = cute.make_rmem_tensor(mst_id.shape, cutlass.BFloat16)
        r_vv_wide = cute.make_tensor(r_vv_old.iterator, cute.make_layout(uw_sid.shape))
        mma_hc = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                cutlass.BFloat16,
                cutlass.Float32,
                (BM, 16, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.TMEM,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
            )
        )
        thr_hc = mma_hc.get_slice(0)
        h16c_base = thr_hc.make_fragment_C(mma_hc.partition_shape_C((BM, 16)))
        ld16c_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), cutlass.Float32
        )
        h0c = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + 32, dtype=cutlass.Float32), h16c_base.layout
        )
        h16c_ld = tcgen05.make_tmem_copy(ld16c_atom, h0c)
        h16c_thr = h16c_ld.get_slice(tidx)
        h16c_id = h16c_thr.partition_D(
            thr_hc.partition_C(cute.make_identity_tensor((BM, 16)))
        )
        r_u2h = cute.make_tensor(r_mst.iterator, cute.make_layout(h16c_id.shape))
        w8b_thrc = w8b_st.get_slice(tidx)
        w8b_idc = w8b_thrc.partition_S(
            thr1.partition_A(cute.make_identity_tensor((BM, 16)))
        )
        rb_u2h = cute.make_tensor(rb32f.iterator, cute.make_layout(w8b_idc.shape))

        cbar = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        lower_bound_log2 = lb2
        if cutlass.const_expr(LOWER_BOUND_M5_ == 1):
            lower_bound_log2 = -7.213475204444817
        deanchor = _exp2f(lower_bound_log2 * 16.0)
        p_oe = cutlass.Int32(1)
        p_inp = cutlass.Int32(0)
        csc = cutlass.Int32(0)
        p_qk = cutlass.Int32(0)
        p_vf = cutlass.Int32(0)
        p_oo = cutlass.Int32(0)
        p_u2a = cutlass.Int32(0)
        p_fin = cutlass.Int32(0)
        for kx in cutlass.range(k1 - k0):
            chain = cute.arch.make_warp_uniform(schain[k0 + kx])
            seq_idx = chain // H_
            hidx = chain % H_
            # piece-of-chain descriptor (v96 split): token window + state
            # routing.  Whole chains: pt0=0, ptn=seq len, src=dst=-1.
            bos = cutlass.Int32(cu[seq_idx]) + cute.arch.make_warp_uniform(
                spt0[k0 + kx]
            )
            seq_len = cute.arch.make_warp_uniform(sptn[k0 + kx])
            simp = cute.arch.make_warp_uniform(ssrc[k0 + kx])
            sexp = cute.arch.make_warp_uniform(sdst[k0 + kx])
            t_tiles = (seq_len + C - 1) // C
            if cutlass.const_expr(TPROBE_ == 1):
                if warp == 0:
                    with cute.arch.elect_one():
                        tprobe[slot * 64 + kx * 2] = cute.arch.globaltimer()
            # mid-chain seed: wait for the producer piece's full-count
            # release (256 per-thread arrivals x fepoch; B300 hazard memo:
            # EVERY consuming thread acquire-polls, per-thread releases)
            if cutlass.const_expr(SPLIT_ == 1):
                if simp >= 0:
                    ftgt = fepoch * 256
                    fcur = cute.arch.atomic_add(
                        mflags.iterator + simp,
                        cutlass.Int32(0),
                        sem="acquire",
                        scope="gpu",
                    )
                    while fcur < ftgt:
                        fcur = cute.arch.atomic_add(
                            mflags.iterator + simp,
                            cutlass.Int32(0),
                            sem="acquire",
                            scope="gpu",
                        )
            # epilogue WG seeds + decays master windows 2-3
            for win in cutlass.range_constexpr(2):
                msth = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 64 + win * 32, dtype=cutlass.Float32),
                    mst_base.layout,
                )
                if cutlass.const_expr((SPLIT_ == 1) or (WARM_SPLIT_ == 1)):
                    if simp >= 0:
                        vv0m = mst_id[0][0]
                        for jj in cutlass.range_constexpr(4):
                            r8i = cute.make_tensor(
                                r_mst.iterator + 8 * jj, cute.make_layout((8,))
                            )
                            g8i = cute.local_tile(
                                midstate, (1, 1, 8), (simp, vv0m, win * 4 + jj)
                            )
                            if cutlass.const_expr(
                                midstate.element_type is cutlass.BFloat16
                            ):
                                for ee in cutlass.range_constexpr(8):
                                    r8i[ee] = cutlass.Float32(g8i[ee])
                            else:
                                _load_state_f32x8(g8i, r8i)
                    else:
                        if cutlass.const_expr(STATE_ == 1):
                            if simp == -1:
                                vv0 = mst_id[0][0]
                                for jj in cutlass.range_constexpr(4):
                                    r8i = cute.make_tensor(
                                        r_mst.iterator + 8 * jj, cute.make_layout((8,))
                                    )
                                    g8i = cute.local_tile(
                                        state0,
                                        (1, 1, 8),
                                        (chain, vv0, win * 4 + jj),
                                    )
                                    _load_state_f32x8(g8i, r8i)
                            else:
                                for e in cutlass.range_constexpr(cute.size(r_mst)):
                                    r_mst[e] = cutlass.Float32(0.0)
                        else:
                            for e in cutlass.range_constexpr(cute.size(r_mst)):
                                r_mst[e] = cutlass.Float32(0.0)
                else:
                    if cutlass.const_expr(STATE_ == 1):
                        vv0 = mst_id[0][0]
                        for jj in cutlass.range_constexpr(4):
                            r8i = cute.make_tensor(
                                r_mst.iterator + 8 * jj, cute.make_layout((8,))
                            )
                            g8i = cute.local_tile(
                                state0, (1, 1, 8), (chain, vv0, win * 4 + jj)
                            )
                            _load_state_f32x8(g8i, r8i)
                    else:
                        for e in cutlass.range_constexpr(cute.size(r_mst)):
                            r_mst[e] = cutlass.Float32(0.0)
                cute.copy(mst_st, r_mst, mst_sthr.partition_D(msth))
            cute.arch.fence_view_async_tmem_store()
            for t in cutlass.range(t_tiles):  # noqa: B007
                cute.arch.mbarrier_wait(mb + MB_QK + csc, p_qk)
                m0 = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 64, dtype=cutlass.Float32),
                    mst_base.layout,
                )
                cute.copy(mst_ld, mst_thr.partition_S(m0), r_mst)
                for win in cutlass.range_constexpr(2):
                    rcur = r_mst if win % 2 == 0 else r_mst2
                    rnxt = r_mst2 if win % 2 == 0 else r_mst
                    msth = cute.make_tensor(
                        cute.recast_ptr(
                            tmem_ptr + 64 + win * 32, dtype=cutlass.Float32
                        ),
                        mst_base.layout,
                    )
                    twin = cute.make_tensor(
                        cute.recast_ptr(tmem_ptr + win * 16, dtype=cutlass.BFloat16),
                        w32b_base.layout,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(win < 1):
                        mnxt = cute.make_tensor(
                            cute.recast_ptr(
                                tmem_ptr + 64 + (win + 1) * 32, dtype=cutlass.Float32
                            ),
                            mst_base.layout,
                        )
                        cute.copy(mst_ld, mst_thr.partition_S(mnxt), rnxt)
                    for gv in cutlass.range_constexpr(4):
                        g8d = cute.make_tensor(
                            r_gt.iterator + 8 * gv, cute.make_layout((8,))
                        )
                        cute.autovec_copy(v_gt8[(win * 4 + gv, None, csc)], g8d)
                    mv = rcur.load()
                    rb32f.store(mv.to(cutlass.BFloat16))
                    rcur.store(mv * r_gt.load())
                    cute.copy(w16b_st, rb32f, w16b_thr.partition_D(twin))
                    cute.copy(mst_st, rcur, mst_sthr.partition_D(msth))
                cute.arch.fence_view_async_tmem_store()
                state_inp_bar.arrive()
                p_oe ^= 1
                p_inp ^= 1

                # QK-ready (waited at the top of this iteration) also makes
                # beta stable.  Prefetch it while V/O are still in flight.
                if cutlass.const_expr(GATE2_ == 1):
                    if cutlass.const_expr(BETA_BF16_ == 1):
                        for ee in cutlass.range_constexpr(cute.size(uw_uid)):
                            r_bt_b_wide[ee] = v_btb[uw_uid[ee][1], csc]
                    else:
                        for ee in cutlass.range_constexpr(cute.size(uw_uid)):
                            r_bt_f_wide[ee] = v_bt[uw_uid[ee][1], csc]
                else:
                    for gv2 in cutlass.range_constexpr(4):
                        if cutlass.const_expr(BETA_BF16_ == 1):
                            b8d = cute.make_tensor(
                                r_bt_b_old.iterator + 8 * gv2, cute.make_layout((8,))
                            )
                            cute.autovec_copy(v_btb8[(gv2, None, csc)], b8d)
                        else:
                            b8d = cute.make_tensor(
                                r_bt_f_old.iterator + 8 * gv2, cute.make_layout((8,))
                            )
                            cute.autovec_copy(v_bt8[(gv2, None, csc)], b8d)
                cute.arch.mbarrier_wait(mb + MB_VFULL + csc, p_vf)
                cute.arch.mbarrier_wait(mb + MB_OOUT + csc, p_oo)
                twri = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16), w32b_base.layout
                )
                twri_dv = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 224, dtype=cutlass.BFloat16),
                    w32b_base.layout,
                )
                if cutlass.const_expr(GATE2_ == 1):
                    cute.copy(u_wide_ld, uw_ldthr.partition_S(u2f), r_u_wide)
                    # Wide transformed views preserve the same logical
                    # (feature, token) ordering while vectorizing V loads.
                    cute.autovec_copy(uw_v[None, None, None, csc], r_vv_wide)
                else:
                    cute.copy(mst_ld, mst_thr.partition_S(t_u), r_u_old)
                    vv1 = mst_id[0][0]
                    for tt in cutlass.range_constexpr(C):
                        r_vv_old[tt] = v_v[tt, vv1, csc]
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(GATE2_ == 1):
                    if cutlass.const_expr(BF16_DV_ == 1):
                        # The residual is immediately consumed as a BF16
                        # MMA3 operand.  Test forming the chain in packed
                        # BF16 against the original FP32 implementation.
                        if cutlass.const_expr(BETA_BF16_ == 1):
                            dvvb = (
                                r_vv_wide.load()
                                - (r_u_wide.load() * deanchor).to(cutlass.BFloat16)
                            ) * r_bt_b_wide.load()
                        else:
                            dvvb = (
                                r_vv_wide.load()
                                - (r_u_wide.load() * deanchor).to(cutlass.BFloat16)
                            ) * r_bt_f_wide.load().to(cutlass.BFloat16)
                        rb_wide.store(dvvb)
                    else:
                        if cutlass.const_expr(BETA_BF16_ == 1):
                            dvv = (
                                r_vv_wide.load().to(cutlass.Float32)
                                - r_u_wide.load() * deanchor
                            ) * r_bt_b_wide.load().to(cutlass.Float32)
                        else:
                            dvv = (
                                r_vv_wide.load().to(cutlass.Float32)
                                - r_u_wide.load() * deanchor
                            ) * r_bt_f_wide.load()
                        rb_wide.store(dvv.to(cutlass.BFloat16))
                    twri_dv_w = cute.make_tensor(twri_dv.iterator, u2b.layout)
                    cute.copy(u_wide_st, rb_wide, uw_sthr.partition_D(twri_dv_w))
                else:
                    if cutlass.const_expr(BF16_DV_ == 1):
                        if cutlass.const_expr(BETA_BF16_ == 1):
                            dvvb = (
                                r_vv_old.load()
                                - (r_u_old.load() * deanchor).to(cutlass.BFloat16)
                            ) * r_bt_b_old.load()
                        else:
                            dvvb = (
                                r_vv_old.load()
                                - (r_u_old.load() * deanchor).to(cutlass.BFloat16)
                            ) * r_bt_f_old.load().to(cutlass.BFloat16)
                        rb32f.store(dvvb)
                    else:
                        if cutlass.const_expr(BETA_BF16_ == 1):
                            dvv = (
                                r_vv_old.load().to(cutlass.Float32)
                                - r_u_old.load() * deanchor
                            ) * r_bt_b_old.load().to(cutlass.Float32)
                        else:
                            dvv = (
                                r_vv_old.load().to(cutlass.Float32)
                                - r_u_old.load() * deanchor
                            ) * r_bt_f_old.load()
                        rb32f.store(dvv.to(cutlass.BFloat16))
                    cute.copy(w16b_st, rb32f, w16b_thr.partition_D(twri_dv))
                cute.arch.fence_view_async_tmem_store()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mb + MB_VFREE + csc)
                u_inp_bar.arrive()

                cute.arch.mbarrier_wait(mb + MB_U2ACC + csc, p_u2a)
                if cutlass.const_expr(GATE2_ == 1):
                    for half in cutlass.range_constexpr(2):
                        t_vst_h = cute.make_tensor(
                            cute.recast_ptr(
                                tmem_ptr + 32 + half * 16, dtype=cutlass.Float32
                            ),
                            h16c_base.layout,
                        )
                        cute.copy(h16c_ld, h16c_thr.partition_S(t_vst_h), r_u2h)
                        cute.arch.fence_view_async_tmem_load()
                        rb_u2h.store(r_u2h.load().to(cutlass.BFloat16))
                        twri_h = cute.make_tensor(
                            cute.recast_ptr(
                                tmem_ptr + half * 8, dtype=cutlass.BFloat16
                            ),
                            w8b_base.layout,
                        )
                        cute.copy(w8b_st, rb_u2h, w8b_thrc.partition_D(twri_h))
                        cute.arch.fence_view_async_tmem_store()
                        if cutlass.const_expr(half == 0):
                            u2_inp_lo_bar.arrive()
                        else:
                            u2_inp_bar.arrive()
                else:
                    cute.copy(mst_ld, mst_thr.partition_S(t_vst), r_u_old)
                    cute.arch.fence_view_async_tmem_load()
                    rb32f.store(r_u_old.load().to(cutlass.BFloat16))
                    cute.copy(w16b_st, rb32f, w16b_thr.partition_D(twri))
                    cute.arch.fence_view_async_tmem_store()
                    u2_inp_bar.arrive()

                cute.arch.mbarrier_wait(mb + MB_FIN + csc, p_fin)
                csc += 1
                if csc == STAGES:
                    csc = 0
                    p_qk ^= 1
                    p_vf ^= 1
                    p_oo ^= 1
                    p_u2a ^= 1
                    p_fin ^= 1

            # State export: split producers always write the midstate ring;
            # terminal stateT writes compile out for this no-final task.
            if cutlass.const_expr((SPLIT_ == 1) or (FINAL_ == 1)):
                for win in cutlass.range_constexpr(2):
                    msth = cute.make_tensor(
                        cute.recast_ptr(
                            tmem_ptr + 64 + win * 32, dtype=cutlass.Float32
                        ),
                        mst_base.layout,
                    )
                    cute.copy(mst_ld, mst_thr.partition_S(msth), r_mst)
                    cute.arch.fence_view_async_tmem_load()
                    vv2 = mst_id[0][0]
                    for jj in cutlass.range_constexpr(4):
                        r8o = cute.make_tensor(
                            r_mst.iterator + 8 * jj, cute.make_layout((8,))
                        )
                        if cutlass.const_expr(SPLIT_ == 1):
                            if sexp >= 0:
                                g8m = cute.local_tile(
                                    midstate, (1, 1, 8), (sexp, vv2, win * 4 + jj)
                                )
                                if cutlass.const_expr(
                                    midstate.element_type is cutlass.BFloat16
                                ):
                                    for ee in cutlass.range_constexpr(8):
                                        g8m[ee] = r8o[ee].to(cutlass.BFloat16)
                                else:
                                    cute.autovec_copy(r8o, g8m)
                            else:
                                if cutlass.const_expr(FINAL_ == 1):
                                    if sexp == -1:
                                        g8o = cute.local_tile(
                                            stateT,
                                            (1, 1, 8),
                                            (chain, vv2, win * 4 + jj),
                                        )
                                        _store_final_f32x8(g8o, r8o)
                        else:
                            if cutlass.const_expr(WARM_SPLIT_ == 1):
                                if sexp == -1:
                                    g8o = cute.local_tile(
                                        stateT,
                                        (1, 1, 8),
                                        (chain, vv2, win * 4 + jj),
                                    )
                                    _store_final_f32x8(g8o, r8o)
                            else:
                                g8o = cute.local_tile(
                                    stateT, (1, 1, 8), (chain, vv2, win * 4 + jj)
                                )
                                _store_final_f32x8(g8o, r8o)
            if cutlass.const_expr(SPLIT_ == 1):
                if sexp >= 0:
                    cute.arch.atomic_add(
                        mflags.iterator + sexp,
                        cutlass.Int32(1),
                        sem="release",
                        scope="gpu",
                    )
            if cutlass.const_expr(TPROBE_ == 1):
                if warp == 0:
                    with cute.arch.elect_one():
                        tprobe[slot * 64 + kx * 2 + 1] = cute.arch.globaltimer()

    # =====================================================================
    # EPILOGUE warpgroup (warps 4-7)
    # =====================================================================
    if (warp >= 4) & (warp < 8):
        if cutlass.const_expr(REG_MODE_ == 2):
            cute.arch.warpgroup_reg_alloc(96)
        else:
            cute.arch.warpgroup_reg_alloc(88)
        etx = tidx - 128
        o16_atom = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x4), cutlass.Float32
        )
        o16_cp = tcgen05.make_tmem_copy(o16_atom, t_out)
        out_thr = o16_cp.get_slice(etx)
        o_id = out_thr.partition_D(thr1.partition_C(cute.make_identity_tensor((BM, C))))
        r_o = cute.make_rmem_tensor(o_id.shape, cutlass.Float32)
        r_ob = cute.make_rmem_tensor(o_id.shape, cutlass.BFloat16)
        stm_np = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(True, 4), cutlass.BFloat16
        )
        ep_bar = pipeline.NamedBarrier(barrier_id=6, num_threads=128)
        # stmatrix.x4.trans lane addressing (PR m128 formula; the XOR
        # is Sw(3,4,3) in element space, matching the l3k TMA box)
        elw = etx >> 5
        lne = etx & 31
        mtx = lne >> 3
        row8 = lne & 7
        so_base = cutlass.Int32(s_out.iterator.toint())

        # ---- decay-help machinery: [128,16] half-window TMEM copies
        # (mma_h exists only to mint the [BM,16] f32 accumulator layout)
        mma_h = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                cutlass.BFloat16,
                cutlass.Float32,
                (BM, 16, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.TMEM,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
            )
        )
        thr_h = mma_h.get_slice(0)
        h16_base = thr_h.make_fragment_C(mma_h.partition_shape_C((BM, 16)))
        ld16_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), cutlass.Float32
        )
        st16_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Float32
        )
        h0t = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + 128, dtype=cutlass.Float32), h16_base.layout
        )
        h_ld = tcgen05.make_tmem_copy(ld16_atom, h0t)
        h_st = tcgen05.make_tmem_copy(st16_atom, h0t)
        h_thr = h_ld.get_slice(etx)
        h_sthr = h_st.get_slice(etx)
        h_id = h_thr.partition_D(thr_h.partition_C(cute.make_identity_tensor((BM, 16))))
        r_e1 = cute.make_rmem_tensor(h_id.shape, cutlass.Float32)
        r_e2 = cute.make_rmem_tensor(h_id.shape, cutlass.Float32)
        gt16 = cute.make_rmem_tensor(h_id.shape, cutlass.Float32)
        w8b_thr = w8b_st.get_slice(etx)
        w8b_id = w8b_thr.partition_S(
            thr1.partition_A(cute.make_identity_tensor((BM, 16)))
        )
        rbh = cute.make_tensor(r_ob.iterator, cute.make_layout(w8b_id.shape))

        ping = cutlass.Int32(0)
        lower_bound_log2 = lb2
        if cutlass.const_expr(LOWER_BOUND_M5_ == 1):
            lower_bound_log2 = -7.213475204444817
        out_deanchor = _exp2f(lower_bound_log2 * 16.0)
        cse = cutlass.Int32(0)
        pe_state = cutlass.Int32(0)
        pe_fin = cutlass.Int32(0)
        csn = cutlass.Int32(0)
        pqn = cutlass.Int32(0)
        for kx in cutlass.range(k1 - k0):
            chain = cute.arch.make_warp_uniform(schain[k0 + kx])
            seq_idx = chain // H_
            hidx = chain % H_
            bos = cutlass.Int32(cu[seq_idx]) + cute.arch.make_warp_uniform(
                spt0[k0 + kx]
            )
            seq_len = cute.arch.make_warp_uniform(sptn[k0 + kx])
            simp = cute.arch.make_warp_uniform(ssrc[k0 + kx])
            sexp = cute.arch.make_warp_uniform(sdst[k0 + kx])
            t_tiles = (seq_len + C - 1) // C
            warm_chunks = cutlass.max(-simp - 1, 0)
            gO = cute.flat_divide(cute.domain_offset((bos, 0, 0), mO), (C, 1, D))
            if cutlass.const_expr(SPLIT_ == 1):
                if simp >= 0:
                    ftgt = fepoch * 256
                    fcur = cute.arch.atomic_add(
                        mflags.iterator + simp,
                        cutlass.Int32(0),
                        sem="acquire",
                        scope="gpu",
                    )
                    while fcur < ftgt:
                        fcur = cute.arch.atomic_add(
                            mflags.iterator + simp,
                            cutlass.Int32(0),
                            sem="acquire",
                            scope="gpu",
                        )
            # seed master windows 2-3 (four 16-ch halves; compute seeds 0-1)
            for hh in cutlass.range_constexpr(4):
                hten = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 128 + hh * 16, dtype=cutlass.Float32),
                    h16_base.layout,
                )
                if cutlass.const_expr((SPLIT_ == 1) or (WARM_SPLIT_ == 1)):
                    if simp >= 0:
                        vv0m = h_id[0][0]
                        for jj in cutlass.range_constexpr(2):
                            r8i = cute.make_tensor(
                                r_e1.iterator + 8 * jj, cute.make_layout((8,))
                            )
                            g8i = cute.local_tile(
                                midstate, (1, 1, 8), (simp, vv0m, 8 + hh * 2 + jj)
                            )
                            if cutlass.const_expr(
                                midstate.element_type is cutlass.BFloat16
                            ):
                                for ee in cutlass.range_constexpr(8):
                                    r8i[ee] = cutlass.Float32(g8i[ee])
                            else:
                                _load_state_f32x8(g8i, r8i)
                    else:
                        if cutlass.const_expr(STATE_ == 1):
                            if simp == -1:
                                vv0 = h_id[0][0]
                                for jj in cutlass.range_constexpr(2):
                                    r8i = cute.make_tensor(
                                        r_e1.iterator + 8 * jj,
                                        cute.make_layout((8,)),
                                    )
                                    g8i = cute.local_tile(
                                        state0,
                                        (1, 1, 8),
                                        (chain, vv0, 8 + hh * 2 + jj),
                                    )
                                    _load_state_f32x8(g8i, r8i)
                            else:
                                for e in cutlass.range_constexpr(cute.size(r_e1)):
                                    r_e1[e] = cutlass.Float32(0.0)
                        else:
                            for e in cutlass.range_constexpr(cute.size(r_e1)):
                                r_e1[e] = cutlass.Float32(0.0)
                else:
                    if cutlass.const_expr(STATE_ == 1):
                        vv0 = h_id[0][0]
                        for jj in cutlass.range_constexpr(2):
                            r8i = cute.make_tensor(
                                r_e1.iterator + 8 * jj, cute.make_layout((8,))
                            )
                            g8i = cute.local_tile(
                                state0, (1, 1, 8), (chain, vv0, 8 + hh * 2 + jj)
                            )
                            _load_state_f32x8(g8i, r8i)
                    else:
                        for e in cutlass.range_constexpr(cute.size(r_e1)):
                            r_e1[e] = cutlass.Float32(0.0)
                cute.copy(h_st, r_e1, h_sthr.partition_D(hten))
            cute.arch.fence_view_async_tmem_store()

            # chunk-0 decay help (running stage slot)
            cute.arch.mbarrier_wait(mb + MB_QK + csn, pqn)
            h0p = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + 128, dtype=cutlass.Float32), h16_base.layout
            )
            cute.copy(h_ld, h_thr.partition_S(h0p), r_e1)
            for hh in cutlass.range_constexpr(4):
                rcur = r_e1 if hh % 2 == 0 else r_e2
                rnxt = r_e2 if hh % 2 == 0 else r_e1
                hmst = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 128 + hh * 16, dtype=cutlass.Float32),
                    h16_base.layout,
                )
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(hh < 3):
                    hnxt = cute.make_tensor(
                        cute.recast_ptr(
                            tmem_ptr + 128 + (hh + 1) * 16, dtype=cutlass.Float32
                        ),
                        h16_base.layout,
                    )
                    cute.copy(h_ld, h_thr.partition_S(hnxt), rnxt)
                for gv in cutlass.range_constexpr(2):
                    g8d = cute.make_tensor(
                        gt16.iterator + 8 * gv, cute.make_layout((8,))
                    )
                    cute.autovec_copy(v_gt8[(8 + hh * 2 + gv, None, csn)], g8d)
                mv = rcur.load()
                rbh.store(mv.to(cutlass.BFloat16))
                rcur.store(mv * gt16.load())
                tw8 = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 32 + hh * 8, dtype=cutlass.BFloat16),
                    w8b_base.layout,
                )
                cute.copy(w8b_st, rbh, w8b_thr.partition_D(tw8))
                cute.copy(h_st, rcur, h_sthr.partition_D(hmst))
            cute.arch.fence_view_async_tmem_store()
            state_inp_bar.arrive()
            csn += 1
            if csn == STAGES:
                csn = 0
                pqn ^= 1

            n_full_tiles = seq_len // C
            for tail_phase in cutlass.range_constexpr(1 if FULL_CHUNKS_ == 1 else 2):
                phase_tiles = (
                    n_full_tiles if tail_phase == 0 else t_tiles - n_full_tiles
                )
                phase_base = 0 if tail_phase == 0 else n_full_tiles
                full = cutlass.const_expr(FULL_CHUNKS_ == 1 or tail_phase == 0)
                for phase_t in cutlass.range(phase_tiles):
                    t = phase_base + phase_t
                    cc_t = seq_len - t * C
                    if cutlass.const_expr(GATE2_ == 0):
                        cute.arch.mbarrier_wait(mb + MB_OFIN + cse, pe_fin)
                    if cutlass.const_expr(True):
                        if t + 1 < t_tiles:
                            if cutlass.const_expr(GATE2_ == 1):
                                cute.arch.mbarrier_wait(mb + MB_FIN + cse, pe_state)
                            cute.arch.mbarrier_wait(mb + MB_QK + csn, pqn)
                            h0q = cute.make_tensor(
                                cute.recast_ptr(tmem_ptr + 128, dtype=cutlass.Float32),
                                h16_base.layout,
                            )
                            cute.copy(h_ld, h_thr.partition_S(h0q), r_e1)
                            for hh in cutlass.range_constexpr(4):
                                rcur = r_e1 if hh % 2 == 0 else r_e2
                                rnxt = r_e2 if hh % 2 == 0 else r_e1
                                hmst = cute.make_tensor(
                                    cute.recast_ptr(
                                        tmem_ptr + 128 + hh * 16,
                                        dtype=cutlass.Float32,
                                    ),
                                    h16_base.layout,
                                )
                                cute.arch.fence_view_async_tmem_load()
                                if cutlass.const_expr(hh < 3):
                                    hnxt = cute.make_tensor(
                                        cute.recast_ptr(
                                            tmem_ptr + 128 + (hh + 1) * 16,
                                            dtype=cutlass.Float32,
                                        ),
                                        h16_base.layout,
                                    )
                                    cute.copy(h_ld, h_thr.partition_S(hnxt), rnxt)
                                for gv in cutlass.range_constexpr(2):
                                    g8d = cute.make_tensor(
                                        gt16.iterator + 8 * gv, cute.make_layout((8,))
                                    )
                                    cute.autovec_copy(
                                        v_gt8[(8 + hh * 2 + gv, None, csn)], g8d
                                    )
                                mv = rcur.load()
                                rbh.store(mv.to(cutlass.BFloat16))
                                rcur.store(mv * gt16.load())
                                tw8 = cute.make_tensor(
                                    cute.recast_ptr(
                                        tmem_ptr + 32 + hh * 8,
                                        dtype=cutlass.BFloat16,
                                    ),
                                    w8b_base.layout,
                                )
                                cute.copy(w8b_st, rbh, w8b_thr.partition_D(tw8))
                                cute.copy(h_st, rcur, h_sthr.partition_D(hmst))
                            cute.arch.fence_view_async_tmem_store()
                            state_inp_bar.arrive()
                            csn += 1
                            if csn == STAGES:
                                csn = 0
                                pqn ^= 1
                    if cutlass.const_expr(GATE2_ == 1):
                        cute.arch.mbarrier_wait(mb + MB_OFIN + cse, pe_fin)
                    cute.copy(o16_cp, out_thr.partition_S(t_out), r_o)
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mb + MB_OEMPTY)
                    if full:
                        if warp == 4:
                            # unconditional: stores may be in flight across
                            # chain boundaries; no-op when nothing pending
                            cute.arch.cp_async_bulk_wait_group(1, read=True)
                        ep_bar.arrive_and_wait()
                        # PTX-free stmatrix: DSL StMatrix atom over the
                        # same lane addresses (probe scripts/stm_probe.py
                        # = byte-identical to the inline-PTX block)
                        r_ob.store((r_o.load() * out_deanchor).to(cutlass.BFloat16))
                        for dh in cutlass.range_constexpr(2):
                            for tg in cutlass.range_constexpr(2):
                                dim_base = elw * 32 + dh * 16 + (mtx & 1) * 8
                                ta = tg * 16 + (mtx >> 1) * 8 + row8
                                tp = ta >> 1
                                par = ta & 1
                                raw_row = tp + (dim_base >> 6) * 16
                                raw_col = (
                                    (dim_base & 63) ^ ((tp & 3) << 4) ^ (par << 3)
                                ) + par * 64
                                addr = (
                                    so_base
                                    + ping * 8192
                                    + (raw_row * 128 + raw_col) * 2
                                )
                                e0 = dh * 16 + tg * 8
                                s8 = cute.make_tensor(
                                    r_ob.iterator + e0, cute.make_layout((8,))
                                )
                                dst = cute.make_tensor(
                                    cute.make_ptr(
                                        cutlass.BFloat16,
                                        addr,
                                        cute.AddressSpace.smem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout((8,)),
                                )
                                cute.copy(stm_np, s8, dst)
                        cute.arch.fence_proxy("async.shared", space="cta")
                        ep_bar.arrive_and_wait()
                        if warp == 4:
                            if t >= warm_chunks:
                                f_o = cute.make_tensor(
                                    p_o + ping * (C * D),
                                    cute.make_layout(
                                        (C, 1, (64, 2)), stride=(64, 0, (1, 2048))
                                    ),
                                )
                                o_s, o_g = cpasync.tma_partition(
                                    tma_o,
                                    0,
                                    cute.make_layout(1),
                                    cute.group_modes(f_o, 0, 3),
                                    cute.group_modes(gO, 0, 3),
                                )
                                cute.copy(tma_o, o_s, o_g[(None, t, hidx, 0)])
                                cute.arch.cp_async_bulk_commit_group()
                        ping ^= 1
                    else:
                        if cutlass.const_expr(H_ == 96):
                            if warp == 4:
                                cute.arch.cp_async_bulk_wait_group(1, read=True)
                            ep_bar.arrive_and_wait()
                            r_ob.store((r_o.load() * out_deanchor).to(cutlass.BFloat16))
                            for dh in cutlass.range_constexpr(2):
                                for tg in cutlass.range_constexpr(2):
                                    dim_base = elw * 32 + dh * 16 + (mtx & 1) * 8
                                    ta = tg * 16 + (mtx >> 1) * 8 + row8
                                    tp = ta >> 1
                                    par = ta & 1
                                    raw_row = tp + (dim_base >> 6) * 16
                                    raw_col = (
                                        (dim_base & 63) ^ ((tp & 3) << 4) ^ (par << 3)
                                    ) + par * 64
                                    addr = (
                                        so_base
                                        + ping * 8192
                                        + (raw_row * 128 + raw_col) * 2
                                    )
                                    e0 = dh * 16 + tg * 8
                                    s8 = cute.make_tensor(
                                        r_ob.iterator + e0, cute.make_layout((8,))
                                    )
                                    dst = cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.BFloat16,
                                            addr,
                                            cute.AddressSpace.smem,
                                            assumed_align=16,
                                        ),
                                        cute.make_layout((8,)),
                                    )
                                    cute.copy(stm_np, s8, dst)
                            cute.arch.fence_proxy("async.shared", space="cta")
                            ep_bar.arrive_and_wait()
                            for tail_pass in cutlass.range_constexpr(4):
                                tail_item = tail_pass * 128 + etx
                                tail_row = tail_item // 16
                                tail_seg = tail_item % 16
                                if (tail_row < cc_t) & (t >= warm_chunks):
                                    src8 = cute.local_tile(
                                        v_o,
                                        (1, 8, 1, 1),
                                        (tail_row, tail_seg & 7, tail_seg >> 3, ping),
                                    )
                                    src8 = cute.group_modes(src8, 0, 4)
                                    tail8 = cute.make_tensor(
                                        r_ob.iterator, cute.make_layout((8,))
                                    )
                                    cute.autovec_copy(src8, tail8)
                                    dst8 = cute.local_tile(
                                        out_raw,
                                        (1, 1, 8),
                                        (bos + t * C + tail_row, hidx, tail_seg),
                                    )
                                    cute.autovec_copy(
                                        tail8, cute.group_modes(dst8, 0, 3)
                                    )
                            ep_bar.arrive_and_wait()
                        else:
                            r_ob.store((r_o.load() * out_deanchor).to(cutlass.BFloat16))
                            for e in cutlass.range_constexpr(cute.size(r_o)):
                                tt2 = o_id[e][1]
                                if (tt2 < cc_t) & (t >= warm_chunks):
                                    out_raw[bos + t * C + tt2, hidx, o_id[e][0]] = r_ob[
                                        e
                                    ]
                    cse += 1
                    if cse == STAGES:
                        cse = 0
                        pe_state ^= 1
                        pe_fin ^= 1
            # Split-state windows 2-3; terminal stateT writes compile out.
            if cutlass.const_expr((SPLIT_ == 1) or (FINAL_ == 1)):
                for hh in cutlass.range_constexpr(4):
                    hten = cute.make_tensor(
                        cute.recast_ptr(
                            tmem_ptr + 128 + hh * 16, dtype=cutlass.Float32
                        ),
                        h16_base.layout,
                    )
                    cute.copy(h_ld, h_thr.partition_S(hten), r_e1)
                    cute.arch.fence_view_async_tmem_load()
                    vv2e = h_id[0][0]
                    for jj in cutlass.range_constexpr(2):
                        r8o = cute.make_tensor(
                            r_e1.iterator + 8 * jj, cute.make_layout((8,))
                        )
                        if cutlass.const_expr(SPLIT_ == 1):
                            if sexp >= 0:
                                g8m = cute.local_tile(
                                    midstate, (1, 1, 8), (sexp, vv2e, 8 + hh * 2 + jj)
                                )
                                if cutlass.const_expr(
                                    midstate.element_type is cutlass.BFloat16
                                ):
                                    for ee in cutlass.range_constexpr(8):
                                        g8m[ee] = r8o[ee].to(cutlass.BFloat16)
                                else:
                                    cute.autovec_copy(r8o, g8m)
                            else:
                                if cutlass.const_expr(FINAL_ == 1):
                                    if sexp == -1:
                                        g8o = cute.local_tile(
                                            stateT,
                                            (1, 1, 8),
                                            (chain, vv2e, 8 + hh * 2 + jj),
                                        )
                                        _store_final_f32x8(g8o, r8o)
                        else:
                            if cutlass.const_expr(WARM_SPLIT_ == 1):
                                if sexp == -1:
                                    g8o = cute.local_tile(
                                        stateT,
                                        (1, 1, 8),
                                        (chain, vv2e, 8 + hh * 2 + jj),
                                    )
                                    _store_final_f32x8(g8o, r8o)
                            else:
                                g8o = cute.local_tile(
                                    stateT, (1, 1, 8), (chain, vv2e, 8 + hh * 2 + jj)
                                )
                                _store_final_f32x8(g8o, r8o)
            if cutlass.const_expr(SPLIT_ == 1):
                if sexp >= 0:
                    cute.arch.atomic_add(
                        mflags.iterator + sexp,
                        cutlass.Int32(1),
                        sem="release",
                        scope="gpu",
                    )
        if warp == 4:
            cute.arch.cp_async_bulk_wait_group(0)

    # =====================================================================
    # WG2: MMA (warp 9), LOAD (warp 10), donors (8, 11)
    # =====================================================================
    if (warp >= 8) & (warp < 12):
        if cutlass.const_expr(REG_MODE_ != 0):
            cute.arch.warpgroup_reg_dealloc(24)
        else:
            cute.arch.warpgroup_reg_dealloc(32)
        if cutlass.const_expr(GRAM_W9_ != 0):
            if warp == 9:
                # The prep instances publish their complete Qd/Kd/Ki image
                # through MB_TAB.  Centralize the SS Gram issue on the idle
                # donor warp so the five plw0 warps no longer concentrate
                # every tcgen05 issue sequence on one SMSP.
                csg = cutlass.Int32(0)
                pg_tab = cutlass.Int32(0)
                thr_gd = mma_g.get_slice(0)
                gm_base_d = thr_gd.make_fragment_C(mma_g.partition_shape_C((2 * C, C)))
                p_g64d = cute.recast_ptr(tmem_ptr + GRAM_COL, dtype=cutlass.Float64)
                for kx in cutlass.range(k1 - k0):
                    chain = cute.arch.make_warp_uniform(schain[k0 + kx])
                    seq_idx = chain // H_
                    seq_len = cute.arch.make_warp_uniform(sptn[k0 + kx])
                    t_tiles = (seq_len + C - 1) // C
                    for _t in cutlass.range(t_tiles):
                        # Form the dynamic stage address while the donor is
                        # otherwise idle, before its publication wait.
                        p_gstage_d = p_g64d + csg * (C // 2)
                        cute.arch.mbarrier_wait(mb + MB_TAB + csg, pg_tab)
                        t_gram_d = cute.make_tensor(
                            cute.recast_ptr(p_gstage_d, dtype=cutlass.Float32),
                            gm_base_d.layout,
                        )
                        mma_g.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(D // 16):
                            cute.gemm(
                                mma_g,
                                t_gram_d,
                                a_qk[None, None, kb, csg],
                                b_ki[None, None, kb, csg],
                                t_gram_d,
                            )
                            mma_g.set(tcgen05.Field.ACCUMULATE, True)
                        with cute.arch.elect_one():
                            tcgen05.commit(mb + MB_GRAM + csg)
                        csg += 1
                        if csg == STAGES:
                            csg = 0
                            pg_tab ^= 1
        mma_warp = 9
        if cutlass.const_expr(GRAM_W9_ != 0):
            mma_warp = 8
        if warp == mma_warp:
            if cutlass.const_expr(True):
                csm = cutlass.Int32(0)
                pm_oe = cutlass.Int32(1)
                for kx in cutlass.range(k1 - k0):
                    chain = cute.arch.make_warp_uniform(schain[k0 + kx])
                    seq_idx = chain // H_
                    seq_len = cute.arch.make_warp_uniform(sptn[k0 + kx])
                    t_tiles = (seq_len + C - 1) // C
                    for _t in cutlass.range(t_tiles):
                        state_inp_bar.arrive_and_wait()
                        mma_1.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(D // 16):
                            cute.gemm(
                                mma_1,
                                t_u,
                                t_inp[None, None, kb],
                                b_kd[None, None, kb, csm],
                                t_u,
                            )
                            mma_1.set(tcgen05.Field.ACCUMULATE, True)
                        with cute.arch.elect_one():
                            tcgen05.commit(mb + MB_OOUT + csm)
                        cute.arch.mbarrier_wait(mb + MB_OEMPTY, pm_oe)
                        mma_1.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(D // 16):
                            cute.gemm(
                                mma_1,
                                t_out,
                                t_inp[None, None, kb],
                                b_qd[None, None, kb, csm],
                                t_out,
                            )
                            mma_1.set(tcgen05.Field.ACCUMULATE, True)
                        with cute.arch.elect_one():
                            tcgen05.commit(mb + MB_RFREE + csm)

                        u_inp_bar.arrive_and_wait()
                        mma_3.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(C // 16):
                            cute.gemm(
                                mma_3,
                                t_vst,
                                t_ri3[None, None, kb],
                                b_inv[None, None, kb, csm],
                                t_vst,
                            )
                            mma_3.set(tcgen05.Field.ACCUMULATE, True)
                        with cute.arch.elect_one():
                            tcgen05.commit(mb + MB_U2ACC + csm)

                        mma_4a.set(tcgen05.Field.ACCUMULATE, True)
                        if cutlass.const_expr(GATE2_ == 1):
                            u2_inp_lo_bar.arrive_and_wait()
                            cute.gemm(
                                mma_4a,
                                t_d4a,
                                t_ri4[None, None, 0],
                                b_fta[None, None, 0, csm],
                                t_d4a,
                            )
                            u2_inp_bar.arrive_and_wait()
                            cute.gemm(
                                mma_4a,
                                t_d4a,
                                t_ri4[None, None, 1],
                                b_fta[None, None, 1, csm],
                                t_d4a,
                            )
                        else:
                            u2_inp_bar.arrive_and_wait()
                            for kb in cutlass.range_constexpr(C // 16):
                                cute.gemm(
                                    mma_4a,
                                    t_d4a,
                                    t_ri4[None, None, kb],
                                    b_fta[None, None, kb, csm],
                                    t_d4a,
                                )
                        with cute.arch.elect_one():
                            tcgen05.commit(mb + MB_FIN + csm)
                        mma_4b.set(tcgen05.Field.ACCUMULATE, True)
                        for kb in cutlass.range_constexpr(C // 16):
                            cute.gemm(
                                mma_4b,
                                t_d4b,
                                t_ri4[None, None, kb],
                                b_ftb[None, None, kb, csm],
                                t_d4b,
                            )
                        with cute.arch.elect_one():
                            tcgen05.commit(mb + MB_OFIN + csm)
                            tcgen05.commit(mb + MB_SFREE + csm)
                        csm += 1
                        if csm == STAGES:
                            csm = 0
                        pm_oe ^= 1
        if warp == 10:
            csl = cutlass.Int32(0)
            pl_vfree = cutlass.Int32(1)
            pl_qk = cutlass.Int32(0)
            gE = cute.flat_divide(mE, (64, 1, 256))
            for kx in cutlass.range(k1 - k0):
                chain = cute.arch.make_warp_uniform(schain[k0 + kx])
                seq_idx = chain // H_
                hidx = chain % H_
                bos = cutlass.Int32(cu[seq_idx]) + cute.arch.make_warp_uniform(
                    spt0[k0 + kx]
                )
                seq_len = cute.arch.make_warp_uniform(sptn[k0 + kx])
                t_tiles = (seq_len + C - 1) // C
                if cutlass.const_expr(STATE_ == 1):
                    simp_l = cute.arch.make_warp_uniform(ssrc[k0 + kx])
                    if simp_l == -1:
                        # This loader runs roughly five chunks ahead of the
                        # recurrence. One elected 64 KiB L2 request warms the
                        # real chain seed without consuming L1 sectors or
                        # extending compute's state fragments.
                        with cute.arch.elect_one():
                            _bulk_prefetch_l2(
                                state0.iterator + chain * (D * D),
                                cutlass.Int32(D * D * 4),
                            )
                if cutlass.const_expr(GATE2_ == 1):
                    gV = cute.flat_divide(
                        cute.domain_offset((0, 0, bos), mV), (D, 1, C)
                    )
                else:
                    gV = cute.flat_divide(
                        cute.domain_offset((bos, 0, 0), mV), (C, 1, D)
                    )
                for t in cutlass.range(t_tiles):
                    cute.arch.mbarrier_wait(mb + MB_VFREE + csl, pl_vfree)
                    cute.arch.mbarrier_wait(mb + MB_QK + csl, pl_qk)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mb + MB_VFULL + csl, C * D * 2
                        )
                    if cutlass.const_expr(GATE2_ == 1):
                        f_v = v_vt[None, None, None, csl]
                        v_d, v_s = cpasync.tma_partition(
                            tma_v,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(f_v, 0, 3),
                            cute.group_modes(gV, 0, 3),
                        )
                        cute.copy(
                            tma_v,
                            v_s[(None, 0, hidx, t)],
                            v_d,
                            tma_bar_ptr=mb + MB_VFULL + csl,
                        )
                    else:
                        f_v = cute.make_tensor(
                            p_v_raw + csl * STAGE_ELTS,
                            cute.make_layout((C, 1, D), stride=(D, 0, 1)),
                        )
                        v_d, v_s = cpasync.tma_partition(
                            tma_v,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(f_v, 0, 3),
                            cute.group_modes(gV, 0, 3),
                        )
                        cute.copy(
                            tma_v,
                            v_s[(None, t, hidx, 0)],
                            v_d,
                            tma_bar_ptr=mb + MB_VFULL + csl,
                        )
                    if cutlass.const_expr(EXPORT_ == 1):
                        if seq_idx == export_seq:
                            # flat byte-image dump of the whole prep stage
                            # (v36 law: reload with the same flat box + the
                            # canonical views reproduces the operands); the
                            # 5-deep ring gives ~4 chunk periods of slack
                            # before prep(t+5) overwrites this stage
                            f_e = cute.make_tensor(
                                cute.recast_ptr(
                                    ar0 + csl * STAGE_BYTES, dtype=cutlass.BFloat16
                                ),
                                cute.make_layout((64, 1, 256), stride=(256, 0, 1)),
                            )
                            e_s, e_g = cpasync.tma_partition(
                                tma_e,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(f_e, 0, 3),
                                cute.group_modes(gE, 0, 3),
                            )
                            cute.copy(tma_e, e_s, e_g[(None, hidx * nc2 + t, 0, 0)])
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(2, read=True)
                            # gt/beta tables (dedicated buffers, scalar STG)
                            for kt5 in cutlass.range_constexpr(4):
                                expt[hidx * nc2 + t, kt5 * 32 + lane] = v_gt[
                                    kt5 * 32 + lane, csl
                                ]
                            expt[hidx * nc2 + t, 128 + lane] = v_bt[lane, csl]
                    csl += 1
                    if csl == STAGES:
                        csl = 0
                        pl_vfree ^= 1
                        pl_qk ^= 1

    # =====================================================================
    # PREP (warps 12-31): 5 instances x 4 warps; instance == stage
    # =====================================================================
    if warp >= 12:
        cute.arch.warpgroup_reg_dealloc(48)
        inst = (warp - 12) >> 2
        plw = (warp - 12) & 3
        ptl = plw * 32 + lane

        mma_a = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(
                cutlass.BFloat16, cutlass.Float32, (16, 8, 16)
            ),
            cute.make_layout((1, 1, 1)),
        )
        thr_a = mma_a.get_slice(lane)
        tIdA = thr_a.partition_C(cute.make_identity_tensor((16, 16)))
        tIdA16 = thr_a.partition_A(cute.make_identity_tensor((16, 16)))
        tIdB16 = thr_a.partition_B(cute.make_identity_tensor((16, 16)))
        # gram result window (one [64,32] f32 window per stage; the
        # issuing instance reuses its own window lap-to-lap in program
        # order, so no free barrier is needed)
        thr_gm = mma_g.get_slice(0)
        gm_base = thr_gm.make_fragment_C(mma_g.partition_shape_C((2 * C, C)))
        # route the per-instance column offset through a 64-bit recast so
        # the 16-DP tmem-load atoms keep their provable 2-col alignment
        p_g64 = cute.recast_ptr(tmem_ptr + GRAM_COL, dtype=cutlass.Float64)
        t_gram = cute.make_tensor(
            cute.recast_ptr(p_g64 + inst * (C // 2), dtype=cutlass.Float32),
            gm_base.layout,
        )
        gram_atom = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x4), cutlass.Float32
        )
        gram_ld = tcgen05.make_tmem_copy(gram_atom, t_gram)
        gm_thr = gram_ld.get_slice(ptl)
        gram_id = gm_thr.partition_D(
            thr_gm.partition_C(cute.make_identity_tensor((2 * C, C)))
        )
        r_gram = cute.make_rmem_tensor(gram_id.shape, cutlass.Float32)
        # Ld16x256b has the thread/value ownership required by Blackwell's
        # matrix-store helper.  Retile the register fragment once and write
        # the combined [G^T; L] image cooperatively instead of issuing 16
        # scalar BF16 shared stores per prep thread.
        gram_st_atom = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.COL_MAJOR, cutlass.BFloat16, cutlass.Float32, gram_ld
        )
        gram_st = cute.make_tiled_copy_D(gram_st_atom, gram_ld)
        gm_st_thr = gram_st.get_slice(ptl)
        r_gram_st = gram_st.retile(r_gram)
        r_gram_b = cute.make_rmem_tensor(r_gram_st.shape, cutlass.BFloat16)
        s_gram_st = gm_st_thr.partition_D(t_sgram[None, None, None, inst])

        lower_bound_log2 = lb2
        if cutlass.const_expr(LOWER_BOUND_M5_ == 1):
            lower_bound_log2 = -7.213475204444817
        lb2h = lower_bound_log2 * 0.5
        anch = lower_bound_log2 * 16.0
        konst2 = _exp2f(anch)
        inv_konst2 = _exp2f(-anch)

        qr = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.BFloat16)
        kr = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.BFloat16)
        g8 = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.Float32)
        rf8 = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.Float32)
        w2 = cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.BFloat16)
        g2 = cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.Float32)
        dtb2 = cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.Float32)

        pp_rfree = cutlass.Int32(1)
        pp_graw = cutlass.Int32(0)
        pp_sfree = cutlass.Int32(1)
        pp_qkraw = cutlass.Int32(0)
        pp_braw = cutlass.Int32(0)
        pp_gram = cutlass.Int32(0)
        ibar = pipeline.NamedBarrier(barrier_id=7 + inst, num_threads=128)

        ph0 = cutlass.Int32(0)
        for kx in cutlass.range(k1 - k0):
            chain = cute.arch.make_warp_uniform(schain[k0 + kx])
            seq_idx = chain // H_
            hidx = chain % H_
            bos = cutlass.Int32(cu[seq_idx]) + cute.arch.make_warp_uniform(
                spt0[k0 + kx]
            )
            seq_len = cute.arch.make_warp_uniform(sptn[k0 + kx])
            simp = cute.arch.make_warp_uniform(ssrc[k0 + kx])
            t_tiles = (seq_len + C - 1) // C
            gQ = cute.flat_divide(cute.domain_offset((bos, 0, 0), mQ), (C, 1, D))
            gK = cute.flat_divide(cute.domain_offset((bos, 0, 0), mK), (C, 1, D))
            gG = cute.flat_divide(cute.domain_offset((bos, 0, 0), mG), (C, 1, D))
            if cutlass.const_expr(BETA_TMA_ == 1):
                gB = cute.flat_divide(cute.domain_offset((bos, 0), mB), (C, 8))
                f_b = v_braw[None, None, inst]
                b_d, b_s = cpasync.tma_partition(
                    tma_b,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(f_b, 0, 2),
                    cute.group_modes(gB, 0, 2),
                )
            ea = cute.math.exp(a_log[hidx], fastmath=True)
            ea2c = 0.5 * ea
            dtbc = ea2c * cutlass.Float32(dt_bias[hidx, ptl])
            # this instance owns local chunks li0, li0+5, ... (< t_tiles):
            # li0 aligns inst with the CTA-global chunk stream position
            li0 = inst - ph0
            if li0 < 0:
                li0 += STAGES
            n_iters = (t_tiles - li0 + STAGES - 1) // STAGES
            ph0 += t_tiles
            ph0 = ph0 % STAGES
            n_full_iters = n_iters
            if cutlass.const_expr(FULL_CHUNKS_ == 0 and PREP_PEEL_ == 1):
                last_ci = li0 + (n_iters - 1) * STAGES
                if (n_iters > 0) & ((last_ci + 1) * C > seq_len):
                    n_full_iters -= 1
            for tail_phase in cutlass.range_constexpr(
                1 if FULL_CHUNKS_ == 1 or PREP_PEEL_ == 0 else 2
            ):
                phase_iters = (
                    n_full_iters if tail_phase == 0 else n_iters - n_full_iters
                )
                phase_base = 0 if tail_phase == 0 else n_full_iters
                if cutlass.const_expr(FULL_CHUNKS_ == 1 or PREP_PEEL_ == 1):
                    peeled_full = cutlass.const_expr(
                        FULL_CHUNKS_ == 1 or tail_phase == 0
                    )
                for phase_it in cutlass.range(phase_iters):
                    it = phase_base + phase_it
                    ci = li0 + it * STAGES
                    r0 = bos + ci * C
                    cc = cutlass.min(cutlass.Int32(C), seq_len - ci * C)
                    if cutlass.const_expr(FULL_CHUNKS_ == 1 or PREP_PEEL_ == 1):
                        full = peeled_full
                    else:
                        full = seq_len >= (ci + 1) * C

                    # -- phase 0: raw g/k TMA (qd/kd slots free after MMA2(ci-5)) --
                    if cutlass.const_expr(BETA_TMA_ == 1):
                        if plw == 2:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    mb + MB_BRAW + inst, C * 8 * 2
                                )
                            cute.copy(
                                tma_b,
                                b_s[(None, ci, hidx >> 3)],
                                b_d,
                                tma_bar_ptr=mb + MB_BRAW + inst,
                            )
                    if full:
                        cute.arch.mbarrier_wait(mb + MB_RFREE + inst, pp_rfree)
                        if plw == 0:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    mb + MB_GRAW + inst, C * D * 2
                                )
                            f_g = cute.make_tensor(
                                cute.recast_ptr(
                                    ar0 + (OFF_QD + inst * STAGE_BYTES),
                                    dtype=cutlass.BFloat16,
                                ),
                                cute.make_layout(
                                    (C, 1, (64, 2)), stride=(64, 0, (1, 4096))
                                ),
                            )
                            g_d, g_s = cpasync.tma_partition(
                                tma_g,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(f_g, 0, 3),
                                cute.group_modes(gG, 0, 3),
                            )
                            cute.copy(
                                tma_g,
                                g_s[(None, ci, hidx, 0)],
                                g_d,
                                tma_bar_ptr=mb + MB_GRAW + inst,
                            )
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    mb + MB_QKRAW + inst, 2 * C * D * 2
                                )
                            f_k = cute.make_tensor(
                                cute.recast_ptr(
                                    ar0 + (OFF_QD + 4096 + inst * STAGE_BYTES),
                                    lay_qd.inner,
                                    dtype=cutlass.BFloat16,
                                ),
                                cute.make_layout(
                                    (C, 1, (16, 4, 2)), stride=(64, 0, (1, 16, 4096))
                                ),
                            )
                            k_d, k_s = cpasync.tma_partition(
                                tma_k,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(f_k, 0, 3),
                                cute.group_modes(gK, 0, 3),
                            )
                            cute.copy(
                                tma_k,
                                k_s[(None, ci, hidx, 0)],
                                k_d,
                                tma_bar_ptr=mb + MB_QKRAW + inst,
                            )

                    # G/beta occupy intervals that are already free before the
                    # final-recurrence SFREE boundary.  Evaluate their first
                    # independent nonlinear results now and carry only those
                    # scalars across the wait, matching the retained overlap schedule.
                    btv = cutlass.Float32(0.0)
                    early_gate0 = cutlass.Float32(0.0)
                    if cutlass.const_expr((FULL_CHUNKS_ == 1) and (QK_ROWPAIR_ == 8)):
                        if plw == 2:
                            if cutlass.const_expr(BETA_TMA_ == 1):
                                cute.arch.mbarrier_wait(mb + MB_BRAW + inst, pp_braw)
                            if lane < cc:
                                if cutlass.const_expr(BETA_TMA_ == 1):
                                    bx = cutlass.Float32(v_braw[lane, hidx & 7, inst])
                                else:
                                    bx = cutlass.Float32(beta[r0 + lane, hidx])
                                btv = 0.5 + 0.5 * _tanhf(0.5 * bx)
                        if cutlass.const_expr(QK_ROWPAIR_ == 8):
                            cute.arch.mbarrier_wait(mb + MB_GRAW + inst, pp_graw)
                            early_gv0 = cutlass.Float32(v_graw[0, ptl, inst])
                            early_gate0 = lb2h * _tanhf(ea2c * early_gv0 + dtbc) + lb2h

                    # -- phase 1: everything else waits MMA4(ci-5) --
                    cute.arch.mbarrier_wait(mb + MB_SFREE + inst, pp_sfree)
                    if full:
                        if plw == 0:
                            f_qr = cute.make_tensor(
                                cute.recast_ptr(
                                    ar0 + (OFF_FT + inst * STAGE_BYTES),
                                    lay_qd.inner,
                                    dtype=cutlass.BFloat16,
                                ),
                                cute.make_layout(
                                    (C, 1, (16, 4, 2)), stride=(64, 0, (1, 16, 2048))
                                ),
                            )
                            q_d, q_s = cpasync.tma_partition(
                                tma_q,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(f_qr, 0, 3),
                                cute.group_modes(gQ, 0, 3),
                            )
                            cute.copy(
                                tma_q,
                                q_s[(None, ci, hidx, 0)],
                                q_d,
                                tma_bar_ptr=mb + MB_QKRAW + inst,
                            )

                    # beta (dedicated buffer -> written pre-walker; read by the
                    # grams two barriers later and by compute after qk_full)
                    if plw == 2:
                        if cutlass.const_expr(
                            (FULL_CHUNKS_ == 0) or (QK_ROWPAIR_ != 8)
                        ):
                            if cutlass.const_expr(BETA_TMA_ == 1):
                                cute.arch.mbarrier_wait(mb + MB_BRAW + inst, pp_braw)
                            if lane < cc:
                                if cutlass.const_expr(BETA_TMA_ == 1):
                                    bx = cutlass.Float32(v_braw[lane, hidx & 7, inst])
                                else:
                                    bx = cutlass.Float32(beta[r0 + lane, hidx])
                                btv = 0.5 + 0.5 * _tanhf(0.5 * bx)
                        if lane < C:
                            v_bt[lane, inst] = btv
                            if cutlass.const_expr(BETA_BF16_ == 1):
                                v_btb[lane, inst] = btv.to(cutlass.BFloat16)

                    if cutlass.const_expr((GATE2_ == 1) or (QK_ROWPAIR_ == 8)):
                        # v111 fixed-shape specialization: one channel per
                        # thread across all four warps, running sum in a register.
                        # This replaces the v98 two-channel/two-warp scan and the
                        # store-gate / reload / walker phase pair (~256 fewer
                        # L1 wavefronts per chunk on the ~86%-busy LSU pipe).
                        # v99 note kept out: TMAs stay full-chunk-gated here.
                        if cutlass.const_expr(
                            (FULL_CHUNKS_ == 0) or (QK_ROWPAIR_ != 8)
                        ):
                            if full:
                                cute.arch.mbarrier_wait(mb + MB_GRAW + inst, pp_graw)
                        accg = cutlass.Float32(0.0)
                        if full:
                            if cutlass.const_expr(GATE_ROLL_ == 1):
                                for rp in cutlass.range(C // 8):
                                    rw0 = rp * 8
                                    for u in cutlass.range_constexpr(8):
                                        g8[u] = cutlass.Float32(
                                            v_graw[rw0 + u, ptl, inst]
                                        )
                                    g8.store(g8.load() * ea2c + dtbc)
                                    for u in cutlass.range_constexpr(8):
                                        g8[u] = _tanhf(g8[u])
                                    g8.store(g8.load() * lb2h + lb2h)
                                    for u in cutlass.range_constexpr(8):
                                        accg += g8[u]
                                        v_gcs[rw0 + u, ptl, inst] = accg
                            else:
                                for rp in cutlass.range_constexpr(C // 16):
                                    rw0 = rp * 16
                                    for u in cutlass.range_constexpr(8):
                                        g8[u] = cutlass.Float32(
                                            v_graw[rw0 + u, ptl, inst]
                                        )
                                        rf8[u] = cutlass.Float32(
                                            v_graw[rw0 + 8 + u, ptl, inst]
                                        )
                                    g8.store(g8.load() * ea2c + dtbc)
                                    rf8.store(rf8.load() * ea2c + dtbc)
                                    for u in cutlass.range_constexpr(8):
                                        g8[u] = _tanhf(g8[u])
                                        rf8[u] = _tanhf(rf8[u])
                                    g8.store(g8.load() * lb2h + lb2h)
                                    rf8.store(rf8.load() * lb2h + lb2h)
                                    if cutlass.const_expr(
                                        (FULL_CHUNKS_ == 1) and (QK_ROWPAIR_ == 8)
                                    ):
                                        g8[0] = early_gate0 if rp == 0 else g8[0]
                                    for u in cutlass.range_constexpr(8):
                                        accg += g8[u]
                                        v_gcs[rw0 + u, ptl, inst] = accg
                                    for u in cutlass.range_constexpr(8):
                                        accg += rf8[u]
                                        v_gcs[rw0 + 8 + u, ptl, inst] = accg
                        else:
                            for rw in cutlass.range_constexpr(C):
                                if rw < cc:
                                    gv = cutlass.Float32(g[r0 + rw, hidx, ptl])
                                    accg += lb2h * _tanhf(ea2c * gv + dtbc) + lb2h
                                v_gcs[rw, ptl, inst] = accg
                        if (ci == 0) & (simp < -1) & (accg >= -50.0):
                            failure_ptr = cute.make_ptr(
                                cutlass.Int32,
                                failure_addr,
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            )
                            cute.arch.store(
                                failure_ptr,
                                fepoch,
                                sem="release",
                                scope="sys",
                            )
                        ibar.arrive_and_wait()
                    # -- phase 3: q/k load + l2norm + anchored decorations --
                    # (unroll=2 measured-closed: 1.569x -> 1.445x fixed_h96 —
                    # the 48-reg prep diet spills, reconfirming the v60 re-roll)
                    if full:
                        cute.arch.mbarrier_wait(mb + MB_QKRAW + inst, pp_qkraw)
                    for wp in cutlass.range(4):
                        # Both retained fixed/varlen policies use the same
                        # four-row half-warp pairing. QK_ROWPAIR_ only selects
                        # the surrounding gate pipeline now.
                        rw2 = wp * 8 + (ptl >> 5) + ((ptl >> 2) & 4)
                        sg2 = ptl & 15
                        if full:
                            q8s = v_ki8[(rw2, None, sg2 & 7, sg2 >> 3, inst)]
                            cute.autovec_copy(q8s, qr)
                            k8s = v_kd8[(rw2, None, sg2 & 7, sg2 >> 3, inst)]
                            cute.autovec_copy(k8s, kr)
                        else:
                            if rw2 < cc:
                                gq8 = cute.local_tile(
                                    q, (1, 1, 8), (r0 + rw2, hidx, sg2)
                                )
                                cute.autovec_copy(gq8, qr)
                                gk8 = cute.local_tile(
                                    k, (1, 1, 8), (r0 + rw2, hidx, sg2)
                                )
                                cute.autovec_copy(gk8, kr)
                            else:
                                for u in cutlass.range_constexpr(8):
                                    qr[u] = cutlass.BFloat16(0.0)
                                    kr[u] = cutlass.BFloat16(0.0)
                        sq = cutlass.Float32(0.0)
                        sk = cutlass.Float32(0.0)
                        for u in cutlass.range_constexpr(8):
                            fq = cutlass.Float32(qr[u])
                            fk = cutlass.Float32(kr[u])
                            sq += fq * fq
                            sk += fk * fk
                        for off in [1, 2, 4, 8]:
                            sq += cute.arch.shuffle_sync_bfly(sq, off)
                            sk += cute.arch.shuffle_sync_bfly(sk, off)
                        # scale folded into rqn: one FMUL replaces 8 per slice
                        rqn = cute.math.rsqrt(sq + 1e-6, fastmath=True) * scale
                        rkn = cute.math.rsqrt(sk + 1e-6, fastmath=True)
                        g8s = v_gcs8[(rw2, None, sg2, inst)]
                        cute.autovec_copy(g8s, g8)
                        gv8 = g8.load()
                        decv = cute.math.exp2(gv8 - anch, fastmath=True)
                        qhv = qr.load().to(cutlass.Float32) * rqn
                        khv = kr.load().to(cutlass.Float32) * rkn
                        qr.store((qhv * decv).to(cutlass.BFloat16))
                        kr.store((khv * decv).to(cutlass.BFloat16))
                        if cutlass.const_expr(RCP_DECOR_ == 1):
                            # The inverse decoration is exactly reciprocal to the
                            # forward decay.  Its consumer is rounded to BF16, so
                            # a native approximate reciprocal avoids a second
                            # vector exp2 while retaining more precision than
                            # that store.  The exp2 control remains for matched
                            # profiling.
                            idecv = cute.math.rcp(decv, approx=True, ftz=True)
                        else:
                            idecv = cute.math.exp2(anch - gv8, fastmath=True)
                        dq8 = v_qd8[(rw2, None, sg2 & 7, sg2 >> 3, inst)]
                        cute.autovec_copy(qr, dq8)
                        dk8 = v_kd8[(rw2, None, sg2 & 7, sg2 >> 3, inst)]
                        cute.autovec_copy(kr, dk8)
                        kr.store((khv * idecv).to(cutlass.BFloat16))
                        di8 = v_ki8[(rw2, None, sg2 & 7, sg2 >> 3, inst)]
                        cute.autovec_copy(kr, di8)
                    # gt/rf (dedicated buffers; gcs row 31 stable since the
                    # post-walker barrier -> rides the decoration phase)
                    if ptl < D:
                        gl_t = cutlass.Float32(v_gcs[C - 1, ptl, inst])
                        rfv = _exp2f(gl_t - anch)
                        v_rf[ptl, inst] = rfv
                        v_gt[ptl, inst] = rfv * konst2
                    if ptl == 0:
                        v_rf[D, inst] = _exp2f(anch)
                    ibar.arrive_and_wait()
                    if plw == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mb + MB_TAB + inst)

                    # -- phase 5: gram via one m64n32k128 SS tcgen05 MMA over
                    # the interleaved qd||kd block x ki.  D rows 0-31 = G^T
                    # (warps 0-1), rows 32-63 = raw L (warps 2-3); each warp
                    # reads its own 16-lane block, masks, and stores. --
                    if cutlass.const_expr(GRAM_W9_ == 0):
                        # Warp 2 balances Gram issue against warp 0's solve path.
                        if plw == 2:
                            mma_g.set(tcgen05.Field.ACCUMULATE, False)
                            for kb in cutlass.range_constexpr(D // 16):
                                cute.gemm(
                                    mma_g,
                                    t_gram,
                                    a_qk[None, None, kb, inst],
                                    b_ki[None, None, kb, inst],
                                    t_gram,
                                )
                                mma_g.set(tcgen05.Field.ACCUMULATE, True)
                            with cute.arch.elect_one():
                                tcgen05.commit(mb + MB_GRAM + inst)
                    cute.arch.mbarrier_wait(mb + MB_GRAM + inst, pp_gram)
                    cute.copy(gram_ld, gm_thr.partition_S(t_gram), r_gram)
                    cute.arch.fence_view_async_tmem_load()
                    if plw < 2:
                        for e in cutlass.range_constexpr(cute.size(r_gram)):
                            crd = gram_id[e]
                            mv = cutlass.Float32(0.0)
                            if crd[1] <= crd[0]:
                                mv = r_gram[e] * inv_konst2
                            r_gram[e] = mv
                    else:
                        for e in cutlass.range_constexpr(cute.size(r_gram)):
                            crd = gram_id[e]
                            gi = crd[0] - C
                            lv = cutlass.Float32(0.0)
                            if crd[1] < gi:
                                lv = r_gram[e] * v_bt[gi, inst]
                            r_gram[e] = lv
                    r_gram_b.store(r_gram_st.load().to(cutlass.BFloat16))
                    cute.copy(gram_st, r_gram_b, s_gram_st)
                    ibar.arrive_and_wait()

                    # -- phase 6: solve, PR-style hierarchy.  Level 1: four
                    # 8x8 unit-lower inverses by in-register shuffle
                    # elimination (zero LDS on the elimination chain);
                    # level 2: both 16-blocks' -B^-1 C A^-1 combines batched
                    # as block-diagonal [16,16] warp-MMAs. --
                    if plw == 0:
                        dgb = lane >> 3
                        lid = lane & 7
                        b8 = dgb * 8
                        cute.autovec_copy(v_lw_diag[(dgb, lid, None, inst)], qr)
                        for c8 in cutlass.range_constexpr(8):
                            lv8 = cutlass.Float32(0.0)
                            if c8 < lid:
                                lv8 = cutlass.Float32(qr[c8])
                            if c8 == lid:
                                lv8 = cutlass.Float32(1.0)
                            g8[c8] = lv8
                        for sr in cutlass.range_constexpr(7):
                            rs8 = cutlass.Float32(0.0) - g8[sr]
                            for pc in cutlass.range_constexpr(7):
                                if cutlass.const_expr(pc < sr):
                                    pv8 = cute.arch.shuffle_sync(g8[pc], b8 + sr)
                                    if lid > sr:
                                        g8[pc] = rs8 * pv8 + g8[pc]
                            if lid > sr:
                                g8[sr] = rs8
                        qr.store(g8.load().to(cutlass.BFloat16))
                        cute.autovec_copy(qr, v_inv[(b8 + lid, None, dgb, inst)])
                        # zero the upper-right 8x8 of each 16-block and the
                        # 32-level upper-right [0:16,16:32)
                        qr.fill(cutlass.BFloat16(0.0))
                        if dgb == 0:
                            cute.autovec_copy(qr, v_inv[(lid, None, 1, inst)])
                        if dgb == 2:
                            cute.autovec_copy(qr, v_inv[(16 + lid, None, 3, inst)])
                        halfl = lane >> 4
                        coll = lane & 15
                        if halfl == 0:
                            cute.autovec_copy(qr, v_inv[(coll, None, 2, inst)])
                            cute.autovec_copy(qr, v_inv[(coll, None, 3, inst)])
                        cute.arch.sync_warp()
                        # level 2: Y = diag(A1inv,B1inv) @ diag(C_A,C_B)
                        Cy2 = mma_a.make_fragment_C(mma_a.partition_shape_C((16, 16)))
                        Cy2.fill(0.0)
                        fa2 = thr_a.make_fragment_A(mma_a.partition_shape_A((16, 16)))
                        fb2 = thr_a.make_fragment_B(mma_a.partition_shape_B((16, 16)))
                        for e in cutlass.range_constexpr(cute.size(fa2)):
                            ac = tIdA16[e]
                            va2 = cutlass.BFloat16(0.0)
                            if (ac[0] >> 3) == (ac[1] >> 3):
                                va2 = v_inv[
                                    8 + 16 * (ac[0] >> 3) + (ac[0] & 7),
                                    ac[1] & 7,
                                    1 + 2 * (ac[0] >> 3),
                                    inst,
                                ]
                            fa2[e] = va2
                        for e in cutlass.range_constexpr(cute.size(fb2)):
                            bc = tIdB16[e]
                            vb2 = cutlass.BFloat16(0.0)
                            if (bc[1] >> 3) == (bc[0] >> 3):
                                vb2 = v_gram[
                                    C + 8 + 16 * (bc[1] >> 3) + (bc[1] & 7),
                                    16 * (bc[1] >> 3) + (bc[0] & 7),
                                    inst,
                                ]
                            fb2[e] = vb2
                        cute.gemm(mma_a, Cy2, fa2, fb2, Cy2)
                        yA2 = thr_a.make_fragment_A(mma_a.partition_shape_A((16, 16)))
                        for e in cutlass.range_constexpr(cute.size(Cy2)):
                            yA2[e] = cutlass.BFloat16(Cy2[e])
                        fbt2 = thr_a.make_fragment_B(mma_a.partition_shape_B((16, 16)))
                        for e in cutlass.range_constexpr(cute.size(fbt2)):
                            bc2 = tIdB16[e]
                            vt2 = cutlass.BFloat16(0.0)
                            if (bc2[1] >> 3) == (bc2[0] >> 3):
                                vt2 = v_inv[
                                    16 * (bc2[1] >> 3) + (bc2[1] & 7),
                                    bc2[0] & 7,
                                    2 * (bc2[1] >> 3),
                                    inst,
                                ]
                            fbt2[e] = vt2
                        Cc2 = mma_a.make_fragment_C(mma_a.partition_shape_C((16, 16)))
                        Cc2.fill(0.0)
                        cute.gemm(mma_a, Cc2, yA2, fbt2, Cc2)
                        for e in cutlass.range_constexpr(cute.size(Cc2)):
                            crd = tIdA[e]
                            if (crd[0] >> 3) == (crd[1] >> 3):
                                v_inv[
                                    8 + 16 * (crd[0] >> 3) + (crd[0] & 7),
                                    crd[1] & 7,
                                    2 * (crd[0] >> 3),
                                    inst,
                                ] = cutlass.BFloat16(0.0 - Cc2[e])
                        cute.arch.sync_warp()
                    # (no barrier: the combine is warp 0's own program order; the
                    #  restore reads only pre-gram-barrier data)

                    # -- phase 7: off-diag combine (warp 0) || restore (warps 1-3) --
                    # INV[16:,:16] = -(Tb @ Lc) @ Ta; fragments loaded elementwise
                    # by identity coords (transposed dynamic-offset views reject
                    # autovec: provenance law).
                    if plw == 0:
                        Cy = mma_a.make_fragment_C(mma_a.partition_shape_C((16, 16)))
                        Cy.fill(0.0)
                        fat = thr_a.make_fragment_A(mma_a.partition_shape_A((16, 16)))
                        fbl = thr_a.make_fragment_B(mma_a.partition_shape_B((16, 16)))
                        for e in cutlass.range_constexpr(cute.size(fat)):
                            ac = tIdA16[e]
                            fat[e] = v_inv[
                                16 + ac[0], ac[1] & 7, 2 + (ac[1] >> 3), inst
                            ]
                        for e in cutlass.range_constexpr(cute.size(fbl)):
                            bc = tIdB16[e]
                            fbl[e] = v_gram[C + 16 + bc[1], bc[0], inst]
                        cute.gemm(mma_a, Cy, fat, fbl, Cy)
                        yA = thr_a.make_fragment_A(mma_a.partition_shape_A((16, 16)))
                        for e in cutlass.range_constexpr(cute.size(Cy)):
                            yA[e] = cutlass.BFloat16(Cy[e])
                        fbt = thr_a.make_fragment_B(mma_a.partition_shape_B((16, 16)))
                        for e in cutlass.range_constexpr(cute.size(fbt)):
                            bc2 = tIdB16[e]
                            fbt[e] = v_inv[bc2[1], bc2[0] & 7, bc2[0] >> 3, inst]
                        Cc = mma_a.make_fragment_C(mma_a.partition_shape_C((16, 16)))
                        Cc.fill(0.0)
                        cute.gemm(mma_a, Cc, yA, fbt, Cc)
                        for pair in cutlass.range_constexpr(4):
                            e = 2 * pair
                            crd = tIdA[e]
                            c2 = cute.make_rmem_tensor(
                                cute.make_layout((2,)), cutlass.BFloat16
                            )
                            c2[0] = cutlass.BFloat16(0.0 - Cc[e])
                            c2[1] = cutlass.BFloat16(0.0 - Cc[e + 1])
                            d2 = cute.local_tile(
                                v_inv,
                                (1, 2, 1, 1),
                                (
                                    16 + crd[0],
                                    (crd[1] & 7) >> 1,
                                    crd[1] >> 3,
                                    inst,
                                ),
                            )
                            cute.autovec_copy(c2, d2)
                    else:
                        # ``96`` restore workers advance by a whole multiple of
                        # the 16 RF segments, so each thread's segment is
                        # invariant across all six row-work iterations.  Load
                        # that shared factor vector once instead of 5--6 times.
                        if cutlass.const_expr(RF_HOIST_ == 1):
                            sg_restore = lane & 15
                            cute.autovec_copy(v_rf8[(sg_restore, None, inst)], rf8)
                        restore_iters = (
                            5 if cutlass.const_expr(RESTORE_TAIL_ == 1) else 6
                        )
                        for wp in cutlass.range(restore_iters):
                            item = wp * 96 + (ptl - 32)
                            if item < C * 16:
                                rw3 = item >> 4
                                sg3 = item & 15
                                si8 = v_ki8[(rw3, None, sg3 & 7, sg3 >> 3, inst)]
                                cute.autovec_copy(si8, kr)
                                if cutlass.const_expr(RF_HOIST_ == 0):
                                    cute.autovec_copy(v_rf8[(sg3, None, inst)], rf8)
                                kr.store(kr.load() * rf8.load().to(cutlass.BFloat16))
                                cute.autovec_copy(kr, si8)
                    # Warps 1--3 restore the first 480 independent row-segments
                    # in five balanced rounds.  Once its triangular solve is
                    # complete, warp 0 restores the 32-segment tail that
                    # previously made one third of the restore workers take a
                    # sixth round.  The shared stage is not published until the
                    # existing four-warp barrier below.
                    if cutlass.const_expr(RESTORE_TAIL_ == 1):
                        if plw == 0:
                            item = 480 + lane
                            rw3 = item >> 4
                            sg3 = item & 15
                            cute.autovec_copy(v_rf8[(sg3, None, inst)], rf8)
                            si8 = v_ki8[(rw3, None, sg3 & 7, sg3 >> 3, inst)]
                            cute.autovec_copy(si8, kr)
                            kr.store(kr.load() * rf8.load().to(cutlass.BFloat16))
                            cute.autovec_copy(kr, si8)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    ibar.arrive_and_wait()
                    if plw == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mb + MB_QK + inst)
                    pp_rfree ^= 1
                    pp_sfree ^= 1
                    pp_gram ^= 1
                    if cutlass.const_expr(BETA_TMA_ == 1):
                        pp_braw ^= 1
                    # GRAW/QKRAW are armed+waited only on full chunks; their
                    # parities must advance 1:1 with actual completions or the
                    # NEXT chain in this persistent slot waits a stale phase
                    # (RFREE/SFREE are compute-armed every chunk -> uncond.)
                    if full:
                        pp_graw ^= 1
                        pp_qkraw ^= 1

    cute.arch.sync_threads()
    tmem.free(tmem_ptr)


@cute.jit
def _launch_pkd(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    state0: cute.Tensor,
    stateT: cute.Tensor,
    out: cute.Tensor,
    cu: cute.Tensor,
    soff: cute.Tensor,
    schain: cute.Tensor,
    spt0: cute.Tensor,
    sptn: cute.Tensor,
    ssrc: cute.Tensor,
    sdst: cute.Tensor,
    midstate: cute.Tensor,
    mflags: cute.Tensor,
    exp_ws: cute.Tensor,
    expt: cute.Tensor,
    tprobe: cute.Tensor,
    failure_addr: cutlass.Int64,
    have_state: cutlass.Int32,
    gcnt: cutlass.Int32,
    do_export: cutlass.Int32,
    export_seq: cutlass.Int32,
    nc2: cutlass.Int32,
    fepoch: cutlass.Int32,
    scale: cutlass.Float32,
    lb2: cutlass.Float32,
    H_: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
    TPROBE_: cutlass.Constexpr[int] = 0,
    GATE2_: cutlass.Constexpr[int] = 0,
    FINAL_: cutlass.Constexpr[int] = 1,
    SPLIT_: cutlass.Constexpr[int] = 1,
    EXPORT_: cutlass.Constexpr[int] = 1,
    STATE_: cutlass.Constexpr[int] = 0,
    BETA_TMA_: cutlass.Constexpr[int] = 0,
    BF16_DV_: cutlass.Constexpr[int] = 0,
    BETA_BF16_: cutlass.Constexpr[int] = 0,
    QK_ROWPAIR_: cutlass.Constexpr[int] = 0,
    RCP_DECOR_: cutlass.Constexpr[int] = 1,
    RF_HOIST_: cutlass.Constexpr[int] = 1,
    RESTORE_TAIL_: cutlass.Constexpr[int] = 0,
    GRAM_W9_: cutlass.Constexpr[int] = 0,
    REG_MODE_: cutlass.Constexpr[int] = 0,
    FULL_CHUNKS_: cutlass.Constexpr[int] = 0,
    LOWER_BOUND_M5_: cutlass.Constexpr[int] = 0,
    WARM_SPLIT_: cutlass.Constexpr[int] = 0,
    PREP_PEEL_: cutlass.Constexpr[int] = 0,
    GATE_ROLL_: cutlass.Constexpr[int] = 0,
    CLUSTER_: cutlass.Constexpr[int] = 1,
):
    mma_1 = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (BM, C, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
    )
    mma_3 = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (BM, C, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
    )
    mma_4 = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (BM, NFT, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
        )
    )
    # v91 split-MMA4: master leg (N=128) commits FIN for the next decay
    # before the OUT leg (N=32, commits OFIN + SFREE); junk pad gone.
    mma_4a = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (BM, D, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
        )
    )
    mma_4b = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (BM, C, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
        )
    )
    # prep gram MMA: D[64,32] = (qd||kd)[64,128] @ ki[32,128]^T, SS operands
    mma_g = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (2 * C, C, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
    )
    lay_qd = sm100_utils.make_smem_layout_b(mma_1, (BM, C, D), cutlass.BFloat16, 1)
    inv_shape = mma_3.partition_shape_B(cute.dice((BM, C, C), (None, 1, 1)))
    inv_atom = sm100_utils.make_smem_layout_atom(
        tcgen05.SmemLayoutAtomKind.K_INTER, cutlass.BFloat16
    )
    lay_inv = sm100_utils.tile_to_mma_shape(
        inv_atom, cute.append(inv_shape, 1), order=(1, 2, 3)
    )
    lay_ft = sm100_utils.make_smem_layout_b(mma_4, (BM, NFT, C), cutlass.BFloat16, 1)
    lay_v0 = sm100_utils.make_smem_layout_a(
        mma_1, (BM, C, C), cutlass.BFloat16, STAGES, is_k_major=False
    )
    lvo = lay_v0.outer
    lay_v = cute.make_composed_layout(
        lay_v0.inner,
        0,
        cute.make_layout(
            (
                lvo.shape[0][0],
                lvo.shape[1],
                (lvo.shape[0][1], lvo.shape[2]),
                lvo.shape[3],
            ),
            stride=(
                lvo.stride[0][0],
                lvo.stride[1],
                (lvo.stride[0][1], lvo.stride[2]),
                STAGE_ELTS,
            ),
        ),
    )
    l3k = cute.make_composed_layout(
        lay_qd.inner,
        0,
        cute.make_layout((C, 1, (16, 4, 2)), stride=(64, 0, (1, 16, 2048))),
    )
    # k lands in the kd rows of the interleaved [64,128] block
    l3k_k = cute.make_composed_layout(
        lay_qd.inner,
        0,
        cute.make_layout((C, 1, (16, 4, 2)), stride=(64, 0, (1, 16, 4096))),
    )
    lg_raw = cute.make_layout((C, 1, D), stride=(D, 0, 1))
    # g stages plain in the qd byte-positions (two 4 KiB halves)
    lg_qk = cute.make_layout((C, 1, (64, 2)), stride=(64, 0, (1, 4096)))
    tma_q, mQ = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), q, l3k, (C, 1, D)
    )
    tma_k, mK = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), k, l3k_k, (C, 1, D)
    )
    tma_g, mG = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), g, lg_qk, (C, 1, D)
    )
    lb_raw = cute.make_layout((C, 8), stride=(8, 1))
    tma_b, mB = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), beta, lb_raw, (C, 8)
    )
    if cutlass.const_expr(GATE2_ == 1):
        vT = cute.make_tensor(
            v.iterator, cute.make_layout((D, H_, v.shape[0]), stride=(1, D, H_ * D))
        )
        lv = cute.select(lay_v, mode=[0, 1, 2])
        tma_v, mV = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), vT, lv, (D, 1, C)
        )
    else:
        tma_v, mV = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), v, lg_raw, (C, 1, D)
        )
    tma_o, mO = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), out, l3k, (C, 1, D)
    )
    le_flat = cute.make_layout((64, 1, 256), stride=(256, 0, 1))
    tma_e, mE = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), exp_ws, le_flat, (64, 1, 256)
    )
    _pkd(
        mma_1,
        mma_3,
        mma_4a,
        mma_4b,
        mma_g,
        tma_q,
        mQ,
        tma_k,
        mK,
        tma_g,
        mG,
        tma_b,
        mB,
        tma_v,
        mV,
        tma_o,
        mO,
        tma_e,
        mE,
        q,
        k,
        g,
        beta,
        a_log,
        dt_bias,
        out,
        state0,
        stateT,
        cu,
        soff,
        schain,
        spt0,
        sptn,
        ssrc,
        sdst,
        midstate,
        mflags,
        expt,
        tprobe,
        failure_addr,
        have_state,
        do_export,
        export_seq,
        nc2,
        fepoch,
        scale,
        lb2,
        lay_qd,
        lay_inv,
        lay_ft,
        lay_v,
        H_,
        TPROBE_,
        GATE2_,
        FINAL_,
        SPLIT_,
        EXPORT_,
        STATE_,
        BETA_TMA_,
        BF16_DV_,
        BETA_BF16_,
        QK_ROWPAIR_,
        RCP_DECOR_,
        RF_HOIST_,
        RESTORE_TAIL_,
        GRAM_W9_,
        REG_MODE_,
        FULL_CHUNKS_,
        LOWER_BOUND_M5_,
        WARM_SPLIT_,
        PREP_PEEL_,
        GATE_ROLL_,
    ).launch(
        grid=(gcnt, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(CLUSTER_, 1, 1),
        stream=stream,
    )
