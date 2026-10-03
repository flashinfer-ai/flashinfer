# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""sm120 (GeForce/consumer Blackwell, CC 12.x) MoE grouped matmul fwd: grouped
persistent scheduler + warp-level MMA.

Per routed group ``g``: ``out[fto[g]:fto[g+1]] = token[range] @ weight[g % E].T``.
The token tensor is one flat K-major ``(S, K)`` matrix whose group boundaries
live in the runtime ``first_token_offset`` vector, so the tile space is
irregular and only the device can enumerate it. A persistent grid of
``grid_num_clusters`` CTAs claims tiles through one global atomic counter (in
``a_tma_workspace``); the scheduler warp of each CTA maps the claimed linear
index onto ``(group, tile_m, tile_n)`` with a warp-parallel prefix scan over
the group sizes and publishes the record through a 2-stage SMEM ring that the
TMA warp and every compute warp consume once.
In GATHER mode, ``token_index`` maps each routed row to a source token; the
source and routed extents are independent.

KEEP IN SYNC WITH ``sm120_matmul.py`` (this tree) and
``../../sm100/kernel_templates/sm100_moe_grouped_matmul_fwd.py``
-----------------------------------------------------------------------
* The mainloop (TMA -> swizzled SMEM -> ldmatrix -> ``mma.sync``), the
  transposed-STG epilogue staging and the register budget are the dense sm120
  kernel's, verbatim; only the tile source changed (scheduler ring instead of
  the launch grid + CLC), and A is K-major only (a MoE token is).
* The scheduler (atomic claim, ``_moe_group_at`` visitation order, the
  ``shfl``/``ballot`` prefix scan, the per-group L2 swizzle) is the sm100 MoE
  kernel's with the cluster broadcast layer removed: CC 12.x has no clusters,
  so every CTA is its own leader and the claimed index needs no DSM broadcast.

What is deliberately NOT here (vs. the sm100 MoE kernel)
-------------------------------------------------------
* No per-group TMA descriptor replacement. In NONE mode A is addressed by COORDINATE on one
  global ``[K, S]`` descriptor: a tile of group ``g`` loads rows
  ``group_begin + tile_m * cta_m ..`` and the ragged tail rows past
  ``group_end`` (the next group's tokens, or hardware zero-fill past ``S``)
  land in accumulator rows the epilogue never stores (``row < group_end``).
  That removes ``tensormap.replace``, the proxy fences and the per-CTA
  descriptor scratch; the workspace holds only the scheduler counter
  (``moe_desc_slots = 0``, so the compiler's counter offset agrees).
  GATHER uses a global source descriptor with one-row boxes and shared::cta
  GATHER4 loads. Group-tail indices are masked before reading token_index.
* No TMA-store epilogue: sm120 stores STG straight from registers, so the
  output needs no re-dimensioned descriptor either.

Multi-GEMM (e.g. SwiGLU: ``silu(A @ B0) * (A @ B1)``)
-------------------------------------------------------
One SMEM tile per DISTINCT operand per stage, one register accumulator per
GEMM (``gemm_a_idx`` / ``gemm_b_idx`` pick each GEMM's operands, as in the
sm100 template). Every GEMM's accumulators stay resident for the whole tile
(``num_gemms * _ACC_REGS`` fp32 per lane); past the compute warp's register
grant ptxas spills them to local memory -- a perf trade-off, never a reject.
The one hard gate is SMEM: the renderer sizes the ring with the epilogue
staging pre-funded (``Sm120KernelTemplate.multi_gemm_reject`` /
``.multi_gemm_ab_stages`` in kernel_registry). The STG epilogue stages the GEMMs' fragments
through the same warp-private buffer one after another and hands the fused
epilogue one fp32 vector per GEMM (``vec_f32``, ``vec_f32_1``, ...).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import cutlass.experimental.primitives as nvvm
# Inlined from cudnn.gemm.frost.tile_helpers
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Tile-level helpers shared by the SM100 and SM120 kernel templates.

A template is RENDERED (its `@@INJECT_*@@` blocks become module-level
constants) and then exec'd from the kernel cache under a synthetic module
name, so it cannot use relative imports and this module is never rendered.
Everything here therefore takes what it needs as ARGUMENTS -- a helper that
reads an injected constant (`mma_size_m`, `tile_swizzle_n`, `ab_dtype`, ...)
has to stay in the template, or receive it explicitly.

Scheduling and gather helpers serve both families. The tcgen05 wrappers are
used only by the SM100 family and retain its instruction-specific contracts.
"""


import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass._mlir.dialects import llvm






TENSOR_MAP_QWORDS = 16


def moe_swizzle_tile(t, nt_m, nt_n, swizzle_w):
    """Group-local linear tile index -> (m, n) under an N-super-block walk.
    ``swizzle_w == nt_n`` reproduces the plain n-fast split; ``1`` gives m-fast.
    """
    blk = cutlass.max(nt_m * swizzle_w, cutlass.Int32(1))
    sb = t // blk
    off = t - sb * blk
    base_n = sb * swizzle_w
    cur_S = cutlass.min(cutlass.Int32(swizzle_w), nt_n - base_n)
    tile_m = off // cur_S
    tile_n = base_n + off - tile_m * cur_S
    return tile_m, tile_n














@cute.jit
def moe_group_at(visit_idx, num_groups, num_experts):
    """Visitation index -> routed group index.

    ``num_groups == num_experts`` (or a non-multiple) walks groups in order. Batched MoE
    (``num_groups == B * num_experts``) walks expert-major -- the B groups sharing expert
    ``g % E`` become consecutive, so the expert weight is fetched once instead of B times.
    """
    per_expert = num_groups // cutlass.max(num_experts, cutlass.Int32(1))
    group = visit_idx
    if per_expert > 1 and per_expert * num_experts == num_groups:
        group = (visit_idx % per_expert) * num_experts + (visit_idx // per_expert)
    return group




@cute.jit
def moe_gather_row(token_index, row, group_end, source_rows):
    # Never read beyond the routed group. An out-of-range source row asks TMA
    # to zero-fill the padding while still completing the expected byte count.
    src = cutlass.Int32(source_rows)
    if row < group_end:
        src = cutlass.Int32(token_index[row])
    return src






@cute.jit
def tma_gather4(dst, desc, k, r0, r1, r2, r3, mbar, mask=None, cta_group: cutlass.Constexpr = 1):
    """Gather four rows from a rank-2 tensor map with box_dims[1] == 1.

    No mask selects shared::cta (SM120); a mask selects SM100 cluster multicast.
    The experimental DSL TMA wrapper currently validates two coordinates for
    this five-coordinate instruction, so issue the PTX directly.
    """
    bar_addr = mbar.data_ptr().toint(dtype=cutlass.Int32)
    if cutlass.const_expr(cta_group == 2):
        bar_addr = bar_addr & cutlass.Int32(0xFEFFFFFF)
    args = [
        dst.data_ptr().toint(dtype=cutlass.Int32).ir_value(),
        desc.toint().ir_value(),
        cutlass.Int32(k).ir_value(),
        cutlass.Int32(r0).ir_value(),
        cutlass.Int32(r1).ir_value(),
        cutlass.Int32(r2).ir_value(),
        cutlass.Int32(r3).ir_value(),
        bar_addr.ir_value(),
    ]
    if cutlass.const_expr(mask is None):
        llvm.inline_asm(
            None,
            args,
            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4." "mbarrier::complete_tx::bytes [$0], [$1, {$2, $3, $4, $5, $6}], [$7];",
            "r,l,r,r,r,r,r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        llvm.inline_asm(
            None,
            args + [cutlass.Int16(mask).ir_value()],
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4."
            "mbarrier::complete_tx::bytes.multicast::cluster."
            f"cta_group::{cta_group} [$0], [$1, {{$2, $3, $4, $5, $6}}], [$7], $8;",
            "r,l,r,r,r,r,r,r,h,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )











_moe_swizzle_tile = moe_swizzle_tile
_moe_group_at = moe_group_at
import cutlass.experimental.cuda.tensor_map as _tma
from cutlass import apply_swizzle as _apply_smem_swizzle
import cutlass
# Inlined from cudnn.gemm.frost.kernel_templates.dynamic_scheduler_counter_initialization
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cutlass
import cutlass.cute as cute


@cute.kernel
def dynamic_scheduler_counter_initialization(workspace: cute.Tensor, counter_qword: cutlass.Int32):
    counter = cute.make_tensor(
        cute.recast_ptr(workspace.iterator + counter_qword, dtype=cutlass.Int32),
        cute.make_layout(1),
    )
    counter[0] = cutlass.Int32(0)


dynamic_scheduler_counter_initialization.set_name_prefix("cudnn", remove_cutlass_symbol=True)

_dynamic_scheduler_counter_initialization = dynamic_scheduler_counter_initialization
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as _cuda

# @@FROST_GEOMETRY@@

if a_is_m_major:
    raise NotImplementedError(f"{__name__}: the MoE token is K-major only (the grouped A walk is a K-major TMA box)")

# A TMA tensormap is 128 bytes = 16 int64 qwords. The workspace is laid out as
# grid_ctas * moe_desc_slots tensormap slots followed by the scheduler counter;
# this kernel patches no descriptor, so its slot count is zero and the counter
# sits at the start of the buffer (the compiler carves it the same way). The
# host zeroes it, stream-ordered, before every launch -- as the sm100 MoE hosts
# do -- so the compiled launchable is self-contained: a caller that exports it
# (or replays it in a CUDA graph) need not reset the counter itself.
TENSOR_MAP_QWORDS = 16
moe_desc_slots = 0

# Scheduler ring depth and the i32 words per record.
SCHED_STAGES = 2
SCHED_SLOT_WORDS = 8

# Programmatic Dependent Launch (PDL, sm_90+; supported on sm_120).
USE_PDL = True

# Named barrier id for cross-warp sync of the compute warps (unused by the STG
# epilogue, kept for parity with the dense kernel's constants).
EPI_SYNC_BAR_ID = 1

# Compute-warp grid over the CTA tile (warp_row x warp_col), derived from the
# injected geometry: one warp tile = mma_size x the 16x16 warp-MMA pair, so
# the grid is cta_tile / warp_tile per axis.
WARPS_M = cta_tile_mnk[0] // (mma_size_m * mma_inst_shape_mnk[0])
WARPS_N = cta_tile_mnk[1] // (mma_size_n * mma_inst_shape_mnk[1])
NUM_COMPUTE_WARPS = WARPS_M * WARPS_N

TMA_WARP_ID = NUM_COMPUTE_WARPS
SCHEDULER_WARP_ID = NUM_COMPUTE_WARPS + 1
NUM_WARPS = threads_per_cta // 32

# Scheduler-ring consumers: every compute warp + the TMA producer each arrive
# once (elected) per consumed record.
NUM_SCHED_CONSUMER_WARPS = NUM_COMPUTE_WARPS + 1

EPI_REG_COUNT = 232
# One budget for the whole producer warpgroup (TMA, scheduler and donor warps).
# PTX setmaxnreg is .sync.aligned over the warpgroup: it is issued once, before
# the per-warp roles diverge, so the count cannot differ per warp.
PROD_REG_COUNT = 24
assert NUM_COMPUTE_WARPS % 4 == 0, "the producer warps must form whole warpgroups"

# ---------------------------------------------------------------------------
# Geometry derived from the injected tile constants (all plain Python ints —
# resolved at render/import time, traced as constants). Verbatim sm120_matmul.
# ---------------------------------------------------------------------------

_ELEM_BITS = ab_dtype.width
_ELEM_BYTES = _ELEM_BITS // 8
_ELEMS_16B = 16 // _ELEM_BYTES
# One k-block = 32 bytes of K = the K extent of one mma.sync (k16 for 16-bit,
# k32 for 8-bit operands).
_K_BLK_ELEMS = (32 * 8) // _ELEM_BITS
_NUM_K_BLOCKS = (cta_tile_mnk[2] * _ELEM_BITS) // (32 * 8)
_CTA_K_ELEMS = cta_tile_mnk[2]

_WARP_TILE_M = cta_tile_mnk[0] // WARPS_M
_WARP_TILE_N = cta_tile_mnk[1] // WARPS_N
_M_FRAGS = _WARP_TILE_M // 16
_N_FRAGS = _WARP_TILE_N // 8
_N_FRAG_PAIRS = _N_FRAGS // 2
_ACC_REGS = _M_FRAGS * _N_FRAGS * 4

# SMEM K-row swizzle: the K-row width IS the swizzle span (the renderer derives
# ab_tma_swizzle from cta_tile_k_bytes). The TMA s{128,64,32}b pattern ==
# cutlass.Swizzle(b, 4, 3) with b = log2(row_bytes / 16); ldmatrix addresses
# below apply the same XOR.
_AB_SMEM_SWIZZLE_BYTES = _CTA_K_ELEMS * _ELEM_BYTES
_AB_SW_BBITS = (_AB_SMEM_SWIZZLE_BYTES // 16).bit_length() - 1
_AB_SWIZZLE = cutlass.Swizzle(_AB_SW_BBITS, 4, 3)

# ---- Transposed STG epilogue staging (fort "Sheet3" scheme) -----------------
_STG_EPI_LANE_QUAD = 4  # one STS.128 = 4 x 32-bit acc regs per lane
_STG_EPI_PAD = 4  # 16B skew after each 128-element batch (sheet's X cells)
_STG_EPI_BATCH_STRIDE = 32 * _STG_EPI_LANE_QUAD + _STG_EPI_PAD  # 132
_STG_EPI_GROUP_FRAGS = 4  # fragments (= STS batches) per 32-column group
_STG_EPI_WARP_ELEMS = _STG_EPI_GROUP_FRAGS * _STG_EPI_BATCH_STRIDE  # 528
_STG_EPI_NGRP = (_N_FRAGS + _STG_EPI_GROUP_FRAGS - 1) // _STG_EPI_GROUP_FRAGS
_STG_V = (vec_bytes_epi * 8) // cd_dtype.width

_STG_EPI_BYTES = 4 * _STG_EPI_WARP_ELEMS * NUM_COMPUTE_WARPS
# One SMEM tile per DISTINCT operand per stage (multi-GEMM holds several B tiles).
_AB_STAGE_BYTES = (num_a_operands * cta_tile_mnk[0] + num_b_operands * cta_tile_mnk[1]) * _CTA_K_ELEMS * _ELEM_BYTES + 16
# Single-GEMM: the staging is funded by giving up whole AB stages (legacy
# accounting; the catalog sweep and every existing kernel count the same way).
# Multi-GEMM: the renderer took the staging off the budget in BYTES before
# sizing the ring (Sm120KernelTemplate.multi_gemm_ab_stages) -- a whole
# multi-operand stage may be all that fits -- so nothing is deducted here.
if not stg_epi_prefunded:
    ab_stages = ab_stages - -(-_STG_EPI_BYTES // _AB_STAGE_BYTES)


# ---------------------------------------------------------------------------
# The warp MMA instruction, resolved from the injected MMA dtypes.
# sm120 tensor cores are warp-scoped: mma.sync.aligned.m16n8k16 (16-bit A/B)
# or .m16n8k32 (8-bit A/B), row.col (both operands K-major), fp32/s32 acc.
# ---------------------------------------------------------------------------

_PTX_AB_TAG = {
    cutlass.BFloat16: "bf16",
    cutlass.Float16: "f16",
    cutlass.Float8E4M3FN: "e4m3",
    cutlass.Float8E5M2: "e5m2",
    cutlass.Int8: "s8",
}
assert mma_a_dtype in _PTX_AB_TAG and mma_b_dtype in _PTX_AB_TAG, f"unsupported sm120 MMA input dtypes: {mma_a_dtype} x {mma_b_dtype}"
_MMA_SHAPE = "m16n8k16" if _ELEM_BITS == 16 else "m16n8k32"
_MMA_C_TAG = "f32" if mma_c_dtype == cutlass.Float32 else "s32"
_MMA_PTX = (
    f"mma.sync.aligned.{_MMA_SHAPE}.row.col"
    f".{_MMA_C_TAG}.{_PTX_AB_TAG[mma_a_dtype]}.{_PTX_AB_TAG[mma_b_dtype]}.{_MMA_C_TAG} "
    "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};"
)


@cute.jit
def _mma_16x8_k32b(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3):
    """One warp-wide mma.sync on a (16, 8, 32-byte-K) fragment.

    A carrier: 4x b32 regs (ldmatrix.x4 of a [16 x 32B] K-major SMEM region).
    B carrier: 2x b32 regs. D/C: 4 accumulator regs (f32 or s32).
    """
    return cute.arch.inline_ptx(
        _MMA_PTX,
        write_only_types=[mma_c_dtype, mma_c_dtype, mma_c_dtype, mma_c_dtype],
        read_only_args=[a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
    )


# ---------------------------------------------------------------------------
# Group-local swizzle width depends on this template's injected constants.
# Tile mapping and group visitation are shared through tile_helpers.
# ---------------------------------------------------------------------------


@cute.jit
def _moe_auto_swizzle_w(group_rows, n, k, nt_n):
    """N-super-block width for one routed group, resolved per group.

    Same rule as the dense path, but the "M side" is THIS group's token slice, not the
    whole token tensor: block along the shorter of (group tokens, expert weight), capped
    by what L2 can hold onto. A group spanning one m-tile makes both orders identical.
    """
    if cutlass.const_expr(tile_swizzle_n > 0):
        return tile_swizzle_n
    budget = cutlass.Int64(swizzle_l2_budget_bytes)
    row_bytes = (cutlass.Int64(ab_dtype.width) * k) // 8
    cap = cutlass.max(budget // (row_bytes * cgrp_tile_mnk[1]), cutlass.Int64(1))
    w = cutlass.min(cutlass.Int64(nt_n), cap)
    rows = cutlass.Int64(group_rows)
    if cutlass.min(rows, n) * row_bytes <= budget and rows <= n:
        w = cutlass.Int64(1)
    return cutlass.Int32(w)


@cute.kernel
def frost_template_kernel(
    m: cutlass.Int64,
    n: cutlass.Int64,
    k: cutlass.Int64,
    num_experts: cutlass.Int32,
    num_groups: cutlass.Int32,
    first_token_offset: cute.Tensor,
    a_tma_workspace: cute.Tensor,
    tma_a_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_b_desc_0: cutlass.GridConstant[_tma.TensorMap],
    mC_tap_0: cute.Tensor,
    out_stride_m_0: cutlass.Int64,
    out_stride_n_0: cutlass.Int64,
    out_stride_l_0: cutlass.Int64,
) -> None:
    tma_a_descs = [tma_a_desc_0]
    tma_b_descs = [tma_b_desc_0]

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    elect_one = nvvm.elect_sync()

    tidx = cute.arch.thread_idx()[0]

    # The dynamic tile scheduler's global counter: the one word this kernel
    # keeps in the workspace (no descriptor slots precede it).
    sched_counter_ptr = cute.make_ptr(
        cutlass.Int32,
        (a_tma_workspace.iterator.raw_ptr() + grid_num_clusters * moe_desc_slots * TENSOR_MAP_QWORDS).toint(),
        mem_space=cute.AddressSpace.generic,
    )

    if warp_idx == TMA_WARP_ID:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)

    # Scheduler ring: one record per stage --
    # [0] expert (B's batch coordinate), [1] tile_m within the group, [2] tile_n,
    # [3] valid, [4] group_begin, [5] group_end, [7] routed group index.
    sched_storage = cutlass.Array(
        cutlass.Int32,
        SCHED_STAGES * SCHED_SLOT_WORDS,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    sched_full_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    sched_empty_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)

    sA_elems = cta_tile_mnk[0] * cta_tile_mnk[2]
    sB_elems = cta_tile_mnk[1] * cta_tile_mnk[2]
    smem_a_list = [
        cutlass.Array(
            ab_dtype,
            sA_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_b_list = [
        cutlass.Array(
            ab_dtype,
            sB_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_b_operands)
    ]

    # Per-compute-warp staging stream for the transposed STG epilogue (raw
    # accumulator dtype; 4 batches x (128 elems + 16B pad) = 528 elems, one
    # 32-column group of one m-frag at a time). Slices are warp-private, so
    # the round trip only needs bar.warp syncs — no CTA barrier.
    smem_stg_epi = cutlass.Array(
        mma_c_dtype,
        _STG_EPI_WARP_ELEMS * NUM_COMPUTE_WARPS,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )

    # ab full: one producer-elected arrive_expect_tx per stage.
    # ab empty: one elected arrive per compute warp per stage.
    # sched full: one elected arrive by the scheduler warp per record.
    # sched empty: one elected arrive per consumer warp (compute + TMA) per record.
    if warp_idx == 0:
        for i in range(ab_stages):
            if elect_one:
                nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), NUM_COMPUTE_WARPS)
        for i in range(SCHED_STAGES):
            if elect_one:
                nvvm.mbarrier_init(sched_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(sched_empty_mbar_ptr.subview(i), NUM_SCHED_CONSUMER_WARPS)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync(0)

    sA_bytes = sA_elems * (ab_dtype.width // 8)
    sB_bytes = sB_elems * (ab_dtype.width // 8)
    num_tma_copy_bytes = num_a_operands * sA_bytes + num_b_operands * sB_bytes

    gC_tap_0_ptr = mC_tap_0.iterator.raw_ptr()
    VEC_BYTES_TAP_0 = vec_bytes_tap_0

    VEC_BYTES = vec_bytes_epi
    vsize = epi_chunk_elems

    M = m
    N = n
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
    # Every group is cut into the same N tiling; only its M tiling is its own.
    tiles_along_n = cute.ceil_div(cutlass.Int32(N), cgrp_tile_mnk[1])
    first_token_arr = cutlass.make_array_view(first_token_offset)

    # -- Producer warpgroup ---------------------------------------------------
    # Warps NUM_COMPUTE_WARPS.. (TMA, scheduler, donors) are whole warpgroups:
    # release their registers once, warpgroup-uniformly, before the roles below
    # diverge per warp. Donor warps have no further role and simply exit.
    if warp_idx >= NUM_COMPUTE_WARPS:
        nvvm.setmaxregister(PROD_REG_COUNT, nvvm.SetMaxRegisterAction.DECREASE)

    # -- Grouped scheduler warp ------------------------------------------------
    # Claim the next GLOBAL linear tile index off the counter, locate the group
    # it falls in (a warp-parallel prefix scan over the group sizes, resumed
    # from the last hit -- claims only ever grow), split the group-local index
    # into (tile_m, tile_n) under the group's own L2 raster, and publish.
    if warp_idx == SCHEDULER_WARP_ID:
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        full_warp_mask = 0xFFFFFFFF
        shfl_idx_clamp = 0x1F
        shfl_up_clamp = 0
        lane = cute.arch.lane_idx()
        sched_stage = cutlass.Int32(0)
        sched_empty_phase = cutlass.Int32(1)
        linear_idx = cutlass.Int32(0)
        start_linear_idx = cutlass.Int32(0)
        total_tiles = cutlass.Int32(0)
        scan_base = cutlass.Int32(0)
        group_idx = cutlass.Int32(0)
        group_begin = cutlass.Int32(0)
        group_end = cutlass.Int32(0)
        is_tile_valid = cutlass.Int32(1)

        while is_tile_valid != 0:
            # Dynamic tile assignment: the CTAs live at any instant sit in one
            # contiguous window of tile space and share L2. No cluster, so the
            # claimed index is broadcast within the warp only.
            claimed = cutlass.Int32(0)
            if lane == 0:
                claimed = nvvm.atomicrmw(
                    "add",
                    sched_counter_ptr,
                    cutlass.Int32(1),
                    mem_order="relaxed",
                    syncscope="gpu",
                )
            linear_idx = nvvm.shfl_sync(full_warp_mask, claimed, 0, shfl_idx_clamp, nvvm.Shfl.IDX)
            if linear_idx >= start_linear_idx + total_tiles:
                is_search_live = cutlass.Int32(1)
                while is_search_live != 0:
                    visit_idx = scan_base + lane
                    my_group = _moe_group_at(visit_idx, num_groups, num_experts)
                    my_begin = cutlass.Int32(0)
                    my_end = cutlass.Int32(0)
                    my_tiles = cutlass.Int32(0)
                    if visit_idx < num_groups:
                        if my_group != 0:
                            my_begin = cutlass.Int32(first_token_arr[my_group])
                        my_end = cutlass.Int32(first_token_arr[my_group + 1])
                        my_tiles = cute.ceil_div(my_end - my_begin, cgrp_tile_mnk[0]) * tiles_along_n
                    prefix_tiles = my_tiles
                    for delta in (1, 2, 4, 8, 16):
                        prefix_delta = nvvm.shfl_sync(
                            full_warp_mask,
                            prefix_tiles,
                            delta,
                            shfl_up_clamp,
                            nvvm.Shfl.UP,
                        )
                        if lane >= delta:
                            prefix_tiles += prefix_delta
                    my_start = start_linear_idx + prefix_tiles - my_tiles
                    thread_succeed = nvvm.vote_sync(
                        full_warp_mask,
                        linear_idx < my_start + my_tiles,
                        nvvm.VoteSync.BALLOT,
                    )
                    if thread_succeed != 0:
                        winning_lane = cutlass.Int32(31) - cute.arch.bfind(cute.arch.brev(thread_succeed)).to(cutlass.Int32)
                        scan_base = nvvm.shfl_sync(full_warp_mask, visit_idx, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        group_idx = nvvm.shfl_sync(full_warp_mask, my_group, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        group_begin = nvvm.shfl_sync(full_warp_mask, my_begin, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        group_end = nvvm.shfl_sync(full_warp_mask, my_end, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        start_linear_idx = nvvm.shfl_sync(full_warp_mask, my_start, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        total_tiles = nvvm.shfl_sync(full_warp_mask, my_tiles, winning_lane, shfl_idx_clamp, nvvm.Shfl.IDX)
                        is_search_live = cutlass.Int32(0)
                    else:
                        start_linear_idx = nvvm.shfl_sync(
                            full_warp_mask,
                            my_start + my_tiles,
                            31,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        scan_base += 32
                        if scan_base >= num_groups:
                            is_tile_valid = cutlass.Int32(0)
                            is_search_live = cutlass.Int32(0)

            coord_expert = cutlass.Int32(0)
            tile_m = cutlass.Int32(0)
            tile_n = cutlass.Int32(0)
            if is_tile_valid != 0:
                local_linear_idx = linear_idx - start_linear_idx
                group_nt_m = total_tiles // tiles_along_n
                tile_m, tile_n = _moe_swizzle_tile(
                    local_linear_idx,
                    group_nt_m,
                    tiles_along_n,
                    _moe_auto_swizzle_w(group_nt_m * cgrp_tile_mnk[0], N, k, tiles_along_n),
                )
                coord_expert = group_idx % num_experts

            while not nvvm.mbarrier_try_wait_parity(
                sched_empty_mbar_ptr.subview(sched_stage),
                sched_empty_phase,
                time_limit=10_000_000,
            ):
                pass
            if lane == 0:
                slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
                (slot.subview(0)).store(coord_expert)
                (slot.subview(1)).store(tile_m)
                (slot.subview(2)).store(tile_n)
                (slot.subview(3)).store(is_tile_valid)
                (slot.subview(4)).store(group_begin)
                (slot.subview(5)).store(group_end)
                (slot.subview(7)).store(group_idx)
                nvvm.mbarrier_arrive(sched_full_mbar_ptr.subview(sched_stage))

            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_empty_phase = sched_empty_phase ^ 1

    # -- TMA producer warp ----------------------------------------------------
    if warp_idx == TMA_WARP_ID:
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        ab_empty_phase_bit = cutlass.Int32(1)
        ab_iter = cutlass.Int32(0)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        while is_valid != 0:
            while not nvvm.mbarrier_try_wait_parity(
                sched_full_mbar_ptr.subview(sched_stage),
                sched_full_phase,
                time_limit=10_000_000,
            ):
                pass
            slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
            coord_expert = (slot.subview(0)).load()
            tile_m = (slot.subview(1)).load()
            tile_n = (slot.subview(2)).load()
            is_valid = (slot.subview(3)).load()
            group_begin = (slot.subview(4)).load()
            group_end = (slot.subview(5)).load()
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

            if is_valid != 0:
                # Routed tile origin. GATHER maps these rows to the source
                # tensor; NONE addresses them directly in the global descriptor.
                coord_m = group_begin + tile_m * cgrp_tile_mnk[0]
                coord_n = tile_n * cgrp_tile_mnk[1]

                for k_tile_idx in range(num_k_tiles):
                    stage = ab_iter % ab_stages
                    if stage == 0 and ab_iter != 0:
                        ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(stage), ab_empty_phase_bit, time_limit=10_000_000):
                        pass

                    coord_k = k_tile_idx * cta_tile_mnk[2]
                    # One elected lane only: the barrier's arrival count is 1, and
                    # the TMA copies deliver exactly num_tma_copy_bytes once.
                    if elect_one:
                        nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        if cutlass.const_expr(moe_gather):
                            for _am in cutlass.range(cta_tile_mnk[0] // 4, unroll_full=True):
                                if elect_one:
                                    row = coord_m + _am * 4
                                    r0 = moe_gather_row(token_index, row, group_end, source_rows)
                                    r1 = moe_gather_row(token_index, row + 1, group_end, source_rows)
                                    r2 = moe_gather_row(token_index, row + 2, group_end, source_rows)
                                    r3 = moe_gather_row(token_index, row + 3, group_end, source_rows)
                                    tma_gather4(
                                        smem_a_list[_ai].subview(sA_elems * stage + _am * 4 * _CTA_K_ELEMS),
                                        tma_a_descs[_ai].get_ptr(),
                                        coord_k,
                                        r0,
                                        r1,
                                        r2,
                                        r3,
                                        ab_full_mbar_ptr.subview(stage),
                                    )
                        else:
                            # K-major box; hardware zero-fills K tails.
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cta_global(
                                    smem_a_list[_ai].subview(sA_elems * stage),
                                    tma_a_descs[_ai].get_ptr(),
                                    (coord_k, coord_m, cutlass.Int32(0)),
                                    ab_full_mbar_ptr.subview(stage),
                                )
                    # The expert's weight is B's batch coordinate. K-major B: box
                    # [K_tile, cta_n] at (k, n, e); N-major B walks N in
                    # b_tma_group_elems-wide groups (same row bytes as a K-major row).
                    for _bj in cutlass.range_constexpr(num_b_operands):
                        if cutlass.const_expr(b_is_n_major):
                            for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cta_global(
                                        smem_b_list[_bj].subview(sB_elems * stage + n_group * b_tma_group_elems * _CTA_K_ELEMS),
                                        tma_b_descs[_bj].get_ptr(),
                                        (coord_n + n_group * b_tma_group_elems, coord_k, coord_expert),
                                        ab_full_mbar_ptr.subview(stage),
                                    )
                        else:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cta_global(
                                    smem_b_list[_bj].subview(sB_elems * stage),
                                    tma_b_descs[_bj].get_ptr(),
                                    (coord_k, coord_n, coord_expert),
                                    ab_full_mbar_ptr.subview(stage),
                                )
                    ab_iter += 1

    # -- Compute warps: mma.sync mainloop + epilogue --------------------------
    if warp_idx < NUM_COMPUTE_WARPS:
        nvvm.setmaxregister(EPI_REG_COUNT, nvvm.SetMaxRegisterAction.INCREASE)
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")

        lane = tidx % 32
        lane_div4 = lane // 4
        lane_mod4 = lane % 4
        warp_row = warp_idx % WARPS_M
        warp_col = warp_idx // WARPS_M

        # ldmatrix lane->address maps (see PTX ldmatrix; addresses are 16B rows).
        # A x4 tile order = (rows 0-7, rows 8-15) x (16B col 0, 16B col 1) —
        # matching the a0..a3 fragment order of mma.sync.
        a_ldm_row = (lane % 8) + 8 * ((lane // 8) % 2)
        a_ldm_col16 = lane // 16
        # B x4 covers TWO 8-col n-frags: (n rows 0-7, n rows 8-15) each split
        # over (16B col 0, 16B col 1) -> regs (b0,b1) frag0 + (b0,b1) frag1.
        b_ldm_pair_row = (lane % 8) + 8 * (lane // 16)
        b_ldm_pair_col16 = (lane // 8) % 2
        # B x2 tail: one n-frag (rows 0-7 x two 16B cols; lanes 16-31 unused).
        b_ldm_tail_row = lane % 8
        b_ldm_tail_col16 = (lane // 8) % 2
        # Transposed (N-major SMEM) maps for the b16 form: rows run along K,
        # 16B units along N. (The 8-bit m16n16.trans.b8 form needs no map: its
        # two tiles' 16 k-row addresses are simply k = kb_base + lane.)
        bt_ldm_k = (lane % 8) + 8 * ((lane // 8) % 2)
        bt_ldm_n8 = lane // 16

        # One register accumulator per GEMM of the chain (multi-GEMM: past the
        # warp's register grant ptxas spills -- a perf trade-off, never a gate).
        acc_list = [cutlass.Array(mma_c_dtype, _ACC_REGS, alignment=16) for _g in range(num_gemms)]

        ab_full_phase_bit = cutlass.Int32(0)
        ab_iter = cutlass.Int32(0)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)
        # The routed output is one flat (S, N) surface: no batch term.
        tile_l = cutlass.Int32(0)

        while not nvvm.mbarrier_try_wait_parity(sched_full_mbar_ptr.subview(sched_stage), sched_full_phase, time_limit=10_000_000):
            pass
        _slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
        tile_m = (_slot.subview(1)).load()
        tile_n = (_slot.subview(2)).load()
        is_valid = (_slot.subview(3)).load()
        group_begin = (_slot.subview(4)).load()
        group_end = (_slot.subview(5)).load()
        group_idx = (_slot.subview(7)).load()
        nvvm.bar_warp_sync(0xFFFFFFFF)
        if elect_one:
            nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
        sched_stage += 1
        if sched_stage == SCHED_STAGES:
            sched_stage = cutlass.Int32(0)
            sched_full_phase = sched_full_phase ^ 1

        while is_valid != 0:
            coord_m = group_begin + tile_m * cgrp_tile_mnk[0]
            coord_n = tile_n * cgrp_tile_mnk[1]

            for _g in cutlass.range_constexpr(num_gemms):
                _acc_g = acc_list[_g]
                for _z in cutlass.range_constexpr(_ACC_REGS):
                    _acc_g[_z] = mma_c_dtype(0)

            for k_tile_idx in range(num_k_tiles):
                stage = ab_iter % ab_stages
                if stage == 0 and ab_iter != 0:
                    ab_full_phase_bit = ab_full_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(ab_full_mbar_ptr.subview(stage), ab_full_phase_bit, time_limit=10_000_000):
                    pass

                sA_ptrs = [smem_a_list[_ai].subview(sA_elems * stage).data_ptr() for _ai in range(num_a_operands)]
                sB_ptrs = [smem_b_list[_bj].subview(sB_elems * stage).data_ptr() for _bj in range(num_b_operands)]

                for k_blk in cutlass.range_constexpr(_NUM_K_BLOCKS):
                    kb_base = k_blk * _K_BLK_ELEMS
                    a_frags_list = []
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        sA_ptr = sA_ptrs[_ai]
                        a_frags = []
                        for mf in cutlass.range_constexpr(_M_FRAGS):
                            a_row = warp_row * _WARP_TILE_M + mf * 16 + a_ldm_row
                            a_off = a_row * _CTA_K_ELEMS + kb_base + a_ldm_col16 * _ELEMS_16B
                            a_frags.append(
                                nvvm.ldmatrix(
                                    _apply_smem_swizzle(sA_ptr + a_off, _AB_SWIZZLE),
                                    4,
                                    nvvm.MMALayout.ROW,
                                )
                            )
                        a_frags_list.append(a_frags)
                    for _bj in cutlass.range_constexpr(num_b_operands):
                        sB_ptr = sB_ptrs[_bj]
                        b_frags = []
                        if cutlass.const_expr(b_is_n_major and _ELEM_BITS == 8):
                            # ldmatrix.m16n16.x2.trans.b8 per n-frag pair: the tile's 16
                            # transposed columns span n-frags (2p, 2p+1), so the result
                            # regs are [b0(2p), b0(2p+1), b1(2p), b1(2p+1)]. Addresses:
                            # k = kb_base + lane, no lane map. (_N_FRAGS is even here.)
                            for npair in cutlass.range_constexpr(_N_FRAG_PAIRS):
                                b_n = warp_col * _WARP_TILE_N + npair * 16
                                b_off = (
                                    (b_n // b_tma_group_elems) * (b_tma_group_elems * _CTA_K_ELEMS)
                                    + (kb_base + lane) * b_tma_group_elems
                                    + b_n % b_tma_group_elems
                                )
                                bv = nvvm.ldmatrix(
                                    _apply_smem_swizzle(sB_ptr + b_off, _AB_SWIZZLE),
                                    4,
                                    nvvm.MMALayout.COL,
                                    shape=nvvm.LoadShape.M16N16,
                                    src_format=nvvm.LoadSrcFormat.B8,
                                )
                                b_frags.append((bv[0], bv[2]))
                                b_frags.append((bv[1], bv[3]))
                        elif cutlass.const_expr(b_is_n_major):
                            for npair in cutlass.range_constexpr(_N_FRAG_PAIRS):
                                b_n = warp_col * _WARP_TILE_N + npair * 16 + bt_ldm_n8 * 8
                                b_off = (
                                    (b_n // b_tma_group_elems) * (b_tma_group_elems * _CTA_K_ELEMS)
                                    + (kb_base + bt_ldm_k) * b_tma_group_elems
                                    + b_n % b_tma_group_elems
                                )
                                bv = nvvm.ldmatrix(
                                    _apply_smem_swizzle(sB_ptr + b_off, _AB_SWIZZLE),
                                    4,
                                    nvvm.MMALayout.COL,
                                )
                                b_frags.append((bv[0], bv[1]))
                                b_frags.append((bv[2], bv[3]))
                            if cutlass.const_expr(_N_FRAGS % 2 == 1):
                                b_n = warp_col * _WARP_TILE_N + (_N_FRAGS - 1) * 8
                                b_off = (
                                    (b_n // b_tma_group_elems) * (b_tma_group_elems * _CTA_K_ELEMS)
                                    + (kb_base + bt_ldm_k) * b_tma_group_elems
                                    + b_n % b_tma_group_elems
                                )
                                bt = nvvm.ldmatrix(
                                    _apply_smem_swizzle(sB_ptr + b_off, _AB_SWIZZLE),
                                    2,
                                    nvvm.MMALayout.COL,
                                )
                                b_frags.append((bt[0], bt[1]))
                        else:
                            for npair in cutlass.range_constexpr(_N_FRAG_PAIRS):
                                b_row = warp_col * _WARP_TILE_N + npair * 16 + b_ldm_pair_row
                                b_off = b_row * _CTA_K_ELEMS + kb_base + b_ldm_pair_col16 * _ELEMS_16B
                                bv = nvvm.ldmatrix(
                                    _apply_smem_swizzle(sB_ptr + b_off, _AB_SWIZZLE),
                                    4,
                                    nvvm.MMALayout.ROW,
                                )
                                b_frags.append((bv[0], bv[1]))
                                b_frags.append((bv[2], bv[3]))
                            if cutlass.const_expr(_N_FRAGS % 2 == 1):
                                b_row = warp_col * _WARP_TILE_N + (_N_FRAGS - 1) * 8 + b_ldm_tail_row
                                b_off = b_row * _CTA_K_ELEMS + kb_base + b_ldm_tail_col16 * _ELEMS_16B
                                bt = nvvm.ldmatrix(
                                    _apply_smem_swizzle(sB_ptr + b_off, _AB_SWIZZLE),
                                    2,
                                    nvvm.MMALayout.ROW,
                                )
                                b_frags.append((bt[0], bt[1]))

                        # Every GEMM fed by this B operand (trace-time selection).
                        for _g in cutlass.range_constexpr(num_gemms):
                            if cutlass.const_expr(gemm_b_idx[_g] == _bj):
                                _acc_g = acc_list[_g]
                                a_frags = a_frags_list[gemm_a_idx[_g]]
                                for mf in cutlass.range_constexpr(_M_FRAGS):
                                    av = a_frags[mf]
                                    for nf in cutlass.range_constexpr(_N_FRAGS):
                                        b0, b1 = b_frags[nf]
                                        _o = (mf * _N_FRAGS + nf) * 4
                                        _acc_g[_o:4] = _mma_16x8_k32b(
                                            av[0],
                                            av[1],
                                            av[2],
                                            av[3],
                                            b0,
                                            b1,
                                            _acc_g[_o + 0],
                                            _acc_g[_o + 1],
                                            _acc_g[_o + 2],
                                            _acc_g[_o + 3],
                                        )
                nvvm.bar_warp_sync(0xFFFFFFFF)
                cute.arch.fence_proxy("async.shared", space="cta")
                if elect_one:
                    nvvm.mbarrier_arrive(ab_empty_mbar_ptr.subview(stage))
                ab_iter += 1

            # -- Epilogue: accumulators are already in registers ------------------

            pass

            _stg_stage = smem_stg_epi.subview(warp_idx * _STG_EPI_WARP_ELEMS)
            for mf in cutlass.range_constexpr(_M_FRAGS):
                for grp in cutlass.range_constexpr(_STG_EPI_NGRP):
                    _nf0 = grp * _STG_EPI_GROUP_FRAGS
                    _grp_frags = min(_STG_EPI_GROUP_FRAGS, _N_FRAGS - _nf0)
                    # -- STS_128: reg-index-order dump, one batch per fragment; then
                    # LDS: 16 contiguous elems = both row-halves of one frag. One GEMM
                    # at a time through the same warp-private buffer (multi-GEMM keeps
                    # every GEMM's segment in registers, not extra SMEM).
                    _seg_list = []
                    for _g in cutlass.range_constexpr(num_gemms):
                        _acc_g = acc_list[_g]
                        for b in cutlass.range_constexpr(_grp_frags):
                            _o = (mf * _N_FRAGS + _nf0 + b) * 4
                            _s_off = b * _STG_EPI_BATCH_STRIDE + lane * _STG_EPI_LANE_QUAD
                            (_stg_stage.data_ptr() + _s_off).store(_acc_g[_o:4], alignment=16)
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                        _seg_list.append((_stg_stage.data_ptr() + lane_mod4 * _STG_EPI_BATCH_STRIDE + lane_div4 * 16).load(alignment=16, count=16))
                        if cutlass.const_expr(_g + 1 < num_gemms):
                            nvvm.bar_warp_sync(0xFFFFFFFF)  # every lane has its segment before the buffer is reused
                    # Short tail group: trailing lanes own no fragment there
                    # (True at trace time for full groups — no guard emitted).
                    _lane_active = True if _grp_frags == _STG_EPI_GROUP_FRAGS else lane_mod4 < _grp_frags
                    if _lane_active:
                        for half in cutlass.range_constexpr(2):
                            row_in_cta = warp_row * _WARP_TILE_M + mf * 16 + half * 8 + lane_div4
                            row = coord_m + row_in_cta
                            # The ragged tail of a group: rows at or past group_end
                            # hold the next group's tokens (or zero-fill) and are
                            # not this expert's output.
                            if row < group_end:
                                _row_list = []
                                for _g in cutlass.range_constexpr(num_gemms):
                                    _seg = _seg_list[_g]
                                    _row = cutlass.Array(mma_c_dtype, 8, alignment=16)
                                    for sj in cutlass.range_constexpr(4):
                                        _row[2 * sj] = _seg[4 * sj + 2 * half]
                                        _row[2 * sj + 1] = _seg[4 * sj + 2 * half + 1]
                                    _row_list.append(_row)
                                for sv in cutlass.range_constexpr(8 // _STG_V):
                                    col = coord_n + warp_col * _WARP_TILE_N + (_nf0 + lane_mod4) * 8 + sv * _STG_V
                                    col_j = col
                                    if col_j + vsize <= N:
                                        # One epilogue input vector per GEMM: GEMM 0 is the
                                        # template's `vec_f32`, GEMMs > 0 are bound as
                                        # `vec_f32_<g>` by the injected STG bindings below.
                                        # NB: Array slices are [start:COUNT], not
                                        # [start:stop] (matches acc[_o:2] above).
                                        c_rmem_vecs = []
                                        for _g in cutlass.range_constexpr(num_gemms):
                                            _vec = _row_list[_g][sv * _STG_V : _STG_V]
                                            if cutlass.const_expr(acc_widen_to_fp32):
                                                _pf = _vec.to(cutlass.Float32)
                                                c_rmem_vecs.append(_pf + cutlass.full_like(_pf, 0.0))
                                            else:
                                                c_rmem_vecs.append(_vec)
                                        vec_f32 = c_rmem_vecs[0]
                                        linear_idx = tile_l * out_stride_l_0 + row * out_stride_m_0 + col_j * out_stride_n_0

                                        pass

                                        _r_mm = (vec_f32).to(cutlass.BFloat16)
                                        _tap_0 = (_r_mm).to(cutlass.BFloat16)
                                        (gC_tap_0_ptr + (row * out_stride_m_0 + col_j)).store(_tap_0, alignment=VEC_BYTES_TAP_0)
                    nvvm.bar_warp_sync(0xFFFFFFFF)

            # Next record.
            while not nvvm.mbarrier_try_wait_parity(sched_full_mbar_ptr.subview(sched_stage), sched_full_phase, time_limit=10_000_000):
                pass
            _slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
            tile_m = (_slot.subview(1)).load()
            tile_n = (_slot.subview(2)).load()
            is_valid = (_slot.subview(3)).load()
            group_begin = (_slot.subview(4)).load()
            group_end = (_slot.subview(5)).load()
            group_idx = (_slot.subview(7)).load()
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

        # No more tiles for this CTA: all its global A/B reads have been issued.
        if cutlass.const_expr(USE_PDL):
            if warp_idx == 0:
                if elect_one:
                    nvvm.griddepcontrol("launch_dependents")


frost_template_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    problem_size: tuple,
    first_token_offset: cute.Tensor,
    a_tma_workspace: cute.Tensor,
    a_0: cute.Tensor,
    b_0: cute.Tensor,
    c_tap_0: cute.Tensor,
    stream: _cuda.CUstream,
) -> None:
    _a_operands = [a_0]
    _b_operands = [b_0]

    m = problem_size[0]
    n = problem_size[1]
    k_sym = problem_size[2]
    num_experts = problem_size[3]
    num_groups = problem_size[4]
    _stride_idx = 5
    _a_stride_sets = []
    for _ in cutlass.range_constexpr(num_a_operands):
        _a_stride_sets.append(
            (
                problem_size[_stride_idx],
                problem_size[_stride_idx + 1],
                problem_size[_stride_idx + 2],
            )
        )
        _stride_idx += 3
    _b_stride_sets = []
    for _ in cutlass.range_constexpr(num_b_operands):
        _b_stride_sets.append(
            (
                problem_size[_stride_idx],
                problem_size[_stride_idx + 1],
                problem_size[_stride_idx + 2],
            )
        )
        _stride_idx += 3

    out_stride_m_0 = problem_size[_stride_idx]
    out_stride_n_0 = problem_size[_stride_idx + 1]
    out_stride_l_0 = problem_size[_stride_idx + 2]
    _stride_idx += 3

    # GATHER uses the original source extent and one-row boxes. NONE walks
    # whole tiles in the routed token matrix.
    tma_a_desc_list = []
    for _a_idx, _a_op in enumerate(_a_operands):
        a_stride_m, a_stride_k, a_stride_l = _a_stride_sets[_a_idx]
        if cutlass.const_expr(moe_gather):
            a_dims = [k_sym, _a_op.shape[0]]
            a_strides = [a_stride_m * ab_dtype.width // 128]
            a_box = [cta_tile_mnk[2], 1]
        else:
            a_dims = [k_sym, m, 1]
            a_strides = [a_stride_m * ab_dtype.width // 128, a_stride_l * ab_dtype.width // 128]
            a_box = [cta_tile_mnk[2], cta_tile_mnk[0], 1]
        tma_a_desc_list.append(
            _tma.create_tensor_map_tiled(
                global_address=_a_op.iterator.toint(),
                dtype=ab_tma_dtype,
                global_dims=a_dims,
                global_strides=a_strides,
                box_dims=a_box,
                swizzle=ab_tma_swizzle,
            )
        )
    # B is batched by expert. K-major: box [K_tile, cta_n]. N-major:
    # [group_elems, K_tile] boxes, one per N group (the group row bytes equal a
    # K-major row's, so both majors share ab_tma_swizzle).
    tma_b_desc_list = []
    for _b_idx, _b_op in enumerate(_b_operands):
        b_stride_n, b_stride_k, b_stride_l = _b_stride_sets[_b_idx]
        if cutlass.const_expr(b_is_n_major):
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[n, k_sym, num_experts],
                    global_strides=[
                        b_stride_k * ab_dtype.width // 128,
                        b_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[b_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=ab_tma_swizzle,
                )
            )
        else:
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[k_sym, n, num_experts],
                    global_strides=[
                        b_stride_n * ab_dtype.width // 128,
                        b_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[1], 1],
                    swizzle=ab_tma_swizzle,
                )
            )

    # Persistent grid: as many CTAs as the device co-schedules; every CTA pulls
    # tiles off the global counter until the group space is exhausted. No
    # cluster launch on sm120 (CC 12.x has no thread-block clusters).
    grid_shape = (grid_num_clusters, 1, 1)
    # Zero the scheduler counter on the launch stream. The PDL main kernel
    # below reads it only after griddepcontrol.wait, i.e. once this has landed.
    counter_qword = grid_num_clusters * moe_desc_slots * TENSOR_MAP_QWORDS
    _dynamic_scheduler_counter_initialization(a_tma_workspace, cutlass.Int32(counter_qword)).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
    frost_template_kernel(
        problem_size[0],
        problem_size[1],
        problem_size[2],
        cutlass.Int32(num_experts),
        cutlass.Int32(num_groups),
        first_token_offset,
        a_tma_workspace,
        tma_a_desc_list[0],
        tma_b_desc_list[0],
        c_tap_0,
        out_stride_m_0,
        out_stride_n_0,
        out_stride_l_0,
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        use_pdl=USE_PDL,
        stream=stream,
    )


@lru_cache(maxsize=None)
def compile() -> Callable:
    out_vec_elems = vec_bytes_epi // (cd_dtype.width // 8)
    ab_stride_elems = 16 // (ab_dtype.width // 8)
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=out_vec_elems)
    # K tails are supported: the K loop is ceil_div and the TMA descriptor's global K
    # extent makes a partial box HW zero-filled. The only real K rule is the 16-byte
    # TMA contiguous-extent one, already gated by _tma_alignment_reject.
    sym_k = cute.sym_int64()
    sym_e = cute.sym_int64()
    sym_g = cute.sym_int64()
    sym_source_m = cute.sym_int64() if moe_gather else sym_m

    def _make_fake_a():
        return make_fake_compact_tensor(
            mma_a_dtype,
            (sym_source_m, sym_k, 1),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            mma_b_dtype,
            (sym_n, sym_k, sym_e),
            stride_order=(0, 1, 2) if b_is_n_major else (1, 0, 2),
            assumed_align=16,
        )

    fake_first_token_offset = make_fake_compact_tensor(
        offset_cutlass_dtype,
        (cute.sym_int64(),),
        stride_order=(0,),
        assumed_align=offset_cutlass_dtype.width // 8,
    )
    # The compiler carves grid_ctas * moe_desc_slots tensormap slots plus one
    # counter slot (16 int64 each); with no descriptor slots that is the one
    # counter slot.
    fake_a_tma_workspace = make_fake_compact_tensor(
        cutlass.Int64,
        (grid_num_clusters * moe_desc_slots * TENSOR_MAP_QWORDS + TENSOR_MAP_QWORDS,),
        stride_order=(0,),
        assumed_align=128,
    )

    def _sym_operand_strides(is_mn_major: bool) -> tuple:
        # Operand is permuted to (M|N, K, L): the unit stride is mode 0 when MN-major, mode 1 when K-major, and never reaches TMA.
        unit = 0 if is_mn_major else 1
        return tuple(cute.sym_int64() if i == unit else cute.sym_int64(divisibility=ab_stride_elems) for i in range(3))

    sym_a_strides = []
    for _ in range(num_a_operands):
        sym_a_strides.extend(_sym_operand_strides(False))
    sym_b_strides = []
    for _ in range(num_b_operands):
        sym_b_strides.extend(_sym_operand_strides(b_is_n_major))
    sym_out_stride_m_0 = cute.sym_int64()
    sym_out_stride_n_0 = cute.sym_int64()
    sym_out_stride_l_0 = cute.sym_int64()
    fake_a_0 = _make_fake_a()
    fake_b_0 = _make_fake_b()
    fake_c_tap_0 = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (sym_m, sym_n, 1),
        stride=(cute.sym_int64(), 1, cute.sym_int64()),
        assumed_align=16,
    )
    problem_size = (
        sym_m,
        sym_n,
        sym_k,
        sym_e,
        sym_g,
        *sym_a_strides,
        *sym_b_strides,
        sym_out_stride_m_0,
        sym_out_stride_n_0,
        sym_out_stride_l_0,
    )
    pass
    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    return cute.compile(_host, problem_size, fake_first_token_offset, fake_a_tma_workspace, fake_a_0, fake_b_0, fake_c_tap_0, stream=_fake_stream, options=frost_compile_options)
