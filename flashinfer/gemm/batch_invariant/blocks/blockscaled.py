# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Block-scaled blocks: the dense blocks with scale-factor operands, for the MXFP8, NVFP4 and MXFP4 classes.

The classes ``gemm.mxfp8.kfold`` / ``gemm.mxfp8.kchunk[C]`` (the sm_100 / sm_103 instruction) and ``gemm.mxfp8.kfold64`` /
``gemm.mxfp8.kchunk64[C]`` (the sm_107 instruction) are the dense K fold with a different opaque step: one
``tcgen05.mma.kind::mxf8f6f4.block_scale`` instruction (K = 32 e4m3 elements on sm_100 / sm_103, 64 on sm_107) whose
inputs are the A and B slices *and* their UE8M0 scale factors (one per 32 elements, the 128x4 swizzled layout
:func:`flashinfer.mxfp8_quantize` produces). Nothing below the instruction is ours: the block scales are applied inside
it, never in software (``spec.kslice_leaf``). What this file adds to the dense blocks, all placements:

* :class:`BlockScaledPlan` (host): the block-scaled tiled MMA (the A2 surface: ``Knobs.instruction("mxfp8")``), the
  SFA / SFB smem layouts (one stage of scale factors next to each A / B stage), the stage counts with the scale factors
  in the smem budget, and the TMEM budget (accumulator stages + the scale-factor columns the instruction reads from
  TMEM); :class:`BlockScaledPlanK64` the same on the sm_107 instruction (two steps per 128-byte stage, the 576-column
  TMEM, the B-reuse record);
* :class:`BlockScaledTmaSetup` (host): TMA atoms for SFA / SFB on top of A / B / C (SFB through a 1-CTA tiled MMA whose
  N is the 128-row scale-factor atom; the 192-wide tile re-addresses its SFB tile, the 64-wide tile shares one SFB tile
  between two N tiles);
* :func:`blockscaled_producer_loop`: the dense producer loading SFA and SFB with every A / B stage;
* :func:`blockscaled_mma_loop`: ``MainloopBlock.lowering()`` (``canon.BlockedFold``, the same object the dense mainloop
  executes) with the per-stage ``tcgen05.cp`` of the scale factors to TMEM and the SFA / SFB operands set on the
  instruction;
* :class:`BlockScaledGemm`: the kernel composing them with the dense epilogue block (the unscaled chain of
  ``mm_mxfp8``: one ``cvt.rn``; the in-CTA chunk fold of the split classes exactly as in the dense body);
  :class:`ScaledBlockScaledGemm` the same body with the alpha chain of the 4-bit classes (``mm_fp4``'s scalar: one
  ``mul.rn``, then the conversion), on byte-packed e2m1 operands viewed as fp4 (:func:`fp4_view`);
* the **in-cluster split** (``Knobs.cluster_k = C > 1``): the ``C`` CTAs of a ``(C, 1, 1)`` cluster each fold one K chunk
  (``k_tile0 = rank * k_tiles / C``: the class ``kchunk[C]``, exactly what the workspace split computes), and
  :func:`blockscaled_cluster_epilogue_loop` folds the fp32 partials **in chunk order from +0 with add.rn** through a
  distributed-shared-memory mailbox in the owner CTA of each epilogue subtile, then applies the chain. One tile per
  cluster (the persistent scheduler with as many clusters as tiles), so every mailbox and barrier is used once;
* the **swapped decode record** (``Knobs.swap_ab``): the caller hands the kernel the weights as A and the tokens as B
  with the transposed output (``api._run_mx``); the body is unchanged, the step is symmetric in its two operands.

Library utilities (CuTe DSL): ``make_blockscaled_trivial_tiled_mma`` (``utils.sm100`` / ``utils.rubin_helpers``),
``blockscaled_layout`` (``make_smem_layout_sfa/sfb``, ``make_tmem_layout_sfa/sfb``), ``cluster_shape_to_tma_atom_SFB``,
``Cp4x32x128bOp`` + ``make_s2t_copy``. The plumbing follows NVIDIA's CuTe-DSL block-scaled persistent example and
FlashInfer's derivative of it, digested, not copied; neither decides a numeric order.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as bs_layout
import cutlass.utils.rubin_helpers as rubin
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.tensor_utils import LayoutEnum
from cutlass.utils.blackwell_helpers import get_smem_store_op, get_tmem_load_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

# FlashInfer's copy of the DSL's persistent tile scheduler (the DSL's own module is deprecated and warns per use)
from flashinfer.cute_dsl.attention.fmha.helpers import (
    static_persistent_tile_scheduler as tile_scheduler,
)

from ....batch_invariant.fp_interface.cute import CuteF32
from ..generator import BREUSE_FMTS, Knobs, MainloopBlock
from ..spec import SF_ATOM_KBLOCKS, SF_ATOM_ROWS, SF_VEC, EpilogueChain, sf_bytes  # noqa: F401  (sf_bytes re-exported)
from .epilogue import apply_chain_pre_cvt, epilogue_loop
from .operands import (
    ACC_DTYPE,
    BAR_EPILOGUE,
    BAR_TMEM_ALLOC,
    EPILOGUE_WARPS,
    IN_DT,
    MBAR_RESERVE_BYTES,
    MMA_WARP,
    NUM_C_STAGE_MIN,
    OCCUPANCY,
    OUT_DT,
    THREADS,
    TMA_WARP,
    TMEM_ARCH,
    build_grid,
)

E8M0_DTYPE = cutlass.Float8E8M0FNU  # the MX block scale
E4M3_DTYPE = cutlass.Float8E4M3FN  # the NVFP4 block scale (and the MXFP8 element)
FP4_DTYPE = (
    cutlass.Float4E2M1FN
)  # the 4-bit element, two per byte in the operands the API hands over
SF_ATOM_BYTES = (
    SF_ATOM_ROWS * SF_ATOM_KBLOCKS
)  # one scale-factor atom: 128 rows x 4 K-blocks
TMEM_COLS = 512  # the sm_100 TMEM budget
TMEM_COLS_SM107 = 576  # the sm_107 TMEM (cute.arch.get_max_tmem_alloc_cols("sm_107"))
#: The A operand ring ends below 256 KB (a placement bound: the stage count it caps is not a numeric parameter). It is
#: empirical: on SM107 (327 KB of shared memory) the narrow tiles, whose B stage is 1-2 KB so the budget admits 16-17 stages
#: of the 16 KB A tile, gave wrong bits once the A ring crossed 256 KB, and are bit-identical to the wide tiles at or below
#: it. The 14-bit start address of the shared-memory descriptors (16-byte units: a 256 KB window) would be a mechanism, but
#: it does not fit the rest of the evidence: the B and scale-factor rings are allocated after A and, for the 128-row 64-wide
#: records at 12 stages, end near 301 KB, and those records are bit-exact in the suite (`tests/gemm/test_batch_invariant_mm_mxfp8.py`,
#: configuration agreement and the split implementations), so the window does not bind the B ring or the `tcgen05.cp`
#: source. The bound therefore covers the A ring, the one buffer the evidence implicates. FlashInfer's own block-scaled
#: kernels plan against the sm_100 budget of 228 KB and never reach it.
A_RING_MAX_BYTES = 1 << 18


def _pow2ceil(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


class BlockScaledPlan:
    """Host-side plan of the block-scaled body: tiled MMAs, tilers, cluster layouts, stages, smem layouts (A, B, SFA,
    SFB, C) and the TMEM budget. Placements only; the instruction variant is the A2 surface. The class attributes are
    the hooks the sm_107 plan overrides; the base is the sm_100 / sm_103 MXFP8 instruction (``kind::mxf8f6f4``, K = 32,
    one UE8M0 scale per row and step)."""

    IN_FMT = "mxfp8"  # the format key: the step (MMA_K_OF), the op name and the lowering's inner radix
    A_DT = cutlass.Float8E4M3FN
    B_DT = cutlass.Float8E4M3FN
    SF_DT = E8M0_DTYPE
    SF_VEC = SF_VEC["mxfp8"]  # elements per block scale
    INST_K = 32  # the instruction's K
    TMEM_COLS = TMEM_COLS  # the TMEM budget planned against
    TMEM_ALLOC_ARCH = TMEM_ARCH  # the allocator's architecture: "sm_100" allows powers of two to 512, "sm_107" also the whole 576
    S2T_BROADCAST = False  # the smem -> TMEM scale-factor copy's source partition (see s2t_copy_and_partition)

    def make_tiled_mma(
        self, cta_group, tile_mn, breuse: bool = False, b_collector=None
    ):
        """The instruction (the A2 surface, ``Knobs.instruction(IN_FMT)``): the sm_100 block-scaled op the library selects
        for the plan's operand / scale formats (``kind::mxf8f6f4`` K = 32 for e4m3 with 32-wide UE8M0 scales). ``breuse`` /
        ``b_collector`` (the B-reuse record: two instructions of half the tile's M per K step through the collector
        buffer) exist on the sm_107 instruction only (:class:`BlockScaledPlanK64`)."""
        assert not breuse and b_collector is None, (
            "B-reuse needs the sm_107 instruction's collector buffer"
        )
        kmaj = cute.nvgpu.OperandMajorMode.K
        return utils.sm100.make_blockscaled_trivial_tiled_mma(
            self.A_DT,
            self.B_DT,
            kmaj,
            kmaj,
            self.SF_DT,
            self.SF_VEC,
            cta_group,
            tile_mn,
        )

    def epilogue_tile(self):
        return utils.sm100.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, self.use_2cta_instrs, self.c_layout, self.c_dtype
        )

    def sf_tmem_cols(self):
        """TMEM columns of one stage of SFA and of SFB, from the TMEM layouts: a TMEM address is ``row << 16 | col``, so the
        cosize of the layout recast to 32-bit columns, masked to its low 16 bits, is the column extent of one stage of
        scale factors (one 128-row x 4-block scale-factor atom = 4 columns of 32 rows x 4 bytes; a stage holds
        ``inner * step / (4 * SF_VEC)`` atoms per 128 rows: one for MXFP8 at either instruction)."""
        sfa = (
            cute.cosize(cute.recast_layout(32, self.SF_DT.width, self.tCtSFA_layout))
            & 0xFFFF
        )
        sfb = (
            cute.cosize(cute.recast_layout(32, self.SF_DT.width, self.tCtSFB_layout))
            & 0xFFFF
        )
        return sfa, sfb

    def tmem_alloc_cols(self, need: int) -> int:
        return max(32, _pow2ceil(need))

    def __init__(self, knobs: Knobs, c_dtype, c_layout, chunks: int = 1):
        """``chunks``: the class's chunk count; with ``knobs.cta_chunks`` the CTA folds all of them (``self.cta_chunks``) and
        the TMEM holds the running sum next to the accumulator stages (``sum_stage``)."""
        assert knobs.well_formed(self.IN_FMT) and knobs.use_tma_store, knobs
        self.knobs = knobs
        self.chunks = chunks
        self.cta_chunks = (
            chunks if (knobs.cta_chunks and chunks > 1) else 1
        )  # chunks folded inside the CTA (1: none)
        self.block = MainloopBlock(self.IN_FMT)
        self.a_dtype, self.b_dtype = self.A_DT, self.B_DT
        self.sf_dtype, self.sf_vec = self.SF_DT, self.SF_VEC
        self.c_dtype, self.c_layout = c_dtype, c_layout
        self.use_2cta_instrs = knobs.use_2cta_instrs
        self.cluster_shape_mn = knobs.cluster_shape_mn
        self.cluster_k = knobs.cluster_k
        self.cta_group = (
            tcgen05.CtaGroup.TWO if knobs.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        # the B-reuse record (tile M = 2 x the instruction's M, the sm_107 class): two instructions per K step on the two
        # halves of the tile sharing one B fragment through the collector buffer (`tiled_mma_bkeep` FILL, `tiled_mma_breuse`
        # LASTUSE); the base tiled MMA (collector ops DISCARD) partitions the operands and the accumulator
        self.breuse = knobs.b_reuse
        assert not self.breuse or self.IN_FMT in BREUSE_FMTS, (self.IN_FMT, knobs)
        # the instruction variant (A2 surface): kind::mxf8f6f4.block_scale, K = 32 (sm_100 op) or 64 (sm_107 op), SF UE8M0 / 32,
        # cta_group 1/2, tile MxN
        self.tiled_mma = self.make_tiled_mma(
            self.cta_group, knobs.mma_tiler_mn, self.breuse
        )
        self.tiled_mma_bkeep = self.tiled_mma_breuse = None
        if self.breuse:
            self.tiled_mma_bkeep = self.make_tiled_mma(
                self.cta_group, knobs.mma_tiler_mn, True, tcgen05.CollectorOp.FILL
            )
            self.tiled_mma_breuse = self.make_tiled_mma(
                self.cta_group, knobs.mma_tiler_mn, True, tcgen05.CollectorOp.LASTUSE
            )
        inst_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        assert inst_k == self.block.step, (inst_k, self.block.step)
        self.atom_thr_size = cute.size(self.tiled_mma.thr_id.shape)
        self.mma_tiler = (
            knobs.mma_tiler_mn[0],
            knobs.mma_tiler_mn[1],
            self.block.k_block,
        )  # inst_k x the lowering's inner radix
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // self.atom_thr_size,
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        cta_m, tn = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]
        # SFB is loaded in whole scale-factor atoms (128 rows of N) through a 1-CTA tiled MMA of that width (and of the
        # instruction's per-CTA M: 128 rows, half of a 2-CTA instruction, whatever the tile's M)
        self.sfb_tile_n = -(-tn // SF_ATOM_ROWS) * SF_ATOM_ROWS
        sfb_m = knobs.inst_m // self.atom_thr_size
        self.tiled_mma_sfb = self.make_tiled_mma(
            tcgen05.CtaGroup.ONE, (sfb_m, self.sfb_tile_n)
        )
        self.mma_tiler_sfb = (sfb_m, self.sfb_tile_n, self.mma_tiler[2])
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*knobs.cluster_shape_mn, 1)),
            (self.tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*knobs.cluster_shape_mn, 1)),
            (self.tiled_mma_sfb.thr_id.shape,),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1
        self.epi_tile = self.epilogue_tile()
        self.smem_capacity = cutlass.memory.get_smem_capacity_in_bytes()
        # the in-cluster split's mailbox: (cluster_k - 1) fp32 tiles (one slot per peer) and one mbarrier per epilogue subtile
        self.subtile_elems = cute.size(self.epi_tile[0]) * cute.size(self.epi_tile[1])
        self.num_subtiles = cta_m * tn // self.subtile_elems
        self.mailbox_elems = (
            (self.cluster_k - 1) * cta_m * tn if self.cluster_k > 1 else 0
        )
        self.reduce_bytes = self.mailbox_elems * 4 + (
            self.num_subtiles * 8 if self.cluster_k > 1 else 0
        )
        # the TMEM layouts of one stage of scale factors (the instruction reads SFA / SFB from TMEM), from the one-stage smem
        # layouts; the stage count does not enter them
        sfa_one = bs_layout.make_smem_layout_sfa(
            self.tiled_mma, self.mma_tiler, self.sf_vec, 1
        )
        sfb_one = bs_layout.make_smem_layout_sfb(
            self.tiled_mma, self.mma_tiler, self.sf_vec, 1
        )
        self.tCtSFA_layout = bs_layout.make_tmem_layout_sfa(
            self.tiled_mma,
            self.mma_tiler,
            self.sf_vec,
            cute.slice_(sfa_one, (None, None, None, 0)),
        )
        self.tCtSFB_layout = bs_layout.make_tmem_layout_sfb(
            self.tiled_mma,
            self.mma_tiler,
            self.sf_vec,
            cute.slice_(sfb_one, (None, None, None, 0)),
        )
        # TMEM budget: accumulator stages x tile N columns + the scale-factor columns; where 2 x tile N + scale factors (+ the
        # running sum) exceed the budget the accumulator gets one stage
        self.num_sfa_tmem_cols, self.num_sfb_tmem_cols = self.sf_tmem_cols()
        # one accumulator stage = tile N columns (two tile-N blocks for the B-reuse record: one accumulator per 256-row half);
        # the in-CTA chunk fold adds one more accumulator-shaped buffer, the running sum (stage index `sum_stage`)
        acc_cols = tn * (2 if self.breuse else 1)
        n_sum = 1 if self.cta_chunks > 1 else 0
        self.num_acc_stage = (
            1
            if acc_cols * (2 + n_sum) + self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
            > self.TMEM_COLS
            else 2
        )
        self.sum_stage = self.num_acc_stage if n_sum else None
        self.num_tmem_stages = self.num_acc_stage + n_sum
        self.num_acc_tmem_cols = acc_cols * self.num_tmem_stages
        self.num_tmem_alloc_cols = self.tmem_alloc_cols(
            self.num_acc_tmem_cols + self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        )
        assert self.num_tmem_alloc_cols <= self.TMEM_COLS, (
            knobs,
            self.num_tmem_alloc_cols,
        )
        self.num_ab_stage, self.num_c_stage = self._stages()
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = bs_layout.make_smem_layout_sfa(
            self.tiled_mma, self.mma_tiler, self.sf_vec, self.num_ab_stage
        )
        self.sfb_smem_layout_staged = bs_layout.make_smem_layout_sfb(
            self.tiled_mma, self.mma_tiler, self.sf_vec, self.num_ab_stage
        )
        self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            c_dtype, c_layout, self.epi_tile, self.num_c_stage
        )

    def _stages(self) -> Tuple[int, int]:
        """A/B/SFA/SFB ring depth from the smem budget after the barriers (and the in-cluster split's mailbox) and two C
        stages; the leftover becomes extra C stages (the dense rule with the scale factors in the stage); the A ring
        below ``A_RING_MAX_BYTES``."""
        a_one = utils.sm100.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler, self.a_dtype, 1
        )
        b_one = utils.sm100.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, self.b_dtype, 1
        )
        sfa_one = bs_layout.make_smem_layout_sfa(
            self.tiled_mma, self.mma_tiler, self.sf_vec, 1
        )
        sfb_one = bs_layout.make_smem_layout_sfb(
            self.tiled_mma, self.mma_tiler, self.sf_vec, 1
        )
        c_one = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, 1
        )
        a_bytes = cute.size_in_bytes(self.a_dtype, a_one)
        ab_bytes = (
            a_bytes
            + cute.size_in_bytes(self.b_dtype, b_one)
            + cute.size_in_bytes(self.sf_dtype, sfa_one)
            + cute.size_in_bytes(self.sf_dtype, sfb_one)
        )
        c_bytes_per_stage = cute.size_in_bytes(self.c_dtype, c_one)
        c_bytes = c_bytes_per_stage * NUM_C_STAGE_MIN
        reserve = MBAR_RESERVE_BYTES + self.reduce_bytes
        num_ab = (self.smem_capacity // OCCUPANCY - (reserve + c_bytes)) // ab_bytes
        num_ab = min(num_ab, (A_RING_MAX_BYTES - 1) // a_bytes)
        assert num_ab >= 2, (self.knobs, num_ab)
        num_c = NUM_C_STAGE_MIN + (
            self.smem_capacity
            - OCCUPANCY * ab_bytes * num_ab
            - OCCUPANCY * (reserve + c_bytes)
        ) // (OCCUPANCY * c_bytes_per_stage)
        return num_ab, num_c


class BlockScaledPlanSM107(BlockScaledPlan):
    """The plan on the sm_107 block-scaled instructions (``kind::mxf8f6f4.block_scale`` with K = 64 e4m3 elements per step,
    two UE8M0 scales per row and step, for ``gemm.mxfp8.kfold64``; ``kind::mxf4nvf4.block_scale`` with K = 128 e2m1
    elements per step for ``gemm.nvfp4.kfold128`` and ``gemm.mxfp4.kfold128``). A wider step folds more products before
    the accumulator is updated, so it is a different per-element fold from the sm_100 step: a new consistency class, not
    a placement. What differs from the base plan, all host-side and all placements but the instruction itself: the tiled
    MMAs from ``rubin_helpers`` at the instruction shape ``(inst_m, tn, INST_K)`` with the collector ops of the B-reuse
    record; two instructions per 128-byte stage (the lowering's inner radix 2); the sm_107 TMEM (576 columns: two
    accumulator stages fit next to the scale factors at every tile N) and the allocation of the whole 576 when the need
    exceeds 512; the epilogue tile of the Rubin block-scaled example; the smem -> TMEM scale-factor copy partitioned on
    the MN-broadcast view (``S2T_BROADCAST``). The format hooks name the class."""

    IN_FMT = "mxfp8_k64"
    INST_K = 64
    TMEM_COLS = TMEM_COLS_SM107
    TMEM_ALLOC_ARCH = (
        "sm_107"  # the allocator accepts the non-power-of-two 576 on sm_107 only
    )
    S2T_BROADCAST = True

    def make_tiled_mma(
        self, cta_group, tile_mn, breuse: bool = False, b_collector=None
    ):
        """The Rubin block-scaled tiled MMA at the instruction shape ``(inst_m, tn, INST_K)``. B-reuse (``breuse``: the tile's
        M is 2 x ``inst_m``, two instructions per K step): the tiled MMA spans the whole tile with ``MMA_M = 2``, permuted so
        that each CTA of the 2-CTA pair owns a contiguous half of the tile's rows (CTA 0 rows 0 .. 255, CTA 1 rows 256 ..
        511: one TMA box and one epilogue block per CTA; the permutation ``(inst_m / 2, 2, 2) : (1, inst_m, inst_m / 2)`` of
        NVIDIA's and FlashInfer's Rubin kernels); ``b_collector`` FILL / LASTUSE for the two instructions of a step, DISCARD
        (default) for the partitioning MMA and the plain records."""
        kmaj = cute.nvgpu.OperandMajorMode.K
        inst_m = tile_mn[0] // 2 if breuse else tile_mn[0]
        perm = None
        if breuse and cta_group == tcgen05.CtaGroup.TWO:
            perm = (
                cute.make_layout((inst_m // 2, 2, 2), stride=(1, inst_m, inst_m // 2)),
                tile_mn[1],
                self.INST_K,
            )
        return rubin.make_blockscaled_trivial_tiled_mma(
            self.A_DT,
            self.B_DT,
            kmaj,
            kmaj,
            self.SF_DT,
            self.SF_VEC,
            cta_group,
            (inst_m, tile_mn[1], self.INST_K),
            a_collector_op=tcgen05.CollectorOp.DISCARD,
            b_collector_op=tcgen05.CollectorOp.DISCARD
            if b_collector is None
            else b_collector,
            permutation_mnk=perm,
        )

    def epilogue_tile(self):
        return rubin.compute_epilogue_tile_shape(
            self.tiled_mma.op,
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

    def tmem_alloc_cols(self, need: int) -> int:
        """Powers of two up to 512; beyond that the whole sm_107 TMEM (576 columns, what NVIDIA's Rubin example allocates)."""
        cols = max(32, _pow2ceil(need))
        return cols if cols <= 512 else TMEM_COLS_SM107


BlockScaledPlanK64 = (
    BlockScaledPlanSM107  # the sm_107 MXFP8 plan: its format hooks are the base's
)


class BlockScaledPlanNVFP4(BlockScaledPlan):
    """e2m1 x e2m1 with e4m3 block scales per 16 elements: the sm_100 / sm_103 ``kind::mxf4nvf4`` instruction, K = 64."""

    IN_FMT = "nvfp4"
    A_DT = FP4_DTYPE
    B_DT = FP4_DTYPE
    SF_DT = E4M3_DTYPE
    SF_VEC = SF_VEC["nvfp4"]
    INST_K = 64


class BlockScaledPlanNVFP4K128(BlockScaledPlanSM107):
    """e2m1 x e2m1 with e4m3 block scales per 16: the sm_107 ``kind::mxf4nvf4`` instruction, K = 128 (eight scales per
    row and step)."""

    IN_FMT = "nvfp4_k128"
    A_DT = FP4_DTYPE
    B_DT = FP4_DTYPE
    SF_DT = E4M3_DTYPE
    SF_VEC = SF_VEC["nvfp4_k128"]
    INST_K = 128


class BlockScaledPlanMXFP4(BlockScaledPlan):
    """e2m1 x e2m1 with UE8M0 block scales per 32 elements: the sm_100 / sm_103 ``kind::mxf4`` instruction, K = 64."""

    IN_FMT = "mxfp4"
    A_DT = FP4_DTYPE
    B_DT = FP4_DTYPE
    SF_DT = E8M0_DTYPE
    SF_VEC = SF_VEC["mxfp4"]
    INST_K = 64


class BlockScaledPlanMXFP4K128(BlockScaledPlanSM107):
    """e2m1 x e2m1 with UE8M0 block scales per 32: the sm_107 ``kind::mxf4nvf4`` instruction at that scale, K = 128 (four
    scales per row and step)."""

    IN_FMT = "mxfp4_k128"
    A_DT = FP4_DTYPE
    B_DT = FP4_DTYPE
    SF_DT = E8M0_DTYPE
    SF_VEC = SF_VEC["mxfp4_k128"]
    INST_K = 128


PLANS = {
    "mxfp8": BlockScaledPlan,
    "mxfp8_k64": BlockScaledPlanK64,
    "nvfp4": BlockScaledPlanNVFP4,
    "nvfp4_k128": BlockScaledPlanNVFP4K128,
    "mxfp4": BlockScaledPlanMXFP4,
    "mxfp4_k128": BlockScaledPlanMXFP4K128,
}


class BlockScaledTmaSetup:
    """TMA atoms and coordinate tensors for A, B, SFA, SFB (loads, zero-fill out of bounds: the padded K tail reads +0
    values and +0 scale bytes) and C (store, clipped); the transaction bytes of one stage of all four operands."""

    def __init__(
        self,
        st: BlockScaledPlan,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
    ):
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        a_one = cute.slice_(st.a_smem_layout_staged, (None, None, None, 0))
        self.atom_a, self.tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_one, st.mma_tiler, st.tiled_mma, st.cluster_layout_vmnk.shape
        )
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        b_one = cute.slice_(st.b_smem_layout_staged, (None, None, None, 0))
        self.atom_b, self.tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_one, st.mma_tiler, st.tiled_mma, st.cluster_layout_vmnk.shape
        )
        sfa_op = utils.sm100.cluster_shape_to_tma_atom_A(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        sfa_one = cute.slice_(st.sfa_smem_layout_staged, (None, None, None, 0))
        self.atom_sfa, self.tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa,
            sfa_one,
            st.mma_tiler,
            st.tiled_mma,
            st.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_op = utils.sm100.cluster_shape_to_tma_atom_SFB(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        sfb_one = cute.slice_(st.sfb_smem_layout_staged, (None, None, None, 0))
        self.atom_sfb, tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb,
            sfb_one,
            st.mma_tiler_sfb,
            st.tiled_mma_sfb,
            st.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        if st.cta_tile_shape_mnk[1] == 192:
            # A 192-wide N tile spans one and a half 128-row scale-factor atoms. Re-address the SFB coordinate tensor so that
            # N tile n (a 256-row box) starts at atom 3(n//2) + n%2: even tiles read atoms 3y, 3y+1; odd tiles atoms 3y+1, 3y+2
            # whose first 64 rows belong to the previous tile and are skipped in TMEM (blockscaled_mma_fold).
            x = tensor_sfb.stride[0][1]
            y = cute.ceil_div(tensor_sfb.shape[0][1], 4)
            new_shape = (
                (tensor_sfb.shape[0][0], ((2, 2), y)),
                tensor_sfb.shape[1],
                tensor_sfb.shape[2],
            )
            new_stride = (
                (tensor_sfb.stride[0][0], ((x, x), 3 * x)),
                tensor_sfb.stride[1],
                tensor_sfb.stride[2],
            )
            tensor_sfb = cute.make_tensor(
                tensor_sfb.iterator, cute.make_layout(new_shape, stride=new_stride)
            )
        self.tensor_sfb = tensor_sfb
        self.tx_bytes = (
            cute.size_in_bytes(st.a_dtype, a_one)
            + cute.size_in_bytes(st.b_dtype, b_one)
            + cute.size_in_bytes(st.sf_dtype, sfa_one)
            + cute.size_in_bytes(st.sf_dtype, sfb_one)
        ) * st.atom_thr_size
        epi_smem = cute.select(st.c_smem_layout_staged, mode=[0, 1])
        self.atom_c, self.tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem, st.epi_tile
        )


def fp4_view(t: cute.Tensor) -> cute.Tensor:
    """The byte-packed ``(rows, K / 2, L)`` uint8 operand (two e2m1 values per byte, K contiguous) as the ``(rows, K, L)`` fp4
    tensor the block-scaled TMA reads: the same bytes, the row and batch strides counted in elements (a view, no data moves)."""
    s0, _, s2 = t.layout.stride
    ptr = cute.recast_ptr(t.iterator, dtype=FP4_DTYPE)
    return cute.make_tensor(
        ptr,
        cute.make_layout(
            (t.shape[0], t.shape[1] * 2, t.shape[2]), stride=(s0 * 2, 1, s2 * 2)
        ),
    )


def sf_view(sf: cute.Tensor, rows, k, l, vec: int, sf_dtype=E8M0_DTYPE):
    """The flat scale-factor buffer (uint8 bytes of UE8M0) as the (rows, k, l) scale tensor the block-scaled TMA reads: the
    128x4 swizzled layout, i.e. tiles of 128 rows x 4 K-blocks (512 B; row r, block j inside a tile at byte
    ``(r % 32) * 16 + (r // 32) * 4 + j``), K-tiles consecutive, row-tiles strided by the K-tiles of the *whole* K. The
    ``l`` mode is the K chunk of the split class (``k`` elements each, a whole number of ``4 * vec``-element K-tiles):
    chunk ``c`` starts ``c * k / (4 vec)`` K-tiles in. ``vec`` consecutive elements share one scale (stride 0)."""
    mt = cute.ceil_div(rows, SF_ATOM_ROWS)
    kt = cute.ceil_div(k, vec * SF_ATOM_KBLOCKS)
    ptr = cute.recast_ptr(sf.iterator, dtype=sf_dtype)
    layout = cute.make_layout(
        (((32, 4), mt), ((vec, SF_ATOM_KBLOCKS), kt), l),
        stride=(
            ((16, 4), SF_ATOM_BYTES * kt * l),
            ((0, 1), SF_ATOM_BYTES),
            SF_ATOM_BYTES * kt,
        ),
    )
    return cute.make_tensor(ptr, layout)


def _append_mn_broadcast_mode(smem_layout: cute.Layout) -> cute.Layout:
    """The compact one-stage scale-factor smem layout ``(((AtomMN, RestMN), (AtomK, RestK)), MMA_MN, MMA_K, STAGE)`` with a
    stride-0 mode of 4 appended to its MN atom. The ``4x32x128b`` copy atom has no broadcast mode of its own, so for the
    wider K step the source partition is formed on this broadcast view (NVIDIA's Rubin block-scaled example does the
    same); a layout manipulation, no data moves."""
    mn = cute.get(smem_layout, mode=[0, 0])
    mn = cute.append(mn, cute.make_layout((4), stride=(0)))
    layout = cute.append(cute.group_modes(mn, 0), cute.get(smem_layout, mode=[0, 1]))
    layout = cute.append(cute.group_modes(layout, 0), cute.get(smem_layout, mode=[1]))
    layout = cute.append(layout, cute.get(smem_layout, mode=[2]))
    return cute.append(layout, cute.get(smem_layout, mode=[3]))


@cute.jit
def s2t_copy_and_partition(
    sSF: cute.Tensor,
    tSF: cute.Tensor,
    cta_group: cutlass.Constexpr,
    broadcast: cutlass.Constexpr = False,
):
    """The smem -> TMEM copy of one stage of scale factors (``tcgen05.cp 4x32x128b``): the tiled copy, the smem descriptor
    partition (all stages) and the TMEM partition. ``broadcast``: the sm_107 plan's source partition on the MN-broadcast
    view of the smem tensor (``_append_mn_broadcast_mode``)."""
    tCsSF = cute.filter_zeros(sSF)
    tCtSF = cute.filter_zeros(tSF)
    atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(cta_group), sSF.element_type)
    tiled_copy = tcgen05.make_s2t_copy(atom, tCtSF)
    thr = tiled_copy.get_slice(0)
    src = tCsSF
    if cutlass.const_expr(broadcast):
        src = cute.make_tensor(tCsSF.iterator, _append_mn_broadcast_mode(tCsSF.layout))
    tCsSF_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy, thr.partition_S(src))
    tCtSF_s2t = thr.partition_D(tCtSF)
    return tiled_copy, tCsSF_s2t, tCtSF_s2t


@dsl_user_op
def _map_shared_rank(
    smem_ptr: cute.Pointer, peer_rank: Int32, *, loc=None, ip=None
) -> Int32:
    """``mapa.shared::cluster``: the address of ``smem_ptr`` in the peer CTA's shared memory."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr.toint(loc=loc, ip=ip).ir_value(), peer_rank.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _store_remote_v4(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Four fp32 bit patterns into a peer CTA's shared memory, completing 16 transaction bytes on the peer's mbarrier
    (``st.async.shared::cluster.mbarrier::complete_tx``): a move of bits, no arithmetic."""
    remote_smem = _map_shared_rank(smem_ptr, peer_rank, loc=loc, ip=ip).ir_value()
    remote_mbar = _map_shared_rank(mbar_ptr, peer_rank, loc=loc, ip=ip).ir_value()
    vals = [v.bitcast(Int32).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)]
    llvm.inline_asm(
        None,
        [remote_smem, *vals, remote_mbar],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];",
        "r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def blockscaled_produce_ktiles(
    k_tile_cnt: Int32,
    k_tile0: Int32,
    ab_producer,
    tma_atom_a: cute.CopyAtom,
    tma_atom_b: cute.CopyAtom,
    tma_atom_sfa: cute.CopyAtom,
    tma_atom_sfb: cute.CopyAtom,
    tAgA_slice: cute.Tensor,
    tBgB_slice: cute.Tensor,
    tAgSFA_slice: cute.Tensor,
    tBgSFB_slice: cute.Tensor,
    tAsA: cute.Tensor,
    tBsB: cute.Tensor,
    tAsSFA: cute.Tensor,
    tBsSFB: cute.Tensor,
    a_mcast_mask,
    b_mcast_mask,
    sfb_mcast_mask,
):
    """The k-tiles of one tile (or of one chunk of a tile), ascending from ``k_tile0``, into the ring: one slot holds A, B,
    SFA and SFB of a k-tile. Returns the producer (loop-carried)."""
    ab_producer.reset()
    peek_empty = ab_producer.try_acquire()
    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
        handle = ab_producer.acquire_and_advance(peek_empty)
        kt = k_tile0 + handle.count
        cute.copy(
            tma_atom_a,
            tAgA_slice[(None, kt)],
            tAsA[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
            mcast_mask=a_mcast_mask,
        )
        cute.copy(
            tma_atom_b,
            tBgB_slice[(None, kt)],
            tBsB[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
            mcast_mask=b_mcast_mask,
        )
        cute.copy(
            tma_atom_sfa,
            tAgSFA_slice[(None, kt)],
            tAsSFA[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
            mcast_mask=a_mcast_mask,
        )
        cute.copy(
            tma_atom_sfb,
            tBgSFB_slice[(None, kt)],
            tBsSFB[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
            mcast_mask=sfb_mcast_mask,
        )
        peek_empty = cutlass.Boolean(1)
        if handle.count + 1 < k_tile_cnt:
            peek_empty = ab_producer.try_acquire()
    return ab_producer


@cute.jit
def blockscaled_producer_loop(
    st,
    k_tile_cnt: Int32,
    k_tile0: Int32,
    tile_sched,
    ab_producer,
    tma_atom_a: cute.CopyAtom,
    tma_atom_b: cute.CopyAtom,
    tma_atom_sfa: cute.CopyAtom,
    tma_atom_sfb: cute.CopyAtom,
    tAgA: cute.Tensor,
    tBgB: cute.Tensor,
    tAgSFA: cute.Tensor,
    tBgSFB: cute.Tensor,
    tAsA: cute.Tensor,
    tBsB: cute.Tensor,
    tAsSFA: cute.Tensor,
    tBsSFB: cute.Tensor,
    a_mcast_mask,
    b_mcast_mask,
    sfb_mcast_mask,
    chunks: cutlass.Constexpr = 1,
):
    """The dense producer with the scale factors: per tile, per k-tile ascending from ``k_tile0`` (0, or the first k-tile of
    this CTA's chunk in the in-cluster split), one ring slot holds A, B, SFA and SFB of that k-tile. A 64-wide N tile reads
    half of a 128-row SFB atom (tile n -> atom n // 2); the 192-wide tile's addressing lives in the SFB coordinate tensor.
    ``chunks > 1`` (the in-CTA chunk fold): the operands are the split class's ``(rows, Kc, C)`` views and the chunks of a
    tile are streamed in order."""
    tn = st.cta_tile_shape_mnk[1]
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        coord = work_tile.tile_idx
        m_tile = coord[0] // st.atom_thr_size
        sfb_n = coord[1]
        if cutlass.const_expr(tn < 128):
            sfb_n = coord[1] // (
                SF_ATOM_ROWS // tn
            )  # 64-wide: half an atom; narrow (swap_ab): one tile at atom 0
        if cutlass.const_expr(chunks == 1):
            ab_producer = blockscaled_produce_ktiles(
                k_tile_cnt,
                k_tile0,
                ab_producer,
                tma_atom_a,
                tma_atom_b,
                tma_atom_sfa,
                tma_atom_sfb,
                tAgA[(None, m_tile, None, coord[2])],
                tBgB[(None, coord[1], None, coord[2])],
                tAgSFA[(None, m_tile, None, coord[2])],
                tBgSFB[(None, sfb_n, None, coord[2])],
                tAsA,
                tBsB,
                tAsSFA,
                tBsSFB,
                a_mcast_mask,
                b_mcast_mask,
                sfb_mcast_mask,
            )
        else:
            for c in cutlass.range(
                0, chunks, 1, unroll=1
            ):  # chunks ascending: the batch mode of the split views
                ab_producer = blockscaled_produce_ktiles(
                    k_tile_cnt,
                    k_tile0,
                    ab_producer,
                    tma_atom_a,
                    tma_atom_b,
                    tma_atom_sfa,
                    tma_atom_sfb,
                    tAgA[(None, m_tile, None, c)],
                    tBgB[(None, coord[1], None, c)],
                    tAgSFA[(None, m_tile, None, c)],
                    tBgSFB[(None, sfb_n, None, c)],
                    tAsA,
                    tBsB,
                    tAsSFA,
                    tBsSFB,
                    a_mcast_mask,
                    b_mcast_mask,
                    sfb_mcast_mask,
                )
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    ab_producer.tail()


@cute.jit
def blockscaled_mma_fold(
    lowering,
    st,
    k_tile_cnt: Int32,
    is_leader_cta,
    n_tile: Int32,
    tiled_mma: cute.TiledMma,
    ab_consumer,
    acc_pipeline,
    acc_state,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCtAcc_base: cute.Tensor,
    acc_tmem_ptr,
    tCtSFA: cute.Tensor,
    tCtSFB: cute.Tensor,
    tiled_copy_sfa,
    tCsSFA_s2t,
    tCtSFA_s2t,
    tiled_copy_sfb,
    tCsSFB_s2t,
    tCtSFB_s2t,
    tiled_mma_bkeep=None,
    tiled_mma_breuse=None,
):
    """One ``kstep`` fold (a tile, or one chunk of a tile) of the block-scaled body into the accumulator stage ``acc_state``
    points at (``blockscaled_mma_loop`` for the issue order), then the stage's commit to the epilogue. Returns the
    participants it advanced."""
    tn = st.cta_tile_shape_mnk[1]
    sfb_col0 = st.num_acc_tmem_cols + st.num_sfa_tmem_cols
    tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
    tCtSFB_mma = tCtSFB
    if cutlass.const_expr(tn == 192):
        offset = (
            Int32(2) if n_tile % 2 == 1 else Int32(0)
        )  # odd tile: skip the atom's first 64 rows
        tCtSFB_mma = cute.make_tensor(
            cute.recast_ptr(acc_tmem_ptr + sfb_col0 + offset, dtype=st.sf_dtype),
            tCtSFB.layout,
        )
    elif cutlass.const_expr(tn == 64):
        offset = Int32((n_tile % 2) * 2)  # second half of the shared atom
        tCtSFB_mma = cute.make_tensor(
            cute.recast_ptr(acc_tmem_ptr + sfb_col0 + offset, dtype=st.sf_dtype),
            tCtSFB.layout,
        )
    # tn < 64 (the swapped decode records): a single N tile (acceptance: all tokens fit it), so its scale factors are the
    # atom's first rows and the TMEM window needs no shift
    ab_consumer.reset()
    peek_full = cutlass.Boolean(1)
    if is_leader_cta:
        peek_full = ab_consumer.try_wait()
    if is_leader_cta:
        acc_pipeline.producer_acquire(
            acc_state
        )  # the epilogue has drained this TMEM buffer
    for k_tile in range(k_tile_cnt):  # blocks ascending (the lowering's outer loop)
        if is_leader_cta:
            handle = ab_consumer.wait_and_advance(peek_full)
            stage = (None, None, None, None, handle.index)
            cute.copy(
                tiled_copy_sfa, tCsSFA_s2t[stage], tCtSFA_s2t
            )  # this block's scale factors -> TMEM
            cute.copy(tiled_copy_sfb, tCsSFB_s2t[stage], tCtSFB_s2t)
            for (
                j
            ) in lowering.inner_steps():  # the lowering's issue order inside a block
                acc_on = lowering.accumulate(
                    k_tile, j
                )  # off on the fold's first step only
                if cutlass.const_expr(st.breuse):
                    for h, mma_h in (
                        (0, tiled_mma_bkeep),
                        (1, tiled_mma_breuse),
                    ):  # the two 256-row halves, one B fragment
                        mma_h.set(tcgen05.Field.ACCUMULATE, acc_on)
                        mma_h.set(tcgen05.Field.SFA, tCtSFA[(None, h, j)].iterator)
                        mma_h.set(tcgen05.Field.SFB, tCtSFB_mma[(None, 0, j)].iterator)
                        cute.gemm(
                            mma_h,
                            tCtAcc[(None, h, 0)],
                            tCrA[(None, h, j, handle.index)],
                            tCrB[(None, 0, j, handle.index)],
                            tCtAcc[(None, h, 0)],
                        )  # one fold step of half the tile's rows
                else:
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, acc_on)
                    tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, j)].iterator)
                    tiled_mma.set(
                        tcgen05.Field.SFB, tCtSFB_mma[(None, None, j)].iterator
                    )
                    crd = (None, None, j, handle.index)
                    cute.gemm(
                        tiled_mma, tCtAcc, tCrA[crd], tCrB[crd], tCtAcc
                    )  # one fold step: MMA(D, A, SFA, B, SFB)
            handle.release()
            peek_full = cutlass.Boolean(1)
            if handle.count + 1 < k_tile_cnt:
                peek_full = ab_consumer.try_wait()
    if is_leader_cta:
        acc_pipeline.producer_commit(
            acc_state
        )  # accumulator complete: hand to the epilogue
    acc_state.advance()
    return ab_consumer, acc_state


@cute.jit
def blockscaled_mma_loop(
    lowering,
    st,
    k_tile_cnt: Int32,
    is_leader_cta,
    tile_sched,
    tiled_mma: cute.TiledMma,
    ab_consumer,
    acc_pipeline,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCtAcc_base: cute.Tensor,
    acc_tmem_ptr,
    tCtSFA: cute.Tensor,
    tCtSFB: cute.Tensor,
    sSFA: cute.Tensor,
    sSFB: cute.Tensor,
    tiled_mma_bkeep=None,
    tiled_mma_breuse=None,
    chunks: cutlass.Constexpr = 1,
):
    """The ``kstep`` fold issued by ``MainloopBlock.lowering()`` (``canon.BlockedFold``): blocks (= smem stages, k-tiles
    ascending) of ``inner`` block-scaled instructions ascending, ``ACCUMULATE`` off on the fold's first step only
    (``D_0 = +0``). Before a block's instructions its scale factors are copied smem -> TMEM (``tcgen05.cp``, in the tcgen05
    issue order, so the instructions of the previous block have consumed the previous copy); each instruction gets the
    SFA / SFB TMEM slices of its K step. One ``cute.gemm`` is one opaque step of the spec, scale factors included. The 64-
    and 192-wide N tiles read their scale factors from a window of the loaded SFB atom(s): a TMEM column shift of 2 per 64
    rows (placement, decided per tile). The B-reuse record (``st.breuse``) issues two instructions per step, one per
    256-row half of the tile (``tCrA`` / ``tCtSFA`` / ``tCtAcc`` mode 1), on the same B fragment: the first keeps B in
    the collector buffer (``tiled_mma_bkeep``, FILL), the second consumes it (``tiled_mma_breuse``, LASTUSE); each half is
    its own accumulator with the same K order from +0, so every output element's fold is unchanged. ``chunks > 1`` (the
    in-CTA chunk fold, ``Knobs.cta_chunks``): ``k_tile_cnt`` is the k-tiles of ONE chunk and the fold is issued once per
    chunk, chunks ascending, each from +0 into the next accumulator stage and committed to the epilogue, which folds the
    chunk partials in order (``blocks/epilogue.py``)."""
    assert cute.size(tCrA, mode=[2]) == lowering.inner, (
        "one smem stage must hold exactly the lowering's inner steps"
    )
    assert cute.size(tCrA, mode=[1]) == (2 if st.breuse else 1), (
        "two instructions along M for the B-reuse record only"
    )
    tiled_copy_sfa, tCsSFA_s2t, tCtSFA_s2t = s2t_copy_and_partition(
        sSFA, tCtSFA, st.cta_group, st.S2T_BROADCAST
    )
    tiled_copy_sfb, tCsSFB_s2t, tCtSFB_s2t = s2t_copy_and_partition(
        sSFB, tCtSFB, st.cta_group, st.S2T_BROADCAST
    )
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, st.num_acc_stage
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        n_tile = work_tile.tile_idx[1]
        if cutlass.const_expr(chunks == 1):
            ab_consumer, acc_state = blockscaled_mma_fold(
                lowering,
                st,
                k_tile_cnt,
                is_leader_cta,
                n_tile,
                tiled_mma,
                ab_consumer,
                acc_pipeline,
                acc_state,
                tCrA,
                tCrB,
                tCtAcc_base,
                acc_tmem_ptr,
                tCtSFA,
                tCtSFB,
                tiled_copy_sfa,
                tCsSFA_s2t,
                tCtSFA_s2t,
                tiled_copy_sfb,
                tCsSFB_s2t,
                tCtSFB_s2t,
                tiled_mma_bkeep,
                tiled_mma_breuse,
            )
        else:
            for _c in cutlass.range(
                0, chunks, 1, unroll=1
            ):  # chunks ascending, each a fold from +0 into the next stage
                ab_consumer, acc_state = blockscaled_mma_fold(
                    lowering,
                    st,
                    k_tile_cnt,
                    is_leader_cta,
                    n_tile,
                    tiled_mma,
                    ab_consumer,
                    acc_pipeline,
                    acc_state,
                    tCrA,
                    tCrB,
                    tCtAcc_base,
                    acc_tmem_ptr,
                    tCtSFA,
                    tCtSFB,
                    tiled_copy_sfa,
                    tCsSFA_s2t,
                    tCtSFA_s2t,
                    tiled_copy_sfb,
                    tCsSFB_s2t,
                    tCtSFB_s2t,
                    tiled_mma_bkeep,
                    tiled_mma_breuse,
                )
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    acc_pipeline.producer_tail(acc_state)


@cute.jit
def blockscaled_cluster_epilogue_loop(
    st,
    chain,
    tidx: Int32,
    warp_idx: Int32,
    tile_sched,
    tma_atom_c: cute.CopyAtom,
    tCtAcc_base: cute.Tensor,
    sC: cute.Tensor,
    tCgC_base: cute.Tensor,
    epi_tile: cute.Tile,
    acc_pipeline,
    alpha,
    rank: Int32,
    mailbox: cute.Tensor,
    reduce_mbar: cute.Pointer,
):
    """The epilogue of the in-cluster split (``cluster_k = C``): the dense epilogue with the ``chunk`` fold of the class
    ``kchunk[C]`` in front of the chain. Per epilogue subtile ``s`` the owner CTA is rank ``s % C``; every other CTA sends
    its fp32 fragment of the subtile into the owner's mailbox slot (``st.async`` + the owner's mbarrier); the owner waits
    for the ``C - 1`` fragments and folds ``acc = +0; for c in 0 .. C-1: acc = add.rn(acc, P_c)`` with ``P_c`` its own
    registers for ``c = rank`` and the mailbox slot of rank ``c`` otherwise -- chunk order from +0, bit for bit the
    workspace fixup (``fixup.SplitKFixup``) -- then the chain and the store. The subtile loop is static so the owner is a
    compile-time rank per subtile; one tile per cluster, so every barrier is used once (phase 0)."""
    C = st.cluster_k
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    copy_atom_t2r = get_tmem_load_op(
        st.cta_tile_shape_mnk,
        st.c_layout,
        st.c_dtype,
        ACC_DTYPE,
        epi_tile,
        st.use_2cta_instrs,
    )
    tAcc_epi = cute.flat_divide(tCtAcc, epi_tile)
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
    )
    thr_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc_base = thr_t2r.partition_S(tAcc_epi)
    tCgC_epi = cute.flat_divide(tCgC, epi_tile)
    tTR_gC = thr_t2r.partition_D(tCgC_epi)
    tTR_rAcc = cute.make_rmem_tensor(
        tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, ACC_DTYPE
    )
    tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, st.c_dtype)
    copy_atom_r2s = get_smem_store_op(
        st.c_layout, st.c_dtype, ACC_DTYPE, tiled_copy_t2r
    )
    tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
    thr_r2s = tiled_copy_r2s.get_slice(tidx)
    tRS_sC = thr_r2s.partition_D(sC)
    tRS_rC = tiled_copy_r2s.retile(tTR_rC)
    bSG_sC, bSG_gC_all = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(tCgC_epi, 0, 2),
    )
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, st.num_acc_stage
    )
    c_pipeline = pipeline.PipelineTmaStore.create(
        num_stages=st.num_c_stage,
        producer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32 * len(EPILOGUE_WARPS)
        ),
    )
    sync = pipeline.NamedBarrier(
        barrier_id=BAR_EPILOGUE, num_threads=32 * len(EPILOGUE_WARPS)
    )
    vpt = cute.size(tTR_rAcc)  # fp32 values of the subtile held by this thread
    assert vpt % 4 == 0 and vpt * 32 * len(EPILOGUE_WARPS) == st.subtile_elems, (
        vpt,
        st.subtile_elems,
    )
    slot_elems = st.subtile_elems  # one peer's fragment of one subtile
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        coord = work_tile.tile_idx
        mnl = (coord[0], coord[1], coord[2])
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
        bSG_gC = bSG_gC_all[(None, None, None, *mnl)]
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_state.index)]
        acc_pipeline.consumer_wait(
            acc_state
        )  # this CTA's chunk partial is complete in TMEM
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
        for s in cutlass.range_constexpr(st.num_subtiles):
            owner = s % C
            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
            mbar = reduce_mbar + s
            if rank != owner:
                # send this chunk's fragment to the owner: slot = rank, or rank - 1 past the owner (C - 1 slots);
                # (rank + C - 1 - owner) // C is 1 exactly when rank > owner (rank < C)
                slot = rank - (rank + (C - 1 - owner)) // C
                base = (s * (C - 1) + slot) * slot_elems + tidx * vpt
                for v in cutlass.range_constexpr(0, vpt, 4):
                    _store_remote_v4(
                        tTR_rAcc[v],
                        tTR_rAcc[v + 1],
                        tTR_rAcc[v + 2],
                        tTR_rAcc[v + 3],
                        mailbox.iterator + base + v,
                        mbar,
                        Int32(owner),
                    )
            else:
                if warp_idx == EPILOGUE_WARPS[0]:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar, (C - 1) * slot_elems * 4
                        )
                cute.arch.mbarrier_wait(mbar, 0)
                # the chunk fold, in chunk order from +0 (the fixup's fold): P_c from the mailbox slot of rank c, own
                # registers at c = rank
                for i in cutlass.range_constexpr(vpt):
                    acc = CuteF32.const(0.0)
                    for c in cutlass.range_constexpr(C):
                        if cutlass.const_expr(c == owner):
                            acc = CuteF32.add(acc, tTR_rAcc[i])
                        else:
                            acc = CuteF32.add(
                                acc,
                                mailbox[
                                    (s * (C - 1) + (c - (1 if c > owner else 0)))
                                    * slot_elems
                                    + tidx * vpt
                                    + i
                                ],
                            )
                    tTR_rAcc[i] = acc
                apply_chain_pre_cvt(chain, tTR_rAcc, alpha)
                acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                tRS_rC.store(acc_vec.to(st.c_dtype))
                # the owner stores every C-th subtile (s = rank, rank + C, ...): the staging buffer follows its own store
                # count, as the pipeline's stage does, not the global subtile index (which would map every store of this
                # CTA to one buffer whenever C and the stage count share a factor, under a store still in flight)
                c_buffer = (s // C) % st.num_c_stage
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                cute.arch.fence_proxy("async.shared", space="cta")
                sync.arrive_and_wait()
                if warp_idx == EPILOGUE_WARPS[0]:
                    cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, s)])
                    c_pipeline.producer_commit()
                    c_pipeline.producer_acquire()
                sync.arrive_and_wait()
        sync.arrive_and_wait()
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_state)
        acc_state.advance()
    c_pipeline.producer_tail()


class BlockScaledGemm:
    """``C[m, n, l] = cvt(sum_k MMA(A, SFA, B, SFB))``: the block-scaled kernel for the knobs; M, N, K dynamic, L static (1, or
    the chunk count of the split class). Operands: A ``(M, K, L)`` and B ``(N, K, L)`` e4m3 (K contiguous), or for the 4-bit
    formats the byte-packed ``(M, K / 2, L)`` / ``(N, K / 2, L)`` uint8 tensors (viewed as fp4 by :func:`fp4_view`), their
    scale factors as flat uint8 buffers in the 128x4 swizzled layout, C ``(M, N, L)`` row-major. ``in_fmt`` selects the plan,
    hence the instruction. With ``knobs.cta_chunks`` (the in-CTA chunk fold of the split class) A, B and their scales are the
    split class's ``(rows, Kc, C)`` views, C has ``L = 1``, and the CTA folds the C chunk partials in order through a TMEM
    running sum (``blocks/epilogue.py``) before the chain. The chain is the specification's object for the output format:
    unscaled here (``mm_mxfp8`` takes no per-tensor scale); :class:`ScaledBlockScaledGemm` sets ``scaled`` and takes the
    4-bit classes' ``alpha``."""

    scaled = False

    def __init__(self, knobs: Knobs, in_fmt: str, out_fmt: str, chunks: int = 1):
        """``chunks``: the operands' batch extent L (1, or the split class's chunk count: the K chunk is the batch index of
        the workspace split AND of the in-CTA chunk fold, whose C has L = 1 and whose CTA folds the chunks itself)."""
        self.knobs = knobs
        self.in_fmt = in_fmt
        self.plan_cls = PLANS[in_fmt]
        self.cluster_shape_mn = knobs.cluster_shape_mn
        self.chunks = chunks
        self.cta_chunks = chunks if (knobs.cta_chunks and chunks > 1) else 1
        self.out_fmt = out_fmt
        self.chain = EpilogueChain.for_class(out_fmt, self.scaled)
        assert self.chain.scaled == self.scaled and self.chain.out_fmt == out_fmt

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self._launch(a, b, sfa, sfb, c, max_active_clusters, stream, None)

    # host side (trace time)
    def _launch(self, a, b, sfa, sfb, c, max_active_clusters, stream, alpha):
        plan_cls = self.plan_cls
        if cutlass.const_expr(plan_cls.A_DT.width == 4):
            a, b = (
                fp4_view(a),
                fp4_view(b),
            )  # the API's byte-packed operands as fp4 tensors of the logical K
        sfa_t = sf_view(
            sfa, a.shape[0], a.shape[1], a.shape[2], plan_cls.SF_VEC, plan_cls.SF_DT
        )
        sfb_t = sf_view(
            sfb, b.shape[0], b.shape[1], b.shape[2], plan_cls.SF_VEC, plan_cls.SF_DT
        )
        st = plan_cls(
            self.knobs, c.element_type, LayoutEnum.from_tensor(c), self.chunks
        )
        self.st = st
        assert OUT_DT[self.out_fmt] is c.element_type, (self.out_fmt, c.element_type)
        assert (
            a.element_type is IN_DT[self.in_fmt]
            and b.element_type is IN_DT[self.in_fmt]
        ), (
            self.in_fmt,
            a.element_type,
            b.element_type,
        )
        self.block = st.block
        assert self.block.in_fmt == self.in_fmt, (self.block.in_fmt, self.in_fmt)
        self.lowering = self.block.lowering()
        assert st.mma_tiler[2] == self.block.k_block, (st.mma_tiler, self.block.k_block)
        tma = BlockScaledTmaSetup(st, a, b, sfa_t, sfb_t, c)
        self.num_tma_load_bytes = tma.tx_bytes
        params, grid = build_grid(st, c, max_active_clusters)
        cluster = (*self.cluster_shape_mn, 1)
        if cutlass.const_expr(st.cluster_k > 1):
            # one tile per cluster: as many persistent "clusters" as tiles; the physical cluster is the K group
            params, grid1 = build_grid(st, c, 1 << 24)
            grid, cluster = (st.cluster_k, 1, grid1[2]), (st.cluster_k, 1, 1)
        self.kernel(
            st.tiled_mma,
            st.tiled_mma_sfb,
            tma.atom_a,
            tma.tensor_a,
            tma.atom_b,
            tma.tensor_b,
            tma.atom_sfa,
            tma.tensor_sfa,
            tma.atom_sfb,
            tma.tensor_sfb,
            tma.atom_c,
            tma.tensor_c,
            st.cluster_layout_vmnk,
            st.cluster_layout_sfb_vmnk,
            st.a_smem_layout_staged,
            st.b_smem_layout_staged,
            st.sfa_smem_layout_staged,
            st.sfb_smem_layout_staged,
            st.c_smem_layout_staged,
            st.epi_tile,
            params,
            alpha,
            st.tiled_mma_bkeep,
            st.tiled_mma_breuse,
        ).launch(grid=grid, block=[THREADS, 1, 1], cluster=cluster, stream=stream)

    # device side
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged,
        epi_tile: cute.Tile,
        tile_sched_params: tile_scheduler.PersistentTileSchedulerParams,
        alpha_t: Optional[cute.Tensor],
        tiled_mma_bkeep: Optional[cute.TiledMma] = None,
        tiled_mma_breuse: Optional[cute.TiledMma] = None,
    ):
        st = self.st
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        alpha = None
        if cutlass.const_expr(self.scaled):
            alpha = alpha_t[
                0
            ]  # the scaled chain's alpha: the caller's scalar, read once per CTA
        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        use_2cta_instrs = atom_thr_size == 2
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % atom_thr_size
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        # --- scaffolding: barriers, pipelines, TMEM allocator (as the dense body); the in-cluster split adds its mailbox
        C = st.cluster_k
        smem = cutlass.memory.SmemAllocator()
        if cutlass.const_expr(C > 1):

            @cute.struct
            class SharedStorageCluster:
                ab_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_ab_stage * 2
                ]
                acc_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_acc_stage * 2
                ]
                tmem_dealloc_mbar: cutlass.Int64
                tmem_holding_buf: cutlass.Int32
                reduce_mbar: cute.struct.MemRange[cutlass.Int64, st.num_subtiles]
                reduce_mailbox: cute.struct.Align[
                    cute.struct.MemRange[Float32, st.mailbox_elems], 16
                ]

            storage = smem.allocate(SharedStorageCluster)
        else:

            @cute.struct
            class SharedStorage:
                ab_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_ab_stage * 2
                ]
                acc_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, st.num_acc_stage * 2
                ]
                tmem_dealloc_mbar: cutlass.Int64
                tmem_holding_buf: cutlass.Int32

            storage = smem.allocate(SharedStorage)
        rank = cta_rank_in_cluster  # the K chunk of this CTA when C > 1
        k_tile0 = Int32(0)
        mailbox = None
        reduce_mbar = None
        if cutlass.const_expr(C > 1):
            reduce_mbar = storage.reduce_mbar.data_ptr()
            mailbox = storage.reduce_mailbox.get_tensor(
                cute.make_layout((st.mailbox_elems,))
            )
            if warp_idx == TMA_WARP:
                with cute.arch.elect_one():
                    for s in cutlass.range_constexpr(st.num_subtiles):
                        cute.arch.mbarrier_init(reduce_mbar + s, 1)
        sync_shape = (
            self.cluster_shape_mn
        )  # the physical cluster: the barrier inits must be visible cluster-wide
        if cutlass.const_expr(C > 1):
            sync_shape = (C, 1)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=st.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Warp),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        ).make_participants()
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=st.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(EPILOGUE_WARPS) * (2 if use_2cta_instrs else 1),
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=BAR_TMEM_ALLOC, num_threads=32 * (len(EPILOGUE_WARPS) + 1)
        )
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=EPILOGUE_WARPS[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=st.TMEM_ALLOC_ARCH,
        )
        pipeline_init_arrive(cluster_shape_mn=sync_shape, is_relaxed=True)

        # --- operand staging: smem rings (A, B swizzled; SFA, SFB plain), multicast masks, global tiles, partitions
        sA = smem.allocate_tensor(
            element_type=st.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=st.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=st.sf_dtype, layout=sfa_smem_layout_staged, byte_alignment=128
        )
        sSFB = smem.allocate_tensor(
            element_type=st.sf_dtype, layout=sfb_smem_layout_staged, byte_alignment=128
        )
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(st.is_a_mcast or st.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(st.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(st.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(st.mma_tiler_sfb, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(st.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        if cutlass.const_expr(C > 1):
            k_tile_cnt = (
                k_tile_cnt // C
            )  # whole k-tiles per chunk (acceptance: K % (k_block C) == 0)
            k_tile0 = rank * k_tile_cnt
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)
        tCrA = tiled_mma.make_fragment_A(sA)  # (MMA, MMA_M, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)  # (MMA, MMA_N, MMA_K, STAGE)
        acc_shape = tiled_mma.partition_shape_C(st.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, st.num_tmem_stages)
        )  # + the running sum of the in-CTA chunk fold
        # TMEM layouts of one block of scale factors (the instruction reads SFA / SFB from TMEM); the plan built the same
        # static layouts host-side for the column budget (an IR layout value cannot cross into the kernel)
        tCtSFA_layout = bs_layout.make_tmem_layout_sfa(
            tiled_mma,
            st.mma_tiler,
            st.sf_vec,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFB_layout = bs_layout.make_tmem_layout_sfb(
            tiled_mma,
            st.mma_tiler,
            st.sf_vec,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        pipeline_init_wait(cluster_shape_mn=sync_shape)

        # the scheduler: the persistent one; for the in-cluster split the K group is folded out of the block index (one tile
        # per cluster, the tile sequence shared by the C ranks)
        sched_bidx, sched_grid = cute.arch.block_idx(), cute.arch.grid_dim()
        if cutlass.const_expr(C > 1):
            sched_bidx = (Int32(0), Int32(0), sched_bidx[2])
            sched_grid = (Int32(1), Int32(1), sched_grid[2])

        # --- roles
        if warp_idx == TMA_WARP:
            tile_sched = tile_scheduler.StaticPersistentTileScheduler.create(
                tile_sched_params, sched_bidx, sched_grid
            )
            blockscaled_producer_loop(
                st,
                k_tile_cnt,
                k_tile0,
                tile_sched,
                ab_producer,
                tma_atom_a,
                tma_atom_b,
                tma_atom_sfa,
                tma_atom_sfb,
                tAgA,
                tBgB,
                tAgSFA,
                tBgSFB,
                tAsA,
                tBsB,
                tAsSFA,
                tBsSFB,
                a_full_mcast_mask,
                b_full_mcast_mask,
                sfb_full_mcast_mask,
                st.cta_chunks,
            )
        if warp_idx == MMA_WARP:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtSFA = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + st.num_acc_tmem_cols, dtype=st.sf_dtype),
                tCtSFA_layout,
            )
            tCtSFB = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + st.num_acc_tmem_cols + st.num_sfa_tmem_cols,
                    dtype=st.sf_dtype,
                ),
                tCtSFB_layout,
            )
            tile_sched = tile_scheduler.StaticPersistentTileScheduler.create(
                tile_sched_params, sched_bidx, sched_grid
            )
            blockscaled_mma_loop(
                self.lowering,
                st,
                k_tile_cnt,
                is_leader_cta,
                tile_sched,
                tiled_mma,
                ab_consumer,
                acc_pipeline,
                tCrA,
                tCrB,
                tCtAcc_base,
                tmem_ptr,
                tCtSFA,
                tCtSFB,
                sSFA,
                sSFB,
                tiled_mma_bkeep,
                tiled_mma_breuse,
                st.cta_chunks,
            )
        sC = smem.allocate_tensor(
            element_type=st.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )
        if warp_idx < MMA_WARP:
            tmem.allocate(st.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = tile_scheduler.StaticPersistentTileScheduler.create(
                tile_sched_params, sched_bidx, sched_grid
            )
            if cutlass.const_expr(C > 1):
                blockscaled_cluster_epilogue_loop(
                    st,
                    self.chain,
                    tidx,
                    warp_idx,
                    tile_sched,
                    tma_atom_c,
                    tCtAcc_base,
                    sC,
                    tCgC,
                    epi_tile,
                    acc_pipeline,
                    alpha,
                    rank,
                    mailbox,
                    reduce_mbar,
                )
            else:
                epilogue_loop(
                    st,
                    self.chain,
                    atom_thr_size,
                    tidx,
                    warp_idx,
                    tile_sched,
                    tma_atom_c,
                    tCtAcc_base,
                    sC,
                    tCgC,
                    epi_tile,
                    acc_pipeline,
                    alpha,
                    st.cta_chunks,
                )
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        if cutlass.const_expr(C > 1):
            cute.arch.cluster_arrive()  # no CTA leaves while a peer may still address its smem
            cute.arch.cluster_wait()


class ScaledBlockScaledGemm(BlockScaledGemm):
    """``C = cvt(mul.rn(sum_k MMA(A, SFA, B, SFB), alpha))``: the 4-bit classes' body (``mm_fp4``); ``alpha`` is the caller's
    1-element fp32 tensor, read once per CTA and applied once per element after the whole fold (the epilogue of the direct
    stores; the workspace split's fixup applies it instead)."""

    scaled = True

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self._launch(a, b, sfa, sfb, c, max_active_clusters, stream, alpha)
