# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Host-side operand block: everything that depends only on the knobs, the formats and the
operand major modes . None of it decides a numeric order; the one
numerically visible choice, the MMA instruction variant, is the A2 surface
(``Knobs.instruction``) and is fixed by dtype x major modes x CTA group x tile.

Built at trace time (inside the ``@cute.jit`` entry point) because the CuTe layout algebra needs
the MLIR context; the results are plain attributes the kernel reads as constants.
"""

from __future__ import annotations

from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

# FlashInfer's copy of the DSL's persistent tile scheduler (the DSL's own module is deprecated and warns per use)
from flashinfer.cute_dsl.attention.fmha.helpers import (
    static_persistent_tile_scheduler as tile_scheduler,
)
from cutlass.cute.nvgpu import cpasync, tcgen05

from ..generator import MMA_K_BLOCKS, Knobs  # noqa: F401  (MMA_K_BLOCKS re-exported: the mainloop's inner radix)

ACC_DTYPE = cutlass.Float32
IN_DT = {"bf16": cutlass.BFloat16}
OUT_DT = {
    "bf16": cutlass.BFloat16,
    "f32": cutlass.Float32,
}  # f32: the split class's workspace partials
ALIGN = {"bf16": 8, "f32": 4}  # elements per 16-byte TMA alignment

NUM_ACC_STAGE = 2  # TMEM accumulator double buffer
NUM_C_STAGE_MIN = 2  # C smem stages before the leftover budget is added
MBAR_RESERVE_BYTES = 1024  # barriers and the TMEM holding word
OCCUPANCY = 1
TMEM_ARCH = "sm_100"

EPILOGUE_WARPS = (0, 1, 2, 3)
MMA_WARP = 4
TMA_WARP = 5
THREADS = 32 * (len(EPILOGUE_WARPS) + 2)
BAR_EPILOGUE = 1  # named barrier ids
BAR_TMEM_ALLOC = 2


class StaticPlan:
    """Tiled MMA, tilers, cluster layout, stage counts, smem layouts, TMEM columns."""

    def __init__(
        self,
        knobs: Knobs,
        a_dtype,
        b_dtype,
        c_dtype,
        a_major,
        b_major,
        c_layout,
        c_in_smem: bool = True,
        extra_smem_bytes: int = 2048,
        chunks: int = 1,
    ):
        """``c_in_smem=False`` reserves no C stages, so the whole leftover budget goes to the A/B ring, minus
        ``extra_smem_bytes`` for a caller's own shared storage (unused by this body). ``chunks`` is the
        class's chunk count: with ``knobs.cta_chunks`` the CTA folds all of them (``cta_chunks`` below) and the
        TMEM holds the running sum next to the accumulator stages."""
        assert knobs.well_formed() and knobs.use_tma_store, knobs
        self.knobs = knobs
        self.chunks = chunks
        self.cta_chunks = (
            chunks if (knobs.cta_chunks and chunks > 1) else 1
        )  # chunks folded inside the CTA (1: none)
        self.c_in_smem = c_in_smem
        self.extra_smem_bytes = extra_smem_bytes
        self.a_dtype, self.b_dtype, self.c_dtype, self.c_layout = (
            a_dtype,
            b_dtype,
            c_dtype,
            c_layout,
        )
        self.use_2cta_instrs = knobs.use_2cta_instrs
        self.cluster_shape_mn = knobs.cluster_shape_mn
        cta_group = (
            tcgen05.CtaGroup.TWO if knobs.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        # the instruction variant (A2 surface): kind::f16 K=16, cta_group 1/2, tile MxN
        self.tiled_mma = utils.sm100.make_trivial_tiled_mma(
            a_dtype, b_dtype, a_major, b_major, ACC_DTYPE, cta_group, knobs.mma_tiler_mn
        )
        inst_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        self.atom_thr_size = cute.size(self.tiled_mma.thr_id.shape)
        self.mma_tiler = (
            knobs.mma_tiler_mn[0],
            knobs.mma_tiler_mn[1],
            inst_k * MMA_K_BLOCKS,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // self.atom_thr_size,
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*knobs.cluster_shape_mn, 1)),
            (self.tiled_mma.thr_id.shape,),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.epi_tile = utils.sm100.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, knobs.use_2cta_instrs, c_layout, c_dtype
        )
        c_one = utils.sm100.make_smem_layout_epi(c_dtype, c_layout, self.epi_tile, 1)
        self.smem_capacity = cutlass.memory.get_smem_capacity_in_bytes()
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._stages(c_one)
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler, a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, b_dtype, self.num_ab_stage
        )
        self.c_smem_layout_staged = (
            utils.sm100.make_smem_layout_epi(
                c_dtype, c_layout, self.epi_tile, self.num_c_stage
            )
            if self.c_in_smem
            else None
        )
        acc_shape = self.tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # the in-CTA chunk fold keeps the running sum in TMEM as one more accumulator-shaped buffer (stage index
        # `num_acc_stage` of the fragment); the accumulator stages shrink to one when three buffers do not fit
        self.sum_stage = self.num_acc_stage if self.cta_chunks > 1 else None
        self.num_tmem_stages = self.num_acc_stage + (1 if self.cta_chunks > 1 else 0)
        if self.cta_chunks > 1:
            # (stages + sum) x tile N columns, a power of two >= 32 (the library's rule; its layout recast does not take the 64-row
            # tile's 16x4 lane split at three stages, so the count is formed here); one accumulator stage holds tile N columns
            self.num_tmem_alloc_cols = max(
                32, 1 << (self.knobs.tile_n * self.num_tmem_stages - 1).bit_length()
            )
        else:
            tCtAcc_fake = self.tiled_mma.make_fragment_C(
                cute.append(acc_shape, self.num_tmem_stages)
            )
            self.num_tmem_alloc_cols = cutlass.memory.get_num_tmem_alloc_cols(
                tCtAcc_fake, arch=TMEM_ARCH
            )
        assert self.num_tmem_alloc_cols <= 512, (knobs, self.num_tmem_alloc_cols)

    def _stages(self, c_one) -> Tuple[int, int, int]:
        """A/B ring depth from the smem budget after the barriers and two C stages; the leftover
        becomes extra C stages. A placement: it moves no arithmetic."""
        a_one = utils.sm100.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler, self.a_dtype, 1
        )
        b_one = utils.sm100.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, self.b_dtype, 1
        )
        ab_bytes = cute.size_in_bytes(self.a_dtype, a_one) + cute.size_in_bytes(
            self.b_dtype, b_one
        )
        c_bytes_per_stage = cute.size_in_bytes(self.c_dtype, c_one)
        num_acc = NUM_ACC_STAGE
        if self.cta_chunks > 1:
            # accumulator stages + the running sum: 3 x tile N columns when they fit the 512-column allocation, else one stage; the
            # 64-row CTA tiles (the 16 x 4 lane split of their accumulator) take one stage too: the DSL's TMEM fragment of three
            # such stages does not compile (a layout-recast limit of DSL 4.8.1, not a budget), two do
            num_acc = (
                2
                if (3 * self.knobs.tile_n <= 512 and self.knobs.cta_tile_mn[0] != 64)
                else 1
            )
        if not self.c_in_smem:
            num_ab = (
                self.smem_capacity // OCCUPANCY
                - MBAR_RESERVE_BYTES
                - self.extra_smem_bytes
            ) // ab_bytes  # the caller's tables
            return num_acc, num_ab, 0
        c_bytes = c_bytes_per_stage * NUM_C_STAGE_MIN
        num_ab = (
            self.smem_capacity // OCCUPANCY - (MBAR_RESERVE_BYTES + c_bytes)
        ) // ab_bytes
        num_c = NUM_C_STAGE_MIN + (
            self.smem_capacity
            - OCCUPANCY * ab_bytes * num_ab
            - OCCUPANCY * (MBAR_RESERVE_BYTES + c_bytes)
        ) // (OCCUPANCY * c_bytes_per_stage)
        return num_acc, num_ab, num_c


class TmaSetup:
    """TMA atoms and coordinate tensors for A, B (loads, zero-fill out of bounds) and C (store,
    clipped), plus the transaction byte count of one A+B stage."""

    def __init__(
        self, st: StaticPlan, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor = None
    ):
        """``c=None``: no TMA store atom (kernels that store from registers)."""
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        a_one = cute.select(
            st.a_smem_layout_staged,
            mode=list(range(cute.rank(st.a_smem_layout_staged) - 1)),
        )
        self.atom_a, self.tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_one, st.mma_tiler, st.tiled_mma, st.cluster_layout_vmnk.shape
        )
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            st.cluster_shape_mn, st.tiled_mma.thr_id
        )
        b_one = cute.select(
            st.b_smem_layout_staged,
            mode=list(range(cute.rank(st.b_smem_layout_staged) - 1)),
        )
        self.atom_b, self.tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_one, st.mma_tiler, st.tiled_mma, st.cluster_layout_vmnk.shape
        )
        self.tx_bytes = (
            cute.size_in_bytes(st.a_dtype, a_one)
            + cute.size_in_bytes(st.b_dtype, b_one)
        ) * st.atom_thr_size
        self.atom_c = self.tensor_c = None
        if c is not None:
            epi_smem = cute.select(st.c_smem_layout_staged, mode=[0, 1])
            self.atom_c, self.tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem, st.epi_tile
            )


def build_grid(st: StaticPlan, c: cute.Tensor, max_active_clusters):
    """Tile count from C and the CTA tile; persistent grid = min(tiles, max_active_clusters x
    cluster). The grid is a launch parameter only (I10): no fold bound depends on it."""
    gc = cute.zipped_divide(
        c, tiler=cute.slice_(st.cta_tile_shape_mnk, (None, None, 0))
    )
    num_ctas_mnl = gc[(0, (None, None, None))].shape
    params = tile_scheduler.PersistentTileSchedulerParams(
        num_ctas_mnl,
        (*st.cluster_shape_mn, 1),
        st.knobs.swizzle_size,
        st.knobs.raster_along == "m",
    )
    grid = tile_scheduler.StaticPersistentTileScheduler.get_grid_shape(
        params, max_active_clusters
    )
    return params, grid


def can_implement(
    knobs: Knobs, M: int, N: int, K: int, b_major: str = "n", chunks: int = 1
) -> bool:
    """The body's acceptance: well-formed knobs (TMA store only, so out-of-bounds tiles are fine), 16-byte alignment of
    each operand's contiguous extent (the K chunk for A and a K-major B, N for an N-major B and for C), a whole
    number of chunks."""
    if not knobs.well_formed() or b_major not in ("n", "k"):
        return False
    if K % chunks:
        return False
    if knobs.cta_chunks and chunks == 1:
        return False  # the in-CTA chunk fold is a record of the split classes only
    Kc, al_in = K // chunks, ALIGN["bf16"]
    # the output is stored directly (no fp32 workspace) by the unsplit class and by the in-CTA chunk fold
    al_out = ALIGN["bf16"] if (chunks == 1 or knobs.cta_chunks) else ALIGN["f32"]
    return (
        Kc % al_in == 0
        and (N if b_major == "n" else Kc) % al_in == 0
        and N % al_out == 0
    )
