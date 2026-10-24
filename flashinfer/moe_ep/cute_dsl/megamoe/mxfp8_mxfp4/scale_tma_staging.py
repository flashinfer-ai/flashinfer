# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block-32 scale-factor and TMA staging for SM100 MXFP8 x MXFP4 MegaMoE.

This component uses the swap-AB orientation of the persistent MegaMoE kernel:

* operand A is K-major MXFP4 (E2M1) expert weight;
* operand B is K-major MXFP8 (E4M3) token activation;
* SFA and SFB are E8M0 scale factors with one scale per 32 K elements.

It owns the GMEM -> SMEM TMA descriptors and partitions, plus the scale-factor
SMEM -> TMEM copies consumed by ``tcgen05.mma.kind::mxf8f6f4``.  Scheduling,
pipeline barriers, routing weights, and MMA issue are intentionally outside
this module.
"""

from typing import Type

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05

from .stage_fit import MX_BLOCK_SIZE, OperandStageBytes


class Mxfp8Mxfp4TmaStaging:
    """Layout and copy factory for one block-scaled MegaMoE mainloop.

    The object contains compile-time configuration only.  Methods return CuTe
    tensors/copy atoms for the integrating kernel to place in its own shared
    storage and pipeline.
    """

    weight_dtype: Type[cutlass.Numeric] = cutlass.Float4E2M1FN
    # tcgen05's mixed-width FP4(A) x FP8(B) path consumes A from byte-wide
    # SMEM containers.  TMA still reads packed FP4 from GMEM, but unpacks each
    # nibble into one Int8 slot on arrival.
    weight_smem_dtype: Type[cutlass.Numeric] = cutlass.Int8
    activation_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN
    scale_dtype: Type[cutlass.Numeric] = cutlass.Float8E8M0FNU
    sf_vec_size = MX_BLOCK_SIZE

    def __init__(
        self,
        *,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        num_stages: int,
    ) -> None:
        if mma_tiler_mn[0] not in (128, 256):
            raise ValueError("mma_tiler_mn[0] must be 128 or 256")
        if mma_tiler_mn[1] not in (64, 128, 192, 256):
            raise ValueError("mma_tiler_mn[1] must be 64, 128, 192, or 256")
        if num_stages < 2:
            raise ValueError("num_stages must be at least two")
        if min(cluster_shape_mn) <= 0 or cluster_shape_mn[0] * cluster_shape_mn[1] > 16:
            raise ValueError("cluster_shape_mn must be positive with at most 16 CTAs")
        if max(cluster_shape_mn) > 4:
            raise ValueError(
                "each cluster dimension must be at most four for SF multicast"
            )
        if any(value & (value - 1) for value in cluster_shape_mn):
            raise ValueError("cluster dimensions must be powers of two")
        if mma_tiler_mn[0] == 256 and cluster_shape_mn[0] % 2:
            raise ValueError("2-CTA MMA requires an even cluster M dimension")

        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.num_stages = num_stages
        self.cta_group = (
            tcgen05.CtaGroup.TWO if mma_tiler_mn[0] == 256 else tcgen05.CtaGroup.ONE
        )

        self.tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.weight_dtype,
            self.activation_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.scale_dtype,
            self.sf_vec_size,
            self.cta_group,
            mma_tiler_mn,
        )
        mma_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (*mma_tiler_mn, mma_k * 4)

        # A separate 1-CTA view is required for SFB multicast and the rounded-N
        # scale layout, including the 2-CTA data path.
        sfb_mn = (
            mma_tiler_mn[0] // (2 if mma_tiler_mn[0] == 256 else 1),
            cute.round_up(mma_tiler_mn[1], 128),
        )
        self.tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.weight_dtype,
            self.activation_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.scale_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            sfb_mn,
        )
        self.mma_tiler_sfb = (*sfb_mn, mma_k * 4)

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*cluster_shape_mn, 1)),
            (self.tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*cluster_shape_mn, 1)),
            (self.tiled_mma_sfb.thr_id.shape,),
        )

        self.weight_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler, self.weight_smem_dtype, num_stages
        )
        self.activation_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, self.activation_dtype, num_stages
        )
        self.sfa_smem_layout = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma, self.mma_tiler, self.sf_vec_size, num_stages
        )
        self.sfb_smem_layout = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma, self.mma_tiler, self.sf_vec_size, num_stages
        )

    @property
    def cta_tile_shape_mnk(self) -> tuple[int, int, int]:
        return (
            self.mma_tiler[0] // cute.size(self.tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

    @property
    def cta_tile_shape_mnk_sfb(self) -> tuple[int, int, int]:
        return (
            self.mma_tiler_sfb[0] // cute.size(self.tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

    def one_stage_layouts(self):
        """Return A, B, SFA, SFB layouts with the pipeline mode sliced away."""
        coord = (None, None, None, 0)
        return (
            cute.slice_(self.weight_smem_layout, coord),
            cute.slice_(self.activation_smem_layout, coord),
            cute.slice_(self.sfa_smem_layout, coord),
            cute.slice_(self.sfb_smem_layout, coord),
        )

    def operand_stage_bytes(self) -> OperandStageBytes:
        """Return exact one-stage byte counts from the concrete CuTe layouts."""
        weight, activation, sfa, sfb = self.one_stage_layouts()
        return OperandStageBytes(
            weight_a=cute.size_in_bytes(self.weight_smem_dtype, weight),
            activation_b=cute.size_in_bytes(self.activation_dtype, activation),
            weight_sfa=cute.size_in_bytes(self.scale_dtype, sfa),
            activation_sfb=cute.size_in_bytes(self.scale_dtype, sfb),
        )

    def tma_transaction_bytes_a(self) -> int:
        """Packed weight plus SFA bytes arriving on the A pipeline stage."""

        weight, _, sfa, _ = self.one_stage_layouts()
        per_atom = cute.size_in_bytes(self.weight_dtype, weight) + cute.size_in_bytes(
            self.scale_dtype, sfa
        )
        return per_atom * cute.size(self.tiled_mma.thr_id.shape)

    def tma_transaction_bytes_b(self) -> int:
        """MXFP8 activation plus SFB bytes arriving on the B pipeline stage."""

        _, activation, _, sfb = self.one_stage_layouts()
        per_atom = cute.size_in_bytes(
            self.activation_dtype, activation
        ) + cute.size_in_bytes(self.scale_dtype, sfb)
        return per_atom * cute.size(self.tiled_mma.thr_id.shape)

    def make_tma_atoms(
        self,
        weight_a: cute.Tensor,
        activation_b: cute.Tensor,
        weight_sfa: cute.Tensor,
        activation_sfb: cute.Tensor,
    ):
        """Build TMA load atoms/tensors for A, B, SFA, and SFB."""
        a_layout, b_layout, sfa_layout, sfb_layout = self.one_stage_layouts()
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )

        tma_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            weight_a,
            a_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=self.weight_smem_dtype,
        )
        tma_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            activation_b,
            b_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            weight_sfa,
            sfa_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Uint16,
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            activation_sfb,
            sfb_layout,
            self.mma_tiler_sfb,
            self.tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Uint16,
        )
        tma_tensor_sfb = self._normalize_sfb_tma_tensor(tma_tensor_sfb)
        return (*tma_a, *tma_b, *tma_sfa, tma_atom_sfb, tma_tensor_sfb)

    def _normalize_sfb_tma_tensor(self, tma_tensor_sfb: cute.Tensor):
        """Identity: N192 SFB tiles are addressed by atom offset instead.

        CUTLASS's dense block-scaled GEMM repairs overlapping 192-column SFB
        blocks with an overlapping-stride layout.  This kernel offsets every
        work tile's scale rows with ``cute.domain_offset`` (per expert), which
        assumes a linear row mapping, so the driver instead offsets N192 tiles
        to the 128-row atom holding their first token and loads a 256-row
        tile from there (``sfb_atom_tile`` is 0 for N192).
        """
        return tma_tensor_sfb

    def make_multicast_masks(
        self,
        block_in_cluster_coord_vmnk,
        block_in_cluster_coord_sfb_vmnk,
    ):
        """Return multicast masks in A, B, SFA, SFB order."""
        num_mcast_a = cute.size(self.cluster_layout_vmnk.shape[2])
        num_mcast_b = cute.size(self.cluster_layout_vmnk.shape[1])
        num_mcast_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        # 2-CTA TMA loads always carry masks, as in the reference swap-AB
        # kernel (kernel_fc12), even when no operand is multicast.
        two_cta = self.cta_group == tcgen05.CtaGroup.TWO
        mask_a = None
        mask_b = None
        mask_sfa = None
        mask_sfb = None
        if cutlass.const_expr(num_mcast_a > 1 or two_cta):
            mask_a = cpasync.create_tma_multicast_mask(
                self.cluster_layout_vmnk,
                block_in_cluster_coord_vmnk,
                mcast_mode=2,
            )
            mask_sfa = mask_a
        if cutlass.const_expr(num_mcast_b > 1 or two_cta):
            mask_b = cpasync.create_tma_multicast_mask(
                self.cluster_layout_vmnk,
                block_in_cluster_coord_vmnk,
                mcast_mode=1,
            )
        if cutlass.const_expr(num_mcast_sfb > 1 or two_cta):
            mask_sfb = cpasync.create_tma_multicast_mask(
                self.cluster_layout_sfb_vmnk,
                block_in_cluster_coord_sfb_vmnk,
                mcast_mode=1,
            )
        return (
            mask_a,
            mask_b,
            mask_sfa,
            mask_sfb,
        )

    def tmem_column_counts(self) -> tuple[int, int, int]:
        """Return (SFA, SFB, total) TMEM column counts."""
        mma_k_tiles = self.mma_tiler[2] // cute.size(self.tiled_mma.shape_mnk, mode=[2])
        sfa_cols = self.cta_tile_shape_mnk[0] // 32 * mma_k_tiles
        sfb_cols = self.cta_tile_shape_mnk_sfb[1] // 32 * mma_k_tiles
        return sfa_cols, sfb_cols, sfa_cols + sfb_cols

    def make_tmem_scale_tensors(self, base_ptr, column_offset: int):
        """Place SFA then SFB at ``column_offset`` in an existing TMEM alloc."""
        sfa_cols, _, _ = self.tmem_column_counts()
        sf_base = cute.recast_ptr(base_ptr + column_offset, dtype=self.scale_dtype)
        sfb_base = cute.recast_ptr(
            base_ptr + column_offset + sfa_cols, dtype=self.scale_dtype
        )
        sfa_layout = blockscaled_utils.make_tmem_layout_sfa(
            self.tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.one_stage_layouts()[2],
        )
        sfb_layout = blockscaled_utils.make_tmem_layout_sfb(
            self.tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.one_stage_layouts()[3],
        )
        return (
            cute.make_tensor(sf_base, sfa_layout),
            cute.make_tensor(sfb_base, sfb_layout),
        )

    @property
    def sfb_tiles_per_atom(self) -> int:
        """Token tiles sharing one 128-row SFB atom (2 for N64, else 1)."""
        return self.cta_tile_shape_mnk_sfb[1] // self.mma_tiler[1]

    def sfb_atom_tile(self, tile_n_idx):
        """Index of the 128-row SFB TMA tile that holds token tile ``tile_n_idx``."""
        if cutlass.const_expr(self.mma_tiler[1] == 192):
            # The driver's SFB row offset already selects the tile's atom.
            return cutlass.Int32(0)
        if cutlass.const_expr(self.sfb_tiles_per_atom == 1):
            return tile_n_idx
        return tile_n_idx // cutlass.Int32(self.sfb_tiles_per_atom)

    def make_mma_tmem_sfb(self, base_ptr, column_offset: int, tile_n_idx, tmem_sfb):
        """SFB TMEM view for the MMA of token tile ``tile_n_idx``.

        SFB always moves in 128-row atoms, so for N64 the S2T copy lands
        the whole atom and the MMA must start at this tile's 32-row group:
        one TMEM word per 32 rows, as in CUTLASS's dense block-scaled GEMM.
        """
        if cutlass.const_expr(self.mma_tiler[1] == 192):
            # N192: the SFB tile starts at the atom holding the tile's first
            # row (driver offset); odd tiles start 64 rows into it, i.e. two
            # TMEM words (as in CUTLASS's dense block-scaled GEMM).
            sfa_cols, _, _ = self.tmem_column_counts()
            shift = (tile_n_idx % cutlass.Int32(2)) * cutlass.Int32(2)
            shifted = cute.recast_ptr(
                base_ptr + column_offset + sfa_cols + shift, dtype=self.scale_dtype
            )
            return cute.make_tensor(shifted, tmem_sfb.layout)
        if cutlass.const_expr(self.sfb_tiles_per_atom == 1):
            return tmem_sfb
        sfa_cols, _, _ = self.tmem_column_counts()
        shift = (tile_n_idx % cutlass.Int32(self.sfb_tiles_per_atom)) * cutlass.Int32(
            self.mma_tiler[1] // 32
        )
        shifted = cute.recast_ptr(
            base_ptr + column_offset + sfa_cols + shift, dtype=self.scale_dtype
        )
        return cute.make_tensor(shifted, tmem_sfb.layout)

    def make_s2t_copy_and_partition(self, smem_sf: cute.Tensor, tmem_sf: cute.Tensor):
        """Create the tcgen05 4x32x128b scale SMEM -> TMEM copy views."""
        smem_compact = cute.filter_zeros(smem_sf)
        tmem_compact = cute.filter_zeros(tmem_sf)
        atom = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), self.scale_dtype
        )
        tiled_copy = tcgen05.make_s2t_copy(atom, tmem_compact)
        thread_copy = tiled_copy.get_slice(0)
        smem_partition = thread_copy.partition_S(smem_compact)
        smem_desc = tcgen05.get_s2t_smem_desc_tensor(tiled_copy, smem_partition)
        tmem_partition = thread_copy.partition_D(tmem_compact)
        return tiled_copy, smem_desc, tmem_partition

    @staticmethod
    def copy_scale_stage(
        tiled_copy,
        smem_partition: cute.Tensor,
        tmem_partition: cute.Tensor,
        stage: int,
    ) -> None:
        """Issue one staged SMEM -> TMEM scale copy after pipeline wait."""
        cute.copy(
            tiled_copy,
            smem_partition[(None, None, None, None, stage)],
            tmem_partition,
        )

    def partition_tma_loads(
        self,
        *,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_sfa,
        tma_tensor_sfa,
        tma_atom_sfb,
        tma_tensor_sfb,
        smem_a,
        smem_b,
        smem_sfa,
        smem_sfb,
        mma_tile_coord_v,
        block_in_cluster_coord_vmnk,
        block_in_cluster_coord_sfb_vmnk,
    ):
        """Partition all four GMEM/SMEM pairs for the caller's TMA warp."""
        g_a = cute.local_tile(
            tma_tensor_a, cute.slice_(self.mma_tiler, (None, 0, None)), (None,) * 3
        )
        g_b = cute.local_tile(
            tma_tensor_b, cute.slice_(self.mma_tiler, (0, None, None)), (None,) * 3
        )
        g_sfa = cute.local_tile(
            tma_tensor_sfa,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None,) * 3,
        )
        g_sfb = cute.local_tile(
            tma_tensor_sfb,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None,) * 3,
        )
        thr_mma = self.tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = self.tiled_mma_sfb.get_slice(mma_tile_coord_v)
        p_a = thr_mma.partition_A(g_a)
        p_b = thr_mma.partition_B(g_b)
        p_sfa = thr_mma.partition_A(g_sfa)
        p_sfb = thr_mma_sfb.partition_B(g_sfb)

        a_cta_layout = cute.make_layout(
            cute.slice_(self.cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(self.cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        sfb_cta_layout = cute.make_layout(
            cute.slice_(self.cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        a_pair = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(smem_a, 0, 3),
            cute.group_modes(p_a, 0, 3),
        )
        b_pair = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(smem_b, 0, 3),
            cute.group_modes(p_b, 0, 3),
        )
        sfa_pair = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(smem_sfa, 0, 3),
            cute.group_modes(p_sfa, 0, 3),
        )
        sfb_pair = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(smem_sfb, 0, 3),
            cute.group_modes(p_sfb, 0, 3),
        )
        return (
            *a_pair,
            *b_pair,
            *(cute.filter_zeros(t) for t in sfa_pair),
            *(cute.filter_zeros(t) for t in sfb_pair),
        )

    def slice_tma_sources(
        self,
        *,
        global_a: cute.Tensor,
        global_b: cute.Tensor,
        global_sfa: cute.Tensor,
        global_sfb: cute.Tensor,
        tile_mnl,
    ):
        """Select one persistent work tile from partitioned TMA sources.

        The returned tensors retain the RestK pipeline-count mode.  The caller
        indexes it with the producer state's monotonically increasing count.
        """
        tile_m, tile_n, tile_l = tile_mnl
        if cutlass.const_expr(self.cta_group == tcgen05.CtaGroup.TWO):
            # A/SFA are tiled by the 256-row pair tile; the scheduler's
            # tile_m is the per-CTA 128-row index (2 * pair + v).
            tile_m = tile_m // 2
        sfb_n = tile_n
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
            sfb_n = tile_n // 2
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            sfb_n = self.sfb_atom_tile(tile_n)
        return (
            global_a[(None, tile_m, None, tile_l)],
            global_b[(None, tile_n, None, tile_l)],
            global_sfa[(None, tile_m, None, tile_l)],
            global_sfb[(None, sfb_n, None, tile_l)],
        )
