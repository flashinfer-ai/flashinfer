# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo (modifications)
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Ported from ResInfer, commit 51b906836a69fa43915040314df57bc9f23030ad.
# Source: csrc/common/fold/sm10x/kernel_sm100_fold.py

from typing import Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait
from cutlass.pipeline import PipelineTmaUmma, PipelineUmmaAsync

_PI_SF_MN_SHAPE = (2, 4, 4)
_PI_SF_ROW_STRIDE = 16


def _pi_permute_sf_mn(sf_layout):
    """Apply the weight-row pi map to the inner 32-row scale-factor mode."""
    (atom_mn, rest_mn), k_mode, l_mode = sf_layout.shape
    (atom_mn_s, rest_mn_s), k_stride, l_stride = sf_layout.stride

    inner, outer = atom_mn
    _inner_s, outer_s = atom_mn_s
    assert inner == 32, f"pi assumes a 32-row SF atom mode, got {inner}"
    assert _inner_s == _PI_SF_ROW_STRIDE, (
        f"pi assumes SF row stride {_PI_SF_ROW_STRIDE}, got {_inner_s}"
    )
    pi_inner_stride = tuple(d * _PI_SF_ROW_STRIDE for d in (1, 8, 2))
    return cute.make_layout(
        (((_PI_SF_MN_SHAPE, outer), rest_mn), k_mode, l_mode),
        stride=(((pi_inner_stride, outer_s), rest_mn_s), k_stride, l_stride),
    )


class Sm100BlockScaledPersistentDenseGemmKernel:
    """Persistent SM100 GEMM with residue folding."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        enable_pdl: bool = True,
        fold_pairs_tma: bool = False,
        fold_on_m: bool = False,
        ab_stage_override: int | None = None,
        mext_direct_store: bool = False,
        pi_via_layout: bool = False,
        dual_alpha: bool = False,
        k_loop: bool = False,
    ):
        """Configure the MMA tile, cluster and fold geometry."""

        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size

        self.fold_pairs_tma = fold_pairs_tma

        self.fold_on_m = fold_on_m

        self.ab_stage_override = ab_stage_override

        self.dual_alpha = dual_alpha

        assert not mext_direct_store or fold_on_m, (
            "mext_direct_store requires the M-ext fold path"
        )
        self.mext_direct_store = mext_direct_store

        assert not pi_via_layout or mext_direct_store, (
            "pi_via_layout only means anything for the A6 (direct_mext) path"
        )
        self.pi_via_layout = pi_via_layout

        self.k_loop = k_loop
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn

        self.mma_tiler = (*mma_tiler_mn, 1)
        self.enable_pdl = enable_pdl
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1

        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )

        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    def _setup_attributes(self):
        """Derive MMA, memory layouts and pipeline stages from the operands."""
        mma_inst_bits_k = 256

        self.mma_inst_shape_mnk = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_bits_k // self.a_dtype.width,
        )

        self.mma_inst_shape_mnk_sfb = (
            self.mma_inst_shape_mnk[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mnk[1], 128),
            self.mma_inst_shape_mnk[2],
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )

        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mnk[0],
            self.mma_inst_shape_mnk[1],
            self.mma_inst_shape_mnk[2] * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mnk_sfb[0],
            self.mma_inst_shape_mnk_sfb[1],
            self.mma_inst_shape_mnk_sfb[2] * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        self.epi_tile_n = cute.size(self.epi_tile[1])

        if self.fold_on_m:
            self.epi_tile_c = (
                cute.size(self.epi_tile[0]) // 2,
                self.epi_tile_n,
            )
        else:
            self.epi_tile_c = (
                cute.size(self.epi_tile[0]),
                self.epi_tile_n // 2,
            )

        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.a_major_mode,
            self.b_dtype,
            self.b_major_mode,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )

        if self.ab_stage_override is not None:
            if self.ab_stage_override > self.num_ab_stage:
                raise ValueError(
                    f"ab_stage_override {self.ab_stage_override} exceeds smem-max "
                    f"{self.num_ab_stage}"
                )
            self.num_ab_stage = self.ab_stage_override

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        if self.fold_pairs_tma and self.fold_on_m:
            self.num_c_stage *= 2
            _epi_m2 = cute.size(self.epi_tile[0]) // 2
            _epi_n = self.epi_tile_n
            self.c_smem_layout_staged = cute.make_layout(
                (_epi_m2, _epi_n, self.num_c_stage),
                stride=(_epi_n, 1, _epi_m2 * _epi_n),
            )
        elif self.fold_pairs_tma:
            self.num_c_stage *= 2
            _epi_m = cute.size(self.epi_tile[0])
            _epi_n2 = self.epi_tile_n // 2
            self.c_smem_layout_staged = cute.make_layout(
                (_epi_m, _epi_n2, self.num_c_stage),
                stride=(1, _epi_m, _epi_m * _epi_n2),
            )
        else:
            self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.c_dtype,
                self.c_layout,
                self.epi_tile,
                self.num_c_stage,
            )

        self.overlapping_accum = self.num_acc_stage == 1
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (
            self.cta_tile_shape_mnk[0] // sf_atom_mn
        ) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (
            self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn
        ) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
            if not self.overlapping_accum
            else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        )
        if (self.num_accumulator_tmem_cols + self.num_sf_tmem_cols) > 512:
            raise ValueError(
                f"TMEM over budget: acc {self.num_accumulator_tmem_cols} + sf "
                f"{self.num_sf_tmem_cols} > 512 cols "
                f"(num_acc_stage={self.num_acc_stage})"
            )

        self.iter_acc_early_release_in_epilogue = (
            self.num_sf_tmem_cols // self.epi_tile_n
        )

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        c_alias_tensor: cute.Tensor = None,
    ):
        """Build descriptors and launch the persistent GEMM."""
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        if cutlass.const_expr(self.fold_pairs_tma and self.fold_on_m):
            self.c_layout = utils.LayoutEnum.COL_MAJOR
        elif cutlass.const_expr(self.fold_pairs_tma):
            self.c_layout = utils.LayoutEnum.ROW_MAJOR
        else:
            self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)

        if cutlass.const_expr(self.pi_via_layout):
            _b_shape_flat = (
                cute.size(b_tensor, mode=[0]),
                cute.size(b_tensor, mode=[1]),
                cute.size(b_tensor, mode=[2]),
            )
        else:
            _b_shape_flat = b_tensor.shape
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            _b_shape_flat, self.sf_vec_size
        )
        if cutlass.const_expr(self.pi_via_layout):
            sfb_layout = _pi_permute_sf_mn(sfb_layout)
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )

            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(
                tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout
            )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile_c if self.fold_pairs_tma else self.epi_tile,
        )

        if cutlass.const_expr(self.mext_direct_store):
            _grid_c = cute.make_tensor(
                c_alias_tensor.iterator,
                layout=cute.make_layout(
                    (
                        cute.size(c_alias_tensor, mode=[0]),
                        cute.size(c_alias_tensor, mode=[1]),
                        cute.size(c_alias_tensor, mode=[2]),
                    )
                ),
            )
        else:
            _grid_c = c_alias_tensor if self.fold_pairs_tma else c_tensor
        self.tile_sched_params, grid = self._compute_grid(
            _grid_c,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    (
                        cute.cosize(self.c_smem_layout_staged)
                        if self.fold_pairs_tma
                        else cute.cosize(self.c_smem_layout_staged.outer)
                    ),
                ],
                self.buffer_align_bytes,
            ]

            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]

            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]

            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            alpha,
            c_alias_tensor if self.fold_pairs_tma else tma_tensor_c,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.enable_pdl,
        )
        return

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
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        pipe_cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
        mC_fold_alias: cute.Tensor,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """

        alpha_value = cutlass.Float32(1.0)
        if cutlass.const_expr(alpha is not None):
            alpha_value = alpha[0].to(cutlass.Float32)
        if cutlass.const_expr(self.dual_alpha):
            alpha_res_value = alpha[1].to(cutlass.Float32)
        else:
            alpha_res_value = alpha_value

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        _tma_rank = cta_rank_in_cluster % cute.size(cluster_layout_vmnk)
        _tma_rank_sfb = cta_rank_in_cluster % cute.size(cluster_layout_sfb_vmnk)
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(_tma_rank)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            _tma_rank_sfb
        )
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.ptr
        tmem_holding_buf = storage.tmem_holding_buf.ptr

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=pipe_cluster_layout_vmnk,
        )

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=pipe_cluster_layout_vmnk,
        )

        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(
                        tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads
                    )
        cute.arch.mbarrier_init_fence()

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed(aligned=True)

        if cutlass.const_expr(self.fold_pairs_tma):
            sC = storage.sC.get_tensor(c_smem_layout_staged)
        else:
            sC = storage.sC.get_tensor(
                c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
            )

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )

        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)

        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )

        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )

        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )

        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )

        if cutlass.const_expr(self.fold_pairs_tma):
            gC_mnl = cute.local_tile(
                mC_fold_alias,
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
            if cutlass.const_expr(self.fold_on_m):
                _fold_cta_tile = (
                    self.cta_tile_shape_mnk[0] // 2,
                    self.cta_tile_shape_mnk[1],
                )
            else:
                _fold_cta_tile = (
                    self.cta_tile_shape_mnk[0],
                    self.cta_tile_shape_mnk[1] // 2,
                )
            gC_fold_mnl = cute.local_tile(
                mC_mnl,
                _fold_cta_tile,
                (None, None, None),
            )
        else:
            gC_mnl = cute.local_tile(
                mC_mnl,
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
            gC_fold_mnl = None
        k_block_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        k_base_block_cnt = cutlass.Int32(cute.size(gB_nkl, mode=[3]))

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

        sfa_cta_layout = a_cta_layout

        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )

        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)

        tCrB = tiled_mma.make_fragment_B(sB)

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )

            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, self.num_acc_stage)
            )

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(
                barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
            )

        griddepcontrol_wait()

        if warp_idx == self.tma_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                _m_coord = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                _l_coord = cur_tile_coord[2]
                mma_tile_coord_mnl = (_m_coord, cur_tile_coord[1], _l_coord)

                _l_weight = mma_tile_coord_mnl[2]

                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]

                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, _l_weight)]

                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2

                tBgSFB_slice = tBgSFB[(None, slice_n, None, _l_weight)]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_block_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )

                for _k_block in cutlass.range(0, k_block_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    _kb_idx = ab_producer_state.count
                    if cutlass.const_expr(self.k_loop):
                        _kb_idx = cutlass.Int32(
                            _kb_idx
                            if _kb_idx < k_base_block_cnt
                            else _kb_idx - k_base_block_cnt
                        )
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, _kb_idx)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, _kb_idx)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_block_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_pipeline.producer_tail(ab_producer_state)

        if warp_idx == self.mma_warp_id:
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )

            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )

            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )

            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                _m_coord = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                _l_coord = cur_tile_coord[2]
                mma_tile_coord_mnl = (_m_coord, cur_tile_coord[1], _l_coord)

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_block_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    offset = (
                        cutlass.Int32(2)
                        if mma_tile_coord_mnl[1] % 2 == 1
                        else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for _k_block in range(k_block_cnt):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        num_kphases = cute.size(tCrA, mode=[2])
                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                            kphase_coord = (
                                None,
                                None,
                                kphase_idx,
                                ab_consumer_state.index,
                            )

                            sf_kphase_coord = (None, None, kphase_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kphase_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma[sf_kphase_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kphase_coord],
                                tCrB[kphase_coord],
                                tCtAcc,
                            )

                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)

                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_block_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        if warp_idx < self.mma_warp_id:
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )

            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )

            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                self.epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
                )
            )

            if cutlass.const_expr(self.fold_pairs_tma):
                if cutlass.const_expr(self.fold_on_m):
                    _PAIR_EVEN: tuple = (((None, 0, None), None), None, None)
                    _PAIR_ODD: tuple = (((None, 1, None), None), None, None)
                else:
                    _PAIR_EVEN = (((0, None), None), None, None)
                    _PAIR_ODD = (((1, None), None), None, None)
                tTR_rC_half = cute.make_rmem_tensor(
                    tTR_rAcc[_PAIR_EVEN].shape, self.c_dtype
                )

                if cutlass.const_expr(self.mext_direct_store):
                    bSG_sC, bSG_gC_partitioned = None, None
                else:
                    gC_fold_epi = cute.flat_divide(gC_fold_mnl, self.epi_tile_c)
                    bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sC, 0, 2),
                        cute.group_modes(gC_fold_epi, 0, 2),
                    )
                _thr_t2r = tiled_copy_t2r.get_slice(epi_tidx)
                if cutlass.const_expr(self.mext_direct_store):
                    _gC_epi = cute.flat_divide(
                        tCgC[((None, None), 0, 0, None, None, None)], epi_tile
                    )
                    tTR_gC_part = _thr_t2r.partition_D(_gC_epi)
                    _sC_alias_layout = None
                elif cutlass.const_expr(self.fold_on_m):
                    tTR_gC_part = None
                    _epi_m_sz = cute.size(epi_tile[0])
                    _epi_n_sz = cute.size(epi_tile[1])
                    _nb_epi = _epi_m_sz // 16
                    _sC_alias_layout = cute.make_layout(
                        ((8, 2, _nb_epi), _epi_n_sz),
                        stride=((_epi_n_sz, 0, 8 * _epi_n_sz), 1),
                    )
                else:
                    tTR_gC_part = None
                    _sC_alias_layout = None
                tiled_copy_r2s, tRS_rC, tRS_sC = None, None, None
            else:
                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                    tiled_copy_t2r, tTR_rC, epi_tidx, sC
                )
                tma_atom_c, bSG_sC, bSG_gC_partitioned = (
                    self.epilog_gmem_copy_and_partition(
                        epi_tidx, tma_atom_c, tCgC, epi_tile, sC
                    )
                )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                _m_coord = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                _l_coord = cur_tile_coord[2]
                mma_tile_coord_mnl = (_m_coord, cur_tile_coord[1], _l_coord)

                if cutlass.const_expr(self.mext_direct_store):
                    tTR_gC = tTR_gC_part[
                        (None, None, None, None, None, *mma_tile_coord_mnl)
                    ]
                    tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
                    bSG_gC = None
                elif cutlass.const_expr(self.fold_pairs_tma):
                    _cta_m_coord = cur_tile_coord[0]
                    bSG_gC = bSG_gC_partitioned[
                        (
                            None,
                            None,
                            None,
                            _cta_m_coord,
                            cur_tile_coord[1],
                            _l_coord,
                        )
                    ]
                else:
                    bSG_gC = bSG_gC_partitioned[
                        (
                            None,
                            None,
                            None,
                            *mma_tile_coord_mnl,
                        )
                    ]

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = (
                        cutlass.Boolean(True)
                        if acc_stage_index == 0
                        else cutlass.Boolean(False)
                    )
                else:
                    acc_stage_index = acc_consumer_state.index

                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_stage_index)
                ]

                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                if cutlass.const_expr(not self.mext_direct_store):
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt

                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = (
                                self.cta_tile_shape_mnk[1] // self.epi_tile_n
                                - 1
                                - subtile_idx
                            )

                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    if cutlass.const_expr(self.fold_pairs_tma):
                        if cutlass.const_expr(self.dual_alpha):
                            folded_half = epilogue_op(
                                (
                                    alpha_value
                                    * tTR_rAcc[_PAIR_EVEN].load().to(cutlass.Float32)
                                    + alpha_res_value
                                    * tTR_rAcc[_PAIR_ODD].load().to(cutlass.Float32)
                                ).to(self.c_dtype)
                            )
                        else:
                            _fold_sum = tTR_rAcc[_PAIR_EVEN].load().to(
                                cutlass.Float32
                            ) + tTR_rAcc[_PAIR_ODD].load().to(cutlass.Float32)
                            folded_half = epilogue_op(
                                (alpha_value * _fold_sum).to(self.c_dtype)
                            )
                        tTR_rC_half.store(folded_half)
                        if cutlass.const_expr(self.mext_direct_store):
                            tTR_gC_subtile = tTR_gC[
                                (None, None, None, real_subtile_idx)
                            ]
                            _dst = tTR_gC_subtile[_PAIR_EVEN]
                            _stg_atom = cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                self.c_dtype,
                                num_bits_per_copy=128,
                            )

                            for _im in cutlass.range_constexpr(
                                cute.size(_dst, mode=[3])
                            ):
                                for _in in cutlass.range_constexpr(
                                    cute.size(_dst, mode=[4])
                                ):
                                    _d1 = _dst[(None, None, 0, _im, _in)]
                                    _s1 = tTR_rC_half[(None, None, 0, _im, _in)]
                                    cute.copy(
                                        _stg_atom,
                                        cute.make_tensor(
                                            _s1.iterator,
                                            cute.make_layout(cute.size(_s1)),
                                        ),
                                        cute.make_tensor(
                                            _d1.iterator,
                                            cute.make_layout(cute.size(_d1)),
                                        ),
                                    )
                        elif cutlass.const_expr(self.fold_on_m):
                            sC_stage = sC[(None, None, c_buffer)]
                            sC_alias = cute.make_tensor(
                                sC_stage.iterator, _sC_alias_layout
                            )
                            sC_alias_epi = cute.flat_divide(sC_alias, epi_tile)
                            tTR_sC = _thr_t2r.partition_D(sC_alias_epi)
                            tTR_sC_sub = tTR_sC[(None, None, None, 0, 0)]
                            cute.autovec_copy(tTR_rC_half, tTR_sC_sub[_PAIR_EVEN])
                        else:
                            sC_row = sC[(epi_tidx, None, c_buffer)]
                            for _i in range(cute.size(sC_row.shape)):
                                sC_row[_i] = tTR_rC_half[_i]
                    else:
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()

                        acc_vec = epilogue_op((alpha_value * acc_vec).to(self.c_dtype))
                        tRS_rC.store(acc_vec)
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )

                    if cutlass.const_expr(not self.mext_direct_store):
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        epilog_threads = 32 * len(self.epilog_warp_id)
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epilog_threads,
                        )

                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, real_subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epilog_threads,
                        )

                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            epilog_threads = 32 * len(self.epilog_warp_id)
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
            )
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(
                        tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                    )
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )

            c_pipeline.producer_tail()

        griddepcontrol_launch_dependents()

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Return the SMEM-to-TMEM scale copy and its source/destination partitions."""

        tCsSF_compact = cute.filter_zeros(sSF)

        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)

        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )

        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Return the TMEM-to-register copy, accumulator partition and register buffer."""
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )

        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)

        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Return the register-to-SMEM copy and source/destination partitions."""
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)

        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)

        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Partition compact shared and global output tensors for TMA stores."""

        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)

        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        a_major_mode: tcgen05.OperandMajorMode,
        b_dtype: Type[cutlass.Numeric],
        b_major_mode: tcgen05.OperandMajorMode,
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """Return (ACC, A/B, C) stage counts within the shared-memory budget."""

        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        num_c_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Return persistent scheduler parameters and launch grid."""
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @cute.jit
    def wrapper(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        l: cutlass.Constexpr,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        current_stream,
        swap_ab: cutlass.Constexpr = False,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Wrap Torch tensors for the compiled GEMM."""

        m = cute.size(mA, mode=[0])
        k_raw = cute.size(mA, mode=[1])

        k_raw_b = cute.size(mB, mode=[1])
        n = cute.size(mB, mode=[0])

        if cutlass.const_expr(
            mA.element_type == cutlass.Uint8 and mB.element_type == cutlass.Uint8
        ):
            k = k_raw * 2
            k_b = k_raw_b * 2
            a_ptr = cute.recast_ptr(mA.iterator, dtype=cutlass.Float4E2M1FN)
            b_ptr = cute.recast_ptr(mB.iterator, dtype=cutlass.Float4E2M1FN)
        elif cutlass.const_expr(mA.element_type != mB.element_type):
            raise TypeError(
                "Unsupported mixed input dtypes for block-scaled GEMM: "
                "mA and mB must have matching element_type "
                "(both Uint8 for FP4 path, or both FP8 for MXFP8 path)."
            )
        else:
            k = k_raw
            k_b = k_raw_b
            a_ptr = mA.iterator
            b_ptr = mB.iterator

        _scale = (
            2
            if cutlass.const_expr(
                mA.element_type == cutlass.Uint8 and mB.element_type == cutlass.Uint8
            )
            else 1
        )
        lda = cute.size(mA.layout.stride[0]) * _scale
        ldb = cute.size(mB.layout.stride[0]) * _scale
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_layout((m, k, l), stride=(lda, 1, m * lda)),
        )
        if cutlass.const_expr(self.pi_via_layout):
            b_tensor = cute.make_tensor(
                b_ptr,
                cute.make_layout(
                    ((2, 4, 4, n // 32), k_b, l),
                    stride=((ldb, 8 * ldb, 2 * ldb, 32 * ldb), 1, n * ldb),
                ),
            )
        else:
            b_tensor = cute.make_tensor(
                b_ptr,
                layout=cute.make_layout((n, k_b, l), stride=(ldb, 1, n * ldb)),
            )

        c_alias_tensor = None
        if cutlass.const_expr(self.fold_pairs_tma and self.fold_on_m):
            c_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_layout((m // 2, n, l), stride=(n, 1, (m // 2) * n)),
            )

            _nb = m // 16
            if cutlass.const_expr(self.mext_direct_store):
                c_alias_tensor = cute.make_tensor(
                    mC.iterator,
                    layout=cute.make_layout(
                        ((8, 2, _nb), ((2, 4, 4), n // 32), l),
                        stride=((n, 0, 8 * n), ((1, 8, 2), 32), (m // 2) * n),
                    ),
                )
            else:
                c_alias_tensor = cute.make_tensor(
                    mC.iterator,
                    layout=cute.make_layout(
                        ((8, 2, _nb), n, l),
                        stride=((n, 0, 8 * n), 1, (m // 2) * n),
                    ),
                )
        elif cutlass.const_expr(self.fold_pairs_tma):
            c_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_layout((m, n // 2, l), stride=(1, m, m * (n // 2))),
            )
            c_alias_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_layout(
                    (m, (2, n // 2), l),
                    stride=(n // 2, (0, 1), m * (n // 2)),
                ),
            )
        elif cutlass.const_expr(swap_ab):
            c_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_ordered_layout((m, n, l), order=(0, 1, 2)),
            )
        else:
            c_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_ordered_layout((m, n, l), order=(1, 0, 2)),
            )

        if cutlass.const_expr(self.k_loop):
            _sfa_k = sf_k * 2
        else:
            _sfa_k = sf_k
        sfa_tensor = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, sf_m, 4, _sfa_k, l),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        sfb_tensor = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, sf_n, 4, sf_k, l),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )

        if cutlass.const_expr(self.fold_pairs_tma):
            self(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                c_tensor,
                alpha_tensor,
                max_active_clusters,
                current_stream,
                epilogue_op,
                c_alias_tensor,
            )
        else:
            self(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                c_tensor,
                alpha_tensor,
                max_active_clusters,
                current_stream,
                epilogue_op,
            )
