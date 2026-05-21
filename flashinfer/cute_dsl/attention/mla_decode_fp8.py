# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Modular FP8 MLA decode kernel — composes role-based building blocks.

This is the FP8 variant of the MLA decode kernel. It differs from the FP16
variant (mla_decode.py) in warp assignments, pipeline topology, and MMA loop
structure, while sharing compute (softmax) and correction roles.

Key structural differences from FP16:
- Two TMA loader warps (K and V) instead of TMA + page-table loaders
- Separate load_k and load_v pipelines instead of unified load_kv + load_pt
- Separate SMEM buffers for KC-latent, KC-rope, and VC (no aliasing)
- mma_o_stage=2 with per-iteration pipeline wait/release
- QK rope tiler K-dim doubled (128 vs 64), iterations_qk_rope=1

AI-assisted port from monolithic mla_decode_fp8.py to the modular framework.
"""

from typing import Type, Tuple, Optional
from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .mla_config import MLAConfig
from .config import AttentionFusion
from .mla_warp_schedule import MLAWarpScheduleFP8, MLA_DECODE_FP8_SCHEDULE
from .mainloop_spec import make_mla_fp8_mainloop_spec
from .collective_builder import build_mla_fp8_launch_params
from .roles.mla_loader_fp8 import MLAFP8LoaderKRole, MLAFP8LoaderVRole
from .roles.mla_mma_fp8 import MLAMmaFP8Role
from .roles.mla_compute import MLAComputeRole
from .roles.mla_correction import MLACorrectionRole
from .scheduler.mla_persistent import (
    LOG2_E,
    MAX_SPLITS,
    MLAStaticTileScheduler,
    MLAStaticTileSchedulerParams,
    create_mla_static_tile_scheduler_params,
    mla_get_split_kv,
    mla_get_split_kv_simplified,
    mla_get_workspace_size,
)

import warnings

warnings.filterwarnings(
    "ignore",
    message="This loop is no longer unrolled and may cause performance regression",
)

from .compat import (
    setmaxregister_decrease as _setmaxregister_decrease,
    setmaxregister_increase as _setmaxregister_increase,
    get_max_tmem_alloc_cols as _get_max_tmem_alloc_cols,
)


class BlackwellMultiLatentAttentionForwardFP8:
    """Modular FP8 MLA decode kernel composing role-based building blocks.

    Follows the same compositional pattern as the FP16 MLA decode kernel
    but uses FP8-specific roles, pipeline topology, and warp schedule.
    """

    def __init__(
        self,
        config: MLAConfig,
        fusion: AttentionFusion | None = None,
        schedule: MLAWarpScheduleFP8 | None = None,
    ):
        self.config = config
        self.fusion = fusion if fusion is not None else AttentionFusion()
        self.schedule = schedule if schedule is not None else MLA_DECODE_FP8_SCHEDULE
        self.mainloop = make_mla_fp8_mainloop_spec(config, self.schedule)
        (
            self.tmem_ptr_sync_bar,
            self.softmax_exchange_sync_bar,
            self.epilogue_exchange_sync_bar,
        ) = self.schedule.make_named_barriers()

    @cute.jit
    def __call__(
        self,
        q_latent: cute.Tensor,
        q_rope: cute.Tensor,
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        page_table: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        workspace: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: Optional[cute.Tensor],
        block_split_kvs: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        params_in: Optional[cute.Tensor],
        stream,
    ):
        self.q_dtype: Type[cutlass.Numeric] = q_latent.element_type
        self.k_dtype: Type[cutlass.Numeric] = c_latent.element_type
        self.v_dtype: Type[cutlass.Numeric] = c_latent.element_type
        self.o_dtype: Type[cutlass.Numeric] = o.element_type

        if cutlass.const_expr(
            self.q_dtype != self.k_dtype or self.q_dtype != self.v_dtype
        ):
            raise TypeError(
                f"Type mismatch: {self.q_dtype} != {self.k_dtype} "
                f"or {self.q_dtype} != {self.v_dtype}"
            )

        def _reinterpret_4d(t):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[2], t.shape[3], t.shape[1], t.shape[0]),
                    stride=(t.stride[2], t.stride[3], t.stride[1], t.stride[0]),
                ),
            )

        q_latent = _reinterpret_4d(q_latent)
        q_rope = _reinterpret_4d(q_rope)
        o = _reinterpret_4d(o)

        def _reinterpret_3d_kv(t):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[1], t.shape[2], t.shape[0]),
                    stride=(t.stride[1], t.stride[2], t.stride[0]),
                ),
            )

        c_latent = _reinterpret_3d_kv(c_latent)
        c_rope = _reinterpret_3d_kv(c_rope)

        page_table = cute.make_tensor(
            page_table.iterator,
            cute.make_layout(
                (page_table.shape[1], page_table.shape[0]),
                stride=(page_table.stride[1], page_table.stride[0]),
            ),
        )

        lse = cute.make_tensor(
            lse.iterator,
            cute.make_layout(
                (lse.shape[2], lse.shape[1], lse.shape[0]),
                stride=(lse.stride[2], lse.stride[1], lse.stride[0]),
            ),
        )

        acc_o, acc_lse = self.initialize_workspace(
            q_latent.shape[0],
            q_latent.shape[1],
            q_latent.shape[2],
            q_latent.shape[3],
            split_kv,
            self.config.acc_dtype,
            workspace,
        )

        c_latent_transpose_layout = cute.select(c_latent.layout, mode=[1, 0, 2])
        c_latent_transpose = cute.make_tensor(
            c_latent.iterator, c_latent_transpose_layout
        )

        self.mainloop = self.mainloop.resolve(self.q_dtype.width)

        params = (
            cute.make_tensor(
                params_in.iterator,
                cute.make_layout(
                    self.fusion.params_shape,
                    stride=self.fusion.params_strides,
                ),
            )
            if cutlass.const_expr(self.fusion.has_params)
            else None
        )

        self.loader_k_role = MLAFP8LoaderKRole(self.config)
        self.loader_v_role = MLAFP8LoaderVRole(self.config)
        self.mma_role = MLAMmaFP8Role(self.config, self.mainloop)
        self.mma_role.set_dtypes(self.q_dtype, self.v_dtype)
        self.compute_role = MLAComputeRole(self.config, fusion=self.fusion)
        self.compute_role.set_dtypes(self.q_dtype)
        self.compute_role.set_barriers(self.softmax_exchange_sync_bar)
        self.correction_role = MLACorrectionRole(
            self.config, fusion=self.fusion, v_dtype=self.v_dtype, o_dtype=self.o_dtype
        )
        self.correction_role.set_barriers(self.epilogue_exchange_sync_bar)

        lp = build_mla_fp8_launch_params(
            self.mainloop,
            self.schedule,
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            c_latent_transpose,
            page_table,
            o,
            lse,
            acc_o,
            acc_lse,
            self.q_dtype,
            self.k_dtype,
            self.v_dtype,
            self.o_dtype,
        )
        self.shared_storage = lp.SharedStorage
        self.tma_copy_q_bytes = lp.tma_copy_q_bytes
        self.tma_copy_kc_bytes = lp.tma_copy_kc_bytes
        self.tma_copy_vc_bytes = lp.tma_copy_vc_bytes

        tile_sched_params, grid = self._compute_grid(
            o,
            split_kv,
            self.config.cluster_shape_mnk,
            self.config.max_active_clusters,
            self.config.is_persistent,
        )

        softmax_scale_log2 = softmax_scale * LOG2_E
        self.split_kv_kernel(
            lp.qk_tiled_mma,
            lp.pv_tiled_mma,
            lp.tma_atom_q_latent,
            lp.tma_tensor_q_latent,
            lp.tma_atom_q_rope,
            lp.tma_tensor_q_rope,
            lp.tma_atom_c_latent,
            lp.tma_tensor_c_latent,
            lp.tma_atom_c_rope,
            lp.tma_tensor_c_rope,
            lp.tma_atom_c_latent_transpose,
            lp.tma_tensor_c_latent_transpose,
            page_table,
            o,
            lse,
            acc_o,
            acc_lse,
            split_kv,
            cache_seqs,
            block_split_kvs,
            softmax_scale_log2,
            output_scale,
            lp.q_latent_smem_layout_staged,
            lp.q_rope_smem_layout_staged,
            lp.kc_latent_smem_layout_staged,
            lp.kc_rope_smem_layout_staged,
            lp.p_smem_layout_staged,
            lp.vc_smem_layout_staged,
            lp.kc_latent_smem_layout_for_tma,
            lp.kc_rope_smem_layout_for_tma,
            lp.vc_smem_layout_for_tma,
            lp.cta_layout_vmnk,
            tile_sched_params,
            lp.SharedStorage,
            params,
        ).launch(
            grid=grid,
            block=[self.schedule.threads_per_cta, 1, 1],
            cluster=self.config.cluster_shape_mnk,
            smem=lp.SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.config.enable_pdl,
        )
        if cutlass.const_expr(acc_o is not None):
            self.reduction_kernel(
                o,
                lse,
                acc_o,
                acc_lse,
                split_kv,
                cache_seqs,
                block_split_kvs,
            ).launch(
                grid=(q_latent.shape[0], q_latent.shape[2], q_latent.shape[3]),
                block=[
                    self.schedule.threads_per_warp * self.config.num_compute_warps,
                    1,
                    1,
                ],
                smem=MAX_SPLITS * self.config.acc_dtype.width // 8,
                stream=stream,
                min_blocks_per_mp=1,
                use_pdl=self.config.enable_pdl,
            )

    @cute.jit
    def _create_pipelines(self, storage, cta_layout_vmnk):
        barrier_ptrs = {
            edge.name: getattr(storage, edge.barrier_field_name).data_ptr()
            for edge in self.mainloop.pipeline_topology.edges
        }
        tx_counts = {
            "q": self.tma_copy_q_bytes,
            "kv": self.tma_copy_kc_bytes,
            "vc": self.tma_copy_vc_bytes,
        }
        return self.mainloop.pipeline_topology.create_pipelines(
            barrier_ptrs,
            tx_counts,
            self.schedule.threads_per_warp,
            cta_layout_vmnk,
        )

    @cute.kernel
    def split_kv_kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_atom_q_latent: Optional[cute.CopyAtom],
        mQL: cute.Tensor,
        tma_atom_q_rope: Optional[cute.CopyAtom],
        mQR: cute.Tensor,
        tma_atom_c_latent: Optional[cute.CopyAtom],
        mCL: cute.Tensor,
        tma_atom_c_rope: Optional[cute.CopyAtom],
        mKR: cute.Tensor,
        tma_atom_c_latent_transpose: Optional[cute.CopyAtom],
        mCLT: cute.Tensor,
        mPT: cute.Tensor,
        mO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mAccO: Optional[cute.Tensor],
        mAccLSE: Optional[cute.Tensor],
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        q_latent_smem_layout_staged: cute.ComposedLayout,
        q_rope_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_staged: cute.ComposedLayout,
        kc_rope_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        vc_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_for_tma: cute.ComposedLayout,
        kc_rope_smem_layout_for_tma: cute.ComposedLayout,
        vc_smem_layout_for_tma: cute.ComposedLayout,
        cta_layout_vmnk: cute.Layout,
        tile_sched_params: MLAStaticTileSchedulerParams,
        SharedStorage: cutlass.Constexpr,
        params: Optional[cute.Tensor] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        if warp_idx == self.schedule.mma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_q_latent)
            cpasync.prefetch_descriptor(tma_atom_q_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent)
            cpasync.prefetch_descriptor(tma_atom_c_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent_transpose)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_ptr_sync_bar,
            allocator_warp_id=self.schedule.mma_warp_id,
            is_two_cta=self.config.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        pipes = self._create_pipelines(storage, cta_layout_vmnk)
        load_q_prod, load_q_cons = pipes["load_q"]
        load_k_prod, load_k_cons = pipes["load_k"]
        load_v_prod, load_v_cons = pipes["load_v"]
        mma_s_prod, mma_s_cons = pipes["mma_s"]
        p_mma_prod, p_mma_cons = pipes["p_mma"]
        p_cor_prod, p_cor_cons = pipes["p_cor"]
        mma_o_prod, mma_o_cons = pipes["mma_o"]

        pipeline_init_arrive(
            cluster_shape_mn=self.config.cluster_shape_mnk,
            is_relaxed=True,
        )

        # SMEM tensor views
        sQ = storage.smem_q_latent.get_tensor(
            q_latent_smem_layout_staged.outer,
            swizzle=q_latent_smem_layout_staged.inner,
        )
        sQ_rope = storage.smem_q_rope.get_tensor(
            q_rope_smem_layout_staged.outer,
            swizzle=q_rope_smem_layout_staged.inner,
        )
        sKC = storage.smem_kc_latent.get_tensor(
            kc_latent_smem_layout_staged.outer,
            swizzle=kc_latent_smem_layout_staged.inner,
        )
        sKC_rope = storage.smem_kc_rope.get_tensor(
            kc_rope_smem_layout_staged.outer,
            swizzle=kc_rope_smem_layout_staged.inner,
        )
        sKC_for_tma = storage.smem_kc_latent.get_tensor(
            kc_latent_smem_layout_for_tma.outer,
            swizzle=kc_latent_smem_layout_for_tma.inner,
        )
        sKC_rope_for_tma = storage.smem_kc_rope.get_tensor(
            kc_rope_smem_layout_for_tma.outer,
            swizzle=kc_rope_smem_layout_for_tma.inner,
        )
        sVC = storage.smem_vc.get_tensor(
            vc_smem_layout_staged.outer,
            swizzle=vc_smem_layout_staged.inner,
        )
        sVC_for_tma = storage.smem_vc.get_tensor(
            vc_smem_layout_for_tma.outer,
            swizzle=vc_smem_layout_for_tma.inner,
        )
        sP = storage.smem_p.get_tensor(
            p_smem_layout_staged.outer,
            swizzle=p_smem_layout_staged.inner,
        )
        softmax_smem_exchange = storage.softmax_smem_exchange.get_tensor(
            cute.make_layout(
                self.config.num_compute_warps * self.schedule.threads_per_warp
            )
        )
        epilogue_smem_exchange = storage.epilogue_smem_exchange.get_tensor(
            cute.make_layout(
                self.config.num_compute_warps * self.schedule.threads_per_warp
            )
        )

        pipeline_init_wait(cluster_shape_mn=self.config.cluster_shape_mnk)

        if cutlass.const_expr(self.config.enable_pdl):
            cute.arch.griddepcontrol_wait()

        # /////////////////////////////////////////////////////////////////////
        #  Empty warps
        # /////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.schedule.empty_warp_ids[0]
            and warp_idx <= self.schedule.empty_warp_ids[-1]
        ):
            _setmaxregister_decrease(self.schedule.other_reg_num)

        # /////////////////////////////////////////////////////////////////////
        #  K loader warp (Q + K loading)
        # /////////////////////////////////////////////////////////////////////
        if warp_idx == self.schedule.load_tma_k_warp_id:
            _setmaxregister_decrease(self.schedule.other_reg_num)
            tma_common_params = SimpleNamespace(mPT=mPT)
            tma_qk_params = SimpleNamespace(
                tiled_mma_qk=tiled_mma_qk,
                tma_atom_q_latent=tma_atom_q_latent,
                tma_atom_q_rope=tma_atom_q_rope,
                tma_atom_c_latent=tma_atom_c_latent,
                tma_atom_c_rope=tma_atom_c_rope,
                mQL=mQL,
                mQR=mQR,
                mCL=mCL,
                mKR=mKR,
                sQ=sQ,
                sQ_rope=sQ_rope,
                sKC=sKC_for_tma,
                sKC_rope=sKC_rope_for_tma,
            )
            self.loader_k_role.run(
                tma_common_params,
                tma_qk_params,
                split_kv,
                cache_seqs,
                block_split_kvs,
                load_q_prod,
                load_k_prod,
                tile_sched_params,
            )

        # /////////////////////////////////////////////////////////////////////
        #  V loader warp
        # /////////////////////////////////////////////////////////////////////
        if warp_idx == self.schedule.load_tma_v_warp_id:
            _setmaxregister_decrease(self.schedule.other_reg_num)
            tma_common_params = SimpleNamespace(mPT=mPT)
            tma_v_params = SimpleNamespace(
                tiled_mma_pv=tiled_mma_pv,
                tma_atom_c_latent_transpose=tma_atom_c_latent_transpose,
                mCLT=mCLT,
                sVC=sVC_for_tma,
            )
            self.loader_v_role.run(
                tma_common_params,
                tma_v_params,
                split_kv,
                cache_seqs,
                block_split_kvs,
                load_v_prod,
                tile_sched_params,
            )

        # /////////////////////////////////////////////////////////////////////
        #  MMA warp
        # /////////////////////////////////////////////////////////////////////
        if warp_idx == self.schedule.mma_warp_id:
            _setmaxregister_decrease(self.schedule.other_reg_num)
            tmem.allocate(_get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.config.acc_dtype)

            self.mma_role.run(
                tiled_mma_qk,
                tiled_mma_pv,
                load_q_cons,
                load_k_cons,
                load_v_cons,
                mma_s_prod,
                p_mma_cons,
                mma_o_prod,
                split_kv,
                cache_seqs,
                block_split_kvs,
                tile_sched_params,
                sQ,
                sQ_rope,
                sKC,
                sKC_rope,
                sP,
                sVC,
                tmem_ptr,
                is_leader_cta,
                mCL.shape[1],
            )

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
            if cutlass.const_expr(self.config.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()

        # /////////////////////////////////////////////////////////////////////
        #  Compute (softmax) warps
        # /////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.schedule.compute_warp_ids[0]
            and warp_idx <= self.schedule.compute_warp_ids[-1]
        ):
            # Fresh TiledMma avoids SSA dominance conflict with the MMA warp's
            # .set(ACCUMULATE) mutations on the same tiled_mma_qk variable.
            compute_tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
                self.q_dtype,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                self.config.acc_dtype,
                tcgen05.CtaGroup.TWO,
                self.config.mma_qk_tiler[:2],
            )
            self.compute_role.run(
                split_kv,
                cache_seqs,
                block_split_kvs,
                tile_sched_params,
                tmem_ptr=None,
                mma_s_consumer=mma_s_cons,
                p_mma_producer=p_mma_prod,
                p_cor_producer=p_cor_prod,
                softmax_smem_exchange=softmax_smem_exchange,
                mAccO=mAccO,
                mO=mO,
                mCL=mCL,
                K=None,
                L=mCL.shape[1],
                tiled_mma_qk=compute_tiled_mma_qk,
                sP=sP,
                softmax_scale_log2=softmax_scale_log2,
                tmem=tmem,
                params=params,
            )

        # /////////////////////////////////////////////////////////////////////
        #  Correction (rescale + epilogue) warps
        # /////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.schedule.correction_warp_ids[0]
            and warp_idx <= self.schedule.correction_warp_ids[-1]
        ):
            _setmaxregister_increase(self.schedule.correction_reg_num)
            tmem.wait_for_alloc()
            tmem_ptr_corr = tmem.retrieve_ptr(self.config.acc_dtype)

            cta_m_offset = (bidx % cute.size(tiled_mma_qk.thr_id.shape)) * (
                self.config.mma_qk_tiler[0] // self.config.cluster_shape_mnk[0]
            )
            corr_common_params = SimpleNamespace(
                smem_exchange=epilogue_smem_exchange,
                mAccO=mAccO,
                mO=mO,
                L=mCL.shape[1],
                H=mQL.shape[0],
                cta_m_offset=cta_m_offset,
            )
            corr_epilogue_params = SimpleNamespace(
                output_scale=output_scale,
                softmax_scale_log2=softmax_scale_log2,
                mAccLSE=mAccLSE,
                mLSE=mLSE,
            )
            self.correction_role.run(
                split_kv,
                cache_seqs,
                block_split_kvs,
                tile_sched_params,
                tmem_ptr_corr,
                p_cor_consumer=p_cor_cons,
                mma_o_consumer=mma_o_cons,
                compute_common_params=corr_common_params,
                epilogue_params=corr_epilogue_params,
                params=params,
            )

        return

    @cute.kernel
    def reduction_kernel(
        self,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mAccO: cute.Tensor,
        mAccLSE: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
    ):
        """Reduction kernel — identical to FP16 version."""
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        blk_coord = (bidx, bidy, bidz)
        local_split_kv = (
            block_split_kvs[blk_coord[2]] if self.config.is_var_split_kv else split_kv
        )
        k_tile_total = cute.ceil_div(
            cache_seqs[blk_coord[2]], self.config.mma_qk_tiler[1]
        )
        k_tile_per_cta = cute.ceil_div(k_tile_total, local_split_kv)
        local_split_kv = cute.ceil_div(k_tile_total, k_tile_per_cta)

        smem = utils.SmemAllocator()
        storage = smem.allocate(MAX_SPLITS * self.config.acc_dtype.width // 8, 16)
        lse_scale_ptr = cute.recast_ptr(storage, dtype=self.config.acc_dtype)
        smem_lse_scale = cute.make_tensor(lse_scale_ptr, cute.make_layout(MAX_SPLITS))

        if cutlass.const_expr(self.config.enable_pdl):
            cute.arch.griddepcontrol_wait()
        gLSE = mAccLSE[blk_coord[0], None, blk_coord[1], blk_coord[2]]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            lse_per_thread = cute.ceil_div(MAX_SPLITS, self.schedule.threads_per_warp)
            local_lse = cute.make_rmem_tensor(
                cute.make_layout(lse_per_thread), self.config.lse_dtype
            )
            lse_max = -self.config.lse_dtype.inf
            for i in cutlass.range_constexpr(lse_per_thread):
                split_kv_idx = tidx + i * self.schedule.threads_per_warp
                local_lse[i] = (
                    gLSE[split_kv_idx]
                    if cute.elem_less(split_kv_idx, local_split_kv)
                    else -self.config.lse_dtype.inf
                )
                lse_max = cute.arch.fmax(lse_max, local_lse[i])
            lse_max = cute.arch.warp_reduction_max(lse_max)
            lse_max = lse_max if lse_max != -self.config.lse_dtype.inf else 0.0
            sum_lse = 0.0
            for i in cutlass.range_constexpr(lse_per_thread):
                sum_lse += cute.math.exp2(local_lse[i] - lse_max, fastmath=True)
            sum_lse = cute.arch.warp_reduction_sum(sum_lse)
            global_lse = (
                lse_max + cute.math.log2(sum_lse, fastmath=True)
                if not sum_lse == self.config.lse_dtype(0.0) or sum_lse != sum_lse  # noqa: SIM201
                else self.config.lse_dtype.inf
            )
            if tidx == 0:
                mLSE[blk_coord[0], blk_coord[1], blk_coord[2]] = global_lse
            for i in cutlass.range_constexpr(lse_per_thread):
                split_kv_idx = tidx + i * self.schedule.threads_per_warp
                if cute.elem_less(split_kv_idx, local_split_kv):
                    smem_lse_scale[split_kv_idx] = cute.math.exp2(
                        local_lse[i] - global_lse, fastmath=True
                    )

        pipeline.sync(barrier_id=4)

        elements_per_thread = cute.ceil_div(
            self.config.latent_dim,
            self.schedule.threads_per_warp * self.config.num_compute_warps,
        )
        gAccO = mAccO[blk_coord[0], None, None, blk_coord[1], blk_coord[2]]
        rAccO = cute.make_rmem_tensor(
            cute.make_layout(elements_per_thread), self.config.acc_dtype
        )
        rO = cute.make_rmem_tensor(cute.make_layout(elements_per_thread), self.o_dtype)
        rAccO.fill(0.0)
        for i in range(local_split_kv):
            for j in cutlass.range_constexpr(elements_per_thread):
                element_idx = (
                    tidx
                    + j * self.schedule.threads_per_warp * self.config.num_compute_warps
                )
                rAccO[j] += gAccO[i, element_idx] * smem_lse_scale[i]
        rO.store(rAccO.load().to(self.o_dtype))
        for j in cutlass.range_constexpr(elements_per_thread):
            element_idx = (
                tidx
                + j * self.schedule.threads_per_warp * self.config.num_compute_warps
            )
            mO[blk_coord[0], element_idx, blk_coord[1], blk_coord[2]] = rO[j]
        if cutlass.const_expr(self.config.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        return

    @staticmethod
    def _compute_grid(
        o: cute.Tensor,
        split_kv: cutlass.Int32,
        cluster_shape_mnk: Tuple[int, int, int],
        max_active_clusters: int,
        is_persistent: bool,
    ) -> Tuple[MLAStaticTileSchedulerParams, Tuple[int, int, int]]:
        o_shape = o.shape
        tile_sched_params = create_mla_static_tile_scheduler_params(
            is_persistent,
            cute.size(o_shape[3]),
            cute.size(o_shape[2]),
            cluster_shape_mnk,
            split_kv,
        )
        grid = MLAStaticTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @cute.jit
    def initialize_workspace(
        self,
        H: cutlass.Int32,
        D: cutlass.Int32,
        S: cutlass.Int32,
        B: cutlass.Int32,
        split_kv: cutlass.Int32,
        acc_dtype: Type[cutlass.Numeric],
        workspace: cute.Tensor,
    ) -> tuple[cute.Tensor, cute.Tensor]:
        """Initialize workspace tensors acc_o and acc_lse for split-KV."""
        acc_o, acc_lse = None, None
        if cutlass.const_expr(workspace is not None):
            workspace_H = cutlass.max(H, cutlass.Int32(128))
            align = 256 // self.q_dtype.width
            acc_o_layout = cute.make_layout(
                (workspace_H, split_kv, D, S, B),
                stride=(
                    cute.assume(split_kv * D, align),
                    cute.assume(D, align),
                    1,
                    cute.assume(split_kv * workspace_H * D, align),
                    cute.assume(workspace_H * split_kv * S * D, align),
                ),
            )
            acc_o_iter = cute.recast_ptr(workspace.iterator, dtype=acc_dtype)
            acc_o = cute.make_tensor(acc_o_iter, acc_o_layout)
            acc_lse_layout = cute.make_layout(
                (workspace_H, split_kv, S, B),
                stride=(
                    split_kv,
                    1,
                    workspace_H * split_kv,
                    workspace_H * split_kv * S,
                ),
            )
            acc_lse_iter = cute.recast_ptr(
                workspace.iterator + cute.cosize(acc_o_layout) * acc_dtype.width // 8,
                dtype=acc_dtype,
            )
            acc_lse = cute.make_tensor(acc_lse_iter, acc_lse_layout)
        return acc_o, acc_lse

    @staticmethod
    def get_split_kv(
        B: int, S: int, K: int, mma_qk_tiler_mn: tuple, max_active_blocks: int
    ) -> int:
        return mla_get_split_kv(B, S, K, mma_qk_tiler_mn, max_active_blocks)

    @staticmethod
    def get_split_kv_simplified(B: int, S: int, max_active_blocks: int) -> int:
        return mla_get_split_kv_simplified(B, S, max_active_blocks)

    @staticmethod
    def get_workspace_size(
        H: int,
        S: int,
        D: int,
        B: int,
        split_kv: int,
        acc_dtype: Type[cutlass.Numeric],
    ) -> int:
        return mla_get_workspace_size(H, S, D, B, split_kv, acc_dtype.width)

    @staticmethod
    def can_implement(
        B: int,
        S: int,
        K: int,
        H: int,
        L: int,
        R: int,
        in_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        lse_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        is_persistent: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        page_size: int,
    ) -> bool:
        return MLAConfig.can_implement_fp8(
            B,
            S,
            K,
            H,
            L,
            R,
            in_dtype,
            out_dtype,
            acc_dtype,
            lse_dtype,
            mma_qk_tiler_mn,
            mma_pv_tiler_mn,
            is_persistent,
            is_var_seq,
            is_var_split_kv,
            page_size,
        )
