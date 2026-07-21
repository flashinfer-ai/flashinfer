"""Native tcgen05 context-parallel GDN kernels for Blackwell SM100."""

from typing import Type

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05

from ..delta_rule_dsl.alpha import AlphaProcessor
from ..delta_rule_dsl.delta_rule_cp_sm120 import CPDeltaRuleMNPrecomputeSm120
from ..delta_rule_dsl.varlen_helper import (
    chunks_for_len,
    varlen_chunk_idx,
    varlen_chunk_valid_len,
)


class CPDeltaRuleMNPrecomputeUtcmma1Sm100(CPDeltaRuleMNPrecomputeSm120):
    """Compute local CP transfer/state while keeping both recurrences in TMEM."""

    def __init__(self, dtype: Type[cutlass.Numeric]):
        self.dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.BLK = 64
        self.D = 128

        self.k_stage = 1
        self.v_stage = 2
        self.t_stage = 1
        self.alpha_stage = 1

        self.compute_group_0_warp_ids = [0, 1, 2, 3]
        self.compute_group_1_warp_ids = [4, 5, 6, 7]
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.alpha_warp_id = 10
        self.idle_warp_id = 11
        self.threads_per_cta = 384

        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mnk = (1, 1, 1)
        self.is_two_sm = False
        self.use_multicast = False
        self.tmem_m_offset = 0
        self.tmem_n_offset = 128
        self.tmem_scratch_offset = 256
        self.tmem_m_inp_offset = 320
        self.tmem_n_inp_offset = 384

        self.num_regs_compute = 216
        self.num_regs_mma = 72
        self.num_regs_other = 24
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32
            * (
                len(self.compute_group_0_warp_ids)
                + len(self.compute_group_1_warp_ids)
                + 1
            ),
        )
        self.tmem_dealloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=1)
        self.manual_cache_key(
            "dtype", "acc_dtype", "BLK", "D", "is_two_sm", "use_multicast"
        )

    @staticmethod
    def transform_partitioned_layout(layout):
        stored_layout = layout
        if isinstance(layout, cute.ComposedLayout):
            layout = layout.outer
        shape = layout.shape
        stride = layout.stride
        new_layout = cute.make_layout(
            ((shape[0][0], shape[1]), (shape[0][1], shape[2]), *shape[3:]),
            stride=((stride[0][0], stride[1]), (stride[0][1], stride[2]), *stride[3:]),
        )
        if isinstance(stored_layout, cute.ComposedLayout):
            new_layout = cute.make_composed_layout(
                stored_layout.inner, stored_layout.offset, new_layout
            )
        return new_layout

    @staticmethod
    def transform_partitioned_tensor_layout(tensor: cute.Tensor) -> cute.Tensor:
        new_layout = CPDeltaRuleMNPrecomputeUtcmma1Sm100.transform_partitioned_layout(
            tensor.layout
        )
        return cute.make_tensor(tensor.iterator, new_layout)

    @cute.jit
    def load_tensor_block_tma_sm100(
        self,
        tma_atom,
        tma_tensor,
        sTensor,
        tensor_pipeline,
        tok_offset,
        t_block_start,
        head_idx,
        blk,
        is_transfer: bool,
        cta_coord,
        cta_layout,
        mcast_mask,
    ):
        handle = tensor_pipeline.acquire_and_advance()
        sTensor_stage = sTensor[None, None, handle.index]
        if cutlass.const_expr(is_transfer):
            mTensor = tma_tensor[None, None, head_idx, t_block_start + blk]
            gTensor = cute.zipped_divide(mTensor, (self.BLK, self.BLK))[
                ((None, None), (0, 0))
            ]
        else:
            mTensor = cute.domain_offset(
                (0, tok_offset + blk * self.BLK), tma_tensor[None, None, head_idx]
            )
            gTensor = cute.zipped_divide(mTensor, (self.D, self.BLK))[
                ((None, None), (0, 0))
            ]
        tTsT, tTgT = cpasync.tma_partition(
            tma_atom,
            cta_coord,
            cta_layout,
            cute.group_modes(sTensor_stage, 0, 2),
            cute.group_modes(gTensor, 0, 2),
        )
        if cutlass.const_expr(self.use_multicast):
            cute.copy(
                tma_atom,
                tTgT,
                tTsT,
                tma_bar_ptr=handle.barrier,
                mcast_mask=mcast_mask,
            )
        else:
            cute.copy(tma_atom, tTgT, tTsT, tma_bar_ptr=handle.barrier)
        return tensor_pipeline

    @cute.jit
    def run_load_alpha_role_sm100(
        self,
        g_alpha,
        sAlpha,
        alpha_producer,
        tok_offset,
        head_idx,
        chunk_len,
        num_blocks,
        num_heads,
        tidx,
    ):
        for blk in cutlass.range(num_blocks, unroll=1):
            handle = alpha_producer.acquire_and_advance()
            cute.arch.fence_view_async_shared()
            self.load_alpha_block(
                g_alpha,
                sAlpha,
                tok_offset,
                head_idx,
                blk,
                chunk_len,
                num_heads,
                handle.index,
                tidx,
            )
            AlphaProcessor().run(
                sAlpha[None, None, handle.index], cutlass.Float32(1.0), True
            )
            self.mask_alpha_oob_block(sAlpha, blk, chunk_len, handle.index, tidx)
            cute.arch.fence_view_async_shared()
            handle.commit()
        return alpha_producer

    @cute.jit
    def _initialize_tmem_matrix(
        self,
        tmem_ptr: cutlass.Int64,
        offset: int,
        tiled_mma,
        tidx: cutlass.Int32,
        identity: bool,
    ):
        acc_shape = tiled_mma.partition_shape_C((self.D, self.D))
        tCtC_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        tCtC = cute.make_tensor(tmem_ptr + offset, tCtC_fake.layout)
        tCtC_mn = self.transform_partitioned_tensor_layout(tCtC)
        cC = cute.make_identity_tensor((self.D, self.D))
        atom_r2t = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tiled_r2t = tcgen05.make_tmem_copy(atom_r2t, tCtC[(None, None), 0, 0, 0])
        thr_r2t = tiled_r2t.get_slice(tidx)
        tRT_cC = thr_r2t.partition_S(cC)
        tRT_tC = thr_r2t.partition_D(tCtC_mn)
        rC = cute.make_rmem_tensor_like(tRT_cC, self.acc_dtype)
        for i in cutlass.range(cute.size(rC), unroll_full=True):
            row, col = tRT_cC[i]
            rC[i] = (
                cutlass.Float32(1.0)
                if cutlass.const_expr(identity) and row == col
                else cutlass.Float32(0.0)
            )
        for sub in cutlass.range(cute.size(rC, mode=[2]), unroll_full=True):
            cute.copy(tiled_r2t, rC[None, 0, sub], tRT_tC[None, 0, sub, 0])
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def _scale_tmem_matrix(
        self,
        tmem_ptr: cutlass.Int64,
        offset: int,
        tiled_mma,
        tidx: cutlass.Int32,
        scale: cutlass.Float32,
    ):
        acc_shape = tiled_mma.partition_shape_C((self.D, self.D))
        tCtC_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        tCtC = cute.make_tensor(tmem_ptr + offset, tCtC_fake.layout)
        tCtC_mn = self.transform_partitioned_tensor_layout(tCtC)
        cC = cute.make_identity_tensor((self.D, self.D))
        atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        atom_r2t = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tiled_t2r = tcgen05.make_tmem_copy(atom_t2r, tCtC[(None, None), 0, 0, 0])
        tiled_r2t = tcgen05.make_tmem_copy(atom_r2t, tCtC[(None, None), 0, 0, 0])
        thr_t2r = tiled_t2r.get_slice(tidx)
        thr_r2t = tiled_r2t.get_slice(tidx)
        tTR_tC = thr_t2r.partition_S(tCtC_mn)
        tTR_cC = thr_t2r.partition_D(cC)
        tRT_tC = thr_r2t.partition_D(tCtC_mn)
        rC = cute.make_rmem_tensor_like(tTR_cC, self.acc_dtype)
        for sub in cutlass.range(cute.size(rC, mode=[2]), unroll_full=True):
            cute.copy(tiled_t2r, tTR_tC[None, 0, sub, 0], rC[None, 0, sub])
            for i in cutlass.range(cute.size(rC, mode=[0]), vectorize=True):
                rC[i, 0, sub] = rC[i, 0, sub] * scale
            cute.copy(tiled_r2t, rC[None, 0, sub], tRT_tC[None, 0, sub, 0])
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def _store_tmem_matrix(
        self,
        tmem_ptr: cutlass.Int64,
        offset: int,
        tiled_mma,
        tidx: cutlass.Int32,
        gC: cute.Tensor,
    ):
        acc_shape = tiled_mma.partition_shape_C((self.D, self.D))
        tCtC_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        tCtC = cute.make_tensor(tmem_ptr + offset, tCtC_fake.layout)
        tCtC_mn = self.transform_partitioned_tensor_layout(tCtC)
        cC = cute.make_identity_tensor((self.D, self.D))
        atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tiled_t2r = tcgen05.make_tmem_copy(atom_t2r, tCtC[(None, None), 0, 0, 0])
        thr_t2r = tiled_t2r.get_slice(tidx)
        tTR_tC = thr_t2r.partition_S(tCtC_mn)
        tTR_cC = thr_t2r.partition_D(cC)
        tCgC = thr_t2r.partition_D(gC)
        tCgC = cute.make_tensor(tCgC.iterator.align(16), tCgC.layout)
        rC = cute.make_rmem_tensor_like(tTR_cC, self.acc_dtype)
        atom_r2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.acc_dtype, num_bits_per_copy=128
        )
        for sub in cutlass.range(cute.size(rC, mode=[2]), unroll_full=True):
            cute.copy(tiled_t2r, tTR_tC[None, 0, sub, 0], rC[None, 0, sub])
            cute.copy(atom_r2g, rC[None, 0, sub], tCgC[None, 0, sub])

    @cute.jit
    def _materialize_x(
        self,
        tmem_ptr: cutlass.Int64,
        tiled_mma_x,
        tidx: cutlass.Int32,
        sX: cute.Tensor,
    ):
        acc_shape = tiled_mma_x.partition_shape_C((self.D, self.BLK))
        tCtX_fake = tiled_mma_x.make_fragment_C(cute.append(acc_shape, 1))
        tCtX = cute.make_tensor(tmem_ptr + self.tmem_scratch_offset, tCtX_fake.layout)
        tCtX_mn = self.transform_partitioned_tensor_layout(tCtX)
        cX = cute.make_identity_tensor((self.D, self.BLK))
        atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tiled_t2r = tcgen05.make_tmem_copy(atom_t2r, tCtX[(None, None), 0, 0, 0])
        thr_t2r = tiled_t2r.get_slice(tidx)
        tTR_tX = thr_t2r.partition_S(tCtX_mn)
        tTR_cX = thr_t2r.partition_D(cX)
        rX = cute.make_rmem_tensor_like(tTR_cX, self.acc_dtype)
        sX_mn = self.transform_partitioned_tensor_layout(sX)
        rX_out = cute.make_rmem_tensor_like(tTR_cX, self.dtype)
        atom_x_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.dtype,
        )
        tiled_x_r2s = cute.make_tiled_copy_D(atom_x_r2s, tiled_t2r)
        thr_x_r2s = tiled_x_r2s.get_slice(tidx)
        tXsX = thr_x_r2s.partition_D(sX_mn)
        tXsX = cute.make_tensor(tXsX.iterator.align(16), tXsX.layout)
        tXrX = tiled_x_r2s.retile(rX_out)
        for iter_m in cutlass.range(cute.size(rX, mode=[1]), unroll_full=True):
            for iter_n in cutlass.range(cute.size(rX, mode=[2]), unroll_full=True):
                cute.copy(
                    tiled_t2r,
                    tTR_tX[None, iter_m, iter_n, 0],
                    rX[None, iter_m, iter_n],
                )
                rX_out[None, iter_m, iter_n].store(
                    rX[None, iter_m, iter_n].load().to(self.dtype)
                )
                cute.copy(
                    tiled_x_r2s,
                    tXrX[None, iter_m, iter_n],
                    tXsX[None, iter_m, iter_n, 0],
                )
        cute.arch.fence_view_async_shared()

    @cute.jit
    def _convert_matrix_to_z_input(
        self,
        tmem_ptr: cutlass.Int64,
        src_offset: int,
        dst_offset: int,
        tiled_mma_update,
        tiled_mma_z,
        tidx: cutlass.Int32,
    ):
        src_shape = tiled_mma_update.partition_shape_C((self.D, self.D))
        tCtSrc_fake = tiled_mma_update.make_fragment_C(cute.append(src_shape, 1))
        tCtSrc = cute.make_tensor(tmem_ptr + src_offset, tCtSrc_fake.layout)
        tCtSrc_mn = self.transform_partitioned_tensor_layout(tCtSrc)
        cSrc = cute.make_identity_tensor((self.D, self.D))
        src_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        src_copy = tcgen05.make_tmem_copy(src_atom, tCtSrc[(None, None), 0, 0, 0])
        src_thr = src_copy.get_slice(tidx)
        tTR_tSrc = src_thr.partition_S(tCtSrc_mn)
        tTR_cSrc = src_thr.partition_D(cSrc)
        rSrc = cute.make_rmem_tensor_like(tTR_cSrc, self.acc_dtype)

        dst_shape = tiled_mma_z.partition_shape_A((self.D, self.D))
        tCtDst_fake = tiled_mma_z.make_fragment_A(cute.append(dst_shape, 1))
        tCtDst = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + dst_offset, dtype=self.dtype), tCtDst_fake.layout
        )
        tCtDst_mn = self.transform_partitioned_tensor_layout(tCtDst)
        cDst = cute.make_identity_tensor((self.D, self.D))
        dst_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), self.dtype
        )
        dst_copy = tcgen05.make_tmem_copy(dst_atom, tCtDst_mn[None, None, 0])
        dst_thr = dst_copy.get_slice(tidx)
        tRT_cDst = dst_thr.partition_S(cDst)
        tRT_tDst = dst_thr.partition_D(tCtDst_mn)
        rDst = cute.make_rmem_tensor_like(tRT_cDst, self.dtype)

        for iter_m in cutlass.range(cute.size(rSrc, mode=[1]), unroll_full=True):
            for iter_n in cutlass.range(cute.size(rSrc, mode=[2]), unroll_full=True):
                cute.copy(
                    src_copy,
                    tTR_tSrc[None, iter_m, iter_n, 0],
                    rSrc[None, iter_m, iter_n],
                )
                rDst[None, iter_m, iter_n].store(
                    rSrc[None, iter_m, iter_n].load().to(self.dtype)
                )
                cute.copy(
                    dst_copy,
                    rDst[None, iter_m, iter_n],
                    tRT_tDst[None, iter_m, iter_n, 0],
                )
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def _convert_scratch_to_update_input(
        self,
        tmem_ptr: cutlass.Int64,
        dst_offset: int,
        tiled_mma_x,
        tiled_mma_update,
        tidx: cutlass.Int32,
    ):
        src_shape = tiled_mma_x.partition_shape_C((self.D, self.BLK))
        tCtSrc_fake = tiled_mma_x.make_fragment_C(cute.append(src_shape, 1))
        tCtSrc = cute.make_tensor(
            tmem_ptr + self.tmem_scratch_offset, tCtSrc_fake.layout
        )
        tCtSrc_mn = self.transform_partitioned_tensor_layout(tCtSrc)
        cSrc = cute.make_identity_tensor((self.D, self.BLK))
        src_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        src_copy = tcgen05.make_tmem_copy(src_atom, tCtSrc[(None, None), 0, 0, 0])
        src_thr = src_copy.get_slice(tidx)
        tTR_tSrc = src_thr.partition_S(tCtSrc_mn)
        tTR_cSrc = src_thr.partition_D(cSrc)
        rSrc = cute.make_rmem_tensor_like(tTR_cSrc, self.acc_dtype)

        dst_shape = tiled_mma_update.partition_shape_A((self.D, self.BLK))
        tCtDst_fake = tiled_mma_update.make_fragment_A(cute.append(dst_shape, 1))
        tCtDst = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + dst_offset, dtype=self.dtype), tCtDst_fake.layout
        )
        tCtDst_mn = self.transform_partitioned_tensor_layout(tCtDst)
        cDst = cute.make_identity_tensor((self.D, self.BLK))
        dst_atom = cute.make_copy_atom(
            tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(8)), self.dtype
        )
        dst_copy = tcgen05.make_tmem_copy(dst_atom, tCtDst_mn[None, None, 0])
        dst_thr = dst_copy.get_slice(tidx)
        tRT_cDst = dst_thr.partition_S(cDst)
        tRT_tDst = dst_thr.partition_D(tCtDst_mn)
        rDst = cute.make_rmem_tensor_like(tRT_cDst, self.dtype)

        for iter_m in cutlass.range(cute.size(rSrc, mode=[1]), unroll_full=True):
            for iter_n in cutlass.range(cute.size(rSrc, mode=[2]), unroll_full=True):
                cute.copy(
                    src_copy,
                    tTR_tSrc[None, iter_m, iter_n, 0],
                    rSrc[None, iter_m, iter_n],
                )
                rDst[None, iter_m, iter_n].store(
                    rSrc[None, iter_m, iter_n].load().to(self.dtype)
                )
                cute.copy(
                    dst_copy,
                    rDst[None, iter_m, iter_n],
                    tRT_tDst[None, iter_m, iter_n, 0],
                )
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def _process_y(
        self,
        tmem_ptr: cutlass.Int64,
        tiled_mma_x,
        tidx: cutlass.Int32,
        sV: cute.Tensor,
        sAlpha: cute.Tensor,
        v_stage: cutlass.Int32,
        alpha_stage: cutlass.Int32,
        block_coeff: cutlass.Float32,
        tiled_mma_update,
    ):
        acc_shape = tiled_mma_x.partition_shape_C((self.D, self.BLK))
        tCtY_fake = tiled_mma_x.make_fragment_C(cute.append(acc_shape, 1))
        tCtY = cute.make_tensor(tmem_ptr + self.tmem_scratch_offset, tCtY_fake.layout)
        tCtY_mn = self.transform_partitioned_tensor_layout(tCtY)
        cY = cute.make_identity_tensor((self.D, self.BLK))
        atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tiled_t2r = tcgen05.make_tmem_copy(atom_t2r, tCtY[(None, None), 0, 0, 0])
        thr_t2r = tiled_t2r.get_slice(tidx)
        tTR_tY = thr_t2r.partition_S(tCtY_mn)
        tTR_cY = thr_t2r.partition_D(cY)
        rY = cute.make_rmem_tensor_like(tTR_cY, self.acc_dtype)
        sAlpha_blk = sAlpha[None, None, alpha_stage]

        dst_shape = tiled_mma_update.partition_shape_A((self.D, self.BLK))
        tCtDst_fake = tiled_mma_update.make_fragment_A(cute.append(dst_shape, 1))
        tCtDst = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_n_inp_offset, dtype=self.dtype),
            tCtDst_fake.layout,
        )
        tCtDst_mn = self.transform_partitioned_tensor_layout(tCtDst)
        cDst = cute.make_identity_tensor((self.D, self.BLK))
        dst_atom = cute.make_copy_atom(
            tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(8)), self.dtype
        )
        dst_copy = tcgen05.make_tmem_copy(dst_atom, tCtDst_mn[None, None, 0])
        dst_thr = dst_copy.get_slice(tidx)
        tRT_cDst = dst_thr.partition_S(cDst)
        tRT_tDst = dst_thr.partition_D(tCtDst_mn)
        rDst = cute.make_rmem_tensor_like(tRT_cDst, self.dtype)
        rV = cute.make_rmem_tensor_like(tRT_cDst, self.dtype)
        atom_v_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.dtype,
        )
        tiled_v_s2r = cute.make_tiled_copy_S(atom_v_s2r, dst_copy)
        thr_v_s2r = tiled_v_s2r.get_slice(tidx)
        tVsV = thr_v_s2r.partition_S(sV)
        tVrV = tiled_v_s2r.retile(rV)
        for iter_m in cutlass.range(cute.size(rY, mode=[1]), unroll_full=True):
            for iter_n in cutlass.range(cute.size(rY, mode=[2]), unroll_full=True):
                cute.copy(
                    tiled_t2r,
                    tTR_tY[None, iter_m, iter_n, 0],
                    rY[None, iter_m, iter_n],
                )
                cute.copy(
                    tiled_v_s2r,
                    tVsV[None, iter_m, iter_n, v_stage],
                    tVrV[None, iter_m, iter_n],
                )
                for i in cutlass.range(cute.size(rY, mode=[0]), unroll_full=True):
                    _, token = tTR_cY[i, iter_m, iter_n]
                    rY[i, iter_m, iter_n] = (
                        block_coeff * rY[i, iter_m, iter_n]
                        + cutlass.Float32(rV[i, iter_m, iter_n])
                        * sAlpha_blk[token, AlphaProcessor.CUMPROD_NEG_END_RCP]
                    )
                rDst[None, iter_m, iter_n].store(
                    rY[None, iter_m, iter_n].load().to(self.dtype)
                )
                cute.copy(
                    dst_copy,
                    rDst[None, iter_m, iter_n],
                    tRT_tDst[None, iter_m, iter_n, 0],
                )
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def run_mma_role(
        self,
        tmem_ptr: cutlass.Int64,
        mma_args: tuple,
        smem_args: tuple,
        load_pipelines: tuple,
        math_pipelines: tuple,
        num_blocks: cutlass.Int32,
        is_state,
    ):
        tiled_mma_x, tiled_mma_z, tiled_mma_update, tiled_mma_update_ss = mma_args
        sK, sK_trans, sT, sX = smem_args
        k_consumer, t_consumer = load_pipelines
        (
            m_init_consumer,
            n_init_consumer,
            x_acc_producer,
            x_ready_consumer,
            m_in_consumer,
            n_in_consumer,
            z_acc_producer,
            z_ready_consumer,
            m_acc_producer,
            y_acc_producer,
            y_ready_consumer,
            n_acc_producer,
        ) = math_pipelines

        if cutlass.const_expr(self.is_two_sm):
            if is_state:
                n_init_consumer.wait_and_advance().release()
            else:
                m_init_consumer.wait_and_advance().release()
        else:
            m_init_consumer.wait_and_advance().release()
            n_init_consumer.wait_and_advance().release()

        x_shape = tiled_mma_x.partition_shape_C((self.D, self.BLK))
        tCtScratch_fake = tiled_mma_x.make_fragment_C(cute.append(x_shape, 1))
        tCtScratch = cute.make_tensor(
            tmem_ptr + self.tmem_scratch_offset, tCtScratch_fake.layout
        )
        update_shape = tiled_mma_update.partition_shape_C((self.D, self.D))
        tCtUpdate_fake = tiled_mma_update.make_fragment_C(cute.append(update_shape, 1))
        tCtM = cute.make_tensor(tmem_ptr + self.tmem_m_offset, tCtUpdate_fake.layout)
        tCtN = cute.make_tensor(tmem_ptr + self.tmem_n_offset, tCtUpdate_fake.layout)

        z_inp_shape = tiled_mma_z.partition_shape_A((self.D, self.D))
        tCtZInp_fake = tiled_mma_z.make_fragment_A(cute.append(z_inp_shape, 1))
        tCtMInp = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_m_inp_offset, dtype=self.dtype),
            tCtZInp_fake.layout,
        )
        tCtNInp = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_n_inp_offset, dtype=self.dtype),
            tCtZInp_fake.layout,
        )
        update_inp_shape = tiled_mma_update.partition_shape_A((self.D, self.BLK))
        tCtUpdateInp_fake = tiled_mma_update.make_fragment_A(
            cute.append(update_inp_shape, 1)
        )
        tCtZInp = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_m_inp_offset, dtype=self.dtype),
            tCtUpdateInp_fake.layout,
        )
        tCtYInp = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_n_inp_offset, dtype=self.dtype),
            tCtUpdateInp_fake.layout,
        )

        tCrK_X = tiled_mma_x.make_fragment_A(sK)
        tCrT_X = tiled_mma_x.make_fragment_B(sT)
        tCrM_Z = tCtMInp
        tCrN_Y = tCtNInp
        tCrK_ZY = tiled_mma_z.make_fragment_B(sK_trans)
        tCrZ_Update = tCtZInp
        tCrY_Update = tCtYInp
        tCrX_Update = tiled_mma_update.make_fragment_B(sX)
        tCrK_Update = tiled_mma_update_ss.make_fragment_A(sK)

        for blk in cutlass.range(num_blocks, unroll=1):
            k_handle = k_consumer.wait_and_advance()
            t_handle = t_consumer.wait_and_advance()
            x_handle = x_acc_producer.acquire_and_advance()
            for kphase in cutlass.range(cute.size(tCrK_X, mode=[2]), unroll_full=True):
                tiled_mma_x.set(tcgen05.Field.ACCUMULATE, kphase != 0)
                cute.gemm(
                    tiled_mma_x,
                    tCtScratch[None, None, None, x_handle.index],
                    tCrK_X[None, None, kphase, k_handle.index],
                    tCrT_X[None, None, kphase, t_handle.index],
                    tCtScratch[None, None, None, x_handle.index],
                )
            x_handle.commit()
            t_handle.release()
            x_ready_consumer.wait_and_advance().release()
            if (not cutlass.const_expr(self.is_two_sm) or not is_state) and blk > 0:
                m_in_handle = m_in_consumer.wait_and_advance()
                for kphase in cutlass.range(
                    cute.size(tCrM_Z, mode=[2]), unroll_full=True
                ):
                    tiled_mma_z.set(tcgen05.Field.ACCUMULATE, kphase != 0)
                    cute.gemm(
                        tiled_mma_z,
                        tCtScratch[None, None, None, 0],
                        tCrM_Z[None, None, kphase, 0],
                        tCrK_ZY[None, None, kphase, k_handle.index],
                        tCtScratch[None, None, None, 0],
                    )
                z_handle = z_acc_producer.acquire_and_advance()
                z_handle.commit()
                z_ready_consumer.wait_and_advance().release()
                m_in_handle.release()

            if not cutlass.const_expr(self.is_two_sm) or not is_state:
                m_handle = m_acc_producer.acquire_and_advance()
                for kphase in cutlass.range(
                    cute.size(tCrX_Update, mode=[2]), unroll_full=True
                ):
                    tiled_mma_update.set(tcgen05.Field.ACCUMULATE, True)
                    if blk == 0:
                        tiled_mma_update_ss.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(
                            tiled_mma_update_ss,
                            tCtM[None, None, None, m_handle.index],
                            tCrK_Update[None, None, kphase, k_handle.index],
                            tCrX_Update[None, None, kphase, 0],
                            tCtM[None, None, None, m_handle.index],
                        )
                    else:
                        cute.gemm(
                            tiled_mma_update,
                            tCtM[None, None, None, m_handle.index],
                            tCrZ_Update[None, None, kphase, 0],
                            tCrX_Update[None, None, kphase, 0],
                            tCtM[None, None, None, m_handle.index],
                        )
                m_handle.commit()

            if not cutlass.const_expr(self.is_two_sm) or is_state:
                n_in_handle = n_in_consumer.wait_and_advance()
                y_handle = y_acc_producer.acquire_and_advance()
                for kphase in cutlass.range(
                    cute.size(tCrN_Y, mode=[2]), unroll_full=True
                ):
                    tiled_mma_z.set(tcgen05.Field.ACCUMULATE, kphase != 0)
                    cute.gemm(
                        tiled_mma_z,
                        tCtScratch[None, None, None, y_handle.index],
                        tCrN_Y[None, None, kphase, 0],
                        tCrK_ZY[None, None, kphase, k_handle.index],
                        tCtScratch[None, None, None, y_handle.index],
                    )
                y_handle.commit()
                n_in_handle.release()
            k_handle.release()

            if not cutlass.const_expr(self.is_two_sm) or is_state:
                y_ready_consumer.wait_and_advance().release()
                n_handle = n_acc_producer.acquire_and_advance()
                for kphase in cutlass.range(
                    cute.size(tCrX_Update, mode=[2]), unroll_full=True
                ):
                    tiled_mma_update.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(
                        tiled_mma_update,
                        tCtN[None, None, None, n_handle.index],
                        tCrY_Update[None, None, kphase, 0],
                        tCrX_Update[None, None, kphase, 0],
                        tCtN[None, None, None, n_handle.index],
                    )
                n_handle.commit()

    @cute.jit
    def run_compute_group_0(
        self,
        tidx: cutlass.Int32,
        tmem_ptr: cutlass.Int64,
        tiled_mma_x,
        tiled_mma_z,
        tiled_mma_update,
        sX: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_consumer,
        math_pipelines: tuple,
        valid_chunk_len: cutlass.Int32,
        num_blocks: cutlass.Int32,
        gM: cute.Tensor,
    ):
        (
            m_init_producer,
            x_acc_consumer,
            x_ready_producer,
            m_in_producer,
            z_acc_consumer,
            z_ready_producer,
            m_acc_consumer,
            done_producer,
        ) = math_pipelines
        self._initialize_tmem_matrix(
            tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, True
        )
        m_init_producer.acquire_and_advance().commit()
        for blk in cutlass.range(num_blocks, unroll=1):
            alpha_handle = alpha_consumer.wait_and_advance()
            x_handle = x_acc_consumer.wait_and_advance()
            self._materialize_x(tmem_ptr, tiled_mma_x, tidx, sX)
            x_handle.release()
            x_ready_producer.acquire_and_advance().commit()

            if blk > 0:
                self._convert_matrix_to_z_input(
                    tmem_ptr,
                    self.tmem_m_offset,
                    self.tmem_m_inp_offset,
                    tiled_mma_update,
                    tiled_mma_z,
                    tidx,
                )
                m_in_producer.acquire_and_advance().commit()
                z_handle = z_acc_consumer.wait_and_advance()
                self._convert_scratch_to_update_input(
                    tmem_ptr,
                    self.tmem_m_inp_offset,
                    tiled_mma_x,
                    tiled_mma_update,
                    tidx,
                )
                z_handle.release()
                z_ready_producer.acquire_and_advance().commit()

            block_start = blk * self.BLK
            valid_len = valid_chunk_len - block_start
            if valid_len > self.BLK:
                valid_len = self.BLK
            block_coeff = cutlass.Float32(
                sAlpha[valid_len - 1, AlphaProcessor.CUMPROD, alpha_handle.index]
            )
            m_handle = m_acc_consumer.wait_and_advance()
            self._scale_tmem_matrix(
                tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, block_coeff
            )
            m_handle.release()
            alpha_handle.release()
        self._store_tmem_matrix(
            tmem_ptr, self.tmem_m_offset, tiled_mma_update, tidx, gM
        )
        done_producer.acquire_and_advance().commit()

    @cute.jit
    def run_compute_group_1(
        self,
        tidx: cutlass.Int32,
        tmem_ptr: cutlass.Int64,
        tiled_mma_x,
        tiled_mma_z,
        tiled_mma_update,
        sX: cute.Tensor,
        sV: cute.Tensor,
        sAlpha: cute.Tensor,
        v_consumer,
        alpha_consumer,
        math_pipelines: tuple,
        valid_chunk_len: cutlass.Int32,
        num_blocks: cutlass.Int32,
        gN: cute.Tensor,
    ):
        (
            n_init_producer,
            x_acc_consumer,
            x_ready_producer,
            n_in_producer,
            y_acc_consumer,
            y_ready_producer,
            n_acc_consumer,
            done_consumer,
        ) = math_pipelines
        self._initialize_tmem_matrix(
            tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, False
        )
        n_init_producer.acquire_and_advance().commit()
        for blk in cutlass.range(num_blocks, unroll=1):
            if cutlass.const_expr(self.is_two_sm):
                x_handle = x_acc_consumer.wait_and_advance()
                self._materialize_x(tmem_ptr, tiled_mma_x, tidx, sX)
                x_handle.release()
                x_ready_producer.acquire_and_advance().commit()
            v_handle = v_consumer.wait_and_advance()
            alpha_handle = alpha_consumer.wait_and_advance()
            self._convert_matrix_to_z_input(
                tmem_ptr,
                self.tmem_n_offset,
                self.tmem_n_inp_offset,
                tiled_mma_update,
                tiled_mma_z,
                tidx,
            )
            n_in_producer.acquire_and_advance().commit()
            y_handle = y_acc_consumer.wait_and_advance()
            block_start = blk * self.BLK
            valid_len = valid_chunk_len - block_start
            if valid_len > self.BLK:
                valid_len = self.BLK
            block_coeff = cutlass.Float32(
                sAlpha[valid_len - 1, AlphaProcessor.CUMPROD, alpha_handle.index]
            )
            self._process_y(
                tmem_ptr,
                tiled_mma_x,
                tidx,
                sV,
                sAlpha,
                v_handle.index,
                alpha_handle.index,
                block_coeff,
                tiled_mma_update,
            )
            y_handle.release()
            self._scale_tmem_matrix(
                tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, block_coeff
            )
            cute.arch.fence_view_async_tmem_store()
            y_ready_producer.acquire_and_advance().commit()
            n_acc_consumer.wait_and_advance().release()
            v_handle.release()
            alpha_handle.release()
        self._store_tmem_matrix(
            tmem_ptr, self.tmem_n_offset, tiled_mma_update, tidx, gN
        )
        if not cutlass.const_expr(self.is_two_sm):
            done_consumer.wait_and_advance().release()

    @cute.jit
    def __call__(
        self,
        g_k: cute.Tensor,
        g_v: cute.Tensor,
        g_t: cute.Tensor,
        g_alpha: cute.Tensor,
        g_transfer_t: cute.Tensor,
        g_state_t: cute.Tensor,
        g_cu_seqlens: cute.Tensor,
        chunk_len: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        max_cp_chunks_per_seq: cutlass.Int32,
        num_seqs: cutlass.Int32,
        stream,
    ):
        def make_mma(mnk, source_a, a_major, b_major):
            op = tcgen05.MmaF16BF16Op(
                self.dtype,
                self.acc_dtype,
                (mnk[0], mnk[1], 16),
                self.cta_group,
                source_a,
                a_major,
                b_major,
            )
            return cute.make_tiled_mma(op)

        tiled_mma_x = make_mma(
            (self.D, self.BLK, self.BLK),
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.MN,
            OperandMajorMode.K,
        )
        tiled_mma_z = make_mma(
            (self.D, self.BLK, self.D),
            tcgen05.OperandSource.TMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        tiled_mma_update = make_mma(
            (self.D, self.D, self.BLK),
            tcgen05.OperandSource.TMEM,
            OperandMajorMode.K,
            OperandMajorMode.MN,
        )
        tiled_mma_update_ss = make_mma(
            (self.D, self.D, self.BLK),
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.MN,
            OperandMajorMode.MN,
        )
        k_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_x, (self.D, self.BLK, self.BLK), self.dtype, self.k_stage
        )
        k_trans_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_z, (self.D, self.BLK, self.D), self.dtype, self.k_stage
        )
        v_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_x, (self.D, self.BLK, self.BLK), self.dtype, self.v_stage
        )
        t_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_x, (self.D, self.BLK, self.BLK), self.dtype, self.t_stage
        )
        x_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_update, (self.D, self.D, self.BLK), self.dtype, 1
        )
        k_semantic_layout = self.transform_partitioned_layout(k_layout)
        v_semantic_layout = self.transform_partitioned_layout(v_layout)
        t_semantic_layout = self.transform_partitioned_layout(t_layout)
        alpha_layout = cute.make_layout(
            (self.BLK, AlphaProcessor.NUM_CHANNELS, self.alpha_stage)
        )

        # The physical cluster is laid out along grid-x. Model that rank as a
        # logical N coordinate so K/T/V are broadcast rather than partitioned.
        cluster_layout_vmnk = cute.make_layout((1, 1, 2 if self.is_two_sm else 1, 1))
        tma_op = (
            cpasync.CopyBulkTensorTileG2SMulticastOp(self.cta_group)
            if self.use_multicast
            else cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        )
        k_tma_layout = cute.slice_(k_semantic_layout, (None, None, 0))
        v_tma_layout = cute.slice_(v_semantic_layout, (None, None, 0))
        t_tma_layout = cute.slice_(t_semantic_layout, (None, None, 0))
        if cutlass.const_expr(self.use_multicast):
            tma_k = cpasync.make_tiled_tma_atom(
                tma_op, g_k, k_tma_layout, (self.D, self.BLK), 2
            )
            tma_v = cpasync.make_tiled_tma_atom(
                tma_op, g_v, v_tma_layout, (self.D, self.BLK), 2
            )
            tma_t = cpasync.make_tiled_tma_atom(
                tma_op, g_t, t_tma_layout, (self.BLK, self.BLK), 2
            )
        else:
            tma_k = cpasync.make_tiled_tma_atom(
                tma_op, g_k, k_tma_layout, (self.D, self.BLK)
            )
            tma_v = cpasync.make_tiled_tma_atom(
                tma_op, g_v, v_tma_layout, (self.D, self.BLK)
            )
            tma_t = cpasync.make_tiled_tma_atom(
                tma_op, g_t, t_tma_layout, (self.BLK, self.BLK)
            )
        tma_atom_k, tma_tensor_k = tma_k
        tma_atom_v, tma_tensor_v = tma_v
        tma_atom_t, tma_tensor_t = tma_t
        dtype_bytes = self.dtype.width // 8
        self.tma_k_bytes = cute.size(k_tma_layout) * dtype_bytes
        self.tma_v_bytes = cute.size(v_tma_layout) * dtype_bytes
        self.tma_t_bytes = cute.size(t_tma_layout) * dtype_bytes

        @cute.struct
        class SharedStorage:
            load_k_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            load_v_mbar: cute.struct.MemRange[cutlass.Int64, self.v_stage * 2]
            load_t_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            alpha_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            m_init_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            n_init_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            x_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            x_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            m_in_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            n_in_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            z_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            z_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            m_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            y_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            y_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            n_acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            done_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32
            sK: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(k_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(v_layout)], 1024
            ]
            sT: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(t_layout)], 1024
            ]
            sX: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(x_layout)], 1024
            ]
            sAlpha: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(alpha_layout)], 16
            ]

        self.shared_storage = SharedStorage  # type: ignore[assignment]
        self.kernel(
            tiled_mma_x,
            tiled_mma_z,
            tiled_mma_update,
            tiled_mma_update_ss,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_t,
            tma_tensor_t,
            g_alpha,
            g_transfer_t,
            g_state_t,
            g_cu_seqlens,
            k_layout,
            k_trans_layout,
            v_layout,
            t_layout,
            x_layout,
            alpha_layout,
            cluster_layout_vmnk,
            chunk_len,
            num_k_heads,
            num_v_heads,
            num_sab_heads,
            total_cp_chunks,
            num_seqs,
        ).launch(
            grid=(
                (2 if self.is_two_sm else 1) * num_sab_heads * max_cp_chunks_per_seq,
                num_seqs,
                1,
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma_x,
        tiled_mma_z,
        tiled_mma_update,
        tiled_mma_update_ss,
        tma_atom_k,
        tma_tensor_k,
        tma_atom_v,
        tma_tensor_v,
        tma_atom_t,
        tma_tensor_t,
        g_alpha,
        g_transfer_t,
        g_state_t,
        g_cu_seqlens,
        k_layout,
        k_trans_layout,
        v_layout,
        t_layout,
        x_layout,
        alpha_layout,
        cluster_layout_vmnk,
        chunk_len,
        num_k_heads,
        num_v_heads,
        num_sab_heads,
        total_cp_chunks,
        num_seqs,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bx, seq_idx, _ = cute.arch.block_idx()
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        is_state = cta_rank_in_cluster == 0
        if cutlass.const_expr(self.is_two_sm):
            bx = bx // 2
        sab_head_idx = bx % num_sab_heads
        chunk_idx_in_seq = bx // num_sab_heads
        k_head_idx = sab_head_idx * num_k_heads // num_sab_heads
        v_head_idx = sab_head_idx * num_v_heads // num_sab_heads
        seq_start = cutlass.Int32(g_cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(g_cu_seqlens[seq_idx + 1])
        seq_len = seq_end - seq_start
        num_cp_chunks = chunks_for_len(seq_len, chunk_len)
        is_valid_chunk = chunk_idx_in_seq < num_cp_chunks

        tok_offset = seq_start + chunk_idx_in_seq * chunk_len
        cp_chunk_idx = varlen_chunk_idx(seq_idx, seq_start, chunk_idx_in_seq, chunk_len)
        valid_chunk_len = varlen_chunk_valid_len(seq_len, chunk_idx_in_seq, chunk_len)
        num_blocks = cute.ceil_div(valid_chunk_len, self.BLK)
        t_blocks_per_cp_chunk = cute.ceil_div(chunk_len, self.BLK)
        t_block_start = varlen_chunk_idx(
            seq_idx, seq_start, chunk_idx_in_seq * t_blocks_per_cp_chunk, self.BLK
        )

        allocator = utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        sK = storage.sK.get_tensor(k_layout.outer, swizzle=k_layout.inner)
        sK_trans = storage.sK.get_tensor(
            k_trans_layout.outer, swizzle=k_trans_layout.inner
        )
        sV = storage.sV.get_tensor(v_layout.outer, swizzle=v_layout.inner)
        sT = storage.sT.get_tensor(t_layout.outer, swizzle=t_layout.inner)
        sX = storage.sX.get_tensor(x_layout.outer, swizzle=x_layout.inner)
        sK_tma = self.transform_partitioned_tensor_layout(sK)
        sV_tma = self.transform_partitioned_tensor_layout(sV)
        sT_tma = self.transform_partitioned_tensor_layout(sT)
        sAlpha = storage.sAlpha.get_tensor(alpha_layout)

        def cg(num_threads):
            return pipeline.CooperativeGroup(pipeline.Agent.Thread, num_threads)

        cg_tma = cg(1)
        cg_mma = cg(1)
        cg_mma_load = cg(2 if self.is_two_sm else 1)
        cg_alpha = cg(32)
        cg_cg0 = cg(128)
        cg_cg1 = cg(128)
        cg_cg1_v = cg(128 if self.use_multicast else 4)
        cg_both = cg(128 if self.is_two_sm else 256)
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=1,
            producer_group=cg_tma,
            consumer_group=cg_mma_load,
            tx_count=self.tma_k_bytes,
            barrier_storage=storage.load_k_mbar.data_ptr(),
            defer_sync=True,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(1, 0),
        ).make_participants()
        load_t_producer, load_t_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=1,
            producer_group=cg_tma,
            consumer_group=cg_mma_load,
            tx_count=self.tma_t_bytes,
            barrier_storage=storage.load_t_mbar.data_ptr(),
            defer_sync=True,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(1, 0),
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.v_stage,
            producer_group=cg_tma,
            consumer_group=cg_cg1_v,
            tx_count=self.tma_v_bytes,
            barrier_storage=storage.load_v_mbar.data_ptr(),
            defer_sync=True,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(1, 0),
            enable_multicast_signaling=self.is_two_sm,
        ).make_participants()
        alpha_producer, alpha_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=cg_alpha,
            consumer_group=cg_both,
            barrier_storage=storage.alpha_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        m_init_producer, m_init_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.m_init_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        n_init_producer, n_init_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg1,
            consumer_group=cg_mma,
            barrier_storage=storage.n_init_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        x_acc_producer, x_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=cg_mma,
            consumer_group=cg_cg0,
            barrier_storage=storage.x_acc_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        x_ready_producer, x_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.x_ready_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        m_in_producer, m_in_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.m_in_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        n_in_producer, n_in_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg1,
            consumer_group=cg_mma,
            barrier_storage=storage.n_in_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        z_acc_producer, z_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=cg_mma,
            consumer_group=cg_cg0,
            barrier_storage=storage.z_acc_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        z_ready_producer, z_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.z_ready_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        m_acc_producer, m_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=cg_mma,
            consumer_group=cg_cg0,
            barrier_storage=storage.m_acc_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        y_acc_producer, y_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.y_acc_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        y_ready_producer, y_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cg_cg1,
            consumer_group=cg_mma,
            barrier_storage=storage.y_ready_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        n_acc_producer, n_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.n_acc_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()
        done_producer, done_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=cg_cg0,
            consumer_group=cg_cg1,
            barrier_storage=storage.done_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()

        pipeline.pipeline_init_arrive(
            cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True
        )
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=(
                self.compute_group_0_warp_ids[0]
                if self.is_two_sm
                else self.compute_group_1_warp_ids[0]
            ),
        )
        tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))

        out_layout = cute.make_layout(
            (self.D, self.D, num_sab_heads, total_cp_chunks),
            stride=(self.D, 1, self.D * self.D, self.D * self.D * num_sab_heads),
        )
        mM = cute.make_tensor(g_transfer_t.iterator, out_layout)
        mN = cute.make_tensor(g_state_t.iterator, out_layout)
        gM = mM[None, None, sab_head_idx, cp_chunk_idx]
        gN = mN[None, None, sab_head_idx, cp_chunk_idx]

        tensor_cta_coord = 0
        tensor_cta_layout = cute.make_layout(1)
        tma_mcast_mask = None
        if cutlass.const_expr(self.use_multicast):
            tensor_cta_coord = block_in_cluster_coord_vmnk[2]
            tensor_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )
            tma_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )

        if not is_valid_chunk:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            if not cutlass.const_expr(self.is_two_sm):
                if warp_idx >= 0 and warp_idx <= self.mma_warp_id:
                    tmem.wait_for_alloc()
                if warp_idx == self.compute_group_1_warp_ids[0]:
                    tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                    tmem.relinquish_alloc_permit()
                    tmem.free(tmem_ptr)
            elif is_state:
                if (
                    warp_idx >= self.compute_group_1_warp_ids[0]
                    and warp_idx <= self.compute_group_1_warp_ids[-1]
                ) or warp_idx == self.mma_warp_id:
                    tmem.wait_for_alloc()
                if (
                    warp_idx >= self.compute_group_1_warp_ids[0]
                    and warp_idx <= self.compute_group_1_warp_ids[-1]
                ):
                    self.tmem_dealloc_barrier.sync()
                elif warp_idx == self.compute_group_0_warp_ids[0]:
                    self.tmem_dealloc_barrier.sync()
                    tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                    tmem.relinquish_alloc_permit()
                    tmem.free(tmem_ptr)
            elif (
                warp_idx >= self.compute_group_0_warp_ids[0]
                and warp_idx <= self.compute_group_0_warp_ids[-1]
            ) or warp_idx == self.mma_warp_id:
                tmem.wait_for_alloc()
                self.tmem_dealloc_barrier.sync()
                if warp_idx == self.compute_group_0_warp_ids[0]:
                    tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                    tmem.relinquish_alloc_permit()
                    tmem.free(tmem_ptr)
        elif (
            is_valid_chunk
            and cutlass.const_expr(self.is_two_sm)
            and is_state
            and warp_idx == self.compute_group_0_warp_ids[0]
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.tmem_dealloc_barrier.sync()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        elif (
            is_valid_chunk
            and warp_idx >= 0
            and warp_idx <= 3
            and (not cutlass.const_expr(self.is_two_sm) or not is_state)
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_compute_group_0(
                tidx,
                tmem_ptr,
                tiled_mma_x,
                tiled_mma_z,
                tiled_mma_update,
                sX,
                sAlpha,
                alpha_consumer,
                (
                    m_init_producer,
                    x_acc_consumer,
                    x_ready_producer,
                    m_in_producer,
                    z_acc_consumer,
                    z_ready_producer,
                    m_acc_consumer,
                    done_producer,
                ),
                valid_chunk_len,
                num_blocks,
                gM,
            )
            if cutlass.const_expr(self.is_two_sm):
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)
        elif (
            is_valid_chunk
            and warp_idx >= 4
            and warp_idx <= 7
            and (not cutlass.const_expr(self.is_two_sm) or is_state)
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_compute_group_1(
                tidx % 128,
                tmem_ptr,
                tiled_mma_x,
                tiled_mma_z,
                tiled_mma_update,
                sX,
                sV_tma,
                sAlpha,
                load_v_consumer,
                alpha_consumer,
                (
                    n_init_producer,
                    x_acc_consumer,
                    x_ready_producer,
                    n_in_producer,
                    y_acc_consumer,
                    y_ready_producer,
                    n_acc_consumer,
                    done_consumer,
                ),
                valid_chunk_len,
                num_blocks,
                gN,
            )
            if cutlass.const_expr(self.is_two_sm):
                self.tmem_dealloc_barrier.sync()
            else:
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)
        elif (
            is_valid_chunk
            and cutlass.const_expr(self.is_two_sm)
            and not is_state
            and warp_idx >= 4
            and warp_idx <= 7
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            if cutlass.const_expr(self.use_multicast):
                for _ in cutlass.range(num_blocks, unroll=1):
                    load_v_consumer.wait_and_advance().release()
        elif is_valid_chunk and warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_t)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            self.run_mma_role(
                tmem_ptr,
                (tiled_mma_x, tiled_mma_z, tiled_mma_update, tiled_mma_update_ss),
                (sK, sK_trans, sT, sX),
                (load_k_consumer, load_t_consumer),
                (
                    m_init_consumer,
                    n_init_consumer,
                    x_acc_producer,
                    x_ready_consumer,
                    m_in_consumer,
                    n_in_consumer,
                    z_acc_producer,
                    z_ready_consumer,
                    m_acc_producer,
                    y_acc_producer,
                    y_ready_consumer,
                    n_acc_producer,
                ),
                num_blocks,
                is_state,
            )
        elif is_valid_chunk and warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            for blk in cutlass.range(num_blocks, unroll=1):
                load_k_producer = self.load_tensor_block_tma_sm100(
                    tma_atom_k,
                    tma_tensor_k,
                    sK_tma,
                    load_k_producer,
                    tok_offset,
                    t_block_start,
                    k_head_idx,
                    blk,
                    False,
                    tensor_cta_coord,
                    tensor_cta_layout,
                    tma_mcast_mask,
                )
                if (
                    not cutlass.const_expr(self.is_two_sm)
                    or is_state
                    or cutlass.const_expr(self.use_multicast)
                ):
                    load_v_producer = self.load_tensor_block_tma_sm100(
                        tma_atom_v,
                        tma_tensor_v,
                        sV_tma,
                        load_v_producer,
                        tok_offset,
                        t_block_start,
                        v_head_idx,
                        blk,
                        False,
                        tensor_cta_coord,
                        tensor_cta_layout,
                        tma_mcast_mask,
                    )
                load_t_producer = self.load_tensor_block_tma_sm100(
                    tma_atom_t,
                    tma_tensor_t,
                    sT_tma,
                    load_t_producer,
                    tok_offset,
                    t_block_start,
                    sab_head_idx,
                    blk,
                    True,
                    tensor_cta_coord,
                    tensor_cta_layout,
                    tma_mcast_mask,
                )
        elif is_valid_chunk and warp_idx == self.alpha_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            alpha_producer = self.run_load_alpha_role_sm100(
                g_alpha,
                sAlpha,
                alpha_producer,
                tok_offset,
                sab_head_idx,
                valid_chunk_len,
                num_blocks,
                num_sab_heads,
                tidx,
            )
        else:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
