# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported from Block-Sparse-Attention's csrc/fwd/sm120_blk64/bsa_fwd_sm120_sage.py
# (native SM120 Sage QK-INT8 / PV-FP8 block-sparse attention forward kernel).
# Structurally a sibling of flash_fwd_sm120.py: same TMA/pipeline/scheduler
# infrastructure, but QK runs through a custom INT8 warp MMA (MmaInt8Op, the
# SM80-era m16n8k32 s8s8s32 atom -- SM120 has no newer native INT8 tensor-core
# instruction) with an INT32 accumulator folded straight into a log2-domain
# online softmax, and PV runs through the native FP8 warp MMA. Does not
# compute LSE (matches the upstream kernel; flashinfer's own vsa_sm120_blk64
# bf16/fp16 kernel does compute LSE, but this Sage kernel is exposed as a
# standalone function, not through the BlockSparseAttentionWrapper/
# _vsa_run_core dispatch path that requires it).

import math
import operator
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cutlass_dsl import T
from cutlass.cute.nvgpu.warp import mma as warp_mma
import cuda.bindings.driver as cuda

from . import layout_utils
from . import kernel_utils
from .batched_static_scheduler import BatchedStaticSchedulerMixin


SM120_SAGE_FWD_BLOCK_SIZE = 64
SAGE_P_QUANT_SCALE = 256.0
SAGE_P_QUANT_LOG2_SCALE = math.log2(SAGE_P_QUANT_SCALE)
SAGE_P_RESCALE_THRESHOLD = math.log2(448.0 / SAGE_P_QUANT_SCALE)


class MmaInt8Trait(warp_mma.Trait):
    pass


@dataclass(frozen=True)
class MmaInt8Op(warp_mma.WarpMmaOp):
    """Warp-level signed INT8 MMA with INT32 accumulation."""

    ab_dtype: type[cutlass.Numeric]
    acc_dtype: type[cutlass.Numeric]
    shape_mnk: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.ab_dtype is not cutlass.Int8:
            raise TypeError("MmaInt8Op requires signed Int8 operands")
        if self.acc_dtype is not cutlass.Int32:
            raise TypeError("MmaInt8Op requires Int32 accumulation")
        if self.shape_mnk != (16, 8, 32):
            raise ValueError("MmaInt8Op requires the m16n8k32 instruction shape")

    def _make_trait(self, *, loc=None, ip=None, **kwargs):
        shape_mnk = warp_mma._pack_shape(self.shape_mnk, loc=loc, ip=ip)
        atom_type = warp_mma._cute_nvgpu_ir.MmaAtomSM80Type.get(
            shape_mnk.type.attribute,
            T.si8(),
            T.si8(),
            self.acc_dtype.mlir_type,
        )
        return MmaInt8Trait(warp_mma.make_atom(atom_type, loc=loc, ip=ip))

    def _verify_fragment_A(self, input, *, loc=None, ip=None) -> bool:
        return True

    def _verify_fragment_B(self, input, *, loc=None, ip=None) -> bool:
        return True


# =============================================================================
# Public kernel class
# =============================================================================
class BlockSparseAttnForwardSageSm120Blk64(BatchedStaticSchedulerMixin):
    def __init__(
        self,
        gqa_ratio: int = 1,
        head_dim: int = 128,
        value_dim: int = 128,
        blocksparse_blocksize_q: int = 64,
        blocksparse_blocksize_k: int = 64,
        has_block_sizes: bool = True,
        has_block_nums: bool = True,
        block_sizes_mode: int = 0,
    ):
        self.qk_acc_dtype = cutlass.Int32
        self.pv_acc_dtype = cutlass.Float16
        self.output_acc_dtype = cutlass.Float32
        self.softmax_p_scale_log2 = SAGE_P_QUANT_LOG2_SCALE
        self.softmax_rescale_threshold = SAGE_P_RESCALE_THRESHOLD

        self.tile_size = SM120_SAGE_FWD_BLOCK_SIZE

        assert blocksparse_blocksize_q == self.tile_size, (
            "Only block_size_m=64 is supported in this kernel."
        )
        assert blocksparse_blocksize_k == self.tile_size, (
            "Only block_size_n=64 is supported in this kernel."
        )
        self.num_threads = 128
        self.kv_stage = 1
        self.q_stage = 1

        assert gqa_ratio >= 1
        assert head_dim == 128, "SM120 blk64 fwd currently requires QK dim 128"
        assert value_dim == 128, "SM120 blk64 fwd currently requires value dim 128"
        self.gqa_ratio = gqa_ratio
        self.qk_dim = head_dim
        self.value_dim = value_dim
        self.tile_shape_qk = (self.tile_size, self.tile_size, self.qk_dim)
        self.tile_shape_pv = (self.tile_size, self.value_dim, self.tile_size)

        self.has_block_sizes = has_block_sizes
        self.has_block_nums = has_block_nums
        self.block_sizes_mode = block_sizes_mode

    def _check_dim(self, tensor: cute.Tensor | list[cute.Tensor], mode: int):
        if isinstance(tensor, list):
            for t in tensor:
                self._check_dim(t, mode)
            return
        assert tensor.shape[mode] == 128, f"dim must be 128 in mode {mode}."
        assert tensor.stride[mode] == 1, f"dim must be contiguous in mode {mode}."

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mVScale: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        blocksparse_indices_q2k: cute.Tensor,
        blocksparse_num_blocks_q2k: cute.Tensor,
        block_sparse_num: cutlass.Int32,
        blocksparse_varblk: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        Q_smem_layout: cute.ComposedLayout,
        K_smem_layout: cute.ComposedLayout,
        V_smem_layout: cute.ComposedLayout,
        O_smem_layout: cute.ComposedLayout,
        scale_softmax_log2e: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        work_desc = self.get_work_desc()
        seqlen = mK.shape[0]
        num_compute_tiles = cute.ceil_div(seqlen, self.tile_size)

        shared_storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage_t)

        if warp_idx == 0 and lane_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_Q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_K)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_V)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_O)

        cg = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # Q, K, and V use independent TMA pipelines.
        Q_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.q_stage,
            producer_group=cg,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_threads // 32
            ),
            tx_count=cute.size_in_bytes(
                self.Q_dtype, cute.select(Q_smem_layout, mode=[0, 1])
            ),
            barrier_storage=shared_storage.Q_barrier.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        Q_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.q_stage
        )
        Q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stage
        )
        K_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stage,
            producer_group=cg,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_threads // 32
            ),
            tx_count=cute.size_in_bytes(
                self.K_dtype, cute.select(K_smem_layout, mode=[0, 1])
            ),
            barrier_storage=shared_storage.K_barrier.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        V_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stage,
            producer_group=cg,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_threads // 32
            ),
            tx_count=cute.size_in_bytes(
                self.V_dtype, cute.select(V_smem_layout, mode=[0, 1])
            ),
            barrier_storage=shared_storage.V_barrier.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        K_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        K_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        V_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        V_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )

        # Partition tensors.
        sQ = shared_storage.Q_smem.get_tensor(
            Q_smem_layout.outer,
            swizzle=Q_smem_layout.inner,
        )
        sK = shared_storage.K_smem.get_tensor(
            K_smem_layout.outer, swizzle=K_smem_layout.inner
        )
        sV = shared_storage.V_smem.get_tensor(
            V_smem_layout.outer, swizzle=V_smem_layout.inner
        )

        mO_slice = mO[None, None, work_desc.qo_head_idx, work_desc.batch_idx]
        mQ_slice = mQ[None, None, work_desc.qo_head_idx, work_desc.batch_idx]
        mK_slice = mK[None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        mV_slice = mV[None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        gQScale = mQScale[None, work_desc.qo_head_idx, work_desc.batch_idx]
        gKScale = mKScale[None, work_desc.kv_head_idx, work_desc.batch_idx]
        gVScale = mVScale[None, work_desc.kv_head_idx, work_desc.batch_idx]
        gO = cute.local_tile(
            mO_slice,
            (self.tile_shape_pv[0], self.tile_shape_pv[1]),
            coord=(work_desc.qo_tile_idx, 0),
        )
        gQ = cute.local_tile(
            mQ_slice,
            (self.tile_shape_qk[0], self.tile_shape_qk[2]),
            coord=(work_desc.qo_tile_idx, 0),
        )
        gK = cute.local_tile(
            mK_slice,
            (self.tile_shape_qk[1], self.tile_shape_qk[2]),
            coord=(None, 0),
        )
        gV = cute.local_tile(
            mV_slice,
            (self.tile_shape_pv[1], self.tile_shape_pv[2]),
            coord=(0, None),
        )

        gIndices = blocksparse_indices_q2k[
            None,
            work_desc.qo_tile_idx,
            work_desc.qo_head_idx,
            work_desc.batch_idx,
        ]
        if cutlass.const_expr(self.has_block_nums):
            num_n_tiles = blocksparse_num_blocks_q2k[
                work_desc.qo_tile_idx, work_desc.qo_head_idx, work_desc.batch_idx
            ]
        else:
            num_n_tiles = block_sparse_num
        if cutlass.const_expr(self.has_block_sizes):
            if cutlass.const_expr(self.block_sizes_mode == 1):
                gBSZ = blocksparse_varblk
            elif cutlass.const_expr(self.block_sizes_mode == 2):
                gBSZ = blocksparse_varblk[None, work_desc.batch_idx]
            else:
                gBSZ = blocksparse_varblk[
                    None, work_desc.qo_head_idx, work_desc.batch_idx
                ]

        # A single-CTA layout disables TMA multicast.
        cta_coord_layout = (0, cute.make_layout(1))
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            *cta_coord_layout,
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ, 0, 2),
        )
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            *cta_coord_layout,
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V,
            *cta_coord_layout,
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV, 0, 2),
        )

        cS = cute.make_identity_tensor(self.tile_shape_qk[:2])

        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        tSsQ = thr_mma_qk.partition_A(sQ)
        tSsK = thr_mma_qk.partition_B(sK)
        tSrQ = tiled_mma_qk.make_fragment_A(tSsQ[None, None, None, 0])
        tSrK = tiled_mma_qk.make_fragment_B(tSsK[None, None, None, 0])
        tSrS_i32 = cute.make_rmem_tensor(
            thr_mma_qk.partition_shape_C(
                (self.tile_shape_qk[0], self.tile_shape_qk[1])
            ),
            self.qk_acc_dtype,
        )
        tSrS = cute.make_rmem_tensor_like(tSrS_i32, cutlass.Float32)
        tScS = thr_mma_qk.partition_C(cS)

        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tOsV = thr_mma_pv.partition_B(sV)
        tOrV = tiled_mma_pv.make_fragment_B(tOsV[None, None, None, 0])
        tOrO = cute.make_rmem_tensor(
            thr_mma_pv.partition_shape_C(
                (self.tile_shape_pv[0], self.tile_shape_pv[1])
            ),
            self.output_acc_dtype,
        )
        tOrO_block = cute.make_rmem_tensor_like(tOrO, self.pv_acc_dtype)
        cO = cute.make_identity_tensor(self.tile_shape_pv[:2])
        tOcO = thr_mma_pv.partition_C(cO)

        atom_copy_ldmatrix_Q = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False,
                num_matrices=4,
            ),
            self.Q_dtype,
        )
        atom_copy_ldmatrix_K = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False,
                num_matrices=4,
            ),
            self.K_dtype,
        )
        atom_copy_ldmatrix_V = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False,
                num_matrices=4,
            ),
            self.V_dtype,
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(atom_copy_ldmatrix_Q, tiled_mma_qk)
        smem_tiled_copy_K = cute.make_tiled_copy_B(atom_copy_ldmatrix_K, tiled_mma_qk)
        smem_tiled_copy_V = cute.make_tiled_copy_B(
            atom_copy_ldmatrix_V,
            tiled_mma_pv,
        )

        thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        thr_copy_V = smem_tiled_copy_V.get_slice(tidx)
        tSsQ_copy = thr_copy_Q.partition_S(sQ)
        tSrQ_copy = thr_copy_Q.retile(tSrQ)
        tSsK_copy = thr_copy_K.partition_S(sK)
        tOsV_copy = thr_copy_V.partition_S(sV)

        max_m_layout = cute.make_layout(
            cute.size(
                layout_utils.reshape_acc_to_mn(tOrO).layout,
                mode=[0],
            )
        )
        max_m = cute.make_rmem_tensor_like(max_m_layout, cutlass.Float32)
        sum_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)
        q_softmax_scale_log2e_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)
        tScS_mn = layout_utils.reshape_acc_to_mn(tScS)
        for m in cutlass.range_constexpr(cute.size(q_softmax_scale_log2e_m)):
            row_idx = work_desc.qo_tile_idx * self.tile_size + tScS_mn[m, 0][0]
            q_scale_idx = (row_idx // 128) * 4 + (row_idx % 128) // 32
            q_softmax_scale_log2e_m[m] = (
                gQScale[q_scale_idx] * scale_softmax_log2e
                if row_idx < mQ.shape[0]
                else cutlass.Float32(0.0)
            )

        tOrO.store(cute.full_like(tOrO, 0.0, self.output_acc_dtype))
        max_m.store(cute.full_like(max_m, float("-inf"), cutlass.Float32))
        sum_m.store(cute.full_like(sum_m, 0.0, cutlass.Float32))

        if warp_idx == 0:
            Q_pipeline.producer_acquire(Q_producer_state)
            cute.copy(
                tma_atom_Q,
                tQgQ,
                tQsQ[None, 0],
                tma_bar_ptr=Q_pipeline.producer_get_barrier(Q_producer_state),
            )
            Q_pipeline.producer_commit(Q_producer_state)
            Q_producer_state.advance()

            preload_count = cutlass.Int32(0)
            if preload_count < num_n_tiles:
                logical_idx = num_n_tiles - 1 - preload_count
                physical_idx = gIndices[logical_idx]
                K_pipeline.producer_acquire(K_producer_state)
                cute.copy(
                    tma_atom_K,
                    tKgK[None, physical_idx],
                    tKsK[None, K_producer_state.index],
                    tma_bar_ptr=K_pipeline.producer_get_barrier(K_producer_state),
                )
                K_pipeline.producer_commit(K_producer_state)
                K_producer_state.advance()
                V_pipeline.producer_acquire(V_producer_state)
                cute.copy(
                    tma_atom_V,
                    tVgV[None, physical_idx],
                    tVsV[None, V_producer_state.index],
                    tma_bar_ptr=V_pipeline.producer_get_barrier(V_producer_state),
                )
                V_pipeline.producer_commit(V_producer_state)
                V_producer_state.advance()
        cute.arch.sync_threads()

        Q_wait_status = Q_pipeline.consumer_try_wait(Q_consumer_state)
        Q_pipeline.consumer_wait(Q_consumer_state, Q_wait_status)
        tQsQ_p = tSsQ_copy[None, None, None, 0]
        for k_block_idx in cutlass.range_constexpr(cute.size(tSrQ, mode=[2])):
            tQsQ_k = tQsQ_p[None, None, k_block_idx]
            tQsQ_k = cute.make_tensor(
                tQsQ_k.iterator.align(16),
                tQsQ_k.layout,
            )
            cute.copy(
                smem_tiled_copy_Q,
                tQsQ_k,
                tSrQ_copy[None, None, k_block_idx],
            )
        Q_pipeline.consumer_release(Q_consumer_state)
        Q_consumer_state.advance()

        n_tile_idx = gIndices[num_n_tiles - 1]
        for load_count in cutlass.range(0, num_n_tiles, 1, unroll=1):
            if cutlass.const_expr(self.has_block_sizes):
                varblk = gBSZ[n_tile_idx]
            else:
                varblk = cutlass.Int32(self.tile_size)
                if n_tile_idx == num_compute_tiles - 1:
                    varblk = seqlen - n_tile_idx * self.tile_size

            K_wait_status = K_pipeline.consumer_try_wait(K_consumer_state)
            K_pipeline.consumer_wait(K_consumer_state, K_wait_status)

            k_stage = K_consumer_state.index

            _gemm_qk_int8(
                tiled_mma_qk,
                tSrS_i32,
                tSrQ,
                tSrK,
                tSsK_copy[None, None, None, k_stage],
                smem_tiled_copy_K,
            )

            tSrS.store(tSrS_i32.load().to(cutlass.Float32))

            k_scale = _load_sage_k_scale_int8(
                gKScale,
                n_tile_idx,
            )

            K_pipeline.consumer_release(K_consumer_state)
            K_consumer_state.advance()

            preload_count = load_count + self.kv_stage
            next_n_tile_idx = cutlass.Int32(0)
            if preload_count < num_n_tiles:
                preload_logical = num_n_tiles - 1 - preload_count
                next_n_tile_idx = gIndices[preload_logical]
                if warp_idx == 0:
                    K_pipeline.producer_acquire(K_producer_state)
                    cute.copy(
                        tma_atom_K,
                        tKgK[None, next_n_tile_idx],
                        tKsK[None, K_producer_state.index],
                        tma_bar_ptr=K_pipeline.producer_get_barrier(K_producer_state),
                    )
                    K_pipeline.producer_commit(K_producer_state)
                    K_producer_state.advance()

            if varblk < self.tile_size:
                _mask_fp8(tSrS, tScS, varblk)
            row_scale = _online_softmax_fp8(
                tSrS,
                max_m,
                sum_m,
                q_softmax_scale_log2e_m,
                self.softmax_p_scale_log2,
                self.softmax_rescale_threshold,
                k_scale,
            )

            # Compute P @ V.
            _rescale_o_for_next_acc_fp8(tOrO, row_scale)
            tOrP = _make_acc_into_fp8_op(
                tSrS,
                tiled_mma_pv.tv_layout_A,
                self.V_dtype,
            )

            V_wait_status = V_pipeline.consumer_try_wait(V_consumer_state)
            V_pipeline.consumer_wait(V_consumer_state, V_wait_status)

            tOrO_block.fill(0.0)
            _gemm_rs_fp8(
                tiled_mma_pv,
                tOrO_block,
                tOrP,
                tOrV,
                tOsV_copy[None, None, None, 0],
                smem_tiled_copy_V,
            )
            _accumulate_o_block_fp8(tOrO, tOrO_block)
            V_pipeline.consumer_release(V_consumer_state)
            V_consumer_state.advance()

            if warp_idx == 0 and preload_count < num_n_tiles:
                V_pipeline.producer_acquire(V_producer_state)
                cute.copy(
                    tma_atom_V,
                    tVgV[None, next_n_tile_idx],
                    tVsV[None, V_producer_state.index],
                    tma_bar_ptr=V_pipeline.producer_get_barrier(V_producer_state),
                )
                V_pipeline.producer_commit(V_producer_state)
                V_producer_state.advance()

            n_tile_idx = next_n_tile_idx

        final_ratio = _finalize_softmax_sage(sum_m)
        _rescale_o_with_sage_v_scale_fp8(
            tOrO,
            tOcO,
            final_ratio,
            gVScale,
        )
        tOrO_cvt = cute.make_rmem_tensor_like(tOrO, self.O_dtype)
        tOrO_cvt.store(tOrO.load().to(self.O_dtype))

        # Use a separate shared-memory buffer for the BF16 output tile.
        sO = shared_storage.O_smem.get_tensor(
            O_smem_layout.outer, swizzle=O_smem_layout.inner
        )
        tiled_copy_o_r2s = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(self.O_layout.is_m_major_c(), 4),
                self.O_dtype,
            ),
            tiled_mma_pv,
        )
        tOrO_cv = tiled_copy_o_r2s.retile(tOrO_cvt)
        tOsO = tiled_copy_o_r2s.get_slice(tidx).partition_D(sO)
        cute.copy(tiled_copy_o_r2s, tOrO_cv, tOsO)

        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        # S2G with explicit TMA store wait.
        tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
            tma_atom_O,
            *cta_coord_layout,
            cute.group_modes(sO, 0, 2),
            cute.group_modes(gO, 0, 2),
        )
        if warp_idx == 0:
            cute.copy(tma_atom_O, tOsO, tOgO)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (head_dim, seqlen, nheads, batch)
        mK: cute.Tensor,  # (head_dim, seqlen, nheads, batch)
        mV: cute.Tensor,  # (value_dim, seqlen, nheads, batch)
        mO: cute.Tensor,  # (value_dim, seqlen, nheads, batch)
        mQScale: cute.Tensor,  # (ceil_div(seqlen_q, 128) * 4, nheads_q, batch)
        mKScale: cute.Tensor,  # (ceil_div(seqlen_k, 64), nheads_kv, batch)
        mVScale: cute.Tensor,  # (value_dim, nheads_kv, batch)
        blocksparse_indices_q2k: cute.Tensor,  # (k, q, nheads, batch)
        blocksparse_num_blocks_q2k: cute.Tensor,  # (q, nheads, batch)
        block_sparse_num: cutlass.Int32,
        blocksparse_varblk: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        # Restore compile-time head dimensions while keeping runtime tensor
        # modes dynamic for AOT and reusable JIT artifacts.
        mQ = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (mQ.shape[0], self.qk_dim, mQ.shape[2], mQ.shape[3]),
                stride=mQ.stride,
            ),
        )
        mK = cute.make_tensor(
            mK.iterator,
            cute.make_layout(
                (mK.shape[0], self.qk_dim, mK.shape[2], mK.shape[3]),
                stride=mK.stride,
            ),
        )
        mV = cute.make_tensor(
            mV.iterator,
            cute.make_layout(
                (self.value_dim, mV.shape[1], mV.shape[2], mV.shape[3]),
                stride=mV.stride,
            ),
        )
        mO = cute.make_tensor(
            mO.iterator,
            cute.make_layout(
                (mO.shape[0], self.value_dim, mO.shape[2], mO.shape[3]),
                stride=mO.stride,
            ),
        )
        mVScale = cute.make_tensor(
            mVScale.iterator,
            cute.make_layout(
                (self.value_dim, mVScale.shape[1], mVScale.shape[2]),
                stride=mVScale.stride,
            ),
        )
        self._check_dim([mQ, mK, mO], 1)
        assert mV.shape[0] == self.value_dim

        Q_layout = utils.LayoutEnum.from_tensor(mQ)
        K_layout = utils.LayoutEnum.from_tensor(mK)
        V_layout = utils.LayoutEnum.from_tensor(mV)
        O_layout = utils.LayoutEnum.from_tensor(mO)

        self.Q_dtype = mQ.element_type
        self.K_dtype = mK.element_type
        self.V_dtype = mV.element_type
        self.O_dtype = mO.element_type
        assert self.Q_dtype is cutlass.Int8
        assert self.K_dtype is cutlass.Int8
        assert self.V_dtype is cutlass.Float8E4M3FN
        assert self.O_dtype is cutlass.BFloat16
        self.Q_layout = Q_layout
        self.K_layout = K_layout
        self.V_layout = V_layout
        self.O_layout = O_layout

        atom_layout_mnk = (4, 1, 1)
        mma_inst_mnk = (16, 8, 32)
        permutation_mnk = (
            atom_layout_mnk[0] * mma_inst_mnk[0],
            atom_layout_mnk[1] * mma_inst_mnk[1] * 2,
            atom_layout_mnk[2] * mma_inst_mnk[2],
        )
        mma_op_qk = MmaInt8Op(
            self.Q_dtype,
            self.qk_acc_dtype,
            mma_inst_mnk,
        )
        mma_op_pv = cute.nvgpu.warp.MmaFP8Op(
            self.V_dtype,
            self.pv_acc_dtype,
            mma_inst_mnk,
        )
        tiled_mma_qk = cute.make_tiled_mma(
            mma_op_qk,
            cute.make_layout(atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )
        tiled_mma_pv = cute.make_tiled_mma(
            mma_op_pv,
            cute.make_layout(atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )

        self.Q_smem_layout = sm90_utils.make_smem_layout_a(
            Q_layout,
            self.tile_shape_qk,
            self.Q_dtype,
            self.q_stage,
        )
        self.K_smem_layout = sm90_utils.make_smem_layout_b(
            K_layout,
            self.tile_shape_qk,
            self.K_dtype,
            self.kv_stage,
        )
        self.V_smem_layout = sm90_utils.make_smem_layout_b(
            V_layout,
            self.tile_shape_pv,
            self.V_dtype,
            self.kv_stage,
        )
        O_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.O_dtype,
            O_layout,
            self.tile_shape_pv[:2],
            1,
        )
        self.O_smem_layout = cute.select(O_smem_layout_staged, mode=[0, 1])

        @cute.struct
        class SharedStorage:
            Q_barrier: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            K_barrier: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            V_barrier: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]

            Q_smem: cute.struct.Align[
                cute.struct.MemRange[self.Q_dtype, cute.cosize(self.Q_smem_layout)],
                128,
            ]
            K_smem: cute.struct.Align[
                cute.struct.MemRange[self.K_dtype, cute.cosize(self.K_smem_layout)],
                128,
            ]
            V_smem: cute.struct.Align[
                cute.struct.MemRange[self.V_dtype, cute.cosize(self.V_smem_layout)],
                128,
            ]
            O_smem: cute.struct.Align[
                cute.struct.MemRange[self.O_dtype, cute.cosize(self.O_smem_layout)],
                128,
            ]

        self.shared_storage_t = SharedStorage

        tma_copy_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mQ,
            self.Q_smem_layout,
            (self.tile_shape_qk[0], self.tile_shape_qk[2]),
            num_multicast=1,
        )
        tma_atom_K, tma_tensor_K = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mK,
            self.K_smem_layout,
            (self.tile_shape_qk[1], self.tile_shape_qk[2]),
            num_multicast=1,
        )
        tma_atom_V, tma_tensor_V = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mV,
            self.V_smem_layout,
            (self.tile_shape_pv[1], self.tile_shape_pv[2]),
            num_multicast=1,
        )

        tma_copy_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_O, tma_tensor_O = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mO,
            self.O_smem_layout,
            (self.tile_shape_pv[0], self.tile_shape_pv[1]),
            num_multicast=1,
        )

        log2_e = 1.44269504088896340736

        grid_config = self.get_grid_config(mQ.shape[0], mQ.shape[2], mQ.shape[3])
        block_config = (self.num_threads, 1, 1)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_O,
            mQScale,
            mKScale,
            mVScale,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            blocksparse_indices_q2k,
            blocksparse_num_blocks_q2k,
            block_sparse_num,
            blocksparse_varblk,
            tiled_mma_qk,
            tiled_mma_pv,
            self.Q_smem_layout,
            self.K_smem_layout,
            self.V_smem_layout,
            self.O_smem_layout,
            softmax_scale * log2_e,
        ).launch(
            grid=grid_config,
            block=block_config,
            cluster=(1, 1, 1),
            smem=self.shared_storage_t.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
        )


# =============================================================================
# Local CuTe helpers
# =============================================================================


def _convert_c_layout_to_a_layout_fp8(c: cute.Layout, a: cute.Layout) -> cute.Layout:
    """Reinterpret a C fragment with the logical shape of an A operand."""
    a_layout = cute.make_layout(a)
    c_atom_size = cute.size(c, mode=[0])
    expansion = cute.size(a) // c_atom_size
    return cute.make_layout(
        (a, c.shape[1], c.shape[2] // expansion),
        stride=(
            a_layout.stride,
            0,
            cute.size(a),
        ),
    )


@cute.jit
def _make_acc_into_fp8_op(
    acc: cute.Tensor,
    operand_layout_tv: cute.Layout,
    element: type[cutlass.Numeric],
) -> cute.Tensor:
    """Convert the score accumulator directly into Sage's FP8 A fragment.

    NOTE: the byte-lane permutation below (via cute.arch.prmt) is the
    inverse-side counterpart of the 16-token physical permutation baked into
    V by the quantization kernel (see sage_quant_sm120.py,
    _quantize_sage_kv_kernel's `physical_row` computation). The two must stay
    bit-for-bit consistent -- porting or editing one without the other
    silently produces wrong (not crashing) numerical results.
    """
    operand = cute.make_rmem_tensor_like(
        _convert_c_layout_to_a_layout_fp8(acc.layout, operand_layout_tv.shape[1]),
        element,
    )
    operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
    operand_as_acc.store(acc.load().to(element))

    # Composing the standard C-to-A exchange with Sage's 16-token V
    # permutation makes the final mapping lane-local. Pack the low and high
    # byte pairs from adjacent registers exactly like Sage's RS_32_to_8.
    values_u32 = cute.recast_tensor(operand, cutlass.Uint32)
    for n in cutlass.range_constexpr(cute.size(values_u32, mode=[1])):
        for k in cutlass.range_constexpr(cute.size(values_u32, mode=[2])):
            for ii in cutlass.range_constexpr(0, 8, 4):
                values_tmp_0 = values_u32[ii // 2, n, k]
                values_tmp_1 = values_u32[ii // 2 + 1, n, k]
                values_u32[ii // 2, n, k] = cute.arch.prmt(
                    values_tmp_0, values_tmp_1, 0x5410
                )
                values_u32[ii // 2 + 1, n, k] = cute.arch.prmt(
                    values_tmp_0, values_tmp_1, 0x7632
                )
    return operand


@cute.jit
def _gemm_qk_int8(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_tiled_copy_B: cute.TiledCopy,
) -> None:
    """Run INT8 QK MMA with Q in registers and K staged in shared memory."""
    acc.fill(0.0)
    tCrB_copy_view = smem_tiled_copy_B.retile(tCrB)
    tCsB_cur = tCsB[None, None, 0]
    tCsB_cur = cute.make_tensor(
        tCsB_cur.iterator.align(16),
        tCsB_cur.layout,
    )
    cute.copy(smem_tiled_copy_B, tCsB_cur, tCrB_copy_view[None, None, 0])
    for k_block_idx in cutlass.range_constexpr(cute.size(tCsB.shape[2])):
        if k_block_idx < cute.size(tCsB.shape[2]) - 1:
            tCsB_next = tCsB[None, None, k_block_idx + 1]
            tCsB_next = cute.make_tensor(
                tCsB_next.iterator.align(16),
                tCsB_next.layout,
            )
            cute.copy(
                smem_tiled_copy_B,
                tCsB_next,
                tCrB_copy_view[None, None, k_block_idx + 1],
            )
        cute.gemm(
            tiled_mma,
            acc,
            tCrA[None, None, k_block_idx],
            tCrB[None, None, k_block_idx],
            acc,
        )


@cute.jit
def _gemm_rs_fp8(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_tiled_copy_B: cute.TiledCopy,
) -> None:
    """Run FP8 PV MMA with V staged in shared memory."""
    tCrB_copy_view = smem_tiled_copy_B.retile(tCrB)
    tCsB_cur = tCsB[None, None, 0]
    cute.copy(smem_tiled_copy_B, tCsB_cur, tCrB_copy_view[None, None, 0])
    for k_block_idx in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if k_block_idx < cute.size(tCrA.shape[2]) - 1:
            tCsB_next = tCsB[None, None, k_block_idx + 1]
            cute.copy(
                smem_tiled_copy_B,
                tCsB_next,
                tCrB_copy_view[None, None, k_block_idx + 1],
            )
        cute.gemm(
            tiled_mma,
            acc,
            tCrA[None, None, k_block_idx],
            tCrB[None, None, k_block_idx],
            acc,
        )


@cute.jit
def _load_sage_k_scale_int8(
    gKScale: cute.Tensor,
    k_tile_idx: cutlass.Int32,
) -> cutlass.Float32:
    """Load the single Sage K64 descale for one physical KV tile."""
    scale_idx = k_tile_idx if k_tile_idx < gKScale.shape[0] else gKScale.shape[0] - 1
    return gKScale[scale_idx]


@cute.jit
def _mask_fp8(
    tSrS: cute.ThrMma,
    tScS: cute.Tensor,
    varblk: cutlass.Int32,
):
    tSrS_mn = layout_utils.reshape_acc_to_mn(tSrS)
    tScS_mn = layout_utils.reshape_acc_to_mn(tScS)

    for n in cutlass.range(cute.size(tSrS_mn, mode=[1]), unroll_full=True):
        should_mask = tScS_mn[0, n][1] >= varblk
        for m in cutlass.range(cute.size(tSrS_mn, mode=[0]), unroll_full=True):
            tSrS_mn[m, n] = -cutlass.Float32.inf if should_mask else tSrS_mn[m, n]


@cute.jit
def _online_softmax_fp8(
    tSrS: cute.ThrMma,
    row_max: cute.Tensor,
    row_sum: cute.Tensor,
    softmax_scale_log2e_m: cute.Tensor,
    exp_scale_log2: cutlass.Float32,
    rescale_threshold: cutlass.Constexpr[float],
    k_scale: cutlass.Float32,
) -> cute.Tensor:
    tSrS_mn = layout_utils.reshape_acc_to_mn(tSrS)
    row_scale = cute.make_rmem_tensor_like(row_max, cutlass.Float32)

    for m in cutlass.range(cute.size(row_max), unroll_full=True):
        score_scale_log2e = k_scale * softmax_scale_log2e_m[m]
        row_max_local = row_max[m]
        # Keep row_max in the final exp2 domain. Sage has one positive K scale
        # per K64 tile, so the unscaled scores can be reduced before scaling.
        group_max = tSrS_mn[m, 0]
        for n in cutlass.range_constexpr(
            1,
            cute.size(tSrS_mn, mode=[1]),
        ):
            group_max = cute.arch.fmax(group_max, tSrS_mn[m, n])
        row_max_local = cute.arch.fmax(
            row_max_local,
            group_max * score_scale_log2e,
        )
        row_max_cur = cute.arch.warp_reduction_max(
            row_max_local,
            threads_in_group=4,
        )

        row_max_prev = row_max[m]
        row_max_safe = (
            cutlass.Float32(0.0) if row_max_cur == -cutlass.Float32.inf else row_max_cur
        )
        row_scale_log2 = row_max_prev - row_max_safe
        if row_scale_log2 >= -rescale_threshold:
            row_max_cur = row_max_prev
            row_max_safe = row_max_prev
            row_scale[m] = 1.0
        else:
            row_scale[m] = cute.math.exp2(
                row_scale_log2,
                fastmath=True,
            )
        row_max[m] = row_max_cur
        # Shift the exponent by log2(P scale) before FP8 conversion. Keeping
        # row_sum in the same scale removes a vector multiply from every tile.
        row_max_shifted = row_max_safe - exp_scale_log2
        for n in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[1])):
            tSrS_mn[m, n] = cute.math.exp2(
                tSrS_mn[m, n] * score_scale_log2e - row_max_shifted,
                fastmath=True,
            )
        acc_S_row_exp = tSrS_mn[m, None].load()
        row_sum[m] = kernel_utils.fadd_reduce(
            acc_S_row_exp,
            init_val=row_sum[m] * row_scale[m],
            arch=80,
        )
        tSrS_mn[m, None].store(acc_S_row_exp)

    return row_scale


@cute.jit
def _finalize_softmax_sage(row_sum: cute.Tensor) -> cute.Tensor:
    """Reduce the scaled denominator and return its reciprocal."""
    row_sum.store(kernel_utils.warp_reduce(row_sum.load(), operator.add, width=4))
    final_ratio = cute.make_rmem_tensor_like(row_sum, cutlass.Float32)

    for m in cutlass.range(cute.size(row_sum), unroll_full=True):
        final_sum = row_sum[m]
        is_zero_or_nan = final_sum == 0.0 or final_sum != final_sum
        final_ratio[m] = cute.arch.rcp_approx(final_sum if not is_zero_or_nan else 1.0)

    return final_ratio


@cute.jit
def _rescale_o_for_next_acc_fp8(
    tOrO: cute.ThrMma,
    prev_ratio_m: cute.Tensor,
):
    tOrO_mn = layout_utils.reshape_acc_to_mn(tOrO)
    for m in cutlass.range(cute.size(prev_ratio_m), unroll_full=True):
        should_rescale = cute.arch.vote_ballot_sync(prev_ratio_m[m] < 1.0) != 0
        if should_rescale:
            tOrO_mn[m, None].store(tOrO_mn[m, None].load() * prev_ratio_m[m])


@cute.jit
def _accumulate_o_block_fp8(
    tOrO: cute.Tensor,
    tOrO_block: cute.Tensor,
) -> None:
    """Promote each FP16 PV result and merge it into the FP32 O accumulator."""
    for i in cutlass.range(cute.size(tOrO), unroll_full=True):
        tOrO[i] += tOrO_block[i].to(cutlass.Float32)


@cute.jit
def _rescale_o_with_sage_v_scale_fp8(
    tOrO: cute.Tensor,
    tOcO: cute.Tensor,
    final_ratio_m: cute.Tensor,
    gVScale: cute.Tensor,
) -> None:
    """Normalize O and apply the per-channel Sage V descale."""
    tOrO_mn = layout_utils.reshape_acc_to_mn(tOrO)
    tOcO_mn = layout_utils.reshape_acc_to_mn(tOcO)
    for n in cutlass.range(cute.size(tOrO_mn, mode=[1]), unroll_full=True):
        v_scale = gVScale[tOcO_mn[0, n][1]]
        for m in cutlass.range(cute.size(final_ratio_m), unroll_full=True):
            tOrO_mn[m, n] *= final_ratio_m[m] * v_scale
