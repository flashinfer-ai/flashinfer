# Supported features:
# - BF16 & FP16 dtype
# - noncausal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128, (192, 128).
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

import math
from typing import Tuple, Callable, Optional, Literal
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL

from quack import copy_utils, layout_utils

from .cute_dsl_utils import assume_tensor_aligned
from . import utils
from . import pipeline as pipeline_custom
from .mask import apply_block_size_mask
from .softmax import SoftmaxSm100
from .seqlen_info import SeqlenInfoQK
from .block_info import BlockInfo
from .pack_gqa import PackGQA, pack_gqa_layout
from . import mma_sm100_desc as sm100_desc
from . import blackwell_helpers as sm100_utils
from .named_barrier import NamedBarrierFwdSm100
from quack.cute_dsl_utils import ParamsBase
import cutlass.pipeline as cutlass_pipeline
from .tile_scheduler import (
    TileSchedulerArguments,
    TileSchedulerProtocol,
    SchedulingMode,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
)


class FlashAttentionForwardSm100:
    def __init__(
        self,
        # dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        pack_gqa: bool = False,
        m_block_size: int = 128,
        n_block_size: int = 128,
        is_persistent: bool = True,
        use_2cta_instrs: bool = False,
        use_clc_scheduler: bool = False,
        allow_empty_block_nums: bool = False,
        has_block_sizes: bool = True,
    ):
        self.use_tma_KV = True
        # self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(
            math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of
        )
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.q_stage = 1
        self.use_2cta_instrs = use_2cta_instrs
        # If split_P_arrive, the softmax warps write some columns of P first, signal to the MMA warp
        # to being the P @ V MMA, then write the rest of P and signal again. This allows some overlap
        # between compute the last couple columns of P and the P @ V MMA.
        self.split_P_arrive = n_block_size // 4 * 3
        self.split_P_arrive = int(self.split_P_arrive / 32) * 32  # multiple of 32
        assert self.split_P_arrive % 32 == 0
        assert self.split_P_arrive < self.n_block_size
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f, (
            "Only SM 10.x and 11.x are supported"
        )

        self.cta_group_size = 2 if self.use_2cta_instrs else 1
        # cta_tiler M includes only 1 CTA, the scheduler will take into account the cluster shape
        self.cta_tiler = (1 * m_block_size, n_block_size, self.head_dim_padded)
        # With 2CTA, the MMA tiler M covers both CTAs, so it's cta_group_size * m_block_size.
        # Each CTA owns m_block_size rows; the 2CTA MMA instruction spans both.
        self.mma_tiler_qk = (
            self.cta_group_size * m_block_size,
            n_block_size,
            self.head_dim_padded,
        )
        self.mma_tiler_pv = (
            self.cta_group_size * m_block_size,
            self.head_dim_v_padded,
            n_block_size,
        )
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)
        self.is_persistent = is_persistent
        # CLC persistent scheduling
        self.use_clc_scheduler = use_clc_scheduler and self.use_tma_KV
        self.sched_stages = 1
        if self.use_clc_scheduler:
            assert self.cluster_shape_mn[1] == 1, (
                f"CLC requires cluster N == 1: {self.cluster_shape_mn}"
            )
            assert self.cluster_shape_mn[0] in (1, 2), (
                f"bad CLC cluster M: {self.cluster_shape_mn}"
            )
            assert self.cluster_shape_mn[0] == self.cta_group_size, (
                f"CLC cluster M != cta_group_size: {self.cluster_shape_mn}, {self.cta_group_size}"
            )
        self.scheduling_mode = (
            SchedulingMode.CLC if self.use_clc_scheduler else SchedulingMode.STATIC
        )
        self.allow_empty_block_nums = allow_empty_block_nums
        self.has_block_sizes = has_block_sizes
        self.is_causal = False
        self.is_local = False
        self.is_varlen_q = False
        self.use_correction_warps_for_epi = False
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = False
        self.pack_gqa = pack_gqa
        if pack_gqa:
            assert m_block_size % self.qhead_per_kvhead == 0, (
                "For PackGQA, m_block_size must be divisible by qhead_per_kvhead"
            )
        is_sm103 = self.arch >= Arch.sm_103 and self.arch <= Arch.sm_103f
        self.enable_ex2_emu = (
            self.head_dim_padded <= 128
            or (
                self.head_dim_padded == 192
                and self.use_2cta_instrs
                and not self.is_causal
                and not self.is_local
            )
        ) and not is_sm103
        self.s0_s1_barrier = False
        self.overlap_sO_sQ = (
            self.head_dim_padded == 192 and self.head_dim_v_padded >= 64
        )
        if self.overlap_sO_sQ:
            self.is_persistent = False

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        self.clc_scheduler_warp_id = (
            self.empty_warp_ids[0] if self.use_clc_scheduler else None
        )
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )

        if self.use_correction_warps_for_epi:
            self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids  # type: ignore[assignment]
            self.epilogue_warp_ids = self.correction_warp_ids  # type: ignore[assignment]

        self.s_stage = 2  # Always 2: for q_stage=1 it's n-direction
        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.s_stage)
        ]  # e.g., 256, 384
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_s_to_p_offset = self.n_block_size // 2
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # 0, 128

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200
            self.num_regs_correction = 64
            self.num_regs_other = 48
        else:
            if not self.enable_ex2_emu:
                self.num_regs_softmax = 184
            else:
                self.num_regs_softmax = 184
            if not self.enable_ex2_emu:
                self.num_regs_correction = 88
            else:
                self.num_regs_correction = 88
            self.num_regs_other = 56

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        smem_size_q = (
            self.q_stage
            * self.m_block_size
            * self.head_dim_padded
            * self.q_dtype.width
            // 8
        )
        smem_size_o = (
            self.s_stage
            * self.m_block_size
            * self.head_dim_v_padded
            * self.o_dtype.width
            // 8
        )
        smem_size_q_o = (
            smem_size_q + smem_size_o
            if not self.overlap_sO_sQ
            else max(smem_size_q, smem_size_o)
        )
        smem_size_k_per_stage = (
            self.n_block_size * self.head_dim_padded * self.k_dtype.width // 8
        )
        smem_size_v_per_stage = (
            self.n_block_size * self.head_dim_v_padded * self.v_dtype.width // 8
        )
        smem_size_kv_per_stage = (
            max(smem_size_k_per_stage, smem_size_v_per_stage) // self.cta_group_size
        )
        kv_stage = (224 * 1024 - smem_size_q_o) // smem_size_kv_per_stage
        if (
            self.head_dim_padded == 192
            and self.head_dim_v_padded == 128
            and kv_stage == 2
        ):
            # For hdim 192,128, we can fit 3 stages if we use uneven_kv_smem
            kv_stage = 3
        self.kv_stage = kv_stage
        # self.s_stage is defined in __init__ (always 2)
        assert self.s_stage >= self.q_stage
        # For hdim 192,128 1CTA, we don't have enough smem to store all 3 stages of KV:
        # 128 x 192 x 2 bytes x 3 stages = 144KB, and we need 96KB for Q.
        # Instead we store smem as [smem_large, smem_small, smem_large], where smem_large is
        # 128 x 192 and smem_small is 128 x 128. We set the stride between the stages to be
        # 128 * 160, so that indexing the 0th and 2nd stages will get the right address,
        # but for the 1st stage we need to add or subtract (depending on phase) 128 x 64.
        self.uneven_kv_smem = (
            self.head_dim_padded == 192
            and self.head_dim_v_padded == 128
            and self.kv_stage == 3
        )
        self.uneven_kv_smem_offset = (
            self.n_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2
            if self.uneven_kv_smem
            else 0
        )
        assert self.uneven_kv_smem_offset % 1024 == 0

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d)
        mK: cute.Tensor,  # (b_k, s_k, h_k, d)
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv)
        mO: cute.Tensor,  # (b, s_q, h, dv)
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mBlockIndex: cute.Tensor,  # (batch, heads, num_q_blocks, max_kv_blocks), int32
        mBlockSizes: cute.Tensor,  # (num_kv_blocks,), int32
        block_sparse_num: Int32,  # runtime scalar, even, >= 2
        mBlockNums: Optional[
            cute.Tensor
        ],  # (batch, heads, num_q_blocks), int32 or None
        stream: cuda.CUstream,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [1, 3, 2, 0]
        mQ = cute.make_tensor(
            mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose)
        )
        # (s_k, d, h_k, b_k)
        KV_layout_transpose = [1, 3, 2, 0]
        mK, mV = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=KV_layout_transpose)
            )
            for t in (mK, mV)
        ]
        O_layout_transpose = [1, 3, 2, 0]
        LSE_layout_transpose = [2, 1, 0]
        num_splits = Int32(1)
        mO = cute.make_tensor(
            mO.iterator, cute.select(mO.layout, mode=O_layout_transpose)
        )
        mLSE = (
            cute.make_tensor(
                mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)
            )
            if const_expr(mLSE is not None)
            else None
        )
        # (s, d, h, b) -> (d, s, h, b)
        V_layout_transpose = [1, 0, 2, 3]
        mV = cute.make_tensor(
            mV.iterator, cute.select(mV.layout, mode=V_layout_transpose)
        )

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()
        self.use_tma_O = self.arch >= Arch.sm_90
        # This can be tuned
        # This is currently very ad-hoc, we should tune it systematically
        self.ex2_emu_freq = 0
        self.ex2_emu_start_frg = 0
        if const_expr(self.enable_ex2_emu):
            self.ex2_emu_freq = 10
            if const_expr(self.head_dim_padded == 128 and self.use_2cta_instrs):
                self.ex2_emu_freq = 12
            if const_expr(
                self.pack_gqa
                and self.head_dim_padded > 64
                and not self.is_causal
                and not self.is_local
            ):
                self.ex2_emu_freq = 10
            if const_expr(self.head_dim_padded > 64 and self.is_causal):
                self.ex2_emu_freq = 10

        cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        q_major_mode = tcgen05.OperandMajorMode.K
        k_major_mode = tcgen05.OperandMajorMode.K
        v_major_mode = tcgen05.OperandMajorMode.MN
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )

        # epi_tile is per-CTA (not full 2CTA) since each CTA writes its own O portion
        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.s_stage
        )
        if const_expr(not self.same_hdim_kv_padded):
            # sK and sV are using the same physical smem so we need to adjust the stride so that they line up
            stride_sK = const_expr(
                max(sK_layout.outer.stride[-1], 0)
            )  # take max to turn tuple to Int32
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(
                max(stride_sK, stride_sV)
                if not self.uneven_kv_smem
                else (stride_sK + stride_sV) // 2
            )
            sK_layout = cute.make_composed_layout(
                sK_layout.inner,
                0,
                cute.make_layout(
                    (*sK_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], stage_stride),
                ),
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner,
                0,
                cute.make_layout(
                    (*sV_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sV_layout.outer.stride[:-1], stage_stride),
                ),
            )

        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(
                    mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1
                )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(
                mX.element_type, cute.select(layout, mode=[0, 1, 2])
            )
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        for name in ("Q", "K", "V"):
            self.tma_copy_bytes[name] *= self.cta_group_size

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )

        # TMA load for K
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )
        # TMA load for V
        tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_pv,
            tiled_mma_pv,
            cta_layout_vmnk.shape,
        )

        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op, mO, cute.select(sO_layout, mode=[0, 1]), self.epi_tile
            )
            gmem_tiled_copy_O = None
        else:
            tma_atom_O = None
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            # So that we don't have to check if we overshoot kBlockM when we store O
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(
                atom_universal_copy, tO_layout, vO_layout
            )

        if const_expr(self.use_clc_scheduler):
            TileScheduler = StaticPersistentTileScheduler
        elif const_expr(not self.is_persistent):
            TileScheduler = SingleTileScheduler  # type: ignore[assignment]
        else:
            TileScheduler = StaticPersistentTileScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            num_splits,
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[
                0
            ],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            cluster_shape_mn=self.cluster_shape_mn,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 0
        sQ_size = (
            cute.cosize(sQ_layout)
            if const_expr(not self.overlap_sO_sQ)
            else cutlass.max(
                cute.cosize(sQ_layout),
                cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width,
            )
        )

        clc_response_size = self.sched_stages * 4 if self.use_clc_scheduler else 0
        clc_mbar_size = self.sched_stages * 2 if self.use_clc_scheduler else 0

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_s0_s1_sequence: cute.struct.MemRange[Int64, 2 * 2]
            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar_ptr: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.s_stage * self.m_block_size * 2]
            # CLC buffers (mbarriers + response storage)
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            clc_response: cute.struct.MemRange[Int32, clc_response_size]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size], self.buffer_align_bytes
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                # cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale
        )

        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            num_splits,
            mBlockIndex,
            mBlockSizes,
            block_sparse_num,
            mBlockNums,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk
            if cute.size(self.cluster_shape_mnk) > 1
            else None,
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b)
        mK: cute.Tensor,  # (s_k, d, h_k, b_k)
        mV: cute.Tensor,  # (d, s_k, h_k, b_k)
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        mBlockIndex: cute.Tensor,
        mBlockSizes: cute.Tensor,
        block_sparse_num: Int32,
        mBlockNums: Optional[cute.Tensor],
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        # Setup cta/thread coordinates
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(cute.size(tiled_mma_qk.thr_id.shape) == 1):
            mma_tile_coord_v = 0
        else:
            mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len(
                (
                    self.mma_warp_id,
                    *self.softmax0_warp_ids,
                    *self.softmax1_warp_ids,
                    *self.correction_warp_ids,
                )
            ),
        )
        # Tensor memory dealloc barrier init
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        ThreadCooperativeGroup = partial(
            pipeline.CooperativeGroup, pipeline.Agent.Thread
        )
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        ThreadCooperativeGroup(len(self.load_warp_ids))
        tma_warp = ThreadCooperativeGroup(1)
        ThreadCooperativeGroup(len(self.softmax0_warp_ids))
        softmax_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
        )
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids)
        )
        epilogue_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        )
        # For UMMA-bridging pipelines: the non-MMA side spans both CTAs in the cluster,
        # so the thread count must include warps from both CTAs.
        softmax_warps_cluster = ThreadCooperativeGroup(
            len(self.softmax0_warp_ids) * self.cta_group_size
        )
        correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids) * self.cta_group_size
        )
        softmax_correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE
            * len(self.softmax0_warp_ids + self.correction_warp_ids)
            * self.cta_group_size
        )
        pipeline_q = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # This pipeline is not the typical producer-consumer pipeline. The "producer" mma warp
        # uses it to signal that S is ready, and the softmax threads wait for S to be ready.
        # When softmax threads write P to tmem and the correction threads have rescaled O, they
        # signal as "consumer". The mma warp then waits for that signal to do the P @ V gemm.
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.s_stage,
            producer_group=mma_warp,
            consumer_group=softmax_correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_warps_cluster,
            consumer_group=mma_warp,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # MMA warp uses this to signal to the correction warps that O is ready.
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.s_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_s0_s1_sequence = None
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        # Should put the NamedBarrier inside the pipeline class so we'll just have pipeline_sm_stats
        sm_stats_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0),
            num_threads=cute.arch.WARP_SIZE * 2,
        )
        pipeline_o_epi = None
        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_O_epi.data_ptr(),
                num_stages=self.s_stage,
                producer_group=correction_threads,
                consumer_group=epilogue_threads,
                defer_sync=True,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer
        )
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype),
                sO_layout.outer,
            )

        sScale = storage.sScale.get_tensor(
            cute.make_layout(self.s_stage * self.m_block_size * 2)
        )

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # This is a fake tensor, by right we need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.s_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        # Need to multiply by width ratio bc tP is in v_dtype but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.v_dtype.width
        # Need to adjust the stage stride manually since the two stages aren't contiguous in tmem
        tP_stage_stride = (
            self.tmem_p_offset[1] - self.tmem_p_offset[0]
        ) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(
                tOrP.layout,
                cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)),
            ),
        )

        block_info = BlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.cta_tiler[0],
            self.cta_tiler[1],
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0]
            if const_expr(not self.pack_gqa)
            else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0],
        )
        # Create tile scheduler (and CLC pipeline if enabled)
        if const_expr(self.use_clc_scheduler):
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()

            clc_pipeline_producer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread
            )
            num_clc_consumer_warps_per_cta = self.threads_per_cta // cute.arch.WARP_SIZE
            num_clc_consumer_warps = (
                num_clc_consumer_warps_per_cta * self.cta_group_size
            )
            clc_pipeline_consumer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread,
                cute.arch.WARP_SIZE * num_clc_consumer_warps,
            )
            clc_pipeline = cutlass_pipeline.PipelineClcFetchAsync.create(
                barrier_storage=clc_mbar_ptr,
                num_stages=self.sched_stages,
                producer_group=clc_pipeline_producer_group,
                consumer_group=clc_pipeline_consumer_group,
                tx_count=16,
                cta_layout_vmnk=cta_layout_vmnk,
            )

            tile_scheduler = self.tile_scheduler_cls.create(
                tile_sched_params, clc_response_ptr=clc_response_ptr
            )
            clc_consumer_state = cutlass_pipeline.make_pipeline_state(
                cutlass_pipeline.PipelineUserType.Consumer, self.sched_stages
            )
            tile_scheduler.set_clc_pipeline(clc_pipeline, clc_consumer_state)
        else:
            clc_pipeline = None
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)

        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY / CLC SCHEDULER WARP
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(self.use_clc_scheduler):
            if warp_idx == self.clc_scheduler_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                if is_leader_cta:
                    self.clc_scheduler_warp(clc_pipeline, tile_scheduler)
                else:
                    self.empty_warp(clc_pipeline, tile_scheduler)
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if (
                    warp_idx == self.empty_warp_ids[i]
                    and warp_idx != self.clc_scheduler_warp_id
                ):
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                    self.empty_warp(clc_pipeline, tile_scheduler)
        else:
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i]:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_q,
                pipeline_kv,
                block_info,
                num_splits,
                SeqlenInfoCls,
                tile_scheduler,
                mBlockIndex,
                block_sparse_num,
                mBlockNums,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Alloc tensor memory buffer
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tStS,
                tOtO,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                is_leader_cta,
                block_info,
                num_splits,
                SeqlenInfoCls,
                tile_scheduler,
                block_sparse_num,
                mBlockNums,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.use_correction_warps_for_epi):
            if (
                warp_idx >= self.epilogue_warp_ids[0]
                and warp_idx <= self.epilogue_warp_ids[-1]
            ):
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.epilogue_s2g(
                    mO,
                    sO,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    pipeline_o_epi,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    tile_scheduler,
                    mma_tile_coord_v,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx <= self.softmax1_warp_ids[-1]:
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
                mBlockIndex=mBlockIndex,
                mBlockSizes=mBlockSizes,
                block_sparse_num=block_sparse_num,
                mBlockNums=mBlockNums,
            )

            if const_expr(not self.s0_s1_barrier):
                stage = Int32(0 if warp_idx < self.softmax1_warp_ids[0] else 1)
                softmax_loop(stage=stage, tStS=tStS)
            else:
                # If there's s0_s1_barrier, it's faster to have 2 WGs having different code
                if warp_idx < self.softmax1_warp_ids[0]:
                    softmax_loop(stage=0, tStS=tStS)
                if (
                    warp_idx < self.correction_warp_ids[0]
                    and warp_idx >= self.softmax1_warp_ids[0]
                ):
                    softmax_loop(stage=1, tStS=tStS)

            tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtO,
                sScale,
                mO,
                mLSE,
                sO,
                pipeline_s_p_o,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                pipeline_o_epi,
                gmem_tiled_copy_O,
                tma_atom_O,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                tile_scheduler,
                block_sparse_num,
                mBlockNums,
            )
            tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def clc_scheduler_warp(
        self,
        clc_pipeline: cutlass_pipeline.PipelineClcFetchAsync,
        tile_scheduler: TileSchedulerProtocol,
    ):
        """Runs on leader CTA's scheduler warp — produces CLC work queries."""
        clc_producer_state = cutlass_pipeline.make_pipeline_state(
            cutlass_pipeline.PipelineUserType.Producer, self.sched_stages
        )
        clc_consumer_state = cutlass_pipeline.make_pipeline_state(
            cutlass_pipeline.PipelineUserType.Consumer, self.sched_stages
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            clc_pipeline.producer_acquire(clc_producer_state)
            mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
            tile_scheduler.advance_to_next_work(mbarrier_addr=mbarrier_addr)
            clc_producer_state.advance()

            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_scheduler.get_current_work()
            clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()
        clc_pipeline.producer_tail(clc_producer_state)

    @cute.jit
    def empty_warp(
        self,
        clc_pipeline: cutlass_pipeline.PipelineClcFetchAsync,
        tile_scheduler: TileSchedulerProtocol,
    ):
        """Runs on empty warps (and non-leader CTA scheduler warp) — consumes CLC responses."""
        clc_consumer_state = cutlass_pipeline.make_pipeline_state(
            cutlass_pipeline.PipelineUserType.Consumer, self.sched_stages
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_scheduler.get_current_work()
            clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        mBlockIndex: cute.Tensor,
        block_sparse_num: Int32,
        mBlockNums: Optional[cute.Tensor],
    ):
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        cute.arch.thread_idx()[0] % num_load_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            SeqlenInfoCls()
            mQ_cur = mQ[None, None, None, batch_idx][None, None, head_idx]
            tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)
            gQ = cute.local_tile(mQ_cur, tiler_gQ, (m_block, 0))  # (128, 128)
            gQ = layout_utils.select(
                cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
            )  # (128, 128, 1)

            head_idx_kv = (
                head_idx // self.qhead_per_kvhead
                if const_expr(not self.pack_gqa)
                else head_idx
            )
            mK_cur, mV_cur = [t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)]
            gK = cute.local_tile(
                mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0)
            )
            gV = cute.local_tile(
                mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None)
            )
            tSgQ = thr_mma_qk.partition_A(gQ)
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )

            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV, 0, 3),
            )

            partial(
                self.load_Q, load_Q_fn, pipeline_q=pipeline_q, phase=q_producer_phase
            )
            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                sK,
                pipeline_kv=pipeline_kv,
                K_or_V="K",
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                sV,
                pipeline_kv=pipeline_kv,
                K_or_V="V",
            )

            # n_block(i): maps logical index i to actual KV block index via q2k_block_index
            # When mBlockNums is provided, raw count may be odd; round up to even for kernel loops.
            # max_i clamps phantom block indices to the last valid entry.
            if const_expr(mBlockNums is not None):
                raw_block_count = mBlockNums[batch_idx, head_idx, m_block]
                process_tile = (
                    raw_block_count > Int32(0)
                    if const_expr(self.allow_empty_block_nums)
                    else True
                )
                block_iter_count = (raw_block_count + 1) & ~1
                n_block = partial(
                    block_info.get_n_block_idx,
                    mBlockIndex,
                    batch_idx,
                    head_idx,
                    m_block,
                    max_i=cutlass.max(raw_block_count - 1, Int32(0)),
                )
            else:
                process_tile = True
                block_iter_count = block_sparse_num
                n_block = partial(
                    block_info.get_n_block_idx,
                    mBlockIndex,
                    batch_idx,
                    head_idx,
                    m_block,
                )

            if process_tile:
                load_K(
                    block=n_block(block_iter_count - 1),
                    producer_state=kv_producer_state,
                    page_idx=None,
                )  # K0
                if (
                    const_expr(len(self.load_warp_ids) == 1)
                    or warp_idx == self.load_warp_ids[0]
                ):
                    pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                    tma_bar_ptr = pipeline_q.sync_object_full.get_barrier(0)
                    load_Q_fn(src_idx=0, dst_idx=0, tma_bar_ptr=tma_bar_ptr)
                kv_producer_state.advance()

                # q_stage=1 intra-warp overlap
                # Load order: K[N-1], Q, K[N-2], {V[N-1-i], K[N-3-i]}x(N-2), V[1], V[0]
                block_loop_count = block_iter_count - 2
                q_producer_phase ^= 1

                # Prologue: K[N-2]
                load_K(
                    block=n_block(block_iter_count - 2),
                    producer_state=kv_producer_state,
                    page_idx=None,
                )
                kv_producer_state.advance()

                # Flat main loop: N-2 iterations, each loads V then K
                for i in cutlass.range(block_loop_count, unroll=1):
                    # V[N-1-i]: V for the block whose S was computed earlier
                    load_V(
                        block=n_block(block_iter_count - 1 - i),
                        producer_state=kv_producer_state,
                        page_idx=None,
                    )
                    kv_producer_state.advance()
                    # K[N-3-i]: K for the next QK GEMM
                    load_K(
                        block=n_block(block_iter_count - 3 - i),
                        producer_state=kv_producer_state,
                        page_idx=None,
                    )
                    kv_producer_state.advance()

                # Epilogue: last 2 V loads
                load_V(
                    block=n_block(1), producer_state=kv_producer_state, page_idx=None
                )
                kv_producer_state.advance()
                load_V(
                    block=n_block(0), producer_state=kv_producer_state, page_idx=None
                )
                kv_producer_state.advance()

            tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.consumer_advance()
            # End of persistent scheduler loop

        pipeline_kv.producer_tail(kv_producer_state)
        # This is equivalent to pipeline_q.producer_tail
        if (
            const_expr(len(self.load_warp_ids) == 1)
            or warp_idx == self.load_warp_ids[0]
        ):
            pipeline_q.producer_acquire_w_index_phase(
                self.q_stage - 1, q_producer_phase
            )

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.ThrMma,
        tiled_mma_pv: cute.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        is_leader_cta: Boolean,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        block_sparse_num: Int32,
        mBlockNums: Optional[cute.Tensor],
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        # q_stage=1: both stages use the same Q (intra-warp overlap across n_blocks)
        (tSrQ[None, None, None, 0], tSrQ[None, None, None, 0])

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        _qk_mma_idesc, _pv_mma_idesc = (
            sm100_desc.mma_op_to_idesc(qk_mma_op),
            sm100_desc.mma_op_to_idesc(pv_mma_op),
        )
        q_smem_base = sm100_desc.smem_desc_base_from_tensor(sQ, sm100_desc.Major.K)
        k_smem_base = sm100_desc.smem_desc_base_from_tensor(sK, sm100_desc.Major.K)
        sm100_desc.smem_desc_base_from_tensor(sV, sm100_desc.Major.MN)
        q_smem_start = [
            sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator)
            for stage in range(self.q_stage)
        ]

        sm100_utils.declare_ptx_smem_desc(
            q_smem_start[self.q_stage - 1],
            q_smem_base,
            tSrQ[None, None, None, 0].layout,
            var_name_prefix="fa_fwd_q_smem_desc",
        )
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
        sm100_utils.declare_ptx_idesc(pv_mma_op, var_name="fa_fwd_pv_mma_idesc")

        sQ_stage_stride = 0  # q_stage=1
        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_precomputed_varname,
                self.tmem_s_offset[stage],
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK[None, None, None, 0].layout,
                smem_var_name_prefix="fa_fwd_q_smem_desc",
                idesc_var_name="fa_fwd_qk_mma_idesc",
                # For q_stage=1, both stages use same Q smem (offset=0).
                smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride,
                zero_init=True,
                cta_group=self.cta_group_size,
            )
            for stage in range(self.s_stage)
        ]
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage],
                tOrP[None, None, None, stage],
                sA=None,
                split_arrive=self.split_P_arrive if self.split_P_arrive > 0 else None,
                cta_group=self.cta_group_size,
            )
            for stage in range(self.s_stage)
        ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        Int32(0)
        # Pipeline s_p_o phases for stage 0 and stage 1.
        # Must persist across tiles (like FA's P_full_O_rescaled_phase)
        # so that mbarrier phase stays in sync when block_iter_count varies.
        phase_s0 = Int32(0)
        phase_s1 = Int32(0)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            SeqlenInfoCls()

            if const_expr(mBlockNums is not None):
                raw_block_count = mBlockNums[batch_idx, head_idx, m_block]
                process_tile = (
                    raw_block_count > Int32(0)
                    if const_expr(self.allow_empty_block_nums)
                    else True
                )
                block_iter_count = (raw_block_count + 1) & ~1
            else:
                process_tile = True
                block_iter_count = block_sparse_num

            if process_tile and is_leader_cta:
                # ================================================================
                # q_stage=1: intra-warp overlap across n_block direction
                # Pipeline KV order: K0, K1, V0, K2, V1, K3, V2, ...
                # GEMM order: S0=Q@K0, S1=Q@K1, {O0+=P0@V0, S0=Q@K2}, {O1+=P1@V1, S1=Q@K3}, ...
                # ================================================================

                # Prologue: wait Q0
                pipeline_q.consumer_wait_w_index_phase(0, mma_q_consumer_phase)
                mma_q_consumer_phase ^= 1

                # S0 = Q @ K0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Ki_index, Ki_phase = (
                    mma_kv_consumer_state.index,
                    mma_kv_consumer_state.phase,
                )
                sK_cur = sK[None, None, None, Ki_index]
                if const_expr(self.uneven_kv_smem):
                    sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                gemm_Si[0](
                    smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(
                        sK_cur.iterator
                    )
                )
                pipeline_s_p_o.producer_commit_w_index(0)  # signal S0 ready
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

                # S1 = Q @ K1
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Ki_index, Ki_phase = (
                    mma_kv_consumer_state.index,
                    mma_kv_consumer_state.phase,
                )
                sK_cur = sK[None, None, None, Ki_index]
                if const_expr(self.uneven_kv_smem):
                    sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                gemm_Si[1](
                    smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(
                        sK_cur.iterator
                    )
                )
                pipeline_s_p_o.producer_commit_w_index(1)  # signal S1 ready
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

                # Flat loop: N-2 iterations, alternating stage 0/1
                block_loop_count = block_iter_count - 2
                O_acc_s0 = False  # per-stage accumulate flags
                O_acc_s1 = False
                # Pre-declare loop variables for DSL type tracking
                Vi_index = mma_kv_consumer_state.index
                Vi_phase = mma_kv_consumer_state.phase
                Ki_index = mma_kv_consumer_state.index
                Ki_phase = mma_kv_consumer_state.phase
                mma_kv_release_state = mma_kv_consumer_state.clone()
                tOrVi = tOrV[None, None, None, Vi_index]
                sV_cur = sV[None, None, None, Vi_index]
                sK_cur = sK[None, None, None, Ki_index]

                pair_count = (
                    block_loop_count // 2
                )  # N even => block_loop_count = N-2 is even
                for _i in cutlass.range(pair_count, unroll=1):
                    for stage in cutlass.range_constexpr(self.s_stage):
                        if const_expr(stage == 0):
                            phase_cur, O_acc_cur = phase_s0, O_acc_s0
                        else:
                            phase_cur, O_acc_cur = phase_s1, O_acc_s1
                        # Wait V
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        mma_kv_release_state = mma_kv_consumer_state.clone()
                        Vi_index, Vi_phase = (
                            mma_kv_consumer_state.index,
                            mma_kv_consumer_state.phase,
                        )
                        tOrVi = tOrV[None, None, None, Vi_index]
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        mma_kv_consumer_state.advance()
                        # Wait K
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = (
                            mma_kv_consumer_state.index,
                            mma_kv_consumer_state.phase,
                        )
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        pipeline_s_p_o.producer_acquire_w_index_phase(stage, phase_cur)
                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_acc_cur,
                            mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(
                                stage
                            )
                            if self.split_P_arrive > 0
                            else None,
                            mbar_phase=phase_cur,
                        )
                        gemm_Si[stage](
                            smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(
                                sK_cur.iterator
                            )
                        )
                        pipeline_s_p_o.producer_commit_w_index(stage)
                        if const_expr(stage == 0):
                            phase_s0 ^= 1
                            O_acc_s0 = True
                        else:
                            phase_s1 ^= 1
                            O_acc_s1 = True
                        # Release V and K
                        pipeline_kv.consumer_release(mma_kv_release_state)
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()

                # release Q0
                pipeline_q.consumer_release_w_index(0)

                # Epilogue: 2 PV GEMMs (N even: stage 0 first, stage 1 second)
                for epi_stage_constexpr in cutlass.range_constexpr(2):
                    if const_expr(epi_stage_constexpr == 0):
                        epi_s, epi_p, epi_zi = 0, phase_s0, not O_acc_s0
                    else:
                        epi_s, epi_p, epi_zi = 1, phase_s1, not O_acc_s1
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Vi_index, Vi_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )
                    tOrVi = tOrV[None, None, None, Vi_index]
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    pipeline_s_p_o.producer_acquire_w_index_phase(epi_s, epi_p)
                    gemm_Pi[epi_stage_constexpr](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        zero_init=epi_zi,
                        mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(
                            epi_s
                        )
                        if self.split_P_arrive > 0
                        else None,
                        mbar_phase=epi_p,
                    )
                    pipeline_o_acc.producer_commit_w_index(epi_s)
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                # Epilogue did one acquire per stage; advance phases for next tile
                phase_s0 ^= 1
                phase_s1 ^= 1

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()
        # End of persistent scheduler loop

        # We don't need pipeline_s_p_o.producer_tail() since there's no dangling mbarrier at the end
        # We don't need pipeline_o_acc.producer_tail() since we don't call
        # pipeline_o_acc.producer_acquire() inside the loop.

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        thr_mma_qk: cute.ThrMma,
        tStS: cute.Tensor,  # ((TILE_M, TILE_N), 1, 1, q_stage)
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        mBlockIndex: cute.Tensor,
        mBlockSizes: cute.Tensor,
        block_sparse_num: Int32,
        mBlockNums: Optional[cute.Tensor],
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * (len(self.softmax0_warp_ids))
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        (
            self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape,
            self.mma_tiler_qk[1],
        )
        tSAcc = tStS[(None, None), 0, 0, stage]  # (128, 128)
        tStScale = cute.composition(tSAcc, cute.make_layout((self.m_block_size, 1)))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)  # (((32,32),1),1,4)

        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)), Float32
        )
        thr_tmem_store_scale = tcgen05.make_tmem_copy(
            tmem_store_scale_atom, tStScale
        ).get_slice(tidx)
        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)  # (((16,32),1),1,4)

        mma_si_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            # n_block(i): maps logical index i to actual KV block index via q2k_block_index
            if const_expr(mBlockNums is not None):
                raw_block_count = mBlockNums[batch_idx, head_idx, m_block]
                has_work = (
                    raw_block_count > Int32(0)
                    if const_expr(self.allow_empty_block_nums)
                    else True
                )
                n_block = partial(
                    block_info.get_n_block_idx,
                    mBlockIndex,
                    batch_idx,
                    head_idx,
                    m_block,
                    max_i=cutlass.max(raw_block_count - 1, Int32(0)),
                )
            else:
                raw_block_count = block_sparse_num
                has_work = True
                n_block = partial(
                    block_info.get_n_block_idx,
                    mBlockIndex,
                    batch_idx,
                    head_idx,
                    m_block,
                )

            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=8.0 if const_expr(self.q_dtype.width == 16) else 0.0,
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
            )

            # Always acquire pipeline_sm_stats to stay in sync with correction
            pipeline_sm_stats.producer_acquire_w_index_phase(
                stage, sm_stats_producer_phase
            )
            sm_stats_producer_phase ^= 1

            if has_work:
                # block_iter_count is even: each WG processes exactly half the blocks
                # WG0 (stage=0): logical indices N-1, N-3, ... (stride 2)
                # WG1 (stage=1): logical indices N-2, N-4, ... (stride 2)
                if const_expr(mBlockNums is not None):
                    block_iter_count = (raw_block_count + 1) & ~1
                else:
                    block_iter_count = block_sparse_num
                wg_count = block_iter_count // 2
                # logical_first is the first logical index for this WG
                logical_first = block_iter_count - 1 - stage

                # 1st block — phantom block (logical_first >= raw_block_count) uses block_size=0
                n_block_first = n_block(logical_first)
                if const_expr(self.has_block_sizes):
                    first_block_size = (
                        Int32(0)
                        if const_expr(mBlockNums is not None)
                        and logical_first >= raw_block_count
                        else mBlockSizes[n_block_first]
                    )
                    first_mask_fn = partial(
                        apply_block_size_mask,
                        block_size=first_block_size,
                        n_block_size=self.n_block_size,
                    )
                elif const_expr(mBlockNums is not None):
                    # No block_sizes but var block nums: phantom block still needs block_size=0 mask;
                    # real blocks get n_block_size which is a no-op in apply_block_size_mask
                    first_block_size = (
                        Int32(0)
                        if logical_first >= raw_block_count
                        else Int32(self.n_block_size)
                    )
                    first_mask_fn = partial(
                        apply_block_size_mask,
                        block_size=first_block_size,
                        n_block_size=self.n_block_size,
                    )
                else:
                    first_mask_fn = None
                mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase = (
                    softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                        mask_fn=first_mask_fn,
                        is_first=True,
                    )
                )
                # Remaining blocks with stride 2 — always valid (logical_n < raw_block_count)
                for n_tile in cutlass.range(wg_count - 1, unroll=1):
                    logical_n = logical_first - self.s_stage * (n_tile + 1)
                    n_block_cur = n_block(logical_n)
                    if const_expr(self.has_block_sizes):
                        remaining_mask_fn = partial(
                            apply_block_size_mask,
                            block_size=mBlockSizes[n_block_cur],
                            n_block_size=self.n_block_size,
                        )
                    else:
                        remaining_mask_fn = None
                    (
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                    ) = softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                        mask_fn=remaining_mask_fn,
                    )

                sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                if const_expr(mLSE is not None or self.q_stage == 1):
                    sScale[
                        tidx
                        + stage * self.m_block_size
                        + self.s_stage * self.m_block_size
                    ] = softmax.row_max[0]
                sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)
            else:
                # Empty tile: arrive barrier once (synthetic "no work" signal for correction)
                sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_sm_stats.producer_tail
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        cta_qk_tiler = (
            self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape,
            self.mma_tiler_qk[1],
        )
        tScS_shape = cta_qk_tiler  # (128, 128)
        tScP_shape = (tScS_shape[0], tilePlikeFP32)  # (128, 64)

        # Wait for Si
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_rmem_tensor(
            thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype
        )
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r)
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        # Notify correction wg that row_max is ready
        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape,
            Float32,
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout
        )
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            # ex2_emu_freq=self.ex2_emu_freq if const_expr(mask_fn is None) else 0,
            ex2_emu_freq=self.ex2_emu_freq,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i]
            )
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = (
                    cute.size(tStP_r2t.shape[2])
                    * self.split_P_arrive
                    // self.n_block_size
                )
                if const_expr(i + 1 == split_P_arrive_idx):
                    # Notify mma warp that the 1st half of P is ready
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        return (
            mma_si_consumer_phase ^ 1,
            sm_stats_producer_phase ^ 1,
            s0_s1_sequence_phase ^ 1,
        )

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_o_epi: pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        block_sparse_num: Int32,
        mBlockNums: Optional[cute.Tensor],
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mma_tile_coord_v = thr_mma_qk.thr_idx

        # First iter: no correction is required
        # Notify mma warp that O has been rescaled
        for stage in cutlass.range(self.s_stage):
            pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls()

            mO_cur = mO[None, None, None, batch_idx][None, None, head_idx]
            # For q_stage=1, gO tiles span 1*m_block_size rows (single Q, combine writes one O)
            tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
            gO = layout_utils.select(
                cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
            )
            gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[
                None, mma_tile_coord_v, None, None
            ]

            # For q_stage=1, always need row_max for combine; use -inf as default
            stats = [
                (
                    Float32(0.0),
                    -Float32.inf
                    if const_expr(mLSE is not None or self.q_stage == 1)
                    else None,
                    True,
                )
            ] * self.s_stage

            if const_expr(mBlockNums is not None):
                has_work = (
                    mBlockNums[batch_idx, head_idx, m_block] > Int32(0)
                    if const_expr(self.allow_empty_block_nums)
                    else True
                )
            else:
                has_work = True

            if has_work:
                # Ignore first signal from softmax as no correction is required
                sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                pipeline_sm_stats.consumer_release_w_index(0)
                # Both q_stage=1 and q_stage=2 have 2 softmax WGs, wait for both
                sm_stats_barrier.arrive_and_wait_w_index(index=1 * 4 + warp_idx)
                sm_stats_consumer_phase ^= 1

                # q_stage=1 correction loop
                if const_expr(mBlockNums is not None):
                    block_iter_count = (
                        mBlockNums[batch_idx, head_idx, m_block] + 1
                    ) & ~1
                else:
                    block_iter_count = block_sparse_num
                corr_pair_count = (block_iter_count - 2) // 2
                # Paired rescale loop (same structure as q_stage=2)
                for _i in cutlass.range(corr_pair_count, unroll=1):
                    for stage in cutlass.range_constexpr(self.s_stage):
                        sm_stats_barrier.arrive_and_wait_w_index(
                            index=stage * 4 + warp_idx
                        )
                        scale = sScale[tidx + stage * self.m_block_size]
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        if should_rescale:
                            self.correction_rescale(
                                thr_mma_pv, tOtO[None, None, None, stage], tidx, scale
                            )
                        pipeline_s_p_o.consumer_release_w_index(stage)
                        pipeline_sm_stats.consumer_release_w_index(
                            self.s_stage - 1 - stage
                        )
                    sm_stats_consumer_phase ^= 1
                # N even: no remainder. Release final sm_stats stage 1.
                pipeline_sm_stats.consumer_release_w_index(1)
                # End of seqlen_corr_loop_steps

                # Even in the case of self.overlap_sO_sQ, we can write to stage 0 of sO without
                # additional sync because the MMA in the top half must have been done.
                # Similarly we can write to stage 1 of sO without additional sync.

                # Read final softmax stats for both stages
                for stage in cutlass.range_constexpr(self.s_stage):
                    sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                    row_sum = sScale[tidx + stage * self.m_block_size]
                    if const_expr(mLSE is not None or self.q_stage == 1):
                        row_max = sScale[
                            tidx
                            + stage * self.m_block_size
                            + self.s_stage * self.m_block_size
                        ]
                    else:
                        row_max = None
                    pipeline_sm_stats.consumer_release_w_index(stage)
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)

                # q_stage=1: combine O0 and O1, then write single output
                row_sum0, row_max0, zero_or_nan0 = stats[0]
                row_sum1, row_max1, zero_or_nan1 = stats[1]

                # Compute combined scales for the two partial O accumulators
                # row_max is in original S space (unscaled). To compute rescale
                # factors we need exp2((row_max - max_combined) * scale_log2).
                # For empty/padding stages (zero_or_nan=True), row_max may be 0.0 instead of -inf
                # due to softmax's safe_max clamping. Use -inf for those to avoid polluting max_combined.
                rm0 = row_max0 if not zero_or_nan0 else -Float32.inf
                rm1 = row_max1 if not zero_or_nan1 else -Float32.inf
                max_combined = cutlass.max(rm0, rm1)
                max_safe = (
                    max_combined if max_combined != -Float32.inf else Float32(0.0)
                )
                scale0 = (
                    cute.math.exp2((rm0 - max_safe) * softmax_scale_log2, fastmath=True)
                    if not zero_or_nan0
                    else Float32(0.0)
                )
                scale1 = (
                    cute.math.exp2((rm1 - max_safe) * softmax_scale_log2, fastmath=True)
                    if not zero_or_nan1
                    else Float32(0.0)
                )
                sum_combined = row_sum0 * scale0 + row_sum1 * scale1
                combined_zero_or_nan = (
                    sum_combined == 0.0 or sum_combined != sum_combined
                )
                inv_sum = cute.arch.rcp_approx(
                    sum_combined if not combined_zero_or_nan else 1.0
                )
                final_scale0 = scale0 * inv_sum
                final_scale1 = scale1 * inv_sum

                # Wait for both O accumulators from MMA warp
                for stage in cutlass.range_constexpr(self.s_stage):
                    pipeline_o_acc.consumer_wait_w_index_phase(
                        stage, o_corr_consumer_phase
                    )
                if const_expr(not self.use_correction_warps_for_epi):
                    pipeline_o_epi.producer_acquire_w_index_phase(
                        0, corr_epi_producer_phase
                    )
                self.correction_epilogue_combine(
                    thr_mma_pv,
                    tOtO[None, None, None, 0],
                    tOtO[None, None, None, 1],
                    tidx,
                    m_block,
                    seqlen.seqlen_q,
                    final_scale0,
                    final_scale1,
                    sO[None, None, 0],
                    mO_cur,
                    gO[None, None, 0],
                    gmem_tiled_copy_O,
                )
                # Release both O buffers in tmem
                for stage in cutlass.range_constexpr(self.s_stage):
                    pipeline_s_p_o.consumer_release_w_index(stage)
                if const_expr(not self.use_correction_warps_for_epi):
                    pipeline_o_epi.producer_commit_w_index(0)

                o_corr_consumer_phase ^= 1
                sm_stats_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            else:
                # Empty tile (block_count == 0): sync pipelines and write O=0.
                # Match softmax's 1 barrier arrive per stage.
                for stage_idx in cutlass.range_constexpr(self.s_stage):
                    sm_stats_barrier.arrive_and_wait_w_index(
                        index=stage_idx * 4 + warp_idx
                    )
                    pipeline_sm_stats.consumer_release_w_index(stage_idx)
                sm_stats_consumer_phase ^= 1
                # Write O=0 via correction_epilogue_combine with scale=0.
                # Note: reads tmem which may have values from a previous tile;
                # 0.0 * finite = 0.0. For the very first tile, tmem is hardware-zero-initialized.
                if const_expr(not self.use_correction_warps_for_epi):
                    pipeline_o_epi.producer_acquire_w_index_phase(
                        0, corr_epi_producer_phase
                    )
                self.correction_epilogue_combine(
                    thr_mma_pv,
                    tOtO[None, None, None, 0],
                    tOtO[None, None, None, 1],
                    tidx,
                    m_block,
                    seqlen.seqlen_q,
                    Float32(0.0),
                    Float32(0.0),
                    sO[None, None, 0],
                    mO_cur,
                    gO[None, None, 0],
                    gmem_tiled_copy_O,
                )
                # Do NOT release pipeline_s_p_o (MMA didn't commit)
                if const_expr(not self.use_correction_warps_for_epi):
                    pipeline_o_epi.producer_commit_w_index(0)
                # o_corr_consumer_phase NOT toggled (pipeline_o_acc not touched)
                corr_epi_producer_phase ^= 1

            if const_expr(mLSE is not None):
                mLSE_cur = mLSE[None, head_idx, batch_idx]
                # q_stage=1: compute combined LSE from two partial softmax stats
                m_tile_idx = m_block * self.cta_group_size + mma_tile_coord_v
                gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_tile_idx,))
                row_sum0, row_max0, zero_or_nan0 = stats[0]
                row_sum1, row_max1, zero_or_nan1 = stats[1]
                rm0_lse = row_max0 if not zero_or_nan0 else -Float32.inf
                rm1_lse = row_max1 if not zero_or_nan1 else -Float32.inf
                max_combined = cutlass.max(rm0_lse, rm1_lse)
                max_safe = (
                    max_combined if max_combined != -Float32.inf else Float32(0.0)
                )
                s0 = (
                    cute.math.exp2(
                        (rm0_lse - max_safe) * softmax_scale_log2, fastmath=True
                    )
                    if not zero_or_nan0
                    else Float32(0.0)
                )
                s1 = (
                    cute.math.exp2(
                        (rm1_lse - max_safe) * softmax_scale_log2, fastmath=True
                    )
                    if not zero_or_nan1
                    else Float32(0.0)
                )
                sum_comb = row_sum0 * s0 + row_sum1 * s1
                comb_zero_or_nan = sum_comb == 0.0 or sum_comb != sum_comb
                LN2 = math.log(2.0)
                lse = (
                    (
                        max_safe * softmax_scale_log2
                        + cute.math.log2(sum_comb, fastmath=True)
                    )
                    * LN2
                    if not comb_zero_or_nan
                    else -Float32.inf
                )
                seqlen_q = (
                    seqlen.seqlen_q
                    if const_expr(not self.pack_gqa)
                    else seqlen.seqlen_q * self.qhead_per_kvhead
                )
                if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                    gLSE[tidx] = lse

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_o_epi.consumer_tail() for the correction warps
        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi.producer_acquire_w_index_phase(
                self.q_stage - 1, corr_epi_producer_phase
            )

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(
            tOtO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOcO_i = cute.composition(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count = self.head_dim_v_padded // corr_tile_size
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_rmem_tensor(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(
                tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout
            )
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            tOtO_r2t_i = cute.make_tensor(
                tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout
            )
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory."""

        corr_tile_size = 8 * 32 // self.o_dtype.width
        # Use CTA 0 mapping for smem partitioning since sO is per-CTA sized
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(
            tOtO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOcO_i = cute.logical_divide(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOsO_i = cute.logical_divide(
            tOsO, cute.make_layout((self.m_block_size, corr_tile_size))
        )

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=self.use_2cta_instrs,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_copy_atom, tOtO_i[(None, None), 0]
        )
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(
            thr_tmem_load, tOsO_i[(None, None), None]
        )
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range(
            self.head_dim_v_padded // corr_tile_size, unroll_full=True
        ):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_rmem_tensor(
                tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype
            )
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            copy_utils.cvt_copy(tiled_smem_store, tOrO_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

        if const_expr(self.use_correction_warps_for_epi):
            assert not self.use_tma_O
            assert gmem_tiled_copy_O is not None
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwdSm100.Epilogue),
                number_of_threads=len(self.epilogue_warp_ids) * cute.arch.WARP_SIZE,
            )
            mma_tile_coord_v = thr_mma.thr_idx
            m_tile_idx = (
                m_block * self.q_stage + stage
            ) * self.cta_group_size + mma_tile_coord_v
            self._store_O_to_gmem(
                sO, gO, mO_cur, gmem_tiled_copy_O, tidx, seqlen_q, m_tile_idx
            )

    @cute.jit
    def correction_epilogue_combine(
        self,
        thr_mma: cute.ThrMma,
        tOtO0: cute.Tensor,
        tOtO1: cute.Tensor,
        tidx: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        scale0: Float32,
        scale1: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
    ):
        """Combine two partial O accumulators (O0, O1) from intra-warp overlap and write to smem/gmem.

        For q_stage=1 intra-warp overlap, O0 and O1 accumulate results from even and odd
        n_blocks respectively. This method reads both from TMEM, applies their respective
        scales (derived from combined softmax stats), adds them, and writes the result.
        """
        corr_tile_size = 8 * 32 // self.o_dtype.width
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO0_i = cute.logical_divide(
            tOtO0, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOtO1_i = cute.logical_divide(
            tOtO1, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOcO_i = cute.logical_divide(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOsO_i = cute.logical_divide(
            tOsO, cute.make_layout((self.m_block_size, corr_tile_size))
        )

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=self.use_2cta_instrs,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_copy_atom, tOtO0_i[(None, None), 0]
        )
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO0_t2r = thr_tmem_load.partition_S(tOtO0_i[(None, None), None])
        tOtO1_t2r = thr_tmem_load.partition_S(tOtO1_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(
            thr_tmem_load, tOsO_i[(None, None), None]
        )
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range(
            self.head_dim_v_padded // corr_tile_size, unroll_full=True
        ):
            tOtO0_t2r_i = tOtO0_t2r[None, 0, 0, i]
            tOtO1_t2r_i = tOtO1_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            frg_shape = tOcO_t2r[None, 0, 0, i].shape
            tOrO0_frg = cute.make_rmem_tensor(frg_shape, self.pv_acc_dtype)
            tOrO1_frg = cute.make_rmem_tensor(frg_shape, self.pv_acc_dtype)
            # When both scales are 0 (empty tile), skip tmem reads to avoid 0*NaN=NaN.
            is_zero_output = scale0 == Float32(0.0) and scale1 == Float32(0.0)
            if not is_zero_output:
                cute.copy(tiled_tmem_load, tOtO0_t2r_i, tOrO0_frg)
                cute.copy(tiled_tmem_load, tOtO1_t2r_i, tOrO1_frg)
                # Combined: O = O0 * scale0 + O1 * scale1
                for j in cutlass.range(0, cute.size(tOrO0_frg), 2, unroll_full=True):
                    o0_a, o0_b = cute.arch.mul_packed_f32x2(
                        (tOrO0_frg[j], tOrO0_frg[j + 1]), (scale0, scale0)
                    )
                    o1_a, o1_b = cute.arch.mul_packed_f32x2(
                        (tOrO1_frg[j], tOrO1_frg[j + 1]), (scale1, scale1)
                    )
                    tOrO0_frg[j], tOrO0_frg[j + 1] = cute.arch.add_packed_f32x2(
                        (o0_a, o0_b), (o1_a, o1_b)
                    )
            else:
                tOrO0_frg.fill(Float32(0.0))
            copy_utils.cvt_copy(tiled_smem_store, tOrO0_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

        if const_expr(self.use_correction_warps_for_epi):
            assert not self.use_tma_O
            assert gmem_tiled_copy_O is not None
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwdSm100.Epilogue),
                number_of_threads=len(self.epilogue_warp_ids) * cute.arch.WARP_SIZE,
            )
            mma_tile_coord_v = thr_mma.thr_idx
            m_tile_idx = m_block * self.cta_group_size + mma_tile_coord_v
            self._store_O_to_gmem(
                sO, gO, mO_cur, gmem_tiled_copy_O, tidx, seqlen_q, m_tile_idx
            )

    @cute.jit
    def _store_O_to_gmem(
        self,
        sO_stage: cute.Tensor,
        gO: cute.Tensor,
        mO_cur: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tidx: Int32,
        seqlen_q: Int32,
        m_tile_idx: Int32,
    ):
        """Copy a single stage of O from smem to gmem via registers."""
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO_stage)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOcO = gmem_thr_copy_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOpO = copy_utils.predicate_k(tOcO, limit=mO_cur.shape[1])
        pack_gqa = PackGQA(
            self.m_block_size,
            self.head_dim_v_padded,
            self.check_hdim_v_oob,
            self.qhead_per_kvhead,
        )

        # load acc O from smem to rmem for wider vectorization
        tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
        cute.autovec_copy(tOsO, tOrO)
        # copy acc O from rmem to gmem
        if const_expr(not self.pack_gqa):
            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                if (
                    t0OcO[0, rest_m, 0][0]
                    < seqlen_q - m_tile_idx * self.m_block_size - tOcO[0][0]
                ):
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO[None, rest_m, None],
                        tOgO[None, rest_m, None],
                        pred=tOpO[None, rest_m, None]
                        if const_expr(self.check_hdim_v_oob)
                        else None,
                    )
        else:
            pack_gqa.store_O(
                mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_tile_idx, seqlen_q
            )

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        pipeline_o_epi: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        mma_tile_coord_v: Int32 = 0,
    ):
        epi_consumer_phase = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls()

            mO_cur = mO[None, None, None, batch_idx][None, None, head_idx]
            tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))  # (128, 128)
            gO = layout_utils.select(
                cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
            )  # (128, 128, 1)
            gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[
                None, mma_tile_coord_v, None, None
            ]

            if const_expr(self.use_tma_O):
                store_O, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_O, 0, cute.make_layout(1), sO, gO
                )
                for stage in cutlass.range(self.q_stage, unroll_full=True):
                    # wait from corr, issue tma store on smem
                    # 1. wait for O0 final
                    pipeline_o_epi.consumer_wait_w_index_phase(
                        stage, epi_consumer_phase
                    )
                    # 2. copy O0 to gmem
                    store_O(src_idx=stage, dst_idx=stage)
                    cute.arch.cp_async_bulk_commit_group()
                for stage in cutlass.range_constexpr(self.q_stage):
                    # Ensure O0 buffer is ready to be released
                    cute.arch.cp_async_bulk_wait_group(
                        self.q_stage - 1 - stage, read=True
                    )
                    pipeline_o_epi.consumer_release_w_index(stage)
            else:
                tidx = cute.arch.thread_idx()[0] % (
                    cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
                )
                for stage in cutlass.range_constexpr(self.q_stage):
                    # wait from corr, issue tma store on smem
                    # 1. wait for O0 final
                    pipeline_o_epi.consumer_wait_w_index_phase(
                        stage, epi_consumer_phase
                    )
                    # 2. copy O0 to gmem
                    m_tile_idx = (
                        m_block * self.q_stage + stage
                    ) * self.cta_group_size + mma_tile_coord_v
                    self._store_O_to_gmem(
                        sO[None, None, stage],
                        gO[None, None, stage],
                        mO_cur,
                        gmem_tiled_copy_O,
                        tidx,
                        seqlen.seqlen_q,
                        m_tile_idx,
                    )
                    pipeline_o_epi.consumer_release_w_index(stage)

            epi_consumer_phase ^= 1

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()

    def load_Q(
        self,
        load_Q_fn: Callable,
        pipeline_q: pipeline.PipelineAsync,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        load_Q_fn(
            src_idx=block,
            dst_idx=stage,
            tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage),
        )

    @cute.jit
    def load_KV(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        sX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
        extra_tx_count: Optional[Int32] = None,
    ):
        assert K_or_V in ("K", "V")
        stage, phase = producer_state.index, producer_state.phase
        extra_tx_count_kv = self.tma_copy_bytes[K_or_V] - self.tma_copy_bytes["K"]
        extra_tx_count = extra_tx_count_kv + (
            extra_tx_count if extra_tx_count is not None else 0
        )
        extra_kwargs = {"extra_tx_count": extra_tx_count}
        pipeline_kv.producer_acquire(producer_state, **extra_kwargs)
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            # Before this round, the smem location was occupied by V, which is smaller than
            # K. So we need to wait for the stage after that (stage 1) to be empty as well.
            if stage == 0:
                pipeline_kv.sync_object_empty.wait(1, phase)

        tXsX_cur = tXsX[None, stage]
        if const_expr(self.uneven_kv_smem):
            # Since this is the producer_state, the phase starts at 1, so we have to invert it
            tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)
        tXgX_cur = (
            tXgX[None, block]
            if const_expr(page_idx is None)
            else tXgX[None, 0, page_idx]
        )
        cute.copy(
            tma_atom,
            tXgX_cur,
            tXsX_cur,
            tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state),
        )

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX
