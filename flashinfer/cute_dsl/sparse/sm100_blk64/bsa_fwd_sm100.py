# Supported features:
# - BF16 & FP16 dtype
# - noncausal attention
# - MHA, GQA, MQA
# - hdim 128.
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

import math
from typing import Tuple, Callable, Optional
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

from . import quack_compat  # noqa: F401
from . import copy_utils, layout_utils

from .cute_dsl_utils import assume_tensor_aligned
from . import kernel_utils as utils
from . import pipeline as pipeline_custom
from .softmax import SoftmaxSm100
from .seqlen_info import SeqlenInfoQK
from .pack_gqa import PackGQA, pack_gqa_layout
from . import bsa_fwd_helpers
from .named_barrier import NamedBarrierFwdSm100
from .cute_dsl_utils import ParamsBase
import cutlass.pipeline as cutlass_pipeline
from .block_sparse_tile_scheduler import (
    TileSchedulerArguments,
    TileSchedulerProtocol,
    SchedulingMode,
    BlockSparsePersistentTileScheduler,
)
from .tile_scheduler import SingleTileScheduler


SAGE_P_QUANT_SCALE = 256.0
SAGE_P_QUANT_LOG2_SCALE = math.log2(SAGE_P_QUANT_SCALE)
SAGE_P_RESCALE_THRESHOLD = math.log2(448.0 / SAGE_P_QUANT_SCALE)


class BlockSparseAttnForwardSm100Blk64:
    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        pack_gqa: cutlass.Constexpr[bool] = False,
        m_block_size: cutlass.Constexpr[int] = 64,
        n_block_size: cutlass.Constexpr[int] = 256,
        sparse_block_size: cutlass.Constexpr[int] = 64,
        is_persistent: cutlass.Constexpr[bool] = True,
        use_clc_scheduler: cutlass.Constexpr[bool] = False,
        allow_empty_block_nums: cutlass.Constexpr[bool] = False,
        has_block_sizes: cutlass.Constexpr[bool] = True,
        num_splits: cutlass.Constexpr[int] = 1,
        use_int64_kv_strides: cutlass.Constexpr[bool] = False,
    ):
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(
            math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of
        )
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.head_dim_v_padded = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        assert head_dim == 128 and head_dim_v == 128, (
            "blk64 CuTeDSL fwd currently requires D=DV=128"
        )
        assert m_block_size == 64, "blk64 CuTeDSL fwd requires 64-row tiles"
        assert n_block_size == 256, "blk64 CuTeDSL fwd requires 256-column tiles"
        assert sparse_block_size == 64, (
            "blk64 CuTeDSL fwd requires 64-token sparse blocks"
        )
        self.m_block_size = m_block_size
        self.sparse_block_size = sparse_block_size
        self.sparse_blocks_per_kv = n_block_size // sparse_block_size
        self.n_block_size = n_block_size
        self.kv_mma_k = 128
        self.output_cols = 128
        self.kv_elems_per_stage = self.n_block_size * self.kv_mma_k
        self.q_stage = 1
        # If split_P_arrive, the softmax warps write some columns of P first, signal to the MMA warp
        # to being the P @ V MMA, then write the rest of P and signal again. This allows some overlap
        # between compute the last couple columns of P and the P @ V MMA.
        # Split the numerator work at 1/4: softmax releases the first 32 P
        # columns early, then MMA waits on
        # p_lastsplit before consuming the remaining P fragments.
        self.split_P_arrive = 32
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert self.arch.is_family_of(Arch.sm_100f) or self.arch.is_family_of(
            Arch.sm_110f
        ), "Only SM 10.x and 11.x are supported"

        self.cta_tiler = (m_block_size, self.n_block_size, self.head_dim_padded)
        self.mma_tiler_qk = (
            m_block_size,
            self.n_block_size,
            self.head_dim_padded,
        )
        # WS TS PV uses M=64, N=256, K=128.
        # The final 128 output columns are recovered by the correction epilogue's
        # WS TMEM load/store mapping.
        self.mma_tiler_pv = (
            m_block_size,
            self.n_block_size,
            self.head_dim_v_padded,
        )
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.is_persistent = is_persistent
        # CLC persistent scheduling
        self.use_clc_scheduler = use_clc_scheduler
        self.sched_stages = 1
        self.scheduling_mode = (
            SchedulingMode.CLC if self.use_clc_scheduler else SchedulingMode.STATIC
        )
        assert num_splits >= 1, "num_splits must be >= 1"
        self.num_splits = num_splits
        self.is_split_kv = num_splits > 1
        self.allow_empty_block_nums = allow_empty_block_nums
        self.has_block_sizes = has_block_sizes
        self.use_int64_kv_strides = use_int64_kv_strides
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = pack_gqa
        if pack_gqa:
            assert m_block_size % self.qhead_per_kvhead == 0, (
                "For PackGQA, m_block_size must be divisible by qhead_per_kvhead"
            )
        is_sm103 = self.arch >= Arch.sm_103 and self.arch <= Arch.sm_103f
        self.enable_ex2_emu = self.head_dim_padded <= 128 and not is_sm103

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.stats_stride = cute.arch.WARP_SIZE * len(self.correction_warp_ids)
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

        self.s_stage = 2  # Always 2: for q_stage=1 it's n-direction
        # Use two 128-column S/P stages and two 128-column O stages.
        self.tmem_s_offset = [0, 128]
        self.tmem_o_offset = [256, 384]
        self.tmem_total = 512
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_p_offset = self.tmem_s_offset
        self.tmem_s_to_p_offset = 0
        self.epi_exchange_f32_count = 4 * 4 * 32 * 32
        # Filled in once the input dtype is known.  This is an offset in
        # elements of the shared K/V allocation, not an offset in bytes.
        self.epi_sO_input_offset = self.epi_exchange_f32_count * 2

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200
            self.num_regs_correction = 64
            self.num_regs_other = 48
        else:
            self.num_regs_softmax = 184
            self.num_regs_correction = 88
            self.num_regs_other = 48

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        # Sage keeps both P and V in E4M3 for one native FP8 PV MMA.  Map the
        # unnormalized softmax probabilities from [0, 1] to the full E4M3
        # nominal range [0, 256]. Values may use the remaining E4M3 headroom up
        # to 448 while online softmax keeps its old reference maximum across
        # small increases, avoiding an O TMEM rescale. The same factor is carried
        # by the softmax sum, so it cancels during normalization and only V's
        # per-channel scale remains.
        self.kv_stage = 3
        # self.s_stage is defined in __init__ (always 2)
        assert self.s_stage >= self.q_stage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, h, s_q, d)
        mK: cute.Tensor,  # (b_k, h_k, s_k, d)
        mV: cute.Tensor,  # (b_k, h_k, s_k, dv)
        mO: cute.Tensor,  # (b, h, s_q, dv)
        mLSE: Optional[cute.Tensor],
        mQScale: Optional[cute.Tensor],  # Sage FP8: (b, h, s_q)
        mKScale: Optional[cute.Tensor],  # Sage FP8: (b, h, ceil(s_k / 16))
        mVScale: Optional[cute.Tensor],  # Sage FP8: (h, dv)
        softmax_scale: Float32,
        mBlockIndex: cute.Tensor,  # (batch, heads, num_q_blocks, max_kv_blocks), int32
        mBlockSizes: Optional[cute.Tensor],  # (num_kv_blocks,), int32 or None
        block_sparse_num: Int32,  # runtime scalar, even, >= 2
        mBlockNums: Optional[
            cute.Tensor
        ],  # (batch, heads, num_q_blocks), int32 or None
        mSplitOffsets: Optional[
            cute.Tensor
        ],  # (batch, heads, num_q_blocks, num_splits + 1), int32 or None
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
        self.epi_sO_input_offset = self.epi_exchange_f32_count * (
            Float32.width // self.k_dtype.width
        )
        self.is_sage_fp8 = mQScale is not None
        if const_expr(self.is_sage_fp8):
            if const_expr(mKScale is None or mVScale is None):
                raise TypeError("Sage FP8 requires Q, K, and V scale tensors")
            if const_expr(self.q_dtype != cutlass.Float8E4M3FN):
                raise TypeError("Sage FP8 inputs must be E4M3")
            if const_expr(
                mQScale.element_type != Float32
                or mKScale.element_type != Float32
                or mVScale.element_type != Float32
            ):
                raise TypeError("Sage FP8 scales must be FP32")
            expected_o_dtype = (
                Float32 if const_expr(self.is_split_kv) else cutlass.BFloat16
            )
            if const_expr(self.o_dtype != expected_o_dtype):
                raise TypeError(
                    "Sage FP8 output must be FP32 partials for split-KV, otherwise BF16"
                )
        elif const_expr(mKScale is not None or mVScale is not None):
            raise TypeError("Q, K, and V scales must be provided together")
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [2, 3, 1, 0]
        mQ_seq = cute.make_tensor(
            mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose)
        )
        seqlen_q_static = mQ_seq.shape[0]
        num_q_heads_static = mQ_seq.shape[2]
        batch_size_static = mQ_seq.shape[3]
        if const_expr(self.is_sage_fp8):
            q_stride_s = Int32(mQ_seq.layout.stride[0])
            q_stride_d = Int32(mQ_seq.layout.stride[1])
            q_stride_h = Int32(mQ_seq.layout.stride[2])
            q_stride_b = Int32(mQ_seq.layout.stride[3])
            mQ = cute.make_tensor(
                mQ_seq.iterator,
                cute.make_layout(
                    (
                        self.head_dim_padded,
                        32,
                        2,
                        mQ_seq.shape[2],
                        cute.ceil_div(mQ_seq.shape[0], self.m_block_size),
                        mQ_seq.shape[3],
                    ),
                    stride=(
                        q_stride_d,
                        q_stride_s,
                        32 * q_stride_s,
                        q_stride_h,
                        self.m_block_size * q_stride_s,
                        q_stride_b,
                    ),
                ),
            )
        else:
            mQ = mQ_seq
        # (s_k, d, h_k, b_k)
        KV_layout_transpose = [2, 3, 1, 0]
        mK_seq, mV_seq = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=KV_layout_transpose)
            )
            for t in (mK, mV)
        ]
        O_layout_transpose = [2, 3, 1, 0]
        LSE_layout_transpose = [2, 1, 0]
        num_splits = Int32(self.num_splits)
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
        # The fast rank-6 view matches the sparse-block layout. CuTe DSL
        # cannot lower an Int64 basis in that rank-6 TMA view, so layouts with
        # large active strides use a rank-5 Int64 view and divide it into
        # sparse blocks in the device kernel instead.
        k_dim_half = self.head_dim_padded // 2
        v_dim_part = self.head_dim_v_padded // 2
        if const_expr(self.use_int64_kv_strides):
            k_stride_s = Int64(mK_seq.layout.stride[0])
            k_stride_d = Int64(mK_seq.layout.stride[1])
            k_stride_h = Int64(mK_seq.layout.stride[2])
            k_stride_b = Int64(mK_seq.layout.stride[3])
            mK = cute.make_tensor(
                mK_seq.iterator,
                cute.make_layout(
                    (
                        mK_seq.shape[0],
                        k_dim_half,
                        2,
                        mK_seq.shape[2],
                        mK_seq.shape[3],
                    ),
                    stride=(
                        k_stride_s,
                        k_stride_d,
                        k_dim_half * k_stride_d,
                        k_stride_h,
                        k_stride_b,
                    ),
                ),
            )
            v_stride_s = Int64(mV_seq.layout.stride[0])
            v_stride_d = Int64(mV_seq.layout.stride[1])
            v_stride_h = Int64(mV_seq.layout.stride[2])
            v_stride_b = Int64(mV_seq.layout.stride[3])
            mV = cute.make_tensor(
                mV_seq.iterator,
                cute.make_layout(
                    (
                        v_dim_part,
                        mV_seq.shape[0],
                        2,
                        mV_seq.shape[2],
                        mV_seq.shape[3],
                    ),
                    stride=(
                        v_stride_d,
                        v_stride_s,
                        v_dim_part * v_stride_d,
                        v_stride_h,
                        v_stride_b,
                    ),
                ),
            )
        else:
            k_stride_s = Int32(mK_seq.layout.stride[0])
            k_stride_d = Int32(mK_seq.layout.stride[1])
            k_stride_h = Int32(mK_seq.layout.stride[2])
            k_stride_b = Int32(mK_seq.layout.stride[3])
            if const_expr(self.is_sage_fp8):
                mK = cute.make_tensor(
                    mK_seq.iterator,
                    cute.make_layout(
                        (
                            self.head_dim_padded,
                            32,
                            2,
                            mK_seq.shape[2],
                            cute.ceil_div(mK_seq.shape[0], self.sparse_block_size),
                            mK_seq.shape[3],
                        ),
                        stride=(
                            k_stride_d,
                            k_stride_s,
                            32 * k_stride_s,
                            k_stride_h,
                            self.sparse_block_size * k_stride_s,
                            k_stride_b,
                        ),
                    ),
                )
            else:
                mK = cute.make_tensor(
                    mK_seq.iterator,
                    cute.make_layout(
                        (
                            self.sparse_block_size,
                            k_dim_half,
                            2,
                            mK_seq.shape[2],
                            cute.ceil_div(mK_seq.shape[0], self.sparse_block_size),
                            mK_seq.shape[3],
                        ),
                        stride=(
                            k_stride_s,
                            k_stride_d,
                            k_dim_half * k_stride_d,
                            k_stride_h,
                            self.sparse_block_size * k_stride_s,
                            k_stride_b,
                        ),
                    ),
                )
            v_stride_s = Int32(mV_seq.layout.stride[0])
            v_stride_d = Int32(mV_seq.layout.stride[1])
            v_stride_h = Int32(mV_seq.layout.stride[2])
            v_stride_b = Int32(mV_seq.layout.stride[3])
            if const_expr(self.is_sage_fp8):
                # FP8 PV uses the native SM100 B-major layout: D is
                # contiguous and a 64-token sparse block is represented as
                # two 32-token MMA-K groups.
                mV = cute.make_tensor(
                    mV_seq.iterator,
                    cute.make_layout(
                        (
                            self.head_dim_v_padded,
                            32,
                            2,
                            mV_seq.shape[2],
                            cute.ceil_div(mV_seq.shape[0], self.sparse_block_size),
                            mV_seq.shape[3],
                        ),
                        stride=(
                            v_stride_d,
                            v_stride_s,
                            32 * v_stride_s,
                            v_stride_h,
                            self.sparse_block_size * v_stride_s,
                            v_stride_b,
                        ),
                    ),
                )
            else:
                mV = cute.make_tensor(
                    mV_seq.iterator,
                    cute.make_layout(
                        (
                            v_dim_part,
                            self.sparse_block_size,
                            2,
                            mV_seq.shape[2],
                            cute.ceil_div(mV_seq.shape[0], self.sparse_block_size),
                            mV_seq.shape[3],
                        ),
                        stride=(
                            v_stride_d,
                            v_stride_s,
                            v_dim_part * v_stride_d,
                            v_stride_h,
                            self.sparse_block_size * v_stride_s,
                            v_stride_b,
                        ),
                    ),
                )

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()
        # Sage V has a different scale for every output channel.  The FP8
        # path therefore uses the register epilogue so that scale can be
        # applied immediately before the BF16 store.
        self.use_tma_O = self.arch >= Arch.sm_90 and not self.is_sage_fp8
        # This can be tuned
        # This is currently very ad-hoc, we should tune it systematically
        self.ex2_emu_freq = 0
        self.ex2_emu_start_frg = 0
        if const_expr(self.enable_ex2_emu):
            self.ex2_emu_freq = 10

        q_major_mode = tcgen05.OperandMajorMode.K
        k_major_mode = tcgen05.OperandMajorMode.K
        v_major_mode = tcgen05.OperandMajorMode.MN
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        # The public tcgen05 tiled-MMA builder does not expose mma.ws. Use this
        # object for CuTe fragment
        # partitioning and issue the WS instruction in ws_qk_gemm below.
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_pv[:2],
            p_source,
        )
        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sQ_tma_layout = (
            cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3),
                0,
                cute.make_layout(
                    (self.head_dim_padded, 32, 2),
                    stride=(1, self.head_dim_padded, 32 * self.head_dim_padded),
                ),
            )
            if const_expr(self.is_sage_fp8)
            else None
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage
        )
        # V dual layout in the MMA-fragment shape expected by CuTe DSL.
        # The K-groups must stay nested as (4,2): the second dim half jumps by
        # 16384 elements, while groups inside one half advance by 1024.
        if const_expr(self.is_sage_fp8):
            sV_layout = sm100_utils_basic.make_smem_layout_b(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.v_dtype,
                self.kv_stage,
            )
        else:
            sV_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3),
                0,
                cute.make_layout(
                    (
                        ((v_dim_part, self.sparse_blocks_per_kv), 16),
                        1,
                        (4, 2),
                        self.kv_stage,
                    ),
                    stride=(
                        ((1, v_dim_part * self.sparse_block_size), v_dim_part),
                        0,
                        (
                            16 * v_dim_part,
                            v_dim_part
                            * self.sparse_block_size
                            * self.sparse_blocks_per_kv,
                        ),
                        self.kv_elems_per_stage,
                    ),
                ),
            )
        # Wide K shared-memory layout:
        #   Sw<3,4,3> o _0 o (_64,_64,_2):(_64,_1,_16384)
        # A sparse sub-block advances by 4096 elements, while the second
        # dim half jumps by the full 256-token half stride.
        if const_expr(self.is_sage_fp8):
            sK_tma_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3),
                0,
                cute.make_layout(
                    (self.head_dim_padded, 32, 2),
                    stride=(1, self.head_dim_padded, 32 * self.head_dim_padded),
                ),
            )
        else:
            sK_tma_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3),
                0,
                cute.make_layout(
                    (self.sparse_block_size, k_dim_half, 2),
                    stride=(k_dim_half, 1, self.n_block_size * k_dim_half),
                ),
            )
        # Wide V shared-memory layout:
        #   Sw<3,4,3> o _0 o (_64,_64,_2):(_1,_64,_4096)
        # It writes both 64-dim halves for one sparse 64-token block while
        # preserving the offsets consumed by the full VDual PV layout.
        if const_expr(self.is_sage_fp8):
            sV_tma_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3),
                0,
                cute.make_layout(
                    (self.head_dim_v_padded, 32, 2),
                    stride=(1, self.head_dim_v_padded, 32 * self.head_dim_v_padded),
                ),
            )
        else:
            sV_tma_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3),
                0,
                cute.make_layout(
                    (v_dim_part, self.sparse_block_size, 2),
                    stride=(
                        1,
                        v_dim_part,
                        v_dim_part * self.sparse_block_size,
                    ),
                ),
            )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.s_stage
        )

        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[3]
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

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        if const_expr(self.is_sage_fp8):
            tma_atom_Q, mQ = cpasync.make_tiled_tma_atom(
                tma_load_op,
                mQ,
                cute.select(sQ_tma_layout, mode=[0, 1, 2]),
                (self.head_dim_padded, 32, 2),
            )
        else:
            tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mQ,
                cute.select(sQ_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
            )

        # K/V are sparse 64-block indexed. Issue four TMAs per
        # 64x256 KV iteration; each copy targets a sub-tile of the full
        # 256x128 WS SMEM layout.
        tma_atom_K, mK = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mK,
            cute.select(sK_tma_layout, mode=[0, 1, 2]),
            (
                (self.head_dim_padded, 32, 2)
                if const_expr(self.is_sage_fp8)
                else (self.sparse_block_size, k_dim_half, 2)
            ),
        )
        tma_atom_V, mV = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mV,
            cute.select(sV_tma_layout, mode=[0, 1, 2]),
            (
                (self.head_dim_v_padded, 32, 2)
                if const_expr(self.is_sage_fp8)
                else (v_dim_part, self.sparse_block_size, 2)
            ),
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
            TileScheduler = BlockSparsePersistentTileScheduler
        elif const_expr(not self.is_persistent):
            TileScheduler = SingleTileScheduler
        else:
            TileScheduler = BlockSparsePersistentTileScheduler
        tile_sched_args = TileSchedulerArguments(
            (
                cute.ceil_div(cute.size(seqlen_q_static), self.cta_tiler[0])
                if const_expr(self.is_sage_fp8)
                else cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0])
            ),
            cute.size(num_q_heads_static)
            if const_expr(self.is_sage_fp8)
            else cute.size(mQ.shape[2]),
            cute.size(batch_size_static)
            if const_expr(self.is_sage_fp8)
            else cute.size(mQ.shape[3]),
            num_splits,
            cute.size(mK_seq.shape[0])
            if const_expr(self.is_sage_fp8)
            else cute.size(mK.shape[0]),
            self.head_dim_padded if const_expr(self.is_sage_fp8) else mQ.shape[1],
            self.head_dim_v_padded,
            total_q=(
                cute.size(seqlen_q_static) * cute.size(batch_size_static)
                if const_expr(self.is_sage_fp8)
                else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3])
            ),
            tile_shape_mn=self.cta_tiler[:2],
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            is_split_kv=self.is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        sO_size = cute.cosize(sO_layout)
        sQ_size = cute.cosize(sQ_layout)
        # K/V storage is reused by the correction exchange and O epilogue.
        # FP8 split-KV writes FP32 partials, so its O scratch is twice as wide
        # as the normal BF16 output and needs a larger backing allocation.
        sO_size_in_k_elements = (
            sO_size * self.o_dtype.width + self.k_dtype.width - 1
        ) // self.k_dtype.width
        sKV_size = max(
            cute.cosize(sK_layout),
            self.epi_sO_input_offset + sO_size_in_k_elements,
        )

        clc_response_size = self.sched_stages * 4 if self.use_clc_scheduler else 0
        clc_mbar_size = self.sched_stages * 2 if self.use_clc_scheduler else 0
        v_scale_cache_size = (
            self.head_dim_v_padded * len(self.correction_warp_ids)
            if self.is_sage_fp8
            else 0
        )
        k_scale_cache_size = self.s_stage * 16 if self.is_sage_fp8 else 0

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
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Per warp-pair reduction barriers for correction exchange.
            reduce_mbar: cute.struct.MemRange[Int64, 2]
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.s_stage * self.stats_stride * 2]
            oStats: cute.struct.MemRange[Float32, 4 * 32 * 2]
            # One 128-wide V-scale cache per correction warp. Separate slices
            # avoid cross-tile overwrite between the two correction warp pairs.
            sVScale: cute.struct.MemRange[Float32, v_scale_cache_size]
            # One 16-value Sage K-scale vector per alternating score stage.
            sKScale: cute.struct.MemRange[Float32, k_scale_cache_size]
            # CLC buffers (mbarriers + response storage)
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[Int32, clc_response_size],
                16,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, sKV_size],
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
            mQScale,
            mKScale,
            mVScale,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            sQ_layout,
            sQ_tma_layout,
            sK_layout,
            sK_tma_layout,
            tP_layout,
            sV_layout,
            sV_tma_layout,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            mBlockIndex,
            mBlockSizes,
            block_sparse_num,
            mBlockNums,
            mSplitOffsets,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b)
        mK: cute.Tensor,  # Rank-5 Int64 or rank-6 sparse-block view
        mV: cute.Tensor,  # Rank-5 Int64 or rank-6 sparse-block view
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mQScale: Optional[cute.Tensor],
        mKScale: Optional[cute.Tensor],
        mVScale: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        sQ_layout: cute.ComposedLayout,
        sQ_tma_layout: Optional[cute.ComposedLayout],
        sK_layout: cute.ComposedLayout,
        sK_tma_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sV_tma_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        mBlockIndex: cute.Tensor,
        mBlockSizes: Optional[cute.Tensor],
        block_sparse_num: Int32,
        mBlockNums: Optional[cute.Tensor],
        mSplitOffsets: Optional[cute.Tensor],
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
        # Tensor memory allocator
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
        )

        ThreadCooperativeGroup = partial(
            pipeline.CooperativeGroup, pipeline.Agent.Thread
        )
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        tma_warp = ThreadCooperativeGroup(1)
        softmax_warps = ThreadCooperativeGroup(len(self.softmax0_warp_ids))
        softmax_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
        )
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        softmax_correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids)
        )
        epilogue_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        )
        pipeline_q = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["Q"],
            defer_sync=True,
        )
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["K"],
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
            consumer_group=softmax_correction_threads,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_warps,
            consumer_group=mma_warp,
            defer_sync=True,
        )
        # MMA warp uses this to signal to the correction warps that O is ready.
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.s_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads,
            defer_sync=True,
        )
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
        reduce_mbar_ptr = storage.reduce_mbar.data_ptr()
        pipeline_o_epi = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_O_epi.data_ptr(),
            num_stages=self.s_stage,
            producer_group=correction_threads,
            consumer_group=epilogue_threads,
            defer_sync=True,
        )

        if warp_idx == self.empty_warp_ids[0]:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(reduce_mbar_ptr + 0, cute.arch.WARP_SIZE * 2)
                cute.arch.mbarrier_init(reduce_mbar_ptr + 1, cute.arch.WARP_SIZE * 2)

        # Fence pipeline barrier initialization before any warp uses it.
        pipeline_init_arrive(is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sKV_ptr = storage.sK.data_ptr()
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(
            cute.recast_ptr(sKV_ptr, sV_layout.inner), sV_layout.outer
        )
        # The epilogue exchange and sO reuse
        # the large KV buffer after the mainloop has finished consuming it.
        sO = cute.make_tensor(
            cute.recast_ptr(
                sKV_ptr + self.epi_sO_input_offset, sO_layout.inner, self.o_dtype
            ),
            sO_layout.outer,
        )

        sScale = storage.sScale.get_tensor(
            cute.make_layout(self.s_stage * self.stats_stride * 2)
        )
        oStats = storage.oStats.get_tensor(cute.make_layout(4 * 32 * 2))
        sVScale = (
            storage.sVScale.get_tensor(cute.make_layout(self.head_dim_v_padded))
            if const_expr(self.is_sage_fp8)
            else sScale
        )
        sKScale = (
            storage.sKScale.get_tensor(cute.make_layout(self.s_stage * 16))
            if const_expr(self.is_sage_fp8)
            else sScale
        )
        oExchange = cute.make_tensor(
            cute.recast_ptr(sKV_ptr, dtype=Float32),
            cute.make_layout(self.epi_exchange_f32_count),
        )

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)

        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO_base = thr_mma_pv.make_fragment_C(pv_acc_shape)
        tOtO = cute.make_tensor(
            tOtO_base.iterator + self.tmem_o_offset[0],
            cute.append(
                tOtO_base.layout,
                cute.make_layout(
                    (self.s_stage,),
                    stride=(self.tmem_o_offset[1] - self.tmem_o_offset[0],),
                ),
            ),
        )
        p_frag_shape = thr_mma_pv.partition_shape_A((self.m_block_size, self.kv_mma_k))
        tOrP_base = thr_mma_pv.make_fragment_A(p_frag_shape)
        # Need to multiply by width ratio bc tP is in v_dtype but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.v_dtype.width
        # Need to adjust the stage stride manually since the two stages aren't contiguous in tmem
        tP_stage_stride = (
            self.tmem_p_offset[1] - self.tmem_p_offset[0]
        ) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP_base.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(
                tOrP_base.layout,
                cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)),
            ),
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=(
                mO.shape[0]
                if const_expr(self.is_sage_fp8)
                else mQ.shape[0]
                if const_expr(not self.pack_gqa)
                else mQ.shape[0][1]
            ),
            seqlen_k_static=(
                mKScale.shape[2] * 16 if const_expr(self.is_sage_fp8) else mK.shape[0]
            ),
        )
        # Split-KV expands O's physical head axis to num_splits * H.  Scale
        # tensors keep the logical H axis, which is what the scheduler and
        # split output addressing need here.
        num_q_heads = (
            cute.size(mQScale.shape[1])
            if const_expr(self.is_sage_fp8)
            else cute.size(mQ.shape[2])
        )
        # Create tile scheduler (and CLC pipeline if enabled)
        if const_expr(self.use_clc_scheduler):
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()

            clc_pipeline_producer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread
            )
            num_clc_consumer_warps = self.threads_per_cta // cute.arch.WARP_SIZE
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

        # Synchronize the CTA before tensor memory allocation.
        pipeline_init_wait()

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY / CLC SCHEDULER WARP
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(self.use_clc_scheduler):
            if warp_idx == self.clc_scheduler_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.clc_scheduler_warp(clc_pipeline, tile_scheduler)
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
                sKV_ptr,
                sQ_tma_layout,
                sK_tma_layout,
                sV_tma_layout,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_q,
                pipeline_kv,
                tile_scheduler,
                mBlockIndex,
                block_sparse_num,
                mBlockNums,
                mSplitOffsets,
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
                sKScale,
                tOtO,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                tile_scheduler,
                mBlockIndex,
                mKScale,
                block_sparse_num,
                mBlockNums,
                mSplitOffsets,
            )
            if const_expr(self.is_sage_fp8):
                # Order the final asynchronous tcgen05 operations before the
                # allocator synchronization/deallocation sequence.
                bsa_fwd_helpers.tcgen05_fence_before_thread_sync()
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.epilogue_warp_ids[0]
            and warp_idx <= self.epilogue_warp_ids[-1]
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.epilogue_s2g(
                mO,
                sO,
                mVScale,
                sVScale,
                gmem_tiled_copy_O,
                tma_atom_O,
                pipeline_o_epi,
                SeqlenInfoCls,
                tile_scheduler,
                num_q_heads,
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
                sScale=sScale,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                tile_scheduler=tile_scheduler,
                mBlockIndex=mBlockIndex,
                mBlockSizes=mBlockSizes,
                mQScale=mQScale,
                mKScale=mKScale,
                sKScale=sKScale,
                block_sparse_num=block_sparse_num,
                mBlockNums=mBlockNums,
                mSplitOffsets=mSplitOffsets,
            )

            # Keep stage constexpr so each softmax path has a fixed stage.
            # Runtime stage selection leaves TMEM offsets, barrier indices, and
            # stats strides as uniform arithmetic in the hot softmax loop.
            if warp_idx < self.softmax1_warp_ids[0]:
                softmax_loop(stage=0)
            if (
                warp_idx < self.correction_warp_ids[0]
                and warp_idx >= self.softmax1_warp_ids[0]
            ):
                softmax_loop(stage=1)

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
                thr_mma_pv,
                tOtO,
                sScale,
                mO,
                mLSE,
                sO,
                pipeline_s_p_o,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                reduce_mbar_ptr,
                pipeline_o_epi,
                gmem_tiled_copy_O,
                tma_atom_O,
                Float32(1.0) if const_expr(self.is_sage_fp8) else softmax_scale_log2,
                oStats,
                oExchange,
                mVScale,
                sVScale,
                SeqlenInfoCls,
                tile_scheduler,
                num_q_heads,
                block_sparse_num,
                mBlockNums,
                mSplitOffsets,
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
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            clc_pipeline.producer_acquire(clc_producer_state)
            mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
            tile_scheduler.advance_to_next_work(mbarrier_addr=mbarrier_addr)
            clc_producer_state.advance()

            work_tile = tile_scheduler.consumer_advance()
        clc_pipeline.producer_tail(clc_producer_state)

    @cute.jit
    def empty_warp(
        self,
        clc_pipeline: cutlass_pipeline.PipelineClcFetchAsync,
        tile_scheduler: TileSchedulerProtocol,
    ):
        """Runs on empty warps (and non-leader CTA scheduler warp) — consumes CLC responses."""
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_tile = tile_scheduler.consumer_advance()

    @cute.jit
    def get_tile_block_count_and_offset(
        self,
        mSplitOffsets: Optional[cute.Tensor],
        mBlockNums: Optional[cute.Tensor],
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        split_idx: Int32,
        block_sparse_num: Int32,
    ) -> Tuple[Int32, Int32]:
        if const_expr(mSplitOffsets is not None):
            split_start = mSplitOffsets[batch_idx, head_idx, m_block, split_idx]
            split_end = mSplitOffsets[batch_idx, head_idx, m_block, split_idx + 1]
            return split_end - split_start, split_start
        if const_expr(self.is_split_kv):
            num_splits = Int32(self.num_splits)
            avg_blocks = block_sparse_num // num_splits
            aligned_base = (avg_blocks // 8) * 8
            split_start = Int32(0)
            split_end = Int32(0)
            if aligned_base > Int32(0):
                remainder = block_sparse_num - aligned_base * num_splits
                split_start = aligned_base * split_idx + cutlass.min(
                    remainder, split_idx * 8
                )
                split_next = split_idx + 1
                split_end = aligned_base * split_next + cutlass.min(
                    remainder, split_next * 8
                )
            else:
                split_start = (
                    block_sparse_num * split_idx + num_splits - 1
                ) // num_splits
                split_end = (
                    block_sparse_num * (split_idx + 1) + num_splits - 1
                ) // num_splits
            return split_end - split_start, split_start
        if const_expr(mBlockNums is not None):
            return mBlockNums[batch_idx, head_idx, m_block], Int32(0)
        return block_sparse_num, Int32(0)

    @cute.jit
    def offset_tile_block_indices(
        self,
        tile_block_indices: cute.Tensor,
        split_offset: Int32,
        mSplitOffsets: Optional[cute.Tensor],
    ) -> cute.Tensor:
        if const_expr(mSplitOffsets is not None or self.is_split_kv):
            return cute.domain_offset((split_offset,), tile_block_indices)
        return tile_block_indices

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
        sKV_ptr: cute.Pointer,
        sQ_tma_layout: Optional[cute.ComposedLayout],
        sK_tma_layout: cute.ComposedLayout,
        sV_tma_layout: cute.ComposedLayout,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        tile_scheduler: TileSchedulerProtocol,
        mBlockIndex: cute.Tensor,
        block_sparse_num: Int32,
        mBlockNums: Optional[cute.Tensor],
        mSplitOffsets: Optional[cute.Tensor],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            if const_expr(self.is_sage_fp8):
                mQ_cur = mQ[None, None, None, head_idx, None, batch_idx]
                gQ_tma = cute.group_modes(mQ_cur, 0, 3)
                tQsQ, tQgQ = cpasync.tma_partition(
                    tma_atom_Q,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(
                        cute.make_tensor(
                            cute.recast_ptr(sQ.iterator, sQ_tma_layout.inner),
                            cute.append(
                                sQ_tma_layout.outer,
                                cute.make_layout(
                                    (self.q_stage,),
                                    stride=(cute.cosize(sQ_tma_layout.outer),),
                                ),
                            ),
                        ),
                        0,
                        3,
                    ),
                    gQ_tma,
                )
            else:
                mQ_cur = mQ[None, None, None, batch_idx][None, None, head_idx]
                gQ = cute.local_tile(mQ_cur, tiler_gQ, (m_block, 0))  # (64, 128)
                gQ = layout_utils.select(
                    cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
                )
                tSgQ = thr_mma_qk.partition_A(gQ)
                load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
                )

            head_idx_kv = (
                head_idx // self.qhead_per_kvhead
                if const_expr(not self.pack_gqa)
                else head_idx
            )
            if const_expr(self.use_int64_kv_strides):
                mK_cur = mK[None, None, None, head_idx_kv, batch_idx]
                mV_cur = mV[None, None, None, head_idx_kv, batch_idx]
                gK_tma = cute.zipped_divide(
                    mK_cur, (self.sparse_block_size, self.head_dim_padded // 2, 2)
                )
                gV_tma = cute.zipped_divide(
                    mV_cur, (self.head_dim_v_padded // 2, self.sparse_block_size, 2)
                )
            else:
                mK_cur = mK[None, None, None, head_idx_kv, None, batch_idx]
                mV_cur = mV[None, None, None, head_idx_kv, None, batch_idx]
                gK_tma = cute.group_modes(mK_cur, 0, 3)
                gV_tma = cute.group_modes(mV_cur, 0, 3)
            tile_block_indices_base = mBlockIndex[batch_idx, head_idx, m_block, None]

            # n_block(i): maps logical index i to actual KV block index via q2k_block_index
            # Group four sparse 64-blocks into one 64x256 KV iteration.
            # raw count is padded to a multiple of 8, then divided by 4 so the
            # number of KV iterations is even.
            # max_i clamps phantom block indices to the last valid entry.
            raw_block_count, split_offset = self.get_tile_block_count_and_offset(
                mSplitOffsets,
                mBlockNums,
                batch_idx,
                head_idx,
                m_block,
                split_idx,
                block_sparse_num,
            )
            tile_block_indices = self.offset_tile_block_indices(
                tile_block_indices_base, split_offset, mSplitOffsets
            )
            process_tile = (
                raw_block_count > Int32(0)
                if const_expr(self.allow_empty_block_nums)
                else True
            )
            block_iter_count = ((raw_block_count + 7) & ~7) // self.sparse_blocks_per_kv
            n_block = partial(
                self.get_tile_n_block_idx,
                tile_block_indices,
                max_i=cutlass.max(raw_block_count - 1, Int32(0)),
            )

            if process_tile:
                # Issue the one-shot Q TMA first, then K[N-1]/K[N-2].
                if (
                    const_expr(len(self.load_warp_ids) == 1)
                    or warp_idx == self.load_warp_ids[0]
                ):
                    pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                    tma_bar_ptr = pipeline_q.sync_object_full.get_barrier(0)
                    if const_expr(self.is_sage_fp8):
                        cute.copy(
                            tma_atom_Q,
                            tQgQ[None, m_block],
                            tQsQ[None, 0],
                            tma_bar_ptr=tma_bar_ptr,
                        )
                    else:
                        load_Q_fn(src_idx=0, dst_idx=0, tma_bar_ptr=tma_bar_ptr)
                q_producer_phase ^= 1

                kv_producer_state = self.load_K_group(
                    tma_atom_K,
                    gK_tma,
                    sK,
                    sKV_ptr,
                    sK_tma_layout,
                    n_block,
                    block_iter_count - 1,
                    kv_producer_state,
                    pipeline_kv,
                )
                # q_stage=1 intra-warp overlap
                # Load order: Q, K[N-1], K[N-2], {V[N-1-i], K[N-3-i]}x(N-2), V[1], V[0]
                block_loop_count = block_iter_count - 2

                # Prologue: K[N-2]
                kv_producer_state = self.load_K_group(
                    tma_atom_K,
                    gK_tma,
                    sK,
                    sKV_ptr,
                    sK_tma_layout,
                    n_block,
                    block_iter_count - 2,
                    kv_producer_state,
                    pipeline_kv,
                )

                # Flat main loop: N-2 iterations, each loads V then K
                for i in cutlass.range(block_loop_count, unroll=1):
                    # V[N-1-i]: V for the block whose S was computed earlier
                    kv_producer_state = self.load_V_group(
                        tma_atom_V,
                        gV_tma,
                        sV,
                        sKV_ptr,
                        sV_tma_layout,
                        n_block,
                        block_iter_count - 1 - i,
                        kv_producer_state,
                        pipeline_kv,
                    )
                    # K[N-3-i]: K for the next QK GEMM
                    kv_producer_state = self.load_K_group(
                        tma_atom_K,
                        gK_tma,
                        sK,
                        sKV_ptr,
                        sK_tma_layout,
                        n_block,
                        block_iter_count - 3 - i,
                        kv_producer_state,
                        pipeline_kv,
                    )

                # Epilogue: last 2 V loads
                kv_producer_state = self.load_V_group(
                    tma_atom_V,
                    gV_tma,
                    sV,
                    sKV_ptr,
                    sV_tma_layout,
                    n_block,
                    1,
                    kv_producer_state,
                    pipeline_kv,
                )
                kv_producer_state = self.load_V_group(
                    tma_atom_V,
                    gV_tma,
                    sV,
                    sKV_ptr,
                    sV_tma_layout,
                    n_block,
                    0,
                    kv_producer_state,
                    pipeline_kv,
                )

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
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sKScale: cute.Tensor,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        tile_scheduler: TileSchedulerProtocol,
        mBlockIndex: cute.Tensor,
        mKScale: Optional[cute.Tensor],
        block_sparse_num: Int32,
        mBlockNums: Optional[cute.Tensor],
        mSplitOffsets: Optional[cute.Tensor],
    ):
        tidx = cute.arch.thread_idx()[0] % cute.arch.WARP_SIZE
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        # q_stage=1: both stages use the same Q (intra-warp overlap across n_blocks)
        tSrQ0 = tSrQ[None, None, None, 0]
        sQ0 = sQ[None, None, None, 0]
        tOrP0 = tOrP[None, None, None, 0]
        tOrP1 = tOrP[None, None, None, 1]

        qk_mma_op = tiled_mma_qk.op
        pv_mma_op = tiled_mma_pv.op

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        q_full_mbar = Int32(pipeline_q.sync_object_full.get_barrier(0).toint())
        spo_empty_mbar0 = Int32(pipeline_s_p_o.sync_object_empty.get_barrier(0).toint())
        spo_empty_mbar1 = Int32(pipeline_s_p_o.sync_object_empty.get_barrier(1).toint())
        # Pipeline s_p_o phases for stage 0 and stage 1.
        # These phases must persist across tiles
        # so that mbarrier phase stays in sync when block_iter_count varies.
        phase_s0 = Int32(0)
        phase_s1 = Int32(0)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            raw_block_count, split_offset = self.get_tile_block_count_and_offset(
                mSplitOffsets,
                mBlockNums,
                batch_idx,
                head_idx,
                m_block,
                split_idx,
                block_sparse_num,
            )
            process_tile = (
                raw_block_count > Int32(0)
                if const_expr(self.allow_empty_block_nums)
                else True
            )
            block_iter_count = ((raw_block_count + 7) & ~7) // self.sparse_blocks_per_kv
            tile_block_indices_base = mBlockIndex[batch_idx, head_idx, m_block, None]
            tile_block_indices = self.offset_tile_block_indices(
                tile_block_indices_base, split_offset, mSplitOffsets
            )
            n_block = partial(
                self.get_tile_n_block_idx,
                tile_block_indices,
                max_i=cutlass.max(raw_block_count - 1, Int32(0)),
            )

            if process_tile:
                # ================================================================
                # q_stage=1: intra-warp overlap across n_block direction
                # Pipeline KV order: K0, K1, V0, K2, V1, K3, V2, ...
                # GEMM order: S0=Q@K0, S1=Q@K1, {O0+=P0@V0, S0=Q@K2}, {O1+=P1@V1, S1=Q@K3}, ...
                # ================================================================

                # Prologue: wait Q0
                bsa_fwd_helpers.mbar_wait(q_full_mbar, mma_q_consumer_phase)
                bsa_fwd_helpers.tcgen05_fence_after_thread_sync()
                mma_q_consumer_phase ^= 1

                # S0 = Q @ K0
                if const_expr(self.is_sage_fp8 and not self.is_split_kv):
                    self.publish_k_scales(
                        mKScale,
                        sKScale,
                        n_block,
                        block_iter_count - 1,
                        0,
                        batch_idx,
                        head_idx,
                        tidx,
                    )
                bsa_fwd_helpers.mbar_wait(
                    Int32(
                        pipeline_kv.sync_object_full.get_barrier(
                            mma_kv_consumer_state.index
                        ).toint()
                    ),
                    mma_kv_consumer_state.phase,
                )
                bsa_fwd_helpers.tcgen05_fence_after_thread_sync()
                Ki_index = mma_kv_consumer_state.index
                sK_cur = sK[None, None, None, Ki_index]
                self.ws_qk_gemm(
                    qk_mma_op, 0, tSrQ0, tSrK[None, None, None, Ki_index], sQ0, sK_cur
                )
                pipeline_s_p_o.producer_commit_w_index(0)  # signal S0 ready
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

                # S1 = Q @ K1
                if const_expr(self.is_sage_fp8 and not self.is_split_kv):
                    self.publish_k_scales(
                        mKScale,
                        sKScale,
                        n_block,
                        block_iter_count - 2,
                        1,
                        batch_idx,
                        head_idx,
                        tidx,
                    )
                bsa_fwd_helpers.mbar_wait(
                    Int32(
                        pipeline_kv.sync_object_full.get_barrier(
                            mma_kv_consumer_state.index
                        ).toint()
                    ),
                    mma_kv_consumer_state.phase,
                )
                bsa_fwd_helpers.tcgen05_fence_after_thread_sync()
                Ki_index = mma_kv_consumer_state.index
                sK_cur = sK[None, None, None, Ki_index]
                self.ws_qk_gemm(
                    qk_mma_op, 1, tSrQ0, tSrK[None, None, None, Ki_index], sQ0, sK_cur
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
                Ki_index = mma_kv_consumer_state.index
                mma_kv_release_state = mma_kv_consumer_state.clone()
                tOrVi = tOrV[None, None, None, Vi_index]
                sV_cur = sV[None, None, None, Vi_index]
                sK_cur = sK[None, None, None, Ki_index]

                pair_count = (
                    block_loop_count // 2
                )  # N even => block_loop_count = N-2 is even
                for i in cutlass.range(pair_count, unroll=1):
                    for stage in cutlass.range_constexpr(self.s_stage):
                        if const_expr(stage == 0):
                            phase_cur, O_acc_cur = phase_s0, O_acc_s0
                            spo_empty_mbar = spo_empty_mbar0
                            tOrP_stage = tOrP0
                        else:
                            phase_cur, O_acc_cur = phase_s1, O_acc_s1
                            spo_empty_mbar = spo_empty_mbar1
                            tOrP_stage = tOrP1
                        bsa_fwd_helpers.mbar_wait(spo_empty_mbar, phase_cur)
                        if const_expr(self.is_sage_fp8 and not self.is_split_kv):
                            self.publish_k_scales(
                                mKScale,
                                sKScale,
                                n_block,
                                block_iter_count - 3 - (i * self.s_stage + stage),
                                stage,
                                batch_idx,
                                head_idx,
                                tidx,
                            )
                        # Wait V
                        bsa_fwd_helpers.mbar_wait(
                            Int32(
                                pipeline_kv.sync_object_full.get_barrier(
                                    mma_kv_consumer_state.index
                                ).toint()
                            ),
                            mma_kv_consumer_state.phase,
                        )
                        bsa_fwd_helpers.tcgen05_fence_after_thread_sync()
                        mma_kv_release_state = mma_kv_consumer_state.clone()
                        Vi_index = mma_kv_consumer_state.index
                        tOrVi = tOrV[None, None, None, Vi_index]
                        sV_cur = sV[None, None, None, Vi_index]
                        mma_kv_consumer_state.advance()
                        self.ws_pv_gemm(
                            pv_mma_op,
                            stage,
                            tOrP_stage,
                            tOrVi,
                            sV_cur,
                            not O_acc_cur,
                            phase_cur,
                            pipeline_p_lastsplit,
                        )
                        pipeline_kv.consumer_release(mma_kv_release_state)
                        # Overlap the independent K wait with the preceding PV issue.
                        bsa_fwd_helpers.mbar_wait(
                            Int32(
                                pipeline_kv.sync_object_full.get_barrier(
                                    mma_kv_consumer_state.index
                                ).toint()
                            ),
                            mma_kv_consumer_state.phase,
                        )
                        bsa_fwd_helpers.tcgen05_fence_after_thread_sync()
                        Ki_index = mma_kv_consumer_state.index
                        sK_cur = sK[None, None, None, Ki_index]
                        self.ws_qk_gemm(
                            qk_mma_op,
                            stage,
                            tSrQ0,
                            tSrK[None, None, None, Ki_index],
                            sQ0,
                            sK_cur,
                        )
                        pipeline_s_p_o.producer_commit_w_index(stage)
                        if const_expr(stage == 0):
                            phase_s0 ^= 1
                            O_acc_s0 = True
                        else:
                            phase_s1 ^= 1
                            O_acc_s1 = True
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()

                # release Q0
                pipeline_q.consumer_release_w_index(0)

                # Epilogue: 2 PV GEMMs (N even: stage 0 first, stage 1 second)
                for epi_stage_constexpr in cutlass.range_constexpr(2):
                    if const_expr(epi_stage_constexpr == 0):
                        epi_s, epi_p, epi_zi = 0, phase_s0, not O_acc_s0
                        spo_empty_mbar = spo_empty_mbar0
                        tOrP_epi = tOrP0
                    else:
                        epi_s, epi_p, epi_zi = 1, phase_s1, not O_acc_s1
                        spo_empty_mbar = spo_empty_mbar1
                        tOrP_epi = tOrP1
                    bsa_fwd_helpers.mbar_wait(spo_empty_mbar, epi_p)
                    bsa_fwd_helpers.mbar_wait(
                        Int32(
                            pipeline_kv.sync_object_full.get_barrier(
                                mma_kv_consumer_state.index
                            ).toint()
                        ),
                        mma_kv_consumer_state.phase,
                    )
                    bsa_fwd_helpers.tcgen05_fence_after_thread_sync()
                    Vi_index = mma_kv_consumer_state.index
                    tOrVi = tOrV[None, None, None, Vi_index]
                    sV_cur = sV[None, None, None, Vi_index]
                    self.ws_pv_gemm(
                        pv_mma_op,
                        epi_stage_constexpr,
                        tOrP_epi,
                        tOrVi,
                        sV_cur,
                        epi_zi,
                        epi_p,
                        pipeline_p_lastsplit,
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

    @cute.jit
    def publish_k_scales(
        self,
        mKScale: cute.Tensor,
        sKScale: cute.Tensor,
        n_block: Callable,
        kv_block_idx: Int32,
        score_stage: int,
        batch_idx: Int32,
        head_idx: Int32,
        tidx: Int32,
    ):
        """Load one KV tile's 16 K scales once for all softmax rows."""
        if tidx < 16:
            logical_sub = tidx // 4
            scale_group = tidx % 4
            key_block = n_block(kv_block_idx * self.sparse_blocks_per_kv + logical_sub)
            sKScale[score_stage * 16 + tidx] = mKScale[
                batch_idx,
                head_idx,
                key_block * 4 + scale_group,
            ]
        cute.arch.sync_warp()
        cute.arch.fence_view_async_shared()

    @cute.jit
    def ws_pv_gemm(
        self,
        pv_mma_op: cute.nvgpu.tcgen05.mma.MmaOp,
        stage: int,
        tCrP: cute.Tensor,
        tCrV: cute.Tensor,
        sV: cute.Tensor,
        zero_init: bool | Boolean,
        phase: Int32,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
    ) -> None:
        bsa_fwd_helpers.gemm_ptx_partial(
            pv_mma_op,
            self.tmem_o_offset[stage],
            tCrP,
            tCrV,
            sA=None,
            sB=sV,
            mbar_ptr=(
                pipeline_p_lastsplit.sync_object_full.get_barrier(stage)
                if const_expr(self.split_P_arrive > 0)
                else None
            ),
            mbar_phase=phase,
            split_arrive=(
                self.split_P_arrive if const_expr(self.split_P_arrive > 0) else None
            ),
            zero_init=zero_init,
            tA_addr=Int32(self.tmem_p_offset[stage]),
        )

    @cute.jit
    def ws_qk_gemm(
        self,
        qk_mma_op: cute.nvgpu.tcgen05.mma.MmaOp,
        stage: int,
        tCrQ: cute.Tensor,
        tCrK: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
    ) -> None:
        bsa_fwd_helpers.gemm_ptx_partial(
            qk_mma_op,
            self.tmem_s_offset[stage],
            tCrQ,
            tCrK,
            sA=sQ,
            sB=sK,
            zero_init=True,
        )

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        tile_scheduler: TileSchedulerProtocol,
        mBlockIndex: cute.Tensor,
        mBlockSizes: Optional[cute.Tensor],
        mQScale: Optional[cute.Tensor],
        mKScale: Optional[cute.Tensor],
        sKScale: cute.Tensor,
        block_sparse_num: Int32,
        mBlockNums: Optional[cute.Tensor],
        mSplitOffsets: Optional[cute.Tensor],
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

        mma_si_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)

        tmem_s_addr = Int32(stage) * Int32(
            self.tmem_s_offset[1] - self.tmem_s_offset[0]
        )
        tmem_p_addr = Int32(stage) * Int32(
            self.tmem_p_offset[1] - self.tmem_p_offset[0]
        )
        sm_stats_bar_index = stage * 4 + warp_idx

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            tile_block_indices_base = mBlockIndex[batch_idx, head_idx, m_block, None]

            # n_block(i): maps logical index i to actual KV block index via q2k_block_index
            raw_block_count, split_offset = self.get_tile_block_count_and_offset(
                mSplitOffsets,
                mBlockNums,
                batch_idx,
                head_idx,
                m_block,
                split_idx,
                block_sparse_num,
            )
            tile_block_indices = self.offset_tile_block_indices(
                tile_block_indices_base, split_offset, mSplitOffsets
            )
            has_work = (
                raw_block_count > Int32(0)
                if const_expr(self.allow_empty_block_nums)
                else True
            )
            n_block = partial(
                self.get_tile_n_block_idx,
                tile_block_indices,
                max_i=cutlass.max(raw_block_count - 1, Int32(0)),
            )

            softmax = SoftmaxSm100.create(
                Float32(1.0) if const_expr(self.is_sage_fp8) else softmax_scale_log2,
                rescale_threshold=(
                    SAGE_P_RESCALE_THRESHOLD
                    if const_expr(self.is_sage_fp8)
                    else 8.0
                    if const_expr(self.q_dtype.width == 16)
                    else 0.0
                ),
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                sScale=sScale,
                stage=stage,
                mKScale=mKScale,
                sKScale=sKScale,
                batch_idx=batch_idx,
                head_idx=head_idx,
                score_scale_log2=softmax_scale_log2,
            )

            if const_expr(self.is_sage_fp8):
                q_row = (
                    m_block * self.m_block_size
                    + (warp_idx & 1) * cute.arch.WARP_SIZE
                    + (tidx % cute.arch.WARP_SIZE)
                )
                q_scale = (
                    mQScale[batch_idx, head_idx, q_row]
                    if q_row < mQScale.shape[2]
                    else Float32(1.0)
                )
            else:
                q_scale = Float32(1.0)

            # Always acquire pipeline_sm_stats to stay in sync with correction
            pipeline_sm_stats.producer_acquire_w_index_phase(
                stage, sm_stats_producer_phase
            )
            sm_stats_producer_phase ^= 1

            if has_work:
                # block_iter_count is the padded number of 4-sparse-block KV groups.
                # WG0 (stage=0): kv groups N-1, N-3, ...
                # WG1 (stage=1): kv groups N-2, N-4, ...
                block_iter_count = (
                    (raw_block_count + 7) & ~7
                ) // self.sparse_blocks_per_kv
                wg_count = block_iter_count // 2
                warp_col = warp_idx // 2

                first_bs_lo, first_bs_hi, first_nb_lo, first_nb_hi = (
                    self.get_softmax_block_info(
                        Int32(0),
                        stage,
                        warp_col,
                        block_iter_count,
                        raw_block_count,
                        n_block,
                        mBlockSizes,
                    )
                )
                mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                    mma_si_consumer_phase,
                    sm_stats_producer_phase,
                    block_size_lo=first_bs_lo,
                    block_size_hi=first_bs_hi,
                    n_block_lo=first_nb_lo,
                    n_block_hi=first_nb_hi,
                    q_scale=q_scale,
                    tidx=tidx,
                    tmem_s_addr=tmem_s_addr,
                    tmem_p_addr=tmem_p_addr,
                    sm_stats_bar_index=sm_stats_bar_index,
                    is_first=True,
                )
                for n_tile in cutlass.range(wg_count - 1, unroll=1):
                    bs_lo, bs_hi, nb_lo, nb_hi = self.get_softmax_block_info(
                        n_tile + 1,
                        stage,
                        warp_col,
                        block_iter_count,
                        raw_block_count,
                        n_block,
                        mBlockSizes,
                    )
                    mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        block_size_lo=bs_lo,
                        block_size_hi=bs_hi,
                        n_block_lo=nb_lo,
                        n_block_hi=nb_hi,
                        q_scale=q_scale,
                        tidx=tidx,
                        tmem_s_addr=tmem_s_addr,
                        tmem_p_addr=tmem_p_addr,
                        sm_stats_bar_index=sm_stats_bar_index,
                    )

                sScale[tidx + stage * self.stats_stride] = softmax.row_sum[0]
                if const_expr(mLSE is not None or self.q_stage == 1):
                    sScale[
                        tidx
                        + stage * self.stats_stride
                        + self.s_stage * self.stats_stride
                    ] = softmax.row_max[0]
                sm_stats_barrier.arrive_w_index(index=sm_stats_bar_index)
            else:
                # Empty tile: arrive barrier once (synthetic "no work" signal for correction)
                sm_stats_barrier.arrive_w_index(index=sm_stats_bar_index)

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_sm_stats.producer_tail
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)

    @cute.jit
    def get_softmax_block_info(
        self,
        kv_iter: Int32,
        stage: int | Int32,
        warp_col: Int32,
        block_iter_count: Int32,
        raw_block_count: Int32,
        n_block: Callable,
        mBlockSizes: Optional[cute.Tensor],
    ) -> Tuple[Int32, Int32, Int32, Int32]:
        kv_block = block_iter_count - 1 - (self.s_stage * kv_iter + Int32(stage))
        logical_lo = kv_block * self.sparse_blocks_per_kv + warp_col
        logical_hi = logical_lo + 2
        if const_expr(self.has_block_sizes):
            bs_lo = (
                Int32(0)
                if logical_lo >= raw_block_count
                else mBlockSizes[n_block(logical_lo)]
            )
            bs_hi = (
                Int32(0)
                if logical_hi >= raw_block_count
                else mBlockSizes[n_block(logical_hi)]
            )
        else:
            bs_lo = (
                Int32(0)
                if logical_lo >= raw_block_count
                else Int32(self.sparse_block_size)
            )
            bs_hi = (
                Int32(0)
                if logical_hi >= raw_block_count
                else Int32(self.sparse_block_size)
            )
        # Invalid padded entries are masked to zero.  Clamp their lookup so no
        # out-of-range global-memory access is generated in the FP8 scale path.
        nb_lo = n_block(
            cutlass.min(logical_lo, cutlass.max(raw_block_count - 1, Int32(0)))
        )
        nb_hi = n_block(
            cutlass.min(logical_hi, cutlass.max(raw_block_count - 1, Int32(0)))
        )
        return bs_lo, bs_hi, nb_lo, nb_hi

    @cute.jit
    def get_tile_n_block_idx(
        self,
        tile_block_indices: cute.Tensor,
        i: Int32,
        max_i: Optional[Int32] = None,
    ) -> Int32:
        idx = cutlass.min(i, max_i) if const_expr(max_i is not None) else i
        return tile_block_indices[idx]

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        softmax: SoftmaxSm100,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        sScale: cute.Tensor,
        stage: int | Int32,
        block_size_lo: Int32,
        block_size_hi: Int32,
        n_block_lo: Int32,
        n_block_hi: Int32,
        q_scale: Float32,
        mKScale: Optional[cute.Tensor],
        sKScale: cute.Tensor,
        batch_idx: Int32,
        head_idx: Int32,
        score_scale_log2: Float32,
        tidx: Int32,
        tmem_s_addr: Int32,
        tmem_p_addr: Int32,
        sm_stats_bar_index: Int32,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32]:
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
        if const_expr(self.is_sage_fp8 and not self.is_split_kv):
            qk_scales = cute.make_rmem_tensor((8,), Float32)
            q_log2_scale = q_scale * score_scale_log2

        # Wait for Si
        bsa_fwd_helpers.mbar_wait(
            Int32(pipeline_s_p_o.sync_object_full.get_barrier(stage).toint()),
            mma_si_consumer_phase,
        )

        if const_expr(self.is_sage_fp8 and not self.is_split_kv):
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
            warp_col = warp_idx // 2
            for block_idx in cutlass.range_constexpr(2):
                for key_scale_group in cutlass.range_constexpr(4):
                    scale_idx = block_idx * 4 + key_scale_group
                    cache_idx = (
                        stage * 16 + (warp_col + block_idx * 2) * 4 + key_scale_group
                    )
                    qk_scales[scale_idx] = q_log2_scale * sKScale[cache_idx]

        tSrS_t2r = cute.make_rmem_tensor((128,), Float32)
        for c in cutlass.range_constexpr(4):
            vals = bsa_fwd_helpers.tmem_load_32dp32b32x(tmem_s_addr + c * 32)
            for j in cutlass.range_constexpr(32):
                tSrS_t2r[c * 32 + j] = vals[j]

        tSrS_blocks = cute.logical_divide(
            tSrS_t2r, cute.make_layout(self.sparse_block_size)
        )
        bsa_fwd_helpers.apply_block_size_mask_64(tSrS_blocks[None, 0], block_size_lo)
        bsa_fwd_helpers.apply_block_size_mask_64(tSrS_blocks[None, 1], block_size_hi)

        if const_expr(self.is_sage_fp8):
            # Q has one scale per row.  K has one scale per 16 consecutive
            # tokens, hence four scale values inside each logical 64-token KV
            # block.  Max-reduce the raw scores within each group first, then
            # scale only the eight group maxima.  The score scale is fused into
            # the later subtract-row-max FMA, avoiding a full extra score pass.
            scaled_group_max = cute.make_rmem_tensor((8,), Float32)
            tSrS_groups = cute.logical_divide(tSrS_t2r, cute.make_layout(16))
            if const_expr(not self.is_split_kv):
                for scale_idx in cutlass.range_constexpr(8):
                    group_max = softmax._compute_row_max(
                        tSrS_groups[None, scale_idx].load()
                    )
                    scaled_group_max[scale_idx] = group_max * qk_scales[scale_idx]
            else:
                qk_scales = cute.make_rmem_tensor((8,), Float32)
                q_log2_scale = q_scale * score_scale_log2
                for block_idx in cutlass.range_constexpr(2):
                    key_block = n_block_lo if const_expr(block_idx == 0) else n_block_hi
                    for key_scale_group in cutlass.range_constexpr(4):
                        scale_idx = block_idx * 4 + key_scale_group
                        qk_scale = (
                            q_log2_scale
                            * mKScale[
                                batch_idx,
                                head_idx,
                                key_block * 4 + key_scale_group,
                            ]
                        )
                        qk_scales[scale_idx] = qk_scale
                        group_max = softmax._compute_row_max(
                            tSrS_groups[None, scale_idx].load()
                        )
                        scaled_group_max[scale_idx] = group_max * qk_scale
            row_max, acc_scale = softmax.update_row_max(
                scaled_group_max.load(), is_first
            )
        else:
            row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            sScale[tidx + stage * self.stats_stride] = acc_scale
        # Notify correction wg that row_max is ready
        sm_stats_barrier.arrive_w_index(index=sm_stats_bar_index)

        if const_expr(self.is_sage_fp8):
            # Generate P * 256 directly in exp2.  The softmax numerator and
            # denominator carry the same fixed factor, so normalization is
            # unchanged while the separate per-element multiply disappears.
            p_log2_scale = Float32(SAGE_P_QUANT_LOG2_SCALE)
            for scale_idx in cutlass.range_constexpr(8):
                score_scale = qk_scales[scale_idx]
                for token_idx in cutlass.range_constexpr(0, 16, 2):
                    j = scale_idx * 16 + token_idx
                    tSrS_t2r[j], tSrS_t2r[j + 1] = cute.arch.fma_packed_f32x2(
                        (tSrS_t2r[j], tSrS_t2r[j + 1]),
                        (score_scale, score_scale),
                        (
                            -row_max + p_log2_scale,
                            -row_max + p_log2_scale,
                        ),
                    )
        else:
            softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        tSrS_frg = cute.logical_divide(tSrS_t2r, cute.make_layout(32))
        if const_expr(not is_first):
            sum0 = (softmax.row_sum[0] * acc_scale, 0.0)
        else:
            sum0 = (0.0, 0.0)
        sum1 = (0.0, 0.0)
        sum2 = (0.0, 0.0)
        sum3 = (0.0, 0.0)
        for i in cutlass.range_constexpr(4):
            p_frag = cute.make_rmem_tensor(
                (8 if const_expr(self.is_sage_fp8) else 16,), Int32
            )
            for j in cutlass.range_constexpr(0, 32, 8):
                e0 = cute.math.exp2(tSrS_frg[j + 0, i], fastmath=True)
                e1 = cute.math.exp2(tSrS_frg[j + 1, i], fastmath=True)
                tSrS_frg[j + 0, i] = e0
                tSrS_frg[j + 1, i] = e1

                e2 = cute.math.exp2(tSrS_frg[j + 2, i], fastmath=True)
                e3 = cute.math.exp2(tSrS_frg[j + 3, i], fastmath=True)
                tSrS_frg[j + 2, i] = e2
                tSrS_frg[j + 3, i] = e3

                e4 = cute.math.exp2(tSrS_frg[j + 4, i], fastmath=True)
                e5 = cute.math.exp2(tSrS_frg[j + 5, i], fastmath=True)
                tSrS_frg[j + 4, i] = e4
                tSrS_frg[j + 5, i] = e5

                e6 = cute.math.exp2(tSrS_frg[j + 6, i], fastmath=True)
                e7 = cute.math.exp2(tSrS_frg[j + 7, i], fastmath=True)
                tSrS_frg[j + 6, i] = e6
                tSrS_frg[j + 7, i] = e7
                if const_expr(self.is_sage_fp8):
                    p_frag[j // 4 + 0] = bsa_fwd_helpers.cvt_f32x4_to_e4m3x4(
                        e0,
                        e1,
                        e2,
                        e3,
                    )
                    p_frag[j // 4 + 1] = bsa_fwd_helpers.cvt_f32x4_to_e4m3x4(
                        e4,
                        e5,
                        e6,
                        e7,
                    )
                else:
                    p_frag[j // 2 + 0] = bsa_fwd_helpers.cvt_f32x2_to_bf16x2(e0, e1)
                    p_frag[j // 2 + 1] = bsa_fwd_helpers.cvt_f32x2_to_bf16x2(e2, e3)
                    p_frag[j // 2 + 2] = bsa_fwd_helpers.cvt_f32x2_to_bf16x2(e4, e5)
                    p_frag[j // 2 + 3] = bsa_fwd_helpers.cvt_f32x2_to_bf16x2(e6, e7)
            if const_expr(self.is_sage_fp8):
                bsa_fwd_helpers.tmem_store_e4m3x8(tmem_p_addr + i * 8, p_frag)
            else:
                bsa_fwd_helpers.tmem_store_bf16x16(tmem_p_addr + i * 16, p_frag)
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = self.split_P_arrive // 32
                if const_expr(i + 1 == split_P_arrive_idx):
                    # Notify mma warp that the 1st half of P is ready
                    cute.arch.fence_view_async_tmem_store()
                    if const_expr(self.is_sage_fp8):
                        bsa_fwd_helpers.tcgen05_fence_before_thread_sync()
                    pipeline_s_p_o.consumer_release_w_index(stage)
            for j in cutlass.range_constexpr(0, 32, 8):
                sum0 = cute.arch.add_packed_f32x2(
                    sum0, (tSrS_frg[j + 0, i], tSrS_frg[j + 1, i])
                )
                sum1 = cute.arch.add_packed_f32x2(
                    sum1, (tSrS_frg[j + 2, i], tSrS_frg[j + 3, i])
                )
                sum2 = cute.arch.add_packed_f32x2(
                    sum2, (tSrS_frg[j + 4, i], tSrS_frg[j + 5, i])
                )
                sum3 = cute.arch.add_packed_f32x2(
                    sum3, (tSrS_frg[j + 6, i], tSrS_frg[j + 7, i])
                )
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.is_sage_fp8):
            bsa_fwd_helpers.tcgen05_fence_before_thread_sync()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        sum0 = cute.arch.add_packed_f32x2(sum0, sum1)
        sum2 = cute.arch.add_packed_f32x2(sum2, sum3)
        sum0 = cute.arch.add_packed_f32x2(sum0, sum2)
        softmax.row_sum[0] = sum0[0] + sum0[1]
        bsa_fwd_helpers.mbar_wait(
            Int32(pipeline_sm_stats.sync_object_empty.get_barrier(stage).toint()),
            sm_stats_producer_phase,
        )
        return mma_si_consumer_phase ^ 1, sm_stats_producer_phase ^ 1

    @cute.jit
    def correction_loop(
        self,
        thr_mma_pv: cute.ThrMma,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        reduce_mbar_ptr: cute.Pointer,
        pipeline_o_epi: pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        oStats: cute.Tensor,
        oExchange: cute.Tensor,
        mVScale: Optional[cute.Tensor],
        sVScale: cute.Tensor,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        num_heads: Int32,
        block_sparse_num: Int32,
        mBlockNums: Optional[cute.Tensor],
        mSplitOffsets: Optional[cute.Tensor],
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        reduce_mbar_addr = Int32((reduce_mbar_ptr + (warp_idx & 1)).toint())

        # First iter: no correction is required
        # Notify mma warp that O has been rescaled
        for stage in cutlass.range_constexpr(self.s_stage):
            pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)
        tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            out_head_idx = (
                head_idx + split_idx * num_heads
                if const_expr(self.is_split_kv)
                else head_idx
            )

            if const_expr(self.is_sage_fp8 and not self.is_split_kv):
                lane_idx = tidx % cute.arch.WARP_SIZE
                for vec in cutlass.range_constexpr(
                    self.head_dim_v_padded // cute.arch.WARP_SIZE
                ):
                    col = lane_idx + vec * cute.arch.WARP_SIZE
                    sVScale[warp_idx * self.head_dim_v_padded + col] = mVScale[
                        head_idx, col
                    ]
                cute.arch.sync_warp()

            mO_cur = mO[None, None, None, batch_idx][None, None, out_head_idx]
            # For q_stage=1, gO tiles span 1*m_block_size rows (single Q, combine writes one O)
            gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
            gO = layout_utils.select(
                cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
            )
            gO = cute.flat_divide(gO, (self.mma_tiler_pv[0],))[None, 0, None, None]

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

            raw_block_count, _ = self.get_tile_block_count_and_offset(
                mSplitOffsets,
                mBlockNums,
                batch_idx,
                head_idx,
                m_block,
                split_idx,
                block_sparse_num,
            )
            has_work = (
                raw_block_count > Int32(0)
                if const_expr(self.allow_empty_block_nums)
                else True
            )

            if has_work:
                # Ignore first signal from softmax as no correction is required
                sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                pipeline_sm_stats.consumer_release_w_index(0)
                # Wait for both softmax warp groups.
                sm_stats_barrier.arrive_and_wait_w_index(index=1 * 4 + warp_idx)
                sm_stats_consumer_phase ^= 1

                # q_stage=1 correction loop
                block_iter_count = (
                    (raw_block_count + 7) & ~7
                ) // self.sparse_blocks_per_kv
                corr_pair_count = (block_iter_count - 2) // 2
                # Rescale the two alternating stages as a pair.
                for _i in cutlass.range(corr_pair_count, unroll=1):
                    for stage in cutlass.range_constexpr(self.s_stage):
                        sm_stats_barrier.arrive_and_wait_w_index(
                            index=stage * 4 + warp_idx
                        )
                        scale = sScale[tidx + stage * self.stats_stride]
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

                # Read final softmax stats for both stages
                for stage in cutlass.range_constexpr(self.s_stage):
                    sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                    row_sum = sScale[tidx + stage * self.stats_stride]
                    if const_expr(mLSE is not None or self.q_stage == 1):
                        row_max = sScale[
                            tidx
                            + stage * self.stats_stride
                            + self.s_stage * self.stats_stride
                        ]
                    else:
                        row_max = None
                    pipeline_sm_stats.consumer_release_w_index(stage)
                    row_is_valid = row_sum > Float32(0.0)
                    stats[stage] = (row_sum, row_max, row_is_valid)

                # q_stage=1: combine O0 and O1, then write single output
                row_sum0, row_max0, valid0 = stats[0]
                row_sum1, row_max1, valid1 = stats[1]

                # Compute combined scales for the two partial O accumulators
                # row_max is in original S space (unscaled). To compute rescale
                # factors we need exp2((row_max - max_combined) * scale_log2).
                # For empty/padding stages, row_max may be 0.0 instead of -inf
                # due to softmax's safe_max clamping. Use -inf for those to avoid polluting max_combined.
                rm0 = row_max0 if valid0 else -Float32.inf
                rm1 = row_max1 if valid1 else -Float32.inf
                max_combined = cutlass.max(rm0, rm1)
                max_safe = max_combined if max_combined > -Float32.inf else Float32(0.0)
                scale0 = (
                    cute.math.exp2((rm0 - max_safe) * softmax_scale_log2, fastmath=True)
                    if valid0
                    else Float32(0.0)
                )
                scale1 = (
                    cute.math.exp2((rm1 - max_safe) * softmax_scale_log2, fastmath=True)
                    if valid1
                    else Float32(0.0)
                )
                sum_combined = row_sum0 * scale0 + row_sum1 * scale1
                my_sum = sum_combined
                my_max = max_safe

                # Wait for both O accumulators from MMA warp
                for stage in cutlass.range_constexpr(self.s_stage):
                    bsa_fwd_helpers.mbar_wait(
                        Int32(
                            pipeline_o_acc.sync_object_full.get_barrier(stage).toint()
                        ),
                        o_corr_consumer_phase,
                    )
                    bsa_fwd_helpers.tcgen05_fence_after_thread_sync()
                pipeline_o_epi.producer_acquire_w_index_phase(
                    0, corr_epi_producer_phase
                )
                mLSE_cur = (
                    mLSE[None, out_head_idx, batch_idx]
                    if const_expr(mLSE is not None)
                    else None
                )
                self.correction_epilogue_combine_ws_raw(
                    tOtO[None, None, None, 0].iterator.toint(),
                    tOtO[None, None, None, 1].iterator.toint(),
                    tidx,
                    m_block,
                    seqlen.seqlen_q,
                    softmax_scale_log2,
                    scale0,
                    scale1,
                    my_sum,
                    my_max,
                    sO[None, None, 0],
                    oStats,
                    oExchange,
                    sVScale,
                    warp_idx * self.head_dim_v_padded,
                    reduce_mbar_addr,
                    mLSE_cur,
                )
                # Release both O buffers in tmem
                for stage in cutlass.range_constexpr(self.s_stage):
                    pipeline_s_p_o.consumer_release_w_index(stage)
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
                # Write O=0 through the raw WS exchange path without reading stale TMEM.
                pipeline_o_epi.producer_acquire_w_index_phase(
                    0, corr_epi_producer_phase
                )
                mLSE_cur = (
                    mLSE[None, out_head_idx, batch_idx]
                    if const_expr(mLSE is not None)
                    else None
                )
                self.correction_epilogue_combine_ws_raw(
                    tOtO[None, None, None, 0].iterator.toint(),
                    tOtO[None, None, None, 1].iterator.toint(),
                    tidx,
                    m_block,
                    seqlen.seqlen_q,
                    softmax_scale_log2,
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    sO[None, None, 0],
                    oStats,
                    oExchange,
                    sVScale,
                    warp_idx * self.head_dim_v_padded,
                    reduce_mbar_addr,
                    mLSE_cur,
                )
                # Do NOT release pipeline_s_p_o (MMA didn't commit)
                pipeline_o_epi.producer_commit_w_index(0)
                # o_corr_consumer_phase NOT toggled (pipeline_o_acc not touched)
                corr_epi_producer_phase ^= 1

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_o_epi.consumer_tail() for the correction warps
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
        assert self.head_dim_v_padded == 128
        bsa_fwd_helpers.tmem_rescale_4x32dp32b32x(Int32(tOtO.iterator.toint()), scale)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue_combine_ws_raw(
        self,
        tmem_o0_addr: Int32,
        tmem_o1_addr: Int32,
        tidx: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        softmax_scale_log2: Float32,
        scale0: Float32,
        scale1: Float32,
        my_sum: Float32,
        my_max: Float32,
        sO: cute.Tensor,
        oStats: cute.Tensor,
        oExchange: cute.Tensor,
        sVScale: cute.Tensor,
        v_scale_warp_base: Int32,
        reduce_mbar_addr: Int32,
        mLSE_cur: Optional[cute.Tensor] = None,
    ):
        """Combine the WS epilogue using raw TMEM/SMEM addressing."""
        corr_warp = tidx // cute.arch.WARP_SIZE
        lane_idx = tidx % cute.arch.WARP_SIZE
        partner_warp = corr_warp ^ 2

        # Exchange the two warp-pair row stats: (0,2) and (1,3).
        oStats[(partner_warp * 64) + lane_idx * 2 + 0] = my_sum
        oStats[(partner_warp * 64) + lane_idx * 2 + 1] = my_max
        bsa_fwd_helpers.mbar_arrive_and_wait(reduce_mbar_addr, Int32(0))

        partner_sum = oStats[(corr_warp * 64) + lane_idx * 2 + 0]
        partner_max = oStats[(corr_warp * 64) + lane_idx * 2 + 1]
        max_total = cutlass.max(my_max, partner_max)
        max_total_safe = max_total if max_total > -Float32.inf else Float32(0.0)
        my_rescale = (
            cute.math.exp2(
                (my_max - max_total_safe) * softmax_scale_log2, fastmath=True
            )
            if my_sum > Float32(0.0)
            else Float32(0.0)
        )
        partner_rescale = (
            cute.math.exp2(
                (partner_max - max_total_safe) * softmax_scale_log2, fastmath=True
            )
            if partner_sum > Float32(0.0)
            else Float32(0.0)
        )
        sum_total = my_sum * my_rescale + partner_sum * partner_rescale
        total_is_valid = sum_total > Float32(0.0)
        inv_sum_total = (
            cute.arch.rcp_approx(sum_total) if total_is_valid else Float32(0.0)
        )
        my_weight = my_rescale * inv_sum_total
        my_scale0 = scale0 * my_weight
        my_scale1 = scale1 * my_weight

        exchange_warp_base = corr_warp * 4 * 32 * 32
        exchange_addr = Int32(
            (oExchange.iterator + exchange_warp_base + lane_idx * 4).toint()
        )
        if const_expr(self.allow_empty_block_nums):
            is_zero_output = my_scale0 == Float32(0.0) and my_scale1 == Float32(0.0)
            if not is_zero_output:
                bsa_fwd_helpers.tmem_combine_store_exchange_4x32dp32b32x(
                    Int32(tmem_o0_addr),
                    Int32(tmem_o1_addr),
                    exchange_addr,
                    my_scale0,
                    my_scale1,
                )
            else:
                bsa_fwd_helpers.smem_zero_store_exchange_4x32dp32b32x(exchange_addr)
        else:
            bsa_fwd_helpers.tmem_combine_store_exchange_4x32dp32b32x(
                Int32(tmem_o0_addr),
                Int32(tmem_o1_addr),
                exchange_addr,
                my_scale0,
                my_scale1,
            )

        bsa_fwd_helpers.mbar_arrive_and_wait(reduce_mbar_addr, Int32(1))

        out_row = (corr_warp & 1) * cute.arch.WARP_SIZE + lane_idx
        if corr_warp < 2 and out_row < self.m_block_size:
            own_warp_base = corr_warp * 4 * 32 * 32
            partner_warp_base = partner_warp * 4 * 32 * 32
            lane_col_swizzle = lane_idx & 7
            for c in cutlass.range_constexpr(4):
                off = c * 32 * 32 + lane_idx * 4
                col0 = (((c * 4) + 0) ^ lane_col_swizzle) * 8
                col1 = (((c * 4) + 1) ^ lane_col_swizzle) * 8
                col2 = (((c * 4) + 2) ^ lane_col_swizzle) * 8
                col3 = (((c * 4) + 3) ^ lane_col_swizzle) * 8
                if const_expr(self.o_dtype == Float32):
                    col0 = (((c * 8) + 0) ^ lane_col_swizzle) * 4
                    col1 = (((c * 8) + 1) ^ lane_col_swizzle) * 4
                    col2 = (((c * 8) + 2) ^ lane_col_swizzle) * 4
                    col3 = (((c * 8) + 3) ^ lane_col_swizzle) * 4
                    col4 = (((c * 8) + 4) ^ lane_col_swizzle) * 4
                    col5 = (((c * 8) + 5) ^ lane_col_swizzle) * 4
                    col6 = (((c * 8) + 6) ^ lane_col_swizzle) * 4
                    col7 = (((c * 8) + 7) ^ lane_col_swizzle) * 4
                    bsa_fwd_helpers.smem_exchange_reduce_store_f32x32(
                        Int32((oExchange.iterator + own_warp_base + off).toint()),
                        Int32((oExchange.iterator + partner_warp_base + off).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col0))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col1))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col2))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col3))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col4))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col5))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col6))).toint()),
                        Int32((sO.iterator + sO.layout((out_row, col7))).toint()),
                    )
                else:
                    if const_expr(self.is_sage_fp8):
                        bsa_fwd_helpers.smem_exchange_reduce_scale_store_bf16x32(
                            Int32((oExchange.iterator + own_warp_base + off).toint()),
                            Int32(
                                (oExchange.iterator + partner_warp_base + off).toint()
                            ),
                            Int32((sO.iterator + sO.layout((out_row, col0))).toint()),
                            Int32((sO.iterator + sO.layout((out_row, col1))).toint()),
                            Int32((sO.iterator + sO.layout((out_row, col2))).toint()),
                            Int32((sO.iterator + sO.layout((out_row, col3))).toint()),
                            Int32(
                                (
                                    sVScale.iterator
                                    + v_scale_warp_base
                                    + sVScale.layout(c * 32 + 0)
                                ).toint()
                            ),
                            Int32(
                                (
                                    sVScale.iterator
                                    + v_scale_warp_base
                                    + sVScale.layout(c * 32 + 8)
                                ).toint()
                            ),
                            Int32(
                                (
                                    sVScale.iterator
                                    + v_scale_warp_base
                                    + sVScale.layout(c * 32 + 16)
                                ).toint()
                            ),
                            Int32(
                                (
                                    sVScale.iterator
                                    + v_scale_warp_base
                                    + sVScale.layout(c * 32 + 24)
                                ).toint()
                            ),
                        )
                    else:
                        bsa_fwd_helpers.smem_exchange_reduce_store_bf16x32(
                            Int32((oExchange.iterator + own_warp_base + off).toint()),
                            Int32(
                                (oExchange.iterator + partner_warp_base + off).toint()
                            ),
                            Int32((sO.iterator + sO.layout((out_row, col0))).toint()),
                            Int32((sO.iterator + sO.layout((out_row, col1))).toint()),
                            Int32((sO.iterator + sO.layout((out_row, col2))).toint()),
                            Int32((sO.iterator + sO.layout((out_row, col3))).toint()),
                        )

        cute.arch.fence_view_async_shared()

        # Keep every lane converged through the two warp-pair exchange barriers
        # above. On a partial final Q tile, predicating this store before the
        # raw inline-assembly mbarrier collectives introduces tail-dependent
        # lane divergence and can deadlock the CTA on SM100.
        if const_expr(mLSE_cur is not None):
            out_row = (corr_warp & 1) * cute.arch.WARP_SIZE + lane_idx
            valid_rows = seqlen_q - m_block * self.m_block_size
            if corr_warp < 2 and out_row < valid_rows:
                LN2 = math.log(2.0)
                lse_log2 = max_total_safe * softmax_scale_log2 + cute.math.log2(
                    sum_total, fastmath=True
                )
                if const_expr(self.is_sage_fp8):
                    lse_log2 -= Float32(SAGE_P_QUANT_LOG2_SCALE)
                lse = lse_log2 * LN2 if total_is_valid else -Float32.inf
                mLSE_cur[m_block * self.m_block_size + out_row] = lse

    @cute.jit
    def _store_O_to_gmem(
        self,
        sO_stage: cute.Tensor,
        gO: cute.Tensor,
        mO_cur: cute.Tensor,
        sVScale: cute.Tensor,
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

        if const_expr(not self.pack_gqa):
            # Stream one row fragment through registers at a time.  Materializing
            # the complete per-thread O fragment before scaling makes the FP8
            # epilogue spill heavily to local memory.
            for rest_m in cutlass.range_constexpr(cute.size(tOsO.shape[1])):
                if (
                    t0OcO[0, rest_m, 0][0]
                    < seqlen_q - m_tile_idx * self.m_block_size - tOcO[0][0]
                ):
                    tOrO_row = cute.make_rmem_tensor_like(
                        tOsO[None, rest_m, None], self.o_dtype
                    )
                    cute.autovec_copy(tOsO[None, rest_m, None], tOrO_row)
                    if const_expr(self.is_sage_fp8 and self.is_split_kv):
                        # P and the softmax denominator carry the same factor
                        # of 256, so only V_scale remains after normalization.
                        assert cute.size(tOrO_row.shape[0]) % 2 == 0
                        for rest_n in cutlass.range_constexpr(
                            cute.size(tOrO_row.shape[1])
                        ):
                            for vec in cutlass.range_constexpr(
                                0, cute.size(tOrO_row.shape[0]), 2
                            ):
                                col0 = tOcO[vec, rest_m, rest_n][1]
                                col1 = tOcO[vec + 1, rest_m, rest_n][1]
                                value0 = tOrO_row[vec, rest_n].to(Float32)
                                value1 = tOrO_row[vec + 1, rest_n].to(Float32)
                                value0, value1 = cute.arch.mul_packed_f32x2(
                                    (value0, value1),
                                    (
                                        sVScale[col0],
                                        sVScale[col1],
                                    ),
                                )
                                tOrO_row[vec, rest_n] = value0.to(self.o_dtype)
                                tOrO_row[vec + 1, rest_n] = value1.to(self.o_dtype)
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO_row,
                        tOgO[None, rest_m, None],
                        pred=tOpO[None, rest_m, None]
                        if const_expr(self.check_hdim_v_oob)
                        else None,
                    )
        else:
            # PackGQA retains the existing whole-fragment path. The supported
            # Sage FP8 v1 contract does not use PackGQA.
            tOrO = cute.make_rmem_tensor_like(tOsO, self.o_dtype)
            cute.autovec_copy(tOsO, tOrO)
            pack_gqa.store_O(
                mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_tile_idx, seqlen_q
            )

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        mVScale: Optional[cute.Tensor],
        sVScale: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        pipeline_o_epi: pipeline.PipelineAsync,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        num_heads: Int32,
    ):
        epi_consumer_phase = Int32(0)
        tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            out_head_idx = (
                head_idx + split_idx * num_heads
                if const_expr(self.is_split_kv)
                else head_idx
            )

            tidx = cute.arch.thread_idx()[0] % (
                cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
            )
            if const_expr(self.is_sage_fp8 and self.is_split_kv):
                # One epilogue warp cooperatively stages all per-channel V
                # scales.  This happens before waiting for O, so the loads can
                # overlap the producer's final correction work.
                for vec in cutlass.range_constexpr(
                    self.head_dim_v_padded // cute.arch.WARP_SIZE
                ):
                    col = tidx + vec * cute.arch.WARP_SIZE
                    sVScale[col] = mVScale[head_idx, col]
                cute.arch.sync_warp()

            mO_cur = mO[None, None, None, batch_idx][None, None, out_head_idx]
            gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
            gO = layout_utils.select(
                cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
            )
            gO = cute.flat_divide(gO, (self.mma_tiler_pv[0],))[None, 0, None, None]

            if const_expr(self.use_tma_O):
                store_O, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_O, 0, cute.make_layout(1), sO, gO
                )
                for stage in cutlass.range_constexpr(self.q_stage):
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
                for stage in cutlass.range_constexpr(self.q_stage):
                    # wait from corr, issue tma store on smem
                    # 1. wait for O0 final
                    pipeline_o_epi.consumer_wait_w_index_phase(
                        stage, epi_consumer_phase
                    )
                    # 2. copy O0 to gmem
                    m_tile_idx = m_block * self.q_stage + stage
                    self._store_O_to_gmem(
                        sO[None, None, stage],
                        gO[None, None, stage],
                        mO_cur,
                        sVScale,
                        gmem_tiled_copy_O,
                        tidx,
                        seqlen.seqlen_q,
                        m_tile_idx,
                    )
                    pipeline_o_epi.consumer_release_w_index(stage)

            epi_consumer_phase ^= 1

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()

    @cute.jit
    def load_K_group(
        self,
        tma_atom_K: cute.CopyAtom,
        gK: cute.Tensor,
        sK: cute.Tensor,
        sKV_ptr: cute.Pointer,
        sK_tma_layout: cute.ComposedLayout,
        n_block: Callable,
        kv_block_idx: Int32,
        producer_state: pipeline.PipelineState,
        pipeline_kv: pipeline.PipelineAsync,
    ):
        stage = producer_state.index
        logical_base = kv_block_idx * self.sparse_blocks_per_kv
        idx0 = n_block(logical_base + 0)
        idx1 = n_block(logical_base + 1)
        idx2 = n_block(logical_base + 2)
        idx3 = n_block(logical_base + 3)

        pipeline_kv.producer_acquire(producer_state)
        tma_bar_ptr = pipeline_kv.producer_get_barrier(producer_state)
        stage_base = stage * self.kv_elems_per_stage

        for sub in cutlass.range_constexpr(self.sparse_blocks_per_kv):
            slot = (
                0
                if const_expr(sub == 0)
                else 2
                if const_expr(sub == 1)
                else 1
                if const_expr(sub == 2)
                else 3
            )
            sparse_idx = (
                idx0
                if const_expr(sub == 0)
                else idx1
                if const_expr(sub == 1)
                else idx2
                if const_expr(sub == 2)
                else idx3
            )
            smem_offset = stage_base + slot * self.sparse_block_size * (
                self.head_dim_padded
                if const_expr(self.is_sage_fp8)
                else self.head_dim_padded // 2
            )
            sK_sub = cute.make_tensor(
                cute.recast_ptr(sKV_ptr + smem_offset, sK_tma_layout.inner),
                sK_tma_layout.outer,
            )
            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,
                cute.make_layout(1),
                cute.group_modes(sK_sub, 0, 3),
                gK,
            )
            cute.copy(
                tma_atom_K,
                tKgK[None, sparse_idx],
                tKsK[None],
                tma_bar_ptr=tma_bar_ptr,
            )
        producer_state.advance()
        return producer_state

    @cute.jit
    def load_V_group(
        self,
        tma_atom_V: cute.CopyAtom,
        gV: cute.Tensor,
        sV: cute.Tensor,
        sKV_ptr: cute.Pointer,
        sV_tma_layout: cute.ComposedLayout,
        n_block: Callable,
        kv_block_idx: Int32,
        producer_state: pipeline.PipelineState,
        pipeline_kv: pipeline.PipelineAsync,
    ):
        stage = producer_state.index
        logical_base = kv_block_idx * self.sparse_blocks_per_kv
        idx0 = n_block(logical_base + 0)
        idx1 = n_block(logical_base + 1)
        idx2 = n_block(logical_base + 2)
        idx3 = n_block(logical_base + 3)

        pipeline_kv.producer_acquire(producer_state)
        tma_bar_ptr = pipeline_kv.producer_get_barrier(producer_state)
        stage_base = stage * self.kv_elems_per_stage

        for sub in cutlass.range_constexpr(self.sparse_blocks_per_kv):
            sparse_idx = (
                idx0
                if const_expr(sub == 0)
                else idx1
                if const_expr(sub == 1)
                else idx2
                if const_expr(sub == 2)
                else idx3
            )
            if const_expr(self.is_sage_fp8):
                # The native FP8 B operand uses the same interleaved 64-token
                # sub-block order as the QK N operand.  Keeping K and V in the
                # same logical order is essential once selected blocks differ.
                slot = (
                    0
                    if const_expr(sub == 0)
                    else 2
                    if const_expr(sub == 1)
                    else 1
                    if const_expr(sub == 2)
                    else 3
                )
                smem_offset = (
                    stage_base + slot * self.head_dim_v_padded * self.sparse_block_size
                )
            else:
                smem_offset = (
                    stage_base
                    + (sub & 1) * self.head_dim_v_padded * self.sparse_block_size
                    + (sub >> 1) * self.head_dim_v_padded * self.sparse_block_size * 2
                )
            sV_sub = cute.make_tensor(
                cute.recast_ptr(sKV_ptr + smem_offset, sV_tma_layout.inner),
                sV_tma_layout.outer,
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,
                cute.make_layout(1),
                cute.group_modes(sV_sub, 0, 3),
                gV,
            )
            cute.copy(
                tma_atom_V,
                tVgV[None, sparse_idx],
                tVsV[None],
                tma_bar_ptr=tma_bar_ptr,
            )
        producer_state.advance()
        return producer_state
