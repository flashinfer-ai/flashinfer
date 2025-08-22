# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import enum
import math
from typing import Type, Tuple, Optional

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Int64, Float32, Boolean

from .patch import pipeline as pipeline_patch

from typing import Callable, Any
from types import SimpleNamespace

import warnings

# Ignore this specific warning
warnings.filterwarnings(
    "ignore",
    message="This loop is no longer unrolled and may cause performance regression",
)

# Or ignore all UserWarnings (more broad)
warnings.filterwarnings("ignore", category=UserWarning)

"""
A fused multi-head attention (FMHA) example for the NVIDIA Blackwell SM100 architecture using CUTE DSL

This example demonstrates an implementation of fused multi-head attention using a TMA + Blackwell SM100
TensorCore warp-specialized persistent kernel. The implementation integrates the Q*K^T matrix multiplication,
softmax normalization, and softmax(Q*K^T)*V into a single kernel, avoiding intermediate data movement between
global memory and shared memory, thus improving computational efficiency.

The kernel implements key optimizations including:
- Warp specialization for different computation phases (load, MMA, softmax, correction, epilogue)
- Pipeline stages between different warps for overlapping computation and memory access
- Support for different precision data types
- Optional causal masking for autoregressive models

To run this example:

.. code-block:: bash

    python examples/blackwell/fmha.py                                     \
      --qk_acc_dtype Float32 --pv_acc_dtype Float32                       \
      --mma_tiler_mn 128,128                                              \
      --q_shape 4,1024,8,64 --k_shape 4,1024,8,64                         \
      --is_persistent

The above example runs FMHA with batch size 4, sequence length 1024, 8 attention heads, and head
dimension 64. The Blackwell tcgen05 MMA tile shape is (128, 128), and the kernel uses fp16 for input/output
with fp32 for accumulation.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/fmha.py                                 \
      --qk_acc_dtype Float32 --pv_acc_dtype Float32                       \
      --mma_tiler_mn 128,128                                              \
      --q_shape 4,1024,8,64 --k_shape 4,1024,8,64                         \
      --is_persistent --warmup_iterations 10                              \
      --iterations 10 --skip_ref_check

Constraints for this example:
* Supported head dimensions: 32, 64, and 128
* Number of heads in Q must be divisible by number of heads in K
* mma_tiler_mn must be 128,128
* Batch size must be the same for Q, K, and V tensors
* For causal masking, use --is_causal (note: specify without =True/False)
* For persistent scheduling, use --is_persistent (note: specify without =True/False)
"""


class FmhaStaticTileSchedulerParams:
    def __init__(
        self,
        is_persistent: bool,
        problem_shape_mbh: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self.is_persistent = is_persistent
        self.problem_shape_mbh = problem_shape_mbh
        self._loc = loc
        self._ip = ip

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.is_persistent, self.problem_shape_mbh]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.is_persistent, self.problem_shape_mbh], self._values_pos
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return FmhaStaticTileSchedulerParams(*(tuple(obj_list)), loc=self._loc)


def create_fmha_static_tile_scheduler_params(
    is_persistent: bool,
    problem_shape_mbh: cute.Shape,
) -> FmhaStaticTileSchedulerParams:
    return FmhaStaticTileSchedulerParams(is_persistent, problem_shape_mbh)


class FmhaStaticTileScheduler:
    def __init__(
        self,
        params: FmhaStaticTileSchedulerParams,
        current_work_linear_idx: Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self._params = params
        self._blk_coord = blk_coord
        self._grid_shape = grid_shape
        self._is_persistent = params.is_persistent
        self._current_work_linear_idx = current_work_linear_idx
        self._problem_shape_mbh = cute.make_layout(
            params.problem_shape_mbh, loc=loc, ip=ip
        )
        self._num_blocks = cute.size(self._problem_shape_mbh, loc=loc, ip=ip)
        self._is_first_block = True
        self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        self._loc = loc
        self._ip = ip

    # called by host
    @staticmethod
    def get_grid_shape(
        params: FmhaStaticTileSchedulerParams,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        if params.is_persistent:
            hardware_info = cutlass.utils.HardwareInfo()
            sm_count = hardware_info.get_device_multiprocessor_count()
            return (
                cutlass.min(
                    sm_count, cute.size(params.problem_shape_mbh, loc=loc, ip=ip)
                ),
                1,
                1,
            )
        else:
            return params.problem_shape_mbh

    @staticmethod
    def check_valid_work_for_seqlen_q(
        q_tiler: int,
        current_idx: Int32,
        seqlen_q: Int32,
    ) -> Boolean:
        return current_idx * q_tiler < seqlen_q

    def get_current_work(self, *, loc=None, ip=None) -> utils.WorkTileInfo:
        is_valid = (
            self._current_work_linear_idx < self._num_blocks
            if self._is_persistent
            else self._is_first_block
        )

        blk_coord = (0, 0, 0)
        if self._is_persistent:
            blk_coord = self._problem_shape_mbh.get_hier_coord(
                self._current_work_linear_idx, loc=loc, ip=ip
            )
        else:
            blk_coord = self._blk_coord

        # cur_tile_coord is (mid, 0, (bid, hid))
        cur_tile_coord = (
            blk_coord[0],
            0,
            (blk_coord[1], blk_coord[2]),
        )

        return utils.WorkTileInfo(cur_tile_coord, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        if self._is_persistent:
            self._current_work_linear_idx += advance_count * self.num_persistent_sm
        self._is_first_block = False

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self._params)
        values.extend(cutlass.extract_mlir_values(self._current_work_linear_idx))
        values.extend(cutlass.extract_mlir_values(self._blk_coord))
        values.extend(cutlass.extract_mlir_values(self._grid_shape))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 10
        new_params = cutlass.new_from_mlir_values(self._params, values[0:3])
        new_current_work_linear_idx = cutlass.new_from_mlir_values(
            self._current_work_linear_idx, [values[3]]
        )
        new_blk_coord = cutlass.new_from_mlir_values(self._blk_coord, values[4:7])
        new_grid_shape = cutlass.new_from_mlir_values(self._grid_shape, values[7:])
        return FmhaStaticTileScheduler(
            new_params, new_current_work_linear_idx, new_blk_coord, new_grid_shape
        )


def create_fmha_static_tile_scheduler(
    params: FmhaStaticTileSchedulerParams,
    blk_coord: cute.Coord,
    grid_shape: cute.Shape,
) -> FmhaStaticTileScheduler:
    return FmhaStaticTileScheduler(params, blk_coord[0], blk_coord, grid_shape)


class MaskType(enum.Enum):
    NO_MASK = enum.auto()
    RESIDUAL_MASK = enum.auto()
    CAUSAL_MASK = enum.auto()
    SLIDING_WINDOW_MASK = enum.auto()


class BlackwellFusedMultiHeadAttentionForward:
    def __init__(
        self,
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
        mma_tiler: Tuple[int, int, int],
        is_persistent: bool,
        mask_type: MaskType,
        num_repeat_kv_heads: int = 1,
        custom_params: Any | None = None,
        logits_transform: Callable | None = None,
        output_transform: Callable | None = None,
        window_left: int = -1,
        M_D_update: Callable | None = None,
        use_attention_sink: bool = False,
    ):
        """Initializes the configuration for a Blackwell Fused Multi-Head Attention (FMHA) kernel.

        This configuration includes several key aspects:

        1.  Data Type Settings:
            - qk_acc_dtype: Data type for Q*K^T matrix multiplication accumulator
            - pv_acc_dtype: Data type for P*V matrix multiplication accumulator

        2.  MMA Instruction Settings:
            - mma_tiler: The (M, N, K) shape of the MMA instruction unit
            - qk_mma_tiler: MMA shape for Q*K^T computation
            - pv_mma_tiler: MMA shape for P*V computation

        3.  Kernel Execution Mode:
            - is_persistent: Boolean indicating whether to use persistent kernel mode
            - mask_type: Specifies the type of mask to use (no mask, residual mask, or causal mask)

        :param qk_acc_dtype: Data type for Q*K^T matrix multiplication accumulator
        :type qk_acc_dtype: Type[cutlass.Numeric]
        :param pv_acc_dtype: Data type for P*V matrix multiplication accumulator
        :type pv_acc_dtype: Type[cutlass.Numeric]
        :param mma_tiler: The (M, N, K) shape of the MMA instruction
        :type mma_tiler: Tuple[int, int, int]
        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param mask_type: Type of mask to use
        :type mask_type: MaskType
        """

        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.cta_tiler = (
            2 * mma_tiler[0],  # 2 Q tile per CTA
            mma_tiler[1],
            mma_tiler[2],
        )
        self.qk_mma_tiler = mma_tiler
        self.pv_mma_tiler = (
            mma_tiler[0],
            mma_tiler[2],
            mma_tiler[1],
        )
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.mask_type = mask_type
        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.epilogue_warp_id = 14
        self.empty_warp_id = 15
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epilogue_warp_id,
                self.empty_warp_id,
            )
        )

        self.cta_sync_bar_id = 0
        self.tmem_alloc_sync_bar_id = 1

        self.tmem_s0_offset = 0
        self.tmem_s1_offset = 128
        self.tmem_o0_offset = 256
        self.tmem_o1_offset = 384
        self.tmem_p0_offset = 32
        self.tmem_p1_offset = 160

        # vec buffer for row_max & row_sum
        self.tmem_vec0_offset = 0
        self.tmem_vec1_offset = 128

        self.num_regs_softmax = 192
        self.num_regs_correction = 96
        self.num_regs_other = 32
        self.num_regs_empty = 24

        self.buffer_align_bytes = 1024

        num_warps_per_warpgroup = 4
        self.softmax_warpgroup_count = (
            len((*self.softmax0_warp_ids, *self.softmax1_warp_ids))
            // num_warps_per_warpgroup
        )

        self.custom_logits_transform = logits_transform is not None
        self.logits_transform = logits_transform
        self.custom_output_transform = output_transform is not None
        self.output_transform = output_transform
        self.window_left = window_left

        self.num_repeat_kv_heads = num_repeat_kv_heads

        self.custom_params = (
            custom_params if custom_params is not None else SimpleNamespace()
        )
        self.custom_M_D_update = M_D_update is not None
        self.M_D_update = M_D_update
        self.use_attention_sink = use_attention_sink
        if use_attention_sink:
            assert M_D_update is not None, (
                "M_D_update is required when use_attention_sink is True"
            )

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.q_stage = 2
        self.kv_stage = 4 if self.q_dtype.width == 8 else 3
        self.acc_stage = 1
        self.softmax_corr_stage = 1
        self.mma_corr_stage = 2
        self.mma_softmax_stage = 1
        self.epi_stage = 2

    @cute.jit
    def __call__(
        self,
        q_iter: cute.Pointer,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        o_iter: cute.Pointer,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32, Int32],
        cum_seqlen_q: cute.Tensor | None,
        s_q_all: Int32,
        cum_seqlen_k: cute.Tensor | None,
        s_k_all: Int32,
        scale_softmax_log2: Float32,
        scale_output: Float32,
        sink_iter: cute.Pointer | None,
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

        :param q_iter: The query tensor pointer
        :type q_iter: cute.Pointer
        :param k_iter: The key tensor pointer
        :type k_iter: cute.Pointer
        :param v_iter: The value tensor pointer
        :type v_iter: cute.Pointer
        :param o_iter: The output tensor pointer
        :type o_iter: cute.Pointer
        :param problem_size: The problem size with shape [b, s_q, s_k, h_q, h_k, d]. If cum_seqlen_q or cum_seqlen_k is not None, s_q and s_k are the max of the cumulative sequence length respectively.
        :type problem_size: Tuple[Int32, Int32, Int32, Int32, Int32, Int32]
        :param cum_seqlen_q: The cumulative sequence length tensor for query
        :type cum_seqlen_q: cute.Tensor | None
        :param cum_seqlen_k: The cumulative sequence length tensor for key
        :type cum_seqlen_k: cute.Tensor | None
        :param scale_softmax_log2: The log2 scale factor for softmax
        :type scale_softmax_log2: Float32
        :param scale_output: The scale factor for the output
        :type scale_output: Float32
        :param sink_iter: The sink tensor pointer
        :type sink_iter: cute.Pointer | None
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream
        :raises TypeError: If tensor data types don't match or aren't supported
        :raises RuntimeError: If tensor layouts aren't in supported formats
        """
        b, s_q, s_k, h_q, h_k, d = problem_size
        h_r = h_q // h_k

        q_offset = 0
        kv_offset = 0
        o_offset = -s_q * d * h_r * h_k
        b_q = 1
        b_kv = 1
        b_o = s_q * (1 + b)
        stride_b_q = 0
        stride_b_kv = 0
        stride_b_o = d * h_r * h_k

        # (s, d, ((h_r, h_k), b))
        q_layout = cute.make_layout(
            (s_q_all, d, ((h_r, h_k), b_q)),
            stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_q)),
        )
        q = cute.make_tensor(q_iter + q_offset, q_layout)
        # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        k_layout = cute.make_layout(
            (s_k_all, d, ((h_r, h_k), b_kv)),
            stride=(d * h_k, 1, ((0, d), stride_b_kv)),
        )
        k = cute.make_tensor(k_iter + kv_offset, k_layout)
        # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        v_layout = cute.make_layout(
            (d, s_k_all, ((h_r, h_k), b_kv)),
            stride=(1, d * h_k, ((0, d), stride_b_kv)),
        )
        v = cute.make_tensor(v_iter + kv_offset, v_layout)
        # (s, d, ((h_r, h_k), b))
        o_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), b_o)),
            stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_o)),
        )
        o = cute.make_tensor(o_iter + o_offset, o_layout)

        sink = (
            cute.make_tensor(sink_iter, cute.make_layout((h_q,)))
            if self.use_attention_sink
            else None
        )

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type

        self.tile_sched_params, grid = self._compute_grid(
            cute.shape((s_q, d, ((h_r, h_k), b))),
            self.cta_tiler,
            self.is_persistent,
        )

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of v is not supported")

        # check type consistency
        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if cutlass.const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.ONE
        # the intermediate tensor p is from tmem & k-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.pv_mma_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.kv_stage,
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.q_dtype,
            self.acc_stage,
        )
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.v_dtype,
            self.kv_stage,
        )
        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.epi_stage,
        )

        # TMA load for Q
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.pv_mma_tiler,
            pv_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])

        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            o,
            o_smem_layout,
            self.epi_tile,
        )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_kv_bytes = k_copy_size

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mma_s0_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            mma_s1_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            s0_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            s1_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            s0_s1_sequence_mbar_ptr: cute.struct.MemRange[
                Int64, self.softmax_warpgroup_count
            ]
            corr_epi_mbar_ptr: cute.struct.MemRange[Int64, self.epi_stage * 2]
            mma_corr_mbar_ptr: cute.struct.MemRange[Int64, self.mma_corr_stage * 2]
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[Int64, 1]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, cute.cosize(o_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            cum_seqlen_q,
            cum_seqlen_k,
            scale_softmax_log2,
            scale_output,
            sink,
            q_smem_layout_staged,
            k_smem_layout_staged,
            p_tmem_layout_staged,
            v_smem_layout_staged,
            o_smem_layout_staged,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        mO_qdl: cute.Tensor,
        cum_seqlen_q: cute.Tensor | None,
        cum_seqlen_k: cute.Tensor | None,
        scale_softmax_log2: Float32,
        scale_output: Float32,
        sink: cute.Tensor | None,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: FmhaStaticTileSchedulerParams,
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

        :param qk_tiled_mma: Tiled MMA for Q*K^T
        :type qk_tiled_mma: cute.TiledMma
        :param pv_tiled_mma: Tiled MMA for P*V
        :type pv_tiled_mma: cute.TiledMma
        :param tma_atom_q: TMA copy atom for query tensor
        :type tma_atom_q: cute.CopyAtom
        :param mQ_qdl: Partitioned query tensor
        :type mQ_qdl: cute.Tensor
        :param tma_atom_k: TMA copy atom for key tensor
        :type tma_atom_k: cute.CopyAtom
        :param mK_kdl: Partitioned key tensor
        :type mK_kdl: cute.Tensor
        :param tma_atom_v: TMA copy atom for value tensor
        :type tma_atom_v: cute.CopyAtom
        :param mV_dkl: Partitioned value tensor
        :type mV_dkl: cute.Tensor
        :param tma_atom_o: TMA copy atom for output tensor
        :type tma_atom_o: cute.CopyAtom
        :param mO_qdl: Partitioned output tensor
        :type mO_qdl: cute.Tensor
        :param scale_softmax_log2: The log2 scale factor for softmax
        :type scale_softmax_log2: Float32
        :param scale_output: The scale factor for the output
        :type scale_output: Float32
        :param sink: The sink tensor
        :type sink: cute.Tensor | None
        :param q_smem_layout_staged: Shared memory layout for query tensor
        :type q_smem_layout_staged: cute.ComposedLayout
        :param k_smem_layout_staged: Shared memory layout for key tensor
        :type k_smem_layout_staged: cute.ComposedLayout
        :param p_tmem_layout_staged: Tensor memory layout for probability matrix
        :type p_tmem_layout_staged: cute.ComposedLayout
        :param v_smem_layout_staged: Shared memory layout for value tensor
        :type v_smem_layout_staged: cute.ComposedLayout
        :param o_smem_layout_staged: Shared memory layout for output tensor
        :type o_smem_layout_staged: cute.ComposedLayout
        :param tile_sched_params: Scheduling parameters for work distribution
        :type tile_sched_params: FmhaStaticTileSchedulerParams
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Prefetch tma desc
        #
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline_patch.make_pipeline_participants(
            pipeline_type=pipeline.PipelineTmaUmma,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            num_stages=self.q_stage,
            producer_thread_count=len([self.load_warp_id]),
            consumer_thread_count=len([self.mma_warp_id]),
            tx_count=self.tma_copy_q_bytes,
        )
        load_kv_producer, load_kv_consumer = pipeline_patch.make_pipeline_participants(
            pipeline_type=pipeline.PipelineTmaUmma,
            barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
            num_stages=self.kv_stage,
            producer_thread_count=len([self.load_warp_id]),
            consumer_thread_count=len([self.mma_warp_id]),
            tx_count=self.tma_copy_kv_bytes,
        )
        mma_s0_producer, mma_s0_consumer = pipeline_patch.make_pipeline_participants(
            pipeline_type=pipeline.PipelineUmmaAsync,
            barrier_storage=storage.mma_s0_mbar_ptr.data_ptr(),
            num_stages=self.mma_softmax_stage,
            producer_thread_count=len([self.mma_warp_id]),
            consumer_thread_count=self.threads_per_warp * len(self.softmax0_warp_ids),
        )
        mma_s1_producer, mma_s1_consumer = pipeline_patch.make_pipeline_participants(
            pipeline_type=pipeline.PipelineUmmaAsync,
            barrier_storage=storage.mma_s1_mbar_ptr.data_ptr(),
            num_stages=self.mma_softmax_stage,
            producer_thread_count=len([self.mma_warp_id]),
            consumer_thread_count=self.threads_per_warp * len(self.softmax1_warp_ids),
        )
        s0_corr_producer, s0_corr_consumer = pipeline_patch.make_pipeline_participants(
            pipeline_type=pipeline.PipelineAsync,
            barrier_storage=storage.s0_corr_mbar_ptr.data_ptr(),
            num_stages=self.softmax_corr_stage,
            producer_thread_count=self.threads_per_warp * len(self.softmax0_warp_ids),
            consumer_thread_count=self.threads_per_warp * len(self.correction_warp_ids),
        )
        s1_corr_producer, s1_corr_consumer = pipeline_patch.make_pipeline_participants(
            pipeline_type=pipeline.PipelineAsync,
            barrier_storage=storage.s1_corr_mbar_ptr.data_ptr(),
            num_stages=self.softmax_corr_stage,
            producer_thread_count=self.threads_per_warp * len(self.softmax1_warp_ids),
            consumer_thread_count=self.threads_per_warp * len(self.correction_warp_ids),
        )
        corr_epi_producer, corr_epi_consumer = (
            pipeline_patch.make_pipeline_participants(
                pipeline_type=pipeline.PipelineAsync,
                barrier_storage=storage.corr_epi_mbar_ptr.data_ptr(),
                num_stages=self.epi_stage,
                producer_thread_count=self.threads_per_warp
                * len(self.correction_warp_ids),
                consumer_thread_count=self.threads_per_warp
                * len([self.epilogue_warp_id]),
            )
        )
        mma_corr_producer, mma_corr_consumer = (
            pipeline_patch.make_pipeline_participants(
                pipeline_type=pipeline.PipelineUmmaAsync,
                barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
                num_stages=self.mma_corr_stage,
                producer_thread_count=len([self.mma_warp_id]),
                consumer_thread_count=self.threads_per_warp
                * len(self.correction_warp_ids),
            )
        )
        s0_s1_sequence_producer, s0_s1_sequence_consumer = (
            pipeline_patch.make_pipeline_participants(
                pipeline_type=pipeline.PipelineAsync,
                barrier_storage=storage.s0_s1_sequence_mbar_ptr.data_ptr(),
                num_stages=1,
                producer_thread_count=self.threads_per_warp
                * len(self.softmax0_warp_ids),
                consumer_thread_count=self.threads_per_warp
                * len(self.softmax1_warp_ids),
            )
        )
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()

        #  Correction & Epilogue & tmem barrier init
        if warp_idx == self.empty_warp_id:
            cute.arch.mbarrier_init(
                tmem_dealloc_mbar_ptr,
                self.threads_per_warp
                * len(
                    (
                        *self.softmax0_warp_ids,
                        *self.softmax1_warp_ids,
                        *self.correction_warp_ids,
                    )
                ),
            )
        cute.arch.mbarrier_init_fence()

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV_ptr = cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner)
        sV = cute.make_tensor(sV_ptr, v_smem_layout_staged.outer)
        sO = storage.sO.get_tensor(
            o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
        )
        qk_thr_mma = qk_tiled_mma.get_slice(0)  # default 1sm
        pv_thr_mma = pv_tiled_mma.get_slice(0)  # default 1sm
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        qk_acc_shape = qk_thr_mma.partition_shape_C(
            (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        tStS = qk_thr_mma.make_fragment_C(qk_acc_shape)
        pv_acc_shape = pv_thr_mma.partition_shape_C(
            (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
        )
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)

        tStS0 = cute.make_tensor(tStS.iterator + self.tmem_s0_offset, tStS.layout)
        tStS1 = cute.make_tensor(tStS.iterator + self.tmem_s1_offset, tStS.layout)
        tOtO0 = cute.make_tensor(tOtO.iterator + self.tmem_o0_offset, tOtO.layout)
        tOtO1 = cute.make_tensor(tOtO.iterator + self.tmem_o1_offset, tOtO.layout)

        tP = cute.make_tensor(tStS.iterator, p_tmem_layout_staged.outer)
        tOrP = pv_thr_mma.make_fragment_A(tP)[None, None, None, 0]
        tOrP0 = cute.make_tensor(
            tOrP.iterator
            + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p0_offset,
            tOrP.layout,
        )
        tOrP1 = cute.make_tensor(
            tOrP.iterator
            + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p1_offset,
            tOrP.layout,
        )
        cute.arch.barrier(
            barrier_id=self.cta_sync_bar_id,
            number_of_threads=self.threads_per_cta,
        )
        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.empty_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            tile_sched = create_fmha_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                # block_coord = (mid, 0, (hid, bid))
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                cuseqlen_q = Int32(0)
                seqlen_q = mQ_qdl.shape[0]
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                            self.cta_tiler[0],
                            curr_block_coord[0],
                            seqlen_q,
                        )
                    )
                if not continue_cond:
                    mQ_qdl_ = mQ_qdl
                    mK_kdl_ = mK_kdl
                    mV_dkl_ = mV_dkl
                    seqlen_k = mK_kdl.shape[0]
                    curr_block_coord_q = curr_block_coord
                    curr_block_coord_kv = curr_block_coord

                    if cutlass.const_expr(cum_seqlen_q is not None):
                        logical_offset_mQ = (
                            cuseqlen_q,
                            0,
                            (0, 0),
                        )
                        mQ_qdl_ = cute.domain_offset(logical_offset_mQ, mQ_qdl)
                        curr_block_coord_q = (
                            curr_block_coord[0],
                            curr_block_coord[1],
                            (curr_block_coord[2][0], Int32(0)),
                        )

                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                        logical_offset_mK = (
                            cuseqlen_k,
                            0,
                            (0, 0),
                        )
                        logical_offset_mV = (
                            0,
                            cuseqlen_k,
                            (0, 0),
                        )
                        mK_kdl_ = cute.domain_offset(logical_offset_mK, mK_kdl)
                        mV_dkl_ = cute.domain_offset(logical_offset_mV, mV_dkl)
                        curr_block_coord_kv = (
                            curr_block_coord[0],
                            curr_block_coord[1],
                            (curr_block_coord[2][0], Int32(0)),
                        )

                    # Local tile partition global tensors
                    # (bM, bK, loopM, loopK, loopL)
                    gQ_qdl = cute.flat_divide(
                        mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2])
                    )
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sQ, 0, 3),
                        cute.group_modes(tSgQ_qdl, 0, 3),
                    )
                    tQgQ = tQgQ_qdl[None, None, 0, curr_block_coord_q[2]]

                    gK_kdl = cute.flat_divide(
                        mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
                    )
                    tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                    tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sK, 0, 3),
                        cute.group_modes(tSgK_kdl, 0, 3),
                    )
                    tKgK = tKgK_kdl[None, None, 0, curr_block_coord_kv[2]]

                    gV_dkl = cute.flat_divide(
                        mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2])
                    )
                    tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                    tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sV, 0, 3),
                        cute.group_modes(tSgV_dkl, 0, 3),
                    )
                    tVgV = tVgV_dkl[None, 0, None, curr_block_coord_kv[2]]

                    # Q0
                    q0_coord = 2 * curr_block_coord_q[0]
                    q0_handle = load_q_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, q0_coord],
                        tQsQ[None, q0_handle.index],
                        tma_bar_ptr=q0_handle.barrier,
                    )
                    # K0
                    kv_coord = self.get_kv_start_block_idx(
                        curr_block_coord, self.cta_tiler, seqlen_k
                    )  # seqlen_kv_loop

                    k_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        tKgK[None, kv_coord],
                        tKsK[None, k_handle.index],
                        tma_bar_ptr=k_handle.barrier,
                    )
                    # Q1
                    q1_coord = q0_coord + 1
                    q1_handle = load_q_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, q1_coord],
                        tQsQ[None, q1_handle.index],
                        tma_bar_ptr=q1_handle.barrier,
                    )
                    # V0
                    v_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_v,
                        tVgV[None, kv_coord],
                        tVsV[None, v_handle.index],
                        tma_bar_ptr=v_handle.barrier,
                    )
                    kv_coord += 1

                    seqlen_kv_loop_steps = (
                        self.get_trip_count(curr_block_coord, self.cta_tiler, seqlen_k)
                        - 1
                    )
                    for _i in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                        # Ki
                        k_handle = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, kv_coord],
                            tKsK[None, k_handle.index],
                            tma_bar_ptr=k_handle.barrier,
                        )
                        # Vi
                        v_handle = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, kv_coord],
                            tVsV[None, v_handle.index],
                            tma_bar_ptr=v_handle.barrier,
                        )
                        kv_coord += 1
                    # End of seqlen_kv loop

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                # End of persistent scheduler loop

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            cute.arch.barrier(
                barrier_id=self.tmem_alloc_sync_bar_id,
                number_of_threads=self.threads_per_warp,
            )
            tile_sched = create_fmha_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                            self.cta_tiler[0],
                            curr_block_coord[0],
                            seqlen_q,
                        )
                    )

                if not continue_cond:
                    seqlen_k = mK_kdl.shape[0]
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

                    # GEMM_QK00 (Q0 * K0 -> S0)
                    # 1. wait for Q0
                    q0_handle = load_q_consumer.wait_and_advance()
                    tSrQ0 = tSrQ[None, None, None, q0_handle.index]
                    # 2. wait for K0
                    k_handle = load_kv_consumer.wait_and_advance()
                    tSrK0 = tSrK[None, None, None, k_handle.index]
                    # 3. acquire empty S0 buffer
                    s0_handle = mma_s0_producer.acquire_and_advance()
                    # 4. gemm
                    num_kphases = cute.size(tSrQ0, mode=[2])
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord_0 = (None, None, kphase_idx)
                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                        cute.gemm(
                            qk_tiled_mma,
                            tStS0,
                            tSrQ0[kphase_coord_0],
                            tSrK0[kphase_coord_0],
                            tStS0,
                        )
                    # 5. release S0
                    s0_handle.commit()
                    # End of GEMM (Q0 * K0 -> S0)

                    # GEMM_QK10 (Q1 * K0 -> S1), K0 is ready in GEMM_QK00
                    # 1. wait for Q1
                    q1_handle = load_q_consumer.wait_and_advance()
                    tSrQ1 = tSrQ[None, None, None, q1_handle.index]
                    # 2. acquire empty S1
                    s1_handle = mma_s1_producer.acquire_and_advance()
                    # 3. gemm
                    num_kphases = cute.size(tSrQ1, mode=[2])
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord_1 = (None, None, kphase_idx)
                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                        cute.gemm(
                            qk_tiled_mma,
                            tStS1,
                            tSrQ1[kphase_coord_1],
                            tSrK0[kphase_coord_1],
                            tStS1,
                        )
                    # 4. release S1
                    s1_handle.commit()
                    # 5. release K0
                    k_handle.release()
                    # End of GEMM (Q1 * K0 -> S1)
                    # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                    # so we need to release them after the seqlen_kv loop

                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    v_handle = load_kv_consumer.wait_and_advance()
                    tOrVi = tOrV[None, None, None, v_handle.index]
                    # 2. acquire corrected O0_partial
                    # Note: acquire corr first to take it out of the critical
                    # path since softmax takes longer
                    o0_handle = mma_corr_producer.acquire_and_advance()
                    # 3. acquire P0
                    # this acquire returns the ownership of all of S0 to the mma warp
                    # including the P0 part (inplaced in S0)
                    s0_handle = mma_s0_producer.acquire_and_advance()
                    # 4. gemm
                    num_kphases = cute.size(tOrP0, mode=[2])
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord_2 = (None, None, kphase_idx)
                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                        cute.gemm(
                            pv_tiled_mma,
                            tOtO0,
                            tOrP0[kphase_coord_2],
                            tOrVi[kphase_coord_2],
                            tOtO0,
                        )
                    # 5. release accumulated O0_partial
                    o0_handle.commit()
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                    seqlen_kv_loop_steps = (
                        self.get_trip_count(curr_block_coord, self.cta_tiler, seqlen_k)
                        - 1
                    )

                    # O1 hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                    pv_whether_acc = False
                    for _i in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                        # GEMM_QK0i (Q0 * Ki -> S0)
                        # 1. wait for Ki
                        k_handle = load_kv_consumer.wait_and_advance()
                        tSrKi = tSrK[None, None, None, k_handle.index]
                        # 2. gemm
                        inner_num_kphases = cute.size(tSrQ0, mode=[2])
                        for kphase_idx in cutlass.range(
                            inner_num_kphases, unroll_full=True
                        ):
                            kphase_coord_3 = (None, None, kphase_idx)
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                            cute.gemm(
                                qk_tiled_mma,
                                tStS0,
                                tSrQ0[kphase_coord_3],
                                tSrKi[kphase_coord_3],
                                tStS0,
                            )
                        # 3. release S0
                        s0_handle.commit()
                        # End of GEMM_QK0i (Q0 * Ki -> S0)

                        # GEMM_PV1(i-1) (P1 * V(i-1) -> O1_partial), V(i-1) is ready in GEMM_PV0(i-1)
                        # 1. acquire corrected O1_partial
                        o1_handle = mma_corr_producer.acquire_and_advance()
                        # 2. acquire P1
                        s1_handle = mma_s1_producer.acquire_and_advance()
                        # 3. gemm
                        inner_num_kphases = cute.size(tOrP0, mode=[2])
                        for kphase_idx in cutlass.range(
                            inner_num_kphases, unroll_full=True
                        ):
                            kphase_coord_4 = (None, None, kphase_idx)
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                            cute.gemm(
                                pv_tiled_mma,
                                tOtO1,
                                tOrP1[kphase_coord_4],
                                tOrVi[kphase_coord_4],
                                tOtO1,
                            )
                            pv_whether_acc = True
                        # 4. release accumulated O1_partial
                        o1_handle.commit()
                        # 5. release V(i-1)
                        v_handle.release()
                        # End of GEMM_PV1(i-1) (P1 * V(i-1) -> O1_partial)

                        # GEMM_QK1i (Q1 * Ki -> S1), Q1 is ready in GEMM_QK10; Ki is ready in GEMM_QK0i
                        # 1. gemm
                        inner_num_kphases = cute.size(tSrQ1, mode=[2])
                        for kphase_idx in cutlass.range(
                            inner_num_kphases, unroll_full=True
                        ):
                            kphase_coord_5 = (None, None, kphase_idx)
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                            cute.gemm(
                                qk_tiled_mma,
                                tStS1,
                                tSrQ1[kphase_coord_5],
                                tSrKi[kphase_coord_5],
                                tStS1,
                            )
                        s1_handle.commit()
                        # 2. release Ki
                        k_handle.release()
                        # End of GEMM_QK1i (Q1 * Ki -> S1)

                        # GEMM_PV0i (P0 * Vi -> O0_partial)
                        # 1. wait for Vi
                        v_handle = load_kv_consumer.wait_and_advance()
                        tOrVi = tOrV[None, None, None, v_handle.index]
                        # 2. acquire corrected O0_partial
                        o0_handle = mma_corr_producer.acquire_and_advance()
                        # 3. acquire P0
                        s0_handle = mma_s0_producer.acquire_and_advance()
                        # 4. gemm
                        inner_num_kphases = cute.size(tOrP0, mode=[2])
                        for kphase_idx in cutlass.range(
                            inner_num_kphases, unroll_full=True
                        ):
                            kphase_coord_6 = (None, None, kphase_idx)
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(
                                pv_tiled_mma,
                                tOtO0,
                                tOrP0[kphase_coord_6],
                                tOrVi[kphase_coord_6],
                                tOtO0,
                            )
                        # 5. release accumulated O0_partial
                        o0_handle.commit()
                        # End of GEMM_PV0i (P0 * Vi -> O0_partial)
                    # End of seqlen_kv loop

                    # release Q0 & Q1
                    q0_handle.release()
                    q1_handle.release()

                    # GEMM_PV1(i_end) (P1 * Vi_end -> O1)
                    # 1. acquire corrected O1_partial
                    o1_handle = mma_corr_producer.acquire_and_advance()
                    # 2. acquire P1
                    s1_handle = mma_s1_producer.acquire_and_advance()
                    # 3. gemm
                    num_kphases = cute.size(tOrP1, mode=[2])
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord_7 = (None, None, kphase_idx)
                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(
                            pv_tiled_mma,
                            tOtO1,
                            tOrP1[kphase_coord_7],
                            tOrVi[kphase_coord_7],
                            tOtO1,
                        )
                    # 4. commit accumulated O1
                    o1_handle.commit()
                    # 5. release Vi_end
                    v_handle.release()
                    # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

                    # Commit S0 and S1
                    s0_handle.commit()
                    s1_handle.commit()

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # End of persistent scheduler loop

            # dealloc tmem buffer
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            #  Retrieving tmem ptr and make acc
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.epilogue_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            tile_sched = create_fmha_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                cuseqlen_q = Int32(0)
                seqlen_q = mQ_qdl.shape[0]

                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                            self.cta_tiler[0],
                            curr_block_coord[0],
                            seqlen_q,
                        )
                    )
                if not continue_cond:
                    curr_block_coord_o = curr_block_coord
                    mO_qdl_ = mO_qdl
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        logical_offset_mO = (
                            mO_qdl_.shape[0] - seqlen_q,
                            0,
                            (0, cuseqlen_q + seqlen_q),
                        )
                        mO_qdl_ = cute.domain_offset(logical_offset_mO, mO_qdl_)
                        curr_block_coord_o = (
                            curr_block_coord[0],
                            curr_block_coord[1],
                            (curr_block_coord[2][0], 0),
                        )

                    o0_coord = 2 * curr_block_coord_o[0]
                    o1_coord = o0_coord + 1
                    gO_qdl = cute.flat_divide(
                        mO_qdl_, cute.select(self.pv_mma_tiler, mode=[0, 1])
                    )
                    gO = gO_qdl[None, None, None, 0, curr_block_coord_o[2]]
                    tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_o,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sO, 0, 2),
                        cute.group_modes(gO, 0, 2),
                    )

                    # O0 O1 using the same pipeline
                    # wait from corr, issue tma store on smem
                    # O0
                    # 1. wait for O0 final
                    o0_handle = corr_epi_consumer.wait_and_advance()
                    # 2. copy O0 to gmem
                    cute.copy(tma_atom_o, tOsO[None, 0], tOgO[None, o0_coord])
                    cute.arch.cp_async_bulk_commit_group()
                    # O1
                    # 1. wait for O1 final
                    o1_handle = corr_epi_consumer.wait_and_advance()
                    # 2. copy O1 to gmem
                    cute.copy(tma_atom_o, tOsO[None, 1], tOgO[None, o1_coord])
                    cute.arch.cp_async_bulk_commit_group()

                    # Ensure O0 buffer is ready to be released
                    cute.arch.cp_async_bulk_wait_group(1, read=True)
                    o0_handle.release()
                    # Ensure O1 buffer is ready to be released
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    o1_handle.release()

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # End of persistent scheduler loop

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax0
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.softmax1_warp_ids[0]:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)

            self.softmax(
                stage=0,
                seqlen_k=mK_kdl.shape[0],
                cum_seqlen_q=cum_seqlen_q,
                cum_seqlen_k=cum_seqlen_k,
                scale_softmax_log2=scale_softmax_log2,
                qk_thr_mma=qk_thr_mma,
                tStS=tStS,
                tStSi=tStS0,
                sink=sink,
                mma_si_consumer=mma_s0_consumer,
                si_corr_producer=s0_corr_producer,
                s0_s1_sequence_consumer=s0_s1_sequence_consumer,
                s0_s1_sequence_producer=s0_s1_sequence_producer,
                tile_sched_params=tile_sched_params,
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax1
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx < self.correction_warp_ids[0]
            and warp_idx >= self.softmax1_warp_ids[0]
        ):
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)

            self.softmax(
                stage=1,
                seqlen_k=mK_kdl.shape[0],
                cum_seqlen_q=cum_seqlen_q,
                cum_seqlen_k=cum_seqlen_k,
                scale_softmax_log2=scale_softmax_log2,
                qk_thr_mma=qk_thr_mma,
                tStS=tStS,
                tStSi=tStS1,
                sink=sink,
                mma_si_consumer=mma_s1_consumer,
                si_corr_producer=s1_corr_producer,
                s0_s1_sequence_consumer=s0_s1_sequence_consumer,
                s0_s1_sequence_producer=s0_s1_sequence_producer,
                tile_sched_params=tile_sched_params,
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)

            cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
            tScS = qk_thr_mma.partition_C(cS)

            tStS_vec_layout = cute.composition(tStS.layout, cute.make_layout((128, 2)))

            tStS_vec0 = cute.make_tensor(
                tStS.iterator + self.tmem_vec0_offset, tStS_vec_layout
            )
            tStS_vec1 = cute.make_tensor(
                tStS.iterator + self.tmem_vec1_offset, tStS_vec_layout
            )

            tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
            tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)

            tmem_load_v_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
                self.qk_acc_dtype,
            )

            tiled_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStS_vec0)
            thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
            thr_tmem_load_vec = tiled_tmem_load_vec.get_slice(thread_idx)

            tTMEM_LOAD_VECtS0 = thr_tmem_load_vec.partition_S(tStS_vec0)
            tTMEM_LOAD_VECtS1 = thr_tmem_load_vec.partition_S(tStS_vec1)
            tTMEM_LOAD_VECcS = thr_tmem_load_vec.partition_D(tScS_vec)

            tile_sched = create_fmha_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                head_coord = curr_block_coord[2][0]
                qo_idx_offset = curr_block_coord[0] * self.cta_tiler[0]
                # qo_head_idx = head_coord

                seqlen_k = mK_kdl.shape[0]
                continue_cond = False

                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                            self.cta_tiler[0],
                            curr_block_coord[0],
                            seqlen_q,
                        )
                    )

                if not continue_cond:
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                    # Ignore first signal from softmax as no correction is required
                    vec0_handle = s0_corr_consumer.wait_and_advance()
                    vec0_handle.release()
                    vec1_handle = s1_corr_consumer.wait_and_advance()
                    seqlen_kv_loop_steps = (
                        self.get_trip_count(curr_block_coord, self.cta_tiler, seqlen_k)
                        - 1
                    )
                    for _i in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                        # wait for vec0 (row_wise current max & previous max)
                        vec0_handle = s0_corr_consumer.wait_and_advance()
                        tTMEM_LOAD_VECrS = cute.make_fragment(
                            tTMEM_LOAD_VECcS.shape, self.qk_acc_dtype
                        )
                        cute.copy(
                            tiled_tmem_load_vec, tTMEM_LOAD_VECtS0, tTMEM_LOAD_VECrS
                        )
                        scale_ = scale_softmax_log2 * (
                            tTMEM_LOAD_VECrS[0] - tTMEM_LOAD_VECrS[1]
                        )
                        scale = cute.arch.exp2(scale_)

                        # wait for o0
                        o0_handle = mma_corr_consumer.wait_and_advance()
                        if cutlass.const_expr(not self.custom_logits_transform):
                            self.correction_rescale(pv_thr_mma, tOtO0, scale)
                        # release vec1 & o0
                        vec1_handle.release()
                        cute.arch.fence_view_async_tmem_store()
                        o0_handle.release()

                        # wait for vec1 (row_wise current max & previous max)
                        vec1_handle = s1_corr_consumer.wait_and_advance()
                        cute.copy(
                            tiled_tmem_load_vec, tTMEM_LOAD_VECtS1, tTMEM_LOAD_VECrS
                        )
                        scale_ = scale_softmax_log2 * (
                            tTMEM_LOAD_VECrS[0] - tTMEM_LOAD_VECrS[1]
                        )
                        scale = cute.arch.exp2(scale_)

                        o1_handle = mma_corr_consumer.wait_and_advance()
                        if cutlass.const_expr(not self.custom_logits_transform):
                            self.correction_rescale(pv_thr_mma, tOtO1, scale)
                        vec0_handle.release()
                        cute.arch.fence_view_async_tmem_store()
                        o1_handle.release()
                    # End of seqlen_corr_loop_steps
                    vec1_handle.release()

                    # wait for vec0 (row_wise global sum)
                    vec0_handle = s0_corr_consumer.wait_and_advance()
                    tTMEM_LOAD_VECrS = cute.make_fragment(
                        tTMEM_LOAD_VECcS.shape, self.qk_acc_dtype
                    )
                    cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtS0, tTMEM_LOAD_VECrS)
                    cute.arch.fence_view_async_tmem_load()
                    vec0_handle.release()
                    # wait for o0
                    o0_handle = mma_corr_consumer.wait_and_advance()
                    o0_final_handle = corr_epi_producer.acquire_and_advance()

                    epilogue_scale = scale_output
                    d = tTMEM_LOAD_VECrS[0]  # row sum
                    m = tTMEM_LOAD_VECrS[1]  # row max
                    self.correction_epilog(
                        pv_thr_mma,
                        tOtO0,
                        epilogue_scale,
                        m,
                        d,
                        sO[None, None, 0],
                        batch_coord,
                        head_coord,
                        qo_idx_offset,
                    )
                    o0_handle.release()
                    o0_final_handle.commit()

                    # wait for vec1 (row_wise global sum)
                    vec1_handle = s1_corr_consumer.wait_and_advance()
                    cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtS1, tTMEM_LOAD_VECrS)
                    cute.arch.fence_view_async_tmem_load()
                    vec1_handle.release()
                    # wait for o1
                    o1_handle = mma_corr_consumer.wait_and_advance()
                    o1_final_handle = corr_epi_producer.acquire_and_advance()

                    epilogue_scale = scale_output
                    d = tTMEM_LOAD_VECrS[0]  # row sum
                    m = tTMEM_LOAD_VECrS[1]  # row max
                    self.correction_epilog(
                        pv_thr_mma,
                        tOtO1,
                        epilogue_scale,
                        m,
                        d,
                        sO[None, None, 1],
                        batch_coord,
                        head_coord,
                        qo_idx_offset + self.qk_mma_tiler[0],
                    )
                    o1_handle.release()
                    o1_final_handle.commit()
                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # End of persistent scheduler loop
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)
        return

    @cute.jit
    def softmax_step(
        self,
        stage: int,
        need_apply_mask: bool,
        iter_args: tuple,
        value_args: tuple,
        pipeline_args: tuple,
        atom_args: tuple,
        tensor_args: tuple,
        sink: cute.Tensor | None,
    ) -> Tuple[
        Float32,
        Float32,
        pipeline_patch.PipelineProducer.ImmutableResourceHandle,
        pipeline_patch.PipelineConsumer,
        pipeline_patch.PipelineProducer,
        pipeline_patch.PipelineConsumer,
        pipeline_patch.PipelineProducer,
    ]:
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

        :param stage: Processing stage (0 for first half, 1 for second half)
        :type stage: int
        :param need_apply_mask: Whether to apply attention masking
        :type need_apply_mask: bool
        :param iter_args: Tuple containing the counting tensor, row_max, row_sum, and vector buffer's handle for current iteration
        :type iter_args: tuple
        :param value_args: Tuple containing seqlen_k and scale_softmax_log2
        :type value_args: tuple
        :param pipeline_args: Tuple containing pipeline related arguments for MMA, correction, and sequence synchronization
        :type pipeline_args: tuple
        :param atom_args: Tuple containing mma & copy atoms
        :type atom_args: tuple
        :param tensor_args: Tuple containing softmax related tensors
        :type tensor_args: tuple
        :return: Updated state values (row_max, row_sum, and pipeline related arguments)
        :rtype: tuple
        """
        cS, row_max, row_sum, vec_i_handle, batch_coord, head_coord = iter_args
        qo_head_idx = head_coord
        kv_head_idx = qo_head_idx // self.num_repeat_kv_heads
        kv_tile_idx = cS[0][1] // self.qk_mma_tiler[1]

        seqlen_k, scale_softmax_log2 = value_args
        (
            mma_si_consumer,
            si_corr_producer,
            s0_s1_sequence_consumer,
            s0_s1_sequence_producer,
        ) = pipeline_args
        (
            qk_thr_mma,
            tiled_tmem_load,
            tiled_tmem_store,
            tiled_tmem_store_vec,
            thr_tmem_load,
            thr_tmem_store,
            thr_tmem_store_vec,
        ) = atom_args
        (
            tTMEM_LOADtS,
            tTMEM_STORE_VECtS,
            tTMEM_STOREtS_x4,
        ) = tensor_args

        if cutlass.const_expr(self.custom_M_D_update):
            self.custom_params.sink = sink
            row_max, row_sum = self.M_D_update(
                self.custom_params,
                kv_tile_idx,
                qo_head_idx,
                row_max,
                row_sum,
                scale_softmax_log2,
            )

        tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.o_dtype.width
        tScS = qk_thr_mma.partition_C(cS)
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)

        tScS_P_layout = cute.composition(
            tScS.layout, cute.make_layout((128, tilePlikeFP32))
        )
        tScS_P = cute.make_tensor(tScS.iterator, tScS_P_layout)
        tTMEM_LOADcS = thr_tmem_load.partition_D(tScS)
        tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(tScS_vec)
        tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P)

        # Wait for Si
        si_handle = mma_si_consumer.wait_and_advance()
        tTMEM_LOADrS = cute.make_fragment(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        cute.copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS)
        if need_apply_mask:
            self.apply_mask(tTMEM_LOADrS, tTMEM_LOADcS, seqlen_k)

        old_row_max = row_max
        row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
        row_max_safe = row_max

        if row_max == -cutlass.Float32.inf:
            row_max_safe = 0.0
        tTMEM_STORE_VECrS = cute.make_fragment(
            tTMEM_STORE_VECcS.shape, self.qk_acc_dtype
        )

        tTMEM_STORE_VECrS[0] = old_row_max
        tTMEM_STORE_VECrS[1] = row_max_safe
        cute.copy(tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS)
        cute.arch.fence_view_async_tmem_store()
        # Notify correction wg that row_max is ready
        vec_i_handle.commit()

        tTMEM_STORErS_x4 = cute.make_fragment(tTMEM_STOREcS.shape, self.qk_acc_dtype)
        tTMEM_STORErS_x4_e = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErS_x4.iterator, dtype=self.q_dtype),
            tTMEM_LOADrS.layout,
        )

        scale = scale_softmax_log2
        minus_row_max_scale = (0.0 - row_max_safe) * scale

        # Sequence barrier wait
        if cutlass.const_expr(stage == 0):
            sequence_producer_handle = s0_s1_sequence_producer.acquire_and_advance()
        else:
            sequence_consumer_handle = s0_s1_sequence_consumer.wait_and_advance()
        frg_cnt = 4
        frg_tile = cute.size(tTMEM_LOADrS) // frg_cnt
        tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))
        tTMEM_STORErS_x4_e_frg = cute.logical_divide(
            tTMEM_STORErS_x4_e, cute.make_layout(frg_tile)
        )
        tTMEM_LOADcS_frg = cute.logical_divide(tTMEM_LOADcS, cute.make_layout(frg_tile))
        ### the softmax computation part ### e^(xi*scale - mi*scale)
        if cutlass.const_expr(not self.custom_logits_transform):
            for j in range(frg_cnt):
                for k in range(0, cute.size(tTMEM_LOADrS_frg, mode=[0]), 2):
                    tTMEM_LOADrS_frg[k, j], tTMEM_LOADrS_frg[k + 1, j] = (
                        cute.arch.fma_packed_f32x2(
                            (tTMEM_LOADrS_frg[k, j], tTMEM_LOADrS_frg[k + 1, j]),
                            (scale, scale),
                            (minus_row_max_scale, minus_row_max_scale),
                        )
                    )
                    tTMEM_LOADrS_frg[k, j] = cute.arch.exp2(tTMEM_LOADrS_frg[k, j])
                    tTMEM_LOADrS_frg[k + 1, j] = cute.arch.exp2(
                        tTMEM_LOADrS_frg[k + 1, j]
                    )
                s_vec = tTMEM_LOADrS_frg[None, j].load()
                tTMEM_STORErS_x4_e_frg[None, j].store(s_vec.to(self.q_dtype))

        else:
            for j in range(frg_cnt):
                for k in range(cute.size(tTMEM_LOADrS_frg, mode=[0])):
                    qo_idx, kv_idx = tTMEM_LOADcS_frg[k, j]
                    tTMEM_LOADrS_frg[k, j] = self.logits_transform(
                        self.custom_params,
                        tTMEM_LOADrS_frg[k, j],
                        batch_coord,
                        qo_idx,
                        kv_idx,
                        qo_head_idx,
                        kv_head_idx,
                    )
                s_vec = tTMEM_LOADrS_frg[None, j].load()
                tTMEM_STORErS_x4_e_frg[None, j].store(s_vec.to(self.q_dtype))

        # Sequence barrier arrive
        if cutlass.const_expr(stage == 0):
            sequence_producer_handle.commit()
        else:
            sequence_consumer_handle.release()
        cute.copy(tiled_tmem_store, tTMEM_STORErS_x4, tTMEM_STOREtS_x4)
        cute.arch.fence_view_async_tmem_store()
        # Notify tensor core warp that softmax(S->P) is ready
        si_handle.release()

        ### di = di-1 * (e^(mi-1 - mi) * scale) + sum e^(xi*scale - mi*scale)
        vec_i_handle = si_corr_producer.acquire_and_advance()
        acc_scale_ = scale * (old_row_max - row_max_safe)
        acc_scale = cute.arch.exp2(acc_scale_) * 0.5
        row_sum *= acc_scale
        local_row_sum_0 = (row_sum, row_sum)
        local_row_sum_1 = (0.0, 0.0)
        local_row_sum_2 = (0.0, 0.0)
        local_row_sum_3 = (0.0, 0.0)

        reduction_unroll = 4
        frg_tile = cute.size(tTMEM_LOADrS) // reduction_unroll
        tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))

        for j in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS_frg, mode=[0]), 2):
            local_row_sum_0 = cute.arch.add_packed_f32x2(
                local_row_sum_0, (tTMEM_LOADrS_frg[j, 0], tTMEM_LOADrS_frg[j + 1, 0])
            )
            local_row_sum_1 = cute.arch.add_packed_f32x2(
                local_row_sum_1, (tTMEM_LOADrS_frg[j, 1], tTMEM_LOADrS_frg[j + 1, 1])
            )
            local_row_sum_2 = cute.arch.add_packed_f32x2(
                local_row_sum_2, (tTMEM_LOADrS_frg[j, 2], tTMEM_LOADrS_frg[j + 1, 2])
            )
            local_row_sum_3 = cute.arch.add_packed_f32x2(
                local_row_sum_3, (tTMEM_LOADrS_frg[j, 3], tTMEM_LOADrS_frg[j + 1, 3])
            )

        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_1)
        local_row_sum_2 = cute.arch.add_packed_f32x2(local_row_sum_2, local_row_sum_3)
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_2)
        row_sum = local_row_sum_0[0] + local_row_sum_0[1]

        return (
            row_max,
            row_sum,
            vec_i_handle,
            mma_si_consumer,
            si_corr_producer,
            s0_s1_sequence_consumer,
            s0_s1_sequence_producer,
        )

    # For both softmax0 and softmax1 warp group
    @cute.jit
    def softmax(
        self,
        stage: int,
        seqlen_k: Int32,
        cum_seqlen_q: cute.Tensor | None,
        cum_seqlen_k: cute.Tensor | None,
        scale_softmax_log2: Float32,
        qk_thr_mma: cute.core.ThrMma,
        tStS: cute.Tensor,
        tStSi: cute.Tensor,
        sink: cute.Tensor | None,
        mma_si_consumer: pipeline_patch.PipelineConsumer,
        si_corr_producer: pipeline_patch.PipelineProducer,
        s0_s1_sequence_consumer: pipeline_patch.PipelineConsumer,
        s0_s1_sequence_producer: pipeline_patch.PipelineProducer,
        tile_sched_params: FmhaStaticTileSchedulerParams,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.

        :param stage: Processing stage (0 for first half, 1 for second half of attention matrix)
        :type stage: int
        :param scale_softmax_log2: Log2 scale factor for softmax operation
        :type scale_softmax_log2: Float32
        :param qk_thr_mma: Thread MMA operation for QK matrix multiplication
        :type qk_thr_mma: cute.core.ThrMma
        :param tStS: Shared tensor for softmax input/output
        :type tStS: cute.Tensor
        :param tStSi: Input tensor containing attention scores
        :type tStSi: cute.Tensor
        :param mma_si_pipeline: Pipeline for synchronizing with MMA operations
        :type mma_si_pipeline: pipeline.PipelineAsync
        :param si_corr_pipeline: Pipeline for synchronizing with correction operations
        :type si_corr_pipeline: pipeline.PipelineAsync
        :param s0_s1_sequence_pipeline: Pipeline for synchronizing between stage 0 and 1
        :type s0_s1_sequence_pipeline: pipeline.PipelineAsync
        :param tile_sched_params: Parameters for tile scheduling
        :type tile_sched_params: FmhaStaticTileSchedulerParams
        :param sink: The sink tensor
        :type sink: cute.Tensor | None
        """
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (
            self.threads_per_warp
            * (
                len(self.softmax0_warp_ids)
                if stage == 0
                else len(self.softmax1_warp_ids)
            )
        )

        cS_base = cute.make_identity_tensor(
            (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        tilePlikeFP32 = self.qk_mma_tiler[1] // 32 * self.o_dtype.width
        tScS = qk_thr_mma.partition_C(cS_base)
        tStS_vec_layout = cute.composition(tStS.layout, cute.make_layout((128, 2)))
        tmem_vec_offset = self.tmem_vec0_offset if stage == 0 else self.tmem_vec1_offset
        tStS_vec = cute.make_tensor(tStS.iterator + tmem_vec_offset, tStS_vec_layout)
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)
        tStS_P_layout = cute.composition(
            tStS.layout, cute.make_layout((128, tilePlikeFP32))
        )
        tmem_p_offset = self.tmem_p0_offset if stage == 0 else self.tmem_p1_offset
        tStS_P = cute.make_tensor(tStS.iterator + tmem_p_offset, tStS_P_layout)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi)
        thread_idx = tidx % (
            self.threads_per_warp
            * (
                len(self.softmax0_warp_ids)
                if stage == 0
                else len(self.softmax1_warp_ids)
            )
        )
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_tmem_load.partition_S(tStSi)
        tmem_store_vec_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store_vec = tcgen05.make_tmem_copy(tmem_store_vec_atom, tStS_vec)
        thr_tmem_store_vec = tiled_tmem_store_vec.get_slice(thread_idx)
        tTMEM_STORE_VECtS = thr_tmem_store_vec.partition_D(tStS_vec)
        tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(tScS_vec)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)
        tTMEM_STOREtS_x4 = thr_tmem_store.partition_D(tStS_P)

        tile_sched = create_fmha_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        while work_tile.is_valid_tile:
            curr_block_coord = work_tile.tile_idx
            batch_coord = curr_block_coord[2][1]
            seqlen_k_ = seqlen_k
            continue_cond = False

            if cutlass.const_expr(cum_seqlen_q is not None):
                cuseqlen_q = cum_seqlen_q[batch_coord]
                seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                continue_cond = (
                    not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0],
                        curr_block_coord[0],
                        seqlen_q,
                    )
                )

            if not continue_cond:
                if cutlass.const_expr(cum_seqlen_k is not None):
                    cuseqlen_k = cum_seqlen_k[batch_coord]
                    seqlen_k_ = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                row_max = -Float32.inf
                row_sum = 0.0
                value_args = (seqlen_k_, scale_softmax_log2)
                atom_args = (
                    qk_thr_mma,
                    tiled_tmem_load,
                    tiled_tmem_store,
                    tiled_tmem_store_vec,
                    thr_tmem_load,
                    thr_tmem_store,
                    thr_tmem_store_vec,
                )
                tensor_args = (
                    tTMEM_LOADtS,
                    tTMEM_STORE_VECtS,
                    tTMEM_STOREtS_x4,
                )

                logical_offset = (
                    curr_block_coord[0] * self.cta_tiler[0]
                    + stage * self.qk_mma_tiler[0],
                    0,
                )
                cS = cute.domain_offset(logical_offset, cS_base)
                vec_i_handle = si_corr_producer.acquire_and_advance()
                unmask_count = self.get_unmasked_trip_count(
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_k_,
                )
                batch_coord = curr_block_coord[2][1]
                head_coord = curr_block_coord[2][0]
                for i in cutlass.range(0, unmask_count, 1, unroll=1):
                    cS_iter = cute.domain_offset((0, i * self.qk_mma_tiler[1]), cS)
                    iter_args = (
                        cS_iter,
                        row_max,
                        row_sum,
                        vec_i_handle,
                        batch_coord,
                        head_coord,
                    )
                    pipeline_args = (
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    )
                    (
                        row_max,
                        row_sum,
                        vec_i_handle,
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    ) = self.softmax_step(
                        stage,
                        False,
                        iter_args,
                        value_args,
                        pipeline_args,
                        atom_args,
                        tensor_args,
                        sink,
                    )

                mask_count = self.get_masked_trip_count(
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_k_,
                )

                for i in cutlass.range(
                    unmask_count, unmask_count + mask_count, 1, unroll=1
                ):
                    cS_iter = cute.domain_offset((0, i * self.qk_mma_tiler[1]), cS)
                    iter_args = (
                        cS_iter,
                        row_max,
                        row_sum,
                        vec_i_handle,
                        batch_coord,
                        head_coord,
                    )
                    pipeline_args = (
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    )
                    (
                        row_max,
                        row_sum,
                        vec_i_handle,
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    ) = self.softmax_step(
                        stage,
                        True,
                        iter_args,
                        value_args,
                        pipeline_args,
                        atom_args,
                        tensor_args,
                        sink,
                    )
                si_handle = mma_si_consumer.wait_and_advance()
                tTMEM_STORE_VECrS = cute.make_fragment(
                    tTMEM_STORE_VECcS.shape, self.qk_acc_dtype
                )
                tTMEM_STORE_VECrS[0] = row_sum
                tTMEM_STORE_VECrS[1] = row_max
                cute.copy(tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS)
                cute.arch.fence_view_async_tmem_store()
                vec_i_handle.commit()
                si_corr_producer.acquire()
                # Empty step to sync against pipe s
                si_handle.release()

            # Advance to next tile
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
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

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.core.ThrMma
        :param tOtO: Tensor representing partial attention output to be rescaled
        :type tOtO: cute.Tensor
        :param scale: Scaling factor to apply to the partial results
        :type scale: Float32
        """
        pv_tiled_mma_shape = (
            self.pv_mma_tiler[0],
            self.pv_mma_tiler[1],
        )
        cO = cute.make_identity_tensor(pv_tiled_mma_shape)
        tOcO = thr_mma.partition_C(cO)

        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )

        tOtO_i_layout = cute.composition(
            tOtO.layout, cute.make_layout((128, corr_tile_size))
        )
        tOcO_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )

        tOtO_i = cute.make_tensor(tOtO.iterator, tOtO_i_layout)
        tOcO_i = cute.make_tensor(tOcO.iterator, tOcO_i_layout)

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i)
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i)
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i)

        tTMEM_STOREtO = thr_tmem_store.partition_D(tOtO_i)

        tTMrO = cute.make_fragment(
            (tTMEM_LOADcO.shape, 128 // corr_tile_size), self.pv_acc_dtype
        )
        for i in range(self.cta_tiler[2] // corr_tile_size):
            tTMrO_i_ = tTMrO[None, i]
            tTMrO_i_layout = cute.composition(
                tTMrO_i_.layout, cute.make_layout(tTMrO.shape[0])
            )
            tTMrO_i = cute.make_tensor(tTMrO_i_.iterator, tTMrO_i_layout)
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size, tTMEM_STOREtO.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i)
            for j in range(0, cute.size(tTMrO_i), 2):
                tTMrO_i[j], tTMrO_i[j + 1] = cute.arch.mul_packed_f32x2(
                    (tTMrO_i[j], tTMrO_i[j + 1]),
                    (scale, scale),
                )
            cute.copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i)

    @cute.jit
    def correction_epilog(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        scale: Float32,
        m: Float32,
        d: Float32,
        sO: cute.Tensor,
        batch_coord: Int32,
        head_coord: Int32,
        qo_idx_offset: Int32,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilog function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.core.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        pv_tiled_mma_shape = (
            self.pv_mma_tiler[0],
            self.pv_mma_tiler[1],
        )
        cO = cute.make_identity_tensor(pv_tiled_mma_shape)
        cO_custom = cute.make_identity_tensor(pv_tiled_mma_shape)

        corr_tile_size = 32 * 8 // self.o_dtype.width
        tOsO = thr_mma.partition_C(sO)
        tOcO = thr_mma.partition_C(cO)
        tOcO_custom = thr_mma.partition_C(cO_custom)

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((128, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((128, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((128, corr_tile_size)))
        tOcO_custom_i = cute.logical_divide(
            tOcO_custom, cute.make_layout((128, corr_tile_size))
        )
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils.get_tmem_load_op(
            self.pv_mma_tiler,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_copy_atom, tOtO_i[(None, None), 0]
        )

        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        smem_copy_atom = sm100_utils.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tTMEM_LOADsO = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
        tTMEM_LOADoO = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        tTMEM_LOADcO_custom = thr_tmem_load.partition_D(
            tOcO_custom_i[(None, None), None]
        )

        scale_rcp_d = scale / d if not self.custom_logits_transform else scale
        rcp_d = 1 / d if m != -Float32.inf else 0.0
        for i in range(self.cta_tiler[2] // corr_tile_size):
            tTMEM_LOADtO_i = tTMEM_LOADtO[None, 0, 0, i]
            tTMEM_LOADsO_i = tTMEM_LOADsO[None, 0, 0, i]
            tTMrO = cute.make_fragment(
                tTMEM_LOADoO[None, 0, 0, i].shape, self.pv_acc_dtype
            )
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
            if cutlass.const_expr(not self.custom_output_transform):
                for j in range(0, cute.size(tTMrO), 2):
                    tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j], tTMrO[j + 1]),
                        (scale_rcp_d, scale_rcp_d),
                    )
            else:
                tTMcO_custom = tTMEM_LOADcO_custom[None, 0, 0, i]
                for j in range(0, cute.size(tTMrO)):
                    qo_idx = qo_idx_offset + tTMcO_custom[j][0]
                    tTMrO[j] = self.output_transform(
                        self.custom_params,
                        tTMrO[j],
                        batch_coord,
                        qo_idx,
                        head_coord,
                        m,
                        rcp_d,
                        scale,
                    )
            tSMrO = cute.make_fragment(tTMrO.shape, self.o_dtype)
            o_vec = tTMrO.load()
            tSMrO.store(o_vec.to(self.o_dtype))
            cute.copy(tiled_smem_store, tSMrO, tTMEM_LOADsO_i)

        # fence view async shared
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    def get_trip_count(
        self,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_k: Int32,
    ) -> Int32:
        result = 0
        if (
            self.mask_type == MaskType.NO_MASK
            or self.mask_type == MaskType.RESIDUAL_MASK
        ):
            result = cute.ceil_div(seqlen_k, tile_shape[1])
        elif self.mask_type == MaskType.CAUSAL_MASK:
            max_blocks_k = cute.ceil_div(seqlen_k, tile_shape[1])
            max_blocks_q = cute.ceil_div(
                (blk_coord[0] + 1) * tile_shape[0], tile_shape[1]
            )
            result = cutlass.min(max_blocks_k, max_blocks_q)
        elif self.mask_type == MaskType.SLIDING_WINDOW_MASK:
            max_blocks_k = cute.ceil_div(self.window_left, tile_shape[1]) + 1
            max_blocks_q = cute.ceil_div(
                (blk_coord[0] + 1) * tile_shape[0], tile_shape[1]
            )
            result = cutlass.min(max_blocks_k, max_blocks_q)
        return result

    @cute.jit
    def get_masked_trip_count(
        self,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_k: Int32,
    ) -> Int32:
        result = 0
        if self.mask_type == MaskType.NO_MASK:
            result = 0
        elif self.mask_type == MaskType.RESIDUAL_MASK:
            if seqlen_k % tile_shape[1] != 0:
                result = 1
            else:
                result = 0
        elif self.mask_type == MaskType.CAUSAL_MASK:
            trip_count = self.get_trip_count(blk_coord, tile_shape, seqlen_k)
            result = cutlass.min(
                trip_count,
                cute.ceil_div(tile_shape[0], tile_shape[1]),
            )
        elif self.mask_type == MaskType.SLIDING_WINDOW_MASK:
            trip_count = self.get_trip_count(blk_coord, tile_shape, seqlen_k)
            result = trip_count
        return result

    @cute.jit
    def get_unmasked_trip_count(
        self,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_k: Int32,
    ) -> Int32:
        result = 0
        if self.mask_type == MaskType.NO_MASK:
            result = self.get_trip_count(blk_coord, tile_shape, seqlen_k)
        elif self.mask_type == MaskType.RESIDUAL_MASK:
            if seqlen_k % tile_shape[1] != 0:
                result = self.get_trip_count(blk_coord, tile_shape, seqlen_k) - 1
            else:
                result = self.get_trip_count(blk_coord, tile_shape, seqlen_k)
        elif self.mask_type == MaskType.CAUSAL_MASK:
            result = self.get_trip_count(
                blk_coord, tile_shape, seqlen_k
            ) - self.get_masked_trip_count(blk_coord, tile_shape, seqlen_k)
        elif self.mask_type == MaskType.SLIDING_WINDOW_MASK:
            result = 0
        return result

    @cute.jit
    def get_kv_start_block_idx(
        self,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_k: Int32,
    ) -> Int32:
        if cutlass.const_expr(self.mask_type == MaskType.SLIDING_WINDOW_MASK):
            num_blocks_k = cute.ceil_div(self.window_left, tile_shape[1])
            block_idx = (
                cute.ceil_div((blk_coord[0] + 1) * tile_shape[0], tile_shape[1]) - 1
            )
            return cutlass.max(0, block_idx - num_blocks_k)
        else:
            return 0

    @cute.jit
    def apply_mask(
        self,
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        seqlen_k: Int32,
    ):
        if self.mask_type == MaskType.RESIDUAL_MASK:
            for i in range(cute.size(acc_qk)):
                pos = index_qk[i]
                if pos[1] >= seqlen_k:
                    acc_qk[i] = -Float32.inf
        elif self.mask_type == MaskType.CAUSAL_MASK:
            for i in range(cute.size(acc_qk)):
                pos = index_qk[i]
                if pos[0] < pos[1] or pos[1] >= seqlen_k:
                    acc_qk[i] = -Float32.inf
        elif self.mask_type == MaskType.SLIDING_WINDOW_MASK:
            for i in range(cute.size(acc_qk)):
                pos = index_qk[i]
                if pos[1] - pos[0] > self.window_left:
                    acc_qk[i] = -Float32.inf

    @staticmethod
    def _compute_grid(
        o_shape: cute.Shape,
        cta_tiler: Tuple[int, int, int],
        is_persistent: bool,
    ) -> Tuple[FmhaStaticTileSchedulerParams, Tuple[int, int, int]]:
        tile_sched_params = create_fmha_static_tile_scheduler_params(
            is_persistent,
            (
                cute.ceil_div(cute.size(o_shape[0]), cta_tiler[0]),
                cute.size(o_shape[2][0]),
                cute.size(o_shape[2][1]),
            ),
        )
        grid = FmhaStaticTileScheduler.get_grid_shape(tile_sched_params)
        return tile_sched_params, grid


def dumb_output_transform(x: cute.Tensor, scale: float) -> cute.Tensor:
    return x * scale * 2.0


def sigmoid_logits_transform(x: cute.Tensor) -> cute.Tensor:
    scale = 1.0 * math.log2(math.exp(1.0))
    bias = 0.0
    return 1 / (1 + cute.arch.exp2(-(x * scale + bias)))


class BatchPrefillCuteDSLWrapper:
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
    ) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

        self._use_cuda_graph = use_cuda_graph

        # Data types will be set in plan() method based on input parameters
        self._in_dtype = None
        self._out_dtype = None
        self._qk_acc_dtype = cutlass.Float32
        self._pv_acc_dtype = cutlass.Float32

    def plan(
        self,
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=None,
        causal=True,
        sm_scale=1.0,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        custom_params: SimpleNamespace | None = None,
        logits_transform: Callable | None = None,
        output_transform: Callable | None = None,
        window_left: int = -1,
        M_D_update: Callable | None = None,
        use_attention_sink: bool = False,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required to run this example!")

        self._batch_size = qo_indptr.shape[0] - 1
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        assert num_qo_heads % num_kv_heads == 0, (
            "num_qo_heads must be divisible by num_kv_heads"
        )
        self._head_dim = head_dim_qk
        assert head_dim_vo is None or head_dim_vo == head_dim_qk, (
            "head_dim_vo must be None or equal to head_dim_qk"
        )
        self._causal = causal
        self._sm_scale = sm_scale
        self._device = qo_indptr.device
        self._is_persistent = True

        h_r = num_qo_heads // num_kv_heads

        self._use_attention_sink = use_attention_sink

        # Set data types based on input parameters
        if q_data_type == torch.bfloat16:
            self._in_dtype = cutlass.BFloat16
            self._out_dtype = cutlass.BFloat16
        elif q_data_type == torch.half:
            self._in_dtype = cutlass.Float16
            self._out_dtype = cutlass.Float16
        elif q_data_type == torch.float8_e4m3fn:
            self._in_dtype = cutlass.Float8E4M3FN
            self._out_dtype = cutlass.Float16  # Output is always Float16 for FP8 input
        else:
            raise ValueError(f"Unsupported input data type: {q_data_type}")

        s_cumsum_q_cute_tensor, s_cumsum_q_torch_tensor = (
            cutlass_torch.cute_tensor_like(
                qo_indptr.to(torch.int32),
                Int32,
                is_dynamic_layout=True,
                assumed_align=16,
            )
        )
        s_q = qo_indptr[1:] - qo_indptr[:-1]

        s_cumsum_k_cute_tensor, s_cumsum_k_torch_tensor = (
            cutlass_torch.cute_tensor_like(
                kv_indptr.to(torch.int32),
                Int32,
                is_dynamic_layout=True,
                assumed_align=16,
            )
        )
        s_k = kv_indptr[1:] - kv_indptr[:-1]

        qo_shape = (1, torch.sum(s_q), h_r * self._num_kv_heads, self._head_dim)
        o_padding = (0, torch.max(s_q), 0, 0, 0)
        kv_shape = (1, torch.sum(s_k), self._num_kv_heads, self._head_dim)

        self._o_padding = o_padding[1]
        self._kv_padding = 0

        q_ref, q_cute, q_torch = create_and_pad_tensor(
            qo_shape,
            (0, 0, 0, 0, 0),
            self._in_dtype,
            s_cumsum=s_cumsum_q_torch_tensor,
            is_dynamic_layout=True,
        )
        k_ref, k_cute, k_torch = create_and_pad_tensor(
            kv_shape,
            (0, 0, 0, 0, 0),
            self._in_dtype,
            s_cumsum=s_cumsum_k_torch_tensor,
            is_dynamic_layout=True,
        )
        v_ref, v_cute, v_torch = create_and_pad_tensor(
            kv_shape,
            (0, 0, 0, 0, 0),
            self._in_dtype,
            s_cumsum=s_cumsum_k_torch_tensor,
            is_dynamic_layout=True,
        )

        _, o_cute, o_torch = create_and_pad_tensor(
            qo_shape,
            o_padding,
            self._out_dtype,
            s_cumsum=s_cumsum_q_torch_tensor,
            is_dynamic_layout=True,
        )

        if use_attention_sink:
            sink = torch.randn(num_qo_heads, dtype=torch.float16, device=self._device)
            sink_cute = from_dlpack(sink, assumed_align=16)

        self._mma_tiler_mn = (128, 128)
        self._mma_tiler = (128, 128, self._head_dim)

        # Create random tensors for compilation
        self._mask_type = MaskType.NO_MASK
        if self._causal:
            self._mask_type = MaskType.CAUSAL_MASK
        elif window_left > 0:
            self._mask_type = MaskType.SLIDING_WINDOW_MASK
        else:
            if s_k.shape[0] > 1:
                for i in range(len(s_k)):
                    if s_k[i] % self._mma_tiler_mn[1] != 0:
                        self._mask_type = MaskType.RESIDUAL_MASK
            else:
                if s_k % self._mma_tiler_mn[1] != 0:
                    self._mask_type = MaskType.RESIDUAL_MASK

        # Create the FMHA instance
        fmha = BlackwellFusedMultiHeadAttentionForward(
            self._qk_acc_dtype,
            self._pv_acc_dtype,
            self._mma_tiler,
            self._is_persistent,
            self._mask_type,
            h_r,
            custom_params,
            logits_transform,
            output_transform,
            window_left,
            M_D_update,
            use_attention_sink,
        )

        problem_size = (
            self._batch_size,
            int(torch.max(s_q).item()),
            int(torch.max(s_k).item()),
            self._num_qo_heads,
            self._num_kv_heads,
            self._head_dim,
        )

        self._problem_size = problem_size
        self._s_cumsum_q_cute_tensor = s_cumsum_q_cute_tensor
        self._s_cumsum_k_cute_tensor = s_cumsum_k_cute_tensor
        self._s_q_all = s_cumsum_q_torch_tensor[-1].item()
        self._s_k_all = s_cumsum_k_torch_tensor[-1].item()

        log2_e = math.log2(
            math.exp(1.0)
        )  # gpu uses exp2 for perf concerns, we need an extra factor 'log2_e' here
        scale_softmax = self._sm_scale
        self._scale_softmax_log2 = scale_softmax * log2_e
        self._scale_output = 1.0

        # Get current CUDA stream from PyTorch
        torch_stream = torch.cuda.current_stream()
        # Get the raw stream pointer as a CUstream
        stream = cuda.CUstream(torch_stream.cuda_stream)

        # compile fmha kernel
        compiled_fmha = cute.compile(
            fmha,
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            o_cute.iterator,
            self._problem_size,
            self._s_cumsum_q_cute_tensor,
            self._s_q_all,
            self._s_cumsum_k_cute_tensor,
            self._s_k_all,
            self._scale_softmax_log2,
            self._scale_output,
            sink_cute.iterator if self._use_attention_sink else None,
            stream,
        )

        self._compiled_fmha = compiled_fmha

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        sink: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Run the prefill attention computation.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor with shape [batch_size, seq_len, num_heads, head_dim].
        k : torch.Tensor
            The key tensor with shape [batch_size, seq_len, num_heads, head_dim].
        v : torch.Tensor
            The value tensor with shape [batch_size, seq_len, num_heads, head_dim].
        out : Optional[torch.Tensor], optional
            The output tensor. If None, a new tensor will be created.
        sink : Optional[torch.Tensor], optional
            The sink tensor with shape [num_heads].
        Returns
        -------
        torch.Tensor
            The output tensor with shape [batch_size, seq_len, num_heads, head_dim].
        """

        if self._compiled_fmha is None:
            raise RuntimeError("Plan the prefill attention computation first!")

        # Create output tensor if not provided
        if out is None:
            out = torch.empty_like(q, device=q.device)

        # Convert tensors to cute format
        # Create dtype cute tensor with offset (gpu)
        q_cute = from_dlpack(q, assumed_align=16)
        k_cute = from_dlpack(k, assumed_align=16)
        v_cute = from_dlpack(v, assumed_align=16)
        o_cute, o_torch = qkv_torch_2_cute(out, self._o_padding, self._out_dtype)

        if self._use_attention_sink:
            assert sink is not None, "sink is required when use_attention_sink is True"
            assert sink.dtype == q.dtype, "sink must have the same dtype as q"
            sink_cute = from_dlpack(sink, assumed_align=16)

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        self._compiled_fmha(
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            o_cute.iterator,
            self._problem_size,
            self._s_cumsum_q_cute_tensor,
            self._s_q_all,
            self._s_cumsum_k_cute_tensor,
            self._s_k_all,
            self._scale_softmax_log2,
            self._scale_output,
            sink_cute.iterator if self._use_attention_sink else None,
            stream,
        )

        return o_torch


def qkv_torch_2_cute(x_torch, padding, dtype, s_cumsum=None, is_dynamic_layout=True):
    # (b, s, h, d)

    # pad tensor in front of the tensor on the second dimension
    x_torch_full = torch.nn.functional.pad(x_torch, (0, 0, 0, 0, padding, 0))

    x_torch = x_torch_full[padding:, :, :].detach()
    x_torch._keep_alive = x_torch_full

    # Create dtype cute tensor with offset (gpu)
    x_cute = from_dlpack(x_torch, assumed_align=16)
    x_cute.element_type = dtype

    return (x_cute, x_torch)


def create_and_pad_tensor(shape, padding, dtype, s_cumsum=None, is_dynamic_layout=True):
    # (b, s, h, d)
    shape_ = tuple(map(lambda x, y: x + y, shape, padding))
    if s_cumsum is not None:
        if shape_[0] != 1 or padding[0] != 0:
            raise ValueError("Invalid tensor creation for variable sequence length")
        # (s_total + padding, h, d)
        shape_ = shape_[1:]
        padding = padding[1:]

    # Create f32 torch tensor (cpu)
    f32_torch_tensor_full = cutlass_torch.create_and_permute_torch_tensor(
        shape_,
        torch.float32,
        permute_order=None,
        init_type=cutlass.torch.TensorInitType.RANDOM,
        init_config=cutlass.torch.RandomInitConfig(
            min_val=-2 if dtype.is_float or dtype.signed else 0, max_val=2
        ),
    )
    # Create dtype cute & torch tensor (gpu)
    _, torch_tensor_full = cutlass_torch.cute_tensor_like(
        f32_torch_tensor_full,
        dtype,
        is_dynamic_layout,
        assumed_align=16,
    )

    # Offset the tensor
    slices = tuple(slice(s, e) for s, e in zip(padding, shape_))
    torch_tensor = torch_tensor_full[slices].detach()
    f32_torch_tensor = f32_torch_tensor_full[slices].detach()
    torch_tensor._keep_alive = torch_tensor_full
    f32_torch_tensor._keep_alive = f32_torch_tensor_full

    # Create dtype cute tensor with offset (gpu)
    cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
    cute_tensor.element_type = dtype

    # From ragged to jagged
    if s_cumsum is not None:
        torch_tensor = torch.nested.nested_tensor_from_jagged(
            values=torch_tensor, offsets=s_cumsum
        )
        f32_torch_tensor = torch.nested.nested_tensor_from_jagged(
            values=f32_torch_tensor, offsets=s_cumsum.cpu()
        )

    return (
        f32_torch_tensor,
        cute_tensor,
        torch_tensor,
    )
