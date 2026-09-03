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


import argparse
from typing import Optional, Tuple, Type, Union


import cutlass
import cutlass.torch
import torch
from cuda.bindings import driver
from cutlass import cute, testing, utils
from cutlass._mlir.dialects import math
from cutlass.cute.arch import (
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import Int32
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    NamedBarrier,
    PipelineAsync,
    PipelineAsyncUmma,
    PipelineTmaStore,
    PipelineTmaUmma,
    PipelineUmmaAsync,
    PipelineUserType,
    make_pipeline_state,
)
from cutlass.utils import blackwell_helpers, blockscaled_layout

from flashinfer.tllm_enums import (
    ActivationType,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
)

from ..moe_utils import (
    normalize_cute_dsl_moe_activation_type,
    normalize_cute_dsl_moe_weight_interleave,
    validate_cute_dsl_moe_situ_config,
)
from .custom_pipeline import PipelineCpAsyncUmma
from .utils import (
    DeviceBoundPersistentTileScheduler,
    compact_sf_layout,
    f32_reciprocal,
    fmax,
    fmin,
    gelu_tanh_f32,
    is_power_of_2,
    make_ptr,
    sigmoid_f32,
    situ_f32,
    tanh_f32,
)

FP32_MAX = torch.finfo(torch.float32).max
"""
High-performance persistent blockscaled contiguous grouped dense GEMM with gather
and FC1 activation fusion for the NVIDIA Blackwell architecture using CUTE DSL.

This kernel performs FC1 layer computation with activation fusion:
1. GEMM: acc = alpha * (SFA * A[token_ids]) * (SFB * B)
2. Activation: SwiGLU, SiTU, tanh-approximate GeGLU, or ReLU^2
3. Optional Quant: Generates scale factor C and quantizes output to NVFP4 or MXFP8

- Matrix A is MxKx1, A can be row-major("K"), ValidM is composed of valid m in different groups
- Matrix B is NxKxL, B can be column-major("K"), L is grouped dimension (number of experts)
  - B weights use the configured 16- or 64-row up/gate interleave
- Matrix C is Mx(N/2)x1 for gated activations and MxNx1 otherwise
- Matrix SFA layout is filled internally according to A shape and BlockScaledBasicChunk,
  which has M×ceil_div(K, sf_vec_size)×1 elements
- Matrix SFB layout is filled internally according to B shape and BlockScaledBasicChunk,
  which has N×ceil_div(K, sf_vec_size)×L elements
- Token ID mapping tensor enables gather operation for A and SFA

Matrix A/C Memory Layout Diagrams:

   ```
    Group 0    Group 1   Group 2
   -+---------+---------+---------+
    |         |         |         |
   K| ValidM0 | ValidM1 | ValidM2 |
    |         |         |         |
   -+---------+---------+---------+
    |<-        ValidM           ->|
   ```
   Note: the Group(L) dimension will be flatted into M dimension, and the rest Group(L) size is 1.
         each ValidM will be aligned to 256 or 128. The alignment is determined by the mma_tiler_mn parameter.
         For NVFP4, 2CTA, the alignment is 256. For NVFP4, 1CTA, the alignment is 128.

This GEMM kernel supports the following features:
    - Uses asynchronous global-to-shared copies for A and SFA with gather
    - Utilizes Tensor Memory Access (TMA) for B and SFB matrices
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. SCHEDULER warp (warp 10): Dispatches tile information to all consumer warps via tile_info_pipeline.
2. A/SFA load warps (warps 4-7):
    - Load A from global memory (GMEM) to shared memory (SMEM) with gather.
    - Load SFA (scale factor A) from GMEM to SMEM.
    - Uses token_id_mapping to perform permutation/gather during load.
3. TMA B/SFB warp (warp 9):
    - Load B and SFB matrices from GMEM to SMEM using TMA operations with multicast.
4. MMA warp (warp 8):
    - Load SFA from shared memory (SMEM) to tensor memory (TMEM) using tcgen05.cp.
    - Load SFB from shared memory (SMEM) to tensor memory (TMEM) using tcgen05.cp.
    - Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
5. EPILOGUE warps (warps 0-3):
    - Load two accumulator subtiles (up and gate) from tensor memory (TMEM) to registers (RMEM).
    - Apply alpha scaling: up_scaled = alpha * up, gate_scaled = alpha * gate
    - Compute the configured FC1 activation
    - If c_dtype is Float4E2M1FN: generate scale factor C (SFC) and quantize output
    - Type convert output to c_dtype.
    - Store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations.

SM100 tcgen05.mma.kind.block_scale instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Read scalefactor A from TMEM
- Read scalefactor B from TMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Constraints:
* Supported input data types: mxf8, mxf4, nvf4
  see detailed valid dtype combinations in below Sm100BlockScaledPersistentDenseGemmKernel class documentation
* A/B tensor must have the same data type, or form a mixed MXFP8 x MXFP4 pair
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 64/128/192/256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.

CUDA Graph Support:
* For CUDA graph support, the tile_idx_to_expert_idx, token_id_mapping, A/C matrices,
  and scale factor A can be padded to a larger size
  (e.g., permuted_m = m*topK + num_local_experts*(256-1),
  example: 4096*8 + (256/32)*255 = 34808)
* Use create_tensors() with permuted_m parameter to automatically pad:
  - tile_idx_to_expert_idx: padded for invalid tiles (set to -2e9 for padding tiles)
  - token_id_mapping: padded to permuted_m size (invalid tokens set to -1)
  - A matrix: padded to permuted_m rows (padding rows contain dummy data)
  - C matrix: padded to permuted_m rows (output buffer for cuda_graph)
  - Scale factor A: padded to match A matrix dimensions
* Kernel handling of padding:
  - Scheduler warp checks if tile_idx >= num_non_exiting_tiles to exit
  - Only valid tiles (tile_idx < num_non_exiting_tiles) are written to tile_info pipeline
  - A/SFA load warps use token_id_mapping predicates to skip invalid tokens (token_id == -1)
  - When no more valid tiles exist, outer loop exits and calls producer_tail()
  - Consumer warps process only valid tiles from pipeline
  - No deadlock or synchronization issues
* Consumer warps check initial tile against num_non_exiting_tiles and set
  is_valid_tile=False if tile_idx >= num_non_exiting_tiles
* Only rows within (aligned_groupm[0]+aligned_groupm[1]+...) contain valid data
* Padding rows in C matrix will not be written by the kernel
"""


# TODO: Remove this hook helper function after nvidia-cutlass-dsl 4.3.x is no longer supported.
def hooked_PersistentTileSchedulerParams_init(
    self,
    problem_shape_ntile_mnl: cute.Shape,
    cluster_shape_mnk: cute.Shape,
    swizzle_size: int = 1,
    raster_along_m: bool = True,
    *,
    loc=None,
    ip=None,
):
    if cluster_shape_mnk[2] != 1:
        raise ValueError(f"unsupported cluster_shape_k {cluster_shape_mnk[2]}")
    if swizzle_size < 1:
        raise ValueError(f"expect swizzle_size >= 1, but get {swizzle_size}")

    self.problem_shape_ntile_mnl = problem_shape_ntile_mnl
    # cluster_shape_mnk is kept for reconstruction
    self._cluster_shape_mnk = cluster_shape_mnk
    self.cluster_shape_mn = cluster_shape_mnk[:2]
    self.swizzle_size = swizzle_size
    self._raster_along_m = raster_along_m
    self._loc = loc

    problem_shape_ncluster_mnl = cute.ceil_div(
        self.problem_shape_ntile_mnl, cluster_shape_mnk[:2], loc=loc, ip=ip
    )

    # Apply swizzle if swizzle_size > 1
    if swizzle_size > 1:
        problem_shape_ncluster_mnl = cute.round_up(
            problem_shape_ncluster_mnl,
            (1, swizzle_size, 1) if raster_along_m else (swizzle_size, 1, 1),
        )

        if raster_along_m:
            self.problem_layout_ncluster_mnl = cute.make_layout(
                (
                    problem_shape_ncluster_mnl[0],
                    (swizzle_size, problem_shape_ncluster_mnl[1] // swizzle_size),
                    problem_shape_ncluster_mnl[2],
                ),
                stride=(
                    swizzle_size,
                    (1, swizzle_size * problem_shape_ncluster_mnl[0]),
                    problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],
                ),
                loc=loc,
                ip=ip,
            )
        else:
            self.problem_layout_ncluster_mnl = cute.make_layout(
                (
                    (swizzle_size, problem_shape_ncluster_mnl[0] // swizzle_size),
                    problem_shape_ncluster_mnl[1],
                    problem_shape_ncluster_mnl[2],
                ),
                stride=(
                    (1, swizzle_size * problem_shape_ncluster_mnl[1]),
                    swizzle_size,
                    problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],
                ),
                loc=loc,
                ip=ip,
            )

    # Create FastDivmod divisors (only when swizzle_size == 1 for correctness)
    # FastDivmod assumes simple col-major/row-major layout, incompatible with swizzled layouts
    if swizzle_size == 1:
        if raster_along_m:
            self.problem_layout_ncluster_mnl = cute.make_layout(
                problem_shape_ncluster_mnl,
                stride=(
                    1,
                    problem_shape_ncluster_mnl[0],
                    problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],
                ),
                loc=loc,
                ip=ip,
            )
        else:
            self.problem_layout_ncluster_mnl = cute.make_layout(
                problem_shape_ncluster_mnl,
                stride=(
                    problem_shape_ncluster_mnl[1],
                    1,
                    problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],
                ),
                loc=loc,
                ip=ip,
            )
        problem_layout_size = cute.size(
            self.problem_layout_ncluster_mnl, loc=loc, ip=ip
        )
        cluster_count_m = self.problem_layout_ncluster_mnl.shape[0]
        cluster_count_n = self.problem_layout_ncluster_mnl.shape[1]

        # batch_fdd: Used to map linear_idx to work_unit_id (handles persistent scheduling)
        self.batch_fdd = cute.fast_divmod_create_divisor(
            problem_layout_size, loc=loc, ip=ip
        )

        # cluster_shape_m_fdd: Used to decode work_unit_id to cluster coordinates
        self.cluster_shape_m_fdd = cute.fast_divmod_create_divisor(
            cluster_count_m, loc=loc, ip=ip
        )

        # cluster_shape_n_fdd: Used for the second level decomposition
        self.cluster_shape_n_fdd = cute.fast_divmod_create_divisor(
            cluster_count_n, loc=loc, ip=ip
        )
    else:
        # FastDivmod not applicable with swizzling, set to None
        self.batch_fdd = None
        self.cluster_shape_m_fdd = None
        self.cluster_shape_n_fdd = None


def hooked_get_cluster_work_idx_with_fastdivmod(
    self, current_work_linear_idx: Int32, *, loc=None, ip=None
) -> Tuple[Int32, Int32, Int32]:
    work_iteration, work_unit_id = divmod(
        current_work_linear_idx, self.params.batch_fdd
    )

    if self.params._raster_along_m:
        # raster_along_m=True means column major (m is fastest)
        # First, get cluster_m using cluster_shape_m_fdd
        cluster_n_batch, cluster_m = divmod(
            work_unit_id, self.params.cluster_shape_m_fdd
        )

        # Then decode cluster_n_batch to get cluster_n and batch_l using FastDivmod
        batch_l, cluster_n = divmod(cluster_n_batch, self.params.cluster_shape_n_fdd)
    else:
        # raster_along_m=False means row major (n is fastest)
        # First, get cluster_n using cluster_shape_n_fdd
        cluster_m_batch, cluster_n = divmod(
            work_unit_id, self.params.cluster_shape_n_fdd
        )

        # Then decode cluster_m_batch to get cluster_m and batch_l using FastDivmod
        batch_l, cluster_m = divmod(cluster_m_batch, self.params.cluster_shape_m_fdd)

    return (cluster_m, cluster_n, batch_l)


# Only apply monkey-patches for cutlass < 4.4.0 which lacks swizzle_size/raster_along_m
# support and FastDivmod in PersistentTileSchedulerParams.
# cutlass.__version__ was added in 4.4.0, so its absence indicates an older version.
if not hasattr(cutlass, "__version__"):
    cutlass.utils.PersistentTileSchedulerParams.__init__ = (
        hooked_PersistentTileSchedulerParams_init
    )
    cutlass.utils.StaticPersistentTileScheduler._get_cluster_work_idx_with_fastdivmod = hooked_get_cluster_work_idx_with_fastdivmod


class BlockScaledContiguousGatherGroupedGemmKernel:
    """This class implements contiguous grouped matrix multiplication with gather operation and SwiGLU fusion
    for FC1 layer computation (C = up * silu(gate), where up/gate come from interleaved GEMM result).

    The computation flow:
    1. GEMM: acc = alpha * (SFA * A[token_ids]) * (SFB * B)
    2. Activation: SwiGLU, SiTU, tanh-approximate GeGLU, or ReLU^2
    3. Optional Quant: Generates SFC and quantizes output to NVFP4 or MXFP8

    Note: Output C has N/2 columns for gated activations and N columns otherwise.

    Key Features:
    - Loads A and SFA asynchronously with gather/permutation capability
    - Uses TMA (Tensor Memory Access) for loading B and SFB matrices with multicast
    - Token ID mapping enables efficient gather operation during A/SFA load
    - FC1 activation fusion in the epilogue
    - Optional NVFP4 or MXFP8 output quantization with scale factor generation
    - Warp specialization: Scheduler (warp 10), A Sync Transform (warp 11, only used when
      use_2cta_instrs is True), A/SFA loads (warps 4-7), TMA B/SFB (warp 9), MMA (warp 8),
      Epilogue (warps 0-3), and dedicated tensor-memory load/store warps (warps 11-14
      for the mutually exclusive 1-CTA path)

    :param sf_vec_size: Scalefactor vector size (16 for NVF4, 32 for MXF4/MXF8/mixed).
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N).
        Note: use_2cta_instrs is automatically inferred from mma_tiler_mn[0]
        (True when M=256, False when M=128).
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]
    :param vectorized_f32: Whether to use vectorized f32x2 operations for better performance.
    :type vectorized_f32: bool

    :note: A and B may have the same element type or form a
        mixed MXFP8 x MXFP4 pair. The gathered operand A must be the >=8-bit (FP8) operand;
        in a mixed-precision pair B must be the <8-bit (FP4) operand.

    :note: Supported combinations of A/B data types, SF data type and SF vector size
        (A=FP4 requires B=FP4, since the gathered A path cannot be unpacked):
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16
        - Mixed MXFP8 x MXFP4: A: Float8E5M2/Float8E4M3FN, B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2
        # Note: Float4E2M1FN output includes SFC generation and quantization support for internal testing.
        - Float4E2M1FN with scale factor generation

    :note: Constraints:
        - MMA M must be 128 or 256 (use_2cta_instrs)
        - MMA N may be 8/16/32/64/128/256 when swapped and 128/256
          otherwise. Swapped FP4 requires N >= 32; 2CTA MMA requires N >= 16.
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Also, Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors

    Example:
        >>> # Note: use_2cta_instrs is auto-inferred from mma_tiler_mn[0]
        >>> # (True when M=256, False when M=128)
        >>> gemm = BlockScaledContiguousGatherGroupedGemmKernel(
        ...     sf_vec_size=16,
        ...     mma_tiler_mn=(256, 128),  # use_2cta_instrs=True since M=256
        ...     cluster_shape_mn=(2, 1),
        ...     vectorized_f32=True,
        ... )
        >>> gemm(
        ...     a=a_tensor,
        ...     b=b_tensor,
        ...     c=c_tensor,
        ...     sfa=sfa_tensor,
        ...     sfb=sfb_tensor,
        ...     sfc_tensor=None,
        ...     norm_const_tensor=None,
        ...     tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        ...     tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        ...     token_id_mapping_tensor=token_id_mapping_tensor,
        ...     num_non_exiting_tiles=num_non_exiting_tiles,
        ...     alpha=alpha,
        ...     max_active_clusters=max_active_clusters,
        ...     stream=stream,
        ... )
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        topk: cutlass.Int64,
        raster_along_m: bool = False,
        pdl_count: Optional[int] = -1,
        split_k: int = 1,
        gated: bool = True,
        swap_ab: bool = False,
        weight_interleave: Optional[int] = None,
        use_alpha: bool = True,
        use_bias: bool = False,
        activation_type: Optional[int] = None,
        swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
        swiglu_beta: float = DEFAULT_SWIGLU_BETA,
        swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
        use_a_per_token_scale: bool = False,
        use_compact_sfc: bool = True,
        bias_expert_stride_factor: int = 1,
    ):
        """Initializes the configuration for a Blackwell blockscaled dense GEMM kernel with
        gather operation and FC1 activation fusion.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Automatically inferred from mma_tiler_mn[0]
              (True when M=256, False when M=128).

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        3.  Scale Factor Configuration:
            - sf_vec_size: Vector size for block-scaled quantization.

        4.  Performance Optimization:
            - vectorized_f32: Enable vectorized f32x2 operations.

        5.  MoE Configuration:
            - topk: Number of experts selected per token (used for token ID mapping).

        :param sf_vec_size: Vector size for scale factors (16 for NVF4, 32 for MXF4/MXF8/mixed).
        :type sf_vec_size: int
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
            use_2cta_instrs is automatically set based on M (True if M=256, False if M=128).
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param vectorized_f32: Enable vectorized f32x2 operations for better performance.
        :type vectorized_f32: bool
        :param topk: Number of experts selected per token (used for token ID mapping).
        :type topk: cutlass.Int64
        :param raster_along_m: If True, raster persistent tiles along the M dimension.
            Swapped 2CTA kernels always use M rasterization for cache locality.
        :type raster_along_m: bool
        :param pdl_count: Persistent K-tile index at which to launch dependent
            grids. The index advances across scheduler work tiles. None disables
            PDL, while -1 releases dependent grids when the kernel completes.
        :type pdl_count: Optional[int]
        :param split_k: Number of cluster-local K partitions. Values greater
            than one are supported by the low-token swap_ab path for both
            1CTA and 2CTA MMA instructions.
        :type split_k: int
        :param gated: Whether GEMM1 output is split into up/gate halves. If
            False, the epilogue computes non-gated ReLU^2.
        :type gated: bool
        :param swap_ab: Whether to swap the MMA-A/MMA-B assignments and M/N roles.
        :type swap_ab: bool
        :param weight_interleave: Physical up/gate interleave. Unswapped mode
            supports 16 or 64; swapped mode supports 16.
        :type weight_interleave: Optional[int]
        :param use_alpha: Whether to apply a per-expert alpha scale.
        :type use_alpha: bool
        :param use_bias: Whether to apply per-expert branch bias.
        :type use_bias: bool
        :param activation_type: FC1 activation type. Use ActivationType.Swiglu
            for gated SwiGLU/OAI/SiTU, ActivationType.GegluTanh for
            tanh-approximate GeGLU, and ActivationType.Relu2 for non-gated
            ReLU^2. Setting situ_beta selects SiTU.
        :type activation_type: Optional[int]
        :param swiglu_alpha: Sigmoid multiplier for parameterized SwiGLU.
        :type swiglu_alpha: float
        :param swiglu_beta: Up-projection bias for parameterized SwiGLU.
        :type swiglu_beta: float
        :param swiglu_limit: Clamp limit for parameterized SwiGLU.
        :type swiglu_limit: float
        :param situ_beta: When set, use the SiTU gate
            ``beta * tanh(gate / beta) * sigmoid(gate)`` instead of SwiGLU.
        :type situ_beta: Optional[float]
        :param situ_linear_beta: Optional SiTU tanh clamp for the up branch.
        :type situ_linear_beta: Optional[float]
        :param use_a_per_token_scale: Whether operand A has an additional
            per-token row scale.
        :type use_a_per_token_scale: bool
        :param use_compact_sfc: Whether swapped output scales use the compact
            row-major SFC layout. Ignored when swap_ab is False.
        :type use_compact_sfc: bool
        """

        self.sf_vec_size = sf_vec_size
        self.pdl_count = pdl_count
        self.topk = topk
        self.gated = gated
        self.acc_dtype = cutlass.Float32
        self.swap_ab = swap_ab
        weight_interleave = normalize_cute_dsl_moe_weight_interleave(
            weight_interleave, swap_ab
        )
        self.weight_interleave = weight_interleave
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_alpha = use_alpha
        self.use_bias = use_bias
        if activation_type is None:
            activation_type = ActivationType.Swiglu if gated else ActivationType.Relu2
        activation_type, expected_gated = normalize_cute_dsl_moe_activation_type(
            activation_type
        )
        if gated != expected_gated:
            raise ValueError(
                f"gated={gated} is inconsistent with activation_type {activation_type!r}"
            )
        if use_bias and not gated:
            raise ValueError("branch bias is only supported by the gated path")
        validate_cute_dsl_moe_situ_config(activation_type, situ_beta, situ_linear_beta)
        self.activation_type = int(activation_type)
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        if use_a_per_token_scale and swap_ab:
            raise ValueError("per-token activation scales do not support swap_ab")
        self.use_a_per_token_scale = use_a_per_token_scale
        self.use_compact_sfc = swap_ab and use_compact_sfc
        self.bias_expert_stride_factor = bias_expert_stride_factor
        self.out_n_factor = 2 if gated else 1
        mma_m = mma_tiler_mn[1] if swap_ab else mma_tiler_mn[0]
        mma_n = mma_tiler_mn[0] if swap_ab else mma_tiler_mn[1]
        self.use_2cta_instrs = mma_m == 256
        self.split_k = split_k
        # Swapped 2CTA uses M rasterization for weight-cache locality.
        self.raster_along_m = raster_along_m or (swap_ab and self.use_2cta_instrs)
        if split_k < 1 or split_k > 16 or not is_power_of_2(split_k):
            raise ValueError(
                "split_k must be a positive power of two no greater than 16, "
                f"got {split_k}"
            )
        cluster_size = (
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1] * self.split_k
        )
        if cluster_size > 16:
            raise ValueError(
                "cluster_shape_mn * split_k must contain no more than 16 "
                f"CTAs, got {cluster_size}"
            )
        if self.split_k > 1 and (not swap_ab or (self.split_k > 4 and mma_n > 32)):
            raise ValueError(
                "split_k > 1 requires swap_ab; factors larger than four "
                "support token tiles up to 32"
            )
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.async_copy_a_warp_id = (
            4,
            5,
            6,
            7,
        )
        self.mma_warp_id = 8
        self.tma_b_warp_id = 9
        self.sched_warp_id = 10
        self.sync_transform_warp_id = 11
        self.threads_per_warp = 32
        active_sync_transform_warps = (
            (self.sync_transform_warp_id,) if self.use_2cta_instrs else ()
        )
        active_warp_ids = (
            self.mma_warp_id,
            *self.async_copy_a_warp_id,
            self.tma_b_warp_id,
            *self.epilog_warp_id,
            self.sched_warp_id,
            *active_sync_transform_warps,
        )
        self.threads_per_cta = self.threads_per_warp * (max(active_warp_ids) + 1)
        self.warps_wo_sched = (
            len(
                (
                    *self.epilog_warp_id,
                    self.mma_warp_id,
                    self.tma_b_warp_id,
                    self.sync_transform_warp_id,
                    *self.async_copy_a_warp_id,
                )
            )
            if self.use_2cta_instrs
            else len(
                (
                    *self.epilog_warp_id,
                    self.mma_warp_id,
                    self.tma_b_warp_id,
                    *self.async_copy_a_warp_id,
                )
            )
        )
        self.threads_wo_sched = self.threads_per_warp * self.warps_wo_sched

        # Set barrier for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_barrier = NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = NamedBarrier(
            barrier_id=3,
            num_threads=32
            * (
                len((self.mma_warp_id, *self.epilog_warp_id))
                + (1 if self.swap_ab and self.use_2cta_instrs else 0)
            ),
        )
        self.sched_sync_barrier = NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )
        self.async_copy_sync_barrier = NamedBarrier(
            barrier_id=5,
            num_threads=self.threads_per_warp * len(self.async_copy_a_warp_id),
        )
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.vectorized_f32 = vectorized_f32

    def setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """
        if cutlass.const_expr(not self.swap_ab):
            mma_m, mma_n = self.mma_tiler[0], self.mma_tiler[1]
            self.smem_alloc_a_dtype = (
                cutlass.Int8
                if self.mxf8f6f4 and self.a_dtype.width < 8
                else self.a_dtype
            )
            self.smem_alloc_b_dtype = (
                cutlass.Int8
                if self.mxf8f6f4 and self.b_dtype.width < 8
                else self.b_dtype
            )
        else:
            mma_m, mma_n = self.mma_tiler[1], self.mma_tiler[0]
            self.smem_alloc_a_dtype = (
                cutlass.Int8
                if self.mxf8f6f4 and self.b_dtype.width < 8
                else self.b_dtype
            )
            self.smem_alloc_b_dtype = (
                cutlass.Int8
                if self.mxf8f6f4 and self.a_dtype.width < 8
                else self.a_dtype
            )

        self.mma_inst_shape_mn = (mma_m, mma_n)
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        # Configure tiled mma
        tiled_mma = blackwell_helpers.make_blockscaled_trivial_tiled_mma(
            self.b_dtype if self.swap_ab else self.a_dtype,
            self.a_dtype if self.swap_ab else self.b_dtype,
            self.b_major_mode if self.swap_ab else self.a_major_mode,
            self.a_major_mode if self.swap_ab else self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = blackwell_helpers.make_blockscaled_trivial_tiled_mma(
            self.b_dtype if self.swap_ab else self.a_dtype,
            self.a_dtype if self.swap_ab else self.b_dtype,
            self.b_major_mode if self.swap_ab else self.a_major_mode,
            self.a_major_mode if self.swap_ab else self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.mma_tiler_sfa = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k // self.sf_vec_size,
        )

        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.mma_tiler_c = (
            self.mma_inst_shape_mn[0] // (self.out_n_factor if self.swap_ab else 1),
            self.mma_inst_shape_mn[1] // (1 if self.swap_ab else self.out_n_factor),
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cta_tile_shape_mnk_sfa = (
            self.mma_tiler_sfa[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfa[1],
            self.mma_tiler_sfa[2],
        )

        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        self.cta_tile_shape_mnk_c = (
            self.mma_tiler_c[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_c[1],
            self.mma_tiler_c[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, self.split_k)),
            (tiled_mma.thr_id.shape,),
        )

        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, self.split_k)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        if cutlass.const_expr(not self.swap_ab):
            self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        else:
            self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[2])
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # N=192 needs 32-column output tiles to preserve 16-row pairs.
        epi_weight = 128 if self.swap_ab and not self.gated else 64
        n192_interleaved_epilogue = (
            not self.swap_ab
            and self.gated
            and self.weight_interleave == 16
            and self.mma_inst_shape_mn[1] == 192
        )
        if cutlass.const_expr(n192_interleaved_epilogue):
            epi_weight = 32
        if cutlass.const_expr(not self.swap_ab):
            epi_token = min(128, self.cta_tile_shape_mnk_c[0])
            self.epi_tile = (epi_token, epi_weight)
        else:
            epi_token = min(32, self.cta_tile_shape_mnk_c[1])
            self.epi_tile = (epi_weight, epi_token)
        self.epi_tile_cnt = (
            self.cta_tile_shape_mnk_c[0] // self.epi_tile[0],
            self.cta_tile_shape_mnk_c[1] // self.epi_tile[1],
        )
        self.sfc_exchange_elems = (
            cute.size(self.epi_tile) // self.sf_vec_size if self.generate_sfc else 1
        )
        # Each split-K mailbox holds one epilogue fragment and both gates.
        self.split_k_values_per_thread = (
            cute.size(self.epi_tile) * self.out_n_factor
        ) // (self.threads_per_warp * len(self.epilog_warp_id))
        self.split_k_mailbox_elems = (
            (self.split_k - 1)
            * self.threads_per_warp
            * len(self.epilog_warp_id)
            * self.split_k_values_per_thread
        )
        self.split_k_reduction_bytes = (
            self.split_k_mailbox_elems * cutlass.Float32.width // 8
            + (16 if self.split_k > 1 else 0)
        )
        # Setup A/B/C/Scale stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_tile_stage,
        ) = self.compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.smem_alloc_a_dtype,
            self.smem_alloc_b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.num_smem_capacity
            - self.sfc_exchange_elems * cutlass.Float32.width // 8
            - self.split_k_reduction_bytes
            - 1024,
            self.occupancy,
            self.distributed_split_k,
        )
        if cutlass.const_expr(not self.distributed_split_k and self.num_c_stage == 15):
            self.num_c_stage = 14
        # Compute A/B/C/Scale shared memory layout
        self.a_smem_layout_staged = blackwell_helpers.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.smem_alloc_a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = blackwell_helpers.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.smem_alloc_b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_layout.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_layout.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )

        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        self.c_smem_layout_staged = (
            blackwell_helpers.make_smem_layout_epi(
                self.c_dtype,
                self.c_layout,
                self.epi_tile,
                self.num_c_stage,
            )
            if not self.distributed_split_k
            else None
        )

        # Overlap and double buffer accumulator when num_acc_stage == 1 for cta_tile_n = 256 case
        self.overlapping_accum = not self.swap_ab and self.num_acc_stage == 1

        # Compute number of TMEM columns for SFA/SFB/Accumulator
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

        self.epi_tile_n_required = self.out_n_factor * cute.size(self.epi_tile[1])
        # Only when overlapping_accum is enabled, we need to release accumulator buffer early in epilogue
        self.iter_acc_early_release_in_epilogue = (
            self.num_sf_tmem_cols // self.epi_tile_n_required
        )

        self.a_elements_per_async_copy = 128 // self.a_dtype.width

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        sfc_tensor: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        bias_up_tensor: Optional[cute.Tensor],
        bias_gate_tensor: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha: Optional[cute.Tensor],
        a_per_token_scale: Optional[cute.Tensor],
        max_active_clusters: cutlass.Constexpr,
        stream: driver.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the contiguous grouped GEMM with gather operation and SwiGLU fusion.

        This method performs FC1 layer computation:
        1. GEMM: acc = alpha * (SFA * A[token_ids]) * (SFB * B)
        2. SwiGLU: C = up * silu(gate), using the configured weight interleave
        3. Optional Quant: When c_dtype is Float4E2M1FN, generates SFC and quantizes output

        Data loading:
        - A and SFA are loaded asynchronously with token-based gather
        - B and SFB are loaded using TMA instructions with multicast
        - B weights use the configured 16- or 64-row up/gate interleave

        Execution steps:
        1. Setup static attributes before smem/grid computation
        2. Setup TMA load/store atoms for B, SFB, and C (no TMA for A/SFA)
        3. Compute grid size with regard to hardware constraints
        4. Define shared storage for kernel
        5. Launch the kernel synchronously with warp specialization:
           - Scheduler warp: Dispatches tile information
           - A/SFA load warps: Load A and SFA with gather
           - A Sync Transform warps: Transform the sync signal of A and SFA from global to
             shared memory when use_2cta_instrs is True
           - TMA warp: Load B and SFB with multicast
           - MMA warp: Perform matrix multiply-accumulate
           - Epilogue warps: Apply SwiGLU activation, optional quantization, and store results

        :param a: Input tensor A (MxKx1), will be gathered using token_id_mapping
        :type a: cute.Tensor
        :param b: Input tensor B (NxKxL), L is the number of experts/groups, weights are interleaved for SwiGLU
        :type b: cute.Tensor
        :param c: Output tensor C (Mx(N/2)x1), N is halved due to SwiGLU fusion
        :type c: cute.Tensor
        :param sfa: Scale factor tensor A, will be gathered using token_id_mapping
        :type sfa: cute.Tensor
        :param sfb: Scale factor tensor B
        :type sfb: cute.Tensor
        :param sfc_tensor: Scale factor tensor C for quantized output (None if not quantizing)
        :type sfc_tensor: Optional[cute.Tensor]
        :param norm_const_tensor: Normalization constant for scale factor generation
            (None if not quantizing)
        :type norm_const_tensor: Optional[cute.Tensor]
        :param tile_idx_to_expert_idx: Mapping from tile index to expert ID,
            shape (permuted_m/cta_tile_m,) where cta_tile_m is the CTA tile M size
        :type tile_idx_to_expert_idx: cute.Tensor
        :param tile_idx_to_mn_limit: Mapping from tile index to M-N dimension limit
            for boundary checking, shape (permuted_m/cta_tile_m,)
        :type tile_idx_to_mn_limit: cute.Tensor
        :param token_id_mapping_tensor: Token ID mapping for gather operation, shape (permuted_m,)
        :type token_id_mapping_tensor: cute.Tensor
        :param num_non_exiting_tiles: Number of valid tiles to process (valid_m/cta_tile_m), shape (1,)
        :type num_non_exiting_tiles: cute.Tensor
        :param alpha: Alpha tensor for each group
        :type alpha: Optional[cute.Tensor]
        :param bias_up_tensor: Optional bias tensor for up activation.
        :type bias_up_tensor: Optional[cute.Tensor]
        :param bias_gate_tensor: Optional bias tensor for gate activation.
        :type bias_gate_tensor: Optional[cute.Tensor]
        :param a_per_token_scale: Optional per-token row scale for operand A.
        :type a_per_token_scale: Optional[cute.Tensor]
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: driver.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.generate_sfc = sfc_tensor is not None
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.fp4_swap = self.swap_ab and self.c_dtype == cutlass.Float4E2M1FN
        self.distributed_split_k = (
            self.split_k > 1
            and self.swap_ab
            and self.gated
            and self.generate_sfc
            and self.c_dtype.width >= 8
        )
        if cutlass.const_expr(self.swap_ab):
            tokens, interm, rest_l = c.shape[0], c.shape[1], c.shape[2]
            # (M=feature, N=token, L)
            c = cute.make_tensor(
                c.iterator,
                cute.make_layout(
                    (interm, tokens, rest_l),
                    stride=(1, interm, interm * tokens),
                ),
            )
            if cutlass.const_expr(self.use_bias):
                bias_l = bias_up_tensor.shape[2]
                bias_expert_stride = interm * self.bias_expert_stride_factor
                # (M=feature, N=token, L)
                bias_up_tensor = cute.make_tensor(
                    bias_up_tensor.iterator,
                    cute.make_layout(
                        (interm, tokens, bias_l), stride=(1, 0, bias_expert_stride)
                    ),
                )
                # (M=feature, N=token, L)
                bias_gate_tensor = cute.make_tensor(
                    bias_gate_tensor.iterator,
                    cute.make_layout(
                        (interm, tokens, bias_l), stride=(1, 0, bias_expert_stride)
                    ),
                )

        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self.mxf8f6f4 = self.needs_unpack_tma(self.a_dtype, self.b_dtype)

        # Check if input data types are compatible with the MMA instruction.
        SUPPORTED_DTYPES = (
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        )
        if cutlass.const_expr(
            self.a_dtype not in SUPPORTED_DTYPES or self.b_dtype not in SUPPORTED_DTYPES
        ):
            raise TypeError(
                f"Unsupported data types: A={self.a_dtype}, B={self.b_dtype}; "
                f"expected in {SUPPORTED_DTYPES}"
            )
        if cutlass.const_expr(self.mxf8f6f4 and self.a_dtype.width < 8):
            raise TypeError(
                f"A=FP4 is only valid when B is FP4, but got A={self.a_dtype} and B={self.b_dtype}"
            )
        if cutlass.const_expr(
            cutlass.const_expr(
                self.swiglu_limit is not None
                or self.swiglu_alpha != 1.0
                or self.swiglu_beta != 0.0
            )
            and not self.gated
        ):
            print("Warning: SwiGLU is not supported when gated is False")

        # Setup attributes that dependent on gemm inputs
        self.setup_attributes()

        if cutlass.const_expr(not self.swap_ab):
            # Setup sfb tensor by filling B tensor to scale factor atom layout
            # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
            sfb_layout = blockscaled_layout.tile_atom_to_shape_SF(
                b.shape, self.sf_vec_size
            )
            weights_sf = cute.make_tensor(sfb.iterator, sfb_layout)
        else:
            sfa_layout = blockscaled_layout.tile_atom_to_shape_SF(
                b.shape, self.sf_vec_size
            )
            weights_sf = cute.make_tensor(sfb.iterator, sfa_layout)

        if cutlass.const_expr(self.generate_sfc):
            if cutlass.const_expr(self.use_compact_sfc):
                output_shape = (tokens, interm, rest_l) if self.swap_ab else c.shape
                # (token, feature // sf_vec_size, L)
                sfc_layout = compact_sf_layout(output_shape, self.sf_vec_size)
            else:
                # ((Atom_M, Rest_M), (Atom_K, Rest_K), RestL) layout.
                output_shape = (tokens, interm, rest_l) if self.swap_ab else c.shape
                sfc_layout = blockscaled_layout.tile_atom_to_shape_SF(
                    output_shape, self.sf_vec_size
                )
            sfc_tensor = cute.make_tensor(sfc_tensor.iterator, sfc_layout)

        tiled_mma = blackwell_helpers.make_blockscaled_trivial_tiled_mma(
            b.element_type if self.swap_ab else a.element_type,
            a.element_type if self.swap_ab else b.element_type,
            (
                utils.LayoutEnum.from_tensor(b).mma_major_mode()
                if self.swap_ab
                else utils.LayoutEnum.from_tensor(a).mma_major_mode()
            ),
            (
                utils.LayoutEnum.from_tensor(a).mma_major_mode()
                if self.swap_ab
                else utils.LayoutEnum.from_tensor(b).mma_major_mode()
            ),
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        # For 2CTA blockscaled kernels, SFB needs to be replicated across peer CTAs.
        tiled_mma_sfb = blackwell_helpers.make_blockscaled_trivial_tiled_mma(
            b.element_type if self.swap_ab else a.element_type,
            a.element_type if self.swap_ab else b.element_type,
            (
                utils.LayoutEnum.from_tensor(b).mma_major_mode()
                if self.swap_ab
                else utils.LayoutEnum.from_tensor(a).mma_major_mode()
            ),
            (
                utils.LayoutEnum.from_tensor(a).mma_major_mode()
                if self.swap_ab
                else utils.LayoutEnum.from_tensor(b).mma_major_mode()
            ),
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        tiled_mma_epilogue = tiled_mma
        if cutlass.const_expr(self.swap_ab and self.gated and self.use_2cta_instrs):
            tiled_mma_epilogue = blackwell_helpers.make_blockscaled_trivial_tiled_mma(
                b.element_type,
                a.element_type,
                utils.LayoutEnum.from_tensor(b).mma_major_mode(),
                utils.LayoutEnum.from_tensor(a).mma_major_mode(),
                self.sf_dtype,
                self.sf_vec_size,
                cute.nvgpu.tcgen05.CtaGroup.ONE,
                (self.mma_inst_shape_mn[0] // 2, self.mma_inst_shape_mn[1]),
            )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        if cutlass.const_expr(not self.swap_ab):
            # Setup TMA load for B
            b_op = blackwell_helpers.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            b_smem_layout = cute.slice_(
                self.b_smem_layout_staged, (None, None, None, 0)
            )
            tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
                b_op,
                b,
                b_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=(
                    self.smem_alloc_b_dtype
                    if (self.mxf8f6f4 and self.b_dtype.width < 8)
                    else None
                ),
            )
        else:
            # Setup TMA load for A
            a_op = blackwell_helpers.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            b_smem_layout = cute.slice_(
                self.a_smem_layout_staged, (None, None, None, 0)
            )
            tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_A(
                a_op,
                b,
                b_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=(
                    self.smem_alloc_a_dtype
                    if (self.mxf8f6f4 and self.b_dtype.width < 8)
                    else None
                ),
            )

        if cutlass.const_expr(not self.swap_ab):
            sfb_op = blackwell_helpers.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            sfb_smem_layout = cute.slice_(
                self.sfb_smem_layout_staged, (None, None, None, 0)
            )
            tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                weights_sf,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                tiled_mma_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )
        else:
            sfa_op = blackwell_helpers.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            sfb_smem_layout = cute.slice_(
                self.sfa_smem_layout_staged, (None, None, None, 0)
            )
            tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op,
                weights_sf,
                sfb_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

        # This modifies the layout to handle overlapping 256x(# of scale factors for a single column of B (nNSF))
        # logical blocks for SFB when cta_tile_shape_n=192.
        if cutlass.const_expr(not self.swap_ab and self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
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

        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (b_copy_size + sfb_copy_size) * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        if cutlass.const_expr(not self.distributed_split_k):
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c,
                epi_smem_layout,
                self.epi_tile,
            )

        # Compute grid size
        self.tile_sched_params, grid = self.compute_grid(
            c,
            self.cta_tile_shape_mnk_c,
            self.cluster_shape_mn,
            max_active_clusters,
            self.raster_along_m,
        )
        grid = (grid[0], grid[1], grid[2] * self.split_k)

        self.buffer_align_bytes = 1024
        c_smem_elems = (
            cute.cosize(self.c_smem_layout_staged.outer)
            if not self.distributed_split_k
            else 0
        )

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage1cta:
            # (bidx, bidy, bidz, valid, mn_limit)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            split_k_reduce_mbar_ptr: cutlass.Int64
            split_k_consumed_mbar_ptr: cutlass.Int64
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            split_k_mailbox: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.split_k_mailbox_elems],
                16,
            ]
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    c_smem_elems,
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.smem_alloc_a_dtype,
                    cute.cosize(self.a_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.smem_alloc_b_dtype,
                    cute.cosize(self.b_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFCExchange: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.sfc_exchange_elems],
                16,
            ]

        @cute.struct
        class SharedStorage2cta:
            # (bidx, bidy, bidz, valid, mn_limit)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            a_sync_transform_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            b_sync_transform_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            sfb_tmem_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            split_k_reduce_mbar_ptr: cutlass.Int64
            split_k_consumed_mbar_ptr: cutlass.Int64
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            split_k_mailbox: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.split_k_mailbox_elems],
                16,
            ]
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    c_smem_elems,
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.smem_alloc_a_dtype,
                    cute.cosize(self.a_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.smem_alloc_b_dtype,
                    cute.cosize(self.b_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFCExchange: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.sfc_exchange_elems],
                16,
            ]

        self.shared_storage = (
            SharedStorage2cta
            if cutlass.const_expr(self.use_2cta_instrs)
            else SharedStorage1cta
        )

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tiled_mma_epilogue,
            a,
            tma_tensor_b,
            tma_tensor_c if not self.distributed_split_k else c,
            sfa,
            tma_tensor_sfb,
            sfc_tensor,
            norm_const_tensor,
            bias_up_tensor,
            bias_gate_tensor,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping_tensor,
            num_non_exiting_tiles,
            alpha,
            a_per_token_scale,
            tma_atom_b,
            tma_atom_sfb,
            tma_atom_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, self.split_k),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[union-attr]
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.pdl_count is not None,
        )
        return

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to
        partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tiled_mma_epilogue: cute.TiledMma,
        mA_mkl: cute.Tensor,
        mB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        mSFB_nkl: cute.Tensor,
        mSFC_mnl: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        mBiasUp_mnl: Optional[cute.Tensor],
        mBiasGate_mnl: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha: Optional[cute.Tensor],
        a_per_token_scale: Optional[cute.Tensor],
        tma_atom_b: cute.CopyAtom,
        tma_atom_sf: cute.CopyAtom,
        tma_atom_c: Optional[cute.CopyAtom],
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_b_warp_id:
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sf)
            if cutlass.const_expr(not self.distributed_split_k):
                cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        gdx, gdy, gdz = cute.arch.grid_dim()
        _, _, split_rank = cute.arch.block_in_cluster_idx()
        scheduler_block_idx = (bidx, bidy, bidz // self.split_k)
        scheduler_grid_dim = (gdx, gdy, gdz // self.split_k)
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        # Preserve the CTA's M/N lane when selecting a split-Z DSMEM peer.
        split_peer_stride = cutlass.Int32(cute.size(self.cluster_shape_mn))
        split_peer_base = cta_rank_in_cluster - split_rank * split_peer_stride
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        if cutlass.const_expr(not self.swap_ab):
            async_copy_mbar = storage.a_mbar_ptr.data_ptr()
            tma_mbar = storage.b_mbar_ptr.data_ptr()
        else:
            async_copy_mbar = storage.b_mbar_ptr.data_ptr()
            tma_mbar = storage.a_mbar_ptr.data_ptr()
        num_tma_producer = self.num_mcast_ctas_b

        # Pipeline Init: Initialize the asynchronous A load pipeline
        # Producer: 4 warps (warps 4-7) with 128 threads total
        # Consumer: MMA warp for consuming A/SFA data
        # If swap_ab, use the B pipeline for asynchronous loads
        a_pipeline_producer_group = CooperativeGroup(
            Agent.Thread,
            self.threads_per_warp * 4,
        )
        a_pipeline = PipelineCpAsyncUmma.create(
            barrier_storage=async_copy_mbar,
            num_stages=self.num_ab_stage,
            producer_group=a_pipeline_producer_group,
            consumer_group=CooperativeGroup(Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init: Initialize A SYNC Transform pipeline when use_2cta_instrs is True
        # Producer: 1 warp (warp 11) for load synchronization transformation
        # Consumer: MMA warp for consuming A/SFA data
        if cutlass.const_expr(self.use_2cta_instrs):
            a_sync_transform_mbar = storage.a_sync_transform_mbar_ptr.data_ptr()
            a_sync_transform_pipeline_producer_group = CooperativeGroup(
                Agent.Thread,
                32 * cute.size(cluster_layout_vmnk, mode=[0]),
            )
            a_sync_transform_pipeline = PipelineAsyncUmma.create(
                barrier_storage=a_sync_transform_mbar,
                num_stages=self.num_ab_stage,
                producer_group=a_sync_transform_pipeline_producer_group,
                consumer_group=CooperativeGroup(Agent.Thread),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )

        if cutlass.const_expr(self.swap_ab and self.use_2cta_instrs):
            # The SFB handoff uses the same rotating stage as the A/B operands.
            sfb_tmem_pipeline = PipelineAsyncUmma.create(
                barrier_storage=storage.sfb_tmem_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=CooperativeGroup(
                    Agent.Thread,
                    self.threads_per_warp * cute.size(cluster_layout_vmnk, mode=[0]),
                ),
                consumer_group=CooperativeGroup(Agent.Thread),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )

        # Pipeline Init: Initialize B pipeline for TMA operations
        # Using PipelineTmaUmma for B/SFB since they use TMA load with multicast support
        # Producer: TMA B/SFB warp (warp 9) - 1 warp issuing TMA operations
        # Consumer: MMA warp for consuming B/SFB data
        # If swap_ab, use TMA Umma for B pipeline, MMA warp for consuming A/SFA data
        b_pipeline_producer_group = CooperativeGroup(Agent.Thread)
        b_pipeline_consumer_group = CooperativeGroup(Agent.Thread, num_tma_producer)
        b_pipeline = PipelineTmaUmma.create(
            barrier_storage=tma_mbar,
            num_stages=self.num_ab_stage,
            producer_group=b_pipeline_producer_group,
            consumer_group=b_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Pipeline Init: Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = CooperativeGroup(Agent.Thread)
        num_acc_consumer_threads = (
            len(self.epilog_warp_id)
            * self.threads_per_warp
            * (2 if use_2cta_instrs else 1)
        )
        acc_pipeline_consumer_group = CooperativeGroup(
            Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Pipeline Init:Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = CooperativeGroup(
            Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = CooperativeGroup(
            Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        split_k_reduce_mbar = storage.split_k_reduce_mbar_ptr.ptr
        split_k_consumed_mbar = storage.split_k_consumed_mbar_ptr.ptr
        split_k_mailbox = storage.split_k_mailbox.data_ptr()
        if cutlass.const_expr(self.split_k > 1):
            if warp_idx == self.epilog_warp_id[0]:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(split_k_reduce_mbar, 1)
                    cute.arch.mbarrier_init(
                        split_k_consumed_mbar,
                        (self.split_k - 1 if self.distributed_split_k else 1),
                    )
            cute.arch.mbarrier_init_fence()

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) * self.split_k > 1:
            cute.arch.cluster_arrive_relaxed()

        #
        # Setup smem tensor A/B/C/Scale
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = None
        if cutlass.const_expr(not self.distributed_split_k):
            sC = storage.sC.get_tensor(
                c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
            )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        # (bidx, bidy, bidz, valid, mn_limit)
        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        sInfo = storage.sInfo.get_tensor(info_layout)
        sSFCExchange = storage.sSFCExchange.get_tensor(
            cute.make_layout(self.sfc_exchange_elems)
        )
        #
        # Compute multicast mask for A/B buffer full
        #
        b_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
            if cutlass.const_expr(not self.swap_ab):
                cluster_sf_layout = cluster_layout_sfb_vmnk
                sf_coord = block_in_cluster_coord_sfb_vmnk
                m_cast_mode = 1
            else:
                cluster_sf_layout = cluster_layout_vmnk
                sf_coord = block_in_cluster_coord_vmnk
                m_cast_mode = 2
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=m_cast_mode
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_sf_layout, sf_coord, mcast_mode=m_cast_mode
            )

        #
        # Local_tile partition global tensors
        #
        if cutlass.const_expr(not self.swap_ab):
            a_tile = cute.slice_(self.cta_tile_shape_mnk, (None, 0, None))
            b_tile = cute.slice_(self.mma_tiler, (0, None, None))
            sfa_tile = cute.slice_(self.cta_tile_shape_mnk_sfa, (None, 0, None))
            sfb_tile = cute.slice_(self.mma_tiler_sfb, (0, None, None))
        else:
            a_tile = cute.slice_(self.mma_tiler, (0, None, None))
            b_tile = cute.slice_(self.cta_tile_shape_mnk, (None, 0, None))
            sfa_tile = cute.slice_(self.mma_tiler_sfa, (0, None, None))
            sfb_tile = cute.slice_(self.mma_tiler, (None, 0, None))

        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl,
            a_tile,
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(mB_nkl, b_tile, (None, None, None))

        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            sfa_tile,
            (None, None, None),
        )

        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            sfb_tile,
            (None, None, None),
        )

        if cutlass.const_expr(not self.swap_ab):
            gToken_ml = cute.local_tile(
                token_id_mapping_tensor,
                cute.slice_(self.cta_tile_shape_mnk, (None, 0, 0)),
                (None,),
            )
        else:
            gToken_ml = cute.local_tile(
                token_id_mapping_tensor,
                cute.slice_(self.cta_tile_shape_mnk, (0, None, 0)),
                (None,),
            )

        c_tile = self.mma_tiler_c
        if cutlass.const_expr(self.swap_ab and self.gated and self.use_2cta_instrs):
            # Partition each CTA's half with a 1CTA MMA layout.
            c_tile = self.cta_tile_shape_mnk_c

        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(c_tile, (None, None, 0)), (None, None, None)
        )
        total_k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))
        k_tiles_per_split = (
            total_k_tile_cnt + cutlass.Int32(self.split_k - 1)
        ) // cutlass.Int32(self.split_k)
        k_tile_start = split_rank * k_tiles_per_split
        if k_tile_start > total_k_tile_cnt:
            k_tile_start = total_k_tile_cnt
        k_tile_end = k_tile_start + k_tiles_per_split
        if k_tile_end > total_k_tile_cnt:
            k_tile_end = total_k_tile_cnt
        k_tile_cnt = k_tile_end - k_tile_start

        # (bM, bN, loopM, loopN, loopL)
        gBiasUp_mnl = None
        gBiasGate_mnl = None
        if cutlass.const_expr(self.use_bias):
            gBiasUp_mnl = cute.local_tile(
                mBiasUp_mnl,
                cute.slice_(c_tile, (None, None, 0)),
                (None, None, None),
            )
            gBiasGate_mnl = cute.local_tile(
                mBiasGate_mnl,
                cute.slice_(c_tile, (None, None, 0)),
                (None, None, None),
            )

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        thr_mma_epilogue = tiled_mma_epilogue.get_slice(0)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgC = thr_mma.partition_C(gC_mnl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgBiasUp = None
        tCgBiasGate = None
        if cutlass.const_expr(self.use_bias):
            tCgBiasUp = thr_mma.partition_C(gBiasUp_mnl)
            tCgBiasGate = thr_mma.partition_C(gBiasGate_mnl)
        if cutlass.const_expr(self.swap_ab and self.gated and self.use_2cta_instrs):
            tCgC = thr_mma_epilogue.partition_C(gC_mnl)
            if cutlass.const_expr(self.use_bias):
                tCgBiasUp = thr_mma_epilogue.partition_C(gBiasUp_mnl)
                tCgBiasGate = thr_mma_epilogue.partition_C(gBiasGate_mnl)

        if cutlass.const_expr(not self.swap_ab):
            # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
            tCgB = thr_mma.partition_B(gB_nkl)
            # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
            tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        else:
            # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
            tCgB = thr_mma.partition_A(gB_nkl)
            # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
            tCgSFB = thr_mma.partition_A(gSFB_nkl)
        #
        # Partition global/shared tensor for TMA load B
        #
        # TMA load B partition_S/D
        if cutlass.const_expr(not self.swap_ab):
            weights_smem = sB
            cta_layout = cute.slice_(cluster_layout_vmnk, (0, None, 0, 0))
            coord = block_in_cluster_coord_vmnk[1]
            sf_layout = cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0))
            coord_sf = block_in_cluster_coord_sfb_vmnk[1]
            smem_sf = sSFB
        else:
            weights_smem = sA
            cta_layout = cute.slice_(cluster_layout_vmnk, (0, 0, None, 0))
            coord = block_in_cluster_coord_vmnk[2]
            sf_layout = cute.slice_(cluster_layout_vmnk, (0, 0, None, 0))
            coord_sf = block_in_cluster_coord_vmnk[2]
            smem_sf = sSFA

        cta_layout = cute.make_layout(cta_layout.shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            coord,
            cta_layout,
            cute.group_modes(weights_smem, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # TMA load SFB partition_S/D
        sf_cta_layout = cute.make_layout(sf_layout.shape)

        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sf,
            coord_sf,
            sf_cta_layout,
            cute.group_modes(smem_sf, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
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

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) * self.split_k > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        # Expert weights, activations, and scheduling depend on routing metadata, only setup can be done before this point
        griddepcontrol_wait()

        num_non_exiting_tiles_value = num_non_exiting_tiles[0]
        create_tile_scheduler = (
            DeviceBoundPersistentTileScheduler.create_n
            if self.swap_ab
            else DeviceBoundPersistentTileScheduler.create_static_n
        )
        tile_sched = create_tile_scheduler(
            tile_sched_params,
            scheduler_block_idx,
            scheduler_grid_dim,
            num_non_exiting_tiles_value,
        )

        #
        # Specialized Schedule Warp
        #
        if warp_idx == self.sched_warp_id:
            #
            # Persistent tile scheduling loop
            #
            # First tile
            work_tile = tile_sched.initial_work_tile_info()
            tile_info_producer_state = make_pipeline_state(
                PipelineUserType.Producer, self.num_tile_stage
            )

            if cutlass.const_expr(self.raster_along_m or self.swap_ab):
                while work_tile.is_valid_tile:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    mma_tile_coord_n = cur_tile_coord[1]
                    if cutlass.const_expr(not self.swap_ab):
                        tile_coord = mma_tile_coord_m
                    else:
                        tile_coord = mma_tile_coord_n
                    if tile_coord < num_non_exiting_tiles_value:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        cur_tile_coord = work_tile.tile_idx
                        expert_idx = tile_idx_to_expert_idx[tile_coord]
                        mn_limit = tile_idx_to_mn_limit[tile_coord]
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[
                                0
                            ]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[
                                1
                            ]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                            # fence view async shared
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            else:
                is_continue = cutlass.Boolean(1)
                while work_tile.is_valid_tile and is_continue:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    mma_tile_coord_n = cur_tile_coord[1]
                    if cutlass.const_expr(not self.swap_ab):
                        tile_coord = mma_tile_coord_m
                    else:
                        tile_coord = mma_tile_coord_n
                    if tile_coord < num_non_exiting_tiles_value:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        cur_tile_coord = work_tile.tile_idx
                        expert_idx = tile_idx_to_expert_idx[tile_coord]
                        mn_limit = tile_idx_to_mn_limit[tile_coord]
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[
                                0
                            ]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[
                                1
                            ]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                            # fence view async shared
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()
                    else:
                        is_continue = cutlass.Boolean(0)

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = work_tile.tile_idx[0]
                sInfo[(1, tile_info_producer_state.index)] = work_tile.tile_idx[1]
                sInfo[(2, tile_info_producer_state.index)] = -1
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(4, tile_info_producer_state.index)] = -1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized activation load warps (warps 4-7)
        # These warps load activation and scale factors from global to shared memory
        # with gather/permutation capability enabled by token_id_mapping
        #
        if (
            warp_idx <= self.async_copy_a_warp_id[-1]
            and warp_idx >= self.async_copy_a_warp_id[0]
        ):
            #
            # Setup asynchronous copy atoms for activation and scale factor
            # A uses eight 128-bit copies per thread with a 128-byte swizzle.
            # SFA uses four 32-bit copies per thread with 512-element block swizzling.
            #
            a_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(
                    cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL
                ),
                mA_mkl.element_type,
                num_bits_per_copy=128,
            )
            a_thread_layout = cute.make_layout((16, 8), stride=(8, 1))
            a_value_layout = cute.make_layout(
                (1, self.a_elements_per_async_copy),
                stride=(self.a_elements_per_async_copy, 1),
            )
            a_tiled_copy = cute.make_tiled_copy_tv(
                a_atom_copy,
                a_thread_layout,
                a_value_layout,
            )

            sfa_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mSFA_mkl.element_type,
                num_bits_per_copy=32,
            )
            tidx_in_warpgroup = tidx % 128

            async_copy_token_tile = self.cta_tile_shape_mnk[0]
            if cutlass.const_expr(not self.swap_ab):
                smem_alloc = sA
                layout = cute.make_layout(
                    (
                        self.cta_tile_shape_mnk[0],
                        self.cta_tile_shape_mnk[2],
                        self.num_ab_stage,
                    ),
                    stride=(
                        self.cta_tile_shape_mnk[2],
                        1,
                        self.cta_tile_shape_mnk[0] * self.cta_tile_shape_mnk[2],
                    ),
                )
            else:
                smem_alloc = sB
                async_copy_token_tile = self.cta_tile_shape_mnk[1] // (
                    2 if self.use_2cta_instrs else 1
                )
                layout = cute.make_layout(
                    (
                        async_copy_token_tile,
                        self.cta_tile_shape_mnk[2],
                        self.num_ab_stage,
                    ),
                    stride=(
                        self.cta_tile_shape_mnk[2],
                        1,
                        async_copy_token_tile * self.cta_tile_shape_mnk[2],
                    ),
                )

            sA_tiled = cute.make_tensor(smem_alloc.iterator, layout=layout)
            a_thr_copy = a_tiled_copy.get_slice(tidx_in_warpgroup)
            tAsA_tiled = a_thr_copy.partition_D(sA_tiled)

            if cutlass.const_expr(not self.swap_ab):
                num_rows_per_thread = 8
            else:
                # Token tile lives on CTA N under swap_ab.
                num_rows_per_thread = min(16, -(-async_copy_token_tile // 16))

            a_token_offset_tensor = cute.make_rmem_tensor(
                cute.make_layout((num_rows_per_thread,)),
                cutlass.Int32,
            )
            a_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((num_rows_per_thread,)),
                cutlass.Boolean,
            )
            sfa_token_offset_tensor = cute.make_rmem_tensor(
                cute.make_layout((1,)),
                cutlass.Int32,
            )
            sfa_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((1,)),
                cutlass.Boolean,
            )
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, scheduler_block_idx, scheduler_grid_dim
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()
            pdl_k_tile = cutlass.Int32(0)

            a_producer_state = make_pipeline_state(
                PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = make_pipeline_state(
                PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Load token IDs for gather operation
                # Each thread loads eight A token offsets and one SFA token offset.
                token_cta_offset = cutlass.Int32(0)
                if cutlass.const_expr(not self.swap_ab):
                    tToken_ml = tile_info[0]
                    token_tile = self.cta_tile_shape_mnk[0]
                else:
                    tToken_ml = tile_info[1]
                    token_tile = self.cta_tile_shape_mnk[1]
                    token_cta_offset = mma_tile_coord_v * async_copy_token_tile

                gToken_ml_tile = gToken_ml[(None, tToken_ml)]
                for i in cutlass.range_constexpr(num_rows_per_thread, unroll=1):
                    token_ml_tile_offset = (
                        (tidx_in_warpgroup // 8)
                        + i * 16
                        + (token_cta_offset if cutlass.const_expr(self.swap_ab) else 0)
                    )
                    a_predicate_tensor[i] = (
                        cutlass.Boolean(1)
                        if token_ml_tile_offset
                        < (
                            token_cta_offset + async_copy_token_tile
                            if cutlass.const_expr(self.swap_ab)
                            else token_tile
                        )
                        and tToken_ml * token_tile + token_ml_tile_offset < tile_info[4]
                        else cutlass.Boolean(0)
                    )
                    a_token_offset_tensor[i] = 0
                    if a_predicate_tensor[i]:
                        a_token_offset_tensor[i] = (
                            gToken_ml_tile[token_ml_tile_offset] // self.topk
                        )

                token_ml_tile_offset_local = (
                    8 * (tidx_in_warpgroup // 32)
                    + 32 * ((tidx_in_warpgroup % 32) // 8)
                    + (tidx_in_warpgroup % 8)
                )
                token_ml_tile_offset = token_ml_tile_offset_local
                sfa_predicate_tensor[0] = (
                    cutlass.Boolean(1)
                    if token_ml_tile_offset_local < token_tile
                    and tToken_ml * token_tile + token_ml_tile_offset < tile_info[4]
                    else cutlass.Boolean(0)
                )
                sfa_token_offset_tensor[0] = 0
                if sfa_predicate_tensor[0]:
                    sfa_token_offset_tensor[0] = (
                        gToken_ml_tile[token_ml_tile_offset] // self.topk
                    )
                relative_sfa_token_offset = sfa_token_offset_tensor[0]

                tAgA = gA_mkl[(None, None, 0, None, 0)]
                A_gmem_thread_offset = cute.assume(
                    (tidx_in_warpgroup % 8) * self.a_elements_per_async_copy,
                    divby=self.a_elements_per_async_copy,
                )
                tAgSFA = gSFA_mkl[(relative_sfa_token_offset, None, 0, None, 0)]
                # ((32, 4), 4)
                sfa_atom_layout = cute.filter_zeros(
                    blockscaled_layout.BlockScaledBasicChunk(self.sf_vec_size).layout
                )
                sfa_atom_scales = cute.size(sfa_atom_layout, mode=[1])
                sfa_atom_storage = cute.cosize(sfa_atom_layout)
                n_sfa_blocks = self.cta_tile_shape_mnk_sfa[2] // sfa_atom_scales
                num_sfa_k_tiles = cute.ceil_div(
                    mSFA_mkl.shape[1], self.cta_tile_shape_mnk_sfa[2]
                )
                # (Rest_K, Atom_K)
                sfa_gmem_copy_layout = cute.composition(
                    cute.make_layout(self.cta_tile_shape_mnk_sfa[2]),
                    cute.make_layout(
                        (n_sfa_blocks, sfa_atom_scales),
                        stride=(sfa_atom_scales, 1),
                    ),
                )
                # (loopK, Rest_K)
                sfa_global_offset_layout = cute.composition(
                    cute.make_layout(num_sfa_k_tiles * self.cta_tile_shape_mnk_sfa[2]),
                    cute.make_layout(
                        (num_sfa_k_tiles, n_sfa_blocks),
                        stride=(
                            self.cta_tile_shape_mnk_sfa[2],
                            sfa_atom_scales,
                        ),
                    ),
                )
                # (Rest_K, Atom_K)
                sfa_smem_copy_layout = cute.composition(
                    cute.make_layout(sfa_atom_storage * n_sfa_blocks),
                    cute.make_layout(
                        (n_sfa_blocks, sfa_atom_scales),
                        stride=(sfa_atom_storage, 1),
                    ),
                )

                if cutlass.const_expr(not self.swap_ab):
                    sSF = sSFA
                else:
                    sSF = sSFB

                tAsSFA = sSF[
                    (
                        (
                            (
                                (
                                    8 * (tidx_in_warpgroup // 32)
                                    + (tidx_in_warpgroup % 8),
                                    (tidx_in_warpgroup % 32) // 8,
                                ),
                                None,
                            ),
                            None,
                        ),
                        None,
                        None,
                        None,
                    )
                ]

                # Peek (try_wait) SCALE buffer empty
                a_producer_state.reset_count()
                peek_a_empty_status = cutlass.Boolean(1)
                if a_producer_state.count < k_tile_cnt:
                    peek_a_empty_status = a_pipeline.producer_try_acquire(
                        a_producer_state
                    )

                #
                # Load A and SFA asynchronously with gather/permutation
                # Each K-tile iteration loads one K-tile of A and SFA from GMEM to SMEM
                # using token-based gather addressing
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                    # Conditionally wait for AB buffer empty
                    a_pipeline.producer_acquire(a_producer_state, peek_a_empty_status)

                    global_k_tile = k_tile_start + a_producer_state.count
                    tAgA_ktile = tAgA[(None, None, global_k_tile)]
                    tAgA_flat = cute.make_tensor(
                        tAgA_ktile.iterator,
                        cute.make_layout(mA_mkl.shape[0] * mA_mkl.shape[1]),
                    )
                    tAgA_vectors = cute.logical_divide(
                        tAgA_flat,
                        cute.make_layout(self.a_elements_per_async_copy),
                    )
                    tAsA_ktile = tAsA_tiled[(None, None, None, a_producer_state.index)]

                    tAgSFA_ktile = tAgSFA[(None, global_k_tile)]
                    # (Rest_K, Atom_K)
                    tAgSFA_blocked = cute.make_tensor(
                        tAgSFA_ktile.iterator, sfa_gmem_copy_layout
                    )
                    tAsSFA_ktile = tAsSFA[
                        (
                            None,
                            None,
                            None,
                            None,
                            a_producer_state.index,
                        )
                    ]
                    tAsSFA_blocked = cute.make_tensor(
                        tAsSFA_ktile.iterator, sfa_smem_copy_layout
                    )
                    is_partial_k_tile = (
                        global_k_tile * self.cta_tile_shape_mnk[2]
                        + self.cta_tile_shape_mnk[2]
                        > mA_mkl.shape[1]
                    )
                    compact_swapped_tile = self.swap_ab and async_copy_token_tile < 16
                    initialize_stage = is_partial_k_tile
                    if cutlass.const_expr(compact_swapped_tile):
                        initialize_stage = cutlass.Boolean(True)
                    if initialize_stage:
                        # Zero aliased rows before predicated compact-tile loads.
                        for i in cutlass.range_constexpr(num_rows_per_thread, unroll=1):
                            initialize_a_slice = cutlass.Boolean(True)
                            if cutlass.const_expr(compact_swapped_tile):
                                initialize_a_slice = (
                                    tidx_in_warpgroup // 8 + i * 16
                                    < async_copy_token_tile
                                )
                            if initialize_a_slice:
                                tAsA_slice = cute.make_tensor(
                                    tAsA_ktile[(None, i, None)].iterator,
                                    layout=cute.make_layout(
                                        (self.a_elements_per_async_copy,)
                                    ),
                                )
                                if cutlass.const_expr(
                                    self.smem_alloc_a_dtype.width < 8
                                ):
                                    tAsA_slice = cute.recast_tensor(
                                        tAsA_slice, cutlass.Uint8
                                    )
                                for j in cutlass.range_constexpr(cute.size(tAsA_slice)):
                                    tAsA_slice[j] = tAsA_slice.element_type(0)

                        # Scale partitions alias: row owners initialize them, while
                        # full-tile copies below populate both token halves.
                        if token_ml_tile_offset_local < async_copy_token_tile:
                            for i in cutlass.range_constexpr(n_sfa_blocks, unroll=1):
                                swizzled_iterator = (
                                    ((tidx_in_warpgroup % 32) // 8) & (n_sfa_blocks - 1)
                                ) ^ i
                                tAsSFA_slice = tAsSFA_blocked[(swizzled_iterator, None)]
                                for j in cutlass.range_constexpr(
                                    cute.size(tAsSFA_slice)
                                ):
                                    tAsSFA_slice[j] = self.sf_dtype(0.0)
                        self.async_copy_sync_barrier.arrive_and_wait()

                    for i in cutlass.range_constexpr(num_rows_per_thread, unroll=1):
                        #
                        # Load A with eight 128-bit copies per thread and a 128-byte swizzle.
                        # Each copy transfers the configured number of elements from GMEM to SMEM.
                        # Global memory address is computed using token offset for gather operation
                        # Predicate mask guards against invalid token IDs (padding tokens marked as -1)
                        #
                        A_gmem_slice_offset = A_gmem_thread_offset + cute.assume(
                            a_token_offset_tensor[i] * tAgA_ktile.layout[0].stride,
                            divby=self.a_elements_per_async_copy,
                        )
                        A_gmem_slice_offset = cute.assume(
                            A_gmem_slice_offset,
                            divby=self.a_elements_per_async_copy,
                        )
                        _, a_vector_idx = cute.idx2crd(
                            A_gmem_slice_offset,
                            (
                                self.a_elements_per_async_copy,
                                cute.size(tAgA_vectors, mode=[1]),
                            ),
                        )
                        tAgA_slice = tAgA_vectors[None, a_vector_idx]

                        tAsA_slice = cute.make_tensor(
                            tAsA_ktile[(None, i, None)].iterator,
                            layout=cute.make_layout((self.a_elements_per_async_copy,)),
                        )
                        a_predicate_slice = cute.make_rmem_tensor(
                            cute.make_layout((1,)), cutlass.Boolean
                        )
                        global_k_offset = (
                            global_k_tile * self.cta_tile_shape_mnk[2]
                            + A_gmem_thread_offset
                        )
                        a_predicate_slice[0] = cutlass.Boolean(False)
                        if a_predicate_tensor[i] and global_k_offset < mA_mkl.shape[1]:
                            a_predicate_slice[0] = cutlass.Boolean(True)

                        # Predicated cp.async zero-fills aliased rows; skip it instead.
                        if cutlass.const_expr(compact_swapped_tile):
                            if a_predicate_slice[0]:
                                cute.copy_atom_call(a_atom_copy, tAgA_slice, tAsA_slice)
                        else:
                            cute.copy_atom_call(
                                a_atom_copy,
                                tAgA_slice,
                                tAsA_slice,
                                pred=a_predicate_slice,
                            )
                    for i in cutlass.range_constexpr(n_sfa_blocks, unroll=1):
                        #
                        # Load SFA with n_sfa_blocks 32-bit copies per thread and
                        # 512-element block swizzling. Each copy transfers four scale factors.
                        # Uses same token offset as A matrix for consistent gather operation
                        #
                        # The swizzle base is reduced modulo n_sfa_blocks (power of 2)
                        # so threads never address a block that does not exist when
                        # there are fewer than 4 blocks (e.g. mixed precision -> 1).
                        swizzled_iterator = (
                            ((tidx_in_warpgroup % 32) // 8) & (n_sfa_blocks - 1)
                        ) ^ i
                        global_sf_offset = sfa_global_offset_layout(
                            (global_k_tile, swizzled_iterator)
                        )
                        tAgSFA_slice = tAgSFA_blocked[(swizzled_iterator, None)]
                        tAsSFA_slice = tAsSFA_blocked[(swizzled_iterator, None)]

                        sfa_copy_predicate = cute.make_rmem_tensor(
                            cute.make_layout((1,)), cutlass.Boolean
                        )
                        sfa_copy_predicate[0] = cutlass.Boolean(False)
                        if (
                            sfa_predicate_tensor[0]
                            and global_sf_offset < mSFA_mkl.shape[1]
                        ):
                            sfa_copy_predicate[0] = cutlass.Boolean(True)

                        if cutlass.const_expr(compact_swapped_tile):
                            if sfa_copy_predicate[0]:
                                cute.copy_atom_call(
                                    sfa_atom_copy, tAgSFA_slice, tAsSFA_slice
                                )
                        else:
                            cute.copy_atom_call(
                                sfa_atom_copy,
                                tAgSFA_slice,
                                tAsSFA_slice,
                                pred=sfa_copy_predicate,
                            )
                        zero_sfa = cutlass.Boolean(False)
                        if cutlass.const_expr(
                            self.swap_ab and not compact_swapped_tile
                        ):
                            if (
                                token_ml_tile_offset_local < async_copy_token_tile
                                and not sfa_predicate_tensor[0]
                            ):
                                zero_sfa = cutlass.Boolean(True)
                        if zero_sfa:
                            for j in cutlass.range_constexpr(cute.size(tAsSFA_slice)):
                                tAsSFA_slice[j] = self.sf_dtype(0.0)
                    if (
                        global_k_tile * self.cta_tile_shape_mnk[2]
                        + self.cta_tile_shape_mnk[2]
                        > mA_mkl.shape[1]
                    ):
                        # Publish explicit K-tail zeros before committing the stage.
                        self.async_copy_sync_barrier.arrive_and_wait()
                    if cutlass.const_expr(self.swap_ab) or is_partial_k_tile:
                        cute.arch.fence_proxy("async.shared", space="cta")
                    a_pipeline.producer_commit(a_producer_state)

                    # Launch dependents early to allow downstream grids to run without activations
                    if cutlass.const_expr(self.pdl_count is not None):
                        if warp_idx == self.async_copy_a_warp_id[0]:
                            if pdl_k_tile == self.pdl_count:
                                griddepcontrol_launch_dependents()
                    pdl_k_tile += 1

                    # Peek (try_wait) A buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    a_producer_state.advance()
                    peek_a_empty_status = cutlass.Boolean(1)
                    if a_producer_state.count < k_tile_cnt:
                        peek_a_empty_status = a_pipeline.producer_try_acquire(
                            a_producer_state
                        )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            #
            # Wait A pipeline buffer empty
            #
            a_pipeline.producer_tail(a_producer_state)

        #
        # Specialized A/SFA Sync Transform Warp (warp 11) when use_2cta_instrs is True
        # This warp serve as sync transformation for A and SFA
        #
        if warp_idx == self.sync_transform_warp_id:
            if cutlass.const_expr(self.use_2cta_instrs):
                #
                # Persistent tile scheduling loop
                #
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, scheduler_block_idx, scheduler_grid_dim
                )
                # First tile
                work_tile = tile_sched.initial_work_tile_info()

                a_consumer_state = make_pipeline_state(
                    PipelineUserType.Consumer, self.num_ab_stage
                )
                a_sync_transform_producer_state = make_pipeline_state(
                    PipelineUserType.Producer, self.num_ab_stage
                )
                if cutlass.const_expr(self.swap_ab):
                    tmem.wait_for_alloc()
                    sfb_tmem_producer_state = make_pipeline_state(
                        PipelineUserType.Producer, self.num_ab_stage
                    )
                tile_info_consumer_state = make_pipeline_state(
                    PipelineUserType.Consumer, self.num_tile_stage
                )

                # Get the first tile info
                tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                if cutlass.const_expr(self.swap_ab):
                    tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                    tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                while is_valid_tile:
                    # Peek (try_wait) A buffer full for k_tile = 0
                    a_consumer_state.reset_count()
                    peek_a_full_status = cutlass.Boolean(0)
                    # Peek (try_wait) a sync transform buffer empty
                    a_sync_transform_producer_state.reset_count()

                    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Conditionally wait for A buffer full
                        a_pipeline.consumer_wait(a_consumer_state, peek_a_full_status)

                        if cutlass.const_expr(self.swap_ab):
                            sfb_tmem_pipeline.producer_acquire(sfb_tmem_producer_state)
                            cute.arch.fence_proxy("async.shared", space="cta")
                            cute.arch.sync_warp()
                            sfb_tmem_pipeline.producer_commit(sfb_tmem_producer_state)
                            sfb_tmem_producer_state.advance()

                        a_sync_transform_pipeline.producer_commit(
                            a_sync_transform_producer_state
                        )
                        a_sync_transform_producer_state.advance()

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        a_consumer_state.advance()
                        peek_a_full_status = cutlass.Boolean(1)
                        if a_consumer_state.count < k_tile_cnt:
                            peek_a_full_status = cutlass.Boolean(0)

                    #
                    # Advance to next tile
                    #
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    if cutlass.const_expr(self.swap_ab):
                        tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                        tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                    tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                #
                # Wait A sync transform buffer empty
                #
                a_sync_transform_pipeline.producer_tail(a_sync_transform_producer_state)
                if cutlass.const_expr(self.swap_ab):
                    sfb_tmem_pipeline.producer_tail(sfb_tmem_producer_state)

        #
        # Specialized TMA B/SFB load warp (warp 9)
        # This warp uses TMA instructions to load B and SFB from global to shared memory
        # with multicast support to reduce L2 memory traffic
        #
        if warp_idx == self.tma_b_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, scheduler_block_idx, scheduler_grid_dim
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            b_producer_state = make_pipeline_state(
                PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = make_pipeline_state(
                PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
            tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), loopK)
                if cutlass.const_expr(not self.swap_ab):
                    weight_tile = mma_tile_coord_mnl[1]
                    weight_tile_sf = weight_tile
                    # Apply special SFB slicing when cta_tile_shape_n=64.
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                        weight_tile_sf = weight_tile // 2
                else:
                    weight_tile = mma_tile_coord_mnl[0]
                    weight_tile_sf = weight_tile
                    if cutlass.const_expr(self.use_2cta_instrs):
                        weight_tile = weight_tile * 2
                tma_g_weights_slice = tBgB[
                    (None, weight_tile, None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tma_g_sf_slice = tBgSFB[
                    (None, weight_tile_sf, None, mma_tile_coord_mnl[2])
                ]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                b_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if b_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = b_pipeline.producer_try_acquire(
                        b_producer_state
                    )
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                    # Conditionally wait for B buffer empty
                    b_pipeline.producer_acquire(b_producer_state, peek_ab_empty_status)

                    tma_g_weights_k = tma_g_weights_slice[
                        (None, k_tile_start + b_producer_state.count)
                    ]
                    tma_g_sf_k = tma_g_sf_slice[
                        (None, k_tile_start + b_producer_state.count)
                    ]
                    tma_s_weights_pipe = tBsB[(None, b_producer_state.index)]
                    tma_s_sf_pipe = tBsSFB[(None, b_producer_state.index)]

                    tma_bar = b_pipeline.producer_get_barrier(b_producer_state)

                    # TMA load B
                    cute.copy(
                        tma_atom_b,
                        tma_g_weights_k,
                        tma_s_weights_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # TMA load SFB
                    cute.copy(
                        tma_atom_sf,
                        tma_g_sf_k,
                        tma_s_sf_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    b_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if b_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = b_pipeline.producer_try_acquire(
                            b_producer_state
                        )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait A/B buffer empty
            #
            b_pipeline.producer_tail(b_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tmem_columns = cute.make_tensor(
                acc_tmem_ptr,
                cute.make_layout(
                    self.num_accumulator_tmem_cols + self.num_sf_tmem_cols
                ),
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                cute.domain_offset(
                    self.num_accumulator_tmem_cols, tmem_columns
                ).iterator,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_layout.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                cute.domain_offset(
                    self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                    tmem_columns,
                ).iterator,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_layout.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            sfb_panel_layout = cute.make_layout(2, stride=2)
            sfb_panel_coord_shape = (
                2,
                cute.ceil_div(cute.size(tBgB, mode=[1]), 2),
            )

            # Partition for S2T copy of SFA/SFB
            #
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, scheduler_block_idx, scheduler_grid_dim
            )
            work_tile = tile_sched.initial_work_tile_info()

            if cutlass.const_expr(self.use_2cta_instrs):
                a_sync_transform_consumer_state = make_pipeline_state(
                    PipelineUserType.Consumer, self.num_ab_stage
                )
            a_consumer_state = make_pipeline_state(
                PipelineUserType.Consumer, self.num_ab_stage
            )

            b_consumer_state = make_pipeline_state(
                PipelineUserType.Consumer, self.num_ab_stage
            )
            if cutlass.const_expr(self.swap_ab and self.use_2cta_instrs):
                sfb_tmem_consumer_state = make_pipeline_state(
                    PipelineUserType.Consumer, self.num_ab_stage
                )
            acc_producer_state = make_pipeline_state(
                PipelineUserType.Producer, self.num_acc_stage
            )
            tile_info_consumer_state = make_pipeline_state(
                PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info from pipeline (scheduler has filtered out tiles >= num_non_exiting_tiles)
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
            tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Peek (try_wait) AB buffer full for k_tile = 0
                if cutlass.const_expr(self.use_2cta_instrs):
                    a_sync_transform_consumer_state.reset_count()
                    peek_a_sync_transform_full_status = cutlass.Boolean(1)
                    if (
                        a_sync_transform_consumer_state.count < k_tile_cnt
                        and is_leader_cta
                    ):
                        peek_a_sync_transform_full_status = (
                            a_sync_transform_pipeline.consumer_try_wait(
                                a_sync_transform_consumer_state
                            )
                        )
                    a_consumer_state.reset_count()
                else:
                    a_consumer_state.reset_count()
                    peek_a_full_status = cutlass.Boolean(1)
                    if a_consumer_state.count < k_tile_cnt:
                        peek_a_full_status = a_pipeline.consumer_try_wait(
                            a_consumer_state
                        )

                b_consumer_state.reset_count()
                peek_b_full_status = cutlass.Boolean(1)
                if b_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_b_full_status = b_pipeline.consumer_try_wait(b_consumer_state)
                if cutlass.const_expr(self.swap_ab and self.use_2cta_instrs):
                    sfb_tmem_consumer_state.reset_count()
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # SFB TMEM 64/192 layouts apply to weight-on-N (non-swap) only.
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(
                    not self.swap_ab and self.cta_tile_shape_mnk[1] in (64, 192)
                ):
                    sfb_panel, _ = cute.idx2crd(
                        mma_tile_coord_mnl[1], sfb_panel_coord_shape
                    )
                    sfb_panel_offset = sfb_panel_layout(sfb_panel)
                    tCtSFB_mma = cute.make_tensor(
                        cute.recast_ptr(
                            acc_tmem_ptr
                            + self.num_accumulator_tmem_cols
                            + self.num_sfa_tmem_cols
                            + sfb_panel_offset,
                            dtype=self.sf_dtype,
                        ),
                        tCtSFB_layout,
                    )
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                #
                # Mma mainloop
                #

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in cutlass.range(k_tile_cnt):  # noqa: B007
                    # Set tensor memory buffer for current tile
                    # (MMA, MMA_M, MMA_N)

                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        if cutlass.const_expr(self.use_2cta_instrs):
                            a_sync_transform_pipeline.consumer_wait(
                                a_sync_transform_consumer_state,
                                peek_a_sync_transform_full_status,
                            )
                        else:
                            a_pipeline.consumer_wait(
                                a_consumer_state, peek_a_full_status
                            )
                        b_pipeline.consumer_wait(b_consumer_state, peek_b_full_status)
                        if cutlass.const_expr(self.swap_ab and self.use_2cta_instrs):
                            sfb_tmem_pipeline.consumer_wait(sfb_tmem_consumer_state)

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            b_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )

                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        # Issue a single block-scaled MMA over the full K with the
                        # scale-factor tensors passed inline as part of the A/B
                        # operands; required for mixed-precision inputs
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        tile_crd = (None, None, None, b_consumer_state.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            [tCrA[tile_crd], tCtSFA],
                            [tCrB[tile_crd], tCtSFB_mma],
                            tCtAcc,
                        )
                        if cutlass.const_expr(self.swap_ab and self.use_2cta_instrs):
                            sfb_tmem_pipeline.consumer_release(sfb_tmem_consumer_state)
                        # Async arrive AB buffer empty
                        a_pipeline.consumer_release(a_consumer_state)
                        if cutlass.const_expr(self.use_2cta_instrs):
                            a_sync_transform_pipeline.consumer_release(
                                a_sync_transform_consumer_state
                            )
                        b_pipeline.consumer_release(b_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    if cutlass.const_expr(self.use_2cta_instrs):
                        a_sync_transform_consumer_state.advance()
                        peek_a_sync_transform_full_status = cutlass.Boolean(1)
                        if a_sync_transform_consumer_state.count < k_tile_cnt:
                            if is_leader_cta:
                                peek_a_sync_transform_full_status = (
                                    a_sync_transform_pipeline.consumer_try_wait(
                                        a_sync_transform_consumer_state
                                    )
                                )
                        a_consumer_state.advance()
                    else:
                        a_consumer_state.advance()
                        peek_a_full_status = cutlass.Boolean(1)
                        if a_consumer_state.count < k_tile_cnt:
                            peek_a_full_status = a_pipeline.consumer_try_wait(
                                a_consumer_state
                            )

                    b_consumer_state.advance()
                    if cutlass.const_expr(self.swap_ab and self.use_2cta_instrs):
                        sfb_tmem_consumer_state.advance()
                    peek_b_full_status = cutlass.Boolean(1)
                    if b_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_b_full_status = b_pipeline.consumer_try_wait(
                                b_consumer_state
                            )

                #
                # Async arrive accumulator buffer full(each kblock)
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx <= self.epilog_warp_id[-1]:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)
            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tiled_copy_epi,
                tTR_tAcc_base,
                tTR_rAcc_own,
                tTR_rAcc_up,
                tTR_rAcc_gate,
                tTR_rAcc_panel,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            #
            # Partition the per-expert bias
            #
            epi_data_tidx = (
                (epi_tidx // 64) * 32 + epi_tidx % 32
                if self.swap_ab and self.gated
                else epi_tidx
            )
            thr_copy_t2r_bias = tiled_copy_epi.get_slice(epi_data_tidx)
            tTR_gBiasUp = None
            tTR_gBiasGate = None
            if cutlass.const_expr(self.use_bias):
                tCgBiasUp_epi = cute.flat_divide(
                    tCgBiasUp[((None, None), 0, 0, None, None, None)], epi_tile
                )
                tCgBiasGate_epi = cute.flat_divide(
                    tCgBiasGate[((None, None), 0, 0, None, None, None)], epi_tile
                )
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
                tTR_gBiasUp = thr_copy_t2r_bias.partition_D(tCgBiasUp_epi)
                tTR_gBiasGate = thr_copy_t2r_bias.partition_D(tCgBiasGate_epi)
            tTR_gC_output = None
            unswapped_fp4_sfc = (
                self.generate_sfc
                and not self.swap_ab
                and self.gated
                and self.weight_interleave == 16
                and self.c_dtype == cutlass.Float4E2M1FN
            )
            output_idC = cute.make_identity_tensor(
                (
                    self.cta_tile_shape_mnk_c[0],
                    self.cta_tile_shape_mnk_c[1],
                )
            )
            output_idC_epi = cute.flat_divide(output_idC, epi_tile)
            output_cid_full = thr_copy_t2r_bias.partition_D(output_idC_epi)
            if cutlass.const_expr(self.distributed_split_k):
                tCgC_epi = cute.flat_divide(
                    tCgC[((None, None), 0, 0, None, None, None)], epi_tile
                )
                tTR_gC_output = thr_copy_t2r_bias.partition_D(tCgC_epi)

            tiled_copy_r2s = None
            tRS_rC = None
            tRS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            output_fragment_shape = (
                tTR_rAcc_own.shape if self.swap_ab and self.gated else tTR_rAcc_up.shape
            )
            tTR_rC = cute.make_rmem_tensor(output_fragment_shape, self.c_dtype)
            tTR_rC_packed = None
            if cutlass.const_expr(self.fp4_swap):
                # (PACKED_C)
                packed_output_shape = cute.make_layout(
                    cute.size(tTR_rAcc_up) * self.c_dtype.width // 16
                )
                tTR_rC_packed = cute.make_rmem_tensor(
                    packed_output_shape, cutlass.Uint16
                )
            if cutlass.const_expr(not self.distributed_split_k):
                if cutlass.const_expr(self.fp4_swap):
                    (
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC,
                    ) = self.epilog_packed_smem_copy_and_partition(
                        epi_tidx,
                        sC,
                        c_smem_layout_staged,
                    )
                else:
                    (
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC,
                    ) = self.epilog_smem_copy_and_partition(
                        tiled_copy_epi, tTR_rC, epi_tidx, sC
                    )
                (
                    tma_atom_c,
                    bSG_sC,
                    bSG_gC_partitioned,
                ) = self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_c, tCgC, epi_tile, sC
                )

            if cutlass.const_expr(self.generate_sfc):
                if cutlass.const_expr(norm_const_tensor is not None):
                    norm_const = norm_const_tensor[0]
                else:
                    norm_const = cutlass.Float32(1.0)
                thr_copy_t2r = tiled_copy_epi.get_slice(epi_data_tidx)
                if cutlass.const_expr(not self.swap_ab):
                    # (EPI_TILE_M, EPI_TILE_N, RestM, RestN, RestL)
                    gSFC_mnl = cute.local_tile(mSFC_mnl, epi_tile, (None, None, None))
                    # (T2R, T2R_M, T2R_N, RestM, RestN, RestL)
                    tCgSFC_mnl = thr_copy_t2r.partition_D(gSFC_mnl)
                    tCgSFC_mnl = cute.filter_zeros(tCgSFC_mnl)
                    if cutlass.const_expr(unswapped_fp4_sfc):
                        sfc_layout = cute.make_layout(
                            cute.size(tTR_rAcc_up) // self.sf_vec_size
                        )
                    else:
                        sfc_layout = tCgSFC_mnl[(None, None, None, 0, 0, 0)].layout
                    tCrSFC = cute.make_rmem_tensor(sfc_layout, self.sf_dtype)
                    tCrSFC_pvscale = cute.make_rmem_tensor_like(tCrSFC, cutlass.Float32)
                else:
                    n_sfc_swap = cute.size(tTR_rAcc_own) // (self.sf_vec_size // 8)
                    if cutlass.const_expr(self.fp4_swap):
                        n_sfc_swap = cute.size(tTR_rAcc_up) // (self.sf_vec_size // 8)
                    tCrSFC = cute.make_rmem_tensor(
                        cute.make_layout(n_sfc_swap), self.sf_dtype
                    )
                    tCrSFC_pvscale = cute.make_rmem_tensor_like(tCrSFC, cutlass.Float32)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, scheduler_block_idx, scheduler_grid_dim
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = make_pipeline_state(
                PipelineUserType.Consumer, self.num_acc_stage
            )
            c_pipeline = None
            if cutlass.const_expr(not self.distributed_split_k):
                c_producer_group = CooperativeGroup(
                    Agent.Thread,
                    32 * len(self.epilog_warp_id),
                )
                c_pipeline = PipelineTmaStore.create(
                    num_stages=self.num_c_stage,
                    producer_group=c_producer_group,
                )

            tile_info_consumer_state = make_pipeline_state(
                PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
            tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            if cutlass.const_expr(self.use_a_per_token_scale):
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                output_tile_coord_mnl = mma_tile_coord_mnl
                if cutlass.const_expr(
                    self.swap_ab and self.gated and self.use_2cta_instrs
                ):
                    output_tile_coord_mnl = (
                        mma_tile_coord_mnl[0] * cute.size(tiled_mma.thr_id.shape)
                        + mma_tile_coord_v,
                        tile_info[1],
                        tile_info[2],
                    )
                #
                # Get alpha for current group (identity when use_alpha=False)
                #
                expert_idx = mma_tile_coord_mnl[2]
                if cutlass.const_expr(self.use_alpha):
                    alpha_val = alpha[expert_idx]
                else:
                    alpha_val = cutlass.Float32(1.0)
                if cutlass.const_expr(self.use_a_per_token_scale):
                    tile_m_start = tile_info[0] * self.cta_tile_shape_mnk[0]
                    permuted_row = tile_m_start + epi_tidx
                    if permuted_row < tile_info[4]:
                        expanded_idx = token_id_mapping_tensor[permuted_row]
                        token_idx_for_scale = expanded_idx // self.topk
                        alpha_val = alpha_val * a_per_token_scale[token_idx_for_scale]

                #
                # Slice to per mma tile index
                #
                bSG_gC = None
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                if cutlass.const_expr(not self.distributed_split_k):
                    bSG_gC = bSG_gC_partitioned[
                        (
                            None,
                            None,
                            None,
                            output_tile_coord_mnl[0],
                            output_tile_coord_mnl[1],
                            0,
                        )
                    ]
                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = (
                        cutlass.Boolean(True)
                        if acc_stage_index == 0
                        else cutlass.Boolean(False)
                    )
                else:
                    acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_stage_index)
                ]

                if cutlass.const_expr(self.generate_sfc and not self.swap_ab):
                    # (T2R, T2R_M, T2R_N, RestM, RestN)
                    tCgSFC_mn = tCgSFC_mnl[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            0,
                        )
                    ]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
                epi_m_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                epi_n_cnt = cute.size(tTR_tAcc.shape, mode=[4])
                unswapped_gated = (
                    not self.swap_ab and self.gated and self.weight_interleave == 16
                )
                weight_step = (
                    1
                    if unswapped_gated and self.cta_tile_shape_mnk[1] != 128
                    else (2 if self.gated else 1)
                )
                if cutlass.const_expr(not self.swap_ab):
                    weight_cnt = epi_n_cnt
                    token_cnt = epi_m_cnt
                else:
                    weight_cnt = epi_m_cnt
                    token_cnt = epi_n_cnt
                if cutlass.const_expr(
                    not self.swap_ab and self.cta_tile_shape_mnk[1] == 192
                ):
                    output_n_start = (
                        output_tile_coord_mnl[1] * self.cta_tile_shape_mnk_c[1]
                    )
                    valid_weight_loops = cute.ceil_div(
                        mC_mnl.shape[1] - output_n_start, self.epi_tile[1]
                    )
                    valid_weight_cnt = valid_weight_loops * weight_step
                    if valid_weight_cnt < weight_cnt:
                        weight_cnt = valid_weight_cnt

                tTR_gBiasUp_tile = None
                tTR_gBiasGate_tile = None
                tTR_gC_output_tile = None
                if cutlass.const_expr(self.gated and self.use_bias):
                    # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
                    tTR_gBiasUp_tile = tTR_gBiasUp[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            output_tile_coord_mnl[0],
                            output_tile_coord_mnl[1],
                            expert_idx,
                        )
                    ]
                    tTR_gBiasGate_tile = tTR_gBiasGate[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            output_tile_coord_mnl[0],
                            output_tile_coord_mnl[1],
                            expert_idx,
                        )
                    ]
                if cutlass.const_expr(self.distributed_split_k):
                    tTR_gC_output_tile = tTR_gC_output[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            output_tile_coord_mnl[0],
                            output_tile_coord_mnl[1],
                            0,
                        )
                    ]
                for weight_loop_idx in cutlass.range(0, weight_cnt, weight_step):
                    weight_idx = weight_loop_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            weight_idx = weight_cnt - weight_step - weight_loop_idx
                    if cutlass.const_expr(self.gated):
                        weight_pair_idx = (
                            weight_idx
                            if unswapped_gated and self.cta_tile_shape_mnk[1] != 128
                            else weight_idx // 2
                        )
                    else:
                        weight_pair_idx = weight_idx

                    for token_idx in cutlass.range(token_cnt):
                        if cutlass.const_expr(self.gated):
                            if cutlass.const_expr(not self.swap_ab):
                                epi_m_up = token_idx
                                epi_n_up = weight_idx
                                epi_m_gate = token_idx
                                epi_n_gate = weight_idx + 1
                                c_m = token_idx
                                c_n = weight_pair_idx
                            else:
                                epi_m_up = weight_idx
                                epi_n_up = token_idx
                                epi_m_gate = weight_idx
                                epi_n_gate = token_idx
                                c_m = weight_pair_idx
                                c_n = token_idx
                        else:
                            if cutlass.const_expr(not self.swap_ab):
                                epi_m_up = token_idx
                                epi_n_up = weight_idx
                                c_m = token_idx
                                c_n = weight_idx
                            else:
                                epi_m_up = weight_idx
                                epi_n_up = token_idx
                                c_m = weight_idx
                                c_n = token_idx
                            epi_m_gate = epi_m_up
                            epi_n_gate = epi_n_up
                        #
                        # Load accumulator from tensor memory buffer to register
                        #
                        if cutlass.const_expr(self.gated):
                            if cutlass.const_expr(not self.swap_ab):
                                if cutlass.const_expr(
                                    unswapped_gated
                                    and self.cta_tile_shape_mnk[1] != 128
                                ):
                                    tTR_tAcc_mn = tTR_tAcc[
                                        (None, None, None, epi_m_up, epi_n_up)
                                    ]
                                    for t2r_m in cutlass.range_constexpr(2):
                                        tTR_tAcc_panel = tTR_tAcc_mn[(None, t2r_m, 0)]
                                        cute.copy(
                                            tiled_copy_t2r,
                                            tTR_tAcc_panel,
                                            tTR_rAcc_panel[(None, t2r_m)],
                                        )
                                elif cutlass.const_expr(unswapped_gated):
                                    for panel_idx in cutlass.range_constexpr(2):
                                        tTR_tAcc_panel = tTR_tAcc[
                                            (
                                                None,
                                                None,
                                                None,
                                                epi_m_up,
                                                weight_idx + panel_idx,
                                            )
                                        ]
                                        cute.copy(
                                            tiled_copy_t2r,
                                            tTR_tAcc_panel,
                                            tTR_rAcc_panel[
                                                (None, None, None, panel_idx)
                                            ],
                                        )
                                else:
                                    tTR_tAcc_mn_up = tTR_tAcc[
                                        (None, None, None, epi_m_up, epi_n_up)
                                    ]
                                    cute.copy(
                                        tiled_copy_t2r, tTR_tAcc_mn_up, tTR_rAcc_up
                                    )
                                    tTR_tAcc_mn_gate = tTR_tAcc[
                                        (None, None, None, epi_m_gate, epi_n_gate)
                                    ]
                                    cute.copy(
                                        tiled_copy_t2r,
                                        tTR_tAcc_mn_gate,
                                        tTR_rAcc_gate,
                                    )
                            else:
                                tTR_tAcc_mn_own = tTR_tAcc[
                                    (None, None, None, epi_m_up, epi_n_up)
                                ]
                                cute.copy(
                                    tiled_copy_t2r,
                                    tTR_tAcc_mn_own,
                                    tTR_rAcc_own,
                                )
                        else:
                            tTR_tAcc_mn = tTR_tAcc[
                                (None, None, None, epi_m_up, epi_n_up)
                            ]
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc_up)

                        #
                        # Async arrive accumulator buffer empty earlier when overlapping_accum is enabled
                        #
                        if cutlass.const_expr(self.overlapping_accum):
                            if (
                                token_idx == token_cnt - 1
                                and weight_pair_idx
                                == self.iter_acc_early_release_in_epilogue
                            ):
                                # Fence for TMEM load
                                cute.arch.fence_view_async_tmem_load()
                                acc_pipeline.consumer_release(acc_consumer_state)
                                acc_consumer_state.advance()

                        if cutlass.const_expr(self.split_k > 1):
                            if k_tile_cnt == 0:
                                tTR_rAcc_own.fill(0.0)

                            values_per_thread = self.split_k_values_per_thread
                            epilog_threads = self.threads_per_warp * len(
                                self.epilog_warp_id
                            )
                            values_per_peer = epilog_threads * values_per_thread
                            reduce_phase = num_prev_subtiles % cutlass.Int32(2)
                            if cutlass.const_expr(self.distributed_split_k):
                                # Delay the empty-mailbox wait until the slot is
                                # reused so peer consumption overlaps this CTA's
                                # preceding activation, quantization, and store.
                                if num_prev_subtiles > 0:
                                    consumed_phase = (
                                        num_prev_subtiles - cutlass.Int32(1)
                                    ) % cutlass.Int32(2)
                                    cute.arch.mbarrier_wait(
                                        split_k_consumed_mbar,
                                        consumed_phase,
                                    )
                                split_output_half = warp_idx & 1
                                output_cid_sub = output_cid_full[
                                    (None, None, None, epi_m_up, epi_n_up)
                                ]
                                split_output_coords = output_cid_sub[
                                    (None, split_output_half, None)
                                ]
                                output_values_per_thread = cute.size(tTR_rAcc_up)
                                split_scale_vectors_per_tile = (
                                    self.cta_tile_shape_mnk_c[0] // self.sf_vec_size
                                )
                                split_interm_base = (
                                    output_tile_coord_mnl[0]
                                    * self.cta_tile_shape_mnk_c[0]
                                )
                                split_slot_base = (
                                    mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk_c[1]
                                )

                                # Every CTA receives one scale-aligned output
                                # shard from every peer CTA.
                                if epi_tidx == 0:
                                    cute.arch.mbarrier_arrive_and_expect_tx(
                                        split_k_reduce_mbar,
                                        (self.split_k - 1)
                                        * epilog_threads
                                        * cute.size(tTR_rAcc_own)
                                        // self.split_k
                                        * cutlass.Float32.width
                                        // 8,
                                    )
                                self.epilog_sync_barrier.arrive_and_wait()

                                for output_idx in cutlass.range_constexpr(
                                    output_values_per_thread
                                ):
                                    split_output_coord = split_output_coords[output_idx]
                                    output_owner = (
                                        (split_slot_base + split_output_coord[1])
                                        * split_scale_vectors_per_tile
                                        + (split_interm_base + split_output_coord[0])
                                        // self.sf_vec_size
                                    ) % self.split_k
                                    if split_rank != output_owner:
                                        sender_idx = split_rank - cutlass.select_(
                                            split_rank > output_owner, 1, 0
                                        )
                                        mailbox_base = sender_idx * values_per_peer
                                        cute.arch.store_async_dsmem(
                                            split_k_mailbox
                                            + mailbox_base
                                            + output_idx * epilog_threads
                                            + epi_tidx,
                                            tTR_rAcc_up[output_idx].bitcast(
                                                cutlass.Int32
                                            ),
                                            split_k_reduce_mbar,
                                            split_peer_base
                                            + output_owner * split_peer_stride,
                                        )
                                        cute.arch.store_async_dsmem(
                                            split_k_mailbox
                                            + mailbox_base
                                            + (output_values_per_thread + output_idx)
                                            * epilog_threads
                                            + epi_tidx,
                                            tTR_rAcc_gate[output_idx].bitcast(
                                                cutlass.Int32
                                            ),
                                            split_k_reduce_mbar,
                                            split_peer_base
                                            + output_owner * split_peer_stride,
                                        )

                                cute.arch.fence_acq_rel_cta()
                                cute.arch.mbarrier_wait(
                                    split_k_reduce_mbar, reduce_phase
                                )
                                for sender_idx in cutlass.range_constexpr(
                                    self.split_k - 1
                                ):
                                    mailbox_base = sender_idx * values_per_peer
                                    for output_idx in cutlass.range_constexpr(
                                        output_values_per_thread
                                    ):
                                        split_output_coord = split_output_coords[
                                            output_idx
                                        ]
                                        output_owner = (
                                            (split_slot_base + split_output_coord[1])
                                            * split_scale_vectors_per_tile
                                            + (
                                                split_interm_base
                                                + split_output_coord[0]
                                            )
                                            // self.sf_vec_size
                                        ) % self.split_k
                                        if output_owner == split_rank:
                                            tTR_rAcc_up[output_idx] = (
                                                tTR_rAcc_up[output_idx]
                                                + split_k_mailbox[
                                                    mailbox_base
                                                    + output_idx * epilog_threads
                                                    + epi_tidx
                                                ]
                                            )
                                            tTR_rAcc_gate[output_idx] = (
                                                tTR_rAcc_gate[output_idx]
                                                + split_k_mailbox[
                                                    mailbox_base
                                                    + (
                                                        output_values_per_thread
                                                        + output_idx
                                                    )
                                                    * epilog_threads
                                                    + epi_tidx
                                                ]
                                            )

                                self.epilog_sync_barrier.arrive_and_wait()
                                if epi_tidx == 0:
                                    for peer_rank in cutlass.range_constexpr(
                                        self.split_k
                                    ):
                                        if split_rank != peer_rank:
                                            cute.arch.mbarrier_arrive(
                                                split_k_consumed_mbar,
                                                peer_cta_rank_in_cluster=(
                                                    split_peer_base
                                                    + peer_rank * split_peer_stride
                                                ),
                                            )
                            else:
                                if split_rank == 0:
                                    if epi_tidx == 0:
                                        cute.arch.mbarrier_arrive_and_expect_tx(
                                            split_k_reduce_mbar,
                                            (self.split_k - 1)
                                            * values_per_peer
                                            * cutlass.Float32.width
                                            // 8,
                                        )

                                # Publish the owner's transaction expectation before
                                # peer CTAs start their DSMEM stores.
                                self.epilog_sync_barrier.arrive_and_wait()
                                if split_rank > 0:
                                    mailbox_base = (split_rank - 1) * values_per_peer
                                    for value_idx in cutlass.range_constexpr(
                                        values_per_thread
                                    ):
                                        cute.arch.store_async_dsmem(
                                            split_k_mailbox
                                            + mailbox_base
                                            + value_idx * epilog_threads
                                            + epi_tidx,
                                            tTR_rAcc_own[value_idx].bitcast(
                                                cutlass.Int32
                                            ),
                                            split_k_reduce_mbar,
                                            split_peer_base,
                                        )
                                    cute.arch.mbarrier_wait(
                                        split_k_consumed_mbar, reduce_phase
                                    )
                                else:
                                    cute.arch.mbarrier_wait(
                                        split_k_reduce_mbar, reduce_phase
                                    )
                                    for peer_idx in cutlass.range_constexpr(
                                        self.split_k - 1
                                    ):
                                        peer_base = peer_idx * values_per_peer
                                        for value_idx in cutlass.range_constexpr(
                                            values_per_thread
                                        ):
                                            tTR_rAcc_own[value_idx] = (
                                                tTR_rAcc_own[value_idx]
                                                + split_k_mailbox[
                                                    peer_base
                                                    + value_idx * epilog_threads
                                                    + epi_tidx
                                                ]
                                            )

                                    # A peer must not reuse its mailbox until every
                                    # owner thread has consumed the current partial.
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if epi_tidx == 0:
                                        for peer_rank in cutlass.range_constexpr(
                                            1, self.split_k
                                        ):
                                            cute.arch.mbarrier_arrive(
                                                split_k_consumed_mbar,
                                                peer_cta_rank_in_cluster=(
                                                    split_peer_base
                                                    + peer_rank * split_peer_stride
                                                ),
                                            )

                        if (
                            cutlass.const_expr(
                                self.split_k == 1 or self.distributed_split_k
                            )
                            or split_rank == 0
                        ):
                            if cutlass.const_expr(not self.gated):
                                acc_vec = tTR_rAcc_up.load()
                                tCompute = cute.make_rmem_tensor(
                                    acc_vec.shape, self.acc_dtype
                                )
                                if cutlass.const_expr(self.vectorized_f32):
                                    for i in cutlass.range_constexpr(
                                        0, cute.size(tTR_rAcc_up), 2
                                    ):
                                        acc_alpha = cute.arch.mul_packed_f32x2(
                                            (acc_vec[i], acc_vec[i + 1]),
                                            (
                                                cutlass.Float32(alpha_val),
                                                cutlass.Float32(alpha_val),
                                            ),
                                        )
                                        r0 = cute.arch.fmax(
                                            acc_alpha[0], cutlass.Float32(0.0)
                                        )
                                        r1 = cute.arch.fmax(
                                            acc_alpha[1], cutlass.Float32(0.0)
                                        )
                                        (
                                            tCompute[i],
                                            tCompute[i + 1],
                                        ) = cute.arch.mul_packed_f32x2(
                                            (r0, r1), (r0, r1)
                                        )
                                else:
                                    for i in cutlass.range_constexpr(
                                        cute.size(tTR_rAcc_up)
                                    ):
                                        v = acc_vec[i] * cutlass.Float32(alpha_val)
                                        v = cute.arch.fmax(v, cutlass.Float32(0.0))
                                        tCompute[i] = v * v
                            else:
                                acc_vec_up = tTR_rAcc_up.load()
                                acc_vec_gate = tTR_rAcc_gate.load()

                                #
                                # Per-expert bias (C-shaped): index by output epi (c_m, c_n)
                                #
                                bias_up_vec = None
                                bias_gate_vec = None
                                if cutlass.const_expr(self.use_bias):
                                    bias_up_frg = tTR_gBiasUp_tile[
                                        (None, None, None, c_m, c_n)
                                    ]
                                    bias_gate_frg = tTR_gBiasGate_tile[
                                        (None, None, None, c_m, c_n)
                                    ]
                                    if cutlass.const_expr(self.swap_ab):
                                        output_half = warp_idx & 1
                                        bias_up_vec = bias_up_frg[
                                            (None, output_half, None)
                                        ].load()
                                        bias_gate_vec = bias_gate_frg[
                                            (None, output_half, None)
                                        ].load()
                                    else:
                                        bias_up_vec = bias_up_frg.load()
                                        bias_gate_vec = bias_gate_frg.load()

                                n_eff = cutlass.const_expr(cute.size(tTR_rAcc_up))
                                tCompute = cute.make_rmem_tensor(
                                    acc_vec_gate.shape, self.acc_dtype
                                )
                                swiglu_alpha = cutlass.Float32(self.swiglu_alpha)
                                swiglu_beta = cutlass.Float32(self.swiglu_beta)
                                alpha_log2e = swiglu_alpha * cutlass.Float32(
                                    1.4426950408889634
                                )
                                if cutlass.const_expr(self.situ_beta is not None):
                                    # Keep the Python float so situ_f32 can fold 1/beta.
                                    situ_beta = self.situ_beta
                                    if cutlass.const_expr(
                                        self.situ_linear_beta is not None
                                    ):
                                        linear_beta = cutlass.Float32(
                                            self.situ_linear_beta
                                        )
                                        inv_linear_beta = cutlass.Float32(
                                            f32_reciprocal(self.situ_linear_beta)
                                        )
                                    for i in cutlass.range_constexpr(n_eff):
                                        up = acc_vec_up[i] * cutlass.Float32(alpha_val)
                                        gate = acc_vec_gate[i] * cutlass.Float32(
                                            alpha_val
                                        )
                                        if cutlass.const_expr(self.use_bias):
                                            up += bias_up_vec[i]
                                            gate += bias_gate_vec[i]
                                        if cutlass.const_expr(
                                            self.situ_linear_beta is not None
                                        ):
                                            up = linear_beta * tanh_f32(
                                                up * inv_linear_beta, fastmath=True
                                            )
                                        tCompute[i] = up * situ_f32(
                                            gate, situ_beta, fastmath=True
                                        )
                                elif cutlass.const_expr(
                                    self.activation_type
                                    == ActivationType.GegluTanh.value
                                ):
                                    for i in cutlass.range_constexpr(n_eff):
                                        up = acc_vec_up[i] * cutlass.Float32(alpha_val)
                                        gate = acc_vec_gate[i] * cutlass.Float32(
                                            alpha_val
                                        )
                                        if cutlass.const_expr(self.use_bias):
                                            up += bias_up_vec[i]
                                            gate += bias_gate_vec[i]
                                        tCompute[i] = up * gelu_tanh_f32(
                                            gate, fastmath=True
                                        )
                                # 2CTA packed-FP4 bias fragments require scalar indexing.
                                elif cutlass.const_expr(
                                    self.vectorized_f32
                                    and not (
                                        self.fp4_swap
                                        and self.use_2cta_instrs
                                        and self.use_bias
                                    )
                                ):
                                    # Generalized scaled/clamped SwiGLU family.
                                    #   out = (up_c + offset) * gate_c * sigmoid(swiglu_alpha * gate_c)
                                    for i in cutlass.range_constexpr(0, n_eff, 2):
                                        acc_vec_up_alpha = cute.arch.mul_packed_f32x2(
                                            (acc_vec_up[i], acc_vec_up[i + 1]),
                                            (
                                                cutlass.Float32(alpha_val),
                                                cutlass.Float32(alpha_val),
                                            ),
                                        )
                                        acc_vec_gate_alpha = cute.arch.mul_packed_f32x2(
                                            (acc_vec_gate[i], acc_vec_gate[i + 1]),
                                            (
                                                cutlass.Float32(alpha_val),
                                                cutlass.Float32(alpha_val),
                                            ),
                                        )
                                        if cutlass.const_expr(self.use_bias):
                                            acc_vec_up_alpha = (
                                                cute.arch.add_packed_f32x2(
                                                    acc_vec_up_alpha,
                                                    (
                                                        bias_up_vec[i],
                                                        bias_up_vec[i + 1],
                                                    ),
                                                )
                                            )
                                            acc_vec_gate_alpha = (
                                                cute.arch.add_packed_f32x2(
                                                    acc_vec_gate_alpha,
                                                    (
                                                        bias_gate_vec[i],
                                                        bias_gate_vec[i + 1],
                                                    ),
                                                )
                                            )
                                        if cutlass.const_expr(
                                            self.swiglu_limit is not None
                                        ):
                                            swiglu_limit = cutlass.Float32(
                                                self.swiglu_limit
                                            )
                                            acc_vec_gate_alpha = (
                                                fmin(
                                                    acc_vec_gate_alpha[0],
                                                    swiglu_limit,
                                                    nan=True,
                                                ),
                                                fmin(
                                                    acc_vec_gate_alpha[1],
                                                    swiglu_limit,
                                                    nan=True,
                                                ),
                                            )
                                            acc_vec_up_alpha = (
                                                fmax(
                                                    fmin(
                                                        acc_vec_up_alpha[0],
                                                        swiglu_limit,
                                                        nan=True,
                                                    ),
                                                    -swiglu_limit,
                                                    nan=True,
                                                ),
                                                fmax(
                                                    fmin(
                                                        acc_vec_up_alpha[1],
                                                        swiglu_limit,
                                                        nan=True,
                                                    ),
                                                    -swiglu_limit,
                                                    nan=True,
                                                ),
                                            )
                                        tCompute_log2e = cute.arch.mul_packed_f32x2(
                                            (
                                                acc_vec_gate_alpha[0],
                                                acc_vec_gate_alpha[1],
                                            ),
                                            (-alpha_log2e, -alpha_log2e),
                                        )
                                        # Clamp exp2 input to [-127, 127] to prevent
                                        # exp2 overflow -> inf -> 0*inf = NaN
                                        log2e_max = cutlass.Float32(127.0)
                                        tCompute_log2e = (
                                            fmin(
                                                fmax(tCompute_log2e[0], -log2e_max),
                                                log2e_max,
                                            ),
                                            fmin(
                                                fmax(tCompute_log2e[1], -log2e_max),
                                                log2e_max,
                                            ),
                                        )
                                        (
                                            tCompute[i],
                                            tCompute[i + 1],
                                        ) = cute.arch.add_packed_f32x2(
                                            (
                                                cute.math.exp2(
                                                    tCompute_log2e[0], fastmath=True
                                                ),
                                                cute.math.exp2(
                                                    tCompute_log2e[1], fastmath=True
                                                ),
                                            ),
                                            (1.0, 1.0),
                                        )
                                        (tCompute[i], tCompute[i + 1]) = (
                                            cute.arch.rcp_approx(tCompute[i]),
                                            cute.arch.rcp_approx(tCompute[i + 1]),
                                        )
                                        # sigmoid(swiglu_alpha * gate_c) = 1 / (1 + exp(-swiglu_alpha * gate_c))
                                        prod = cute.arch.mul_packed_f32x2(
                                            (
                                                acc_vec_gate_alpha[0],
                                                acc_vec_gate_alpha[1],
                                            ),
                                            cute.arch.add_packed_f32x2(
                                                (
                                                    acc_vec_up_alpha[0],
                                                    acc_vec_up_alpha[1],
                                                ),
                                                (swiglu_beta, swiglu_beta),
                                            ),
                                        )
                                        # Clamp gate*(up+β) to [-fp32_max, fp32_max]
                                        # to prevent overflow
                                        fp32m = cutlass.Float32(FP32_MAX)
                                        prod = (
                                            fmin(fmax(prod[0], -fp32m), fp32m),
                                            fmin(fmax(prod[1], -fp32m), fp32m),
                                        )
                                        (
                                            tCompute[i],
                                            tCompute[i + 1],
                                        ) = cute.arch.mul_packed_f32x2(
                                            (tCompute[i], tCompute[i + 1]),
                                            prod,
                                        )
                                else:
                                    for i in cutlass.range_constexpr(n_eff):
                                        acc_vec_up_alpha = acc_vec_up[
                                            i
                                        ] * cutlass.Float32(alpha_val)
                                        acc_vec_gate_alpha = acc_vec_gate[
                                            i
                                        ] * cutlass.Float32(alpha_val)
                                        if cutlass.const_expr(self.use_bias):
                                            acc_vec_up_alpha += bias_up_vec[i]
                                            acc_vec_gate_alpha += bias_gate_vec[i]
                                        if cutlass.const_expr(
                                            self.swiglu_limit is not None
                                        ):
                                            swiglu_limit = cutlass.Float32(
                                                self.swiglu_limit
                                            )
                                            acc_vec_up_alpha = fmax(
                                                fmin(
                                                    acc_vec_up_alpha,
                                                    swiglu_limit,
                                                    nan=True,
                                                ),
                                                -swiglu_limit,
                                                nan=True,
                                            )
                                            acc_vec_gate_alpha = fmin(
                                                acc_vec_gate_alpha,
                                                swiglu_limit,
                                                nan=True,
                                            )
                                        # Clamp sigmoid input to [-88, 88] to prevent
                                        # exp overflow
                                        gate_clamped = cute.arch.fmin(
                                            cute.arch.fmax(
                                                acc_vec_gate_alpha,
                                                cutlass.Float32(-88.0),
                                            ),
                                            cutlass.Float32(88.0),
                                        )
                                        # Clamp product to fp32_max before sigmoid scales.
                                        prod_s = (
                                            acc_vec_up_alpha + swiglu_beta
                                        ) * acc_vec_gate_alpha
                                        fp32m_s = cutlass.Float32(FP32_MAX)
                                        prod_s = cute.arch.fmin(
                                            cute.arch.fmax(prod_s, -fp32m_s), fp32m_s
                                        )
                                        tCompute[i] = prod_s * sigmoid_f32(
                                            gate_clamped * swiglu_alpha, fastmath=True
                                        )

                            if cutlass.const_expr(self.generate_sfc):
                                #
                                # Quantization path for Float4E2M1FN output:
                                # 1. Compute per-vector absolute max from SwiGLU result
                                # 2. Generate scale factor C (SFC) based on max values
                                # 3. Store SFC to global memory
                                # 4. Quantize output by scaling with reciprocal of SFC
                                #
                                if cutlass.const_expr(not self.swap_ab):
                                    sfc_subtile_idx_mn = (
                                        tile_info[0] * self.epi_tile_cnt[0] + c_m,
                                        tile_info[1] * self.epi_tile_cnt[1] + c_n,
                                    )
                                    tCgSFC = tCgSFC_mn[
                                        (
                                            None,
                                            None,
                                            None,
                                            *sfc_subtile_idx_mn,
                                        )
                                    ]

                                #
                                # Get absolute max across a vector and Compute SFC
                                #
                                is_unswapped_fp4_output = (
                                    not self.swap_ab
                                    and self.c_dtype == cutlass.Float4E2M1FN
                                )
                                needs_unswapped_fp4_scale_tail_store = (
                                    unswapped_fp4_sfc
                                    and self.c_dtype == cutlass.Float4E2M1FN
                                    and self.epi_tile[1] == 32
                                )
                                rcp_lim = self.get_dtype_rcp_limits(self.c_dtype)
                                if cutlass.const_expr(is_unswapped_fp4_output):
                                    self.reduce_unswapped_fp4_sfc(
                                        tCompute,
                                        output_cid_full,
                                        c_m,
                                        c_n,
                                        epi_tidx,
                                        sSFCExchange,
                                    )
                                    tTR_rAcc_frg = cute.logical_divide(
                                        tCompute, cute.make_layout(1)
                                    )
                                elif cutlass.const_expr(not self.swap_ab):
                                    tTR_rAcc_frg = cute.logical_divide(
                                        tCompute, cute.make_layout(self.sf_vec_size)
                                    )
                                else:
                                    sf_within = self.sf_vec_size // 8
                                    tTR_rAcc_frg = cute.logical_divide(
                                        tCompute, cute.make_layout(sf_within)
                                    )
                                acc_frg = tTR_rAcc_frg.load()
                                acc_frg = epilogue_op(acc_frg)

                                # Apply element-wise absolute value using math.absf (supports vectors)
                                abs_acc_frg_ir = math.absf(acc_frg.ir_value())
                                abs_acc_frg = type(acc_frg)(
                                    abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype
                                )

                                if cutlass.const_expr(
                                    is_unswapped_fp4_output
                                    and not needs_unswapped_fp4_scale_tail_store
                                ):
                                    self.generate_unswapped_fp4_sfc(
                                        output_cid_full,
                                        c_m,
                                        c_n,
                                        sSFCExchange,
                                        tCrSFC,
                                        tCrSFC_pvscale,
                                        rcp_lim,
                                        norm_const,
                                    )
                                elif cutlass.const_expr(not self.swap_ab):
                                    for vi in cutlass.range_constexpr(
                                        abs_acc_frg.shape[1]
                                    ):
                                        tCrSFC_pvscale[vi] = abs_acc_frg[
                                            None, vi
                                        ].reduce(
                                            cute.ReductionOp.MAX,
                                            cutlass.Float32(0.0),
                                            0,  # Use 0.0 as init for abs values
                                        )
                                    if cutlass.const_expr(self.vectorized_f32):
                                        for vi in cutlass.range_constexpr(
                                            0, abs_acc_frg.shape[1], 2
                                        ):
                                            (
                                                tCrSFC_pvscale[vi],
                                                tCrSFC_pvscale[vi + 1],
                                            ) = cute.arch.mul_packed_f32x2(
                                                (
                                                    tCrSFC_pvscale[vi],
                                                    tCrSFC_pvscale[vi + 1],
                                                ),
                                                (rcp_lim, rcp_lim),
                                            )
                                            (
                                                tCrSFC_pvscale[vi],
                                                tCrSFC_pvscale[vi + 1],
                                            ) = cute.arch.mul_packed_f32x2(
                                                (
                                                    tCrSFC_pvscale[vi],
                                                    tCrSFC_pvscale[vi + 1],
                                                ),
                                                (norm_const, norm_const),
                                            )
                                    else:
                                        scale = rcp_lim * norm_const
                                        for vi in cutlass.range_constexpr(
                                            abs_acc_frg.shape[1]
                                        ):
                                            tCrSFC_pvscale[vi] *= scale
                                else:
                                    zero_f32 = cutlass.Float32(0.0)
                                    output_half = warp_idx & 1
                                    sfc_cid_sub = output_cid_full[
                                        (None, None, None, epi_m_up, epi_n_up)
                                    ]
                                    output_coords = sfc_cid_sub[
                                        (None, output_half, None)
                                    ]
                                    if cutlass.const_expr(self.sf_vec_size == 16):
                                        sfc_fragment_layout = cute.make_layout(
                                            (2, cute.size(tCrSFC) // 2),
                                            stride=(1, 4),
                                        )
                                        # Each SF16 block is owned by one warp:
                                        # eight lanes contribute two values each.
                                        # The four modulo-4 lane groups reduce four
                                        # independent blocks without shared atomics.
                                        for ti in cutlass.range_constexpr(
                                            cute.size(tCrSFC)
                                        ):
                                            e0 = sfc_fragment_layout(ti)
                                            e1 = e0 + 2
                                            value0 = tCompute[e0]
                                            value1 = tCompute[e1]
                                            pair_max = cute.arch.fmax(
                                                cute.arch.fmax(
                                                    value0, zero_f32 - value0
                                                ),
                                                cute.arch.fmax(
                                                    value1, zero_f32 - value1
                                                ),
                                            )
                                            block_max = pair_max
                                            for offset in (4, 8, 16):
                                                block_max = cute.arch.fmax(
                                                    block_max,
                                                    cute.arch.shuffle_sync_bfly(
                                                        block_max, offset=offset
                                                    ),
                                                )
                                            tCrSFC_pvscale[ti] = (
                                                block_max * rcp_lim * norm_const
                                            )
                                    else:
                                        self.generate_swapped_blk32_sfc(
                                            tCompute,
                                            output_coords,
                                            epi_tidx,
                                            sSFCExchange,
                                            tCrSFC,
                                            tCrSFC_pvscale,
                                            rcp_lim,
                                            norm_const,
                                        )

                                # TODO: need to add f32x2 -> f8x2 conversion
                                if cutlass.const_expr(
                                    not needs_unswapped_fp4_scale_tail_store
                                ):
                                    tCrSFC.store(
                                        tCrSFC_pvscale.load().to(self.sf_dtype)
                                    )

                                #
                                # Store SFC to global memory
                                #
                                if cutlass.const_expr(not self.swap_ab):
                                    if cutlass.const_expr(
                                        unswapped_fp4_sfc
                                        and self.c_dtype == cutlass.Float4E2M1FN
                                        and self.sf_vec_size == 16
                                        and not needs_unswapped_fp4_scale_tail_store
                                    ):
                                        output_n_base = (
                                            output_tile_coord_mnl[1]
                                            * self.cta_tile_shape_mnk_c[1]
                                        )
                                        if (
                                            output_n_base + self.cta_tile_shape_mnk_c[1]
                                            <= mC_mnl.shape[1]
                                        ):
                                            # Preserve partitioned ownership between
                                            # peer CTAs for full 2CTA tiles.
                                            cute.autovec_copy(
                                                tCrSFC,
                                                cute.group_modes(
                                                    tCgSFC,
                                                    0,
                                                    cute.rank(tCgSFC),
                                                ),
                                            )
                                        else:
                                            self.store_unswapped_sfc_tail(
                                                tCrSFC,
                                                output_cid_full,
                                                epi_m_up,
                                                epi_n_up,
                                                tile_info[0],
                                                output_tile_coord_mnl[1],
                                                mC_mnl,
                                                mSFC_mnl,
                                            )
                                    elif cutlass.const_expr(
                                        self.c_dtype == cutlass.Float4E2M1FN
                                    ):
                                        scales_per_row = (
                                            self.epi_tile[1] // self.sf_vec_size
                                        )
                                        sfc_threads = self.threads_per_warp * len(
                                            self.epilog_warp_id
                                        )
                                        sfc_thread_layout = cute.make_ordered_layout(
                                            (
                                                sfc_threads,
                                                cute.ceil_div(
                                                    self.sfc_exchange_elems,
                                                    sfc_threads,
                                                ),
                                            ),
                                            order=(0, 1),
                                        )
                                        exchange = self.make_sfc_exchange(
                                            sSFCExchange,
                                            (self.epi_tile[0], scales_per_row),
                                        )
                                        for si in cutlass.range_constexpr(
                                            cute.ceil_div(
                                                self.sfc_exchange_elems,
                                                sfc_threads,
                                            )
                                        ):
                                            scale_idx = sfc_thread_layout(
                                                (epi_tidx, si)
                                            )
                                            if scale_idx < self.sfc_exchange_elems:
                                                local_scale, local_row = cute.idx2crd(
                                                    scale_idx,
                                                    (
                                                        scales_per_row,
                                                        self.epi_tile[0],
                                                    ),
                                                )
                                                g_row = (
                                                    tile_info[0]
                                                    * self.cta_tile_shape_mnk_c[0]
                                                    + epi_m_up * self.epi_tile[0]
                                                    + local_row
                                                )
                                                g_col = (
                                                    output_tile_coord_mnl[1]
                                                    * self.cta_tile_shape_mnk_c[1]
                                                    + c_n * self.epi_tile[1]
                                                    + local_scale * self.sf_vec_size
                                                )
                                                if cute.elem_less(
                                                    (g_row, g_col, 0), mC_mnl.shape
                                                ):
                                                    mSFC_mnl[(g_row, g_col, 0)] = (
                                                        self.sf_dtype(
                                                            exchange[
                                                                (local_row, local_scale)
                                                            ]
                                                            * rcp_lim
                                                            * norm_const
                                                        )
                                                    )
                                    else:
                                        cute.autovec_copy(tCrSFC, tCgSFC)
                                else:
                                    interm_base = (
                                        output_tile_coord_mnl[0]
                                        * self.cta_tile_shape_mnk_c[0]
                                        + epi_m_up * self.epi_tile[0]
                                    )
                                    slot_base = (
                                        output_tile_coord_mnl[1]
                                        * self.cta_tile_shape_mnk_c[1]
                                        + epi_n_up * self.epi_tile[1]
                                    )
                                    scales_per_token = (
                                        self.epi_tile[0] // self.sf_vec_size
                                    )
                                    scale_owner_layout = cute.make_layout(
                                        (
                                            self.cta_tile_shape_mnk_c[0]
                                            // self.sf_vec_size,
                                            mC_mnl.shape[1],
                                        )
                                    )
                                    if cutlass.const_expr(self.sf_vec_size == 16):
                                        # Each modulo-4 group has the same
                                        # reduced scale and coordinate. Its
                                        # first lane writes the warp-owned block.
                                        for ti in cutlass.range_constexpr(
                                            cute.size(tCrSFC)
                                        ):
                                            sfc_ti = sfc_fragment_layout(ti)
                                            lcoord = output_coords[sfc_ti]
                                            g_slot = slot_base + (
                                                lcoord[1] % self.epi_tile[1]
                                            )
                                            g_interm = interm_base + (
                                                lcoord[0] % self.epi_tile[0]
                                            )
                                            scale_owner = (
                                                scale_owner_layout(
                                                    (
                                                        g_interm // self.sf_vec_size,
                                                        g_slot,
                                                    )
                                                )
                                                % self.split_k
                                            )
                                            if (
                                                cute.arch.lane_idx() < 4
                                                and (
                                                    cutlass.const_expr(
                                                        not self.distributed_split_k
                                                    )
                                                    or split_rank == scale_owner
                                                )
                                                and cute.elem_less(
                                                    (g_interm, g_slot, 0),
                                                    mC_mnl.shape,
                                                )
                                            ):
                                                self.store_swapped_sfc(
                                                    mSFC_mnl,
                                                    g_slot,
                                                    g_interm,
                                                    tCrSFC[ti],
                                                )
                                    elif epi_tidx < self.sfc_exchange_elems:
                                        local_scale, local_slot = cute.idx2crd(
                                            epi_tidx,
                                            (scales_per_token, self.epi_tile[1]),
                                        )
                                        exchange = self.make_sfc_exchange(
                                            sSFCExchange,
                                            (self.epi_tile[1], scales_per_token),
                                        )
                                        g_slot = slot_base + local_slot
                                        g_interm = (
                                            interm_base + local_scale * self.sf_vec_size
                                        )
                                        scale_owner = (
                                            scale_owner_layout(
                                                (
                                                    g_interm // self.sf_vec_size,
                                                    g_slot,
                                                )
                                            )
                                            % self.split_k
                                        )
                                        if (
                                            cutlass.const_expr(
                                                not self.distributed_split_k
                                            )
                                            or split_rank == scale_owner
                                        ) and cute.elem_less(
                                            (g_interm, g_slot, 0),
                                            mC_mnl.shape,
                                        ):
                                            quant_scale = self.sf_dtype(
                                                exchange[(local_slot, local_scale)]
                                                * rcp_lim
                                                * norm_const
                                            )
                                            self.store_swapped_sfc(
                                                mSFC_mnl,
                                                g_slot,
                                                g_interm,
                                                quant_scale,
                                            )

                                #
                                # Compute quantized output values and convert to C type
                                #
                                # TODO: need to add f8x2 -> f32x2 conversion
                                if cutlass.const_expr(not is_unswapped_fp4_output):
                                    tCrSFC_qpvscale_up = tCrSFC.load().to(
                                        cutlass.Float32
                                    )
                                fp32_max = cutlass.Float32(FP32_MAX)
                                if cutlass.const_expr(is_unswapped_fp4_output):
                                    self.quantize_unswapped_fp4(
                                        tCompute,
                                        output_cid_full,
                                        c_m,
                                        c_n,
                                        sSFCExchange,
                                        rcp_lim,
                                        norm_const,
                                    )
                                elif cutlass.const_expr(
                                    not self.swap_ab and self.vectorized_f32
                                ):
                                    for vi in cutlass.range_constexpr(
                                        0, cute.size(tCrSFC), 2
                                    ):
                                        acc_scale = cute.arch.mul_packed_f32x2(
                                            (
                                                cute.arch.rcp_approx(
                                                    tCrSFC_qpvscale_up[vi]
                                                ),
                                                cute.arch.rcp_approx(
                                                    tCrSFC_qpvscale_up[vi + 1]
                                                ),
                                            ),
                                            (norm_const, norm_const),
                                        )
                                        acc_scale_min0 = fmin(
                                            acc_scale[0], fp32_max, nan=True
                                        )
                                        acc_scale_min1 = fmin(
                                            acc_scale[1], fp32_max, nan=True
                                        )

                                        vec0 = tTR_rAcc_frg[None, vi]
                                        vec1 = tTR_rAcc_frg[None, vi + 1]
                                        for ei in cutlass.range_constexpr(
                                            self.sf_vec_size
                                        ):
                                            vec0[ei], vec1[ei] = (
                                                cute.arch.mul_packed_f32x2(
                                                    (vec0[ei], vec1[ei]),
                                                    (acc_scale_min0, acc_scale_min1),
                                                )
                                            )
                                elif cutlass.const_expr(not self.swap_ab):
                                    for vi in cutlass.range_constexpr(
                                        cute.size(tCrSFC)
                                    ):
                                        # TODO:Need to add E8M0 rcp approximation
                                        acc_scale = norm_const * cute.arch.rcp_approx(
                                            tCrSFC_qpvscale_up[vi]
                                        )
                                        acc_scale = fmin(acc_scale, fp32_max, nan=True)

                                        vec = tTR_rAcc_frg[None, vi]
                                        for ei in cutlass.range_constexpr(
                                            self.sf_vec_size
                                        ):
                                            vec[ei] = vec[ei] * acc_scale
                                elif cutlass.const_expr(self.sf_vec_size == 16):
                                    for e in cutlass.range_constexpr(
                                        cute.size(tCompute)
                                    ):
                                        sfc_idx = (e & 1) + 2 * (e >> 2)
                                        acc_scale = norm_const * cute.arch.rcp_approx(
                                            tCrSFC_qpvscale_up[sfc_idx]
                                        )
                                        acc_scale = fmin(acc_scale, fp32_max, nan=True)
                                        tCompute[e] = tCompute[e] * acc_scale
                                else:
                                    for e in cutlass.range_constexpr(
                                        cute.size(tCompute)
                                    ):
                                        lcoord = output_coords[e]
                                        scale_idx = (
                                            lcoord[1] % self.epi_tile[1]
                                        ) * scales_per_token + (
                                            lcoord[0] % self.epi_tile[0]
                                        ) // self.sf_vec_size
                                        quant_scale = self.sf_dtype(
                                            sSFCExchange[scale_idx]
                                            * rcp_lim
                                            * norm_const
                                        )
                                        acc_scale = norm_const * cute.arch.rcp_approx(
                                            cutlass.Float32(quant_scale)
                                        )
                                        tCompute[e] *= fmin(
                                            acc_scale, fp32_max, nan=True
                                        )

                                if cutlass.const_expr(self.swap_ab and self.gated):
                                    output_half = warp_idx & 1
                                    converted = tCompute.load().to(self.c_dtype)
                                    if cutlass.const_expr(self.fp4_swap):
                                        tTR_rC_packed.store(
                                            converted.bitcast(cutlass.Uint16)
                                        )
                                    else:
                                        tTR_rC[(None, output_half, None)].store(
                                            converted
                                        )
                                else:
                                    acc_vec = tiled_copy_r2s.retile(tCompute).load()
                                    tRS_rC.store(acc_vec.to(self.c_dtype))
                            else:
                                #
                                # Convert to C type
                                #
                                if cutlass.const_expr(self.swap_ab and self.gated):
                                    output_half = warp_idx & 1
                                    acc_vec = epilogue_op(
                                        tCompute.load().to(self.c_dtype)
                                    )
                                    tTR_rC[(None, output_half, None)].store(acc_vec)
                                else:
                                    acc_vec = tiled_copy_r2s.retile(tCompute).load()
                                    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                                    tRS_rC.store(acc_vec)

                            if cutlass.const_expr(self.distributed_split_k):
                                split_output_half = warp_idx & 1
                                output_cid_sub = output_cid_full[
                                    (None, None, None, epi_m_up, epi_n_up)
                                ]
                                store_output_coords = output_cid_sub[
                                    (None, split_output_half, None)
                                ]
                                output_gC_frg = tTR_gC_output_tile[
                                    (None, None, None, c_m, c_n)
                                ]
                                output_gC = output_gC_frg[
                                    (None, split_output_half, None)
                                ]
                                output_rC = tTR_rC[(None, split_output_half, None)]
                                store_scale_vectors_per_tile = (
                                    self.cta_tile_shape_mnk_c[0] // self.sf_vec_size
                                )
                                store_interm_base = (
                                    output_tile_coord_mnl[0]
                                    * self.cta_tile_shape_mnk_c[0]
                                )
                                store_slot_base = (
                                    output_tile_coord_mnl[1]
                                    * self.cta_tile_shape_mnk_c[1]
                                )
                                output_pred = cute.make_rmem_tensor(
                                    output_rC.shape, cutlass.Boolean
                                )
                                for output_idx in cutlass.range_constexpr(
                                    cute.size(output_rC)
                                ):
                                    store_output_coord = store_output_coords[output_idx]
                                    store_output_owner = (
                                        (store_slot_base + store_output_coord[1])
                                        * store_scale_vectors_per_tile
                                        + (store_interm_base + store_output_coord[0])
                                        // self.sf_vec_size
                                    ) % self.split_k
                                    output_pred[output_idx] = cutlass.Boolean(False)
                                    if split_rank == store_output_owner:
                                        output_pred[output_idx] = cute.elem_less(
                                            (
                                                store_interm_base
                                                + store_output_coord[0],
                                                store_slot_base + store_output_coord[1],
                                                0,
                                            ),
                                            mC_mnl.shape,
                                        )
                                cute.basic_copy_if(output_pred, output_rC, output_gC)

                        #
                        # Store C to shared memory
                        #
                        num_prev_subtiles = num_prev_subtiles + 1
                        if cutlass.const_expr(not self.distributed_split_k):
                            c_buffer = num_prev_subtiles % self.num_c_stage
                            if cutlass.const_expr(self.split_k == 1) or split_rank == 0:
                                if cutlass.const_expr(self.swap_ab and self.gated):
                                    output_half = warp_idx & 1
                                    if cutlass.const_expr(self.fp4_swap):
                                        self.pack_fp4(
                                            tTR_rC_packed,
                                            tRS_rC,
                                        )
                                        output_rC = tRS_rC
                                        output_sC = tRS_sC[(None, c_m, c_n, c_buffer)]
                                    else:
                                        output_rC = tRS_rC[(None, output_half, None)]
                                        output_sC = tRS_sC[
                                            (
                                                None,
                                                output_half,
                                                None,
                                                c_buffer,
                                            )
                                        ]
                                    cute.copy(
                                        tiled_copy_r2s,
                                        output_rC,
                                        output_sC,
                                    )
                                else:
                                    cute.copy(
                                        tiled_copy_r2s,
                                        tRS_rC,
                                        tRS_sC[(None, None, None, c_buffer)],
                                    )
                                # Fence and barrier to make sure smem write is visible to TMA
                                cute.arch.fence_proxy(
                                    "async.shared",
                                    space="cta",
                                )
                                self.epilog_sync_barrier.arrive_and_wait()
                                #
                                # TMA store C to global memory
                                #
                                if warp_idx == self.epilog_warp_id[0]:
                                    cute.copy(
                                        tma_atom_c,
                                        bSG_sC[(None, c_buffer)],
                                        bSG_gC[(None, c_m, c_n)],
                                    )
                                    c_pipeline.producer_commit()
                                    c_pipeline.producer_acquire()
                                self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                if cutlass.const_expr(not self.overlapping_accum):
                    # Complete TMEM loads before releasing the accumulator stage.
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                if cutlass.const_expr(self.use_a_per_token_scale):
                    tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        if cutlass.const_expr(not self.distributed_split_k):
            if warp_idx <= self.epilog_warp_id[-1]:
                c_pipeline = PipelineTmaStore.create(
                    num_stages=self.num_c_stage,
                    producer_group=CooperativeGroup(
                        Agent.Thread,
                        32 * len(self.epilog_warp_id),
                    ),
                )
                # Drain pending C stores.
                c_pipeline.producer_tail()

        if cutlass.const_expr(self.pdl_count is not None):
            griddepcontrol_launch_dependents()

    @cute.jit
    def pack_fp4(
        self,
        output_rC: cute.Tensor,
        packed_rC: cute.Tensor,
    ) -> None:
        """Pack an FP4 register fragment for the M-major SMEM tile."""
        source_column_pair, source_row_pair, output_row_half = cute.idx2crd(
            cute.arch.lane_idx(), (4, 4, 2)
        )
        # (source_column_pair, source_row_half, source_row_pair)
        source_lane_layout = cute.make_layout((4, 2, 4), stride=(1, 4, 8))
        source_lane0 = source_lane_layout((source_column_pair, 0, source_row_pair))
        source_lane1 = source_lane_layout((source_column_pair, 1, source_row_pair))
        nibble_shift_layout = cute.make_layout(2, stride=8)
        for output_idx in cutlass.range_constexpr(cute.size(output_rC)):
            source_word = cutlass.Uint32(output_rC[output_idx])
            source_word0 = cute.arch.shuffle_sync(source_word, source_lane0)
            source_word1 = cute.arch.shuffle_sync(source_word, source_lane1)
            nibble_shift = cutlass.Uint32(nibble_shift_layout(output_row_half))
            output_byte0 = cutlass.Uint8(
                ((source_word0 >> nibble_shift) & 0xF)
                | (((source_word1 >> nibble_shift) & 0xF) << 4)
            )
            output_byte1 = cutlass.Uint8(
                ((source_word0 >> (nibble_shift + 4)) & 0xF)
                | (((source_word1 >> (nibble_shift + 4)) & 0xF) << 4)
            )
            packed_rC[2 * output_idx] = output_byte0
            packed_rC[2 * output_idx + 1] = output_byte1

    @cute.jit
    def store_swapped_sfc(
        self,
        mSFC_mnl: cute.Tensor,
        g_slot: cutlass.Int32,
        g_interm: cutlass.Int32,
        value,
    ) -> None:
        """Store one swapped SFC value."""
        if cutlass.const_expr(self.use_compact_sfc):
            # (token, feature // sf_vec_size, L)
            mSFC_mnl[(g_slot, g_interm // self.sf_vec_size, 0)] = value
        else:
            # ((Atom_M, Rest_M), (Atom_K, Rest_K), RestL)
            mSFC_mnl[(g_slot, g_interm, 0)] = value

    @cute.jit
    def make_sfc_exchange(self, sSFCExchange: cute.Tensor, shape) -> cute.Tensor:
        """View the flat exchange allocation by its logical scale coordinates."""
        return cute.make_tensor(
            sSFCExchange.iterator,
            cute.make_ordered_layout(shape, order=(1, 0)),
        )

    @cute.jit
    def sfc_exchange_coord(self, output_coord):
        """Map an epilogue output coordinate to its logical exchange slot."""
        local_m, _ = cute.idx2crd(
            output_coord[0],
            (
                self.epi_tile[0],
                cute.ceil_div(self.cta_tile_shape_mnk_c[0], self.epi_tile[0]),
            ),
        )
        local_n, _ = cute.idx2crd(
            output_coord[1],
            (
                self.epi_tile[1],
                cute.ceil_div(self.cta_tile_shape_mnk_c[1], self.epi_tile[1]),
            ),
        )
        if cutlass.const_expr(self.swap_ab):
            _, local_scale = cute.idx2crd(
                local_m,
                (self.sf_vec_size, self.epi_tile[0] // self.sf_vec_size),
            )
            return local_n, local_scale
        _, local_scale = cute.idx2crd(
            local_n,
            (self.sf_vec_size, self.epi_tile[1] // self.sf_vec_size),
        )
        return local_m, local_scale

    @cute.jit
    def generate_swapped_blk32_sfc(
        self,
        tCompute: cute.Tensor,
        output_coords: cute.Tensor,
        epi_tidx: cutlass.Int32,
        sSFCExchange: cute.Tensor,
        tCrSFC: cute.Tensor,
        tCrSFC_pvscale: cute.Tensor,
        rcp_lim: cutlass.Float32,
        norm_const: cutlass.Float32,
    ) -> None:
        """Reduce and compute swapped block-32 SFC through shared memory."""
        zero_f32 = cutlass.Float32(0.0)
        scales_per_token = self.epi_tile[0] // self.sf_vec_size
        exchange = self.make_sfc_exchange(
            sSFCExchange, (self.epi_tile[1], scales_per_token)
        )
        exchange_layout = exchange.layout
        fragment_coord_layout = cute.make_layout(
            (2, cute.size(tCompute) // 4), stride=(1, 4)
        )
        if epi_tidx < self.sfc_exchange_elems:
            local_scale, local_slot = cute.idx2crd(
                epi_tidx, (scales_per_token, self.epi_tile[1])
            )
            exchange[(local_slot, local_scale)] = zero_f32
        self.epilog_sync_barrier.arrive_and_wait()
        for pair_idx in cutlass.range_constexpr(cute.size(tCompute) // 2):
            e0 = fragment_coord_layout(pair_idx)
            e1 = e0 + 2
            lcoord0 = output_coords[e0]
            lcoord1 = output_coords[e1]
            scale_coord0 = self.sfc_exchange_coord(lcoord0)
            scale_coord1 = self.sfc_exchange_coord(lcoord1)
            scale_idx0 = exchange_layout(scale_coord0)
            scale_idx1 = exchange_layout(scale_coord1)
            value0 = tCompute[e0]
            value1 = tCompute[e1]
            abs_value0 = cute.arch.fmax(value0, zero_f32 - value0)
            abs_value1 = cute.arch.fmax(value1, zero_f32 - value1)
            if scale_idx0 == scale_idx1:
                cute.arch.atomic_fmax(
                    cute.domain_offset(scale_coord0, exchange).iterator,
                    cute.arch.fmax(abs_value0, abs_value1),
                    sem="relaxed",
                    scope="cta",
                )
            else:
                cute.arch.atomic_fmax(
                    cute.domain_offset(scale_coord0, exchange).iterator,
                    abs_value0,
                    sem="relaxed",
                    scope="cta",
                )
                cute.arch.atomic_fmax(
                    cute.domain_offset(scale_coord1, exchange).iterator,
                    abs_value1,
                    sem="relaxed",
                    scope="cta",
                )
        self.epilog_sync_barrier.arrive_and_wait()

        for ti in cutlass.range_constexpr(cute.size(tCrSFC)):
            sfc_ti = fragment_coord_layout(ti)
            lcoord = output_coords[sfc_ti]
            scale_coord = self.sfc_exchange_coord(lcoord)
            tCrSFC_pvscale[ti] = exchange[scale_coord] * rcp_lim * norm_const

    # Unswapped FP4 scales require coordinate-based cross-warp reduction.
    @cute.jit
    def store_unswapped_sfc_tail(
        self,
        tCrSFC: cute.Tensor,
        output_cid_full: cute.Tensor,
        c_m,
        c_n,
        output_m_tile_idx,
        output_n_tile_idx,
        mC_mnl: cute.Tensor,
        mSFC_mnl: cute.Tensor,
    ) -> None:
        """Store an unswapped FP4 SFC tail by logical coordinate."""
        output_coords = output_cid_full[(None, None, None, c_m, c_n)]
        sfc_cid_frg = cute.logical_divide(
            output_coords,
            cute.make_layout(self.sf_vec_size),
        )
        # The scheduler M coordinate identifies the physical CTA tile. The MMA
        # coordinate is divided by the CTA-group size and therefore cannot
        # distinguish the peer CTA in a 2CTA instruction.
        output_m_base = output_m_tile_idx * self.cta_tile_shape_mnk_c[0]
        output_n_base = output_n_tile_idx * self.cta_tile_shape_mnk_c[1]
        for ti in cutlass.range_constexpr(cute.size(tCrSFC)):
            lcoord = sfc_cid_frg[(0, ti)]
            global_coord = (
                output_m_base + lcoord[0],
                output_n_base + lcoord[1],
                0,
            )
            if cute.elem_less(global_coord, mC_mnl.shape):
                mSFC_mnl[global_coord] = tCrSFC[ti]

    @cute.jit
    def reduce_unswapped_fp4_sfc(
        self,
        tCompute: cute.Tensor,
        output_cid_full: cute.Tensor,
        c_m,
        c_n,
        epi_tidx: cutlass.Int32,
        sSFCExchange: cute.Tensor,
    ) -> None:
        """Reduce unswapped FP4 output absolute maxima into shared memory."""
        zero_f32 = cutlass.Float32(0.0)
        output_coords = output_cid_full[(None, None, None, c_m, c_n)]
        scales_per_row = self.epi_tile[1] // self.sf_vec_size
        exchange = self.make_sfc_exchange(
            sSFCExchange, (self.epi_tile[0], scales_per_row)
        )
        sfc_threads = self.threads_per_warp * len(self.epilog_warp_id)
        sfc_thread_layout = cute.make_ordered_layout(
            (
                sfc_threads,
                cute.ceil_div(self.sfc_exchange_elems, sfc_threads),
            ),
            order=(0, 1),
        )
        for si in cutlass.range_constexpr(
            cute.ceil_div(self.sfc_exchange_elems, sfc_threads)
        ):
            scale_idx = sfc_thread_layout((epi_tidx, si))
            if scale_idx < self.sfc_exchange_elems:
                local_scale, local_row = cute.idx2crd(
                    scale_idx, (scales_per_row, self.epi_tile[0])
                )
                exchange[(local_row, local_scale)] = zero_f32
        self.epilog_sync_barrier.arrive_and_wait()
        for e in cutlass.range_constexpr(cute.size(tCompute)):
            lcoord = output_coords[e]
            scale_coord = self.sfc_exchange_coord(lcoord)
            value = tCompute[e]
            cute.arch.atomic_fmax(
                cute.domain_offset(scale_coord, exchange).iterator,
                cute.arch.fmax(value, zero_f32 - value),
                sem="relaxed",
                scope="cta",
            )
        self.epilog_sync_barrier.arrive_and_wait()

    @cute.jit
    def generate_unswapped_fp4_sfc(
        self,
        output_cid_full: cute.Tensor,
        c_m,
        c_n,
        sSFCExchange: cute.Tensor,
        tCrSFC: cute.Tensor,
        tCrSFC_pvscale: cute.Tensor,
        rcp_lim: cutlass.Float32,
        norm_const: cutlass.Float32,
    ) -> None:
        """Compute unswapped FP4 SFC from shared absolute maxima."""
        output_coords = output_cid_full[(None, None, None, c_m, c_n)]
        scales_per_row = self.epi_tile[1] // self.sf_vec_size
        exchange = self.make_sfc_exchange(
            sSFCExchange, (self.epi_tile[0], scales_per_row)
        )
        sfc_cid_frg = cute.logical_divide(
            output_coords,
            cute.make_layout(self.sf_vec_size),
        )
        for ti in cutlass.range_constexpr(cute.size(tCrSFC)):
            lcoord = sfc_cid_frg[(0, ti)]
            scale_coord = self.sfc_exchange_coord(lcoord)
            tCrSFC_pvscale[ti] = exchange[scale_coord] * rcp_lim * norm_const

    @cute.jit
    def quantize_unswapped_fp4(
        self,
        tCompute: cute.Tensor,
        output_cid_full: cute.Tensor,
        c_m,
        c_n,
        sSFCExchange: cute.Tensor,
        rcp_lim: cutlass.Float32,
        norm_const: cutlass.Float32,
    ) -> None:
        """Quantize unswapped FP4 output using shared absolute maxima."""
        output_coords = output_cid_full[(None, None, None, c_m, c_n)]
        scales_per_row = self.epi_tile[1] // self.sf_vec_size
        exchange = self.make_sfc_exchange(
            sSFCExchange, (self.epi_tile[0], scales_per_row)
        )
        fp32_max = cutlass.Float32(FP32_MAX)
        for e in cutlass.range_constexpr(cute.size(tCompute)):
            lcoord = output_coords[e]
            scale_coord = self.sfc_exchange_coord(lcoord)
            quant_scale = self.sf_dtype(exchange[scale_coord] * rcp_lim * norm_const)
            acc_scale = norm_const * cute.arch.rcp_approx(cutlass.Float32(quant_scale))
            tCompute[e] *= fmin(acc_scale, fp32_max, nan=True)

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[
        cute.TiledCopy,
        cute.TiledCopy,
        cute.Tensor,
        cute.Tensor,
        cute.Tensor,
        cute.Tensor,
        cute.Tensor,
    ]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory
        (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing
            (tiled_copy_t2r, tiled_copy_epi, tTR_tAcc, tTR_rAcc_full,
            tTR_rAcc_up, tTR_rAcc_gate, tTR_rAcc_panel) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tiled_copy_epi: The copy defining logical output ownership
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc_full: The full 32-row load fragment
            - tTR_rAcc_up: The partitioned accumulator tensor for acc up
            - tTR_rAcc_gate: The partitioned accumulator tensor for acc gate
            - tTR_rAcc_panel: A 16-row interleaved load fragment
        """
        # Make tiledCopy for tensor memory load
        unswapped_gated = (
            not self.swap_ab and self.gated and self.weight_interleave == 16
        )
        if cutlass.const_expr(unswapped_gated and self.cta_tile_shape_mnk[1] != 128):
            copy_atom_epi = cute.make_copy_atom(
                tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(epi_tile[1] // 2)),
                self.acc_dtype,
            )
            copy_atom_t2r = cute.make_copy_atom(
                tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(epi_tile[1])),
                self.acc_dtype,
            )
        else:
            copy_atom_epi = blackwell_helpers.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                self.c_layout,
                (cutlass.Float8E4M3FN if self.fp4_swap else self.c_dtype),
                self.acc_dtype,
                epi_tile,
                use_2cta_instrs,
            )
            copy_atom_t2r = blackwell_helpers.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                self.c_layout,
                (cutlass.Float8E4M3FN if self.fp4_swap else self.c_dtype),
                self.acc_dtype,
                epi_tile,
                use_2cta_instrs,
            )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        tiled_copy_epi = tcgen05.make_tmem_copy(
            copy_atom_epi, tAcc_epi[(None, None, 0, 0, 0)]
        )
        if cutlass.const_expr(unswapped_gated and self.cta_tile_shape_mnk[1] != 128):
            tmem_epi_tile = (epi_tile[0], epi_tile[1] * 2)
            tAcc_t2r = cute.flat_divide(tAcc[((None, None), 0, 0, None)], tmem_epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_t2r[(None, None, 0, 0, 0)]
            )
        else:
            tAcc_t2r = cute.flat_divide(
                tAcc[((None, None), 0, 0, None)],
                epi_tile,
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_t2r[(None, None, 0, 0, 0)]
            )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_t2r)
        if cutlass.const_expr(unswapped_gated and self.cta_tile_shape_mnk[1] != 128):
            tTR_cAcc_panel = thr_copy_t2r.partition_D(
                cute.make_identity_tensor(tmem_epi_tile)
            )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        thr_copy_epi = tiled_copy_epi.get_slice(tidx)
        tTR_gC = thr_copy_epi.partition_D(gC_mnl_epi)

        tTR_slot_shape = tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape
        if cutlass.const_expr(unswapped_gated):
            if cutlass.const_expr(self.cta_tile_shape_mnk[1] != 128):
                panel_layout = cute.make_layout(tTR_cAcc_panel[(None, 0, 0)].shape)
            else:
                panel_layout = cute.make_layout(tTR_slot_shape)
            # Load both panels; views select the interleaved up/gate groups.
            tTR_rAcc_full = cute.make_rmem_tensor(
                cute.append(
                    panel_layout,
                    cute.make_layout(2, stride=cute.cosize(panel_layout)),
                ),
                self.acc_dtype,
            )
            fragment_layout = cute.make_layout(tTR_slot_shape)
            fragment_size = cute.size(tTR_slot_shape)
            pairs_per_panel = epi_tile[1] // 32
            deinterleave_layout = cute.make_layout(
                (16, pairs_per_panel, 2),
                stride=(1, 32, fragment_size),
            )
            deinterleaved_fragment_layout = cute.composition(
                deinterleave_layout, fragment_layout
            )
            tTR_rAcc_deinterleaved = cute.make_tensor(
                tTR_rAcc_full.iterator,
                cute.append(
                    deinterleaved_fragment_layout,
                    cute.make_layout(2, stride=16),
                ),
            )
            tTR_rAcc_up = tTR_rAcc_deinterleaved[(None, None, None, 0)]
            tTR_rAcc_gate = tTR_rAcc_deinterleaved[(None, None, None, 1)]
            tTR_rAcc_panel = tTR_rAcc_full
        elif cutlass.const_expr(self.swap_ab and self.gated):
            tTR_rAcc_full = cute.make_rmem_tensor(tTR_slot_shape, self.acc_dtype)
            # Use up/gate as layout views into
            # the same RMEM tensor
            tTR_rAcc_up = tTR_rAcc_full[(None, 0, None)]
            tTR_rAcc_gate = tTR_rAcc_full[(None, 1, None)]
        else:
            tTR_rAcc_up = cute.make_rmem_tensor(tTR_slot_shape, self.acc_dtype)
            if cutlass.const_expr(self.gated):
                tTR_rAcc_gate = cute.make_rmem_tensor(tTR_slot_shape, self.acc_dtype)
            else:
                tTR_rAcc_gate = tTR_rAcc_up
            tTR_rAcc_full = tTR_rAcc_up
        if cutlass.const_expr(not unswapped_gated):
            tTR_rAcc_panel = tTR_rAcc_up
        return (
            tiled_copy_t2r,
            tiled_copy_epi,
            tTR_tAcc,
            tTR_rAcc_full,
            tTR_rAcc_up,
            tTR_rAcc_gate,
            tTR_rAcc_panel,
        )

    def epilog_packed_smem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
        sC_layout: cute.ComposedLayout,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Partition packed output bytes for the M-major SMEM store."""
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Uint8,
            num_bits_per_copy=cutlass.Uint8.width,
        )
        # ((feature_pair, token_pair, warp), (byte, repetition))
        layout_tv = cute.make_layout(
            ((4, 8, 4), (2, self.epi_tile[1] // 8)),
            stride=((64, 1, 8), (32, 256)),
        )
        tiled_copy = cute.make_tiled_copy(
            copy_atom,
            layout_tv,
            (self.epi_tile[0] // 2, self.epi_tile[1]),
        )
        # (M_byte, N, STAGE)
        packed_sC = cute.make_tensor(
            cute.recast_ptr(
                sC.iterator,
                sC_layout.inner,
                dtype=cutlass.Uint8,
            ),
            cute.recast_layout(
                cutlass.Uint8.width,
                self.c_dtype.width,
                sC_layout.outer,
            ),
        )
        thread_copy = tiled_copy.get_slice(tidx)
        tRS_sC = thread_copy.partition_D(packed_sC)
        tRS_rC = cute.make_fragment_like(tRS_sC[(None, 0, 0, 0)])
        return tiled_copy, tRS_rC, tRS_sC

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register
        array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = blackwell_helpers.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        if cutlass.const_expr(self.swap_ab and self.gated):
            output_tidx = (tidx // 64) * 32 + tidx % 32
        else:
            output_tidx = tidx
        thr_copy_r2s = tiled_copy_r2s.get_slice(output_tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
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
        """Make tiledCopy for global memory store, then use it to:
        - partition register array (source) and global memory (destination) for none TMA store version;
        - partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing :
            - For TMA store: (tma_atom_c, bSG_sC, bSG_gC) where:
                - tma_atom_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        num_smem_capacity: int,
        occupancy: int,
        distributed_split_k: bool,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Vector size of scale factor.
        :type sf_vec_size: int
        :param num_smem_capacity: Total available shared memory capacity in bytes.
        :type num_smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # Default ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        num_c_stage = 0 if distributed_split_k else 2

        # Default Tile info stages
        num_tile_stage = 2

        # Calculate smem layout and size for one stage of A, B, and C
        a_smem_layout_stage_one = blackwell_helpers.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = blackwell_helpers.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )

        sfa_smem_layout_staged_one = blockscaled_layout.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        sfb_smem_layout_staged_one = blockscaled_layout.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = (
            blackwell_helpers.make_smem_layout_epi(
                c_dtype,
                c_layout,
                epi_tile,
                1,
            )
            if not distributed_split_k
            else None
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        # 1024B alignment
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = (
            cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
            if not distributed_split_k
            else 0
        )
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (
            num_smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        if not distributed_split_k:
            num_c_stage += (
                num_smem_capacity
                - occupancy * ab_bytes_per_stage * num_ab_stage
                - occupancy * (mbar_helpers_bytes + c_bytes)
            ) // (occupancy * c_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_c_stage, num_tile_stage  # type: ignore[return-value]

    @staticmethod
    def compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        raster_along_m: bool = False,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, raster_along_m=raster_along_m
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def needs_unpack_tma(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Decide whether TMA must use the UNPACK_U8 variant (U4_UNPACK_U8 /
        U6_UNPACK_U8) for narrow-precision operands.

        Unpack is required when:
          * Operand widths differ (mxf8f6f4 mixed-precision) — A and B must
            share a uniform byte-per-element SMEM layout, so the narrower
            operand is unpacked into 1B/elem containers in SMEM.
          * Either operand is 6-bit — there is no packed U6 TMA format,
            only U6_UNPACK_U8 exists.

        Otherwise (same-width and no 6-bit operand, e.g. f4xf4 / f8xf8 /
        f8E4M3xf8E5M2) TMA can use the natural packed format (U4 for 4-bit,
        U8 for 8-bit).

        :param a_dtype: Element data type of the A operand
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: Element data type of the B operand
        :type b_dtype: Type[cutlass.Numeric]
        :return: True if UNPACK_U8 TMA format must be used, False otherwise
        :rtype: bool
        """
        if a_dtype.width != b_dtype.width:
            return True
        if a_dtype.width == 6 or b_dtype.width == 6:
            return True
        return False

    @staticmethod
    def get_tma_atom_kind(
        atom_sm_cnt: cutlass.Int32, mcast: cutlass.Boolean
    ) -> Union[
        cpasync.CopyBulkTensorTileG2SMulticastOp, cpasync.CopyBulkTensorTileG2SOp
    ]:
        """
        Select the appropriate TMA copy atom based on the number of SMs and the multicast flag.

        :param atom_sm_cnt: The number of SMs
        :type atom_sm_cnt: cutlass.Int32
        :param mcast: The multicast flag
        :type mcast: cutlass.Boolean

        :return: The appropriate TMA copy atom kind
        :rtype: cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp

        :raise ValueError: If the atom_sm_cnt is invalid
        """
        if atom_sm_cnt == 2 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 2 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 1 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
        elif atom_sm_cnt == 1 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        raise ValueError(f"Invalid atom_sm_cnt: {atom_sm_cnt} and {mcast}")

    @staticmethod
    def get_dtype_rcp_limits(dtype: Type[cutlass.Numeric]) -> float:
        """
        Calculates the reciprocal of the maximum absolute value for a given data type.

        :param dtype: Data type
        :type dtype: Type[cutlass.Numeric]

        :return: An float representing the reciprocal of the maximum absolute value
        :rtype: float
        """
        if dtype == cutlass.Float4E2M1FN:
            return 1 / 6.0
        if dtype == cutlass.Float8E4M3FN:
            return 1 / 448.0
        if dtype == cutlass.Float8E5M2:
            return 1 / 128.0
        return 1.0

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of the A operand
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of the B operand
        :type b_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        SUPPORTED_DTYPES = {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }
        if a_dtype not in SUPPORTED_DTYPES or b_dtype not in SUPPORTED_DTYPES:
            is_valid = False
        if (
            BlockScaledContiguousGatherGroupedGemmKernel.needs_unpack_tma(
                a_dtype, b_dtype
            )
            and a_dtype.width < 8
        ):
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if {a_dtype, b_dtype} & {
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        } != set() and sf_vec_size == 16:
            is_valid = False

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
            cutlass.Float4E2M1FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
        swap_ab: bool = False,
    ) -> bool:
        """
        Check if layouts and dtypes are valid combinations

        :param a_dtype: The data type of the A operand
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of the B operand
        :type b_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param c_major: The major dimension of the C tensor
        :type c_major: str
        :param swap_ab: Whether A/B (and MN) roles are swapped
        :type swap_ab: bool

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        if a_dtype is cutlass.Float4E2M1FN and a_major != "k":
            is_valid = False
        if b_dtype is cutlass.Float4E2M1FN and b_major != "k":
            is_valid = False
        if (c_major == "m") != swap_ab:
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        swap_ab: bool = False,
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid

        :param mma_tiler_mn: User (token_tile, weight_tile). Under swap_ab,
            MMA M = weight_tile and MMA N = token_tile.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param swap_ab: Whether A/B (and MN) roles are swapped
        :type swap_ab: bool

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        mma_m = mma_tiler_mn[1] if swap_ab else mma_tiler_mn[0]
        mma_n = mma_tiler_mn[0] if swap_ab else mma_tiler_mn[1]

        # MMA M must be 128 or 256 (2CTA when 256)
        if mma_m not in (128, 256):
            is_valid = False
        # MMA N: swapped supports 8..256 powers of two; unswapped supports 128/192/256.
        if swap_ab:
            if mma_n not in (8, 16, 32, 64, 128, 256):
                is_valid = False
            # SM100 2CTA MMA requires MMA N >= 16.
            if mma_m == 256 and mma_n < 16:
                is_valid = False
        else:
            if mma_n not in (128, 192, 256):
                is_valid = False

        if cluster_shape_mn[0] <= 0 or cluster_shape_mn[1] <= 0:
            return False

        # Skip illegal cluster shape (CTA M along MMA M is always 128)
        if (mma_m // cluster_shape_mn[0]) != 128:
            is_valid = False

        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False

        # We only support cluster shape n = 1 for now
        # TODO: Support cluster shape n > 1
        if cluster_shape_mn[1] != 1:
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: cutlass.Int64
        :param n: The number of columns in the B tensor
        :type n: cutlass.Int64
        :param k: The number of columns in the A tensor
        :type k: cutlass.Int64
        :param l: The number of columns in the C tensor
        :type l: cutlass.Int64
        :param a_dtype: The data type of the A operand
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of the B operand
        :type b_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        def check_contigous_128_alignment(dtype, is_mode0_major, tensor_shape):
            if dtype.width >= 8:
                return True
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            return num_major_elements % 128 == 0

        if (
            not check_contigous_16B_alignment(a_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(b_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False

        # UNPACK_U8 requires the contiguous dimension of a sub-byte operand to
        # be a multiple of 128 elements.  In this gather kernel B is the only
        # unpacked operand, but keep the check symmetric for clarity.
        if BlockScaledContiguousGatherGroupedGemmKernel.needs_unpack_tma(
            a_dtype, b_dtype
        ) and (
            not check_contigous_128_alignment(a_dtype, a_major == "m", (m, k, l))
            or not check_contigous_128_alignment(b_dtype, b_major == "n", (n, k, l))
        ):
            is_valid = False
        return is_valid

    @classmethod
    def can_implement(
        cls,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        a_major: str,
        b_major: str,
        c_major: str,
        swap_ab: bool = False,
        gated: bool = True,
        split_k: int = 1,
        weight_interleave: Optional[int] = None,
        use_compact_sfc: bool = True,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param a_dtype: The data type of the A operand
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of the B operand
        :type b_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: User (token_tile, weight_tile)
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: cutlass.Int64
        :param n: The number of columns in the B tensor
        :type n: cutlass.Int64
        :param k: The number of columns in the A tensor
        :type k: cutlass.Int64
        :param l: The number of columns in the C tensor
        :type l: cutlass.Int64
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor (API layout)
        :type c_major: str
        :param swap_ab: Whether A/B (and MN) roles are swapped.
        :type swap_ab: bool
        :param gated: Whether the epilogue applies a gated activation.
        :type gated: bool
        :param split_k: Number of physical cluster-local K partitions.
        :type split_k: int
        :param weight_interleave: Physical up/gate interleave. Unswapped mode
            supports 16 or 64; swapped mode supports 16.
        :type weight_interleave: Optional[int]
        :param use_compact_sfc: Whether swapped output scales use compact
            row-major storage.
        :type use_compact_sfc: bool
        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        if c_major != "n":
            can_implement = False
        try:
            weight_interleave = normalize_cute_dsl_moe_weight_interleave(
                weight_interleave, swap_ab
            )
        except ValueError:
            return False
        if gated and n % (2 * weight_interleave) != 0:
            can_implement = False
        # Skip unsupported types
        if not cls.is_valid_dtypes_and_scale_factor_vec_size(
            a_dtype, b_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        if swap_ab:
            mma_c_major = "m"
        else:
            mma_c_major = c_major
        if not cls.is_valid_layouts(
            a_dtype,
            b_dtype,
            c_dtype,
            a_major,
            b_major,
            mma_c_major,
            swap_ab,
        ):
            can_implement = False

        # Skip invalid mma tile shape and cluster shape
        if not cls.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn, swap_ab=swap_ab
        ):
            can_implement = False

        mma_n = mma_tiler_mn[0] if swap_ab else mma_tiler_mn[1]
        if not swap_ab and mma_n == 192 and (not gated or weight_interleave != 16):
            can_implement = False
        if (
            swap_ab
            and c_dtype is cutlass.Float4E2M1FN
            and mma_tiler_mn == (16, 256)
            and (a_dtype.width == 8 or b_dtype.width == 8)
        ):
            can_implement = False
        logical_output_n = n // 2 if gated else n
        if c_dtype == cutlass.Float4E2M1FN and not (swap_ab and use_compact_sfc):
            scale_m = logical_output_n if swap_ab else m
            scale_k = m if swap_ab else logical_output_n
            if scale_m % 128 != 0 or scale_k % (sf_vec_size * 4) != 0:
                can_implement = False
        if split_k < 1 or not is_power_of_2(split_k):
            can_implement = False
        if cluster_shape_mn[0] * cluster_shape_mn[1] * split_k > 16:
            can_implement = False
        if split_k > 1 and (
            not swap_ab or (split_k > 4 and mma_n > 32) or split_k > 16
        ):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not cls.is_valid_tensor_alignment(
            m,
            n,
            k,
            l,
            a_dtype,
            b_dtype,
            c_dtype,
            a_major,
            b_major,
            c_major,
        ):
            can_implement = False
        # Skip unsupported A/B layout
        if not (a_major == "k" and b_major == "k"):
            can_implement = False
        return can_implement

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        c_sf_ptr: cute.Pointer,
        alpha_ptr: Optional[cute.Pointer],
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        token_id_mapping_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        global_sf_ptr: cute.Pointer,
        bias_up_ptr: Optional[cute.Pointer],
        bias_gate_ptr: Optional[cute.Pointer],
        a_per_token_scale_ptr: Optional[cute.Pointer],
        orig_m: cutlass.Int64,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: driver.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        scale_k = k // scaling_vector_size
        interm_size = n // self.out_n_factor
        num_tiles = m // tile_size
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((orig_m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2))
        )
        a_sf = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout((orig_m, scale_k, 1), order=(1, 0, 2)),
        )
        b_sf = cute.make_tensor(
            b_sf_ptr,
            layout=blockscaled_layout.tile_atom_to_shape_SF(
                (n, k, l), scaling_vector_size
            ),
        )
        c = cute.make_tensor(
            c_ptr, layout=cute.make_ordered_layout((m, interm_size, 1), order=(1, 0, 2))
        )
        if cutlass.const_expr(c_sf_ptr is not None):
            c_sf_layout = (
                compact_sf_layout((m, interm_size, l), scaling_vector_size)
                if self.use_compact_sfc
                else blockscaled_layout.tile_atom_to_shape_SF(
                    (m, interm_size, l), scaling_vector_size
                )
            )
            c_sf = cute.make_tensor(
                c_sf_ptr,
                layout=c_sf_layout,
            )
        else:
            c_sf = None
        if cutlass.const_expr(self.use_alpha):
            assert alpha_ptr is not None
            alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((l,)))
        else:
            assert alpha_ptr is None
            alpha = None

        tile_idx_to_group_idx = cute.make_tensor(
            tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        token_id_mapping = cute.make_tensor(
            token_id_mapping_ptr, layout=cute.make_layout((m,))
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )
        if cutlass.const_expr(global_sf_ptr is not None):
            global_sf = cute.make_tensor(global_sf_ptr, layout=cute.make_layout((1,)))
        else:
            global_sf = None
        if cutlass.const_expr(self.use_a_per_token_scale):
            a_per_token_scale = cute.make_tensor(
                a_per_token_scale_ptr, layout=cute.make_layout((orig_m,))
            )
        else:
            a_per_token_scale = None

        if cutlass.const_expr(self.use_bias):
            assert bias_up_ptr is not None
            assert bias_gate_ptr is not None
            bias_expert_stride = interm_size * self.bias_expert_stride_factor
            bias_up = cute.make_tensor(
                bias_up_ptr,
                layout=cute.make_layout(
                    (m, interm_size, l), stride=(0, 1, bias_expert_stride)
                ),
            )
            bias_gate = cute.make_tensor(
                bias_gate_ptr,
                layout=cute.make_layout(
                    (m, interm_size, l), stride=(0, 1, bias_expert_stride)
                ),
            )
        else:
            assert bias_up_ptr is None
            assert bias_gate_ptr is None
            bias_up = None
            bias_gate = None
        return self(
            a,
            b,
            c,
            a_sf,
            b_sf,
            c_sf,
            global_sf,
            bias_up,
            bias_gate,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            alpha,
            a_per_token_scale,
            max_active_clusters=max_active_clusters,
            stream=stream,
            epilogue_op=epilogue_op,
        )


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


def interleave_up_gate(x, group_size):
    """Convert conceptual [all up, all gate] rows to kernel physical ordering."""
    num_experts, n, k = x.shape
    return (
        x.view(num_experts, 2, n // (2 * group_size), group_size, k)
        .transpose(1, 2)
        .contiguous()
        .view_as(x)
    )


def deinterleave_up_gate(x, group_size):
    """Undo interleave_up_gate after quantize/dequantize."""
    num_experts, n, k = x.shape
    return (
        x.view(num_experts, n // (2 * group_size), 2, group_size, k)
        .transpose(1, 2)
        .contiguous()
        .view_as(x)
    )


def quantize_operand(source, dtype, sf_dtype, sf_vec_size, swizzled):
    """Encode one 2D operand with unit block scales.

    :return: (quantized tensor, scale factors, effective float32 values)
    """
    if dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
        q = source.to(cutlass.torch.dtype(dtype))
    elif dtype is cutlass.Float4E2M1FN:
        magnitude = source.abs()
        encoded = torch.zeros_like(source, dtype=torch.uint8)
        for midpoint in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
            encoded.add_(magnitude > midpoint)
        encoded |= (source < 0).to(torch.uint8) << 3
        q = (encoded[:, 0::2] | (encoded[:, 1::2] << 4)).to(torch.uint8)
    else:
        raise ValueError(
            f"unsupported quantization: {dtype=} {sf_dtype=} {sf_vec_size=}"
        )

    rows, k = source.shape
    sf_k = k // sf_vec_size
    if sf_dtype is cutlass.Float8E8M0FNU:
        unit = 127
    elif sf_dtype is cutlass.Float8E4M3FN:
        unit = 0x38
    else:
        raise ValueError(f"unsupported scale factor type: {sf_dtype}")
    sf_linear = torch.full((rows, sf_k), unit, dtype=torch.uint8, device=source.device)
    if swizzled:
        padded_rows = ((rows + 127) // 128) * 128
        padded_sf_k = ((sf_k + 3) // 4) * 4
        sf = torch.full(
            (padded_rows, padded_sf_k),
            unit,
            dtype=torch.uint8,
            device=source.device,
        )
    else:
        sf = sf_linear
    effective = dequantize_blockscaled_values(
        q, sf_linear, dtype, sf_dtype, sf_vec_size
    )
    return q, sf, effective.float()


def quantize_scaled_output(source, dtype, sf_dtype, sf_vec_size, global_scale):
    """Quantize an output using the kernel's generated block scales."""
    rows, k = source.shape
    blocks = source.float().view(rows, k // sf_vec_size, sf_vec_size)
    limit = (
        6.0
        if dtype is cutlass.Float4E2M1FN
        else torch.finfo(cutlass.torch.dtype(dtype)).max
    )
    ratio = blocks.abs().amax(dim=-1) * (global_scale / limit)
    if sf_dtype is cutlass.Float8E8M0FNU:
        # E8M0 rounds to the nearest power of two; rounding up avoids
        # overflowing the output type (NaN for E4M3), matching the kernel.
        ratio = torch.where(ratio > 0, torch.exp2(torch.ceil(torch.log2(ratio))), ratio)
    scale = ratio.to(cutlass.torch.dtype(sf_dtype))
    scale_f32 = scale.float()
    multiplier = torch.where(
        scale_f32 == 0,
        torch.zeros_like(scale_f32),
        global_scale / scale_f32,
    )
    scaled = (blocks * multiplier.unsqueeze(-1)).view(rows, k)
    q, _, _ = quantize_operand(scaled, dtype, sf_dtype, sf_vec_size, False)
    effective = dequantize_blockscaled_values(
        q, scale.view(torch.uint8), dtype, sf_dtype, sf_vec_size
    )
    return q, scale.view(torch.uint8), effective / global_scale


def convert_sf_to_mma_layout(
    sf: torch.Tensor,
    m: int,
    k: int,
    num_groups: int = 1,
    sf_vec_size: int = 16,
) -> torch.Tensor:
    """View swizzled scale storage in the MMA scale-factor layout."""
    sf_k = (k + sf_vec_size - 1) // sf_vec_size
    m_tiles = (m + 127) // 128
    k_tiles = (sf_k + 3) // 4
    expected = num_groups * m_tiles * k_tiles * 32 * 4 * 4
    if sf.numel() != expected:
        raise ValueError(f"scale tensor has {sf.numel()} elements; expected {expected}")
    return sf.view(num_groups, m_tiles, k_tiles, 32, 4, 4).permute(3, 4, 1, 5, 2, 0)


def dequantize_blockscaled_values(encoded, linear_sf, dtype, sf_dtype, sf_vec_size):
    """Decode quantized values with already-linear block scales to float32."""
    import torch

    if sf_dtype is cutlass.Float8E8M0FNU:
        exponent = linear_sf.view(torch.uint8).to(torch.int32) - 127
        scales = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    else:
        scales = linear_sf.view(torch.float8_e4m3fn).float()
    scales = scales.repeat_interleave(sf_vec_size, dim=-1)
    if dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
        values = encoded.float()
    else:
        lut = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            device=encoded.device,
        )
        packed = encoded.view(torch.uint8).long()
        values = torch.stack((lut[packed & 0xF], lut[packed >> 4]), dim=-1).reshape(
            encoded.shape[0], -1
        )
    return values * scales


def cvt_sf_M32x4xrm_K4xrk_L_to_MKL_torch(sf, m, k, sf_vec_size):
    """Map MMA SF layout M(32x4xrest_m)xK(4xrest_k)xL to ordinary row-major scales."""
    sf_k = k // sf_vec_size
    padded_sf_k = ((sf_k + 3) // 4) * 4
    rows = torch.arange(m, device=sf.device)[:, None]
    cols = torch.arange(sf_k, device=sf.device)[None, :]
    offsets = (
        cols % 4
        + (cols // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + (rows // 128) * (128 * padded_sf_k)
    )
    return sf.reshape(-1)[offsets]


def create_routing_tensors(num_tokens, topk, num_experts, tile_m, device):
    """Create deterministic expert-grouped routing metadata for the gather kernel.

    :return: token_id_mapping, row_experts, tile_idx_to_expert_idx,
             tile_idx_to_mn_limit, num_non_exiting_tiles
    """
    import torch

    assignments = [[] for _ in range(num_experts)]
    for token in range(num_tokens):
        for rank in range(topk):
            assignments[(token * topk + rank) % num_experts].append(token * topk + rank)

    aligned = [((len(x) + tile_m - 1) // tile_m) * tile_m for x in assignments]
    valid_m = sum(aligned)
    token_ids = torch.full((valid_m,), -1, dtype=torch.int32, device=device)
    row_experts = torch.full((valid_m,), -1, dtype=torch.int64, device=device)
    tile_experts, tile_limits = [], []
    offset = 0
    for expert, expanded_ids in enumerate(assignments):
        count = len(expanded_ids)
        if count:
            token_ids[offset : offset + count] = torch.tensor(
                expanded_ids, dtype=torch.int32, device=device
            )
            row_experts[offset : offset + count] = expert
        for tile in range(aligned[expert] // tile_m):
            tile_experts.append(expert)
            tile_limits.append(offset + min((tile + 1) * tile_m, count))
        offset += aligned[expert]

    return (
        token_ids,
        row_experts,
        torch.tensor(tile_experts, dtype=torch.int32, device=device),
        torch.tensor(tile_limits, dtype=torch.int32, device=device),
        torch.tensor([len(tile_experts)], dtype=torch.int32, device=device),
    )


def create_tensors_ab(
    num_tokens,
    num_experts,
    n,
    k,
    a_dtype,
    b_dtype,
    sf_dtype,
    sf_vec_size,
    gated,
    group_size,
    init_normal,
    normal_mean,
    normal_std,
    device,
    generator,
):
    """Create quantized A/B operands, scale factors, and float32 reference tensors.

    Analogous to create_tensors_abc_for_all_groups / create_tensors_sfasfb_for_all_groups
    in the grouped blockscaled GEMM example, adapted for contiguous gather MoE.
    """
    if init_normal:
        a_source = torch.normal(
            normal_mean,
            normal_std,
            size=(num_tokens, k),
            generator=generator,
            device=device,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        b_conceptual = torch.normal(
            normal_mean,
            normal_std,
            size=(num_experts, n, k),
            generator=generator,
            device=device,
            dtype=torch.float32,
        ).to(torch.bfloat16)
    else:
        a_source = torch.randint(
            -2,
            3,
            (num_tokens, k),
            generator=generator,
            device=device,
            dtype=torch.int32,
        ).to(torch.bfloat16)
        b_conceptual = torch.randint(
            -2,
            3,
            (num_experts, n, k),
            generator=generator,
            device=device,
            dtype=torch.int32,
        ).to(torch.bfloat16)
    b_source = interleave_up_gate(b_conceptual, group_size) if gated else b_conceptual

    a, a_scale, a_ref = quantize_operand(
        a_source, a_dtype, sf_dtype, sf_vec_size, False
    )
    b_flat, b_scale_swizzled, b_ref_flat = quantize_operand(
        b_source.view(num_experts * n, k),
        b_dtype,
        sf_dtype,
        sf_vec_size,
        True,
    )
    b = b_flat.view(num_experts, n, -1)
    b_ref = b_ref_flat.view(num_experts, n, k)
    if gated:
        b_ref = deinterleave_up_gate(b_ref, group_size)
    b_scale = convert_sf_to_mma_layout(
        b_scale_swizzled,
        m=n,
        k=k,
        num_groups=num_experts,
        sf_vec_size=sf_vec_size,
    )
    return a, b, a_scale, b_scale, a_ref, b_ref


def create_output_tensors(
    valid_m,
    intermediate_dim,
    c_dtype,
    sf_vec_size,
    swap_ab,
    use_compact_sfc,
    generate_sfc,
    output_global_scale,
    device,
):
    """Create output C and optional SFC tensors for the gather kernel."""
    if c_dtype is cutlass.Float4E2M1FN:
        out = torch.empty(
            (valid_m, intermediate_dim // 2), dtype=torch.uint8, device=device
        )
    else:
        out = torch.empty(
            (valid_m, intermediate_dim),
            dtype=cutlass.torch.dtype(c_dtype),
            device=device,
        )
    if generate_sfc:
        scale_k = intermediate_dim // sf_vec_size
        if swap_ab and use_compact_sfc:
            out_scale = torch.empty(
                (valid_m, ((scale_k + 15) // 16) * 16),
                dtype=torch.uint8,
                device=device,
            )
        else:
            out_scale = torch.empty(
                (
                    32,
                    4,
                    (valid_m + 127) // 128,
                    4,
                    intermediate_dim // (sf_vec_size * 4),
                    1,
                ),
                dtype=torch.uint8,
                device=device,
            )
        global_scale = torch.tensor(
            [output_global_scale], dtype=torch.float32, device=device
        )
    else:
        out_scale = global_scale = None
    return out, out_scale, global_scale


def compute_reference(
    a_ref,
    b_ref,
    token_id_mapping,
    row_experts,
    topk,
    intermediate_dim,
    gated,
    alpha,
    bias_up,
    bias_gate,
    swiglu_alpha,
    swiglu_beta,
    swiglu_limit,
):
    """Compute float32 reference for valid permuted rows (gather GEMM + activation)."""
    import torch

    ref = torch.zeros(
        (token_id_mapping.numel(), intermediate_dim),
        dtype=torch.float32,
        device=a_ref.device,
    )
    valid = token_id_mapping >= 0
    for expert in range(b_ref.shape[0]):
        rows = valid & (row_experts == expert)
        if not rows.any():
            continue
        tokens = token_id_mapping[rows].long() // topk
        acc = a_ref[tokens] @ b_ref[expert].T
        if alpha is not None:
            acc *= alpha[expert]
        if gated:
            up, gate = acc.split(intermediate_dim, dim=-1)
            up = up + bias_up[expert]
            gate = gate + bias_gate[expert]
            if swiglu_limit is not None:
                up = up.clamp(-swiglu_limit, swiglu_limit)
                gate = torch.minimum(
                    gate, torch.tensor(swiglu_limit, device=gate.device)
                )
            ref[rows] = (up + swiglu_beta) * gate * torch.sigmoid(swiglu_alpha * gate)
        else:
            ref[rows] = torch.relu(acc).square()
    return ref, valid


def decode_scaled_output(
    out,
    out_scale,
    c_dtype,
    sf_dtype,
    sf_vec_size,
    valid_m,
    intermediate_dim,
    output_global_scale,
    blockscaled_sfc,
):
    """Decode kernel-generated FP8/FP4 output to logical row-major float32."""
    scale_k = intermediate_dim // sf_vec_size
    if blockscaled_sfc:
        sf_storage = out_scale.view(
            32,
            4,
            (valid_m + 127) // 128,
            4,
            scale_k // 4,
            1,
        )
        sf = cvt_sf_M32x4xrm_K4xrk_L_to_MKL_torch(
            sf_storage, valid_m, intermediate_dim, sf_vec_size
        )
    else:
        sf = out_scale[:valid_m, :scale_k]

    decoded = dequantize_blockscaled_values(out, sf, c_dtype, sf_dtype, sf_vec_size)
    return decoded / output_global_scale


@cute.jit
def cvt_sf_M32x4xrm_K4xrk_L_to_MKL(
    sf_swizzled_tensor: cute.Tensor,
    sf_unswizzled_tensor: cute.Tensor,
):
    """Convert scale factor tensor from mma specification M(32x4xrest_m)xK(4xrest_k)xL layout to MKL layout"""
    # sf_swizzled_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_swizzled_tensor = cute.group_modes(sf_swizzled_tensor, 0, 3)
    sf_swizzled_tensor = cute.group_modes(sf_swizzled_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_unswizzled_tensor)):
        mkl_coord = sf_unswizzled_tensor.layout.get_hier_coord(i)
        sf_unswizzled_tensor[mkl_coord] = sf_swizzled_tensor[mkl_coord]


def run(
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    topk: int,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    init_normal: bool = False,
    normal_mean: float = 0.0,
    normal_std: float = 1.0,
    swap_ab: bool = False,
    gated: bool = True,
    vectorized_f32: bool = True,
    raster_along_m: bool = False,
    pdl_count: Optional[int] = -1,
    use_alpha: bool = True,
    expert_alpha_value: float = 1.0,
    use_bias: bool = False,
    bias_value: Optional[float] = None,
    combined_bias: bool = False,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
    swiglu_limit: Optional[float] = None,
    seed: int = 2025,
    generate_scaled_output: bool = False,
    output_global_scale: float = 1.0,
    split_k: int = 1,
    activation_type: Optional[int] = None,
    weight_interleave: Optional[int] = None,
    use_compact_sfc: bool = True,
):
    """Run SM100 contiguous gather grouped blockscaled GEMM example with specified configurations.

    :param use_cold_l2: Whether to use circular buffer strategy to ensure cold L2 cache, defaults to False
    :type use_cold_l2: bool, optional
    :param init_normal: Whether to initialize tensors using normal distribution
        instead of uniform random, defaults to False.
    :type init_normal: bool, optional
    :param normal_mean: Mean for normal distribution initialization, defaults to 0.0.
    :type normal_mean: float, optional
    :param normal_std: Standard deviation for normal distribution initialization,
        defaults to 1.0.
    :type normal_std: float, optional
    :param swap_ab: Whether to swap MMA-A/MMA-B assignments and M/N roles,
        defaults to False.
    :type swap_ab: bool, optional
    :param gated: Whether to use SwiGLU instead of ReLU2, defaults to True.
    :type gated: bool, optional
    :param vectorized_f32: Whether to use packed float32 epilogue math, defaults to True.
    :type vectorized_f32: bool, optional
    :param raster_along_m: Whether to rasterize along M, defaults to False.
    :type raster_along_m: bool, optional
    :param pdl_count: Persistent K-tile index at which to launch dependent
        grids. None disables PDL; -1 releases grids at kernel completion.
    :type pdl_count: Optional[int]
    :param use_alpha: Whether to apply per-expert alpha, defaults to True.
    :type use_alpha: bool, optional
    :param use_bias: Whether to apply branch bias, defaults to False.
    :type use_bias: bool, optional
    :param generate_scaled_output: Whether to generate output block scales, defaults to False.
    :type generate_scaled_output: bool, optional
    :param use_compact_sfc: Whether swapped output scales use compact row-major
        storage, defaults to True.
    :type use_compact_sfc: bool, optional
    :param split_k: Number of cluster-local K partitions, defaults to 1.
    :type split_k: int, optional
    :param activation_type: FC1 activation type. When omitted, derives SwiGLU
        or ReLU^2 from ``gated``. Reference checking supports SwiGLU and ReLU^2.
    :type activation_type: Optional[int]
    :param weight_interleave: Physical up/gate interleave. Unswapped mode
        supports 16 or 64; swapped mode supports 16.
    :type weight_interleave: Optional[int]
    :return: Execution time of the GEMM kernel in microseconds
    :rtype: float
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    if activation_type is None:
        activation_type = ActivationType.Swiglu if gated else ActivationType.Relu2
    activation_type, expected_gated = normalize_cute_dsl_moe_activation_type(
        activation_type
    )
    if gated != expected_gated:
        raise ValueError(
            f"gated={gated} is inconsistent with activation_type {activation_type!r}"
        )
    if not skip_ref_check and activation_type not in (
        ActivationType.Swiglu,
        ActivationType.Relu2,
    ):
        raise ValueError(
            f"Reference checking does not support activation_type {activation_type!r}; "
            "set skip_ref_check=True"
        )

    positive = {
        "num_tokens": num_tokens,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "num_experts": num_experts,
        "topk": topk,
        "tolerance": tolerance,
        "output_global_scale": output_global_scale,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError(f"expected positive values, got {positive}")
    if normal_std < 0:
        raise ValueError("normal_std must be non-negative")
    if topk > num_experts:
        raise ValueError(f"topk={topk} must not exceed num_experts={num_experts}")
    if len(mma_tiler_mn) != 2 or len(cluster_shape_mn) != 2:
        raise ValueError(
            "mma_tiler_mn and cluster_shape_mn must each contain exactly 2 values"
        )
    if warmup_iterations < 0 or iterations <= 0:
        raise ValueError("warmup_iterations must be >= 0 and iterations must be > 0")
    if pdl_count is not None and pdl_count < -1:
        raise ValueError("pdl_count must be -1 or a non-negative K-tile index")
    if sf_vec_size == 16 and sf_dtype is not cutlass.Float8E4M3FN:
        raise ValueError("sf_vec_size=16 requires E4M3 scale factors")
    if sf_vec_size == 32 and sf_dtype is not cutlass.Float8E8M0FNU:
        raise ValueError("sf_vec_size=32 requires UE8M0 scale factors")
    if use_bias and not gated:
        raise ValueError("branch bias is only supported by the gated SwiGLU path")
    if swiglu_limit is not None and swiglu_limit <= 0:
        raise ValueError("swiglu_limit must be positive")
    if not gated and (
        swiglu_alpha != 1.0 or swiglu_beta != 0.0 or swiglu_limit is not None
    ):
        raise ValueError("SwiGLU controls require gated=True")
    if generate_scaled_output and c_dtype not in {
        cutlass.Float8E4M3FN,
        cutlass.Float4E2M1FN,
    }:
        raise ValueError("--generate_scaled_output requires E4M3 FP8 or FP4 output")
    if (
        generate_scaled_output
        and (sf_dtype is not cutlass.Float8E8M0FNU or sf_vec_size != 32)
        and c_dtype is not cutlass.Float4E2M1FN
    ):
        raise ValueError(
            "scaled FP8 output requires UE8M0 scale factors and sf_vec_size=32"
        )
    if (
        output_global_scale != 1.0
        and c_dtype is not cutlass.Float4E2M1FN
        and not generate_scaled_output
    ):
        raise ValueError(
            "output_global_scale requires FP4 output or --generate_scaled_output"
        )

    device = torch.device("cuda")
    n = intermediate_dim * (2 if gated else 1)
    k = hidden_dim
    weight_interleave = normalize_cute_dsl_moe_weight_interleave(
        weight_interleave, swap_ab
    )
    if k % sf_vec_size or (k // sf_vec_size) % 4:
        raise ValueError(
            f"hidden_dim={k} must provide a multiple of four {sf_vec_size}-value "
            "scale blocks"
        )
    tile_m = mma_tiler_mn[0]
    group_size = weight_interleave
    if gated and intermediate_dim % group_size:
        raise ValueError(
            f"gated intermediate_dim={intermediate_dim} must be divisible by "
            f"the {group_size}-row physical interleave group"
        )

    print("Running Blackwell Contiguous Gather Grouped GEMM test with:")
    print(f"Tokens: {num_tokens}, Experts: {num_experts}, TopK: {topk}")
    print(f"Hidden (K): {hidden_dim}, Intermediate: {intermediate_dim}, Gated: {gated}")
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, SF dtype: {sf_dtype}, SF Vec size: {sf_vec_size}"
    )
    print(f"C dtype: {c_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")
    print(f"Swap AB: {swap_ab}")
    print(f"Vectorized F32: {vectorized_f32}")
    print(f"Raster along M: {raster_along_m}")
    print(f"PDL count: {pdl_count} (-1 = release at kernel completion)")
    print(f"Split K: {split_k}")
    print(f"Weight interleave: {weight_interleave}")
    print(
        f"Expert alpha: {use_alpha}" + (f" ({expert_alpha_value})" if use_alpha else "")
    )
    print(
        f"Bias: {use_bias}"
        + (f" ({bias_value})" if use_bias and bias_value is not None else "")
    )
    print(f"SwiGLU alpha: {swiglu_alpha}, beta: {swiglu_beta}, limit: {swiglu_limit}")
    print(f"Generate scaled output: {generate_scaled_output}")
    print(f"Compact SFC: {swap_ab and use_compact_sfc}")
    print(f"Output global scale: {output_global_scale}")
    print(f"Seed: {seed}")
    print(f"Normal initialization: {init_normal}")

    initial_routing_tensors = (
        token_id_mapping,
        row_experts,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        num_non_exiting,
    ) = create_routing_tensors(num_tokens, topk, num_experts, tile_m, device)
    valid_m = token_id_mapping.numel()
    print(f"Permuted M: {valid_m}")

    if not BlockScaledContiguousGatherGroupedGemmKernel.can_implement(
        a_dtype,
        b_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        valid_m,
        n,
        k,
        num_experts,
        a_major,
        b_major,
        c_major,
        swap_ab=swap_ab,
        gated=gated,
        split_k=split_k,
        weight_interleave=weight_interleave,
        use_compact_sfc=use_compact_sfc,
    ):
        raise cutlass.testing.CantImplementError(
            f"Unsupported testcase {a_dtype}, {b_dtype}, {sf_dtype}, "
            f"{sf_vec_size}, {c_dtype}, {mma_tiler_mn}, {cluster_shape_mn}, "
            f"shape=({valid_m},{n},{k},{num_experts}), {a_major}, {b_major}, "
            f"{c_major}, swap_ab={swap_ab}"
        )

    torch.manual_seed(seed)
    generate_sfc = c_dtype is cutlass.Float4E2M1FN or generate_scaled_output
    compact_sfc = swap_ab and use_compact_sfc
    if generate_sfc and intermediate_dim % sf_vec_size:
        raise ValueError(
            f"intermediate_dim={intermediate_dim} must be divisible by "
            f"sf_vec_size={sf_vec_size}"
        )
    scale_m = intermediate_dim if swap_ab else valid_m
    scale_k = valid_m if swap_ab else intermediate_dim
    if (
        generate_sfc
        and not compact_sfc
        and (scale_m % 128 or scale_k % (sf_vec_size * 4))
    ):
        raise ValueError(
            "generated output scales require scale-m divisible by 128 and "
            f"scale-k divisible by {sf_vec_size * 4}; got ({scale_m}, {scale_k})"
        )

    gmem = cute.AddressSpace.gmem
    current_stream = cutlass.torch.default_stream()

    def create_workspace(routing_tensors=None):
        if routing_tensors is None:
            routing_tensors = create_routing_tensors(
                num_tokens, topk, num_experts, tile_m, device
            )
        (
            token_id_mapping,
            row_experts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            num_non_exiting,
        ) = routing_tensors

        generator = torch.Generator(device=device).manual_seed(seed)
        a, b, a_scale, b_scale, a_ref, b_ref = create_tensors_ab(
            num_tokens,
            num_experts,
            n,
            k,
            a_dtype,
            b_dtype,
            sf_dtype,
            sf_vec_size,
            gated,
            group_size,
            init_normal,
            normal_mean,
            normal_std,
            device,
            generator,
        )

        alpha = (
            torch.full(
                (num_experts,),
                expert_alpha_value,
                dtype=torch.float32,
                device=device,
            )
            if use_alpha
            else None
        )
        if use_bias and bias_value is None:
            bias_up = (
                torch.randn(
                    (num_experts, intermediate_dim),
                    generator=generator,
                    device=device,
                ).float()
                * 0.01
            )
            bias_gate = (
                torch.randn(
                    (num_experts, intermediate_dim),
                    generator=generator,
                    device=device,
                ).float()
                * 0.01
            )
        else:
            fill = bias_value if use_bias else 0.0
            bias_up = torch.full(
                (num_experts, intermediate_dim),
                fill,
                dtype=torch.float32,
                device=device,
            )
            bias_gate = torch.full_like(bias_up, fill)
        if combined_bias:
            bias_up, bias_gate = torch.cat((bias_up, bias_gate), dim=1).chunk(2, dim=1)

        out, out_scale, global_scale = create_output_tensors(
            valid_m,
            intermediate_dim,
            c_dtype,
            sf_vec_size,
            swap_ab,
            use_compact_sfc,
            generate_sfc,
            output_global_scale,
            device,
        )

        a_ptr = make_ptr(a_dtype, a.data_ptr(), gmem, assumed_align=32)
        b_ptr = make_ptr(b_dtype, b.data_ptr(), gmem, assumed_align=32)
        a_sf_ptr = make_ptr(sf_dtype, a_scale.data_ptr(), gmem, assumed_align=16)
        b_sf_ptr = make_ptr(sf_dtype, b_scale.data_ptr(), gmem, assumed_align=16)
        c_ptr = make_ptr(c_dtype, out.data_ptr(), gmem, assumed_align=32)
        c_sf_ptr = (
            make_ptr(sf_dtype, out_scale.data_ptr(), gmem, assumed_align=16)
            if out_scale is not None
            else None
        )
        alpha_ptr = (
            make_ptr(cutlass.Float32, alpha.data_ptr(), gmem)
            if alpha is not None
            else None
        )
        norm_const_ptr = (
            make_ptr(cutlass.Float32, global_scale.data_ptr(), gmem)
            if global_scale is not None
            else None
        )
        tile_idx_ptr = make_ptr(cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), gmem)
        mn_limit_ptr = make_ptr(cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), gmem)
        token_id_ptr = make_ptr(cutlass.Int32, token_id_mapping.data_ptr(), gmem)
        num_tiles_ptr = make_ptr(cutlass.Int32, num_non_exiting.data_ptr(), gmem)
        bias_up_ptr = (
            make_ptr(cutlass.Float32, bias_up.data_ptr(), gmem) if use_bias else None
        )
        bias_gate_ptr = (
            make_ptr(cutlass.Float32, bias_gate.data_ptr(), gmem) if use_bias else None
        )

        jit_args = cutlass.testing.JitArguments(
            a_ptr,
            b_ptr,
            a_sf_ptr,
            b_sf_ptr,
            c_ptr,
            c_sf_ptr,
            alpha_ptr,
            tile_idx_ptr,
            mn_limit_ptr,
            token_id_ptr,
            num_tiles_ptr,
            norm_const_ptr,
            bias_up_ptr,
            bias_gate_ptr,
            None,  # a_per_token_scale_ptr
            num_tokens,
            valid_m,
            n,
            k,
            num_experts,
            current_stream,
        )
        kernel_tensors = [
            a,
            b,
            a_scale,
            b_scale,
            out,
            alpha,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting,
            global_scale,
            bias_up,
            bias_gate,
            out_scale,
        ]
        kernel_tensors = [tensor for tensor in kernel_tensors if tensor is not None]
        jit_args.add_to_scope(kernel_tensors)
        return jit_args, {
            "a_ref": a_ref,
            "b_ref": b_ref,
            "out": out,
            "out_scale": out_scale,
            "global_scale": global_scale,
            "token_id_mapping": token_id_mapping,
            "row_experts": row_experts,
            "alpha": alpha,
            "bias_up": bias_up,
            "bias_gate": bias_gate,
            "kernel_tensors": kernel_tensors,
        }

    initial_args, initial_tensors = create_workspace(initial_routing_tensors)

    gemm = BlockScaledContiguousGatherGroupedGemmKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        vectorized_f32=vectorized_f32,
        topk=topk,
        raster_along_m=raster_along_m,
        pdl_count=pdl_count,
        split_k=split_k,
        weight_interleave=weight_interleave,
        gated=gated,
        swap_ab=swap_ab,
        use_alpha=use_alpha,
        use_bias=use_bias,
        activation_type=activation_type,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        use_compact_sfc=use_compact_sfc,
        bias_expert_stride_factor=2 if combined_bias else 1,
    )
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        gemm.cluster_shape_mn[0] * gemm.cluster_shape_mn[1] * gemm.split_k
    )

    compiled = cute.compile(
        gemm.wrapper,
        *initial_args.args[:-1],
        tile_size=tile_m,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_active_clusters,
        stream=current_stream,
    )

    if not skip_ref_check:
        compiled(*initial_args.args, **initial_args.kwargs)
        torch.cuda.synchronize()
        print("Verifying results...")
        reference, valid_rows = compute_reference(
            initial_tensors["a_ref"],
            initial_tensors["b_ref"],
            initial_tensors["token_id_mapping"],
            initial_tensors["row_experts"],
            topk,
            intermediate_dim,
            gated,
            initial_tensors["alpha"],
            initial_tensors["bias_up"],
            initial_tensors["bias_gate"],
            swiglu_alpha,
            swiglu_beta,
            swiglu_limit,
        )
        if generate_sfc:
            actual = decode_scaled_output(
                initial_tensors["out"],
                initial_tensors["out_scale"],
                c_dtype,
                sf_dtype,
                sf_vec_size,
                valid_m,
                intermediate_dim,
                float(initial_tensors["global_scale"].item()),
                not compact_sfc,
            )
            if output_global_scale == 1.0:
                _, expected_scale, expected = quantize_scaled_output(
                    reference.to(torch.bfloat16),
                    c_dtype,
                    sf_dtype,
                    sf_vec_size,
                    output_global_scale,
                )
            else:
                expected = reference
        else:
            actual = initial_tensors["out"].float()
            expected = reference.to(initial_tensors["out"].dtype).float()
        actual_valid = actual[valid_rows]
        expected_valid = expected[valid_rows]
        if (
            c_dtype is cutlass.Float4E2M1FN
            and generate_sfc
            and output_global_scale == 1.0
        ):
            # Allow one scale ULP and two normalized FP4 quantization steps.
            sfc_k = intermediate_dim // sf_vec_size
            if swap_ab and use_compact_sfc:
                actual_sf_codes = initial_tensors["out_scale"][valid_rows, :sfc_k]
            else:
                actual_sf_codes = cvt_sf_M32x4xrm_K4xrk_L_to_MKL_torch(
                    initial_tensors["out_scale"],
                    valid_m,
                    intermediate_dim,
                    sf_vec_size,
                )[valid_rows, :sfc_k]
            expected_sf_codes = expected_scale.view(valid_m, -1)[valid_rows, :sfc_k]
            torch.testing.assert_close(
                actual_sf_codes.to(torch.int16),
                expected_sf_codes.to(torch.int16),
                atol=1,
                rtol=0,
            )
            if sf_dtype is cutlass.Float8E8M0FNU:
                actual_sf_values = torch.ldexp(
                    torch.ones_like(actual_sf_codes, dtype=torch.float32),
                    actual_sf_codes.to(torch.int32) - 127,
                )
                expected_sf_values = torch.ldexp(
                    torch.ones_like(expected_sf_codes, dtype=torch.float32),
                    expected_sf_codes.to(torch.int32) - 127,
                )
            else:
                actual_sf_values = actual_sf_codes.view(torch.float8_e4m3fn).float()
                expected_sf_values = expected_sf_codes.view(torch.float8_e4m3fn).float()
            value_scale = torch.maximum(
                actual_sf_values, expected_sf_values
            ).repeat_interleave(sf_vec_size, dim=-1)
            value_scale = value_scale.clamp_min(torch.finfo(torch.float32).tiny)
            torch.testing.assert_close(
                actual_valid / value_scale,
                expected_valid / value_scale,
                atol=2.0,
                rtol=1e-02,
            )
        else:
            torch.testing.assert_close(
                actual_valid,
                expected_valid,
                atol=tolerance,
                rtol=1e-02,
            )

    def generate_tensors():
        jit_args, _ = create_workspace()
        return jit_args

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in initial_tensors["kernel_tensors"]
        )
        workspace_count = cutlass.testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = cutlass.testing.benchmark(
        compiled,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    runtime_s = exec_time / 1.0e6
    flop = 2 * valid_m * n * k
    gflops = (flop / 1.0e9) / runtime_s
    print("Average Runtime : ", exec_time / 1000, "ms")
    print("GFLOPS          : ", gflops)
    return exec_time


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError as err:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            ) from err

    parser = argparse.ArgumentParser(
        description="Example of Contiguous Gather Grouped Blockscaled GEMM on Blackwell."
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=4096,
        help="Number of tokens",
    )
    parser.add_argument("--hidden_dim", type=int, default=7168, help="Hidden size K")
    parser.add_argument(
        "--intermediate_dim",
        type=int,
        default=2048,
        help="Intermediate size (N/2 when gated)",
    )
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--topk", type=int, default=2, help="Top-K experts per token")
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="Mma tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument("--a_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--b_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=0, help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference checking"
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )
    testing.add_tensor_init_args(parser, supports_int_dtypes=False)

    parser.add_argument("--swap_ab", action="store_true", help="Enable A/B swap path")
    parser.add_argument(
        "--gated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SwiGLU; --no-gated selects ReLU2",
    )
    parser.add_argument(
        "--vectorized_f32", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--raster_along_m", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--pdl_count",
        type=int,
        default=-1,
        help="K-tile index for launching dependent grids; -1 releases at exit",
    )
    parser.add_argument(
        "--split_k",
        type=int,
        default=1,
        help="Number of cluster-local K partitions",
    )
    parser.add_argument(
        "--weight_interleave",
        type=int,
        choices=(16, 64),
        default=None,
        help="Physical up/gate weight interleave; defaults by swap mode",
    )
    parser.add_argument(
        "--use_alpha",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--expert_alpha_value", type=float, default=1.0)
    parser.add_argument(
        "--use_bias", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--bias_value",
        type=float,
        default=None,
        help="Constant branch bias; omit for deterministic generated biases",
    )
    parser.add_argument("--swiglu_alpha", type=float, default=1.0)
    parser.add_argument("--swiglu_beta", type=float, default=0.0)
    parser.add_argument("--swiglu_limit", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--generate_scaled_output",
        "--scaled_output",
        action="store_true",
        help="Generate and decode block scales for FP8/FP4 output",
    )
    parser.add_argument(
        "--use_compact_sfc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use compact swapped output scales",
    )
    parser.add_argument("--output_global_scale", type=float, default=1.0)
    args = parser.parse_args()

    testing.validate_tensor_init_args(args, parser)

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")
    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    run(
        args.num_tokens,
        args.hidden_dim,
        args.intermediate_dim,
        args.num_experts,
        args.topk,
        args.a_dtype,
        args.b_dtype,
        args.sf_dtype,
        args.sf_vec_size,
        args.c_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        args.init_normal,
        args.normal_mean,
        args.normal_std,
        args.swap_ab,
        args.gated,
        args.vectorized_f32,
        args.raster_along_m,
        args.pdl_count,
        args.use_alpha,
        args.expert_alpha_value,
        args.use_bias,
        args.bias_value,
        args.swiglu_alpha,
        args.swiglu_beta,
        args.swiglu_limit,
        args.seed,
        args.generate_scaled_output,
        args.output_global_scale,
        args.split_k,
        weight_interleave=args.weight_interleave,
        use_compact_sfc=args.use_compact_sfc,
    )
    print("PASS")
