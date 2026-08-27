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
import math
from typing import Optional, Tuple, Type, Union


import cutlass
import cutlass.torch
import torch
from cuda.bindings import driver
from cutlass import cute, testing, utils
from cutlass.cute.arch import (
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    NamedBarrier,
    PipelineAsync,
    PipelineTmaUmma,
    PipelineUmmaAsync,
    PipelineUserType,
    make_pipeline_state,
)
from cutlass.utils import blackwell_helpers, blockscaled_layout

from .utils import (
    DeviceBoundPersistentTileScheduler,
    blk_copy,
    blk_reduce_bf16,
    blk_reduce_fp16,
    blk_reduce_fp32,
    compact_sf_layout,
    is_power_of_2,
    make_ptr,
)

"""
High-performance persistent blockscaled contiguous grouped dense GEMM (C = alpha * (SFA * A) * (SFB * B)) example for
the NVIDIA Blackwell architecture using CUTE DSL.
- Matrix A is MxKx1, A can be row-major("K"), ValidM is composed of valid m in different groups
- Matrix B is NxKxL, B can be column-major("K"), L is grouped dimension
- Matrix C is SxNX1, C can be row-major("N"), ValidM is composed of valid m in different groups
- Matrix SFA layout is filled internally according to A shape and BlockScaledBasicChunk, which has
  M x ceil_div(K, sf_vec_size) x L elements respectively
- Matrix SFB layout is filled internally according to B shape and BlockScaledBasicChunk, which has
  N x ceil_div(K, sf_vec_size) x L elements respectively

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
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. TMA warp: Prefetch expert weights and their scale factors, then load the
   dependency-bound token activations and their scale factors into SMEM.
2. MMA warp:
    - Load SFA and SFB from SMEM to TMEM with tcgen05.cp.
    - Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. EPILOGUE warps (warps 0-3, with Fused Finalize for MoE):
    - Load completed accumulator from tensor memory (TMEM) to registers (RMEM).
    - Apply alpha scaling: acc_scaled = alpha * acc
    - **Fused Finalize Logic** (following TensorRT-LLM's sm90_visitor_scatter.hpp pattern):
      a) Use permuted_idx_to_expanded_idx to map from permuted row to token/topk indices
      b) Load router_scale directly from global memory to register (no shared memory)
      c) Apply router_scale: Final = router_scale * acc_scaled
    - Type convert Final matrix to output type.
    - Store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations.

SM100 tcgen05.mma.kind.block_scale instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Read scalefactor A from TMEM
- Read scalefactor B from TMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

.. code-block:: bash

    python blackwell/blockscaled_contiguous_grouped_gemm_finalize_fusion.py         \
      --a_dtype Float4E2M1FN --b_dtype Float4E2M1FN --out_dtype BFloat16         \
      --sf_dtype Float8E4M3FN --sf_vec_size 16                                   \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
      --benchmark 1024x7168x2048x64

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python blackwell/blockscaled_contiguous_grouped_gemm_finalize_fusion.py        \\     \
      --a_dtype Float4E2M1FN --b_dtype Float4E2M1FN --out_dtype BFloat16           \
      --sf_dtype Float8E4M3FN --sf_vec_size 16                                   \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                             \
      --benchmark [80,120,160]x7168x2048x64

Constraints:
* Supported input data types: mxf8, mxf4, nvf4
  see detailed valid dtype combinations in below Sm100BlockScaledPersistentDenseGemmKernel class documentation
* A/B tensors may use mixed MXFP4 and MXFP8 data types in either operand order.
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 64/128/192/256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.

CUDA Graph Support:
* For CUDA graph support, the tile_idx_to_expert_idx, A/C matrices, and scale factor A can be padded to a larger size
  (e.g., permuted_m = m*topK + num_local_experts*(256-1), example: 4096*8 + (256/32)*255 = 34808)
* Use create_tensors() with permuted_m parameter to automatically pad:
  - tile_idx_to_expert_idx: padded for invalid tiles
  - A matrix: padded to permuted_m rows (padding rows contain dummy data)
  - C matrix: padded to permuted_m rows (output buffer for cuda_graph)
  - Scale factor A: padded to match A matrix dimensions
* Kernel handling of padding (similar to masked_grouped_gemm.py):
  - Scheduler warp checks if tile_idx >= num_non_exiting_tiles to exit
  - Only valid tiles (tile_idx < num_non_exiting_tiles) are written to tile_info pipeline
  - When no more valid tiles exist, outer loop exits and calls producer_tail()
  - Consumer warps process only valid tiles from pipeline
  - No deadlock or synchronization issues
* Consumer warps check initial tile against num_non_exiting_tiles and set is_valid_tile=False if
  tile_idx >= num_non_exiting_tiles
* Only rows within (aligned_groupm[0]+aligned_groupm[1]+...) contain valid data
* Padding rows in C matrix will not be written by the kernel
"""


# TODO(zhichenj): Remove this hook helper function after nvidia-cutlass-dsl 4.3.x is no longer supported.
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
    self, current_work_linear_idx: cutlass.Int32, *, loc=None, ip=None
) -> Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
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


class Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel:
    """This class implements batched matrix multiplication (C = A x SFA x B x SFB) with support for various data types
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - Mixed MXF4 x MXF8: A/B: Float4E2M1FN and Float8E5M2/Float8E4M3FN in either order + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2

    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        - MMA tiler N must be 64/128/192/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Also, Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors

    Example:
        >>> gemm = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel(
        ...     sf_vec_size=16, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 1)
        ... )
        >>> gemm(
        ...     a_tensor, b_tensor, sfa_tensor, sfb_tensor, out_tensor, max_active_clusters, stream
        ... )
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        raster_along_m: bool = False,
        pdl_count: Optional[int] = -1,
        swap_ab: bool = False,
        apply_expert_alpha: bool = True,
        use_a_per_token_scale: bool = False,
        use_fused_finalize: bool = True,
        use_compact_sfb: bool = True,
    ):
        """Initializes the configuration for a Blackwell blockscaled dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        :param sf_vec_size: Vector size for block-scaled scale factors.
        :type sf_vec_size: int
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param raster_along_m: Boolean, True to use raster along M.
        :type raster_along_m: bool
        :param pdl_count: Persistent K-tile index at which to launch dependent
            grids. The index advances across scheduler work tiles. None disables
            PDL, while -1 releases dependent grids when the kernel completes.
        :type pdl_count: Optional[int]
        :param swap_ab: Whether to swap the MMA-A/MMA-B assignments and M/N roles.
        :type swap_ab: bool
        :param apply_expert_alpha: Whether to apply a per-expert alpha scale.
        :type apply_expert_alpha: bool
        :param use_a_per_token_scale: Whether operand A has an additional
            per-token row scale.
        :type use_a_per_token_scale: bool
        :param use_fused_finalize: Whether to apply routing weights and
            atomically reduce the GEMM output into token rows.
        :type use_fused_finalize: bool
        :param use_compact_sfb: Whether swapped activation scales use the compact
            row-major SFB layout. Ignored when swap_ab is False.
        :type use_compact_sfb: bool
        """

        self.sf_vec_size = sf_vec_size
        self.pdl_count = pdl_count
        self.acc_dtype = cutlass.Float32
        self.cluster_shape_mn = cluster_shape_mn
        self.raster_along_m = raster_along_m
        self.swap_ab = swap_ab
        self.apply_expert_alpha = apply_expert_alpha
        self.use_a_per_token_scale = use_a_per_token_scale
        self.use_fused_finalize = use_fused_finalize
        self.use_compact_sfb = swap_ab and use_compact_sfb
        # K dimension is deferred in setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        mma_m = mma_tiler_mn[1] if swap_ab else mma_tiler_mn[0]
        self.use_2cta_instrs = mma_m == 256

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.activation_load_warp_id = 5
        self.sched_warp_id = 6
        self.meta_load_warp_id = 7
        self.weight_load_warp_id = 8
        self.threads_per_warp = 32
        active_warp_ids = (
            *self.epilog_warp_id,
            self.mma_warp_id,
            self.activation_load_warp_id,
            self.sched_warp_id,
            self.meta_load_warp_id,
            self.weight_load_warp_id,
        )
        self.threads_per_cta = self.threads_per_warp * (max(active_warp_ids) + 1)
        self.threads_wo_sched = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.activation_load_warp_id,
                self.meta_load_warp_id,
                self.weight_load_warp_id,
            )
        )

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
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.sched_sync_barrier = NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        # TMEM offset for final accumulator
        self.tmem_final_offset = 384

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

        self.mma_inst_shape_mn = (
            mma_m,
            mma_n,
        )
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

        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
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

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        if cutlass.const_expr(not self.swap_ab):
            self.epi_tile = blackwell_helpers.compute_epilogue_tile_shape(
                self.cta_tile_shape_mnk,
                self.use_2cta_instrs,
                self.gemm_output_layout,
                self.out_dtype,
            )
            self.epi_token = cute.size(self.epi_tile[0])
            self.epi_weight = cute.size(self.epi_tile[1])
        else:
            # Keep a full 128-wide weight row contiguous for the fused
            # scatter-reduce while processing at most 32 token rows at once.
            self.epi_weight = self.cta_tile_shape_mnk[0]
            self.epi_token = min(32, self.cta_tile_shape_mnk[1])
            self.epi_tile = (self.epi_weight, self.epi_token)

        # Setup A/B/C/Scale stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_tile_stage,
            self.num_meta_stage,
        ) = self.compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.smem_alloc_a_dtype,
            self.smem_alloc_b_dtype,
            self.out_dtype,
            self.cta_tile_shape_mnk,
            self.sf_dtype,
            self.sf_vec_size,
            self.final_scale_dtype,
            self.swap_ab,
            self.num_smem_capacity,
            self.occupancy,
        )

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

        swizzled_pad = 16 // (self.out_dtype.width // 8)
        if cutlass.const_expr(not self.swap_ab):
            self.c_smem_layout_staged = cute.make_layout(
                (
                    self.cta_tile_shape_mnk[0],
                    self.cta_tile_shape_mnk[1],
                    self.num_c_stage,
                ),
                stride=(
                    self.cta_tile_shape_mnk[1] + swizzled_pad,
                    1,
                    self.cta_tile_shape_mnk[0] * (self.cta_tile_shape_mnk[1] + 8),
                ),
            )
        else:
            self.c_smem_layout_staged = cute.make_layout(
                (
                    self.cta_tile_shape_mnk[0],
                    self.cta_tile_shape_mnk[1],
                    self.num_c_stage,
                ),
                stride=(
                    1,
                    self.cta_tile_shape_mnk[0] + swizzled_pad,
                    self.cta_tile_shape_mnk[1]
                    * (self.cta_tile_shape_mnk[0] + swizzled_pad),
                ),
            )

        # Overlap and double buffer accumulator when num_acc_stage == 1 for cta_tile_n = 256 case
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

        self.iter_acc_early_release_in_epilogue = (
            self.num_sf_tmem_cols // self.epi_weight
        )

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = 512

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        out: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        alpha: Optional[cute.Tensor],
        max_active_clusters: cutlass.Constexpr,
        stream: driver.CUstream,
        permuted_idx_to_expanded_idx: cute.Tensor,
        token_final_scales: cute.Tensor,
        a_per_token_scale: Optional[cute.Tensor],
        down_bias: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param out: Finalized output tensor (shape [seq_len, n])
        :type out: cute.Tensor
        :param sfa: Scale factor tensor A
        :type sfa: cute.Tensor
        :param sfb: Scale factor tensor B
        :type sfb: cute.Tensor
        :param tile_idx_to_expert_idx: Mapping from tile index to expert ID, shape (permuted_m/cta_tile_m,) where
        cta_tile_m is the CTA tile M size
        :type tile_idx_to_expert_idx: cute.Tensor
        :param num_non_exiting_tiles: Number of valid tiles (valid_m/cta_tile_m), shape (1,)
        :type num_non_exiting_tiles: cute.Tensor
        :param tile_idx_to_mn_limit: M-N boundary limit for each tile.
        :type tile_idx_to_mn_limit: cute.Tensor
        :param alpha: Alpha tensor for each group, or None when apply_expert_alpha=False
        :type alpha: Optional[cute.Tensor]
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: driver.CUstream
        :param permuted_idx_to_expanded_idx: Mapping from permuted index to expanded index, shape (permuted_m,)
        :type permuted_idx_to_expanded_idx: cute.Tensor
        :param token_final_scales: Token-wise scaling factors, shape (m, topK)
        :type token_final_scales: cute.Tensor
        :param a_per_token_scale: Optional per-row scale for operand A.
        :type a_per_token_scale: Optional[cute.Tensor]
        :param down_bias: Down-projection bias tensor.
        :type down_bias: cute.Tensor
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.out_dtype: Type[cutlass.Numeric] = out.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.final_scale_dtype: Type[cutlass.Numeric] = token_final_scales.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.gemm_output_layout = (
            utils.LayoutEnum.COL_MAJOR if self.swap_ab else utils.LayoutEnum.ROW_MAJOR
        )
        self.mxf8f6f4 = self.needs_unpack_tma(self.a_dtype, self.b_dtype)
        if cutlass.const_expr(self.swap_ab and not self.use_compact_sfb):
            if cutlass.const_expr(self.mma_tiler[0] == 256):
                raise ValueError("SFB does not support a 256-row activation tile")

        mma_bias = down_bias
        if cutlass.const_expr(self.swap_ab):
            tokens, weights, experts = (
                down_bias.shape[0],
                down_bias.shape[1],
                down_bias.shape[2],
            )
            # (M=weight, N=token, L)
            mma_bias = cute.make_tensor(
                down_bias.iterator,
                cute.make_layout(
                    (weights, tokens, experts),
                    stride=(1, 0, weights),
                ),
            )

        self.topK = token_final_scales.shape[1]
        # Check if input data types are compatible with MMA instruction
        SUPPORTED_DTYPES = (
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        )
        if cutlass.const_expr(
            self.a_dtype not in SUPPORTED_DTYPES or self.b_dtype not in SUPPORTED_DTYPES
        ):
            raise TypeError(
                f"Unsupported data types: A={self.a_dtype}, B={self.b_dtype}; expected in {SUPPORTED_DTYPES}"
            )

        # Setup attributes that dependent on gemm inputs
        self.setup_attributes()
        if cutlass.const_expr(not self.swap_ab):
            # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestLayout)
            sfa_layout = blockscaled_layout.tile_atom_to_shape_SF(
                a.shape, self.sf_vec_size
            )
            mma_sfa = cute.make_tensor(sfa.iterator, sfa_layout)
            sfb_layout = blockscaled_layout.tile_atom_to_shape_SF(
                b.shape, self.sf_vec_size
            )
            mma_sfb = cute.make_tensor(sfb.iterator, sfb_layout)
            mma_a, mma_b = a, b
        else:
            sfa_layout = blockscaled_layout.tile_atom_to_shape_SF(
                b.shape, self.sf_vec_size
            )
            mma_sfa = cute.make_tensor(sfb.iterator, sfa_layout)
            sfb_layout = blockscaled_layout.tile_atom_to_shape_SF(
                a.shape, self.sf_vec_size
            )
            mma_sfb = cute.make_tensor(sfa.iterator, sfb_layout)
            mma_a, mma_b = b, a

        tiled_mma = blackwell_helpers.make_blockscaled_trivial_tiled_mma(
            mma_a.element_type,
            mma_b.element_type,
            utils.LayoutEnum.from_tensor(mma_a).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mma_b).mma_major_mode(),
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = blackwell_helpers.make_blockscaled_trivial_tiled_mma(
            mma_a.element_type,
            mma_b.element_type,
            utils.LayoutEnum.from_tensor(mma_a).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mma_b).mma_major_mode(),
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        tiled_mma_epilogue = tiled_mma
        if cutlass.const_expr(self.swap_ab and self.use_2cta_instrs):
            tiled_mma_epilogue = blackwell_helpers.make_blockscaled_trivial_tiled_mma(
                mma_a.element_type,
                mma_b.element_type,
                utils.LayoutEnum.from_tensor(mma_a).mma_major_mode(),
                utils.LayoutEnum.from_tensor(mma_b).mma_major_mode(),
                self.sf_dtype,
                self.sf_vec_size,
                cute.nvgpu.tcgen05.CtaGroup.ONE,
                (self.mma_inst_shape_mn[0] // 2, self.mma_inst_shape_mn[1]),
            )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = blackwell_helpers.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            mma_a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                self.smem_alloc_a_dtype
                if (self.mxf8f6f4 and mma_a.element_type.width < 8)
                else None
            ),
        )

        # Setup TMA load for B
        b_op = blackwell_helpers.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            mma_b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                self.smem_alloc_b_dtype
                if (self.mxf8f6f4 and mma_b.element_type.width < 8)
                else None
            ),
        )

        # Setup TMA load for SFA
        sfa_op = blackwell_helpers.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            mma_sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        sfb_op = blackwell_helpers.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            mma_sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

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

        a_copy_size = cute.size_in_bytes(mma_a.element_type, a_smem_layout)
        b_copy_size = cute.size_in_bytes(mma_b.element_type, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        tma_sfa_copy_size = sfa_copy_size
        tma_sfb_copy_size = (
            sfb_copy_size if not self.swap_ab or not self.use_compact_sfb else 0
        )
        if cutlass.const_expr(self.swap_ab):
            self.num_activation_load_bytes = (
                b_copy_size + tma_sfb_copy_size
            ) * atom_thr_size
            self.num_weight_load_bytes = (
                a_copy_size + tma_sfa_copy_size
            ) * atom_thr_size
        else:
            self.num_activation_load_bytes = (
                a_copy_size + tma_sfa_copy_size
            ) * atom_thr_size
            self.num_weight_load_bytes = (
                b_copy_size + tma_sfb_copy_size
            ) * atom_thr_size

        gemm_shape = a.shape
        if cutlass.const_expr(self.swap_ab):
            gemm_shape = (b.shape[0], a.shape[0], a.shape[2])
        else:
            gemm_shape = (a.shape[0], b.shape[0], a.shape[2])
        self.tile_sched_params, grid = self.compute_grid(
            gemm_shape,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.raster_along_m,
        )

        self.buffer_align_bytes = 1024

        #### finalized epi layout ####
        epi_tile_m = cute.size(self.epi_tile[0])
        epi_tile_n = cute.size(self.epi_tile[1])
        epi_tile_size = epi_tile_m * epi_tile_n
        num_epilogue_threads = 32 * len(self.epilog_warp_id)
        self.ttr_racc_size = epi_tile_size // num_epilogue_threads
        self.copy_size = (
            self.cta_tile_shape_mnk[0] if self.swap_ab else self.cta_tile_shape_mnk[1]
        ) * (self.out_dtype.width // 8)

        if cutlass.const_expr(self.out_dtype == cutlass.BFloat16):
            # 8-element vectorization for BF16
            self.epi_layout = cute.make_layout(
                shape=(self.ttr_racc_size // 8, 4, 2), stride=(8, 2, 1)
            )
            self.epi_loop_size = self.ttr_racc_size // 8
            self.element_offset = 8

        elif cutlass.const_expr(self.out_dtype == cutlass.Float32):
            # 2-element vectorization for FP32
            self.epi_layout = cute.make_layout(
                shape=(self.ttr_racc_size // 2, 2), stride=(2, 1)
            )
            self.epi_loop_size = self.ttr_racc_size // 2
            self.element_offset = 2
        else:
            # Scalar fallback
            self.epi_layout = cute.make_layout(shape=(self.ttr_racc_size,), stride=(1,))
            self.epi_loop_size = self.ttr_racc_size
            self.element_offset = 1

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            # (bidx, bidy, bidz, valid, mn_limit)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            activation_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            weight_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            meta_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_meta_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
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

            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.out_dtype,
                    cute.cosize(self.c_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]
            meta_token_idx: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32,
                    (
                        self.cta_tile_shape_mnk[1]
                        if self.swap_ab
                        else self.cta_tile_shape_mnk[0]
                    )
                    * self.num_meta_stage,
                ],
                1,
            ]
            meta_scale: cute.struct.Align[
                cute.struct.MemRange[
                    self.final_scale_dtype,
                    (
                        self.cta_tile_shape_mnk[1]
                        if self.swap_ab
                        else self.cta_tile_shape_mnk[0]
                    )
                    * self.num_meta_stage,
                ],
                1,
            ]
            meta_bias_scale: cute.struct.Align[
                cute.struct.MemRange[
                    self.final_scale_dtype,
                    (
                        self.cta_tile_shape_mnk[1]
                        if self.swap_ab
                        else self.cta_tile_shape_mnk[0]
                    )
                    * self.num_meta_stage,
                ],
                1,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tiled_mma_epilogue,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            sfa,
            out,
            tile_idx_to_expert_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            permuted_idx_to_expanded_idx,
            token_final_scales,
            a_per_token_scale,
            mma_bias,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.epi_layout,
            self.topK,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
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
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and
        tensor memory (destination).

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

    @cute.jit
    def get_sfb_atom_rows(self):
        """Return the row extent encoded by the block-scaled SFB atom."""
        return cute.size(
            blockscaled_layout.BlockScaledBasicChunk(self.sf_vec_size).layout,
            mode=[0],
        )

    @cute.jit
    def make_blockscaled_sfb_word_layout(self, rows, k) -> cute.Layout:
        """Create a block-scaled SFB layout: (row, four-scale word)."""
        sfb_atom_layout = blockscaled_layout.BlockScaledBasicChunk(
            self.sf_vec_size
        ).layout
        sfb_atom_rows = cute.size(sfb_atom_layout, mode=[0])
        sfb_layout = cute.tile_to_shape(
            sfb_atom_layout,
            (cute.round_up(rows, sfb_atom_rows), k),
            (2, 1),
        )
        sfb_layout = cute.filter_zeros(sfb_layout)
        return cute.recast_layout(
            cutlass.Uint32.width,
            self.sf_dtype.width,
            sfb_layout,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tiled_mma_epilogue: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        token_sf_raw: cute.Tensor,
        out: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        alpha: Optional[cute.Tensor],
        permuted_idx_to_expanded_idx: cute.Tensor,
        token_final_scales: cute.Tensor,
        a_per_token_scale: Optional[cute.Tensor],
        mBias_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        epi_layout: cute.Layout,
        topK: cutlass.Int32,
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
        if warp_idx == self.activation_load_warp_id:
            if cutlass.const_expr(self.swap_ab):
                cpasync.prefetch_descriptor(tma_atom_b)
                if cutlass.const_expr(not self.use_compact_sfb):
                    cpasync.prefetch_descriptor(tma_atom_sfb)
            else:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_sfa)
        if warp_idx == self.weight_load_warp_id:
            if cutlass.const_expr(self.swap_ab):
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_sfa)
            else:
                cpasync.prefetch_descriptor(tma_atom_b)
                cpasync.prefetch_descriptor(tma_atom_sfb)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
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

        # Independent pipelines let weight TMA run ahead of dependent activations.
        activation_num_consumers = (
            self.num_mcast_ctas_b if self.swap_ab else self.num_mcast_ctas_a
        )
        weight_num_consumers = (
            self.num_mcast_ctas_a if self.swap_ab else self.num_mcast_ctas_b
        )
        activation_mcast_mode_mn = (0, 1) if self.swap_ab else (1, 0)
        weight_mcast_mode_mn = (1, 0) if self.swap_ab else (0, 1)
        activation_pipeline = PipelineTmaUmma.create(
            barrier_storage=storage.activation_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, activation_num_consumers),
            tx_count=self.num_activation_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=activation_mcast_mode_mn,
        )
        weight_pipeline = PipelineTmaUmma.create(
            barrier_storage=storage.weight_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, weight_num_consumers),
            tx_count=self.num_weight_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=weight_mcast_mode_mn,
        )

        # Initialize acc_pipeline (barrier) and states
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

        # Initialize tile info pipeline (barrier) and states
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

        # Initialize metadata pipeline (meta loader warp -> epilogue warps)
        meta_pipeline_producer_group = CooperativeGroup(
            Agent.Thread,
            self.threads_per_warp * 1,
        )
        meta_pipeline_consumer_group = CooperativeGroup(
            Agent.Thread,
            self.threads_per_warp * len(self.epilog_warp_id),
        )
        meta_pipeline = PipelineAsync.create(
            barrier_storage=storage.meta_mbar_ptr.data_ptr(),
            num_stages=self.num_meta_stage,
            producer_group=meta_pipeline_producer_group,
            consumer_group=meta_pipeline_consumer_group,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        #
        # Setup smem tensor A/B/C/Scale/ExpandedIdx
        #
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

        sC = storage.sC.get_tensor(c_smem_layout_staged)

        # (bidx, bidy, bidz, valid)
        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        sInfo = storage.sInfo.get_tensor(info_layout)

        # Per-row finalize metadata staged by the meta loader warp: (row, stage)
        meta_token_tile = (
            self.cta_tile_shape_mnk[1] if self.swap_ab else self.cta_tile_shape_mnk[0]
        )
        meta_layout = cute.make_layout(
            (meta_token_tile, self.num_meta_stage),
            stride=(1, meta_token_tile),
        )
        sMetaTokenIdx = storage.meta_token_idx.get_tensor(meta_layout)
        sMetaScale = storage.meta_scale.get_tensor(meta_layout)
        sMetaBiasScale = storage.meta_bias_scale.get_tensor(meta_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
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

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )

        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )

        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )

        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        #
        # Partition global tensor for TiledMMA_A/B
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        thr_mma_epilogue = tiled_mma_epilogue.get_slice(0)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMA load SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)

        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )

        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
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

        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )
            # (MMA, MMA_M, MMA_N, STAGE)
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
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, self.num_acc_stage)
            )

        epilogue_tile = self.mma_tiler
        if cutlass.const_expr(self.swap_ab and use_2cta_instrs):
            epilogue_tile = self.cta_tile_shape_mnk
        gBias_mnl = cute.local_tile(
            mBias_mnl, cute.slice_(epilogue_tile, (None, None, 0)), (None, None, None)
        )
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        if cutlass.const_expr(self.swap_ab and use_2cta_instrs):
            tCgBias = thr_mma_epilogue.partition_C(gBias_mnl)
            tCgC = thr_mma_epilogue.partition_C(gBias_mnl)
        else:
            tCgBias = thr_mma.partition_C(gBias_mnl)
            tCgC = thr_mma.partition_C(gBias_mnl)

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        num_valid_tiles = num_non_exiting_tiles[0]
        create_tile_scheduler = (
            DeviceBoundPersistentTileScheduler.create_n
            if self.swap_ab
            else DeviceBoundPersistentTileScheduler.create_static_n
        )
        tile_sched = create_tile_scheduler(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            num_valid_tiles,
        )

        #
        # Specialized Schedule warp
        #
        if warp_idx == self.sched_warp_id:
            #
            # Persistent tile scheduling loop, starting after the pre-emitted
            # first tile.
            #
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
                    tile_idx = (
                        cur_tile_coord[1]
                        if cutlass.const_expr(self.swap_ab)
                        else mma_tile_coord_m
                    )
                    expert_idx = tile_idx_to_expert_idx[tile_idx]
                    if tile_idx < num_valid_tiles:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        mn_limit = tile_idx_to_mn_limit[tile_idx]
                        if cutlass.const_expr(not self.swap_ab):
                            num_n_tiles = tile_sched_params.problem_shape_ntile_mnl[1]
                            n_coord = cutlass.min(cur_tile_coord[1], num_n_tiles - 1)
                            mn_limit = mn_limit * cutlass.Int32(
                                cur_tile_coord[1] < num_n_tiles
                            )
                        else:
                            n_coord = cur_tile_coord[1]
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[
                                0
                            ]
                            sInfo[(1, tile_info_producer_state.index)] = n_coord
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
                    tile_idx = (
                        cur_tile_coord[1]
                        if cutlass.const_expr(self.swap_ab)
                        else mma_tile_coord_m
                    )
                    expert_idx = tile_idx_to_expert_idx[tile_idx]
                    if tile_idx < num_valid_tiles:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        mn_limit = tile_idx_to_mn_limit[tile_idx]
                        if cutlass.const_expr(not self.swap_ab):
                            num_n_tiles = tile_sched_params.problem_shape_ntile_mnl[1]
                            n_coord = cutlass.min(cur_tile_coord[1], num_n_tiles - 1)
                            mn_limit = mn_limit * cutlass.Int32(
                                cur_tile_coord[1] < num_n_tiles
                            )
                        else:
                            n_coord = cur_tile_coord[1]
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[
                                0
                            ]
                            sInfo[(1, tile_info_producer_state.index)] = n_coord
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
                sInfo[(4, tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized activation TMA load warp
        #
        if warp_idx == self.activation_load_warp_id:
            pdl_k_tile = cutlass.Int32(0)
            ab_producer_state = make_pipeline_state(
                PipelineUserType.Producer, self.num_ab_stage
            )

            if cutlass.const_expr(self.use_compact_sfb):
                sfb_atom_copy = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(),
                    cutlass.Uint32,
                    num_bits_per_copy=32,
                )
                sfb_copy_predicate = cute.make_rmem_tensor(
                    cute.make_layout((1,)), cutlass.Boolean
                )
                activation_rows = self.cta_tile_shape_mnk[1]
                row_groups = cute.ceil_div(activation_rows, self.threads_per_warp)
                # (lane, iteration)
                row_copy_layout = cute.make_ordered_layout(
                    (self.threads_per_warp, row_groups),
                    order=(0, 1),
                )
                # (N, loopN)
                activation_row_layout = cute.make_ordered_layout(
                    (
                        activation_rows,
                        cute.ceil_div(mA_mkl.shape[0], activation_rows),
                    ),
                    order=(0, 1),
                )
                # (N, K_word)
                sfb_stage_word_layout = self.make_blockscaled_sfb_word_layout(
                    activation_rows,
                    self.cta_tile_shape_mnk[2],
                )
                sf_groups_per_k_tile = cute.size(sfb_stage_word_layout, mode=[1])
                # (N, K_scale)
                compact_sfb = token_sf_raw[(None, None, 0)]
                # (N, K_word)
                sfb_source_words = cute.make_tensor(
                    cute.recast_ptr(
                        compact_sfb.iterator.align(4), dtype=cutlass.Uint32
                    ),
                    cute.recast_layout(
                        cutlass.Uint32.width,
                        self.sf_dtype.width,
                        compact_sfb.layout,
                    ),
                )
                source_sf_groups = cute.size(sfb_source_words, mode=[1])
                # (K_word, loopK)
                source_sf_group_layout = cute.make_ordered_layout(
                    (
                        sf_groups_per_k_tile,
                        cute.ceil_div(source_sf_groups, sf_groups_per_k_tile),
                    ),
                    order=(0, 1),
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
                    tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, 0)]
                    tBgB_slice = tBgB[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]
                    tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, 0)]
                else:
                    tAgA_slice = tAgA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                    tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, 0)]
                    tAgSFA_slice = tAgSFA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                    if cutlass.const_expr(not self.use_compact_sfb):
                        # (N_subtile, Atom_N)
                        sfb_subtiles_per_atom = (
                            self.get_sfb_atom_rows() // self.cta_tile_shape_mnk[1]
                        )
                        _, sfb_atom_tile = cute.idx2crd(
                            mma_tile_coord_mnl[1],
                            (
                                sfb_subtiles_per_atom,
                                cute.size(tBgSFB, mode=[1]),
                            ),
                        )
                        tBgSFB_slice = tBgSFB[(None, sfb_atom_tile, None, 0)]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = activation_pipeline.producer_try_acquire(
                        ab_producer_state
                    )
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                    tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                    tBgB_k = tBgB_slice[(None, ab_producer_state.count)]
                    tAgSFA_k = tAgSFA_slice[(None, ab_producer_state.count)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]
                    tAsSFA_pipe = tAsSFA[(None, ab_producer_state.index)]
                    tma_bar = activation_pipeline.producer_get_barrier(
                        ab_producer_state
                    )

                    # Conditionally wait for AB buffer empty
                    activation_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    griddepcontrol_wait()

                    if cutlass.const_expr(self.use_compact_sfb):
                        lane = cute.arch.lane_idx()
                        activation_tile = mma_tile_coord_mnl[1]
                        sf_stage = sSFB[(None, None, None, ab_producer_state.index)]
                        # (N, K_word)
                        sfb_stage_words = cute.make_tensor(
                            cute.recast_ptr(
                                sf_stage.iterator.align(4), dtype=cutlass.Uint32
                            ),
                            sfb_stage_word_layout,
                        )
                        for group in cutlass.range_constexpr(row_groups):
                            local_row = row_copy_layout((lane, group))
                            if local_row < activation_rows:
                                global_row = activation_row_layout(
                                    (local_row, activation_tile)
                                )
                                for sf_group in cutlass.range_constexpr(
                                    sf_groups_per_k_tile
                                ):
                                    source_sf_group = source_sf_group_layout(
                                        (sf_group, ab_producer_state.count)
                                    )
                                    source = cute.local_tile(
                                        sfb_source_words,
                                        (1, 1),
                                        (global_row, source_sf_group),
                                    )
                                    destination = cute.local_tile(
                                        sfb_stage_words,
                                        (1, 1),
                                        (local_row, sf_group),
                                    )
                                    source = cute.group_modes(
                                        source, 0, cute.rank(source)
                                    )
                                    destination = cute.group_modes(
                                        destination, 0, cute.rank(destination)
                                    )
                                    sfb_copy_predicate[0] = (
                                        global_row < tile_info[4]
                                        and source_sf_group < source_sf_groups
                                    )
                                    cute.copy_atom_call(
                                        sfb_atom_copy,
                                        source,
                                        destination,
                                        pred=sfb_copy_predicate,
                                    )
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_proxy("async.shared", space="cta")
                        cute.arch.sync_warp()

                    if cutlass.const_expr(self.swap_ab):
                        cute.copy(
                            tma_atom_b,
                            tBgB_k,
                            tBsB_pipe,
                            tma_bar_ptr=tma_bar,
                            mcast_mask=b_full_mcast_mask,
                        )
                        if cutlass.const_expr(not self.use_compact_sfb):
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, ab_producer_state.count)],
                                tBsSFB[(None, ab_producer_state.index)],
                                tma_bar_ptr=tma_bar,
                                mcast_mask=sfb_full_mcast_mask,
                            )
                    else:
                        cute.copy(
                            tma_atom_a,
                            tAgA_k,
                            tAsA_pipe,
                            tma_bar_ptr=tma_bar,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_k,
                            tAsSFA_pipe,
                            tma_bar_ptr=tma_bar,
                            mcast_mask=sfa_full_mcast_mask,
                        )

                    if cutlass.const_expr(self.pdl_count is not None):
                        if pdl_k_tile == self.pdl_count:
                            griddepcontrol_launch_dependents()
                    pdl_k_tile += 1

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = activation_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

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
            # Wait activation buffers empty
            #
            activation_pipeline.producer_tail(ab_producer_state)

        # The weight warp has no grid dependency and can fill every available stage.
        if warp_idx == self.weight_load_warp_id:
            weight_producer_state = make_pipeline_state(
                PipelineUserType.Producer, self.num_ab_stage
            )
            tile_info_consumer_state = make_pipeline_state(
                PipelineUserType.Consumer, self.num_tile_stage
            )

            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
            tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                if cutlass.const_expr(self.swap_ab):
                    tAgA_weight = tAgA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                    tAgSFA_weight = tAgSFA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                else:
                    tBgB_weight = tBgB[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]
                    slice_n = mma_tile_coord_mnl[1]
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                        slice_n = mma_tile_coord_mnl[1] // 2
                    tBgSFB_weight = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                weight_producer_state.reset_count()
                peek_weight_empty_status = cutlass.Boolean(1)
                if weight_producer_state.count < k_tile_cnt:
                    peek_weight_empty_status = weight_pipeline.producer_try_acquire(
                        weight_producer_state
                    )

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                    weight_bar = weight_pipeline.producer_get_barrier(
                        weight_producer_state
                    )
                    weight_pipeline.producer_acquire(
                        weight_producer_state, peek_weight_empty_status
                    )
                    if cutlass.const_expr(self.swap_ab):
                        cute.copy(
                            tma_atom_a,
                            tAgA_weight[(None, weight_producer_state.count)],
                            tAsA[(None, weight_producer_state.index)],
                            tma_bar_ptr=weight_bar,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_weight[(None, weight_producer_state.count)],
                            tAsSFA[(None, weight_producer_state.index)],
                            tma_bar_ptr=weight_bar,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                    else:
                        cute.copy(
                            tma_atom_b,
                            tBgB_weight[(None, weight_producer_state.count)],
                            tBsB[(None, weight_producer_state.index)],
                            tma_bar_ptr=weight_bar,
                            mcast_mask=b_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_weight[(None, weight_producer_state.count)],
                            tBsSFB[(None, weight_producer_state.index)],
                            tma_bar_ptr=weight_bar,
                            mcast_mask=sfb_full_mcast_mask,
                        )

                    weight_producer_state.advance()
                    peek_weight_empty_status = cutlass.Boolean(1)
                    if weight_producer_state.count < k_tile_cnt:
                        peek_weight_empty_status = weight_pipeline.producer_try_acquire(
                            weight_producer_state
                        )

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            weight_pipeline.producer_tail(weight_producer_state)

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

            # SFA remains on the direct SMEM -> TMEM multicast path.
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

            activation_consumer_state = make_pipeline_state(
                PipelineUserType.Consumer, self.num_ab_stage
            )
            weight_consumer_state = make_pipeline_state(
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
                # Peek both operand pipelines for k_tile = 0.
                activation_consumer_state.reset_count()
                weight_consumer_state.reset_count()
                peek_activation_full_status = cutlass.Boolean(1)
                peek_weight_full_status = cutlass.Boolean(1)
                if activation_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_activation_full_status = activation_pipeline.consumer_try_wait(
                        activation_consumer_state
                    )
                    peek_weight_full_status = weight_pipeline.consumer_try_wait(
                        weight_consumer_state
                    )

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

                tCtSFB_tile = tCtSFB
                use_sfb_panel = (
                    not self.swap_ab and self.cta_tile_shape_mnk[1] in (64, 192)
                ) or (
                    self.swap_ab
                    and not self.use_compact_sfb
                    and self.cta_tile_shape_mnk[1] == 64
                )
                if cutlass.const_expr(use_sfb_panel):
                    sfb_panel, _ = cute.idx2crd(
                        mma_tile_coord_mnl[1], sfb_panel_coord_shape
                    )
                    sfb_panel_offset = sfb_panel_layout(sfb_panel)
                    # (MMA, MMA_N, MMA_K, STAGE)
                    tCtSFB_tile = cute.make_tensor(
                        cute.recast_ptr(
                            acc_tmem_ptr
                            + self.num_accumulator_tmem_cols
                            + self.num_sfa_tmem_cols
                            + sfb_panel_offset,
                            dtype=self.sf_dtype,
                        ),
                        tCtSFB_layout,
                    )
                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in cutlass.range(k_tile_cnt):  # noqa: B007
                    if is_leader_cta:
                        activation_pipeline.consumer_wait(
                            activation_consumer_state,
                            peek_activation_full_status,
                        )
                        weight_pipeline.consumer_wait(
                            weight_consumer_state,
                            peek_weight_full_status,
                        )

                        if cutlass.const_expr(
                            self.swap_ab
                            and not self.use_compact_sfb
                            and self.cta_tile_shape_mnk[1] <= 32
                        ):
                            activation_rows = self.cta_tile_shape_mnk[1]
                            panel_subtiles = self.get_sfb_atom_rows() // activation_rows
                            sfb_subtile, _ = cute.idx2crd(
                                mma_tile_coord_mnl[1],
                                (
                                    panel_subtiles,
                                    cute.ceil_div(
                                        cute.size(tBgB, mode=[1]),
                                        panel_subtiles,
                                    ),
                                ),
                            )
                            # (N, N_subtile)
                            sfb_row_layout = cute.make_ordered_layout(
                                (activation_rows, panel_subtiles), order=(0, 1)
                            )
                            if sfb_subtile != 0:
                                lane = cute.arch.lane_idx()
                                if lane < activation_rows:
                                    source_row = sfb_row_layout((lane, sfb_subtile))
                                    scale_stage = sSFB[
                                        (
                                            None,
                                            None,
                                            None,
                                            activation_consumer_state.index,
                                        )
                                    ]
                                    scale_stage_words = cute.make_tensor(
                                        cute.recast_ptr(
                                            scale_stage.iterator.align(4),
                                            dtype=cutlass.Uint32,
                                        ),
                                        self.make_blockscaled_sfb_word_layout(
                                            activation_rows,
                                            self.cta_tile_shape_mnk[2],
                                        ),
                                    )
                                    source_word = cute.local_tile(
                                        scale_stage_words,
                                        (1, 1),
                                        (source_row, 0),
                                    )
                                    destination_word = cute.local_tile(
                                        scale_stage_words,
                                        (1, 1),
                                        (lane, 0),
                                    )
                                    destination_word[0] = source_word[0]
                                cute.arch.fence_proxy("async.shared", space="cta")
                                cute.arch.sync_warp()

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            activation_consumer_state.index,
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
                        tCtSFB_mma = tCtSFB_tile

                        # NVFP4 follows the mainline per-kblock sequence.
                        # MXFP8 keeps the combined operand/scale form.
                        if cutlass.const_expr(self.a_dtype == cutlass.Float4E2M1FN):
                            num_kblocks = cute.size(tCrA, mode=[2])
                            for kblock_idx in cutlass.range(
                                num_kblocks, unroll_full=True
                            ):
                                kblock_coord = (
                                    None,
                                    None,
                                    kblock_idx,
                                    activation_consumer_state.index,
                                )
                                sf_kblock_coord = (None, None, kblock_idx)
                                tiled_mma.set(
                                    tcgen05.Field.SFA,
                                    tCtSFA[sf_kblock_coord].iterator,
                                )
                                tiled_mma.set(
                                    tcgen05.Field.SFB,
                                    tCtSFB_mma[sf_kblock_coord].iterator,
                                )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    tCrA[kblock_coord],
                                    tCrB[kblock_coord],
                                    tCtAcc,
                                )
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        else:
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                            tile_crd = (
                                None,
                                None,
                                None,
                                activation_consumer_state.index,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                [tCrA[tile_crd], tCtSFA],
                                [tCrB[tile_crd], tCtSFB_mma],
                                tCtAcc,
                            )

                        activation_pipeline.consumer_release(activation_consumer_state)
                        weight_pipeline.consumer_release(weight_consumer_state)

                    # Peek both operand pipelines for k_tile = k_tile + 1.
                    activation_consumer_state.advance()
                    weight_consumer_state.advance()
                    peek_activation_full_status = cutlass.Boolean(1)
                    peek_weight_full_status = cutlass.Boolean(1)
                    if activation_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_activation_full_status = (
                                activation_pipeline.consumer_try_wait(
                                    activation_consumer_state
                                )
                            )
                            peek_weight_full_status = weight_pipeline.consumer_try_wait(
                                weight_consumer_state
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
        # Specialized metadata loader warp
        #
        if warp_idx == self.meta_load_warp_id:
            meta_lane = tidx % self.threads_per_warp

            tile_info_consumer_state = make_pipeline_state(
                PipelineUserType.Consumer, self.num_tile_stage
            )
            meta_producer_state = make_pipeline_state(
                PipelineUserType.Producer, self.num_meta_stage
            )
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
            tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            # Token routing metadata and final scales are produced by the
            # preceding grid; weight TMA prefetch remains independent
            griddepcontrol_wait()

            while is_valid_tile:
                if cutlass.const_expr(not self.swap_ab):
                    tile_m_start = tile_info[0] * self.cta_tile_shape_mnk[0]
                    token_tile = self.cta_tile_shape_mnk[0]
                else:
                    tile_m_start = tile_info[1] * self.cta_tile_shape_mnk[1]
                    token_tile = self.cta_tile_shape_mnk[1]
                expert_idx = tile_info[2]
                if cutlass.const_expr(self.apply_expert_alpha):
                    alpha_val = alpha[expert_idx]
                else:
                    alpha_val = cutlass.Float32(1.0)

                meta_pipeline.producer_acquire(meta_producer_state)
                meta_stage = meta_producer_state.index
                # Strided row assignment keeps the permuted_idx loads and smem
                # stores coalesced (each fixed j: 32 lanes touch 32 contiguous
                # rows). Rows beyond mn_limit are padding: their expanded_idx is
                # not guaranteed to be in range, so gate the token_final_scales
                # gather to token 0 for them (branchless) to keep it in-bounds.
                # The epilogue ignores padding rows via its own is_valid_row, so
                # the value staged for them is irrelevant.
                for j in cutlass.range(
                    cute.ceil_div(token_tile, self.threads_per_warp),
                    unroll_full=True,
                ):
                    r = meta_lane + j * self.threads_per_warp
                    if r < token_tile:
                        permuted_row = tile_m_start + r
                        expanded_idx = permuted_idx_to_expanded_idx[permuted_row]
                        safe_idx = cutlass.max(expanded_idx, cutlass.Int32(0))
                        token_idx = safe_idx // topK
                        topk_idx = safe_idx % topK
                        is_valid_row = cutlass.Int32(permuted_row < tile_info[4])
                        gather_tok = token_idx * is_valid_row
                        token_scale = token_final_scales[(gather_tok, topk_idx)]
                        if cutlass.const_expr(not self.use_fused_finalize):
                            token_idx = safe_idx
                            token_scale = self.final_scale_dtype(1.0)
                        if cutlass.const_expr(self.use_a_per_token_scale):
                            token_scale = cutlass.Float32(
                                token_scale
                            ) * cutlass.Float32(a_per_token_scale[permuted_row])
                        sMetaTokenIdx[(r, meta_stage)] = token_idx
                        sMetaScale[(r, meta_stage)] = alpha_val * token_scale
                        sMetaBiasScale[(r, meta_stage)] = token_scale
                cute.arch.fence_proxy("async.shared", space="cta")
                meta_pipeline.producer_commit(meta_producer_state)
                meta_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            meta_pipeline.producer_tail(meta_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
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
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.out_dtype)
            tiled_copy_r2s, tRS_rC, _tRS_sC = self.epilog_smem_copy_and_partition(
                epi_tidx, tTR_rC, sC, tiled_copy_t2r
            )
            c_smem_layout_stage = cute.make_layout(
                (
                    c_smem_layout_staged.shape[0],
                    c_smem_layout_staged.shape[1],
                    1,
                ),
                stride=c_smem_layout_staged.stride,
            )

            thr_copy_t2r_bias = tiled_copy_t2r.get_slice(epi_tidx)
            tCgBias_epi = cute.flat_divide(
                tCgBias[((None, None), 0, 0, None, None, None)], epi_tile
            )
            tTR_gBias = thr_copy_t2r_bias.partition_D(tCgBias_epi)

            tTR_cId = None
            if cutlass.const_expr(self.swap_ab):
                c_id = cute.make_identity_tensor(
                    (
                        self.cta_tile_shape_mnk[0],
                        self.cta_tile_shape_mnk[1],
                    )
                )
                c_id_epi = cute.flat_divide(c_id, epi_tile)
                tTR_cId = thr_copy_t2r_bias.partition_D(c_id_epi)
                tTR_cId = cute.group_modes(tTR_cId, 3, cute.rank(tTR_cId))

            acc_consumer_state = make_pipeline_state(
                PipelineUserType.Consumer, self.num_acc_stage
            )
            tile_info_consumer_state = make_pipeline_state(
                PipelineUserType.Consumer, self.num_tile_stage
            )
            meta_consumer_state = make_pipeline_state(
                PipelineUserType.Consumer, self.num_meta_stage
            )

            # Get the first tile info
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
                # Read per-row finalize metadata prefetched by the meta loader
                # warp (token_idx for scatter, combined_scale = alpha * token_scale).
                meta_pipeline.consumer_wait(meta_consumer_state)
                # The compact swapped output uses two C stages. Keep the
                # larger unswapped output single-buffered to avoid an
                # occupancy loss from the additional shared memory.
                c_stage = cutlass.Int32(0)
                if cutlass.const_expr(self.swap_ab):
                    c_stage = meta_consumer_state.index
                sC_stage = cute.domain_offset((0, 0, c_stage), sC)
                sC_stage = cute.make_tensor(
                    sC_stage.iterator,
                    c_smem_layout_stage,
                )
                tRS_sC_stage = tiled_copy_r2s.get_slice(epi_tidx).partition_D(sC_stage)
                token_idx = cutlass.Int32(0)
                if cutlass.const_expr(not self.swap_ab):
                    tile_m_start = tile_info[0] * self.cta_tile_shape_mnk[0]
                    permuted_row = tile_m_start + epi_tidx
                    is_valid_row = permuted_row < tile_info[4]
                    token_idx = sMetaTokenIdx[(epi_tidx, meta_consumer_state.index)]
                    meta_scale = sMetaScale[(epi_tidx, meta_consumer_state.index)]
                    bias_scale = sMetaBiasScale[(epi_tidx, meta_consumer_state.index)]

                tTR_gBias_tile = tTR_gBias[
                    (
                        None,
                        None,
                        None,
                        None,
                        None,
                        tile_info[0],
                        tile_info[1],
                        tile_info[2],
                    )
                ]
                tTR_gBias_tile = cute.group_modes(
                    tTR_gBias_tile, 3, cute.rank(tTR_gBias_tile)
                )

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

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                #
                # Process sub-tiles with vectorized scatter-add
                # Following TensorRT-LLM's direct G2R (global to register) approach
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = subtile_cnt - 1 - subtile_idx
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]

                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    # The N=256 accumulator stages overlap in TMEM. Consume the
                    # aliased region first, then release it so the MMA warp can
                    # start producing the next stage while this warp drains the
                    # remaining non-overlapping subtiles.
                    if cutlass.const_expr(self.overlapping_accum and not self.swap_ab):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            cute.arch.fence_view_async_tmem_load()
                            acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                    # Get vectorized accumulator and apply the combined finalize scale,
                    # then add the router-weighted per-expert down bias
                    acc_vec = tTR_rAcc.load()
                    bias_vec = tTR_gBias_tile[
                        (None, None, None, real_subtile_idx)
                    ].load()
                    if cutlass.const_expr(not self.swap_ab):
                        acc_vec_final = meta_scale * acc_vec + bias_scale * bias_vec
                    else:
                        acc_vec_final = cute.make_rmem_tensor(
                            tTR_rAcc.shape, self.acc_dtype
                        )
                        c_id_vec = tTR_cId[(None, None, None, real_subtile_idx)]
                        for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                            token_local = c_id_vec[i][1]
                            element_scale = sMetaScale[
                                (token_local, meta_consumer_state.index)
                            ]
                            element_bias_scale = sMetaBiasScale[
                                (token_local, meta_consumer_state.index)
                            ]
                            acc_vec_final[i] = (
                                element_scale * acc_vec[i]
                                + element_bias_scale * bias_vec[i]
                            )

                    if cutlass.const_expr(not self.swap_ab):
                        tRS_rC.store(epilogue_op(acc_vec_final.to(self.out_dtype)))
                    else:
                        tRS_rC.store(
                            epilogue_op(acc_vec_final.load().to(self.out_dtype))
                        )
                    if cutlass.const_expr(self.swap_ab) or is_valid_row:
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC_stage[(None, None, real_subtile_idx, None)],
                        )

                # Make all R2S smem writes visible to the async bulk-reduce proxy.
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                token_tile_start = (
                    tile_info[1] * self.cta_tile_shape_mnk[1]
                    if cutlass.const_expr(self.swap_ab)
                    else tile_info[0] * self.cta_tile_shape_mnk[0]
                )
                token_tile_size = (
                    self.cta_tile_shape_mnk[1]
                    if cutlass.const_expr(self.swap_ab)
                    else self.cta_tile_shape_mnk[0]
                )
                is_partial_tile = tile_info[4] < token_tile_start + token_tile_size
                #
                # Async arrive accumulator buffer empty
                #
                if cutlass.const_expr(not self.overlapping_accum or self.swap_ab):
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                num_epilogue_threads = self.threads_per_warp * len(self.epilog_warp_id)
                needs_reduce_sync = (
                    self.swap_ab
                    or is_partial_tile
                    or token_tile_size > num_epilogue_threads
                )
                if needs_reduce_sync:
                    self.epilog_sync_barrier.arrive_and_wait()

                for reduce_group in cutlass.range_constexpr(
                    cute.ceil_div(token_tile_size, num_epilogue_threads)
                ):
                    reduce_row = epi_tidx + reduce_group * num_epilogue_threads
                    if is_partial_tile:
                        reduce_row = (
                            (epi_tidx % self.threads_per_warp)
                            * len(self.epilog_warp_id)
                            + (epi_tidx // self.threads_per_warp)
                            + reduce_group * num_epilogue_threads
                        )
                    reduce_permuted_row = token_tile_start + reduce_row
                    is_valid_reduce_row = (
                        reduce_row < token_tile_size
                        and reduce_permuted_row < tile_info[4]
                    )

                    if is_valid_reduce_row:
                        token_idx = sMetaTokenIdx[
                            (reduce_row, meta_consumer_state.index)
                        ]
                        coord_n = (
                            tile_info[0] * self.cta_tile_shape_mnk[0]
                            if cutlass.const_expr(self.swap_ab)
                            else tile_info[1] * self.cta_tile_shape_mnk[1]
                        )
                        tile_columns = (
                            self.cta_tile_shape_mnk[0]
                            if cutlass.const_expr(self.swap_ab)
                            else self.cta_tile_shape_mnk[1]
                        )
                        valid_columns = cutlass.min(
                            cutlass.Int64(out.shape[1]) - coord_n,
                            cutlass.Int64(tile_columns),
                        )
                        # Padded CTAs beyond N have no valid transfer.
                        if valid_columns > 0:
                            scatter_out_offset = cute.domain_offset(
                                (token_idx, coord_n, 0), out
                            )
                            valid_bytes = cutlass.Int32(
                                valid_columns * (self.out_dtype.width // 8)
                            )
                            smem_row = (
                                sC[None, reduce_row, c_stage]
                                if cutlass.const_expr(self.swap_ab)
                                else sC[reduce_row, None, c_stage]
                            )
                            if cutlass.const_expr(not self.use_fused_finalize):
                                blk_copy(scatter_out_offset, smem_row, valid_bytes)
                            elif cutlass.const_expr(self.out_dtype == cutlass.BFloat16):
                                blk_reduce_bf16(
                                    scatter_out_offset, smem_row, valid_bytes
                                )
                            elif cutlass.const_expr(self.out_dtype == cutlass.Float32):
                                blk_reduce_fp32(
                                    scatter_out_offset, smem_row, valid_bytes
                                )
                            elif cutlass.const_expr(self.out_dtype == cutlass.Float16):
                                blk_reduce_fp16(
                                    scatter_out_offset, smem_row, valid_bytes
                                )

                cute.arch.cp_async_bulk_commit_group()
                if cutlass.const_expr(self.swap_ab):
                    # Leave the newest reduction in flight while the next tile
                    # is converted and staged into the other C buffer.
                    cute.arch.cp_async_bulk_wait_group(1, read=True)
                else:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                self.epilog_sync_barrier.arrive_and_wait()

                # Release the prefetched metadata slot for this tile.
                meta_pipeline.consumer_release(meta_consumer_state)
                meta_consumer_state.advance()

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
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            # Drain the final reduction before releasing dependent work.
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        if cutlass.const_expr(self.pdl_count is not None):
            griddepcontrol_launch_dependents()

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array
        (destination).

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

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = blackwell_helpers.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.gemm_output_layout,
            self.out_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )

        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tTR_rC: cute.Tensor,
        sC: cute.Tensor,
        tiled_copy_t2r: cute.TiledCopy,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Create tiled copy for register to shared memory (R2S).
        """
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.out_dtype,
        )

        tiled_copy_r2s = cute.make_tiled_copy_D(atom, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    @staticmethod
    def compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        cta_tile: cute.Tile,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        final_scale_dtype: Type[cutlass.Numeric],
        swap_ab: bool,
        num_smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param out_dtype: Data type of operand C (output).
        :type out_dtype: type[cutlass.Numeric]
        :param cta_tile: The CTA tile shape.
        :type cta_tile: cute.Tile
        :param sf_dtype: Data type of scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Vector size of scale factor.
        :type sf_vec_size: int
        :param swap_ab: Whether A/B and M/N roles are swapped.
        :type swap_ab: bool
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

        # Ping-pong the compact swapped C tile so reduction N can overlap
        # staging N+1. Doubling the larger unswapped tile costs occupancy.
        num_c_stage = 2 if swap_ab else 1

        # Default Tile info stages
        num_tile_stage = 2

        # Metadata loader pipeline depth (meta warp -> epilogue). Lets the loader
        # prefetch the per-row {token_idx, combined_scale} ahead of the epilogue.
        num_meta_stage = 2
        token_idx_bytes = cutlass.Int32.width // 8
        scale_bytes = final_scale_dtype.width // 8
        meta_rows = cta_tile[1] if swap_ab else cta_tile[0]
        meta_smem_bytes = (
            meta_rows * (token_idx_bytes + scale_bytes + scale_bytes) * num_meta_stage
        )

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

        # satisfy 16B alignment for the output tensor
        swizzled_pad = 16 // (out_dtype.width // 8)
        if swap_ab:
            c_smem_layout_staged_one = cute.make_layout(
                (cta_tile[0], cta_tile[1]),
                stride=(1, cta_tile[0] + swizzled_pad),
            )
        else:
            c_smem_layout_staged_one = cute.make_layout(
                (cta_tile[0], cta_tile[1]),
                stride=(cta_tile[1] + swizzled_pad, 1),
            )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        # Reserve one block for pipeline barriers and one for padding between
        # the independently 1024-byte-aligned operand and scale buffers.
        mbar_helpers_bytes = 2 * 1024

        c_bytes_per_stage = cute.size_in_bytes(out_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes, initial C stages bytes, and the per-row
        # metadata smem held by the loader warp
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (
            num_smem_capacity // occupancy
            - (mbar_helpers_bytes + c_bytes + meta_smem_bytes)
        ) // ab_bytes_per_stage

        return num_acc_stage, num_ab_stage, num_c_stage, num_tile_stage, num_meta_stage  # type: ignore[return-value]

    @staticmethod
    def compute_grid(
        gemm_shape: Tuple[int, int, int],
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        raster_along_m: bool,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size based on GEMM shape.

        :param gemm_shape: The GEMM computation shape (M, N, L)
        :type gemm_shape: tuple[int, int, int]
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr
        :param raster_along_m: Boolean, True to use raster along M.
        :type raster_along_m: bool

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        (m, n, l) = gemm_shape  # noqa: E741

        num_ctas_m = cute.ceil_div(m, cta_tile_shape_mnk[0])
        num_ctas_n = cute.ceil_div(n, cta_tile_shape_mnk[1])
        num_ctas_l = l

        num_ctas_mnl = (num_ctas_m, num_ctas_n, num_ctas_l)
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
    def is_valid_dtypes_and_scale_factor_vec_size(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        out_dtype: Type[cutlass.Numeric],
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
        :param out_dtype: The data type of the output tensor
        :type out_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        SUPPORTED_DTYPES = (
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        )
        if a_dtype not in SUPPORTED_DTYPES or b_dtype not in SUPPORTED_DTYPES:
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

        if out_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.BFloat16}:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        out_major: str,
        swap_ab: bool = False,
    ) -> bool:
        """
        Check if layouts and dtypes are valid combinations

        :param a_dtype: The data type of the A operand
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of the B operand
        :type b_dtype: Type[cutlass.Numeric]
        :param out_dtype: The data type of the output tensor
        :type out_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param out_major: The major dimension of the C tensor
        :type out_major: str
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
        if (out_major == "m") != swap_ab:
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

        :param use_2cta_instrs: Whether to use 2 CTA groups
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        mma_m = mma_tiler_mn[1] if swap_ab else mma_tiler_mn[0]
        mma_n = mma_tiler_mn[0] if swap_ab else mma_tiler_mn[1]

        if mma_m not in (128, 256):
            is_valid = False
        if swap_ab:
            if mma_n not in (8, 16, 32, 64, 128, 256):
                is_valid = False
            # SM100 2CTA MMA requires MMA N >= 16.
            if mma_m == 256 and mma_n < 16:
                is_valid = False
        elif mma_n not in (64, 128, 192, 256):
            is_valid = False
        if cluster_shape_mn[0] <= 0 or cluster_shape_mn[1] <= 0:
            return False

        # Skip illegal cluster shape
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

        if swap_ab and cluster_shape_mn[1] != 1:
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
        out_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        out_major: str,
        mma_tiler_mn: Tuple[int, int],
        swap_ab: bool = False,
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
        :param out_dtype: The data type of the output tensor
        :type out_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param out_major: The major axis of the C tensor
        :type out_major: str
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param swap_ab: Whether A/B (and MN) roles are swapped.
        :type swap_ab: bool

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        def check_contigous_128_alignment(dtype, is_mode0_major, shape):
            if dtype.width >= 8:
                return True
            return shape[0 if is_mode0_major else 1] % 128 == 0

        needs_unpack = (
            Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.needs_unpack_tma(
                a_dtype, b_dtype
            )
        )

        if (
            not check_contigous_16B_alignment(a_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(b_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(out_dtype, out_major == "m", (m, n, l))
        ):
            is_valid = False
        if needs_unpack and (
            not check_contigous_128_alignment(a_dtype, a_major == "m", (m, k, l))
            or not check_contigous_128_alignment(b_dtype, b_major == "n", (n, k, l))
        ):
            is_valid = False

        mma_m = mma_tiler_mn[1] if swap_ab else mma_tiler_mn[0]
        use_2cta_instrs = mma_m == 256
        cta_div = 2 if use_2cta_instrs else 1
        if (
            needs_unpack
            and a_major == "m"
            and a_dtype.width < 8
            and (mma_tiler_mn[0] // cta_div) % 128 != 0
        ):
            is_valid = False
        if (
            needs_unpack
            and b_major == "n"
            and b_dtype.width < 8
            and (mma_tiler_mn[1] // cta_div) % 128 != 0
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
        out_dtype: Type[cutlass.Numeric],
        final_scale_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        a_major: str,
        b_major: str,
        out_major: str,
        swap_ab: bool = False,
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
        :param out_dtype: The data type of the output tensor
        :type out_dtype: Type[cutlass.Numeric]
        :param final_scale_dtype: The data type of the router scales (token_final_scales)
        :type final_scale_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
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
        :param out_major: The major axis of the C tensor
        :type out_major: str
        :param swap_ab: Whether A/B (and MN) roles are swapped.
        :type swap_ab: bool
        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        if out_major != "n":
            can_implement = False
        # Skip unsupported types
        if not cls.is_valid_dtypes_and_scale_factor_vec_size(
            a_dtype, b_dtype, sf_dtype, sf_vec_size, out_dtype
        ):
            can_implement = False
        if a_dtype != b_dtype and n % 128 != 0:
            # Mixed E8M0 weight scales are described in complete 128-row
            # atoms even when the MMA N tile itself is 64 or 192.
            can_implement = False
        if swap_ab and out_dtype is cutlass.Float32 and mma_tiler_mn[0] == 256:
            # Two ping-pong FP32 C stages for a 256-row token tile consume the
            # SM100 shared-memory budget before any positive A/B stage fits.
            can_implement = False

        # Skip unsupported layouts
        mma_out_major = "m" if swap_ab else out_major
        if not cls.is_valid_layouts(
            a_dtype,
            b_dtype,
            out_dtype,
            a_major,
            b_major,
            mma_out_major,
            swap_ab,
        ):
            can_implement = False

        # Skip invalid mma tile shape and cluster shape
        if not cls.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn, swap_ab=swap_ab
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
            out_dtype,
            a_major,
            b_major,
            out_major,
            mma_tiler_mn,
            swap_ab,
        ):
            can_implement = False
        # Skip unsupported A/B layout
        if not (a_major == "k" and b_major == "k"):
            can_implement = False

        # Skip unsupported final scale dtype, only Float32 is supported
        if final_scale_dtype != cutlass.Float32:
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
        alpha_ptr: Optional[cute.Pointer],
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        permuted_idx_to_expanded_idx_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        token_final_scales_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        a_per_token_scale_ptr: Optional[cute.Pointer],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        num_tokens: cutlass.Int64,
        top_k: cutlass.Int64,
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: driver.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        num_tiles = m // tile_size
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2))
        )
        if cutlass.const_expr(not self.use_compact_sfb):
            # ((Atom_M, Rest_M), (Atom_K, Rest_K), 1)
            a_sf_layout = blockscaled_layout.tile_atom_to_shape_SF(
                (m, k, 1), scaling_vector_size
            )
        else:
            # (M, K // scaling_vector_size, 1)
            a_sf_layout = compact_sf_layout((m, k, 1), scaling_vector_size)
        a_sf = cute.make_tensor(
            a_sf_ptr,
            layout=a_sf_layout,
        )
        b_sf = cute.make_tensor(
            b_sf_ptr,
            layout=blockscaled_layout.tile_atom_to_shape_SF(
                (n, k, l), scaling_vector_size
            ),
        )
        output_rows = num_tokens if self.use_fused_finalize else num_tokens * top_k
        c = cute.make_tensor(
            c_ptr, layout=cute.make_ordered_layout((output_rows, n, 1), order=(1, 0, 2))
        )
        if cutlass.const_expr(self.apply_expert_alpha):
            alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((l,)))
        else:
            alpha = None

        tile_idx_to_group_idx = cute.make_tensor(
            tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        permuted_idx_to_expanded_idx = cute.make_tensor(
            permuted_idx_to_expanded_idx_ptr, layout=cute.make_layout((m,))
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )
        token_final_scales = cute.make_tensor(
            token_final_scales_ptr,
            layout=cute.make_ordered_layout((num_tokens, top_k), order=(1, 0)),
        )
        # Per-expert down bias broadcast over the permuted-row (m) dimension.
        down_bias = cute.make_tensor(
            bias_ptr, layout=cute.make_layout((m, n, l), stride=(0, 1, n))
        )
        if cutlass.const_expr(self.use_a_per_token_scale):
            a_per_token_scale = cute.make_tensor(
                a_per_token_scale_ptr, layout=cute.make_layout((m,))
            )
        else:
            a_per_token_scale = None

        return self(
            a,
            b,
            c,
            a_sf,
            b_sf,
            tile_idx_to_group_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            max_active_clusters=max_active_clusters,
            stream=stream,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            token_final_scales=token_final_scales,
            a_per_token_scale=a_per_token_scale,
            down_bias=down_bias,
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


def create_finalize_routing_tensors(
    num_tokens: int,
    topk: int,
    num_experts: int,
    tile_m: int,
    device: torch.device,
    use_compact_sfb: bool,
):
    """Create deterministic expert-grouped routing metadata for the example."""
    assignments: list[list[int]] = [[] for _ in range(num_experts)]
    for token in range(num_tokens):
        for rank in range(topk):
            expanded_idx = token * topk + rank
            assignments[expanded_idx % num_experts].append(expanded_idx)

    # Only the block-scaled MMA layout requires complete 128-row atoms.
    group_alignment = tile_m if use_compact_sfb else math.lcm(tile_m, 128)
    aligned = [
        ((len(group) + group_alignment - 1) // group_alignment) * group_alignment
        for group in assignments
    ]
    permuted_m = sum(aligned)
    permuted_idx = torch.full((permuted_m,), -1, dtype=torch.int32, device=device)
    row_experts = torch.full((permuted_m,), -1, dtype=torch.int64, device=device)
    tile_experts, tile_limits = [], []
    offset = 0
    for expert, expanded_indices in enumerate(assignments):
        count = len(expanded_indices)
        if count:
            permuted_idx[offset : offset + count] = torch.tensor(
                expanded_indices, dtype=torch.int32, device=device
            )
            row_experts[offset : offset + count] = expert
        for tile in range(aligned[expert] // tile_m):
            tile_experts.append(expert)
            tile_limits.append(offset + min((tile + 1) * tile_m, count))
        offset += aligned[expert]

    return (
        permuted_idx,
        row_experts,
        torch.tensor(tile_experts, dtype=torch.int32, device=device),
        torch.tensor(tile_limits, dtype=torch.int32, device=device),
        torch.tensor([len(tile_experts)], dtype=torch.int32, device=device),
    )


def create_finalize_tensors_ab(
    permuted_m: int,
    num_experts: int,
    n: int,
    k: int,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    swap_ab: bool,
    init_normal: bool,
    normal_mean: float,
    normal_std: float,
    device: torch.device,
    generator: torch.Generator,
    use_compact_sfb: bool = True,
):
    """Create finalize operands with the activation scales expected by the kernel."""
    from .blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
        convert_sf_to_mma_layout,
        quantize_operand,
    )

    if init_normal:
        a_source = torch.normal(
            normal_mean,
            normal_std,
            size=(permuted_m, k),
            generator=generator,
            device=device,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        b_source = torch.normal(
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
            (permuted_m, k),
            generator=generator,
            device=device,
            dtype=torch.int32,
        ).to(torch.bfloat16)
        b_source = torch.randint(
            -2,
            3,
            (num_experts, n, k),
            generator=generator,
            device=device,
            dtype=torch.int32,
        ).to(torch.bfloat16)

    compact_sfb = swap_ab and use_compact_sfb
    a, a_scale_swizzled, a_ref = quantize_operand(
        a_source,
        a_dtype,
        sf_dtype,
        sf_vec_size,
        a_dtype is cutlass.Float4E2M1FN and not compact_sfb,
    )
    b_flat, b_scale_swizzled, b_ref_flat = quantize_operand(
        b_source.view(num_experts * n, k),
        b_dtype,
        sf_dtype,
        sf_vec_size,
        True,
    )
    if n % 128:
        padded_n = cute.ceil_div(n, 128) * 128
        padded_scale_k = cute.ceil_div(k // sf_vec_size, 4) * 4
        grouped_b_scale = b_scale_swizzled.new_empty(
            (num_experts, padded_n, padded_scale_k)
        )
        grouped_b_scale[:] = b_scale_swizzled[0, 0]
        grouped_b_scale[:, :n] = b_scale_swizzled[: num_experts * n].view(
            num_experts, n, padded_scale_k
        )
        b_scale_swizzled = grouped_b_scale
    if not compact_sfb:
        a_scale = convert_sf_to_mma_layout(
            a_scale_swizzled,
            m=permuted_m,
            k=k,
            sf_vec_size=sf_vec_size,
        )
    else:
        scale_k = k // sf_vec_size
        a_scale = torch.zeros(
            (permuted_m, ((scale_k + 15) // 16) * 16),
            dtype=torch.uint8,
            device=device,
        )
        a_scale[:, :scale_k] = a_scale_swizzled.view(torch.uint8).reshape(
            permuted_m, scale_k
        )
    b_scale = convert_sf_to_mma_layout(
        b_scale_swizzled,
        m=n,
        k=k,
        num_groups=num_experts,
        sf_vec_size=sf_vec_size,
    )
    return (
        a,
        b_flat.view(num_experts, n, -1),
        a_scale,
        b_scale,
        a_ref,
        b_ref_flat.view(num_experts, n, k),
    )


def compute_finalize_reference(
    a_ref: torch.Tensor,
    b_ref: torch.Tensor,
    permuted_idx: torch.Tensor,
    row_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    alpha: Optional[torch.Tensor],
    down_bias: torch.Tensor,
    num_tokens: int,
    topk: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute the grouped GEMM, scale, bias, and scatter-reduce reference."""
    reference = torch.zeros(
        (num_tokens, b_ref.shape[1]), dtype=out_dtype, device=a_ref.device
    )
    for expert in range(b_ref.shape[0]):
        rows = (permuted_idx >= 0) & (row_experts == expert)
        if not rows.any():
            continue
        expanded = permuted_idx[rows].long()
        tokens = expanded // topk
        ranks = expanded % topk
        contribution = a_ref[rows] @ b_ref[expert].T
        if alpha is not None:
            contribution *= alpha[expert]
        contribution += down_bias[expert]
        contribution *= token_final_scales[tokens, ranks, None]
        # The kernel converts each expert contribution before its atomic
        # scatter-reduce, so mirror that rounding point in the reference.
        reference.index_add_(0, tokens, contribution.to(out_dtype))
    return reference


def run(
    num_tokens: int,
    hidden_dim: int,
    output_dim: int,
    num_experts: int,
    topk: int,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    out_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    out_major: str,
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
    raster_along_m: bool = False,
    pdl_count: int = -1,
    expert_alpha_enabled: bool = True,
    expert_alpha_value: float = 1.0,
    bias_enabled: bool = False,
    bias_value: Optional[float] = None,
    router_scale_value: Optional[float] = None,
    seed: int = 2025,
    use_compact_sfb: bool = True,
    **kwargs,
):
    """Run and benchmark the SM100 grouped GEMM finalize-fusion example."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    positive = {
        "num_tokens": num_tokens,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "num_experts": num_experts,
        "topk": topk,
        "tolerance": tolerance,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError(f"expected positive values, got {positive}")
    if topk > num_experts:
        raise ValueError(f"topk={topk} must not exceed num_experts={num_experts}")
    if normal_std < 0:
        raise ValueError("normal_std must be non-negative")
    if warmup_iterations < 0 or iterations <= 0:
        raise ValueError("warmup_iterations must be >= 0 and iterations must be > 0")
    if pdl_count < -1:
        raise ValueError("pdl_count must be -1 or a non-negative K-tile index")
    if len(mma_tiler_mn) != 2 or len(cluster_shape_mn) != 2:
        raise ValueError(
            "mma_tiler_mn and cluster_shape_mn must each contain exactly 2 values"
        )
    nvfp4_with_ue8m0_scales = (
        a_dtype is cutlass.Float4E2M1FN
        and b_dtype is cutlass.Float4E2M1FN
        and sf_dtype is cutlass.Float8E8M0FNU
    )
    if (
        sf_vec_size == 16
        and sf_dtype is not cutlass.Float8E4M3FN
        and not nvfp4_with_ue8m0_scales
    ):
        raise ValueError("sf_vec_size=16 requires E4M3 scale factors")
    if sf_vec_size == 32 and sf_dtype is not cutlass.Float8E8M0FNU:
        raise ValueError("sf_vec_size=32 requires UE8M0 scale factors")

    device = torch.device("cuda")
    m_tile = mma_tiler_mn[0]
    n, k = output_dim, hidden_dim
    if k % sf_vec_size or (k // sf_vec_size) % 4:
        raise ValueError(
            f"hidden_dim={k} must provide a multiple of four {sf_vec_size}-value "
            "scale blocks"
        )

    compact_sfb = swap_ab and use_compact_sfb
    routing = create_finalize_routing_tensors(
        num_tokens,
        topk,
        num_experts,
        m_tile,
        device,
        compact_sfb,
    )
    permuted_m = routing[0].numel()
    if not Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.can_implement(
        a_dtype,
        b_dtype,
        sf_dtype,
        sf_vec_size,
        out_dtype,
        cutlass.Float32,
        mma_tiler_mn,
        cluster_shape_mn,
        permuted_m,
        n,
        k,
        num_experts,
        a_major,
        b_major,
        out_major,
        swap_ab=swap_ab,
    ):
        raise testing.CantImplementError(
            f"Unsupported testcase {a_dtype}, {b_dtype}, {sf_dtype}, "
            f"{sf_vec_size}, {out_dtype}, {mma_tiler_mn}, {cluster_shape_mn}, "
            f"shape=({permuted_m},{n},{k},{num_experts}), {a_major}, {b_major}, "
            f"{out_major}, swap_ab={swap_ab}"
        )

    print("Running Blackwell Contiguous Grouped GEMM Finalize Fusion with:")
    print(f"Tokens: {num_tokens}, Experts: {num_experts}, TopK: {topk}")
    print(f"Permuted M: {permuted_m}, Output N: {n}, Hidden K: {k}")
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, SF dtype: {sf_dtype}, "
        f"SF Vec size: {sf_vec_size}, Output dtype: {out_dtype}"
    )
    print(
        f"MMA Tiler: {mma_tiler_mn}, Cluster Shape: {cluster_shape_mn}, "
        f"Swap AB: {swap_ab}"
    )
    print(
        f"Warmup iterations: {warmup_iterations}, Iterations: {iterations}, "
        f"Cold L2: {use_cold_l2}"
    )
    print(f"PDL count: {pdl_count} (-1 = release at kernel completion)")

    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    gmem = cute.AddressSpace.gmem
    current_stream = cutlass.torch.default_stream()

    def create_workspace(routing_tensors=None):
        if routing_tensors is None:
            routing_tensors = create_finalize_routing_tensors(
                num_tokens,
                topk,
                num_experts,
                m_tile,
                device,
                compact_sfb,
            )
        (
            permuted_idx,
            row_experts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            num_non_exiting,
        ) = routing_tensors

        generator = torch.Generator(device=device).manual_seed(seed)
        a, b, a_scale, b_scale, a_ref, b_ref = create_finalize_tensors_ab(
            permuted_m,
            num_experts,
            n,
            k,
            a_dtype,
            b_dtype,
            sf_dtype,
            sf_vec_size,
            swap_ab,
            init_normal,
            normal_mean,
            normal_std,
            device,
            generator,
            use_compact_sfb,
        )
        out = torch.zeros(
            (num_tokens, n),
            dtype=cutlass.torch.dtype(out_dtype),
            device=device,
        )
        alpha = (
            torch.full(
                (num_experts,),
                expert_alpha_value,
                dtype=torch.float32,
                device=device,
            )
            if expert_alpha_enabled
            else None
        )
        if bias_enabled and bias_value is None:
            down_bias = (
                torch.randn(
                    (num_experts, n), generator=generator, device=device
                ).float()
                * 0.01
            )
        else:
            down_bias = torch.full(
                (num_experts, n),
                bias_value if bias_enabled else 0.0,
                dtype=torch.float32,
                device=device,
            )
        if router_scale_value is None:
            token_final_scales = torch.rand(
                (num_tokens, topk),
                generator=generator,
                dtype=torch.float32,
                device=device,
            )
            token_final_scales /= token_final_scales.sum(dim=1, keepdim=True)
        else:
            token_final_scales = torch.full(
                (num_tokens, topk),
                router_scale_value,
                dtype=torch.float32,
                device=device,
            )

        a_ptr = make_ptr(a_dtype, a.data_ptr(), gmem, assumed_align=32)
        b_ptr = make_ptr(b_dtype, b.data_ptr(), gmem, assumed_align=32)
        a_sf_ptr = make_ptr(sf_dtype, a_scale.data_ptr(), gmem, assumed_align=16)
        b_sf_ptr = make_ptr(sf_dtype, b_scale.data_ptr(), gmem, assumed_align=16)
        out_ptr = make_ptr(out_dtype, out.data_ptr(), gmem, assumed_align=32)
        alpha_ptr = (
            make_ptr(cutlass.Float32, alpha.data_ptr(), gmem)
            if alpha is not None
            else None
        )
        tile_idx_ptr = make_ptr(cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), gmem)
        mn_limit_ptr = make_ptr(cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), gmem)
        permuted_idx_ptr = make_ptr(cutlass.Int32, permuted_idx.data_ptr(), gmem)
        num_tiles_ptr = make_ptr(cutlass.Int32, num_non_exiting.data_ptr(), gmem)
        final_scales_ptr = make_ptr(
            cutlass.Float32, token_final_scales.data_ptr(), gmem
        )
        bias_ptr = make_ptr(cutlass.Float32, down_bias.data_ptr(), gmem)

        jit_args = testing.JitArguments(
            a_ptr,
            b_ptr,
            a_sf_ptr,
            b_sf_ptr,
            out_ptr,
            alpha_ptr,
            tile_idx_ptr,
            mn_limit_ptr,
            permuted_idx_ptr,
            num_tiles_ptr,
            final_scales_ptr,
            bias_ptr,
            None,
            permuted_m,
            n,
            k,
            num_experts,
            num_tokens,
            topk,
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
            permuted_idx,
            num_non_exiting,
            token_final_scales,
            down_bias,
        ]
        jit_args.add_to_scope(
            [tensor for tensor in kernel_tensors if tensor is not None]
        )
        return jit_args, {
            "a_ref": a_ref,
            "b_ref": b_ref,
            "out": out,
            "alpha": alpha,
            "permuted_idx": permuted_idx,
            "row_experts": row_experts,
            "token_final_scales": token_final_scales,
            "down_bias": down_bias,
            "kernel_tensors": kernel_tensors,
        }

    initial_args, initial_tensors = create_workspace(routing)
    gemm = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        raster_along_m=raster_along_m,
        pdl_count=pdl_count,
        swap_ab=swap_ab,
        apply_expert_alpha=expert_alpha_enabled,
        use_compact_sfb=use_compact_sfb,
    )
    compiled = cute.compile(
        gemm.wrapper,
        *initial_args.args[:-1],
        tile_size=m_tile,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_active_clusters,
        stream=current_stream,
    )

    if not skip_ref_check:
        compiled(*initial_args.args, **initial_args.kwargs)
        torch.cuda.synchronize()
        print("Verifying results...")
        reference = compute_finalize_reference(
            initial_tensors["a_ref"],
            initial_tensors["b_ref"],
            initial_tensors["permuted_idx"],
            initial_tensors["row_experts"],
            initial_tensors["token_final_scales"],
            initial_tensors["alpha"],
            initial_tensors["down_bias"],
            num_tokens,
            topk,
            initial_tensors["out"].dtype,
        )
        actual_float = initial_tensors["out"].float()
        reference_float = reference.float()
        abs_diff = (actual_float - reference_float).abs()
        print(
            "Reference stats: "
            f"actual_std={actual_float.std().item():.6g}, "
            f"expected_std={reference_float.std().item():.6g}, "
            f"max_abs_diff={abs_diff.max().item():.6g}, "
            f"mean_abs_diff={abs_diff.mean().item():.6g}"
        )
        torch.testing.assert_close(
            actual_float,
            reference_float,
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
            if tensor is not None
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    runtime_s = exec_time / 1.0e6
    flop = 2 * num_tokens * topk * n * k
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
        description="Contiguous grouped blockscaled GEMM finalize fusion on Blackwell."
    )
    parser.add_argument("--num_tokens", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=7168, help="Hidden size K")
    parser.add_argument("--output_dim", type=int, default=2048, help="Output size N")
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="Logical token/weight tile shape",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
    )
    parser.add_argument("--a_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--b_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--out_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--a_major", choices=["k", "m"], default="k")
    parser.add_argument("--b_major", choices=["k", "n"], default="k")
    parser.add_argument("--out_major", choices=["n", "m"], default="n")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-01,
        help="Absolute tolerance for the scatter-reduced output reference",
    )
    parser.add_argument("--warmup_iterations", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--use_cold_l2", action="store_true")
    testing.add_tensor_init_args(parser, supports_int_dtypes=False)
    parser.add_argument("--swap_ab", action="store_true")
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
        "--expert_alpha_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--expert_alpha_value", type=float, default=1.0)
    parser.add_argument(
        "--bias_enabled", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--bias_value",
        type=float,
        default=None,
        help="Constant down bias; omit for deterministic generated bias",
    )
    parser.add_argument(
        "--router_scale_value",
        type=float,
        default=None,
        help="Constant router scale; omit for normalized random scales",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--use_compact_sfb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use compact swapped activation scales",
    )
    args = parser.parse_args()

    testing.validate_tensor_init_args(args, parser)
    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")
    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    run(
        args.num_tokens,
        args.hidden_dim,
        args.output_dim,
        args.num_experts,
        args.topk,
        args.a_dtype,
        args.b_dtype,
        args.sf_dtype,
        args.sf_vec_size,
        args.out_dtype,
        args.a_major,
        args.b_major,
        args.out_major,
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
        args.raster_along_m,
        args.pdl_count,
        args.expert_alpha_enabled,
        args.expert_alpha_value,
        args.bias_enabled,
        args.bias_value,
        args.router_scale_value,
        args.seed,
        args.use_compact_sfb,
    )
    print("PASS")
