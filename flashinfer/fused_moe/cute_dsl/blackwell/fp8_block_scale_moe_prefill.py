# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""FP8 block-scale MoE prefill pipeline for SM103 (CuTe DSL).

Specialized for one expert geometry: 512 local experts, top-10 softmax/renormalize
routing from bf16 logits, hidden 4096, intermediate 1024, SwiGLU, DeepSeek-FP8
block scales (activations 1x128, weights 128x128), bf16 output. The data path is
the one ``trtllm_fp8_block_scale_moe`` serves: GEMM1's output and the SwiGLU
output are each stored as FP8 E4M3 with one fp32 scale per (token, 128 columns),
``scale = max(amax / 448, FLT_MIN)``; accumulation is fp32.

Pipeline (all launches on the caller's current stream; workspace sized from the
token count only):

1. routing: softmax over the 512 logits, top-10, renormalize (one warp per token);
2. permutation into 128-row expert segments (rank, scan, row map, placement);
3. gather of the routed activation rows (compact rows for ``T <= 8192``);
4. GEMM1: persistent warp-specialized block-scaled grouped GEMM (tcgen05) whose
   epilogue quantizes each (row, 128-column) tile to FP8;
5. SwiGLU of the dequantized GEMM1 output, re-quantized to FP8 per (row, 128);
6. GEMM2: the same grouped GEMM with a bf16 epilogue;
7. top-k weighted combine into the bf16 output.

Each GEMM has two builds selected by the token count: for ``T <= 8192`` the
epilogues store 16-row blocks and skip blocks of padding rows (GEMM1 included);
above, 64-row blocks, GEMM1 storing every row and GEMM2 skipping padding blocks.
No kernel reads a padding row of either output.

Every CuTe DSL program is compiled once per device for a dynamic token count
(:func:`compile_kernels`); :func:`run` only looks compiled programs up, so it is
safe under CUDA-graph capture once the programs exist.
"""

import threading
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

import math


C_ROW_BLK = 16


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
                cutlass.Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class BlockwiseContiguousGroupedGemmKernel:
    """This class implements batched matrix multiplication (C = (SFA * A) * (SFB * B)) with support for fp8 (e4m3fn, e5m2)
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: Supported A/B data types:
        - Float8E4M3FN

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float16/BFloat16
        - Other data types are not supported for accuracy issues

    :note: Constraints:
        - MMA tiler M must be 64/128/256
        - MMA tiler N must be 128
        - Cluster shape M must be multiple of 2
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16

    Example:
        >>> gemm = BlockwiseContiguousGroupedGemmKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     use_2cta_instrs=True,
        ...     mma_tiler_mn=(128, 128),
        ...     cluster_shape_mn=(2, 2)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor, max_active_clusters, stream)
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        fuse_quant: bool = False,
        row_blk: int = 16,
        skip_pad_store: bool = True,
    ):
        """Initializes the configuration for a Blackwell blockwise dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Boolean indicating if the tcgen05 MMA variant
              with cta_group=2 should be used.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        """

        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        # fuse_quant: C is FP8, each (row, 128-col N tile) quantised with scale max(amax/448, FLT_MIN)
        self.fuse_quant = fuse_quant
        self.row_blk = row_blk
        # skip TMA stores of row blocks that are all padding (always for bf16 C; opt-in for FP8 C)
        self.skip_pad_store = skip_pad_store or not fuse_quant
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.use_newer_backend_codegen = cutlass.target_version(min_version="13.1")
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # Set specialized warp ids
        self.acc_update_warp_id = (0, 1, 2, 3)
        self.epilog_warp_id = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.scale_warp_id = 10
        self.sched_warp_id = 11
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.acc_update_warp_id,
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.scale_warp_id,
                self.sched_warp_id,
            )
        )
        self.threads_wo_sched = self.threads_per_warp * len(
            (
                *self.acc_update_warp_id,
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.scale_warp_id,
            )
        )
        self.num_regs_uniform_warps = 64
        self.num_regs_sched_warps = 64
        self.num_regs_epilogue_warps = 216
        self.num_regs_acc_update_warps = 216

        # Set barrier for epilogue sync and tmem ptr sync
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32
            * len((self.mma_warp_id, *self.epilog_warp_id, *self.acc_update_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_warp,
        )
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        # TMEM offset for final accumulator
        self.tmem_final_offset = 384

    def _setup_attributes(self):
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
        # Configure tiled mma
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.scale_granularity_m = 1
        self.scale_granularity_n = 128
        self.scale_granularity_k = 128
        self.scale_m_per_tile = self.cta_tile_shape_mnk[0] // self.scale_granularity_m
        self.scale_n_per_tile = self.cta_tile_shape_mnk[1] // self.scale_granularity_n
        self.scale_k_per_tile = self.cta_tile_shape_mnk[2] // self.scale_granularity_k

        if self.scale_k_per_tile != 1:
            raise ValueError("scale_k_per_tile must be 1")
        if self.scale_m_per_tile != self.cta_tile_shape_mnk[0]:
            raise ValueError("scale_m_per_tile must be cta_tile_m")
        if self.scale_n_per_tile != 1:
            raise ValueError("scale_n_per_tile must be 1")

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # Setup A/B/C/Scale stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_scale_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sfa_dtype,
            self.sfb_dtype,
            self.scale_m_per_tile * self.scale_k_per_tile,
            self.scale_n_per_tile * self.scale_k_per_tile,
            self.num_smem_capacity,
            self.occupancy,
        )

        # Compute A/B/C/Scale shared memory layout
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
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )
        self.sfa_smem_layout_staged = cute.make_layout(
            (
                (self.scale_granularity_m, self.scale_m_per_tile),
                (self.scale_granularity_k, self.scale_k_per_tile),
                self.num_scale_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_m_per_tile,
            ),
        )
        self.sfb_smem_layout_staged = cute.make_layout(
            (
                (self.scale_granularity_n, self.scale_n_per_tile),
                (self.scale_granularity_k, self.scale_k_per_tile),
                self.num_scale_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_n_per_tile,
            ),
        )

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = 512

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        gidx_mapping: cute.Tensor,
        sco: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
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
        :param c: Output tensor C
        :type c: cute.Tensor
        :param sfa: Scale factor tensor A
        :type sfa: cute.Tensor
        :param sfb: Scale factor tensor B
        :type sfb: cute.Tensor
        :param gidx_mapping: Mapping from m index to group index
        :type gidx_mapping: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.sfa_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.sfb_dtype: Type[cutlass.Numeric] = sfb.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = self._get_tma_atom_kind(atom_thr_size, self.is_a_mcast)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if a.element_type is cutlass.Float32 else None
            ),
        )

        # Setup TMA load for B
        b_op = self._get_tma_atom_kind(atom_thr_size, self.is_b_mcast)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if b.element_type is cutlass.Float32 else None
            ),
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        c_cta_v_layout = cute.composition(
            cute.make_identity_layout(c.shape), self.epi_tile
        )
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout,
            c_cta_v_layout,
        )
        # 32-row box store used for tiles that end in padding rows: only blocks holding real rows
        # are written back
        epi_tile_blk = (self.row_blk, cute.size(self.epi_tile[1]))
        epi_smem_layout_blk = cute.slice_(
            sm100_utils.make_smem_layout_epi(
                self.c_dtype, self.c_layout, epi_tile_blk, 1
            ),
            (None, None, 0),
        )
        tma_atom_cb, tma_tensor_cb = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout_blk,
            cute.composition(cute.make_identity_layout(c.shape), epi_tile_blk),
        )

        tensor_sfa = cute.make_tensor(
            sfa.iterator,
            cute.make_layout(
                (
                    (self.scale_granularity_m, sfa.shape[0]),
                    (self.scale_granularity_k, sfa.shape[1]),
                    sfa.shape[2],
                ),
                stride=(
                    (0, sfa.layout.stride[0]),
                    (0, sfa.layout.stride[1]),
                    sfa.layout.stride[2],
                ),
            ),
        )
        tensor_sfb = cute.make_tensor(
            sfb.iterator,
            cute.make_layout(
                (
                    (self.scale_granularity_n, sfb.shape[0]),
                    (self.scale_granularity_k, sfb.shape[1]),
                    sfb.shape[2],
                ),
                stride=(
                    (0, sfb.layout.stride[0]),
                    (0, sfb.layout.stride[1]),
                    sfb.layout.stride[2],
                ),
            ),
        )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        self.buffer_align_bytes = 1024

        c_smem_size = cute.cosize(self.c_smem_layout_staged.outer)

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            # (bidx, bidy, bidz, valid)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 4 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            scale_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_scale_stage * 2
            ]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            epi_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1 * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    c_smem_size,
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sfa_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sfb_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tma_atom_cb,
            tma_tensor_cb,
            tensor_sfa,
            tensor_sfb,
            gidx_mapping,
            sco,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tma_atom_cb: cute.CopyAtom,
        mCb_mnl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        mSFB_nkl: cute.Tensor,
        gidx_mapping: cute.Tensor,
        sco: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

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
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize mainloop scale_pipeline (barrier) and states
        scale_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        scale_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.epilog_warp_id),
        )
        scale_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.scale_mbar_ptr.data_ptr(),
            num_stages=self.num_scale_stage,
            producer_group=scale_pipeline_producer_group,
            consumer_group=scale_pipeline_consumer_group,
            defer_sync=True,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize epilogue pipeline (barrier) and states
        epi_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.acc_update_warp_id),
        )
        epi_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.epilog_warp_id),
        )
        epi_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.epi_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=epi_pipeline_producer_group,
            consumer_group=epi_pipeline_consumer_group,
            defer_sync=True,
        )

        # Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
            defer_sync=True,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/C/Scale
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
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
        # (bidx, bidy, bidz, valid)
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        sInfo = storage.sInfo.get_tensor(info_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
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
        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        # (bM, bK, loopM, loopK, loopL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.cta_tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # coordinate
        cSFA_mkl = cute.make_identity_tensor(cute.shape(mSFA_mkl))
        cSFB_nkl = cute.make_identity_tensor(cute.shape(mSFB_nkl))
        # (bM, bK, loopM, loopK, loopL)
        cSFA = cute.local_tile(
            cSFA_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        cSFB = cute.local_tile(
            cSFB_nkl,
            cute.slice_(self.cta_tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCgCb = thr_mma.partition_C(
            cute.local_tile(
                mCb_mnl,
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
        )

        # scale viewed as C tensor
        sSFA_view_as_C_layout = cute.make_layout(
            (
                (self.scale_granularity_m, self.scale_m_per_tile),
                self.cta_tile_shape_mnk[1],
                self.num_scale_stage,
            ),
            stride=((0, 1), 0, self.scale_m_per_tile),
        )
        sSFB_view_as_C_layout = cute.make_layout(
            (
                self.cta_tile_shape_mnk[0],
                (self.scale_granularity_n, self.scale_n_per_tile),
                self.num_scale_stage,
            ),
            stride=(0, (0, 1), self.scale_n_per_tile),
        )
        sSFA_view_as_C = cute.make_tensor(sSFA.iterator, sSFA_view_as_C_layout)
        sSFB_view_as_C = cute.make_tensor(sSFB.iterator, sSFB_view_as_C_layout)

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

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # load scaleA/scaleB
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mSFA_mkl.element_type,
            num_bits_per_copy=mSFA_mkl.element_type.width,
        )
        tiled_copy_sfa = cute.make_tiled_copy_tv(
            atom_copy, cute.make_layout((32,)), cute.make_layout((1,))
        )
        tiled_copy_sfb = cute.make_tiled_copy_tv(
            atom_copy, cute.make_layout((32,)), cute.make_layout((1,))
        )
        thr_copy_sfa = tiled_copy_sfa.get_slice(lane_idx)
        thr_copy_sfb = tiled_copy_sfb.get_slice(lane_idx)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tAgSFA_mkl = thr_copy_sfa.partition_S(gSFA_mkl)  # noqa: F841
        tAsSFA = thr_copy_sfa.partition_D(sSFA)
        tAcSFA = thr_copy_sfa.partition_S(cSFA)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopN, loopK, loopL)
        tBgSFB_nkl = thr_copy_sfb.partition_S(gSFB_nkl)
        tBsSFB = thr_copy_sfb.partition_D(sSFB)
        tBcSFB = thr_copy_sfb.partition_S(cSFB)

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
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Specialized Schedule warp
        #
        if warp_idx == self.sched_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_sched_warps)
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            sched_valid = cutlass.Boolean(work_tile.is_valid_tile)
            if sched_valid:
                if gidx_mapping[work_tile.tile_idx[0] * self.cta_tile_shape_mnk[0]] < 0:
                    sched_valid = cutlass.Boolean(False)
            while sched_valid:
                # query next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                # acquire tile info pipeline
                tile_info_pipeline.producer_acquire(tile_info_producer_state)

                # get the group info
                cur_tile_coord = work_tile.tile_idx
                gidx = 0
                valid_i = cutlass.Int32(0)
                if work_tile.is_valid_tile:
                    gidx = gidx_mapping[cur_tile_coord[0] * self.cta_tile_shape_mnk[0]]
                    if gidx > 0:
                        gidx = gidx & 1023
                    if gidx >= 0:
                        valid_i = cutlass.Int32(1)
                sched_valid = valid_i == 1

                # store the tile info
                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                    sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                    sInfo[(2, tile_info_producer_state.index)] = gidx
                    sInfo[(3, tile_info_producer_state.index)] = valid_i

                # fence view async shared
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                self.sched_sync_barrier.arrive_and_wait()
                # commit tile info pipeline
                tile_info_pipeline.producer_commit(tile_info_producer_state)

                # advance to next tile
                tile_info_producer_state.advance()

            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            # Get tile coord from tile scheduler
            cur_tile_coord = work_tile.tile_idx
            gidx = 0
            if work_tile.is_valid_tile:
                gidx = gidx_mapping[cur_tile_coord[0] * self.cta_tile_shape_mnk[0]]
                if gidx > 0:
                    gidx = gidx & 1023
            # initialize the tile info
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = gidx
            tile_info[3] = work_tile.is_valid_tile
            if gidx < 0:
                tile_info[3] = cutlass.Int32(0)

            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

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
                # GEMM1 A rows are stored compactly: shift the tile's rows by the expert offset
                a_code = gidx_mapping[tile_info[0] * self.cta_tile_shape_mnk[0]]
                a_shift = cutlass.Int32(0) - (a_code >> 10)
                gA_s = cute.local_tile(
                    cute.domain_offset((a_shift, 0, 0), mA_mkl),
                    cute.slice_(self.mma_tiler, (None, 0, None)),
                    (None, None, None),
                )
                tCgA_s = thr_mma.partition_A(gA_s)
                _tAsA_s, tAgA_s = cpasync.tma_partition(
                    tma_atom_a,
                    block_in_cluster_coord_vmnk[2],
                    a_cta_layout,
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(tCgA_s, 0, 3),
                )
                tAgA_slice = tAgA_s[(None, mma_tile_coord_mnl[0], None, 0)]
                # ((atom_v, rest_v), loopK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                    tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                    tBgB_k = tBgB_slice[(None, ab_producer_state.count)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]

                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)

                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
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
            ab_pipeline.producer_tail(ab_producer_state)
        #
        # Specialized Scale load warp
        #
        if warp_idx == self.scale_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            scale_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_scale_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            # Get tile coord from tile scheduler
            cur_tile_coord = work_tile.tile_idx
            gidx = 0
            if work_tile.is_valid_tile:
                gidx = gidx_mapping[cur_tile_coord[0] * self.cta_tile_shape_mnk[0]]
                if gidx > 0:
                    gidx = gidx & 1023
            # initialize the tile info
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = gidx
            tile_info[3] = work_tile.is_valid_tile
            if gidx < 0:
                tile_info[3] = cutlass.Int32(0)

            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                #
                # Prepare the mask for scaleA/scaleB
                #
                tApSFA = cute.make_rmem_tensor(
                    cute.make_layout(
                        cute.filter_zeros(
                            cute.slice_(tAsSFA, (None, None, None, 0))
                        ).shape
                    ),
                    cutlass.Boolean,
                )
                tBpSFB = cute.make_rmem_tensor(
                    cute.make_layout(
                        cute.filter_zeros(
                            cute.slice_(tBsSFB, (None, None, None, 0))
                        ).shape
                    ),
                    cutlass.Boolean,
                )

                s_code = gidx_mapping[tile_info[0] * self.cta_tile_shape_mnk[0]]
                s_shift = cutlass.Int32(0) - (s_code >> 10)
                gSFA_s = cute.local_tile(
                    cute.domain_offset((s_shift, 0, 0), mSFA_mkl),
                    cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
                    (None, None, None),
                )
                tAgSFA_s = thr_copy_sfa.partition_S(gSFA_s)

                # Peek (try_wait) SCALE buffer empty
                scale_producer_state.reset_count()
                peek_scale_empty_status = cutlass.Boolean(1)
                if scale_producer_state.count < k_tile_cnt:
                    peek_scale_empty_status = scale_pipeline.producer_try_acquire(
                        scale_producer_state
                    )

                #
                # load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                    #
                    # Slice to per mma tile index
                    #
                    tAsSFA_pipe = cute.filter_zeros(
                        tAsSFA[(None, None, None, scale_producer_state.index)]
                    )
                    tBsSFB_pipe = cute.filter_zeros(
                        tBsSFB[(None, None, None, scale_producer_state.index)]
                    )
                    tAgSFA_k = cute.filter_zeros(
                        tAgSFA_s[
                            (
                                None,
                                None,
                                None,
                                tile_info[0],
                                scale_producer_state.count,
                                0,
                            )
                        ]
                    )
                    tBgSFB_k = cute.filter_zeros(
                        tBgSFB_nkl[
                            (
                                None,
                                None,
                                None,
                                tile_info[1],
                                scale_producer_state.count,
                                tile_info[2],
                            )
                        ]
                    )

                    tAcSFA_compact = cute.filter_zeros(
                        cute.slice_(
                            tAcSFA,
                            (
                                None,
                                None,
                                None,
                                tile_info[0],
                                scale_producer_state.count,
                                0,
                            ),
                        )
                    )
                    tBcSFB_compact = cute.filter_zeros(
                        cute.slice_(
                            tBcSFB,
                            (
                                None,
                                None,
                                None,
                                tile_info[1],
                                scale_producer_state.count,
                                tile_info[2],
                            ),
                        )
                    )
                    for i in cutlass.range_constexpr(cute.size(tApSFA, mode=[1])):
                        tApSFA[((0, 0), i, (0, 0))] = cute.elem_less(
                            tAcSFA_compact[(i)][0], mSFA_mkl.shape[0]
                        )
                    for i in cutlass.range_constexpr(cute.size(tBpSFB, mode=[1])):
                        tBpSFB[((0, 0), i, (0, 0))] = cute.elem_less(
                            tBcSFB_compact[(i)][0], mSFB_nkl.shape[0]
                        )

                    # Conditionally wait for Scale buffer empty
                    scale_pipeline.producer_acquire(
                        scale_producer_state, peek_scale_empty_status
                    )

                    # load scaleA/scaleB
                    cute.copy(tiled_copy_sfa, tAgSFA_k, tAsSFA_pipe, pred=tApSFA)
                    if cutlass.const_expr(self.fuse_quant):
                        if lane_idx == 0:
                            cute.copy(
                                tiled_copy_sfb, tBgSFB_k, tBsSFB_pipe, pred=tBpSFB
                            )
                    else:
                        cute.copy(tiled_copy_sfb, tBgSFB_k, tBsSFB_pipe, pred=tBpSFB)

                    scale_pipeline.producer_commit(scale_producer_state)

                    # Peek (try_wait) Scale buffer empty
                    scale_producer_state.advance()
                    peek_scale_empty_status = cutlass.Boolean(1)
                    if scale_producer_state.count < k_tile_cnt:
                        peek_scale_empty_status = scale_pipeline.producer_try_acquire(
                            scale_producer_state
                        )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            #
            # Wait Scale buffer empty
            #
            scale_pipeline.producer_tail(scale_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #
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

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            # Get tile coord from tile scheduler
            cur_tile_coord = work_tile.tile_idx
            # MMA warp don't care about gidx
            gidx = 0
            if work_tile.is_valid_tile:
                gidx = gidx_mapping[cur_tile_coord[0] * self.cta_tile_shape_mnk[0]]
                if gidx > 0:
                    gidx = gidx & 1023
            # initialize the tile info
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = gidx
            tile_info[3] = work_tile.is_valid_tile
            if gidx < 0:
                tile_info[3] = cutlass.Int32(0)

            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                # Peek (try_wait) Acc buffer empty for k_tile = 0
                acc_producer_state.reset_count()
                peek_acc_empty_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_acc_empty_status = acc_pipeline.producer_try_acquire(
                        acc_producer_state
                    )

                #
                # Mma mainloop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                    # Set tensor memory buffer for current tile
                    # (MMA, MMA_M, MMA_N)
                    tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                    #
                    # Wait for accumulator buffer empty
                    #
                    if is_leader_cta:
                        acc_pipeline.producer_acquire(
                            acc_producer_state, peek_acc_empty_status
                        )

                    #
                    # Reset the ACCUMULATE field for each tile
                    #
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                    #
                    # Async arrive accumulator buffer full(each kblock)
                    #
                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)

                    # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                    acc_producer_state.advance()
                    if acc_producer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_acc_empty_status = acc_pipeline.producer_try_acquire(
                                acc_producer_state
                            )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
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
        # Specialized acc update warps
        #
        if warp_idx <= self.acc_update_warp_id[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_acc_update_warps)
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
            tCtAcc_final = cute.make_tensor(
                tCtAcc_base.iterator + self.tmem_final_offset, tCtAcc_base.layout
            )

            #
            # Partition for epilogue
            #
            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tiled_copy_r2t,
                tTR_tAcc_base,
                tTR_rAcc,
                tTR_rAcc_final,
                tTR_sSFA,
                tTR_sSFB,
                tRT_rAcc,
                tRT_tAcc_base,
            ) = self.acc_update_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_base,
                tCtAcc_final,
                tCgC,
                sSFA_view_as_C,
                sSFB_view_as_C,
                epi_tile,
            )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            scale_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_scale_stage
            )

            epi_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            # Get tile coord from tile scheduler
            cur_tile_coord = work_tile.tile_idx
            # Acc update warp don't care about gidx
            gidx = 0
            if work_tile.is_valid_tile:
                gidx = gidx_mapping[cur_tile_coord[0] * self.cta_tile_shape_mnk[0]]
                if gidx > 0:
                    gidx = gidx & 1023
            # initialize the tile info
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = gidx
            tile_info[3] = work_tile.is_valid_tile
            if gidx < 0:
                tile_info[3] = cutlass.Int32(0)

            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                # initialize the final accumulator
                tTR_rAcc_final.fill(0.0)

                tTR_rSFA = cute.make_rmem_tensor(  # noqa: F841
                    cute.slice_(tTR_sSFA, (None, None, None, 0, None, 0)).shape,
                    self.acc_dtype,
                )
                tTR_rSFB = cute.make_rmem_tensor(  # noqa: F841
                    cute.slice_(tTR_sSFB, (None, None, None, 0, None, 0)).shape,
                    self.acc_dtype,
                )

                scale_consumer_state.reset_count()
                peek_scale_full_status = cutlass.Boolean(1)
                if scale_consumer_state.count < k_tile_cnt:
                    peek_scale_full_status = scale_pipeline.consumer_try_wait(
                        scale_consumer_state
                    )

                acc_consumer_state.reset_count()
                peek_acc_full_status = cutlass.Boolean(1)
                if acc_consumer_state.count < k_tile_cnt:
                    peek_acc_full_status = acc_pipeline.consumer_try_wait(
                        acc_consumer_state
                    )

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                    # Set tensor memory buffer for current tile
                    # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, acc_consumer_state.index)
                    ]

                    #
                    # Wait for scale buffer full
                    #
                    scale_pipeline.consumer_wait(
                        scale_consumer_state, peek_scale_full_status
                    )

                    tTR_sSFA_slice = cute.slice_(
                        tTR_sSFA,
                        (None, None, None, 0, None, scale_consumer_state.index),
                    )
                    tTR_sSFB_slice = cute.slice_(
                        tTR_sSFB,
                        (None, None, None, 0, None, scale_consumer_state.index),
                    )

                    # every element a thread owns lies in one row of one 128x128 scale block, so the
                    # combined scale is a single scalar per thread per k-tile
                    scale = tTR_sSFA_slice[0] * tTR_sSFB_slice[0]

                    #
                    # Wait for accumulator buffer full
                    #

                    acc_pipeline.consumer_wait(acc_consumer_state, peek_acc_full_status)

                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                        tTR_rAcc_subtile = tTR_rAcc_final[
                            (None, None, None, subtile_idx)
                        ]
                        tTR_rAcc_subtile.store(
                            tTR_rAcc.load() * scale + tTR_rAcc_subtile.load()
                        )

                    #
                    # Async arrive accumulator buffer empty
                    #
                    scale_pipeline.consumer_release(scale_consumer_state)
                    scale_consumer_state.advance()

                    peek_scale_full_status = cutlass.Boolean(1)
                    if scale_consumer_state.count < k_tile_cnt:
                        peek_scale_full_status = scale_pipeline.consumer_try_wait(
                            scale_consumer_state
                        )
                    #
                    # Async arrive accumulator buffer empty
                    #
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                    peek_acc_full_status = cutlass.Boolean(1)
                    if acc_consumer_state.count < k_tile_cnt:
                        peek_acc_full_status = acc_pipeline.consumer_try_wait(
                            acc_consumer_state
                        )

                tRT_tAcc = tRT_tAcc_base[(None, None, None, None, None, 0)]
                tRT_tAcc = cute.group_modes(tRT_tAcc, 3, cute.rank(tRT_tAcc))

                #
                # Wait for epilogue buffer empty
                #
                epi_pipeline.producer_acquire(epi_producer_state)

                # copy the accumulator to tensor memory buffer
                cute.copy(tiled_copy_r2t, tTR_rAcc_final, tRT_tAcc)
                cute.arch.fence_view_async_tmem_store()

                #
                # Async arrive epilogue buffer full
                #
                epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

        #
        # Specialized epilogue warps
        #
        if warp_idx <= self.epilog_warp_id[-1] and warp_idx >= self.epilog_warp_id[0]:
            cute.arch.setmaxregister_increase(self.num_regs_epilogue_warps)
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
            tCtAcc_base_ = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAcc_final = cute.make_tensor(
                tCtAcc_base_.iterator + self.tmem_final_offset, tCtAcc_base_.layout
            )

            #
            # Partition for epilogue
            #
            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_final, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = None
            tiled_copy_r2s = None
            simt_atom = None  # noqa: F841
            tRS_rC = None
            tRS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC, epi_tile, sC
            )
            blk_tile = (self.row_blk, cute.size(epi_tile[1]))
            # ((ATOM_V, REST_V), ROW_BLK, 1, STAGE) / ((ATOM_V, REST_V), ROW_BLK, EPI_N, loopM, loopN, loopL)
            bSG_sCb, bSG_gCb_partitioned = cpasync.tma_partition(
                tma_atom_cb,
                0,
                cute.make_layout(1),
                cute.group_modes(cute.flat_divide(sC, blk_tile), 0, 2),
                cute.group_modes(
                    cute.flat_divide(
                        tCgCb[((None, None), 0, 0, None, None, None)], blk_tile
                    ),
                    0,
                    2,
                ),
            )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            epi_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )

            c_pipeline = None
            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            # Get tile coord from tile scheduler
            cur_tile_coord = work_tile.tile_idx
            # Epilogue warp don't care about gidx
            gidx = 0
            if work_tile.is_valid_tile:
                gidx = gidx_mapping[cur_tile_coord[0] * self.cta_tile_shape_mnk[0]]
                if gidx > 0:
                    gidx = gidx & 1023
            # initialize the tile info
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = gidx
            tile_info[3] = work_tile.is_valid_tile
            if gidx < 0:
                tile_info[3] = cutlass.Int32(0)

            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            num_prev_subtiles = cutlass.Int32(0)

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Slice to per mma tile index
                #
                bSG_gC = None
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        mma_tile_coord_mnl[0],
                        mma_tile_coord_mnl[1],
                        0,
                    )
                ]

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, epi_consumer_state.index)
                ]

                n_row_blk = self.cta_tile_shape_mnk[0] // self.row_blk
                m_base = tile_info[0] * self.cta_tile_shape_mnk[0]
                tile_full = True
                blkv = cute.make_rmem_tensor((n_row_blk,), cutlass.Int32)
                if cutlass.const_expr(self.skip_pad_store):
                    # gidx < 0 marks padding rows; a 32-row block is stored iff its first row is real.
                    # Loaded before the accumulator wait so the latency overlaps it.
                    for rb in cutlass.range_constexpr(n_row_blk):
                        blkv[rb] = gidx_mapping[m_base + rb * self.row_blk]
                    last_row = gidx_mapping[m_base + self.cta_tile_shape_mnk[0] - 1]
                    tile_full = last_row >= 0

                #
                # Wait for accumulator buffer full
                #
                epi_pipeline.consumer_wait(epi_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                q_inv = cutlass.Float32(1.0)
                if cutlass.const_expr(self.fuse_quant):
                    # pass 1: per-row amax over the 128-column tile (thread == TMEM lane == row)
                    amax = cutlass.Float32(0.0)
                    for subtile_idx in cutlass.range(subtile_cnt):
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[(None, None, None, subtile_idx)],
                            tTR_rAcc,
                        )
                        acc_values = tTR_rAcc.load()
                        abs_values_ir = mlir_math.absf(acc_values.ir_value())
                        abs_values = type(acc_values)(
                            abs_values_ir, acc_values.shape, acc_values.dtype
                        )
                        subtile_amax = abs_values.reduce(
                            cute.ReductionOp.MAX, cutlass.Float32(0.0), 0
                        )
                        amax = cute.arch.fmax(amax, subtile_amax)
                    q_scale = cute.arch.fmax(
                        amax / 448.0, cutlass.Float32(1.1754943508222875e-38)
                    )
                    q_inv = cutlass.Float32(1.0) / q_scale
                    sco[
                        tile_info[1],
                        tile_info[0] * self.cta_tile_shape_mnk[0] + epi_tidx,
                    ] = q_scale
                for subtile_idx in cutlass.range(subtile_cnt):
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    #
                    # Convert to C type
                    #
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    if cutlass.const_expr(self.fuse_quant):
                        acc_vec = acc_vec * q_inv
                    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                    tRS_rC.store(acc_vec)

                    #
                    # Store C to shared memory
                    #
                    num_prev_subtiles = num_prev_subtiles + 1
                    c_buffer = num_prev_subtiles % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    self.epilog_sync_barrier.arrive_and_wait()

                    #
                    # TMA store C to global memory
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        if tile_full:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, subtile_idx)],
                            )
                        else:
                            for rb in cutlass.range_constexpr(n_row_blk):
                                if blkv[rb] >= 0:
                                    cute.copy(
                                        tma_atom_cb,
                                        bSG_sCb[(None, rb, 0, c_buffer)],
                                        bSG_gCb_partitioned[
                                            (
                                                None,
                                                rb,
                                                subtile_idx,
                                                mma_tile_coord_mnl[0],
                                                mma_tile_coord_mnl[1],
                                                0,
                                            )
                                        ],
                                    )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                epi_pipeline.consumer_release(epi_consumer_state)
                epi_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
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
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

    def acc_update_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        tAcc_final: cute.Tensor,
        gC_mnl: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        epi_tile: cute.Tile,
    ) -> Tuple[
        cute.TiledCopy,
        cute.TiledCopy,
        cute.Tensor,
        cute.Tensor,
        cute.Tensor,
        cute.Tensor,
        cute.Tensor,
        cute.Tensor,
        cute.Tensor,
    ]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).
        Make tiledCopy for tensor memory store, then use it to partition register array (source) and tensor memory (destination).
        Partition the scale factor tensor for related copy operations.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param tAcc_final: The final accumulator tensor to be copied and partitioned
        :type tAcc_final: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param sSFA: The scale factor tensor for A
        :type sSFA: cute.Tensor
        :param sSFB: The scale factor tensor for B
        :type sSFB: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tiled_copy_r2t: The tiled copy operation for register to tmem copy(r2t)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
            - tTR_rAcc_final: The accumulated tensor in register used to hold all t2r results
            - tTR_sSFA: The partitioned tensor SFA by tiled_copy_t2r
            - tTR_sSFB: The partitioned tensor SFB by tiled_copy_t2r
            - tRT_rAcc_final: The accumulated tensor in register used to hold all r2t results
            - tRT_tAcc_final: The partitioned accumulator tensor by tiled_copy_r2t
        :rtype: Tuple[cute.TiledCopy, cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        tmem_load_atom = None
        tmem_store_atom = None
        if cutlass.const_expr(self.mma_tiler[0] == 64):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
                self.acc_dtype,
            )
        else:
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )
        if cutlass.const_expr(self.mma_tiler[0] == 64):
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(8)),
                self.acc_dtype,
            )
        else:
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )

        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tAcc_final_epi = cute.flat_divide(
            tAcc_final[((None, None), 0, 0, None)], epi_tile
        )

        tiled_copy_t2r = tcgen05.make_tmem_copy(
            tmem_load_atom, tAcc_epi[(None, None, 0, 0, 0)]
        )
        tiled_copy_r2t = tcgen05.make_tmem_copy(
            tmem_store_atom, tAcc_final_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        thr_copy_r2t = tiled_copy_r2t.get_slice(tidx)

        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        sSFA_epi = cute.flat_divide(sSFA, epi_tile)
        sSFB_epi = cute.flat_divide(sSFB, epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_sSFA = thr_copy_t2r.partition_D(sSFA_epi)
        tTR_sSFB = thr_copy_t2r.partition_D(sSFB_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_rAcc_final_ = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, None, None, 0, 0, 0)].shape, self.acc_dtype
        )
        tTR_rAcc_final = cute.group_modes(
            tTR_rAcc_final_, 3, cute.rank(tTR_rAcc_final_)
        )

        tRT_gC = thr_copy_r2t.partition_S(gC_mnl_epi)
        tRT_tAcc_final = thr_copy_r2t.partition_D(tAcc_final_epi)
        # (R2T, R2T_M, R2T_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tRT_rAcc_final_ = cute.make_rmem_tensor(
            tRT_gC[(None, None, None, None, None, 0, 0, 0)].shape, self.acc_dtype
        )
        # (R2T, R2T_M, R2T_N, (EPI_M, EPI_N))
        tRT_rAcc_final = cute.group_modes(
            tRT_rAcc_final_, 3, cute.rank(tRT_rAcc_final_)
        )

        return (
            tiled_copy_t2r,
            tiled_copy_r2t,
            tTR_tAcc,
            tTR_rAcc,
            tTR_rAcc_final,
            tTR_sSFA,
            tTR_sSFB,
            tRT_rAcc_final,
            tRT_tAcc_final,
        )

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

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
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
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
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

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
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
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
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sfa_dtype: Type[cutlass.Numeric],
        sfb_dtype: Type[cutlass.Numeric],
        sfa_count: int,
        sfb_count: int,
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
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout of operand C.
        :type c_layout: utils.LayoutEnum
        :param num_smem_capacity: Total available shared memory capacity in bytes.
        :type num_smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # Default ACC stages
        num_acc_stage = 3 if mma_tiler_mnk[0] / tiled_mma.thr_id.shape == 128 else 6

        # Default C stages
        num_c_stage = 2

        # Default ScaleA/B stages
        num_scale_stage = 10

        # Default Tile info stages
        num_tile_stage = 2

        # Calculate smem layout and size for one stage of A, B, and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        # 1024B alignment
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage
        sfa_bytes = sfa_count * (sfa_dtype.width // 8) * num_scale_stage
        sfb_bytes = sfb_count * (sfb_dtype.width // 8) * num_scale_stage
        scale_bytes = math.ceil((sfa_bytes + sfb_bytes) / 1024) * 1024

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (
            num_smem_capacity // occupancy
            - (mbar_helpers_bytes + c_bytes + scale_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            num_smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes + scale_bytes)
        ) // (occupancy * c_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_c_stage, num_scale_stage, num_tile_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
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
            num_ctas_mnl, cluster_shape_mnl, raster_along_m=False
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _get_tma_atom_kind(
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


# ---------------------------------------------------------------------------
# Auxiliary CuTe DSL kernels: routing, permutation, gather, SwiGLU + FP8
# re-quantization, top-k weighted combine.
# ---------------------------------------------------------------------------

_FP8_MAX = 448.0
_FLT_MIN = 1.1754943508222875e-38


@cute.kernel
def _act_kernel(
    c1: cute.Tensor,
    sc1: cute.Tensor,
    a2: cute.Tensor,
    s2: cute.Tensor,
    rstate: cute.Tensor,
):
    # c1 is GEMM1's FP8 output with scales sc1[n_block, row]. 4 rows per CTA, 64 threads per row;
    # 8 lanes cover one 128-col block, each lane owns 16 cols
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    row = bidx * 4 + tidx // 64
    q = tidx % 64
    j = q // 8
    st = rstate[row]
    if st == 0:
        fz = cute.make_rmem_tensor((16,), cutlass.Float8E4M3FN)
        fz.fill(0.0)
        cute.autovec_copy(fz, cute.local_tile(a2[row, None], (16,), (q,)))
        if q % 8 == 0:
            s2[j, row] = cutlass.Float32(0.0)
    if st > 0:
        crow = c1[row, None]
        fu8 = cute.make_rmem_tensor((16,), cutlass.Float8E4M3FN)
        fg8 = cute.make_rmem_tensor((16,), cutlass.Float8E4M3FN)
        cute.autovec_copy(cute.local_tile(crow, (16,), (q,)), fu8)
        cute.autovec_copy(cute.local_tile(crow, (16,), (q + 64,)), fg8)
        su = sc1[j, row]
        sg = sc1[j + 8, row]
        fu = cute.make_rmem_tensor((16,), cutlass.Float32)
        fg = cute.make_rmem_tensor((16,), cutlass.Float32)
        fu.store(fu8.load().to(cutlass.Float32) * su)
        fg.store(fg8.load().to(cutlass.Float32) * sg)
        ma = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(16):
            g = fg[i]
            d = cute.arch.fmin(
                cutlass.Float32(1.0) + cute.math.exp(-g), cutlass.Float32(3.0e38)
            )
            r0 = cute.arch.rcp_approx(d)
            r = _fma_rn(r0, _fma_rn(-d, r0, cutlass.Float32(1.0)), r0)
            q0 = g * r
            a = _fma_rn(_fma_rn(-q0, d, g), r, q0) * fu[i]
            fu[i] = a
            ma = cute.arch.fmax(ma, cute.math.absf(a))
        ma = cute.arch.warp_reduction_max(ma, threads_in_group=8)
        sa = cute.arch.fmax(ma / _FP8_MAX, cutlass.Float32(_FLT_MIN))
        isa = cutlass.Float32(1.0) / sa
        fq = cute.make_rmem_tensor((16,), cutlass.Float8E4M3FN)
        fq.store((fu.load() * isa).to(cutlass.Float8E4M3FN))
        cute.autovec_copy(fq, cute.local_tile(a2[row, None], (16,), (q,)))
        if q % 8 == 0:
            s2[j, row] = sa


@cute.jit
def _act_host(
    c1: cute.Tensor,
    sc1: cute.Tensor,
    a2: cute.Tensor,
    s2: cute.Tensor,
    rstate: cute.Tensor,
    stream: cuda.CUstream,
):
    _act_kernel(c1, sc1, a2, s2, rstate).launch(
        grid=(c1.shape[0] // 4, 1, 1), block=(256, 1, 1), stream=stream
    )


@cute.kernel
def _combine_kernel(
    c2: cute.Tensor, rows: cute.Tensor, wts: cute.Tensor, out: cute.Tensor
):
    tidx, _, _ = cute.arch.thread_idx()
    t, _, _ = cute.arch.block_idx()
    acc = cute.make_rmem_tensor((16,), cutlass.Float32)
    acc.fill(0.0)
    fr = cute.make_rmem_tensor((16,), cutlass.BFloat16)
    for k in cutlass.range_constexpr(10):
        r = rows[t, k]
        wk = wts[t, k]
        cute.autovec_copy(cute.local_tile(c2[r, None], (16,), (tidx,)), fr)
        acc.store(acc.load() + fr.load().to(cutlass.Float32) * wk)
    fo = cute.make_rmem_tensor((16,), cutlass.BFloat16)
    fo.store(acc.load().to(cutlass.BFloat16))
    cute.autovec_copy(fo, cute.local_tile(out[t, None], (16,), (tidx,)))


@cute.jit
def _combine_host(
    c2: cute.Tensor,
    rows: cute.Tensor,
    wts: cute.Tensor,
    out: cute.Tensor,
    stream: cuda.CUstream,
):
    _combine_kernel(c2, rows, wts, out).launch(
        grid=(out.shape[0], 1, 1), block=(256, 1, 1), stream=stream
    )


def _compile_aux():
    m = cute.sym_int64()
    c1 = make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (m, 2048), stride_order=(1, 0), assumed_align=16
    )
    sc1 = make_fake_compact_tensor(
        cutlass.Float32, (16, m), stride_order=(1, 0), assumed_align=16
    )
    a2 = make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (m, 1024), stride_order=(1, 0), assumed_align=16
    )
    s2 = make_fake_compact_tensor(
        cutlass.Float32, (8, m), stride_order=(1, 0), assumed_align=16
    )
    gi = make_fake_compact_tensor(cutlass.Int32, (m,), assumed_align=16)
    act = cute.compile(
        _act_host, c1, sc1, a2, s2, gi, make_fake_stream(), options="--enable-tvm-ffi"
    )
    m2 = cute.sym_int64()
    t = cute.sym_int64()
    c2 = make_fake_compact_tensor(
        cutlass.BFloat16, (m2, 4096), stride_order=(1, 0), assumed_align=16
    )
    rows = make_fake_compact_tensor(
        cutlass.Int32, (t, 10), stride_order=(1, 0), assumed_align=16
    )
    w = make_fake_compact_tensor(
        cutlass.Float32, (t, 10), stride_order=(1, 0), assumed_align=16
    )
    out = make_fake_compact_tensor(
        cutlass.BFloat16, (t, 4096), stride_order=(1, 0), assumed_align=16
    )
    comb = cute.compile(
        _combine_host, c2, rows, w, out, make_fake_stream(), options="--enable-tvm-ffi"
    )
    return act, comb


@cute.kernel
def _route_kernel(logits: cute.Tensor, idx_out: cute.Tensor, w_out: cute.Tensor):
    # one warp per token: softmax over 512 experts (lane owns 16), top-10 by warp argmax, renorm
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    t = bidx * 4 + tidx // 32
    if t < logits.shape[0]:
        fl = cute.make_rmem_tensor((16,), cutlass.BFloat16)
        cute.autovec_copy(cute.local_tile(logits[t, None], (16,), (lane,)), fl)
        p = cute.make_rmem_tensor((16,), cutlass.Float32)
        p.store(fl.load().to(cutlass.Float32))
        mx = p[0]
        for i in cutlass.range_constexpr(1, 16):
            mx = cute.arch.fmax(mx, p[i])
        mx = cute.arch.warp_reduction_max(mx)
        sm = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(16):
            e = cute.math.exp(p[i] - mx)
            p[i] = e
            sm = sm + e
        sm = cute.arch.warp_reduction_sum(sm)
        for i in cutlass.range_constexpr(16):
            p[i] = p[i] / sm
        vals = cute.make_rmem_tensor((10,), cutlass.Float32)
        tot = cutlass.Float32(0.0)
        for k in cutlass.range_constexpr(10):
            bv = cutlass.Float32(-1.0)
            bi = cutlass.Int32(0)
            for i in cutlass.range_constexpr(16):
                if p[i] > bv:
                    bv = p[i]
                    bi = lane * 16 + i
            for off in cutlass.range_constexpr(5):
                ov = cute.arch.shuffle_sync_bfly(bv, offset=16 >> off)
                oi = cute.arch.shuffle_sync_bfly(bi, offset=16 >> off)
                if (ov > bv) | ((ov == bv) & (oi < bi)):
                    bv = ov
                    bi = oi
            for i in cutlass.range_constexpr(16):
                if lane * 16 + i == bi:
                    p[i] = cutlass.Float32(-2.0)
            vals[k] = bv
            tot = tot + bv
            if lane == 0:
                idx_out[t, k] = bi
        if lane == 0:
            for k in cutlass.range_constexpr(10):
                w_out[t, k] = vals[k] / tot


@cute.jit
def _route_host(
    logits: cute.Tensor, idx_out: cute.Tensor, w_out: cute.Tensor, stream: cuda.CUstream
):
    _route_kernel(logits, idx_out, w_out).launch(
        grid=((logits.shape[0] + 3) // 4, 1, 1), block=(128, 1, 1), stream=stream
    )


def _compile_route():
    t = cute.sym_int64()
    lg = make_fake_compact_tensor(
        cutlass.BFloat16, (t, 512), stride_order=(1, 0), assumed_align=16
    )
    io = make_fake_compact_tensor(
        cutlass.Int32, (t, 10), stride_order=(1, 0), assumed_align=16
    )
    wo = make_fake_compact_tensor(
        cutlass.Float32, (t, 10), stride_order=(1, 0), assumed_align=16
    )
    return cute.compile(
        _route_host, lg, io, wo, make_fake_stream(), options="--enable-tvm-ffi"
    )


@cute.kernel
def _perm_rank_kernel(idx: cute.Tensor, rank: cute.Tensor, cnt: cute.Tensor):
    # each CTA ranks 4096 assignments with shared atomics, then reserves its per-expert range with one
    # global atomic per (CTA, expert); rank = global base + local rank (cnt must start zeroed)
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = cutlass.utils.SmemAllocator()
    sc = smem.allocate_tensor(
        cutlass.Int32, cute.make_layout((512,)), byte_alignment=16
    )
    if tidx < 512:
        sc[tidx] = cutlass.Int32(0)
    cute.arch.barrier()
    n = idx.shape[0]
    lr = cute.make_rmem_tensor((4,), cutlass.Int32)
    le = cute.make_rmem_tensor((4,), cutlass.Int32)
    for k in cutlass.range_constexpr(4):
        i = bidx * 4096 + k * 1024 + tidx
        le[k] = cutlass.Int32(0)
        lr[k] = cutlass.Int32(0)
        if i < n:
            e = idx[i]
            le[k] = e
            lr[k] = cute.arch.atomic_add(
                (sc.iterator + e).llvm_ptr, cutlass.Int32(1), scope="cta"
            )
    cute.arch.barrier()
    if tidx < 512:
        c = sc[tidx]
        b = cutlass.Int32(0)
        if c > 0:
            b = cute.arch.atomic_add((cnt.iterator + tidx).llvm_ptr, c, scope="gpu")
        sc[tidx] = b
    cute.arch.barrier()
    for k in cutlass.range_constexpr(4):
        i = bidx * 4096 + k * 1024 + tidx
        if i < n:
            rank[i] = sc[le[k]] + lr[k]


@cute.jit
def _perm_rank_host(
    idx: cute.Tensor, rank: cute.Tensor, cnt: cute.Tensor, stream: cuda.CUstream
):
    _perm_rank_kernel(idx, rank, cnt).launch(
        grid=((idx.shape[0] + 4095) // 4096, 1, 1),
        block=(1024, 1, 1),
        smem=4096,
        stream=stream,
    )


@cute.kernel
def _perm_scan_kernel(
    cnt: cute.Tensor, pad_off: cute.Tensor, pad_end: cute.Tensor, shift_e: cute.Tensor
):
    # exclusive scan of the 128-padded per-expert counts (512 entries, one warp, 16 per lane).
    # shfl.sync.up must clamp at lane 0 (packed c = 0): CuTe DSL < 4.7 defaults shuffle_sync_up
    # to c = 31, which makes every lane read its own value and turns the scan into garbage
    # offsets past the workspace, so the clamp is passed explicitly.
    tidx, _, _ = cute.arch.thread_idx()
    v = cute.make_rmem_tensor((16,), cutlass.Int32)
    s = cutlass.Int32(0)
    for k in cutlass.range_constexpr(16):
        p = (cnt[tidx * 16 + k] + 127) // 128 * 128
        v[k] = p
        s = s + p
    incl = s
    for off in cutlass.range_constexpr(5):
        o = cute.arch.shuffle_sync_up(incl, offset=1 << off, mask_and_clamp=0)
        if tidx >= (1 << off):
            incl = incl + o
    acc = incl - s
    cs = cutlass.Int32(0)
    for k in cutlass.range_constexpr(16):
        cs = cs + cnt[tidx * 16 + k]
    cincl = cs
    for off in cutlass.range_constexpr(5):
        o = cute.arch.shuffle_sync_up(cincl, offset=1 << off, mask_and_clamp=0)
        if tidx >= (1 << off):
            cincl = cincl + o
    cacc = cincl - cs
    for k in cutlass.range_constexpr(16):
        pad_off[tidx * 16 + k] = acc
        shift_e[tidx * 16 + k] = cacc - acc
        cacc = cacc + cnt[tidx * 16 + k]
        acc = acc + v[k]
        pad_end[tidx * 16 + k] = acc


@cute.jit
def _perm_scan_host(
    cnt: cute.Tensor,
    pad_off: cute.Tensor,
    pad_end: cute.Tensor,
    shift_e: cute.Tensor,
    stream: cuda.CUstream,
):
    _perm_scan_kernel(cnt, pad_off, pad_end, shift_e).launch(
        grid=(1, 1, 1), block=(32, 1, 1), stream=stream
    )


@cute.kernel
def _perm_rows_kernel(
    pad_end: cute.Tensor,
    pad_off: cute.Tensor,
    cnt: cute.Tensor,
    gidx: cute.Tensor,
    src_tok: cute.Tensor,
    rstate: cute.Tensor,
    shift_e: cute.Tensor,
    gidx_enc: cute.Tensor,
):
    # row -> expert (first e with pad_end[e] > row, -1 past the last segment); padding rows read token 0
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    r = bidx * 256 + tidx
    if r < gidx.shape[0]:
        _row_lookup(pad_end, pad_off, cnt, gidx, src_tok, rstate, r)
        g = gidx[r]
        ge = cutlass.Int32(-1)
        if g >= 0:
            ge = g + (cutlass.Int32(0) - shift_e[g]) * 1024
        gidx_enc[r] = ge


@cute.jit
def _row_lookup(
    pad_end: cute.Tensor,
    pad_off: cute.Tensor,
    cnt: cute.Tensor,
    gidx: cute.Tensor,
    src_tok: cute.Tensor,
    rstate: cute.Tensor,
    r: cutlass.Int32,
):
    lo = cutlass.Int32(0)
    hi = cutlass.Int32(512)
    for _ in cutlass.range_constexpr(10):
        if lo < hi:
            mid = (lo + hi) // 2
            if pad_end[mid] > r:
                hi = mid
            else:
                lo = mid + 1
    g = lo
    s = cutlass.Int32(-1)
    if lo >= 512:
        g = cutlass.Int32(-1)
    else:
        s = cutlass.Int32(0)
        if r - pad_off[lo] < cnt[lo]:
            s = cutlass.Int32(1)
        else:
            g = cutlass.Int32(-2)
    gidx[r] = g
    rstate[r] = s
    src_tok[r] = cutlass.Int32(0)


@cute.jit
def _perm_rows_host(
    pad_end: cute.Tensor,
    pad_off: cute.Tensor,
    cnt: cute.Tensor,
    gidx: cute.Tensor,
    src_tok: cute.Tensor,
    rstate: cute.Tensor,
    shift_e: cute.Tensor,
    gidx_enc: cute.Tensor,
    stream: cuda.CUstream,
):
    _perm_rows_kernel(
        pad_end, pad_off, cnt, gidx, src_tok, rstate, shift_e, gidx_enc
    ).launch(
        grid=((gidx.shape[0] + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
    )


@cute.kernel
def _perm_place_kernel(
    idx: cute.Tensor,
    rank: cute.Tensor,
    pad_off: cute.Tensor,
    row_of: cute.Tensor,
    src_tok: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 256 + tidx
    if i < idx.shape[0]:
        dest = pad_off[idx[i]] + rank[i]
        row_of[i] = dest
        src_tok[dest] = i // 10


@cute.jit
def _perm_place_host(
    idx: cute.Tensor,
    rank: cute.Tensor,
    pad_off: cute.Tensor,
    row_of: cute.Tensor,
    src_tok: cute.Tensor,
    stream: cuda.CUstream,
):
    _perm_place_kernel(idx, rank, pad_off, row_of, src_tok).launch(
        grid=((idx.shape[0] + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
    )


def _compile_perm():
    tk = cute.sym_int64()
    m = cute.sym_int64()
    idx = make_fake_compact_tensor(cutlass.Int32, (tk,), assumed_align=16)
    rank = make_fake_compact_tensor(cutlass.Int32, (tk,), assumed_align=16)
    po = make_fake_compact_tensor(cutlass.Int32, (512,), assumed_align=16)
    pe = make_fake_compact_tensor(cutlass.Int32, (512,), assumed_align=16)
    gi = make_fake_compact_tensor(cutlass.Int32, (m,), assumed_align=16)
    st = make_fake_compact_tensor(cutlass.Int32, (m,), assumed_align=16)
    ro = make_fake_compact_tensor(cutlass.Int32, (tk,), assumed_align=16)
    opts = "--enable-tvm-ffi"
    cn = make_fake_compact_tensor(cutlass.Int32, (512,), assumed_align=16)
    rs = make_fake_compact_tensor(cutlass.Int32, (m,), assumed_align=16)
    k1 = cute.compile(_perm_rank_host, idx, rank, cn, make_fake_stream(), options=opts)
    se = make_fake_compact_tensor(cutlass.Int32, (512,), assumed_align=16)
    ge = make_fake_compact_tensor(cutlass.Int32, (m,), assumed_align=16)
    k1b = cute.compile(
        _perm_scan_host, cn, po, pe, se, make_fake_stream(), options=opts
    )
    k2 = cute.compile(
        _perm_rows_host,
        pe,
        po,
        cn,
        gi,
        st,
        rs,
        se,
        ge,
        make_fake_stream(),
        options=opts,
    )
    k3 = cute.compile(
        _perm_place_host, idx, rank, po, ro, st, make_fake_stream(), options=opts
    )
    return k1, k1b, k2, k3


@cute.kernel
def _gather_kernel(
    hs: cute.Tensor,
    hss: cute.Tensor,
    src_tok: cute.Tensor,
    rstate: cute.Tensor,
    gidx_enc: cute.Tensor,
    a1: cute.Tensor,
    sfa1: cute.Tensor,
):
    # 16 permuted rows per CTA, one warp per row. Each lane copies eight 16-byte vectors,
    # plus one activation scale, into compact (unpadded) rows.
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    row = bidx * 16 + tidx // 32
    q = tidx % 32
    st = rstate[row]
    if st > 0:
        t = src_tok[row]
        dst = row - (gidx_enc[row] >> 10)
        # issue all eight 16-byte loads before any store (stores may alias the source)
        fr = cute.make_rmem_tensor((128,), cutlass.Uint8)
        for j in cutlass.range_constexpr(8):
            cute.autovec_copy(
                cute.local_tile(hs[t, None], (16,), (q + 32 * j,)),
                cute.local_tile(fr, (16,), (j,)),
            )
        sv = hss[q, t]
        for j in cutlass.range_constexpr(8):
            cute.autovec_copy(
                cute.local_tile(fr, (16,), (j,)),
                cute.local_tile(a1[dst, None], (16,), (q + 32 * j,)),
            )
        sfa1[q, dst] = sv
    # A rows are compact (no padding); a tile may over-read up to 127 rows past the last real
    # row, so those rows are zeroed
    tk = hs.shape[0] * 10
    if (row >= tk) & (row < tk + 128):
        fz = cute.make_rmem_tensor((16,), cutlass.Uint8)
        fz.fill(0)
        for j in cutlass.range_constexpr(8):
            cute.autovec_copy(fz, cute.local_tile(a1[row, None], (16,), (q + 32 * j,)))
        sfa1[q, row] = cutlass.Float32(0.0)


@cute.jit
def _gather_host(
    hs: cute.Tensor,
    hss: cute.Tensor,
    src_tok: cute.Tensor,
    rstate: cute.Tensor,
    gidx_enc: cute.Tensor,
    a1: cute.Tensor,
    sfa1: cute.Tensor,
    stream: cuda.CUstream,
):
    _gather_kernel(hs, hss, src_tok, rstate, gidx_enc, a1, sfa1).launch(
        grid=(a1.shape[0] // 16, 1, 1), block=(512, 1, 1), stream=stream
    )


def _compile_gather():
    t = cute.sym_int64()
    m = cute.sym_int64()
    hs = make_fake_compact_tensor(
        cutlass.Uint8, (t, 4096), stride_order=(1, 0), assumed_align=16
    )
    hss = make_fake_compact_tensor(
        cutlass.Float32, (32, t), stride_order=(1, 0), assumed_align=16
    )
    st = make_fake_compact_tensor(cutlass.Int32, (m,), assumed_align=16)
    rs = make_fake_compact_tensor(cutlass.Int32, (m,), assumed_align=16)
    a1 = make_fake_compact_tensor(
        cutlass.Uint8, (m, 4096), stride_order=(1, 0), assumed_align=16
    )
    sf = make_fake_compact_tensor(
        cutlass.Float32, (32, m), stride_order=(1, 0), assumed_align=16
    )
    ge = make_fake_compact_tensor(cutlass.Int32, (m,), assumed_align=16)
    return cute.compile(
        _gather_host,
        hs,
        hss,
        st,
        rs,
        ge,
        a1,
        sf,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


@cute.kernel
def _gather_pad_kernel(
    hs: cute.Tensor,
    hss: cute.Tensor,
    src_tok: cute.Tensor,
    rstate: cute.Tensor,
    a1: cute.Tensor,
    sfa1: cute.Tensor,
):
    # 16 permuted rows per CTA, one warp per row. Each lane copies eight 16-byte vectors,
    # plus one activation scale; padding rows are zero-filled.
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    row = bidx * 16 + tidx // 32
    q = tidx % 32
    st = rstate[row]
    if st > 0:
        t = src_tok[row]
        # issue all eight 16-byte loads before any store (stores may alias the source)
        fr = cute.make_rmem_tensor((128,), cutlass.Uint8)
        for j in cutlass.range_constexpr(8):
            cute.autovec_copy(
                cute.local_tile(hs[t, None], (16,), (q + 32 * j,)),
                cute.local_tile(fr, (16,), (j,)),
            )
        sv = hss[q, t]
        for j in cutlass.range_constexpr(8):
            cute.autovec_copy(
                cute.local_tile(fr, (16,), (j,)),
                cute.local_tile(a1[row, None], (16,), (q + 32 * j,)),
            )
        sfa1[q, row] = sv
    if st == 0:
        fz = cute.make_rmem_tensor((16,), cutlass.Uint8)
        fz.fill(0)
        for j in cutlass.range_constexpr(8):
            cute.autovec_copy(fz, cute.local_tile(a1[row, None], (16,), (q + 32 * j,)))
        sfa1[q, row] = cutlass.Float32(0.0)


@cute.jit
def _gather_pad_host(
    hs: cute.Tensor,
    hss: cute.Tensor,
    src_tok: cute.Tensor,
    rstate: cute.Tensor,
    a1: cute.Tensor,
    sfa1: cute.Tensor,
    stream: cuda.CUstream,
):
    _gather_pad_kernel(hs, hss, src_tok, rstate, a1, sfa1).launch(
        grid=(a1.shape[0] // 16, 1, 1), block=(512, 1, 1), stream=stream
    )


def _compile_gather_pad():
    t = cute.sym_int64()
    m = cute.sym_int64()
    hs = make_fake_compact_tensor(
        cutlass.Uint8, (t, 4096), stride_order=(1, 0), assumed_align=16
    )
    hss = make_fake_compact_tensor(
        cutlass.Float32, (32, t), stride_order=(1, 0), assumed_align=16
    )
    st = make_fake_compact_tensor(cutlass.Int32, (m,), assumed_align=16)
    rs = make_fake_compact_tensor(cutlass.Int32, (m,), assumed_align=16)
    a1 = make_fake_compact_tensor(
        cutlass.Uint8, (m, 4096), stride_order=(1, 0), assumed_align=16
    )
    sf = make_fake_compact_tensor(
        cutlass.Float32, (32, m), stride_order=(1, 0), assumed_align=16
    )
    return cute.compile(
        _gather_pad_host,
        hs,
        hss,
        st,
        rs,
        a1,
        sf,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


# ---------------------------------------------------------------------------
# Host pipeline: per-device precompile, lookup-only launch.
# ---------------------------------------------------------------------------

NUM_EXPERTS = 512
TOP_K = 10
HIDDEN = 4096
INTER = 1024
BLK = 128
CLUSTER_N = 2
# Compact (unpadded) GEMM1 activation rows pay off where padding is a large share
# of the rows.
COMPACT_A_MAX_T = 8192
# Largest token count the pipeline was validated for (workspace element offsets
# stay below 2**31; the 32768-token bucket uses 392704 permuted rows).
MAX_TOKENS = 32768
# Grouped-GEMM builds: (fuse_quant, epilogue store-block rows, skip padding
# blocks). The *_compact builds serve T <= COMPACT_A_MAX_T.
_GEMM_BUILDS = {
    "gemm1": (True, 64, False),
    "gemm1_compact": (True, 16, True),
    "gemm2": (False, 64, True),
    "gemm2_compact": (False, 16, True),
}
# Number of CuTe DSL programs compiled per device (9 auxiliary + 4 GEMM builds);
# none depends on the token count.
NUM_PROGRAMS = 9 + len(_GEMM_BUILDS)

# Rows of the dummy operands used to compile the grouped GEMMs (the layouts are
# dynamic; a multiple of 128 like every real padded row count).
_COMPILE_ROWS = 3 * BLK


class _Programs(NamedTuple):
    act: Callable[..., Any]
    comb: Callable[..., Any]
    route: Callable[..., Any]
    gather: Callable[..., Any]
    rank: Callable[..., Any]
    scan: Callable[..., Any]
    rows: Callable[..., Any]
    place: Callable[..., Any]
    gather_pad: Callable[..., Any]
    gemm1: Callable[..., Any]
    gemm1_compact: Callable[..., Any]
    gemm2: Callable[..., Any]
    gemm2_compact: Callable[..., Any]


_PROGRAMS: Dict[int, _Programs] = {}
_COMPILE_LOCK = threading.Lock()


def padded_rows(num_tokens: int) -> int:
    """Rows of the permuted (128-row expert segment) workspace for ``num_tokens``:
    every assignment plus the worst-case padding of all 512 segments."""
    tk = num_tokens * TOP_K
    return ((tk + NUM_EXPERTS * (BLK - 1) + BLK - 1) // BLK) * BLK


def workspace_bytes(num_tokens: int) -> int:
    """Device bytes :func:`run` allocates for ``num_tokens`` (excluding ``output``)."""
    m = padded_rows(num_tokens)
    tk = num_tokens * TOP_K
    ints = 4 * (num_tokens * TOP_K * 2 + tk + 4 * NUM_EXPERTS + 4 * m)
    scales = 4 * m * (HIDDEN // BLK + 2 * INTER // BLK + INTER // BLK)
    data = m * (HIDDEN + 2 * INTER + INTER) + 2 * m * HIDDEN
    return ints + scales + data


def _fp8_view(t, lead):
    ct = from_dlpack(
        t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t, assumed_align=16
    )
    if t.dtype == torch.float8_e4m3fn:
        ct.element_type = cutlass.Float8E4M3FN
    return ct.mark_layout_dynamic(leading_dim=lead)


def _as_mk1(x):  # [M, K] row-major -> (M, K, 1)
    return x.unsqueeze(0).permute(1, 2, 0)


def _gemm_operands(a, b, c, sfa, sfb, gidx, sco):
    """a [M,K] fp8, b [E,N,K] fp8, c [M,N], sfa [K/128, M] fp32, sfb [E,N/128,K/128] fp32."""
    return (
        _fp8_view(_as_mk1(a), 1),
        _fp8_view(b.permute(1, 2, 0), 1),
        _fp8_view(_as_mk1(c), 1),
        _fp8_view(sfa.unsqueeze(0).permute(2, 1, 0), 0),
        _fp8_view(sfb.permute(1, 2, 0), 1),
        from_dlpack(gidx, assumed_align=16).mark_layout_dynamic(),
        from_dlpack(sco, assumed_align=16).mark_layout_dynamic(leading_dim=1),
    )


def _compile_grouped_gemm(
    device: torch.device, fuse_quant: bool, row_blk: int, skip_pad_store: bool
):
    """Compile a GEMM1 build (``fuse_quant``: FP8 output + per-(row, 128) scales)
    or a GEMM2 build (bf16 output) against small dummy operands of the real dtypes
    and layouts."""
    m = _COMPILE_ROWS
    if fuse_quant:
        n, k, c_dtype = 2 * INTER, HIDDEN, torch.float8_e4m3fn
    else:
        n, k, c_dtype = HIDDEN, INTER, torch.bfloat16
    a = torch.zeros(m, k, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
    b = torch.zeros(2, n, k, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
    c = torch.zeros(m, n, dtype=torch.uint8 if fuse_quant else c_dtype, device=device)
    if fuse_quant:
        c = c.view(torch.float8_e4m3fn)
    sfa = torch.zeros(k // BLK, m, dtype=torch.float32, device=device)
    sfb = torch.zeros(2, n // BLK, k // BLK, dtype=torch.float32, device=device)
    gidx = torch.zeros(m, dtype=torch.int32, device=device)
    sco = torch.zeros(2 * INTER // BLK, m, dtype=torch.float32, device=device)
    operands = _gemm_operands(a, b, c, sfa, sfb, gidx, sco)
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    gemm = BlockwiseContiguousGroupedGemmKernel(
        cutlass.Float32,
        False,
        (128, 128),
        (1, CLUSTER_N),
        fuse_quant=fuse_quant,
        row_blk=row_blk,
        skip_pad_store=skip_pad_store,
    )
    mac = utils.HardwareInfo(device_id=device.index).get_max_active_clusters(CLUSTER_N)
    opt = "--opt-level 2" if gemm.use_newer_backend_codegen else "--opt-level 3"
    return cute.compile(gemm, *operands, mac, stream, options=opt)


def compile_kernels(device: torch.device) -> None:
    """Compile every program of the pipeline for ``device`` (idempotent, thread-safe).

    Never call this under CUDA-graph capture: it queries the device and allocates.
    """
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _PROGRAMS:
        return
    with _COMPILE_LOCK:
        if index in _PROGRAMS:
            return
        dev = torch.device("cuda", index)
        with torch.cuda.device(dev):
            act, comb = _compile_aux()
            route = _compile_route()
            gather = _compile_gather()
            rank, scan, rows, place = _compile_perm()
            gather_pad = _compile_gather_pad()
            gemms = {
                name: _compile_grouped_gemm(dev, *build)
                for name, build in _GEMM_BUILDS.items()
            }
        _PROGRAMS[index] = _Programs(
            act, comb, route, gather, rank, scan, rows, place, gather_pad, **gemms
        )


def is_compiled(device: torch.device) -> bool:
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return index in _PROGRAMS


def compiled_program_count() -> int:
    """Programs compiled in this process, over all devices."""
    return NUM_PROGRAMS * len(_PROGRAMS)


def _grouped_gemm(fn, a, b, c, sfa, sfb, gidx, sco, stream):
    fn(*_gemm_operands(a, b, c, sfa, sfb, gidx, sco), stream)


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output: torch.Tensor,
    compact: Optional[bool] = None,
) -> torch.Tensor:
    """Run the pipeline into ``output`` [T, 4096] bf16 on the current stream.

    The caller guarantees the contract (dtypes, shapes, contiguity, 16-byte
    alignment, ``1 <= T <= MAX_TOKENS``) and that :func:`compile_kernels` ran for
    the tensors' device. Raises ``RuntimeError`` instead of compiling on a miss.
    ``compact`` overrides the token-count choice between the compact
    (``T <= COMPACT_A_MAX_T``) and padded programs; both are exact for any T (the
    warm-up launches each once).
    """
    dev = hidden_states.device
    programs = _PROGRAMS.get(dev.index)
    if programs is None:
        raise RuntimeError(
            f"fp8 block-scale MoE prefill programs are not compiled for {dev}"
        )
    num_tokens = hidden_states.shape[0]
    TK = num_tokens * TOP_K
    M = padded_rows(num_tokens)
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)

    # routing: softmax -> top-k -> renormalise
    idx32 = torch.empty(num_tokens, TOP_K, dtype=torch.int32, device=dev)
    wts = torch.empty(num_tokens, TOP_K, dtype=torch.float32, device=dev)
    programs.route(routing_logits, idx32, wts, stream)
    flat_e = idx32.view(-1)
    rank = torch.empty(TK, dtype=torch.int32, device=dev)
    pad_off = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=dev)
    pad_end = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=dev)
    gidx = torch.empty(M, dtype=torch.int32, device=dev)
    src_tok = torch.empty(M, dtype=torch.int32, device=dev)
    row_of = torch.empty(TK, dtype=torch.int32, device=dev)
    cnt = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=dev)
    rstate = torch.empty(M, dtype=torch.int32, device=dev)
    shift_e = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=dev)
    gidx_enc = torch.empty(M, dtype=torch.int32, device=dev)
    programs.rank(flat_e, rank, cnt, stream)
    programs.scan(cnt, pad_off, pad_end, shift_e, stream)
    programs.rows(
        pad_end, pad_off, cnt, gidx, src_tok, rstate, shift_e, gidx_enc, stream
    )
    programs.place(flat_e, rank, pad_off, row_of, src_tok, stream)

    a1u = torch.empty(M, HIDDEN, dtype=torch.uint8, device=dev)
    sfa1 = torch.empty(HIDDEN // BLK, M, dtype=torch.float32, device=dev)
    compact_a = num_tokens <= COMPACT_A_MAX_T if compact is None else compact
    if compact_a:
        programs.gather(
            hidden_states.view(torch.uint8),
            hidden_states_scale,
            src_tok,
            rstate,
            gidx_enc,
            a1u,
            sfa1,
            stream,
        )
    else:
        programs.gather_pad(
            hidden_states.view(torch.uint8),
            hidden_states_scale,
            src_tok,
            rstate,
            a1u,
            sfa1,
            stream,
        )
    c1 = torch.empty(M, 2 * INTER, dtype=torch.float8_e4m3fn, device=dev)
    sc1 = torch.empty(2 * INTER // BLK, M, dtype=torch.float32, device=dev)
    _grouped_gemm(
        programs.gemm1_compact if compact_a else programs.gemm1,
        a1u.view(torch.float8_e4m3fn),
        gemm1_weights,
        c1,
        sfa1,
        gemm1_weights_scale,
        gidx_enc if compact_a else gidx,
        sc1,
        stream,
    )
    # GEMM1's operands are dead: let the allocator reuse their blocks (stream-ordered).
    del a1u, sfa1

    a2 = torch.empty(M, INTER, dtype=torch.float8_e4m3fn, device=dev)
    sfa2 = torch.empty(INTER // BLK, M, dtype=torch.float32, device=dev)
    programs.act(c1, sc1, a2, sfa2, rstate, stream)
    del c1

    c2 = torch.empty(M, HIDDEN, dtype=torch.bfloat16, device=dev)
    _grouped_gemm(
        programs.gemm2_compact if compact_a else programs.gemm2,
        a2,
        gemm2_weights,
        c2,
        sfa2,
        gemm2_weights_scale,
        gidx,
        sc1,
        stream,
    )

    rows = row_of.view(num_tokens, TOP_K)
    programs.comb(c2, rows, wts, output, stream)
    return output
