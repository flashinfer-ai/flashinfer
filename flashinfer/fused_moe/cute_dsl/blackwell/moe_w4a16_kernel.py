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

"""SM100 grouped GEMM for NVFP4 expert weights and BF16 activations.

Each routed-row tile selects one expert through FlashInfer's MoE sort metadata.
The kernel decodes each 16-value E2M1 weight block and its E4M3 scale to BF16,
then executes BF16 tensor-core MMA with FP32 accumulation.
"""

from math import ceil, log2
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.mixed_input_helpers as mixed_input_utils
from cutlass.utils.mixed_input_helpers import TransformMode
from cutlass.cute.nvgpu import cpasync, tcgen05

from flashinfer.tllm_enums import (
    ActivationType,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
)

from .moe_w4a16_utils import decode_nvfp4_fragment_to_bf16
from .utils import (
    blk_reduce_bf16,
    fmin,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)


class Sm100W4A16GroupedGemmKernel:
    """Warp-specialized grouped GEMM for the W4A16 MoE pipeline."""

    def __init__(
        self,
        scale_granularity_m: int,
        scale_granularity_k: int,
        acc_dtype: type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        group_count: int,
        activation_type: Optional[int],
        swiglu_alpha: float,
        swiglu_beta: float,
        swiglu_limit: float,
        use_fused_finalize: bool,
        enable_pdl: bool,
    ):
        """
        Initializes the mixed-input GEMM kernel with a specified configuration.
        """
        # Scale granularity defines how many elements share the same scale factor
        # along the M and K modes.
        self.scale_granularity_m = scale_granularity_m
        self.scale_granularity_k = scale_granularity_k
        # Set transform mode
        if cutlass.const_expr(
            self.scale_granularity_m == 0 and self.scale_granularity_k == 0
        ):
            self.scale_mode = TransformMode.ConvertOnly
        else:
            self.scale_mode = TransformMode.ConvertScale
        self.group_count = group_count
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = mma_tiler_mnk
        if activation_type not in (
            None,
            ActivationType.Swiglu.value,
            ActivationType.Relu2.value,
        ):
            raise ValueError(
                f"unsupported W4A16 epilogue activation: {activation_type}"
            )
        self.fuse_activation = activation_type is not None
        self.gated = activation_type == ActivationType.Swiglu.value
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit
        self.parameterized_swiglu = (
            swiglu_alpha != DEFAULT_SWIGLU_ALPHA
            or swiglu_beta != DEFAULT_SWIGLU_BETA
            or swiglu_limit != DEFAULT_SWIGLU_LIMIT
        )
        self.output_m_factor = 2 if self.gated else 1
        self.use_fused_finalize = use_fused_finalize
        self.enable_pdl = enable_pdl
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        # Set specialized warp ids
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.scale_tma_warp_id = 6
        # Schedule warp assigns routed-row tiles to experts.
        self.schedule_warp_id = 7
        self.transform_warp_id = (
            8,
            9,
            10,
            11,
        )
        # Define expected register count for different warps
        self.num_regs_epilogue_warps = 176
        self.num_regs_mma_warp = 96
        self.num_regs_tma_warps = 80
        self.num_regs_transform_warps = 224
        self.num_regs_schedule_warp = 64
        self.threads_per_cta = 32 * (
            max(
                (
                    self.mma_warp_id,
                    self.tma_warp_id,
                    self.scale_tma_warp_id,
                    *self.epilog_warp_id,
                    *self.transform_warp_id,
                )
            )
            + 1
        )

        # Set barrier id for cta sync, epilogue sync, and tmem ptr sync
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            1, 32 * len(self.epilog_warp_id)
        )
        self.tmem_ptr_sync_barrier = pipeline.NamedBarrier(2, self.threads_per_cta)
        self.cta_sync_barrier = pipeline.NamedBarrier(4, self.threads_per_cta)
        self.sched_sync_barrier = pipeline.NamedBarrier(5, 32)

        self.smem_buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Deduce where the transformed A tensor is stored
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue sub-tile
        - Setting up A/scale/B/C stage counts in shared memory
        - Setting up transformed A stage count in shared memory or tensor memory
        - Computing A/transformed A/scale/B/C memory layout
        - Computing tensor memory allocation columns
        """
        # Deduce where the transformed A tensor is stored, shared memory(SMEM) or tensor memory(TMEM)
        self.transform_a_source = mixed_input_utils.get_transform_a_source(
            self.a_major_mode
        )
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.mma_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
            self.transform_a_source,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.mma_tiler_c = (
            self.mma_tiler[0] // self.output_m_factor,
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_c = (
            self.cta_tile_shape_mnk[0] // self.output_m_factor,
            self.cta_tile_shape_mnk[1],
            self.cta_tile_shape_mnk[2],
        )
        self.cluster_tile_shape_mnk = (
            self.cluster_shape_mn[0] * self.cta_tile_shape_mnk[0],
            self.cluster_shape_mn[1] * self.cta_tile_shape_mnk[1],
            self.cta_tile_shape_mnk[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        default_epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.epi_tile = (
            (self.cta_tile_shape_mnk_c[0], default_epi_tile[1])
            if self.gated
            else default_epi_tile
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        # Compute tensor memory(TMEM) columns and stages for each pipeline
        (
            self.num_load2trans_stage,
            self.num_scale_load2trans_stage,
            self.num_trans2mma_stage,
            self.num_acc_stage,
            self.num_c_stage,
            self.num_tile_info_stage,
            self.num_acc_tmem_cols,
            self.num_a_tmem_cols,
        ) = self._compute_stages_and_tmem_cols(
            tiled_mma,
            self.mma_tiler,
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.c_layout,
            self.transform_a_source,
            self.scale_granularity_m,
            self.scale_granularity_k,
            self.smem_buffer_align_bytes,
            self.scale_mode,
            self.use_fused_finalize,
            self.gated,
        )
        self.num_c_store_stage = self.num_c_stage - (2 if self.gated else 0)

        # Align TMEM columns for allocation
        # TMEM allocation requires power-of-2 column alignment
        # and must meet minimum allocation requirements
        self.num_tmem_alloc_cols = cute.round_up(
            self.num_acc_tmem_cols + self.num_a_tmem_cols,
            cute.arch.get_min_tmem_alloc_cols("sm_100"),
        )
        self.num_tmem_alloc_cols = 2 ** (ceil(log2(self.num_tmem_alloc_cols)))
        # Get smem layout for C tensor
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_store_stage,
        )
        self.c_acc_smem_layout = (
            sm100_utils.make_smem_layout_epi(
                self.c_dtype,
                self.c_layout,
                default_epi_tile,
                1,
            )
            if self.gated
            else None
        )
        # Get smem layout for A, transformed A, and B
        (
            self.smem_layout_a,
            self.smem_layout_a_transform,
            self.smem_layout_b,
        ) = mixed_input_utils.compute_smem_layout(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.num_load2trans_stage,
            self.num_trans2mma_stage,
        )
        # Get smem layout for scale tensor
        self.smem_layout_scale_per_stage = None
        self.smem_layout_scale = None
        if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
            # Get scale tile shape and smem layout for scale tensor
            (
                self.scale_tile_shape,
                self.smem_layout_scale_per_stage,
                self.smem_layout_scale,
            ) = mixed_input_utils.get_smem_layout_scale(
                self.mma_tiler,
                self.use_2cta_instrs,
                self.scale_granularity_m,
                self.scale_granularity_k,
                self.scale_major_mode,
                self.a_scale_dtype,
                self.num_scale_load2trans_stage,
            )

    def _validate_inputs(
        self,
        a: cute.Tensor,
        a_scale: Optional[cute.Tensor],
        b: cute.Tensor,
        c: cute.Tensor,
    ) -> None:
        """
        Validates input tensors and their properties.

        :param a: Input tensor A.
        :type a: cute.Tensor
        :param a_scale: Scale tensor for tensor A (None for ConvertOnly mode).
        :type a_scale: Optional[cute.Tensor]
        :param b: Input tensor B.
        :type b: cute.Tensor
        :param c: Output tensor C.
        :type c: cute.Tensor
        :raises ValueError: If inputs don't meet kernel requirements.
        """
        # Validate scale tensor major mode
        if cutlass.const_expr(
            self.scale_mode == TransformMode.ConvertScale
            and utils.LayoutEnum.from_tensor(a_scale).mma_major_mode()
            != tcgen05.OperandMajorMode.MN
        ):
            raise ValueError("scale_major_mode must be M-major")

    @cute.jit
    def wrapper(
        self,
        weight_ptr: cute.Pointer,
        weight_sf_ptr: cute.Pointer,
        activation_ptr: cute.Pointer,
        tile_idx_to_expert_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        permuted_idx_to_expanded_idx_ptr: Optional[cute.Pointer],
        token_final_scales_ptr: Optional[cute.Pointer],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        num_tokens: cutlass.Int64,
        top_k: cutlass.Int64,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Build logical tensors over FlashInfer's packed NVFP4 storage."""
        scale_k = k // self.scale_granularity_k
        weights = cute.make_tensor(
            weight_ptr,
            cute.make_ordered_layout((m, k, self.group_count), order=(1, 0, 2)),
        )
        weight_sf = cute.make_tensor(
            weight_sf_ptr,
            cute.make_ordered_layout((m, scale_k, self.group_count), order=(0, 1, 2)),
        )
        activations = cute.make_tensor(
            activation_ptr,
            cute.make_ordered_layout((n, k, 1), order=(1, 0, 2)),
        )
        output_m = m // self.output_m_factor
        output_layout = cute.make_ordered_layout((output_m, n, 1), order=(0, 1, 2))
        output = cute.make_tensor(output_ptr, output_layout)
        final_output = (
            cute.make_tensor(
                output_ptr,
                cute.make_ordered_layout((output_m, num_tokens, 1), order=(0, 1, 2)),
            )
            if cutlass.const_expr(self.use_fused_finalize)
            else output
        )
        num_route_tiles = n // self.mma_tiler[1]
        tile_idx_to_expert_idx = cute.make_tensor(
            tile_idx_to_expert_idx_ptr, cute.make_layout((num_route_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, cute.make_layout((num_route_tiles,))
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, cute.make_layout((1,))
        )
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout((self.group_count,)))
        permuted_idx_to_expanded_idx = (
            cute.make_tensor(permuted_idx_to_expanded_idx_ptr, cute.make_layout((n,)))
            if cutlass.const_expr(self.use_fused_finalize)
            else None
        )
        token_final_scales = (
            cute.make_tensor(
                token_final_scales_ptr,
                cute.make_ordered_layout((num_tokens, top_k), order=(1, 0)),
            )
            if cutlass.const_expr(self.use_fused_finalize)
            else None
        )
        return self(
            weights,
            weight_sf,
            activations,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            num_non_exiting_tiles,
            alpha,
            output,
            final_output,
            permuted_idx_to_expanded_idx,
            token_final_scales,
            max_active_clusters,
            stream,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        a_scale: Optional[cute.Tensor],  # None for ConvertOnly mode
        b: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha: cute.Tensor,
        c: cute.Tensor,
        final_output: cute.Tensor,
        permuted_idx_to_expanded_idx: Optional[cute.Tensor],
        token_final_scales: Optional[cute.Tensor],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Configure and launch the grouped GEMM."""
        self.a_dtype: type[cutlass.Numeric] = a.element_type
        self.a_scale_dtype: type[cutlass.Numeric] = (
            a_scale.element_type
            if self.scale_mode is TransformMode.ConvertScale
            else None
        )
        self.b_dtype: type[cutlass.Numeric] = b.element_type
        self.c_dtype: type[cutlass.Numeric] = c.element_type
        self.final_scale_dtype: Optional[type[cutlass.Numeric]] = (
            token_final_scales.element_type if self.use_fused_finalize else None
        )
        self.mma_dtype = self.b_dtype

        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.scale_major_mode = (
            utils.LayoutEnum.from_tensor(a_scale).mma_major_mode()
            if self.scale_mode is TransformMode.ConvertScale
            else None
        )
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
            # The public NVFP4 contract stores scales in the 6D MMA layout
            # M(32x4xrest_m)xK(4xrest_k)xL. Preserve that physical layout so
            # in-place scale updates are visible without a host-side copy.
            m, k, group_count = a.shape
            scale_k = cute.ceil_div(k, self.scale_granularity_k)
            m_tiles = cute.ceil_div(m, 128)
            k_tiles = cute.ceil_div(scale_k, 4)
            tile_elements = 32 * 4 * 4
            self.gmem_layout_scale = cute.make_layout(
                (
                    (self.scale_granularity_m, (32, 4, m_tiles)),
                    (self.scale_granularity_k, (4, k_tiles)),
                    group_count,
                ),
                stride=(
                    (0, (16, 4, k_tiles * tile_elements)),
                    (0, (1, tile_elements)),
                    m_tiles * k_tiles * tile_elements,
                ),
            )

        # Validate inputs
        self._validate_inputs(a, a_scale, b, c)

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.mma_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
            self.transform_a_source,
        )
        # Set up gmem copy atoms for A, scale, and B
        a_op = mixed_input_utils.get_tma_atom_kind(
            self.is_a_mcast, self.use_2cta_instrs, False
        )
        b_op = mixed_input_utils.get_tma_atom_kind(
            self.is_b_mcast, self.use_2cta_instrs, True
        )
        a_scale_op = a_op
        # Deduce TMA copy atom and TMA tensor for A, scale, and B
        smem_layout_a_per_stage = cute.slice_(self.smem_layout_a, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            smem_layout_a_per_stage,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if a.element_type is cutlass.Float32 else None
            ),
        )

        tma_atom_scale, tma_tensor_scale = None, None
        if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
            # Partition smem layout for scale tensor to make it compatible with TMA atom
            smem_layout_for_tma_atom = cute.get(
                tiled_mma._thrfrg_A(self.smem_layout_scale_per_stage.outer), mode=[1]
            )
            # ((MMA_M, MMA_K), REST_M, REST_K)
            smem_layout_for_tma_atom = cute.dice(
                smem_layout_for_tma_atom,
                (1, (1,) * cute.rank(self.smem_layout_scale_per_stage.outer)),
            )
            tma_atom_scale, tma_tensor_scale = cute.nvgpu.make_tiled_tma_atom_A(
                a_scale_op,
                cute.make_tensor(a_scale.iterator, self.gmem_layout_scale),
                smem_layout_for_tma_atom,
                # (SCALE_M, 1, SCALE_K)
                (self.scale_tile_shape[0], 1, self.scale_tile_shape[1]),
                tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=(
                    cutlass.TFloat32
                    if a_scale.element_type is cutlass.Float32
                    else None
                ),
            )

        smem_layout_b_per_stage = cute.slice_(self.smem_layout_b, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            smem_layout_b_per_stage,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if b.element_type is cutlass.Float32 else None
            ),
        )

        # Calculate copy size for tensor A, B, and scale
        a_copy_size = cute.size_in_bytes(self.a_dtype, smem_layout_a_per_stage)
        b_copy_size = cute.size_in_bytes(self.b_dtype, smem_layout_b_per_stage)
        a_scale_copy_size = (
            cute.size_in_bytes(self.a_scale_dtype, self.smem_layout_scale_per_stage)
            if self.scale_mode is TransformMode.ConvertScale
            else 0
        )

        self.num_tma_load_bytes_a = a_copy_size
        self.num_tma_load_bytes_b = b_copy_size * cute.size(tiled_mma.thr_id.shape)
        self.num_tma_load_bytes_scale = a_scale_copy_size
        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk_c,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout,
            self.epi_tile,
        )

        @cute.struct
        class SharedStorage:
            # Routed-tile scheduling metadata.
            tile_info: cute.struct.MemRange[cutlass.Int32, 4 * self.num_tile_info_stage]
            a_load2trans_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_load2trans_stage
            ]
            a_load2trans_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_load2trans_stage
            ]
            a_scale_load2trans_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_scale_load2trans_stage
            ]
            a_scale_load2trans_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_scale_load2trans_stage
            ]
            a_trans2mma_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_trans2mma_stage
            ]
            a_trans2mma_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_trans2mma_stage
            ]
            b_load2mma_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_load2trans_stage
            ]
            b_load2mma_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_load2trans_stage
            ]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tile_info_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_info_stage
            ]
            tile_info_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_info_stage
            ]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        # Launch kernel
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_scale,
            tma_tensor_scale,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            c,
            final_output,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            num_non_exiting_tiles,
            alpha,
            permuted_idx_to_expanded_idx,
            token_final_scales,
            self.group_count,
            self.cluster_layout_vmnk,
            self.smem_layout_a,
            self.smem_layout_scale,
            self.smem_layout_a_transform,
            self.smem_layout_b,
            self.c_smem_layout_staged,
            self.c_acc_smem_layout,
            self.epi_tile,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.enable_pdl,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_s: Optional[cute.CopyAtom],
        mS_mkl: Optional[cute.Tensor],
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tensor_c: cute.Tensor,
        final_output: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha: cute.Tensor,
        permuted_idx_to_expanded_idx: Optional[cute.Tensor],
        token_final_scales: Optional[cute.Tensor],
        group_count: cutlass.Constexpr[int],
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        scale_smem_layout: cute.ComposedLayout,
        a_smem_layout_transform: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        c_acc_smem_layout: Optional[cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        """
        GPU device kernel performing the Persistent Mixed-Input Grouped GEMM computation.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        # Prefetch TMA descriptors
        if warp_idx == self.epilog_warp_id[0]:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
                cpasync.prefetch_descriptor(tma_atom_s)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, bidy, bidz = cute.arch.block_idx()
        # Compute how many k_tiles share the same scale
        num_k_tiles_per_scale = max(
            1, self.scale_granularity_k // self.cta_tile_shape_mnk[2]
        )

        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize load2transform pipeline, which tracks the dependencies between TMA's loading
        # of A and B, and the transformation of A and MMA's consumption
        transform_thread_idx = (
            tidx - 32 * self.transform_warp_id[0]
            if tidx >= 32 * self.transform_warp_id[0]
            else tidx
        )
        a_load2trans_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.a_load2trans_full_mbar_ptr.data_ptr(),
            num_stages=self.num_load2trans_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mcast_ctas_a * len(self.transform_warp_id),
            ),
            tx_count=self.num_tma_load_bytes_a,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=transform_thread_idx,
            mcast_mode_mn=(1, 0),  # multicast for A will only happen on the M-mode
            defer_sync=True,
        )
        # Initialize scale_load2trans pipeline, which tracks the dependencies between TMA's loading
        # of scale, and the transformation of A
        scale_load2trans_pipeline = None
        if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
            num_producers_a_scale = self.num_mcast_ctas_a
            scale_load2trans_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.a_scale_load2trans_full_mbar_ptr.data_ptr(),
                num_stages=self.num_scale_load2trans_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    num_producers_a_scale
                    * len(self.transform_warp_id)
                    * num_k_tiles_per_scale,
                ),
                tx_count=self.num_tma_load_bytes_scale,
                cta_layout_vmnk=cluster_layout_vmnk,
                tidx=transform_thread_idx,
                mcast_mode_mn=(
                    1,
                    0,
                ),  # multicast for scale_a will only happen on the M-mode
                defer_sync=True,
            )
        # Initialize transform2mma pipeline, which tracks the dependencies between the transformation
        # of A and MMA's consumption of transformed A
        cta_v_size = cute.size(cluster_layout_vmnk, mode=[0])
        trans2mma_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.a_trans2mma_full_mbar_ptr.data_ptr(),
            num_stages=self.num_trans2mma_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.transform_warp_id) * cta_v_size,
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # Initialize pipeline for tensor B load to MMA
        # MMA warp informs TMA warp to proceed to load next tile of B tensor
        b_load2mma_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_load2mma_full_mbar_ptr.data_ptr(),
            num_stages=self.num_load2trans_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mcast_ctas_b
            ),
            tx_count=self.num_tma_load_bytes_b,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(0, 1),  # multicast for B will only happen on the N-mode
            defer_sync=True,
        )
        # Initialize accumulator pipeline, which tracks the dependencies between
        # MMA's computation of accumulators and epilogue warps' consumption of accumulators
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, cta_v_size * len(self.epilog_warp_id)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # Initialize tile info pipeline, which tracks the dependencies between
        # tile scheduling warp and other warps
        # Skip scheduler warp and TMA scale load warp when scale_mode is ConvertOnly
        # when computing consumer thread count
        num_tile_info_pipeline_consumer_threads = (
            self.threads_per_cta
            - 32
            - (32 if self.scale_mode is TransformMode.ConvertOnly else 0)
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_full_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_info_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                num_tile_info_pipeline_consumer_threads,
            ),
            defer_sync=True,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_ptr_sync_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # Setup smem tensor A/scale/B/C
        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=self.smem_buffer_align_bytes,
            swizzle=c_smem_layout_staged.inner,
        )
        sC_acc = (
            smem.allocate_tensor(
                element_type=self.c_dtype,
                layout=c_acc_smem_layout.outer,
                byte_alignment=self.smem_buffer_align_bytes,
                swizzle=c_acc_smem_layout.inner,
            )
            if cutlass.const_expr(self.gated)
            else None
        )
        # Fused finalize reuses the first C stage as a linear
        # [route, hidden] tile for descriptor-free bulk reduction.
        sFinalize = (
            cute.make_tensor(
                sC.iterator,
                cute.make_layout(
                    (self.epi_tile_n, self.cta_tile_shape_mnk[0]),
                    stride=(self.cta_tile_shape_mnk[0], 1),
                ),
            )
            if cutlass.const_expr(self.use_fused_finalize)
            else None
        )
        sA_input = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=self.smem_buffer_align_bytes,
            swizzle=a_smem_layout.inner,
        )
        sS_input = (
            smem.allocate_tensor(
                element_type=self.a_scale_dtype,
                layout=scale_smem_layout.outer,
                byte_alignment=self.smem_buffer_align_bytes,
                swizzle=scale_smem_layout.inner,
            )
            if self.scale_mode is TransformMode.ConvertScale
            else None
        )
        sB_input = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=self.smem_buffer_align_bytes,
            swizzle=b_smem_layout.inner,
        )
        sA_transform = None
        # Get smem tensor for transformed A when transform_a_source is SMEM
        if cutlass.const_expr(self.transform_a_source == tcgen05.OperandSource.SMEM):
            sA_transform = smem.allocate_tensor(
                element_type=self.mma_dtype,
                layout=a_smem_layout_transform.outer,
                byte_alignment=self.smem_buffer_align_bytes,
                swizzle=a_smem_layout_transform.inner,
            )
        sFinalizeScale = (
            smem.allocate_tensor(
                element_type=cutlass.Float32,
                layout=cute.make_layout((self.epi_tile_n,)),
                byte_alignment=16,
            )
            if cutlass.const_expr(self.use_fused_finalize)
            else None
        )
        sTile_info = storage.tile_info.get_tensor(
            cute.make_layout((4, self.num_tile_info_stage), stride=(1, 4))
        )

        # Compute multicast mask for A/B buffer full
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        s_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            # Scale tensor shares the same multicast mask as the A tensor
            s_full_mcast_mask = a_full_mcast_mask
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # local_tile partition global tensors
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bM, bK, loopM, loopK, loopL)
        gS_mkl = (
            cute.local_tile(
                mS_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            if self.scale_mode is TransformMode.ConvertScale
            else None
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.mma_tiler_c, (None, None, 0)),
            (None, None, None),
        )
        gC_mnl_simt = cute.local_tile(
            tensor_c,
            cute.slice_(self.mma_tiler_c, (None, None, 0)),
            (None, None, None),
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        # Partition global tensor for TiledMMA_A/B/C
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        tCgS = (
            thr_mma.partition_A(gS_mkl)
            if self.scale_mode is TransformMode.ConvertScale
            else None
        )
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCgC_simt = thr_mma.partition_C(gC_mnl_simt)

        # Setup copy atom to load A from shared memory for further transformation
        copy_atom_a_input = (
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.a_dtype, num_bits_per_copy=32
            )
            if self.scale_mode is TransformMode.ConvertScale
            else None
        )
        a_smem_shape = tiled_mma.partition_shape_A(
            cute.dice(self.mma_tiler, (1, None, 1))
        )
        # Setup copy atom to store transformed A into tensor memory or shared memory
        copy_atom_a_transform = mixed_input_utils.get_copy_atom_a_transform(
            self.mma_dtype,
            self.use_2cta_instrs,
            self.transform_a_source,
            a_smem_shape,
            self.a_dtype,
        )

        # Partition global/shared tensor for TMA load A/B
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
            cute.group_modes(sA_input, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )

        tCsS = None
        tSsS = None
        tSgS = None
        if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
            thr_mma_leader_cta = tiled_mma.get_slice(0)
            # (MMA, MMA_M, MMA_K, STAGE)
            tCsS = thr_mma_leader_cta.partition_A(sS_input)
            # ((atom_v, rest_v), STAGE)
            # ((atom_v, rest_v), loopM, loopK, loopL)
            tSsS, tSgS = mixed_input_utils.scale_tma_partition(
                tCsS,
                tCgS,
                tma_atom_s,
                block_in_cluster_coord_vmnk,
                a_cta_layout,
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
            cute.group_modes(sB_input, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB_input)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        # Cluster wait before TMEM alloc and ensure pipelines are ready
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        griddepcontrol_wait()

        # TMEM allocation
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()

        # Schedule warp
        if warp_idx == self.schedule_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_schedule_warp)
            # Persistent tile scheduling loop
            tile_sched = utils.StaticPersistentRuntimeTileScheduler.create(
                tile_sched_params,
                (bidx, bidy, bidz),
                cute.arch.grid_dim(),
                inner_mode=0,
            )
            work_tile = tile_sched.initial_work_tile_info()
            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_info_stage
            )
            num_non_exiting_tiles_value = num_non_exiting_tiles[0]
            not_last_tile = cutlass.Boolean(1)
            while not_last_tile:
                tile_info_pipeline.producer_acquire(tile_info_producer_state)
                cluster_tile_coord_mnl = work_tile.tile_idx
                cta_tile_coord_m = (
                    cluster_tile_coord_mnl[0] * self.cluster_shape_mn[0]
                    + block_in_cluster_coord_vmnk[1] * cute.size(tiled_mma.thr_id.shape)
                    + block_in_cluster_coord_vmnk[0]
                )
                cta_tile_offset_n = block_in_cluster_coord_vmnk[2]
                route_tile_idx = (
                    cluster_tile_coord_mnl[1] * self.cluster_shape_mn[1]
                    + cta_tile_offset_n
                )
                cur_sTile_info = sTile_info[(None, tile_info_producer_state.index)]
                not_last_tile = work_tile.is_valid_tile and (
                    route_tile_idx < num_non_exiting_tiles_value
                )
                # Store tile info into shared memory buffer
                with cute.arch.elect_one():
                    cur_sTile_info[0] = cta_tile_coord_m
                    if not_last_tile:
                        route_start = route_tile_idx * self.cta_tile_shape_mnk[1]
                        cur_sTile_info[1] = route_start
                        cur_sTile_info[2] = tile_idx_to_expert_idx[route_tile_idx]
                        cur_sTile_info[3] = (
                            tile_idx_to_mn_limit[route_tile_idx] - route_start
                        )
                    else:
                        cur_sTile_info[1] = -1
                        cur_sTile_info[2] = group_count
                        cur_sTile_info[3] = -1
                # Fence and barrier to ensure tile info store has finished
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                self.sched_sync_barrier.arrive_and_wait()
                # Commit tile info pipeline
                tile_info_pipeline.producer_commit(tile_info_producer_state)
                # Advance to next tile
                tile_info_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        # Specialized TMA load warp for A/B tensor
        if warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_tma_warps)
            # Persistent tile scheduling loop
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                group_count, sTile_info[(None, tile_info_consumer_state.index)]
            )
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()
            a_load2trans_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_load2trans_stage
            )
            b_load2mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_load2trans_stage
            )

            while work_tile.is_valid_tile:
                tAgA_slice = tAgA[
                    (
                        None,
                        work_tile.cta_coord_m // cute.size(tiled_mma.thr_id.shape),
                        None,
                        work_tile.group_idx,
                    )
                ]
                # Select the routed-row tile assigned by the scheduler.
                coord_n_offset = (
                    (work_tile.coord_n, 0)
                    if cutlass.const_expr(
                        self.b_major_mode == tcgen05.OperandMajorMode.MN
                    )
                    else (0, work_tile.coord_n)
                )
                tBgB_slice = cute.make_tensor(
                    (
                        tBgB.iterator[0] + coord_n_offset[0],
                        coord_n_offset[1] + tBgB.iterator[1],
                    ),
                    cute.slice_(tBgB.layout, (None, 0, None, 0)),
                )

                a_load2trans_producer_state.reset_count()
                peek_load2trans_empty_status = cutlass.Boolean(1)
                if a_load2trans_producer_state.count < k_tile_cnt:
                    peek_load2trans_empty_status = (
                        a_load2trans_pipeline.producer_try_acquire(
                            a_load2trans_producer_state
                        )
                    )
                b_load2mma_producer_state.reset_count()
                for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    a_load2trans_pipeline.producer_acquire(
                        a_load2trans_producer_state, peek_load2trans_empty_status
                    )
                    b_load2mma_pipeline.producer_acquire(b_load2mma_producer_state)
                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, a_load2trans_producer_state.count)],
                        tAsA[(None, a_load2trans_producer_state.index)],
                        tma_bar_ptr=a_load2trans_pipeline.producer_get_barrier(
                            a_load2trans_producer_state
                        ),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, b_load2mma_producer_state.count)],
                        tBsB[(None, b_load2mma_producer_state.index)],
                        tma_bar_ptr=b_load2mma_pipeline.producer_get_barrier(
                            b_load2mma_producer_state
                        ),
                        mcast_mask=b_full_mcast_mask,
                    )
                    a_load2trans_pipeline.producer_commit(a_load2trans_producer_state)
                    b_load2mma_pipeline.producer_commit(b_load2mma_producer_state)
                    a_load2trans_producer_state.advance()
                    b_load2mma_producer_state.advance()
                    if a_load2trans_producer_state.count < k_tile_cnt:
                        peek_load2trans_empty_status = (
                            a_load2trans_pipeline.producer_try_acquire(
                                a_load2trans_producer_state
                            )
                        )
                # Advance to next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                    group_count, sTile_info[(None, tile_info_consumer_state.index)]
                )
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            # Wait A/B buffer empty
            a_load2trans_pipeline.producer_tail(a_load2trans_producer_state)
            b_load2mma_pipeline.producer_tail(b_load2mma_producer_state)

        # Specialized TMA load for scale tensor
        if warp_idx == self.scale_tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_tma_warps)
            if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
                # Persistent tile scheduling loop
                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
                )
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                    group_count, sTile_info[(None, tile_info_consumer_state.index)]
                )
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
                scale_load2trans_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_scale_load2trans_stage
                )
                scale_k_tile_cnt = k_tile_cnt

                while work_tile.is_valid_tile:
                    # ((atom_v, rest_v), RestK)
                    tSgS_slice = tSgS[
                        (
                            None,
                            work_tile.cta_coord_m // cute.size(tiled_mma.thr_id.shape),
                            None,
                            work_tile.group_idx,
                        )
                    ]
                    # Filter zeros in rest mode
                    rest_filtered = cute.filter_zeros(tSgS_slice[(0, None)].layout)
                    tSgS_slice_filtered = cute.make_tensor(
                        tSgS_slice.iterator,
                        cute.make_layout(
                            (tSgS_slice.layout[0].shape, rest_filtered.shape),
                            stride=(tSgS_slice.layout[0].stride, rest_filtered.stride),
                        ),
                    )

                    scale_load2trans_producer_state.reset_count()
                    peek_scale_load2trans_empty_status = cutlass.Boolean(1)
                    if scale_load2trans_producer_state.count < scale_k_tile_cnt:
                        peek_scale_load2trans_empty_status = (
                            scale_load2trans_pipeline.producer_try_acquire(
                                scale_load2trans_producer_state
                            )
                        )
                    for _k_tile in cutlass.range(0, scale_k_tile_cnt, 1, unroll=1):
                        scale_load2trans_pipeline.producer_acquire(
                            scale_load2trans_producer_state,
                            peek_scale_load2trans_empty_status,
                        )
                        # TMA load scale
                        cute.copy(
                            tma_atom_s,
                            tSgS_slice_filtered[
                                (None, scale_load2trans_producer_state.count)
                            ],
                            tSsS[(None, scale_load2trans_producer_state.index)],
                            tma_bar_ptr=scale_load2trans_pipeline.producer_get_barrier(
                                scale_load2trans_producer_state
                            ),
                            mcast_mask=s_full_mcast_mask,
                        )

                        scale_load2trans_producer_state.advance()
                        peek_scale_load2trans_empty_status = cutlass.Boolean(1)
                        if scale_load2trans_producer_state.count < scale_k_tile_cnt:
                            peek_scale_load2trans_empty_status = (
                                scale_load2trans_pipeline.producer_try_acquire(
                                    scale_load2trans_producer_state
                                )
                            )
                    # Advance to next tile
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                        group_count, sTile_info[(None, tile_info_consumer_state.index)]
                    )
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()
                # Wait scale buffer empty
                scale_load2trans_pipeline.producer_tail(scale_load2trans_producer_state)

        # Specialized transform warps
        if warp_idx >= self.transform_warp_id[0]:
            cute.arch.setmaxregister_increase(self.num_regs_transform_warps)
            transform_local_tidx = tidx - 32 * self.transform_warp_id[0]
            # Get the pointer to the TMEM buffer
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            accumulators = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            tCrA = None
            if cutlass.const_expr(
                self.transform_a_source == tcgen05.OperandSource.TMEM
            ):
                tmem_ptr_transform = cute.recast_ptr(
                    accumulators.iterator + self.num_acc_tmem_cols, dtype=self.mma_dtype
                )
                tCrA = cute.make_tensor(
                    tmem_ptr_transform,
                    tiled_mma.make_fragment_A(a_smem_layout_transform.outer).layout,
                )
            else:
                tCrA = tiled_mma.make_fragment_A(sA_transform)
            # Partition tensors for transform input and output and set up the copy atom
            # used for loading and storing transformed A tensor
            src_copy_a, dst_copy_a, tAsA_input, tAsA_transform = (
                mixed_input_utils.transform_partition(
                    self.transform_a_source,
                    self.scale_mode,
                    copy_atom_a_input,
                    copy_atom_a_transform,
                    sA_input,
                    (
                        tCrA
                        if self.transform_a_source == tcgen05.OperandSource.TMEM
                        else sA_transform
                    ),
                    transform_local_tidx,
                )
            )
            # make fragment for input A and transformed A
            tArA = cute.make_rmem_tensor(
                tAsA_input[(None, None, None, None, 0)].shape, tAsA_input.element_type
            )
            tArA_transform = cute.make_rmem_tensor(
                tAsA_input[(None, None, None, None, 0)].shape, self.mma_dtype
            )
            # Partition scale tensor
            smem_thr_copy_S = None
            tSsS_trans = None
            tSrS_copy = None
            tSrS = None
            if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
                smem_thr_copy_S = src_copy_a.get_slice(transform_local_tidx)
                tSsS_trans = smem_thr_copy_S.partition_S(tCsS)
                tSsS_layout_per_stage = tSsS_trans[(None, None, None, None, 0)].layout
                tSrS_copy = cute.make_rmem_tensor(
                    cute.filter_zeros(tSsS_layout_per_stage).shape,
                    self.a_scale_dtype,
                )
                tSrS = cute.make_tensor(
                    tSrS_copy.iterator,
                    cute.make_layout(
                        tSsS_layout_per_stage.shape,
                        stride=tSrS_copy.layout.stride,
                    ),
                )
                assert cute.size(tSrS, mode=[0]) == cute.size(tArA, mode=[0]), (
                    "tSrS and tArA have different leading dimension"
                )
                assert cute.size(tSrS) == cute.size(tArA), (
                    "tSrS and tArA have different shape"
                )
            # Deduce a sub-tile size and tile tensors
            transform_tiler_size = min(
                cute.size(cute.coalesce(tAsA_input.layout), mode=[0]), 128
            )
            transform_tiler = cute.make_layout(transform_tiler_size)
            tArA_load = cute.flat_divide(tArA, transform_tiler)
            tArA_load = cute.group_modes(tArA_load, 1, cute.rank(tArA_load))
            tSrS_load = (
                cute.flat_divide(tSrS, transform_tiler)
                if self.scale_mode is TransformMode.ConvertScale
                else None
            )
            tSrS_load = (
                cute.group_modes(tSrS_load, 1, cute.rank(tSrS_load))
                if self.scale_mode is TransformMode.ConvertScale
                else None
            )
            tArA_transform_store = cute.flat_divide(tArA_transform, transform_tiler)
            tArA_transform_store = cute.group_modes(
                tArA_transform_store, 1, cute.rank(tArA_transform_store)
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                group_count, sTile_info[(None, tile_info_consumer_state.index)]
            )
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()
            a_load2trans_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_load2trans_stage,
            )
            scale_load2trans_consumer_state = (
                pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    self.num_scale_load2trans_stage,
                )
                if self.scale_mode is TransformMode.ConvertScale
                else None
            )
            trans2mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_trans2mma_stage,
            )
            while work_tile.is_valid_tile:
                a_load2trans_consumer_state.reset_count()
                peek_load2trans_full_status = cutlass.Boolean(1)
                if a_load2trans_consumer_state.count < k_tile_cnt:
                    peek_load2trans_full_status = (
                        a_load2trans_pipeline.consumer_try_wait(
                            a_load2trans_consumer_state
                        )
                    )
                peek_scale_load2trans_full_status = cutlass.Boolean(1)
                if cutlass.const_expr(self.scale_mode == TransformMode.ConvertScale):
                    scale_load2trans_consumer_state.reset_count()
                    peek_scale_load2trans_full_status = (
                        scale_load2trans_pipeline.consumer_try_wait(
                            scale_load2trans_consumer_state
                        )
                    )
                trans2mma_producer_state.reset_count()
                peek_trans2mma_empty_status = cutlass.Boolean(1)
                if trans2mma_producer_state.count < k_tile_cnt:
                    peek_trans2mma_empty_status = (
                        trans2mma_pipeline.producer_try_acquire(
                            trans2mma_producer_state
                        )
                    )

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    a_load2trans_pipeline.consumer_wait(
                        a_load2trans_consumer_state, peek_load2trans_full_status
                    )
                    tAsA_input_slice = tAsA_input[
                        (None, None, None, None, a_load2trans_consumer_state.index)
                    ]
                    tAsA_input_slice = cute.flat_divide(
                        tAsA_input_slice, transform_tiler
                    )
                    tAsA_input_slice = cute.group_modes(
                        tAsA_input_slice, 1, cute.rank(tAsA_input_slice)
                    )
                    if cutlass.const_expr(
                        self.scale_mode == TransformMode.ConvertScale
                    ):
                        scale_load2trans_pipeline.consumer_wait(
                            scale_load2trans_consumer_state,
                            peek_scale_load2trans_full_status,
                        )
                    trans2mma_pipeline.producer_acquire(
                        trans2mma_producer_state, peek_trans2mma_empty_status
                    )
                    # load scale tensor when needed
                    if cutlass.const_expr(
                        self.scale_mode == TransformMode.ConvertScale
                    ):
                        if k_tile % num_k_tiles_per_scale == 0:
                            tSsS_slice = tSsS_trans[
                                (
                                    None,
                                    None,
                                    None,
                                    None,
                                    scale_load2trans_consumer_state.index,
                                )
                            ]
                            tSsS_slice_filtered = cute.make_tensor(
                                tSsS_slice.iterator,
                                cute.filter_zeros(tSsS_slice.layout),
                            )
                            cute.autovec_copy(tSsS_slice_filtered, tSrS_copy)
                        cur_scale_load2trans_consumer_state = (
                            scale_load2trans_consumer_state.clone()
                        )
                        if (k_tile + 1) % num_k_tiles_per_scale == 0:
                            scale_load2trans_consumer_state.advance()

                    cur_a_load2trans_consumer_state = (
                        a_load2trans_consumer_state.clone()
                    )
                    for idx in cutlass.range_constexpr(cute.size(tArA_load, mode=[1])):
                        # Load A from shared memory
                        cute.autovec_copy(
                            tAsA_input_slice[(None, idx)],
                            tArA_load[(None, idx)],
                        )
                        if cutlass.const_expr(
                            idx == cute.size(tArA_load, mode=[1]) - 1
                        ):
                            a_load2trans_consumer_state.advance()
                            if a_load2trans_consumer_state.count < k_tile_cnt:
                                peek_load2trans_full_status = (
                                    a_load2trans_pipeline.consumer_try_wait(
                                        a_load2trans_consumer_state
                                    )
                                )
                                if cutlass.const_expr(
                                    self.scale_mode == TransformMode.ConvertScale
                                ):
                                    peek_scale_load2trans_full_status = (
                                        scale_load2trans_pipeline.consumer_try_wait(
                                            scale_load2trans_consumer_state
                                        )
                                    )
                        if cutlass.const_expr(
                            self.scale_mode == TransformMode.ConvertScale
                        ):
                            packed_fragment = cute.recast_tensor(
                                tArA_load[(None, idx)], cutlass.Uint8
                            ).load()
                            scale_fragment = cute.recast_tensor(
                                tSrS_load[(None, (None, idx))], cutlass.Uint8
                            ).load()
                            tensor_transformed = decode_nvfp4_fragment_to_bf16(
                                packed_fragment,
                                scale_fragment,
                            )
                        else:
                            tensor_transformed = mixed_input_utils.cvt_tensor_a(
                                tArA_load[(None, idx)],
                                self.mma_dtype,
                                False,
                            )
                        tArA_transform_store[(None, idx)].store(tensor_transformed)
                    # Store transformed A to tensor memory or shared memory
                    mixed_input_utils.store_transformed_a(
                        tArA_transform,
                        tAsA_transform[
                            (None, None, None, None, trans2mma_producer_state.index)
                        ],
                        dst_copy_a,
                    )
                    if cutlass.const_expr(
                        self.transform_a_source == tcgen05.OperandSource.TMEM
                    ):
                        cute.arch.fence_view_async_tmem_store()
                    else:
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                    if cutlass.const_expr(
                        self.scale_mode == TransformMode.ConvertScale
                    ):
                        scale_load2trans_pipeline.consumer_release(
                            cur_scale_load2trans_consumer_state
                        )

                    a_load2trans_pipeline.consumer_release(
                        cur_a_load2trans_consumer_state
                    )
                    # Signal the completion of transformation
                    trans2mma_pipeline.producer_commit(trans2mma_producer_state)
                    trans2mma_producer_state.advance()
                    if trans2mma_producer_state.count < k_tile_cnt:
                        peek_trans2mma_empty_status = (
                            trans2mma_pipeline.producer_try_acquire(
                                trans2mma_producer_state
                            )
                        )
                # Advance to next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                    group_count, sTile_info[(None, tile_info_consumer_state.index)]
                )
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            # Wait a_transform buffer empty
            trans2mma_pipeline.producer_tail(trans2mma_producer_state)

        # Specialized MMA warp
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma_warp)
            # Get the pointer to the TMEM buffer
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            accumulators = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCrA = None
            if cutlass.const_expr(
                self.transform_a_source == tcgen05.OperandSource.TMEM
            ):
                tmem_ptr_transform = cute.recast_ptr(
                    accumulators.iterator + self.num_acc_tmem_cols, dtype=self.mma_dtype
                )
                tCrA = cute.make_tensor(
                    tmem_ptr_transform,
                    tiled_mma.make_fragment_A(a_smem_layout_transform.outer).layout,
                )
            else:
                tCrA = tiled_mma.make_fragment_A(sA_transform)
            tCtAcc_base = accumulators
            # Persistent tile scheduling loop
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                group_count, sTile_info[(None, tile_info_consumer_state.index)]
            )
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()
            trans2mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_trans2mma_stage
            )
            b_load2mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_load2trans_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                b_load2mma_consumer_state.reset_count()
                trans2mma_consumer_state.reset_count()
                peek_trans2mma_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    if trans2mma_consumer_state.count < k_tile_cnt:
                        peek_trans2mma_full_status = (
                            trans2mma_pipeline.consumer_try_wait(
                                trans2mma_consumer_state
                            )
                        )
                    acc_pipeline.producer_acquire(acc_producer_state)

                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    # Mma mainloop
                    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        trans2mma_pipeline.consumer_wait(
                            trans2mma_consumer_state, peek_trans2mma_full_status
                        )
                        b_load2mma_pipeline.consumer_wait(b_load2mma_consumer_state)
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord_a = (
                                None,
                                None,
                                kblock_idx,
                                trans2mma_consumer_state.index,
                            )
                            kblock_coord_b = (
                                None,
                                None,
                                kblock_idx,
                                b_load2mma_consumer_state.index,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord_a],
                                tCrB[kblock_coord_b],
                                tCtAcc,
                            )
                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        trans2mma_pipeline.consumer_release(trans2mma_consumer_state)
                        b_load2mma_pipeline.consumer_release(b_load2mma_consumer_state)
                        trans2mma_consumer_state.advance()
                        b_load2mma_consumer_state.advance()
                        peek_trans2mma_full_status = cutlass.Boolean(1)
                        if trans2mma_consumer_state.count < k_tile_cnt:
                            peek_trans2mma_full_status = (
                                trans2mma_pipeline.consumer_try_wait(
                                    trans2mma_consumer_state
                                )
                            )
                    # Async arrive accumulator buffer full
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                # Advance to next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                    group_count, sTile_info[(None, tile_info_consumer_state.index)]
                )
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            # Wait for accumulator buffer empty
            acc_pipeline.producer_tail(acc_producer_state)

        # Specialized epilogue warps
        if warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_epilogue_warps)
            epi_tidx = tidx
            # Get the pointer to the TMEM buffer
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            accumulators = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAcc_base = accumulators
            # Partition for epilogue
            fused_tTR_tAcc_base = None
            fused_tiled_copy_t2r = None
            fused_tTR_identity = None
            fused_tTR_rAcc = None
            fused_tiled_copy_r2s = None
            fused_tRS_rC = None
            fused_tRS_sC = None
            if cutlass.const_expr(self.gated):
                # Pair up and gate rows through a logical full-M exchange tile
                # before storing the reduced activation tile with TMA.
                fused_epi_tile = (
                    self.cta_tile_shape_mnk[0],
                    self.epi_tile_n,
                )
                fused_copy_atom_t2r = sm100_utils.get_tmem_load_op(
                    self.cta_tile_shape_mnk,
                    self.c_layout,
                    self.c_dtype,
                    self.acc_dtype,
                    fused_epi_tile,
                    self.use_2cta_instrs,
                )
                fused_tAcc_epi = cute.flat_divide(
                    tCtAcc_base[((None, None), 0, 0, None)],
                    fused_epi_tile,
                )
                fused_tiled_copy_t2r = tcgen05.make_tmem_copy(
                    fused_copy_atom_t2r,
                    fused_tAcc_epi[(None, None, 0, 0, 0)],
                )
                fused_thr_copy_t2r = fused_tiled_copy_t2r.get_slice(epi_tidx)
                fused_tTR_tAcc_base = fused_thr_copy_t2r.partition_S(fused_tAcc_epi)
                fused_identity = cute.make_identity_tensor(self.cta_tile_shape_mnk[:2])
                fused_identity_epi = cute.flat_divide(fused_identity, fused_epi_tile)
                fused_tTR_identity = fused_thr_copy_t2r.partition_D(fused_identity_epi)
                fused_tTR_rAcc = cute.make_rmem_tensor(
                    fused_tTR_identity[(None, None, None, 0, 0)].shape,
                    self.acc_dtype,
                )
                fused_tTR_rC = cute.make_rmem_tensor(
                    fused_tTR_rAcc.shape,
                    self.c_dtype,
                )
                (
                    fused_tiled_copy_r2s,
                    fused_tRS_rC,
                    fused_tRS_sC,
                ) = mixed_input_utils.epilog_smem_copy_and_partition(
                    self.c_layout,
                    self.c_dtype,
                    self.acc_dtype,
                    fused_tiled_copy_t2r,
                    fused_tTR_rC,
                    epi_tidx,
                    sC_acc,
                )
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                mixed_input_utils.epilog_tmem_copy_and_partition(
                    self.cta_tile_shape_mnk,
                    self.c_layout,
                    self.c_dtype,
                    self.acc_dtype,
                    epi_tidx,
                    tCtAcc_base,
                    tCgC,
                    epi_tile,
                    self.use_2cta_instrs,
                )
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = (
                mixed_input_utils.epilog_smem_copy_and_partition(
                    self.c_layout,
                    self.c_dtype,
                    self.acc_dtype,
                    tiled_copy_t2r,
                    tTR_rC,
                    epi_tidx,
                    sC,
                )
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
                simt_atom,
                tTR_gC_partitioned,
            ) = mixed_input_utils.epilog_gmem_copy_and_partition(
                self.c_dtype,
                epi_tidx,
                tma_atom_c,
                tiled_copy_t2r,
                tCgC,
                tCgC_simt,
                epi_tile,
                sC,
            )
            if cutlass.const_expr(self.gated):
                gC_epi = cute.local_tile(mC_mnl, epi_tile, (None, None, None))
                bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    cute.group_modes(gC_epi, 0, 2),
                )

            # Predicates
            thr_mapping = cute.make_identity_tensor(
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1])
            )
            thr_mapping_mn = cute.flat_divide(thr_mapping, epi_tile)
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            m_thr_offset = thr_copy_t2r.partition_D(thr_mapping_mn)
            m_thr_offset = cute.group_modes(m_thr_offset, 3, cute.rank(m_thr_offset))

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_store_stage,
                producer_group=c_producer_group,
            )

            # Persistent tile scheduling loop
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stage
            )
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                group_count, sTile_info[(None, tile_info_consumer_state.index)]
            )
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()
            num_prev_subtiles = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                alpha_val = alpha[work_tile.group_idx]
                if cutlass.const_expr(self.gated):
                    bSG_gC = bSG_gC_partitioned[(None, work_tile.cta_coord_m, None, 0)]
                else:
                    bSG_gC = bSG_gC_partitioned[
                        (
                            None,
                            None,
                            None,
                            work_tile.cta_coord_m // cute.size(tiled_mma.thr_id.shape),
                            0,
                            0,
                        )
                    ]
                    tma_store_offset_coord = (
                        (work_tile.coord_n, 0)
                        if cutlass.const_expr(self.c_layout.is_n_major_c())
                        else (0, work_tile.coord_n)
                    )
                    bSG_gC = cute.make_tensor(
                        (
                            tma_store_offset_coord[0] + bSG_gC.iterator[0],
                            tma_store_offset_coord[1] + bSG_gC.iterator[1],
                        ),
                        bSG_gC.layout,
                    )
                tTR_gC = tTR_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        None,
                        None,
                        work_tile.cta_coord_m // cute.size(tiled_mma.thr_id.shape),
                        0,
                        0,
                    )
                ]
                tTR_gC = cute.make_tensor(
                    tTR_gC.iterator + (work_tile.coord_n * tensor_c.layout.stride[1]),
                    tTR_gC.layout,
                )

                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]
                # Wait for accumulator buffer full
                acc_pipeline.consumer_wait(acc_consumer_state)
                if cutlass.const_expr(not self.gated):
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                if cutlass.const_expr(self.gated):
                    fused_tTR_tAcc = fused_tTR_tAcc_base[
                        (None, None, None, None, None, acc_consumer_state.index)
                    ]
                    fused_subtile_cnt = cute.size(fused_tTR_tAcc.shape, mode=[4])
                    gated_alpha_f32 = cutlass.Float32(alpha_val)
                    for fused_subtile_idx in cutlass.range(fused_subtile_cnt):
                        fused_tTR_tAcc_mn = fused_tTR_tAcc[
                            (None, None, None, 0, fused_subtile_idx)
                        ]
                        cute.copy(
                            fused_tiled_copy_t2r,
                            fused_tTR_tAcc_mn,
                            fused_tTR_rAcc,
                        )
                        fused_acc = fused_tiled_copy_r2s.retile(fused_tTR_rAcc).load()
                        fused_acc = (gated_alpha_f32 * fused_acc).to(self.c_dtype)
                        fused_tRS_rC.store(fused_acc)
                        cute.copy(
                            fused_tiled_copy_r2s,
                            fused_tRS_rC,
                            fused_tRS_sC[(None, None, None, 0)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        num_prev_subtiles += 1
                        c_buffer = num_prev_subtiles % self.num_c_store_stage
                        output_elements_per_thread = (
                            self.cta_tile_shape_mnk_c[0] * self.epi_tile_n
                        ) // (32 * len(self.epilog_warp_id))
                        swiglu_alpha = cutlass.Float32(self.swiglu_alpha)
                        swiglu_beta = cutlass.Float32(self.swiglu_beta)
                        swiglu_limit = cutlass.Float32(self.swiglu_limit)
                        log2_e = cutlass.Float32(1.4426950408889634)
                        neg_swiglu_alpha_log2_e = -(swiglu_alpha * log2_e)
                        for i in cutlass.range_constexpr(
                            0, output_elements_per_thread, 2
                        ):
                            output_idx_0 = epi_tidx + i * (
                                32 * len(self.epilog_warp_id)
                            )
                            output_idx_1 = output_idx_0 + (
                                32 * len(self.epilog_warp_id)
                            )
                            coord_m_0 = output_idx_0 % self.cta_tile_shape_mnk_c[0]
                            coord_n_0 = output_idx_0 // self.cta_tile_shape_mnk_c[0]
                            coord_m_1 = output_idx_1 % self.cta_tile_shape_mnk_c[0]
                            coord_n_1 = output_idx_1 // self.cta_tile_shape_mnk_c[0]
                            up = (
                                cutlass.Float32(sC_acc[(coord_m_0, coord_n_0, 0)]),
                                cutlass.Float32(sC_acc[(coord_m_1, coord_n_1, 0)]),
                            )
                            gate = (
                                cutlass.Float32(
                                    sC_acc[
                                        (
                                            coord_m_0 + self.cta_tile_shape_mnk_c[0],
                                            coord_n_0,
                                            0,
                                        )
                                    ]
                                ),
                                cutlass.Float32(
                                    sC_acc[
                                        (
                                            coord_m_1 + self.cta_tile_shape_mnk_c[0],
                                            coord_n_1,
                                            0,
                                        )
                                    ]
                                ),
                            )
                            if cutlass.const_expr(self.parameterized_swiglu):
                                gate = (
                                    fmin(gate[0], swiglu_limit, nan=True),
                                    fmin(gate[1], swiglu_limit, nan=True),
                                )
                                up = (
                                    -fmin(
                                        -fmin(up[0], swiglu_limit, nan=True),
                                        swiglu_limit,
                                        nan=True,
                                    ),
                                    -fmin(
                                        -fmin(up[1], swiglu_limit, nan=True),
                                        swiglu_limit,
                                        nan=True,
                                    ),
                                )
                            gate_log2e = cute.arch.mul_packed_f32x2(
                                gate,
                                (
                                    neg_swiglu_alpha_log2_e,
                                    neg_swiglu_alpha_log2_e,
                                ),
                            )
                            sigmoid = cute.arch.add_packed_f32x2(
                                (
                                    cute.math.exp2(gate_log2e[0], fastmath=True),
                                    cute.math.exp2(gate_log2e[1], fastmath=True),
                                ),
                                (1.0, 1.0),
                            )
                            sigmoid = (
                                cute.arch.rcp_approx(sigmoid[0]),
                                cute.arch.rcp_approx(sigmoid[1]),
                            )
                            silu = cute.arch.mul_packed_f32x2(gate, sigmoid)
                            if cutlass.const_expr(self.parameterized_swiglu):
                                up = cute.arch.add_packed_f32x2(
                                    up, (swiglu_beta, swiglu_beta)
                                )
                            result = cute.arch.mul_packed_f32x2(up, silu)
                            sC[(coord_m_0, coord_n_0, c_buffer)] = self.c_dtype(
                                result[0]
                            )
                            sC[(coord_m_1, coord_n_1, c_buffer)] = self.c_dtype(
                                result[1]
                            )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[
                                    (
                                        None,
                                        work_tile.coord_n // self.epi_tile_n
                                        + fused_subtile_idx,
                                    )
                                ],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
                tma_distance_to_boundary = work_tile.distance_to_boundary

                # Store accumulator to global memory in subtiles
                subtile_cnt = (
                    0
                    if cutlass.const_expr(self.gated)
                    else cute.size(tTR_tAcc.shape, mode=[3])
                )
                for subtile_idx in cutlass.range(subtile_cnt):
                    # Load accumulator from tensor memory buffer to register
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    if cutlass.const_expr(self.fuse_activation):
                        alpha_f32 = cutlass.Float32(alpha_val)
                        acc_up = tTR_rAcc.load()
                        for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
                            value = cute.arch.mul_packed_f32x2(
                                (acc_up[i], acc_up[i + 1]),
                                (alpha_f32, alpha_f32),
                            )
                            relu = (
                                cute.arch.fmax(value[0], cutlass.Float32(0.0)),
                                cute.arch.fmax(value[1], cutlass.Float32(0.0)),
                            )
                            tTR_rAcc[i], tTR_rAcc[i + 1] = cute.arch.mul_packed_f32x2(
                                relu, relu
                            )
                    if cutlass.const_expr(self.use_fused_finalize):
                        finalize_thr_slice = m_thr_offset[
                            (None, None, None, subtile_idx)
                        ]
                        top_k = token_final_scales.shape[1]
                        finalize_scale_route = epi_tidx
                        finalize_scale_route_in_tile = (
                            subtile_idx * self.epi_tile_n + finalize_scale_route
                        )
                        if finalize_scale_route < self.epi_tile_n:
                            finalize_scale_permuted_row = (
                                work_tile.coord_n + finalize_scale_route_in_tile
                            )
                            finalize_scale_expanded_idx = permuted_idx_to_expanded_idx[
                                finalize_scale_permuted_row
                            ]
                            finalize_scale_safe_idx = cutlass.max(
                                finalize_scale_expanded_idx, cutlass.Int32(0)
                            )
                            finalize_scale_token_idx = finalize_scale_safe_idx // top_k
                            finalize_scale_topk_idx = finalize_scale_safe_idx % top_k
                            finalize_scale_is_valid = cutlass.Int32(
                                finalize_scale_route_in_tile
                                < work_tile.distance_to_boundary
                            )
                            finalize_token_scale = token_final_scales[
                                (
                                    finalize_scale_token_idx * finalize_scale_is_valid,
                                    finalize_scale_topk_idx,
                                )
                            ]
                            sFinalizeScale[finalize_scale_route] = (
                                cutlass.Float32(alpha_val)
                                * cutlass.Float32(finalize_token_scale)
                                * cutlass.Float32(finalize_scale_is_valid)
                            )

                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        for i in cutlass.range(cute.size(tTR_rC), unroll_full=True):
                            finalize_route = finalize_thr_slice[(i)][1]
                            finalize_value = sFinalizeScale[
                                finalize_route % self.epi_tile_n
                            ] * cutlass.Float32(tTR_rAcc[i])
                            sFinalize[
                                (
                                    finalize_route % self.epi_tile_n,
                                    finalize_thr_slice[(i)][0],
                                )
                            ] = self.c_dtype(finalize_value)

                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                        reduce_route = epi_tidx
                        reduce_route_in_tile = (
                            subtile_idx * self.epi_tile_n + reduce_route
                        )
                        if (
                            reduce_route < self.epi_tile_n
                            and reduce_route_in_tile < work_tile.distance_to_boundary
                        ):
                            reduce_permuted_row = (
                                work_tile.coord_n + reduce_route_in_tile
                            )
                            reduce_expanded_idx = permuted_idx_to_expanded_idx[
                                reduce_permuted_row
                            ]
                            reduce_token_idx = reduce_expanded_idx // top_k
                            hidden_base = (
                                work_tile.cta_coord_m * self.cta_tile_shape_mnk[0]
                            )
                            scatter_out = cute.domain_offset(
                                (hidden_base, reduce_token_idx, 0), final_output
                            )
                            copy_elements = cutlass.min(
                                cutlass.Int32(self.cta_tile_shape_mnk[0]),
                                cutlass.Int32(final_output.shape[0]) - hidden_base,
                            )
                            blk_reduce_bf16(
                                scatter_out,
                                sFinalize[(reduce_route, None)],
                                copy_elements * (self.c_dtype.width // 8),
                            )

                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        self.epilog_sync_barrier.arrive_and_wait()
                    elif tma_distance_to_boundary >= self.cta_tile_shape_mnk[1]:
                        # Convert to C type
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        if cutlass.const_expr(not self.fuse_activation):
                            acc_vec = cutlass.Float32(alpha_val) * acc_vec
                        acc_vec = acc_vec.to(self.c_dtype)
                        tRS_rC.store(acc_vec)
                        num_prev_subtiles += 1
                        c_buffer = num_prev_subtiles % self.num_c_stage
                        # Store C to shared memory
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
                        # TMA store C to global memory
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()
                    else:
                        # Convert to C type
                        acc_vec = tTR_rAcc.load()
                        if cutlass.const_expr(not self.fuse_activation):
                            acc_vec = cutlass.Float32(alpha_val) * acc_vec
                        acc_vec = acc_vec.to(self.c_dtype)
                        tTR_rC.store(acc_vec)
                        # Compute predicate for SIMT store
                        tCpC = cute.make_rmem_tensor(
                            cute.make_layout(tTR_rC.shape),
                            cutlass.Boolean,
                        )
                        m_thr_slice = m_thr_offset[(None, None, None, subtile_idx)]
                        for i in cutlass.range(cute.size(tCpC), unroll_full=True):
                            tCpC[i] = (
                                m_thr_slice[(i)][0]
                                + work_tile.cta_coord_m * self.cta_tile_shape_mnk_c[0]
                                < cute.size(tensor_c.shape[0])
                            ) and (m_thr_slice[(i)][1] < work_tile.distance_to_boundary)
                        # Store C to global memory
                        cute.copy(
                            simt_atom,
                            cute.flatten(tTR_rC),
                            cute.flatten(tTR_gC[(None, None, None, subtile_idx)]),
                            pred=cute.flatten(tCpC),
                        )
                # Async arrive accumulator buffer empty
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
                # Advance to next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                work_tile = mixed_input_utils.make_contiguous_group_work_tile_info(
                    group_count, sTile_info[(None, tile_info_consumer_state.index)]
                )
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            if cutlass.const_expr(not self.use_fused_finalize):
                c_pipeline.producer_tail()

        griddepcontrol_launch_dependents()

    @staticmethod
    def _compute_stages_and_tmem_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: tuple[int, int, int],
        cta_tile_shape_mnk: tuple[int, int, int],
        epi_tile: cute.Tile,
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        transform_a_source: tcgen05.OperandSource,
        scale_granularity_m: int,
        scale_granularity_k: int,
        smem_buffer_align_bytes: int,
        scale_mode: TransformMode,
        use_fused_finalize: bool,
        gated: bool,
    ) -> tuple[int, int, int, int, int, int, int, int]:
        """
        Compute pipeline stages and TMEM column allocation configurations.

        This method calculates the number of pipeline stages for different operations
        (tile_info, load2trans, trans2mma, accumulator, etc.) and determines TMEM column allocation
        based on available memory resources and tile configuration.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param c_dtype: Data type of operand C.
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum of operand C.
        :type c_layout: utils.LayoutEnum
        :param transform_a_source: The source of the transformed A tensor.
        :type transform_a_source: tcgen05.OperandSource
        :param scale_granularity_m: The granularity of the scale tensor along the M mode.
        :type scale_granularity_m: int
        :param scale_granularity_k: The granularity of the scale tensor along the K mode.
        :type scale_granularity_k: int
        :param smem_buffer_align_bytes: The alignment of the shared memory buffer.
        :type smem_buffer_align_bytes: int
        :param scale_mode: The transform mode.
        :type scale_mode: TransformMode
        :param use_fused_finalize: Whether the epilogue atomically reduces route
            outputs into token rows.
        :type use_fused_finalize: bool

        :return: A tuple containing the number of stages for:
                 (load2trans, scale_load2trans, transform2mma, accumulator, c, tile_info, tmem_acc_cols, tmem_a_cols)
        :rtype: tuple[int, int, int, int, int, int, int]
        - num_load2trans_stage: Stages for load-to-transform A and B tensors pipeline
        - num_scale_load2trans_stage: Stages for scale load-to-transform A tensor pipeline
        - num_trans2mma_stage: Stages for transform-to-MMA pipeline
        - num_acc_stage: Stages for accumulator-to-epilogue pipeline
        - num_c_stage: Stages for epilogue-to-output C pipeline
        - num_tile_info_stage: Stages for buffers storing tile info
        - num_acc_tmem_cols: TMEM columns for accumulator
        - num_a_tmem_cols: TMEM columns for transformed A tensor
        """
        # Compute tmem columns required for accumulator
        acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        tCtAcc_stage1 = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        num_tmem_acc_col_per_stage = utils.get_num_tmem_alloc_cols(tCtAcc_stage1, True)
        # Heuristic to decide the number of stages for accumulator
        sm100_tmem_columns = cute.arch.get_max_tmem_alloc_cols("sm_100")
        accumulator_stage_count = sm100_tmem_columns // num_tmem_acc_col_per_stage
        if transform_a_source == tcgen05.OperandSource.TMEM:
            if num_tmem_acc_col_per_stage < 128:
                accumulator_stage_count = 3
            elif num_tmem_acc_col_per_stage < 256:
                accumulator_stage_count = 2
            else:
                accumulator_stage_count = 1
        # transformed A in 16bit, thus 1 tmem column could hold 2 elements
        num_elts_per_tmem_col = 32 // tiled_mma.op.a_dtype.width
        num_tmem_cols_a_per_stage = cute.round_up(
            (
                cta_tile_shape_mnk[2] // num_elts_per_tmem_col
                if transform_a_source == tcgen05.OperandSource.TMEM
                else 0
            ),
            4,
        )

        bytes_per_pipeline_stage = 16
        # By default, we use 2 stages for tile info
        num_tile_info_stage = 2
        tile_info_bytes = (
            cute.size_in_bytes(cute.Int32, cute.make_layout((4, num_tile_info_stage)))
            + bytes_per_pipeline_stage * num_tile_info_stage
        )

        c_store_stage_count = 1 if use_fused_finalize else 2
        c_stage_count = c_store_stage_count + (2 if gated else 0)
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * c_stage_count
        finalize_metadata_bytes = (
            cute.size_in_bytes(
                cutlass.Float32,
                cute.make_layout((cute.size(epi_tile[1]),)),
            )
            if use_fused_finalize
            else 0
        )

        smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        if scale_mode == TransformMode.ConvertOnly:
            scale_load2trans_stage_count = 0
            a_scale_bytes_per_stage = 0
        else:
            # Ensure we have 4 buffers for scale tiles needed for 1 CTA tile
            a_scale_k_mode = max(cta_tile_shape_mnk[2] // scale_granularity_k, 1)
            a_scale_m_mode = max(cta_tile_shape_mnk[0] // scale_granularity_m, 1)
            scale_load2trans_stage_count = 4
            a_scale_bytes_per_stage = cute.round_up(
                cute.size_in_bytes(
                    tiled_mma.op.a_dtype,
                    cute.make_layout((a_scale_m_mode, a_scale_k_mode)),
                ),
                smem_buffer_align_bytes,
            )
        a_scale_bytes = (
            a_scale_bytes_per_stage + bytes_per_pipeline_stage
        ) * scale_load2trans_stage_count
        carveout_smem_bytes = (
            bytes_per_pipeline_stage * accumulator_stage_count
            + a_scale_bytes
            + c_bytes
            + finalize_metadata_bytes
            + tile_info_bytes
        )

        # Compute transform stages if A is in TMEM
        num_tmem_acc_cols = cute.round_up(
            accumulator_stage_count * num_tmem_acc_col_per_stage, 4
        )

        transform2mma_stage_count_a_source_tmem_potential = (
            (sm100_tmem_columns - num_tmem_acc_cols) // num_tmem_cols_a_per_stage
            if transform_a_source == tcgen05.OperandSource.TMEM
            else -1
        )
        if (
            transform_a_source == tcgen05.OperandSource.TMEM
            and transform2mma_stage_count_a_source_tmem_potential <= 0
        ):
            raise ValueError("Not enough TMEM capacity for selected tile size")
        a_load_bytes_per_stage = cute.round_up(
            cute.size_in_bytes(
                a_dtype,
                cute.make_layout((cta_tile_shape_mnk[0], cta_tile_shape_mnk[2])),
            ),
            smem_buffer_align_bytes,
        )
        b_load_bytes_per_stage = cute.round_up(
            cute.size_in_bytes(
                b_dtype,
                cute.make_layout(
                    (
                        cta_tile_shape_mnk[1] // cute.size(tiled_mma.thr_id),
                        cta_tile_shape_mnk[2],
                    )
                ),
            ),
            smem_buffer_align_bytes,
        )
        ab_load_bytes_per_stage = (
            a_load_bytes_per_stage
            + b_load_bytes_per_stage
            + 2 * bytes_per_pipeline_stage
        )
        a_transform_bytes_per_stage = (
            cute.round_up(
                cute.size_in_bytes(
                    tiled_mma.op.a_dtype,
                    cute.make_layout((cta_tile_shape_mnk[0], cta_tile_shape_mnk[2])),
                ),
                smem_buffer_align_bytes,
            )
            if transform_a_source == tcgen05.OperandSource.SMEM
            else 0
        )

        a_transform_bytes_per_stage = (
            a_transform_bytes_per_stage + bytes_per_pipeline_stage
        )
        transform2mma_stage_count_a_source_smem_potential = (
            smem_capacity - carveout_smem_bytes
        ) // (ab_load_bytes_per_stage + a_transform_bytes_per_stage)
        transform2mma_stage_count = (
            min(
                transform2mma_stage_count_a_source_tmem_potential,
                transform2mma_stage_count_a_source_smem_potential,
            )
            if transform_a_source == tcgen05.OperandSource.TMEM
            else transform2mma_stage_count_a_source_smem_potential
        )
        load2transform_stage_count = (
            smem_capacity
            - carveout_smem_bytes
            - (transform2mma_stage_count * a_transform_bytes_per_stage)
        ) // ab_load_bytes_per_stage
        if (
            load2transform_stage_count < 2
            or transform2mma_stage_count < 2
            or accumulator_stage_count < 1
        ):
            raise ValueError("Not enough SMEM or TMEM capacity for selected tile size")
        num_tmem_a_cols = transform2mma_stage_count * num_tmem_cols_a_per_stage
        # Fused finalize reuses its single C stage as reduction scratch.
        if not use_fused_finalize:
            c_stage_count += (
                smem_capacity
                - load2transform_stage_count * ab_load_bytes_per_stage
                - transform2mma_stage_count * a_transform_bytes_per_stage
                - scale_load2trans_stage_count * a_scale_bytes_per_stage
                - c_bytes
            ) // c_bytes_per_stage

        return (
            load2transform_stage_count,
            scale_load2trans_stage_count,
            transform2mma_stage_count,
            accumulator_stage_count,
            c_stage_count,
            num_tile_info_stage,
            num_tmem_acc_cols,
            num_tmem_a_cols,
        )

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        """
        Use persistent tile scheduler to compute the grid size for the output tensor C.
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = (cluster_shape_mn[0], cluster_shape_mn[1], max_active_clusters)

        return tile_sched_params, grid

    @staticmethod
    def can_implement(
        mnkl: tuple[int, int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        c_dtype: type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
        scale_granularity_m: int,
        scale_granularity_k: int,
        mma_tiler: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        use_2cta_instrs: bool,
    ) -> bool:
        """
        Check if the kernel can be implemented for the given tensor shapes and data types.
        """
        m, n, k, l = mnkl

        if not mixed_input_utils.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler, cluster_shape_mn, use_2cta_instrs
        ):
            return False
        # Unlike the generic mixed-input kernel, W4A16 consumes multiple
        # 16-value NVFP4 scale blocks within each MMA K tile.
        if (
            scale_granularity_m != 1
            or scale_granularity_k != 16
            or k % scale_granularity_k != 0
            or mma_tiler[2] % scale_granularity_k != 0
        ):
            return False
        if not mixed_input_utils.is_valid_tensor_alignment(
            m,
            n,
            k,
            a_dtype,
            b_dtype,
            c_dtype,
            b_dtype,
            a_major,
            b_major,
            c_major,
            mma_tiler,
            use_2cta_instrs,
            cluster_shape_mn,
            scale_granularity_m,
            scale_granularity_k,
        ):
            return False
        return True
