# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""SM120 NVFP4 block-scaled implicit-GEMM Conv3d kernel."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from flashinfer.cute_dsl.sm120_blockscaled import (
    make_sm120_blockscaled_smem_layouts,
)

from ._sm120_conv3d_descriptor import Sm120Nvfp4Conv3dDescriptor
from ._sm120_conv3d_mainloop import CompactSfaPingpongMainloop


class Sm120Nvfp4Conv3dKernel(
    CompactSfaPingpongMainloop,
    Sm120Nvfp4Conv3dDescriptor,
):
    """Specialized 3x3x3 W4A4 Conv3d for SM120.

    The input activation is packed physical-halo NDHWC, activation scales are
    compact NDHWC16, and the prepared weight uses KTRSC reduction order.
    """

    def __init__(
        self,
        *,
        a_copy_bits: int = 64,
        a_copy_layout: str = "row",
        a_producer_warps: int = 4,
        n_pair: bool = False,
        fuse_alpha: bool = True,
        fuse_bias: bool = False,
        raster_order: str = "n",
        swizzle_size: int = 1,
    ):
        if a_copy_bits not in (32, 64, 128):
            raise ValueError(f"a_copy_bits must be 32, 64, or 128; got {a_copy_bits}")
        if a_copy_layout not in ("row", "coalesced"):
            raise ValueError(
                f"a_copy_layout must be 'row' or 'coalesced'; got {a_copy_layout!r}"
            )
        if a_producer_warps not in (1, 2, 4):
            raise ValueError(
                f"a_producer_warps must be 1, 2, or 4; got {a_producer_warps}"
            )
        if fuse_bias and not fuse_alpha:
            raise ValueError("fuse_bias requires fuse_alpha")
        if raster_order not in ("m", "n"):
            raise ValueError(f"raster_order must be 'm' or 'n'; got {raster_order!r}")
        if swizzle_size not in (1, 2, 4, 8):
            raise ValueError(f"swizzle_size must be 1, 2, 4, or 8; got {swizzle_size}")

        self.raster_order = raster_order
        self.swizzle_size = swizzle_size
        self.p3_a_copy_bits = a_copy_bits
        self.p3_a_copy_layout = a_copy_layout
        self.p3_a_producer_warps = a_producer_warps
        self.p3_n_pair = n_pair
        self.p3_fuse_alpha = fuse_alpha
        self.p3_fuse_bias = fuse_bias

        super().__init__(
            acc_dtype=cutlass.Float32,
            sf_vec_size=16,
            tile_shape_mnk=(128, 128, 128),
            epi_tile=(64, 32),
        )

    def _setup_attributes(self):
        super()._setup_attributes()
        if not self.p3_n_pair:
            return

        self.ab_stage = 3
        single_n_layouts = make_sm120_blockscaled_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.smem_alloc_a_dtype,
            self.a_layout,
            self.smem_alloc_b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )
        dual_n_layouts = make_sm120_blockscaled_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.smem_alloc_a_dtype,
            self.a_layout,
            self.smem_alloc_b_dtype,
            self.b_layout,
            2 * self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )
        selected_layouts = (
            single_n_layouts[0],
            dual_n_layouts[1],
            single_n_layouts[2],
            dual_n_layouts[3],
            single_n_layouts[4],
        )
        selected_dtypes = (
            self.smem_alloc_a_dtype,
            self.smem_alloc_b_dtype,
            self.sf_dtype,
            self.sf_dtype,
            self.c_dtype,
        )
        # Account for both pipeline barrier arrays, the order barriers, and
        # each 1024-byte-aligned staging buffer in SharedStorage.
        required_smem_bytes = (4 * self.ab_stage + 2) * 8
        for layout, dtype in zip(selected_layouts, selected_dtypes, strict=True):
            required_smem_bytes = (
                (required_smem_bytes + self.buffer_align_bytes - 1)
                // self.buffer_align_bytes
                * self.buffer_align_bytes
            )
            required_smem_bytes += cute.cosize(layout) * dtype.width // 8
        required_smem_bytes = (
            (required_smem_bytes + self.buffer_align_bytes - 1)
            // self.buffer_align_bytes
            * self.buffer_align_bytes
        )
        if required_smem_bytes > self.smem_capacity:
            raise ValueError(
                "n-pair Conv3d staging exceeds SM120 shared-memory capacity: "
                f"requires {required_smem_bytes} bytes, "
                f"capacity is {self.smem_capacity} bytes"
            )

        self.a_smem_layout_staged = single_n_layouts[0]
        self.b_smem_layout_staged = dual_n_layouts[1]
        self.sfa_smem_layout_staged = single_n_layouts[2]
        self.sfb_smem_layout_staged = dual_n_layouts[3]
        self.epi_smem_layout_staged = single_n_layouts[4]

    def _compute_grid(self, c, tile_shape_mnk, max_active_clusters):
        if self.p3_n_pair:
            c_shape = (tile_shape_mnk[0], 2 * tile_shape_mnk[1])
        else:
            c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            (1, 1, 1),
            swizzle_size=self.swizzle_size,
            raster_along_m=self.raster_order == "m",
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            params,
            max_active_clusters,
        )
        return params, grid

    @cute.jit
    def wrapper(
        self,
        packed_input: cute.Tensor,
        packed_weight: cute.Tensor,
        input_scale: cute.Tensor,
        weight_scale: cute.Tensor,
        alpha_and_bias: cute.Tensor,
        output: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        current_stream: cuda.CUstream,
    ):
        input_n = cute.size(packed_input, mode=[0])
        input_d = cute.size(packed_input, mode=[1])
        input_h = cute.size(packed_input, mode=[2])
        input_w = cute.size(packed_input, mode=[3])
        input_c = cute.size(packed_input, mode=[4]) * 2
        output_k = cute.size(packed_weight, mode=[0])
        filter_t = cute.size(packed_weight, mode=[1])
        filter_r = cute.size(packed_weight, mode=[2])
        filter_s = cute.size(packed_weight, mode=[3])

        input_tensor = cute.make_tensor(
            cute.recast_ptr(packed_input.iterator, dtype=cutlass.Float4E2M1FN),
            cute.make_ordered_layout(
                (input_n, input_d, input_h, input_w, input_c),
                order=(4, 3, 2, 1, 0),
            ),
        )
        weight_tensor = cute.make_tensor(
            cute.recast_ptr(packed_weight.iterator, dtype=cutlass.Float4E2M1FN),
            cute.make_ordered_layout(
                (output_k, filter_t, filter_r, filter_s, input_c),
                order=(4, 3, 2, 1, 0),
            ),
        )
        input_scale_tensor = cute.make_tensor(
            cute.recast_ptr(input_scale.iterator, dtype=cutlass.Float8E4M3FN),
            cute.make_ordered_layout(
                (input_n, input_d, input_h, input_w, input_c // 16),
                order=(4, 3, 2, 1, 0),
            ),
        )
        weight_scale_tensor = cute.make_tensor(
            cute.recast_ptr(weight_scale.iterator, dtype=cutlass.Float8E4M3FN),
            cute.make_layout(cute.size(weight_scale)),
        )

        self(
            a=input_tensor,
            a_zero=alpha_and_bias,
            b=weight_tensor,
            sfa=input_scale_tensor,
            sfb=weight_scale_tensor,
            c=output,
            max_active_clusters=max_active_clusters,
            stream=current_stream,
        )


__all__ = ["Sm120Nvfp4Conv3dKernel"]
