"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Fused Add + RMSNorm + FP4 Quantization using CuTe-DSL
======================================================

High-performance fused kernel for element-wise addition followed by RMS normalization
and FP4 quantization. Supports both NVFP4 (block_size=16, E4M3 scales) and MXFP4
(block_size=32, UE8M0 scales) formats.

Operation:
    1. residual = residual + input (in-place update)
    2. output = (residual / sqrt(mean(residual²) + eps)) * weight
    3. quantize output to FP4

The residual tensor is modified in-place to contain the fused value (input + residual).

"""

import functools
from typing import Callable, Tuple, Union

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint8

from ..api_logging import flashinfer_api
from .fp4_common import (
    # Constants
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    COPY_BITS,
    # Architecture detection
    get_sm_version,
    # PTX intrinsics - basic ops
    st_global_u64,
    get_ptr_as_int64,
    rcp_approx_ftz,
    fmin_f32,
    fmax_f32,
    # Half2 SIMD intrinsics
    hmax2,
    hmax_to_f32,
    # BFloat2 SIMD intrinsics
    bfloat2_hmax2,
    bfloat2_hmax_to_f32,
    # FP8 and FP4 conversion
    cvt_f32_to_e4m3,
    fp8_e4m3_to_f32_and_rcp,
    cvt_f32_to_ue8m0,
    ue8m0_to_output_scale,
    # Reduction utilities
    row_reduce,
    # Predicate utility
    predicate_k,
    # Helper functions for SF block processing
    load_8_half2,
    half2_mul_8,
    bfloat2_mul_8,
    half2_max_abs_8,
    bfloat2_max_abs_8,
    half2_to_float16,
    bfloat2_to_float16,
    quantize_and_pack_16,
    # Helper functions for Float32 shared memory processing
    load_f32_16_from_smem,
    compute_y_and_max_abs_f32,
)


# =============================================================================
# CuTe-DSL Kernel Class
# =============================================================================


class AddRMSNormFP4QuantKernel:
    """
    Fused Add + RMSNorm + FP4 Quantization Kernel.

    Computes:
        1. residual = input + residual (in-place update)
        2. y = RMSNorm(residual) * weight
        3. quantize y to FP4

    The residual tensor is modified in-place.
    Supports both NVFP4 (block_size=16) and MXFP4 (block_size=32) formats.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        block_size: int,
        output_swizzled: bool,
        is_fp16: bool,
        sm_version: int | None = None,
        scale_format: str | None = None,
        output_both_sf_layouts: bool = False,
    ):
        self.dtype = dtype
        self.H = H
        self.block_size = block_size
        self.output_swizzled = output_swizzled
        self.is_fp16 = is_fp16
        self.sm_version = sm_version if sm_version is not None else get_sm_version()
        self.output_both_sf_layouts = output_both_sf_layouts

        if scale_format is None:
            self.scale_format = "ue8m0" if block_size == 32 else "e4m3"
        else:
            self.scale_format = scale_format

        assert block_size in (16, 32), f"block_size must be 16 or 32, got {block_size}"
        assert self.scale_format in ("e4m3", "ue8m0"), (
            "scale_format must be 'e4m3' or 'ue8m0'"
        )

        self.cluster_n = self._compute_cluster_n(H, dtype, self.sm_version)
        self.H_per_cta = H // self.cluster_n

        self.threads_per_row = self._compute_threads_per_row(self.H_per_cta)
        self.num_threads = self._compute_num_threads(self.H_per_cta)
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

        elem_bytes = dtype.width // 8
        self.vec_size = COPY_BITS // 8 // elem_bytes
        self.num_vec_blocks = max(
            1,
            (self.H_per_cta // self.vec_size + self.threads_per_row - 1)
            // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

        self.num_sf_blocks_per_row = H // block_size

        # Need swizzle params if output_swizzled or output_both_sf_layouts
        if output_swizzled or output_both_sf_layouts:
            num_col_vecs = H // block_size
            self.num_k_tiles = (num_col_vecs + 3) // 4
            self.k_tile_stride = 512

    @staticmethod
    def _compute_cluster_n(H: int, dtype: cutlass.Numeric, sm_version: int) -> int:
        """Compute optimal cluster size based on H and device shared memory.

        Dynamically determines the minimum cluster_n that fits within the
        device's shared memory limit, making it compatible with different
        GPU architectures (e.g., SM100 with 228KB vs SM120 with 128KB).
        """
        if sm_version < 90:
            return 1

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        max_smem_bytes = props.shared_memory_per_block_optin
        elem_size = dtype.width // 8

        for cluster_n in [1, 2, 4, 8, 16]:
            if H % cluster_n != 0:
                continue
            smem_needed = AddRMSNormFP4QuantKernel._estimate_smem_bytes(
                H, cluster_n, elem_size
            )
            if smem_needed <= max_smem_bytes:
                return cluster_n

        return 16

    @staticmethod
    def _compute_threads_per_row(H_per_cta: int) -> int:
        """Compute optimal threads per row."""
        if H_per_cta <= 64:
            return 8
        elif H_per_cta <= 128:
            return 16
        elif H_per_cta <= 3072:
            return 32
        elif H_per_cta <= 6144:
            return 64
        elif H_per_cta <= 16384:
            return 128
        else:
            return 256

    @staticmethod
    def _compute_num_threads(H_per_cta: int) -> int:
        """Compute total threads per block."""
        return 128 if H_per_cta <= 16384 else 256

    @staticmethod
    def _estimate_smem_bytes(H: int, cluster_n: int, elem_size: int) -> int:
        """Estimate shared memory bytes needed for given configuration.

        This is used to dynamically determine cluster_n based on device
        shared memory limits.
        """
        H_per_cta = H // cluster_n
        threads_per_row = AddRMSNormFP4QuantKernel._compute_threads_per_row(H_per_cta)
        num_threads = AddRMSNormFP4QuantKernel._compute_num_threads(H_per_cta)
        rows_per_block = num_threads // threads_per_row
        warps_per_row = max(threads_per_row // 32, 1)

        vec_size = COPY_BITS // 8 // elem_size
        num_vec_blocks = max(
            1, (H_per_cta // vec_size + threads_per_row - 1) // threads_per_row
        )
        cols_per_tile = vec_size * num_vec_blocks * threads_per_row

        tile_bytes = rows_per_block * cols_per_tile * elem_size

        if cluster_n == 1:
            # 4 tiles: sX, sR, sW, sH + reduction buffer
            return 4 * tile_bytes + rows_per_block * warps_per_row * 4
        else:
            # 2 tiles: sX, sR + larger reduction buffer + mbarrier
            return 2 * tile_bytes + rows_per_block * warps_per_row * cluster_n * 4 + 8

    @staticmethod
    def _make_tv_layout(
        threads_per_row: int,
        rows_per_block: int,
        vec_size: int,
        num_vec_blocks: int,
    ) -> tuple:
        """Create Thread-Value layout for coalesced vectorized memory access."""
        shape = (
            (threads_per_row, rows_per_block),
            (vec_size, num_vec_blocks),
        )
        stride = (
            (vec_size * rows_per_block, 1),
            (rows_per_block, rows_per_block * vec_size * threads_per_row),
        )
        return shape, stride

    def _smem_size_in_bytes(self) -> int:
        """Calculate shared memory requirement."""
        elem_size = self.dtype.width // 8
        x_tile_bytes = self.rows_per_block * self.cols_per_tile * elem_size
        r_tile_bytes = self.rows_per_block * self.cols_per_tile * elem_size

        if self.cluster_n == 1:
            w_tile_bytes = self.rows_per_block * self.cols_per_tile * elem_size
            h_tile_bytes = self.rows_per_block * self.cols_per_tile * elem_size
            reduction_bytes = self.rows_per_block * self.warps_per_row * 4
        else:
            w_tile_bytes = 0
            h_tile_bytes = 0
            reduction_bytes = (
                self.rows_per_block * self.warps_per_row * self.cluster_n * 4
            )

        mbar_bytes = 8 if self.cluster_n > 1 else 0

        return (
            x_tile_bytes
            + r_tile_bytes
            + w_tile_bytes
            + h_tile_bytes
            + reduction_bytes
            + mbar_bytes
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        mS_unswizzled: cute.Tensor,
        mGlobalScale: cute.Tensor,
        M: Int32,
        eps: Float32,
        stream,
    ):
        """Host function to launch the kernel.

        Takes tensors directly via TVM-FFI.
        - mX: Input tensor, shape (M, H), row-major (read-only)
        - mR: Residual tensor, shape (M, H), row-major (modified in-place to input + residual)
        - mW: Weight tensor, shape (H,)
        - mY: Output FP4 tensor, shape (M, H // 2), row-major (packed)
        - mS: Scale factor tensor, shape depends on swizzle mode
        - mS_unswizzled: Unswizzled scale factor tensor (used when output_both_sf_layouts=True)
        - mGlobalScale: Global scale tensor, shape (1,), float32
        """

        tv_shape, tv_stride = self._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        self.kernel(
            mX, mR, mW, mY, mS, mS_unswizzled, mGlobalScale, M, eps, tv_layout, tiler_mn
        ).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), self.cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1]
            if cutlass.const_expr(self.cluster_n > 1)
            else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        mS_unswizzled: cute.Tensor,
        mGlobalScale: cute.Tensor,
        M: Int32,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        """Device kernel with cluster sync and Half2 SIMD.

        Performs:
        1. h = input + residual (writes h back to mR in-place)
        2. y = h * rstd * w / global_scale = rmsnorm(h, w) / global_scale
        3. quantizes y to FP4

        mGlobalScale contains the global scale value. The kernel reads it and
        computes 1/global_scale, which is multiplied with rstd to apply:
        y = h * rstd * w / global_scale = rmsnorm(h, w) / global_scale
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        H = self.H
        block_size = self.block_size
        num_sf_blocks_per_row = self.num_sf_blocks_per_row
        is_fp16 = self.is_fp16
        cluster_n = self.cluster_n

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        threads_per_row = tv_layout.shape[0][0]
        warps_per_row = max(threads_per_row // 32, 1)
        rows_per_block = tiler_mn[0]

        lane_in_row = tidx % threads_per_row
        row_in_block = tidx // threads_per_row

        fp4_max_rcp = rcp_approx_ftz(Float32(FLOAT4_E2M1_MAX))

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()

        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        sR = smem.allocate_tensor(
            mR.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        if cutlass.const_expr(cluster_n == 1):
            sW = smem.allocate_tensor(
                mW.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            sH = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(Int64, num_elems=1)

        # Initialize cluster
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # Create identity tensor
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        if cutlass.const_expr(cluster_n == 1):
            mW_expanded_layout = cute.prepend(
                mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
            gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # TiledCopy for loads
        copy_atom_load_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=COPY_BITS,
        )

        # TiledCopy for stores (to write h back to residual)
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mR.element_type,
            num_bits_per_copy=COPY_BITS,
        )

        tiled_copy_load = cute.make_tiled_copy(
            copy_atom_load_async, tv_layout, tiler_mn
        )
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_R = tiled_copy_load.get_slice(tidx)
        thr_copy_R_store = tiled_copy_store.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tRgR = thr_copy_R.partition_S(gR)
        tRsR = thr_copy_R.partition_D(sR)
        tXcX = thr_copy_X.partition_S(cX)

        # Output partition for writing h back to residual (in-place update)
        tRgO = thr_copy_R_store.partition_D(gR)

        if cutlass.const_expr(cluster_n == 1):
            thr_copy_W = tiled_copy_load.get_slice(tidx)
            tWgW = thr_copy_W.partition_S(gW)
            tWsW = thr_copy_W.partition_D(sW)
            tHsH = thr_copy_X.partition_D(sH)

        tXrX = cute.make_fragment_like(tXgX)
        tRrR = cute.make_fragment_like(tRgR)
        tRrO = cute.make_fragment_like(tRgO)  # Register fragment for residual output

        # Bounds checking
        tXpX = predicate_k(tXcX, limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # Phase 1: Async copy
        if row_in_bounds:
            cute.copy(copy_atom_load_async, tXgX, tXsX, pred=tXpX)
            cute.copy(copy_atom_load_async, tRgR, tRsR, pred=tXpX)

        if cutlass.const_expr(cluster_n == 1):
            cute.copy(copy_atom_load_async, tWgW, tWsW, pred=tXpX)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        # Phase 2: h = x + r and sum of squares
        cute.autovec_copy(tXsX, tXrX)
        cute.autovec_copy(tRsR, tRrR)

        x_vals = tXrX.load().to(Float32)
        r_vals = tRrR.load().to(Float32)

        h_vals = x_vals + r_vals
        h_sq = h_vals * h_vals

        # Store h to residual output registers (for in-place update of residual)
        tRrO.store(h_vals.to(self.dtype))

        if cutlass.const_expr(cluster_n == 1):
            h_elem = h_vals.to(mX.element_type)
            tHsH.store(h_elem)

        sum_sq = row_reduce(
            h_sq,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer,
            mbar_ptr,
            cluster_n,
            Float32(0.0),
        )

        mean_sq = sum_sq / H
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        # Read global_scale from device memory (CUDA graph compatible)
        # Note: global_scale is incorporated into the block scale, NOT applied to input
        global_scale_val = mGlobalScale[0]

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # Write h back to residual tensor (in-place update: residual = input + residual)
        if row_in_bounds:
            cute.copy(copy_atom_store, tRrO, tRgO, pred=tXpX)

        # In cluster mode, Phase 3 reads from mR across the FULL hidden dimension,
        # including slices written by other CTAs.  We must ensure all CTAs' global
        # memory writes are visible before any CTA proceeds to Phase 3.
        if cutlass.const_expr(cluster_n > 1):
            cute.arch.fence_acq_rel_cluster()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        actual_row_idx = bidx * rows_per_block + row_in_block

        # Phase 3: RMSNorm + Quantize
        if actual_row_idx < M:
            num_sf_per_thread = (
                num_sf_blocks_per_row + threads_per_row - 1
            ) // threads_per_row

            for sf_iter in range(num_sf_per_thread):
                sf_idx = lane_in_row + sf_iter * threads_per_row

                if sf_idx < num_sf_blocks_per_row:
                    block_start = sf_idx * block_size

                    if cutlass.const_expr(block_size == 16):
                        if cutlass.const_expr(cluster_n == 1):
                            # Shared memory path - use helper functions
                            h_f32 = load_f32_16_from_smem(sH, row_in_block, block_start)
                            w_f32 = load_f32_16_from_smem(sW, row_in_block, block_start)
                            y_f32, max_abs = compute_y_and_max_abs_f32(
                                h_f32, w_f32, rstd
                            )

                            # E4M3: global_scale is incorporated into block scale
                            scale_float = global_scale_val * max_abs * fp4_max_rcp
                            scale_float = fmin_f32(
                                scale_float, Float32(FLOAT8_E4M3_MAX)
                            )
                            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
                            scale_fp8 = Uint8(scale_fp8_u32 & Uint32(0xFF))
                            inv_scale = (
                                fp8_e4m3_to_f32_and_rcp(scale_fp8_u32)
                                * global_scale_val
                            )

                            if cutlass.const_expr(self.output_both_sf_layouts):
                                inner_k_idx = sf_idx % Int32(4)
                                inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                                outer_m_idx = actual_row_idx % Int32(32)
                                k_tile_idx = sf_idx // Int32(4)
                                m_tile_idx = actual_row_idx // Int32(128)
                                m_tile_stride = self.num_k_tiles * self.k_tile_stride
                                swizzled_offset = (
                                    m_tile_idx * m_tile_stride
                                    + k_tile_idx * self.k_tile_stride
                                    + outer_m_idx * Int32(16)
                                    + inner_m_idx * Int32(4)
                                    + inner_k_idx
                                )
                                mS[swizzled_offset] = scale_fp8
                                mS_unswizzled[actual_row_idx, sf_idx] = scale_fp8
                            elif cutlass.const_expr(self.output_swizzled):
                                inner_k_idx = sf_idx % Int32(4)
                                inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                                outer_m_idx = actual_row_idx % Int32(32)
                                k_tile_idx = sf_idx // Int32(4)
                                m_tile_idx = actual_row_idx // Int32(128)
                                m_tile_stride = self.num_k_tiles * self.k_tile_stride
                                swizzled_offset = (
                                    m_tile_idx * m_tile_stride
                                    + k_tile_idx * self.k_tile_stride
                                    + outer_m_idx * Int32(16)
                                    + inner_m_idx * Int32(4)
                                    + inner_k_idx
                                )
                                mS[swizzled_offset] = scale_fp8
                            else:
                                mS[actual_row_idx, sf_idx] = scale_fp8

                            packed64 = quantize_and_pack_16(y_f32, inv_scale)
                            out_offset = block_start // 2
                            out_ptr = get_ptr_as_int64(
                                mY, actual_row_idx * (H // 2) + out_offset
                            )
                            st_global_u64(out_ptr, packed64)

                        else:
                            # Global memory path (cluster mode) - use helper functions
                            # mR contains h = x + r, so load from mR
                            h_h2, w_h2 = load_8_half2(
                                mR, mW, actual_row_idx, block_start, H
                            )

                            if cutlass.const_expr(is_fp16):
                                hw_h2 = half2_mul_8(h_h2, w_h2)
                                max_hw = half2_max_abs_8(hw_h2)
                                max_xw = hmax_to_f32(max_hw)
                                y_f32 = half2_to_float16(hw_h2, rstd)
                            else:
                                hw_h2 = bfloat2_mul_8(h_h2, w_h2)
                                max_hw = bfloat2_max_abs_8(hw_h2)
                                max_xw = bfloat2_hmax_to_f32(max_hw)
                                y_f32 = bfloat2_to_float16(hw_h2, rstd)

                            max_abs = max_xw * rstd

                            # E4M3: global_scale is incorporated into block scale
                            scale_float = global_scale_val * max_abs * fp4_max_rcp
                            scale_float = fmin_f32(
                                scale_float, Float32(FLOAT8_E4M3_MAX)
                            )
                            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
                            scale_fp8 = Uint8(scale_fp8_u32 & Uint32(0xFF))
                            inv_scale = (
                                fp8_e4m3_to_f32_and_rcp(scale_fp8_u32)
                                * global_scale_val
                            )

                            if cutlass.const_expr(self.output_both_sf_layouts):
                                inner_k_idx = sf_idx % Int32(4)
                                inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                                outer_m_idx = actual_row_idx % Int32(32)
                                k_tile_idx = sf_idx // Int32(4)
                                m_tile_idx = actual_row_idx // Int32(128)
                                m_tile_stride = self.num_k_tiles * self.k_tile_stride
                                swizzled_offset = (
                                    m_tile_idx * m_tile_stride
                                    + k_tile_idx * self.k_tile_stride
                                    + outer_m_idx * Int32(16)
                                    + inner_m_idx * Int32(4)
                                    + inner_k_idx
                                )
                                mS[swizzled_offset] = scale_fp8
                                mS_unswizzled[actual_row_idx, sf_idx] = scale_fp8
                            elif cutlass.const_expr(self.output_swizzled):
                                inner_k_idx = sf_idx % Int32(4)
                                inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                                outer_m_idx = actual_row_idx % Int32(32)
                                k_tile_idx = sf_idx // Int32(4)
                                m_tile_idx = actual_row_idx // Int32(128)
                                m_tile_stride = self.num_k_tiles * self.k_tile_stride
                                swizzled_offset = (
                                    m_tile_idx * m_tile_stride
                                    + k_tile_idx * self.k_tile_stride
                                    + outer_m_idx * Int32(16)
                                    + inner_m_idx * Int32(4)
                                    + inner_k_idx
                                )
                                mS[swizzled_offset] = scale_fp8
                            else:
                                mS[actual_row_idx, sf_idx] = scale_fp8

                            packed64 = quantize_and_pack_16(y_f32, inv_scale)
                            out_offset = block_start // 2
                            out_ptr = get_ptr_as_int64(
                                mY, actual_row_idx * (H // 2) + out_offset
                            )
                            st_global_u64(out_ptr, packed64)

                    else:
                        # block_size == 32 (MXFP4) - process in two chunks of 16
                        if cutlass.const_expr(cluster_n == 1):
                            # Shared memory path - use helper functions
                            # Load and compute first 16 elements
                            h_f32_c0 = load_f32_16_from_smem(
                                sH, row_in_block, block_start
                            )
                            w_f32_c0 = load_f32_16_from_smem(
                                sW, row_in_block, block_start
                            )
                            y_f32_c0, max_abs_c0 = compute_y_and_max_abs_f32(
                                h_f32_c0, w_f32_c0, rstd
                            )

                            # Load and compute second 16 elements
                            h_f32_c1 = load_f32_16_from_smem(
                                sH, row_in_block, block_start + Int32(16)
                            )
                            w_f32_c1 = load_f32_16_from_smem(
                                sW, row_in_block, block_start + Int32(16)
                            )
                            y_f32_c1, max_abs_c1 = compute_y_and_max_abs_f32(
                                h_f32_c1, w_f32_c1, rstd
                            )

                            # Combine max_abs from both chunks
                            max_abs = fmax_f32(max_abs_c0, max_abs_c1)

                            # Compute scale factor (E4M3 or UE8M0)
                            if cutlass.const_expr(self.scale_format == "ue8m0"):
                                scale_float = max_abs * fp4_max_rcp
                                scale_ue8m0 = cvt_f32_to_ue8m0(scale_float)
                                scale_u8 = Uint8(scale_ue8m0 & Uint32(0xFF))
                                inv_scale = ue8m0_to_output_scale(scale_ue8m0)
                            else:
                                scale_float = global_scale_val * max_abs * fp4_max_rcp
                                scale_float = fmin_f32(
                                    scale_float, Float32(FLOAT8_E4M3_MAX)
                                )
                                scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
                                scale_u8 = Uint8(scale_fp8_u32 & Uint32(0xFF))
                                inv_scale = (
                                    fp8_e4m3_to_f32_and_rcp(scale_fp8_u32)
                                    * global_scale_val
                                )

                            if cutlass.const_expr(self.output_both_sf_layouts):
                                inner_k_idx = sf_idx % Int32(4)
                                inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                                outer_m_idx = actual_row_idx % Int32(32)
                                k_tile_idx = sf_idx // Int32(4)
                                m_tile_idx = actual_row_idx // Int32(128)
                                m_tile_stride = self.num_k_tiles * self.k_tile_stride
                                swizzled_offset = (
                                    m_tile_idx * m_tile_stride
                                    + k_tile_idx * self.k_tile_stride
                                    + outer_m_idx * Int32(16)
                                    + inner_m_idx * Int32(4)
                                    + inner_k_idx
                                )
                                mS[swizzled_offset] = scale_u8
                                mS_unswizzled[actual_row_idx, sf_idx] = scale_u8
                            elif cutlass.const_expr(self.output_swizzled):
                                inner_k_idx = sf_idx % Int32(4)
                                inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                                outer_m_idx = actual_row_idx % Int32(32)
                                k_tile_idx = sf_idx // Int32(4)
                                m_tile_idx = actual_row_idx // Int32(128)
                                m_tile_stride = self.num_k_tiles * self.k_tile_stride
                                swizzled_offset = (
                                    m_tile_idx * m_tile_stride
                                    + k_tile_idx * self.k_tile_stride
                                    + outer_m_idx * Int32(16)
                                    + inner_m_idx * Int32(4)
                                    + inner_k_idx
                                )
                                mS[swizzled_offset] = scale_u8
                            else:
                                mS[actual_row_idx, sf_idx] = scale_u8

                            # Quantize and store both chunks
                            packed64_c0 = quantize_and_pack_16(y_f32_c0, inv_scale)
                            packed64_c1 = quantize_and_pack_16(y_f32_c1, inv_scale)

                            fp4_offset = actual_row_idx * (H // 2) + sf_idx * (
                                block_size // 2
                            )
                            fp4_ptr_0 = get_ptr_as_int64(mY, fp4_offset)
                            fp4_ptr_1 = get_ptr_as_int64(mY, fp4_offset + Int32(8))
                            st_global_u64(fp4_ptr_0, packed64_c0)
                            st_global_u64(fp4_ptr_1, packed64_c1)

                        else:
                            # Global memory path (cluster mode) - use helper functions
                            # mR contains h = x + r, load in two chunks
                            h_h2_c0, w_h2_c0 = load_8_half2(
                                mR, mW, actual_row_idx, block_start, H
                            )
                            h_h2_c1, w_h2_c1 = load_8_half2(
                                mR, mW, actual_row_idx, block_start + Int32(16), H
                            )

                            if cutlass.const_expr(is_fp16):
                                hw_h2_c0 = half2_mul_8(h_h2_c0, w_h2_c0)
                                hw_h2_c1 = half2_mul_8(h_h2_c1, w_h2_c1)
                                max_c0_h2 = half2_max_abs_8(hw_h2_c0)
                                max_c1_h2 = half2_max_abs_8(hw_h2_c1)
                                max_hw = hmax2(max_c0_h2, max_c1_h2)
                                max_xw = hmax_to_f32(max_hw)
                                y_f32_c0 = half2_to_float16(hw_h2_c0, rstd)
                                y_f32_c1 = half2_to_float16(hw_h2_c1, rstd)
                            else:
                                hw_h2_c0 = bfloat2_mul_8(h_h2_c0, w_h2_c0)
                                hw_h2_c1 = bfloat2_mul_8(h_h2_c1, w_h2_c1)
                                max_c0_h2 = bfloat2_max_abs_8(hw_h2_c0)
                                max_c1_h2 = bfloat2_max_abs_8(hw_h2_c1)
                                max_hw = bfloat2_hmax2(max_c0_h2, max_c1_h2)
                                max_xw = bfloat2_hmax_to_f32(max_hw)
                                y_f32_c0 = bfloat2_to_float16(hw_h2_c0, rstd)
                                y_f32_c1 = bfloat2_to_float16(hw_h2_c1, rstd)

                            max_abs = max_xw * rstd

                            # Compute scale factor (E4M3 or UE8M0)
                            if cutlass.const_expr(self.scale_format == "ue8m0"):
                                scale_float = max_abs * fp4_max_rcp
                                scale_ue8m0 = cvt_f32_to_ue8m0(scale_float)
                                scale_u8 = Uint8(scale_ue8m0 & Uint32(0xFF))
                                inv_scale = ue8m0_to_output_scale(scale_ue8m0)
                            else:
                                scale_float = global_scale_val * max_abs * fp4_max_rcp
                                scale_float = fmin_f32(
                                    scale_float, Float32(FLOAT8_E4M3_MAX)
                                )
                                scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
                                scale_u8 = Uint8(scale_fp8_u32 & Uint32(0xFF))
                                inv_scale = (
                                    fp8_e4m3_to_f32_and_rcp(scale_fp8_u32)
                                    * global_scale_val
                                )

                            if cutlass.const_expr(self.output_both_sf_layouts):
                                inner_k_idx = sf_idx % Int32(4)
                                inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                                outer_m_idx = actual_row_idx % Int32(32)
                                k_tile_idx = sf_idx // Int32(4)
                                m_tile_idx = actual_row_idx // Int32(128)
                                m_tile_stride = self.num_k_tiles * self.k_tile_stride
                                swizzled_offset = (
                                    m_tile_idx * m_tile_stride
                                    + k_tile_idx * self.k_tile_stride
                                    + outer_m_idx * Int32(16)
                                    + inner_m_idx * Int32(4)
                                    + inner_k_idx
                                )
                                mS[swizzled_offset] = scale_u8
                                mS_unswizzled[actual_row_idx, sf_idx] = scale_u8
                            elif cutlass.const_expr(self.output_swizzled):
                                inner_k_idx = sf_idx % Int32(4)
                                inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                                outer_m_idx = actual_row_idx % Int32(32)
                                k_tile_idx = sf_idx // Int32(4)
                                m_tile_idx = actual_row_idx // Int32(128)
                                m_tile_stride = self.num_k_tiles * self.k_tile_stride
                                swizzled_offset = (
                                    m_tile_idx * m_tile_stride
                                    + k_tile_idx * self.k_tile_stride
                                    + outer_m_idx * Int32(16)
                                    + inner_m_idx * Int32(4)
                                    + inner_k_idx
                                )
                                mS[swizzled_offset] = scale_u8
                            else:
                                mS[actual_row_idx, sf_idx] = scale_u8

                            # Quantize and store both chunks
                            packed64_c0 = quantize_and_pack_16(y_f32_c0, inv_scale)
                            packed64_c1 = quantize_and_pack_16(y_f32_c1, inv_scale)

                            fp4_offset = actual_row_idx * (H // 2) + sf_idx * (
                                block_size // 2
                            )
                            fp4_ptr_0 = get_ptr_as_int64(mY, fp4_offset)
                            fp4_ptr_1 = get_ptr_as_int64(mY, fp4_offset + Int32(8))
                            st_global_u64(fp4_ptr_0, packed64_c0)
                            st_global_u64(fp4_ptr_1, packed64_c1)


# =============================================================================
# PyTorch API Functions
# =============================================================================


@functools.cache
def _get_compiled_kernel(
    hidden_size: int,
    block_size: int,
    is_fp16: bool,
    sm_version: int,
    scale_format: str,
    is_sf_swizzled_layout: bool,
    output_both_sf_layouts: bool = False,
) -> Callable:
    """
    Get a compiled kernel closure that takes torch.Tensor directly.

    Uses TVM-FFI for efficient tensor passing without manual pointer construction.
    """
    cutlass_dtype = cutlass.Float16 if is_fp16 else cutlass.BFloat16

    kernel_obj = AddRMSNormFP4QuantKernel(
        dtype=cutlass_dtype,
        H=hidden_size,
        block_size=block_size,
        output_swizzled=is_sf_swizzled_layout,
        is_fp16=is_fp16,
        sm_version=sm_version,
        scale_format=scale_format,
        output_both_sf_layouts=output_both_sf_layouts,
    )

    # Use symbolic size for dynamic M dimension
    sym_m = cute.sym_int()

    # Create fake tensors for compilation with TVM-FFI
    # Use stride_order=(1, 0) for row-major layout and assumed_align=128 for async copy
    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_m, hidden_size), stride_order=(1, 0), assumed_align=128
    )
    r_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_m, hidden_size), stride_order=(1, 0), assumed_align=128
    )
    w_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (hidden_size,), assumed_align=128
    )
    y_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, hidden_size // 2), stride_order=(1, 0), assumed_align=128
    )

    # Scale factor tensor layout depends on swizzle mode or output_both_sf_layouts
    if is_sf_swizzled_layout or output_both_sf_layouts:
        # For swizzled mode, use 1D layout - the swizzle pattern is computed in kernel
        # Size is: num_m_tiles * num_k_tiles * 512, which is independent of M
        # Use a separate symbolic variable for this size
        sym_swizzled_size = cute.sym_int()
        s_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (sym_swizzled_size,), assumed_align=128
        )
    else:
        # For non-swizzled mode, use 2D row-major layout
        s_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8,
            (sym_m, hidden_size // block_size),
            stride_order=(1, 0),
            assumed_align=128,
        )

    # Unswizzled scale factor tensor (always 2D row-major, used when output_both_sf_layouts=True)
    s_unswizzled_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, hidden_size // block_size),
        stride_order=(1, 0),
        assumed_align=128,
    )

    # Create fake stream that uses environment stream at runtime
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Global scale fake tensor (shape [1], float32)
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )

    # Compile with TVM-FFI enabled
    compiled_kernel = cute.compile(
        kernel_obj,
        x_fake,
        r_fake,
        w_fake,
        y_fake,
        s_fake,
        s_unswizzled_fake,
        global_scale_fake,
        Int32(1),  # Dummy M
        Float32(1e-6),  # Dummy eps
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        x: torch.Tensor,
        r: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
        s: torch.Tensor,
        s_unswizzled: torch.Tensor,
        global_scale: torch.Tensor,
        M: int,
        eps: float,
    ) -> None:
        """Runtime API that passes torch tensors directly via TVM-FFI."""
        nonlocal compiled_kernel
        # For swizzled mode or output_both_sf_layouts, flatten the scale tensor to 1D
        s_tensor = (
            s.flatten()
            if (is_sf_swizzled_layout or output_both_sf_layouts)
            else s.contiguous()
        )
        # View y as uint8 since kernel expects uint8 but caller may pass float4_e2m1fn_x2
        y_uint8 = y.view(torch.uint8)
        compiled_kernel(
            x,
            r,
            w,
            y_uint8,
            s_tensor,
            s_unswizzled.contiguous(),
            global_scale,
            Int32(M),
            Float32(eps),
        )

    return tensor_api


@flashinfer_api
def add_rmsnorm_fp4quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    y_fp4: torch.Tensor | None = None,
    block_scale: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
    eps: float = 1e-6,
    block_size: int = 16,
    scale_format: str | None = None,
    is_sf_swizzled_layout: bool = False,
    output_both_sf_layouts: bool = False,
    block_scale_unswizzled: torch.Tensor | None = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """
    Fused Add + RMS normalization + FP4 quantization using CuTe-DSL.

    Computes:
        1. ``residual = residual + input`` (in-place update)
        2. ``y = RMSNorm(residual) * weight``
        3. Optionally applies global scaling (``y = y / global_scale``)
        4. Quantizes ``y`` to FP4

    The residual tensor is modified in-place to contain the fused value.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, shape ``(batch_size, hidden_size)`` or ``(batch_size, seq_len, hidden_size)``.
        Must be ``torch.float16`` or ``torch.bfloat16``. Read-only.
    residual : torch.Tensor
        Residual tensor. Must have the same shape and dtype as ``input``.
        **Modified in-place** to contain ``residual + input``.
    weight : torch.Tensor
        Weight tensor for RMSNorm, shape ``(hidden_size,)``.
        Must have the same dtype as input.
    y_fp4 : torch.Tensor, optional
        Output tensor for quantized values in FP4_E2M1 format with dtype
        ``torch.float4_e2m1fn_x2``.
        Shape must be ``(batch_size, hidden_size // 2)`` or matching 3D input.
        If ``None``, will be allocated automatically.
    block_scale : torch.Tensor, optional
        Output tensor for per-block scale factors.

        - If ``is_sf_swizzled_layout=False`` and ``output_both_sf_layouts=False``: row-major
          layout with shape ``(batch_size, hidden_size // block_size)`` or matching 3D input.
        - If ``is_sf_swizzled_layout=True`` or ``output_both_sf_layouts=True``: swizzled layout
          for efficient tensor core access, with shape
          ``(batch_size * hidden_size // block_size,)`` flattened.
          The swizzle pattern uses 128x4 tiles where scales are arranged as:
          ``[m_tile][k_tile][outer_m (32)][inner_m (4)][inner_k (4)]``.

        Dtype should be ``torch.float8_e4m3fn`` for E4M3 format or ``torch.uint8``
        for UE8M0 format. If ``None``, will be allocated automatically.
    global_scale : torch.Tensor, optional
        Global scale factor tensor of shape ``(1,)`` with dtype ``torch.float32``.
        If provided, the RMSNorm output is divided by this value before quantization:
        ``y = rmsnorm(h, w) / global_scale`` where ``h = input + residual``.
        This is used for NVFP4 format where a pre-computed global scale lifts
        per-block scales into optimal dynamic range.
        If ``None``, no global scaling is applied (equivalent to global_scale=1.0).
    eps : float
        Epsilon for numerical stability in RMSNorm. Default is ``1e-6``.
    block_size : int
        Number of elements per quantization block. Default is ``16``.

        - ``16``: NVFP4 format with E4M3 scale factors
        - ``32``: MXFP4 format with UE8M0 scale factors
    scale_format : str, optional
        Scale factor format: ``"e4m3"`` or ``"ue8m0"``.
        If ``None``, auto-selects based on ``block_size``:
        ``"e4m3"`` for block_size=16, ``"ue8m0"`` for block_size=32.
    is_sf_swizzled_layout : bool
        If ``True``, output scale factors in swizzled layout optimized for
        tensor core GEMM operations. The swizzle uses 128x4 tiles with the pattern:
        ``[m_tile_idx * k_tiles * 512 + k_tile_idx * 512 + outer_m * 16 + inner_m * 4 + inner_k]``
        where ``outer_m = row % 32``, ``inner_m = (row % 128) // 32``, etc.
        Default is ``False`` (row-major layout).
        Note: This parameter is ignored when ``output_both_sf_layouts=True``.
    output_both_sf_layouts : bool
        If ``True``, return both swizzled and unswizzled scale factors.
        When enabled, ``block_scale`` contains the swizzled layout and
        ``block_scale_unswizzled`` contains the row-major layout.
        This overrides ``is_sf_swizzled_layout``.
        Default is ``False``.
    block_scale_unswizzled : torch.Tensor, optional
        Output tensor for unswizzled per-block scale factors (row-major layout).
        Only used when ``output_both_sf_layouts=True``.
        Shape is ``(batch_size, hidden_size // block_size)`` or matching 3D input.
        Dtype should be ``torch.float8_e4m3fn`` for E4M3 format or ``torch.uint8``
        for UE8M0 format. If ``None``, will be allocated automatically when
        ``output_both_sf_layouts=True``.

    Returns
    -------
    Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        When ``output_both_sf_layouts=False``:
            A tuple of ``(y_fp4, block_scale)``:

            - ``y_fp4``: Quantized FP4 values packed as uint8.
            - ``block_scale``: Per-block scale factors (swizzled or row-major based on
              ``is_sf_swizzled_layout``).

        When ``output_both_sf_layouts=True``:
            A tuple of ``(y_fp4, block_scale, block_scale_unswizzled)``:

            - ``y_fp4``: Quantized FP4 values packed as uint8.
            - ``block_scale``: Per-block scale factors in swizzled layout.
            - ``block_scale_unswizzled``: Per-block scale factors in row-major layout.

    Notes
    -----
    - Requires SM100+ (Blackwell) for FP4 quantization PTX intrinsics.
    - For block_size=16 (NVFP4): uses E4M3 scale factors (max value 448.0).
    - For block_size=32 (MXFP4): uses UE8M0 scale factors (power-of-2 scales).
    - FP4 E2M1 format has a max representable value of 6.0.
    """
    is_3d = input.dim() == 3
    if is_3d:
        B, S, H = input.shape
        input_2d = input.view(B * S, H).contiguous()
        residual_2d = residual.view(B * S, H).contiguous()
    else:
        input_2d = input
        residual_2d = residual

    batch_size, hidden_size = input_2d.shape
    dtype = input.dtype

    assert hidden_size % block_size == 0, "hidden_size must be divisible by block_size"
    assert hidden_size >= 64, "hidden_size must be >= 64"
    assert block_size in [16, 32], "block_size must be 16 or 32"

    is_fp16 = dtype == torch.float16
    actual_scale_format = (
        scale_format if scale_format else ("ue8m0" if block_size == 32 else "e4m3")
    )
    sm_version = get_sm_version(input.device)

    # Determine scale dtype based on format
    scale_dtype = torch.uint8 if actual_scale_format == "ue8m0" else torch.float8_e4m3fn
    num_sf_blocks_per_row = hidden_size // block_size

    # Allocate output tensors if not provided
    if y_fp4 is None:
        if is_3d:
            y_fp4 = torch.empty(
                (B, S, hidden_size // 2),
                dtype=torch.float4_e2m1fn_x2,
                device=input.device,
            )
        else:
            y_fp4 = torch.empty(
                (batch_size, hidden_size // 2),
                dtype=torch.float4_e2m1fn_x2,
                device=input.device,
            )

    if block_scale is None:
        # When output_both_sf_layouts=True, block_scale is always swizzled
        if is_sf_swizzled_layout or output_both_sf_layouts:
            # Swizzled layout: flattened with 128x4 tile pattern
            num_m_tiles = (batch_size + 127) // 128
            num_k_tiles = (num_sf_blocks_per_row + 3) // 4
            k_tile_stride = 512
            swizzled_size = num_m_tiles * num_k_tiles * k_tile_stride
            block_scale = torch.empty(
                (swizzled_size,), dtype=scale_dtype, device=input.device
            )
        else:
            if is_3d:
                block_scale = torch.empty(
                    (B, S, num_sf_blocks_per_row),
                    dtype=scale_dtype,
                    device=input.device,
                )
            else:
                block_scale = torch.empty(
                    (batch_size, num_sf_blocks_per_row),
                    dtype=scale_dtype,
                    device=input.device,
                )

    # Allocate unswizzled scale factor tensor
    # When output_both_sf_layouts=False, we still need a correctly-shaped tensor
    # for TVM-FFI validation, even though the kernel won't write to it
    if block_scale_unswizzled is None:
        if is_3d:
            block_scale_unswizzled = torch.empty(
                (B, S, num_sf_blocks_per_row),
                dtype=scale_dtype,
                device=input.device,
            )
        else:
            block_scale_unswizzled = torch.empty(
                (batch_size, num_sf_blocks_per_row),
                dtype=scale_dtype,
                device=input.device,
            )

    # Get 2D views for kernel
    if is_3d:
        y_fp4_2d = y_fp4.view(B * S, -1)
        block_scale_2d = (
            block_scale.view(B * S, -1)
            if not (is_sf_swizzled_layout or output_both_sf_layouts)
            else block_scale
        )
        # Always convert to 2D for kernel call
        block_scale_unswizzled_2d = block_scale_unswizzled.view(B * S, -1)
    else:
        y_fp4_2d = y_fp4
        block_scale_2d = block_scale
        block_scale_unswizzled_2d = block_scale_unswizzled

    # Create global_scale tensor if not provided (1.0 = no scaling)
    if global_scale is None:
        global_scale = torch.ones(1, dtype=torch.float32, device=input.device)

    tensor_api = _get_compiled_kernel(
        hidden_size,
        block_size,
        is_fp16,
        sm_version,
        actual_scale_format,
        is_sf_swizzled_layout,
        output_both_sf_layouts,
    )
    tensor_api(
        input_2d.contiguous(),
        residual_2d.contiguous(),
        weight.contiguous(),
        y_fp4_2d,
        block_scale_2d.view(torch.uint8),
        block_scale_unswizzled_2d.view(torch.uint8),
        global_scale.contiguous(),
        batch_size,
        eps,
    )

    if output_both_sf_layouts:
        return y_fp4, block_scale, block_scale_unswizzled
    else:
        return y_fp4, block_scale


__all__ = [
    "AddRMSNormFP4QuantKernel",
    "add_rmsnorm_fp4quant",
    "get_sm_version",
]
