# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# mypy: disable-error-code="assignment, attr-defined, misc"

"""Descriptor construction and launch for the SM120 NVFP4 Conv3d kernel."""

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

from flashinfer.cute_dsl.sm120_blockscaled import (
    make_sm120_tma_load_atom_and_tensor,
    make_sm120_tma_store_atom_and_tensor,
)

from ._sm120_nvfp4_base import Sm120Nvfp4KernelBase


class Sm120Nvfp4Conv3dDescriptor(Sm120Nvfp4KernelBase):
    """Build TMA descriptors for the specialized physical-halo Conv3d."""

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        a_zero: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.sf_dtype = sfa.element_type

        # NDHWC, KTRSC, and NZPQK all expose their GEMM-contiguous dimension
        # as the innermost physical dimension.
        self.a_layout = utils.LayoutEnum.ROW_MAJOR
        self.b_layout = utils.LayoutEnum.ROW_MAJOR
        self.c_layout = utils.LayoutEnum.ROW_MAJOR
        self._setup_attributes()

        def add_dummy_batch_dimension(tensor):
            new_layout = cute.append(tensor.layout, cute.make_layout(1))
            return cute.make_tensor(tensor.iterator, new_layout)

        input_d = cute.size(a, mode=[1])
        input_h = cute.size(a, mode=[2])
        input_w = cute.size(a, mode=[3])
        conv_n = cute.size(c, mode=[0])
        conv_z = cute.size(c, mode=[1])
        conv_p = cute.size(c, mode=[2])
        conv_q = cute.size(c, mode=[3])
        conv_k = cute.size(c, mode=[4])
        conv_c = cute.size(b, mode=[4])
        gemm_m = conv_n * conv_z * conv_p * conv_q
        gemm_k = conv_c * 27

        # A is already converted to physical-halo NDHWC by the fused activation
        # producer. The descriptor performs only the 3x3x3 im2col traversal.
        mA = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[3, 2, 1, 0, 4]))
        mA = cute.group_modes(mA, begin=0, end=4)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a = cpasync.make_im2col_tma_atom(
            cpasync.CopyBulkTensorIm2ColG2SOp(),
            mA,
            a_smem_layout,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            lower_corner_whd=(0, 0, 0),
            upper_corner_whd=(-2, -2, -2),
            lower_padding_whd=(0, 0, 0),
            upper_padding_whd=(0, 0, 0),
            stride_whd=(1, 1, 1),
            lower_srt=(0, 0, 0),
            stride_srt=(1, 1, 1),
        )
        tma_tensor_a = add_dummy_batch_dimension(tma_tensor_a)

        # KTRSC is compact in the same C/S/R/T reduction order as im2col A.
        mB = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[0, 4, 3, 2, 1]))
        mB = cute.group_modes(mB, begin=1, end=5)
        mB = cute.make_tensor(
            b.iterator,
            cute.make_layout(
                (cute.size(mB, mode=[0]), cute.size(mB, mode=[1])),
                stride=(cute.size(mB, mode=[1]), 1),
            ),
        )
        mB = add_dummy_batch_dimension(mB)
        tma_atom_b, tma_tensor_b = make_sm120_tma_load_atom_and_tensor(
            mB,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
        )

        # Public validation requires batch one, so NZPQK is a compact MxK view.
        mC = cute.make_tensor(
            c.iterator,
            cute.make_layout((gemm_m, conv_k), stride=(conv_k, 1)),
        )
        mC = add_dummy_batch_dimension(mC)

        # Compact activation scales stay in physical-halo NDHWC16 storage. The
        # DMA warp maps them directly into the staged SFA MMA layout.
        mSFA = cute.make_tensor(sfa.iterator, sfa.layout)
        mSFA = cute.group_modes(mSFA, begin=0, end=4)
        tma_tensor_sfa = add_dummy_batch_dimension(mSFA)

        # Prepared weight scales already follow the logical GEMM SFB layout.
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            (conv_k, gemm_k, 1),
            self.sf_vec_size,
        )
        sfb_tensor = cute.make_tensor(sfb.iterator, sfb_layout)
        tma_atom_sfb, tma_tensor_sfb = make_sm120_tma_load_atom_and_tensor(
            sfb_tensor,
            self.sfb_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            internal_type=cutlass.Int16,
        )

        # SFA uses cp.async in the Conv-owned mainloop. A valid copy atom keeps
        # the uniform kernel signature without constructing an unused SFA TMA.
        tma_atom_sfa = tma_atom_b
        tma_atom_c, tma_tensor_c = make_sm120_tma_store_atom_and_tensor(
            mC,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )
        tile_sched_params, grid = self._compute_grid(
            mC,
            self.tile_shape_mnk,
            max_active_clusters,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sfa_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            math_wg_order_barrier_array_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.smem_alloc_a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.smem_alloc_b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.threads_per_cta = (self.threads_per_cta + 127) // 128 * 128

        self.conv_kernel(
            tma_atom_a,
            tma_tensor_a,
            a,
            a_zero,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            mC,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
            3,
            3,
            3,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            input_d,
            input_h,
            input_w,
            conv_z,
            conv_p,
            conv_q,
            conv_n,
            conv_c,
            self.tile_shape_mnk[2],
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            max_number_threads=[self.threads_per_cta, 1, 1],
            min_blocks_per_mp=1,
        )


__all__ = ["Sm120Nvfp4Conv3dDescriptor"]
