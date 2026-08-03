# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# mypy: disable-error-code="assignment, attr-defined, misc"

from typing import Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

from ._sm120_blockscaled_gemm import Sm120BlockScaledGemmKernel


def compute_zpq(
    dhw: Tuple[int, int, int],
    trs: Tuple[int, int, int],
    stride_dhw: Tuple[int, int, int],
    upper_padding_dhw: Tuple[int, int, int],
    lower_padding_dhw: Tuple[int, int, int],
    dilation_dhw: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    d, h, w = dhw
    t, r, s = trs
    sd, sh, sw = stride_dhw
    up_d, up_h, up_w = upper_padding_dhw
    lo_d, lo_h, lo_w = lower_padding_dhw
    dil_d, dil_h, dil_w = dilation_dhw
    z = ((d + up_d + lo_d - dil_d * (t - 1) - 1) // sd) + 1
    p = ((h + up_h + lo_h - dil_h * (r - 1) - 1) // sh) + 1
    q = ((w + up_w + lo_w - dil_w * (s - 1) - 1) // sw) + 1
    return z, p, q


class Sm120BlockScaledConv3dKernel(Sm120BlockScaledGemmKernel):
    """Narrow SM120 NVFP4 Conv3d descriptor and launch frontend.

    This prototype uses true im2col TMA for A and Conv filter/output layouts.
    It can either consume materialized logical GEMM SFA (baseline path), or a
    natural input-scale tensor exposed as a zero-stride logical NDHWC view and
    loaded through im2col TMA.
    """

    def __init__(
        self,
        acc_dtype,
        sf_vec_size,
        tile_shape_mnk,
        epi_tile,
        filter_trs: Tuple[int, int, int],
        upper_padding_dhw: Tuple[int, int, int],
        lower_padding_dhw: Tuple[int, int, int],
        stride_dhw: Tuple[int, int, int],
        dilation_dhw: Tuple[int, int, int],
        sfa_layout_mode: str = "materialized",
        use_conv_owned_kernel: bool = False,
        a_load_mode: str = "tma",
        a_copy_bits: int = 32,
        a_copy_layout_mode: str = "single",
        sfb_load_mode: str = "tma",
        epilogue_store_mode: str = "tma",
        output_z_override: int = 0,
        output_z_offset: int = 0,
        scale_exactn_fastpath: bool = True,
    ):
        valid_modes = {
            "sfa_layout_mode": {
                "materialized",
                "natural_cpasync",
                "natural_cpasync_inline",
                "compact_im2col_cpasync_inline",
                "natural_im2col",
                "natural_expanded_im2col",
            },
            "a_load_mode": {"tma"},
            "a_copy_layout_mode": {"single", "row", "coalesced"},
            "sfb_load_mode": {"tma", "cpasync_inline"},
            "epilogue_store_mode": {"tma"},
        }
        selected_modes = {
            "sfa_layout_mode": sfa_layout_mode,
            "a_load_mode": a_load_mode,
            "a_copy_layout_mode": a_copy_layout_mode,
            "sfb_load_mode": sfb_load_mode,
            "epilogue_store_mode": epilogue_store_mode,
        }
        for name, value in selected_modes.items():
            if value not in valid_modes[name]:
                raise ValueError(
                    f"{name} must be one of {sorted(valid_modes[name])}; got {value!r}"
                )

        super().__init__(acc_dtype, sf_vec_size, tile_shape_mnk, epi_tile)
        self.filter_trs = filter_trs
        self.upper_padding_dhw = upper_padding_dhw
        self.lower_padding_dhw = lower_padding_dhw
        self.stride_dhw = stride_dhw
        self.dilation_dhw = dilation_dhw
        self.sfa_layout_mode = sfa_layout_mode
        self.use_conv_owned_kernel = use_conv_owned_kernel
        self.a_load_mode = a_load_mode
        self.a_copy_bits = a_copy_bits
        self.a_copy_layout_mode = a_copy_layout_mode
        self.sfb_load_mode = sfb_load_mode
        self.epilogue_store_mode = epilogue_store_mode
        self.output_z_override = output_z_override
        self.output_z_offset = output_z_offset
        self.scale_exactn_fastpath = scale_exactn_fastpath
        self.sfa_cpasync_warp_start = self.tma_load_warp_id + 1
        self.sfa_cpasync_warp_count = 4
        self.sfa_cpasync_threads = (
            self.sfa_cpasync_warp_count * self.num_threads_per_warp
        )
        if sfa_layout_mode == "natural_cpasync":
            self.threads_per_cta = max(
                self.threads_per_cta,
                (self.sfa_cpasync_warp_start + self.sfa_cpasync_warp_count)
                * self.num_threads_per_warp,
            )

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

        # Conv tensors are NDHWC, KTRSC, NZPQK. After regrouping they become
        # GEMM-style A(M,K), B(N,K), C(M,N), all K/N contiguous as row-major.
        self.a_layout = utils.LayoutEnum.ROW_MAJOR
        self.b_layout = utils.LayoutEnum.ROW_MAJOR
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        self._setup_attributes()

        def add_dummy_batch_dimension(tensor):
            new_layout = cute.append(tensor.layout, cute.make_layout(1))
            return cute.make_tensor(tensor.iterator, new_layout)

        fast_1x1_gemm_path = (
            self.filter_trs == (1, 1, 1)
            and self.upper_padding_dhw == (0, 0, 0)
            and self.lower_padding_dhw == (0, 0, 0)
            and self.stride_dhw == (1, 1, 1)
            and self.dilation_dhw == (1, 1, 1)
        )

        # A: (N,D,H,W,C) -> ((Q/P/Z/N output rows), im2col K). The 1x1/no-pad
        # case is exactly GEMM A(M,C), so avoid the heavier im2col descriptor.
        mA = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[3, 2, 1, 0, 4]))
        mA = cute.group_modes(mA, begin=0, end=4)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        pad_upper_d, pad_upper_h, pad_upper_w = self.upper_padding_dhw
        pad_lower_d, pad_lower_h, pad_lower_w = self.lower_padding_dhw
        stride_d, stride_h, stride_w = self.stride_dhw
        dilation_d, dilation_h, dilation_w = self.dilation_dhw
        filter_t, filter_r, filter_s = self.filter_trs
        if cutlass.const_expr(fast_1x1_gemm_path):
            mA = cute.make_tensor(
                a.iterator,
                cute.make_layout(
                    (cute.size(mA, mode=[0]), cute.size(mA, mode=[1])),
                    stride=(cute.size(mA, mode=[1]), 1),
                ),
            )
            mA = add_dummy_batch_dimension(mA)
            tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
                mA,
                self.a_smem_layout_staged,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                1,
                internal_type=self.tma_internal_a_dtype,
            )
        else:
            tma_atom_a, tma_tensor_a = cpasync.make_im2col_tma_atom(
                cpasync.CopyBulkTensorIm2ColG2SOp(),
                mA,
                a_smem_layout,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                lower_corner_whd=(-pad_lower_w, -pad_lower_h, -pad_lower_d),
                upper_corner_whd=(
                    pad_upper_w - ((filter_s - 1) * dilation_w),
                    pad_upper_h - ((filter_r - 1) * dilation_h),
                    pad_upper_d - ((filter_t - 1) * dilation_d),
                ),
                lower_padding_whd=(pad_lower_w, pad_lower_h, pad_lower_d),
                upper_padding_whd=(pad_upper_w, pad_upper_h, pad_upper_d),
                stride_whd=(stride_w, stride_h, stride_d),
                lower_srt=(0, 0, 0),
                stride_srt=(dilation_w, dilation_h, dilation_d),
                internal_type=self.tma_internal_a_dtype,
            )
            tma_tensor_a = add_dummy_batch_dimension(tma_tensor_a)

        # B: filter (K,T,R,S,C) is already compact as GEMM B(K, T*R*S*C)
        # in memory. The flat K order is C-fastest, then S/R/T, matching A's
        # im2col traversal. Use an explicit compact layout instead of a
        # composed select/group view so regular TMA sees a dense GEMM operand.
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
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            mB,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
            internal_type=self.tma_internal_b_dtype,
        )

        # C: output (N,Z,P,Q,K) -> ((Q,P,Z,N), K).  The optional Z override
        # and Z offset are a staging hook for Wan D=4 temporal split: a T=1
        # sub-kernel can write directly into a slice of a larger final output.
        c_iterator = c.iterator
        conv_n = cute.size(c, mode=[0])
        conv_z_storage = cute.size(c, mode=[1])
        conv_p = cute.size(c, mode=[2])
        conv_q = cute.size(c, mode=[3])
        conv_k = cute.size(c, mode=[4])
        conv_z = conv_z_storage
        if cutlass.const_expr(self.output_z_override != 0):
            conv_z = self.output_z_override
        if cutlass.const_expr(self.output_z_offset != 0):
            c_iterator = c.iterator + self.output_z_offset * conv_p * conv_q * conv_k
        mC = cute.make_tensor(c_iterator, cute.select(c.layout, mode=[3, 2, 1, 0, 4]))
        mC = cute.group_modes(mC, begin=0, end=4)
        mC = add_dummy_batch_dimension(mC)

        conv_c = cute.size(b, mode=[4])
        conv_t, conv_r, conv_s = self.filter_trs
        gemm_m = conv_n * conv_z * conv_p * conv_q
        gemm_k = conv_c * conv_t * conv_r * conv_s
        if cutlass.const_expr(fast_1x1_gemm_path or conv_n == 1):
            mC = cute.make_tensor(
                c_iterator,
                cute.make_layout((gemm_m, conv_k), stride=(conv_k, 1)),
            )
            mC = add_dummy_batch_dimension(mC)

        if cutlass.const_expr(
            self.sfa_layout_mode
            in (
                "natural_cpasync",
                "natural_cpasync_inline",
                "compact_im2col_cpasync_inline",
            )
        ):
            # Natural cp.async path: compact SFA storage is (N,D,H,W,C/sf_vec).
            # The Conv-owned kernel maps output/filter coordinates to this
            # compact tensor and writes the SM120 SFA SMEM layout directly.
            mSFA = cute.make_tensor(sfa.iterator, sfa.layout)
            if cutlass.const_expr(
                self.sfa_layout_mode in ("natural_cpasync", "natural_cpasync_inline")
            ):
                mSFA = cute.group_modes(mSFA, begin=0, end=4)
            tma_atom_sfa = None
            tma_tensor_sfa = add_dummy_batch_dimension(mSFA)
        elif cutlass.const_expr(self.sfa_layout_mode == "natural_im2col"):
            # Natural SFA input is compact (N,D,H,W,ceil(C/sf_vec)). Build a
            # logical im2col source view ((W,H,D,N),(sf_vec,ceil(C/sf_vec)))
            # with stride 0 in the sf_vec submode, so it matches A's logical C
            # traversal without materializing per-channel scale values.
            sfa_spatial_layout = cute.select(sfa.layout, mode=[3, 2, 1, 0])
            sfa_channel_layout = cute.make_layout(
                (self.sf_vec_size, cute.size(sfa, mode=[4])),
                stride=(0, sfa.stride[4]),
            )
            sfa_im2col_layout = cute.make_layout(
                (sfa_spatial_layout.shape, sfa_channel_layout.shape),
                stride=(sfa_spatial_layout.stride, sfa_channel_layout.stride),
            )
            mSFA = cute.make_tensor(sfa.iterator, sfa_im2col_layout)
            sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, 0))
            tma_atom_sfa, tma_tensor_sfa = cpasync.make_im2col_tma_atom(
                cpasync.CopyBulkTensorIm2ColG2SOp(),
                mSFA,
                sfa_smem_layout,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                lower_corner_whd=(-pad_lower_w, -pad_lower_h, -pad_lower_d),
                upper_corner_whd=(
                    pad_upper_w - ((filter_s - 1) * dilation_w),
                    pad_upper_h - ((filter_r - 1) * dilation_h),
                    pad_upper_d - ((filter_t - 1) * dilation_d),
                ),
                lower_padding_whd=(pad_lower_w, pad_lower_h, pad_lower_d),
                upper_padding_whd=(pad_upper_w, pad_upper_h, pad_upper_d),
                stride_whd=(stride_w, stride_h, stride_d),
                lower_srt=(0, 0, 0),
                stride_srt=(dilation_w, dilation_h, dilation_d),
                internal_type=cutlass.Int16,
            )
            tma_tensor_sfa = add_dummy_batch_dimension(tma_tensor_sfa)
        elif cutlass.const_expr(self.sfa_layout_mode == "natural_expanded_im2col"):
            # Experimental bridge: natural SFA is stored as expanded
            # (N,D,H,W,C), with identical values repeated inside each sf_vec
            # group. This avoids a zero-stride global SFA layout while keeping
            # the same im2col coordinate mapping as A.
            mSFA = cute.make_tensor(
                sfa.iterator, cute.select(sfa.layout, mode=[3, 2, 1, 0, 4])
            )
            mSFA = cute.group_modes(mSFA, begin=0, end=4)
            sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, 0))
            tma_atom_sfa, tma_tensor_sfa = cpasync.make_im2col_tma_atom(
                cpasync.CopyBulkTensorIm2ColG2SOp(),
                mSFA,
                sfa_smem_layout,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                lower_corner_whd=(-pad_lower_w, -pad_lower_h, -pad_lower_d),
                upper_corner_whd=(
                    pad_upper_w - ((filter_s - 1) * dilation_w),
                    pad_upper_h - ((filter_r - 1) * dilation_h),
                    pad_upper_d - ((filter_t - 1) * dilation_d),
                ),
                lower_padding_whd=(pad_lower_w, pad_lower_h, pad_lower_d),
                upper_padding_whd=(pad_upper_w, pad_upper_h, pad_upper_d),
                stride_whd=(stride_w, stride_h, stride_d),
                lower_srt=(0, 0, 0),
                stride_srt=(dilation_w, dilation_h, dilation_d),
                internal_type=cutlass.Int16,
            )
            tma_tensor_sfa = add_dummy_batch_dimension(tma_tensor_sfa)
        else:
            # Baseline path: materialized logical GEMM scale tensor.
            sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (gemm_m, gemm_k, 1), self.sf_vec_size
            )
            sfa_tensor = cute.make_tensor(sfa.iterator, sfa_layout)
            tma_atom_sfa, tma_tensor_sfa = self._make_tma_atoms_and_tensors(
                sfa_tensor,
                self.sfa_smem_layout_staged,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                1,
                internal_type=cutlass.Int16,
            )

        if cutlass.const_expr(self.sfb_load_mode == "cpasync_inline"):
            # Experimental C96 path: SFB is supplied as compact
            # (padded-output-channel, padded-scale-group) storage and staged
            # directly into the normal SFB SMEM layout by the TMA load warp.
            tma_atom_sfb = tma_atom_b
            tma_tensor_sfb = sfb
        else:
            # Baseline path: materialized logical GEMM SFB loaded by TMA.
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (conv_k, gemm_k, 1), self.sf_vec_size
            )
            sfb_tensor = cute.make_tensor(sfb.iterator, sfb_layout)

            tma_atom_sfb, tma_tensor_sfb = self._make_tma_atoms_and_tensors(
                sfb_tensor,
                self.sfb_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
                1,
                internal_type=cutlass.Int16,
            )
        if cutlass.const_expr(
            self.sfa_layout_mode
            in (
                "natural_cpasync",
                "natural_cpasync_inline",
                "compact_im2col_cpasync_inline",
            )
        ):
            # The cp.async path does not use an SFA TMA descriptor. Pass a real
            # copy atom to keep the kernel signature uniform; all SFA TMA uses
            # are compile-time skipped for this mode.
            tma_atom_sfa = tma_atom_b

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
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
        self.threads_per_cta = (self.threads_per_cta + 128) // 128 * 128

        if cutlass.const_expr(self.use_conv_owned_kernel):
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
                conv_t,
                conv_r,
                conv_s,
                stride_d,
                stride_h,
                stride_w,
                dilation_d,
                dilation_h,
                dilation_w,
                pad_lower_d,
                pad_lower_h,
                pad_lower_w,
                cute.size(a, mode=[1]),
                cute.size(a, mode=[2]),
                cute.size(a, mode=[3]),
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
        else:
            self.kernel(
                tma_atom_a,
                tma_tensor_a,
                tma_atom_b,
                tma_tensor_b,
                tma_atom_sfa,
                tma_tensor_sfa,
                tma_atom_sfb,
                tma_tensor_sfb,
                tma_atom_c,
                tma_tensor_c,
                self.tiled_mma,
                self.cta_layout_mnk,
                self.a_smem_layout_staged,
                self.b_smem_layout_staged,
                self.sfa_smem_layout_staged,
                self.sfb_smem_layout_staged,
                self.epi_smem_layout_staged,
                tile_sched_params,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=[1, 1, 1],
                stream=stream,
                max_number_threads=[self.threads_per_cta, 1, 1],
                min_blocks_per_mp=1,
            )
