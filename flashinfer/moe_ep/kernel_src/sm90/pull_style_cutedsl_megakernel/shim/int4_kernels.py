# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Signed INT4 weights on the packed-weight SM90 MegaMoE pipeline.

Metadata copies and register conversion specialize the shared kernel.
Float4E2M1 serves only as the four-bit storage type for LDSM.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync

from moe_hopper_fp8.kernel_mxfp4_fp8_glu_fc12_swapab import (
    Sm90SwapABSwigluMxfp4Fp8Fc12Kernel,
    _copy_offset_row_zfill,
)
from moe_hopper_fp8.megamoe_kernel_fp8 import Sm90MegaMoESwapABMxfp4Fp8Kernel

from .int4_conversion import (
    INT4_FOLD_BLOCK_BYTES,
    convert_int4_group,
    make_expanded_offset_view,
    make_expanded_offset_view_k256,
    make_offset_smem_layout,
    make_offset_smem_layout_k256,
)


class Sm90SwapABSwigluInt4Fp8Fc12Kernel(Sm90SwapABSwigluMxfp4Fp8Fc12Kernel):
    _fp4_format_name = "INT4"
    _folded_scale_rows = 8
    _folded_scale_block_bytes = INT4_FOLD_BLOCK_BYTES
    _make_scale_smem_layout = staticmethod(make_offset_smem_layout)
    _make_scale_smem_layout_k256 = staticmethod(make_offset_smem_layout_k256)
    _make_expanded_scale_view = staticmethod(make_expanded_offset_view)
    _make_expanded_scale_view_k256 = staticmethod(make_expanded_offset_view_k256)

    def __init__(self, *args, **kwargs):
        if kwargs.get("fp8_scale_mode", "per_tensor") != "mxfp4_hybrid":
            raise NotImplementedError("INT4 requires fp8_scale_mode='mxfp4_hybrid'")
        super().__init__(*args, **kwargs)

    def _weight_aux_smem_dtype(self):
        return cutlass.Uint64

    @cute.jit
    def _convert_packed_a_kblock(
        self,
        packed_registers,
        fp8_fragment_a,
        partitioned_scales,
        k_block: cutlass.Constexpr,
        stage_idx,
    ):
        # K256 issues K32 fragments in order, with a wait between K128 halves.
        # Convert each half once; retain the inherited fence/MMA sequence.
        if cutlass.const_expr(k_block % 4 == 0):
            convert_int4_group(
                packed_registers,
                fp8_fragment_a,
                partitioned_scales,
                k_block,
                k_block,
                4,
                stage_idx,
            )

    @cute.jit
    def _convert_weight_group(
        self,
        packed_registers,
        fp8_fragment_a,
        partitioned_offsets,
        local_begin: cutlass.Constexpr,
        global_begin: cutlass.Constexpr,
        count: cutlass.Constexpr,
        stage_idx,
        smem_weight_sf,
    ):
        convert_int4_group(
            packed_registers,
            fp8_fragment_a,
            partitioned_offsets,
            local_begin,
            global_begin,
            count,
            stage_idx,
        )

    @cute.jit
    def _copy_int4_scale_rows(
        self,
        weight_sf_gemm: cute.Tensor,
        smem_weight_sf: cute.Tensor,
        work_tile_info,
        scale_handle,
        tidx,
        k128_blocks: cutlass.Constexpr,
    ):
        lane = tidx % cutlass.Int32(32)
        m_blocks = self.mma_tiler[0] // 64
        vectors_per_m = 32 * k128_blocks
        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Uint64,
            num_bits_per_copy=128,
        )
        for item in cutlass.range_constexpr(m_blocks * k128_blocks):
            vector = lane + cutlass.Int32(item * 32)
            local_m = vector // cutlass.Int32(vectors_per_m)
            in_m = vector % cutlass.Int32(vectors_per_m)
            local_k = in_m // cutlass.Int32(32)
            flat = (in_m % cutlass.Int32(32)) * 2
            row = flat // cutlass.Int32(8)
            column = flat % cutlass.Int32(8)
            global_m = work_tile_info.tile_m_idx * cutlass.Int32(m_blocks) + local_m
            global_k = scale_handle.count * cutlass.Int32(k128_blocks) + local_k
            src = weight_sf_gemm.iterator + cute.crd2idx(
                (work_tile_info.expert_idx, global_m, global_k, row, column),
                weight_sf_gemm.layout,
            )
            dst = smem_weight_sf.iterator + cute.crd2idx(
                (column, row, local_m, local_k, scale_handle.index),
                smem_weight_sf.layout,
            )
            # Each vector contains two uint64 entries; modulo indexing hides
            # the guaranteed sixteen-byte alignment from the compiler.
            src_vec = cute.make_tensor(
                cute.make_ptr(src.dtype, src.toint(), src.memspace, assumed_align=16),
                cute.make_layout(2),
            )
            dst_vec = cute.make_tensor(
                cute.make_ptr(dst.dtype, dst.toint(), dst.memspace, assumed_align=16),
                cute.make_layout(2),
            )
            if cutlass.const_expr(self._weight_offset_m_padding_possible()):
                src_bytes = cutlass.Int32(16)
                if global_m >= weight_sf_gemm.shape[1]:
                    src_bytes = cutlass.Int32(0)
                _copy_offset_row_zfill(dst_vec.iterator, src_vec.iterator, src_bytes)
            else:
                cute.copy(copy_atom, src_vec, dst_vec)
        scale_handle.commit()

    @cute.jit
    def _copy_weight_scale_cpasync(
        self,
        weight_sf_gemm,
        smem_weight_sf,
        work_tile_info,
        output_scale_block_base,
        scale_handle,
        tidx,
    ):
        self._copy_int4_scale_rows(
            weight_sf_gemm, smem_weight_sf, work_tile_info, scale_handle, tidx, 1
        )

    @cute.jit
    def _copy_weight_scale_cpasync_k256_rows(
        self,
        weight_sf_gemm,
        smem_weight_sf,
        work_tile_info,
        output_scale_block_base,
        scale_handle,
        tidx,
    ):
        self._copy_int4_scale_rows(
            weight_sf_gemm, smem_weight_sf, work_tile_info, scale_handle, tidx, 2
        )


class Sm90MegaMoESwapABInt4Fp8Kernel(
    Sm90MegaMoESwapABMxfp4Fp8Kernel, Sm90SwapABSwigluInt4Fp8Fc12Kernel
):
    """Fused MegaMoE with signed INT4 weights and K128 register tables."""
