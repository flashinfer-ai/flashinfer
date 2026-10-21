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

"""Dense SM120 GEMM device implementation."""

from b12x._lib.quant.block_codec import BLOCK_CODECS
from b12x._lib.fp6 import FLOAT6_E2M3_MAX
from b12x._lib.fp6 import FLOAT6_E3M2_MAX
from b12x._lib.intrinsics.constants import FLOAT8_E4M3_MAX
from cutlass import Int32
from cutlass import Int64
from typing import Literal
from typing import Optional
from typing import Tuple
from cutlass import Uint32
from cutlass import Uint64
from cutlass import Uint8
from cutlass.cute.nvgpu.warp.mma import Field as WarpField
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo
from b12x._lib.intrinsics.packed import bfloat2_to_float2_scaled
from b12x._lib.quant.block_codec import block_codec
import cutlass.utils.blockscaled_layout as blockscaled_utils
from b12x._lib.intrinsics.memory import cp_async4_shared_global
from b12x._lib.intrinsics.memory import cp_async_bulk_g2s_mbar
from cutlass.cute.nvgpu import cpasync
import cuda.bindings.driver as cuda
import cutlass.cute as cute
import cutlass
from b12x._lib.fp6 import cvt_f32_to_e2m3x2
from b12x._lib.fp6 import cvt_f32_to_e3m2x2
from b12x._lib.fp6 import cvt_f32_to_e4m3x2
from b12x._lib.intrinsics.conversion import cvt_f32x4_to_e4m3x4
from b12x._lib.intrinsics.memory import elem_pointer
from b12x._lib.dense_gemm_mxfp6 import emit_mxfp6_dense_mma_k_block
from b12x._lib.intrinsics.math import fabs_f32
from b12x._lib.intrinsics.math import fmax_f32
from b12x._lib.fp6 import fp6_block_ue8m0_exact
from b12x._lib.intrinsics.memory import get_ptr_as_int64
from b12x._lib.intrinsics.trellis import iq2_xs_descriptor_pair_to_bf16x2x2
from b12x._lib.intrinsics.trellis import iq2_xxs_descriptor_pair
from b12x._lib.utils.fp6_geometry import is_mxfp6_ab_dtype
from b12x._lib.intrinsics.memory import ld_global_b16
from b12x._lib.intrinsics.memory import ld_global_v4_u32
from b12x._lib.fp6 import mx_gs_numerator
from b12x._lib.utils.fp6_geometry import mxfp6_tile_k
from b12x._lib.intrinsics.packed import mxfp8_pair_to_bf16x2_sm120
from b12x._lib.intrinsics.packed import nvfp4_pair_to_bf16x2_sm120
import cutlass.pipeline as pipeline
from b12x._lib.intrinsics.conversion import pow2_ceil_ue8m0
from b12x._lib.intrinsics.trellis import q8_0_pair_to_bf16x2
from b12x._lib.intrinsics.quantization import quantize_block_fp8_mx
from b12x._lib.intrinsics.atomics import scatter_add_bf16
from b12x._lib.intrinsics.atomics import scatter_add_bf16x2
from b12x._lib.intrinsics.memory import shared_ptr_to_u32
from b12x._lib.utils.scale_layout import sm120_make_smem_layout_sfa
from b12x._lib.utils.scale_layout import sm120_make_smem_layout_sfb
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.hopper_helpers as sm90_utils
from b12x._lib.intrinsics.memory import st_global_u64
from b12x._lib.intrinsics.memory import st_shared_u16
from b12x._lib.intrinsics.memory import st_shared_u8
from b12x._lib.intrinsics.memory import u32_as_f32
from b12x._lib.fp6 import ue8m0_output_scale_exact
from b12x._lib.intrinsics.conversion import ue8m0_to_output_scale
import cutlass.utils as utils
from b12x._lib.intrinsics.reduction import warp_reduce
from b12x._lib.dense_gemm.config import (
    _DENSE_FUSED_QUANT,
    _DENSE_LOAD_PATHS,
    _PACKED_B_EXPAND_AHEAD,
    _emit_plain_fp8_dense_mma_k_block,
    _expand_packed_b_stage_smem,
    _reshape_acc_to_mn,
)


class DenseGemmKernel:
    """Implements batched matrix multiplication (C = A x SFA x B x SFB) for
    Blackwell GeForce architecture using warp-level MMA.

    Key architectural differences from the tcgen05 donor path:
    - No TMEM, no tcgen05, no 2-CTA instructions, no multi-cluster
    - Warp-level MMA: MmaMXF4NVF4Op atom m16n8k64, atom_layout=(4,2,1)
    - 256 MMA threads + 32 DMA = 288 total threads
    - PipelineTmaAsync (not PipelineTmaUmma)
    - Manual atom unroll workaround for CuTe DSL compiler SF address space bug
    - Cluster shape always (1,1,1)

    Notes:
        - Supported combinations:
            * NVF4: A/B: Float4E2M1FN, SF: Float8E4M3FN, sf_vec_size: 16
            * MXF4: A/B: Float4E2M1FN, SF: Float8E8M0FNU, sf_vec_size: 32
            * MXFP8: A/B: Float8E4M3FN, SF: Float8E8M0FNU, sf_vec_size: 32
            * MX-FP6: A/B: Float6E3M2FN or Float6E2M3FN codes carried in
              Float8E4M3FN byte-containers, SF: Float8E8M0FNU, sf_vec_size: 32
              (inline ``mxf8f6f4`` MMA, m16n8k32; see ``mxfp6_fmt_a/b``)
        - Tile shape constraints:
            * tile_m must be divisible by 128
            * tile_n must be divisible by 128
            * tile_k must be divisible by 64
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        mma_k: int = 64,
        tile_k: Optional[int] = None,
        single_work_tile_per_cta: bool = False,
        use_prefetch: bool = False,
        direct_one_m_tile_scheduler: bool = False,
        split_k_slices: int = 1,
        split_k_atomic_bf16: bool = False,
        large_m_unroll: bool = False,
        use_m1_non_tma_a: bool = False,
        use_m1_non_tma_c: bool = False,
        use_m1_non_tma_sfa: bool = False,
        load_path: Literal["tma", "cpasync"] = "tma",
        swap_ab: bool = False,
        sfb_k_reuse: bool = False,
        fused_quant_a: bool = False,
        fused_quant_a_inner_span: int = 0,
        fused_quant_a_row_stride: int = 0,
        fused_quant_a_l_stride: int = 0,
        fused_quant_a_inv_rope: bool = False,
        fused_quant_a_head_dim: int = 0,
        fused_quant_a_nope_dim: int = 0,
        fused_quant_a_rope_dim: int = 0,
        fused_quant_a_wide: bool = False,
        atom_shape_24: bool = False,
        b_tile_major: bool = False,
        quantize_c: bool = False,
        alpha_is_one: bool = False,
        row_scale: bool = False,
        direct_sfa_live16: bool = False,
        direct_m1_wo_a_inputs: bool = False,
        target_occupancy: int = 1,
        mxfp6_fmt: Optional[str] = None,
        mxfp6_fmt_a: Optional[str] = None,
        mxfp6_fmt_b: Optional[str] = None,
        b_packed: bool = False,
        plain_fp8: bool = False,
        fused_quant_bf16: Optional[bool] = None,
        block_fp8: bool = False,
        weight_only: Optional[str] = None,
        mma_warps: Optional[int] = None,
        alpha_reciprocal: bool = False,
    ):
        if weight_only not in (None, "nvfp4", "mxfp8", "iq2_xs", "iq2_xxs", "q8_0"):
            raise ValueError("weight_only must be nvfp4, mxfp8, or iq2_xs")
        if mma_warps is not None and not (
            type(mma_warps) is int
            and mma_warps in (2, 4)
            and weight_only == "mxfp8"
            and mma_tiler_mn in ((16, 64), (16, 128))
        ):
            raise ValueError("explicit compute warps require MXFP8 A16 M16 N64/N128")
        self.weight_only = weight_only
        self.a16 = weight_only is not None
        self.a16_fp4 = weight_only == "nvfp4"
        self.a16_block = weight_only in BLOCK_CODECS
        self.a16_iq2_xxs = weight_only == "iq2_xxs"
        self.a16_q8 = weight_only == "q8_0"
        self.a16_lut_bytes = (
            block_codec(weight_only).lut_bytes() if self.a16_block else 0
        )
        self.a16_block_transposed = self.a16_block and swap_ab
        self.a16_pack_factor = (
            block_codec(weight_only).pack_factor
            if self.a16_block
            else 2
            if self.a16_fp4
            else 1
        )
        self.alpha_reciprocal = bool(alpha_reciprocal)
        if self.a16:
            if (
                load_path != "tma"
                or (swap_ab and not self.a16_block_transposed)
                or fused_quant_a
                or plain_fp8
                or block_fp8
                or b_packed
                or quantize_c
                or sfb_k_reuse
            ):
                raise ValueError("A16 requires ordinary K-major TMA operands")
            if mma_k != 16 or sf_vec_size != (16 if self.a16_fp4 else 32):
                raise ValueError(
                    "A16 requires BF16 K16 MMA and matching weight scale groups"
                )
            if self.a16_block_transposed and (
                mma_tiler_mn[0] != 8 or split_k_slices != 1
            ):
                raise ValueError(
                    "transposed IQ2_XS requires an eight-row tile without split-K"
                )
        # When set, A/B operands are MX codes carried in Float8E4M3FN
        # byte-containers: the whole kernel runs the MXFP8 smem/TMA/ldmatrix
        # machinery, and only the mainloop MMA is emitted as the inline
        # ``mxf8f6f4`` instruction (cutlass has no working 6-bit smem layout).
        # ``mxfp6_fmt_a`` / ``mxfp6_fmt_b`` may differ (W6A8: e4m3 acts, e2m3
        # weights). A single ``mxfp6_fmt`` still means both operands match.
        if mxfp6_fmt_a is None and mxfp6_fmt_b is None:
            mxfp6_fmt_a = mxfp6_fmt
            mxfp6_fmt_b = mxfp6_fmt
        elif mxfp6_fmt_a is None or mxfp6_fmt_b is None:
            raise ValueError(
                "mxfp6_fmt_a and mxfp6_fmt_b must both be set or both None"
            )
        self.mxfp6_fmt_a = mxfp6_fmt_a
        self.mxfp6_fmt_b = mxfp6_fmt_b
        self.block_fp8 = bool(block_fp8)
        self.plain_fp8 = bool(plain_fp8 or block_fp8)
        if self.plain_fp8:
            assert mxfp6_fmt_a is None
            assert not b_packed
            assert not fused_quant_a
            assert not quantize_c
        if self.block_fp8:
            assert sf_vec_size == 128
            assert (tile_k or 128) == 128
            assert load_path == "tma"
            assert not swap_ab
            assert not sfb_k_reuse
            assert not b_tile_major
        if mxfp6_fmt_a is not None:
            # Upstream mainloop variants not wired for the FP6 byte-container
            # path; fail loudly instead of silently miscomputing.
            assert load_path == "tma", (
                "MX-FP6 is only wired for the TMA load path, not "
                f"load_path={load_path!r}"
            )
            assert not swap_ab, "MX-FP6 is not wired for swap_ab"
            assert split_k_slices == 1, "MX-FP6 is not wired for split-K"
            assert not fused_quant_a, (
                "MX-FP6 uses its own fused activation-quant prologue, not the "
                "MXFP8 fused_quant_a machinery"
            )
            assert not b_tile_major, "MX-FP6 is not wired for tile-major B"
            assert not quantize_c, "MX-FP6 is not wired for quantize_c"
            assert not sfb_k_reuse, "MX-FP6 is not wired for sfb_k_reuse"
        # Native packed-FP6 streaming: B arrives 3:4-packed ``(N, 3K/4, L)`` in
        # gmem, TMA stages the packed tile into a plain smem buffer, and the MMA
        # warps expand it into the swizzled byte-container sB right after
        # consumer_wait. Cuts B HBM traffic by 25% vs the byte-container layout;
        # the ldmatrix/MMA path is unchanged.
        assert not b_packed or mxfp6_fmt_b is not None, (
            "b_packed requires the MX-FP6 path (mxfp6_fmt_b set)"
        )
        # The in-kernel expansion computes swizzled sB addresses assuming the
        # 8-bit K-major SW128 atom (8x128 = 1024 B): physical =
        # flat ^ ((row & 7) << 4). That atom is selected iff the smem major
        # size is 128, i.e. tile_k == 128 (always true for MX-FP6).
        assert not b_packed or (tile_k or sf_vec_size * 8) == 128, (
            "b_packed expansion assumes tile_k == 128 (SW128 smem atom)"
        )
        self.b_packed = b_packed
        # The fused flag changes code generation and must be explicit in every
        # memoized launch key. The environment is only a direct-construction
        # fallback.
        if fused_quant_bf16 is None:
            fused_quant_bf16 = _DENSE_FUSED_QUANT
        self.a_bf16_fused = bool(fused_quant_bf16) and use_m1_non_tma_a
        if self.a_bf16_fused:
            _fused_fmt = mxfp6_fmt_a or "e4m3"
            self._fused_gs_num = mx_gs_numerator(_fused_fmt)
            self._fused_act_fmt = _fused_fmt
            self._fused_fmt_max = {
                "e4m3": FLOAT8_E4M3_MAX,
                "e3m2": FLOAT6_E3M2_MAX,
                "e2m3": FLOAT6_E2M3_MAX,
            }[_fused_fmt]
        else:
            self._fused_gs_num = 0.0
            self._fused_act_fmt = ""
            self._fused_fmt_max = 0.0
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.mma_k = mma_k
        if tile_k is None:
            tile_k = sf_vec_size * 8
        self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], tile_k)
        self.manual_bk64_sf = not self.a16 and sf_vec_size == 32 and tile_k == 64
        self.mma_tile_shape_mnk = (
            (mma_tiler_mn[1], mma_tiler_mn[0], tile_k)
            if swap_ab
            else self.tile_shape_mnk
        )
        self.sfa_tile_shape_mk = (max(128, mma_tiler_mn[0]), tile_k)
        self.sfa_tiles_per_block = self.sfa_tile_shape_mk[0] // mma_tiler_mn[0]
        self.sfb_tile_shape_nk = (max(128, mma_tiler_mn[1]), tile_k)
        self.sfb_tiles_per_block = self.sfb_tile_shape_nk[0] // mma_tiler_mn[1]
        self.cluster_shape_mnk = (1, 1, 1)  # Always (1,1,1) on the current target
        self.epi_tile = mma_tiler_mn
        self.single_work_tile_per_cta = single_work_tile_per_cta
        self.use_prefetch = use_prefetch
        self.direct_one_m_tile_scheduler = direct_one_m_tile_scheduler
        self.split_k_slices = split_k_slices
        self.split_k_atomic_bf16 = split_k_atomic_bf16
        self.large_m_unroll = large_m_unroll
        self.use_m1_non_tma_a = use_m1_non_tma_a
        self.use_m1_non_tma_c = use_m1_non_tma_c
        self.use_m1_non_tma_sfa = use_m1_non_tma_sfa
        self.load_path = load_path
        self.swap_ab = swap_ab
        # SFB bytes are k-replicated within a 128-wide k tile (128x128 block
        # weight scales expanded to per-32): load one byte per stage and feed
        # every k block from it.
        self.sfb_k_reuse = sfb_k_reuse
        self.fused_quant_a = fused_quant_a
        # When >0, the BF16 A source is stored L-blocked along K (physical
        # [K/span, M, span], e.g. the WO tmp group-major view over [groups, M,
        # rank]): flat k = outer * span + inner reads element
        # outer * (M * span) + row * span + inner. 0 keeps contiguous [M, K].
        self.fused_quant_a_inner_span = fused_quant_a_inner_span
        # Grouped (L>1) BF16 A source, e.g. WO-A reading attention output
        # [M, groups, group_width] flat rows: element offset is
        # row * row_stride + l * l_stride + k (both strides in elements;
        # row_stride 0 keeps the contiguous shape[1] row pitch).
        self.fused_quant_a_row_stride = fused_quant_a_row_stride
        self.fused_quant_a_l_stride = fused_quant_a_l_stride
        # Inverse-RoPE applied in the quantizing A load: the trailing rope_dim
        # of every head_dim block is de-rotated with cos/sin at positions[row]
        # before MXFP8 quantization (head_dim/nope_dim aligned to 32-value
        # scale blocks; adjacent-pair rotation stays inside one load).
        self.fused_quant_a_inv_rope = fused_quant_a_inv_rope
        self.fused_quant_a_head_dim = fused_quant_a_head_dim
        self.fused_quant_a_nope_dim = fused_quant_a_nope_dim
        self.fused_quant_a_rope_dim = fused_quant_a_rope_dim
        # M=1 layout: 4 lanes per 32-value scale block (16 active lanes per
        # 128-wide k tile) instead of one, cutting the DMA-warp quantization
        # latency that serializes deep-K small-N pipelines.
        self.fused_quant_a_wide = fused_quant_a_wide
        self.b_tile_major = b_tile_major
        self.quantize_c = quantize_c
        self.alpha_is_one = alpha_is_one
        # Per-row output scale applied in the epilogue, replacing a separate
        # ``result.mul_(inv_gs)`` launch. It is NOT foldable into ``alpha``:
        # the eager multiply rounds to c_dtype twice (once after alpha, once
        # after the row scale), and the epilogue must reproduce both roundings
        # to stay bit-identical. See the epilogue application site.
        self.row_scale = row_scale
        if row_scale:
            # Each of these takes a store path that bypasses the r2s register
            # stage where the row scale is applied.
            assert not quantize_c, "row_scale is not wired for quantize_c"
            assert split_k_slices == 1, "row_scale is not wired for split-K"
            assert not swap_ab, "row_scale is not wired for swap_ab"
        self.direct_sfb_representative = (
            sfb_k_reuse
            and b_tile_major
            and (
                (
                    not fused_quant_a
                    and self.tile_shape_mnk in ((16, 64, 128), (32, 64, 128))
                )
                or (fused_quant_a and self.tile_shape_mnk == (16, 128, 128))
            )
        )
        self.direct_m1_wo_a_inputs = direct_m1_wo_a_inputs
        # Exact B16 consumes only rows 0-15 of each 128-row SFA atom. Those
        # rows are the contiguous first 256 bytes in both the packed global
        # and shared-memory layouts.
        self.direct_sfa_prefix = direct_sfa_live16 and self.direct_sfb_representative
        mma_atom_mn = (self.mma_tile_shape_mnk[0], self.mma_tile_shape_mnk[1])
        small_m_warps = (
            mma_warps
            if mma_warps is not None
            else 8
            if self.a16_block
            else 4
            if weight_only == "mxfp8"
            else 2
        )
        if self.a16_block_transposed:
            self.atom_shape = (mma_atom_mn[0] // 16, 1, 1)
        elif mma_atom_mn in ((16, 64), (16, 128)):
            # This table sets the MMA atom tiling only. The warp count is a
            # SEPARATE table below and both must list the same tiles: an atom
            # shape covering two warps of work under a launch geometry sized
            # for eight leaves warps 2-7 with no valid tile, reading shared
            # memory past the end of the staged operands.
            self.atom_shape = (1, small_m_warps, 1)
        elif mma_atom_mn in ((32, 64), (32, 128)):
            self.atom_shape = (2, 2, 1)
        elif atom_shape_24:
            self.atom_shape = (2, 4, 1)
        else:
            self.atom_shape = (4, 2, 1)

        self.tiled_mma = None
        self.occupancy = target_occupancy
        if self.a16_block_transposed:
            self.num_mma_warps = mma_atom_mn[0] // 16
        elif mma_atom_mn in ((16, 64), (16, 128)):
            self.num_mma_warps = small_m_warps
        elif mma_atom_mn in ((32, 64), (32, 128)):
            self.num_mma_warps = 4
        else:
            self.num_mma_warps = 8
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + 1  # 1 warp for DMA
        ) * self.num_threads_per_warp

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None

        self.buffer_align_bytes = 1024

        self.mma_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        mma_sf_dtype = cutlass.Float8E8M0FNU if self.block_fp8 else self.sf_dtype
        if self.a16:
            mma_op = cute.nvgpu.warp.MmaF16BF16Op(
                cutlass.BFloat16, self.acc_dtype, (16, 8, 16)
            )
        elif cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
            mma_op = cute.nvgpu.warp.MmaMXF8Op(
                self.a_dtype,
                self.acc_dtype,
                mma_sf_dtype,
            )
        elif cutlass.const_expr(
            self.a_dtype == cutlass.Float6E3M2FN or self.a_dtype == cutlass.Float6E2M3FN
        ):
            # MX-FP6 uses inline ``mxf8f6f4`` MMA in the mainloop. Build tiled_mma
            # with the MXFP8 op so smem/SF layouts match m16n8k32 geometry.
            mma_op = cute.nvgpu.warp.MmaMXF8Op(
                cutlass.Float8E4M3FN,
                self.acc_dtype,
                self.sf_dtype,
            )
        elif cutlass.const_expr(self.sf_vec_size == 32):
            mma_op = cute.nvgpu.warp.MmaMXF4Op(
                self.a_dtype,
                self.acc_dtype,
                self.sf_dtype,
            )
        else:
            mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
                self.a_dtype,
                self.acc_dtype,
                self.sf_dtype,
            )
        atom_shape = self.atom_shape
        atom_layout = cute.make_layout(atom_shape)
        permutation_mnk = (
            self.mma_tile_shape_mnk
            if self.a16
            else sm120_utils.get_permutation_mnk(
                self.mma_tile_shape_mnk,
                32 if self.block_fp8 else self.sf_vec_size,
                cutlass.const_expr(
                    self.a_dtype == cutlass.Float8E4M3FN
                    or self.a_dtype == cutlass.Float6E3M2FN
                    or self.a_dtype == cutlass.Float6E2M3FN
                ),
            )
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )
        # Bare atom for manual unroll workaround (avoids hasAuxTensor address space bug)
        self.mma_atom = cute.make_mma_atom(mma_op)
        # Compute atom loop bounds from tile shape and atom/layout shape
        # MMA atom: m16n8k64 for FP4, m16n8k32 for MXFP8.
        mma_m, mma_n, mma_k = 16, 8, self.mma_k
        self.num_m_tiles = self.mma_tile_shape_mnk[0] // (mma_m * atom_shape[0])
        self.num_n_tiles = self.mma_tile_shape_mnk[1] // (mma_n * atom_shape[1])
        self.num_k_blocks = self.mma_tile_shape_mnk[2] // mma_k

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        # Compute the smem size of SFA/SFB
        if self.a16:
            self.a16_scale_bytes = (
                (
                    128
                    * block_codec(self.weight_only).metadata_bytes
                    * (self.tile_shape_mnk[2] // 32 if self.a16_q8 else 1)
                )
                if self.a16_block
                else max(1, self.tile_shape_mnk[2] // (self.sf_vec_size * 4)) * 512
            )
            sfa_smem_layout_per_stage = cute.make_layout((1, 1))
            sfb_smem_layout_per_stage = cute.make_layout((self.a16_scale_bytes, 1))
        elif self.block_fp8:
            sfa_smem_layout_per_stage = cute.make_layout((1, 1))
            sfb_smem_layout_per_stage = cute.make_layout((1, 1))
        else:
            sfa_smem_layout_per_stage = sm120_make_smem_layout_sfa(
                self.tiled_mma,
                self.tile_shape_mnk,
                self.sf_vec_size,
                1,
            )
            sfb_smem_layout_per_stage = sm120_make_smem_layout_sfb(
                self.tiled_mma,
                self.tile_shape_mnk,
                self.sf_vec_size,
                1,
            )

        # MX-FP6 operands use Float8E4M3FN byte containers in global and
        # shared memory. The explicit format is therefore the reliable policy
        # discriminator; ``a_dtype`` alone also matches ordinary MXFP8.
        if self.mxfp6_fmt_a is not None:

            def _probe_stages(epi_tile: tuple, epi_stage_cap: int) -> tuple:
                return self._compute_stages(
                    self.tile_shape_mnk,
                    self.a_dtype,
                    self.b_dtype,
                    self.sf_dtype,
                    sfa_smem_layout_per_stage,
                    sfb_smem_layout_per_stage,
                    epi_tile,
                    self.c_dtype,
                    self.smem_capacity,
                    self.occupancy,
                    self.b_packed,
                    epi_stage_cap,
                    decode_stage3=True,
                )

            self.epi_tile, epi_stage_cap = self._choose_epilogue(
                (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
                (16 * self.atom_shape[0], 8 * self.atom_shape[1]),
                _probe_stages,
                stages_through_smem=not self.use_m1_non_tma_c,
            )
            self.ab_stage, self.epi_stage = _probe_stages(self.epi_tile, epi_stage_cap)
        elif self.a16:

            def _probe_a16_stages():
                return self._compute_stages(
                    self.tile_shape_mnk,
                    self.a_dtype,
                    cutlass.Float4E2M1FN if self.a16_fp4 else cutlass.Uint8,
                    self.sf_dtype,
                    sfa_smem_layout_per_stage,
                    sfb_smem_layout_per_stage,
                    self.epi_tile,
                    self.c_dtype,
                    self.smem_capacity - self.a16_lut_bytes * self.occupancy,
                    self.occupancy,
                    epi_stage_cap=1,
                    minimum_ab_stage=0,
                    b_storage_bits=8 // self.a16_pack_factor
                    if self.a16_block
                    else None,
                )

            self.ab_stage, self.epi_stage = _probe_a16_stages()
            # A wide Q8 tile plus FP32 split-K partials can leave no room for
            # even one mainloop stage. Subtile the existing epilogue instead
            # of clamping an impossible stage count to one. Keep each tile
            # at least as large as the MMA atom, or its stores omit data.
            while self.ab_stage == 0:
                epi_m, epi_n = self.epi_tile
                if epi_n // 2 >= 8 * self.atom_shape[1]:
                    self.epi_tile = (epi_m, epi_n // 2)
                elif epi_m // 2 >= 16 * self.atom_shape[0]:
                    self.epi_tile = (epi_m // 2, epi_n)
                else:
                    raise ValueError("A16 tile cannot fit one shared-memory stage")
                self.ab_stage, self.epi_stage = _probe_a16_stages()
        else:
            # Other families use the generic stage policy.
            self.ab_stage, self.epi_stage = self._compute_stages(
                self.tile_shape_mnk,
                self.a_dtype,
                self.b_dtype,
                self.sf_dtype,
                sfa_smem_layout_per_stage,
                sfb_smem_layout_per_stage,
                self.epi_tile,
                self.c_dtype,
                self.smem_capacity,
                self.occupancy,
                self.b_packed,
            )

        assert self.epi_stage > 0, (
            "epi_stage <= 0, not enough shared memory. This configuration will be skipped."
        )

        # Decided here because it depends on the computed stage depth; see the
        # _PACKED_B_EXPAND_AHEAD module comment for the >= 4 stage rationale.
        self.packed_expand_ahead = (
            self.b_packed and _PACKED_B_EXPAND_AHEAD and self.ab_stage >= 4
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
            self.block_fp8 or self.a16,
        )

        # Plain (non-swizzled) k-major staging layout for the 3:4-packed B
        # tile, ALIASED into the bottom of each sB stage: TMA writes the 96
        # packed bytes/row there (16-byte aligned box, within TMA's
        # non-swizzled constraints) and the MMA warps expand IN PLACE into the
        # full swizzled 128 B/row stage (two-phase, see
        # _expand_packed_b_stage_smem). No separate staging buffer means the
        # packed mode pays zero extra smem -> one more pipeline stage. The
        # stage stride is sB's full stage size, NOT the packed tile size.
        if self.a16:
            self.sfb_smem_layout_staged = cute.make_layout(
                (self.a16_scale_bytes, 1, self.ab_stage)
            )
            self.a16_packed_k = self.tile_shape_mnk[2] // self.a16_pack_factor
            self.a16_b_smem_layout = cute.make_layout(
                (self.tile_shape_mnk[1], self.a16_packed_k, self.ab_stage),
                stride=(
                    self.a16_packed_k,
                    1,
                    self.tile_shape_mnk[1] * self.a16_packed_k,
                ),
            )
        if self.b_packed:
            self.b_packed_smem_layout_staged = cute.make_layout(
                (
                    self.tile_shape_mnk[1],
                    self.tile_shape_mnk[2] * 3 // 4,
                    self.ab_stage,
                ),
                stride=(
                    self.tile_shape_mnk[2] * 3 // 4,
                    1,
                    self.tile_shape_mnk[1] * self.tile_shape_mnk[2],
                ),
            )
        else:
            self.b_packed_smem_layout_staged = None

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        quant_a_source: cute.Tensor,
        quant_a_positions: cute.Tensor,
        quant_a_cos_sin: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        quant_c_values: cute.Tensor,
        quant_c_scale_rows: cute.Tensor,
        quant_c_scale_mma: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        x_bf16: cute.Tensor = None,
        w_gscale: cute.Tensor = None,
    ):
        """Execute the GEMM operation.

        Args:
            a: Input tensor A (byte-containers, or dummy when a_bf16_fused)
            b: Input tensor B
            sfa: Scale factor tensor for A (or dummy when a_bf16_fused)
            sfb: Scale factor tensor for B
            c: Output tensor C
            alpha: Alpha scaling factor tensor, shape (1,), float32
            max_active_clusters: Max active clusters
            stream: CUDA stream
            epilogue_op: Elementwise epilogue function
            x_bf16: BF16 activation input (MX-FP6 fused quant mode only)
            w_gscale: Weight global scale, shape (1,), f32 (fused mode only)
        """
        # Dead kernel arguments on every non-fused path; substitute a
        # type-compatible live tensor so the traced signature stays uniform.
        # const_expr is required: DSL >= 4.6 otherwise traces this as a
        # runtime if-region and rejects the rebind (CONTAINER_STRUCTURE_CHANGED).
        if cutlass.const_expr(x_bf16 is None):
            x_bf16 = alpha
        if cutlass.const_expr(w_gscale is None):
            w_gscale = alpha
        # Setup static attributes
        self.a_dtype = a.element_type
        self.b_dtype = cutlass.BFloat16 if self.a16 else b.element_type
        self.c_dtype = c.element_type
        self.sf_dtype = sfa.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = (
            utils.LayoutEnum.ROW_MAJOR
            if self.b_tile_major or self.a16_block
            else utils.LayoutEnum.from_tensor(b)
        )
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        # Regular block-FP8 carries compact FP32 scales directly. Native MX
        # paths retain the MMA scale-factor atom layout.
        if cutlass.const_expr(self.block_fp8 or self.a16):
            sfa_tensor = sfa
            sfb_tensor = sfb
        else:
            self.sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
                a.shape, self.sf_vec_size
            )
            sfa_tensor = cute.make_tensor(sfa.iterator, self.sfa_layout)

            # With packed B the gmem extent is 3K/4 bytes, but scale-factor
            # geometry follows the LOGICAL K.
            if cutlass.const_expr(self.b_packed):
                b_logical_shape = (
                    cute.size(b.shape[0]),
                    cute.size(b.shape[1]) * 4 // 3,
                    cute.size(b.shape[2]),
                )
            else:
                b_logical_shape = (
                    cute.size(b.shape[0]),
                    cute.size(b.shape[1]),
                    cute.size(b.shape[2]),
                )
            self.sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
                b_logical_shape, self.sf_vec_size
            )
            sfb_tensor = cute.make_tensor(sfb.iterator, self.sfb_layout)

        if cutlass.const_expr(self.a16):
            tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
                b,
                self.a16_b_smem_layout,
                (self.tile_shape_mnk[1], self.a16_packed_k),
                1,
            )
        elif cutlass.const_expr(self.b_packed):
            # TMA loads the packed tile (tile_n, 3*tile_k/4 bytes) into the
            # plain staging layout; the swizzled sB is filled in-kernel.
            tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
                b,
                self.b_packed_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2] * 3 // 4),
                1,
            )
        else:
            tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
                b,
                self.b_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
                1,
            )
        if cutlass.const_expr(self.fused_quant_a or self.direct_m1_wo_a_inputs):
            # A does not use a TMA descriptor on these paths. Reuse B's as a
            # type-compatible placeholder for the dead kernel argument.
            tma_atom_a = tma_atom_b
            tma_tensor_a = a
        else:
            tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
                a,
                self.a_smem_layout_staged,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                1,
            )
        if cutlass.const_expr(
            self.block_fp8
            or self.a16
            or self.fused_quant_a
            or self.use_m1_non_tma_sfa
            or self.manual_bk64_sf
            or self.direct_sfa_prefix
        ):
            tma_atom_sfa = tma_atom_b
            tma_tensor_sfa = sfa_tensor
        else:
            tma_atom_sfa, tma_tensor_sfa = self._make_tma_atoms_and_tensors(
                sfa_tensor,
                self.sfa_smem_layout_staged,
                self.sfa_tile_shape_mk,
                1,
                internal_type=cutlass.Int16,
            )
        if cutlass.const_expr(
            self.block_fp8
            or self.a16
            or self.manual_bk64_sf
            or self.direct_sfb_representative
        ):
            tma_atom_sfb = tma_atom_b
            tma_tensor_sfb = sfb_tensor
        else:
            tma_atom_sfb, tma_tensor_sfb = self._make_tma_atoms_and_tensors(
                sfb_tensor,
                self.sfb_smem_layout_staged,
                self.sfb_tile_shape_nk,
                1,
                internal_type=cutlass.Int16,
            )
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
            self.direct_one_m_tile_scheduler,
            self.split_k_slices,
            self.large_m_unroll,
            self.a16,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8 if self.a16 else self.b_dtype,
                    cute.cosize(self.a16_b_smem_layout)
                    if self.a16
                    else cute.cosize(self.b_smem_layout_staged),
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

        # Unused (never traced) when b_packed is off; pass the expanded layout
        # as a stand-in so the kernel signature stays uniform.
        if cutlass.const_expr(self.a16):
            b_packed_smem_layout_arg = self.a16_b_smem_layout
        elif cutlass.const_expr(self.b_packed):
            b_packed_smem_layout_arg = self.b_packed_smem_layout_staged
        else:
            b_packed_smem_layout_arg = self.b_smem_layout_staged

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            a,
            quant_a_source,
            quant_a_positions,
            quant_a_cos_sin,
            tma_atom_b,
            tma_tensor_b,
            b,
            tma_atom_sfa,
            tma_tensor_sfa,
            sfa if self.manual_bk64_sf else sfa_tensor,
            tma_atom_sfb,
            tma_tensor_sfb,
            sfb if self.manual_bk64_sf else sfb_tensor,
            tma_atom_c,
            tma_tensor_c,
            c,
            quant_c_values,
            quant_c_scale_rows,
            quant_c_scale_mma,
            self.tiled_mma,
            self.mma_atom,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            b_packed_smem_layout_arg,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
            epilogue_op,
            alpha,
            x_bf16,
            w_gscale,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )
        return

    def _partition_fragment_SFA(
        self,
        sfa_tensor: cute.Tensor,
        thr_mma: cute.ThrMma,
        tidx: int,
    ):
        return sm120_utils.partition_fragment_SFA(sfa_tensor, thr_mma, tidx)

    def _partition_fragment_SFB(
        self,
        sfb_tensor: cute.Tensor,
        thr_mma: cute.ThrMma,
        tidx: int,
    ):
        return sm120_utils.partition_fragment_SFB(sfb_tensor, thr_mma, tidx)

    def _thrfrg_SFA(self, sfa_tensor, tiled_mma: cute.TiledMma):
        return sm120_utils.thrfrg_SFA(sfa_tensor, tiled_mma)

    def _thrfrg_SFB(self, sfb_tensor, tiled_mma: cute.TiledMma):
        return sm120_utils.thrfrg_SFB(sfb_tensor, tiled_mma)

    def _get_layoutSFA_TV(self, tiled_mma: cute.TiledMma):
        return sm120_utils.get_layoutSFA_TV(tiled_mma)

    def _get_layoutSFB_TV(self, tiled_mma: cute.TiledMma):
        return sm120_utils.get_layoutSFB_TV(tiled_mma)

    @cute.jit
    def _fill_replicated_sfb_fragment(self, fragment: cute.Tensor, scale) -> None:
        flat = cute.group_modes(cute.flatten(fragment), 0, cute.rank(fragment))
        for idx in cutlass.range_constexpr(cute.size(flat)):
            flat[idx] = scale

    @cute.jit
    def _accumulate_block_fp8_stage(
        self,
        accumulators: cute.Tensor,
        stage_accumulators: cute.Tensor,
        coord_mn: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        tile_coord_mnl,
        k_tile_global: Int32,
    ) -> None:
        accum_mn = _reshape_acc_to_mn(accumulators)
        stage_accum_mn = _reshape_acc_to_mn(stage_accumulators)
        scale_n = (tile_coord_mnl[1] * Int32(self.tile_shape_mnk[1])) // Int32(128)
        scale_b = cutlass.Float32(sfb[(scale_n, k_tile_global, tile_coord_mnl[2])])
        for acc_m in cutlass.range_constexpr(cute.size(accum_mn.shape[0])):
            coord = coord_mn[acc_m, 0]
            m_coord = tile_coord_mnl[0] * Int32(self.tile_shape_mnk[0]) + coord[0]
            scale_a = cutlass.Float32(0.0)
            if m_coord < Int32(sfa.shape[0]):
                scale_a = cutlass.Float32(
                    sfa[(m_coord, k_tile_global, tile_coord_mnl[2])]
                )
            scale_ab = scale_a * scale_b
            for acc_n in cutlass.range_constexpr(cute.size(accum_mn.shape[1])):
                accum_mn[acc_m, acc_n] += stage_accum_mn[acc_m, acc_n] * scale_ab
                stage_accum_mn[acc_m, acc_n] = 0.0

    @cute.jit
    def _make_cpasync_tiled_copy(
        self,
        dtype: cutlass.Constexpr,
        tile_cols: cutlass.Constexpr[int],
    ) -> cute.TiledCopy:
        copy_bits = 128
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=copy_bits,
        )
        async_copy_elems = copy_bits // dtype.width
        t_shape_dim_1 = tile_cols // async_copy_elems
        assert self.num_threads_per_warp % t_shape_dim_1 == 0
        t_layout = cute.make_ordered_layout(
            (self.num_threads_per_warp // t_shape_dim_1, t_shape_dim_1),
            order=(1, 0),
        )
        v_layout = cute.make_layout((1, async_copy_elems))
        return cute.make_tiled_copy_tv(atom_async_copy, t_layout, v_layout)

    @cute.jit
    def _make_scale_tiled_copy(
        self,
        dtype: cutlass.Constexpr,
    ) -> cute.TiledCopy:
        copy_bits = dtype.width
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=copy_bits,
        )
        return cute.make_tiled_copy_tv(
            atom_async_copy,
            cute.make_layout((self.num_threads_per_warp,)),
            cute.make_layout((copy_bits // dtype.width,)),
        )

    @cute.jit
    def _predicate_cpasync_rows(
        self,
        tCc: cute.Tensor,
        row_limit: Int32,
    ) -> cute.Tensor:
        tPred = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    cute.size(tCc, mode=[0, 1]),
                    cute.size(tCc, mode=[1]),
                    cute.size(tCc, mode=[2]),
                ),
                stride=(cute.size(tCc, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tPred.shape[0]):
            for rest_k in cutlass.range_constexpr(tPred.shape[2]):
                tPred[rest_v, 0, rest_k] = tCc[(0, rest_v), 0, rest_k][0] < row_limit
        return tPred

    @cute.jit
    def _cpasync_copy_2d(
        self,
        tiled_copy: cute.TiledCopy,
        tG: cute.Tensor,
        tS: cute.Tensor,
        tC: cute.Tensor,
        row_limit: Int32,
        predicate_rows: cutlass.Constexpr[bool],
    ) -> None:
        if cutlass.const_expr(predicate_rows):
            tP = self._predicate_cpasync_rows(tC, row_limit)
        for rest_m in cutlass.range_constexpr(cute.size(tS.shape[1])):
            if cutlass.const_expr(predicate_rows):
                cute.copy(
                    tiled_copy,
                    tG[None, rest_m, None],
                    tS[None, rest_m, None],
                    pred=tP[None, rest_m, None],
                )
            else:
                cute.copy(
                    tiled_copy,
                    tG[None, rest_m, None],
                    tS[None, rest_m, None],
                )

    @cute.jit
    def _scale_copy_2d(
        self,
        tiled_copy: cute.TiledCopy,
        tG: cute.Tensor,
        tS: cute.Tensor,
        tC: cute.Tensor,
        row_limit: Int32,
    ) -> None:
        tP = cute.make_rmem_tensor(cute.make_layout(tS.shape), cutlass.Boolean)
        for i in cutlass.range_constexpr(cute.size(tP)):
            tP[i] = cute.elem_less(tC[i][0][0][0], row_limit)
        for rest_m in cutlass.range_constexpr(cute.size(tS.shape[1])):
            cute.copy(
                tiled_copy,
                tG[None, rest_m, None],
                tS[None, rest_m, None],
                pred=tP[None, rest_m, None],
            )

    @cute.jit
    def _load_a16_b_fragment(
        self,
        fragment: cute.Tensor,
        coordinates: cute.Tensor,
        packed: cute.Tensor,
        scales: cute.Tensor,
        lut: cute.Tensor,
        stage: Int32,
        n_tile: Int32,
        k_tile: Int32,
        size_n: cutlass.Constexpr,
        size_k: cutlass.Constexpr,
    ):
        pairs = cute.recast_tensor(fragment, Uint32)
        packed_u16 = cute.recast_tensor(packed, cutlass.Uint16)
        if cutlass.const_expr(self.a16_block):
            packed_u32 = cute.recast_tensor(packed, Uint32)
            bases = cute.recast_tensor(scales, cutlass.Uint16)
            rows_per_fragment = 2 if self.a16_block_transposed else 1
            for i in cutlass.range_constexpr(
                cute.size(fragment) // (4 * rows_per_fragment)
            ):
                for row in cutlass.range_constexpr(rows_per_fragment):
                    local_n, local_k = coordinates[i * 4 * rows_per_fragment + row * 2]
                    scale_n = (
                        local_n
                        + (n_tile % self.sfb_tiles_per_block) * self.tile_shape_mnk[1]
                    )
                    block_k = (k_tile * self.tile_shape_mnk[2] + local_k) % 256
                    if cutlass.const_expr(self.a16_q8):
                        base = Uint32(bases[local_k // 32 * 128 + scale_n, 0, stage])
                        p0 = q8_0_pair_to_bf16x2(
                            Uint32(packed_u16[local_n, local_k // 2, stage]), base
                        )
                        p1 = q8_0_pair_to_bf16x2(
                            Uint32(packed_u16[local_n, (local_k + 8) // 2, stage]), base
                        )
                    else:
                        if cutlass.const_expr(self.a16_iq2_xxs):
                            grids = Uint32(
                                packed_u32[local_n, local_k // 32 * 2, stage]
                            )
                            signs_scale = Uint32(
                                packed_u32[local_n, local_k // 32 * 2 + 1, stage]
                            )
                            descriptors = iq2_xxs_descriptor_pair(
                                grids, signs_scale, Int32(local_k // 16 % 2)
                            )
                            subscale = signs_scale >> 28
                        else:
                            scale_pair = Uint32(
                                scales[256 + (block_k // 32) * 128 + scale_n, 0, stage]
                            )
                            subscale = (scale_pair >> ((block_k // 16 % 2) * 4)) & 15
                            descriptors = Uint32(
                                packed_u32[local_n, local_k // 16, stage]
                            )
                        p0, p1 = iq2_xs_descriptor_pair_to_bf16x2x2(
                            descriptors,
                            Uint32(bases[scale_n, 0, stage]),
                            subscale,
                            Int64(shared_ptr_to_u32(lut.iterator)),
                            Int32(local_k % 8),
                        )
                    pairs[i * 2 * rows_per_fragment + row] = p0
                    pairs[i * 2 * rows_per_fragment + row + rows_per_fragment] = p1
        else:
            for i in cutlass.range_constexpr(cute.size(fragment) // 2):
                local_n, local_k = coordinates[i * 2]
                scale_n = (
                    local_n
                    + (n_tile % self.sfb_tiles_per_block) * self.tile_shape_mnk[1]
                )
                valid = cutlass.Boolean(True)
                if cutlass.const_expr(size_k % self.tile_shape_mnk[2] != 0):
                    valid = Int64(k_tile) * self.tile_shape_mnk[2] + local_k < size_k
                packed_pair = Uint32(0)
                if valid:
                    group = local_k // self.sf_vec_size
                    if cutlass.const_expr(
                        self.tile_shape_mnk[2] < self.sf_vec_size * 4
                    ):
                        group += (k_tile % 2) * 2
                    scale_offset = (
                        group // 4 * 512
                        + (scale_n % 32) * 16
                        + (scale_n // 32) * 4
                        + group % 4
                    )
                    byte = Uint32(scales[scale_offset, 0, stage])
                    if cutlass.const_expr(self.a16_fp4):
                        packed_pair = nvfp4_pair_to_bf16x2_sm120(
                            Uint32(packed[local_n, local_k // 2, stage]),
                            byte,
                        )
                    else:
                        word = Uint32(packed_u16[local_n, local_k // 2, stage])
                        packed_pair = mxfp8_pair_to_bf16x2_sm120(word, byte)
                pairs[i] = packed_pair

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        directA_mkl: cute.Tensor,
        quantA_mkl: cute.Tensor,
        quantA_positions: cute.Tensor,
        quantA_cos_sin: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        directB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        directSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        directSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        directC_mnl: cute.Tensor,
        quantC_values: cute.Tensor,
        quantC_scale_rows: cute.Tensor,
        quantC_scale_mma: cute.Tensor,
        tiled_mma: cute.TiledMma,
        mma_atom: cute.MmaAtom,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        b_packed_smem_layout_staged,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
        directX_bf16: cute.Tensor,
        w_gscale: cute.Tensor,
    ):
        # Keep alpha in FP32 for precision
        alpha_value = alpha[0].to(cutlass.Float32)
        if cutlass.const_expr(self.alpha_reciprocal):
            alpha_value = cutlass.Float32(1.0) / alpha_value

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == 0:
            if cutlass.const_expr(
                self.load_path == "tma"
                and not self.use_m1_non_tma_a
                and not self.fused_quant_a
                and not self.direct_m1_wo_a_inputs
            ):
                cpasync.prefetch_descriptor(tma_atom_a)
            if cutlass.const_expr(self.load_path == "tma"):
                cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(
                self.load_path == "tma"
                and not (self.block_fp8 or self.a16)
                and not self.use_m1_non_tma_sfa
                and not self.fused_quant_a
                and not self.manual_bk64_sf
                and not self.direct_sfa_prefix
            ):
                cpasync.prefetch_descriptor(tma_atom_sfa)
            if cutlass.const_expr(
                self.load_path == "tma"
                and not (self.block_fp8 or self.a16)
                and not self.manual_bk64_sf
                and not self.direct_sfb_representative
            ):
                cpasync.prefetch_descriptor(tma_atom_sfb)
            if cutlass.const_expr(not self.use_m1_non_tma_c):
                cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        # B's TMA transaction covers the packed staging tile when b_packed (the
        # expanded sB is filled by the MMA warps, not by TMA).
        if cutlass.const_expr(self.b_packed):
            b_tma_smem_layout = cute.slice_(
                b_packed_smem_layout_staged, (None, None, 0)
            )
        else:
            b_tma_smem_layout = b_smem_layout
        if cutlass.const_expr(self.a16):
            tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout)
            tma_copy_bytes += (
                self.tile_shape_mnk[1] * self.a16_packed_k + self.a16_scale_bytes
            )
        elif cutlass.const_expr(self.block_fp8):
            tma_copy_bytes = cute.size_in_bytes(
                self.a_dtype, a_smem_layout
            ) + cute.size_in_bytes(self.b_dtype, b_tma_smem_layout)
        elif cutlass.const_expr(self.fused_quant_a):
            tma_copy_bytes = cute.size_in_bytes(
                self.b_dtype, b_tma_smem_layout
            ) + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        elif cutlass.const_expr(self.manual_bk64_sf):
            tma_copy_bytes = cute.size_in_bytes(self.b_dtype, b_tma_smem_layout)
            if cutlass.const_expr(not self.use_m1_non_tma_a):
                tma_copy_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)
        elif cutlass.const_expr(self.use_m1_non_tma_sfa):
            tma_copy_bytes = cute.size_in_bytes(
                self.b_dtype, b_tma_smem_layout
            ) + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
            if cutlass.const_expr(not self.use_m1_non_tma_a):
                tma_copy_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)
        else:
            tma_copy_bytes = (
                cute.size_in_bytes(self.b_dtype, b_tma_smem_layout)
                + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
                + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
            )
            if cutlass.const_expr(self.direct_m1_wo_a_inputs):
                tma_copy_bytes += 128
            else:
                tma_copy_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)
        if cutlass.const_expr(self.direct_sfb_representative):
            tma_copy_bytes -= cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
            tma_copy_bytes += 16
        if cutlass.const_expr(self.direct_sfa_prefix):
            tma_copy_bytes -= cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            tma_copy_bytes += 256

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        a16_lut = directSFA_mkl
        if cutlass.const_expr(self.a16_lut_bytes > 0):
            a16_lut = smem.allocate_tensor(
                cutlass.Uint8,
                cute.make_layout((self.a16_lut_bytes,)),
                byte_alignment=16,
            )
            lut_address = shared_ptr_to_u32(a16_lut.iterator)
            for chunk_base in cutlass.range_constexpr(
                (self.a16_lut_bytes // 16 + self.threads_per_cta - 1)
                // self.threads_per_cta
            ):
                chunk = Int32(tidx) + chunk_base * self.threads_per_cta
                if chunk < self.a16_lut_bytes // 16:
                    cp_async4_shared_global(
                        lut_address + chunk * 16,
                        get_ptr_as_int64(directSFA_mkl, Int64(chunk) * 16),
                    )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)

        # Pipeline setup
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        # A16 scalar shared loads must complete in every consumer thread.
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_mma_warps * (self.num_threads_per_warp if self.a16 else 1),
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        if cutlass.const_expr(self.load_path == "cpasync"):
            mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_threads_per_warp,
            )
            mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_warps * self.num_threads_per_warp,
            )
            mainloop_pipeline = pipeline.PipelineAsync.create(
                num_stages=self.ab_stage,
                producer_group=mainloop_pipeline_producer_group,
                consumer_group=mainloop_pipeline_consumer_group,
                barrier_storage=mainloop_pipeline_array_ptr,
            )
        else:
            mainloop_pipeline = pipeline.PipelineTmaAsync.create(
                num_stages=self.ab_stage,
                producer_group=mainloop_pipeline_producer_group,
                consumer_group=mainloop_pipeline_consumer_group,
                tx_count=tma_copy_bytes,
                barrier_storage=mainloop_pipeline_array_ptr,
                cta_layout_vmnk=cta_layout_vmnk,
            )

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # Generate smem tensors
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        if cutlass.const_expr(self.a16):
            sBPacked = storage.sB.get_tensor(b_packed_smem_layout_staged)
            # This view supplies the BF16 MMA fragment layout; A16 loads the
            # compressed storage directly into those registers below.
            sB = cute.make_tensor(
                cute.recast_ptr(storage.sB.data_ptr(), dtype=cutlass.BFloat16),
                b_smem_layout_staged,
            )
        else:
            sB = storage.sB.get_tensor(
                b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
            )
        if cutlass.const_expr(self.b_packed):
            # Packed TMA destination ALIASED into sB's storage (bottom 96 B of
            # each row span, stage stride = full sB stage): no separate buffer,
            # expansion happens in place (see _expand_packed_b_stage_smem).
            sBPacked = cute.make_tensor(
                storage.sB.data_ptr(), b_packed_smem_layout_staged
            )
            # Raw u32 smem address for the packed->container expansion. Two
            # constraints force this exact spot: (1) cute.recast_tensor strips
            # sB's swizzle (see _expand_packed_b_stage_smem), so the expansion
            # needs raw addresses; (2) @cute.struct instances cannot be
            # flattened across DYNAMIC ifs (NVIDIA/cutlass#3268) and the if-
            # capture analysis is syntactic, so ``storage`` must never be
            # referenced inside the warp-dispatch branches - only this Int32
            # address may cross into them.
            sb_base_addr = shared_ptr_to_u32(storage.sB.data_ptr())
        if cutlass.const_expr(self.a_bf16_fused):
            sa_base_addr = shared_ptr_to_u32(storage.sA.data_ptr())
            ssfa_base_addr = shared_ptr_to_u32(storage.sSFA.data_ptr())
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        # Local_tile partition global tensors
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        if cutlass.const_expr(self.a16):
            gB_nkl = cute.local_tile(
                mB_nkl,
                (self.tile_shape_mnk[1], self.a16_packed_k),
                (None, None, None),
            )
        elif cutlass.const_expr(self.b_packed):
            # Packed gmem extent: 96 bytes per 128-wide logical K-tile, same
            # K-tile count as the A side ((3K/4)/96 == K/128).
            gB_nkl = cute.local_tile(
                mB_nkl,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2] * 3 // 4),
                (None, None, None),
            )
        else:
            gB_nkl = cute.local_tile(
                mB_nkl,
                cute.slice_(self.tile_shape_mnk, (0, None, None)),
                (None, None, None),
            )
        if cutlass.const_expr(
            not (self.block_fp8 or self.a16)
            and not self.use_m1_non_tma_sfa
            and not self.fused_quant_a
        ):
            gSFA_mkl = cute.local_tile(
                mSFA_mkl,
                self.sfa_tile_shape_mk,
                (None, None, None),
            )
        if cutlass.const_expr(not self.a16):
            gSFB_nkl = cute.local_tile(
                mSFB_nkl,
                self.sfb_tile_shape_nk,
                (None, None, None),
            )
        if cutlass.const_expr(self.load_path == "cpasync"):
            gA_cpasync_mkl = cute.local_tile(
                directA_mkl,
                cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                (None, None, None),
            )
            gB_cpasync_nkl = cute.local_tile(
                directB_nkl,
                cute.slice_(self.tile_shape_mnk, (0, None, None)),
                (None, None, None),
            )
            gSFA_cpasync_mkl = cute.local_tile(
                directSFA_mkl,
                self.sfa_tile_shape_mk,
                (None, None, None),
            )
            gSFB_cpasync_nkl = cute.local_tile(
                directSFB_nkl,
                self.sfb_tile_shape_nk,
                (None, None, None),
            )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # Partition for TiledMMA
        thr_mma = tiled_mma.get_slice(tidx)

        # TMA partitions for A
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        if cutlass.const_expr(
            self.load_path == "tma"
            and not self.use_m1_non_tma_a
            and not self.fused_quant_a
            and not self.direct_m1_wo_a_inputs
        ):
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                a_cta_crd,
                a_cta_layout,
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA_mkl, 0, 2),
            )

        # TMA partitions for B (targets the packed staging buffer when b_packed)
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        if cutlass.const_expr(self.load_path == "tma" and (self.b_packed or self.a16)):
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                b_cta_crd,
                b_cta_layout,
                cute.group_modes(sBPacked, 0, 2),
                cute.group_modes(gB_nkl, 0, 2),
            )
        elif cutlass.const_expr(self.load_path == "tma"):
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                b_cta_crd,
                b_cta_layout,
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB_nkl, 0, 2),
            )

        # TMA partitions for SFA
        if cutlass.const_expr(
            self.load_path == "tma"
            and not (self.block_fp8 or self.a16)
            and not self.use_m1_non_tma_sfa
            and not self.fused_quant_a
            and not self.manual_bk64_sf
            and not self.direct_sfa_prefix
        ):
            tAsSFA, tAgSFA = cpasync.tma_partition(
                tma_atom_sfa,
                a_cta_crd,
                a_cta_layout,
                cute.group_modes(sSFA, 0, 2),
                cute.group_modes(gSFA_mkl, 0, 2),
            )
            tAsSFA = cute.filter_zeros(tAsSFA)
            tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA partitions for SFB
        if cutlass.const_expr(
            self.load_path == "tma"
            and not (self.block_fp8 or self.a16)
            and not self.manual_bk64_sf
            and not self.direct_sfb_representative
        ):
            tBsSFB, tBgSFB = cpasync.tma_partition(
                tma_atom_sfb,
                b_cta_crd,
                b_cta_layout,
                cute.group_modes(sSFB, 0, 2),
                cute.group_modes(gSFB_nkl, 0, 2),
            )
            tBsSFB = cute.filter_zeros(tBsSFB)
            tBgSFB = cute.filter_zeros(tBgSFB)

        if cutlass.const_expr(self.load_path == "cpasync"):
            cpasync_tiled_copy_A = self._make_cpasync_tiled_copy(
                self.a_dtype,
                self.tile_shape_mnk[2],
            )
            cpasync_tiled_copy_B = self._make_cpasync_tiled_copy(
                self.b_dtype,
                self.tile_shape_mnk[2],
            )
            cpasync_tiled_copy_SF = self._make_scale_tiled_copy(self.sf_dtype)
            cA_mkl = cute.make_identity_tensor(cute.shape(directA_mkl))
            cA_cpasync_mkl = cute.local_tile(
                cA_mkl,
                cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                (None, None, None),
            )
            cB_nkl = cute.make_identity_tensor(cute.shape(directB_nkl))
            cB_cpasync_nkl = cute.local_tile(
                cB_nkl,
                cute.slice_(self.tile_shape_mnk, (0, None, None)),
                (None, None, None),
            )
            cSFA_mkl = cute.make_identity_tensor(cute.shape(directSFA_mkl))
            cSFA_cpasync_mkl = cute.local_tile(
                cSFA_mkl,
                self.sfa_tile_shape_mk,
                (None, None, None),
            )
            cSFB_nkl = cute.make_identity_tensor(cute.shape(directSFB_nkl))
            cSFB_cpasync_nkl = cute.local_tile(
                cSFB_nkl,
                self.sfb_tile_shape_nk,
                (None, None, None),
            )

            cpasync_lane = tidx % self.num_threads_per_warp
            thr_cpasync_A = cpasync_tiled_copy_A.get_slice(cpasync_lane)
            thr_cpasync_B = cpasync_tiled_copy_B.get_slice(cpasync_lane)
            thr_cpasync_SF = cpasync_tiled_copy_SF.get_slice(cpasync_lane)
            tAgA_cpasync_mkl = thr_cpasync_A.partition_S(gA_cpasync_mkl)
            tAsA_cpasync = thr_cpasync_A.partition_D(sA)
            tAcA_cpasync_mkl = thr_cpasync_A.partition_S(cA_cpasync_mkl)
            tBgB_cpasync_nkl = thr_cpasync_B.partition_S(gB_cpasync_nkl)
            tBsB_cpasync = thr_cpasync_B.partition_D(sB)
            tBcB_cpasync_nkl = thr_cpasync_B.partition_S(cB_cpasync_nkl)
            tAgSFA_cpasync_mkl = thr_cpasync_SF.partition_S(gSFA_cpasync_mkl)
            tAsSFA_cpasync = thr_cpasync_SF.partition_D(sSFA)
            tAcSFA_cpasync_mkl = thr_cpasync_SF.partition_S(cSFA_cpasync_mkl)
            tBgSFB_cpasync_nkl = thr_cpasync_SF.partition_S(gSFB_cpasync_nkl)
            tBsSFB_cpasync = thr_cpasync_SF.partition_D(sSFB)
            tBcSFB_cpasync_nkl = thr_cpasync_SF.partition_S(cSFB_cpasync_nkl)

        # Make fragments. swap_ab keeps public C[M,N] unchanged but presents
        # B as MMA-A and A as MMA-B.
        if cutlass.const_expr(self.swap_ab):
            tCsA = thr_mma.partition_A(sB)
            tCsB = thr_mma.partition_B(sA)
        else:
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)

        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        if cutlass.const_expr(self.a16):
            if cutlass.const_expr(self.swap_ab):
                tCgC = thr_mma.partition_C(
                    cute.make_identity_tensor(
                        (self.tile_shape_mnk[1], self.tile_shape_mnk[0])
                    )
                )
                a16_weight_fragments = tCrA
                a16_b_coordinates = thr_mma.partition_A(
                    cute.make_identity_tensor(
                        (self.tile_shape_mnk[1], self.tile_shape_mnk[2])
                    )
                )
            else:
                tCgC = thr_mma.partition_C(gC_mnl)
                a16_weight_fragments = tCrB
                a16_b_coordinates = thr_mma.partition_B(
                    cute.make_identity_tensor(
                        (self.tile_shape_mnk[1], self.tile_shape_mnk[2])
                    )
                )
        elif cutlass.const_expr(self.block_fp8):
            tCgC = thr_mma.partition_C(gC_mnl)
        elif cutlass.const_expr(self.swap_ab):
            tCrSFA_full = self._partition_fragment_SFA(
                sSFB[None, None, 0], thr_mma, tidx
            )
            tCrSFB_full = self._partition_fragment_SFB(
                sSFA[None, None, 0], thr_mma, tidx
            )
            c_mma = cute.make_identity_tensor(
                (self.tile_shape_mnk[1], self.tile_shape_mnk[0])
            )
            tCgC = thr_mma.partition_C(c_mma)
        else:
            tCrSFA_full = self._partition_fragment_SFA(
                sSFA[None, None, 0], thr_mma, tidx
            )
            tCrSFB_full = self._partition_fragment_SFB(
                sSFB[None, None, 0], thr_mma, tidx
            )
            tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        if cutlass.const_expr(self.block_fp8):
            stage_accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
            block_fp8_c_identity = cute.make_identity_tensor(
                cute.slice_(self.tile_shape_mnk, (None, None, 0))
            )
            block_fp8_coord_mn = _reshape_acc_to_mn(
                thr_mma.partition_C(block_fp8_c_identity)
            )
        if cutlass.const_expr(self.swap_ab):
            swap_ab_acc_mn = _reshape_acc_to_mn(accumulators, transpose=True)
            swap_ab_c_identity = cute.make_identity_tensor(
                (self.tile_shape_mnk[1], self.tile_shape_mnk[0])
            )
            swap_ab_coord_mn = _reshape_acc_to_mn(
                thr_mma.partition_C(swap_ab_c_identity),
                transpose=True,
            )
        if cutlass.const_expr(self.split_k_slices > 1):
            split_k_acc_mn = _reshape_acc_to_mn(accumulators)
            split_k_c_identity = cute.make_identity_tensor(
                cute.slice_(self.tile_shape_mnk, (None, None, 0))
            )
            split_k_coord_mn = _reshape_acc_to_mn(
                thr_mma.partition_C(split_k_c_identity)
            )

        # Cluster/thread sync
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        block_idx = cute.arch.block_idx()
        k_tile_start = Int32(0)
        k_tile_iter_cnt = k_tile_cnt
        if cutlass.const_expr(self.split_k_slices > 1):
            if cutlass.const_expr(self.a16):
                k_tile_start = Int32(block_idx[1]) * k_tile_cnt // self.split_k_slices
                k_tile_iter_cnt = (
                    Int32(block_idx[1]) + 1
                ) * k_tile_cnt // self.split_k_slices - k_tile_start
            else:
                k_tiles_per_split = k_tile_cnt // self.split_k_slices
                k_tile_start = Int32(block_idx[1]) * Int32(k_tiles_per_split)
                k_tile_iter_cnt = k_tiles_per_split

        # Tile scheduler
        if cutlass.const_expr(self.direct_one_m_tile_scheduler):
            direct_tile_valid = Int32(block_idx[2]) < Int32(
                tile_sched_params.problem_shape_ntile_mnl[1]
            )
            work_tile = WorkTileInfo(
                (
                    Int32(block_idx[0]) if self.a16 else Int32(0),
                    Int32(block_idx[2]),
                    Int32(0),
                ),
                direct_tile_valid,
            )
        else:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, block_idx, cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

        # Pipeline states
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        if cutlass.const_expr(self.packed_expand_ahead):
            # Second consumer-side view of the mainloop pipeline, kept exactly
            # one stage ahead of mainloop_consumer_state: it advances once at
            # each work-tile prologue and once per k_tile at k_block 0, for a
            # total of k_tile_cnt per tile — the same as the main state's
            # (k_tile_cnt - 1) in-loop advances plus 1 in the hoisted tail —
            # so the two stay phase-aligned across persistent work tiles.
            packed_b_lookahead_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )

        # MMA warp group
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # Copy atoms for SMEM->RMEM
            if cutlass.const_expr(self.swap_ab):
                atom_copy_ldmatrix_A = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                    self.b_dtype,
                )
                atom_copy_ldmatrix_B = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(
                        self.a_layout.is_m_major_a(), 2 if self.a16 else 4
                    ),
                    self.a_dtype,
                )
            else:
                atom_copy_ldmatrix_A = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                    self.a_dtype,
                )
                atom_copy_ldmatrix_B = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                    self.b_dtype,
                )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            if cutlass.const_expr(not (self.block_fp8 or self.a16)):
                atom_copy_ldmatrix_SF = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.sf_dtype,
                )
                smem_tiled_copy_SFA = cute.make_tiled_copy(
                    atom_copy_ldmatrix_SF,
                    self._get_layoutSFA_TV(tiled_mma),
                    (
                        cute.size(tiled_mma.permutation_mnk[0]),
                        cute.size(tiled_mma.permutation_mnk[2]),
                    ),
                )
                smem_tiled_copy_SFB = cute.make_tiled_copy(
                    atom_copy_ldmatrix_SF,
                    self._get_layoutSFB_TV(tiled_mma),
                    (
                        cute.size(tiled_mma.permutation_mnk[1]),
                        cute.size(tiled_mma.permutation_mnk[2]),
                    ),
                )

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(
                sB if cutlass.const_expr(self.swap_ab) else sA
            )
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(
                sA if cutlass.const_expr(self.swap_ab) else sB
            )
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            if cutlass.const_expr(not (self.block_fp8 or self.a16)):
                thr_copy_ldmatrix_SFA = smem_tiled_copy_SFA.get_slice(tidx)
                thr_copy_ldmatrix_SFB = smem_tiled_copy_SFB.get_slice(tidx)
                tCsSFA_copy_view_full = thr_copy_ldmatrix_SFA.partition_S(
                    sSFB if cutlass.const_expr(self.swap_ab) else sSFA
                )
                tCrSFA_copy_view_full = thr_copy_ldmatrix_SFA.retile(tCrSFA_full)
                tCsSFB_copy_view_full = thr_copy_ldmatrix_SFB.partition_S(
                    sSFA if cutlass.const_expr(self.swap_ab) else sSFB
                )
                tCrSFB_copy_view_full = thr_copy_ldmatrix_SFB.retile(tCrSFB_full)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                sfa_tile_offset = tile_coord_mnl[0] % self.sfa_tiles_per_block
                sfb_tile_offset = tile_coord_mnl[1] % self.sfb_tiles_per_block
                if cutlass.const_expr(self.block_fp8 or self.a16):
                    pass
                elif cutlass.const_expr(self.swap_ab):
                    if cutlass.const_expr(self.sfb_tiles_per_block > 1):
                        sSFB_tile = cute.local_tile(
                            sSFB,
                            cute.slice_(self.tile_shape_mnk, (0, None, None)),
                            (sfb_tile_offset, 0, None),
                        )
                        tCsSFA_tile_copy_view = thr_copy_ldmatrix_SFA.partition_S(
                            sSFB_tile
                        )
                        tCrSFA_tile = self._partition_fragment_SFA(
                            sSFB_tile[None, None, 0], thr_mma, tidx
                        )
                        tCrSFA_tile_copy_view = thr_copy_ldmatrix_SFA.retile(
                            tCrSFA_tile
                        )
                    else:
                        tCsSFA_tile_copy_view = tCsSFA_copy_view_full
                        tCrSFA_tile = tCrSFA_full
                        tCrSFA_tile_copy_view = tCrSFA_copy_view_full
                    if cutlass.const_expr(self.sfa_tiles_per_block > 1):
                        sSFA_tile = cute.local_tile(
                            sSFA,
                            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                            (sfa_tile_offset, 0, None),
                        )
                        tCsSFB_tile_copy_view = thr_copy_ldmatrix_SFB.partition_S(
                            sSFA_tile
                        )
                        tCrSFB_tile = self._partition_fragment_SFB(
                            sSFA_tile[None, None, 0], thr_mma, tidx
                        )
                        tCrSFB_tile_copy_view = thr_copy_ldmatrix_SFB.retile(
                            tCrSFB_tile
                        )
                    else:
                        tCsSFB_tile_copy_view = tCsSFB_copy_view_full
                        tCrSFB_tile = tCrSFB_full
                        tCrSFB_tile_copy_view = tCrSFB_copy_view_full
                else:
                    if cutlass.const_expr(self.sfa_tiles_per_block > 1):
                        sSFA_tile = cute.local_tile(
                            sSFA,
                            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                            (sfa_tile_offset, 0, None),
                        )
                        tCsSFA_tile_copy_view = thr_copy_ldmatrix_SFA.partition_S(
                            sSFA_tile
                        )
                        tCrSFA_tile = self._partition_fragment_SFA(
                            sSFA_tile[None, None, 0], thr_mma, tidx
                        )
                        tCrSFA_tile_copy_view = thr_copy_ldmatrix_SFA.retile(
                            tCrSFA_tile
                        )
                    else:
                        tCsSFA_tile_copy_view = tCsSFA_copy_view_full
                        tCrSFA_tile = tCrSFA_full
                        tCrSFA_tile_copy_view = tCrSFA_copy_view_full
                    if cutlass.const_expr(self.sfb_tiles_per_block > 1):
                        sSFB_tile = cute.local_tile(
                            sSFB,
                            cute.slice_(self.tile_shape_mnk, (0, None, None)),
                            (sfb_tile_offset, 0, None),
                        )
                        tCsSFB_tile_copy_view = thr_copy_ldmatrix_SFB.partition_S(
                            sSFB_tile
                        )
                        tCrSFB_tile = self._partition_fragment_SFB(
                            sSFB_tile[None, None, 0], thr_mma, tidx
                        )
                        tCrSFB_tile_copy_view = thr_copy_ldmatrix_SFB.retile(
                            tCrSFB_tile
                        )
                    else:
                        tCsSFB_tile_copy_view = tCsSFB_copy_view_full
                        tCrSFB_tile = tCrSFB_full
                        tCrSFB_tile_copy_view = tCrSFB_copy_view_full
                accumulators.fill(0.0)
                if cutlass.const_expr(self.block_fp8):
                    stage_accumulators.fill(0.0)

                # Pipelined MAINLOOP
                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_iter_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                if cutlass.const_expr(self.b_packed):
                    # Expand the TMA-staged packed B tile in place into the
                    # swizzled byte-container sB BEFORE any ldmatrix touches
                    # this stage (two-phase; internal read/write barrier).
                    _expand_packed_b_stage_smem(
                        sb_base_addr,
                        mainloop_consumer_state.index,
                        Int32(tidx),
                        self.tile_shape_mnk[1],
                        self.tile_shape_mnk[2],
                        self.num_mma_warps * self.num_threads_per_warp,
                        self.mma_sync_barrier,
                    )
                    self.mma_sync_barrier.arrive_and_wait()
                    if cutlass.const_expr(self.packed_expand_ahead):
                        # Lookahead expanded this stage too (same index as the
                        # consumer state at the prologue); move it one ahead.
                        packed_b_lookahead_state.advance()
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                if cutlass.const_expr(not (self.block_fp8 or self.a16)):
                    tCsSFA_p = tCsSFA_tile_copy_view[
                        None, None, None, mainloop_consumer_state.index
                    ]
                    tCsSFB_p = tCsSFB_tile_copy_view[
                        None, None, None, mainloop_consumer_state.index
                    ]
                if cutlass.const_expr(self.a16_block_transposed):
                    cute.copy(
                        smem_tiled_copy_B,
                        tCsB_p[None, None, 0],
                        tCrB_copy_view[None, None, 0],
                    )
                else:
                    cute.copy(
                        smem_tiled_copy_A,
                        tCsA_p[None, None, 0],
                        tCrA_copy_view[None, None, 0],
                    )
                if cutlass.const_expr(self.a16):
                    self._load_a16_b_fragment(
                        a16_weight_fragments[None, None, 0],
                        a16_b_coordinates[None, None, 0],
                        sBPacked,
                        sSFB,
                        a16_lut,
                        mainloop_consumer_state.index,
                        tile_coord_mnl[1],
                        k_tile_start,
                        cute.size(directB_nkl, mode=[0]),
                        cute.size(directB_nkl, mode=[1]) * self.a16_pack_factor,
                    )
                else:
                    cute.copy(
                        smem_tiled_copy_B,
                        tCsB_p[None, None, 0],
                        tCrB_copy_view[None, None, 0],
                    )

                if cutlass.const_expr(not (self.block_fp8 or self.a16)):
                    tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                    tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                    tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_tile_copy_view)
                    tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_tile_copy_view)

                    # Whole-stage SF copy: scale bytes for all k blocks of the
                    # acquired stage load in one bulk copy.
                    cute.copy(
                        smem_tiled_copy_SFA,
                        tCsSFA_p_filtered,
                        tCrSFA_copy_view_filtered,
                    )
                    if cutlass.const_expr(self.direct_sfb_representative):
                        self._fill_replicated_sfb_fragment(
                            tCrSFB_tile[None, None, 0],
                            sSFB[
                                (
                                    Int32(0),
                                    Int32(0),
                                    mainloop_consumer_state.index,
                                )
                            ],
                        )
                    elif cutlass.const_expr(self.sfb_k_reuse):
                        cute.copy(
                            smem_tiled_copy_SFB,
                            tCsSFB_p_filtered[None, None, 0],
                            tCrSFB_copy_view_filtered[None, None, 0],
                        )
                    else:
                        cute.copy(
                            smem_tiled_copy_SFB,
                            tCsSFB_p_filtered,
                            tCrSFB_copy_view_filtered,
                        )

                for k_tile in range(
                    0,
                    k_tile_iter_cnt - 1,
                    1,
                    unroll=4 if self.large_m_unroll else 2,
                ):
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )

                        if cutlass.const_expr(self.packed_expand_ahead):
                            if k_block_idx == 0:
                                # Expand-ahead: wait for stage s+1 (producer
                                # runs >= 2 stages ahead at this depth, so the
                                # wait is usually free) and expand it now, so
                                # the whole expansion overlaps stage s's MMA
                                # k-blocks instead of sitting at the stage
                                # boundary. The matching write->ldmatrix fence
                                # is the deferred barrier after the last
                                # k-block's MMA below.
                                lookahead_peek = mainloop_pipeline.consumer_try_wait(
                                    packed_b_lookahead_state
                                )
                                mainloop_pipeline.consumer_wait(
                                    packed_b_lookahead_state, lookahead_peek
                                )
                                _expand_packed_b_stage_smem(
                                    sb_base_addr,
                                    packed_b_lookahead_state.index,
                                    Int32(tidx),
                                    self.tile_shape_mnk[1],
                                    self.tile_shape_mnk[2],
                                    self.num_mma_warps * self.num_threads_per_warp,
                                    self.mma_sync_barrier,
                                )
                                packed_b_lookahead_state.advance()

                        if k_block_idx == num_k_blocks - 1:
                            if cutlass.const_expr(self.a16):
                                pipeline.PipelineAsync.consumer_release(
                                    mainloop_pipeline, mainloop_consumer_state
                                )
                            else:
                                mainloop_pipeline.consumer_release(
                                    mainloop_consumer_state
                                )
                            mainloop_consumer_state.advance()

                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )

                            tCsA_p = tCsA_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsB_p = tCsB_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            if cutlass.const_expr(not (self.block_fp8 or self.a16)):
                                tCsSFA_p = tCsSFA_tile_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                tCsSFB_p = tCsSFB_tile_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )
                            if cutlass.const_expr(
                                self.b_packed and not self.packed_expand_ahead
                            ):
                                # Shallow-pipeline fallback (< 4 stages): the
                                # new stage just became full (packed bytes
                                # only); expand before the k_block_next=0
                                # ldmatrix of this stage later in this
                                # iteration. Reads of the PREVIOUS stage
                                # finished at the prior iteration's copies, so
                                # writing here is safe. The matching barrier
                                # sits AFTER the MMA block below: the last
                                # k-block's MMA reads only registers, so it
                                # overlaps expansion stragglers instead of
                                # waiting on the barrier first. (In expand-
                                # ahead mode this stage was already expanded
                                # at k_block 0; the consumer_wait above then
                                # returns immediately.)
                                _expand_packed_b_stage_smem(
                                    sb_base_addr,
                                    mainloop_consumer_state.index,
                                    Int32(tidx),
                                    self.tile_shape_mnk[1],
                                    self.tile_shape_mnk[2],
                                    self.num_mma_warps * self.num_threads_per_warp,
                                    self.mma_sync_barrier,
                                )

                        if cutlass.const_expr(self.a16):
                            cute.gemm(
                                tiled_mma,
                                accumulators,
                                tCrA[None, None, k_block_idx],
                                tCrB[None, None, k_block_idx],
                                accumulators,
                            )
                        # Manual atom unroll: avoids hasAuxTensor address space bug
                        for _mt in range(self.num_m_tiles):
                            for _nt in range(self.num_n_tiles):
                                if cutlass.const_expr(self.a16):
                                    pass
                                elif cutlass.const_expr(self.plain_fp8):
                                    _emit_plain_fp8_dense_mma_k_block(
                                        (
                                            stage_accumulators
                                            if cutlass.const_expr(self.block_fp8)
                                            else accumulators
                                        ),
                                        tCrA,
                                        tCrB,
                                        _mt,
                                        _nt,
                                        k_block_idx,
                                    )
                                elif cutlass.const_expr(self.mxfp6_fmt_a is not None):
                                    emit_mxfp6_dense_mma_k_block(
                                        accumulators,
                                        tCrA,
                                        tCrB,
                                        tCrSFA_tile,
                                        tCrSFB_tile,
                                        _mt,
                                        _nt,
                                        k_block_idx,
                                        self.mxfp6_fmt_a,
                                        self.mxfp6_fmt_b,
                                    )
                                else:
                                    mma_atom.set(
                                        WarpField.SFA,
                                        tCrSFA_tile[None, _mt, k_block_idx].iterator,
                                    )
                                    if cutlass.const_expr(self.sfb_k_reuse):
                                        mma_atom.set(
                                            WarpField.SFB,
                                            tCrSFB_tile[None, _nt, 0].iterator,
                                        )
                                    else:
                                        mma_atom.set(
                                            WarpField.SFB,
                                            tCrSFB_tile[
                                                None, _nt, k_block_idx
                                            ].iterator,
                                        )
                                    cute.gemm(
                                        mma_atom,
                                        accumulators[None, _mt, _nt],
                                        tCrA[None, _mt, k_block_idx],
                                        tCrB[None, _nt, k_block_idx],
                                        accumulators[None, _mt, _nt],
                                    )
                        if cutlass.const_expr(self.block_fp8):
                            if k_block_idx == num_k_blocks - 1:
                                self._accumulate_block_fp8_stage(
                                    accumulators,
                                    stage_accumulators,
                                    block_fp8_coord_mn,
                                    directSFA_mkl,
                                    directSFB_nkl,
                                    tile_coord_mnl,
                                    k_tile_start + Int32(k_tile),
                                )
                        if cutlass.const_expr(self.b_packed):
                            # Deferred expansion barrier (see expansion call
                            # above): must precede the k_block_next=0 ldmatrix
                            # of the just-expanded next stage right below.
                            if k_block_idx == num_k_blocks - 1:
                                self.mma_sync_barrier.arrive_and_wait()
                        if cutlass.const_expr(self.a16_block_transposed):
                            cute.copy(
                                smem_tiled_copy_B,
                                tCsB_p[None, None, k_block_next],
                                tCrB_copy_view[None, None, k_block_next],
                            )
                        else:
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, k_block_next],
                                tCrA_copy_view[None, None, k_block_next],
                            )
                        if cutlass.const_expr(self.a16):
                            self._load_a16_b_fragment(
                                a16_weight_fragments[None, None, k_block_next],
                                a16_b_coordinates[None, None, k_block_next],
                                sBPacked,
                                sSFB,
                                a16_lut,
                                mainloop_consumer_state.index,
                                tile_coord_mnl[1],
                                k_tile_start
                                + Int32(k_tile)
                                + (1 if k_block_next == 0 else 0),
                                cute.size(directB_nkl, mode=[0]),
                                cute.size(directB_nkl, mode=[1]) * self.a16_pack_factor,
                            )
                        else:
                            cute.copy(
                                smem_tiled_copy_B,
                                tCsB_p[None, None, k_block_next],
                                tCrB_copy_view[None, None, k_block_next],
                            )

                        if k_block_idx == num_k_blocks - 1:
                            if cutlass.const_expr(not (self.block_fp8 or self.a16)):
                                # New stage acquired above: bulk-load its whole
                                # SF tile once.
                                tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                                tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                                tCrSFA_copy_view_filtered = cute.filter_zeros(
                                    tCrSFA_tile_copy_view
                                )
                                tCrSFB_copy_view_filtered = cute.filter_zeros(
                                    tCrSFB_tile_copy_view
                                )
                                cute.copy(
                                    smem_tiled_copy_SFA,
                                    tCsSFA_p_filtered,
                                    tCrSFA_copy_view_filtered,
                                )
                                if cutlass.const_expr(self.direct_sfb_representative):
                                    self._fill_replicated_sfb_fragment(
                                        tCrSFB_tile[None, None, 0],
                                        sSFB[
                                            (
                                                Int32(0),
                                                Int32(0),
                                                mainloop_consumer_state.index,
                                            )
                                        ],
                                    )
                                elif cutlass.const_expr(self.sfb_k_reuse):
                                    cute.copy(
                                        smem_tiled_copy_SFB,
                                        tCsSFB_p_filtered[None, None, 0],
                                        tCrSFB_copy_view_filtered[None, None, 0],
                                    )
                                else:
                                    cute.copy(
                                        smem_tiled_copy_SFB,
                                        tCsSFB_p_filtered,
                                        tCrSFB_copy_view_filtered,
                                    )

                # Hoist out last k_tile
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )

                    if k_block_idx == num_k_blocks - 1:
                        if cutlass.const_expr(self.a16):
                            pipeline.PipelineAsync.consumer_release(
                                mainloop_pipeline, mainloop_consumer_state
                            )
                        else:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if k_block_next > 0:
                        if cutlass.const_expr(self.a16_block_transposed):
                            cute.copy(
                                smem_tiled_copy_B,
                                tCsB_p[None, None, k_block_next],
                                tCrB_copy_view[None, None, k_block_next],
                            )
                        else:
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, k_block_next],
                                tCrA_copy_view[None, None, k_block_next],
                            )
                        if cutlass.const_expr(self.a16):
                            self._load_a16_b_fragment(
                                a16_weight_fragments[None, None, k_block_next],
                                a16_b_coordinates[None, None, k_block_next],
                                sBPacked,
                                sSFB,
                                a16_lut,
                                mainloop_consumer_state.index,
                                tile_coord_mnl[1],
                                k_tile_start + Int32(k_tile_iter_cnt - 1),
                                cute.size(directB_nkl, mode=[0]),
                                cute.size(directB_nkl, mode=[1]) * self.a16_pack_factor,
                            )
                        else:
                            cute.copy(
                                smem_tiled_copy_B,
                                tCsB_p[None, None, k_block_next],
                                tCrB_copy_view[None, None, k_block_next],
                            )
                        # SF registers for the whole stage were bulk-loaded at
                        # stage acquisition; nothing to reload per k block.
                    if cutlass.const_expr(self.a16):
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[None, None, k_block_idx],
                            tCrB[None, None, k_block_idx],
                            accumulators,
                        )
                    # Manual atom unroll: avoids hasAuxTensor address space bug
                    for _mt in range(self.num_m_tiles):
                        for _nt in range(self.num_n_tiles):
                            if cutlass.const_expr(self.a16):
                                pass
                            elif cutlass.const_expr(self.plain_fp8):
                                _emit_plain_fp8_dense_mma_k_block(
                                    (
                                        stage_accumulators
                                        if cutlass.const_expr(self.block_fp8)
                                        else accumulators
                                    ),
                                    tCrA,
                                    tCrB,
                                    _mt,
                                    _nt,
                                    k_block_idx,
                                )
                            elif cutlass.const_expr(self.mxfp6_fmt_a is not None):
                                emit_mxfp6_dense_mma_k_block(
                                    accumulators,
                                    tCrA,
                                    tCrB,
                                    tCrSFA_tile,
                                    tCrSFB_tile,
                                    _mt,
                                    _nt,
                                    k_block_idx,
                                    self.mxfp6_fmt_a,
                                    self.mxfp6_fmt_b,
                                )
                            else:
                                mma_atom.set(
                                    WarpField.SFA,
                                    tCrSFA_tile[None, _mt, k_block_idx].iterator,
                                )
                                if cutlass.const_expr(self.sfb_k_reuse):
                                    mma_atom.set(
                                        WarpField.SFB,
                                        tCrSFB_tile[None, _nt, 0].iterator,
                                    )
                                else:
                                    mma_atom.set(
                                        WarpField.SFB,
                                        tCrSFB_tile[None, _nt, k_block_idx].iterator,
                                    )
                                cute.gemm(
                                    mma_atom,
                                    accumulators[None, _mt, _nt],
                                    tCrA[None, _mt, k_block_idx],
                                    tCrB[None, _nt, k_block_idx],
                                    accumulators[None, _mt, _nt],
                                )
                    if cutlass.const_expr(self.block_fp8):
                        if k_block_idx == num_k_blocks - 1:
                            self._accumulate_block_fp8_stage(
                                accumulators,
                                stage_accumulators,
                                block_fp8_coord_mn,
                                directSFA_mkl,
                                directSFB_nkl,
                                tile_coord_mnl,
                                k_tile_start + Int32(k_tile_iter_cnt - 1),
                            )

                if cutlass.const_expr(self.swap_ab):
                    for acc_m in cutlass.range_constexpr(
                        cute.size(swap_ab_acc_mn.shape[0])
                    ):
                        for acc_n in cutlass.range_constexpr(
                            cute.size(swap_ab_acc_mn.shape[1])
                        ):
                            coord = swap_ab_coord_mn[acc_m, acc_n]
                            m_coord = (
                                tile_coord_mnl[0] * Int32(self.tile_shape_mnk[0])
                                + coord[1]
                            )
                            n_coord = (
                                tile_coord_mnl[1] * Int32(self.tile_shape_mnk[1])
                                + coord[0]
                            )
                            if m_coord < Int32(
                                directC_mnl.shape[0]
                            ) and n_coord < Int32(directC_mnl.shape[1]):
                                directC_mnl[
                                    (
                                        m_coord,
                                        n_coord,
                                        tile_coord_mnl[2],
                                    )
                                ] = epilogue_op(
                                    (alpha_value * swap_ab_acc_mn[acc_m, acc_n]).to(
                                        self.c_dtype
                                    )
                                )
                    if cutlass.const_expr(self.single_work_tile_per_cta):
                        work_tile = WorkTileInfo(
                            work_tile.tile_idx,
                            cutlass.Boolean(0),
                        )
                    else:
                        tile_sched.advance_to_next_work()
                        work_tile = tile_sched.get_current_work()

                if cutlass.const_expr(not self.swap_ab):
                    # EPILOGUE
                    _is_m_major = self.c_layout.is_m_major_c()
                    if cutlass.const_expr(self.c_dtype.width == 16):
                        copy_atom_r2s = cute.make_copy_atom(
                            cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2),
                            self.c_dtype,
                        )
                    else:
                        copy_atom_r2s = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            self.c_dtype,
                        )

                    if cutlass.const_expr(self.c_dtype.width == 16):
                        copy_atom_C = cute.make_copy_atom(
                            cute.nvgpu.warp.StMatrix8x8x16bOp(
                                self.c_layout.is_m_major_c(),
                                2,
                            ),
                            self.c_dtype,
                        )
                    else:
                        copy_atom_C = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(), self.c_dtype
                        )

                    tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(
                        copy_atom_C, tiled_mma
                    )

                    tiled_copy_r2s = cute.make_tiled_copy_S(
                        copy_atom_r2s,
                        tiled_copy_C_Atom,
                    )

                    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                    tRS_sD = thr_copy_r2s.partition_D(sC)
                    tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                    rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                    tRS_rD_layout = cute.make_layout(rD_shape[:3])
                    tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)

                    if cutlass.const_expr(self.row_scale):
                        # Retiled through the SAME copy as tRS_rAcc, so element
                        # ``i`` of tRS_cAcc is the (m,n) tile coordinate of
                        # element ``i`` of tRS_rAcc, and therefore of tRS_rD.
                        # Partitioning an identity tensor against sC directly
                        # would not be safe here: sC is a swizzled composed
                        # layout and the identity is not.
                        tRS_cAcc = tiled_copy_r2s.retile(
                            thr_mma.partition_C(
                                cute.make_identity_tensor(
                                    cute.slice_(self.tile_shape_mnk, (None, None, 0))
                                )
                            )
                        )
                        tRS_rRowScale = cute.make_rmem_tensor(
                            tRS_rD_layout.shape, self.acc_dtype
                        )

                    sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                    tcgc_for_tma_partition = cute.zipped_divide(
                        gC_mnl_slice, self.epi_tile
                    )

                    bSG_sD, bSG_gD = cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        sepi_for_tma_partition,
                        tcgc_for_tma_partition,
                    )

                    epi_rest_m = bSG_gD.shape[1][0]
                    epi_rest_n = bSG_gD.shape[1][1]
                    epi_tile_m = self.epi_tile[0]
                    epi_tile_n = self.epi_tile[1]
                    mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rAcc, mode=[1])
                    mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rAcc, mode=[2])
                    # Persistent CTAs also reuse sC across multiple stores.
                    has_multi_epi_store = cutlass.const_expr(
                        not (
                            self.single_work_tile_per_cta
                            and self.epi_stage == 1
                            and epi_rest_m == 1
                            and epi_rest_n == 1
                        )
                    )
                    tma_store_producer_group = pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        self.num_mma_warps * self.num_threads_per_warp,
                    )
                    tma_store_pipeline = pipeline.PipelineTmaStore.create(
                        num_stages=self.epi_stage,
                        producer_group=tma_store_producer_group,
                    )

                    for epi_m in cutlass.range_constexpr(epi_rest_m):
                        for epi_n in cutlass.range_constexpr(epi_rest_n):
                            MmaMPerEpiM = epi_tile_m // mma_tile_m
                            MmaNPerEpiN = epi_tile_n // mma_tile_n
                            for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                                for mma_m_in_epi in cutlass.range_constexpr(
                                    MmaMPerEpiM
                                ):
                                    mma_n = (epi_n * MmaNPerEpiN) + mma_n_in_epi
                                    mma_m = (epi_m * MmaMPerEpiM) + mma_m_in_epi
                                    tRS_rD_slice = tRS_rD[
                                        (None, mma_m_in_epi, mma_n_in_epi)
                                    ]
                                    tRS_rAcc_slice = tRS_rAcc[(None, mma_m, mma_n)]
                                    for elem_idx in cutlass.range_constexpr(
                                        cute.size(tRS_rD_slice)
                                    ):
                                        tRS_rD_slice[elem_idx] = tRS_rAcc_slice[
                                            elem_idx
                                        ]
                                    if cutlass.const_expr(self.row_scale):
                                        tRS_cAcc_slice = tRS_cAcc[(None, mma_m, mma_n)]
                                        tRS_rRowScale_slice = tRS_rRowScale[
                                            (None, mma_m_in_epi, mma_n_in_epi)
                                        ]
                                        for elem_idx in cutlass.range_constexpr(
                                            cute.size(tRS_rD_slice)
                                        ):
                                            m_coord = (
                                                tile_coord_mnl[0]
                                                * Int32(self.tile_shape_mnk[0])
                                                + tRS_cAcc_slice[elem_idx][0]
                                            )
                                            # Out-of-range rows are never
                                            # stored, so any finite filler is
                                            # fine; 1.0 keeps the multiply from
                                            # manufacturing NaN/Inf in a lane
                                            # whose accumulator is garbage.
                                            row_scale_value = cutlass.Float32(1.0)
                                            if m_coord < Int32(quantC_values.shape[0]):
                                                row_scale_value = quantC_values[
                                                    m_coord
                                                ].to(cutlass.Float32)
                                            tRS_rRowScale_slice[elem_idx] = (
                                                row_scale_value
                                            )

                            gmem_coord = (epi_m, epi_n)
                            if cutlass.const_expr(self.split_k_slices > 1):
                                if cutlass.const_expr(self.split_k_atomic_bf16):
                                    for acc_m in cutlass.range_constexpr(
                                        cute.size(split_k_acc_mn.shape[0])
                                    ):
                                        for acc_n_pair in cutlass.range_constexpr(
                                            cute.size(split_k_acc_mn.shape[1]) // 2
                                        ):
                                            acc_n0 = acc_n_pair * 2
                                            acc_n1 = acc_n0 + 1
                                            coord0 = split_k_coord_mn[acc_m, acc_n0]
                                            coord1 = split_k_coord_mn[acc_m, acc_n1]
                                            m_coord0 = (
                                                tile_coord_mnl[0]
                                                * Int32(self.tile_shape_mnk[0])
                                                + coord0[0]
                                            )
                                            n_coord0 = (
                                                tile_coord_mnl[1]
                                                * Int32(self.tile_shape_mnk[1])
                                                + coord0[1]
                                            )
                                            m_coord1 = (
                                                tile_coord_mnl[0]
                                                * Int32(self.tile_shape_mnk[0])
                                                + coord1[0]
                                            )
                                            n_coord1 = (
                                                tile_coord_mnl[1]
                                                * Int32(self.tile_shape_mnk[1])
                                                + coord1[1]
                                            )
                                            if (
                                                m_coord0 < Int32(directC_mnl.shape[0])
                                                and m_coord1
                                                < Int32(directC_mnl.shape[0])
                                                and n_coord0
                                                < Int32(directC_mnl.shape[1])
                                                and n_coord1
                                                < Int32(directC_mnl.shape[1])
                                            ):
                                                c_offset = cute.crd2idx(
                                                    (
                                                        m_coord0,
                                                        n_coord0,
                                                        Int32(0),
                                                    ),
                                                    directC_mnl.layout,
                                                )
                                                scatter_add_bf16x2(
                                                    get_ptr_as_int64(
                                                        directC_mnl,
                                                        c_offset,
                                                    ),
                                                    alpha_value
                                                    * split_k_acc_mn[acc_m, acc_n0],
                                                    alpha_value
                                                    * split_k_acc_mn[acc_m, acc_n1],
                                                )
                                        if cutlass.const_expr(
                                            cute.size(split_k_acc_mn.shape[1]) % 2 == 1
                                        ):
                                            acc_n = (
                                                cute.size(split_k_acc_mn.shape[1]) - 1
                                            )
                                            coord = split_k_coord_mn[acc_m, acc_n]
                                            m_coord = (
                                                tile_coord_mnl[0]
                                                * Int32(self.tile_shape_mnk[0])
                                                + coord[0]
                                            )
                                            n_coord = (
                                                tile_coord_mnl[1]
                                                * Int32(self.tile_shape_mnk[1])
                                                + coord[1]
                                            )
                                            if m_coord < Int32(
                                                directC_mnl.shape[0]
                                            ) and n_coord < Int32(directC_mnl.shape[1]):
                                                c_offset = cute.crd2idx(
                                                    (
                                                        m_coord,
                                                        n_coord,
                                                        Int32(0),
                                                    ),
                                                    directC_mnl.layout,
                                                )
                                                scatter_add_bf16(
                                                    get_ptr_as_int64(
                                                        directC_mnl,
                                                        c_offset,
                                                    ),
                                                    alpha_value
                                                    * split_k_acc_mn[acc_m, acc_n],
                                                )
                                else:
                                    split_idx = Int32(block_idx[1])
                                    for acc_m in cutlass.range_constexpr(
                                        cute.size(split_k_acc_mn.shape[0])
                                    ):
                                        for acc_n in cutlass.range_constexpr(
                                            cute.size(split_k_acc_mn.shape[1])
                                        ):
                                            coord = split_k_coord_mn[acc_m, acc_n]
                                            m_coord = (
                                                tile_coord_mnl[0]
                                                * Int32(self.tile_shape_mnk[0])
                                                + coord[0]
                                            )
                                            n_coord = (
                                                tile_coord_mnl[1]
                                                * Int32(self.tile_shape_mnk[1])
                                                + coord[1]
                                            )
                                            if m_coord < Int32(
                                                directC_mnl.shape[0]
                                            ) and n_coord < Int32(directC_mnl.shape[1]):
                                                directC_mnl[
                                                    (m_coord, n_coord, split_idx)
                                                ] = (
                                                    alpha_value
                                                    * split_k_acc_mn[acc_m, acc_n]
                                                )
                            else:
                                # Type conversion with alpha scaling
                                tRS_rD_out = cute.make_rmem_tensor(
                                    tRS_rD_layout.shape, self.c_dtype
                                )
                                acc_vec = tRS_rD.load()
                                # Multiply alpha in FP32 before converting to c_dtype
                                # to avoid overflow when c_dtype is FP16
                                if cutlass.const_expr(self.alpha_is_one):
                                    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                                else:
                                    acc_vec = epilogue_op(
                                        (alpha_value * acc_vec).to(self.c_dtype)
                                    )
                                if cutlass.const_expr(self.row_scale):
                                    # Deliberately a SECOND rounding to c_dtype,
                                    # applied to the already-rounded alpha
                                    # result. This reproduces the eager
                                    # ``result.mul_(inv_gs)`` it replaces, which
                                    # rounds once when the GEMM writes C and
                                    # again after the multiply. Folding the row
                                    # scale into alpha, or multiplying the fp32
                                    # accumulator, would each round only once
                                    # and would not be bit-identical.
                                    acc_vec = (
                                        acc_vec.to(cutlass.Float32)
                                        * tRS_rRowScale.load()
                                    ).to(self.c_dtype)
                                tRS_rD_out.store(acc_vec)

                                # Register to shared memory
                                epi_buffer = (epi_m * epi_rest_n + epi_n) % cute.size(
                                    tRS_sD, mode=[3]
                                )
                                if has_multi_epi_store:
                                    self.epilog_sync_barrier.arrive_and_wait()
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rD_out,
                                    tRS_sD[(None, None, None, epi_buffer)],
                                )
                                cute.arch.fence_proxy(
                                    "async.shared",
                                    space="cta",
                                )
                                self.epilog_sync_barrier.arrive_and_wait()

                                if cutlass.const_expr(self.quantize_c):
                                    quant_chunks_per_epi = self.epi_tile[1] // 32
                                    quant_active_warps = min(self.num_mma_warps, 4)
                                    quant_subgroups = quant_active_warps * 8
                                    quant_lane = Int32(tidx % 4)
                                    quant_subgroup = Int32(tidx // 4)
                                    quant_iters = (
                                        16 * quant_chunks_per_epi + quant_subgroups - 1
                                    ) // quant_subgroups
                                    if warp_idx < Int32(quant_active_warps):
                                        for quant_iter in cutlass.range(
                                            quant_iters,
                                            unroll=1,
                                            at_least_once=True,
                                        ):
                                            quant_task = quant_subgroup + Int32(
                                                quant_iter * quant_subgroups
                                            )
                                            quant_row = quant_task // Int32(
                                                quant_chunks_per_epi
                                            )
                                            quant_chunk = (
                                                quant_task
                                                - quant_row
                                                * Int32(quant_chunks_per_epi)
                                            )
                                            quant_row_valid = quant_row < Int32(
                                                quantC_values.shape[0]
                                            )
                                            quant_row_safe = quant_row
                                            if quant_row_safe >= Int32(
                                                quantC_values.shape[0]
                                            ):
                                                quant_row_safe = Int32(0)
                                            if quant_row_safe < Int32(
                                                quantC_values.shape[0]
                                            ):
                                                quant_elem0 = quant_chunk * Int32(
                                                    32
                                                ) + quant_lane * Int32(8)
                                                quant_v0 = cutlass.Float32(
                                                    sC[
                                                        (
                                                            quant_row_safe,
                                                            quant_elem0,
                                                            epi_buffer,
                                                        )
                                                    ]
                                                )
                                                quant_v1 = cutlass.Float32(
                                                    sC[
                                                        (
                                                            quant_row_safe,
                                                            quant_elem0 + Int32(1),
                                                            epi_buffer,
                                                        )
                                                    ]
                                                )
                                                quant_v2 = cutlass.Float32(
                                                    sC[
                                                        (
                                                            quant_row_safe,
                                                            quant_elem0 + Int32(2),
                                                            epi_buffer,
                                                        )
                                                    ]
                                                )
                                                quant_v3 = cutlass.Float32(
                                                    sC[
                                                        (
                                                            quant_row_safe,
                                                            quant_elem0 + Int32(3),
                                                            epi_buffer,
                                                        )
                                                    ]
                                                )
                                                quant_v4 = cutlass.Float32(
                                                    sC[
                                                        (
                                                            quant_row_safe,
                                                            quant_elem0 + Int32(4),
                                                            epi_buffer,
                                                        )
                                                    ]
                                                )
                                                quant_v5 = cutlass.Float32(
                                                    sC[
                                                        (
                                                            quant_row_safe,
                                                            quant_elem0 + Int32(5),
                                                            epi_buffer,
                                                        )
                                                    ]
                                                )
                                                quant_v6 = cutlass.Float32(
                                                    sC[
                                                        (
                                                            quant_row_safe,
                                                            quant_elem0 + Int32(6),
                                                            epi_buffer,
                                                        )
                                                    ]
                                                )
                                                quant_v7 = cutlass.Float32(
                                                    sC[
                                                        (
                                                            quant_row_safe,
                                                            quant_elem0 + Int32(7),
                                                            epi_buffer,
                                                        )
                                                    ]
                                                )
                                                quant_max = fmax_f32(
                                                    fmax_f32(
                                                        fmax_f32(
                                                            fabs_f32(quant_v0),
                                                            fabs_f32(quant_v1),
                                                        ),
                                                        fmax_f32(
                                                            fabs_f32(quant_v2),
                                                            fabs_f32(quant_v3),
                                                        ),
                                                    ),
                                                    fmax_f32(
                                                        fmax_f32(
                                                            fabs_f32(quant_v4),
                                                            fabs_f32(quant_v5),
                                                        ),
                                                        fmax_f32(
                                                            fabs_f32(quant_v6),
                                                            fabs_f32(quant_v7),
                                                        ),
                                                    ),
                                                )
                                                for (
                                                    quant_shift
                                                ) in cutlass.range_constexpr(2):
                                                    quant_max = fmax_f32(
                                                        quant_max,
                                                        cute.arch.shuffle_sync_bfly(
                                                            quant_max,
                                                            offset=1 << quant_shift,
                                                        ),
                                                    )
                                                _, quant_scale_byte = pow2_ceil_ue8m0(
                                                    quant_max
                                                    * cutlass.Float32(
                                                        1.0 / FLOAT8_E4M3_MAX
                                                    )
                                                )
                                                if quant_max == cutlass.Float32(0.0):
                                                    quant_scale_byte = cutlass.Uint32(
                                                        127
                                                    )
                                                quant_inv_scale = ue8m0_to_output_scale(
                                                    quant_scale_byte
                                                )
                                                quant_payload_lo = cvt_f32x4_to_e4m3x4(
                                                    quant_v0 * quant_inv_scale,
                                                    quant_v1 * quant_inv_scale,
                                                    quant_v2 * quant_inv_scale,
                                                    quant_v3 * quant_inv_scale,
                                                )
                                                quant_payload_hi = cvt_f32x4_to_e4m3x4(
                                                    quant_v4 * quant_inv_scale,
                                                    quant_v5 * quant_inv_scale,
                                                    quant_v6 * quant_inv_scale,
                                                    quant_v7 * quant_inv_scale,
                                                )
                                                quant_payload = (
                                                    Uint64(quant_payload_hi)
                                                    << Uint64(32)
                                                ) | Uint64(quant_payload_lo)
                                                quant_global_chunk = (
                                                    tile_coord_mnl[2]
                                                    * Int32(directC_mnl.shape[1] // 32)
                                                    + tile_coord_mnl[1]
                                                    * Int32(
                                                        self.tile_shape_mnk[1] // 32
                                                    )
                                                    + Int32(
                                                        epi_n * quant_chunks_per_epi
                                                    )
                                                    + quant_chunk
                                                )
                                                quant_global_col = (
                                                    quant_global_chunk * Int32(32)
                                                    + quant_lane * Int32(8)
                                                )
                                                if quant_row_valid:
                                                    st_global_u64(
                                                        get_ptr_as_int64(
                                                            quantC_values,
                                                            quant_row
                                                            * Int32(
                                                                quantC_values.shape[1]
                                                            )
                                                            + quant_global_col,
                                                        ),
                                                        quant_payload,
                                                    )
                                                    if quant_lane == Int32(0):
                                                        quant_scale = cutlass.Uint8(
                                                            quant_scale_byte
                                                        ).bitcast(cutlass.Float8E8M0FNU)
                                                        quantC_scale_rows[
                                                            (
                                                                quant_row,
                                                                quant_global_chunk,
                                                            )
                                                        ] = quant_scale
                                                        quantC_scale_mma[
                                                            (
                                                                quant_row,
                                                                Int32(0),
                                                                Int32(0),
                                                                quant_global_chunk
                                                                % Int32(4),
                                                                quant_global_chunk
                                                                // Int32(4),
                                                                Int32(0),
                                                            )
                                                        ] = quant_scale

                                    # Only synchronize before a persistent CTA
                                    # reuses sC for another work tile. Kernel
                                    # completion orders terminal-tile stores.
                                    if cutlass.const_expr(
                                        self.single_work_tile_per_cta
                                    ):
                                        work_tile = WorkTileInfo(
                                            work_tile.tile_idx,
                                            cutlass.Boolean(0),
                                        )
                                    else:
                                        tile_sched.advance_to_next_work()
                                        work_tile = tile_sched.get_current_work()
                                    if work_tile.is_valid_tile:
                                        self.epilog_sync_barrier.arrive_and_wait()

                                # Copy from shared memory to global memory
                                if cutlass.const_expr(
                                    self.use_m1_non_tma_c and not self.quantize_c
                                ):
                                    for n_iter in cutlass.range_constexpr(
                                        (
                                            self.epi_tile[1]
                                            + self.num_mma_warps
                                            * self.num_threads_per_warp
                                            - 1
                                        )
                                        // (
                                            self.num_mma_warps
                                            * self.num_threads_per_warp
                                        )
                                    ):
                                        n_local = Int32(tidx) + Int32(
                                            n_iter
                                            * self.num_mma_warps
                                            * self.num_threads_per_warp
                                        )
                                        n_coord = (
                                            tile_coord_mnl[1]
                                            * Int32(self.tile_shape_mnk[1])
                                            + Int32(epi_n * self.epi_tile[1])
                                            + n_local
                                        )
                                        if n_local < Int32(
                                            self.epi_tile[1]
                                        ) and n_coord < Int32(directC_mnl.shape[1]):
                                            directC_mnl[
                                                (
                                                    Int32(0),
                                                    n_coord,
                                                    tile_coord_mnl[2],
                                                )
                                            ] = sC[(Int32(0), n_local, epi_buffer)]
                                elif cutlass.const_expr(not self.quantize_c):
                                    if warp_idx == 0:
                                        cute.copy(
                                            tma_atom_c,
                                            bSG_sD[(None, epi_buffer)],
                                            bSG_gD[(None, gmem_coord)],
                                        )
                                        if has_multi_epi_store:
                                            tma_store_pipeline.producer_commit()
                                            tma_store_pipeline.producer_acquire()

                    # Advance to the next work tile
                    if cutlass.const_expr(not self.quantize_c):
                        if cutlass.const_expr(self.single_work_tile_per_cta):
                            work_tile = WorkTileInfo(
                                work_tile.tile_idx,
                                cutlass.Boolean(0),
                            )
                        else:
                            tile_sched.advance_to_next_work()
                            work_tile = tile_sched.get_current_work()
                    if has_multi_epi_store and cutlass.const_expr(
                        self.split_k_slices == 1
                    ):
                        tma_store_pipeline.producer_tail()

        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                if cutlass.const_expr(
                    self.load_path == "tma"
                    and not self.use_m1_non_tma_a
                    and not self.fused_quant_a
                    and not self.direct_m1_wo_a_inputs
                ):
                    tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                if cutlass.const_expr(self.load_path == "tma"):
                    tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                if cutlass.const_expr(
                    self.load_path == "tma"
                    and not (self.block_fp8 or self.a16)
                    and not self.use_m1_non_tma_sfa
                    and not self.fused_quant_a
                    and not self.manual_bk64_sf
                    and not self.direct_sfa_prefix
                ):
                    sfa_tile_coord_m = tile_coord_mnl[0] // self.sfa_tiles_per_block
                    tAgSFA_mkl = tAgSFA[
                        (None, sfa_tile_coord_m, None, tile_coord_mnl[2])
                    ]
                if cutlass.const_expr(
                    self.load_path == "tma"
                    and not (self.block_fp8 or self.a16)
                    and not self.manual_bk64_sf
                    and not self.direct_sfb_representative
                ):
                    sfb_tile_coord_n = tile_coord_mnl[1] // self.sfb_tiles_per_block
                    tBgSFB_nkl = tBgSFB[
                        (None, sfb_tile_coord_n, None, tile_coord_mnl[2])
                    ]
                if cutlass.const_expr(self.load_path == "cpasync"):
                    cpasync_sfa_tile_coord_m = (
                        tile_coord_mnl[0] // self.sfa_tiles_per_block
                    )
                    cpasync_sfb_tile_coord_n = (
                        tile_coord_mnl[1] // self.sfb_tiles_per_block
                    )

                mainloop_producer_state.reset_count()

                if cutlass.const_expr(self.a_bf16_fused):
                    fq_lane = Int32(tidx % self.num_threads_per_warp)
                    full_k = Int32(directX_bf16.shape[1])
                    nvec = full_k // Int32(8)
                    bf16_base = get_ptr_as_int64(directX_bf16, Int32(0))
                    local_amax = cutlass.Float32(0.0)
                    i_vec = fq_lane
                    while i_vec < nvec:
                        w0, w1, w2, w3 = ld_global_v4_u32(
                            bf16_base + cutlass.Int64(i_vec) * cutlass.Int64(16)
                        )
                        for w in (w0, w1, w2, w3):
                            hi = u32_as_f32(w & Uint32(0x7FFF0000))
                            lo = u32_as_f32((w << Uint32(16)) & Uint32(0x7FFF0000))
                            local_amax = fmax_f32(local_amax, fmax_f32(hi, lo))
                        i_vec += Int32(self.num_threads_per_warp)
                    fused_amax = warp_reduce(local_amax, fmax_f32)
                    fused_amax_c = fmax_f32(fused_amax, cutlass.Float32(1e-6))
                    fused_gs = cutlass.Float32(self._fused_gs_num) / fused_amax_c
                    _fq_k_base = Int32(0)
                    _fq_sa_stage = Int32(0)
                    _fq_packed_scales = Uint32(0)
                    _fq_k_abs = Int32(0)
                    _fq_val = cutlass.Float32(0.0)
                    _fq_bmax = cutlass.Float32(0.0)
                    _fq_su32 = Uint32(0)
                    _fq_inv = cutlass.Float32(0.0)
                    _fq_scaled = cutlass.Float32(0.0)
                    _fq_pair = Uint32(0)
                    _fq_code = Uint8(0)
                    _fq_ssfa_stage = Int32(0)
                    _fq_lin = Int32(0)
                    _fq_m = Int32(0)
                    _fq_sg_idx = Int32(0)
                    _fq_sf_off = Int32(0)
                    _fq_sb = Uint8(0)
                    _fq_sfa_sg = self.tile_shape_mnk[2] // self.sf_vec_size
                    _fq_sfa_slots = self.sfa_tile_shape_mk[0] * _fq_sfa_sg
                    _fq_sg = 0
                    _fq_si = 0

                for _k_tile in range(0, k_tile_iter_cnt, 1, unroll=2):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    if cutlass.const_expr(self.load_path == "tma"):
                        # Order consumers' generic-proxy reads before the TMA refill.
                        cute.arch.fence_view_async_shared()

                    k_tile_global = k_tile_start + mainloop_producer_state.count
                    if cutlass.const_expr(self.load_path == "tma"):
                        tBgB_k = tBgB_nkl[(None, k_tile_global)]
                        tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]
                        if cutlass.const_expr(
                            not self.use_m1_non_tma_a
                            and not self.fused_quant_a
                            and not self.direct_m1_wo_a_inputs
                        ):
                            tAgA_k = tAgA_mkl[(None, k_tile_global)]
                            tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                        if cutlass.const_expr(
                            not (self.block_fp8 or self.a16)
                            and not self.use_m1_non_tma_sfa
                            and not self.fused_quant_a
                            and not self.manual_bk64_sf
                            and not self.direct_sfa_prefix
                        ):
                            tAgSFA_k = tAgSFA_mkl[(None, k_tile_global)]
                            tAsSFA_pipe = tAsSFA[(None, mainloop_producer_state.index)]

                        if cutlass.const_expr(
                            not (self.block_fp8 or self.a16)
                            and not self.manual_bk64_sf
                            and not self.direct_sfb_representative
                        ):
                            tBgSFB_k = tBgSFB_nkl[(None, k_tile_global)]
                            tBsSFB_pipe = tBsSFB[(None, mainloop_producer_state.index)]

                        if cutlass.const_expr(self.fused_quant_a and self.b_tile_major):
                            # Start the large weight transfer before synchronous
                            # A quantization. SFB stays below as a post-fence
                            # doorbell because TMA producer_commit is a no-op.
                            cute.copy(
                                tma_atom_b,
                                tBgB_k,
                                tBsB_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                                cache_policy=Int64(0x12F0000000000000),
                            )

                    if cutlass.const_expr(self.load_path == "cpasync"):
                        tAgA_cpasync_k = tAgA_cpasync_mkl[
                            (
                                None,
                                None,
                                None,
                                tile_coord_mnl[0],
                                k_tile_global,
                                tile_coord_mnl[2],
                            )
                        ]
                        tAsA_cpasync_pipe = tAsA_cpasync[
                            (None, None, None, mainloop_producer_state.index)
                        ]
                        tAcA_cpasync_k = cute.slice_(
                            tAcA_cpasync_mkl,
                            (
                                None,
                                None,
                                None,
                                tile_coord_mnl[0],
                                k_tile_global,
                                tile_coord_mnl[2],
                            ),
                        )
                        tBgB_cpasync_k = tBgB_cpasync_nkl[
                            (
                                None,
                                None,
                                None,
                                tile_coord_mnl[1],
                                k_tile_global,
                                tile_coord_mnl[2],
                            )
                        ]
                        tBsB_cpasync_pipe = tBsB_cpasync[
                            (None, None, None, mainloop_producer_state.index)
                        ]
                        tBcB_cpasync_k = cute.slice_(
                            tBcB_cpasync_nkl,
                            (
                                None,
                                None,
                                None,
                                tile_coord_mnl[1],
                                k_tile_global,
                                tile_coord_mnl[2],
                            ),
                        )
                        tAgSFA_cpasync_k = cute.filter_zeros(
                            tAgSFA_cpasync_mkl[
                                (
                                    None,
                                    None,
                                    None,
                                    cpasync_sfa_tile_coord_m,
                                    k_tile_global,
                                    tile_coord_mnl[2],
                                )
                            ]
                        )
                        tAsSFA_cpasync_pipe = cute.filter_zeros(
                            tAsSFA_cpasync[
                                (None, None, None, mainloop_producer_state.index)
                            ]
                        )
                        tAcSFA_cpasync_k = cute.filter_zeros(
                            cute.slice_(
                                tAcSFA_cpasync_mkl,
                                (
                                    None,
                                    None,
                                    None,
                                    cpasync_sfa_tile_coord_m,
                                    k_tile_global,
                                    tile_coord_mnl[2],
                                ),
                            )
                        )
                        tBgSFB_cpasync_k = cute.filter_zeros(
                            tBgSFB_cpasync_nkl[
                                (
                                    None,
                                    None,
                                    None,
                                    cpasync_sfb_tile_coord_n,
                                    k_tile_global,
                                    tile_coord_mnl[2],
                                )
                            ]
                        )
                        tBsSFB_cpasync_pipe = cute.filter_zeros(
                            tBsSFB_cpasync[
                                (None, None, None, mainloop_producer_state.index)
                            ]
                        )
                        tBcSFB_cpasync_k = cute.filter_zeros(
                            cute.slice_(
                                tBcSFB_cpasync_nkl,
                                (
                                    None,
                                    None,
                                    None,
                                    cpasync_sfb_tile_coord_n,
                                    k_tile_global,
                                    tile_coord_mnl[2],
                                ),
                            )
                        )
                        self._cpasync_copy_2d(
                            cpasync_tiled_copy_A,
                            tAgA_cpasync_k,
                            tAsA_cpasync_pipe,
                            tAcA_cpasync_k,
                            Int32(directA_mkl.shape[0]),
                            True,
                        )
                        self._cpasync_copy_2d(
                            cpasync_tiled_copy_B,
                            tBgB_cpasync_k,
                            tBsB_cpasync_pipe,
                            tBcB_cpasync_k,
                            Int32(directC_mnl.shape[1]),
                            True,
                        )
                        self._scale_copy_2d(
                            cpasync_tiled_copy_SF,
                            tAgSFA_cpasync_k,
                            tAsSFA_cpasync_pipe,
                            tAcSFA_cpasync_k,
                            Int32(directA_mkl.shape[0]),
                        )
                        self._scale_copy_2d(
                            cpasync_tiled_copy_SF,
                            tBgSFB_cpasync_k,
                            tBsSFB_cpasync_pipe,
                            tBcSFB_cpasync_k,
                            Int32(directC_mnl.shape[1]),
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                    elif cutlass.const_expr(
                        self.fused_quant_a and self.fused_quant_a_wide
                    ):
                        # M=1 wide layout: 4 lanes cooperate on each 32-value
                        # scale block (16B load per lane, butterfly max), so a
                        # k-tile quantizes with 16 lanes instead of 4 and stops
                        # throttling deep-K producer pipelines. Lanes 16..31
                        # mirror blocks 0..3 (clamped index, stores predicated
                        # off) so the warp stays converged at the shuffles.
                        lane = Int32(tidx % self.num_threads_per_warp)
                        scale_group_raw = lane // Int32(4)
                        scale_group = scale_group_raw % Int32(4)
                        lane4 = lane % Int32(4)
                        row_global = tile_coord_mnl[0] * Int32(self.tile_shape_mnk[0])
                        # Uniform across the warp: at M=1 there is a single
                        # m tile, so this only guards the degenerate case.
                        if row_global < Int32(quantA_mkl.shape[0]):
                            values = cute.make_rmem_tensor((8,), cutlass.Float32)
                            k_local0 = scale_group * Int32(32)
                            k_global0 = (
                                k_tile_global * Int32(self.tile_shape_mnk[2]) + k_local0
                            )
                            if cutlass.const_expr(self.fused_quant_a_inner_span > 0):
                                span = Int32(self.fused_quant_a_inner_span)
                                outer = k_global0 // span
                                linear_offset = (
                                    outer * (Int32(quantA_mkl.shape[0]) * span)
                                    + row_global * span
                                    + (k_global0 - outer * span)
                                )
                            else:
                                if cutlass.const_expr(
                                    self.fused_quant_a_row_stride > 0
                                ):
                                    linear_offset = (
                                        row_global
                                        * Int32(self.fused_quant_a_row_stride)
                                        + k_global0
                                    )
                                else:
                                    linear_offset = (
                                        row_global * Int32(quantA_mkl.shape[1])
                                        + k_global0
                                    )
                                if cutlass.const_expr(self.fused_quant_a_l_stride > 0):
                                    linear_offset = linear_offset + tile_coord_mnl[
                                        2
                                    ] * Int32(self.fused_quant_a_l_stride)
                            elem0 = lane4 * Int32(8)
                            source_base = get_ptr_as_int64(
                                quantA_mkl, linear_offset + elem0
                            )
                            max_abs = cutlass.Float32(0.0)
                            w0, w1, w2, w3 = ld_global_v4_u32(source_base)
                            v0, v1 = bfloat2_to_float2_scaled(w0, cutlass.Float32(1.0))
                            v2, v3 = bfloat2_to_float2_scaled(w1, cutlass.Float32(1.0))
                            v4, v5 = bfloat2_to_float2_scaled(w2, cutlass.Float32(1.0))
                            v6, v7 = bfloat2_to_float2_scaled(w3, cutlass.Float32(1.0))
                            values[0] = v0
                            values[1] = v1
                            values[2] = v2
                            values[3] = v3
                            values[4] = v4
                            values[5] = v5
                            values[6] = v6
                            values[7] = v7
                            if cutlass.const_expr(self.fused_quant_a_inv_rope):
                                head_d0 = k_global0 % Int32(self.fused_quant_a_head_dim)
                                if head_d0 >= Int32(self.fused_quant_a_nope_dim):
                                    pos = Int32(quantA_positions[row_global])
                                    half_rope = Int32(self.fused_quant_a_rope_dim // 2)
                                    cs_base = pos * Int32(self.fused_quant_a_rope_dim)
                                    rl_half0 = (
                                        head_d0
                                        - Int32(self.fused_quant_a_nope_dim)
                                        + elem0
                                    ) // Int32(2)
                                    for pair in cutlass.range_constexpr(4):
                                        cs_idx = cs_base + rl_half0 + Int32(pair)
                                        cos_v = cutlass.Float32(quantA_cos_sin[cs_idx])
                                        sin_v = cutlass.Float32(
                                            quantA_cos_sin[cs_idx + half_rope]
                                        )
                                        v_even = values[pair * 2]
                                        v_odd = values[pair * 2 + 1]
                                        values[pair * 2] = (
                                            v_even * cos_v + v_odd * sin_v
                                        )
                                        values[pair * 2 + 1] = (
                                            v_odd * cos_v - v_even * sin_v
                                        )
                            for elem in cutlass.range_constexpr(8):
                                max_abs = fmax_f32(max_abs, fabs_f32(values[elem]))
                            for shift in cutlass.range_constexpr(2):
                                max_abs = fmax_f32(
                                    max_abs,
                                    cute.arch.shuffle_sync_bfly(
                                        max_abs, offset=1 << shift
                                    ),
                                )
                            _, scale_byte = pow2_ceil_ue8m0(
                                max_abs * cutlass.Float32(1.0 / FLOAT8_E4M3_MAX)
                            )
                            if max_abs == cutlass.Float32(0.0):
                                scale_byte = cutlass.Uint32(127)
                            inv_scale = ue8m0_to_output_scale(scale_byte)
                            payload0 = cvt_f32x4_to_e4m3x4(
                                values[0] * inv_scale,
                                values[1] * inv_scale,
                                values[2] * inv_scale,
                                values[3] * inv_scale,
                            )
                            payload1 = cvt_f32x4_to_e4m3x4(
                                values[4] * inv_scale,
                                values[5] * inv_scale,
                                values[6] * inv_scale,
                                values[7] * inv_scale,
                            )
                            if scale_group_raw < Int32(4):
                                for byte in cutlass.range_constexpr(4):
                                    raw0 = cutlass.Uint8(
                                        payload0 >> cutlass.Uint32(byte * 8)
                                    )
                                    raw1 = cutlass.Uint8(
                                        payload1 >> cutlass.Uint32(byte * 8)
                                    )
                                    sA[
                                        (
                                            Int32(0),
                                            k_local0 + elem0 + Int32(byte),
                                            mainloop_producer_state.index,
                                        )
                                    ] = raw0.bitcast(cutlass.Float8E4M3FN)
                                    sA[
                                        (
                                            Int32(0),
                                            k_local0 + elem0 + Int32(4 + byte),
                                            mainloop_producer_state.index,
                                        )
                                    ] = raw1.bitcast(cutlass.Float8E4M3FN)
                                if lane4 == Int32(0):
                                    sSFA[
                                        (
                                            Int32(0),
                                            k_local0,
                                            mainloop_producer_state.index,
                                        )
                                    ] = cutlass.Uint8(scale_byte).bitcast(
                                        cutlass.Float8E8M0FNU
                                    )
                        cute.arch.fence_proxy("async.shared", space="cta")
                    elif cutlass.const_expr(self.fused_quant_a):
                        lane = Int32(tidx % self.num_threads_per_warp)
                        row_local = lane // Int32(4)
                        scale_group = lane % Int32(4)
                        row_global = (
                            tile_coord_mnl[0] * Int32(self.tile_shape_mnk[0])
                            + row_local
                        )
                        if row_global < Int32(quantA_mkl.shape[0]):
                            values = cute.make_rmem_tensor((32,), cutlass.Float32)
                            k_local0 = scale_group * Int32(32)
                            k_global0 = (
                                k_tile_global * Int32(self.tile_shape_mnk[2]) + k_local0
                            )
                            if cutlass.const_expr(self.fused_quant_a_inner_span > 0):
                                # L-blocked source: each 32-value scale block
                                # lies within one span (span % 32 == 0).
                                span = Int32(self.fused_quant_a_inner_span)
                                outer = k_global0 // span
                                linear_offset = (
                                    outer * (Int32(quantA_mkl.shape[0]) * span)
                                    + row_global * span
                                    + (k_global0 - outer * span)
                                )
                            else:
                                if cutlass.const_expr(
                                    self.fused_quant_a_row_stride > 0
                                ):
                                    linear_offset = (
                                        row_global
                                        * Int32(self.fused_quant_a_row_stride)
                                        + k_global0
                                    )
                                else:
                                    linear_offset = (
                                        row_global * Int32(quantA_mkl.shape[1])
                                        + k_global0
                                    )
                                if cutlass.const_expr(self.fused_quant_a_l_stride > 0):
                                    linear_offset = linear_offset + tile_coord_mnl[
                                        2
                                    ] * Int32(self.fused_quant_a_l_stride)
                            source_base = get_ptr_as_int64(quantA_mkl, linear_offset)
                            max_abs = cutlass.Float32(0.0)
                            for vec in cutlass.range_constexpr(4):
                                w0, w1, w2, w3 = ld_global_v4_u32(
                                    source_base + Int64(vec * 16)
                                )
                                v0, v1 = bfloat2_to_float2_scaled(
                                    w0, cutlass.Float32(1.0)
                                )
                                v2, v3 = bfloat2_to_float2_scaled(
                                    w1, cutlass.Float32(1.0)
                                )
                                v4, v5 = bfloat2_to_float2_scaled(
                                    w2, cutlass.Float32(1.0)
                                )
                                v6, v7 = bfloat2_to_float2_scaled(
                                    w3, cutlass.Float32(1.0)
                                )
                                values[vec * 8 + 0] = v0
                                values[vec * 8 + 1] = v1
                                values[vec * 8 + 2] = v2
                                values[vec * 8 + 3] = v3
                                values[vec * 8 + 4] = v4
                                values[vec * 8 + 5] = v5
                                values[vec * 8 + 6] = v6
                                values[vec * 8 + 7] = v7
                                max_abs = fmax_f32(max_abs, fabs_f32(v0))
                                max_abs = fmax_f32(max_abs, fabs_f32(v1))
                                max_abs = fmax_f32(max_abs, fabs_f32(v2))
                                max_abs = fmax_f32(max_abs, fabs_f32(v3))
                                max_abs = fmax_f32(max_abs, fabs_f32(v4))
                                max_abs = fmax_f32(max_abs, fabs_f32(v5))
                                max_abs = fmax_f32(max_abs, fabs_f32(v6))
                                max_abs = fmax_f32(max_abs, fabs_f32(v7))
                            if cutlass.const_expr(self.fused_quant_a_inv_rope):
                                # nope_dim % 32 == 0, so a scale block is
                                # entirely nope (left as loaded) or entirely
                                # rope: de-rotate adjacent pairs with cos/sin
                                # at positions[row] and recompute the block
                                # max over the rotated values.
                                head_d0 = k_global0 % Int32(self.fused_quant_a_head_dim)
                                if head_d0 >= Int32(self.fused_quant_a_nope_dim):
                                    pos = Int32(quantA_positions[row_global])
                                    half_rope = Int32(self.fused_quant_a_rope_dim // 2)
                                    cs_base = pos * Int32(self.fused_quant_a_rope_dim)
                                    rl_half0 = (
                                        head_d0 - Int32(self.fused_quant_a_nope_dim)
                                    ) // Int32(2)
                                    max_abs = cutlass.Float32(0.0)
                                    for pair in cutlass.range_constexpr(16):
                                        cs_idx = cs_base + rl_half0 + Int32(pair)
                                        cos_v = cutlass.Float32(quantA_cos_sin[cs_idx])
                                        sin_v = cutlass.Float32(
                                            quantA_cos_sin[cs_idx + half_rope]
                                        )
                                        v_even = values[pair * 2]
                                        v_odd = values[pair * 2 + 1]
                                        values[pair * 2] = (
                                            v_even * cos_v + v_odd * sin_v
                                        )
                                        values[pair * 2 + 1] = (
                                            v_odd * cos_v - v_even * sin_v
                                        )
                                        max_abs = fmax_f32(
                                            max_abs,
                                            fabs_f32(values[pair * 2]),
                                        )
                                        max_abs = fmax_f32(
                                            max_abs,
                                            fabs_f32(values[pair * 2 + 1]),
                                        )
                            payload, scale_byte = quantize_block_fp8_mx(values, max_abs)
                            if max_abs == cutlass.Float32(0.0):
                                scale_byte = cutlass.Uint32(127)
                            for word in cutlass.range_constexpr(8):
                                for byte in cutlass.range_constexpr(4):
                                    raw_byte = cutlass.Uint8(
                                        payload[word] >> cutlass.Uint32(byte * 8)
                                    )
                                    sA[
                                        (
                                            row_local,
                                            k_local0 + Int32(word * 4 + byte),
                                            mainloop_producer_state.index,
                                        )
                                    ] = raw_byte.bitcast(cutlass.Float8E4M3FN)
                            sSFA[
                                (
                                    row_local,
                                    k_local0,
                                    mainloop_producer_state.index,
                                )
                            ] = cutlass.Uint8(scale_byte).bitcast(cutlass.Float8E8M0FNU)
                        cute.arch.fence_proxy("async.shared", space="cta")
                    elif cutlass.const_expr(self.a_bf16_fused):
                        # MX-FP6 fused activation quantization (m=1 decode):
                        # quantize this K-tile's 32-element blocks from the
                        # BF16 row straight into sA/sSFA using the row-wide
                        # gs derived in the work-tile prologue above.
                        _fq_k_base = k_tile_global * Int32(self.tile_shape_mnk[2])
                        _fq_sa_stage = mainloop_producer_state.index * Int32(
                            self.tile_shape_mnk[0] * self.tile_shape_mnk[2]
                        )
                        _fq_packed_scales = Uint32(0)

                        for _fq_sg in cutlass.range_constexpr(
                            self.tile_shape_mnk[2] // self.sf_vec_size
                        ):
                            _fq_k_abs = (
                                _fq_k_base + Int32(_fq_sg * self.sf_vec_size) + fq_lane
                            )
                            _fq_val = cutlass.Float32(
                                directX_bf16[(Int32(0), _fq_k_abs)]
                            )
                            _fq_bmax = warp_reduce(fabs_f32(_fq_val), fmax_f32)
                            _fq_su32 = fp6_block_ue8m0_exact(
                                _fq_bmax,
                                fused_gs,
                                cutlass.Float32(self._fused_fmt_max),
                            )
                            _fq_inv = ue8m0_output_scale_exact(_fq_su32, fused_gs)
                            _fq_scaled = _fq_val * _fq_inv
                            if cutlass.const_expr(self._fused_act_fmt == "e4m3"):
                                _fq_pair = cvt_f32_to_e4m3x2(
                                    cutlass.Float32(0.0), _fq_scaled
                                )
                            elif cutlass.const_expr(self._fused_act_fmt == "e3m2"):
                                _fq_pair = cvt_f32_to_e3m2x2(
                                    cutlass.Float32(0.0), _fq_scaled
                                )
                            else:
                                _fq_pair = cvt_f32_to_e2m3x2(
                                    cutlass.Float32(0.0), _fq_scaled
                                )
                            _fq_code = Uint8(_fq_pair & Uint32(0xFF))

                            # sA row-0 store (SW128 XOR is zero for row 0)
                            st_shared_u8(
                                sa_base_addr
                                + _fq_sa_stage
                                + Int32(_fq_sg * self.sf_vec_size)
                                + fq_lane,
                                _fq_code,
                            )

                            _fq_packed_scales = _fq_packed_scales | (
                                (_fq_su32 & Uint32(0xFF)) << Uint32(_fq_sg * 8)
                            )

                        # Broadcast scale bytes to all 128 SFA M-rows.
                        # SFA atom layout (Sw<3,4,3> UE8M0 block): flat offset
                        # for (m_row, sg) = (m%32)*16 + (m//32)*4 + sg.
                        _fq_sfa_sg = self.tile_shape_mnk[2] // self.sf_vec_size
                        _fq_sfa_slots = self.sfa_tile_shape_mk[0] * _fq_sfa_sg
                        _fq_ssfa_stage = mainloop_producer_state.index * Int32(
                            (self.sfa_tile_shape_mk[0] // 128) * 128 * _fq_sfa_sg
                        )
                        for _fq_si in cutlass.range_constexpr(
                            (_fq_sfa_slots + self.num_threads_per_warp - 1)
                            // self.num_threads_per_warp
                        ):
                            _fq_lin = fq_lane + Int32(
                                _fq_si * self.num_threads_per_warp
                            )
                            if _fq_lin < Int32(_fq_sfa_slots):
                                _fq_m = _fq_lin // Int32(_fq_sfa_sg)
                                _fq_sg_idx = _fq_lin - _fq_m * Int32(_fq_sfa_sg)
                                _fq_sf_off = (
                                    (_fq_m & Int32(31)) * Int32(16)
                                    + (_fq_m >> Int32(5)) * Int32(4)
                                    + _fq_sg_idx
                                )
                                _fq_sb = Uint8(
                                    (
                                        _fq_packed_scales
                                        >> (Uint32(_fq_sg_idx) * Uint32(8))
                                    )
                                    & Uint32(0xFF)
                                )
                                st_shared_u8(
                                    ssfa_base_addr + _fq_ssfa_stage + _fq_sf_off,
                                    _fq_sb,
                                )

                        cute.arch.fence_proxy("async.shared", space="cta")
                    elif cutlass.const_expr(self.use_m1_non_tma_a):
                        lane = Int32(tidx % self.num_threads_per_warp)
                        for a_iter in cutlass.range_constexpr(
                            (self.tile_shape_mnk[2] + self.num_threads_per_warp - 1)
                            // self.num_threads_per_warp
                        ):
                            k_local = lane + Int32(a_iter * self.num_threads_per_warp)
                            if k_local < Int32(self.tile_shape_mnk[2]):
                                k_coord = (
                                    k_tile_global * Int32(self.tile_shape_mnk[2])
                                    + k_local
                                )
                                sA[
                                    (
                                        Int32(0),
                                        k_local,
                                        mainloop_producer_state.index,
                                    )
                                ] = directA_mkl[
                                    (
                                        Int32(0),
                                        k_coord,
                                        tile_coord_mnl[2],
                                    )
                                ]
                    elif cutlass.const_expr(self.direct_m1_wo_a_inputs):
                        lane = Int32(tidx % self.num_threads_per_warp)
                        if lane == Int32(0):
                            a_offset = tile_coord_mnl[2] * Int32(
                                directA_mkl.shape[0]
                            ) * Int32(directA_mkl.shape[1]) + k_tile_global * Int32(
                                self.tile_shape_mnk[2]
                            )
                            cp_async_bulk_g2s_mbar(
                                shared_ptr_to_u32(
                                    elem_pointer(
                                        sA,
                                        (
                                            Int32(0),
                                            Int32(0),
                                            mainloop_producer_state.index,
                                        ),
                                    )
                                ),
                                get_ptr_as_int64(directA_mkl, a_offset),
                                Int32(128),
                                shared_ptr_to_u32(
                                    mainloop_pipeline.producer_get_barrier(
                                        mainloop_producer_state
                                    )
                                ),
                            )
                    else:
                        cute.copy(
                            tma_atom_a,
                            tAgA_k,
                            tAsA_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                        )

                    if cutlass.const_expr(self.a16):
                        if Int32(tidx % self.num_threads_per_warp) == 0:
                            logical_k = (
                                cute.size(directB_nkl, mode=[1]) * self.a16_pack_factor
                            )
                            if cutlass.const_expr(self.a16_block):
                                if cutlass.const_expr(self.a16_iq2_xxs or self.a16_q8):
                                    size_n = cute.size(directB_nkl, mode=[0])
                                    base_n = (
                                        Int64(tile_coord_mnl[1])
                                        * self.tile_shape_mnk[1]
                                        // 128
                                        * 128
                                    )
                                    block_k = 32 if self.a16_q8 else 256
                                    barrier = shared_ptr_to_u32(
                                        mainloop_pipeline.producer_get_barrier(
                                            mainloop_producer_state
                                        )
                                    )
                                    for unit in cutlass.range_constexpr(
                                        self.a16_scale_bytes // 256
                                    ):
                                        base_k = (
                                            Int64(k_tile_global)
                                            * self.tile_shape_mnk[2]
                                            // block_k
                                            + unit
                                        )
                                        if base_k >= logical_k // block_k:
                                            base_k = Int64(logical_k // block_k - 1)
                                        dst = shared_ptr_to_u32(
                                            elem_pointer(
                                                sSFB,
                                                (
                                                    unit * 256,
                                                    0,
                                                    mainloop_producer_state.index,
                                                ),
                                            )
                                        )
                                        base_chunk = 0
                                        base_row = Int64(0)
                                        if base_n + 128 <= size_n:
                                            cp_async_bulk_g2s_mbar(
                                                dst,
                                                get_ptr_as_int64(
                                                    directSFB_nkl,
                                                    (base_k * size_n + base_n) * 2,
                                                ),
                                                Int32(256),
                                                barrier,
                                            )
                                        else:
                                            # N8 alignment permits whole 16-byte transfers;
                                            # masked channels reuse a valid base span.
                                            for base_chunk in cutlass.range_constexpr(
                                                16
                                            ):
                                                base_row = base_n + Int64(
                                                    base_chunk * 8
                                                )
                                                if base_row >= size_n:
                                                    base_row = Int64(0)
                                                cp_async_bulk_g2s_mbar(
                                                    dst + Int32(base_chunk * 16),
                                                    get_ptr_as_int64(
                                                        directSFB_nkl,
                                                        (base_k * size_n + base_row)
                                                        * 2,
                                                    ),
                                                    Int32(16),
                                                    barrier,
                                                )
                                else:
                                    scale_offset = (
                                        Int64(tile_coord_mnl[1])
                                        * self.tile_shape_mnk[1]
                                        // 128
                                        * (logical_k // 256)
                                        + Int64(k_tile_global)
                                        * self.tile_shape_mnk[2]
                                        // 256
                                    ) * 1280
                                    cp_async_bulk_g2s_mbar(
                                        shared_ptr_to_u32(
                                            elem_pointer(
                                                sSFB,
                                                (0, 0, mainloop_producer_state.index),
                                            )
                                        ),
                                        get_ptr_as_int64(directSFB_nkl, scale_offset),
                                        Int32(1280),
                                        shared_ptr_to_u32(
                                            mainloop_pipeline.producer_get_barrier(
                                                mainloop_producer_state
                                            )
                                        ),
                                    )
                            else:
                                scale_k_tiles = (logical_k // self.sf_vec_size + 3) // 4
                                for unit in cutlass.range_constexpr(
                                    self.a16_scale_bytes // 512
                                ):
                                    scale_k = (
                                        Int64(k_tile_global)
                                        * self.tile_shape_mnk[2]
                                        // (self.sf_vec_size * 4)
                                        + unit
                                    )
                                    if cutlass.const_expr(
                                        logical_k % self.tile_shape_mnk[2] != 0
                                    ):
                                        if scale_k >= scale_k_tiles:
                                            scale_k = Int64(scale_k_tiles - 1)
                                    scale_offset = (
                                        Int64(tile_coord_mnl[1])
                                        * self.tile_shape_mnk[1]
                                        // 128
                                        * scale_k_tiles
                                        + scale_k
                                    ) * 512
                                    cp_async_bulk_g2s_mbar(
                                        shared_ptr_to_u32(
                                            elem_pointer(
                                                sSFB,
                                                (
                                                    unit * 512,
                                                    0,
                                                    mainloop_producer_state.index,
                                                ),
                                            )
                                        ),
                                        get_ptr_as_int64(directSFB_nkl, scale_offset),
                                        Int32(512),
                                        shared_ptr_to_u32(
                                            mainloop_pipeline.producer_get_barrier(
                                                mainloop_producer_state
                                            )
                                        ),
                                    )
                    elif cutlass.const_expr(
                        self.load_path == "cpasync"
                        or self.block_fp8
                        or self.fused_quant_a
                    ):
                        pass
                    elif cutlass.const_expr(self.a_bf16_fused):
                        pass  # scales already written above
                    elif cutlass.const_expr(self.manual_bk64_sf):
                        lane = Int32(tidx % self.num_threads_per_warp)
                        sf_k_tiles = Int32(k_tile_cnt // 2)
                        sf_k_tile = k_tile_global // Int32(2)
                        sf_k_half = k_tile_global - sf_k_tile * Int32(2)
                        sfa_tile = tile_coord_mnl[0]
                        sfb_tile = tile_coord_mnl[1] // Int32(self.sfb_tiles_per_block)
                        # directSFA/directSFB retain the physical packed-scale
                        # storage order [L, MN-tile, K128-tile, 32, 4, 4].
                        # The original BK64 address arithmetic covered only
                        # L=1; without these strides every later grouped GEMM
                        # batch silently consumed batch 0's scales.
                        sfa_l_stride = (
                            Int32(tile_sched_params.problem_shape_ntile_mnl[0])
                            * sf_k_tiles
                            * Int32(512)
                        )
                        problem_n_tiles = Int32(
                            tile_sched_params.problem_shape_ntile_mnl[1]
                        )
                        sfb_scale_n_tiles = (
                            problem_n_tiles + Int32(self.sfb_tiles_per_block - 1)
                        ) // Int32(self.sfb_tiles_per_block)
                        sfb_l_stride = sfb_scale_n_tiles * sf_k_tiles * Int32(512)
                        l_coord = tile_coord_mnl[2]
                        for sf_iter in cutlass.range_constexpr(4):
                            mn_local = lane + Int32(sf_iter * self.num_threads_per_warp)
                            mn_outer = mn_local // Int32(32)
                            mn_inner = mn_local - mn_outer * Int32(32)
                            atom_offset = (
                                mn_inner * Int32(16)
                                + mn_outer * Int32(4)
                                + sf_k_half * Int32(2)
                            )
                            sfa_offset = (
                                l_coord * sfa_l_stride
                                + (sfa_tile * sf_k_tiles + sf_k_tile) * Int32(512)
                                + atom_offset
                            )
                            sfb_offset = (
                                l_coord * sfb_l_stride
                                + (sfb_tile * sf_k_tiles + sf_k_tile) * Int32(512)
                                + atom_offset
                            )
                            sfa_pair = ld_global_b16(
                                get_ptr_as_int64(directSFA_mkl, sfa_offset)
                            )
                            # The persistent swizzle rounds N tiles up to its
                            # raster width. TMA zero-fills these extra B tiles,
                            # but the packed scale allocation contains only
                            # ceil(N/128) atoms. Scalar loads need the same
                            # bound; unit scales keep the zero-filled tile inert.
                            if cutlass.const_expr(
                                tile_sched_params.swizzle_size > 1
                                and ((directB_nkl.shape[0] + 127) // 128)
                                % tile_sched_params.swizzle_size
                                != 0
                            ):
                                sfb_pair = Uint32(0x7F7F)
                                if sfb_tile < Int32(
                                    (directB_nkl.shape[0] + 127) // 128
                                ):
                                    sfb_pair = ld_global_b16(
                                        get_ptr_as_int64(directSFB_nkl, sfb_offset)
                                    )
                            else:
                                sfb_pair = ld_global_b16(
                                    get_ptr_as_int64(directSFB_nkl, sfb_offset)
                                )
                            sfa_smem_addr = shared_ptr_to_u32(
                                elem_pointer(
                                    sSFA,
                                    (
                                        mn_local,
                                        Int32(0),
                                        mainloop_producer_state.index,
                                    ),
                                )
                            )
                            sfb_smem_addr = shared_ptr_to_u32(
                                elem_pointer(
                                    sSFB,
                                    (
                                        mn_local,
                                        Int32(0),
                                        mainloop_producer_state.index,
                                    ),
                                )
                            )
                            st_shared_u16(sfa_smem_addr, sfa_pair)
                            st_shared_u16(sfb_smem_addr, sfb_pair)
                        cute.arch.fence_proxy("async.shared", space="cta")
                    elif cutlass.const_expr(self.direct_sfa_prefix):
                        lane = Int32(tidx % self.num_threads_per_warp)
                        if lane == Int32(0):
                            scale_k_tiles = (
                                Int32(directA_mkl.shape[1]) + Int32(127)
                            ) // Int32(128)
                            scale_offset = (
                                tile_coord_mnl[2] * scale_k_tiles + k_tile_global
                            ) * Int32(512)
                            cp_async_bulk_g2s_mbar(
                                shared_ptr_to_u32(
                                    elem_pointer(
                                        sSFA,
                                        (
                                            Int32(0),
                                            Int32(0),
                                            mainloop_producer_state.index,
                                        ),
                                    )
                                ),
                                get_ptr_as_int64(directSFA_mkl, scale_offset),
                                Int32(256),
                                shared_ptr_to_u32(
                                    mainloop_pipeline.producer_get_barrier(
                                        mainloop_producer_state
                                    )
                                ),
                            )
                    elif cutlass.const_expr(self.use_m1_non_tma_sfa):
                        lane = Int32(tidx % self.num_threads_per_warp)
                        scale_groups_per_k_tile = (
                            self.tile_shape_mnk[2] // self.sf_vec_size
                        )
                        sfa_slots = self.sfa_tile_shape_mk[0] * scale_groups_per_k_tile
                        for sfa_iter in cutlass.range_constexpr(
                            (sfa_slots + self.num_threads_per_warp - 1)
                            // self.num_threads_per_warp
                        ):
                            linear = lane + Int32(sfa_iter * self.num_threads_per_warp)
                            m_local = linear // Int32(scale_groups_per_k_tile)
                            scale_group = linear - m_local * Int32(
                                scale_groups_per_k_tile
                            )
                            k_local_sfa = scale_group * Int32(self.sf_vec_size)
                            k_coord_sfa = (
                                k_tile_global * Int32(self.tile_shape_mnk[2])
                                + k_local_sfa
                            )
                            if linear < Int32(sfa_slots):
                                sSFA[
                                    (
                                        m_local,
                                        k_local_sfa,
                                        mainloop_producer_state.index,
                                    )
                                ] = directSFA_mkl[
                                    (
                                        Int32(0),
                                        k_coord_sfa,
                                        tile_coord_mnl[2],
                                    )
                                ]
                        cute.arch.fence_proxy("async.shared", space="cta")
                    else:
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_k,
                            tAsSFA_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                        )
                    if cutlass.const_expr(self.direct_sfb_representative):
                        lane = Int32(tidx % self.num_threads_per_warp)
                        if lane == Int32(0):
                            scale_n_tiles = (
                                Int32(directC_mnl.shape[1]) + Int32(127)
                            ) // Int32(128)
                            scale_k_tiles = (
                                Int32(directA_mkl.shape[1]) + Int32(127)
                            ) // Int32(128)
                            scale_n_tile = (
                                tile_coord_mnl[1] * Int32(self.tile_shape_mnk[1])
                            ) // Int32(128)
                            scale_offset = (
                                (tile_coord_mnl[2] * scale_n_tiles + scale_n_tile)
                                * scale_k_tiles
                                + k_tile_global
                            ) * Int32(512)
                            cp_async_bulk_g2s_mbar(
                                shared_ptr_to_u32(
                                    elem_pointer(
                                        sSFB,
                                        (
                                            Int32(0),
                                            Int32(0),
                                            mainloop_producer_state.index,
                                        ),
                                    )
                                ),
                                get_ptr_as_int64(directSFB_nkl, scale_offset),
                                Int32(16),
                                shared_ptr_to_u32(
                                    mainloop_pipeline.producer_get_barrier(
                                        mainloop_producer_state
                                    )
                                ),
                            )
                    if cutlass.const_expr(self.load_path == "tma"):
                        if cutlass.const_expr(
                            not (self.fused_quant_a and self.b_tile_major)
                        ):
                            if cutlass.const_expr(
                                self.a16_block
                                and not self.a16_q8
                                and self.tile_shape_mnk[2] == 256
                                and self.tile_shape_mnk[1] in (64, 128)
                                and cute.size(directB_nkl, mode=[0]) % 128 == 0
                            ):
                                if Int32(tidx % self.num_threads_per_warp) == 0:
                                    size_n = cute.size(directB_nkl, mode=[0])
                                    storage_k = cute.size(directB_nkl, mode=[1])
                                    n_start = (
                                        Int64(tile_coord_mnl[1])
                                        * self.tile_shape_mnk[1]
                                    )
                                    descriptor_offset = (
                                        Int64(tile_coord_mnl[2]) * size_n * storage_k
                                        + (
                                            n_start // 128 * (storage_k // 64)
                                            + Int64(k_tile_global)
                                        )
                                        * 8192
                                        + n_start % 128 * 64
                                    )
                                    cp_async_bulk_g2s_mbar(
                                        shared_ptr_to_u32(
                                            elem_pointer(
                                                sBPacked,
                                                (0, 0, mainloop_producer_state.index),
                                            )
                                        ),
                                        get_ptr_as_int64(
                                            directB_nkl, descriptor_offset
                                        ),
                                        Int32(self.tile_shape_mnk[1] * 64),
                                        shared_ptr_to_u32(
                                            mainloop_pipeline.producer_get_barrier(
                                                mainloop_producer_state
                                            )
                                        ),
                                    )
                            elif cutlass.const_expr(self.b_tile_major):
                                cute.copy(
                                    tma_atom_b,
                                    tBgB_k,
                                    tBsB_pipe,
                                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                        mainloop_producer_state
                                    ),
                                    cache_policy=Int64(0x12F0000000000000),
                                )
                            else:
                                if cutlass.const_expr(self.occupancy > 1):
                                    cute.copy(
                                        tma_atom_b,
                                        tBgB_k,
                                        tBsB_pipe,
                                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                            mainloop_producer_state
                                        ),
                                        cache_policy=Int64(0x12F0000000000000),
                                    )
                                else:
                                    cute.copy(
                                        tma_atom_b,
                                        tBgB_k,
                                        tBsB_pipe,
                                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                            mainloop_producer_state
                                        ),
                                    )
                        if cutlass.const_expr(
                            not (self.block_fp8 or self.a16)
                            and not self.manual_bk64_sf
                            and not self.direct_sfb_representative
                        ):
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_k,
                                tBsSFB_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                            )
                    if cutlass.const_expr(self.load_path == "cpasync"):
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                if cutlass.const_expr(self.single_work_tile_per_cta):
                    work_tile = WorkTileInfo(
                        work_tile.tile_idx,
                        cutlass.Boolean(0),
                    )
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple,
        a_dtype,
        b_dtype,
        sf_dtype,
        sfa_smem_layout,
        sfb_smem_layout,
        epi_tile: tuple,
        c_dtype,
        smem_capacity: int,
        occupancy: int,
        b_packed: bool = False,
        epi_stage_cap: int = 0,
        decode_stage3: bool = False,
        b_storage_bits: int | None = None,
        minimum_ab_stage: int = 1,
    ) -> tuple:
        epi_stage_max = (tile_shape_mnk[1] // epi_tile[1]) * (
            tile_shape_mnk[0] // epi_tile[0]
        )
        epi_stage = min(epi_stage_max, 4)
        if epi_stage_cap:
            epi_stage = max(1, min(epi_stage, epi_stage_cap))
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape)
            * (b_dtype.width if b_storage_bits is None else b_storage_bits)
            // 8
        )
        # b_packed costs no extra smem: the packed TMA tile is aliased into the
        # bottom of each sB stage and expanded in place.
        sf_bytes_per_stage = (
            cute.size(cute.filter_zeros(sfa_smem_layout).shape) * sf_dtype.width // 8
            + cute.size(cute.filter_zeros(sfb_smem_layout).shape) * sf_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        raw_ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // (ab_bytes_per_stage + sf_bytes_per_stage)
        ab_stage = max(minimum_ab_stage, min(raw_ab_stage, 4))
        if tile_shape_mnk[0] in (16, 64) and tile_shape_mnk[1] == 128:
            ab_stage = max(minimum_ab_stage, min(raw_ab_stage, 5))
        if b_packed:
            # In-place packed staging freed 12 KB/stage; deeper pipelines give
            # the producer the lookahead the packed consumer chain needs.
            ab_stage = max(minimum_ab_stage, min(raw_ab_stage, 5))
        if decode_stage3 and occupancy >= 2 and tile_shape_mnk[0] <= 16:
            ab_stage = max(minimum_ab_stage, min(raw_ab_stage, 3))
        return ab_stage, epi_stage

    @staticmethod
    def _choose_epilogue(
        mma_tiler_mn: tuple,
        mma_atom_tile_mn: tuple,
        probe,
        stages_through_smem: bool = True,
    ) -> tuple:
        """Choose an MX-FP6 epilogue tile without reducing mainloop depth."""
        full = (mma_tiler_mn[0], mma_tiler_mn[1])
        if not stages_through_smem:
            # The m=1 epilogue stores straight out of registers rather than
            # staging through sC (the use_m1_non_tma_c branches below), so a
            # sub-tiled multi-stage buffer does not describe what it does. It
            # does not fail loudly either: it writes garbage rows.
            return full, 0
        atom_m, atom_n = mma_atom_tile_mn

        def _legal(epi_tile: tuple) -> bool:
            # The epilogue walks MmaMPerEpiM = epi_m // mma_tile_m atoms per
            # staged tile. An epilogue smaller than one atom floors that to
            # zero, so the accumulator is never copied in and the TMA stores
            # whatever was in shared memory - silent NaN, not a compile error.
            return (
                epi_tile[0] >= atom_m
                and epi_tile[1] >= atom_n
                and epi_tile[0] % atom_m == 0
                and epi_tile[1] % atom_n == 0
            )

        full_ab, _ = probe(full, 0)
        half = (mma_tiler_mn[0] // 2, mma_tiler_mn[1] // 2)
        if not _legal(half):
            return full, 0
        # Halving the tile alone is inert - epi_stage_max rises by the same
        # factor - so the cap is what actually reclaims the bytes.
        half_ab, _ = probe(half, 2)
        if half_ab > full_ab:
            return half, 2
        return full, 0

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple,
        epi_tile: tuple,
        a_dtype,
        a_layout,
        b_dtype,
        b_layout,
        ab_stage: int,
        c_dtype,
        c_layout,
        epi_stage: int,
        sf_vec_size: int,
        tiled_mma,
        block_fp8: bool = False,
    ) -> tuple:
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.is_k_major_a()
        b_is_k_major = b_layout.is_k_major_b()
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]

        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        if block_fp8:
            sfa_smem_layout_staged = cute.make_layout((1, 1, ab_stage))
            sfb_smem_layout_staged = cute.make_layout((1, 1, ab_stage))
        else:
            sfa_smem_layout_staged = sm120_make_smem_layout_sfa(
                tiled_mma,
                tile_shape_mnk,
                sf_vec_size,
                ab_stage,
            )
            sfb_smem_layout_staged = sm120_make_smem_layout_sfb(
                tiled_mma,
                tile_shape_mnk,
                sf_vec_size,
                ab_stage,
            )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                c_layout,
                c_dtype,
                c_major_mode_size,
            ),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )

        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            epi_smem_layout_staged,
        )

    @staticmethod
    def _compute_grid(
        c,
        tile_shape_mnk: tuple,
        max_active_clusters,
        direct_one_m_tile_scheduler: bool,
        split_k_slices: int,
        large_m_unroll: bool,
        split_k_all_m: bool = False,
    ) -> tuple:
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            cluster_shape_mnl,
            swizzle_size=(
                16 if tile_shape_mnk == (128, 128, 64) and not large_m_unroll else 1
            ),
        )
        if cutlass.const_expr(split_k_slices > 1 or split_k_all_m):
            grid = (
                num_ctas_mnl[0] if split_k_all_m else 1,
                split_k_slices,
                num_ctas_mnl[1],
            )
        else:
            grid = utils.StaticPersistentTileScheduler.get_grid_shape(
                tile_sched_params, max_active_clusters
            )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c,
        epi_smem_layout_staged,
        epi_tile: tuple,
    ) -> tuple:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )
        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor,
        smem_layout_staged,
        smem_tile: tuple,
        mcast_dim: int,
        internal_type=None,
    ) -> tuple:
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
            internal_type=internal_type,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def can_implement(
        ab_dtype,
        sf_dtype,
        sf_vec_size: int,
        c_dtype,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
        *,
        load_path: str = "tma",
        swap_ab: bool = False,
        block_fp8: bool = False,
        weight_only: Optional[str] = None,
    ) -> bool:
        if weight_only is not None:
            return (
                weight_only in ("nvfp4", "mxfp8", "iq2_xs", "iq2_xxs", "q8_0")
                and ab_dtype == cutlass.BFloat16
                and sf_dtype == cutlass.Uint8
                and sf_vec_size == (16 if weight_only == "nvfp4" else 32)
                and c_dtype in (cutlass.BFloat16, cutlass.Float32)
                and (
                    mma_tiler_mn
                    in ((16, 64), (16, 128), (32, 64), (32, 128), (64, 64), (64, 128))
                    or weight_only in BLOCK_CODECS
                    and mma_tiler_mn in ((8, 64), (8, 128))
                )
                and cluster_shape_mn == (1, 1)
                and l == 1
                and a_major == b_major == "k"
                and c_major == "n"
                and load_path == "tma"
                and not block_fp8
                and swap_ab == (weight_only in BLOCK_CODECS and mma_tiler_mn[0] == 8)
                and n % 8 == 0
                and k
                % (
                    block_codec(weight_only).block_weights
                    if weight_only in BLOCK_CODECS
                    else 32
                )
                == 0
            )
        # The current target only supports cluster (1,1)
        if cluster_shape_mn != (1, 1):
            return False
        if load_path not in _DENSE_LOAD_PATHS:
            return False
        if block_fp8 and (
            ab_dtype != cutlass.Float8E4M3FN
            or sf_dtype != cutlass.Float32
            or sf_vec_size != 128
            or load_path != "tma"
            or swap_ab
            or l != 1
            or n % 128 != 0
        ):
            return False
        if swap_ab:
            if l != 1:
                return False
            if not (
                (ab_dtype == cutlass.Float4E2M1FN and sf_vec_size == 16)
                or (ab_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32)
            ):
                return False
        if load_path == "cpasync" and (
            ab_dtype != cutlass.Float4E2M1FN or sf_vec_size != 16 or l != 1
        ):
            return False
        # FP4 experiments allow narrow N tiles. The scale-factor smem paths
        # still allocate full 128-element SF blocks, but the live MMA tile may
        # consume only 16/32 columns.
        mma_check_mn = (mma_tiler_mn[1], mma_tiler_mn[0]) if swap_ab else mma_tiler_mn
        if ab_dtype == cutlass.Float8E4M3FN or is_mxfp6_ab_dtype(ab_dtype):
            if mma_check_mn not in (
                (16, 64),
                (16, 128),
                (32, 64),
                (32, 128),
            ):
                if mma_check_mn[0] % 64 != 0 or mma_check_mn[1] % 64 != 0:
                    return False
        elif ab_dtype == cutlass.Float4E2M1FN:
            if (
                mma_tiler_mn[0] % 64 != 0
                or mma_tiler_mn[1] % 16 != 0
                or mma_tiler_mn[1] > 128
                or (mma_tiler_mn[1] < 64 and not swap_ab)
            ):
                return False
        else:
            if mma_check_mn[0] % 64 != 0 or mma_check_mn[1] % 64 != 0:
                return False
        # The current target supports FP4, MXFP8, and MX-FP6 warp MMA paths.
        if ab_dtype not in (
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            cutlass.Float6E3M2FN,
            cutlass.Float6E2M3FN,
        ):
            return False
        # Current target MMA constraints:
        #   sf_vec_size=16 requires sf_dtype=Float8E4M3FN
        #   sf_vec_size=32 requires sf_dtype=Float8E8M0FNU
        if not block_fp8:
            if sf_vec_size == 16 and sf_dtype != cutlass.Float8E4M3FN:
                return False
            if sf_vec_size == 32 and sf_dtype != cutlass.Float8E8M0FNU:
                return False
            if ab_dtype == cutlass.Float8E4M3FN and sf_vec_size != 32:
                return False
        if is_mxfp6_ab_dtype(ab_dtype) and sf_vec_size != 32:
            return False
        # Public output is 16-bit; split-K internally uses FP32 partial output.
        if c_dtype not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32):
            return False
        # A must be K-major, B must be K-major
        if a_major != "k" or b_major != "k":
            return False
        # Alignment: K must be divisible by tile_k
        if block_fp8:
            tile_k = 128
        elif ab_dtype == cutlass.Float8E4M3FN or is_mxfp6_ab_dtype(ab_dtype):
            tile_k = mxfp6_tile_k() if is_mxfp6_ab_dtype(ab_dtype) else 128
        else:
            tile_k = sf_vec_size * 8
        return k % tile_k == 0
