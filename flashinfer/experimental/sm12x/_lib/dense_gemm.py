# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/gemm/dense.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
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

# This file is ported from the CUTLASS dense block-scaled GEMM example
# and adapted for the current Blackwell GeForce target.

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.hopper_helpers as sm90_utils
import functools
import logging
import os
import time
import torch
import triton
import triton.language as tl
from cutlass import Int32, Int64, Uint64
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp.mma import Field as WarpField
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo

from flashinfer.experimental.sm12x._lib.compiler import (
    KernelCompileSpec,
    compile as sm12x_compile,
)
from flashinfer.experimental.sm12x._lib.utils import (
    cuda_stream_from_int_or_current,
    cuda_stream_to_int,
    current_cuda_stream,
    cutlass_to_torch_dtype,
    get_cutlass_dtype,
    get_max_active_clusters,
    get_num_sm,
    make_ptr,
    sm120_make_smem_layout_sfa,
    sm120_make_smem_layout_sfb,
)
from flashinfer.experimental.sm12x._lib.intrinsics import (
    FLOAT8_E4M3_MAX,
    bfloat2_to_float2_scaled,
    cp_async_bulk_g2s_mbar,
    cvt_f32x4_to_e4m3x4,
    elem_pointer,
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    ld_global_b16,
    ld_global_v4_u32,
    pow2_ceil_ue8m0,
    quantize_block_fp8_mx,
    scatter_add_bf16,
    scatter_add_bf16x2,
    shared_ptr_to_u32,
    st_global_u32,
    st_global_u64,
    st_shared_u16,
    ue8m0_to_output_scale,
)
from flashinfer.experimental.sm12x._lib.runtime_control import (
    raise_if_kernel_resolution_frozen,
)

logger = logging.getLogger(__name__)
_WO_SPARK_MAX_SMS = 64
_DENSE_SPARK_MAX_SMS = 64


def _dense_spark_policy_for_sm_count(sm_count: int) -> bool:
    """Select dense-GEMM tactics measured on the low-SM DGX Spark class."""
    return int(sm_count) <= _DENSE_SPARK_MAX_SMS


_FLASHINFER_EXP_SM12X_TIMING = (
    os.getenv("FLASHINFER_EXP_SM12X_TIMING", "0") == "1"
    or os.getenv("VLLM_FLASHINFER_EXP_SM12X_TIMING", "0") == "1"
)
_FLASHINFER_EXP_SM12X_TIMING_THRESHOLD_MS = float(
    os.getenv(
        "FLASHINFER_EXP_SM12X_TIMING_THRESHOLD_MS",
        os.getenv("VLLM_FLASHINFER_EXP_SM12X_TIMING_THRESHOLD_MS", "0"),
    )
)
_FLASHINFER_EXP_SM12X_DENSE_SPLITK_TURBO = (
    os.getenv("FLASHINFER_EXP_SM12X_DENSE_SPLITK_TURBO", "1") == "1"
)
_FLASHINFER_EXP_SM12X_DENSE_ATOM_24 = (
    os.getenv("FLASHINFER_EXP_SM12X_DENSE_ATOM_24", "0") == "1"
)
_DENSE_LOAD_PATHS = ("tma", "cpasync")


@dataclass(frozen=True)
class _DenseGemmPlan:
    mma_tiler_mn: Tuple[int, int]
    load_path: Literal["tma", "cpasync"]
    swap_ab: bool


@triton.jit
def _reduce_split_k2_bf16_kernel(
    partials, out, total: tl.constexpr, BLOCK: tl.constexpr
) -> None:
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    accum = tl.load(partials + offs, mask=mask).to(tl.float32)
    accum += tl.load(partials + total + offs, mask=mask).to(tl.float32)
    tl.store(out + offs, accum, mask=mask)


def _reduce_split_k2_bf16(
    partials: torch.Tensor, out: torch.Tensor, *, m: int, n: int
) -> None:
    """Fused 2-way split-K FP32-partials reduction (exact); faster than torch.add.

    Falls back to torch.add when the scratch/output layout is not the expected
    [m, n, 2] / [m, n, 1] contiguous-row form.
    """
    total = int(m) * int(n)
    if (
        partials.shape == (m, n, 2)
        and partials.stride() == (n, 1, total)
        and out.shape == (m, n, 1)
        and out.stride()[0] == n
        and out.stride()[1] == 1
    ):
        block = 1024
        grid = (triton.cdiv(total, block),)
        _reduce_split_k2_bf16_kernel[grid](partials, out, total, BLOCK=block)
    else:
        torch.add(partials[:, :, 0], partials[:, :, 1], out=out[:, :, 0])


# @dsl_user_op on PersistentTileSchedulerParams.__init__ can rename attributes
# (e.g. raster_along_m -> _raster_along_m, cluster_shape_major_fdd ->
# cluster_shape_m_fdd) but __extract_mlir_values__ (used by TVM-FFI)
# still references the original names.
_orig_extract = utils.PersistentTileSchedulerParams.__extract_mlir_values__

# Map of source-code attr name -> runtime attr name set by @dsl_user_op
_ATTR_RENAMES = {
    "raster_along_m": "_raster_along_m",
    "cluster_shape_major_fdd": "cluster_shape_m_fdd",
    "cluster_shape_minor_fdd": "cluster_shape_n_fdd",
}


def _patched_extract(self):
    for src_name, dst_name in _ATTR_RENAMES.items():
        if not hasattr(self, src_name) and hasattr(self, dst_name):
            setattr(self, src_name, getattr(self, dst_name))
    return _orig_extract(self)


utils.PersistentTileSchedulerParams.__extract_mlir_values__ = _patched_extract


def _convert_layout_acc_mn(
    acc_layout: cute.Layout, transpose: bool = False
) -> cute.Layout:
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),
        *acc_layout_col_major.stride[3:],
    )
    if cutlass.const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


def _reshape_acc_to_mn(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(
        acc.iterator, _convert_layout_acc_mn(acc.layout, transpose=transpose)
    )


@dataclass(frozen=True)
class _DenseGemmPolicy:
    single_work_tile_per_cta: bool
    direct_one_m_tile_scheduler: bool
    use_m1_non_tma: bool
    split_k_slices: int
    split_k_atomic_bf16: bool
    large_m_unroll: bool


def _max_active_clusters_for(
    cluster_shape_mn: Tuple[int, int],
    sm_count: int,
) -> int:
    cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
    # For the default single-cluster launch, occupancy is bounded only by
    # the SM count. Avoid the CUTLASS hardware-info probe here because it
    # can fail on some driver/runtime combinations with INVALID_HANDLE
    # while providing no additional information for cluster_size == 1.
    return (
        sm_count
        if cluster_size == 1
        else min(get_max_active_clusters(cluster_size), sm_count)
    )


def _use_direct_sfa_live16(
    *,
    m: int,
    n: int,
    k: int,
    l: int,
    sf_vec_size: int,
    tile_k: int,
    mma_tiler_mn: Tuple[int, int],
    load_path: str,
    swap_ab: bool,
    b_tile_major: bool,
    sfb_k_reuse: bool,
    alpha_is_one: bool,
    is_mxfp8: bool,
) -> bool:
    return (
        m == 16
        and is_mxfp8
        and sf_vec_size == 32
        and tile_k == 128
        and mma_tiler_mn == (32, 64)
        and load_path == "tma"
        and not swap_ab
        and b_tile_major
        and sfb_k_reuse
        and alpha_is_one
        and (n, k, l) in ((1024, 4096, 4), (4096, 4096, 1))
    )


def _use_direct_m1_wo_a_inputs(
    *,
    m: int,
    n: int,
    k: int,
    l: int,
    sf_vec_size: int,
    tile_k: int,
    mma_tiler_mn: Tuple[int, int],
    load_path: str,
    swap_ab: bool,
    b_tile_major: bool,
    sfb_k_reuse: bool,
    is_mxfp8: bool,
) -> bool:
    return (
        m == 1
        and (n, k, l) == (1024, 4096, 4)
        and is_mxfp8
        and sf_vec_size == 32
        and tile_k == 128
        and mma_tiler_mn == (16, 64)
        and load_path == "tma"
        and not swap_ab
        and b_tile_major
        and sfb_k_reuse
    )


def _dense_gemm_policy_for(
    *,
    m: int,
    n: int,
    k: int,
    l: int,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    sm_count: int,
    expected_m: Optional[int] = None,
) -> _DenseGemmPolicy:
    max_active_clusters = _max_active_clusters_for(cluster_shape_mn, sm_count)
    tile_m, tile_n = mma_tiler_mn
    one_work_tile_per_cta = ((m + tile_m - 1) // tile_m) * (
        (n + tile_n - 1) // tile_n
    ) * l <= max_active_clusters
    single_work_tile_per_cta = (
        one_work_tile_per_cta and m < 16 and m <= tile_m and l == 1
    )
    direct_one_m_tile_scheduler = (
        one_work_tile_per_cta and m == 1 and m <= tile_m and l == 1
    )
    use_m1_non_tma = ab_dtype == cutlass.Float8E4M3FN and m == 1
    split_k_candidate = (
        single_work_tile_per_cta
        and ab_dtype == cutlass.Float8E4M3FN
        and c_dtype == cutlass.BFloat16
        and m <= 8
        and n >= 4096
        and k >= 4096
        and k % 256 == 0
        and l == 1
    )
    split_k_slices = 1
    if split_k_candidate:
        split_k_slices = (
            4 if m == 8 and (n, k) == (4096, 4096) and mma_tiler_mn == (16, 128) else 2
        )
    # A declared expected_m owns compile-time tuning for its regime. Without a
    # hint, keep the unroll choice stable throughout the existing persistent
    # scheduler regime (m >= 16); otherwise warming a large prefill and serving
    # a smaller live prefill resolves a second kernel under frozen resolution.
    # Tiny M already has distinct scheduler/load policies and is warmed
    # separately by contract. M=4096 unrolling is a Spark win, but the RTX
    # audit measured regressions on q_b and wo_b, so RTX keeps the M=8192
    # threshold.
    large_m_unroll_threshold = (
        4096 if _dense_spark_policy_for_sm_count(sm_count) else 8192
    )
    use_large_m_unroll = (
        expected_m >= large_m_unroll_threshold
        if expected_m is not None
        else not single_work_tile_per_cta
        and not direct_one_m_tile_scheduler
        and not use_m1_non_tma
    )
    return _DenseGemmPolicy(
        single_work_tile_per_cta=single_work_tile_per_cta,
        direct_one_m_tile_scheduler=direct_one_m_tile_scheduler,
        use_m1_non_tma=use_m1_non_tma,
        split_k_slices=split_k_slices,
        split_k_atomic_bf16=_FLASHINFER_EXP_SM12X_DENSE_SPLITK_TURBO,
        large_m_unroll=(
            ab_dtype == cutlass.Float8E4M3FN and use_large_m_unroll and l == 1
        ),
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
        direct_sfa_live16: bool = False,
        direct_m1_wo_a_inputs: bool = False,
        target_occupancy: int = 1,
    ):
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.mma_k = mma_k
        if tile_k is None:
            tile_k = sf_vec_size * 8
        self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], tile_k)
        self.manual_bk64_sf = sf_vec_size == 32 and tile_k == 64
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
        self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])
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
        if mma_atom_mn in ((16, 64), (16, 128)):
            self.atom_shape = (1, 2, 1)
        elif mma_atom_mn in ((32, 64), (32, 128)):
            self.atom_shape = (2, 2, 1)
        elif atom_shape_24:
            self.atom_shape = (2, 4, 1)
        else:
            self.atom_shape = (4, 2, 1)

        self.tiled_mma = None
        self.occupancy = target_occupancy
        if mma_atom_mn in ((16, 64), (16, 128)):
            self.num_mma_warps = 2
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
        if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
            mma_op = cute.nvgpu.warp.MmaMXF8Op(
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
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.mma_tile_shape_mnk,
            self.sf_vec_size,
            cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN),
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

        # Compute stage before compute smem layout
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
        )

        assert self.epi_stage > 0, (
            "epi_stage <= 0, not enough shared memory. This configuration will be skipped."
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
        )

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
    ):
        """Execute the GEMM operation.

        Args:
            a: Input tensor A
            b: Input tensor B
            sfa: Scale factor tensor for A
            sfb: Scale factor tensor for B
            c: Output tensor C
            alpha: Alpha scaling factor tensor, shape (1,), float32
            max_active_clusters: Max active clusters
            stream: CUDA stream
            epilogue_op: Elementwise epilogue function
        """
        # Setup static attributes
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.sf_dtype = sfa.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = (
            utils.LayoutEnum.ROW_MAJOR
            if self.b_tile_major
            else utils.LayoutEnum.from_tensor(b)
        )
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        self.sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa.iterator, self.sfa_layout)

        b_logical_shape = (
            cute.size(b.shape[0]),
            cute.size(b.shape[1]),
            cute.size(b.shape[2]),
        )
        self.sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_logical_shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb.iterator, self.sfb_layout)

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
        if cutlass.const_expr(self.fused_quant_a):
            tma_atom_sfa = tma_atom_b
            tma_tensor_sfa = sfa_tensor
        elif cutlass.const_expr(
            self.use_m1_non_tma_sfa or self.manual_bk64_sf or self.direct_sfa_prefix
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
        if cutlass.const_expr(self.manual_bk64_sf or self.direct_sfb_representative):
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
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
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
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
            epilogue_op,
            alpha,
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
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
    ):
        # Keep alpha in FP32 for precision
        alpha_value = alpha[0].to(cutlass.Float32)

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
                and not self.use_m1_non_tma_sfa
                and not self.fused_quant_a
                and not self.manual_bk64_sf
                and not self.direct_sfa_prefix
            ):
                cpasync.prefetch_descriptor(tma_atom_sfa)
            if cutlass.const_expr(
                self.load_path == "tma"
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
        if cutlass.const_expr(self.fused_quant_a):
            tma_copy_bytes = cute.size_in_bytes(
                self.b_dtype, b_smem_layout
            ) + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        elif cutlass.const_expr(self.manual_bk64_sf):
            tma_copy_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
            if cutlass.const_expr(not self.use_m1_non_tma_a):
                tma_copy_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)
        elif cutlass.const_expr(self.use_m1_non_tma_sfa):
            tma_copy_bytes = cute.size_in_bytes(
                self.b_dtype, b_smem_layout
            ) + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
            if cutlass.const_expr(not self.use_m1_non_tma_a):
                tma_copy_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)
        else:
            tma_copy_bytes = (
                cute.size_in_bytes(self.b_dtype, b_smem_layout)
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

        # Pipeline setup
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
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
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
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
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        if cutlass.const_expr(not self.use_m1_non_tma_sfa and not self.fused_quant_a):
            gSFA_mkl = cute.local_tile(
                mSFA_mkl,
                self.sfa_tile_shape_mk,
                (None, None, None),
            )
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

        # TMA partitions for B
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        if cutlass.const_expr(self.load_path == "tma"):
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
        if cutlass.const_expr(self.swap_ab):
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
            k_tiles_per_split = k_tile_cnt // self.split_k_slices
            k_tile_start = Int32(block_idx[1]) * Int32(k_tiles_per_split)
            k_tile_iter_cnt = k_tiles_per_split

        # Tile scheduler
        if cutlass.const_expr(self.direct_one_m_tile_scheduler):
            direct_tile_valid = Int32(block_idx[2]) < Int32(
                tile_sched_params.problem_shape_ntile_mnl[1]
            )
            work_tile = WorkTileInfo(
                (Int32(0), Int32(block_idx[2]), Int32(0)),
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
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
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
                if cutlass.const_expr(self.swap_ab):
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
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsSFA_p = tCsSFA_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                tCsSFB_p = tCsSFB_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

                tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_tile_copy_view)
                tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_tile_copy_view)

                # Whole-stage SF copy: scale bytes for all k blocks of the
                # acquired stage load in one bulk copy (per-k_block SF reloads
                # dominated the LDS/issue budget at prefill M).
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

                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
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
                            tCsSFA_p = tCsSFA_tile_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsSFB_p = tCsSFB_tile_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )

                        # Manual atom unroll: avoids hasAuxTensor address space bug
                        for _mt in range(self.num_m_tiles):
                            for _nt in range(self.num_n_tiles):
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
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )

                        if k_block_idx == num_k_blocks - 1:
                            # New stage acquired above: bulk-load its whole SF
                            # tile once. The current tile's k_block MMAs have
                            # all consumed their SF registers by this point.
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
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                        # SF registers for the whole stage were bulk-loaded at
                        # stage acquisition; nothing to reload per k block.
                    # Manual atom unroll: avoids hasAuxTensor address space bug
                    for _mt in range(self.num_m_tiles):
                        for _nt in range(self.num_n_tiles):
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

                if cutlass.const_expr(self.swap_ab):
                    acc_mn = _reshape_acc_to_mn(accumulators, transpose=True)
                    c_identity = cute.make_identity_tensor(
                        (self.tile_shape_mnk[1], self.tile_shape_mnk[0])
                    )
                    coord_mn = _reshape_acc_to_mn(
                        thr_mma.partition_C(c_identity),
                        transpose=True,
                    )
                    for acc_m in cutlass.range_constexpr(cute.size(acc_mn.shape[0])):
                        for acc_n in cutlass.range_constexpr(
                            cute.size(acc_mn.shape[1])
                        ):
                            coord = coord_mn[acc_m, acc_n]
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
                                    (alpha_value * acc_mn[acc_m, acc_n]).to(
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
                    has_multi_epi_store = cutlass.const_expr(
                        not (
                            self.epi_stage == 1 and epi_rest_m == 1 and epi_rest_n == 1
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

                            gmem_coord = (epi_m, epi_n)
                            if cutlass.const_expr(self.split_k_slices > 1):
                                acc_mn = _reshape_acc_to_mn(accumulators)
                                c_identity = cute.make_identity_tensor(
                                    cute.slice_(self.tile_shape_mnk, (None, None, 0))
                                )
                                coord_mn = _reshape_acc_to_mn(
                                    thr_mma.partition_C(c_identity)
                                )
                                if cutlass.const_expr(self.split_k_atomic_bf16):
                                    for acc_m in cutlass.range_constexpr(
                                        cute.size(acc_mn.shape[0])
                                    ):
                                        for acc_n_pair in cutlass.range_constexpr(
                                            cute.size(acc_mn.shape[1]) // 2
                                        ):
                                            acc_n0 = acc_n_pair * 2
                                            acc_n1 = acc_n0 + 1
                                            coord0 = coord_mn[acc_m, acc_n0]
                                            coord1 = coord_mn[acc_m, acc_n1]
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
                                                    alpha_value * acc_mn[acc_m, acc_n0],
                                                    alpha_value * acc_mn[acc_m, acc_n1],
                                                )
                                        if cutlass.const_expr(
                                            cute.size(acc_mn.shape[1]) % 2 == 1
                                        ):
                                            acc_n = cute.size(acc_mn.shape[1]) - 1
                                            coord = coord_mn[acc_m, acc_n]
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
                                                    alpha_value * acc_mn[acc_m, acc_n],
                                                )
                                else:
                                    split_idx = Int32(block_idx[1])
                                    for acc_m in cutlass.range_constexpr(
                                        cute.size(acc_mn.shape[0])
                                    ):
                                        for acc_n in cutlass.range_constexpr(
                                            cute.size(acc_mn.shape[1])
                                        ):
                                            coord = coord_mn[acc_m, acc_n]
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
                                                ] = alpha_value * acc_mn[acc_m, acc_n]
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

                for k_tile in range(0, k_tile_iter_cnt, 1, unroll=2):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

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
                            not self.use_m1_non_tma_sfa
                            and not self.fused_quant_a
                            and not self.manual_bk64_sf
                            and not self.direct_sfa_prefix
                        ):
                            tAgSFA_k = tAgSFA_mkl[(None, k_tile_global)]
                            tAsSFA_pipe = tAsSFA[(None, mainloop_producer_state.index)]

                        if cutlass.const_expr(
                            not self.manual_bk64_sf
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

                    if cutlass.const_expr(self.load_path == "cpasync"):
                        pass
                    elif cutlass.const_expr(self.fused_quant_a):
                        pass
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
                            if cutlass.const_expr(self.b_tile_major):
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
                            not self.manual_bk64_sf
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
    ) -> tuple:
        epi_stage_max = (tile_shape_mnk[1] // epi_tile[1]) * (
            tile_shape_mnk[0] // epi_tile[0]
        )
        epi_stage = min(epi_stage_max, 4)
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
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
        ab_stage = max(1, min(raw_ab_stage, 4))
        if tile_shape_mnk[0] in (16, 64) and tile_shape_mnk[1] == 128:
            ab_stage = max(1, min(raw_ab_stage, 5))
        return ab_stage, epi_stage

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
        if cutlass.const_expr(split_k_slices > 1):
            grid = (1, split_k_slices, num_ctas_mnl[1])
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
    ) -> bool:
        # The current target only supports cluster (1,1)
        if cluster_shape_mn != (1, 1):
            return False
        if load_path not in _DENSE_LOAD_PATHS:
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
        if ab_dtype == cutlass.Float8E4M3FN:
            if mma_check_mn not in ((16, 64), (16, 128), (32, 64), (32, 128)):
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
        # The current target supports FP4 and MXFP8 warp MMA paths.
        if ab_dtype not in (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN):
            return False
        # Current target MMA constraints:
        #   sf_vec_size=16 requires sf_dtype=Float8E4M3FN
        #   sf_vec_size=32 requires sf_dtype=Float8E8M0FNU
        if sf_vec_size == 16 and sf_dtype != cutlass.Float8E4M3FN:
            return False
        if sf_vec_size == 32 and sf_dtype != cutlass.Float8E8M0FNU:
            return False
        if ab_dtype == cutlass.Float8E4M3FN and sf_vec_size != 32:
            return False
        # Public output is 16-bit; split-K internally uses FP32 partial output.
        if c_dtype not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32):
            return False
        # A must be K-major, B must be K-major
        if a_major != "k" or b_major != "k":
            return False
        # Alignment: K must be divisible by tile_k
        tile_k = 128 if ab_dtype == cutlass.Float8E4M3FN else sf_vec_size * 8
        if k % tile_k != 0:
            return False
        return True


class _DenseGemmLaunch:
    def __init__(
        self,
        n: int,
        k: int,
        l: int,
        c_l: int,
        a_major: str,
        b_major: str,
        c_major: str,
        ab_dtype: torch.dtype,
        sf_dtype: torch.dtype,
        c_dtype: torch.dtype,
        alpha_dtype: torch.dtype,
        sf_vec_size: int,
        mma_k: int,
        tile_k: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        policy: _DenseGemmPolicy,
        sm_count: int,
        sm_version: str,
        load_path: str,
        swap_ab: bool,
        sfb_k_reuse: bool,
        b_tile_major: bool = False,
        quantize_c: bool = False,
        alpha_is_one: bool = False,
        direct_sfa_live16: bool = False,
        direct_m1_wo_a_inputs: bool = False,
        target_occupancy: int = 1,
    ):
        self._n = n
        self._k = k
        self._l = l
        self._c_l = c_l
        self._a_major = a_major
        self._b_major = b_major
        self._c_major = c_major
        self._ab_dtype = ab_dtype
        self._sf_dtype = sf_dtype
        self._c_dtype = c_dtype
        self._alpha_dtype = alpha_dtype
        self._sf_vec_size = sf_vec_size
        self._mma_k = mma_k
        self._tile_k = tile_k
        self._mma_tiler_mn = mma_tiler_mn
        self._cluster_shape_mn = cluster_shape_mn
        self._policy = policy
        self._sm_count = sm_count
        self._sm_version = sm_version
        self._load_path = load_path
        self._swap_ab = swap_ab
        # This experimental atom choice changes generated code. Capture the
        # import-time setting on the launch so both in-process resolution and
        # the persistent object cache distinguish it.
        self._atom_shape_24 = _FLASHINFER_EXP_SM12X_DENSE_ATOM_24
        self._sfb_k_reuse = sfb_k_reuse
        self._b_tile_major = b_tile_major
        self._quantize_c = quantize_c
        self._alpha_is_one = alpha_is_one
        self._direct_sfa_live16 = direct_sfa_live16
        self._direct_m1_wo_a_inputs = direct_m1_wo_a_inputs
        self._target_occupancy = target_occupancy
        if b_tile_major:
            if (n, k, l) == (1024, 4096, 4):
                self._b_tile_n = 64
            elif (n, k, l) == (4096, 4096, 1):
                self._b_tile_n = 128
            else:
                raise ValueError(
                    "tile-major B is restricted to production WO-A/WO-B shapes, "
                    f"got {(n, k, l)}"
                )
        else:
            self._b_tile_n = 0
        if not DenseGemmKernel.can_implement(
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            c_dtype,
            mma_tiler_mn,
            cluster_shape_mn,
            n,
            k,
            l,
            a_major,
            b_major,
            c_major,
            load_path=load_path,
            swap_ab=swap_ab,
        ):
            raise TypeError(
                "dense_gemm launch is unsupported with "
                f"{ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype}, "
                f"{mma_tiler_mn}, {cluster_shape_mn}, {n}, {k}, {l}, "
                f"{a_major}, {b_major}, {c_major}, "
                f"load_path={load_path}, swap_ab={swap_ab}"
            )

        self._max_active_clusters = (
            _max_active_clusters_for(self._cluster_shape_mn, sm_count)
            * self._target_occupancy
        )
        if (
            mma_tiler_mn == (32, 64)
            and tile_k == 128
            and b_tile_major
            and sfb_k_reuse
            and alpha_is_one
            and (n, k, l) in ((1024, 4096, 4), (4096, 4096, 1))
        ):
            self._max_active_clusters = min(self._max_active_clusters, 40)

    def compile_key(self) -> tuple[object, ...]:
        """Return every value that can specialize the generated kernel."""

        return (
            self._n,
            self._k,
            self._l,
            self._c_l,
            self._a_major,
            self._b_major,
            self._c_major,
            self._ab_dtype,
            self._sf_dtype,
            self._c_dtype,
            self._alpha_dtype,
            self._sf_vec_size,
            self._mma_k,
            self._tile_k,
            self._mma_tiler_mn,
            self._cluster_shape_mn,
            self._policy,
            self._sm_count,
            self._max_active_clusters,
            self._sm_version,
            self._load_path,
            self._swap_ab,
            self._atom_shape_24,
            self._sfb_k_reuse,
            self._b_tile_major,
            self._b_tile_n,
            self._quantize_c,
            self._alpha_is_one,
            self._direct_sfa_live16,
            self._direct_m1_wo_a_inputs,
            self._target_occupancy,
        )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        quant_c_values_ptr: cute.Pointer,
        quant_c_scale_rows_ptr: cute.Pointer,
        quant_c_scale_mma_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        m: cutlass.Int32,
        current_stream: cuda.CUstream,
    ):
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout(
                (m, self._k, self._l),
                order=(0, 1, 2) if self._a_major == "m" else (1, 0, 2),
            ),
        )
        if cutlass.const_expr(self._b_tile_major):
            b_layout = cute.make_layout(
                (
                    (self._b_tile_n, self._n // self._b_tile_n),
                    (128, self._k // 128),
                    self._l,
                ),
                stride=(
                    (128, self._b_tile_n * self._k),
                    (1, self._b_tile_n * 128),
                    self._n * self._k,
                ),
            )
        else:
            b_layout = cute.make_ordered_layout(
                (self._n, self._k, self._l),
                order=(0, 1, 2) if self._b_major == "n" else (1, 0, 2),
            )
        b_tensor = cute.make_tensor(b_ptr, layout=b_layout)
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout(
                (m, self._n, self._c_l),
                order=(0, 1, 2) if self._c_major == "m" else (1, 0, 2),
            ),
        )
        quant_c_width = self._n * self._l
        quant_c_chunks = max(1, (quant_c_width + 31) // 32)
        quant_c_k_tiles = max(1, (quant_c_width + 127) // 128)
        quant_c_values_tensor = cute.make_tensor(
            quant_c_values_ptr,
            layout=cute.make_ordered_layout((m, quant_c_width), order=(1, 0)),
        )
        quant_c_scale_rows_tensor = cute.make_tensor(
            quant_c_scale_rows_ptr,
            layout=cute.make_ordered_layout((m, quant_c_chunks), order=(1, 0)),
        )
        quant_c_scale_mma_tensor = cute.make_tensor(
            quant_c_scale_mma_ptr,
            layout=cute.make_layout(
                (32, 4, 1, 4, quant_c_k_tiles, 1),
                stride=(
                    16,
                    4,
                    quant_c_k_tiles * 512,
                    1,
                    512,
                    quant_c_k_tiles * 512,
                ),
            ),
        )
        alpha_tensor = cute.make_tensor(
            alpha_ptr,
            layout=cute.make_ordered_layout((1,), order=(0,)),
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, layout=cute.make_layout((1,)))
        sfb_tensor = cute.make_tensor(sfb_ptr, layout=cute.make_layout((1,)))
        policy = self._policy
        DenseGemmKernel(
            sf_vec_size=self._sf_vec_size,
            mma_tiler_mn=self._mma_tiler_mn,
            cluster_shape_mn=self._cluster_shape_mn,
            mma_k=self._mma_k,
            tile_k=self._tile_k,
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            split_k_slices=policy.split_k_slices,
            split_k_atomic_bf16=policy.split_k_atomic_bf16,
            large_m_unroll=policy.large_m_unroll,
            # M=1 FP8 benefits from normal TMA loads for A/SFA on the
            # standalone tiny-M profile. Keep C on the direct epilogue path;
            # the normal TMA store did not beat it in the DSV4F TP=2 GPU5 run.
            use_m1_non_tma_a=False,
            use_m1_non_tma_c=policy.use_m1_non_tma and not self._swap_ab,
            use_m1_non_tma_sfa=False,
            load_path=self._load_path,
            swap_ab=self._swap_ab,
            sfb_k_reuse=self._sfb_k_reuse,
            atom_shape_24=self._atom_shape_24,
            b_tile_major=self._b_tile_major,
            quantize_c=self._quantize_c,
            alpha_is_one=self._alpha_is_one,
            direct_sfa_live16=self._direct_sfa_live16,
            direct_m1_wo_a_inputs=self._direct_m1_wo_a_inputs,
            target_occupancy=self._target_occupancy,
        )(
            a_tensor,
            a_tensor,
            alpha_tensor,
            alpha_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            quant_c_values_tensor,
            quant_c_scale_rows_tensor,
            quant_c_scale_mma_tensor,
            alpha_tensor,
            self._max_active_clusters,
            current_stream,
        )


class _DenseGemmFusedQuantALaunch(_DenseGemmLaunch):
    """Small-M MXFP8 launch that quantizes BF16 A into each CTA's stages."""

    def __init__(
        self,
        *args,
        fused_quant_a_inner_span: int = 0,
        fused_quant_a_wide: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._fused_quant_a_inner_span = int(fused_quant_a_inner_span)
        self._fused_quant_a_wide = bool(fused_quant_a_wide)

    def compile_key(self) -> tuple[object, ...]:
        # Keep the fused entry point separate even if every ordinary launch
        # specialization matches. Deriving the rest from the parent key avoids
        # silently omitting a future codegen field here.
        return (
            "fused_quant_a",
            self._fused_quant_a_inner_span,
            self._fused_quant_a_wide,
            *super().compile_key(),
        )

    @cute.jit
    def __call__(
        self,
        a_placeholder_ptr: cute.Pointer,
        a_source_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_placeholder_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        m: cutlass.Int32,
        current_stream: cuda.CUstream,
    ):
        a_tensor = cute.make_tensor(
            a_placeholder_ptr,
            layout=cute.make_ordered_layout((m, self._k, 1), order=(1, 0, 2)),
        )
        a_source = cute.make_tensor(
            a_source_ptr,
            layout=cute.make_ordered_layout((m, self._k, 1), order=(1, 0, 2)),
        )
        if cutlass.const_expr(self._b_tile_major):
            b_layout = cute.make_layout(
                (
                    (self._b_tile_n, self._n // self._b_tile_n),
                    (128, self._k // 128),
                    1,
                ),
                stride=(
                    (128, self._b_tile_n * self._k),
                    (1, self._b_tile_n * 128),
                    self._n * self._k,
                ),
            )
        else:
            b_layout = cute.make_ordered_layout((self._n, self._k, 1), order=(1, 0, 2))
        b_tensor = cute.make_tensor(b_ptr, layout=b_layout)
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout((m, self._n, self._c_l), order=(1, 0, 2)),
        )
        alpha_tensor = cute.make_tensor(
            alpha_ptr,
            layout=cute.make_ordered_layout((1,), order=(0,)),
        )
        sfa_tensor = cute.make_tensor(
            sfa_placeholder_ptr, layout=cute.make_layout((1,))
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, layout=cute.make_layout((1,)))
        policy = self._policy
        DenseGemmKernel(
            sf_vec_size=self._sf_vec_size,
            mma_tiler_mn=self._mma_tiler_mn,
            cluster_shape_mn=self._cluster_shape_mn,
            mma_k=self._mma_k,
            tile_k=self._tile_k,
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            split_k_slices=policy.split_k_slices,
            split_k_atomic_bf16=policy.split_k_atomic_bf16,
            large_m_unroll=False,
            use_m1_non_tma_a=False,
            use_m1_non_tma_c=policy.use_m1_non_tma,
            use_m1_non_tma_sfa=False,
            load_path="tma",
            swap_ab=False,
            sfb_k_reuse=self._sfb_k_reuse,
            fused_quant_a=True,
            fused_quant_a_inner_span=self._fused_quant_a_inner_span,
            fused_quant_a_wide=self._fused_quant_a_wide,
            atom_shape_24=self._atom_shape_24,
            b_tile_major=self._b_tile_major,
            target_occupancy=self._target_occupancy,
        )(
            a_tensor,
            a_source,
            alpha_tensor,
            alpha_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            a_tensor,
            sfa_tensor,
            sfa_tensor,
            alpha_tensor,
            self._max_active_clusters,
            current_stream,
        )


@functools.cache
def _get_compiled_dense_gemm_fused_quant_a(
    n: int,
    k: int,
    c_dtype: Type[cutlass.Numeric],
    policy: _DenseGemmPolicy,
    mma_tiler_mn: Tuple[int, int],
    sm_count: int,
    sfb_k_reuse: bool,
    b_tile_major: bool,
    a_inner_span: int = 0,
    kernel_c_l: int = 1,
    a_wide: bool = False,
) -> Callable:
    launch = _DenseGemmFusedQuantALaunch(
        n=n,
        k=k,
        l=1,
        c_l=kernel_c_l,
        a_major="k",
        b_major="k",
        c_major="n",
        ab_dtype=cutlass.Float8E4M3FN,
        sf_dtype=cutlass.Float8E8M0FNU,
        c_dtype=c_dtype,
        alpha_dtype=cutlass.Float32,
        sf_vec_size=32,
        mma_k=32,
        tile_k=128,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        policy=policy,
        sm_count=sm_count,
        sm_version="sm_120",
        load_path="tma",
        swap_ab=False,
        sfb_k_reuse=sfb_k_reuse,
        b_tile_major=b_tile_major,
        fused_quant_a_inner_span=a_inner_span,
        fused_quant_a_wide=a_wide,
    )
    compile_key = launch.compile_key()
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=compile_key,
    )
    placeholders = [16] * 7
    compiled = sm12x_compile(
        launch,
        make_ptr(
            cutlass.Float8E4M3FN,
            placeholders[0],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.BFloat16, placeholders[1], cute.AddressSpace.gmem, assumed_align=16
        ),
        make_ptr(
            cutlass.Float8E4M3FN,
            placeholders[2],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float8E8M0FNU,
            placeholders[3],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float8E8M0FNU,
            placeholders[4],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(c_dtype, placeholders[5], cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(
            cutlass.Float32, placeholders[6], cute.AddressSpace.gmem, assumed_align=16
        ),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "gemm.dense_fused_quant_a", 4, compile_key
        ),
    )

    def tensor_api(
        source: torch.Tensor,
        b: torch.Tensor,
        sfb: torch.Tensor,
        out: torch.Tensor,
        alpha: torch.Tensor,
        stream_int: Optional[int],
    ) -> torch.Tensor:
        source_ptr = source.data_ptr()
        compiled(
            make_ptr(
                cutlass.Float8E4M3FN,
                source_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.BFloat16, source_ptr, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Float8E4M3FN,
                b.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float8E8M0FNU,
                source_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float8E8M0FNU,
                sfb.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(c_dtype, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(
                cutlass.Float32,
                alpha.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            int(source.shape[0]),
            cuda_stream_from_int_or_current(stream_int),
        )
        return out

    return tensor_api


class _DenseGemmFusedQuantAGroupedLaunch(_DenseGemmLaunch):
    """Grouped small-M MXFP8 launch quantizing a strided (optionally
    inverse-RoPE) BF16 A source into each CTA's stages (WO-A)."""

    def __init__(
        self,
        *args,
        fused_quant_a_row_stride: int,
        fused_quant_a_l_stride: int,
        fused_quant_a_inv_rope: bool,
        fused_quant_a_head_dim: int,
        fused_quant_a_nope_dim: int,
        fused_quant_a_rope_dim: int,
        fused_quant_a_wide: bool,
        positions_dtype: Type[cutlass.Numeric],
        cos_sin_dtype: Type[cutlass.Numeric],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._fused_quant_a_row_stride = int(fused_quant_a_row_stride)
        self._fused_quant_a_l_stride = int(fused_quant_a_l_stride)
        self._fused_quant_a_inv_rope = bool(fused_quant_a_inv_rope)
        self._fused_quant_a_head_dim = int(fused_quant_a_head_dim)
        self._fused_quant_a_nope_dim = int(fused_quant_a_nope_dim)
        self._fused_quant_a_rope_dim = int(fused_quant_a_rope_dim)
        self._fused_quant_a_wide = bool(fused_quant_a_wide)
        self._positions_dtype = positions_dtype
        self._cos_sin_dtype = cos_sin_dtype

    def compile_key(self) -> tuple[object, ...]:
        return (
            "fused_quant_a_grouped",
            self._fused_quant_a_row_stride,
            self._fused_quant_a_l_stride,
            self._fused_quant_a_inv_rope,
            self._fused_quant_a_head_dim,
            self._fused_quant_a_nope_dim,
            self._fused_quant_a_rope_dim,
            self._fused_quant_a_wide,
            self._positions_dtype,
            self._cos_sin_dtype,
            *super().compile_key(),
        )

    @cute.jit
    def __call__(
        self,
        a_placeholder_ptr: cute.Pointer,
        a_source_ptr: cute.Pointer,
        positions_ptr: cute.Pointer,
        cos_sin_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_placeholder_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        m: cutlass.Int32,
        cos_sin_len: cutlass.Int32,
        current_stream: cuda.CUstream,
    ):
        a_tensor = cute.make_tensor(
            a_placeholder_ptr,
            layout=cute.make_ordered_layout((m, self._k, 1), order=(1, 0, 2)),
        )
        a_source = cute.make_tensor(
            a_source_ptr,
            layout=cute.make_ordered_layout((m, self._k, 1), order=(1, 0, 2)),
        )
        positions_tensor = cute.make_tensor(
            positions_ptr, layout=cute.make_layout((m,))
        )
        cos_sin_tensor = cute.make_tensor(
            cos_sin_ptr, layout=cute.make_layout((cos_sin_len,))
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout(
                (self._n, self._k, self._l), order=(1, 0, 2)
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout((m, self._n, self._c_l), order=(1, 0, 2)),
        )
        alpha_tensor = cute.make_tensor(
            alpha_ptr,
            layout=cute.make_ordered_layout((1,), order=(0,)),
        )
        sfa_tensor = cute.make_tensor(
            sfa_placeholder_ptr, layout=cute.make_layout((1,))
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, layout=cute.make_layout((1,)))
        policy = self._policy
        DenseGemmKernel(
            sf_vec_size=self._sf_vec_size,
            mma_tiler_mn=self._mma_tiler_mn,
            cluster_shape_mn=self._cluster_shape_mn,
            mma_k=self._mma_k,
            tile_k=self._tile_k,
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            split_k_slices=policy.split_k_slices,
            split_k_atomic_bf16=policy.split_k_atomic_bf16,
            large_m_unroll=False,
            use_m1_non_tma_a=False,
            use_m1_non_tma_c=policy.use_m1_non_tma,
            use_m1_non_tma_sfa=False,
            load_path="tma",
            swap_ab=False,
            sfb_k_reuse=self._sfb_k_reuse,
            fused_quant_a=True,
            fused_quant_a_row_stride=self._fused_quant_a_row_stride,
            fused_quant_a_l_stride=self._fused_quant_a_l_stride,
            fused_quant_a_inv_rope=self._fused_quant_a_inv_rope,
            fused_quant_a_head_dim=self._fused_quant_a_head_dim,
            fused_quant_a_nope_dim=self._fused_quant_a_nope_dim,
            fused_quant_a_rope_dim=self._fused_quant_a_rope_dim,
            fused_quant_a_wide=self._fused_quant_a_wide,
            atom_shape_24=self._atom_shape_24,
            target_occupancy=self._target_occupancy,
        )(
            a_tensor,
            a_source,
            positions_tensor,
            cos_sin_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            a_tensor,
            sfa_tensor,
            sfa_tensor,
            alpha_tensor,
            self._max_active_clusters,
            current_stream,
        )


def _cutlass_positions_dtype(dtype: torch.dtype) -> Type[cutlass.Numeric]:
    if dtype == torch.int64:
        return cutlass.Int64
    if dtype == torch.int32:
        return cutlass.Int32
    raise ValueError(f"fused inv-RoPE positions must be int32/int64, got {dtype}")


def _cutlass_cos_sin_dtype(dtype: torch.dtype) -> Type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float32:
        return cutlass.Float32
    raise ValueError(f"fused inv-RoPE cos/sin cache must be bf16/fp32, got {dtype}")


@functools.cache
def _get_compiled_dense_gemm_fused_quant_a_grouped(
    n: int,
    k: int,
    l: int,
    policy: _DenseGemmPolicy,
    mma_tiler_mn: Tuple[int, int],
    sm_count: int,
    sfb_k_reuse: bool,
    a_row_stride: int,
    a_l_stride: int,
    inv_rope: bool,
    head_dim: int,
    nope_dim: int,
    rope_dim: int,
    a_wide: bool,
    positions_dtype: Type[cutlass.Numeric],
    cos_sin_dtype: Type[cutlass.Numeric],
) -> Callable:
    launch = _DenseGemmFusedQuantAGroupedLaunch(
        n=n,
        k=k,
        l=l,
        c_l=l,
        a_major="k",
        b_major="k",
        c_major="n",
        ab_dtype=cutlass.Float8E4M3FN,
        sf_dtype=cutlass.Float8E8M0FNU,
        c_dtype=cutlass.BFloat16,
        alpha_dtype=cutlass.Float32,
        sf_vec_size=32,
        mma_k=32,
        tile_k=128,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        policy=policy,
        sm_count=sm_count,
        sm_version="sm_120",
        load_path="tma",
        swap_ab=False,
        sfb_k_reuse=sfb_k_reuse,
        fused_quant_a_row_stride=a_row_stride,
        fused_quant_a_l_stride=a_l_stride,
        fused_quant_a_inv_rope=inv_rope,
        fused_quant_a_head_dim=head_dim,
        fused_quant_a_nope_dim=nope_dim,
        fused_quant_a_rope_dim=rope_dim,
        fused_quant_a_wide=a_wide,
        positions_dtype=positions_dtype,
        cos_sin_dtype=cos_sin_dtype,
    )
    compile_key = launch.compile_key()
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=compile_key,
    )
    placeholders = [16] * 9
    compiled = sm12x_compile(
        launch,
        make_ptr(
            cutlass.Float8E4M3FN,
            placeholders[0],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.BFloat16, placeholders[1], cute.AddressSpace.gmem, assumed_align=16
        ),
        make_ptr(
            positions_dtype, placeholders[2], cute.AddressSpace.gmem, assumed_align=8
        ),
        make_ptr(
            cos_sin_dtype, placeholders[3], cute.AddressSpace.gmem, assumed_align=4
        ),
        make_ptr(
            cutlass.Float8E4M3FN,
            placeholders[4],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float8E8M0FNU,
            placeholders[5],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float8E8M0FNU,
            placeholders[6],
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.BFloat16, placeholders[7], cute.AddressSpace.gmem, assumed_align=16
        ),
        make_ptr(
            cutlass.Float32, placeholders[8], cute.AddressSpace.gmem, assumed_align=16
        ),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "gemm.dense_fused_quant_a_grouped", 2, compile_key
        ),
    )

    def tensor_api(
        source: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        b: torch.Tensor,
        sfb: torch.Tensor,
        out: torch.Tensor,
        alpha: torch.Tensor,
        stream_int: Optional[int],
    ) -> torch.Tensor:
        source_ptr = source.data_ptr()
        compiled(
            make_ptr(
                cutlass.Float8E4M3FN,
                source_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.BFloat16, source_ptr, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                positions_dtype,
                positions.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            make_ptr(
                cos_sin_dtype,
                cos_sin.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                cutlass.Float8E4M3FN,
                b.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float8E8M0FNU,
                source_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float8E8M0FNU,
                sfb.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.BFloat16,
                out.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float32,
                alpha.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            int(source.shape[0]),
            int(cos_sin.numel()),
            cuda_stream_from_int_or_current(stream_int),
        )
        return out

    return tensor_api


def dense_gemm_fused_quant_a_grouped(
    source: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    *,
    groups: int,
    out: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    cos_sin_cache: Optional[torch.Tensor] = None,
    head_dim: int = 0,
    nope_dim: int = 0,
    rope_dim: int = 0,
    expected_m: Optional[int] = None,
    sfb_k_replicated: bool = False,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    stream: object = None,
) -> torch.Tensor:
    """Small-M grouped BF16-A -> MXFP8 GEMM quantizing A in each CTA (WO-A).

    `source` is `[M, groups, K]` BF16 with contiguous trailing dims (rows may
    be strided); logical GEMM operands are per-group `[M, K] x [N, K]`. When
    `positions`/`cos_sin_cache` are given, the trailing `rope_dim` of every
    `head_dim` block is inverse-RoPE-rotated before quantization.
    """

    if source.dtype != torch.bfloat16 or source.ndim != 3:
        raise ValueError(
            "fused grouped MXFP8 quantization requires BF16 [M, groups, K]"
        )
    m = int(source.shape[0])
    k = int(source.shape[2])
    if int(source.shape[1]) != groups:
        raise ValueError(
            f"source groups {int(source.shape[1])} != weight groups {groups}"
        )
    if source.stride(2) != 1 or source.stride(1) != k:
        raise ValueError(
            f"fused grouped MXFP8 A needs contiguous [groups, K] rows, got strides {source.stride()}"
        )
    row_stride = int(source.stride(0))
    if m < 1 or m > 8 or k % 128 != 0 or row_stride % 8 != 0:
        raise ValueError(
            f"fused grouped MXFP8 quantization requires 1<=M<=8, K%128=0, row stride%8=0; "
            f"got M={m}, K={k}, row_stride={row_stride}"
        )
    if b.ndim != 3 or int(b.shape[1]) != k or int(b.shape[2]) != groups:
        raise ValueError(f"B must have shape [N,{k},{groups}], got {tuple(b.shape)}")
    n = int(b.shape[0])
    inv_rope = positions is not None or cos_sin_cache is not None
    if inv_rope:
        if positions is None or cos_sin_cache is None:
            raise ValueError("inverse-RoPE needs both positions and cos_sin_cache")
        if positions.shape != (m,):
            raise ValueError(
                f"positions must have shape {(m,)}, got {tuple(positions.shape)}"
            )
        if not positions.is_contiguous() or not cos_sin_cache.is_contiguous():
            raise ValueError("positions and cos_sin_cache must be contiguous")
        if cos_sin_cache.ndim != 2 or int(cos_sin_cache.shape[1]) != rope_dim:
            raise ValueError(
                f"cos_sin_cache must have shape [max_pos, {rope_dim}], got {tuple(cos_sin_cache.shape)}"
            )
        if (
            head_dim <= 0
            or nope_dim + rope_dim != head_dim
            or head_dim % 32
            or nope_dim % 32
            or rope_dim % 32
            or k % head_dim
        ):
            raise ValueError(
                "fused inverse-RoPE requires 32-aligned head_dim = nope_dim + rope_dim "
                f"dividing K, got head={head_dim}, nope={nope_dim}, rope={rope_dim}, K={k}"
            )
        positions_dtype = _cutlass_positions_dtype(positions.dtype)
        cos_sin_dtype = _cutlass_cos_sin_dtype(cos_sin_cache.dtype)
    else:
        head_dim = nope_dim = rope_dim = 0
        positions = source
        cos_sin_cache = source
        positions_dtype = cutlass.Int64
        cos_sin_dtype = cutlass.BFloat16
    sm_count = get_num_sm(source.device)
    if mma_tiler_mn is None:
        plan = _select_default_dense_gemm_plan(
            m, n, k, sm_count, is_mxfp8=True, expected_m=expected_m
        )
        if plan.swap_ab or plan.load_path != "tma":
            raise ValueError(
                "fused grouped MXFP8 quantization requires the unswapped TMA plan"
            )
        mma_tiler_mn = plan.mma_tiler_mn
    policy = _dense_gemm_policy_for(
        m=m,
        n=n,
        k=k,
        l=groups,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        sm_count=sm_count,
        expected_m=expected_m,
    )
    if policy.split_k_slices != 1:
        raise ValueError("fused grouped MXFP8 quantization does not support split-K")
    if out is None:
        out = _empty_dense_gemm_output(
            m, n, groups, dtype=torch.bfloat16, device=source.device
        )
    if out.shape != (m, n, groups) or out.dtype != torch.bfloat16:
        raise ValueError(
            f"out must be BF16 with shape {(m, n, groups)}, got {out.dtype} {tuple(out.shape)}"
        )
    compiled = _get_compiled_dense_gemm_fused_quant_a_grouped(
        n,
        k,
        groups,
        policy,
        mma_tiler_mn,
        sm_count,
        bool(sfb_k_replicated),
        row_stride,
        k,
        bool(inv_rope),
        int(head_dim),
        int(nope_dim),
        int(rope_dim),
        m == 1,
        positions_dtype,
        cos_sin_dtype,
    )
    return compiled(
        source,
        positions,
        cos_sin_cache,
        b,
        sfb,
        out,
        _cached_alpha_one(source.device),
        cuda_stream_to_int(stream),
    )


def _dense_gemm_target_occupancy(
    *,
    n: int,
    k: int,
    l: int,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    tile_k: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    sm_count: int,
    load_path: str,
    swap_ab: bool,
    b_tile_major: bool,
) -> int:
    tile_m, tile_n = mma_tiler_mn
    n_tiles = ((n + tile_n - 1) // tile_n) * l
    return (
        2
        if ab_dtype == cutlass.Float8E4M3FN
        and c_dtype == cutlass.BFloat16
        and tile_k == 128
        and tile_m == 16
        and k <= 1024
        and cluster_shape_mn == (1, 1)
        and load_path == "tma"
        and not swap_ab
        and not b_tile_major
        and n_tiles >= 2 * sm_count
        else 1
    )


@functools.cache
def _get_compiled_dense_gemm(
    n: int,
    k: int,
    l: int,
    c_l: int,
    a_major: str,
    b_major: str,
    c_major: str,
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    alpha_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    mma_k: int,
    tile_k: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    policy: _DenseGemmPolicy,
    sm_count: int,
    sm_version: str,
    load_path: str,
    swap_ab: bool,
    sfb_k_reuse: bool,
    b_tile_major: bool,
    quantize_c: bool = False,
    alpha_is_one: bool = False,
    direct_sfa_live16: bool = False,
    direct_m1_wo_a_inputs: bool = False,
) -> Callable:
    def _make_runtime_pointers(
        input_tensors: Optional[List[torch.Tensor]],
        quant_c_tensors: Optional[List[torch.Tensor]] = None,
    ) -> List[cute.Pointer]:
        if input_tensors is None:
            (
                a_data_ptr,
                b_data_ptr,
                sfa_data_ptr,
                sfb_data_ptr,
                c_data_ptr,
                alpha_data_ptr,
            ) = [16 for _ in range(6)]
        else:
            (
                a_tensor_gpu,
                b_tensor_gpu,
                sfa_tensor_gpu,
                sfb_tensor_gpu,
                c_tensor_gpu,
                alpha_tensor_gpu,
            ) = input_tensors
            (
                a_data_ptr,
                b_data_ptr,
                sfa_data_ptr,
                sfb_data_ptr,
                c_data_ptr,
                alpha_data_ptr,
            ) = (
                a_tensor_gpu.data_ptr(),
                b_tensor_gpu.data_ptr(),
                sfa_tensor_gpu.data_ptr(),
                sfb_tensor_gpu.data_ptr(),
                c_tensor_gpu.data_ptr(),
                alpha_tensor_gpu.data_ptr(),
            )
        if quant_c_tensors is None:
            quant_c_values_ptr = 16
            quant_c_scale_rows_ptr = 16
            quant_c_scale_mma_ptr = 16
        else:
            (
                quant_c_values_gpu,
                quant_c_scale_rows_gpu,
                quant_c_scale_mma_gpu,
            ) = quant_c_tensors
            quant_c_values_ptr = quant_c_values_gpu.data_ptr()
            quant_c_scale_rows_ptr = quant_c_scale_rows_gpu.data_ptr()
            quant_c_scale_mma_ptr = quant_c_scale_mma_gpu.data_ptr()

        return [
            make_ptr(ab_dtype, a_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(ab_dtype, b_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(sf_dtype, sfa_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(sf_dtype, sfb_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(c_dtype, c_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(
                cutlass.Float8E4M3FN,
                quant_c_values_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float8E8M0FNU,
                quant_c_scale_rows_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float8E8M0FNU,
                quant_c_scale_mma_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                alpha_dtype, alpha_data_ptr, cute.AddressSpace.gmem, assumed_align=16
            ),
        ]

    launch = _DenseGemmLaunch(
        n=n,
        k=k,
        l=l,
        c_l=c_l,
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        alpha_dtype=alpha_dtype,
        sf_vec_size=sf_vec_size,
        mma_k=mma_k,
        tile_k=tile_k,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        policy=policy,
        sm_count=sm_count,
        sm_version=sm_version,
        load_path=load_path,
        swap_ab=swap_ab,
        sfb_k_reuse=sfb_k_reuse,
        b_tile_major=b_tile_major,
        quantize_c=quantize_c,
        alpha_is_one=alpha_is_one,
        direct_sfa_live16=direct_sfa_live16,
        direct_m1_wo_a_inputs=direct_m1_wo_a_inputs,
        target_occupancy=_dense_gemm_target_occupancy(
            n=n,
            k=k,
            l=l,
            ab_dtype=ab_dtype,
            c_dtype=c_dtype,
            tile_k=tile_k,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sm_count=sm_count,
            load_path=load_path,
            swap_ab=swap_ab,
            b_tile_major=b_tile_major,
        ),
    )
    compile_key = launch.compile_key()
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=compile_key,
    )
    compiled_kernel = sm12x_compile(
        launch,
        *_make_runtime_pointers(None),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("gemm.dense", 3, compile_key),
    )

    def tensor_api(
        a_tensor_gpu: torch.Tensor,
        b_tensor_gpu: torch.Tensor,
        sfa_tensor_gpu: torch.Tensor,
        sfb_tensor_gpu: torch.Tensor,
        c_tensor_gpu: Optional[torch.Tensor] = None,
        alpha_tensor_gpu: Optional[torch.Tensor] = None,
        stream_int: Optional[int] = None,
        quant_c_values_gpu: Optional[torch.Tensor] = None,
        quant_c_scale_rows_gpu: Optional[torch.Tensor] = None,
        quant_c_scale_mma_gpu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m = a_tensor_gpu.shape[0]
        if c_tensor_gpu is None:
            c_tensor_gpu = torch.empty(
                (m, n, c_l),
                dtype=cutlass_to_torch_dtype(c_dtype),
                device=a_tensor_gpu.device,
            )
        if alpha_tensor_gpu is None:
            alpha_tensor_gpu = _cached_alpha_one(a_tensor_gpu.device)
        quant_c_tensors = None
        if quantize_c:
            if (
                quant_c_values_gpu is None
                or quant_c_scale_rows_gpu is None
                or quant_c_scale_mma_gpu is None
            ):
                raise ValueError("quantized C output tensors are required")
            quant_c_tensors = [
                quant_c_values_gpu,
                quant_c_scale_rows_gpu,
                quant_c_scale_mma_gpu,
            ]

        nonlocal compiled_kernel
        compiled_kernel(
            *_make_runtime_pointers(
                [
                    a_tensor_gpu,
                    b_tensor_gpu,
                    sfa_tensor_gpu,
                    sfb_tensor_gpu,
                    c_tensor_gpu,
                    alpha_tensor_gpu,
                ],
                quant_c_tensors,
            ),
            m,
            cuda_stream_from_int_or_current(stream_int),
        )
        return c_tensor_gpu

    return tensor_api


def _dense_gemm_launch_flat(
    a_tensor_gpu: torch.Tensor,
    b_tensor_gpu: torch.Tensor,
    sfa_tensor_gpu: torch.Tensor,
    sfb_tensor_gpu: torch.Tensor,
    c_tensor_gpu: torch.Tensor,
    alpha_tensor_gpu: torch.Tensor,
    n: int,
    k: int,
    l: int,
    c_l: int,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    alpha_dtype: str,
    sf_vec_size: int,
    mma_k: int,
    tile_k: int,
    mma_tile_m: int,
    mma_tile_n: int,
    cluster_shape_m: int,
    cluster_shape_n: int,
    sm_count: int,
    single_work_tile_per_cta: bool,
    direct_one_m_tile_scheduler: bool,
    use_m1_non_tma: bool,
    split_k_slices: int,
    split_k_atomic_bf16: bool,
    large_m_unroll: bool,
    load_path: str,
    swap_ab: bool,
    sfb_k_reuse: bool,
    alpha_is_one: bool,
    stream_int: Optional[int],
) -> None:
    b_tile_major = b_tensor_gpu.ndim == 5
    policy = _DenseGemmPolicy(
        single_work_tile_per_cta=single_work_tile_per_cta,
        direct_one_m_tile_scheduler=direct_one_m_tile_scheduler,
        use_m1_non_tma=use_m1_non_tma,
        split_k_slices=split_k_slices,
        split_k_atomic_bf16=split_k_atomic_bf16,
        large_m_unroll=large_m_unroll,
    )
    compiled = _get_compiled_dense_gemm(
        n=n,
        k=k,
        l=l,
        c_l=c_l,
        a_major="k",
        b_major="k",
        c_major="n",
        ab_dtype=get_cutlass_dtype(ab_dtype),
        sf_dtype=get_cutlass_dtype(sf_dtype),
        c_dtype=get_cutlass_dtype(c_dtype),
        alpha_dtype=get_cutlass_dtype(alpha_dtype),
        sf_vec_size=sf_vec_size,
        mma_k=mma_k,
        tile_k=tile_k,
        mma_tiler_mn=(mma_tile_m, mma_tile_n),
        cluster_shape_mn=(cluster_shape_m, cluster_shape_n),
        policy=policy,
        sm_count=sm_count,
        sm_version="sm_120",
        load_path=load_path,
        swap_ab=swap_ab,
        sfb_k_reuse=sfb_k_reuse,
        b_tile_major=b_tile_major,
        alpha_is_one=alpha_is_one,
        direct_sfa_live16=_use_direct_sfa_live16(
            m=int(a_tensor_gpu.shape[0]),
            n=n,
            k=k,
            l=l,
            sf_vec_size=sf_vec_size,
            tile_k=tile_k,
            mma_tiler_mn=(mma_tile_m, mma_tile_n),
            load_path=load_path,
            swap_ab=swap_ab,
            b_tile_major=b_tile_major,
            sfb_k_reuse=sfb_k_reuse,
            alpha_is_one=alpha_is_one,
            is_mxfp8=ab_dtype == "float8_e4m3fn",
        ),
        direct_m1_wo_a_inputs=_use_direct_m1_wo_a_inputs(
            m=int(a_tensor_gpu.shape[0]),
            n=n,
            k=k,
            l=l,
            sf_vec_size=sf_vec_size,
            tile_k=tile_k,
            mma_tiler_mn=(mma_tile_m, mma_tile_n),
            load_path=load_path,
            swap_ab=swap_ab,
            b_tile_major=b_tile_major,
            sfb_k_reuse=sfb_k_reuse,
            is_mxfp8=ab_dtype == "float8_e4m3fn",
        ),
    )
    compiled(
        a_tensor_gpu=a_tensor_gpu,
        b_tensor_gpu=b_tensor_gpu,
        sfa_tensor_gpu=sfa_tensor_gpu,
        sfb_tensor_gpu=sfb_tensor_gpu,
        c_tensor_gpu=c_tensor_gpu,
        alpha_tensor_gpu=alpha_tensor_gpu,
        stream_int=stream_int,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::dense_gemm_launch",
    mutates_args=("c_tensor_gpu",),
)
def _dense_gemm_launch_op(
    a_tensor_gpu: torch.Tensor,
    b_tensor_gpu: torch.Tensor,
    sfa_tensor_gpu: torch.Tensor,
    sfb_tensor_gpu: torch.Tensor,
    c_tensor_gpu: torch.Tensor,
    alpha_tensor_gpu: torch.Tensor,
    n: int,
    k: int,
    l: int,
    c_l: int,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    alpha_dtype: str,
    sf_vec_size: int,
    mma_k: int,
    tile_k: int,
    mma_tile_m: int,
    mma_tile_n: int,
    cluster_shape_m: int,
    cluster_shape_n: int,
    sm_count: int,
    single_work_tile_per_cta: bool,
    direct_one_m_tile_scheduler: bool,
    use_m1_non_tma: bool,
    split_k_slices: int,
    split_k_atomic_bf16: bool,
    large_m_unroll: bool,
    load_path: str,
    swap_ab: bool,
    sfb_k_reuse: bool,
    alpha_is_one: bool,
    stream_int: Optional[int],
) -> None:
    _dense_gemm_launch_flat(
        a_tensor_gpu,
        b_tensor_gpu,
        sfa_tensor_gpu,
        sfb_tensor_gpu,
        c_tensor_gpu,
        alpha_tensor_gpu,
        n,
        k,
        l,
        c_l,
        ab_dtype,
        sf_dtype,
        c_dtype,
        alpha_dtype,
        sf_vec_size,
        mma_k,
        tile_k,
        mma_tile_m,
        mma_tile_n,
        cluster_shape_m,
        cluster_shape_n,
        sm_count,
        single_work_tile_per_cta,
        direct_one_m_tile_scheduler,
        use_m1_non_tma,
        split_k_slices,
        split_k_atomic_bf16,
        large_m_unroll,
        load_path,
        swap_ab,
        sfb_k_reuse,
        alpha_is_one,
        stream_int,
    )


@_dense_gemm_launch_op.register_fake
def _dense_gemm_launch_fake(
    a_tensor_gpu: torch.Tensor,
    b_tensor_gpu: torch.Tensor,
    sfa_tensor_gpu: torch.Tensor,
    sfb_tensor_gpu: torch.Tensor,
    c_tensor_gpu: torch.Tensor,
    alpha_tensor_gpu: torch.Tensor,
    n: int,
    k: int,
    l: int,
    c_l: int,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    alpha_dtype: str,
    sf_vec_size: int,
    mma_k: int,
    tile_k: int,
    mma_tile_m: int,
    mma_tile_n: int,
    cluster_shape_m: int,
    cluster_shape_n: int,
    sm_count: int,
    single_work_tile_per_cta: bool,
    direct_one_m_tile_scheduler: bool,
    use_m1_non_tma: bool,
    split_k_slices: int,
    split_k_atomic_bf16: bool,
    large_m_unroll: bool,
    load_path: str,
    swap_ab: bool,
    sfb_k_reuse: bool,
    alpha_is_one: bool,
    stream_int: Optional[int],
) -> None:
    return None


_ALPHA_ONE_CACHE: dict = {}


def _cached_alpha_one(device: torch.device | str) -> torch.Tensor:
    # Per-device cached scalar-one alpha, to avoid a per-call torch.ones((1,))
    # host/device alloc on the generic FP8 dense-GEMM path. Mirrors
    # wo_projection._cached_alpha_one (not imported -- wo_projection imports
    # dense, so importing back would be circular).
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    key = (resolved.type, resolved.index)
    alpha = _ALPHA_ONE_CACHE.get(key)
    if alpha is None or alpha.device != resolved:
        alpha = torch.ones((1,), dtype=torch.float32, device=resolved)
        _ALPHA_ONE_CACHE[key] = alpha
    return alpha


def _empty_dense_gemm_output(
    m: int,
    n: int,
    l: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Allocate an `[M,N,L]` dense-GEMM output in the layout the kernel writes.

    The CuTe dense GEMM hardcodes ``c_major='n'`` and builds the C tensor from
    the data pointer with order ``(1,0,2)`` -- i.e. it writes the grouped output
    as physical ``[L,M,N]`` (an ``[M,N,L]`` view with strides ``(N,1,M*N)``) and
    ignores the runtime tensor's actual strides. A plain contiguous ``(M,N,L)``
    buffer (strides ``(N*L,L,1)``) would scatter the ``L`` groups to the wrong
    offsets, so back ``L>1`` with ``[L,M,N]`` physical storage. ``L==1`` is the
    same either way. Mirrors ``empty_dense_gemm_mnl_view`` in wo_projection.
    """
    if l > 1:
        return torch.empty((l, m, n), dtype=dtype, device=device).as_strided(
            (m, n, l), (n, 1, m * n)
        )
    return torch.empty((m, n, l), dtype=dtype, device=device)


@torch.library.custom_op(
    "flashinfer_sm12x::dense_gemm_launch_functional",
    mutates_args=(),
)
def _dense_gemm_launch_functional_op(
    a_tensor_gpu: torch.Tensor,
    b_tensor_gpu: torch.Tensor,
    sfa_tensor_gpu: torch.Tensor,
    sfb_tensor_gpu: torch.Tensor,
    alpha_tensor_gpu: torch.Tensor,
    n: int,
    k: int,
    l: int,
    kernel_c_l: int,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    kernel_c_dtype: str,
    alpha_dtype: str,
    sf_vec_size: int,
    mma_k: int,
    tile_k: int,
    mma_tile_m: int,
    mma_tile_n: int,
    cluster_shape_m: int,
    cluster_shape_n: int,
    sm_count: int,
    single_work_tile_per_cta: bool,
    direct_one_m_tile_scheduler: bool,
    use_m1_non_tma: bool,
    split_k_slices: int,
    split_k_atomic_bf16: bool,
    large_m_unroll: bool,
    load_path: str,
    swap_ab: bool,
    sfb_k_reuse: bool,
    alpha_is_one: bool,
    stream_int: Optional[int],
) -> torch.Tensor:
    m = int(a_tensor_gpu.shape[0])
    out = _empty_dense_gemm_output(
        m,
        n,
        l,
        dtype=cutlass_to_torch_dtype(get_cutlass_dtype(c_dtype)),
        device=a_tensor_gpu.device,
    )
    split_k_output = int(split_k_slices) > 1
    if split_k_output and split_k_atomic_bf16:
        c_tensor_gpu = out
        out.zero_()
    elif split_k_output:
        split_storage = torch.empty(
            (split_k_slices, m, n),
            dtype=torch.float32,
            device=a_tensor_gpu.device,
        )
        c_tensor_gpu = split_storage.permute(1, 2, 0)
    else:
        c_tensor_gpu = out

    _dense_gemm_launch_flat(
        a_tensor_gpu,
        b_tensor_gpu,
        sfa_tensor_gpu,
        sfb_tensor_gpu,
        c_tensor_gpu,
        alpha_tensor_gpu,
        n,
        k,
        l,
        kernel_c_l,
        ab_dtype,
        sf_dtype,
        kernel_c_dtype,
        alpha_dtype,
        sf_vec_size,
        mma_k,
        tile_k,
        mma_tile_m,
        mma_tile_n,
        cluster_shape_m,
        cluster_shape_n,
        sm_count,
        single_work_tile_per_cta,
        direct_one_m_tile_scheduler,
        use_m1_non_tma,
        split_k_slices,
        split_k_atomic_bf16,
        large_m_unroll,
        load_path,
        swap_ab,
        sfb_k_reuse,
        alpha_is_one,
        stream_int,
    )
    if split_k_output and not split_k_atomic_bf16:
        _reduce_split_k2_bf16(c_tensor_gpu, out, m=m, n=n)
    return out


@_dense_gemm_launch_functional_op.register_fake
def _dense_gemm_launch_functional_fake(
    a_tensor_gpu: torch.Tensor,
    b_tensor_gpu: torch.Tensor,
    sfa_tensor_gpu: torch.Tensor,
    sfb_tensor_gpu: torch.Tensor,
    alpha_tensor_gpu: torch.Tensor,
    n: int,
    k: int,
    l: int,
    kernel_c_l: int,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    kernel_c_dtype: str,
    alpha_dtype: str,
    sf_vec_size: int,
    mma_k: int,
    tile_k: int,
    mma_tile_m: int,
    mma_tile_n: int,
    cluster_shape_m: int,
    cluster_shape_n: int,
    sm_count: int,
    single_work_tile_per_cta: bool,
    direct_one_m_tile_scheduler: bool,
    use_m1_non_tma: bool,
    split_k_slices: int,
    split_k_atomic_bf16: bool,
    large_m_unroll: bool,
    load_path: str,
    swap_ab: bool,
    sfb_k_reuse: bool,
    alpha_is_one: bool,
    stream_int: Optional[int],
) -> torch.Tensor:
    del (
        b_tensor_gpu,
        sfa_tensor_gpu,
        sfb_tensor_gpu,
        alpha_tensor_gpu,
        k,
        kernel_c_l,
        ab_dtype,
        sf_dtype,
        kernel_c_dtype,
        alpha_dtype,
        sf_vec_size,
        mma_k,
        tile_k,
        mma_tile_m,
        mma_tile_n,
        cluster_shape_m,
        cluster_shape_n,
        sm_count,
        single_work_tile_per_cta,
        direct_one_m_tile_scheduler,
        use_m1_non_tma,
        split_k_slices,
        split_k_atomic_bf16,
        large_m_unroll,
        load_path,
        swap_ab,
        sfb_k_reuse,
        alpha_is_one,
        stream_int,
    )
    return _empty_dense_gemm_output(
        int(a_tensor_gpu.shape[0]),
        n,
        l,
        dtype=cutlass_to_torch_dtype(get_cutlass_dtype(c_dtype)),
        device=a_tensor_gpu.device,
    )


def _select_default_mma_tiler_mn(
    m: int,
    n: int,
    sm_count: int,
    *,
    is_mxfp8: bool,
    expected_m: Optional[int] = None,
    k: Optional[int] = None,
) -> Tuple[int, int]:
    coarse_tile = (128, 128)
    # The serving WO-B prefill GEMM is [M,4096] x [4096,4096]. DeepGEMM's
    # specialized O-projection dispatch switches it from BM64/BK128 to
    # BM128/BK64 at M>=2048, and the same exact-shape switch wins in sm12x.
    # WO-A is deliberately excluded: its four grouped [M,512]x[1024,512]
    # GEMMs remain faster with BM64/BK128 on this kernel.
    if (
        is_mxfp8
        and expected_m is not None
        and expected_m >= 2048
        and k is not None
        and (n, k) == (4096, 4096)
    ):
        return (128, 128)
    if is_mxfp8 and n > 1536:
        # DeepGEMM-style regime hint. When a caller declares expected_m, pick the
        # per-regime optimal tile and key the compile on it: ONE kernel per
        # (N,K,expected_m), reused for every live M in that regime under frozen
        # resolution (M-independent within the regime). Probe optima
        # (benchmarks/probe_dense_fp8_tile_sweep.py): exact M=1 -> 16x64
        # (flushed common-shape decode sweep); expected_m=2..8 -> 16x128.
        # The DSV4 TP2 q_b shape keeps that tile through M=16; its 16-row tile
        # sustains two resident CTAs per SM, while 32x128 drops to one and loses
        # the evict-first short-K load policy. Other wide-N shapes retain the
        # existing 32x128 small-batch regime.
        # <=128 (small batch) -> 32x128 (~25% faster than 64x128 at M=32..128);
        # else -> 64x128 (the M-independent default, good to prefill).
        if expected_m is not None:
            # BM64/BK128 is the 20-SM Spark q_b winner. The 188-SM RTX audit
            # keeps the following BM128/BK64 specialization instead.
            if (
                expected_m >= 2048
                and (n, k) == (16384, 1024)
                and _dense_spark_policy_for_sm_count(sm_count)
            ):
                return (64, 128)
            if expected_m >= 2048 and n >= 16384 and k is not None and k <= 1024:
                return (128, 128)
            if expected_m == 1:
                return (16, 64)
            # 48-SM Spark q_b decode: (16,128) yields 128 N-tiles over the 96
            # resident CTAs (occupancy 2), so the remainder wave streams B with
            # only 32 CTAs and drops below the sustained-read ceiling. (16,64)
            # quantizes the tail at 64 columns with 2/3 of CTAs still active,
            # matching the M=1 (16,64) profile that already runs at ceiling.
            # RTX keeps the probe-swept (16,128).
            if (
                expected_m <= 16
                and (n, k) == (16384, 1024)
                and _dense_spark_policy_for_sm_count(sm_count)
            ):
                return (16, 64)
            if expected_m <= 8 or (expected_m <= 16 and (n, k) == (16384, 1024)):
                return (16, 128)
            if expected_m <= 128:
                return (32, 128)
            return (64, 128)
        # No regime hint: keep the true single-token decode specialization and
        # use the decode-tuned tile for tiny standalone graph shapes. Broader
        # live-M reuse still falls back to the M-independent prefill-safe tile.
        if m == 1:
            return (16, 64)
        if m <= 8:
            return (16, 128)
        # Wide-N MXFP8: the 128x128 pin spans only ceil(N/128) column tiles, so
        # at small/medium M it launches ~32-64 CTAs and runs flat (~80us, B-BW
        # starved). It is in fact the WORST tile at every M (geomean ~121us over
        # M=2..4096; see benchmarks/probe_dense_fp8_tile_sweep.py). 64x128 is the
        # best M-INDEPENDENT tile: it beats 128x128 at every M (1.1x-2.4x; geomean
        # ~69us) with byte-identical output. M-independence is required because
        # dense serving warms one kernel per (N,K) and reuses it for all live M
        # under frozen resolution (see test_block_fp8_linear_small_live_m_reuses_
        # prefill_dense_kernel) -- an M-dependent tile forces an illegal recompile
        # mid-serve. (Smaller 32x128/16x128 are faster at M<=128 but regress
        # prefill M>=2k and would break that one-kernel-per-(N,K) reuse contract.)
        return (64, 128)

    if is_mxfp8:
        # Narrow-N MXFP8 (n <= 1536; the n > 1536 case returned above). The
        # (128,128) coarse tile spans only ceil(N/128) column tiles (<=12 at
        # N<=1536), so at M>=512 it launches ~32-48 CTAs on a 188-SM part and
        # runs CTA-starved -- 2x-3.5x slower than a CTA-multiplying tile
        # (probe_dense_fp8_tile_sweep.py: N=1024 M=512 (128,128)=63.5us vs
        # (64,64)=18.4us; N=1536 M=512 (128,128)=65.5us vs (64,128)=24.6us).
        # Mirror the wide-N expected_m design where we have data. Exact M=1
        # gets the flushed common-shape decode winner (16,64). Declared prefill
        # (expected_m>128) -> (64,128): the best narrow-N tile at M>=512 for
        # both N=1024 and N=1536 across M=512..8192 (probe sweep), recovering
        # both the M~512 cliff and the large-M tail (N=1024 M=4096:
        # (64,128)=80us vs (64,64)=105us vs (128,128)=125us). Other
        # decode/small and the no-hint default use the M-independent (64,64)
        # (max CTAs; best at M<=512), preserving the one-kernel-per-(N,K) reuse
        # contract.
        if expected_m == 1 or (expected_m is None and m == 1):
            return (16, 64)
        if expected_m is not None and expected_m > 128:
            return (64, 128)
        return (64, 64)

    plan_m = expected_m if expected_m is not None else m
    if plan_m == 1 and k is not None:
        # Flushed M=1 FP4 probe (benchmarks/probe_dense_fp4_tile_load_sweep.py)
        # across the repo's common shapes:
        #   * wide/medium N: (64,128)/TMA has the best geomean and wins nearly all
        #     shapes.
        #   * N=1024,K=5376: (64,64)/TMA wins the boundary by a small margin.
        #   * N<=512 with long K: (64,32)/TMA+swap_ab is the only clear tiny-N win.
        # Keep the tile selector tile-only; the launch planner below attaches
        # swap_ab to the narrow tile.
        if n <= 512 and k >= 4096:
            return (64, 32)
        if n <= 1024:
            return (64, 64)
        return (64, 128)

    coarse_tiles = ((m + coarse_tile[0] - 1) // coarse_tile[0]) * (
        (n + coarse_tile[1] - 1) // coarse_tile[1]
    )
    # The coarse CTA-count heuristic misses exact-small-M, wide-N cases: a wide
    # N dimension can generate plenty of CTAs even while each 128-row M tile is
    # mostly empty. Keep using the narrower 64x128 tile while the 128x128 plan
    # still leaves the GPU below the existing half-SM occupancy proxy.
    if n > 1536:
        if m <= 64:
            return (64, 128)
        if m <= 256 and coarse_tiles < max(1, sm_count // 2):
            return (64, 128)
    if m <= 128 and coarse_tiles < max(1, sm_count // 2):
        if n > 1536:
            return (64, 128)
        medium_tile = (128, 64)
        medium_tiles = ((m + medium_tile[0] - 1) // medium_tile[0]) * (
            (n + medium_tile[1] - 1) // medium_tile[1]
        )
        if medium_tiles < max(1, sm_count // 2):
            return (64, 64)
        return (128, 64)
    return coarse_tile


def _select_mxfp8_tile_k(
    m: int,
    n: int,
    k: int,
    expected_m: Optional[int],
    sm_count: int,
) -> int:
    # Keep tile-M and tile-K as one hardware-specific q_b plan: Spark uses
    # BM64/BK128, while the RTX specialization below remains BM128/BK64.
    if (
        expected_m is not None
        and expected_m >= 2048
        and (n, k) == (16384, 1024)
        and _dense_spark_policy_for_sm_count(sm_count)
    ):
        return 128
    # BK64 is an explicitly hinted production specialization. Choosing it from
    # live M when expected_m is absent would change both tile K and generated
    # code at M=2048, violating the no-hint frozen-resolution reuse contract.
    hinted_bk64 = (
        expected_m is not None
        and expected_m >= 2048
        and ((n >= 16384 and k <= 1024) or (n, k) == (4096, 4096))
    )
    return 64 if hinted_bk64 else 128


def _validate_mxfp8_bk64_plan(
    tile_k: int,
    mma_tiler_mn: Tuple[int, int],
    swap_ab: bool,
) -> None:
    if tile_k == 64 and (mma_tiler_mn[0] != 128 or swap_ab):
        raise ValueError(
            "MXFP8 BK64 packed-scale staging requires an unswapped "
            f"128-row tile, got tile={mma_tiler_mn}, swap_ab={swap_ab}"
        )


def _select_default_dense_gemm_plan(
    m: int,
    n: int,
    k: int,
    sm_count: int,
    *,
    is_mxfp8: bool,
    expected_m: Optional[int] = None,
) -> _DenseGemmPlan:
    tile = _select_default_mma_tiler_mn(
        m,
        n,
        sm_count,
        is_mxfp8=is_mxfp8,
        expected_m=expected_m,
        k=k,
    )
    return _DenseGemmPlan(
        mma_tiler_mn=tile,
        load_path="tma",
        swap_ab=(not is_mxfp8 and tile[1] < 64),
    )


def dense_gemm_fused_quant_a(
    source: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    expected_m: Optional[int] = None,
    sfb_k_replicated: bool = False,
    rhs_values_tiled: Optional[torch.Tensor] = None,
    a_inner_span: int = 0,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    _atomic_output_precleared: bool = False,
    stream: object = None,
) -> torch.Tensor:
    """Small-M BF16-A -> MXFP8 GEMM with activation quantization in each CTA.

    a_inner_span > 0 reads A from an L-blocked source instead of contiguous
    rows: `source` is the `[M, span, K/span]` dense-GEMM mnl view over physical
    `[K/span, M, span]` storage (the WO tmp group-major layout). Follows the
    dense_gemm split-K policy (FP32 partials + fused reduce) instead of forcing
    a single un-split kernel, which loses ~2x at M=1 for N,K >= 4096.
    """

    a_inner_span = int(a_inner_span)
    if source.dtype != torch.bfloat16:
        raise ValueError("fused MXFP8 activation quantization requires BF16 A")
    if a_inner_span == 0:
        if source.ndim != 2 or not source.is_contiguous():
            raise ValueError(
                "fused MXFP8 activation quantization requires contiguous BF16 [M,K]"
            )
        m, k = map(int, source.shape)
    else:
        if a_inner_span % 32 != 0:
            raise ValueError(
                f"fused MXFP8 a_inner_span must be a multiple of 32, got {a_inner_span}"
            )
        if source.ndim != 3 or int(source.shape[1]) != a_inner_span:
            raise ValueError(
                "L-blocked fused MXFP8 A requires an [M, span, K/span] view, "
                f"got {tuple(source.shape)} for span={a_inner_span}"
            )
        m = int(source.shape[0])
        k = a_inner_span * int(source.shape[2])
        if source.stride() != (a_inner_span, 1, m * a_inner_span):
            raise ValueError(
                "L-blocked fused MXFP8 A must be a dense-GEMM mnl view over "
                f"physical [K/span, M, span] storage, got strides {source.stride()}"
            )
    if m < 1 or m > 8 or k % 128 != 0:
        raise ValueError(
            f"fused MXFP8 activation quantization requires 1<=M<=8 and K%128=0, got M={m}, K={k}"
        )
    if b.ndim != 3 or int(b.shape[1]) != k or int(b.shape[2]) != 1:
        raise ValueError(f"B must have shape [N,{k},1], got {tuple(b.shape)}")
    n = int(b.shape[0])
    sm_count = get_num_sm(source.device)
    if mma_tiler_mn is None:
        plan = _select_default_dense_gemm_plan(
            m, n, k, sm_count, is_mxfp8=True, expected_m=expected_m
        )
        if plan.swap_ab or plan.load_path != "tma":
            raise ValueError(
                "fused MXFP8 activation quantization requires the unswapped TMA plan"
            )
        mma_tiler_mn = plan.mma_tiler_mn
    b_launch = b
    if rhs_values_tiled is not None:
        expected_tiled_shape = (1, 32, 32, 128, 128)
        if (n, k) != (4096, 4096) or mma_tiler_mn not in (
            (16, 64),
            (16, 128),
        ):
            raise ValueError(
                "tile-major fused-quant RHS is restricted to the production "
                "WO-B 16xN/BK128 plans"
            )
        if (
            rhs_values_tiled.shape != expected_tiled_shape
            or rhs_values_tiled.dtype != b.dtype
            or rhs_values_tiled.device != b.device
            or not rhs_values_tiled.is_contiguous()
        ):
            raise ValueError(
                "tile-major fused-quant RHS must be contiguous with shape "
                f"{expected_tiled_shape}, dtype {b.dtype}, and device {b.device}; "
                f"got shape={tuple(rhs_values_tiled.shape)}, "
                f"dtype={rhs_values_tiled.dtype}, device={rhs_values_tiled.device}"
            )
        b_launch = rhs_values_tiled
    policy = _dense_gemm_policy_for(
        m=m,
        n=n,
        k=k,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        sm_count=sm_count,
        expected_m=expected_m,
    )
    split_k_slices = policy.split_k_slices
    split_k_output = split_k_slices > 1
    split_k_atomic_bf16 = split_k_output and policy.split_k_atomic_bf16
    if out is None:
        if _atomic_output_precleared:
            raise ValueError("a precleared fused-quant output must be caller-owned")
        out = torch.empty((m, n, 1), dtype=torch.bfloat16, device=source.device)
    if out.shape != (m, n, 1) or out.dtype != torch.bfloat16:
        raise ValueError(
            f"out must be BF16 with shape {(m, n, 1)}, got {out.dtype} {tuple(out.shape)}"
        )
    split_storage = None
    if split_k_atomic_bf16:
        if not _atomic_output_precleared:
            out.zero_()
        kernel_c_l = 1
        kernel_c_dtype = cutlass.BFloat16
        c_tensor_gpu = out
    elif split_k_output:
        split_storage = torch.empty(
            (split_k_slices, m, n), dtype=torch.float32, device=source.device
        )
        kernel_c_l = split_k_slices
        kernel_c_dtype = cutlass.Float32
        c_tensor_gpu = split_storage
    else:
        kernel_c_l = 1
        kernel_c_dtype = cutlass.BFloat16
        c_tensor_gpu = out
    compiled = _get_compiled_dense_gemm_fused_quant_a(
        n,
        k,
        kernel_c_dtype,
        policy,
        mma_tiler_mn,
        sm_count,
        bool(sfb_k_replicated),
        rhs_values_tiled is not None,
        a_inner_span,
        kernel_c_l,
        m == 1,
    )
    compiled(
        source,
        b_launch,
        sfb,
        c_tensor_gpu,
        _cached_alpha_one(source.device),
        cuda_stream_to_int(stream),
    )
    if split_storage is not None:
        _reduce_split_k2_bf16(split_storage.permute(1, 2, 0), out, m=m, n=n)
    return out


def dense_gemm(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: Optional[torch.Tensor] = None,
    *,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    sm_count: Optional[int] = None,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    alpha: Optional[torch.Tensor] = None,
    alpha_dtype: Optional[str] = None,
    expected_m: Optional[int] = None,
    load_path: Optional[Literal["tma", "cpasync"]] = None,
    swap_ab: Optional[bool] = None,
    sfb_k_replicated: bool = False,
    rhs_values_tiled: Optional[torch.Tensor] = None,
    _quantized_c: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    stream: object = None,
) -> torch.Tensor:
    """Execute dense block-scaled GEMM for one expert-major batch stack.

    expected_m: optional regime hint (DeepGEMM-style). When set, the default tile
    is chosen for that representative M instead of being M-independent, giving a
    per-regime-optimal kernel that is still reused across all live M in the regime
    (e.g. expected_m<=128 selects a decode-tuned tile). Ignored when mma_tiler_mn
    is given. Live M stays a runtime arg; only the tile (a compile key) changes.
    """
    a_torch, sfa_torch = lhs
    b_torch, sfb_torch = rhs
    if load_path is not None and load_path not in _DENSE_LOAD_PATHS:
        raise ValueError(
            f"dense_gemm load_path must be one of {_DENSE_LOAD_PATHS}, got {load_path!r}"
        )

    m, k, l = a_torch.shape
    n, _, _ = b_torch.shape
    if sm_count is None:
        sm_count = get_num_sm(a_torch.device)
    if ab_dtype == "float4_e2m1fn":
        is_mxfp8 = False
        k *= 2
        mma_k = 64
        tile_k = sf_vec_size * 8
    elif ab_dtype == "float8_e4m3fn":
        is_mxfp8 = True
        mma_k = 32
        tile_k = _select_mxfp8_tile_k(m, n, k, expected_m, sm_count)
    else:
        raise TypeError(f"dense_gemm unsupported ab_dtype: {ab_dtype}")

    ab_cutlass_dtype = get_cutlass_dtype(ab_dtype)
    c_cutlass_dtype = get_cutlass_dtype(c_dtype)
    if mma_tiler_mn is None or load_path is None or swap_ab is None:
        default_plan = _select_default_dense_gemm_plan(
            m,
            n,
            k,
            sm_count,
            is_mxfp8=is_mxfp8,
            expected_m=expected_m,
        )
        if mma_tiler_mn is None:
            mma_tiler_mn = default_plan.mma_tiler_mn
        if load_path is None:
            load_path = default_plan.load_path
        if swap_ab is None:
            swap_ab = default_plan.swap_ab if mma_tiler_mn[1] < 64 else False
    assert load_path is not None
    assert swap_ab is not None
    b_launch_torch = b_torch
    if rhs_values_tiled is not None:
        tile_n = 0
        supported_plan = False
        if (n, k, l) == (1024, 4096, 4):
            tile_n = 64
            supported_plan = mma_tiler_mn in ((16, 64), (32, 64), (64, 64))
        elif (n, k, l) == (4096, 4096, 1):
            tile_n = 128
            supported_plan = mma_tiler_mn in (
                (16, 128),
                (32, 64),
                (32, 128),
            )
        if not is_mxfp8 or not supported_plan or swap_ab or load_path != "tma":
            raise ValueError(
                "tile-major MXFP8 RHS is restricted to production WO-A/WO-B "
                "Ntile/BK128 TMA plans"
            )
        expected_tiled_shape = (l, n // tile_n, k // 128, tile_n, 128)
        if (
            rhs_values_tiled.shape != expected_tiled_shape
            or rhs_values_tiled.dtype != b_torch.dtype
            or rhs_values_tiled.device != b_torch.device
            or not rhs_values_tiled.is_contiguous()
        ):
            raise ValueError(
                "tile-major MXFP8 RHS must be contiguous with shape "
                f"{expected_tiled_shape}, dtype {b_torch.dtype}, and device "
                f"{b_torch.device}; got shape={tuple(rhs_values_tiled.shape)}, "
                f"dtype={rhs_values_tiled.dtype}, device={rhs_values_tiled.device}"
            )
        b_launch_torch = rhs_values_tiled
    if is_mxfp8:
        _validate_mxfp8_bk64_plan(tile_k, mma_tiler_mn, swap_ab)
    # k-reuse relies on SFB being the 128x128-block weight operand; with
    # swap_ab the smem B slot holds activations, so force it off there.
    sfb_k_reuse = bool(sfb_k_replicated) and not swap_ab and is_mxfp8
    if alpha_dtype is None:
        alpha_dtype = "float32" if alpha is None else str(alpha.dtype).split(".")[-1]
    policy = _dense_gemm_policy_for(
        m=m,
        n=n,
        k=k,
        l=l,
        ab_dtype=ab_cutlass_dtype,
        c_dtype=c_cutlass_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sm_count=sm_count,
        expected_m=expected_m,
    )
    split_k_slices = policy.split_k_slices
    if swap_ab and split_k_slices != 1:
        policy = _DenseGemmPolicy(
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            use_m1_non_tma=policy.use_m1_non_tma,
            split_k_slices=1,
            split_k_atomic_bf16=False,
            large_m_unroll=policy.large_m_unroll,
        )
        split_k_slices = 1
    split_k_output = split_k_slices > 1
    split_k_atomic_bf16 = split_k_output and policy.split_k_atomic_bf16
    if split_k_atomic_bf16:
        kernel_c_l = l
    elif split_k_output:
        kernel_c_l = split_k_slices
    else:
        kernel_c_l = l
    alpha_is_one = alpha is None
    if alpha is None:
        alpha = _cached_alpha_one(a_torch.device)
    stream_int = cuda_stream_to_int(stream)
    kernel_c_dtype_name = (
        "float32" if split_k_output and not split_k_atomic_bf16 else c_dtype
    )
    if _quantized_c is not None:
        quant_c_values, quant_c_scale_rows, quant_c_scale_mma = _quantized_c
        quant_c_width = n * l
        expected_scale_mma_shape = (
            32,
            4,
            1,
            4,
            quant_c_width // 128,
            1,
        )
        if (
            split_k_output
            or swap_ab
            or c_dtype != "bfloat16"
            or m < 1
            or m > 16
            or n % 32 != 0
            or quant_c_width % 128 != 0
            or mma_tiler_mn[1] != 64
            or quant_c_values.shape != (m, quant_c_width)
            or not quant_c_values.is_contiguous()
            or quant_c_values.dtype != torch.float8_e4m3fn
            or quant_c_scale_rows.shape != (m, quant_c_width // 32)
            or not quant_c_scale_rows.is_contiguous()
            or quant_c_scale_rows.dtype != torch.float8_e8m0fnu
            or quant_c_scale_mma.shape != expected_scale_mma_shape
            or quant_c_scale_mma.dtype != torch.float8_e8m0fnu
        ):
            raise ValueError("quantized C is restricted to the BF16 WO-A decode layout")
        if out is None:
            out = _empty_dense_gemm_output(
                m,
                n,
                l,
                dtype=torch.bfloat16,
                device=a_torch.device,
            )
        compiled_quant_c = _get_compiled_dense_gemm(
            n=n,
            k=k,
            l=l,
            c_l=kernel_c_l,
            a_major="k",
            b_major="k",
            c_major="n",
            ab_dtype=ab_cutlass_dtype,
            sf_dtype=get_cutlass_dtype(sf_dtype),
            c_dtype=c_cutlass_dtype,
            alpha_dtype=get_cutlass_dtype(alpha_dtype),
            sf_vec_size=sf_vec_size,
            mma_k=mma_k,
            tile_k=tile_k,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            policy=policy,
            sm_count=sm_count,
            sm_version="sm_120",
            load_path=load_path,
            swap_ab=swap_ab,
            sfb_k_reuse=sfb_k_reuse,
            b_tile_major=rhs_values_tiled is not None,
            quantize_c=True,
            alpha_is_one=alpha_is_one,
            direct_sfa_live16=_use_direct_sfa_live16(
                m=m,
                n=n,
                k=k,
                l=l,
                sf_vec_size=sf_vec_size,
                tile_k=tile_k,
                mma_tiler_mn=mma_tiler_mn,
                load_path=load_path,
                swap_ab=swap_ab,
                b_tile_major=rhs_values_tiled is not None,
                sfb_k_reuse=sfb_k_reuse,
                alpha_is_one=alpha_is_one,
                is_mxfp8=is_mxfp8,
            ),
        )
        return compiled_quant_c(
            a_tensor_gpu=a_torch,
            b_tensor_gpu=b_launch_torch,
            sfa_tensor_gpu=sfa_torch,
            sfb_tensor_gpu=sfb_torch,
            c_tensor_gpu=out,
            alpha_tensor_gpu=alpha,
            stream_int=stream_int,
            quant_c_values_gpu=quant_c_values,
            quant_c_scale_rows_gpu=quant_c_scale_rows,
            quant_c_scale_mma_gpu=quant_c_scale_mma,
        )
    if out is None:
        # No caller-owned output buffer: functional launch (allocate + return
        # inside the opaque op). The compile graph then carries no
        # auto_functionalized dense node mutating a (possibly strided) caller
        # view -- which inductor's decompose pass cannot remove. No is_compiling;
        # purely caller-intent, behaviorally identical to the eager out=None path.
        return torch.ops.flashinfer_sm12x.dense_gemm_launch_functional(
            a_torch,
            b_launch_torch,
            sfa_torch,
            sfb_torch,
            alpha,
            n,
            k,
            l,
            kernel_c_l,
            ab_dtype,
            sf_dtype,
            c_dtype,
            kernel_c_dtype_name,
            alpha_dtype,
            sf_vec_size,
            mma_k,
            tile_k,
            mma_tiler_mn[0],
            mma_tiler_mn[1],
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            sm_count,
            policy.single_work_tile_per_cta,
            policy.direct_one_m_tile_scheduler,
            policy.use_m1_non_tma,
            policy.split_k_slices,
            policy.split_k_atomic_bf16,
            policy.large_m_unroll,
            load_path,
            swap_ab,
            sfb_k_reuse,
            alpha_is_one,
            stream_int,
        )
    split_storage = None
    split_scratch = None
    if split_k_output:
        if out is None:
            out = torch.empty(
                (m, n, l),
                dtype=cutlass_to_torch_dtype(c_cutlass_dtype),
                device=a_torch.device,
            )
        if split_k_atomic_bf16:
            out.zero_()
        else:
            split_storage = torch.empty(
                (split_k_slices, m, n),
                dtype=torch.float32,
                device=a_torch.device,
            )
            split_scratch = split_storage.permute(1, 2, 0)
    elif out is None:
        out = torch.empty(
            (m, n, l),
            dtype=cutlass_to_torch_dtype(c_cutlass_dtype),
            device=a_torch.device,
        )
    if alpha is None:
        alpha = _cached_alpha_one(a_torch.device)

    t0 = time.perf_counter() if _FLASHINFER_EXP_SM12X_TIMING else 0.0
    cache_before = (
        _get_compiled_dense_gemm.cache_info() if _FLASHINFER_EXP_SM12X_TIMING else None
    )
    t_compiled = t0
    kernel_c_dtype_name = (
        "float32" if split_k_output and not split_k_atomic_bf16 else c_dtype
    )
    c_tensor_gpu = (
        out if split_k_atomic_bf16 else split_scratch if split_k_output else out
    )
    assert c_tensor_gpu is not None
    torch.ops.flashinfer_sm12x.dense_gemm_launch(
        a_torch,
        b_launch_torch,
        sfa_torch,
        sfb_torch,
        c_tensor_gpu,
        alpha,
        n,
        k,
        l,
        kernel_c_l,
        ab_dtype,
        sf_dtype,
        kernel_c_dtype_name,
        alpha_dtype,
        sf_vec_size,
        mma_k,
        tile_k,
        mma_tiler_mn[0],
        mma_tiler_mn[1],
        cluster_shape_mn[0],
        cluster_shape_mn[1],
        sm_count,
        policy.single_work_tile_per_cta,
        policy.direct_one_m_tile_scheduler,
        policy.use_m1_non_tma,
        policy.split_k_slices,
        policy.split_k_atomic_bf16,
        policy.large_m_unroll,
        load_path,
        swap_ab,
        sfb_k_reuse,
        alpha_is_one,
        stream_int,
    )
    result = out
    if split_k_output and not split_k_atomic_bf16:
        assert split_scratch is not None
        assert out is not None
        _reduce_split_k2_bf16(split_scratch, out, m=m, n=n)
        result = out
    if _FLASHINFER_EXP_SM12X_TIMING:
        t_launch = time.perf_counter()
        cache_after = _get_compiled_dense_gemm.cache_info()
        assert cache_before is not None
        compile_ms = (t_compiled - t0) * 1000.0
        launch_ms = (t_launch - t_compiled) * 1000.0
        total_ms = (t_launch - t0) * 1000.0
        if total_ms >= _FLASHINFER_EXP_SM12X_TIMING_THRESHOLD_MS:
            logger.warning(
                "sm12x_dense_gemm timing m=%d n=%d k=%d l=%d ab=%s sf=%s c=%s "
                "tile=%s load=%s swap_ab=%s cache_hit=%s compile_or_lookup=%.3fms "
                "launch_enqueue=%.3fms total=%.3fms cache=%s",
                m,
                n,
                k,
                l,
                ab_dtype,
                sf_dtype,
                c_dtype,
                mma_tiler_mn,
                load_path,
                swap_ab,
                cache_after.hits > cache_before.hits,
                compile_ms,
                launch_ms,
                total_ms,
                cache_after,
            )
    return result
