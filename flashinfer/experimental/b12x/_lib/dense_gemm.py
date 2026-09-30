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

from b12x._lib.program_cache import program_cache


from b12x._lib.quant.block_codec import BLOCK_CODECS, block_codec
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
from cutlass import Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp.mma import Field as WarpField
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo

from b12x._lib.compiler import (
    KernelCompileSpec,
    compile as b12x_compile,
)
from b12x._lib.compile_plan import attach_programs
from b12x._lib.utils import (
    cuda_stream_from_int_or_current,
    cuda_stream_to_int,
    current_cuda_stream,
    cutlass_to_torch_dtype,
    get_cutlass_dtype,
    get_max_active_clusters,
    get_num_sm,
    is_mxfp6_ab_dtype,
    make_ptr,
    mxfp6_logical_k_from_packed_bytes,
    mxfp6_tile_k,
    sm120_make_smem_layout_sfa,
    sm120_make_smem_layout_sfb,
)
from b12x._lib.intrinsics import (
    FLOAT8_E4M3_MAX,
    bfloat2_to_float2_scaled,
    cp_async_bulk_g2s_mbar,
    cp_async4_shared_global,
    cvt_f32x4_to_e4m3x4,
    elem_pointer,
    fabs_f32,
    fmax_f32,
    nvfp4_pair_to_bf16x2_sm120,
    iq2_xs_descriptor_pair_to_bf16x2x2,
    iq2_xxs_descriptor_pair,
    q8_0_pair_to_bf16x2,
    mxfp8_pair_to_bf16x2_sm120,
    get_ptr_as_int64,
    ld_global_b16,
    ld_global_v4_u32,
    ld_shared_v4_u32,
    mma_m16n8k32_f32_e4m3,
    pow2_ceil_ue8m0,
    quantize_block_fp8_mx,
    scatter_add_bf16,
    scatter_add_bf16x2,
    shared_ptr_to_u32,
    st_global_u64,
    st_shared_u16,
    st_shared_u8,
    st_shared_v4_u32,
    u32_as_f32,
    ue8m0_to_output_scale,
    warp_reduce,
)
from b12x._lib.dense_gemm_mxfp6 import emit_mxfp6_dense_mma_k_block
from b12x._lib.fp6 import (
    FLOAT6_E2M3_MAX,
    FLOAT6_E3M2_MAX,
    cvt_f32_to_e2m3x2,
    cvt_f32_to_e3m2x2,
    cvt_f32_to_e4m3x2,
    expand_mxfp6_packed_to_bytes,
    fp6_block_ue8m0_exact,
    mx_gs_numerator,
    ue8m0_output_scale_exact,
)
from b12x._lib.runtime_control import (
    raise_if_kernel_resolution_frozen,
)

logger = logging.getLogger(__name__)
_DENSE_LOW_SM_MAX_SMS = 64
_WO_SPARK_MAX_SMS = _DENSE_LOW_SM_MAX_SMS


def _use_low_sm_dense_tactics(sm_count: int) -> bool:
    return int(sm_count) <= _DENSE_LOW_SM_MAX_SMS


def _dense_spark_policy_for_sm_count(sm_count: int) -> bool:
    return _use_low_sm_dense_tactics(sm_count)


_B12X_TIMING = (
    os.getenv("B12X_TIMING", "0") == "1" or os.getenv("VLLM_B12X_TIMING", "0") == "1"
)
_B12X_TIMING_THRESHOLD_MS = float(
    os.getenv(
        "B12X_TIMING_THRESHOLD_MS",
        os.getenv("VLLM_B12X_TIMING_THRESHOLD_MS", "0"),
    )
)
_B12X_DENSE_SPLITK_TURBO = os.getenv("B12X_DENSE_SPLITK_TURBO", "1") == "1"

# MX-FP6 decode uses at most three mainloop stages when two CTAs share an SM.
_FP6_DECODE_TILE = (16, 64)
_FP6_PREFILL_TILE = (128, 128)
_B12X_DENSE_ATOM_24 = os.getenv("B12X_DENSE_ATOM_24", "0") == "1"
_DENSE_LOAD_PATHS = ("tma", "cpasync")

# Expand-ahead for packed-B: at k_block 0 the MMA warps wait for stage s+1 and
# expand it in place, overlapping the expansion with ALL of stage s's MMA work
# instead of putting it on the critical path at the stage boundary. It requires
# at least four pipeline stages of producer slack.
_PACKED_B_EXPAND_AHEAD = os.environ.get(
    "B12X_PACKED_B_EXPAND_AHEAD", "1"
).lower() not in ("0", "false")

# MX-FP6 fused activation quantization: fuse BF16 activation quantization into
# the GEMM's DMA producer prologue, eliminating the separate quant kernel and
# the HBM round-trip for activation codes+scales. Currently m=1 only (decode
# hot path). The GEMM's producer warp does a full-row amax scan, derives
# gs/alpha, then quantizes each K-tile's 32-element blocks directly into
# sA/sSFA smem. Distinct from the MXFP8 ``fused_quant_a`` machinery.
_DENSE_FUSED_QUANT = os.environ.get("B12X_DENSE_FUSED_QUANT", "0").lower() not in (
    "0",
    "false",
)


@dataclass(frozen=True)
class _DenseGemmPlan:
    mma_tiler_mn: Tuple[int, int]
    load_path: Literal["tma", "cpasync"]
    swap_ab: bool


@triton.jit(do_not_specialize=["total"], do_not_specialize_on_alignment=["total"])
def _reduce_split_k2_bf16_kernel(
    partials, out, total, BLOCK: tl.constexpr
) -> None:
    pid = tl.program_id(0).to(tl.int64)
    total = total.to(tl.int64)
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


@cute.jit
def _emit_plain_fp8_dense_mma_k_block(
    accumulators: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    mt: int,
    nt: int,
    k_block_idx: int,
) -> None:
    acc = accumulators[None, mt, nt]
    a_frag = cute.flatten(
        cute.recast_tensor(tCrA[None, mt, k_block_idx], cutlass.Uint32)
    )
    b_frag = cute.flatten(
        cute.recast_tensor(tCrB[None, nt, k_block_idx], cutlass.Uint32)
    )
    d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        a_frag[0],
        a_frag[1],
        a_frag[2],
        a_frag[3],
        b_frag[0],
        b_frag[1],
    )
    acc[0] = d0
    acc[1] = d1
    acc[2] = d2
    acc[3] = d3


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


def _tile_major_cluster_limit(
    max_active_clusters: int,
    *,
    n: int,
    l: int,
    tile_n: int,
) -> int:
    """Apply the qualified 64-output-tile launch cap by grid geometry."""
    output_tiles = ((n + tile_n - 1) // tile_n) * l
    return min(max_active_clusters, 40) if output_tiles == 64 else max_active_clusters


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
    tile_k: int = 128,
    expected_m: Optional[int] = None,
    generalize_mxfp8_split_k: bool = False,
    generalize_block_fp8_split_k: bool = False,
) -> _DenseGemmPolicy:
    # A declared row bound owns scheduling as well as tile and unroll tuning.
    if expected_m is not None:
        m = expected_m
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
    block_fp8_slices = _select_block_fp8_decode_slices(m, n, k, sm_count)
    planned_block_fp8_split = (
        generalize_block_fp8_split_k
        and mma_tiler_mn == (32, 64)
        and block_fp8_slices > 1
    )
    low_sm_four_way_split = (
        generalize_mxfp8_split_k
        and _use_low_sm_dense_tactics(sm_count)
        and m <= 6
        and n >= 64 * sm_count
        and 4096 <= k <= 6144
        and k % (4 * 128) == 0
        and l == 1
        and mma_tiler_mn == (32, 64)
    )
    if planned_block_fp8_split:
        split_k_slices = block_fp8_slices
    elif low_sm_four_way_split:
        # Joint tile/slice sweeps show that once the 64-column output grid
        # reaches one SM wave, four shorter K ranges amortize the atomic BF16
        # epilogue through the decode regime. Below one wave, direct 16x64 is
        # consistently faster. Keep the K bound at the qualified stage-count
        # range; deeper-K counterexamples flatten or reverse the gain.
        split_k_slices = 4
    elif split_k_candidate:
        if generalize_block_fp8_split_k:
            pass
        elif generalize_mxfp8_split_k:
            if not _use_low_sm_dense_tactics(sm_count):
                work_tiles = (
                    ((m + tile_m - 1) // tile_m)
                    * ((n + tile_n - 1) // tile_n)
                    * l
                )
                four_way_fits = (
                    (m == 8 or k >= 5120)
                    and k % (4 * 128) == 0
                    and work_tiles * 6 >= max_active_clusters
                    and work_tiles * 4 <= max_active_clusters
                )
                if four_way_fits:
                    split_k_slices = 4
                elif work_tiles * 4 >= max_active_clusters:
                    split_k_slices = 2
        else:
            # Block-FP8 and plain-FP8 have not been qualified by the MXFP8
            # slice sweep. Preserve their established policy until they are.
            split_k_slices = (
                4
                if m == 8
                and (n, k) == (4096, 4096)
                and mma_tiler_mn == (16, 128)
                else 2
            )
    if not _B12X_DENSE_SPLITK_TURBO:
        # The FP32 workspace reducer consumes exactly two partials.
        split_k_slices = min(split_k_slices, 2)
    # A declared expected_m owns compile-time tuning for its regime. Without a
    # hint, keep the unroll choice stable throughout the persistent scheduler
    # regime so one warmed kernel covers every live M in that regime.
    large_m_unroll_threshold = (
        4096 if _use_low_sm_dense_tactics(sm_count) else 8192
    )
    use_large_m_unroll = (
        expected_m >= large_m_unroll_threshold
        if expected_m is not None
        else not single_work_tile_per_cta
        and not direct_one_m_tile_scheduler
        and not use_m1_non_tma
    )
    if (
        generalize_mxfp8_split_k
        and _use_low_sm_dense_tactics(sm_count)
        and mma_tiler_mn == (128, 128)
        and tile_k == 64
        and expected_m is not None
    ):
        if 1536 <= expected_m <= 2048 and n >= 4096 and k >= 2048:
            # The bounded medium-prefill BK64 plan needs four-way mainloop
            # unrolling and the unswizzled persistent scheduler.
            use_large_m_unroll = True
        elif expected_m >= 4096 and k >= 4096:
            # At large M the 16-way persistent swizzle is the structural win
            # for BK64. Coupling it off through large_m_unroll causes severe
            # WO-B/Qwen cliffs (up to ~3x at M=8192).
            use_large_m_unroll = False
    if (
        generalize_block_fp8_split_k
        and _use_low_sm_dense_tactics(sm_count)
        and mma_tiler_mn == (128, 128)
        and expected_m is not None
        and expected_m >= 2048
        and k >= 10240
    ):
        # K128 block scaling lengthens each staged accumulation. Once the
        # reduction reaches 80 scale blocks, a 128-row tile wins by reusing B;
        # keeping two-way unroll and the 16-way scheduler swizzle avoids the
        # large-M cliff seen with the generic FP8 unroll threshold.
        use_large_m_unroll = False
    return _DenseGemmPolicy(
        single_work_tile_per_cta=single_work_tile_per_cta,
        direct_one_m_tile_scheduler=direct_one_m_tile_scheduler,
        use_m1_non_tma=use_m1_non_tma,
        split_k_slices=split_k_slices,
        split_k_atomic_bf16=_B12X_DENSE_SPLITK_TURBO,
        large_m_unroll=(
            ab_dtype == cutlass.Float8E4M3FN and use_large_m_unroll and l == 1
        ),
    )


def _select_block_fp8_decode_slices(
    m: int,
    n: int,
    k: int,
    sm_count: int,
) -> int:
    """Select a K128 block-FP8 decode split count from SM/grid geometry."""
    if not _use_low_sm_dense_tactics(sm_count):
        return 2 if 2 <= m <= 6 and k >= 4096 and k % (2 * 128) == 0 else 1

    n_tiles_64 = (n + 63) // 64
    minimum_tiles = (2 * sm_count + 2) // 3
    if (
        m > 6
        or n_tiles_64 < minimum_tiles
        or k < 4096
    ):
        return 1
    if k % (4 * 128) == 0:
        return 4
    if k <= 4352 and k % (2 * 128) == 0:
        return 2
    return 1


@cute.jit
def _spread_fp6_group_u32(g: Uint32) -> Uint32:
    """Spread 4 packed 6-bit codes (24 bits) into 4 byte lanes (bits[5:0] each).

    Output byte ``i`` holds ``(g >> 6*i) & 0x3F`` — identical per-group math to
    :func:`b12x._lib.fp6.expand_mxfp6_packed_to_bytes`. Two-step binary
    spread (12-bit halves to 16-bit lanes, then 6-bit fields to byte lanes): 2
    shifts + 2 LOP3-fusable mask/or pairs, ~30% fewer ops than masking each
    field out of ``g`` individually.
    """
    a = (g & Uint32(0x00000FFF)) | ((g << Uint32(4)) & Uint32(0x0FFF0000))
    return (a & Uint32(0x003F003F)) | ((a << Uint32(2)) & Uint32(0x3F003F00))


@cute.jit
def _expand_packed_b_triplet(
    a: Uint32, b: Uint32, c: Uint32
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Expand 12 packed bytes (3 LE u32 words = 16 FP6 codes) to 4 output words.

    Little-endian regroup into four 24-bit groups of 4 codes, then byte-lane
    spread per group.
    """
    g0 = a & Uint32(0x00FFFFFF)
    g1 = (a >> Uint32(24)) | ((b & Uint32(0xFFFF)) << Uint32(8))
    g2 = (b >> Uint32(16)) | ((c & Uint32(0xFF)) << Uint32(16))
    g3 = c >> Uint32(8)
    return (
        _spread_fp6_group_u32(g0),
        _spread_fp6_group_u32(g1),
        _spread_fp6_group_u32(g2),
        _spread_fp6_group_u32(g3),
    )


@cute.jit
def _expand_packed_b_stage_smem(
    sb_base_addr: Int32,
    stage: Int32,
    tidx: Int32,
    tile_n: cutlass.Constexpr,
    tile_k: cutlass.Constexpr,
    num_threads: cutlass.Constexpr,
    sync_barrier,
) -> None:
    """Expand one 3:4-packed B stage IN PLACE into the byte-container sB stage.

    TMA stages the packed tile (96 B/row, plain k-major, no swizzle) into the
    BOTTOM ``tile_n * 3*tile_k/4`` bytes of the sB stage itself — no separate
    staging buffer, which buys an extra pipeline stage (3 -> 4 for the decode
    tile). The expanded output (128 B/row, swizzled) overlaps the packed input
    region, so expansion is two-phase: every thread loads ALL of its packed
    rows into registers, one MMA-group named-barrier, then writes the expanded
    swizzled rows. Each thread owns whole rows (``tile_n / num_threads`` of
    them), so within a row reads always precede writes; the barrier orders
    them across threads.

    Raw addressing is deliberate: ``cute.recast_tensor(sB, Uint8)`` STRIPS the
    smem swizzle on this build (confirmed in the fused-MoE FC2 requant path).
    The 8-bit K-major SW128 atom (Sw<3,4,3>, 8x128 = 1024 B) gives
    ``physical = flat ^ ((row & 7) << 4)`` — the XOR touches bits 4..6 only,
    so every 16-byte unit stays contiguous and the traffic is fully 128-bit
    (6 x ld.shared.v4 in, 8 x st.shared.v4 out per row). Per-group bit math
    matches :func:`b12x._lib.fp6.expand_mxfp6_packed_to_bytes` exactly,
    so downstream ldmatrix/MMA sees bytes identical to the pre-expanded path.

    Runs on the MMA warp group only; the caller must still follow with the
    MMA-group named barrier before any ldmatrix reads of this stage.
    """
    # Whole-row ownership; ceil-divide so configs with more threads than rows
    # (e.g. the 256-thread (128,128) tile) stay correct — guarded threads skip
    # the row I/O but ALL threads reach the phase barrier.
    rows_per_thread = (tile_n + num_threads - 1) // num_threads
    packed_row_bytes = tile_k * 3 // 4
    words_per_row = packed_row_bytes // 4
    sb_stage = sb_base_addr + stage * Int32(tile_n * tile_k)

    # Phase 1: read all owned packed rows into registers (static rmem layout).
    w = cute.make_rmem_tensor((rows_per_thread * words_per_row,), Uint32)
    for r_i in cutlass.range_constexpr(rows_per_thread):
        row = Int32(tidx) + Int32(r_i * num_threads)
        if row < Int32(tile_n):
            src = sb_stage + row * Int32(packed_row_bytes)
            for c in cutlass.range_constexpr(words_per_row // 4):
                w0, w1, w2, w3 = ld_shared_v4_u32(src + Int32(c * 16))
                w[r_i * words_per_row + c * 4 + 0] = w0
                w[r_i * words_per_row + c * 4 + 1] = w1
                w[r_i * words_per_row + c * 4 + 2] = w2
                w[r_i * words_per_row + c * 4 + 3] = w3

    # All packed reads must complete before any expanded write lands.
    sync_barrier.arrive_and_wait()

    # Phase 2: expand and write the swizzled byte-container rows.
    for r_i in cutlass.range_constexpr(rows_per_thread):
        row = Int32(tidx) + Int32(r_i * num_threads)
        if row < Int32(tile_n):
            sw = (row & Int32(7)) << Int32(4)
            flat = row * Int32(tile_k)
            for o in cutlass.range_constexpr(tile_k // 16):
                base = r_i * words_per_row + o * 3
                o0, o1, o2, o3 = _expand_packed_b_triplet(
                    w[base + 0], w[base + 1], w[base + 2]
                )
                st_shared_v4_u32(
                    sb_stage + ((flat + Int32(o * 16)) ^ sw), o0, o1, o2, o3
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
        alpha_reciprocal: bool = False,
    ):
        if weight_only not in (None, "nvfp4", "mxfp8", "iq2_xs", "iq2_xxs", "q8_0"):
            raise ValueError("weight_only must be nvfp4, mxfp8, or iq2_xs")
        self.weight_only = weight_only
        self.a16 = weight_only is not None
        self.a16_fp4 = weight_only == "nvfp4"
        self.a16_block = weight_only in BLOCK_CODECS
        self.a16_iq2_xxs = weight_only == "iq2_xxs"
        self.a16_q8 = weight_only == "q8_0"
        self.a16_lut_bytes = block_codec(weight_only).lut_bytes() if self.a16_block else 0
        self.a16_block_transposed = self.a16_block and swap_ab
        self.a16_pack_factor = block_codec(weight_only).pack_factor if self.a16_block else 2 if self.a16_fp4 else 1
        self.alpha_reciprocal = bool(alpha_reciprocal)
        if self.a16:
            if (load_path != "tma" or (swap_ab and not self.a16_block_transposed) or fused_quant_a or plain_fp8
                    or block_fp8 or b_packed or quantize_c or sfb_k_reuse):
                raise ValueError("A16 requires ordinary K-major TMA operands")
            if mma_k != 16 or sf_vec_size != (16 if self.a16_fp4 else 32):
                raise ValueError("A16 requires BF16 K16 MMA and matching weight scale groups")
            if self.a16_block_transposed and (mma_tiler_mn[0] != 8 or split_k_slices != 1):
                raise ValueError("transposed IQ2_XS requires an eight-row tile without split-K")
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
        if self.a16_block_transposed:
            self.atom_shape = (mma_atom_mn[0] // 16, 1, 1)
        elif mma_atom_mn in ((16, 64), (16, 128)):
            # This table sets the MMA atom tiling only. The warp count is a
            # SEPARATE table below and both must list the same tiles: an atom
            # shape covering two warps of work under a launch geometry sized
            # for eight leaves warps 2-7 with no valid tile, reading shared
            # memory past the end of the staged operands.
            self.atom_shape = (1, 8 if self.a16_block else 2, 1)
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
            self.num_mma_warps = 8 if self.a16_block else 2
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
        permutation_mnk = self.mma_tile_shape_mnk if self.a16 else sm120_utils.get_permutation_mnk(
            self.mma_tile_shape_mnk,
            32 if self.block_fp8 else self.sf_vec_size,
            cutlass.const_expr(
                self.a_dtype == cutlass.Float8E4M3FN
                or self.a_dtype == cutlass.Float6E3M2FN
                or self.a_dtype == cutlass.Float6E2M3FN
            ),
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
            self.a16_scale_bytes = (128 * block_codec(self.weight_only).metadata_bytes * (self.tile_shape_mnk[2] // 32 if self.a16_q8 else 1)) if self.a16_block else max(1, self.tile_shape_mnk[2] // (self.sf_vec_size * 4)) * 512
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
                    self.tile_shape_mnk, self.a_dtype,
                    cutlass.Float4E2M1FN if self.a16_fp4 else cutlass.Uint8,
                    self.sf_dtype, sfa_smem_layout_per_stage, sfb_smem_layout_per_stage,
                    self.epi_tile, self.c_dtype,
                    self.smem_capacity - self.a16_lut_bytes * self.occupancy,
                    self.occupancy, epi_stage_cap=1, minimum_ab_stage=0,
                    b_storage_bits=8 // self.a16_pack_factor if self.a16_block else None,
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
            self.sfb_smem_layout_staged = cute.make_layout((self.a16_scale_bytes, 1, self.ab_stage))
            self.a16_packed_k = self.tile_shape_mnk[2] // self.a16_pack_factor
            self.a16_b_smem_layout = cute.make_layout(
                (self.tile_shape_mnk[1], self.a16_packed_k, self.ab_stage),
                stride=(self.a16_packed_k, 1, self.tile_shape_mnk[1] * self.a16_packed_k),
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
                b, self.a16_b_smem_layout,
                (self.tile_shape_mnk[1], self.a16_packed_k), 1,
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
            self.block_fp8 or self.a16 or self.manual_bk64_sf or self.direct_sfb_representative
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
                    cute.cosize(self.a16_b_smem_layout) if self.a16 else cute.cosize(self.b_smem_layout_staged)
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
        self, fragment: cute.Tensor, coordinates: cute.Tensor,
        packed: cute.Tensor, scales: cute.Tensor, lut: cute.Tensor, stage: Int32,
        n_tile: Int32, k_tile: Int32, size_n: cutlass.Constexpr,
        size_k: cutlass.Constexpr,
    ):
        pairs = cute.recast_tensor(fragment, Uint32)
        packed_u16 = cute.recast_tensor(packed, cutlass.Uint16)
        if cutlass.const_expr(self.a16_block):
            packed_u32 = cute.recast_tensor(packed, Uint32)
            bases = cute.recast_tensor(scales, cutlass.Uint16)
            rows_per_fragment = 2 if self.a16_block_transposed else 1
            for i in cutlass.range_constexpr(cute.size(fragment) // (4 * rows_per_fragment)):
                for row in cutlass.range_constexpr(rows_per_fragment):
                    local_n, local_k = coordinates[i * 4 * rows_per_fragment + row * 2]
                    scale_n = local_n + (n_tile % self.sfb_tiles_per_block) * self.tile_shape_mnk[1]
                    block_k = (k_tile * self.tile_shape_mnk[2] + local_k) % 256
                    if cutlass.const_expr(self.a16_q8):
                        base = Uint32(bases[local_k // 32 * 128 + scale_n, 0, stage])
                        p0 = q8_0_pair_to_bf16x2(Uint32(packed_u16[local_n, local_k // 2, stage]), base)
                        p1 = q8_0_pair_to_bf16x2(Uint32(packed_u16[local_n, (local_k + 8) // 2, stage]), base)
                    else:
                        if cutlass.const_expr(self.a16_iq2_xxs):
                            grids = Uint32(packed_u32[local_n, local_k // 32 * 2, stage])
                            signs_scale = Uint32(packed_u32[local_n, local_k // 32 * 2 + 1, stage])
                            descriptors = iq2_xxs_descriptor_pair(grids, signs_scale, Int32(local_k // 16 % 2))
                            subscale = signs_scale >> 28
                        else:
                            scale_pair = Uint32(scales[256 + (block_k // 32) * 128 + scale_n, 0, stage])
                            subscale = (scale_pair >> ((block_k // 16 % 2) * 4)) & 15
                            descriptors = Uint32(packed_u32[local_n, local_k // 16, stage])
                        p0, p1 = iq2_xs_descriptor_pair_to_bf16x2x2(
                            descriptors,
                            Uint32(bases[scale_n, 0, stage]), subscale,
                            Int64(shared_ptr_to_u32(lut.iterator)), Int32(local_k % 8),
                        )
                    pairs[i * 2 * rows_per_fragment + row] = p0
                    pairs[i * 2 * rows_per_fragment + row + rows_per_fragment] = p1
        else:
            for i in cutlass.range_constexpr(cute.size(fragment) // 2):
                local_n, local_k = coordinates[i * 2]
                scale_n = local_n + (n_tile % self.sfb_tiles_per_block) * self.tile_shape_mnk[1]
                valid = cutlass.Boolean(True)
                if cutlass.const_expr(size_k % self.tile_shape_mnk[2] != 0):
                    valid = Int64(k_tile) * self.tile_shape_mnk[2] + local_k < size_k
                packed_pair = Uint32(0)
                if valid:
                    group = local_k // self.sf_vec_size
                    if cutlass.const_expr(self.tile_shape_mnk[2] < self.sf_vec_size * 4):
                        group += (k_tile % 2) * 2
                    scale_offset = (group // 4 * 512 + (scale_n % 32) * 16
                                    + (scale_n // 32) * 4 + group % 4)
                    byte = Uint32(scales[scale_offset, 0, stage])
                    if cutlass.const_expr(self.a16_fp4):
                        packed_pair = nvfp4_pair_to_bf16x2_sm120(
                            Uint32(packed[local_n, local_k // 2, stage]), byte,
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
            tma_copy_bytes += self.tile_shape_mnk[1] * self.a16_packed_k + self.a16_scale_bytes
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
            a16_lut = smem.allocate_tensor(cutlass.Uint8, cute.make_layout((self.a16_lut_bytes,)), byte_alignment=16)
            lut_address = shared_ptr_to_u32(a16_lut.iterator)
            for chunk_base in cutlass.range_constexpr((self.a16_lut_bytes // 16 + self.threads_per_cta - 1) // self.threads_per_cta):
                chunk = Int32(tidx) + chunk_base * self.threads_per_cta
                if chunk < self.a16_lut_bytes // 16:
                    cp_async4_shared_global(lut_address + chunk * 16,
                                           get_ptr_as_int64(directSFA_mkl, Int64(chunk) * 16))
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)

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
                mB_nkl, (self.tile_shape_mnk[1], self.a16_packed_k),
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
                mSFB_nkl, self.sfb_tile_shape_nk, (None, None, None),
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
                tCgC = thr_mma.partition_C(cute.make_identity_tensor(
                    (self.tile_shape_mnk[1], self.tile_shape_mnk[0])
                ))
                a16_weight_fragments = tCrA
                a16_b_coordinates = thr_mma.partition_A(cute.make_identity_tensor(
                    (self.tile_shape_mnk[1], self.tile_shape_mnk[2])
                ))
            else:
                tCgC = thr_mma.partition_C(gC_mnl)
                a16_weight_fragments = tCrB
                a16_b_coordinates = thr_mma.partition_B(cute.make_identity_tensor(
                    (self.tile_shape_mnk[1], self.tile_shape_mnk[2])
                ))
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
                k_tile_iter_cnt = (Int32(block_idx[1]) + 1) * k_tile_cnt // self.split_k_slices - k_tile_start
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
                (Int32(block_idx[0]) if self.a16 else Int32(0), Int32(block_idx[2]), Int32(0)),
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
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 2 if self.a16 else 4),
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
                    cute.copy(smem_tiled_copy_B, tCsB_p[None, None, 0],
                              tCrB_copy_view[None, None, 0])
                else:
                    cute.copy(smem_tiled_copy_A, tCsA_p[None, None, 0],
                              tCrA_copy_view[None, None, 0])
                if cutlass.const_expr(self.a16):
                    self._load_a16_b_fragment(
                        a16_weight_fragments[None, None, 0], a16_b_coordinates[None, None, 0],
                        sBPacked, sSFB, a16_lut, mainloop_consumer_state.index,
                        tile_coord_mnl[1], k_tile_start,
                        cute.size(directB_nkl, mode=[0]), cute.size(directB_nkl, mode=[1]) * self.a16_pack_factor,
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
                            cute.gemm(tiled_mma, accumulators,
                                      tCrA[None, None, k_block_idx],
                                      tCrB[None, None, k_block_idx], accumulators)
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
                            cute.copy(smem_tiled_copy_B, tCsB_p[None, None, k_block_next],
                                      tCrB_copy_view[None, None, k_block_next])
                        else:
                            cute.copy(smem_tiled_copy_A, tCsA_p[None, None, k_block_next],
                                      tCrA_copy_view[None, None, k_block_next])
                        if cutlass.const_expr(self.a16):
                            self._load_a16_b_fragment(
                                a16_weight_fragments[None, None, k_block_next], a16_b_coordinates[None, None, k_block_next],
                                sBPacked, sSFB, a16_lut, mainloop_consumer_state.index,
                                tile_coord_mnl[1], k_tile_start + Int32(k_tile) + (1 if k_block_next == 0 else 0),
                                cute.size(directB_nkl, mode=[0]), cute.size(directB_nkl, mode=[1]) * self.a16_pack_factor,
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
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if k_block_next > 0:
                        if cutlass.const_expr(self.a16_block_transposed):
                            cute.copy(smem_tiled_copy_B, tCsB_p[None, None, k_block_next],
                                      tCrB_copy_view[None, None, k_block_next])
                        else:
                            cute.copy(smem_tiled_copy_A, tCsA_p[None, None, k_block_next],
                                      tCrA_copy_view[None, None, k_block_next])
                        if cutlass.const_expr(self.a16):
                            self._load_a16_b_fragment(
                                a16_weight_fragments[None, None, k_block_next], a16_b_coordinates[None, None, k_block_next],
                                sBPacked, sSFB, a16_lut, mainloop_consumer_state.index,
                                tile_coord_mnl[1], k_tile_start + Int32(k_tile_iter_cnt - 1),
                                cute.size(directB_nkl, mode=[0]), cute.size(directB_nkl, mode=[1]) * self.a16_pack_factor,
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
                        cute.gemm(tiled_mma, accumulators,
                                  tCrA[None, None, k_block_idx],
                                  tCrB[None, None, k_block_idx], accumulators)
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
                            logical_k = cute.size(directB_nkl, mode=[1]) * self.a16_pack_factor
                            if cutlass.const_expr(self.a16_block):
                                if cutlass.const_expr(self.a16_iq2_xxs or self.a16_q8):
                                    size_n = cute.size(directB_nkl, mode=[0])
                                    base_n = Int64(tile_coord_mnl[1]) * self.tile_shape_mnk[1] // 128 * 128
                                    block_k = 32 if self.a16_q8 else 256
                                    barrier = shared_ptr_to_u32(mainloop_pipeline.producer_get_barrier(mainloop_producer_state))
                                    for unit in cutlass.range_constexpr(self.a16_scale_bytes // 256):
                                        base_k = Int64(k_tile_global) * self.tile_shape_mnk[2] // block_k + unit
                                        if base_k >= logical_k // block_k:
                                            base_k = Int64(logical_k // block_k - 1)
                                        dst = shared_ptr_to_u32(elem_pointer(sSFB, (unit * 256, 0, mainloop_producer_state.index)))
                                        base_chunk = 0
                                        base_row = Int64(0)
                                        if base_n + 128 <= size_n:
                                            cp_async_bulk_g2s_mbar(dst, get_ptr_as_int64(directSFB_nkl, (base_k * size_n + base_n) * 2), Int32(256), barrier)
                                        else:
                                            # N8 alignment permits whole 16-byte transfers;
                                            # masked channels reuse a valid base span.
                                            for base_chunk in cutlass.range_constexpr(16):
                                                base_row = base_n + Int64(base_chunk * 8)
                                                if base_row >= size_n:
                                                    base_row = Int64(0)
                                                cp_async_bulk_g2s_mbar(dst + Int32(base_chunk * 16), get_ptr_as_int64(directSFB_nkl, (base_k * size_n + base_row) * 2), Int32(16), barrier)
                                else:
                                    scale_offset = (
                                        Int64(tile_coord_mnl[1]) * self.tile_shape_mnk[1] // 128 * (logical_k // 256)
                                        + Int64(k_tile_global) * self.tile_shape_mnk[2] // 256
                                    ) * 1280
                                    cp_async_bulk_g2s_mbar(
                                        shared_ptr_to_u32(elem_pointer(sSFB, (0, 0, mainloop_producer_state.index))),
                                        get_ptr_as_int64(directSFB_nkl, scale_offset), Int32(1280),
                                        shared_ptr_to_u32(mainloop_pipeline.producer_get_barrier(mainloop_producer_state)),
                                    )
                            else:
                                scale_k_tiles = (logical_k // self.sf_vec_size + 3) // 4
                                for unit in cutlass.range_constexpr(self.a16_scale_bytes // 512):
                                    scale_k = Int64(k_tile_global) * self.tile_shape_mnk[2] // (self.sf_vec_size * 4) + unit
                                    if cutlass.const_expr(logical_k % self.tile_shape_mnk[2] != 0):
                                        if scale_k >= scale_k_tiles:
                                            scale_k = Int64(scale_k_tiles - 1)
                                    scale_offset = (
                                        Int64(tile_coord_mnl[1]) * self.tile_shape_mnk[1] // 128 * scale_k_tiles
                                        + scale_k
                                    ) * 512
                                    cp_async_bulk_g2s_mbar(
                                        shared_ptr_to_u32(elem_pointer(sSFB, (unit * 512, 0, mainloop_producer_state.index))),
                                        get_ptr_as_int64(directSFB_nkl, scale_offset), Int32(512),
                                        shared_ptr_to_u32(mainloop_pipeline.producer_get_barrier(mainloop_producer_state)),
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
                                % tile_sched_params.swizzle_size != 0
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
                                    n_start = Int64(tile_coord_mnl[1]) * self.tile_shape_mnk[1]
                                    descriptor_offset = (
                                        Int64(tile_coord_mnl[2]) * size_n * storage_k
                                        + (n_start // 128 * (storage_k // 64)
                                           + Int64(k_tile_global)) * 8192
                                        + n_start % 128 * 64
                                    )
                                    cp_async_bulk_g2s_mbar(
                                        shared_ptr_to_u32(elem_pointer(
                                            sBPacked, (0, 0, mainloop_producer_state.index),
                                        )),
                                        get_ptr_as_int64(directB_nkl, descriptor_offset),
                                        Int32(self.tile_shape_mnk[1] * 64),
                                        shared_ptr_to_u32(mainloop_pipeline.producer_get_barrier(mainloop_producer_state)),
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
            + cute.size(b_shape) * (b_dtype.width if b_storage_bits is None else b_storage_bits) // 8
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
            grid = (num_ctas_mnl[0] if split_k_all_m else 1, split_k_slices, num_ctas_mnl[1])
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
                and (mma_tiler_mn in ((16, 64), (16, 128), (32, 64), (32, 128), (64, 64), (64, 128))
                     or weight_only in BLOCK_CODECS and mma_tiler_mn in ((8, 64), (8, 128)))
                and cluster_shape_mn == (1, 1) and l == 1
                and a_major == b_major == "k" and c_major == "n"
                and load_path == "tma" and not block_fp8
                and swap_ab == (weight_only in BLOCK_CODECS and mma_tiler_mn[0] == 8)
                and n % 8 == 0 and k % (block_codec(weight_only).block_weights if weight_only in BLOCK_CODECS else 32) == 0
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
        plain_fp8: bool = False,
        block_fp8: bool = False,
        weight_only: Optional[str] = None,
        alpha_reciprocal: bool = False,
        input_k: Optional[int] = None,
    ):
        self._weight_only = weight_only
        self._alpha_reciprocal = alpha_reciprocal
        self._input_k = k if input_k is None else input_k
        self._storage_k = k // (block_codec(weight_only).pack_factor if weight_only in BLOCK_CODECS else 2 if weight_only == "nvfp4" else 1)
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
        self._atom_shape_24 = _B12X_DENSE_ATOM_24
        self._sfb_k_reuse = sfb_k_reuse
        self._b_tile_major = b_tile_major
        self._quantize_c = quantize_c
        self._alpha_is_one = alpha_is_one
        self._direct_sfa_live16 = direct_sfa_live16
        self._direct_m1_wo_a_inputs = direct_m1_wo_a_inputs
        self._target_occupancy = target_occupancy
        self._plain_fp8 = bool(plain_fp8)
        self._block_fp8 = bool(block_fp8)
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
            block_fp8=block_fp8,
            weight_only=weight_only,
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
        ):
            self._max_active_clusters = _tile_major_cluster_limit(
                self._max_active_clusters,
                n=n,
                l=l,
                tile_n=mma_tiler_mn[1],
            )

    def compile_key(self) -> tuple[object, ...]:
        """Return every value that can specialize the generated kernel."""

        key = (
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
            self._plain_fp8,
        )

        return key if self._weight_only is None else key + (
            self._weight_only, self._alpha_reciprocal, self._input_k,
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
                (m, self._input_k, self._l),
                order=(0, 1, 2) if self._a_major == "m" else (1, 0, 2),
            ),
        )
        if cutlass.const_expr(self._weight_only in BLOCK_CODECS):
            descriptor_tile_n = 128 if self._n % 128 == 0 else 8
            spec = block_codec(self._weight_only)
            b_layout = cute.make_layout(
                ((descriptor_tile_n, self._n // descriptor_tile_n), (spec.payload_bytes, self._k // spec.block_weights), self._l),
                stride=((spec.payload_bytes, descriptor_tile_n * self._storage_k),
                        (1, descriptor_tile_n * spec.payload_bytes), self._n * self._storage_k),
            )
        elif cutlass.const_expr(self._b_tile_major):
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
                (self._n, self._storage_k, self._l),
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
        if cutlass.const_expr(self._weight_only is not None):
            scale_bytes = ((self._n + 127) // 128) * ((self._k // self._sf_vec_size + 3) // 4) * 512
            if cutlass.const_expr(self._weight_only in BLOCK_CODECS):
                scale_bytes = (self._n * (self._k // block_codec(self._weight_only).block_weights) * 2 if self._weight_only != "iq2_xs"
                               else ((self._n + 127) // 128) * (self._k // 256) * 1280)
            sfa_tensor = cute.make_tensor(sfa_ptr, layout=cute.make_layout((max(1, block_codec(self._weight_only).lut_bytes()) if self._weight_only in BLOCK_CODECS else 1,)))
            sfb_tensor = cute.make_tensor(sfb_ptr, layout=cute.make_layout((scale_bytes,)))
        elif cutlass.const_expr(self._block_fp8):
            sfa_tensor = cute.make_tensor(
                sfa_ptr,
                layout=cute.make_ordered_layout(
                    (m, self._k // 128, self._l), order=(1, 0, 2)
                ),
            )
            sfb_tensor = cute.make_tensor(
                sfb_ptr,
                layout=cute.make_ordered_layout(
                    (self._n // 128, self._k // 128, self._l),
                    order=(1, 0, 2),
                ),
            )
        else:
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
            plain_fp8=self._plain_fp8,
            block_fp8=self._block_fp8,
            weight_only=self._weight_only,
            alpha_reciprocal=self._alpha_reciprocal,
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


class _DenseGemmMxfp6Launch(_DenseGemmLaunch):
    """MX-FP6 (W6A6/W6A8) launch: FP6 codes in Float8E4M3FN byte-containers.

    Adds the per-operand MX sub-formats, optional native 3:4-packed B
    streaming, and the optional fused BF16->FP6 activation-quant prologue
    (B12X_DENSE_FUSED_QUANT). Only the unswapped single-slice TMA plan
    is wired; the constructor of DenseGemmKernel asserts the rest.
    """

    def __init__(
        self,
        *args,
        mxfp6_fmt: Optional[str] = None,
        mxfp6_fmt_a: Optional[str] = None,
        mxfp6_fmt_b: Optional[str] = None,
        b_packed: bool = False,
        a_preexpanded: bool = False,
        b_preexpanded: bool = False,
        row_scale: bool = False,
        fused_quant: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if mxfp6_fmt_a is None and mxfp6_fmt_b is None:
            mxfp6_fmt_a = mxfp6_fmt
            mxfp6_fmt_b = mxfp6_fmt
        elif mxfp6_fmt_a is None or mxfp6_fmt_b is None:
            raise ValueError(
                "mxfp6_fmt_a and mxfp6_fmt_b must both be set or both None"
            )
        self._mxfp6_fmt_a = mxfp6_fmt_a
        self._mxfp6_fmt_b = mxfp6_fmt_b
        self._b_packed = bool(b_packed)
        # Key-only host-side flags (do not change codegen, kept for cache
        # hygiene) plus the import-time fused-quant knob, which DOES change
        # the generated kernel via DenseGemmKernel.a_bf16_fused.
        self._a_preexpanded = bool(a_preexpanded)
        self._b_preexpanded = bool(b_preexpanded)
        self._fused_quant_env = (
            _DENSE_FUSED_QUANT if fused_quant is None else bool(fused_quant)
        )
        self._row_scale = bool(row_scale)

    def compile_key(self) -> tuple[object, ...]:
        return (
            "mxfp6",
            self._mxfp6_fmt_a,
            self._mxfp6_fmt_b,
            self._b_packed,
            self._a_preexpanded,
            self._b_preexpanded,
            self._fused_quant_env,
            self._row_scale,
            *super().compile_key(),
        )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        x_bf16_ptr: cute.Pointer,
        w_gscale_ptr: cute.Pointer,
        row_scale_ptr: cute.Pointer,
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
        # Packed B carries 3 bytes per 4 codes: gmem extent is 3K/4.
        b_k_extent = self._k * 3 // 4 if self._b_packed else self._k
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout(
                (self._n, b_k_extent, self._l),
                order=(0, 1, 2) if self._b_major == "n" else (1, 0, 2),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout(
                (m, self._n, self._c_l),
                order=(0, 1, 2) if self._c_major == "m" else (1, 0, 2),
            ),
        )
        alpha_tensor = cute.make_tensor(
            alpha_ptr,
            layout=cute.make_ordered_layout((1,), order=(0,)),
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, layout=cute.make_layout((1,)))
        sfb_tensor = cute.make_tensor(sfb_ptr, layout=cute.make_layout((1,)))
        x_bf16_tensor = cute.make_tensor(
            x_bf16_ptr,
            layout=cute.make_ordered_layout((m, self._k), order=(0, 1)),
        )
        w_gscale_tensor = cute.make_tensor(
            w_gscale_ptr,
            layout=cute.make_ordered_layout((Int32(1),), order=(0,)),
        )
        if cutlass.const_expr(self._row_scale):
            # Row scaling and quantized-C are mutually exclusive, so the
            # existing quant_c_values operand carries the FP6 epilogue scale.
            row_scale_tensor = cute.make_tensor(
                row_scale_ptr,
                layout=cute.make_ordered_layout((m,), order=(0,)),
            )
        else:
            row_scale_tensor = alpha_tensor
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
            # The M=1 FP6 shape uses a 128-row MMA/TMA tile. In this lowering,
            # the narrow A/SFA/C tensor-map paths illegal-instruction instead
            # of relying on OOB handling, so keep them as explicit direct paths
            # (also required by the fused activation-quant prologue).
            use_m1_non_tma_a=policy.use_m1_non_tma,
            use_m1_non_tma_c=policy.use_m1_non_tma,
            use_m1_non_tma_sfa=policy.use_m1_non_tma,
            load_path=self._load_path,
            swap_ab=self._swap_ab,
            sfb_k_reuse=self._sfb_k_reuse,
            alpha_is_one=self._alpha_is_one,
            mxfp6_fmt_a=self._mxfp6_fmt_a,
            mxfp6_fmt_b=self._mxfp6_fmt_b,
            b_packed=self._b_packed,
            target_occupancy=self._target_occupancy,
            row_scale=self._row_scale,
            fused_quant_bf16=self._fused_quant_env,
        )(
            a_tensor,
            a_tensor,
            alpha_tensor,
            alpha_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            row_scale_tensor,
            alpha_tensor,
            alpha_tensor,
            alpha_tensor,
            self._max_active_clusters,
            current_stream,
            x_bf16=x_bf16_tensor,
            w_gscale=w_gscale_tensor,
        )


@program_cache
def _get_compiled_dense_gemm_mxfp6(
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
    mxfp6_fmt_a: Optional[str],
    mxfp6_fmt_b: Optional[str],
    b_packed: bool,
    a_preexpanded: bool,
    b_preexpanded: bool,
    alpha_is_one: bool,
    row_scale: bool,
    fused_quant: bool,
) -> Callable:
    def _make_runtime_pointers(
        input_tensors: Optional[List[torch.Tensor]],
    ) -> List[cute.Pointer]:
        if input_tensors is None:
            (
                a_data_ptr,
                b_data_ptr,
                sfa_data_ptr,
                sfb_data_ptr,
                c_data_ptr,
                alpha_data_ptr,
                x_bf16_data_ptr,
                w_gscale_data_ptr,
                row_scale_data_ptr,
            ) = [16 for _ in range(9)]
        else:
            (
                a_tensor_gpu,
                b_tensor_gpu,
                sfa_tensor_gpu,
                sfb_tensor_gpu,
                c_tensor_gpu,
                alpha_tensor_gpu,
                x_bf16_tensor_gpu,
                w_gscale_tensor_gpu,
                row_scale_tensor_gpu,
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
            x_bf16_data_ptr = (
                x_bf16_tensor_gpu.data_ptr() if x_bf16_tensor_gpu is not None else 16
            )
            w_gscale_data_ptr = (
                w_gscale_tensor_gpu.data_ptr()
                if w_gscale_tensor_gpu is not None
                else 16
            )
            row_scale_data_ptr = (
                row_scale_tensor_gpu.data_ptr()
                if row_scale_tensor_gpu is not None
                else 16
            )

        return [
            make_ptr(ab_dtype, a_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(ab_dtype, b_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(sf_dtype, sfa_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(sf_dtype, sfb_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(c_dtype, c_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(
                alpha_dtype, alpha_data_ptr, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16,
                x_bf16_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float32,
                w_gscale_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                c_dtype,
                row_scale_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
        ]

    launch = _DenseGemmMxfp6Launch(
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
        load_path="tma",
        swap_ab=False,
        sfb_k_reuse=False,
        alpha_is_one=alpha_is_one,
        mxfp6_fmt_a=mxfp6_fmt_a,
        mxfp6_fmt_b=mxfp6_fmt_b,
        b_packed=b_packed,
        a_preexpanded=a_preexpanded,
        b_preexpanded=b_preexpanded,
        row_scale=row_scale,
        fused_quant=fused_quant,
        # MX-FP6 uses the same occupancy policy as the other dense kernels.
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
            load_path="tma",
            swap_ab=False,
            b_tile_major=False,
            is_mxfp6=True,
        ),
    )
    compile_key = launch.compile_key()
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=compile_key,
    )
    compiled_kernel = b12x_compile(
        launch,
        *_make_runtime_pointers(None),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("gemm.dense.mxfp6", 1, compile_key),
    )

    def tensor_api(
        a_tensor_gpu: torch.Tensor,
        b_tensor_gpu: torch.Tensor,
        sfa_tensor_gpu: torch.Tensor,
        sfb_tensor_gpu: torch.Tensor,
        c_tensor_gpu: Optional[torch.Tensor] = None,
        alpha_tensor_gpu: Optional[torch.Tensor] = None,
        stream_int: Optional[int] = None,
        x_bf16_tensor_gpu: Optional[torch.Tensor] = None,
        w_gscale_tensor_gpu: Optional[torch.Tensor] = None,
        row_scale_tensor_gpu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m = a_tensor_gpu.shape[0]
        if c_tensor_gpu is None:
            c_tensor_gpu = _empty_dense_gemm_output(
                int(m),
                n,
                c_l,
                dtype=cutlass_to_torch_dtype(c_dtype),
                device=a_tensor_gpu.device,
            )
        if alpha_tensor_gpu is None:
            alpha_tensor_gpu = _cached_alpha_one(a_tensor_gpu.device)

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
                    x_bf16_tensor_gpu,
                    w_gscale_tensor_gpu,
                    row_scale_tensor_gpu,
                ]
            ),
            m,
            cuda_stream_from_int_or_current(stream_int),
        )
        return c_tensor_gpu

    return attach_programs(tensor_api, compiled_kernel)


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


@program_cache
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
    compiled = b12x_compile(
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

    return attach_programs(tensor_api, compiled)


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


@program_cache
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
    compiled = b12x_compile(
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

    return attach_programs(tensor_api, compiled)


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
        tile_k=128,
        expected_m=expected_m,
        generalize_mxfp8_split_k=True,
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
        (expected_m if expected_m is not None else m) == 1,
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
    is_mxfp6: bool = False,
) -> int:
    tile_m, tile_n = mma_tiler_mn
    n_tiles = ((n + tile_n - 1) // tile_n) * l
    if (
        # NOT is_mxfp6_ab_dtype(ab_dtype): the MX-FP6 path rewrites ab_dtype to
        # Float8E4M3FN byte-containers before compiling, so the operand dtype
        # cannot distinguish the two families here.
        is_mxfp6
        and c_dtype == cutlass.BFloat16
        and tile_k == 128
        and mma_tiler_mn == (16, 64)
        and cluster_shape_mn == (1, 1)
        and load_path == "tma"
        and not swap_ab
        and not b_tile_major
        and n_tiles > sm_count
    ):
        # Two resident CTAs avoid a partial tail wave when the decode grid is
        # larger than the SM count. Occupancy does not change MMA ordering.
        return 2
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


@program_cache
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
    plain_fp8: bool = False,
    block_fp8: bool = False,
    target_occupancy_override: Optional[int] = None,
    weight_only: Optional[str] = None,
    alpha_reciprocal: bool = False,
    input_k: Optional[int] = None,
    device_ordinal: Optional[int] = None,
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
            make_ptr(cutlass.Uint8 if weight_only else ab_dtype, b_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
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
        plain_fp8=plain_fp8,
        block_fp8=block_fp8,
        weight_only=weight_only,
        alpha_reciprocal=alpha_reciprocal,
        input_k=input_k,
        target_occupancy=(
            target_occupancy_override
            if target_occupancy_override is not None
            else _dense_gemm_target_occupancy(
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
            )
        ),
    )
    if weight_only is not None and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("A16 dense GEMM must be prewarmed before CUDA graph capture")
    compile_key = launch.compile_key()
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=compile_key,
    )
    compiled_kernel = b12x_compile(
        launch,
        *_make_runtime_pointers(None),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("gemm.dense", 4, compile_key),
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

    return attach_programs(tensor_api, compiled_kernel)


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
    target_occupancy_override: Optional[int],
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
        target_occupancy_override=target_occupancy_override,
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
    "b12x::dense_gemm_launch",
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
    target_occupancy_override: Optional[int],
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
        target_occupancy_override,
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
    target_occupancy_override: Optional[int],
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
    "b12x::dense_gemm_launch_functional",
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
        None,
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
    is_mxfp6: bool = False,
    expected_m: Optional[int] = None,
    k: Optional[int] = None,
) -> Tuple[int, int]:
    coarse_tile = (128, 128)
    if is_mxfp6:
        # Decode and speculative verification fit in one 16-row M tile. A
        # declared expected-M regime hint owns the decision when present.
        plan_m = expected_m if expected_m is not None else m
        if n > 1536 and plan_m <= 16:
            return _FP6_DECODE_TILE
        if n > 1536:
            # The wide-N prefill choice is independent of live M, preserving
            # one kernel per expected-M regime and (N, K) shape.
            return _FP6_PREFILL_TILE
        return coarse_tile
    # The serving WO-B prefill GEMM is [M,4096] x [4096,4096]. DeepGEMM's
    # specialized O-projection dispatch switches it from BM64/BK128 to
    # BM128/BK64 at M>=2048, and the same exact-shape switch wins in b12x.
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
            if (
                expected_m >= 2048
                and (n, k) == (16384, 1024)
                and _use_low_sm_dense_tactics(sm_count)
            ):
                return (64, 128)
            if _use_high_sm_mxfp8_bk64_prefill(
                expected_m, n, k, sm_count
            ):
                return (128, 128)
            if (
                expected_m >= 2048
                and not _use_low_sm_dense_tactics(sm_count)
                and k is not None
                and k >= 2048
                and n <= 8192
            ):
                return (128, 64)
            if expected_m >= 2048 and n >= 16384 and k is not None and k <= 1024:
                return (128, 128)
            if expected_m == 1:
                return (16, 64)
            # The 64-column short-K tile keeps the remainder wave populated on
            # parts whose two-CTA resident grid is smaller than this output.
            if (
                expected_m <= 16
                and (n, k) == (16384, 1024)
                and _use_low_sm_dense_tactics(sm_count)
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
        # N<=1536), so at M>=512 it can leave a large-SM launch CTA-starved --
        # 2x-3.5x slower than a CTA-multiplying tile
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
        if (
            expected_m is not None
            and expected_m >= 2048
            and not _use_low_sm_dense_tactics(sm_count)
            and k is not None
            and k >= 2048
        ):
            return (128, 64)
        if expected_m is not None and expected_m > 128:
            return (64, 128)
        return (64, 64)

    plan_m = expected_m if expected_m is not None else m
    if (
        not _use_low_sm_dense_tactics(sm_count)
        and plan_m <= 128
        and k is not None
        and k >= 4096
        and (n <= 4096 or (plan_m >= 64 and n <= 6144))
    ):
        return (64, 64)
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

    if (
        48 <= plan_m <= 64
        and n >= 4096
        and k is not None
        and k >= 5120
        and _use_low_sm_dense_tactics(sm_count)
    ):
        # Joint tile/BK/load sweeps across Qwen TP=1/2/4/8 and the common
        # corpus found a bounded medium-M window where doubling the N-tile
        # count wins after swapping the logical operands. TMA remains the load
        # path; _select_default_dense_gemm_plan attaches swapped storage to the
        # narrow tile.
        return (64, 32)

    planned_coarse_tiles = ((plan_m + coarse_tile[0] - 1) // coarse_tile[0]) * (
        (n + coarse_tile[1] - 1) // coarse_tile[1]
    )
    if (
        not _use_low_sm_dense_tactics(sm_count)
        and 1024 <= plan_m <= 2048
        and planned_coarse_tiles < max(1, sm_count // 2)
    ):
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
            # Very wide outputs already expose multiple full 128-column waves.
            # A 64-column tile gives the producer/epilogue a smaller working
            # set without relying on extra K splits; keep 128 columns for
            # narrower grids where duplicating A traffic does not pay back.
            if (n + 127) // 128 >= 2 * sm_count:
                return (64, 64)
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


def _use_high_sm_mxfp8_bk64_prefill(
    expected_m: Optional[int],
    n: int,
    k: Optional[int],
    sm_count: int,
) -> bool:
    if (
        expected_m is None
        or expected_m < 2048
        or k is None
        or _use_low_sm_dense_tactics(sm_count)
    ):
        return False
    return (
        (n >= 4096 and k <= 1536)
        or (n == 4096 and 4096 <= k <= 6144)
        or (n >= 8192 and k <= 6144)
    )


def _select_mxfp8_tile_k(
    m: int,
    n: int,
    k: int,
    expected_m: Optional[int],
    sm_count: int,
) -> int:
    plan_m = expected_m if expected_m is not None else m
    if _use_high_sm_mxfp8_bk64_prefill(expected_m, n, k, sm_count):
        return 64
    if (
        expected_m is not None
        and k <= 1024
        and n >= 4096
        and _use_low_sm_dense_tactics(sm_count)
    ):
        medium_prefill_bk64 = (
            1536 <= plan_m <= 2048
            and (n + 127) // 128 <= 2 * sm_count
        )
        return 64 if medium_prefill_bk64 else 128
    if (
        1536 <= plan_m <= 2048
        and n >= 4096
        and k >= 2048
        and _use_low_sm_dense_tactics(sm_count)
    ):
        return 64
    # Keep tile M and K tied for the short-K, wide-output prefill plan.
    if (
        expected_m is not None
        and expected_m >= 2048
        and (n, k) == (16384, 1024)
        and _use_low_sm_dense_tactics(sm_count)
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


def _select_fp4_tile_k(
    m: int,
    n: int,
    k: int,
    expected_m: Optional[int],
    sm_count: int,
    mma_tiler_mn: Tuple[int, int],
) -> int:
    """Select the staged K depth for prequantized NVFP4 GEMM."""
    plan_m = expected_m if expected_m is not None else m
    if plan_m <= 128 and k >= 4096 and k % 256 == 0:
        tile_n = mma_tiler_mn[1]
        n_tiles = (n + tile_n - 1) // tile_n
        if _use_low_sm_dense_tactics(sm_count):
            minimum_tiles = max(1, sm_count // 2)
        elif plan_m >= 48:
            minimum_tiles = (sm_count + 2) // 3
        else:
            minimum_tiles = sm_count + 1
        if tile_n <= 64 or n_tiles >= minimum_tiles:
            return 256
    return 128


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
    is_mxfp6: bool = False,
    block_fp8: bool = False,
    expected_m: Optional[int] = None,
    select_swapped_output_storage: bool = False,
) -> _DenseGemmPlan:
    tile = _select_default_mma_tiler_mn(
        m,
        n,
        sm_count,
        is_mxfp8=is_mxfp8,
        is_mxfp6=is_mxfp6,
        expected_m=expected_m,
        k=k,
    )
    plan_m = expected_m if expected_m is not None else m
    if (
        block_fp8
        and _select_block_fp8_decode_slices(plan_m, n, k, sm_count) > 1
    ):
        tile = (32, 64)
    elif (
        block_fp8
        and not _use_low_sm_dense_tactics(sm_count)
        and plan_m >= 2048
    ):
        tile = (64, 128)
    elif (
        block_fp8
        and not _use_low_sm_dense_tactics(sm_count)
        and 96 <= plan_m <= 128
    ):
        work_tiles = ((plan_m + 63) // 64) * ((n + 127) // 128)
        if (sm_count + 1) // 2 <= work_tiles <= sm_count:
            tile = (64, 128)
    elif (
        is_mxfp8
        and not block_fp8
        and k <= 1024
        and n >= 4096
        and _use_low_sm_dense_tactics(sm_count)
    ):
        n_tiles_128 = (n + 127) // 128
        if plan_m <= 6:
            # Short-K decode is output-grid-bound. A 64-column tile supplies
            # at least one full CTA wave down through TP=8 while preserving
            # the two-CTA occupancy policy for the 16-row MMA tile.
            tile = (16, 64)
        elif 1536 <= plan_m <= 2048 and n_tiles_128 <= 2 * sm_count:
            # Smaller medium-prefill shards benefit from BK64 scale staging;
            # wider grids retain the 64x128/BK128 plan.
            tile = (128, 128)
        elif plan_m > 128:
            tile = (64, 128)
    elif (
        is_mxfp8
        and not block_fp8
        and plan_m <= 6
        and k % (4 * 128) == 0
        and 4096 <= k <= 6144
        and (n + 63) // 64 >= sm_count
        and _use_low_sm_dense_tactics(sm_count)
    ):
        # The four-slice decode kernel uses one 64-column CTA per output tile.
        # Select the jointly tuned 32-row tile only once that grid spans a full
        # SM wave; smaller grids retain the direct 16-row plan.
        tile = (32, 64)
    elif (
        is_mxfp8
        and not block_fp8
        and 1536 <= plan_m <= 2048
        and n >= 4096
        and k >= 2048
        and _use_low_sm_dense_tactics(sm_count)
    ):
        # Across Qwen TP=1/2/4/8 and the N=4096 K-boundary corpus, the
        # 128x128/BK64 plan is the bounded medium-prefill winner. The M bounds
        # exclude measured 1024 and 3072 counterexamples.
        tile = (128, 128)
    elif (
        is_mxfp8
        and not block_fp8
        and 512 <= plan_m <= 3072
        and 1536 < n < 4096
        and _use_low_sm_dense_tactics(sm_count)
    ):
        # A narrow-wide output does not provide enough 128-column tiles to
        # offset repeated B loads from the 64-row prefill default. Doubling M
        # while halving N preserves the output grid and wins throughout the
        # qualified prefill window; M=4096 counterexamples retain 64x128.
        tile = (128, 64)
    if (
        block_fp8
        and 96 <= plan_m <= 128
        and k >= 8192
        and (n + 127) // 128 < sm_count
        and _use_low_sm_dense_tactics(sm_count)
    ):
        # Deep-K medium batches benefit from B reuse before the full prefill
        # regime begins; the narrow output grid remains a full M-expanded wave.
        tile = (128, 128)
    if (
        is_mxfp8
        and not block_fp8
        and expected_m is not None
        and expected_m <= 8
        and k >= 8192
        and (n + 127) // 128 < sm_count
    ):
        # A deep-K, narrow-N decode shape otherwise exposes fewer than one
        # 128-column wave and falls back to split-K. Two 64-column waves keep
        # the same CTA parallelism without partial-output reduction.
        tile = (16, 64)
    elif (
        is_mxfp8
        and not block_fp8
        and expected_m is not None
        and expected_m >= 2048
        and k >= 8192
        and (n + 127) // 128 < sm_count
    ):
        # Deep-K prefill reloads the full weight matrix once per M tile. A
        # 128-row tile halves those reloads while the large M dimension still
        # leaves a deeply oversubscribed grid. Large-SM grids retain 64 output
        # columns to expose enough independent CTAs.
        tile = (
            (128, 128)
            if _use_low_sm_dense_tactics(sm_count)
            else (128, 64)
        )
    if (
        block_fp8
        and expected_m is not None
        and expected_m >= 2048
        and k >= 10240
        and _use_low_sm_dense_tactics(sm_count)
    ):
        # Block-FP8 accumulates and rescales every K128 stage. Deep-K prefill
        # benefits from doubling tile M: it halves the number of CTAs that
        # reload each weight/scaling block without starving the large-M grid.
        tile = (128, 128)
    plan = _DenseGemmPlan(
        mma_tiler_mn=tile,
        load_path="tma",
        swap_ab=(not is_mxfp8 and not is_mxfp6 and tile[1] < 64),
    )
    if not (is_mxfp8 and select_swapped_output_storage):
        return plan

    # Swapping operands reverses the logical MMA tile axes. Transpose the
    # tuned default so expected-M, N, K, and SM-count policy remains intact.
    # A 64x32 tile gives the qualified narrow-output path more independent N
    # tiles than the square default without changing the public output shape.
    swapped_tile = (
        (64, 32) if n < 64 or (n <= 256 and tile == (64, 64)) else (tile[1], tile[0])
    )
    return _DenseGemmPlan(swapped_tile, plan.load_path, True)


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
    _split_k_workspace: Optional[torch.Tensor] = None,
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
        tile_k=128,
        expected_m=expected_m,
        generalize_mxfp8_split_k=True,
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
        split_storage = _dense_split_storage(
            _split_k_workspace, split_k_slices, m, n, source.device
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
        (expected_m if expected_m is not None else m) == 1,
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


def _expand_packed_mxfp6_ab(t: torch.Tensor, num_codes: int) -> torch.Tensor:
    """Expand a 3:4-packed FP6 operand ``(X, packed_k, L)`` to byte-containers.

    Returns an ``(X, num_codes, L)`` uint8 view whose underlying memory is laid out
    K-major (matching ``a_major``/``b_major == "k"``), so the compiled kernel reads
    each FP6 code from one byte. The launch only uses ``data_ptr`` plus the compiled
    ``(X, K, L)`` layout, so the returned view's strides need not be contiguous; the
    contiguous backing buffer is kept alive by the returned view.
    """
    t_lxk = t.permute(2, 0, 1).contiguous()  # (L, X, packed_k), packed_k fastest
    e_lxk = expand_mxfp6_packed_to_bytes(t_lxk, num_codes)  # (L, X, num_codes)
    return e_lxk.permute(1, 2, 0)  # (X, num_codes, L), K stride 1


def _dense_split_storage(workspace, slices, m, n, device):
    if workspace is None:
        return torch.empty((slices, m, n), dtype=torch.float32, device=device)
    if (workspace.dtype != torch.float32 or workspace.device != device
            or not workspace.is_contiguous() or workspace.data_ptr() % 16
            or workspace.numel() < slices * m * n):
        raise ValueError("split-K workspace must be aligned contiguous FP32 storage with sufficient capacity")
    return workspace.view(-1)[:slices * m * n].view(slices, m, n)


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
    a_preexpanded: bool = False,
    b_preexpanded: bool = False,
    b_packed: bool = False,
    a_fmt: Optional[str] = None,
    b_fmt: Optional[str] = None,
    x_bf16: Optional[torch.Tensor] = None,
    w_gscale: Optional[torch.Tensor] = None,
    plain_fp8: bool = False,
    row_scale: Optional[torch.Tensor] = None,
    block_fp8: bool = False,
    _tile_k_override: Optional[int] = None,
    _split_k_slices_override: Optional[int] = None,
    _large_m_unroll_override: Optional[bool] = None,
    _target_occupancy_override: Optional[int] = None,
    _split_k_workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute dense block-scaled GEMM for one expert-major batch stack.

    expected_m: optional regime hint (DeepGEMM-style). When set, the default tile
    is chosen for that representative M instead of being M-independent, giving a
    per-regime-optimal kernel that is still reused across all live M in the regime
    (e.g. expected_m<=128 selects a decode-tuned tile). Ignored when mma_tiler_mn
    is given. Live M stays a runtime arg; only the tile (a compile key) changes.

    MX-FP6 (``ab_dtype`` in ``float6_e3m2fn``/``float6_e2m3fn``) parameters:

    ``b_preexpanded``: when True the RHS is already in 1-byte-per-code FP8
    container layout (the output of ``_expand_packed_mxfp6_ab``) and the
    per-call weight expansion is skipped. Callers hoist the expansion to load
    time so the static weight is unpacked once instead of every token.

    ``a_preexpanded``: same for the LHS — its shape is then ``(M, K, L)`` with
    one code per byte (e.g. the quantizer's ``emit="bytes"`` output), so the
    logical K is read directly from the shape and no expansion runs.

    ``b_packed``: native packed-FP6 streaming. The RHS stays in the 3:4-packed
    wire format ``(N, 3K/4, L)``; the kernel TMA-streams the packed bytes and
    expands to byte-containers in smem. 25% less B HBM traffic than the
    byte-container layout and no expanded copy resident in VRAM. MX-FP6 only;
    mutually exclusive with ``b_preexpanded``.

    ``a_fmt`` / ``b_fmt``: optional per-operand MX sub-formats (``e2m3`` /
    ``e3m2`` / ``e4m3``). When omitted, both operands use the format implied by
    ``ab_dtype``. W6A8 dense passes ``a_fmt="e4m3"`` (activations) and
    ``b_fmt="e2m3"`` (weights) with ``ab_dtype="float6_e2m3fn"`` so the weight
    packing / expansion path stays on the MX-FP6 branch.

    ``x_bf16`` / ``w_gscale``: fused BF16 activation-quant inputs, only used
    when B12X_DENSE_FUSED_QUANT is enabled on an m=1 MX-FP6 launch.

    ``plain_fp8``: emit non-block-scaled E4M3 warp MMA while reusing the
    SM12x dense pipeline. The scalar ``alpha`` carries the combined activation
    and weight dequantization scale.

    ``row_scale``: optional contiguous ``(M,)`` tensor in the C dtype, applied
    per output row in the epilogue. MX-FP6 only.

    ``block_fp8``: accumulate ordinary E4M3 MMA over each K128 block, then
    apply compact FP32 activation ``[M,K/128]`` and weight
    ``[N/128,K/128]`` scales before adding it to the final accumulator.
    """
    a_torch, sfa_torch = lhs
    b_torch, sfb_torch = rhs
    if load_path is not None and load_path not in _DENSE_LOAD_PATHS:
        raise ValueError(
            f"dense_gemm load_path must be one of {_DENSE_LOAD_PATHS}, got {load_path!r}"
        )
    if b_packed and b_preexpanded:
        raise ValueError("b_packed and b_preexpanded are mutually exclusive")

    m, k, l = a_torch.shape
    n, _, _ = b_torch.shape
    if sm_count is None:
        sm_count = get_num_sm(a_torch.device)
    mxfp6_fmt_a: Optional[str] = None
    mxfp6_fmt_b: Optional[str] = None
    if ab_dtype == "float4_e2m1fn":
        is_mxfp8 = False
        is_mxfp6 = False
        k *= 2
        mma_k = 64
        tile_k = 128
    elif ab_dtype == "float8_e4m3fn":
        is_mxfp8 = True
        is_mxfp6 = False
        mma_k = 32
        tile_k = (
            128 if block_fp8 else _select_mxfp8_tile_k(m, n, k, expected_m, sm_count)
        )
    elif ab_dtype in ("float6_e3m2fn", "float6_e2m3fn"):
        is_mxfp8 = False
        is_mxfp6 = True
        if sf_vec_size != 32:
            raise ValueError("MX-FP6 dense_gemm requires sf_vec_size=32")
        if sf_dtype != "float8_e8m0fnu":
            raise ValueError("MX-FP6 dense_gemm requires sf_dtype='float8_e8m0fnu'")
        if not a_preexpanded:
            k = mxfp6_logical_k_from_packed_bytes(k)
        mma_k = 32
        tile_k = mxfp6_tile_k(sf_vec_size)
        weight_fmt = "e3m2" if ab_dtype == "float6_e3m2fn" else "e2m3"
        mxfp6_fmt_b = b_fmt if b_fmt is not None else weight_fmt
        mxfp6_fmt_a = a_fmt if a_fmt is not None else weight_fmt
        for name, fmt in (("a_fmt", mxfp6_fmt_a), ("b_fmt", mxfp6_fmt_b)):
            if fmt not in ("e2m3", "e3m2", "e4m3"):
                raise ValueError(f"unsupported {name}={fmt!r}")
    else:
        raise TypeError(f"dense_gemm unsupported ab_dtype: {ab_dtype}")
    if _tile_k_override is not None:
        if ab_dtype == "float4_e2m1fn":
            valid_tile_k = (128, 256, 512)
            format_name = "NVFP4"
        elif (
            ab_dtype == "float8_e4m3fn"
            and not block_fp8
            and not plain_fp8
            and sf_dtype == "float8_e8m0fnu"
            and sf_vec_size == 32
        ):
            valid_tile_k = (64, 128)
            format_name = "MXFP8"
        else:
            raise ValueError(
                "_tile_k_override is restricted to NVFP4 or MXFP8 autotuning"
            )
        if _tile_k_override not in valid_tile_k or k % _tile_k_override:
            if format_name == "NVFP4":
                requirement = "128, 256, or 512"
            else:
                requirement = "one of (64, 128)"
            raise ValueError(
                f"{format_name} _tile_k_override must be {requirement} and divide "
                f"logical K={k}, got {_tile_k_override}"
            )
        tile_k = _tile_k_override
    if _target_occupancy_override is not None:
        if ab_dtype != "float4_e2m1fn":
            raise ValueError(
                "_target_occupancy_override is restricted to NVFP4 autotuning"
            )
        if _target_occupancy_override not in (1, 2, 3, 4):
            raise ValueError(
                "NVFP4 _target_occupancy_override must be 1, 2, 3, or 4, got "
                f"{_target_occupancy_override}"
            )
        if out is None:
            raise ValueError(
                "NVFP4 _target_occupancy_override requires a caller-owned output"
            )
    if block_fp8:
        plain_fp8 = True
        expected_sfa_shape = (m, k // 128)
        expected_sfb_shape = (n // 128, k // 128)
        if (
            sf_dtype != "float32"
            or sf_vec_size != 128
            or l != 1
            or n % 128 != 0
            or k % 128 != 0
            or sfa_torch.shape != expected_sfa_shape
            or sfb_torch.shape != expected_sfb_shape
            or sfa_torch.dtype != torch.float32
            or sfb_torch.dtype != torch.float32
            or not sfa_torch.is_contiguous()
            or not sfb_torch.is_contiguous()
        ):
            raise ValueError(
                "block_fp8 requires L=1, N/K divisible by 128, "
                "sf_dtype='float32', sf_vec_size=128, and contiguous FP32 "
                f"scales shaped {expected_sfa_shape} and {expected_sfb_shape}"
            )
        if (
            load_path not in (None, "tma")
            or swap_ab not in (None, False)
            or rhs_values_tiled is not None
            or _quantized_c is not None
            or sfb_k_replicated
        ):
            raise ValueError(
                "block_fp8 currently requires unswapped TMA operands without "
                "tiled RHS, quantized output, or replicated weight scales"
            )
        load_path = "tma"
        swap_ab = False
    if plain_fp8 and not is_mxfp8:
        raise ValueError("plain_fp8 requires ab_dtype='float8_e4m3fn'")
    if plain_fp8 and (
        rhs_values_tiled is not None or _quantized_c is not None or sfb_k_replicated
    ):
        raise ValueError(
            "plain_fp8 does not support tiled/quantized output or replicated scales"
        )
    if b_packed:
        if mxfp6_fmt_b is None:
            raise ValueError("b_packed requires an MX-FP6 ab_dtype")
        if b_torch.shape[1] * 4 != k * 3:
            raise ValueError(
                f"b_packed expects (N, 3K/4, L); got packed K bytes "
                f"{b_torch.shape[1]} for logical K {k}"
            )
    if not is_mxfp6 and (
        a_preexpanded
        or b_preexpanded
        or a_fmt is not None
        or b_fmt is not None
        or x_bf16 is not None
        or w_gscale is not None
    ):
        raise ValueError(
            "a_preexpanded/b_preexpanded/a_fmt/b_fmt/x_bf16/w_gscale are only "
            f"supported with an MX-FP6 ab_dtype, got ab_dtype={ab_dtype!r}"
        )
    if (x_bf16 is None) != (w_gscale is None):
        raise ValueError("x_bf16 and w_gscale must be provided together")
    if x_bf16 is not None and w_gscale is not None:
        if not is_mxfp6:
            raise ValueError("fused quantization inputs require an MX-FP6 ab_dtype")
        if (
            x_bf16.shape != (m, k)
            or x_bf16.dtype != torch.bfloat16
            or not x_bf16.is_contiguous()
            or x_bf16.device != a_torch.device
            or x_bf16.data_ptr() % 16 != 0
        ):
            raise ValueError(
                "x_bf16 must be a contiguous, 16-byte-aligned BF16 tensor "
                f"with shape {(m, k)} on {a_torch.device}; got shape "
                f"{tuple(x_bf16.shape)}, dtype {x_bf16.dtype}, device "
                f"{x_bf16.device}, data_ptr alignment {x_bf16.data_ptr() % 16}"
            )
        if (
            w_gscale.shape != (1,)
            or w_gscale.dtype != torch.float32
            or not w_gscale.is_contiguous()
            or w_gscale.device != a_torch.device
            or w_gscale.data_ptr() % 16 != 0
        ):
            raise ValueError(
                "w_gscale must be a contiguous, 16-byte-aligned float32 tensor "
                f"with shape (1,) on {a_torch.device}; got shape "
                f"{tuple(w_gscale.shape)}, dtype {w_gscale.dtype}, device "
                f"{w_gscale.device}, data_ptr alignment {w_gscale.data_ptr() % 16}"
            )

    ab_cutlass_dtype = get_cutlass_dtype(ab_dtype)
    if mxfp6_fmt_a is not None:
        # Stage 1: carry MX codes in Float8E4M3FN byte-containers so the kernel
        # uses cutlass's native 8-bit smem/TMA/ldmatrix path (cutlass cannot build
        # a 6-bit smem layout). Expand the 3:4-packed inputs to one code per byte
        # at this load boundary; the on-disk/wire format stays packed. E4M3
        # activations are already one-byte-per-code (a_preexpanded required).
        ab_cutlass_dtype = cutlass.Float8E4M3FN
        if not a_preexpanded:
            if mxfp6_fmt_a == "e4m3":
                raise ValueError("e4m3 activations require a_preexpanded=True")
            a_torch = _expand_packed_mxfp6_ab(a_torch, k)
        if not (b_preexpanded or b_packed):
            b_torch = _expand_packed_mxfp6_ab(b_torch, k)
    c_cutlass_dtype = get_cutlass_dtype(c_dtype)
    c_row_stride_bytes = n * c_cutlass_dtype.width // 8
    output_requires_swapped_store = (m > 1 or l > 1) and c_row_stride_bytes % 16 != 0
    use_default_mma_tiler = mma_tiler_mn is None
    use_default_output_storage = mma_tiler_mn is None and swap_ab is None
    if mma_tiler_mn is None or load_path is None or swap_ab is None:
        default_plan = _select_default_dense_gemm_plan(
            m,
            n,
            k,
            sm_count,
            is_mxfp8=is_mxfp8,
            is_mxfp6=is_mxfp6,
            block_fp8=block_fp8,
            expected_m=expected_m,
            select_swapped_output_storage=(
                use_default_output_storage
                and l == 1
                and (n < 64 or output_requires_swapped_store)
            ),
        )
        if mma_tiler_mn is None:
            mma_tiler_mn = default_plan.mma_tiler_mn
        if load_path is None:
            load_path = default_plan.load_path
        if swap_ab is None:
            if use_default_mma_tiler:
                swap_ab = default_plan.swap_ab
            else:
                swap_ab = default_plan.swap_ab if mma_tiler_mn[1] < 64 else False
    assert load_path is not None
    assert swap_ab is not None
    if ab_dtype == "float4_e2m1fn" and _tile_k_override is None:
        tile_k = _select_fp4_tile_k(
            m,
            n,
            k,
            expected_m,
            sm_count,
            mma_tiler_mn,
        )
    if l > 1 and swap_ab:
        raise ValueError(
            "swapped dense_gemm output storage supports L=1 only; pad N for "
            f"grouped output, got L={l}, N={n}"
        )
    if output_requires_swapped_store and not swap_ab:
        remedy = (
            "pad N; swapped output storage is unsupported when L > 1"
            if l > 1
            else "use a supported swapped plan or pad N"
        )
        raise ValueError(
            "the unswapped dense_gemm TMA epilogue requires a 16-byte-aligned "
            f"C row stride, but N={n} and c_dtype={c_dtype!r} produce "
            f"{c_row_stride_bytes} bytes; {remedy}"
        )
    if is_mxfp8 and _tile_k_override is None and mma_tiler_mn[0] != 128:
        # A caller's explicit M tile can differ from the default tile that
        # justified automatic BK64 selection. Keep its native BK128 path;
        # an explicitly forced incompatible BK64 still fails validation below.
        tile_k = 128
    if is_mxfp8 and swap_ab:
        # BK64 packed-scale staging requires the weight operand to remain in
        # the unswapped 128-row slot. Swapped storage therefore uses BK128.
        tile_k = 128
    if is_mxfp6:
        # Only the unswapped single-slice TMA mainloop is wired for the FP6
        # byte-container path; fail loudly instead of silently miscomputing.
        if swap_ab:
            raise ValueError("MX-FP6 dense_gemm does not support swap_ab")
        if load_path != "tma":
            raise ValueError(
                f"MX-FP6 dense_gemm only supports load_path='tma', got {load_path!r}"
            )
        if _quantized_c is not None:
            raise ValueError("MX-FP6 dense_gemm does not support quantized C output")
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
        tile_k=tile_k,
        expected_m=expected_m,
        generalize_mxfp8_split_k=(is_mxfp8 and not block_fp8 and not plain_fp8),
        generalize_block_fp8_split_k=block_fp8,
    )
    if _split_k_slices_override is not None:
        mxfp8_autotune = (
            is_mxfp8
            and not block_fp8
            and not plain_fp8
            and sf_vec_size == 32
            and sf_dtype == "float8_e8m0fnu"
        )
        block_fp8_autotune = (
            is_mxfp8
            and block_fp8
            and sf_vec_size == 128
            and sf_dtype == "float32"
        )
        if not (mxfp8_autotune or block_fp8_autotune):
            raise ValueError(
                "_split_k_slices_override is restricted to MXFP8 or block-FP8 "
                "autotuning"
            )
        if _split_k_slices_override not in (1, 2, 4):
            raise ValueError(
                "FP8 _split_k_slices_override must be 1, 2, or 4, got "
                f"{_split_k_slices_override}"
            )
        if _split_k_slices_override > 1:
            if m > 8 or m > mma_tiler_mn[0] or l != 1 or swap_ab:
                raise ValueError(
                    "split-K FP8 autotuning requires M<=8 within one M tile, "
                    f"L=1, and an unswapped plan; got M={m}, L={l}, "
                    f"tile={mma_tiler_mn}, swap_ab={swap_ab}"
                )
            if k % (tile_k * _split_k_slices_override):
                raise ValueError(
                    "split-K FP8 autotuning requires the staged K-tile count "
                    f"to divide evenly across slices; got K={k}, BK={tile_k}, "
                    f"slices={_split_k_slices_override}"
                )
            if (
                _split_k_slices_override > 2
                and not _B12X_DENSE_SPLITK_TURBO
            ):
                raise ValueError(
                    "four-way split-K requires atomic-BF16 or an explicit "
                    "caller-owned FP32 reduction"
                )
        policy = _DenseGemmPolicy(
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            use_m1_non_tma=policy.use_m1_non_tma,
            split_k_slices=_split_k_slices_override,
            split_k_atomic_bf16=(
                _split_k_slices_override > 1 and _B12X_DENSE_SPLITK_TURBO
            ),
            large_m_unroll=policy.large_m_unroll,
        )
    if _large_m_unroll_override is not None:
        if not is_mxfp8 or is_mxfp6 or l != 1:
            raise ValueError(
                "_large_m_unroll_override is restricted to FP8 autotuning with L=1"
            )
        if not isinstance(_large_m_unroll_override, bool):
            raise ValueError("_large_m_unroll_override must be a bool")
        policy = _DenseGemmPolicy(
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            use_m1_non_tma=policy.use_m1_non_tma,
            split_k_slices=policy.split_k_slices,
            split_k_atomic_bf16=policy.split_k_atomic_bf16,
            large_m_unroll=_large_m_unroll_override,
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
    if is_mxfp6 and (policy.split_k_slices != 1 or policy.large_m_unroll):
        # The policy helper sees the FP8 byte-container dtype and may pick
        # MXFP8 tactics that are not wired for MX-FP6.
        policy = _DenseGemmPolicy(
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            use_m1_non_tma=policy.use_m1_non_tma,
            split_k_slices=1,
            split_k_atomic_bf16=False,
            large_m_unroll=False,
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
    if row_scale is not None:
        if not is_mxfp6:
            raise ValueError("row_scale is only wired for the MX-FP6 path")
        if (
            row_scale.dim() != 1
            or row_scale.shape[0] != m
            or not row_scale.is_contiguous()
            or row_scale.dtype != cutlass_to_torch_dtype(c_cutlass_dtype)
            # The epilogue reads it through a raw device pointer, so a
            # host or wrong-device tensor is an illegal access, not an error.
            or row_scale.device != a_torch.device
            or row_scale.data_ptr() % 16 != 0
        ):
            raise ValueError(
                "row_scale must be a contiguous 1-D tensor of shape (M,) in the "
                f"C dtype on the operand device; got shape "
                f"{tuple(row_scale.shape)} dtype {row_scale.dtype} device "
                f"{row_scale.device} for M={m}, C dtype "
                f"{cutlass_to_torch_dtype(c_cutlass_dtype)}, operand device "
                f"{a_torch.device}"
            )
    if is_mxfp6:
        # Dedicated MX-FP6 launch path (bypasses the torch custom ops, like
        # the quantized-C path): the byte-container operands plus the
        # x_bf16/w_gscale fused-quant tensors flow straight into the compiled
        # launch, and the FP6 fmt/packed axes key the compile cache.
        compiled_mxfp6 = _get_compiled_dense_gemm_mxfp6(
            n=n,
            k=k,
            l=l,
            c_l=l,
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
            mxfp6_fmt_a=mxfp6_fmt_a,
            mxfp6_fmt_b=mxfp6_fmt_b,
            b_packed=b_packed,
            a_preexpanded=a_preexpanded,
            b_preexpanded=b_preexpanded,
            alpha_is_one=alpha_is_one,
            row_scale=row_scale is not None,
            # Fused quantization changes code generation and is part of the
            # compiled-kernel key.
            fused_quant=_DENSE_FUSED_QUANT and x_bf16 is not None,
        )
        if out is None:
            out = _empty_dense_gemm_output(
                m,
                n,
                l,
                dtype=cutlass_to_torch_dtype(c_cutlass_dtype),
                device=a_torch.device,
            )
        return compiled_mxfp6(
            a_tensor_gpu=a_torch,
            b_tensor_gpu=b_torch,
            sfa_tensor_gpu=sfa_torch,
            sfb_tensor_gpu=sfb_torch,
            c_tensor_gpu=out,
            alpha_tensor_gpu=alpha,
            stream_int=stream_int,
            x_bf16_tensor_gpu=x_bf16,
            w_gscale_tensor_gpu=w_gscale,
            row_scale_tensor_gpu=row_scale,
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
    if out is None and not plain_fp8:
        # No caller-owned output buffer: functional launch (allocate + return
        # inside the opaque op). The compile graph then carries no
        # auto_functionalized dense node mutating a (possibly strided) caller
        # view -- which inductor's decompose pass cannot remove. No is_compiling;
        # purely caller-intent, behaviorally identical to the eager out=None path.
        return torch.ops.b12x.dense_gemm_launch_functional(
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
            split_storage = _dense_split_storage(
                _split_k_workspace, split_k_slices, m, n, a_torch.device
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

    t0 = time.perf_counter() if _B12X_TIMING else 0.0
    cache_before = _get_compiled_dense_gemm.cache_info() if _B12X_TIMING else None
    t_compiled = t0
    kernel_c_dtype_name = (
        "float32" if split_k_output and not split_k_atomic_bf16 else c_dtype
    )
    c_tensor_gpu = (
        out if split_k_atomic_bf16 else split_scratch if split_k_output else out
    )
    assert c_tensor_gpu is not None
    if plain_fp8:
        compiled_plain_fp8 = _get_compiled_dense_gemm(
            n=n,
            k=k,
            l=l,
            c_l=kernel_c_l,
            a_major="k",
            b_major="k",
            c_major="n",
            ab_dtype=ab_cutlass_dtype,
            sf_dtype=get_cutlass_dtype(sf_dtype),
            c_dtype=get_cutlass_dtype(kernel_c_dtype_name),
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
            sfb_k_reuse=False,
            b_tile_major=False,
            alpha_is_one=alpha_is_one,
            plain_fp8=True,
            block_fp8=block_fp8,
        )
        compiled_plain_fp8(
            a_tensor_gpu=a_torch,
            b_tensor_gpu=b_launch_torch,
            sfa_tensor_gpu=sfa_torch,
            sfb_tensor_gpu=sfb_torch,
            c_tensor_gpu=c_tensor_gpu,
            alpha_tensor_gpu=alpha,
            stream_int=stream_int,
        )
    else:
        torch.ops.b12x.dense_gemm_launch(
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
            _target_occupancy_override,
            stream_int,
        )
    result = out
    if split_k_output and not split_k_atomic_bf16:
        assert split_scratch is not None
        assert out is not None
        _reduce_split_k2_bf16(split_scratch, out, m=m, n=n)
        result = out
    if _B12X_TIMING:
        t_launch = time.perf_counter()
        cache_after = _get_compiled_dense_gemm.cache_info()
        assert cache_before is not None
        compile_ms = (t_compiled - t0) * 1000.0
        launch_ms = (t_launch - t_compiled) * 1000.0
        total_ms = (t_launch - t0) * 1000.0
        if total_ms >= _B12X_TIMING_THRESHOLD_MS:
            logger.warning(
                "b12x_dense_gemm timing m=%d n=%d k=%d l=%d ab=%s sf=%s c=%s "
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


@dataclass(frozen=True)
class _DenseLowering:
    m: int
    n: int
    k: int
    l: int
    a_storage_k: int
    b_storage_k: int
    ab_dtype: str
    sf_dtype: str
    c_dtype: str
    alpha_dtype: str
    sf_vec_size: int
    sm_count: int
    mma_k: int
    tile_k: int
    mma_tiler_mn: tuple[int, int]
    cluster_shape_mn: tuple[int, int]
    policy: _DenseGemmPolicy
    load_path: str
    swap_ab: bool
    expected_m: int | None
    kernel_c_l: int
    alpha_is_one: bool
    is_mxfp6: bool
    mxfp6_fmt_a: str | None
    mxfp6_fmt_b: str | None
    a_preexpanded: bool
    b_preexpanded: bool
    b_packed: bool
    plain_fp8: bool
    block_fp8: bool
    sfb_k_reuse: bool
    b_tile_major: bool
    quantize_c: bool
    fused_quant: bool
    row_scale: bool
    output_provided: bool
    target_occupancy_override: int | None
    direct_sfa_live16: bool
    direct_m1_wo_a_inputs: bool

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["policy"] = _DenseGemmPolicy(**values["policy"])
        for name in ("mma_tiler_mn", "cluster_shape_mn"):
            values[name] = tuple(values[name])
        return cls(**values)

def _lower_dense_gemm(
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
    a_preexpanded: bool = False,
    b_preexpanded: bool = False,
    b_packed: bool = False,
    a_fmt: Optional[str] = None,
    b_fmt: Optional[str] = None,
    x_bf16: Optional[torch.Tensor] = None,
    w_gscale: Optional[torch.Tensor] = None,
    plain_fp8: bool = False,
    row_scale: Optional[torch.Tensor] = None,
    block_fp8: bool = False,
    _tile_k_override: Optional[int] = None,
    _split_k_slices_override: Optional[int] = None,
    _large_m_unroll_override: Optional[bool] = None,
    _target_occupancy_override: Optional[int] = None,
    _split_k_workspace: Optional[torch.Tensor] = None,
) -> _DenseLowering:
    """Lower the existing joint host rules without allocating or launching."""
    a_torch, sfa_torch = lhs
    b_torch, sfb_torch = rhs
    if load_path is not None and load_path not in _DENSE_LOAD_PATHS:
        raise ValueError(
            f"dense_gemm load_path must be one of {_DENSE_LOAD_PATHS}, got {load_path!r}"
        )
    if b_packed and b_preexpanded:
        raise ValueError("b_packed and b_preexpanded are mutually exclusive")

    m, k, l = a_torch.shape
    n, _, _ = b_torch.shape
    if sm_count is None:
        sm_count = get_num_sm(a_torch.device)
    mxfp6_fmt_a: Optional[str] = None
    mxfp6_fmt_b: Optional[str] = None
    if ab_dtype == "float4_e2m1fn":
        is_mxfp8 = False
        is_mxfp6 = False
        k *= 2
        mma_k = 64
        tile_k = 128
    elif ab_dtype == "float8_e4m3fn":
        is_mxfp8 = True
        is_mxfp6 = False
        mma_k = 32
        tile_k = (
            128 if block_fp8 else _select_mxfp8_tile_k(m, n, k, expected_m, sm_count)
        )
    elif ab_dtype in ("float6_e3m2fn", "float6_e2m3fn"):
        is_mxfp8 = False
        is_mxfp6 = True
        if sf_vec_size != 32:
            raise ValueError("MX-FP6 dense_gemm requires sf_vec_size=32")
        if sf_dtype != "float8_e8m0fnu":
            raise ValueError("MX-FP6 dense_gemm requires sf_dtype='float8_e8m0fnu'")
        if not a_preexpanded:
            k = mxfp6_logical_k_from_packed_bytes(k)
        mma_k = 32
        tile_k = mxfp6_tile_k(sf_vec_size)
        weight_fmt = "e3m2" if ab_dtype == "float6_e3m2fn" else "e2m3"
        mxfp6_fmt_b = b_fmt if b_fmt is not None else weight_fmt
        mxfp6_fmt_a = a_fmt if a_fmt is not None else weight_fmt
        for name, fmt in (("a_fmt", mxfp6_fmt_a), ("b_fmt", mxfp6_fmt_b)):
            if fmt not in ("e2m3", "e3m2", "e4m3"):
                raise ValueError(f"unsupported {name}={fmt!r}")
    else:
        raise TypeError(f"dense_gemm unsupported ab_dtype: {ab_dtype}")
    if _tile_k_override is not None:
        if ab_dtype == "float4_e2m1fn":
            valid_tile_k = (128, 256, 512)
            format_name = "NVFP4"
        elif (
            ab_dtype == "float8_e4m3fn"
            and not block_fp8
            and not plain_fp8
            and sf_dtype == "float8_e8m0fnu"
            and sf_vec_size == 32
        ):
            valid_tile_k = (64, 128)
            format_name = "MXFP8"
        else:
            raise ValueError(
                "_tile_k_override is restricted to NVFP4 or MXFP8 autotuning"
            )
        if _tile_k_override not in valid_tile_k or k % _tile_k_override:
            if format_name == "NVFP4":
                requirement = "128, 256, or 512"
            else:
                requirement = "one of (64, 128)"
            raise ValueError(
                f"{format_name} _tile_k_override must be {requirement} and divide "
                f"logical K={k}, got {_tile_k_override}"
            )
        tile_k = _tile_k_override
    if _target_occupancy_override is not None:
        if ab_dtype != "float4_e2m1fn":
            raise ValueError(
                "_target_occupancy_override is restricted to NVFP4 autotuning"
            )
        if _target_occupancy_override not in (1, 2, 3, 4):
            raise ValueError(
                "NVFP4 _target_occupancy_override must be 1, 2, 3, or 4, got "
                f"{_target_occupancy_override}"
            )
        if out is None:
            raise ValueError(
                "NVFP4 _target_occupancy_override requires a caller-owned output"
            )
    if block_fp8:
        plain_fp8 = True
        expected_sfa_shape = (m, k // 128)
        expected_sfb_shape = (n // 128, k // 128)
        if (
            sf_dtype != "float32"
            or sf_vec_size != 128
            or l != 1
            or n % 128 != 0
            or k % 128 != 0
            or sfa_torch.shape != expected_sfa_shape
            or sfb_torch.shape != expected_sfb_shape
            or sfa_torch.dtype != torch.float32
            or sfb_torch.dtype != torch.float32
            or not sfa_torch.is_contiguous()
            or not sfb_torch.is_contiguous()
        ):
            raise ValueError(
                "block_fp8 requires L=1, N/K divisible by 128, "
                "sf_dtype='float32', sf_vec_size=128, and contiguous FP32 "
                f"scales shaped {expected_sfa_shape} and {expected_sfb_shape}"
            )
        if (
            load_path not in (None, "tma")
            or swap_ab not in (None, False)
            or rhs_values_tiled is not None
            or _quantized_c is not None
            or sfb_k_replicated
        ):
            raise ValueError(
                "block_fp8 currently requires unswapped TMA operands without "
                "tiled RHS, quantized output, or replicated weight scales"
            )
        load_path = "tma"
        swap_ab = False
    if plain_fp8 and not is_mxfp8:
        raise ValueError("plain_fp8 requires ab_dtype='float8_e4m3fn'")
    if plain_fp8 and (
        rhs_values_tiled is not None or _quantized_c is not None or sfb_k_replicated
    ):
        raise ValueError(
            "plain_fp8 does not support tiled/quantized output or replicated scales"
        )
    if b_packed:
        if mxfp6_fmt_b is None:
            raise ValueError("b_packed requires an MX-FP6 ab_dtype")
        if b_torch.shape[1] * 4 != k * 3:
            raise ValueError(
                f"b_packed expects (N, 3K/4, L); got packed K bytes "
                f"{b_torch.shape[1]} for logical K {k}"
            )
    if not is_mxfp6 and (
        a_preexpanded
        or b_preexpanded
        or a_fmt is not None
        or b_fmt is not None
        or x_bf16 is not None
        or w_gscale is not None
    ):
        raise ValueError(
            "a_preexpanded/b_preexpanded/a_fmt/b_fmt/x_bf16/w_gscale are only "
            f"supported with an MX-FP6 ab_dtype, got ab_dtype={ab_dtype!r}"
        )
    if (x_bf16 is None) != (w_gscale is None):
        raise ValueError("x_bf16 and w_gscale must be provided together")
    if x_bf16 is not None and w_gscale is not None:
        if not is_mxfp6:
            raise ValueError("fused quantization inputs require an MX-FP6 ab_dtype")
        if (
            x_bf16.shape != (m, k)
            or x_bf16.dtype != torch.bfloat16
            or not x_bf16.is_contiguous()
            or x_bf16.device != a_torch.device
            or x_bf16.data_ptr() % 16 != 0
        ):
            raise ValueError(
                "x_bf16 must be a contiguous, 16-byte-aligned BF16 tensor "
                f"with shape {(m, k)} on {a_torch.device}; got shape "
                f"{tuple(x_bf16.shape)}, dtype {x_bf16.dtype}, device "
                f"{x_bf16.device}, data_ptr alignment {x_bf16.data_ptr() % 16}"
            )
        if (
            w_gscale.shape != (1,)
            or w_gscale.dtype != torch.float32
            or not w_gscale.is_contiguous()
            or w_gscale.device != a_torch.device
            or w_gscale.data_ptr() % 16 != 0
        ):
            raise ValueError(
                "w_gscale must be a contiguous, 16-byte-aligned float32 tensor "
                f"with shape (1,) on {a_torch.device}; got shape "
                f"{tuple(w_gscale.shape)}, dtype {w_gscale.dtype}, device "
                f"{w_gscale.device}, data_ptr alignment {w_gscale.data_ptr() % 16}"
            )

    ab_cutlass_dtype = get_cutlass_dtype(ab_dtype)
    if mxfp6_fmt_a is not None:
        # Stage 1: carry MX codes in Float8E4M3FN byte-containers so the kernel
        # uses cutlass's native 8-bit smem/TMA/ldmatrix path (cutlass cannot build
        # a 6-bit smem layout). Expand the 3:4-packed inputs to one code per byte
        # at this load boundary; the on-disk/wire format stays packed. E4M3
        # activations are already one-byte-per-code (a_preexpanded required).
        ab_cutlass_dtype = cutlass.Float8E4M3FN
        if not a_preexpanded:
            if mxfp6_fmt_a == "e4m3":
                raise ValueError("e4m3 activations require a_preexpanded=True")
    c_cutlass_dtype = get_cutlass_dtype(c_dtype)
    c_row_stride_bytes = n * c_cutlass_dtype.width // 8
    output_requires_swapped_store = (m > 1 or l > 1) and c_row_stride_bytes % 16 != 0
    use_default_mma_tiler = mma_tiler_mn is None
    use_default_output_storage = mma_tiler_mn is None and swap_ab is None
    if mma_tiler_mn is None or load_path is None or swap_ab is None:
        default_plan = _select_default_dense_gemm_plan(
            m,
            n,
            k,
            sm_count,
            is_mxfp8=is_mxfp8,
            is_mxfp6=is_mxfp6,
            block_fp8=block_fp8,
            expected_m=expected_m,
            select_swapped_output_storage=(
                use_default_output_storage
                and l == 1
                and (n < 64 or output_requires_swapped_store)
            ),
        )
        if mma_tiler_mn is None:
            mma_tiler_mn = default_plan.mma_tiler_mn
        if load_path is None:
            load_path = default_plan.load_path
        if swap_ab is None:
            if use_default_mma_tiler:
                swap_ab = default_plan.swap_ab
            else:
                swap_ab = default_plan.swap_ab if mma_tiler_mn[1] < 64 else False
    assert load_path is not None
    assert swap_ab is not None
    if ab_dtype == "float4_e2m1fn" and _tile_k_override is None:
        tile_k = _select_fp4_tile_k(
            m,
            n,
            k,
            expected_m,
            sm_count,
            mma_tiler_mn,
        )
    if l > 1 and swap_ab:
        raise ValueError(
            "swapped dense_gemm output storage supports L=1 only; pad N for "
            f"grouped output, got L={l}, N={n}"
        )
    if output_requires_swapped_store and not swap_ab:
        remedy = (
            "pad N; swapped output storage is unsupported when L > 1"
            if l > 1
            else "use a supported swapped plan or pad N"
        )
        raise ValueError(
            "the unswapped dense_gemm TMA epilogue requires a 16-byte-aligned "
            f"C row stride, but N={n} and c_dtype={c_dtype!r} produce "
            f"{c_row_stride_bytes} bytes; {remedy}"
        )
    if is_mxfp8 and _tile_k_override is None and mma_tiler_mn[0] != 128:
        tile_k = 128
    if is_mxfp8 and swap_ab:
        # BK64 packed-scale staging requires the weight operand to remain in
        # the unswapped 128-row slot. Swapped storage therefore uses BK128.
        tile_k = 128
    if is_mxfp6:
        # Only the unswapped single-slice TMA mainloop is wired for the FP6
        # byte-container path; fail loudly instead of silently miscomputing.
        if swap_ab:
            raise ValueError("MX-FP6 dense_gemm does not support swap_ab")
        if load_path != "tma":
            raise ValueError(
                f"MX-FP6 dense_gemm only supports load_path='tma', got {load_path!r}"
            )
        if _quantized_c is not None:
            raise ValueError("MX-FP6 dense_gemm does not support quantized C output")
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
        tile_k=tile_k,
        expected_m=expected_m,
        generalize_mxfp8_split_k=(is_mxfp8 and not block_fp8 and not plain_fp8),
        generalize_block_fp8_split_k=block_fp8,
    )
    if _split_k_slices_override is not None:
        mxfp8_autotune = (
            is_mxfp8
            and not block_fp8
            and not plain_fp8
            and sf_vec_size == 32
            and sf_dtype == "float8_e8m0fnu"
        )
        block_fp8_autotune = (
            is_mxfp8
            and block_fp8
            and sf_vec_size == 128
            and sf_dtype == "float32"
        )
        if not (mxfp8_autotune or block_fp8_autotune):
            raise ValueError(
                "_split_k_slices_override is restricted to MXFP8 or block-FP8 "
                "autotuning"
            )
        if _split_k_slices_override not in (1, 2, 4):
            raise ValueError(
                "FP8 _split_k_slices_override must be 1, 2, or 4, got "
                f"{_split_k_slices_override}"
            )
        if _split_k_slices_override > 1:
            if m > 8 or m > mma_tiler_mn[0] or l != 1 or swap_ab:
                raise ValueError(
                    "split-K FP8 autotuning requires M<=8 within one M tile, "
                    f"L=1, and an unswapped plan; got M={m}, L={l}, "
                    f"tile={mma_tiler_mn}, swap_ab={swap_ab}"
                )
            if k % (tile_k * _split_k_slices_override):
                raise ValueError(
                    "split-K FP8 autotuning requires the staged K-tile count "
                    f"to divide evenly across slices; got K={k}, BK={tile_k}, "
                    f"slices={_split_k_slices_override}"
                )
            if _split_k_slices_override > 2 and not _B12X_DENSE_SPLITK_TURBO:
                raise ValueError(
                    "four-way split-K requires the atomic-BF16 reduction path"
                )
        policy = _DenseGemmPolicy(
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            use_m1_non_tma=policy.use_m1_non_tma,
            split_k_slices=_split_k_slices_override,
            split_k_atomic_bf16=(
                _split_k_slices_override > 1 and _B12X_DENSE_SPLITK_TURBO
            ),
            large_m_unroll=policy.large_m_unroll,
        )
    if _large_m_unroll_override is not None:
        if not is_mxfp8 or is_mxfp6 or l != 1:
            raise ValueError(
                "_large_m_unroll_override is restricted to FP8 autotuning with L=1"
            )
        if not isinstance(_large_m_unroll_override, bool):
            raise ValueError("_large_m_unroll_override must be a bool")
        policy = _DenseGemmPolicy(
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            use_m1_non_tma=policy.use_m1_non_tma,
            split_k_slices=policy.split_k_slices,
            split_k_atomic_bf16=policy.split_k_atomic_bf16,
            large_m_unroll=_large_m_unroll_override,
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
    if is_mxfp6 and (policy.split_k_slices != 1 or policy.large_m_unroll):
        # The policy helper sees the FP8 byte-container dtype and may pick
        # MXFP8 tactics that are not wired for MX-FP6.
        policy = _DenseGemmPolicy(
            single_work_tile_per_cta=policy.single_work_tile_per_cta,
            direct_one_m_tile_scheduler=policy.direct_one_m_tile_scheduler,
            use_m1_non_tma=policy.use_m1_non_tma,
            split_k_slices=1,
            split_k_atomic_bf16=False,
            large_m_unroll=False,
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

    common = dict(
        m=m, n=n, k=k, l=l, sf_vec_size=sf_vec_size, tile_k=tile_k,
        mma_tiler_mn=mma_tiler_mn, load_path=load_path, swap_ab=swap_ab,
        b_tile_major=rhs_values_tiled is not None, sfb_k_reuse=sfb_k_reuse,
        is_mxfp8=is_mxfp8,
    )
    return _DenseLowering(
        m=m, n=n, k=k, l=l, a_storage_k=lhs[0].shape[1], b_storage_k=rhs[0].shape[1],
        ab_dtype=ab_dtype, sf_dtype=sf_dtype, c_dtype=c_dtype, alpha_dtype=alpha_dtype,
        sf_vec_size=sf_vec_size, sm_count=sm_count, mma_k=mma_k, tile_k=tile_k,
        mma_tiler_mn=tuple(mma_tiler_mn), cluster_shape_mn=tuple(cluster_shape_mn),
        policy=policy, load_path=load_path, swap_ab=swap_ab, expected_m=expected_m,
        kernel_c_l=kernel_c_l, alpha_is_one=alpha_is_one, is_mxfp6=is_mxfp6,
        mxfp6_fmt_a=mxfp6_fmt_a, mxfp6_fmt_b=mxfp6_fmt_b,
        a_preexpanded=a_preexpanded, b_preexpanded=b_preexpanded, b_packed=b_packed,
        plain_fp8=plain_fp8, block_fp8=block_fp8, sfb_k_reuse=sfb_k_reuse,
        b_tile_major=rhs_values_tiled is not None, quantize_c=_quantized_c is not None,
        fused_quant=_DENSE_FUSED_QUANT and x_bf16 is not None, row_scale=row_scale is not None,
        output_provided=out is not None, target_occupancy_override=_target_occupancy_override,
        direct_sfa_live16=_use_direct_sfa_live16(**common, alpha_is_one=alpha_is_one),
        direct_m1_wo_a_inputs=_use_direct_m1_wo_a_inputs(**common),
    )


@program_cache(scope="preparation")
def _compile_dense_lowering(payload, device_ordinal):
    """Compile exact production launchers from metadata, without tensor storage."""
    p = _DenseLowering.from_dict(payload)
    split = p.policy.split_k_slices > 1
    atomic = split and p.policy.split_k_atomic_bf16
    output_type = "float32" if split and not atomic else p.c_dtype
    common = dict(
        n=p.n, k=p.k, l=p.l, c_l=p.kernel_c_l,
        a_major="k", b_major="k", c_major="n",
        ab_dtype=cutlass.Float8E4M3FN if p.is_mxfp6 else get_cutlass_dtype(p.ab_dtype),
        sf_dtype=get_cutlass_dtype(p.sf_dtype), c_dtype=get_cutlass_dtype(output_type),
        alpha_dtype=get_cutlass_dtype(p.alpha_dtype), sf_vec_size=p.sf_vec_size,
        mma_k=p.mma_k, tile_k=p.tile_k, mma_tiler_mn=p.mma_tiler_mn,
        cluster_shape_mn=p.cluster_shape_mn, policy=p.policy, sm_count=p.sm_count,
        sm_version="sm_120", alpha_is_one=p.alpha_is_one,
    )
    with torch.cuda.device(device_ordinal):
        if p.is_mxfp6:
            gemm = _get_compiled_dense_gemm_mxfp6(
                **common, mxfp6_fmt_a=p.mxfp6_fmt_a, mxfp6_fmt_b=p.mxfp6_fmt_b,
                b_packed=p.b_packed, a_preexpanded=p.a_preexpanded,
                b_preexpanded=p.b_preexpanded, row_scale=p.row_scale,
                fused_quant=p.fused_quant,
            )
        else:
            gemm = _get_compiled_dense_gemm(
                **common, load_path=p.load_path, swap_ab=p.swap_ab,
                sfb_k_reuse=False if p.plain_fp8 else p.sfb_k_reuse,
                b_tile_major=False if p.plain_fp8 else p.b_tile_major,
                quantize_c=p.quantize_c, plain_fp8=p.plain_fp8, block_fp8=p.block_fp8,
                direct_sfa_live16=False if p.plain_fp8 else p.direct_sfa_live16,
                direct_m1_wo_a_inputs=False if p.plain_fp8 or p.quantize_c else p.direct_m1_wo_a_inputs,
                target_occupancy_override=p.target_occupancy_override,
            )
        programs = {"gemm": gemm}
        if split and not atomic:
            if p.policy.split_k_slices != 2:
                raise ValueError("non-atomic dense reduction requires two slices")
            programs["reduce"] = _reduce_split_k2_bf16_kernel.warmup(
                torch.float32, torch.bfloat16, p.m * p.n, BLOCK=1024,
                grid=(triton.cdiv(p.m * p.n, 1024),),
            )
        return programs


def _require_dense_value(tensor, shape, dtype, device, name):
    if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(f"{name} differs from prepared dense shape, dtype, or device")
    # Pointer-based launchers use physical [L,M,K] / [L,M,N], ignoring strides
    # on singleton axes. Reject padded rows rather than silently misaddress them.
    strides = (shape[1], 1, shape[0] * shape[1])
    if any(size > 1 and actual != expected for size, actual, expected in zip(shape, tensor.stride(), strides)):
        raise ValueError(f"{name} must retain the prepared group-major layout")
    if tensor.numel() and tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")


@dataclass(frozen=True)
class _DenseExecutionState:
    lowering: _DenseLowering
    device: torch.device
    gemm: Callable
    reduction: object | None
    alpha_one: torch.Tensor | None

    def run(self, lhs, rhs, out=None, *, alpha=None, stream=None,
            rhs_values_tiled=None, quantized_c=None, x_bf16=None, w_gscale=None,
            row_scale=None, split_k_workspace=None):
        p = self.lowering
        a, sfa = lhs
        b, sfb = rhs
        m = int(a.shape[0])
        if m > p.m:
            raise ValueError(
                "dense execution exceeds its planned capacity"
            )
        value_dtype = torch.uint8 if p.ab_dtype != "float8_e4m3fn" else torch.float8_e4m3fn
        a_dtype = torch.float8_e4m3fn if p.mxfp6_fmt_a == "e4m3" else value_dtype
        _require_dense_value(a, (m, p.a_storage_k, p.l), a_dtype, self.device, "A")
        _require_dense_value(b, (p.n, p.b_storage_k, p.l), value_dtype, self.device, "B")
        for name, scale in (("SFA", sfa), ("SFB", sfb)):
            if scale.device != self.device or scale.dtype != getattr(torch, p.sf_dtype):
                raise ValueError(f"{name} differs from prepared dense dtype/device")
        if (alpha is None) != p.alpha_is_one:
            raise ValueError("dense scalar-alpha presence differs from preparation")
        if alpha is not None and (
            alpha.device != self.device or alpha.dtype != getattr(torch, p.alpha_dtype)
            or alpha.numel() != 1 or not alpha.is_contiguous()
        ):
            raise ValueError("dense alpha must retain its declared scalar dtype/device")
        if (rhs_values_tiled is not None) != p.b_tile_major:
            raise ValueError("dense tiled weight representation changed")
        if (quantized_c is not None) != p.quantize_c or (row_scale is not None) != p.row_scale:
            raise ValueError("dense auxiliary operand presence changed")
        if out is not None:
            _require_dense_value(out, (m, p.n, p.l), getattr(torch, p.c_dtype), self.device, "C")
        if m == 0:
            return out if out is not None else _empty_dense_gemm_output(
                0, p.n, p.l, dtype=getattr(torch, p.c_dtype), device=self.device,
            )
        if p.is_mxfp6:
            if not p.a_preexpanded:
                a = _expand_packed_mxfp6_ab(a, p.k)
            if not (p.b_preexpanded or p.b_packed):
                b = _expand_packed_mxfp6_ab(b, p.k)
        if out is None:
            out = _empty_dense_gemm_output(m, p.n, p.l, dtype=getattr(torch, p.c_dtype), device=self.device)
        alpha_value = self.alpha_one if alpha is None else alpha
        stream_int = cuda_stream_to_int(stream)
        if p.is_mxfp6:
            if row_scale is not None and (
                row_scale.shape != (m,) or row_scale.dtype != getattr(torch, p.c_dtype)
                or row_scale.device != self.device or not row_scale.is_contiguous()
                or row_scale.data_ptr() % 16
            ):
                raise ValueError("dense FP6 row scale differs from its planned layout")
            return self.gemm(
                a_tensor_gpu=a, b_tensor_gpu=b, sfa_tensor_gpu=sfa, sfb_tensor_gpu=sfb,
                c_tensor_gpu=out, alpha_tensor_gpu=alpha_value, stream_int=stream_int,
                x_bf16_tensor_gpu=x_bf16, w_gscale_tensor_gpu=w_gscale,
                row_scale_tensor_gpu=row_scale,
            )
        b_launch = b if rhs_values_tiled is None else rhs_values_tiled
        if p.quantize_c:
            values, rows, mma = quantized_c
            return self.gemm(
                a_tensor_gpu=a, b_tensor_gpu=b_launch, sfa_tensor_gpu=sfa, sfb_tensor_gpu=sfb,
                c_tensor_gpu=out, alpha_tensor_gpu=alpha_value, stream_int=stream_int,
                quant_c_values_gpu=values, quant_c_scale_rows_gpu=rows, quant_c_scale_mma_gpu=mma,
            )
        split = p.policy.split_k_slices > 1
        atomic = split and p.policy.split_k_atomic_bf16
        temporary = None
        if atomic:
            out.zero_()
        elif split:
            temporary = _dense_split_storage(split_k_workspace, p.policy.split_k_slices, m, p.n, self.device)
        self.gemm(
            a_tensor_gpu=a, b_tensor_gpu=b_launch, sfa_tensor_gpu=sfa, sfb_tensor_gpu=sfb,
            c_tensor_gpu=out if temporary is None else temporary.permute(1, 2, 0),
            alpha_tensor_gpu=alpha_value, stream_int=stream_int,
        )
        if temporary is not None:
            self.reduction[(triton.cdiv(m * p.n, 1024), 1, 1)](temporary, out, m * p.n, 1024)
        return out


def _materialize_dense(lowering, device):
    programs = _compile_dense_lowering(lowering.to_dict(), device.index)
    return _DenseExecutionState(
        lowering, device, programs["gemm"], programs.get("reduce"),
        _cached_alpha_one(device) if lowering.alpha_is_one else None,
    )


@dataclass(frozen=True)
class _DenseFusedQuantLowering:
    m: int
    n: int
    k: int
    groups: int
    grouped: bool
    source_shape: tuple[int, ...]
    source_strides: tuple[int, ...]
    b_dtype: str
    sfb_dtype: str
    expected_m: int | None
    sm_count: int
    mma_tiler_mn: tuple[int, int]
    policy: _DenseGemmPolicy
    sfb_k_replicated: bool
    b_tile_major: bool
    a_inner_span: int
    inv_rope: bool
    head_dim: int
    nope_dim: int
    rope_dim: int
    positions_dtype: str
    cos_sin_dtype: str
    output_provided: bool
    atomic_output_precleared: bool

    @property
    def workspace_nbytes(self):
        if self.policy.split_k_slices > 1 and not self.policy.split_k_atomic_bf16:
            return self.policy.split_k_slices * self.m * self.n * 4
        return 0

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        values["policy"] = _DenseGemmPolicy(**values["policy"])
        for name in ("source_shape", "source_strides", "mma_tiler_mn"):
            values[name] = tuple(values[name])
        return cls(**values)


def _lower_dense_gemm_fused_quant_a(
    source, b, sfb, *, sm_count, groups=None, out=None, positions=None,
    cos_sin_cache=None, head_dim=0, nope_dim=0, rope_dim=0, expected_m=None,
    sfb_k_replicated=False, rhs_values_tiled=None, a_inner_span=0,
    mma_tiler_mn=None, _atomic_output_precleared=False,
) -> _DenseFusedQuantLowering:
    """Lower the existing fused-quant branches without allocation or compilation."""
    grouped = groups is not None
    groups = 1 if groups is None else int(groups)
    a_inner_span = int(a_inner_span)
    if source.dtype != torch.bfloat16:
        raise ValueError("fused MXFP8 activation quantization requires BF16 A")
    if grouped:
        if a_inner_span or rhs_values_tiled is not None or _atomic_output_precleared:
            raise ValueError("grouped fused quantization does not use WO-B storage options")
        if source.ndim != 3 or int(source.shape[1]) != groups:
            raise ValueError("grouped fused quantization requires [M,groups,K]")
        m, k = int(source.shape[0]), int(source.shape[2])
        if source.stride(2) != 1 or source.stride(1) != k or source.stride(0) % 8:
            raise ValueError("grouped fused A requires contiguous trailing dimensions and an aligned row stride")
    elif not a_inner_span:
        if source.ndim != 2 or not source.is_contiguous():
            raise ValueError("fused MXFP8 activation quantization requires contiguous BF16 [M,K]")
        m, k = map(int, source.shape)
    else:
        if a_inner_span % 32 or source.ndim != 3 or int(source.shape[1]) != a_inner_span:
            raise ValueError("L-blocked fused A requires an aligned [M,span,K/span] view")
        m, k = int(source.shape[0]), a_inner_span * int(source.shape[2])
        if source.stride() != (a_inner_span, 1, m * a_inner_span):
            raise ValueError("L-blocked fused A must retain the group-major source layout")
    if not 1 <= m <= 8 or k % 128:
        raise ValueError("fused MXFP8 activation quantization requires M1..8 and K divisible by128")
    if b.ndim != 3 or tuple(b.shape[1:]) != (k, groups):
        raise ValueError(f"fused B must have shape [N,{k},{groups}]")
    n = int(b.shape[0])
    inv_rope = positions is not None or cos_sin_cache is not None
    if inv_rope:
        if not grouped or positions is None or cos_sin_cache is None:
            raise ValueError("grouped inverse-RoPE requires positions and cos_sin_cache")
        if positions.shape != (m,) or not positions.is_contiguous() or not cos_sin_cache.is_contiguous():
            raise ValueError("inverse-RoPE positions/cache must retain contiguous planned geometry")
        if cos_sin_cache.ndim != 2 or int(cos_sin_cache.shape[1]) != rope_dim:
            raise ValueError("inverse-RoPE cache width differs from rope_dim")
        if (head_dim <= 0 or nope_dim + rope_dim != head_dim or head_dim % 32
                or nope_dim % 32 or rope_dim % 32 or k % head_dim):
            raise ValueError("inverse-RoPE head/nope/rope dimensions must preserve the aligned source contract")
        _cutlass_positions_dtype(positions.dtype)
        _cutlass_cos_sin_dtype(cos_sin_cache.dtype)
        positions_dtype = str(positions.dtype).removeprefix("torch.")
        cos_sin_dtype = str(cos_sin_cache.dtype).removeprefix("torch.")
    else:
        head_dim = nope_dim = rope_dim = 0
        positions_dtype, cos_sin_dtype = "int64", "bfloat16"
    if mma_tiler_mn is None:
        selected = _select_default_dense_gemm_plan(
            m, n, k, sm_count, is_mxfp8=True, expected_m=expected_m,
        )
        if selected.swap_ab or selected.load_path != "tma":
            raise ValueError("fused quantization requires the unswapped TMA plan")
        mma_tiler_mn = selected.mma_tiler_mn
    mma_tiler_mn = tuple(mma_tiler_mn)
    if rhs_values_tiled is not None:
        if (n, k) != (4096, 4096) or mma_tiler_mn not in ((16, 64), (16, 128)):
            raise ValueError("tile-major fused RHS requires the existing WO-B4096 plan")
        if (rhs_values_tiled.shape != (1, 32, 32, 128, 128)
                or rhs_values_tiled.dtype != b.dtype or rhs_values_tiled.device != b.device
                or not rhs_values_tiled.is_contiguous()):
            raise ValueError("tile-major fused RHS differs from its native storage contract")
    policy = _dense_gemm_policy_for(
        m=m, n=n, k=k, l=groups, ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16, mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=(1, 1), sm_count=sm_count, tile_k=128,
        expected_m=expected_m, generalize_mxfp8_split_k=True,
    )
    if grouped and policy.split_k_slices != 1:
        raise ValueError("grouped fused quantization does not support split-K")
    if out is None and _atomic_output_precleared:
        raise ValueError("a precleared fused output must be caller-owned")
    if out is not None and (out.shape != (m, n, groups) or out.dtype != torch.bfloat16):
        raise ValueError("fused output must retain the planned BF16 shape")
    return _DenseFusedQuantLowering(
        m, n, k, groups, grouped, tuple(source.shape), tuple(source.stride()),
        str(b.dtype).removeprefix("torch."), str(sfb.dtype).removeprefix("torch."),
        expected_m, int(sm_count), mma_tiler_mn, policy, bool(sfb_k_replicated),
        rhs_values_tiled is not None, a_inner_span, inv_rope, int(head_dim),
        int(nope_dim), int(rope_dim), positions_dtype, cos_sin_dtype,
        out is not None, bool(_atomic_output_precleared),
    )


@program_cache(scope="preparation")
def _compile_dense_fused_quant_lowering(payload, device_ordinal):
    p = _DenseFusedQuantLowering.from_dict(payload)
    one_m = (p.expected_m if p.expected_m is not None else p.m) == 1
    split = p.policy.split_k_slices > 1
    partials = split and not p.policy.split_k_atomic_bf16
    with torch.cuda.device(device_ordinal):
        if p.grouped:
            gemm = _get_compiled_dense_gemm_fused_quant_a_grouped(
                p.n, p.k, p.groups, p.policy, p.mma_tiler_mn, p.sm_count,
                p.sfb_k_replicated, p.source_strides[0], p.k, p.inv_rope,
                p.head_dim, p.nope_dim, p.rope_dim, one_m,
                _cutlass_positions_dtype(getattr(torch, p.positions_dtype)),
                _cutlass_cos_sin_dtype(getattr(torch, p.cos_sin_dtype)),
            )
        else:
            gemm = _get_compiled_dense_gemm_fused_quant_a(
                p.n, p.k, cutlass.Float32 if partials else cutlass.BFloat16,
                p.policy, p.mma_tiler_mn, p.sm_count, p.sfb_k_replicated,
                p.b_tile_major, p.a_inner_span, p.policy.split_k_slices if partials else 1, one_m,
            )
        programs = {"gemm": gemm}
        if partials:
            if p.policy.split_k_slices != 2:
                raise ValueError("fused non-atomic split reduction requires two slices")
            programs["reduce"] = _reduce_split_k2_bf16_kernel.warmup(
                torch.float32, torch.bfloat16, p.m * p.n, BLOCK=1024,
                grid=(triton.cdiv(p.m * p.n, 1024),),
            )
        return programs


@dataclass(frozen=True)
class _DenseFusedQuantState:
    lowering: _DenseFusedQuantLowering
    device: torch.device
    gemm: Callable
    reduction: object | None
    alpha_one: torch.Tensor

    def run(self, source, b, sfb, out=None, *, positions=None, cos_sin_cache=None,
            rhs_values_tiled=None, split_k_workspace=None, stream=None):
        p = self.lowering
        m = int(source.shape[0])
        shape_matches = (tuple(source.shape[1:]) == p.source_shape[1:] and m <= p.m
                         and (not p.grouped or m == p.m))
        if (not shape_matches or source.stride() != p.source_strides
                or source.dtype != torch.bfloat16 or source.device != self.device):
            raise ValueError("fused A exceeds its prepared capacity or differs in layout/device")
        _require_dense_value(b, (p.n, p.k, p.groups), getattr(torch, p.b_dtype), self.device, "fused B")
        if sfb.device != self.device or sfb.dtype != getattr(torch, p.sfb_dtype):
            raise ValueError("fused SFB differs from its prepared dtype/device")
        if (out is not None) != p.output_provided:
            raise ValueError("fused output allocation form changed")
        if (rhs_values_tiled is not None) != p.b_tile_major:
            raise ValueError("fused tiled RHS presence changed")
        if p.inv_rope:
            if (positions is None or positions.shape != (m,)
                    or positions.dtype != getattr(torch, p.positions_dtype)
                    or positions.device != self.device or not positions.is_contiguous()
                    or cos_sin_cache is None or cos_sin_cache.ndim != 2
                    or int(cos_sin_cache.shape[1]) != p.rope_dim
                    or cos_sin_cache.dtype != getattr(torch, p.cos_sin_dtype)
                    or cos_sin_cache.device != self.device or not cos_sin_cache.is_contiguous()):
                raise ValueError("fused inverse-RoPE operands differ from preparation")
        elif positions is not None or cos_sin_cache is not None:
            raise ValueError("inverse-RoPE was not declared")
        if out is None:
            out = _empty_dense_gemm_output(m, p.n, p.groups, dtype=torch.bfloat16, device=self.device)
        else:
            _require_dense_value(out, (m, p.n, p.groups), torch.bfloat16, self.device, "fused C")
        if m == 0:
            return out
        stream_int = cuda_stream_to_int(stream)
        if p.grouped:
            self.gemm(
                source, positions if p.inv_rope else source, cos_sin_cache if p.inv_rope else source,
                b, sfb, out, self.alpha_one, stream_int,
            )
            return out
        split = p.policy.split_k_slices > 1
        atomic = split and p.policy.split_k_atomic_bf16
        temporary = None
        if atomic:
            if not p.atomic_output_precleared:
                out.zero_()
        elif split:
            temporary = _dense_split_storage(split_k_workspace, p.policy.split_k_slices, m, p.n, self.device)
        b_launch = b
        if rhs_values_tiled is not None:
            if (rhs_values_tiled.shape != (1, 32, 32, 128, 128)
                    or rhs_values_tiled.dtype != b.dtype or rhs_values_tiled.device != self.device
                    or not rhs_values_tiled.is_contiguous()):
                raise ValueError("fused tile-major RHS storage changed")
            b_launch = rhs_values_tiled
        self.gemm(source, b_launch, sfb, out if temporary is None else temporary, self.alpha_one, stream_int)
        if temporary is not None:
            self.reduction[(triton.cdiv(m * p.n, 1024), 1, 1)](temporary, out, m * p.n, 1024)
        return out


def _materialize_dense_fused_quant(lowering, device):
    programs = _compile_dense_fused_quant_lowering(lowering.to_dict(), device.index)
    return _DenseFusedQuantState(
        lowering, device, programs["gemm"], programs.get("reduce"), _cached_alpha_one(device),
    )
