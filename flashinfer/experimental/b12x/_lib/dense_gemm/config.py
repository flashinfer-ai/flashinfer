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

"""Dense GEMM configuration and device helpers."""

from cutlass import Int32
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Type
from cutlass import Uint32
import cutlass.cute as cute
import cutlass
from dataclasses import dataclass
from b12x._lib.utils.device import get_max_active_clusters
from b12x._lib.intrinsics.memory import ld_shared_v4_u32
import logging
from b12x._lib.intrinsics.mma import mma_m16n8k32_f32_e4m3
import os
from b12x._lib.intrinsics.memory import st_shared_v4_u32
import cutlass.utils as utils


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
                    ((m + tile_m - 1) // tile_m) * ((n + tile_n - 1) // tile_n) * l
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
                if m == 8 and (n, k) == (4096, 4096) and mma_tiler_mn == (16, 128)
                else 2
            )
    if not _B12X_DENSE_SPLITK_TURBO:
        # The FP32 workspace reducer consumes exactly two partials.
        split_k_slices = min(split_k_slices, 2)
    # A declared expected_m owns compile-time tuning for its regime. Without a
    # hint, keep the unroll choice stable throughout the persistent scheduler
    # regime so one warmed kernel covers every live M in that regime.
    large_m_unroll_threshold = 4096 if _use_low_sm_dense_tactics(sm_count) else 8192
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
    if m > 6 or n_tiles_64 < minimum_tiles or k < 4096:
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
