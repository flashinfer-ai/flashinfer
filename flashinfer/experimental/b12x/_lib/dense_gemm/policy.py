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

"""Dense GEMM tile and scheduling selection."""

from typing import Optional
from typing import Tuple
from b12x._lib.dense_gemm.config import (
    _DenseGemmPlan,
    _FP6_DECODE_TILE,
    _FP6_PREFILL_TILE,
    _select_block_fp8_decode_slices,
    _use_low_sm_dense_tactics,
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
            if _use_high_sm_mxfp8_bk64_prefill(expected_m, n, k, sm_count):
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
            1536 <= plan_m <= 2048 and (n + 127) // 128 <= 2 * sm_count
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
    if block_fp8 and _select_block_fp8_decode_slices(plan_m, n, k, sm_count) > 1:
        tile = (32, 64)
    elif block_fp8 and not _use_low_sm_dense_tactics(sm_count) and plan_m >= 2048:
        tile = (64, 128)
    elif block_fp8 and not _use_low_sm_dense_tactics(sm_count) and 96 <= plan_m <= 128:
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
        tile = (128, 128) if _use_low_sm_dense_tactics(sm_count) else (128, 64)
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
