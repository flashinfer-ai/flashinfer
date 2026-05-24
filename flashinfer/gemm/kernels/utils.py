"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ...fused_moe.utils import last_positive_power_of_2

_SM100_MMA_TILER_MN_CANDIDATES = [
    (128, 64),
    (256, 64),
    (128, 128),
    (256, 128),
    (128, 192),
    (256, 192),
    (128, 256),
    (256, 256),
]

# Tactic cache: (n, real_k, sm_count) -> dict[(m_bucket, is_8_aligned) -> tactic_tuple]
# Bounded by the number of unique (N, K) pairs in the model (typically < 50).
_SM100_MM_FP4_TACTIC_CACHE: dict[tuple, dict] = {}

# M bucket boundaries — powers of 2 for fast bucketing via
# last_positive_power_of_2 (imported from flashinfer.fused_moe.utils).
# Each bucket is precomputed for both 8-aligned and non-8-aligned M,
# keyed as (bucket, is_8_aligned).
_M_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _compute_tactic_for_m(rep_m, n, real_k, sm_count, m_aligned):
    """Compute the best tactic for a specific (M, N, K) on a GPU with sm_count SMs.

    Selects swap_ab, tile shape, and cluster shape sequentially:

    1. **swap_ab**: Swap A and B operands when M is small (8-16) and
       8-aligned, putting the larger N dimension on the M-axis to increase
       the number of CTAs.  Also swaps when N is not 8-aligned (required
       for memory alignment).

    2. **Tile shape**: Scores all 8 candidates from _SM100_MMA_TILER_MN_CANDIDATES.
       The score balances three factors:
         - Tile quantization: M and N padding waste from rounding up to tile
           boundaries. Smaller tiles waste less for small dimensions.
         - Wave quantization: How well total CTAs fill the available SMs.
           Ideal when total_ctas is a multiple of sm_count.
         - Tile throughput: Larger, balanced tiles have higher per-CTA
           throughput. Penalized for small K (<=2048) where the K-pipeline
           can't hide launch latency of large tiles.
       The combined score is: m_eff * wave_eff * n_eff * tile_throughput.

    3. **Cluster shape**: For narrow GEMMs (prob_m fits in one tile row),
       clusters in the N dimension (up to 4) for scale-factor multicast.
       Otherwise uses (1, 1).  tile_m=256 forces cluster_m=2 (HW constraint).

    4. **Prefetch**: Disabled.
    """
    n_aligned = n % 8 == 0

    swap_ab = False
    if m_aligned and 8 <= rep_m <= 16 and n > rep_m:
        swap_ab = True
    if not swap_ab and not n_aligned and m_aligned:
        swap_ab = True

    prob_m = n if swap_ab else rep_m
    prob_n = rep_m if swap_ab else n

    # Small-K penalty factor (loop-invariant).
    if real_k <= 1024:
        large_tile_penalty = 0.50
    elif real_k <= 2048:
        large_tile_penalty = 0.80
    else:
        large_tile_penalty = 1.0

    # Score all 8 tile candidates.
    max_tile_area = 256 * 256
    best_tile_m = 128
    best_tile_n = 128
    best_score = -1.0

    for tile_m, tile_n in _SM100_MMA_TILER_MN_CANDIDATES:
        n_tiles = (prob_n + tile_n - 1) // tile_n
        n_eff = prob_n / (n_tiles * tile_n)
        tile_area_factor = ((tile_m * tile_n) / max_tile_area) ** 0.5
        tile_bal = min(tile_m, tile_n) / max(tile_m, tile_n)
        tile_tp = tile_area_factor * (tile_bal**0.25)
        ns = n_eff * tile_tp

        if tile_m * tile_n > 128 * 128:
            ns *= large_tile_penalty

        m_tiles = (prob_m + tile_m - 1) // tile_m
        total_ctas = m_tiles * n_tiles
        num_waves = (total_ctas + sm_count - 1) // sm_count
        score = prob_m * total_ctas * ns / (tile_m * num_waves * sm_count)
        if score > best_score:
            best_score = score
            best_tile_m = tile_m
            best_tile_n = tile_n

    # Cluster: N-only for small prob_m, else (1,1).
    tiles_on_n = (prob_n + best_tile_n - 1) // best_tile_n
    if prob_m <= best_tile_m:
        if tiles_on_n % 4 == 0 or tiles_on_n > 10:
            cga_n = 4
        elif tiles_on_n % 2 == 0:
            cga_n = 2
        else:
            cga_n = 1
        cga_m = 1
    else:
        cga_m = 1
        cga_n = 1

    if best_tile_m == 256 and cga_m < 2:
        cga_m = 2

    return (
        (best_tile_m, best_tile_n),
        (cga_m, cga_n),
        swap_ab,
        False,
        "sm100",
        None,
    )


def _select_sm100_mm_fp4_cute_dsl_tactic(m, n, real_k, sm_count):
    """Select the best tactic for mm_fp4(backend='cute-dsl').

    On the first call for a given (N, K), precomputes the optimal tactic
    for each M bucket (~13 buckets, ~55-86 usec).  Subsequent calls with
    any M just look up the bucket — runs in ~0.2 usec.

    Args:
        m: M dimension of the GEMM problem.
        n: N dimension of the GEMM problem.
        real_k: K dimension (unpacked, i.e. 2x the packed FP4 dimension).
        sm_count: Number of SMs on the target GPU.

    Returns:
        Tactic tuple: (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch,
                        kernel_type, use_tma_store)
    """
    cache_key = (n, real_k, sm_count)
    bucket_tactics = _SM100_MM_FP4_TACTIC_CACHE.get(cache_key)
    if bucket_tactics is None:
        bucket_tactics = {}
        for rep_m in _M_BUCKETS:
            for aligned in (True, False):
                bucket_tactics[(rep_m, aligned)] = _compute_tactic_for_m(
                    rep_m, n, real_k, sm_count, aligned
                )
        _SM100_MM_FP4_TACTIC_CACHE[cache_key] = bucket_tactics

    bucket = min(last_positive_power_of_2(m), _M_BUCKETS[-1])
    return bucket_tactics[(bucket, m % 8 == 0)]
