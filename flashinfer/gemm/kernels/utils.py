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

from ...fused_moe.utils import next_positive_power_of_2

_SM100_MMA_TILER_MN_CANDIDATES = [
    (128, 8),
    (128, 16),
    (128, 32),
    (128, 64),
    (256, 64),
    (128, 128),
    (256, 128),
    (128, 192),
    (256, 192),
    (128, 256),
    (256, 256),
]

# Tactic cache: (n, real_k, sm_count) -> dict[m_bucket -> tactic_tuple]
# Bounded by the number of unique (N, K) pairs in the model (typically < 50).
_SM100_MM_FP4_TACTIC_CACHE: dict[tuple, dict] = {}

# M bucket boundaries — powers of 2 for fast bucketing via
# next_positive_power_of_2 (imported from flashinfer.fused_moe.utils).
_M_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


_SM100_CLUSTER_SHAPE_MN_CANDIDATES = [
    (1, 1),
    (1, 2),
    (1, 4),
    (2, 1),
    (2, 2),
    (2, 4),
    (4, 1),
    (4, 2),
    (4, 4),
]


def _compute_tactic_for_m(rep_m, n, real_k, sm_count):
    """Compute the best tactic for a specific (M, N, K) on a GPU with sm_count SMs.

    Argmax of _score_sm100_mm_fp4_tactic over (tile, cluster, swap_ab), with
    prefetch disabled.  Measured against the previous rule-based selection on
    B200 over the 217 unique shapes of the mm_fp4 tuning + holdout testlists:
    geomean -2.6%, wins up to 2x.
    """
    n_aligned = n % 8 == 0
    best_tactic = None
    best_score = -1.0
    for tile in _SM100_MMA_TILER_MN_CANDIDATES:
        for cluster in _SM100_CLUSTER_SHAPE_MN_CANDIDATES:
            if tile[0] == 256 and cluster[0] < 2:
                continue  # 2-CTA MMA (tile_m == 256) requires cluster_m >= 2
            # swap_ab is only valid for 8-aligned n
            for swap_ab in (False,) if not n_aligned else (False, True):
                score = _score_sm100_mm_fp4_tactic(
                    rep_m, n, real_k, sm_count, tile, cluster, swap_ab
                )
                if score > best_score:
                    best_score = score
                    best_tactic = (tile, cluster, swap_ab, False, "sm100", None)
    return best_tactic


def _score_sm100_mm_fp4_tactic(
    m, n, real_k, sm_count, mma_tiler_mn, cluster_shape_mn, swap_ab
):
    """Score a mm_fp4 cute-dsl tactic (higher = better), modeling execution
    time as num_waves x per-CTA-tile time.

    Ranks autotune candidates and drives _compute_tactic_for_m's no-autotune
    argmax.  Validated against exhaustive per-tactic measurements on B200
    over 16 (m, n, k) shapes: the top-16 configs by this score contain a
    tactic within 0.1% of the exhaustive optimum on every shape.
    """
    tile_m, tile_n = mma_tiler_mn
    cga_m, cga_n = cluster_shape_mn
    prob_m = n if swap_ab else m
    prob_n = m if swap_ab else n

    # Tile-quantization efficiency: padding waste from rounding the problem
    # up to tile boundaries.
    m_tiles = (prob_m + tile_m - 1) // tile_m
    n_tiles = (prob_n + tile_n - 1) // tile_n
    m_eff = prob_m / (m_tiles * tile_m)
    n_eff = prob_n / (n_tiles * tile_n)

    # tile_m == 256 runs as 2-CTA cooperative MMA (the reason it requires
    # cluster_m >= 2), so each output tile occupies two SMs.
    cta_group = 2 if tile_m == 256 else 1
    ctas_m = m_tiles * cta_group
    # Wave count over the real CTA grid, padded to the cluster shape:
    # partially-filled clusters still occupy whole clusters' worth of SMs.
    padded_ctas = ((ctas_m + cga_m - 1) // cga_m * cga_m) * (
        (n_tiles + cga_n - 1) // cga_n * cga_n
    )
    num_waves = (padded_ctas + sm_count - 1) // sm_count

    # Per-CTA tile is (128, tile_n) regardless of tile_m; its throughput
    # scales ~sqrt(tile_n).  Wide tiles are penalized at small K where the
    # K-pipeline cannot hide their latency.
    throughput = (tile_n / 256) ** 0.5
    if tile_n > 128:
        if real_k <= 1024:
            throughput *= 0.50
        elif real_k <= 2048:
            throughput *= 0.80

    score = m_eff * n_eff * throughput / (num_waves * tile_n)

    # Tie-breakers keep the ranking strict where the wave model is degenerate
    # (equal waves and quantization).  Exact values sit on a wide flat
    # optimum of the validation sweep.
    if cta_group == 2:
        # 2-CTA MMA: higher per-SM throughput at equal per-CTA tile
        score *= 1.05
    if prob_m > tile_m:
        # cluster multicast hurts when the m-grid is thicker than one tile
        # (measured up to 1.8x slower); cga values 1/2/4 -> exponent 0/1/2
        score *= 0.95 ** (0, 1, 2)[cga_n.bit_length() - 1]
        score *= 0.99 ** (0, 1, 2)[cga_m.bit_length() - 1]

    # Demote (rather than exclude) narrow tiles that cannot cover prob_n,
    # keeping the ranking total over all valid tactics.
    if tile_n < 64 and prob_n > tile_n:
        score *= 1e-6

    # The swap comparison is only roughly calibrated, but large-m swap
    # winners can beat their twin by 12-16%: penalize deviations from the
    # swap rule mildly, so both swap variants stay in the profiled top-N.
    rule_swap = (n % 8 == 0) and (1 <= m <= 32) and n > m
    if swap_ab != rule_swap:
        score *= 0.95
    return score


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
            bucket_tactics[rep_m] = _compute_tactic_for_m(rep_m, n, real_k, sm_count)
        _SM100_MM_FP4_TACTIC_CACHE[cache_key] = bucket_tactics

    bucket = min(next_positive_power_of_2(m), _M_BUCKETS[-1])
    return bucket_tactics[bucket]
