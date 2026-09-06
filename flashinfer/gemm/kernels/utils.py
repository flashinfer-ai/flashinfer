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

from flashinfer.utils import next_positive_power_of_2

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

# Tactic cache: (n, real_k, sm_count, sf_vec_size) -> dict[m_bucket -> tactic_tuple]
# Bounded by the number of unique (N, K) pairs in the model (typically < 50).
_SM100_MM_FP4_TACTIC_CACHE: dict[tuple, dict] = {}

# M bucket boundaries — powers of 2 for fast bucketing via
# next_positive_power_of_2 (imported from flashinfer.utils).
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


def _compute_tactic_for_m(rep_m, n, real_k, sm_count, sf_vec_size):
    """Compute the best tactic for a specific (M, N, K) on a GPU with sm_count SMs.

    Used for mm_fp4(backend='cute-dsl') without autotune by taking the
    argmax of _score_sm100_mm_fp4_tactic over kernel-feasible
    (tile, cluster, swap_ab), with prefetch disabled.
    """
    import cutlass

    from .dense_blockscaled_gemm_sm100 import Sm100BlockScaledPersistentDenseGemmKernel

    sf_dtype = cutlass.Float8E4M3FN if sf_vec_size == 16 else cutlass.Float8E8M0FNU

    def is_feasible(tile, cluster, swap_ab):
        kernel_m, kernel_n = (n, rep_m) if swap_ab else (rep_m, n)
        return Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
            cutlass.Float4E2M1FN,
            sf_dtype,
            sf_vec_size,
            cutlass.BFloat16,  # Note: BF16 or FP16 does not impact can_implement outcome
            tile,
            cluster,
            kernel_m,
            kernel_n,
            real_k,
            1,
            "k",
            "k",
            "m" if swap_ab else "n",
        )

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
                if score > best_score and is_feasible(tile, cluster, swap_ab):
                    best_score = score
                    best_tactic = (tile, cluster, swap_ab, False, "sm100", None)
    return best_tactic


def _score_sm100_mm_fp4_tactic(
    m, n, real_k, sm_count, mma_tiler_mn, cluster_shape_mn, swap_ab
):
    """Score a mm_fp4 cute-dsl tactic (higher means better).
    Used for ranking candidates for both top-1 and autotune scenarios.
    """
    if m == 0 or n == 0:
        return 0.0
    tile_m, tile_n = mma_tiler_mn
    cga_m, cga_n = cluster_shape_mn
    prob_m = n if swap_ab else m
    prob_n = m if swap_ab else n

    # 1. Check tile-quantization efficiency from padding waste
    m_tiles = (prob_m + tile_m - 1) // tile_m
    n_tiles = (prob_n + tile_n - 1) // tile_n
    m_eff = prob_m / (m_tiles * tile_m)
    n_eff = prob_n / (n_tiles * tile_n)

    # tile_m == 256 runs as 2-CTA cooperative MMA
    cta_group = 2 if tile_m == 256 else 1
    ctas_m = m_tiles * cta_group
    # 2. Wave quantization effects: count waves over the real CTA grid
    padded_ctas = ((ctas_m + cga_m - 1) // cga_m * cga_m) * (
        (n_tiles + cga_n - 1) // cga_n * cga_n
    )
    num_waves = (padded_ctas + sm_count - 1) // sm_count

    # 3. Calculate per-CTA throughput: per-CTA tile is (128, tile_n).
    # Wide tiles are penalized at small K where the
    # K-pipeline cannot hide their latency.
    throughput = (tile_n / 256) ** 0.5
    if tile_n > 128:
        if real_k <= 1024:
            throughput *= 0.50
        elif real_k <= 2048:
            throughput *= 0.80

    score = m_eff * n_eff * throughput / (num_waves * tile_n)

    # 4. Tie-breaking
    if cta_group == 2:
        # 2-CTA MMA: higher per-SM throughput at equal per-CTA tile
        score *= 1.05
    if prob_m > tile_m:
        # Cluster multicast hurts when the m-grid is thicker than one tile
        # cga_x.bit_length() for cga_x = 1/2/4 -> exponent 0/1/2
        score *= 0.95 ** (cga_n.bit_length() - 1)
        score *= 0.99 ** (cga_m.bit_length() - 1)

    # 5. Demote narrow tiles that cannot cover prob_n,
    # keeping the ranking total over all valid tactics.
    if tile_n < 64 and prob_n > tile_n:
        score *= 1e-6

    # 6. Penalize deviations from the swap rule mildly, so both swap variants stay in the top-N.
    rule_swap = (n % 8 == 0) and (1 <= m <= 32) and n > m
    if swap_ab != rule_swap:
        score *= 0.95
    return score


def _select_sm100_mm_fp4_cute_dsl_tactic(m, n, real_k, sm_count, sf_vec_size):
    """Select the best tactic for mm_fp4(backend='cute-dsl').

    On the first call for a given (N, K), precomputes the optimal tactic
    for each M bucket. Subsequent calls with any M just look up the bucket
    — runs in ~0.2 usec.

    Args:
        m: M dimension of the GEMM problem.
        n: N dimension of the GEMM problem.
        real_k: K dimension (unpacked, i.e. 2x the packed FP4 dimension).
        sm_count: Number of SMs on the target GPU.
        sf_vec_size: Scale-factor vector size (16 for nvfp4, 32 for mxfp4).

    Returns:
        Tactic tuple: (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch,
                        kernel_type, use_tma_store)
    """
    cache_key = (n, real_k, sm_count, sf_vec_size)
    bucket_tactics = _SM100_MM_FP4_TACTIC_CACHE.get(cache_key)
    if bucket_tactics is None:
        bucket_tactics = {}
        for rep_m in _M_BUCKETS:
            bucket_tactics[rep_m] = _compute_tactic_for_m(
                rep_m, n, real_k, sm_count, sf_vec_size
            )
        _SM100_MM_FP4_TACTIC_CACHE[cache_key] = bucket_tactics

    bucket = min(next_positive_power_of_2(m), _M_BUCKETS[-1])
    return bucket_tactics[bucket]


# --- sm107 (Rubin) ----------------------------------------------------------
#
# sm107 does not replace the sm100 candidate set, it extends it. The sm100
# kernel runs on Rubin (sm_100f family target) and gemm_base.get_valid_tactics
# enumerates sm100 tactics alongside sm107 ones there, so both are reachable.
# The autotuner picks from both: over 110 tuned fp4_gemm shapes on DeepSeek-R1
# nvfp4 tp4 (VR200) its winners split sm100 29 / sm107 1 at M<=32, sm100 16 /
# sm107 4 at M 33-512, and sm107 41 / sm100 19 above that -- narrow sm100 tiles
# win decode, wide sm107 tiles win prefill.
#
# So this scores the union with the same _score_sm100_mm_fp4_tactic used for
# sm100 and for the top-N ranking, and emits whichever kernel won.
_SM107_MMA_TILER_MN_CANDIDATES = [
    (128, 64),
    (256, 64),
    (128, 128),
    (256, 128),
    (128, 192),
    (256, 192),
    (128, 256),
    (256, 256),
]

# (n, real_k, sm_count, sf_vec_size) -> dict[m_bucket -> tactic_tuple]
_SM107_MM_FP4_TACTIC_CACHE: dict[tuple, dict] = {}

# Fixed for FP4, matching the tactic enumeration in gemm_base.py.
_SM107_MMA_INST_SHAPE_K = 128
_SM107_MMA_TILER_K = 256

# Kernel default since 60a54049f ("Default prefetch off for Rubin dense GEMM"):
# TRT-LLM's auto depth (num_ab_stage, 6-11 tiles) costs 25-40% untuned at
# generic shapes on Rubin.
_SM107_UNTUNED_PREFETCH_DIST = 0


def _compute_sm107_tactic_for_m(rep_m, n, real_k, sm_count, sf_vec_size):
    """Best mm_fp4 tactic for one (M, N, K) on sm107.

    Same structure as _compute_tactic_for_m, over the union of the sm100 and
    sm107 candidate sets, ranked by the same scorer.
    """
    import cutlass

    from .dense_blockscaled_gemm_sm100 import Sm100BlockScaledPersistentDenseGemmKernel

    try:
        from .dense_blockscaled_gemm_sm107 import (
            Sm107BlockScaledPersistentDenseGemmKernel,
        )
    except ImportError:
        Sm107BlockScaledPersistentDenseGemmKernel = None

    sf_dtype = cutlass.Float8E4M3FN if sf_vec_size == 16 else cutlass.Float8E8M0FNU
    n_aligned = n % 8 == 0

    def sm100_feasible(tile, cluster, swap_ab):
        kernel_m, kernel_n = (n, rep_m) if swap_ab else (rep_m, n)
        return Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
            cutlass.Float4E2M1FN,
            sf_dtype,
            sf_vec_size,
            cutlass.BFloat16,
            tile,
            cluster,
            kernel_m,
            kernel_n,
            real_k,
            1,
            "k",
            "k",
            "m" if swap_ab else "n",
        )

    def sm107_feasible(tile, inst_shape, cluster, swap_ab):
        kernel_m, kernel_n = (n, rep_m) if swap_ab else (rep_m, n)
        return Sm107BlockScaledPersistentDenseGemmKernel.can_implement(
            cutlass.Float4E2M1FN,
            sf_dtype,
            sf_vec_size,
            cutlass.BFloat16,
            tile,
            inst_shape,
            cluster,
            kernel_m,
            kernel_n,
            real_k,
            1,
            "k",
            "k",
            "m" if swap_ab else "n",
        )

    best_tactic = None
    best_score = -1.0

    # sm107 candidates are scored FIRST so that ties resolve to the
    # Rubin-native kernel. _score_sm100_mm_fp4_tactic reads only tile, cluster
    # and swap_ab, so a tile offered by both kernels scores identically and the
    # strict `>` below keeps whichever was seen first. The autotuner's measured
    # preference is sm107 at large M (41/60 winners above M=512), so sm107 wins
    # ties; sm100's narrow tiles still win at decode because they score strictly
    # higher there.
    # sm107 candidates -- wider tiles on the Rubin-native kernel.
    #
    # mma_inst_shape_m is legal iff tiler_m == inst_m or tiler_m == 2 * inst_m
    # ("b_reuse"). Pinned to tiler_m (the equal case) so it agrees with the
    # scorer, which already treats tile_m == 256 as the 2-CTA cooperative MMA.
    #
    # swap_ab additionally requires m % 8 == 0 here: gemm_base gates the sm107
    # tactic path on m_aligned, so M=1 cannot swap on this kernel.
    if Sm107BlockScaledPersistentDenseGemmKernel is not None:
        m_aligned = rep_m % 8 == 0
        for tile in _SM107_MMA_TILER_MN_CANDIDATES:
            inst_shape = (tile[0], tile[1], _SM107_MMA_INST_SHAPE_K)
            for cluster in _SM100_CLUSTER_SHAPE_MN_CANDIDATES:
                if tile[0] == 256 and cluster[0] < 2:
                    continue
                swap_options = (False, True) if (n_aligned and m_aligned) else (False,)
                for swap_ab in swap_options:
                    if swap_ab and not n_aligned:
                        continue
                    score = _score_sm100_mm_fp4_tactic(
                        rep_m, n, real_k, sm_count, tile, cluster, swap_ab
                    )
                    if score > best_score and sm107_feasible(
                        tile, inst_shape, cluster, swap_ab
                    ):
                        best_score = score
                        best_tactic = (
                            tile,
                            cluster,
                            swap_ab,
                            False,  # use_prefetch is sm100-only
                            "sm107",
                            (
                                inst_shape[0],
                                tile[1],
                                _SM107_MMA_INST_SHAPE_K,
                                _SM107_MMA_TILER_K,
                                _SM107_UNTUNED_PREFETCH_DIST,
                            ),
                        )

    # sm100 candidates -- the only ones that can express tile_n < 64, which is
    # what wins at decode M once swap_ab puts the small dimension on the N axis.
    for tile in _SM100_MMA_TILER_MN_CANDIDATES:
        for cluster in _SM100_CLUSTER_SHAPE_MN_CANDIDATES:
            if tile[0] == 256 and cluster[0] < 2:
                continue  # 2-CTA MMA (tile_m == 256) requires cluster_m >= 2
            for swap_ab in (False,) if not n_aligned else (False, True):
                score = _score_sm100_mm_fp4_tactic(
                    rep_m, n, real_k, sm_count, tile, cluster, swap_ab
                )
                if score > best_score and sm100_feasible(tile, cluster, swap_ab):
                    best_score = score
                    best_tactic = (tile, cluster, swap_ab, False, "sm100", None)

    return best_tactic


def _select_sm107_mm_fp4_cute_dsl_tactic(m, n, real_k, sm_count, sf_vec_size):
    """Select the best mm_fp4(backend='cute-dsl') tactic on sm107.

    Mirrors _select_sm100_mm_fp4_cute_dsl_tactic: the per-M-bucket table is
    computed once per (N, K, sm_count, sf_vec_size) and then looked up.

    Returns a tactic tuple whose kernel_type is "sm100" or "sm107" depending on
    which candidate scored highest; both run on Rubin.
    """
    cache_key = (n, real_k, sm_count, sf_vec_size)
    bucket_tactics = _SM107_MM_FP4_TACTIC_CACHE.get(cache_key)
    if bucket_tactics is None:
        bucket_tactics = {}
        for rep_m in _M_BUCKETS:
            bucket_tactics[rep_m] = _compute_sm107_tactic_for_m(
                rep_m, n, real_k, sm_count, sf_vec_size
            )
        _SM107_MM_FP4_TACTIC_CACHE[cache_key] = bucket_tactics

    bucket = min(next_positive_power_of_2(m), _M_BUCKETS[-1])
    return bucket_tactics[bucket]
