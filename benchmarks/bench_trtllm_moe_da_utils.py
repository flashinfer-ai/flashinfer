from __future__ import annotations

import numpy as np

from flashinfer.fused_moe.dist_aware.da_utils import (
    DADistributionSpec,
    _exp_floor_probs_for_target_eff,
    _sparse_probs,
    _symmetric_dirichlet_probs_for_target_eff,
    da_distribution_target_effective_experts,
)


def _counts_from_probs(probs: np.ndarray, total: int) -> np.ndarray:
    """Round probabilities into integer counts while preserving the total."""

    raw = probs * float(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        counts[order[:remainder]] += 1
    return counts


def da_distribution_counts(
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    distribution: DADistributionSpec,
) -> np.ndarray:
    """Deterministic expert-count vector for a DA synthetic benchmark distribution.

    The returned counts always sum to ``num_tokens * top_k``. This keeps the
    benchmark's generated assignments consistent across TP/EP comparisons.
    """

    total = int(num_tokens) * int(top_k)
    n = int(num_local_experts)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    if total <= 0:
        return np.zeros(n, dtype=np.int64)

    label, kind, param = distribution
    del label
    if kind == "uniform":
        counts = np.full(n, total // n, dtype=np.int64)
        counts[: total % n] += 1
        return counts
    if kind == "single":
        counts = np.zeros(n, dtype=np.int64)
        counts[0] = total
        return counts
    if kind in ("sparse_eff", "sparse_factor"):
        return _counts_from_probs(_sparse_probs(kind, param, n), total)
    if kind == "ddist_factor":
        probs = _symmetric_dirichlet_probs_for_target_eff(
            da_distribution_target_effective_experts(distribution, n),
            n,
        )
        return _counts_from_probs(probs, total)
    if kind != "exp_factor":
        raise ValueError(f"Unknown DA distribution kind: {kind!r}")

    target_eff = da_distribution_target_effective_experts(distribution, n)
    return _counts_from_probs(_exp_floor_probs_for_target_eff(target_eff, n), total)
