# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only acceptance and stationary metadata fixtures for ReplaySSM.

T includes the verifier token. Acceptance lengths are independent between
steps; conditional acceptance within a draft prefix has probability p.
"""

import bisect
import csv
import math
import random
import statistics
from pathlib import Path
from dataclasses import dataclass


def fit_acceptance(draft_length: int, mean: float) -> tuple[float, tuple[float, ...]]:
    """Fit constant conditional acceptance; the mean includes the verifier token."""
    if draft_length < 0 or not math.isfinite(mean) or not 1 <= mean <= draft_length + 1:
        raise ValueError("Require draft_length >= 0 and 1 <= mean <= draft_length + 1")
    low, high = 0.0, 1.0
    for _ in range(80):
        p = (low + high) / 2
        if sum(p**k for k in range(draft_length + 1)) < mean:
            low = p
        else:
            high = p
    p = 0.0 if mean == 1 else 1.0 if mean == draft_length + 1 else (low + high) / 2
    return p, tuple(p**k * (1 - p) for k in range(draft_length)) + (p**draft_length,)


def advance_history(history: int, accepted: int, tokens: int, window: int) -> int:
    return accepted if history + tokens > window else history + accepted


@dataclass(frozen=True)
class Metadata:
    history: int
    previous_accepted: int
    ring_start: int


@dataclass(frozen=True)
class StationaryWorkload:
    tokens: int
    window: int
    acceptance: tuple[float, ...]
    # Sorted (history, previous accepted, probability) states.
    joint: tuple[tuple[int, int, float], ...]
    balance_residual: float

    @property
    def mean_acceptance(self) -> float:
        return sum((i + 1) * mass for i, mass in enumerate(self.acceptance))

    @property
    def flush_share(self) -> float:
        return sum(mass for h, _, mass in self.joint if h + self.tokens > self.window)

    def draws(self, batch: int, draws: int, seed: int) -> list[list[Metadata]]:
        """Stratify over draws*batch, then shuffle to cover the law even at B1."""
        if batch < 1 or draws < 1:
            raise ValueError("batch and draws must be positive")
        rng = random.Random(seed)
        cdf, cumulative = [], 0.0
        for _, _, mass in self.joint:
            cumulative += mass
            cdf.append(cumulative)
        cdf[-1] = 1.0
        count = batch * draws
        result = []
        for index in range(count):
            quantile = (index + rng.random()) / count
            history, previous, _ = self.joint[bisect.bisect_left(cdf, quantile)]
            result.append(
                Metadata(history, previous, rng.randrange(self.window + self.tokens))
            )
        rng.shuffle(result)
        return [result[i : i + batch] for i in range(0, count, batch)]


def stationary_workload(
    acceptance: tuple[float, ...], window: int = 16
) -> StationaryWorkload:
    """Reuse the existing SSU Markov balance and retain the incoming acceptance."""
    import numpy as np
    from bench_ssu_checkpoint_mixed import markov_stationary

    tokens = len(acceptance)
    if not 1 <= tokens <= window or any(
        not math.isfinite(p) or p < 0 for p in acceptance
    ):
        raise ValueError("Invalid acceptance distribution/window")
    total = sum(acceptance)
    if total <= 0:
        raise ValueError("Acceptance law needs positive mass")
    acceptance = tuple(p / total for p in acceptance)
    marginal = markov_stationary(np.array((0.0, *acceptance)), tokens, window)
    joint = {}
    for history in range(1, window + 1):
        for accepted, probability in enumerate(acceptance, 1):
            key = advance_history(history, accepted, tokens, window), accepted
            joint[key] = joint.get(key, 0.0) + float(marginal[history]) * probability
    states = tuple((h, a, p) for (h, a), p in sorted(joint.items()) if p > 0)
    residual = max(
        abs(sum(p for h, _, p in states if h == i) - marginal[i])
        for i in range(1, window + 1)
    )
    if residual > 1e-12:
        raise ArithmeticError(f"Stationary balance residual is {residual}")
    return StationaryWorkload(tokens, window, acceptance, states, float(residual))


BOOTSTRAP_RESAMPLES = 4096
BOOTSTRAP_SEED = 51019


def iqm(values):
    """Central 50 percent mean with fractional endpoint weights."""
    ordered = sorted(values)
    if not ordered or not all(math.isfinite(value) for value in ordered):
        raise ValueError("IQM needs finite complete-pass observations")
    lower, upper = len(ordered) / 4, 3 * len(ordered) / 4
    total = sum(
        value * max(0.0, min(index + 1, upper) - max(index, lower))
        for index, value in enumerate(ordered)
    )
    return total / (upper - lower)


def paired_pass_iqm(pairs, *, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """IQM of paired PASS mean deltas and conservative within-process sensitivity.

    Each pair must already average ALL calls in a full workload pass. Bootstrap
    units are whole paired passes, and adjacent AB/BA two-pass cycles. This does
    not trim/reweight any raw token/history entry or imply fresh-run confidence.
    """
    pairs = sorted(pairs, key=lambda pair: pair["round"])
    if not pairs or len({pair["round"] for pair in pairs}) != len(pairs):
        raise ValueError("need unique complete paired-pass rounds")
    if type(resamples) is not int or resamples < 2:
        raise ValueError("bootstrap needs at least two resamples")
    deltas = [pair["us_saved"] for pair in pairs]
    count = len(deltas)
    if count % 2:
        raise ValueError("balanced AB/BA bootstrap needs an even paired-pass count")
    if [pair["round"] for pair in pairs] != list(range(count)):
        raise ValueError("balanced cycles require contiguous rounds starting at zero")
    rng = random.Random(seed)
    independent = [
        iqm([deltas[rng.randrange(count)] for _ in range(count)])
        for _ in range(resamples)
    ]
    cycles = [deltas[index : index + 2] for index in range(0, count, 2)]
    rng = random.Random(seed + 1)
    blocked = [
        iqm(
            [
                value
                for _ in range(len(cycles))
                for value in cycles[rng.randrange(len(cycles))]
            ]
        )
        for _ in range(resamples)
    ]
    pair_se, cycle_se = statistics.stdev(independent), statistics.stdev(blocked)
    return {
        "iqm_us_saved": iqm(deltas),
        "iqm_fractional_improvement": iqm(
            [pair["fractional_improvement"] for pair in pairs]
        ),
        "paired_pass_bootstrap_se_us": pair_se,
        "ab_ba_cycle_bootstrap_se_us": cycle_se,
        "conservative_bootstrap_se_us": max(pair_se, cycle_se),
        "paired_pass_count": count,
        "ab_ba_cycle_count": len(cycles),
        "trim_fraction_per_tail": 0.25,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "estimator": "IQM of whole paired-pass arithmetic-mean differences",
        "unit": (
            "within-process paired passes/cycles; "
            "not independent fresh-process confidence"
        ),
    }


def load_acceptance_csv(path: str | Path) -> tuple[float, ...]:
    """Read contiguous accepted-token bins; normalize counts or probabilities."""
    with Path(path).open() as file:
        rows = list(csv.DictReader(line for line in file if not line.startswith("#")))
    if [int(row["accepted_tokens"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("accepted_tokens must be consecutive bins starting at 1")
    weights = tuple(float(row["probability"]) for row in rows)
    if (
        not weights
        or any(not math.isfinite(p) or p < 0 for p in weights)
        or sum(weights) <= 0
    ):
        raise ValueError("acceptance weights must be finite, nonnegative and nonzero")
    return tuple(p / sum(weights) for p in weights)
