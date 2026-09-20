"""Pre-declared statistics for the sampling quality fuzzer.

The distribution layer compares ``N`` independent trial outcomes against the oracle's
per-class probability ``p`` and accepts a class when

    |p_hat - p| <= sqrt(log(2 * K / alpha) / (2 * N))

where ``K`` is the number of (config x effective class) comparisons in the whole round and
``alpha`` is the whole-round false-positive budget.  Both are declared in advance (here and in
``test_sampling_quality_fuzz.py``), never derived from the data.

The bound is the conservative bounded-Bernoulli plan: for one class Hoeffding gives
``P(|p_hat - p| > hw) <= 2 exp(-2 N hw^2)``, and ``hw = sqrt(log(2K/alpha) / (2N))`` makes that
``<= alpha / K``, so a union bound over the K comparisons caps the whole round at ``alpha``
without assuming anything about how the classes are correlated.

``calibrate()`` measures both sides of that plan against pure reference sampling (an explicit
inverse-CDF draw -- no kernel, no torch): the false-positive rate under the null, and the
detection power against a known injected bias, including the injected class permutation that a
wrong row mapping produces.  It imports only the standard library, so the numbers recorded in
NOTES.md are reproducible on a box with neither torch nor numpy:

    python3 tests/sampling/sampling_stats.py
"""

from __future__ import annotations

import bisect
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# Whole-round false-positive budget (declared in advance).
ALPHA = 0.01
# Independent trials per distribution case in the fuzz round (declared in advance).
N_TRIALS = 200_000
# Fraction of probability mass swapped between two classes by the reference bias injection.
INJECTED_DELTA = 0.02


def half_width(n_trials: int, n_comparisons: int, alpha: float = ALPHA) -> float:
    """Acceptance half-width for one class, from the whole-round budget."""
    if n_trials <= 0 or n_comparisons <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError(
            f"need n_trials>0, n_comparisons>0, 0<alpha<1; got "
            f"{n_trials}, {n_comparisons}, {alpha}"
        )
    return math.sqrt(math.log(2.0 * n_comparisons / alpha) / (2.0 * n_trials))


def violations(
    observed: Sequence[float], expected: Sequence[float], hw: float
) -> List[int]:
    """Classes whose observed frequency leaves the declared band."""
    return [c for c in range(len(expected)) if abs(observed[c] - expected[c]) > hw]


def draw_counts(p: Sequence[float], n_trials: int, rng: random.Random) -> List[int]:
    """Explicit inverse-CDF sampling from ``p`` (no torch, no multinomial)."""
    cdf: List[float] = []
    acc = 0.0
    for x in p:
        acc += x
        cdf.append(acc)
    counts = [0] * len(p)
    total = cdf[-1]
    for _ in range(n_trials):
        counts[bisect.bisect_right(cdf, rng.random() * total)] += 1
    return counts


def swap_mass(p: Sequence[float], delta: float) -> List[float]:
    """Shift ``delta`` of probability from the largest class to the next largest."""
    order = sorted(range(len(p)), key=lambda i: (-p[i], i))
    out = list(p)
    out[order[0]] -= delta
    out[order[1]] += delta
    return out


@dataclass(frozen=True)
class Calibration:
    """Measured behaviour of the acceptance rule (recorded by the test and in NOTES.md)."""

    n_trials: int
    n_comparisons: int
    alpha: float
    reps: int
    hw: float
    fp_reps: int  # replications with at least one class outside the band
    fp_comparisons: int  # individual class comparisons outside the band
    comparisons: int
    power_permutation: float  # rejections of a permuted class mapping
    power_curve: Tuple[Tuple[float, float], ...]  # (injected delta, rejection rate)

    @property
    def fp_rate(self) -> float:
        return self.fp_reps / self.reps

    @property
    def per_comparison_fp(self) -> float:
        return self.fp_comparisons / self.comparisons

    @property
    def detection_floor(self) -> float:
        """Smallest injected delta whose measured rejection rate is >= 0.5."""
        hits = [d for d, rate in self.power_curve if rate >= 0.5]
        return min(hits) if hits else float("nan")

    def power_at(self, delta: float) -> float:
        """Rejection rate measured for exactly this injected delta."""
        for d, rate in self.power_curve:
            if d == delta:
                return rate
        raise KeyError(f"delta {delta!r} not in the measured power curve")

    def as_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "n_trials": self.n_trials,
            "n_comparisons": self.n_comparisons,
            "alpha": self.alpha,
            "reps": self.reps,
            "half_width": round(self.hw, 6),
            "false_positive_replications": self.fp_reps,
            "false_positive_rate": self.fp_rate,
            "per_comparison_false_positive_rate": round(self.per_comparison_fp, 8),
            "per_comparison_budget": round(self.alpha / self.n_comparisons, 8),
            "power_permuted_mapping": self.power_permutation,
            "detection_floor": self.detection_floor,
        }
        for delta, rate in self.power_curve:
            out[f"power_injected_delta_{delta:g}"] = rate
        return out


def calibrate(
    p: Sequence[float],
    *,
    n_trials: int = 50_000,
    n_comparisons: int,
    alpha: float = ALPHA,
    reps: int = 40,
    deltas: Sequence[float] = (0.0, 0.003, 0.004, 0.005, 0.008, INJECTED_DELTA),
    seed: int = 20260919,
) -> Calibration:
    """Measure the acceptance rule's false-positive rate and power by reference sampling.

    Each replication draws ``n_trials`` independent outcomes from a reference distribution and
    applies the rule to all ``len(p)`` classes, exactly as the GPU distribution layer does.
    ``delta = 0`` is the null (its rejection rate is the false-positive rate); the remaining
    deltas inject a known mass shift.  A permuted class mapping (the signature of a wrong row
    mapping) is measured by applying the rule to the reversed observation vector, which needs
    no extra draw.
    """
    hw = half_width(n_trials, n_comparisons, alpha)
    rng = random.Random(seed)
    injections = [(d, swap_mass(p, d)) for d in deltas]
    fp_reps = fp_comparisons = 0
    power_perm = 0
    power_hits = [0] * len(injections)
    for _ in range(reps):
        for slot, (delta, q) in enumerate(injections):
            counts = draw_counts(q, n_trials, rng)
            observed = [c / n_trials for c in counts]
            # Always judged against the DECLARED reference p: for delta == 0 that is the null
            # (so this counts false positives), for delta > 0 it is the injected bias the rule
            # is supposed to catch.
            bad = violations(observed, p, hw)
            if delta == 0.0:
                fp_reps += int(bool(bad))
                fp_comparisons += len(bad)
            else:
                power_hits[slot] += int(bool(bad))
        counts = draw_counts(p, n_trials, rng)
        power_perm += int(
            bool(violations([c / n_trials for c in reversed(counts)], p, hw))
        )
    return Calibration(
        n_trials=n_trials,
        n_comparisons=n_comparisons,
        alpha=alpha,
        reps=reps,
        hw=hw,
        fp_reps=fp_reps,
        fp_comparisons=fp_comparisons,
        comparisons=reps * len(p),
        power_permutation=power_perm / reps,
        power_curve=tuple(
            (delta, power_hits[i] / reps) for i, (delta, _) in enumerate(injections)
        ),
    )


if __name__ == "__main__":
    # The recorded calibration: the fuzz round's own N/K/alpha, and the reference support the
    # distribution layer's smallest case uses (8 classes, one of them exactly zero, so the run
    # also exercises the "a zero-probability class is never drawn" rule).
    REFERENCE_P = (0.28, 0.22, 0.17, 0.12, 0.09, 0.07, 0.05, 0.0)
    hw = half_width(N_TRIALS, 44)
    result = calibrate(
        REFERENCE_P,
        n_trials=N_TRIALS,
        n_comparisons=44,
        reps=30,
        deltas=(0.0, 0.5 * hw, hw, 2.0 * hw, INJECTED_DELTA),
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
