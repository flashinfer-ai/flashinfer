"""
Copyright (c) 2026 by FlashInfer team.

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

import math


def required_num_trials(
    expected_probs,
    cosine_threshold=0.99,
    compare_two_samples=True,
    safety_factor=1.25,
    min_trials=10_000,
):
    """Size a cosine-similarity frequency check for one categorical distribution.

    Pass probabilities after any filtering. ``compare_two_samples=False``
    compares empirical frequencies with known probabilities; the default
    compares two independent empirical frequency vectors. The variance-based
    approximation uses a safety margin and a floor, not a tail bound.
    """
    probs = expected_probs.float()
    probs = probs / probs.sum()

    sum_p2 = probs.square().sum().item()
    effective_support = 1.0 / sum_p2

    if compare_two_samples:
        multiplier = cosine_threshold / (1.0 - cosine_threshold)
    else:
        multiplier = cosine_threshold**2 / (1.0 - cosine_threshold**2)

    return max(
        min_trials,
        math.ceil(safety_factor * multiplier * effective_support),
    )


def required_num_collision_rows(
    expected_probs, max_failure_probability=1e-12, comparisons=2
):
    """Select a row prefix unlikely to match completely in independent draws.

    Each row's collision probability is sum(p**2), the inverse effective
    support. Their product bounds an all-match assertion's false failure;
    a union bound covers ``comparisons`` pairs, even with a shared baseline.
    Pass probabilities after filtering. The independence model does not
    guarantee detection of partial RNG defects or preserve batch coverage.
    """
    if not 0 < max_failure_probability < 1 or comparisons < 1:
        raise ValueError("Require a failure probability in (0, 1) and comparisons >= 1")
    if expected_probs.ndim != 2 or expected_probs.shape[0] == 0:
        raise ValueError("Expected a nonempty matrix of row probabilities")

    probs = expected_probs.double()
    row_mass = probs.sum(dim=-1, keepdim=True)
    valid = (
        probs.isfinite().all()
        & (probs >= 0).all()
        & row_mass.isfinite().all()
        & (row_mass > 0).all()
    )
    if not valid:
        raise ValueError(
            "Require finite nonnegative probabilities with positive row mass"
        )
    probs = probs / row_mass
    log_all_match = probs.square().sum(dim=-1).clamp(max=1).log().cumsum(0)
    sufficient = (
        log_all_match <= math.log(max_failure_probability) - math.log(comparisons)
    ).nonzero()
    if len(sufficient) == 0:
        raise ValueError("Candidate rows do not meet the all-match failure budget")
    return int(sufficient[0].item()) + 1
