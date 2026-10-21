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
