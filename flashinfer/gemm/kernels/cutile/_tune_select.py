# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deterministic ranking of cuTile ``exhaustive_search`` measurements.

``exhaustive_search`` stops sampling a candidate once its error margin drops
below max(1% of mean, 0.5 us). For fast kernels this margin is of the same
order as the gap between the top candidates, so a raw argmin over the mean
latencies picks a different winner from process to process. This module
breaks such statistical ties with a deterministic config key so that repeated
tuning of the same shape on the same GPU selects the same kernel.
"""

from typing import Any, Callable, Sequence

# Two configs whose means differ by less than this fraction are considered
# tied even when their confidence intervals do not overlap: run-to-run drift
# of the mean itself is of this order, so ranking inside the band is noise.
TIE_REL_TOL = 0.02


def rank_measurements(
    successes: Sequence[Any],
    config_key: Callable[[Any], tuple],
) -> list:
    """Rank measurements fastest first, ordering ties by ``config_key``.

    A measurement is tied with the fastest one when the gap between their
    means is within the combined 95% error margins or within ``TIE_REL_TOL``
    of the best mean. Tied measurements are ordered by ``config_key`` so the
    selection is stable across runs; the rest keep the latency order.

    ``config_key`` must map a config to a tuple of primitives forming a total
    order. It only ever reorders statistically indistinguishable candidates,
    so it cannot trade away measurable performance.
    """
    by_time = sorted(successes, key=lambda m: m.mean_us)
    best = by_time[0]
    tol = max(best.error_margin_us, TIE_REL_TOL * best.mean_us)

    def _tied(m):
        return m.mean_us - best.mean_us <= m.error_margin_us + tol

    band = [m for m in by_time if _tied(m)]
    rest = [m for m in by_time if not _tied(m)]
    return sorted(band, key=lambda m: config_key(m.config)) + rest
