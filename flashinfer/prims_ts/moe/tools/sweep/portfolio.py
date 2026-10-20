# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-only portfolio selection for the Prims-TS MoE sweep.

This module deliberately knows nothing about CUDA, tensors, or the unfinished
sweep driver.  Candidate and shape identifiers are opaque hashable values, so
the search driver can use compile-signature hashes while tests can use strings.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias


Identifier: TypeAlias = Hashable


class TimingKind(str, Enum):
    """State of one candidate/shape measurement."""

    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"


@dataclass(frozen=True)
class TimingValue:
    """A measured time or an explicit non-measurement state."""

    kind: TimingKind
    time: float | None = None
    spread: float = 0.0

    def __post_init__(self) -> None:
        if self.kind is TimingKind.MEASURED:
            if self.time is None or not math.isfinite(self.time) or self.time <= 0:
                raise ValueError("a measured time must be positive and finite")
            if not math.isfinite(self.spread) or self.spread < 0:
                raise ValueError("spread must be nonnegative and finite")
        elif self.time is not None:
            raise ValueError("non-measured timing values cannot contain a time")

    @classmethod
    def measured(cls, time: float, spread: float = 0.0) -> TimingValue:
        return cls(TimingKind.MEASURED, float(time), float(spread))

    @classmethod
    def not_applicable(cls) -> TimingValue:
        return cls(TimingKind.NOT_APPLICABLE)

    @classmethod
    def failed(cls) -> TimingValue:
        return cls(TimingKind.FAILED)

    @property
    def is_measured(self) -> bool:
        return self.kind is TimingKind.MEASURED


NOT_APPLICABLE = TimingValue.not_applicable()
FAILED = TimingValue.failed()
TimingLike: TypeAlias = float | TimingValue | None


def as_timing(value: TimingLike) -> TimingValue:
    """Normalize the compact values accepted by the public helpers.

    ``None`` means that a config is not applicable.  Failures must be explicit
    via :data:`FAILED`, preserving the distinction in reports and search state.
    """

    if isinstance(value, TimingValue):
        return value
    if value is None:
        return NOT_APPLICABLE
    return TimingValue.measured(float(value))


@dataclass(frozen=True)
class Shape:
    """One weighted deployment shape in the portfolio objective."""

    shape_id: Identifier
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("shape weight must be positive and finite")


@dataclass(frozen=True)
class PortfolioCandidate:
    """A candidate config and the tile-N menu whose cap it consumes."""

    candidate_id: Identifier
    tile_n: Identifier


@dataclass(frozen=True)
class ShapeRegret:
    """Selected and globally best measurement for one shape."""

    shape_id: Identifier
    best_time: float
    selected_time: float | None
    regret: float


@dataclass(frozen=True)
class PortfolioSelection:
    """Result of cap-constrained greedy portfolio selection."""

    selected: tuple[Identifier, ...]
    by_tile_n: Mapping[Identifier, tuple[Identifier, ...]]
    marginal_contributions: Mapping[Identifier, float]
    weighted_log_regret: float
    average_regret: float
    worst_regret: float
    per_shape: Mapping[Identifier, ShapeRegret]


def _measured_time(
    timings: Mapping[Identifier, Mapping[Identifier, TimingLike]],
    candidate_id: Identifier,
    shape_id: Identifier,
) -> float | None:
    value = as_timing(timings.get(candidate_id, {}).get(shape_id))
    return value.time if value.is_measured else None


def best_times(
    candidates: Sequence[PortfolioCandidate],
    shapes: Sequence[Shape],
    timings: Mapping[Identifier, Mapping[Identifier, TimingLike]],
) -> dict[Identifier, float]:
    """Return the best successful measurement found for every covered shape."""

    result: dict[Identifier, float] = {}
    for shape in shapes:
        measured = [
            time
            for candidate in candidates
            if (time := _measured_time(timings, candidate.candidate_id, shape.shape_id))
            is not None
        ]
        if measured:
            result[shape.shape_id] = min(measured)
    return result


def _objective(
    selected: Iterable[Identifier],
    shapes: Sequence[Shape],
    timings: Mapping[Identifier, Mapping[Identifier, TimingLike]],
    global_best: Mapping[Identifier, float],
    regret_cap: float,
) -> float:
    selected = tuple(selected)
    total = 0.0
    for shape in shapes:
        best = global_best.get(shape.shape_id)
        if best is None:
            continue
        selected_times = [
            time
            for candidate_id in selected
            if (time := _measured_time(timings, candidate_id, shape.shape_id))
            is not None
        ]
        log_regret = (
            min(math.log(min(selected_times) / best), regret_cap)
            if selected_times
            else regret_cap
        )
        total += shape.weight * max(0.0, log_regret)
    return total


def _caps(
    candidates: Sequence[PortfolioCandidate],
    menu_size: int | Mapping[Identifier, int],
) -> dict[Identifier, int]:
    tile_ns = {candidate.tile_n for candidate in candidates}
    if isinstance(menu_size, int):
        if menu_size < 0:
            raise ValueError("menu_size must be nonnegative")
        return {tile_n: menu_size for tile_n in tile_ns}
    result = {}
    for tile_n in tile_ns:
        cap = int(menu_size.get(tile_n, 0))
        if cap < 0:
            raise ValueError("menu caps must be nonnegative")
        result[tile_n] = cap
    return result


def select_greedy_menu(
    candidates: Sequence[PortfolioCandidate],
    shapes: Sequence[Shape],
    timings: Mapping[Identifier, Mapping[Identifier, TimingLike]],
    *,
    menu_size: int | Mapping[Identifier, int] = 4,
    min_gain: float | None = None,
    noise_multiplier: float = 1.0,
    regret_cap: float = math.log(10.0),
) -> PortfolioSelection:
    """Select a weighted log-regret menu with an independent cap per tile N.

    Missing, inapplicable, and failed values receive ``regret_cap`` in the
    objective.  By default the menu grows while an entry improves at least one
    shape by more than the combined timing spread.  ``min_gain`` restores an
    explicit average weighted log-regret threshold when requested.
    """

    if min_gain is not None and (min_gain < 0 or not math.isfinite(min_gain)):
        raise ValueError("min_gain must be nonnegative and finite")
    if noise_multiplier < 0 or not math.isfinite(noise_multiplier):
        raise ValueError("noise_multiplier must be nonnegative and finite")
    if regret_cap <= 0 or not math.isfinite(regret_cap):
        raise ValueError("regret_cap must be positive and finite")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate identifiers must be unique")
    shape_ids = [shape.shape_id for shape in shapes]
    if len(set(shape_ids)) != len(shape_ids):
        raise ValueError("shape identifiers must be unique")

    global_best = best_times(candidates, shapes, timings)
    relevant_weight = sum(
        shape.weight for shape in shapes if shape.shape_id in global_best
    )
    caps = _caps(candidates, menu_size)
    used = {tile_n: 0 for tile_n in caps}
    selected: list[Identifier] = []
    objective = _objective(selected, shapes, timings, global_best, regret_cap)

    def improves_beyond_noise(candidate_id: Identifier) -> bool:
        for shape in shapes:
            candidate_value = as_timing(
                timings.get(candidate_id, {}).get(shape.shape_id)
            )
            if not candidate_value.is_measured:
                continue
            selected_values = [
                as_timing(timings.get(selected_id, {}).get(shape.shape_id))
                for selected_id in selected
            ]
            selected_values = [value for value in selected_values if value.is_measured]
            if not selected_values:
                return True
            current = min(selected_values, key=lambda value: float(value.time))
            assert candidate_value.time is not None and current.time is not None
            noise = noise_multiplier * math.hypot(
                candidate_value.spread, current.spread
            )
            if candidate_value.time + noise < current.time:
                return True
        return False

    while relevant_weight:
        winner: PortfolioCandidate | None = None
        winner_objective = objective
        winner_gain = 0.0
        for candidate in candidates:
            if candidate.candidate_id in selected:
                continue
            if used[candidate.tile_n] >= caps[candidate.tile_n]:
                continue
            if min_gain is None and not improves_beyond_noise(candidate.candidate_id):
                continue
            trial_objective = _objective(
                (*selected, candidate.candidate_id),
                shapes,
                timings,
                global_best,
                regret_cap,
            )
            gain = (objective - trial_objective) / relevant_weight
            # Candidate order is the deterministic tie breaker.
            if gain > winner_gain + 1e-15:
                winner = candidate
                winner_objective = trial_objective
                winner_gain = gain
        if winner is None or (min_gain is not None and winner_gain + 1e-15 < min_gain):
            break
        selected.append(winner.candidate_id)
        used[winner.tile_n] += 1
        objective = winner_objective

    contributions = marginal_contributions(
        selected,
        shapes,
        timings,
        global_best=global_best,
        regret_cap=regret_cap,
    )
    per_shape: dict[Identifier, ShapeRegret] = {}
    weighted_regret = 0.0
    worst_regret = 0.0
    uncovered = False
    for shape in shapes:
        best = global_best.get(shape.shape_id)
        if best is None:
            continue
        selected_times = [
            time
            for candidate_id in selected
            if (time := _measured_time(timings, candidate_id, shape.shape_id))
            is not None
        ]
        if selected_times:
            chosen = min(selected_times)
            regret = max(0.0, chosen / best - 1.0)
            weighted_regret += shape.weight * regret
            worst_regret = max(worst_regret, regret)
        else:
            chosen = None
            regret = math.inf
            uncovered = True
        per_shape[shape.shape_id] = ShapeRegret(shape.shape_id, best, chosen, regret)

    by_tile: dict[Identifier, list[Identifier]] = {tile_n: [] for tile_n in caps}
    tile_by_id = {candidate.candidate_id: candidate.tile_n for candidate in candidates}
    for candidate_id in selected:
        by_tile[tile_by_id[candidate_id]].append(candidate_id)
    average_regret = (
        math.inf
        if uncovered
        else weighted_regret / relevant_weight
        if relevant_weight
        else 0.0
    )
    return PortfolioSelection(
        selected=tuple(selected),
        by_tile_n={key: tuple(value) for key, value in by_tile.items()},
        marginal_contributions=contributions,
        weighted_log_regret=objective / relevant_weight if relevant_weight else 0.0,
        average_regret=average_regret,
        worst_regret=math.inf if uncovered else worst_regret,
        per_shape=per_shape,
    )


def marginal_contributions(
    menu: Sequence[Identifier],
    shapes: Sequence[Shape],
    timings: Mapping[Identifier, Mapping[Identifier, TimingLike]],
    *,
    global_best: Mapping[Identifier, float] | None = None,
    regret_cap: float = math.log(10.0),
) -> dict[Identifier, float]:
    """Compute each entry's leave-one-out weighted log-regret contribution."""

    if global_best is None:
        pseudo_candidates = [
            PortfolioCandidate(candidate_id, 0) for candidate_id in timings.keys()
        ]
        global_best = best_times(pseudo_candidates, shapes, timings)
    relevant_weight = sum(
        shape.weight for shape in shapes if shape.shape_id in global_best
    )
    if not relevant_weight:
        return {candidate_id: 0.0 for candidate_id in menu}
    full = _objective(menu, shapes, timings, global_best, regret_cap)
    return {
        candidate_id: max(
            0.0,
            (
                _objective(
                    [other for other in menu if other != candidate_id],
                    shapes,
                    timings,
                    global_best,
                    regret_cap,
                )
                - full
            )
            / relevant_weight,
        )
        for candidate_id in menu
    }


@dataclass(frozen=True)
class RetimePool:
    """Deduplicated candidates selected for the final high-precision timing."""

    candidates: tuple[Identifier, ...]
    reasons: Mapping[Identifier, tuple[str, ...]] = field(default_factory=dict)


def build_final_retime_pool(
    candidates: Sequence[PortfolioCandidate],
    shapes: Sequence[Shape],
    timings: Mapping[Identifier, Mapping[Identifier, TimingLike]],
    *,
    top_k: int = 5,
    interim_menu: Iterable[Identifier] = (),
    contributors: Iterable[Identifier] | Mapping[Identifier, float] = (),
    incumbents: Iterable[Identifier] = (),
    near_best_fraction: float = 0.03,
    max_near_best_per_shape: int | None = 10,
) -> RetimePool:
    """Build a bounded, priority-ordered final re-timing pool.

    Explicit portfolio entries and incumbents come first.  Per-shape top-k and
    near-best candidates are then interleaved by rank so a deadline cannot let
    one flat shape consume the whole re-timing budget.
    """

    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    if near_best_fraction < 0 or not math.isfinite(near_best_fraction):
        raise ValueError("near_best_fraction must be nonnegative and finite")
    if max_near_best_per_shape is not None and max_near_best_per_shape < 0:
        raise ValueError("max_near_best_per_shape must be nonnegative")
    reasons: dict[Identifier, set[str]] = {}
    ordered: list[Identifier] = []
    ordered_set: set[Identifier] = set()

    def add(candidate_id: Identifier, reason: str) -> None:
        reasons.setdefault(candidate_id, set()).add(reason)
        if candidate_id not in ordered_set:
            ordered.append(candidate_id)
            ordered_set.add(candidate_id)

    # These entries already affect the deployable portfolio and therefore get
    # first claim on a bounded finalization tail.
    for candidate_id in interim_menu:
        add(candidate_id, "interim_menu")
    contributor_ids = (
        contributors.keys() if isinstance(contributors, Mapping) else contributors
    )
    for candidate_id in contributor_ids:
        add(candidate_id, "contributor")
    for candidate_id in incumbents:
        add(candidate_id, "incumbent")

    ranked_by_shape: list[tuple[Identifier, list[tuple[float, int, Identifier]]]] = []
    for shape in shapes:
        ranked = sorted(
            (
                (time, order, candidate.candidate_id)
                for order, candidate in enumerate(candidates)
                if (
                    time := _measured_time(
                        timings, candidate.candidate_id, shape.shape_id
                    )
                )
                is not None
            ),
            key=lambda item: (item[0], item[1]),
        )
        if not ranked:
            continue
        ranked_by_shape.append((shape.shape_id, ranked))

    # Round-robin by rank preserves representation from every shape when the
    # hard sweep deadline truncates final re-timing.
    for rank in range(top_k):
        for shape_id, ranked in ranked_by_shape:
            if rank < len(ranked):
                add(ranked[rank][2], f"top_k:{shape_id}")

    near_best_by_shape: list[tuple[Identifier, list[Identifier]]] = []
    for shape_id, ranked in ranked_by_shape:
        limit = ranked[0][0] * (1.0 + near_best_fraction)
        near_best = [candidate_id for time, _, candidate_id in ranked if time <= limit]
        if max_near_best_per_shape is not None:
            near_best = near_best[:max_near_best_per_shape]
        near_best_by_shape.append((shape_id, near_best))
    max_near_best = max(
        (len(candidate_ids) for _, candidate_ids in near_best_by_shape),
        default=0,
    )
    for rank in range(max_near_best):
        for shape_id, candidate_ids in near_best_by_shape:
            if rank < len(candidate_ids):
                add(candidate_ids[rank], f"near_best:{shape_id}")

    return RetimePool(
        candidates=tuple(ordered),
        reasons={
            candidate_id: tuple(sorted(candidate_reasons))
            for candidate_id, candidate_reasons in reasons.items()
        },
    )
