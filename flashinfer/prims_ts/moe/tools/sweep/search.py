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

"""Host-safe adaptive-search primitives for the Prims-TS MoE sweep."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, TypeAlias

from .portfolio import PortfolioCandidate, Shape, TimingLike, TimingValue, as_timing


Identifier: TypeAlias = Hashable


class CandidateOutcome(str, Enum):
    """Noise-aware outcome used by move arms and two-step exploration."""

    SUCCESS = "success"
    TWO_STEP = "two_step"
    MISS = "miss"
    FAILED = "failed"


class RejectionReason(str, Enum):
    """Stable reason names for proposals that were not evaluated."""

    ILLEGAL = "illegal"
    NO_EFFECTIVE_MUTATION = "no_effective_mutation"
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    DUPLICATE_SIGNATURE = "duplicate_signature"
    EXHAUSTED_PARENT_MOVE = "exhausted_parent_move"
    NO_LEGAL_PARENT = "no_legal_parent"
    NO_LEGAL_NEIGHBOR = "no_legal_neighbor"
    VALIDATION_ERROR = "validation_error"


@dataclass(frozen=True)
class NoiseAwareThresholds:
    """Regret bands with an allowance for measured timing spread."""

    success_fraction: float = 0.03
    two_step_fraction: float = 0.10
    noise_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.success_fraction)
            or self.success_fraction < 0
            or not math.isfinite(self.two_step_fraction)
            or self.two_step_fraction < self.success_fraction
            or not math.isfinite(self.noise_multiplier)
            or self.noise_multiplier < 0
        ):
            raise ValueError("invalid success, two-step, or noise threshold")

    def classify_one(
        self,
        candidate: float | TimingValue,
        best: float | TimingValue,
    ) -> CandidateOutcome:
        """Classify one shape, expanding both bands by relative timing noise."""

        candidate_value = as_timing(candidate)
        best_value = as_timing(best)
        if not candidate_value.is_measured or not best_value.is_measured:
            return CandidateOutcome.FAILED
        assert candidate_value.time is not None and best_value.time is not None
        relative_noise = self.noise_multiplier * math.hypot(
            candidate_value.spread / candidate_value.time,
            best_value.spread / best_value.time,
        )
        regret = candidate_value.time / best_value.time - 1.0
        if regret <= self.success_fraction + relative_noise:
            return CandidateOutcome.SUCCESS
        if regret <= self.two_step_fraction + relative_noise:
            return CandidateOutcome.TWO_STEP
        return CandidateOutcome.MISS

    def classify(
        self,
        candidate_times: Mapping[Identifier, TimingLike],
        best_times: Mapping[Identifier, TimingLike],
    ) -> CandidateOutcome:
        """Classify a candidate by its best outcome on any measured shape."""

        outcomes = [
            self.classify_one(candidate_times[shape_id], best)
            for shape_id, best in best_times.items()
            if shape_id in candidate_times
        ]
        if CandidateOutcome.SUCCESS in outcomes:
            return CandidateOutcome.SUCCESS
        if CandidateOutcome.TWO_STEP in outcomes:
            return CandidateOutcome.TWO_STEP
        if CandidateOutcome.MISS in outcomes:
            return CandidateOutcome.MISS
        return CandidateOutcome.FAILED

    def classify_against_parent(
        self,
        candidate_times: Mapping[Identifier, TimingLike],
        parent_times: Mapping[Identifier, TimingLike],
        best_times: Mapping[Identifier, TimingLike],
        *,
        parent_improvement_fraction: float = 0.005,
    ) -> CandidateOutcome:
        """Reward only near-best children that materially improve their parent.

        The broader two-step band remains relative to the best known timing so
        promising intermediate configurations can still seed a short lineage.
        """

        if (
            not math.isfinite(parent_improvement_fraction)
            or parent_improvement_fraction < 0
            or parent_improvement_fraction >= 1
        ):
            raise ValueError("parent_improvement_fraction must be in [0, 1)")
        for shape_id, best in best_times.items():
            if shape_id not in candidate_times or shape_id not in parent_times:
                continue
            candidate = as_timing(candidate_times[shape_id])
            parent = as_timing(parent_times[shape_id])
            best_value = as_timing(best)
            if not (
                candidate.is_measured and parent.is_measured and best_value.is_measured
            ):
                continue
            assert (
                candidate.time is not None
                and parent.time is not None
                and best_value.time is not None
            )
            parent_noise = self.noise_multiplier * math.hypot(
                candidate.spread / candidate.time,
                parent.spread / parent.time,
            )
            parent_improvement = (parent.time - candidate.time) / parent.time
            near_best = candidate.time / best_value.time - 1.0 <= self.success_fraction
            if (
                near_best
                and parent_improvement >= parent_improvement_fraction + parent_noise
            ):
                return CandidateOutcome.SUCCESS

        broad_outcome = self.classify(candidate_times, best_times)
        if broad_outcome is CandidateOutcome.SUCCESS:
            return CandidateOutcome.TWO_STEP
        return broad_outcome


@dataclass
class BetaArm:
    """Bernoulli arm accounting used for one move kind in one search group."""

    attempts: int = 0
    successes: int = 0
    failures: int = 0
    rejected: int = 0
    duplicates: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            min(
                self.attempts,
                self.successes,
                self.failures,
                self.rejected,
                self.duplicates,
            )
            < 0
        ):
            raise ValueError("arm counts must be nonnegative")
        if self.successes + self.failures != self.attempts:
            raise ValueError("attempts must equal successes plus failures")
        if any(count < 0 for count in self.rejection_reasons.values()):
            raise ValueError("rejection-reason counts must be nonnegative")
        if sum(self.rejection_reasons.values()) > self.rejected:
            raise ValueError("rejection-reason counts cannot exceed rejected")

    def record(self, success: bool) -> None:
        self.attempts += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1

    def record_outcome(self, outcome: CandidateOutcome) -> None:
        """Only a caller-classified success updates the positive posterior."""

        self.record(outcome is CandidateOutcome.SUCCESS)

    def record_rejected(self, reason: RejectionReason | str | None = None) -> None:
        self.rejected += 1
        if reason is not None:
            key = reason.value if isinstance(reason, RejectionReason) else str(reason)
            self.rejection_reasons[key] = self.rejection_reasons.get(key, 0) + 1

    def record_duplicate(self) -> None:
        self.duplicates += 1

    @property
    def proposed(self) -> int:
        return self.attempts + self.rejected + self.duplicates

    @property
    def evaluation_yield(self) -> float:
        return self.attempts / self.proposed if self.proposed else 0.0

    @property
    def duplicate_rate(self) -> float:
        return self.duplicates / self.proposed if self.proposed else 0.0

    @property
    def duplicate_penalty(self) -> float:
        """Smoothed novelty probability used to down-rank duplicate-heavy arms."""

        return (1.0 + self.proposed - self.duplicates) / (1.0 + self.proposed)

    def sample(
        self,
        rng: random.Random,
        *,
        prior_success: float = 1.0,
        prior_failure: float = 1.0,
    ) -> float:
        if prior_success <= 0 or prior_failure <= 0:
            raise ValueError("beta prior parameters must be positive")
        return rng.betavariate(
            prior_success + self.successes,
            prior_failure + self.failures,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "proposed": self.proposed,
            "evaluation_yield": self.evaluation_yield,
            "duplicate_rate": self.duplicate_rate,
            "duplicate_penalty": self.duplicate_penalty,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BetaArm:
        return cls(
            attempts=int(value["attempts"]),
            successes=int(value["successes"]),
            failures=int(value["failures"]),
            rejected=int(value.get("rejected", 0)),
            duplicates=int(value.get("duplicates", 0)),
            rejection_reasons={
                str(reason): int(count)
                for reason, count in value.get("rejection_reasons", {}).items()
            },
        )


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


class AdaptiveMoveSelector:
    """Per-group Beta Thompson sampler with uniform exploration."""

    def __init__(
        self,
        *,
        seed: int = 0,
        exploration_floor: float = 0.05,
        prior_success: float = 1.0,
        prior_failure: float = 1.0,
    ) -> None:
        if not 0.0 <= exploration_floor <= 1.0:
            raise ValueError("exploration_floor must be in [0, 1]")
        if prior_success <= 0 or prior_failure <= 0:
            raise ValueError("beta prior parameters must be positive")
        self.exploration_floor = float(exploration_floor)
        self.prior_success = float(prior_success)
        self.prior_failure = float(prior_failure)
        self.rng = random.Random(seed)
        self.arms: dict[str, dict[str, BetaArm]] = {}

    def arm(self, group: str, move_kind: str) -> BetaArm:
        return self.arms.setdefault(group, {}).setdefault(move_kind, BetaArm())

    def choose(self, group: str, move_kinds: Sequence[str]) -> str:
        """Draw one legal move, using candidate order as the score tie-breaker."""

        if not move_kinds:
            raise ValueError("move_kinds cannot be empty")
        if len(set(move_kinds)) != len(move_kinds):
            raise ValueError("move_kinds must be unique")
        for move_kind in move_kinds:
            self.arm(group, move_kind)
        if self.rng.random() < self.exploration_floor:
            return move_kinds[self.rng.randrange(len(move_kinds))]
        samples = [
            self.arm(group, move_kind).sample(
                self.rng,
                prior_success=self.prior_success,
                prior_failure=self.prior_failure,
            )
            * self.arm(group, move_kind).duplicate_penalty
            for move_kind in move_kinds
        ]
        return move_kinds[max(range(len(move_kinds)), key=samples.__getitem__)]

    def record(
        self,
        group: str,
        move_kind: str,
        outcome: CandidateOutcome | bool,
    ) -> None:
        arm = self.arm(group, move_kind)
        if isinstance(outcome, bool):
            arm.record(outcome)
        else:
            arm.record_outcome(outcome)

    def record_rejected(
        self,
        group: str,
        move_kind: str,
        reason: RejectionReason | str | None = None,
    ) -> None:
        self.arm(group, move_kind).record_rejected(reason)

    def record_duplicate(self, group: str, move_kind: str) -> None:
        self.arm(group, move_kind).record_duplicate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "exploration_floor": self.exploration_floor,
            "prior_success": self.prior_success,
            "prior_failure": self.prior_failure,
            "rng_state": self.rng.getstate(),
            "arms": {
                group: {
                    move_kind: arm.to_dict()
                    for move_kind, arm in sorted(group_arms.items())
                }
                for group, group_arms in sorted(self.arms.items())
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AdaptiveMoveSelector:
        result = cls(
            exploration_floor=float(value["exploration_floor"]),
            prior_success=float(value["prior_success"]),
            prior_failure=float(value["prior_failure"]),
        )
        result.rng.setstate(_nested_tuple(value["rng_state"]))
        result.arms = {
            str(group): {
                str(move_kind): BetaArm.from_dict(arm)
                for move_kind, arm in group_arms.items()
            }
            for group, group_arms in value.get("arms", {}).items()
        }
        return result


@dataclass
class ParentMoveTracker:
    """Track duplicate-heavy or proven-empty ``(parent, move)`` pairs."""

    duplicate_limit: int = 2
    exhausted: set[tuple[Identifier, str]] = field(default_factory=set)
    duplicate_counts: dict[tuple[Identifier, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.duplicate_limit <= 0:
            raise ValueError("duplicate_limit must be positive")
        if any(count < 0 for count in self.duplicate_counts.values()):
            raise ValueError("duplicate counts must be nonnegative")
        self.exhausted.update(
            key
            for key, count in self.duplicate_counts.items()
            if count >= self.duplicate_limit
        )

    def is_exhausted(self, parent_id: Identifier, move_kind: str) -> bool:
        return (parent_id, move_kind) in self.exhausted

    def mark_exhausted(self, parent_id: Identifier, move_kind: str) -> None:
        self.exhausted.add((parent_id, move_kind))

    def record_duplicate(self, parent_id: Identifier, move_kind: str) -> bool:
        """Record a duplicate and return whether the pair became exhausted."""

        key = (parent_id, move_kind)
        count = self.duplicate_counts.get(key, 0) + 1
        self.duplicate_counts[key] = count
        if count >= self.duplicate_limit:
            self.exhausted.add(key)
        return key in self.exhausted

    def available_moves(
        self, parent_id: Identifier, move_kinds: Sequence[str]
    ) -> tuple[str, ...]:
        return tuple(
            move_kind
            for move_kind in move_kinds
            if not self.is_exhausted(parent_id, move_kind)
        )

    def parent_is_exhausted(
        self, parent_id: Identifier, move_kinds: Sequence[str]
    ) -> bool:
        return not self.available_moves(parent_id, move_kinds)


@dataclass(frozen=True)
class ParentScore:
    """Why a measured config was retained as a local-search parent."""

    candidate_id: Identifier
    coverage_weight: float
    contribution: float
    score: float
    in_menu: bool
    mean_regret: float = math.inf
    is_fallback: bool = False


@dataclass(frozen=True)
class ParentPool:
    """Capped, ranked parent set and its diagnostics."""

    parents: tuple[Identifier, ...]
    scores: Mapping[Identifier, ParentScore]


def rank_parents_by_mean_regret(
    candidate_ids: Sequence[Identifier],
    shapes: Sequence[Shape],
    timings: Mapping[Identifier, Mapping[Identifier, TimingLike]],
    best_times: Mapping[Identifier, TimingLike],
    *,
    top_k: int,
) -> tuple[tuple[Identifier, float], ...]:
    """Return measured candidates with the lowest weighted mean regret."""

    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    shape_by_id = {shape.shape_id: shape for shape in shapes}
    ranked: list[tuple[float, int, Identifier]] = []
    for order, candidate_id in enumerate(candidate_ids):
        weighted_regret = 0.0
        measured_weight = 0.0
        for shape_id, best in best_times.items():
            shape = shape_by_id.get(shape_id)
            candidate = as_timing(timings.get(candidate_id, {}).get(shape_id))
            best_value = as_timing(best)
            if shape is None or not candidate.is_measured or not best_value.is_measured:
                continue
            assert candidate.time is not None and best_value.time is not None
            weighted_regret += shape.weight * max(
                0.0, candidate.time / best_value.time - 1.0
            )
            measured_weight += shape.weight
        if measured_weight:
            ranked.append((weighted_regret / measured_weight, order, candidate_id))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return tuple(
        (candidate_id, mean_regret)
        for mean_regret, _order, candidate_id in ranked[:top_k]
    )


def select_parent_pool(
    candidate_ids: Sequence[Identifier],
    shapes: Sequence[Shape],
    timings: Mapping[Identifier, Mapping[Identifier, TimingLike]],
    best_times: Mapping[Identifier, TimingLike],
    *,
    menu: Iterable[Identifier] = (),
    contributions: Mapping[Identifier, float] | None = None,
    max_parents: int = 64,
    near_best_fraction: float = 0.03,
    noise_multiplier: float = 1.0,
    contribution_weight: float = 1.0,
    fallback_top_k: int = 8,
    min_parents: int = 4,
) -> ParentPool:
    """Rank parents by coverage, falling back to the lowest mean regret."""

    if max_parents < 0:
        raise ValueError("max_parents must be nonnegative")
    if fallback_top_k < 0 or min_parents < 0:
        raise ValueError("fallback_top_k and min_parents must be nonnegative")
    if near_best_fraction < 0 or noise_multiplier < 0 or contribution_weight < 0:
        raise ValueError("parent-pool tuning values must be nonnegative")
    menu_set = set(menu)
    contributions = contributions or {}
    total_weight = sum(shape.weight for shape in shapes)
    shape_by_id = {shape.shape_id: shape for shape in shapes}
    thresholds = NoiseAwareThresholds(
        success_fraction=near_best_fraction,
        two_step_fraction=near_best_fraction,
        noise_multiplier=noise_multiplier,
    )
    ranked: list[tuple[tuple[float, ...], int, ParentScore]] = []
    for order, candidate_id in enumerate(candidate_ids):
        coverage_weight = 0.0
        for shape_id, best in best_times.items():
            if shape_id not in shape_by_id:
                continue
            value = timings.get(candidate_id, {}).get(shape_id)
            if value is None:
                continue
            if (
                thresholds.classify_one(as_timing(value), as_timing(best))
                is CandidateOutcome.SUCCESS
            ):
                coverage_weight += shape_by_id[shape_id].weight
        normalized_coverage = coverage_weight / total_weight if total_weight else 0.0
        contribution = max(0.0, float(contributions.get(candidate_id, 0.0)))
        score = normalized_coverage + contribution_weight * contribution
        parent_score = ParentScore(
            candidate_id=candidate_id,
            coverage_weight=coverage_weight,
            contribution=contribution,
            score=score,
            in_menu=candidate_id in menu_set,
        )
        # Menu entries remain parents when possible; everything else competes
        # by actual weighted coverage and leave-one-out contribution.
        rank = (
            float(candidate_id in menu_set),
            score,
            normalized_coverage,
            contribution,
        )
        if candidate_id in menu_set or coverage_weight > 0 or contribution > 0:
            ranked.append((rank, order, parent_score))
    if not ranked and max_parents and fallback_top_k:
        fallback = rank_parents_by_mean_regret(
            candidate_ids,
            shapes,
            timings,
            best_times,
            top_k=min(max_parents, fallback_top_k),
        )
        fallback_scores = [
            ParentScore(
                candidate_id=candidate_id,
                coverage_weight=0.0,
                contribution=0.0,
                score=-mean_regret,
                in_menu=False,
                mean_regret=mean_regret,
                is_fallback=True,
            )
            for candidate_id, mean_regret in fallback
        ]
        return ParentPool(
            parents=tuple(score.candidate_id for score in fallback_scores),
            scores={score.candidate_id: score for score in fallback_scores},
        )
    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected_scores = [item[2] for item in ranked[:max_parents]]
    selected_ids = {score.candidate_id for score in selected_scores}
    if len(selected_scores) < min(min_parents, max_parents):
        fallback = rank_parents_by_mean_regret(
            (
                candidate_id
                for candidate_id in candidate_ids
                if candidate_id not in selected_ids
            ),
            shapes,
            timings,
            best_times,
            top_k=min(min_parents - len(selected_scores), max_parents),
        )
        for candidate_id, mean_regret in fallback:
            score = ParentScore(
                candidate_id=candidate_id,
                coverage_weight=0.0,
                contribution=0.0,
                score=-mean_regret,
                in_menu=False,
                mean_regret=mean_regret,
                is_fallback=True,
            )
            selected_scores.append(score)
            selected_ids.add(candidate_id)
            ranked.append(
                (
                    (0.0, score.score, 0.0, 0.0),
                    len(ranked),
                    score,
                )
            )
    return ParentPool(
        parents=tuple(item.candidate_id for item in selected_scores),
        scores={item[2].candidate_id: item[2] for item in ranked},
    )


@dataclass
class GroupProgress:
    """Improvement/stagnation state for one (FC, tile-N) group."""

    attempts: int = 0
    since_improvement: int = 0
    best_objective: float | None = None
    improvement_anchor: float | None = None
    improvement_history: list[bool] = field(default_factory=list)

    def record(self, objective: float, *, improvement_fraction: float = 0.005) -> bool:
        if not math.isfinite(objective) or objective < 0:
            raise ValueError("objective must be nonnegative and finite")
        if not 0 <= improvement_fraction < 1:
            raise ValueError("improvement_fraction must be in [0, 1)")
        self.attempts += 1
        if self.improvement_anchor is None:
            self.improvement_anchor = objective
            improved = True
        else:
            denominator = max(self.improvement_anchor, 1e-15)
            improved = (
                self.improvement_anchor - objective
            ) / denominator >= improvement_fraction
        if self.best_objective is None or objective < self.best_objective:
            self.best_objective = objective
        if improved:
            self.since_improvement = 0
            self.improvement_anchor = objective
        else:
            self.since_improvement += 1
        self.improvement_history.append(improved)
        return improved

    def recent_improvement_rate(self, window: int = 40) -> float:
        """Return a smoothed improvement rate over the latest observations."""

        if window <= 0:
            raise ValueError("improvement-rate window must be positive")
        history = self.improvement_history[-window:]
        if not history:
            return 1.0
        # The beta(1, 1) prior keeps new groups competitive and avoids making
        # one short unlucky run indistinguishable from sustained stagnation.
        return (1.0 + sum(history)) / (2.0 + len(history))

    def stagnation_windows(self, window: int = 40) -> int:
        if window <= 0:
            raise ValueError("stagnation window must be positive")
        return self.since_improvement // window

    def is_frozen(self, window: int = 40, *, after_windows: int = 2) -> bool:
        if after_windows <= 0:
            raise ValueError("freeze window count must be positive")
        return self.stagnation_windows(window) >= after_windows

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GroupProgress:
        best = value.get("best_objective")
        anchor = value.get("improvement_anchor", best)
        return cls(
            attempts=int(value.get("attempts", 0)),
            since_improvement=int(value.get("since_improvement", 0)),
            best_objective=None if best is None else float(best),
            improvement_anchor=None if anchor is None else float(anchor),
            improvement_history=[
                bool(item) for item in value.get("improvement_history", ())
            ],
        )


def group_best_objective(
    candidate_ids: Iterable[Identifier],
    shapes: Sequence[Shape],
    timings: Mapping[Identifier, Mapping[Identifier, TimingLike]],
) -> float | None:
    """Return the weighted geometric mean of per-shape group-best timings."""

    candidate_ids = tuple(candidate_ids)
    weighted_log_time = 0.0
    covered_weight = 0.0
    for shape in shapes:
        measured = [
            value.time
            for candidate_id in candidate_ids
            if (
                value := as_timing(timings.get(candidate_id, {}).get(shape.shape_id))
            ).is_measured
        ]
        if not measured:
            continue
        best = min(float(value) for value in measured if value is not None)
        weighted_log_time += shape.weight * math.log(best)
        covered_weight += shape.weight
    if not covered_weight:
        return None
    return math.exp(weighted_log_time / covered_weight)


@dataclass(frozen=True)
class GroupCompetition:
    """Cross-group competitiveness weights."""

    weights: Mapping[str, float]


def contested_group_weights(
    candidates: Sequence[PortfolioCandidate],
    shapes: Sequence[Shape],
    timings: Mapping[Identifier, Mapping[Identifier, TimingLike]],
    *,
    contest_margin: float = 0.03,
    min_samples: int = 20,
    eligible: Mapping[str, set[Identifier] | frozenset[Identifier]] | None = None,
) -> GroupCompetition:
    """Weight tile groups by shapes they can contest.

    Under-sampled group/shape pairs remain optimistic.
    """

    if contest_margin < 0 or not math.isfinite(contest_margin):
        raise ValueError("contest margin must be finite and non-negative")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")

    by_group: dict[str, list[Identifier]] = {}
    for candidate in candidates:
        by_group.setdefault(str(candidate.tile_n), []).append(candidate.candidate_id)
    weights = {group: 0.0 for group in by_group}

    for shape in shapes:
        measured_by_group: dict[str, list[float]] = {}
        for group, candidate_ids in by_group.items():
            if eligible is not None and shape.shape_id not in eligible.get(
                group, frozenset()
            ):
                measured_by_group[group] = []
                continue
            measured_by_group[group] = [
                float(value.time)
                for candidate_id in candidate_ids
                if (
                    value := as_timing(
                        timings.get(candidate_id, {}).get(shape.shape_id)
                    )
                ).is_measured
                and value.time is not None
            ]
        all_times = [
            timing
            for group_times in measured_by_group.values()
            for timing in group_times
        ]
        global_best = min(all_times) if all_times else None
        for group, group_times in measured_by_group.items():
            if eligible is not None and shape.shape_id not in eligible.get(
                group, frozenset()
            ):
                continue
            if len(group_times) < min_samples or global_best is None:
                weights[group] += shape.weight
                continue
            group_best = min(group_times)
            regret = group_best / global_best - 1.0
            if regret <= contest_margin:
                weights[group] += shape.weight

    return GroupCompetition(weights)


@dataclass(frozen=True)
class GroupBudgetAllocator:
    """Allocate dispatches by shape weight, with floors and stagnation decay."""

    minimum_share: float = 0.05
    stagnation_window: int = 40
    stagnation_reduction: float = 0.5
    freeze_after_windows: int = 2

    def _validate(self, count: int) -> None:
        if count and self.minimum_share * count > 1.0 + 1e-12:
            raise ValueError("minimum shares exceed the total budget")
        if self.minimum_share < 0 or self.stagnation_window <= 0:
            raise ValueError("invalid floor or stagnation window")
        if not 0.0 <= self.stagnation_reduction <= 1.0:
            raise ValueError("stagnation_reduction must be in [0, 1]")
        if self.freeze_after_windows <= 0:
            raise ValueError("freeze_after_windows must be positive")

    def shares(
        self,
        weights: Mapping[str, float],
        progress: Mapping[str, GroupProgress] | None = None,
    ) -> dict[str, float]:
        """Return exact shares; stagnant groups lose only their non-floor share."""

        if not weights:
            return {}
        progress = progress or {}
        for weight in weights.values():
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("group weights must be nonnegative and finite")
        active = {group: weight for group, weight in weights.items() if weight > 0}
        if not active:
            return {group: 0.0 for group in weights}
        self._validate(len(active))
        adjusted: dict[str, float] = {}
        for group, weight in active.items():
            state = progress.get(group)
            windows = (
                state.stagnation_windows(self.stagnation_window)
                if state is not None
                else 0
            )
            reduction = (
                0.0
                if windows >= self.freeze_after_windows
                else self.stagnation_reduction**windows
            )
            improvement_rate = (
                state.recent_improvement_rate(self.stagnation_window)
                if state is not None
                else 1.0
            )
            adjusted[group] = weight * improvement_rate * reduction
        adjusted_total = sum(adjusted.values())
        if adjusted_total == 0:
            # A normalized scheduler cannot leave every group at only its
            # floor. Resume all-frozen groups uniformly to keep making progress.
            adjusted = {group: 1.0 for group in active}
            adjusted_total = float(len(adjusted))
        remainder = 1.0 - self.minimum_share * len(active)
        shares = {
            group: self.minimum_share + remainder * adjusted[group] / adjusted_total
            for group in active
        }
        return {group: shares.get(group, 0.0) for group in weights}

    def allocate(
        self,
        total: int,
        weights: Mapping[str, float],
        progress: Mapping[str, GroupProgress] | None = None,
    ) -> dict[str, int]:
        """Round shares with largest remainders while preserving ``total``."""

        if total < 0:
            raise ValueError("total budget must be nonnegative")
        shares = self.shares(weights, progress)
        exact = {group: total * share for group, share in shares.items()}
        result = {group: math.floor(value) for group, value in exact.items()}
        left = total - sum(result.values())
        order = sorted(
            weights,
            key=lambda group: (exact[group] - result[group], group),
            reverse=True,
        )
        for group in order[:left]:
            result[group] += 1
        return result


@dataclass
class SearchState:
    """JSON-serializable state needed to resume the adaptive search."""

    selector: AdaptiveMoveSelector = field(default_factory=AdaptiveMoveSelector)
    groups: dict[str, GroupProgress] = field(default_factory=dict)
    evaluated: list[str] = field(default_factory=list)
    two_step_remaining: dict[str, int] = field(default_factory=dict)
    two_step_depth: dict[str, int] = field(default_factory=dict)
    pending_confirmations: dict[str, list[str]] = field(default_factory=dict)
    group_dispatch_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    dispatched: int = 0
    completed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 3,
            "selector": self.selector.to_dict(),
            "groups": {
                group: state.to_dict() for group, state in sorted(self.groups.items())
            },
            "evaluated": list(self.evaluated),
            "two_step_remaining": dict(sorted(self.two_step_remaining.items())),
            "two_step_depth": dict(sorted(self.two_step_depth.items())),
            "pending_confirmations": {
                candidate_id: sorted(shape_ids)
                for candidate_id, shape_ids in sorted(
                    self.pending_confirmations.items()
                )
            },
            "group_dispatch_stats": self.group_dispatch_stats,
            "dispatched": self.dispatched,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SearchState:
        version = int(value.get("version", 1))
        if version not in (2, 3):
            raise ValueError("unsupported search-state version")
        selector = AdaptiveMoveSelector.from_dict(value["selector"])
        groups = {
            str(group): GroupProgress.from_dict(state)
            for group, state in value.get("groups", {}).items()
        }
        pending_confirmations = {
            str(candidate_id): [str(shape_id) for shape_id in shape_ids]
            for candidate_id, shape_ids in value.get(
                "pending_confirmations", {}
            ).items()
        }
        # Older v3 checkpoints treated a failed confirmation as permanently
        # invalid. Preserve the exclusion while exposing it as unconfirmed so
        # a resumed run can retry and report it.
        for pair in value.get("invalidated_pairs", ()):
            candidate_id, shape_id = str(pair[0]), str(pair[1])
            pending = pending_confirmations.setdefault(candidate_id, [])
            if shape_id not in pending:
                pending.append(shape_id)
        return cls(
            selector=selector,
            groups=groups,
            evaluated=[str(item) for item in value.get("evaluated", ())],
            two_step_remaining={
                str(key): max(0, int(remaining))
                for key, remaining in value.get("two_step_remaining", {}).items()
                if int(remaining) > 0
            },
            two_step_depth={
                str(key): int(depth)
                for key, depth in value.get("two_step_depth", {}).items()
            },
            pending_confirmations=pending_confirmations,
            group_dispatch_stats={
                str(group): dict(stats)
                for group, stats in value.get("group_dispatch_stats", {}).items()
            },
            dispatched=int(value.get("dispatched", 0)),
            completed=int(value.get("completed", 0)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> SearchState:
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise ValueError("search state must decode to an object")
        return cls.from_dict(decoded)
