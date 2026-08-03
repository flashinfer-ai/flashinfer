from __future__ import annotations

import csv
import gzip
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Iterable

from .models import base_function_for_nodeid, source_file_for_nodeid


PREFERRED_FALLBACK_PROFILES = frozenset({"sm103-cuda12", "sm103-cuda13"})
MIN_AGGREGATE_COVERAGE = 0.8


@dataclass(frozen=True)
class DurationEstimate:
    profile: str
    nodeid: str
    estimated_seconds: float
    sample_count: int


@dataclass(frozen=True)
class OverheadEstimate:
    profile: str
    source_file: str
    process_startup_seconds: float
    source_warmup_seconds: float
    sample_count: int


@dataclass(frozen=True)
class DurationLookup:
    seconds: float
    source: str


@dataclass(frozen=True)
class DurationCoverage:
    function_totals: dict[str, int]
    source_totals: dict[str, int]
    function_observed: dict[tuple[str, str], int]
    source_observed: dict[tuple[str, str], int]

    def sufficient(
        self,
        profile: str,
        key: str,
        level: str,
    ) -> bool:
        if level == "function":
            observed = self.function_observed.get((profile, key), 0)
            population = self.function_totals.get(key, 0)
        elif level == "source":
            observed = self.source_observed.get((profile, key), 0)
            population = self.source_totals.get(key, 0)
        else:
            raise ValueError(f"unknown aggregate level: {level}")
        return population > 0 and observed / population >= MIN_AGGREGATE_COVERAGE


def nearest_rank_p90(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("p90 requires at least one value")
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]


class EstimateBook:
    def __init__(
        self,
        durations: Iterable[DurationEstimate] = (),
        overheads: Iterable[OverheadEstimate] = (),
    ) -> None:
        self.durations = tuple(durations)
        self.overheads = tuple(overheads)
        self._exact = {
            (estimate.profile, estimate.nodeid): estimate for estimate in self.durations
        }
        self._by_node: dict[str, list[DurationEstimate]] = defaultdict(list)
        self._overhead = {
            (estimate.profile, estimate.source_file): estimate
            for estimate in self.overheads
        }
        self._overhead_profiles = frozenset(
            estimate.profile for estimate in self.overheads
        )
        overhead_seconds = [
            estimate.process_startup_seconds + estimate.source_warmup_seconds
            for estimate in self.overheads
        ]
        # Each row predicts one source file, so sample_count must not bias the
        # default toward a few heavily sampled files.
        self._default_overhead_seconds = (
            median(overhead_seconds) if overhead_seconds else 0.0
        )
        self._function: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._source: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._suite: dict[str, list[float]] = defaultdict(list)
        for estimate in self.durations:
            self._by_node[estimate.nodeid].append(estimate)
            key = (estimate.profile, base_function_for_nodeid(estimate.nodeid))
            self._function[key].append(estimate.estimated_seconds)
            source_key = (estimate.profile, source_file_for_nodeid(estimate.nodeid))
            self._source[source_key].append(estimate.estimated_seconds)
            self._suite[estimate.profile].append(estimate.estimated_seconds)
        self._function_p90 = {
            key: nearest_rank_p90(values) for key, values in self._function.items()
        }
        self._source_p90 = {
            key: nearest_rank_p90(values) for key, values in self._source.items()
        }
        self._suite_mean = {key: fmean(values) for key, values in self._suite.items()}
        self._duration_profiles = frozenset(self._suite_mean)

    @staticmethod
    def _fallback_profiles(
        profile: str, available_profiles: frozenset[str]
    ) -> frozenset[str]:
        alternatives = available_profiles - {profile}
        preferred = alternatives & PREFERRED_FALLBACK_PROFILES
        if profile not in available_profiles and preferred:
            return preferred
        return alternatives

    @classmethod
    def from_files(
        cls, duration_path: Path, overhead_path: Path | None = None
    ) -> "EstimateBook":
        durations: list[DurationEstimate] = []
        if duration_path.exists():
            opener = gzip.open if duration_path.suffix == ".gz" else open
            with opener(duration_path, "rt", newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    durations.append(
                        DurationEstimate(
                            profile=row["profile"],
                            nodeid=row["nodeid"],
                            estimated_seconds=float(row["estimated_seconds"]),
                            sample_count=int(row["sample_count"]),
                        )
                    )
        overheads: list[OverheadEstimate] = []
        if overhead_path is not None and overhead_path.exists():
            with overhead_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    overheads.append(
                        OverheadEstimate(
                            profile=row["profile"],
                            source_file=row["source_file"],
                            process_startup_seconds=float(
                                row["process_startup_seconds"]
                            ),
                            source_warmup_seconds=float(row["source_warmup_seconds"]),
                            sample_count=int(row["sample_count"]),
                        )
                    )
        return cls(durations, overheads)

    def lookup(
        self,
        nodeid: str,
        profile: str,
        unknown_floor_seconds: float,
        *,
        coverage: DurationCoverage | None = None,
    ) -> DurationLookup:
        exact = self._exact.get((profile, nodeid))
        if exact is not None:
            return DurationLookup(exact.estimated_seconds, "exact-current-profile")

        fallback_profiles = self._fallback_profiles(profile, self._duration_profiles)
        exact_other = [
            item.estimated_seconds
            for item in self._by_node.get(nodeid, ())
            if item.profile in fallback_profiles
        ]
        if exact_other:
            return DurationLookup(max(exact_other), "exact-other-profile")

        function = base_function_for_nodeid(nodeid)
        source = source_file_for_nodeid(nodeid)
        current_levels = (
            (
                self._function_p90.get((profile, function)),
                function,
                "function",
                "function-current-profile",
            ),
            (
                self._source_p90.get((profile, source)),
                source,
                "source",
                "source-current-profile",
            ),
        )
        for value, key, level, current_name in current_levels:
            if value is not None and (
                coverage is None or coverage.sufficient(profile, key, level)
            ):
                return DurationLookup(max(unknown_floor_seconds, value), current_name)
        suite_mean = self._suite_mean.get(profile)
        if suite_mean is not None:
            return DurationLookup(
                max(unknown_floor_seconds, suite_mean),
                "suite-mean-current-profile",
            )
        other_levels = (
            (
                [
                    (other_profile, value)
                    for (
                        other_profile,
                        other_function,
                    ), value in self._function_p90.items()
                    if other_profile in fallback_profiles and other_function == function
                ],
                function,
                "function",
                "function-other-profile",
            ),
            (
                [
                    (other_profile, value)
                    for (other_profile, other_source), value in self._source_p90.items()
                    if other_profile in fallback_profiles and other_source == source
                ],
                source,
                "source",
                "source-other-profile",
            ),
        )
        for candidates, key, level, other_name in other_levels:
            covered = [
                value
                for other_profile, value in candidates
                if coverage is None or coverage.sufficient(other_profile, key, level)
            ]
            if covered:
                return DurationLookup(
                    max(unknown_floor_seconds, max(covered)), other_name
                )
        other_suite_means = [
            value
            for other_profile, value in self._suite_mean.items()
            if other_profile in fallback_profiles
        ]
        if other_suite_means:
            return DurationLookup(
                max(unknown_floor_seconds, max(other_suite_means)),
                "suite-mean-other-profile",
            )
        return DurationLookup(unknown_floor_seconds, "unknown-floor")

    def coverage(self, nodeids: Iterable[str]) -> DurationCoverage:
        function_totals: dict[str, int] = defaultdict(int)
        source_totals: dict[str, int] = defaultdict(int)
        function_observed: dict[tuple[str, str], int] = defaultdict(int)
        source_observed: dict[tuple[str, str], int] = defaultdict(int)
        for nodeid in nodeids:
            function = base_function_for_nodeid(nodeid)
            source = source_file_for_nodeid(nodeid)
            function_totals[function] += 1
            source_totals[source] += 1
            for observed_profile in {
                estimate.profile for estimate in self._by_node.get(nodeid, ())
            }:
                function_observed[(observed_profile, function)] += 1
                source_observed[(observed_profile, source)] += 1
        return DurationCoverage(
            dict(function_totals),
            dict(source_totals),
            dict(function_observed),
            dict(source_observed),
        )

    def overhead_ms(self, source_file: str, profile: str) -> int:
        exact = self._overhead.get((profile, source_file))
        if exact is None:
            fallback_profiles = self._fallback_profiles(
                profile, self._overhead_profiles
            )
            alternatives = [
                value
                for (other_profile, source), value in self._overhead.items()
                if source == source_file and other_profile in fallback_profiles
            ]
            if not alternatives:
                seconds = self._default_overhead_seconds
            else:
                seconds = max(
                    item.process_startup_seconds + item.source_warmup_seconds
                    for item in alternatives
                )
        else:
            seconds = exact.process_startup_seconds + exact.source_warmup_seconds
        return max(0, round(seconds * 1000))
