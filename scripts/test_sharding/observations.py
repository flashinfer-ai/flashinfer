from __future__ import annotations

import csv
import gzip
import io
import statistics
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from .estimates import (
    DurationEstimate,
    EstimateBook,
    OverheadEstimate,
    nearest_rank_p90,
)
from .io import atomic_write_bytes, atomic_write_text
from .models import base_function_for_nodeid, source_file_for_nodeid


@dataclass(frozen=True)
class ObservedCase:
    profile: str
    nodeid: str
    source_file: str
    base_function: str
    outcome: str
    seconds: float
    adjusted_seconds: float
    synthetic: bool
    run_id: str
    batch_id: str
    first_in_batch: bool


@dataclass(frozen=True)
class EstimateRefresh:
    duration_file: Path
    overhead_file: Path
    summary_file: Path
    keep_nodeids: set[str] | None = None
    prune_profile: str | None = None


@dataclass(frozen=True)
class ObservedOverhead:
    profile: str
    source_file: str
    process_startup_seconds: float
    source_warmup_seconds: float
    run_id: str
    batch_id: str


def _format_seconds(value: float) -> str:
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


def adjust_first_case_warmup(
    observations: Iterable[ObservedCase],
) -> tuple[list[ObservedCase], dict[tuple[str, str], float]]:
    values = list(observations)
    siblings: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item in values:
        siblings[(item.profile, item.base_function)].append(item.seconds)
    adjusted: list[ObservedCase] = []
    warmup: dict[tuple[str, str], float] = defaultdict(float)
    for item in values:
        peers = list(siblings[(item.profile, item.base_function)])
        with suppress(ValueError):
            peers.remove(item.seconds)
        if item.first_in_batch and len(peers) >= 10:
            median = statistics.median(peers)
            if median > 0 and item.seconds >= 10 * median:
                adjusted.append(replace(item, adjusted_seconds=median))
                warmup[(item.run_id, item.batch_id)] += item.seconds - median
                continue
        adjusted.append(replace(item, adjusted_seconds=item.seconds))
    return adjusted, dict(warmup)


def _write_duration_file(path: Path, rows: Iterable[DurationEstimate]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["profile", "nodeid", "estimated_seconds", "sample_count"])
    for row in sorted(
        rows, key=lambda item: (item.profile.encode(), item.nodeid.encode())
    ):
        writer.writerow(
            [
                row.profile,
                row.nodeid,
                _format_seconds(row.estimated_seconds),
                row.sample_count,
            ]
        )
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0
    ) as compressed:
        compressed.write(stream.getvalue().encode("utf-8"))
    atomic_write_bytes(path, buffer.getvalue())


def _write_overhead_file(path: Path, rows: Iterable[OverheadEstimate]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "profile",
            "source_file",
            "process_startup_seconds",
            "source_warmup_seconds",
            "sample_count",
        ]
    )
    for row in sorted(
        rows, key=lambda item: (item.profile.encode(), item.source_file.encode())
    ):
        writer.writerow(
            [
                row.profile,
                row.source_file,
                _format_seconds(row.process_startup_seconds),
                _format_seconds(row.source_warmup_seconds),
                row.sample_count,
            ]
        )
    atomic_write_text(path, stream.getvalue())


def _write_summary(path: Path, rows: Iterable[DurationEstimate]) -> None:
    aggregate: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        aggregate[
            (
                row.profile,
                source_file_for_nodeid(row.nodeid),
                base_function_for_nodeid(row.nodeid),
            )
        ].append(row.estimated_seconds)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "profile",
            "source_file",
            "base_function",
            "node_count",
            "total_estimated_seconds",
            "p90_estimated_seconds",
        ]
    )
    for key, estimates in sorted(
        aggregate.items(), key=lambda item: tuple(value.encode() for value in item[0])
    ):
        writer.writerow(
            [
                *key,
                len(estimates),
                _format_seconds(sum(estimates)),
                _format_seconds(nearest_rank_p90(estimates)),
            ]
        )
    atomic_write_text(path, stream.getvalue())


def refresh_estimates(
    observations: Iterable[ObservedCase],
    overhead_observations: Iterable[ObservedOverhead],
    request: EstimateRefresh,
) -> tuple[list[DurationEstimate], list[OverheadEstimate]]:
    duration_file = request.duration_file
    overhead_file = request.overhead_file
    summary_file = request.summary_file
    book = EstimateBook.from_files(duration_file, overhead_file)
    existing = {(row.profile, row.nodeid): row for row in book.durations}
    eligible: dict[tuple[str, str], list[ObservedCase]] = defaultdict(list)
    for observation in observations:
        if not observation.synthetic and observation.outcome in {"passed", "failed"}:
            eligible[(observation.profile, observation.nodeid)].append(observation)
    updated = dict(existing)
    for duration_key, duration_samples in eligible.items():
        old_duration = existing.get(duration_key)
        passing = [
            sample.adjusted_seconds
            for sample in duration_samples
            if sample.outcome == "passed"
        ]
        failing = [
            sample.adjusted_seconds
            for sample in duration_samples
            if sample.outcome == "failed"
        ]
        candidates: list[float] = []
        if passing:
            passing_candidate = 1.2 * nearest_rank_p90(passing)
            candidates.append(
                max(0.9 * old_duration.estimated_seconds, passing_candidate)
                if old_duration is not None
                else passing_candidate
            )
        if failing:
            failure_candidate = 1.2 * nearest_rank_p90(failing)
            candidates.append(
                max(old_duration.estimated_seconds, failure_candidate)
                if old_duration is not None
                else failure_candidate
            )
        estimate = max(candidates)
        updated[duration_key] = DurationEstimate(
            profile=duration_key[0],
            nodeid=duration_key[1],
            estimated_seconds=estimate,
            sample_count=(old_duration.sample_count if old_duration is not None else 0)
            + len(duration_samples),
        )
    if request.keep_nodeids is not None and request.prune_profile is not None:
        updated = {
            key: row
            for key, row in updated.items()
            if key[0] != request.prune_profile or key[1] in request.keep_nodeids
        }

    overhead_existing = {(row.profile, row.source_file): row for row in book.overheads}
    grouped_overheads: dict[tuple[str, str], list[ObservedOverhead]] = defaultdict(list)
    for overhead_observation in overhead_observations:
        grouped_overheads[
            (overhead_observation.profile, overhead_observation.source_file)
        ].append(overhead_observation)
    overhead_updated = dict(overhead_existing)
    for overhead_key, overhead_samples in grouped_overheads.items():
        old_overhead = overhead_existing.get(overhead_key)
        startup_candidate = 1.2 * nearest_rank_p90(
            sample.process_startup_seconds for sample in overhead_samples
        )
        warmup_candidate = 1.2 * nearest_rank_p90(
            sample.source_warmup_seconds for sample in overhead_samples
        )
        if old_overhead is not None:
            startup_candidate = max(
                0.9 * old_overhead.process_startup_seconds, startup_candidate
            )
            warmup_candidate = max(
                0.9 * old_overhead.source_warmup_seconds, warmup_candidate
            )
        overhead_updated[overhead_key] = OverheadEstimate(
            profile=overhead_key[0],
            source_file=overhead_key[1],
            process_startup_seconds=startup_candidate,
            source_warmup_seconds=warmup_candidate,
            sample_count=(old_overhead.sample_count if old_overhead is not None else 0)
            + len(overhead_samples),
        )

    duration_rows = list(updated.values())
    overhead_rows = list(overhead_updated.values())
    _write_duration_file(duration_file, duration_rows)
    _write_overhead_file(overhead_file, overhead_rows)
    _write_summary(summary_file, duration_rows)
    return duration_rows, overhead_rows
