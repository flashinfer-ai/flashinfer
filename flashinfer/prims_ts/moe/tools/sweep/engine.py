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

"""Driver and subprocess entry points for Prims-TS MoE sweeps."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import resource
import statistics
import subprocess
from types import SimpleNamespace
import sys
import tempfile
import threading
import time
from typing import Iterable, Mapping, Sequence

from flashinfer.prims_ts.batched_gemm.config_types import DType
from flashinfer.tllm_enums import ActivationType

from .calibration import CalibrationData, measure_worker_gpu_footprint
from .db import JSONLWriter, SweepCheckpoint, atomic_write_json, deterministic_id
from .defaults import (
    CANDIDATE_TIMEOUT_SECONDS,
    COMPILE_TIMEOUT_SECONDS,
    DEFAULT_CONFIRM_FRACTION,
    DEFAULT_MAX_ACCUMULATOR_STAGES,
    DEFAULT_MAX_C_SMEM_STAGES,
    DEFAULT_MAX_INPUT_STAGES,
    DEFAULT_MAX_NEAR_BEST_PER_SHAPE,
    DEFAULT_MAX_ROUTING_REGRET,
    DEFAULT_MAX_SF_RING_RATIO,
    DEFAULT_MAX_WORKID_STAGES,
    DEFAULT_REFERENCE_RESERVE_MINUTES,
    DEFAULT_RETIME_ROUTINGS,
    DEFAULT_SEARCH_ROUTING,
    MAX_SEEDS,
    PROGRESS_INTERVAL_SECONDS,
    RANDOM_SEED,
    ROUTING_SEED,
    SEARCH_REPEAT_ITERS,
    SEARCH_WARMUP_ITERS,
    SEED_LIMIT,
    SHAPE_TIMEOUT_SECONDS,
    WORKER_RESTART_CONFIGS,
)
from .measure import (
    TimingProtocol,
    compile_candidate,
    evaluate_candidate,
    final_timing_protocol,
    reference_check_candidate,
    resolve_timing_backend,
)
from .portfolio import (
    FAILED,
    PortfolioCandidate,
    Shape as PortfolioShape,
    TimingValue,
    build_final_retime_pool,
    select_greedy_menu,
)
from .report import make_summary, refresh_environment_provenance, write_summary
from .search import (
    AdaptiveMoveSelector,
    CandidateOutcome,
    GroupBudgetAllocator,
    GroupProgress,
    NoiseAwareThresholds,
    ParentMoveTracker,
    SearchState,
    contested_group_weights,
    group_best_objective,
    select_parent_pool,
)
from .shapes import (
    MoeSweepShape,
    build_sweep_shapes,
    merge_duplicate_shapes,
    model_moe_shape,
)
from .space import (
    MOVE_KINDS,
    Candidate,
    ConfigContext,
    StageLimits,
    STAGE_MOVE_KINDS,
    applicable_move_kinds,
    enumerate_incumbents,
    enumerate_stage_neighbors,
    filter_legal_move_kinds,
    fixed_seed_variants,
    generate_move,
    is_applicable,
    materialize_candidate,
    repair_structural_candidate,
    structural_options_key,
)
from .workers import (
    FailureKind,
    PersistentProcessWorker,
    ProcessResult,
    WorkerActivity,
    WorkerStatus,
)
from .utils import git_output, repo_root, resolve_visible_gpu


_ADAPTIVE_MOVES = tuple(move for move in MOVE_KINDS if move != "crossover")
REFERENCE_DEVIATION_TOLERANCE = 0.25
_NUMERIC_MOVES = (
    "stages_all",
    "stages_one",
    "stages_trade",
    "tmem_stages",
    "epilogue",
    "warps",
    "flags",
)


def shape_id(shape: MoeSweepShape, fc: str) -> str:
    return deterministic_id(asdict(shape.launch_key(fc=fc)), length=20)


def cap_candidates_round_robin(
    candidates: Sequence[Candidate], *, limit: int
) -> list[Candidate]:
    """Apply a deterministic cap while preserving every tile-N group."""
    if limit <= 0:
        return []
    groups: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.tile_n, []).append(candidate)
    result: list[Candidate] = []
    while len(result) < limit:
        added = False
        for tile_n in sorted(groups):
            group = groups[tile_n]
            if group:
                result.append(group.pop(0))
                added = True
                if len(result) == limit:
                    break
        if not added:
            break
    return result


def cap_and_prune_candidates(
    candidates: dict[str, Candidate],
    incumbent_ids: set[str],
    pending: Sequence[Candidate],
    *,
    limit: int,
) -> tuple[list[Candidate], set[str]]:
    """Cap pending incumbents and keep candidate/incumbent indexes consistent."""

    retained = cap_candidates_round_robin(pending, limit=limit)
    retained_ids = {candidate.candidate_id for candidate in retained}
    dropped_ids = {
        candidate.candidate_id
        for candidate in pending
        if candidate.candidate_id not in retained_ids
    }
    incumbent_ids.difference_update(dropped_ids)
    for candidate_id in dropped_ids:
        candidates.pop(candidate_id, None)
    return retained, dropped_ids


def code_fingerprint() -> dict[str, object]:
    """Identify the sweep code a run used, including uncommitted edits."""
    package_root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())

    status = git_output("status", "--porcelain", "--untracked-files=no")
    dirty_files: list[str] | int = (
        -1
        if status is None
        else sorted(
            {line[3:] for line in status.splitlines() if len(line) > 3 and line[3:]}
        )
    )
    return {
        "module_path": str(Path(__file__).resolve().parent),
        "python": sys.executable,
        "git_commit": git_output("rev-parse", "--short=12", "HEAD"),
        "git_dirty": None if dirty_files == -1 else bool(dirty_files),
        "git_dirty_files": dirty_files,
        "prims_ts_source_digest": digest.hexdigest()[:16],
    }


def gpu_identity(selector: str) -> dict[str, str | None]:
    """Resolve a stable GPU name/UUID without initializing CUDA."""

    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=name,uuid",
                "--format=csv,noheader",
            ),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {"gpu_name": None, "gpu_uuid": None}
    devices = []
    for line in completed.stdout.splitlines():
        try:
            name, uuid = (value.strip() for value in line.rsplit(",", maxsplit=1))
        except ValueError:
            continue
        devices.append((name, uuid))
    try:
        resolved = resolve_visible_gpu(selector)
    except ValueError:
        return {"gpu_name": None, "gpu_uuid": None}
    if resolved.startswith("GPU-"):
        match = next((device for device in devices if device[1] == resolved), None)
    else:
        try:
            match = devices[int(resolved)]
        except (ValueError, IndexError):
            match = None
    return {
        "gpu_name": None if match is None else match[0],
        "gpu_uuid": None if match is None else match[1],
    }


def _short_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 120.0:
        return f"{seconds:.0f}s"
    return f"{seconds / 60.0:.1f}m"


def describe_worker_failure(
    result: ProcessResult,
    *,
    compile_timeout_seconds: float,
    run_timeout_seconds: float,
) -> str:
    """One-line cause for a failed request, placed ahead of the stderr tail."""
    phase = result.phase or "unknown"
    where = f"{phase} ({result.phase_unit})" if result.phase_unit else phase
    phase_seconds = result.phase_elapsed_seconds or 0.0
    if result.status is WorkerStatus.TIMED_OUT:
        limit = compile_timeout_seconds if phase == "compile" else run_timeout_seconds
        reason = (
            f"{phase} limit {limit:.0f}s"
            if phase != "lock_wait" and phase_seconds >= limit
            else "sweep deadline"
        )
        return (
            f"killed in {where} after {phase_seconds:.1f}s in phase, "
            f"{result.elapsed_seconds:.1f}s total ({reason})"
        )
    return (
        f"worker {result.failure.value} in {where} after "
        f"{result.elapsed_seconds:.1f}s (exit code {result.returncode})"
    )


_ACTIVITY_PHASES = ("compile", "lock_wait", "run")


def summarize_worker_activity(
    activities: Iterable[WorkerActivity], *, now: float
) -> tuple[tuple[tuple[str, int], ...], str]:
    """Return a change-detection key and a phase breakdown of in-flight requests.

    The key holds only per-phase counts so a heartbeat can stay quiet while
    nothing moves between phases, even though ages keep increasing.
    """
    by_phase: dict[str, list[WorkerActivity]] = {}
    for activity in activities:
        by_phase.setdefault(activity.phase, []).append(activity)
    if not by_phase:
        return (), "no worker requests active"
    order = [phase for phase in _ACTIVITY_PHASES if phase in by_phase]
    order.extend(sorted(set(by_phase) - set(_ACTIVITY_PHASES)))
    key = tuple((phase, len(by_phase[phase])) for phase in order)
    parts = [" ".join(f"{phase}={count}" for phase, count in key)]
    for phase in order:
        if phase == "run":
            continue
        oldest = min(activity.phase_started for activity in by_phase[phase])
        parts.append(f"oldest {phase} {_short_duration(now - oldest)}")
    running = sorted(
        by_phase.get("run", ()), key=lambda activity: activity.request_started
    )
    for activity in running[:2]:
        progress = activity.unit.split(" ", 1)[0] or "?"
        parts.append(
            f"timing shape {progress} "
            f"(request age {_short_duration(now - activity.request_started)})"
        )
    if len(running) > 2:
        parts.append(f"+{len(running) - 2} more timing")
    return key, "; ".join(parts)


def _shape_count_text(counts: Sequence[int]) -> str:
    if not counts:
        return "0"
    low, high = min(counts), max(counts)
    return str(low) if low == high else f"{low}-{high}"


def manifest_identity(manifest: Mapping[str, object]) -> dict[str, object]:
    """Return fields that affect candidate identity or recorded measurements.

    The result is JSON-normalized so an in-memory manifest (tuples, from
    ``asdict``) compares equal to the same manifest reloaded from disk (lists).
    """
    timing = dict(manifest.get("timing", {}))
    search = dict(manifest.get("search", {}))
    identity = {
        "target": manifest.get("target"),
        "timing": {
            key: timing.get(key)
            for key in (
                "warmup_iters",
                "repeat_iters",
                "timing_backend",
                "cold_l2_cache",
                "gpu_name",
                "gpu_uuid",
                "protocol_version",
            )
        },
        "search": {
            key: search.get(key)
            for key in (
                "seed",
                "menu_size",
                "max_input_stages",
                "max_accumulator_stages",
                "max_c_smem_stages",
                "max_workid_stages",
                "max_sf_ring_ratio",
            )
        },
    }
    return json.loads(json.dumps(identity, sort_keys=True))


def _with_context_resume_defaults(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Fill context fields absent from manifests written by older sweep code."""

    normalized = json.loads(json.dumps(manifest))
    target = normalized.get("target")
    if not isinstance(target, dict):
        return normalized
    context = target.get("context")
    if not isinstance(context, dict):
        return normalized
    for key in (
        "gemm1_alpha_value",
        "gemm1_beta_value",
        "gemm1_clamp_limit_value",
    ):
        context.setdefault(key, None)
    return normalized


def manifest_identity_mismatches(
    prior: Mapping[str, object], current: Mapping[str, object]
) -> list[str]:
    """Describe each identity field that differs between two manifests."""
    mismatches: list[str] = []

    def compare(path: str, old: object, new: object) -> None:
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                compare(f"{path}.{key}", old.get(key), new.get(key))
        elif (
            isinstance(old, list)
            and isinstance(new, list)
            and len(old) == len(new)
            and any(isinstance(value, (dict, list)) for value in old + new)
        ):
            for index, (old_value, new_value) in enumerate(zip(old, new, strict=True)):
                compare(f"{path}[{index}]", old_value, new_value)
        elif old != new:
            mismatches.append(f"{path}: resumed={old!r} current={new!r}")

    compare(
        "manifest",
        manifest_identity(_with_context_resume_defaults(prior)),
        manifest_identity(_with_context_resume_defaults(current)),
    )
    return mismatches


def speedup_vs_pre_sweep(
    records: Sequence[Mapping[str, object]],
    incumbent_ids: set[str],
    shape_ids: Sequence[str],
    *,
    selected_ids: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Per shape: fastest pre-sweep (incumbent) time over fastest time found.

    Only candidates the menu could use count: a successful search record with
    no failed measurement, matching ``SweepEngine._timing_inputs``.
    """
    latest: dict[str, Mapping[str, object]] = {}
    for record in records:
        if record.get("phase") == "search":
            latest[str(record.get("candidate_id"))] = record
    pre_sweep: dict[str, float] = {}
    best: dict[str, tuple[float, str]] = {}
    for candidate_id, record in latest.items():
        measurements = list(record.get("measurements", ()))
        if record.get("status") != "ok" or any(
            measurement.get("status") not in ("ok", "not_applicable")
            for measurement in measurements
        ):
            continue
        for measurement in measurements:
            if measurement.get("status") != "ok":
                continue
            key = str(measurement["shape_id"])
            time_us = float(measurement["median_us"])
            if (selected_ids is None or candidate_id in selected_ids) and (
                key not in best or time_us < best[key][0]
            ):
                best[key] = (time_us, candidate_id)
            if candidate_id in incumbent_ids and time_us < pre_sweep.get(key, math.inf):
                pre_sweep[key] = time_us
    rows: dict[str, dict[str, object]] = {}
    for key in shape_ids:
        baseline = pre_sweep.get(key)
        found = best.get(key)
        rows[key] = {
            "pre_sweep_us": baseline,
            "best_us": None if found is None else found[0],
            "best_candidate_id": None if found is None else found[1],
            "speedup": (
                None if baseline is None or found is None else baseline / found[0]
            ),
        }
    return rows


def selection_portfolio_shapes(
    shapes: Sequence[MoeSweepShape],
    *,
    fc: str,
    routings: Sequence[str],
) -> list[PortfolioShape]:
    """Return final-selection shapes with routing-neutral logical weights."""

    routing_set = set(routings)
    if not routing_set:
        raise ValueError("selection routings must not be empty")
    scale = 1.0 / len(routing_set)
    return [
        PortfolioShape(shape_id(shape, fc), shape.weight * scale)
        for shape in shapes
        if shape.routing in routing_set
    ]


def reference_consistency(
    records: Sequence[Mapping[str, object]],
    *,
    selected_ids: set[str] | None = None,
    tolerance: float = REFERENCE_DEVIATION_TOLERANCE,
) -> dict[str, object]:
    """Compare seeded reference error statistics with incumbent sanity checks."""

    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("reference consistency tolerance must be nonnegative")
    checks_by_shape: dict[str, list[tuple[str, str, Mapping[str, object]]]] = {}
    sanity_by_shape: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        phase = str(record.get("phase"))
        if phase not in ("reference_sanity", "reference"):
            continue
        candidate_id = str(record.get("candidate_id"))
        for value in record.get("checks", ()):
            if not isinstance(value, Mapping) or value.get("passed") is None:
                continue
            shape_key = str(value.get("shape_id"))
            checks_by_shape.setdefault(shape_key, []).append(
                (candidate_id, phase, value)
            )
            if phase == "reference_sanity":
                sanity_by_shape.setdefault(shape_key, []).append(value)

    def finite(check: Mapping[str, object], field: str) -> float | None:
        try:
            result = float(check[field])
        except (KeyError, TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    baselines: dict[str, dict[str, float]] = {}
    for shape_key, values in checks_by_shape.items():
        source = sanity_by_shape.get(shape_key) or [check for _, _, check in values]
        metrics = {}
        for field in ("mean_abs", "max_abs", "max_rel"):
            samples = [
                sample
                for check in source
                if (sample := finite(check, field)) is not None
            ]
            if samples:
                metrics[field] = statistics.median(samples)
        baselines[shape_key] = metrics

    def ratio(value: float, baseline: float) -> float | None:
        if baseline == 0:
            return 1.0 if value == 0 else None
        return value / baseline

    flagged_rows: list[dict[str, object]] = []
    mismatch_count = 0
    checked_count = 0
    for shape_key, values in checks_by_shape.items():
        baseline = baselines[shape_key]
        for candidate_id, phase, check in values:
            mean_abs = finite(check, "mean_abs")
            max_abs = finite(check, "max_abs")
            max_rel = finite(check, "max_rel")
            mean_ratio = (
                None
                if mean_abs is None or "mean_abs" not in baseline
                else ratio(mean_abs, baseline["mean_abs"])
            )
            max_ratio = (
                None
                if max_abs is None or "max_abs" not in baseline
                else ratio(max_abs, baseline["max_abs"])
            )
            max_rel_matches = (
                max_rel is not None
                and "max_rel" in baseline
                and max_rel == baseline["max_rel"]
            )
            if max_rel is not None and "max_rel" in baseline and not max_rel_matches:
                mismatch_count += 1
            checked_count += 1
            is_flagged = any(
                current is not None
                and (
                    (base == 0 and current != 0)
                    or (base != 0 and current / base > 1.0 + tolerance)
                )
                for current, base in (
                    (mean_abs, baseline.get("mean_abs")),
                    (max_abs, baseline.get("max_abs")),
                )
                if base is not None
            )
            if is_flagged:
                flagged_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "shape_id": shape_key,
                        "phase": phase,
                        "mean_abs_ratio": mean_ratio,
                        "max_abs_ratio": max_ratio,
                        "max_rel_matches": max_rel_matches,
                        "selected": (
                            selected_ids is not None and candidate_id in selected_ids
                        ),
                    }
                )
    return {
        "tolerance": tolerance,
        "checked_count": checked_count,
        "flagged_count": len(flagged_rows),
        "max_rel_mismatch_count": mismatch_count,
        "flagged_rows": flagged_rows,
        "flagged_menu_rows": [row for row in flagged_rows if bool(row.get("selected"))],
    }


def format_speedup_summary(
    rows: Mapping[str, Mapping[str, object]],
    labels: Mapping[str, str],
    *,
    title: str = "speedup vs pre-sweep best, per shape",
) -> str:
    speedups = {
        key: float(row["speedup"])
        for key, row in rows.items()
        if row.get("speedup") is not None
    }
    missing = len(rows) - len(speedups)
    missing_text = f"; {missing} without a pre-sweep timing" if missing else ""
    if not speedups:
        return f"{title}: no comparable shapes yet{missing_text}"

    def extreme(key: str) -> str:
        value = speedups[key]
        ties = sum(round(other, 2) == round(value, 2) for other in speedups.values())
        more = f" +{ties - 1} more" if ties > 1 else ""
        return f"{value:.2f}x ({labels.get(key, key)}{more})"

    low = min(speedups, key=speedups.__getitem__)
    high = max(speedups, key=speedups.__getitem__)
    return f"{title}: min {extreme(low)}, max {extreme(high)}{missing_text}"


def menu_speedup_curve(
    selected_ids: Sequence[str],
    shapes: Sequence[PortfolioShape],
    timings: Mapping[str, Mapping[str, TimingValue]],
    incumbent_ids: set[str],
) -> list[dict[str, object]]:
    """Return deployable speedup after each greedy menu entry."""

    result: list[dict[str, object]] = []
    prefix: list[str] = []
    for candidate_id in selected_ids:
        prefix.append(candidate_id)
        weighted_log_speedup = 0.0
        covered_weight = 0.0
        speedups: list[float] = []
        for shape in shapes:
            selected_times = [
                value.time
                for selected_id in prefix
                if (value := timings.get(selected_id, {}).get(str(shape.shape_id)))
                is not None
                and value.time is not None
            ]
            incumbent_times = [
                value.time
                for incumbent_id in incumbent_ids
                if (value := timings.get(incumbent_id, {}).get(str(shape.shape_id)))
                is not None
                and value.time is not None
            ]
            if not selected_times or not incumbent_times:
                continue
            speedup = min(incumbent_times) / min(selected_times)
            speedups.append(speedup)
            weighted_log_speedup += shape.weight * math.log(speedup)
            covered_weight += shape.weight
        result.append(
            {
                "entries": len(prefix),
                "added_candidate_id": candidate_id,
                "covered_shapes": len(speedups),
                "geomean_speedup_vs_incumbent": (
                    math.exp(weighted_log_speedup / covered_weight)
                    if covered_weight
                    else None
                ),
                "minimum_speedup_vs_incumbent": (min(speedups) if speedups else None),
            }
        )
    return result


def retryable_failed_ids(records: Sequence[Mapping[str, object]]) -> list[str]:
    """Return candidates whose latest search result was transiently unusable."""
    latest: dict[str, Mapping[str, object]] = {}
    for record in records:
        if record.get("phase") == "search":
            latest[str(record.get("candidate_id"))] = record
    result = []
    for candidate_id, record in latest.items():
        status = record.get("status")
        if status in ("timeout", "worker_error"):
            result.append(candidate_id)
            continue
        if status == "error" and "outofmemoryerror" in json.dumps(record).lower():
            result.append(candidate_id)
    return result


def _resolved_signature(hashes: Mapping[object, object]) -> tuple[str, ...]:
    return tuple(sorted(f"{key}:{value}" for key, value in hashes.items()))


def _shape_payload(shape: MoeSweepShape, fc: str) -> dict[str, object]:
    payload = asdict(shape)
    payload["shape_id"] = shape_id(shape, fc)
    # Origins are reporting metadata, not launch inputs.
    payload.pop("origins", None)
    return payload


def _candidate_payload(candidate: Candidate) -> dict[str, object]:
    return {
        "options": dict(candidate.options),
        "tile_n": candidate.tile_n,
        "source_global_index": candidate.source_global_index,
        "source_raw_index": candidate.source_raw_index,
        "source_comment": candidate.source_comment,
        "scheduler_union_variant": candidate.scheduler_union_variant,
        "parent_ids": list(candidate.parent_ids),
        "move_kind": candidate.move_kind,
        "repair_kind": candidate.repair_kind,
    }


def _candidate_from_payload(value: Mapping[str, object]) -> Candidate:
    return Candidate(
        options=dict(value["options"]),
        tile_n=int(value["tile_n"]),
        source_global_index=(
            None
            if value.get("source_global_index") is None
            else int(value["source_global_index"])
        ),
        source_raw_index=(
            None
            if value.get("source_raw_index") is None
            else int(value["source_raw_index"])
        ),
        source_comment=str(value.get("source_comment", "")),
        scheduler_union_variant=bool(value.get("scheduler_union_variant", False)),
        parent_ids=tuple(str(item) for item in value.get("parent_ids", ())),
        move_kind=str(value.get("move_kind", "incumbent")),
        repair_kind=(
            None if value.get("repair_kind") is None else str(value["repair_kind"])
        ),
    )


def _dtype_context(dtype: str, fc: str) -> tuple[int, int, int, str]:
    if dtype == "NvFP4xNvFP4":
        return (
            int(DType.E2M1),
            int(DType.E2M1),
            int(DType.E2M1 if fc == "fc1" else DType.BF16),
            "NVFP4xNVFP4",
        )
    if dtype == "MxFP4xMxFP8":
        return (
            int(DType.MXE2M1),
            int(DType.MXE4M3),
            int(DType.MXE4M3 if fc == "fc1" else DType.BF16),
            "MXFP4xMXFP8",
        )
    if dtype == "MxFP8xMxFP8":
        return (
            int(DType.MXE4M3),
            int(DType.MXE4M3),
            int(DType.MXE4M3 if fc == "fc1" else DType.BF16),
            "MXFP8xMXFP8",
        )
    if dtype == "BF16":
        return int(DType.BF16), int(DType.BF16), int(DType.BF16), "BF16"
    if dtype == "MxFP4xBf16":
        return (
            int(DType.MXE2M1),
            int(DType.BF16),
            int(DType.BF16),
            "MXFP4xBF16",
        )
    raise ValueError(f"sweep context for dtype {dtype!r} is not implemented")


def context_for_shapes(
    shapes: Sequence[MoeSweepShape],
    *,
    fc: str,
    sm_arch: int,
    has_bias: bool = False,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    gemm1_clamp_limit: float | None = None,
) -> ConfigContext:
    if not shapes:
        raise ValueError("at least one shape is required")
    first = shapes[0]
    if any(
        (shape.dtype, shape.activation, shape.top_k)
        != (first.dtype, first.activation, first.top_k)
        for shape in shapes
    ):
        raise ValueError("one sweep may only contain one dtype/activation/top-k family")
    dtype_a, dtype_b, dtype_c, label = _dtype_context(first.dtype, fc)
    try:
        activation = int(getattr(ActivationType, first.activation))
    except AttributeError as exc:
        raise ValueError(f"unknown activation {first.activation!r}") from exc
    gemm1_values = (gemm1_alpha, gemm1_beta, gemm1_clamp_limit)
    if any(value is not None for value in gemm1_values) and activation not in (
        int(ActivationType.Swiglu),
        int(ActivationType.Situ),
    ):
        raise ValueError(
            "gemm1 alpha/beta/clamp_limit require Swiglu or Situ activation"
        )
    is_fc1 = fc == "fc1"
    return ConfigContext(
        sm_arch=sm_arch,
        fc=fc,
        activation_type=activation,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        dtype_c=dtype_c,
        top_k=first.top_k,
        num_local_experts=max(shape.local_num_experts for shape in shapes),
        has_bias=has_bias,
        has_gemm1_alpha=is_fc1 and gemm1_alpha is not None,
        has_gemm1_beta=is_fc1 and gemm1_beta is not None,
        has_gemm1_clamp_limit=is_fc1 and gemm1_clamp_limit is not None,
        gemm1_alpha_value=gemm1_alpha if is_fc1 else None,
        gemm1_beta_value=gemm1_beta if is_fc1 else None,
        gemm1_clamp_limit_value=gemm1_clamp_limit if is_fc1 else None,
        dtype_label=label,
        out_hiddens=tuple(sorted({int(getattr(shape, f"{fc}_n")) for shape in shapes})),
    )


def _problem_ks(shapes: Iterable[object], fc: str) -> tuple[int, ...]:
    return tuple(sorted({int(getattr(shape, f"{fc}_k")) for shape in shapes}))


def _conservative_confirmed_timing(
    search: TimingValue,
    confirmation: TimingValue,
) -> TimingValue:
    """Keep the slower timing so one anomalously fast sample cannot win."""

    if search.time is None or confirmation.time is None:
        raise ValueError("confirmation requires two measured timings")
    return TimingValue.measured(
        max(search.time, confirmation.time),
        max(search.spread, confirmation.spread),
    )


def _parent_has_unexhausted_move(
    tracker: ParentMoveTracker,
    parent_id: str,
    local_moves: Sequence[str],
    *,
    has_crossover_alternative: bool,
) -> bool:
    """Treat crossover as usable only when a distinct second parent exists."""

    if tracker.available_moves(parent_id, local_moves):
        return True
    return has_crossover_alternative and not tracker.is_exhausted(
        parent_id, "crossover"
    )


def _is_cuda_fault(error: object) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "illegal memory access",
            "device-side assert",
            "misaligned address",
            "unspecified launch failure",
            "cuda error",
        )
    )


def _is_cuda_oom(value: object) -> bool:
    """Return whether a nested worker result contains a CUDA OOM."""

    if isinstance(value, str):
        return "out of memory" in value.lower()
    if isinstance(value, Mapping):
        return any(_is_cuda_oom(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_is_cuda_oom(item) for item in value)
    return False


def summarize_gpu_clocks(
    records: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Summarize per-measurement SM clocks by sweep phase."""

    grouped: dict[str, list[tuple[int, int | None, Sequence[object]]]] = {}
    for record in records:
        phase = str(record.get("phase", "unknown"))
        for measurement in record.get("measurements", ()) or ():
            if not isinstance(measurement, Mapping):
                continue
            clock = measurement.get("sm_clock_mhz")
            if clock is None:
                continue
            maximum = measurement.get("sm_clock_max_mhz")
            reasons = measurement.get("clock_event_reasons", ()) or ()
            grouped.setdefault(phase, []).append(
                (
                    int(clock),
                    None if maximum is None else int(maximum),
                    reasons
                    if isinstance(reasons, Sequence)
                    and not isinstance(reasons, (str, bytes))
                    else (),
                )
            )
    result: dict[str, dict[str, object]] = {}
    for phase, samples in sorted(grouped.items()):
        clocks = [clock for clock, _, _ in samples]
        low_count = sum(
            maximum is not None and clock < 0.97 * maximum
            for clock, maximum, _ in samples
        )
        reason_counts: dict[str, int] = {}
        for _, _, reasons in samples:
            for reason in reasons:
                key = str(reason)
                reason_counts[key] = reason_counts.get(key, 0) + 1
        result[phase] = {
            "samples": len(samples),
            "min_sm_clock_mhz": min(clocks),
            "median_sm_clock_mhz": statistics.median(clocks),
            "max_sm_clock_mhz": max(clocks),
            "max_rated_sm_clock_mhz": max(
                (maximum for _, maximum, _ in samples if maximum is not None),
                default=None,
            ),
            "low_clock_count": low_count,
            "low_clock_fraction": low_count / len(samples),
            "clock_event_reasons": dict(sorted(reason_counts.items())),
        }
    return result


def _best_final_time(
    final_timings: Mapping[str, Mapping[str, TimingValue]],
    search_timings: Mapping[str, Mapping[str, TimingValue]],
    shape_key: str,
    fallback: float | None,
) -> float | None:
    """Prefer the final protocol; use search timing only if none was re-timed."""

    for rows in (final_timings, search_timings):
        values = [
            value.time
            for row in rows.values()
            if (value := row.get(shape_key)) is not None and value.time is not None
        ]
        if values:
            return min(values)
    return fallback


def _prune_failed_finalists(
    candidates: Sequence[PortfolioCandidate],
    timings: Mapping[str, Mapping[str, TimingValue]],
    failed: set[str],
) -> tuple[list[PortfolioCandidate], dict[str, dict[str, TimingValue]]]:
    """Remove failed finalists from both selection inputs."""

    return (
        [candidate for candidate in candidates if candidate.candidate_id not in failed],
        {
            candidate_id: dict(row)
            for candidate_id, row in timings.items()
            if candidate_id not in failed
        },
    )


def _shape_label(shape: MoeSweepShape, fc: str) -> str:
    return (
        f"T={shape.timing_num_tokens} E={shape.local_num_experts} "
        f"N={getattr(shape, f'{fc}_n')} K={getattr(shape, f'{fc}_k')} "
        f"TP/EP={shape.tp}/{shape.ep} R={shape.routing}"
    )


def _record_gpu_memory(record: dict[str, object]) -> None:
    """Attach allocator and optional NVML process memory without failing work."""

    try:
        footprint = measure_worker_gpu_footprint()
    except (ImportError, RuntimeError, OSError):
        return
    record["gpu_memory_footprint"] = footprint.to_dict()
    record["peak_gpu_memory_bytes"] = max(
        int(record.get("peak_gpu_memory_bytes") or 0),
        footprint.effective_bytes,
    )


def _reference_lock(payload: Mapping[str, object]):
    if not payload.get("reference_gpu_lock"):
        return nullcontext()

    from .workers import gpu_lock, resolve_gpu_lock_identity

    return gpu_lock(
        resolve_gpu_lock_identity(str(payload.get("gpu_lock_identity", "0"))),
    )


def run_worker_payload(
    payload: Mapping[str, object],
    *,
    weight_cache: dict[tuple[object, ...], tuple[object, object]] | None = None,
    progress_path: Path | None = None,
) -> dict[str, object]:
    """Execute one killable candidate evaluation or correctness check."""
    current_phase = "compile"

    def progress(phase: str, unit: str) -> None:
        nonlocal current_phase
        current_phase = phase
        if progress_path is not None:
            atomic_write_json(
                progress_path,
                {"phase": phase, "unit": unit, "timestamp": time.time()},
                fsync=False,
            )

    candidate = _candidate_from_payload(payload["candidate"])
    context = ConfigContext(**dict(payload["context"]))
    shapes = [SimpleNamespace(**dict(value)) for value in payload["shapes"]]
    materialized = materialize_candidate(
        candidate,
        context,
        _problem_ks(shapes, context.fc),
        check_task_manager=True,
    )
    action = str(payload.get("action", "evaluate"))
    if action == "compile":
        progress("compile", candidate.candidate_id)
        compile_ms = compile_candidate(
            materialized,
            context,
            shapes,
            gpu_lock_identity=payload.get("gpu_lock_identity"),
        )
        record = {
            "candidate_id": candidate.candidate_id,
            "status": "ok",
            "compile_ms": compile_ms,
            "measurements": [],
            "materialized": {
                "source_hash": materialized.source_hash,
                "resolved_hashes": {
                    str(key): value
                    for key, value in materialized.resolved_hashes.items()
                },
                "compile_signatures": {
                    str(key): value
                    for key, value in materialized.compile_signatures.items()
                },
                "smem_bytes": materialized.smem_bytes,
                "smem_regime_kib": materialized.smem_regime_kib,
                "tmem_columns": materialized.tmem_columns,
                "multicast_operands": list(materialized.multicast_operands),
            },
        }
        record["peak_rss_bytes"] = (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        )
        _record_gpu_memory(record)
        return record
    if action == "reference":
        checks = []
        cuda_fault = False
        for shape in shapes:
            applicable, reason = is_applicable(materialized, context, shape)
            if not applicable:
                checks.append(
                    {
                        "shape_id": str(shape.shape_id),
                        "passed": None,
                        "skipped": True,
                        "error": reason,
                        "elapsed_seconds": 0.0,
                    }
                )
                continue
            lock_wait_seconds = 0.0
            started: float | None = None
            details: dict[str, object] = {}
            try:
                if payload.get("reference_gpu_lock"):
                    progress("lock_wait", str(shape.shape_id))
                    lock_wait_started = time.perf_counter()
                with _reference_lock(payload):
                    if payload.get("reference_gpu_lock"):
                        lock_wait_seconds = time.perf_counter() - lock_wait_started
                    progress("run", str(shape.shape_id))
                    started = time.perf_counter()
                    result = reference_check_candidate(materialized, context, shape)
                if isinstance(result, Mapping):
                    details = dict(result)
                    passed_value = details.pop("passed", None)
                    passed = None if passed_value is None else bool(passed_value)
                elif hasattr(result, "passed"):
                    details = asdict(result)
                    passed = bool(details.pop("passed"))
                else:
                    passed = bool(result)
                error = None
            except Exception as exc:
                passed = None
                error = f"{type(exc).__name__}: {exc}"
                details["oom"] = "out of memory" in error.lower()
                cuda_fault = not details["oom"] and _is_cuda_fault(error)
            finally:
                try:
                    import torch

                    torch.cuda.empty_cache()
                except (ImportError, RuntimeError):
                    pass
            checks.append(
                {
                    "shape_id": str(shape.shape_id),
                    "passed": passed,
                    "error": error,
                    "elapsed_seconds": (
                        0.0 if started is None else time.perf_counter() - started
                    ),
                    "lock_wait_seconds": lock_wait_seconds,
                    **details,
                }
            )
            if cuda_fault:
                break
        record = {
            "candidate_id": candidate.candidate_id,
            "status": (
                "failed"
                if any(check.get("passed") is False for check in checks)
                else (
                    "oom"
                    if any(check.get("oom") for check in checks)
                    else (
                        "reference_error"
                        if any(
                            check["passed"] is None and not check.get("skipped")
                            for check in checks
                        )
                        else (
                            "ok"
                            if any(check["passed"] is True for check in checks)
                            else "failed"
                        )
                    )
                )
            ),
            "checks": checks,
        }
        if cuda_fault:
            record["failure"] = FailureKind.CUDA_FAULT.value
            record["_terminate_worker"] = True
        try:
            import torch

            record["gpu_memory_allocated_bytes"] = int(torch.cuda.memory_allocated())
            record["gpu_memory_reserved_bytes"] = int(torch.cuda.memory_reserved())
        except (ImportError, RuntimeError):
            pass
        _record_gpu_memory(record)
        return record
    if action != "evaluate":
        raise ValueError(f"unknown worker action {action!r}")
    protocol = TimingProtocol(**dict(payload["protocol"]))
    evaluation = evaluate_candidate(
        materialized,
        context,
        shapes,
        protocol,
        weight_cache=weight_cache,
        gpu_lock_identity=payload.get("gpu_lock_identity"),
        progress=progress,
    )
    record = evaluation.to_record()
    record["peak_rss_bytes"] = (
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    )
    _record_gpu_memory(record)
    record["materialized"] = {
        "source_hash": materialized.source_hash,
        "resolved_hashes": {
            str(key): value for key, value in materialized.resolved_hashes.items()
        },
        "compile_signatures": {
            str(key): value for key, value in materialized.compile_signatures.items()
        },
        "smem_bytes": materialized.smem_bytes,
        "smem_regime_kib": materialized.smem_regime_kib,
        "tmem_columns": materialized.tmem_columns,
        "multicast_operands": list(materialized.multicast_operands),
    }
    if not _is_cuda_oom(record) and (
        evaluation.error
        and _is_cuda_fault(evaluation.error)
        or any(
            measurement.error and _is_cuda_fault(measurement.error)
            for measurement in evaluation.measurements
        )
    ):
        record["failure"] = FailureKind.CUDA_FAULT.value
        record["_terminate_worker"] = True
    elif evaluation.error:
        record["failure"] = (
            FailureKind.COMPILE_ERROR.value
            if current_phase == "compile"
            else FailureKind.RUN_ERROR.value
        )
    elif any(
        measurement.status == "run_error" for measurement in evaluation.measurements
    ):
        record["failure"] = FailureKind.RUN_ERROR.value
    return record


def run_worker_loop() -> None:
    """Serve requests while retaining process-local compiled kernels and IO."""

    weight_cache: OrderedDict[tuple[object, ...], tuple[object, object]] = OrderedDict()
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        payload_path = Path(request["payload"])
        output_path = Path(request["output"])
        progress_path = Path(request["progress"])
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            result = run_worker_payload(
                payload, weight_cache=weight_cache, progress_path=progress_path
            )
        except Exception as exc:
            try:
                failed_phase = json.loads(
                    progress_path.read_text(encoding="utf-8")
                ).get("phase", "compile")
            except (OSError, ValueError, json.JSONDecodeError):
                failed_phase = "compile"
            result = {
                "status": "worker_error",
                "failure": (
                    FailureKind.COMPILE_ERROR.value
                    if failed_phase == "compile"
                    else FailureKind.RUN_ERROR.value
                ),
                "error": f"{type(exc).__name__}: {exc}",
            }
        terminate_worker = bool(result.pop("_terminate_worker", False))
        if not terminate_worker:
            try:
                import torch

                torch.cuda.synchronize()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if not _is_cuda_oom(error) and _is_cuda_fault(error):
                    result["status"] = "worker_error"
                    result["failure"] = FailureKind.CUDA_FAULT.value
                    result["error"] = error
                    terminate_worker = True
        if not terminate_worker:
            from flashinfer.prims_ts.moe.compile_cache import (
                clear_compiled_gemm_cache,
            )

            clear_compiled_gemm_cache()
            gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            # A sticky CUDA fault can make allocator cleanup fail.  The result
            # still has to be journaled so the parent can restart this worker.
            pass
        atomic_write_json(output_path, result)
        if terminate_worker:
            break


@dataclass(frozen=True)
class SweepOptions:
    out_dir: Path
    fc: str
    model: str | None
    sm_arch: int
    tp: tuple[int, ...]
    ep: tuple[int, ...]
    tokens: tuple[int, ...]
    final_tokens: tuple[int, ...]
    budget_minutes: float
    tail_minutes: float
    menu_size: int
    worker_count: int
    gpu: str
    seed_limit: int = SEED_LIMIT
    seed: int = RANDOM_SEED
    candidate_timeout_seconds: float = CANDIDATE_TIMEOUT_SECONDS
    max_seeds: int = MAX_SEEDS
    max_near_best_per_shape: int = DEFAULT_MAX_NEAR_BEST_PER_SHAPE
    reference_reserve_minutes: float = DEFAULT_REFERENCE_RESERVE_MINUTES
    tp_ep: tuple[tuple[int, int], ...] | None = None
    dtype: str | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_experts: int | None = None
    top_k: int | None = None
    activation: str | None = None
    has_bias: bool = False
    gemm1_alpha: float | None = None
    gemm1_beta: float | None = None
    gemm1_clamp_limit: float | None = None
    dry_run: bool = False
    warmup_iters: int = SEARCH_WARMUP_ITERS
    repeat_iters: int = SEARCH_REPEAT_ITERS
    resume: bool = False
    strict_resume: bool = False
    max_input_stages: int = DEFAULT_MAX_INPUT_STAGES
    max_accumulator_stages: int = DEFAULT_MAX_ACCUMULATOR_STAGES
    max_c_smem_stages: int = DEFAULT_MAX_C_SMEM_STAGES
    max_workid_stages: int = DEFAULT_MAX_WORKID_STAGES
    max_sf_ring_ratio: int = DEFAULT_MAX_SF_RING_RATIO
    compile_timeout_seconds: float | None = COMPILE_TIMEOUT_SECONDS
    shape_timeout_seconds: float | None = SHAPE_TIMEOUT_SECONDS
    worker_restart_configs: int = WORKER_RESTART_CONFIGS
    progress_interval_seconds: float = PROGRESS_INTERVAL_SECONDS
    search_routing: str = DEFAULT_SEARCH_ROUTING
    retime_routings: tuple[str, ...] = DEFAULT_RETIME_ROUTINGS
    selection_routings: tuple[str, ...] | None = None
    routing_seed: int = ROUTING_SEED
    max_routing_regret: float = DEFAULT_MAX_ROUTING_REGRET
    enable_cluster_y: bool = False
    retry_failed: bool = False
    confirm_fraction: float = DEFAULT_CONFIRM_FRACTION

    def __post_init__(self) -> None:
        if self.fc not in ("fc1", "fc2"):
            raise ValueError(f"fc must be fc1 or fc2, got {self.fc!r}")
        if self.worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if self.budget_minutes <= 0:
            raise ValueError("budget_minutes must be positive")
        if self.tail_minutes < 0 or self.tail_minutes >= self.budget_minutes:
            raise ValueError("tail_minutes must be nonnegative and less than budget")
        if self.menu_size <= 0 or self.seed_limit < 0 or self.max_seeds < 0:
            raise ValueError("menu_size must be positive and seed limits nonnegative")
        if self.max_near_best_per_shape < 0:
            raise ValueError("max_near_best_per_shape must be nonnegative")
        if (
            not math.isfinite(self.reference_reserve_minutes)
            or self.reference_reserve_minutes < 0
        ):
            raise ValueError("reference_reserve_minutes must be nonnegative and finite")
        if self.candidate_timeout_seconds <= 0:
            raise ValueError("candidate_timeout_seconds must be positive")
        if (
            self.compile_timeout_seconds is not None
            and self.compile_timeout_seconds <= 0
        ):
            raise ValueError("compile_timeout_seconds must be positive")
        if self.shape_timeout_seconds is not None and self.shape_timeout_seconds <= 0:
            raise ValueError("shape_timeout_seconds must be positive")
        if self.worker_restart_configs <= 0:
            raise ValueError("worker_restart_configs must be positive")
        if self.progress_interval_seconds < 0:
            raise ValueError("progress_interval_seconds must be nonnegative")
        if not self.final_tokens:
            raise ValueError("final_tokens must not be empty")
        if not self.search_routing or not self.retime_routings:
            raise ValueError("routing selections must not be empty")
        selection_routings = (
            self.retime_routings
            if self.selection_routings is None
            else tuple(dict.fromkeys(self.selection_routings))
        )
        if not selection_routings:
            raise ValueError("selection_routings must not be empty")
        missing_selection_routings = set(selection_routings) - set(self.retime_routings)
        if missing_selection_routings:
            raise ValueError(
                "selection_routings must be included in retime_routings: "
                f"{sorted(missing_selection_routings)}"
            )
        if self.search_routing not in selection_routings:
            raise ValueError(
                "selection_routings must include search_routing so the summary "
                "has a stable top-level routing"
            )
        object.__setattr__(self, "selection_routings", selection_routings)
        if self.max_routing_regret < 0 or not math.isfinite(self.max_routing_regret):
            raise ValueError("max_routing_regret must be non-negative and finite")
        if self.warmup_iters < 0 or self.repeat_iters <= 0:
            raise ValueError("timing iteration counts must be nonnegative/positive")
        if not 0 < self.confirm_fraction < 1 or not math.isfinite(
            self.confirm_fraction
        ):
            raise ValueError("confirm_fraction must be finite and in (0, 1)")
        if (
            min(
                self.max_input_stages,
                self.max_accumulator_stages,
                self.max_c_smem_stages,
                self.max_workid_stages,
                self.max_sf_ring_ratio,
            )
            < 1
        ):
            raise ValueError("stage bounds must be positive")
        resolve_visible_gpu(self.gpu)

    @property
    def stage_limits(self) -> StageLimits:
        return StageLimits(
            inputs=self.max_input_stages,
            accumulator=self.max_accumulator_stages,
            c_smem=self.max_c_smem_stages,
            workid=self.max_workid_stages,
        )


class SweepEngine:
    def __init__(
        self,
        options: SweepOptions,
        *,
        calibration: CalibrationData | None = None,
    ) -> None:
        self.options = options
        self.calibration = calibration
        calibrated_noise = 0.0 if calibration is None else calibration.noise.relative
        self.thresholds = NoiseAwareThresholds(
            success_fraction=max(0.03, 2.0 * calibrated_noise),
            two_step_fraction=max(0.10, 3.0 * calibrated_noise),
        )
        self.rng = random.Random(options.seed)
        self.search = SearchState(selector=AdaptiveMoveSelector(seed=options.seed))
        self._parent_moves = ParentMoveTracker()
        self.started = time.monotonic()
        self.deadline = self.started + options.budget_minutes * 60.0
        self.search_deadline = self.deadline - options.tail_minutes * 60.0
        self.options.out_dir.mkdir(parents=True, exist_ok=True)
        self._worker_init_lock = threading.Lock()
        self._worker_queue: queue.Queue[PersistentProcessWorker] | None = None
        self._workers: list[PersistentProcessWorker] = []
        self._executor: ThreadPoolExecutor | None = None
        self._inflight: dict[Future[dict[str, object]], tuple[Candidate, float]] = {}
        self._adaptive_proposal_metadata: dict[str, tuple[str, str, str]] = {}
        self._adaptive_reference_best: dict[str, dict[str, TimingValue]] = {}
        self._adaptive_resolved_signatures: dict[str, tuple[str, ...]] = {}
        self._smem_regime_cache: dict[str, int] = {}
        self._rejection_reasons: dict[str, dict[str, dict[str, int]]] = {}
        self._search_stop_reason = "deadline"
        self._search_stop_minute: float | None = None
        self._last_progress = 0.0
        self._last_clock_warning = 0.0
        self._search_loop_seconds: dict[str, float] = {}
        self._search_loop_calls: dict[str, int] = {}
        self._search_loop_wall_seconds = 0.0
        self._search_worker_busy_seconds = 0.0
        self._search_worker_capacity_seconds = 0.0
        self._search_loop_active_since: float | None = None
        self._search_profile_lock = threading.Lock()
        self._last_search_profile_progress = 0.0
        self.journal_path = self.options.out_dir / "results.jsonl"
        self.state_path = self.options.out_dir / "state.json"
        self.manifest_path = self.options.out_dir / "run_manifest.json"
        self.shapes = build_sweep_shapes(
            model=options.model,
            dtype=options.dtype,
            hidden_size=options.hidden_size,
            intermediate_size=options.intermediate_size,
            num_experts=options.num_experts,
            top_k=options.top_k,
            activation=options.activation,
            global_num_tokens=options.tokens,
            tp=options.tp,
            ep=options.ep,
            tp_ep=options.tp_ep,
            routing=options.search_routing,
            routing_seed=options.routing_seed,
            fc=options.fc,
        )
        self.final_shapes = merge_duplicate_shapes(
            (
                shape
                for routing in options.retime_routings
                for shape in build_sweep_shapes(
                    model=options.model,
                    dtype=options.dtype,
                    hidden_size=options.hidden_size,
                    intermediate_size=options.intermediate_size,
                    num_experts=options.num_experts,
                    top_k=options.top_k,
                    activation=options.activation,
                    global_num_tokens=options.final_tokens,
                    tp=options.tp,
                    ep=options.ep,
                    tp_ep=options.tp_ep,
                    routing=routing,
                    routing_seed=options.routing_seed,
                    fc=options.fc,
                )
            ),
            fc=options.fc,
        )
        self.context = context_for_shapes(
            self.shapes,
            fc=options.fc,
            sm_arch=options.sm_arch,
            has_bias=options.has_bias,
            gemm1_alpha=options.gemm1_alpha,
            gemm1_beta=options.gemm1_beta,
            gemm1_clamp_limit=options.gemm1_clamp_limit,
        )
        tile_ns = {tile_n for shape in self.shapes for tile_n in shape.selected_tile_ns}
        incumbents = enumerate_incumbents(self.context, tile_ns)
        self.candidates: dict[str, Candidate] = {
            candidate.candidate_id: candidate for candidate in incumbents
        }
        self.incumbent_ids = set(self.candidates)
        self._incumbents_dropped_by_cap: set[str] = set()
        self._shape_labels = {
            shape_id(shape, options.fc): _shape_label(shape, options.fc)
            for shape in (*self.shapes, *self.final_shapes)
        }
        self.records: list[dict[str, object]] = []
        self._records_version = 0
        self._timing_inputs_cache: dict[tuple[object, ...], object] = {}
        self._menu_cache: dict[tuple[object, ...], object] = {}
        self._latest_records: dict[tuple[str, str], Mapping[str, object]] = {}
        self._reference_failures: set[str] = set()
        self._confirm_timings: dict[tuple[str, str], TimingValue] = {}
        self._seed_candidates: list[Candidate] = []
        self._seed_selection_complete = False
        self.resume_segment = 0
        self.code = code_fingerprint()
        self.timing_backend = resolve_timing_backend()
        self.gpu_identity = gpu_identity(options.gpu)
        if self.gpu_identity["gpu_name"] is None and calibration is not None:
            self.gpu_identity["gpu_name"] = calibration.environment.gpu
        manifest = {
            "schema_version": 1,
            "code": self.code,
            "target": {
                "fc": options.fc,
                "model": options.model,
                "dtype": options.dtype,
                "sm_arch": options.sm_arch,
                "tp": list(options.tp),
                "ep": list(options.ep),
                "tp_ep": (
                    None
                    if options.tp_ep is None
                    else [list(pair) for pair in options.tp_ep]
                ),
                "tokens": list(options.tokens),
                "final_tokens": list(options.final_tokens),
                "search_routing": options.search_routing,
                "retime_routings": list(options.retime_routings),
                "selection_routings": list(options.selection_routings or ()),
                "routing_seed": options.routing_seed,
                "max_routing_regret": options.max_routing_regret,
                "enable_cluster_y": options.enable_cluster_y,
                "context": asdict(self.context),
                "shape_launch_keys": [
                    asdict(shape.launch_key(fc=options.fc)) for shape in self.shapes
                ],
            },
            "timing": {
                "warmup_iters": options.warmup_iters,
                "repeat_iters": options.repeat_iters,
                "candidate_timeout_seconds": options.candidate_timeout_seconds,
                "compile_timeout_seconds": options.compile_timeout_seconds,
                "shape_timeout_seconds": options.shape_timeout_seconds,
                "timing_backend": self.timing_backend,
                "cold_l2_cache": True,
                **self.gpu_identity,
                "protocol_version": TimingProtocol.protocol_version,
            },
            "search": {
                "gpu": options.gpu,
                "worker_count": options.worker_count,
                "menu_size": options.menu_size,
                "seed_limit": options.seed_limit,
                "max_seeds": options.max_seeds,
                "max_near_best_per_shape": options.max_near_best_per_shape,
                "reference_reserve_minutes": options.reference_reserve_minutes,
                "seed": options.seed,
                "max_input_stages": options.max_input_stages,
                "max_accumulator_stages": options.max_accumulator_stages,
                "max_c_smem_stages": options.max_c_smem_stages,
                "max_workid_stages": options.max_workid_stages,
                "max_sf_ring_ratio": options.max_sf_ring_ratio,
                "worker_restart_configs": options.worker_restart_configs,
                "retry_failed": options.retry_failed,
                "confirm_fraction": options.confirm_fraction,
            },
            "calibration": (
                None
                if calibration is None
                else {
                    "schema_fingerprint": calibration.schema_fingerprint,
                    "environment_digest": calibration.environment.digest,
                    "measurement_digest": deterministic_id(calibration.to_dict()),
                }
            ),
        }
        has_prior_state = self.journal_path.exists() or self.state_path.exists()
        if has_prior_state and not options.resume:
            raise FileExistsError(
                f"{options.out_dir} already contains sweep state; pass --resume "
                "to continue it"
            )
        if options.resume:
            if not self.manifest_path.exists():
                raise FileNotFoundError(
                    f"cannot resume without {self.manifest_path.name}"
                )
            prior_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            mismatches = manifest_identity_mismatches(prior_manifest, manifest)
            if mismatches:
                shown = "\n  ".join(mismatches[:10])
                more = (
                    f"\n  ... and {len(mismatches) - 10} more"
                    if len(mismatches) > 10
                    else ""
                )
                raise ValueError(
                    "resume manifest does not match this sweep invocation:\n  "
                    f"{shown}{more}"
                )
            history = list(prior_manifest.get("resume_history", ()))
            prior_code = dict(prior_manifest.get("code", {}))
            code_changed = prior_code.get("prims_ts_source_digest") != self.code.get(
                "prims_ts_source_digest"
            )
            if code_changed and options.strict_resume:
                raise ValueError(
                    "resume code fingerprint changed; omit --strict-resume "
                    "to accept a hotfix"
                )
            if code_changed:
                print(
                    "WARNING: resuming after a Prims-TS source change; "
                    "new records will be stored in a new resume segment",
                    file=sys.stderr,
                )
            self.resume_segment = (
                int(history[-1].get("segment", len(history))) if history else 0
            )
            if code_changed:
                self.resume_segment += 1
                history.append(
                    {
                        "segment": self.resume_segment,
                        "timestamp": time.time(),
                        "code": self.code,
                        "timing": manifest["timing"],
                        "search": manifest["search"],
                    }
                )
            prior_manifest["code"] = self.code
            prior_manifest["timing"] = manifest["timing"]
            prior_manifest["search"] = manifest["search"]
            prior_manifest["resume_history"] = history
            atomic_write_json(self.manifest_path, prior_manifest)
        else:
            atomic_write_json(self.manifest_path, manifest)
        if options.resume and self.journal_path.exists():
            from .db import read_jsonl

            self.records = read_jsonl(self.journal_path)
            self._records_version = len(self.records)
            for record in self.records:
                self._index_record(record)
                value = record.get("candidate")
                if isinstance(value, Mapping):
                    candidate = _candidate_from_payload(value)
                    self.candidates[candidate.candidate_id] = candidate
        if options.resume and self.state_path.exists():
            checkpoint = SweepCheckpoint.load(self.state_path)
            state_value = checkpoint.extra.get("search")
            if isinstance(state_value, Mapping):
                self.search = SearchState.from_dict(state_value)
            seed_values = checkpoint.extra.get("seed_candidates", ())
            if isinstance(seed_values, Sequence) and not isinstance(
                seed_values, (str, bytes)
            ):
                self._seed_candidates = [
                    _candidate_from_payload(value)
                    for value in seed_values
                    if isinstance(value, Mapping)
                ]
            self._seed_selection_complete = bool(
                checkpoint.extra.get("seed_selection_complete", False)
            )
            saved_incumbents = checkpoint.extra.get("incumbent_ids")
            if isinstance(saved_incumbents, Sequence) and not isinstance(
                saved_incumbents, (str, bytes)
            ):
                self.incumbent_ids = {
                    str(candidate_id)
                    for candidate_id in saved_incumbents
                    if str(candidate_id) in self.candidates
                }
            self._incumbents_dropped_by_cap.update(
                str(candidate_id)
                for candidate_id in checkpoint.extra.get(
                    "incumbents_dropped_by_cap", ()
                )
            )
            rejection_value = checkpoint.extra.get("rejection_reasons")
            if isinstance(rejection_value, Mapping):
                self._rejection_reasons = {
                    str(group): {
                        str(move): {
                            str(reason): int(count)
                            for reason, count in dict(reasons).items()
                        }
                        for move, reasons in dict(moves).items()
                    }
                    for group, moves in rejection_value.items()
                }
            tracker_value = checkpoint.extra.get("parent_moves")
            if isinstance(tracker_value, Mapping):
                self._parent_moves = ParentMoveTracker(
                    duplicate_limit=int(tracker_value.get("duplicate_limit", 2)),
                    exhausted={
                        (str(value[0]), str(value[1]))
                        for value in tracker_value.get("exhausted", ())
                    },
                    duplicate_counts={
                        (str(value[0]), str(value[1])): int(value[2])
                        for value in tracker_value.get("duplicate_counts", ())
                    },
                )
            self.rng = checkpoint.restore_rng(self.rng)
            profile_value = checkpoint.extra.get("search_loop_profile")
            if isinstance(profile_value, Mapping):
                self._search_loop_wall_seconds = float(
                    profile_value.get("wall_seconds", 0.0)
                )
                self._search_worker_busy_seconds = float(
                    profile_value.get("worker_busy_seconds", 0.0)
                )
                self._search_worker_capacity_seconds = float(
                    profile_value.get(
                        "worker_capacity_seconds",
                        self._search_loop_wall_seconds * self.options.worker_count,
                    )
                )
                phases = profile_value.get("phases", {})
                if isinstance(phases, Mapping):
                    for name, values in phases.items():
                        if not isinstance(values, Mapping):
                            continue
                        self._search_loop_seconds[str(name)] = float(
                            values.get("seconds", 0.0)
                        )
                        self._search_loop_calls[str(name)] = int(values.get("calls", 0))
            if checkpoint.sequence > len(self.records):
                raise ValueError("checkpoint sequence is ahead of the results journal")
            tail = self.records[checkpoint.sequence :]
            search_tail = [record for record in tail if record.get("phase") == "search"]
            if search_tail:
                # A crash can happen after durable result appends but before
                # the next atomic checkpoint. Reconcile those records instead
                # of silently restoring stale counters and arm statistics.
                prior_best: dict[str, TimingValue] = {}
                prior_timings: dict[str, dict[str, TimingValue]] = {}
                for record in self.records[: checkpoint.sequence]:
                    if record.get("phase") != "search":
                        continue
                    candidate_id = str(record.get("candidate_id"))
                    row = prior_timings.setdefault(candidate_id, {})
                    for measurement in record.get("measurements", ()):
                        if measurement.get("status") != "ok":
                            continue
                        shape_key = str(measurement["shape_id"])
                        value = TimingValue.measured(
                            float(measurement["median_us"]),
                            float(measurement.get("mad_us") or 0.0),
                        )
                        row[shape_key] = value
                        if (
                            shape_key not in prior_best
                            or value.time < prior_best[shape_key].time
                        ):
                            prior_best[shape_key] = value
                for record in search_tail:
                    candidate_id = str(record.get("candidate_id"))
                    if candidate_id in self.search.evaluated:
                        continue
                    candidate = self.candidates.get(candidate_id)
                    if candidate is None or candidate.move_kind not in MOVE_KINDS:
                        continue
                    group = str(candidate.tile_n)
                    if record.get("status") == "duplicate_signature":
                        self.search.selector.record_duplicate(
                            group, candidate.move_kind
                        )
                        continue
                    if record.get("status") == "validation_error":
                        self.search.selector.record_rejected(group, candidate.move_kind)
                        continue
                    child_times = {
                        str(measurement["shape_id"]): TimingValue.measured(
                            float(measurement["median_us"]),
                            float(measurement.get("mad_us") or 0.0),
                        )
                        for measurement in record.get("measurements", ())
                        if measurement.get("status") == "ok"
                    }
                    parent_id = candidate.parent_ids[0] if candidate.parent_ids else ""
                    self._mark_suspicious_records((record,), prior_best)
                    outcome = self.thresholds.classify_against_parent(
                        child_times,
                        prior_timings.get(parent_id, {}),
                        prior_best,
                    )
                    self.search.selector.record(
                        group,
                        candidate.move_kind,
                        outcome,
                    )
                    self._record_two_step_child(
                        candidate_id,
                        parent_id,
                        outcome,
                        status=str(record.get("status")),
                    )
                    self.search.evaluated.append(candidate_id)
                    self.search.dispatched += 1
                    self.search.completed += 1
                    pending_shapes = set(
                        self.search.pending_confirmations.get(candidate_id, ())
                    )
                    accepted_times = {
                        shape_key: value
                        for shape_key, value in child_times.items()
                        if shape_key not in pending_shapes
                    }
                    if accepted_times:
                        prior_timings[candidate_id] = accepted_times
                        for shape_key, value in accepted_times.items():
                            if (
                                shape_key not in prior_best
                                or value.time < prior_best[shape_key].time
                            ):
                                prior_best[shape_key] = value
                resume_seed = deterministic_id(
                    {
                        "checkpoint_sequence": checkpoint.sequence,
                        "tail_candidate_ids": [
                            str(record.get("candidate_id")) for record in search_tail
                        ],
                        "tail_statuses": [
                            str(record.get("status")) for record in search_tail
                        ],
                    }
                )
                self.rng.seed(int(resume_seed, 16))
                self.search.selector.rng.seed(int(resume_seed[-16:], 16))

    def _checkpoint(self, *, phase: str) -> None:
        checkpoint = SweepCheckpoint.capture(
            self.rng,
            sequence=len(self.records),
            extra={
                "phase": phase,
                "search": self.search.to_dict(),
                "seed_candidates": [
                    _candidate_payload(candidate) for candidate in self._seed_candidates
                ],
                "seed_selection_complete": self._seed_selection_complete,
                "incumbent_ids": sorted(self.incumbent_ids),
                "incumbents_dropped_by_cap": sorted(self._incumbents_dropped_by_cap),
                "rejection_reasons": self._rejection_reasons,
                "parent_moves": {
                    "duplicate_limit": self._parent_moves.duplicate_limit,
                    "exhausted": [
                        list(value) for value in sorted(self._parent_moves.exhausted)
                    ],
                    "duplicate_counts": [
                        [parent_id, move, count]
                        for (parent_id, move), count in sorted(
                            self._parent_moves.duplicate_counts.items()
                        )
                    ],
                },
                "search_speedup_vs_pre_sweep": {
                    key: {"shape": self._shape_labels[key], **row}
                    for key, row in self._speedup_rows().items()
                },
                "search_menu_speedup_vs_pre_sweep": {
                    key: {"shape": self._shape_labels[key], **row}
                    for key, row in self._menu_speedup_rows().items()
                },
                "search_loop_profile": self._search_loop_profile(),
            },
        )
        checkpoint.save(self.state_path)

    def _speedup_rows(self) -> dict[str, dict[str, object]]:
        search_shape_ids = [shape_id(shape, self.options.fc) for shape in self.shapes]
        return speedup_vs_pre_sweep(self.records, self.incumbent_ids, search_shape_ids)

    def _menu_speedup_rows(self) -> dict[str, dict[str, object]]:
        candidates, shapes, timings = self._timing_inputs(phase="search")
        selected: set[str] = set()
        if candidates:
            menu = self._select_menu(candidates, shapes, timings)
            selected = {str(candidate_id) for candidate_id in menu.selected}
        return speedup_vs_pre_sweep(
            self.records,
            self.incumbent_ids,
            [shape_id(shape, self.options.fc) for shape in self.shapes],
            selected_ids=selected,
        )

    def _report_speedup(self) -> None:
        best_summary = format_speedup_summary(self._speedup_rows(), self._shape_labels)
        menu_summary = format_speedup_summary(
            self._menu_speedup_rows(),
            self._shape_labels,
            title="menu-constrained speedup",
        )
        self._progress(
            f"{best_summary} | {menu_summary}",
            force=True,
        )

    def _payload(
        self,
        candidate: Candidate,
        shapes: Sequence[MoeSweepShape],
        *,
        protocol: TimingProtocol,
        action: str = "evaluate",
        reference_gpu_lock: bool = False,
    ) -> dict[str, object]:
        return {
            "action": action,
            "candidate": _candidate_payload(candidate),
            "context": asdict(self.context),
            "shapes": [_shape_payload(shape, self.options.fc) for shape in shapes],
            "protocol": asdict(protocol),
            "gpu_lock_identity": self.options.gpu,
            "reference_gpu_lock": reference_gpu_lock,
        }

    def _smem_regime(self, candidate: Candidate, problem_ks: Sequence[int]) -> int:
        cached = self._smem_regime_cache.get(candidate.candidate_id)
        if cached is not None:
            return cached
        materialized = materialize_candidate(
            candidate,
            self.context,
            problem_ks,
            check_task_manager=True,
        )
        self._smem_regime_cache[candidate.candidate_id] = materialized.smem_regime_kib
        return materialized.smem_regime_kib

    def _base_worker_env(self) -> dict[str, str]:
        environment = dict(os.environ)
        inherited_visible = environment.get("CUDA_VISIBLE_DEVICES")
        selected_gpu = resolve_visible_gpu(self.options.gpu, inherited_visible)
        environment["CUDA_VISIBLE_DEVICES"] = selected_gpu
        environment.setdefault("OMP_NUM_THREADS", "1")
        allocator_config = environment.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in allocator_config:
            environment["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(
                value
                for value in (allocator_config, "expandable_segments:True")
                if value
            )
        return environment

    def _make_persistent_worker(
        self, *, name: str, restart_after: int
    ) -> PersistentProcessWorker:
        worker_env = self._base_worker_env()
        return PersistentProcessWorker(
            (
                sys.executable,
                "-m",
                "flashinfer.prims_ts.moe.tools.sweep",
                "worker-loop",
            ),
            cwd=repo_root(),
            env=worker_env,
            log_path=self.options.out_dir / ".worker_logs" / f"{name}.log",
            restart_after=restart_after,
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _progress(self, message: str, *, force: bool = False) -> None:
        interval = self.options.progress_interval_seconds
        if interval <= 0:
            return
        now = time.monotonic()
        if not force and now - self._last_progress < interval:
            return
        self._last_progress = now
        elapsed = self._format_duration(now - self.started)
        remaining = self._format_duration(self.deadline - now)
        print(
            f"[sweep elapsed={elapsed} remaining={remaining}] {message}",
            file=sys.stdout,
            flush=True,
        )

    def _start_search_profile(self) -> None:
        now = time.monotonic()
        with self._search_profile_lock:
            if self._search_loop_active_since is None:
                self._search_loop_active_since = now
        self._last_search_profile_progress = now

    def _stop_search_profile(self) -> None:
        now = time.monotonic()
        with self._search_profile_lock:
            if self._search_loop_active_since is not None:
                elapsed = now - self._search_loop_active_since
                self._search_loop_wall_seconds += elapsed
                self._search_worker_capacity_seconds += (
                    elapsed * self.options.worker_count
                )
                self._search_loop_active_since = None

    def _record_search_phase(self, name: str, started: float) -> None:
        duration = max(0.0, time.monotonic() - started)
        with self._search_profile_lock:
            self._search_loop_seconds[name] = (
                self._search_loop_seconds.get(name, 0.0) + duration
            )
            self._search_loop_calls[name] = self._search_loop_calls.get(name, 0) + 1

    def _record_search_worker_busy(self, started: float) -> None:
        duration = max(0.0, time.monotonic() - started)
        with self._search_profile_lock:
            if self._search_loop_active_since is not None:
                self._search_worker_busy_seconds += duration

    def _search_loop_profile(self) -> dict[str, object]:
        now = time.monotonic()
        with self._search_profile_lock:
            wall_seconds = self._search_loop_wall_seconds
            worker_capacity_seconds = self._search_worker_capacity_seconds
            if self._search_loop_active_since is not None:
                active_seconds = now - self._search_loop_active_since
                wall_seconds += active_seconds
                worker_capacity_seconds += active_seconds * self.options.worker_count
            worker_busy_seconds = self._search_worker_busy_seconds
            phase_seconds = dict(self._search_loop_seconds)
            phase_calls = dict(self._search_loop_calls)
        worker_utilization = (
            worker_busy_seconds / worker_capacity_seconds
            if worker_capacity_seconds > 0
            else 0.0
        )
        profiled_seconds = sum(phase_seconds.values())
        return {
            "wall_seconds": wall_seconds,
            "worker_busy_seconds": worker_busy_seconds,
            "worker_capacity_seconds": worker_capacity_seconds,
            "worker_utilization": worker_utilization,
            "profiled_main_thread_seconds": profiled_seconds,
            "unprofiled_main_thread_seconds": max(0.0, wall_seconds - profiled_seconds),
            "phases": {
                name: {
                    "seconds": seconds,
                    "calls": phase_calls.get(name, 0),
                    "wall_fraction": (
                        seconds / wall_seconds if wall_seconds > 0 else 0.0
                    ),
                }
                for name, seconds in sorted(phase_seconds.items())
            },
        }

    def _report_search_profile(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_search_profile_progress < 60.0:
            return
        self._last_search_profile_progress = now
        profile = self._search_loop_profile()
        phases = profile["phases"]
        assert isinstance(phases, Mapping)
        busiest = sorted(
            (
                (name, float(values.get("seconds", 0.0)))
                for name, values in phases.items()
                if isinstance(values, Mapping)
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:4]
        phase_text = ", ".join(f"{name}={seconds:.1f}s" for name, seconds in busiest)
        self._progress(
            "adaptive profile: "
            f"worker utilization={float(profile['worker_utilization']):.1%}; "
            f"main-thread {phase_text or 'no samples'}",
            force=True,
        )

    def _ensure_worker_pool(self) -> queue.Queue[PersistentProcessWorker]:
        if self._worker_queue is not None:
            return self._worker_queue
        with self._worker_init_lock:
            if self._worker_queue is not None:
                return self._worker_queue
            worker_queue: queue.Queue[PersistentProcessWorker] = queue.Queue()
            for index in range(self.options.worker_count):
                worker = self._make_persistent_worker(
                    name=f"worker-{index}",
                    restart_after=self.options.worker_restart_configs,
                )
                self._workers.append(worker)
                worker_queue.put(worker)
            self._worker_queue = worker_queue
            return worker_queue

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._inflight.clear()
        self._adaptive_resolved_signatures.clear()
        for worker in self._workers:
            worker.close()
        self._workers.clear()
        self._worker_queue = None

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=max(1, self.options.worker_count)
            )
        return self._executor

    def _consume_future(
        self,
        writer: JSONLWriter,
        future: Future[dict[str, object]],
        candidate: Candidate,
        *,
        phase: str,
        record_results: bool = True,
    ) -> dict[str, object]:
        try:
            result = future.result()
        except Exception as exc:
            result = {
                "candidate_id": candidate.candidate_id,
                "status": "driver_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        return (
            self._record(writer, candidate, result, phase=phase)
            if record_results
            else {
                **dict(result),
                "candidate_id": candidate.candidate_id,
            }
        )

    def _harvest_search_inflight(
        self,
        writer: JSONLWriter,
        *,
        wait_for_all: bool = False,
        wait_for_one: bool = False,
    ) -> list[dict[str, object]]:
        if not self._inflight:
            return []
        futures = set(self._inflight)
        done = (
            wait(futures)[0]
            if wait_for_all
            else (
                wait(futures, return_when=FIRST_COMPLETED)[0]
                if wait_for_one
                else {future for future in futures if future.done()}
            )
        )
        records = []
        for future in done:
            candidate, _ = self._inflight.pop(future)
            records.append(
                self._consume_future(
                    writer,
                    future,
                    candidate,
                    phase="search",
                )
            )
        return records

    def dispatch(
        self,
        candidate: Candidate,
        shapes: Sequence[MoeSweepShape],
        *,
        protocol: TimingProtocol,
        action: str = "evaluate",
        cache_tag: str | None = None,
        deadline: float | None = None,
        reference_gpu_lock: bool = False,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory(
            prefix="flashinfer-prims-ts-sweep-"
        ) as temporary:
            root = Path(temporary)

            def invoke(worker_action: str) -> dict[str, object]:
                payload_path = root / f"{worker_action}-payload.json"
                output_path = root / f"{worker_action}-output.json"
                progress_path = root / f"{worker_action}-progress.json"
                atomic_write_json(
                    payload_path,
                    self._payload(
                        candidate,
                        shapes,
                        protocol=protocol,
                        action=worker_action,
                        reference_gpu_lock=reference_gpu_lock,
                    ),
                    fsync=False,
                )
                worker_queue = None
                if cache_tag is None:
                    worker_queue = self._ensure_worker_pool()
                    worker = worker_queue.get()
                else:
                    worker = self._make_persistent_worker(
                        name=f"calibration-{cache_tag}", restart_after=1
                    )
                compile_timeout = (
                    self.options.compile_timeout_seconds
                    or self.options.candidate_timeout_seconds
                )
                run_timeout = (
                    self.options.shape_timeout_seconds
                    or self.options.candidate_timeout_seconds
                )
                worker_started = time.monotonic()
                try:
                    process_result = worker.run_request(
                        payload_path,
                        output_path,
                        progress_path,
                        compile_timeout_seconds=compile_timeout,
                        run_timeout_seconds=run_timeout,
                        hard_timeout_seconds=max(
                            0.001,
                            (self.deadline if deadline is None else deadline)
                            - time.monotonic(),
                        ),
                    )
                finally:
                    self._record_search_worker_busy(worker_started)
                    if worker_queue is None:
                        worker.close()
                    else:
                        worker_queue.put(worker)
                if not process_result.ok:
                    timed_out = process_result.failure in (
                        FailureKind.COMPILE_TIMEOUT,
                        FailureKind.LOCK_TIMEOUT,
                        FailureKind.RUN_TIMEOUT,
                    )
                    summary = describe_worker_failure(
                        process_result,
                        compile_timeout_seconds=compile_timeout,
                        run_timeout_seconds=run_timeout,
                    )
                    return {
                        "candidate_id": candidate.candidate_id,
                        "status": "timeout" if timed_out else "worker_error",
                        "failure": process_result.failure.value,
                        "failure_phase": process_result.phase,
                        "failure_unit": process_result.phase_unit,
                        "failure_phase_seconds": process_result.phase_elapsed_seconds,
                        "error": f"{summary}\n{process_result.stderr[-4000:]}",
                        "elapsed_seconds": process_result.elapsed_seconds,
                    }
                return json.loads(output_path.read_text(encoding="utf-8"))

            return invoke(action)

    def _batches(
        self, items: Sequence[Candidate]
    ) -> Iterable[tuple[str, Sequence[Candidate]]]:
        size = max(1, self.options.worker_count)
        total = math.ceil(len(items) / size)
        for index, offset in enumerate(range(0, len(items), size), start=1):
            yield f"batch {index}/{total}", items[offset : offset + size]

    def _index_record(self, record: Mapping[str, object]) -> None:
        phase = str(record.get("phase"))
        candidate_id = str(record.get("candidate_id"))
        self._latest_records[(phase, candidate_id)] = record
        if phase == "reference" and record.get("status") == "failed":
            self._reference_failures.add(candidate_id)
        if phase == "confirm":
            for measurement in record.get("measurements", ()) or ():
                if measurement.get("status") != "ok":
                    continue
                shape_key = str(measurement["shape_id"])
                self._confirm_timings[(candidate_id, shape_key)] = TimingValue.measured(
                    float(measurement["median_us"]),
                    float(measurement.get("mad_us") or 0.0),
                )

    def _record(
        self,
        writer: JSONLWriter,
        candidate: Candidate,
        result: Mapping[str, object],
        *,
        phase: str,
    ) -> dict[str, object]:
        record = {
            **dict(result),
            "schema_version": 1,
            "phase": phase,
            "candidate": _candidate_payload(candidate),
            "candidate_id": candidate.candidate_id,
            "fc": self.options.fc,
            "tile_n": candidate.tile_n,
            "move_kind": candidate.move_kind,
            "parent_ids": list(candidate.parent_ids),
            "dispatch_order": len(self.records),
            "timestamp": time.time(),
            "resume_segment": self.resume_segment,
            "environment_digest": (
                None
                if self.calibration is None
                else self.calibration.environment.digest
            ),
        }
        low_clocks = [
            (
                int(measurement["sm_clock_mhz"]),
                int(measurement["sm_clock_max_mhz"]),
            )
            for measurement in record.get("measurements", ()) or ()
            if isinstance(measurement, Mapping)
            and measurement.get("sm_clock_mhz") is not None
            and measurement.get("sm_clock_max_mhz") is not None
            and int(measurement["sm_clock_mhz"])
            < 0.97 * int(measurement["sm_clock_max_mhz"])
        ]
        now = time.monotonic()
        if low_clocks and now - self._last_clock_warning >= 300.0:
            current, maximum = min(low_clocks, key=lambda value: value[0] / value[1])
            self._progress(
                f"WARNING: low GPU SM clock {current}/{maximum} MHz during {phase}",
                force=True,
            )
            self._last_clock_warning = now
        writer.append(record)
        self.records.append(record)
        self._index_record(record)
        self._records_version += 1
        self._timing_inputs_cache.clear()
        self._menu_cache.clear()
        return record

    def _completed_ids(self, phase: str = "search") -> set[str]:
        return {
            str(record["candidate_id"])
            for record in self.records
            if record.get("phase") == phase
            and record.get("status")
            in (
                "ok",
                "error",
                "timeout",
                "worker_error",
                "validation_error",
                "duplicate_signature",
            )
        }

    def _evaluate_batch(
        self,
        writer: JSONLWriter,
        candidates: Sequence[Candidate],
        *,
        protocol: TimingProtocol,
        phase: str,
        progress_label: str | None = None,
        shapes_by_candidate: Mapping[str, Sequence[MoeSweepShape]] | None = None,
        record_results: bool = True,
        deadline: float | None = None,
        allow_stragglers: bool = False,
    ) -> list[dict[str, object]]:
        if not candidates:
            return []
        label = progress_label or phase
        batch_started = time.monotonic()
        shapes_for = {
            candidate.candidate_id: (
                list(shapes_by_candidate[candidate.candidate_id])
                if shapes_by_candidate is not None
                else [
                    shape
                    for shape in self.shapes
                    if candidate.tile_n in shape.selected_tile_ns
                ]
            )
            for candidate in candidates
        }
        empty_candidates = [
            candidate
            for candidate in candidates
            if not shapes_for[candidate.candidate_id]
        ]
        results: list[dict[str, object]] = []
        for candidate in empty_candidates:
            result = {
                "status": "validation_error",
                "error": "no_live_shapes",
                "measurements": [],
            }
            results.append(
                self._record(writer, candidate, result, phase=phase)
                if record_results
                else {**result, "candidate_id": candidate.candidate_id}
            )
        if empty_candidates:
            empty_ids = {candidate.candidate_id for candidate in empty_candidates}
            candidates = [
                candidate
                for candidate in candidates
                if candidate.candidate_id not in empty_ids
            ]
        if not candidates:
            return results
        search_reference_best = (
            self._best_by_shape(self._timing_inputs(phase="search")[2])
            if phase == "search"
            else {}
        )
        searched = {
            str(record.get("candidate_id")): str(record.get("status"))
            for record in self.records
            if record.get("phase") == "search"
        }
        searched_ok = sum(status == "ok" for status in searched.values())
        self._progress(
            f"{label}: dispatching {len(candidates)} candidate(s) x "
            f"{_shape_count_text([len(v) for v in shapes_for.values()])} shape(s) "
            f"on {self.options.worker_count} worker(s); so far "
            f"{len(searched)} candidate(s) searched ({searched_ok} ok), "
            f"{len(self.records)} journal record(s)",
            force=True,
        )
        status_counts: dict[str, int] = {}
        heartbeat_key: tuple[object, ...] | None = None
        heartbeat_at = batch_started
        heartbeat_quiet_seconds = max(
            60.0, 4.0 * self.options.progress_interval_seconds
        )
        if deadline is None and phase == "search":
            deadline = min(self.deadline, self.search_deadline + 30.0)
        pool = self._ensure_executor()
        futures = {
            pool.submit(
                self.dispatch,
                candidate,
                shapes_for[candidate.candidate_id],
                protocol=protocol,
                deadline=deadline,
            ): candidate
            for candidate in candidates
        }
        pending = set(futures)
        completion_durations: list[float] = []
        poll_seconds = (
            None
            if self.options.progress_interval_seconds <= 0
            else min(5.0, max(1.0, self.options.progress_interval_seconds))
        )
        while pending:
            done, pending = wait(
                pending,
                timeout=poll_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                now = time.monotonic()
                snapshots = [worker.activity for worker in list(self._workers)]
                activity_key, activity = summarize_worker_activity(
                    (value for value in snapshots if value is not None),
                    now=now,
                )
                key = (len(results), activity_key)
                # Ages grow on every poll; only report when requests move
                # between phases, or periodically so a stall stays visible.
                if (
                    key != heartbeat_key
                    or now - heartbeat_at >= heartbeat_quiet_seconds
                ):
                    before = self._last_progress
                    self._progress(
                        f"{label}: {len(results)}/{len(candidates)} done, "
                        f"{len(pending)} in flight "
                        f"(batch {_short_duration(now - batch_started)}) | "
                        f"{activity}"
                    )
                    if self._last_progress != before:
                        heartbeat_key = key
                        heartbeat_at = now
            for future in done:
                candidate = futures[future]
                record = self._consume_future(
                    writer,
                    future,
                    candidate,
                    phase=phase,
                    record_results=record_results,
                )
                results.append(record)
                completion_durations.append(time.monotonic() - batch_started)
                status = str(record.get("status", "unknown"))
                failure = record.get("failure")
                if status != "ok" and failure not in (None, "", "none"):
                    status = f"{status}/{failure}"
                status_counts[status] = status_counts.get(status, 0) + 1
            completed = len(results)
            elapsed = max(time.monotonic() - batch_started, 1e-9)
            rate = completed / elapsed
            eta = (
                self._format_duration((len(candidates) - completed) / rate)
                if rate > 0
                else "unknown"
            )
            statuses = ", ".join(
                f"{key}={value}" for key, value in sorted(status_counts.items())
            )
            self._progress(
                f"{label}: completed {completed}/{len(candidates)} "
                f"({rate * 60.0:.1f}/min, batch ETA {eta}); {statuses}",
                force=completed == len(candidates),
            )
            if allow_stragglers and pending and completion_durations:
                straggler_limit = max(1, self.options.worker_count // 8)
                straggler_age = max(
                    30.0,
                    3.0 * statistics.median(completion_durations),
                )
                if len(pending) <= straggler_limit and elapsed >= straggler_age:
                    for future in pending:
                        self._inflight[future] = (
                            futures[future],
                            batch_started,
                        )
                    self._progress(
                        f"{label}: carrying {len(pending)} straggler(s) "
                        "into the next adaptive round",
                        force=True,
                    )
                    break
        if search_reference_best:
            self._mark_suspicious_records(results, search_reference_best)
        return results

    def _mark_suspicious_records(
        self,
        records: Sequence[Mapping[str, object]],
        reference_best: Mapping[str, TimingValue],
    ) -> None:
        for record in records:
            candidate_id = str(record.get("candidate_id"))
            suspicious = []
            for measurement in record.get("measurements", ()):
                if measurement.get("status") != "ok":
                    continue
                shape_key = str(measurement["shape_id"])
                baseline = reference_best.get(shape_key)
                if (
                    baseline is not None
                    and baseline.time is not None
                    and float(measurement["median_us"])
                    < baseline.time * (1.0 - self.options.confirm_fraction)
                ):
                    suspicious.append(shape_key)
            if suspicious:
                pending = self.search.pending_confirmations.setdefault(candidate_id, [])
                pending.extend(
                    shape_key for shape_key in suspicious if shape_key not in pending
                )

    def _process_adaptive_records(
        self,
        records: Sequence[Mapping[str, object]],
        timings_before: Mapping[str, Mapping[str, TimingValue]],
    ) -> None:
        if not records:
            return
        _, _, updated_timings = self._timing_inputs(phase="search")
        touched_groups: set[str] = set()
        for record in records:
            candidate_id = str(record.get("candidate_id"))
            self._adaptive_resolved_signatures.pop(candidate_id, None)
            metadata = self._adaptive_proposal_metadata.pop(candidate_id, None)
            if metadata is None or candidate_id not in self.candidates:
                continue
            group, move, parent_id = metadata
            touched_groups.add(group)
            child_times = updated_timings.get(candidate_id, {})
            if not child_times:
                child_times = {
                    str(measurement["shape_id"]): TimingValue.measured(
                        float(measurement["median_us"]),
                        float(measurement.get("mad_us") or 0.0),
                    )
                    for measurement in record.get("measurements", ())
                    if measurement.get("status") == "ok"
                }
            reference_best = self._adaptive_reference_best.pop(
                candidate_id,
                self._best_by_shape(timings_before),
            )
            self._mark_suspicious_records((record,), reference_best)
            outcome = self.thresholds.classify_against_parent(
                child_times,
                updated_timings.get(parent_id, timings_before.get(parent_id, {})),
                reference_best,
            )
            self.search.selector.record(group, move, outcome)
            self.search.evaluated.append(candidate_id)
            self.search.completed += 1
            self.search.dispatched += 1
            self._record_two_step_child(
                candidate_id,
                parent_id,
                outcome,
                status=str(record.get("status")),
            )

        updated_candidates, updated_shapes, updated_timings = self._timing_inputs(
            phase="search"
        )
        eligible = {
            str(tile_n): {
                shape_id(shape, self.options.fc)
                for shape in self.shapes
                if tile_n in shape.selected_tile_ns
            }
            for tile_n in {candidate.tile_n for candidate in updated_candidates}
        }
        for group in touched_groups:
            live_shapes = [
                shape
                for shape in updated_shapes
                if shape.shape_id in eligible.get(group, set())
            ]
            group_objective = group_best_objective(
                (
                    candidate.candidate_id
                    for candidate in updated_candidates
                    if candidate.tile_n == int(group)
                ),
                live_shapes,
                updated_timings,
            )
            if group_objective is not None:
                self.search.groups.setdefault(group, GroupProgress()).record(
                    group_objective
                )

    def _record_two_step_child(
        self,
        candidate_id: str,
        parent_id: str,
        outcome: CandidateOutcome,
        *,
        status: str,
    ) -> None:
        if status != "ok" or outcome is not CandidateOutcome.TWO_STEP:
            return
        child_depth = self.search.two_step_depth.get(parent_id, -1) + 1
        if child_depth < 2:
            self.search.two_step_remaining[candidate_id] = 4
            self.search.two_step_depth[candidate_id] = child_depth

    def _confirm_pending(
        self,
        writer: JSONLWriter,
        protocol: TimingProtocol,
        *,
        deadline: float | None = None,
    ) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            return
        pending = {
            candidate_id: tuple(shape_ids)
            for candidate_id, shape_ids in self.search.pending_confirmations.items()
            if candidate_id in self.candidates and shape_ids
        }
        if not pending:
            return
        shape_by_id = {shape_id(shape, self.options.fc): shape for shape in self.shapes}
        candidates = []
        shapes_by_candidate: dict[str, list[MoeSweepShape]] = {}
        for candidate_id, shape_ids in pending.items():
            shapes = [
                shape_by_id[shape_key]
                for shape_key in shape_ids
                if shape_key in shape_by_id
            ]
            if not shapes:
                continue
            candidates.append(self.candidates[candidate_id])
            shapes_by_candidate[candidate_id] = shapes
        if not candidates:
            return
        records = self._evaluate_batch(
            writer,
            candidates,
            protocol=protocol,
            phase="confirm",
            progress_label="confirm suspicious improvements",
            shapes_by_candidate=shapes_by_candidate,
            deadline=deadline,
        )
        confirmed = {
            (str(record.get("candidate_id")), str(measurement["shape_id"]))
            for record in records
            for measurement in record.get("measurements", ())
            if measurement.get("status") == "ok"
        }
        for candidate_id, shape_key in confirmed:
            values = self.search.pending_confirmations.get(candidate_id)
            if values is None:
                continue
            if shape_key in values:
                values.remove(shape_key)
            if not values:
                self.search.pending_confirmations.pop(candidate_id, None)

    def _select_menu(
        self,
        candidates: Sequence[PortfolioCandidate],
        shapes: Sequence[PortfolioShape],
        timings: Mapping[str, Mapping[str, TimingValue]],
    ):
        key = (
            self._records_version,
            tuple(candidate.candidate_id for candidate in candidates),
            tuple(shape.shape_id for shape in shapes),
            tuple(
                (candidate_id, tuple(sorted(shape_ids)))
                for candidate_id, shape_ids in sorted(
                    self.search.pending_confirmations.items()
                )
            ),
        )
        cached = self._menu_cache.get(key)
        if cached is None:
            cached = select_greedy_menu(
                candidates,
                shapes,
                timings,
                menu_size=self.options.menu_size,
            )
            self._menu_cache[key] = cached
        return cached

    def _timing_inputs(
        self,
        *,
        phase: str,
        allowed: set[str] | None = None,
        shapes: Sequence[MoeSweepShape] | None = None,
    ) -> tuple[
        list[PortfolioCandidate],
        list[PortfolioShape],
        dict[str, dict[str, TimingValue]],
    ]:
        cache_key = (
            self._records_version,
            phase,
            None if allowed is None else tuple(sorted(allowed)),
            None
            if shapes is None
            else tuple(shape_id(shape, self.options.fc) for shape in shapes),
            tuple(
                (candidate_id, tuple(sorted(shape_ids)))
                for candidate_id, shape_ids in sorted(
                    self.search.pending_confirmations.items()
                )
            ),
        )
        cached = self._timing_inputs_cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        candidate_ids = []
        timings: dict[str, dict[str, TimingValue]] = {}
        failed_ids = set(self._reference_failures)
        latest_records: dict[str, Mapping[str, object]] = {}
        for (record_phase, candidate_id), record in self._latest_records.items():
            if record_phase != phase:
                continue
            if allowed is not None and candidate_id not in allowed:
                continue
            if candidate_id not in self.candidates:
                continue
            latest_records[candidate_id] = record
        for candidate_id, record in latest_records.items():
            candidate_ids.append(candidate_id)
            row = timings.setdefault(candidate_id, {})
            if record.get("status") != "ok":
                failed_ids.add(candidate_id)
            for measurement in record.get("measurements", ()):
                if measurement.get("status") == "ok":
                    shape_key = str(measurement["shape_id"])
                    pair = (candidate_id, shape_key)
                    if (
                        phase == "search"
                        and shape_key
                        in self.search.pending_confirmations.get(candidate_id, ())
                    ):
                        continue
                    value = TimingValue.measured(
                        float(measurement["median_us"]),
                        float(measurement.get("mad_us") or 0.0),
                    )
                    confirmation = (
                        self._confirm_timings.get(pair) if phase == "search" else None
                    )
                    if confirmation is not None:
                        value = _conservative_confirmed_timing(value, confirmation)
                    row[shape_key] = value
                elif measurement.get("status") not in ("not_applicable",):
                    row[str(measurement["shape_id"])] = FAILED
                    failed_ids.add(candidate_id)
        ordered_ids = [
            candidate_id
            for candidate_id in dict.fromkeys(candidate_ids)
            if candidate_id not in failed_ids
            and any(
                value.is_measured for value in timings.get(candidate_id, {}).values()
            )
        ]
        timings = {
            candidate_id: row
            for candidate_id, row in timings.items()
            if candidate_id not in failed_ids
        }
        portfolio_candidates = [
            PortfolioCandidate(candidate_id, self.candidates[candidate_id].tile_n)
            for candidate_id in ordered_ids
        ]
        portfolio_shapes = [
            PortfolioShape(shape_id(shape, self.options.fc), shape.weight)
            for shape in (self.shapes if shapes is None else shapes)
        ]
        result = (portfolio_candidates, portfolio_shapes, timings)
        self._timing_inputs_cache[cache_key] = result
        return result

    def _baseline(self, writer: JSONLWriter, protocol: TimingProtocol) -> None:
        completed = self._completed_ids()
        raw_pending = [
            candidate
            for candidate_id, candidate in self.candidates.items()
            if candidate_id not in completed
        ]
        self._progress(
            f"baseline: validating {len(raw_pending)} pending incumbent(s); "
            f"{len(completed)} already complete",
            force=True,
        )
        signature_owner: dict[tuple[str, ...], str] = {}
        for record in self.records:
            materialized_record = record.get("materialized", {})
            if not isinstance(materialized_record, Mapping):
                continue
            hashes = materialized_record.get("resolved_hashes", {})
            if isinstance(hashes, Mapping) and hashes:
                signature_owner[_resolved_signature(hashes)] = str(
                    record["candidate_id"]
                )
        pending = []
        for candidate in raw_pending:
            try:
                materialized = materialize_candidate(
                    candidate,
                    self.context,
                    _problem_ks(self.shapes, self.options.fc),
                    check_task_manager=True,
                )
            except Exception as exc:
                self._record(
                    writer,
                    candidate,
                    {
                        "status": "validation_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    phase="search",
                )
                continue
            signature = _resolved_signature(materialized.resolved_hashes)
            owner = signature_owner.get(signature)
            if owner is not None:
                self._record(
                    writer,
                    candidate,
                    {
                        "status": "duplicate_signature",
                        "duplicate_of": owner,
                        "materialized": {
                            "resolved_hashes": {
                                str(key): value
                                for key, value in materialized.resolved_hashes.items()
                            }
                        },
                    },
                    phase="search",
                )
                continue
            signature_owner[signature] = candidate.candidate_id
            pending.append(candidate)
        pending, dropped_ids = cap_and_prune_candidates(
            self.candidates,
            self.incumbent_ids,
            pending,
            limit=self.options.max_seeds,
        )
        self._incumbents_dropped_by_cap.update(dropped_ids)
        self._progress(
            f"baseline: {len(pending)} unique candidate(s) ready to evaluate",
            force=True,
        )
        for batch_label, batch in self._batches(pending):
            if time.monotonic() >= self.search_deadline:
                break
            self._evaluate_batch(
                writer,
                batch,
                protocol=protocol,
                phase="search",
                progress_label=f"baseline {batch_label}",
            )
            self._checkpoint(phase="baseline")

    def _retry_failed(self, writer: JSONLWriter, protocol: TimingProtocol) -> None:
        pending = [
            self.candidates[candidate_id]
            for candidate_id in retryable_failed_ids(self.records)
            if candidate_id in self.candidates
        ]
        self._progress(
            f"retry: {len(pending)} transiently failed candidate(s)",
            force=True,
        )
        for batch_label, batch in self._batches(pending):
            if time.monotonic() >= self.search_deadline:
                break
            self._evaluate_batch(
                writer,
                batch,
                protocol=protocol,
                phase="search",
                progress_label=f"retry {batch_label}",
            )
            self._checkpoint(phase="retry")

    def _reference_shapes_for_candidate(
        self,
        candidate: Candidate,
        *,
        shapes: Sequence[MoeSweepShape] | None = None,
    ) -> list[MoeSweepShape]:
        """Choose one representative check per K using realistic routing."""

        source = self.final_shapes if shapes is None else shapes
        materialized = materialize_candidate(
            candidate,
            self.context,
            _problem_ks(source, self.options.fc),
            check_task_manager=False,
        )
        applicable = [
            shape
            for shape in source
            if candidate.tile_n in shape.selected_tile_ns
            and is_applicable(materialized, self.context, shape)[0]
        ]
        by_k: dict[int, list[MoeSweepShape]] = {}
        for shape in applicable:
            by_k.setdefault(int(getattr(shape, f"{self.options.fc}_k")), []).append(
                shape
            )
        checks: list[MoeSweepShape] = []
        for k_shapes in by_k.values():
            preferred = [
                shape
                for shape in k_shapes
                if shape.routing == self.options.search_routing
            ]
            if not preferred:
                preferred = [
                    shape
                    for shape in k_shapes
                    if shape.routing != "balanced_round_robin"
                ] or k_shapes
            ordered = sorted(preferred, key=lambda value: value.global_num_tokens)
            checks.append(ordered[len(ordered) // 2])
        # Balanced-only invocations still need a non-uniform tail.
        if checks and all(shape.routing == "balanced_round_robin" for shape in checks):
            center = checks[0]
            uneven = model_moe_shape(
                model=None,
                dtype=center.dtype,
                hidden_size=center.hidden_size,
                intermediate_size=center.intermediate_size,
                num_experts=center.num_experts,
                top_k=center.top_k,
                activation=center.activation,
                global_num_tokens=center.global_num_tokens + 1,
                tp=center.tp,
                ep=center.ep,
                routing=center.routing,
            )
            if (
                candidate.tile_n in uneven.selected_tile_ns
                and is_applicable(materialized, self.context, uneven)[0]
            ):
                checks.append(uneven)
        return checks

    def _reference_batch(
        self,
        writer: JSONLWriter,
        candidates: Sequence[Candidate],
        *,
        protocol: TimingProtocol,
        phase: str,
        shapes_by_candidate: Mapping[str, Sequence[MoeSweepShape]] | None = None,
        retries: int = 2,
        gpu_lock: bool = True,
    ) -> list[dict[str, object]]:
        """Reference-check candidates concurrently and journal every result."""

        prior = {
            str(record.get("candidate_id")): record
            for record in self.records
            if record.get("phase") == phase and record.get("status") in ("ok", "failed")
        }
        pending = [
            candidate for candidate in candidates if candidate.candidate_id not in prior
        ]
        results = list(prior.values())

        def invoke(candidate: Candidate) -> tuple[Candidate, dict[str, object]]:
            checks = (
                list(shapes_by_candidate[candidate.candidate_id])
                if shapes_by_candidate is not None
                else self._reference_shapes_for_candidate(candidate)
            )
            result: dict[str, object] = {}
            for attempt in range(retries + 1):
                result = self.dispatch(
                    candidate,
                    checks,
                    protocol=protocol,
                    action="reference",
                    reference_gpu_lock=gpu_lock,
                )
                result["attempt"] = attempt + 1
                if result.get("status") != "oom":
                    break
            result["shapes"] = [
                _shape_payload(shape, self.options.fc) for shape in checks
            ]
            return candidate, result

        with ThreadPoolExecutor(max_workers=max(1, self.options.worker_count)) as pool:
            futures = [pool.submit(invoke, candidate) for candidate in pending]
            for future in futures:
                candidate, result = future.result()
                results.append(self._record(writer, candidate, result, phase=phase))
        return results

    def _sanity_check_incumbents(
        self, writer: JSONLWriter, protocol: TimingProtocol
    ) -> None:
        """Validate the reference path against the best incumbent per tile group."""

        candidates, _, timings = self._timing_inputs(phase="search")
        chosen: list[Candidate] = []
        for tile_n in sorted({candidate.tile_n for candidate in candidates}):
            tile_candidates = [
                candidate
                for candidate in candidates
                if candidate.tile_n == tile_n
                and candidate.candidate_id in self.incumbent_ids
            ]
            if not tile_candidates:
                continue
            best = min(
                tile_candidates,
                key=lambda candidate: min(
                    (
                        value.time
                        for value in timings.get(candidate.candidate_id, {}).values()
                        if value.time is not None
                    ),
                    default=math.inf,
                ),
            )
            chosen.append(self.candidates[best.candidate_id])
        if not chosen:
            raise RuntimeError("reference sanity check found no measured incumbents")
        records = self._reference_batch(
            writer,
            chosen,
            protocol=protocol,
            phase="reference_sanity",
        )
        attempted = [record for record in records if record.get("status") != "oom"]
        if not attempted:
            raise RuntimeError(
                "reference sanity check was inconclusive: every incumbent tile "
                "group exhausted GPU memory"
            )
        if not any(record.get("status") == "ok" for record in attempted):
            raise RuntimeError(
                "reference sanity check failed for every incumbent tile group; "
                "fix the reference path before sweeping"
            )

    def _best_by_shape(
        self, timings: Mapping[str, Mapping[str, TimingValue]]
    ) -> dict[str, TimingValue]:
        result: dict[str, TimingValue] = {}
        for row in timings.values():
            for key, value in row.items():
                if value.time is None:
                    continue
                if key not in result or value.time < result[key].time:
                    result[key] = value
        return result

    def _search_elapsed_minutes(self) -> float:
        timestamps = [
            float(record["timestamp"])
            for record in self.records
            if record.get("phase") == "search"
            and isinstance(record.get("timestamp"), (int, float))
        ]
        if not timestamps:
            return 0.0
        return max(0.0, (time.time() - min(timestamps)) / 60.0)

    def _adaptive(self, writer: JSONLWriter, protocol: TimingProtocol) -> None:
        generated = 0
        round_index = 0
        empty_rounds = 0
        self._start_search_profile()
        self._progress("adaptive: starting anytime candidate search", force=True)
        allocator = GroupBudgetAllocator(
            minimum_share=min(
                0.05, 1.0 / max(1, len({c.tile_n for c in self.candidates.values()}))
            )
        )
        while time.monotonic() < self.search_deadline:
            self._report_search_profile()
            phase_started = time.monotonic()
            _, _, timings_before_harvest = self._timing_inputs(phase="search")
            self._record_search_phase("timing_inputs", phase_started)
            phase_started = time.monotonic()
            carried_records = self._harvest_search_inflight(writer)
            self._process_adaptive_records(
                carried_records,
                timings_before_harvest,
            )
            self._record_search_phase("harvest_wait", phase_started)
            if len(self._inflight) >= self.options.worker_count:
                phase_started = time.monotonic()
                _, _, timings_before_wait = self._timing_inputs(phase="search")
                self._record_search_phase("timing_inputs", phase_started)
                phase_started = time.monotonic()
                waited_records = self._harvest_search_inflight(
                    writer,
                    wait_for_one=True,
                )
                self._process_adaptive_records(waited_records, timings_before_wait)
                self._record_search_phase("harvest_wait", phase_started)
            phase_started = time.monotonic()
            self._confirm_pending(
                writer,
                protocol,
                deadline=self.search_deadline,
            )
            self._record_search_phase("confirm_pending", phase_started)
            phase_started = time.monotonic()
            portfolio_candidates, portfolio_shapes, timings = self._timing_inputs(
                phase="search"
            )
            self._record_search_phase("timing_inputs", phase_started)
            if not portfolio_candidates:
                self._search_stop_reason = "exhausted"
                break
            round_index += 1
            self._progress(
                f"adaptive round {round_index}: proposing up to "
                f"{self.options.worker_count} candidate(s); "
                f"generated={generated}, evaluated={self.search.completed}",
                force=True,
            )
            phase_started = time.monotonic()
            interim = self._select_menu(portfolio_candidates, portfolio_shapes, timings)
            self._record_search_phase("select_menu", phase_started)
            best = self._best_by_shape(timings)
            groups = sorted(
                {candidate.tile_n for candidate in self.candidates.values()}
            )
            phase_started = time.monotonic()
            competition = contested_group_weights(
                portfolio_candidates,
                portfolio_shapes,
                timings,
                eligible={
                    str(tile_n): {
                        shape_id(shape, self.options.fc)
                        for shape in self.shapes
                        if tile_n in shape.selected_tile_ns
                    }
                    for tile_n in groups
                },
            )
            self._record_search_phase("contested_group_weights", phase_started)
            weights = {
                str(tile_n): competition.weights.get(str(tile_n), 0.0)
                for tile_n in groups
            }
            if not any(weight > 0 for weight in weights.values()):
                self._search_stop_reason = "exhausted"
                break
            shares = allocator.shares(weights, self.search.groups)
            round_stats: dict[str, dict[str, int]] = {}
            disabled_groups: set[str] = set()

            def note_group_event(
                group: str, event: str, reason: str | None = None
            ) -> None:
                persistent = self.search.group_dispatch_stats.setdefault(
                    group,
                    {"draws": 0, "proposals": 0, "drops_by_reason": {}},
                )
                current = round_stats.setdefault(
                    group,
                    {"draws": 0, "proposals": 0, "drops": 0},
                )
                if event == "draw":
                    persistent["draws"] = int(persistent.get("draws", 0)) + 1
                    current["draws"] += 1
                elif event == "proposal":
                    persistent["proposals"] = int(persistent.get("proposals", 0)) + 1
                    current["proposals"] += 1
                elif event == "drop":
                    drops = persistent.setdefault("drops_by_reason", {})
                    assert isinstance(drops, dict)
                    key = reason or "unknown"
                    drops[key] = int(drops.get(key, 0)) + 1
                    current["drops"] += 1
                    if current["drops"] >= 3 and current["proposals"] == 0:
                        disabled_groups.add(group)

            phase_started = time.monotonic()
            existing_keys = {
                _resolved_signature(
                    record.get("materialized", {}).get("resolved_hashes", {})
                )
                for record in self.records
                if record.get("phase") == "search"
                and isinstance(record.get("materialized"), Mapping)
            }
            existing_keys.update(self._adaptive_resolved_signatures.values())
            self._record_search_phase("existing_keys", phase_started)
            proposed: list[Candidate] = []
            proposal_shapes: dict[str, list[MoeSweepShape]] = {}
            max_batch = max(0, self.options.worker_count - len(self._inflight))
            proposal_deadline = min(
                self.search_deadline,
                time.monotonic() + max(1.0, 0.25 * max_batch),
            )
            phase_started = time.monotonic()
            while len(proposed) < max_batch and time.monotonic() < proposal_deadline:
                active_shares = {
                    key: share
                    for key, share in shares.items()
                    if key not in disabled_groups and share > 0
                }
                share_total = sum(active_shares.values())
                if share_total <= 0:
                    break
                draw = self.rng.random()
                cumulative = 0.0
                group = next(reversed(active_shares))
                for key, share in active_shares.items():
                    cumulative += share / share_total
                    if draw <= cumulative:
                        group = key
                        break
                note_group_event(group, "draw")
                tile_n = int(group)
                group_ids = [
                    candidate.candidate_id
                    for candidate in portfolio_candidates
                    if int(candidate.tile_n) == tile_n
                ]
                parent_pool = select_parent_pool(
                    group_ids,
                    portfolio_shapes,
                    timings,
                    best,
                    menu=interim.selected,
                    contributions=interim.marginal_contributions,
                    max_parents=32,
                    near_best_fraction=self.thresholds.success_fraction,
                )

                def mean_regret(candidate_id: str) -> float:
                    ratios = []
                    for shape_key, value in timings.get(candidate_id, {}).items():
                        baseline = best.get(shape_key)
                        if (
                            value.time is not None
                            and baseline is not None
                            and baseline.time is not None
                            and baseline.time > 0
                        ):
                            ratios.append(math.log(value.time / baseline.time))
                    return sum(ratios) / len(ratios) if ratios else math.inf

                ranked_group_ids = sorted(group_ids, key=mean_regret)
                parent_ids = list(
                    dict.fromkeys((*parent_pool.parents, *ranked_group_ids[:8]))
                )
                base_moves = applicable_move_kinds(
                    self.context,
                    (value for value in _ADAPTIVE_MOVES if value != "crossover"),
                )
                if not parent_ids:
                    note_group_event(group, "drop", "no_parent")
                    generated += 1
                    continue
                band_parents = [
                    candidate_id
                    for candidate_id, remaining in self.search.two_step_remaining.items()
                    if remaining > 0
                    and candidate_id in self.candidates
                    and self.candidates[candidate_id].tile_n == tile_n
                    and not self._parent_moves.parent_is_exhausted(
                        candidate_id, (*base_moves, "crossover")
                    )
                ]
                remaining_parents = list(dict.fromkeys((*band_parents, *parent_ids)))
                if len(remaining_parents) > len(band_parents):
                    tail = remaining_parents[len(band_parents) :]
                    self.rng.shuffle(tail)
                    remaining_parents[len(band_parents) :] = tail
                selected = None
                for parent_id in remaining_parents:
                    parent = self.candidates[parent_id]
                    has_crossover_alternative = any(
                        value != parent_id
                        and structural_options_key(self.candidates[value])
                        != structural_options_key(parent)
                        for value in parent_ids
                    )
                    if not _parent_has_unexhausted_move(
                        self._parent_moves,
                        parent_id,
                        base_moves,
                        has_crossover_alternative=has_crossover_alternative,
                    ):
                        continue
                    available_moves = self._parent_moves.available_moves(
                        parent_id, base_moves
                    )
                    parent_moves = applicable_move_kinds(
                        self.context,
                        available_moves,
                        parent=parent,
                    )
                    for unsupported_move in (
                        move for move in available_moves if move not in parent_moves
                    ):
                        by_move = self._rejection_reasons.setdefault(
                            group, {}
                        ).setdefault(unsupported_move, {})
                        by_move["unsupported"] = by_move.get("unsupported", 0) + 1
                        self._parent_moves.mark_exhausted(parent_id, unsupported_move)
                    legality = filter_legal_move_kinds(
                        parent,
                        parent_moves,
                        stage_limits=self.options.stage_limits,
                        max_sf_ring_ratio=self.options.max_sf_ring_ratio,
                        context=self.context,
                        seed=generated,
                        enable_cluster_y=self.options.enable_cluster_y,
                    )
                    for rejected_move, reason in legality.rejected.items():
                        self.search.selector.record_rejected(
                            group, rejected_move, reason
                        )
                        by_move = self._rejection_reasons.setdefault(
                            group, {}
                        ).setdefault(rejected_move, {})
                        by_move[reason] = by_move.get(reason, 0) + 1
                        self._parent_moves.mark_exhausted(parent_id, rejected_move)
                    available_moves = legality.legal
                    alternatives = [
                        value
                        for value in parent_ids
                        if value != parent_id
                        and structural_options_key(self.candidates[value])
                        != structural_options_key(parent)
                    ]
                    if alternatives and not self._parent_moves.is_exhausted(
                        parent_id, "crossover"
                    ):
                        available_moves = (*available_moves, "crossover")
                    if not available_moves:
                        continue
                    if self.search.two_step_remaining.get(parent_id, 0) > 0:
                        numeric_moves = tuple(
                            move for move in _NUMERIC_MOVES if move in available_moves
                        )
                        fallback_moves = tuple(
                            move for move in available_moves if move != "crossover"
                        )
                        if not numeric_moves and not fallback_moves:
                            continue
                        remaining = self.search.two_step_remaining[parent_id] - 1
                        if remaining > 0:
                            self.search.two_step_remaining[parent_id] = remaining
                        else:
                            self.search.two_step_remaining.pop(parent_id, None)
                        move = self.search.selector.choose(
                            group, numeric_moves or fallback_moves
                        )
                        other_parent = None
                    else:
                        local_moves = tuple(
                            value for value in available_moves if value != "crossover"
                        )
                        if (
                            alternatives
                            and "crossover" in available_moves
                            and self.rng.random() < 0.10
                        ):
                            move = "crossover"
                            other_parent = self.candidates[
                                self.rng.choice(alternatives)
                            ]
                        elif local_moves:
                            move = self.search.selector.choose(group, local_moves)
                            other_parent = None
                        else:
                            continue
                    selected = (
                        parent_id,
                        parent,
                        move,
                        other_parent,
                        legality.generated.get(move),
                    )
                    break
                if selected is None:
                    note_group_event(group, "drop", "no_move")
                    generated += 1
                    continue
                parent_id, parent, move, other_parent, generated_child = selected
                try:
                    problem_ks = _problem_ks(self.shapes, self.options.fc)
                    parent_regime = self._smem_regime(parent, problem_ks)
                    if move in STAGE_MOVE_KINDS:
                        unseen_neighbors = [
                            candidate
                            for candidate in enumerate_stage_neighbors(
                                parent,
                                move,
                                stage_limits=self.options.stage_limits,
                                max_sf_ring_ratio=self.options.max_sf_ring_ratio,
                                context=self.context,
                            )
                            if candidate.candidate_id not in self.candidates
                        ]
                        if not unseen_neighbors:
                            self._parent_moves.mark_exhausted(parent_id, move)
                            raise ValueError("no unseen legal stage neighbor")
                        child = self.rng.choice(unseen_neighbors)
                    elif generated_child is not None:
                        child = generated_child
                    else:
                        child = generate_move(
                            parent,
                            move,
                            self.rng,
                            other_parent=other_parent,
                            stage_limits=self.options.stage_limits,
                            max_sf_ring_ratio=self.options.max_sf_ring_ratio,
                            repair_index=generated % 3,
                            context=self.context,
                            enable_cluster_y=self.options.enable_cluster_y,
                        )
                    child = repair_structural_candidate(
                        child,
                        self.context,
                        _problem_ks(self.shapes, self.options.fc),
                        repair_index=generated % 3,
                        stage_limits=self.options.stage_limits,
                        max_sf_ring_ratio=self.options.max_sf_ring_ratio,
                        target_smem_regime_kib=parent_regime,
                    )
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    self.search.selector.record_rejected(group, move, reason)
                    by_move = self._rejection_reasons.setdefault(group, {}).setdefault(
                        move, {}
                    )
                    by_move[reason] = by_move.get(reason, 0) + 1
                    if any(
                        marker in reason.lower()
                        for marker in (
                            "num_registers",
                            "register allocation",
                            "every trial illegal",
                            "no legal neighbor",
                            "no unseen legal",
                        )
                    ):
                        self._parent_moves.mark_exhausted(parent_id, move)
                    generated += 1
                    continue
                if child.candidate_id in self.candidates:
                    self.search.selector.record_duplicate(group, move)
                    self._parent_moves.record_duplicate(parent_id, move)
                    by_move = self._rejection_reasons.setdefault(group, {}).setdefault(
                        move, {}
                    )
                    by_move["duplicate_candidate"] = (
                        by_move.get("duplicate_candidate", 0) + 1
                    )
                    generated += 1
                    continue
                try:
                    materialized = materialize_candidate(
                        child,
                        self.context,
                        _problem_ks(self.shapes, self.options.fc),
                        check_task_manager=True,
                    )
                    applicable_shapes: list[MoeSweepShape] = []
                    applicability_errors: list[str] = []
                    for shape in self.shapes:
                        if child.tile_n not in shape.selected_tile_ns:
                            continue
                        applicable, reason = is_applicable(
                            materialized, self.context, shape
                        )
                        if applicable:
                            applicable_shapes.append(shape)
                        else:
                            applicability_errors.append(reason)
                    if not applicable_shapes:
                        reason = next(
                            (
                                value
                                for value in applicability_errors
                                if value.startswith("cluster_span:")
                            ),
                            "no_live_shapes",
                        )
                        self.candidates[child.candidate_id] = child
                        self._record(
                            writer,
                            child,
                            {"status": "validation_error", "error": reason},
                            phase="search",
                        )
                        self.search.selector.record_rejected(group, move, reason)
                        by_move = self._rejection_reasons.setdefault(
                            group, {}
                        ).setdefault(move, {})
                        by_move[reason] = by_move.get(reason, 0) + 1
                        note_group_event(group, "drop", "no_live_shapes")
                        generated += 1
                        continue
                    resolved_key = _resolved_signature(materialized.resolved_hashes)
                    if resolved_key in existing_keys:
                        self.search.selector.record_duplicate(group, move)
                        self._parent_moves.record_duplicate(parent_id, move)
                        by_move = self._rejection_reasons.setdefault(
                            group, {}
                        ).setdefault(move, {})
                        by_move["duplicate_signature"] = (
                            by_move.get("duplicate_signature", 0) + 1
                        )
                        generated += 1
                        continue
                except Exception as exc:
                    self.candidates[child.candidate_id] = child
                    self._record(
                        writer,
                        child,
                        {
                            "status": "validation_error",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        phase="search",
                    )
                    reason = f"{type(exc).__name__}: {exc}"
                    self.search.selector.record_rejected(group, move, reason)
                    by_move = self._rejection_reasons.setdefault(group, {}).setdefault(
                        move, {}
                    )
                    by_move[reason] = by_move.get(reason, 0) + 1
                    if any(
                        marker in reason.lower()
                        for marker in (
                            "num_registers",
                            "register allocation",
                            "every trial illegal",
                            "no legal neighbor",
                        )
                    ):
                        self._parent_moves.mark_exhausted(parent_id, move)
                    generated += 1
                    continue
                existing_keys.add(resolved_key)
                self.candidates[child.candidate_id] = child
                proposed.append(child)
                proposal_shapes[child.candidate_id] = applicable_shapes
                self._adaptive_proposal_metadata[child.candidate_id] = (
                    group,
                    move,
                    parent_id,
                )
                self._adaptive_reference_best[child.candidate_id] = dict(best)
                self._adaptive_resolved_signatures[child.candidate_id] = resolved_key
                note_group_event(group, "proposal")
                generated += 1
            self._record_search_phase("proposing", phase_started)
            for group, share in shares.items():
                current = round_stats.get(group, {})
                if (
                    share >= 0.2
                    and int(current.get("draws", 0)) >= 3
                    and int(current.get("proposals", 0)) == 0
                ):
                    persistent = self.search.group_dispatch_stats.setdefault(
                        group,
                        {"draws": 0, "proposals": 0, "drops_by_reason": {}},
                    )
                    persistent["starvation_warnings"] = (
                        int(persistent.get("starvation_warnings", 0)) + 1
                    )
                    self._progress(
                        f"adaptive: tile N {group} received {share:.1%} of "
                        f"round {round_index} draws but produced no proposals",
                        force=True,
                    )
            if not proposed:
                if self._inflight:
                    phase_started = time.monotonic()
                    _, _, timings_before_wait = self._timing_inputs(phase="search")
                    self._record_search_phase("timing_inputs", phase_started)
                    phase_started = time.monotonic()
                    waited_records = self._harvest_search_inflight(
                        writer, wait_for_one=True
                    )
                    self._process_adaptive_records(waited_records, timings_before_wait)
                    self._record_search_phase("harvest_wait", phase_started)
                    empty_rounds = 0
                    continue
                empty_rounds += 1
                if empty_rounds >= 3:
                    self._search_stop_reason = "no_proposals"
                    break
                continue
            empty_rounds = 0
            phase_started = time.monotonic()
            records = self._evaluate_batch(
                writer,
                proposed,
                protocol=protocol,
                phase="search",
                progress_label=f"adaptive round {round_index}",
                shapes_by_candidate=proposal_shapes,
                allow_stragglers=True,
            )
            self._process_adaptive_records(records, timings)
            self._record_search_phase("evaluate_batch", phase_started)
            phase_started = time.monotonic()
            updated_candidates, updated_shapes, updated_timings = self._timing_inputs(
                phase="search"
            )
            self._record_search_phase("timing_inputs", phase_started)
            phase_started = time.monotonic()
            interim_after = self._select_menu(
                updated_candidates, updated_shapes, updated_timings
            )
            self._record_search_phase("select_menu", phase_started)
            interim_candidates = [
                self.candidates[str(candidate_id)]
                for candidate_id in interim_after.selected
                if str(candidate_id) in self.candidates
            ]
            phase_started = time.monotonic()
            self._reference_batch(
                writer,
                interim_candidates,
                protocol=protocol,
                phase="reference",
                gpu_lock=True,
            )
            self._record_search_phase("reference_batch", phase_started)
            phase_started = time.monotonic()
            self._checkpoint(phase="search")
            self._record_search_phase("checkpoint", phase_started)
            phase_started = time.monotonic()
            self._report_speedup()
            self._record_search_phase("report_speedup", phase_started)
        phase_started = time.monotonic()
        _, _, timings_before_drain = self._timing_inputs(phase="search")
        self._record_search_phase("timing_inputs", phase_started)
        phase_started = time.monotonic()
        drained_records = self._harvest_search_inflight(
            writer,
            wait_for_all=True,
        )
        self._process_adaptive_records(drained_records, timings_before_drain)
        self._record_search_phase("harvest_wait", phase_started)
        confirmation_deadline = max(
            self.search_deadline,
            self.deadline - self.options.reference_reserve_minutes * 60.0,
        )
        phase_started = time.monotonic()
        self._confirm_pending(
            writer,
            protocol,
            deadline=confirmation_deadline,
        )
        self._record_search_phase("confirm_pending", phase_started)
        phase_started = time.monotonic()
        self._checkpoint(phase="search")
        self._record_search_phase("checkpoint", phase_started)
        self._stop_search_profile()
        self._report_search_profile(force=True)
        if self._search_stop_minute is None:
            self._search_stop_minute = self._search_elapsed_minutes()

    def _seed(self, writer: JSONLWriter, protocol: TimingProtocol) -> None:
        self._progress("seed: generating fixed incumbent variants", force=True)
        portfolio_candidates, _, timings = self._timing_inputs(phase="search")
        if not portfolio_candidates:
            return
        scores = {
            candidate.candidate_id: min(
                (
                    value.time
                    for value in timings.get(candidate.candidate_id, {}).values()
                    if value.time is not None
                ),
                default=math.inf,
            )
            for candidate in portfolio_candidates
        }
        problem_ks = _problem_ks(self.shapes, self.options.fc)
        if self._seed_selection_complete:
            seeds = list(self._seed_candidates)
        else:
            raw_seeds: list[Candidate] = []
            for tile_n in sorted(
                {candidate.tile_n for candidate in self.candidates.values()}
            ):
                ranked = sorted(
                    (
                        candidate
                        for candidate_id, candidate in self.candidates.items()
                        if candidate_id in self.incumbent_ids
                        and candidate.tile_n == tile_n
                    ),
                    key=lambda candidate: scores.get(candidate.candidate_id, math.inf),
                )
                raw_seeds.extend(
                    fixed_seed_variants(
                        ranked,
                        limit=self.options.seed_limit,
                        sm_arch=self.options.sm_arch,
                    )
                )
            seeds = []
            for candidate in raw_seeds:
                parent_candidate = (
                    self.candidates.get(candidate.parent_ids[0])
                    if candidate.parent_ids
                    else None
                )
                parent_regime = (
                    None
                    if parent_candidate is None
                    else self._smem_regime(parent_candidate, problem_ks)
                )
                for repair_index in range(3):
                    try:
                        seeds.append(
                            repair_structural_candidate(
                                candidate,
                                self.context,
                                problem_ks,
                                repair_index=repair_index,
                                stage_limits=self.options.stage_limits,
                                max_sf_ring_ratio=self.options.max_sf_ring_ratio,
                                target_smem_regime_kib=parent_regime,
                            )
                        )
                    except Exception:
                        continue
            seeds = list(
                {candidate.candidate_id: candidate for candidate in seeds}.values()
            )
            self._seed_candidates = seeds
            self._seed_selection_complete = True
            # Persist the exact seed set before the first dispatch so a crash
            # cannot reselect seeds from a partially updated timing journal.
            self._checkpoint(phase="seed_selection")
        signature_owner: dict[tuple[str, ...], str] = {}
        for record in self.records:
            materialized_record = record.get("materialized", {})
            if not isinstance(materialized_record, Mapping):
                continue
            hashes = materialized_record.get("resolved_hashes", {})
            if isinstance(hashes, Mapping) and hashes:
                signature_owner[_resolved_signature(hashes)] = str(
                    record["candidate_id"]
                )
        pending: list[Candidate] = []
        for candidate in seeds:
            if candidate.candidate_id in self.candidates:
                continue
            self.candidates[candidate.candidate_id] = candidate
            try:
                materialized = materialize_candidate(
                    candidate,
                    self.context,
                    problem_ks,
                    check_task_manager=True,
                )
            except Exception as exc:
                self._record(
                    writer,
                    candidate,
                    {
                        "status": "validation_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    phase="search",
                )
                continue
            signature = _resolved_signature(materialized.resolved_hashes)
            owner = signature_owner.get(signature)
            if owner is not None:
                self._record(
                    writer,
                    candidate,
                    {
                        "status": "duplicate_signature",
                        "duplicate_of": owner,
                        "materialized": {
                            "resolved_hashes": {
                                str(key): value
                                for key, value in materialized.resolved_hashes.items()
                            }
                        },
                    },
                    phase="search",
                )
                continue
            signature_owner[signature] = candidate.candidate_id
            pending.append(candidate)
        self._progress(
            f"seed: {len(pending)} unique candidate(s) ready to evaluate",
            force=True,
        )
        for batch_label, batch in self._batches(pending):
            if time.monotonic() >= self.search_deadline:
                break
            self._evaluate_batch(
                writer,
                batch,
                protocol=protocol,
                phase="search",
                progress_label=f"seed {batch_label}",
            )
            self._checkpoint(phase="seed")
            self._report_speedup()

    def _finalize(self, writer: JSONLWriter, search_protocol: TimingProtocol) -> None:
        self._progress("finalize: building the re-timing pool", force=True)
        # Drop allocator reservations and process-local module state accumulated
        # during search. On-disk CuTe-DSL caches remain intact.
        self.close()
        candidates, shapes, timings = self._timing_inputs(phase="search")
        if not candidates:
            raise RuntimeError("sweep produced no successful measurements")
        interim = self._select_menu(candidates, shapes, timings)
        pool = build_final_retime_pool(
            candidates,
            shapes,
            timings,
            top_k=5,
            interim_menu=interim.selected,
            contributors=interim.marginal_contributions,
            incumbents=self.incumbent_ids,
            near_best_fraction=self.thresholds.success_fraction,
            max_near_best_per_shape=self.options.max_near_best_per_shape,
        )
        final_protocol = final_timing_protocol(search_protocol)
        sanity_seconds = [
            float(check.get("elapsed_seconds") or 0.0)
            for record in self.records
            if record.get("phase") == "reference_sanity"
            for check in record.get("checks", ())
            if check.get("passed") is not None
        ]
        tile_group_count = len({candidate.tile_n for candidate in candidates})
        checks_per_candidate = max(
            (
                len(
                    {
                        int(getattr(shape, f"{self.options.fc}_k"))
                        for shape in self.final_shapes
                        if candidate.tile_n in shape.selected_tile_ns
                    }
                )
                for candidate in candidates
            ),
            default=1,
        )
        measured_reserve_seconds = (
            statistics.median(sanity_seconds)
            * self.options.menu_size
            * max(1, tile_group_count)
            * checks_per_candidate
            * 1.25
            if sanity_seconds
            else self.options.reference_reserve_minutes * 60.0
        )
        reference_reserve_seconds = min(
            max(0.0, measured_reserve_seconds),
            max(0.0, self.options.tail_minutes * 60.0),
        )
        retime_deadline = self.deadline - reference_reserve_seconds
        prior_retimed, _, _ = self._timing_inputs(phase="retime")
        completed = {candidate.candidate_id for candidate in prior_retimed}
        retime_candidates = [
            self.candidates[candidate_id]
            for candidate_id in pool.candidates
            if candidate_id not in completed
        ]
        self._progress(
            f"finalize: re-timing {len(retime_candidates)} candidate(s) "
            f"from a pool of {len(pool.candidates)}",
            force=True,
        )
        for batch_label, batch in self._batches(retime_candidates):
            if time.monotonic() >= retime_deadline:
                break
            final_shapes_by_candidate = {
                candidate.candidate_id: [
                    shape
                    for shape in self.final_shapes
                    if candidate.tile_n in shape.selected_tile_ns
                ]
                for candidate in batch
            }
            self._evaluate_batch(
                writer,
                batch,
                protocol=final_protocol,
                phase="retime",
                progress_label=f"retime {batch_label}",
                shapes_by_candidate=final_shapes_by_candidate,
                deadline=retime_deadline,
            )
        latest_retime = {
            str(record.get("candidate_id")): record
            for record in self.records
            if record.get("phase") == "retime"
        }
        oom_retries = [
            self.candidates[candidate_id]
            for candidate_id in pool.candidates
            if candidate_id in latest_retime
            and _is_cuda_oom(latest_retime[candidate_id])
        ]
        if oom_retries and time.monotonic() < retime_deadline:
            self._progress(
                f"finalize: retrying {len(oom_retries)} CUDA OOM candidate(s)",
                force=True,
            )
        for batch_label, batch in self._batches(oom_retries):
            if time.monotonic() >= retime_deadline:
                break
            retry_shapes_by_candidate = {
                candidate.candidate_id: [
                    shape
                    for shape in self.final_shapes
                    if candidate.tile_n in shape.selected_tile_ns
                ]
                for candidate in batch
            }
            self._evaluate_batch(
                writer,
                batch,
                protocol=final_protocol,
                phase="retime",
                progress_label=f"retime OOM retry {batch_label}",
                shapes_by_candidate=retry_shapes_by_candidate,
                deadline=retime_deadline,
            )
        allowed = set(pool.candidates)
        retimed_candidates, _all_final_shapes, retimed_timings = self._timing_inputs(
            phase="retime", allowed=allowed, shapes=self.final_shapes
        )
        selection_shape_values = [
            shape
            for shape in self.final_shapes
            if shape.routing in set(self.options.selection_routings or ())
        ]
        if not selection_shape_values:
            raise RuntimeError(
                "final shape set does not include any selection routing "
                f"{self.options.selection_routings!r}"
            )
        final_shapes = selection_portfolio_shapes(
            self.final_shapes,
            fc=self.options.fc,
            routings=self.options.selection_routings or (),
        )
        successfully_retimed = {
            candidate.candidate_id for candidate in retimed_candidates
        }
        latest_retime_records = {
            str(record.get("candidate_id")): record
            for record in self.records
            if record.get("phase") == "retime"
        }
        usable_ids = set(successfully_retimed)
        candidate_by_id = {
            candidate.candidate_id: candidate for candidate in candidates
        }
        final_candidates = [
            candidate_by_id[candidate_id]
            for candidate_id in pool.candidates
            if candidate_id in usable_ids and candidate_id in candidate_by_id
        ]
        final_timings = {
            candidate_id: dict(retimed_timings.get(candidate_id, {}))
            for candidate_id in usable_ids
        }
        finalization_complete = successfully_retimed == allowed
        failed: set[str] = set()
        correctness_failures: list[dict[str, object]] = []
        verification_errors: list[dict[str, object]] = [
            {
                "candidate_id": candidate_id,
                "tile_n": self.candidates[candidate_id].tile_n,
                "move_kind": self.candidates[candidate_id].move_kind,
                "status": "retime_incomplete",
                "error": self._retime_incomplete_reason(
                    latest_retime_records.get(candidate_id)
                ),
            }
            for candidate_id in sorted(allowed - successfully_retimed)
        ]
        prior_reference_records = {
            str(record.get("candidate_id")): record
            for record in self.records
            if record.get("phase") == "reference"
        }
        reference_verified: set[str] = {
            candidate_id
            for candidate_id, record in prior_reference_records.items()
            if record.get("status") == "ok"
        }
        failed.update(
            candidate_id
            for candidate_id, record in prior_reference_records.items()
            if record.get("status") == "failed"
        )
        while True:
            selectable = [
                candidate
                for candidate in final_candidates
                if candidate.candidate_id not in failed
            ]
            selection = self._select_menu(
                selectable,
                final_shapes,
                final_timings,
            )
            pending_selection = [
                self.candidates[str(candidate_id)]
                for candidate_id in selection.selected
                if str(candidate_id) not in reference_verified
            ]
            if not pending_selection:
                break
            if time.monotonic() >= self.deadline:
                finalization_complete = False
                for candidate in pending_selection:
                    verification_errors.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "tile_n": candidate.tile_n,
                            "move_kind": candidate.move_kind,
                            "status": "deadline",
                            "error": (
                                "hard deadline reached before reference checks "
                                "completed"
                            ),
                        }
                    )
                break
            self._progress(
                f"verify: checking {len(pending_selection)} menu contender(s) "
                "in parallel",
                force=True,
            )
            batch_results = self._reference_batch(
                writer,
                pending_selection,
                protocol=final_protocol,
                phase="reference",
                gpu_lock=True,
            )
            batch_by_id = {
                str(record.get("candidate_id")): record
                for record in batch_results
                if str(record.get("candidate_id"))
                in {candidate.candidate_id for candidate in pending_selection}
            }
            removed = False
            for candidate in pending_selection:
                result = batch_by_id[candidate.candidate_id]
                if result.get("status") == "ok":
                    reference_verified.add(candidate.candidate_id)
                    continue
                removed = True
                failed.add(candidate.candidate_id)
                if result.get("status") == "failed":
                    failure = {
                        "candidate_id": candidate.candidate_id,
                        "tile_n": candidate.tile_n,
                        "move_kind": candidate.move_kind,
                        "candidate": candidate.as_json_row(),
                        "result": result,
                    }
                    correctness_failures.append(failure)
                else:
                    finalization_complete = False
                    verification_errors.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "tile_n": candidate.tile_n,
                            "move_kind": candidate.move_kind,
                            "result": result,
                        }
                    )
            if removed:
                continue
            break
        final_candidates, final_timings = _prune_failed_finalists(
            final_candidates, final_timings, failed
        )
        selected_by_tile: dict[int, list[Mapping[str, object]]] = {}
        for candidate_id in selection.selected:
            candidate = self.candidates[str(candidate_id)]
            selected_by_tile.setdefault(candidate.tile_n, []).append(
                candidate.as_json_row()
            )
        per_shape = {}
        report_timings = {
            **timings,
            **final_timings,
        }

        def logical_shape_key(shape: MoeSweepShape) -> tuple[object, ...]:
            return (
                shape.timing_num_tokens,
                shape.local_num_experts,
                shape.top_k,
                getattr(shape, f"{self.options.fc}_n"),
                getattr(shape, f"{self.options.fc}_k"),
            )

        def estimated_active_experts(shape: MoeSweepShape) -> float | None:
            expert_count = shape.local_num_experts
            token_count = shape.timing_num_tokens
            local_experts = shape.local_num_experts
            if shape.routing == "balanced_round_robin":
                return float(min(local_experts, token_count * shape.top_k))
            if shape.routing == "uniform":
                inactive = (1.0 - shape.top_k / expert_count) ** token_count
                return local_experts * (1.0 - inactive)
            if shape.routing.startswith("zipf:"):
                try:
                    alpha = float(shape.routing.split(":", maxsplit=1)[1])
                except ValueError:
                    return None
                weights = [1.0 / (index + 1) ** alpha for index in range(expert_count)]
                total_weight = sum(weights)
                return sum(
                    1.0
                    - (1.0 - min(1.0, shape.top_k * weight / total_weight))
                    ** token_count
                    for weight in weights[:local_experts]
                )
            return None

        def expert_weight_bytes(shape: MoeSweepShape) -> float | None:
            dtype = shape.dtype.lower()
            bytes_per_element = (
                0.5
                if "fp4" in dtype
                else 1.0
                if "fp8" in dtype
                else 2.0
                if "bf16" in dtype
                else None
            )
            if bytes_per_element is None:
                return None
            return (
                int(getattr(shape, f"{self.options.fc}_n"))
                * int(getattr(shape, f"{self.options.fc}_k"))
                * bytes_per_element
            )

        final_shapes_by_logical: dict[tuple[object, ...], list[MoeSweepShape]] = {}
        for shape in self.final_shapes:
            final_shapes_by_logical.setdefault(logical_shape_key(shape), []).append(
                shape
            )
        report_shape_values = [
            shape
            for shape in self.final_shapes
            if shape.routing == self.options.search_routing
        ]
        for selected_shape in report_shape_values:
            shape_id_value = shape_id(selected_shape, self.options.fc)
            # Selection omits shapes no finalist measured, e.g. when re-timing
            # ran out of budget; report those as uncovered.
            regret = selection.per_shape.get(shape_id_value)
            best_time = None if regret is None else regret.best_time
            selected_time = None if regret is None else regret.selected_time
            regret_value = math.inf if regret is None else regret.regret
            incumbent_times = [
                value.time
                for candidate_id, row in report_timings.items()
                if candidate_id in self.incumbent_ids
                and (value := row.get(shape_id_value)) is not None
                and value.time is not None
            ]
            route_results: dict[str, dict[str, object]] = {}
            for route_shape in final_shapes_by_logical.get(
                logical_shape_key(selected_shape), ()
            ):
                route_key = shape_id(route_shape, self.options.fc)
                route_all = [
                    value.time
                    for row in final_timings.values()
                    if (value := row.get(route_key)) is not None
                    and value.time is not None
                ]
                route_incumbent = [
                    value.time
                    for candidate_id, row in final_timings.items()
                    if candidate_id in self.incumbent_ids
                    and (value := row.get(route_key)) is not None
                    and value.time is not None
                ]
                route_selected = [
                    value.time
                    for candidate_id in selection.selected
                    if (
                        value := final_timings.get(str(candidate_id), {}).get(route_key)
                    )
                    is not None
                    and value.time is not None
                ]
                best_route = min(route_all) if route_all else None
                route_active_experts = estimated_active_experts(route_shape)
                route_expert_weight_bytes = expert_weight_bytes(route_shape)
                route_results[route_shape.routing] = {
                    "best_found_us": best_route,
                    "best_existing_us": (
                        min(route_incumbent) if route_incumbent else None
                    ),
                    "selected_us": (min(route_selected) if route_selected else None),
                    "speedup_vs_pre_sweep": (
                        min(route_incumbent) / best_route
                        if route_incumbent and best_route is not None and best_route > 0
                        else None
                    ),
                    "menu_speedup_vs_pre_sweep": (
                        min(route_incumbent) / min(route_selected)
                        if route_incumbent and route_selected
                        else None
                    ),
                    "active_experts": route_active_experts,
                    "expert_weight_bytes": route_expert_weight_bytes,
                    "estimated_weight_bandwidth_tb_s": (
                        route_active_experts
                        * route_expert_weight_bytes
                        / best_route
                        / 1e6
                        if route_active_experts is not None
                        and route_expert_weight_bytes is not None
                        and best_route is not None
                        and best_route > 0
                        else None
                    ),
                }
            active_experts = estimated_active_experts(selected_shape)
            weight_bytes_per_expert = expert_weight_bytes(selected_shape)
            best_found = _best_final_time(
                final_timings,
                timings,
                str(shape_id_value),
                best_time,
            )
            per_shape[str(shape_id_value)] = {
                "shape_label": self._shape_labels.get(
                    str(shape_id_value), str(shape_id_value)
                ),
                "best_existing_us": (min(incumbent_times) if incumbent_times else None),
                "best_found_us": best_found,
                "selected_us": selected_time,
                "menu_speedup_vs_incumbent": (
                    min(incumbent_times) / selected_time
                    if incumbent_times and selected_time is not None
                    else None
                ),
                "regret": regret_value,
                "routing": selected_shape.routing,
                "active_experts": active_experts,
                "expert_weight_bytes": weight_bytes_per_expert,
                "estimated_weight_bandwidth_tb_s": (
                    active_experts * weight_bytes_per_expert / best_found / 1e6
                    if active_experts is not None
                    and weight_bytes_per_expert is not None
                    and best_found is not None
                    and best_found > 0
                    else None
                ),
                "per_routing": route_results,
            }
        move_stats = {
            group: {move: arm.to_dict() for move, arm in sorted(group_arms.items())}
            for group, group_arms in sorted(self.search.selector.arms.items())
        }
        covered = bool(selection.selected) and all(
            (value := selection.per_shape.get(shape.shape_id)) is not None
            and math.isfinite(value.regret)
            for shape in final_shapes
        )
        routing_robustness: dict[str, dict[str, float | int | None]] = {}
        routing_robust = True
        for routing in self.options.retime_routings:
            route_shapes = [
                shape for shape in self.final_shapes if shape.routing == routing
            ]
            regrets: list[float] = []
            for shape in route_shapes:
                shape_key = shape_id(shape, self.options.fc)
                all_times = [
                    value.time
                    for row in final_timings.values()
                    if (value := row.get(shape_key)) is not None
                    and value.time is not None
                ]
                selected_times = [
                    value.time
                    for candidate_id in selection.selected
                    if (
                        value := final_timings.get(str(candidate_id), {}).get(shape_key)
                    )
                    is not None
                    and value.time is not None
                ]
                if all_times and selected_times:
                    regrets.append(max(0.0, min(selected_times) / min(all_times) - 1.0))
            worst = max(regrets, default=None)
            average = sum(regrets) / len(regrets) if regrets else None
            routing_robustness[routing] = {
                "covered_shapes": len(regrets),
                "average_regret": average,
                "worst_regret": worst,
            }
            if (
                routing != self.options.search_routing
                and worst is not None
                and worst > self.options.max_routing_regret
            ):
                routing_robust = False
                verification_errors.append(
                    {
                        "status": "routing_regret",
                        "routing": routing,
                        "worst_regret": worst,
                        "limit": self.options.max_routing_regret,
                    }
                )
        verified = finalization_complete and covered and routing_robust
        environment = (
            {} if self.calibration is None else self.calibration.environment.to_dict()
        )
        environment["measurement_protocol"] = asdict(final_protocol)
        environment = refresh_environment_provenance(
            environment,
            repository=repo_root(),
        )
        summary = make_summary(
            fc=self.options.fc,
            model=self.options.model or "custom",
            sm_arch=self.options.sm_arch,
            selected_by_tile_n=selected_by_tile,
            per_shape=per_shape,
            verified=verified,
            correctness_failures=correctness_failures,
            move_statistics=move_stats,
            environment=environment,
            activation_params={
                "activation": self.shapes[0].activation,
                "bias": self.context.has_bias,
                "alpha": self.context.gemm1_alpha_value,
                "beta": self.context.gemm1_beta_value,
                "clamp": self.context.gemm1_clamp_limit_value,
            },
        )
        summary["finalization"] = {
            "complete": finalization_complete,
            "retime_pool_size": len(allowed),
            "successfully_retimed": len(successfully_retimed),
            "reference_reserve_minutes": self.options.reference_reserve_minutes,
            "measured_reference_reserve_minutes": (reference_reserve_seconds / 60.0),
            "reference_verified": len(reference_verified),
            "reference_failed": len(correctness_failures),
            "reference_errors": sum(
                record.get("status") not in ("ok", "failed")
                for record in prior_reference_records.values()
            ),
            "verified_count": sum(
                str(candidate_id) in reference_verified
                for candidate_id in selection.selected
            ),
            "unverified_count": sum(
                str(candidate_id) not in reference_verified
                for candidate_id in selection.selected
            ),
            "failed_count": len(correctness_failures),
            "verification_errors": verification_errors,
        }
        summary["search_termination"] = {
            "reason": self._search_stop_reason,
            "stop_minute": self._search_stop_minute,
        }
        summary["search_confirmation"] = {
            "unconfirmed_candidates": len(self.search.pending_confirmations),
            "unconfirmed_pairs": sum(
                len(shape_ids)
                for shape_ids in self.search.pending_confirmations.values()
            ),
        }
        summary["portfolio"] = {
            "average_regret": selection.average_regret,
            "worst_regret": selection.worst_regret,
            "weighted_log_regret": selection.weighted_log_regret,
            "gain_vs_size": menu_speedup_curve(
                [str(candidate_id) for candidate_id in selection.selected],
                final_shapes,
                report_timings,
                self.incumbent_ids,
            ),
        }
        summary["routing_robustness"] = routing_robustness
        summary["reference_consistency"] = reference_consistency(
            self.records,
            selected_ids={str(candidate_id) for candidate_id in selection.selected},
        )
        summary["search_failures"] = self._search_failure_summary()
        summary["gpu_clocks"] = summarize_gpu_clocks(self.records)
        summary["feature_effects"] = self._feature_effects()
        summary["rejection_reasons"] = self._rejection_reasons
        summary["group_dispatch_stats"] = self.search.group_dispatch_stats
        search_loop_profile = self._search_loop_profile()
        summary["search_loop_profile"] = search_loop_profile
        summary["worker_utilization"] = search_loop_profile["worker_utilization"]
        summary["lineage_contributions"] = self._lineage_contributions(
            [str(candidate_id) for candidate_id in selection.selected]
        )
        selected_ids = {str(candidate_id) for candidate_id in selection.selected}
        summary["existing_entries_never_selected"] = [
            self.candidates[candidate_id].as_json_row()
            for candidate_id in sorted(self.incumbent_ids - selected_ids)
        ]
        summary["incumbents_dropped_by_cap"] = sorted(self._incumbents_dropped_by_cap)
        write_summary(self.options.out_dir, summary)
        self._checkpoint(phase="complete")

    def _search_failure_summary(self) -> list[dict[str, object]]:
        """Aggregate non-finalization failures recorded in the run journal."""

        failing_statuses = {
            "error",
            "oom",
            "worker_error",
            "timeout",
            "validation_error",
            "reference_error",
            "driver_error",
            "failed",
        }
        grouped: dict[
            tuple[str, str, str, int | None, int | None, str],
            dict[str, object],
        ] = {}
        for record in self.records:
            status = str(record.get("status", ""))
            failure_value = str(record.get("failure") or "none")
            if status not in failing_statuses and failure_value in ("", "none"):
                continue
            error = self._record_error(record)
            is_oom = _is_cuda_oom(record)
            failure_value = "cuda_oom" if is_oom else failure_value
            if is_oom and not error:
                error = "CUDA out of memory"
            exit_match = re.search(r"exit code (-?\d+)", error)
            exit_code = int(exit_match.group(1)) if exit_match else None
            key = (
                str(record.get("phase", "unknown")),
                status,
                failure_value,
                exit_code,
                (int(record["tile_n"]) if record.get("tile_n") is not None else None),
                str(record.get("move_kind") or "unknown"),
            )
            row = grouped.setdefault(
                key,
                {
                    "phase": key[0],
                    "status": key[1],
                    "failure": key[2],
                    "exit_code": key[3],
                    "tile_n": key[4],
                    "move_kind": key[5],
                    "count": 0,
                    "example_candidate_ids": [],
                    "error": error.splitlines()[0] if error else "",
                },
            )
            row["count"] = int(row["count"]) + 1
            examples = row["example_candidate_ids"]
            if isinstance(examples, list) and len(examples) < 3:
                candidate_id = str(record.get("candidate_id") or "")
                if candidate_id and candidate_id not in examples:
                    examples.append(candidate_id)
        return [
            grouped[key]
            for key in sorted(
                grouped,
                key=lambda item: tuple(
                    "" if value is None else str(value) for value in item
                ),
            )
        ]

    @staticmethod
    def _record_error(record: Mapping[str, object]) -> str:
        error = str(record.get("error") or "")
        if error:
            return error
        for measurement in record.get("measurements", ()) or ():
            if isinstance(measurement, Mapping) and measurement.get("error"):
                return str(measurement["error"])
        return ""

    @classmethod
    def _retime_incomplete_reason(cls, record: Mapping[str, object] | None) -> str:
        if record is None:
            return "candidate was not re-timed before the deadline"
        if _is_cuda_oom(record):
            return "CUDA out of memory during final re-timing"
        error = cls._record_error(record)
        if error:
            return error
        failure = str(record.get("failure") or "")
        status = str(record.get("status") or "unknown")
        return f"final re-timing ended with {status}/{failure or 'unknown'}"

    def _feature_effects(self) -> dict[str, object]:
        """Summarize parent/child feature flips per tile-N group."""
        features: dict[str, dict[str, dict[str, object]]] = {
            "large_carveout": {},
            "multicast": {},
            "tma_store": {},
        }
        latest: dict[str, Mapping[str, object]] = {}
        for record in self.records:
            if record.get("phase") not in ("search", "retime"):
                continue
            candidate_id = str(record.get("candidate_id"))
            latest[candidate_id] = record
        paired_ratios: dict[tuple[str, int], list[float]] = {}
        for candidate_id, child_record in latest.items():
            candidate = self.candidates.get(candidate_id)
            child_materialized = child_record.get("materialized", {})
            if candidate is None or not isinstance(child_materialized, Mapping):
                continue
            child_times = {
                str(measurement["shape_id"]): float(measurement["median_us"])
                for measurement in child_record.get("measurements", ())
                if measurement.get("status") == "ok"
            }
            for parent_id in candidate.parent_ids:
                parent_record = latest.get(parent_id)
                if parent_record is None:
                    continue
                parent_candidate = self.candidates.get(parent_id)
                parent_materialized = parent_record.get("materialized", {})
                if not isinstance(parent_materialized, Mapping):
                    continue
                parent_times = {
                    str(measurement["shape_id"]): float(measurement["median_us"])
                    for measurement in parent_record.get("measurements", ())
                    if measurement.get("status") == "ok"
                }
                feature_values = {
                    "large_carveout": (
                        int(parent_materialized.get("smem_regime_kib") or 227) > 227,
                        int(child_materialized.get("smem_regime_kib") or 227) > 227,
                    ),
                    "multicast": (
                        bool(parent_materialized.get("multicast_operands")),
                        bool(child_materialized.get("multicast_operands")),
                    ),
                    "tma_store": (
                        bool(
                            parent_candidate.options.get("use_tma_store", True)
                            if parent_candidate is not None
                            else True
                        ),
                        bool(candidate.options.get("use_tma_store", True)),
                    ),
                }
                for feature, (parent_on, child_on) in feature_values.items():
                    if parent_on == child_on:
                        continue
                    for shape_key in child_times.keys() & parent_times.keys():
                        on_time = (
                            child_times[shape_key]
                            if child_on
                            else parent_times[shape_key]
                        )
                        off_time = (
                            parent_times[shape_key]
                            if child_on
                            else child_times[shape_key]
                        )
                        paired_ratios.setdefault(
                            (feature, candidate.tile_n), []
                        ).append(on_time / off_time)
        for feature in features:
            tile_ns = sorted(
                tile_n for name, tile_n in paired_ratios if name == feature
            )
            for tile_n in tile_ns:
                ratios = paired_ratios[(feature, tile_n)]
                wins_on = sum(value < 1.0 for value in ratios)
                wins_off = len(ratios) - wins_on
                features[feature][str(tile_n)] = {
                    "paired_measurements": len(ratios),
                    "feature_on_wins": wins_on,
                    "feature_off_wins": wins_off,
                    "geomean_on_over_off": (
                        math.exp(sum(math.log(value) for value in ratios) / len(ratios))
                        if ratios
                        else None
                    ),
                }
        return features

    def _lineage_contributions(self, selected_ids: Sequence[str]) -> dict[str, object]:
        def lineage(candidate_id: str, seen: set[str]) -> tuple[set[str], set[str]]:
            if candidate_id in seen or candidate_id not in self.candidates:
                return set(), set()
            seen.add(candidate_id)
            candidate = self.candidates[candidate_id]
            moves = {candidate.move_kind}
            candidate_ids = {candidate_id}
            for parent_id in candidate.parent_ids:
                parent_moves, parent_ids = lineage(parent_id, seen)
                moves.update(parent_moves)
                candidate_ids.update(parent_ids)
            return moves, candidate_ids

        selected_lineages = {
            candidate_id: lineage(candidate_id, set()) for candidate_id in selected_ids
        }
        seed_ids = {candidate.candidate_id for candidate in self._seed_candidates}

        def contains_lineage(
            mechanism: str,
            value: tuple[set[str], set[str]],
        ) -> bool:
            moves, candidate_ids = value
            if mechanism == "two_step":
                return bool(candidate_ids.intersection(self.search.two_step_depth))
            return mechanism in moves

        return {
            "incumbent": {
                "selected_entries": sum(
                    candidate_id in self.incumbent_ids for candidate_id in selected_ids
                ),
            },
            "seed": {
                "selected_entries": sum(
                    candidate_id in seed_ids for candidate_id in selected_ids
                ),
            },
            "crossover_lineage": {
                "selected_entries": sum(
                    contains_lineage("crossover", values)
                    for values in selected_lineages.values()
                ),
            },
            "two_step_lineage": {
                "selected_entries": sum(
                    contains_lineage("two_step", values)
                    for values in selected_lineages.values()
                ),
            },
        }

    def _banner_lines(self) -> list[str]:
        options = self.options
        code = self.code
        dirty = code.get("git_dirty_files")
        commit = code.get("git_commit") or "unknown"
        if dirty == -1:
            commit = f"{commit} + dirty status unavailable"
        elif dirty:
            commit = f"{commit} + {dirty} modified file(s)"
        compile_timeout = (
            options.compile_timeout_seconds or options.candidate_timeout_seconds
        )
        run_timeout = options.shape_timeout_seconds or options.candidate_timeout_seconds
        retime = final_timing_protocol(
            TimingProtocol(
                warmup_iters=options.warmup_iters,
                repeat_iters=options.repeat_iters,
            )
        )
        calibration = (
            "none"
            if self.calibration is None
            else (
                f"{self.calibration.environment.digest[:12]} "
                f"(knee={self.calibration.worker_count} workers)"
            )
        )
        return [
            f"start: fc={options.fc} model={options.model} dtype={options.dtype} "
            f"sm={options.sm_arch} shapes={len(self.shapes)} "
            f"incumbents={len(self.incumbent_ids)} resume={options.resume} "
            f"out={options.out_dir}",
            f"code: {code.get('module_path')} commit={commit} "
            f"source_digest={code.get('prims_ts_source_digest')} "
            f"python={code.get('python')}",
            f"limits: compile={compile_timeout:.0f}s run/shape={run_timeout:.0f}s "
            f"(lock wait bounded only by the sweep deadline) "
            f"budget={options.budget_minutes:g}m tail={options.tail_minutes:g}m",
            f"search: workers={options.worker_count} "
            f"restart_after={options.worker_restart_configs} "
            f"seed_limit={options.seed_limit} max_seeds={options.max_seeds} "
            f"final_near_best_cap={options.max_near_best_per_shape} "
            f"reference_reserve={options.reference_reserve_minutes:g}m "
            f"retry_failed={options.retry_failed}",
            f"routing: search={options.search_routing} "
            f"retime={','.join(options.retime_routings)} "
            f"selection={','.join(options.selection_routings or ())} "
            f"seed={options.routing_seed} final_shapes={len(self.final_shapes)}",
            f"timing: warmup={options.warmup_iters} repeat={options.repeat_iters} "
            f"retime_warmup={retime.warmup_iters} retime_repeat={retime.repeat_iters} "
            f"calibration={calibration}",
        ]

    def run(self) -> None:
        search_protocol = TimingProtocol(
            warmup_iters=self.options.warmup_iters,
            repeat_iters=self.options.repeat_iters,
            timing_backend=self.timing_backend,
        )
        if self.timing_backend != "cupti":
            self._progress(
                "WARNING: CUPTI is unavailable; using the hot-input "
                f"{self.timing_backend} timing fallback",
                force=True,
            )
        for line in self._banner_lines():
            self._progress(line, force=True)
        with JSONLWriter(self.journal_path, fsync=True) as writer:
            if self.options.dry_run:
                dry_candidates = dict(self.candidates)
                incumbents_by_tile: dict[int, list[Candidate]] = {}
                for candidate in self.candidates.values():
                    incumbents_by_tile.setdefault(candidate.tile_n, []).append(
                        candidate
                    )
                problem_ks = _problem_ks(self.shapes, self.options.fc)
                dry_rng = random.Random(self.options.seed)
                for _tile_n, incumbents in sorted(incumbents_by_tile.items()):
                    for seed in fixed_seed_variants(
                        incumbents,
                        limit=self.options.seed_limit,
                        sm_arch=self.options.sm_arch,
                    ):
                        for repair_index in range(3):
                            try:
                                repaired = repair_structural_candidate(
                                    seed,
                                    self.context,
                                    problem_ks,
                                    repair_index=repair_index,
                                    stage_limits=self.options.stage_limits,
                                    max_sf_ring_ratio=self.options.max_sf_ring_ratio,
                                )
                            except Exception:
                                continue
                            dry_candidates[repaired.candidate_id] = repaired
                    parent = incumbents[0]
                    for move in applicable_move_kinds(self.context, parent=parent):
                        other = None
                        if move == "crossover":
                            parent_structure = structural_options_key(parent)
                            other = next(
                                (
                                    candidate
                                    for candidate in incumbents[1:]
                                    if structural_options_key(candidate)
                                    != parent_structure
                                ),
                                None,
                            )
                        if move == "crossover" and other is None:
                            continue
                        try:
                            child = generate_move(
                                parent,
                                move,
                                dry_rng,
                                other_parent=other,
                                stage_limits=self.options.stage_limits,
                                max_sf_ring_ratio=self.options.max_sf_ring_ratio,
                                context=self.context,
                                enable_cluster_y=self.options.enable_cluster_y,
                            )
                            child = repair_structural_candidate(
                                child,
                                self.context,
                                problem_ks,
                                repair_index=0,
                                stage_limits=self.options.stage_limits,
                                max_sf_ring_ratio=self.options.max_sf_ring_ratio,
                            )
                        except Exception:
                            continue
                        dry_candidates[child.candidate_id] = child
                for candidate in dry_candidates.values():
                    try:
                        materialized = materialize_candidate(
                            candidate,
                            self.context,
                            _problem_ks(self.shapes, self.options.fc),
                            check_task_manager=True,
                        )
                        result = {
                            "status": "validated",
                            "materialized": {
                                "resolved_hashes": materialized.resolved_hashes,
                                "smem_bytes": materialized.smem_bytes,
                                "tmem_columns": materialized.tmem_columns,
                            },
                        }
                    except Exception as exc:
                        result = {
                            "status": "validation_error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    self._record(writer, candidate, result, phase="dry_run")
                self._checkpoint(phase="dry_run")
                self._progress(
                    f"dry-run complete: {len(self.records)} candidate record(s)",
                    force=True,
                )
                return
            self._baseline(writer, search_protocol)
            self._progress(
                f"baseline complete: {len(self.records)} total record(s)",
                force=True,
            )
            self._sanity_check_incumbents(writer, search_protocol)
            if self.options.retry_failed:
                self._retry_failed(writer, search_protocol)
            self._seed(writer, search_protocol)
            self._progress(
                f"seed complete: {len(self.records)} total record(s)",
                force=True,
            )
            self._adaptive(writer, search_protocol)
            self._progress(
                f"adaptive complete: generated={self.search.dispatched}, "
                f"evaluated={self.search.completed}, records={len(self.records)}",
                force=True,
            )
            self._finalize(writer, search_protocol)
            self._progress(
                f"complete: summary written to {self.options.out_dir}",
                force=True,
            )
            self.close()
