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

"""Summary and human-readable reporting for Prims-TS MoE sweeps."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from .utils import git_output, nvml_driver_version, nvml_handle_for_uuid


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                _json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def make_summary(
    *,
    fc: str,
    model: str,
    sm_arch: int,
    selected_by_tile_n: Mapping[int, Sequence[Mapping[str, object]]],
    per_shape: Mapping[str, Mapping[str, object]],
    verified: bool,
    correctness_failures: Sequence[Mapping[str, object]] = (),
    move_statistics: Mapping[str, Mapping[str, object]] | None = None,
    environment: Mapping[str, object] | None = None,
    activation_params: Mapping[str, object] | None = None,
) -> dict[str, object]:
    selected = {
        str(tile_n): [dict(row) for row in rows]
        for tile_n, rows in sorted(selected_by_tile_n.items())
    }
    return {
        "schema_version": 1,
        "fc": fc,
        "model": model,
        "sm_arch": int(sm_arch),
        "verified": bool(verified),
        "selected_menu": selected,
        "menu_size_per_tile_n": {
            tile_n: len(rows) for tile_n, rows in selected.items()
        },
        "per_shape": {key: dict(value) for key, value in per_shape.items()},
        "correctness_failures": [dict(value) for value in correctness_failures],
        "move_statistics": {
            key: dict(value) for key, value in (move_statistics or {}).items()
        },
        "environment": dict(environment or {}),
        "activation_params": dict(activation_params or {}),
    }


def estimate_weight_bandwidth_tb_s(
    active_experts: object,
    expert_weight_bytes: object,
    time_us: object,
) -> float | None:
    """Estimate effective weight bandwidth from the modeled active weights.

    This deliberately reports an estimate rather than hardware DRAM traffic:
    cache reuse and repeated reads are not modeled.
    """

    try:
        experts = float(active_experts)
        bytes_per_expert = float(expert_weight_bytes)
        duration_us = float(time_us)
    except (TypeError, ValueError):
        return None
    if (
        experts < 0
        or bytes_per_expert < 0
        or duration_us <= 0
        or not all(
            math.isfinite(value) for value in (experts, bytes_per_expert, duration_us)
        )
    ):
        return None
    return experts * bytes_per_expert / duration_us / 1e6


def capture_environment_provenance(
    repository: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Capture best-effort, JSON-safe run provenance without requiring CUDA."""

    provenance: dict[str, object] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": None,
        "flashinfer_sha": None,
        "git_dirty": None,
        "git_dirty_files": -1,
        "gpu_name": None,
        "gpu_uuid": None,
        "gpu_total_memory_bytes": None,
        "gpu_sm_count": None,
        "gpu_compute_capability": None,
        "gpu_clock_mhz": None,
        "gpu_max_sm_clock_mhz": None,
        "driver": None,
    }
    device_index = 0
    commit = git_output("rev-parse", "HEAD", repository=repository)
    status = git_output(
        "status", "--porcelain=v1", "--untracked-files=no", repository=repository
    )
    dirty_files: list[str] | int = (
        -1
        if status is None
        else sorted(
            {line[3:] for line in status.splitlines() if len(line) > 3 and line[3:]}
        )
    )
    provenance.update(
        {
            "git_commit": commit,
            "flashinfer_sha": commit,
            "git_dirty": None if dirty_files == -1 else bool(dirty_files),
            "git_dirty_files": dirty_files,
        }
    )
    provenance["driver"] = nvml_driver_version()
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            device_index = int(device)
            properties = torch.cuda.get_device_properties(device)
            provenance.update(
                {
                    "gpu_name": properties.name,
                    "gpu_uuid": str(getattr(properties, "uuid", "")) or None,
                    "gpu_total_memory_bytes": int(properties.total_memory),
                    "gpu_sm_count": int(properties.multi_processor_count),
                    "gpu_compute_capability": (
                        f"{int(properties.major)}.{int(properties.minor)}"
                    ),
                }
            )
            driver_version = getattr(torch._C, "_cuda_getDriverVersion", None)
            if provenance["driver"] is None and callable(driver_version):
                provenance["driver"] = str(driver_version())
    except (AttributeError, ImportError, RuntimeError):
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            uuid = provenance.get("gpu_uuid")
            if uuid:
                uuid_text = str(uuid)
                if not uuid_text.startswith("GPU-"):
                    uuid_text = f"GPU-{uuid_text}"
            else:
                uuid_text = ""
            handle = nvml_handle_for_uuid(pynvml, uuid_text, index=device_index)
            provenance["gpu_clock_mhz"] = int(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            )
            provenance["gpu_max_sm_clock_mhz"] = int(
                pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
            )
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    return provenance


def refresh_environment_provenance(
    environment: Mapping[str, object] | None = None,
    *,
    repository: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Refresh checkout/device fields while retaining prior session snapshots."""

    result = dict(environment or {})
    current = capture_environment_provenance(repository)
    history_value = result.get("provenance_history", ())
    history = (
        [dict(value) for value in history_value if isinstance(value, Mapping)]
        if isinstance(history_value, Sequence)
        else []
    )
    prior = {key: result.get(key) for key in current if key in result}
    if prior and (not history or history[-1] != prior):
        history.append(prior)
    history.append(dict(current))
    result.update(current)
    result["provenance_history"] = history
    return result


def render_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# Prims-TS MoE sweep report",
        "",
        _finalization_status_line(summary),
        "",
        f"- Model: `{summary.get('model')}`",
        f"- FC: `{summary.get('fc')}`",
        f"- SM architecture: `{summary.get('sm_arch')}`",
        f"- Correctness verified: `{bool(summary.get('verified'))}`",
    ]
    worker_utilization = summary.get("worker_utilization")
    profile = summary.get("search_loop_profile", {})
    if isinstance(worker_utilization, (int, float)) and isinstance(profile, Mapping):
        lines.append(
            "- Adaptive-search worker utilization: "
            f"`{float(worker_utilization):.1%}` "
            f"({float(profile.get('worker_busy_seconds', 0.0)):.1f} worker-s / "
            f"{float(profile.get('worker_capacity_seconds', 0.0)):.1f} capacity-s)"
        )
    confirmation = summary.get("search_confirmation", {})
    if isinstance(confirmation, Mapping):
        lines.append(
            "- Unconfirmed suspicious search timings: "
            f"`{int(confirmation.get('unconfirmed_pairs', 0))}` "
            f"across `{int(confirmation.get('unconfirmed_candidates', 0))}` "
            "candidate(s)"
        )
    clock_values = summary.get("gpu_clocks", {})
    clock_phases = clock_values if isinstance(clock_values, Mapping) else {}
    low_clock_phases = [
        str(phase)
        for phase, values in clock_phases.items()
        if isinstance(values, Mapping)
        and float(values.get("low_clock_fraction") or 0.0) > 0.05
    ]
    if low_clock_phases:
        lines[2:2] = [
            "**Warning: sustained low GPU SM clocks were observed during "
            + ", ".join(sorted(low_clock_phases))
            + ".**",
            "",
        ]
    activation_params = summary.get("activation_params")
    if isinstance(activation_params, Mapping) and activation_params:
        lines.append(
            "- Activation parameters: `"
            + ", ".join(
                f"{name}={activation_params.get(name)}"
                for name in ("activation", "bias", "alpha", "beta", "clamp")
            )
            + "`"
        )
    lines.extend(["", "## GPU clocks", ""])
    lines.extend(
        _markdown_table(
            (
                "Phase",
                "Samples",
                "Min MHz",
                "Median MHz",
                "Max MHz",
                "Rated max MHz",
                "Below 97%",
                "Event reasons",
            ),
            _gpu_clock_rows(summary),
            empty_message="GPU clock telemetry was unavailable.",
        )
    )
    lines.extend(["", "## Selected menu", ""])
    menu = summary.get("selected_menu", {})
    if isinstance(menu, Mapping):
        for tile_n, rows in sorted(menu.items(), key=lambda item: int(item[0])):
            lines.append(f"- tile N {tile_n}: {len(rows)} entries")
    lines.extend(["", "## Per-shape results", ""])
    per_shape = summary.get("per_shape", {})
    shape_rows = (
        [
            (str(shape_id), row)
            for shape_id, row in per_shape.items()
            if isinstance(row, Mapping)
        ]
        if isinstance(per_shape, Mapping)
        else []
    )
    routing_names = sorted(
        {routing for _, row in shape_rows for routing in _routing_results(row)}
    )
    shape_headers = [
        "Shape",
        "Pre-sweep best (µs)",
        "Best found (µs)",
        "Best speedup",
        "Selected (µs)",
        "Menu speedup",
        "Est. weight BW (TB/s)",
    ]
    shape_headers.extend(
        f"{routing} best / best speedup / menu speedup / BW"
        for routing in routing_names
    )
    lines.extend(
        _markdown_table(
            shape_headers,
            _shape_table_rows(shape_rows, routing_names),
        )
    )
    lines.extend(["", "## Menu size trade-off", ""])
    portfolio = summary.get("portfolio", {})
    curve = portfolio.get("gain_vs_size", ()) if isinstance(portfolio, Mapping) else ()
    if curve:
        lines.extend(
            _markdown_table(
                (
                    "Entries",
                    "Added candidate",
                    "Geomean speedup",
                    "Minimum speedup",
                    "Covered shapes",
                ),
                [
                    [
                        _escape_cell(point.get("entries")),
                        f"`{_escape_cell(point.get('added_candidate_id', '—'))}`",
                        _number(
                            point.get("geomean_speedup_vs_incumbent"),
                            suffix="×",
                        ),
                        _number(
                            point.get("minimum_speedup_vs_incumbent"),
                            suffix="×",
                        ),
                        _escape_cell(point.get("covered_shapes", "—")),
                    ]
                    for point in curve
                    if isinstance(point, Mapping)
                ],
            )
        )
    else:
        lines.append("- No comparable menu prefixes were available.")
    failures = _failure_rows(summary)
    lines.extend(["", "## Finalization failures", ""])
    lines.extend(
        _markdown_table(
            ("Candidate", "tile_n", "Move", "Shape", "Reason", "Max error"),
            failures,
            empty_message="None",
        )
    )
    lines.extend(["", "## Reference consistency deviations", ""])
    lines.extend(
        _markdown_table(
            (
                "Candidate",
                "Shape",
                "Mean abs / baseline",
                "Max abs / baseline",
                "Max rel exact match",
            ),
            _reference_consistency_rows(summary),
            empty_message="No selected menu entry deviated from the seeded reference baseline.",
        )
    )
    lines.extend(["", "## Search failures", ""])
    lines.extend(
        _markdown_table(
            (
                "Phase",
                "Status",
                "Failure",
                "Exit",
                "tile_n",
                "Move",
                "Count",
                "Examples",
                "Error",
            ),
            _search_failure_rows(summary),
            empty_message="None",
        )
    )
    lines.extend(["", "## Move statistics", ""])
    lines.extend(
        _markdown_table(
            (
                "tile_n",
                "Move",
                "Proposed",
                "Attempts",
                "Successes",
                "Failures",
                "Duplicates",
                "Rejected",
                "Yield",
                "Top rejection",
            ),
            _move_statistics_rows(summary),
            empty_message="No adaptive moves were recorded.",
        )
    )
    lines.extend(["", "## Group dispatch health", ""])
    lines.extend(
        _markdown_table(
            ("tile_n", "Draws", "Proposals", "Dropped draws", "Top drop", "Warnings"),
            _group_dispatch_rows(summary),
            empty_message="No adaptive group dispatch statistics were recorded.",
        )
    )
    lines.extend(["", "## Feature effects", ""])
    lines.extend(
        _markdown_table(
            (
                "Feature",
                "tile_n",
                "Pairs",
                "Feature-on wins",
                "Feature-off wins",
                "Geomean on/off",
            ),
            _feature_effect_rows(summary),
            empty_message="No matched feature measurements were available.",
        )
    )
    lines.extend(["", "## Selected-menu lineage", ""])
    lines.append(
        "Rows overlap. Lineage rows include the selected entry and all of its "
        "ancestors."
    )
    lines.append("")
    lines.extend(
        _markdown_table(
            ("Origin or lineage", "Selected entries"),
            _lineage_rows(summary),
            empty_message="No selected-menu lineage information was available.",
        )
    )
    return "\n".join(lines) + "\n"


def _finalization_counts(summary: Mapping[str, object]) -> tuple[int, int, int]:
    finalization = summary.get("finalization", {})
    if not isinstance(finalization, Mapping):
        finalization = {}
    failures = summary.get("correctness_failures", ())
    failed = int(finalization.get("failed_count", len(failures or ())))
    errors = finalization.get("verification_errors", ())
    unverified = int(finalization.get("unverified_count", len(errors or ())))
    successfully_retimed = int(finalization.get("successfully_retimed", 0))
    verified = int(
        finalization.get(
            "verified_count",
            max(0, successfully_retimed - failed - unverified),
        )
    )
    return verified, unverified, failed


def _finalization_status_line(summary: Mapping[str, object]) -> str:
    finalization = summary.get("finalization", {})
    complete = (
        bool(finalization.get("complete"))
        if isinstance(finalization, Mapping) and "complete" in finalization
        else bool(summary.get("verified"))
    )
    verified, unverified, failed = _finalization_counts(summary)
    retimed = (
        int(finalization.get("successfully_retimed", 0))
        if isinstance(finalization, Mapping)
        else 0
    )
    retime_pool = (
        int(finalization.get("retime_pool_size", 0))
        if isinstance(finalization, Mapping)
        else 0
    )
    reference_errors = (
        int(finalization.get("reference_errors", 0))
        if isinstance(finalization, Mapping)
        else 0
    )
    menu = summary.get("selected_menu", {})
    has_selected_entries = isinstance(menu, Mapping) and any(
        isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and rows
        for rows in menu.values()
    )
    if complete and bool(summary.get("verified")):
        status = "VERIFIED"
    elif has_selected_entries:
        status = "UNVERIFIED"
    elif failed:
        status = "FAILED"
    else:
        status = "INCOMPLETE"
    search_failures = summary.get("search_failures", ())
    search_failure_count = (
        sum(
            int(row.get("count", 0))
            for row in search_failures
            if isinstance(row, Mapping)
        )
        if isinstance(search_failures, Sequence)
        and not isinstance(search_failures, (str, bytes))
        else 0
    )
    return (
        f"**Finalization: {status}** — verified: {verified}; "
        f"unverified: {unverified}; failed: {failed}; "
        f"re-timed: {retimed}/{retime_pool}; reference errors: {reference_errors}; "
        f"search failures: {search_failure_count}."
    )


def _routing_results(row: Mapping[str, object]) -> Mapping[str, object]:
    value = row.get("per_routing")
    return value if isinstance(value, Mapping) else {}


def _first_value(row: Mapping[str, object], *names: str) -> object:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return None


def _number(value: object, *, suffix: str = "", digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _escape_cell(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}{suffix}"


def _count(value: object) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number) or not number.is_integer():
        return "—"
    return str(int(number))


def _shape_metrics(row: Mapping[str, object]) -> tuple[object, object, object]:
    existing = row.get("best_existing_us")
    found = row.get("best_found_us")
    speedup = row.get("speedup_vs_pre_sweep")
    if speedup is None:
        try:
            if float(found) > 0:
                speedup = float(existing) / float(found)
        except (TypeError, ValueError):
            pass
    bandwidth = row.get("estimated_weight_bandwidth_tb_s")
    if bandwidth is None:
        bandwidth = estimate_weight_bandwidth_tb_s(
            row.get("active_experts"),
            row.get("expert_weight_bytes"),
            found,
        )
    return found, speedup, bandwidth


def _shape_table_rows(
    shape_rows: Sequence[tuple[str, Mapping[str, object]]],
    routing_names: Sequence[str],
) -> list[list[str]]:
    result: list[list[str]] = []
    for shape_id, row in shape_rows:
        found, speedup, bandwidth = _shape_metrics(row)
        values = [
            f"`{_escape_cell(row.get('shape_label', shape_id))}`",
            _number(row.get("best_existing_us")),
            _number(found),
            _number(speedup, suffix="×"),
            _number(row.get("selected_us")),
            _number(row.get("menu_speedup_vs_incumbent"), suffix="×"),
            _number(bandwidth),
        ]
        routing = _routing_results(row)
        for name in routing_names:
            route_value = routing.get(name)
            if isinstance(route_value, Mapping):
                route_found, route_speedup, route_bandwidth = _shape_metrics(
                    route_value
                )
                route_menu_speedup = route_value.get("menu_speedup_vs_pre_sweep")
            else:
                route_found, route_speedup, route_bandwidth = route_value, None, None
                route_menu_speedup = None
            values.append(
                " / ".join(
                    (
                        _number(route_found),
                        _number(route_speedup, suffix="×"),
                        _number(route_menu_speedup, suffix="×"),
                        _number(route_bandwidth),
                    )
                )
            )
        result.append(values)
    return result


def _search_failure_rows(summary: Mapping[str, object]) -> list[list[str]]:
    failures = summary.get("search_failures", ())
    if not isinstance(failures, Sequence) or isinstance(failures, (str, bytes)):
        return []
    rows = []
    for failure in failures:
        if not isinstance(failure, Mapping):
            continue
        examples = failure.get("example_candidate_ids", ())
        shown_examples = (
            ", ".join(f"`{str(value)[:12]}`" for value in examples)
            if isinstance(examples, Sequence) and not isinstance(examples, (str, bytes))
            else "—"
        )
        rows.append(
            [
                _escape_cell(failure.get("phase", "—")),
                _escape_cell(failure.get("status", "—")),
                _escape_cell(failure.get("failure", "—")),
                _escape_cell(
                    failure.get("exit_code")
                    if failure.get("exit_code") is not None
                    else "—"
                ),
                _escape_cell(
                    failure.get("tile_n") if failure.get("tile_n") is not None else "—"
                ),
                _escape_cell(failure.get("move_kind", "—")),
                _escape_cell(failure.get("count", "—")),
                shown_examples,
                _compact_reason(failure.get("error") or "—"),
            ]
        )
    return rows


def _gpu_clock_rows(summary: Mapping[str, object]) -> list[list[str]]:
    phases = summary.get("gpu_clocks", {})
    if not isinstance(phases, Mapping):
        return []
    rows = []
    for phase, values in sorted(phases.items()):
        if not isinstance(values, Mapping):
            continue
        reasons = values.get("clock_event_reasons", {})
        reason_text = (
            ", ".join(f"{name}={count}" for name, count in sorted(reasons.items()))
            if isinstance(reasons, Mapping) and reasons
            else "—"
        )
        rows.append(
            [
                _escape_cell(phase),
                _escape_cell(values.get("samples", "—")),
                _number(values.get("min_sm_clock_mhz"), digits=0),
                _number(values.get("median_sm_clock_mhz"), digits=0),
                _number(values.get("max_sm_clock_mhz"), digits=0),
                _number(values.get("max_rated_sm_clock_mhz"), digits=0),
                _number(
                    100.0 * float(values.get("low_clock_fraction") or 0.0),
                    suffix="%",
                    digits=1,
                ),
                _compact_reason(reason_text),
            ]
        )
    return rows


def _move_statistics_rows(summary: Mapping[str, object]) -> list[list[str]]:
    statistics = summary.get("move_statistics", {})
    if not isinstance(statistics, Mapping):
        return []
    rows = []
    for tile_n, by_move in sorted(statistics.items(), key=lambda item: int(item[0])):
        if not isinstance(by_move, Mapping):
            continue
        for move, values in sorted(by_move.items()):
            if not isinstance(values, Mapping):
                continue
            rejection_reasons = values.get("rejection_reasons", {})
            top_rejection = "—"
            if isinstance(rejection_reasons, Mapping) and rejection_reasons:
                reason, count = max(
                    rejection_reasons.items(),
                    key=lambda item: int(item[1]),
                )
                top_rejection = f"{_compact_reason(reason)} ({count})"
            rows.append(
                [
                    _escape_cell(tile_n),
                    f"`{_escape_cell(move)}`",
                    _escape_cell(values.get("proposed", "—")),
                    _escape_cell(values.get("attempts", "—")),
                    _escape_cell(values.get("successes", "—")),
                    _escape_cell(values.get("failures", "—")),
                    _escape_cell(values.get("duplicates", "—")),
                    _escape_cell(values.get("rejected", "—")),
                    _number(values.get("evaluation_yield")),
                    top_rejection,
                ]
            )
    return rows


def _group_dispatch_rows(summary: Mapping[str, object]) -> list[list[str]]:
    statistics = summary.get("group_dispatch_stats", {})
    if not isinstance(statistics, Mapping):
        return []
    rows = []
    for tile_n, values in sorted(statistics.items(), key=lambda item: int(item[0])):
        if not isinstance(values, Mapping):
            continue
        drops = values.get("drops_by_reason", {})
        drop_total = (
            sum(int(count) for count in drops.values())
            if isinstance(drops, Mapping)
            else 0
        )
        top_drop = "—"
        if isinstance(drops, Mapping) and drops:
            reason, count = max(drops.items(), key=lambda item: int(item[1]))
            top_drop = f"{_escape_cell(reason)} ({count})"
        rows.append(
            [
                _escape_cell(tile_n),
                _escape_cell(values.get("draws", 0)),
                _escape_cell(values.get("proposals", 0)),
                _escape_cell(drop_total),
                top_drop,
                _escape_cell(values.get("starvation_warnings", 0)),
            ]
        )
    return rows


def _feature_effect_rows(summary: Mapping[str, object]) -> list[list[str]]:
    effects = summary.get("feature_effects", {})
    if not isinstance(effects, Mapping):
        return []
    rows = []
    for feature, by_tile in sorted(effects.items()):
        if not isinstance(by_tile, Mapping):
            continue
        for tile_n, values in sorted(by_tile.items(), key=lambda item: int(item[0])):
            if not isinstance(values, Mapping):
                continue
            rows.append(
                [
                    f"`{_escape_cell(feature)}`",
                    _escape_cell(tile_n),
                    _count(values.get("paired_measurements")),
                    _count(values.get("feature_on_wins")),
                    _count(values.get("feature_off_wins")),
                    _number(values.get("geomean_on_over_off"), suffix="×"),
                ]
            )
    return rows


def _lineage_rows(summary: Mapping[str, object]) -> list[list[str]]:
    lineage = summary.get("lineage_contributions", {})
    if not isinstance(lineage, Mapping):
        return []
    labels = {
        "incumbent": "incumbent (own origin)",
        "seed": "seed (own origin)",
        "crossover_lineage": "crossover in lineage",
        "two_step_lineage": "two-step in lineage",
    }
    rows = []
    for mechanism, values in sorted(lineage.items()):
        if not isinstance(values, Mapping):
            continue
        rows.append(
            [
                f"`{_escape_cell(labels.get(str(mechanism), mechanism))}`",
                _escape_cell(values.get("selected_entries", "—")),
            ]
        )
    return rows


def _failure_rows(summary: Mapping[str, object]) -> list[list[str]]:
    failures: list[object] = list(summary.get("correctness_failures", ()) or ())
    finalization = summary.get("finalization", {})
    if isinstance(finalization, Mapping):
        failures.extend(finalization.get("verification_errors", ()) or ())
    rows: list[list[str]] = []
    for value in failures:
        if not isinstance(value, Mapping):
            rows.append(["—", "—", "—", "—", _escape_cell(value), "—"])
            continue
        candidate = value.get("candidate")
        candidate_data = candidate if isinstance(candidate, Mapping) else {}
        result = value.get("result")
        result_data = result if isinstance(result, Mapping) else {}
        checks = result_data.get("checks")
        check_rows = (
            [check for check in checks if isinstance(check, Mapping)]
            if isinstance(checks, Sequence) and not isinstance(checks, (str, bytes))
            else []
        )
        failed_checks = [
            check
            for check in check_rows
            if check.get("passed") is not True and not check.get("skipped")
        ]
        if not failed_checks:
            failed_checks = [None]
        for check in failed_checks:
            check_data = check or {}
            reason = _first_value(check_data, "failure_reason", "reason", "error")
            if (
                reason is None
                and check is not None
                and check_data.get("passed") is False
            ):
                reason = "reference mismatch"
            reason = reason or _first_value(
                value, "failure_reason", "reason", "error", "status"
            )
            reason = reason or _first_value(
                result_data, "failure_reason", "reason", "error", "status"
            )
            max_error = _format_max_error(check_data)
            if max_error == "—":
                max_error = _format_max_error(value)
            rows.append(
                [
                    f"`{_escape_cell(value.get('candidate_id', '—'))}`",
                    _escape_cell(
                        _first_value(value, "tile_n")
                        or _first_value(candidate_data, "tile_n")
                        or "—"
                    ),
                    _escape_cell(
                        _first_value(value, "move", "move_kind")
                        or _first_value(candidate_data, "move", "move_kind")
                        or "—"
                    ),
                    _escape_cell(
                        _first_value(check_data, "shape_id", "shape")
                        or _first_value(value, "shape_id", "shape")
                        or "—"
                    ),
                    _compact_reason(reason or "unknown failure"),
                    max_error,
                ]
            )
    return rows


def _reference_consistency_rows(
    summary: Mapping[str, object],
) -> list[list[str]]:
    consistency = summary.get("reference_consistency", {})
    if not isinstance(consistency, Mapping):
        return []
    values = consistency.get("flagged_menu_rows", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    rows = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        rows.append(
            [
                f"`{_escape_cell(value.get('candidate_id', '—'))}`",
                f"`{_escape_cell(value.get('shape_id', '—'))}`",
                _number(value.get("mean_abs_ratio"), suffix="×"),
                _number(value.get("max_abs_ratio"), suffix="×"),
                _escape_cell(value.get("max_rel_matches", "—")),
            ]
        )
    return rows


def _format_max_error(value: Mapping[str, object]) -> str:
    absolute = _first_value(value, "max_abs_error", "max_error")
    relative = _first_value(value, "max_rel_error")
    if absolute is None and relative is None:
        return "—"
    parts = []
    if absolute is not None:
        parts.append(f"abs={_number(absolute)}")
    if relative is not None:
        parts.append(f"rel={_number(relative)}")
    return ", ".join(parts)


def _escape_cell(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", "<br>")


def _compact_reason(value: object, *, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return _escape_cell(text)


def _markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    empty_message: str = "No per-shape measurements were available.",
) -> list[str]:
    if not rows:
        return [f"- {empty_message}"]
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    result.extend(
        "| " + " | ".join(_escape_cell(value) for value in row) + " |" for row in rows
    )
    return result


def write_summary(out_dir: Path, summary: Mapping[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_dir / "summary.json", dict(summary))
    (out_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
