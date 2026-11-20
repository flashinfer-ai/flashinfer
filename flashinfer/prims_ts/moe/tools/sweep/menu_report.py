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

"""Machine-readable and Markdown reports for joint MoE menu selection."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

from .db import atomic_write_json
from .menu_config import AutotuneCost
from .menu_select import JointSelection, menu_metrics


def _fmt(value: float, digits: int = 4) -> str:
    return "∞" if not math.isfinite(value) else f"{value:.{digits}f}x"


def _incumbent_metrics(selection: JointSelection, cost: AutotuneCost):
    incumbent_ids = {
        candidate_id
        for candidate_id, candidate in selection.candidates.items()
        if candidate.shipped_incumbent
    }
    return menu_metrics(
        incumbent_ids,
        candidates=selection.candidates,
        shapes=selection.shapes,
        timings=selection.timings,
        global_best=selection.global_best,
        cost=cost,
        deployments=selection.deployments,
    )


def _held_out_by_n(selection: JointSelection) -> dict[int, tuple[float, float]]:
    values: dict[int, list[float]] = {}
    for result in selection.leave_one_model_out.values():
        for row in result.get("held_out_curve", ()):
            value = float(row["geomean_regret"])
            values.setdefault(int(row["n"]), []).append(value)
    return {
        n: (
            math.exp(math.fsum(math.log(value) for value in rows) / len(rows)),
            max(rows),
        )
        for n, rows in values.items()
    }


def _knee(
    selection: JointSelection, held_out: Mapping[int, tuple[float, float]]
) -> int:
    if not selection.order:
        return 0
    start = max(1, selection.floor_size)
    boundaries = []
    for index, entry in enumerate(selection.order):
        next_entry = (
            selection.order[index + 1] if index + 1 < len(selection.order) else None
        )
        grouped = entry.reason == "greedy_pair" or entry.reason.startswith("floor:")
        if grouped and next_entry is not None and next_entry.reason == entry.reason:
            continue
        boundaries.append(entry.rank)
    boundaries = [n for n in boundaries if n >= start]
    if held_out:
        last = len(selection.order)
        for n in boundaries:
            if n == last:
                continue
            current = held_out.get(n)
            future_n = next((value for value in boundaries if value > n), last)
            future = held_out.get(future_n)
            if current is not None and future is not None:
                if current[0] - future[0] < 0.001:
                    return n
    for entry in selection.order:
        if entry.rank not in boundaries:
            continue
        if entry.marginal_gain < 0.001:
            return entry.rank
    return len(selection.order)


def _standalone_score(selection: JointSelection, candidate_id: str) -> float:
    candidate = selection.candidates[candidate_id]
    weighted_logs = 0.0
    total_weight = 0.0
    for shape in selection.shapes.values():
        if candidate.tile_n not in shape.selected_tile_ns:
            continue
        if (
            candidate.fc == "fc1"
            and candidate.activation_class != shape.activation_class
        ):
            continue
        value = selection.timings.get(candidate_id, {}).get(shape.shape_id)
        if value is None or value.time is None:
            continue
        peers = [
            timing.time
            for other_id, other in selection.candidates.items()
            if other.fc == candidate.fc
            and other.activation_class == candidate.activation_class
            and other.tile_n == candidate.tile_n
            and (timing := selection.timings.get(other_id, {}).get(shape.shape_id))
            is not None
            and timing.time is not None
        ]
        if not peers:
            continue
        weighted_logs += shape.weight * math.log(value.time / min(peers))
        total_weight += shape.weight
    return math.exp(weighted_logs / total_weight) if total_weight else 10.0


def _breakdown(
    selection: JointSelection,
    selected_ids: Iterable[str],
    *,
    cost: AutotuneCost,
) -> list[dict[str, object]]:
    selected = set(selected_ids)
    result = []
    for deployment in selection.deployments:
        shapes = {
            shape_id: shape
            for shape_id, shape in selection.shapes.items()
            if shape.deployment == deployment.deployment
            and shape.tp == deployment.tp
            and shape.ep == deployment.ep
        }
        metrics = menu_metrics(
            selected,
            candidates=selection.candidates,
            shapes=shapes,
            timings=selection.timings,
            global_best=selection.global_best,
            cost=cost,
            deployments=(deployment,),
        )
        result.append(
            {
                "deployment": deployment.deployment,
                "activation_class": deployment.activation_class,
                "tp": deployment.tp,
                "ep": deployment.ep,
                "weight": deployment.weight,
                **asdict(metrics),
            }
        )
    return result


def selection_payload(
    selection: JointSelection,
    *,
    cost: AutotuneCost,
    order_by: str,
    input_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the complete JSON-serializable selection result."""

    inputs = {
        key: input_manifest[key]
        for key in ("target", "quant_contexts", "code", "calibration")
        if input_manifest is not None and key in input_manifest
    }
    incumbent = _incumbent_metrics(selection, cost)
    entries = []
    for entry in selection.order:
        candidate = selection.candidates[entry.candidate_id]
        entries.append(
            {
                "rank": entry.rank,
                "candidate_id": entry.candidate_id,
                "fc": candidate.fc,
                "activation_class": candidate.activation_class,
                "tile_n": candidate.tile_n,
                "options": dict(candidate.options),
                "provenance": list(candidate.provenance),
                "shipped_incumbent": candidate.shipped_incumbent,
                "standalone_regret": _standalone_score(selection, entry.candidate_id),
                "reason": entry.reason,
                "marginal_gain": entry.marginal_gain,
                "delta_kernels": entry.delta_kernels,
                "delta_pairs": entry.delta_pairs,
                "metrics": asdict(entry.metrics),
                "speedup_over_incumbents": (
                    incumbent.geomean_regret / entry.metrics.geomean_regret
                    if math.isfinite(entry.metrics.geomean_regret)
                    else 0.0
                ),
            }
        )
    all_ids = tuple(entry.candidate_id for entry in selection.order)
    held_out = _held_out_by_n(selection)
    knee = _knee(selection, held_out)
    checkpoints = sorted(
        {
            selection.floor_size,
            knee,
            len(all_ids),
        }
        - {0}
    )
    return {
        "schema_version": 2,
        "order_by": order_by,
        "floor_size": selection.floor_size,
        "floor_tile_ns": list(selection.floor_tile_ns),
        "floor_plausible_ratio": selection.floor_plausible_ratio,
        "candidate_count": len(selection.candidates),
        "shape_count": len(selection.shapes),
        "reference_shape_count": len(selection.global_best),
        "not_applicable_measurement_count": sum(
            timing.kind.value == "not_applicable"
            for candidate_timings in selection.timings.values()
            for timing in candidate_timings.values()
        ),
        "unmeasured_candidate_count": sum(
            not any(
                timing.is_measured
                for timing in selection.timings.get(candidate_id, {}).values()
            )
            for candidate_id in selection.candidates
        ),
        "inputs": inputs,
        "candidate_sources": {
            source: sum(
                source in candidate.provenance
                for candidate in selection.candidates.values()
            )
            for source in sorted(
                {
                    source
                    for candidate in selection.candidates.values()
                    for source in candidate.provenance
                }
            )
        },
        "incumbent_metrics": asdict(incumbent),
        "incumbent_breakdown": _breakdown(
            selection,
            (
                candidate_id
                for candidate_id, candidate in selection.candidates.items()
                if candidate.shipped_incumbent
            ),
            cost=cost,
        ),
        "knee": knee,
        "entries": entries,
        "not_selected": sorted(set(selection.candidates) - set(all_ids)),
        "leave_one_model_out": selection.leave_one_model_out,
        "breakdowns": {
            str(n): _breakdown(selection, all_ids[:n], cost=cost) for n in checkpoints
        },
    }


def write_menu_report(
    selection: JointSelection,
    *,
    out_dir: str | Path,
    cost: AutotuneCost,
    order_by: str,
    input_manifest: Mapping[str, object] | None = None,
) -> tuple[Path, Path]:
    """Write `menu_order.json` and a concise Markdown trade-off report."""

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = selection_payload(
        selection,
        cost=cost,
        order_by=order_by,
        input_manifest=input_manifest,
    )
    order_path = destination / "menu_order.json"
    report_path = destination / "menu_report.md"
    atomic_write_json(order_path, payload)

    incumbent = payload["incumbent_metrics"]
    held_out = _held_out_by_n(selection)
    inputs = payload["inputs"]
    raw_target = inputs.get("target", {}) if isinstance(inputs, Mapping) else {}
    target = raw_target if isinstance(raw_target, Mapping) else {}
    raw_quant_contexts = (
        inputs.get("quant_contexts", {}) if isinstance(inputs, Mapping) else {}
    )
    quant_contexts = (
        raw_quant_contexts if isinstance(raw_quant_contexts, Mapping) else {}
    )
    lines = [
        "# Prims-TS MoE menu report",
        "",
        f"- Target: **SM{target.get('sm_arch', 'unknown')} "
        f"{target.get('dtype', 'unknown')}**",
        f"- Deployments: **{len(target.get('deployments', ())) if target else 0}**",
        f"- Candidates: **{payload['candidate_count']}**",
        f"- Weighted shapes: **{payload['shape_count']}**",
        f"- Shapes with a measured FC1+FC2 reference: "
        f"**{payload['reference_shape_count']}**",
        f"- Not-applicable measurements: "
        f"**{payload['not_applicable_measurement_count']}**",
        f"- Candidates without a successful measurement: "
        f"**{payload['unmeasured_candidate_count']}**",
        f"- Mandatory floor: **{payload['floor_size']} entries**",
        "- Extra floor tiles: "
        f"**{', '.join(str(value) for value in payload['floor_tile_ns']) or 'none'}** "
        f"(plausible within {float(payload['floor_plausible_ratio']):.2f}x)",
        f"- Estimated knee: **N={payload['knee']}**",
        f"- Ordering objective: **{order_by}**",
        "- Shipped-menu geomean regret: "
        f"**{_fmt(float(incumbent['geomean_regret']))}**",
        "",
        "## Inputs",
        "",
        "Candidate sources:",
        *[
            f"- `{source}`: {count}"
            for source, count in payload["candidate_sources"].items()
        ],
        "",
        "Quantization contexts:",
        *[
            f"- `{name}`: `{json.dumps(context, sort_keys=True)}`"
            for name, context in quant_contexts.items()
        ],
        "",
        "## Prefix trade-off",
        "",
        "| N | Added | FC/class/tile | Geomean regret | p95 | Worst | "
        "Speedup vs shipped | Held-out geo / worst | Kernels | "
        "Pairs/deployment mean / max | Autotune/deployment mean / max |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in payload["entries"]:
        metrics = entry["metrics"]
        held = held_out.get(int(entry["rank"]))
        held_text = "—" if held is None else f"{_fmt(held[0])} / {_fmt(held[1])}"
        marker = " (floor)" if str(entry["reason"]).startswith("floor:") else ""
        class_name = entry["activation_class"] or "all"
        lines.append(
            f"| {entry['rank']} | `{str(entry['candidate_id'])[:12]}`{marker} | "
            f"{entry['fc']}/{class_name}/{entry['tile_n']} | "
            f"{_fmt(float(metrics['geomean_regret']))} | "
            f"{_fmt(float(metrics['p95_regret']))} | "
            f"{_fmt(float(metrics['worst_regret']))} | "
            f"{float(entry['speedup_over_incumbents']):.4f}x | {held_text} | "
            f"{metrics['runtime_kernels']} | "
            f"{float(metrics['deployment_pairs_mean']):.1f} / "
            f"{metrics['deployment_pairs_max']} | "
            f"{float(metrics['deployment_seconds_mean']):.1f}s / "
            f"{float(metrics['deployment_seconds_max']):.1f}s |"
        )
    lines.extend(
        [
            "",
            "Selection stopped when no remaining entry improved any shape beyond "
            "the combined MAD noise band.",
            "",
            "## Per-deployment checkpoints",
            "",
        ]
    )
    checkpoint_rows = [
        ("Shipped incumbents", payload["incumbent_breakdown"]),
        *[(f"N={n}", rows) for n, rows in payload["breakdowns"].items()],
    ]
    for label, rows in checkpoint_rows:
        lines.extend(
            [
                f"### {label}",
                "",
                "| Deployment | TP:EP | Class | Geomean regret | p95 | Worst | "
                "Covered | Kernels | Pairs | Estimated autotune |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row['deployment']}` | {row['tp']}:{row['ep']} | "
                f"{row['activation_class']} | "
                f"{_fmt(float(row['geomean_regret']))} | "
                f"{_fmt(float(row['p95_regret']))} | "
                f"{_fmt(float(row['worst_regret']))} | "
                f"{row['covered_shapes']}/{row['total_shapes']} | "
                f"{float(row['deployment_kernels_mean']):.0f} | "
                f"{float(row['deployment_pairs_mean']):.0f} | "
                f"{float(row['deployment_seconds_mean']):.1f}s |"
            )
        lines.append("")
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return order_path, report_path
