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

"""Host-only joint FC1/FC2 menu selection."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .db import canonical_json, deterministic_id, read_jsonl
from .menu_config import AutotuneCost, activation_class, runtime_twin
from .portfolio import TimingValue
from .shapes import _SUPPORTED_TILES_BY_DTYPE, normalize_dtype


@dataclass(frozen=True)
class JointCandidate:
    candidate_id: str
    fc: str
    activation_class: str | None
    tile_n: int
    options: Mapping[str, object]
    provenance: tuple[str, ...] = ()
    shipped_incumbent: bool = False


@dataclass(frozen=True)
class JointShape:
    shape_id: str
    deployment: str
    model: str | None
    activation_class: str
    tp: int
    ep: int
    tokens: int
    routing: str
    selected_tile_ns: tuple[int, ...]
    weight: float

    @property
    def bucket_key(self) -> tuple[object, ...]:
        return (
            self.deployment,
            self.activation_class,
            self.tp,
            self.ep,
            self.tokens,
            self.selected_tile_ns,
        )


@dataclass(frozen=True)
class DeploymentBuckets:
    deployment: str
    model: str | None
    activation_class: str
    tp: int
    ep: int
    weight: float
    # (selected tile window, number of runtime token buckets with that window)
    windows: tuple[tuple[tuple[int, ...], int], ...]

    @property
    def key(self) -> tuple[str, int, int]:
        return self.deployment, self.tp, self.ep


@dataclass(frozen=True)
class MenuMetrics:
    objective: float
    geomean_regret: float
    p95_regret: float
    worst_regret: float
    covered_shapes: int
    total_shapes: int
    runtime_kernels: int
    deployment_kernels_mean: float
    deployment_pairs_mean: float
    deployment_pairs_max: int
    deployment_seconds_mean: float
    deployment_seconds_max: float


@dataclass(frozen=True)
class DeploymentCost:
    deployment: str
    model: str | None
    activation_class: str
    tp: int
    ep: int
    weight: float
    kernels: int
    pairs: int
    seconds: float

    @property
    def key(self) -> tuple[str, int, int]:
        return self.deployment, self.tp, self.ep


@dataclass(frozen=True)
class AutotuneCostBreakdown:
    runtime_kernels: int
    deployment_kernels_mean: float
    deployment_pairs_mean: float
    deployment_pairs_max: int
    deployment_seconds_mean: float
    deployment_seconds_max: float
    per_deployment: tuple[DeploymentCost, ...]


@dataclass(frozen=True)
class OrderedEntry:
    rank: int
    candidate_id: str
    reason: str
    marginal_gain: float
    delta_kernels: float
    delta_pairs: float
    metrics: MenuMetrics


@dataclass(frozen=True)
class JointSelection:
    candidates: Mapping[str, JointCandidate]
    shapes: Mapping[str, JointShape]
    deployments: tuple[DeploymentBuckets, ...]
    timings: Mapping[str, Mapping[str, TimingValue]]
    global_best: Mapping[str, float]
    order: tuple[OrderedEntry, ...]
    floor_size: int
    floor_tile_ns: tuple[int, ...]
    floor_plausible_ratio: float
    leave_one_model_out: Mapping[str, Mapping[str, object]]

    @property
    def selected_ids(self) -> tuple[str, ...]:
        return tuple(entry.candidate_id for entry in self.order)


def _collapse_windows(
    windows: Iterable[tuple[int, ...]],
) -> tuple[tuple[tuple[int, ...], int], ...]:
    counts: dict[tuple[int, ...], int] = {}
    for window in windows:
        counts[window] = counts.get(window, 0) + 1
    return tuple(sorted(counts.items()))


def deployment_buckets_from_target(
    target: Mapping[str, object],
) -> tuple[DeploymentBuckets, ...]:
    """Build the runtime autotune buckets described by a manifest target."""

    raw_deployments = target.get("deployments", ())
    if not isinstance(raw_deployments, Sequence) or isinstance(
        raw_deployments, (str, bytes)
    ):
        return ()

    # Lazy imports keep the menu matrix/report modules host-only until runtime
    # bucket reconstruction is actually requested.
    from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets
    from flashinfer.prims_ts.moe.config_mapper import _selected_tile_ns

    result = []
    for raw_deployment in raw_deployments:
        if not isinstance(raw_deployment, Mapping):
            continue
        name = str(raw_deployment.get("name") or raw_deployment.get("model") or "")
        model_value = raw_deployment.get("model")
        model = None if model_value is None else str(model_value)
        dtype = normalize_dtype(str(raw_deployment.get("dtype", target.get("dtype"))))
        tiles = _SUPPORTED_TILES_BY_DTYPE[dtype]
        num_experts = int(raw_deployment["num_experts"])
        top_k = int(raw_deployment["top_k"])
        class_name = activation_class(str(raw_deployment["activation"]))
        deployment_weight = float(raw_deployment.get("weight", 1.0))
        raw_points = raw_deployment.get("parallelism", raw_deployment.get("tp_ep", ()))
        if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
            continue
        for raw_point in raw_points:
            if not isinstance(raw_point, Mapping):
                continue
            if "tp" in raw_point and "ep" in raw_point:
                tp, ep = int(raw_point["tp"]), int(raw_point["ep"])
            else:
                tp_ep = raw_point.get("tp_ep")
                pieces = (
                    str(tp_ep).split(":")
                    if isinstance(tp_ep, str)
                    else list(tp_ep)
                    if isinstance(tp_ep, Sequence)
                    else ()
                )
                if len(pieces) != 2:
                    continue
                tp, ep = (int(value) for value in pieces)
            tokens = tuple(int(value) for value in raw_point.get("tokens", ()))
            if not tokens:
                continue
            tune_max = int(raw_point.get("tune_max_num_tokens") or max(tokens))
            if tune_max < max(tokens):
                raise ValueError(
                    f"{name} TP:EP {tp}:{ep} tune_max_num_tokens={tune_max} "
                    f"is below its largest token count {max(tokens)}"
                )
            buckets = get_hybrid_num_tokens_buckets(tune_max, 1)
            windows = (
                _selected_tile_ns(
                    num_tokens=bucket,
                    top_k=top_k,
                    num_local_experts=num_experts // ep,
                    num_experts=num_experts,
                    supported_tiles=tiles,
                )
                for bucket in buckets
            )
            result.append(
                DeploymentBuckets(
                    deployment=name,
                    model=model,
                    activation_class=class_name,
                    tp=tp,
                    ep=ep,
                    weight=deployment_weight * float(raw_point.get("weight", 1.0)),
                    windows=_collapse_windows(windows),
                )
            )
    return tuple(result)


def deployment_buckets_from_shapes(
    shapes: Iterable[JointShape],
) -> tuple[DeploymentBuckets, ...]:
    """Fall back to measured shape buckets when target metadata is unavailable."""

    grouped: dict[tuple[str, str | None, str, int, int], list[JointShape]] = {}
    for shape in shapes:
        key = (
            shape.deployment,
            shape.model,
            shape.activation_class,
            shape.tp,
            shape.ep,
        )
        grouped.setdefault(key, []).append(shape)
    result = []
    for (deployment, model, class_name, tp, ep), values in sorted(grouped.items()):
        # Routing variants repeat the same runtime token bucket. Count each
        # token/window once, while retaining the full group's aggregate weight.
        token_windows = {(shape.tokens, shape.selected_tile_ns) for shape in values}
        result.append(
            DeploymentBuckets(
                deployment=deployment,
                model=model,
                activation_class=class_name,
                tp=tp,
                ep=ep,
                weight=math.fsum(shape.weight for shape in values),
                windows=_collapse_windows(window for _tokens, window in token_windows),
            )
        )
    return tuple(result)


def _logical_shape_id(row: Mapping[str, object]) -> str:
    return deterministic_id(
        {
            "deployment": row.get("deployment"),
            "model": row.get("model"),
            "activation_class": row.get("activation_class"),
            "tp": row.get("tp"),
            "ep": row.get("ep"),
            "tokens": row.get("tokens"),
            "routing": row.get("routing"),
            "selected_tile_ns": row.get("selected_tile_ns"),
        },
        length=20,
    )


def load_joint_inputs(
    retime_dir: str | Path,
) -> tuple[
    dict[str, JointCandidate],
    dict[str, JointShape],
    dict[str, dict[str, TimingValue]],
    AutotuneCost,
    tuple[DeploymentBuckets, ...],
]:
    """Load the menu manifest and flat re-time matrix."""

    root = Path(retime_dir)
    manifest = json.loads((root / "menu_manifest.json").read_text(encoding="utf-8"))
    target = manifest["target"]
    cost_value = target.get("autotune_cost", {})
    cost = AutotuneCost(
        seconds_per_pair=float(cost_value.get("seconds_per_pair", 0.0)),
        seconds_per_compile=float(cost_value.get("seconds_per_compile", 0.0)),
    )
    candidates = {
        candidate_id: JointCandidate(
            candidate_id=candidate_id,
            fc=str(value["fc"]),
            activation_class=(
                None
                if value.get("activation_class") is None
                else str(value["activation_class"])
            ),
            tile_n=int(value["tile_n"]),
            options=dict(value["options"]),
            provenance=tuple(str(item) for item in value.get("provenance", ())),
            shipped_incumbent=bool(value.get("shipped_incumbent", False)),
        )
        for candidate_id, value in manifest["candidates"].items()
    }
    shapes: dict[str, JointShape] = {}
    timings: dict[str, dict[str, TimingValue]] = {}
    for row in read_jsonl(root / "matrix.jsonl"):
        if row.get("record_type") == "candidate_context":
            candidate_id = str(row["candidate_id"])
            context_key = (
                f"__context__:{row.get('deployment')}:{row.get('fc')}:"
                f"{row.get('activation_class')}"
            )
            timings.setdefault(candidate_id, {})[context_key] = (
                TimingValue.not_applicable()
                if row.get("status") == "not_applicable"
                else TimingValue.failed()
            )
            continue
        shape_key = _logical_shape_id(row)
        class_name = str(row.get("activation_class") or "")
        if not class_name:
            # FC2 rows share the FC1 activation class of the deployment.
            deployment = str(row.get("deployment"))
            matching = [
                value
                for value in shapes.values()
                if value.deployment == deployment
                and value.tp == int(row["tp"])
                and value.ep == int(row["ep"])
                and value.tokens == int(row["tokens"])
                and value.routing == str(row["routing"])
            ]
            if matching:
                class_name = matching[0].activation_class
                row = {**row, "activation_class": class_name}
                shape_key = _logical_shape_id(row)
        if not class_name:
            # The matrix is normally ordered FC1 then FC2. Defer an unusual
            # FC2-first row until its matching FC1 row appears.
            continue
        shapes.setdefault(
            shape_key,
            JointShape(
                shape_id=shape_key,
                deployment=str(row.get("deployment")),
                model=None if row.get("model") is None else str(row["model"]),
                activation_class=class_name,
                tp=int(row["tp"]),
                ep=int(row["ep"]),
                tokens=int(row["tokens"]),
                routing=str(row["routing"]),
                selected_tile_ns=tuple(int(v) for v in row["selected_tile_ns"]),
                weight=float(row.get("weight", 1.0)),
            ),
        )
        candidate_id = str(row["candidate_id"])
        if row.get("status") == "ok":
            timings.setdefault(candidate_id, {})[shape_key] = TimingValue.measured(
                float(row["median_us"]), float(row.get("mad_us") or 0.0)
            )
        elif row.get("status") == "not_applicable":
            timings.setdefault(candidate_id, {})[shape_key] = (
                TimingValue.not_applicable()
            )
        else:
            timings.setdefault(candidate_id, {})[shape_key] = TimingValue.failed()
    # Absence still means the candidate was outside that sub-run's coverage.
    deployments = deployment_buckets_from_target(target)
    if not deployments:
        deployments = deployment_buckets_from_shapes(shapes.values())
    return candidates, shapes, timings, cost, deployments


def _best_for(
    candidate_ids: Iterable[str],
    *,
    shape_id: str,
    timings: Mapping[str, Mapping[str, TimingValue]],
) -> TimingValue | None:
    values = [
        timings[candidate_id][shape_id]
        for candidate_id in candidate_ids
        if shape_id in timings.get(candidate_id, {})
        and timings[candidate_id][shape_id].is_measured
    ]
    return min(values, key=lambda value: float(value.time)) if values else None


class _TimingIndex:
    """Per-shape, per-tile FC1/FC2 timings sorted fastest first.

    The greedy evaluates every shape for every candidate move, so scanning all
    selected candidates per lookup dominates selection time. Equal times break
    ties by candidate id.
    """

    def __init__(
        self,
        candidates: Mapping[str, JointCandidate],
        shapes: Mapping[str, JointShape],
        timings: Mapping[str, Mapping[str, TimingValue]],
    ) -> None:
        groups: dict[tuple[str, str | None, int], list[str]] = {}
        for candidate_id, candidate in candidates.items():
            class_name = candidate.activation_class if candidate.fc == "fc1" else None
            groups.setdefault((candidate.fc, class_name, candidate.tile_n), []).append(
                candidate_id
            )

        def ranked(
            key: tuple[str, str | None, int], shape_id: str
        ) -> tuple[tuple[str, TimingValue], ...]:
            values = [
                (candidate_id, timings[candidate_id][shape_id])
                for candidate_id in groups.get(key, ())
                if shape_id in timings.get(candidate_id, {})
                and timings[candidate_id][shape_id].is_measured
            ]
            values.sort(key=lambda value: (float(value[1].time), value[0]))
            return tuple(values)

        self._tiles = {
            shape.shape_id: tuple(
                (
                    ranked(("fc1", shape.activation_class, tile_n), shape.shape_id),
                    ranked(("fc2", None, tile_n), shape.shape_id),
                )
                for tile_n in shape.selected_tile_ns
            )
            for shape in shapes.values()
        }

    def shape_time(
        self, selected: set[str], shape_id: str
    ) -> tuple[float, float] | None:
        best: tuple[float, float] | None = None
        for fc1_ranked, fc2_ranked in self._tiles[shape_id]:
            fc1 = next((value for cid, value in fc1_ranked if cid in selected), None)
            if fc1 is None:
                continue
            fc2 = next((value for cid, value in fc2_ranked if cid in selected), None)
            if fc2 is None:
                continue
            assert fc1.time is not None and fc2.time is not None
            pair = (
                fc1.time + fc2.time,
                math.hypot(fc1.spread, fc2.spread),
            )
            if best is None or pair[0] < best[0]:
                best = pair
        return best


def _shape_time(
    selected: set[str],
    shape: JointShape,
    candidates: Mapping[str, JointCandidate],
    timings: Mapping[str, Mapping[str, TimingValue]],
    index: _TimingIndex | None = None,
) -> tuple[float, float] | None:
    if index is not None:
        return index.shape_time(selected, shape.shape_id)
    best: tuple[float, float] | None = None
    for tile_n in shape.selected_tile_ns:
        fc1 = _best_for(
            (
                candidate_id
                for candidate_id in selected
                if candidates[candidate_id].fc == "fc1"
                and candidates[candidate_id].activation_class == shape.activation_class
                and candidates[candidate_id].tile_n == tile_n
            ),
            shape_id=shape.shape_id,
            timings=timings,
        )
        fc2 = _best_for(
            (
                candidate_id
                for candidate_id in selected
                if candidates[candidate_id].fc == "fc2"
                and candidates[candidate_id].tile_n == tile_n
            ),
            shape_id=shape.shape_id,
            timings=timings,
        )
        if fc1 is None or fc2 is None:
            continue
        assert fc1.time is not None and fc2.time is not None
        pair = (
            fc1.time + fc2.time,
            math.hypot(fc1.spread, fc2.spread),
        )
        if best is None or pair[0] < best[0]:
            best = pair
    return best


def global_best_times(
    candidates: Mapping[str, JointCandidate],
    shapes: Mapping[str, JointShape],
    timings: Mapping[str, Mapping[str, TimingValue]],
    index: _TimingIndex | None = None,
) -> dict[str, float]:
    all_ids = set(candidates)
    result = {}
    for shape in shapes.values():
        value = _shape_time(all_ids, shape, candidates, timings, index)
        if value is not None:
            result[shape.shape_id] = value[0]
    return result


def autotune_cost(
    selected: set[str],
    *,
    candidates: Mapping[str, JointCandidate],
    deployments: Sequence[DeploymentBuckets],
    cost: AutotuneCost,
) -> AutotuneCostBreakdown:
    """Estimate what one process autotunes for each deployment target."""

    # Match menu emission before counting. Geometry-identical FC1 entries from
    # different activation classes collapse to one emitted row. If an explicit
    # drain+throttle entry is the runtime twin of another selected row, emission
    # keeps the base, merges their activation classes, and lets the mapper append
    # the twin. This can make both variants available to a class that selected
    # only one of them.
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for candidate_id in selected:
        candidate = candidates[candidate_id]
        options = dict(candidate.options)
        key = candidate.fc, canonical_json(options)
        value = grouped.setdefault(
            key,
            {
                "fc": candidate.fc,
                "tile_n": candidate.tile_n,
                "options": options,
                "classes": set(),
            },
        )
        if candidate.fc == "fc1":
            value["classes"].add(str(candidate.activation_class))
    redundant_twins = set()
    for _key, value in tuple(grouped.items()):
        twin = runtime_twin(value["options"])
        if twin is None:
            continue
        twin_key = str(value["fc"]), canonical_json(twin)
        explicit_twin = grouped.get(twin_key)
        if explicit_twin is None:
            continue
        value["classes"].update(explicit_twin["classes"])
        redundant_twins.add(twin_key)

    fc1: dict[tuple[str, int], set[tuple[object, ...]]] = {}
    fc2: dict[int, set[tuple[object, ...]]] = {}
    all_kernels: set[tuple[object, ...]] = set()
    selected_has_elementwise = any(
        candidate.fc == "fc1" and candidate.activation_class == "elementwise"
        for candidate_id, candidate in candidates.items()
        if candidate_id in selected
    )
    all_classes = {deployment.activation_class for deployment in deployments}
    for grouped_key, value in grouped.items():
        if grouped_key in redundant_twins:
            continue
        fc = str(value["fc"])
        tile_n = int(value["tile_n"])
        options = dict(value["options"])
        variants = [options]
        twin = runtime_twin(options)
        if twin is not None:
            variants.append(twin)
        classes = (
            set(value["classes"])
            if fc == "fc1" and selected_has_elementwise
            else all_classes
            if fc == "fc1"
            else {None}
        )
        for class_name in classes:
            for variant in variants:
                key = fc, class_name, tile_n, canonical_json(variant)
                all_kernels.add(key)
                if fc == "fc1":
                    fc1.setdefault((str(class_name), tile_n), set()).add(key)
                else:
                    fc2.setdefault(tile_n, set()).add(key)

    per_deployment = []
    for deployment in deployments:
        reachable_tiles = {
            tile_n for window, _count in deployment.windows for tile_n in window
        }
        kernels = set()
        for tile_n in reachable_tiles:
            kernels.update(fc1.get((deployment.activation_class, tile_n), ()))
            kernels.update(fc2.get(tile_n, ()))
        pairs = 0
        for window, count in deployment.windows:
            menu_pairs = sum(
                len(fc1.get((deployment.activation_class, tile_n), ()))
                * len(fc2.get(tile_n, ()))
                for tile_n in window
            )
            # Every Prims-TS MoE runner prepends tactic -1 to the JSON menu.
            # It is profiled once per bucket in addition to the Cartesian menu.
            pairs += count * (menu_pairs + int(menu_pairs > 0))
        seconds = (
            len(kernels) * cost.seconds_per_compile + pairs * cost.seconds_per_pair
        )
        per_deployment.append(
            DeploymentCost(
                deployment=deployment.deployment,
                model=deployment.model,
                activation_class=deployment.activation_class,
                tp=deployment.tp,
                ep=deployment.ep,
                weight=deployment.weight,
                kernels=len(kernels),
                pairs=pairs,
                seconds=seconds,
            )
        )

    total_weight = math.fsum(value.weight for value in per_deployment)

    def weighted_mean(field: str) -> float:
        if not total_weight:
            return 0.0
        return (
            math.fsum(
                value.weight * float(getattr(value, field)) for value in per_deployment
            )
            / total_weight
        )

    return AutotuneCostBreakdown(
        runtime_kernels=len(all_kernels),
        deployment_kernels_mean=weighted_mean("kernels"),
        deployment_pairs_mean=weighted_mean("pairs"),
        deployment_pairs_max=max((value.pairs for value in per_deployment), default=0),
        deployment_seconds_mean=weighted_mean("seconds"),
        deployment_seconds_max=max(
            (value.seconds for value in per_deployment), default=0.0
        ),
        per_deployment=tuple(per_deployment),
    )


def menu_metrics(
    selected: set[str],
    *,
    candidates: Mapping[str, JointCandidate],
    shapes: Mapping[str, JointShape],
    timings: Mapping[str, Mapping[str, TimingValue]],
    global_best: Mapping[str, float],
    cost: AutotuneCost,
    deployments: Sequence[DeploymentBuckets] | None = None,
    regret_cap: float = math.log(10.0),
    index: _TimingIndex | None = None,
) -> MenuMetrics:
    weighted_logs = 0.0
    total_weight = 0.0
    regrets: list[float] = []
    covered = 0
    for shape in shapes.values():
        best = global_best.get(shape.shape_id)
        if best is None:
            continue
        total_weight += shape.weight
        selected_value = _shape_time(selected, shape, candidates, timings, index)
        if selected_value is None:
            log_regret = regret_cap
        else:
            covered += 1
            regret = max(1.0, selected_value[0] / best)
            regrets.append(regret)
            log_regret = min(regret_cap, math.log(regret))
        weighted_logs += shape.weight * log_regret
    resolved_deployments = (
        tuple(deployments)
        if deployments is not None
        else deployment_buckets_from_shapes(shapes.values())
    )
    autotune = autotune_cost(
        selected,
        candidates=candidates,
        deployments=resolved_deployments,
        cost=cost,
    )
    ordered = sorted(regrets)
    total_shapes = sum(shape.shape_id in global_best for shape in shapes.values())
    uncovered = covered < total_shapes
    capped_regret = math.exp(regret_cap)
    return MenuMetrics(
        objective=weighted_logs / total_weight if total_weight else 0.0,
        geomean_regret=(
            math.exp(weighted_logs / total_weight) if total_weight else 1.0
        ),
        p95_regret=(
            capped_regret
            if uncovered
            else ordered[math.ceil(0.95 * len(ordered)) - 1]
            if ordered
            else capped_regret
        ),
        worst_regret=(
            capped_regret if uncovered else max(ordered, default=capped_regret)
        ),
        covered_shapes=covered,
        total_shapes=total_shapes,
        runtime_kernels=autotune.runtime_kernels,
        deployment_kernels_mean=autotune.deployment_kernels_mean,
        deployment_pairs_mean=autotune.deployment_pairs_mean,
        deployment_pairs_max=autotune.deployment_pairs_max,
        deployment_seconds_mean=autotune.deployment_seconds_mean,
        deployment_seconds_max=autotune.deployment_seconds_max,
    )


def _improves_beyond_noise(
    addition: set[str],
    selected: set[str],
    *,
    candidates: Mapping[str, JointCandidate],
    shapes: Mapping[str, JointShape],
    timings: Mapping[str, Mapping[str, TimingValue]],
    index: _TimingIndex | None = None,
    current_times: Mapping[str, tuple[float, float] | None] | None = None,
) -> bool:
    trial = selected | addition
    for shape in shapes.values():
        current = (
            current_times[shape.shape_id]
            if current_times is not None
            else _shape_time(selected, shape, candidates, timings, index)
        )
        proposed = _shape_time(trial, shape, candidates, timings, index)
        if proposed is None:
            continue
        if current is None:
            return True
        combined_noise = math.hypot(current[1], proposed[1])
        if proposed[0] + combined_noise < current[0]:
            return True
    return False


def _floor_moves(
    candidates: Mapping[str, JointCandidate],
    shapes: Mapping[str, JointShape],
    timings: Mapping[str, Mapping[str, TimingValue]],
    index: _TimingIndex | None = None,
) -> list[tuple[str, set[str]]]:
    requirements = sorted(
        {
            (shape.activation_class, min(shape.selected_tile_ns))
            for shape in shapes.values()
        }
    )
    moves: list[tuple[str, set[str]]] = []
    selected: set[str] = set()
    for class_name, tile_n in requirements:
        relevant = {
            shape_id: shape
            for shape_id, shape in shapes.items()
            if shape.activation_class == class_name
            and min(shape.selected_tile_ns) == tile_n
        }
        fc1 = [
            candidate_id
            for candidate_id, candidate in candidates.items()
            if candidate.fc == "fc1"
            and candidate.activation_class == class_name
            and candidate.tile_n == tile_n
        ]
        fc2 = [
            candidate_id
            for candidate_id, candidate in candidates.items()
            if candidate.fc == "fc2" and candidate.tile_n == tile_n
        ]
        best_move: set[str] | None = None
        best_objective = math.inf
        best_all = global_best_times(candidates, relevant, timings, index)
        if not best_all:
            # No kernel ran on any shape of this group, so no menu can cover
            # it; the report counts these shapes as uncovered.
            continue
        for left in fc1:
            for right in fc2:
                trial = selected | {left, right}
                metrics = menu_metrics(
                    trial,
                    candidates=candidates,
                    shapes=relevant,
                    timings=timings,
                    global_best=best_all,
                    cost=AutotuneCost(),
                    index=index,
                )
                if metrics.covered_shapes == 0:
                    continue
                if metrics.objective < best_objective:
                    best_objective = metrics.objective
                    best_move = {left, right}
        if best_move is None:
            raise ValueError(
                f"no measured floor pair for class={class_name}, tile_n={tile_n}"
            )
        additions = best_move - selected
        if additions:
            moves.append((f"floor:{class_name}:{tile_n}", additions))
            selected.update(additions)
    return moves


def _extra_tile_floor_moves(
    candidates: Mapping[str, JointCandidate],
    shapes: Mapping[str, JointShape],
    timings: Mapping[str, Mapping[str, TimingValue]],
    global_best: Mapping[str, float],
    tile_ns: Sequence[int],
    plausible_ratio: float,
    *,
    selected: set[str],
    index: _TimingIndex | None = None,
) -> list[tuple[str, set[str]]]:
    """Choose isolated FC1/FC2 floor pairs for requested extra tiles."""

    moves: list[tuple[str, set[str]]] = []
    selected = set(selected)
    classes = sorted({shape.activation_class for shape in shapes.values()})
    for class_name in classes:
        for tile_n in tile_ns:
            has_fc1 = any(
                candidates[candidate_id].fc == "fc1"
                and candidates[candidate_id].activation_class == class_name
                and candidates[candidate_id].tile_n == tile_n
                for candidate_id in selected
            )
            has_fc2 = any(
                candidates[candidate_id].fc == "fc2"
                and candidates[candidate_id].tile_n == tile_n
                for candidate_id in selected
            )
            if has_fc1 and has_fc2:
                continue

            eligible = {
                shape_id: shape
                for shape_id, shape in shapes.items()
                if shape.activation_class == class_name
                and tile_n in shape.selected_tile_ns
                and shape_id in global_best
            }
            if not eligible:
                continue
            fc1 = sorted(
                candidate_id
                for candidate_id, candidate in candidates.items()
                if candidate.fc == "fc1"
                and candidate.activation_class == class_name
                and candidate.tile_n == tile_n
            )
            fc2 = sorted(
                candidate_id
                for candidate_id, candidate in candidates.items()
                if candidate.fc == "fc2" and candidate.tile_n == tile_n
            )
            measured_any = any(
                shape_id in timings.get(candidate_id, {})
                and timings[candidate_id][shape_id].is_measured
                for candidate_id in (*fc1, *fc2)
                for shape_id in eligible
            )
            if not measured_any:
                continue

            tile_candidates = set(fc1) | set(fc2)
            tile_best = {
                shape_id: value[0]
                for shape_id, shape in eligible.items()
                if (
                    value := _shape_time(
                        tile_candidates, shape, candidates, timings, index
                    )
                )
                is not None
            }
            if not tile_best:
                raise ValueError(
                    f"no measured floor pair for class={class_name}, tile_n={tile_n}"
                )
            plausible = {
                shape_id: shape
                for shape_id, shape in eligible.items()
                if shape_id in tile_best
                and tile_best[shape_id] <= plausible_ratio * global_best[shape_id]
            }
            relevant = plausible or eligible

            best_move: tuple[str, str] | None = None
            best_key: tuple[float, str, str] | None = None
            for left in fc1:
                for right in fc2:
                    metrics = menu_metrics(
                        {left, right},
                        candidates=candidates,
                        shapes=relevant,
                        timings=timings,
                        global_best=global_best,
                        cost=AutotuneCost(),
                        index=index,
                    )
                    if metrics.covered_shapes == 0:
                        continue
                    key = (metrics.objective, left, right)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_move = (left, right)
            if best_move is None:
                raise ValueError(
                    f"no measured floor pair for class={class_name}, tile_n={tile_n}"
                )
            additions = set(best_move) - selected
            if additions:
                moves.append((f"floor:{class_name}:{tile_n}", additions))
                selected.update(additions)
    return moves


def _curve_for_order(
    order: Sequence[str],
    *,
    candidates: Mapping[str, JointCandidate],
    shapes: Mapping[str, JointShape],
    timings: Mapping[str, Mapping[str, TimingValue]],
    global_best: Mapping[str, float],
    cost: AutotuneCost,
    deployments: Sequence[DeploymentBuckets],
    index: _TimingIndex | None = None,
) -> list[dict[str, object]]:
    selected: set[str] = set()
    result = []
    for candidate_id in order:
        selected.add(candidate_id)
        result.append(
            {
                "n": len(selected),
                **menu_metrics(
                    selected,
                    candidates=candidates,
                    shapes=shapes,
                    timings=timings,
                    global_best=global_best,
                    cost=cost,
                    deployments=deployments,
                    index=index,
                ).__dict__,
            }
        )
    return result


def select_joint_menu(
    candidates: Mapping[str, JointCandidate],
    shapes: Mapping[str, JointShape],
    timings: Mapping[str, Mapping[str, TimingValue]],
    *,
    cost: AutotuneCost | None = None,
    deployments: Sequence[DeploymentBuckets] | None = None,
    order_by: str = "cost",
    compute_lomo: bool = True,
    floor_tile_ns: Sequence[int] = (),
    floor_plausible_ratio: float = 1.25,
) -> JointSelection:
    """Build a nested, floor-preserving, noise-gated joint menu."""

    if cost is None:
        cost = AutotuneCost()
    if order_by not in ("cost", "gain"):
        raise ValueError("order_by must be 'cost' or 'gain'")
    resolved_floor_tile_ns = tuple(sorted(set(int(value) for value in floor_tile_ns)))
    if any(value <= 0 for value in resolved_floor_tile_ns):
        raise ValueError("floor_tile_ns values must be positive")
    if floor_plausible_ratio < 1.0:
        raise ValueError("floor_plausible_ratio must be at least 1")
    resolved_deployments = (
        tuple(deployments)
        if deployments is not None
        else deployment_buckets_from_shapes(shapes.values())
    )
    index = _TimingIndex(candidates, shapes, timings)
    best = global_best_times(candidates, shapes, timings, index)
    selected: set[str] = set()
    entries: list[OrderedEntry] = []
    previous = menu_metrics(
        selected,
        candidates=candidates,
        shapes=shapes,
        timings=timings,
        global_best=best,
        cost=cost,
        deployments=resolved_deployments,
        index=index,
    )

    def add(candidate_id: str, reason: str, group_gain: float = 0.0) -> None:
        nonlocal previous
        if candidate_id in selected:
            return
        selected.add(candidate_id)
        current = menu_metrics(
            selected,
            candidates=candidates,
            shapes=shapes,
            timings=timings,
            global_best=best,
            cost=cost,
            deployments=resolved_deployments,
            index=index,
        )
        entries.append(
            OrderedEntry(
                rank=len(entries) + 1,
                candidate_id=candidate_id,
                reason=reason,
                marginal_gain=max(group_gain, previous.objective - current.objective),
                delta_kernels=(
                    current.deployment_kernels_mean - previous.deployment_kernels_mean
                ),
                delta_pairs=(
                    current.deployment_pairs_mean - previous.deployment_pairs_mean
                ),
                metrics=current,
            )
        )
        previous = current

    floor_moves = _floor_moves(candidates, shapes, timings, index)
    floor_selected = {
        candidate_id for _reason, move in floor_moves for candidate_id in move
    }
    floor_moves.extend(
        _extra_tile_floor_moves(
            candidates,
            shapes,
            timings,
            best,
            resolved_floor_tile_ns,
            floor_plausible_ratio,
            selected=floor_selected,
            index=index,
        )
    )
    for reason, move in floor_moves:
        before = previous.objective
        # Add FC1 first deterministically; the paired FC2 follows immediately.
        ordered_move = sorted(move, key=lambda value: (candidates[value].fc, value))
        for candidate_id in ordered_move:
            add(candidate_id, reason)
        group_gain = before - previous.objective
        if entries:
            last = entries[-1]
            entries[-1] = OrderedEntry(
                rank=last.rank,
                candidate_id=last.candidate_id,
                reason=last.reason,
                marginal_gain=group_gain,
                delta_kernels=last.delta_kernels,
                delta_pairs=last.delta_pairs,
                metrics=last.metrics,
            )
    floor_size = len(entries)

    while True:
        winner: tuple[str, ...] | None = None
        winner_metrics: MenuMetrics | None = None
        winner_key: tuple[float, float, str] | None = None
        moves: set[tuple[str, ...]] = {
            (candidate_id,)
            for candidate_id in candidates
            if candidate_id not in selected
        }
        # A previously unused tile cannot improve the objective one side at a
        # time. Include two-entry moves so greedy can open that tile.
        for shape in shapes.values():
            for tile_n in shape.selected_tile_ns:
                has_fc1 = any(
                    candidates[candidate_id].fc == "fc1"
                    and candidates[candidate_id].activation_class
                    == shape.activation_class
                    and candidates[candidate_id].tile_n == tile_n
                    for candidate_id in selected
                )
                has_fc2 = any(
                    candidates[candidate_id].fc == "fc2"
                    and candidates[candidate_id].tile_n == tile_n
                    for candidate_id in selected
                )
                if has_fc1 or has_fc2:
                    continue
                fc1_ids = [
                    candidate_id
                    for candidate_id, candidate in candidates.items()
                    if candidate_id not in selected
                    and candidate.fc == "fc1"
                    and candidate.activation_class == shape.activation_class
                    and candidate.tile_n == tile_n
                ]
                fc2_ids = [
                    candidate_id
                    for candidate_id, candidate in candidates.items()
                    if candidate_id not in selected
                    and candidate.fc == "fc2"
                    and candidate.tile_n == tile_n
                ]
                moves.update(
                    tuple(sorted((left, right)))
                    for left in fc1_ids
                    for right in fc2_ids
                )
        current_times = {
            shape_id: index.shape_time(selected, shape_id) for shape_id in shapes
        }
        for move in sorted(moves):
            addition = set(move)
            if not _improves_beyond_noise(
                addition,
                selected,
                candidates=candidates,
                shapes=shapes,
                timings=timings,
                index=index,
                current_times=current_times,
            ):
                continue
            current = menu_metrics(
                selected | addition,
                candidates=candidates,
                shapes=shapes,
                timings=timings,
                global_best=best,
                cost=cost,
                deployments=resolved_deployments,
                index=index,
            )
            gain = previous.objective - current.objective
            if gain <= 1e-15:
                continue
            delta_seconds = (
                current.deployment_seconds_mean - previous.deployment_seconds_mean
            )
            delta_raw = (
                current.deployment_pairs_mean
                - previous.deployment_pairs_mean
                + current.deployment_kernels_mean
                - previous.deployment_kernels_mean
            )
            denominator = delta_seconds if delta_seconds > 0 else max(1, delta_raw)
            score = gain if order_by == "gain" else gain / denominator
            move_key = ",".join(move)
            key = (score, -denominator, move_key)
            if winner_key is None or key > winner_key:
                winner = move
                winner_metrics = current
                winner_key = key
        if winner is None or winner_metrics is None:
            break
        before = previous.objective
        reason = "greedy_pair" if len(winner) > 1 else "greedy"
        for candidate_id in sorted(
            winner, key=lambda value: (candidates[value].fc, value)
        ):
            add(candidate_id, reason)
        group_gain = before - previous.objective
        last = entries[-1]
        entries[-1] = OrderedEntry(
            rank=last.rank,
            candidate_id=last.candidate_id,
            reason=last.reason,
            marginal_gain=group_gain,
            delta_kernels=last.delta_kernels,
            delta_pairs=last.delta_pairs,
            metrics=last.metrics,
        )

    lomo: dict[str, Mapping[str, object]] = {}

    def lomo_group(shape: JointShape) -> str:
        return shape.model or shape.deployment

    models = sorted({lomo_group(shape) for shape in shapes.values()})
    if compute_lomo and len(models) >= 2:
        for model in models:
            training = {
                shape_id: shape
                for shape_id, shape in shapes.items()
                if lomo_group(shape) != model
            }
            held_out = {
                shape_id: shape
                for shape_id, shape in shapes.items()
                if lomo_group(shape) == model
            }
            training_deployments = tuple(
                deployment
                for deployment in resolved_deployments
                if (deployment.model or deployment.deployment) != model
            )
            held_out_deployments = tuple(
                deployment
                for deployment in resolved_deployments
                if (deployment.model or deployment.deployment) == model
            )
            trained = select_joint_menu(
                candidates,
                training,
                timings,
                cost=cost,
                deployments=training_deployments,
                order_by=order_by,
                compute_lomo=False,
                floor_tile_ns=resolved_floor_tile_ns,
                floor_plausible_ratio=floor_plausible_ratio,
            )
            lomo[model] = {
                "order": list(trained.selected_ids),
                "held_out_curve": _curve_for_order(
                    trained.selected_ids,
                    candidates=candidates,
                    shapes=held_out,
                    timings=timings,
                    global_best=best,
                    cost=cost,
                    deployments=held_out_deployments,
                    index=index,
                ),
            }
    return JointSelection(
        candidates=candidates,
        shapes=shapes,
        deployments=resolved_deployments,
        timings=timings,
        global_best=best,
        order=tuple(entries),
        floor_size=floor_size,
        floor_tile_ns=resolved_floor_tile_ns,
        floor_plausible_ratio=floor_plausible_ratio,
        leave_one_model_out=lomo,
    )
