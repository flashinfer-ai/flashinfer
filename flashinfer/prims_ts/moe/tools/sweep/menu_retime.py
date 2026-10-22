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

"""Candidate collection and execution support for MoE menu re-timing."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .db import JSONLWriter, atomic_write_json, deterministic_id, read_jsonl
from .engine import (
    SweepEngine,
    SweepOptions,
    _candidate_from_payload,
    code_fingerprint,
    context_for_shapes,
    shape_id,
)
from .measure import TimingProtocol, final_timing_protocol
from .menu_config import (
    DeploymentTarget,
    MenuTarget,
    activation_class,
    build_deployment_shapes,
    build_menu_shapes,
)
from .shapes import MODEL_PRESETS, normalize_dtype
from .space import Candidate, enumerate_incumbents

_ACTIVATION_FIELDS = frozenset(("fused_act", "act", "eltwise_act_type"))
_TRANSIENT_STATUSES = frozenset(("timeout", "worker_error", "oom"))
_QUANT_FIELDS = (
    "dtype_a",
    "dtype_b",
    "dtype_c",
    "use_deepseek_fp8",
    "weight_layout",
    "use_per_token_sf_b",
)


@dataclass
class MenuCandidate:
    """One activation-normalized FC candidate with accumulated provenance."""

    candidate: Candidate
    fc: str
    activation_class: str | None
    provenance: set[str] = field(default_factory=set)
    shipped_incumbent: bool = False

    @property
    def menu_id(self) -> str:
        return menu_candidate_id(
            self.candidate,
            fc=self.fc,
            activation_class_name=self.activation_class,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.menu_id,
            "fc": self.fc,
            "activation_class": self.activation_class,
            "tile_n": self.candidate.tile_n,
            "options": dict(self.candidate.options),
            "provenance": sorted(self.provenance),
            "shipped_incumbent": self.shipped_incumbent,
        }


def _is_transient_failure(record: Mapping[str, object]) -> bool:
    status = record.get("status")
    if status in _TRANSIENT_STATUSES:
        return True
    return status == "error" and "outofmemoryerror" in json.dumps(record).lower()


def normalize_activation_options(options: Mapping[str, object]) -> dict[str, object]:
    """Remove runtime activation spelling from a geometry candidate."""

    return {
        key: value for key, value in options.items() if key not in _ACTIVATION_FIELDS
    }


def menu_candidate_id(
    candidate: Candidate,
    *,
    fc: str,
    activation_class_name: str | None,
) -> str:
    return deterministic_id(
        {
            "fc": fc,
            "activation_class": activation_class_name if fc == "fc1" else None,
            "tile_n": candidate.tile_n,
            "options": normalize_activation_options(candidate.options),
        }
    )


def _activation_classes_from_options(
    options: Mapping[str, object], *, default: str | None
) -> tuple[str, ...]:
    if bool(options.get("fused_act", False)):
        act = str(options.get("act", "")).lower()
        return ("gated",) if act in ("swiglu", "geglu", "situ") else ()
    eltwise = str(options.get("eltwise_act_type", "none")).lower()
    if eltwise not in ("", "none", "false", "0"):
        return ("elementwise",)
    return () if default is None else (default,)


def _candidate(
    options: Mapping[str, object],
    *,
    move_kind: str,
    source_comment: str = "",
) -> Candidate:
    normalized = normalize_activation_options(options)
    tile_n = int(normalized["tile_n"])
    return Candidate(
        options=normalized,
        tile_n=tile_n,
        source_comment=source_comment,
        move_kind=move_kind,
    )


def _add_candidate(
    result: dict[str, MenuCandidate],
    candidate: Candidate,
    *,
    fc: str,
    classes: Iterable[str | None],
    provenance: str,
    incumbent: bool = False,
) -> None:
    for class_name in classes:
        item = MenuCandidate(
            candidate=replace(
                candidate, options=normalize_activation_options(candidate.options)
            ),
            fc=fc,
            activation_class=class_name if fc == "fc1" else None,
            provenance={provenance},
            shipped_incumbent=incumbent,
        )
        existing = result.get(item.menu_id)
        if existing is None:
            result[item.menu_id] = item
        else:
            existing.provenance.add(provenance)
            existing.shipped_incumbent |= incumbent


def _manifest_target(
    run_dir: Path,
) -> tuple[str, str | None, str, tuple[object, ...]]:
    with (run_dir / "run_manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)
    target = manifest.get("target", manifest)
    fc = str(target.get("fc", ""))
    if fc not in ("fc1", "fc2"):
        raise ValueError(f"{run_dir}: manifest does not identify fc1 or fc2")
    dtype = target.get("dtype")
    if dtype is None:
        model = target.get("model")
        preset = MODEL_PRESETS.get(str(model))
        dtype = None if preset is None else preset.dtype
    launch_keys = target.get("shape_launch_keys", ())
    if dtype is None and launch_keys:
        dtype = launch_keys[0].get("dtype")
    if dtype is None:
        raise ValueError(f"{run_dir}: manifest does not identify a dtype")
    context = target.get("context", {})
    quant_key = tuple(context.get(field) for field in _QUANT_FIELDS)
    if fc == "fc2":
        return fc, None, normalize_dtype(str(dtype)), quant_key
    activation = target.get("activation")
    if activation is None:
        model = target.get("model")
        preset = MODEL_PRESETS.get(str(model))
        activation = None if preset is None else preset.activation
    if activation is None and launch_keys:
        activation = launch_keys[0].get("activation")
    if activation is None:
        raise ValueError(f"{run_dir}: FC1 manifest does not identify activation")
    return (
        fc,
        activation_class(str(activation)),
        normalize_dtype(str(dtype)),
        quant_key,
    )


def collect_run_candidates(
    run_dirs: Iterable[str | Path],
    *,
    candidate_source: str = "selected",
    expected_dtype: str | None = None,
    expected_quant_keys: Mapping[str, tuple[object, ...]] | None = None,
) -> dict[str, MenuCandidate]:
    """Collect and exactly deduplicate selected menus or final re-time pools."""

    if candidate_source not in ("selected", "pool"):
        raise ValueError("candidate_source must be 'selected' or 'pool'")
    result: dict[str, MenuCandidate] = {}
    quant_keys: dict[str, tuple[object, ...]] = {}
    for raw_dir in run_dirs:
        run_dir = Path(raw_dir)
        fc, default_class, dtype, quant_key = _manifest_target(run_dir)
        if expected_dtype is not None and dtype != normalize_dtype(expected_dtype):
            raise ValueError(
                f"{run_dir}: dtype={dtype} conflicts with target "
                f"dtype={normalize_dtype(expected_dtype)}"
            )
        if expected_quant_keys is not None and quant_key != expected_quant_keys[fc]:
            raise ValueError(
                f"{run_dir}: quantization context conflicts with target {fc}: "
                f"{quant_key} != {expected_quant_keys[fc]}"
            )
        prior_quant_key = quant_keys.setdefault(fc, quant_key)
        if quant_key != prior_quant_key:
            raise ValueError(
                f"{run_dir}: quantization context conflicts with another {fc} run: "
                f"{quant_key} != {prior_quant_key}"
            )
        candidates: list[Candidate] = []
        if candidate_source == "selected":
            with (run_dir / "summary.json").open(encoding="utf-8") as file:
                summary = json.load(file)
            for rows in summary.get("selected_menu", {}).values():
                for row in rows:
                    candidates.append(
                        _candidate(
                            row,
                            move_kind="sweep",
                            source_comment=f"selected:{run_dir}",
                        )
                    )
        else:
            latest: dict[str, Mapping[str, object]] = {}
            for record in read_jsonl(run_dir / "results.jsonl"):
                if record.get("phase") == "retime":
                    latest[str(record.get("candidate_id"))] = record
            candidates.extend(
                _candidate_from_payload(record["candidate"])
                for record in latest.values()
                if record.get("status") == "ok"
                and isinstance(record.get("candidate"), Mapping)
            )

        for candidate in candidates:
            classes: tuple[str | None, ...]
            if fc == "fc1":
                found = _activation_classes_from_options(
                    candidate.options, default=default_class
                )
                classes = found or ((default_class,) if default_class else ())
            else:
                classes = (None,)
            _add_candidate(
                result,
                candidate,
                fc=fc,
                classes=classes,
                provenance=f"{candidate_source}:{run_dir}",
            )
    return result


def collect_incumbents(target: MenuTarget) -> dict[str, MenuCandidate]:
    """Collect shipped candidates applicable to any configured deployment."""

    result: dict[str, MenuCandidate] = {}
    tile_ns = {
        tile_n
        for shape in build_menu_shapes(target)
        for tile_n in shape.selected_tile_ns
    }
    for deployment in target.deployments:
        class_name = activation_class(deployment.activation)
        for fc in ("fc1", "fc2"):
            shapes = list(build_deployment_shapes(target, deployment, fc=fc))
            context = context_for_shapes(
                shapes,
                fc=fc,
                sm_arch=target.sm_arch,
                has_bias=deployment.has_bias,
                gemm1_alpha=deployment.gemm1_alpha,
                gemm1_beta=deployment.gemm1_beta,
                gemm1_clamp_limit=deployment.gemm1_clamp_limit,
            )
            for candidate in enumerate_incumbents(context, tile_ns):
                _add_candidate(
                    result,
                    candidate,
                    fc=fc,
                    classes=(class_name if fc == "fc1" else None,),
                    provenance="shipped",
                    incumbent=True,
                )
    return result


def collect_menu_candidates(
    target: MenuTarget,
    run_dirs: Sequence[str | Path],
    *,
    candidate_source: str = "selected",
) -> tuple[MenuCandidate, ...]:
    """Return the union of shipped and sweep candidates for one quant mode."""

    combined = collect_incumbents(target)
    target_contexts = _target_quant_contexts(target)
    expected_quant_keys: dict[str, tuple[object, ...]] = {}
    for fc in ("fc1", "fc2"):
        keys = {
            tuple(context[field] for field in _QUANT_FIELDS)
            for name, context in target_contexts.items()
            if name.endswith(f"/{fc}")
        }
        if len(keys) != 1:
            raise ValueError(
                f"target deployments have conflicting quantization contexts for {fc}"
            )
        expected_quant_keys[fc] = keys.pop()
    for candidate_id, candidate in collect_run_candidates(
        run_dirs,
        candidate_source=candidate_source,
        expected_dtype=target.dtype,
        expected_quant_keys=expected_quant_keys,
    ).items():
        existing = combined.get(candidate_id)
        if existing is None:
            combined[candidate_id] = candidate
        else:
            existing.provenance.update(candidate.provenance)
    return tuple(combined[key] for key in sorted(combined))


def _deployment_shapes(target: MenuTarget, deployment: DeploymentTarget, *, fc: str):
    return build_deployment_shapes(target, deployment, fc=fc)


class RetimeEngine(SweepEngine):
    """A SweepEngine specialization that only re-times an explicit pool."""

    def __init__(
        self,
        *,
        target: MenuTarget,
        deployment: DeploymentTarget,
        fc: str,
        candidates: Sequence[MenuCandidate],
        out_dir: Path,
        worker_count: int,
        gpu: str,
        resume: bool,
        calibration=None,
        budget_minutes: float = 1440.0,
    ) -> None:
        points = deployment.parallelism
        all_tokens = tuple(
            sorted({token for point in points for token in point.tokens})
        )
        options = SweepOptions(
            out_dir=out_dir,
            fc=fc,
            model=deployment.model,
            sm_arch=target.sm_arch,
            dtype=None if deployment.model is not None else deployment.dtype,
            hidden_size=(
                None if deployment.model is not None else deployment.hidden_size
            ),
            intermediate_size=(
                None if deployment.model is not None else deployment.intermediate_size
            ),
            num_experts=(
                None if deployment.model is not None else deployment.num_experts
            ),
            top_k=None if deployment.model is not None else deployment.top_k,
            activation=(
                None if deployment.model is not None else deployment.activation
            ),
            tp=tuple(sorted({point.tp for point in points})),
            ep=tuple(sorted({point.ep for point in points})),
            tp_ep=tuple((point.tp, point.ep) for point in points),
            tokens=all_tokens,
            final_tokens=all_tokens,
            budget_minutes=budget_minutes,
            tail_minutes=0.0,
            menu_size=1,
            worker_count=worker_count,
            gpu=gpu,
            has_bias=deployment.has_bias,
            gemm1_alpha=deployment.gemm1_alpha,
            gemm1_beta=deployment.gemm1_beta,
            gemm1_clamp_limit=deployment.gemm1_clamp_limit,
            search_routing=target.routings[0],
            retime_routings=target.routings,
            selection_routings=target.routings,
            routing_seed=target.routing_seed,
            resume=resume,
        )
        super().__init__(options, calibration=calibration)
        explicit_shapes = _deployment_shapes(target, deployment, fc=fc)
        self.shapes = explicit_shapes
        self.final_shapes = explicit_shapes
        self.context = context_for_shapes(
            explicit_shapes,
            fc=fc,
            sm_arch=target.sm_arch,
            has_bias=deployment.has_bias,
            gemm1_alpha=deployment.gemm1_alpha,
            gemm1_beta=deployment.gemm1_beta,
            gemm1_clamp_limit=deployment.gemm1_clamp_limit,
        )
        class_name = activation_class(deployment.activation)
        selected = [
            item
            for item in candidates
            if item.fc == fc and (fc == "fc2" or item.activation_class == class_name)
        ]
        explicit = {item.candidate.candidate_id: item.candidate for item in selected}
        # Preserve candidates recovered from a prior journal, then add this
        # invocation's explicit pool.
        self.candidates.update(explicit)
        self.incumbent_ids = {
            item.candidate.candidate_id for item in selected if item.shipped_incumbent
        }
        self.menu_candidates = {item.candidate.candidate_id: item for item in selected}
        self.menu_label = f"{deployment.name}/{fc}"
        self._shape_labels = {
            shape_id(shape, fc): (
                f"{deployment.name} TP/EP={shape.tp}/{shape.ep} "
                f"T={shape.global_num_tokens} R={shape.routing}"
            )
            for shape in explicit_shapes
        }
        atomic_write_json(
            out_dir / "shape_index.json",
            {
                shape_id(shape, fc): {
                    **asdict(shape),
                    "deployment": deployment.name,
                    "activation_class": class_name,
                }
                for shape in explicit_shapes
            },
        )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        menu_identity = {
            "deployment": deployment.name,
            "activation_class": class_name,
            "candidate_ids": sorted(item.menu_id for item in selected),
            "context": asdict(self.context),
            "shape_launch_keys": [
                asdict(shape.launch_key(fc=fc)) for shape in explicit_shapes
            ],
        }
        menu_identity = json.loads(json.dumps(menu_identity))
        prior_identity = manifest.get("menu_retime")
        if resume and prior_identity is not None and prior_identity != menu_identity:
            raise ValueError(f"{out_dir}: resume candidate pool or deployment changed")
        manifest["menu_retime"] = menu_identity
        atomic_write_json(self.manifest_path, manifest)

    def _plan_reference_checks(
        self, writer: JSONLWriter, candidates: Sequence[Candidate]
    ) -> dict[str, list]:
        """Choose reference shapes, journaling candidates this FC cannot run.

        The pool spans models, so a candidate valid in its source sweep can fail
        host-side materialization here (another K, bias, or SMEM regime).
        """

        checks: dict[str, list] = {}
        for candidate in candidates:
            try:
                shapes = self._reference_shapes_for_candidate(candidate)
                error = None if shapes else "no applicable shape"
            except Exception as exc:
                shapes, error = [], f"{type(exc).__name__}: {exc}"
            if error is None:
                checks[candidate.candidate_id] = shapes
            else:
                self._record(
                    writer,
                    candidate,
                    {"status": "not_applicable", "error": error, "checks": []},
                    phase="menu_reference",
                )
        return checks

    def run_retime(self) -> None:
        protocol = final_timing_protocol(
            TimingProtocol(
                warmup_iters=self.options.warmup_iters,
                repeat_iters=self.options.repeat_iters,
                timing_backend=self.timing_backend,
            )
        )
        candidates = list(self.menu_candidates)
        with JSONLWriter(self.journal_path, fsync=True) as writer:
            reference_status = {
                str(record.get("candidate_id")): str(record.get("status"))
                for record in self.records
                if record.get("phase") == "menu_reference"
            }
            pending_reference = [
                self.candidates[candidate_id]
                for candidate_id in candidates
                if reference_status.get(candidate_id)
                not in ("ok", "failed", "not_applicable")
            ]
            checks = self._plan_reference_checks(writer, pending_reference)
            if pending_reference:
                self._progress(
                    f"{self.menu_label} reference: checking {len(checks)} "
                    f"candidate(s); {len(pending_reference) - len(checks)} "
                    "not applicable",
                    force=True,
                )
            if checks:
                self._reference_batch(
                    writer,
                    [
                        candidate
                        for candidate in pending_reference
                        if candidate.candidate_id in checks
                    ],
                    protocol=protocol,
                    phase="menu_reference",
                    shapes_by_candidate=checks,
                )
            reference_status = {
                str(record.get("candidate_id")): str(record.get("status"))
                for record in self.records
                if record.get("phase") == "menu_reference"
            }
            latest_timing: dict[str, Mapping[str, object]] = {}
            for record in self.records:
                if record.get("phase") == "menu_retime":
                    latest_timing[str(record.get("candidate_id"))] = record
            # Budget expiry, worker crashes and OOM say nothing about the
            # candidate itself, so a resumed run must time those again.
            timed = {
                candidate_id
                for candidate_id, record in latest_timing.items()
                if not _is_transient_failure(record)
            }
            pending = [
                self.candidates[candidate_id]
                for candidate_id in candidates
                if reference_status.get(candidate_id) == "ok"
                and candidate_id not in timed
            ]
            if pending:
                self._evaluate_batch(
                    writer,
                    pending,
                    protocol=protocol,
                    phase="menu_retime",
                    progress_label=f"{self.menu_label} re-time",
                )


def _matrix_rows(
    run_dir: Path,
    *,
    fc: str,
    class_name: str | None,
) -> list[dict[str, object]]:
    shape_index = json.loads((run_dir / "shape_index.json").read_text(encoding="utf-8"))
    deployment_name = next(
        (
            str(shape.get("deployment"))
            for shape in shape_index.values()
            if shape.get("deployment") is not None
        ),
        run_dir.parent.name,
    )
    rows: list[dict[str, object]] = []
    latest: dict[str, Mapping[str, object]] = {}
    reference_status: dict[str, str] = {}
    for record in read_jsonl(run_dir / "results.jsonl"):
        if record.get("phase") == "menu_retime":
            latest[str(record.get("candidate_id"))] = record
        elif record.get("phase") == "menu_reference":
            reference_status[str(record.get("candidate_id"))] = str(
                record.get("status")
            )
    for record in latest.values():
        payload = record.get("candidate")
        if not isinstance(payload, Mapping):
            continue
        candidate = _candidate_from_payload(payload)
        candidate_id = menu_candidate_id(
            candidate, fc=fc, activation_class_name=class_name
        )
        measurements = list(record.get("measurements", ()))
        if not measurements and record.get("status") != "ok":
            rows.append(
                {
                    "record_type": "candidate_context",
                    "candidate_id": candidate_id,
                    "fc": fc,
                    "activation_class": class_name,
                    "deployment": deployment_name,
                    "status": (
                        "not_applicable"
                        if record.get("status") == "validation_error"
                        else record.get("status")
                    ),
                    "error": record.get("error"),
                }
            )
        for measurement in measurements:
            shape_key = str(measurement.get("shape_id"))
            shape = shape_index.get(shape_key, {})
            status = measurement.get("status")
            error = measurement.get("error")
            if status == "ok" and reference_status.get(candidate.candidate_id) != "ok":
                status = "reference_failed"
                error = (
                    f"reference status="
                    f"{reference_status.get(candidate.candidate_id, 'missing')}"
                )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "fc": fc,
                    "activation_class": class_name,
                    "tile_n": candidate.tile_n,
                    "shape_id": shape_key,
                    "deployment": shape.get("deployment"),
                    "model": shape.get("model"),
                    "tp": shape.get("tp"),
                    "ep": shape.get("ep"),
                    "tokens": shape.get("global_num_tokens"),
                    "routing": shape.get("routing"),
                    "weight": shape.get("weight", 1.0),
                    "selected_tile_ns": shape.get("selected_tile_ns", ()),
                    "status": status,
                    "median_us": measurement.get("median_us"),
                    "mad_us": measurement.get("mad_us"),
                    "error": error,
                }
            )
    return rows


def _target_quant_contexts(target: MenuTarget) -> dict[str, dict[str, object]]:
    contexts = {}
    for deployment in target.deployments:
        for fc in ("fc1", "fc2"):
            context = context_for_shapes(
                build_deployment_shapes(target, deployment, fc=fc),
                fc=fc,
                sm_arch=target.sm_arch,
                has_bias=deployment.has_bias,
                gemm1_alpha=deployment.gemm1_alpha,
                gemm1_beta=deployment.gemm1_beta,
                gemm1_clamp_limit=deployment.gemm1_clamp_limit,
            )
            contexts[f"{deployment.name}/{fc}"] = {
                field: getattr(context, field) for field in _QUANT_FIELDS
            }
    return contexts


def run_menu_retime(
    target: MenuTarget,
    candidates: Sequence[MenuCandidate],
    *,
    out_dir: str | Path,
    worker_count: int,
    gpu: str = "0",
    resume: bool = False,
    calibration=None,
    budget_minutes: float = 1440.0,
) -> Path:
    """Re-time a common candidate pool and write a flat selection matrix."""

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    menu_manifest_path = destination / "menu_manifest.json"
    menu_manifest = {
        "schema_version": 1,
        "created_at": time.time(),
        "target": json.loads(json.dumps(asdict(target))),
        "candidate_ids": sorted(item.menu_id for item in candidates),
        "candidates": {item.menu_id: item.to_dict() for item in candidates},
        "code": code_fingerprint(),
        "quant_contexts": _target_quant_contexts(target),
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
    if resume:
        if not menu_manifest_path.exists():
            raise FileNotFoundError(f"cannot resume without {menu_manifest_path.name}")
        prior = json.loads(menu_manifest_path.read_text(encoding="utf-8"))
        for field in ("target", "candidate_ids", "quant_contexts", "calibration"):
            if prior.get(field) != menu_manifest[field]:
                raise ValueError(f"menu resume {field} changed")
        # Like SweepEngine's non-strict resume, accept a code hotfix; each
        # sub-run journals later records in a new resume segment.
        if prior.get("code") != menu_manifest["code"]:
            print(
                "WARNING: resuming the menu re-time after a code change",
                file=sys.stderr,
            )
        menu_manifest["created_at"] = prior.get("created_at")
    atomic_write_json(menu_manifest_path, menu_manifest)
    matrix: list[dict[str, object]] = []
    for deployment in target.deployments:
        class_name = activation_class(deployment.activation)
        for fc in ("fc1", "fc2"):
            run_dir = destination / deployment.name / fc
            # An interrupted run leaves later sub-runs unstarted; those begin
            # fresh while started ones resume.
            engine = RetimeEngine(
                target=target,
                deployment=deployment,
                fc=fc,
                candidates=candidates,
                out_dir=run_dir,
                worker_count=worker_count,
                gpu=gpu,
                resume=resume and (run_dir / "run_manifest.json").exists(),
                calibration=calibration,
                budget_minutes=budget_minutes,
            )
            try:
                engine.run_retime()
            finally:
                engine.close()
            matrix.extend(
                _matrix_rows(
                    run_dir,
                    fc=fc,
                    class_name=class_name,
                )
            )
    matrix_path = destination / "matrix.jsonl"
    temporary_matrix = destination / ".matrix.jsonl.tmp"
    temporary_matrix.unlink(missing_ok=True)
    with JSONLWriter(temporary_matrix, recover=False) as writer:
        for row in matrix:
            writer.append(row)
    temporary_matrix.replace(matrix_path)
    return matrix_path
