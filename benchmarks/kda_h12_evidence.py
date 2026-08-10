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

"""Pure-Python schema and CUPTI reduction for the Phase-A H12 KDA harness.

This module intentionally has no torch, CUDA, or FlashInfer import.  The GPU
runner converts CUPTI records into the dataclasses below; CPU tests can then
audit preset identity, activity correlation, and report schema without loading
the CUDA runtime.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


FLASH_KDA_REPOSITORY = "https://github.com/MoonshotAI/FlashKDA.git"
FLASH_KDA_BASELINE_REVISION = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
FLASH_KDA_CUTLASS_REVISION = "5c149f52a436782210263fb2f19b354443a61c6a"
FLASHINFER_PHASE_A_UPSTREAM_MAIN_REVISION = "2ab910c58fdd2392914ea05e2a8714946ac0eef6"
FLASHINFER_H12_ROUTE_REVISION = "38bf507f9c9eba6b4544bee016d2bdf9c4fed02b"
PRESET_SCHEMA_VERSION = 1
SUPPORTED_ARCHITECTURES = {(10, 0): "sm100a", (10, 3): "sm103a"}

PUBLIC_TIMING_SCOPE = (
    "public_recurrent_kda_first_to_last_correlated_gpu_activity_"
    "including_beta_pack_and_recurrence"
)
PREPARED_TIMING_SCOPE = (
    "prepared_recurrence_kernel_activity_selected_from_the_same_public_call"
)
BETA_PACK_ACTIVITY_MARKER = "PackBetaForTmaKernel"
RECURRENCE_ACTIVITY_MARKER = "kernel_flashkda_bf16_fused_m128"


class EvidenceSchemaError(ValueError):
    """Raised when checked-in evidence input or activity shape is invalid."""


@dataclass(frozen=True)
class CasePreset:
    """One deterministic H12 evidence case."""

    name: str
    layout: str
    seq_lens: tuple[int, ...]
    seed: int

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)


@dataclass(frozen=True)
class EvidencePreset:
    """Strictly validated checked-in Phase-A preset."""

    name: str
    common: dict
    aggregation: str
    cases: tuple[CasePreset, ...]
    path: str
    sha256: str


@dataclass(frozen=True)
class CpuBracket:
    """CUPTI-clock host timestamps around one measured invocation."""

    start_ns: int
    submitted_ns: int
    synchronized_ns: int


@dataclass(frozen=True)
class LaunchActivity:
    """One CUPTI runtime or driver API activity."""

    start_ns: int
    end_ns: int
    correlation_id: int
    kind: str
    name: str


@dataclass(frozen=True)
class GpuActivity:
    """One correlated GPU kernel, memcpy, or memset activity."""

    start_ns: int
    end_ns: int
    correlation_id: int
    kind: str
    name: str


_EXPECTED_CASES = (
    CasePreset("h12_packed_512x32", "packed", (512,) * 32, 12000),
    CasePreset("h12_packed_128x8", "packed", (128,) * 8, 12001),
    CasePreset("h12_fixed_512", "fixed", (512,), 12002),
    CasePreset("h12_fixed_8192", "fixed", (8192,), 12003),
    CasePreset(
        "h12_packed_mixed",
        "packed",
        (1300, 547, 2048, 963, 271, 3063),
        12004,
    ),
    CasePreset("h12_packed_1024x8", "packed", (1024,) * 8, 12005),
)
_EXPECTED_COMMON = {
    "num_heads": 12,
    "head_dim_qk": 128,
    "head_dim_vo": 128,
    "dtype": "bfloat16",
    "initial_state": "provided",
    "use_qk_l2norm_in_kernel": True,
    "use_gate_in_kernel": True,
    "beta_is_logit": True,
    "lower_bound": -5.0,
}


def _require_exact_keys(mapping: dict, keys: set[str], label: str) -> None:
    actual = set(mapping)
    if actual != keys:
        raise EvidenceSchemaError(
            f"{label} keys must be exactly {sorted(keys)}, got {sorted(actual)}"
        )


def load_preset(path: Path) -> EvidencePreset:
    """Load and strictly validate the checked-in six-case H12 preset."""

    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise EvidenceSchemaError(f"invalid preset JSON at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise EvidenceSchemaError("preset root must be an object")
    _require_exact_keys(
        payload,
        {"schema_version", "name", "common", "aggregation", "cases"},
        "preset",
    )
    if payload["schema_version"] != PRESET_SCHEMA_VERSION:
        raise EvidenceSchemaError(
            "preset schema_version must be "
            f"{PRESET_SCHEMA_VERSION}, got {payload['schema_version']!r}"
        )
    if payload["name"] != "recurrent_kda_prefill_h12_phase_a":
        raise EvidenceSchemaError(f"unexpected preset name {payload['name']!r}")
    if payload["common"] != _EXPECTED_COMMON:
        raise EvidenceSchemaError(
            f"preset common contract must be {_EXPECTED_COMMON!r}, "
            f"got {payload['common']!r}"
        )
    if payload["aggregation"] != "per_case_only":
        raise EvidenceSchemaError(
            "Phase-A promotion evidence is per-case only; cross-shape aggregation "
            f"is forbidden, got {payload['aggregation']!r}"
        )
    if not isinstance(payload["cases"], list):
        raise EvidenceSchemaError("preset cases must be an array")

    cases = []
    for index, raw_case in enumerate(payload["cases"]):
        if not isinstance(raw_case, dict):
            raise EvidenceSchemaError(f"case {index} must be an object")
        _require_exact_keys(
            raw_case,
            {"name", "layout", "seq_lens", "seed"},
            f"case {index}",
        )
        if raw_case["layout"] not in {"fixed", "packed"}:
            raise EvidenceSchemaError(
                f"case {index} has invalid layout {raw_case['layout']!r}"
            )
        seq_lens = raw_case["seq_lens"]
        if (
            not isinstance(seq_lens, list)
            or not seq_lens
            or any(type(value) is not int or value <= 0 for value in seq_lens)
        ):
            raise EvidenceSchemaError(
                f"case {index} seq_lens must be a non-empty array of positive ints"
            )
        if raw_case["layout"] == "fixed" and len(seq_lens) != 1:
            raise EvidenceSchemaError(
                f"fixed case {raw_case['name']!r} must contain one sequence length"
            )
        cases.append(
            CasePreset(
                name=raw_case["name"],
                layout=raw_case["layout"],
                seq_lens=tuple(seq_lens),
                seed=raw_case["seed"],
            )
        )
    if tuple(cases) != _EXPECTED_CASES:
        raise EvidenceSchemaError(
            "preset cases do not match the frozen six-case H12/seed contract"
        )
    return EvidencePreset(
        name=payload["name"],
        common=dict(payload["common"]),
        aggregation=payload["aggregation"],
        cases=tuple(cases),
        path=str(path.resolve()),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"failed to verify FlashKDA source provenance at {root}: "
            f"{error.output.strip()}"
        ) from error


def verify_flash_kda_provenance(
    *,
    package_path: Path,
    extension_path: Path,
    source_dir: Path,
    git_output: Callable[..., str] = _git_output,
) -> dict:
    """Verify that Python and extension bits come from the exact pinned checkout."""

    source_dir = source_dir.resolve(strict=True)
    package_path = package_path.resolve(strict=True)
    extension_path = extension_path.resolve(strict=True)
    for label, path in (
        ("flash_kda package", package_path),
        ("flash_kda_C extension", extension_path),
    ):
        if not path.is_relative_to(source_dir):
            raise RuntimeError(
                f"{label} must resolve inside the verified source checkout: "
                f"artifact={path}, checkout={source_dir}"
            )

    source_commit = git_output(source_dir, "rev-parse", "HEAD")
    if source_commit != FLASH_KDA_BASELINE_REVISION:
        raise RuntimeError(
            "unexpected FlashKDA source revision: expected "
            f"{FLASH_KDA_BASELINE_REVISION}, got {source_commit}"
        )
    cutlass_dir = source_dir / "cutlass"
    cutlass_commit = git_output(cutlass_dir, "rev-parse", "HEAD")
    if cutlass_commit != FLASH_KDA_CUTLASS_REVISION:
        raise RuntimeError(
            "unexpected FlashKDA CUTLASS revision: expected "
            f"{FLASH_KDA_CUTLASS_REVISION}, got {cutlass_commit}"
        )
    gitlink = git_output(source_dir, "ls-tree", "HEAD", "cutlass").split()
    if len(gitlink) < 3 or gitlink[2] != cutlass_commit:
        raise RuntimeError(
            "FlashKDA CUTLASS checkout does not match the pinned gitlink"
        )
    tracked_changes = git_output(
        source_dir,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_changes:
        raise RuntimeError(
            f"verified FlashKDA checkout has tracked modifications:\n{tracked_changes}"
        )
    return {
        "repository": FLASH_KDA_REPOSITORY,
        "source_dir": str(source_dir),
        "source_commit": source_commit,
        "cutlass_commit": cutlass_commit,
        "package_path": str(package_path),
        "package_sha256": _sha256(package_path),
        "extension_path": str(extension_path),
        "extension_sha256": _sha256(extension_path),
    }


def _validate_timestamp_range(start_ns: int, end_ns: int, label: str) -> None:
    if start_ns < 0 or end_ns < start_ns:
        raise EvidenceSchemaError(
            f"{label} has invalid timestamps start={start_ns}, end={end_ns}"
        )


def _interval_metrics(activities: Sequence[GpuActivity]) -> dict:
    if not activities:
        raise EvidenceSchemaError("timing scope contains no correlated GPU activities")
    ordered = sorted(
        activities, key=lambda activity: (activity.start_ns, activity.end_ns)
    )
    for activity in ordered:
        _validate_timestamp_range(
            activity.start_ns,
            activity.end_ns,
            f"GPU activity {activity.name!r}",
        )
    span_ns = max(item.end_ns for item in ordered) - ordered[0].start_ns
    activity_sum_ns = sum(item.end_ns - item.start_ns for item in ordered)
    kernel_sum_ns = sum(
        item.end_ns - item.start_ns for item in ordered if item.kind == "kernel"
    )
    current_start = ordered[0].start_ns
    current_end = ordered[0].end_ns
    active_union_ns = 0
    for item in ordered[1:]:
        if item.start_ns <= current_end:
            current_end = max(current_end, item.end_ns)
        else:
            active_union_ns += current_end - current_start
            current_start = item.start_ns
            current_end = item.end_ns
    active_union_ns += current_end - current_start
    return {
        "gpu_span_ms": span_ns / 1e6,
        "gpu_activity_sum_ms": activity_sum_ns / 1e6,
        "kernel_sum_ms": kernel_sum_ns / 1e6,
        "active_union_ms": active_union_ns / 1e6,
        "inter_kernel_gap_ms": (span_ns - active_union_ns) / 1e6,
    }


def _activity_identity(activity: GpuActivity) -> dict:
    payload = asdict(activity)
    payload["duration_ms"] = (activity.end_ns - activity.start_ns) / 1e6
    return payload


def _launch_identity(launch: LaunchActivity) -> dict:
    payload = asdict(launch)
    payload["duration_ms"] = (launch.end_ns - launch.start_ns) / 1e6
    return payload


def _correlated_sample(
    *,
    sample_index: int,
    bracket: CpuBracket,
    launches: Sequence[LaunchActivity],
    activities: Sequence[GpuActivity],
    require_h12_public_route: bool,
) -> dict:
    _validate_timestamp_range(bracket.start_ns, bracket.submitted_ns, "CPU submission")
    _validate_timestamp_range(
        bracket.submitted_ns,
        bracket.synchronized_ns,
        "CPU synchronization",
    )
    selected_launches = [
        launch
        for launch in launches
        if bracket.start_ns <= launch.start_ns <= bracket.submitted_ns
    ]
    correlation_ids = {launch.correlation_id for launch in selected_launches}
    selected_activities = sorted(
        (
            activity
            for activity in activities
            if activity.correlation_id in correlation_ids
        ),
        key=lambda activity: (activity.start_ns, activity.end_ns),
    )
    if not selected_activities:
        raise EvidenceSchemaError(
            f"sample {sample_index} has no GPU activities correlated to its host call"
        )
    if any(
        activity.end_ns > bracket.synchronized_ns for activity in selected_activities
    ):
        raise EvidenceSchemaError(
            f"sample {sample_index} contains an activity after its synchronized boundary"
        )
    contributing_ids = {activity.correlation_id for activity in selected_activities}
    selected_launches = [
        launch
        for launch in selected_launches
        if launch.correlation_id in contributing_ids
    ]
    if not selected_launches:
        raise EvidenceSchemaError(
            f"sample {sample_index} has no contributing launch activities"
        )

    kernels = [
        activity for activity in selected_activities if activity.kind == "kernel"
    ]
    copies = [
        activity
        for activity in selected_activities
        if activity.kind in {"memcpy", "memset"}
    ]
    public_metrics = _interval_metrics(selected_activities)
    result = {
        "sample_index": sample_index,
        **public_metrics,
        "submission_ms": (bracket.submitted_ns - bracket.start_ns) / 1e6,
        "synchronized_e2e_ms": (bracket.synchronized_ns - bracket.start_ns) / 1e6,
        "launch_activity_count": len(selected_launches),
        "launch_activity_names": [launch.name for launch in selected_launches],
        "launch_activity_order": [
            _launch_identity(launch) for launch in selected_launches
        ],
        "gpu_activity_count": len(selected_activities),
        "gpu_activity_names": [activity.name for activity in selected_activities],
        "kernel_activity_count": len(kernels),
        "kernel_activity_names": [activity.name for activity in kernels],
        "copy_activity_count": len(copies),
        "copy_activity_names": [activity.name for activity in copies],
        "activity_order": [_activity_identity(item) for item in selected_activities],
    }
    if require_h12_public_route:
        beta_pack = [
            activity
            for activity in kernels
            if BETA_PACK_ACTIVITY_MARKER in activity.name
        ]
        recurrence = [
            activity
            for activity in kernels
            if RECURRENCE_ACTIVITY_MARKER in activity.name
        ]
        if len(beta_pack) != 1 or len(recurrence) != 1:
            raise EvidenceSchemaError(
                "H12 public timing requires exactly one beta-pack and one M128 "
                "recurrence activity per call; got "
                f"pack={len(beta_pack)}, recurrence={len(recurrence)}, "
                f"names={result['kernel_activity_names']!r}"
            )
        if beta_pack[0].end_ns > recurrence[0].start_ns:
            raise EvidenceSchemaError(
                "H12 public beta-pack must complete before the recurrence activity"
            )
        recurrence_launches = [
            launch
            for launch in selected_launches
            if launch.correlation_id == recurrence[0].correlation_id
        ]
        prepared_metrics = _interval_metrics(recurrence)
        result["prepared_recurrence"] = {
            **prepared_metrics,
            "launch_activity_count": len(recurrence_launches),
            "launch_activity_names": [launch.name for launch in recurrence_launches],
            "launch_activity_order": [
                _launch_identity(launch) for launch in recurrence_launches
            ],
            "gpu_activity_count": len(recurrence),
            "gpu_activity_names": [activity.name for activity in recurrence],
            "kernel_activity_count": len(recurrence),
            "kernel_activity_names": [activity.name for activity in recurrence],
            "activity_order": [_activity_identity(item) for item in recurrence],
        }
    return result


def correlate_samples(
    *,
    brackets: Sequence[CpuBracket],
    launches: Iterable[LaunchActivity],
    activities: Iterable[GpuActivity],
    require_h12_public_route: bool = False,
) -> list[dict]:
    """Correlate CPU brackets to named GPU activities and reduce every sample."""

    if not brackets:
        raise EvidenceSchemaError("at least one CPU bracket is required")
    launch_list = sorted(launches, key=lambda item: item.start_ns)
    activity_list = sorted(activities, key=lambda item: item.start_ns)
    launch_starts = [item.start_ns for item in launch_list]
    samples = []
    for index, bracket in enumerate(brackets):
        lo = bisect.bisect_left(launch_starts, bracket.start_ns)
        hi = bisect.bisect_right(launch_starts, bracket.submitted_ns)
        samples.append(
            _correlated_sample(
                sample_index=index,
                bracket=bracket,
                launches=launch_list[lo:hi],
                activities=activity_list,
                require_h12_public_route=require_h12_public_route,
            )
        )
    return samples


_PUBLIC_NUMERIC_FIELDS = (
    "gpu_span_ms",
    "gpu_activity_sum_ms",
    "kernel_sum_ms",
    "active_union_ms",
    "inter_kernel_gap_ms",
    "submission_ms",
    "synchronized_e2e_ms",
    "launch_activity_count",
    "gpu_activity_count",
    "kernel_activity_count",
    "copy_activity_count",
)
_PREPARED_NUMERIC_FIELDS = (
    "gpu_span_ms",
    "gpu_activity_sum_ms",
    "kernel_sum_ms",
    "active_union_ms",
    "inter_kernel_gap_ms",
    "launch_activity_count",
    "gpu_activity_count",
    "kernel_activity_count",
)


def _summarize_numeric(samples: Sequence[dict], fields: Sequence[str]) -> dict:
    report = {}
    for field in fields:
        values = [sample[field] for sample in samples]
        report[f"{field}_samples"] = values
        report[f"median_{field}"] = float(statistics.median(values))
    return report


def summarize_samples(
    samples: Sequence[dict], *, require_h12_public_route: bool
) -> dict:
    """Preserve raw arrays/names/order and add medians without cross-shape math."""

    if not samples:
        raise EvidenceSchemaError("cannot summarize an empty sample list")
    report = {
        "timing_scope": (
            PUBLIC_TIMING_SCOPE
            if require_h12_public_route
            else "backend_call_first_to_last_correlated_gpu_activity"
        ),
        "timing_backend": "cupti_activity",
        "cold_l2": True,
        "raw_samples": list(samples),
        "launch_activity_names_samples": [
            sample["launch_activity_names"] for sample in samples
        ],
        "launch_activity_order_samples": [
            sample["launch_activity_order"] for sample in samples
        ],
        "gpu_activity_names_samples": [
            sample["gpu_activity_names"] for sample in samples
        ],
        "kernel_activity_names_samples": [
            sample["kernel_activity_names"] for sample in samples
        ],
        "copy_activity_names_samples": [
            sample["copy_activity_names"] for sample in samples
        ],
        "activity_order_samples": [sample["activity_order"] for sample in samples],
        **_summarize_numeric(samples, _PUBLIC_NUMERIC_FIELDS),
    }
    if require_h12_public_route:
        report["call_path"] = "flashinfer.kda.recurrent_kda"
        report["includes_beta_preparation"] = True
        prepared = [sample["prepared_recurrence"] for sample in samples]
        report["prepared_recurrence"] = {
            "call_path": (
                "recurrence activity derived from flashinfer.kda.recurrent_kda"
            ),
            "timing_scope": PREPARED_TIMING_SCOPE,
            "timing_backend": "cupti_activity",
            "derived_from_same_public_samples": True,
            "includes_beta_pack": False,
            "raw_samples": prepared,
            "launch_activity_names_samples": [
                sample["launch_activity_names"] for sample in prepared
            ],
            "launch_activity_order_samples": [
                sample["launch_activity_order"] for sample in prepared
            ],
            "gpu_activity_names_samples": [
                sample["gpu_activity_names"] for sample in prepared
            ],
            "kernel_activity_names_samples": [
                sample["kernel_activity_names"] for sample in prepared
            ],
            "activity_order_samples": [sample["activity_order"] for sample in prepared],
            **_summarize_numeric(prepared, _PREPARED_NUMERIC_FIELDS),
        }
    return report
