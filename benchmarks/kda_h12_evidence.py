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
import math
import os
import re
import statistics
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


FLASH_KDA_REPOSITORY = "https://github.com/MoonshotAI/FlashKDA.git"
FLASH_KDA_BASELINE_REVISION = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
FLASH_KDA_CUTLASS_REVISION = "5c149f52a436782210263fb2f19b354443a61c6a"
FLASHINFER_PHASE_A_UPSTREAM_MAIN_REVISION = "2ab910c58fdd2392914ea05e2a8714946ac0eef6"
FLASHINFER_H12_ROUTE_REVISION = "38bf507f9c9eba6b4544bee016d2bdf9c4fed02b"
PRESET_SCHEMA_VERSION = 1
EVIDENCE_REPORT_SCHEMA_VERSION = 3
FLASH_KDA_BUILD_MANIFEST_SCHEMA_VERSION = 2
DUAL_ARCH_PROMOTION_SCHEMA_VERSION = 2
FROZEN_PRESET_SHA256 = (
    "eef38e8697e2818822186f6c0537c34c1defa41e0b0e08ee272448103a3cf314"
)
SUPPORTED_ARCHITECTURES = {(10, 0): "sm100a", (10, 3): "sm103a"}
REQUIRED_ARCHITECTURES = ("sm100a", "sm103a")

GRAPH_TEST_SOURCE = "tests/kda/test_recurrent_kda_prefill.py"
GRAPH_TEST_SOURCE_LINE_RANGE = (981, 1042)
GRAPH_TEST_NODE_ID = (
    "tests/kda/test_recurrent_kda_prefill.py::"
    "test_frozen_prefill_non_aligned_heads_graph_refreshes_beta"
)
REQUIRED_TIMING_PATHS = (
    "flashinfer_public",
    "flash_kda_raw",
    "flash_kda_public_semantics_adapted",
    "fla_triton",
)
REQUIRED_TIMING_CALL_PATHS = {
    "flashinfer_public": "flashinfer.kda.recurrent_kda",
    "flash_kda_raw": "flash_kda._fwd_raw",
    "flash_kda_public_semantics_adapted": (
        "flash_kda._fwd_raw followed by final-state copy-back"
    ),
    "fla_triton": "fla.ops.kda.chunk_kda (backend dispatch disabled)",
}
REQUIRED_CANDIDATE_IMPORTED_MODULES = (
    "flashinfer.kda",
    "flashinfer.kda_prefill",
)
REQUIRED_CANDIDATE_SOURCE_PATHS = (
    "flashinfer/kda.py",
    "flashinfer/kda_prefill.py",
    "csrc/kda/flashkda_binding_common.cuh",
    "csrc/kda/flashkda_bf16_fused_m128_binding.cu",
    "csrc/kda/flashkda_bf16_fused_m128.cu",
    "benchmarks/bench_recurrent_kda_prefill_h12_phase_a.py",
    "benchmarks/build_flash_kda_phase_a.py",
    "benchmarks/kda_h12_evidence.py",
    "benchmarks/presets/recurrent_kda_prefill_h12_phase_a.json",
    "benchmarks/reduce_kda_h12_phase_a.py",
    GRAPH_TEST_SOURCE,
)

PUBLIC_TIMING_SCOPE = (
    "public_recurrent_kda_first_to_last_correlated_gpu_activity_"
    "including_beta_pack_and_recurrence"
)
PREPARED_TIMING_SCOPE = (
    "prepared_recurrence_kernel_activity_selected_from_the_same_public_call"
)
PHASE_A_MEASUREMENT_CONTRACT = {
    "timing_backend": "cupti_activity",
    "cold_l2": True,
    "warmup_iters_per_block": 5,
    "repeat_iters_per_block": 20,
    "blocks": 2,
    "public_metric": (
        "first-to-last correlated activity span including beta pack and recurrence"
    ),
    "prepared_metric": (
        "recurrence activity from the same public sample, reported separately"
    ),
    "synchronized_e2e_is_diagnostic_only": True,
    "cross_shape_geomean": False,
    "promotion_unit": "complete six-case denominator on one architecture",
}
PHASE_A_EXPECTED_SAMPLE_COUNT = (
    PHASE_A_MEASUREMENT_CONTRACT["blocks"]
    * PHASE_A_MEASUREMENT_CONTRACT["repeat_iters_per_block"]
)
BF16_CORRECTNESS_ATOL = 1e-2
BF16_CORRECTNESS_RTOL = 1e-2
BETA_PACK_ACTIVITY_MARKER = "PackBetaForTmaKernel"
RECURRENCE_ACTIVITY_MARKER = "kernel_flashkda_bf16_fused_m128"
CUPTI_MEMCPY_KIND_DEVICE_TO_DEVICE = 8


class EvidenceSchemaError(ValueError):
    """Raised when checked-in evidence input or activity shape is invalid."""


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace one JSON receipt without exposing a partial file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                allow_nan=False,
                ensure_ascii=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


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
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if sha256 != FROZEN_PRESET_SHA256:
        raise EvidenceSchemaError(
            "preset bytes do not match the frozen Phase-A identity: expected "
            f"{FROZEN_PRESET_SHA256}, got {sha256}"
        )
    return EvidencePreset(
        name=payload["name"],
        common=dict(payload["common"]),
        aggregation=payload["aggregation"],
        cases=tuple(cases),
        path=str(path.resolve()),
        sha256=sha256,
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


def _verify_clean_checkout(
    *,
    root: Path,
    label: str,
    git_output: Callable[..., str],
) -> None:
    status = git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        raise RuntimeError(
            f"{label} has tracked or nonignored untracked changes:\n{status}"
        )


def verify_flash_kda_checkout(
    *,
    source_dir: Path,
    git_output: Callable[..., str] = _git_output,
) -> dict:
    """Verify the exact clean FlashKDA source and CUTLASS identities."""

    source_dir = source_dir.resolve(strict=True)
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
    _verify_clean_checkout(
        root=source_dir,
        label="FlashKDA checkout",
        git_output=git_output,
    )
    _verify_clean_checkout(
        root=cutlass_dir,
        label="FlashKDA CUTLASS checkout",
        git_output=git_output,
    )
    return {
        "repository": FLASH_KDA_REPOSITORY,
        "source_dir": str(source_dir),
        "source_commit": source_commit,
        "cutlass_commit": cutlass_commit,
        "worktree_clean_including_untracked": True,
    }


_FLASH_KDA_BUILD_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "source",
    "build",
    "toolchain",
    "allocation",
    "hardware",
    "artifacts",
}
_FLASH_KDA_BUILD_SOURCE_KEYS = {
    "repository",
    "source_dir",
    "source_commit",
    "cutlass_commit",
    "worktree_clean_including_untracked",
}
_FLASH_KDA_BUILD_KEYS = {"command", "cwd", "environment"}
_FLASH_KDA_BUILD_ENVIRONMENT_KEYS = {
    "CC",
    "CXX",
    "CUDA_HOME",
    "FLASH_KDA_CUDA_ARCHS",
    "MAX_JOBS",
    "NVCC_PREPEND_FLAGS",
    "NVCC_THREADS",
    "TORCH_CUDA_ARCH_LIST",
}
_FLASH_KDA_TOOLCHAIN_KEYS = {
    "python_executable",
    "python_version",
    "platform",
    "torch_version",
    "torch_cuda_version",
    "cuda_home",
    "nvcc_path",
    "nvcc_version",
    "cxx_path",
    "cxx_version",
}
_FLASH_KDA_ALLOCATION_KEYS = {
    "slurm_job_id",
    "slurm_cluster_name",
    "slurm_partition",
    "slurm_node_list",
}
_FLASH_KDA_HARDWARE_KEYS = {
    "cuda_available",
    "cuda_arch",
    "compute_capability",
    "device_name",
    "device_uuid",
}
_FLASH_KDA_ARTIFACT_KEYS = {
    "package_path",
    "package_sha256",
    "extension_path",
    "extension_sha256",
}


def validate_flash_kda_build_manifest_schema(payload: dict) -> dict:
    """Validate an allocation-build manifest without importing CUDA."""

    if not isinstance(payload, dict):
        raise EvidenceSchemaError("FlashKDA build manifest root must be an object")
    _require_exact_keys(payload, _FLASH_KDA_BUILD_MANIFEST_KEYS, "build manifest")
    if payload["schema_version"] != FLASH_KDA_BUILD_MANIFEST_SCHEMA_VERSION:
        raise EvidenceSchemaError(
            "FlashKDA build manifest schema_version must be "
            f"{FLASH_KDA_BUILD_MANIFEST_SCHEMA_VERSION}"
        )
    if payload["kind"] != "phase_a_flash_kda_in_allocation_build":
        raise EvidenceSchemaError("unexpected FlashKDA build manifest kind")
    nested_contracts = (
        ("source", _FLASH_KDA_BUILD_SOURCE_KEYS),
        ("build", _FLASH_KDA_BUILD_KEYS),
        ("toolchain", _FLASH_KDA_TOOLCHAIN_KEYS),
        ("allocation", _FLASH_KDA_ALLOCATION_KEYS),
        ("hardware", _FLASH_KDA_HARDWARE_KEYS),
        ("artifacts", _FLASH_KDA_ARTIFACT_KEYS),
    )
    for label, keys in nested_contracts:
        value = payload[label]
        if not isinstance(value, dict):
            raise EvidenceSchemaError(f"build manifest {label} must be an object")
        _require_exact_keys(value, keys, f"build manifest {label}")

    source = payload["source"]
    if source["repository"] != FLASH_KDA_REPOSITORY:
        raise EvidenceSchemaError("build manifest has unexpected FlashKDA repository")
    if source["source_commit"] != FLASH_KDA_BASELINE_REVISION:
        raise EvidenceSchemaError("build manifest has unexpected FlashKDA revision")
    if source["cutlass_commit"] != FLASH_KDA_CUTLASS_REVISION:
        raise EvidenceSchemaError("build manifest has unexpected CUTLASS revision")
    if source["worktree_clean_including_untracked"] is not True:
        raise EvidenceSchemaError("build manifest source was not clean")
    source_dir_value = source["source_dir"]
    if (
        not isinstance(source_dir_value, str)
        or not Path(source_dir_value).is_absolute()
    ):
        raise EvidenceSchemaError("build manifest source_dir must be an absolute path")

    build = payload["build"]
    command = build["command"]
    if (
        not isinstance(command, list)
        or any(not isinstance(item, str) or not item for item in command)
        or len(command) != 5
        or command[1:] != ["setup.py", "build_ext", "--inplace", "--force"]
    ):
        raise EvidenceSchemaError(
            "build manifest command must be exactly Python setup.py build_ext "
            "--inplace --force"
        )
    if build["cwd"] != source_dir_value:
        raise EvidenceSchemaError("build manifest cwd must equal its source_dir")
    if not isinstance(build["environment"], dict):
        raise EvidenceSchemaError(
            "build manifest environment must map strings to strings or null"
        )
    _require_exact_keys(
        build["environment"],
        _FLASH_KDA_BUILD_ENVIRONMENT_KEYS,
        "build manifest environment",
    )
    if any(
        value is not None and not isinstance(value, str)
        for value in build["environment"].values()
    ):
        raise EvidenceSchemaError(
            "build manifest environment must map strings to strings or null"
        )
    if build["environment"]["FLASH_KDA_CUDA_ARCHS"] != "auto":
        raise EvidenceSchemaError(
            "Phase-A FlashKDA build must use FLASH_KDA_CUDA_ARCHS=auto"
        )
    if build["environment"]["NVCC_PREPEND_FLAGS"] is not None:
        raise EvidenceSchemaError(
            "Phase-A FlashKDA build forbids ambient NVCC_PREPEND_FLAGS"
        )
    if build["environment"]["TORCH_CUDA_ARCH_LIST"] is not None:
        raise EvidenceSchemaError(
            "Phase-A FlashKDA build forbids ambient TORCH_CUDA_ARCH_LIST"
        )
    nvcc_threads = build["environment"]["NVCC_THREADS"]
    if (
        not isinstance(nvcc_threads, str)
        or not nvcc_threads.isdigit()
        or int(nvcc_threads) <= 0
    ):
        raise EvidenceSchemaError(
            "Phase-A FlashKDA build NVCC_THREADS must be a positive integer"
        )
    toolchain = payload["toolchain"]
    for key in _FLASH_KDA_TOOLCHAIN_KEYS:
        if not isinstance(toolchain[key], str) or not toolchain[key]:
            raise EvidenceSchemaError(
                f"build manifest toolchain {key} must be a non-empty string"
            )
    if command[0] != toolchain["python_executable"]:
        raise EvidenceSchemaError(
            "build command interpreter does not match recorded Python toolchain"
        )

    allocation = payload["allocation"]
    for key in _FLASH_KDA_ALLOCATION_KEYS:
        value = allocation[key]
        if not isinstance(value, str) or not value or value == "unknown":
            raise EvidenceSchemaError(
                f"FlashKDA build manifest allocation {key} must be resolved"
            )
    hardware = payload["hardware"]
    if hardware["cuda_available"] is not True:
        raise EvidenceSchemaError("FlashKDA build did not run with an allocated GPU")
    if hardware["cuda_arch"] not in REQUIRED_ARCHITECTURES:
        raise EvidenceSchemaError(
            "FlashKDA build architecture must be sm100a or sm103a"
        )
    expected_capabilities = {"sm100a": [10, 0], "sm103a": [10, 3]}
    if hardware["compute_capability"] != expected_capabilities[hardware["cuda_arch"]]:
        raise EvidenceSchemaError(
            "FlashKDA build architecture and compute capability disagree"
        )
    for key in ("device_name", "device_uuid"):
        value = hardware[key]
        if not isinstance(value, str) or not value or value == "unavailable":
            raise EvidenceSchemaError(
                f"FlashKDA build manifest hardware {key} must be resolved"
            )

    artifacts = payload["artifacts"]
    source_dir = Path(source_dir_value)
    for key in ("package_path", "extension_path"):
        value = artifacts[key]
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or not Path(value).is_relative_to(source_dir)
        ):
            raise EvidenceSchemaError(
                f"build manifest artifact {key} must be inside source_dir"
            )
    for key in ("package_sha256", "extension_sha256"):
        _require_sha256(artifacts[key], f"build manifest artifact {key}")
    return payload


def load_flash_kda_build_manifest(path: Path) -> tuple[dict, str]:
    """Load a structurally valid build manifest and return its SHA-256."""

    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise EvidenceSchemaError(
            f"invalid FlashKDA build manifest JSON at {path}: {error}"
        ) from error
    manifest = validate_flash_kda_build_manifest_schema(payload)
    return manifest, flash_kda_build_manifest_sha256(manifest)


def flash_kda_build_manifest_sha256(payload: dict) -> str:
    """Hash the validated manifest canonically for embedding in receipts."""

    manifest = validate_flash_kda_build_manifest_schema(payload)
    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def verify_flash_kda_provenance(
    *,
    package_path: Path,
    extension_path: Path,
    source_dir: Path,
    build_manifest_path: Path,
    git_output: Callable[..., str] = _git_output,
) -> dict:
    """Bind imported artifacts to an exact in-allocation rebuild manifest."""

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

    checkout = verify_flash_kda_checkout(
        source_dir=source_dir,
        git_output=git_output,
    )
    build_manifest_path = build_manifest_path.resolve(strict=True)
    if build_manifest_path.is_relative_to(source_dir):
        raise RuntimeError(
            "FlashKDA build manifest must be written outside the source checkout"
        )
    manifest, manifest_sha256 = load_flash_kda_build_manifest(build_manifest_path)
    source = manifest["source"]
    if Path(source["source_dir"]).resolve() != source_dir:
        raise RuntimeError(
            "FlashKDA build manifest source_dir does not match verified checkout"
        )
    if (
        source["source_commit"] != checkout["source_commit"]
        or source["cutlass_commit"] != checkout["cutlass_commit"]
    ):
        raise RuntimeError(
            "FlashKDA build manifest source identity does not match checkout"
        )
    artifacts = manifest["artifacts"]
    expected_paths = {
        "package_path": package_path,
        "extension_path": extension_path,
    }
    for key, actual_path in expected_paths.items():
        if Path(artifacts[key]).resolve() != actual_path:
            raise RuntimeError(
                f"imported {key} does not match the allocation build manifest"
            )
    package_sha256 = _sha256(package_path)
    extension_sha256 = _sha256(extension_path)
    if package_sha256 != artifacts["package_sha256"]:
        raise RuntimeError(
            "imported FlashKDA package hash does not match allocation build manifest"
        )
    if extension_sha256 != artifacts["extension_sha256"]:
        raise RuntimeError(
            "imported FlashKDA extension hash does not match allocation build manifest"
        )
    return {
        **checkout,
        "package_path": str(package_path),
        "package_sha256": package_sha256,
        "extension_path": str(extension_path),
        "extension_sha256": extension_sha256,
        "build_manifest_path": str(build_manifest_path),
        "build_manifest_sha256": manifest_sha256,
        "build_manifest": manifest,
    }


def verify_flash_kda_current_receipt_binding(
    manifest: dict,
    *,
    allocation: dict[str, str],
    hardware: dict[str, object],
    runtime: dict[str, str],
) -> dict:
    """Bind one FlashKDA build to this receipt's allocation, GPU, and runtime."""

    manifest = validate_flash_kda_build_manifest_schema(manifest)
    expected_allocation = manifest["allocation"]
    if set(allocation) != _FLASH_KDA_ALLOCATION_KEYS:
        raise EvidenceSchemaError(
            "current receipt allocation must contain the exact Slurm identity"
        )
    for key, value in allocation.items():
        if not isinstance(value, str) or not value or value == "unknown":
            raise EvidenceSchemaError(
                f"current receipt allocation {key} must be resolved"
            )
    if allocation != expected_allocation:
        raise EvidenceSchemaError(
            "FlashKDA build manifest does not belong to the current Slurm allocation"
        )

    expected_hardware = {
        key: manifest["hardware"][key]
        for key in (
            "cuda_arch",
            "compute_capability",
            "device_name",
            "device_uuid",
        )
    }
    if hardware != expected_hardware:
        raise EvidenceSchemaError(
            "FlashKDA build manifest does not belong to the current GPU"
        )

    runtime_keys = {
        "python_executable",
        "python_version",
        "platform",
        "torch_version",
        "torch_cuda_version",
        "cuda_home",
    }
    if set(runtime) != runtime_keys:
        raise EvidenceSchemaError(
            "current receipt runtime must contain the exact Python/Torch/CUDA identity"
        )
    expected_runtime = {key: manifest["toolchain"][key] for key in runtime_keys}
    if runtime != expected_runtime:
        raise EvidenceSchemaError(
            "FlashKDA build manifest runtime differs from the current receipt runtime"
        )
    return {
        "schema_version": 1,
        "same_slurm_allocation": True,
        "same_gpu": True,
        "same_python_torch_cuda_runtime": True,
        "allocation": dict(allocation),
        "hardware": dict(hardware),
        "runtime": dict(runtime),
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


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvidenceSchemaError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_commit(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvidenceSchemaError(f"{label} must be a full lowercase Git commit")
    return value


def _require_finite_number(
    value: object,
    label: str,
    *,
    positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value <= 0)
    ):
        qualifier = "positive " if positive else "non-negative "
        raise EvidenceSchemaError(f"{label} must be a finite {qualifier}number")
    return float(value)


def _validate_activity_records(
    records: object,
    names: object,
    expected_count: int,
    *,
    label: str,
    allowed_kinds: set[str],
) -> list[dict]:
    if (
        not isinstance(records, list)
        or len(records) != expected_count
        or not isinstance(names, list)
        or len(names) != expected_count
    ):
        raise EvidenceSchemaError(f"{label} activity records/count disagree")
    expected_keys = {
        "start_ns",
        "end_ns",
        "correlation_id",
        "kind",
        "name",
        "duration_ms",
    }
    previous_start = -1
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise EvidenceSchemaError(f"{label} activity {index} must be an object")
        _require_exact_keys(record, expected_keys, f"{label} activity {index}")
        start_ns = record["start_ns"]
        end_ns = record["end_ns"]
        if (
            type(start_ns) is not int
            or type(end_ns) is not int
            or type(record["correlation_id"]) is not int
            or record["correlation_id"] < 0
            or start_ns < previous_start
            or start_ns < 0
            or end_ns < start_ns
            or record["kind"] not in allowed_kinds
            or not isinstance(record["name"], str)
            or not record["name"]
            or record["name"] != names[index]
            or (
                record["kind"] in {"runtime", "driver"}
                and not record["name"].startswith(f"{record['kind']}:")
            )
        ):
            raise EvidenceSchemaError(f"{label} activity {index} is malformed")
        duration_ms = _require_finite_number(
            record["duration_ms"], f"{label} activity {index} duration"
        )
        if duration_ms != (end_ns - start_ns) / 1e6:
            raise EvidenceSchemaError(
                f"{label} activity {index} duration does not match timestamps"
            )
        previous_start = start_ns
    return records


def _validate_prepared_raw_sample(sample: object, *, label: str) -> None:
    expected_keys = {
        *_PREPARED_NUMERIC_FIELDS,
        "launch_activity_names",
        "launch_activity_order",
        "gpu_activity_names",
        "kernel_activity_names",
        "activity_order",
    }
    if not isinstance(sample, dict):
        raise EvidenceSchemaError(f"{label} must be an object")
    _require_exact_keys(sample, expected_keys, label)
    for field in _PREPARED_NUMERIC_FIELDS:
        if field.endswith("_count"):
            if type(sample[field]) is not int or sample[field] <= 0:
                raise EvidenceSchemaError(f"{label} {field} must be a positive integer")
        else:
            _require_finite_number(
                sample[field],
                f"{label} {field}",
                positive=field in {"gpu_span_ms", "kernel_sum_ms", "active_union_ms"},
            )
    launches = _validate_activity_records(
        sample["launch_activity_order"],
        sample["launch_activity_names"],
        sample["launch_activity_count"],
        label=f"{label} launch",
        allowed_kinds={"runtime", "driver"},
    )
    activities = _validate_activity_records(
        sample["activity_order"],
        sample["gpu_activity_names"],
        sample["gpu_activity_count"],
        label=f"{label} GPU",
        allowed_kinds={"kernel", "memcpy", "memset"},
    )
    kernel_names = [
        record["name"] for record in activities if record["kind"] == "kernel"
    ]
    if (
        sample["kernel_activity_count"] != 1
        or sample["gpu_activity_count"] != 1
        or kernel_names != sample["kernel_activity_names"]
        or len(kernel_names) != 1
        or RECURRENCE_ACTIVITY_MARKER not in kernel_names[0]
        or {record["correlation_id"] for record in activities}
        != {record["correlation_id"] for record in launches}
    ):
        raise EvidenceSchemaError(f"{label} is not recurrence-only CUPTI evidence")
    recomputed = _interval_metrics(
        [
            GpuActivity(
                start_ns=record["start_ns"],
                end_ns=record["end_ns"],
                correlation_id=record["correlation_id"],
                kind=record["kind"],
                name=record["name"],
            )
            for record in activities
        ]
    )
    if any(sample[field] != value for field, value in recomputed.items()):
        raise EvidenceSchemaError(f"{label} interval metrics do not match raw activity")


def _validate_raw_sample(
    sample: object,
    *,
    path_name: str,
    block_index: int,
    order_index: int,
    sample_index: int,
    expected_final_state_bytes: int,
) -> None:
    expected_keys = {
        "sample_index",
        *_PUBLIC_NUMERIC_FIELDS,
        "launch_activity_names",
        "launch_activity_order",
        "gpu_activity_names",
        "kernel_activity_names",
        "copy_activity_names",
        "activity_order",
        "block_index",
        "order_index",
    }
    if path_name == "flashinfer_public":
        expected_keys.add("prepared_recurrence")
    label = f"path {path_name} block {block_index} sample {sample_index}"
    if not isinstance(sample, dict):
        raise EvidenceSchemaError(f"{label} must be an object")
    _require_exact_keys(sample, expected_keys, label)
    if (
        sample["sample_index"] != sample_index
        or sample["block_index"] != block_index
        or sample["order_index"] != order_index
    ):
        raise EvidenceSchemaError(f"{label} sample/block/order identity is wrong")
    for field in _PUBLIC_NUMERIC_FIELDS:
        if field.endswith("_count"):
            if type(sample[field]) is not int or sample[field] < 0:
                raise EvidenceSchemaError(
                    f"{label} {field} must be a non-negative integer"
                )
        else:
            _require_finite_number(
                sample[field],
                f"{label} {field}",
                positive=field in {"gpu_span_ms", "kernel_sum_ms", "active_union_ms"},
            )
    if (
        sample["launch_activity_count"] <= 0
        or sample["gpu_activity_count"] <= 0
        or sample["kernel_activity_count"] <= 0
        or sample["synchronized_e2e_ms"] < sample["submission_ms"]
        or sample["synchronized_e2e_ms"] < sample["gpu_span_ms"]
    ):
        raise EvidenceSchemaError(f"{label} has incomplete correlated timing")
    launches = _validate_activity_records(
        sample["launch_activity_order"],
        sample["launch_activity_names"],
        sample["launch_activity_count"],
        label=f"{label} launch",
        allowed_kinds={"runtime", "driver"},
    )
    activities = _validate_activity_records(
        sample["activity_order"],
        sample["gpu_activity_names"],
        sample["gpu_activity_count"],
        label=f"{label} GPU",
        allowed_kinds={"kernel", "memcpy", "memset"},
    )
    kernel_names = [
        record["name"] for record in activities if record["kind"] == "kernel"
    ]
    copy_names = [
        record["name"]
        for record in activities
        if record["kind"] in {"memcpy", "memset"}
    ]
    if (
        kernel_names != sample["kernel_activity_names"]
        or len(kernel_names) != sample["kernel_activity_count"]
        or copy_names != sample["copy_activity_names"]
        or len(copy_names) != sample["copy_activity_count"]
        or {record["correlation_id"] for record in activities}
        != {record["correlation_id"] for record in launches}
    ):
        raise EvidenceSchemaError(f"{label} activity counts/names are inconsistent")
    recomputed = _interval_metrics(
        [
            GpuActivity(
                start_ns=record["start_ns"],
                end_ns=record["end_ns"],
                correlation_id=record["correlation_id"],
                kind=record["kind"],
                name=record["name"],
            )
            for record in activities
        ]
    )
    if any(sample[field] != value for field, value in recomputed.items()):
        raise EvidenceSchemaError(f"{label} interval metrics do not match raw activity")
    kernel_records = [record for record in activities if record["kind"] == "kernel"]
    copy_records = [
        record for record in activities if record["kind"] in {"memcpy", "memset"}
    ]
    recurrence_records = [
        record
        for record in kernel_records
        if RECURRENCE_ACTIVITY_MARKER in record["name"]
    ]
    if path_name == "flashinfer_public":
        beta_pack_records = [
            record
            for record in kernel_records
            if BETA_PACK_ACTIVITY_MARKER in record["name"]
        ]
        if (
            len(kernel_records) != 2
            or len(activities) != 2
            or copy_records
            or len(beta_pack_records) != 1
            or len(recurrence_records) != 1
            or beta_pack_records[0]["end_ns"] > recurrence_records[0]["start_ns"]
        ):
            raise EvidenceSchemaError(
                f"{label} is not the exact nonoverlapping pack-to-recurrence route"
            )
        _validate_prepared_raw_sample(
            sample["prepared_recurrence"], label=f"{label} prepared recurrence"
        )
    elif path_name == "flash_kda_raw":
        if (
            len(activities) != 1
            or len(kernel_records) != 1
            or copy_records
            or len(recurrence_records) != 1
        ):
            raise EvidenceSchemaError(
                f"{label} is not the exact pinned FlashKDA recurrence route"
            )
    elif path_name == "flash_kda_public_semantics_adapted":
        expected_copy_name = (
            "MEMCPY(copy_kind="
            f"{CUPTI_MEMCPY_KIND_DEVICE_TO_DEVICE},"
            f"bytes={expected_final_state_bytes})"
        )
        if (
            len(activities) != 2
            or len(kernel_records) != 1
            or len(recurrence_records) != 1
            or len(copy_records) != 1
            or copy_records[0]["kind"] != "memcpy"
            or copy_records[0]["name"] != expected_copy_name
            or copy_records[0]["start_ns"] < recurrence_records[0]["end_ns"]
        ):
            raise EvidenceSchemaError(
                f"{label} lacks the exact post-recurrence full-state D2D copy-back"
            )
    elif path_name == "fla_triton":
        if not any("kda" in record["name"].lower() for record in kernel_records):
            raise EvidenceSchemaError(
                f"{label} lacks an identifiable FLA KDA kernel activity"
            )


def _validate_timing_summary(
    timing: object,
    *,
    path_name: str,
    expected_sample_count: int,
    expected_final_state_bytes: int,
) -> None:
    if not isinstance(timing, dict):
        raise EvidenceSchemaError(f"path {path_name} timing must be an object")
    expected_scope = (
        PUBLIC_TIMING_SCOPE
        if path_name == "flashinfer_public"
        else "backend_call_first_to_last_correlated_gpu_activity"
    )
    expected_keys = {
        "timing_scope",
        "timing_backend",
        "cold_l2",
        "raw_samples",
        "launch_activity_names_samples",
        "launch_activity_order_samples",
        "gpu_activity_names_samples",
        "kernel_activity_names_samples",
        "copy_activity_names_samples",
        "activity_order_samples",
        "call_path",
        *(f"{field}_samples" for field in _PUBLIC_NUMERIC_FIELDS),
        *(f"median_{field}" for field in _PUBLIC_NUMERIC_FIELDS),
    }
    if path_name == "flashinfer_public":
        expected_keys.update({"includes_beta_preparation", "prepared_recurrence"})
    _require_exact_keys(timing, expected_keys, f"path {path_name} timing")
    if (
        timing["timing_scope"] != expected_scope
        or timing["timing_backend"] != "cupti_activity"
        or timing["cold_l2"] is not True
        or timing["call_path"] != REQUIRED_TIMING_CALL_PATHS[path_name]
    ):
        raise EvidenceSchemaError(f"path {path_name} timing scope is not exact")
    raw_samples = timing["raw_samples"]
    if not isinstance(raw_samples, list) or len(raw_samples) != expected_sample_count:
        raise EvidenceSchemaError(f"path {path_name} lacks the exact raw denominator")
    base_order = list(REQUIRED_TIMING_PATHS)
    samples_per_block = PHASE_A_MEASUREMENT_CONTRACT["repeat_iters_per_block"]
    for flat_index, sample in enumerate(raw_samples):
        block_index = flat_index // samples_per_block
        sample_index = flat_index % samples_per_block
        block_order = base_order if block_index % 2 == 0 else list(reversed(base_order))
        order_index = block_order.index(path_name)
        _validate_raw_sample(
            sample,
            path_name=path_name,
            block_index=block_index,
            order_index=order_index,
            sample_index=sample_index,
            expected_final_state_bytes=expected_final_state_bytes,
        )
    derived_arrays = {
        "launch_activity_names_samples": [
            sample["launch_activity_names"] for sample in raw_samples
        ],
        "launch_activity_order_samples": [
            sample["launch_activity_order"] for sample in raw_samples
        ],
        "gpu_activity_names_samples": [
            sample["gpu_activity_names"] for sample in raw_samples
        ],
        "kernel_activity_names_samples": [
            sample["kernel_activity_names"] for sample in raw_samples
        ],
        "copy_activity_names_samples": [
            sample["copy_activity_names"] for sample in raw_samples
        ],
        "activity_order_samples": [sample["activity_order"] for sample in raw_samples],
    }
    if any(timing[key] != value for key, value in derived_arrays.items()):
        raise EvidenceSchemaError(
            f"path {path_name} summary arrays differ from raw data"
        )
    expected_numeric = _summarize_numeric(raw_samples, _PUBLIC_NUMERIC_FIELDS)
    if any(timing[key] != value for key, value in expected_numeric.items()):
        raise EvidenceSchemaError(f"path {path_name} medians differ from raw data")

    if path_name != "flashinfer_public":
        return
    if timing["includes_beta_preparation"] is not True:
        raise EvidenceSchemaError("public timing must include beta preparation")
    prepared = timing["prepared_recurrence"]
    prepared_keys = {
        "call_path",
        "timing_scope",
        "timing_backend",
        "derived_from_same_public_samples",
        "includes_beta_pack",
        "raw_samples",
        "launch_activity_names_samples",
        "launch_activity_order_samples",
        "gpu_activity_names_samples",
        "kernel_activity_names_samples",
        "activity_order_samples",
        *(f"{field}_samples" for field in _PREPARED_NUMERIC_FIELDS),
        *(f"median_{field}" for field in _PREPARED_NUMERIC_FIELDS),
    }
    if not isinstance(prepared, dict):
        raise EvidenceSchemaError("public timing lacks prepared recurrence object")
    _require_exact_keys(prepared, prepared_keys, "prepared recurrence timing")
    prepared_raw = [sample["prepared_recurrence"] for sample in raw_samples]
    if (
        prepared["call_path"]
        != "recurrence activity derived from flashinfer.kda.recurrent_kda"
        or prepared["timing_scope"] != PREPARED_TIMING_SCOPE
        or prepared["timing_backend"] != "cupti_activity"
        or prepared["derived_from_same_public_samples"] is not True
        or prepared["includes_beta_pack"] is not False
        or prepared["raw_samples"] != prepared_raw
    ):
        raise EvidenceSchemaError("prepared timing is not derived from public samples")
    prepared_arrays = {
        "launch_activity_names_samples": [
            sample["launch_activity_names"] for sample in prepared_raw
        ],
        "launch_activity_order_samples": [
            sample["launch_activity_order"] for sample in prepared_raw
        ],
        "gpu_activity_names_samples": [
            sample["gpu_activity_names"] for sample in prepared_raw
        ],
        "kernel_activity_names_samples": [
            sample["kernel_activity_names"] for sample in prepared_raw
        ],
        "activity_order_samples": [sample["activity_order"] for sample in prepared_raw],
    }
    expected_prepared_numeric = _summarize_numeric(
        prepared_raw, _PREPARED_NUMERIC_FIELDS
    )
    if any(prepared[key] != value for key, value in prepared_arrays.items()) or any(
        prepared[key] != value for key, value in expected_prepared_numeric.items()
    ):
        raise EvidenceSchemaError("prepared summary differs from raw public samples")


def _require_oracle(case: dict, key: str, expected: CasePreset) -> None:
    correctness = case.get("correctness")
    if not isinstance(correctness, dict):
        raise EvidenceSchemaError(f"case {case.get('name')!r} lacks correctness")
    oracle = correctness.get(key)
    if not isinstance(oracle, dict):
        raise EvidenceSchemaError(
            f"case {case.get('name')!r} lacks required oracle {key}"
        )
    _require_exact_keys(oracle, {"output", "final_state"}, f"oracle {key}")
    expected_numel = {
        "output": expected.total_tokens * 12 * 128,
        "final_state": len(expected.seq_lens) * 12 * 128 * 128,
    }
    for result_name in ("output", "final_state"):
        result = oracle.get(result_name)
        if isinstance(result, dict):
            _require_exact_keys(
                result,
                {
                    "passed",
                    "max_abs",
                    "max_allowed_abs",
                    "mismatch_count",
                    "atol",
                    "rtol",
                    "compared_dtype",
                    "compared_numel",
                },
                f"oracle {key} {result_name}",
            )
        max_abs = result.get("max_abs") if isinstance(result, dict) else None
        max_allowed_abs = (
            result.get("max_allowed_abs") if isinstance(result, dict) else None
        )
        if (
            not isinstance(result, dict)
            or result.get("passed") is not True
            or isinstance(max_abs, bool)
            or not isinstance(max_abs, (int, float))
            or not math.isfinite(max_abs)
            or max_abs < 0
            or isinstance(max_allowed_abs, bool)
            or not isinstance(max_allowed_abs, (int, float))
            or not math.isfinite(max_allowed_abs)
            or max_allowed_abs < 0
            or max_abs > max_allowed_abs
            or type(result.get("mismatch_count")) is not int
            or result["mismatch_count"] != 0
            or result.get("atol") != BF16_CORRECTNESS_ATOL
            or result.get("rtol") != BF16_CORRECTNESS_RTOL
            or result.get("compared_dtype") != "bfloat16"
            or type(result.get("compared_numel")) is not int
            or result["compared_numel"] != expected_numel[result_name]
        ):
            raise EvidenceSchemaError(
                f"case {case.get('name')!r} oracle {key} {result_name} "
                "does not prove the exact BF16 full-tensor contract"
            )


def _validate_case_receipt(
    case: dict,
    expected: CasePreset,
    *,
    expected_sample_count: int,
) -> None:
    expected_identity = {
        "name": expected.name,
        "layout": expected.layout,
        "seq_lens": list(expected.seq_lens),
        "total_tokens": expected.total_tokens,
        "num_sequences": len(expected.seq_lens),
        "seed": expected.seed,
        "num_heads": 12,
        "head_dim_qk": 128,
        "head_dim_vo": 128,
        "dtype": "bfloat16",
        "initial_state": "provided_bfloat16",
        "variant": "m128",
    }
    _require_exact_keys(
        case,
        {
            *expected_identity,
            "correctness",
            "timings",
            "measurement_order",
            "per_case_speedups",
            "cross_shape_aggregate",
        },
        f"case {expected.name!r}",
    )
    for key, value in expected_identity.items():
        if case.get(key) != value:
            raise EvidenceSchemaError(
                f"case {expected.name!r} {key} must be {value!r}, got {case.get(key)!r}"
            )
    correctness = case.get("correctness")
    if (
        not isinstance(correctness, dict)
        or correctness.get("passed") is not True
        or correctness.get("public_output_and_full_final_state") is not True
    ):
        raise EvidenceSchemaError(
            f"case {expected.name!r} lacks complete public correctness"
        )
    _require_exact_keys(
        correctness,
        {
            "passed",
            "public_output_and_full_final_state",
            "independent_bf16_recurrence",
            "pinned_flash_kda",
            "fla_triton",
        },
        f"case {expected.name!r} correctness",
    )
    for oracle in (
        "independent_bf16_recurrence",
        "pinned_flash_kda",
        "fla_triton",
    ):
        _require_oracle(case, oracle, expected)

    timings = case.get("timings")
    if not isinstance(timings, dict) or set(timings) != set(REQUIRED_TIMING_PATHS):
        raise EvidenceSchemaError(
            f"case {expected.name!r} must time exactly {REQUIRED_TIMING_PATHS!r}"
        )
    for path_name in REQUIRED_TIMING_PATHS:
        _validate_timing_summary(
            timings[path_name],
            path_name=path_name,
            expected_sample_count=expected_sample_count,
            expected_final_state_bytes=(len(expected.seq_lens) * 12 * 128 * 128 * 2),
        )
    public = timings["flashinfer_public"]
    for names in public["kernel_activity_names_samples"]:
        if (
            len(names) != 2
            or BETA_PACK_ACTIVITY_MARKER not in names[0]
            or RECURRENCE_ACTIVITY_MARKER not in names[1]
        ):
            raise EvidenceSchemaError(
                f"case {expected.name!r} public timing lost exact pack/recurrence order"
            )
    if any(
        "flashkda" in name.lower()
        for names in timings["fla_triton"]["kernel_activity_names_samples"]
        for name in names
    ):
        raise EvidenceSchemaError(
            f"case {expected.name!r} FLA timing routed through FlashKDA"
        )
    base_order = list(REQUIRED_TIMING_PATHS)
    expected_measurement_order = []
    for block_index in range(PHASE_A_MEASUREMENT_CONTRACT["blocks"]):
        block_order = base_order if block_index % 2 == 0 else list(reversed(base_order))
        expected_measurement_order.extend(
            {
                "block_index": block_index,
                "order_index": order_index,
                "path": path_name,
            }
            for order_index, path_name in enumerate(block_order)
        )
    if case["measurement_order"] != expected_measurement_order:
        raise EvidenceSchemaError(
            f"case {expected.name!r} measurement order is not exact forward/reverse"
        )
    speedups = case["per_case_speedups"]
    expected_speedup_keys = {
        "vs_pinned_flash_kda_raw": "flash_kda_raw",
        "vs_pinned_flash_kda_public_semantics_adapted": (
            "flash_kda_public_semantics_adapted"
        ),
        "vs_fla_triton": "fla_triton",
    }
    if not isinstance(speedups, dict) or set(speedups) != set(expected_speedup_keys):
        raise EvidenceSchemaError(f"case {expected.name!r} speedup keys are incomplete")
    candidate_ms = public["median_gpu_span_ms"]
    for speedup_name, timing_name in expected_speedup_keys.items():
        expected_speedup = timings[timing_name]["median_gpu_span_ms"] / candidate_ms
        value = _require_finite_number(
            speedups[speedup_name],
            f"case {expected.name!r} {speedup_name}",
            positive=True,
        )
        if value != expected_speedup:
            raise EvidenceSchemaError(
                f"case {expected.name!r} {speedup_name} differs from raw timing"
            )
    if case["cross_shape_aggregate"] is not None:
        raise EvidenceSchemaError(
            f"case {expected.name!r} injected a forbidden cross-shape aggregate"
        )


def _validate_graph_receipt(graph: object) -> None:
    if not isinstance(graph, dict):
        raise EvidenceSchemaError("receipt lacks changed-beta CUDA Graph evidence")
    _require_exact_keys(
        graph,
        {
            "source",
            "source_line_range",
            "source_sha256",
            "node_id",
            "parameterization",
            "command",
            "returncode",
            "stdout",
            "stderr",
            "passed",
        },
        "changed-beta CUDA Graph receipt",
    )
    if graph.get("passed") is not True or graph.get("returncode") != 0:
        raise EvidenceSchemaError("changed-beta CUDA Graph test did not pass")
    if graph.get("node_id") != GRAPH_TEST_NODE_ID:
        raise EvidenceSchemaError("CUDA Graph receipt has the wrong pytest node")
    if graph.get("source") != GRAPH_TEST_SOURCE or graph.get(
        "source_line_range"
    ) != list(GRAPH_TEST_SOURCE_LINE_RANGE):
        raise EvidenceSchemaError("CUDA Graph receipt has the wrong source range")
    if graph.get("parameterization") != {"num_heads": [6, 12]}:
        raise EvidenceSchemaError("CUDA Graph receipt omitted the H6/H12 cases")
    command = graph.get("command")
    if not isinstance(command, list) or command[1:] != [
        "-m",
        "pytest",
        "-q",
        GRAPH_TEST_NODE_ID,
    ]:
        raise EvidenceSchemaError("CUDA Graph receipt command is not the exact target")
    if (
        not isinstance(command[0], str)
        or not Path(command[0]).is_absolute()
        or not isinstance(graph["stdout"], str)
        or not isinstance(graph["stderr"], str)
        or re.search(r"\b2 passed\b", graph["stdout"]) is None
    ):
        raise EvidenceSchemaError("CUDA Graph receipt process evidence is malformed")
    _require_sha256(graph.get("source_sha256"), "CUDA Graph source hash")


def validate_per_arch_receipt(
    report: dict,
    *,
    expected_arch: str,
    expected_candidate_commit: str,
    expected_fla_commit: str,
    preset: EvidencePreset,
) -> dict:
    """Validate one complete per-architecture receipt and return its identity."""

    _require_commit(expected_candidate_commit, "expected FlashInfer commit")
    _require_commit(expected_fla_commit, "expected FLA commit")
    if not isinstance(report, dict):
        raise EvidenceSchemaError("per-architecture receipt must be an object")
    _require_exact_keys(
        report,
        {
            "schema_version",
            "suite",
            "preset",
            "candidate_provenance",
            "baselines",
            "hardware",
            "changed_beta_cuda_graph_test",
            "measurement",
            "cases",
            "complete_per_arch_denominator",
        },
        "per-architecture receipt",
    )
    if report.get("schema_version") != EVIDENCE_REPORT_SCHEMA_VERSION:
        raise EvidenceSchemaError("unexpected per-architecture receipt schema")
    if report.get("suite") != "recurrent_kda_prefill_h12_phase_a":
        raise EvidenceSchemaError("unexpected per-architecture receipt suite")
    if report.get("complete_per_arch_denominator") is not True:
        raise EvidenceSchemaError(
            f"{expected_arch} receipt is not a complete per-architecture denominator"
        )
    preset_payload = report.get("preset")
    expected_preset_cases = [
        {
            "name": case.name,
            "layout": case.layout,
            "seq_lens": list(case.seq_lens),
            "total_tokens": case.total_tokens,
            "seed": case.seed,
        }
        for case in preset.cases
    ]
    if (
        not isinstance(preset_payload, dict)
        or set(preset_payload)
        != {"name", "path", "sha256", "common", "aggregation", "cases"}
        or preset_payload.get("sha256") != FROZEN_PRESET_SHA256
        or preset_payload.get("name") != preset.name
        or preset_payload.get("common") != preset.common
        or preset_payload.get("aggregation") != "per_case_only"
        or preset_payload.get("cases") != expected_preset_cases
        or not isinstance(preset_payload.get("path"), str)
        or Path(preset_payload["path"]).name != "recurrent_kda_prefill_h12_phase_a.json"
    ):
        raise EvidenceSchemaError("receipt preset identity is not frozen Phase A")

    hardware = report.get("hardware")
    expected_capabilities = {"sm100a": [10, 0], "sm103a": [10, 3]}
    if expected_arch not in expected_capabilities:
        raise EvidenceSchemaError(f"unsupported promotion architecture {expected_arch}")
    if (
        not isinstance(hardware, dict)
        or set(hardware)
        != {
            "device_name",
            "device_index",
            "device_uuid",
            "compute_capability",
            "cuda_arch",
            "multiprocessor_count",
            "total_memory_bytes",
            "l2_cache_bytes",
            "torch_version",
            "torch_cuda_version",
        }
        or hardware.get("cuda_arch") != expected_arch
        or hardware.get("compute_capability") != expected_capabilities[expected_arch]
        or not isinstance(hardware.get("device_name"), str)
        or not hardware["device_name"]
        or not isinstance(hardware.get("device_uuid"), str)
        or not hardware["device_uuid"]
        or type(hardware.get("device_index")) is not int
        or hardware["device_index"] < 0
        or any(
            type(hardware.get(key)) is not int or hardware[key] <= 0
            for key in ("multiprocessor_count", "total_memory_bytes", "l2_cache_bytes")
        )
        or not isinstance(hardware.get("torch_version"), str)
        or not hardware["torch_version"]
        or not isinstance(hardware.get("torch_cuda_version"), str)
        or not hardware["torch_cuda_version"]
    ):
        raise EvidenceSchemaError(
            f"receipt hardware does not match required {expected_arch}"
        )

    measurement = report.get("measurement")
    expected_measurement_keys = set(PHASE_A_MEASUREMENT_CONTRACT) | {
        "cupti_python_version"
    }
    cupti_python_version = (
        measurement.get("cupti_python_version")
        if isinstance(measurement, dict)
        else None
    )
    cupti_major = (
        cupti_python_version.split(".", 1)[0]
        if isinstance(cupti_python_version, str)
        else ""
    )
    if (
        not isinstance(measurement, dict)
        or set(measurement) != expected_measurement_keys
        or any(
            measurement.get(key) != value
            for key, value in PHASE_A_MEASUREMENT_CONTRACT.items()
        )
        or not isinstance(cupti_python_version, str)
        or not cupti_python_version
        or not cupti_major.isdigit()
        or int(cupti_major) < 13
    ):
        raise EvidenceSchemaError("receipt measurement contract is not exact Phase A")
    expected_sample_count = PHASE_A_EXPECTED_SAMPLE_COUNT

    candidate = report.get("candidate_provenance")
    required_ancestors = {
        "phase_a_upstream_main": FLASHINFER_PHASE_A_UPSTREAM_MAIN_REVISION,
        "non_aligned_h12_public_route_pr_4351": FLASHINFER_H12_ROUTE_REVISION,
    }
    if (
        not isinstance(candidate, dict)
        or set(candidate)
        != {
            "repository",
            "source_dir",
            "source_commit",
            "required_ancestor_revisions",
            "worktree_clean_including_untracked",
            "imported_module_paths",
            "imported_module_sha256",
            "source_sha256",
        }
        or candidate.get("repository")
        != "https://github.com/flashinfer-ai/flashinfer.git"
        or not isinstance(candidate.get("source_dir"), str)
        or not Path(candidate["source_dir"]).is_absolute()
        or candidate.get("source_commit") != expected_candidate_commit
        or candidate.get("required_ancestor_revisions") != required_ancestors
        or candidate.get("worktree_clean_including_untracked") is not True
    ):
        raise EvidenceSchemaError("receipt candidate identity is not frozen/clean")
    imported_paths = candidate.get("imported_module_paths")
    imported_hashes = candidate.get("imported_module_sha256")
    source_hashes = candidate.get("source_sha256")
    if (
        not isinstance(imported_paths, dict)
        or set(imported_paths) != set(REQUIRED_CANDIDATE_IMPORTED_MODULES)
        or not isinstance(imported_hashes, dict)
        or set(imported_hashes) != set(REQUIRED_CANDIDATE_IMPORTED_MODULES)
        or not isinstance(source_hashes, dict)
        or set(source_hashes) != set(REQUIRED_CANDIDATE_SOURCE_PATHS)
    ):
        raise EvidenceSchemaError("candidate source/module hash key sets are not exact")
    source_dir = Path(candidate["source_dir"])
    if source_dir != source_dir.resolve(strict=False):
        raise EvidenceSchemaError("candidate source_dir must be a normalized path")
    imported_source_paths = {
        "flashinfer.kda": "flashinfer/kda.py",
        "flashinfer.kda_prefill": "flashinfer/kda_prefill.py",
    }
    for name, path in imported_paths.items():
        expected_path = source_dir / imported_source_paths[name]
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or Path(path) != expected_path
        ):
            raise EvidenceSchemaError(
                f"candidate imported module path {name} is invalid"
            )
    for label, hashes in (
        ("imported_module_sha256", imported_hashes),
        ("source_sha256", source_hashes),
    ):
        for name, digest in hashes.items():
            _require_sha256(digest, f"candidate {label} {name}")
    for module_name, source_path in imported_source_paths.items():
        if imported_hashes[module_name] != source_hashes[source_path]:
            raise EvidenceSchemaError(
                f"candidate imported module {module_name} hash differs from source"
            )

    baselines = report.get("baselines")
    if not isinstance(baselines, dict) or set(baselines) != {
        "flash_kda",
        "fla_triton",
    }:
        raise EvidenceSchemaError("receipt baselines are missing")
    flash_kda = baselines.get("flash_kda")
    expected_flash_kda_keys = {
        "available",
        "required_revision",
        "repository",
        "source_dir",
        "source_commit",
        "cutlass_commit",
        "worktree_clean_including_untracked",
        "package_path",
        "package_sha256",
        "extension_path",
        "extension_sha256",
        "build_manifest_path",
        "build_manifest_sha256",
        "build_manifest",
        "current_receipt_binding",
    }
    if (
        not isinstance(flash_kda, dict)
        or set(flash_kda) != expected_flash_kda_keys
        or flash_kda.get("available") is not True
        or flash_kda.get("required_revision") != FLASH_KDA_BASELINE_REVISION
        or flash_kda.get("repository") != FLASH_KDA_REPOSITORY
        or flash_kda.get("source_commit") != FLASH_KDA_BASELINE_REVISION
        or flash_kda.get("cutlass_commit") != FLASH_KDA_CUTLASS_REVISION
        or flash_kda.get("worktree_clean_including_untracked") is not True
    ):
        raise EvidenceSchemaError("pinned FlashKDA peer is missing or unclean")
    _require_sha256(flash_kda.get("package_sha256"), "FlashKDA package hash")
    _require_sha256(flash_kda.get("extension_sha256"), "FlashKDA extension hash")
    _require_sha256(
        flash_kda.get("build_manifest_sha256"), "FlashKDA build manifest hash"
    )
    build_manifest = validate_flash_kda_build_manifest_schema(
        flash_kda.get("build_manifest")
    )
    if (
        build_manifest["hardware"]["cuda_arch"] != expected_arch
        or flash_kda_build_manifest_sha256(build_manifest)
        != flash_kda["build_manifest_sha256"]
        or build_manifest["artifacts"]["package_sha256"] != flash_kda["package_sha256"]
        or build_manifest["artifacts"]["extension_sha256"]
        != flash_kda["extension_sha256"]
    ):
        raise EvidenceSchemaError(
            "FlashKDA build manifest is not bound to this architecture/artifact"
        )
    flash_source_dir = Path(flash_kda["source_dir"])
    package_path = Path(flash_kda["package_path"])
    extension_path = Path(flash_kda["extension_path"])
    build_manifest_path = Path(flash_kda["build_manifest_path"])
    if (
        not flash_source_dir.is_absolute()
        or flash_source_dir != flash_source_dir.resolve(strict=False)
        or build_manifest["source"]["source_dir"] != str(flash_source_dir)
        or package_path != Path(build_manifest["artifacts"]["package_path"])
        or extension_path != Path(build_manifest["artifacts"]["extension_path"])
        or not package_path.is_relative_to(flash_source_dir)
        or not extension_path.is_relative_to(flash_source_dir)
        or not build_manifest_path.is_absolute()
        or build_manifest_path.is_relative_to(flash_source_dir)
    ):
        raise EvidenceSchemaError(
            "FlashKDA source/artifact/manifest paths are not exactly bound"
        )
    current_binding = flash_kda.get("current_receipt_binding")
    expected_binding_hardware = {
        key: build_manifest["hardware"][key]
        for key in (
            "cuda_arch",
            "compute_capability",
            "device_name",
            "device_uuid",
        )
    }
    expected_binding_runtime = {
        key: build_manifest["toolchain"][key]
        for key in (
            "python_executable",
            "python_version",
            "platform",
            "torch_version",
            "torch_cuda_version",
            "cuda_home",
        )
    }
    if (
        not isinstance(current_binding, dict)
        or set(current_binding)
        != {
            "schema_version",
            "same_slurm_allocation",
            "same_gpu",
            "same_python_torch_cuda_runtime",
            "allocation",
            "hardware",
            "runtime",
        }
        or current_binding.get("schema_version") != 1
        or current_binding.get("same_slurm_allocation") is not True
        or current_binding.get("same_gpu") is not True
        or current_binding.get("same_python_torch_cuda_runtime") is not True
        or current_binding.get("allocation") != build_manifest["allocation"]
        or current_binding.get("hardware") != expected_binding_hardware
        or current_binding.get("runtime") != expected_binding_runtime
    ):
        raise EvidenceSchemaError(
            "FlashKDA build is not bound to the current receipt allocation/GPU/runtime"
        )
    receipt_binding_hardware = {
        key: hardware[key]
        for key in ("cuda_arch", "compute_capability", "device_name", "device_uuid")
    }
    if (
        receipt_binding_hardware != current_binding["hardware"]
        or hardware["torch_version"] != current_binding["runtime"]["torch_version"]
        or hardware["torch_cuda_version"]
        != current_binding["runtime"]["torch_cuda_version"]
    ):
        raise EvidenceSchemaError(
            "receipt hardware/runtime differs from its FlashKDA build binding"
        )

    fla = baselines.get("fla_triton")
    expected_fla_keys = {
        "available",
        "implementation",
        "distribution_version",
        "package_path",
        "package_sha256",
        "op_path",
        "op_sha256",
        "forced_environment",
        "git_source_dir",
        "git_revision",
        "worktree_clean_including_untracked",
    }
    if (
        not isinstance(fla, dict)
        or set(fla) != expected_fla_keys
        or fla.get("available") is not True
        or fla.get("implementation") != "fla.ops.kda.chunk_kda (Triton forced)"
        or fla.get("git_revision") != expected_fla_commit
        or fla.get("worktree_clean_including_untracked") is not True
        or fla.get("forced_environment")
        != {"FLA_FLASH_KDA": "0", "FLA_DISABLE_BACKEND_DISPATCH": "1"}
    ):
        raise EvidenceSchemaError("required FLA/Triton peer is missing or unclean")
    _require_sha256(fla.get("package_sha256"), "FLA package hash")
    _require_sha256(fla.get("op_sha256"), "FLA KDA op hash")
    fla_source_dir = Path(fla["git_source_dir"])
    fla_package_path = Path(fla["package_path"])
    fla_op_path = Path(fla["op_path"])
    distribution_version = fla["distribution_version"]
    if (
        not fla_source_dir.is_absolute()
        or fla_source_dir != fla_source_dir.resolve(strict=False)
        or not fla_package_path.is_absolute()
        or not fla_package_path.is_relative_to(fla_source_dir)
        or fla_package_path.relative_to(fla_source_dir) != Path("fla/__init__.py")
        or not fla_op_path.is_absolute()
        or not fla_op_path.is_relative_to(fla_source_dir)
        or fla_op_path.relative_to(fla_source_dir) != Path("fla/ops/kda/chunk.py")
        or (
            distribution_version is not None
            and (not isinstance(distribution_version, str) or not distribution_version)
        )
    ):
        raise EvidenceSchemaError("FLA source/package/op identity is not exact")

    graph_receipt = report.get("changed_beta_cuda_graph_test")
    _validate_graph_receipt(graph_receipt)
    if (
        graph_receipt["source_sha256"] != source_hashes[GRAPH_TEST_SOURCE]
        or graph_receipt["command"][0]
        != current_binding["runtime"]["python_executable"]
    ):
        raise EvidenceSchemaError(
            "CUDA Graph source/runtime differs from candidate receipt provenance"
        )
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != len(preset.cases):
        raise EvidenceSchemaError("receipt does not contain exactly six cases")
    for case, expected_case in zip(cases, preset.cases, strict=True):
        if not isinstance(case, dict):
            raise EvidenceSchemaError("receipt case must be an object")
        _validate_case_receipt(
            case,
            expected_case,
            expected_sample_count=expected_sample_count,
        )

    return {
        "preset_sha256": FROZEN_PRESET_SHA256,
        "candidate": {
            "source_commit": candidate["source_commit"],
            "required_ancestor_revisions": candidate["required_ancestor_revisions"],
            "imported_module_sha256": candidate["imported_module_sha256"],
            "source_sha256": candidate["source_sha256"],
        },
        "flash_kda": {
            "repository": flash_kda["repository"],
            "source_commit": flash_kda["source_commit"],
            "cutlass_commit": flash_kda["cutlass_commit"],
            "package_sha256": flash_kda["package_sha256"],
        },
        "fla_triton": {
            "git_revision": fla["git_revision"],
            "distribution_version": fla["distribution_version"],
            "package_sha256": fla["package_sha256"],
            "op_sha256": fla["op_sha256"],
        },
        "graph_test_source_sha256": report["changed_beta_cuda_graph_test"][
            "source_sha256"
        ],
        "measurement_contract": dict(measurement),
    }


def reduce_dual_arch_receipts(
    *,
    sm100a_report: dict,
    sm103a_report: dict,
    sm100a_receipt_sha256: str,
    sm103a_receipt_sha256: str,
    expected_candidate_commit: str,
    expected_fla_commit: str,
    preset: EvidencePreset,
) -> dict:
    """Fail closed unless both architecture receipts form one frozen result."""

    identities = {
        "sm100a": validate_per_arch_receipt(
            sm100a_report,
            expected_arch="sm100a",
            expected_candidate_commit=expected_candidate_commit,
            expected_fla_commit=expected_fla_commit,
            preset=preset,
        ),
        "sm103a": validate_per_arch_receipt(
            sm103a_report,
            expected_arch="sm103a",
            expected_candidate_commit=expected_candidate_commit,
            expected_fla_commit=expected_fla_commit,
            preset=preset,
        ),
    }
    if identities["sm100a"] != identities["sm103a"]:
        raise EvidenceSchemaError(
            "SM100a and SM103a receipts do not have exactly matching "
            "preset/candidate/peer/graph identities"
        )
    return {
        "schema_version": DUAL_ARCH_PROMOTION_SCHEMA_VERSION,
        "suite": "recurrent_kda_prefill_h12_phase_a_dual_arch_promotion",
        "frozen_identity": identities["sm100a"],
        "receipts": {
            "sm100a": {
                "sha256": _require_sha256(sm100a_receipt_sha256, "SM100a receipt")
            },
            "sm103a": {
                "sha256": _require_sha256(sm103a_receipt_sha256, "SM103a receipt")
            },
        },
        "cross_shape_aggregate": None,
        "promotion_complete_dual_arch": True,
    }
