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

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


BENCHMARKS_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "kda_h12_evidence",
    BENCHMARKS_DIR / "kda_h12_evidence.py",
)
assert SPEC is not None and SPEC.loader is not None
evidence = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evidence
SPEC.loader.exec_module(evidence)


def _preset_path() -> Path:
    return BENCHMARKS_DIR / "presets" / "recurrent_kda_prefill_h12_phase_a.json"


def _runner_path() -> Path:
    return BENCHMARKS_DIR / "bench_recurrent_kda_prefill_h12_phase_a.py"


def _build_helper_path() -> Path:
    return BENCHMARKS_DIR / "build_flash_kda_phase_a.py"


def _reducer_path() -> Path:
    return BENCHMARKS_DIR / "reduce_kda_h12_phase_a.py"


def test_checked_in_preset_is_exact_and_per_case_only():
    preset = evidence.load_preset(_preset_path())

    assert preset.common == {
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
    assert preset.aggregation == "per_case_only"
    assert [(case.layout, case.seq_lens, case.seed) for case in preset.cases] == [
        ("packed", (512,) * 32, 12000),
        ("packed", (128,) * 8, 12001),
        ("fixed", (512,), 12002),
        ("fixed", (8192,), 12003),
        ("packed", (1300, 547, 2048, 963, 271, 3063), 12004),
        ("packed", (1024,) * 8, 12005),
    ]
    assert len(preset.sha256) == 64


def test_preset_rejects_cross_shape_aggregation(tmp_path):
    payload = (
        _preset_path()
        .read_text()
        .replace(
            '"aggregation": "per_case_only"',
            '"aggregation": "geomean"',
        )
    )
    path = tmp_path / "bad.json"
    path.write_text(payload)

    with pytest.raises(
        evidence.EvidenceSchemaError,
        match="cross-shape aggregation is forbidden",
    ):
        evidence.load_preset(path)


def _flash_kda_paths(tmp_path):
    source_dir = tmp_path / "FlashKDA"
    package_path = source_dir / "flash_kda" / "__init__.py"
    extension_path = source_dir / "flash_kda_C.so"
    cutlass_dir = source_dir / "cutlass"
    package_path.parent.mkdir(parents=True)
    cutlass_dir.mkdir(parents=True)
    package_path.write_text("# pinned package\n")
    extension_path.write_bytes(b"pinned extension")
    return source_dir, package_path, extension_path, cutlass_dir


def _build_manifest(tmp_path, source_dir, package_path, extension_path, arch="sm100a"):
    capability = [10, 0] if arch == "sm100a" else [10, 3]
    payload = {
        "schema_version": evidence.FLASH_KDA_BUILD_MANIFEST_SCHEMA_VERSION,
        "kind": "phase_a_flash_kda_in_allocation_build",
        "source": {
            "repository": evidence.FLASH_KDA_REPOSITORY,
            "source_dir": str(source_dir.resolve()),
            "source_commit": evidence.FLASH_KDA_BASELINE_REVISION,
            "cutlass_commit": evidence.FLASH_KDA_CUTLASS_REVISION,
            "worktree_clean_including_untracked": True,
        },
        "build": {
            "command": [
                "/venv/bin/python",
                "setup.py",
                "build_ext",
                "--inplace",
                "--force",
            ],
            "cwd": str(source_dir.resolve()),
            "environment": {
                "CC": None,
                "CXX": None,
                "CUDA_HOME": None,
                "FLASH_KDA_CUDA_ARCHS": "auto",
                "MAX_JOBS": None,
                "NVCC_PREPEND_FLAGS": None,
                "NVCC_THREADS": "32",
                "TORCH_CUDA_ARCH_LIST": None,
            },
        },
        "toolchain": {
            "python_executable": "/venv/bin/python",
            "python_version": "3.12.0",
            "platform": "Linux",
            "torch_version": "2.8.0",
            "torch_cuda_version": "12.9",
            "cuda_home": "/usr/local/cuda",
            "nvcc_path": "/usr/local/cuda/bin/nvcc",
            "nvcc_version": "Cuda compilation tools, release 12.9",
            "cxx_path": "/usr/bin/c++",
            "cxx_version": "c++ 13",
        },
        "allocation": {
            "slurm_job_id": "1234",
            "slurm_cluster_name": "test",
            "slurm_partition": "gpu",
            "slurm_node_list": "node1",
        },
        "hardware": {
            "cuda_available": True,
            "cuda_arch": arch,
            "compute_capability": capability,
            "device_name": "Blackwell",
            "device_uuid": "GPU-test",
        },
        "artifacts": {
            "package_path": str(package_path.resolve()),
            "package_sha256": evidence._sha256(package_path),
            "extension_path": str(extension_path.resolve()),
            "extension_sha256": evidence._sha256(extension_path),
        },
    }
    path = tmp_path / f"flash-kda-build-{arch}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path, payload


def _current_receipt_binding(build_manifest):
    hardware = {
        key: build_manifest["hardware"][key]
        for key in (
            "cuda_arch",
            "compute_capability",
            "device_name",
            "device_uuid",
        )
    }
    runtime = {
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
    return evidence.verify_flash_kda_current_receipt_binding(
        build_manifest,
        allocation=dict(build_manifest["allocation"]),
        hardware=hardware,
        runtime=runtime,
    )


def _exact_flash_kda_git(source_dir, cutlass_dir, *, source_status=""):
    def exact_revision(root, *args):
        root = Path(root)
        if root == source_dir and args == ("rev-parse", "HEAD"):
            return evidence.FLASH_KDA_BASELINE_REVISION
        if root == cutlass_dir and args == ("rev-parse", "HEAD"):
            return evidence.FLASH_KDA_CUTLASS_REVISION
        if root == source_dir and args == ("ls-tree", "HEAD", "cutlass"):
            return f"160000 commit {evidence.FLASH_KDA_CUTLASS_REVISION}\tcutlass"
        if args == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ):
            return source_status if root == source_dir else ""
        raise AssertionError(f"unexpected git query at {root}: {args!r}")

    return exact_revision


def test_flash_kda_identity_mismatch_is_rejected(tmp_path):
    source_dir, package_path, extension_path, _ = _flash_kda_paths(tmp_path)

    def wrong_revision(root, *args):
        assert Path(root).is_relative_to(source_dir)
        if args == ("rev-parse", "HEAD"):
            return "0" * 40
        raise AssertionError(f"unexpected git query {args!r}")

    with pytest.raises(RuntimeError, match="unexpected FlashKDA source revision"):
        evidence.verify_flash_kda_provenance(
            package_path=package_path,
            extension_path=extension_path,
            source_dir=source_dir,
            build_manifest_path=tmp_path / "not-needed.json",
            git_output=wrong_revision,
        )


def test_flash_kda_exact_identity_is_recorded(tmp_path):
    source_dir, package_path, extension_path, cutlass_dir = _flash_kda_paths(tmp_path)
    manifest_path, _ = _build_manifest(
        tmp_path, source_dir, package_path, extension_path
    )

    provenance = evidence.verify_flash_kda_provenance(
        package_path=package_path,
        extension_path=extension_path,
        source_dir=source_dir,
        build_manifest_path=manifest_path,
        git_output=_exact_flash_kda_git(source_dir, cutlass_dir),
    )

    assert provenance["source_commit"] == evidence.FLASH_KDA_BASELINE_REVISION
    assert provenance["cutlass_commit"] == evidence.FLASH_KDA_CUTLASS_REVISION
    assert len(provenance["package_sha256"]) == 64
    assert len(provenance["extension_sha256"]) == 64
    assert provenance["build_manifest"]["allocation"]["slurm_job_id"] == "1234"


def test_flash_kda_nonignored_untracked_source_is_rejected(tmp_path):
    source_dir, package_path, extension_path, cutlass_dir = _flash_kda_paths(tmp_path)
    manifest_path, _ = _build_manifest(
        tmp_path, source_dir, package_path, extension_path
    )

    with pytest.raises(RuntimeError, match="nonignored untracked"):
        evidence.verify_flash_kda_provenance(
            package_path=package_path,
            extension_path=extension_path,
            source_dir=source_dir,
            build_manifest_path=manifest_path,
            git_output=_exact_flash_kda_git(
                source_dir,
                cutlass_dir,
                source_status="?? rogue_kernel.cu",
            ),
        )


def test_flash_kda_extension_must_match_allocation_build_manifest(tmp_path):
    source_dir, package_path, extension_path, cutlass_dir = _flash_kda_paths(tmp_path)
    manifest_path, _ = _build_manifest(
        tmp_path, source_dir, package_path, extension_path
    )
    extension_path.write_bytes(b"arbitrary stale extension")

    with pytest.raises(RuntimeError, match="extension hash"):
        evidence.verify_flash_kda_provenance(
            package_path=package_path,
            extension_path=extension_path,
            source_dir=source_dir,
            build_manifest_path=manifest_path,
            git_output=_exact_flash_kda_git(source_dir, cutlass_dir),
        )


def test_flash_kda_manifest_schema_validate_only_is_cpu_safe(tmp_path):
    source_dir, package_path, extension_path, _ = _flash_kda_paths(tmp_path)
    manifest_path, payload = _build_manifest(
        tmp_path, source_dir, package_path, extension_path
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(_build_helper_path()),
            "--validate-only",
            "--manifest",
            str(manifest_path),
        ],
        cwd=BENCHMARKS_DIR.parent,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert json.loads(completed.stdout)["validation"] == "schema_only_no_cuda_import"
    payload["allocation"]["slurm_job_id"] = ""
    with pytest.raises(
        evidence.EvidenceSchemaError,
        match="allocation slurm_job_id",
    ):
        evidence.validate_flash_kda_build_manifest_schema(payload)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("FLASH_KDA_CUDA_ARCHS", "all", "CUDA_ARCHS=auto"),
        ("NVCC_THREADS", "0", "positive integer"),
        ("NVCC_PREPEND_FLAGS", "--use_fast_math", "forbids ambient"),
        ("TORCH_CUDA_ARCH_LIST", "9.0", "forbids ambient"),
    ],
)
def test_flash_kda_manifest_rejects_noncanonical_build_environment(
    tmp_path,
    key,
    value,
    message,
):
    source_dir, package_path, extension_path, _ = _flash_kda_paths(tmp_path)
    _, payload = _build_manifest(tmp_path, source_dir, package_path, extension_path)
    payload["build"]["environment"][key] = value

    with pytest.raises(evidence.EvidenceSchemaError, match=message):
        evidence.validate_flash_kda_build_manifest_schema(payload)


def test_flash_kda_manifest_rejects_prefixed_build_command(tmp_path):
    source_dir, package_path, extension_path, _ = _flash_kda_paths(tmp_path)
    _, payload = _build_manifest(tmp_path, source_dir, package_path, extension_path)
    payload["build"]["command"].insert(1, "-I")

    with pytest.raises(evidence.EvidenceSchemaError, match="must be exactly"):
        evidence.validate_flash_kda_build_manifest_schema(payload)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("allocation", "slurm_job_id", "stale-job", "Slurm allocation"),
        ("hardware", "device_uuid", "GPU-other", "current GPU"),
        ("runtime", "torch_version", "9.9.9", "current receipt runtime"),
    ],
)
def test_flash_kda_current_receipt_binding_rejects_stale_identity(
    tmp_path,
    section,
    key,
    value,
    message,
):
    source_dir, package_path, extension_path, _ = _flash_kda_paths(tmp_path)
    _, manifest = _build_manifest(
        tmp_path,
        source_dir,
        package_path,
        extension_path,
    )
    allocation = dict(manifest["allocation"])
    hardware = {
        name: manifest["hardware"][name]
        for name in (
            "cuda_arch",
            "compute_capability",
            "device_name",
            "device_uuid",
        )
    }
    runtime = {
        name: manifest["toolchain"][name]
        for name in (
            "python_executable",
            "python_version",
            "platform",
            "torch_version",
            "torch_cuda_version",
            "cuda_home",
        )
    }
    current = {
        "allocation": allocation,
        "hardware": hardware,
        "runtime": runtime,
    }
    current[section][key] = value
    with pytest.raises(evidence.EvidenceSchemaError, match=message):
        evidence.verify_flash_kda_current_receipt_binding(
            manifest,
            allocation=allocation,
            hardware=hardware,
            runtime=runtime,
        )


def test_flash_kda_cutlass_identity_mismatch_is_rejected(tmp_path):
    source_dir, package_path, extension_path, cutlass_dir = _flash_kda_paths(tmp_path)

    def wrong_cutlass(root, *args):
        root = Path(root)
        if root == source_dir and args == ("rev-parse", "HEAD"):
            return evidence.FLASH_KDA_BASELINE_REVISION
        if root == cutlass_dir and args == ("rev-parse", "HEAD"):
            return "0" * 40
        raise AssertionError(f"unexpected git query at {root}: {args!r}")

    with pytest.raises(RuntimeError, match="unexpected FlashKDA CUTLASS revision"):
        evidence.verify_flash_kda_provenance(
            package_path=package_path,
            extension_path=extension_path,
            source_dir=source_dir,
            build_manifest_path=tmp_path / "not-needed.json",
            git_output=wrong_cutlass,
        )


def test_h12_public_activity_reduction_preserves_span_decomposition_and_names():
    brackets = [evidence.CpuBracket(100, 250, 1000)]
    launches = [
        evidence.LaunchActivity(110, 120, 7, "runtime", "runtime:cudaLaunchKernel"),
        evidence.LaunchActivity(130, 140, 8, "runtime", "runtime:cudaLaunchKernel"),
        evidence.LaunchActivity(115, 125, 7, "driver", "driver:cuLaunchKernel"),
        evidence.LaunchActivity(135, 145, 8, "driver", "driver:cuLaunchKernel"),
        evidence.LaunchActivity(300, 310, 9, "runtime", "runtime:outside"),
    ]
    activities = [
        evidence.GpuActivity(
            300,
            400,
            7,
            "kernel",
            "void PackBetaForTmaKernel<bf16>",
        ),
        evidence.GpuActivity(
            450,
            900,
            8,
            "kernel",
            "kernel_flashkda_bf16_fused_m128",
        ),
        evidence.GpuActivity(320, 330, 9, "kernel", "outside"),
    ]

    samples = evidence.correlate_samples(
        brackets=brackets,
        launches=launches,
        activities=activities,
        require_h12_public_route=True,
    )
    report = evidence.summarize_samples(
        samples,
        require_h12_public_route=True,
    )

    assert report["gpu_span_ms_samples"] == [0.0006]
    assert report["kernel_sum_ms_samples"] == [0.00055]
    assert report["active_union_ms_samples"] == [0.00055]
    assert report["inter_kernel_gap_ms_samples"] == [0.00005]
    assert report["submission_ms_samples"] == [0.00015]
    assert report["synchronized_e2e_ms_samples"] == [0.0009]
    assert report["launch_activity_count_samples"] == [4]
    assert [
        launch["correlation_id"]
        for launch in report["launch_activity_order_samples"][0]
    ] == [7, 7, 8, 8]
    assert report["kernel_activity_count_samples"] == [2]
    assert report["kernel_activity_names_samples"] == [
        [
            "void PackBetaForTmaKernel<bf16>",
            "kernel_flashkda_bf16_fused_m128",
        ]
    ]
    prepared = report["prepared_recurrence"]
    assert report["call_path"] == "flashinfer.kda.recurrent_kda"
    assert report["includes_beta_preparation"] is True
    assert prepared["derived_from_same_public_samples"] is True
    assert prepared["includes_beta_pack"] is False
    assert prepared["gpu_span_ms_samples"] == [0.00045]
    assert prepared["launch_activity_count_samples"] == [2]
    assert prepared["launch_activity_names_samples"] == [
        ["runtime:cudaLaunchKernel", "driver:cuLaunchKernel"]
    ]
    assert prepared["kernel_activity_count_samples"] == [1]


def test_h12_public_activity_reduction_rejects_missing_pack():
    with pytest.raises(
        evidence.EvidenceSchemaError,
        match="exactly one beta-pack",
    ):
        evidence.correlate_samples(
            brackets=[evidence.CpuBracket(100, 200, 600)],
            launches=[
                evidence.LaunchActivity(
                    110,
                    120,
                    1,
                    "runtime",
                    "runtime:cudaLaunchKernel",
                )
            ],
            activities=[
                evidence.GpuActivity(
                    300,
                    500,
                    1,
                    "kernel",
                    "kernel_flashkda_bf16_fused_m128",
                )
            ],
            require_h12_public_route=True,
        )


def test_runner_validate_only_is_cpu_safe_and_reports_frozen_identities():
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(_runner_path()), "--validate-only"],
        cwd=BENCHMARKS_DIR.parent,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["gpu_execution"] == "not_requested"
    assert payload["flash_kda_required_revision"] == (
        "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
    )
    assert payload["flashinfer_required_ancestor_revisions"] == {
        "phase_a_upstream_main": "2ab910c58fdd2392914ea05e2a8714946ac0eef6",
        "non_aligned_h12_public_route_pr_4351": (
            "38bf507f9c9eba6b4544bee016d2bdf9c4fed02b"
        ),
    }
    assert len(payload["preset"]["cases"]) == 6
    assert payload["preset"]["aggregation"] == "per_case_only"
    assert payload["fla_triton"]["required"] is True
    assert payload["flash_kda_build_manifest"] == {
        "required": True,
        "schema_version": evidence.FLASH_KDA_BUILD_MANIFEST_SCHEMA_VERSION,
        "helper": "benchmarks/build_flash_kda_phase_a.py",
        "requires_slurm_gpu_allocation": True,
        "requires_force_rebuild": True,
    }
    assert payload["changed_beta_cuda_graph_test"]["source_line_range"] == [
        981,
        1042,
    ]
    assert payload["changed_beta_cuda_graph_test"]["parameterization"] == {
        "num_heads": [6, 12]
    }
    assert payload["promotion"] == {
        "per_arch_flag": "complete_per_arch_denominator",
        "required_architectures": ["sm100a", "sm103a"],
        "reducer": "benchmarks/reduce_kda_h12_phase_a.py",
        "dual_arch_flag": "promotion_complete_dual_arch",
    }


def _timing_receipt(name, *, num_sequences):
    base_order = list(evidence.REQUIRED_TIMING_PATHS)
    raw_samples = []
    for block_index in range(evidence.PHASE_A_MEASUREMENT_CONTRACT["blocks"]):
        block_order = base_order if block_index % 2 == 0 else list(reversed(base_order))
        order_index = block_order.index(name)
        for sample_index in range(
            evidence.PHASE_A_MEASUREMENT_CONTRACT["repeat_iters_per_block"]
        ):
            base = (block_index * 100 + sample_index) * 100_000
            launches = [
                evidence.LaunchActivity(
                    start_ns=base + 1_000,
                    end_ns=base + 1_100,
                    correlation_id=1,
                    kind="runtime",
                    name="runtime:cbid=13",
                )
            ]
            if name == "flashinfer_public":
                launches.append(
                    evidence.LaunchActivity(
                        start_ns=base + 2_000,
                        end_ns=base + 2_100,
                        correlation_id=2,
                        kind="runtime",
                        name="runtime:cbid=13",
                    )
                )
                activities = [
                    evidence.GpuActivity(
                        start_ns=base + 6_000,
                        end_ns=base + 6_500,
                        correlation_id=1,
                        kind="kernel",
                        name="void PackBetaForTmaKernel<bf16>",
                    ),
                    evidence.GpuActivity(
                        start_ns=base + 7_000,
                        end_ns=base + 9_000,
                        correlation_id=2,
                        kind="kernel",
                        name="kernel_flashkda_bf16_fused_m128",
                    ),
                ]
            elif name == "flash_kda_public_semantics_adapted":
                launches.append(
                    evidence.LaunchActivity(
                        start_ns=base + 2_000,
                        end_ns=base + 2_100,
                        correlation_id=2,
                        kind="runtime",
                        name="runtime:cbid=41",
                    )
                )
                activities = [
                    evidence.GpuActivity(
                        start_ns=base + 6_000,
                        end_ns=base + 8_000,
                        correlation_id=1,
                        kind="kernel",
                        name="kernel_flashkda_bf16_fused_m128",
                    ),
                    evidence.GpuActivity(
                        start_ns=base + 8_200,
                        end_ns=base + 8_700,
                        correlation_id=2,
                        kind="memcpy",
                        name=(
                            "MEMCPY(copy_kind=8,bytes="
                            f"{num_sequences * 12 * 128 * 128 * 2})"
                        ),
                    ),
                ]
            else:
                kernel_name = (
                    "kernel_flashkda_bf16_fused_m128"
                    if name == "flash_kda_raw"
                    else "triton_red_fused_kda"
                )
                activities = [
                    evidence.GpuActivity(
                        start_ns=base + 6_000,
                        end_ns=base + 8_000,
                        correlation_id=1,
                        kind="kernel",
                        name=kernel_name,
                    )
                ]
            sample = evidence._correlated_sample(
                sample_index=sample_index,
                bracket=evidence.CpuBracket(
                    start_ns=base,
                    submitted_ns=base + 4_000,
                    synchronized_ns=base + 12_000,
                ),
                launches=launches,
                activities=activities,
                require_h12_public_route=name == "flashinfer_public",
            )
            sample["block_index"] = block_index
            sample["order_index"] = order_index
            raw_samples.append(sample)
    timing = evidence.summarize_samples(
        raw_samples,
        require_h12_public_route=name == "flashinfer_public",
    )
    timing["call_path"] = evidence.REQUIRED_TIMING_CALL_PATHS[name]
    return timing


def _complete_per_arch_report(tmp_path, arch):
    preset = evidence.load_preset(_preset_path())
    source_dir, package_path, extension_path, _ = _flash_kda_paths(tmp_path / arch)
    build_manifest_path, build_manifest = _build_manifest(
        tmp_path / arch,
        source_dir,
        package_path,
        extension_path,
        arch=arch,
    )
    candidate_commit = "1" * 40
    fla_commit = "2" * 40
    cases = []
    for case in preset.cases:
        oracle_result = {
            "output": {
                "passed": True,
                "max_abs": 0.0,
                "max_allowed_abs": 0.01,
                "mismatch_count": 0,
                "atol": 0.01,
                "rtol": 0.01,
                "compared_dtype": "bfloat16",
                "compared_numel": case.total_tokens * 12 * 128,
            },
            "final_state": {
                "passed": True,
                "max_abs": 0.0,
                "max_allowed_abs": 0.01,
                "mismatch_count": 0,
                "atol": 0.01,
                "rtol": 0.01,
                "compared_dtype": "bfloat16",
                "compared_numel": len(case.seq_lens) * 12 * 128 * 128,
            },
        }
        timings = {
            name: _timing_receipt(name, num_sequences=len(case.seq_lens))
            for name in evidence.REQUIRED_TIMING_PATHS
        }
        base_order = list(evidence.REQUIRED_TIMING_PATHS)
        measurement_order = []
        for block_index in range(evidence.PHASE_A_MEASUREMENT_CONTRACT["blocks"]):
            block_order = (
                base_order if block_index % 2 == 0 else list(reversed(base_order))
            )
            measurement_order.extend(
                {
                    "block_index": block_index,
                    "order_index": order_index,
                    "path": name,
                }
                for order_index, name in enumerate(block_order)
            )
        candidate_ms = timings["flashinfer_public"]["median_gpu_span_ms"]
        cases.append(
            {
                "name": case.name,
                "layout": case.layout,
                "seq_lens": list(case.seq_lens),
                "total_tokens": case.total_tokens,
                "num_sequences": len(case.seq_lens),
                "seed": case.seed,
                "num_heads": 12,
                "head_dim_qk": 128,
                "head_dim_vo": 128,
                "dtype": "bfloat16",
                "initial_state": "provided_bfloat16",
                "variant": "m128",
                "correctness": {
                    "passed": True,
                    "public_output_and_full_final_state": True,
                    "independent_bf16_recurrence": json.loads(
                        json.dumps(oracle_result)
                    ),
                    "pinned_flash_kda": json.loads(json.dumps(oracle_result)),
                    "fla_triton": json.loads(json.dumps(oracle_result)),
                },
                "timings": timings,
                "measurement_order": measurement_order,
                "per_case_speedups": {
                    "vs_pinned_flash_kda_raw": (
                        timings["flash_kda_raw"]["median_gpu_span_ms"] / candidate_ms
                    ),
                    "vs_pinned_flash_kda_public_semantics_adapted": (
                        timings["flash_kda_public_semantics_adapted"][
                            "median_gpu_span_ms"
                        ]
                        / candidate_ms
                    ),
                    "vs_fla_triton": (
                        timings["fla_triton"]["median_gpu_span_ms"] / candidate_ms
                    ),
                },
                "cross_shape_aggregate": None,
            }
        )
    capability = [10, 0] if arch == "sm100a" else [10, 3]
    report = {
        "schema_version": evidence.EVIDENCE_REPORT_SCHEMA_VERSION,
        "suite": "recurrent_kda_prefill_h12_phase_a",
        "preset": {
            "name": preset.name,
            "path": preset.path,
            "sha256": preset.sha256,
            "common": preset.common,
            "aggregation": "per_case_only",
            "cases": [
                {
                    "name": case.name,
                    "layout": case.layout,
                    "seq_lens": list(case.seq_lens),
                    "total_tokens": case.total_tokens,
                    "seed": case.seed,
                }
                for case in preset.cases
            ],
        },
        "candidate_provenance": {
            "repository": "https://github.com/flashinfer-ai/flashinfer.git",
            "source_dir": "/src/flashinfer",
            "source_commit": candidate_commit,
            "required_ancestor_revisions": {
                "phase_a_upstream_main": (
                    evidence.FLASHINFER_PHASE_A_UPSTREAM_MAIN_REVISION
                ),
                "non_aligned_h12_public_route_pr_4351": (
                    evidence.FLASHINFER_H12_ROUTE_REVISION
                ),
            },
            "worktree_clean_including_untracked": True,
            "imported_module_paths": {
                "flashinfer.kda": "/src/flashinfer/flashinfer/kda.py",
                "flashinfer.kda_prefill": ("/src/flashinfer/flashinfer/kda_prefill.py"),
            },
            "imported_module_sha256": {
                name: "b" * 64 for name in evidence.REQUIRED_CANDIDATE_IMPORTED_MODULES
            },
            "source_sha256": {
                path: "b" * 64 for path in evidence.REQUIRED_CANDIDATE_SOURCE_PATHS
            },
        },
        "baselines": {
            "flash_kda": {
                "available": True,
                "required_revision": evidence.FLASH_KDA_BASELINE_REVISION,
                "repository": evidence.FLASH_KDA_REPOSITORY,
                "source_dir": build_manifest["source"]["source_dir"],
                "source_commit": evidence.FLASH_KDA_BASELINE_REVISION,
                "cutlass_commit": evidence.FLASH_KDA_CUTLASS_REVISION,
                "worktree_clean_including_untracked": True,
                "package_path": build_manifest["artifacts"]["package_path"],
                "package_sha256": build_manifest["artifacts"]["package_sha256"],
                "extension_path": build_manifest["artifacts"]["extension_path"],
                "extension_sha256": build_manifest["artifacts"]["extension_sha256"],
                "build_manifest_path": str(build_manifest_path.resolve()),
                "build_manifest_sha256": (
                    evidence.flash_kda_build_manifest_sha256(build_manifest)
                ),
                "build_manifest": build_manifest,
                "current_receipt_binding": _current_receipt_binding(build_manifest),
            },
            "fla_triton": {
                "available": True,
                "implementation": "fla.ops.kda.chunk_kda (Triton forced)",
                "distribution_version": "0.3.2",
                "package_path": str(
                    (tmp_path / arch / "fla" / "fla" / "__init__.py").resolve()
                ),
                "package_sha256": "d" * 64,
                "op_path": str(
                    (
                        tmp_path / arch / "fla" / "fla" / "ops" / "kda" / "chunk.py"
                    ).resolve()
                ),
                "op_sha256": "e" * 64,
                "git_source_dir": str((tmp_path / arch / "fla").resolve()),
                "git_revision": fla_commit,
                "worktree_clean_including_untracked": True,
                "forced_environment": {
                    "FLA_FLASH_KDA": "0",
                    "FLA_DISABLE_BACKEND_DISPATCH": "1",
                },
            },
        },
        "hardware": {
            "device_name": build_manifest["hardware"]["device_name"],
            "device_index": 0,
            "device_uuid": build_manifest["hardware"]["device_uuid"],
            "compute_capability": capability,
            "cuda_arch": arch,
            "multiprocessor_count": 100,
            "total_memory_bytes": 1_000_000,
            "l2_cache_bytes": 100_000,
            "torch_version": build_manifest["toolchain"]["torch_version"],
            "torch_cuda_version": build_manifest["toolchain"]["torch_cuda_version"],
        },
        "measurement": {
            **evidence.PHASE_A_MEASUREMENT_CONTRACT,
            "cupti_python_version": "13.0.0",
        },
        "changed_beta_cuda_graph_test": {
            "source": evidence.GRAPH_TEST_SOURCE,
            "source_line_range": list(evidence.GRAPH_TEST_SOURCE_LINE_RANGE),
            "source_sha256": "b" * 64,
            "node_id": evidence.GRAPH_TEST_NODE_ID,
            "parameterization": {"num_heads": [6, 12]},
            "command": [
                "/venv/bin/python",
                "-m",
                "pytest",
                "-q",
                evidence.GRAPH_TEST_NODE_ID,
            ],
            "returncode": 0,
            "stdout": "2 passed\n",
            "stderr": "",
            "passed": True,
        },
        "cases": cases,
        "complete_per_arch_denominator": True,
    }
    return report, candidate_commit, fla_commit, preset


def test_dual_arch_reducer_requires_matching_complete_receipts(tmp_path):
    sm100a, candidate_commit, fla_commit, preset = _complete_per_arch_report(
        tmp_path, "sm100a"
    )
    sm103a, _, _, _ = _complete_per_arch_report(tmp_path, "sm103a")

    result = evidence.reduce_dual_arch_receipts(
        sm100a_report=sm100a,
        sm103a_report=sm103a,
        sm100a_receipt_sha256="1" * 64,
        sm103a_receipt_sha256="2" * 64,
        expected_candidate_commit=candidate_commit,
        expected_fla_commit=fla_commit,
        preset=preset,
    )

    assert result["promotion_complete_dual_arch"] is True
    assert result["cross_shape_aggregate"] is None
    assert set(result["receipts"]) == {"sm100a", "sm103a"}


def test_dual_arch_reducer_cli_emits_only_after_both_receipts(tmp_path):
    sm100a, candidate_commit, fla_commit, _ = _complete_per_arch_report(
        tmp_path, "sm100a"
    )
    sm103a, _, _, _ = _complete_per_arch_report(tmp_path, "sm103a")
    sm100a_path = tmp_path / "sm100a.json"
    sm103a_path = tmp_path / "sm103a.json"
    result_path = tmp_path / "dual-arch.json"
    sm100a_path.write_text(json.dumps(sm100a))
    sm103a_path.write_text(json.dumps(sm103a))

    completed = subprocess.run(
        [
            sys.executable,
            str(_reducer_path()),
            "--sm100a",
            str(sm100a_path),
            "--sm103a",
            str(sm103a_path),
            "--expected-flashinfer-commit",
            candidate_commit,
            "--expected-fla-commit",
            fla_commit,
            "--json",
            str(result_path),
        ],
        cwd=BENCHMARKS_DIR.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert json.loads(completed.stdout)["promotion_complete_dual_arch"] is True
    assert json.loads(result_path.read_text())["promotion_complete_dual_arch"] is True


def test_dual_arch_reducer_rejects_missing_oracle_graph_or_identity(tmp_path):
    sm100a, candidate_commit, fla_commit, preset = _complete_per_arch_report(
        tmp_path, "sm100a"
    )
    sm103a, _, _, _ = _complete_per_arch_report(tmp_path, "sm103a")

    missing_oracle = json.loads(json.dumps(sm100a))
    del missing_oracle["cases"][0]["correctness"]["fla_triton"]
    with pytest.raises(evidence.EvidenceSchemaError, match="correctness keys"):
        evidence.validate_per_arch_receipt(
            missing_oracle,
            expected_arch="sm100a",
            expected_candidate_commit=candidate_commit,
            expected_fla_commit=fla_commit,
            preset=preset,
        )

    failed_graph = json.loads(json.dumps(sm100a))
    failed_graph["changed_beta_cuda_graph_test"]["passed"] = False
    with pytest.raises(evidence.EvidenceSchemaError, match="Graph test did not pass"):
        evidence.validate_per_arch_receipt(
            failed_graph,
            expected_arch="sm100a",
            expected_candidate_commit=candidate_commit,
            expected_fla_commit=fla_commit,
            preset=preset,
        )

    stale_build = json.loads(json.dumps(sm100a))
    stale_build["baselines"]["flash_kda"]["current_receipt_binding"][
        "same_slurm_allocation"
    ] = False
    with pytest.raises(
        evidence.EvidenceSchemaError,
        match="current receipt allocation/GPU/runtime",
    ):
        evidence.validate_per_arch_receipt(
            stale_build,
            expected_arch="sm100a",
            expected_candidate_commit=candidate_commit,
            expected_fla_commit=fla_commit,
            preset=preset,
        )

    sm103a["candidate_provenance"]["source_sha256"]["flashinfer/kda.py"] = "0" * 64
    sm103a["candidate_provenance"]["imported_module_sha256"]["flashinfer.kda"] = (
        "0" * 64
    )
    with pytest.raises(evidence.EvidenceSchemaError, match="exactly matching"):
        evidence.reduce_dual_arch_receipts(
            sm100a_report=sm100a,
            sm103a_report=sm103a,
            sm100a_receipt_sha256="1" * 64,
            sm103a_receipt_sha256="2" * 64,
            expected_candidate_commit=candidate_commit,
            expected_fla_commit=fla_commit,
            preset=preset,
        )


def _refresh_timing_summary_arrays(timing):
    raw_samples = timing["raw_samples"]
    timing.update(
        evidence._summarize_numeric(raw_samples, evidence._PUBLIC_NUMERIC_FIELDS)
    )
    for summary_key, sample_key in (
        ("launch_activity_names_samples", "launch_activity_names"),
        ("launch_activity_order_samples", "launch_activity_order"),
        ("gpu_activity_names_samples", "gpu_activity_names"),
        ("kernel_activity_names_samples", "kernel_activity_names"),
        ("copy_activity_names_samples", "copy_activity_names"),
        ("activity_order_samples", "activity_order"),
    ):
        timing[summary_key] = [sample[sample_key] for sample in raw_samples]


def _rewrite_first_case_kernel(payload, path_name, kernel_name):
    timing = payload["cases"][0]["timings"][path_name]
    for sample in timing["raw_samples"]:
        for record in sample["activity_order"]:
            if record["kind"] == "kernel":
                record["name"] = kernel_name
        sample["gpu_activity_names"] = [
            record["name"] for record in sample["activity_order"]
        ]
        sample["kernel_activity_names"] = [
            record["name"]
            for record in sample["activity_order"]
            if record["kind"] == "kernel"
        ]
    _refresh_timing_summary_arrays(timing)


def _rewrite_first_adapted_copy(payload, *, kind, name):
    timing = payload["cases"][0]["timings"]["flash_kda_public_semantics_adapted"]
    for sample in timing["raw_samples"]:
        copy = next(
            record
            for record in sample["activity_order"]
            if record["kind"] in {"memcpy", "memset"}
        )
        copy["kind"] = kind
        copy["name"] = name
        sample["gpu_activity_names"] = [
            record["name"] for record in sample["activity_order"]
        ]
        sample["copy_activity_names"] = [name]
    _refresh_timing_summary_arrays(timing)


def _overlap_first_public_pack_and_recurrence(payload):
    timing = payload["cases"][0]["timings"]["flashinfer_public"]
    sample = timing["raw_samples"][0]
    pack, recurrence = sample["activity_order"]
    pack["end_ns"] = recurrence["start_ns"] + 1
    pack["duration_ms"] = (pack["end_ns"] - pack["start_ns"]) / 1e6
    metrics = evidence._interval_metrics(
        [
            evidence.GpuActivity(
                start_ns=record["start_ns"],
                end_ns=record["end_ns"],
                correlation_id=record["correlation_id"],
                kind=record["kind"],
                name=record["name"],
            )
            for record in sample["activity_order"]
        ]
    )
    sample.update(metrics)
    _refresh_timing_summary_arrays(timing)


def _overlap_first_adapted_copy(payload):
    timing = payload["cases"][0]["timings"]["flash_kda_public_semantics_adapted"]
    sample = timing["raw_samples"][0]
    recurrence, copy = sample["activity_order"]
    copy["start_ns"] = recurrence["end_ns"] - 1
    copy["duration_ms"] = (copy["end_ns"] - copy["start_ns"]) / 1e6
    metrics = evidence._interval_metrics(
        [
            evidence.GpuActivity(
                start_ns=record["start_ns"],
                end_ns=record["end_ns"],
                correlation_id=record["correlation_id"],
                kind=record["kind"],
                name=record["name"],
            )
            for record in sample["activity_order"]
        ]
    )
    sample.update(metrics)
    _refresh_timing_summary_arrays(timing)


def test_per_arch_reducer_rejects_phase_a_contract_mutations(tmp_path):
    report, candidate_commit, fla_commit, preset = _complete_per_arch_report(
        tmp_path, "sm100a"
    )

    def reject(mutator, message):
        mutated = json.loads(json.dumps(report))
        mutator(mutated)
        with pytest.raises(evidence.EvidenceSchemaError, match=message):
            evidence.validate_per_arch_receipt(
                mutated,
                expected_arch="sm100a",
                expected_candidate_commit=candidate_commit,
                expected_fla_commit=fla_commit,
                preset=preset,
            )

    reject(
        lambda payload: payload["measurement"].__setitem__("warmup_iters_per_block", 1),
        "not exact Phase A",
    )
    reject(
        lambda payload: payload["measurement"].__setitem__(
            "cupti_python_version", "not-a-version"
        ),
        "not exact Phase A",
    )
    reject(
        lambda payload: payload["cases"][0]["correctness"][
            "independent_bf16_recurrence"
        ]["output"].__setitem__("compared_dtype", "float32"),
        "exact BF16 full-tensor",
    )
    reject(
        lambda payload: payload["cases"][0]["correctness"]["pinned_flash_kda"][
            "final_state"
        ].__setitem__("compared_numel", 1),
        "exact BF16 full-tensor",
    )
    reject(
        lambda payload: payload["cases"][0]["correctness"]["fla_triton"][
            "output"
        ].__setitem__("atol", 999.0),
        "exact BF16 full-tensor",
    )
    reject(
        lambda payload: payload["cases"][0]["correctness"][
            "independent_bf16_recurrence"
        ]["output"].__setitem__("mismatch_count", 1),
        "exact BF16 full-tensor",
    )
    reject(
        lambda payload: payload["cases"][0]["correctness"][
            "independent_bf16_recurrence"
        ]["output"].__setitem__("max_abs", 1e30),
        "exact BF16 full-tensor",
    )
    reject(
        lambda payload: payload["cases"][0]["timings"]["flash_kda_raw"]["raw_samples"][
            0
        ].pop("activity_order"),
        "keys must be exactly",
    )
    reject(
        lambda payload: payload["cases"][0]["timings"]["fla_triton"].__setitem__(
            "median_gpu_span_ms", 99.0
        ),
        "medians differ",
    )
    reject(
        lambda payload: payload["cases"][0]["timings"]["flashinfer_public"].__setitem__(
            "cold_l2", False
        ),
        "timing scope is not exact",
    )
    reject(
        lambda payload: payload["cases"][0]["timings"]["flash_kda_raw"]["raw_samples"][
            0
        ].__setitem__("block_index", 1),
        "sample/block/order identity",
    )
    reject(
        lambda payload: payload["cases"][0]["timings"][
            "flash_kda_public_semantics_adapted"
        ]["raw_samples"][0].__setitem__("copy_activity_count", 0),
        "activity counts/names",
    )
    reject(
        lambda payload: _rewrite_first_case_kernel(
            payload,
            "flash_kda_raw",
            "totally_unrelated_kernel",
        ),
        "exact pinned FlashKDA recurrence",
    )
    reject(
        lambda payload: _rewrite_first_case_kernel(
            payload,
            "fla_triton",
            "totally_unrelated_kernel",
        ),
        "identifiable FLA KDA kernel",
    )
    reject(
        lambda payload: _rewrite_first_adapted_copy(
            payload,
            kind="memcpy",
            name="MEMCPY(copy_kind=8,bytes=1)",
        ),
        "exact post-recurrence full-state D2D copy-back",
    )
    reject(
        _overlap_first_adapted_copy,
        "exact post-recurrence full-state D2D copy-back",
    )
    reject(
        lambda payload: _rewrite_first_adapted_copy(
            payload,
            kind="memset",
            name="MEMSET(value=0,bytes=1)",
        ),
        "exact post-recurrence full-state D2D copy-back",
    )
    reject(
        _overlap_first_public_pack_and_recurrence,
        "exact nonoverlapping pack-to-recurrence",
    )
    reject(
        lambda payload: payload["cases"][0]["timings"]["flash_kda_raw"]["raw_samples"][
            0
        ]["launch_activity_order"][0].__setitem__("correlation_id", 99),
        "activity counts/names are inconsistent",
    )
    reject(
        lambda payload: payload["cases"][0]["measurement_order"].reverse(),
        "measurement order",
    )
    reject(
        lambda payload: payload["preset"]["common"].__setitem__("beta_is_logit", False),
        "preset identity",
    )
    reject(
        lambda payload: payload["cases"][0].__setitem__(
            "initial_state", "provided_float32"
        ),
        "initial_state must be",
    )
    reject(
        lambda payload: payload["candidate_provenance"]["source_sha256"].pop(
            "flashinfer/kda.py"
        ),
        "hash key sets",
    )
    reject(
        lambda payload: payload["candidate_provenance"][
            "imported_module_paths"
        ].__setitem__("flashinfer.kda", "/src/flashinfer/not_the_source/evil_kda.py"),
        "imported module path",
    )
    reject(
        lambda payload: payload["candidate_provenance"][
            "imported_module_sha256"
        ].__setitem__("flashinfer.kda", "c" * 64),
        "hash differs from source",
    )
    reject(
        lambda payload: payload["changed_beta_cuda_graph_test"].__setitem__(
            "source_sha256", "0" * 64
        ),
        "differs from candidate",
    )
    reject(
        lambda payload: payload["changed_beta_cuda_graph_test"].__setitem__(
            "stdout", ""
        ),
        "process evidence is malformed",
    )
    reject(
        lambda payload: payload["changed_beta_cuda_graph_test"]["command"].__setitem__(
            0, "/different/python"
        ),
        "source/runtime differs",
    )
    reject(
        lambda payload: payload["hardware"].__setitem__(
            "device_uuid", "GPU-not-the-build-device"
        ),
        "hardware/runtime differs",
    )
    reject(
        lambda payload: payload["hardware"].__setitem__("torch_version", "0.fake"),
        "hardware/runtime differs",
    )
    reject(
        lambda payload: payload["baselines"]["flash_kda"].__setitem__(
            "unexpected", True
        ),
        "pinned FlashKDA peer",
    )
    reject(
        lambda payload: payload["baselines"]["flash_kda"].__setitem__(
            "repository", "https://example.invalid/fake.git"
        ),
        "pinned FlashKDA peer",
    )
    reject(
        lambda payload: payload["baselines"]["fla_triton"].__setitem__(
            "package_path", "/outside/fla/__init__.py"
        ),
        "FLA source/package/op identity",
    )
    reject(
        lambda payload: payload["cases"][0].__setitem__(
            "cross_shape_aggregate", {"geomean": 2.0}
        ),
        "forbidden cross-shape",
    )
    reject(
        lambda payload: payload.__setitem__("cross_shape_geomean", 2.0),
        "keys must be exactly",
    )


def test_dual_arch_reducer_rejects_different_cupti_contract_identity(tmp_path):
    sm100a, candidate_commit, fla_commit, preset = _complete_per_arch_report(
        tmp_path, "sm100a"
    )
    sm103a, _, _, _ = _complete_per_arch_report(tmp_path, "sm103a")
    sm103a["measurement"]["cupti_python_version"] = "13.1.0"

    with pytest.raises(evidence.EvidenceSchemaError, match="exactly matching"):
        evidence.reduce_dual_arch_receipts(
            sm100a_report=sm100a,
            sm103a_report=sm103a,
            sm100a_receipt_sha256="1" * 64,
            sm103a_receipt_sha256="2" * 64,
            expected_candidate_commit=candidate_commit,
            expected_fla_commit=fla_commit,
            preset=preset,
        )


def test_runner_rejects_non_exact_phase_a_sampling_without_gpu_import(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(_runner_path()),
            "--flash-kda-source-dir",
            str(tmp_path / "flash-kda"),
            "--flash-kda-build-manifest",
            str(tmp_path / "manifest.json"),
            "--warmup-iters",
            "1",
            "--json",
            str(tmp_path / "receipt.json"),
        ],
        cwd=BENCHMARKS_DIR.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    assert "exact --warmup-iters 5 --repeat-iters 20 --blocks 2" in completed.stderr


def test_runner_has_no_reportable_timer_fallback_or_top_level_gpu_import():
    source = _runner_path().read_text()
    tree = ast.parse(source)
    top_level_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".", 1)[0])

    assert "torch" not in top_level_imports
    assert "flashinfer" not in top_level_imports
    assert "cupti" not in top_level_imports
    for forbidden in (
        "time.perf_counter",
        "time.time",
        "torch.cuda.Event",
        "bench_gpu_time",
    ):
        assert forbidden not in source
    assert source.count(".finalize()") == 1


def test_independent_recurrence_allocates_a_bf16_output_buffer():
    tree = ast.parse(_runner_path().read_text())
    recurrence = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_independent_bf16_recurrence"
    )
    out_assignment = next(
        node
        for node in ast.walk(recurrence)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "out"
            for target in node.targets
        )
    )
    assert isinstance(out_assignment.value, ast.Call)
    dtype = next(
        keyword.value
        for keyword in out_assignment.value.keywords
        if keyword.arg == "dtype"
    )
    assert ast.unparse(dtype) == "torch.bfloat16"
    contraction_signatures = [
        node.value
        for node in ast.walk(recurrence)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "->" in node.value
    ]
    assert contraction_signatures.count("hk,hvk->hv") == 2
    assert "nhk,nhvk->nhv" not in contraction_signatures
