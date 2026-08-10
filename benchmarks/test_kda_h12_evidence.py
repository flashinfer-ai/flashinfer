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
            "environment": {},
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
        "schema_version": 1,
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


def _timing_receipt(name):
    names = [f"{name}_kernel"]
    if name == "flashinfer_public":
        names = [
            "void PackBetaForTmaKernel<bf16>",
            "kernel_flashkda_bf16_fused_m128",
        ]
    timing = {
        "timing_backend": "cupti_activity",
        "raw_samples": [{"gpu_span_ms": 1.0}, {"gpu_span_ms": 1.1}],
        "kernel_activity_names_samples": [list(names), list(names)],
    }
    if name == "flashinfer_public":
        timing.update(
            {
                "includes_beta_preparation": True,
                "prepared_recurrence": {
                    "derived_from_same_public_samples": True,
                    "includes_beta_pack": False,
                    "raw_samples": [
                        {"gpu_span_ms": 0.9},
                        {"gpu_span_ms": 1.0},
                    ],
                    "kernel_activity_names_samples": [
                        ["kernel_flashkda_bf16_fused_m128"],
                        ["kernel_flashkda_bf16_fused_m128"],
                    ],
                },
            }
        )
    return timing


def _complete_per_arch_report(tmp_path, arch):
    preset = evidence.load_preset(_preset_path())
    source_dir, package_path, extension_path, _ = _flash_kda_paths(tmp_path / arch)
    _, build_manifest = _build_manifest(
        tmp_path / arch,
        source_dir,
        package_path,
        extension_path,
        arch=arch,
    )
    candidate_commit = "1" * 40
    fla_commit = "2" * 40
    oracle_result = {
        "output": {"passed": True},
        "final_state": {"passed": True},
    }
    cases = []
    for case in preset.cases:
        cases.append(
            {
                "name": case.name,
                "layout": case.layout,
                "seq_lens": list(case.seq_lens),
                "seed": case.seed,
                "num_heads": 12,
                "head_dim_qk": 128,
                "head_dim_vo": 128,
                "dtype": "bfloat16",
                "correctness": {
                    "passed": True,
                    "public_output_and_full_final_state": True,
                    "independent_bf16_recurrence": json.loads(
                        json.dumps(oracle_result)
                    ),
                    "pinned_flash_kda": json.loads(json.dumps(oracle_result)),
                    "fla_triton": json.loads(json.dumps(oracle_result)),
                },
                "timings": {
                    name: _timing_receipt(name)
                    for name in evidence.REQUIRED_TIMING_PATHS
                },
            }
        )
    capability = [10, 0] if arch == "sm100a" else [10, 3]
    report = {
        "schema_version": evidence.EVIDENCE_REPORT_SCHEMA_VERSION,
        "suite": "recurrent_kda_prefill_h12_phase_a",
        "preset": {
            "name": preset.name,
            "sha256": preset.sha256,
            "aggregation": "per_case_only",
        },
        "candidate_provenance": {
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
            "imported_module_sha256": {"flashinfer.kda": "a" * 64},
            "source_sha256": {"flashinfer/kda.py": "b" * 64},
        },
        "baselines": {
            "flash_kda": {
                "available": True,
                "repository": evidence.FLASH_KDA_REPOSITORY,
                "source_commit": evidence.FLASH_KDA_BASELINE_REVISION,
                "cutlass_commit": evidence.FLASH_KDA_CUTLASS_REVISION,
                "worktree_clean_including_untracked": True,
                "package_sha256": build_manifest["artifacts"]["package_sha256"],
                "extension_sha256": build_manifest["artifacts"]["extension_sha256"],
                "build_manifest_sha256": (
                    evidence.flash_kda_build_manifest_sha256(build_manifest)
                ),
                "build_manifest": build_manifest,
                "current_receipt_binding": _current_receipt_binding(build_manifest),
            },
            "fla_triton": {
                "available": True,
                "git_revision": fla_commit,
                "worktree_clean_including_untracked": True,
                "distribution_version": "0.3.2",
                "package_sha256": "d" * 64,
                "op_sha256": "e" * 64,
                "forced_environment": {
                    "FLA_FLASH_KDA": "0",
                    "FLA_DISABLE_BACKEND_DISPATCH": "1",
                },
            },
        },
        "hardware": {"cuda_arch": arch, "compute_capability": capability},
        "measurement": {
            "timing_backend": "cupti_activity",
            "cross_shape_geomean": False,
            "blocks": 2,
            "repeat_iters_per_block": 1,
        },
        "changed_beta_cuda_graph_test": {
            "source": evidence.GRAPH_TEST_SOURCE,
            "source_line_range": list(evidence.GRAPH_TEST_SOURCE_LINE_RANGE),
            "source_sha256": "f" * 64,
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
    with pytest.raises(evidence.EvidenceSchemaError, match="required oracle"):
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
