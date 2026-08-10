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

import importlib.util
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
    payload = _preset_path().read_text().replace(
        '"aggregation": "per_case_only"',
        '"aggregation": "geomean"',
    )
    path = tmp_path / "bad.json"
    path.write_text(payload)

    with pytest.raises(
        evidence.EvidenceSchemaError,
        match="cross-shape aggregation is forbidden",
    ):
        evidence.load_preset(path)


def test_flash_kda_identity_mismatch_is_rejected(tmp_path):
    source_dir = tmp_path / "FlashKDA"
    package_path = source_dir / "flash_kda" / "__init__.py"
    extension_path = source_dir / "build" / "flash_kda_C.so"
    cutlass_dir = source_dir / "cutlass"
    package_path.parent.mkdir(parents=True)
    extension_path.parent.mkdir(parents=True)
    cutlass_dir.mkdir(parents=True)
    package_path.write_text("# package\n")
    extension_path.write_bytes(b"extension")

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
            git_output=wrong_revision,
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
    assert report["kernel_activity_count_samples"] == [2]
    assert report["kernel_activity_names_samples"] == [
        [
            "void PackBetaForTmaKernel<bf16>",
            "kernel_flashkda_bf16_fused_m128",
        ]
    ]
    prepared = report["prepared_recurrence"]
    assert prepared["derived_from_same_public_samples"] is True
    assert prepared["includes_beta_pack"] is False
    assert prepared["gpu_span_ms_samples"] == [0.00045]
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
