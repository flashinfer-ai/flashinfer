# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""CPU-only tests for durable Prims-TS MoE sweep foundations."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from flashinfer.prims_ts.moe.tools.sweep.calibration import (
    CALIBRATION_SCHEMA_FINGERPRINT,
    CalibrationCompatibilityError,
    CalibrationData,
    CompileScalingPoint,
    EnvironmentFingerprint,
    ShapeGPUCost,
    derive_noise_band,
    select_compile_scaling_knee,
)
from flashinfer.prims_ts.moe.tools.sweep.db import (
    JSONLWriter,
    JournalCorruptionError,
    SweepCheckpoint,
    read_jsonl,
    recover_jsonl,
)


def test_sweep_package_import_does_not_eagerly_import_submodules() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import flashinfer.prims_ts.moe.tools.sweep; "
                "assert not any(name.startswith("
                "'flashinfer.prims_ts.moe.tools.sweep.'"
                ") for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_jsonl_recovers_only_truncated_trailing_line(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"candidate_id":"a","time":1}\n{"candidate_id":"b"')

    recovery = recover_jsonl(path)

    assert [record["candidate_id"] for record in recovery.records] == ["a"]
    assert recovery.truncated_bytes > 0
    assert path.read_bytes() == b'{"candidate_id":"a","time":1}\n'

    with JSONLWriter(path, fsync=True) as writer:
        writer.append({"candidate_id": "c", "time": 2})
    assert [record["candidate_id"] for record in read_jsonl(path)] == ["a", "c"]


def test_jsonl_retains_valid_last_record_without_newline(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"candidate_id":"a"}')

    recovery = recover_jsonl(path)

    assert recovery.added_newline
    assert recovery.records == ({"candidate_id": "a"},)
    assert path.read_bytes().endswith(b"\n")


def test_jsonl_does_not_hide_complete_line_corruption(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"candidate_id":"a"}\nnot-json\n')

    with pytest.raises(JournalCorruptionError):
        recover_jsonl(path)


def test_checkpoint_roundtrip_restores_rng_and_resume_order(tmp_path: Path) -> None:
    rng = random.Random(2026)
    prefix = [rng.random() for _ in range(3)]
    assert len(prefix) == 3
    checkpoint = SweepCheckpoint.capture(
        rng,
        sequence=17,
        extra={"phase": "search", "search": {"version": 3}},
    )
    expected = [rng.random() for _ in range(5)]

    path = tmp_path / "state.json"
    checkpoint.save(path)
    loaded = SweepCheckpoint.load(path)
    resumed_rng = loaded.restore_rng()

    assert [resumed_rng.random() for _ in range(5)] == expected
    assert loaded.sequence == 17
    assert loaded.extra == checkpoint.extra
    # The checkpoint itself contains only portable JSON types.
    encoded = json.loads(path.read_text())
    assert encoded["rng_state"]["internal"]
    assert set(encoded) == {
        "schema_version",
        "rng_state",
        "sequence",
        "extra",
    }


def _environment(**overrides: str) -> EnvironmentFingerprint:
    values = {
        "gpu": "NVIDIA Test GPU",
        "driver": "600.1",
        "cuda": "13.0",
        "cute_dsl": "4.7.0",
        "cpu": "Test CPU",
        "flashinfer_sha": "0123456789abcdef",
        "protocol": "prims-ts-moe-sweep-v1",
    }
    values.update(overrides)
    return EnvironmentFingerprint(**values)


def _calibration() -> CalibrationData:
    noise = derive_noise_band([9.9, 10.2, 10.1, 9.8, 10.0])
    return CalibrationData(
        environment=_environment(),
        compile_scaling=[
            CompileScalingPoint(1, 1.0, 100),
            CompileScalingPoint(2, 1.9, 190),
            CompileScalingPoint(4, 2.05, 370),
        ],
        worker_count=2,
        noise=noise,
        shape_costs=[ShapeGPUCost("fc1:N=128:K=64", 0.5)],
    )


def test_calibration_roundtrip_and_compatibility_override(tmp_path: Path) -> None:
    calibration = _calibration()
    path = tmp_path / "calibration.json"
    calibration.save(path)

    loaded = CalibrationData.load(path)

    assert loaded.to_dict() == calibration.to_dict()
    assert loaded.schema_fingerprint == CALIBRATION_SCHEMA_FINGERPRINT
    assert loaded.validate_compatibility(_environment()) == []
    assert (
        loaded.validate_compatibility(_environment(flashinfer_sha="fedcba9876543210"))
        == []
    )
    changed = _environment(driver="601.0")
    with pytest.raises(CalibrationCompatibilityError, match="driver"):
        loaded.validate_compatibility(changed)
    mismatches = loaded.validate_compatibility(changed, force=True)
    assert len(mismatches) == 1
    assert mismatches[0].startswith("driver:")

    unknown_driver = replace(
        loaded,
        environment=_environment(driver="unknown"),
    )
    assert unknown_driver.validate_compatibility(changed) == []


def test_compile_scaling_knee_and_memory_failure() -> None:
    points = [
        CompileScalingPoint(1, 1.0),
        CompileScalingPoint(2, 1.9),
        CompileScalingPoint(4, 2.05),
        CompileScalingPoint(8, 0.0, failed=True, failure="out of memory"),
    ]
    assert select_compile_scaling_knee(points) == 2
    assert (
        select_compile_scaling_knee(
            [
                CompileScalingPoint(1, 1.0, 100),
                CompileScalingPoint(2, 1.8, 250),
            ],
            max_peak_rss_bytes=200,
        )
        == 1
    )
    assert (
        select_compile_scaling_knee(
            [
                CompileScalingPoint(1, 4.4),
                CompileScalingPoint(2, 11.7),
                CompileScalingPoint(4, 12.6),
                CompileScalingPoint(8, 37.0),
                CompileScalingPoint(16, 50.0),
            ]
        )
        == 16
    )
    assert (
        select_compile_scaling_knee(
            [
                CompileScalingPoint(1, 5.0),
                CompileScalingPoint(2, 12.0),
                CompileScalingPoint(4, 22.0),
                CompileScalingPoint(8, 23.0),
                CompileScalingPoint(16, 23.5),
            ]
        )
        == 4
    )


def test_noise_band_is_robust_and_json_safe() -> None:
    band = derive_noise_band([100.0, 99.0, 101.0, 100.5, 99.5])

    assert band.median == 100.0
    assert band.lower < band.median < band.upper
    assert band.relative > 0
    json.dumps(band.to_dict(), allow_nan=False)
