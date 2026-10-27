"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Frozen-source guard for the generated ``sm90_bf16_bf16_bf16_push_cake`` GEMMs (CPU only).

The FC1 / FC1+clamp / FC2 expert-grouped GEMM sources, their early-start
variants (FC1 releasing per-tile readiness counters, FC2 claiming ready tiles)
and their tvm-ffi bindings are generated and sealed by
``cake_sm90_bf16_megamoe_manifest.json``.
These tests pin the seal: every file is present and matches its sha256, each
stage resolves to exactly one module with the expected launch geometry and
FFI entry, the host constants mirrored in the Python shim equal the
generator's, and a tampered source fails the loader closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe.shim import cake_gemm as _gemm
from flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe.shim import cake_jit as _jit
from flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe.shim import (
    cake_weights as _weights,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_LAUNCH = {
    "block": [384, 1, 1],
    "dynamic_smem_bytes": 196704,
    "cluster": [2, 1, 1],
    "cooperative": False,
    "use_pdl": True,
}
EXPECTED_ARG_PLAN = [
    ["parameter", "num_experts"],
    ["parameter", "shape_n"],
    ["parameter", "shape_k"],
    ["parameter", "clamp_limit"],
    ["tma_buffer", "A"],
    ["tma_buffer", "W"],
    ["buffer", "offsets"],
    ["buffer", "D"],
    ["grid", "grid_x"],
    ["grid", "grid_y"],
    ["grid", "grid_z"],
]
# early-start stage -> the regular stage it replaces in a decode-window round
EARLY_STAGES = {
    "fc1_gated_signal": "fc1",
    "fc1_gated_clamp_signal": "fc1",
    "fc2_early": "fc2",
}
EARLY_DYNAMIC_SMEM_BYTES = 196768


def _expected_launch(stage: str) -> dict:
    launch = dict(EXPECTED_LAUNCH)
    if stage in EARLY_STAGES:
        launch["dynamic_smem_bytes"] = EARLY_DYNAMIC_SMEM_BYTES
    return launch


def _expected_arg_plan(stage: str) -> list:
    if stage not in EARLY_STAGES:
        return EXPECTED_ARG_PLAN
    extra = [["buffer", "sched"]]
    if stage == "fc2_early":
        extra += [["buffer", "fc2_done"], ["parameter", "use_fc2_done"]]
    grid = EXPECTED_ARG_PLAN.index(["grid", "grid_x"])
    return EXPECTED_ARG_PLAN[:grid] + extra + EXPECTED_ARG_PLAN[grid:]


@pytest.fixture(autouse=True)
def _fresh_manifest_cache():
    _jit.grouped_gemm_manifest.cache_clear()
    yield
    _jit.grouped_gemm_manifest.cache_clear()


def test_manifest_identity() -> None:
    manifest = _jit.grouped_gemm_manifest()
    assert manifest["schema"] == "cake.library_export.v5"
    assert manifest["library"] == "flashinfer"
    assert manifest["name"] == "cake_sm90_bf16_megamoe"
    assert manifest["contract"]["ffi_entry"] == "run"
    assert sorted(manifest["contract"]["stages"]) == sorted(_jit.GROUPED_GEMM_STAGES)


def test_package_data_declares_frozen_sources() -> None:
    pyproject_path = _PROJECT_ROOT / "pyproject.toml"
    if not pyproject_path.is_file():
        pytest.skip("pyproject.toml is only available in source-tree test runs")
    pyproject = pyproject_path.read_text(encoding="utf-8")
    key = '"flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe" = ['
    assert key in pyproject
    package_block = pyproject.split(key, maxsplit=1)[1].split("]", maxsplit=1)[0]
    assert '"src/*.cu"' in package_block
    assert '"src/*.json"' in package_block


def test_every_sealed_file_is_present_and_matches() -> None:
    manifest = _jit.grouped_gemm_manifest()
    source_dir = _jit._source_dir()
    assert manifest["files"], "empty file inventory"
    for item in manifest["files"]:
        path = source_dir / Path(item["path"]).name
        assert path.is_file(), item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item[
            "path"
        ]


@pytest.mark.parametrize("stage", _jit.GROUPED_GEMM_STAGES)
def test_stage_record(stage: str) -> None:
    record = _jit.grouped_gemm_record(stage)
    sealed = {item["path"] for item in _jit.grouped_gemm_manifest()["files"]}
    units = record["translation_units"]
    assert units["device"] in sealed and units["binding"] in sealed
    assert record["arch"] == "sm_90a"
    assert record["ffi_entry"] == "run"
    assert record["kernel_symbol"] == f"kernel_cake_sm90_bf16_megamoe_{stage}"
    assert record["route"]["gated"] is (not stage.startswith("fc2"))
    assert record["route"]["use_clamp"] is ("clamp" in stage)
    assert record["route"].get("early_start") == EARLY_STAGES.get(stage)
    expected_launch = _expected_launch(stage)
    assert {k: record["launch"][k] for k in expected_launch} == expected_launch
    assert record["arg_plan"] == _expected_arg_plan(stage)
    assert record["compile_flags"] == []
    # Both translation units resolve and re-hash cleanly.
    assert _jit._sealed_source(units["device"]).is_file()
    assert _jit._sealed_source(units["binding"]).is_file()


def test_closure_digests_match_records() -> None:
    """Each module's ``closure_sha256`` (whose prefix names its JIT module) is the
    digest of its own record, so a record edited without a reseal fails here."""
    for record in _jit.grouped_gemm_manifest()["modules"]:
        ident = {
            key: value
            for key, value in record.items()
            if key not in ("arg_plan_sha256", "closure_sha256")
        }
        text = json.dumps(ident, sort_keys=True, separators=(",", ":"), allow_nan=False)
        assert hashlib.sha256(text.encode()).hexdigest() == record["closure_sha256"], (
            record["name"]
        )


def test_arg_plan_digests_match_records() -> None:
    for record in _jit.grouped_gemm_manifest()["modules"]:
        text = json.dumps(record["arg_plan"], separators=(",", ":"))
        assert hashlib.sha256(text.encode()).hexdigest() == record["arg_plan_sha256"], (
            record["name"]
        )


def test_host_constants_match_generator() -> None:
    constants = _jit.grouped_gemm_manifest()["contract"]["host_constants"]
    assert constants["block_m"] == _gemm.BLOCK_M
    assert constants["block_n"] == _gemm.BLOCK_N
    assert constants["block_k"] == _gemm.BLOCK_K
    assert constants["gate_up_group"] == _weights.GATE_UP_GROUP
    assert constants["threads"] == EXPECTED_LAUNCH["block"][0]


def test_unknown_stage_rejected() -> None:
    with pytest.raises(ValueError, match="unknown grouped GEMM stage"):
        _jit.grouped_gemm_record("fc3")


def test_tampered_source_fails_closed(tmp_path: Path, monkeypatch) -> None:
    source_dir = _jit._source_dir()
    manifest = _jit.grouped_gemm_manifest()
    record = _jit.grouped_gemm_record("fc2")
    staged = tmp_path / "src"
    staged.mkdir()
    for item in manifest["files"]:
        shutil.copy(source_dir / Path(item["path"]).name, staged)
    device = staged / Path(record["translation_units"]["device"]).name
    device.write_bytes(device.read_bytes() + b"\n// tampered\n")
    staged_manifest = staged / _jit._manifest_path().name
    staged_manifest.write_text(json.dumps(manifest))
    monkeypatch.setattr(_jit, "_source_dir", lambda: staged)
    monkeypatch.setattr(_jit, "_manifest_path", lambda: staged_manifest)
    _jit.grouped_gemm_manifest.cache_clear()
    with pytest.raises(RuntimeError, match="generated source hash mismatch"):
        _jit._sealed_source(record["translation_units"]["device"])
    # The binding of the same module is untouched and still resolves.
    assert _jit._sealed_source(record["translation_units"]["binding"]).is_file()


def test_manifest_schema_drift_rejected(tmp_path: Path, monkeypatch) -> None:
    manifest = copy.deepcopy(_jit.grouped_gemm_manifest())
    manifest["schema"] = "cake.library_export.v4"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(_jit, "_manifest_path", lambda: path)
    _jit.grouped_gemm_manifest.cache_clear()
    with pytest.raises(RuntimeError, match="invalid SM90 BF16 grouped GEMM manifest"):
        _jit.grouped_gemm_manifest()
