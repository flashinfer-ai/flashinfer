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

CPU contracts for the internal persistent-offset SM90 BF16 GEMM.
"""

from __future__ import annotations

from dataclasses import replace
from importlib import resources
from importlib.util import find_spec
from pathlib import Path

import pytest


_PACKAGE_NAME = "flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = (
    _PROJECT_ROOT
    / "flashinfer"
    / "moe_ep"
    / "kernel_src"
    / "sm90"
    / "push_style_megamoe"
)


def _package_text(*parts: str) -> str:
    source_tree = _PACKAGE_ROOT.joinpath(*parts)
    if source_tree.is_file():
        return source_tree.read_text(encoding="utf-8")

    resource = resources.files(_PACKAGE_NAME)
    for part in parts:
        resource = resource / part
    return resource.read_text(encoding="utf-8")


def _module_text(module_name: str) -> str:
    spec = find_spec(module_name)
    if spec is None or spec.origin is None:
        raise AssertionError(f"module source is unavailable: {module_name}")
    return Path(spec.origin).read_text(encoding="utf-8")


def test_archived_persistent_engine_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_persistent_gemm import (
        _check_archived_gate,
    )

    monkeypatch.delenv("SM90_PUSH_BF16_ENABLE_ARCHIVED", raising=False)
    with pytest.raises(RuntimeError, match="bf16_single_gpu_20260817"):
        _check_archived_gate()
    monkeypatch.setenv("SM90_PUSH_BF16_ENABLE_ARCHIVED", "1")
    _check_archived_gate()


def test_persistent_offsets_source_has_no_cutlass_prepare_launch() -> None:
    kernel = _package_text("src", "bf16_persistent_gemm", "bf16_persistent_gemm.cuh")
    binding = _package_text(
        "src", "bf16_persistent_gemm", "bf16_persistent_gemm_binding.cu"
    )
    source = kernel + binding

    assert "grouped_bf16_persistent_offsets_kernel" in source
    assert "prepare_schedule_and_arguments_kernel" not in source
    assert "grouped_run_prepared" not in source
    assert "int64_t const* offsets" in source
    assert "grouped_run" in binding
    assert "kernel_resource_usage" in binding
    assert "cudaLaunchAttributeClusterDimension" in kernel
    assert "SM90_TMA_LOAD_MULTICAST_2D" in kernel
    assert "kOperandAElements = kOperandARows * kBlockK" in kernel
    assert "inner * Traits::kOperandARows * kMmaK" in kernel
    assert "inner * Traits::kOperandBRows * kMmaK" in kernel
    assert "if constexpr (Traits::kSchedule == 0)" in kernel


def test_persistent_offsets_adapter_uses_one_grouped_run_seam() -> None:
    shim = _package_text("shim", "bf16_persistent_gemm.py")

    assert "del prepare_schedule" in shim
    assert "self.ffi_runner.grouped_run(" in shim
    assert "grouped_run_prepared" not in shim
    assert "self.trusted_offsets" in shim
    assert "self._map_identity != map_identity" in shim
    assert "same activation and weight storage" in shim
    assert '"implementation": "persistent_offsets"' in shim
    assert '"comparison_scope": "full_implementation_matched_tactic"' in shim
    assert '"wait_policy"' in shim
    assert '"active_rows_source": "offsets[-1]"' in shim
    assert '"capacity_tail_written": False' in shim


def test_persistent_offsets_jit_snapshot_covers_all_inputs() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        bf16_persistent_gemm,
    )

    snapshot = bf16_persistent_gemm._capture_source_snapshot()

    assert tuple(name for name, _ in snapshot.sources) == (
        "bf16_persistent_gemm.cuh",
        "bf16_persistent_gemm_binding.cu",
    )
    assert snapshot.generator
    assert snapshot.tactic_generator
    assert tuple(name for name, _ in snapshot.dependencies) == (
        "csrc/tvm_ffi_utils.h",
        "csrc/nv_internal/tensorrt_llm/deep_gemm/mma_utils.cuh",
        "csrc/nv_internal/tensorrt_llm/deep_gemm/tma_utils.cuh",
        "csrc/nv_internal/tensorrt_llm/deep_gemm/utils.cuh",
        "include/flashinfer/attention/hopper.cuh",
        "include/flashinfer/cp_async.cuh",
        "include/flashinfer/layout.cuh",
        "include/flashinfer/mma.cuh",
        "include/flashinfer/permuted_smem.cuh",
    )
    assert all(content for _, content in snapshot.dependencies)


def test_persistent_offsets_jit_spec_uses_shape_and_tactic_flags(
    tmp_path, monkeypatch
) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        bf16_persistent_gemm,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_tactics import (
        CORE_BF16_GEMM_TACTICS,
    )

    captured: dict[str, object] = {}

    def fake_gen_jit_spec(uri, sources, **kwargs):
        captured.update(uri=uri, sources=sources, **kwargs)
        return captured

    tactic = CORE_BF16_GEMM_TACTICS[0]
    snapshot = bf16_persistent_gemm._capture_source_snapshot()
    digest = bf16_persistent_gemm._source_digest(snapshot, tactic, 256, 128)
    monkeypatch.setattr(
        bf16_persistent_gemm, "is_cuda_version_at_least", lambda _: True
    )
    monkeypatch.setattr(bf16_persistent_gemm, "gen_jit_spec", fake_gen_jit_spec)
    monkeypatch.setattr(
        bf16_persistent_gemm.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path
    )

    result = bf16_persistent_gemm._make_jit_spec(snapshot, digest, tactic, 256, 128)

    assert result is captured
    assert tactic.tag in str(captured["uri"])
    assert str(captured["uri"]).endswith(digest)
    assert [Path(path).name for path in captured["sources"]] == [
        "bf16_persistent_gemm_binding.cu"
    ]
    flags = captured["extra_cuda_cflags"]
    assert "-DSM90_PUSH_BF16_SHAPE_N=256" in flags
    assert "-DSM90_PUSH_BF16_SHAPE_K=128" in flags
    assert any(str(flag).startswith("-DSM90_PUSH_BF16_FAMILY_MASK=") for flag in flags)
    materialized = tmp_path / str(captured["uri"]) / "bf16_persistent_gemm"
    assert (materialized / "bf16_persistent_gemm.cuh").read_bytes()
    assert (materialized / "bf16_persistent_gemm_binding.cu").read_bytes()
    snapshot_root = materialized.parent
    assert (snapshot_root / "tvm_ffi_utils.h").read_bytes()
    assert (
        snapshot_root / "nv_internal/tensorrt_llm/deep_gemm/mma_utils.cuh"
    ).read_bytes()
    assert (
        snapshot_root / "nv_internal/tensorrt_llm/deep_gemm/tma_utils.cuh"
    ).read_bytes()
    assert (snapshot_root / "include/flashinfer/attention/hopper.cuh").read_bytes()
    assert (snapshot_root / "include/flashinfer/permuted_smem.cuh").read_bytes()
    generated_root = tmp_path / str(captured["uri"])
    assert (generated_root / "tvm_ffi_utils.h").read_bytes()
    assert (
        generated_root / "nv_internal/tensorrt_llm/deep_gemm/mma_utils.cuh"
    ).read_bytes()
    assert (generated_root / "include/flashinfer/attention/hopper.cuh").read_bytes()


def test_persistent_offsets_digest_covers_source_tactic_and_shape() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        bf16_persistent_gemm,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_tactics import (
        CORE_BF16_GEMM_TACTICS,
    )

    snapshot = bf16_persistent_gemm._capture_source_snapshot()
    tactic = CORE_BF16_GEMM_TACTICS[0]
    baseline = bf16_persistent_gemm._source_digest(snapshot, tactic, 128, 128)
    source_name, source = snapshot.sources[0]
    changed_source = replace(
        snapshot,
        sources=((source_name, source + b"\nchanged"), *snapshot.sources[1:]),
    )
    changed_tactics = replace(
        snapshot, tactic_generator=snapshot.tactic_generator + b"\nchanged"
    )

    assert (
        bf16_persistent_gemm._source_digest(changed_source, tactic, 128, 128)
        != baseline
    )
    assert (
        bf16_persistent_gemm._source_digest(changed_tactics, tactic, 128, 128)
        != baseline
    )
    assert (
        bf16_persistent_gemm._source_digest(
            snapshot, CORE_BF16_GEMM_TACTICS[1], 128, 128
        )
        != baseline
    )
    assert bf16_persistent_gemm._source_digest(snapshot, tactic, 256, 128) != baseline
    assert bf16_persistent_gemm._source_digest(snapshot, tactic, 128, 256) != baseline


def test_persistent_offsets_uri_names_implementation_tactic_and_shape() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_persistent_gemm import (
        sm90_push_bf16_persistent_gemm_uri,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_tactics import (
        CORE_BF16_GEMM_TACTICS,
    )

    uris = [
        sm90_push_bf16_persistent_gemm_uri(tactic, n=256, k=128)
        for tactic in CORE_BF16_GEMM_TACTICS
    ]

    assert len(uris) == len(set(uris))
    for tactic, uri in zip(CORE_BF16_GEMM_TACTICS, uris, strict=True):
        assert uri.startswith("sm90_push_bf16_persistent_gemm_n256_k128_")
        assert tactic.tag in uri


def test_persistent_offsets_rejects_shape_tactic_tile_mismatches() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_persistent_gemm import (
        _validate_tactic_shape,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_tactics import (
        Bf16GemmFamilyTactic,
        Bf16GemmTactic,
    )

    tactic = Bf16GemmTactic(
        "m64", m64=Bf16GemmFamilyTactic(64, 128, 128, 3, 1, "pingpong")
    )

    assert _validate_tactic_shape(tactic, 256, 256) == (256, 256)
    with pytest.raises(ValueError, match="BlockN"):
        _validate_tactic_shape(tactic, 192, 256)
    with pytest.raises(ValueError, match="BlockK"):
        _validate_tactic_shape(tactic, 256, 192)


def test_persistent_offsets_auto_selector_keeps_64_aligned_shapes_executable() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_persistent_gemm import (
        _validate_tactic_shape,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_tactics import (
        select_sm90_push_bf16_gemm_tactic,
    )

    tactic, _ = select_sm90_push_bf16_gemm_tactic(
        expected_m=96,
        n=2880,
        k=2880,
        sm_count=132,
    )

    assert {family.block_n for family in tactic.families} == {64}
    assert {family.block_k for family in tactic.families} == {64}
    assert _validate_tactic_shape(tactic, 2880, 2880) == (2880, 2880)


def test_persistent_offsets_reuses_the_explicit_finite_tactic_matrix() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_persistent_gemm import (
        _cuda_flags,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_tactics import (
        CORE_BF16_GEMM_TACTICS,
        SUPPORTED_BF16_GEMM_FAMILY_TACTICS,
        SUPPORTED_BF16_GEMM_TACTICS,
    )

    assert SUPPORTED_BF16_GEMM_TACTICS
    assert set(CORE_BF16_GEMM_TACTICS).issubset(SUPPORTED_BF16_GEMM_TACTICS)
    assert {tactic.family_mode for tactic in SUPPORTED_BF16_GEMM_TACTICS} == {
        "m64",
        "m128",
        "dual",
    }
    assert {family.block_n for family in SUPPORTED_BF16_GEMM_FAMILY_TACTICS} == {
        64,
        128,
    }
    assert {family.block_k for family in SUPPORTED_BF16_GEMM_FAMILY_TACTICS} == {
        64,
        128,
    }
    assert {family.stages for family in SUPPORTED_BF16_GEMM_FAMILY_TACTICS} == {
        2,
        3,
        4,
    }
    assert {family.cluster_m for family in SUPPORTED_BF16_GEMM_FAMILY_TACTICS} == {
        1,
        2,
    }
    assert {family.schedule for family in SUPPORTED_BF16_GEMM_FAMILY_TACTICS} == {
        "pingpong",
        "cooperative",
    }
    assert {tactic.swap_ab for tactic in SUPPORTED_BF16_GEMM_TACTICS} == {
        False,
        True,
    }
    assert all(
        "-DSM90_PUSH_BF16_SHAPE_N=256" in _cuda_flags(t, 256, 128)
        for t in CORE_BF16_GEMM_TACTICS
    )
    assert all(
        "-DSM90_PUSH_BF16_SHAPE_K=128" in _cuda_flags(t, 256, 128)
        for t in CORE_BF16_GEMM_TACTICS
    )


def test_persistent_offsets_sources_are_packaged_but_not_aot_registered() -> None:
    pyproject_path = _PROJECT_ROOT / "pyproject.toml"
    if not pyproject_path.is_file():
        pytest.skip("pyproject.toml is only available in source-tree test runs")
    pyproject = pyproject_path.read_text(encoding="utf-8")
    aot = _module_text("flashinfer.aot")
    package_key = '"flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe" = ['
    package_data = pyproject.split(package_key, maxsplit=1)[1].split("]", maxsplit=1)[0]

    assert '"src/bf16_persistent_gemm/*.cu"' in package_data
    assert '"src/bf16_persistent_gemm/*.cuh"' in package_data
    assert "gen_sm90_push_bf16_persistent_gemm_module" not in aot


def test_persistent_offsets_stays_out_of_public_production_config() -> None:
    public_sources = (
        _module_text(
            "flashinfer.moe_ep.backends.mega.kernel.sm90."
            "bf16_bf16_bf16_push_cuda.config"
        ),
        _module_text("flashinfer.moe_ep"),
        _module_text("flashinfer.moe_ep.backends.mega.kernel"),
    )

    for source in public_sources:
        assert "persistent_offsets" not in source
        assert "Bf16Persistent" not in source


def test_persistent_offsets_is_an_internal_runner_axis_with_cutlass_default() -> None:
    runner = _package_text("shim", "bf16_runner.py")

    assert 'gemm_implementation: str = "cutlass_prepared"' in runner
    assert 'implementation not in ("cutlass_prepared", "persistent_offsets")' in runner
    assert "create_sm90_push_bf16_persistent_gemm_runner" in runner
    assert 'if gemm_implementation == "cutlass_prepared"' in runner
