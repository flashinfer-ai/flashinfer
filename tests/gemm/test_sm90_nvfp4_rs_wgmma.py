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
"""

import os
import re

import pytest
import torch

from flashinfer.fused_moe.nvfp4_checkpoint import (
    NVFP4Checkpoint,
    reference_dequantize_nvfp4,
)
from flashinfer.jit.cpp_ext import is_cuda_version_at_least
from flashinfer.utils import is_sm90a_supported


requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cuda_version_at_least("12.0")
    or not is_sm90a_supported(torch.device("cuda")),
    reason="requires SM90 and CUDA Toolkit 12.0+",
)


def _grouped_reference(
    activation: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    result = torch.empty(
        activation.shape[0],
        weights.shape[1],
        dtype=torch.float32,
        device=activation.device,
    )
    for expert in range(weights.shape[0]):
        begin = int(offsets[expert].item())
        end = int(offsets[expert + 1].item())
        result[begin:end] = activation[begin:end].float() @ weights[expert].float().T
    return result


def _dequantize_nvfp4_streams(
    payload: torch.Tensor,
    scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    experts, rows, packed_k = payload.shape
    checkpoint = NVFP4Checkpoint(
        payload,
        scales,
        alpha,
        (experts, rows, packed_k * 2),
        tuple(range(experts)),
        "flashinfer.sm90_push.nvfp4.rs_test",
    )
    return reference_dequantize_nvfp4(checkpoint)


def test_sm90_nvfp4_rs_uri_is_explicit():
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        get_sm90_push_nvfp4_rs_gemm_uri,
    )

    uri = get_sm90_push_nvfp4_rs_gemm_uri("rs_wgmma", 64, 3, 64)
    assert uri.startswith("sm90_push_nvfp4_rs_gemm_v5_")
    assert "_rs_wgmma_n64_s3_k64_" in uri
    with pytest.raises(ValueError, match="implementation"):
        get_sm90_push_nvfp4_rs_gemm_uri("rs_bf16", 64, 3)
    with pytest.raises(ValueError, match="stages"):
        get_sm90_push_nvfp4_rs_gemm_uri("rs_wgmma", 64, 2, 64)
    with pytest.raises(ValueError, match="stage_k"):
        get_sm90_push_nvfp4_rs_gemm_uri("rs_wgmma", 64, 3, 32)


def test_sm90_nvfp4_rs_retires_each_commit_before_accumulator_reuse():
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_rs_gemm as rs_gemm,
    )

    sources = dict(rs_gemm._capture_source_snapshot().sources)
    kernel = sources["sm90_nvfp4_rs_kernel.cuh"].decode("utf-8")
    assert "SM90_NVFP4_RS_DIAG_SINGLE_WGMMA_GROUP" not in kernel
    assert re.search(
        r"wgmma_commit\(\);\s*compiler_fence\(accumulator\);\s*"
        r"(?://[^\n]*\s*)?wgmma_wait<0>\(\);\s*compiler_fence\(accumulator\);",
        kernel,
    )
    assert "wgmma_wait<1>()" not in kernel
    assert "a_fragments[2]" not in kernel
    assert re.search(
        r"wgmma_wait<0>\(\);\s*compiler_fence\(accumulator\);\s*"
        r"#pragma unroll\s*for \(int member = 0; member < kWgmmaGroup; \+\+member\) \{\s*"
        r"compiler_fence\(a_fragments\[member\]\);",
        kernel,
    )


def test_sm90_nvfp4_rs_frozen_spec_ignores_experiment_environment(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "9.0a")

    from flashinfer.compilation_context import CompilationContext
    from flashinfer.jit import core as jit_core
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        gen_sm90_push_nvfp4_rs_gemm_module,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_rs_gemm as rs_gemm,
    )

    monkeypatch.setattr(jit_core, "current_compilation_context", CompilationContext())
    monkeypatch.setattr(rs_gemm.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)
    monkeypatch.setattr(rs_gemm, "is_cuda_version_at_least", lambda _version: True)
    for name in (
        "FLASHINFER_SM90_PUSH_NVFP4_RS_WGMMA_GROUP",
        "FLASHINFER_SM90_PUSH_NVFP4_RS_STATIC_SCHED",
        "FLASHINFER_SM90_PUSH_NVFP4_RS_NO_UNION",
    ):
        monkeypatch.delenv(name, raising=False)
    baseline = gen_sm90_push_nvfp4_rs_gemm_module(use_environment=False)
    assert baseline.name == rs_gemm.get_sm90_push_nvfp4_rs_gemm_uri()

    monkeypatch.setenv("FLASHINFER_SM90_PUSH_NVFP4_RS_WGMMA_GROUP", "2")
    monkeypatch.setenv("FLASHINFER_SM90_PUSH_NVFP4_RS_STATIC_SCHED", "1")
    monkeypatch.setenv("FLASHINFER_SM90_PUSH_NVFP4_RS_NO_UNION", "1")
    repeated = gen_sm90_push_nvfp4_rs_gemm_module(use_environment=False)

    assert repeated.name == baseline.name
    assert repeated.extra_cuda_cflags == baseline.extra_cuda_cflags
    assert "-DSM90_NVFP4_RS_WGMMA_GROUP=1" in baseline.extra_cuda_cflags
    assert "-DSM90_NVFP4_RS_STATIC_SCHED=0" in baseline.extra_cuda_cflags
    assert "-DSM90_NVFP4_RS_NO_UNION=0" in baseline.extra_cuda_cflags


def test_sm90_nvfp4_rs_jit_spec_consumes_the_hashed_source_snapshot(
    monkeypatch, tmp_path
):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_rs_gemm as rs_gemm,
    )

    source_directory = tmp_path / "live" / "csrc" / "fused_moe" / "sm90_nvfp4_rs_gemm"
    source_directory.mkdir(parents=True)
    binding = source_directory / "sm90_nvfp4_rs_binding.cu"
    kernel = source_directory / "kernel.cuh"
    binding.write_bytes(b"binding-v1\r\n")
    kernel.write_bytes(b"kernel-v1")
    (tmp_path / "live" / "csrc" / "tvm_ffi_utils.h").write_bytes(b"ffi-v1")
    layout = tmp_path / "live" / "include" / "flashinfer" / "layout.cuh"
    layout.parent.mkdir(parents=True)
    layout.write_bytes(b"layout-v1")
    monkeypatch.setattr(
        rs_gemm,
        "_SOURCE_NAMES",
        ("kernel.cuh", "sm90_nvfp4_rs_binding.cu"),
    )
    monkeypatch.setattr(rs_gemm, "_source_directory", lambda: source_directory)
    monkeypatch.setattr(rs_gemm, "_csrc_directory", lambda: tmp_path / "live" / "csrc")
    monkeypatch.setattr(
        rs_gemm.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "generated"
    )
    monkeypatch.setattr(rs_gemm, "is_cuda_version_at_least", lambda _version: True)

    class _Spec:
        def __init__(self, name, sources, options):
            self.name = name
            self.sources = sources
            self.options = options

    monkeypatch.setattr(
        rs_gemm,
        "gen_jit_spec",
        lambda name, sources, **options: _Spec(name, sources, options),
    )
    expected_snapshot = rs_gemm._capture_source_snapshot()
    knobs = rs_gemm._experiment_knobs()
    expected_digest = rs_gemm._source_digest(
        "rs_wgmma", 64, 3, 64, knobs, snapshot=expected_snapshot
    )
    binding.write_bytes(b"binding-v1\n")
    assert rs_gemm._source_digest("rs_wgmma", 64, 3, 64, knobs) == expected_digest
    binding.write_bytes(b"binding-v1\r\n")
    materialize = rs_gemm._materialize_source_snapshot

    def mutate_then_materialize(uri, snapshot):
        kernel.write_bytes(b"kernel-v2")
        return materialize(uri, snapshot)

    monkeypatch.setattr(
        rs_gemm, "_materialize_source_snapshot", mutate_then_materialize
    )
    spec = rs_gemm.gen_sm90_push_nvfp4_rs_gemm_module()

    snapshotted_binding = spec.sources[0]
    snapshotted_kernel = snapshotted_binding.parent / "kernel.cuh"
    assert spec.name.endswith(f"_{expected_digest}")
    assert snapshotted_binding != binding
    assert snapshotted_binding.read_bytes() == b"binding-v1\n"
    assert snapshotted_kernel.read_bytes() == b"kernel-v1"
    assert (
        snapshotted_binding.parent.parent / "flashinfer/layout.cuh"
    ).read_bytes() == b"layout-v1"
    assert spec.options["extra_include_paths"] == [
        snapshotted_binding.parent.parent,
        snapshotted_binding.parent,
    ]
    assert kernel.read_bytes() == b"kernel-v2"


def test_sm90_nvfp4_rs_snapshot_write_accepts_an_identical_concurrent_winner(
    monkeypatch, tmp_path
):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_rs_gemm as rs_gemm,
    )

    destination = tmp_path / "kernel.cuh"

    def concurrent_replace(_source, target):
        target.write_bytes(b"identical")
        raise PermissionError("destination is open by another process")

    monkeypatch.setattr(rs_gemm.os, "replace", concurrent_replace)
    rs_gemm._write_snapshot_file(destination, b"identical")

    assert destination.read_bytes() == b"identical"
    assert set(tmp_path.iterdir()) == {destination}


def test_sm90_nvfp4_rs_snapshot_write_rejects_a_conflicting_concurrent_winner(
    monkeypatch, tmp_path
):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_rs_gemm as rs_gemm,
    )

    destination = tmp_path / "kernel.cuh"

    def conflicting_replace(_source, target):
        target.write_bytes(b"conflict")
        raise PermissionError("destination is open by another process")

    monkeypatch.setattr(rs_gemm.os, "replace", conflicting_replace)
    with pytest.raises(PermissionError, match="open by another process"):
        rs_gemm._write_snapshot_file(destination, b"expected")

    assert destination.read_bytes() == b"conflict"
    assert set(tmp_path.iterdir()) == {destination}


def test_sm90_nvfp4_rs_loaded_module_cache_tracks_source_digest(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_rs_gemm as rs_gemm,
    )

    source_directory = tmp_path / "csrc" / "fused_moe" / "sm90_nvfp4_rs_gemm"
    source_directory.mkdir(parents=True)
    kernel = source_directory / "kernel.cuh"
    kernel.write_text("first", encoding="utf-8")
    (tmp_path / "csrc" / "tvm_ffi_utils.h").write_text("binding", encoding="utf-8")
    layout = tmp_path / "include" / "flashinfer" / "layout.cuh"
    layout.parent.mkdir(parents=True)
    layout.write_text("layout-first", encoding="utf-8")
    monkeypatch.setattr(rs_gemm, "_SOURCE_NAMES", ("kernel.cuh",))
    monkeypatch.setattr(rs_gemm, "_source_directory", lambda: source_directory)
    monkeypatch.setattr(rs_gemm, "_csrc_directory", lambda: tmp_path / "csrc")

    loaded_modules = []

    class _Spec:
        def build_and_load(self):
            module = object()
            loaded_modules.append(module)
            return module

    monkeypatch.setattr(
        rs_gemm,
        "_gen_sm90_push_nvfp4_rs_gemm_module",
        lambda *_args: _Spec(),
    )
    cache = rs_gemm._load_sm90_push_nvfp4_rs_gemm_module_cached
    cache.cache_clear()
    try:
        first = rs_gemm.load_sm90_push_nvfp4_rs_gemm_module()
        unchanged = rs_gemm.load_sm90_push_nvfp4_rs_gemm_module()
        layout.write_text("layout-second", encoding="utf-8")
        changed = rs_gemm.load_sm90_push_nvfp4_rs_gemm_module()
    finally:
        cache.cache_clear()

    assert unchanged is first
    assert changed is not first
    assert loaded_modules == [first, changed]


@pytest.mark.parametrize(
    "name",
    [
        "FLASHINFER_SM90_PUSH_NVFP4_RS_STATIC_SCHED",
        "FLASHINFER_SM90_PUSH_NVFP4_RS_NO_UNION",
    ],
)
def test_sm90_nvfp4_rs_boolean_build_knobs_are_strict(monkeypatch, name):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        get_sm90_push_nvfp4_rs_gemm_uri,
    )

    monkeypatch.setenv(name, "true")
    with pytest.raises(ValueError, match=f"{name} must be 0 or 1"):
        get_sm90_push_nvfp4_rs_gemm_uri()


def test_sm90_nvfp4_rs_environment_uses_final_names_only(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        get_sm90_push_nvfp4_rs_gemm_uri,
    )

    for name in (
        "FLASHINFER_SM90_PUSH_NVFP4_RS_WGMMA_GROUP",
        "FLASHINFER_SM90_PUSH_NVFP4_RS_STATIC_SCHED",
        "FLASHINFER_SM90_PUSH_NVFP4_RS_NO_UNION",
    ):
        monkeypatch.delenv(name, raising=False)
    baseline = get_sm90_push_nvfp4_rs_gemm_uri()
    monkeypatch.setenv("FI_RS_WGMMA_GROUP", "4")
    monkeypatch.setenv("FI_RS_STATIC_SCHED", "1")
    monkeypatch.setenv("FI_RS_NO_UNION", "1")
    monkeypatch.setenv("FLASHINFER_NVFP4_RS_WGMMA_GROUP", "4")
    monkeypatch.setenv("FLASHINFER_NVFP4_RS_STATIC_SCHED", "1")
    monkeypatch.setenv("FLASHINFER_NVFP4_RS_NO_UNION", "1")

    assert get_sm90_push_nvfp4_rs_gemm_uri() == baseline
    monkeypatch.setenv("FLASHINFER_SM90_PUSH_NVFP4_RS_WGMMA_GROUP", "2")
    monkeypatch.setenv("FLASHINFER_SM90_PUSH_NVFP4_RS_STATIC_SCHED", "1")
    monkeypatch.setenv("FLASHINFER_SM90_PUSH_NVFP4_RS_NO_UNION", "1")
    assert get_sm90_push_nvfp4_rs_gemm_uri() != baseline


def test_sm90_nvfp4_rs_wgmma_group_must_divide_stage(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_rs_gemm import (
        get_sm90_push_nvfp4_rs_gemm_uri,
    )

    monkeypatch.setenv("FLASHINFER_SM90_PUSH_NVFP4_RS_WGMMA_GROUP", "4")
    with pytest.raises(ValueError, match="does not divide stage_k/32=2"):
        get_sm90_push_nvfp4_rs_gemm_uri(stage_k=64)


def test_sm90_nvfp4_rs_payload_contract_round_trip():
    from flashinfer.fused_moe.sm90_nvfp4_repack import (
        repack_nvfp4_payload_v2,
        unpack_nvfp4_payload_v2,
    )

    packed = torch.arange(2 * 128 * 16, dtype=torch.int64).to(torch.uint8)
    packed = packed.view(2, 128, 16)
    payload_rs = repack_nvfp4_payload_v2(packed)
    reconstructed = unpack_nvfp4_payload_v2(payload_rs)
    torch.testing.assert_close(reconstructed, packed)


@requires_sm90
def test_sm90_nvfp4_rs_rejects_non_divisible_stage_k():
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        create_sm90_push_nvfp4_rs_gemm_runner,
    )

    runner = create_sm90_push_nvfp4_rs_gemm_runner(
        "rs_wgmma", n_tactic=64, stages=3, stage_k=64
    )
    with pytest.raises(Exception, match="K must be divisible by 64"):
        runner.get_workspace_size(32, 2, 64, 96)


@requires_sm90
@pytest.mark.parametrize("n_tactic", [64, 96, 128])
@pytest.mark.parametrize("stages", [3])
@pytest.mark.parametrize("stage_k", [64, 128])
@pytest.mark.parametrize("rows", (0, 1, 31, 32, 63, 64, 65, 127, 128, 129))
def test_sm90_nvfp4_rs_wgmma_matches_direct_oracle(n_tactic, stages, stage_k, rows):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        create_sm90_push_nvfp4_rs_gemm_runner,
    )
    from flashinfer.fused_moe.sm90_nvfp4_repack import (
        build_nvfp4_rs_weight_view,
    )

    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    groups, n, k = 3, 128, stage_k
    offsets = torch.tensor([0, 0, 0, rows], dtype=torch.int64, device=device)
    activation = (torch.randn(rows, k, dtype=torch.float32, device=device) * 0.25).to(
        torch.bfloat16
    )
    packed = torch.randint(
        0, 256, (groups, n, k // 2), dtype=torch.uint8, device=device
    )
    scale_values = torch.tensor(
        [0.25, 0.5, 0.75, 1.0],
        dtype=torch.float32,
        device=device,
    ).repeat(k // 64)
    scales = (
        scale_values.view(1, 1, -1)
        .expand(groups, n, -1)
        .contiguous()
        .to(torch.float8_e4m3fn)
    )
    alpha = torch.tensor([0.5, 1.25, 2.0], dtype=torch.float32, device=device)
    view = build_nvfp4_rs_weight_view(packed, scales, alpha)
    canonical_weights = _dequantize_nvfp4_streams(
        packed, scales, torch.ones_like(alpha)
    ).to(torch.bfloat16)
    alpha_folded_weights = _dequantize_nvfp4_streams(packed, scales, alpha)
    runner = create_sm90_push_nvfp4_rs_gemm_runner(
        "rs_wgmma", n_tactic=n_tactic, stages=stages, stage_k=stage_k
    )
    size = int(runner.get_workspace_size(rows, groups, n, k))
    workspace = torch.empty(max(size, 1), dtype=torch.uint8, device=device)
    runner.configure_workspace(workspace)
    production = torch.empty(rows, n, dtype=torch.bfloat16, device=device)
    oracle = torch.empty_like(production)
    runner.grouped_run(
        production,
        activation,
        view.payload,
        view.scales,
        view.alpha,
        offsets,
        False,
    )
    runner.oracle_run(
        oracle,
        activation,
        canonical_weights,
        alpha,
        offsets,
        False,
    )
    assert torch.equal(production.view(torch.int16), oracle.view(torch.int16))

    expected = _grouped_reference(activation, alpha_folded_weights, offsets)
    torch.testing.assert_close(production.float(), expected, rtol=3e-2, atol=5e-2)
    torch.testing.assert_close(oracle.float(), expected, rtol=3e-2, atol=5e-2)
    if (n_tactic, stages, stage_k) == (64, 3, 64):
        scalar = create_sm90_push_nvfp4_rs_gemm_runner("scalar", n_tactic=64, stages=3)
        scalar_workspace_size = int(scalar.get_workspace_size(rows, groups, n, k))
        scalar_workspace = torch.empty(
            max(scalar_workspace_size, 1), dtype=torch.uint8, device=device
        )
        scalar.configure_workspace(scalar_workspace)
        scalar_output = torch.empty_like(production)
        scalar.grouped_run(
            scalar_output,
            activation,
            view.payload,
            view.scales,
            view.alpha,
            offsets,
            False,
        )
        torch.testing.assert_close(
            scalar_output.float(), expected, rtol=3e-2, atol=5e-2
        )


@requires_sm90
@pytest.mark.parametrize("n_tactic", [64, 96, 128])
@pytest.mark.parametrize("stages", [3])
@pytest.mark.parametrize("stage_k", [64, 128])
def test_sm90_nvfp4_rs_stage_reuse_matches_oracle(n_tactic, stages, stage_k):
    from flashinfer.fused_moe.sm90_nvfp4_repack import (
        build_nvfp4_rs_weight_view,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        create_sm90_push_nvfp4_rs_gemm_runner,
    )

    device = torch.device("cuda", 0)
    groups, rows, n = 2, 33, 64
    k = stage_k * (stages + 1)
    offsets = torch.tensor([0, 17, rows], dtype=torch.int64, device=device)
    activation = (torch.randn(rows, k, dtype=torch.float32, device=device) * 0.25).to(
        torch.bfloat16
    )
    packed = torch.randint(
        0, 256, (groups, n, k // 2), dtype=torch.uint8, device=device
    )
    scales = torch.ones(groups, n, k // 16, dtype=torch.float8_e4m3fn, device=device)
    alpha = torch.tensor([0.75, 1.25], dtype=torch.float32, device=device)
    view = build_nvfp4_rs_weight_view(packed, scales, alpha)
    canonical = _dequantize_nvfp4_streams(packed, scales, torch.ones_like(alpha)).to(
        torch.bfloat16
    )
    alpha_folded = _dequantize_nvfp4_streams(packed, scales, alpha)
    runner = create_sm90_push_nvfp4_rs_gemm_runner(
        "rs_wgmma", n_tactic=n_tactic, stages=stages, stage_k=stage_k
    )
    workspace_size = int(runner.get_workspace_size(rows, groups, n, k))
    workspace = torch.empty(max(workspace_size, 1), dtype=torch.uint8, device=device)
    runner.configure_workspace(workspace)
    output = torch.empty(rows, n, dtype=torch.bfloat16, device=device)
    oracle = torch.empty_like(output)
    runner.grouped_run(
        output,
        activation,
        view.payload,
        view.scales,
        view.alpha,
        offsets,
        False,
    )
    runner.oracle_run(
        oracle,
        activation,
        canonical,
        alpha,
        offsets,
        False,
    )
    torch.cuda.synchronize()
    assert torch.equal(output.view(torch.int16), oracle.view(torch.int16))
    expected = _grouped_reference(activation, alpha_folded, offsets)
    torch.testing.assert_close(output.float(), expected, rtol=3e-2, atol=5e-2)


@requires_sm90
@pytest.mark.parametrize("stage_k", [64, 128])
def test_sm90_nvfp4_rs_padded_prefix_matches_oracle(stage_k):
    from flashinfer.fused_moe.sm90_nvfp4_repack import (
        build_nvfp4_rs_weight_view,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        create_sm90_push_nvfp4_rs_gemm_runner,
    )

    device = torch.device("cuda", 0)
    groups, rows, n, k = 3, 79, 128, stage_k
    padded_offsets = torch.tensor([0, 24, 24, 88], dtype=torch.int64, device=device)
    tile_prefix = torch.tensor([0, 1, 1, 2], dtype=torch.int64, device=device)
    activation = torch.randn(rows, k, dtype=torch.bfloat16, device=device)
    padded_activation = torch.zeros(88, k, dtype=torch.bfloat16, device=device)
    padded_activation[0:17].copy_(activation[0:17])
    padded_activation[24:86].copy_(activation[17:79])
    packed = torch.randint(
        0, 256, (groups, n, k // 2), dtype=torch.uint8, device=device
    )
    scales = torch.ones(groups, n, k // 16, dtype=torch.float8_e4m3fn, device=device)
    alpha = torch.tensor([0.5, 1.25, 2.0], dtype=torch.float32, device=device)
    view = build_nvfp4_rs_weight_view(packed, scales, alpha)
    canonical = _dequantize_nvfp4_streams(packed, scales, torch.ones_like(alpha)).to(
        torch.bfloat16
    )
    alpha_folded = _dequantize_nvfp4_streams(packed, scales, alpha)
    runner = create_sm90_push_nvfp4_rs_gemm_runner(
        "rs_wgmma", n_tactic=64, stages=3, stage_k=stage_k
    )
    workspace_size = int(runner.get_workspace_size(88, groups, n, k))
    workspace = torch.empty(max(workspace_size, 1), dtype=torch.uint8, device=device)
    runner.configure_workspace(workspace)
    output = torch.empty(88, n, dtype=torch.bfloat16, device=device)
    oracle = torch.empty_like(output)
    runner.grouped_run_padded(
        output,
        padded_activation,
        view.payload,
        view.scales,
        view.alpha,
        padded_offsets,
        tile_prefix,
        False,
    )
    runner.oracle_run(
        oracle,
        padded_activation,
        canonical,
        alpha,
        padded_offsets,
        False,
    )
    assert torch.equal(output.view(torch.int16), oracle.view(torch.int16))
    expected = _grouped_reference(padded_activation, alpha_folded, padded_offsets)
    torch.testing.assert_close(output.float(), expected, rtol=3e-2, atol=5e-2)


@requires_sm90
@pytest.mark.parametrize("stage_k", [64, 128])
def test_sm90_nvfp4_rs_s3_large_m_soak(stage_k):
    from flashinfer.fused_moe.sm90_nvfp4_repack import (
        build_nvfp4_rs_weight_view,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        create_sm90_push_nvfp4_rs_gemm_runner,
    )

    device = torch.device("cuda", 0)
    groups, rows, n, k = 1, 12288, 4096, 2048
    offsets = torch.tensor([0, rows], dtype=torch.int64, device=device)
    activation = (torch.randn(rows, k, dtype=torch.float32, device=device) * 0.25).to(
        torch.bfloat16
    )
    packed = torch.randint(
        0, 256, (groups, n, k // 2), dtype=torch.uint8, device=device
    )
    scales = torch.ones(groups, n, k // 16, dtype=torch.float8_e4m3fn, device=device)
    alpha = torch.ones(groups, dtype=torch.float32, device=device)
    view = build_nvfp4_rs_weight_view(packed, scales, alpha)
    reference_weights = _dequantize_nvfp4_streams(packed, scales, alpha)
    expected = _grouped_reference(activation, reference_weights, offsets)
    runner = create_sm90_push_nvfp4_rs_gemm_runner(
        "rs_wgmma", n_tactic=128, stages=3, stage_k=stage_k
    )
    workspace_size = int(runner.get_workspace_size(rows, groups, n, k))
    workspace = torch.empty(max(workspace_size, 1), dtype=torch.uint8, device=device)
    runner.configure_workspace(workspace)
    output = torch.empty(rows, n, dtype=torch.bfloat16, device=device)
    repetitions = int(os.environ.get("SM90_NVFP4_RS_RACE_SOAK_REPS", "10"))
    for _ in range(repetitions):
        runner.grouped_run(
            output,
            activation,
            view.payload,
            view.scales,
            view.alpha,
            offsets,
            False,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(output.float(), expected, rtol=3e-2, atol=5e-2)
