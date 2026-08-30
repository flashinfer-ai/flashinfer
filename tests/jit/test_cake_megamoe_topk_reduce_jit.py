# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.version import Version

from flashinfer.jit import cake_megamoe_topk_reduce
from flashinfer.jit import core as jit_core


def _write_bundle(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    source = (
        cake_megamoe_topk_reduce._get_csrc_dir() / cake_megamoe_topk_reduce._SOURCE_FILE
    ).read_bytes()
    (directory / cake_megamoe_topk_reduce._SOURCE_FILE).write_bytes(source)
    (directory / cake_megamoe_topk_reduce._BINDING_HEADER).write_text(
        "// test binding\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "arch": "sm_100a",
        "compile_flags": [],
        "tma_abi": "pointer",
        "kernel_count": 1,
        "launch": copy.deepcopy(cake_megamoe_topk_reduce._LAUNCH),
        "constraints": copy.deepcopy(cake_megamoe_topk_reduce._CONSTRAINTS),
        "kernel_symbols": [cake_megamoe_topk_reduce._KERNEL_SYMBOL],
        "source_sha256": hashlib.sha256(source).hexdigest(),
    }
    (directory / cake_megamoe_topk_reduce._MANIFEST_FILE).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def test_packaged_bundle_matches_frozen_manifest():
    source, manifest = cake_megamoe_topk_reduce._program_source()

    assert source.name == cake_megamoe_topk_reduce._SOURCE_FILE
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        cake_megamoe_topk_reduce._SOURCE_SHA256
    )
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["kernel_symbols"] == [cake_megamoe_topk_reduce._KERNEL_SYMBOL]


def test_manifest_loader_accepts_only_the_frozen_identity(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    manifest = _write_bundle(bundle)
    monkeypatch.setattr(cake_megamoe_topk_reduce, "_get_csrc_dir", lambda: bundle)

    source, loaded = cake_megamoe_topk_reduce._program_source()

    assert source == bundle / cake_megamoe_topk_reduce._SOURCE_FILE
    assert loaded == manifest


def test_manifest_loader_rejects_joint_source_and_manifest_drift(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    manifest = _write_bundle(bundle)
    source_path = bundle / cake_megamoe_topk_reduce._SOURCE_FILE
    source_path.write_bytes(source_path.read_bytes() + b"\n// drift\n")
    manifest["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    (bundle / cake_megamoe_topk_reduce._MANIFEST_FILE).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(cake_megamoe_topk_reduce, "_get_csrc_dir", lambda: bundle)

    with pytest.raises(RuntimeError, match="source identity is invalid"):
        cake_megamoe_topk_reduce._program_source()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("arch", "sm_103a"),
        ("compile_flags", ["--use_fast_math"]),
        ("kernel_count", 2),
        ("tma_abi", "inline"),
        (
            "launch",
            {
                "block_threads": 64,
                "dynamic_smem_bytes": 0,
                "grid_x": "4 * num_tokens",
            },
        ),
        (
            "constraints",
            {
                "capacities": [256],
                "dtype": "bfloat16",
                "hidden_size": 4096,
                "top_k": 6,
            },
        ),
        ("kernel_symbols", ["different_kernel"]),
        ("source_sha256", "0" * 64),
    ],
)
def test_manifest_loader_rejects_identity_drift(monkeypatch, tmp_path, field, value):
    bundle = tmp_path / "bundle"
    manifest = _write_bundle(bundle)
    manifest[field] = value
    (bundle / cake_megamoe_topk_reduce._MANIFEST_FILE).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(cake_megamoe_topk_reduce, "_get_csrc_dir", lambda: bundle)

    with pytest.raises(RuntimeError, match="manifest identity is invalid"):
        cake_megamoe_topk_reduce._program_source()


def test_manifest_loader_rejects_duplicate_and_unknown_keys(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    manifest = _write_bundle(bundle)
    manifest_path = bundle / cake_megamoe_topk_reduce._MANIFEST_FILE
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8")[:-1] + ', "arch": "sm_100a"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(cake_megamoe_topk_reduce, "_get_csrc_dir", lambda: bundle)
    with pytest.raises(RuntimeError, match="duplicate key 'arch'"):
        cake_megamoe_topk_reduce._program_source()

    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest identity is invalid"):
        cake_megamoe_topk_reduce._program_source()


def test_manifest_loader_rejects_incomplete_bundle(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setattr(cake_megamoe_topk_reduce, "_get_csrc_dir", lambda: bundle)

    with pytest.raises(RuntimeError, match="source package is incomplete"):
        cake_megamoe_topk_reduce._program_source()


def test_jit_spec_is_content_addressed_and_binds_frozen_launch(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    include = tmp_path / "include"
    include.mkdir()
    generated = tmp_path / "generated"
    monkeypatch.setattr(cake_megamoe_topk_reduce, "_get_csrc_dir", lambda: bundle)
    monkeypatch.setattr(cake_megamoe_topk_reduce, "_get_include_dir", lambda: include)
    monkeypatch.setattr(
        cake_megamoe_topk_reduce.jit_env,
        "FLASHINFER_GEN_SRC_DIR",
        generated,
    )
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, "0a")},
    )
    cake_megamoe_topk_reduce.gen_cake_megamoe_topk_reduce_module.cache_clear()

    spec = cake_megamoe_topk_reduce.gen_cake_megamoe_topk_reduce_module()
    binding = spec.sources[0].read_text(encoding="utf-8")

    assert spec.name.startswith("cake_megamoe_topk_reduce_sm100a_")
    assert spec.sources == [
        generated / spec.name / "cake_megamoe_topk_reduce_binding.cu"
    ]
    assert "-gencode=arch=compute_100a,code=sm_100a" in spec.extra_cuda_cflags
    assert "-use_fast_math" not in spec.extra_cuda_cflags
    assert (
        f'#define CAKE_MEGAMOE_TOPK_REDUCE_BODY_FILE "{cake_megamoe_topk_reduce._SOURCE_FILE}"'
        in binding
    )
    assert (
        f"#define CAKE_MEGAMOE_TOPK_REDUCE_KERNEL {cake_megamoe_topk_reduce._KERNEL_SYMBOL}"
        in binding
    )
    assert "#define CAKE_MEGAMOE_TOPK_REDUCE_THREADS 128" in binding
    assert "#define CAKE_MEGAMOE_TOPK_REDUCE_SMEM_BYTES 0" in binding
    assert cake_megamoe_topk_reduce.gen_cake_megamoe_topk_reduce_module() is spec
    cake_megamoe_topk_reduce.gen_cake_megamoe_topk_reduce_module.cache_clear()


def test_python_launch_uses_partials_current_stream(monkeypatch):
    calls = []
    module = SimpleNamespace(run=lambda *args: calls.append(args))
    stream = SimpleNamespace(cuda_stream=12345)
    partials = SimpleNamespace(device="cuda:1")
    out = object()
    monkeypatch.setattr(
        cake_megamoe_topk_reduce,
        "get_cake_megamoe_topk_reduce_module",
        lambda: module,
    )
    monkeypatch.setattr(
        cake_megamoe_topk_reduce.torch.cuda,
        "current_stream",
        lambda *, device: stream if device == "cuda:1" else None,
    )

    cake_megamoe_topk_reduce.run_cake_megamoe_topk_reduce(partials, out, 17)

    assert calls == [(partials, out, 17, 12345)]


def test_binding_enforces_full_runtime_contract():
    binding = (
        cake_megamoe_topk_reduce._get_csrc_dir()
        / cake_megamoe_topk_reduce._BINDING_HEADER
    ).read_text(encoding="utf-8")

    assert "CHECK_INPUT_TYPE(partials, dl_bfloat16)" in binding
    assert "CHECK_INPUT_TYPE(out, dl_bfloat16)" in binding
    assert "partials must have shape [256 or 4096, 6, 4096]" in binding
    assert "out must have shape [capacity, 4096] matching partials" in binding
    assert "num_tokens >= 0 && num_tokens <= capacity" in binding
    assert "partials must be 128-byte aligned" in binding
    assert "out must be 128-byte aligned" in binding
    assert 'CheckNoOverlap(partials, "partials", out, "out")' in binding
    assert "major == 10 && minor == 0" in binding
    assert "if (num_tokens == 0)" in binding
    assert "kGridCTAsPerToken * num_tokens" in binding
    assert "reinterpret_cast<cudaStream_t>(cuda_stream)" in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run" in binding


@pytest.mark.parametrize(
    ("target_archs", "cuda_version", "expected"),
    [
        ({(10, "0a")}, "12.8", True),
        ({(10, "0a")}, "13.0", True),
        ({(10, "0a")}, "12.7", False),
        ({(10, "0f")}, "12.9", False),
        ({(10, "3a")}, "12.9", False),
    ],
)
def test_aot_detects_exact_sm100a_reducer(
    monkeypatch, target_archs, cuda_version, expected
):
    from flashinfer import aot

    class FakeCompilationContext:
        TARGET_CUDA_ARCHS = target_archs

        def get_nvcc_flags_list(self, supported_major_versions=None):
            del supported_major_versions
            return [
                f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
                for major, minor in sorted(self.TARGET_CUDA_ARCHS)
            ]

    monkeypatch.setattr(aot, "CompilationContext", FakeCompilationContext)
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version(cuda_version))

    assert aot.detect_sm_capabilities()["cake_megamoe_topk_reduce_sm100a"] is expected


@pytest.mark.parametrize("enabled", [False, True])
def test_aot_inventory_registers_reducer_only_when_enabled(monkeypatch, enabled):
    from flashinfer import aot

    calls = []

    def named(name):
        return SimpleNamespace(name=name)

    monkeypatch.setattr(
        aot,
        "gen_cake_megamoe_topk_reduce_module",
        lambda: calls.append("cake_megamoe_topk_reduce")
        or named("cake_megamoe_topk_reduce"),
    )
    monkeypatch.setattr(aot, "gen_spdlog_module", lambda: named("spdlog"))
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(aot, "gen_gemm_module", lambda: named("gemm"))
    monkeypatch.setattr(aot, "gen_bgmv_moe_module", lambda: named("bgmv"))
    monkeypatch.setattr(aot, "gen_hash_topk_module", lambda: named("hash_topk"))
    monkeypatch.setattr(aot, "gen_cudnn_fmha_module", lambda: named("cudnn"))

    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        {"cake_megamoe_topk_reduce_sm100a": enabled},
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    )

    assert calls == (["cake_megamoe_topk_reduce"] if enabled else [])
    assert [spec.name for spec in specs] == [
        "spdlog",
        "gemm",
        "bgmv",
        "hash_topk",
        *(["cake_megamoe_topk_reduce"] if enabled else []),
        "cudnn",
    ]
