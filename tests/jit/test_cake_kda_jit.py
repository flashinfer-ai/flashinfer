# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.version import Version

from flashinfer.jit import cake_kda
from flashinfer.jit import core as jit_core


def test_cake_kda_unbounded_prefill_manifest_matches_frozen_source():
    csrc_dir = cake_kda._get_cake_kda_csrc_dir()
    manifest = json.loads(
        (
            csrc_dir
            / "cake_kda_bf16_fused_m128_unbounded_softplus_import_manifest.json"
        ).read_text()
    )

    assert manifest["schema_version"] == 2
    assert set(manifest["profiles"]) == {
        "sm100_unbounded_softplus",
        "sm103_unbounded_softplus",
    }
    assert set(manifest["variants"]) == {"unbounded_softplus"}
    record = manifest["variants"]["unbounded_softplus"]
    frozen = csrc_dir / "cake_kda_bf16_fused_m128_unbounded_softplus.cu"
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == record["frozen_sha256"]
    assert record["frozen_sha256"] == (
        "94d641ecb2a28235ff1da1cc8160d0e9698446ec14c6eb01a05eb31662f460f2"
    )
    assert record["module_ident"] == "cake_kda_bf16_fused_m128_d7a7b33c69"
    assert record["source_module_ident"] == "flashkda_bf16_fused_m128_d7a7b33c69"
    assert (
        Path(__file__).resolve().parents[2] / "tools" / "import-cake-kda-prefill"
    ).is_file()


@pytest.mark.parametrize(
    ("target", "target_arch", "expected_flag", "expected_define"),
    [
        (
            "sm100a",
            (10, "0a"),
            "-gencode=arch=compute_100a,code=sm_100a",
            "-DFLASHINFER_CAKE_KDA_TARGET_MINOR=0",
        ),
        (
            "sm103a",
            (10, "3a"),
            "-gencode=arch=compute_103a,code=sm_103a",
            "-DFLASHINFER_CAKE_KDA_TARGET_MINOR=3",
        ),
    ],
)
def test_cake_kda_unbounded_prefill_jit_spec(
    monkeypatch, target, target_arch, expected_flag, expected_define
):
    monkeypatch.setattr(
        jit_core.current_compilation_context, "TARGET_CUDA_ARCHS", {target_arch}
    )
    cake_kda.gen_cake_kda_module.cache_clear()

    variant = "m128_unbounded_softplus"
    uri = cake_kda.get_cake_kda_uri(variant, target)
    spec = cake_kda.gen_cake_kda_module(variant, target)

    assert uri == f"cake_kda_bf16_fused_{variant}_d7a7b33c69_{target}"
    assert spec.name == uri
    assert spec.sources == [
        cake_kda._get_cake_kda_csrc_dir()
        / "cake_kda_bf16_fused_m128_unbounded_softplus_binding.cu"
    ]
    assert expected_flag in spec.extra_cuda_cflags
    assert expected_define in spec.extra_cuda_cflags


@pytest.mark.parametrize(
    ("target_archs", "cuda_version", "expected_sm100a", "expected_sm103a"),
    [
        ({(10, "0a")}, "12.8", True, False),
        ({(10, "0a")}, "12.9", True, False),
        ({(10, "3a")}, "12.8", False, False),
        ({(10, "3a")}, "12.9", False, True),
        ({(10, "0a"), (10, "3a")}, "13.0", True, True),
    ],
)
def test_aot_detects_cake_kda_prefill_exact_targets(
    monkeypatch, target_archs, cuda_version, expected_sm100a, expected_sm103a
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

    capabilities = aot.detect_sm_capabilities()
    assert capabilities["cake_kda_prefill_sm100a"] is expected_sm100a
    assert capabilities["cake_kda_prefill_sm103a"] is expected_sm103a


def test_aot_registers_cake_kda_unbounded_prefill_targets(monkeypatch):
    from flashinfer import aot

    calls = []

    def fake_cake_kda(target):
        calls.append(target)
        return SimpleNamespace(name=f"cake_kda_unbounded_{target}")

    monkeypatch.setattr(
        aot, "gen_cake_kda_m128_unbounded_softplus_module", fake_cake_kda
    )
    monkeypatch.setattr(
        aot, "gen_spdlog_module", lambda: SimpleNamespace(name="spdlog")
    )
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(
        aot, "gen_cudnn_fmha_module", lambda: SimpleNamespace(name="cudnn")
    )

    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        {"cake_kda_prefill_sm100a": True, "cake_kda_prefill_sm103a": True},
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    assert calls == ["sm100a", "sm103a"]
    assert [spec.name for spec in specs] == [
        "spdlog",
        "cake_kda_unbounded_sm100a",
        "cake_kda_unbounded_sm103a",
        "cudnn",
    ]
