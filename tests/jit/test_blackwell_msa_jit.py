# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

from types import SimpleNamespace

import pytest
from packaging.version import Version

from flashinfer.jit import blackwell_msa


def test_variant_manifest_matches_each_target_source_directory() -> None:
    csrc_dir = blackwell_msa._get_blackwell_msa_csrc_dir()
    assert set(blackwell_msa.BLACKWELL_MSA_VARIANTS_BY_TARGET) == {
        "sm100a",
        "sm103a",
    }
    assert len(blackwell_msa.BLACKWELL_MSA_VARIANTS_BY_TARGET["sm100a"]) == 38
    assert len(blackwell_msa.BLACKWELL_MSA_VARIANTS_BY_TARGET["sm103a"]) == 37
    for target, variants in blackwell_msa.BLACKWELL_MSA_VARIANTS_BY_TARGET.items():
        target_dir = csrc_dir / target
        bodies = tuple(
            sorted(
                path.stem.removeprefix("blackwell_msa_")
                for path in target_dir.glob("blackwell_msa_*.cu")
                if not path.name.endswith("_binding.cu")
            )
        )
        bindings = tuple(
            sorted(
                path.name.removeprefix("blackwell_msa_").removesuffix("_binding.cu")
                for path in target_dir.glob("blackwell_msa_*_binding.cu")
            )
        )
        assert bodies == variants
        assert bindings == variants
    assert (
        "long_prefill_paged_bf16_gqa16_direct_group_sm100"
        in blackwell_msa.BLACKWELL_MSA_VARIANTS_BY_TARGET["sm100a"]
    )
    assert (
        "long_prefill_paged_bf16_gqa16_direct_group_sm100"
        not in blackwell_msa.BLACKWELL_MSA_VARIANTS_BY_TARGET["sm103a"]
    )
    for variants in blackwell_msa.BLACKWELL_MSA_VARIANTS_BY_TARGET.values():
        assert "decode_m16_bf16_paged_topk4_exact512" in variants
        assert not any("active8" in variant for variant in variants)


@pytest.mark.parametrize(
    ("target", "expected_flag", "expected_define", "forbidden"),
    [
        (
            "sm100a",
            "-gencode=arch=compute_100a,code=sm_100a",
            "-DFLASHINFER_BLACKWELL_MSA_TARGET_MINOR=0",
            ("compute_103a", "compute_120"),
        ),
        (
            "sm103a",
            "-gencode=arch=compute_103a,code=sm_103a",
            "-DFLASHINFER_BLACKWELL_MSA_TARGET_MINOR=3",
            ("compute_100a", "compute_120"),
        ),
    ],
)
def test_uri_and_jit_specs(target, expected_flag, expected_define, forbidden) -> None:
    blackwell_msa.gen_blackwell_msa_module.cache_clear()
    for variant in blackwell_msa.BLACKWELL_MSA_VARIANTS_BY_TARGET[target]:
        uri = blackwell_msa.get_blackwell_msa_uri(variant, target)
        spec = blackwell_msa.gen_blackwell_msa_module(variant, target)
        assert uri == f"blackwell_msa_{variant}_{target}"
        assert spec.name == uri
        assert len(spec.sources) == 1
        assert spec.sources[0].name == f"blackwell_msa_{variant}_binding.cu"
        assert spec.sources[0].parent.name == target
        assert (spec.sources[0].parent / f"blackwell_msa_{variant}.cu").is_file()
        assert expected_flag in spec.extra_cuda_cflags
        assert expected_define in spec.extra_cuda_cflags
        assert "-use_fast_math" in spec.extra_cuda_cflags
        assert not any(
            token in flag for token in forbidden for flag in spec.extra_cuda_cflags
        )


def test_validation_getter_and_cache(monkeypatch) -> None:
    with pytest.raises(ValueError, match="unsupported Blackwell MSA target"):
        blackwell_msa.get_blackwell_msa_uri("topk", "unknown")
    with pytest.raises(ValueError, match="variant/target"):
        blackwell_msa.get_blackwell_msa_uri(
            "long_prefill_paged_bf16_gqa16_direct_group_sm100", "sm103a"
        )
    sentinel = object()
    monkeypatch.setattr(
        blackwell_msa,
        "load_blackwell_msa_module",
        lambda variant, target: (sentinel, variant, target),
    )
    assert blackwell_msa.get_blackwell_msa_module("topk", "sm103a") == (
        sentinel,
        "topk",
        "sm103a",
    )


@pytest.mark.parametrize(
    ("target_archs", "cuda_version", "expected_sm100a", "expected_sm103a"),
    [
        ({(10, "0a")}, "12.8", True, False),
        ({(10, "0a")}, "12.9", True, False),
        ({(10, "3a")}, "12.8", False, False),
        ({(10, "3a")}, "12.9", False, True),
        ({(10, "3f")}, "13.0", False, True),
        ({(10, "0a"), (10, "3a")}, "13.0", True, True),
        ({(12, "0f")}, "13.0", False, False),
    ],
)
def test_aot_detects_exact_targets(
    monkeypatch, target_archs, cuda_version, expected_sm100a, expected_sm103a
) -> None:
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
    assert capabilities["blackwell_msa_sm100a"] is expected_sm100a
    assert capabilities["blackwell_msa_sm103a"] is expected_sm103a


@pytest.mark.parametrize("target", ["sm100a", "sm103a"])
def test_aot_registers_target_specific_modules(monkeypatch, target) -> None:
    from flashinfer import aot

    calls = []

    def fake_blackwell_msa(variant, selected_target):
        calls.append((variant, selected_target))
        return SimpleNamespace(name=f"blackwell_msa_{variant}_{selected_target}")

    monkeypatch.setattr(aot, "gen_blackwell_msa_module", fake_blackwell_msa)
    monkeypatch.setattr(
        aot, "gen_spdlog_module", lambda: SimpleNamespace(name="spdlog")
    )
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(
        aot, "gen_cudnn_fmha_module", lambda: SimpleNamespace(name="cudnn")
    )
    capabilities = {f"blackwell_msa_{target}": True}
    aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        capabilities,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    assert calls == [
        (variant, target)
        for variant in blackwell_msa.BLACKWELL_MSA_VARIANTS_BY_TARGET[target]
    ]
