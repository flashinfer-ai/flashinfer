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

from types import SimpleNamespace

import pytest
from packaging.version import Version

from flashinfer.jit import cake_msa
from flashinfer.jit import core as jit_core


EXPECTED_VARIANTS = (
    "decode_bf16_flat",
    "decode_bf16_paged",
    "decode_fp16_flat",
    "decode_fp16_paged",
    "decode_fp8_flat",
    "decode_fp8_paged",
    "decode_m16_bf16_flat",
    "decode_m16_bf16_paged",
    "prefill_m128_bf16_flat",
    "prefill_m128_bf16_gqa16_flat",
    "prefill_m128_bf16_gqa16_paged",
    "prefill_m128_bf16_paged",
    "prefill_m128_fp16_flat",
    "prefill_m128_fp16_paged",
    "prefill_m128_fp8_flat",
    "prefill_m128_fp8_paged",
    "prefill_m64_bf16_flat",
    "topk",
)


def test_cake_msa_variant_manifest_matches_sources():
    csrc_dir = cake_msa._get_cake_msa_csrc_dir()
    binding_variants = tuple(
        sorted(
            path.name.removeprefix("cake_msa_").removesuffix("_binding.cu")
            for path in csrc_dir.glob("cake_msa_*_binding.cu")
        )
    )
    body_variants = tuple(
        sorted(
            path.stem.removeprefix("cake_msa_")
            for path in csrc_dir.glob("cake_msa_*.cu")
            if not path.name.endswith("_binding.cu")
        )
    )

    assert cake_msa.CAKE_MSA_VARIANTS == EXPECTED_VARIANTS
    assert binding_variants == tuple(sorted(EXPECTED_VARIANTS))
    assert body_variants == tuple(sorted(EXPECTED_VARIANTS))


@pytest.mark.parametrize(
    ("target", "target_arch", "expected_flag", "expected_define", "forbidden"),
    [
        (
            "sm100a",
            (10, "0a"),
            "-gencode=arch=compute_100a,code=sm_100a",
            "-DFLASHINFER_CAKE_MSA_TARGET_MINOR=0",
            ("compute_100f", "compute_103a", "compute_120"),
        ),
        (
            "sm100f",
            (10, "0f"),
            "-gencode=arch=compute_100f,code=sm_100f",
            "-DFLASHINFER_CAKE_MSA_TARGET_FAMILY=100",
            ("compute_100a", "compute_103a", "compute_120"),
        ),
    ],
)
def test_cake_msa_uri_and_jit_specs(
    monkeypatch,
    target,
    target_arch,
    expected_flag,
    expected_define,
    forbidden,
):
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {target_arch},
    )
    cake_msa.gen_cake_msa_module.cache_clear()

    for variant in EXPECTED_VARIANTS:
        uri = cake_msa.get_cake_msa_uri(variant, target)
        spec = cake_msa.gen_cake_msa_module(variant, target)

        assert uri == f"cake_msa_{variant}_{target}"
        assert spec.name == uri
        assert len(spec.sources) == 1
        assert spec.sources[0].name == f"cake_msa_{variant}_binding.cu"
        assert spec.sources[0].is_file()
        assert (spec.sources[0].parent / f"cake_msa_{variant}.cu").is_file()
        assert expected_flag in spec.extra_cuda_cflags
        target_defines = [
            flag
            for flag in spec.extra_cuda_cflags
            if flag.startswith("-DFLASHINFER_CAKE_MSA_TARGET_")
        ]
        assert target_defines == [expected_define]
        assert "-use_fast_math" in spec.extra_cuda_cflags
        assert not any(
            compute in flag for compute in forbidden for flag in spec.extra_cuda_cflags
        )
        binding_text = spec.sources[0].read_text()
        assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run," in binding_text
        assert "CheckCakeMsaTarget" in binding_text


def test_cake_msa_validation_getter_and_cache(monkeypatch):
    with pytest.raises(ValueError, match="unsupported CAKE MSA variant"):
        cake_msa.get_cake_msa_uri("unknown", "sm100f")
    with pytest.raises(ValueError, match="unsupported CAKE MSA target"):
        cake_msa.get_cake_msa_uri("topk", "sm103a")

    with monkeypatch.context() as getter_patch:
        sentinel = object()
        getter_patch.setattr(
            cake_msa,
            "load_cake_msa_module",
            lambda variant, target: (sentinel, variant, target),
        )
        assert cake_msa.get_cake_msa_module("topk", "sm100f") == (
            sentinel,
            "topk",
            "sm100f",
        )

    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, "0a"), (10, "0f")},
    )
    cake_msa.gen_cake_msa_module.cache_clear()
    sm100a = cake_msa.gen_cake_msa_module("topk", "sm100a")
    sm100f = cake_msa.gen_cake_msa_module("topk", "sm100f")
    sm100f_cached = cake_msa.gen_cake_msa_module("topk", "sm100f")
    assert sm100a is not sm100f
    assert sm100f is sm100f_cached


@pytest.mark.parametrize(
    ("target_archs", "cuda_version", "expected_legacy", "expected_family"),
    [
        ({(10, "0a")}, "12.8", True, False),
        ({(10, "0a")}, "12.9", False, True),
        ({(10, "0f")}, "12.9", False, True),
        ({(10, "3a")}, "12.8", False, False),
        ({(10, "3a")}, "12.9", False, True),
        ({(10, "3f")}, "13.0", False, True),
        ({(10, "0a"), (10, "3a")}, "13.0", False, True),
        ({(12, "0f")}, "13.0", False, False),
    ],
)
def test_aot_detects_cake_msa_target_matrix(
    monkeypatch,
    target_archs,
    cuda_version,
    expected_legacy,
    expected_family,
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
    assert capabilities["cake_msa_sm100a"] is expected_legacy
    assert capabilities["cake_msa_sm100f"] is expected_family


@pytest.mark.parametrize(
    ("capabilities", "expected_target"),
    [
        ({"cake_msa_sm100a": True}, "sm100a"),
        ({"cake_msa_sm100f": True}, "sm100f"),
    ],
)
def test_aot_registers_all_cake_msa_modules(
    monkeypatch,
    capabilities,
    expected_target,
):
    from flashinfer import aot

    calls = []

    def fake_cake_msa(variant, target):
        calls.append((variant, target))
        return SimpleNamespace(name=f"cake_msa_{variant}_{target}")

    monkeypatch.setattr(aot, "gen_cake_msa_module", fake_cake_msa)
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
        (variant, expected_target) for variant in cake_msa.CAKE_MSA_VARIANTS
    ]
    assert [spec.name for spec in specs] == [
        "spdlog",
        *[
            f"cake_msa_{variant}_{expected_target}"
            for variant in cake_msa.CAKE_MSA_VARIANTS
        ],
        "cudnn",
    ]
