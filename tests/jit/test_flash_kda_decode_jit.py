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

from flashinfer.jit import core as jit_core
from flashinfer.jit import flash_kda_decode


FROZEN_VARIANTS = flash_kda_decode.FLASH_KDA_DECODE_VARIANTS

_FROZEN_BODY_BEGIN = "// BEGIN FROZEN GENERATED BODY\n"
_FROZEN_BODY_END = "// END FROZEN GENERATED BODY\n"

_PHYSICAL_TARGET_CASES = [
    *(
        (
            variant,
            "sm100a",
            (10, "0a"),
            "-gencode=arch=compute_100a,code=sm_100a",
            1000,
        )
        for variant in FROZEN_VARIANTS
    ),
    *(
        (
            variant,
            "sm100f",
            (10, "0f"),
            "-gencode=arch=compute_100f,code=sm_100f",
            100,
        )
        for variant in FROZEN_VARIANTS
    ),
    *(
        (
            variant,
            "sm103a",
            (10, "3a"),
            "-gencode=arch=compute_103a,code=sm_103a",
            1003,
        )
        for variant in (
            "d128_t1_precomputed_direct_split16",
            "d128_t1_precomputed_direct_split8",
        )
    ),
]


@pytest.mark.parametrize(
    ("variant", "target", "target_arch", "expected_flag", "target_kind"),
    _PHYSICAL_TARGET_CASES,
)
def test_flash_kda_decode_jit_spec_and_frozen_body(
    monkeypatch,
    tmp_path,
    variant,
    target,
    target_arch,
    expected_flag,
    target_kind,
):
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {target_arch},
    )
    monkeypatch.setattr(
        flash_kda_decode.jit_env,
        "FLASHINFER_GEN_SRC_DIR",
        tmp_path,
    )
    flash_kda_decode.gen_flash_kda_decode_module.cache_clear()

    uri = flash_kda_decode.get_flash_kda_decode_uri(variant, target)
    spec = flash_kda_decode.gen_flash_kda_decode_module(variant, target)

    assert uri == f"flash_kda_decode_{variant}_{target}"
    assert spec.name == uri
    assert len(spec.sources) == 1
    assert spec.sources[0] == tmp_path / uri / "cake_kda_decode_binding.cu"
    assert spec.sources[0].is_file()
    assert expected_flag in spec.extra_cuda_cflags
    target_defines = [
        flag
        for flag in spec.extra_cuda_cflags
        if flag.startswith("-DFLASHINFER_CAKE_KDA_DECODE_TARGET_KIND=")
    ]
    assert target_defines == [f"-DFLASHINFER_CAKE_KDA_DECODE_TARGET_KIND={target_kind}"]
    assert "-use_fast_math" in spec.extra_cuda_cflags
    assert "--maxrregcount=128" in spec.extra_cuda_cflags
    assert sum("-gencode=arch=compute_" in flag for flag in spec.extra_cuda_cflags) == 1
    assert not any("compute_120" in flag for flag in spec.extra_cuda_cflags)

    frozen_source = flash_kda_decode._get_csrc_dir() / f"flashkda_decode_{variant}.cu"
    frozen_text = frozen_source.read_text()
    assert "Frozen Cake recurrent-KDA export; do not edit by hand." in frozen_text
    assert "SHA256" not in frozen_text
    # Public sources carry the frozen body without generator provenance:
    # no private URLs, internal merge-request IDs, commits, or body digests.
    for private_provenance in (
        "gitlab-master.nvidia.com",
        "merge_requests/",
        "Cake commit",
        "CAKE commit",
        "MR !",
    ):
        assert private_provenance not in frozen_text

    before_body, begin_marker, remainder = frozen_text.partition(_FROZEN_BODY_BEGIN)
    generated_body, end_marker, after_body = remainder.partition(_FROZEN_BODY_END)
    assert begin_marker == _FROZEN_BODY_BEGIN
    assert end_marker == _FROZEN_BODY_END
    assert _FROZEN_BODY_BEGIN not in generated_body
    assert _FROZEN_BODY_END not in generated_body
    assert before_body.rstrip().endswith(
        "// clang-format off\n// Frozen Cake recurrent-KDA export; do not edit by hand."
    )
    assert after_body.strip() == "// clang-format on"
    metadata = flash_kda_decode.FLASH_KDA_DECODE_VARIANT_METADATA[variant]
    expected_gate_kind = metadata.gate_kind
    if metadata.direct_impl:
        assert "#define THREADS 32" in generated_body
        assert "kernel_flashinfer_recurrent_kda_t1_direct" in generated_body
        assert "bool has_token = raw_token_pos >= 0" in generated_body
        assert "int token_pos = ((has_token) ? raw_token_pos : 0)" in generated_body
        assert "else if (has_token && k_lane == 0)" in generated_body
        assert "#define GATE_KIND" not in generated_body
        assert "#define DIRECT_PREFIX_CHECKPOINT" not in generated_body
        assert "#define BLOCK_CHECKPOINT_MMA" not in generated_body
    else:
        assert f"#define GATE_KIND {expected_gate_kind}" in generated_body
        assert "#define DIRECT_PREFIX_CHECKPOINT 0" in generated_body
        assert "#define BLOCK_CHECKPOINT_MMA 0" in generated_body

    binding_text = spec.sources[0].read_text()
    assert (
        f'#define CAKE_KDA_DECODE_BODY_FILE "flashkda_decode_{variant}.cu"'
        in binding_text
    )
    assert f"#define CAKE_KDA_DECODE_HEAD_DIM {metadata.head_dim}" in binding_text
    assert f"#define CAKE_KDA_DECODE_TOKENS {metadata.tokens}" in binding_text
    assert f"#define CAKE_KDA_DECODE_GATE_KIND {expected_gate_kind}" in binding_text
    assert f"#define CAKE_KDA_DECODE_VALUE_SPLIT {metadata.value_split}" in binding_text
    assert (
        f"#define CAKE_KDA_DECODE_LAUNCH_THREADS {metadata.launch_threads}"
        in binding_text
    )
    assert "#define CAKE_KDA_DECODE_WARPS_PER_CTA 1" in binding_text
    assert (
        "#define CAKE_KDA_DECODE_DIRECT_IMPL 1" in binding_text
    ) is metadata.direct_impl
    assert '#include "cake_kda_decode_binding.cuh"' in binding_text
    flash_kda_decode.gen_flash_kda_decode_module.cache_clear()


def test_flash_kda_decode_binding_contract():
    csrc_dir = flash_kda_decode._get_csrc_dir()
    binding = (csrc_dir / "cake_kda_decode_binding.cuh").read_text()
    common = (csrc_dir / "cake_kda_decode_binding_common.cuh").read_text()
    impl = (csrc_dir / "cake_kda_decode_binding_impl.cuh").read_text()
    direct_impl = (csrc_dir / "cake_kda_decode_binding_direct_impl.cuh").read_text()

    assert "CAKE_KDA_DECODE_BODY_FILE" in binding
    assert "#include CAKE_KDA_DECODE_BODY_FILE" in binding
    assert "#ifdef CAKE_KDA_DECODE_DIRECT_IMPL" in binding
    assert '#include "cake_kda_decode_binding_direct_impl.cuh"' in binding
    assert '#include "cake_kda_decode_binding_impl.cuh"' in binding
    assert "#ifndef FLASHINFER_CAKE_KDA_DECODE_TARGET_KIND" in common
    assert "kCakeKDADecodeTargetKind == kCakeKDADecodeFamilyTarget" in common
    assert "kCakeKDADecodeTargetKind == kCakeKDADecodeExactSM100aTarget" in common
    assert "kCakeKDADecodeTargetKind == kCakeKDADecodeExactSM103aTarget" in common
    assert "major == 10 && (minor == 0 || minor == 3)" in common
    assert "major == 10 && minor == expected_minor" in common
    assert "CheckCakeKDADecodeTarget(device_id)" in common
    assert "struct VariantTraits" in common
    assert "static_assert(Tokens >= 1)" in common
    assert "GateKind == 0 || GateKind == 1 || GateKind == 2" in common
    assert "ValueSplit == 16" in common
    assert "HeadDim % ValueSplit == 0" in common
    assert "state.stride(0) >= num_value_heads * head_dim * head_dim" in common
    assert "gate.stride(1) >= num_value_heads * head_dim" in common
    assert "g must be compact in its [HV, K] trailing dimensions" in common
    assert 'CheckNoOverlap(out, "output", state, "initial_state")' in common
    assert "lower-bound gate variants require at least H A_log values" in common
    assert "lower-bound gate variants require at least H * D dt_bias values" in common
    assert "finite negative lower_bound" in common
    assert "only the T1 unbounded-softplus variant accepts beta logits" in common
    assert "torch.cuda.current_stream" not in impl
    assert "int64_t beta_is_logit, int64_t cuda_stream" in impl
    assert "CAKE_KDA_DECODE_VALUE_SPLIT" in impl
    assert "SMEM_TOTAL > 0" in impl
    assert "kernel_flashinfer_recurrent_kda_wy_vtile_short" in impl
    assert "kernel_flashinfer_recurrent_kda_t1_direct" in direct_impl
    assert "SMEM_TOTAL" not in direct_impl
    assert "int64_t beta_is_logit, int64_t cuda_stream" in direct_impl


def test_flash_kda_decode_variant_validation_and_getter(monkeypatch):
    expected_variants = (
        "d128_t1_precomputed_direct_split16",
        "d128_t1_precomputed_direct_split8",
        "d128_t2_precomputed_split4",
        "d128_t2_precomputed_split8",
        "d128_t3_lower_bound_split4",
        "d128_t4_precomputed_split1",
        "d128_t4_precomputed_split2",
        "d128_t4_precomputed_split4",
        "d128_t4_precomputed_split8",
        "d128_t5_precomputed_gram_split1",
        "d128_t5_precomputed_gram_split2",
        "d128_t5_precomputed_gram_split4",
        "d128_t5_precomputed_gram_split8",
        "d128_t6_precomputed_gram_split1",
        "d128_t6_precomputed_gram_split2",
        "d128_t6_precomputed_gram_split4",
        "d128_t6_precomputed_gram_split8",
    )
    assert expected_variants == flash_kda_decode.FLASH_KDA_DECODE_VARIANTS
    assert expected_variants == tuple(
        flash_kda_decode.FLASH_KDA_DECODE_VARIANT_METADATA
    )
    assert {
        variant: metadata.launch_threads
        for variant, metadata in (
            flash_kda_decode.FLASH_KDA_DECODE_VARIANT_METADATA.items()
        )
    } == {
        "d128_t1_precomputed_direct_split16": 32,
        "d128_t1_precomputed_direct_split8": 32,
        "d128_t2_precomputed_split4": 64,
        "d128_t2_precomputed_split8": 64,
        "d128_t3_lower_bound_split4": 96,
        "d128_t4_precomputed_split1": 256,
        "d128_t4_precomputed_split2": 128,
        "d128_t4_precomputed_split4": 128,
        "d128_t4_precomputed_split8": 128,
        "d128_t5_precomputed_gram_split1": 256,
        "d128_t5_precomputed_gram_split2": 160,
        "d128_t5_precomputed_gram_split4": 160,
        "d128_t5_precomputed_gram_split8": 160,
        "d128_t6_precomputed_gram_split1": 256,
        "d128_t6_precomputed_gram_split2": 192,
        "d128_t6_precomputed_gram_split4": 192,
        "d128_t6_precomputed_gram_split8": 192,
    }
    direct_variants = {
        variant
        for variant, metadata in (
            flash_kda_decode.FLASH_KDA_DECODE_VARIANT_METADATA.items()
        )
        if metadata.direct_impl
    }
    assert direct_variants == {
        "d128_t1_precomputed_direct_split16",
        "d128_t1_precomputed_direct_split8",
    }
    assert set(flash_kda_decode.FLASH_KDA_DECODE_DIRECT_VARIANTS) == direct_variants
    for removed_variant in (
        "d128_t4_precomputed",
        "d128_t5_precomputed",
        "d128_t5_precomputed_gram",
        "d128_t5_precomputed_gram_split3",
    ):
        with pytest.raises(ValueError, match="unsupported FlashKDA decode variant"):
            flash_kda_decode.get_flash_kda_decode_uri(removed_variant, "sm100f")
    with pytest.raises(ValueError, match="unsupported FlashKDA decode target"):
        flash_kda_decode.get_flash_kda_decode_uri(expected_variants[0], "sm120a")
    non_direct_variant = next(
        variant for variant in expected_variants if variant not in direct_variants
    )
    with pytest.raises(ValueError, match="only retained for direct T=1"):
        flash_kda_decode.get_flash_kda_decode_uri(non_direct_variant, "sm103a")
    sentinel = object()
    monkeypatch.setattr(
        flash_kda_decode,
        "load_flash_kda_decode_module",
        lambda variant, target: (sentinel, variant, target),
    )
    for variant in expected_variants:
        assert flash_kda_decode.get_flash_kda_decode_module(variant, "sm100f") == (
            sentinel,
            variant,
            "sm100f",
        )


def test_flash_kda_decode_physical_targets_have_deliberate_cache_keys(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, "0a"), (10, "3a")},
    )
    monkeypatch.setattr(
        flash_kda_decode.jit_env,
        "FLASHINFER_GEN_SRC_DIR",
        tmp_path,
    )
    flash_kda_decode.gen_flash_kda_decode_module.cache_clear()

    variant = flash_kda_decode.FLASH_KDA_DECODE_DIRECT_VARIANTS[0]
    family = flash_kda_decode.gen_flash_kda_decode_module(variant, "sm100f")
    cached_family = flash_kda_decode.gen_flash_kda_decode_module(variant, "sm100f")
    legacy = flash_kda_decode.gen_flash_kda_decode_module(variant, "sm100a")
    gb300_direct = flash_kda_decode.gen_flash_kda_decode_module(variant, "sm103a")

    assert family is cached_family
    assert len({family.name, legacy.name, gb300_direct.name}) == 3
    assert family.name == f"flash_kda_decode_{variant}_sm100f"
    assert legacy.name == f"flash_kda_decode_{variant}_sm100a"
    assert gb300_direct.name == f"flash_kda_decode_{variant}_sm103a"
    flash_kda_decode.gen_flash_kda_decode_module.cache_clear()


@pytest.mark.parametrize(
    (
        "target_archs",
        "cuda_version",
        "expected_legacy",
        "expected_family",
        "expected_sm103_direct",
    ),
    [
        ({(10, "0a")}, "12.8", True, False, False),
        ({(10, "0a")}, "12.9", False, True, False),
        ({(10, "0f")}, "13.0", False, True, False),
        ({(10, "3a")}, "12.8", False, False, False),
        ({(10, "3a")}, "12.9", False, True, True),
        ({(10, "3f")}, "13.0", False, True, True),
        ({(10, "0a"), (10, "3a")}, "13.0", False, True, True),
        ({(12, "0f")}, "13.0", False, False, False),
    ],
)
def test_aot_detects_flash_kda_decode_physical_targets(
    monkeypatch,
    target_archs,
    cuda_version,
    expected_legacy,
    expected_family,
    expected_sm103_direct,
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
    assert capabilities["flash_kda_decode_sm100a_legacy"] is expected_legacy
    assert capabilities["flash_kda_decode_sm100f"] is expected_family
    assert capabilities["flash_kda_decode_sm103a_direct"] is expected_sm103_direct


@pytest.mark.parametrize(
    ("capabilities", "expected_targets"),
    [
        (
            {"flash_kda_decode_sm100a_legacy": True},
            [
                (variant, "sm100a")
                for variant in flash_kda_decode.FLASH_KDA_DECODE_VARIANTS
            ],
        ),
        (
            {"flash_kda_decode_sm100f": True},
            [
                (variant, "sm100f")
                for variant in flash_kda_decode.FLASH_KDA_DECODE_VARIANTS
            ],
        ),
        (
            {
                "flash_kda_decode_sm100f": True,
                "flash_kda_decode_sm103a_direct": True,
            },
            [
                *[
                    (variant, "sm100f")
                    for variant in flash_kda_decode.FLASH_KDA_DECODE_VARIANTS
                ],
                *[
                    (variant, "sm103a")
                    for variant in flash_kda_decode.FLASH_KDA_DECODE_DIRECT_VARIANTS
                ],
            ],
        ),
    ],
)
def test_aot_registers_flash_kda_decode_physical_portfolio(
    monkeypatch, capabilities, expected_targets
):
    from flashinfer import aot

    calls = []

    def fake_flash_kda_decode(variant, target):
        calls.append((variant, target))
        return SimpleNamespace(name=f"flash_kda_decode_{variant}_{target}")

    monkeypatch.setattr(
        aot,
        "gen_flash_kda_decode_module",
        fake_flash_kda_decode,
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
        capabilities,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )

    assert calls == expected_targets
    assert [spec.name for spec in specs] == [
        "spdlog",
        *(
            f"flash_kda_decode_{variant}_{target}"
            for variant, target in expected_targets
        ),
        "cudnn",
    ]
