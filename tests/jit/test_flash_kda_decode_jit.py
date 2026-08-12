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

import hashlib
import re
from types import SimpleNamespace

import pytest
from packaging.version import Version

from flashinfer.jit import core as jit_core
from flashinfer.jit import flash_kda_decode


# Each tuple is (raw frozen-body SHA256, normalized generated-body SHA256).
# The normalization removes only identity-derived temporary suffixes; all
# schedule-relevant source text remains covered by the stable digest.
FROZEN_GENERATED_BODY_SHA256 = {
    "d128_t1_precomputed_split1": (
        "c50a65997603f729d1558bfd9e14ce8d3415add9dc32aab2f958ab414307a3b4",
        "a12ff529c81d7dc375e31bf57137d348b1eb39a93a6206d6aacc99515dae26f4",
    ),
    "d128_t1_precomputed_split2": (
        "227b3b95337bb231798e7b5c2509a2d117a7337a6a6279010be6b71d3c8e0fb1",
        "f0c24746584b0f6a19b5aefa977b71852cb920c9d1a626c73616dbb458ebd13e",
    ),
    "d128_t1_precomputed_split4": (
        "30ff1318b9a7bc9904202e7bed93234378b6a7f231fa67ec00a39d2ae5e98f72",
        "f7eba8a6b9ac933d7b54bcd333400fd640b90a54bf7dcb12c308f588a6533d6b",
    ),
    "d128_t1_precomputed_split8": (
        "7b613b50ac99df6cc55bb349deb8dc01adf0f49b049636775a486a41df2a77ae",
        "a38718a0b4383d2f2d2c3ec7eb1746be44e649265b805a3dbcebdd20415612d3",
    ),
    "d128_t1_precomputed_direct_split16": (
        "9119163b3b5cb6a8760b6a17a7ce01788a0e1c0078f8c812255225f27e5989e5",
        "9119163b3b5cb6a8760b6a17a7ce01788a0e1c0078f8c812255225f27e5989e5",
    ),
    "d128_t1_precomputed_direct_split8": (
        "116612419179c171252cbdfa11b2e531a2c2a12ddf5b8eecb2874ecd2dea396f",
        "116612419179c171252cbdfa11b2e531a2c2a12ddf5b8eecb2874ecd2dea396f",
    ),
    "d128_t2_precomputed_split1": (
        "4eaf6b81491db3bd7969ddd8f41797bd7d5e472c591b4ed6a9a8fd7172eb6047",
        "ab378468d5b81e203e09405adc49004f30a0e178ec29986f40e31583de1dd291",
    ),
    "d128_t2_precomputed_split2": (
        "3d08321d288f816d21bfffc93bf25b7df4631b9f500799a4a7036a226c4250b2",
        "54eadc2257ca591d7af94d8176f6562a564b4e72bc90e959a18b85a7a1af597a",
    ),
    "d128_t2_precomputed_split4": (
        "e0cf2ece1b50e851579df8822fa2f03262a3399e25643a6bd3df03c875cc5ea2",
        "bd7a8ae1bd202df4fc534f628be755db3c515a6755770b3a9fea9cae08cba780",
    ),
    "d128_t2_precomputed_split8": (
        "b420190365e5f60432301956dec4ecfbd1b6dcba3c6034a5d1847f8c300a1501",
        "1cee7e4aee0953f701e09f39c7edfdf1dac2abb6037628b291bec71bb34fd2af",
    ),
    "d128_t3_lower_bound_split4": (
        "70c838b717a9eb7765bf9291db786b2b7a0387fbf80a3c337d5c04e7a553fe21",
        "76976b198f567bd68ed246343d9e795b9ca2883dce89033e20e2125cb4a84b82",
    ),
    "d128_t4_precomputed_split1": (
        "d9eb05f21e8c76a92b25d1ce288db603c9ed8454e271cca1c11171b6d818a326",
        "996815363f81f0d79dd517fcc7ed20216ea9509c87073275bda6d2ecf70a4c58",
    ),
    "d128_t4_precomputed_split2": (
        "6ee50e300a2f20f305252a49e3e84fde957542ccb97b7fcb7ba7f075c7733077",
        "581c722ec4d490853a382dfaa6204c909aa778637818de960627f1aa1e2b3fc0",
    ),
    "d128_t4_precomputed_split4": (
        "1fa5ac877060a9c7a003e454eb6d69a4fb9a7720871c89a7aaf403de186fd934",
        "da9042cd0bd0ad972ecf423ac7f56eaa2b8d37718a8a273552ac003f8c7aceb2",
    ),
    "d128_t4_precomputed_split8": (
        "1a648fba1d1e3d899582d0ef85af869754fb8e2276d25cfc2b4d01c3480fe967",
        "e0f65928fe3fc77bebf664bc369d8f35e0ecfde0cd1db539f1702bf805d320b3",
    ),
    "d128_t5_precomputed_gram_split1": (
        "7d44765cc20864dca2fc5f96ed2ae653e4d421f963c20fed5ba825d4989c8b4e",
        "3716601286aae19dde3d52020a97966b5f6b973ffcb128dfee6995855843c851",
    ),
    "d128_t5_precomputed_gram_split2": (
        "d8c24892ac7e456fd04c51fe820ecedfc677c3be0e823cc4919043da7ae025af",
        "9bdbf25cd89369d522e7a30e60ebc20c39dc5c5049a63930d57bd7d1ecfc7789",
    ),
    "d128_t5_precomputed_gram_split4": (
        "93020b1e878a584146d7f9c1a4e46c98bd05bcf8fa4f83c41fb6dcd4e616377d",
        "97eb2e934f3e12342ca5dc699cea23228e47a62afd52554cc47daf73482aa569",
    ),
    "d128_t5_precomputed_gram_split8": (
        "2307b896466dd58ff1daba770763b0a7142451e73225e940e9e9461a21bb9452",
        "1de157a38002ebc51ab603e79545eeea92d6fc13d53ed307e595ee29d04a8a02",
    ),
    "d128_t6_precomputed_gram_split1": (
        "6684bee1c3e1354082e8603574fe43a5b37991dd2767c9b33104d97cd84bbc55",
        "f2ea538c2bb191ca896efdaf5ec81c10922a36fe6aac4781e28cec1f793bff0b",
    ),
    "d128_t6_precomputed_gram_split2": (
        "27e047dc15f2bb4972718b15143d1ebff6cfe21c42aa5688db8245a22c7eba55",
        "eb79e4a0a7ce02b84ef52e71ff34bf5c3bb1d7595a76fbcf0f03c83124c8cd82",
    ),
    "d128_t6_precomputed_gram_split4": (
        "f38dc82927095886ce8b030becd0761124d9a0aab3a8373efd17818f7f760463",
        "bcd1f00f81fbeb45803def145b3dd11517a8427925a2cc383667d7934cd7da64",
    ),
    "d128_t6_precomputed_gram_split8": (
        "96593303ff3d6dbb7e9c603f1b76d9eb87eb21f03341a796ea63cdd9e8ee1dac",
        "cf36709e7727d702c34ee87b4123e4f9d77f9f8757eb0a26aecb25b1dd690cbd",
    ),
}

_FROZEN_BODY_BEGIN = "// BEGIN FROZEN GENERATED BODY\n"
_FROZEN_BODY_END = "// END FROZEN GENERATED BODY\n"
_VOLATILE_TEMP = re.compile(r"\b_(bval|bits|addr)_([0-9]{6,})\b")

_PHYSICAL_TARGET_CASES = [
    *(
        (
            variant,
            "sm100a",
            (10, "0a"),
            "-gencode=arch=compute_100a,code=sm_100a",
            1000,
        )
        for variant in FROZEN_GENERATED_BODY_SHA256
    ),
    *(
        (
            variant,
            "sm100f",
            (10, "0f"),
            "-gencode=arch=compute_100f,code=sm_100f",
            100,
        )
        for variant in FROZEN_GENERATED_BODY_SHA256
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


def _normalize_generated_body(source):
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    volatile_ids = {}

    def replace_volatile_temp(match):
        kind, identity = match.groups()
        ordinal = volatile_ids.setdefault(identity, f"{len(volatile_ids) + 1:04d}")
        return f"_{kind}_volatile{ordinal}"

    source = _VOLATILE_TEMP.sub(replace_volatile_temp, source)
    source = "\n".join(line.rstrip() for line in source.splitlines())
    return source.rstrip() + "\n"


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
    raw_sha256, normalized_sha256 = FROZEN_GENERATED_BODY_SHA256[variant]
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
    assert spec.sources[0] == tmp_path / uri / "flashkda_decode_binding.cu"
    assert spec.sources[0].is_file()
    assert expected_flag in spec.extra_cuda_cflags
    target_defines = [
        flag
        for flag in spec.extra_cuda_cflags
        if flag.startswith("-DFLASHINFER_FLASH_KDA_DECODE_TARGET_KIND=")
    ]
    assert target_defines == [
        f"-DFLASHINFER_FLASH_KDA_DECODE_TARGET_KIND={target_kind}"
    ]
    assert "-use_fast_math" in spec.extra_cuda_cflags
    assert "--maxrregcount=128" in spec.extra_cuda_cflags
    assert sum("-gencode=arch=compute_" in flag for flag in spec.extra_cuda_cflags) == 1
    assert not any("compute_120" in flag for flag in spec.extra_cuda_cflags)

    frozen_source = flash_kda_decode._get_csrc_dir() / f"flashkda_decode_{variant}.cu"
    frozen_text = frozen_source.read_text()
    assert "Generated from a recurrent-KDA Loom schedule." in frozen_text
    assert f"Raw generated body SHA256: {raw_sha256}" in frozen_text
    assert f"Normalized generated SHA256: {normalized_sha256}" in frozen_text
    # Public sources describe the generator and immutable body, without
    # publishing private GitLab URLs, internal merge-request IDs, or commits.
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
    assert hashlib.sha256(generated_body.encode()).hexdigest() == raw_sha256
    normalized_body = _normalize_generated_body(generated_body)
    assert hashlib.sha256(normalized_body.encode()).hexdigest() == normalized_sha256
    assert f"Raw generated body SHA256: {raw_sha256}" in before_body
    assert f"Normalized generated SHA256: {normalized_sha256}" in before_body
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
        f'#define FLASHKDA_DECODE_BODY_FILE "flashkda_decode_{variant}.cu"'
        in binding_text
    )
    assert f"#define FLASHKDA_DECODE_HEAD_DIM {metadata.head_dim}" in binding_text
    assert f"#define FLASHKDA_DECODE_TOKENS {metadata.tokens}" in binding_text
    assert f"#define FLASHKDA_DECODE_GATE_KIND {expected_gate_kind}" in binding_text
    assert f"#define FLASHKDA_DECODE_VALUE_SPLIT {metadata.value_split}" in binding_text
    assert (
        f"#define FLASHKDA_DECODE_LAUNCH_THREADS {metadata.launch_threads}"
        in binding_text
    )
    assert (
        "#define FLASHKDA_DECODE_DIRECT_IMPL 1" in binding_text
    ) is metadata.direct_impl
    assert '#include "flashkda_decode_binding.cuh"' in binding_text
    flash_kda_decode.gen_flash_kda_decode_module.cache_clear()


def test_flash_kda_decode_binding_contract():
    csrc_dir = flash_kda_decode._get_csrc_dir()
    binding = (csrc_dir / "flashkda_decode_binding.cuh").read_text()
    common = (csrc_dir / "flashkda_decode_binding_common.cuh").read_text()
    impl = (csrc_dir / "flashkda_decode_binding_impl.cuh").read_text()
    direct_impl = (csrc_dir / "flashkda_decode_binding_direct_impl.cuh").read_text()

    assert "FLASHKDA_DECODE_BODY_FILE" in binding
    assert "#include FLASHKDA_DECODE_BODY_FILE" in binding
    assert "#ifdef FLASHKDA_DECODE_DIRECT_IMPL" in binding
    assert '#include "flashkda_decode_binding_direct_impl.cuh"' in binding
    assert "#ifndef FLASHINFER_FLASH_KDA_DECODE_TARGET_KIND" in common
    assert "kFlashKDADecodeTargetKind == kFlashKDADecodeFamilyTarget" in common
    assert "kFlashKDADecodeTargetKind == kFlashKDADecodeExactSM100aTarget" in common
    assert "kFlashKDADecodeTargetKind == kFlashKDADecodeExactSM103aTarget" in common
    assert "major == 10 && (minor == 0 || minor == 3)" in common
    assert "major == 10 && minor == expected_minor" in common
    assert "CheckFlashKDADecodeTarget(device_id)" in common
    assert "struct VariantTraits" in common
    assert "static_assert(Tokens >= 1)" in common
    assert "ValueSplit == 16" in common
    assert "HeadDim % ValueSplit == 0" in common
    assert "state.stride(0) >= num_value_heads * head_dim * head_dim" in common
    assert "gate.stride(1) >= num_value_heads * head_dim" in common
    assert "g must be compact in its [HV, K] trailing dimensions" in common
    assert 'CheckNoOverlap(out, "output", state, "initial_state")' in common
    assert "lower-bound gate variants require at least H A_log values" in common
    assert "lower-bound gate variants require at least H * D dt_bias values" in common
    assert "finite negative lower_bound" in common
    assert "torch.cuda.current_stream" not in impl
    assert "cuda_stream" in impl
    assert "VALUE_SPLIT" in impl
    assert "FLASHKDA_DECODE_EXPECTED_SMEM" not in impl
    assert "SMEM_TOTAL > 0" in impl
    assert "kernel_flashinfer_recurrent_kda_t1_direct" in direct_impl
    assert "SMEM_TOTAL" not in direct_impl
    assert "cuda_stream" in direct_impl


def test_flash_kda_decode_variant_validation_and_getter(monkeypatch):
    expected_variants = (
        "d128_t1_precomputed_split1",
        "d128_t1_precomputed_split2",
        "d128_t1_precomputed_split4",
        "d128_t1_precomputed_split8",
        "d128_t1_precomputed_direct_split16",
        "d128_t1_precomputed_direct_split8",
        "d128_t2_precomputed_split1",
        "d128_t2_precomputed_split2",
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
        "d128_t1_precomputed_split1": 256,
        "d128_t1_precomputed_split2": 128,
        "d128_t1_precomputed_split4": 64,
        "d128_t1_precomputed_split8": 32,
        "d128_t1_precomputed_direct_split16": 32,
        "d128_t1_precomputed_direct_split8": 32,
        "d128_t2_precomputed_split1": 256,
        "d128_t2_precomputed_split2": 128,
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
    with pytest.raises(ValueError, match="only retained for direct T=1"):
        flash_kda_decode.get_flash_kda_decode_uri(expected_variants[0], "sm103a")
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
