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

import pytest
from packaging.version import Version

from flashinfer.jit import core as jit_core
from flashinfer.jit import flash_kda_decode


# Each tuple is (raw frozen-body SHA256, normalized generated-body SHA256).
# The normalization removes only identity-derived temporary suffixes; all
# schedule-relevant source text remains covered by the stable digest.
FROZEN_GENERATED_BODY_SHA256 = {
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
}

_FROZEN_BODY_BEGIN = "// BEGIN FROZEN GENERATED BODY\n"
_FROZEN_BODY_END = "// END FROZEN GENERATED BODY\n"
_VOLATILE_TEMP = re.compile(r"\b_(bval|bits|addr)_([0-9]{6,})\b")


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
    ("variant", "body_hashes"),
    FROZEN_GENERATED_BODY_SHA256.items(),
)
def test_flash_kda_decode_jit_spec_and_frozen_body(monkeypatch, variant, body_hashes):
    raw_sha256, normalized_sha256 = body_hashes
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, "0a")},
    )
    flash_kda_decode.gen_flash_kda_decode_module.cache_clear()

    uri = flash_kda_decode.get_flash_kda_decode_uri(variant)
    spec = flash_kda_decode.gen_flash_kda_decode_module(variant)

    assert uri == f"flash_kda_decode_{variant}_sm100a"
    assert spec.name == uri
    assert len(spec.sources) == 1
    assert spec.sources[0].name == f"flashkda_decode_{variant}_binding.cu"
    assert spec.sources[0].is_file()
    assert "-gencode=arch=compute_100a,code=sm_100a" in spec.extra_cuda_cflags
    assert "-use_fast_math" in spec.extra_cuda_cflags
    assert "--maxrregcount=128" in spec.extra_cuda_cflags
    assert not any(
        "compute_103" in flag or "compute_120" in flag
        for flag in spec.extra_cuda_cflags
    )

    frozen_source = spec.sources[0].parent / f"flashkda_decode_{variant}.cu"
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
    assert "#define GATE_KIND 0" in generated_body
    assert "#define DIRECT_PREFIX_CHECKPOINT 0" in generated_body
    assert "#define BLOCK_CHECKPOINT_MMA 0" in generated_body

    split = int(variant.rpartition("split")[2])
    binding_text = spec.sources[0].read_text()
    assert f"#define FLASHKDA_DECODE_VALUE_SPLIT {split}" in binding_text


def test_flash_kda_decode_binding_contract():
    csrc_dir = flash_kda_decode._get_csrc_dir()
    common = (csrc_dir / "flashkda_decode_binding_common.cuh").read_text()
    impl = (csrc_dir / "flashkda_decode_binding_impl.cuh").read_text()

    assert "CheckExactSm100a" in common
    assert "state.stride(0) >= num_value_heads * head_dim * head_dim" in common
    assert "gate.stride(1) >= num_value_heads * head_dim" in common
    assert "g must be compact in its [HV, K] trailing dimensions" in common
    assert 'CheckNoOverlap(out, "output", state, "initial_state")' in common
    assert "torch.cuda.current_stream" not in impl
    assert "cuda_stream" in impl
    assert "VALUE_SPLIT" in impl


def test_flash_kda_decode_variant_validation_and_getter(monkeypatch):
    expected_variants = tuple(FROZEN_GENERATED_BODY_SHA256)
    assert expected_variants == flash_kda_decode.FLASH_KDA_DECODE_VARIANTS
    for removed_variant in (
        "d128_t4_precomputed",
        "d128_t5_precomputed",
        "d128_t5_precomputed_gram",
        "d128_t5_precomputed_gram_split3",
    ):
        with pytest.raises(ValueError, match="unsupported FlashKDA decode variant"):
            flash_kda_decode.get_flash_kda_decode_uri(removed_variant)

    sentinel = object()
    monkeypatch.setattr(
        flash_kda_decode,
        "load_flash_kda_decode_module",
        lambda variant: (sentinel, variant),
    )
    for variant in FROZEN_GENERATED_BODY_SHA256:
        assert flash_kda_decode.get_flash_kda_decode_module(variant) == (
            sentinel,
            variant,
        )


@pytest.mark.parametrize(
    ("target_archs", "expected_exact"),
    [
        ({(10, "0a")}, True),
        ({(10, "0f")}, False),
        ({(10, "3a")}, False),
        ({(12, "0f")}, False),
    ],
)
def test_aot_detects_only_exact_sm100a(monkeypatch, target_archs, expected_exact):
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
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version("13.0"))
    assert aot.detect_sm_capabilities()["sm100a_exact"] is expected_exact
