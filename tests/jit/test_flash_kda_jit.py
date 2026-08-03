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

import pytest
from packaging.version import Version

from flashinfer.jit import core as jit_core
from flashinfer.jit import flash_kda


@pytest.mark.parametrize(
    ("variant", "smem_bytes", "generated_sha256"),
    [
        (
            "m64",
            219136,
            "c28aacd475983c72ffe84acac7321a0b2e1c495d7c6e9cdc4a80ada112d76515",
        ),
        (
            "m128",
            227328,
            "e6ea814f0f2e0e0cb33c1562458de9e47272760562dbcb2364855c5b48f0b6ce",
        ),
    ],
)
def test_flash_kda_uri_and_jit_spec(monkeypatch, variant, smem_bytes, generated_sha256):
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, "0a")},
    )
    flash_kda.gen_flash_kda_module.cache_clear()

    uri = flash_kda.get_flash_kda_uri(variant)
    spec = flash_kda.gen_flash_kda_module(variant)

    assert uri == f"flash_kda_bf16_fused_{variant}_sm100a"
    assert spec.name == uri
    assert len(spec.sources) == 1
    assert spec.sources[0].name == f"flashkda_bf16_fused_{variant}_binding.cu"
    assert spec.sources[0].is_file()
    assert "-gencode=arch=compute_100a,code=sm_100a" in spec.extra_cuda_cflags
    assert "-use_fast_math" in spec.extra_cuda_cflags
    assert not any(
        "compute_103" in flag or "compute_120" in flag
        for flag in spec.extra_cuda_cflags
    )
    frozen_source = spec.sources[0].parent / f"flashkda_bf16_fused_{variant}.cu"
    frozen_text = frozen_source.read_text()
    assert f"Provenance: generated Loom schedule 'flashkda_bf16_fused_{variant}'" in (
        frozen_text
    )
    assert f"#define SMEM_TOTAL {smem_bytes}" in frozen_text
    assert frozen_text.count("// clang-format off") == 1
    assert frozen_text.rstrip().endswith("// clang-format on")
    generated_body = frozen_text.partition("// clang-format off\n")[2].rpartition(
        "// clang-format on"
    )[0]
    integration_begin = (
        "    // FLASHINFER INTEGRATION BEGIN: acquire global tensor maps\n"
    )
    integration_end = "    // FLASHINFER INTEGRATION END: acquire global tensor maps\n"
    generated_prefix, begin_marker, integration_tail = generated_body.partition(
        integration_begin
    )
    integration_prologue, end_marker, generated_suffix = integration_tail.partition(
        integration_end
    )
    assert begin_marker == integration_begin
    assert end_marker == integration_end
    assert integration_prologue.count("fence.proxy.tensormap::generic.acquire.gpu") == 6
    assert integration_prologue.count("], 128;") == 6
    assert integration_prologue.count("__syncthreads();") == 1
    for tensor_map in ("q_tma", "k_tma", "v_tma", "g_tma", "beta_tma", "out_tma"):
        assert f'"l"({tensor_map})' in integration_prologue
    generated_body_without_tma_integration = generated_prefix + generated_suffix
    alias_begin = "// FLASHINFER INTEGRATION BEGIN: allow exact state alias\n"
    alias_end = "// FLASHINFER INTEGRATION END: allow exact state alias\n"
    alias_prefix, begin_marker, alias_tail = (
        generated_body_without_tma_integration.partition(alias_begin)
    )
    alias_signature, end_marker, alias_suffix = alias_tail.partition(alias_end)
    assert begin_marker == alias_begin
    assert end_marker == alias_end
    assert alias_signature.count("__nv_bfloat16* initial_state") == 1
    assert alias_signature.count("__nv_bfloat16* final_state") == 1
    restricted_alias_signature = alias_signature.replace(
        "__nv_bfloat16* initial_state",
        "__nv_bfloat16* __restrict__ initial_state",
    ).replace(
        "__nv_bfloat16* final_state",
        "__nv_bfloat16* __restrict__ final_state",
    )
    # Keep the exporter output immutable outside the two narrowly marked
    # FlashInfer integration patches.
    normalized_generated_body = alias_prefix + restricted_alias_signature + alias_suffix
    assert (
        hashlib.sha256(normalized_generated_body.encode()).hexdigest()
        == generated_sha256
    )
    assert [
        line for line in frozen_text.splitlines() if line.startswith("#include")
    ] == [
        "#include <cuda_bf16.h>",
        "#include <math_constants.h>",
    ]

    binding_text = spec.sources[0].read_text()
    assert "#define uint64_t flashkda_generated_uint64_t" in binding_text
    assert "TensorView descriptor_storage, int64_t prepare_descriptors" in binding_text


def test_flash_kda_descriptor_workspace_contract():
    common_source = flash_kda._get_flash_kda_csrc_dir() / (
        "flashkda_binding_common.cuh"
    )
    common_text = common_source.read_text()

    assert "static_assert(sizeof(CUtensorMap) == 128);" in common_text
    assert "kTensorMapAlignment = 64" in common_text
    assert 'CheckDtype(descriptor_storage, "descriptor_storage", dl_uint8)' in (
        common_text
    )
    assert "PublishTensorMaps<<<1, 128, 0, stream>>>" in common_text
    assert "prepare_descriptors must be 0 during CUDA graph capture" in common_text
    assert "cudaMemcpyAsync(TMA descriptors)" not in common_text


def test_flash_kda_variant_validation_and_public_getter(monkeypatch):
    with pytest.raises(ValueError, match="unsupported FlashKDA variant"):
        flash_kda.get_flash_kda_uri("m32")

    sentinel = object()
    monkeypatch.setattr(
        flash_kda,
        "load_flash_kda_module",
        lambda variant: (sentinel, variant),
    )
    assert flash_kda.get_flash_kda_prefill_module("m128") == (
        sentinel,
        "m128",
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

    capabilities = aot.detect_sm_capabilities()
    assert capabilities["sm100a_exact"] is expected_exact
