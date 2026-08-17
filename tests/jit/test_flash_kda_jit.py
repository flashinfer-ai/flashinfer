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
from flashinfer.jit import flash_kda


@pytest.mark.parametrize(
    (
        "variant",
        "smem_bytes",
        "module_ident",
        "generated_tensor_map_acquire",
        "tensor_map_count",
    ),
    [
        (
            "m64",
            221696,
            "9a5566f3be",
            True,
            6,
        ),
        (
            "m128",
            227328,
            "ea022a2f1f",
            True,
            6,
        ),
        (
            "m128_n16",
            219136,
            "ef8b47d690",
            True,
            6,
        ),
        (
            "persistent_m128",
            221696,
            "64bc19d01c",
            True,
            6,
        ),
        (
            "small_bh_m128",
            227328,
            "73369168de",
            True,
            7,
        ),
    ],
)
@pytest.mark.parametrize(
    (
        "target",
        "target_arch",
        "expected_flag",
        "expected_define",
        "forbidden_compute",
    ),
    [
        (
            "sm100a",
            (10, "0a"),
            "-gencode=arch=compute_100a,code=sm_100a",
            "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0",
            ("compute_100f", "compute_103a"),
        ),
        (
            "sm100f",
            (10, "3a"),
            "-gencode=arch=compute_100f,code=sm_100f",
            "-DFLASHINFER_FLASH_KDA_TARGET_FAMILY=100",
            ("compute_100a", "compute_103a"),
        ),
    ],
)
def test_flash_kda_uri_and_jit_spec(
    monkeypatch,
    variant,
    smem_bytes,
    module_ident,
    generated_tensor_map_acquire,
    tensor_map_count,
    target,
    target_arch,
    expected_flag,
    expected_define,
    forbidden_compute,
):
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {target_arch},
    )
    flash_kda.gen_flash_kda_module.cache_clear()

    uri = flash_kda.get_flash_kda_uri(variant, target)
    spec = flash_kda.gen_flash_kda_module(variant, target)

    assert uri == f"flash_kda_bf16_{variant}_{module_ident}_{target}"
    assert spec.name == uri
    assert len(spec.sources) == 1
    assert spec.sources[0].name == (
        f"{flash_kda._FLASH_KDA_BINDING_STEMS[variant]}_binding.cu"
    )
    assert spec.sources[0].is_file()
    assert expected_flag in spec.extra_cuda_cflags
    target_defines = [
        flag
        for flag in spec.extra_cuda_cflags
        if flag.startswith("-DFLASHINFER_FLASH_KDA_TARGET_")
    ]
    assert target_defines == [expected_define]
    assert "-use_fast_math" in spec.extra_cuda_cflags
    assert not any(
        compute in flag
        for compute in forbidden_compute
        for flag in spec.extra_cuda_cflags
    )
    assert not any("compute_120" in flag for flag in spec.extra_cuda_cflags)
    frozen_source = spec.sources[0].parent / (
        f"{flash_kda._FLASH_KDA_BINDING_STEMS[variant]}.cu"
    )
    frozen_text = frozen_source.read_text()
    schedule_symbol = (
        "flashkda_bf16_small_bh_m128"
        if variant == "small_bh_m128"
        else "flashkda_bf16_persistent_m128"
        if variant == "persistent_m128"
        else "flashkda_bf16_fused_m64"
        if variant == "m64"
        else "flashkda_bf16_fused_m128"
    )
    assert f"Generated schedule '{schedule_symbol}'" in frozen_text
    assert f"#define SMEM_TOTAL {smem_bytes}" in frozen_text
    assert frozen_text.count("// clang-format off") == 1
    assert frozen_text.rstrip().endswith("// clang-format on")
    generated_body = frozen_text.partition("// clang-format off\n")[2].rpartition(
        "// clang-format on"
    )[0]
    if generated_tensor_map_acquire:
        assert "FLASHINFER INTEGRATION BEGIN: acquire global tensor maps" not in (
            generated_body
        )
        assert (
            generated_body.count("fence.proxy.tensormap::generic.acquire.sys")
            == tensor_map_count
        )
        assert generated_body.count("__syncthreads();") >= 1
        generated_body_without_tma_integration = generated_body
    else:
        integration_begin = (
            "    // FLASHINFER INTEGRATION BEGIN: acquire global tensor maps\n"
        )
        integration_end = (
            "    // FLASHINFER INTEGRATION END: acquire global tensor maps\n"
        )
        generated_prefix, begin_marker, integration_tail = generated_body.partition(
            integration_begin
        )
        integration_prologue, end_marker, generated_suffix = integration_tail.partition(
            integration_end
        )
        assert begin_marker == integration_begin
        assert end_marker == integration_end
        assert (
            integration_prologue.count("fence.proxy.tensormap::generic.acquire.gpu")
            == 6
        )
        assert integration_prologue.count("], 128;") == 6
        assert integration_prologue.count("__syncthreads();") == 1
        for tensor_map in (
            "q_tma",
            "k_tma",
            "v_tma",
            "g_tma",
            "beta_tma",
            "out_tma",
        ):
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
    if variant == "persistent_m128":
        assert alias_signature.count("__nv_bfloat16* __restrict__ initial_state") == 1
    else:
        assert alias_signature.count("__nv_bfloat16* initial_state") == 1
    assert alias_signature.count("__nv_bfloat16* final_state") == 1
    restricted_alias_signature = alias_signature.replace(
        "__nv_bfloat16* final_state",
        "__nv_bfloat16* __restrict__ final_state",
    )
    if variant != "persistent_m128":
        restricted_alias_signature = restricted_alias_signature.replace(
            "__nv_bfloat16* initial_state",
            "__nv_bfloat16* __restrict__ initial_state",
        )
    if variant == "persistent_m128":
        inplace_store = (
            "/* FLASHINFER INTEGRATION: persistent in-place state */ initial_state +"
        )
        assert alias_suffix.count(inplace_store) == 4
        alias_suffix = alias_suffix.replace(inplace_store, "final_state +")
    normalized_generated_body = alias_prefix + restricted_alias_signature + alias_suffix
    assert "kernel_flashkda_bf16_" in normalized_generated_body
    assert "FlashKDATensorMap" in normalized_generated_body
    assert [
        line for line in frozen_text.splitlines() if line.startswith("#include")
    ] == [
        "#include <cuda_bf16.h>",
        "#include <math_constants.h>",
    ]

    binding_text = spec.sources[0].read_text()
    assert "#define uint64_t flashkda_generated_uint64_t" in binding_text
    assert "TensorView descriptor_storage" in binding_text
    assert "int64_t prepare_descriptors" in binding_text
    assert "CheckFlashKDATarget(device_id)" in binding_text
    chunk_tokens = 16 if variant == "m128_n16" else 32
    value_rows = 64 if variant == "m64" else 128
    assert f"EncodeTmaPointers<{value_rows}, {chunk_tokens}>" in binding_text
    if variant in {"m128", "m128_n16"}:
        assert "TensorView state_indices" in binding_text
        assert "TensorView state_checkpoints" in binding_text
        assert "int64_t beta_token_stride" in binding_text
        assert "int64_t checkpoint_every_n_tokens" in binding_text
        assert "reinterpret_cast<uintptr_t>(state_indices.data_ptr())" in binding_text
        assert (
            "reinterpret_cast<uintptr_t>(state_checkpoints.data_ptr())" in binding_text
        )
        assert (
            "reinterpret_cast<uintptr_t>(checkpoint_cu_starts.data_ptr())"
            in binding_text
        )
    if variant == "persistent_m128":
        assert "TensorView task_ids" in binding_text
        assert "TensorView task_offsets" in binding_text
        assert "one caller-owned in-place state tensor" in binding_text
        assert "sm_count == 148 || sm_count == 152" in binding_text
        assert "CheckFlashKDAPersistentDevice(device_id)" in binding_text
    if variant == "small_bh_m128":
        assert "TensorView packet_workspace" in binding_text
        assert "TensorView packet_ready" in binding_text
        assert "TensorView packet_consumed" in binding_text
        assert "TensorView helper_done" in binding_text
        assert "kSmallBHGroupSize * total_tasks" in binding_text
        assert "EncodeSmallBHPacketTma(packet_workspace)" in binding_text
    assert (
        "#define FlashKDATensorMap flashkda_generated_FlashKDATensorMap" in binding_text
    )
    assert (
        "reinterpret_cast<flashkda_generated_FlashKDATensorMap const*>" in binding_text
    )


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
    assert "defined(FLASHINFER_FLASH_KDA_TARGET_MINOR)" in common_text
    assert "defined(FLASHINFER_FLASH_KDA_TARGET_FAMILY)" in common_text
    assert "kFlashKDATargetMinor == 0" in common_text
    assert "kFlashKDATargetFamily == 100" in common_text
    assert "minor == 0 || minor == 3" in common_text
    assert "major == 10 && minor == kFlashKDATargetMinor" in common_text
    assert "CheckFlashKDATarget" in common_text
    assert "PackBetaForTmaKernel" in common_text
    assert "RoundUpBetaTmaHeads(num_heads)" in common_text
    assert "padded_num_heads == num_heads" in common_text
    assert "linear_index / padded_num_heads" in common_text
    assert "linear_index % padded_num_heads" in common_text
    assert (
        'CheckNoPartialOverlapOrExactAlias(beta, "beta", beta_tma, "beta_tma")'
        in common_text
    )
    assert (
        'CheckNoOverlap(initial_state, "initial_state", beta_tma, "beta_tma")'
        in common_text
    )

    m128_binding = (
        flash_kda._get_flash_kda_csrc_dir() / "flashkda_bf16_fused_m128_binding.cu"
    ).read_text()
    assert (
        "PackBetaForTmaIfNeeded(beta, beta_tma, num_heads, beta_token_stride, stream);"
        in m128_binding
    )
    assert "EncodeTmaPointers<128, 32>" in m128_binding

    m128_n16_binding = (
        flash_kda._get_flash_kda_csrc_dir()
        / "cake_flashkda_bf16_fused_m128_n16_binding.cu"
    ).read_text()
    assert (
        "PackBetaForTmaIfNeeded(beta, beta_tma, num_heads, beta_token_stride, stream);"
        in m128_n16_binding
    )
    assert "EncodeTmaPointers<128, 16>" in m128_n16_binding


def test_flash_kda_variant_validation_and_public_getter(monkeypatch):
    with pytest.raises(ValueError, match="unsupported FlashKDA variant"):
        flash_kda.get_flash_kda_uri("m32", "sm100a")
    with pytest.raises(ValueError, match="unsupported FlashKDA target"):
        flash_kda.get_flash_kda_uri("m128", "sm120a")
    with pytest.raises(ValueError, match="unsupported FlashKDA target"):
        flash_kda.get_flash_kda_uri("persistent_m128", "sm103a")

    sentinel = object()
    monkeypatch.setattr(
        flash_kda,
        "load_flash_kda_module",
        lambda variant, target: (sentinel, variant, target),
    )
    assert flash_kda.get_flash_kda_prefill_module("m128_n16", "sm100f") == (
        sentinel,
        "m128_n16",
        "sm100f",
    )


def test_flash_kda_legacy_and_family_targets_have_independent_cache_keys(monkeypatch):
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, "0a"), (10, "3a")},
    )
    flash_kda.gen_flash_kda_module.cache_clear()

    sm100a = flash_kda.gen_flash_kda_module("m128", "sm100a")
    sm100f = flash_kda.gen_flash_kda_module("m128", "sm100f")
    sm100f_cached = flash_kda.gen_flash_kda_module("m128", "sm100f")
    n16_sm100a = flash_kda.gen_flash_kda_module("m128_n16", "sm100a")
    n16_sm100f = flash_kda.gen_flash_kda_module("m128_n16", "sm100f")
    small_sm100a = flash_kda.gen_flash_kda_module("small_bh_m128", "sm100a")
    small_sm100f = flash_kda.gen_flash_kda_module("small_bh_m128", "sm100f")

    assert sm100a is not sm100f
    assert sm100a.name == "flash_kda_bf16_m128_ea022a2f1f_sm100a"
    assert sm100f.name == "flash_kda_bf16_m128_ea022a2f1f_sm100f"
    assert sm100f is sm100f_cached
    assert n16_sm100a is not n16_sm100f
    assert n16_sm100a.name == "flash_kda_bf16_m128_n16_ef8b47d690_sm100a"
    assert n16_sm100f.name == "flash_kda_bf16_m128_n16_ef8b47d690_sm100f"
    assert small_sm100a is not small_sm100f
    assert small_sm100a.name == "flash_kda_bf16_small_bh_m128_73369168de_sm100a"
    assert small_sm100f.name == "flash_kda_bf16_small_bh_m128_73369168de_sm100f"


@pytest.mark.parametrize(
    (
        "target_archs",
        "cuda_version",
        "expected_sm100a",
        "expected_sm100f",
    ),
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
def test_aot_detects_flash_kda_target_matrix(
    monkeypatch, target_archs, cuda_version, expected_sm100a, expected_sm100f
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
    assert capabilities["flash_kda_prefill_sm100a"] is expected_sm100a
    assert capabilities["flash_kda_prefill_sm100f"] is expected_sm100f


@pytest.mark.parametrize(
    ("capabilities", "expected_targets"),
    [
        ({"flash_kda_prefill_sm100a": True}, ("sm100a",)),
        ({"flash_kda_prefill_sm100f": True}, ("sm100f",)),
        (
            {
                "flash_kda_prefill_sm100a": True,
                "flash_kda_prefill_sm100f": True,
            },
            ("sm100a", "sm100f"),
        ),
    ],
)
def test_aot_registers_complete_flash_kda_modules(
    monkeypatch, capabilities, expected_targets
):
    from flashinfer import aot

    calls = []

    def fake_flash_kda(variant, target):
        calls.append((variant, target))
        return SimpleNamespace(name=f"flash_kda_{variant}_{target}")

    monkeypatch.setattr(
        aot,
        "gen_flash_kda_m64_module",
        lambda target: fake_flash_kda("m64", target),
    )
    monkeypatch.setattr(
        aot,
        "gen_flash_kda_m128_module",
        lambda target: fake_flash_kda("m128", target),
    )
    monkeypatch.setattr(
        aot,
        "gen_flash_kda_m128_n16_module",
        lambda target: fake_flash_kda("m128_n16", target),
    )
    monkeypatch.setattr(
        aot,
        "gen_flash_kda_persistent_m128_module",
        lambda target: fake_flash_kda("persistent_m128", target),
    )
    monkeypatch.setattr(
        aot,
        "gen_flash_kda_small_bh_m128_module",
        lambda target: fake_flash_kda("small_bh_m128", target),
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

    expected_calls = []
    for target in expected_targets:
        expected_calls.extend(
            (variant, target)
            for variant in ("m64", "m128", "m128_n16", "small_bh_m128")
        )
        expected_calls.append(("persistent_m128", target))
    assert calls == expected_calls
    assert [spec.name for spec in specs] == [
        "spdlog",
        *[f"flash_kda_{variant}_{target}" for variant, target in expected_calls],
        "cudnn",
    ]
