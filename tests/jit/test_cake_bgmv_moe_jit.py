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

import re

import pytest

from flashinfer.jit import cake_bgmv_moe
from flashinfer.jit import core as jit_core


@pytest.mark.parametrize("arch", cake_bgmv_moe.CAKE_BGMV_MOE_ARCHES)
@pytest.mark.parametrize(
    ("hidden_size", "num_tokens", "expected"),
    [
        (3072, 1, "token_owned_t64"),
        (3072, 4, "token_owned_t64"),
        (3072, 8, "token_owned_t64"),
        (3072, 32, "token_owned"),
        (3072, 256, "token_owned"),
        (3072, 512, "token_owned_dual_col"),
        (3072, 1024, "token_owned_dual_col"),
        (2688, 1, "token_owned_t64"),
        (2688, 4, "token_owned_t64"),
        (2688, 8, "token_owned_t64"),
        (2688, 32, "token_owned"),
        (2688, 256, "token_owned"),
        (2688, 512, "token_owned"),
        (2688, 1024, "token_owned_dual_col"),
    ],
)
def test_selector_matches_measured_shape_portfolio(
    arch, hidden_size, num_tokens, expected
):
    assert (
        cake_bgmv_moe.select_cake_bgmv_moe_schedule(hidden_size, num_tokens, arch)
        == expected
    )


def test_selector_rejects_unsupported_shapes():
    with pytest.raises(ValueError, match="hidden_size"):
        cake_bgmv_moe.select_cake_bgmv_moe_schedule(2048, 32)
    with pytest.raises(ValueError, match="positive"):
        cake_bgmv_moe.select_cake_bgmv_moe_schedule(3072, 0)
    with pytest.raises(ValueError, match="arch"):
        cake_bgmv_moe.select_cake_bgmv_moe_schedule(3072, 8, "sm120a")


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((9, 0), "sm90a"),
        ((10, 0), "sm100a"),
        ((10, 3), "sm103a"),
        ((8, 0), None),
        ((10, 1), None),
        ((12, 0), None),
        ((12, 1), None),
    ],
)
def test_arch_for_capability(capability, expected):
    assert cake_bgmv_moe.cake_bgmv_moe_arch_for_capability(capability) == expected


@pytest.mark.parametrize(
    ("arch", "cuda_arch", "gencode", "cc"),
    [
        ("sm90a", (9, "0a"), "-gencode=arch=compute_90a,code=sm_90a", (9, 0)),
        ("sm100a", (10, "0a"), "-gencode=arch=compute_100a,code=sm_100a", (10, 0)),
        ("sm103a", (10, "3a"), "-gencode=arch=compute_103a,code=sm_103a", (10, 3)),
    ],
)
@pytest.mark.parametrize("hidden_size", cake_bgmv_moe.CAKE_BGMV_MOE_HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", cake_bgmv_moe.CAKE_BGMV_MOE_DTYPES)
def test_jit_spec_binds_generated_source_per_arch(
    monkeypatch, tmp_path, arch, cuda_arch, gencode, cc, hidden_size, dtype
):
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {cuda_arch},
    )
    monkeypatch.setattr(cake_bgmv_moe.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)
    cake_bgmv_moe.gen_cake_bgmv_moe_module.cache_clear()

    spec = cake_bgmv_moe.gen_cake_bgmv_moe_module(hidden_size, dtype, arch)
    uri = cake_bgmv_moe.get_cake_bgmv_moe_uri(hidden_size, dtype, arch)
    metadata = cake_bgmv_moe._metadata(hidden_size, dtype)

    assert spec.name == uri
    assert uri.endswith(f"_{arch}")
    assert spec.sources == [tmp_path / uri / "cake_bgmv_moe_binding.cu"]
    assert gencode in spec.extra_cuda_cflags
    assert "-use_fast_math" in spec.extra_cuda_cflags
    assert cake_bgmv_moe._get_csrc_dir().parent in spec.extra_include_dirs
    assert cake_bgmv_moe._get_include_dir() in spec.extra_include_dirs
    body = (cake_bgmv_moe._get_csrc_dir() / metadata.body).read_text()
    for symbol in metadata[1:]:
        assert symbol in body
    binding = spec.sources[0].read_text()
    assert f'#define CAKE_BGMV_MOE_BODY_FILE "{metadata.body}"' in binding
    assert f"#define CAKE_BGMV_MOE_HIDDEN {hidden_size}" in binding
    assert f"#define CAKE_BGMV_MOE_CC_MAJOR {cc[0]}" in binding
    assert f"#define CAKE_BGMV_MOE_CC_MINOR {cc[1]}" in binding
    assert '#include "cake_bgmv_moe_binding.cuh"' in binding
    cake_bgmv_moe.gen_cake_bgmv_moe_module.cache_clear()


def test_arch_modules_do_not_share_a_uri():
    uris = {
        cake_bgmv_moe.get_cake_bgmv_moe_uri(hidden_size, dtype, arch)
        for arch in cake_bgmv_moe.CAKE_BGMV_MOE_ARCHES
        for hidden_size in cake_bgmv_moe.CAKE_BGMV_MOE_HIDDEN_SIZES
        for dtype in cake_bgmv_moe.CAKE_BGMV_MOE_DTYPES
    }
    assert len(uris) == (
        len(cake_bgmv_moe.CAKE_BGMV_MOE_ARCHES)
        * len(cake_bgmv_moe.CAKE_BGMV_MOE_HIDDEN_SIZES)
        * len(cake_bgmv_moe.CAKE_BGMV_MOE_DTYPES)
    )


def test_binding_preserves_graph_and_tensor_contracts():
    binding = (cake_bgmv_moe._get_csrc_dir() / "cake_bgmv_moe_binding.cuh").read_text()
    assert "CheckCompiledArch" in binding
    assert "CheckExactSM100" not in binding
    assert (
        "major == CAKE_BGMV_MOE_CC_MAJOR && minor == CAKE_BGMV_MOE_CC_MINOR" in binding
    )
    assert "kShrinkDecodeSmemBytes = 221696" in binding
    assert "kShrinkPrefillSmemBytes = 36992" in binding
    assert "cudaDevAttrMaxSharedMemoryPerBlockOptin" in binding
    assert "cudaFuncAttributeMaxDynamicSharedMemorySize" in binding
    assert "cudaMemsetAsync" not in binding
    assert "EXPAND_PAIR" not in binding
    assert "CAKE_BGMV_MOE_SHRINK_DECODE<<<" in binding
    assert "CAKE_BGMV_MOE_EXPAND_TOKEN_DUAL<<<" in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(configure" in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run" in binding

    for hidden_size in cake_bgmv_moe.CAKE_BGMV_MOE_HIDDEN_SIZES:
        for dtype in cake_bgmv_moe.CAKE_BGMV_MOE_DTYPES:
            body = (
                cake_bgmv_moe._get_csrc_dir()
                / cake_bgmv_moe._metadata(hidden_size, dtype).body
            ).read_text()
            smem_totals = re.findall(r"#define SMEM_TOTAL (\d+)", body)
            assert smem_totals[:2] == ["221696", "36992"]
            assert "atomicAdd(" not in body
            assert "expand_pair_owned" not in body


# ---- generic-shape bundles -------------------------------------------------


@pytest.mark.parametrize(
    ("hidden_size", "rank", "expected"),
    [
        (2688, 32, "specialized"),
        (3072, 32, "specialized"),
        (3072, 16, "generic"),
        (2688, 64, "generic"),
        (2048, 32, "generic"),
        (736, 8, "generic"),
        (8, 8, "generic"),
        (2052, 32, None),
        (0, 32, None),
        (3072, 12, None),
        (3072, 128, None),
    ],
)
def test_variant_routing(hidden_size, rank, expected):
    assert cake_bgmv_moe.cake_bgmv_moe_variant(hidden_size, rank) == expected


@pytest.mark.parametrize("arch", cake_bgmv_moe.CAKE_BGMV_MOE_ARCHES)
@pytest.mark.parametrize(
    ("hidden_size", "num_tokens", "expected"),
    [
        (2048, 1, "token_owned_t64"),
        (2048, 8, "token_owned_t64"),
        (2048, 9, "token_owned_t128"),
        (736, 512, "token_owned_t128"),
    ],
)
def test_generic_selector(arch, hidden_size, num_tokens, expected):
    assert (
        cake_bgmv_moe.select_cake_bgmv_moe_generic_schedule(
            hidden_size, num_tokens, arch
        )
        == expected
    )


def test_generic_selector_rejects_unsupported_inputs():
    with pytest.raises(ValueError, match="multiple of 8"):
        cake_bgmv_moe.select_cake_bgmv_moe_generic_schedule(2052, 8)
    with pytest.raises(ValueError, match="positive"):
        cake_bgmv_moe.select_cake_bgmv_moe_generic_schedule(2048, 0)
    with pytest.raises(ValueError, match="arch"):
        cake_bgmv_moe.select_cake_bgmv_moe_generic_schedule(2048, 8, "sm120a")
    with pytest.raises(ValueError, match="rank"):
        cake_bgmv_moe.get_cake_bgmv_moe_generic_uri(12, "bfloat16")


@pytest.mark.parametrize(
    ("arch", "cuda_arch", "gencode", "cc"),
    [
        ("sm90a", (9, "0a"), "-gencode=arch=compute_90a,code=sm_90a", (9, 0)),
        ("sm100a", (10, "0a"), "-gencode=arch=compute_100a,code=sm_100a", (10, 0)),
        ("sm103a", (10, "3a"), "-gencode=arch=compute_103a,code=sm_103a", (10, 3)),
    ],
)
@pytest.mark.parametrize("rank", cake_bgmv_moe.CAKE_BGMV_MOE_GENERIC_RANKS)
@pytest.mark.parametrize("dtype", cake_bgmv_moe.CAKE_BGMV_MOE_DTYPES)
def test_generic_jit_spec_binds_generated_source_per_arch(
    monkeypatch, tmp_path, arch, cuda_arch, gencode, cc, rank, dtype
):
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {cuda_arch},
    )
    monkeypatch.setattr(cake_bgmv_moe.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)
    cake_bgmv_moe.gen_cake_bgmv_moe_generic_module.cache_clear()

    spec = cake_bgmv_moe.gen_cake_bgmv_moe_generic_module(rank, dtype, arch)
    uri = cake_bgmv_moe.get_cake_bgmv_moe_generic_uri(rank, dtype, arch)
    metadata = cake_bgmv_moe._generic_metadata(rank, dtype)

    assert spec.name == uri
    assert (
        uri == f"cake_bgmv_moe_generic_{cake_bgmv_moe._dtype_tag(dtype)}_r{rank}_{arch}"
    )
    assert spec.sources == [tmp_path / uri / "cake_bgmv_moe_generic_binding.cu"]
    assert gencode in spec.extra_cuda_cflags
    assert "-use_fast_math" in spec.extra_cuda_cflags
    body = (cake_bgmv_moe._get_csrc_dir() / metadata.body).read_text()
    for symbol in metadata[1:]:
        assert symbol in body
    for macro in (
        "CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_DECODE 221696",
        "CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL 36992",
        "CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T64 ",
        "CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T128 ",
    ):
        assert macro in body
    assert "atomicAdd(" not in body
    binding = spec.sources[0].read_text()
    assert f'#define CAKE_BGMV_MOE_BODY_FILE "{metadata.body}"' in binding
    assert f"#define CAKE_BGMV_MOE_RANK {rank}" in binding
    assert f"#define CAKE_BGMV_MOE_CC_MAJOR {cc[0]}" in binding
    assert f"#define CAKE_BGMV_MOE_CC_MINOR {cc[1]}" in binding
    assert '#include "cake_bgmv_moe_generic_binding.cuh"' in binding
    cake_bgmv_moe.gen_cake_bgmv_moe_generic_module.cache_clear()


def test_generic_and_specialized_modules_do_not_share_a_uri():
    specialized = {
        cake_bgmv_moe.get_cake_bgmv_moe_uri(hidden_size, dtype, arch)
        for arch in cake_bgmv_moe.CAKE_BGMV_MOE_ARCHES
        for hidden_size in cake_bgmv_moe.CAKE_BGMV_MOE_HIDDEN_SIZES
        for dtype in cake_bgmv_moe.CAKE_BGMV_MOE_DTYPES
    }
    generic = {
        cake_bgmv_moe.get_cake_bgmv_moe_generic_uri(rank, dtype, arch)
        for arch in cake_bgmv_moe.CAKE_BGMV_MOE_ARCHES
        for rank in cake_bgmv_moe.CAKE_BGMV_MOE_GENERIC_RANKS
        for dtype in cake_bgmv_moe.CAKE_BGMV_MOE_DTYPES
    }
    assert len(generic) == (
        len(cake_bgmv_moe.CAKE_BGMV_MOE_ARCHES)
        * len(cake_bgmv_moe.CAKE_BGMV_MOE_GENERIC_RANKS)
        * len(cake_bgmv_moe.CAKE_BGMV_MOE_DTYPES)
    )
    assert not (specialized & generic)


def test_generic_binding_preserves_graph_and_tensor_contracts():
    binding = (
        cake_bgmv_moe._get_csrc_dir() / "cake_bgmv_moe_generic_binding.cuh"
    ).read_text()
    assert "CheckCompiledArch" in binding
    assert (
        "major == CAKE_BGMV_MOE_CC_MAJOR && minor == CAKE_BGMV_MOE_CC_MINOR" in binding
    )
    assert (
        "kShrinkDecodeSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_DECODE" in binding
    )
    assert "cudaDevAttrMaxSharedMemoryPerBlockOptin" in binding
    assert "cudaFuncAttributeMaxDynamicSharedMemorySize" in binding
    assert "cudaMemsetAsync" not in binding
    assert "x.size(1) % kVec == 0" in binding
    assert "CAKE_BGMV_MOE_SHRINK_DECODE<<<" in binding
    assert "CAKE_BGMV_MOE_SHRINK_PREFILL<<<" in binding
    assert "CAKE_BGMV_MOE_EXPAND_T64<<<" in binding
    assert "CAKE_BGMV_MOE_EXPAND_T128<<<" in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(configure" in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run" in binding
