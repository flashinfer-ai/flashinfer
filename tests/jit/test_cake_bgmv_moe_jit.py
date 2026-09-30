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
        ((10, 0), "sm100a"),
        ((10, 3), "sm103a"),
        ((9, 0), None),
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
