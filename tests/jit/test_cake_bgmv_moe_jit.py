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
    assert "TensorView route_index" in binding
    assert "kExpandT64SmemBytes = CAKE_BGMV_MOE_SMEM_EXPAND_T64" in binding
    assert "kExpandDualSmemBytes = CAKE_BGMV_MOE_SMEM_EXPAND_DUAL" in binding
    assert "CheckCompiledArch" in binding
    assert "CheckExactSM100" not in binding
    assert (
        "major == CAKE_BGMV_MOE_CC_MAJOR && minor == CAKE_BGMV_MOE_CC_MINOR" in binding
    )
    assert "kShrinkDecodeSmemBytes = CAKE_BGMV_MOE_SMEM_SHRINK_DECODE" in binding
    assert "kShrinkPrefillSmemBytes = CAKE_BGMV_MOE_SMEM_SHRINK_PREFILL" in binding
    assert "static_assert(kShrinkDecodeSmemBytes == 221696" in binding
    assert "static_assert(kShrinkPrefillSmemBytes == 36992" in binding
    assert "cudaDevAttrMaxSharedMemoryPerBlockOptin" in binding
    assert "cudaFuncAttributeMaxDynamicSharedMemorySize" in binding
    assert "cudaMemsetAsync" not in binding
    assert "EXPAND_PAIR" not in binding
    assert "CAKE_BGMV_MOE_SHRINK_DECODE<<<" in binding
    assert "cudaLaunchKernelEx(&config, CAKE_BGMV_MOE_EXPAND_TOKEN_DUAL," in binding
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
            # The shrink kernels publish the token->pair route index; the expand
            # kernels keep one owner per output (no output atomics).
            assert body.count("atomicAdd(") == 2
            assert "atomicAdd(&reinterpret_cast<float" not in body
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


@pytest.mark.parametrize(
    ("hidden_size", "num_tokens", "expected"),
    [
        (3072, 1, "generic"),
        (3072, 16, "generic"),
        (3072, 31, "generic"),
        (3072, 32, "specialized"),
        (3072, 64, "specialized"),
        (3072, 128, "specialized"),
        (2688, 512, "specialized"),
        (3072, 1024, "specialized"),
        (2688, 1025, "generic"),
        (3072, 2048, "generic"),
        (3072, 4096, "generic"),
        (2048, 4096, "generic"),
    ],
)
def test_variant_routing_by_token_count(hidden_size, num_tokens, expected):
    assert cake_bgmv_moe.CAKE_BGMV_MOE_SPECIALIZED_TOKEN_WINDOW == {
        "sm90a": None,
        "sm100a": (32, 1024),
        "sm103a": (32, 1024),
    }
    for arch in ("sm100a", "sm103a"):
        assert (
            cake_bgmv_moe.cake_bgmv_moe_variant(hidden_size, 32, num_tokens, arch)
            == expected
        )
    # Hopper: the generic bundle wins or ties at every token count.
    assert (
        cake_bgmv_moe.cake_bgmv_moe_variant(hidden_size, 32, num_tokens, "sm90a")
        == "generic"
    )
    # Support queries (no token count) keep the specialized answer.
    assert cake_bgmv_moe.cake_bgmv_moe_variant(3072, 32) == "specialized"


def test_pdl_mode_policy():
    pdl = cake_bgmv_moe.cake_bgmv_moe_pdl_mode
    # 16 tokens x 23 column CTAs = 368 CTAs: small on every part.
    assert pdl("sm90a", 16, 2944, 132) == 1
    assert pdl("sm100a", 16, 2944, 148) == 1
    assert pdl("sm103a", 16, 2944, 148) == 1
    # 128 tokens x 12 column CTAs = 1536 CTAs: still small on Hopper (12/SM),
    # large on Blackwell (8/SM).
    assert pdl("sm90a", 128, 1472, 132) == 1
    assert pdl("sm100a", 128, 1472, 148) == 2
    # 512 tokens x 6 column CTAs = 3072 CTAs: large.
    assert pdl("sm90a", 512, 736, 132) == 0
    assert pdl("sm100a", 512, 736, 148) == 2
    assert pdl("sm103a", 512, 736, 148) == 2
    # The specialized bodies take plain launches at every grid size.
    assert pdl("sm100a", 512, 3072, 148, "specialized") == 0
    assert pdl("sm103a", 1024, 2688, 148, "specialized") == 0
    assert pdl("sm100a", 32, 3072, 148, "specialized") == 0
    assert pdl("sm90a", 32, 3072, 132, "specialized") == 0


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
        "CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_DECODE 221824",
        "CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL 37120",
        "CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_DECODE_PDL 221824",
        "CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL_PDL 37120",
        "CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL_S3 ",
        "CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL_S3_PDL ",
        "CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL_REMAP 37120",
        "CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL_REMAP_PDL 37120",
        "CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T64 ",
        "CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T128 ",
        "CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T64_PF ",
        "CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T128_PF ",
        "CAKE_BGMV_MOE_GENERIC_SMEM_GROUP_HIST ",
        "CAKE_BGMV_MOE_GENERIC_SMEM_GROUP_SCAN ",
        "CAKE_BGMV_MOE_GENERIC_SMEM_GROUP_SCATTER ",
        "CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED ",
        "CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_GROUPED ",
        "CAKE_BGMV_MOE_GENERIC_SMEM_COMBINE_GROUPED ",
        "CAKE_BGMV_MOE_GENERIC_SMEM_ORDER_BUILD ",
    ):
        assert macro in body
    # Route-index publication in the eight shrink kernels plus the three grouping
    # counters (shared-memory histogram and fill counters, per-token route
    # counters) in group_hist / group_scatter and the two block-local bin
    # counters (histogram, cursor) of the order_build prologue; the expands
    # keep one owner per output (no output atomics).
    assert body.count("atomicAdd(") == 13
    assert "atomicAdd(&reinterpret_cast<float" not in body
    binding = spec.sources[0].read_text()
    assert f'#define CAKE_BGMV_MOE_BODY_FILE "{metadata.body}"' in binding
    assert f"#define CAKE_BGMV_MOE_RANK {rank}" in binding
    assert f"#define CAKE_BGMV_MOE_CC_MAJOR {cc[0]}" in binding
    assert f"#define CAKE_BGMV_MOE_CC_MINOR {cc[1]}" in binding
    # Lever 3c: sm90a runs the operand-ring grouped shrink on single-K-tile rows too;
    # Blackwell keeps the register-direct form there.
    single_tile_ring = 1 if arch == "sm90a" else 0
    assert (
        f"#define CAKE_BGMV_MOE_GROUP_SHRINK_RING_SINGLE_TILE {single_tile_ring}"
        in binding
    )
    # Lever 34: the bf16 Blackwell bundles run the mixed-precision ring grouped shrink.
    mixed = (
        1
        if arch in ("sm100a", "sm103a") and cake_bgmv_moe._dtype_tag(dtype) == "bf16"
        else 0
    )
    assert f"#define CAKE_BGMV_MOE_GROUP_SHRINK_MIXED {mixed}" in binding
    assert (
        f"#define CAKE_BGMV_MOE_SHRINK_GROUPED_RING_MIXED {metadata.shrink_grouped_ring_mixed_symbol}"
        in binding
    )
    if cake_bgmv_moe._dtype_tag(dtype) == "bf16":
        assert (
            "_shrink_grouped_ring_mixed_" in metadata.shrink_grouped_ring_mixed_symbol
        )
    else:
        assert (
            metadata.shrink_grouped_ring_mixed_symbol
            == metadata.shrink_grouped_ring_symbol
        )
        assert (
            metadata.shrink_grouped_ring_mixed_single_symbol
            == metadata.shrink_grouped_ring_single_symbol
        )
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
    assert "CAKE_BGMV_MOE_SHRINK_DECODE_PDL<<<" in binding
    assert "CAKE_BGMV_MOE_SHRINK_PREFILL_PDL<<<" in binding
    # The expand is a programmatic dependent launch of the shrink (PDL): the
    # expand grid may start while the shrink drains and waits in-kernel.
    assert "cudaLaunchKernelEx(&config, expand_kernel_t64," in binding
    assert "cudaLaunchKernelEx(&config, expand_kernel_t128," in binding
    # PDL launches select the register-prefetch expand forms; plain launches the
    # lower-register interleaved forms (both are rendered into every bundle).
    assert "const bool prefetch_form = pdl_mode != 0;" in binding
    assert (
        "prefetch_form ? CAKE_BGMV_MOE_EXPAND_T64_PF : CAKE_BGMV_MOE_EXPAND_T64"
        in binding
    )
    assert (
        "prefetch_form ? CAKE_BGMV_MOE_EXPAND_T128_PF : CAKE_BGMV_MOE_EXPAND_T128"
        in binding
    )
    assert "cudaLaunchAttributeProgrammaticStreamSerialization" in binding
    assert "programmaticStreamSerializationAllowed = pdl_mode != 0 ? 1 : 0" in binding
    # clang-format may wrap the signature; match across whitespace.
    assert re.search(r"int64_t pdl_mode,\s*int64_t cuda_stream\)", binding)
    assert re.search(r"kRouteAdvance,\s*hidden\)", binding)
    assert "TensorView route_index" in binding
    assert "CHECK_INPUT_TYPE(route_index, dl_int32)" in binding
    assert "kRouteIndexWordsPerToken = 3 + kRouteIndexMaxRoutes" in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(configure" in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run" in binding


def test_route_index_workspace_sizing():
    assert cake_bgmv_moe.CAKE_BGMV_MOE_ROUTE_INDEX_MAX_ROUTES == 16
    assert cake_bgmv_moe.CAKE_BGMV_MOE_ROUTE_INDEX_HEADER_WORDS == 4
    assert cake_bgmv_moe.CAKE_BGMV_MOE_ROUTE_INDEX_WORDS_PER_TOKEN == 19
    split_words = 8 * 128 * 64 + 128 * 8
    assert cake_bgmv_moe.CAKE_BGMV_MOE_SHRINK_SPLIT_MAX == 8
    assert cake_bgmv_moe.CAKE_BGMV_MOE_SHRINK_SPLIT_MAX_PAIRS == 128
    assert cake_bgmv_moe.cake_bgmv_moe_route_index_words(1) == 4 + 19
    assert cake_bgmv_moe.cake_bgmv_moe_route_index_numel(1) == 4 + 19 + split_words
    assert (
        cake_bgmv_moe.cake_bgmv_moe_route_index_numel(4096)
        == 4 + 4096 * 19 + split_words
    )
    for hidden in cake_bgmv_moe.CAKE_BGMV_MOE_HIDDEN_SIZES:
        body = (
            cake_bgmv_moe._get_csrc_dir() / f"cake_bgmv_moe_bf16_h{hidden}.cu"
        ).read_text()
        assert "#define CAKE_BGMV_MOE_SMEM_EXPAND_T64 " in body
        assert "#define CAKE_BGMV_MOE_SMEM_EXPAND_TOKEN " in body
        assert "#define CAKE_BGMV_MOE_SMEM_EXPAND_DUAL " in body


@pytest.mark.parametrize(
    ("num_pairs", "rank", "hidden_size", "expected"),
    [
        # 16 tokens x top-k 2 at hidden 7168, rank 32: 128 CTAs already -> no split;
        # small grid -> three-stage ring (form 2)
        (32, 32, 7168, (2, 1)),
        # rank 8 at 32 pairs: 32 CTAs -> 4 splits (7 tiles available)
        (32, 8, 7168, (2, 4)),
        # 4 tokens at hidden 5888: 32 CTAs -> 4 of the 6 tiles' worth of splits
        (8, 32, 5888, (2, 4)),
        # hidden 736 has one tile: no split possible
        (8, 64, 736, (2, 1)),
        # wide prefill grids never split and run the two-stage ring (form 0)
        (8192, 32, 3072, (0, 1)),
        (256, 64, 4096, (0, 1)),
        # ring-depth break-even on SM90: 512 CTAs still three stages, 520 and 1024 two
        (64, 64, 7168, (2, 1)),
        (65, 64, 7168, (0, 1)),
        (128, 64, 7168, (0, 1)),
        (256, 16, 4096, (2, 1)),
        (512, 16, 4096, (0, 1)),
    ],
)
def test_generic_shrink_launch_selection(num_pairs, rank, hidden_size, expected):
    assert (
        cake_bgmv_moe.select_cake_bgmv_moe_generic_shrink(
            num_pairs, rank, hidden_size, "sm90a"
        )
        == expected
    )


@pytest.mark.parametrize(
    "num_pairs,rank,hidden_size,arch,expected",
    [
        # lever 11d: Blackwell keeps three stages up to 1024 CTAs when hidden >= 4096
        (65, 64, 7168, "sm100a", (2, 1)),
        (128, 64, 7168, "sm100a", (2, 1)),
        (128, 64, 7168, "sm103a", (2, 1)),
        (512, 16, 4096, "sm100a", (2, 1)),
        (1024, 8, 5888, "sm103a", (2, 1)),
        # ... but not beyond one wave, and not for short K loops
        (129, 64, 7168, "sm100a", (0, 1)),
        (1025, 8, 7168, "sm103a", (0, 1)),
        (512, 16, 3072, "sm100a", (0, 1)),
        (1024, 8, 2048, "sm103a", (0, 1)),
        (1024, 8, 768, "sm100a", (0, 1)),
        # SM90 never takes the Blackwell extension
        (128, 64, 7168, "sm90a", (0, 1)),
        (1024, 8, 7168, "sm90a", (0, 1)),
        # the <= 512 rule is arch-independent
        (64, 64, 7168, "sm100a", (2, 1)),
        (512, 8, 768, "sm90a", (2, 1)),
    ],
)
def test_generic_shrink_launch_selection_blackwell_deep_ring(
    num_pairs, rank, hidden_size, arch, expected
):
    assert (
        cake_bgmv_moe.select_cake_bgmv_moe_generic_shrink(
            num_pairs, rank, hidden_size, arch
        )
        == expected
    )


def test_generic_shrink_launch_selection_rejects_bad_inputs():
    with pytest.raises(ValueError):
        cake_bgmv_moe.select_cake_bgmv_moe_generic_shrink(0, 32, 3072)
    with pytest.raises(ValueError):
        cake_bgmv_moe.select_cake_bgmv_moe_generic_shrink(8, 24, 3072)


def test_grouped_workspace_sizing_and_selector():
    assert cake_bgmv_moe.CAKE_BGMV_MOE_GROUP_TILE_TOKENS == 16
    assert cake_bgmv_moe.CAKE_BGMV_MOE_GROUP_BINS_MAX == 4096
    assert cake_bgmv_moe.CAKE_BGMV_MOE_GROUP_HEADER_WORDS == 4
    # 8192 routes over 1024 bins: 1024 + 1024 tiles at most
    assert cake_bgmv_moe.cake_bgmv_moe_group_max_tiles(8192, 1024) == 1536
    words = cake_bgmv_moe.cake_bgmv_moe_grouped_workspace_words(8192, 4096, 1024)
    assert cake_bgmv_moe.cake_bgmv_moe_group_hist_ctas(8192) == 8
    assert cake_bgmv_moe.cake_bgmv_moe_group_hist_ctas(1) == 1
    assert cake_bgmv_moe.cake_bgmv_moe_group_hist_ctas(1 << 20) == 64
    assert words == 4 + 1025 + 1536 + 8192 + 4096 + 4096 * 16 + 2 * 8 * 1024
    # weight reuse only pays off once routes clearly outnumber the bins
    select = cake_bgmv_moe.select_cake_bgmv_moe_generic_grouped
    assert select(8192, 4096, 8, 128, 4096, 32)
    assert select(4096, 2048, 8, 128, 3072, 32)  # 4 routes per bin
    assert not select(2048, 1024, 8, 128, 3072, 32)  # 2 routes per bin
    # 2048 <= routes < 4096: only rank 64 reuses enough weight per route
    assert select(2048, 1024, 4, 128, 4096, 64)
    assert not select(2048, 1024, 4, 128, 4096, 32)
    assert not select(3072, 1536, 4, 128, 7168, 8)
    assert select(4096, 2048, 4, 128, 7168, 8)
    assert not select(1024, 512, 8, 128, 4096, 32)
    assert not select(8192, 4096, 64, 128, 4096, 32)  # too many bins
    assert not select(8192, 4096, 0, 128, 4096, 32)
    # small per-pair weights: the fixed grouping cost exceeds the reuse win
    assert not select(8192, 4096, 8, 128, 768, 8)
    assert select(
        8192, 4096, 8, 128, 2048, 8
    )  # 16384 weight elems: wins 5-13 % with the multi-CTA prologue
    assert select(8192, 4096, 8, 128, 4096, 8)
    assert select(8192, 4096, 8, 128, 5888, 8)
    assert select(
        8192, 4096, 8, 128, 768, 16
    )  # 12288 weight elems: wins 13-16 % with the multi-CTA prologue
    assert not select(8192, 4096, 8, 128, 512, 16)  # 8192 weight elems: below the floor
    assert select(8192, 4096, 8, 128, 1024, 16)
    assert select(8192, 4096, 8, 128, 2048, 16)
    assert select(8192, 4096, 8, 128, 768, 32)
    for rank in cake_bgmv_moe.CAKE_BGMV_MOE_GENERIC_RANKS:
        metadata = cake_bgmv_moe._generic_metadata(rank, "bfloat16")
        body = (cake_bgmv_moe._get_csrc_dir() / metadata.body).read_text()
        for symbol in (
            metadata.group_hist_symbol,
            metadata.group_scan_symbol,
            metadata.group_scatter_symbol,
            metadata.shrink_grouped_symbol,
            metadata.shrink_grouped_single_symbol,
            metadata.shrink_grouped_ring_symbol,
            metadata.shrink_grouped_ring_single_symbol,
            metadata.shrink_grouped_ring_mixed_symbol,
            metadata.shrink_grouped_ring_mixed_single_symbol,
            metadata.expand_grouped_symbol,
            metadata.combine_grouped_symbol,
            metadata.order_build_symbol,
        ):
            assert body.count(f"{symbol}(") == 1, symbol
        for macro in (
            "CAKE_BGMV_MOE_GENERIC_SMEM_GROUP_HIST",
            "CAKE_BGMV_MOE_GENERIC_SMEM_GROUP_SCAN",
            "CAKE_BGMV_MOE_GENERIC_SMEM_GROUP_SCATTER",
            "CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED",
            "CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED_RING",
            "CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED_RING_MIXED",
            "CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_GROUPED",
            "CAKE_BGMV_MOE_GENERIC_SMEM_COMBINE_GROUPED",
            "CAKE_BGMV_MOE_GENERIC_SMEM_ORDER_BUILD",
        ):
            assert f"#define {macro} " in body, macro
        # Lever 3: the ring form stages two K tiles of x + weight rows (49664 B) behind the
        # 512 B reduction scratch of the direct form.
        assert "#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED 512\n" in body
        assert "#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED_RING 49664\n" in body
        # Lever 34: the mixed-precision form stages the weight rows only (two 16 KiB K tiles).
        assert (
            "#define CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED_RING_MIXED 33280\n"
            in body
        )


def test_order_remap_selector_and_workspace():
    assert cake_bgmv_moe.cake_bgmv_moe_order_workspace_words(3000) == 3000
    assert cake_bgmv_moe.CAKE_BGMV_MOE_ORDER_REMAP_MAX_PAIRS == 4096
    select = cake_bgmv_moe.select_cake_bgmv_moe_order_remap
    # 1024 routes x 7168 x 64 x 2 B = 939 MB of per-route A traffic: SM90 only.
    assert select(1024, 512, 8, 128, 7168, 64, "sm90a")
    assert not select(1024, 512, 8, 128, 7168, 64, "sm100a")
    assert not select(1024, 512, 8, 128, 7168, 64, "sm103a")
    assert select(2048, 1024, 8, 128, 7168, 16, "sm90a")  # 470 MB
    assert select(1024, 512, 8, 128, 7168, 16, "sm90a")  # 235 MB: above the floor
    assert not select(1024, 512, 8, 128, 2048, 64, "sm90a")  # rank 64 below hidden 4096
    assert not select(1024, 512, 8, 128, 3072, 64, "sm90a")
    assert select(1024, 512, 8, 128, 4096, 64, "sm90a")
    assert select(
        1024, 512, 8, 128, 16384, 8, "sm90a"
    )  # 1024 CTAs is a two-stage grid again (deep ring <= 512), 268 MB
    assert not select(1024, 512, 8, 128, 8192, 8, "sm90a")  # 134 MB: below the floor
    assert not select(1024, 512, 8, 128, 3072, 32, "sm90a")  # 201 MB: below it
    assert not select(512, 256, 8, 128, 7168, 64, "sm90a")  # too few routes
    assert not select(8192, 4096, 8, 128, 7168, 64, "sm90a")  # beyond one CTA
    assert not select(2048, 1024, 64, 128, 7168, 64, "sm90a")  # 8192 bins
