"""CPU-side packaging and public-routing tests for Cake FMHA."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import flashinfer
import flashinfer.cake_fmha as cake_api
import flashinfer.decode as decode
import flashinfer.prefill as prefill
import pytest
import torch
from flashinfer.jit.cake_fmha import (
    CAKE_FMHA_FLASHINFER_BINDINGS_SHA256,
    CAKE_FMHA_FLASHINFER_MATRIX_REVISION,
    CAKE_FMHA_MANIFEST_SHA256,
    gen_cake_fmha_compat_module,
    gen_cake_fmha_context_bf16_module,
    gen_cake_fmha_context_fp16_hd256_module,
    gen_cake_fmha_context_fp8_module,
    gen_cake_fmha_context_fp8_hd256_module,
    gen_cake_fmha_context_nvfp4_module,
    gen_cake_fmha_decode_native_bf16_module,
    gen_cake_fmha_decode_native_fp16_hd512_module,
    gen_cake_fmha_decode_native_fp16_nhd_module,
    gen_cake_fmha_decode_quant_bf16q_module,
    gen_cake_fmha_decode_quant_fp8_module,
    gen_cake_fmha_decode_quant_nvfp4_module,
    get_cake_fmha_compat_uri,
    get_cake_fmha_csrc_dir,
    get_cake_fmha_context_bf16_uri,
    get_cake_fmha_context_fp16_hd256_uri,
    get_cake_fmha_context_fp8_uri,
    get_cake_fmha_context_fp8_hd256_uri,
    get_cake_fmha_context_nvfp4_uri,
    get_cake_fmha_decode_native_bf16_uri,
    get_cake_fmha_decode_native_fp16_hd512_uri,
    get_cake_fmha_decode_native_fp16_nhd_uri,
    get_cake_fmha_decode_quant_bf16q_uri,
    get_cake_fmha_decode_quant_fp8_uri,
    get_cake_fmha_decode_quant_nvfp4_uri,
    get_cake_fmha_manifest,
)
from tests.test_helpers.cake_fmha_capability import (
    PINNED_FLASHINFER_REVISION,
    replay_selectors,
)


def test_cake_fmha_manifest_is_authenticated_and_complete() -> None:
    manifest = get_cake_fmha_manifest()
    assert manifest["product"] == "cake_fmha"
    assert manifest["flashinfer_matrix_revision"] == (
        CAKE_FMHA_FLASHINFER_MATRIX_REVISION
    )
    assert manifest["publication"]["promotion_ready"] is True
    assert manifest["capability"]["complete"] is True
    assert manifest["capability"]["cake_coverage_ratio"] == 1.0
    assert manifest["capability"]["upstream_valid_cases"] == 57_280
    assert manifest["capability"]["cake_covered_cases"] == 57_280
    assert manifest["capability"]["route_counts"]["cake_fmha_compat_v1"] == 55_482
    assert len(manifest["route_probes"]) == 29
    assert {probe["label"] for probe in manifest["route_probes"]} >= {
        "correctness_compat_decode_fp8_nhd_separate_group5",
        "correctness_compat_decode_fp8_hnd_shared_group8_partial",
        "correctness_decode_fp8_hnd_shared_group8_full_blocks",
    }
    assert len(manifest["artifacts"]) == 135
    dcp_addon = manifest["add_ons"]["cake_fmha_dcp_spec"]
    assert dcp_addon["installed"] is True
    assert dcp_addon["selection_key"] == "causal_seqlens_kv_global"
    assert set(dcp_addon["manifest"]["families"]) == {
        "dcp_spec_bf16_fp8",
        "dcp_spec_bf16_v1",
        "dcp_spec_bf16_v4",
    }
    assert manifest["components"]["compat_v1"]["launch_binding"] == (
        "cake_fmha_launch_compat_v1"
    )
    assert len(CAKE_FMHA_MANIFEST_SHA256) == 64
    assert len(CAKE_FMHA_FLASHINFER_BINDINGS_SHA256) == 64


def test_cake_fmha_public_manifest_is_defensive_copy() -> None:
    public_manifest = cake_api.cake_fmha_manifest()
    public_manifest["product"] = "mutated"
    assert cake_api.cake_fmha_manifest()["product"] == "cake_fmha"


def test_cake_fmha_registry_accounts_for_manifest_routes_and_components() -> None:
    manifest = cake_api.cake_fmha_manifest()
    manifest_route_counts = manifest["capability"]["route_counts"]
    manifest_optimized_routes = set(manifest_route_counts) - {"cake_fmha_compat_v1"}
    assert manifest_optimized_routes <= set(cake_api._PRODUCT_ROUTE_COMPONENTS)
    assert cake_api._manifest_optimized_route_accounting() == (1_798, 1_798)
    assert cake_api._manifest_authenticated_route_accounting() == (1_798, 1_798)
    for route_name, components in cake_api._PRODUCT_ROUTE_COMPONENTS.items():
        manifest_components = tuple(
            dict.fromkeys(
                item["component"]
                for item in manifest["routes"][route_name]["components"]
            )
        )
        assert components == manifest_components
    routed_components = {
        component
        for components in cake_api._PRODUCT_ROUTE_COMPONENTS.values()
        for component in components
    }
    assert routed_components | {"compat_v1"} == set(manifest["components"])
    assert cake_api._AUTHENTICATED_JIT_COMPONENTS == routed_components | {
        "compat_v1"
    }


def test_cake_fmha_high_level_selectors_match_pinned_capability_corpus(
    monkeypatch,
) -> None:
    """Replay every valid public cell, not only manifest route-name totals."""

    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    report = replay_selectors()
    manifest = get_cake_fmha_manifest()

    assert PINNED_FLASHINFER_REVISION == CAKE_FMHA_FLASHINFER_MATRIX_REVISION
    assert report.raw_cases == 80_768
    assert report.valid_cases == 57_280
    assert report.optimized_cases == 1_798
    assert report.compat_cases == 55_482
    assert report.route_counts == dict(
        sorted(manifest["capability"]["route_counts"].items())
    )
    assert report.digest == (
        "d47bf01c2d27409c6a39759d02e30bb9df65e98c353f53d7335081dd26b3f3a8"
    )


def test_cake_fmha_jit_spec_uses_versioned_standalone_sources(monkeypatch) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    for target in ("sm100a", "sm103a"):
        spec = gen_cake_fmha_compat_module(target)
        assert spec.name == get_cake_fmha_compat_uri(target)
        source_names = {Path(source).name for source in spec.sources}
        assert source_names == {
            "cake_fmha_compat_v1.cu",
            "cake_fmha_compat_v1_binding.cu",
            "cake_fmha_jit_binding.cu",
        }


def test_cake_fmha_decode_native_bf16_jit_selects_one_manifest_member(
    monkeypatch,
) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    spec = gen_cake_fmha_decode_native_bf16_module(
        "sm100a",
        2,
        1,
        4,
        2,
        has_sink=False,
        has_window=False,
        use_scale_ptr=True,
        retain_kv_l2=True,
    )
    assert spec.name == get_cake_fmha_decode_native_bf16_uri(
        "sm100a",
        2,
        1,
        4,
        2,
        has_sink=False,
        has_window=False,
        use_scale_ptr=True,
        retain_kv_l2=True,
    )
    assert {Path(source).name for source in spec.sources} == {
        "has_sink0_has_window0_retain_kv_l21_use_scale_ptr1.cu",
        "cake_fmha_decode_native_bf16_binding.cu",
        "cake_fmha_decode_native_bf16_jit_binding.cu",
    }
    assert "-DBATCH_SIZE=2" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_USE_SCALE_PTR=1" in spec.extra_cuda_cflags


@pytest.mark.parametrize(
    ("target", "manifest_arch"),
    (("sm100a", "sm_100a"), ("sm103a", "sm_103a")),
)
def test_cake_fmha_decode_native_bf16_jit_selects_all_exact_manifest_members(
    monkeypatch, target, manifest_arch
) -> None:
    import flashinfer.jit.core as jit_core
    from flashinfer.jit import cake_fmha as cake_jit

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    component = get_cake_fmha_manifest()["components"]["decode_native_bf16"]
    exact_members = [
        member
        for member in component["source_family"]
        if "BATCH_SIZE" in member["selector"]
    ]
    assert len(exact_members) == 5

    selected_batches = set()
    csrc_dir = cake_jit.get_cake_fmha_csrc_dir()
    for member in exact_members:
        selector = member["selector"]
        selected_batches.add(selector["BATCH_SIZE"])
        spec = gen_cake_fmha_decode_native_bf16_module(
            target,
            selector["BATCH_SIZE"],
            selector["Q_LEN"],
            selector["NUM_Q_HEADS"],
            selector["NUM_KV_HEADS"],
            has_sink=bool(selector["HAS_SINK"]),
            has_window=bool(selector["HAS_WINDOW"]),
            use_scale_ptr=bool(selector["USE_SCALE_PTR"]),
            retain_kv_l2=bool(selector["RETAIN_KV_L2"]),
        )
        assert Path(spec.sources[0]) == csrc_dir / member["sources"][manifest_arch]
        launch_override = member.get("launch_override") or {}
        expected_binding = launch_override.get(
            "binding_source", component["binding_source"]
        )
        assert Path(spec.sources[1]) == csrc_dir / expected_binding

        route = cake_api.CakeFmhaDecodeRoute(
            target=target,
            batch_size=selector["BATCH_SIZE"],
            q_len=selector["Q_LEN"],
            num_q_heads=selector["NUM_Q_HEADS"],
            num_kv_heads=selector["NUM_KV_HEADS"],
            has_sink=bool(selector["HAS_SINK"]),
            has_window=bool(selector["HAS_WINDOW"]),
            use_scale_ptr=bool(selector["USE_SCALE_PTR"]),
            retain_kv_l2=bool(selector["RETAIN_KV_L2"]),
        )
        assert cake_api.cake_fmha_route_is_optimized(route)

        if selector["BATCH_SIZE"] == 4:
            assert Path(spec.sources[1]).name == (
                "cake_fmha_decode_native_bf16_b4_exact_cga_binding.cu"
            )
        if selector["BATCH_SIZE"] == 256:
            assert selector == {
                "BATCH_SIZE": 256,
                "HAS_SINK": 1,
                "HAS_WINDOW": 0,
                "NUM_KV_HEADS": 4,
                "NUM_Q_HEADS": 32,
                "Q_LEN": 1,
                "RETAIN_KV_L2": 0,
                "USE_SCALE_PTR": 0,
            }

    assert selected_batches == {4, 128, 256}


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize(
    ("has_sink", "retain_kv_l2"),
    ((True, False), (False, False), (False, True), (True, True)),
)
def test_cake_fmha_decode_native_bf16_absent_selectors_fall_back_to_compat(
    monkeypatch, target, has_sink, retain_kv_l2
) -> None:
    compat_sentinel = object()
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    monkeypatch.setattr(
        cake_api,
        "load_cake_fmha_compat_module",
        lambda selected_target: (
            compat_sentinel
            if selected_target == target
            else pytest.fail("compat target mismatch")
        ),
    )
    monkeypatch.setattr(
        cake_api,
        "load_cake_fmha_decode_native_bf16_module",
        lambda *args, **kwargs: pytest.fail("absent BF16 selector must not load"),
    )

    batch_size = 2
    query = torch.empty((batch_size, 4, 128), dtype=torch.bfloat16)
    key = torch.empty((4, 2, 16, 128), dtype=torch.bfloat16)
    max_seq_len = 31 if retain_kv_l2 else 1153
    route = cake_api.select_cake_fmha_decode_route(
        query.device,
        query=query,
        key_cache=key,
        value_cache=torch.empty_like(key),
        out=torch.empty_like(query),
        workspace_buffer=torch.empty(4096, dtype=torch.uint8),
        block_tables=torch.zeros((batch_size, 2), dtype=torch.int32),
        seq_lens=torch.full((batch_size,), max_seq_len, dtype=torch.int32),
        batch_size=batch_size,
        q_len=1,
        max_seq_len=max_seq_len,
        window_left=127,
        bmm1_scale=torch.ones(1, dtype=torch.float32),
        bmm2_scale=1.0,
        o_scale=1.0,
        sinks=torch.zeros(4, dtype=torch.float32) if has_sink else None,
        kv_layout="HND",
        uses_shared_paged_kv_idx=True,
        cum_seq_lens_q=None,
        key_block_scales=None,
        value_block_scales=None,
        skip_softmax_threshold_scale_factor=None,
        enable_block_sparse_attention=False,
    )
    assert route is None
    assert (
        cake_api.get_cake_fmha_decode_module(torch.device("cpu"), route)
        is compat_sentinel
    )


def test_cake_fmha_decode_native_fp16_nhd_jit_selects_one_manifest_member(
    monkeypatch,
) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    spec = gen_cake_fmha_decode_native_fp16_nhd_module(
        "sm103a",
        2,
        1,
        4,
        2,
        has_sink=False,
        has_window=False,
        use_scale_ptr=True,
        retain_kv_l2=True,
    )
    assert spec.name == get_cake_fmha_decode_native_fp16_nhd_uri(
        "sm103a",
        2,
        1,
        4,
        2,
        has_sink=False,
        has_window=False,
        use_scale_ptr=True,
        retain_kv_l2=True,
    )
    assert {Path(source).name for source in spec.sources} == {
        "has_sink0_has_window0_retain_kv_l21_use_scale_ptr1.cu",
        "cake_fmha_decode_native_fp16_nhd_binding.cu",
        "cake_fmha_decode_native_fp16_nhd_jit_binding.cu",
    }
    assert "-DNUM_KV_HEADS=2" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_USE_SCALE_PTR=1" in spec.extra_cuda_cflags


def test_cake_fmha_decode_native_fp16_hd512_jit_selects_one_manifest_member(
    monkeypatch,
) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    spec = gen_cake_fmha_decode_native_fp16_hd512_module(
        "sm100a",
        2,
        1,
        4,
        2,
        has_window=True,
        use_scale_ptr=True,
        retain_kv_l2=False,
    )
    assert spec.name == get_cake_fmha_decode_native_fp16_hd512_uri(
        "sm100a",
        2,
        1,
        4,
        2,
        has_window=True,
        use_scale_ptr=True,
        retain_kv_l2=False,
    )
    assert {Path(source).name for source in spec.sources} == {
        "has_window1_retain_kv_l20_use_scale_ptr1.cu",
        "cake_fmha_decode_native_fp16_hd512_binding.cu",
        "cake_fmha_decode_native_fp16_hd512_jit_binding.cu",
    }
    assert "-DCAKE_FMHA_HAS_WINDOW=1" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_USE_SCALE_PTR=1" in spec.extra_cuda_cflags
    assert not any("HAS_SINK" in flag for flag in spec.extra_cuda_cflags)


def test_cake_fmha_decode_quant_bf16q_jit_selects_one_manifest_member(
    monkeypatch,
) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    spec = gen_cake_fmha_decode_quant_bf16q_module(
        "sm103a",
        2,
        1,
        4,
        2,
        32,
    )
    assert spec.name == get_cake_fmha_decode_quant_bf16q_uri(
        "sm103a",
        2,
        1,
        4,
        2,
        32,
    )
    assert {Path(source).name for source in spec.sources} == {
        "page_size32.cu",
        "cake_fmha_decode_quant_bf16q_binding.cu",
        "cake_fmha_decode_quant_bf16q_jit_binding.cu",
    }
    assert "-DBATCH_SIZE=2" in spec.extra_cuda_cflags
    assert "-DNUM_Q_HEADS=4" in spec.extra_cuda_cflags
    assert "-DNUM_KV_HEADS=2" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_PAGE_SIZE=32" in spec.extra_cuda_cflags


def test_cake_fmha_decode_quant_fp8_jit_selects_main_and_reducer(
    monkeypatch,
) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    spec = gen_cake_fmha_decode_quant_fp8_module(
        "sm103a",
        2,
        1,
        16,
        2,
        32,
        full_blocks=True,
    )
    assert spec.name == get_cake_fmha_decode_quant_fp8_uri(
        "sm103a",
        2,
        1,
        16,
        2,
        32,
        full_blocks=True,
    )
    assert {Path(source).name for source in spec.sources} == {
        "full_blocks1_page_size32.cu",
        "cake_fmha_decode_quant_fp8_binding.cu",
        "default.cu",
        "cake_fmha_decode_quant_fp8_reduce_binding.cu",
        "cake_fmha_decode_quant_fp8_jit_binding.cu",
    }
    assert "-DBATCH_SIZE=2" in spec.extra_cuda_cflags
    assert "-DNUM_Q_HEADS=16" in spec.extra_cuda_cflags
    assert "-DNUM_KV_HEADS=2" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_PAGE_SIZE=32" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_FULL_BLOCKS=1" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_NVFP4=0" in spec.extra_cuda_cflags


def test_cake_fmha_decode_quant_nvfp4_jit_selects_main_and_reducer(
    monkeypatch,
) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    spec = gen_cake_fmha_decode_quant_nvfp4_module(
        "sm103a", 2, 1, 4, 2, 32
    )
    assert spec.name == get_cake_fmha_decode_quant_nvfp4_uri(
        "sm103a", 2, 1, 4, 2, 32
    )
    assert {Path(source).name for source in spec.sources} == {
        "page_size32.cu",
        "cake_fmha_decode_quant_nvfp4_binding.cu",
        "default.cu",
        "cake_fmha_decode_quant_fp8_reduce_binding.cu",
        "cake_fmha_decode_quant_fp8_jit_binding.cu",
    }
    assert not any("native_qmul4" in str(source) for source in spec.sources)
    assert "-DBATCH_SIZE=2" in spec.extra_cuda_cflags
    assert "-DNUM_Q_HEADS=4" in spec.extra_cuda_cflags
    assert "-DNUM_KV_HEADS=2" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_PAGE_SIZE=32" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_FULL_BLOCKS=0" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_NVFP4=1" in spec.extra_cuda_cflags


def test_cake_fmha_context_bf16_jit_selects_one_manifest_member(
    monkeypatch,
) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    spec = gen_cake_fmha_context_bf16_module(
        "sm103a",
        1,
        4,
        2,
        2,
        16,
        1,
        is_causal=True,
        return_lse=True,
        enable_sink=False,
    )
    assert spec.name == get_cake_fmha_context_bf16_uri(
        "sm103a",
        1,
        4,
        2,
        2,
        16,
        1,
        is_causal=True,
        return_lse=True,
        enable_sink=False,
    )
    assert {Path(source).name for source in spec.sources} == {
        "enable_sink0_is_causal1_return_lse1.cu",
        "cake_fmha_context_bf16_binding.cu",
        "cake_fmha_context_bf16_jit_binding.cu",
    }
    assert "-DNUM_M_BLOCKS=1" in spec.extra_cuda_cflags
    assert "-DHEADS_PER_GROUP=2" in spec.extra_cuda_cflags
    assert "-DPACK_G=2" in spec.extra_cuda_cflags
    assert "-DTOK_PER_STAGE=64" in spec.extra_cuda_cflags


@pytest.mark.parametrize(
    ("profile", "args", "expected"),
    [
        (
            "q511",
            ("sm100a", 11, 10, 2, 5, 32, 1),
            {
                "ENABLE_SINK": 0,
                "HEADS_PER_GROUP": 5,
                "IS_CAUSAL": 1,
                "L2_SWIZZLE": 1,
                "NUM_M_BLOCKS": 11,
                "NUM_Q_HEADS": 10,
                "PACK_G": 5,
                "PAGE_SIZE": 32,
                "RETURN_LSE": 0,
                "SINGLE_MASK_LOOP": 1,
                "TOK_PER_STAGE": 25,
            },
        ),
        (
            "q257",
            ("sm103a", 6, 10, 2, 5, 1024, 8),
            {
                "ENABLE_SINK": 0,
                "HEADS_PER_GROUP": 5,
                "IS_CAUSAL": 1,
                "L2_SWIZZLE": 8,
                "NUM_M_BLOCKS": 6,
                "NUM_Q_HEADS": 10,
                "PACK_G": 5,
                "PAGE_SIZE": 1024,
                "RETURN_LSE": 0,
                "SINGLE_MASK_LOOP": 1,
                "TOK_PER_STAGE": 25,
            },
        ),
    ],
)
def test_cake_fmha_context_bf16_exact_profile_selector(
    profile, args, expected
) -> None:
    from flashinfer.jit import cake_fmha as cake_jit

    assert cake_jit._validate_context_specialization(
        *args,
        is_causal=True,
        return_lse=False,
        enable_sink=False,
        exact_profile=profile,
    ) == expected
    with pytest.raises(ValueError, match="does not match its fixed selector"):
        cake_jit._validate_context_specialization(
            *args[:-1],
            8 if args[-1] == 1 else 1,
            is_causal=True,
            return_lse=False,
            enable_sink=False,
            exact_profile=profile,
        )


def test_cake_fmha_context_fp8_jit_selects_one_manifest_member(monkeypatch) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    spec = gen_cake_fmha_context_fp8_module(
        "sm100a",
        1,
        32,
        4,
        8,
        64,
        8,
        is_causal=True,
        return_lse=False,
        enable_sink=False,
    )
    assert spec.name == get_cake_fmha_context_fp8_uri(
        "sm100a",
        1,
        32,
        4,
        8,
        64,
        8,
        is_causal=True,
        return_lse=False,
        enable_sink=False,
    )
    assert {Path(source).name for source in spec.sources} == {
        "enable_sink0_is_causal1_return_lse0.cu",
        "cake_fmha_context_fp8_binding.cu",
        "cake_fmha_context_fp8_jit_binding.cu",
    }
    assert "-DNUM_Q_HEADS=32" in spec.extra_cuda_cflags
    assert "-DHEADS_PER_GROUP=8" in spec.extra_cuda_cflags
    assert "-DPACK_G=8" in spec.extra_cuda_cflags
    assert "-DTOK_PER_STAGE=16" in spec.extra_cuda_cflags


def test_cake_fmha_context_nvfp4_jit_selects_fused_member(
    monkeypatch,
) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    spec = gen_cake_fmha_context_nvfp4_module(
        "sm100a", 1, 32, 4, 8, 16, 8
    )
    assert spec.name == get_cake_fmha_context_nvfp4_uri(
        "sm100a", 1, 32, 4, 8, 16, 8
    )
    assert {Path(source).name for source in spec.sources} == {
        "enable_sink0_is_causal1_return_lse0_static_one_tile1.cu",
        "cake_fmha_context_nvfp4_binding.cu",
        "cake_fmha_context_fp8_jit_binding.cu",
    }
    assert "-DNUM_KV_HEADS=4" in spec.extra_cuda_cflags
    assert "-DPAGE_SIZE=16" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_CONTEXT_NVFP4=1" in spec.extra_cuda_cflags
    adapter = next(
        Path(source)
        for source in spec.sources
        if Path(source).name == "cake_fmha_context_fp8_jit_binding.cu"
    ).read_text(encoding="utf-8")
    assert "cake_fmha_launch_context_nvfp4(" in adapter
    assert "cake_fmha_launch_context_nvfp4_dequant" not in adapter
    assert adapter.count("reinterpret_cast<CakeFmhaTensorMap const*>") >= 5
    assert "unsigned int grid_x = total_tiles;" in adapter


@pytest.mark.parametrize(
    "relative_path",
    [
        "cake_fmha_jit_binding.cu",
        "jit/cake_fmha_context_bf16_jit_binding.cu",
        "jit/cake_fmha_context_fp8_jit_binding.cu",
        "jit/cake_fmha_context_hd256_jit_binding.cu",
    ],
)
def test_cake_fmha_context_adapters_match_public_feature_abi(
    relative_path,
) -> None:
    adapter = (get_cake_fmha_csrc_dir() / relative_path).read_text(encoding="utf-8")
    signature_start = adapter.index("void cake_paged_attention_context(")
    signature_end = adapter.index(") {", signature_start)
    signature = adapter[signature_start:signature_end]
    shared = signature.index("Optional<bool> uses_shared_paged_kv_idx")
    fp16 = signature.index("Optional<bool> use_fp16_softmax")
    spcompress = signature.index("Optional<bool> uses_spcompress")
    causal = signature.index("bool is_causal")
    assert shared < fp16 < spcompress < causal
    assert "TVM_FFI_ICHECK(!use_fp16_softmax.value_or(false));" in adapter
    assert "TVM_FFI_ICHECK(!uses_spcompress.value_or(false));" in adapter


@pytest.mark.parametrize(
    ("kind", "generator", "uri", "body", "binding", "fp8_flag"),
    [
        (
            "fp16",
            gen_cake_fmha_context_fp16_hd256_module,
            get_cake_fmha_context_fp16_hd256_uri,
            "is_causal0.cu",
            "cake_fmha_context_fp16_hd256_binding.cu",
            "0",
        ),
        (
            "fp8",
            gen_cake_fmha_context_fp8_hd256_module,
            get_cake_fmha_context_fp8_hd256_uri,
            "is_causal1_output_bf161.cu",
            "cake_fmha_context_fp8_hd256_binding.cu",
            "1",
        ),
    ],
)
def test_cake_fmha_context_hd256_jit_selects_main_and_support(
    monkeypatch, kind, generator, uri, body, binding, fp8_flag
) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    spec = generator("sm103a", 2, 10, 2, 32)
    assert spec.name == uri("sm103a", 2, 10, 2, 32)
    assert {Path(source).name for source in spec.sources} == {
        body,
        binding,
        "cake_fmha_hd256_support.cu",
        "cake_fmha_context_hd256_jit_binding.cu",
    }
    assert "-DNUM_M_BLOCKS=2" in spec.extra_cuda_cflags
    assert "-DNUM_Q_HEADS=10" in spec.extra_cuda_cflags
    assert "-DNUM_KV_HEADS=2" in spec.extra_cuda_cflags
    assert "-DHEADS_PER_GROUP=5" in spec.extra_cuda_cflags
    assert f"-DCAKE_FMHA_HD256_FP8={fp8_flag}" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_SOURCE_PAGE_SIZE=32" in spec.extra_cuda_cflags


def test_cake_fmha_decode_route_is_optimized_only_on_exact_bf16_domain(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    query = torch.empty((2, 4, 128), dtype=torch.bfloat16)
    key = torch.empty((4, 2, 16, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    out = torch.empty_like(query)
    block_tables = torch.zeros((2, 2), dtype=torch.int32)
    seq_lens = torch.tensor([31, 29], dtype=torch.int32)
    kwargs = dict(
        query=query,
        key_cache=key,
        value_cache=value,
        out=out,
        workspace_buffer=torch.empty(4096, dtype=torch.uint8),
        block_tables=block_tables,
        seq_lens=seq_lens,
        batch_size=2,
        q_len=1,
        max_seq_len=31,
        window_left=-1,
        bmm1_scale=torch.ones(1, dtype=torch.float32),
        bmm2_scale=1.0,
        o_scale=1.0,
        sinks=None,
        kv_layout="HND",
        uses_shared_paged_kv_idx=True,
        cum_seq_lens_q=None,
        key_block_scales=None,
        value_block_scales=None,
        skip_softmax_threshold_scale_factor=None,
        enable_block_sparse_attention=False,
    )
    route = cake_api.select_cake_fmha_decode_route(query.device, **kwargs)
    assert route == cake_api.CakeFmhaDecodeRoute(
        target="sm100a",
        batch_size=2,
        q_len=1,
        num_q_heads=4,
        num_kv_heads=2,
        has_sink=False,
        has_window=False,
        use_scale_ptr=True,
        retain_kv_l2=True,
    )
    assert cake_api.cake_fmha_route_is_optimized(route)
    for invalid_workspace in (
        torch.empty(16, dtype=torch.uint8),
        torch.empty((4096, 2), dtype=torch.uint8)[:, 0],
    ):
        assert (
            cake_api.select_cake_fmha_decode_route(
                query.device,
                **{**kwargs, "workspace_buffer": invalid_workspace},
            )
            is None
        )
    assert (
        cake_api.select_cake_fmha_decode_route(
            query.device, **{**kwargs, "uses_shared_paged_kv_idx": False}
        )
        is None
    )
    assert (
        cake_api.select_cake_fmha_decode_route(
            query.device,
            **{
                **kwargs,
                "block_tables": torch.zeros((2, 4), dtype=torch.int32)[:, ::2],
            },
        )
        is None
    )
    valid_lse = torch.empty((2, 4), dtype=torch.float32)
    assert cake_api.select_cake_fmha_decode_route(
        query.device, **{**kwargs, "lse": valid_lse}
    ) == route
    for invalid_lse in (
        torch.empty((2, 4), dtype=torch.float16),
        torch.empty((2, 5), dtype=torch.float32)[:, :4],
    ):
        assert (
            cake_api.select_cake_fmha_decode_route(
                query.device, **{**kwargs, "lse": invalid_lse}
            )
            is None
        )
    misaligned_key = torch.empty((4, 2, 16, 129), dtype=torch.bfloat16)[..., :128]
    assert misaligned_key.stride() == (4128, 2064, 129, 1)
    assert (
        cake_api.select_cake_fmha_decode_route(
            query.device,
            **{
                **kwargs,
                "key_cache": misaligned_key,
                "value_cache": torch.empty_like(misaligned_key),
            },
        )
        is None
    )


def test_cake_fmha_exact_sink_no_lse_member_falls_back_for_caller_lse(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    batch_size = 256
    num_q_heads = 32
    query = torch.empty((batch_size, num_q_heads, 128), dtype=torch.bfloat16)
    key = torch.empty((1, 4, 16, 128), dtype=torch.bfloat16)
    kwargs = dict(
        query=query,
        key_cache=key,
        value_cache=torch.empty_like(key),
        out=torch.empty_like(query),
        workspace_buffer=torch.empty(2 << 20, dtype=torch.uint8),
        block_tables=torch.zeros((batch_size, 256), dtype=torch.int32),
        seq_lens=torch.full((batch_size,), 4096, dtype=torch.int32),
        batch_size=batch_size,
        q_len=1,
        max_seq_len=4096,
        window_left=-1,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
        o_scale=1.0,
        sinks=torch.zeros(num_q_heads, dtype=torch.float32),
        kv_layout="HND",
        uses_shared_paged_kv_idx=True,
        cum_seq_lens_q=None,
        key_block_scales=None,
        value_block_scales=None,
        skip_softmax_threshold_scale_factor=None,
        enable_block_sparse_attention=False,
    )
    route = cake_api.select_cake_fmha_decode_route(query.device, **kwargs)
    assert route == cake_api.CakeFmhaDecodeRoute(
        target="sm100a",
        batch_size=batch_size,
        q_len=1,
        num_q_heads=num_q_heads,
        num_kv_heads=4,
        has_sink=True,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=False,
    )
    assert cake_api.cake_fmha_route_is_optimized(route)

    caller_lse = torch.empty((batch_size, num_q_heads), dtype=torch.float32)
    assert (
        cake_api.select_cake_fmha_decode_route(
            query.device, **{**kwargs, "lse": caller_lse}
        )
        is None
    )


def test_cake_fmha_decode_candidate_selection_for_adapter_families(monkeypatch) -> None:
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")

    def select(
        query,
        key,
        out,
        *,
        kv_layout,
        shared,
        scales=None,
        workspace_bytes=65536,
    ):
        batch_size = 2
        return cake_api.select_cake_fmha_decode_route(
            query.device,
            query=query,
            key_cache=key,
            value_cache=torch.empty_like(key),
            out=out,
            workspace_buffer=torch.empty(workspace_bytes, dtype=torch.uint8),
            block_tables=torch.zeros(
                (batch_size, 2) if shared else (batch_size, 2, 2),
                dtype=torch.int32,
            ),
            seq_lens=torch.tensor([32, 32], dtype=torch.int32),
            batch_size=batch_size,
            q_len=1,
            max_seq_len=32,
            window_left=-1,
            bmm1_scale=0.125,
            bmm2_scale=1.0,
            o_scale=1.0,
            sinks=None,
            kv_layout=kv_layout,
            uses_shared_paged_kv_idx=shared,
            cum_seq_lens_q=None,
            key_block_scales=None if scales is None else scales,
            value_block_scales=None if scales is None else torch.empty_like(scales),
            skip_softmax_threshold_scale_factor=None,
            enable_block_sparse_attention=False,
            lse=None,
        )

    fp16_q = torch.empty((2, 4, 128), dtype=torch.float16)
    # decode.py normalizes logical NHD with a zero-copy transpose before route
    # selection. The FP16 adapter must accept this exact non-contiguous view.
    logical_nhd = torch.empty((4, 32, 2, 128), dtype=torch.float16)
    normalized_nhd = logical_nhd.transpose(-3, -2)
    assert normalized_nhd.shape == (4, 2, 32, 128)
    assert normalized_nhd.stride() == (8192, 128, 256, 1)
    assert not normalized_nhd.is_contiguous()
    fp16_route = select(
        fp16_q,
        normalized_nhd,
        torch.empty_like(fp16_q),
        kv_layout="NHD",
        shared=False,
    )
    assert fp16_route is not None
    assert fp16_route.component == "decode_native_fp16_nhd"
    assert (
        cake_api.select_cake_fmha_decode_route(
            fp16_q.device,
            query=fp16_q,
            key_cache=normalized_nhd,
            value_cache=torch.empty_like(normalized_nhd),
            out=torch.empty_like(fp16_q),
            workspace_buffer=torch.empty(4096, dtype=torch.uint8),
            block_tables=torch.zeros((2, 2, 4), dtype=torch.int32)[:, :, ::2],
            seq_lens=torch.tensor([32, 32], dtype=torch.int32),
            batch_size=2,
            q_len=1,
            max_seq_len=32,
            window_left=-1,
            bmm1_scale=0.125,
            bmm2_scale=1.0,
            o_scale=1.0,
            sinks=None,
            kv_layout="NHD",
            uses_shared_paged_kv_idx=False,
            cum_seq_lens_q=None,
            key_block_scales=None,
            value_block_scales=None,
            skip_softmax_threshold_scale_factor=None,
            enable_block_sparse_attention=False,
            lse=None,
        )
        is None
    )
    misaligned_nhd = torch.empty((4, 32, 2, 129), dtype=torch.float16)[
        ..., :128
    ].transpose(-3, -2)
    assert misaligned_nhd.stride() == (8256, 129, 258, 1)
    assert (
        select(
            fp16_q,
            misaligned_nhd,
            torch.empty_like(fp16_q),
            kv_layout="NHD",
            shared=False,
        )
        is None
    )

    bf16_q = torch.empty((2, 4, 128), dtype=torch.bfloat16)
    bf16q_route = select(
        bf16_q,
        torch.empty((4, 2, 16, 128), dtype=torch.float8_e4m3fn),
        torch.empty_like(bf16_q),
        kv_layout="HND",
        shared=True,
    )
    assert bf16q_route is not None
    assert bf16q_route.component == "decode_quant_bf16q"
    bf16q_group7 = select(
        torch.empty((2, 14, 128), dtype=torch.bfloat16),
        torch.empty((4, 2, 16, 128), dtype=torch.float8_e4m3fn),
        torch.empty((2, 14, 128), dtype=torch.bfloat16),
        kv_layout="HND",
        shared=True,
    )
    assert bf16q_group7 is not None
    assert bf16q_group7.component == "decode_quant_bf16q"
    assert (
        select(
            torch.empty((2, 16, 128), dtype=torch.bfloat16),
            torch.empty((4, 2, 16, 128), dtype=torch.float8_e4m3fn),
            torch.empty((2, 16, 128), dtype=torch.bfloat16),
            kv_layout="HND",
            shared=True,
        )
        is None
    )
    assert (
        select(
            bf16_q,
            torch.empty((4, 2, 16, 128), dtype=torch.float8_e4m3fn),
            torch.empty_like(bf16_q),
            kv_layout="HND",
            shared=True,
            workspace_bytes=4096,
        )
        is None
    )

    fp8_q = torch.empty((2, 4, 128), dtype=torch.float8_e4m3fn)
    nvfp4_route = select(
        fp8_q,
        torch.empty((4, 2, 16, 64), dtype=torch.uint8),
        torch.empty_like(fp8_q),
        kv_layout="HND",
        shared=True,
        scales=torch.empty((4, 2, 16, 8), dtype=torch.float8_e4m3fn),
    )
    assert nvfp4_route is not None
    assert nvfp4_route.component == "decode_quant_nvfp4"
    nvfp4_separate_route = select(
        fp8_q,
        torch.empty((4, 2, 16, 64), dtype=torch.uint8),
        torch.empty_like(fp8_q),
        kv_layout="HND",
        shared=False,
        scales=torch.empty((4, 2, 16, 8), dtype=torch.float8_e4m3fn),
    )
    assert nvfp4_separate_route is not None
    assert nvfp4_separate_route.component == "decode_quant_nvfp4"
    bad_scale_stride = torch.empty(
        (4, 2, 16, 9), dtype=torch.float8_e4m3fn
    )[..., :8]
    assert bad_scale_stride.stride(2) == 9
    assert (
        select(
            fp8_q,
            torch.empty((4, 2, 16, 64), dtype=torch.uint8),
            torch.empty_like(fp8_q),
            kv_layout="HND",
            shared=True,
            scales=bad_scale_stride,
        )
        is None
    )
    assert (
        select(
            fp8_q,
            torch.empty((4, 2, 16, 64), dtype=torch.uint8),
            torch.empty_like(fp8_q),
            kv_layout="HND",
            shared=True,
            scales=torch.empty((4, 2, 16, 8), dtype=torch.float8_e4m3fn),
            workspace_bytes=4096,
        )
        is None
    )
    too_many_group_heads = torch.empty(
        (2, 18, 128), dtype=torch.float8_e4m3fn
    )
    assert (
        select(
            too_many_group_heads,
            torch.empty((4, 2, 16, 64), dtype=torch.uint8),
            torch.empty_like(too_many_group_heads),
            kv_layout="HND",
            shared=True,
            scales=torch.empty((4, 2, 16, 8), dtype=torch.float8_e4m3fn),
        )
        is None
    )

    assert cake_api.cake_fmha_route_is_optimized(fp16_route)
    assert cake_api.cake_fmha_route_is_optimized(bf16q_route)
    assert cake_api.cake_fmha_route_is_optimized(nvfp4_route)

    hd512_q = torch.empty((2, 4, 512), dtype=torch.float16)
    stacked_hnd = torch.empty((4, 2, 2, 64, 512), dtype=torch.float16)
    normalized_hnd = stacked_hnd[:, 0]
    assert normalized_hnd.stride() == (131072, 32768, 512, 1)
    hd512_route = select(
        hd512_q,
        normalized_hnd,
        torch.empty_like(hd512_q),
        kv_layout="HND",
        shared=True,
    )
    assert hd512_route is not None
    assert hd512_route.component == "decode_native_fp16_hd512"
    assert cake_api.cake_fmha_route_is_optimized(hd512_route)
    misaligned_hd512 = torch.empty((4, 2, 64, 513), dtype=torch.float16)[
        ..., :512
    ]
    assert misaligned_hd512.stride() == (65664, 32832, 513, 1)
    assert (
        select(
            hd512_q,
            misaligned_hd512,
            torch.empty_like(hd512_q),
            kv_layout="HND",
            shared=True,
        )
        is None
    )


def test_cake_fmha_fp8_decode_route_requires_exact_full_block_bucket(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    batch_size, num_q_heads, num_kv_heads, page_size = 2, 16, 2, 16
    query = torch.empty(
        (batch_size, num_q_heads, 128), dtype=torch.float8_e4m3fn
    )
    key = torch.empty(
        (64, num_kv_heads, page_size, 128), dtype=torch.float8_e4m3fn
    )
    out = torch.empty_like(query)
    block_tables = torch.zeros((batch_size, 32), dtype=torch.int32)

    def select(seq_lens: torch.Tensor, max_seq_len: int):
        return cake_api.select_cake_fmha_decode_route(
            query.device,
            query=query,
            key_cache=key,
            value_cache=torch.empty_like(key),
            out=out,
            workspace_buffer=torch.empty(1 << 20, dtype=torch.uint8),
            block_tables=block_tables,
            seq_lens=seq_lens,
            batch_size=batch_size,
            q_len=1,
            max_seq_len=max_seq_len,
            window_left=-1,
            bmm1_scale=0.125,
            bmm2_scale=1.0,
            o_scale=1.0,
            sinks=None,
            kv_layout="HND",
            uses_shared_paged_kv_idx=True,
            cum_seq_lens_q=None,
            key_block_scales=None,
            value_block_scales=None,
            skip_softmax_threshold_scale_factor=None,
            enable_block_sparse_attention=False,
            lse=None,
        )

    route = select(torch.tensor([512, 512], dtype=torch.int32), 512)
    assert route is not None
    assert route.component == "decode_quant_fp8"
    assert cake_api.cake_fmha_route_is_optimized(route)
    assert select(torch.tensor([511, 512], dtype=torch.int32), 512) is None
    assert select(torch.tensor([512, 640], dtype=torch.int32), 640) is None


def test_cake_fmha_fp16_nhd_route_loads_its_authenticated_adapter(monkeypatch) -> None:
    fp16_sentinel = object()
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    monkeypatch.setattr(
        cake_api,
        "load_cake_fmha_decode_native_fp16_nhd_module",
        lambda *args, **kwargs: fp16_sentinel,
    )
    monkeypatch.setattr(
        cake_api,
        "load_cake_fmha_decode_native_bf16_module",
        lambda *args, **kwargs: pytest.fail("BF16 loader must not serve FP16 NHD"),
    )
    route = cake_api.CakeFmhaDecodeRoute(
        target="sm100a",
        batch_size=2,
        q_len=1,
        num_q_heads=4,
        num_kv_heads=2,
        has_sink=False,
        has_window=False,
        use_scale_ptr=True,
        retain_kv_l2=True,
        component="decode_native_fp16_nhd",
        page_size=32,
    )
    assert cake_api.get_cake_fmha_decode_module(torch.device("cpu"), route) is fp16_sentinel


def test_cake_fmha_fp16_hd512_route_loads_its_authenticated_adapter(monkeypatch) -> None:
    hd512_sentinel = object()
    observed = {}

    def load_hd512(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return hd512_sentinel

    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    monkeypatch.setattr(
        cake_api,
        "load_cake_fmha_decode_native_fp16_hd512_module",
        load_hd512,
    )
    monkeypatch.setattr(
        cake_api,
        "load_cake_fmha_decode_native_fp16_nhd_module",
        lambda *args, **kwargs: pytest.fail("NHD loader must not serve head-dim 512"),
    )
    route = cake_api.CakeFmhaDecodeRoute(
        target="sm103a",
        batch_size=2,
        q_len=1,
        num_q_heads=4,
        num_kv_heads=2,
        has_sink=False,
        has_window=True,
        use_scale_ptr=True,
        retain_kv_l2=False,
        component="decode_native_fp16_hd512",
        page_size=64,
    )
    assert cake_api.get_cake_fmha_decode_module(torch.device("cpu"), route) is hd512_sentinel
    assert observed == {
        "args": ("sm103a", 2, 1, 4, 2),
        "kwargs": {
            "has_window": True,
            "use_scale_ptr": True,
            "retain_kv_l2": False,
        },
    }


def test_cake_fmha_bf16q_route_loads_its_authenticated_adapter(monkeypatch) -> None:
    bf16q_sentinel = object()
    observed = {}

    def load_bf16q(*args):
        observed["args"] = args
        return bf16q_sentinel

    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    monkeypatch.setattr(
        cake_api,
        "load_cake_fmha_decode_quant_bf16q_module",
        load_bf16q,
    )
    route = cake_api.CakeFmhaDecodeRoute(
        target="sm100a",
        batch_size=2,
        q_len=1,
        num_q_heads=4,
        num_kv_heads=2,
        has_sink=False,
        has_window=False,
        use_scale_ptr=True,
        retain_kv_l2=True,
        component="decode_quant_bf16q",
        page_size=32,
    )
    assert cake_api.get_cake_fmha_decode_module(torch.device("cpu"), route) is bf16q_sentinel
    assert observed == {"args": ("sm100a", 2, 1, 4, 2, 32)}


def test_cake_fmha_fp8_route_loads_its_authenticated_adapter(monkeypatch) -> None:
    fp8_sentinel = object()
    observed = {}

    def load_fp8(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return fp8_sentinel

    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    monkeypatch.setattr(
        cake_api,
        "load_cake_fmha_decode_quant_fp8_module",
        load_fp8,
    )
    route = cake_api.CakeFmhaDecodeRoute(
        target="sm103a",
        batch_size=2,
        q_len=1,
        num_q_heads=16,
        num_kv_heads=2,
        has_sink=False,
        has_window=False,
        use_scale_ptr=True,
        retain_kv_l2=True,
        component="decode_quant_fp8",
        page_size=16,
    )
    assert (
        cake_api.get_cake_fmha_decode_module(torch.device("cpu"), route)
        is fp8_sentinel
    )
    assert observed == {
        "args": ("sm103a", 2, 1, 16, 2, 16),
        "kwargs": {"full_blocks": True},
    }


def test_cake_fmha_nvfp4_route_loads_authenticated_adapter(monkeypatch) -> None:
    sentinel = object()
    observed = {}

    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    monkeypatch.setattr(
        cake_api,
        "load_cake_fmha_decode_quant_nvfp4_module",
        lambda *args: observed.setdefault("args", args) and sentinel,
    )
    route = cake_api.CakeFmhaDecodeRoute(
        target="sm103a",
        batch_size=2,
        q_len=1,
        num_q_heads=4,
        num_kv_heads=2,
        has_sink=False,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=True,
        component="decode_quant_nvfp4",
        page_size=32,
    )
    assert cake_api.get_cake_fmha_decode_module(torch.device("cpu"), route) is sentinel
    assert observed == {"args": ("sm103a", 2, 1, 4, 2, 32)}


def test_cake_fmha_nvfp4_load_failure_fails_closed_to_compat(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    monkeypatch.setattr(
        cake_api,
        "load_cake_fmha_decode_quant_nvfp4_module",
        lambda *args: (_ for _ in ()).throw(RuntimeError("JIT unavailable")),
    )
    monkeypatch.setattr(
        cake_api, "load_cake_fmha_compat_module", lambda target: sentinel
    )
    route = cake_api.CakeFmhaDecodeRoute(
        target="sm103a",
        batch_size=2,
        q_len=1,
        num_q_heads=4,
        num_kv_heads=2,
        has_sink=False,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=True,
        component="decode_quant_nvfp4",
        page_size=32,
    )
    with pytest.warns(RuntimeWarning, match="failed closed to compat_v1"):
        assert (
            cake_api.get_cake_fmha_decode_module(torch.device("cpu"), route)
            is sentinel
        )


def test_cake_fmha_context_candidate_selection_for_adapter_families(monkeypatch) -> None:
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")

    def select(
        query,
        key,
        out,
        *,
        kv_layout,
        shared,
        causal,
        scales=None,
        workspace_bytes=1 << 20,
    ):
        return cake_api.select_cake_fmha_context_route(
            query.device,
            query=query,
            key_cache=key,
            value_cache=torch.empty_like(key),
            out=out,
            block_tables=torch.zeros(
                (2, 2) if shared else (2, 2, 2), dtype=torch.int32
            ),
            seq_lens=torch.tensor([16, 16], dtype=torch.int32),
            batch_size=2,
            max_q_len=8,
            max_kv_len=16,
            window_left=-1,
            bmm1_scale=0.125,
            bmm2_scale=1.0,
            sinks=None,
            uses_shared_paged_kv_idx=shared,
            cum_seq_lens_q=torch.tensor([0, 8, 16], dtype=torch.int32),
            cum_seq_lens_kv=torch.tensor([0, 16, 32], dtype=torch.int32),
            key_block_scales=None if scales is None else scales,
            value_block_scales=None if scales is None else torch.empty_like(scales),
            skip_softmax_threshold_scale_factor=None,
            is_causal=causal,
            lse=None,
            kv_layout=kv_layout,
            workspace_buffer=torch.empty(workspace_bytes, dtype=torch.uint8),
        )

    fp16_q = torch.empty((16, 4, 256), dtype=torch.float16)
    fp16_route = select(
        fp16_q,
        torch.empty((4, 2, 16, 256), dtype=torch.float16),
        torch.empty_like(fp16_q),
        kv_layout="NHD",
        shared=False,
        causal=False,
    )
    assert fp16_route is not None
    assert fp16_route.component == "context_fp16_hd256"

    fp8_q = torch.empty((16, 4, 256), dtype=torch.float8_e4m3fn)
    fp8_hd256_route = select(
        fp8_q,
        torch.empty((4, 2, 16, 256), dtype=torch.float8_e4m3fn),
        torch.empty_like(fp8_q, dtype=torch.bfloat16),
        kv_layout="NHD",
        shared=False,
        causal=True,
    )
    assert fp8_hd256_route is not None
    assert fp8_hd256_route.component == "context_fp8_hd256"

    fp8_q = torch.empty((16, 4, 128), dtype=torch.float8_e4m3fn)
    nvfp4_route = select(
        fp8_q,
        torch.empty((4, 2, 16, 64), dtype=torch.uint8),
        torch.empty_like(fp8_q),
        kv_layout="HND",
        shared=True,
        causal=True,
        scales=torch.empty((4, 2, 16, 8), dtype=torch.float8_e4m3fn),
    )
    assert nvfp4_route is not None
    assert nvfp4_route.component == "context_nvfp4"
    metadata_only_nvfp4_route = select(
        fp8_q,
        torch.empty((4, 2, 16, 64), dtype=torch.uint8),
        torch.empty_like(fp8_q),
        kv_layout="HND",
        shared=True,
        causal=True,
        scales=torch.empty((4, 2, 16, 8), dtype=torch.float8_e4m3fn),
        workspace_bytes=64,
    )
    assert metadata_only_nvfp4_route is not None
    assert metadata_only_nvfp4_route.component == "context_nvfp4"
    assert (
        select(
            fp8_q,
            torch.empty((4, 2, 16, 64), dtype=torch.uint8),
            torch.empty_like(fp8_q),
            kv_layout="HND",
            shared=True,
            causal=True,
            scales=torch.empty((4, 2, 16, 8), dtype=torch.float8_e4m3fn),
            workspace_bytes=1,
        )
        is None
    )

    for route in (fp16_route, fp8_hd256_route):
        assert cake_api.cake_fmha_route_is_optimized(route)
    assert cake_api.cake_fmha_route_is_optimized(nvfp4_route)


def test_cake_fmha_context_bf16_exact_route_loads_exact_member(monkeypatch) -> None:
    sentinel = object()
    observed = {}

    def load(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    monkeypatch.setattr(cake_api, "load_cake_fmha_context_bf16_module", load)
    route = cake_api.CakeFmhaContextRoute(
        target="sm103a",
        component="context_bf16",
        num_m_blocks=11,
        num_q_heads=10,
        num_kv_heads=2,
        pack_g=5,
        page_size=32,
        l2_swizzle=1,
        is_causal=True,
        return_lse=False,
        enable_sink=False,
        exact_profile="q511",
    )
    assert cake_api.get_cake_fmha_context_module(torch.device("cpu"), route) is sentinel
    assert observed == {
        "args": ("sm103a", 11, 10, 2, 5, 32, 1),
        "kwargs": {
            "is_causal": True,
            "return_lse": False,
            "enable_sink": False,
            "exact_profile": "q511",
        },
    }


@pytest.mark.parametrize(
    ("component", "loader_name"),
    [
        ("context_fp16_hd256", "load_cake_fmha_context_fp16_hd256_module"),
        ("context_fp8_hd256", "load_cake_fmha_context_fp8_hd256_module"),
    ],
)
def test_cake_fmha_context_hd256_route_loads_authenticated_chain(
    monkeypatch, component, loader_name
) -> None:
    sentinel = object()
    observed = {}

    def load(*args):
        observed["args"] = args
        return sentinel

    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    monkeypatch.setattr(cake_api, loader_name, load)
    route = cake_api.CakeFmhaContextRoute(
        target="sm103a",
        component=component,
        num_m_blocks=2,
        num_q_heads=10,
        num_kv_heads=2,
        pack_g=1,
        page_size=32,
        l2_swizzle=1,
        is_causal=component == "context_fp8_hd256",
        return_lse=False,
        enable_sink=False,
    )
    assert cake_api.get_cake_fmha_context_module(torch.device("cpu"), route) is sentinel
    assert observed == {"args": ("sm103a", 2, 10, 2, 32)}


def test_cake_fmha_context_nvfp4_route_loads_authenticated_chain(
    monkeypatch,
) -> None:
    sentinel = object()
    observed = {}

    def load(*args):
        observed["args"] = args
        return sentinel

    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    monkeypatch.setattr(cake_api, "load_cake_fmha_context_nvfp4_module", load)
    route = cake_api.CakeFmhaContextRoute(
        target="sm103a",
        component="context_nvfp4",
        num_m_blocks=2,
        num_q_heads=32,
        num_kv_heads=4,
        pack_g=8,
        page_size=16,
        l2_swizzle=8,
        is_causal=True,
        return_lse=False,
        enable_sink=False,
    )
    assert cake_api.get_cake_fmha_context_module(torch.device("cpu"), route) is sentinel
    assert observed == {"args": ("sm103a", 2, 32, 4, 8, 16, 8)}


def test_cake_fmha_context_nvfp4_load_failure_fails_closed(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    monkeypatch.setattr(
        cake_api,
        "load_cake_fmha_context_nvfp4_module",
        lambda *args: (_ for _ in ()).throw(OSError("JIT unavailable")),
    )
    monkeypatch.setattr(
        cake_api, "load_cake_fmha_compat_module", lambda target: sentinel
    )
    route = cake_api.CakeFmhaContextRoute(
        target="sm103a",
        component="context_nvfp4",
        num_m_blocks=2,
        num_q_heads=32,
        num_kv_heads=4,
        pack_g=8,
        page_size=16,
        l2_swizzle=8,
        is_causal=True,
        return_lse=False,
        enable_sink=False,
    )
    with pytest.warns(RuntimeWarning, match="failed closed to compat_v1"):
        assert (
            cake_api.get_cake_fmha_context_module(torch.device("cpu"), route)
            is sentinel
        )


def test_decode_route_miss_fails_closed_to_compat(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    monkeypatch.setattr(cake_api, "load_cake_fmha_compat_module", lambda target: sentinel)
    assert cake_api.get_cake_fmha_decode_module(torch.device("cpu"), None) is sentinel


def test_context_route_miss_fails_closed_to_compat(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    monkeypatch.setattr(cake_api, "load_cake_fmha_compat_module", lambda target: sentinel)
    assert cake_api.get_cake_fmha_context_module(torch.device("cpu"), None) is sentinel


@pytest.mark.parametrize(
    ("skip_softmax_threshold_scale_factor", "expected_ffi_value"),
    [(1e-30, None), (1e-4, 1e-4)],
)
def test_cake_public_decode_route_miss_canonicalizes_only_pinned_noop_skip(
    monkeypatch,
    skip_softmax_threshold_scale_factor,
    expected_ffi_value,
) -> None:
    observed = {}

    def run(*args):
        observed["args"] = args

    compat_module = SimpleNamespace(cake_paged_attention_decode=run)
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    monkeypatch.setattr(
        cake_api, "load_cake_fmha_compat_module", lambda target: compat_module
    )
    monkeypatch.setattr(decode, "get_device_sm_count", lambda device: 1)

    query = torch.empty((2, 4, 256), dtype=torch.bfloat16)
    key = torch.empty((4, 2, 16, 256), dtype=torch.bfloat16)
    result = cake_api.cake_batch_decode_with_kv_cache(
        query,
        (key, torch.empty_like(key)),
        torch.empty(4096, dtype=torch.uint8),
        torch.zeros((2, 2), dtype=torch.int32),
        torch.tensor([16, 16], dtype=torch.int32),
        16,
        bmm1_scale=0.125,
        bmm2_scale=1.0,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
    )

    assert result.shape == query.shape
    assert observed["args"][26] == expected_ffi_value


@pytest.mark.parametrize(
    ("skip_softmax_threshold_scale_factor", "expected_ffi_value"),
    [(1e-30, None), (1e-4, 1e-4)],
)
def test_cake_public_context_route_miss_canonicalizes_only_pinned_noop_skip(
    monkeypatch,
    skip_softmax_threshold_scale_factor,
    expected_ffi_value,
) -> None:
    observed = {}

    def run(*args):
        observed["args"] = args

    compat_module = SimpleNamespace(cake_paged_attention_context=run)
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    monkeypatch.setattr(
        cake_api, "load_cake_fmha_compat_module", lambda target: compat_module
    )
    monkeypatch.setattr(prefill, "get_device_sm_count", lambda device: 1)

    query = torch.empty((4, 4, 256), dtype=torch.bfloat16)
    key = torch.empty((4, 2, 16, 256), dtype=torch.bfloat16)
    result = cake_api.cake_batch_context_with_kv_cache(
        query,
        (key, torch.empty_like(key)),
        torch.empty(4096, dtype=torch.uint8),
        torch.zeros((2, 2), dtype=torch.int32),
        torch.tensor([16, 16], dtype=torch.int32),
        max_q_len=2,
        max_kv_len=16,
        bmm1_scale=0.125,
        bmm2_scale=1.0,
        batch_size=2,
        cum_seq_lens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
        cum_seq_lens_kv=torch.tensor([0, 16, 32], dtype=torch.int32),
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
    )

    assert result.shape == query.shape
    assert observed["args"][26] == expected_ffi_value


def test_cake_public_nvfp4_loader_failure_materializes_compat_scale_abi(
    monkeypatch,
) -> None:
    observed = {}

    def run(*args):
        observed["args"] = args

    compat_module = SimpleNamespace(cake_paged_attention_decode=run)
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    monkeypatch.setattr(
        cake_api,
        "load_cake_fmha_decode_quant_nvfp4_module",
        lambda *args: (_ for _ in ()).throw(RuntimeError("JIT unavailable")),
    )
    monkeypatch.setattr(
        cake_api, "load_cake_fmha_compat_module", lambda target: compat_module
    )
    monkeypatch.setattr(decode, "get_compute_capability", lambda device: (10, 3))
    monkeypatch.setattr(decode, "get_device_sm_count", lambda device: 1)

    query = torch.empty((2, 4, 128), dtype=torch.float8_e4m3fn)
    key = torch.empty((4, 2, 16, 64), dtype=torch.uint8)
    scale = torch.empty((4, 2, 16, 8), dtype=torch.float8_e4m3fn)
    with pytest.warns(RuntimeWarning, match="failed closed to compat_v1"):
        result = cake_api.cake_batch_decode_with_kv_cache(
            query,
            (key, torch.empty_like(key)),
            torch.empty(1 << 20, dtype=torch.uint8),
            torch.zeros((2, 2), dtype=torch.int32),
            torch.tensor([16, 16], dtype=torch.int32),
            16,
            bmm1_scale=torch.tensor(0.125, dtype=torch.float32),
            bmm2_scale=torch.tensor(0.75, dtype=torch.float32),
            kv_cache_sf=(scale, torch.empty_like(scale)),
        )

    assert result.shape == query.shape
    assert isinstance(observed["args"][11], float)
    assert isinstance(observed["args"][12], float)
    assert observed["args"][11] == pytest.approx(0.125)
    assert observed["args"][12] == pytest.approx(0.75)


def test_cake_fmha_context_route_is_optimized_only_on_exact_bf16_domain(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    query = torch.empty((14, 4, 128), dtype=torch.bfloat16)
    key = torch.empty((4, 2, 16, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    out = torch.empty_like(query)
    lse = torch.empty((14, 4), dtype=torch.float32)
    kwargs = dict(
        query=query,
        key_cache=key,
        value_cache=value,
        out=out,
        block_tables=torch.zeros((2, 2), dtype=torch.int32),
        seq_lens=torch.tensor([31, 29], dtype=torch.int32),
        batch_size=2,
        max_q_len=7,
        max_kv_len=31,
        window_left=-1,
        bmm1_scale=0.125,
        bmm2_scale=1.0,
        sinks=None,
        uses_shared_paged_kv_idx=True,
        cum_seq_lens_q=torch.tensor([0, 7, 14], dtype=torch.int32),
        cum_seq_lens_kv=torch.tensor([0, 31, 60], dtype=torch.int32),
        key_block_scales=None,
        value_block_scales=None,
        skip_softmax_threshold_scale_factor=None,
        is_causal=True,
        lse=lse,
    )
    route = cake_api.select_cake_fmha_context_route(query.device, **kwargs)
    assert route == cake_api.CakeFmhaContextRoute(
        target="sm103a",
        component="context_bf16",
        num_m_blocks=1,
        num_q_heads=4,
        num_kv_heads=2,
        pack_g=2,
        page_size=16,
        l2_swizzle=1,
        is_causal=True,
        return_lse=True,
        enable_sink=False,
    )
    assert cake_api.cake_fmha_route_is_optimized(route)
    assert (
        cake_api.select_cake_fmha_context_route(
            query.device,
            **{
                **kwargs,
                "bmm1_scale": torch.ones(1, dtype=torch.float32),
            },
        )
        is None
    )
    assert cake_api.select_cake_fmha_context_route(
        query.device,
        **{
            **kwargs,
            "skip_softmax_threshold_scale_factor": 1e-30,
        },
    ) == route
    assert (
        cake_api.select_cake_fmha_context_route(
            query.device,
            **{
                **kwargs,
                "skip_softmax_threshold_scale_factor": 1e-4,
            },
        )
        is None
    )

    fp8_query = torch.empty((64, 32, 128), dtype=torch.float8_e4m3fn)
    fp8_key = torch.empty((4, 4, 64, 128), dtype=torch.float8_e4m3fn)
    fp8_route = cake_api.select_cake_fmha_context_route(
        fp8_query.device,
        query=fp8_query,
        key_cache=fp8_key,
        value_cache=torch.empty_like(fp8_key),
        out=torch.empty_like(fp8_query),
        block_tables=torch.zeros((2, 2, 1), dtype=torch.int32),
        seq_lens=torch.tensor([64, 64], dtype=torch.int32),
        batch_size=2,
        max_q_len=32,
        max_kv_len=64,
        window_left=-1,
        bmm1_scale=0.03125,
        bmm2_scale=0.75,
        sinks=None,
        uses_shared_paged_kv_idx=False,
        cum_seq_lens_q=torch.tensor([0, 32, 64], dtype=torch.int32),
        cum_seq_lens_kv=torch.tensor([0, 64, 128], dtype=torch.int32),
        key_block_scales=None,
        value_block_scales=None,
        skip_softmax_threshold_scale_factor=None,
        is_causal=True,
        lse=None,
    )
    assert fp8_route == cake_api.CakeFmhaContextRoute(
        target="sm103a",
        component="context_fp8",
        num_m_blocks=1,
        num_q_heads=32,
        num_kv_heads=4,
        pack_g=8,
        page_size=64,
        l2_swizzle=8,
        is_causal=True,
        return_lse=False,
        enable_sink=False,
    )
    assert cake_api.select_cake_fmha_context_route(
        fp8_query.device,
        query=fp8_query,
        key_cache=fp8_key,
        value_cache=torch.empty_like(fp8_key),
        out=torch.empty_like(fp8_query),
        block_tables=torch.zeros((2, 2, 1), dtype=torch.int32),
        seq_lens=torch.tensor([64, 64], dtype=torch.int32),
        batch_size=2,
        max_q_len=32,
        max_kv_len=64,
        window_left=-1,
        bmm1_scale=torch.tensor(0.03125, dtype=torch.float32),
        bmm2_scale=torch.tensor(0.75, dtype=torch.float32),
        sinks=None,
        uses_shared_paged_kv_idx=False,
        cum_seq_lens_q=torch.tensor([0, 32, 64], dtype=torch.int32),
        cum_seq_lens_kv=torch.tensor([0, 64, 128], dtype=torch.int32),
        key_block_scales=None,
        value_block_scales=None,
        skip_softmax_threshold_scale_factor=1e-30,
        is_causal=True,
        lse=None,
    ) == fp8_route
    assert cake_api.cake_fmha_route_is_optimized(fp8_route)


@pytest.mark.parametrize(
    (
        "profile",
        "q_len",
        "kv_len",
        "page_size",
        "uses_shared",
        "num_m_blocks",
        "l2_swizzle",
    ),
    [
        ("q511", 511, 2047, 32, True, 11, 1),
        ("q257", 257, 1024, 1024, False, 6, 8),
    ],
)
def test_cake_context_exact_mask_profile_requires_uniform_runtime_lengths(
    monkeypatch,
    profile,
    q_len,
    kv_len,
    page_size,
    uses_shared,
    num_m_blocks,
    l2_swizzle,
) -> None:
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    batch_size = 4
    query = torch.empty((batch_size * q_len, 10, 128), dtype=torch.bfloat16)
    pages_per_sequence = (kv_len + page_size - 1) // page_size
    key = torch.empty(
        (batch_size * 2 * pages_per_sequence, 2, page_size, 128),
        dtype=torch.bfloat16,
    )
    block_tables = (
        torch.zeros((batch_size, pages_per_sequence), dtype=torch.int32)
        if uses_shared
        else torch.zeros((batch_size, 2, pages_per_sequence), dtype=torch.int32)
    )
    kwargs = dict(
        query=query,
        key_cache=key,
        value_cache=torch.empty_like(key),
        out=torch.empty_like(query),
        block_tables=block_tables,
        seq_lens=torch.full((batch_size,), kv_len, dtype=torch.int32),
        batch_size=batch_size,
        max_q_len=q_len,
        max_kv_len=kv_len,
        window_left=-1,
        bmm1_scale=0.125,
        bmm2_scale=1.0,
        sinks=None,
        uses_shared_paged_kv_idx=uses_shared,
        cum_seq_lens_q=torch.arange(
            0, (batch_size + 1) * q_len, q_len, dtype=torch.int32
        ),
        cum_seq_lens_kv=torch.arange(
            0, (batch_size + 1) * kv_len, kv_len, dtype=torch.int32
        ),
        key_block_scales=None,
        value_block_scales=None,
        skip_softmax_threshold_scale_factor=None,
        is_causal=True,
        lse=None,
        kv_layout="HND",
    )
    route = cake_api.select_cake_fmha_context_route(query.device, **kwargs)
    assert route == cake_api.CakeFmhaContextRoute(
        target="sm103a",
        component="context_bf16",
        num_m_blocks=num_m_blocks,
        num_q_heads=10,
        num_kv_heads=2,
        pack_g=5,
        page_size=page_size,
        l2_swizzle=l2_swizzle,
        is_causal=True,
        return_lse=False,
        enable_sink=False,
        exact_profile=profile,
    )

    nonuniform = kwargs["seq_lens"].clone()
    nonuniform[-1] -= 1
    generic = cake_api.select_cake_fmha_context_route(
        query.device, **{**kwargs, "seq_lens": nonuniform}
    )
    assert generic is not None
    assert generic.exact_profile is None


def test_cake_decode_public_entrypoint_forces_cake_backend(monkeypatch) -> None:
    observed = {}

    def fake_decode(*args, **kwargs):
        observed.update(kwargs)
        return "decode-result"

    monkeypatch.setattr(decode, "trtllm_batch_decode_with_kv_cache", fake_decode)
    assert cake_api.cake_batch_decode_with_kv_cache("query") == "decode-result"
    assert observed["backend"] == "cake"


def test_cake_context_public_entrypoint_forces_cake_backend(monkeypatch) -> None:
    observed = {}

    def fake_context(*args, **kwargs):
        observed.update(kwargs)
        return "context-result"

    monkeypatch.setattr(prefill, "trtllm_batch_context_with_kv_cache", fake_context)
    assert cake_api.cake_batch_context_with_kv_cache("query") == "context-result"
    assert observed["backend"] == "cake"


def test_cake_public_symbols_are_top_level() -> None:
    assert flashinfer.cake_batch_decode_with_kv_cache is (
        cake_api.cake_batch_decode_with_kv_cache
    )
    assert flashinfer.cake_batch_context_with_kv_cache is (
        cake_api.cake_batch_context_with_kv_cache
    )
    assert flashinfer.cake_fmha_manifest is cake_api.cake_fmha_manifest


def test_cake_fmha_aot_registers_each_exact_blackwell_target(monkeypatch) -> None:
    from flashinfer import aot

    monkeypatch.setattr(
        aot, "gen_spdlog_module", lambda: SimpleNamespace(name="spdlog")
    )
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(
        aot, "gen_cudnn_fmha_module", lambda: SimpleNamespace(name="cudnn")
    )
    monkeypatch.setattr(
        aot,
        "gen_cake_fmha_compat_module",
        lambda target: SimpleNamespace(name=f"cake-{target}"),
    )

    specs = aot.gen_all_modules(
        [torch.bfloat16],
        [torch.float8_e4m3fn],
        [(128, 128)],
        [(128, 128)],
        [False],
        [False],
        {"sm100a_exact": True, "sm103a_exact": True},
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    assert {spec.name for spec in specs} == {
        "spdlog",
        "cudnn",
        "cake-sm100a",
        "cake-sm103a",
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cake_decode_bf16_matches_flashinfer_reference() -> None:
    from tests.attention.test_trtllm_gen_attention_decode import (
        _test_trtllm_batch_decode,
    )

    _test_trtllm_batch_decode(
        backend="cake",
        kv_layout="HND",
        batch_size=2,
        q_len_per_req=1,
        page_size=16,
        num_kv_heads=2,
        head_grp_size=2,
        window_left=-1,
        q_dtype="bf16",
        o_dtype="bf16",
        kv_dtype="bf16",
        enable_pdl=False,
        enable_sink=False,
        max_in_kv_len=31,
        head_dim=128,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cake_decode_fp16_nhd_matches_flashinfer_reference() -> None:
    from tests.attention.test_trtllm_gen_attention_decode import (
        _test_trtllm_batch_decode,
    )

    _test_trtllm_batch_decode(
        backend="cake",
        kv_layout="NHD",
        batch_size=2,
        q_len_per_req=1,
        page_size=32,
        num_kv_heads=2,
        head_grp_size=2,
        window_left=-1,
        q_dtype="fp16",
        o_dtype="fp16",
        kv_dtype="fp16",
        enable_pdl=False,
        enable_sink=False,
        max_in_kv_len=127,
        head_dim=128,
        uses_shared_paged_kv_idx=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cake_decode_fp16_hd512_matches_flashinfer_reference() -> None:
    from tests.attention.test_trtllm_gen_attention_decode import (
        _test_trtllm_batch_decode,
    )

    _test_trtllm_batch_decode(
        backend="cake",
        kv_layout="HND",
        batch_size=2,
        q_len_per_req=1,
        page_size=64,
        num_kv_heads=2,
        head_grp_size=2,
        window_left=96,
        q_dtype="fp16",
        o_dtype="fp16",
        kv_dtype="fp16",
        enable_pdl=False,
        enable_sink=False,
        max_in_kv_len=255,
        head_dim=512,
        uses_shared_paged_kv_idx=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cake_context_bf16_separate_tables_matches_reference(monkeypatch) -> None:
    from tests.attention.test_trtllm_gen_attention_prefill import (
        _test_trtllm_batch_prefill,
    )

    original = prefill.trtllm_batch_context_with_kv_cache

    def cake_context(*args, **kwargs):
        return original(*args, backend="cake", **kwargs)

    monkeypatch.setattr(prefill, "trtllm_batch_context_with_kv_cache", cake_context)
    _test_trtllm_batch_prefill(
        kv_layout="NHD",
        batch_size=2,
        page_size=32,
        num_kv_heads=2,
        head_grp_size=2,
        causal=False,
        window_left=-1,
        q_dtype="bf16",
        o_dtype="bf16",
        kv_dtype="bf16",
        enable_pdl=False,
        enable_sink=False,
        max_q_len=7,
        max_kv_len=31,
        device_scale=False,
        head_dim=128,
        uses_shared_paged_kv_idx=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cake_context_fp8_nhd_device_scale_skip_matches_reference(monkeypatch) -> None:
    from tests.attention.test_trtllm_gen_attention_prefill import (
        _test_trtllm_batch_prefill,
    )

    original = prefill.trtllm_batch_context_with_kv_cache

    def cake_context(*args, **kwargs):
        return original(*args, backend="cake", **kwargs)

    monkeypatch.setattr(prefill, "trtllm_batch_context_with_kv_cache", cake_context)
    _test_trtllm_batch_prefill(
        kv_layout="NHD",
        batch_size=2,
        page_size=64,
        num_kv_heads=4,
        head_grp_size=8,
        causal=True,
        window_left=-1,
        q_dtype="fp8",
        o_dtype="fp8",
        kv_dtype="fp8",
        enable_pdl=False,
        enable_sink=False,
        max_q_len=32,
        max_kv_len=159,
        device_scale=True,
        head_dim=128,
        uses_shared_paged_kv_idx=True,
        skips_softmax=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cake_base_decode_cuda_graph_capture_replay() -> None:
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Cake FMHA requires SM100 or SM103")

    from flashinfer.utils import (
        get_device_sm_count,
        get_trtllm_gen_multi_ctas_kv_counter_bytes,
    )

    device = torch.device("cuda")
    batch_size, num_q_heads, num_kv_heads = 2, 4, 2
    query = torch.randn(
        (batch_size, num_q_heads, 128), dtype=torch.bfloat16, device=device
    )
    key = torch.randn((4, num_kv_heads, 16, 128), dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([31, 29], dtype=torch.int32, device=device)
    workspace = torch.empty(1 << 20, dtype=torch.uint8, device=device)
    out = torch.empty_like(query)
    counter = torch.zeros(
        get_trtllm_gen_multi_ctas_kv_counter_bytes(
            batch_size, num_q_heads, get_device_sm_count(device)
        ),
        dtype=torch.uint8,
        device=device,
    )

    def run():
        return cake_api.cake_batch_decode_with_kv_cache(
            query,
            (key, value),
            workspace,
            block_tables,
            seq_lens,
            31,
            out=out,
            multi_ctas_kv_counter_buffer=counter,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    expected = out.clone()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
