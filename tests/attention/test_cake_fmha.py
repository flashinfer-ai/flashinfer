"""CPU-side packaging and public-routing tests for Cake FMHA."""

from __future__ import annotations

import dataclasses
import math

from pathlib import Path
from types import SimpleNamespace

import flashinfer
import flashinfer.cake_fmha as cake_api
import flashinfer.decode as decode
import flashinfer.prefill as prefill
import pytest
import torch
from flashinfer.jit.cake_fmha import (
    cake_fmha_smallm_component_name,
    gen_cake_fmha_decode_native_bf16_hd256_smallm_module,
    get_cake_fmha_decode_native_bf16_hd256_smallm_uri,
    CAKE_FMHA_BALANCED_FP8_Q_DTYPES,
    CAKE_FMHA_BALANCED_HD256_PAGE_SIZES,
    CAKE_FMHA_FLASHINFER_MATRIX_REVISION,
    cake_fmha_balanced_fp8_component_name,
    cake_fmha_balanced_hd256_component_name,
    gen_cake_fmha_compat_module,
    gen_cake_fmha_context_bf16_module,
    gen_cake_fmha_context_fp16_hd256_module,
    gen_cake_fmha_context_fp8_module,
    gen_cake_fmha_context_fp8_hd256_module,
    gen_cake_fmha_context_nvfp4_module,
    gen_cake_fmha_decode_balanced_bf16_module,
    gen_cake_fmha_decode_balanced_fp8_module,
    gen_cake_fmha_decode_balanced_hd256_module,
    gen_cake_fmha_decode_balanced_hd64_module,
    gen_cake_fmha_decode_balanced_module,
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
    get_cake_fmha_decode_balanced_bf16_uri,
    get_cake_fmha_decode_balanced_fp8_uri,
    get_cake_fmha_decode_balanced_hd256_uri,
    get_cake_fmha_decode_balanced_hd64_uri,
    get_cake_fmha_decode_balanced_uri,
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
    select_case,
)


def test_cake_fmha_registry_is_complete() -> None:
    manifest = get_cake_fmha_manifest()
    assert manifest["product"] == "cake_fmha"
    assert manifest["flashinfer_matrix_revision"] == (
        CAKE_FMHA_FLASHINFER_MATRIX_REVISION
    )
    assert manifest["capability"]["complete"] is True
    assert manifest["capability"]["cake_coverage_ratio"] == 1.0
    assert manifest["capability"]["upstream_valid_cases"] == 57_280
    assert manifest["capability"]["cake_covered_cases"] == 57_280
    assert manifest["capability"]["route_counts"]["cake_fmha_compat_v1"] == 55_374
    # 29 pinned product probes plus the 19 correctness rows the balanced routes
    # (bf16, fp16, fp8, bf16q, fp16q, bf16 hd64, bf16 hd256) add to the contract.
    assert len(manifest["route_probes"]) == 48
    assert {probe["label"] for probe in manifest["route_probes"]} >= {
        "correctness_compat_decode_fp8_nhd_separate_group5",
        "correctness_compat_decode_fp8_hnd_shared_group8_partial",
        "correctness_decode_fp8_hnd_shared_group8_full_blocks",
    }
    assert manifest["components"]["compat_v1"]["launch_binding"] == (
        "cake_fmha_launch_compat_v1"
    )


def test_cake_fmha_public_manifest_is_defensive_copy() -> None:
    public_manifest = cake_api.cake_fmha_manifest()
    public_manifest["product"] = "mutated"
    assert cake_api.cake_fmha_manifest()["product"] == "cake_fmha"


def test_cake_fmha_registry_accounts_for_manifest_routes_and_components() -> None:
    manifest = cake_api.cake_fmha_manifest()
    manifest_route_counts = manifest["capability"]["route_counts"]
    manifest_optimized_routes = set(manifest_route_counts) - {"cake_fmha_compat_v1"}
    assert manifest_optimized_routes <= set(cake_api._PRODUCT_ROUTE_COMPONENTS)
    assert cake_api._manifest_optimized_route_accounting() == (1_906, 1_906)
    assert cake_api._manifest_authenticated_route_accounting() == (1_906, 1_906)
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
    assert routed_components | {"compat_v1"} == cake_api._AUTHENTICATED_JIT_COMPONENTS


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
    assert report.optimized_cases == 1_906
    assert report.compat_cases == 55_374
    assert report.route_counts == dict(
        sorted(manifest["capability"]["route_counts"].items())
    )
    assert report.digest == (
        "bac35578d289126db5e078275890f6f87c5237895ae7750ca419b06be13feb9a"
    )


def test_cake_fmha_capability_replay_covers_q257_exact_context_profile(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    case = {
        "mode": "context",
        "batch_size": 4,
        "q_len": 257,
        "kv_len": 1024,
        "page_size": 1024,
        "num_kv_heads": 2,
        "num_qo_heads": 10,
        "window_left": -1,
        "q_dtype": "bf16",
        "kv_dtype": "bf16",
        "o_dtype": "bf16",
        "enable_pdl": None,
        "enable_sink": False,
        "head_dim": 128,
        "non_contiguous_query": False,
        "skip_softmax": False,
        "uses_shared_paged_kv_idx": False,
        "kv_layout": "HND",
        "causal": True,
    }

    assert select_case(case) == "ctx_bf16_hnd_hd128_hgpack_03df_v2"


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
        "default.cu",
        "cake_fmha_decode_native_bf16_binding.cu",
        "cake_fmha_decode_native_bf16_jit_binding.cu",
    }
    assert "-DBATCH_SIZE=2" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_USE_SCALE_PTR=1" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_RETAIN_KV_L2=1" in spec.extra_cuda_cflags


@pytest.mark.parametrize(
    ("component", "macros", "generic_count", "exact_key", "prefix", "sources"),
    (
        (
            "decode_native_bf16",
            ("HAS_SINK", "HAS_WINDOW", "RETAIN_KV_L2", "USE_SCALE_PTR"),
            11,
            "BATCH_SIZE",
            "CAKE_FMHA_",
            {
                "sm_100a": "cuda/decode_native_bf16/default.cu",
                "sm_103a": "cuda/decode_native_bf16/default.cu",
            },
        ),
        (
            "decode_native_fp16_nhd",
            ("HAS_SINK", "HAS_WINDOW", "RETAIN_KV_L2", "USE_SCALE_PTR"),
            16,
            "BATCH_SIZE",
            "CAKE_FMHA_",
            {
                "sm_100a": "cuda/decode_native_fp16_nhd/default.cu",
                "sm_103a": "cuda/decode_native_fp16_nhd/default.cu",
            },
        ),
        (
            "decode_native_fp16_hd512",
            ("HAS_WINDOW", "RETAIN_KV_L2", "USE_SCALE_PTR"),
            8,
            "BATCH_SIZE",
            "CAKE_FMHA_",
            {
                "sm_100a": "cuda/decode_native_fp16_hd512/default.cu",
                "sm_103a": "cuda/decode_native_fp16_hd512/default.cu",
            },
        ),
        (
            "context_fp8",
            ("ENABLE_SINK", "IS_CAUSAL", "RETURN_LSE"),
            6,
            "NUM_M_BLOCKS",
            "CAKE_FMHA_CONTEXT_",
            {
                "sm_100a": "cuda/context_fp8/default.cu",
                "sm_103a": "cuda/context_fp8/default.cu",
            },
        ),
        (
            "context_bf16",
            ("ENABLE_SINK", "IS_CAUSAL", "RETURN_LSE"),
            6,
            "NUM_M_BLOCKS",
            "CAKE_FMHA_CONTEXT_",
            {
                "sm_100a": "cuda/context_bf16/default.cu",
                "sm_103a": "cuda/context_bf16/default.cu",
            },
        ),
    ),
)
def test_cake_fmha_generic_members_share_one_folded_source(
    component, macros, generic_count, exact_key, prefix, sources
) -> None:
    """The generic members compile one source whose flags come from the loader defines."""

    members = get_cake_fmha_manifest()["components"][component]["source_family"]
    generic_members = [m for m in members if exact_key not in m["selector"]]
    assert len(generic_members) == generic_count
    for arch, source in sources.items():
        assert {member["sources"][arch] for member in generic_members} == {source}
    for source in sorted(set(sources.values())):
        text = (get_cake_fmha_csrc_dir() / source).read_text(encoding="utf-8")
        for macro in macros:
            assert f"#define {macro} {prefix}{macro}\n" in text
            assert f"#define {macro} 0\n" not in text
            assert f"#define {macro} 1\n" not in text
            assert (
                text.count(f"#if {macro}\n") == text.count(f"#endif  // {macro}\n") > 0
            )
    flag_sets = {
        tuple(member["selector"][macro] for macro in macros)
        for member in generic_members
    }
    assert len(flag_sets) == generic_count
    assert all(set(member["selector"]) == set(macros) for member in generic_members)


@pytest.mark.parametrize(
    ("components", "knobs", "source"),
    (
        (
            ("decode_balanced_bf16", "decode_balanced_fp16"),
            {"CAKE_FMHA_BALANCED_FP16": ()},
            "cuda/decode_balanced_16bit/default.cu",
        ),
        (
            (
                "decode_balanced_bf16_mtp_n32",
                "decode_balanced_bf16_mtp_n64",
                "decode_balanced_fp16_mtp_n32",
                "decode_balanced_fp16_mtp_n64",
            ),
            {"CAKE_FMHA_BALANCED_N_ROWS": (64,), "CAKE_FMHA_BALANCED_FP16": ()},
            "cuda/decode_balanced_mtp/default.cu",
        ),
        (
            (
                "decode_balanced_bf16_hd256_p16",
                "decode_balanced_bf16_hd256_p32",
                "decode_balanced_bf16_hd256_p64",
            ),
            {"CAKE_FMHA_BALANCED_PAGE_SIZE": (32, 64)},
            "cuda/decode_balanced_hd256/default.cu",
        ),
        (
            ("decode_balanced_bf16q", "decode_balanced_fp16q"),
            {"CAKE_FMHA_BALANCED_Q_DTYPE": (2,)},
            "cuda/decode_balanced_q16bit/default.cu",
        ),
        (
            ("decode_balanced_bf16_hd64", "decode_balanced_bf16_hd64_g16"),
            {"CAKE_FMHA_BALANCED_MAX_GROUPS": (16,)},
            "cuda/decode_balanced_hd64/default.cu",
        ),
    ),
)
def test_cake_fmha_balanced_decode_components_share_one_folded_source(
    components, knobs, source
) -> None:
    """One folded source per balanced decode family, selected by the loader defines."""

    manifest_components = get_cake_fmha_manifest()["components"]
    text = (get_cake_fmha_csrc_dir() / source).read_text(encoding="utf-8")
    for component in components:
        (member,) = manifest_components[component]["source_family"]
        assert member["selector"] == {}
        assert member["sources"] == {"sm_100a": source, "sm_103a": source}
        assert f"{manifest_components[component]['public_symbol']}(" in text
    for knob, values in knobs.items():
        assert f"#define {knob}" not in text
        opens = [
            line
            for line in text.splitlines()
            if line == f"#if {knob}"
            or line == f"#if !{knob}"
            or line.startswith((f"#if {knob} == ", f"#if {knob} != "))
        ]
        assert text.count(f"#if {knob}") == len(opens)
        assert text.count(f"#endif  // {knob}\n") == len(opens) > 0
        tested = {int(line.rsplit(" ", 1)[1]) for line in opens if " " in line[4:]}
        assert tested <= set(values)


def test_cake_fmha_context_bf16_generic_members_share_one_arch_source() -> None:
    """Both arches compile one context_bf16 source split only by __CUDA_ARCH__."""

    members = get_cake_fmha_manifest()["components"]["context_bf16"]["source_family"]
    generic_members = [m for m in members if "NUM_M_BLOCKS" not in m["selector"]]
    # The q511 / q257 one-tile programs ship as the generated unit
    # csrc/cake_fmha_context_bf16/, not as registry members.
    assert len(generic_members) == len(members) == 6
    assert {
        member["sources"][arch]
        for member in generic_members
        for arch in ("sm_100a", "sm_103a")
    } == {"cuda/context_bf16/default.cu"}
    csrc_dir = get_cake_fmha_csrc_dir()
    for arch in ("sm_100a", "sm_103a"):
        assert not (csrc_dir / f"cuda/context_bf16/{arch}").exists()
    text = (csrc_dir / "cuda/context_bf16/default.cu").read_text(encoding="utf-8")
    assert "#define __CUDA_ARCH__" not in text
    assert (
        text.count("#if __CUDA_ARCH__ == 1030\n")
        == text.count("#endif  // __CUDA_ARCH__\n")
        > 0
    )
    # The arch guards only split the kernel body; the host-visible preamble
    # (defines, includes, kernel signature) is shared by both arches.
    assert text.index("#if __CUDA_ARCH__ == 1030\n") > text.index("__global__ ")
    assert text.rindex("#endif  // __CUDA_ARCH__\n") < text.index('} // extern "C"')


def test_cake_fmha_smallm_components_share_one_folded_source() -> None:
    """The small-M hd256 components share one source driven by the loader defines."""

    folded = "cuda/decode_native_bf16_hd256_smallm/default.cu"
    csrc_dir = get_cake_fmha_csrc_dir()
    components = get_cake_fmha_manifest()["components"]
    names = [
        cake_fmha_smallm_component_name(n_rows, page_size)
        for n_rows in (32, 64)
        for page_size in (16, 32, 64)
    ]
    for name in names:
        members = components[name]["source_family"]
        assert len(members) == 1
        assert members[0]["selector"] == {}
        assert members[0]["sources"] == {"sm_100a": folded, "sm_103a": folded}
        assert not (csrc_dir / "cuda" / name).exists()
    lines = (csrc_dir / folded).read_text(encoding="utf-8").split("\n")
    for name in names:
        symbol = components[name]["public_symbol"]
        assert sum(line.startswith(f"{symbol}(") for line in lines) == 1
    fixed_defines = (
        "#define N_ROWS ",
        "#define PAGE_SIZE ",
        "#define CAKE_FMHA_SMALLM_",
    )
    assert not [line for line in lines if line.startswith(fixed_defines)]
    knobs = {"CAKE_FMHA_SMALLM_N_ROWS": {64}, "CAKE_FMHA_SMALLM_PAGE_SIZE": {32, 64}}
    for knob, values in knobs.items():
        opens = [line for line in lines if line.startswith(f"#if {knob} ")]
        assert {int(line.rsplit(" ", 1)[1]) for line in opens} == values
        assert len(opens) == lines.count(f"#endif  // {knob}") > 0


@pytest.mark.parametrize(
    ("component", "sources"),
    (
        (
            "decode_quant_bf16q",
            {
                "sm_100a": "cuda/decode_quant_bf16q/default.cu",
                "sm_103a": "cuda/decode_quant_bf16q/default.cu",
            },
        ),
        (
            "decode_quant_nvfp4",
            {
                "sm_100a": "cuda/decode_quant_nvfp4/default.cu",
                "sm_103a": "cuda/decode_quant_nvfp4/default.cu",
            },
        ),
        (
            "decode_quant_fp8",
            {
                "sm_100a": "cuda/decode_quant_fp8/default.cu",
                "sm_103a": "cuda/decode_quant_fp8/default.cu",
            },
        ),
    ),
)
def test_cake_fmha_decode_quant_members_share_one_folded_source(
    component, sources
) -> None:
    """The page-size members compile one source whose PAGE_SIZE is a loader define."""

    members = get_cake_fmha_manifest()["components"][component]["source_family"]
    # Runtime split-plan members, one per page size; decode_quant_fp8 adds the fixed 8 x 16 split-plan
    # members (NUM_SPLITS/BLOCKS_PER_SPLIT/ENABLE_PDL loader defines of the same source).
    runtime = [member for member in members if "NUM_SPLITS" not in member["selector"]]
    fixed = [member for member in members if "NUM_SPLITS" in member["selector"]]
    assert sorted(member["selector"]["PAGE_SIZE"] for member in runtime) == [16, 32]
    assert sorted(member["selector"]["PAGE_SIZE"] for member in fixed) in ([], [16, 32])
    for arch, source in sources.items():
        assert {member["sources"][arch] for member in members} == {source}
    for source in sorted(set(sources.values())):
        text = (get_cake_fmha_csrc_dir() / source).read_text(encoding="utf-8")
        defines = [
            line for line in text.splitlines() if line.startswith("#define PAGE_SIZE ")
        ]
        assert defines == ["#define PAGE_SIZE CAKE_FMHA_PAGE_SIZE"]
        guards = [
            line for line in text.splitlines() if line.startswith("#if PAGE_SIZE")
        ]
        assert guards
        assert len(guards) == text.count("#endif  // PAGE_SIZE\n")


def test_cake_fmha_decode_quant_fp8_members_share_one_arch_source() -> None:
    """Both arches compile one decode_quant_fp8 source split only by __CUDA_ARCH__."""

    folded = "cuda/decode_quant_fp8/default.cu"
    arches = ("sm_100a", "sm_103a")
    component = get_cake_fmha_manifest()["components"]["decode_quant_fp8"]
    members = component["source_family"]
    # Runtime split-plan members, one per page size; decode_quant_fp8 adds the fixed 8 x 16 split-plan
    # members (NUM_SPLITS/BLOCKS_PER_SPLIT/ENABLE_PDL loader defines of the same source).
    runtime = [member for member in members if "NUM_SPLITS" not in member["selector"]]
    fixed = [member for member in members if "NUM_SPLITS" in member["selector"]]
    assert sorted(member["selector"]["PAGE_SIZE"] for member in runtime) == [16, 32]
    assert sorted(member["selector"]["PAGE_SIZE"] for member in fixed) in ([], [16, 32])
    sources = {member["sources"][arch] for member in members for arch in arches}
    assert sources == {folded}
    csrc_dir = get_cake_fmha_csrc_dir()
    for arch in arches:
        assert not (csrc_dir / f"cuda/decode_quant_fp8/{arch}").exists()
    text = (csrc_dir / folded).read_text(encoding="utf-8")
    lines = text.split("\n")
    assert "#define __CUDA_ARCH__" not in text
    openers = ("#if __CUDA_ARCH__ == 1030", "#if __CUDA_ARCH__ != 1030")
    arch_opens = [line for line in lines if line in openers]
    assert len(arch_opens) == lines.count("#endif  // __CUDA_ARCH__") > 0
    # The earlier page-size fold is untouched: one PAGE_SIZE knob define and
    # balanced PAGE_SIZE guards.
    defines = [line for line in lines if line.startswith("#define PAGE_SIZE ")]
    assert defines == ["#define PAGE_SIZE CAKE_FMHA_PAGE_SIZE"]
    page_opens = [line for line in lines if line.startswith("#if PAGE_SIZE")]
    assert len(page_opens) == lines.count("#endif  // PAGE_SIZE") > 0
    # One kernel whose signature and extern "C" block are shared by both arches.
    assert sum(line.startswith("__global__ ") for line in lines) == 1
    assert text.rindex("#endif  // __CUDA_ARCH__\n") < text.index('} // extern "C"')


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
        shape_args = (
            target,
            selector["BATCH_SIZE"],
            selector["Q_LEN"],
            selector["NUM_Q_HEADS"],
            selector["NUM_KV_HEADS"],
        )
        flag_kwargs = dict(
            has_sink=bool(selector["HAS_SINK"]),
            has_window=bool(selector["HAS_WINDOW"]),
            use_scale_ptr=bool(selector["USE_SCALE_PTR"]),
            retain_kv_l2=bool(selector["RETAIN_KV_L2"]),
        )
        spec = gen_cake_fmha_decode_native_bf16_module(*shape_args, **flag_kwargs)
        assert Path(spec.sources[0]) == csrc_dir / member["sources"][manifest_arch]
        launch_override = member.get("launch_override") or {}
        if "by_arch" in launch_override:
            launch_override = {
                **launch_override,
                **launch_override["by_arch"][manifest_arch],
            }
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

        # Exact-shape members never write ``LSE_ptr``.  A caller-provided LSE
        # output is served by the generic member with the same semantic flags
        # (its own module name), or by no optimized route when that member is
        # absent.
        semantic_selector = {
            key: value
            for key, value in selector.items()
            if key not in ("BATCH_SIZE", "Q_LEN", "NUM_Q_HEADS", "NUM_KV_HEADS")
        }
        generic_member = cake_jit._get_component_member(
            "decode_native_bf16", semantic_selector, required=False
        )
        lse_route = dataclasses.replace(route, return_lse=True)
        assert cake_api.cake_fmha_route_is_optimized(lse_route) == (
            generic_member is not None
        )
        if generic_member is not None:
            lse_spec = gen_cake_fmha_decode_native_bf16_module(
                *shape_args, **flag_kwargs, return_lse=True
            )
            assert Path(lse_spec.sources[0]) == (
                csrc_dir / generic_member["sources"][manifest_arch]
            )
            assert Path(lse_spec.sources[1]) == csrc_dir / component["binding_source"]
            assert lse_spec.name == get_cake_fmha_decode_native_bf16_uri(
                *shape_args, **flag_kwargs, return_lse=True
            )
            assert lse_spec.name != spec.name
            assert lse_spec.name.replace("_lse_", "_", 1) == spec.name
            assert lse_spec.extra_cuda_cflags == spec.extra_cuda_cflags
        else:
            assert selector["BATCH_SIZE"] == 256
            with pytest.raises(RuntimeError, match="absent from the registry"):
                gen_cake_fmha_decode_native_bf16_module(
                    *shape_args, **flag_kwargs, return_lse=True
                )

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


@pytest.mark.parametrize(
    ("target", "manifest_arch", "shared_memory_bytes"),
    (("sm100a", "sm_100a", 178176), ("sm103a", "sm_103a", 210944)),
)
def test_cake_fmha_decode_native_bf16_exact_sink_grid_matches_selector(
    monkeypatch, target, manifest_arch, shared_memory_bytes
) -> None:
    import flashinfer.jit.core as jit_core
    from flashinfer.jit import cake_fmha as cake_jit

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    manifest = get_cake_fmha_manifest()
    component = manifest["components"]["decode_native_bf16"]
    sink_members = [
        member
        for member in component["source_family"]
        if "BATCH_SIZE" in member["selector"]
        and member["selector"].get("HAS_SINK") == 1
    ]
    assert len(sink_members) == 1
    sink_member = sink_members[0]
    selector = sink_member["selector"]
    assert {
        key: selector[key]
        for key in (
            "BATCH_SIZE",
            "Q_LEN",
            "NUM_Q_HEADS",
            "NUM_KV_HEADS",
            "HAS_SINK",
            "HAS_WINDOW",
            "USE_SCALE_PTR",
        )
    } == {
        "BATCH_SIZE": 256,
        "Q_LEN": 1,
        "NUM_Q_HEADS": 32,
        "NUM_KV_HEADS": 4,
        "HAS_SINK": 1,
        "HAS_WINDOW": 0,
        "USE_SCALE_PTR": 0,
    }

    sink_binding = "bindings/cake_fmha_decode_native_bf16_sink_peer_clc_binding.cu"
    launch_override = sink_member["launch_override"]
    arch_override = launch_override["by_arch"][manifest_arch]
    # Both arches name one host binding; the arch's dynamic shared memory
    # reaches it through the loader define.
    assert arch_override["binding_source"] == sink_binding
    assert arch_override["dynamic_shared_memory_bytes"] == shared_memory_bytes
    body, binding = cake_jit._get_component_launch_sources(
        "decode_native_bf16", target, selector
    )
    assert body == get_cake_fmha_csrc_dir() / sink_member["sources"][manifest_arch]
    assert binding == get_cake_fmha_csrc_dir() / sink_binding
    spec = gen_cake_fmha_decode_native_bf16_module(
        target,
        selector["BATCH_SIZE"],
        selector["Q_LEN"],
        selector["NUM_Q_HEADS"],
        selector["NUM_KV_HEADS"],
        has_sink=True,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=bool(selector["RETAIN_KV_L2"]),
    )
    assert Path(spec.sources[1]) == binding
    assert (
        f"-DCAKE_FMHA_SINK_PEER_CLC_DYNAMIC_SMEM_BYTES={shared_memory_bytes}"
        in spec.extra_cuda_cflags
    )
    assert launch_override["grid"] == ["Q_LEN", "NUM_KV_HEADS", "BATCH_SIZE"]
    assert launch_override["use_pdl"] is True

    adapter = (
        get_cake_fmha_csrc_dir() / "jit/cake_fmha_decode_native_bf16_jit_binding.cu"
    ).read_text(encoding="utf-8")
    exact_guard = (
        "#if BATCH_SIZE == 256 && Q_LEN == 1 && NUM_Q_HEADS == 32 && "
        "NUM_KV_HEADS == 4 && \\\n"
        "    CAKE_FMHA_HAS_SINK == 1 && CAKE_FMHA_HAS_WINDOW == 0 && "
        "CAKE_FMHA_USE_SCALE_PTR == 0"
    )
    assert adapter.count(exact_guard) == 1
    exact_branch = adapter.split(exact_guard, 1)[1].split("#else", 1)[0]
    assert "grid_x = Q_LEN;" in exact_branch
    assert "grid_y = NUM_KV_HEADS;" in exact_branch
    assert "grid_z = BATCH_SIZE;" in exact_branch


def test_cake_fmha_exact_sink_member_shares_one_host_binding(monkeypatch) -> None:
    """Both arches compile one sink binding; the loader define carries the smem."""

    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    macro = "CAKE_FMHA_SINK_PEER_CLC_DYNAMIC_SMEM_BYTES"
    component = get_cake_fmha_manifest()["components"]["decode_native_bf16"]
    sink_members = [
        member
        for member in component["source_family"]
        if "BATCH_SIZE" in member["selector"]
        and member["selector"].get("HAS_SINK") == 1
    ]
    (sink_member,) = sink_members
    launch_override = sink_member["launch_override"]
    shared_binding = "bindings/cake_fmha_decode_native_bf16_sink_peer_clc_binding.cu"
    # Both per-arch records name one binding; the registry keeps the per-arch
    # dynamic shared memory (the manifest schema is unchanged).
    assert launch_override["by_arch"] == {
        "sm_100a": {
            "binding_source": shared_binding,
            "dynamic_shared_memory_bytes": 178176,
        },
        "sm_103a": {
            "binding_source": shared_binding,
            "dynamic_shared_memory_bytes": 210944,
        },
    }
    by_arch_members = [
        member
        for member in component["source_family"]
        if "by_arch" in (member.get("launch_override") or {})
    ]
    assert by_arch_members == [sink_member]

    csrc_dir = get_cake_fmha_csrc_dir()
    for arch in ("sm_100a", "sm_103a"):
        old_binding = f"cake_fmha_decode_native_bf16_sink_peer_clc_{arch}_binding.cu"
        assert not (csrc_dir / "bindings" / old_binding).exists()
    text = (csrc_dir / shared_binding).read_text(encoding="utf-8")
    assert f"#ifndef {macro}\n#error " in text
    assert text.count(f"        {macro});\n") == 1
    assert text.count(f"    config.dynamicSmemBytes = {macro};\n") == 1
    assert "178176" not in text
    assert "210944" not in text

    selector = sink_member["selector"]
    for target, arch in (("sm100a", "sm_100a"), ("sm103a", "sm_103a")):
        spec = gen_cake_fmha_decode_native_bf16_module(
            target,
            selector["BATCH_SIZE"],
            selector["Q_LEN"],
            selector["NUM_Q_HEADS"],
            selector["NUM_KV_HEADS"],
            has_sink=True,
            has_window=False,
            use_scale_ptr=False,
            retain_kv_l2=False,
        )
        assert Path(spec.sources[1]) == csrc_dir / shared_binding
        smem = launch_override["by_arch"][arch]["dynamic_shared_memory_bytes"]
        assert spec.extra_cuda_cflags.count(f"-D{macro}={smem}") == 1
        assert sum(macro in flag for flag in spec.extra_cuda_cflags) == 1
    # A member whose launch override is not split by arch keeps its own binding
    # and receives no define.
    b4_spec = gen_cake_fmha_decode_native_bf16_module(
        "sm100a",
        4,
        1,
        32,
        4,
        has_sink=False,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=True,
    )
    assert Path(b4_spec.sources[1]).name == (
        "cake_fmha_decode_native_bf16_b4_exact_cga_binding.cu"
    )
    assert not any(macro in flag for flag in b4_spec.extra_cuda_cflags)


def test_cake_fmha_decode_native_bf16_exact_b4_grid_matches_cga_selector() -> None:
    adapter = (
        get_cake_fmha_csrc_dir() / "jit/cake_fmha_decode_native_bf16_jit_binding.cu"
    ).read_text(encoding="utf-8")
    grid_policy = adapter.split("unsigned int grid_z = 1;", 1)[1].split(
        "cudaError_t status", 1
    )[0]
    exact_guard = (
        "#elif BATCH_SIZE == 4 && Q_LEN == 1 && NUM_Q_HEADS == 32 && "
        "NUM_KV_HEADS == 4 && \\\n"
        "    CAKE_FMHA_HAS_SINK == 0 && CAKE_FMHA_HAS_WINDOW == 0 && "
        "CAKE_FMHA_USE_SCALE_PTR == 0 && \\\n"
        "    CAKE_FMHA_RETAIN_KV_L2 == 1"
    )
    assert adapter.count(exact_guard) == 1
    exact_branch = grid_policy.split(exact_guard, 1)[1].split("#else", 1)[0]
    assert (
        "unsigned int total_tiles = BATCH_SIZE * Q_LEN * NUM_KV_HEADS;" in exact_branch
    )
    assert "unsigned int logical_grid_x =" in exact_branch
    assert "grid_x = 2 * logical_grid_x;" in exact_branch


def test_cake_fmha_decode_native_bf16_other_grids_stay_persistent() -> None:
    adapter = (
        get_cake_fmha_csrc_dir() / "jit/cake_fmha_decode_native_bf16_jit_binding.cu"
    ).read_text(encoding="utf-8")
    grid_policy = adapter.split("unsigned int grid_z = 1;", 1)[1].split(
        "cudaError_t status", 1
    )[0]
    persistent_branch = grid_policy.split("#else", 1)[1].split("#endif", 1)[0]
    assert "unsigned int total_tiles = BATCH_SIZE * Q_LEN * NUM_KV_HEADS;" in (
        persistent_branch
    )
    assert (
        "grid_x = std::min<unsigned int>(static_cast<unsigned int>(sm_count), "
        "total_tiles);"
    ) in persistent_branch
    assert "unsigned int grid_y = 1;" in adapter
    assert "unsigned int grid_z = 1;" in adapter
    assert "grid_x, grid_y, grid_z, stream);" in adapter


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
        "default.cu",
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
        "default.cu",
        "cake_fmha_decode_native_fp16_hd512_binding.cu",
        "cake_fmha_decode_native_fp16_hd512_jit_binding.cu",
    }
    assert "-DCAKE_FMHA_HAS_WINDOW=1" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_USE_SCALE_PTR=1" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_RETAIN_KV_L2=0" in spec.extra_cuda_cflags
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
        "default.cu",
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
        full_blocks=False,
    )
    assert spec.name == get_cake_fmha_decode_quant_fp8_uri(
        "sm103a",
        2,
        1,
        16,
        2,
        32,
        full_blocks=False,
    )
    assert {Path(source).name for source in spec.sources} == {
        "default.cu",
        "cake_fmha_decode_quant_fp8_binding.cu",
        "cake_fmha_decode_quant_fp8_reduce_binding.cu",
        "cake_fmha_decode_quant_fp8_jit_binding.cu",
    }
    assert any(
        str(source).endswith("cuda/decode_quant_fp8/default.cu")
        for source in spec.sources
    )
    assert "-DBATCH_SIZE=2" in spec.extra_cuda_cflags
    assert "-DNUM_Q_HEADS=16" in spec.extra_cuda_cflags
    assert "-DNUM_KV_HEADS=2" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_PAGE_SIZE=32" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_FULL_BLOCKS=0" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_NVFP4=0" in spec.extra_cuda_cflags
    assert not any("FIXED_NUM_SPLITS" in flag for flag in spec.extra_cuda_cflags)


def test_cake_fmha_decode_quant_fp8_jit_fixed_split_member(monkeypatch) -> None:
    """The fixed 8 x 16 split-plan member: same sources, the plan and PDL as macros."""

    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    spec = gen_cake_fmha_decode_quant_fp8_module(
        "sm100a", 4, 1, 32, 4, 32, full_blocks=False, fixed_split=True
    )
    assert spec.name == get_cake_fmha_decode_quant_fp8_uri(
        "sm100a", 4, 1, 32, 4, 32, full_blocks=False, fixed_split=True
    )
    assert "_split8x16_pdl_" in spec.name
    assert spec.name != get_cake_fmha_decode_quant_fp8_uri(
        "sm100a", 4, 1, 32, 4, 32, full_blocks=False
    )
    assert any(
        str(source).endswith("cuda/decode_quant_fp8/default.cu")
        for source in spec.sources
    )
    assert any(
        str(source).endswith("cuda/decode_quant_fp8_reduce/default.cu")
        for source in spec.sources
    )
    for flag in (
        "-DCAKE_FMHA_FIXED_NUM_SPLITS=8",
        "-DCAKE_FMHA_FIXED_BLOCKS_PER_SPLIT=16",
        "-DCAKE_FMHA_ENABLE_PDL=1",
        "-DCAKE_FMHA_REDUCE_FIXED_NUM_SPLITS=8",
        "-DCAKE_FMHA_REDUCE_USE_PDL=1",
        "-DCAKE_FMHA_FULL_BLOCKS=0",
    ):
        assert flag in spec.extra_cuda_cflags
    with pytest.raises(ValueError):
        gen_cake_fmha_decode_quant_fp8_module(
            "sm100a", 4, 1, 32, 4, 32, full_blocks=True, fixed_split=True
        )


@pytest.mark.parametrize(
    ("batch_size", "num_kv_heads", "max_seq_len", "sm_count", "fixed"),
    [
        (4, 4, 16384, 148, True),
        (4, 4, 16384, 152, True),
        (8, 2, 16384, 148, True),
        (4, 4, 16129, 148, True),
        (4, 4, 16128, 148, False),
        (4, 4, 16385, 148, False),
        (4, 4, 8192, 148, False),
        (256, 4, 8192, 148, False),
        (4, 2, 16384, 148, False),
    ],
)
def test_cake_fmha_fp8_decode_fixed_split_plan(
    batch_size, num_kv_heads, max_seq_len, sm_count, fixed
) -> None:
    """The route takes the fixed-plan member exactly when the binding's host plan
    (max_seq_len and the SM count only) is 8 splits x 16 blocks."""

    assert (
        cake_api._decode_quant_fp8_fixed_split(
            batch_size, num_kv_heads, max_seq_len, sm_count
        )
        is fixed
    )


def test_cake_fmha_decode_quant_nvfp4_jit_selects_main_and_reducer(
    monkeypatch,
) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    spec = gen_cake_fmha_decode_quant_nvfp4_module("sm103a", 2, 1, 4, 2, 32)
    assert spec.name == get_cake_fmha_decode_quant_nvfp4_uri("sm103a", 2, 1, 4, 2, 32)
    assert {Path(source).name for source in spec.sources} == {
        "default.cu",
        "cake_fmha_decode_quant_nvfp4_binding.cu",
        "cake_fmha_decode_quant_fp8_reduce_binding.cu",
        "cake_fmha_decode_quant_fp8_jit_binding.cu",
    }
    assert any(
        str(source).endswith("cuda/decode_quant_nvfp4/default.cu")
        for source in spec.sources
    )
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
        "default.cu",
        "cake_fmha_context_bf16_binding.cu",
        "cake_fmha_context_bf16_jit_binding.cu",
    }
    assert "-DNUM_M_BLOCKS=1" in spec.extra_cuda_cflags
    assert "-DHEADS_PER_GROUP=2" in spec.extra_cuda_cflags
    assert "-DPACK_G=2" in spec.extra_cuda_cflags
    assert "-DTOK_PER_STAGE=64" in spec.extra_cuda_cflags


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
        "default.cu",
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
    spec = gen_cake_fmha_context_nvfp4_module("sm100a", 1, 32, 4, 8, 16, 8)
    assert spec.name == get_cake_fmha_context_nvfp4_uri("sm100a", 1, 32, 4, 8, 16, 8)
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
    "relative_path",
    [
        "jit/cake_fmha_context_bf16_jit_binding.cu",
        "jit/cake_fmha_context_fp8_jit_binding.cu",
        "jit/cake_fmha_context_hd256_jit_binding.cu",
    ],
)
def test_cake_fmha_optimized_context_adapters_require_signed_seq_lens(
    relative_path,
) -> None:
    adapter = (get_cake_fmha_csrc_dir() / relative_path).read_text(encoding="utf-8")
    assert "TVM_FFI_ICHECK_EQ(seq_lens.dtype(), dl_int32);" in adapter
    assert "seq_lens.dtype() == dl_uint32" not in adapter


@pytest.mark.parametrize(
    ("relative_path", "descriptor_count"),
    [
        ("jit/cake_fmha_context_bf16_jit_binding.cu", 3),
        ("jit/cake_fmha_context_fp8_jit_binding.cu", 5),
        ("jit/cake_fmha_context_hd256_jit_binding.cu", 3),
        ("jit/cake_fmha_decode_native_bf16_jit_binding.cu", 3),
        ("jit/cake_fmha_decode_native_fp16_hd512_jit_binding.cu", 3),
        ("jit/cake_fmha_decode_native_fp16_nhd_jit_binding.cu", 3),
        ("jit/cake_fmha_decode_quant_bf16q_jit_binding.cu", 2),
        ("jit/cake_fmha_decode_quant_fp8_jit_binding.cu", 5),
    ],
)
def test_cake_fmha_typed_launch_adapters_use_typed_tensor_maps(
    relative_path,
    descriptor_count,
) -> None:
    adapter = (get_cake_fmha_csrc_dir() / relative_path).read_text(encoding="utf-8")
    assert adapter.count("reinterpret_cast<CakeFmhaTensorMap const*>") == (
        descriptor_count
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "jit/cake_fmha_decode_balanced_jit_binding.cu",
        "jit/cake_fmha_decode_balanced_fp8_jit_binding.cu",
        "jit/cake_fmha_decode_balanced_hd64_jit_binding.cu",
        "jit/cake_fmha_decode_balanced_hd256_jit_binding.cu",
        "jit/cake_fmha_dcp_spec_bf16_balanced_jit_binding.cu",
        "jit/cake_fmha_dcp_spec_bf16_fp8_balanced_jit_binding.cu",
        "jit/cake_fmha_dcp_spec_bf16_fp8_d256_balanced_jit_binding.cu",
    ],
)
def test_cake_fmha_balanced_launch_adapters_pass_descriptors_by_value(
    relative_path,
) -> None:
    # The balanced kernels take their TMA descriptors as ``__grid_constant__``
    # parameters: the adapter encodes each descriptor on the host and hands it
    # to the launch binding by reference, with no device descriptor slots.
    adapter = (get_cake_fmha_csrc_dir() / relative_path).read_text(encoding="utf-8")
    assert adapter.count("CUtensorMap const& p_") == 3
    assert "CakeFmhaTensorMap" not in adapter
    assert "TmaDeviceSlot" not in adapter
    assert "RecordTmaDeviceSlotUses" not in adapter
    assert "cuMemAlloc" not in adapter


@pytest.mark.parametrize(
    ("relative_path", "expected_records"),
    [
        (
            "jit/cake_fmha_context_bf16_jit_binding.cu",
            ("{q_slot, k_slot, v_slot}",),
        ),
        (
            "jit/cake_fmha_context_fp8_jit_binding.cu",
            (
                "{q_slot, k_slot, v_slot, ksf_slot, vsf_slot}",
                "{q_slot, k_slot, v_slot}",
            ),
        ),
        (
            "jit/cake_fmha_context_hd256_jit_binding.cu",
            ("{q_slot, k_slot, v_slot}",),
        ),
        (
            "jit/cake_fmha_decode_native_bf16_jit_binding.cu",
            ("{q_slot, k_slot, v_slot}",),
        ),
        (
            "jit/cake_fmha_decode_native_fp16_hd512_jit_binding.cu",
            ("{q_slot, k_slot, v_slot}",),
        ),
        (
            "jit/cake_fmha_decode_native_fp16_nhd_jit_binding.cu",
            ("{q_slot, k_slot, v_slot}",),
        ),
        (
            "jit/cake_fmha_decode_quant_bf16q_jit_binding.cu",
            ("{k_slot, v_slot}",),
        ),
        (
            "jit/cake_fmha_decode_quant_fp8_jit_binding.cu",
            (
                "{q_slot, k_slot, v_slot, ksf_slot, vsf_slot}",
                "{q_slot, k_slot, v_slot}",
            ),
        ),
    ],
)
def test_cake_fmha_tma_adapters_track_descriptor_completion(
    relative_path,
    expected_records,
) -> None:
    csrc_dir = get_cake_fmha_csrc_dir()
    adapter = (csrc_dir / relative_path).read_text(encoding="utf-8")
    # The slot arena and the completion record are shared fragments; the
    # adapter keeps the slot lookup and its per-launch record calls.
    arena = (csrc_dir / "include/cake_fmha_tma_device_arena.inc").read_text(
        encoding="utf-8"
    )
    record = (csrc_dir / "include/cake_fmha_tma_device_slot_uses.inc").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(adapter.split())
    assert "TmaDeviceSlotLease" in adapter
    assert "cudaEventQuery" in arena
    assert "cudaEventRecord" in record
    assert "cuCtxGetId" in adapter
    for text in (adapter, arena, record):
        assert "size_t used = 0" not in text
        assert "++arena.used" not in text
    for leases in expected_records:
        assert f"RecordTmaDeviceSlotUses( {leases}, stream);" in normalized or (
            f"RecordTmaDeviceSlotUses({leases}, stream);" in normalized
        )


def test_cake_fmha_adapters_share_the_host_helper_fragments() -> None:
    """Every adapter includes the host-helper fragments it used to define inline."""

    csrc_dir = get_cake_fmha_csrc_dir()
    adapters = sorted((csrc_dir / "jit").glob("*_jit_binding.cu"))
    adapters.append(csrc_dir / "cake_fmha_jit_binding.cu")
    assert len(adapters) == 17
    tma_adapters = {
        "cake_fmha_context_bf16_jit_binding.cu",
        "cake_fmha_context_fp8_jit_binding.cu",
        "cake_fmha_context_hd256_jit_binding.cu",
        "cake_fmha_decode_native_bf16_jit_binding.cu",
        "cake_fmha_decode_native_fp16_hd512_jit_binding.cu",
        "cake_fmha_decode_native_fp16_nhd_jit_binding.cu",
        "cake_fmha_decode_quant_bf16q_jit_binding.cu",
        "cake_fmha_decode_quant_fp8_jit_binding.cu",
    }
    # fragment -> (the definitions it carries, the adapters that include it)
    fragments = {
        "include/cake_fmha_host_helpers.inc": (
            ("void CheckSameDevice(",),
            {adapter.name for adapter in adapters},
        ),
        "include/cake_fmha_tma_device_arena.inc": (
            (
                "struct TmaDeviceSlotState {",
                "struct TmaDeviceArena {",
                "struct TmaDeviceSlotLease {",
                "bool TmaDeviceSlotReady(",
                "void AddTmaDeviceSlotChunk(",
                "std::mutex& TmaDeviceSlotMutex()",
                "TmaDeviceArena>& TmaDeviceArenas()",
            ),
            tma_adapters,
        ),
        "include/cake_fmha_tma_device_slot_uses.inc": (
            ("void RecordTmaDeviceSlotUses(",),
            tma_adapters,
        ),
    }
    for fragment, (definitions, _) in fragments.items():
        text = (csrc_dir / fragment).read_text(encoding="utf-8")
        assert text.count("#pragma once\n") == 1
        # A textual fragment: it is included inside the adapter's anonymous
        # namespace, so it brings no includes and opens no namespace of its own.
        assert "#include" not in text
        assert not any(line.startswith("namespace") for line in text.splitlines())
        for definition in definitions:
            assert text.count(definition) == 1
    for adapter in adapters:
        text = adapter.read_text(encoding="utf-8")
        assert text.count("\nnamespace {\n") == 1
        anonymous = text.split("\nnamespace {\n", 1)[1].split("\n}  // namespace\n", 1)[
            0
        ]
        for fragment, (definitions, users) in fragments.items():
            include = f'#include "{fragment}"\n'
            if adapter.name in users:
                assert text.count(include) == 1 and include in anonymous
            else:
                assert include not in text
            for definition in definitions:
                assert definition not in text


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
    monkeypatch.setattr(
        cake_api,
        "_is_cake_fmha_decode_native_bf16_available",
        lambda *args, **kwargs: True,
    )
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
    lse_route = cake_api.select_cake_fmha_decode_route(
        query.device, **{**kwargs, "lse": valid_lse}
    )
    assert lse_route == dataclasses.replace(route, return_lse=True)
    assert cake_api.cake_fmha_route_is_optimized(lse_route)
    # No exact-shape member exists here, so the LSE request shares the module.
    assert get_cake_fmha_decode_native_bf16_uri(
        "sm100a",
        2,
        1,
        4,
        2,
        has_sink=False,
        has_window=False,
        use_scale_ptr=True,
        retain_kv_l2=True,
        return_lse=True,
    ) == get_cake_fmha_decode_native_bf16_uri(
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


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize(
    ("max_seq_len", "retain_kv_l2", "device_scale"),
    (
        (1024, True, False),
        # 2048 (not 4096): 128 x 4 tiles at a scalar scale stay on the present
        # route only inside the aligned hd128 selector's c1 carve-out.
        (2048, False, False),
        (4096, False, True),
    ),
)
def test_cake_fmha_exact_b128_members_serve_caller_lse_from_generic_member(
    monkeypatch, target, max_seq_len, retain_kv_l2, device_scale
) -> None:
    """A caller LSE at an exact b128 shape selects the generic member."""

    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    batch_size = 128
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
        seq_lens=torch.full((batch_size,), max_seq_len, dtype=torch.int32),
        batch_size=batch_size,
        q_len=1,
        max_seq_len=max_seq_len,
        window_left=-1,
        bmm1_scale=torch.ones(1, dtype=torch.float32) if device_scale else 1.0,
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
    # The generated DEC_B128 program serves the 2048 launch; the 1024 launch
    # (at most nine KV blocks) and the 4096 launch (a tensor bmm1 with a host
    # bmm2) stay on the JIT member.
    program = ("decode_bf16", "DEC_B128") if max_seq_len == 2048 else None
    assert route == cake_api.CakeFmhaDecodeRoute(
        target=target,
        batch_size=batch_size,
        q_len=1,
        num_q_heads=num_q_heads,
        num_kv_heads=4,
        has_sink=False,
        has_window=False,
        use_scale_ptr=device_scale,
        retain_kv_l2=retain_kv_l2,
        program=program,
    )
    assert cake_api.cake_fmha_route_is_optimized(route)

    caller_lse = torch.empty((batch_size, num_q_heads), dtype=torch.float32)
    lse_route = cake_api.select_cake_fmha_decode_route(
        query.device, **{**kwargs, "lse": caller_lse}
    )
    # A caller LSE keeps the route (and so its JIT member, which writes the LSE).
    assert lse_route == dataclasses.replace(route, return_lse=True)
    assert cake_api.cake_fmha_route_is_optimized(lse_route)

    flag_kwargs = dict(
        has_sink=False,
        has_window=False,
        use_scale_ptr=device_scale,
        retain_kv_l2=retain_kv_l2,
    )
    spec = gen_cake_fmha_decode_native_bf16_module(
        target, batch_size, 1, num_q_heads, 4, **flag_kwargs
    )
    lse_spec = gen_cake_fmha_decode_native_bf16_module(
        target, batch_size, 1, num_q_heads, 4, **flag_kwargs, return_lse=True
    )
    assert Path(lse_spec.sources[0]).name == "default.cu"
    assert Path(lse_spec.sources[1]).name == "cake_fmha_decode_native_bf16_binding.cu"
    assert "-DBATCH_SIZE=128" in lse_spec.extra_cuda_cflags
    assert lse_spec.extra_cuda_cflags == spec.extra_cuda_cflags
    assert Path(spec.sources[0]).name.startswith("batch_size128_")
    assert lse_spec.name != spec.name
    assert lse_spec.name.replace("_lse_", "_", 1) == spec.name


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
    # The generated DEC_SINK program serves the launch without an LSE.
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
        program=("decode_bf16", "DEC_SINK"),
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
    bad_scale_stride = torch.empty((4, 2, 16, 9), dtype=torch.float8_e4m3fn)[..., :8]
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
    too_many_group_heads = torch.empty((2, 18, 128), dtype=torch.float8_e4m3fn)
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
    misaligned_hd512 = torch.empty((4, 2, 64, 513), dtype=torch.float16)[..., :512]
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


def test_cake_fmha_fp8_decode_route_reads_only_host_scalars(
    monkeypatch,
) -> None:
    """The FP8 route plans from ``max_seq_len`` alone: the kernel reads
    ``seq_lens`` at run time and masks every tail, so odd block counts, tails
    and shorter requests route without a device-to-host read of the lengths."""

    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    batch_size, num_q_heads, num_kv_heads, page_size = 2, 16, 2, 16
    query = torch.empty((batch_size, num_q_heads, 128), dtype=torch.float8_e4m3fn)
    key = torch.empty((64, num_kv_heads, page_size, 128), dtype=torch.float8_e4m3fn)
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

    def no_host_read(*args, **kwargs):
        raise AssertionError("the FP8 decode route must not read seq_lens on the host")

    for lengths, max_seq_len in (([511, 512], 512), ([512, 640], 640), ([0, 1], 4096)):
        seq_lens = torch.tensor(lengths, dtype=torch.int32)
        with monkeypatch.context() as patched:
            for name in ("tolist", "item", "cpu", "numpy"):
                patched.setattr(torch.Tensor, name, no_host_read)
            route = select(seq_lens, max_seq_len)
        assert route is not None and route.component == "decode_quant_fp8"
    # Host scalars still bound the domain: fewer than four 128-token blocks.
    short = select(torch.tensor([256, 256], dtype=torch.int32), 256)
    assert short is None or short.component != "decode_quant_fp8"


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
    assert (
        cake_api.get_cake_fmha_decode_module(torch.device("cpu"), route)
        is fp16_sentinel
    )


def test_cake_fmha_fp16_hd512_route_loads_its_authenticated_adapter(
    monkeypatch,
) -> None:
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
    assert (
        cake_api.get_cake_fmha_decode_module(torch.device("cpu"), route)
        is hd512_sentinel
    )
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
    assert (
        cake_api.get_cake_fmha_decode_module(torch.device("cpu"), route)
        is bf16q_sentinel
    )
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
        cake_api.get_cake_fmha_decode_module(torch.device("cpu"), route) is fp8_sentinel
    )
    assert observed == {
        "args": ("sm103a", 2, 1, 16, 2, 16),
        "kwargs": {"full_blocks": False, "fixed_split": False},
    }
    fixed_route = dataclasses.replace(route, num_split=8)
    assert (
        cake_api.get_cake_fmha_decode_module(torch.device("cpu"), fixed_route)
        is fp8_sentinel
    )
    assert observed["kwargs"] == {"full_blocks": False, "fixed_split": True}


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
            cake_api.get_cake_fmha_decode_module(torch.device("cpu"), route) is sentinel
        )


def test_cake_fmha_context_candidate_selection_for_adapter_families(
    monkeypatch,
) -> None:
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
        workspace_bytes=2 << 20,
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


@pytest.mark.parametrize(
    ("query_dtype", "out_dtype", "causal", "required_workspace_bytes"),
    [
        (torch.float16, torch.float16, False, 1_114_224),
        (torch.float8_e4m3fn, torch.bfloat16, True, 819_312),
    ],
)
def test_cake_fmha_context_hd256_route_requires_exact_workspace(
    monkeypatch,
    query_dtype,
    out_dtype,
    causal,
    required_workspace_bytes,
) -> None:
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    query = torch.empty((16, 4, 256), dtype=query_dtype)
    key = torch.empty((4, 2, 16, 256), dtype=query_dtype)

    def select(workspace_buffer):
        return cake_api.select_cake_fmha_context_route(
            query.device,
            query=query,
            key_cache=key,
            value_cache=torch.empty_like(key),
            out=torch.empty_like(query, dtype=out_dtype),
            block_tables=torch.zeros((2, 2, 2), dtype=torch.int32),
            seq_lens=torch.tensor([16, 16], dtype=torch.int32),
            batch_size=2,
            max_q_len=8,
            max_kv_len=16,
            window_left=-1,
            bmm1_scale=0.125,
            bmm2_scale=1.0,
            sinks=None,
            uses_shared_paged_kv_idx=False,
            cum_seq_lens_q=torch.tensor([0, 8, 16], dtype=torch.int32),
            cum_seq_lens_kv=torch.tensor([0, 16, 32], dtype=torch.int32),
            key_block_scales=None,
            value_block_scales=None,
            skip_softmax_threshold_scale_factor=None,
            is_causal=causal,
            lse=None,
            kv_layout="NHD",
            workspace_buffer=workspace_buffer,
        )

    exact = torch.empty(required_workspace_bytes, dtype=torch.uint8)
    assert select(exact) is not None
    assert select(torch.empty(required_workspace_bytes - 1, dtype=torch.uint8)) is None
    noncontiguous = torch.empty(required_workspace_bytes * 2, dtype=torch.uint8)[::2]
    assert select(noncontiguous) is None
    assert select(torch.empty(required_workspace_bytes, device="meta")) is None
    misaligned = torch.empty(required_workspace_bytes + 1, dtype=torch.uint8)[1:]
    assert select(misaligned) is None


@pytest.mark.parametrize(
    ("target", "profile", "shape", "program"),
    [
        ("sm100a", "q511", (11, 10, 2, 5, 32, 1), "Q511_SM100A"),
        ("sm103a", "q511", (11, 10, 2, 5, 32, 1), "Q511_SM103A"),
        ("sm100a", "q257", (6, 10, 2, 5, 1024, 8), "Q257_SM100A"),
        ("sm103a", "q257", (6, 10, 2, 5, 1024, 8), "Q257_SM103A"),
    ],
)
def test_cake_fmha_context_bf16_exact_route_loads_generated_program(
    monkeypatch, target, profile, shape, program
) -> None:
    sentinel = object()
    observed = []
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    monkeypatch.setattr(
        cake_api,
        "_load_cake_fmha_context_program",
        lambda route: observed.append(route) or sentinel,
    )
    monkeypatch.setattr(
        cake_api,
        "load_cake_fmha_context_bf16_module",
        lambda *args, **kwargs: pytest.fail("the JIT module serves no exact route"),
    )
    num_m_blocks, num_q_heads, num_kv_heads, pack_g, page_size, l2_swizzle = shape
    route = cake_api.CakeFmhaContextRoute(
        target=target,
        component="context_bf16",
        num_m_blocks=num_m_blocks,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        pack_g=pack_g,
        page_size=page_size,
        l2_swizzle=l2_swizzle,
        is_causal=True,
        return_lse=False,
        enable_sink=False,
        exact_profile=profile,
        program=program,
    )
    assert cake_api.get_cake_fmha_context_module(torch.device("cpu"), route) is sentinel
    assert observed == [route]
    with pytest.raises(RuntimeError, match="has no program"):
        cake_api.get_cake_fmha_context_module(
            torch.device("cpu"), dataclasses.replace(route, program=None)
        )


def test_cake_fmha_context_fp8_route_omits_bf16_exact_profile(monkeypatch) -> None:
    sentinel = object()
    observed = {}

    def load(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    monkeypatch.setattr(cake_api, "load_cake_fmha_context_fp8_module", load)
    route = cake_api.CakeFmhaContextRoute(
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
    assert cake_api.get_cake_fmha_context_module(torch.device("cpu"), route) is sentinel
    assert observed == {
        "args": ("sm103a", 1, 32, 4, 8, 64, 8),
        "kwargs": {
            "is_causal": True,
            "return_lse": False,
            "enable_sink": False,
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
    monkeypatch.setattr(
        cake_api, "load_cake_fmha_compat_module", lambda target: sentinel
    )
    assert cake_api.get_cake_fmha_decode_module(torch.device("cpu"), None) is sentinel


def test_context_route_miss_fails_closed_to_compat(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm103a")
    monkeypatch.setattr(
        cake_api, "load_cake_fmha_compat_module", lambda target: sentinel
    )
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

    def select_route(_device, **kwargs):
        observed["selector_skip"] = kwargs["skip_softmax_threshold_scale_factor"]
        return None

    compat_module = SimpleNamespace(cake_paged_attention_decode=run)
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    monkeypatch.setattr(
        cake_api, "load_cake_fmha_compat_module", lambda target: compat_module
    )
    monkeypatch.setattr(cake_api, "select_cake_fmha_decode_route", select_route)
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
    assert observed["selector_skip"] == expected_ffi_value
    assert len(observed["args"]) == 33
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
        window_right=0,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
    )

    assert result.shape == query.shape
    assert len(observed["args"]) == 34
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
                "seq_lens": kwargs["seq_lens"].to(torch.uint32),
            },
        )
        is None
    )
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
    assert (
        cake_api.select_cake_fmha_context_route(
            query.device,
            **{
                **kwargs,
                "skip_softmax_threshold_scale_factor": 1e-30,
            },
        )
        == route
    )
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
        program="B256_KV4",
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
    ) == dataclasses.replace(
        fp8_route, device_scales=True, program="B256_SCALE_PTR_KV4"
    )
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
def test_cake_context_exact_mask_profile_selects_on_host_scalars(
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
        program=f"{profile.upper()}_SM103A",
    )

    # The programs read the lengths at run time: shorter KV lengths keep the
    # route, and selection never reads a tensor value (meta tensors).
    nonuniform = kwargs["seq_lens"].clone()
    nonuniform[-1] -= 1
    assert (
        cake_api.select_cake_fmha_context_route(
            query.device, **{**kwargs, "seq_lens": nonuniform}
        )
        == route
    )
    meta = {
        name: value.to("meta")
        for name, value in kwargs.items()
        if isinstance(value, torch.Tensor)
    }
    meta_route = cake_api.select_cake_fmha_context_route(
        torch.device("meta"), **{**kwargs, **meta}
    )
    assert meta_route == route
    # The switch back to uniform lengths: the host reads the lengths, serves a
    # uniform batch and keeps a non-uniform one on the generic JIT route.
    monkeypatch.setattr(cake_api, "_CONTEXT_EXACT_PROFILE_NEEDS_UNIFORM_LENGTHS", True)
    assert cake_api.select_cake_fmha_context_route(query.device, **kwargs) == route
    generic = cake_api.select_cake_fmha_context_route(
        query.device, **{**kwargs, "seq_lens": nonuniform}
    )
    assert generic is not None
    assert generic.exact_profile is None and generic.program is None
    monkeypatch.setattr(cake_api, "_CONTEXT_EXACT_PROFILE_NEEDS_UNIFORM_LENGTHS", False)
    # A query row count off batch x max_q_len (ragged query lengths) keeps the
    # generic program.
    ragged = cake_api.select_cake_fmha_context_route(
        query.device,
        **{**kwargs, "query": query[:-1], "out": kwargs["out"][:-1]},
    )
    assert ragged is not None
    assert ragged.exact_profile is None
    assert ragged.program is None


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
    # Other AOT providers can independently register Blackwell modules.
    assert sorted(spec.name for spec in specs if spec.name.startswith("cake-")) == [
        "cake-sm100a",
        "cake-sm103a",
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)
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
def test_cake_decode_fully_masked_row_returns_zero() -> None:
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        pytest.skip("Cake FMHA requires SM100 or SM103")

    query = torch.zeros((1, 4, 128), dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros((1, 2, 2, 16, 128), dtype=torch.bfloat16, device=device)
    actual = cake_api.cake_batch_decode_with_kv_cache(
        query,
        kv_cache,
        torch.empty(1024 * 1024, dtype=torch.uint8, device=device),
        torch.zeros((1, 1), dtype=torch.int32, device=device),
        torch.zeros((1,), dtype=torch.int32, device=device),
        1,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.zeros_like(actual))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cake_decode_exact_sink_matches_independent_reference() -> None:
    """Compile and validate the exact sink override against PyTorch math."""

    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        pytest.skip("Cake FMHA exact sink requires SM100 or SM103")

    torch.manual_seed(0)
    batch_size = 256
    q_len = 1
    num_q_heads = 32
    num_kv_heads = 4
    page_size = 16
    head_dim = 128
    kv_len = 4096
    pages_per_seq = kv_len // page_size

    query = (
        torch.randn(
            (batch_size * q_len, num_q_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    kv_cache = (
        torch.randn(
            (pages_per_seq, 2, num_kv_heads, page_size, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    # Reuse one complete physical page row for every request.  This keeps the
    # exact logical B256/K4096 route while bounding the test's device footprint.
    block_tables = torch.arange(pages_per_seq, dtype=torch.int32, device=device).repeat(
        batch_size, 1
    )
    seq_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    sinks = torch.linspace(-2.0, 8.0, num_q_heads, dtype=torch.float32, device=device)
    reference_workspace = torch.empty(
        256 * 1024 * 1024, dtype=torch.uint8, device=device
    )
    cake_workspace = torch.empty_like(reference_workspace)
    reference_out = torch.empty_like(query)
    cake_out = torch.empty_like(query)
    common = dict(
        query=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=kv_len,
        bmm1_scale=float(head_dim**-0.5),
        bmm2_scale=1.0,
        window_left=-1,
        sinks=sinks,
        kv_layout="HND",
        enable_pdl=True,
        q_len_per_req=q_len,
        uses_shared_paged_kv_idx=True,
        return_lse=False,
    )

    route = cake_api.select_cake_fmha_decode_route(
        device,
        query=query,
        key_cache=kv_cache[:, 0],
        value_cache=kv_cache[:, 1],
        out=cake_out,
        workspace_buffer=cake_workspace,
        block_tables=block_tables,
        seq_lens=seq_lens,
        batch_size=batch_size,
        q_len=q_len,
        max_seq_len=kv_len,
        window_left=-1,
        bmm1_scale=common["bmm1_scale"],
        bmm2_scale=common["bmm2_scale"],
        o_scale=1.0,
        sinks=sinks,
        kv_layout="HND",
        uses_shared_paged_kv_idx=True,
        cum_seq_lens_q=None,
        key_block_scales=None,
        value_block_scales=None,
        skip_softmax_threshold_scale_factor=None,
        enable_block_sparse_attention=False,
        lse=None,
    )
    assert route is not None
    assert route.component == "decode_native_bf16"
    assert (
        route.batch_size,
        route.q_len,
        route.num_q_heads,
        route.num_kv_heads,
        route.has_sink,
        route.has_window,
        route.use_scale_ptr,
        route.retain_kv_l2,
        route.page_size,
    ) == (256, 1, 32, 4, True, False, False, False, 16)
    assert cake_api.cake_fmha_route_is_optimized(route)
    # This call forces the member-level body and launch_override binding to JIT
    # compile before the public API correctness comparison below.
    assert cake_api.get_cake_fmha_decode_module(device, route) is not None

    reference = decode.trtllm_batch_decode_with_kv_cache(
        workspace_buffer=reference_workspace,
        out=reference_out,
        backend="trtllm-gen",
        **common,
    )
    actual = cake_api.cake_batch_decode_with_kv_cache(
        workspace_buffer=cake_workspace,
        out=cake_out,
        **common,
    )
    assert reference.data_ptr() == reference_out.data_ptr()
    assert actual.data_ptr() == cake_out.data_ptr()

    # Both production kernels accumulate this long reduction in BF16-specific
    # orders, so comparing them directly compounds their independent rounding
    # errors.  Use bounded BF16 inputs and the same sink-attention definition as
    # FlashInfer's reference tests, evaluated in FP32, and judge each
    # implementation against it at the standard BF16 tolerance.
    key = kv_cache[:, 0].permute(0, 2, 1, 3).reshape(kv_len, num_kv_heads, head_dim)
    value = kv_cache[:, 1].permute(0, 2, 1, 3).reshape(kv_len, num_kv_heads, head_dim)
    key = torch.repeat_interleave(key, num_q_heads // num_kv_heads, dim=1).contiguous()
    value = torch.repeat_interleave(
        value, num_q_heads // num_kv_heads, dim=1
    ).contiguous()
    logits = torch.einsum("bhd,khd->bhk", query.float(), key.float())
    logits *= common["bmm1_scale"]
    sink_logits = sinks.view(1, num_q_heads, 1).expand(batch_size, -1, -1)
    weights = torch.softmax(torch.cat([logits, sink_logits], dim=-1), dim=-1)
    independent = torch.einsum("bhk,khd->bhd", weights[..., :-1], value.float()).to(
        query
    )

    torch.testing.assert_close(
        actual.float(), independent.float(), rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        reference.float(), independent.float(), rtol=1e-2, atol=1e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize(
    ("batch_size", "kv_len", "device_scale"),
    (
        # Exact b4 CGA member shape (RETAIN_KV_L2=1).
        (4, 1024, False),
        # Exact b128 RETAIN_KV_L2=1 member shape.
        (128, 1024, False),
        # Exact b128 RETAIN_KV_L2=0 USE_SCALE_PTR=0 member shape.
        (128, 4096, False),
        # Exact b128 RETAIN_KV_L2=0 USE_SCALE_PTR=1 member shape.
        (128, 4096, True),
    ),
)
def test_cake_decode_exact_shapes_return_lse_match_reference(
    batch_size, kv_len, device_scale
) -> None:
    """``return_lse=True`` at the exact decode shapes writes a correct LSE."""

    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        pytest.skip("Cake FMHA requires SM100 or SM103")

    torch.manual_seed(0)
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    page_size = 16
    pages_per_seq = kv_len // page_size
    query = (
        torch.randn(
            (batch_size, num_q_heads, head_dim), dtype=torch.bfloat16, device=device
        )
        * 0.125
    )
    kv_cache = (
        torch.randn(
            (pages_per_seq, 2, num_kv_heads, page_size, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    block_tables = torch.arange(pages_per_seq, dtype=torch.int32, device=device).repeat(
        batch_size, 1
    )
    seq_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    scale = float(head_dim**-0.5)
    bmm1_scale = (
        torch.tensor(scale, dtype=torch.float32, device=device)
        if device_scale
        else scale
    )
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    caller_lse = torch.empty(
        (batch_size, num_q_heads), dtype=torch.float32, device=device
    )

    route_kwargs = dict(
        query=query,
        key_cache=kv_cache[:, 0],
        value_cache=kv_cache[:, 1],
        out=torch.empty_like(query),
        workspace_buffer=workspace,
        block_tables=block_tables,
        seq_lens=seq_lens,
        batch_size=batch_size,
        q_len=1,
        max_seq_len=kv_len,
        window_left=-1,
        bmm1_scale=bmm1_scale,
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
    route = cake_api.select_cake_fmha_decode_route(device, **route_kwargs)
    lse_route = cake_api.select_cake_fmha_decode_route(
        device, **route_kwargs, lse=caller_lse
    )
    assert route is not None
    assert route.component == "decode_native_bf16"
    assert (route.retain_kv_l2, route.use_scale_ptr) == (kv_len <= 1152, device_scale)
    assert lse_route == dataclasses.replace(route, return_lse=True)
    assert cake_api.cake_fmha_route_is_optimized(route)
    assert cake_api.cake_fmha_route_is_optimized(lse_route)
    lse_spec = gen_cake_fmha_decode_native_bf16_module(
        route.target,
        batch_size,
        1,
        num_q_heads,
        num_kv_heads,
        has_sink=False,
        has_window=False,
        use_scale_ptr=device_scale,
        retain_kv_l2=route.retain_kv_l2,
        return_lse=True,
    )
    assert not Path(lse_spec.sources[0]).name.startswith("batch_size")

    common = dict(
        query=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=kv_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=1.0,
        window_left=-1,
        kv_layout="HND",
        enable_pdl=False,
        q_len_per_req=1,
        uses_shared_paged_kv_idx=True,
    )
    actual, actual_lse = cake_api.cake_batch_decode_with_kv_cache(
        workspace_buffer=workspace,
        out=torch.empty_like(query),
        lse=caller_lse,
        return_lse=True,
        **common,
    )
    assert actual_lse.data_ptr() == caller_lse.data_ptr()
    plain = cake_api.cake_batch_decode_with_kv_cache(
        workspace_buffer=torch.empty_like(workspace),
        out=torch.empty_like(query),
        **common,
    )

    reference_wrapper = decode.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        "HND",
        use_tensor_cores=True,
    )
    reference_wrapper.plan(
        torch.arange(
            0,
            (batch_size + 1) * pages_per_seq,
            pages_per_seq,
            dtype=torch.int32,
            device=device,
        ),
        block_tables.reshape(-1),
        torch.full((batch_size,), page_size, dtype=torch.int32, device=device),
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        kv_data_type=torch.bfloat16,
        q_data_type=torch.bfloat16,
        sm_scale=scale,
    )
    reference, reference_lse = reference_wrapper.run(query, kv_cache, return_lse=True)

    # Independent FP32 definition: base-2 log-sum-exp of the scaled logits.
    key = kv_cache[:, 0].permute(0, 2, 1, 3).reshape(kv_len, num_kv_heads, head_dim)
    value = kv_cache[:, 1].permute(0, 2, 1, 3).reshape(kv_len, num_kv_heads, head_dim)
    key = torch.repeat_interleave(key, num_q_heads // num_kv_heads, dim=1).contiguous()
    value = torch.repeat_interleave(
        value, num_q_heads // num_kv_heads, dim=1
    ).contiguous()
    logits = torch.einsum("bhd,khd->bhk", query.float(), key.float()) * scale
    independent = torch.einsum(
        "bhk,khd->bhd", torch.softmax(logits, dim=-1), value.float()
    )
    independent_lse = torch.logsumexp(logits, dim=-1) / math.log(2.0)

    assert torch.isfinite(actual_lse).all()
    torch.testing.assert_close(actual.float(), independent, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(plain.float(), independent, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(reference.float(), independent, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_lse, independent_lse, rtol=1e-3, atol=3e-3)
    torch.testing.assert_close(actual_lse, reference_lse.float(), rtol=1e-3, atol=3e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)
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


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)
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


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)
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


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)
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


def test_cake_fmha_decode_native_bf16_hd256_smallm_jit_selects_component(
    monkeypatch,
) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    spec = gen_cake_fmha_decode_native_bf16_hd256_smallm_module(
        "sm100a", 64, 64, 8, 8, 148
    )
    assert spec.name == get_cake_fmha_decode_native_bf16_hd256_smallm_uri(
        "sm100a", 64, 64, 8, 8, 148
    )
    assert cake_fmha_smallm_component_name(64, 64) == (
        "decode_native_bf16_hd256_smallm_n64_p64"
    )
    assert {Path(source).name for source in spec.sources} == {
        "default.cu",
        "cake_fmha_decode_native_bf16_hd256_smallm_n64_p64_binding.cu",
        "cake_fmha_decode_native_bf16_hd256_smallm_jit_binding.cu",
    }
    assert any(
        "decode_native_bf16_hd256_smallm/default.cu" in str(s) for s in spec.sources
    )
    assert "-DQ_LEN=8" in spec.extra_cuda_cflags
    assert "-DQ_BOX_ROWS=8" in spec.extra_cuda_cflags
    assert "-DGROUP=8" in spec.extra_cuda_cflags
    assert "-DNUM_SPLIT=148" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_SMALLM_N_ROWS=64" in spec.extra_cuda_cflags
    assert "-DCAKE_FMHA_SMALLM_PAGE_SIZE=64" in spec.extra_cuda_cflags
    assert (
        "-DCAKE_FMHA_SMALLM_LAUNCH=cake_fmha_launch_decode_native_bf16_hd256_smallm_n64_p64"
        in spec.extra_cuda_cflags
    )
    with pytest.raises(ValueError):
        gen_cake_fmha_decode_native_bf16_hd256_smallm_module(
            "sm100a", 64, 64, 4, 8, 148
        )
    with pytest.raises(ValueError):
        cake_fmha_smallm_component_name(64, 128)


def test_cake_fmha_smallm_num_split_is_one_resident_wave() -> None:
    assert (
        cake_api._smallm_num_split(tiles=1, sm_count=148, max_seq_len=60008, n_rows=64)
        == 148
    )
    assert (
        cake_api._smallm_num_split(tiles=2, sm_count=148, max_seq_len=60004, n_rows=64)
        == 74
    )
    assert (
        cake_api._smallm_num_split(tiles=1, sm_count=148, max_seq_len=1032, n_rows=64)
        == 64
    )
    assert (
        cake_api._smallm_num_split(tiles=200, sm_count=148, max_seq_len=1032, n_rows=64)
        == 0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize(
    (
        "batch_size",
        "q_len_per_req",
        "num_kv_heads",
        "head_grp_size",
        "page_size",
        "max_in_kv_len",
    ),
    (
        # Qwen3.5-35B-A3B TP=2 rank: 8 heads over one KV head.  SGLang NEXTN with 7
        # draft tokens verifies 7 query rows (56 packed rows padded to 64); a
        # 7-draft + root verify gives 8 rows (64 packed).
        (1, 7, 1, 8, 64, 60000),
        (1, 8, 1, 8, 64, 60000),
        (1, 8, 1, 8, 16, 8192),
        # TP=1 rank with four draft tokens: two KV heads, 64 packed rows each.
        (1, 4, 2, 8, 64, 4096),
        # 32 packed rows.
        (2, 4, 1, 8, 32, 1024),
    ),
)
def test_cake_decode_bf16_hd256_smallm_matches_flashinfer_reference(
    batch_size, q_len_per_req, num_kv_heads, head_grp_size, page_size, max_in_kv_len
) -> None:
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        pytest.skip("Cake FMHA requires SM100 or SM103")
    from tests.attention.test_trtllm_gen_attention_decode import (
        _test_trtllm_batch_decode,
    )

    _test_trtllm_batch_decode(
        backend="cake",
        kv_layout="HND",
        batch_size=batch_size,
        q_len_per_req=q_len_per_req,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_grp_size=head_grp_size,
        window_left=-1,
        q_dtype="bf16",
        o_dtype="bf16",
        kv_dtype="bf16",
        enable_pdl=False,
        enable_sink=False,
        max_in_kv_len=max_in_kv_len,
        head_dim=256,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cake_decode_bf16_hd256_smallm_route_is_selected() -> None:
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        pytest.skip("Cake FMHA requires SM100 or SM103")
    batch, q_len, heads, kv_heads, page, kv_len = 1, 8, 8, 1, 64, 60008
    pages = (kv_len + page - 1) // page
    query = torch.randn(
        (batch * q_len, heads, 256), dtype=torch.bfloat16, device=device
    )
    key_cache = torch.randn(
        (pages, kv_heads, page, 256), dtype=torch.bfloat16, device=device
    )
    value_cache = torch.randn_like(key_cache)
    out = torch.empty_like(query)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    block_tables = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)
    seq_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)
    route = cake_api.select_cake_fmha_decode_route(
        device,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        out=out,
        workspace_buffer=workspace,
        block_tables=block_tables,
        seq_lens=seq_lens,
        batch_size=batch,
        q_len=q_len,
        max_seq_len=kv_len,
        window_left=-1,
        bmm1_scale=256**-0.5,
        bmm2_scale=1.0,
        o_scale=None,
        sinks=None,
        kv_layout="HND",
        uses_shared_paged_kv_idx=True,
        cum_seq_lens_q=None,
        key_block_scales=None,
        value_block_scales=None,
        skip_softmax_threshold_scale_factor=None,
        enable_block_sparse_attention=False,
    )
    assert route is not None
    assert route.component == "decode_native_bf16_hd256_smallm"
    assert route.page_size == page
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    assert route.num_split == min(sm_count, 256)
    assert cake_api.cake_fmha_route_is_optimized(route)

    # Run the public API through the selected route and check O and LSE against
    # dense bottom-right-causal attention (BF16 atol=rtol=1e-2, LSE 1e-2).
    scale = 256**-0.5
    out, lse = decode.trtllm_batch_decode_with_kv_cache(
        query,
        torch.stack((key_cache, value_cache), dim=1),
        workspace,
        block_tables,
        seq_lens,
        kv_len,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        backend="cake",
        q_len_per_req=q_len,
        kv_layout="HND",
        return_lse=True,
    )
    keys = key_cache.reshape(pages * page, kv_heads, 256)[:kv_len].float()
    values = value_cache.reshape(pages * page, kv_heads, 256)[:kv_len].float()
    keys = keys.repeat_interleave(heads // kv_heads, dim=1)
    values = values.repeat_interleave(heads // kv_heads, dim=1)
    scores = torch.einsum("qhd,khd->hqk", query.float(), keys) * scale
    positions = torch.arange(kv_len, device=device)
    visible = (kv_len - q_len) + torch.arange(q_len, device=device)[:, None]
    scores.masked_fill_((positions[None, :] > visible).unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    expected = torch.einsum("hqk,khd->qhd", probabilities, values)
    expected_lse = torch.logsumexp(scores, dim=-1).transpose(0, 1) * math.log2(math.e)
    torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse.float(), expected_lse, atol=1e-2, rtol=1e-2)


_BALANCED_AGENTX = [
    237569,
    120001,
    65536,
    4097,
    32768,
    8192,
    190000,
    512,
    100000,
    76543,
    2048,
    150000,
    33000,
    64000,
    12345,
    200000,
]


def _balanced_decode_kwargs(
    *,
    batch_size: int,
    q_len: int,
    num_kv_heads: int,
    seq_lens,
    sm_count: int = 148,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    page_size: int = 16,
    group: int = 8,
    q_dtype: torch.dtype | None = None,
    kv_dtype: torch.dtype | None = None,
    o_dtype: torch.dtype | None = None,
    bmm1_scale: float | None = None,
    bmm2_scale: float = 1.0,
):
    """Host-side ``select_cake_fmha_decode_route`` inputs of the balanced
    families.  ``dtype`` is the Q / KV / O default; ``q_dtype`` / ``kv_dtype`` /
    ``o_dtype`` override it per tensor (E4M3 cache families)."""

    q_dtype = dtype if q_dtype is None else q_dtype
    kv_dtype = q_dtype if kv_dtype is None else kv_dtype
    o_dtype = q_dtype if o_dtype is None else o_dtype
    num_q_heads = group * num_kv_heads
    query = torch.empty((batch_size * q_len, num_q_heads, head_dim), dtype=q_dtype)
    key = torch.empty((4, num_kv_heads, page_size, head_dim), dtype=kv_dtype)
    value = torch.empty_like(key)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    max_seq_len = int(seq_lens.max())
    return dict(
        query=query,
        key_cache=key,
        value_cache=value,
        out=torch.empty_like(query, dtype=o_dtype),
        workspace_buffer=torch.empty(
            cake_api.cake_fmha_balanced_workspace_bytes(
                sm_count, q_len, head_dim=head_dim
            ),
            dtype=torch.uint8,
        ),
        block_tables=torch.zeros(
            (batch_size, (max_seq_len + page_size - 1) // page_size),
            dtype=torch.int32,
        ),
        seq_lens=seq_lens,
        batch_size=batch_size,
        q_len=q_len,
        max_seq_len=max_seq_len,
        window_left=-1,
        bmm1_scale=head_dim**-0.5 if bmm1_scale is None else bmm1_scale,
        bmm2_scale=bmm2_scale,
        o_scale=1.0,
        sinks=None,
        kv_layout="HND",
        uses_shared_paged_kv_idx=True,
        cum_seq_lens_q=None,
        key_block_scales=None,
        value_block_scales=None,
        skip_softmax_threshold_scale_factor=None,
        enable_block_sparse_attention=False,
        multi_ctas_kv_counter_buffer=torch.zeros(
            cake_api.cake_fmha_balanced_counter_bytes(sm_count, head_dim=head_dim),
            dtype=torch.uint8,
        ),
    )


@pytest.mark.parametrize("dtype_name", ["bf16", "fp16"])
def test_cake_fmha_balanced_decode_route_owns_long_ragged_gqa8_decode(
    monkeypatch, dtype_name
) -> None:
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    component = f"decode_balanced_{dtype_name}"
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    monkeypatch.setattr(cake_api, "_smallm_sm_count", lambda device: 148)
    monkeypatch.setattr(
        cake_api,
        "_is_cake_fmha_decode_native_bf16_available",
        lambda *args, **kwargs: True,
    )
    agentx = [
        237569,
        120001,
        65536,
        4097,
        32768,
        8192,
        190000,
        512,
        100000,
        76543,
        2048,
        150000,
        33000,
        64000,
        12345,
        200000,
    ]
    kwargs = _balanced_decode_kwargs(
        batch_size=16, q_len=1, num_kv_heads=1, seq_lens=agentx, dtype=torch_dtype
    )
    route = cake_api.select_cake_fmha_decode_route(kwargs["query"].device, **kwargs)
    assert route == cake_api.CakeFmhaDecodeRoute(
        target="sm100a",
        batch_size=16,
        q_len=1,
        num_q_heads=8,
        num_kv_heads=1,
        has_sink=False,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=False,
        component=component,
        page_size=16,
    )
    assert cake_api.cake_fmha_route_is_optimized(route)
    assert cake_api._route_components(route) == (
        component,
        f"{component}_mtp_n32",
        f"{component}_mtp_n64",
    )

    # Packed-row MTP draft lengths 3..8 stay on the balanced route; q_len 2 and
    # 9 fall through to the incumbent grid-stride kernel.
    for q_len in (3, 4, 5, 8):
        mtp = _balanced_decode_kwargs(
            batch_size=64,
            q_len=q_len,
            num_kv_heads=1,
            seq_lens=[65536] * 64,
            dtype=torch_dtype,
        )
        mtp_route = cake_api.select_cake_fmha_decode_route(mtp["query"].device, **mtp)
        assert mtp_route is not None and mtp_route.component == component, q_len
    for q_len in (2, 9):
        other = _balanced_decode_kwargs(
            batch_size=64,
            q_len=q_len,
            num_kv_heads=1,
            seq_lens=[65536] * 64,
            dtype=torch_dtype,
        )
        other_route = cake_api.select_cake_fmha_decode_route(
            other["query"].device, **other
        )
        assert other_route is None or other_route.component != component, q_len

    # Host-metadata band (q_len 1 only).  BF16: below 32 K KV or below nine
    # (request, KV head) work tiles the incumbent grid-stride route keeps the
    # batch.  FP16 (HND, page 16, GQA-8) has no other optimized route -- the
    # previous contract was the compat fallback -- so the band admits every shape.
    short = _balanced_decode_kwargs(
        batch_size=16, q_len=1, num_kv_heads=1, seq_lens=[32767] * 16, dtype=torch_dtype
    )
    short_route = cake_api.select_cake_fmha_decode_route(short["query"].device, **short)
    few = _balanced_decode_kwargs(
        batch_size=8, q_len=1, num_kv_heads=1, seq_lens=[65536] * 8, dtype=torch_dtype
    )
    few_route = cake_api.select_cake_fmha_decode_route(few["query"].device, **few)
    below_band = "decode_native_bf16" if dtype_name == "bf16" else component
    assert short_route is not None and short_route.component == below_band
    assert few_route is not None and few_route.component == below_band
    # The packed MTP tiles take the balanced kernel at every KV length and
    # tile count (1.6-11.8x faster than the grid-stride route on every
    # measured short / few-tile shape).
    for q_len, batch_size, seq_lens in (
        (3, 8, [4096] * 8),
        (7, 1, [60007]),
        (7, 4, [300, 257, 5000, 777]),
        (8, 2, [128, 16]),
    ):
        mtp_small = _balanced_decode_kwargs(
            batch_size=batch_size,
            q_len=q_len,
            num_kv_heads=1,
            seq_lens=seq_lens,
            dtype=torch_dtype,
        )
        mtp_small_route = cake_api.select_cake_fmha_decode_route(
            mtp_small["query"].device, **mtp_small
        )
        assert mtp_small_route is not None and mtp_small_route.component == component, (
            q_len,
            batch_size,
            seq_lens,
        )

    # Fail closed on every field the exported kernels do not serve.
    for mutation in (
        {"sinks": torch.zeros(8, dtype=torch.float32)},
        {"window_left": 1024},
        {"bmm1_scale": torch.ones(1, dtype=torch.float32)},
        {"bmm2_scale": 0.5},
        {"uses_shared_paged_kv_idx": False},
        {"kv_layout": "NHD"},
        {"lse": torch.empty((16, 8), dtype=torch.float32)},
        {"workspace_buffer": torch.empty(4096, dtype=torch.uint8)},
        {"multi_ctas_kv_counter_buffer": torch.zeros(64, dtype=torch.uint8)},
        {
            "batch_size": 1025,
            "query": torch.empty((1025, 8, 128), dtype=torch_dtype),
            "out": torch.empty((1025, 8, 128), dtype=torch_dtype),
            "seq_lens": torch.full((1025,), 65536, dtype=torch.int32),
            "block_tables": torch.zeros((1025, 4096), dtype=torch.int32),
        },
    ):
        mutated = {**kwargs, **mutation}
        selected = cake_api.select_cake_fmha_decode_route(
            mutated["query"].device, **mutated
        )
        assert selected is None or selected.component != component, mutation


def _select_balanced_kwargs(kwargs):
    return cake_api.select_cake_fmha_decode_route(kwargs["query"].device, **kwargs)


@pytest.mark.parametrize("q_dtype_name", ["fp8", "bf16q", "fp16q"])
def test_cake_fmha_balanced_fp8_kv_route_owns_gqa8_e4m3_decode(
    monkeypatch, q_dtype_name
) -> None:
    q_dtype = {
        "fp8": torch.float8_e4m3fn,
        "bf16q": torch.bfloat16,
        "fp16q": torch.float16,
    }[q_dtype_name]
    component = f"decode_balanced_{q_dtype_name}"
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    monkeypatch.setattr(cake_api, "_smallm_sm_count", lambda device: 148)

    def build(**overrides):
        params = dict(
            batch_size=16,
            q_len=1,
            num_kv_heads=1,
            seq_lens=_BALANCED_AGENTX,
            q_dtype=q_dtype,
            kv_dtype=torch.float8_e4m3fn,
            bmm1_scale=0.02,
            bmm2_scale=1.5,
        )
        params.update(overrides)
        return _balanced_decode_kwargs(**params)

    kwargs = build()
    route = _select_balanced_kwargs(kwargs)
    assert route == cake_api.CakeFmhaDecodeRoute(
        target="sm100a",
        batch_size=16,
        q_len=1,
        num_q_heads=8,
        num_kv_heads=1,
        has_sink=False,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=False,
        component=component,
        page_size=16,
    )
    assert cake_api.cake_fmha_route_is_optimized(route)
    assert cake_api._route_components(route) == (component,)

    # The band admits every shape: short KV bounds and few work tiles included.
    for batch_size, seq_lens in ((1, [300]), (4, [512, 16, 4096, 1]), (8, [65536] * 8)):
        short = _select_balanced_kwargs(build(batch_size=batch_size, seq_lens=seq_lens))
        assert short is not None and short.component == component, (
            batch_size,
            seq_lens,
        )

    # q_len 1 only (no packed MTP tile), eight query heads per KV head only,
    # 16-token pages only: everything else stays on the quantized routes or
    # the compat kernel.
    for overrides in (
        dict(q_len=3),
        dict(q_len=8),
        dict(group=4),
        dict(group=7),
        dict(page_size=32),
        dict(batch_size=1025, seq_lens=[4096] * 1025),
    ):
        other = _select_balanced_kwargs(build(**overrides))
        assert other is None or other.component != component, overrides

    # Host scalars only: device-tensor scales, non-positive scales and every
    # extension the exported kernels do not serve fail closed.
    for mutation in (
        {"bmm1_scale": torch.ones(1, dtype=torch.float32)},
        {"bmm2_scale": torch.ones(1, dtype=torch.float32)},
        {"bmm1_scale": 0.0},
        {"bmm2_scale": 0.0},
        {"o_scale": 0.5},
        {"sinks": torch.zeros(8, dtype=torch.float32)},
        {"window_left": 1024},
        {"uses_shared_paged_kv_idx": False},
        {"kv_layout": "NHD"},
        {"lse": torch.empty((16, 8), dtype=torch.float32)},
        {
            "key_block_scales": torch.empty((4, 1, 16, 8), dtype=torch.float8_e4m3fn),
            "value_block_scales": torch.empty((4, 1, 16, 8), dtype=torch.float8_e4m3fn),
        },
        {"workspace_buffer": torch.empty(4096, dtype=torch.uint8)},
        {"multi_ctas_kv_counter_buffer": torch.zeros(64, dtype=torch.uint8)},
    ):
        mutated = {**kwargs, **mutation}
        selected = _select_balanced_kwargs(mutated)
        assert selected is None or selected.component != component, mutation


def test_cake_fmha_balanced_hd64_route_owns_bf16_gqa_decode(monkeypatch) -> None:
    component = "decode_balanced_bf16_hd64"
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    monkeypatch.setattr(cake_api, "_smallm_sm_count", lambda device: 148)

    def build(**overrides):
        params = dict(
            batch_size=16,
            q_len=1,
            num_kv_heads=1,
            seq_lens=_BALANCED_AGENTX,
            head_dim=64,
        )
        params.update(overrides)
        return _balanced_decode_kwargs(**params)

    kwargs = build()
    route = _select_balanced_kwargs(kwargs)
    assert route == cake_api.CakeFmhaDecodeRoute(
        target="sm100a",
        batch_size=16,
        q_len=1,
        num_q_heads=8,
        num_kv_heads=1,
        has_sink=False,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=False,
        component=component,
        page_size=16,
    )
    assert cake_api.cake_fmha_route_is_optimized(route)
    # Two structural instances of one body: eight softmax columns for head
    # groups 1..8, sixteen for 9..16 (selected by the host from the group).
    assert cake_api._route_components(route) == (
        component,
        "decode_balanced_bf16_hd64_g16",
    )

    # One to sixteen query heads per KV head, at every KV length and tile count.
    for group, batch_size, seq_lens in (
        (1, 16, _BALANCED_AGENTX),
        (5, 2, [300, 4097]),
        (16, 1, [65536]),
        (16, 4, [1, 16, 17, 8000]),
    ):
        selected = _select_balanced_kwargs(
            build(group=group, batch_size=batch_size, seq_lens=seq_lens)
        )
        assert selected is not None and selected.component == component, (
            group,
            batch_size,
        )
        assert selected.num_q_heads == group

    # No other product route serves head_dim 64: group 17, MTP draft rows,
    # other page sizes, a non-unit bmm2 scale and more than 1024 requests fall
    # through to the compat kernel.
    for overrides in (
        dict(group=17),
        dict(q_len=3),
        dict(q_len=8),
        dict(page_size=32),
        dict(bmm2_scale=2.0),
        dict(batch_size=1025, seq_lens=[4096] * 1025),
    ):
        assert _select_balanced_kwargs(build(**overrides)) is None, overrides
    for mutation in (
        {"sinks": torch.zeros(8, dtype=torch.float32)},
        {"window_left": 1024},
        {"bmm1_scale": torch.ones(1, dtype=torch.float32)},
        {"o_scale": 0.5},
        {"uses_shared_paged_kv_idx": False},
        {"kv_layout": "NHD"},
        {"lse": torch.empty((16, 8), dtype=torch.float32)},
        {"workspace_buffer": torch.empty(4096, dtype=torch.uint8)},
        {"multi_ctas_kv_counter_buffer": torch.zeros(64, dtype=torch.uint8)},
    ):
        assert _select_balanced_kwargs({**kwargs, **mutation}) is None, mutation


def test_cake_fmha_balanced_hd256_route_owns_bf16_row_tiles_per_page_size(
    monkeypatch,
) -> None:
    component = "decode_balanced_bf16_hd256"
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    monkeypatch.setattr(cake_api, "_smallm_sm_count", lambda device: 148)

    def build(**overrides):
        params = dict(
            batch_size=16,
            q_len=1,
            num_kv_heads=1,
            seq_lens=_BALANCED_AGENTX,
            head_dim=256,
            page_size=16,
        )
        params.update(overrides)
        return _balanced_decode_kwargs(**params)

    for page_size in (16, 32, 64):
        kwargs = build(page_size=page_size)
        route = _select_balanced_kwargs(kwargs)
        assert route == cake_api.CakeFmhaDecodeRoute(
            target="sm100a",
            batch_size=16,
            q_len=1,
            num_q_heads=8,
            num_kv_heads=1,
            has_sink=False,
            has_window=False,
            use_scale_ptr=False,
            retain_kv_l2=False,
            component=component,
            page_size=page_size,
        ), page_size
        assert cake_api.cake_fmha_route_is_optimized(route)
        assert cake_api._route_components(route) == (
            "decode_balanced_bf16_hd256_p16",
            "decode_balanced_bf16_hd256_p32",
            "decode_balanced_bf16_hd256_p64",
        )

    # Uniform q_len 1..8 as per-row tiles wherever the resident small-M
    # instance does not pack the rows (q_len * group <= 16, or more work tiles
    # than SMs).
    for q_len, group, batch_size, seq_lens in (
        (2, 8, 16, [4096] * 16),
        (4, 4, 4, [4097, 65536, 300, 33000]),
        (8, 2, 2, [8192, 60000]),
        (8, 8, 160, [2048] * 160),
        (1, 5, 2, [4097, 65536]),
    ):
        selected = _select_balanced_kwargs(
            build(q_len=q_len, group=group, batch_size=batch_size, seq_lens=seq_lens)
        )
        assert selected is not None and selected.component == component, (
            q_len,
            group,
            batch_size,
        )
    # The small-M route keeps its packed tiles (17 <= q_len * group <= 64)
    # ahead of this route.
    for q_len, group in ((8, 4), (4, 8), (3, 8)):
        packed = _select_balanced_kwargs(
            build(q_len=q_len, group=group, batch_size=16, seq_lens=[4096] * 16)
        )
        assert packed is not None, (q_len, group)
        assert packed.component == "decode_native_bf16_hd256_smallm", (q_len, group)

    # q_len 9, 128-token pages, group 9, a non-unit bmm2 scale and more than
    # 1024 requests fall through to the compat kernel.
    for overrides in (
        dict(q_len=9),
        dict(page_size=128),
        dict(group=9),
        dict(bmm2_scale=2.0),
        dict(batch_size=1025, seq_lens=[4096] * 1025),
    ):
        assert _select_balanced_kwargs(build(**overrides)) is None, overrides
    kwargs = build()
    for mutation in (
        {"sinks": torch.zeros(8, dtype=torch.float32)},
        {"window_left": 1024},
        {"bmm1_scale": torch.ones(1, dtype=torch.float32)},
        {"o_scale": 0.5},
        {"uses_shared_paged_kv_idx": False},
        {"kv_layout": "NHD"},
        {"lse": torch.empty((16, 8), dtype=torch.float32)},
        {"workspace_buffer": torch.empty(4096, dtype=torch.uint8)},
        {"multi_ctas_kv_counter_buffer": torch.zeros(64, dtype=torch.uint8)},
    ):
        assert _select_balanced_kwargs({**kwargs, **mutation}) is None, mutation


def test_cake_fmha_balanced_decode_buffer_sizing_is_shape_independent() -> None:
    from flashinfer.jit import cake_fmha as cake_jit

    # 148 SMs: 8 * 148 split tiles, 16 * 148 split items.
    assert cake_api._balanced_workspace_bounds(148) == (2368, 1184)
    # q_len 1: one u32 per split tile, then four 16-byte-aligned queue counters.
    assert cake_api.cake_fmha_balanced_counter_bytes(148, 1) == 1184 * 4 + 16
    # MTP: four counter words per split tile (arrivals, reduce queue A/B, published flag).
    assert cake_api.cake_fmha_balanced_counter_bytes(148, 8) == 1184 * 4 * 4 + 16
    assert cake_api.cake_fmha_balanced_counter_bytes(
        148
    ) == cake_api.cake_fmha_balanced_counter_bytes(148, 8)
    # Partials: FP32 O tile + statistics per split item, stats 256-byte aligned,
    # plus one reserved stats slot for the device planner's plan facts.
    assert (
        cake_api.cake_fmha_balanced_workspace_bytes(148, 1)
        == 2368 * 8 * 128 * 4 + (2368 + 1) * 16 * 4
    )
    assert (
        cake_api.cake_fmha_balanced_workspace_bytes(148, 4)
        == 2368 * 64 * 128 * 4 + (2368 + 1) * 128 * 4
    )
    # head_dim 64 (16 x 64 partial tile, 32 statistics) and head_dim 256
    # (8 x 256, 16 statistics) are q_len independent: every query row is its
    # own tile, so only the packed head_dim-128 MTP kernel needs four counter
    # words per split tile.
    assert (
        cake_api.cake_fmha_balanced_workspace_bytes(148, 1, head_dim=64)
        == 2368 * 16 * 64 * 4 + (2368 + 1) * 32 * 4
    )
    assert (
        cake_api.cake_fmha_balanced_workspace_bytes(148, 1, head_dim=256)
        == 2368 * 8 * 256 * 4 + (2368 + 1) * 16 * 4
    )
    assert cake_api.cake_fmha_balanced_workspace_bytes(
        148, 8, head_dim=256
    ) == cake_api.cake_fmha_balanced_workspace_bytes(148, 1, head_dim=256)
    assert cake_api.cake_fmha_balanced_workspace_bytes(
        148, 1, head_dim=128
    ) == cake_api.cake_fmha_balanced_workspace_bytes(148, 1)
    assert cake_api.cake_fmha_balanced_counter_bytes(148, 1, head_dim=64) == (
        1184 * 4 + 16
    )
    assert cake_api.cake_fmha_balanced_counter_bytes(
        148, 4, head_dim=256
    ) == cake_api.cake_fmha_balanced_counter_bytes(148, 1)
    assert cake_api.cake_fmha_balanced_counter_bytes(148, 4, head_dim=128) == (
        1184 * 4 * 4 + 16
    )
    for q_len, n_rows, component in (
        (1, 0, "decode_balanced_bf16"),
        (3, 32, "decode_balanced_bf16_mtp_n32"),
        (4, 32, "decode_balanced_bf16_mtp_n32"),
        (5, 64, "decode_balanced_bf16_mtp_n64"),
        (8, 64, "decode_balanced_bf16_mtp_n64"),
    ):
        assert cake_jit.cake_fmha_balanced_n_rows(q_len) == n_rows
        assert cake_jit.cake_fmha_balanced_component_name(q_len) == component
        assert cake_jit.cake_fmha_balanced_component_name(
            q_len, "fp16"
        ) == component.replace("bf16", "fp16")
    with pytest.raises(ValueError):
        cake_jit.cake_fmha_balanced_n_rows(2)
    with pytest.raises(ValueError):
        cake_jit.cake_fmha_balanced_component_name(1, "fp8")


@pytest.mark.parametrize("dtype_name", ["bf16", "fp16"])
def test_cake_fmha_decode_balanced_jit_selects_the_q_len_component(
    monkeypatch, dtype_name
) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    for q_len, suffix, n_rows in ((1, "", 0), (4, "_mtp_n32", 32), (7, "_mtp_n64", 64)):
        component = f"decode_balanced_{dtype_name}{suffix}"
        spec = gen_cake_fmha_decode_balanced_module("sm100a", q_len, dtype_name)
        assert spec.name == get_cake_fmha_decode_balanced_uri(
            "sm100a", q_len, dtype_name
        )
        if dtype_name == "bf16":
            assert spec is gen_cake_fmha_decode_balanced_bf16_module("sm100a", q_len)
            assert spec.name == get_cake_fmha_decode_balanced_bf16_uri("sm100a", q_len)
        assert {Path(source).name for source in spec.sources} == {
            "default.cu",
            f"cake_fmha_{component}_binding.cu",
            "cake_fmha_decode_balanced_jit_binding.cu",
        }
        assert f"-DQ_LEN={q_len}" in spec.extra_cuda_cflags
        assert f"-DCAKE_FMHA_BALANCED_N_ROWS={n_rows}" in spec.extra_cuda_cflags
        assert (
            f"-DCAKE_FMHA_BALANCED_FP16={int(dtype_name == 'fp16')}"
            in spec.extra_cuda_cflags
        )
        assert (
            f"-DCAKE_FMHA_BALANCED_LAUNCH=cake_fmha_launch_{component}"
            in spec.extra_cuda_cflags
        )
        folded = "decode_balanced_16bit" if n_rows == 0 else "decode_balanced_mtp"
        assert any(
            f"/cuda/{folded}/default.cu" in str(source) for source in spec.sources
        )


@pytest.mark.parametrize(
    ("q_dtype", "q_dtype_flag"), [("fp8", 0), ("bf16q", 1), ("fp16q", 2)]
)
def test_cake_fmha_decode_balanced_fp8_family_jit_selects_the_q_dtype_component(
    monkeypatch, q_dtype, q_dtype_flag
) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    component = f"decode_balanced_{q_dtype}"
    assert CAKE_FMHA_BALANCED_FP8_Q_DTYPES == ("fp8", "bf16q", "fp16q")
    assert cake_fmha_balanced_fp8_component_name(q_dtype) == component
    spec = gen_cake_fmha_decode_balanced_fp8_module("sm100a", q_dtype)
    assert spec.name == get_cake_fmha_decode_balanced_fp8_uri("sm100a", q_dtype)
    assert spec.name.startswith(f"cake_fmha_{component}_sm100a_")
    assert {Path(source).name for source in spec.sources} == {
        "default.cu",
        f"cake_fmha_{component}_binding.cu",
        "cake_fmha_decode_balanced_fp8_jit_binding.cu",
    }
    folded = "decode_balanced_fp8" if q_dtype == "fp8" else "decode_balanced_q16bit"
    assert any(f"/cuda/{folded}/default.cu" in str(source) for source in spec.sources)
    assert f"-DCAKE_FMHA_BALANCED_Q_DTYPE={q_dtype_flag}" in spec.extra_cuda_cflags
    assert (
        f"-DCAKE_FMHA_BALANCED_LAUNCH=cake_fmha_launch_{component}"
        in spec.extra_cuda_cflags
    )
    for bad_q_dtype in ("bf16", "fp16", "nvfp4"):
        with pytest.raises(ValueError):
            cake_fmha_balanced_fp8_component_name(bad_q_dtype)


@pytest.mark.parametrize("max_group", [8, 16])
def test_cake_fmha_decode_balanced_hd64_jit_selects_the_component(
    monkeypatch, max_group
) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    component = (
        "decode_balanced_bf16_hd64"
        if max_group == 8
        else "decode_balanced_bf16_hd64_g16"
    )
    spec = gen_cake_fmha_decode_balanced_hd64_module("sm100a", max_group)
    assert spec.name == get_cake_fmha_decode_balanced_hd64_uri("sm100a", max_group)
    assert spec.name.startswith(f"cake_fmha_{component}_sm100a_")
    assert {Path(source).name for source in spec.sources} == {
        "default.cu",
        f"cake_fmha_{component}_binding.cu",
        "cake_fmha_decode_balanced_hd64_jit_binding.cu",
    }
    assert any(
        "/cuda/decode_balanced_hd64/default.cu" in str(source)
        for source in spec.sources
    )
    assert f"-DCAKE_FMHA_BALANCED_MAX_GROUPS={max_group}" in spec.extra_cuda_cflags
    assert (
        f"-DCAKE_FMHA_BALANCED_LAUNCH=cake_fmha_launch_{component}"
        in spec.extra_cuda_cflags
    )


@pytest.mark.parametrize("page_size", [16, 32, 64])
def test_cake_fmha_decode_balanced_hd256_jit_selects_the_page_size_component(
    monkeypatch, page_size
) -> None:
    import flashinfer.jit.core as jit_core

    monkeypatch.setattr(jit_core, "check_cuda_arch", lambda: None)
    component = f"decode_balanced_bf16_hd256_p{page_size}"
    assert CAKE_FMHA_BALANCED_HD256_PAGE_SIZES == (16, 32, 64)
    assert cake_fmha_balanced_hd256_component_name(page_size) == component
    spec = gen_cake_fmha_decode_balanced_hd256_module("sm100a", page_size)
    assert spec.name == get_cake_fmha_decode_balanced_hd256_uri("sm100a", page_size)
    assert spec.name.startswith(f"cake_fmha_{component}_sm100a_")
    assert {Path(source).name for source in spec.sources} == {
        "default.cu",
        f"cake_fmha_{component}_binding.cu",
        "cake_fmha_decode_balanced_hd256_jit_binding.cu",
    }
    assert any(
        "/cuda/decode_balanced_hd256/default.cu" in str(source)
        for source in spec.sources
    )
    assert f"-DCAKE_FMHA_BALANCED_PAGE_SIZE={page_size}" in spec.extra_cuda_cflags
    assert (
        f"-DCAKE_FMHA_BALANCED_LAUNCH=cake_fmha_launch_{component}"
        in spec.extra_cuda_cflags
    )
    for bad_page_size in (8, 128):
        with pytest.raises(ValueError):
            cake_fmha_balanced_hd256_component_name(bad_page_size)


def _balanced_reference_decode(
    query, kv_cache, block_tables, seq_lens, *, q_len, scale, page_size=16
):
    """FP32 bottom-right-causal GQA decode over the paged HND cache (row j of a
    request attends to ``S - (q_len - 1 - j)`` positions).  ``head_dim`` follows
    the tensors; E4M3 inputs are read at their stored (unscaled) values."""

    batch = int(seq_lens.numel())
    num_q_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = kv_cache.shape[2]
    group = num_q_heads // num_kv_heads
    out = torch.empty(query.shape, dtype=torch.float32, device=query.device)
    for b in range(batch):
        length = int(seq_lens[b])
        pages = block_tables[b, : (length + page_size - 1) // page_size]
        k = (
            kv_cache[pages, 0]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_dim)[:, :length]
            .float()
        )
        v = (
            kv_cache[pages, 1]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_dim)[:, :length]
            .float()
        )
        for j in range(q_len):
            visible = length - (q_len - 1 - j)
            q = query[b * q_len + j].float().view(num_kv_heads, group, head_dim)
            logits = torch.einsum("hgd,hnd->hgn", q, k[:, :visible]) * scale
            probs = torch.softmax(logits, dim=-1)
            out[b * q_len + j] = torch.einsum(
                "hgn,hnd->hgd", probs, v[:, :visible]
            ).reshape(num_q_heads, head_dim)
    return out


def _balanced_case(
    device,
    *,
    q_len,
    num_kv_heads,
    seq_lens,
    seed,
    dtype=torch.bfloat16,
    head_dim=128,
    page_size=16,
    group=8,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = len(seq_lens)
    num_q_heads = group * num_kv_heads
    pages_per_req = [(s + page_size - 1) // page_size for s in seq_lens]
    total_pages = sum(pages_per_req)
    query = torch.randn((batch * q_len, num_q_heads, head_dim), generator=generator).to(
        device=device, dtype=dtype
    )
    kv_cache = torch.randn(
        (total_pages, 2, num_kv_heads, page_size, head_dim), generator=generator
    ).to(device=device, dtype=dtype)
    max_pages = max(pages_per_req)
    block_tables = torch.zeros((batch, max_pages), dtype=torch.int32)
    order = torch.randperm(total_pages, generator=generator)
    cursor = 0
    for b, n in enumerate(pages_per_req):
        block_tables[b, :n] = order[cursor : cursor + n]
        cursor += n
    return (
        query,
        kv_cache,
        block_tables.to(device),
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
    )


def _e4m3_quantize(tensor):
    """Per-tensor E4M3 quantization with an amax scale: ``(quantized, scale)``."""

    scale = float(tensor.float().abs().amax()) / 448.0
    quantized = (tensor.float() / scale).clamp_(-448.0, 448.0)
    return quantized.to(torch.float8_e4m3fn), scale


def _balanced_shorter_lengths(seq_lens, q_len):
    """A second length vector under the same captured KV bound: the longest
    request keeps its length, every other request loses a third."""

    shorter = [max(q_len, s - (s // 3)) for s in seq_lens]
    longest = max(range(len(seq_lens)), key=seq_lens.__getitem__)
    shorter[longest] = seq_lens[longest]
    return shorter


def _select_balanced_route(
    device,
    *,
    query,
    kv_cache,
    out,
    workspace,
    block_tables,
    seq_lens_t,
    q_len,
    max_seq_len,
    bmm1_scale,
    bmm2_scale=1.0,
):
    return cake_api.select_cake_fmha_decode_route(
        device,
        query=query,
        key_cache=kv_cache[:, 0],
        value_cache=kv_cache[:, 1],
        out=out,
        workspace_buffer=workspace,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        batch_size=int(seq_lens_t.numel()),
        q_len=q_len,
        max_seq_len=max_seq_len,
        window_left=-1,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
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


def _check_balanced_graph_replay(
    run,
    *,
    seq_lens,
    seq_lens_t,
    q_len,
    expected,
    expected_short,
    dequant,
    atol,
    rtol,
):
    """Capture ``run`` in a CUDA graph and replay it under a shorter length
    vector with the same KV bound: the plan is derived on device, so the
    captured launch must follow the new lengths without a re-plan or a memset."""

    device = seq_lens_t.device
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replayed = run()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(dequant(replayed), expected, atol=atol, rtol=rtol)
    shorter = _balanced_shorter_lengths(seq_lens, q_len)
    seq_lens_t.copy_(torch.tensor(shorter, dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(dequant(replayed), expected_short, atol=atol, rtol=rtol)
    seq_lens_t.copy_(torch.tensor(seq_lens, dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(dequant(replayed), expected, atol=atol, rtol=rtol)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)
@pytest.mark.parametrize(
    ("q_len", "num_kv_heads", "seq_lens"),
    [
        (1, 1, _BALANCED_AGENTX),
        (1, 4, [65536] * 4 + [4097] * 4 + [33000] * 4 + [512] * 4),
        (4, 1, [40004, 32771, 36000, 33000, 45000, 34000, 50000, 60000, 38000, 4100]),
        (7, 2, [65536, 32775, 40000, 4103, 12345, 50000, 33000, 70000]),
    ],
)
@pytest.mark.parametrize("dtype_name", ["bf16", "fp16"])
def test_cake_decode_balanced_route_matches_reference_and_replays(
    q_len, num_kv_heads, seq_lens, dtype_name
) -> None:
    device = torch.device("cuda")
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    query, kv_cache, block_tables, seq_lens_t = _balanced_case(
        device,
        q_len=q_len,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        seed=685_000 + q_len,
        dtype=torch_dtype,
    )
    scale = 128**-0.5
    max_seq_len = max(seq_lens)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    route = cake_api.select_cake_fmha_decode_route(
        device,
        query=query,
        key_cache=kv_cache[:, 0],
        value_cache=kv_cache[:, 1],
        out=torch.empty_like(query),
        workspace_buffer=workspace,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        batch_size=len(seq_lens),
        q_len=q_len,
        max_seq_len=max_seq_len,
        window_left=-1,
        bmm1_scale=scale,
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
    assert route is not None and route.component == f"decode_balanced_{dtype_name}"

    def run():
        return cake_api.cake_batch_decode_with_kv_cache(
            query,
            kv_cache,
            workspace,
            block_tables,
            seq_lens_t,
            max_seq_len,
            bmm1_scale=scale,
            bmm2_scale=1.0,
            window_left=-1,
            kv_layout="HND",
            q_len_per_req=q_len,
        )

    actual = run()
    expected = _balanced_reference_decode(
        query, kv_cache, block_tables, seq_lens_t, q_len=q_len, scale=scale
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)

    # Graph replay with a different length vector under the same KV bound:
    # the plan is derived on device, so the captured launch must follow the
    # new lengths without a re-plan or a memset.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replayed = run()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed.float(), expected.float(), atol=1e-2, rtol=1e-2)
    shorter = [max(q_len, s - (s // 3)) for s in seq_lens]
    longest = max(range(len(seq_lens)), key=seq_lens.__getitem__)
    shorter[longest] = seq_lens[longest]  # keep the captured KV bound
    seq_lens_t.copy_(torch.tensor(shorter, dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()
    expected_short = _balanced_reference_decode(
        query, kv_cache, block_tables, seq_lens_t, q_len=q_len, scale=scale
    )
    torch.testing.assert_close(
        replayed.float(), expected_short.float(), atol=1e-2, rtol=1e-2
    )
    seq_lens_t.copy_(torch.tensor(seq_lens, dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)
@pytest.mark.parametrize(
    ("num_kv_heads", "seq_lens"),
    [
        (1, _BALANCED_AGENTX),
        (4, [65536] * 4 + [4097] * 4 + [33000] * 4 + [512] * 4),
    ],
)
@pytest.mark.parametrize("q_dtype_name", ["fp8", "bf16q", "fp16q"])
def test_cake_decode_balanced_fp8_family_matches_reference_and_replays(
    q_dtype_name, num_kv_heads, seq_lens
) -> None:
    device = torch.device("cuda")
    query, kv_cache, block_tables, seq_lens_t = _balanced_case(
        device,
        q_len=1,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        seed=685_200 + num_kv_heads,
        dtype=torch.bfloat16,
    )
    # Per-tensor E4M3 quantization with amax scales.  The reference reads the
    # same E4M3 values, so only the kernel arithmetic is under test.
    key_q, k_scale = _e4m3_quantize(kv_cache[:, 0])
    value_q, v_scale = _e4m3_quantize(kv_cache[:, 1])
    kv_cache_q = torch.empty(kv_cache.shape, dtype=torch.float8_e4m3fn, device=device)
    kv_cache_q[:, 0] = key_q
    kv_cache_q[:, 1] = value_q
    del key_q, value_q, kv_cache
    if q_dtype_name == "fp8":
        query_q, q_scale = _e4m3_quantize(query)
    else:
        query_q = query.to(torch.bfloat16 if q_dtype_name == "bf16q" else torch.float16)
        q_scale = 1.0
    bmm1_scale = q_scale * k_scale / 128**0.5
    max_seq_len = max(seq_lens)
    kv_cache_f32 = kv_cache_q.float()
    expected = (
        _balanced_reference_decode(
            query_q, kv_cache_f32, block_tables, seq_lens_t, q_len=1, scale=bmm1_scale
        )
        * v_scale
    )
    expected_short = (
        _balanced_reference_decode(
            query_q,
            kv_cache_f32,
            block_tables,
            torch.tensor(
                _balanced_shorter_lengths(seq_lens, 1), dtype=torch.int32, device=device
            ),
            q_len=1,
            scale=bmm1_scale,
        )
        * v_scale
    )
    del kv_cache_f32
    if q_dtype_name == "fp8":
        # E4M3 output: one output scale must cover both length vectors the
        # graph replays.
        o_scale = (
            max(float(expected.abs().amax()), float(expected_short.abs().amax()))
            / 448.0
        )
        atol = rtol = 0.1
    else:
        o_scale = 1.0
        atol = rtol = 2e-2
    bmm2_scale = v_scale / o_scale
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    out = torch.empty_like(query_q)
    route = _select_balanced_route(
        device,
        query=query_q,
        kv_cache=kv_cache_q,
        out=out,
        workspace=workspace,
        block_tables=block_tables,
        seq_lens_t=seq_lens_t,
        q_len=1,
        max_seq_len=max_seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
    )
    assert route is not None and route.component == f"decode_balanced_{q_dtype_name}"

    def dequant(tensor):
        return tensor.float() * o_scale

    def run():
        return cake_api.cake_batch_decode_with_kv_cache(
            query_q,
            kv_cache_q,
            workspace,
            block_tables,
            seq_lens_t,
            max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=-1,
            kv_layout="HND",
            q_len_per_req=1,
            out=out,
        )

    torch.testing.assert_close(dequant(run()), expected, atol=atol, rtol=rtol)
    _check_balanced_graph_replay(
        run,
        seq_lens=seq_lens,
        seq_lens_t=seq_lens_t,
        q_len=1,
        expected=expected,
        expected_short=expected_short,
        dequant=dequant,
        atol=atol,
        rtol=rtol,
    )


_FP8_FULL_BLOCK_SEQ_LENS = [
    # The smallest admitted shape: four 128-token blocks, one split.
    pytest.param([512, 512, 512, 512], id="512x4"),
    # Eight blocks, every length a multiple of 256: the binding splits each
    # (request, KV head) tile in two and ``decode_quant_fp8_reduce`` combines
    # the partials.
    pytest.param([1024, 1024, 1024, 1024], id="1024x4-split2"),
    # Sixteen blocks: four partials per tile.
    pytest.param([2048, 2048, 2048, 2048], id="2048x4-split4"),
]


def _fp8_full_block_decode_case(device, *, seq_lens, page_size, seed):
    """E4M3 decode inputs for the optimized full-block FP8 route.

    One BF16 draw is quantized per tensor with amax scales (``_e4m3_quantize``).
    The FP32 reference reads the stored E4M3 query / K / V values with
    ``bmm1_scale = q_scale * k_scale / sqrt(128)`` and multiplies by
    ``v_scale``, so only the kernel arithmetic is under test.  The output is
    E4M3 too: ``o_scale = amax(expected) / 448`` maps the largest reference
    value to the E4M3 maximum and ``bmm2_scale = v_scale / o_scale`` is the
    kernel's output scale.  ``query_bf16`` / ``kv_cache_bf16`` hold the
    dequantized tensors for FlashInfer's BF16 decode reference.
    """

    query, kv_cache, block_tables, seq_lens_t = _balanced_case(
        device,
        q_len=1,
        num_kv_heads=2,
        seq_lens=seq_lens,
        seed=seed,
        dtype=torch.bfloat16,
        page_size=page_size,
    )
    query_q, q_scale = _e4m3_quantize(query)
    key_q, k_scale = _e4m3_quantize(kv_cache[:, 0])
    value_q, v_scale = _e4m3_quantize(kv_cache[:, 1])
    kv_cache_q = torch.empty(kv_cache.shape, dtype=torch.float8_e4m3fn, device=device)
    kv_cache_q[:, 0] = key_q
    kv_cache_q[:, 1] = value_q
    kv_cache_bf16 = torch.empty_like(kv_cache)
    kv_cache_bf16[:, 0] = key_q.float() * k_scale
    kv_cache_bf16[:, 1] = value_q.float() * v_scale
    bmm1_scale = q_scale * k_scale / 128**0.5
    expected = (
        _balanced_reference_decode(
            query_q,
            kv_cache_q.float(),
            block_tables,
            seq_lens_t,
            q_len=1,
            scale=bmm1_scale,
            page_size=page_size,
        )
        * v_scale
    )
    amax = float(expected.abs().amax())
    o_scale = amax / 448.0
    return SimpleNamespace(
        query=query_q,
        kv_cache=kv_cache_q,
        block_tables=block_tables,
        seq_lens=list(seq_lens),
        seq_lens_t=seq_lens_t,
        bmm1_scale=bmm1_scale,
        bmm2_scale=v_scale / o_scale,
        o_scale=o_scale,
        expected=expected,
        # Output E4M3 rounding is at most 1/16 of each value (three mantissa
        # bits) and 16/448 of ``amax`` in the top binade; the E4M3 P tile of
        # the PV MMA adds a few percent of the output scale.  ``rtol=0.1`` is
        # the balanced E4M3 family's bound; the absolute term follows the
        # output quantization grid instead of that test's fixed 0.1.
        atol=0.1 * amax,
        query_bf16=(query_q.float() * q_scale).to(torch.bfloat16),
        kv_cache_bf16=kv_cache_bf16,
    )


def _paged_decode_wrapper_reference(
    query, kv_cache, block_tables, seq_lens, *, page_size, scale
):
    """FlashInfer's BF16 tensor-core paged decode over the same page tables."""

    device = query.device
    pages_per_req = [(length + page_size - 1) // page_size for length in seq_lens]
    indptr = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(torch.tensor(pages_per_req, dtype=torch.int32), dim=0)
    indices = torch.cat(
        [block_tables[b, :pages] for b, pages in enumerate(pages_per_req)]
    )
    last_page_len = torch.tensor(
        [
            seq_lens[b] - (pages_per_req[b] - 1) * page_size
            for b in range(len(seq_lens))
        ],
        dtype=torch.int32,
    )
    wrapper = decode.BatchDecodeWithPagedKVCacheWrapper(
        torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        "HND",
        use_tensor_cores=True,
    )
    wrapper.plan(
        indptr.to(device),
        indices,
        last_page_len.to(device),
        query.shape[1],
        kv_cache.shape[2],
        query.shape[2],
        page_size,
        pos_encoding_mode="NONE",
        kv_data_type=torch.bfloat16,
        q_data_type=torch.bfloat16,
        sm_scale=scale,
    )
    return wrapper.run(query, kv_cache)


def _check_fp8_full_block_decode(device, *, seq_lens, page_size, device_scale):
    case = _fp8_full_block_decode_case(
        device, seq_lens=seq_lens, page_size=page_size, seed=685_300 + page_size
    )
    max_seq_len = max(seq_lens)
    if device_scale:
        bmm1_scale = torch.tensor([case.bmm1_scale], dtype=torch.float32, device=device)
        bmm2_scale = torch.tensor([case.bmm2_scale], dtype=torch.float32, device=device)
    else:
        bmm1_scale, bmm2_scale = case.bmm1_scale, case.bmm2_scale
    # Below the balanced E4M3 route's workspace bound (about ten MiB), so the
    # page-16 host-scale shape also reaches the quantized route; the quantized
    # route's split-KV partials need a few hundred KiB at these shapes.
    workspace = torch.empty(4 * 1024 * 1024, dtype=torch.uint8, device=device)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    assert workspace.numel() < cake_api.cake_fmha_balanced_workspace_bytes(sm_count, 1)
    out = torch.empty_like(case.query)
    route = _select_balanced_route(
        device,
        query=case.query,
        kv_cache=case.kv_cache,
        out=out,
        workspace=workspace,
        block_tables=case.block_tables,
        seq_lens_t=case.seq_lens_t,
        q_len=1,
        max_seq_len=max_seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
    )
    assert route is not None and route.component == "decode_quant_fp8"
    assert (route.page_size, route.use_scale_ptr) == (page_size, device_scale)
    assert (route.batch_size, route.num_q_heads, route.num_kv_heads) == (4, 16, 2)
    assert cake_api.cake_fmha_route_is_optimized(route)
    assert cake_api._route_components(route) == (
        "decode_quant_fp8",
        "decode_quant_fp8_reduce",
    )
    _, optimized_loaded = cake_api._resolve_cake_fmha_decode_module(device, route)
    assert optimized_loaded

    actual = cake_api.cake_batch_decode_with_kv_cache(
        case.query,
        case.kv_cache,
        workspace,
        case.block_tables,
        case.seq_lens_t,
        max_seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        window_left=-1,
        kv_layout="HND",
        q_len_per_req=1,
        out=out,
    )
    assert actual.dtype == torch.float8_e4m3fn and actual.data_ptr() == out.data_ptr()
    dequant = actual.float() * case.o_scale
    torch.testing.assert_close(dequant, case.expected, atol=case.atol, rtol=0.1)

    # FlashInfer's BF16 decode over the dequantized cache agrees with the FP32
    # definition (BF16 inputs and output) and brackets the E4M3 result.
    reference = _paged_decode_wrapper_reference(
        case.query_bf16,
        case.kv_cache_bf16,
        case.block_tables,
        seq_lens,
        page_size=page_size,
        scale=128**-0.5,
    ).float()
    torch.testing.assert_close(reference, case.expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(dequant, reference, atol=case.atol, rtol=0.1)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)
@pytest.mark.parametrize("seq_lens", _FP8_FULL_BLOCK_SEQ_LENS)
@pytest.mark.parametrize("page_size", [16, 32], ids=["page16", "page32"])
@pytest.mark.parametrize(
    "device_scale", [False, True], ids=["host-scale", "device-scale"]
)
def test_cake_decode_fp8_full_block_route_matches_reference(
    seq_lens, page_size, device_scale
) -> None:
    """E4M3 query, KV cache and output on the optimized full-block FP8 route.

    The route (``decode_quant_fp8`` plus ``decode_quant_fp8_reduce``) owns
    HND E4M3 decode with a shared page table, eight query heads per KV head,
    16- or 32-token pages and KV lengths that are whole 128-token blocks, at
    least 512 tokens and in one evened block bucket.  Host scalar and device
    tensor scales are both admitted; the balanced E4M3 route takes the
    page-16 host-scale shape only when its larger workspace is available.
    """

    _check_fp8_full_block_decode(
        torch.device("cuda"),
        seq_lens=seq_lens,
        page_size=page_size,
        device_scale=device_scale,
    )


_FP8_RUNTIME_LENGTH_CASES = [
    # Five- and six-block requests share the evened six-block bucket: the
    # five-block request's padding block is masked token by token.
    pytest.param([640, 768, 768, 640], 768, 16, id="odd-blocks"),
    pytest.param([512, 300, 1, 777], 777, 32, id="nonuniform-tails"),
    # Eight tiles below the SM count split; a zero-length request's tiles are
    # empty (partial max -inf, sum 0) and its output row is zero.
    pytest.param([0, 513, 0, 2048], 2048, 16, id="zero-lengths-split"),
    pytest.param([0, 0, 0, 0], 1024, 16, id="all-zero"),
    pytest.param([5, 17, 129, 3], 8192, 32, id="short-large-max"),
    pytest.param([4095, 1, 2048, 4096], 4096, 16, id="split-tails"),
]


def _fp8_runtime_length_case(device, *, seq_lens, max_seq_len, page_size, seed):
    """Every request owns ``max_seq_len`` tokens of pages (finite E4M3 data);
    ``seq_lens`` (at most ``max_seq_len``) select the attended prefix."""

    case = _fp8_full_block_decode_case(
        device, seq_lens=[max_seq_len] * len(seq_lens), page_size=page_size, seed=seed
    )
    v_scale = case.bmm2_scale * case.o_scale
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    expected = (
        _balanced_reference_decode(
            case.query,
            case.kv_cache.float(),
            case.block_tables,
            seq_lens_t,
            q_len=1,
            scale=case.bmm1_scale,
            page_size=page_size,
        )
        * v_scale
    )
    amax = float(expected.abs().amax())
    o_scale = amax / 448.0 if amax > 0 else 1.0
    case.seq_lens, case.seq_lens_t = list(seq_lens), seq_lens_t
    case.expected, case.o_scale, case.atol = expected, o_scale, 0.1 * amax
    case.bmm2_scale = v_scale / o_scale
    return case


def _run_fp8_runtime_length(case, *, max_seq_len, block_tables=None, out=None):
    device = case.query.device
    workspace = torch.zeros(4 * 1024 * 1024, dtype=torch.uint8, device=device)
    out = torch.empty_like(case.query) if out is None else out
    return cake_api.cake_batch_decode_with_kv_cache(
        case.query,
        case.kv_cache,
        workspace,
        case.block_tables if block_tables is None else block_tables,
        case.seq_lens_t,
        max_seq_len,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        window_left=-1,
        kv_layout="HND",
        q_len_per_req=1,
        out=out,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)
@pytest.mark.parametrize(
    ("seq_lens", "max_seq_len", "page_size"), _FP8_RUNTIME_LENGTH_CASES
)
def test_cake_decode_fp8_runtime_lengths_match_reference(
    seq_lens, max_seq_len, page_size
) -> None:
    """Non-uniform, odd-block, tail, zero and short lengths under one host
    ``max_seq_len``: the route never reads ``seq_lens``; zero-length requests
    write zero bytes and the rest match the FP32 reference; reruns are bitwise."""

    device = torch.device("cuda")
    case = _fp8_runtime_length_case(
        device,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        page_size=page_size,
        seed=685_400 + page_size,
    )
    workspace = torch.zeros(4 * 1024 * 1024, dtype=torch.uint8, device=device)
    route = _select_balanced_route(
        device,
        query=case.query,
        kv_cache=case.kv_cache,
        out=torch.empty_like(case.query),
        workspace=workspace,
        block_tables=case.block_tables,
        seq_lens_t=case.seq_lens_t,
        q_len=1,
        max_seq_len=max_seq_len,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
    )
    assert route is not None and route.component == "decode_quant_fp8"
    actual = _run_fp8_runtime_length(case, max_seq_len=max_seq_len)
    again = _run_fp8_runtime_length(case, max_seq_len=max_seq_len)
    assert torch.equal(actual.view(torch.uint8), again.view(torch.uint8))
    for b, length in enumerate(seq_lens):
        if length == 0:
            assert int(actual[b].view(torch.uint8).count_nonzero()) == 0
    live = [b for b, length in enumerate(seq_lens) if length > 0]
    if live:
        dequant = actual[live].float() * case.o_scale
        torch.testing.assert_close(
            dequant, case.expected[live], atol=case.atol, rtol=0.1
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)
@pytest.mark.parametrize("page_size", [16, 32], ids=["page16", "page32"])
def test_cake_decode_fp8_runtime_lengths_ignore_page_table_past_lengths(
    page_size,
) -> None:
    """Page-table entries past each request's pages (other pages, negative or
    out-of-range ids) never change the output bytes."""

    device = torch.device("cuda")
    seq_lens, max_seq_len = [0, 300, 1, 1500], 2048
    case = _fp8_runtime_length_case(
        device,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        page_size=page_size,
        seed=685_500 + page_size,
    )
    clean = _run_fp8_runtime_length(case, max_seq_len=max_seq_len)
    garbage = case.block_tables.clone()
    generator = torch.Generator(device="cpu").manual_seed(7)
    for b, length in enumerate(seq_lens):
        used = (length + page_size - 1) // page_size
        junk = torch.randint(
            -(2**20),
            2**30,
            (garbage.shape[1] - used,),
            generator=generator,
            dtype=torch.int32,
        )
        garbage[b, used:] = junk.to(device)
    noisy = _run_fp8_runtime_length(case, max_seq_len=max_seq_len, block_tables=garbage)
    assert torch.equal(clean.view(torch.uint8), noisy.view(torch.uint8))


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)
def test_cake_decode_fp8_runtime_lengths_cuda_graph_replay() -> None:
    """A graph captured under one ``max_seq_len`` replays new lengths: the
    plan reads no device value, so the replay equals an eager call."""

    device = torch.device("cuda")
    max_seq_len, page_size = 2048, 16
    case = _fp8_runtime_length_case(
        device,
        seq_lens=[2048, 1024, 640, 2048],
        max_seq_len=max_seq_len,
        page_size=page_size,
        seed=685_600,
    )
    out = torch.empty_like(case.query)
    _run_fp8_runtime_length(
        case, max_seq_len=max_seq_len, out=out
    )  # prewarm the binding's descriptors
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_fp8_runtime_length(case, max_seq_len=max_seq_len, out=out)
    case.seq_lens_t.copy_(
        torch.tensor([0, 513, 2047, 1], dtype=torch.int32, device=device)
    )
    graph.replay()
    torch.cuda.synchronize()
    replayed = out.clone()
    eager = _run_fp8_runtime_length(case, max_seq_len=max_seq_len)
    assert torch.equal(replayed.view(torch.uint8), eager.view(torch.uint8))
    assert int(replayed[0].view(torch.uint8).count_nonzero()) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)
@pytest.mark.parametrize("page_size", [16, 32], ids=["page16", "page32"])
def test_cake_decode_fp8_fixed_split_matches_runtime_plan(
    monkeypatch, page_size
) -> None:
    """The fixed 8 x 16 split-plan member (decode -> reducer PDL) writes the runtime-plan
    member's bytes, also for zero, short and tail lengths, and matches the FP32 reference."""

    device = torch.device("cuda")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    seq_lens, max_seq_len = [16384, 1, 0, 16000, 8193, 300, 16384, 12345], 16384
    if not cake_api._decode_quant_fp8_fixed_split(
        len(seq_lens), 2, max_seq_len, sm_count
    ):
        pytest.skip(f"the host split plan is not 8 x 16 on {sm_count} SMs")
    case = _fp8_runtime_length_case(
        device,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        page_size=page_size,
        seed=685_700 + page_size,
    )
    route_args = dict(
        query=case.query,
        kv_cache=case.kv_cache,
        out=torch.empty_like(case.query),
        workspace=torch.zeros(4 * 1024 * 1024, dtype=torch.uint8, device=device),
        block_tables=case.block_tables,
        seq_lens_t=case.seq_lens_t,
        q_len=1,
        max_seq_len=max_seq_len,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
    )
    route = _select_balanced_route(device, **route_args)
    assert (
        route is not None
        and route.component == "decode_quant_fp8"
        and route.num_split == 8
    )
    fixed = _run_fp8_runtime_length(case, max_seq_len=max_seq_len)
    again = _run_fp8_runtime_length(case, max_seq_len=max_seq_len)
    assert torch.equal(fixed.view(torch.uint8), again.view(torch.uint8))
    monkeypatch.setattr(cake_api, "_decode_quant_fp8_fixed_split", lambda *args: False)
    route = _select_balanced_route(device, **route_args)
    assert (
        route is not None
        and route.component == "decode_quant_fp8"
        and route.num_split == 0
    )
    runtime = _run_fp8_runtime_length(case, max_seq_len=max_seq_len)
    assert torch.equal(fixed.view(torch.uint8), runtime.view(torch.uint8))
    assert int(fixed[2].view(torch.uint8).count_nonzero()) == 0
    live = [b for b, length in enumerate(seq_lens) if length > 0]
    dequant = fixed[live].float() * case.o_scale
    torch.testing.assert_close(dequant, case.expected[live], atol=case.atol, rtol=0.1)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)
@pytest.mark.parametrize(
    ("num_kv_heads", "group", "seq_lens"),
    [
        (1, 8, _BALANCED_AGENTX),
        (2, 16, [4096, 300, 65536, 1, 777, 12345]),
        (4, 4, [43, 200, 512, 1, 8000, 3000]),
    ],
)
def test_cake_decode_balanced_hd64_matches_reference_and_replays(
    num_kv_heads, group, seq_lens
) -> None:
    device = torch.device("cuda")
    query, kv_cache, block_tables, seq_lens_t = _balanced_case(
        device,
        q_len=1,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        seed=685_300 + group,
        head_dim=64,
        group=group,
    )
    scale = 64**-0.5
    max_seq_len = max(seq_lens)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    route = _select_balanced_route(
        device,
        query=query,
        kv_cache=kv_cache,
        out=torch.empty_like(query),
        workspace=workspace,
        block_tables=block_tables,
        seq_lens_t=seq_lens_t,
        q_len=1,
        max_seq_len=max_seq_len,
        bmm1_scale=scale,
    )
    assert route is not None and route.component == "decode_balanced_bf16_hd64"

    def run():
        return cake_api.cake_batch_decode_with_kv_cache(
            query,
            kv_cache,
            workspace,
            block_tables,
            seq_lens_t,
            max_seq_len,
            bmm1_scale=scale,
            bmm2_scale=1.0,
            window_left=-1,
            kv_layout="HND",
            q_len_per_req=1,
        )

    expected = _balanced_reference_decode(
        query, kv_cache, block_tables, seq_lens_t, q_len=1, scale=scale
    )
    expected_short = _balanced_reference_decode(
        query,
        kv_cache,
        block_tables,
        torch.tensor(
            _balanced_shorter_lengths(seq_lens, 1), dtype=torch.int32, device=device
        ),
        q_len=1,
        scale=scale,
    )
    torch.testing.assert_close(run().float(), expected, atol=1e-2, rtol=1e-2)
    _check_balanced_graph_replay(
        run,
        seq_lens=seq_lens,
        seq_lens_t=seq_lens_t,
        q_len=1,
        expected=expected,
        expected_short=expected_short,
        dequant=lambda tensor: tensor.float(),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)
@pytest.mark.parametrize(
    ("page_size", "q_len", "group", "num_kv_heads", "seq_lens"),
    [
        (16, 1, 8, 1, _BALANCED_AGENTX),
        (32, 1, 5, 2, [4097, 65536, 300, 33000]),
        (64, 8, 2, 2, [8192, 4160, 1000, 60000]),
        (16, 4, 4, 1, [40004, 32771, 36000, 33000, 45000, 34000]),
    ],
)
def test_cake_decode_balanced_hd256_matches_reference_and_replays(
    page_size, q_len, group, num_kv_heads, seq_lens
) -> None:
    # q_len * group <= 16 rows: the resident small-M instance (packed 32/64-row
    # tiles) does not claim these shapes, so the per-row balanced kernel owns
    # them in production.
    device = torch.device("cuda")
    query, kv_cache, block_tables, seq_lens_t = _balanced_case(
        device,
        q_len=q_len,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        seed=685_400 + page_size + q_len,
        head_dim=256,
        page_size=page_size,
        group=group,
    )
    scale = 256**-0.5
    max_seq_len = max(seq_lens)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    route = _select_balanced_route(
        device,
        query=query,
        kv_cache=kv_cache,
        out=torch.empty_like(query),
        workspace=workspace,
        block_tables=block_tables,
        seq_lens_t=seq_lens_t,
        q_len=q_len,
        max_seq_len=max_seq_len,
        bmm1_scale=scale,
    )
    assert route is not None and route.component == "decode_balanced_bf16_hd256"
    assert route.page_size == page_size

    def run():
        return cake_api.cake_batch_decode_with_kv_cache(
            query,
            kv_cache,
            workspace,
            block_tables,
            seq_lens_t,
            max_seq_len,
            bmm1_scale=scale,
            bmm2_scale=1.0,
            window_left=-1,
            kv_layout="HND",
            q_len_per_req=q_len,
        )

    expected = _balanced_reference_decode(
        query,
        kv_cache,
        block_tables,
        seq_lens_t,
        q_len=q_len,
        scale=scale,
        page_size=page_size,
    )
    expected_short = _balanced_reference_decode(
        query,
        kv_cache,
        block_tables,
        torch.tensor(
            _balanced_shorter_lengths(seq_lens, q_len),
            dtype=torch.int32,
            device=device,
        ),
        q_len=q_len,
        scale=scale,
        page_size=page_size,
    )
    torch.testing.assert_close(run().float(), expected, atol=1e-2, rtol=1e-2)
    _check_balanced_graph_replay(
        run,
        seq_lens=seq_lens,
        seq_lens_t=seq_lens_t,
        q_len=q_len,
        expected=expected,
        expected_short=expected_short,
        dequant=lambda tensor: tensor.float(),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cake_decode_bf16_hd256_smallm_cuda_graph_without_prewarm() -> None:
    """Serving-style CUDA graph use of the small-M hd256 route.

    Capture with no eager prewarm and a transient query/output (graph-pool
    addresses), then replay across growing KV lengths with other GPU work
    between launches, the way a decode loop interleaves layers. Guards the
    by-value TMA descriptors, the self-resetting merge tickets kept in
    ``multi_ctas_kv_counter_buffer`` (no per-launch memset node), and the
    softmax/correction ``corr_scale`` ordering whose mbarrier parity could
    alias and deadlock a CTA under exactly this interleaving.
    """
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        pytest.skip("Cake FMHA requires SM100 or SM103")

    from flashinfer.utils import (
        get_device_sm_count,
        get_trtllm_gen_multi_ctas_kv_counter_bytes,
    )

    # Qwen3.5-35B-A3B TP=2 verify: 7 query rows x 8 heads over one KV head.
    q_len, num_q_heads, head_dim, page_size, max_ctx = 7, 8, 256, 64, 65536
    gen = torch.Generator(device=device).manual_seed(0)
    num_pages = max_ctx // page_size + 8
    key_cache = 0.5 * torch.randn(
        (num_pages, 1, page_size, head_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=gen,
    )
    value_cache = torch.randn(
        (num_pages, 1, page_size, head_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=gen,
    )
    block_tables = (
        torch.randperm(num_pages, device=device, generator=gen)[: max_ctx // page_size]
        .to(torch.int32)
        .view(1, -1)
    )
    seq_lens = torch.tensor([60000], dtype=torch.int32, device=device)
    q_src = torch.empty(
        (q_len, num_q_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    workspace = torch.zeros(256 << 20, dtype=torch.uint8, device=device)
    counter = torch.zeros(
        get_trtllm_gen_multi_ctas_kv_counter_bytes(
            1, num_q_heads, get_device_sm_count(device)
        ),
        dtype=torch.uint8,
        device=device,
    )
    scale = head_dim**-0.5

    def run():
        query = q_src * 1.0  # transient tensor, graph-pool address inside capture
        return decode.trtllm_batch_decode_with_kv_cache(
            query=query,
            kv_cache=(key_cache, value_cache),
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_ctx,
            bmm1_scale=scale,
            bmm2_scale=1.0,
            window_left=-1,
            out_dtype=torch.bfloat16,
            q_len_per_req=q_len,
            multi_ctas_kv_counter_buffer=counter,
            backend="cake",
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):  # no eager call before capture
        out = run()

    pages = block_tables[0].long()
    k_all = key_cache[pages, 0].reshape(-1, head_dim).float()
    v_all = value_cache[pages, 0].reshape(-1, head_dim).float()
    for kv_len in range(60000, 60000 + 3 * 128, 3):  # crosses page/block edges
        q_src.normal_(generator=gen)
        seq_lens.fill_(kv_len)
        graph.replay()
        # Dense bottom-right-causal reference; also the interleaved GPU work.
        scores = (q_src.float() @ k_all[:kv_len].T) * scale
        pos = torch.arange(kv_len, device=device)
        limit = kv_len - q_len + torch.arange(q_len, device=device)
        scores = scores.masked_fill(pos > limit[:, None, None], float("-inf"))
        ref = torch.softmax(scores, dim=-1) @ v_all[:kv_len]
        torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
    torch.cuda.synchronize()
    assert counter.count_nonzero().item() == 0, "merge tickets must self-reset"
