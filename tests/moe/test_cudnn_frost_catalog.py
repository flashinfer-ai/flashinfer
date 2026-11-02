# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Generic kernel identities, legality, candidate budgets and unseen MoE shapes."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe.backends.cudnn_frost import catalog, heuristics
from flashinfer.fused_moe.backends.cudnn_frost.activations import ACTIVATIONS
from flashinfer.fused_moe.backends.cudnn_frost.manifest import read_manifest
from flashinfer.fused_moe.backends.cudnn_frost.runtime import artifact_root

DTYPES = ("bf16", "mxfp8", "nvfp4", "mxfp8_mxfp4")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires SM107",
)
@pytest.mark.parametrize(
    "name", ["geglu", "geglu_tanh", "swiglu", "situ", "swiglu_step"]
)
def test_nvfp4_gated_activation_product_normal_tma_graph(name, monkeypatch):
    from tests.moe import test_cudnn_frost_kernels as helpers

    def tma_pair(pool):
        choices = tuple(
            k
            for k in pool
            if not k.swap_ab and k.tactic_metadata["store_mode"] == "tma"
        )
        assert len(choices) >= 2
        return choices[:2]

    monkeypatch.setattr(helpers, "_pick_stage_pair", tma_pair)
    helpers.test_nvfp4_moe_full_pipeline_and_dynamic_graph(
        name, False, "bf16", monkeypatch, geometry=(17, 512, 768, 16, 4)
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_generic_identity_coalesces_shapes_and_preserves_numerical_contract(dtype):
    record = read_manifest(artifact_root(dtype) / "cudnn_frost_selected_kernels.json")[
        "kernels"
    ][0]
    alias = deepcopy(record)
    alias["id"] = "other-shape"
    alias["contract"].update(n=768, k=512, experts=16, groups=16)
    records, identities = catalog.generalize_records([record, alias], dtype)
    assert len(records) == 1
    assert identities[record["id"]] == identities["other-shape"]
    assert records[0]["source"] == record["source"]
    numerical = {
        k: v
        for k, v in record["contract"].items()
        if k not in ("s", "n", "k", "experts", "groups")
    }
    assert all(records[0]["contract"][k] == v for k, v in numerical.items())
    changed = deepcopy(record)
    changed["source"]["parameters"]["constants"].append(["new_constant", "1"])
    assert catalog.kernel_id(changed, dtype) == catalog.kernel_id(record, dtype)
    changed = deepcopy(record)
    changed["contract"]["output_dtype"] = "different"
    assert catalog.kernel_id(changed, dtype) != catalog.kernel_id(record, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "change",
    [
        {},
        {"n": 64},
        {"k": 129},
        {"s": 0},
        {"experts": 1025},
        {"groups": 0},
        {"n": (1 << 20) + 128},
    ],
)
def test_generic_legality_uses_bounds_and_alignment(dtype, change):
    record = read_manifest(artifact_root(dtype) / "cudnn_frost_selected_kernels.json")[
        "kernels"
    ][0]
    dimensions = dict(s=68, n=768, k=512, experts=16, groups=16)
    dimensions.update(change)
    assert catalog.can_implement(
        SimpleNamespace(contract=record["contract"]), dimensions
    ) == (not change)


@pytest.mark.parametrize(
    "name,value",
    [
        ("split_k_slices", "2"),
        ("moe_aligned_offsets", "True"),
        ("moe_token_alignment", "128"),
        ("offset_cutlass_dtype", "cutlass.Int64"),
    ],
)
def test_generic_contract_rejects_incompatible_frozen_recipes(name, value):
    record = read_manifest(artifact_root("bf16") / "cudnn_frost_selected_kernels.json")[
        "kernels"
    ][0]
    record["source"]["parameters"]["constants"].append([name, value])
    with pytest.raises(ValueError, match="unsupported generic"):
        catalog.generic_contract(record)


@pytest.mark.parametrize(
    "token_dtype,weight_dtype,expected",
    [
        ("bfloat16", "bfloat16", "z-wide-k"),
        ("float8_e4m3fn", "float8_e4m3fn", "a-narrow-k"),
        ("float4_e2m1fn_x2", "float4_e2m1fn_x2", "a-narrow-k"),
        ("float8_e4m3fn", "float4_e2m1fn_x2", "a-narrow-k"),
    ],
)
def test_heuristic_reduction_tiles_use_actual_operand_bytes(
    token_dtype, weight_dtype, expected
):
    def kernel(identity, kb):
        return SimpleNamespace(
            artifact_id=identity,
            swap_ab=False,
            gated=False,
            contract={"token_dtype": token_dtype, "weight_dtype": weight_dtype},
            tactic_metadata={
                "cta_tile": {"m": 64, "n": 128, "k_bytes": kb},
                "tile": "CONFIG_sm120_64x128x128_16x8x32_cluster1x1",
                "cta_group": 1,
                "store_mode": "stg",
                "swap_ab": False,
            },
        )

    chosen = heuristics.stage_candidates(
        [kernel("a-narrow-k", 128), kernel("z-wide-k", 256)],
        rows=128,
        n=128,
        k=128,
        experts=1,
        sm_count=80,
        budget=1,
    )
    assert chosen[0].artifact_id == expected


@pytest.mark.parametrize("arch", ["sm_107a", "sm_120a"])
def test_bf16_unmeasured_shapes_have_bounded_candidates(
    arch, monkeypatch, frost_metadata_only
):
    from flashinfer.fused_moe.backends.cudnn_frost.bf16 import moe
    from flashinfer.fused_moe import SwiGLU

    monkeypatch.setattr(moe.runtime, "_arch_for", lambda _: arch)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=80),
    )
    moe.runtime.clear_artifact_cache()
    try:
        first, second = moe._selected_kernels(17, 512, 768, 16, 4, "cpu", SwiGLU())
        assert 1 <= len(first) <= heuristics.STAGE_BUDGET
        assert 1 <= len(second) <= heuristics.STAGE_BUDGET
        assert all(kernel.arch == arch for kernel in (*first, *second))
    finally:
        moe.runtime.clear_artifact_cache()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires SM107",
)
@pytest.mark.parametrize(
    "dtype,name,fused",
    [
        (dtype, name, fused)
        for dtype in DTYPES
        for name in ACTIVATIONS
        for fused in ([False] if dtype == "bf16" else [False, True])
    ],
)
def test_unmeasured_shapes_all_dtypes_and_activations_graph(
    dtype, name, fused, monkeypatch
):
    from tests.moe.test_cudnn_frost_kernels import (
        _assert_mxfp8_moe_full_pipeline_and_graph,
        _bf16_moe_reference,
        bf16_config,
        bf16_packs,
    )

    if dtype in ("mxfp8", "mxfp8_mxfp4"):
        _assert_mxfp8_moe_full_pipeline_and_graph(
            name,
            False,
            monkeypatch,
            mixed=dtype == "mxfp8_mxfp4",
            fc1_output="fp8_tma" if fused else "bf16",
            geometry=(17, 512, 768, 16, 4),
        )
        return
    if dtype == "nvfp4":
        # The NVFP4 test has its own scale/weight preparation; use the same
        # generic shape for both fused and unfused paths below.
        from tests.moe.test_cudnn_frost_kernels import (
            test_nvfp4_moe_full_pipeline_and_dynamic_graph,
        )

        test_nvfp4_moe_full_pipeline_and_dynamic_graph(
            name,
            False,
            "fp4_tma" if fused else "bf16",
            monkeypatch,
            geometry=(17, 512, 768, 16, 4),
        )
        return
    from flashinfer.fused_moe.backends.cudnn_frost.bf16 import moe

    torch.manual_seed(131)
    activation = ACTIVATIONS[name]()
    config = replace(
        bf16_config(topk=4, experts=16, intermediate=768), activation=activation
    )
    act, weights = bf16_packs(
        tokens=17, topk=4, experts=16, hidden=512, intermediate=768
    )
    if not activation.is_gated:
        view = weights.get_view("cutlass_bf16")
        view["fc1_expert_weights"] = view["fc1_expert_weights"][:, :768].contiguous()
    runner = moe.CudnnFrostBf16MoeRunner(config, "cuda")
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(act, weights)
    from flashinfer.autotuner import autotune

    with autotune():
        choices = runner.get_valid_tactics(inputs, None)
    assert len(choices) <= 4
    output = runner.forward(inputs, choices[0])
    reference = _bf16_moe_reference(act, weights, activation)
    assert (output.float() - reference.float()).norm() / reference.float().norm() < 0.01
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.forward(inputs, choices[0])
    act.topk_ids.fill_(15)
    graph.replay()
    reference = _bf16_moe_reference(act, weights, activation)
    assert (output.float() - reference.float()).norm() / reference.float().norm() < 0.01
