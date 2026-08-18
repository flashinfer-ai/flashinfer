"""Correctness gates for static hot-prefix folded NVFP4 execution."""

from __future__ import annotations

from typing import Literal

import pytest
import torch

from flashinfer.fused_moe.nvfp4_checkpoint import reference_dequantize_nvfp4

from ._sm90_push_fp8_reference import reference_moe
from .test_sm90_fp8_nvfp4_bf16_push_cuda_backend import (
    INTERMEDIATE,
    LOCAL_EXPERTS,
    TOKEN_CAPACITY,
    TOP_K,
    _build_prepared_layer,
    _forward,
    _make_inputs,
    _make_weights,
    _normalized_l2,
    requires_sm90_fp8,
)


def _checkpoints(device: torch.device, seed: int):
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda import (
        quantize_bf16_to_nvfp4_checkpoint,
    )

    w13, w2 = _make_weights(LOCAL_EXPERTS, seed, device)
    return (
        quantize_bf16_to_nvfp4_checkpoint(w13),
        quantize_bf16_to_nvfp4_checkpoint(w2),
    )


def _nvfp4_config(
    *,
    payload_layout: int = 4,
    weight_policy: Literal["packed", "folded", "hot_folded", "dual"] = "packed",
    hot_expert_count: int = 0,
    acknowledge_dual_residency: bool = False,
):
    from flashinfer.moe_ep import Sm90_Fp8_Nvfp4_Bf16_PushCuda_MegaMoeConfig

    return Sm90_Fp8_Nvfp4_Bf16_PushCuda_MegaMoeConfig(
        intermediate_size=INTERMEDIATE,
        top_k=TOP_K,
        nvfp4_mode="w4a8",
        payload_dtype="bf16",
        combine_dtype="bf16",
        grouped_combine=False,
        fuse_act=True,
        payload_layout=payload_layout,
        allow_legacy_layout=payload_layout == 3,
        weight_policy=weight_policy,
        hot_expert_count=hot_expert_count,
        acknowledge_dual_residency=acknowledge_dual_residency,
    )


def _dense_folded(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    experts, rows, columns = weight.shape
    tiled = weight.float().reshape(experts, rows // 128, 128, columns // 128, 128)
    return (tiled * scales[:, :, None, :, None]).reshape(experts, rows, columns)


@requires_sm90_fp8
@pytest.mark.parametrize("hot_experts", [0, LOCAL_EXPERTS])
@pytest.mark.parametrize("payload_layout", [3, 4])
def test_hot_folded_endpoints_match_existing_engines(
    hot_experts: int, payload_layout: int
) -> None:
    from flashinfer.moe_ep import Sm90PushFp8MegaMoeConfig
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda import (
        make_folded_fp8_weights_from_checkpoints,
        make_hot_folded_weights_from_checkpoints,
        make_transformed_weights_from_checkpoints,
    )

    device = torch.device("cuda", 0)
    w13, w2 = _checkpoints(device, 301)
    if hot_experts == 0:
        hybrid = make_transformed_weights_from_checkpoints(
            w13,
            w2,
            nvfp4_mode="w4a8",
            group_size=128,
            residual_scheme="generic",
            payload_layout=payload_layout,
        )
        anchor_weights = hybrid
        anchor_config = _nvfp4_config(payload_layout=payload_layout)
        hybrid_config = anchor_config
    else:
        hybrid = make_hot_folded_weights_from_checkpoints(
            w13,
            w2,
            hot_experts=hot_experts,
            payload_layout=payload_layout,
        )
        anchor_weights = make_folded_fp8_weights_from_checkpoints(w13, w2)
        anchor_config = Sm90PushFp8MegaMoeConfig(
            intermediate_size=INTERMEDIATE,
            top_k=TOP_K,
            payload_dtype="bf16",
            combine_dtype="bf16",
            grouped_combine=False,
            fuse_fc1_epilogue=False,
        )
        hybrid_config = _nvfp4_config(
            payload_layout=payload_layout,
            weight_policy="folded",
            hot_expert_count=LOCAL_EXPERTS,
        )
    hybrid_layer = _build_prepared_layer(hybrid_config, hybrid)
    anchor_layer = _build_prepared_layer(anchor_config, anchor_weights)
    x, ids, weights = _make_inputs(TOKEN_CAPACITY, LOCAL_EXPERTS, 302, device)
    try:
        actual = _forward(hybrid_layer, x, ids, weights).clone()
        expected = _forward(anchor_layer, x, ids, weights).clone()
        torch.cuda.synchronize()
    finally:
        hybrid_layer.destroy()
        anchor_layer.destroy()
    if hot_experts == 0:
        assert torch.equal(actual, expected)
    else:
        assert _normalized_l2(actual, expected) <= 0.02


@requires_sm90_fp8
@pytest.mark.parametrize("payload_layout", [3, 4])
def test_hot_folded_mixed_output_matches_hybrid_weight_oracle(
    payload_layout: int,
) -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda import (
        make_hot_folded_weights_from_checkpoints,
    )

    device = torch.device("cuda", 0)
    w13, w2 = _checkpoints(device, 311)
    hybrid = make_hot_folded_weights_from_checkpoints(
        w13,
        w2,
        hot_experts=1,
        payload_layout=payload_layout,
    )
    layer = _build_prepared_layer(
        _nvfp4_config(
            payload_layout=payload_layout,
            weight_policy="hot_folded",
            hot_expert_count=1,
        ),
        hybrid,
    )
    x, ids, weights = _make_inputs(TOKEN_CAPACITY, LOCAL_EXPERTS, 312, device)
    try:
        actual = _forward(layer, x, ids, weights).clone()
        torch.cuda.synchronize()
    finally:
        layer.destroy()

    assert hybrid.hot_fp8 is not None
    hot = hybrid.hot_fp8
    dense_w13 = torch.cat(
        (
            _dense_folded(hot.w13_fp8, hot.w13_sf),
            reference_dequantize_nvfp4(w13)[1:],
        )
    )
    dense_w2 = torch.cat(
        (
            _dense_folded(hot.w2_fp8, hot.w2_sf),
            reference_dequantize_nvfp4(w2)[1:],
        )
    )
    expected = reference_moe(x, dense_w13, dense_w2, ids, weights)
    assert torch.isfinite(actual.float()).all()
    assert _normalized_l2(actual, expected) <= 0.35


@requires_sm90_fp8
def test_dual_policy_matches_folded_execution_and_keeps_both_views() -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda import (
        make_dual_weights_from_checkpoints,
        make_hot_folded_weights_from_checkpoints,
    )

    device = torch.device("cuda", 0)
    w13, w2 = _checkpoints(device, 316)
    dual = make_dual_weights_from_checkpoints(w13, w2)
    folded = make_hot_folded_weights_from_checkpoints(
        w13,
        w2,
        hot_experts=LOCAL_EXPERTS,
    )
    dual_layer = _build_prepared_layer(
        _nvfp4_config(
            weight_policy="dual",
            acknowledge_dual_residency=True,
        ),
        dual,
    )
    folded_layer = _build_prepared_layer(
        _nvfp4_config(
            weight_policy="folded",
            hot_expert_count=LOCAL_EXPERTS,
        ),
        folded,
    )
    inputs = _make_inputs(16, LOCAL_EXPERTS, 317, device)
    try:
        actual = _forward(dual_layer, *inputs).clone()
        expected = _forward(folded_layer, *inputs).clone()
        torch.cuda.synchronize()
    finally:
        dual_layer.destroy()
        folded_layer.destroy()

    assert torch.equal(actual, expected)
    assert dual.packed_bytes > 0
    assert dual.folded_bytes > 0
    assert dual.resident_bytes == dual.packed_bytes + dual.folded_bytes


@requires_sm90_fp8
def test_hot_folded_shared_workspace_rebind_a_b_a_preserves_results() -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda import (
        make_hot_folded_weights_from_checkpoints,
    )

    device = torch.device("cuda", 0)
    weights_a = make_hot_folded_weights_from_checkpoints(
        *_checkpoints(device, 321), hot_experts=1
    )
    weights_b = make_hot_folded_weights_from_checkpoints(
        *_checkpoints(device, 322), hot_experts=1
    )
    config = _nvfp4_config(weight_policy="hot_folded", hot_expert_count=1)
    layer_a = _build_prepared_layer(config, weights_a)
    layer_b = _build_prepared_layer(config, weights_b)
    x, ids, weights = _make_inputs(16, LOCAL_EXPERTS, 323, device)
    try:
        first_a = _forward(layer_a, x, ids, weights).clone()
        result_b = _forward(layer_b, x, ids, weights).clone()
        assert layer_a._workspace is layer_b._workspace
        second_a = _forward(layer_a, x, ids, weights).clone()
        torch.cuda.synchronize()
    finally:
        layer_a.destroy()
        layer_b.destroy()
    assert not torch.equal(first_a, result_b)
    assert torch.equal(first_a, second_a)


@requires_sm90_fp8
def test_hot_folded_two_layers_share_workspace_and_graph_replay() -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda import (
        make_hot_folded_weights_from_checkpoints,
    )

    device = torch.device("cuda", 0)
    hybrid_a = make_hot_folded_weights_from_checkpoints(
        *_checkpoints(device, 331), hot_experts=1
    )
    hybrid_b = make_hot_folded_weights_from_checkpoints(
        *_checkpoints(device, 332), hot_experts=1
    )
    config = _nvfp4_config(weight_policy="hot_folded", hot_expert_count=1)
    layer_a = _build_prepared_layer(config, hybrid_a)
    layer_b = _build_prepared_layer(config, hybrid_b)
    inputs = [
        _make_inputs(16, LOCAL_EXPERTS, 332 + index, device) for index in range(2)
    ]
    eager = [
        (
            _forward(layer_a, *values).clone(),
            _forward(layer_b, *values).clone(),
        )
        for values in inputs
    ]
    assert layer_a._workspace is layer_b._workspace
    static_x = torch.empty_like(inputs[0][0]).copy_(inputs[0][0])
    static_ids = torch.empty_like(inputs[0][1]).copy_(inputs[0][1])
    static_weights = torch.empty_like(inputs[0][2]).copy_(inputs[0][2])
    for _ in range(2):
        _forward(layer_a, static_x, static_ids, static_weights)
        _forward(layer_b, static_x, static_ids, static_weights)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output_a = _forward(layer_a, static_x, static_ids, static_weights)
        static_output_b = _forward(layer_b, static_x, static_ids, static_weights)
    try:
        for values, expected in zip(inputs, eager, strict=True):
            static_x.copy_(values[0])
            static_ids.copy_(values[1])
            static_weights.copy_(values[2])
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(static_output_a, expected[0])
            assert torch.equal(static_output_b, expected[1])
    finally:
        layer_a.destroy()
        layer_b.destroy()
