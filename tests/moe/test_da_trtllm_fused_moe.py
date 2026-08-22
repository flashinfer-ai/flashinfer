"""User-facing tuning, eager dispatch, capture, and replay coverage for TRTLLM DA MoE."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from benchmarks.bench_trtllm_moe_da import (
    _canonical_inputs,
    _capture,
    _matching_diagnostic,
    _prepare_precision,
    _realization,
    _temporary_environment,
)
from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    QuantVariant,
    TrtllmBf16Config,
    TrtllmFp4Config,
    TrtllmFp8PerTensorConfig,
    trtllm_bf16_moe,
    trtllm_bf16_routed_moe,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_per_tensor_scale_moe,
    trtllm_moe_acquire_da_graph_leases,
    trtllm_moe_allocate_routing_metadata,
    trtllm_moe_da_diagnostics,
    trtllm_moe_release_da_resources,
)
from flashinfer.fused_moe.da_tuner import DADistribution, RoutingRealizationFactory
from flashinfer.tllm_enums import (
    RoutingInputMode,
    RoutingMethodType,
)

from tests.moe.da_acceptance_utils import (
    PRODUCTION_PRECISIONS,
    compact_shape,
    deepseek_l0_shape,
    require_sm100,
    run_matched_public_graphs,
)


# Shared-plan capture ownership


def _bf16_weights(shape):
    """Prepare the public blocked BF16 weights for one benchmark shape."""
    hidden, w1, w2, _, _ = _canonical_inputs(shape)
    view = TrtllmBf16Config.prepare_weights(
        w1,
        w2,
        num_local_experts=shape.local_num_experts,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        device=hidden.device,
    )
    return hidden, view["gemm1_weights"], view["gemm2_weights"]


def _matching_from_logits_diagnostic(shape, distributions):
    """Return the exact FP8 FromLogits diagnostic for one EP problem."""
    expected_distributions = [DADistribution.parse(item).name for item in distributions]
    matches = []
    for item in trtllm_moe_da_diagnostics():
        operation_key = json.loads(str(item["operation_key"]))
        config_identity = json.loads(operation_key["config_identity"])
        if (
            operation_key["custom_op"] == "flashinfer::trtllm_fp8_per_tensor_scale_moe"
            and operation_key["routing_input_mode"] == RoutingInputMode.FromLogits.value
            and operation_key["num_tokens"] == shape.num_tokens
            and operation_key["num_experts"] == shape.num_experts
            and operation_key["local_expert_offset"] == shape.local_expert_offset
            and operation_key["num_local_experts"] == shape.local_num_experts
            and operation_key["top_k"] == shape.top_k
            and config_identity["distributions"] == expected_distributions
        ):
            matches.append(item)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one FP8 FromLogits DA diagnostic, found {len(matches)}"
        )
    return matches[0]


def test_same_shape_layers_share_one_serial_workspace_lane() -> None:
    """Serial same-domain layers must share one inspected graph-owned workspace lane."""
    require_sm100()
    shape = deepseek_l0_shape()
    distributions = (
        "uniform",
        "ddist:1.1",
        "ddist:1.5",
        "ddist:2",
        "ddist:3",
        "ddist:4",
    )
    first = _prepare_precision("fp8_per_tensor", shape)
    second = _prepare_precision("fp8_per_tensor", shape)
    unprepared = _prepare_precision("fp8_per_tensor", shape)

    # Tune the shared operation domain once and prepare two independent exact binding sets.
    with _temporary_environment(
        FLASHINFER_DIST_AWARE_AUTOTUNE="1",
        FLASHINFER_DA_DISTRIBUTIONS=",".join(distributions),
        FLASHINFER_DA_BASELINE_GUARD="0",
    ):
        with autotune(True, tuning_buckets=(shape.num_tokens,)):
            first.invoke()
            second.invoke()
        factory = RoutingRealizationFactory()
        ids, weights = _realization(factory, shape, "ddist:4")
        first.stage(ids, weights)
        second.stage(ids, weights)
        unprepared.stage(ids, weights)
        first.invoke()
        second.invoke()
        torch.cuda.synchronize()

        # The final unprepared invocation deliberately falls back after two successful injections;
        # it must neither evict their resources nor clear the pending graph lease.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            (first.invoke(), second.invoke(), unprepared.invoke())[-1]
        leases = trtllm_moe_acquire_da_graph_leases(graph)

    try:
        graph.replay()
        torch.cuda.synchronize()
        diagnostic = _matching_diagnostic("fp8_per_tensor", shape, distributions)
        assert torch.isfinite(first.output).all()
        assert torch.isfinite(second.output).all()
        assert torch.isfinite(unprepared.output).all()
        if diagnostic["policy"] != "da_switch":
            assert diagnostic["policy"] in {"da_single_body", "da_fallback"}
            pytest.skip("natural autotuning did not compile a DA switch plan")
        if not leases:
            assert diagnostic["capture_fallback_reason"]
            pytest.skip(
                "runtime resource admission deliberately used pristine NoDA capture fallback"
            )
        # Binding diagnostics are intentionally cumulative lightweight pointer signatures. Other
        # same-domain tests may have run first, while this graph still contributes two bindings.
        assert diagnostic["binding_record_count"] >= 2
        assert diagnostic["prepared_workspace_lane_count"] == 1
        assert diagnostic["leased_workspace_lane_count"] == 1
        assert diagnostic["prepared_body_workspace_count"] == 1
        assert diagnostic["capture_stream_count"] == 1
        assert diagnostic["topology"]["conditional_node_count"] == 2, diagnostic[
            "topology"
        ]
        assert diagnostic["topology"]["is_workspace_lane_serialized"] is True
        assert diagnostic["topology"]["workspace_lane_invocation_count"] == 2
        assert len(leases) == 1
        assert leases[0].resource_count == 1
    finally:
        torch.cuda.synchronize()
        graph.reset()
        for lease in leases:
            lease.release()
    released = _matching_diagnostic("fp8_per_tensor", shape, distributions)
    assert released["binding_record_count"] >= 2
    assert released["prepared_workspace_lane_count"] == 1
    assert released["leased_workspace_lane_count"] == 0
    assert trtllm_moe_release_da_resources() >= 1
    released = _matching_diagnostic("fp8_per_tensor", shape, distributions)
    assert released["prepared_workspace_lane_count"] == 0
    assert released["prepared_body_workspace_count"] == 0
    assert released["capture_stream_count"] == 0


# Routing-input and expert-parallel contracts


def test_from_logits_ep_tuning_accepts_global_selector_ids() -> None:
    """Grouped FromLogits tuning must fingerprint valid global IDs outside the local shard."""
    require_sm100()
    shape = replace(
        compact_shape(num_tokens=32),
        num_experts=256,
        local_num_experts=32,
        local_expert_offset=0,
        top_k=8,
        n_group=8,
        topk_group=4,
    )
    hidden, w1, w2, _, _ = _canonical_inputs(shape)
    input_scale = torch.tensor(1.0, device=hidden.device)
    intermediate_scale = torch.tensor(1.0, device=hidden.device)
    hidden_q, _ = TrtllmFp8PerTensorConfig.prepare_activations(
        hidden, hidden_states_scale_global=input_scale
    )
    view = TrtllmFp8PerTensorConfig.prepare_weights(
        w1,
        w2,
        hidden_states_scale_global=input_scale,
        intermediate_scale_global=intermediate_scale,
        num_local_experts=shape.local_num_experts,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        device=hidden.device,
    )
    routing_logits = torch.full(
        (shape.num_tokens, shape.num_experts),
        -16.0,
        device=hidden.device,
        dtype=torch.bfloat16,
    )
    routing_logits[:, shape.local_num_experts :] = torch.linspace(
        0.0,
        8.0,
        shape.num_experts - shape.local_num_experts,
        device=hidden.device,
        dtype=torch.bfloat16,
    )
    routing_bias = torch.zeros(
        shape.num_experts, device=hidden.device, dtype=torch.bfloat16
    )
    output = torch.empty_like(hidden)

    def invoke() -> torch.Tensor:
        """Invoke grouped expert-parallel FromLogits through the public FP8 API."""
        return trtllm_fp8_per_tensor_scale_moe(
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            hidden_states=hidden_q,
            gemm1_weights=view["gemm1_weights"],
            output1_scales_scalar=view["output1_scales_scalar"],
            output1_scales_gate_scalar=view["output1_scales_gate_scalar"],
            gemm2_weights=view["gemm2_weights"],
            output2_scales_scalar=view["output2_scales_scalar"],
            num_experts=shape.num_experts,
            top_k=shape.top_k,
            n_group=shape.n_group,
            topk_group=shape.topk_group,
            intermediate_size=shape.intermediate_size,
            local_expert_offset=shape.local_expert_offset,
            local_num_experts=shape.local_num_experts,
            routed_scaling_factor=1.0,
            use_routing_scales_on_input=False,
            routing_method_type=RoutingMethodType.DeepSeekV3.value,
            output=output,
            tune_max_num_tokens=shape.tune_max_num_tokens,
        )

    distributions = ("uniform", "ddist:4")
    with _temporary_environment(
        FLASHINFER_DIST_AWARE_AUTOTUNE="1",
        FLASHINFER_DA_DISTRIBUTIONS=",".join(distributions),
        FLASHINFER_DA_BASELINE_GUARD="0",
    ):
        # This tuning phase used to reject the router's valid global IDs as non-local.
        with autotune(True, tuning_buckets=(shape.num_tokens,)):
            invoke()
        invoke()
        torch.cuda.synchronize()
        graph = _capture(invoke)
        leases = trtllm_moe_acquire_da_graph_leases(graph)

    try:
        graph.replay()
        torch.cuda.synchronize()
        diagnostic = _matching_from_logits_diagnostic(shape, distributions)
        assert diagnostic["policy"] in {"da_single_body", "da_switch"}
        assert torch.isfinite(output).all()
    finally:
        torch.cuda.synchronize()
        graph.reset()
        for lease in leases:
            lease.release()


@pytest.mark.parametrize("num_tokens", (2048, 4096, 8192))
def test_from_logits_large_token_capture_falls_back_deliberately(
    num_tokens: int,
) -> None:
    """Large public FromLogits calls must execute DA or a diagnosed ordinary fallback."""
    require_sm100()
    shape = replace(
        compact_shape(num_tokens=num_tokens),
        num_experts=256,
        local_num_experts=256,
        local_expert_offset=0,
        top_k=8,
        n_group=None,
        topk_group=None,
    )
    hidden, w1, w2, _, _ = _canonical_inputs(shape)
    input_scale = torch.tensor(1.0, device=hidden.device)
    intermediate_scale = torch.tensor(1.0, device=hidden.device)
    hidden_q, _ = TrtllmFp8PerTensorConfig.prepare_activations(
        hidden, hidden_states_scale_global=input_scale
    )
    view = TrtllmFp8PerTensorConfig.prepare_weights(
        w1,
        w2,
        hidden_states_scale_global=input_scale,
        intermediate_scale_global=intermediate_scale,
        num_local_experts=shape.local_num_experts,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        device=hidden.device,
    )
    routing_logits = torch.randn(
        shape.num_tokens,
        shape.num_experts,
        device=hidden.device,
        dtype=torch.bfloat16,
    )
    output = torch.empty_like(hidden)

    def invoke() -> torch.Tensor:
        """Invoke one public FP8 FromLogits problem at the requested token count."""
        return trtllm_fp8_per_tensor_scale_moe(
            routing_logits=routing_logits,
            routing_bias=None,
            hidden_states=hidden_q,
            gemm1_weights=view["gemm1_weights"],
            output1_scales_scalar=view["output1_scales_scalar"],
            output1_scales_gate_scalar=view["output1_scales_gate_scalar"],
            gemm2_weights=view["gemm2_weights"],
            output2_scales_scalar=view["output2_scales_scalar"],
            num_experts=shape.num_experts,
            top_k=shape.top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=shape.intermediate_size,
            local_expert_offset=0,
            local_num_experts=shape.local_num_experts,
            routed_scaling_factor=1.0,
            use_routing_scales_on_input=False,
            routing_method_type=RoutingMethodType.Renormalize.value,
            output=output,
            tune_max_num_tokens=shape.num_tokens,
        )

    distributions = ("uniform", "ddist:4")
    with _temporary_environment(
        FLASHINFER_DIST_AWARE_AUTOTUNE="1",
        FLASHINFER_DA_DISTRIBUTIONS=",".join(distributions),
        FLASHINFER_DA_BASELINE_GUARD="0",
    ):
        # The native 256-expert preamble admits 2,048 tokens. Larger shapes deliberately publish
        # an ordinary policy before DA candidate preparation instead of leaking an allocator error.
        with autotune(True, tuning_buckets=(shape.num_tokens,)):
            invoke()
        invoke()
        graph = _capture(invoke)
        leases = trtllm_moe_acquire_da_graph_leases(graph)

    try:
        graph.replay()
        torch.cuda.synchronize()
        diagnostic = _matching_from_logits_diagnostic(shape, distributions)
        if num_tokens == 2048:
            assert diagnostic["policy"] in {"da_single_body", "da_switch"}
        else:
            assert diagnostic["policy"] == "da_fallback"
            assert "supports at most 2048 tokens" in str(
                diagnostic["capture_fallback_reason"]
            )
            assert not leases
        assert torch.isfinite(output).all()
    finally:
        torch.cuda.synchronize()
        graph.reset()
        for lease in leases:
            lease.release()


# Public router replay outputs


@pytest.mark.parametrize("num_tokens", (2, 32, 256))
def test_llama4_public_routing_replay_writes_selected_ids(num_tokens: int) -> None:
    """Llama4 must publish top-1 replay IDs through every public routing topology."""
    require_sm100()
    shape = replace(compact_shape(num_tokens=num_tokens), top_k=1)
    hidden, gemm1_weights, gemm2_weights = _bf16_weights(shape)
    routing_logits = torch.randn(
        shape.num_tokens,
        shape.num_experts,
        device=hidden.device,
        dtype=torch.bfloat16,
    )
    replay_ids = torch.full(
        (shape.num_tokens, 1), -1, device=hidden.device, dtype=torch.int16
    )

    # Exercise the ordinary public router with DA disabled so replay-output ownership is isolated.
    with _temporary_environment(FLASHINFER_DIST_AWARE_AUTOTUNE="0"):
        output = trtllm_bf16_moe(
            routing_logits=routing_logits,
            routing_bias=None,
            hidden_states=hidden,
            gemm1_weights=gemm1_weights,
            gemm2_weights=gemm2_weights,
            num_experts=shape.num_experts,
            top_k=1,
            n_group=None,
            topk_group=None,
            intermediate_size=shape.intermediate_size,
            local_expert_offset=0,
            local_num_experts=shape.local_num_experts,
            routed_scaling_factor=1.0,
            routing_method_type=RoutingMethodType.Llama4.value,
            routing_replay_out=replay_ids,
            tune_max_num_tokens=num_tokens,
        )
    torch.cuda.synchronize()

    expected = routing_logits.argmax(dim=1, keepdim=True).to(torch.int16)
    assert torch.equal(replay_ids, expected)
    assert torch.isfinite(output).all()


# Auxiliary body ABI


def test_lora_public_capture_uses_graph_stable_da_outputs() -> None:
    """A DA switch must publish stable, selected-body LoRA auxiliary outputs."""
    require_sm100()
    shape = compact_shape(num_tokens=48)
    hidden, gemm1_weights, gemm2_weights = _bf16_weights(shape)
    factory = RoutingRealizationFactory()
    expert_ids, routing_weights = _realization(factory, shape, "ddist:4")
    packed = expert_ids.bitwise_left_shift(16).bitwise_or(
        routing_weights.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
    )
    lora_delta = torch.randn(
        shape.num_tokens,
        shape.top_k,
        2 * shape.intermediate_size,
        device=hidden.device,
        dtype=torch.bfloat16,
    ).mul_(0.05)
    output = torch.empty_like(hidden)

    def invoke() -> list[torch.Tensor]:
        """Invoke the public routed BF16 LoRA ABI and preserve all auxiliary results."""
        result = trtllm_bf16_routed_moe(
            topk_ids=packed,
            hidden_states=hidden,
            gemm1_weights=gemm1_weights,
            gemm2_weights=gemm2_weights,
            gemm1_lora_delta=lora_delta,
            num_experts=shape.num_experts,
            top_k=shape.top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=shape.intermediate_size,
            local_expert_offset=shape.local_expert_offset,
            local_num_experts=shape.local_num_experts,
            routed_scaling_factor=1.0,
            routing_method_type=RoutingMethodType.Renormalize.value,
            output=output,
            tune_max_num_tokens=shape.tune_max_num_tokens,
        )
        assert isinstance(result, list)
        return result

    distributions = ("uniform", "ddist:1.1", "ddist:2", "ddist:4")
    graph = torch.cuda.CUDAGraph()
    leases = []
    with _temporary_environment(
        FLASHINFER_DIST_AWARE_AUTOTUNE="1",
        FLASHINFER_DA_DISTRIBUTIONS=",".join(distributions),
        FLASHINFER_DA_BASELINE_GUARD="0",
    ):
        # Tune and prepare the exact LoRA input ABI before entering graph capture.
        with autotune(True, tuning_buckets=(shape.num_tokens,)):
            result = invoke()
        assert len(result) == 3
        diagnostic = _matching_diagnostic("bf16", shape, distributions)
        if diagnostic["policy"] != "da_switch":
            pytest.skip("natural autotuning did not compile a DA switch plan")

        with torch.cuda.graph(graph):
            captured = invoke()
        leases = trtllm_moe_acquire_da_graph_leases(graph)

    try:
        if not leases:
            pytest.skip("DA resource admission used ordinary capture fallback")
        assert len(captured) == 3
        stable_ptrs = tuple(tensor.data_ptr() for tensor in captured)

        def canonical_activation(outputs: list[torch.Tensor]) -> torch.Tensor:
            """Gather one tactic-specific padded FC1 buffer into token-slot order."""
            mapping = outputs[1].reshape(-1).to(torch.int64)
            activation = outputs[2]
            valid = mapping >= 0
            assert torch.all(mapping[valid] < activation.shape[0])
            canonical = torch.zeros(
                mapping.numel(),
                activation.shape[-1],
                dtype=activation.dtype,
                device=activation.device,
            )
            canonical[valid] = activation[mapping[valid]]
            return canonical

        # Replay two routing spectra that may choose different bodies. The
        # public pointers stay fixed and both output plus exposed FC1 values
        # match the ordinary multi-output contract.
        for distribution in ("uniform", "ddist:4"):
            ids, weights = _realization(factory, shape, distribution)
            packed.copy_(
                ids.bitwise_left_shift(16).bitwise_or(
                    weights.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
                )
            )
            graph.replay()
            torch.cuda.synchronize()
            da_output = captured[0].clone()
            da_activation = canonical_activation(captured)
            assert tuple(tensor.data_ptr() for tensor in captured) == stable_ptrs

            with _temporary_environment(FLASHINFER_DIST_AWARE_AUTOTUNE="0"):
                ordinary = invoke()
            torch.cuda.synchronize()
            torch.testing.assert_close(da_output, ordinary[0], rtol=3e-2, atol=3e-2)
            torch.testing.assert_close(
                da_activation,
                canonical_activation(ordinary),
                rtol=3e-2,
                atol=3e-2,
            )
    finally:
        torch.cuda.synchronize()
        graph.reset()
        for lease in leases:
            lease.release()


# Supported precision families and live replay inputs


def test_public_bf16_da_supports_512_global_experts() -> None:
    """The public DA path must admit the selector's full global-expert capacity."""
    require_sm100()
    shape = replace(compact_shape(num_tokens=24), num_experts=512)
    with _temporary_environment(FLASHINFER_DA_BASELINE_GUARD="0"):
        rows = run_matched_public_graphs(
            "bf16",
            shape=shape,
            distributions=("uniform", "ddist:1.1"),
        )
    assert {int(row["num_experts"]) for row in rows} == {512}
    # Capture may deliberately fall back when runtime resources are unavailable.
    policies = {str(row["policy"]) for row in rows}
    assert len(policies) == 1
    policy = policies.pop()
    assert policy in {"da_single_body", "da_switch"}


@pytest.mark.parametrize("precision", PRODUCTION_PRECISIONS)
def test_public_routed_precision_matches_ordinary_graph(precision: str) -> None:
    """Every supported precision must tune, capture, and replay numerically."""
    require_sm100()
    rows = run_matched_public_graphs(precision)
    assert {str(row["distribution"]) for row in rows} == {"uniform", "ddist:4"}


def test_live_distribution_selects_distinct_reachable_bodies() -> None:
    """One captured graph must select distinct complete bodies after routing mutation."""
    require_sm100()
    with _temporary_environment(FLASHINFER_DA_BASELINE_GUARD="0"):
        rows = run_matched_public_graphs(
            "fp8_per_tensor",
            shape=deepseek_l0_shape(),
            distributions=(
                "uniform",
                "ddist:1.1",
                "ddist:1.5",
                "ddist:2",
                "ddist:3",
                "ddist:4",
            ),
        )
    capture_policies = {row["capture_policy"] for row in rows}
    if capture_policies != {"da_switch"}:
        assert capture_policies <= {
            "da_single_body",
            "da_fallback",
            "noda_capture_fallback",
        }
        pytest.skip(
            "natural autotuning or runtime resource admission did not capture a DA switch"
        )
    assert {row["policy"] for row in rows} == {"da_switch"}
    selected_bodies = {int(row["selected_body"]) for row in rows}
    assert len(selected_bodies) >= 2
    assert all(0 <= int(row["selected_body"]) < int(row["num_bodies"]) for row in rows)


def test_fp32_unpacked_routing_weights_remain_live_during_da_replay() -> None:
    """A captured public FP4 graph must read changing caller-owned FP32 weights."""
    require_sm100()
    shape = deepseek_l0_shape()
    hidden, w1, w2, expert_ids, _ = _canonical_inputs(shape)
    routing_weights = torch.linspace(
        0.125,
        0.875,
        expert_ids.numel(),
        device=expert_ids.device,
        dtype=torch.float32,
    ).reshape_as(expert_ids)
    metadata = trtllm_moe_allocate_routing_metadata(
        expert_ids,
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        local_expert_offset=shape.local_expert_offset,
        num_local_experts=shape.local_num_experts,
        tile_n=32,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        topk_weights=routing_weights,
    )
    assert metadata.expert_weights.data_ptr() == routing_weights.data_ptr()
    assert metadata.expert_weights.dtype == torch.float32
    hidden_quantized, hidden_scale = TrtllmFp4Config.prepare_activations(
        hidden, variant=QuantVariant.NVFP4
    )
    view = TrtllmFp4Config.prepare_weights(
        w1,
        w2,
        variant=QuantVariant.NVFP4,
        num_local_experts=shape.local_num_experts,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        device=hidden.device,
    )
    ordinary_output = torch.empty_like(hidden)
    da_output = torch.empty_like(hidden)

    def invoke(output: torch.Tensor) -> torch.Tensor:
        """Invoke the public NVFP4 routed ABI with caller-owned FP32 weights."""
        result = trtllm_fp4_block_scale_routed_moe(
            topk_ids=(expert_ids, routing_weights),
            routing_bias=None,
            hidden_states=hidden_quantized,
            hidden_states_scale=hidden_scale,
            gemm1_weights=view["gemm1_weights"],
            gemm1_weights_scale=view["gemm1_weights_scale"],
            gemm1_bias=None,
            gemm1_alpha=view.get("gemm1_alpha"),
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=view["gemm2_weights"],
            gemm2_weights_scale=view["gemm2_weights_scale"],
            gemm2_bias=None,
            output1_scale_scalar=view.get("output1_scale_scalar"),
            output1_scale_gate_scalar=view.get("output1_scale_gate_scalar"),
            output2_scale_scalar=view.get("output2_scale_scalar"),
            num_experts=shape.num_experts,
            top_k=shape.top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=shape.intermediate_size,
            local_expert_offset=shape.local_expert_offset,
            local_num_experts=shape.local_num_experts,
            routed_scaling_factor=1.0,
            routing_method_type=RoutingMethodType.Renormalize.value,
            output=output,
            tune_max_num_tokens=shape.tune_max_num_tokens,
        )
        return result[0] if isinstance(result, list) else result

    factory = RoutingRealizationFactory()
    initial_ids, _ = _realization(factory, shape, "uniform")
    expert_ids.copy_(initial_ids)
    original_pointer = routing_weights.data_ptr()
    with _temporary_environment(FLASHINFER_DIST_AWARE_AUTOTUNE="0"):
        with autotune(True, tuning_buckets=(shape.num_tokens,)):
            invoke(ordinary_output)
        ordinary_graph = _capture(lambda: invoke(ordinary_output))

    with _temporary_environment(
        FLASHINFER_DIST_AWARE_AUTOTUNE="1",
        FLASHINFER_DA_DISTRIBUTIONS="uniform,ddist:4",
        FLASHINFER_DA_BASELINE_GUARD="0",
    ):
        with autotune(True, tuning_buckets=(shape.num_tokens,)):
            invoke(da_output)
        invoke(da_output)
        torch.cuda.synchronize()
        da_graph = _capture(lambda: invoke(da_output))
        leases = trtllm_moe_acquire_da_graph_leases(da_graph)

    first_da_output = None
    try:
        for replay_index, values in enumerate(
            (
                torch.linspace(
                    0.125,
                    0.875,
                    routing_weights.numel(),
                    device=routing_weights.device,
                    dtype=torch.float32,
                ).reshape_as(routing_weights),
                torch.linspace(
                    0.875,
                    0.125,
                    routing_weights.numel(),
                    device=routing_weights.device,
                    dtype=torch.float32,
                ).reshape_as(routing_weights),
            )
        ):
            routing_weights.copy_(values)
            assert routing_weights.data_ptr() == original_pointer
            with torch.cuda.nvtx.range(f"FP32_NODA_REPLAY_{replay_index}"):
                ordinary_graph.replay()
                torch.cuda.synchronize()
            with torch.cuda.nvtx.range(f"FP32_DA_REPLAY_{replay_index}"):
                da_graph.replay()
                torch.cuda.synchronize()
            max_abs_error = float(
                (da_output.float() - ordinary_output.float()).abs().max().item()
            )
            assert torch.equal(da_output, ordinary_output), max_abs_error
            if first_da_output is None:
                first_da_output = da_output.clone()
            else:
                assert not torch.equal(first_da_output, da_output)
        fp32_diagnostics = []
        for diagnostic in trtllm_moe_da_diagnostics():
            operation_key = json.loads(str(diagnostic["operation_key"]))
            if operation_key["num_tokens"] != shape.num_tokens:
                continue
            input_identity = operation_key["input_identity"]
            if any(item[1] == "torch.float32" for item in input_identity):
                fp32_diagnostics.append(diagnostic)
        assert len(fp32_diagnostics) == 1
        diagnostic = fp32_diagnostics[0]
        assert diagnostic["tuned"] is True
        assert diagnostic["policy"] in {"da_switch", "da_single_body"}
        if diagnostic["policy"] == "da_switch":
            if not leases:
                assert diagnostic["capture_fallback_reason"]
                pytest.skip(
                    "runtime resource admission deliberately used pristine NoDA capture fallback"
                )
            assert diagnostic["topology"]["conditional_node_count"] == 1
    finally:
        torch.cuda.synchronize()
        ordinary_graph.reset()
        da_graph.reset()
        for lease in leases:
            lease.release()


# Graph-free host dispatch


@pytest.mark.parametrize(
    ("precision", "distributions", "expected_distribution"),
    (
        ("bf16", ("uniform", "ddist:1.1"), "ddist:1.1"),
        ("fp8_per_tensor", ("uniform", "ddist:3"), "uniform"),
    ),
)
def test_public_eager_call_uses_preferred_tuned_body_without_da_graph(
    precision: str,
    distributions: tuple[str, ...],
    expected_distribution: str,
) -> None:
    """Graph-free execution must use the preferred measured body on the host."""
    require_sm100()
    shape = compact_shape(num_tokens=24)
    prepared = _prepare_precision(precision, shape)
    ids, weights = _realization(RoutingRealizationFactory(), shape, "ddist:3")
    prepared.stage(ids, weights)

    with _temporary_environment(FLASHINFER_DIST_AWARE_AUTOTUNE="0"):
        with autotune(True, tuning_buckets=(shape.num_tokens,)):
            prepared.invoke()
        torch.cuda.synchronize()
        ordinary = prepared.output.clone()

    with _temporary_environment(
        FLASHINFER_DIST_AWARE_AUTOTUNE="1",
        FLASHINFER_DA_DISTRIBUTIONS=",".join(distributions),
    ):
        with autotune(True, tuning_buckets=(shape.num_tokens,)):
            prepared.invoke()
        prepared.invoke()
        torch.cuda.synchronize()
        eager = prepared.output.clone()

    torch.testing.assert_close(eager, ordinary, rtol=3e-2, atol=3e-2)
    diagnostic = _matching_diagnostic(precision, shape, distributions)
    assert diagnostic["tuned"] is True
    assert diagnostic["eager_distribution"] == expected_distribution
    assert diagnostic["eager_body"] is not None
    assert diagnostic["topology"] is None
