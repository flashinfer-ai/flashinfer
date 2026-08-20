"""User-facing tuning, eager dispatch, capture, and replay coverage for TRTLLM DA MoE."""

from __future__ import annotations

import json
from collections.abc import Callable
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
    populate_trtllm_moe_routing_metadata_,
    trtllm_bf16_moe,
    trtllm_bf16_routed_moe,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_per_tensor_scale_moe,
    trtllm_moe_acquire_da_graph_leases,
    trtllm_moe_allocate_routing_metadata,
    trtllm_moe_allocate_routing_metadata_multi_tile,
    trtllm_moe_da_diagnostics,
    trtllm_moe_release_da_resources,
)
from flashinfer.fused_moe.da_tuner import DADistribution, RoutingRealizationFactory
from flashinfer.fused_moe.tactic_search import (
    FactorizedSearch,
    FactorizedTactic,
    FactorizedTacticSpace,
)
from flashinfer.tllm_enums import RoutingInputMode, RoutingMethodType

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


def _assert_routing_metadata_slots_bit_exact(actual, expected) -> None:
    """Compare every initialized body-facing routing-metadata element exactly."""
    assert torch.equal(actual.total_num_padded_tokens, expected.total_num_padded_tokens)
    assert torch.equal(
        actual.expanded_idx_to_permuted_idx,
        expected.expanded_idx_to_permuted_idx,
    )
    assert torch.equal(actual.expert_weights, expected.expert_weights)
    assert torch.equal(actual.num_tokens_per_expert, expected.num_tokens_per_expert)
    assert torch.equal(actual.num_non_exiting_ctas, expected.num_non_exiting_ctas)

    # Padded permutation holes and capacity tails are intentionally undefined. Compare only row
    # indices proven live by the already-exact expanded mapping, plus the live grouped-GEMM prefix.
    live_permuted_indices = actual.expanded_idx_to_permuted_idx
    live_permuted_indices = live_permuted_indices[live_permuted_indices >= 0]
    num_ctas = int(actual.num_non_exiting_ctas.item())
    assert torch.equal(
        actual.permuted_idx_to_token_idx[live_permuted_indices],
        expected.permuted_idx_to_token_idx[live_permuted_indices],
    )
    assert torch.equal(
        actual.cta_idx_xy_to_batch_idx[:num_ctas],
        expected.cta_idx_xy_to_batch_idx[:num_ctas],
    )
    assert torch.equal(
        actual.cta_idx_xy_to_mn_limit[:num_ctas],
        expected.cta_idx_xy_to_mn_limit[:num_ctas],
    )


def _minimum_average_cuda_time_ms(invoke, *, iterations: int = 100) -> float:
    """Return the least-contended average CUDA-event time across five trials."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(5):
        start.record()
        for _ in range(iterations):
            invoke()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return min(samples)


# Fused routing capacity and exactness


@pytest.mark.parametrize("num_tokens", (2048, 2049, 8192))
@pytest.mark.parametrize(
    "routing_input_mode,expert_id_dtype",
    (
        (RoutingInputMode.PackedPrecomputed, torch.int32),
        (RoutingInputMode.UnpackedPrecomputed, torch.int16),
        (RoutingInputMode.UnpackedPrecomputed, torch.int32),
    ),
)
def test_fused_multi_tile_routing_matches_independent_tiles_at_capacity_boundaries(
    num_tokens: int,
    routing_input_mode: RoutingInputMode,
    expert_id_dtype: torch.dtype,
) -> None:
    """Fused packed and unpacked routing must match independent tile launches bit-exactly."""
    require_sm100()
    num_experts = 256
    top_k = 8
    tile_ns = (8, 16, 64)

    # Give every token unique experts with deterministic wraparound, plus nonuniform live weights.
    token_index = torch.arange(num_tokens, dtype=torch.int64).unsqueeze(1)
    topk_index = torch.arange(top_k, dtype=torch.int64).unsqueeze(0)
    expert_ids = ((token_index * 13 + topk_index * 17) % num_experts).to(
        device="cuda", dtype=expert_id_dtype
    )
    routing_weights = (
        (topk_index + 1).expand(num_tokens, -1).to(torch.bfloat16)
        / float(top_k * (top_k + 1) // 2)
    ).to(device="cuda")
    if routing_input_mode is RoutingInputMode.PackedPrecomputed:
        routing_ids = (
            expert_ids.to(torch.int32)
            .bitwise_left_shift(16)
            .bitwise_or(
                routing_weights.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
            )
        )
        routing_weights_arg = None
    else:
        routing_ids = expert_ids
        routing_weights_arg = routing_weights

    # Materialize an arbitrary tile mixture with one fused launch, then independently materialize
    # the same slots through the public one-tile ABI to establish the exact metadata reference.
    fused = trtllm_moe_allocate_routing_metadata_multi_tile(
        routing_ids,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=0,
        num_local_experts=num_experts,
        tile_ns=tile_ns,
        routing_input_mode=routing_input_mode,
        topk_weights=routing_weights_arg,
    )
    populate_trtllm_moe_routing_metadata_(fused, routing_ids, routing_weights_arg)
    references = []
    for tile_n in tile_ns:
        single = trtllm_moe_allocate_routing_metadata_multi_tile(
            routing_ids,
            num_experts=num_experts,
            top_k=top_k,
            local_expert_offset=0,
            num_local_experts=num_experts,
            tile_ns=(tile_n,),
            routing_input_mode=routing_input_mode,
            topk_weights=routing_weights_arg,
        )
        populate_trtllm_moe_routing_metadata_(single, routing_ids, routing_weights_arg)
        references.append(single.slots[0])
    torch.cuda.synchronize()

    for actual, expected in zip(fused.slots, references, strict=True):
        assert actual.tile_n == expected.tile_n
        _assert_routing_metadata_slots_bit_exact(actual, expected)


def test_fused_multi_tile_routing_beats_independent_launches(
    record_property,
) -> None:
    """One exported fused population call must beat three independent tile launches."""
    require_sm100()
    num_tokens = 8192
    num_experts = 256
    top_k = 8
    tile_ns = (8, 16, 64)
    token_index = torch.arange(num_tokens, device="cuda", dtype=torch.int32).unsqueeze(
        1
    )
    topk_index = torch.arange(top_k, device="cuda", dtype=torch.int32).unsqueeze(0)
    expert_ids = (token_index * 13 + topk_index * 17) % num_experts
    weights = torch.full(
        (num_tokens, top_k), 1.0 / top_k, device="cuda", dtype=torch.bfloat16
    )
    packed = expert_ids.bitwise_left_shift(16).bitwise_or(
        weights.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
    )

    # Allocate graph-stable outputs once so the measurement contains only the exported fused
    # population kernel or its matched three-launch decomposition.
    fused = trtllm_moe_allocate_routing_metadata_multi_tile(
        packed,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=0,
        num_local_experts=num_experts,
        tile_ns=tile_ns,
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
    )
    independent = tuple(
        trtllm_moe_allocate_routing_metadata_multi_tile(
            packed,
            num_experts=num_experts,
            top_k=top_k,
            local_expert_offset=0,
            num_local_experts=num_experts,
            tile_ns=(tile_n,),
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
        )
        for tile_n in tile_ns
    )

    def populate_fused() -> None:
        """Populate every tile through one public fused FFI call."""
        populate_trtllm_moe_routing_metadata_(fused, packed)

    def populate_independent() -> None:
        """Populate the matched tile outputs through three public FFI calls."""
        for metadata in independent:
            populate_trtllm_moe_routing_metadata_(metadata, packed)

    # Warm both paths before measuring their best-of-five CUDA-event averages on one stream.
    for _ in range(5):
        populate_fused()
        populate_independent()
    torch.cuda.synchronize()
    fused_ms = _minimum_average_cuda_time_ms(populate_fused)
    independent_ms = _minimum_average_cuda_time_ms(populate_independent)
    record_property("fused_multi_tile_routing_ms", fused_ms)
    record_property("independent_routing_ms", independent_ms)
    record_property("fused_speedup", independent_ms / fused_ms)
    assert fused_ms < independent_ms


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


@pytest.mark.parametrize("num_tokens", (2048, 2049, 8192))
def test_from_logits_large_token_capture_installs_da_switch(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary-sized public FromLogits calls must capture a complete DA switch."""
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

    distributions = (
        "uniform",
        "ddist:1.1",
        "ddist:1.5",
        "ddist:2",
        "ddist:3",
        "ddist:4",
    )
    selection_index = 0

    def select_distinct_tile_body(
        self: FactorizedSearch,
        space: FactorizedTacticSpace,
        measure: Callable[[FactorizedTactic, bool], float],
    ) -> FactorizedTactic:
        """Choose alternating legal tile anchors without relying on runtime timings."""
        del self, measure
        nonlocal selection_index
        if len(space.tiles) < 2:
            raise RuntimeError("The capacity test requires two legal routing tiles")
        selected = space.anchor(space.tiles[selection_index % 2])
        selection_index += 1
        return selected

    monkeypatch.setattr(FactorizedSearch, "search", select_distinct_tile_body)
    with _temporary_environment(
        FLASHINFER_DIST_AWARE_AUTOTUNE="1",
        FLASHINFER_DA_DISTRIBUTIONS=",".join(distributions),
        FLASHINFER_DA_BASELINE_GUARD="0",
    ):
        # Alternate legal tile anchors across admitted distributions so capture exercises a
        # deterministic multi-body path at the old boundary, immediately above it, and at the
        # new immutable capacity. Candidate bodies are still profiled by the public DA lifecycle.
        with autotune(True, tuning_buckets=(shape.num_tokens,)):
            invoke()
        invoke()
        graph = _capture(invoke)
        leases = trtllm_moe_acquire_da_graph_leases(graph)

    try:
        graph.replay()
        torch.cuda.synchronize()
        diagnostic = _matching_from_logits_diagnostic(shape, distributions)
        assert diagnostic["policy"] == "da_switch"
        assert diagnostic["capture_fallback_reason"] is None
        assert diagnostic["topology"]["conditional_node_count"] == 1
        assert leases
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


# Auxiliary body ABI fallback


def test_lora_public_capture_uses_ordinary_moe_contract_when_da_is_enabled() -> None:
    """LoRA's auxiliary public outputs must bypass a DA switch and remain capturable."""
    require_sm100()
    shape = compact_shape(num_tokens=48)
    hidden, gemm1_weights, gemm2_weights = _bf16_weights(shape)
    factory = RoutingRealizationFactory()
    expert_ids, routing_weights = _realization(factory, shape, "ddist:4")
    packed = expert_ids.bitwise_left_shift(16).bitwise_or(
        routing_weights.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
    )
    lora_delta = torch.zeros(
        shape.num_tokens,
        shape.top_k,
        2 * shape.intermediate_size,
        device=hidden.device,
        dtype=torch.bfloat16,
    )
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

    with _temporary_environment(
        FLASHINFER_DIST_AWARE_AUTOTUNE="1",
        FLASHINFER_DA_DISTRIBUTIONS="uniform,ddist:4",
        FLASHINFER_DA_BASELINE_GUARD="0",
    ):
        # Ordinary autotuning remains enabled, but DA must not claim this multi-output ABI.
        with autotune(True, tuning_buckets=(shape.num_tokens,)):
            result = invoke()
        assert len(result) == 3
        graph = _capture(lambda: invoke()[0])

    try:
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
    finally:
        torch.cuda.synchronize()
        graph.reset()


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
