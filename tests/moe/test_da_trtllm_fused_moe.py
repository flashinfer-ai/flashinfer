"""User-facing tuning, eager dispatch, capture, and replay coverage for TRTLLM DA MoE."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

import pytest
import torch
from cuda.bindings import runtime as cudart

from benchmarks.bench_moe_da import (
    _canonical_inputs,
    _capture,
    _matching_diagnostic,
    _prepare_precision,
    _realization,
    _temporary_environment,
)
from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    QuantConfig,
    QuantFormat,
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


def _assert_routing_metadata_slots_equivalent(actual, expected, expert_ids) -> None:
    """Check metadata and bijective expert-grouped mappings, allowing atomic row order."""
    assert torch.equal(actual.total_num_padded_tokens, expected.total_num_padded_tokens)
    assert torch.equal(actual.expert_weights, expected.expert_weights)
    assert torch.equal(actual.num_tokens_per_expert, expected.num_tokens_per_expert)
    assert torch.equal(actual.num_non_exiting_ctas, expected.num_non_exiting_ctas)
    num_ctas = int(actual.num_non_exiting_ctas.item())
    assert torch.equal(
        actual.cta_idx_xy_to_batch_idx[:num_ctas],
        expected.cta_idx_xy_to_batch_idx[:num_ctas],
    )
    assert torch.equal(
        actual.cta_idx_xy_to_mn_limit[:num_ctas],
        expected.cta_idx_xy_to_mn_limit[:num_ctas],
    )
    ids = expert_ids.flatten().long()
    counts = torch.bincount(ids, minlength=actual.num_tokens_per_expert.numel())
    assert torch.equal(actual.num_tokens_per_expert, counts)
    padded = (counts + actual.tile_n - 1) // actual.tile_n * actual.tile_n
    offsets = padded.cumsum(0) - padded
    num_rows = int(padded.sum().item())
    assert int(actual.total_num_padded_tokens.item()) == num_rows
    for slot in (actual, expected):
        permutation = slot.expanded_idx_to_permuted_idx.flatten().long()
        # Each assignment has one unique row within its expert's live prefix.
        assert permutation.unique().numel() == ids.numel()
        assert torch.all(permutation >= offsets[ids])
        assert torch.all(permutation < offsets[ids] + counts[ids])
        inverse = torch.full((num_rows,), -1, dtype=torch.int32, device=ids.device)
        inverse[permutation] = torch.arange(
            expert_ids.shape[0], dtype=torch.int32, device=ids.device
        ).repeat_interleave(expert_ids.shape[1])
        assert torch.equal(slot.permuted_idx_to_token_idx[:num_rows], inverse)


def _capture_kernel_node_count(invoke) -> int:
    """Capture one invocation and count kernel nodes without timing it."""
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    try:
        with torch.cuda.graph(graph):
            invoke()

        raw_graph = getattr(graph, "raw_cuda_graph", None)
        if raw_graph is None:
            pytest.skip("This PyTorch build does not expose raw CUDA Graph handles")
        graph_handle = cudart.cudaGraph_t(int(raw_graph()))
        status, _, node_count = cudart.cudaGraphGetNodes(graph_handle)
        assert status == cudart.cudaError_t.cudaSuccess
        status, nodes, _ = cudart.cudaGraphGetNodes(graph_handle, node_count)
        assert status == cudart.cudaError_t.cudaSuccess
        kernel_count = 0
        for node in nodes:
            status, node_type = cudart.cudaGraphNodeGetType(node)
            assert status == cudart.cudaError_t.cudaSuccess
            kernel_count += (
                node_type == cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel
            )
        return kernel_count
    finally:
        graph.reset()


# Fused routing capacity and exactness


@pytest.mark.parametrize("num_tokens", (2048, 2049, 8192))
@pytest.mark.parametrize("num_experts,top_k", ((256, 8), (512, 22)))
@pytest.mark.parametrize(
    "tile_ns",
    (
        (256,),
        (64, 128),
        (8, 16, 64),
        (64, 128, 192),
        (8, 16, 32, 64, 128),
        (8, 16, 32, 64, 96, 128, 192, 256),
    ),
)
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
    num_experts: int,
    top_k: int,
    routing_input_mode: RoutingInputMode,
    expert_id_dtype: torch.dtype,
    tile_ns: tuple[int, ...],
) -> None:
    """Fused and independent tiles must describe the same expert-grouped assignments."""
    require_sm100()

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
    # the same slots through the public one-tile ABI to establish equivalent metadata.
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
    for slot in fused.slots:
        slot.permuted_idx_to_token_idx.fill_(-12345)
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
        _assert_routing_metadata_slots_equivalent(actual, expected, expert_ids)


@pytest.mark.parametrize("num_tokens", (2730, 2731, 8192))
@pytest.mark.parametrize("representation", ("int16", "int32", "packed"))
@pytest.mark.parametrize("geometry", ((384, 48, 317, 6), (1024, 640, 317, 32)))
def test_native_da_selector_replays_live_local_histograms(
    num_tokens: int, representation: str, geometry: tuple[int, int, int, int]
) -> None:
    """Real SWITCH bodies follow a CPU oracle across the histogram launch boundary."""
    from flashinfer.fused_moe.core import (
        TrtllmDaSwitchCaptureState,
        _get_trtllm_da_body_capture_lock,
        _get_trtllm_da_body_capture_stream,
        get_trtllm_moe_sm100_module,
    )
    from flashinfer.fused_moe.da_moe import DAGraphTopology

    require_sm100()
    runtime = get_trtllm_moe_sm100_module()
    num_experts, local_experts, offset, top_k = geometry
    assignments = torch.arange(num_tokens * top_k).reshape(num_tokens, top_k)
    # Every row has distinct local experts and one remote expert. Include empty local work.
    routes = []
    for active in (local_experts, top_k, 2 * top_k):
        ids = assignments.remainder(active) + offset
        ids[:, -1] = 0
        routes.append(ids)
    routes.append(assignments.remainder(top_k))

    def spectrum(ids):
        local = ids[(ids >= offset) & (ids < offset + local_experts)] - offset
        counts = (
            torch.bincount(local, minlength=local_experts)
            .double()
            .sort(descending=True)
            .values
        )
        return torch.nn.functional.pad(counts, (0, num_experts - local_experts))

    reference = torch.stack([spectrum(ids) for ids in routes[:3]])
    reference = torch.nn.functional.normalize(reference, dim=1)
    spectra = reference.float().cuda()
    bodies = torch.tensor([0, 1, 0], dtype=torch.int32, device="cuda")
    weights = torch.full(
        (num_tokens, top_k), 1 / top_k, dtype=torch.bfloat16, device="cuda"
    )

    def encode(ids):
        if representation == "packed":
            return (ids.to(device="cuda", dtype=torch.int32) << 16) | (
                weights.view(torch.int16).to(torch.int32) & 0xFFFF
            )
        return ids.to(device="cuda", dtype=getattr(torch, representation))

    live = encode(routes[0])
    mode = (
        RoutingInputMode.PackedPrecomputed
        if representation == "packed"
        else RoutingInputMode.UnpackedPrecomputed
    )
    weight_arg = None if representation == "packed" else weights
    metadata = trtllm_moe_allocate_routing_metadata_multi_tile(
        live,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=offset,
        num_local_experts=local_experts,
        tile_ns=(8, 64),
        routing_input_mode=mode,
        topk_weights=weight_arg,
    )
    selected = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    observed = [torch.full_like(selected, -1) for _ in range(2)]
    executed = [torch.full_like(selected, -1) for _ in range(2)]
    live_inputs = (live, encode(routes[1]))
    scratch = runtime.allocate_da_selector_workspace(live, local_experts)
    device_index = torch.cuda.current_device()
    stream = _get_trtllm_da_body_capture_stream(device_index)
    for output, stamp in zip(observed, executed, strict=True):
        output.copy_(selected)
        stamp.fill_(0)
        stamp.fill_(1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            capture_id, previous_node = 0, 0
            for ids, output, stamp in zip(live_inputs, observed, executed, strict=True):
                # Different inputs share one serialized metadata/histogram lane.
                state = TrtllmDaSwitchCaptureState.from_native(
                    runtime.begin_da_switch_capture(
                        ids,
                        num_experts,
                        top_k,
                        offset,
                        local_experts,
                        list(metadata.tile_ns),
                        metadata.flat_tensors(),
                        int(mode),
                        weight_arg,
                        spectra,
                        bodies,
                        3,
                        selected,
                        scratch,
                        2,
                        capture_id,
                        previous_node,
                    )
                )
                with _get_trtllm_da_body_capture_lock(device_index):
                    for body, handle in enumerate(state.body_graph_handles):
                        runtime.begin_da_body_capture(
                            device_index, stream.handle, handle
                        )
                        with torch.cuda.stream(stream.external_stream):
                            stamp.fill_(body)
                        runtime.end_da_body_capture(device_index, stream.handle, handle)
                topology = DAGraphTopology.from_native(
                    runtime.finish_da_switch_capture(ids, state.to_native())
                )
                assert topology.is_selector_preamble_parallelizable
                assert topology.is_workspace_lane_serialized
                output.copy_(selected)
                capture_id = topology.capture_id
                previous_node = state.conditional_node_handle
        for ids in (*routes, routes[1], routes[0]):
            second_ids = routes[0] if torch.equal(ids, routes[1]) else routes[1]
            expected = []
            for live_ids, payload in zip(live_inputs, (ids, second_ids), strict=True):
                exemplar = (reference @ spectrum(payload)).argmax().item()
                expected.append((0, 1, 0)[exemplar])
                live_ids.copy_(encode(payload))
            for _ in range(3):
                # Poison retained scratch to catch missing writes and stale replay accumulation.
                scratch.fill_(123456)
                selected.fill_(-1)
                for output, stamp in zip(observed, executed, strict=True):
                    output.fill_(-1)
                    stamp.fill_(-1)
                graph.replay()
                torch.cuda.synchronize()
                assert [output.item() for output in observed] == expected
                assert [stamp.item() for stamp in executed] == expected
    finally:
        graph.reset()


@pytest.mark.parametrize("tile_ns", ((8,), (64, 128, 256), (64, 128, 192)))
@pytest.mark.parametrize("num_experts", (256, 512))
def test_long_multi_tile_routing_handles_local_and_nonlocal_experts(
    tile_ns, num_experts
) -> None:
    """The extended-capacity permutation pass must ignore a nonlocal route."""
    require_sm100()
    num_tokens = 2049
    local_expert_offset = 112
    num_local_experts = 56
    expert_ids = torch.empty((num_tokens, 2), device="cuda", dtype=torch.int32)
    expert_ids[:, 0] = local_expert_offset
    expert_ids[:, 1] = 0
    routing_weights = torch.full(
        (num_tokens, 2), 0.5, device="cuda", dtype=torch.bfloat16
    )

    metadata = trtllm_moe_allocate_routing_metadata_multi_tile(
        expert_ids,
        num_experts=num_experts,
        top_k=2,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        tile_ns=tile_ns,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        topk_weights=routing_weights,
    )
    for slot in metadata.slots:
        slot.permuted_idx_to_token_idx.fill_(-12345)
    populate_trtllm_moe_routing_metadata_(metadata, expert_ids, routing_weights)
    torch.cuda.synchronize()

    for slot in metadata.slots:
        expanded = slot.expanded_idx_to_permuted_idx.view(num_tokens, 2)
        assert torch.all(expanded[:, 0] >= 0)
        assert torch.all(expanded[:, 1] == -1)
        assert slot.num_tokens_per_expert[local_expert_offset] == num_tokens
        assert slot.num_tokens_per_expert[0] == 0
        num_rows = int(slot.total_num_padded_tokens.item())
        inverse = torch.full((num_rows,), -1, device="cuda", dtype=torch.int32)
        inverse[expanded[:, 0].long()] = torch.arange(
            num_tokens, device="cuda", dtype=torch.int32
        )
        assert torch.equal(slot.permuted_idx_to_token_idx[:num_rows], inverse)


@pytest.mark.parametrize(
    "routing_input_mode",
    (RoutingInputMode.PackedPrecomputed, RoutingInputMode.UnpackedPrecomputed),
)
@pytest.mark.parametrize("tile_ns", ((8, 32), (64, 128, 256)))
def test_fused_multi_tile_routing_supports_da_capacity_bounds(
    routing_input_mode: RoutingInputMode,
    tile_ns: tuple[int, ...],
) -> None:
    """The fused preamble must cover 1024 global experts and top-k 32."""
    require_sm100()
    num_tokens = 257
    num_experts = 1024
    top_k = 32
    local_expert_offset = 480
    num_local_experts = 64
    token_index = torch.arange(num_tokens, device="cuda", dtype=torch.int32).unsqueeze(
        1
    )
    topk_index = torch.arange(top_k, device="cuda", dtype=torch.int32).unsqueeze(0)
    expert_ids = (token_index * 37 + topk_index * 53) % num_experts
    routing_weights = torch.full(
        (num_tokens, top_k),
        1.0 / top_k,
        device="cuda",
        dtype=torch.bfloat16,
    )
    if routing_input_mode is RoutingInputMode.PackedPrecomputed:
        routing_ids = expert_ids.bitwise_left_shift(16).bitwise_or(
            routing_weights.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
        )
        routing_weights_arg = None
    else:
        routing_ids = expert_ids
        routing_weights_arg = routing_weights

    metadata = trtllm_moe_allocate_routing_metadata_multi_tile(
        routing_ids,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        tile_ns=tile_ns,
        routing_input_mode=routing_input_mode,
        topk_weights=routing_weights_arg,
    )
    populate_trtllm_moe_routing_metadata_(metadata, routing_ids, routing_weights_arg)
    torch.cuda.synchronize()

    expected_counts = torch.bincount(
        expert_ids.flatten().to(torch.int64), minlength=num_experts
    ).to(torch.int32)
    local_mask = torch.zeros(num_experts, device="cuda", dtype=torch.bool)
    local_mask[local_expert_offset : local_expert_offset + num_local_experts] = True
    expected_counts[~local_mask] = 0
    expected_live = local_mask[expert_ids.to(torch.int64)].flatten()
    for slot in metadata.slots:
        assert torch.equal(slot.num_tokens_per_expert, expected_counts)
        assert torch.equal(slot.expanded_idx_to_permuted_idx >= 0, expected_live)


def test_fused_multi_tile_routing_uses_one_kernel_node() -> None:
    """Fused population captures one kernel instead of one kernel per tile."""
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

    # Allocate graph-stable outputs once so capture contains only the exported fused
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

    populate_fused()
    populate_independent()
    torch.cuda.synchronize()
    assert _capture_kernel_node_count(populate_fused) == 1
    assert _capture_kernel_node_count(populate_independent) == len(tile_ns)


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
        hidden,
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
    )
    view = TrtllmFp4Config.prepare_weights(
        w1,
        w2,
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
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
