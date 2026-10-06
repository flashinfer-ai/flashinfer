"""Exact GPU oracle and CUDA-graph coverage for the W4A8 migration corpus."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace

import pytest
import torch

from b12x.moe._shared.kernels.reference import compare_to_reference, moe_reference_w4a8_mx

from b12x.testing.reference.helpers import make_tp_moe_fp4_binding, require_b12x
from .test_w4a8_dynamic_kernel import _run_w4a8_dynamic
from .test_w4a8_mx_tp_moe import (
    _E,
    _K,
    _N,
    _routed_inputs,
    _weights,
)


def _capture_and_replay(launch) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
        launch()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    return graph


def _assert_dynamic_oracle(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    context: object,
) -> None:
    assert output.abs().sum().item() > 0, (context, "all-zero output")
    metrics = compare_to_reference(output.float(), reference)
    assert metrics.cos > 0.999, (context, metrics)
    ref_rms = reference.float().square().mean().sqrt().item()
    assert metrics.rmse <= max(0.03 * ref_rms, 5e-3), (
        context,
        metrics,
        ref_rms,
    )


def _assert_dynamic_live_graph_replay(
    output: torch.Tensor,
    graph: torch.cuda.CUDAGraph,
    state: dict[str, object],
    *,
    context: object,
    num_experts: int,
) -> None:
    groups = (
        state["live_inputs"],
        state["read_only_inputs"],
        state["mutable_allocations"],
    )
    assert all(isinstance(group, dict) for group in groups)
    stable_tensors = {
        f"{group_index}:{name}": tensor
        for group_index, group in enumerate(groups)
        for name, tensor in group.items()
        if isinstance(tensor, torch.Tensor)
    }
    stable_ptrs = {name: tensor.data_ptr() for name, tensor in stable_tensors.items()}
    first_output = output.clone()

    live_inputs = state["live_inputs"]
    assert isinstance(live_inputs, dict)
    x = live_inputs["x"]
    topk_ids = live_inputs["topk_ids"]
    topk_weights = live_inputs["topk_weights"]
    assert isinstance(x, torch.Tensor)
    assert isinstance(topk_ids, torch.Tensor)
    assert isinstance(topk_weights, torch.Tensor)
    x.mul_(-0.75)
    topk_ids.add_(1).remainder_(num_experts)
    topk_weights.copy_(topk_weights.flip(-1))

    current_reference = state["current_reference"]
    assert callable(current_reference)
    reference = current_reference()
    output.zero_()
    allocated_before_replay = torch.cuda.memory_allocated()
    reserved_before_replay = torch.cuda.memory_reserved()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated_before_replay
    assert torch.cuda.memory_reserved() == reserved_before_replay
    assert {
        name: tensor.data_ptr() for name, tensor in stable_tensors.items()
    } == stable_ptrs
    _assert_dynamic_oracle(output, reference, context=(context, "live-replay"))
    assert not torch.equal(output, first_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tile_m", [16, 32, 64, 128], ids=lambda value: f"m{value}")
@pytest.mark.parametrize("recipe", ["w4a8_mx", "w4a8_nvfp4"], ids=["mx", "nvfp4"])
@pytest.mark.parametrize("activation", ["silu", "relu2"])
def test_w4a8_direct_tile_recipe_activation_matches_oracle_under_graph(
    tile_m: int,
    recipe: str,
    activation: str,
) -> None:
    """Cover every direct tile/recipe/activation specialization under replay."""

    require_b12x()
    # M16 also supplies the decode-sized boundary; the remaining inputs cross
    # their tile boundary so tail predicates execute in every specialization.
    m = {16: 1, 32: 33, 64: 65, 128: 129}[tile_m]
    output, reference, launch, state = _run_w4a8_dynamic(
        recipe=recipe,
        activation=activation,
        E=4,
        m=m,
        K=256,
        n=128,
        top_k=2,
        seed=1_000 + tile_m,
        tile_m=tile_m,
        return_launcher=True,
        return_state=True,
    )
    context = (tile_m, recipe, activation, m)
    _assert_dynamic_oracle(output, reference, context=context)

    graph = _capture_and_replay(launch)
    _assert_dynamic_live_graph_replay(
        output,
        graph,
        state,
        context=context,
        num_experts=4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("recipe", ["w4a8_mx", "w4a8_nvfp4"], ids=["mx", "nvfp4"])
@pytest.mark.parametrize("activation", ["silu", "relu2"])
def test_w4a8_packed_prefill_matches_oracle_under_graph(
    recipe: str,
    activation: str,
) -> None:
    """Exercise activation packing and routed prefill at serving-scale M."""

    require_b12x()
    output, reference, launch, state = _run_w4a8_dynamic(
        recipe=recipe,
        activation=activation,
        E=4,
        m=4096,
        K=256,
        n=128,
        top_k=2,
        seed=2_000,
        tile_m=64,
        return_launcher=True,
        return_state=True,
    )
    context = ("packed-prefill", recipe, activation)
    _assert_dynamic_oracle(output, reference, context=context)

    graph = _capture_and_replay(launch)
    _assert_dynamic_live_graph_replay(
        output,
        graph,
        state,
        context=context,
        num_experts=4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tile_m,m", [(32, 128), (64, 4096)])
def test_w4a8_materialized_routing_phase1_phase2_matches_oracle_under_graph(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    tile_m: int,
    m: int,
) -> None:
    """Prove the serving materialized route/phase1/phase2 prefill graph."""

    require_b12x()
    from b12x.moe.fused_moe._impl import b12x_moe_fp4, clear_tp_moe_caches

    monkeypatch.setenv("B12X_DYNAMIC_TILE_MN", f"{tile_m}x128")
    monkeypatch.setenv("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
    clear_tp_moe_caches()
    device = torch.device("cuda")
    weights = _weights(seed=3_000)
    x, topk_ids, topk_weights = _routed_inputs(m, 3_001)
    reference = moe_reference_w4a8_mx(
        x.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        None,
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        None,
        weights["alphas"],
        topk_ids,
        topk_weights,
        _E,
        _K,
        _N,
        activation="silu",
    )
    # Weight preparation below is deliberately destructive: the runtime layout
    # reuses the checkpoint allocations.  Build both logical-weight oracles
    # before that ownership transfer rather than interpreting repacked bytes as
    # the original checkpoint layout on live replay.
    live_x_value, live_ids_value, live_weights_value = _routed_inputs(m, 3_002)
    live_reference = moe_reference_w4a8_mx(
        live_x_value.float(),
        weights["w13_fp4"],
        weights["w13_mx"],
        None,
        weights["alphas"],
        weights["w2_fp4"],
        weights["w2_mx"],
        None,
        weights["alphas"],
        live_ids_value,
        live_weights_value,
        _E,
        _K,
        _N,
        activation="silu",
    )
    from b12x.moe import fused_moe

    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat.MXFP4_E8M0_K32,
            w13_layout=fused_moe.W13Layout.W13,
        ),
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A8, nonlinearity="silu", io_dtype=x.dtype,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=_E, hidden_size=_K, intermediate_size=_N,
        ),
    )
    prepared = fused_moe.prepare_weights(
        plan=weight_plan,
        weights=fused_moe.PackedWeights(
            w13=weights["w13_fp4"], w2=weights["w2_fp4"],
            w13_block_scales=weights["w13_mx"], w2_block_scales=weights["w2_mx"],
            w13_global_scales=weights["alphas"], w2_global_scales=weights["alphas"],
            input_scale=weights["input_scale"], intermediate_scale=weights["input_scale"],
        ),
    )
    output = torch.zeros(m, _K, dtype=torch.bfloat16, device=device)
    bindings = ExitStack()
    request.addfinalizer(bindings.close)
    binding = bindings.enter_context(make_tp_moe_fp4_binding(
        a=x,
        experts=prepared,
        topk_weights=topk_weights.contiguous(),
        topk_ids=topk_ids.contiguous(),
        output=output,
        input_scales_static=True,
        quant_mode="w4a8_mx",
    ))
    assert binding.deterministic_output
    assert binding.route_output is not None
    assert tuple(binding.route_output.shape) == (m * topk_ids.shape[1], _K)
    assert binding.materialized_intermediate is not None
    route_begin = binding.route_output.data_ptr()
    route_end = route_begin + (
        binding.route_output.numel() * binding.route_output.element_size()
    )
    intermediate_begin = binding.materialized_intermediate.data_ptr()
    intermediate_end = intermediate_begin + (
        binding.materialized_intermediate.numel()
        * binding.materialized_intermediate.element_size()
    )
    assert route_end <= intermediate_begin or intermediate_end <= route_begin

    def launch() -> None:
        b12x_moe_fp4(binding=binding)

    launch()
    torch.cuda.synchronize()
    context = ("materialized-prefill", tile_m, m)
    _assert_dynamic_oracle(output, reference, context=context)

    graph = _capture_and_replay(launch)
    first_output = output.clone()
    deterministic_replays = []
    for sentinel in (float("nan"), 997.0, -733.0):
        output.fill_(sentinel)
        # Shared activation rows are indexed by token, while deterministic
        # scatter metadata contains routed-pair indices. Poison padding so
        # a missing pair-to-token conversion cannot consume allocator data.
        for tensor in (
            binding.packed_input,
            binding.packed_input_scale,
            binding.materialized_intermediate,
        ):
            tensor.view(torch.uint8).fill_(0x7F)
        graph.replay()
        torch.cuda.synchronize()
        deterministic_replays.append(output.clone())
    assert all(
        torch.equal(deterministic_replays[0], replay)
        for replay in deterministic_replays[1:]
    )
    assert torch.equal(first_output, deterministic_replays[0])
    stable_tensors = {
        name: tensor
        for name, tensor in vars(binding).items()
        if isinstance(tensor, torch.Tensor)
    }
    stable_ptrs = {name: tensor.data_ptr() for name, tensor in stable_tensors.items()}
    live_x = binding.a
    live_topk_ids = binding.topk_ids
    live_topk_weights = binding.topk_weights
    live_x.copy_(live_x_value)
    live_topk_ids.copy_(live_ids_value)
    live_topk_weights.copy_(live_weights_value)
    output.zero_()
    allocated_before_replay = torch.cuda.memory_allocated()
    reserved_before_replay = torch.cuda.memory_reserved()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated_before_replay
    assert torch.cuda.memory_reserved() == reserved_before_replay
    assert {
        name: tensor.data_ptr() for name, tensor in stable_tensors.items()
    } == stable_ptrs
    graph_output = output.clone()
    output.zero_()
    launch()
    torch.cuda.synchronize()
    eager_output = output.clone()
    _assert_dynamic_oracle(
        eager_output,
        live_reference,
        context=(context, "live-eager-control"),
    )
    _assert_dynamic_oracle(
        graph_output,
        live_reference,
        context=(context, "live-replay"),
    )
    replay_eager_cos = torch.nn.functional.cosine_similarity(
        graph_output.float().flatten(),
        eager_output.float().flatten(),
        dim=0,
    ).item()
    assert replay_eager_cos > 0.9999, (context, replay_eager_cos)
    assert not torch.equal(graph_output, first_output)

    # The serving default is atomic scatter.  It must not reserve the routed
    # deterministic output (M * top-k * K BF16 values), but it must preserve
    # the same fixed-address graph and poison-overwrite contracts.
    monkeypatch.setenv("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "0")
    clear_tp_moe_caches()
    atomic_output = torch.full_like(output, float("nan"))
    atomic_binding = bindings.enter_context(make_tp_moe_fp4_binding(
        a=live_x,
        experts=prepared,
        topk_weights=live_topk_weights,
        topk_ids=live_topk_ids,
        output=atomic_output,
        input_scales_static=True,
        quant_mode="w4a8_mx",
    ))
    assert not atomic_binding.deterministic_output
    assert atomic_binding.route_output is not None
    assert tuple(atomic_binding.route_output.shape) == (1, _K)

    def atomic_launch() -> None:
        b12x_moe_fp4(binding=atomic_binding)

    atomic_launch()
    torch.cuda.synchronize()
    _assert_dynamic_oracle(
        atomic_output,
        live_reference,
        context=(context, "atomic-eager"),
    )
    atomic_graph = _capture_and_replay(atomic_launch)
    atomic_tensors = {
        name: tensor
        for name, tensor in vars(atomic_binding).items()
        if isinstance(tensor, torch.Tensor)
    }
    atomic_ptrs = {name: tensor.data_ptr() for name, tensor in atomic_tensors.items()}
    for sentinel in (float("nan"), 997.0, -733.0):
        atomic_output.fill_(sentinel)
        for tensor in (
            atomic_binding.packed_input,
            atomic_binding.packed_input_scale,
            atomic_binding.materialized_intermediate,
        ):
            tensor.view(torch.uint8).fill_(0x7F)
        allocated_before_replay = torch.cuda.memory_allocated()
        reserved_before_replay = torch.cuda.memory_reserved()
        atomic_graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated_before_replay
        assert torch.cuda.memory_reserved() == reserved_before_replay
        assert {
            name: tensor.data_ptr() for name, tensor in atomic_tensors.items()
        } == atomic_ptrs
        _assert_dynamic_oracle(
            atomic_output,
            live_reference,
            context=(context, "atomic-poison-replay", sentinel),
        )

    with pytest.raises(
        ValueError,
        match="deterministic route-output capacity mismatch",
    ):
        b12x_moe_fp4(binding=replace(binding, route_output=binding.route_output[:1]))
    del atomic_graph, graph
    bindings.close()
