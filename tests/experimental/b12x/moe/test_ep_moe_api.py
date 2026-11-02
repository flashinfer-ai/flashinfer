from __future__ import annotations

import pytest
import torch
from b12x.moe import ep_moe as public_ep_moe
from b12x.preparation import PreparedCall, PreparationSession

import b12x.moe.ep_moe._impl as ep_moe
from b12x.moe.ep_moe._impl import (
    EPMoEScratchCaps,
    _materialize_layout,
    prepare_ep_expert_map,
)
from b12x.moe.fused_moe._impl import plan_b12x_fp4_moe_weights
from b12x.moe._shared.kernels.w4a16.host import (
    max_packed_route_slots,
    route_block_sizes_for_capacity,
    route_pack_token_capacity,
)
from b12x.testing.reference.helpers import prepare_tp_moe_fp4_experts, run_tp_moe_fp4

swizzle_block_scale = (
    None  # placeholder: undefined in the b12x migration; set the real value
)


def _weight_plan(*, local_experts: int = 4, dtype: torch.dtype = torch.bfloat16):
    return plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=dtype,
        num_experts=local_experts,
        hidden_size=128,
        intermediate_size=128,
        w13_layout="w13",
    )


def test_prepare_ep_expert_map_accepts_linear_and_round_robin_placement() -> None:
    linear = prepare_ep_expert_map(
        torch.tensor([-1, -1, 0, 1, -1], dtype=torch.int32),
        local_num_experts=2,
        global_num_experts=5,
    )
    round_robin = prepare_ep_expert_map(
        torch.tensor([0, -1, 1, -1, 2], dtype=torch.int32),
        local_num_experts=3,
        global_num_experts=5,
    )

    assert linear.global_num_experts == 5
    assert linear.local_num_experts == 2
    assert round_robin.global_num_experts == 5
    assert round_robin.local_num_experts == 3


@pytest.mark.parametrize(
    ("values", "local_experts", "match"),
    [
        ([-1, 0, 0], 2, "exactly once"),
        ([-1, 0, 2], 2, "valid local expert ids"),
        ([-2, 0, 1], 2, "valid local expert ids"),
    ],
)
def test_prepare_ep_expert_map_rejects_unsafe_values(
    values: list[int],
    local_experts: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        prepare_ep_expert_map(
            torch.tensor(values, dtype=torch.int32),
            local_num_experts=local_experts,
        )


def test_prepared_ep_expert_map_rejects_mutation() -> None:
    tensor = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    prepared = prepare_ep_expert_map(tensor, local_num_experts=2)

    tensor.copy_(torch.tensor([-1, 0, -1, 1], dtype=torch.int32))

    with pytest.raises(RuntimeError, match="mutated"):
        prepared.validate_static()


def test_prepare_ep_expert_map_accepts_inference_tensor() -> None:
    with torch.inference_mode():
        tensor = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
        prepared = prepare_ep_expert_map(tensor, local_num_experts=2)

    prepared.validate_static()


def test_ep_scratch_sizes_global_route_state_separately_from_local_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ep_moe, "get_num_sm", lambda _device: 120)
    plan = _materialize_layout(
        EPMoEScratchCaps(
            max_tokens=24,
            num_topk=2,
            global_num_experts=10,
            device="cpu",
            weight_plan=_weight_plan(local_experts=3),
        )
    )

    layout = {spec.name: spec for spec in plan._layout}
    assert layout["expert_offsets"].elements == 11
    assert layout["intermediate_cache2"].elements == 24 * 2 * 128
    assert plan.scratch_specs()[0].name == "ep_moe.scratch"
    assert plan.scratch_specs()[0].dtype == torch.uint8


def test_ep_scratch_reserves_route_pack_power_of_two_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ep_moe, "get_num_sm", lambda _device: 120)
    plan = _materialize_layout(
        EPMoEScratchCaps(
            max_tokens=3,
            num_topk=2,
            global_num_experts=10,
            device="cpu",
            weight_plan=_weight_plan(local_experts=3),
        )
    )

    layout = {spec.name: spec for spec in plan._layout}
    block_sizes = route_block_sizes_for_capacity(3, 2, 10)
    capacity_rows = route_pack_token_capacity(3, 2) * 2
    expected_slots = max(
        max_packed_route_slots(capacity_rows, block_size, 10)
        for block_size in block_sizes
    )
    expected_blocks = max(
        (max_packed_route_slots(capacity_rows, block_size, 10) + block_size - 1)
        // block_size
        for block_size in block_sizes
    )
    assert layout["packed_route_indices"].elements == expected_slots
    assert layout["block_expert_ids"].elements == expected_blocks
    assert layout["intermediate_cache2"].elements == 3 * 2 * 128


def test_ep_contract_requires_w4a16_bf16() -> None:
    with pytest.raises(TypeError, match="BF16"):
        EPMoEScratchCaps(
            max_tokens=1,
            num_topk=1,
            global_num_experts=4,
            device="cpu",
            weight_plan=_weight_plan(local_experts=2, dtype=torch.float16),
        )


def _make_modelopt_weights(
    *,
    experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[torch.Tensor, ...]:
    w13 = torch.randint(
        0,
        256,
        (experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = swizzle_block_scale(
        (
            torch.rand(experts, 2 * intermediate_size, hidden_size // 16, device="cuda")
            * 0.25
            + 0.03125
        ).to(torch.float8_e4m3fn)
    )
    w2_scale = swizzle_block_scale(
        (
            torch.rand(experts, hidden_size, intermediate_size // 16, device="cuda")
            * 0.25
            + 0.03125
        ).to(torch.float8_e4m3fn)
    )
    w13_alpha = (torch.rand(experts, device="cuda") * 0.1 + 0.05).float()
    w2_alpha = (torch.rand(experts, device="cuda") * 0.1 + 0.05).float()
    return w13, w13_scale, w13_alpha, w2, w2_scale, w2_alpha


def _prepare_experts(
    a: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    expert_ids: torch.Tensor,
):
    w13, w13_scale, w13_alpha, w2, w2_scale, w2_alpha = weights
    selected = expert_ids.to(device=w13.device, dtype=torch.long)
    local_e = int(selected.numel())
    unit = torch.ones(local_e, dtype=torch.float32, device=a.device)
    return prepare_tp_moe_fp4_experts(
        a=a,
        a1_gscale=unit,
        w1_fp4=w13.index_select(0, selected).clone(),
        w1_blockscale=w13_scale.index_select(0, selected).clone(),
        w1_alphas=w13_alpha.index_select(0, selected).clone(),
        a2_gscale=unit,
        w2_fp4=w2.index_select(0, selected).clone(),
        w2_blockscale=w2_scale.index_select(0, selected).clone(),
        w2_alphas=w2_alpha.index_select(0, selected).clone(),
        activation="silu",
        quant_mode="w4a16",
    )


def _run_ep_rank(
    *,
    a: torch.Tensor,
    experts,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
) -> tuple[torch.Tensor, object]:
    prepared_map = prepare_ep_expert_map(
        expert_map,
        local_num_experts=experts.num_experts,
        global_num_experts=int(expert_map.numel()),
        device=a.device,
    )
    output = torch.empty_like(a)
    caps = EPMoEScratchCaps(
        max_tokens=int(a.shape[0]),
        num_topk=int(topk_ids.shape[1]),
        global_num_experts=int(expert_map.numel()),
        device=a.device,
        weight_plan=experts.plan,
    )
    declaration = public_ep_moe.plan(
        caps,
        expert_map=prepared_map,
        invocation=public_ep_moe.invocation_from_tensors(
            a=a,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
            expert_map=prepared_map,
        ),
    )
    scratch = {}

    def prepare_call(state):
        spec = state.layout.scratch_specs()[0]
        scratch["value"] = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind(
            scratch=scratch["value"],
            a=a,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
        )
        return PreparedCall(run=lambda: state.run(binding))

    session = PreparationSession(device=a.device, autotune=False, compile_workers=2)
    session.prepare(
        (
            declaration.request(
                name=f"ep-{prepared_map.tensor.data_ptr()}",
                prepare_call=prepare_call,
            ),
        )
    )
    plan = declaration
    binding = public_ep_moe.bind(
        plan,
        scratch=scratch["value"],
        a=a,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
    )
    return public_ep_moe.run(binding=binding), binding


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ep_rank_partials_sum_to_full_w4a16_moe() -> None:
    torch.manual_seed(20260630)
    global_e, hidden_size, intermediate_size = 7, 128, 128
    m, topk = 24, 3
    a = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    weights = _make_modelopt_weights(
        experts=global_e,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    topk_ids = torch.randint(
        0,
        global_e,
        (m, topk),
        dtype=torch.int32,
        device="cuda",
    )
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    global_experts = _prepare_experts(
        a,
        weights,
        torch.arange(global_e),
    )
    expected = run_tp_moe_fp4(
        a=a,
        experts=global_experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=torch.empty_like(a),
        quant_mode="w4a16",
    )

    partials = []
    for rank in range(2):
        global_ids = torch.arange(rank, global_e, 2)
        local_experts = _prepare_experts(a, weights, global_ids)
        expert_map = torch.full(
            (global_e,),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        expert_map[global_ids.to(device="cuda")] = torch.arange(
            global_ids.numel(),
            dtype=torch.int32,
            device="cuda",
        )
        partial, _ = _run_ep_rank(
            a=a,
            experts=local_experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
        )
        partials.append(partial.clone())

    actual = partials[0] + partials[1]
    torch.cuda.synchronize()
    assert int(torch.count_nonzero(expected).item()) > 0
    assert int(torch.count_nonzero(actual).item()) > 0
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(),
        expected.float().flatten(),
        dim=0,
    )
    assert float(cosine.item()) > 0.999
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ep_binding_replays_with_changed_routes_under_cuda_graph() -> None:
    torch.manual_seed(20260631)
    global_e, hidden_size, intermediate_size = 4, 128, 128
    m, topk = 8, 2
    a = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    weights = _make_modelopt_weights(
        experts=global_e,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    global_ids = torch.tensor([0, 2])
    experts = _prepare_experts(a, weights, global_ids)
    expert_map = torch.tensor([0, -1, 1, -1], dtype=torch.int32, device="cuda")
    topk_ids = (
        torch.tensor([[1, 3]], dtype=torch.int32, device="cuda")
        .expand(m, -1)
        .contiguous()
    )
    topk_weights = torch.full((m, topk), 0.5, dtype=torch.float32, device="cuda")
    nonlocal_output, binding = _run_ep_rank(
        a=a,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )
    torch.cuda.synchronize()
    assert int(torch.count_nonzero(nonlocal_output).item()) == 0

    topk_ids.copy_(
        torch.tensor([[0, 1]], dtype=torch.int32, device="cuda").expand(m, -1)
    )
    # Resolve all compiled route-pack/GEMM launch variants before capture.
    binding.run()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        binding.run()

    topk_ids.copy_(
        torch.tensor([[2, 3]], dtype=torch.int32, device="cuda").expand(m, -1)
    )
    graph.replay()
    torch.cuda.synchronize()
    replayed = binding.output.clone()
    binding.run()
    torch.cuda.synchronize()

    torch.testing.assert_close(replayed, binding.output, rtol=0, atol=0)
