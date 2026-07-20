# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""moe.ep_moe: two-rank expert-parallel partials must sum to the full fused
MoE output (the strongest single consistency check across both moe ops), and
a captured binding must observe changed routes on replay.

Curated from b12x tests/test_ep_moe_api.py; scratch-sizing and contract
validation unit tests stay in the b12x repo.
"""

from __future__ import annotations

import torch

from flashinfer.experimental.sm12x.moe import ep_moe

from .._reference.helpers import run_tp_moe_fp4
from ..conftest import require_sm12x
from .test_fused_moe import make_modelopt_weights, prepare_experts


def _run_ep_rank(
    *,
    a: torch.Tensor,
    experts,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
):
    prepared_map = ep_moe.prepare_expert_map(
        expert_map,
        local_num_experts=experts.num_experts,
        global_num_experts=int(expert_map.numel()),
        device=a.device,
    )
    plan = ep_moe.plan(
        ep_moe.Caps(
            max_tokens=int(a.shape[0]),
            num_topk=int(topk_ids.shape[1]),
            global_num_experts=int(expert_map.numel()),
            device=a.device,
            weight_plan=experts.plan,
        )
    )
    scratch = torch.empty(
        plan.scratch_specs()[0].shape, dtype=torch.uint8, device=a.device
    )
    output = torch.empty_like(a)
    binding = ep_moe.bind(
        plan,
        scratch=scratch,
        a=a,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=prepared_map,
        output=output,
    )
    return ep_moe.run(binding=binding), binding


def test_ep_rank_partials_sum_to_full_fused_moe() -> None:
    require_sm12x()
    torch.manual_seed(20260630)

    global_e, hidden_size, intermediate_size = 7, 128, 128
    m, topk = 24, 3
    a = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    weights = make_modelopt_weights(
        experts=global_e, hidden_size=hidden_size, intermediate_size=intermediate_size
    )
    topk_ids = torch.randint(0, global_e, (m, topk), dtype=torch.int32, device="cuda")
    topk_weights = torch.softmax(torch.randn(m, topk, device="cuda"), dim=-1)

    global_experts = prepare_experts(a, weights, torch.arange(global_e))
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
        local_experts = prepare_experts(a, weights, global_ids)
        expert_map = torch.full((global_e,), -1, dtype=torch.int32, device="cuda")
        expert_map[global_ids.to(device="cuda")] = torch.arange(
            global_ids.numel(), dtype=torch.int32, device="cuda"
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
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    assert float(cosine.item()) > 0.999
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)


def test_binding_replays_with_changed_routes_under_cuda_graph() -> None:
    require_sm12x()
    torch.manual_seed(20260631)

    global_e, hidden_size, intermediate_size = 4, 128, 128
    m, topk = 8, 2
    a = (torch.randn(m, hidden_size, device="cuda") * 0.25).to(torch.bfloat16)
    weights = make_modelopt_weights(
        experts=global_e, hidden_size=hidden_size, intermediate_size=intermediate_size
    )
    global_ids = torch.tensor([0, 2])
    experts = prepare_experts(a, weights, global_ids)
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
    # No local expert routed -> partial must be exactly zero.
    assert int(torch.count_nonzero(nonlocal_output).item()) == 0

    topk_ids.copy_(
        torch.tensor([[0, 1]], dtype=torch.int32, device="cuda").expand(m, -1)
    )
    binding.run()  # resolve all route-pack/GEMM variants before capture
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        binding.run()

    # Replay must observe route changes staged after capture.
    topk_ids.copy_(
        torch.tensor([[2, 3]], dtype=torch.int32, device="cuda").expand(m, -1)
    )
    graph.replay()
    torch.cuda.synchronize()
    replayed = binding.output.clone()
    binding.run()
    torch.cuda.synchronize()

    torch.testing.assert_close(replayed, binding.output, rtol=0, atol=0)
