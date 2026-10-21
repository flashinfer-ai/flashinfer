import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe._shared.kernels.w4a16.kernel.sum_launch import (
    _w4a16_topk_sum_launch_flat,
    compile_w4a16_topk_sum,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_route_sum_ignores_unwritten_routes(ids_dtype, mapped, weighted):
    torch.manual_seed(911)
    rows, topk, hidden, experts = 64, 9, 256, 8
    ids = torch.randint(experts, (rows, topk), device="cuda", dtype=ids_dtype)
    values = torch.empty((rows, topk, hidden), device="cuda", dtype=torch.bfloat16)
    weights = torch.randint(1, 5, (rows, topk), device="cuda").float() / 4
    output = torch.empty((rows, hidden), device="cuda", dtype=torch.bfloat16)
    expert_map = (
        torch.tensor([3, 2, -1, 1, 0, -1, 4, 5], device="cuda", dtype=torch.int32)
        if mapped
        else None
    )
    launcher = compile_w4a16_topk_sum(
        m=rows,
        topk=topk,
        hidden_size=hidden,
        route_ids_dtype=ids_dtype,
        num_experts=experts if mapped else 0,
        route_num_experts=experts if mapped else 0,
        use_expert_map=mapped,
        apply_topk_weights=weighted,
    )

    def run():
        _w4a16_topk_sum_launch_flat(
            values,
            output,
            rows,
            topk,
            hidden,
            "bf16",
            torch.cuda.current_stream().cuda_stream,
            num_experts=experts,
            route_expert_ids=ids,
            expert_map=expert_map,
            topk_weights=weights if weighted else None,
            launcher=launcher,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard(), torch.cuda.graph(graph):
        run()
    for live in (64, 40, 8, 0, 56, 64):
        ids.random_(0, experts)
        ids[live:] = -1
        ids[0, 0] = experts
        ids[0, 1] = -7
        valid = (ids >= 0) & (ids < experts)
        if mapped:
            valid &= expert_map[ids.clamp(0, experts - 1).long()] >= 0
        values.normal_()
        values[~valid] = float("nan")
        expected_values = values.float().masked_fill(~valid[..., None], 0)
        if weighted:
            expected_values *= weights[..., None]
        expected = expected_values.sum(dim=1).bfloat16()
        output.fill_(float("nan"))
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
