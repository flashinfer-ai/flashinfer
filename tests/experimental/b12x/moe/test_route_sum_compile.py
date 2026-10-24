"""Deterministic route reduction remains present in compiled CUDA graphs."""

import pytest
import torch

from b12x.moe.fused_moe._impl import _launch_dynamic_topk_sum
from ..conftest import require_b12x


@pytest.mark.parametrize("tokens", [1, 4, 512])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_compiled_route_sum(tokens, output_dtype):
    device = require_b12x()
    routes = torch.randn(tokens * 10, 2560, dtype=torch.bfloat16, device=device)
    output = torch.empty(tokens, 2560, dtype=output_dtype, device=device)

    def run():
        _launch_dynamic_topk_sum(
            route_output=routes,
            output=output,
            m=tokens,
            num_topk=10,
            k=2560,
            stream=None,
        )

    run()
    expected = output.clone()
    torch.testing.assert_close(
        output, routes.reshape(tokens, 10, 2560).float().sum(1).to(output_dtype)
    )
    compiled = torch.compile(run, fullgraph=True, options={"triton.cudagraphs": False})
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for _ in range(2):
        routes.normal_()
        run()
        expected.copy_(output)
        output.fill_(float("nan"))
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
