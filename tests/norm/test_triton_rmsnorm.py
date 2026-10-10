import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from flashinfer.triton.norm import rms_norm


@pytest.mark.parametrize("rows", [1, 4, 4096])
@pytest.mark.parametrize("width", [512, 1536, 4096])
@pytest.mark.parametrize("round_before_weight", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_strided_rmsnorm_graph(rows, width, round_before_weight, compiled):
    torch.manual_seed(53)
    stride = max(2048, width + 512)
    source_storage = torch.empty((rows, stride), device="cuda", dtype=torch.bfloat16)
    source = source_storage[:, stride - width :]
    output = torch.empty_like(source)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    source.normal_()

    def run():
        rms_norm(source, weight, output, 1e-6, round_before_weight=round_before_weight)

    if compiled:
        run = torch.compile(make_fx(run)(), fullgraph=True, dynamic=False)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for step in range(3):
        if step == 0:
            source[:, ::2] = 1
            source[:, 1::2] = 2
        else:
            source.normal_()
        value = source.double()
        normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        if round_before_weight:
            normalized = normalized.to(source.dtype).double()
        expected = (normalized * weight.double()).to(output.dtype)
        output.fill_(float("nan"))
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        graph.replay()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        for key in ("allocation.all.allocated", "allocation.all.freed"):
            assert before[key] == after[key]
        if step == 0:
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        else:
            assert torch.isfinite(output).all()
            ulps = (output.view(torch.int16).int() - expected.view(torch.int16).int()).abs()
            # Rounding before and after weighting permits two BF16 rounding steps.
            assert ulps.max() <= (2 if round_before_weight else 1)
