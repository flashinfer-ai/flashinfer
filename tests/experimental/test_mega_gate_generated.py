"""Independent routing values, mapping, current-stream and changed-input replay."""
import pytest
import torch
from flashinfer.mega_gate import prepare_mega_gate


@pytest.mark.parametrize("M,deterministic,physical", [
    (1, False, True), (3, False, True), (16, False, True), (128, False, True),
    (512, False, True), (1024, False, True), (2048, False, True),
    (4096, False, True), (8192, False, True), (16, True, True), (16, False, False)])
def test_routing_values_stream_and_replay(M, deterministic, physical):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("SM103a required")
    if torch.cuda.get_device_properties(0).multi_processor_count != 152:
        pytest.skip("The exported schedule requires 152 SMs")
    K, E, topk = 5120, 384, 6
    x = torch.ones((M, K), dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros((E, K), dtype=torch.bfloat16, device="cuda")
    bias = torch.arange(E, dtype=torch.float32, device="cuda")
    counts = torch.full((E,), 2, dtype=torch.int32, device="cuda") if physical else None
    mapping = torch.stack((torch.arange(E, device="cuda", dtype=torch.int32),
                           torch.arange(E, device="cuda", dtype=torch.int32) + E), 1) if physical else None
    unmapped = torch.empty((M, topk), dtype=torch.int64, device="cuda") if deterministic or not physical else None
    ep_rank = 7 if deterministic else 0
    plan = prepare_mega_gate(x, weight, topk, bias=bias, to_physical_map=mapping,
        logical_count=counts, unmapped_topk_idx=unmapped, ep_rank=ep_rank, deterministic=deterministic)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    stream.synchronize()
    for changed in (False, True, False):
        with torch.cuda.stream(stream):
            bias.copy_(torch.arange(E, device="cuda", dtype=torch.float32).flip(0) if changed else
                       torch.arange(E, device="cuda", dtype=torch.float32))
            plan.outputs[0].fill_(-1)
            plan.outputs[1].fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        logical = torch.arange(topk, device="cuda", dtype=torch.int64) if changed else torch.arange(E-1, E-topk-1, -1, device="cuda")
        expected = logical[None].expand(M, -1)
        if physical:
            duplicate = ((ep_rank + torch.arange(M, device="cuda") * 23333) % 2)[:, None]
            expected = expected + duplicate * E
        assert torch.equal(plan.outputs[0], expected)
        torch.testing.assert_close(plan.outputs[1], torch.full_like(plan.outputs[1], 0.25), atol=1e-5, rtol=1e-5)
        diff = ((plan.outputs[1].double() - 0.25).square().sum() /
                (plan.outputs[1].double().square().sum() + M * topk * 0.25**2)).item()
        assert diff < 1e-11
        if unmapped is not None:
            assert torch.equal(unmapped, logical[None].expand(M, -1))
