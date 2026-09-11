# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import math

import pytest
import torch

from flashinfer.deepseek_v41 import deepseek_v41_index_key


@pytest.mark.parametrize("batch", [1, 2, 4, 16])
@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
def test_index_key_fp64_boundaries_changing_graph_and_alias(batch, norm_dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("index-key projection requires SM100/SM103")
    torch.manual_seed(41389 + batch)
    x = torch.randn(batch, 512, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    norm = (torch.rand(128, device="cuda") + 0.5).to(norm_dtype)
    out = torch.empty(batch, 128, device="cuda", dtype=torch.bfloat16)

    def run():
        return deepseek_v41_index_key(x, weight, norm, out=out)

    def verify():
        projected = (
            torch.nn.functional.linear(x.double(), weight.double()).bfloat16().double()
        )
        expected = (
            projected
            * torch.rsqrt(projected.square().mean(-1, keepdim=True) + 1e-20)
            * norm.double()
        ).bfloat16()
        delta = out.double() - expected.double()
        assert bool(torch.isfinite(out).all())
        assert float(delta.norm() / expected.double().norm().clamp_min(1e-20)) <= 0.004
        assert (
            float(delta.abs().max() / expected.double().abs().max().clamp_min(1e-20))
            <= 0.008
        )
        projected_math = torch.nn.functional.linear(x, weight).float()
        math_result = (
            projected_math
            * torch.rsqrt(projected_math.square().mean(-1, keepdim=True) + 1e-20)
            * norm.float()
        ).bfloat16()
        delta = out.double() - math_result.double()
        assert (
            float(delta.norm() / math_result.double().norm().clamp_min(1e-20)) <= 0.004
        )
        assert (
            float(delta.abs().max() / math_result.double().abs().max().clamp_min(1e-20))
            <= 0.008
        )

    run()
    verify()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        weight.mul_(0.875)
        norm.mul_(0.9375)
        graph.replay()
        verify()
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_index_key(
            x, weight, norm, out=x.flatten()[: batch * 128].view_as(out)
        )
    x.zero_()
    graph.replay()
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    with pytest.raises(ValueError, match="inference-only"):
        deepseek_v41_index_key(x.requires_grad_(), weight, norm, out=out)
