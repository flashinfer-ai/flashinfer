# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import math

import pytest
import torch

from flashinfer.deepseek_v41 import deepseek_v41_compressor_projection


def gate():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("CSA2 projection requires SM100/SM103")


def check(actual, expected):
    a, e = actual.double(), expected.double()
    error = a - e
    assert bool(torch.isfinite(a).all())
    assert float(error.norm() / e.norm().clamp_min(1e-20)) <= 1e-5
    assert float(error.abs().max() / e.abs().max().clamp_min(1e-20)) <= 2e-5


@pytest.mark.parametrize(
    "batch,promoted", [(1, False), (2, False), (4, True), (16, False)]
)
def test_compressor_projection_fp64_changed_graph_and_workspace(batch, promoted):
    gate()
    torch.manual_seed(41289 + batch)
    x = torch.randn(batch, 5120, device="cuda", dtype=torch.bfloat16)
    weights = [
        torch.randn(512, 5120, device="cuda") / math.sqrt(5120) for _ in range(2)
    ]
    if promoted:
        weights = [w.bfloat16().float() for w in weights]
    workspace = torch.empty(8, 2, batch, 512, device="cuda")
    kv, score = [torch.empty(batch, 512, device="cuda") for _ in range(2)]

    def run():
        return deepseek_v41_compressor_projection(
            x, *weights, workspace=workspace, kv=kv, score=score
        )

    def verify():
        for actual, weight in zip((kv, score), weights, strict=True):
            expected = torch.nn.functional.linear(x.double(), weight.double())
            check(actual, expected)

    run()
    verify()
    previous = kv.clone(), score.clone()
    run()
    for a, e in zip((kv, score), previous, strict=True):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.copy_(torch.randn_like(x))
    weights[0].mul_(0.75)
    weights[1].add_(0.03125)
    graph.replay()
    verify()
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_compressor_projection(
            x,
            *weights,
            workspace=weights[0].flatten()[: workspace.numel()].view_as(workspace),
            kv=kv,
            score=score,
        )
