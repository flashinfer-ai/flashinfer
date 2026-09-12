# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import math

import pytest
import torch

from flashinfer.deepseek_v41 import (
    deepseek_v41_compressor_decode,
    deepseek_v41_compressor_projection,
    deepseek_v41_compressor_step,
)


@pytest.mark.parametrize("batch", [1, 4, 16])
@pytest.mark.parametrize("norm_dtype", [torch.float32, torch.bfloat16])
def test_fused_compressor_matches_components_bitwise_with_changing_graph(
    batch, norm_dtype
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("CSA2 compressor step requires SM100/SM103")
    torch.manual_seed(41317 + batch)
    x = torch.randn(batch, 5120, device="cuda", dtype=torch.bfloat16)
    weights = [
        torch.randn(512, 5120, device="cuda") / math.sqrt(5120) for _ in range(2)
    ]
    norm = (torch.rand(512, device="cuda") + 0.5).to(norm_dtype)
    starts = torch.arange(batch, device="cuda", dtype=torch.int32)
    states = [torch.randn(batch, 2, 512, device="cuda") for _ in range(2)]
    control_states = [state.clone() for state in states]
    workspace = torch.empty(8, 2, batch, 512, device="cuda")
    control_workspace = torch.empty_like(workspace)
    projected = [torch.empty(batch, 512, device="cuda") for _ in range(2)]
    out = torch.full((batch, 512), 17.0, device="cuda", dtype=torch.bfloat16)
    control_out = out.clone()
    positions = torch.empty(batch, device="cuda", dtype=torch.int32)
    control_positions = torch.empty_like(positions)

    def run():
        return deepseek_v41_compressor_step(
            x,
            *weights,
            *states,
            norm,
            starts,
            workspace=workspace,
            out=out,
            positions=positions,
        )

    def verify():
        deepseek_v41_compressor_projection(
            x,
            *weights,
            workspace=control_workspace,
            kv=projected[0],
            score=projected[1],
        )
        deepseek_v41_compressor_decode(
            *projected,
            *control_states,
            norm,
            starts,
            out=control_out,
            positions=control_positions,
        )
        for actual, expected in zip(
            (*states, out, positions),
            (*control_states, control_out, control_positions),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    run()
    verify()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for step in range(5):
        starts.add_(1)
        x.copy_(torch.randn_like(x))
        weights[0].mul_(0.875)
        weights[1].add_(0.000125 * (step + 1))
        norm.mul_(0.9375)
        graph.replay()
        verify()
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_compressor_step(
            x,
            *weights,
            *states,
            norm,
            starts,
            workspace=weights[0].flatten()[: workspace.numel()].view_as(workspace),
            out=out,
            positions=positions,
        )
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_compressor_step(
            x,
            *weights,
            states[0],
            states[0],
            norm,
            starts,
            workspace=workspace,
            out=out,
            positions=positions,
        )
