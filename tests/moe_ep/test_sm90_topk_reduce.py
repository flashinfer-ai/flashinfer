# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.arch_hopper
@pytest.mark.parametrize("hidden", [12, 256, 4096])
@pytest.mark.parametrize("topk", [2, 17])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_sm90_bf16_reduction_graph(hidden, topk, weighted, masked):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires Hopper")

    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.topk_reduce import (
        CombineFormat,
        TopkReduce,
    )

    tokens = 17
    terms = torch.empty(tokens, topk, hidden, device="cuda", dtype=torch.bfloat16)
    scores = torch.empty(tokens, topk, device="cuda", dtype=torch.float32)
    ids = torch.empty(tokens, topk, device="cuda", dtype=torch.int64)
    output = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    offsets = torch.arange(terms.numel(), device="cuda").reshape(terms.shape)
    slots = torch.arange(ids.numel(), device="cuda").reshape(ids.shape)
    alignment = 8 if hidden == 12 else 16
    arguments = dict(
        combine_quant=from_dlpack(terms, assumed_align=alignment),
        combine_sf=None,
        reduced_output=from_dlpack(output, assumed_align=alignment),
        topk_score=from_dlpack(scores, assumed_align=4) if weighted else None,
        stream=CUstream(torch.cuda.current_stream().cuda_stream),
        topk_idx=from_dlpack(ids, assumed_align=8) if masked else None,
    )
    reducer = TopkReduce(hidden, topk, CombineFormat.parse("bf16"), sm_arch="sm_90a")
    executable = cute.compile(reducer, **arguments)

    def prepare(step):
        values = ((offsets + step) % 9 - 4).float() * 0.5
        weights = ((slots + step) % 5 - 2).float() * 0.5
        ids.copy_((slots + step) % 3)
        if masked:
            ids.masked_fill_((slots + step) % 3 == 0, -1)
            ids[0].fill_(-1)
        live = ids >= 0
        terms.copy_(values)
        scores.copy_(weights)
        if masked:
            # Invalid slots may contain stale NaNs in both data and scores.
            terms.masked_fill_(~live[:, :, None], float("nan"))
            scores.masked_fill_(~live, float("nan"))
        output.fill_(float("nan"))
        expected = torch.zeros(tokens, hidden, device="cuda", dtype=torch.float32)
        for slot in range(topk):
            value = values[:, slot]
            if weighted:
                value = value * weights[:, slot, None]
            expected += torch.where(live[:, slot, None], value, 0)
        return expected.bfloat16()

    expected = prepare(0)
    executable(**arguments)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            arguments["stream"] = CUstream(torch.cuda.current_stream().cuda_stream)
            executable(**arguments)
        for step in range(1, 4):
            expected = prepare(step)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        graph.reset()
