# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quantized combine ignores masked payloads, scales and scores on replay."""

import pytest
import torch


@pytest.mark.arch_rubin
@pytest.mark.parametrize("combine", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tokens,hidden", [(1, 256), (33, 768)])
@pytest.mark.parametrize("topk", [8, 33])
@pytest.mark.parametrize("weighted", [False, True])
def test_quantized_masked_combine_graph(combine, tokens, hidden, topk, weighted):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Requires SM107")
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as pkg

    pkg.require_sm107_dsl()
    pkg.bootstrap_paths()
    from sources.kernel_src.rubin.inference.mega.topk_reduce import TopkReduce
    from sources.quant_def import CombineFormat

    experts = 128
    fp4 = combine == "nvfp4"
    payload = torch.empty(
        (tokens, topk, hidden // (2 if fp4 else 1)),
        device="cuda",
        dtype=torch.uint8,
    )
    scales = torch.empty(
        (tokens, topk, hidden // (16 if fp4 else 32)),
        device="cuda",
        dtype=torch.bfloat16 if fp4 else torch.float8_e8m0fnu,
    )
    ids = torch.zeros((tokens, topk), device="cuda", dtype=torch.int32)
    scores = torch.ones((tokens, topk), device="cuda", dtype=torch.float32)
    output = torch.empty((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    terms = from_dlpack(
        payload.view(torch.float4_e2m1fn_x2 if fp4 else torch.float8_e4m3fn),
        assumed_align=16,
    )
    reducer = TopkReduce(
        hidden,
        topk,
        CombineFormat.parse("16e2m1xbf16" if fp4 else "32e4m3xe8m0"),
        num_experts=experts,
    )
    kwargs = dict(
        combine_quant=terms,
        combine_sf=from_dlpack(scales, assumed_align=16),
        reduced_output=from_dlpack(output, assumed_align=16),
        topk_score=from_dlpack(scores, assumed_align=4) if weighted else None,
        topk_indices=from_dlpack(ids, assumed_align=4),
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    executable = cute.compile(reducer, **kwargs)
    ids.fill_(-1)
    executable(**kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            kwargs["stream"] = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            executable(**kwargs)
        base = torch.arange(tokens * topk, device="cuda", dtype=torch.int32).reshape_as(
            ids
        )
        for iteration, state in enumerate(
            ("active", "mixed", "empty", "last", "restore")
        ):
            ids.copy_(base % experts)
            if state == "mixed":
                ids.view(-1)[::3] = -1
            elif state == "empty":
                ids.fill_(-1)
            elif state == "last":
                ids.fill_(-1)
                ids[-1, -1] = experts - 1
            valid = ids != -1
            value = 1 << (iteration % 3)
            if fp4:
                payload.fill_((2 + 2 * (iteration % 3)) * 17)
                scales.fill_(6)
                scales.masked_fill_(~valid.unsqueeze(-1), float("nan"))
            else:
                payload.view(torch.float8_e4m3fn).fill_(value)
                scales.view(torch.uint8).fill_(127)
                scales.view(torch.uint8).masked_fill_(~valid.unsqueeze(-1), 255)
            payload.masked_fill_(~valid.unsqueeze(-1), 255)
            scores.copy_(((base % 4).float() + 1) * 0.25)
            scores.masked_fill_(
                ~valid, (float("nan"), float("inf"), -float("inf"))[iteration % 3]
            )
            expected = torch.zeros((tokens, hidden), device="cuda", dtype=torch.float32)
            for k in range(topk):
                contribution = torch.where(valid[:, k], value, 0).float()
                if weighted:
                    contribution *= torch.where(valid[:, k], scores[:, k], 0.0)
                expected += contribution[:, None]
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected.bfloat16(), rtol=0, atol=0)
    finally:
        graph.reset()
