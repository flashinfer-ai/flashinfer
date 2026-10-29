# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 combine must ignore stale payloads and scores for invalid routes."""

import pytest
import torch


@pytest.mark.arch_rubin
@pytest.mark.parametrize("tokens", [1, 33])
@pytest.mark.parametrize("hidden", [256, 520])
@pytest.mark.parametrize("topk", [16, 17, 32, 33])
@pytest.mark.parametrize("weighted", [False, True])
def test_masked_combine_graph(tokens, hidden, topk, weighted):
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
    # Exercise the advertised 16-byte alignment without relying on 32-byte alignment.
    terms = torch.empty(
        tokens * topk * hidden + 8, device="cuda", dtype=torch.bfloat16
    )[8:].reshape(tokens, topk, hidden)
    output = torch.empty(tokens * hidden + 8, device="cuda", dtype=torch.bfloat16)[
        8:
    ].reshape(tokens, hidden)
    assert terms.data_ptr() % 32 == output.data_ptr() % 32 == 16
    ids = torch.zeros((tokens, topk), device="cuda", dtype=torch.int32)
    scores = torch.ones((tokens, topk), device="cuda", dtype=torch.float32)
    terms.fill_(1)
    reducer = TopkReduce(hidden, topk, CombineFormat.parse("bf16"), num_experts=experts)
    kwargs = dict(
        combine_quant=from_dlpack(terms, assumed_align=16),
        combine_sf=None,
        reduced_output=from_dlpack(output, assumed_align=16),
        topk_score=from_dlpack(scores, assumed_align=4) if weighted else None,
        topk_indices=from_dlpack(ids, assumed_align=4),
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    executable = cute.compile(reducer, **kwargs)
    for _ in range(2):
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
            ("active", "mixed-invalid", "empty", "last-only", "restore")
        ):
            ids.copy_((base + iteration) % experts)
            if state == "mixed-invalid":
                ids.view(-1)[::3] = -1
                ids.view(-1)[1::3] = experts
            elif state == "empty":
                ids.fill_(torch.iinfo(torch.int32).min)
            elif state == "last-only":
                ids.fill_(torch.iinfo(torch.int32).max)
                ids[-1, -1] = experts - 1
            valid = (ids >= 0) & (ids < experts)
            values = (
                (
                    torch.arange(terms.numel(), device="cuda").reshape_as(terms)
                    + iteration
                )
                % 9
                - 4
            ).to(torch.bfloat16)
            terms.copy_(values).masked_fill_(~valid.unsqueeze(-1), float("nan"))
            scores.copy_(((base % 4).float() + 1) * 0.25)
            scores.masked_fill_(~valid, float("nan"))
            # Dyadic values make the ordered FP32 reference exactly representable.
            expected = torch.zeros((tokens, hidden), device="cuda", dtype=torch.float32)
            for k in range(topk):
                value = torch.where(valid[:, k, None], values[:, k].float(), 0.0)
                if weighted:
                    value *= torch.where(valid[:, k, None], scores[:, k, None], 0.0)
                expected += value
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.isfinite(output).all(), state
            torch.testing.assert_close(output, expected.bfloat16(), rtol=0, atol=0)
    finally:
        graph.reset()
