# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared output joins the routed FP32 sum before its final BF16 rounding."""

import pytest
import torch


@pytest.mark.arch_rubin
@pytest.mark.parametrize("live", [1, 17, 33])
@pytest.mark.parametrize("hidden", [256, 520])
@pytest.mark.parametrize("weighted", [False, True])
def test_shared_reduce_rounding_and_live_output(live, hidden, weighted):
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

    capacity, topk = 64, 6
    terms = torch.zeros((capacity, topk, hidden), device="cuda", dtype=torch.bfloat16)
    shared = torch.empty((capacity, 1, hidden), device="cuda", dtype=torch.bfloat16)
    output = torch.full((capacity, hidden), 1024, device="cuda", dtype=torch.bfloat16)
    ids = torch.full((capacity, topk), -1, device="cuda", dtype=torch.int32)
    scores = torch.ones((capacity, topk), device="cuda")
    reducer = TopkReduce(hidden, topk, CombineFormat.parse("bf16"), num_experts=128)
    kwargs = dict(
        combine_quant=from_dlpack(terms[:live], assumed_align=16),
        combine_sf=None,
        reduced_output=from_dlpack(output[:live], assumed_align=16),
        topk_score=from_dlpack(scores[:live], assumed_align=4) if weighted else None,
        topk_indices=from_dlpack(ids[:live], assumed_align=4),
        shared_output=from_dlpack(shared[:live], assumed_align=16),
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    executable = cute.compile(reducer, **kwargs)
    executable(**kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            kwargs["stream"] = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            executable(**kwargs)
        for active in (True, False, True):
            ids.fill_(-1)
            terms.fill_(float("nan"))
            scores.fill_(float("inf"))
            shared.fill_(-256 if active else 3)
            if active:
                ids[:, :2] = 0
                terms[:, 0] = 256
                terms[:, 1] = 0.5
                scores[:, :2] = 1
            output.fill_(1024)
            graph.replay()
            torch.cuda.synchronize()
            # Rounding the routed sum to BF16 before adding -256 would give zero.
            expected = torch.full_like(output[:live], 0.5 if active else 3)
            torch.testing.assert_close(output[:live], expected, rtol=0, atol=0)
            assert (output[live:] == 1024).all()
    finally:
        graph.reset()
