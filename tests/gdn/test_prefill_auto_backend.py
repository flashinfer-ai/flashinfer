# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Keep default GDN prefill calls on the established FlashInfer kernels."""

import json

import pytest
import torch

from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.utils import is_sm100a_supported


@pytest.mark.parametrize(
    "batch_size,seq_len,num_q_heads,num_v_heads",
    [(4, 2048, 16, 32), (1, 64, 4, 8)],
)
def test_prefill_default_backend_preserves_flashinfer_graph(
    batch_size, seq_len, num_q_heads, num_v_heads, tmp_path
):
    if not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires Blackwell")
    if int(torch.version.cuda.split(".")[0]) < 13:
        pytest.skip("Blackwell GDN prefill requires CUDA 13+")

    generator = torch.Generator(device="cuda").manual_seed(42)
    total = batch_size * seq_len
    q = torch.randn(
        (total, num_q_heads, 128), generator=generator, device="cuda"
    ).to(torch.bfloat16)
    k = torch.randn(q.shape, generator=generator, device="cuda").to(q.dtype)
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    v = torch.randn(
        (total, num_v_heads, 128), generator=generator, device="cuda"
    ).to(q.dtype)
    g = torch.rand((total, num_v_heads), generator=generator, device="cuda")
    beta = torch.rand(g.shape, generator=generator, device="cuda")
    cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int64, device="cuda")
    cu_seqlens *= seq_len
    results = []
    for backend in ("flashinfer", "auto", None):
        output = torch.empty_like(v)
        state = torch.empty(
            (batch_size, num_v_heads, 128, 128), dtype=torch.float32, device="cuda"
        )

        def launch():
            return chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                cu_seqlens=cu_seqlens,
                output_final_state=True,
                output=output,
                output_state=state,
                **({"backend": backend} if backend is not None else {}),
            )

        for _ in range(3):
            launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            graph.replay()
            torch.cuda.synchronize()
        trace = tmp_path / f"{backend}.json"
        profile.export_chrome_trace(str(trace))
        kernels = [
            event["name"]
            for event in json.loads(trace.read_text())["traceEvents"]
            if event.get("cat") == "kernel"
        ]
        assert kernels, "CUDA profiling must capture the prefill kernel launches"
        results.append((output, state, kernels))

    expected_output, expected_state, expected_kernels = results[0]
    for output, state, kernels in results[1:]:
        # Numerical checks alone miss a slower, numerically valid backend.
        assert kernels == expected_kernels
        torch.testing.assert_close(output, expected_output, atol=0, rtol=0)
        torch.testing.assert_close(state, expected_state, atol=0, rtol=0)
