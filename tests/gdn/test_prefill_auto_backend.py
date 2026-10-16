# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The default GDN prefill backend is ``auto``: it matches an explicit
``auto`` call kernel for kernel, follows the cuDNN policy region on SM100 and
keeps the FlashInfer kernels everywhere else."""

import json

import pytest
import torch

from flashinfer import gdn_prefill
from flashinfer.cudnn import linear_attention
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.utils import is_sm100a_supported


@pytest.fixture
def native_auto_policy():
    """This module exercises the auto resolution itself, so the native-only
    pin of the directory's conftest does not apply here."""
    yield


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
    q = torch.randn((total, num_q_heads, 128), generator=generator, device="cuda").to(
        torch.bfloat16
    )
    k = torch.randn(q.shape, generator=generator, device="cuda").to(q.dtype)
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    v = torch.randn((total, num_v_heads, 128), generator=generator, device="cuda").to(
        q.dtype
    )
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

    native_output, native_state, native_kernels = results[0]
    auto_output, auto_state, auto_kernels = results[1]
    default_output, default_state, default_kernels = results[2]
    # The default backend is "auto": the same kernels and the same bits.
    assert default_kernels == auto_kernels
    torch.testing.assert_close(default_output, auto_output, atol=0, rtol=0)
    torch.testing.assert_close(default_state, auto_state, atol=0, rtol=0)
    # Numerical checks alone miss a slower, numerically valid backend, so the
    # kernel list is the routing check in both directions.
    prefer_cudnn = (
        linear_attention._linear_attention_auto_available()
        and gdn_prefill._prefer_cudnn_gdn_prefill(q, v, cu_seqlens, False)
    )
    if prefer_cudnn:
        assert auto_kernels != native_kernels

        def rel_err(actual, expected):
            diff = (actual.float() - expected.float()).norm()
            return (diff / expected.float().norm().clamp_min(1e-6)).item()

        assert rel_err(auto_output, native_output) < 5e-3
        assert rel_err(auto_state, native_state) < 5e-3
    else:
        assert auto_kernels == native_kernels
        torch.testing.assert_close(auto_output, native_output, atol=0, rtol=0)
        torch.testing.assert_close(auto_state, native_state, atol=0, rtol=0)
