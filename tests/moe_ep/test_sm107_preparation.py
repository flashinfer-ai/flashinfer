"""Staging parity and preparation helpers under CUDA graph replay."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.arch_rubin,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
        reason="Requires SM107",
    ),
]


def _buffers(capacity, hidden, topk, dtype):
    return (
        torch.empty(capacity, hidden // 2, device="cuda", dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        ),
        torch.empty(capacity, hidden // 16, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(capacity, topk, device="cuda", dtype=dtype),
        torch.empty(capacity, topk, device="cuda"),
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("norm", [1.0, 0.3, 0.7, 2.0])
def test_nvfp4_fused_staging_matches_torch(monkeypatch, dtype, norm):
    from flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
        unpack_fp4_to_f32,
    )

    hidden, capacity, topk = 512, 64, 8
    actual = _buffers(capacity, hidden, topk, dtype)
    expected = _buffers(capacity, hidden, topk, dtype)
    for live in (64, 8, 0, 32):
        x = torch.randn(live, hidden, device="cuda", dtype=torch.bfloat16)
        if live:
            x[0, :64] = 0
            x[0, 64:128] = 1e-6
            x[0, 128:192] = 1000
        ids = torch.arange(topk, device="cuda", dtype=dtype).repeat(live, 1)
        scores = torch.rand(live, topk, device="cuda")
        for enabled, out in (("0", expected), ("1", actual)):
            monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", enabled)
            stage_mega_moe_inputs(x, scores, ids, *out, input_norm_const=norm)
        torch.cuda.synchronize()
        if live:
            torch.testing.assert_close(
                unpack_fp4_to_f32(actual[0][:live]),
                unpack_fp4_to_f32(expected[0][:live]),
                rtol=0,
                atol=0,
            )
        assert torch.equal(
            actual[1][:live].view(torch.uint8), expected[1][:live].view(torch.uint8)
        )
        assert torch.equal(actual[2], expected[2])
        assert torch.equal(actual[3][:live], expected[3][:live])


@pytest.mark.parametrize("layout", ["contiguous", "unaligned", "strided"])
def test_staging_graph_restores_shared_workspace_tail(monkeypatch, layout):
    from flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )

    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "1")
    live, capacity, hidden, topk = 8, 64, 512, 8
    x = torch.randn(live, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(live * topk + 1, device="cuda", dtype=torch.int32) % 128
    ids = (
        ids[1:].reshape(live, topk)
        if layout == "unaligned"
        else ids[:-1].reshape(topk, live).T
        if layout == "strided"
        else ids[:-1].reshape(live, topk)
    )
    scores = torch.rand(live, topk, device="cuda")
    outputs = _buffers(capacity, hidden, topk, torch.int32)

    def stage():
        stage_mega_moe_inputs(x, scores, ids, *outputs)

    stage()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stage()
    for shift in (1, 7):
        ids.add_(shift)
        outputs[2].fill_(5)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(outputs[2][:live], ids)
        assert (outputs[2][live:] == -1).all()
    graph.reset()


@pytest.mark.parametrize("ffi", ["0", "1"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(0, 8), (1, 1), (8, 8), (512, 8), (1024, 32)])
def test_routing_check_rebinds_inputs_and_replays(monkeypatch, ffi, dtype, shape):
    from flashinfer.moe_ep.backends.mega.kernel.sm107.validation import (
        validate_routing_values,
    )

    monkeypatch.setenv("FLASHINFER_SM107_PREP_TVM_FFI", ffi)
    monkeypatch.setenv("FLASHINFER_SM107_FUSED_ROUTE_CHECK", "1")
    tokens, topk = shape
    ids = torch.arange(topk, device="cuda", dtype=dtype).repeat(tokens, 1)
    scores = torch.ones(tokens, topk, device="cuda")
    ws = SimpleNamespace()
    validate_routing_values(ids, scores, 128, workspace=ws)
    torch.cuda.synchronize()

    # Observe invalid replay inputs without poisoning the worker's CUDA context.
    verdict = torch.empty((), dtype=torch.bool, device="cuda")

    def observe_check(valid, message):
        verdict.copy_(valid.reshape(()))

    monkeypatch.setattr(torch, "_assert_async", observe_check)
    original_ids, original_scores = ids, scores
    ids, scores = ids.clone(), scores.clone()
    original_ids.fill_(128)
    original_scores.fill_(float("nan"))
    validate_routing_values(ids, scores, 128, workspace=ws)
    torch.cuda.synchronize()
    assert verdict.item()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        validate_routing_values(ids, scores, 128, workspace=ws)
    ids.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert verdict.item()
    if tokens:
        ids[-1, -1] = 128
        graph.replay()
        torch.cuda.synchronize()
        assert not verdict.item()
        ids[-1, -1] = -1
        scores[-1, -1] = float("nan")
        graph.replay()
        torch.cuda.synchronize()
        assert not verdict.item()
        scores.fill_(1)
        graph.replay()
        torch.cuda.synchronize()
        assert verdict.item()
    graph.reset()


@pytest.mark.parametrize("ffi", ["0", "1"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("score_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("topk", [1, 8, 32])
def test_routing_check_reuses_kernel_across_row_counts(
    monkeypatch, ffi, dtype, score_dtype, topk
):
    from flashinfer.moe_ep.backends.mega.kernel.sm107 import routing_validation

    monkeypatch.setenv("FLASHINFER_SM107_PREP_TVM_FFI", ffi)
    monkeypatch.setattr(routing_validation, "_COMPILED", {})
    ws = SimpleNamespace()
    ids = torch.arange(topk, device="cuda", dtype=dtype).repeat(8, 1)
    scores = torch.ones(8, topk, device="cuda", dtype=score_dtype)
    routing_validation.check_routing(ids, scores, 128, workspace=ws)
    torch.cuda.synchronize()

    def unexpected_compile(*args, **kwargs):
        pytest.fail("A new row count must reuse the warmed routing kernel")

    monkeypatch.setattr(routing_validation.cute, "compile", unexpected_compile)
    graphs = []
    inputs = []
    try:
        for tokens in (0, 1, 129, 513, 1024):
            ids = torch.arange(topk, device="cuda", dtype=dtype).repeat(tokens, 1)
            scores = torch.ones(tokens, topk, device="cuda", dtype=score_dtype)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                routing_validation.check_routing(ids, scores, 128, workspace=ws)
            graphs.append(graph)
            inputs.append((ids, scores))
        for graph in reversed(graphs):
            graph.replay()
            torch.cuda.synchronize()
    finally:
        for graph in graphs:
            graph.reset()
