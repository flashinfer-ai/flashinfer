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
@pytest.mark.parametrize(
    "shape",
    [
        (0, 8),
        (1, 1),
        (8, 8),
        (512, 8),
        (1024, 32),
        (1025, 8),
        (2048, 8),
        (4609, 32),
        (16385, 6),
    ],
)
@pytest.mark.parametrize("fused", ["0", "1"])
def test_routing_check_rebinds_inputs_and_replays(
    monkeypatch, ffi, dtype, shape, fused
):
    from flashinfer.moe_ep.backends.mega.kernel.sm107.validation import (
        validate_routing_values,
    )

    monkeypatch.setenv("FLASHINFER_SM107_PREP_TVM_FFI", ffi)
    monkeypatch.setenv("FLASHINFER_SM107_FUSED_ROUTE_CHECK", fused)
    tokens, topk = shape
    ids = torch.arange(topk, device="cuda", dtype=dtype).repeat(tokens, 1)
    scores = torch.ones(tokens, topk, device="cuda")
    ws = SimpleNamespace()
    validate_routing_values(ids, scores, 128, workspace=ws)
    torch.cuda.synchronize()
    if fused == "1":
        assert ws._routing_checks

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
        invalid_ids = [-2, 128]
        if dtype == torch.int64:
            invalid_ids += [-(2**32), 2**32, -(2**63), 2**63 - 1]
        for row in sorted({0, min(1024, tokens - 1), tokens - 1}):
            for expert in invalid_ids:
                ids[row, -1] = expert
                graph.replay()
                torch.cuda.synchronize()
                assert not verdict.item()
            ids[row, -1] = -1
            if topk > 1:
                ids[row, :2] = 7
                graph.replay()
                torch.cuda.synchronize()
                assert not verdict.item()
                ids[row, :2] = -1
            for score in (float("nan"), float("inf"), -float("inf")):
                scores[row, -1] = score
                graph.replay()
                torch.cuda.synchronize()
                assert verdict.item()
                ids[row, -1] = 0
                graph.replay()
                torch.cuda.synchronize()
                assert not verdict.item()
                ids[row, -1] = -1
            scores[row, -1] = 1
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
        for tokens in (0, 1, 129, 513, 1024, 1025, 2048, 4609, 65537):
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


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"])
@pytest.mark.parametrize("prequantized", [False, True])
@pytest.mark.parametrize("fused", ["0", "1"])
@pytest.mark.parametrize("score_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_staging_zeroes_masked_scores(
    monkeypatch, kind, prequantized, fused, score_dtype
):
    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from tests.moe_ep.test_moe_ep_sm107_block_scaled_mega_multirank import (
        _megakernel_config,
    )
    from tests.moe_ep.sm107_test_utils import quantize

    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", fused)
    live, capacity, hidden, topk = 8, 16, 512, 4
    x = torch.randn(live, hidden, device="cuda", dtype=torch.bfloat16)
    xq, sf = quantize(x, kind)
    ids = torch.arange(topk, device="cuda", dtype=torch.int64).repeat(live, 1)
    scores = torch.rand(live, topk, device="cuda", dtype=score_dtype)
    ws = SimpleNamespace(
        config=SimpleNamespace(num_total_experts=8),
        x=torch.empty((capacity, xq.shape[1]), device="cuda", dtype=xq.dtype),
        x_sf=torch.empty((capacity, sf.shape[1]), device="cuda", dtype=sf.dtype),
        topk_idx=torch.empty(capacity, topk, device="cuda", dtype=torch.int32),
        topk_weights=torch.empty(capacity, topk, device="cuda"),
        fc1_alpha=torch.ones(8, device="cuda"),
        fc2_alpha=torch.ones(8, device="cuda"),
        fc1_norm_const=torch.ones(8, device="cuda"),
        note_staged_tokens=lambda n: None,
    )
    kernel = create_mega_kernel(_megakernel_config(kind))
    tensors = MoEEpTensors(
        xq if prequantized else x, ids, scores, scales=sf if prequantized else None
    )

    def stage():
        kernel.stage_inputs(tensors, ws, quantize_input=not prequantized)

    stage()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stage()
    for column, value in enumerate((float("nan"), float("inf"), -float("inf"))):
        ids.copy_(torch.arange(topk, device="cuda").repeat(live, 1))
        scores.fill_(0.25)
        ids[:, column] = -1
        ids[0] = -1
        scores[:, column] = value
        scores[0] = value
        expected = scores.float().masked_fill(ids == -1, 0)
        for launch in (stage, graph.replay):
            ws.topk_weights.fill_(123)
            launch()
            torch.cuda.synchronize()
            torch.testing.assert_close(ws.topk_weights[:live], expected, rtol=0, atol=0)
            assert torch.equal(ws.topk_idx[:live], ids.to(torch.int32))
            assert (ws.topk_idx[live:] == -1).all()
            assert (
                scores[ids == -1].isnan() if column == 0 else scores[ids == -1] == value
            ).all()
    graph.reset()


@pytest.mark.parametrize("ffi", ["0", "1"])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("score_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "unaligned"])
def test_route_staging_rebinds_and_replays(
    monkeypatch, ffi, id_dtype, score_dtype, layout
):
    from flashinfer.moe_ep.backends.mega.kernel.sm107 import routing_staging

    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "1")
    monkeypatch.setenv("FLASHINFER_SM107_PREP_TVM_FFI", ffi)
    monkeypatch.setattr(routing_staging, "_COMPILED", {})
    topk, capacity = 6, 1032
    workspace = SimpleNamespace()
    ids_out = torch.full((capacity, topk), -7, device="cuda", dtype=torch.int32)
    scores_out = torch.full((capacity, topk), -7.0, device="cuda")

    def inputs(live):
        stride = 2 if layout == "strided" else 1
        offset = 1 if layout == "unaligned" else 0
        ids = torch.empty(live * topk * stride + offset, device="cuda", dtype=id_dtype)
        scores = torch.empty(ids.numel(), device="cuda", dtype=score_dtype)
        return tuple(t[offset::stride].reshape(live, topk) for t in (ids, scores))

    ids, scores = inputs(8)
    ids.zero_()
    scores.fill_(1)
    routing_staging.stage_routes(ids, scores, ids_out, scores_out, workspace=workspace)
    torch.cuda.synchronize()
    assert len(routing_staging._COMPILED) == 1

    def unexpected_compile(*args, **kwargs):
        pytest.fail("A new row count must reuse the warmed route-staging kernel")

    import cutlass.cute as cute

    monkeypatch.setattr(cute, "compile", unexpected_compile)
    captures = []
    try:
        for live in (1, 129, 1025, 0):
            ids, scores = inputs(live)
            ids.copy_(torch.arange(topk, device="cuda").repeat(live, 1))
            scores.fill_(0.5)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                routing_staging.stage_routes(
                    ids, scores, ids_out, scores_out, workspace=workspace
                )
            captures.append((graph, ids, scores))
        for graph, ids, scores in reversed(captures):
            live = ids.shape[0]
            for value in (float("nan"), float("inf"), -float("inf")):
                ids[:, 1] = -1
                scores[:, 1] = value
                ids_out.fill_(-7)
                scores_out.fill_(-7)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(ids_out[:live], ids.to(torch.int32))
                torch.testing.assert_close(
                    scores_out[:live],
                    scores.float().masked_fill(ids == -1, 0),
                    rtol=0,
                    atol=0,
                )
                assert (ids_out[live:] == -7).all()
                assert (scores_out[live:] == -7).all()
                assert (
                    scores[:, 1].isnan() if value != value else scores[:, 1] == value
                ).all()
    finally:
        for graph, _, _ in captures:
            graph.reset()


@pytest.mark.parametrize("kind", ["mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"])
@pytest.mark.parametrize("hidden,topk", [(512, 6), (2880, 6), (7168, 6), (512, 129)])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_mxfp8_fused_staging_matches_torch(monkeypatch, kind, hidden, topk, id_dtype):
    from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp8_bf16_cutedsl.staging import (
        _data_dtype,
        stage_mega_moe_inputs,
    )

    capacity = 64

    def buffers():
        return (
            torch.empty(capacity, hidden, device="cuda", dtype=_data_dtype(kind)),
            torch.empty(
                capacity,
                (hidden // 32 + 3) // 4 * 4,
                device="cuda",
                dtype=torch.float8_e8m0fnu,
            ),
            torch.empty(capacity, topk, device="cuda", dtype=torch.int32),
            torch.empty(capacity, topk, device="cuda"),
        )

    actual, expected = buffers(), buffers()
    for live in (64, 8, 0, 32):
        x = torch.randn(live, hidden, device="cuda", dtype=torch.bfloat16)
        if live:
            x[0, :32] = 0
            x[0, 32:64] = 1e-38
            x[0, 64:96] = 1e-6
            x[0, 96:128] = 1000
        ids = torch.arange(topk, device="cuda", dtype=id_dtype).repeat(live, 1)
        scores = torch.rand(live, topk, device="cuda")
        ids[:, 0] = -1
        scores[:, 0] = float("nan")
        for enabled, out in (("0", expected), ("1", actual)):
            monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", enabled)
            stage_mega_moe_inputs(x, scores, ids, *out, kind=kind)
        torch.cuda.synchronize()
        for index in (0, 1):
            torch.testing.assert_close(
                actual[index][:live].view(torch.uint8),
                expected[index][:live].view(torch.uint8),
                rtol=0,
                atol=0,
            )
        assert torch.equal(actual[2], expected[2])
        assert torch.equal(actual[3][:live], expected[3][:live])


@pytest.mark.parametrize("kind", ["mxfp8_e4m3", "mxfp8_e5m2"])
def test_mxfp8_fused_staging_finite_bf16(monkeypatch, kind):
    from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp8_bf16_cutedsl.staging import (
        _data_dtype,
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
        quantize_mxfp8_block32,
    )

    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "1")
    # Cover every finite BF16 value, including signed zero and subnormals.
    x = torch.arange(65536, device="cuda").to(torch.int16).view(torch.bfloat16)
    x = x.nan_to_num(nan=0, posinf=0, neginf=0).reshape(128, 512)
    ids = torch.arange(8, device="cuda", dtype=torch.int32).repeat(128, 1)
    scores = torch.full((128, 8), 0.125, device="cuda")
    q = torch.empty_like(x, dtype=_data_dtype(kind))
    sf = torch.empty(128, 16, device="cuda", dtype=torch.float8_e8m0fnu)
    stage_mega_moe_inputs(
        x,
        scores,
        ids,
        q,
        sf,
        torch.empty_like(ids),
        torch.empty_like(scores),
        kind=kind,
    )
    expected_q, expected_sf = quantize_mxfp8_block32(x.float(), _data_dtype(kind))
    torch.testing.assert_close(
        q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        sf.view(torch.uint8), expected_sf.view(torch.uint8), rtol=0, atol=0
    )


@pytest.mark.parametrize("kind", ["mxfp8_e4m3", "mxfp8_e5m2"])
def test_mxfp8_fused_staging_keeps_scale_policies_separate(kind):
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import fused_quant_stage

    dtype = torch.float8_e4m3fn if kind == "mxfp8_e4m3" else torch.float8_e5m2
    x = torch.zeros(8, 512, device="cuda", dtype=torch.bfloat16)
    x[:, 32:64] = 1e-38
    ids = torch.arange(8, device="cuda", dtype=torch.int32).repeat(8, 1)
    scores = torch.full((8, 8), 0.125, device="cuda")
    q = torch.empty_like(x, dtype=dtype)
    sf = torch.empty(8, 16, device="cuda", dtype=torch.float8_e8m0fnu)
    ids_out, scores_out = torch.empty_like(ids), torch.empty_like(scores)
    fused_quant_stage(
        x, ids, scores, q, sf, ids_out.to(torch.int64), scores_out, quant_type=kind
    )
    legacy_q = q.clone()
    for normal in (False, True, False, True):
        fused_quant_stage(
            x,
            ids,
            scores,
            q,
            sf,
            ids_out,
            scores_out,
            quant_type=kind,
            mxfp8_normal_scales=normal,
        )
        torch.cuda.synchronize()
        expected = (x.float() / (2.0**-126)).to(dtype) if normal else legacy_q
        assert torch.equal(q.view(torch.uint8), expected.view(torch.uint8))
        assert (sf.view(torch.uint8) == int(normal)).all()
        assert torch.equal(ids_out, ids) and torch.equal(scores_out, scores)
