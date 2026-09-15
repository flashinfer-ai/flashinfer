# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from flashinfer.deepseek_v41 import deepseek_v41_index_scores_fp32


def gate():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")


def dequant(data, scales):
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=data.device,
        dtype=torch.float64,
    )
    codes = torch.stack((data & 15, data >> 4), -1).flatten(-2).long()
    return lut[codes] * torch.exp2(scales.double() - 127).repeat_interleave(32, -1)


def quantize(x, *, format="mxfp4", data=None, scales=None):
    # Test-only input generator. The oracle independently decodes the bytes;
    # production quantization is deliberately outside this PR's scope.
    assert format == "mxfp4"
    x = x.float().reshape(-1, 4, 32)
    exponent = torch.ceil(torch.log2(x.abs().amax(-1).clamp_min(1e-8) / 6))
    normalized = x * torch.exp2(-exponent[..., None])
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], device=x.device)
    codes = torch.bucketize(normalized.abs().contiguous(), thresholds).to(torch.uint8)
    codes |= (normalized < 0).to(torch.uint8) * 8
    codes = codes.flatten(1)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    sf = (exponent + 127).to(torch.uint8)
    if data is not None:
        data.copy_(packed)
        scales.copy_(sf)
        return data, scales
    return packed, sf


def make_cache(x, *, page_size=64, out=None):
    kd, ks = quantize(x)
    pages = (x.shape[0] + page_size - 1) // page_size
    stride = (page_size * 68 + 511) // 512 * 512
    if out is None:
        storage = torch.zeros((pages, stride), device=x.device, dtype=torch.uint8)
        out = storage.as_strided((pages, page_size, 1, 68), (stride, 68, 68, 1))
    flat = out.flatten(1)
    flat[:, : page_size * 64].copy_(kd.reshape(pages, -1))
    flat[:, page_size * 64 : page_size * 68].copy_(ks.reshape(pages, -1))
    return out


@pytest.mark.parametrize(
    "batch,context,page,candidate_count",
    [
        (1, 65, 32, None),
        (2, 2049, 64, None),
        (4, 32769, 128, None),
        (2, 32769, 64, 2048),
        (3, 127, 32, 17),
        (1, 1, 32, 1),
        (2, 257, 64, 3),
        (64, 8193, 64, 1025),
    ],
)
@pytest.mark.parametrize("backend", ["triton", "cute_dsl"])
def test_tiled_indexer_fp64_paging_candidates_padding_and_changed_graph(
    batch, context, page, candidate_count, backend
):
    gate()
    if backend == "cute_dsl" and candidate_count is None:
        pytest.skip("CuTe candidate backend; full scorer is covered separately")
    torch.manual_seed(42501 + batch + context + page)
    pages = (context + page - 1) // page
    q = torch.randn(batch * 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * pages * page, 128, device="cuda", dtype=torch.bfloat16)
    qd, qs = quantize(q, format="mxfp4")
    kd, ks = quantize(k, format="mxfp4")
    cache = make_cache(k, page_size=page)
    qd, qs = qd.view(batch, 32, 64), qs.view(batch, 32, 4)
    weights = (
        ((torch.arange(32, device="cuda") - 16) / 32).bfloat16()[None].repeat(batch, 1)
    )
    visible = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    if batch > 1:
        visible[0] = 0
    table = torch.randperm(batch * pages, device="cuda").int().view(batch, pages)
    candidates = None
    if candidate_count is not None:
        candidates = torch.randint(
            (context + 7) // 8 + 8,
            (batch, candidate_count),
            device="cuda",
            dtype=torch.int32,
        )
        candidates[:, -2:] = -1
    width = context if candidates is None else candidate_count * 8
    stride = (width + 511) // 512 * 512
    storage = torch.full((batch, stride), 17, device="cuda", dtype=torch.bfloat16)
    out = storage[:, :width]

    def run():
        return deepseek_v41_index_scores_fp32(
            qd,
            qs,
            cache,
            weights,
            visible,
            table,
            max_context_len=context,
            candidates=candidates,
            out=out,
            backend=backend,
        )

    def verify(exact):
        logical = torch.arange(width, device="cuda")[None].expand(batch, -1)
        if candidates is not None:
            logical = (
                candidates.long()[..., None] * 8 + torch.arange(8, device="cuda")
            ).flatten(1)
        valid = (logical >= 0) & (logical < visible[:, None]) & (logical < context)
        physical = (
            table.gather(1, logical.clamp(0, context - 1).long() // page) * page
            + logical.clamp(0, context - 1) % page
        )
        query = dequant(qd, qs)
        keys = dequant(kd, ks)[physical.long()]
        expected = (
            torch.einsum("bhd,btd->bht", query, keys).relu()
            * weights.double()[..., None]
        ).sum(1)
        assert torch.isneginf(out[~valid]).all()
        actual = out.double().masked_fill(~valid, 0)
        expected = expected.masked_fill(~valid, 0)
        relative = (actual - expected).norm() / expected.norm().clamp_min(1e-20)
        maximum = (actual - expected).abs().max() / expected.abs().max().clamp_min(
            1e-20
        )
        assert relative < 0.0021 and maximum < 0.0042, (float(relative), float(maximum))
        if exact:
            # Dyadic weights bound the full sum's significand: all reduction
            # schedules must agree before the required BF16 result boundary.
            torch.testing.assert_close(
                out[valid], expected.bfloat16()[valid], atol=0, rtol=0
            )
        torch.testing.assert_close(
            storage[:, width:], torch.full_like(storage[:, width:], 17), atol=0, rtol=0
        )

    run()
    verify(True)
    allocated = deepseek_v41_index_scores_fp32(
        qd,
        qs,
        cache,
        weights,
        visible,
        table,
        max_context_len=context,
        candidates=candidates,
        backend=backend,
    )
    torch.testing.assert_close(allocated, out, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for step in range(2):
        qd.bitwise_xor_(0x88)
        qs.add_(1)
        weights.copy_(torch.randn_like(weights))
        k.copy_(torch.randn_like(k) * 0.75)
        quantize(k, format="mxfp4", data=kd, scales=ks)
        make_cache(k, page_size=page, out=cache)
        table.copy_(table.flip(1))
        visible.copy_(
            torch.randint(context + 1, (batch,), device="cuda", dtype=torch.int32)
        )
        visible[-1] = context - step
        if candidates is not None:
            candidates.copy_(
                torch.randint(
                    (context + 7) // 8 + 8,
                    candidates.shape,
                    device="cuda",
                    dtype=torch.int32,
                )
            )
            candidates[:, 0] = -1
        graph.replay()
        verify(False)
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    alias = (
        cache.as_strided((batch * stride * 2,), (1,))
        .view(torch.bfloat16)
        .view(batch, stride)[:, :width]
    )
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_index_scores_fp32(
            qd,
            qs,
            cache,
            weights,
            visible,
            table,
            max_context_len=context,
            candidates=candidates,
            out=alias,
            backend=backend,
        )
    with pytest.raises(ValueError, match="inference"):
        deepseek_v41_index_scores_fp32(
            qd,
            qs,
            cache,
            weights.requires_grad_(),
            visible,
            table,
            max_context_len=context,
            candidates=candidates,
            out=out,
            backend=backend,
        )


@pytest.mark.parametrize("query_scale,key_scale", [(0, 254), (254, 0)])
@pytest.mark.parametrize("backend", ["triton", "cute_dsl"])
def test_native_mxfp4_extreme_e8m0_scales(query_scale, key_scale, backend):
    gate()
    # Both dequantized operands are finite BF16, including 2**-128.
    # Every channel product is exactly 1/4; 128 channels and 32 equally
    # weighted heads give a score of 32, in either scale orientation.
    qd = torch.full((1, 32, 64), 0x11, device="cuda", dtype=torch.uint8)
    qs = torch.full((1, 32, 4), query_scale, device="cuda", dtype=torch.uint8)
    storage = torch.zeros(2, 2560, device="cuda", dtype=torch.uint8)
    cache = storage.as_strided((2, 32, 1, 68), (2560, 68, 68, 1))
    storage[:, : 32 * 64].fill_(0x11)
    storage[:, 32 * 64 : 32 * 68].fill_(key_scale)
    weights = torch.full((1, 32), 1 / 32, device="cuda", dtype=torch.bfloat16)
    visible = torch.tensor([63], device="cuda", dtype=torch.int32)
    table = torch.tensor([[1, 0]], device="cuda", dtype=torch.int32)
    out = deepseek_v41_index_scores_fp32(
        qd,
        qs,
        cache,
        weights,
        visible,
        table,
        max_context_len=64,
        candidates=torch.arange(8, device="cuda", dtype=torch.int32)[None],
        backend=backend,
    )
    torch.testing.assert_close(
        out[:, :63], torch.full_like(out[:, :63], 32), atol=0, rtol=0
    )
    assert torch.isneginf(out[:, 63]).all()


@pytest.mark.parametrize("invalid_candidate", [2**29, -(2**31), 2**31 - 1])
@pytest.mark.parametrize("backend", ["triton", "cute_dsl"])
def test_candidate_token_offset_does_not_wrap(invalid_candidate, backend):
    gate()
    qd = torch.full((1, 32, 64), 0x11, device="cuda", dtype=torch.uint8)
    qs = torch.full((1, 32, 4), 127, device="cuda", dtype=torch.uint8)
    storage = torch.zeros(1, 2560, device="cuda", dtype=torch.uint8)
    cache = storage.as_strided((1, 32, 1, 68), (2560, 68, 68, 1))
    storage[:, : 32 * 64].fill_(0x11)
    storage[:, 32 * 64 : 32 * 68].fill_(127)
    weights = torch.full((1, 32), 1 / 32, device="cuda", dtype=torch.bfloat16)
    visible = torch.tensor([32], device="cuda", dtype=torch.int32)
    table = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    candidates = torch.tensor(
        [[invalid_candidate, 0]], device="cuda", dtype=torch.int32
    )
    out = deepseek_v41_index_scores_fp32(
        qd,
        qs,
        cache,
        weights,
        visible,
        table,
        max_context_len=32,
        candidates=candidates,
        backend=backend,
    )
    assert torch.isneginf(out[:, :8]).all()
    torch.testing.assert_close(
        out[:, 8:], torch.full_like(out[:, 8:], 32), atol=0, rtol=0
    )


def _constant_case(batch=4, count=17):
    gate()
    qd = torch.full((batch, 32, 64), 0x11, device="cuda", dtype=torch.uint8)
    qs = torch.full((batch, 32, 4), 127, device="cuda", dtype=torch.uint8)
    storage = torch.zeros((1, 2560), device="cuda", dtype=torch.uint8)
    cache = storage.as_strided((1, 32, 1, 68), (2560, 68, 68, 1))
    storage[:, :2048].fill_(0x11)
    storage[:, 2048:2176].fill_(127)
    w = torch.full((batch, 32), 1 / 32, device="cuda", dtype=torch.bfloat16)
    visible = torch.full((batch,), 31, device="cuda", dtype=torch.int32)
    table = torch.zeros((batch, 1), device="cuda", dtype=torch.int32)
    c = (torch.arange(count, device="cuda", dtype=torch.int32) % 4)[None].repeat(
        batch, 1
    )
    return [qd, qs, cache, w, visible, table], c


def test_cute_workspace_and_alignment_contracts():
    from flashinfer.experimental.deepseek_v41.indexer_cute import workspace_size

    inputs, candidates = _constant_case()
    size = sum(workspace_size(4, 17))
    workspace = torch.empty(size, device="cuda", dtype=torch.uint8)
    kwargs = dict(max_context_len=32, candidates=candidates, backend="cute_dsl")
    for invalid in (workspace[:-1], workspace.view(1, -1), workspace.to(torch.int32)):
        with pytest.raises(ValueError, match="workspace"):
            deepseek_v41_index_scores_fp32(*inputs, **kwargs, workspace=invalid)
    # A declared contiguous tensor can have an unaligned storage offset.
    shifted = torch.empty(inputs[0].numel() + 1, device="cuda", dtype=torch.uint8)[
        1:
    ].view_as(inputs[0])
    with pytest.raises(ValueError, match="16-byte-aligned"):
        deepseek_v41_index_scores_fp32(shifted, *inputs[1:], **kwargs)
    alias = inputs[0].flatten()[:size]
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_index_scores_fp32(*inputs, **kwargs, workspace=alias)
    with pytest.raises(ValueError, match="candidate scoring only"):
        deepseek_v41_index_scores_fp32(*inputs, max_context_len=32, backend="cute_dsl")
    with pytest.raises(ValueError, match="backend"):
        deepseek_v41_index_scores_fp32(*inputs, max_context_len=32, backend="unknown")


def test_cute_two_graphs_have_independent_workspace():
    from flashinfer.experimental.deepseek_v41.indexer_cute import workspace_size

    calls = []
    for _ in range(2):
        inputs, candidates = _constant_case()
        workspace = torch.empty(
            sum(workspace_size(4, 17)), device="cuda", dtype=torch.uint8
        )
        out = torch.empty_strided(
            (4, 136), (512, 1), device="cuda", dtype=torch.bfloat16
        )

        def run(inputs=inputs, candidates=candidates, workspace=workspace, out=out):
            return deepseek_v41_index_scores_fp32(
                *inputs,
                max_context_len=32,
                candidates=candidates,
                out=out,
                backend="cute_dsl",
                workspace=workspace,
            )

        run()
        stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            run()
        calls.append((inputs, candidates, workspace, out, stream, graph))
    for step in range(3):
        for i, (inputs, _c, workspace, out, stream, graph) in enumerate(calls):
            inputs[1].fill_(126 + i + step)
            workspace.fill_(0xA5)
            out.fill_(float("nan"))
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
        for i, (_, c, _, out, stream, _) in enumerate(calls):
            stream.synchronize()
            valid = (c[:, :, None].long() * 8 + torch.arange(8, device="cuda")).flatten(
                1
            ) < 31
            assert torch.isneginf(out[~valid]).all()
            torch.testing.assert_close(
                out[valid],
                torch.full_like(out[valid], 16 * 2 ** (i + step)),
                atol=0,
                rtol=0,
            )


@pytest.mark.parametrize("backend", ["triton", "cute_dsl"])
def test_kv_stride_above_32_gib(backend):
    inputs, candidates = _constant_case(batch=1, count=4)
    stride = 40 * 1024**3
    free, _ = torch.cuda.mem_get_info()
    if free < stride + 2 * 1024**3:
        pytest.skip("requires40GiB free for64-bit KV address coverage")
    cache = torch.empty_strided(
        (2, 32, 1, 68), (stride, 68, 68, 1), device="cuda", dtype=torch.uint8
    )
    # Only the second physical page is initialized or live.
    flat = cache.flatten(1)
    flat[1, :2048].fill_(0x11)
    flat[1, 2048:].fill_(127)
    inputs[2] = cache
    inputs[-1].fill_(1)
    out = deepseek_v41_index_scores_fp32(
        *inputs, max_context_len=32, candidates=candidates, backend=backend
    )
    torch.testing.assert_close(
        out[:, :31], torch.full_like(out[:, :31], 32), atol=0, rtol=0
    )
    assert torch.isneginf(out[:, 31]).all()


def test_cute_rejects_int32_padded_output_extent():
    inputs, candidates = _constant_case(batch=1, count=4)
    # A singleton row can declare this stride with only64 bytes allocated.
    # The kernel's flattened view still multiplies batch*row_stride in int32.
    out = torch.empty_strided((1, 32), (2**31, 1), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="padded output"):
        deepseek_v41_index_scores_fp32(
            *inputs,
            max_context_len=32,
            candidates=candidates,
            out=out,
            backend="cute_dsl",
        )


@pytest.mark.parametrize("backend", ["triton", "cute_dsl"])
@pytest.mark.parametrize("invalid_page", [-1, 1, 2**31 - 1, -(2**31)])
def test_invalid_physical_pages_are_masked(backend, invalid_page):
    inputs, candidates = _constant_case(batch=1, count=4)
    inputs[-1].fill_(invalid_page)
    out = deepseek_v41_index_scores_fp32(
        *inputs, max_context_len=32, candidates=candidates, backend=backend
    )
    assert torch.isneginf(out).all()
