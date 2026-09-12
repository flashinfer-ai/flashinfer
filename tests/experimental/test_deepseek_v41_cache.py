# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Exact encoded-byte contract for the optional decode cache writer."""

import pytest
import torch

from flashinfer.deepseek_v41 import deepseek_v41_quantize_cache


def gate():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("DS4.1 decode cache preparation requires SM100")


def reference(x, fmt):
    group = 16 if fmt == "main_kv_fp4" else 32
    g = x.float().view(x.shape[0], -1, group)
    amax = g.abs().amax(-1)
    if fmt == "main_kv_fp4":
        sf = (amax.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn)
    else:
        bound, minimum = 448, 1e-4
        sf = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(minimum) / bound))).to(
            torch.float8_e8m0fnu
        )
    v = (g / sf.float().unsqueeze(-1)).view_as(x)
    if fmt == "swa_mxfp8":
        data = v.to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
        distances = (v.abs().clamp_max(6)[..., None] - levels).abs()
        # Resolve exact halfway cases to an even code independently of PTX.
        minimum = distances.amin(-1, keepdim=True)
        candidates = distances == minimum
        codes = torch.arange(8, device=x.device)
        priority = codes + (codes % 2) * 8
        rank = torch.where(candidates, priority, 100).argmin(-1).to(torch.uint8)
        rank |= torch.signbit(v).to(torch.uint8) * 8
        data = rank[:, 0::2] | (rank[:, 1::2] << 4)
    return data.contiguous(), sf.view(torch.uint8).contiguous()


def unpack_bytes(cache, rows, fp4):
    width, sf = (256, 32) if fp4 else (512, 16)
    pages = cache.view(cache.shape[0], -1)
    return (
        pages[:, : 64 * width].reshape(-1, width)[:rows],
        pages[:, 64 * width :].reshape(-1, sf)[:rows],
    )


@pytest.mark.parametrize("fmt", ["main_kv_fp4", "swa_mxfp8"])
@pytest.mark.parametrize("rows", [1, 19, 257])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_exact_cache_bytes_and_changed_graph(fmt, rows, dtype):
    gate()
    torch.manual_seed(841 + rows)
    x = torch.randn(rows, 512, device="cuda", dtype=dtype)
    x[0, :32] = 0
    out = deepseek_v41_quantize_cache(x, format=fmt)

    def check():
        for actual, expected in zip(
            unpack_bytes(out, rows, fmt == "main_kv_fp4"),
            reference(x, fmt),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        deepseek_v41_quantize_cache(x, format=fmt, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        x.copy_(torch.randn_like(x))
        out.fill_(197)
        graph.replay()
        check()
    out.view(-1)[0].bitwise_xor_(1)
    with pytest.raises(AssertionError):
        check()


@pytest.mark.parametrize(
    "fmt,width,sf", [("main_kv_fp4", 256, 32), ("swa_mxfp8", 512, 16)]
)
def test_scatter_changed_slots_and_untouched_bytes(fmt, width, sf):
    gate()
    x = torch.randn(5, 512, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([67, 1, -1, 63, 64], device="cuda", dtype=torch.int32)
    out = torch.full((2, 64, 1, width + sf), 173, device="cuda", dtype=torch.uint8)
    expected = torch.empty_like(out).view(2, -1)
    deepseek_v41_quantize_cache(x, format=fmt, out=out, slots=slots)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_quantize_cache(x, format=fmt, out=out, slots=slots)
    for step in range(3):
        x.copy_(torch.randn_like(x))
        slots.copy_(torch.randperm(128, device="cuda")[:5].int())
        slots[step] = -1
        expected.fill_(173)
        out.fill_(173)
        graph.replay()
        data, scales = reference(x, fmt)
        for row, slot in enumerate(slots.tolist()):
            if slot < 0:
                continue
            page, local = divmod(slot, 64)
            expected[page, local * width : (local + 1) * width] = data[row]
            expected[page, 64 * width + local * sf : 64 * width + (local + 1) * sf] = (
                scales[row]
            )
        torch.testing.assert_close(out.flatten(), expected.flatten(), rtol=0, atol=0)


def test_fp4_halfway_even_and_signed_zero():
    gate()
    values = [
        0.0,
        -0.0,
        0.25,
        0.75,
        1.25,
        1.75,
        2.5,
        3.5,
        5.0,
        6.0,
        -0.25,
        -0.75,
        -1.25,
        -1.75,
        -2.5,
        -3.5,
    ]
    x = torch.tensor([values * 32], device="cuda")
    cache = deepseek_v41_quantize_cache(x, format="main_kv_fp4")
    actual = unpack_bytes(cache, 1, True)
    assert actual[0][0, 0].item() == 128
    for a, e in zip(actual, reference(x, "main_kv_fp4"), strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("fmt,width", [("main_kv_fp4", 288), ("swa_mxfp8", 528)])
def test_cache_write_above_2gib(fmt, width):
    gate()
    page = (2**31 + 64 * width - 1) // (64 * width)
    out = torch.full((page + 1, 64, 1, width), 197, device="cuda", dtype=torch.uint8)
    x = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
    slot = torch.tensor([page * 64], device="cuda", dtype=torch.int32)
    deepseek_v41_quantize_cache(x, format=fmt, out=out, slots=slot)
    for a, e in zip(
        unpack_bytes(out[page:], 1, fmt == "main_kv_fp4"),
        reference(x, fmt),
        strict=True,
    ):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    assert (out[0] == 197).all()
    assert (out[page - 1] == 197).all()


@pytest.mark.parametrize("fmt", ["main_kv_fp4", "swa_mxfp8"])
def test_empty_and_rejected_declarations(fmt):
    gate()
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    out = deepseek_v41_quantize_cache(x, format=fmt)
    empty = deepseek_v41_quantize_cache(x[:0], format=fmt)
    assert empty.shape[0] == 0
    with pytest.raises(ValueError, match="page_size=64"):
        deepseek_v41_quantize_cache(x, format=fmt, page_size=32)
    with pytest.raises(ValueError, match="cache format"):
        deepseek_v41_quantize_cache(x, format="mxfp4")
    with pytest.raises(ValueError, match="QAT"):
        deepseek_v41_quantize_cache(x.requires_grad_(), format=fmt)
    x = x.detach()
    with pytest.raises(ValueError, match="overlap"):
        alias = x.view(torch.uint8).flatten()[: out.numel()].view_as(out)
        deepseek_v41_quantize_cache(x, format=fmt, out=alias)
    with pytest.raises(ValueError, match="overlap"):
        slots = out.flatten()[: x.shape[0] * 4].view(torch.int32)
        deepseek_v41_quantize_cache(x, format=fmt, out=out, slots=slots)
    with pytest.raises(ValueError, match="capacity"):
        deepseek_v41_quantize_cache(x, format=fmt, out=out[:0])
    torch.cuda.set_sync_debug_mode("error")
    try:
        deepseek_v41_quantize_cache(x, format=fmt, out=out)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_prepared_cache_feeds_decode():
    gate()
    from flashinfer.deepseek_v41 import deepseek_v41_decode
    from .test_deepseek_v41_decode import check

    x = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
    swa = deepseek_v41_quantize_cache(x, format="swa_mxfp8")
    main = deepseek_v41_quantize_cache(x, format="main_kv_fp4")
    ids = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 1, 128)
    q = torch.randn(1, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(64, device="cuda")
    args = q, swa, main, ids, ids, sink
    out, lse, _ = deepseek_v41_decode(*args)
    check(args, out, lse)
