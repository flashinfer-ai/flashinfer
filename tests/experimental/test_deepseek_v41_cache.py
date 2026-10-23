# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Exact encoded-byte contract for the optional decode cache writer."""

import pytest
import torch

from flashinfer.deepseek_v41 import deepseek_v41_quantize_cache


def gate():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("DS4.1 decode cache preparation requires SM100, SM103 or SM107")


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


@pytest.mark.parametrize(
    "fmt,width,sf", [("main_kv_fp4", 256, 32), ("swa_mxfp8", 512, 16)]
)
def test_invalid_slots_are_masked_under_graph_replay(fmt, width, sf):
    gate()
    x = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor(
        [0, 127, -1, -2, 128, 2**31 - 1, 64], device="cuda", dtype=torch.int32
    )
    out = torch.full((2, 64, 1, width + sf), 173, device="cuda", dtype=torch.uint8)
    deepseek_v41_quantize_cache(x, format=fmt, out=out, slots=slots)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_quantize_cache(x, format=fmt, out=out, slots=slots)
    for step in range(2):
        if step:
            slots.fill_(-2)
        x.copy_(torch.randn_like(x))
        out.fill_(173)
        expected = torch.full_like(out, 173).view(2, -1)
        graph.replay()
        data, scales = reference(x, fmt)
        for row, slot in enumerate(slots.tolist()):
            if not 0 <= slot < 128:
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
    from flashinfer.deepseek_v41 import deepseek_v41_decode
    from .test_deepseek_v41_decode import check, gate as decode_gate

    decode_gate()
    x = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
    swa = deepseek_v41_quantize_cache(x, format="swa_mxfp8")
    main = deepseek_v41_quantize_cache(x, format="main_kv_fp4")
    ids = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 1, 128)
    q = torch.randn(1, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(64, device="cuda")
    args = q, swa, main, ids, ids, sink
    out, lse, _ = deepseek_v41_decode(*args)
    check(args, out, lse)


# Ordinary unit-scale FP8 has its own reference and tests. Do not route it
# through the legacy MXFP8 scale-plane oracle above.
def _swa_fp8_reference(x):
    return x.float().clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)


def _swa_fp8_bytes(cache):
    return cache.view(torch.uint8).view(-1, 512)


@pytest.mark.parametrize("rows", [1, 19, 257])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_swa_fp8_exact_bytes_and_changed_graph(rows, dtype):
    gate()
    torch.manual_seed(941 + rows)
    x = torch.randn(rows, 512, device="cuda", dtype=dtype) * 300
    x[0, :32] = 0
    x[0, 1] = -0.0
    out = deepseek_v41_quantize_cache(x, format="swa_fp8")
    assert out.dtype == torch.float8_e4m3fn
    assert out.shape == ((rows + 63) // 64, 64, 1, 512)
    assert out.is_contiguous()

    def check():
        torch.testing.assert_close(
            _swa_fp8_bytes(out)[:rows], _swa_fp8_reference(x), rtol=0, atol=0
        )

    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        assert deepseek_v41_quantize_cache(x, format="swa_fp8", out=out) is out
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        x.copy_(torch.randn_like(x) * 300)
        out.view(torch.uint8).fill_(197)
        graph.replay()
        check()
        # The writer owns only the supplied rows, not trailing page padding.
        assert (_swa_fp8_bytes(out)[rows:] == 197).all()
    out.view(torch.uint8).flatten()[0].bitwise_xor_(1)
    with pytest.raises(AssertionError):
        check()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_swa_fp8_all_finite_codes(dtype):
    gate()
    # E4M3FN has precisely two NaN byte codes; every other code is finite,
    # including both zeros and the positive/negative subnormal families.
    codes = torch.tensor(
        [code for code in range(256) if code not in (127, 255)],
        device="cuda",
        dtype=torch.uint8,
    ).repeat(3)[:512]
    x = codes.view(torch.float8_e4m3fn).to(dtype).view(1, 512)
    out = deepseek_v41_quantize_cache(x, format="swa_fp8")
    torch.testing.assert_close(
        _swa_fp8_bytes(out)[:1], codes.view(1, 512), rtol=0, atol=0
    )
    torch.testing.assert_close(
        _swa_fp8_bytes(out)[:1], _swa_fp8_reference(x), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_swa_fp8_halfway_even_saturation_and_signed_zero(dtype):
    gate()
    values = [
        0.0,
        -0.0,
        2**-10,
        -(2**-10),
        3 * 2**-10,
        -3 * 2**-10,
        5 * 2**-10,
        -5 * 2**-10,
        7 * 2**-10,
        -7 * 2**-10,
        2**-9,
        -(2**-9),
        7 * 2**-9,
        -7 * 2**-9,
        15 * 2**-10,
        -15 * 2**-10,
        1.0625,
        -1.0625,
        1.1875,
        -1.1875,
        400.0,
        -400.0,
        432.0,
        -432.0,
        448.0,
        -448.0,
        449.0,
        -449.0,
        1000.0,
        -1000.0,
        1e30,
        -1e30,
    ]
    expected_codes = [
        0x00,
        0x80,
        0x00,
        0x80,
        0x02,
        0x82,
        0x02,
        0x82,
        0x04,
        0x84,
        0x01,
        0x81,
        0x07,
        0x87,
        0x08,
        0x88,
        0x38,
        0xB8,
        0x3A,
        0xBA,
        0x7C,
        0xFC,
        0x7E,
        0xFE,
        0x7E,
        0xFE,
        0x7E,
        0xFE,
        0x7E,
        0xFE,
        0x7E,
        0xFE,
    ]
    x = torch.tensor([values * 16], device="cuda", dtype=dtype)
    expected = torch.tensor([expected_codes * 16], device="cuda", dtype=torch.uint8)
    out = deepseek_v41_quantize_cache(x, format="swa_fp8")
    torch.testing.assert_close(_swa_fp8_bytes(out)[:1], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        _swa_fp8_bytes(out)[:1], _swa_fp8_reference(x), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_swa_fp8_scatter_changed_slots_invalid_and_untouched_bytes(dtype):
    gate()
    x = torch.randn(7, 512, device="cuda", dtype=dtype) * 300
    slots = torch.tensor(
        [67, 1, -1, 63, 64, 128, 2**31 - 1], device="cuda", dtype=torch.int32
    )
    out_bytes = torch.full((2, 64, 1, 512), 173, device="cuda", dtype=torch.uint8)
    out = out_bytes.view(torch.float8_e4m3fn)
    expected = torch.empty_like(out_bytes)

    def check():
        expected.fill_(173)
        data = _swa_fp8_reference(x)
        for row, slot in enumerate(slots.tolist()):
            if 0 <= slot < 128:
                expected.view(-1, 512)[slot].copy_(data[row])
        torch.testing.assert_close(out_bytes, expected, rtol=0, atol=0)

    assert deepseek_v41_quantize_cache(x, format="swa_fp8", out=out, slots=slots) is out
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_quantize_cache(x, format="swa_fp8", out=out, slots=slots)
    changed_slots = (
        [67, 1, -1, 63, 64, 128, 2**31 - 1],
        [2**31 - 1, 65, 127, -2, 0, -1, 128],
        [-1, -2, 128, 2**31 - 1, -3, 129, 2**31 - 2],
    )
    for updated in changed_slots:
        slots.copy_(torch.tensor(updated, device="cuda", dtype=torch.int32))
        x.copy_(torch.randn_like(x) * 300)
        out_bytes.fill_(173)
        graph.replay()
        check()


def test_swa_fp8_cache_write_above_2gib():
    gate()
    # Match the existing large-pool test policy (the shared hook handles OOM).
    page = (2**31 + 64 * 512 - 1) // (64 * 512)
    out_bytes = torch.full(
        (page + 1, 64, 1, 512), 197, device="cuda", dtype=torch.uint8
    )
    out = out_bytes.view(torch.float8_e4m3fn)
    x = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
    slot = torch.tensor([page * 64], device="cuda", dtype=torch.int32)
    deepseek_v41_quantize_cache(x, format="swa_fp8", out=out, slots=slot)
    torch.testing.assert_close(
        _swa_fp8_bytes(out[page:])[:1], _swa_fp8_reference(x), rtol=0, atol=0
    )
    assert (out_bytes[0] == 197).all()
    assert (out_bytes[page - 1] == 197).all()
    assert (out_bytes[page, 1:] == 197).all()


def test_swa_fp8_empty_and_rejected_declarations():
    gate()
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    out_bytes = torch.full((1, 64, 1, 512), 173, device="cuda", dtype=torch.uint8)
    out = out_bytes.view(torch.float8_e4m3fn)
    empty = deepseek_v41_quantize_cache(x[:0], format="swa_fp8")
    assert empty.shape == (0, 64, 1, 512)
    assert empty.dtype == torch.float8_e4m3fn
    empty_slots = torch.empty(0, device="cuda", dtype=torch.int32)
    assert (
        deepseek_v41_quantize_cache(x[:0], format="swa_fp8", out=out, slots=empty_slots)
        is out
    )
    assert (out_bytes == 173).all()
    with pytest.raises(ValueError, match="caller-owned"):
        deepseek_v41_quantize_cache(x[:0], format="swa_fp8", slots=empty_slots)
    with pytest.raises(ValueError, match="page_size=64"):
        deepseek_v41_quantize_cache(x, format="swa_fp8", page_size=32)
    with pytest.raises(ValueError, match="cache format"):
        deepseek_v41_quantize_cache(x, format="fp8")
    for invalid in (
        x.half(),
        x.to(torch.float8_e4m3fn),
        x[0],
        x[:, :256],
        torch.empty((64, 1024), device="cuda", dtype=x.dtype)[:, ::2],
    ):
        with pytest.raises(ValueError, match="contiguous BF16/FP32"):
            deepseek_v41_quantize_cache(invalid, format="swa_fp8")
    for invalid in (
        out_bytes,
        torch.empty((1, 64, 1, 528), device="cuda", dtype=torch.uint8),
        torch.empty((1, 64, 1, 528), device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty((1, 32, 1, 512), device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty_like(x).view(1, 64, 1, 512),
        torch.empty_like(out, device="cpu"),
        out.expand(2, -1, -1, -1),
    ):
        with pytest.raises(ValueError, match="swa_fp8 output"):
            deepseek_v41_quantize_cache(x, format="swa_fp8", out=invalid)
    slots = torch.arange(64, device="cuda", dtype=torch.int32)
    for invalid in (
        slots.long(),
        slots.view(64, 1),
        torch.arange(128, device="cuda", dtype=torch.int32)[::2],
        torch.arange(64, dtype=torch.int32),
    ):
        with pytest.raises(ValueError, match="slots must"):
            deepseek_v41_quantize_cache(x, format="swa_fp8", out=out, slots=invalid)
    with pytest.raises(ValueError, match="QAT"):
        deepseek_v41_quantize_cache(x.requires_grad_(), format="swa_fp8")
    x = x.detach()
    with pytest.raises(ValueError, match="QAT"):
        deepseek_v41_quantize_cache(
            x, format="swa_fp8", out=out.detach().requires_grad_()
        )
    with pytest.raises(ValueError, match="overlap"):
        alias = (
            x.view(torch.uint8).flatten()[: out.numel()].view(torch.float8_e4m3fn)
        ).view_as(out)
        deepseek_v41_quantize_cache(x, format="swa_fp8", out=alias)
    with pytest.raises(ValueError, match="overlap"):
        alias_slots = out_bytes.flatten()[: x.shape[0] * 4].view(torch.int32)
        deepseek_v41_quantize_cache(x, format="swa_fp8", out=out, slots=alias_slots)
    with pytest.raises(ValueError, match="capacity"):
        deepseek_v41_quantize_cache(x, format="swa_fp8", out=out[:0])


def test_swa_fp8_writer_has_no_host_sync():
    gate()
    x = torch.randn(7, 512, device="cuda", dtype=torch.float32)
    slots = torch.tensor(
        [0, 127, -1, -2, 128, 2**31 - 1, 64], device="cuda", dtype=torch.int32
    )
    out_bytes = torch.full((2, 64, 1, 512), 173, device="cuda", dtype=torch.uint8)
    out = out_bytes.view(torch.float8_e4m3fn)
    # Warm compilation before enabling the same synchronization guard used by
    # the legacy writer tests. Slot values remain device-side metadata.
    deepseek_v41_quantize_cache(x, format="swa_fp8", out=out, slots=slots)
    torch.cuda.set_sync_debug_mode("error")
    try:
        assert (
            deepseek_v41_quantize_cache(x, format="swa_fp8", out=out, slots=slots)
            is out
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")
    expected = torch.full_like(out_bytes, 173).view(-1, 512)
    data = _swa_fp8_reference(x)
    for row, slot in enumerate(slots.tolist()):
        if 0 <= slot < 128:
            expected[slot].copy_(data[row])
    torch.testing.assert_close(_swa_fp8_bytes(out), expected, rtol=0, atol=0)


@pytest.mark.parametrize("compressed_k", [0, 128, 192, 512])
def test_swa_fp8_prepared_cache_feeds_prims_decode(compressed_k):
    from .test_deepseek_v41_decode import (
        check_fp8,
        decode_case,
        window_case,
        fp8_oracle,
        prims_ts_gate,
    )

    prims_ts_gate()
    torch.manual_seed(941 + compressed_k)
    x = torch.randn(max(128, compressed_k), 512, device="cuda", dtype=torch.bfloat16)
    x.clamp_(-0.5, 0.5)
    x[0, -1] = 3  # The final group16 scale must survive direct reconstruction.
    swa = deepseek_v41_quantize_cache(x[:128], format="swa_fp8")
    main = (
        deepseek_v41_quantize_cache(x, format="main_kv_fp4") if compressed_k else None
    )
    wi = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 1, 128)
    ci = (
        torch.arange(compressed_k, device="cuda", dtype=torch.int32).view(
            1, 1, compressed_k
        )
        if compressed_k
        else None
    )
    q_source = torch.randn(1, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    q = q_source.to(torch.float8_e4m3fn)
    sink = torch.randn(64, device="cuda")
    args = q, swa, main, wi, ci, sink

    def run(buffers=None):
        if compressed_k:
            return decode_case(*args, buffers=buffers, backend="prims-ts")
        return window_case(q, swa, wi, sink, buffers=buffers, backend="prims-ts")

    def check():
        # Byte gates are separate from the represented-value FP64 decode gate.
        torch.testing.assert_close(
            _swa_fp8_bytes(swa)[:128], _swa_fp8_reference(x[:128]), rtol=0, atol=0
        )
        if main is not None:
            for actual, expected in zip(
                unpack_bytes(main, x.shape[0], True),
                reference(x, "main_kv_fp4"),
                strict=True,
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not hasattr(buffers.views, "main_scale")
        check_fp8(out, lse, fp8_oracle(*args))

    out, lse, buffers = run()
    check()
    if compressed_k:
        previous_bytes = main.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_quantize_cache(x[:128], format="swa_fp8", out=swa)
        if compressed_k:
            deepseek_v41_quantize_cache(x, format="main_kv_fp4", out=main)
        run(buffers)
    for step in range(3):
        x.normal_().clamp_(-1, 1).mul_(step + 1)
        # Live writer group scales change, and the joint consumer must reread
        # payload/scale bytes without any decode-time amax or request descale.
        x[0, -1] = 6 * (step + 1)
        q_source.normal_()
        q.copy_(q_source.to(torch.float8_e4m3fn))
        sink.normal_()
        wi.copy_(torch.randint(0, 128, wi.shape, device="cuda", dtype=torch.int32))
        wi[..., ::11] = -1
        if compressed_k:
            ci.copy_(
                torch.randint(
                    0, compressed_k, ci.shape, device="cuda", dtype=torch.int32
                )
            )
            ci[..., ::17] = -1
            ci[..., 1] = 0  # Keep the changed maximum selected on live replays.
            main.fill_(255)
        if step == 2:
            wi.fill_(-1)
            if compressed_k:
                ci.fill_(-1)
        swa.view(torch.uint8).fill_(255)
        out.view(torch.uint8).fill_(255)
        lse.fill_(float("nan"))
        graph.replay()
        check()
        if compressed_k:
            _, written_scales = unpack_bytes(main, x.shape[0], True)
            assert written_scales[0, -1].item() == (0x38, 0x40, 0x44)[step]
            assert not torch.equal(main, previous_bytes)
            previous_bytes.copy_(main)


@pytest.mark.parametrize(
    "fmt,width,dtype",
    [
        ("main_kv_fp4", 288, torch.uint8),
        ("swa_fp8", 512, torch.float8_e4m3fn),
    ],
)
def test_default_nhd_and_explicit_hnd_writer_bytes_and_identity(fmt, width, dtype):
    gate()
    x = torch.randn(65, 512, device="cuda", dtype=torch.bfloat16)
    nhd = deepseek_v41_quantize_cache(x, format=fmt)
    hnd = deepseek_v41_quantize_cache(x, format=fmt, kv_layout="HND")
    assert nhd.shape == (2, 64, 1, width) and hnd.shape == (2, 1, 64, width)
    assert nhd.dtype == hnd.dtype == dtype
    if fmt == "swa_fp8":
        torch.testing.assert_close(
            _swa_fp8_bytes(nhd)[:65], _swa_fp8_reference(x), rtol=0, atol=0
        )
        torch.testing.assert_close(
            _swa_fp8_bytes(nhd)[:65], _swa_fp8_bytes(hnd)[:65], rtol=0, atol=0
        )
    else:
        for actual, expected, other in zip(
            unpack_bytes(nhd, 65, True),
            reference(x, fmt),
            unpack_bytes(hnd, 65, True),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual, other, rtol=0, atol=0)
    assert deepseek_v41_quantize_cache(x, format=fmt, out=nhd) is nhd
    assert deepseek_v41_quantize_cache(x, format=fmt, out=hnd, kv_layout="HND") is hnd
