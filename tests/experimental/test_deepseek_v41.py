# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from flashinfer.deepseek_v41 import (
    deepseek_v41_pack_cache,
    deepseek_v41_quantize,
    deepseek_v41_quantize_cache,
    deepseek_v41_quantize_index_cache,
)


def gate():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("V4.1 kernels require SM100/SM103")


@pytest.mark.parametrize("page", [32, 64, 128])
def test_index_cache_regions_padding_scatter_and_graph(page):
    gate()
    torch.manual_seed(410 + page)
    tokens = page + 7
    x = torch.randn(tokens, 128, dtype=torch.bfloat16, device="cuda")
    pages, stride = 3, ((page * 68 + 511) // 512) * 512
    storage = torch.full((pages, stride), 197, dtype=torch.uint8, device="cuda")
    cache = storage.as_strided((pages, page, 1, 68), (stride, 68, 68, 1))
    slots = torch.randperm(pages * page, device="cuda")[:tokens].int()
    slots[2] = -1
    expected = storage.clone()

    def update_reference():
        data, scales = reference(x, "mxfp4")
        valid = slots >= 0
        page_ids, local = slots[valid].long() // page, slots[valid].long() % page
        expected[
            page_ids[:, None], local[:, None] * 64 + torch.arange(64, device="cuda")
        ] = data[valid]
        expected[
            page_ids[:, None],
            page * 64 + local[:, None] * 4 + torch.arange(4, device="cuda"),
        ] = scales[valid]

    deepseek_v41_quantize_index_cache(x, page_size=page, out=cache, slots=slots)
    update_reference()
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_quantize_index_cache(x, page_size=page, out=cache, slots=slots)
    x.mul_(0.75)
    slots.copy_(torch.randperm(pages * page, device="cuda")[:tokens].int())
    graph.replay()
    update_reference()
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)
    allocated = deepseek_v41_quantize_index_cache(x, page_size=page)
    assert allocated.shape == (2, page, 1, 68)
    assert allocated.stride(0) == stride
    bad = torch.empty_strided(
        (2, page, 1, 68), (stride + 4, 68, 68, 1), dtype=torch.uint8, device="cuda"
    )
    with pytest.raises(ValueError, match="512-byte-aligned"):
        deepseek_v41_quantize_index_cache(x, page_size=page, out=bad)


def reference(x, fmt):
    group = 16 if fmt == "main_kv_fp4" else 32
    g = x.float().view(x.shape[0], -1, group)
    amax = g.abs().amax(-1)
    if fmt == "main_kv_fp4":
        sf = (amax.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn)
    else:
        bound, minimum = (448, 1e-4) if fmt == "swa_mxfp8" else (6, 6 * 2**-126)
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


@pytest.mark.parametrize("fmt", ["mxfp4", "main_kv_fp4", "swa_mxfp8"])
@pytest.mark.parametrize("shape", [(1, 128), (19, 512), (257, 512)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_quantization_exact_bytes_and_graph(fmt, shape, dtype):
    gate()
    torch.manual_seed(414)
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    x[0, :32] = 0
    expected = reference(x, fmt)
    actual = deepseek_v41_quantize(x, format=fmt)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_quantize(x, format=fmt, data=actual[0], scales=actual[1])
    graph.replay()
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


def test_fp4_halfway_even_and_signed_zero():
    gate()
    # Amax=6 fixes MX scale to 1; values hit every E2M1 halfway boundary.
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
    x = torch.tensor(values * 8, device="cuda").view(1, 128)
    actual = deepseek_v41_quantize(x, format="mxfp4")
    expected = reference(x, "mxfp4")
    assert actual[0][0, 0].item() == 128
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize(
    "fmt,width,sf", [("main_kv_fp4", 256, 32), ("swa_mxfp8", 512, 16)]
)
@pytest.mark.parametrize("page", [32, 64, 128])
def test_page_layout_scatter_and_untouched_slots(fmt, width, sf, page):
    gate()
    x = torch.randn(5, 512, device="cuda", dtype=torch.bfloat16)
    data, scales = deepseek_v41_quantize(x, format=fmt)
    slots = torch.tensor(
        [page + 3, 1, -1, page - 1, page], device="cuda", dtype=torch.int32
    )
    output = torch.full((2, page, 1, width + sf), 173, device="cuda", dtype=torch.uint8)
    expected = output.clone().view(2, -1)
    for row, slot in enumerate([page + 3, 1, -1, page - 1, page]):
        if slot < 0:
            continue
        block, local = divmod(slot, page)
        expected[block, local * width : (local + 1) * width] = data[row]
        expected[block, page * width + local * sf : page * width + (local + 1) * sf] = (
            scales[row]
        )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        deepseek_v41_pack_cache(data, scales, page_size=page, out=output, slots=slots)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(output.flatten(), expected.flatten(), rtol=0, atol=0)
    fused = torch.full_like(output, 173)
    deepseek_v41_quantize_cache(x, format=fmt, page_size=page, out=fused, slots=slots)
    torch.testing.assert_close(fused.flatten(), expected.flatten(), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_quantize_cache(
            x, format=fmt, page_size=page, out=fused, slots=slots
        )
    graph.replay()
    torch.testing.assert_close(fused.flatten(), expected.flatten(), rtol=0, atol=0)


def test_reject_implicit_qat_and_wrong_cache_format():
    gate()
    x = torch.randn(2, 128, device="cuda", requires_grad=True)
    with pytest.raises(ValueError, match="QAT"):
        deepseek_v41_quantize(x, format="mxfp4")
    data, scales = deepseek_v41_quantize(x.detach(), format="mxfp4")
    with pytest.raises(ValueError, match="D512"):
        deepseek_v41_pack_cache(data, scales)
