# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from flashinfer.deepseek_v41 import deepseek_v41_rope_quantize_cache
from .test_deepseek_v41 import reference


def gate():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("V4.1 RoPE/cache requires SM100/SM103")


def rotate_reference(x, freqs, positions):
    result = x.clone()
    pairs = torch.view_as_complex(x[:, -64:].float().reshape(-1, 32, 2))
    rotations = torch.view_as_complex(freqs)[positions.clamp_min(0).long()]
    result[:, -64:] = torch.view_as_real(pairs * rotations).flatten(-2).bfloat16()
    return result


@pytest.mark.parametrize("page", [32, 64, 128])
@pytest.mark.parametrize("fmt", ["index_mxfp4", "main_kv_fp4", "swa_mxfp8"])
def test_rope_quant_cache_exact_regions_and_changing_graph(page, fmt):
    gate()
    torch.manual_seed(41533 + page)
    n = page + 7
    dim, width, sf = (
        (128, 64, 4)
        if fmt == "index_mxfp4"
        else (512, 256, 32)
        if fmt == "main_kv_fp4"
        else (512, 512, 16)
    )
    x = torch.randn(n, dim, device="cuda", dtype=torch.bfloat16)
    phases = torch.randn(65537, 32, device="cuda") * 10
    freqs = torch.view_as_real(torch.polar(torch.ones_like(phases), phases))
    positions = torch.randint(65537, (n,), device="cuda", dtype=torch.int32)
    positions[:4] = torch.tensor([0, 1, 65536, -1], device="cuda", dtype=torch.int32)
    stride = page * (width + sf)
    if fmt == "index_mxfp4":
        stride = (stride + 511) // 512 * 512
    storage = torch.full((3, stride), 197, device="cuda", dtype=torch.uint8)
    cache = storage.as_strided(
        (3, page, 1, width + sf), (stride, width + sf, width + sf, 1)
    )
    slots = torch.randperm(3 * page, device="cuda")[:n].int()
    slots[2] = -1
    expected = storage.clone()

    def run():
        return deepseek_v41_rope_quantize_cache(
            x, freqs, positions, format=fmt, out=cache, slots=slots, page_size=page
        )

    def verify():
        rotated = rotate_reference(x, freqs, positions)
        data, scales = reference(rotated, "mxfp4" if fmt == "index_mxfp4" else fmt)
        valid = (slots >= 0) & (positions >= 0)
        pages, local = slots[valid].long() // page, slots[valid].long() % page
        expected[
            pages[:, None], local[:, None] * width + torch.arange(width, device="cuda")
        ] = data[valid]
        expected[
            pages[:, None],
            page * width + local[:, None] * sf + torch.arange(sf, device="cuda"),
        ] = scales[valid]
        torch.testing.assert_close(storage, expected, rtol=0, atol=0)

    original = x.clone()
    run()
    verify()
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for step in range(3):
        x.copy_(torch.randn_like(x))
        positions.copy_(torch.randint(65537, (n,), device="cuda", dtype=torch.int32))
        positions[step] = -1
        slots.copy_(torch.randperm(3 * page, device="cuda")[:n].int())
        freqs.copy_(freqs.flip(-1))
        graph.replay()
        verify()
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_rope_quantize_cache(
            x,
            freqs,
            positions,
            format=fmt,
            out=freqs.view(torch.uint8)
            .flatten()[: storage.numel()]
            .view_as(storage)
            .as_strided(cache.shape, cache.stride()),
            slots=slots,
            page_size=page,
        )


@pytest.mark.parametrize("fmt", ["index_mxfp4", "main_kv_fp4", "swa_mxfp8"])
def test_rope_quant_cache_long_batch_rounding_boundary(fmt):
    gate()
    from flashinfer.deepseek_v41 import (
        deepseek_v41_quantize_cache,
        deepseek_v41_quantize_index_cache,
    )

    torch.manual_seed(415331)
    n, dim = 32768, 128 if fmt == "index_mxfp4" else 512
    x = torch.randn(n, dim, device="cuda", dtype=torch.bfloat16)
    phase = torch.randn(n, 32, device="cuda") * 100
    freqs = torch.view_as_real(torch.polar(torch.ones_like(phase), phase))
    positions = torch.arange(n, device="cuda", dtype=torch.int32)
    rotated = rotate_reference(x, freqs, positions)
    if fmt == "index_mxfp4":
        expected = deepseek_v41_quantize_index_cache(rotated)
    else:
        expected = deepseek_v41_quantize_cache(rotated, format=fmt)
    out = torch.empty_strided(
        expected.shape, expected.stride(), device="cuda", dtype=torch.uint8
    )
    deepseek_v41_rope_quantize_cache(
        x, freqs, positions, format=fmt, out=out, slots=positions
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("fmt", ["index_mxfp4", "main_kv_fp4", "swa_mxfp8"])
def test_rope_and_component_cache_byte_offsets_beyond_int32(fmt):
    gate()
    from flashinfer.deepseek_v41 import (
        deepseek_v41_pack_cache,
        deepseek_v41_quantize_cache,
        deepseek_v41_quantize_index_cache,
    )

    dim, width, sf = (
        (128, 64, 4)
        if fmt == "index_mxfp4"
        else (512, 256, 32)
        if fmt == "main_kv_fp4"
        else (512, 512, 16)
    )
    page = 64
    stride = page * (width + sf)
    if fmt == "index_mxfp4":
        stride = (stride + 511) // 512 * 512
    pages = (1 << 31) // stride + 2
    # Allocate virtual cache capacity, initialize only the guarded/target pages.
    storage = torch.empty(pages, stride, device="cuda", dtype=torch.uint8)
    cache = storage.as_strided(
        (pages, page, 1, width + sf), (stride, width + sf, width + sf, 1)
    )
    storage[0].fill_(213)
    x = torch.linspace(-6, 6, dim, device="cuda").bfloat16().view(1, dim)
    freqs = torch.zeros(1, 32, 2, device="cuda")
    freqs[:, :, 0] = 1
    position = torch.zeros(1, device="cuda", dtype=torch.int32)
    slot = torch.tensor([pages * page - 1], device="cuda", dtype=torch.int32)
    data, scale = reference(x, "mxfp4" if fmt == "index_mxfp4" else fmt)
    expected = torch.full((stride,), 197, device="cuda", dtype=torch.uint8)
    expected[(page - 1) * width : page * width] = data[0]
    expected[page * width + (page - 1) * sf : page * (width + sf)] = scale[0]
    for arm in ("fused", "quantize", "pack"):
        if arm == "pack" and fmt == "index_mxfp4":
            # The separate pack API only declares D512 main/window formats.
            continue
        storage[-1].fill_(197)
        if arm == "fused":
            deepseek_v41_rope_quantize_cache(
                x, freqs, position, format=fmt, out=cache, slots=slot
            )
        elif arm == "pack":
            deepseek_v41_pack_cache(data, scale, out=cache, slots=slot)
        elif fmt == "index_mxfp4":
            deepseek_v41_quantize_index_cache(x, out=cache, slots=slot)
        else:
            deepseek_v41_quantize_cache(x, format=fmt, out=cache, slots=slot)
        torch.testing.assert_close(storage[-1], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            storage[0], torch.full_like(storage[0], 213), rtol=0, atol=0
        )
