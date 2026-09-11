# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from flashinfer.deepseek_v41 import (
    deepseek_v41_decode_fp32,
    deepseek_v41_decode_bf16x3,
    deepseek_v41_quantize_cache,
)
from .test_deepseek_v41 import gate, reference


@pytest.fixture(params=[deepseek_v41_decode_fp32, deepseek_v41_decode_bf16x3])
def decode(request):
    return request.param


def decoded(values, fmt):
    data, scales = reference(values, fmt)
    if fmt == "swa_mxfp8":
        return (
            data.view(torch.float8_e4m3fn).float()
            * scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32, -1)
        ).bfloat16()
    codes = torch.stack((data & 15, data >> 4), -1).flatten(-2)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=values.device)
    return (
        levels[(codes & 7).long()]
        * torch.where(codes & 8 != 0, -1, 1)
        * scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    ).bfloat16()


@pytest.mark.parametrize(
    "batch,sq,page,mixed",
    [(1, 1, 32, True), (2, 5, 64, True), (4, 1, 128, True), (2, 5, 64, False)],
)
def test_decode_fp32_mixed_pages_fp64_changed_graph_empty_sink_and_alias(
    batch, sq, page, mixed, decode
):
    gate()
    torch.manual_seed(42323 + batch + sq + page)
    q = torch.randn(batch, sq, 64, 512, device="cuda", dtype=torch.bfloat16)
    swa_values = torch.randn(page * 5, 512, device="cuda", dtype=torch.bfloat16)
    main_values = torch.randn(page * 7, 512, device="cuda", dtype=torch.bfloat16)
    swa = deepseek_v41_quantize_cache(swa_values, format="swa_mxfp8", page_size=page)
    main = (
        deepseek_v41_quantize_cache(main_values, format="main_kv_fp4", page_size=page)
        if mixed
        else None
    )
    swa_k = 128 if mixed else 192
    swa_ids = torch.randint(
        page * 5, (batch, sq, swa_k), device="cuda", dtype=torch.int32
    )
    main_ids = (
        torch.randint(page * 7, (batch, sq, 512), device="cuda", dtype=torch.int32)
        if mixed
        else None
    )
    sink = torch.randn(64, device="cuda")
    out, lse, workspace = decode(q, swa, main, swa_ids, main_ids, sink)

    def run():
        return decode(
            q, swa, main, swa_ids, main_ids, sink, out=out, lse=lse, workspace=workspace
        )

    def verify():
        values = decoded(swa_values, "swa_mxfp8")
        indices = swa_ids.flatten(0, 1).long()
        if mixed:
            values = torch.cat((values, decoded(main_values, "main_kv_fp4")))
            offset_ids = torch.where(main_ids >= 0, main_ids + page * 5, -1)
            indices = torch.cat((indices, offset_ids.flatten(0, 1)), -1)
        valid = indices >= 0
        kv = values[indices.clamp_min(0)].double()
        score = torch.einsum("qhd,qkd->qhk", q.flatten(0, 1).double(), kv) * (512**-0.5)
        score = score.masked_fill(~valid[:, None, :], -torch.inf)
        augmented = torch.cat(
            (score, sink.double()[None, :, None].expand(batch * sq, -1, 1)), -1
        )
        probability = augmented.softmax(-1)[..., :-1]
        expected = torch.einsum("qhk,qkd->qhd", probability, kv).bfloat16().view_as(out)
        expected_lse = score.logsumexp(-1).view(batch, sq, 64).transpose(1, 2).float()
        difference = out.float() - expected.float()
        relative = difference.norm() / expected.float().norm().clamp_min(1e-20)
        maximum = difference.abs().max() / expected.float().abs().max().clamp_min(1e-20)
        assert torch.isfinite(out).all()
        assert relative < 3e-4, float(relative)
        assert maximum < 0.004, float(maximum)
        torch.testing.assert_close(lse, expected_lse, atol=3e-6, rtol=1e-5)

    verify()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for step in range(3):
        q.copy_(torch.randn_like(q))
        swa_values.copy_(torch.randn_like(swa_values))
        deepseek_v41_quantize_cache(
            swa_values, format="swa_mxfp8", page_size=page, out=swa
        )
        swa_ids.copy_(
            torch.randint(page * 5, swa_ids.shape, device="cuda", dtype=torch.int32)
        )
        swa_ids[..., step * 32 + 1 :] = -1
        if mixed:
            main_values.copy_(torch.randn_like(main_values))
            deepseek_v41_quantize_cache(
                main_values, format="main_kv_fp4", page_size=page, out=main
            )
            main_ids.copy_(
                torch.randint(
                    page * 7, main_ids.shape, device="cuda", dtype=torch.int32
                )
            )
            main_ids[..., step * 64 :] = -1
        sink.copy_(torch.randn_like(sink))
        sink[0], sink[1] = 1000, -1000
        if step == 2:
            swa_ids[0, 0] = -1
            if mixed:
                main_ids[0, 0] = -1
        q_before = q.clone()
        swa_before = swa.clone()
        graph.replay()
        torch.testing.assert_close(q, q_before, atol=0, rtol=0)
        torch.testing.assert_close(swa, swa_before, atol=0, rtol=0)
        verify()
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    with pytest.raises(ValueError, match="overlap"):
        decode(
            q, swa, main, swa_ids, main_ids, sink, out=q, lse=lse, workspace=workspace
        )
    with pytest.raises(ValueError, match="inference"):
        decode(
            q.requires_grad_(),
            swa,
            main,
            swa_ids,
            main_ids,
            sink,
            out=out,
            lse=lse,
            workspace=workspace,
        )


@pytest.mark.parametrize(
    "scale_byte,data_byte,expected_power",
    [(0, 0x78, -120), (254, 1, 117), (0, 0x10, -133)],
)
def test_decode_fp32_e8m0_extreme_finite_scales(
    scale_byte, data_byte, expected_power, decode
):
    gate()
    q = torch.zeros(1, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(1, 64, 1, 528, device="cuda", dtype=torch.uint8)
    raw = cache.flatten()
    raw[:512] = data_byte
    raw[64 * 512 : 64 * 512 + 16] = scale_byte
    indices = torch.full((1, 1, 128), -1, device="cuda", dtype=torch.int32)
    indices[..., 0] = 0
    sink = torch.zeros(64, device="cuda")
    out, lse, _ = decode(q, cache, None, indices, None, sink)
    # Equal zero key/sink logits produce exactly half the one decoded value.
    # Check bits without norms that could underflow/overflow at these scales.
    torch.testing.assert_close(
        out.view(torch.uint8),
        torch.full_like(out, 2.0**expected_power).view(torch.uint8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(lse, torch.zeros_like(lse), atol=0, rtol=0)
