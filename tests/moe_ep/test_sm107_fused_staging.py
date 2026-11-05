# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compare SM107 fused staging with its canonical Torch fallback."""

import pytest
import torch

from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp8_bf16_cutedsl import (
    staging,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Requires SM107",
)


def compare(monkeypatch, x, topk=10, kind="mxfp8_e4m3", graph=False, strided=False):
    rows, hidden = x.shape
    capacity = rows + 4
    dtype = torch.float8_e5m2 if kind == "mxfp8_e5m2" else torch.float8_e4m3fn
    ids = torch.randint(-1, 128, (rows, topk), device="cuda", dtype=torch.int64)
    weights = torch.rand(rows, topk, device="cuda")

    def buffers():
        q = torch.zeros(
            capacity, hidden * (2 if strided else 1), device="cuda", dtype=dtype
        )
        if strided:
            q = q[:, ::2]
        return (
            q,
            torch.full(
                (capacity, hidden // 32 + 8), 255, device="cuda", dtype=torch.uint8
            ),
            torch.full((capacity, topk), 99, device="cuda", dtype=torch.int32),
            torch.full((capacity, topk), 99.0, device="cuda"),
        )

    expected, actual = buffers(), buffers()
    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "0")
    assert staging.stage_mega_moe_inputs(x, weights, ids, *expected, kind=kind) == rows
    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "1")
    assert staging.stage_mega_moe_inputs(x, weights, ids, *actual, kind=kind) == rows

    def check():
        # Match NaN locations, but do not impose a NaN payload/sign ABI.
        en = torch.isnan(expected[0].float())
        an = torch.isnan(actual[0].float())
        assert torch.equal(en, an)
        assert torch.equal(
            expected[0].view(torch.uint8)[~en], actual[0].view(torch.uint8)[~an]
        )
        for a, b in zip(expected[1:], actual[1:], strict=True):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))

    check()
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            staging.stage_mega_moe_inputs(x, weights, ids, *actual, kind=kind)
        for _ in range(8):
            actual[1][:rows].fill_(255)
            actual[2].fill_(99)
            g.replay()
        torch.cuda.synchronize()
        check()


@pytest.mark.parametrize(
    "rows,hidden,topk",
    [
        (0, 768, 10),
        (1, 32, 1),
        (7, 96, 3),
        (4, 768, 10),
        (8, 768, 10),
        (256, 768, 10),
        (512, 768, 10),
        (9, 1024, 8),
        (3, 8192, 128),
    ],
)
def test_layouts(monkeypatch, rows, hidden, topk):
    compare(
        monkeypatch,
        torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16),
        topk,
        graph=rows > 0,
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_finite_bf16_encodings(monkeypatch, mixed):
    codes = torch.arange(65536, device="cuda", dtype=torch.int32)
    values = codes[(codes & 0x7F80) != 0x7F80].to(torch.int16).view(torch.bfloat16)
    x = (
        values.reshape(-1, 768)
        if mixed
        else values.repeat_interleave(32).reshape(-1, 768)
    )
    compare(monkeypatch, x)


def test_nonfinite(monkeypatch):
    x = torch.randn(8, 768, device="cuda", dtype=torch.bfloat16)
    x[0, 0] = float("nan")
    x[1, 0] = float("inf")
    x[2, 0] = -float("inf")
    x[3, :32] = float("nan")
    x[4, :32] = 0
    x[5, :32] = -0.0
    x[6, 0] = float("nan")
    x[6, 1] = 450 * 2.0**-126
    compare(monkeypatch, x, graph=True)


@pytest.mark.parametrize("kind,strided", [("mxfp8_e5m2", False), ("mxfp8_e4m3", True)])
def test_fallback(monkeypatch, kind, strided):
    compare(
        monkeypatch,
        torch.randn(7, 768, device="cuda", dtype=torch.bfloat16),
        kind=kind,
        strided=strided,
    )
