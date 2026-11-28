# SPDX-License-Identifier: Apache-2.0
"""Exercise eager and replay staging, including a shrinking batch in one workspace."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp8_bf16_cutedsl.staging import (
    stage_mega_moe_inputs,
    try_stage_mxfp4_with_shared,
)
from flashinfer.moe_ep.backends.mega.kernel.sm107.shared_expert import (
    stage_shared_inputs,
)
from flashinfer.moe_ep.tensors import MoEEpTensors


def workspace(capacity, hidden, topk, kind):
    sf_cols = hidden // 32
    sf_rows = ((capacity + 127) // 128) * 128
    fp8_dtype = torch.float8_e4m3fn if kind == "mxfp8_e4m3" else torch.float8_e5m2
    return SimpleNamespace(
        config=SimpleNamespace(shared_expert=SimpleNamespace(kind=kind)),
        x=torch.empty(
            (capacity, hidden // 2), device="cuda", dtype=torch.float4_e2m1fn_x2
        ),
        x_sf=torch.empty(
            (capacity, sf_cols), device="cuda", dtype=torch.float8_e8m0fnu
        ),
        shared_x=torch.empty((capacity, hidden), device="cuda", dtype=fp8_dtype),
        shared_x_sf=torch.empty(
            (sf_rows * sf_cols,), device="cuda", dtype=torch.float8_e8m0fnu
        ),
        shared_x_sf_linear=torch.empty(
            (capacity, sf_cols), device="cuda", dtype=torch.float8_e8m0fnu
        ),
        topk_idx=torch.full((capacity, topk), -1, device="cuda", dtype=torch.int32),
        topk_weights=torch.zeros((capacity, topk), device="cuda", dtype=torch.float32),
    )


def inputs(rows, hidden, topk):
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, 128, (rows, topk), device="cuda")
    ids[::3, 0] = -1
    return MoEEpTensors(x, ids, torch.rand(rows, topk, device="cuda"))


def reference(t, w):
    stage_shared_inputs(t, w, quantize_input=True)
    stage_mega_moe_inputs(
        t.hidden_states,
        t.topk_weights,
        t.topk_ids,
        w.x,
        w.x_sf,
        w.topk_idx,
        w.topk_weights,
        kind="mxfp4",
    )


def compare(t, fast, ref):
    n = t.num_tokens
    for key, all_rows in (
        ("x", False),
        ("x_sf", False),
        ("shared_x", True),
        ("shared_x_sf", True),
        ("topk_idx", True),
        ("topk_weights", False),
    ):
        actual = getattr(fast, key).view(torch.uint8)
        expected = getattr(ref, key).view(torch.uint8)
        if not all_rows:
            actual, expected = actual[:n], expected[:n]
        if not torch.equal(actual, expected):
            raise AssertionError(
                f"M={n} {key}: {(actual != expected).sum().item()} differing bytes"
            )


@pytest.mark.arch_rubin
@pytest.mark.parametrize(
    ("kind", "hidden", "capacity", "topk"),
    [
        ("mxfp8_e4m3", 5120, 128, 3),
        ("mxfp8_e4m3", 5120, 12288, 3),
        ("mxfp8_e5m2", 384, 131, 6),
    ],
)
def test_sm107_mxfp4_shared_staging_eager_and_replay(
    monkeypatch, kind, hidden, capacity, topk
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("SM107 GPU required")
    pytest.importorskip("triton")
    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "1")
    monkeypatch.setenv("FLASHINFER_SM107_MEGA_CHECK_ROUTES", "0")
    torch.manual_seed(1743)
    fast = workspace(capacity, hidden, topk, kind)
    ref = workspace(capacity, hidden, topk, kind)
    sizes = [
        n
        for n in (
            capacity,
            0,
            1,
            3,
            16,
            17,
            31,
            127,
            128,
            129,
            4097,
            8193,
            capacity - 1,
        )
        if n <= capacity
    ]
    for n in sizes:
        t = inputs(n, hidden, topk)
        assert try_stage_mxfp4_with_shared(t, fast)
        reference(t, ref)
        compare(t, fast, ref)
    graphs = []
    for n in [1, min(129, capacity), capacity - 1, 0]:
        t = inputs(n, hidden, topk)
        assert try_stage_mxfp4_with_shared(t, fast)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            assert try_stage_mxfp4_with_shared(t, fast)
        graphs.append((g, t))
    for i in (2, 0, 1, 3, 2, 0):
        g, t = graphs[i]
        g.replay()
        reference(t, ref)
        compare(t, fast, ref)
    for g, _ in graphs:
        g.reset()
    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "0")
    assert not try_stage_mxfp4_with_shared(inputs(1, hidden, topk), fast)
