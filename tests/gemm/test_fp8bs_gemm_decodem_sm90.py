# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the specialized SM90 decode-M backend of
fp8_blockscale_gemm_sm90 (agent-generated kernel, "specialized" backend)."""

import pytest
import torch

from flashinfer.gemm import fp8_blockscale_gemm_sm90
from flashinfer.gemm.fp8bs_decodem_sm90 import (
    FP8BS_GEMM_BACKENDS,
    SPECIALIZED_KERNEL_DISABLE_ENV,
    _dispatch_table,
    _eligibility,
    _fp8bs_decodem_stats,
    _reset_fp8bs_decodem_stats,
)

requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 9,
    reason="needs an SM90 device",
)

G = 128


def _make(M, N, K, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((M, K), generator=g, device=device,
                    dtype=torch.float32).to(torch.bfloat16)
    wq = torch.randn((N, K), generator=g, device=device,
                     dtype=torch.float32).to(torch.float8_e4m3fn)
    pow2 = torch.tensor([0.0625, 0.125, 0.25], device=device)
    ws = pow2[torch.randint(0, 3, ((N + G - 1) // G, K // G),
                            generator=g, device=device)].float().contiguous()
    return x, wq, ws


def _reference(x, wq, ws, eps=1e-4):
    """fp32 emulation of the contract: fused 1x128 activation quantization
    (amax/448) then dequantized matmul."""
    M, K = x.shape
    N = wq.shape[0]
    xf = x.float().view(M, K // G, G)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    xs = amax / 448.0
    xq = (xf / xs).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    xdq = (xq * xs).view(M, K)
    wf = wq.float().view(N, K // G, G)
    ws_rows = ws.repeat_interleave(G, dim=0)[:N]  # (N, K//G)
    wdq = (wf * ws_rows.unsqueeze(-1)).view(N, K)
    return xdq @ wdq.t()


def _rel_fro(a, b):
    return float(((a.float() - b).norm() / b.norm().clamp_min(1e-12)).item())


def test_backends_tuple_and_dispatch_table():
    table = _dispatch_table()
    assert table is not None, "packaged allowlist must load"
    assert table["backends"] == list(FP8BS_GEMM_BACKENDS)
    assert table["kill_switch"] == SPECIALIZED_KERNEL_DISABLE_ENV
    op = table["ops"]["fp8_blockscale_gemm_sm90"]
    assert op["m_band"] == [1, 24]
    assert op["compute_capability"] == [9, 0]


@requires_sm90
def test_unknown_backend_raises():
    x, wq, ws = _make(4, 256, 256)
    with pytest.raises(ValueError, match="never silently swapped"):
        fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="fastest")


@requires_sm90
@pytest.mark.parametrize("backend", ["auto", "specialized"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 8192, 1536),   # captured qbproj class
        (2, 2048, 7168),   # captured qkva class
        (16, 7168, 2048),  # captured oproj class
        (24, 2624, 6144),  # captured glm qkva class
        (13, 2048, 2048),  # interior (band-interpolated) point
    ],
)
def test_specialized_matches_reference(backend, shape):
    M, N, K = shape
    x, wq, ws = _make(M, N, K, seed=42 + M)
    ref = _reference(x, wq, ws)

    _reset_fp8bs_decodem_stats()
    out = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend=backend)
    st = _fp8bs_decodem_stats()
    assert st["backend_dispatches"] == {"specialized": 1, "stock": 0}, st
    assert out.shape == (M, N) and out.dtype == torch.bfloat16
    rel = _rel_fro(out, ref)
    assert rel <= 0.03, rel

    # bitwise determinism
    out2 = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend=backend)
    assert torch.equal(out.view(torch.int16), out2.view(torch.int16))


@requires_sm90
def test_explicit_stock_never_swapped_and_matches():
    x, wq, ws = _make(8, 8192, 1536, seed=7)
    ref = _reference(x, wq, ws)
    _reset_fp8bs_decodem_stats()
    out = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="stock")
    st = _fp8bs_decodem_stats()
    assert st["backend_dispatches"] == {"specialized": 0, "stock": 1}, st
    assert st["fallback_reasons"] == {}, st
    assert _rel_fro(out, ref) <= 0.03


@requires_sm90
def test_kill_switch_forces_stock(monkeypatch):
    x, wq, ws = _make(4, 2048, 2048, seed=3)
    out_spec = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="specialized")
    monkeypatch.setenv(SPECIALIZED_KERNEL_DISABLE_ENV, "1")
    ok, reason = _eligibility(x, wq, None, ws, out_spec)
    assert not ok and reason == "kill_switch"
    _reset_fp8bs_decodem_stats()
    out = fp8_blockscale_gemm_sm90(x, wq, None, ws)  # auto
    st = _fp8bs_decodem_stats()
    assert st["backend_dispatches"] == {"specialized": 0, "stock": 1}, st
    assert st["fallback_reasons"] == {"auto: kill_switch": 1}, st
    monkeypatch.delenv(SPECIALIZED_KERNEL_DISABLE_ENV)
    # both backends agree against the fp32 reference
    ref = _reference(x, wq, ws)
    assert _rel_fro(out, ref) <= 0.03 and _rel_fro(out_spec, ref) <= 0.03


@requires_sm90
def test_off_coverage_specialized_falls_back_recorded():
    # above the measured decode-M band
    x, wq, ws = _make(32, 2048, 2048, seed=5)
    _reset_fp8bs_decodem_stats()
    out = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="specialized")
    st = _fp8bs_decodem_stats()
    assert st["backend_dispatches"] == {"specialized": 0, "stock": 1}, st
    assert st["fallback_reasons"] == {"specialized: m_band": 1}, st
    assert _rel_fro(out, _reference(x, wq, ws)) <= 0.03

    # per-token (1x128) weight-scale granularity stays on the stock route
    x2, wq2, _ = _make(4, 2048, 2048, seed=6)
    ws_pt = torch.full((2048, 2048 // G), 0.125, device="cuda",
                       dtype=torch.float32)
    _reset_fp8bs_decodem_stats()
    fp8_blockscale_gemm_sm90(x2, wq2, None, ws_pt, backend="specialized")
    st = _fp8bs_decodem_stats()
    assert st["backend_dispatches"]["specialized"] == 0, st
    assert st["fallback_reasons"] == {
        "specialized: weight_scale_granularity": 1}, st


@requires_sm90
def test_auto_equals_specialized_bitwise():
    x, wq, ws = _make(12, 2624, 6144, seed=9)
    a = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="auto")
    b = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="specialized")
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@requires_sm90
def test_graph_capture_replays_bitwise_equal_to_eager():
    M, N, K = 8, 8192, 1536
    x, wq, ws = _make(M, N, K, seed=11)
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    # warm (JIT + route) before capture, as documented
    fp8_blockscale_gemm_sm90(x, wq, None, ws, out=out, backend="specialized")
    torch.cuda.synchronize()
    eager = out.clone()

    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fp8_blockscale_gemm_sm90(x, wq, None, ws, out=out,
                                 backend="specialized")
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(g):
        fp8_blockscale_gemm_sm90(x, wq, None, ws, out=out,
                                 backend="specialized")
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
