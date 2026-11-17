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
    GraphCaptureJitError,
    _dispatch_table,
    _eligibility,
    _fp8bs_decodem_stats,
    _reset_fp8bs_decodem_stats,
)

requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="needs an SM90 device",
)

G = 128


def _make(M, N, K, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((M, K), generator=g, device=device, dtype=torch.float32).to(
        torch.bfloat16
    )
    wq = torch.randn((N, K), generator=g, device=device, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    pow2 = torch.tensor([0.0625, 0.125, 0.25], device=device)
    ws = (
        pow2[
            torch.randint(0, 3, ((N + G - 1) // G, K // G), generator=g, device=device)
        ]
        .float()
        .contiguous()
    )
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
        (1, 8192, 1536),  # captured qbproj class
        (2, 2048, 7168),  # captured qkva class
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
    ws_pt = torch.full((2048, 2048 // G), 0.125, device="cuda", dtype=torch.float32)
    _reset_fp8bs_decodem_stats()
    fp8_blockscale_gemm_sm90(x2, wq2, None, ws_pt, backend="specialized")
    st = _fp8bs_decodem_stats()
    assert st["backend_dispatches"]["specialized"] == 0, st
    assert st["fallback_reasons"] == {"specialized: weight_scale_granularity": 1}, st


@requires_sm90
def test_misaligned_offset_views_fall_back():
    # Contiguous storage-offset views break the kernel's 16-byte vector
    # loads, so eligibility must refuse them (the stock route faults on such
    # views too — `misaligned address` — so the full call is not exercised
    # here; the guard keeps the specialized kernel at behavior parity).
    M, N, K = 4, 2048, 2048
    x, wq, ws = _make(M, N, K, seed=21)
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

    xbuf = torch.empty(M * K + 8, device="cuda", dtype=torch.bfloat16)
    x_off = xbuf[1 : 1 + M * K].view(M, K)
    x_off.copy_(x)
    assert x_off.is_contiguous() and x_off.data_ptr() % 16 != 0
    ok, reason = _eligibility(x_off, wq, None, ws, out)
    assert not ok and reason == "alignment"

    wbuf = torch.empty(N * K + 16, device="cuda", dtype=torch.float8_e4m3fn)
    w_off = wbuf[1 : 1 + N * K].view(N, K)
    w_off.copy_(wq)
    assert w_off.is_contiguous() and w_off.data_ptr() % 16 != 0
    ok, reason = _eligibility(x, w_off, None, ws, out)
    assert not ok and reason == "alignment"

    # an aligned call with the same shapes stays eligible
    ok, reason = _eligibility(x, wq, None, ws, out)
    assert ok, reason


@requires_sm90
def test_cross_device_tensors_fall_back():
    x, wq, ws = _make(4, 2048, 2048, seed=15)
    out = torch.empty((4, 2048), device="cuda", dtype=torch.bfloat16)
    ok, reason = _eligibility(x, wq.cpu(), None, ws, out)
    assert not ok and reason == "device"
    ok, reason = _eligibility(x, wq, None, ws.cpu(), out)
    assert not ok and reason == "device"
    ok, reason = _eligibility(x, wq, None, ws, out.cpu())
    assert not ok and reason == "device"


@requires_sm90
def test_k_zero_stays_on_stock_contract():
    # the stock route rejects K == 0 (null input); the specialized backend
    # must not serve it either (several routes would leave `out` unwritten)
    M, N, K = 4, 2048, 0
    x = torch.zeros((M, K), device="cuda", dtype=torch.bfloat16)
    wq = torch.zeros((N, K), device="cuda", dtype=torch.float8_e4m3fn)
    ws = torch.zeros(((N + G - 1) // G, 0), device="cuda", dtype=torch.float32)
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    ok, reason = _eligibility(x, wq, None, ws, out)
    assert not ok and reason == "k_zero"
    for backend in ("auto", "specialized", "stock"):
        with pytest.raises(RuntimeError, match="input is null"):
            fp8_blockscale_gemm_sm90(x, wq, None, ws, backend=backend)


@requires_sm90
def test_jit_build_failure_falls_back_to_stock(monkeypatch):
    import flashinfer.gemm.fp8bs_decodem_sm90 as mod

    x, wq, ws = _make(4, 2048, 2048, seed=13)
    ref = _reference(x, wq, ws)

    def ninja_boom():
        raise RuntimeError("Ninja build failed.")

    monkeypatch.setitem(mod._stats, "jit_build", "not attempted")
    monkeypatch.setattr(mod, "_get_module", ninja_boom)
    _reset_fp8bs_decodem_stats()
    out = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="specialized")
    st = _fp8bs_decodem_stats()
    assert st["backend_dispatches"] == {"specialized": 0, "stock": 1}, st
    assert st["fallback_reasons"] == {"specialized: jit_build_failure": 1}, st
    assert _rel_fro(out, ref) <= 0.03

    # the capture-without-warmup error keeps surfacing instead of falling back
    def capture_boom():
        raise GraphCaptureJitError("not built yet under capture")

    monkeypatch.setattr(mod, "_get_module", capture_boom)
    with pytest.raises(GraphCaptureJitError):
        fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="specialized")


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
        fp8_blockscale_gemm_sm90(x, wq, None, ws, out=out, backend="specialized")
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(g):
        fp8_blockscale_gemm_sm90(x, wq, None, ws, out=out, backend="specialized")
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
