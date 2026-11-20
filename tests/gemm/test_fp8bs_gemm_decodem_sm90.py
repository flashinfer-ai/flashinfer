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


def _reference(x, wq, ws):
    """FP32 reference: quantize with 448/amax (1 for an all-zero group),
    then dequantize using the reciprocal multiplier before the dot product."""
    M, K = x.shape
    N = wq.shape[0]
    xf = x.float().view(M, K // G, G)
    amax = xf.abs().amax(dim=-1, keepdim=True)
    inv = torch.where(amax != 0, 448.0 / amax, torch.ones_like(amax))
    xq = (xf * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    xdq = (xq * (1.0 / inv)).view(M, K)
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
    assert st["backend_dispatches"] == {"specialized": 1, "deepgemm": 0}, st
    assert out.shape == (M, N) and out.dtype == torch.bfloat16
    rel = _rel_fro(out, ref)
    assert rel <= 0.03, rel

    # bitwise determinism
    out2 = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend=backend)
    assert torch.equal(out.view(torch.int16), out2.view(torch.int16))


@requires_sm90
@pytest.mark.parametrize("backend", ["auto", "specialized"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 8192, 1536),
        (4, 2048, 2048),
        (12, 2624, 6144),
        (24, 6144, 2048),
    ],
)
def test_splitk_cancellation_exact(backend, shape):
    """Reviewer repro (exactly representable): BF16 activations all ones;
    each FP8 weight row is +1 for the first K/2 entries and -1 for the rest,
    with its first entry 2; unit scales.  The exact result is 1 everywhere.
    BF16-packed split-K partials cancel to 0 instead; FP32 partials must
    reproduce the stock result exactly."""
    M, N, K = shape
    x = torch.ones((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.ones((N, K), device="cuda", dtype=torch.float32)
    w[:, K // 2 :] = -1.0
    w[:, 0] = 2.0
    wq = w.to(torch.float8_e4m3fn)
    ws = torch.ones(((N + G - 1) // G, K // G), device="cuda", dtype=torch.float32)
    out = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend=backend)
    assert torch.equal(out, torch.ones_like(out)), (
        float(out.min()),
        float(out.max()),
    )
    stock = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="deepgemm")
    assert torch.equal(out.view(torch.int16), stock.view(torch.int16))


@requires_sm90
@pytest.mark.parametrize("backend", ["auto", "specialized"])
@pytest.mark.parametrize("exp", [-20, -34])
def test_small_amax_matches_stock_quantization(backend, exp):
    """Reviewer repro: tiny uniform activations with a compensating weight
    scale on a non-split route.  The exact result is 128; the pre-fix
    epsilon-clamped scale returned 135 (A=2**-20) or 0 (A=2**-34).

    The specialized kernel's scale arithmetic on this probe is exact powers
    of two, so it returns exactly 128 under every toolkit.  The incumbent's
    own last ~2% here is toolkit-dependent (13.x-built incumbents and the
    review's native run give exactly 128; a 12.9-built incumbent returns a
    uniform 125.5 at A=2**-34), so the incumbent comparison is bounded, with
    a bound that still fails the pre-fix clamp (135 -> +5.5%, 0 -> -100%)."""
    M, N, K = 4, 64, 128
    x = torch.full((M, K), 2.0**exp, device="cuda", dtype=torch.bfloat16)
    wq = torch.ones((N, K), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    ws = torch.full((1, 1), 2.0**-exp, device="cuda", dtype=torch.float32)
    out = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend=backend)
    assert torch.equal(out, torch.full_like(out, 128.0)), (
        float(out.min()),
        float(out.max()),
    )
    stock = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="deepgemm")
    rel = float(
        ((out.float() - stock.float()).abs() / stock.float().abs().clamp_min(1e-12))
        .max()
        .item()
    )
    assert rel <= 0.03, (rel, float(stock.min()), float(stock.max()))


@requires_sm90
@pytest.mark.parametrize("backend", ["auto", "specialized"])
def test_normal_amax_below_replacement_floor(backend, record_property):
    # Both FP32 scale factors are normal. With the floor the maximum maps
    # near 420 and rounds to E4M3 416, rather than the unclamped 448.
    x = torch.full((4, G), (15 / 16) * 2.0**-117, device="cuda", dtype=torch.bfloat16)
    wq = torch.ones((64, G), device="cuda").to(torch.float8_e4m3fn)
    ws = torch.full((1, 1), 2.0**117, device="cuda", dtype=torch.float32)
    _reset_fp8bs_decodem_stats()
    out = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend=backend)
    assert _fp8bs_decodem_stats()["backend_dispatches"] == {
        "specialized": 1,
        "deepgemm": 0,
    }
    assert torch.equal(out, torch.full_like(out, 120))  # 128 * 15/16
    # The external H100/cu129 stock build returned zero on this extreme
    # input; the H100/cu132 build returned 120. Keep the exact analytic
    # oracle above, and record stock separately instead of treating it as
    # ground truth. Other ordinary-range stock comparisons are unchanged.
    stock = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="deepgemm")
    record_property("stock_min", stock.min().item())
    record_property("stock_max", stock.max().item())
    record_property("analytic_expected", 120)


@requires_sm90
def test_zero_amax_group_matches_stock_identity_scale():
    """An all-zero 1x128 activation group contributes exactly 0, as it does
    on the stock route (quantized zeros); nonzero groups in the same row are
    unaffected and match stock bitwise."""
    M, N, K = 4, 128, 256
    x = torch.zeros((M, K), device="cuda", dtype=torch.bfloat16)
    x[:, G:] = 1.0  # group 0 all-zero, group 1 all-ones
    wq = torch.ones((N, K), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    ws = torch.ones((1, 2), device="cuda", dtype=torch.float32)
    out = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="specialized")
    assert torch.equal(out, torch.full_like(out, 128.0))
    stock = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="deepgemm")
    assert torch.equal(out.view(torch.int16), stock.view(torch.int16))


@requires_sm90
@pytest.mark.parametrize(
    "shape",
    [
        (24, 4096, 128),  # single group, no next prefetch
        (24, 3136, 1536),  # half-width last HMMA tile
        (16, 3136, 24576),  # long enough to promote across a chunk boundary
    ],
)
def test_hmma_prefetch_and_tail(shape):
    x, wq, ws = _make(*shape, seed=31)
    ref = _reference(x, wq, ws)
    out = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="specialized")
    assert _rel_fro(out, ref) <= 0.03
    if shape[1] % 128:
        assert _rel_fro(out[:, -64:], ref[:, -64:]) <= 0.03


@requires_sm90
def test_explicit_stock_never_swapped_and_matches():
    x, wq, ws = _make(8, 8192, 1536, seed=7)
    ref = _reference(x, wq, ws)
    _reset_fp8bs_decodem_stats()
    out = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="deepgemm")
    st = _fp8bs_decodem_stats()
    assert st["backend_dispatches"] == {"specialized": 0, "deepgemm": 1}, st
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
    assert st["backend_dispatches"] == {"specialized": 0, "deepgemm": 1}, st
    assert st["fallback_reasons"] == {"auto: kill_switch": 1}, st
    # explicit "specialized" raises under the kill switch (never falls back)
    with pytest.raises(NotImplementedError, match="kill_switch"):
        fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="specialized")
    monkeypatch.delenv(SPECIALIZED_KERNEL_DISABLE_ENV)
    # both backends agree against the fp32 reference
    ref = _reference(x, wq, ws)
    assert _rel_fro(out, ref) <= 0.03 and _rel_fro(out_spec, ref) <= 0.03


@requires_sm90
def test_off_coverage_explicit_raises_auto_takes_stock():
    # above the measured decode-M band: explicit "specialized" raises with
    # the reason (never falls back); auto keeps the stock route
    x, wq, ws = _make(32, 2048, 2048, seed=5)
    _reset_fp8bs_decodem_stats()
    with pytest.raises(NotImplementedError, match="m_band"):
        fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="specialized")
    st = _fp8bs_decodem_stats()
    assert st["backend_dispatches"]["specialized"] == 0, st
    assert st["fallback_reasons"] == {"specialized: m_band": 1}, st
    _reset_fp8bs_decodem_stats()
    out = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="auto")
    st = _fp8bs_decodem_stats()
    assert st["backend_dispatches"] == {"specialized": 0, "deepgemm": 1}, st
    assert st["fallback_reasons"] == {"auto: m_band": 1}, st
    assert _rel_fro(out, _reference(x, wq, ws)) <= 0.03

    # per-token (1x128) weight-scale granularity: explicit raises too
    x2, wq2, _ = _make(4, 2048, 2048, seed=6)
    ws_pt = torch.full((2048, 2048 // G), 0.125, device="cuda", dtype=torch.float32)
    _reset_fp8bs_decodem_stats()
    with pytest.raises(NotImplementedError, match="weight_scale_granularity"):
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
def test_overlapping_output_falls_back():
    # the kernel reads activations and weights while CTAs write `out`; an
    # `out` that aliases an input is refused (auto serves via the stock
    # route, whose aliasing behavior is its own contract)
    M, N, K = 4, 2048, 2048
    x, wq, ws = _make(M, N, K, seed=23)
    ok, reason = _eligibility(x, wq, None, ws, x)
    assert not ok and reason == "overlap"
    with pytest.raises(NotImplementedError, match="overlap"):
        fp8_blockscale_gemm_sm90(x, wq, None, ws, out=x, backend="specialized")
    # a distinct output buffer with the same shapes stays eligible
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
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
def test_n_zero_stays_on_stock_contract():
    x, wq, ws = _make(4, 0, 128)
    _reset_fp8bs_decodem_stats()
    with pytest.raises(NotImplementedError, match="n_zero"):
        fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="specialized")
    for backend in ("auto", "deepgemm"):
        with pytest.raises(RuntimeError, match=r"(?i)weight.*null"):
            fp8_blockscale_gemm_sm90(x, wq, None, ws, backend=backend)
    st = _fp8bs_decodem_stats()
    assert st["backend_dispatches"] == {"specialized": 0, "deepgemm": 2}
    assert st["fallback_reasons"] == {"specialized: n_zero": 1, "auto: n_zero": 1}


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
    # auto routes around the refusal onto the stock path, which rejects
    # K == 0 itself; explicit "deepgemm" goes there directly
    for backend in ("auto", "deepgemm"):
        with pytest.raises(RuntimeError, match="input is null"):
            fp8_blockscale_gemm_sm90(x, wq, None, ws, backend=backend)
    # explicit "specialized" never falls back: it raises with the
    # eligibility reason instead of reaching the stock route
    with pytest.raises(NotImplementedError, match="k_zero"):
        fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="specialized")


@requires_sm90
def test_jit_build_failure_falls_back_to_stock(monkeypatch):
    import flashinfer.gemm.fp8bs_decodem_sm90 as mod

    x, wq, ws = _make(4, 2048, 2048, seed=13)
    ref = _reference(x, wq, ws)

    calls = {"n": 0}

    def ninja_boom():
        calls["n"] += 1
        raise RuntimeError("Ninja build failed.")

    monkeypatch.setitem(mod._stats, "jit_build", "not attempted")
    monkeypatch.setattr(mod, "_get_module", ninja_boom)
    _reset_fp8bs_decodem_stats()
    # explicit "specialized" raises on the build failure (never falls back)
    with pytest.raises(NotImplementedError, match="JIT build"):
        fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="specialized")
    st = _fp8bs_decodem_stats()
    assert st["fallback_reasons"] == {"specialized: jit_build_failure": 1}, st
    # auto serves the same call on the stock route
    _reset_fp8bs_decodem_stats()
    out = fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="auto")
    st = _fp8bs_decodem_stats()
    assert st["backend_dispatches"] == {"specialized": 0, "deepgemm": 1}, st
    assert st["fallback_reasons"] == {"auto: jit_build_failure": 1}, st
    assert _rel_fro(out, ref) <= 0.03
    # the failure is cached: no rebuild on later calls, explicit calls keep
    # raising with the recorded failure instead of re-running the compiler
    assert calls["n"] == 1, calls
    with pytest.raises(NotImplementedError, match="previously failed"):
        fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="specialized")
    assert calls["n"] == 1, calls

    # the capture-without-warmup error keeps surfacing instead of falling back
    def capture_boom():
        raise GraphCaptureJitError("not built yet under capture")

    monkeypatch.setitem(mod._stats, "jit_build", "not attempted")
    monkeypatch.setattr(mod, "_get_module", capture_boom)
    with pytest.raises(GraphCaptureJitError):
        fp8_blockscale_gemm_sm90(x, wq, None, ws, backend="specialized")


@requires_sm90
@pytest.mark.parametrize(
    "concurrent", [False, True], ids=["auto-first", "two-cold-callers"]
)
def test_cold_auto_build_failure(monkeypatch, caplog, concurrent):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from unittest.mock import Mock

    import flashinfer.gemm.fp8bs_decodem_sm90 as mod

    x, wq, ws = _make(4, 128, 256, seed=37)
    ref = _reference(x, wq, ws)
    torch.cuda.synchronize(x.device)
    build = Mock(side_effect=RuntimeError("injected cold build failure"))
    monkeypatch.setattr(mod, "_get_module", build)
    monkeypatch.setitem(mod._stats, "jit_build", "not attempted")
    _reset_fp8bs_decodem_stats()

    def call(backend):
        return fp8_blockscale_gemm_sm90(x, wq, None, ws, backend=backend)

    if concurrent:
        # Put both callers at lock entry, past the *old* failure check.
        gate = Barrier(2, timeout=30)
        lock = mod._jit_lock

        class ColdLock:
            def __enter__(self):
                gate.wait()
                lock.acquire()

            def __exit__(self, *exc):
                lock.release()

        with monkeypatch.context() as ctx, ThreadPoolExecutor(2) as pool:
            ctx.setattr(mod, "_jit_lock", ColdLock())
            auto = pool.submit(call, "auto")
            explicit = pool.submit(call, "specialized")
            out = auto.result()
            with pytest.raises(NotImplementedError, match="JIT build"):
                explicit.result()
    else:
        out = call("auto")

    assert _rel_fro(out, ref) <= 0.03
    st = _fp8bs_decodem_stats()
    assert st["backend_dispatches"] == {"specialized": 0, "deepgemm": 1}
    reasons = {"auto: jit_build_failure": 1}
    if concurrent:
        reasons["specialized: jit_build_failure"] = 1
    assert st["fallback_reasons"] == reasons
    with pytest.raises(NotImplementedError, match="previously failed"):
        call("specialized")
    build.assert_called_once_with()
    assert (
        sum(
            r.name == mod.logger.name
            and r.levelname == "WARNING"
            and "JIT build failed" in r.getMessage()
            for r in caplog.records
        )
        == 1
    )


@pytest.fixture(scope="module")
def native_rejection_module(tmp_path_factory):
    from uuid import uuid4

    from flashinfer.jit.gemm import fp8_blockscale as recipe

    directory = tmp_path_factory.mktemp("decodem_native_rejection")
    gen_spec = recipe.gen_jit_spec
    operations = {
        "fp8bs_gemm_decodem_sm90.cu": ("cudaLaunchKernelEx", 2),
        "fp8bs_gemm_decodem_sm90_mwg.cu": ("cudaFuncSetAttribute", 1),
    }

    def private_spec(name, sources, **kwargs):
        copies = []
        for src in sources:
            text = src.read_text()
            if src.name in operations:
                op, sites = operations[src.name]
                # Redirect only the CUDA calls, not the code consuming their
                # return values. Variadic forwarding preserves template commas.
                assert text.count(op + "(") == sites
                text = text.replace(op + "(", "FI_TEST_CUDA_CALL(")
                anchor = '#include "fp8bs_gemm_decodem_sm90.h"'
                assert text.count(anchor) == 1
                wrapper = (
                    "\n#include <cstdlib>\n"
                    "#define FI_TEST_CUDA_CALL(...) "
                    '(std::getenv("FLASHINFER_TEST_DECODEM_REJECT") != nullptr '
                    f"? cudaErrorNotSupported : {op}(__VA_ARGS__))\n"
                )
                text = text.replace(anchor, anchor + wrapper)
            copy = directory / src.name
            copy.write_text(text)
            copies.append(copy)
        # Let the real recipe supply all flags, libraries and include paths.
        return gen_spec(f"{name}_test_{uuid4().hex}", copies, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(recipe, "gen_jit_spec", private_spec)
        spec = recipe.gen_fp8bs_gemm_decodem_sm90_module()
    return spec.build_and_load()


@requires_sm90
@pytest.mark.parametrize("backend", ["auto", "specialized"])
@pytest.mark.parametrize(
    "shape,site",
    [
        ((4, 2048, 2048), "launch_ku"),
        ((2, 7168, 2048), "launch_splitk_load_balanced"),
        ((1, 8192, 1536), "mwg shared-memory opt-in"),
    ],
)
def test_native_rejection_and_recovery(
    monkeypatch, native_rejection_module, backend, shape, site
):
    import flashinfer.gemm.fp8bs_decodem_sm90 as mod

    monkeypatch.setattr(mod, "_get_module", lambda: native_rejection_module)
    monkeypatch.setitem(mod._stats, "jit_build", "built")
    monkeypatch.delenv(SPECIALIZED_KERNEL_DISABLE_ENV, raising=False)
    M, N, K = shape
    x, wq, ws = _make(M, N, K, seed=17)
    ref = _reference(x, wq, ws)
    poison = torch.full((M, N), -1234, device=x.device, dtype=x.dtype)
    out = poison.clone()

    def call():
        return fp8_blockscale_gemm_sm90(x, wq, None, ws, out=out, backend=backend)

    _reset_fp8bs_decodem_stats()
    monkeypatch.setenv("FLASHINFER_TEST_DECODEM_REJECT", "1")
    # The failing branch returns a constant, without submitting work or
    # setting CUDA's last-error slot. A post-launch check alone cannot pass.
    with pytest.raises(RuntimeError, match=rf"{site}: operation not supported"):
        call()
    st = _fp8bs_decodem_stats()
    assert st["backend_dispatches"] == {"specialized": 0, "deepgemm": 0}
    assert st["fallback_reasons"] == {}
    assert torch.equal(out, poison)

    monkeypatch.delenv("FLASHINFER_TEST_DECODEM_REJECT")
    call()
    assert _rel_fro(out, ref) <= 0.03
    st = _fp8bs_decodem_stats()
    assert st["backend_dispatches"] == {"specialized": 1, "deepgemm": 0}
    assert st["fallback_reasons"] == {}


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
