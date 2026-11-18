"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Tests for the "specialized" SM120 backend of ``gemm_fp8_nt_groupwise``
(agent-generated fp8 1x128/128x128 blockwise-scale decode GEMM kernels).

Covers the backend-dispatch semantics (auto routing on the shipped
allowlist, never-swap-explicit, composable fallback for explicit
``backend="specialized"`` off-allowlist, kill switch), numeric correctness
against a dequant+matmul reference for every allowlisted (N, K) class at the
dispatch-route M boundaries of combo_entry.cu (including the 128-unaligned
N=64 class) over several seeds and both scale granularities, with every
checked output NaN-poisoned before the call, and CUDA-graph capture/replay
with exact-shape warm-up keying.
"""

import pytest
import torch


def _is_sm120_exactly():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) == (12, 0)


def _has_min_toolkit():
    try:
        from flashinfer.jit.cpp_ext import get_cuda_version

        v = get_cuda_version()
        return (v.major, v.minor) >= (12, 9)
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _is_sm120_exactly(),
        reason="the specialized fp8-blockwise GEMM backend is SM120-only",
    ),
    pytest.mark.skipif(
        not _has_min_toolkit(),
        reason="the specialized backend JIT-builds an sm_120a module "
        "(CUDA >= 12.9 toolkit required, matching the dispatch guard)",
    ),
]

DEVICE = "cuda"
# L2-relative bound from the backend's measured accuracy class (every
# allowlisted shape verifies at <= 1e-2 against the fp32 dequant reference;
# many rows are bitwise-equal to the bf16 cast of it).
REL_TOL = 2e-2


class _StockReached(Exception):
    """Raised by the sentinel when the stock SM120 cutlass module is hit."""


class _StockSentinel:
    def __getattr__(self, _name):
        def _raise(*args, **kwargs):
            raise _StockReached("stock sm120 module invoked")

        return _raise


@pytest.fixture
def stock_sentinel(monkeypatch):
    """Replace the stock SM120 module getter so any dispatch that reaches
    the stock cutlass kernel raises ``_StockReached``."""
    import flashinfer.gemm.gemm_base as gemm_base

    monkeypatch.setattr(gemm_base, "get_gemm_sm120_module", lambda: _StockSentinel())
    return _StockReached


def _gen_inputs(m, n, k, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a = (torch.randn(m, k, generator=gen) * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator=gen) * 0.5).to(torch.float8_e4m3fn)
    a_scale = torch.rand(m, k // 128, generator=gen) * 0.5 + 0.5
    b_scale = torch.rand((n + 127) // 128, k // 128, generator=gen) * 0.5 + 0.5
    return (
        a.to(DEVICE),
        b.to(DEVICE),
        a_scale.float().contiguous().to(DEVICE),
        b_scale.float().contiguous().to(DEVICE),
    )


def _reference(a, b, a_scale, b_scale):
    """Dequantize both operands blockwise and matmul in fp32."""
    m, k = a.shape
    n = b.shape[0]
    kb = k // 128
    a32 = (a.float().view(m, kb, 128) * a_scale.view(m, kb, 1)).view(m, k)
    bs = b_scale.repeat_interleave(128, dim=0)[:n]
    bs = bs.repeat_interleave(128, dim=1)[:, :k]
    b32 = b.float() * bs
    return (a32 @ b32.T).to(torch.bfloat16)


def _rel(out, ref):
    return (
        (out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12)
    ).item()


def _call(a, b, a_scale, b_scale, backend, out=None):
    from flashinfer.gemm import gemm_fp8_nt_groupwise

    return gemm_fp8_nt_groupwise(
        a,
        b,
        a_scale,
        b_scale,
        scale_major_mode="K",
        out=out,
        out_dtype=torch.bfloat16,
        backend=backend,
    )


# Every allowlisted (N, K) class, exercised at the dispatch-route boundaries
# of combo_entry.cu (each class picks a different kernel per M range, and some
# routes have exact-M forms, e.g. the N=64 cluster launches at M=1/16/32 and
# the N=1024 switch at M<=2), with M capped at the class's admitted max.
# Loaded from the SHIPPED allowlist (the same file the dispatcher reads), so
# the parametrization cannot drift from what auto-routing actually admits.
def _shipped_allowlist():
    from flashinfer.gemm.specialized_fp8_blockwise_sm120 import _allowlist

    return _allowlist()


ALLOWLIST_NK_MMAX = _shipped_allowlist()
ROUTE_BOUNDARY_MS = (1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 32)
ROUTE_CASES = [
    (m, n, k)
    for (n, k), m_max in sorted(ALLOWLIST_NK_MMAX.items())
    # always include the class's own admitted max, so a widened allowlist
    # entry is exercised at its new boundary even before this tuple is told
    for m in sorted(set(ROUTE_BOUNDARY_MS) | {m_max})
    if m <= m_max
]


def _nan_out(m, n):
    """NaN-poisoned output buffer: every correctness check writes into a
    buffer pre-filled with NaN, so an element the kernel never writes (or a
    wrong-route dispatch that writes a subset) cannot pass the comparison."""
    return torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=DEVICE)


@pytest.mark.parametrize("m,n,k", ROUTE_CASES)
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_auto_routes_to_specialized_on_allowlist(stock_sentinel, m, n, k, seed):
    """``backend="auto"`` on every allowlisted (N, K) class at every route
    boundary dispatches the specialized kernels (stock proven not called)
    and matches the dequant reference into a NaN-poisoned output."""
    from flashinfer.gemm import specialized_fp8_blockwise_sm120 as specmod

    a, b, a_scale, b_scale = _gen_inputs(m, n, k, seed=seed * 1000 + m)
    ref = _reference(a, b, a_scale, b_scale)
    before = specmod.dispatch_stats.get("specialized", 0)
    out = _call(a, b, a_scale, b_scale, backend="auto", out=_nan_out(m, n))
    torch.cuda.synchronize()
    assert specmod.dispatch_stats.get("specialized", 0) == before + 1
    assert _rel(out, ref) < REL_TOL


def test_auto_off_allowlist_uses_stock(stock_sentinel):
    """An off-allowlist shape (M above the admitted cap for its (N, K)
    class) keeps the stock path under ``backend="auto"``."""
    a, b, a_scale, b_scale = _gen_inputs(32, 9216, 2048, seed=43)
    with pytest.raises(stock_sentinel):
        _call(a, b, a_scale, b_scale, backend="auto")


def test_explicit_backend_is_never_swapped(stock_sentinel):
    """An explicitly selected other backend must reach the stock kernel
    even on an allowlisted shape."""
    a, b, a_scale, b_scale = _gen_inputs(4, 1024, 2048, seed=44)
    with pytest.raises(stock_sentinel):
        _call(a, b, a_scale, b_scale, backend="cutlass")


def test_explicit_specialized_composable_fallback(stock_sentinel):
    """Explicit ``backend="specialized"`` off-allowlist falls back to the
    composable torch path (correct output, never another fused backend) —
    including the 128-unaligned N=64 class."""
    from flashinfer.gemm import specialized_fp8_blockwise_sm120 as specmod

    for m, n, k, seed in ((33, 64, 2048, 45), (64, 9216, 2048, 46)):
        a, b, a_scale, b_scale = _gen_inputs(m, n, k, seed=seed)
        ref = _reference(a, b, a_scale, b_scale)
        before = specmod.dispatch_stats.get("specialized_composable_fallback", 0)
        out = _call(a, b, a_scale, b_scale, backend="specialized", out=_nan_out(m, n))
        torch.cuda.synchronize()
        assert (
            specmod.dispatch_stats.get("specialized_composable_fallback", 0)
            == before + 1
        )
        assert _rel(out, ref) < REL_TOL


def test_kill_switch_disables_backend(stock_sentinel, monkeypatch):
    """``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1``: auto goes stock on an
    allowlisted shape; explicit ``specialized`` serves the call through the
    composable path (still never another fused backend)."""
    from flashinfer.gemm import specialized_fp8_blockwise_sm120 as specmod

    monkeypatch.setenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1")
    # Stock-routing assertion on an aligned-N class: the stock cutlass
    # groupwise checker rejects 128-unaligned N on SM120 (#9555), so an N=64
    # auto call with the kill switch set raises BackendSupportedError before
    # it could reach the sentinel.
    a, b, a_scale, b_scale = _gen_inputs(4, 1024, 2048, seed=47)
    with pytest.raises(stock_sentinel):
        _call(a, b, a_scale, b_scale, backend="auto")
    reasons = {reason for _, reason in specmod.fallback_reasons}
    assert "kill_switch" in reasons
    # Explicit ``specialized`` under the kill switch serves through the
    # composable path, including the 128-unaligned N=64 class.
    a, b, a_scale, b_scale = _gen_inputs(4, 64, 2048, seed=48)
    ref = _reference(a, b, a_scale, b_scale)
    out = _call(a, b, a_scale, b_scale, backend="specialized", out=_nan_out(4, 64))
    torch.cuda.synchronize()
    assert _rel(out, ref) < REL_TOL


@pytest.mark.parametrize("granularity", ((1, 128, 128), (128, 128, 128)))
def test_explicit_specialized_both_scale_granularities(stock_sentinel, granularity):
    """Numerics at both scale granularities of the op: (1, 128, 128) is the
    kernel contract; (128, 128, 128) is off-contract for the kernel and must
    be served by the composable fallback with correct numerics (never
    another fused backend)."""
    from flashinfer.gemm import gemm_fp8_nt_groupwise

    gm, gn, gk = granularity
    m, n, k = 8, 1024, 2048
    gen = torch.Generator(device="cpu").manual_seed(7)
    a = (torch.randn(m, k, generator=gen) * 0.5).to(torch.float8_e4m3fn).to(DEVICE)
    b = (torch.randn(n, k, generator=gen) * 0.5).to(torch.float8_e4m3fn).to(DEVICE)
    a_scale = (
        (torch.rand((m + gm - 1) // gm, k // gk, generator=gen) * 0.5 + 0.5)
        .float()
        .to(DEVICE)
    )
    b_scale = (
        (torch.rand((n + gn - 1) // gn, k // gk, generator=gen) * 0.5 + 0.5)
        .float()
        .to(DEVICE)
    )
    a32 = (
        a.float() * a_scale.repeat_interleave(gm, 0)[:m].repeat_interleave(gk, 1)[:, :k]
    )
    b32 = (
        b.float() * b_scale.repeat_interleave(gn, 0)[:n].repeat_interleave(gk, 1)[:, :k]
    )
    ref = (a32 @ b32.T).to(torch.bfloat16)
    out = gemm_fp8_nt_groupwise(
        a,
        b,
        a_scale,
        b_scale,
        scale_major_mode="K",
        scale_granularity_mnk=granularity,
        out=_nan_out(m, n),
        out_dtype=torch.bfloat16,
        backend="specialized",
    )
    torch.cuda.synchronize()
    assert _rel(out, ref) < REL_TOL


def test_capture_guard_is_keyed_by_exact_m(monkeypatch):
    """An eager launch of one M must not admit a different M of the same
    (N, K) class under capture: the dispatcher picks a different kernel per
    M range, so warm-up state is keyed by the exact dispatched shape."""
    from flashinfer.gemm import specialized_fp8_blockwise_sm120 as specmod

    snapshot = set(specmod._warmed)
    specmod._warmed.clear()
    try:
        a, b, a_scale, b_scale = _gen_inputs(1, 64, 2048, seed=49)
        out = torch.full((1, 64), float("nan"), dtype=torch.bfloat16, device=DEVICE)
        _call(a, b, a_scale, b_scale, backend="specialized", out=out)  # warms M=1
        torch.cuda.synchronize()
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        a4, b4, s4, t4 = _gen_inputs(4, 64, 2048, seed=50)
        assert not specmod.is_specialized_supported(a4, b4, s4, t4, "K", (1, 128, 128))
        assert specmod.fallback_reasons[-1] == ("specialized", "capture:unwarmed")
        # ... while the M actually launched stays admissible under capture.
        assert specmod.is_specialized_supported(
            a, b, a_scale, b_scale, "K", (1, 128, 128)
        )
    finally:
        specmod._warmed |= snapshot


def test_cuda_graph_replay_matches_eager():
    """After ``warmup()`` an auto-routed allowlisted call is capturable and
    the replay reproduces the eager output bitwise."""
    from flashinfer.gemm import specialized_fp8_blockwise_sm120 as specmod

    specmod.warmup()
    a, b, a_scale, b_scale = _gen_inputs(8, 64, 2048, seed=48)
    out = _nan_out(8, 64)
    eager = _call(a, b, a_scale, b_scale, backend="auto", out=out).clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call(a, b, a_scale, b_scale, backend="auto", out=out)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        _call(a, b, a_scale, b_scale, backend="auto", out=out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
