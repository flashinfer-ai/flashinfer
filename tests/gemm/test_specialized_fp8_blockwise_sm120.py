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
against a dequant+matmul reference (including a 128-unaligned N=64 case),
and CUDA-graph capture/replay.
"""

import pytest
import torch


def _is_sm120_exactly():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) == (12, 0)


def _has_cuda_13_toolkit():
    try:
        from flashinfer.jit.cpp_ext import get_cuda_version

        return get_cuda_version().major >= 13
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _is_sm120_exactly(),
        reason="the specialized fp8-blockwise GEMM backend is SM120-only",
    ),
    pytest.mark.skipif(
        not _has_cuda_13_toolkit(),
        reason="the specialized backend JIT-builds an sm_120a module "
        "(CUDA 13 toolkit required)",
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


@pytest.mark.parametrize(
    "m,n,k",
    [
        (4, 64, 2048),  # 128-unaligned N (partial scale block)
        (1, 1024, 2048),
        (16, 2048, 4096),
    ],
)
def test_auto_routes_to_specialized_on_allowlist(stock_sentinel, m, n, k):
    """``backend="auto"`` on an allowlisted shape dispatches the specialized
    kernels (stock proven not called) and matches the dequant reference."""
    from flashinfer.gemm import specialized_fp8_blockwise_sm120 as specmod

    a, b, a_scale, b_scale = _gen_inputs(m, n, k, seed=42)
    ref = _reference(a, b, a_scale, b_scale)
    before = specmod.dispatch_stats.get("specialized", 0)
    out = _call(a, b, a_scale, b_scale, backend="auto")
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
        out = _call(a, b, a_scale, b_scale, backend="specialized")
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
    a, b, a_scale, b_scale = _gen_inputs(4, 64, 2048, seed=47)
    with pytest.raises(stock_sentinel):
        _call(a, b, a_scale, b_scale, backend="auto")
    reasons = {reason for _, reason in specmod.fallback_reasons}
    assert "kill_switch" in reasons
    ref = _reference(a, b, a_scale, b_scale)
    out = _call(a, b, a_scale, b_scale, backend="specialized")
    torch.cuda.synchronize()
    assert _rel(out, ref) < REL_TOL


def test_cuda_graph_replay_matches_eager():
    """After ``warmup()`` an auto-routed allowlisted call is capturable and
    the replay reproduces the eager output bitwise."""
    from flashinfer.gemm import specialized_fp8_blockwise_sm120 as specmod

    specmod.warmup()
    a, b, a_scale, b_scale = _gen_inputs(8, 64, 2048, seed=48)
    out = torch.empty(8, 64, dtype=torch.bfloat16, device=DEVICE)
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
