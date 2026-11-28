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

Tests for the generic full-op mHC PRE entry (mhc_pre) and its SM103
"specialized" backend: correctness vs an fp32 reference, parity with the
big-fuse composition, backend-dispatch semantics (auto / explicit / kill
switch / allowlist boundary), and back-to-back determinism (the kernels use
programmatic dependent launch, so overlap with a predecessor grid is the
racy regime).
"""

import os

import pytest
import torch

import flashinfer.mhc_pre_glm_cuda as spec_mod
from flashinfer.mhc import mhc_pre

requires_sm103 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the specialized GLM mHC PRE kernel is SM103-only",
)

HC, H, NF, EPS, SINK = 4, 4096, 24, 1e-6, 20

# The audited tolerance contract the kernel was accepted under (per-tensor:
# |err| <= atol + rtol * |ref| for >= 99.5% of elements, max |err| <= cap).
ATOL, RTOL, MATCHED, CAP = 0.004, 0.01, 0.995, 0.05


def _make_inputs(T, seed=0):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    residual = (
        torch.randn((T, HC, H), generator=g, device=dev, dtype=torch.float32) * 0.5
    ).to(torch.bfloat16)
    fn = torch.randn((NF, HC * H), generator=g, device=dev, dtype=torch.float32) * 0.02
    hc_scale = torch.randn((3,), generator=g, device=dev, dtype=torch.float32) * 0.1
    hc_base = torch.randn((NF,), generator=g, device=dev, dtype=torch.float32) * 0.1
    return residual, fn, hc_scale, hc_base


def _reference(residual, fn, hc_scale, hc_base):
    """fp32 reference: verbatim serve math (vLLM mhc_pre_torch)."""
    T, hc, Hd = residual.shape
    x = residual.reshape(T, hc * Hd).to(torch.float32)
    mixes = x @ fn.t()
    sqrsum = x.square().sum(-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc * Hd) + EPS)
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + EPS
    post = torch.sigmoid(mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = mixes[:, 2 * hc :].reshape(T, hc, hc) * hc_scale[2] + hc_base[
        2 * hc :
    ].reshape(1, hc, hc)
    comb = torch.softmax(comb, -1) + EPS
    comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    for _ in range(SINK - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(1).to(torch.bfloat16)
    return post.reshape(T, hc, 1), comb, layer_input


def _assert_tol(got, want):
    for g, w in zip(got, want, strict=True):
        gf, wf = g.float(), w.float()
        assert torch.isfinite(gf).all()
        err = (gf - wf).abs()
        lim = ATOL + RTOL * wf.abs()
        frac = float((err <= lim).float().mean())
        mx = float(err.max())
        assert frac >= MATCHED, f"matched ratio {frac} < {MATCHED}"
        assert mx <= CAP, f"max error {mx} > {CAP}"


@requires_sm103
@pytest.mark.parametrize("T", [1, 3, 37, 64, 128, 256, 333, 512, 1024, 2048, 4096])
def test_specialized_matches_reference(T):
    inputs = _make_inputs(T, seed=T)
    out = mhc_pre(*inputs, backend="specialized")
    assert spec_mod.LAST_FALLBACK_REASONS == []
    ref = _reference(*inputs)
    assert out[0].shape == (T, HC, 1) and out[0].dtype == torch.float32
    assert out[1].shape == (T, HC, HC) and out[1].dtype == torch.float32
    assert out[2].shape == (T, H) and out[2].dtype == torch.bfloat16
    _assert_tol(out, ref)


@requires_sm103
@pytest.mark.parametrize("T", [37, 256, 2048])
def test_specialized_matches_big_fuse(T):
    inputs = _make_inputs(T, seed=100 + T)
    spec = mhc_pre(*inputs, backend="specialized")
    fused = mhc_pre(*inputs, backend="big-fuse")
    for s, f in zip(spec, fused, strict=True):
        assert s.shape == f.shape and s.dtype == f.dtype
    # Both must satisfy the audited contract vs the fp32 reference; they are
    # not required to be bitwise equal to each other.
    ref = _reference(*inputs)
    _assert_tol(spec, ref)
    _assert_tol(fused, ref)


@requires_sm103
@pytest.mark.parametrize("backend", ["big-fuse", "auto"])
def test_big_fuse_projection_unaffected_by_autocast(backend):
    """The big-fuse projection must stay fp32 under torch.autocast: the
    downstream kernel requires fp32 dot_mix, and results must match the
    non-autocast call bitwise. For "auto", a non-default rms_eps forces the
    big-fuse fallback path so autocast is exercised on the fallback too."""
    inputs = _make_inputs(3, seed=7)
    kwargs = {} if backend == "big-fuse" else {"rms_eps": 2e-6}
    ref_out = mhc_pre(*inputs, backend=backend, **kwargs)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ac_out = mhc_pre(*inputs, backend=backend, **kwargs)
    for a, r in zip(ac_out, ref_out, strict=True):
        assert a.dtype == r.dtype
        assert torch.equal(a, r)


@requires_sm103
@pytest.mark.parametrize("T", [1, 37, 64, 128, 256, 512, 1024, 2048])
def test_back_to_back_determinism(T):
    """Three back-to-back pairs on the same inputs must be bitwise equal.
    T=256/512 exercise the programmatic-dependent-launch (overlapped) arms,
    where a missing fence would show up as run-to-run divergence; T<=64 and
    T=128 exercise the cluster arms whose owner-push DSMEM partials and
    staging-slab reuse are published with release arrivals."""
    inputs = _make_inputs(T, seed=7)
    runs = [mhc_pre(*inputs, backend="specialized") for _ in range(6)]
    torch.cuda.synchronize()
    first = runs[0]
    for later in runs[1:]:
        for a, b in zip(first, later, strict=True):
            assert torch.equal(a, b)


@requires_sm103
@pytest.mark.parametrize("T", [1, 37, 64, 128, 1024, 2048])
def test_graph_replay_determinism(T):
    """Capture one specialized call in a CUDA graph and replay it 20 times,
    checking every replay bitwise against the eager result on the same
    inputs. T<=64 and T=128 are the cluster arms whose DSMEM publication
    uses release arrivals; graph replay keeps launch-to-launch timing tight,
    the adversarial regime for a publication race."""
    inputs = _make_inputs(T, seed=11)
    eager = mhc_pre(*inputs, backend="specialized")  # eager build before capture
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        mhc_pre(*inputs, backend="specialized")  # warm the capture stream
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mhc_pre(*inputs, backend="specialized")
    for _ in range(20):
        graph.replay()
        torch.cuda.synchronize()
        for a, b in zip(eager, captured, strict=True):
            assert torch.equal(a, b)


@requires_sm103
def test_auto_routes_specialized_on_allowlist():
    inputs = _make_inputs(64, seed=1)
    before = dict(spec_mod.DISPATCH_STATS)
    out = mhc_pre(*inputs, backend="auto")
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1
    assert spec_mod.LAST_FALLBACK_REASONS == []
    _assert_tol(out, _reference(*inputs))


@requires_sm103
def test_auto_falls_back_off_allowlist():
    # Non-default sinkhorn_repeat is off the specialized allowlist: "auto"
    # must serve it on the big-fuse path, with the reason recorded.
    inputs = _make_inputs(16, seed=2)
    before = dict(spec_mod.DISPATCH_STATS)
    out = mhc_pre(*inputs, sinkhorn_repeat=5, backend="auto")
    assert spec_mod.DISPATCH_STATS["fallback"] == before["fallback"] + 1
    assert "sinkhorn_repeat_pinned" in spec_mod.LAST_FALLBACK_REASONS
    assert out[0].shape == (16, HC, 1)


@requires_sm103
def test_explicit_specialized_never_swapped():
    inputs = _make_inputs(16, seed=3)
    with pytest.raises(ValueError, match="never"):
        mhc_pre(*inputs, sinkhorn_repeat=5, backend="specialized")


@requires_sm103
def test_kill_switch_disables_specialized():
    inputs = _make_inputs(16, seed=4)
    os.environ[spec_mod.KILL_SWITCH_ENV] = "1"
    try:
        out = mhc_pre(*inputs, backend="auto")
        assert spec_mod.LAST_FALLBACK_REASONS == [
            "kill_switch:" + spec_mod.KILL_SWITCH_ENV + "=1"
        ]
        with pytest.raises(ValueError, match="never"):
            mhc_pre(*inputs, backend="specialized")
    finally:
        os.environ.pop(spec_mod.KILL_SWITCH_ENV, None)
    _assert_tol(out, _reference(*inputs))


@requires_sm103
def test_big_fuse_explicit_never_touches_specialized():
    inputs = _make_inputs(16, seed=5)
    before = dict(spec_mod.DISPATCH_STATS)
    mhc_pre(*inputs, backend="big-fuse")
    assert before == spec_mod.DISPATCH_STATS


def test_backend_validation():
    residual = torch.zeros(2, HC, H, dtype=torch.bfloat16)
    fn = torch.zeros(NF, HC * H)
    with pytest.raises(ValueError, match="backend"):
        mhc_pre(residual, fn, torch.zeros(3), torch.zeros(NF), backend="fastest")


@requires_sm103
def test_allowlist_reasons_cover_geometry():
    residual, fn, hc_scale, hc_base = _make_inputs(8, seed=6)
    reasons = spec_mod.supported_with_reasons(
        residual.half(), fn, hc_scale, hc_base, EPS, EPS, EPS, 1.0, SINK
    )
    assert "residual_geometry" in reasons
    reasons = spec_mod.supported_with_reasons(
        residual, fn.t().contiguous(), hc_scale, hc_base, EPS, EPS, EPS, 1.0, SINK
    )
    assert "fn_geometry" in reasons
    reasons = spec_mod.supported_with_reasons(
        residual, fn, hc_scale, hc_base, EPS, EPS, EPS, 1.0, SINK
    )
    assert reasons == []


@requires_sm103
def test_unaligned_base_falls_back():
    residual, fn, hc_scale, hc_base = _make_inputs(8, seed=8)
    backing = torch.empty(NF * HC * H + 4, dtype=torch.float32, device=fn.device)
    fn_unaligned = backing[1 : 1 + NF * HC * H].view(NF, HC * H)
    fn_unaligned.copy_(fn)
    if fn_unaligned.data_ptr() % 16 == 0:
        pytest.skip("could not construct an unaligned base on this allocator")
    reasons = spec_mod.supported_with_reasons(
        residual, fn_unaligned, hc_scale, hc_base, EPS, EPS, EPS, 1.0, SINK
    )
    assert "fn_not_16B_aligned" in reasons
    out = mhc_pre(residual, fn_unaligned, hc_scale, hc_base, backend="auto")
    _assert_tol(out, _reference(residual, fn_unaligned, hc_scale, hc_base))
