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

"""Tests for the "specialized" SM120 backend of ``SSDCombined`` (the
agent-generated Mamba2 SSD chunk-scan prefill kernel: single-sequence,
zero-initial-state, chunk_size=128, headdim=64, dstate=128, ngroups=8,
bf16 io, dt_softplus, no z, no final states).

On SM120 the stock backends refuse the arch outright, so the dispatch tests
verify fail-closed semantics: auto constructs the specialized backend
exactly on its dispatch surface and keeps today's refusal everywhere else;
an explicitly selected backend is never silently swapped; the kill switch
restores the refusal. Numeric correctness is checked against a sequential
fp64 reference recurrence, plus a CUDA-graph capture/replay smoke.
"""

import pytest
import torch
import torch.nn.functional as F


def _is_sm120_exactly():
    # Check the device the suite actually runs on (the current device, which
    # every DEVICE="cuda" tensor resolves to), not ordinal 0.
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(torch.cuda.current_device()) == (12, 0)


def _has_min_cuda_toolkit():
    # Require the actual nvcc the JIT build uses: get_cuda_version() falls
    # back to torch.version.cuda when nvcc is absent, which would enable
    # these JIT tests on a host that cannot compile the module.  The Ninja
    # build honors FLASHINFER_NVCC, so probe the same compiler it selects.
    try:
        import os
        import re
        import subprocess

        from packaging.version import Version

        from flashinfer.jit.cpp_ext import get_cuda_path

        nvcc = os.environ.get(
            "FLASHINFER_NVCC", os.path.join(get_cuda_path(), "bin", "nvcc")
        )
        txt = subprocess.check_output([nvcc, "--version"], text=True)
        matches = re.findall(r"release (\d+\.\d+),", txt)
        return bool(matches) and Version(matches[0]) >= Version("13.3")
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _is_sm120_exactly(),
        reason="the specialized SSD prefill backend is SM120-only",
    ),
]

# Only the tests that actually launch the kernel JIT-build the sm_120a
# module; the construction/refusal tests run without a toolkit.
requires_toolkit = pytest.mark.skipif(
    not _has_min_cuda_toolkit(),
    reason="the specialized backend JIT-builds an sm_120a module "
    "(CUDA >= 13.3 toolkit required)",
)


def _skip_unless_resident(nheads):
    # The NH128 variant is one cooperative launch of nheads + 8 CTAs at one
    # CTA per SM; a device with fewer SMs refuses it (insufficient_sm_count)
    # while NH64 remains fully supported there.
    if nheads >= 96:
        sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        if sms < nheads + 8:
            pytest.skip(
                f"NH{nheads} needs {nheads + 8} resident CTAs, device has {sms} SMs"
            )


DEVICE = "cuda"
HD, DS, NG, CHUNK = 64, 128, 8, 128
REL_TOL = 2e-2


def _make_ssd(nheads, headdim=HD, backend=None):
    from flashinfer.mamba import SSDCombined

    kwargs = dict(
        chunk_size=CHUNK,
        nheads=nheads,
        headdim=headdim,
        dstate=DS,
        ngroups=NG,
    )
    if backend is not None:
        kwargs["backend"] = backend
    return SSDCombined(**kwargs)


def _make_inputs(t, nheads, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen).to(torch.bfloat16).to(DEVICE)

    a_coef = ((-torch.rand(nheads, generator=gen) - 0.05) * 3.0).float()
    return dict(
        x=rnd(t, nheads, HD),
        dt=rnd(t, nheads),
        A=a_coef.to(DEVICE),
        B=rnd(t, NG, DS),
        C=rnd(t, NG, DS),
        D=rnd(nheads),
        dt_bias=rnd(nheads),
    )


def _reference(p, t, nheads):
    """Sequential fp64 SSM recurrence (the op's defining equations)."""
    rep = nheads // NG
    gidx = torch.arange(nheads, device=DEVICE) // rep
    x = p["x"].to(torch.float64)
    b = p["B"].to(torch.float64)
    c = p["C"].to(torch.float64)
    a = p["A"].to(torch.float64)
    dts = F.softplus(p["dt"].to(torch.float64) + p["dt_bias"].to(torch.float64)[None])
    d_coef = p["D"].to(torch.float64)
    y = torch.empty(t, nheads, HD, dtype=torch.float64, device=DEVICE)
    state = torch.zeros(nheads, HD, DS, dtype=torch.float64, device=DEVICE)
    for step in range(t):
        decay = torch.exp(dts[step] * a)
        b_t, c_t = b[step][gidx], c[step][gidx]
        state.mul_(decay[:, None, None]).add_(
            (dts[step][:, None] * x[step])[:, :, None] * b_t[:, None, :]
        )
        y[step] = torch.einsum("hpn,hn->hp", state, c_t) + d_coef[:, None] * x[step]
    return y


def _run(ssd, p, t, nheads, out=None):
    o, final_states = ssd.run(
        x=p["x"].view(1, t, nheads, HD),
        dt=p["dt"].view(1, t, nheads),
        A=p["A"],
        B=p["B"].view(1, t, NG, DS),
        C=p["C"].view(1, t, NG, DS),
        D=p["D"],
        dt_bias=p["dt_bias"],
        dt_softplus=True,
        return_final_states=False,
        out=out,
    )
    assert final_states is None
    return o.view(t, nheads, HD)


def _rel_max(out, ref):
    return ((out.double() - ref).abs().max() / (ref.abs().max() + 1e-9)).item()


@requires_toolkit
@pytest.mark.parametrize("t,nheads", [(120, 64), (500, 128)])
def test_auto_constructs_specialized_and_matches_reference(t, nheads):
    from flashinfer.mamba import specialized_ssd_prefill_stats

    _skip_unless_resident(nheads)
    ssd = _make_ssd(nheads)
    p = _make_inputs(t, nheads, seed=t)
    before = specialized_ssd_prefill_stats()["specialized_dispatches"]
    out = _run(ssd, p, t, nheads)
    torch.cuda.synchronize()
    after = specialized_ssd_prefill_stats()["specialized_dispatches"]
    assert after == before + 1
    rel = _rel_max(out.float(), _reference(p, t, nheads))
    assert rel <= REL_TOL, f"rel {rel} at T={t} nheads={nheads}"


@requires_toolkit
@pytest.mark.parametrize("nheads", [64, 128])
@pytest.mark.parametrize("t,step_pos", [(3, 0), (129, 0), (129, 64)])
def test_large_step_decay_protection(nheads, t, step_pos):
    """A dt step that dominates the fp32 mantissa (2**30) must not erase the
    later tokens' decays: subtracting its cumulative prefix used to collapse
    every following decay to exp(0)=1, silently returning [1, 1, 1] where the
    recurrence gives [1, 0.5, 0.25] (Codex review on the 64-head path; also
    exercised across a chunk boundary and on the protected 128-head path)."""
    _skip_unless_resident(nheads)
    ssd = _make_ssd(nheads)
    dt = torch.zeros(t, nheads, dtype=torch.bfloat16, device=DEVICE)
    dt[step_pos] = 2.0**30
    bc = torch.zeros(t, NG, DS, dtype=torch.bfloat16, device=DEVICE)
    bc[:, :, 0] = 1.0
    p = dict(
        x=torch.full((t, nheads, HD), 2.0**-30, dtype=torch.bfloat16, device=DEVICE),
        dt=dt,
        A=torch.full((nheads,), -1.0, device=DEVICE),
        B=bc,
        C=bc.clone(),
        D=torch.zeros(nheads, dtype=torch.bfloat16, device=DEVICE),
        dt_bias=torch.zeros(nheads, dtype=torch.bfloat16, device=DEVICE),
    )
    out = _run(ssd, p, t, nheads)
    rel = _rel_max(out.float(), _reference(p, t, nheads))
    assert rel <= REL_TOL, f"rel {rel} at T={t} nheads={nheads} step at {step_pos}"


def test_backends_declared_and_bogus_backend_raises():
    from flashinfer.mamba import SSD_BACKENDS

    assert "specialized" in SSD_BACKENDS
    with pytest.raises(ValueError):
        _make_ssd(64, backend="bogus")


def test_explicit_stock_backend_keeps_refusal():
    """An explicitly selected stock backend must never be swapped to the
    specialized kernel: it keeps today's SM120 refusal."""
    with pytest.raises(ValueError):
        _make_ssd(64, backend="cute")


def test_auto_off_surface_keeps_refusal():
    """Off the dispatch surface (headdim 128), auto keeps today's refusal
    (fail-closed)."""
    with pytest.raises(ValueError):
        _make_ssd(64, headdim=128)


def test_explicit_specialized_off_surface_raises():
    with pytest.raises(NotImplementedError, match="specialized"):
        _make_ssd(64, headdim=128, backend="specialized")


def test_unservable_call_form_raises():
    ssd = _make_ssd(64)
    p = _make_inputs(256, 64, seed=1)
    with pytest.raises(NotImplementedError, match="dt_softplus_required"):
        ssd.run(
            x=p["x"].view(1, 256, 64, HD),
            dt=p["dt"].view(1, 256, 64),
            A=p["A"],
            B=p["B"].view(1, 256, NG, DS),
            C=p["C"].view(1, 256, NG, DS),
            D=p["D"],
            dt_bias=p["dt_bias"],
            dt_softplus=False,
            return_final_states=False,
        )


def test_kill_switch(monkeypatch):
    # Set at construction time: auto resolves to today's refusal.
    monkeypatch.setenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1")
    with pytest.raises(ValueError):
        _make_ssd(64)
    # Set after construction: the call raises with the recorded reason.
    monkeypatch.delenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE")
    ssd = _make_ssd(64)
    monkeypatch.setenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1")
    p = _make_inputs(128, 64, seed=2)
    with pytest.raises(NotImplementedError, match="kill_switch"):
        _run(ssd, p, 128, 64)


@requires_toolkit
def test_cuda_graph_replay_matches_eager():
    from flashinfer.mamba import specialized_ssd_prefill_warmup

    specialized_ssd_prefill_warmup(64, device=DEVICE)
    ssd = _make_ssd(64)
    t = 256
    p = _make_inputs(t, 64, seed=3)
    eager = _run(ssd, p, t, 64).clone()
    out = torch.empty(1, t, 64, HD, dtype=torch.bfloat16, device=DEVICE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(ssd, p, t, 64, out=out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(t, 64, HD), eager)
