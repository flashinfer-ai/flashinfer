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

import math

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
    import os
    import re
    import subprocess

    from packaging.version import Version

    nvcc = os.environ.get("FLASHINFER_NVCC")
    if nvcc is None:
        from flashinfer.jit.cpp_ext import get_cuda_path

        try:
            nvcc = os.path.join(get_cuda_path(), "bin", "nvcc")
        except (FileNotFoundError, RuntimeError):
            return False
    try:
        txt = subprocess.check_output([nvcc, "--version"], text=True)
    except FileNotFoundError:
        if "FLASHINFER_NVCC" in os.environ:
            raise
        return False
    match = re.search(r"release (\d+\.\d+),", txt)
    if match is None:
        raise ValueError(f"Unrecognized {nvcc} --version output: {txt}")
    return Version(match[1]) >= Version("13.3")


pytestmark = [
    pytest.mark.skipif(
        not _is_sm120_exactly(),
        reason="the specialized SSD prefill backend is SM120-only",
    ),
]


# Only the tests that actually launch the kernel JIT-build the sm_120a
# module; the construction/refusal tests run without a toolkit.
@pytest.fixture(scope="module")
def _launch_toolkit():
    if not _has_min_cuda_toolkit():
        pytest.skip("the specialized backend requires a CUDA >= 13.3 toolkit")


requires_toolkit = pytest.mark.usefixtures("_launch_toolkit")


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
@pytest.mark.parametrize("t,nheads", [(120, 64), (128, 128), (500, 128)])
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


def test_missing_default_nvcc(monkeypatch, tmp_path):
    from flashinfer.jit import cpp_ext

    monkeypatch.delenv("FLASHINFER_NVCC", raising=False)
    monkeypatch.setattr(cpp_ext, "get_cuda_path", lambda: tmp_path)
    assert not _has_min_cuda_toolkit()


@pytest.mark.parametrize("failure", ["missing", "exit", "banner"])
def test_invalid_nvcc_override(monkeypatch, tmp_path, failure):
    import subprocess

    compiler = tmp_path / "nvcc"
    monkeypatch.setenv("FLASHINFER_NVCC", str(compiler))
    if failure != "missing":
        compiler.write_text(
            "#!/bin/sh\n" + ("exit 2\n" if failure == "exit" else "echo unknown\n")
        )
        compiler.chmod(0o755)
    error = {
        "missing": FileNotFoundError,
        "exit": subprocess.CalledProcessError,
        "banner": ValueError,
    }
    with pytest.raises(error[failure]):
        _has_min_cuda_toolkit()


@pytest.mark.parametrize("release,supported", [("13.2", False), ("13.3", True)])
def test_nvcc_override_without_default_toolkit(monkeypatch, release, supported):
    import subprocess

    from flashinfer.jit import cpp_ext

    compiler = "/override/bin/nvcc"
    monkeypatch.setenv("FLASHINFER_NVCC", compiler)
    monkeypatch.setattr(
        cpp_ext, "get_cuda_path", lambda: pytest.fail("unexpected lookup")
    )

    def version(argv, *, text):
        assert argv == [compiler, "--version"] and text
        return f"Cuda compilation tools, release {release}, V{release}.0"

    monkeypatch.setattr(subprocess, "check_output", version)
    assert _has_min_cuda_toolkit() is supported


@requires_toolkit
def test_nh128_score_cancellation(record_property):
    nheads, t = 128, 2
    _skip_unless_resident(nheads)
    p = _make_inputs(t, nheads, seed=0)
    for v in p.values():
        v.zero_()
    p["A"].fill_(-(2.0**-30))
    p["x"][0].fill_(1)
    p["x"][1].fill_(-1)
    # Row 0 emits nothing. Row 1 subtracts dots 1+2^-9 and 1,
    # which are distinct in the recurrence but equal if staged in bf16.
    p["B"][:, :, 0] = 1
    p["B"][0, :, 1] = 2.0**-9
    p["C"][1, :, :2] = 1
    ref = _reference(p, t, nheads)
    out = _run(_make_ssd(nheads), p, t, nheads)
    rel = _rel_max(out, ref)
    actual, expected = out[1, 0, 0].item(), ref[1, 0, 0].item()
    record_property("row1_actual", actual)
    record_property("row1_fp64", expected)
    record_property("rel_max", rel)
    assert rel <= REL_TOL, (actual, expected, rel)


@requires_toolkit
@pytest.mark.parametrize("row", [15, 29, 56])
def test_nh128_weighted_score_cancellation(row):
    # An exactly representable raw Gram does not suffice: the two steps
    # below differ by less than one BF16 spacing. Exercise both diagonal
    # halves and the off-diagonal tile loads on the full-window route.
    t, nh = 64, 128
    _skip_unless_resident(nh)
    p = _make_inputs(t, nh, seed=1)
    for value in p.values():
        value.zero_()
    p["B"][:, :, 0] = 1
    p["C"][row, :, 0] = 1
    p["x"][1].fill_(1)
    p["x"][row].fill_(-1)
    p["dt"][row].fill_(2.0**-10)
    out = _run(_make_ssd(nh), p, t, nh)
    ref = _reference(p, t, nh)
    assert ref[row, 0, 0].abs().item() > 4e-4
    assert _rel_max(out, ref) <= REL_TOL


@requires_toolkit
@pytest.mark.parametrize("nh", [64, 128])
def test_large_steps_scaled_before_interval_and_suffix_sum(nh):
    _skip_unless_resident(nh)
    t = 129
    p = _make_inputs(t, nh, seed=0)
    for value in p.values():
        value.zero_()
    p["A"].fill_(-(2.0**-126))
    p["x"][125].fill_(1)
    p["B"][125, :, 0] = 1
    p["dt"][126:128].fill_(2.0**127)
    p["C"][127:, :, 0] = 1
    # Both steps scale to -2; their unscaled FP32 sum would be infinite.
    # The last two readouts probe the intra-chunk interval and carried state.
    ref = _reference(p, t, nh)
    out = _run(_make_ssd(nh, backend="specialized"), p, t, nh)
    for row in (127, 128):
        assert ref[row, 0, 0].item() > 0.01
        assert _rel_max(out[row], ref[row]) <= REL_TOL


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


@pytest.mark.parametrize(
    "case",
    [
        "out_rank",
        "out_cpu",
        "device_mismatch",
        "alignment",
        "out_overlap",
        "capture_unwarmed",
    ],
)
def test_refusal_before_kernel_entry(monkeypatch, case):
    from flashinfer.mamba.specialized import ssd_prefill_sm120 as backend

    monkeypatch.setattr(backend, "_kernel_entry", lambda: pytest.fail("entered JIT"))
    t, nh = 256, 64
    p = _make_inputs(t, nh, seed=1)
    ssd = _make_ssd(nh, backend="specialized")
    out = None
    reason = "out_form" if case in ("out_rank", "out_cpu") else case
    if case == "out_rank":
        out = torch.empty_like(p["x"])
    elif case == "out_cpu":
        out = torch.empty_like(p["x"].unsqueeze(0), device="cpu")
    elif reason == "device_mismatch":
        p["D"] = p["D"].cpu()
    elif reason == "alignment":
        b = p["B"]
        p["B"] = b.new_empty(b.numel() + 1)[1:].view_as(b)
    elif reason == "out_overlap":
        x = p["x"]
        backing = x.new_empty(x.numel() + 8)
        p["x"] = backing[:-8].view_as(x)
        out = backing[8:].view(1, t, nh, HD)
    else:
        monkeypatch.setattr(backend, "_warmed", set())
        monkeypatch.setattr(backend, "_native_device", None)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    before = backend.specialized_ssd_prefill_stats()["fallback_reasons"].get(reason, 0)
    with pytest.raises(NotImplementedError, match=rf"reason: {reason}\)"):
        _run(ssd, p, t, nh, out=out)
    assert (
        backend.specialized_ssd_prefill_stats()["fallback_reasons"][reason]
        == before + 1
    )


@requires_toolkit
@pytest.mark.parametrize("nh", [64, 128])
def test_native_operand_checks(nh):
    from flashinfer.mamba.specialized import ssd_prefill_sm120 as backend

    _skip_unless_resident(nh)
    t = 256
    p = _make_inputs(t, nh, seed=1)
    run = backend._kernel_entry()
    y = torch.empty_like(p["x"])
    keys = ("x", "dt", "A", "B", "C", "D", "dt_bias")

    def call(inputs, dst):
        run(*(inputs[k] for k in keys), dst)

    # Every accepted call submits the primary SSD kernel before any recovery.
    # Check its device events independently of Python dispatch statistics.
    def profile():
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )

    def launches(trace):
        return sum(
            e.device_type == torch.autograd.DeviceType.CUDA
            and "ssd_primary2051" in e.name
            for e in trace.events()
        )

    with profile() as control:
        call(p, y)
        torch.cuda.synchronize()
    assert launches(control) == 1
    assert _rel_max(y, _reference(p, t, nh)) <= REL_TOL
    saved_y = y.clone()
    with profile() as refused:
        # Keep full-size backing even for the short destination. A missing
        # check must fail the test, not write beyond a test allocation.
        for key, view, message in (
            ("x", p["x"].unsqueeze(-1), "x must have rank 3"),
            ("x", p["x"][:0], "invalid SSD sequence length"),
            ("x", p["x"][:, :32], "SSD requires 64 or 128 heads"),
            ("y", y[:-1], "y has wrong shape"),
            ("dt", p["dt"].view(torch.float16), "dt has wrong dtype"),
            (
                "C",
                p["C"].transpose(1, 2).contiguous().transpose(1, 2),
                "C must be contiguous",
            ),
        ):
            with pytest.raises(RuntimeError, match=message):
                call(dict(p, **{key: view}), view if key == "y" else y)
            assert torch.equal(y, saved_y)
        for key in ("x", "B", "C", "y"):
            value = y if key == "y" else p[key]
            backing = value.new_full((value.numel() + 1,), 3)
            view = backing[1:].view_as(value)
            assert view.data_ptr() % 16 == 2
            bad = dict(p, **{key: view})
            with pytest.raises(RuntimeError, match="16-byte aligned"):
                call(bad, view if key == "y" else y)
            assert (backing == 3).all() and torch.equal(y, saved_y)
        for key in keys:
            # Distinct, aligned starts; exactly 16 bytes in common. Use a
            # byte view so A's fp32 footprint is tested as bytes, not bf16s.
            value = p[key]
            n = value.numel() * value.element_size()
            backing = torch.full(
                (n + y.numel() * 2 - 16,), 7, device=DEVICE, dtype=torch.uint8
            )
            src = backing[:n].view(value.dtype).view_as(value)
            dst = backing[n - 16 :].view(y.dtype).view_as(y)
            with pytest.raises(RuntimeError, match=f"y must not overlap {key}"):
                call(dict(p, **{key: src}), dst)
            assert (backing == 7).all()
        torch.cuda.synchronize()
    assert launches(refused) == 0


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


def _retry_worker(a, b, nh):
    from flashinfer.mamba.specialized import ssd_prefill_sm120 as backend

    t = 3
    torch.cuda.set_device(a)
    run = backend._kernel_entry()
    p = _make_inputs(t, nh, seed=4)
    y = torch.full_like(p["x"], 7)
    torch.cuda.synchronize(a)
    # Only the submission uses the foreign stream. Attribute setup (and
    # cooperative occupancy validation) uses the valid current device A.
    owner = torch.cuda.Stream(device=b)
    foreign = torch.cuda.ExternalStream(owner.cuda_stream, device=a)
    with (
        torch.cuda.stream(foreign),
        pytest.raises(RuntimeError, match="stream belongs to another CUDA context"),
    ):
        run(*p.values(), y)
    assert (y == 7).all()

    torch.cuda.set_device(b)
    q = _make_inputs(t, nh, seed=5)
    dst = torch.full_like(q["x"], float("nan"))
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as trace:
        run(*q.values(), dst)
        torch.cuda.synchronize(b)
    assert (
        sum(
            e.device_type == torch.autograd.DeviceType.CUDA
            and "ssd_primary2051" in e.name
            for e in trace.events()
        )
        == 1
    )
    assert _rel_max(dst, _reference(q, t, nh)) <= REL_TOL
    torch.cuda.set_device(a)
    with pytest.raises(RuntimeError, match="SSD bound to another CUDA device"):
        run(*p.values(), y)
    assert (y == 7).all()


@requires_toolkit
@pytest.mark.parametrize("nh", [64, 128])
def test_failed_first_submission_retries_on_another_device(nh):
    import multiprocessing

    devices = [
        i
        for i in range(torch.cuda.device_count())
        if torch.cuda.get_device_capability(i) == (12, 0)
        and (
            nh == 64
            or torch.cuda.get_device_properties(i).multi_processor_count >= nh + 8
        )
    ]
    if len(devices) < 2:
        pytest.skip(f"requires two eligible SM120 devices for NH{nh}")
    child = multiprocessing.get_context("spawn").Process(
        target=_retry_worker, args=(*devices[:2], nh)
    )
    child.start()
    try:
        child.join(timeout=600)
        assert child.exitcode == 0, f"retry worker exit code: {child.exitcode}"
    finally:
        if child.is_alive():
            child.kill()
            child.join()


def _graph_worker(device, nh, t):
    from flashinfer.mamba import specialized_ssd_prefill_warmup

    torch.cuda.set_device(device)
    specialized_ssd_prefill_warmup(nh, device=device)  # T=128 only
    ssd = _make_ssd(nh)
    p = _make_inputs(t, nh, seed=3)
    out = torch.empty(1, t, nh, HD, dtype=torch.bfloat16, device=DEVICE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(ssd, p, t, nh, out=out)
    for sign in (1, -1):
        p["x"].mul_(sign)
        out.fill_(float("nan"))
        graph.replay()
        eager = _run(ssd, p, t, nh)
        assert eager.count_nonzero() > 0
        assert torch.equal(out.view_as(eager), eager)


@requires_toolkit
@pytest.mark.parametrize("nh,t", [(64, 256), (128, 129), (128, 256 * 64 + 1)])
def test_cuda_graph_replay_matches_eager(nh, t):
    import multiprocessing

    _skip_unless_resident(nh)
    child = multiprocessing.get_context("spawn").Process(
        target=_graph_worker, args=(torch.cuda.current_device(), nh, t)
    )
    child.start()
    try:
        child.join(timeout=600)
        assert child.exitcode == 0, f"graph worker exit code: {child.exitcode}"
    finally:
        if child.is_alive():
            child.kill()
            child.join()


@requires_toolkit
@pytest.mark.parametrize("nheads", [64, 128])
@pytest.mark.parametrize("readout", [64, 192])
@pytest.mark.parametrize("amplitude", [1.0, 1024.0])
@pytest.mark.parametrize("backend", ["auto", "specialized"])
def test_carried_state_cancellation(nheads, readout, amplitude, backend):
    from flashinfer.mamba import SSDCombined

    t = readout + 1
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("requires SM120")
    if (
        nheads == 128
        and torch.cuda.get_device_properties(device).multi_processor_count < 136
    ):
        pytest.skip("NH128 requires 136 resident CTAs")
    x = torch.zeros((1, t, nheads, 64), dtype=torch.bfloat16, device=device)
    dt = torch.zeros((1, t, nheads), dtype=torch.bfloat16, device=device)
    A = torch.full((nheads,), -1.0, dtype=torch.float32, device=device)
    B = torch.zeros((1, t, 8, 128), dtype=torch.bfloat16, device=device)
    C = torch.zeros_like(B)
    D = torch.zeros((nheads,), dtype=torch.bfloat16, device=device)
    bias = torch.zeros_like(D)
    x[:, readout - 2 : readout].fill_(amplitude)
    B[:, readout - 2, :, :2] = 1.0
    B[:, readout - 1, :, 0] = 2.0**-10
    C[:, readout, :, 0] = 1.0
    C[:, readout, :, 1] = -1.0
    # softplus(0)=ln(2), exp(-ln(2))=1/2. The equal carried
    # components cancel, leaving the small injection times one decay.
    expected = torch.zeros_like(x)
    expected[:, readout].fill_(math.log(2.0) * amplitude * 2.0**-11)
    op = SSDCombined(
        chunk_size=128,
        nheads=nheads,
        headdim=64,
        dstate=128,
        ngroups=8,
        backend=backend,
    )
    for supplied in (False, True):
        out = torch.full_like(x, float("nan")) if supplied else None
        actual, state = op.run(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=bias,
            dt_softplus=True,
            return_final_states=False,
            out=out,
        )
        assert state is None
        if supplied:
            assert actual is out
        assert torch.equal(actual, expected)


@requires_toolkit
@pytest.mark.parametrize("nheads", [64, 128])
@pytest.mark.parametrize("backend", ["auto", "specialized"])
@pytest.mark.parametrize(
    "case", ["weighted_state", "gram_cancellation", "gram_underflow"]
)
def test_precision_before_staging(nheads, backend, case):
    from flashinfer.mamba import SSDCombined
    from flashinfer.mamba.specialized import ssd_prefill_sm120 as impl

    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("requires SM120")
    if (
        nheads == 128
        and torch.cuda.get_device_properties(device).multi_processor_count < 136
    ):
        pytest.skip("NH128 requires 136 resident CTAs")
    length = 65 if case == "weighted_state" else 2

    def zeros(*shape):
        return torch.zeros(shape, device=device, dtype=torch.bfloat16)

    x = zeros(1, length, nheads, 64)
    dt = zeros(1, length, nheads)
    A = torch.zeros(nheads, device=device, dtype=torch.float32)
    B, C = zeros(1, length, 8, 128), zeros(1, length, 8, 128)
    D, bias = zeros(nheads), zeros(nheads)
    if case == "weighted_state":
        # With A=0 the two injections are 256 and -256.5.
        # Rounding both weights to BF16 before injection erases the result.
        bias.fill_(256)
        dt[:, 63] = 0.5
        x[:, 62:64] = 1
        B[:, 62, :, 0], B[:, 63, :, 0] = 1, -1
        C[:, 64, :, 0] = 1
        expected_value = -0.5
    elif case == "gram_cancellation":
        # Sequential FP32 Gram summation loses 2^-30 between +1 and -1.
        dt.fill_(32)
        x[:, 0] = 1
        B[:, 0, :, 0], B[:, 0, :, 1], B[:, 0, :, 2] = 1, 2.0**-30, -1
        C[:, 1, :, :3] = 1
        expected_value = 2.0**-25
    else:
        # B*C underflows FP32 before the large X rescales the normal output.
        dt.fill_(32)
        x[:, 0] = 2.0**100
        B[:, 0, :, 0] = C[:, 1, :, 0] = 2.0**-80
        expected_value = 2.0**-55
    expected = torch.zeros_like(x)
    expected[:, -1] = expected_value
    op = SSDCombined(
        chunk_size=128,
        nheads=nheads,
        headdim=64,
        dstate=128,
        ngroups=8,
        backend=backend,
    )
    for supplied in (False, True):
        out = torch.full_like(x, float("nan")) if supplied else None
        before = dict(impl._stats)
        actual, state = op.run(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=bias,
            dt_softplus=True,
            return_final_states=False,
            out=out,
        )
        assert state is None
        if supplied:
            assert actual is out
        assert (
            impl._stats["specialized_dispatches"]
            == before["specialized_dispatches"] + 1
        )
        assert impl._stats["fallback_dispatches"] == before["fallback_dispatches"]
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@requires_toolkit
@pytest.mark.parametrize("nheads", [64, 128])
@pytest.mark.parametrize(
    "failure", ["optional_oom", "small_oom", "conversion_oom", "other_error"]
)
def test_framework_workspace_allocation_failures(nheads, failure, monkeypatch):
    import tvm_ffi
    from flashinfer.mamba.specialized import ssd_prefill_sm120 as impl

    _skip_unless_resident(nheads)
    t = 65
    p = _make_inputs(t, nheads, seed=91)
    p["B"].zero_()
    p["C"].zero_()
    p["D"].fill_(1)
    ssd = _make_ssd(nheads)
    out = torch.empty_like(p["x"]).unsqueeze(0)
    _run(ssd, p, t, nheads, out)
    torch.cuda.synchronize()
    assert torch.equal(out[0], p["x"])
    base_bytes = 1032 * nheads
    real_empty, real_convert = torch.empty, tvm_ffi.from_dlpack
    requests, conversions = [], []

    def allocate(size, *args, **kwargs):
        nbytes = size[0] if isinstance(size, tuple) else size
        if kwargs.get("dtype") == torch.uint8:
            requests.append(nbytes)
            if failure == "small_oom":
                raise torch.OutOfMemoryError("QA small allocation refusal")
            if failure == "optional_oom" and nbytes > base_bytes:
                raise torch.OutOfMemoryError("QA optional allocation refusal")
            if failure == "other_error":
                raise RuntimeError("QA unrelated allocation error")
        return real_empty(size, *args, **kwargs)

    def convert(tensor, *args, **kwargs):
        conversions.append(tensor.numel())
        if failure == "conversion_oom":
            raise torch.OutOfMemoryError("QA conversion refusal")
        return real_convert(tensor, *args, **kwargs)

    out.fill_(-19)
    torch.cuda.synchronize()
    stamp = impl.specialized_ssd_prefill_stats()["specialized_dispatches"]
    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", allocate)
        patch.setattr(tvm_ffi, "from_dlpack", convert)
        if failure == "optional_oom":
            _run(ssd, p, t, nheads, out)
            torch.cuda.synchronize()
            assert torch.equal(out[0], p["x"])
            assert len(requests) == 2 and requests[0] > base_bytes
            assert requests[1] == base_bytes and conversions == [base_bytes]
            assert (
                impl.specialized_ssd_prefill_stats()["specialized_dispatches"]
                == stamp + 1
            )
        else:
            message = {
                "small_oom": "QA small allocation refusal",
                "conversion_oom": "QA conversion refusal",
                "other_error": "QA unrelated allocation error",
            }[failure]
            with pytest.raises(Exception, match=message):
                _run(ssd, p, t, nheads, out)
            torch.cuda.synchronize()
            assert (out == -19).all()
            assert (
                impl.specialized_ssd_prefill_stats()["specialized_dispatches"] == stamp
            )
            assert len(requests) == (2 if failure == "small_oom" else 1)
            if failure == "conversion_oom":
                assert len(conversions) == 1
    _run(ssd, p, t, nheads, out)
    torch.cuda.synchronize()
    assert torch.equal(out[0], p["x"])


@requires_toolkit
@pytest.mark.parametrize("nh", [64, 128])
def test_capture_guard_uses_operand_device(nh, monkeypatch):
    from flashinfer.mamba.specialized import ssd_prefill_sm120 as impl

    devices = [
        i
        for i in range(torch.cuda.device_count())
        if torch.cuda.get_device_capability(i) == (12, 0)
        and (
            nh == 64 or torch.cuda.get_device_properties(i).multi_processor_count >= 136
        )
    ]
    if len(devices) < 2:
        pytest.skip("requires two eligible SM120 devices")
    a, b = devices[:2]
    t = 65
    with torch.cuda.device(a):
        ssd = _make_ssd(nh)
        p = _make_inputs(t, nh, seed=71)
        out = torch.empty_like(p["x"]).unsqueeze(0)
        _run(ssd, p, t, nh, out)
        torch.cuda.synchronize(a)
        stream = torch.cuda.Stream(device=a)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream), torch.cuda.device(b):
            _run(ssd, p, t, nh, out)
            assert torch.cuda.current_device() == b
        for _ in range(3):
            p["x"].neg_()
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(a)
            with torch.cuda.device(b):
                expected = _run(ssd, p, t, nh)
                assert torch.cuda.current_device() == b
            torch.cuda.synchronize(a)
            assert torch.equal(out[0].view(torch.int16), expected.view(torch.int16))
        # Capture status remains dynamic even after immutable facts are cached.
        with monkeypatch.context() as patch:
            patch.setattr(impl, "_warmed", set())
            empty = torch.cuda.CUDAGraph()
            with torch.cuda.graph(empty, stream=stream), torch.cuda.device(b):
                with pytest.raises(NotImplementedError, match="capture_unwarmed"):
                    _run(ssd, p, t, nh, out)
                assert torch.cuda.current_device() == b
