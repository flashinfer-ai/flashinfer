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

Backend suite of the Cake SiLU-gated RMSNorm training package of the
non-deterministic family
(``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train``, the backend of
:mod:`flashinfer.cake_rmsnorm_silu_gate_nondet_train`):
``RMSNorm(x) * w * SiLU(z)``.

``y``, ``dx`` and ``dz`` must lie within one BF16 ulp of an independent FP64
reference, and the weight gradient within the deterministic suite's budget,
``|dw - dw64| <= |dw64| * 2^-7 + sum|terms| * 2^-20``.  Every output (``y``,
``dx``, ``dz`` and ``dw``, BF16 and FP32 weight gradients, accumulated or not)
must be the same bits on every launch and on every path: streams, CUDA Graph
replays, autograd, ``compute_grad_weight``, grids smaller than the chunks (the
static and the claimed backward at 1, 3 and 7 CTAs), a workspace reused across
shapes.  The matrix covers dynamic token and head counts (``H = 48`` and small
or irregular row counts), ``[T, H, 128]`` strided and flattened layouts,
near-zero variance, saturated gates, FP32 and BF16 weight gradients,
accumulation, ``compute_grad_weight=False``, the short backward (one cluster up
to 256 rows, BF16 weight gradient: no workspace made or touched, a caller
workspace checked and left as it is, the first row count past it on the
backward), empty work, a workspace reused across shapes (zero after every
call), a workspace with nonzero scratch returned to zero by
``cake_rmsnorm_silu_gate_nondet_train_reset_workspace``, two streams with their own
workspaces, CUDA Graph capture and replay (captured calls without a
workspace: one workspace per capture and capture stream, zero-filled outside
the graph, no fill kernel in it; two such graphs replayed at the same time on
two streams; the fill inside the graph past the graph-workspace limit), the
autograd wrapper (non-reentrant checkpointing included), the state a
checkpointed step keeps (FlashInfer issue #7039), the registry's routes, chunks
and grids, and the mode policy: the mode is in the name, so no entry point
takes a ``deterministic`` keyword (passing one, of any value, is a
``TypeError`` before any launch), every entry point runs without an alert or a
warning under ``torch.use_deterministic_algorithms(True)`` (strict and
``warn_only``, CUDA Graph capture included) and returns the bits it returns
with the setting off, and the deterministic package's FP32 workspace is refused
naming both modes' workspaces.

This suite exercises the package alone, named once through
``tests.test_helpers.cake_package`` (so a vendored copy runs it unchanged), and
imports nothing else from flashinfer.  The comparisons against the
deterministic package (each output's FP64 error at most 1.1 times the
deterministic package's on the same inputs), the facade and the ``flashinfer``
top level are ``test_cake_rmsnorm_silu_gate_nondet_train_api.py``'s.
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import numpy
import pytest
import torch
import torch.utils.checkpoint

from tests.test_helpers.cake_package import cake_package

KEY = "cake_rmsnorm_silu_gate_nondet_train"
pkg = cake_package(KEY)
cake_build = cake_package(KEY, "cake_build")
loader = cake_package(KEY, "cake_jit")
forward = pkg.cake_rmsnorm_silu_gate_nondet_train_forward
backward = pkg.cake_rmsnorm_silu_gate_nondet_train_backward
workspace_for = pkg.cake_rmsnorm_silu_gate_nondet_train_backward_workspace
workspace_bytes = pkg.cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes
reset_workspace = pkg.cake_rmsnorm_silu_gate_nondet_train_reset_workspace
gated_rmsnorm = pkg.cake_rmsnorm_silu_gate_nondet

D = 128
EPS = 1e-6
# One BF16 unit in the last place relative to the value (8 significant bits).
BF16_RTOL = 2.0**-7
# The FP32 accumulation term of the weight-gradient budget, relative to the sum
# of the terms' magnitudes (the deterministic suite's).
DW_MAG_TOL = 2.0**-20
REPEATS = 10
DET = "flashinfer.cake_rmsnorm_silu_gate_det_train"
# The bytes of a backward workspace: the u64 windows of the exact
# weight-gradient sum (4 slots x 7 windows x 128 columns), the FP32 class words
# (128) and the u32 arrival ticket and claim counter, each buffer 16-byte
# aligned.
WORKSPACE_BYTES = 29192
WORKSPACE_LAYOUT = (
    ("dw_win", 0, 4 * 7 * 128),
    ("dw_spec", 28672, 128),
    ("dw_done", 29184, 2),
)

# The accuracy matrix; test_cake_rmsnorm_silu_gate_nondet_train_api.py runs the same
# cases against the deterministic package.
SHAPES = [
    (1, 1),
    (1, 48),
    (3, 48),
    (5, 48),
    (13, 48),
    (17, 7),
    (255, 2),
    (1031, 48),
    (4097, 16),
    (2, 64),
    (33, 1),
]
LAYOUTS = ["flat", "projection", "head_major", "misaligned"]
EDGE_DATA = ["near_zero", "saturated"]
EPSILONS = [1e-5, 1e-3]
ISSUE_TOKENS = [30193, 32737, 32388]
FP32_SHAPES = [(7, 48), (1031, 48), (32737, 48)]


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and pkg.is_cake_rmsnorm_silu_gate_nondet_train_supported(torch.device("cuda"))
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)


@pytest.fixture(autouse=True)
def deterministic_algorithms_off():
    """Run with ``torch.use_deterministic_algorithms`` off (tests that need it
    on turn it on); restore the caller's setting."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The module names this package's registry launches, in order."""
    launched: list[str] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        launched.append(name)
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    return launched


def _routes() -> dict[str, str]:
    return loader.ROUTES[loader.device_arch(torch.cuda.current_device())]


def _module(role: str, rows: int) -> str:
    """The module ``module_for_rows`` runs for ``role`` and ``rows`` rows on
    this device."""
    device_index = torch.cuda.current_device()
    return loader.module_for_rows(
        loader.device_arch(device_index),
        role,
        rows,
        loader.device_sm_count(device_index),
    )


def _forward_module(rows: int) -> str:
    """The forward module ``module_for_rows`` runs for ``rows`` rows on this
    device (the short forward where the architecture's rule routes them)."""
    return _module("fwd", rows)


def _backward_module(rows: int, dw_dtype: torch.dtype = torch.bfloat16) -> str:
    """The backward module ``module_for_rows`` runs for ``rows`` rows and a
    ``dw_dtype`` weight gradient on this device: the short backward up to 256
    rows (BF16), the static backward while the rows' chunks fit one wave of
    CTAs, the claimed backward past it."""
    return _module("bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16", rows)


def fp64_reference(dy, x, z, w, eps):
    """An independent FP64 forward and closed-form backward from the BF16 inputs:
    ``(y, dx, dz, dw)`` in FP64 (unrounded)."""
    xd, zd, wd, gd = x.double(), z.double(), w.double(), dy.double()
    rstd = 1.0 / torch.sqrt(xd.square().mean(dim=-1, keepdim=True) + eps)
    n = xd * rstd
    s = torch.sigmoid(zd)
    silu = zd * s
    y = n * wd * silu
    a = gd * silu
    dz = gd * n * wd * s * (1.0 + zd * (1.0 - s))
    dn = a * wd
    dx = rstd * dn - xd * rstd**3 * (dn * xd).mean(dim=-1, keepdim=True)
    dw = (a * n).reshape(-1, D).sum(dim=0)
    return y, dx, dz, dw


def magnitudes(dy, x, z, eps):
    """``sum |a * n|`` per column: the scale of the weight gradient's rounding."""
    terms = (dy.double() * torch.nn.functional.silu(z.double())) * (
        x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + eps)
    )
    return terms.abs().reshape(-1, D).sum(0)


def assert_bf16_close(got, want, *, what, scale=None):
    """``got`` (BF16) within one BF16 ulp of ``want`` plus an absolute floor of
    1/64 ulp of the tensor's largest magnitude (cancellation in ``dx``)."""
    want = want.double()
    floor = (scale if scale is not None else want.abs().max().item()) * BF16_RTOL / 64
    err = (got.double() - want).abs()
    bound = want.abs() * BF16_RTOL + floor
    bad = err > bound
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} of {bad.numel()} elements beyond one BF16 ulp; "
        f"worst err {err.max().item():.3e}"
    )


def assert_dw_within(dw, want, mag, *, what):
    """The weight-gradient budget: ``|dw - want| <= |want| * 2^-7 + mag * 2^-20``."""
    err = (dw.double() - want.double()).abs()
    bound = want.double().abs() * BF16_RTOL + mag * DW_MAG_TOL
    bad = err > bound
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} of {bad.numel()} columns beyond the budget; "
        f"worst excess {(err - bound).max().item():.3e}"
    )


def make(T, H, *, layout="contiguous", data="normal", seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, generator=g).bfloat16()

    if layout == "contiguous":
        x, z, dy = randn(T, H, D), randn(T, H, D), randn(T, H, D)
    elif layout == "flat":
        x, z, dy = randn(T * H, D), randn(T * H, D), randn(T * H, D)
    elif layout == "projection":
        # z (and dy) as the gate columns of wider projections, read in place.
        x = randn(T, H, D)
        z = randn(T, 2 * H * D + 16)[:, H * D : 2 * H * D].view(T, H, D)
        dy = randn(T, 2 * H * D + 16)[:, : H * D].view(T, H, D)
    elif layout == "head_major":
        x = randn(H, T, D).permute(1, 0, 2)
        z = randn(H, T, D).permute(1, 0, 2)
        dy = randn(T, H, D)
    elif layout == "misaligned":
        # Storage offsets of 8 elements: materialized once by the host.
        x = randn(T * H * D + 8)[8:].view(T, H, D)
        z = randn(T * H * D + 8)[8:].view(T, H, D)
        dy = randn(T * H * D + 8)[8:].view(T, H, D)
    else:
        raise ValueError(layout)
    if data == "near_zero":
        flat = torch.arange(T * H, device=device).view(T, H)
        xv = x.view(T, H, D)
        xv[flat % 3 == 0] = 0
        xv[flat % 3 == 1] = 2.0**-60
    elif data == "saturated":
        values = torch.tensor(
            [-1e4, -100.0, -90.0, -30.0, -10.0, 10.0, 30.0, 90.0, 100.0, 1e4],
            device=device,
        ).bfloat16()
        pick = torch.randint(0, 10, z.shape, device=device, generator=g)
        z.copy_(
            torch.where(
                torch.rand(z.shape, device=device, generator=g) < 0.5, values[pick], z
            )
        )
    w = (1.0 + 0.5 * torch.randn(D, device=device, generator=g)).bfloat16()
    return x, z, w, dy


def check_all(x, z, w, dy, eps, *, dw_dtype=torch.bfloat16, repeats=REPEATS):
    """``repeats`` forward and backward launches against the FP64 reference,
    every output the same bits on every launch, the weight gradient also when
    accumulated into a nonzero buffer; returns ``(y, dx, dz, dw)`` of the first
    launch."""
    y64, dx64, dz64, dw64 = fp64_reference(dy, x, z, w, eps)
    mag = magnitudes(dy, x, z, eps)

    def out(fill=None):
        if fill is not None:
            return torch.full((D,), fill, dtype=dw_dtype, device=x.device)
        return (
            None
            if dw_dtype == torch.bfloat16
            else torch.empty(D, dtype=dw_dtype, device=x.device)
        )

    first = None
    for i in range(repeats):
        y = forward(x, z, w, eps)
        assert y.dtype == torch.bfloat16 and y.shape == x.shape
        dx, dz, dw = backward(dy, x, z, w, eps, grad_weight=out())
        assert dx.dtype == dz.dtype == torch.bfloat16 and dw.dtype == dw_dtype
        assert dx.shape == dz.shape == x.shape and dw.shape == (D,)
        main = out(0.25)
        backward(dy, x, z, w, eps, grad_weight=main, accumulate_grad_weight=True)
        if first is None:
            first = y, dx, dz, dw, main
            assert_bf16_close(y, y64, what="y vs FP64")
            assert_bf16_close(dz, dz64, what="dz vs FP64")
            assert_bf16_close(dx, dx64, what="dx vs FP64")
            assert_dw_within(dw, dw64, mag, what="dw vs FP64")
            assert_dw_within(main, 0.25 + dw64, mag, what="accumulated dw vs FP64")
        for what, got, want in zip(
            ("y", "dx", "dz", "dw", "accumulated dw"),
            (y, dx, dz, dw, main),
            first,
            strict=True,
        ):
            assert torch.equal(got, want), f"{what}: launch {i} vs the first launch"
    return first[:4]


@pytest.mark.parametrize("T,H", SHAPES)
def test_forward_backward_shapes(T, H):
    x, z, w, dy = make(T, H)
    check_all(x, z, w, dy, EPS)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_layouts(layout):
    x, z, w, dy = make(65, 48, layout=layout, seed=1)
    check_all(x, z, w, dy, EPS)


@pytest.mark.parametrize("data", EDGE_DATA)
def test_edge_values(data):
    x, z, w, dy = make(37, 48, data=data, seed=2)
    check_all(x, z, w, dy, EPS)


@pytest.mark.parametrize("eps", EPSILONS)
def test_epsilon(eps):
    x, z, w, dy = make(19, 48, seed=3)
    check_all(x, z, w, dy, eps)


@pytest.mark.parametrize("T", ISSUE_TOKENS)
def test_issue_shapes(T):
    x, z, w, dy = make(T, 48, seed=4)
    check_all(x, z, w, dy, EPS)


@pytest.mark.parametrize("T,H", FP32_SHAPES)
def test_fp32_weight_gradient(T, H):
    x, z, w, dy = make(T, H, seed=5)
    check_all(x, z, w, dy, EPS, dw_dtype=torch.float32)


def test_weight_gradient_accumulation():
    x, z, w, dy = make(1031, 48, seed=6)
    x2, z2, _, dy2 = make(1031, 48, seed=7)
    dw64 = fp64_reference(dy, x, z, w, EPS)[3]
    dw64b = fp64_reference(dy2, x2, z2, w, EPS)[3]
    mag = magnitudes(dy, x, z, EPS) + magnitudes(dy2, x2, z2, EPS)
    first = 0.25 + dw64
    want = first + dw64b
    # Two roundings into the accumulator's dtype, each within its relative unit.
    bound = (first.abs() + want.abs()) * BF16_RTOL + mag * DW_MAG_TOL
    for dtype in (torch.float32, torch.bfloat16):
        results = []
        for _ in range(3):
            main = torch.full((D,), 0.25, dtype=dtype, device="cuda")
            for args in ((dy, x, z), (dy2, x2, z2)):
                backward(*args, w, EPS, grad_weight=main, accumulate_grad_weight=True)
            results.append(main)
        assert ((main.double() - want).abs() <= bound).all(), dtype
        if dtype == torch.float32:
            assert torch.allclose(main.double(), want, rtol=1e-5, atol=1e-3)
        for other in results[1:]:
            assert torch.equal(other, results[0]), dtype
    # autograd accumulation into .grad over two microbatches: the sums of the
    # functional backward's gradients
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    gated_rmsnorm(xa, za, wa, EPS).backward(dy)
    gated_rmsnorm(xa, za, wa, EPS).backward(dy)
    dx1, dz1, dw1 = backward(dy, x, z, w, EPS)
    assert torch.equal(xa.grad, dx1 + dx1)
    assert torch.equal(za.grad, dz1 + dz1)
    assert torch.equal(wa.grad, dw1 + dw1)
    # Two BF16 weight gradients within budget, summed in BF16 by autograd.
    mag1 = magnitudes(dy, x, z, EPS)
    err = (wa.grad.double() - 2 * dw64).abs()
    assert (err <= (2 * dw64).abs() * 2.0**-6 + mag1 * 2.0**-18).all()


def test_compute_grad_weight_false_touches_no_weight_state(monkeypatch):
    x, z, w, dy = make(257, 48, seed=12)
    # dx and dz do not depend on whether the weight gradient is computed
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    ws = workspace_for(257 * 48, x.device)
    main = torch.full((D,), 3.0, dtype=torch.float32, device="cuda")
    dx, dz, dw = backward(
        dy, x, z, w, EPS, grad_weight=main, compute_grad_weight=False, workspace=ws
    )
    assert dw is None
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
    torch.cuda.synchronize()
    assert torch.equal(main, torch.full_like(main, 3.0))
    assert not bool(ws.any())
    # without a workspace: the launch still takes the arrival ticket (and the
    # claimed backward claims its chunks), on the stream's workspace of the
    # BF16 module, left zero
    monkeypatch.setattr(pkg, "_STREAM_WORKSPACES", {})
    dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
    assert dw is None
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
    torch.cuda.synchronize()
    key = (
        torch.cuda.current_device(),
        torch.cuda.current_stream().cuda_stream,
        _backward_module(257 * 48),
    )
    assert list(pkg._STREAM_WORKSPACES) == [key]
    views = pkg._STREAM_WORKSPACES[key]
    assert sorted(views) == ["dw_done", "dw_spec", "dw_win"]
    # (through bytes: uint64 tensors take few operations)
    assert not any(bool(view.view(torch.uint8).any()) for view in views.values())
    # autograd with a weight that needs no gradient
    xa, za = (t.clone().requires_grad_(True) for t in (x, z))
    gated_rmsnorm(xa, za, w, EPS).backward(dy)
    assert torch.equal(xa.grad, dx_d) and torch.equal(za.grad, dz_d)
    assert w.grad is None


@pytest.mark.parametrize("ctas", [1, 3, 7])
@pytest.mark.parametrize("T", [129, 300], ids=["static", "claimed"])
def test_backward_at_grids_smaller_than_the_chunks(T, ctas, launches, monkeypatch):
    """The host's grid forced to ``ctas`` CTAs for ``T * 48`` rows (774 chunks
    of one 8-row block, which the static backward runs while they fit one wave
    of CTAs, and 1800, which the claimed backward runs on up to 225 SMs), so
    every CTA runs many chunks: every output (``dx``, ``dz``, BF16 and FP32
    ``dw``, alone and accumulated) the bits of the host's grid;
    ``compute_grad_weight=False`` leaves ``grad_weight`` as it is; the
    workspace is zero after every launch."""
    rows = T * 48
    role = loader.MODULES[_backward_module(rows)]["role"]
    tier = "bwd_bf16" if T == 129 else "bwd_bf16_mid"
    if role != tier:
        pytest.skip(f"{rows} rows run {role} on this device's SMs, not {tier}")
    x, z, w, dy = make(T, 48, seed=12)
    dw64 = fp64_reference(dy, x, z, w, EPS)[3]
    mag = magnitudes(dy, x, z, EPS)
    want = {}
    for dtype in (torch.bfloat16, torch.float32):
        dw = torch.empty(D, dtype=dtype, device="cuda")
        dx_d, dz_d, _ = backward(dy, x, z, w, EPS, grad_weight=dw)
        main = torch.full((D,), 0.25, dtype=dtype, device="cuda")
        backward(dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True)
        want[dtype] = dw, main
    ws = workspace_for(rows, x.device)
    monkeypatch.setattr(pkg, "backward_ctas", lambda name, rows, sm_count: ctas)
    launches.clear()
    for dtype in (torch.bfloat16, torch.float32):
        what = f"{dtype} at {ctas} CTAs"
        dw = torch.empty(D, dtype=dtype, device="cuda")
        dx, dz, _ = backward(dy, x, z, w, EPS, grad_weight=dw, workspace=ws)
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), what
        assert torch.equal(dw, want[dtype][0]), f"dw {what}"
        assert_dw_within(dw, dw64, mag, what=f"dw {what}")
        assert not bool(ws.any()), what
        main = torch.full((D,), 0.25, dtype=dtype, device="cuda")
        backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=main,
            accumulate_grad_weight=True,
            workspace=ws,
        )
        assert torch.equal(main, want[dtype][1]), f"accumulated dw {what}"
        assert_dw_within(main, 0.25 + dw64, mag, what=f"accumulated dw {what}")
        assert not bool(ws.any()), what
        main = torch.full((D,), 3.0, dtype=dtype, device="cuda")
        dx, dz, dw = backward(
            dy, x, z, w, EPS, grad_weight=main, compute_grad_weight=False, workspace=ws
        )
        assert dw is None
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), what
        assert torch.equal(main, torch.full_like(main, 3.0)), what
        assert not bool(ws.any()), what
    assert launches == [
        _backward_module(rows),
        _backward_module(rows),
        _backward_module(rows),
        _backward_module(rows, torch.float32),
        _backward_module(rows, torch.float32),
        _backward_module(rows, torch.float32),
    ], launches


@pytest.mark.parametrize("shape", [(0, 48, D), (5, 0, D), (0, D)])
def test_empty_work_keeps_the_accumulator(shape, launches):
    x = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(D, dtype=torch.bfloat16, device="cuda")
    ws = workspace_for(0, x.device)
    assert forward(x, x, w).shape == x.shape
    for dtype in (torch.bfloat16, torch.float32):
        main = torch.full((D,), 1.5, dtype=dtype, device="cuda")
        dx, dz, dw = backward(
            x, x, x, w, grad_weight=main, accumulate_grad_weight=True, workspace=ws
        )
        assert dw is main and torch.equal(main, torch.full_like(main, 1.5)), dtype
        assert dx.shape == dz.shape == x.shape
        backward(x, x, x, w, grad_weight=main)
        assert torch.equal(main, torch.zeros_like(main)), dtype
    _, _, dw = backward(x, x, x, w)
    assert dw.dtype == torch.bfloat16 and torch.equal(dw, torch.zeros_like(dw))
    assert launches == []
    assert not bool(ws.any())


def test_workspace_reused_across_shapes_without_reset():
    """One zero-filled byte workspace of ``WORKSPACE_BYTES`` serves every
    backward module and row count: each backward on it is within the FP64
    budgets, the bits of the same call on the stream's own workspace, and
    leaves it zero."""
    device = torch.device("cuda")
    ws = workspace_for(1, device)
    assert ws.dtype == torch.uint8 and ws.is_contiguous() and not bool(ws.any())
    assert ws.data_ptr() % 16 == 0
    nbytes = workspace_bytes(1, device)
    assert ws.numel() == nbytes == workspace_bytes(10**7, device) == WORKSPACE_BYTES
    arch = loader.device_arch(torch.cuda.current_device())
    for role, name in loader.ROUTES[arch].items():
        if not role.startswith("bwd"):
            continue
        if role.endswith("_short"):
            assert loader.workspace_layout(name) == (), role
            assert loader.workspace_bytes(name) == 0, role
        else:
            assert loader.workspace_layout(name) == WORKSPACE_LAYOUT, role
            assert loader.workspace_bytes(name) == nbytes, role
    cases = (
        (1031, 48, torch.bfloat16, False),
        (7, 48, torch.float32, False),
        (4097, 16, torch.bfloat16, False),
        (1, 1, torch.float32, True),
        (30193, 48, torch.bfloat16, False),
        (33, 7, torch.float32, True),
    )
    for i, (T, H, dtype, accumulate) in enumerate(cases):
        x, z, w, dy = make(T, H, seed=20 + i)
        start = 0.5 if accumulate else 0.0
        gw = (
            torch.full((D,), start, dtype=dtype, device=device)
            if accumulate or dtype != torch.bfloat16
            else None
        )
        dx, dz, dw = backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=gw,
            accumulate_grad_weight=accumulate,
            workspace=ws,
        )
        torch.cuda.synchronize()
        assert not bool(ws.any()), (T, H, dtype)
        _, dx64, dz64, dw64 = fp64_reference(dy, x, z, w, EPS)
        assert_bf16_close(dx, dx64, what=f"dx T={T} H={H}")
        assert_bf16_close(dz, dz64, what=f"dz T={T} H={H}")
        assert dw.dtype == dtype
        want = dw64 + start
        assert_dw_within(
            dw, want, magnitudes(dy, x, z, EPS), what=f"dw T={T} H={H} {dtype}"
        )
        # the same call on the stream's workspace: the same bits
        gw2 = None if gw is None else torch.full_like(gw, start)
        again = backward(
            dy, x, z, w, EPS, grad_weight=gw2, accumulate_grad_weight=accumulate
        )
        for a, b in zip(again, (dx, dz, dw), strict=True):
            assert torch.equal(a, b), (T, H, dtype)


def _unfit_workspaces(nbytes: int) -> list[torch.Tensor]:
    """Buffers that are not a backward workspace of ``nbytes`` bytes: too
    small, strided, misaligned, of another dtype."""
    big = torch.zeros(nbytes + 64, dtype=torch.uint8, device="cuda")
    return [
        torch.zeros(nbytes - 1, dtype=torch.uint8, device="cuda"),
        torch.zeros(2 * nbytes, dtype=torch.uint8, device="cuda")[::2],
        big[1 : 1 + nbytes],
        big[8 : 8 + nbytes],
        torch.zeros(nbytes, dtype=torch.float64, device="cuda"),
        torch.zeros(nbytes, dtype=torch.int8, device="cuda"),
    ]


def test_workspace_errors():
    x, z, w, dy = make(65, 48, seed=13)
    nbytes = workspace_for(1, x.device).numel()
    # an FP32 buffer (the deterministic package's kind)
    fp32 = torch.zeros(nbytes, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="deterministic mode"):
        backward(dy, x, z, w, EPS, workspace=fp32)
    for bad in _unfit_workspaces(nbytes):
        with pytest.raises(ValueError, match="uint8"):
            backward(dy, x, z, w, EPS, workspace=bad)
        assert not bool(bad.any())


def test_reset_workspace_clears_nonzero_scratch():
    """A caller workspace whose integer windows, class words, arrival ticket
    and claim counter hold garbage (written by something other than the
    backward) is zeroed by ``reset_workspace``; backward launches at two
    different shapes and weight-gradient dtypes then run on it in sequence,
    within the FP64 budget and the bits of the same calls on a fresh
    workspace, and leave it zero."""
    device = torch.device("cuda")
    ws = workspace_for(1, device)
    g = torch.Generator(device="cuda").manual_seed(60)
    ws.copy_(
        torch.randint(1, 256, ws.shape, device=device, generator=g).to(torch.uint8)
    )
    offset = dict((b, o) for b, o, _ in WORKSPACE_LAYOUT)["dw_done"]
    # a stale arrival ticket and a stale claim count
    ws[offset : offset + 8].view(torch.int32).fill_(3)
    assert bool(ws.ne(0).all())
    reset_workspace(ws)
    torch.cuda.synchronize()
    assert not bool(ws.any())
    # 37 x 7 = 259 rows: past the short backward's 256, so both calls use the
    # workspace
    for i, (T, H, dtype) in enumerate(
        ((1031, 48, torch.bfloat16), (37, 7, torch.float32))
    ):
        x, z, w, dy = make(T, H, seed=61 + i)
        gw = (
            None
            if dtype == torch.bfloat16
            else torch.empty(D, dtype=dtype, device=device)
        )
        dx, dz, dw = backward(dy, x, z, w, EPS, grad_weight=gw, workspace=ws)
        _, dx64, dz64, dw64 = fp64_reference(dy, x, z, w, EPS)
        assert_bf16_close(dx, dx64, what=f"dx T={T} H={H}")
        assert_bf16_close(dz, dz64, what=f"dz T={T} H={H}")
        assert dw.dtype == dtype
        assert_dw_within(
            dw, dw64, magnitudes(dy, x, z, EPS), what=f"dw T={T} H={H} {dtype}"
        )
        gw2 = None if gw is None else torch.empty_like(gw)
        fresh = backward(
            dy, x, z, w, EPS, grad_weight=gw2, workspace=workspace_for(1, device)
        )
        for a, b in zip(fresh, (dx, dz, dw), strict=True):
            assert torch.equal(a, b), (T, H)
        torch.cuda.synchronize()
        assert not bool(ws.any()), (T, H)


def test_reset_workspace_of_a_zero_workspace_is_a_no_op(launches):
    """Resetting a zero workspace keeps it zero and in place, launches no module
    and does not synchronize the host; the next backward is unaffected."""
    x, z, w, dy = make(129, 48, seed=62)
    ws = workspace_for(129 * 48, x.device)
    ptr = ws.data_ptr()
    torch.cuda.set_sync_debug_mode("error")
    try:
        reset_workspace(ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert ws.data_ptr() == ptr and not bool(ws.any())
    assert launches == []
    _, _, dw = backward(dy, x, z, w, EPS, workspace=ws)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    assert_dw_within(dw, dw64, mag, what="dw after a reset of a zero workspace")
    torch.cuda.synchronize()
    assert not bool(ws.any())


def test_reset_workspace_errors():
    nbytes = workspace_for(1, torch.device("cuda")).numel()
    with pytest.raises(TypeError):
        reset_workspace(None)
    fp32 = torch.zeros(nbytes, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="deterministic mode"):
        reset_workspace(fp32)
    for bad in _unfit_workspaces(nbytes):
        with pytest.raises(ValueError, match="uint8"):
            reset_workspace(bad)
        assert not bool(bad.any())
    with pytest.raises(ValueError, match="CUDA"):
        reset_workspace(torch.zeros(nbytes, dtype=torch.uint8))
    ws = workspace_for(1, torch.device("cuda"))
    # the mode is in the name: a mode keyword is a TypeError
    for value in (False, True, None):
        with pytest.raises(TypeError, match="deterministic"):
            pkg.cake_rmsnorm_silu_gate_nondet_train_reset_workspace(
                ws, deterministic=value
            )


def test_two_streams_with_their_own_workspaces():
    cases = [make(4097, 48, seed=30), make(1031, 16, seed=31)]
    # dx and dz of one launch on the current stream
    refs = [backward(dy, x, z, w, EPS) for x, z, w, dy in cases]
    references = [
        (fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS))
        for x, z, w, dy in cases
    ]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    workspaces = [workspace_for(1, torch.device("cuda")) for _ in streams]
    outputs: list[list] = [[], []]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for _ in range(5):
        for i, stream in enumerate(streams):
            x, z, w, dy = cases[i]
            with torch.cuda.stream(stream):
                outputs[i].append(backward(dy, x, z, w, EPS, workspace=workspaces[i]))
    torch.cuda.synchronize()
    for i in range(2):
        dx_d, dz_d, dw_d = refs[i]
        dw64, mag = references[i]
        assert_dw_within(dw_d, dw64, mag, what=f"stream {i} dw")
        for dx, dz, dw in outputs[i]:
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), i
            assert torch.equal(dw, dw_d), i
        assert not bool(workspaces[i].any()), i


def test_side_stream_and_no_host_sync():
    x, z, w, dy = make(2049, 48, seed=9)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    ws = workspace_for(x.shape[0] * x.shape[1], x.device)
    # Load both modules before the sync-checked region; the bits of y, dx and dz
    # on the current stream.
    want_y = forward(x, z, w, EPS)
    dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS, workspace=ws)
    assert_dw_within(dw_d, dw64, mag, what="dw")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.cuda.stream(stream):
            y = forward(x, z, w, EPS)
            got = backward(dy, x, z, w, EPS, workspace=ws)
            fresh = backward(dy, x, z, w, EPS)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert torch.equal(y, want_y)
    for dx, dz, dw in (got, fresh):
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
        assert torch.equal(dw, dw_d)
    assert not bool(ws.any())


def test_cuda_graph_capture_and_replay():
    T, H = 300, 48
    x, z, w, dy = make(T, H, seed=10)
    xs, zs, dys = x.clone(), z.clone(), dy.clone()
    y = torch.empty_like(x)
    dx, dz = torch.empty_like(x), torch.empty_like(x)
    dw = torch.empty(D, dtype=torch.bfloat16, device="cuda")
    main = torch.zeros(D, dtype=torch.float32, device="cuda")
    ws = workspace_for(T * H, x.device)

    def step():
        forward(xs, zs, w, EPS, out=y)
        backward(
            dys, xs, zs, w, EPS, grad_x=dx, grad_z=dz, grad_weight=dw, workspace=ws
        )
        # FP32 main-grad accumulation with the workspace allocated in the graph
        backward(dys, xs, zs, w, EPS, grad_weight=main, accumulate_grad_weight=True)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # loads the modules before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    main.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    total = torch.zeros(D, dtype=torch.float64, device="cuda")
    total_mag = torch.zeros(D, dtype=torch.float64, device="cuda")
    main_e = torch.zeros(D, dtype=torch.float32, device="cuda")
    for k in range(3):
        xn, zn, _, dyn = make(T, H, seed=40 + k)
        xs.copy_(xn)
        zs.copy_(zn)
        dys.copy_(dyn)
        graph.replay()
        torch.cuda.synchronize()
        # the replay against eager launches on the same inputs: the same bits
        assert torch.equal(y, forward(xn, zn, w, EPS)), k
        dx_e, dz_e, dw_e = backward(dyn, xn, zn, w, EPS)
        assert torch.equal(dx, dx_e) and torch.equal(dz, dz_e), k
        assert torch.equal(dw, dw_e), k
        backward(dyn, xn, zn, w, EPS, grad_weight=main_e, accumulate_grad_weight=True)
        assert torch.equal(main, main_e), k
        dw64, mag = fp64_reference(dyn, xn, zn, w, EPS)[3], magnitudes(dyn, xn, zn, EPS)
        assert_dw_within(dw, dw64, mag, what=f"replay {k} dw")
        assert not bool(ws.any()), k
        total += dw64
        total_mag += mag
        # three FP32 roundings of the accumulator at most, plus the sums' budget
        err = (main.double() - total).abs()
        assert (err <= total.abs() * 2.0**-21 + total_mag * DW_MAG_TOL).all(), k


def _kept_graph() -> torch.cuda.CUDAGraph:
    try:
        return torch.cuda.CUDAGraph(keep_graph=True)
    except TypeError:
        pytest.skip("listing graph nodes needs torch.cuda.CUDAGraph(keep_graph=True)")


def _graph_node_types(graph: torch.cuda.CUDAGraph) -> list[str]:
    """The node types of a graph captured with ``keep_graph=True``."""
    driver = pytest.importorskip("cuda.bindings.driver")
    handle = driver.CUgraph(graph.raw_cuda_graph())
    _, _, count = driver.cuGraphGetNodes(handle)
    _, nodes, count = driver.cuGraphGetNodes(handle, count)
    return [
        driver.cuGraphNodeGetType(node)[1].name.replace("CU_GRAPH_NODE_TYPE_", "")
        for node in nodes[:count]
    ]


@pytest.fixture
def reserve(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Fresh reserve and capture tables; the reserve of the current device,
    filled and seen ready by eager calls without a workspace."""
    monkeypatch.setattr(pkg, "_RESERVES", {})
    monkeypatch.setattr(pkg, "_CAPTURE_WORKSPACES", {})
    x, z, w, dy = make(17, 48, seed=69)
    backward(dy, x, z, w, EPS)  # loads the module, fills the reserve
    torch.cuda.synchronize()
    backward(dy, x, z, w, EPS)  # sees the fill complete
    torch.cuda.synchronize()
    state = pkg._RESERVES[torch.cuda.current_device()]
    assert len(state["ready"]) == pkg._RESERVE_SIZE and not state["pending"]
    return state


def test_capture_without_workspace_takes_a_reserve_workspace(reserve):
    """Backwards without a caller workspace captured in one graph on one
    stream share one zero-filled workspace taken from the reserve: the graph
    holds the backward kernels only (no fill kernel, memset or event wait), for
    both weight-gradient dtypes; every replay matches eager launches and leaves
    the workspace zero; eager calls do not touch it."""
    pytest.importorskip("cuda.bindings.runtime")
    T, H = 129, 48
    x, z, w, dy = make(T, H, seed=70)
    dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    main = torch.zeros(D, dtype=torch.float32, device="cuda")
    backward(dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True)
    torch.cuda.synchronize()
    main.zero_()
    ready = list(reserve["ready"])

    def step():
        outs = [backward(dy, x, z, w, EPS) for _ in range(2)]
        backward(dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True)
        return outs

    graph = _kept_graph()
    with torch.cuda.graph(graph):
        outs = step()
    types = _graph_node_types(graph)
    assert types == ["KERNEL"] * 3, types
    taken = [ws for ws in ready if all(ws is not r for r in reserve["ready"])]
    assert len(taken) == 1 and len(reserve["ready"]) == len(ready) - 1
    ws = taken[0]
    for k in range(3):
        backward(dy, x, z, w, EPS)  # an eager call between replays
        graph.replay()
        torch.cuda.synchronize()
        for dx, dz, dw in outs:
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), k
            assert torch.equal(dw, dw_d), k
            assert_dw_within(dw, dw64, mag, what=f"replay {k} dw")
        assert not bool(ws.any()), k
        err = (main.double() - (k + 1) * dw64).abs()
        assert (err <= (k + 1) * (dw64.abs() * 2.0**-21 + mag * DW_MAG_TOL)).all(), k


def test_graphs_without_workspace_replay_together_on_two_streams(reserve):
    """Two graphs, each capturing a backward without a caller workspace, own
    two reserve workspaces and replay at the same time on two streams, the
    first replays right after the captures with no host synchronization."""
    cases = [make(4097, 48, seed=71), make(1031, 16, seed=72)]
    refs = [backward(dy, x, z, w, EPS) for x, z, w, dy in cases]
    budgets = [
        (fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS))
        for x, z, w, dy in cases
    ]
    torch.cuda.synchronize()
    graphs, outs, taken = [], [], []
    for x, z, w, dy in cases:
        ready = list(reserve["ready"])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs.append(backward(dy, x, z, w, EPS))
        graphs.append(graph)
        taken += [ws for ws in ready if all(ws is not r for r in reserve["ready"])]
    assert len(taken) == 2 and taken[0].data_ptr() != taken[1].data_ptr()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    results: list[list] = [[], []]
    for _ in range(5):
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                graphs[i].replay()
                results[i].append(tuple(t.clone() for t in outs[i]))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for i in range(2):
        dx_d, dz_d, dw_d = refs[i]
        dw64, mag = budgets[i]
        assert_dw_within(dw_d, dw64, mag, what=f"graph {i} dw")
        for dx, dz, dw in results[i]:
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), i
            assert torch.equal(dw, dw_d), i
        assert not bool(taken[i].any()), i


def test_captured_calls_without_capture_ids_take_one_workspace_each(
    reserve, monkeypatch
):
    """Without cuda-python's capture ids, each captured call takes its own
    reserve workspace; replays stay correct and leave them zero."""
    monkeypatch.setattr(pkg, "_cudart", lambda: None)
    x, z, w, dy = make(129, 48, seed=74)
    dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    torch.cuda.synchronize()
    ready = list(reserve["ready"])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [backward(dy, x, z, w, EPS) for _ in range(3)]
    taken = [ws for ws in ready if all(ws is not r for r in reserve["ready"])]
    assert len(taken) == 3
    for k in range(2):
        graph.replay()
        torch.cuda.synchronize()
        for dx, dz, dw in outs:
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), k
            assert torch.equal(dw, dw_d), k
            assert_dw_within(dw, dw64, mag, what=f"replay {k} dw")
        assert not any(bool(ws.any()) for ws in taken), k


def test_capture_with_no_ready_workspace_fills_in_the_graph(monkeypatch):
    """With no ready reserve workspace a captured backward without a caller
    workspace zero-fills one from the graph's pool inside the graph (a memset
    or fill kernel before the backward kernel) and stays correct."""
    monkeypatch.setattr(pkg, "_RESERVES", {})
    monkeypatch.setattr(pkg, "_CAPTURE_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_RESERVE_SIZE", 0)
    x, z, w, dy = make(129, 48, seed=73)
    dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    torch.cuda.synchronize()
    assert pkg._RESERVES[torch.cuda.current_device()]["blocks"] == []
    graph = _kept_graph()
    with torch.cuda.graph(graph):
        dx, dz, dw = backward(dy, x, z, w, EPS)
    types = _graph_node_types(graph)
    assert types in (["MEMSET", "KERNEL"], ["KERNEL", "KERNEL"]), types
    for k in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), k
        assert torch.equal(dw, dw_d), k
        assert_dw_within(dw, dw64, mag, what=f"replay {k} dw")


def test_autograd_matches_functional_and_checkpoint(launches):
    x, z, w, dy = make(513, 48, seed=8)
    # the functional entry points' bits
    y_ref = forward(x, z, w, EPS)
    dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    assert_dw_within(dw_d, dw64, mag, what="dw vs FP64")

    def check(y, xg, zg, wg, what):
        assert torch.equal(y, y_ref), what
        assert torch.equal(xg, dx_d) and torch.equal(zg, dz_d), what
        assert wg.dtype == torch.bfloat16, what
        assert torch.equal(wg, dw_d), what

    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    y = gated_rmsnorm(xa, za, wa, EPS)
    y.backward(dy)
    check(y, xa.grad, za.grad, wa.grad, "autograd")

    # Saved state: only the inputs, no FP32 temporary.
    saved = []

    def pack(t):
        saved.append((t.dtype, t.numel(), t.data_ptr()))
        return t

    xb, zb, wb = (t.clone().requires_grad_(True) for t in (x, z, w))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        gated_rmsnorm(xb, zb, wb, EPS)
    assert sorted(saved) == sorted(
        [
            (torch.bfloat16, x.numel(), xb.data_ptr()),
            (torch.bfloat16, z.numel(), zb.data_ptr()),
            (torch.bfloat16, D, wb.data_ptr()),
        ]
    )

    # Non-reentrant activation checkpointing replays the forward, in the same
    # mode, in the backward.
    launches.clear()
    xc, zc, wc = (t.clone().requires_grad_(True) for t in (x, z, w))
    yc = torch.utils.checkpoint.checkpoint(
        gated_rmsnorm, xc, zc, wc, EPS, use_reentrant=False
    )
    yc.backward(dy)
    check(yc, xc.grad, zc.grad, wc.grad, "checkpoint")
    fwd = _forward_module(x.numel() // D)
    assert launches == [fwd, fwd, _backward_module(x.numel() // D)], launches


def test_nondeterministic_mode_launches_this_package(launches):
    x, z, w, dy = make(129, 48, seed=14)
    forward(x, z, w, EPS)
    backward(dy, x, z, w, EPS)
    backward(
        dy, x, z, w, EPS, grad_weight=torch.empty(D, dtype=torch.float32, device="cuda")
    )
    fwd = _forward_module(x.numel() // D)
    rows = 129 * 48
    bf16, f32 = _backward_module(rows), _backward_module(rows, torch.float32)
    assert launches == [fwd, bf16, f32], launches
    for name in launches:
        assert name.startswith("cake_rmsnorm_silu_gate_train_"), name
    assert loader.MODULES[bf16]["role"] in ("bwd_bf16", "bwd_bf16_mid")
    assert loader.MODULES[f32]["role"] in ("bwd_f32", "bwd_f32_mid")
    # A mode keyword is a TypeError before any launch.
    launches.clear()
    with pytest.raises(TypeError, match="deterministic"):
        pkg.cake_rmsnorm_silu_gate_nondet_train_forward(
            x, z, w, EPS, deterministic=False
        )
    with pytest.raises(TypeError, match="deterministic"):
        pkg.cake_rmsnorm_silu_gate_nondet_train_backward(
            dy, x, z, w, EPS, deterministic=False
        )
    assert launches == []


def test_empty_and_errors():
    x = torch.empty(0, 48, D, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(D, dtype=torch.bfloat16, device="cuda")
    assert forward(x, x, w).shape == x.shape
    dx, dz, dw = backward(x, x, x, w)
    assert dx.shape == x.shape and torch.equal(dw, torch.zeros_like(dw))
    good = torch.randn(4, 2, D, device="cuda").bfloat16()
    with pytest.raises(ValueError):
        forward(good.float(), good, w)
    with pytest.raises(ValueError):
        forward(good, good[:, :1], w)
    with pytest.raises(ValueError):
        forward(
            torch.randn(4, 64, device="cuda").bfloat16(),
            torch.randn(4, 64, device="cuda").bfloat16(),
            w[:64],
        )
    with pytest.raises(ValueError):
        backward(good, good, good, w, accumulate_grad_weight=True)
    with pytest.raises(ValueError):
        backward(
            good,
            good,
            good,
            w,
            grad_weight=torch.empty(D, dtype=torch.float16, device="cuda"),
        )


def test_registry_records_reference_the_build_manifest():
    """Every role ships one module per architecture (a tier role at most one);
    every module record names its ``build.json`` library (schema
    ``cake-build/1``), whose entries carry the package's kernel and binding
    sources and the precise-arithmetic flags; the static and the claimed
    backward records cut their rows into chunks and carry their workspace
    buffers, the claimed one running past the static one's CTAs per SM; the
    short backward records launch one cluster and take no workspace."""
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert (
        manifest["package"]
        == "flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train"
    )
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    for arch in loader.ARCHES:
        routed = set(loader.ROUTES[arch])
        tiers = (
            loader.XLONG_ROLES
            + loader.LONG_ROLES
            + loader.MID_ROLES
            + loader.SHORT_ROLES
        )
        assert set(loader.ROLES) <= routed <= set(loader.ROLES + tiers)
        for role, name in loader.ROUTES[arch].items():
            record = loader.MODULES[name]
            assert record["role"] == role and arch in record["arches"], (arch, role)
            assert loader.library_name(name, arch) in libraries, (arch, role, name)
    routed = {name for by_role in loader.ROUTES.values() for name in by_role.values()}
    assert set(loader.MODULES) == routed
    for name, record in loader.MODULES.items():
        assert name.startswith("cake_rmsnorm_silu_gate_train_"), name
        assert record["library"] == record["kernel"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["grid_rule"]["kind"] in loader.GRID_KINDS
        assert record["launch"]["cooperative"] is False
        assert record["launch"]["use_pdl"] is False
        clustered = record["grid_rule"]["kind"] == "cluster"
        assert (record["launch"]["cluster"] != (1, 1, 1)) == clustered, name
        plan = [key for _kind, key in record["arg_plan"]]
        if record["role"] in ("fwd", "fwd_short"):
            assert record["grid_rule"]["kind"] == "row_blocks"
            assert record.get("workspace") is None
        elif record["role"] in ("fwd_xlong", "fwd_long", "fwd_mid"):
            assert record["grid_rule"]["kind"] == "persistent_row_blocks"
            assert record.get("workspace") is None
        elif record["role"] in loader.SHORT_ROLES:
            assert record["role"] == "bwd_bf16_short"
            assert clustered
            assert record["launch"]["cluster"] == (record["grid_rule"]["cluster"], 1, 1)
            assert {"dw", "weight_grad"} <= set(plan), name
            assert not {"dw_win", "dw_spec", "dw_done"} & set(plan), name
            assert record.get("workspace") is None
            assert loader.workspace_bytes(name) == 0
        else:
            rule = record["grid_rule"]
            assert rule["kind"] == "chunks", name
            assert rule["rows_per_block"] == 8 and rule["chunk_limit"] == 8192
            assert rule["ctas_per_sm"] == 8
            assert record["launch"]["block"] == (128, 1, 1)
            assert {
                "dw_win",
                "dw_spec",
                "dw_done",
                "dw",
                "weight_grad",
                "blocks_per_chunk",
            } <= set(plan), name
            assert record["workspace"] == {
                "dw_done": 2,
                "dw_spec": D,
                "dw_win": 4 * 7 * D,
            }, name
            assert loader.workspace_bytes(name) == WORKSPACE_BYTES
            if record["role"].endswith("_mid"):
                static = loader.MODULES[
                    loader.ROUTES[record["arches"][0]][record["role"][: -len("_mid")]]
                ]
                assert (
                    rule["more_chunks_than_per_sm"]
                    == (static["grid_rule"]["ctas_per_sm"])
                )
                assert record["launch"] == static["launch"], name
            else:
                assert "more_chunks_than_per_sm" not in rule, name
        for arch in record["arches"]:
            entry = libraries[loader.library_name(name, arch)]
            for flag in ("--prec-div=true", "--prec-sqrt=true", "--ftz=false"):
                assert flag in entry["nvcc_flags"], (name, arch, flag)
            assert "--use_fast_math" not in entry["nvcc_flags"], (name, arch)
            for source in entry["sources"]:
                assert (package / source).is_file(), (name, source)


# (rows, SMs, blocks per chunk, chunks, CTAs) of the static and the claimed
# backward: 8-row blocks, at most 8192 chunks of the fewest consecutive blocks
# each (the deterministic backward's chunks), at most 8 CTAs per SM.
BACKWARD_GRIDS = [
    (1, 148, 1, 1, 1),
    (48, 148, 1, 6, 6),
    (6191, 148, 1, 774, 774),
    (64 * 148, 148, 1, 1184, 1184),  # the static backward's last row count
    (64 * 148 + 1, 148, 1, 1185, 1184),  # the claimed backward's first
    (49488, 148, 1, 6186, 1184),
    (8 * 8192, 148, 1, 8192, 1184),
    (8 * 8192 + 1, 148, 2, 4097, 1184),  # two blocks a chunk
    (32737 * 48, 152, 24, 8185, 1216),
    (1440000, 160, 22, 8182, 1280),
    (64 * 216, 216, 1, 1728, 1728),
    (2**31 - 1, 216, 32768, 8192, 1728),
]


@pytest.mark.parametrize(
    ("rows", "sm_count", "per_chunk", "chunks", "ctas"), BACKWARD_GRIDS
)
def test_backward_cta_rule(rows, sm_count, per_chunk, chunks, ctas):
    """The static and the claimed backward: ``blocks_per_chunk = max(1,
    ceil(blocks / 8192))`` blocks a chunk, ``ceil(blocks / blocks_per_chunk)``
    chunks, ``min(chunks, 8 * SMs)`` CTAs (each running chunks ``b``, ``b +
    grid``, ... in the static backward; claiming runs of chunks in the claimed
    one), so a smaller grid still covers every row; the static backward runs
    while every chunk has a CTA, the claimed one past it."""
    for arch in loader.ARCHES:
        for role in ("bwd_bf16", "bwd_f32"):
            for name in (loader.ROUTES[arch][role], loader.ROUTES[arch][role + "_mid"]):
                record = loader.MODULES[name]
                assert loader.backward_chunks(name, rows) == (per_chunk, chunks)
                got = loader.backward_ctas(name, rows, sm_count)
                assert got == ctas == min(chunks, 8 * sm_count), (name, rows)
                grid = loader.launch_grid(
                    record, rows=rows, sm_count=sm_count, ctas=got
                )
                assert grid == (ctas, 1, 1)
            want = role + ("_mid" if chunks > 8 * sm_count else "")
            if rows <= 256 and role + "_short" in loader.ROUTES[arch]:
                want = role + "_short"
            got = loader.module_for_rows(arch, role, rows, sm_count)
            assert got == loader.ROUTES[arch][want], (arch, role, rows)


# The extra-long, long and mid forward roles each architecture routes, with the
# rows per SM they run from (sm_107a: the extra-long forward from the long
# forward's 1740, no long forward).  The claimed backward (``bwd_*_mid``) runs
# past the static backward's 8 CTAs per SM in chunks, on every architecture.
TIER_ROUTES = {
    "sm_100a": {"fwd_long": 2400, "fwd_mid": 976},
    "sm_103a": {"fwd_xlong": 3000, "fwd_long": 2400, "fwd_mid": 976},
    "sm_107a": {"fwd_xlong": 1740, "fwd_mid": 800},
}
CLAIMED_ROUTES = {"bwd_bf16_mid": 8, "bwd_f32_mid": 8}


# The short roles each architecture routes, with the bound of their rows: rows
# per SM (``max_rows_per_sm``, one count per architecture: the short forward,
# whose grid of one CTA per row block grows with the rows) or rows
# (``max_rows``: the short backward, whose grid is one cluster of 16 CTAs
# whatever the rows; BF16 weight gradients only).
SHORT_ROUTES = {
    "sm_100a": {
        "fwd_short": ("max_rows_per_sm", 12),
        "bwd_bf16_short": ("max_rows", 256),
    },
    "sm_103a": {
        "fwd_short": ("max_rows_per_sm", 12),
        "bwd_bf16_short": ("max_rows", 256),
    },
    "sm_107a": {
        "fwd_short": ("max_rows_per_sm", 59),
        "bwd_bf16_short": ("max_rows", 256),
    },
}


def _short_end(arch, role, sm_count):
    """The most rows ``role``'s short module runs by ``SHORT_ROUTES``, or
    ``None`` where ``arch`` routes none."""
    short = SHORT_ROUTES[arch].get(role + "_short")
    if short is None:
        return None
    key, bound = short
    return bound * sm_count if key == "max_rows_per_sm" else bound


def _tier_module(arch, role, rows, sm_count):
    """The module of ``role`` for ``rows`` rows by ``SHORT_ROUTES``,
    ``TIER_ROUTES`` and ``CLAIMED_ROUTES``: the short module up to its rows
    (per SM or in all), else the extra-long forward from its rows per SM, else
    the long forward from its own, else the mid forward from its own or the
    claimed backward past its chunks per SM, else the role's module."""
    end = _short_end(arch, role, sm_count)
    if end is not None and rows <= end:
        return loader.ROUTES[arch][role + "_short"]
    for tier in ("_xlong", "_long", "_mid"):
        start = TIER_ROUTES[arch].get(role + tier)
        if start is not None and rows >= start * sm_count:
            return loader.ROUTES[arch][role + tier]
    per_sm = CLAIMED_ROUTES.get(role + "_mid")
    if per_sm is not None:
        blocks = -(-rows // 8)
        chunks = -(-blocks // max(1, -(-blocks // 8192)))
        if chunks > per_sm * sm_count:
            return loader.ROUTES[arch][role + "_mid"]
    return loader.route_module(arch, role)


@pytest.mark.parametrize("sm_count", [148, 152, 212, 216])
def test_short_long_and_mid_modules_run_from_their_rows_per_sm(sm_count):
    """``module_for_rows``: a role's short module up to ``max_rows`` rows of
    its grid rule (the short backward) or ``max_rows_per_sm * sm_count`` (the
    short forward), above that the forward's extra-long and long modules from
    ``min_rows_per_sm * sm_count`` rows of their own rule on the architecture,
    below that its mid module from its own rule's rows, the claimed backward
    where the rows have more chunks than ``more_chunks_than_per_sm * sm_count``
    (on the architectures that route them), the role's module between and on
    the others."""
    assert loader.TIERS == ("_xlong", "_long", "_mid")
    assert loader.XLONG_ROLES == ("fwd_xlong",)
    assert loader.LONG_ROLES == ("fwd_long",)
    assert loader.MID_ROLES == ("fwd_mid", "bwd_bf16_mid", "bwd_f32_mid")
    assert loader.SHORT_ROLES == ("fwd_short", "bwd_bf16_short")
    for arch in loader.ARCHES:
        shorts = {}
        for role, name in loader.ROUTES[arch].items():
            if role in loader.SHORT_ROLES:
                rule = loader.MODULES[name]["grid_rule"]
                key = "max_rows" if "max_rows" in rule else "max_rows_per_sm"
                assert ("max_rows" in rule) != ("max_rows_per_sm" in rule), name
                if key == "max_rows_per_sm":
                    shorts[role] = (key, loader.max_rows_per_sm(name, arch))
                else:
                    shorts[role] = (key, rule[key])
        assert shorts == SHORT_ROUTES[arch], arch
        tiers, claimed = {}, {}
        for role, name in loader.ROUTES[arch].items():
            rule = loader.MODULES[name]["grid_rule"]
            if role in CLAIMED_ROUTES:
                assert "min_rows_per_sm" not in rule, name
                claimed[role] = rule["more_chunks_than_per_sm"]
            elif role in loader.XLONG_ROLES + loader.LONG_ROLES + loader.MID_ROLES:
                tiers[role] = loader.min_rows_per_sm(name, arch)
        assert tiers == TIER_ROUTES[arch], arch
        assert claimed == CLAIMED_ROUTES, arch
        for role, name in loader.ROUTES[arch].items():
            assert loader.MODULES[name]["role"] == role
        edges = {1, 48, 255, 256, 257, 8 * 8192, 8 * 8192 + 1, 2**31 - 1}
        for start in (10, 12, 59, 64, 800, 976, 1740, 2400, 3000):
            edges |= {start * sm_count - 1, start * sm_count, start * sm_count + 1}
        for role in loader.ROLES:
            for rows in sorted(edges):
                want = _tier_module(arch, role, rows, sm_count)
                got = loader.module_for_rows(arch, role, rows, sm_count)
                assert got == want, (arch, role, rows)
    # the forward on sm_100a: the short forward up to 12 rows per SM, the
    # forward above, then the mid forward from 976 rows per SM, then the long
    # forward from 2400
    routes = loader.ROUTES["sm_100a"]
    for rows, role in (
        (1, "fwd_short"),
        (12 * sm_count, "fwd_short"),
        (12 * sm_count + 1, "fwd"),
        (976 * sm_count - 1, "fwd"),
        (976 * sm_count, "fwd_mid"),
        (2400 * sm_count - 1, "fwd_mid"),
        (2400 * sm_count, "fwd_long"),
    ):
        assert loader.module_for_rows("sm_100a", "fwd", rows, sm_count) == routes[role]
    # the forward on sm_103a: the long forward from 2400 rows per SM, the
    # extra-long forward from 3000
    routes103 = loader.ROUTES["sm_103a"]
    for rows, role in (
        (2400 * sm_count, "fwd_long"),
        (3000 * sm_count - 1, "fwd_long"),
        (3000 * sm_count, "fwd_xlong"),
        (2**31 - 1, "fwd_xlong"),
    ):
        got = loader.module_for_rows("sm_103a", "fwd", rows, sm_count)
        assert got == routes103[role], (rows, role)
    # the backward on every architecture: the short backward up to 256 rows
    # whatever the SMs (BF16 weight gradient), the static backward up to 64
    # rows per SM (8 chunks of one 8-row block per SM), the claimed backward
    # past it, at every larger row count
    for arch in loader.ARCHES:
        for role in ("bwd_bf16", "bwd_f32"):
            short = "_short" if role == "bwd_bf16" else ""
            for rows, tier in (
                (1, short),
                (256, short),
                (257, ""),
                (64 * sm_count, ""),
                (64 * sm_count + 1, "_mid"),
                (8 * 8192 + 1, "_mid"),
                (2**31 - 1, "_mid"),
            ):
                got = loader.module_for_rows(arch, role, rows, sm_count)
                assert got == loader.ROUTES[arch][role + tier], (arch, role, rows)
    # the forward on sm_107a: the short forward up to 59 rows per SM, the
    # forward above, then the mid forward from 800 rows per SM, then the
    # extra-long forward from 1740 (sm_107a routes no long forward)
    routes = loader.ROUTES["sm_107a"]
    for rows, role in (
        (1, "fwd_short"),
        (10 * sm_count + 1, "fwd_short"),
        (59 * sm_count, "fwd_short"),
        (59 * sm_count + 1, "fwd"),
        (800 * sm_count - 1, "fwd"),
        (800 * sm_count, "fwd_mid"),
        (1740 * sm_count - 1, "fwd_mid"),
        (1740 * sm_count, "fwd_xlong"),
        (2**31 - 1, "fwd_xlong"),
    ):
        assert loader.module_for_rows("sm_107a", "fwd", rows, sm_count) == routes[role]
    assert "fwd_long" not in routes


# (rows, SMs, CTAs) of the long forward: 32-row blocks, the fewest whole waves
# of 3 CTAs per SM whose CTAs walk at most 6 blocks, never more CTAs than blocks.
LONG_FORWARD_GRIDS = [
    (1, 148, 1),
    (32 * 444, 148, 444),  # a block per CTA of one wave
    (32 * 444 * 6, 148, 444),  # one wave, 6 blocks a CTA
    (32 * 444 * 6 + 1, 148, 888),  # a block more: two waves
    (2400 * 148, 148, 2220),  # B200 threshold: 11100 blocks, 5 waves
    (32737 * 48, 148, 8436),  # 49106 blocks, 19 waves
    (2400 * 152, 152, 2280),  # GB300 threshold: 11400 blocks, 5 waves
    (32737 * 48, 152, 8208),  # 18 waves of 456
    (1740 * 212, 212, 2544),  # R200 threshold: 11528 blocks, 4 waves
    (8192 * 48, 212, 2544),  # 12288 blocks, 4 waves of 636
    (2**31 - 1, 148, 25192 * 444),  # 2**26 blocks, 25192 waves
]


@pytest.mark.parametrize(("rows", "sm_count", "ctas"), LONG_FORWARD_GRIDS)
def test_long_forward_grid_is_whole_waves_of_at_most_six_blocks_a_cta(
    rows, sm_count, ctas
):
    for arch in ("sm_100a", "sm_103a"):
        name = loader.ROUTES[arch]["fwd_long"]
        assert name == loader.ROUTES["sm_100a"]["fwd_long"]
    record = loader.MODULES[name]
    assert record["grid_rule"] == {
        "ctas_per_sm": 3,
        "kind": "persistent_row_blocks",
        "max_blocks_per_cta": 6,
        "min_rows_per_sm": {"sm_100a": 2400, "sm_103a": 2400},
        "rows_per_block": 32,
    }
    assert record["arches"] == ["sm_100a", "sm_103a"]
    # sm_107a's extra-long forward launches the same grid
    xlong = loader.MODULES[loader.ROUTES["sm_107a"]["fwd_xlong"]]
    assert loader.launch_grid(xlong, rows=rows, sm_count=sm_count) == (ctas, 1, 1)
    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count)
    assert grid == (ctas, 1, 1)
    blocks, wave = -(-rows // 32), 3 * sm_count
    assert -(-blocks // ctas) <= 6
    assert ctas == blocks or ctas % wave == 0
    assert ctas == blocks or (ctas - wave) * 6 < blocks
    # the grid is the long forward's whatever ``ctas`` the call passes
    assert loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=7) == grid
    # the forward: one 64-thread CTA per 8-row block
    fwd = loader.MODULES[loader.route_module("sm_100a", "fwd")]
    assert fwd["launch"]["block"] == (64, 1, 1)
    assert loader.launch_grid(fwd, rows=rows, sm_count=sm_count) == (
        -(-rows // 8),
        1,
        1,
    )


def test_extra_long_forward_is_the_long_forwards_grid_from_its_rows_per_sm():
    # the extra-long forward: the long forward's grid rule (32-row blocks,
    # whole waves of 3 CTAs per SM walking at most 6 blocks) from 3000 rows per
    # SM on sm_103a and from 1740 on sm_107a (the long forward's start there),
    # 256 threads, no PDL
    name = loader.ROUTES["sm_103a"]["fwd_xlong"]
    assert loader.ROUTES["sm_107a"]["fwd_xlong"] == name
    assert "fwd_xlong" not in loader.ROUTES["sm_100a"]
    record = loader.MODULES[name]
    assert record["arches"] == ["sm_103a", "sm_107a"]
    assert record["role"] == "fwd_xlong"
    long_record = loader.MODULES[loader.ROUTES["sm_103a"]["fwd_long"]]
    assert record["grid_rule"] == {
        **long_record["grid_rule"],
        "min_rows_per_sm": {"sm_103a": 3000, "sm_107a": 1740},
    }
    assert record["launch"]["block"] == (256, 1, 1)
    assert record["launch"]["use_pdl"] is False
    for rows, sm_count in (
        (3000 * 152, 152),
        (32737 * 48, 152),
        (2**31 - 1, 152),
        (1740 * 212, 212),
        (32737 * 48, 212),
        (2**31 - 1, 216),
    ):
        assert loader.launch_grid(
            record, rows=rows, sm_count=sm_count
        ) == loader.launch_grid(long_record, rows=rows, sm_count=sm_count)


# (rows, SMs, CTAs) of the mid forward: 16-row blocks, the fewest whole waves
# of 6 CTAs per SM whose CTAs walk at most 2 blocks, never more CTAs than blocks.
MID_FORWARD_GRIDS = [
    (1, 148, 1),
    (16 * 888, 148, 888),  # a block per CTA of one wave
    (16 * 888 * 2, 148, 888),  # one wave, 2 blocks a CTA
    (16 * 888 * 2 + 1, 148, 1776),  # a block more: two waves
    (976 * 148, 148, 5328),  # B200 threshold: 9028 blocks, 6 waves
    (2400 * 148 - 1, 148, 11544),  # below the long threshold: 22200 blocks, 13 waves
    (4097 * 48, 148, 6216),  # 12291 blocks, 7 waves of 888
    (976 * 152, 152, 5472),  # GB300 threshold: 9272 blocks, 6 waves
    (4097 * 48, 152, 6384),  # 12291 blocks, 7 waves of 912
    (800 * 212, 212, 6360),  # R200 threshold: 10600 blocks, 5 waves
    (1740 * 212 - 1, 212, 12720),  # below its long threshold: 23055 blocks, 10 waves
    (2**31 - 1, 148, 67109712),  # 2**27 blocks, 75574 waves
]


@pytest.mark.parametrize(("rows", "sm_count", "ctas"), MID_FORWARD_GRIDS)
def test_mid_forward_grid_is_whole_waves_of_at_most_two_blocks_a_cta(
    rows, sm_count, ctas
):
    # each architecture's mid forward module (sm_100a's and sm_103a's warm
    # later blocks in L2) has this grid rule with the architecture's rows per SM
    starts = {"sm_100a": 976, "sm_103a": 976, "sm_107a": 800}
    records = []
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        record = loader.MODULES[loader.ROUTES[arch]["fwd_mid"]]
        assert arch in record["arches"]
        assert record["grid_rule"] == {
            "ctas_per_sm": 6,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 2,
            "min_rows_per_sm": {a: starts[a] for a in record["arches"]},
            "rows_per_block": 16,
        }
        assert record["launch"]["block"] == (128, 1, 1)
        records.append(record)
    record = records[0]
    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count)
    for other in records[1:]:
        assert loader.launch_grid(other, rows=rows, sm_count=sm_count) == grid
    assert grid == (ctas, 1, 1)
    blocks, wave = -(-rows // 16), 6 * sm_count
    assert -(-blocks // ctas) <= 2
    assert ctas == blocks or ctas % wave == 0
    assert ctas == blocks or (ctas - wave) * 2 < blocks
    assert loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=7) == grid


# (rows, SMs) of the short forward: one CTA per 16-row block, up to 12 rows per
# SM on sm_100a and sm_103a and 59 on sm_107a (each row count is checked on
# every architecture: past an architecture's bound the forward runs it), so at
# most the SMs less 16 CTAs on sm_100a and sm_103a, and on sm_107a, at the 212
# and 216 SMs listed (the bound needs 205 or more, SHORT_FORWARD_MIN_SMS), at
# most one wave of the 4 CTAs per SM its launch bounds give on the SMs less 16.
SHORT_FORWARD_GRIDS = [
    (1, 148),
    (48, 148),  # T 1, H 48: 3 blocks
    (13 * 7, 148),
    (21 * 48, 148),  # 6.8 rows per SM on B200
    (10 * 148, 148),  # B200 at 10 rows per SM: 93 blocks
    (10 * 152, 152),  # GB300 at 10 rows per SM: 95 blocks
    (12 * 148, 148),  # B200 at 12 rows per SM: 111 blocks
    (12 * 152, 152),  # GB300 at 12 rows per SM: 114 blocks
    (10 * 212, 212),  # R200 at 10 rows per SM: 133 blocks
    (30 * 212, 212),  # R200 at 30 rows per SM: 398 blocks
    (1786 * 7, 212),  # T 1786, H 7: 782 blocks, 58.97 rows per SM
    (59 * 212, 212),  # R200 at 59 rows per SM: 782 blocks
    (59 * 216, 216),  # 216-SM R200 at 59 rows per SM: 797 blocks
]
# The most CTAs of the short forward's grid on sm_count SMs, from
# SHORT_FORWARD_MIN_SMS SMs (64 on sm_100a and sm_103a, 205 on sm_107a; on
# fewer SMs, as on a small MIG instance, the grid at the architecture's
# threshold has more): at most the SMs less 16 on sm_100a and sm_103a, one
# wave of 4 CTAs per SM on the SMs less 16 on sm_107a.
SHORT_FORWARD_MAX_CTAS = {
    "sm_100a": lambda sm_count: sm_count - 16,
    "sm_103a": lambda sm_count: sm_count - 16,
    "sm_107a": lambda sm_count: 4 * (sm_count - 16),
}
# The fewest SMs on which the short forward's grid at the architecture's
# threshold keeps SHORT_FORWARD_MAX_CTAS: ceil(12 * SMs / 16) <= SMs - 16 from
# 64 SMs on sm_100a and sm_103a, ceil(59 * SMs / 16) <= 4 * (SMs - 16) from 205
# SMs on sm_107a (212 and 216 keep it; a small MIG instance does not).
SHORT_FORWARD_MIN_SMS = {"sm_100a": 64, "sm_103a": 64, "sm_107a": 205}


@pytest.mark.parametrize(("rows", "sm_count"), SHORT_FORWARD_GRIDS)
def test_short_forward_grid_is_one_cta_a_block_beside_16_held_sms(rows, sm_count):
    name = loader.ROUTES["sm_100a"]["fwd_short"]
    routed = []
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        assert loader.ROUTES[arch]["fwd_short"] == name
        end = SHORT_ROUTES[arch]["fwd_short"][1]
        assert loader.max_rows_per_sm(name, arch) == end
        want = loader.route_module(arch, "fwd")
        if rows <= end * sm_count:
            routed.append(arch)
            want = name
        assert loader.module_for_rows(arch, "fwd", rows, sm_count) == want
    assert routed, (rows, sm_count)
    record = loader.MODULES[name]
    assert record["grid_rule"] == {
        "kind": "row_blocks",
        "max_rows_per_sm": {"sm_100a": 12, "sm_103a": 12, "sm_107a": 59},
        "rows_per_cta": 16,
    }
    assert record["launch"]["block"] == (256, 1, 1)
    assert record["launch"]["dynamic_smem_bytes"] == 0
    assert record["arches"] == ["sm_100a", "sm_103a", "sm_107a"]
    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count)
    assert grid == (-(-rows // 16), 1, 1)
    for arch in routed:
        assert grid[0] <= SHORT_FORWARD_MAX_CTAS[arch](sm_count), (arch, grid)
    assert loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=7) == grid
    # the forward it replaces: one 64-thread CTA per 8-row block
    fwd = loader.MODULES[loader.route_module("sm_100a", "fwd")]
    assert loader.launch_grid(fwd, rows=rows, sm_count=sm_count) == (
        -(-rows // 8),
        1,
        1,
    )


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_short_forward_grid_bound_holds_from_its_fewest_sms(arch):
    name = loader.ROUTES[arch]["fwd_short"]
    record = loader.MODULES[name]
    end = loader.max_rows_per_sm(name, arch)
    for sm_count in range(1, 1025):
        grid = loader.launch_grid(record, rows=end * sm_count, sm_count=sm_count)
        fits = grid[0] <= SHORT_FORWARD_MAX_CTAS[arch](sm_count)
        assert fits == (sm_count >= SHORT_FORWARD_MIN_SMS[arch]), (
            arch,
            sm_count,
            grid,
        )


def test_short_forward_grid_at_its_threshold_keeps_the_bound_on_this_device():
    """On the running device, the short forward's grid at the architecture's
    threshold (the most rows it runs) keeps SHORT_FORWARD_MAX_CTAS; skipped
    where the device has fewer SMs than the bound needs."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    sm_count = loader.device_sm_count(device_index)
    name = loader.ROUTES[arch].get("fwd_short")
    if name is None:
        pytest.skip(f"{arch} routes no fwd_short module")
    if sm_count < SHORT_FORWARD_MIN_SMS[arch]:
        pytest.skip(
            f"{sm_count} SMs: the short forward's grid bound on {arch} needs "
            f"{SHORT_FORWARD_MIN_SMS[arch]} SMs or more"
        )
    rows = loader.max_rows_per_sm(name, arch) * sm_count
    assert loader.module_for_rows(arch, "fwd", rows, sm_count) == name
    assert loader.module_for_rows(arch, "fwd", rows + 1, sm_count) != name
    grid = loader.launch_grid(loader.MODULES[name], rows=rows, sm_count=sm_count)
    assert grid == (-(-rows // 16), 1, 1)
    assert grid[0] <= SHORT_FORWARD_MAX_CTAS[arch](sm_count), (arch, sm_count, grid)


# (rows, SMs) of the short backward: one cluster of 16 CTAs whatever the rows
# and the SMs, up to 256 rows, so no CTA takes more than one 16-row block.
SHORT_BACKWARD_GRIDS = [
    (1, 148),
    (48, 148),  # T 1, H 48
    (13 * 7, 152),
    (5 * 48, 148),
    (256, 148),
    (256, 152),
    (256, 160),
    (256, 212),
]


@pytest.mark.parametrize(("rows", "sm_count"), SHORT_BACKWARD_GRIDS)
def test_short_backward_is_one_cluster_up_to_256_rows(rows, sm_count):
    """The BF16 weight gradient's short backward: one cluster of 16 CTAs up to
    256 rows, no workspace; the FP32 weight gradient runs the static backward
    at every row count up to 64 rows per SM."""
    name = loader.ROUTES["sm_100a"]["bwd_bf16_short"]
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        assert loader.ROUTES[arch]["bwd_bf16_short"] == name
        assert "bwd_f32_short" not in loader.ROUTES[arch]
        assert loader.module_for_rows(arch, "bwd_bf16", rows, sm_count) == name
        base = loader.route_module(arch, "bwd_bf16")
        assert loader.module_for_rows(arch, "bwd_bf16", 257, sm_count) == base
        f32 = loader.route_module(arch, "bwd_f32")
        assert loader.module_for_rows(arch, "bwd_f32", rows, sm_count) == f32
    record = loader.MODULES[name]
    assert record["grid_rule"] == {
        "cluster": 16,
        "kind": "cluster",
        "max_rows": 256,
    }
    assert record["launch"]["block"] == (256, 1, 1)
    assert record["launch"]["cluster"] == (16, 1, 1)
    assert record["launch"]["dynamic_smem_bytes"] == 6144
    assert record["arches"] == ["sm_100a", "sm_103a", "sm_107a"]
    assert loader.backward_ctas(name, rows, sm_count) == 16
    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=16)
    assert grid == (16, 1, 1)
    assert -(-rows // 16) <= grid[0]
    assert loader.workspace_bytes(name) == 0
    with pytest.raises(ValueError, match="chunks"):
        loader.backward_chunks(name, rows)


# Short-backward shapes (T, H, layout): T * H rows up to 256 in every layout,
# and the first row counts past it.
SHORT_BACKWARD_SHAPES = [
    (1, 1, "contiguous"),
    (1, 48, "contiguous"),
    (2, 48, "contiguous"),
    (5, 48, "contiguous"),
    (13, 7, "contiguous"),
    (8, 32, "contiguous"),
    (16, 16, "contiguous"),
    (256, 1, "flat"),
    (5, 48, "projection"),
    (13, 7, "head_major"),
    (2, 48, "misaligned"),
    (257, 1, "flat"),
    (37, 7, "contiguous"),
]


def _short_backward(rows: int, dw_dtype: torch.dtype) -> tuple[str, str | None]:
    """``(the module module_for_rows runs, the short backward module or None)``
    for ``rows`` rows and a ``dw_dtype`` weight gradient on this device."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    name = loader.module_for_rows(
        arch, role, rows, loader.device_sm_count(device_index)
    )
    return name, (name if name == loader.ROUTES[arch].get(role + "_short") else None)


@pytest.mark.parametrize("T,H,layout", SHORT_BACKWARD_SHAPES)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_short_backward_dw_is_the_same_bits_and_takes_no_workspace(
    T, H, layout, dw_dtype, launches, monkeypatch
):
    """Within the FP64 budgets and the same bits on every launch
    (``check_all``), the module ``module_for_rows`` names launched; on the
    short backward (up to 256 rows, BF16 weight gradient) no stream workspace
    is made, a caller workspace holding anything is left as it is (the results
    the same bits) and one that is not a workspace is refused."""
    monkeypatch.setattr(pkg, "_STREAM_WORKSPACES", {})
    x, z, w, dy = make(T, H, layout=layout, seed=80)
    name, short = _short_backward(T * H, dw_dtype)
    check_all(x, z, w, dy, EPS, dw_dtype=dw_dtype, repeats=5)
    assert launches == [_forward_module(T * H), name, name] * 5, launches
    if short is None:
        return
    assert pkg._STREAM_WORKSPACES == {}

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x.device)

    want = backward(dy, x, z, w, EPS, grad_weight=out())
    ws = workspace_for(T * H, x.device)
    ws.fill_(7)
    got = backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
    torch.cuda.synchronize()
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)
    assert bool((ws == 7).all()), "the short backward wrote the workspace"
    with pytest.raises(ValueError, match="deterministic mode"):
        backward(dy, x, z, w, EPS, workspace=ws.float())
    with pytest.raises(ValueError, match="uint8"):
        backward(dy, x, z, w, EPS, workspace=ws[1:])
    assert pkg._STREAM_WORKSPACES == {}


def test_short_backward_accumulates_and_skips_the_weight_gradient():
    """On the short backward (BF16 weight gradient): ``accumulate_grad_weight``
    adds to the accumulator before the one rounding (within the budget, the
    same bits on every launch), ``compute_grad_weight=False`` leaves
    ``grad_weight`` as it is, and ``dx`` and ``dz`` are the same bits either
    way."""
    dw_dtype = torch.bfloat16
    x, z, w, dy = make(5, 48, seed=81)
    _name, short = _short_backward(5 * 48, dw_dtype)
    if short is None:
        pytest.skip("this architecture routes no short backward")
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    first = None
    for _ in range(3):
        main = torch.full((D,), 0.25, dtype=dw_dtype, device="cuda")
        dx, dz, dw = backward(
            dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True
        )
        assert dw is main
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
        first = main if first is None else first
        assert torch.equal(main, first)
    assert_dw_within(main, 0.25 + dw64, mag, what=f"accumulated dw {dw_dtype}")
    main = torch.full((D,), 3.0, dtype=dw_dtype, device="cuda")
    dx, dz, dw = backward(dy, x, z, w, EPS, grad_weight=main, compute_grad_weight=False)
    torch.cuda.synchronize()
    assert dw is None
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
    assert torch.equal(main, torch.full_like(main, 3.0))


def test_calls_without_workspace_share_one_buffer_per_stream():
    x, z, w, dy = make(65, 7)
    for _ in range(2):
        backward(dy, x, z, w, 1e-6)
    torch.cuda.synchronize()
    stream = torch.cuda.current_stream().cuda_stream
    held = [v for k, v in pkg._STREAM_WORKSPACES.items() if k[1] == stream]
    assert held, "a call without a workspace keeps one per stream"
    for views in held:
        assert sorted(views) == ["dw_done", "dw_spec", "dw_win"]
        assert views["dw_win"].dtype == torch.uint64
        assert views["dw_spec"].dtype == torch.float32
        assert views["dw_done"].dtype == torch.uint32
        for view in views.values():
            assert not bool(view.view(torch.uint8).any())
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        backward(dy, x, z, w, 1e-6)
    side.synchronize()
    assert any(k[1] == side.cuda_stream for k in pkg._STREAM_WORKSPACES)


def test_checkpointed_steps_keep_only_the_stream_workspace(monkeypatch):
    """FlashInfer issue #7039: a non-reentrant checkpointed forward keeps nothing
    alive but its output, and once the first step has run, every checkpointed
    forward and backward leaves ``torch.cuda.memory_allocated()`` unchanged.  The
    buffers kept across steps are the workspace of the backward without a
    caller workspace, one per (device, stream, backward module) of
    ``workspace_bytes`` bytes, and the device's reserve of zero-filled
    workspaces for CUDA Graph captures (one allocation)."""
    monkeypatch.setattr(pkg, "_STREAM_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_RESERVES", {})
    # Small enough for the caching allocator's small pool, whose blocks are the
    # requested sizes rounded up to 512 bytes.
    x, z, w, dy = make(17, 48, seed=63)
    leaves = [t.clone().requires_grad_(True) for t in (x, z, w)]
    for leaf in leaves:
        leaf.grad = torch.zeros_like(leaf)

    def checkpointed():
        return torch.utils.checkpoint.checkpoint(
            gated_rmsnorm, *leaves, EPS, use_reentrant=False
        )

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    checkpointed().backward(dy)  # loads the modules, allocates the workspace
    torch.cuda.synchronize()
    kept = torch.cuda.memory_allocated() - before
    nbytes = workspace_bytes(1, x.device)
    key = (
        torch.cuda.current_device(),
        torch.cuda.current_stream().cuda_stream,
        _backward_module(17 * 48),
    )
    assert list(pkg._STREAM_WORKSPACES) == [key]
    for view in pkg._STREAM_WORKSPACES[key].values():
        assert view.untyped_storage().nbytes() == nbytes
    (block,) = pkg._RESERVES[torch.cuda.current_device()]["blocks"]
    reserved = block.untyped_storage().nbytes()
    # 256-byte aligned workspaces: 29440 bytes each, about 1.8 MiB in all
    assert reserved == pkg._RESERVE_SIZE * 29440, reserved
    assert nbytes + reserved <= kept < nbytes + reserved + 512, (kept, nbytes)
    base = torch.cuda.memory_allocated()
    for step in range(3):
        y = checkpointed()
        torch.cuda.synchronize()
        retained = torch.cuda.memory_allocated() - base - y.untyped_storage().nbytes()
        assert retained == 0, (step, retained)
        y.backward(dy)
        del y
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == base, step
    assert list(pkg._STREAM_WORKSPACES) == [key]


# Values a caller might pass as ``deterministic``.
MODES = (True, False, None, 0, 1, "False", numpy.bool_(False))


def test_no_entry_point_takes_a_mode_keyword(launches):
    """The mode is in the name (``nondet``): no entry point of the package has a
    ``deterministic`` parameter, and passing one is a ``TypeError`` before any
    launch, whatever its value and PyTorch's deterministic-algorithms setting;
    the autograd function takes no fifth input."""
    x, z, w, dy = make(5, 48, seed=64)
    device = torch.device("cuda")
    ws = workspace_for(1, device)
    calls = {
        "cake_rmsnorm_silu_gate_nondet_train_forward": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet_train_forward(x, z, w, EPS, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet_train_backward(dy, x, z, w, EPS, **m),
        "cake_rmsnorm_silu_gate_nondet": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet(x, z, w, EPS, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet_train_backward_workspace(1, device, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(1, device, **m),
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws, **m),
        "is_cake_rmsnorm_silu_gate_nondet_train_supported": lambda **m: pkg.is_cake_rmsnorm_silu_gate_nondet_train_supported(device, **m),
    }  # fmt: skip
    assert set(calls) | {"CakeRMSNormSiLUGateNondetFunction"} == set(pkg.__all__) - {
        "ARCHES",
        "SUPPORTED_COMPUTE_CAPABILITIES",
    }
    for name in calls:
        assert "deterministic" not in inspect.signature(getattr(pkg, name)).parameters
    assert list(
        inspect.signature(pkg.CakeRMSNormSiLUGateNondetFunction.forward).parameters
    ) == ["ctx", "x", "z", "weight", "eps"]
    for enabled, warn_only in ((False, False), (True, False), (True, True)):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        for call in calls.values():
            for value in MODES:
                with pytest.raises(TypeError, match="deterministic"):
                    call(deterministic=value)
        for value in MODES:
            with pytest.raises(TypeError):
                pkg.CakeRMSNormSiLUGateNondetFunction.apply(x, z, w, EPS, value)
    assert launches == []
    torch.cuda.synchronize()
    assert not bool(ws.any())


def _forward_rows() -> dict[str, int]:
    """One row count of each forward module this device routes:
    ``{module: rows}``."""
    sm_count = loader.device_sm_count(torch.cuda.current_device())
    found: dict[str, int] = {}
    for rows_per_sm in (1, 12, 59, 100, 900, 1200, 2600, 3200):
        rows = rows_per_sm * sm_count
        found.setdefault(_forward_module(rows), rows)
    return found


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_no_entry_point_alerts_under_deterministic_algorithms(warn_only, launches):
    """Every output repeats bit for bit, so under
    ``torch.use_deterministic_algorithms(True)`` (strict and ``warn_only``)
    with every warning an error, every entry point runs without an alert and
    returns the bits it returns with the setting off, launching the same
    modules: the forward on every forward module this device routes; the
    short, the static and the claimed backward (BF16 and FP32 weight
    gradients, accumulation, ``compute_grad_weight=False``, no rows); the
    workspace entry points, the reset and the support query; the autograd
    backward, also of a node built before the setting was enabled, and
    non-reentrant checkpointing."""
    forward_rows = _forward_rows()
    assert len(forward_rows) >= 3, forward_rows
    cases = [make(T, 48, seed=69 + i) for i, T in enumerate((5, 129, 300))]
    empty = torch.empty(0, 48, D, dtype=torch.bfloat16, device="cuda")
    inputs = {rows: make(rows, 1, seed=72) for rows in forward_rows.values()}
    ws = workspace_for(1, empty.device)

    def leaves(x, z, w, weight_grad=True):
        return [
            x.clone().requires_grad_(True),
            z.clone().requires_grad_(True),
            w.clone().requires_grad_(weight_grad),
        ]

    def run(nodes):
        out = [forward(x, z, w, EPS) for x, z, w, _ in inputs.values()]
        for x, z, w, dy in cases:
            out += backward(dy, x, z, w, EPS)
            out += backward(dy, x, z, w, EPS, grad_weight=torch.empty(D, device="cuda"))
            main = torch.full((D,), 0.25, dtype=torch.float32, device="cuda")
            out += backward(
                dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True
            )
            out += backward(dy, x, z, w, EPS, compute_grad_weight=False)[:2]
            out += backward(dy, x, z, w, EPS, workspace=ws)
        out += backward(empty, empty, empty, cases[0][2], EPS)
        assert workspace_bytes(1, empty.device) == WORKSPACE_BYTES
        assert workspace_for(1, empty.device).numel() == WORKSPACE_BYTES
        reset_workspace(ws)
        assert pkg.is_cake_rmsnorm_silu_gate_nondet_train_supported(empty.device)
        for (x, z, w, dy), node in zip(cases, nodes, strict=True):
            y, grads = node
            y.backward(dy)
            out += [y, *(t.grad for t in grads)]
            ts = leaves(x, z, w)
            yc = torch.utils.checkpoint.checkpoint(
                gated_rmsnorm, *ts, EPS, use_reentrant=False
            )
            yc.backward(dy)
            out += [yc, *(t.grad for t in ts)]
            ts = leaves(x, z, w, weight_grad=False)
            gated_rmsnorm(*ts, EPS).backward(dy)
            out += [ts[0].grad, ts[1].grad]
            assert ts[2].grad is None
        return out

    def nodes():
        built = []
        for x, z, w, _ in cases:
            grads = leaves(x, z, w)
            built.append((gated_rmsnorm(*grads, EPS), grads))
        return built

    want = run(nodes())
    torch.cuda.synchronize()
    want_launches = list(launches)
    launches.clear()
    built = nodes()  # with the setting off
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = run(built)
        torch.cuda.synchronize()
    assert len(got) == len(want)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert torch.equal(a, b), i
    assert launches == want_launches, launches
    modules = set(launches)
    for x, *_ in cases:
        rows = x.numel() // D
        assert {
            _backward_module(rows),
            _backward_module(rows, torch.float32),
        } <= modules
    roles = {loader.MODULES[name]["role"] for name in modules}
    assert {"bwd_bf16", "bwd_f32"} <= roles, roles
    assert not bool(ws.any())


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_graph_capture_under_deterministic_algorithms(warn_only):
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and
    ``warn_only``) with every warning an error, a CUDA Graph captures the
    forward and the backward (caller workspace and none, BF16 and FP32 weight
    gradients) without an alert; its replays return the bits of eager calls,
    the workspace zero after each."""
    T, H = 300, 48
    x, z, w, dy = make(T, H, seed=75)
    ws = workspace_for(T * H, x.device)
    y = torch.empty_like(x)
    dx, dz = torch.empty_like(x), torch.empty_like(x)
    dw = torch.empty(D, dtype=torch.bfloat16, device="cuda")
    main = torch.empty(D, dtype=torch.float32, device="cuda")

    def step():
        forward(x, z, w, EPS, out=y)
        backward(dy, x, z, w, EPS, grad_x=dx, grad_z=dz, grad_weight=dw, workspace=ws)
        backward(dy, x, z, w, EPS, grad_weight=main)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # loads the modules before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    want = [t.clone() for t in (y, dx, dz, dw, main)]
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for k in range(3):
            for t in (y, dx, dz, dw, main):
                t.fill_(7.0)
            graph.replay()
            torch.cuda.synchronize()
            for i, (a, b) in enumerate(zip((y, dx, dz, dw, main), want, strict=True)):
                assert torch.equal(a, b), (k, i)
            assert not bool(ws.any()), k


def test_deterministic_workspace_is_refused_naming_both_modes():
    """The backward (past the short backward and on it, which checks a caller
    workspace too) and the reset refuse an FP32 tensor, the deterministic
    package's workspace, naming both modes' workspaces, and leave it as it is;
    any other unfit buffer gets the byte-workspace requirement."""
    device = torch.device("cuda")
    nbytes = workspace_for(1, device).numel()
    fp32 = torch.zeros(8192, dtype=torch.float32, device="cuda")
    want = (
        "workspace is FP32, a workspace of the deterministic mode "
        f"({DET}.cake_rmsnorm_silu_gate_det_train_backward_workspace); the "
        "non-deterministic backward needs its own zero-filled uint8 workspace "
        f"({pkg.__name__}.cake_rmsnorm_silu_gate_nondet_train_backward_workspace)"
    )
    for T in (65, 5):
        x, z, w, dy = make(T, 48, seed=73)
        for dtype in (torch.bfloat16, torch.float32):
            with pytest.raises(ValueError) as info:
                backward(
                    dy,
                    x,
                    z,
                    w,
                    EPS,
                    grad_weight=torch.empty(D, dtype=dtype, device="cuda"),
                    workspace=fp32,
                )
            assert str(info.value) == want, (T, dtype)
    with pytest.raises(ValueError) as info:
        reset_workspace(fp32)
    assert str(info.value) == want
    torch.cuda.synchronize()
    assert not bool(fp32.any())
    x, z, w, dy = make(65, 48, seed=74)
    device = x.device
    for bad in _unfit_workspaces(nbytes):
        with pytest.raises(ValueError) as info:
            backward(dy, x, z, w, EPS, workspace=bad)
        assert str(info.value) == (
            f"workspace must be a contiguous torch.uint8 tensor on {device} with at "
            f"least {nbytes} bytes starting at a 16-byte aligned address, zero "
            "before its first use "
            "(cake_rmsnorm_silu_gate_nondet_train_backward_workspace(rows, device), "
            "or cake_rmsnorm_silu_gate_nondet_train_reset_workspace(workspace))"
        )
