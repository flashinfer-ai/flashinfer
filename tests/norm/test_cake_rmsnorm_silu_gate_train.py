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

Backend suite of the Cake SiLU-gated RMSNorm training package with the opt-in
non-deterministic weight gradient (``flashinfer.cake_rmsnorm_silu_gate_train_backend``,
the ``deterministic=False`` path of :mod:`flashinfer.cake_rmsnorm_silu_gate_train`):
``RMSNorm(x) * w * SiLU(z)``.

``y``, ``dx`` and ``dz`` must lie within one BF16 ulp of an independent FP64
reference, and ``dx`` and ``dz`` must be the same bits on every launch and on
every path (streams, CUDA Graph replays, autograd, ``compute_grad_weight``,
grids smaller than the row blocks).  The weight gradient, whose FP32 runs and
FP64 cross-CTA sum depend on which row blocks each CTA claims, is held to the
deterministic suite's budget, ``|dw - dw64| <= |dw64| * 2^-7 + sum|terms| *
2^-20``, worst case over repeated launches.  The matrix covers dynamic token and
head counts (``H = 48`` and small or irregular row counts), ``[T, H, 128]``
strided and flattened layouts, near-zero variance, saturated gates, FP32 and
BF16 weight gradients, accumulation, ``compute_grad_weight=False``, grids of 1,
3 and 7 CTAs (every CTA claims), empty work, a workspace reused across shapes
(zero after every call), a workspace with nonzero scratch returned to zero by
``cake_rmsnorm_silu_gate_train_reset_workspace``, two streams with their own
workspaces, CUDA Graph capture and replay (captured calls without a
workspace: one workspace per capture and capture stream, zero-filled outside
the graph, no fill kernel in it; two such graphs replayed at the same time on
two streams; the fill inside the graph past the graph-workspace limit), the
autograd wrapper (non-reentrant checkpointing included), the state a
checkpointed step keeps (FlashInfer issue #7039) and the mode policy: every
compute and workspace entry point refuses any ``deterministic`` but ``False``,
and the backward alerts under ``torch.use_deterministic_algorithms(True)``.

This suite exercises the package alone, named once through
``tests.test_helpers.cake_package`` (so a vendored copy runs it unchanged), and
imports nothing else from flashinfer.  The comparisons against the
deterministic package (``y``, ``dx`` and ``dz`` bitwise, ``dw`` within the
budget of the deterministic ``dw``), the facade, the ``flashinfer`` top level,
the default (deterministic) dispatch and the facade's interplay with
``torch.use_deterministic_algorithms`` are
``test_cake_rmsnorm_silu_gate_train_api.py``'s.
"""

from __future__ import annotations

import functools
import warnings
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

from tests.test_helpers.cake_package import cake_package

KEY = "cake_rmsnorm_silu_gate_train"
pkg = cake_package(KEY)
cake_build = cake_package(KEY, "cake_build")
loader = cake_package(KEY, "cake_jit")
forward = functools.partial(
    pkg.cake_rmsnorm_silu_gate_train_forward, deterministic=False
)
backward = functools.partial(
    pkg.cake_rmsnorm_silu_gate_train_backward, deterministic=False
)
workspace_for = functools.partial(
    pkg.cake_rmsnorm_silu_gate_train_backward_workspace, deterministic=False
)
workspace_bytes = functools.partial(
    pkg.cake_rmsnorm_silu_gate_train_backward_workspace_bytes, deterministic=False
)
reset_workspace = functools.partial(
    pkg.cake_rmsnorm_silu_gate_train_reset_workspace, deterministic=False
)
gated_rmsnorm = functools.partial(pkg.cake_rmsnorm_silu_gate, deterministic=False)

D = 128
EPS = 1e-6
# One BF16 unit in the last place relative to the value (8 significant bits).
BF16_RTOL = 2.0**-7
# The FP32 accumulation term of the weight-gradient budget, relative to the sum
# of the terms' magnitudes (the deterministic suite's).
DW_MAG_TOL = 2.0**-20
REPEATS = 10
# The refusal of every entry point for a ``deterministic`` other than ``False``.
REFUSAL = "computes non-deterministically and takes deterministic=False"
# PyTorch's alert of the backward under its deterministic algorithms.
ALERT = "does not have a deterministic implementation"

# The accuracy matrix; test_cake_rmsnorm_silu_gate_train_api.py runs the same
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
    return torch.cuda.is_available() and pkg.is_cake_rmsnorm_silu_gate_train_supported(
        torch.device("cuda"), deterministic=False
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)


@pytest.fixture(autouse=True)
def nondeterministic_allowed():
    """Run with ``torch.use_deterministic_algorithms`` off; restore the caller's setting."""
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


def check_all(x, z, w, dy, eps, *, dw_dtype=torch.bfloat16, repeats=REPEATS) -> int:
    """Forward and ``repeats`` backward launches against the FP64 reference,
    ``dx`` and ``dz`` the same bits on every launch; returns the number of
    distinct ``dw`` bit patterns over the launches."""
    y = forward(x, z, w, eps)
    assert y.dtype == torch.bfloat16 and y.shape == x.shape
    y64, dx64, dz64, dw64 = fp64_reference(dy, x, z, w, eps)
    mag = magnitudes(dy, x, z, eps)
    assert_bf16_close(y, y64, what="y vs FP64")

    def out():
        return (
            None
            if dw_dtype == torch.bfloat16
            else torch.empty(D, dtype=dw_dtype, device=x.device)
        )

    patterns = set()
    first = None
    for _ in range(repeats):
        dx, dz, dw = backward(dy, x, z, w, eps, grad_weight=out())
        assert dx.dtype == dz.dtype == torch.bfloat16 and dw.dtype == dw_dtype
        assert dx.shape == dz.shape == x.shape and dw.shape == (D,)
        if first is None:
            first = dx, dz
        assert torch.equal(dx, first[0]), "dx vs the first launch"
        assert torch.equal(dz, first[1]), "dz vs the first launch"
        assert_dw_within(dw, dw64, mag, what="dw vs FP64")
        patterns.add(bytes(dw.view(torch.uint8).cpu().numpy()))
    assert_bf16_close(dz, dz64, what="dz vs FP64")
    assert_bf16_close(dx, dx64, what="dx vs FP64")
    return len(patterns)


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
    patterns = check_all(x, z, w, dy, EPS)
    print(f"DW_PATTERNS T={T} H=48 repeats={REPEATS} distinct={patterns}")


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
        main = torch.full((D,), 0.25, dtype=dtype, device="cuda")
        for args in ((dy, x, z), (dy2, x2, z2)):
            backward(*args, w, EPS, grad_weight=main, accumulate_grad_weight=True)
        assert ((main.double() - want).abs() <= bound).all(), dtype
        if dtype == torch.float32:
            assert torch.allclose(main.double(), want, rtol=1e-5, atol=1e-3)
    # autograd accumulation into .grad over two microbatches
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    gated_rmsnorm(xa, za, wa, EPS).backward(dy)
    gated_rmsnorm(xa, za, wa, EPS).backward(dy)
    dx1, dz1, _ = backward(dy, x, z, w, EPS)
    assert torch.equal(xa.grad, dx1 + dx1)
    assert torch.equal(za.grad, dz1 + dz1)
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
    # without a workspace: the launch still takes the arrival ticket and claims
    # row blocks, on the stream's workspace of the BF16 module, left zero
    monkeypatch.setattr(pkg, "_STREAM_WORKSPACES", {})
    dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
    assert dw is None
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
    torch.cuda.synchronize()
    key = (
        torch.cuda.current_device(),
        torch.cuda.current_stream().cuda_stream,
        _routes()["bwd_bf16"],
    )
    assert list(pkg._STREAM_WORKSPACES) == [key]
    sums, done = pkg._STREAM_WORKSPACES[key]
    assert not bool(sums.any()) and done.view(torch.int32).tolist() == [0, 0]
    # autograd with a weight that needs no gradient
    xa, za = (t.clone().requires_grad_(True) for t in (x, z))
    gated_rmsnorm(xa, za, w, EPS).backward(dy)
    assert torch.equal(xa.grad, dx_d) and torch.equal(za.grad, dz_d)
    assert w.grad is None


@pytest.mark.parametrize("ctas", [1, 3, 7])
def test_backward_at_grids_smaller_than_the_row_blocks(ctas, monkeypatch):
    """The host's grid forced to ``ctas`` CTAs for 771 row blocks, so every CTA
    claims blocks from the claim counter (on any SM count): ``dx`` and ``dz``
    the bits of the host's grid; ``dw`` (BF16 and FP32, alone and accumulated)
    within budget; ``compute_grad_weight=False`` leaves ``grad_weight`` as it
    is; the workspace is zero after every launch."""
    x, z, w, dy = make(257, 48, seed=12)
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    dw64 = fp64_reference(dy, x, z, w, EPS)[3]
    mag = magnitudes(dy, x, z, EPS)
    ws = workspace_for(257 * 48, x.device)
    monkeypatch.setattr(pkg, "backward_ctas", lambda name, rows, sm_count: ctas)
    for dtype in (torch.bfloat16, torch.float32):
        what = f"{dtype} at {ctas} CTAs"
        dw = torch.empty(D, dtype=dtype, device="cuda")
        dx, dz, _ = backward(dy, x, z, w, EPS, grad_weight=dw, workspace=ws)
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), what
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
    device = torch.device("cuda")
    ws = workspace_for(1, device)
    assert ws.dtype == torch.float64 and not bool(ws.any())
    nbytes = workspace_bytes(1, device)
    assert ws.numel() * 8 == nbytes == workspace_bytes(10**7, device)
    arch = loader.device_arch(torch.cuda.current_device())
    for role, name in loader.ROUTES[arch].items():
        if role.startswith("bwd"):
            assert loader.workspace_elements(name) * 8 <= nbytes, role
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


def test_workspace_errors():
    x, z, w, dy = make(65, 48, seed=13)
    elements = workspace_for(1, x.device).numel()
    # an FP32 buffer of the same size (the deterministic package's kind)
    fp32 = torch.zeros(2 * elements, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="FP64"):
        backward(dy, x, z, w, EPS, workspace=fp32)
    with pytest.raises(ValueError, match="FP64"):
        backward(
            dy,
            x,
            z,
            w,
            EPS,
            workspace=torch.zeros(elements - 1, dtype=torch.float64, device="cuda"),
        )
    strided = torch.zeros(2 * elements, dtype=torch.float64, device="cuda")[::2]
    with pytest.raises(ValueError, match="FP64"):
        backward(dy, x, z, w, EPS, workspace=strided)


def test_reset_workspace_clears_nonzero_scratch():
    """A caller workspace whose FP64 accumulators, arrival ticket and claim
    counter hold garbage (written by something other than the backward) is
    zeroed by ``reset_workspace``; backward launches at two different shapes and
    weight-gradient dtypes then run on it in sequence, within the FP64 budget,
    and leave it zero."""
    device = torch.device("cuda")
    ws = workspace_for(1, device)
    g = torch.Generator(device="cuda").manual_seed(60)
    ws.copy_(
        torch.randn(ws.shape, dtype=torch.float64, device=device, generator=g) * 1e3
    )
    arch = loader.device_arch(torch.cuda.current_device())
    for role in ("bwd_bf16", "bwd_f32"):
        word = loader.workspace_elements(loader.route_module(arch, role)) - 1
        # both 32-bit words of the dw_done element: a stale arrival ticket and
        # a stale block count
        ws[word : word + 1].view(torch.int32).fill_(3)
    assert bool(ws.ne(0).all())
    reset_workspace(ws)
    torch.cuda.synchronize()
    assert not bool(ws.any())
    for i, (T, H, dtype) in enumerate(
        ((1031, 48, torch.bfloat16), (33, 7, torch.float32))
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
    elements = workspace_for(1, torch.device("cuda")).numel()
    with pytest.raises(TypeError):
        reset_workspace(None)
    for bad in (
        torch.zeros(elements, dtype=torch.float32, device="cuda"),
        torch.zeros(elements - 1, dtype=torch.float64, device="cuda"),
        torch.zeros(2 * elements, dtype=torch.float64, device="cuda")[::2],
    ):
        with pytest.raises(ValueError, match="FP64"):
            reset_workspace(bad)
        assert not bool(bad.any())
    with pytest.raises(ValueError, match="CUDA"):
        reset_workspace(torch.zeros(elements, dtype=torch.float64))
    ws = workspace_for(1, torch.device("cuda"))
    for mode in ({}, {"deterministic": True}, {"deterministic": None}):
        with pytest.raises(ValueError, match=REFUSAL):
            pkg.cake_rmsnorm_silu_gate_train_reset_workspace(ws, **mode)


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
        dx_d, dz_d, _ = refs[i]
        dw64, mag = references[i]
        for dx, dz, dw in outputs[i]:
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), i
            assert_dw_within(dw, dw64, mag, what=f"stream {i} dw")
        assert not bool(workspaces[i].any()), i


def test_side_stream_and_no_host_sync():
    x, z, w, dy = make(2049, 48, seed=9)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    ws = workspace_for(x.shape[0] * x.shape[1], x.device)
    # Load both modules before the sync-checked region; the bits of y, dx and dz
    # on the current stream.
    want_y = forward(x, z, w, EPS)
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS, workspace=ws)
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
        assert_dw_within(dw, dw64, mag, what="side-stream dw")
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
    for k in range(3):
        xn, zn, _, dyn = make(T, H, seed=40 + k)
        xs.copy_(xn)
        zs.copy_(zn)
        dys.copy_(dyn)
        graph.replay()
        torch.cuda.synchronize()
        # the replay against eager launches on the same inputs
        assert torch.equal(y, forward(xn, zn, w, EPS)), k
        dx_e, dz_e, _ = backward(dyn, xn, zn, w, EPS)
        assert torch.equal(dx, dx_e) and torch.equal(dz, dz_e), k
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
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
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
        dx_d, dz_d, _ = refs[i]
        dw64, mag = budgets[i]
        for dx, dz, dw in results[i]:
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), i
            assert_dw_within(dw, dw64, mag, what=f"graph {i} dw")
        assert not bool(taken[i].any()), i


def test_captured_calls_without_capture_ids_take_one_workspace_each(
    reserve, monkeypatch
):
    """Without cuda-python's capture ids, each captured call takes its own
    reserve workspace; replays stay correct and leave them zero."""
    monkeypatch.setattr(pkg, "_cudart", lambda: None)
    x, z, w, dy = make(129, 48, seed=74)
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
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
            assert_dw_within(dw, dw64, mag, what=f"replay {k} dw")
        assert not any(bool(ws.any()) for ws in taken), k


def test_capture_with_no_ready_workspace_fills_in_the_graph(monkeypatch):
    """With no ready reserve workspace a captured backward without a caller
    workspace zero-fills one from the graph's pool inside the graph (a fill
    kernel before the backward kernel) and stays correct."""
    monkeypatch.setattr(pkg, "_RESERVES", {})
    monkeypatch.setattr(pkg, "_CAPTURE_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_RESERVE_SIZE", 0)
    x, z, w, dy = make(129, 48, seed=73)
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    torch.cuda.synchronize()
    assert pkg._RESERVES[torch.cuda.current_device()]["blocks"] == []
    graph = _kept_graph()
    with torch.cuda.graph(graph):
        dx, dz, dw = backward(dy, x, z, w, EPS)
    types = _graph_node_types(graph)
    assert types == ["KERNEL", "KERNEL"], types
    for k in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), k
        assert_dw_within(dw, dw64, mag, what=f"replay {k} dw")


def test_autograd_matches_functional_and_checkpoint(launches):
    x, z, w, dy = make(513, 48, seed=8)
    # the functional entry points' bits
    y_ref = forward(x, z, w, EPS)
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)

    def check(y, xg, zg, wg, what):
        assert torch.equal(y, y_ref), what
        assert torch.equal(xg, dx_d) and torch.equal(zg, dz_d), what
        assert wg.dtype == torch.bfloat16, what
        assert_dw_within(wg, dw64, mag, what=f"{what} dw vs FP64")

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
    routes = _routes()
    launches.clear()
    xc, zc, wc = (t.clone().requires_grad_(True) for t in (x, z, w))
    yc = torch.utils.checkpoint.checkpoint(
        gated_rmsnorm, xc, zc, wc, EPS, use_reentrant=False
    )
    yc.backward(dy)
    check(yc, xc.grad, zc.grad, wc.grad, "checkpoint")
    assert launches == [routes["fwd"], routes["fwd"], routes["bwd_bf16"]], launches


def test_nondeterministic_mode_launches_this_package(launches):
    x, z, w, dy = make(129, 48, seed=14)
    routes = _routes()
    forward(x, z, w, EPS)
    backward(dy, x, z, w, EPS)
    backward(
        dy, x, z, w, EPS, grad_weight=torch.empty(D, dtype=torch.float32, device="cuda")
    )
    assert launches == [routes["fwd"], routes["bwd_bf16"], routes["bwd_f32"]], launches
    for name in launches:
        assert name.startswith("cake_rmsnorm_silu_gate_train_"), name
    assert loader.MODULES[routes["bwd_bf16"]]["role"] == "bwd_bf16"
    assert loader.MODULES[routes["bwd_f32"]]["role"] == "bwd_f32"
    # The default is refused before any launch.
    launches.clear()
    with pytest.raises(ValueError, match=REFUSAL):
        pkg.cake_rmsnorm_silu_gate_train_forward(x, z, w, EPS)
    with pytest.raises(ValueError, match=REFUSAL):
        pkg.cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, EPS)
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
    """Every role ships one module per architecture (a long role at most one);
    every module record names its ``build.json`` library (schema
    ``cake-build/1``), whose entries carry the package's kernel and binding
    sources and the precise-arithmetic flags; the backward records size their
    workspace from their accumulator slots."""
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == "flashinfer.cake_rmsnorm_silu_gate_train_backend"
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    for arch in loader.ARCHES:
        routed = set(loader.ROUTES[arch])
        tiers = loader.LONG_ROLES + loader.MID_ROLES + loader.SHORT_ROLES
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
        if record["role"] in ("fwd", "fwd_short"):
            assert record["grid_rule"]["kind"] == "row_blocks"
        elif record["role"] in ("fwd_long", "fwd_mid"):
            assert record["grid_rule"]["kind"] == "persistent_row_blocks"
        else:
            assert record["grid_rule"]["kind"] == "row_claims"
            plan = [key for _kind, key in record["arg_plan"]]
            assert {"dw_sums", "dw_done", "dw", "weight_grad"} <= set(plan), name
            assert loader.workspace_elements(name) == record["dw_slots"] * D + 1
        for arch in record["arches"]:
            entry = libraries[loader.library_name(name, arch)]
            for flag in ("--prec-div=true", "--prec-sqrt=true", "--ftz=false"):
                assert flag in entry["nvcc_flags"], (name, arch, flag)
            assert "--use_fast_math" not in entry["nvcc_flags"], (name, arch)
            for source in entry["sources"]:
                assert (package / source).is_file(), (name, source)


@pytest.mark.parametrize("sm_count", [148, 160, 224])
@pytest.mark.parametrize("rows", [1, 48, 6191, 49488, 98304, 1440000, 1584000])
def test_backward_cta_rule(rows, sm_count):
    # One CTA per block of rows_per_cta rows, at most ctas_per_sm CTAs per SM
    # (one wave); past one wave the CTAs claim the blocks in chunks from the
    # workspace's claim counter, so a smaller grid still covers every row.
    for role in ("bwd_bf16", "bwd_f32"):
        name = loader.route_module("sm_100a", role)
        record = loader.MODULES[name]
        rule = record["grid_rule"]
        ctas = loader.backward_ctas(name, rows, sm_count)
        blocks = -(-rows // rule["rows_per_cta"])
        wave = rule["ctas_per_sm"] * sm_count
        assert ctas == min(blocks, wave)
        grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=ctas)
        assert grid == (ctas, 1, 1)


# The long and mid roles each architecture routes, with the rows per SM they
# run from.
TIER_ROUTES = {
    "sm_100a": {
        "fwd_long": 2400,
        "fwd_mid": 976,
        "bwd_bf16_long": 1280,
        "bwd_f32_long": 1280,
    },
    "sm_103a": {
        "fwd_long": 2400,
        "fwd_mid": 976,
        "bwd_bf16_long": 1280,
        "bwd_f32_long": 1280,
    },
    "sm_107a": {"bwd_bf16_long": 1024, "bwd_f32_long": 1024},
}


# The short roles each architecture routes, with the rows per SM they run up
# to.
SHORT_ROUTES = {
    "sm_100a": {"fwd_short": 10},
    "sm_103a": {"fwd_short": 10},
    "sm_107a": {},
}


def _tier_module(arch, role, rows, sm_count):
    """The module of ``role`` for ``rows`` rows by ``SHORT_ROUTES`` and
    ``TIER_ROUTES``: the short module up to its rows per SM, else the long
    module from its rows per SM, else the mid module from its own, else the
    role's module."""
    end = SHORT_ROUTES[arch].get(role + "_short")
    if end is not None and rows <= end * sm_count:
        return loader.ROUTES[arch][role + "_short"]
    for tier in ("_long", "_mid"):
        start = TIER_ROUTES[arch].get(role + tier)
        if start is not None and rows >= start * sm_count:
            return loader.ROUTES[arch][role + tier]
    return loader.route_module(arch, role)


@pytest.mark.parametrize("sm_count", [148, 152, 212, 216])
def test_short_long_and_mid_modules_run_from_their_rows_per_sm(sm_count):
    """``module_for_rows``: a role's short module up to ``max_rows_per_sm *
    sm_count`` rows of its grid rule, above that its long module from
    ``min_rows_per_sm * sm_count`` rows of its own rule, below that its mid
    module from its own rule's rows (on the architectures that route them),
    the role's module between and on the others."""
    assert loader.TIERS == ("_long", "_mid")
    for arch in loader.ARCHES:
        shorts = {
            role: loader.MODULES[name]["grid_rule"]["max_rows_per_sm"]
            for role, name in loader.ROUTES[arch].items()
            if role in loader.SHORT_ROLES
        }
        assert shorts == SHORT_ROUTES[arch], arch
        tiers = {
            role: loader.MODULES[name]["grid_rule"]["min_rows_per_sm"]
            for role, name in loader.ROUTES[arch].items()
            if role in loader.LONG_ROLES + loader.MID_ROLES
        }
        assert tiers == TIER_ROUTES[arch], arch
        for role, name in loader.ROUTES[arch].items():
            assert loader.MODULES[name]["role"] == role
        edges = {1, 48, 2**31 - 1}
        for start in (10, 1280, 976, 1024, 2400):
            edges |= {start * sm_count - 1, start * sm_count, start * sm_count + 1}
        for role in loader.ROLES:
            for rows in sorted(edges):
                want = _tier_module(arch, role, rows, sm_count)
                got = loader.module_for_rows(arch, role, rows, sm_count)
                assert got == want, (arch, role, rows)
    # the forward on sm_100a: the short forward up to 10 rows per SM, the
    # forward above, then the mid forward from 976 rows per SM, then the long
    # forward from 2400
    routes = loader.ROUTES["sm_100a"]
    for rows, role in (
        (1, "fwd_short"),
        (10 * sm_count, "fwd_short"),
        (10 * sm_count + 1, "fwd"),
        (976 * sm_count - 1, "fwd"),
        (976 * sm_count, "fwd_mid"),
        (2400 * sm_count - 1, "fwd_mid"),
        (2400 * sm_count, "fwd_long"),
    ):
        assert loader.module_for_rows("sm_100a", "fwd", rows, sm_count) == routes[role]
    # the backward on sm_100a: the backward, then the long backward from 1280
    # rows per SM
    for role in ("bwd_bf16", "bwd_f32"):
        for rows, tier in ((1280 * sm_count - 1, ""), (1280 * sm_count, "_long")):
            got = loader.module_for_rows("sm_100a", role, rows, sm_count)
            assert got == routes[role + tier], (role, rows)


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
    (2**31 - 1, 148, 25192 * 444),  # 2**26 blocks, 25192 waves
]


@pytest.mark.parametrize(("rows", "sm_count", "ctas"), LONG_FORWARD_GRIDS)
def test_long_forward_grid_is_whole_waves_of_at_most_six_blocks_a_cta(
    rows, sm_count, ctas
):
    name = loader.ROUTES["sm_100a"]["fwd_long"]
    record = loader.MODULES[name]
    assert record["grid_rule"] == {
        "ctas_per_sm": 3,
        "kind": "persistent_row_blocks",
        "max_blocks_per_cta": 6,
        "min_rows_per_sm": 2400,
        "rows_per_block": 32,
    }
    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count)
    assert grid == (ctas, 1, 1)
    blocks, wave = -(-rows // 32), 3 * sm_count
    assert -(-blocks // ctas) <= 6
    assert ctas == blocks or ctas % wave == 0
    assert ctas == blocks or (ctas - wave) * 6 < blocks
    # the grid is the long forward's whatever ``ctas`` the call passes
    assert loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=7) == grid
    fwd = loader.MODULES[loader.route_module("sm_100a", "fwd")]
    assert loader.launch_grid(fwd, rows=rows, sm_count=sm_count) == (
        -(-rows // 16),
        1,
        1,
    )


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
    (2**31 - 1, 148, 67109712),  # 2**27 blocks, 75574 waves
]


@pytest.mark.parametrize(("rows", "sm_count", "ctas"), MID_FORWARD_GRIDS)
def test_mid_forward_grid_is_whole_waves_of_at_most_two_blocks_a_cta(
    rows, sm_count, ctas
):
    for arch in ("sm_100a", "sm_103a"):
        name = loader.ROUTES[arch]["fwd_mid"]
        assert name == loader.ROUTES["sm_100a"]["fwd_mid"]
    assert "fwd_mid" not in loader.ROUTES["sm_107a"]
    record = loader.MODULES[name]
    assert record["grid_rule"] == {
        "ctas_per_sm": 6,
        "kind": "persistent_row_blocks",
        "max_blocks_per_cta": 2,
        "min_rows_per_sm": 976,
        "rows_per_block": 16,
    }
    assert record["launch"]["block"] == (128, 1, 1)
    assert record["arches"] == ["sm_100a", "sm_103a"]
    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count)
    assert grid == (ctas, 1, 1)
    blocks, wave = -(-rows // 16), 6 * sm_count
    assert -(-blocks // ctas) <= 2
    assert ctas == blocks or ctas % wave == 0
    assert ctas == blocks or (ctas - wave) * 2 < blocks
    assert loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=7) == grid


# (rows, SMs) of the short forward: one CTA per 16-row block, up to 10 rows per
# SM, so fewer CTAs than SMs.
SHORT_FORWARD_GRIDS = [
    (1, 148),
    (48, 148),  # T 1, H 48: 3 blocks
    (13 * 7, 148),
    (21 * 48, 148),  # 6.8 rows per SM on B200
    (10 * 148, 148),  # B200 at 10 rows per SM: 93 blocks
    (10 * 152, 152),  # GB300 at 10 rows per SM: 95 blocks
]


@pytest.mark.parametrize(("rows", "sm_count"), SHORT_FORWARD_GRIDS)
def test_short_forward_grid_is_one_cta_a_block_below_one_cta_per_sm(rows, sm_count):
    for arch in ("sm_100a", "sm_103a"):
        name = loader.ROUTES[arch]["fwd_short"]
        assert name == loader.ROUTES["sm_100a"]["fwd_short"]
        assert loader.module_for_rows(arch, "fwd", rows, sm_count) == name
    assert "fwd_short" not in loader.ROUTES["sm_107a"]
    record = loader.MODULES[name]
    assert record["grid_rule"] == {
        "kind": "row_blocks",
        "max_rows_per_sm": 10,
        "rows_per_cta": 16,
    }
    assert record["launch"]["block"] == (256, 1, 1)
    assert record["launch"]["dynamic_smem_bytes"] == 0
    assert record["arches"] == ["sm_100a", "sm_103a"]
    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count)
    assert grid == (-(-rows // 16), 1, 1)
    assert grid[0] < sm_count
    assert loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=7) == grid
    # the deterministic forward's grid: one 128-thread CTA per 16-row block
    fwd = loader.MODULES[loader.route_module("sm_100a", "fwd")]
    assert loader.launch_grid(fwd, rows=rows, sm_count=sm_count) == grid


@pytest.mark.parametrize("sm_count", [208, 212, 216])
@pytest.mark.parametrize("rows", [1, 1024, 217087, 217088, 1571376])
def test_long_backward_cta_rule(rows, sm_count):
    # sm_107a's long backward: 8-row blocks of 128 threads, at most 8 CTAs per
    # SM (one wave); the workspace is the 256-thread module's.
    for role in ("bwd_bf16", "bwd_f32"):
        name = loader.ROUTES["sm_107a"][f"{role}_long"]
        record = loader.MODULES[name]
        assert record["grid_rule"] == {
            "ctas_per_sm": 8,
            "kind": "row_claims",
            "min_rows_per_sm": 1024,
            "rows_per_cta": 8,
        }
        assert record["launch"]["block"] == (128, 1, 1)
        ctas = loader.backward_ctas(name, rows, sm_count)
        assert ctas == min(-(-rows // 8), 8 * sm_count)
        grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=ctas)
        assert grid == (ctas, 1, 1)
        base = loader.route_module("sm_107a", role)
        assert loader.workspace_elements(name) == loader.workspace_elements(base)


@pytest.mark.parametrize("sm_count", [148, 152, 160])
@pytest.mark.parametrize(
    "rows", [1, 1280 * 148 - 1, 1280 * 148, 2048 * 48, 32737 * 48, 2**31 - 1]
)
def test_bulk_long_backward_cta_rule(rows, sm_count):
    # sm_100a's and sm_103a's long backward: the backward's 256-thread CTAs of
    # 16-row blocks, at most 5 CTAs per SM (a quarter more than an SM holds at
    # once), from 1280 rows per SM; the workspace is the backward's.
    for arch in ("sm_100a", "sm_103a"):
        for role in ("bwd_bf16", "bwd_f32"):
            name = loader.ROUTES[arch][f"{role}_long"]
            base = loader.route_module(arch, role)
            record = loader.MODULES[name]
            assert record["arches"] == ["sm_100a", "sm_103a"]
            assert record["grid_rule"] == {
                "ctas_per_sm": 5,
                "kind": "row_claims",
                "min_rows_per_sm": 1280,
                "rows_per_cta": 16,
            }
            assert record["launch"] == loader.MODULES[base]["launch"]
            assert record["dw_slots"] == loader.MODULES[base]["dw_slots"]
            ctas = loader.backward_ctas(name, rows, sm_count)
            assert ctas == min(-(-rows // 16), 5 * sm_count)
            grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=ctas)
            assert grid == (ctas, 1, 1)
            assert loader.workspace_elements(name) == loader.workspace_elements(base)


def test_calls_without_workspace_share_one_buffer_per_stream():
    x, z, w, dy = make(65, 7)
    for _ in range(2):
        backward(dy, x, z, w, 1e-6)
    torch.cuda.synchronize()
    stream = torch.cuda.current_stream().cuda_stream
    held = [v for k, v in pkg._STREAM_WORKSPACES.items() if k[1] == stream]
    assert held, "a call without a workspace keeps one per stream"
    for sums, done in held:
        assert sums.abs().max().item() == 0
        assert done.view(torch.int32).tolist() == [0, 0]
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
        _routes()["bwd_bf16"],
    )
    assert list(pkg._STREAM_WORKSPACES) == [key]
    sums, _ = pkg._STREAM_WORKSPACES[key]
    assert sums.untyped_storage().nbytes() == nbytes
    (block,) = pkg._RESERVES[torch.cuda.current_device()]["blocks"]
    reserved = block.untyped_storage().nbytes()
    assert reserved == pkg._RESERVE_SIZE * 544 * 8, reserved
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


# ``deterministic`` values other than the bool ``False`` (``{}``: the default).
MODES = (
    {},
    {"deterministic": True},
    {"deterministic": None},
    {"deterministic": 0},
    {"deterministic": 1},
    {"deterministic": "False"},
)


def _mode_calls(**mode) -> dict:
    """One call per entry point of the package with ``mode`` on valid operands."""
    x, z, w, dy = make(5, 48, seed=64)
    ws = workspace_for(1, x.device)
    flag = tuple(mode.values())
    return {
        "cake_rmsnorm_silu_gate_train_forward": lambda: pkg.cake_rmsnorm_silu_gate_train_forward(x, z, w, EPS, **mode),
        "cake_rmsnorm_silu_gate_train_backward": lambda: pkg.cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, EPS, **mode),
        "cake_rmsnorm_silu_gate_train_backward_workspace": lambda: pkg.cake_rmsnorm_silu_gate_train_backward_workspace(1, x.device, **mode),
        "cake_rmsnorm_silu_gate_train_backward_workspace_bytes": lambda: pkg.cake_rmsnorm_silu_gate_train_backward_workspace_bytes(1, x.device, **mode),
        "cake_rmsnorm_silu_gate_train_reset_workspace": lambda: pkg.cake_rmsnorm_silu_gate_train_reset_workspace(ws, **mode),
        "cake_rmsnorm_silu_gate": lambda: pkg.cake_rmsnorm_silu_gate(x, z, w, EPS, **mode),
        "CakeRMSNormSiLUGateFunction": lambda: pkg.CakeRMSNormSiLUGateFunction.apply(x, z, w, EPS, *flag),
    }  # fmt: skip


@pytest.mark.parametrize(
    "mode", MODES, ids=["default", "true", "none", "zero", "one", "string"]
)
def test_entry_points_refuse_a_deterministic_mode(mode, launches):
    """Any ``deterministic`` but the bool ``False`` (the default ``True``
    included) is a ``ValueError`` naming the entry point and the deterministic
    package, before any launch, also under PyTorch's deterministic algorithms
    (the refusal precedes the alert).  The support query, a capability probe
    that launches nothing, answers for every value."""
    calls = _mode_calls(**mode)
    assert set(calls) == set(pkg.__all__) - {
        "ARCHES",
        "SUPPORTED_COMPUTE_CAPABILITIES",
        "is_cake_rmsnorm_silu_gate_train_supported",
    }
    for enabled, warn_only in ((False, False), (True, False), (True, True)):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        assert (
            pkg.is_cake_rmsnorm_silu_gate_train_supported(torch.device("cuda"), **mode)
            is True
        )
        for name, call in calls.items():
            with pytest.raises(ValueError, match=f"^{name} {REFUSAL}") as info:
                call()
            assert (
                "flashinfer.cake_rmsnorm_silu_gate_det_train is the deterministic package"
                in str(info.value)
            ), name
    assert launches == []


def test_backward_alerts_under_deterministic_algorithms(launches):
    """Under ``torch.use_deterministic_algorithms(True)`` the paths that launch
    the unordered weight-gradient sum -- the backward with
    ``compute_grad_weight=True`` and the autograd backward when ``weight`` needs
    a gradient, also for a node built before the mode was enabled -- raise
    PyTorch's alert before any launch; under ``warn_only=True`` they warn and
    run the non-deterministic backward (never a fallback).  The forward, the
    workspace entry points, the reset, the support query and the backward
    without the weight gradient run without a warning."""
    x, z, w, dy = make(257, 48, seed=65)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    ws = workspace_for(1, x.device)
    routes = _routes()

    def leaves(weight_grad=True):
        return (
            x.clone().requires_grad_(True),
            z.clone().requires_grad_(True),
            w.clone().requires_grad_(weight_grad),
        )

    node = gated_rmsnorm(*leaves(), EPS)  # built with the mode off
    torch.use_deterministic_algorithms(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        forward(x, z, w, EPS)
        assert workspace_bytes(1, x.device) == workspace_for(1, x.device).numel() * 8
        reset_workspace(ws)
        assert pkg.is_cake_rmsnorm_silu_gate_train_supported(
            x.device, deterministic=False
        )
        assert backward(dy, x, z, w, EPS, compute_grad_weight=False)[2] is None
        xa, za, wa = leaves(weight_grad=False)
        gated_rmsnorm(xa, za, wa, EPS).backward(dy)
        assert xa.grad is not None and wa.grad is None
    launches.clear()
    with pytest.raises(RuntimeError, match=ALERT):
        backward(dy, x, z, w, EPS)
    with pytest.raises(RuntimeError, match=ALERT):
        node.backward(dy)
    assert launches == []
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning, match=ALERT):
        _, _, dw = backward(dy, x, z, w, EPS)
    assert_dw_within(dw, dw64, mag, what="warn_only dw")
    xa, za, wa = leaves()
    y = gated_rmsnorm(xa, za, wa, EPS)
    with pytest.warns(UserWarning, match=ALERT):
        y.backward(dy)
    assert_dw_within(wa.grad, dw64, mag, what="warn_only autograd dw")
    assert launches == [routes["bwd_bf16"], routes["fwd"], routes["bwd_bf16"]], launches
