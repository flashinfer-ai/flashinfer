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

Tests of the Cake SiLU-gated RMSNorm training forward / backward
(:mod:`flashinfer.cake_rmsnorm_silu_gate_det_train`): ``RMSNorm(x) * w * SiLU(z)``
against the eager FP32 operation (one final BF16 cast) and an independent
FP64 reference, over dynamic token and head counts, irregular tails, strided
layouts, near-zero variance, saturated gates, gate inputs at the edges of the
forward's gate (finite ones, and infinite and NaN ones), gradient
accumulation, checkpointing, side streams and CUDA Graphs.  ``y``, ``dx``,
``dz`` and the BF16, FP32 and accumulated FP32 weight gradients are the same
bits at every call with the same inputs: at the issue's shapes, small and odd
shapes, the row counts at the edges of the forward's grids (one CTA a row
block below ``persistent_from_rows_per_sm`` rows per SM, whole waves from
there) and of the backward's grid on the device, on every layout (on an
architecture that routes a forward of packed rows, on both forward routes), on
two streams at once, in CUDA Graph replays and under
``torch.use_deterministic_algorithms``.  A backward of packed rows runs its
role's long module from the long module's ``min_rows_per_sm`` rows per SM, and
strided rows, a single token and fewer rows run the role's module; the two
give the same ``dx``, ``dz`` and weight-gradient bits on the same inputs and
grid.  Far past
one wave of the backward's grid (CTAs claiming row blocks) and with the grids
of other SM counts the weight gradient lies within its FP64 budget.  The
backward's workspace is zero after every call (``compute_grad_weight=False``
and accumulation included), serves every shape and both weight-gradient
dtypes, is zeroed by ``cake_rmsnorm_silu_gate_det_train_reset_workspace``, and
the non-deterministic package's workspaces (FP64 or ``uint8``) and unfit FP32
buffers are refused; a call without a workspace uses its stream's and, during
CUDA Graph capture, one of the device's reserve (or one the graph zero-fills).
Without a device, the registry's records and grids are checked at several SM
counts.

This suite exercises the backend package alone: it names the package once,
through ``tests.cake.test_helpers.cake_package``, and imports nothing else from
flashinfer.  The ``flashinfer.cake_rmsnorm_silu_gate_det_train`` facade and the
``flashinfer`` top-level exposure are covered by
``test_cake_rmsnorm_silu_gate_det_train_api.py``.
"""

import inspect
import warnings
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

from tests.cake.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_silu_gate_det_train")
cake_build = cake_package("cake_rmsnorm_silu_gate_det_train", "cake_build")
loader = cake_package("cake_rmsnorm_silu_gate_det_train", "cake_jit")
cake_rmsnorm_silu_gate_det = pkg.cake_rmsnorm_silu_gate_det
cake_rmsnorm_silu_gate_det_train_backward = (
    pkg.cake_rmsnorm_silu_gate_det_train_backward
)
cake_rmsnorm_silu_gate_det_train_backward_workspace = (
    pkg.cake_rmsnorm_silu_gate_det_train_backward_workspace
)
cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes = (
    pkg.cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes
)
cake_rmsnorm_silu_gate_det_train_forward = pkg.cake_rmsnorm_silu_gate_det_train_forward
cake_rmsnorm_silu_gate_det_train_reset_workspace = (
    pkg.cake_rmsnorm_silu_gate_det_train_reset_workspace
)
is_cake_rmsnorm_silu_gate_det_train_supported = (
    pkg.is_cake_rmsnorm_silu_gate_det_train_supported
)
forward = cake_rmsnorm_silu_gate_det_train_forward
backward = cake_rmsnorm_silu_gate_det_train_backward
workspace_for = cake_rmsnorm_silu_gate_det_train_backward_workspace
workspace_bytes = cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes
reset_workspace = cake_rmsnorm_silu_gate_det_train_reset_workspace

D = 128
EPS = 1e-6
# One BF16 unit in the last place relative to the value (8 significant bits).
BF16_RTOL = 2.0**-7

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cake_rmsnorm_silu_gate_det_train_supported(torch.device("cuda")),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)


def eager(x, z, w, eps):
    """The issue's reference operation in FP32, one final cast."""
    xf, zf, wf = x.float(), z.float(), w.float()
    rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (((xf * rstd) * wf) * torch.nn.functional.silu(zf)).to(x.dtype)


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


def assert_bf16_close(got, want, *, what, scale=None):
    """``got`` (BF16) within one BF16 ulp of ``want`` plus an absolute floor of
    1/64 ulp of the tensor's largest magnitude (cancellation in ``dx``), or of
    ``scale`` (a number, or per-row scales that broadcast); a NaN fails."""
    want = want.double()
    floor = (scale if scale is not None else want.abs().max().item()) * BF16_RTOL / 64
    err = (got.double() - want).abs()
    bound = want.abs() * BF16_RTOL + floor
    bad = ~(err <= bound)
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} of {bad.numel()} elements beyond one BF16 ulp; "
        f"worst err {err.max().item():.3e}"
    )


def assert_y_elementwise(y, x, z, w, eps, *, what):
    """``y`` within one BF16 ulp of the FP64 reference plus a floor of 1/64 ulp
    of its row's largest finite ungated magnitude ``|x * rstd * w|``, times the
    gate's magnitude where that exceeds one: each element's own bound, which
    large gates elsewhere in the row or the tensor do not widen.  Elements whose
    reference is not finite are left to the callers."""
    xd, zd, wd = x.double(), z.double(), w.double()
    rstd = 1.0 / torch.sqrt(xd.square().mean(dim=-1, keepdim=True) + eps)
    ungated = xd * rstd * wd
    scale = torch.where(ungated.isfinite(), ungated.abs(), 0.0).amax(
        dim=-1, keepdim=True
    )
    silu = zd * torch.sigmoid(zd)
    want = ungated * silu
    floor = scale * silu.abs().clamp_min(1.0) * BF16_RTOL / 64
    err = (y.double() - want).abs()
    bad = want.isfinite() & ~(err <= want.abs() * BF16_RTOL + floor)
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} of {bad.numel()} elements beyond their own "
        f"bound; worst err {err[bad].max().item():.3e}"
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
        xv = x.view(T, H, D) if layout != "flat" else x.view(T, H, D)
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


def eager_grads(dy, x, z, w, eps):
    xr = x.detach().clone().requires_grad_(True)
    zr = z.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    y = eager(xr, zr, wr, eps)
    y.backward(dy)
    return y.detach(), xr.grad, zr.grad, wr.grad


def row_scales(dy, x, z, w, eps):
    """``(dx_scale, dz_scale)``: each row's scale of the backward's documented
    floor, in FP64 from the inputs, ``[..., 1]``: for ``dz`` the row's largest
    ``|dy * n * w|``; for ``dx`` the largest magnitudes of its two terms, the
    row's largest ``|dn * rstd|`` (``dn = dy * silu(z) * w``) plus its largest
    ``|x|`` times ``rstd^3`` times the mean of ``|dn * x|``."""
    xd, zd, wd, gd = x.double(), z.double(), w.double(), dy.double()
    rstd = 1.0 / torch.sqrt(xd.square().mean(dim=-1, keepdim=True) + eps)
    dz_scale = (gd * (xd * rstd) * wd).abs().amax(dim=-1, keepdim=True)
    dn = gd * torch.nn.functional.silu(zd) * wd
    coef = rstd**3 * (dn * xd).abs().mean(dim=-1, keepdim=True)
    dx_scale = (dn * rstd).abs().amax(dim=-1, keepdim=True) + xd.abs().amax(
        dim=-1, keepdim=True
    ) * coef
    return dx_scale, dz_scale


def check_all(x, z, w, dy, eps):
    y = cake_rmsnorm_silu_gate_det_train_forward(x, z, w, eps)
    dx, dz, dw = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, eps)
    assert y.dtype == dx.dtype == dz.dtype == dw.dtype == torch.bfloat16
    assert y.shape == dx.shape == dz.shape == x.shape and dw.shape == (D,)
    y64, dx64, dz64, dw64 = fp64_reference(dy, x, z, w, eps)
    assert_bf16_close(y, y64, what="y vs FP64")
    assert_bf16_close(dz, dz64, what="dz vs FP64")
    assert_bf16_close(dx, dx64, what="dx vs FP64")
    # dw: the sum of T * H terms; the FP64 sum rounded once, within one BF16 ulp
    # plus the FP32 chunk accumulation's error (relative to the sum of magnitudes).
    terms = (dy.double() * torch.nn.functional.silu(z.double())) * (
        x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + eps)
    )
    mag = terms.abs().reshape(-1, D).sum(0)
    err = (dw.double() - dw64).abs()
    assert (err <= dw64.abs() * BF16_RTOL + mag * 2.0**-20).all(), "dw vs FP64"
    ye, dxe, dze, dwe = eager_grads(dy, x, z, w, eps)
    assert_bf16_close(y, ye, what="y vs eager", scale=y64.abs().max().item())
    assert_bf16_close(dz, dze, what="dz vs eager", scale=dz64.abs().max().item())
    assert_bf16_close(dx, dxe, what="dx vs eager", scale=dx64.abs().max().item())
    # The documented bound: one BF16 ulp of the FP32 formula (the eager
    # autograd) plus 1/64 ulp of the row's own scale.
    dx_scale, dz_scale = row_scales(dy, x, z, w, eps)
    assert_bf16_close(dz, dze, what="dz vs eager, row floor", scale=dz_scale)
    assert_bf16_close(dx, dxe, what="dx vs eager, row floor", scale=dx_scale)
    assert (
        (dw.double() - dwe.double()).abs()
        <= dwe.double().abs() * BF16_RTOL + mag * 2.0**-20
    ).all(), "dw vs eager"


def dw_reference(dy, x, z, eps):
    """``(dw64, mag)``: the FP64 weight gradient and the FP64 sum of its terms'
    magnitudes per column (the terms of ``check_all``'s bound)."""
    terms = (dy.double() * torch.nn.functional.silu(z.double())) * (
        x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + eps)
    )
    terms = terms.reshape(-1, D)
    return terms.sum(0), terms.abs().sum(0)


def assert_dw_within(dw, want, mag, *, what):
    """``dw`` within ``check_all``'s weight-gradient bound of ``want``: one
    BF16 ulp plus ``2^-20`` of the terms' magnitudes."""
    want = want.double()
    err = (dw.double() - want).abs()
    bad = ~(err <= want.abs() * BF16_RTOL + mag * 2.0**-20)
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} of {bad.numel()} columns beyond the bound; "
        f"worst err {err.max().item():.3e}"
    )


def assert_same_bits(got, want, what):
    assert got.dtype == want.dtype and got.shape == want.shape, what
    view = torch.int16 if got.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(got.view(view), want.view(view)), what


def assert_zero(workspace, what="the workspace"):
    """Every bit of ``workspace`` zero (an FP32 comparison would pass ``-0``)."""
    torch.cuda.synchronize()
    assert not bool(workspace.view(torch.int32).any()), f"{what} is not zero"


def _device():
    """``(arch, SMs)`` of the current device."""
    index = torch.cuda.current_device()
    return loader.device_arch(index), loader.device_sm_count(index)


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The module names the registry launches, in order."""
    launched: list[str] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        launched.append(name)
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    return launched


def outputs(x, z, w, dy, *, workspace=None):
    """The forward, then the backward with a BF16, an FP32 and an FP32 weight
    gradient accumulated onto 0.25: ``(y, dx, dz, dw_bf16, dw_f32, dw_acc)``."""
    y = forward(x, z, w, EPS)
    dx, dz, dw = backward(dy, x, z, w, EPS, workspace=workspace)
    f32 = torch.empty(D, dtype=torch.float32, device=x.device)
    backward(dy, x, z, w, EPS, grad_weight=f32, workspace=workspace)
    acc = torch.full((D,), 0.25, dtype=torch.float32, device=x.device)
    backward(
        dy,
        x,
        z,
        w,
        EPS,
        grad_weight=acc,
        accumulate_grad_weight=True,
        workspace=workspace,
    )
    return y, dx, dz, dw, f32, acc


OUTPUTS = ("y", "dx", "dz", "dw_bf16", "dw_f32", "dw_accumulated")


def assert_outputs_equal(got, want, what):
    for a, b, name in zip(got, want, OUTPUTS, strict=True):
        assert_same_bits(a, b, f"{what}: {name}")


REPEATS = 5


@pytest.mark.parametrize(
    "T,H",
    [
        (1, 1),
        (1, 48),
        (3, 48),
        (17, 7),
        (255, 2),
        (1031, 48),
        (4097, 16),
        (2, 64),
        (33, 1),
    ],
)
def test_forward_backward_shapes(T, H):
    x, z, w, dy = make(T, H)
    check_all(x, z, w, dy, EPS)


@pytest.mark.parametrize("layout", ["flat", "projection", "head_major", "misaligned"])
def test_layouts(layout):
    x, z, w, dy = make(65, 48, layout=layout, seed=1)
    check_all(x, z, w, dy, EPS)


@pytest.mark.parametrize("layout", ["flat", "projection", "head_major", "misaligned"])
def test_layouts_give_the_contiguous_bits(layout):
    """The strides change no bits: every output of each layout (rows read in
    place with their own strides, or a misaligned input copied once) is the
    bits of the same rows given as contiguous ``[T, H, 128]`` tensors."""
    T, H = 65, 48
    x, z, w, dy = make(T, H, layout=layout, seed=1)
    got = outputs(x, z, w, dy)
    xc, zc, dyc = (t.reshape(T, H, D).contiguous() for t in (x, z, dy))
    want = outputs(xc, zc, w, dyc)
    for a, b, name in zip(got, want, OUTPUTS, strict=True):
        assert_same_bits(a.reshape(b.shape), b, f"{layout} {name}")


@pytest.mark.parametrize("data", ["near_zero", "saturated"])
def test_edge_values(data):
    x, z, w, dy = make(37, 48, data=data, seed=2)
    check_all(x, z, w, dy, EPS)
    assert_y_elementwise(forward(x, z, w, EPS), x, z, w, EPS, what=f"{data} y")


@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_epsilon(eps):
    x, z, w, dy = make(19, 48, seed=3)
    check_all(x, z, w, dy, eps)


@pytest.mark.parametrize("T", [30193, 32737, 32388])
def test_issue_shapes(T):
    x, z, w, dy = make(T, 48, seed=4)
    check_all(x, z, w, dy, EPS)


# Finite gate inputs at and past the edges of the forward's gate: below about
# -87.3 its reciprocal flushes to zero and below about -88.7 its exponential
# overflows (the gate is -0 in both), subnormal and zero ones, beside
# saturated and ordinary ones.
GATE_EDGE_VALUES = [
    -1e4,
    -100.0,
    -89.0,
    -88.5,
    -88.0,
    -87.5,
    -87.0,
    -86.5,
    -42.0,
    -41.0,
    -40.0,
    -1e-30,
    -1e-39,
    -0.0,
    0.0,
    1e-39,
    1e-30,
    40.0,
    88.0,
    89.0,
    100.0,
    1e4,
]
# The largest BF16 gate input below -87.3: from there down the gate is -0.
GATE_FLUSHED = -87.5
# The BF16 gate inputs whose FP32 sigmoid is subnormal (``z`` from about -87.34
# to -88.72): the backward's sigmoid factor ``s`` is subnormal there too, not
# flushed (the gate ``z * s`` is a normal number there).
GATE_SUBNORMAL = (-88.5, -87.5)


def _gate_edges(z, seed):
    """``z`` with gate edge values in a quarter of the elements of two of
    every three rows (so some threads of a row see them and others do not)."""
    g = torch.Generator(device=z.device).manual_seed(seed)
    values = torch.tensor(GATE_EDGE_VALUES, device=z.device).bfloat16()
    pick = torch.randint(
        0, len(GATE_EDGE_VALUES), z.shape, device=z.device, generator=g
    )
    edge = torch.rand(z.shape, device=z.device, generator=g) < 0.25
    rows = torch.arange(z.numel() // D, device=z.device).view(*z.shape[:-1], 1)
    edge &= rows % 3 != 0
    z.copy_(torch.where(edge, values[pick], z))


@pytest.mark.parametrize(
    "T,H", [(1, 48), (13, 7), (37, 48), (129, 48), (260, 7), (1031, 48)]
)
def test_gate_edge_values(T, H):
    """Gate inputs at the edges of the forward's gate (``GATE_EDGE_VALUES``)
    within ``check_all``'s bounds and ``y`` within each element's own bound
    (``assert_y_elementwise``); where ``z <= -87.5`` the forward's gate is
    ``-0``, so ``y`` is zero there.  Where the FP32 sigmoid is subnormal
    (``GATE_SUBNORMAL``; ``check_all``'s floor would pass a zero ``dz``),
    each ``dz`` lies within one BF16 ulp of the FP64 reference plus half the
    smallest BF16 subnormal (``2^-134``).  The FP32 weight-gradient program
    gives the BF16 program's ``dx`` and ``dz`` bits."""
    x, z, w, dy = make(T, H, seed=23)
    _gate_edges(z, 24)
    check_all(x, z, w, dy, EPS)
    y = forward(x, z, w, EPS)
    assert_y_elementwise(y, x, z, w, EPS, what="y at the gate edges")
    flushed = z.float() <= GATE_FLUSHED
    assert bool(flushed.any())
    assert not bool(y[flushed].float().ne(0).any()), "y where z <= -87.5"
    dx, dz, _ = backward(dy, x, z, w, EPS)
    f32 = torch.empty(D, dtype=torch.float32, device=x.device)
    dx32, dz32, _ = backward(dy, x, z, w, EPS, grad_weight=f32)
    assert_same_bits(dx32, dx, "dx of the FP32 weight-gradient program")
    assert_same_bits(dz32, dz, "dz of the FP32 weight-gradient program")
    low, high = GATE_SUBNORMAL
    band = (z.float() >= low) & (z.float() <= high)
    assert bool(band.any())
    dz = dz[band].double()
    want = fp64_reference(dy, x, z, w, EPS)[2][band]
    bad = ~((dz - want).abs() <= want.abs() * BF16_RTOL + 2.0**-134)
    assert not bad.any(), (
        f"dz where the FP32 sigmoid is subnormal: {int(bad.sum())} of "
        f"{bad.numel()} elements beyond one BF16 ulp plus 2^-134"
    )


def test_infinite_and_nan_gates():
    """Gates of ``+inf``, ``-inf`` and NaN in six rows, placed so that weight
    gradient column 0 sums to ``+inf``, column 4 to ``-inf`` and columns 1, 2
    (``+inf`` and ``-inf`` terms) and 3 (``silu(-inf)`` is NaN) to NaN: ``y``
    is NaN where the FP64 reference is NaN, the same infinity where it is
    infinite and within ``check_all``'s bounds elsewhere; ``dx`` and ``dz`` of
    the rows with finite gates lie within those bounds; the BF16 and the FP32
    weight gradient are NaN in the FP64 reference's NaN columns, its infinity
    in its infinite ones and within budget in the others; the workspace is zero
    after each call and gives the next call the bits of a fresh workspace."""
    T, H = 257, 48
    x, z, w, dy = make(T, H, seed=25)
    # (token, head, column, gate, upstream gradient); x is 1 there, so the
    # term's sign is the gradient's
    cases = (
        (0, 0, 0, float("inf"), 1.0),
        (1, 0, 1, float("nan"), 1.0),
        (2, 0, 2, float("inf"), 1.0),
        (3, 5, 2, float("inf"), -1.0),
        (4, 0, 3, float("-inf"), 1.0),
        (5, 7, 4, float("inf"), -1.0),
    )
    for t, h, c, gate, grad in cases:
        z[t, h, c] = gate
        dy[t, h, c] = grad
        x[t, h, c] = 1.0
    y64, dx64, dz64, dw64 = fp64_reference(dy, x, z, w, EPS)
    _, mag = dw_reference(dy, x, z, EPS)
    assert dw64[0].item() == float("inf") and dw64[4].item() == float("-inf")
    assert bool(dw64[1:4].isnan().all())

    def assert_classes(got, want, what):
        assert torch.equal(got.isnan(), want.isnan()), f"{what}: NaN elements"
        inf = want.isinf()
        assert torch.equal(got.isinf(), inf), f"{what}: infinite elements"
        assert torch.equal(got[inf].double(), want[inf]), f"{what}: infinities"
        return want.isfinite()

    y = forward(x, z, w, EPS)
    ok = assert_classes(y, y64, "y")
    assert_bf16_close(y[ok], y64[ok], what="y of the finite elements")
    ws = workspace_for(T * H, x.device)
    f32 = torch.empty(D, dtype=torch.float32, device=x.device)
    dx, dz, dw = backward(dy, x, z, w, EPS, workspace=ws)
    assert_zero(ws, "the workspace after the BF16 weight gradient")
    backward(dy, x, z, w, EPS, grad_weight=f32, workspace=ws)
    assert_zero(ws, "the workspace after the FP32 weight gradient")
    for got, what in ((dw, "BF16 dw"), (f32, "FP32 dw")):
        ok = assert_classes(got, dw64, what)
        assert_dw_within(got[ok], dw64[ok], mag[ok], what=f"{what}, finite columns")
    rows = torch.isfinite(z.float()).all(-1)
    assert not bool(rows.all())
    assert_bf16_close(dx[rows], dx64[rows], what="dx of the rows with finite gates")
    assert_bf16_close(dz[rows], dz64[rows], what="dz of the rows with finite gates")
    xf, zf, wf, dyf = make(T, H, seed=26)
    got = backward(dyf, xf, zf, wf, EPS, workspace=ws)
    want = backward(dyf, xf, zf, wf, EPS, workspace=workspace_for(T * H, x.device))
    for a, b, name in zip(got, want, ("dx", "dz", "dw"), strict=True):
        assert_same_bits(a, b, f"{name} on the workspace after the non-finite call")
    assert_zero(ws)


@pytest.mark.parametrize("T,H", [(1, 1), (257, 47)], ids=["1x1", "257x47"])
def test_infinite_terms_in_the_last_row_of_a_partial_block(T, H):
    """A row count that is not a multiple of the 16-row block, with infinite
    weight-gradient terms in the last row (a gate of ``+inf`` in column 0, an
    upstream gradient of ``+inf`` in column 1): the rows of the last block past
    the last row read that row and add nothing, so both columns are ``+inf`` in
    the BF16 and the FP32 weight gradient, as in the FP64 reference, and the
    other columns lie within budget; the workspace is zero after each call."""
    x, z, w, dy = make(T, H, seed=27)
    assert (T * H) % 16
    t, h = T - 1, H - 1
    z[t, h, 0], dy[t, h, 0], x[t, h, 0] = float("inf"), 1.0, 1.0
    z[t, h, 1], dy[t, h, 1], x[t, h, 1] = 1.0, float("inf"), 1.0
    dw64, mag = dw_reference(dy, x, z, EPS)
    assert dw64[0].item() == dw64[1].item() == float("inf")
    assert bool(dw64[2:].isfinite().all())
    ws = workspace_for(T * H, x.device)
    f32 = torch.empty(D, dtype=torch.float32, device=x.device)
    dw = backward(dy, x, z, w, EPS, workspace=ws)[2]
    assert_zero(ws, "the workspace after the BF16 weight gradient")
    backward(dy, x, z, w, EPS, grad_weight=f32, workspace=ws)
    assert_zero(ws, "the workspace after the FP32 weight gradient")
    for got, what in ((dw, "BF16 dw"), (f32, "FP32 dw")):
        assert got[0].item() == got[1].item() == float("inf"), what
        assert bool(got[2:].isfinite().all()), what
        assert_dw_within(got[2:], dw64[2:], mag[2:], what=f"{what}, finite columns")


def test_weight_gradient_is_deterministic():
    x, z, w, dy = make(32737, 48, seed=5)
    first = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)[2]
    for _ in range(5):
        again = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)[2]
        assert torch.equal(first.view(torch.int16), again.view(torch.int16))
    fp32 = torch.empty(D, dtype=torch.float32, device="cuda")
    cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS, grad_weight=fp32)
    for _ in range(3):
        again = torch.empty_like(fp32)
        cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS, grad_weight=again)
        assert torch.equal(fp32.view(torch.int32), again.view(torch.int32))


@pytest.mark.parametrize(
    "T,H",
    [
        pytest.param(T, H, id=f"{T}x{H}")
        for T, H in (
            (30193, 48),
            (32388, 48),
            (32737, 48),
            (1, 1),
            (1, 48),
            (3, 48),
            (17, 7),
            (255, 2),
            (33, 1),
            (2, 64),
            (1031, 48),
            (4097, 16),
        )
    ],
)
def test_outputs_are_the_same_bits_at_every_call(T, H):
    """``REPEATS`` calls of the forward and of the backward (BF16, FP32 and
    accumulated FP32 weight gradient), on the stream's workspace and on a
    caller workspace, give the bits of the first call; the caller workspace is
    zero after them."""
    x, z, w, dy = make(T, H, seed=5)
    want = outputs(x, z, w, dy)
    ws = workspace_for(T * H, x.device)
    for i in range(REPEATS):
        assert_outputs_equal(outputs(x, z, w, dy), want, f"{T}x{H} call {i}")
        assert_outputs_equal(
            outputs(x, z, w, dy, workspace=ws),
            want,
            f"{T}x{H} call {i}, caller workspace",
        )
    assert_zero(ws)


# Row counts at the edges of the grids on this device: the forward's
# ``persistent_from_rows_per_sm * SMs`` (a row less: the most rows of one CTA a
# block; from there whole waves) and its fewest whole waves of CTAs that each
# walk ``max_blocks_per_cta`` row blocks at or above that (a row less: a
# partial last block; a row more: another wave), the backward's one wave of
# ``ctas_per_sm * SMs`` row blocks (a row less: a partial last block; a row and
# a block more, and a row more than two waves: CTAs claim blocks), one row and
# one block and a row.
GRID_EDGES = (
    "one_row",
    "one_block+1",
    "fwd_persistent-1",
    "fwd_persistent",
    "fwd_persistent+1",
    "fwd_waves-1",
    "fwd_waves",
    "fwd_waves+1",
    "bwd_one_wave-1",
    "bwd_one_wave",
    "bwd_one_wave+1",
    "bwd_one_wave+block",
    "bwd_two_waves+1",
)


def _persistent_waves_rows(rule, sm_count):
    """``(start, rows)``: the forward's first row count of whole waves on
    ``sm_count`` SMs and the fewest rows at or above it that give exactly
    whole waves of ``max_blocks_per_cta`` blocks a CTA."""
    start = rule["persistent_from_rows_per_sm"] * sm_count
    waves_rows = (
        rule["rows_per_block"]
        * rule["ctas_per_sm"]
        * sm_count
        * rule["max_blocks_per_cta"]
    )
    return start, max(1, -(-start // waves_rows)) * waves_rows


def _edge_rows(edge):
    arch, sms = _device()
    fwd = loader.MODULES[loader.route_module(arch, "fwd")]["grid_rule"]
    bwd = loader.MODULES[loader.route_module(arch, "bwd_bf16")]["grid_rule"]
    start, waves = _persistent_waves_rows(fwd, sms)
    bwd_wave = bwd["rows_per_block"] * bwd["ctas_per_sm"] * sms
    return {
        "one_row": 1,
        "one_block+1": bwd["rows_per_block"] + 1,
        "fwd_persistent-1": start - 1,
        "fwd_persistent": start,
        "fwd_persistent+1": start + 1,
        "fwd_waves-1": waves - 1,
        "fwd_waves": waves,
        "fwd_waves+1": waves + 1,
        "bwd_one_wave-1": bwd_wave - 1,
        "bwd_one_wave": bwd_wave,
        "bwd_one_wave+1": bwd_wave + 1,
        "bwd_one_wave+block": bwd_wave + bwd["rows_per_block"],
        "bwd_two_waves+1": 2 * bwd_wave + 1,
    }[edge]


@pytest.mark.parametrize("edge", GRID_EDGES)
def test_grid_edges_on_this_device(edge):
    """At each of ``GRID_EDGES`` (one flat ``[rows, 128]`` input):
    ``check_all``, and ``REPEATS`` calls on a caller workspace give the bits of
    the first call; the workspace is zero after them."""
    rows = _edge_rows(edge)
    x, z, w, dy = make(rows, 1, layout="flat", seed=36)
    check_all(x, z, w, dy, EPS)
    want = outputs(x, z, w, dy)
    ws = workspace_for(rows, x.device)
    for i in range(REPEATS):
        assert_outputs_equal(
            outputs(x, z, w, dy, workspace=ws), want, f"{edge} ({rows} rows) call {i}"
        )
    assert_zero(ws)


def _packed_route():
    """``(arch, the packed forward)`` of this device; skips without one."""
    arch, _ = _device()
    name = loader.ROUTES[arch].get("fwd_packed")
    if name is None:
        pytest.skip(f"{arch} routes no forward of packed rows")
    return arch, name


# Contiguous shapes of both forward routes: the issue's, small and odd ones.
ROUTE_SHAPES = ((30193, 48), (32737, 48), (2, 64), (3, 48), (17, 7), (255, 2))
ROUTE_SHAPES += ((33, 1), (1031, 48), (4097, 16))
ROUTE_DATA = ("normal", "near_zero", "saturated", "gate_edges", "non_finite")


def _route_inputs(shape, data):
    """``(x, z, w, tokens)`` of one ``ROUTE_SHAPES`` shape (``TxH``, contiguous)
    or one ``GRID_EDGES`` row count (``edge:<name>``, flat ``[rows, 128]``)
    with ``data``: ``make``'s, gate edge values, or infinite and NaN gates and
    infinite ``x`` in every fifth row."""
    if shape.startswith("edge:"):
        T, H, layout = _edge_rows(shape[len("edge:") :]), 1, "flat"
    else:
        T, H = (int(v) for v in shape.split("x"))
        layout = "contiguous"
    kind = data if data in ("near_zero", "saturated") else "normal"
    x, z, w, _ = make(T, H, layout=layout, data=kind, seed=41)
    if data == "gate_edges":
        _gate_edges(z, 42)
    elif data == "non_finite":
        zr, xr = z.view(-1, D), x.view(-1, D)
        zr[::5, 7] = float("inf")
        zr[1::5, 9] = float("-inf")
        zr[2::5, 11] = float("nan")
        xr[3::5, 13] = float("inf")
    return x, z, w, T


@pytest.mark.parametrize("data", ROUTE_DATA)
@pytest.mark.parametrize(
    "shape",
    [*(f"{T}x{H}" for T, H in ROUTE_SHAPES), *(f"edge:{e}" for e in GRID_EDGES)],
)
def test_forward_routes_give_the_same_bits(shape, data, launches, monkeypatch):
    """On an architecture that routes a forward of packed rows: ``REPEATS``
    forwards of contiguous inputs run it (a single token runs ``fwd``) and give
    the bits of the ``fwd`` program on the same inputs (``forward_module``
    patched to ``fwd``), finite or not."""
    arch, packed = _packed_route()
    x, z, w, tokens = _route_inputs(shape, data)
    strided = loader.ROUTES[arch]["fwd"]
    launches.clear()
    got = [forward(x, z, w, EPS) for _ in range(REPEATS)]
    assert launches == [packed if tokens > 1 else strided] * REPEATS, launches
    monkeypatch.setattr(
        loader,
        "forward_module",
        lambda arch, rows, heads, sm_count, *strides: loader.module_for_rows(
            arch, "fwd", rows, sm_count
        ),
    )
    launches.clear()
    want = forward(x, z, w, EPS)
    assert launches == [strided], launches
    for i, y in enumerate(got):
        assert_same_bits(y, want, f"{shape} {data} call {i}")


@pytest.mark.parametrize(
    "layout,T,route",
    [
        ("contiguous", 65, "fwd_packed"),
        ("flat", 65, "fwd_packed"),
        ("misaligned", 65, "fwd_packed"),
        ("contiguous", 1, "fwd"),
        ("projection", 65, "fwd"),
        ("head_major", 65, "fwd"),
        ("broadcast", 65, "fwd"),
    ],
)
def test_forward_route_follows_the_layout(layout, T, route, launches):
    """On an architecture that routes a forward of packed rows, inputs the
    kernel reads as packed rows (contiguous, flat, and misaligned ones once
    copied) run it; a single token (token stride 0), a gate sliced from a
    projection, head-major rows and a token-stride-0 broadcast run ``fwd``.
    Each gives the bits of its rows as contiguous tensors."""
    arch, _ = _packed_route()
    H = 48
    if layout == "broadcast":
        x, z, w, _ = make(1, H, seed=43)
        x, z = x.expand(T, H, D), z.expand(T, H, D)
    else:
        x, z, w, _ = make(T, H, layout=layout, seed=43)
    launches.clear()
    y = forward(x, z, w, EPS)
    assert launches == [loader.ROUTES[arch][route]], (layout, launches)
    xc, zc = (t.reshape(T, H, D).contiguous() for t in (x, z))
    assert_same_bits(y.reshape(T, H, D), forward(xc, zc, w, EPS), layout)


def test_forward_module_runs_packed_rows_only():
    """``forward_module`` (host only): an architecture's ``fwd_packed`` program
    for head strides of 128 and token strides of ``H * 128`` on both ``x`` and
    ``z`` and fewer than ``PACKED_ROWS`` rows; ``fwd`` for any other strides, a
    single token (token stride 0), a token-stride-0 broadcast and from
    ``PACKED_ROWS`` rows, and always on an architecture without one."""
    assert loader.PACKED_ROLES == ("fwd_packed",)
    assert loader.PACKED_ROWS == 2**31 // D
    H, last = 48, loader.PACKED_ROWS - 1
    row = H * D
    for arch in loader.ARCHES:
        fwd = loader.ROUTES[arch]["fwd"]
        packed = loader.ROUTES[arch].get("fwd_packed", fwd)
        for rows, heads, strides, want in (
            (65 * H, H, (row, D, row, D), packed),
            (65, 1, (D, D, D, D), packed),
            (last, 1, (D, D, D, D), packed),
            (loader.PACKED_ROWS, 1, (D, D, D, D), fwd),
            (2**31 - 1, 1, (D, D, D, D), fwd),
            (H, H, (0, D, 0, D), fwd),
            (65 * H, H, (0, D, 0, D), fwd),
            (65 * H, H, (row, D, 2 * row + 16, D), fwd),
            (65 * H, H, (2 * row + 16, D, row, D), fwd),
            (65 * H, H, (D, 65 * D, D, 65 * D), fwd),
            (65 * H, H, (row, D, D, 65 * D), fwd),
        ):
            got = loader.forward_module(arch, rows, heads, 148, *strides)
            assert got == want, (arch, rows, heads, strides)


# The rows per SM from which each architecture's backward runs the long module
# of its role on packed rows (the long records' ``min_rows_per_sm``).
LONG_MIN_ROWS_PER_SM = {"sm_100a": 2560, "sm_103a": 2560, "sm_107a": 1536}
# The roles of ``_long_calls``' backward calls, in order.
LONG_CALL_ROLES = ("bwd_bf16", "bwd_bf16", "bwd_bf16", "bwd_f32", "bwd_f32")


def _long_calls(dy, x, z, w, ws):
    """The backward calls of the long-module tests, of the roles
    ``LONG_CALL_ROLES``: a BF16 weight gradient, one accumulated onto 0.5 (on
    ``ws``), ``compute_grad_weight=False``, an FP32 one (on ``ws``) and one
    accumulated onto 0.25.  Their ``dx``, ``dz`` and weight gradients."""
    device = x.device
    bf16 = torch.full((D,), 0.5, dtype=torch.bfloat16, device=device)
    f32 = torch.empty(D, dtype=torch.float32, device=device)
    acc = torch.full((D,), 0.25, dtype=torch.float32, device=device)
    got = []
    for kwargs in (
        {},
        {"grad_weight": bf16, "accumulate_grad_weight": True, "workspace": ws},
        {"compute_grad_weight": False},
        {"grad_weight": f32, "workspace": ws},
        {"grad_weight": acc, "accumulate_grad_weight": True},
    ):
        dx, dz, dw = backward(dy, x, z, w, EPS, **kwargs)
        got += [dx, dz] if dw is None else [dx, dz, dw]
    return got


def _route_to(suffix):
    """A ``backward_module`` that names the module of the call's role
    (``suffix`` ``""``) or of its long role (``LONG``) whatever the call."""

    def module(arch, role, *_strides):
        return loader.ROUTES[arch][role + suffix]

    return module


# Layouts around this device's first row count of the long modules: flat
# ``[rows, 128]`` rows a row below, at and a row above it; ``[T, 48, 128]``
# rows at the fewest tokens that reach it (contiguous a token fewer too,
# misaligned ones copied by the host, a gate and ``grad_out`` sliced from
# projections, head-major ``x`` and ``z``, ``grad_out`` alone sliced from a
# projection, a token-stride-0 broadcast of ``x`` and ``z``); a single token
# of that many heads.
LONG_ROUTE_LAYOUTS = (
    "flat-1",
    "flat",
    "flat+1",
    "contiguous-1",
    "contiguous",
    "misaligned",
    "projection",
    "head_major",
    "dy_projection",
    "broadcast",
    "single_token",
)
# The layouts the kernel reads as packed rows at or above that row count.
LONG_ROUTED = ("flat", "flat+1", "contiguous", "misaligned")


@pytest.mark.parametrize("layout", LONG_ROUTE_LAYOUTS)
def test_backward_route_follows_the_layout_and_rows(layout, launches):
    """On this device a backward runs its role's long module when the kernel
    reads ``grad_out``, ``x`` and ``z`` as packed rows (``LONG_ROUTED``) from
    ``min_rows_per_sm * SMs`` rows, and its role's module a row or a token
    fewer, on strided, head-major and broadcast inputs and for a single
    token, for a BF16 and an FP32 weight gradient, accumulated or not, and
    for ``compute_grad_weight=False``; the workspace is zero after the calls,
    and the long module gives the bits of the role's module on the same
    calls."""
    arch, sms = _device()
    start = LONG_MIN_ROWS_PER_SM[arch] * sms
    H = 48
    tokens = -(-start // H)
    if layout.startswith("flat"):
        rows = start + {"flat-1": -1, "flat": 0, "flat+1": 1}[layout]
        x, z, w, dy = make(rows, 1, layout="flat", seed=44)
    elif layout == "single_token":
        x, z, w, dy = make(1, start, seed=44)
    elif layout == "broadcast":
        x, z, w, _ = make(1, H, seed=44)
        x, z = x.expand(tokens, H, D), z.expand(tokens, H, D)
        dy = make(tokens, H, seed=45)[3]
    elif layout == "dy_projection":
        x, z, w, _ = make(tokens, H, seed=44)
        dy = make(tokens, H, layout="projection", seed=45)[3]
    else:
        T = tokens - 1 if layout == "contiguous-1" else tokens
        x, z, w, dy = make(T, H, layout=layout.removesuffix("-1"), seed=44)
    long = layout in LONG_ROUTED
    rows = x.numel() // D
    assert (rows >= start) == (long or layout not in ("flat-1", "contiguous-1"))
    routes = loader.ROUTES[arch]
    ws = workspace_for(rows, x.device)
    launches.clear()
    got = _long_calls(dy, x, z, w, ws)
    suffix = loader.LONG if long else ""
    want = [routes[role + suffix] for role in LONG_CALL_ROLES]
    assert launches == want, (layout, launches)
    assert_zero(ws)
    if not long:
        return
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pkg, "backward_module", _route_to(""))
        launches.clear()
        want = _long_calls(dy, x, z, w, ws)
        assert launches == [routes[role] for role in LONG_CALL_ROLES], launches
    assert_zero(ws)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert_same_bits(a, b, f"{layout} ({rows} rows): output {i}")


# Packed shapes on which each long module is compared with its role's module:
# small and odd ones, the issue's (blocks far past the prefetch share of the
# CTAs), and flat rows around this device's first row count of the long
# modules.  Up to ``LONG_BIT_GRID_ROWS`` rows also on the grids of
# ``LONG_BIT_SM_COUNTS`` SMs, whose CTAs claim many blocks each.
LONG_BIT_SHAPES = ("1x1", "33x1", "17x7", "2049x33", "4097x33", "32737x48")
LONG_BIT_SHAPES += ("long-1", "long", "long+1")
LONG_BIT_SM_COUNTS = (1, 16, 66)
LONG_BIT_GRID_ROWS = 4097 * 33


@pytest.mark.parametrize("shape", LONG_BIT_SHAPES)
def test_long_modules_give_the_bits_of_their_roles(shape, launches):
    """Each backward role's long module (forced through ``backward_module``)
    gives the bits of the role's module on the same packed inputs and grid:
    ``dx``, ``dz``, the BF16 and FP32 weight gradients, accumulated or not,
    and ``compute_grad_weight=False``, on this device's grid and, up to
    ``LONG_BIT_GRID_ROWS`` rows, on the grids of ``LONG_BIT_SM_COUNTS`` SMs
    (``device_sm_count`` patched); the workspace is zero after the calls."""
    arch, sms = _device()
    if shape.startswith("long"):
        start = LONG_MIN_ROWS_PER_SM[arch] * sms
        rows = start + {"long-1": -1, "long": 0, "long+1": 1}[shape]
        x, z, w, dy = make(rows, 1, layout="flat", seed=46)
    else:
        T, H = (int(v) for v in shape.split("x"))
        rows = T * H
        x, z, w, dy = make(T, H, seed=46)
    ws = workspace_for(rows, x.device)
    routes = loader.ROUTES[arch]
    counts = [sms, *LONG_BIT_SM_COUNTS] if rows <= LONG_BIT_GRID_ROWS else [sms]
    for sm_count in counts:
        runs = []
        for suffix in ("", loader.LONG):
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(pkg, "backward_module", _route_to(suffix))
                patch.setattr(
                    loader, "device_sm_count", lambda index, count=sm_count: count
                )
                launches.clear()
                runs.append(_long_calls(dy, x, z, w, ws))
                want = [routes[role + suffix] for role in LONG_CALL_ROLES]
                assert launches == want, (suffix, launches)
            assert_zero(ws)
        what = f"{shape} on the grids of {sm_count} SMs"
        for i, (a, b) in enumerate(zip(runs[1], runs[0], strict=True)):
            assert_same_bits(a, b, f"{what}: output {i}")


def test_claimed_backward_matches_fp64():
    """65536 x 48 rows, many times one wave of the backward's row blocks on
    any device, so its CTAs claim chunks of blocks over several rounds:
    ``check_all``; the FP32 and the accumulated FP32 weight gradient within the
    budget and the same bits at a second call; the workspace zero after every
    call."""
    arch, sms = _device()
    T, H = 65536, 48
    rule = loader.MODULES[loader.route_module(arch, "bwd_bf16")]["grid_rule"]
    blocks = -(-T * H // rule["rows_per_block"])
    assert blocks >= 16 * rule["ctas_per_sm"] * sms, (blocks, sms)
    x, z, w, dy = make(T, H, seed=37)
    check_all(x, z, w, dy, EPS)
    dw64, mag = dw_reference(dy, x, z, EPS)
    ws = workspace_for(T * H, x.device)
    runs = []
    for _ in range(2):
        f32 = torch.empty(D, dtype=torch.float32, device=x.device)
        backward(dy, x, z, w, EPS, grad_weight=f32, workspace=ws)
        assert_zero(ws)
        acc = torch.full((D,), 0.25, dtype=torch.float32, device=x.device)
        backward(
            dy, x, z, w, EPS, grad_weight=acc, accumulate_grad_weight=True, workspace=ws
        )
        assert_zero(ws)
        assert_dw_within(f32, dw64, mag, what="FP32 dw")
        assert_dw_within(acc, dw64 + 0.25, mag, what="accumulated FP32 dw")
        runs.append((f32, acc))
    for a, b, name in zip(runs[1], runs[0], ("dw_f32", "dw_accumulated"), strict=True):
        assert_same_bits(a, b, f"{name} at the second call")


# SM counts whose grids the host computes on this device: small MIG-like
# instances, the products' counts and one larger than any device.
OTHER_SM_COUNTS = (1, 7, 16, 24, 66, 148, 152, 160, 212, 216, 1024)


@pytest.mark.parametrize("sm_count", OTHER_SM_COUNTS)
def test_grids_of_other_sm_counts(sm_count):
    """The grids computed for ``sm_count`` SMs (``device_sm_count`` patched)
    on this device: ``y``, ``dx`` and ``dz`` are the bits of the device's own
    grids; the BF16, FP32 and accumulated FP32 weight gradients lie within the
    budget and are the same bits at a second call; the workspace is zero after
    every call.  On few SMs the larger shapes reach the forward's whole
    waves."""
    for T, H in ((17, 7), (257, 48), (1031, 48), (4097, 16)):
        x, z, w, dy = make(T, H, seed=38)
        want = outputs(x, z, w, dy)
        dw64, mag = dw_reference(dy, x, z, EPS)
        ws = workspace_for(T * H, x.device)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(loader, "device_sm_count", lambda index: sm_count)
            got = outputs(x, z, w, dy, workspace=ws)
            assert_zero(ws)
            again = outputs(x, z, w, dy, workspace=ws)
            assert_zero(ws)
        what = f"{T}x{H} on the grids of {sm_count} SMs"
        for a, b, name in zip(got[:3], want[:3], OUTPUTS, strict=False):
            assert_same_bits(a, b, f"{what}: {name}")
        assert_outputs_equal(again, got, f"{what}, second call")
        assert_dw_within(got[3], dw64, mag, what=f"{what}: BF16 dw")
        assert_dw_within(got[4], dw64, mag, what=f"{what}: FP32 dw")
        assert_dw_within(got[5], dw64 + 0.25, mag, what=f"{what}: accumulated dw")


def test_weight_gradient_accumulation():
    x, z, w, dy = make(1031, 48, seed=6)
    x2, z2, _, dy2 = make(1031, 48, seed=7)
    _, _, _, dw64 = fp64_reference(dy, x, z, w, EPS)
    _, _, _, dw64b = fp64_reference(dy2, x2, z2, w, EPS)
    main = torch.full((D,), 0.25, dtype=torch.float32, device="cuda")
    cake_rmsnorm_silu_gate_det_train_backward(
        dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True
    )
    cake_rmsnorm_silu_gate_det_train_backward(
        dy2, x2, z2, w, EPS, grad_weight=main, accumulate_grad_weight=True
    )
    want = 0.25 + dw64 + dw64b
    assert torch.allclose(main.double(), want, rtol=1e-5, atol=1e-3)
    # autograd accumulation into .grad over two microbatches
    xa = x.clone().requires_grad_(True)
    wa = w.clone().requires_grad_(True)
    za = z.clone().requires_grad_(True)
    cake_rmsnorm_silu_gate_det(xa, za, wa, EPS).backward(dy)
    cake_rmsnorm_silu_gate_det(xa, za, wa, EPS).backward(dy)
    single = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)
    assert torch.equal(xa.grad, single[0] + single[0])
    assert torch.equal(za.grad, single[1] + single[1])
    assert torch.equal(wa.grad, single[2] + single[2])


def test_autograd_matches_functional_and_checkpoint():
    x, z, w, dy = make(513, 48, seed=8)
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    y = cake_rmsnorm_silu_gate_det(xa, za, wa, EPS)
    y.backward(dy)
    y_ref = cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS)
    dx, dz, dw = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)
    assert torch.equal(y, y_ref)
    assert (
        torch.equal(xa.grad, dx)
        and torch.equal(za.grad, dz)
        and torch.equal(wa.grad, dw)
    )

    # Saved state: only the inputs, no FP32 temporary.
    saved = []

    def pack(t):
        saved.append((t.dtype, t.numel(), t.data_ptr()))
        return t

    xb, zb, wb = (t.clone().requires_grad_(True) for t in (x, z, w))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        cake_rmsnorm_silu_gate_det(xb, zb, wb, EPS)
    assert sorted(saved) == sorted(
        [
            (torch.bfloat16, x.numel(), xb.data_ptr()),
            (torch.bfloat16, z.numel(), zb.data_ptr()),
            (torch.bfloat16, D, wb.data_ptr()),
        ]
    )

    # Non-reentrant activation checkpointing replays the forward in the backward.
    xc, zc, wc = (t.clone().requires_grad_(True) for t in (x, z, w))
    yc = torch.utils.checkpoint.checkpoint(
        cake_rmsnorm_silu_gate_det, xc, zc, wc, EPS, use_reentrant=False
    )
    yc.backward(dy)
    assert torch.equal(yc, y_ref)
    assert (
        torch.equal(xc.grad, dx)
        and torch.equal(zc.grad, dz)
        and torch.equal(wc.grad, dw)
    )


def _step(x, z, w, dy, *, checkpoint=False):
    """One autograd step: ``(y, dx, dz, dw)`` (``dw`` BF16), the forward under
    non-reentrant checkpointing when asked."""
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    if checkpoint:
        y = torch.utils.checkpoint.checkpoint(
            cake_rmsnorm_silu_gate_det, xa, za, wa, EPS, use_reentrant=False
        )
    else:
        y = cake_rmsnorm_silu_gate_det(xa, za, wa, EPS)
    y.backward(dy)
    return y.detach(), xa.grad, za.grad, wa.grad


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_steps_neither_raise_nor_warn_under_deterministic_algorithms(warn_only):
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and
    ``warn_only``) with warnings as errors, the functional calls (on the
    stream's workspace and on a caller workspace, after its reset), an
    autograd step and a non-reentrant checkpointed step raise nothing, warn
    nothing and give the bits of the same calls with the setting off."""
    x, z, w, dy = make(1031, 48, seed=35)
    ws = workspace_for(1031 * 48, x.device)

    def run():
        reset_workspace(ws)
        return (
            *outputs(x, z, w, dy),
            *outputs(x, z, w, dy, workspace=ws),
            *_step(x, z, w, dy),
            *_step(x, z, w, dy, checkpoint=True),
        )

    want = run()
    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = run()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert_same_bits(a, b, f"warn_only={warn_only} output {i}")
    assert_zero(ws)


@pytest.mark.parametrize("T,H", [(1, 1), (17, 7), (1031, 48), (32737, 48)])
def test_each_call_launches_one_module_of_its_role(T, H, launches):
    """One module a call: the forward of contiguous inputs runs
    ``forward_module`` of their strides (``ROUTES[arch]["fwd_packed"]`` where
    routed, except for a single token; else ``ROUTES[arch]["fwd"]``); a
    backward runs one kernel, ``backward_module`` of its role and strides:
    ``bwd_bf16`` for a BF16 weight gradient and for
    ``compute_grad_weight=False``, ``bwd_f32`` for an FP32 one, accumulating
    or not, each its long module (``LONG_ROLES``) where these packed rows
    reach the long module's ``min_rows_per_sm`` rows per SM (32737 x 48 on
    every product); the reset launches no module."""
    arch, sms = _device()
    routes = loader.ROUTES[arch]
    for role in loader.ROLES:
        assert loader.module_for_rows(arch, role, T * H, sms) == routes[role]
    tok = H * D if T > 1 else 0
    forward_name = loader.forward_module(arch, T * H, H, sms, tok, D, tok, D)
    packed = routes.get("fwd_packed", routes["fwd"])
    assert forward_name == (packed if T > 1 else routes["fwd"])
    long = T > 1 and LONG_MIN_ROWS_PER_SM[arch] * sms <= T * H
    names = {"fwd": forward_name}
    for role in ("bwd_bf16", "bwd_f32"):
        names[role] = routes[role + loader.LONG] if long else routes[role]
        got = loader.backward_module(arch, role, T * H, H, sms, tok, D, tok, D, tok, D)
        assert got == names[role], (role, got)
    x, z, w, dy = make(T, H, seed=39)
    ws = workspace_for(T * H, x.device)
    bf16 = torch.zeros(D, dtype=torch.bfloat16, device=x.device)
    f32 = torch.zeros(D, dtype=torch.float32, device=x.device)
    calls = (
        ("fwd", lambda: forward(x, z, w, EPS)),
        ("bwd_bf16", lambda: backward(dy, x, z, w, EPS)),
        (
            "bwd_bf16",
            lambda: backward(
                dy,
                x,
                z,
                w,
                EPS,
                grad_weight=bf16,
                accumulate_grad_weight=True,
                workspace=ws,
            ),
        ),
        (
            "bwd_bf16",
            lambda: backward(
                dy, x, z, w, EPS, grad_weight=f32, compute_grad_weight=False
            ),
        ),
        ("bwd_f32", lambda: backward(dy, x, z, w, EPS, grad_weight=f32, workspace=ws)),
        (
            "bwd_f32",
            lambda: backward(
                dy, x, z, w, EPS, grad_weight=f32, accumulate_grad_weight=True
            ),
        ),
        (None, lambda: reset_workspace(ws)),
    )
    for role, call in calls:
        launches.clear()
        call()
        want = [] if role is None else [names[role]]
        assert launches == want, (role, launches)
    assert_zero(ws)


def test_compute_grad_weight_false_touches_no_weight_state(launches):
    """``compute_grad_weight=False`` returns ``None`` for the weight gradient,
    leaves ``grad_weight`` as it is and gives the ``dx`` and ``dz`` bits of a
    call that computes it; its launch (the BF16 backward module) still takes
    the arrival ticket and claims row blocks, and leaves a caller workspace
    and, without one, the stream's workspace zero.  Autograd with a weight that
    needs no gradient gives the same ``dx`` and ``dz``."""
    arch, _ = _device()
    x, z, w, dy = make(257, 48, seed=12)
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    ws = workspace_for(257 * 48, x.device)
    main = torch.full((D,), 3.0, dtype=torch.float32, device="cuda")
    launches.clear()
    dx, dz, dw = backward(
        dy, x, z, w, EPS, grad_weight=main, compute_grad_weight=False, workspace=ws
    )
    assert launches == [loader.route_module(arch, "bwd_bf16")]
    assert dw is None
    assert_same_bits(dx, dx_d, "dx")
    assert_same_bits(dz, dz_d, "dz")
    torch.cuda.synchronize()
    assert torch.equal(main, torch.full_like(main, 3.0))
    assert_zero(ws)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pkg, "_STREAM_WORKSPACES", {})
        dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
        assert dw is None
        assert_same_bits(dx, dx_d, "dx on the stream's workspace")
        assert_same_bits(dz, dz_d, "dz on the stream's workspace")
        key = (torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream)
        assert list(pkg._STREAM_WORKSPACES) == [key]
        for view in pkg._STREAM_WORKSPACES[key]:
            assert_zero(view, "the stream's workspace")
    xa, za = (t.clone().requires_grad_(True) for t in (x, z))
    cake_rmsnorm_silu_gate_det(xa, za, w, EPS).backward(dy)
    assert_same_bits(xa.grad, dx_d, "autograd dx")
    assert_same_bits(za.grad, dz_d, "autograd dz")
    assert w.grad is None


def test_workspace_size_and_layout():
    """The backward's workspace: ``workspace_elements`` FP32 elements (4738,
    18952 bytes) for every backward module (both weight-gradient dtypes and
    their long modules, so one stream workspace serves them all) and every row
    count, zero-filled, 16-byte aligned; negative rows are a ``ValueError``.
    Its views are the ``uint64`` digit counters (``slots * limbs * 128``), one
    ``uint32`` flag word a column and the two ``uint32`` words of the arrival
    ticket and the claim counter, in that order."""
    device = torch.device("cuda")
    arch, _ = _device()
    names = [
        loader.route_module(arch, role)
        for role in ("bwd_bf16", "bwd_f32", *loader.LONG_ROLES)
    ]
    assert len(set(names)) == 4
    elements = loader.workspace_elements(names[0])
    for name in names:
        shape = loader.MODULES[name]["workspace"]
        assert (
            loader.workspace_elements(name)
            == elements
            == 2 * shape["slots"] * shape["limbs"] * D + D + 2
        )
    assert elements == 4738
    for rows in (0, 1, 129 * 48, 10**7):
        ws = workspace_for(rows, device)
        assert ws.dtype == torch.float32 and ws.shape == (elements,), rows
        assert ws.is_contiguous() and ws.data_ptr() % 16 == 0, rows
        assert ws.device.type == "cuda", rows
        assert_zero(ws, f"the workspace for {rows} rows")
        assert workspace_bytes(rows, device) == elements * 4 == 18952, rows
    for entry in (workspace_for, workspace_bytes):
        with pytest.raises(ValueError, match="non-negative"):
            entry(-1, device)
    for name in names:
        shape = loader.MODULES[name]["workspace"]
        acc, flags, done = pkg._workspace_views(ws, name, ws.device)
        assert acc.dtype == torch.uint64
        assert acc.numel() == shape["slots"] * shape["limbs"] * D
        assert flags.dtype == torch.uint32 and flags.numel() == D
        assert done.dtype == torch.uint32 and done.numel() == 2
        assert acc.data_ptr() == ws.data_ptr()
        assert flags.data_ptr() == ws.data_ptr() + 8 * acc.numel()
        assert done.data_ptr() == flags.data_ptr() + 4 * D


def test_workspace_is_zero_after_every_call_and_serves_every_shape():
    """One caller workspace, never reset, over a sequence of shapes, both
    weight-gradient dtypes, accumulation, ``compute_grad_weight=False`` and a
    call with no rows: it is zero after every call, every call gives the bits
    of the same call on the stream's workspace, ``dx`` and ``dz`` lie within
    ``check_all``'s bounds and the weight gradient within budget."""
    device = torch.device("cuda")
    ws = workspace_for(1, device)
    # (T, H, weight-gradient dtype, accumulate, compute the weight gradient)
    cases = (
        (1031, 48, torch.bfloat16, False, True),
        (7, 48, torch.float32, False, True),
        (4097, 16, torch.bfloat16, True, True),
        (257, 48, torch.float32, False, False),
        (1, 1, torch.float32, True, True),
        (0, 48, torch.float32, True, True),
        (30193, 48, torch.bfloat16, False, True),
        (33, 7, torch.float32, True, True),
        (65, 48, torch.bfloat16, False, False),
        (2, 64, torch.bfloat16, True, True),
    )
    for i, (T, H, dtype, accumulate, compute) in enumerate(cases):
        what = f"{T}x{H} {dtype} accumulate={accumulate} compute={compute}"
        x, z, w, dy = make(T, H, seed=20 + i)
        start = 0.5 if accumulate else 0.0

        kwargs = dict(accumulate_grad_weight=accumulate, compute_grad_weight=compute)
        got = backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=torch.full((D,), start, dtype=dtype, device=device),
            workspace=ws,
            **kwargs,
        )
        assert_zero(ws, f"the workspace after {what}")
        want = backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=torch.full((D,), start, dtype=dtype, device=device),
            **kwargs,
        )
        for a, b, name in zip(got, want, ("dx", "dz", "dw"), strict=True):
            if b is None:
                assert a is None, (what, name)
            else:
                assert_same_bits(a, b, f"{what}: {name}")
        assert (got[2] is None) == (not compute), what
        if T == 0:
            if compute:
                assert torch.equal(got[2], torch.full_like(got[2], start)), what
            continue
        _, dx64, dz64, dw64 = fp64_reference(dy, x, z, w, EPS)
        assert_bf16_close(got[0], dx64, what=f"{what}: dx")
        assert_bf16_close(got[1], dz64, what=f"{what}: dz")
        if compute:
            assert got[2].dtype == dtype
            _, mag = dw_reference(dy, x, z, EPS)
            assert_dw_within(got[2], dw64 + start, mag, what=f"{what}: dw")


def test_calls_without_workspace_share_one_buffer_per_stream():
    """Calls without a caller workspace (both weight-gradient dtypes) share the
    zero-filled workspace of their (device, stream), left zero; another stream
    gets its own."""
    device = torch.cuda.current_device()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pkg, "_STREAM_WORKSPACES", {})
        x, z, w, dy = make(65, 7)
        backward(dy, x, z, w, EPS)
        f32 = torch.empty(D, dtype=torch.float32, device=x.device)
        backward(dy, x, z, w, EPS, grad_weight=f32)
        key = (device, torch.cuda.current_stream().cuda_stream)
        assert list(pkg._STREAM_WORKSPACES) == [key]
        for view in pkg._STREAM_WORKSPACES[key]:
            assert_zero(view, "the stream's workspace")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            backward(dy, x, z, w, EPS)
        side.synchronize()
        assert set(pkg._STREAM_WORKSPACES) == {key, (device, side.cuda_stream)}
        for view in pkg._STREAM_WORKSPACES[(device, side.cuda_stream)]:
            assert_zero(view, "the side stream's workspace")


def test_checkpointed_steps_keep_only_the_stream_workspace():
    """A non-reentrant checkpointed forward keeps nothing alive but its output,
    and once the first step has run, every checkpointed forward and backward
    leaves ``torch.cuda.memory_allocated()`` unchanged.  The buffers kept
    across steps are the stream's workspace of the backward without a caller
    workspace (``backward_workspace_bytes`` bytes) and the device's reserve of
    zero-filled workspaces for CUDA Graph captures (one allocation)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pkg, "_STREAM_WORKSPACES", {})
        patch.setattr(pkg, "_RESERVES", {})
        # Small enough for the caching allocator's small pool, whose blocks are
        # the requested sizes rounded up to 512 bytes.
        x, z, w, dy = make(17, 48, seed=63)
        leaves = [t.clone().requires_grad_(True) for t in (x, z, w)]
        for leaf in leaves:
            leaf.grad = torch.zeros_like(leaf)

        def checkpointed():
            return torch.utils.checkpoint.checkpoint(
                cake_rmsnorm_silu_gate_det, *leaves, EPS, use_reentrant=False
            )

        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        checkpointed().backward(dy)  # loads the modules, allocates the workspace
        torch.cuda.synchronize()
        kept = torch.cuda.memory_allocated() - before
        nbytes = workspace_bytes(1, x.device)
        key = (torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream)
        assert list(pkg._STREAM_WORKSPACES) == [key]
        acc, _, _ = pkg._STREAM_WORKSPACES[key]
        assert acc.untyped_storage().nbytes() == nbytes
        (block,) = pkg._RESERVES[torch.cuda.current_device()]["blocks"]
        reserved = block.untyped_storage().nbytes()
        # 256-byte aligned workspaces, one after the other
        stride = -(-(nbytes // 4) // 64) * 64
        assert reserved == pkg._RESERVE_SIZE * stride * 4, reserved
        assert nbytes + reserved <= kept < nbytes + reserved + 512, (kept, nbytes)
        base = torch.cuda.memory_allocated()
        for step in range(3):
            y = checkpointed()
            torch.cuda.synchronize()
            retained = (
                torch.cuda.memory_allocated() - base - y.untyped_storage().nbytes()
            )
            assert retained == 0, (step, retained)
            y.backward(dy)
            del y
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == base, step
        assert list(pkg._STREAM_WORKSPACES) == [key]


# The texts of the workspace refusals.
NONDETERMINISTIC_WORKSPACE = (
    "flashinfer.cake_rmsnorm_silu_gate_nondet_train."
    "cake_rmsnorm_silu_gate_nondet_train_backward_workspace"
)
DETERMINISTIC_WORKSPACE = (
    "flashinfer.cake_rmsnorm_silu_gate_det_train."
    "cake_rmsnorm_silu_gate_det_train_backward_workspace"
)


def refusal(dtype):
    """The refusal of a workspace of the non-deterministic package."""
    name = {torch.float64: "FP64", torch.uint8: "uint8"}[dtype]
    return (
        f"workspace is {name}, a workspace of the non-deterministic mode "
        f"({NONDETERMINISTIC_WORKSPACE}); the deterministic backward needs its "
        f"own zero-filled FP32 workspace ({DETERMINISTIC_WORKSPACE})"
    )


def unfit(device, elements):
    """The refusal of another unfit buffer."""
    return (
        "workspace must be a contiguous, 16-byte aligned FP32 tensor on "
        f"{device} with at least {elements} elements, zero before its first use "
        "(cake_rmsnorm_silu_gate_det_train_backward_workspace(rows, device), "
        "or cake_rmsnorm_silu_gate_det_train_reset_workspace(workspace))"
    )


def _unfit_buffers(elements):
    """FP32 buffers of ones the backward and the reset refuse: too small, not
    16-byte aligned, strided, and an FP16 one."""
    base = torch.ones(2 * elements + 4, dtype=torch.float32, device="cuda")
    return (
        torch.ones(elements - 1, dtype=torch.float32, device="cuda"),
        base[1 : elements + 1],
        base[: 2 * elements : 2],
        torch.ones(elements, dtype=torch.float16, device="cuda"),
    )


def test_backward_refuses_unfit_workspaces(launches):
    """The backward refuses an FP64 or ``uint8`` tensor (a workspace of the
    non-deterministic package) naming both modes' workspaces, with rows and
    with none, for both weight-gradient dtypes, and keeps ``grad_weight``;
    another unfit buffer (too small, not 16-byte aligned, strided, FP16, on
    the host) gets the FP32 requirement.  Nothing is launched and no refused
    buffer is written."""
    x, z, w, dy = make(65, 48, seed=13)
    empty = torch.empty(0, 48, D, dtype=torch.bfloat16, device="cuda")
    elements = workspace_for(1, x.device).numel()
    for dtype in (torch.float64, torch.uint8):
        other = torch.ones(elements, dtype=dtype, device="cuda")
        for args in ((dy, x, z), (empty, empty, empty)):
            for gdtype in (torch.bfloat16, torch.float32):
                main = torch.full((D,), 1.5, dtype=gdtype, device="cuda")
                with pytest.raises(ValueError) as info:
                    backward(*args, w, EPS, grad_weight=main, workspace=other)
                assert str(info.value) == refusal(dtype), str(info.value)
                assert torch.equal(main, torch.full_like(main, 1.5))
        assert bool(other.eq(1).all())
    for bad in (
        *_unfit_buffers(elements),
        torch.ones(elements, dtype=torch.float32),
    ):
        with pytest.raises(ValueError) as info:
            backward(dy, x, z, w, EPS, workspace=bad)
        assert str(info.value) == unfit(x.device, elements), str(info.value)
        assert bool(bad.eq(1).all())
    assert launches == []


def test_reset_workspace_clears_nonzero_scratch():
    """A caller workspace whose digit counters, flag words, arrival ticket and
    claim counter hold garbage (written by something other than the backward)
    is zeroed by ``reset_workspace``; backward launches at two shapes and
    weight-gradient dtypes then give the bits of a fresh workspace and leave
    it zero."""
    device = torch.device("cuda")
    ws = workspace_for(1, device)
    g = torch.Generator(device="cuda").manual_seed(60)
    ws.view(torch.int32).copy_(
        torch.randint(
            1, 2**31 - 1, ws.shape, dtype=torch.int32, device=device, generator=g
        )
    )
    assert bool(ws.view(torch.int32).ne(0).all())
    reset_workspace(ws)
    assert_zero(ws)
    for i, (T, H, dtype) in enumerate(
        ((1031, 48, torch.bfloat16), (37, 7, torch.float32))
    ):
        x, z, w, dy = make(T, H, seed=61 + i)

        def call(workspace, x=x, z=z, w=w, dy=dy, dtype=dtype):
            gw = torch.empty(D, dtype=dtype, device=device)
            return backward(dy, x, z, w, EPS, grad_weight=gw, workspace=workspace)

        got = call(ws)
        want = call(workspace_for(T * H, device))
        for a, b, name in zip(got, want, ("dx", "dz", "dw"), strict=True):
            assert_same_bits(a, b, f"{T}x{H} {dtype}: {name}")
        assert_zero(ws)


def test_reset_workspace_of_a_zero_workspace_is_a_no_op(launches):
    """Resetting a zero workspace keeps it zero and in place, launches no module
    and does not synchronize the host; the next backward is unaffected."""
    x, z, w, dy = make(129, 48, seed=62)
    want = backward(dy, x, z, w, EPS)
    ws = workspace_for(129 * 48, x.device)
    ptr = ws.data_ptr()
    launches.clear()
    torch.cuda.set_sync_debug_mode("error")
    try:
        reset_workspace(ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert ws.data_ptr() == ptr
    assert_zero(ws)
    assert launches == []
    got = backward(dy, x, z, w, EPS, workspace=ws)
    for a, b, name in zip(got, want, ("dx", "dz", "dw"), strict=True):
        assert_same_bits(a, b, f"{name} after a reset of a zero workspace")
    assert_zero(ws)


def test_reset_workspace_errors():
    """``reset_workspace`` refuses a non-tensor (``TypeError``), the
    non-deterministic package's FP64 and ``uint8`` workspaces naming both
    modes' workspaces, other unfit buffers with the FP32 requirement and a
    host tensor, each left as it is; it takes no mode keyword."""
    device = torch.device("cuda")
    elements = workspace_for(1, device).numel()
    with pytest.raises(TypeError):
        reset_workspace(None)
    for dtype in (torch.float64, torch.uint8):
        other = torch.ones(elements, dtype=dtype, device="cuda")
        with pytest.raises(ValueError) as info:
            reset_workspace(other)
        assert str(info.value) == refusal(dtype), str(info.value)
        assert bool(other.eq(1).all())
    for bad in _unfit_buffers(elements):
        with pytest.raises(ValueError) as info:
            reset_workspace(bad)
        assert str(info.value) == unfit(bad.device, elements), str(info.value)
        assert bool(bad.eq(1).all())
    with pytest.raises(ValueError, match="CUDA"):
        reset_workspace(torch.zeros(elements, dtype=torch.float32))
    ws = workspace_for(1, device)
    for value in (False, True, None):
        with pytest.raises(TypeError, match="deterministic"):
            pkg.cake_rmsnorm_silu_gate_det_train_reset_workspace(
                ws, deterministic=value
            )


def test_two_streams_with_their_own_workspaces():
    """Two shapes on two side streams at once, five rounds, each stream with
    its own caller workspace and with its stream's workspace: every output is
    the bits of the same calls on the current stream, and each caller
    workspace is zero after."""
    cases = [make(4097, 48, seed=30), make(1031, 16, seed=31)]
    refs = [outputs(x, z, w, dy) for x, z, w, dy in cases]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    workspaces = [workspace_for(1, torch.device("cuda")) for _ in streams]
    results: list[list] = [[], []]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for _ in range(5):
        for i, stream in enumerate(streams):
            x, z, w, dy = cases[i]
            with torch.cuda.stream(stream):
                results[i].append(outputs(x, z, w, dy, workspace=workspaces[i]))
                results[i].append(outputs(x, z, w, dy))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for i in range(2):
        for k, got in enumerate(results[i]):
            assert_outputs_equal(got, refs[i], f"stream {i} call {k}")
        assert_zero(workspaces[i], f"the workspace of stream {i}")


def test_side_stream_and_no_host_sync():
    x, z, w, dy = make(2049, 48, seed=9)
    want_y = cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS)
    want = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)
    ws = cake_rmsnorm_silu_gate_det_train_backward_workspace(
        x.shape[0] * x.shape[1], x.device
    )
    assert ws.numel() * 4 == cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes(
        x.shape[0] * x.shape[1], x.device
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.cuda.stream(stream):
            y = cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS)
            got = cake_rmsnorm_silu_gate_det_train_backward(
                dy, x, z, w, EPS, workspace=ws
            )
            fresh = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(y, want_y)
    for out in (got, fresh):
        for a, b in zip(out, want, strict=True):
            assert torch.equal(a, b)
    assert_zero(ws)


def test_cuda_graph_capture():
    """The forward and the backward with a caller workspace, and an FP32
    accumulation without one, captured once and replayed three times on new
    inputs copied into the captured tensors: each replay gives the bits of
    eager calls on those inputs (the accumulated weight gradient the bits of
    the same accumulation run eagerly), and the caller workspace is zero after
    every replay."""
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
    eager_main = torch.zeros(D, dtype=torch.float32, device="cuda")
    for k in range(3):
        xn, zn, _, dyn = make(T, H, seed=40 + k)
        xs.copy_(xn)
        zs.copy_(zn)
        dys.copy_(dyn)
        for out in (y, dx, dz, dw):
            out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert_same_bits(y, forward(xn, zn, w, EPS), f"replay {k} y")
        want = backward(dyn, xn, zn, w, EPS)
        for a, b, name in zip((dx, dz, dw), want, ("dx", "dz", "dw"), strict=True):
            assert_same_bits(a, b, f"replay {k} {name}")
        backward(
            dyn, xn, zn, w, EPS, grad_weight=eager_main, accumulate_grad_weight=True
        )
        assert_same_bits(main, eager_main, f"replay {k} accumulated dw")
        assert_zero(ws, f"the workspace after replay {k}")


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
    both weight-gradient dtypes; every replay gives the bits of eager calls
    (the accumulated weight gradient those of the same accumulation run
    eagerly) and leaves the workspace zero; eager calls do not touch it."""
    pytest.importorskip("cuda.bindings.runtime")
    T, H = 129, 48
    x, z, w, dy = make(T, H, seed=70)
    want = backward(dy, x, z, w, EPS)
    main = torch.zeros(D, dtype=torch.float32, device="cuda")
    backward(dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True)
    torch.cuda.synchronize()
    main.zero_()
    eager_main = torch.zeros_like(main)
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
        backward(dy, x, z, w, EPS, grad_weight=eager_main, accumulate_grad_weight=True)
        torch.cuda.synchronize()
        for out in outs:
            for a, b, name in zip(out, want, ("dx", "dz", "dw"), strict=True):
                assert_same_bits(a, b, f"replay {k} {name}")
        assert_same_bits(main, eager_main, f"replay {k} accumulated dw")
        assert_zero(ws, f"the reserve workspace after replay {k}")


def test_capture_runs_long_and_default_modules_on_shared_workspaces(reserve, launches):
    """Backwards of this device's first row count of the long modules
    (``LONG_MIN_ROWS_PER_SM * SMs`` flat packed rows) captured in one graph on
    one stream: a BF16 weight gradient on a caller workspace and an FP32 one
    accumulated without a workspace run their roles' long modules, and a BF16
    one whose ``grad_out`` is strided (read in place, so not packed) runs the
    ``bwd_bf16`` module on the reserve workspace the FP32 call took.  The
    graph holds the three backward kernels only (no fill kernel, memset or
    event wait).  Every replay on new inputs gives the bits of eager calls
    forced to the roles' default modules (the accumulated weight gradient
    those of the same accumulation run eagerly) and leaves the caller
    workspace and the reserve workspace zero."""
    pytest.importorskip("cuda.bindings.runtime")
    arch, sms = _device()
    routes = loader.ROUTES[arch]
    rows = LONG_MIN_ROWS_PER_SM[arch] * sms
    xs, zs, w, dys = make(rows, 1, layout="flat", seed=76)
    # ``grad_out`` read in place with token stride 2 * 128 + 16, so not packed
    wide = torch.zeros(rows, 2 * D + 16, dtype=torch.bfloat16, device=xs.device)
    dy_strided = wide[:, :D]
    dy_strided.copy_(dys)
    main = torch.zeros(D, dtype=torch.float32, device=xs.device)
    ws = workspace_for(rows, xs.device)

    def step():
        return (
            backward(dys, xs, zs, w, EPS, workspace=ws),
            backward(
                dys, xs, zs, w, EPS, grad_weight=main, accumulate_grad_weight=True
            ),
            backward(dy_strided, xs, zs, w, EPS),
        )

    want = [
        routes["bwd_bf16" + loader.LONG],
        routes["bwd_f32" + loader.LONG],
        routes["bwd_bf16"],
    ]
    launches.clear()
    step()  # loads the long modules before the capture
    torch.cuda.synchronize()
    assert launches == want, launches
    assert_zero(ws)
    main.zero_()
    ready = list(reserve["ready"])
    launches.clear()
    graph = _kept_graph()
    with torch.cuda.graph(graph):
        outs = step()
    assert launches == want, launches
    types = _graph_node_types(graph)
    assert types == ["KERNEL"] * 3, types
    taken = [t for t in ready if all(t is not r for r in reserve["ready"])]
    assert len(taken) == 1 and len(reserve["ready"]) == len(ready) - 1
    eager_main = torch.zeros_like(main)
    defaults = [routes["bwd_bf16"], routes["bwd_f32"], routes["bwd_bf16"]]
    for k in range(3):
        xn, zn, _, dyn = make(rows, 1, layout="flat", seed=77 + k)
        xs.copy_(xn)
        zs.copy_(zn)
        dys.copy_(dyn)
        dy_strided.copy_(dyn)
        graph.replay()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pkg, "backward_module", _route_to(""))
            launches.clear()
            refs = (
                backward(dyn, xn, zn, w, EPS),
                backward(
                    dyn,
                    xn,
                    zn,
                    w,
                    EPS,
                    grad_weight=eager_main,
                    accumulate_grad_weight=True,
                ),
                backward(dy_strided, xn, zn, w, EPS),
            )
            assert launches == defaults, launches
        torch.cuda.synchronize()
        for i, (out, ref) in enumerate(zip(outs, refs, strict=True)):
            for a, b, name in zip(out, ref, ("dx", "dz", "dw"), strict=True):
                assert_same_bits(a, b, f"replay {k} call {i} ({want[i]}) {name}")
        assert_zero(ws, f"the caller workspace after replay {k}")
        assert_zero(taken[0], f"the reserve workspace after replay {k}")


def test_graphs_without_workspace_replay_together_on_two_streams(reserve):
    """Two graphs, each capturing a backward without a caller workspace, own
    two reserve workspaces and replay at the same time on two streams, the
    first replays right after the captures with no host synchronization: the
    bits of eager calls, and both workspaces zero after."""
    cases = [make(4097, 48, seed=71), make(1031, 16, seed=72)]
    refs = [backward(dy, x, z, w, EPS) for x, z, w, dy in cases]
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
        for k, got in enumerate(results[i]):
            for a, b, name in zip(got, refs[i], ("dx", "dz", "dw"), strict=True):
                assert_same_bits(a, b, f"graph {i} replay {k} {name}")
        assert_zero(taken[i], f"the workspace of graph {i}")


def test_captured_calls_without_capture_ids_take_one_workspace_each(reserve):
    """Without cuda-python's capture ids, each captured call takes its own
    reserve workspace; replays give the bits of eager calls and leave them
    zero."""
    x, z, w, dy = make(129, 48, seed=74)
    want = backward(dy, x, z, w, EPS)
    torch.cuda.synchronize()
    ready = list(reserve["ready"])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pkg, "_cudart", lambda: None)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [backward(dy, x, z, w, EPS) for _ in range(3)]
    taken = [ws for ws in ready if all(ws is not r for r in reserve["ready"])]
    assert len(taken) == 3
    for k in range(2):
        graph.replay()
        torch.cuda.synchronize()
        for out in outs:
            for a, b, name in zip(out, want, ("dx", "dz", "dw"), strict=True):
                assert_same_bits(a, b, f"replay {k} {name}")
        for ws in taken:
            assert_zero(ws, f"a reserve workspace after replay {k}")


def test_capture_with_no_ready_workspace_fills_in_the_graph(monkeypatch):
    """With no ready reserve workspace a captured backward without a caller
    workspace zero-fills one from the graph's pool inside the graph (a memset
    or fill kernel before the backward kernel) and gives the bits of an eager
    call at every replay."""
    monkeypatch.setattr(pkg, "_RESERVES", {})
    monkeypatch.setattr(pkg, "_CAPTURE_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_RESERVE_SIZE", 0)
    x, z, w, dy = make(129, 48, seed=73)
    want = backward(dy, x, z, w, EPS)
    torch.cuda.synchronize()
    assert pkg._RESERVES[torch.cuda.current_device()]["blocks"] == []
    graph = _kept_graph()
    with torch.cuda.graph(graph):
        got = backward(dy, x, z, w, EPS)
    types = _graph_node_types(graph)
    assert types in (["MEMSET", "KERNEL"], ["KERNEL", "KERNEL"]), types
    for k in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for a, b, name in zip(got, want, ("dx", "dz", "dw"), strict=True):
            assert_same_bits(a, b, f"replay {k} {name}")


def test_capture_before_any_eager_call_without_workspace(monkeypatch):
    """A capture before any call without a caller workspace (the modules
    loaded by calls with one): the forward and the backward without a caller
    workspace (BF16 and accumulated FP32 weight gradient) capture with no
    stream workspace and no reserve made, each backward on a workspace the
    graph zero-fills; replays give the bits of eager calls, also after eager
    calls without a workspace have made the stream's workspace and the
    reserve."""
    monkeypatch.setattr(pkg, "_RESERVES", {})
    monkeypatch.setattr(pkg, "_CAPTURE_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_STREAM_WORKSPACES", {})
    T, H = 300, 48
    x, z, w, dy = make(T, H, seed=75)
    outputs(x, z, w, dy, workspace=workspace_for(T * H, x.device))
    torch.cuda.synchronize()
    assert pkg._RESERVES == {} and pkg._STREAM_WORKSPACES == {}
    main = torch.zeros(D, dtype=torch.float32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = forward(x, z, w, EPS)
        got = backward(dy, x, z, w, EPS)
        backward(dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True)
    assert pkg._RESERVES == {} and pkg._STREAM_WORKSPACES == {}
    assert pkg._CAPTURE_WORKSPACES == {}
    eager_main = torch.zeros_like(main)
    for k in range(4):
        if k == 2:
            outputs(x, z, w, dy)  # makes the stream's workspace and the reserve
            torch.cuda.synchronize()
            assert pkg._STREAM_WORKSPACES and pkg._RESERVES
        graph.replay()
        backward(dy, x, z, w, EPS, grad_weight=eager_main, accumulate_grad_weight=True)
        torch.cuda.synchronize()
        assert_same_bits(y, forward(x, z, w, EPS), f"replay {k} y")
        want = backward(dy, x, z, w, EPS)
        for a, b, name in zip(got, want, ("dx", "dz", "dw"), strict=True):
            assert_same_bits(a, b, f"replay {k} {name}")
        assert_same_bits(main, eager_main, f"replay {k} accumulated dw")


def test_empty_and_errors():
    x = torch.empty(0, 48, D, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(D, dtype=torch.bfloat16, device="cuda")
    assert cake_rmsnorm_silu_gate_det_train_forward(x, x, w).shape == x.shape
    dx, dz, dw = cake_rmsnorm_silu_gate_det_train_backward(x, x, x, w)
    assert dx.shape == x.shape and torch.equal(dw, torch.zeros_like(dw))
    good = torch.randn(4, 2, D, device="cuda").bfloat16()
    with pytest.raises(ValueError):
        cake_rmsnorm_silu_gate_det_train_forward(good.float(), good, w)
    with pytest.raises(ValueError):
        cake_rmsnorm_silu_gate_det_train_forward(good, good[:, :1], w)
    with pytest.raises(ValueError):
        cake_rmsnorm_silu_gate_det_train_forward(
            torch.randn(4, 64, device="cuda").bfloat16(),
            torch.randn(4, 64, device="cuda").bfloat16(),
            w[:64],
        )
    with pytest.raises(ValueError):
        cake_rmsnorm_silu_gate_det_train_backward(
            good, good, good, w, accumulate_grad_weight=True
        )


def test_registry_records_reference_the_build_manifest():
    """Every call role (``ROLES``: the forward and the backward of each
    weight-gradient dtype) ships one module per architecture, every
    architecture routes the long module of each backward role
    (``LONG_ROLES``), an architecture may also route a forward of packed rows
    (``PACKED_ROLES``), and no other role is routed; every module record
    names its ``build.json`` library (schema ``cake-build/1``), whose entries
    carry the package's kernel and binding sources and the precise-arithmetic
    flags."""
    assert loader.ROLES == ("fwd", "bwd_bf16", "bwd_f32")
    assert loader.LONG == "_long"
    assert loader.LONG_ROLES == ("bwd_bf16_long", "bwd_f32_long")
    assert loader.GRID_KINDS == ("persistent_row_blocks", "claimed_row_blocks")
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert (
        manifest["package"]
        == "flashinfer.experimental.cake.cake_rmsnorm_silu_gate_det_train"
    )
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    for arch in loader.ARCHES:
        roles = set(loader.ROUTES[arch])
        assert {*loader.ROLES, *loader.LONG_ROLES} <= roles
        assert roles <= {*loader.ROLES, *loader.LONG_ROLES, *loader.PACKED_ROLES}
        for role, name in loader.ROUTES[arch].items():
            record = loader.MODULES[name]
            assert record["role"] == role and arch in record["arches"], (arch, role)
            assert loader.library_name(name, arch) in libraries, (arch, role, name)
    routed = {name for by_role in loader.ROUTES.values() for name in by_role.values()}
    assert set(loader.MODULES) == routed
    for name, record in loader.MODULES.items():
        assert record["library"] == record["kernel"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["grid_rule"]["kind"] in loader.GRID_KINDS
        assert record["launch"]["cooperative"] is False
        assert record["launch"]["use_pdl"] is False
        for arch in record["arches"]:
            entry = libraries[loader.library_name(name, arch)]
            for flag in ("--prec-div=true", "--prec-sqrt=true", "--ftz=false"):
                assert flag in entry["nvcc_flags"], (name, arch, flag)
            assert "--use_fast_math" not in entry["nvcc_flags"], (name, arch)
            for source in entry["sources"]:
                assert (package / source).is_file(), (name, source)


# The logical arguments the host passes to each kind of module.
FORWARD_ARGUMENTS = {
    "x",
    "z",
    "w",
    "y",
    "rows",
    "heads",
    "x_tok",
    "x_head",
    "z_tok",
    "z_head",
    "eps",
}
BACKWARD_ARGUMENTS = {
    "dy",
    "x",
    "z",
    "w",
    "dx",
    "dz",
    "acc",
    "flags",
    "done",
    "dw",
    "rows",
    "heads",
    "dy_tok",
    "dy_head",
    "x_tok",
    "x_head",
    "z_tok",
    "z_head",
    "eps",
    "accumulate",
    "weight_grad",
}


def test_records_launch_alone_and_take_the_hosts_arguments():
    """Every record launches without a cluster, cooperatively or with
    programmatic dependent launch, and its kernel parameters map exactly onto
    the logical arguments the host passes.  Each forward (``fwd`` and
    ``fwd_packed``): 128 threads, no dynamic shared memory, no workspace, a
    ``persistent_row_blocks`` rule of positive ``ctas_per_sm``,
    ``max_blocks_per_cta``, ``persistent_from_rows_per_sm`` and
    ``rows_per_block``; an architecture's two forwards launch alike and take
    the same arguments.
    Each backward: 256 threads and the ``claimed_row_blocks`` rule of 4 CTAs
    per SM over 16-row blocks, and a workspace of 2 slots of 9 limbs, the same
    for both weight-gradient dtypes; a long module's rule adds its
    ``min_rows_per_sm`` (``LONG_MIN_ROWS_PER_SM``) and ``packed_rows``, and it
    launches and takes its arguments as its role's module does."""
    for name, record in loader.MODULES.items():
        launch = record["launch"]
        assert tuple(launch["cluster"]) == (1, 1, 1), name
        assert launch["cooperative"] is False and launch["use_pdl"] is False, name
        logical = record["logical"]
        plan = record["arg_plan"]
        for kind, key in plan:
            if kind == "grid":
                assert key in ("grid_x", "grid_y", "grid_z"), (name, key)
            else:
                assert key in logical, (name, kind, key)
        assert {key for kind, key in plan if kind != "grid"} == set(logical), name
        rule = record["grid_rule"]
        if record["role"] in ("fwd", *loader.PACKED_ROLES):
            assert rule["kind"] == "persistent_row_blocks", name
            positive = (
                "ctas_per_sm",
                "max_blocks_per_cta",
                "persistent_from_rows_per_sm",
                "rows_per_block",
            )
            assert set(rule) == {"kind", *positive}, name
            for key in positive:
                assert isinstance(rule[key], int) and rule[key] >= 1, (name, key)
            assert tuple(launch["block"]) == (128, 1, 1), name
            assert launch["dynamic_smem_bytes"] == 0, name
            assert "workspace" not in record, name
            assert set(logical.values()) == FORWARD_ARGUMENTS, name
        else:
            want = {
                "ctas_per_sm": 4,
                "kind": "claimed_row_blocks",
                "rows_per_block": 16,
            }
            if record["role"] in loader.LONG_ROLES:
                want["min_rows_per_sm"] = LONG_MIN_ROWS_PER_SM
                want["packed_rows"] = True
            assert rule == want, name
            assert tuple(launch["block"]) == (256, 1, 1), name
            assert record["workspace"] == {"limbs": 9, "slots": 2}, name
            assert set(logical.values()) == BACKWARD_ARGUMENTS, name
    for arch in loader.ARCHES:
        bf16, f32 = (
            loader.MODULES[loader.route_module(arch, role)]
            for role in ("bwd_bf16", "bwd_f32")
        )
        for key in ("grid_rule", "workspace", "logical"):
            assert bf16[key] == f32[key], (arch, key)
        for role in ("bwd_bf16", "bwd_f32"):
            short = loader.MODULES[loader.route_module(arch, role)]
            long = loader.MODULES[loader.route_module(arch, role + loader.LONG)]
            for key in ("workspace", "logical", "arg_plan", "ffi_entry", "arches"):
                assert short[key] == long[key], (arch, role, key)
            for key in ("block", "cluster", "cooperative", "use_pdl"):
                assert short["launch"][key] == long["launch"][key], (arch, role, key)
            rule = dict(long["grid_rule"])
            del rule["min_rows_per_sm"], rule["packed_rows"]
            assert rule == short["grid_rule"], (arch, role)
        if "fwd_packed" in loader.ROUTES[arch]:
            fwd, packed = (
                loader.MODULES[loader.route_module(arch, role)]
                for role in ("fwd", "fwd_packed")
            )
            for key in ("grid_rule", "launch", "logical", "arg_plan", "ffi_entry"):
                assert fwd[key] == packed[key], (arch, key)


# Without a device: the grids of both rules at several SM counts (small
# MIG-like instances and the products' counts).
GRID_SM_COUNTS = (1, 2, 7, 14, 16, 24, 33, 66, 148, 152, 160, 212, 216)
# The issue's row counts and the kernels' 32-bit row limit.
ISSUE_ROWS = (30193 * 48, 32388 * 48, 32737 * 48, 65536 * 48, 2**31 - 1)


def _grid_rows(rule, sm_count):
    """Row counts at the edges of ``rule``'s grids on ``sm_count`` SMs: a
    row, a block, the rows around one and two waves (of
    ``max_blocks_per_cta`` blocks a CTA for the forward), for the forward
    also around ``persistent_from_rows_per_sm`` rows per SM and one, two and
    seven whole waves from there, and the issue's."""
    per_block = rule["rows_per_block"]
    wave = rule["ctas_per_sm"] * sm_count
    per_cta = rule.get("max_blocks_per_cta", 1)
    rows = {1, per_block - 1, per_block, per_block + 1, *ISSUE_ROWS}
    edges = [
        wave * per_block,
        (wave + 1) * per_block,
        wave * per_cta * per_block,
        (wave * per_cta + 1) * per_block,
        2 * wave * per_cta * per_block,
        (2 * wave * per_cta + 1) * per_block,
        (7 * wave * per_cta + 3) * per_block,
    ]
    if "persistent_from_rows_per_sm" in rule:
        start, waves = _persistent_waves_rows(rule, sm_count)
        step = wave * per_cta * per_block
        edges += [start, waves, waves + step, waves + 6 * step + 3 * per_block]
    for edge in edges:
        rows |= {edge - 1, edge, edge + 1}
    return sorted(r for r in rows if r >= 1)


def test_launch_grid_takes_rows_and_the_sm_count_only():
    """A grid depends on the record, the rows and the SM count only; a rule
    of another kind is a ``ValueError``."""
    assert list(inspect.signature(loader.launch_grid).parameters) == [
        "record",
        "rows",
        "sm_count",
    ]
    rule = {"ctas_per_sm": 1, "kind": "row_blocks", "rows_per_block": 16}
    with pytest.raises(ValueError, match="row_blocks"):
        loader.launch_grid({"grid_rule": rule}, rows=1, sm_count=1)


@pytest.mark.parametrize("sm_count", GRID_SM_COUNTS)
def test_forward_grid_is_one_cta_a_block_then_whole_waves(sm_count):
    """The forward's grid: below ``persistent_from_rows_per_sm * SMs`` rows one
    CTA a row block; from there never more CTAs than row blocks, no CTA with
    more than ``max_blocks_per_cta`` blocks, whole waves of ``ctas_per_sm *
    SMs`` CTAs unless one CTA a block, and the fewest such waves (exactly ``k``
    waves at the fewest rows of ``k`` whole waves of ``max_blocks_per_cta``
    blocks a CTA from there, ``k + 1`` from a row more), for every forward
    record.  ``module_for_rows`` names the one ``fwd`` module whatever the
    rows."""
    forwards = [
        (arch, role)
        for arch in loader.ARCHES
        for role in ("fwd", *loader.PACKED_ROLES)
        if role in loader.ROUTES[arch]
    ]
    for arch, role in forwards:
        name = loader.route_module(arch, role)
        record = loader.MODULES[name]
        rule = record["grid_rule"]
        per_block = rule["rows_per_block"]
        most = rule["max_blocks_per_cta"]
        wave = rule["ctas_per_sm"] * sm_count
        start, waves_rows = _persistent_waves_rows(rule, sm_count)
        for rows in _grid_rows(rule, sm_count):
            if role == "fwd":
                assert loader.module_for_rows(arch, "fwd", rows, sm_count) == name
            grid = loader.launch_grid(record, rows=rows, sm_count=sm_count)
            ctas, blocks = grid[0], -(-rows // per_block)
            what = (arch, rows, grid)
            assert grid[1:] == (1, 1) and 1 <= ctas <= blocks, what
            if rows < start:
                assert ctas == blocks, what
                continue
            assert -(-blocks // ctas) <= most, what
            assert ctas == blocks or ctas % wave == 0, what
            assert ctas == blocks or (ctas - wave) * most < blocks, what
        k = waves_rows // (wave * most * per_block)
        for rows, ctas in (
            (start - 1, -(-(start - 1) // per_block)),
            (waves_rows, min(k * wave, -(-waves_rows // per_block))),
            (waves_rows + 1, min((k + 1) * wave, waves_rows // per_block + 1)),
            (per_block, 1),
        ):
            if rows >= 1:
                got = loader.launch_grid(record, rows=rows, sm_count=sm_count)
                assert got == (ctas, 1, 1), (arch, rows, got)


@pytest.mark.parametrize("sm_count", GRID_SM_COUNTS)
def test_backward_grid_is_at_most_one_wave(sm_count):
    """Each backward's grid (both weight-gradient dtypes and their long
    modules) is ``min(row blocks, ctas_per_sm * SMs)`` CTAs: one CTA a block
    up to one wave, one wave past it (where CTAs claim blocks).
    ``module_for_rows`` names the role's module for rows that are not packed,
    at any row count, and for packed rows the role's long module from its
    ``min_rows_per_sm * SMs`` rows."""
    for arch in loader.ARCHES:
        for role in ("bwd_bf16", "bwd_f32"):
            name = loader.route_module(arch, role)
            long = loader.route_module(arch, role + loader.LONG)
            start = loader.min_rows_per_sm(long, arch) * sm_count
            for module in (name, long):
                record = loader.MODULES[module]
                rule = record["grid_rule"]
                wave = rule["ctas_per_sm"] * sm_count
                edges = {*_grid_rows(rule, sm_count), start - 1, start, start + 1}
                for rows in sorted(edges):
                    got = loader.module_for_rows(arch, role, rows, sm_count)
                    assert got == name, (arch, role, rows, got)
                    got = loader.module_for_rows(arch, role, rows, sm_count, True)
                    assert got == (long if rows >= start else name), (arch, rows)
                    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count)
                    blocks = -(-rows // rule["rows_per_block"])
                    want = blocks if blocks <= wave else wave
                    assert grid == (want, 1, 1), (arch, module, rows, grid)


@pytest.mark.parametrize("sm_count", GRID_SM_COUNTS)
def test_backward_module_runs_long_modules_on_packed_rows_only(sm_count):
    """``backward_module`` (host only): a backward role's long module from
    ``min_rows_per_sm * SMs`` rows (``LONG_MIN_ROWS_PER_SM``) when ``dy``,
    ``x`` and ``z`` all have head stride 128 and token stride ``H * 128``
    (also with one head, and at the kernels' row limit); the role's module a
    row fewer, for a single token (token stride 0), a token-stride-0
    broadcast of any one tensor, a gate or a ``dy`` sliced from a
    projection, head-major rows and a head stride other than 128.
    ``module_for_rows`` takes the packed flag; the forward has no long
    module."""
    H = 48
    row = H * D
    packed = (row, D, row, D, row, D)
    one_head = (D, D, D, D, D, D)
    for arch in loader.ARCHES:
        fwd = loader.ROUTES[arch]["fwd"]
        assert loader.module_for_rows(arch, "fwd", 2**31 - 1, sm_count, True) == fwd
        for role in ("bwd_bf16", "bwd_f32"):
            short = loader.ROUTES[arch][role]
            long = loader.ROUTES[arch][role + loader.LONG]
            assert loader.min_rows_per_sm(long, arch) == LONG_MIN_ROWS_PER_SM[arch]
            start = LONG_MIN_ROWS_PER_SM[arch] * sm_count
            for rows, heads, strides, want in (
                (start - 1, H, packed, short),
                (start, H, packed, long),
                (start + 1, H, packed, long),
                (2**31 - 1, H, packed, long),
                (1, 1, one_head, short),
                (start - 1, 1, one_head, short),
                (start, 1, one_head, long),
                (start, start, (0, D, 0, D, 0, D), short),
                (start, H, (0, D, row, D, row, D), short),
                (start, H, (row, D, 0, D, row, D), short),
                (start, H, (row, D, row, D, 0, D), short),
                (start, H, (row, D, row, D, 2 * row + 16, D), short),
                (start, H, (2 * row + 16, D, row, D, row, D), short),
                (start, H, (row, D, D, 65 * D, D, 65 * D), short),
                (start, H, (row, D, row, D, row, 2 * D), short),
            ):
                got = loader.backward_module(
                    arch, role, rows, heads, sm_count, *strides
                )
                assert got == want, (arch, role, rows, heads, strides)
            assert loader.module_for_rows(arch, role, start, sm_count) == short
            assert loader.module_for_rows(arch, role, start, sm_count, True) == long
            got = loader.module_for_rows(arch, role, start - 1, sm_count, True)
            assert got == short, (arch, role)
