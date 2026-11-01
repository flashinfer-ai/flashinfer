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

Tests of the Cake SiLU-gated RMSNorm training kernels with the opt-in
non-deterministic weight gradient (``deterministic=False`` of
:mod:`flashinfer.cake_rmsnorm_silu_gate_train`): ``RMSNorm(x) * w * SiLU(z)``.

``y``, ``dx`` and ``dz`` must equal the deterministic package's bits and lie
within one BF16 ulp of an independent FP64 reference.  The weight gradient,
whose FP64 cross-CTA sum depends on scheduling, is held to the deterministic
suite's budget, ``|dw - dw64| <= |dw64| * 2^-7 + sum|terms| * 2^-20``, worst
case over repeated launches, and to the same budget against the deterministic
package's ``dw``.  The matrix covers dynamic token and head counts (``H = 48``
and small or irregular row counts), ``[T, H, 128]`` strided and flattened
layouts, near-zero variance, saturated gates, FP32 and BF16 weight gradients,
accumulation, ``compute_grad_weight=False``, empty work, a workspace reused
across shapes (zero after every call), two streams with their own workspaces,
CUDA Graph capture and replay and the autograd wrapper (non-reentrant
checkpointing included).

This suite exercises the backend packages alone, through
``tests.test_helpers.cake_package``; the facade, the ``flashinfer`` top level,
the default (deterministic) dispatch and the interplay with
``torch.use_deterministic_algorithms`` are
``test_cake_rmsnorm_silu_gate_train_api.py``'s.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_silu_gate_train")
cake_build = cake_package("cake_rmsnorm_silu_gate_train", "cake_build")
loader = cake_package("cake_rmsnorm_silu_gate_train", "cake_jit")
deterministic_package = cake_package("cake_rmsnorm_silu_gate_det_train")
deterministic_loader = cake_package("cake_rmsnorm_silu_gate_det_train", "cake_jit")
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
gated_rmsnorm = functools.partial(pkg.cake_rmsnorm_silu_gate, deterministic=False)
deterministic_forward = deterministic_package.cake_rmsnorm_silu_gate_det_train_forward
deterministic_backward = deterministic_package.cake_rmsnorm_silu_gate_det_train_backward

D = 128
EPS = 1e-6
# One BF16 unit in the last place relative to the value (8 significant bits).
BF16_RTOL = 2.0**-7
# The FP32 accumulation term of the weight-gradient budget, relative to the sum
# of the terms' magnitudes (the deterministic suite's).
DW_MAG_TOL = 2.0**-20
REPEATS = 10


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
    """Forward and ``repeats`` backward launches against the FP64 reference and
    the deterministic package; returns the number of distinct ``dw`` bit
    patterns over the launches."""
    y = forward(x, z, w, eps)
    assert y.dtype == torch.bfloat16 and y.shape == x.shape
    assert torch.equal(y, deterministic_forward(x, z, w, eps)), (
        "y vs the deterministic package"
    )
    y64, dx64, dz64, dw64 = fp64_reference(dy, x, z, w, eps)
    mag = magnitudes(dy, x, z, eps)
    assert_bf16_close(y, y64, what="y vs FP64")

    def out():
        return (
            None
            if dw_dtype == torch.bfloat16
            else torch.empty(D, dtype=dw_dtype, device=x.device)
        )

    dx_d, dz_d, dw_d = deterministic_backward(dy, x, z, w, eps, grad_weight=out())
    patterns = set()
    for _ in range(repeats):
        dx, dz, dw = backward(dy, x, z, w, eps, grad_weight=out())
        assert dx.dtype == dz.dtype == torch.bfloat16 and dw.dtype == dw_dtype
        assert dx.shape == dz.shape == x.shape and dw.shape == (D,)
        assert torch.equal(dx, dx_d), "dx vs the deterministic package"
        assert torch.equal(dz, dz_d), "dz vs the deterministic package"
        assert_dw_within(dw, dw64, mag, what="dw vs FP64")
        assert_dw_within(dw, dw_d, mag, what="dw vs the deterministic package")
        patterns.add(bytes(dw.view(torch.uint8).cpu().numpy()))
    assert_bf16_close(dz, dz64, what="dz vs FP64")
    assert_bf16_close(dx, dx64, what="dx vs FP64")
    return len(patterns)


@pytest.mark.parametrize(
    "T,H",
    [
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
    ],
)
def test_forward_backward_shapes(T, H):
    x, z, w, dy = make(T, H)
    check_all(x, z, w, dy, EPS)


@pytest.mark.parametrize("layout", ["flat", "projection", "head_major", "misaligned"])
def test_layouts(layout):
    x, z, w, dy = make(65, 48, layout=layout, seed=1)
    check_all(x, z, w, dy, EPS)


@pytest.mark.parametrize("data", ["near_zero", "saturated"])
def test_edge_values(data):
    x, z, w, dy = make(37, 48, data=data, seed=2)
    check_all(x, z, w, dy, EPS)


@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_epsilon(eps):
    x, z, w, dy = make(19, 48, seed=3)
    check_all(x, z, w, dy, eps)


@pytest.mark.parametrize("T", [30193, 32737, 32388])
def test_issue_shapes(T):
    x, z, w, dy = make(T, 48, seed=4)
    patterns = check_all(x, z, w, dy, EPS)
    print(f"DW_PATTERNS T={T} H=48 repeats={REPEATS} distinct={patterns}")


@pytest.mark.parametrize("T,H", [(7, 48), (1031, 48), (32737, 48)])
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
        main_reference = main.clone()
        for args in ((dy, x, z), (dy2, x2, z2)):
            backward(*args, w, EPS, grad_weight=main, accumulate_grad_weight=True)
            deterministic_backward(
                *args, w, EPS, grad_weight=main_reference, accumulate_grad_weight=True
            )
        assert ((main.double() - want).abs() <= bound).all(), dtype
        assert ((main.double() - main_reference.double()).abs() <= 2 * bound).all(), (
            dtype
        )
        if dtype == torch.float32:
            assert torch.allclose(main.double(), want, rtol=1e-5, atol=1e-3)
    # autograd accumulation into .grad over two microbatches
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    gated_rmsnorm(xa, za, wa, EPS).backward(dy)
    gated_rmsnorm(xa, za, wa, EPS).backward(dy)
    dx_d, dz_d, _ = deterministic_backward(dy, x, z, w, EPS)
    assert torch.equal(xa.grad, dx_d + dx_d)
    assert torch.equal(za.grad, dz_d + dz_d)
    # Two BF16 weight gradients within budget, summed in BF16 by autograd.
    mag1 = magnitudes(dy, x, z, EPS)
    err = (wa.grad.double() - 2 * dw64).abs()
    assert (err <= (2 * dw64).abs() * 2.0**-6 + mag1 * 2.0**-18).all()


def test_compute_grad_weight_false_touches_no_weight_state():
    x, z, w, dy = make(257, 48, seed=12)
    dx_d, dz_d, _ = deterministic_backward(dy, x, z, w, EPS, compute_grad_weight=False)
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
    # without a workspace: nothing is allocated for the weight gradient
    dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
    assert dw is None
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
    # autograd with a weight that needs no gradient
    xa, za = (t.clone().requires_grad_(True) for t in (x, z))
    gated_rmsnorm(xa, za, w, EPS).backward(dy)
    assert torch.equal(xa.grad, dx_d) and torch.equal(za.grad, dz_d)
    assert w.grad is None


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
    for role in ("bwd_bf16", "bwd_f32"):
        assert loader.workspace_elements(loader.route_module(arch, role)) * 8 <= nbytes
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
        dx_d, dz_d, _ = deterministic_backward(dy, x, z, w, EPS)
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), (T, H)
        assert dw.dtype == dtype
        want = fp64_reference(dy, x, z, w, EPS)[3] + start
        assert_dw_within(
            dw, want, magnitudes(dy, x, z, EPS), what=f"dw T={T} H={H} {dtype}"
        )


def test_workspace_errors():
    x, z, w, dy = make(65, 48, seed=13)
    deterministic_ws = (
        deterministic_package.cake_rmsnorm_silu_gate_det_train_backward_workspace(
            65 * 48, x.device
        )
    )
    with pytest.raises(ValueError, match="FP64"):
        backward(dy, x, z, w, EPS, workspace=deterministic_ws)
    elements = workspace_for(1, x.device).numel()
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


def test_two_streams_with_their_own_workspaces():
    cases = [make(4097, 48, seed=30), make(1031, 16, seed=31)]
    refs = [deterministic_backward(dy, x, z, w, EPS) for x, z, w, dy in cases]
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
    want_y = deterministic_forward(x, z, w, EPS)
    dx_d, dz_d, _ = deterministic_backward(dy, x, z, w, EPS)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    ws = workspace_for(x.shape[0] * x.shape[1], x.device)
    # Load both modules before the sync-checked region.
    forward(x, z, w, EPS)
    backward(dy, x, z, w, EPS, workspace=ws)
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
        assert torch.equal(y, deterministic_forward(xn, zn, w, EPS)), k
        dx_d, dz_d, _ = deterministic_backward(dyn, xn, zn, w, EPS)
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), k
        dw64, mag = fp64_reference(dyn, xn, zn, w, EPS)[3], magnitudes(dyn, xn, zn, EPS)
        assert_dw_within(dw, dw64, mag, what=f"replay {k} dw")
        assert not bool(ws.any()), k
        total += dw64
        total_mag += mag
        # three FP32 roundings of the accumulator at most, plus the sums' budget
        err = (main.double() - total).abs()
        assert (err <= total.abs() * 2.0**-21 + total_mag * DW_MAG_TOL).all(), k


def test_autograd_matches_functional_and_checkpoint(launches):
    x, z, w, dy = make(513, 48, seed=8)
    y_ref = deterministic_forward(x, z, w, EPS)
    dx_d, dz_d, dw_d = deterministic_backward(dy, x, z, w, EPS)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)

    def check(y, xg, zg, wg, what):
        assert torch.equal(y, y_ref), what
        assert torch.equal(xg, dx_d) and torch.equal(zg, dz_d), what
        assert wg.dtype == torch.bfloat16, what
        assert_dw_within(wg, dw64, mag, what=f"{what} dw vs FP64")
        assert_dw_within(wg, dw_d, mag, what=f"{what} dw vs the deterministic package")

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
        assert name not in deterministic_loader.MODULES, name
    assert loader.MODULES[routes["bwd_bf16"]]["role"] == "bwd_bf16"
    assert loader.MODULES[routes["bwd_f32"]]["role"] == "bwd_f32"
    # The default runs the deterministic package: nothing through this registry.
    launches.clear()
    pkg.cake_rmsnorm_silu_gate_train_forward(x, z, w, EPS)
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
    """Every role ships one module per architecture; every module record names
    its ``build.json`` library (schema ``cake-build/1``), whose entries carry the
    package's kernel and binding sources and the precise-arithmetic flags; the
    backward records size their workspace from their accumulator slots."""
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == "flashinfer.cake_rmsnorm_silu_gate_train_backend"
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    for arch in loader.ARCHES:
        assert set(loader.ROUTES[arch]) == set(loader.ROLES), arch
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
        if record["role"] == "fwd":
            assert record["grid_rule"]["kind"] == "row_blocks"
        else:
            assert record["grid_rule"]["kind"] == "row_chunks"
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
def test_backward_chunk_rule(rows, sm_count):
    # The fewest whole waves of ctas_per_sm CTAs per SM with at most
    # max_blocks_per_cta blocks per CTA, the fewest blocks per CTA that fit
    # those waves (no partly filled last wave beyond one CTA's blocks); every
    # row covered.
    for role in ("bwd_bf16", "bwd_f32"):
        name = loader.route_module("sm_100a", role)
        rule = loader.MODULES[name]["grid_rule"]
        chunks, per_cta = loader.backward_chunks(name, rows, sm_count)
        blocks = -(-rows // rule["rows_per_cta"])
        wave = rule["ctas_per_sm"] * sm_count
        waves = -(-blocks // (wave * rule["max_blocks_per_cta"]))
        assert 1 <= per_cta <= rule["max_blocks_per_cta"]
        assert (chunks - 1) * per_cta < blocks <= chunks * per_cta
        assert (waves - 1) * wave < chunks <= waves * wave
        assert per_cta == 1 or blocks > (per_cta - 1) * waves * wave


def test_calls_without_workspace_share_one_buffer_per_stream():
    x, z, w, dy = make(65, 7)
    for _ in range(2):
        backward(dy, x, z, w, 1e-6)
    torch.cuda.synchronize()
    stream = torch.cuda.current_stream().cuda_stream
    held = [v for k, v in pkg._STREAM_WORKSPACES.items() if k[1] == stream]
    assert held, "a call without a workspace keeps one per stream"
    for sums, done in held:
        assert sums.abs().max().item() == 0 and int(done.item()) == 0
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        backward(dy, x, z, w, 1e-6)
    side.synchronize()
    assert any(k[1] == side.cuda_stream for k in pkg._STREAM_WORKSPACES)
