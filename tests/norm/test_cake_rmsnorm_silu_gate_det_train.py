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
layouts, near-zero variance, saturated gates, gradient accumulation,
checkpointing, side streams and repeated-call determinism of the weight
gradient.  The forward's routed modules (the short forward and the persistent
mid, long and extra-long forwards, chosen from the row count and the SM count)
return the plain forward's bits at each threshold and one row on either side,
on every layout, one-head inputs and edge values included, and the backward
after each gives the bits it gives after the plain forward (BF16 and FP32
weight gradients, accumulation, checkpointing, repeats, CUDA Graph replays, two
streams, and ``torch.use_deterministic_algorithms``); without a device, the
registry's routes, thresholds and grids are checked at several SM counts.

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer.  The ``flashinfer.cake_rmsnorm_silu_gate_det_train`` facade and the
``flashinfer`` top-level exposure are covered by
``test_cake_rmsnorm_silu_gate_det_train_api.py``.
"""

import warnings
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

from tests.test_helpers.cake_package import cake_package

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
is_cake_rmsnorm_silu_gate_det_train_supported = (
    pkg.is_cake_rmsnorm_silu_gate_det_train_supported
)

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
    assert (
        (dw.double() - dwe.double()).abs()
        <= dwe.double().abs() * BF16_RTOL + mag * 2.0**-20
    ).all(), "dw vs eager"


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
    check_all(x, z, w, dy, EPS)


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
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(y, want_y)
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)


def test_cuda_graph_capture():
    x, z, w, dy = make(300, 48, seed=10)
    want_y = cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS)
    want = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)
    y = torch.empty_like(x)
    dx, dz = torch.empty_like(x), torch.empty_like(x)
    dw = torch.empty(D, dtype=torch.bfloat16, device="cuda")
    ws = cake_rmsnorm_silu_gate_det_train_backward_workspace(300 * 48, x.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS, out=y)
        cake_rmsnorm_silu_gate_det_train_backward(
            dy, x, z, w, EPS, grad_x=dx, grad_z=dz, grad_weight=dw, workspace=ws
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, want_y)
    assert (
        torch.equal(dx, want[0])
        and torch.equal(dz, want[1])
        and torch.equal(dw, want[2])
    )


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
    """Every call role ships one module per architecture and every routed
    forward role one where the architecture routes it (``ROUTED_FORWARDS``);
    every module record names its ``build.json`` library (schema
    ``cake-build/1``), whose entries carry the package's kernel and binding
    sources and the precise-arithmetic flags."""
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert (
        manifest["package"]
        == "flashinfer.experimental.cake.cake_rmsnorm_silu_gate_det_train"
    )
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    for arch in loader.ARCHES:
        assert set(loader.ROUTES[arch]) == set(loader.ROLES) | ROUTED_FORWARDS[arch]
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


# The forward's routed roles each architecture routes, and the rows per SM the
# short forward runs up to and the mid, long and extra-long forwards from (this
# package's own thresholds).
ROUTED_FORWARDS = {
    "sm_100a": {"fwd_short", "fwd_mid", "fwd_long"},
    "sm_103a": {"fwd_short", "fwd_mid", "fwd_long", "fwd_xlong"},
    "sm_107a": {"fwd_short", "fwd_mid", "fwd_xlong"},
}
SHORT_FORWARD_ROWS_PER_SM = {"sm_100a": 12, "sm_103a": 12, "sm_107a": 59}
FORWARD_STARTS = {
    "fwd_xlong": {"sm_103a": 3000, "sm_107a": 1740},
    "fwd_long": {"sm_100a": 2400, "sm_103a": 2400},
    "fwd_mid": {"sm_100a": 976, "sm_103a": 976, "sm_107a": 800},
}
TIERS = ("fwd_xlong", "fwd_long", "fwd_mid")
ROUTED = ("fwd_short", "fwd_mid", "fwd_long", "fwd_xlong")
REPEATS = 10


def _want_forward(arch, rows, sm_count):
    """The forward role ``rows`` rows run on ``arch`` by the tables above: the
    short forward up to its rows per SM, else the first of the extra-long,
    long and mid forwards the architecture routes from its rows per SM, else
    the forward."""
    if rows <= SHORT_FORWARD_ROWS_PER_SM[arch] * sm_count:
        return "fwd_short"
    for tier in TIERS:
        start = FORWARD_STARTS[tier].get(arch)
        if start is not None and rows >= start * sm_count:
            return tier
    return "fwd"


def _device():
    """``(arch, SMs)`` of the current device."""
    index = torch.cuda.current_device()
    return loader.device_arch(index), loader.device_sm_count(index)


def _bound(tier, arch, sm_count):
    """The rows at ``tier``'s threshold: the most rows the short forward runs,
    the fewest a persistent forward runs."""
    if tier == "fwd_short":
        return SHORT_FORWARD_ROWS_PER_SM[arch] * sm_count
    return FORWARD_STARTS[tier][arch] * sm_count


def _routed_or_skip(tier):
    arch, sms = _device()
    if tier not in ROUTED_FORWARDS[arch]:
        pytest.skip(f"{arch} routes no {tier} module")
    assert loader.ROUTES[arch][tier] == loader.module_for_rows(
        arch, "fwd", _bound(tier, arch, sms), sms
    )
    return arch, sms


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


def _plain(arch, role, rows, sm_count):
    return loader.route_module(arch, role)


def plain_forward(x, z, w, eps, **kwargs):
    """The forward on the plain forward module whatever the rows (the routing
    before the routed forwards shipped): the reference of their bits."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loader, "module_for_rows", _plain)
        return cake_rmsnorm_silu_gate_det_train_forward(x, z, w, eps, **kwargs)


def assert_same_bits(got, want, what):
    assert got.dtype == want.dtype and got.shape == want.shape, what
    view = torch.int16 if got.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(got.view(view), want.view(view)), what


def _check_routed(x, z, w, eps, launches, *, what):
    """The forward of ``x`` runs ``module_for_rows``' module, the one the tables
    give, and returns the plain forward's bits."""
    arch, sms = _device()
    rows = x.numel() // D
    launches.clear()
    y = cake_rmsnorm_silu_gate_det_train_forward(x, z, w, eps)
    served = list(launches)
    name = loader.module_for_rows(arch, "fwd", rows, sms)
    assert served == [name], (what, served)
    assert loader.MODULES[name]["role"] == _want_forward(arch, rows, sms), (what, rows)
    launches.clear()
    want = plain_forward(x, z, w, eps)
    assert launches == [loader.route_module(arch, "fwd")], (what, launches)
    assert_same_bits(y, want, f"{what}: y vs the plain forward")
    return name


@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("tier", ROUTED)
def test_routed_forwards_return_the_plain_forwards_bits_at_their_thresholds(
    tier, delta, launches
):
    """One row below, at and above each routed forward's threshold on this
    device, as one flat ``[rows, 128]`` input."""
    arch, sms = _routed_or_skip(tier)
    rows = _bound(tier, arch, sms) + delta
    x, z, w, _ = make(rows, 1, layout="flat", seed=31)
    _check_routed(x, z, w, EPS, launches, what=f"{tier} {delta:+d}")


@pytest.mark.parametrize("heads", [48, 7, 1])
@pytest.mark.parametrize(
    "layout", ["contiguous", "flat", "projection", "head_major", "misaligned"]
)
@pytest.mark.parametrize("tier", ROUTED)
def test_routed_forwards_return_the_plain_forwards_bits_on_every_layout(
    tier, layout, heads, launches
):
    """At each routed forward's threshold (the most whole tokens the short
    forward runs, the fewest a persistent forward runs) on every layout, one
    head included (``[T, 1, 128]`` and ``[T, 128]``, read in place with head
    stride 128; a gate column slice; head-major; a misaligned copy)."""
    arch, sms = _routed_or_skip(tier)
    bound = _bound(tier, arch, sms)
    tokens = bound // heads if tier == "fwd_short" else -(-bound // heads)
    x, z, w, _ = make(tokens, heads, layout=layout, seed=32)
    name = _check_routed(x, z, w, EPS, launches, what=f"{tier} {layout} H={heads}")
    assert name == loader.ROUTES[arch][tier]


# Gate inputs at and past the edges of the packed gate's range: below -41,
# subnormal and zero, infinite and NaN, beside saturated and ordinary ones.
GATE_EDGE_VALUES = [
    -1e4,
    -100.0,
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
    100.0,
    1e4,
    float("inf"),
    float("-inf"),
    float("nan"),
]


def _gate_edges(z, seed):
    """``z`` with gate edge values in a quarter of the elements of two of
    every three rows (so the packed gate's range check fails for some threads
    of a row and holds for others)."""
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
    "case", ["near_zero", "saturated", "gate_edges", "eps_1e-5", "eps_1e-3"]
)
@pytest.mark.parametrize("tier", ROUTED)
def test_routed_forwards_return_the_plain_forwards_bits_on_edge_values(
    tier, case, launches
):
    """Near-zero variance, saturated gates, gate inputs at and past the packed
    gate's range (NaN and infinities included) and other epsilons, at each
    routed forward's threshold."""
    arch, sms = _routed_or_skip(tier)
    bound = _bound(tier, arch, sms)
    tokens = bound // 48 if tier == "fwd_short" else -(-bound // 48)
    data = case if case in ("near_zero", "saturated") else "normal"
    x, z, w, _ = make(tokens, 48, data=data, seed=33)
    if case == "gate_edges":
        _gate_edges(z, 34)
    eps = float(case[4:]) if case.startswith("eps_") else EPS
    _check_routed(x, z, w, eps, launches, what=f"{tier} {case}")


# The non-deterministic suite's forward cases around and past the short, the
# mid, the long and the extra-long forward's thresholds: (tokens, heads,
# layout); ``heads=None`` takes ``threshold * SMs + tokens`` rows (the short
# threshold for ``flat-short``, the mid one for ``flat-mid``, the extra-long
# one for ``flat-xlong``, the long one for ``flat``) as one flat [rows, 128]
# tensor.
LONG_FORWARD_CASES = [
    pytest.param(1, 48, "contiguous", id="1x48"),
    pytest.param(13, 7, "contiguous", id="13x7"),
    pytest.param(13, 48, "projection", id="13x48-projection"),
    pytest.param(21, 48, "head_major", id="21x48-head_major"),
    pytest.param(0, None, "flat-short", id="short-threshold"),
    pytest.param(1, None, "flat-short", id="short-threshold+1"),
    pytest.param(-1, None, "flat-mid", id="mid-threshold-1"),
    pytest.param(1, None, "flat-mid", id="mid-threshold+1"),
    pytest.param(2048, 48, "contiguous", id="2048x48"),
    pytest.param(4097, 48, "contiguous", id="4097x48"),
    pytest.param(4097, 48, "projection", id="4097x48-projection"),
    pytest.param(4097, 48, "head_major", id="4097x48-head_major"),
    pytest.param(6000, 32, "contiguous", id="6000x32"),
    pytest.param(-1, None, "flat", id="threshold-1"),
    pytest.param(1, None, "flat", id="threshold+1"),
    pytest.param(-1, None, "flat-xlong", id="xlong-threshold-1"),
    pytest.param(0, None, "flat-xlong", id="xlong-threshold"),
    pytest.param(1, None, "flat-xlong", id="xlong-threshold+1"),
    pytest.param(7685, 48, "contiguous", id="7685x48"),
    pytest.param(7685, 48, "projection", id="7685x48-projection"),
    pytest.param(7685, 48, "head_major", id="7685x48-head_major"),
    pytest.param(9500, 48, "contiguous", id="9500x48"),
    pytest.param(9500, 48, "projection", id="9500x48-projection"),
    pytest.param(9500, 48, "head_major", id="9500x48-head_major"),
    pytest.param(16384, 32, "contiguous", id="16384x32"),
    pytest.param(8192, 48, "contiguous", id="8192x48"),
    pytest.param(8192, 48, "projection", id="8192x48-projection"),
    pytest.param(8192, 48, "head_major", id="8192x48-head_major"),
    pytest.param(32737, 48, "contiguous", id="32737x48"),
    pytest.param(32737, 32, "contiguous", id="32737x32"),
]


@pytest.mark.parametrize("tokens,heads,layout", LONG_FORWARD_CASES)
def test_forward_cases_of_the_non_deterministic_suite_return_the_plain_bits(
    tokens, heads, layout, launches
):
    arch, sms = _device()
    if heads is None:
        tier = {
            "flat-short": "fwd_short",
            "flat-mid": "fwd_mid",
            "flat-xlong": "fwd_xlong",
        }.get(layout, "fwd_long" if arch != "sm_107a" else "fwd_xlong")
        if tier not in ROUTED_FORWARDS[arch]:
            pytest.skip(f"{arch} routes no {tier} module")
        x, z, w, _ = make(_bound(tier, arch, sms) + tokens, 1, layout="flat", seed=21)
    else:
        x, z, w, _ = make(tokens, heads, layout=layout, seed=21)
    _check_routed(x, z, w, EPS, launches, what=f"{tokens} {heads} {layout}")


# The non-deterministic suite's gate-edge inputs: those the short forward runs
# and those at its threshold on 148 and 152 SMs (12 rows per SM) and on 212
# SMs (59 rows per SM), and one token past it.
GATE_EDGE_CASES = [
    pytest.param(1, 48, id="1x48"),
    pytest.param(13, 48, id="13x48"),
    pytest.param(30, 48, id="30x48"),
    pytest.param(31, 48, id="31x48"),
    pytest.param(33, 48, id="33x48"),
    pytest.param(37, 48, id="37x48"),
    pytest.param(38, 48, id="38x48"),
    pytest.param(13, 7, id="13x7"),
    pytest.param(217, 7, id="217x7"),
    pytest.param(253, 7, id="253x7"),
    pytest.param(260, 7, id="260x7"),
    pytest.param(55, 32, id="55x32"),
    pytest.param(57, 32, id="57x32"),
    pytest.param(129, 48, id="129x48"),
    pytest.param(260, 48, id="260x48"),
    pytest.param(261, 48, id="261x48"),
    pytest.param(1786, 7, id="1786x7"),
    pytest.param(1787, 7, id="1787x7"),
    pytest.param(390, 32, id="390x32"),
    pytest.param(391, 32, id="391x32"),
]


@pytest.mark.parametrize("tokens,heads", GATE_EDGE_CASES)
def test_gate_edge_cases_of_the_non_deterministic_suite_return_the_plain_bits(
    tokens, heads, launches
):
    x, z, w, _ = make(tokens, heads, seed=23)
    _gate_edges(z, 24)
    _check_routed(x, z, w, EPS, launches, what=f"{tokens}x{heads} gate edges")


def _tier_inputs(tier):
    """``(x, z, w, dy)`` of ``[T, 48, 128]`` at ``tier``'s threshold."""
    arch, sms = _routed_or_skip(tier)
    bound = _bound(tier, arch, sms)
    tokens = bound // 48 if tier == "fwd_short" else -(-bound // 48)
    x, z, w, dy = make(tokens, 48, seed=35)
    assert (
        loader.module_for_rows(arch, "fwd", tokens * 48, sms)
        == (loader.ROUTES[arch][tier])
    )
    return x, z, w, dy


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


def _functional(x, z, w, dy):
    """The forward, then the backward with a BF16, an FP32 and an accumulated
    FP32 weight gradient: ``(y, dx, dz, dw_bf16, dw_f32, dw_acc)``."""
    y = cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS)
    dx, dz, dw = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)
    f32 = torch.empty(D, dtype=torch.float32, device=x.device)
    cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS, grad_weight=f32)
    acc = torch.full((D,), 0.25, dtype=torch.float32, device=x.device)
    for _ in range(2):
        cake_rmsnorm_silu_gate_det_train_backward(
            dy, x, z, w, EPS, grad_weight=acc, accumulate_grad_weight=True
        )
    return y, dx, dz, dw, f32, acc


def _under_plain_routing(fn, *args, **kwargs):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loader, "module_for_rows", _plain)
        return fn(*args, **kwargs)


OUTPUTS = ("y", "dx", "dz", "dw")
FUNCTIONAL_OUTPUTS = ("y", "dx", "dz", "dw_bf16", "dw_f32", "dw_accumulated")


@pytest.mark.parametrize("tier", ROUTED)
def test_backward_after_each_routed_forward_is_the_plain_runs_bits(tier, launches):
    """Autograd (BF16 weight gradient) with and without non-reentrant
    checkpointing, and the functional backward with a BF16, an FP32 and an
    accumulated FP32 weight gradient, each after the routed forward, give the
    bits of the same steps after the plain forward, on every one of
    ``REPEATS`` repeats."""
    arch, _ = _device()
    x, z, w, dy = _tier_inputs(tier)
    want = _under_plain_routing(_step, x, z, w, dy)
    want_functional = _under_plain_routing(_functional, x, z, w, dy)
    launches.clear()
    got = _step(x, z, w, dy)
    assert launches[0] == loader.ROUTES[arch][tier], launches
    for checkpoint in (False, True):
        for _ in range(REPEATS):
            got = _step(x, z, w, dy, checkpoint=checkpoint)
            for a, b, what in zip(got, want, OUTPUTS, strict=True):
                assert_same_bits(a, b, f"{tier} checkpoint={checkpoint} {what}")
    for _ in range(REPEATS):
        got = _functional(x, z, w, dy)
        for a, b, what in zip(got, want_functional, FUNCTIONAL_OUTPUTS, strict=True):
            assert_same_bits(a, b, f"{tier} functional {what}")


@pytest.mark.parametrize("tier", ROUTED)
def test_routed_steps_replay_in_a_cuda_graph_and_on_two_streams(tier):
    """The forward and the backward at each routed forward's threshold,
    captured in a CUDA Graph and replayed, and run on two side streams at
    once with their own workspaces, give the plain run's bits."""
    x, z, w, dy = _tier_inputs(tier)
    rows = x.numel() // D
    want_y = plain_forward(x, z, w, EPS)
    want = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)
    # the routed module loads (and gives the plain bits) before the capture
    routed = cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS)
    assert_same_bits(routed, want_y, f"{tier} eager y")
    y = torch.empty_like(x)
    dx, dz = torch.empty_like(x), torch.empty_like(x)
    dw = torch.empty(D, dtype=torch.bfloat16, device=x.device)
    ws = cake_rmsnorm_silu_gate_det_train_backward_workspace(rows, x.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS, out=y)
        cake_rmsnorm_silu_gate_det_train_backward(
            dy, x, z, w, EPS, grad_x=dx, grad_z=dz, grad_weight=dw, workspace=ws
        )
    for replay in range(3):
        for out in (y, dx, dz, dw):
            out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert_same_bits(y, want_y, f"{tier} graph replay {replay} y")
        for a, b, what in zip((dx, dz, dw), want, OUTPUTS[1:], strict=True):
            assert_same_bits(a, b, f"{tier} graph replay {replay} {what}")
    streams = [torch.cuda.Stream() for _ in range(2)]
    workspaces = [
        cake_rmsnorm_silu_gate_det_train_backward_workspace(rows, x.device)
        for _ in streams
    ]
    outs = []
    for stream, workspace in zip(streams, workspaces, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            outs.append(
                (
                    cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS),
                    *cake_rmsnorm_silu_gate_det_train_backward(
                        dy, x, z, w, EPS, workspace=workspace
                    ),
                )
            )
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for i, out in enumerate(outs):
        for a, b, what in zip(out, (want_y, *want), OUTPUTS, strict=True):
            assert_same_bits(a, b, f"{tier} stream {i} {what}")


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
@pytest.mark.parametrize("tier", ROUTED)
def test_routed_steps_neither_raise_nor_warn_under_deterministic_algorithms(
    tier, warn_only
):
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and
    ``warn_only``) with warnings as errors, an autograd step at each routed
    forward's threshold raises nothing, warns nothing and gives the bits of
    the step with the setting off."""
    x, z, w, dy = _tier_inputs(tier)
    want = _step(x, z, w, dy)
    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = _step(x, z, w, dy)
            got_checkpointed = _step(x, z, w, dy, checkpoint=True)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)
    for out in (got, got_checkpointed):
        for a, b, what in zip(out, want, OUTPUTS, strict=True):
            assert_same_bits(a, b, f"{tier} warn_only={warn_only} {what}")


# Without a device: the registry's routes, thresholds and grids (the
# non-deterministic suite's tests of the same modules, which this package
# routes with its own thresholds).


@pytest.mark.parametrize("sm_count", [148, 152, 212, 216])
def test_routed_forwards_run_from_their_rows_per_sm(sm_count):
    """``module_for_rows``: the short forward up to its ``max_rows_per_sm``
    rows per SM, else the extra-long, the long and the mid forward from their
    ``min_rows_per_sm`` rows per SM, in that order, where the architecture
    routes them, else the forward; the backward and the reduces route none."""
    assert loader.TIERS == ("_xlong", "_long", "_mid")
    assert loader.XLONG_ROLES == ("fwd_xlong",)
    assert loader.LONG_ROLES == ("fwd_long",)
    assert loader.MID_ROLES == ("fwd_mid",)
    assert loader.SHORT_ROLES == ("fwd_short",)
    assert loader.ROLES == ("fwd", "bwd", "dw_bf16", "dw_f32")
    for arch in loader.ARCHES:
        routes = loader.ROUTES[arch]
        assert set(routes) - set(loader.ROLES) == ROUTED_FORWARDS[arch], arch
        for role, name in routes.items():
            assert loader.MODULES[name]["role"] == role
        short = routes["fwd_short"]
        assert loader.max_rows_per_sm(short, arch) == SHORT_FORWARD_ROWS_PER_SM[arch]
        for tier in TIERS:
            if tier in routes:
                assert (
                    loader.min_rows_per_sm(routes[tier], arch)
                    == (FORWARD_STARTS[tier][arch])
                ), (arch, tier)
            else:
                assert arch not in FORWARD_STARTS[tier], (arch, tier)
        edges = {1, 48, 255, 256, 257, 2**31 - 1}
        for start in (12, 59, 800, 976, 1740, 2400, 3000):
            edges |= {start * sm_count - 1, start * sm_count, start * sm_count + 1}
        for rows in sorted(edges):
            want = routes[_want_forward(arch, rows, sm_count)]
            assert loader.module_for_rows(arch, "fwd", rows, sm_count) == want, (
                arch,
                rows,
            )
            for role in ("bwd", "dw_bf16", "dw_f32"):
                got = loader.module_for_rows(arch, role, rows, sm_count)
                assert got == routes[role], (arch, role, rows)
    # sm_100a: the short forward up to 12 rows per SM, the forward, the mid
    # forward from 976, the long forward from 2400; sm_103a the same with the
    # extra-long forward from 3000; sm_107a the short forward up to 59, the
    # mid forward from 800, the extra-long forward from 1740 (no long forward)
    for arch, table in (
        (
            "sm_100a",
            ((1, "fwd_short"), (12, "fwd_short"), (13, "fwd"), (975, "fwd")),
        ),
        ("sm_100a", ((976, "fwd_mid"), (2399, "fwd_mid"), (2400, "fwd_long"))),
        ("sm_103a", ((12, "fwd_short"), (976, "fwd_mid"), (2400, "fwd_long"))),
        ("sm_103a", ((2999, "fwd_long"), (3000, "fwd_xlong"))),
        ("sm_107a", ((59, "fwd_short"), (60, "fwd"), (799, "fwd"))),
        ("sm_107a", ((800, "fwd_mid"), (1739, "fwd_mid"), (1740, "fwd_xlong"))),
    ):
        for per_sm, role in table:
            got = loader.module_for_rows(arch, "fwd", per_sm * sm_count, sm_count)
            assert got == loader.ROUTES[arch][role], (arch, per_sm)
    assert "fwd_long" not in loader.ROUTES["sm_107a"]
    assert "fwd_xlong" not in loader.ROUTES["sm_100a"]


# (role, rows, SMs, CTAs) of the persistent forwards: the long and the
# extra-long forward 32-row blocks, the fewest whole waves of 3 CTAs per SM
# whose CTAs walk at most 6 blocks; the mid forward 16-row blocks, whole waves
# of 6 CTAs per SM walking at most 2; never more CTAs than blocks.
PERSISTENT_FORWARD_GRIDS = [
    ("fwd_long", 1, 148, 1),
    ("fwd_long", 32 * 444, 148, 444),
    ("fwd_long", 32 * 444 * 6, 148, 444),
    ("fwd_long", 32 * 444 * 6 + 1, 148, 888),
    ("fwd_long", 2400 * 148, 148, 2220),
    ("fwd_long", 32737 * 48, 148, 8436),
    ("fwd_long", 2400 * 152, 152, 2280),
    ("fwd_long", 32737 * 48, 152, 8208),
    ("fwd_long", 2**31 - 1, 148, 25192 * 444),
    ("fwd_xlong", 3000 * 152, 152, 2736),
    ("fwd_xlong", 1740 * 212, 212, 2544),
    ("fwd_xlong", 8192 * 48, 212, 2544),
    ("fwd_mid", 1, 148, 1),
    ("fwd_mid", 16 * 888, 148, 888),
    ("fwd_mid", 16 * 888 * 2, 148, 888),
    ("fwd_mid", 16 * 888 * 2 + 1, 148, 1776),
    ("fwd_mid", 976 * 148, 148, 5328),
    ("fwd_mid", 2400 * 148 - 1, 148, 11544),
    ("fwd_mid", 4097 * 48, 148, 6216),
    ("fwd_mid", 976 * 152, 152, 5472),
    ("fwd_mid", 4097 * 48, 152, 6384),
    ("fwd_mid", 800 * 212, 212, 6360),
    ("fwd_mid", 1740 * 212 - 1, 212, 12720),
    ("fwd_mid", 2**31 - 1, 148, 67109712),
]


@pytest.mark.parametrize(("role", "rows", "sm_count", "ctas"), PERSISTENT_FORWARD_GRIDS)
def test_persistent_forward_grids_are_whole_waves_of_a_few_blocks_a_cta(
    role, rows, sm_count, ctas
):
    arch = "sm_107a" if role == "fwd_xlong" else "sm_100a"
    record = loader.MODULES[loader.ROUTES[arch][role]]
    long = role != "fwd_mid"
    per_block, per_sm, most = (32, 3, 6) if long else (16, 6, 2)
    assert record["grid_rule"] == {
        "ctas_per_sm": per_sm,
        "kind": "persistent_row_blocks",
        "max_blocks_per_cta": most,
        "min_rows_per_sm": {a: FORWARD_STARTS[role][a] for a in record["arches"]},
        "rows_per_block": per_block,
    }
    assert record["launch"]["block"] == ((256 if long else 128), 1, 1)
    assert record["launch"]["use_pdl"] is False
    assert record["launch"]["cooperative"] is False
    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count)
    assert grid == (ctas, 1, 1)
    blocks, wave = -(-rows // per_block), per_sm * sm_count
    assert -(-blocks // ctas) <= most
    assert ctas == blocks or ctas % wave == 0
    assert ctas == blocks or (ctas - wave) * most < blocks
    # the same grid on every architecture that routes the role; the extra-long
    # forward's is the long forward's
    for other in loader.ARCHES:
        name = loader.ROUTES[other].get(role)
        if name is not None:
            got = loader.launch_grid(loader.MODULES[name], rows=rows, sm_count=sm_count)
            assert got == grid, other
    if role == "fwd_xlong":
        long_record = loader.MODULES[loader.ROUTES["sm_103a"]["fwd_long"]]
        assert loader.launch_grid(long_record, rows=rows, sm_count=sm_count) == grid
    # no other grid depends on the SM count
    fwd = loader.MODULES[loader.route_module(arch, "fwd")]
    assert loader.launch_grid(fwd, rows=rows, sm_count=sm_count) == (
        -(-rows // 16),
        1,
        1,
    )
    reduce = loader.MODULES[loader.route_module(arch, "dw_bf16")]
    assert loader.launch_grid(reduce, rows=rows, sm_count=sm_count) == (128, 1, 1)


# The most CTAs of the short forward's grid at its threshold from the fewest
# SMs that keep it (64 on sm_100a and sm_103a, 205 on sm_107a; a small MIG
# instance has fewer): the SMs less 16 on sm_100a and sm_103a, one wave of its
# 4 CTAs per SM on the SMs less 16 on sm_107a.
SHORT_FORWARD_MAX_CTAS = {
    "sm_100a": lambda sm_count: sm_count - 16,
    "sm_103a": lambda sm_count: sm_count - 16,
    "sm_107a": lambda sm_count: 4 * (sm_count - 16),
}
SHORT_FORWARD_MIN_SMS = {"sm_100a": 64, "sm_103a": 64, "sm_107a": 205}


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_short_forward_grid_bound_holds_from_its_fewest_sms(arch):
    name = loader.ROUTES[arch]["fwd_short"]
    record = loader.MODULES[name]
    assert record["grid_rule"] == {
        "kind": "row_blocks",
        "max_rows_per_sm": {"sm_100a": 12, "sm_103a": 12, "sm_107a": 59},
        "rows_per_cta": 16,
    }
    assert record["launch"]["block"] == (256, 1, 1)
    end = loader.max_rows_per_sm(name, arch)
    for sm_count in range(1, 1025):
        grid = loader.launch_grid(record, rows=end * sm_count, sm_count=sm_count)
        assert grid == (-(-(end * sm_count) // 16), 1, 1)
        fits = grid[0] <= SHORT_FORWARD_MAX_CTAS[arch](sm_count)
        assert fits == (sm_count >= SHORT_FORWARD_MIN_SMS[arch]), (arch, sm_count)


def test_routed_forwards_share_the_forwards_arguments_and_launch_alone():
    """Every forward module reads and writes the forward's arguments (the
    same ``arg_plan`` and ``logical`` map), launches neither a cluster, nor
    cooperatively, nor with programmatic dependent launch, and takes no
    dynamic shared memory."""
    forward = loader.MODULES[loader.route_module("sm_100a", "fwd")]
    names = {name for routes in loader.ROUTES.values() for name in routes.values()}
    forwards = [loader.MODULES[n] for n in names if loader.MODULES[n]["role"] != "bwd"]
    forwards = [r for r in forwards if r["role"].startswith("fwd")]
    assert sorted(r["role"] for r in forwards) == sorted(("fwd", *ROUTED))
    for record in forwards:
        assert record["arg_plan"] == forward["arg_plan"], record["role"]
        assert record["logical"] == forward["logical"], record["role"]
        assert tuple(record["launch"]["cluster"]) == (1, 1, 1)
        assert record["launch"]["dynamic_smem_bytes"] == 0
        assert record["launch"]["cooperative"] is False
        assert record["launch"]["use_pdl"] is False


def test_short_forward_grid_at_its_threshold_keeps_the_bound_on_this_device():
    """On the running device, the short forward's grid at the architecture's
    threshold keeps ``SHORT_FORWARD_MAX_CTAS``; skipped where the device has
    fewer SMs than the bound needs."""
    arch, sm_count = _device()
    name = loader.ROUTES[arch]["fwd_short"]
    if sm_count < SHORT_FORWARD_MIN_SMS[arch]:
        pytest.skip(
            f"{sm_count} SMs: the short forward's grid bound on {arch} needs "
            f"{SHORT_FORWARD_MIN_SMS[arch]} SMs or more"
        )
    rows = loader.max_rows_per_sm(name, arch) * sm_count
    assert loader.module_for_rows(arch, "fwd", rows, sm_count) == name
    assert loader.module_for_rows(arch, "fwd", rows + 1, sm_count) != name
    grid = loader.launch_grid(loader.MODULES[name], rows=rows, sm_count=sm_count)
    assert grid[0] <= SHORT_FORWARD_MAX_CTAS[arch](sm_count), (arch, sm_count, grid)
