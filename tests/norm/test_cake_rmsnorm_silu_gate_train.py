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
(:mod:`flashinfer.cake_rmsnorm_silu_gate_train`): ``RMSNorm(x) * w * SiLU(z)``
against the eager FP32 operation (one final BF16 cast) and an independent
FP64 reference, over dynamic token and head counts, irregular tails, strided
layouts, near-zero variance, saturated gates, gradient accumulation,
checkpointing, side streams and repeated-call determinism of the weight
gradient.

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer.  The ``flashinfer.cake_rmsnorm_silu_gate_train`` facade and the
``flashinfer`` top-level exposure are covered by
``test_cake_rmsnorm_silu_gate_train_api.py``.
"""

from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_silu_gate_train")
cake_build = cake_package("cake_rmsnorm_silu_gate_train", "cake_build")
loader = cake_package("cake_rmsnorm_silu_gate_train", "cake_jit")
cake_rmsnorm_silu_gate = pkg.cake_rmsnorm_silu_gate
cake_rmsnorm_silu_gate_train_backward = pkg.cake_rmsnorm_silu_gate_train_backward
cake_rmsnorm_silu_gate_train_backward_workspace = (
    pkg.cake_rmsnorm_silu_gate_train_backward_workspace
)
cake_rmsnorm_silu_gate_train_backward_workspace_bytes = (
    pkg.cake_rmsnorm_silu_gate_train_backward_workspace_bytes
)
cake_rmsnorm_silu_gate_train_forward = pkg.cake_rmsnorm_silu_gate_train_forward
is_cake_rmsnorm_silu_gate_train_supported = (
    pkg.is_cake_rmsnorm_silu_gate_train_supported
)

D = 128
EPS = 1e-6
# One BF16 unit in the last place relative to the value (8 significant bits).
BF16_RTOL = 2.0**-7

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cake_rmsnorm_silu_gate_train_supported(torch.device("cuda")),
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
    y = cake_rmsnorm_silu_gate_train_forward(x, z, w, eps)
    dx, dz, dw = cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, eps)
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
    first = cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, EPS)[2]
    for _ in range(5):
        again = cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, EPS)[2]
        assert torch.equal(first.view(torch.int16), again.view(torch.int16))
    fp32 = torch.empty(D, dtype=torch.float32, device="cuda")
    cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, EPS, grad_weight=fp32)
    for _ in range(3):
        again = torch.empty_like(fp32)
        cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, EPS, grad_weight=again)
        assert torch.equal(fp32.view(torch.int32), again.view(torch.int32))


def test_weight_gradient_accumulation():
    x, z, w, dy = make(1031, 48, seed=6)
    x2, z2, _, dy2 = make(1031, 48, seed=7)
    _, _, _, dw64 = fp64_reference(dy, x, z, w, EPS)
    _, _, _, dw64b = fp64_reference(dy2, x2, z2, w, EPS)
    main = torch.full((D,), 0.25, dtype=torch.float32, device="cuda")
    cake_rmsnorm_silu_gate_train_backward(
        dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True
    )
    cake_rmsnorm_silu_gate_train_backward(
        dy2, x2, z2, w, EPS, grad_weight=main, accumulate_grad_weight=True
    )
    want = 0.25 + dw64 + dw64b
    assert torch.allclose(main.double(), want, rtol=1e-5, atol=1e-3)
    # autograd accumulation into .grad over two microbatches
    xa = x.clone().requires_grad_(True)
    wa = w.clone().requires_grad_(True)
    za = z.clone().requires_grad_(True)
    cake_rmsnorm_silu_gate(xa, za, wa, EPS).backward(dy)
    cake_rmsnorm_silu_gate(xa, za, wa, EPS).backward(dy)
    single = cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, EPS)
    assert torch.equal(xa.grad, single[0] + single[0])
    assert torch.equal(za.grad, single[1] + single[1])
    assert torch.equal(wa.grad, single[2] + single[2])


def test_autograd_matches_functional_and_checkpoint():
    x, z, w, dy = make(513, 48, seed=8)
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    y = cake_rmsnorm_silu_gate(xa, za, wa, EPS)
    y.backward(dy)
    y_ref = cake_rmsnorm_silu_gate_train_forward(x, z, w, EPS)
    dx, dz, dw = cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, EPS)
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
        cake_rmsnorm_silu_gate(xb, zb, wb, EPS)
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
        cake_rmsnorm_silu_gate, xc, zc, wc, EPS, use_reentrant=False
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
    want_y = cake_rmsnorm_silu_gate_train_forward(x, z, w, EPS)
    want = cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, EPS)
    ws = cake_rmsnorm_silu_gate_train_backward_workspace(
        x.shape[0] * x.shape[1], x.device
    )
    assert ws.numel() * 4 == cake_rmsnorm_silu_gate_train_backward_workspace_bytes(
        x.shape[0] * x.shape[1], x.device
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.cuda.stream(stream):
            y = cake_rmsnorm_silu_gate_train_forward(x, z, w, EPS)
            got = cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, EPS, workspace=ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(y, want_y)
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)


def test_cuda_graph_capture():
    x, z, w, dy = make(300, 48, seed=10)
    want_y = cake_rmsnorm_silu_gate_train_forward(x, z, w, EPS)
    want = cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, EPS)
    y = torch.empty_like(x)
    dx, dz = torch.empty_like(x), torch.empty_like(x)
    dw = torch.empty(D, dtype=torch.bfloat16, device="cuda")
    ws = cake_rmsnorm_silu_gate_train_backward_workspace(300 * 48, x.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cake_rmsnorm_silu_gate_train_forward(x, z, w, EPS, out=y)
        cake_rmsnorm_silu_gate_train_backward(
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
    assert cake_rmsnorm_silu_gate_train_forward(x, x, w).shape == x.shape
    dx, dz, dw = cake_rmsnorm_silu_gate_train_backward(x, x, x, w)
    assert dx.shape == x.shape and torch.equal(dw, torch.zeros_like(dw))
    good = torch.randn(4, 2, D, device="cuda").bfloat16()
    with pytest.raises(ValueError):
        cake_rmsnorm_silu_gate_train_forward(good.float(), good, w)
    with pytest.raises(ValueError):
        cake_rmsnorm_silu_gate_train_forward(good, good[:, :1], w)
    with pytest.raises(ValueError):
        cake_rmsnorm_silu_gate_train_forward(
            torch.randn(4, 64, device="cuda").bfloat16(),
            torch.randn(4, 64, device="cuda").bfloat16(),
            w[:64],
        )
    with pytest.raises(ValueError):
        cake_rmsnorm_silu_gate_train_backward(
            good, good, good, w, accumulate_grad_weight=True
        )


def test_registry_records_reference_the_build_manifest():
    """Every role ships one module per architecture; every module record names
    its ``build.json`` library (schema ``cake-build/1``), whose entries carry the
    package's kernel and binding sources and the precise-arithmetic flags."""
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
