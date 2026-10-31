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

Tests of the flashinfer public surface of the Cake SiLU-gated RMSNorm training
package with the deterministic default and the opt-in non-deterministic weight
gradient: the ``flashinfer.cake_rmsnorm_silu_gate_train`` facade, its re-export
from the ``flashinfer`` top level, the ``deterministic`` flag of every entry
point and its interplay with ``torch.use_deterministic_algorithms``.  The
default call returns the bits of ``flashinfer.cake_rmsnorm_silu_gate_det_train``;
``deterministic=False`` runs this package's modules.  The numerical contract of
the non-deterministic kernels is ``test_cake_rmsnorm_silu_gate_train.py``'s.
The tests of the flag's resolution stub the kernels and run without a device.
"""

from __future__ import annotations

import inspect
import warnings

import pytest
import torch
import torch.utils.checkpoint

import flashinfer
import flashinfer.cake_rmsnorm_silu_gate_det_train as deterministic_facade
import flashinfer.cake_rmsnorm_silu_gate_train_backend as backend
from flashinfer.cake_rmsnorm_silu_gate_train import (
    CakeRMSNormSiLUGateFunction,
    cake_rmsnorm_silu_gate,
    cake_rmsnorm_silu_gate_train_backward,
    cake_rmsnorm_silu_gate_train_backward_workspace,
    cake_rmsnorm_silu_gate_train_backward_workspace_bytes,
    cake_rmsnorm_silu_gate_train_forward,
    is_cake_rmsnorm_silu_gate_train_supported,
)
from flashinfer.cake_rmsnorm_silu_gate_train_backend import cake_jit as loader

D = 128
EPS = 1e-6
ENTRY_POINTS = (
    cake_rmsnorm_silu_gate_train_forward,
    cake_rmsnorm_silu_gate_train_backward,
    cake_rmsnorm_silu_gate_train_backward_workspace,
    cake_rmsnorm_silu_gate_train_backward_workspace_bytes,
    cake_rmsnorm_silu_gate,
    is_cake_rmsnorm_silu_gate_train_supported,
    backend.cake_rmsnorm_silu_gate_train_forward,
    backend.cake_rmsnorm_silu_gate_train_backward,
    backend.cake_rmsnorm_silu_gate,
)
DETERMINISTIC_ENTRY_POINTS = (
    "cake_rmsnorm_silu_gate_det_train_forward",
    "cake_rmsnorm_silu_gate_det_train_backward",
    "cake_rmsnorm_silu_gate_det_train_backward_workspace",
    "cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes",
)


def _supported() -> bool:
    return torch.cuda.is_available() and is_cake_rmsnorm_silu_gate_train_supported(
        torch.device("cuda"), deterministic=False
    )


requires_device = pytest.mark.skipif(
    not _supported(),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)


@pytest.fixture
def determinism_setting():
    """Restore the caller's ``torch.use_deterministic_algorithms`` setting
    after the test; the test starts with it off."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _cpu_tensors():
    x = torch.zeros(4, 2, D, dtype=torch.bfloat16)
    w = torch.ones(D, dtype=torch.bfloat16)
    return x, w


def _inputs(T: int, H: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x, z, dy = (
        torch.randn(T, H, D, device="cuda", generator=g).bfloat16() for _ in range(3)
    )
    w = (1.0 + 0.5 * torch.randn(D, device="cuda", generator=g)).bfloat16()
    return x, z, w, dy


# --------------------------------------------------------------------------- the surface (no device)


def test_top_level_exports_are_the_facade() -> None:
    assert flashinfer.cake_rmsnorm_silu_gate is cake_rmsnorm_silu_gate
    assert (
        flashinfer.cake_rmsnorm_silu_gate_train_forward
        is cake_rmsnorm_silu_gate_train_forward
    )
    assert (
        flashinfer.cake_rmsnorm_silu_gate_train_backward
        is cake_rmsnorm_silu_gate_train_backward
    )
    assert (
        flashinfer.cake_rmsnorm_silu_gate_train_backward_workspace_bytes
        is cake_rmsnorm_silu_gate_train_backward_workspace_bytes
    )
    assert flashinfer.CakeRMSNormSiLUGateFunction is CakeRMSNormSiLUGateFunction
    assert CakeRMSNormSiLUGateFunction is backend.CakeRMSNormSiLUGateFunction
    # the det names stay the det package's objects
    assert (
        flashinfer.cake_rmsnorm_silu_gate_det
        is deterministic_facade.cake_rmsnorm_silu_gate_det
    )
    assert (
        flashinfer.cake_rmsnorm_silu_gate_det_train_forward
        is deterministic_facade.cake_rmsnorm_silu_gate_det_train_forward
    )
    assert (
        flashinfer.CakeRMSNormSiLUGateDetFunction
        is deterministic_facade.CakeRMSNormSiLUGateDetFunction
    )


@pytest.mark.parametrize(
    "entry", ENTRY_POINTS, ids=lambda f: f"{f.__module__}.{f.__name__}"
)
def test_every_entry_point_defaults_to_deterministic(entry) -> None:
    parameter = inspect.signature(entry).parameters["deterministic"]
    assert parameter.default is True
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_autograd_function_takes_the_flag_as_an_input() -> None:
    parameters = inspect.signature(CakeRMSNormSiLUGateFunction.forward).parameters
    assert list(parameters) == ["ctx", "x", "z", "weight", "eps", "deterministic"]
    assert parameters["deterministic"].default is True


@pytest.mark.parametrize(
    "call", ["forward", "backward", "workspace", "workspace_bytes", "autograd"]
)
def test_nondeterministic_request_is_refused_under_deterministic_algorithms(
    call: str, monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """``deterministic=False`` while ``torch.use_deterministic_algorithms(True)``
    raises before any kernel or device query; with ``warn_only=True`` it warns
    and runs the det package (stubbed here: no device needed)."""
    x, w = _cpu_tensors()
    calls: list[str] = []
    for name in DETERMINISTIC_ENTRY_POINTS:
        monkeypatch.setattr(
            backend._deterministic_backend,
            name,
            lambda *a, _name=name, **k: calls.append(_name) or x.clone(),
        )
    run = {
        "forward": lambda: cake_rmsnorm_silu_gate_train_forward(
            x, x, w, EPS, deterministic=False
        ),
        "backward": lambda: cake_rmsnorm_silu_gate_train_backward(
            x, x, x, w, EPS, deterministic=False
        ),
        "workspace": lambda: cake_rmsnorm_silu_gate_train_backward_workspace(
            8, "cpu", deterministic=False
        ),
        "workspace_bytes": lambda: (
            cake_rmsnorm_silu_gate_train_backward_workspace_bytes(
                8, "cpu", deterministic=False
            )
        ),
        "autograd": lambda: cake_rmsnorm_silu_gate(x, x, w, EPS, deterministic=False),
    }[call]
    torch.use_deterministic_algorithms(True)
    with pytest.raises(RuntimeError, match="deterministic=True") as refused:
        run()
    assert "cake_rmsnorm_silu_gate_det" in str(refused.value)
    assert calls == []
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning, match="running the deterministic kernels"):
        run()
    assert len(calls) == 1 and calls[0].startswith("cake_rmsnorm_silu_gate_det_train_")


def test_deterministic_request_never_warns(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    x, w = _cpu_tensors()
    monkeypatch.setattr(
        backend._deterministic_backend,
        "cake_rmsnorm_silu_gate_det_train_forward",
        lambda *a, **k: x.clone(),
    )
    torch.use_deterministic_algorithms(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cake_rmsnorm_silu_gate_train_forward(x, x, w, EPS)
        cake_rmsnorm_silu_gate_train_forward(x, x, w, EPS, deterministic=True)


def test_autograd_backward_resolves_the_flag_again(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """The backward of a ``deterministic=False`` forward resolves the flag when
    it runs: refused once ``torch.use_deterministic_algorithms(True)`` is in
    effect, the deterministic backward with ``warn_only=True``."""
    calls: list[str] = []
    monkeypatch.setattr(
        backend, "_forward", lambda x, z, w, eps, *, out: calls.append("fwd") or x * 1
    )
    monkeypatch.setattr(
        backend, "_backward", lambda *a, **k: calls.append("bwd") or (None,) * 3
    )
    monkeypatch.setattr(
        backend._deterministic_backend,
        "cake_rmsnorm_silu_gate_det_train_backward",
        lambda grad_out, x, z, w, eps, **k: (
            calls.append("deterministic_bwd")
            or (torch.zeros_like(x), torch.zeros_like(z), torch.zeros_like(w))
        ),
    )

    def leaves():
        x, w = _cpu_tensors()
        return (
            x.clone().requires_grad_(True),
            x.clone().requires_grad_(True),
            w.clone().requires_grad_(True),
        )

    x, z, w = leaves()
    y = backend.cake_rmsnorm_silu_gate(x, z, w, EPS, deterministic=False)
    torch.use_deterministic_algorithms(True)
    with pytest.raises(RuntimeError, match="deterministic=True"):
        y.backward(torch.ones_like(y))
    assert calls == ["fwd"]
    torch.use_deterministic_algorithms(False)
    x, z, w = leaves()
    y = backend.cake_rmsnorm_silu_gate(x, z, w, EPS, deterministic=False)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning, match="running the deterministic kernels"):
        y.backward(torch.ones_like(y))
    assert calls == ["fwd", "fwd", "deterministic_bwd"]


def test_warn_only_fallback_drops_a_nondeterministic_workspace(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """The deterministic backward a ``deterministic=False`` call falls back to
    under ``warn_only=True`` allocates its own buffer in place of the FP64
    workspace of the non-deterministic backward; every other workspace, and
    every ``deterministic=True`` call, passes through unchanged."""
    x, w = _cpu_tensors()
    seen: list = []
    monkeypatch.setattr(
        backend._deterministic_backend,
        "cake_rmsnorm_silu_gate_det_train_backward",
        lambda *a, workspace=None, **k: seen.append(workspace) or (x, x, None),
    )
    fp64 = torch.zeros(4 * D + 1, dtype=torch.float64)
    fp32 = torch.zeros(D, dtype=torch.float32)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning):
        cake_rmsnorm_silu_gate_train_backward(
            x, x, x, w, EPS, workspace=fp64, deterministic=False
        )
    with pytest.warns(UserWarning):
        cake_rmsnorm_silu_gate_train_backward(
            x, x, x, w, EPS, workspace=fp32, deterministic=False
        )
    cake_rmsnorm_silu_gate_train_backward(x, x, x, w, EPS, workspace=fp64)
    assert seen[0] is None and seen[1] is fp32 and seen[2] is fp64


def test_support_query_ignores_torch_determinism(determinism_setting) -> None:
    torch.use_deterministic_algorithms(True)
    cpu = torch.device("cpu")
    assert is_cake_rmsnorm_silu_gate_train_supported(cpu) is False
    assert is_cake_rmsnorm_silu_gate_train_supported(cpu, deterministic=False) is False


# --------------------------------------------------------------------------- on the device


@requires_device
def test_default_returns_the_deterministic_package_bits(determinism_setting) -> None:
    """The default call of every entry point is the det package's call: the
    same bits, the same workspace and the same errors."""
    T, H = 129, 48
    x, z, w, dy = _inputs(T, H, 11)
    device = x.device
    y = cake_rmsnorm_silu_gate_train_forward(x, z, w, EPS)
    assert torch.equal(
        y, deterministic_facade.cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS)
    )
    for dtype in (None, torch.float32):

        def out(dtype=dtype):
            return None if dtype is None else torch.empty(D, dtype=dtype, device=device)

        got = cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, EPS, grad_weight=out())
        want = deterministic_facade.cake_rmsnorm_silu_gate_det_train_backward(
            dy, x, z, w, EPS, grad_weight=out()
        )
        for a, b in zip(got, want, strict=True):
            assert torch.equal(a, b), dtype
    main = torch.full((D,), 0.25, dtype=torch.float32, device=device)
    main_reference = main.clone()
    for _ in range(2):
        cake_rmsnorm_silu_gate_train_backward(
            dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True
        )
        deterministic_facade.cake_rmsnorm_silu_gate_det_train_backward(
            dy, x, z, w, EPS, grad_weight=main_reference, accumulate_grad_weight=True
        )
    assert torch.equal(main, main_reference)
    nbytes = cake_rmsnorm_silu_gate_train_backward_workspace_bytes(T * H, device)
    assert (
        nbytes
        == deterministic_facade.cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes(
            T * H, device
        )
    )
    ws = cake_rmsnorm_silu_gate_train_backward_workspace(T * H, device)
    assert ws.dtype == torch.float32 and ws.numel() * 4 == nbytes
    got = cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, EPS, workspace=ws)
    want = deterministic_facade.cake_rmsnorm_silu_gate_det_train_backward(
        dy, x, z, w, EPS
    )
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)
    # the det package's errors: it refuses the non-deterministic FP64 workspace
    fp64 = cake_rmsnorm_silu_gate_train_backward_workspace(
        T * H, device, deterministic=False
    )
    with pytest.raises(ValueError, match="FP32"):
        cake_rmsnorm_silu_gate_train_backward(dy, x, z, w, EPS, workspace=fp64)
    # autograd and non-reentrant checkpointing
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    xb, zb, wb = (t.clone().requires_grad_(True) for t in (x, z, w))
    ya = cake_rmsnorm_silu_gate(xa, za, wa, EPS)
    yb = deterministic_facade.cake_rmsnorm_silu_gate_det(xb, zb, wb, EPS)
    assert torch.equal(ya, yb)
    ya.backward(dy)
    yb.backward(dy)
    for a, b in ((xa, xb), (za, zb), (wa, wb)):
        assert torch.equal(a.grad, b.grad)
    xc, zc, wc = (t.clone().requires_grad_(True) for t in (x, z, w))
    yc = torch.utils.checkpoint.checkpoint(
        cake_rmsnorm_silu_gate, xc, zc, wc, EPS, use_reentrant=False
    )
    yc.backward(dy)
    assert torch.equal(yc, yb)
    for a, b in ((xc, xb), (zc, zb), (wc, wb)):
        assert torch.equal(a.grad, b.grad)


@requires_device
def test_nondeterministic_flag_selects_this_package(
    determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deterministic=False`` launches this package's forward and its one
    backward kernel (and no det module): ``y``, ``dx`` and ``dz`` are the det
    package's bits, ``dw`` within one BF16 rounding plus the FP32 accumulation
    budget of the det ``dw``."""
    launched: list[str] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        launched.append(name)
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    x, z, w, dy = _inputs(257, 48, 12)
    y = cake_rmsnorm_silu_gate_train_forward(x, z, w, EPS, deterministic=False)
    dx, dz, dw = cake_rmsnorm_silu_gate_train_backward(
        dy, x, z, w, EPS, deterministic=False
    )
    routes = loader.ROUTES[loader.device_arch(torch.cuda.current_device())]
    assert launched == [routes["fwd"], routes["bwd_bf16"]], launched
    assert torch.equal(
        y, deterministic_facade.cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS)
    )
    dx_d, dz_d, dw_d = deterministic_facade.cake_rmsnorm_silu_gate_det_train_backward(
        dy, x, z, w, EPS
    )
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
    terms = (dy.double() * torch.nn.functional.silu(z.double())) * (
        x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + EPS)
    )
    mag = terms.abs().reshape(-1, D).sum(0)
    err = (dw.double() - dw_d.double()).abs()
    assert (err <= dw_d.double().abs() * 2.0**-7 + mag * 2.0**-20).all()
