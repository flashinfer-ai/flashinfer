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
kernels (SM100 / SM103 / SM107): the ``flashinfer.cake_rmsnorm_silu_gate_det_train``
facade and its re-export from the ``flashinfer`` top-level package.  The
facade's entry points are pass-throughs to
``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_det_train``; this
module checks that identity (same objects at the top level, same bits as the
backend) and the mode rules of the deterministic package: no entry point takes a
``deterministic`` keyword (``TypeError``, checked without a device),
``torch.use_deterministic_algorithms(True)`` (strict or ``warn_only``) neither
raises nor warns in any call, the backward's workspace is a zero-filled FP32
tensor of one size for every row count that
``cake_rmsnorm_silu_gate_det_train_reset_workspace`` zeroes again, and the
backward and the reset refuse the non-deterministic package's workspaces (FP64
or ``uint8``), the backward also in a call with no rows.  The reset is not a
``flashinfer`` top-level name.  The numerical contract is
``test_cake_rmsnorm_silu_gate_det_train.py``'s.
"""

from __future__ import annotations

import inspect
import warnings

import pytest
import torch
import torch.utils.checkpoint

import flashinfer
import flashinfer.cake_rmsnorm_silu_gate_det_train as facade
import flashinfer.experimental.cake.cake_rmsnorm_silu_gate_det_train as backend
from flashinfer.cake_rmsnorm_silu_gate_det_train import (
    CakeRMSNormSiLUGateDetFunction,
    cake_rmsnorm_silu_gate_det,
    cake_rmsnorm_silu_gate_det_train_backward,
    cake_rmsnorm_silu_gate_det_train_backward_workspace,
    cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes,
    cake_rmsnorm_silu_gate_det_train_forward,
    cake_rmsnorm_silu_gate_det_train_reset_workspace,
    is_cake_rmsnorm_silu_gate_det_train_supported,
)

D = 128
EPS = 1e-6
requires_device = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cake_rmsnorm_silu_gate_det_train_supported(torch.device("cuda")),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)
# The entry points of both deterministic namespaces (facade and backend package).
ENTRY_NAMES = (
    "cake_rmsnorm_silu_gate_det",
    "cake_rmsnorm_silu_gate_det_train_backward",
    "cake_rmsnorm_silu_gate_det_train_backward_workspace",
    "cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate_det_train_forward",
    "cake_rmsnorm_silu_gate_det_train_reset_workspace",
    "is_cake_rmsnorm_silu_gate_det_train_supported",
)
NAMESPACES = {"facade": facade, "backend": backend}
# Elements and bytes of the backward's workspace, whatever the rows.
WORKSPACE_ELEMENTS = 4738
WORKSPACE_BYTES = 18952


def _refusal(dtype: torch.dtype) -> str:
    """The refusal of a workspace of the non-deterministic package."""
    name = {torch.float64: "FP64", torch.uint8: "uint8"}[dtype]
    return (
        f"workspace is {name}, a workspace of the non-deterministic mode "
        "(flashinfer.cake_rmsnorm_silu_gate_nondet_train."
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace); the "
        "deterministic backward needs its own zero-filled FP32 workspace "
        "(flashinfer.cake_rmsnorm_silu_gate_det_train."
        "cake_rmsnorm_silu_gate_det_train_backward_workspace)"
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


# --------------------------------------------------------------------------- the surface (no device)


@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
@pytest.mark.parametrize("name", ENTRY_NAMES)
def test_det_entry_points_take_no_mode_keyword(namespace: str, name: str) -> None:
    entry = getattr(NAMESPACES[namespace], name)
    assert "deterministic" not in inspect.signature(entry).parameters, name


@pytest.mark.parametrize("mode", [True, False, None], ids=["true", "false", "none"])
def test_a_mode_keyword_is_a_type_error(mode) -> None:
    """Every entry point of both deterministic namespaces raises ``TypeError``
    for a ``deterministic`` keyword before it reads any argument, and the
    autograd function takes no fifth input (no device needed)."""
    x = torch.zeros(4, 2, D, dtype=torch.bfloat16)
    w = torch.ones(D, dtype=torch.bfloat16)
    for module in NAMESPACES.values():
        runs = {
            "cake_rmsnorm_silu_gate_det": lambda m=module: m.cake_rmsnorm_silu_gate_det(
                x, x, w, EPS, deterministic=mode
            ),
            "cake_rmsnorm_silu_gate_det_train_forward": lambda m=module: (
                m.cake_rmsnorm_silu_gate_det_train_forward(
                    x, x, w, EPS, deterministic=mode
                )
            ),
            "cake_rmsnorm_silu_gate_det_train_backward": lambda m=module: (
                m.cake_rmsnorm_silu_gate_det_train_backward(
                    x, x, x, w, EPS, deterministic=mode
                )
            ),
            "cake_rmsnorm_silu_gate_det_train_backward_workspace": lambda m=module: (
                m.cake_rmsnorm_silu_gate_det_train_backward_workspace(
                    8, "cpu", deterministic=mode
                )
            ),
            "cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes": lambda m=module: (
                m.cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes(
                    8, "cpu", deterministic=mode
                )
            ),
            "cake_rmsnorm_silu_gate_det_train_reset_workspace": lambda m=module: (
                m.cake_rmsnorm_silu_gate_det_train_reset_workspace(
                    torch.zeros(8), deterministic=mode
                )
            ),
            "is_cake_rmsnorm_silu_gate_det_train_supported": lambda m=module: (
                m.is_cake_rmsnorm_silu_gate_det_train_supported(
                    "cpu", deterministic=mode
                )
            ),
        }
        assert set(runs) == set(ENTRY_NAMES)
        for run in runs.values():
            with pytest.raises(TypeError, match="deterministic"):
                run()
        with pytest.raises(TypeError):
            module.CakeRMSNormSiLUGateDetFunction.apply(x, x, w, EPS, mode)


def test_workspace_entries_refuse_negative_rows() -> None:
    """Both namespaces' workspace and workspace size refuse a negative row
    count with a ``ValueError`` before they look at the device (no device
    needed)."""
    for module in NAMESPACES.values():
        for entry in (
            module.cake_rmsnorm_silu_gate_det_train_backward_workspace,
            module.cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes,
        ):
            with pytest.raises(ValueError, match="non-negative"):
                entry(-1, "cpu")


def test_reset_workspace_passes_through(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """The facade's reset resets through the backend whatever
    ``torch.use_deterministic_algorithms`` says, with no warning (no device
    needed)."""
    resets: list = []
    monkeypatch.setattr(
        backend,
        "cake_rmsnorm_silu_gate_det_train_reset_workspace",
        lambda workspace, **k: resets.append((workspace, k)),
    )
    ws = torch.zeros(8, dtype=torch.float32)
    for enabled, warn_only in ((False, False), (True, False), (True, True)):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cake_rmsnorm_silu_gate_det_train_reset_workspace(ws)
    assert resets == [(ws, {})] * 3


# --------------------------------------------------------------------------- on the device


@requires_device
def test_top_level_exports_are_the_facade():
    assert flashinfer.cake_rmsnorm_silu_gate_det is cake_rmsnorm_silu_gate_det
    assert (
        flashinfer.cake_rmsnorm_silu_gate_det_train_forward
        is cake_rmsnorm_silu_gate_det_train_forward
    )
    assert (
        flashinfer.cake_rmsnorm_silu_gate_det_train_backward
        is cake_rmsnorm_silu_gate_det_train_backward
    )
    assert (
        flashinfer.cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes
        is cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes
    )
    # the reset is the facade's and the backend's, not a top-level name
    assert not hasattr(flashinfer, "cake_rmsnorm_silu_gate_det_train_reset_workspace")
    # the autograd class is the backend package's, reached through the facade
    # and the top level
    assert facade.CakeRMSNormSiLUGateDetFunction is CakeRMSNormSiLUGateDetFunction
    assert CakeRMSNormSiLUGateDetFunction is backend.CakeRMSNormSiLUGateDetFunction
    assert flashinfer.CakeRMSNormSiLUGateDetFunction is CakeRMSNormSiLUGateDetFunction


@requires_device
def test_facade_matches_backend_bits():
    g = torch.Generator(device="cuda").manual_seed(11)
    x, z, dy = (
        torch.randn(129, 48, 128, device="cuda", generator=g).bfloat16()
        for _ in range(3)
    )
    w = (1.0 + 0.5 * torch.randn(128, device="cuda", generator=g)).bfloat16()
    y = cake_rmsnorm_silu_gate_det_train_forward(x, z, w, 1e-6)
    assert torch.equal(
        y, backend.cake_rmsnorm_silu_gate_det_train_forward(x, z, w, 1e-6)
    )
    got = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, 1e-6)
    want = backend.cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, 1e-6)
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    ya = cake_rmsnorm_silu_gate_det(xa, za, wa, 1e-6)
    ya.backward(dy)
    assert torch.equal(ya, y)
    assert (
        torch.equal(xa.grad, got[0])
        and torch.equal(za.grad, got[1])
        and torch.equal(wa.grad, got[2])
    )


@requires_device
@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_deterministic_algorithms_neither_raise_nor_warn(
    warn_only: bool, determinism_setting
) -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and
    ``warn_only``) with every warning an error, the forward, the backward
    (both weight-gradient dtypes, accumulation, ``compute_grad_weight=False``,
    no rows), the workspace entry points and the reset, the support query,
    autograd and a non-reentrant checkpoint run as with the mode off: the same
    bits."""
    g = torch.Generator(device="cuda").manual_seed(13)
    x, z, dy = (
        torch.randn(257, 48, D, device="cuda", generator=g).bfloat16() for _ in range(3)
    )
    w = (1.0 + 0.5 * torch.randn(D, device="cuda", generator=g)).bfloat16()
    empty = torch.empty(0, 48, D, dtype=torch.bfloat16, device="cuda")

    def run():
        out = [cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS)]
        out += cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)
        main = torch.full((D,), 0.25, dtype=torch.float32, device="cuda")
        ws = cake_rmsnorm_silu_gate_det_train_backward_workspace(
            x.numel() // D, x.device
        )
        cake_rmsnorm_silu_gate_det_train_reset_workspace(ws)
        out += cake_rmsnorm_silu_gate_det_train_backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=main,
            accumulate_grad_weight=True,
            workspace=ws,
        )
        out += cake_rmsnorm_silu_gate_det_train_backward(
            dy, x, z, w, EPS, compute_grad_weight=False
        )[:2]
        out += cake_rmsnorm_silu_gate_det_train_backward(empty, empty, empty, w, EPS)
        assert (
            cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes(
                x.numel() // D, x.device
            )
            == ws.numel() * 4
        )
        assert is_cake_rmsnorm_silu_gate_det_train_supported(x.device)
        leaves = [t.clone().requires_grad_(True) for t in (x, z, w)]
        y = cake_rmsnorm_silu_gate_det(*leaves, EPS)
        y.backward(dy)
        out += [y, *(t.grad for t in leaves)]
        leaves = [t.clone().requires_grad_(True) for t in (x, z, w)]
        y = torch.utils.checkpoint.checkpoint(
            cake_rmsnorm_silu_gate_det, *leaves, EPS, use_reentrant=False
        )
        y.backward(dy)
        out += [y, *(t.grad for t in leaves)]
        return out

    want = run()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = run()
    torch.cuda.synchronize()
    assert len(got) == len(want)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert torch.equal(a, b), i


@requires_device
def test_workspace_is_zero_filled_and_reset_zeroes_it() -> None:
    """Both namespaces' workspace is a zero-filled, contiguous FP32 CUDA tensor
    of ``backward_workspace_bytes`` bytes, 18952 for every row count.  The
    facade's reset zeroes a workspace something else wrote into, and a
    backward on it then gives the bits of a fresh workspace and leaves it
    zero."""
    device = torch.device("cuda")
    for module in NAMESPACES.values():
        for rows in (0, 1, 129 * 48, 10**7):
            ws = module.cake_rmsnorm_silu_gate_det_train_backward_workspace(
                rows, device
            )
            assert ws.dtype == torch.float32 and ws.is_contiguous(), rows
            assert ws.device.type == "cuda" and ws.numel() == WORKSPACE_ELEMENTS
            assert not bool(ws.view(torch.int32).any()), rows
            nbytes = module.cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes(
                rows, device
            )
            assert nbytes == ws.numel() * 4 == WORKSPACE_BYTES, rows
    g = torch.Generator(device="cuda").manual_seed(17)
    x, z, dy = (
        torch.randn(129, 48, D, device="cuda", generator=g).bfloat16() for _ in range(3)
    )
    w = (1.0 + 0.5 * torch.randn(D, device="cuda", generator=g)).bfloat16()
    ws = cake_rmsnorm_silu_gate_det_train_backward_workspace(129 * 48, device)
    ws.view(torch.int32).fill_(7)
    cake_rmsnorm_silu_gate_det_train_reset_workspace(ws)
    torch.cuda.synchronize()
    assert not bool(ws.view(torch.int32).any())
    got = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS, workspace=ws)
    want = cake_rmsnorm_silu_gate_det_train_backward(
        dy,
        x,
        z,
        w,
        EPS,
        workspace=cake_rmsnorm_silu_gate_det_train_backward_workspace(129 * 48, device),
    )
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)
    torch.cuda.synchronize()
    assert not bool(ws.view(torch.int32).any())


@requires_device
@pytest.mark.parametrize("dtype", [torch.float64, torch.uint8], ids=["fp64", "uint8"])
def test_backward_refuses_the_nondeterministic_workspace(dtype: torch.dtype) -> None:
    """The backward and the reset refuse an FP64 or ``uint8`` tensor (the
    non-deterministic package's workspace) naming both modes' workspaces and
    leave it as it is; another unfit FP32 buffer keeps its own message."""
    x = torch.randn(129, 48, D, device="cuda").bfloat16()
    w = torch.ones(D, dtype=torch.bfloat16, device="cuda")
    other = torch.zeros(4 * D + 1, dtype=dtype, device="cuda")
    for module in NAMESPACES.values():
        with pytest.raises(ValueError) as info:
            module.cake_rmsnorm_silu_gate_det_train_backward(
                x, x, x, w, EPS, workspace=other
            )
        assert str(info.value) == _refusal(dtype), str(info.value)
        with pytest.raises(ValueError) as info:
            module.cake_rmsnorm_silu_gate_det_train_reset_workspace(other)
        assert str(info.value) == _refusal(dtype), str(info.value)
        with pytest.raises(ValueError, match="contiguous, 16-byte aligned FP32 tensor"):
            module.cake_rmsnorm_silu_gate_det_train_backward(
                x,
                x,
                x,
                w,
                EPS,
                workspace=torch.zeros(1, dtype=torch.float32, device="cuda"),
            )
    torch.cuda.synchronize()
    assert not bool(other.any())


@requires_device
@pytest.mark.parametrize("dtype", [torch.float64, torch.uint8], ids=["fp64", "uint8"])
@pytest.mark.parametrize("shape", [(0, D), (0, 48, D)], ids=["0x128", "0x48x128"])
def test_backward_with_no_rows_refuses_the_nondeterministic_workspace(
    shape: tuple, dtype: torch.dtype
) -> None:
    """A call with no rows refuses the non-deterministic package's FP64 or
    ``uint8`` workspace with the ``ValueError`` of a call with rows, before it
    returns (``grad_weight`` keeps its value).  With this package's FP32
    workspace it returns zero-sized ``dx`` and ``dz`` and a zero ``dw``, or
    keeps ``grad_weight`` when accumulating."""
    empty = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(129, 48, D, device="cuda").bfloat16()
    w = torch.ones(D, dtype=torch.bfloat16, device="cuda")
    other = torch.zeros(4 * D + 1, dtype=dtype, device="cuda")
    ws = cake_rmsnorm_silu_gate_det_train_backward_workspace(0, empty.device)
    assert ws.dtype == torch.float32
    for module in NAMESPACES.values():
        backward = module.cake_rmsnorm_silu_gate_det_train_backward
        with pytest.raises(ValueError) as with_rows:
            backward(x, x, x, w, EPS, workspace=other)
        main = torch.full((D,), 1.5, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError) as no_rows:
            backward(empty, empty, empty, w, EPS, grad_weight=main, workspace=other)
        assert str(no_rows.value) == str(with_rows.value), str(no_rows.value)
        assert str(no_rows.value) == _refusal(dtype), str(no_rows.value)
        assert torch.equal(main, torch.full_like(main, 1.5))
        dx, dz, dw = backward(empty, empty, empty, w, EPS, workspace=ws)
        assert dx.shape == dz.shape == empty.shape and dx.numel() == dz.numel() == 0
        assert dw.dtype == torch.bfloat16 and torch.equal(dw, torch.zeros_like(dw))
        _, _, dw = backward(
            empty,
            empty,
            empty,
            w,
            EPS,
            grad_weight=main,
            accumulate_grad_weight=True,
            workspace=ws,
        )
        assert dw is main and torch.equal(main, torch.full_like(main, 1.5))
        _, _, dw = backward(empty, empty, empty, w, EPS, grad_weight=main, workspace=ws)
        assert dw is main and torch.equal(main, torch.zeros_like(main))
