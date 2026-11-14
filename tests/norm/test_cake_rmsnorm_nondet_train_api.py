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

Tests of the flashinfer public surface of the Cake fused BF16 RMSNorm training
package with the non-deterministic weight gradient: the
``flashinfer.cake_rmsnorm_nondet_train`` facade (the only way to reach it: the
``flashinfer`` top level binds none of its names), the absent
``deterministic`` keyword (the mode is part of the name only; passing it is
Python's ``TypeError``), the interplay with
``torch.use_deterministic_algorithms`` (only the backward and the autograd
backward alert: ``RuntimeError``; with ``warn_only``, one warning naming the
det entry point, then this package's kernels), the standalone import of the
backend, the workspace contract (one buffer serves any sequence of layouts and
is at rest after every call), the one-kernel launch of a single-kernel window
and the N-D autograd gradients.  The names of both packages and the old names
that fail at import are ``test_cake_rmsnorm_det_train_names.py``'s; the
numerical contract of the kernels is ``test_cake_rmsnorm_nondet_train.py``'s.
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
import warnings
from typing import Optional

import pytest
import torch

import flashinfer
import flashinfer.cake_rmsnorm_det_train as det
import flashinfer.experimental.cake as cake_namespace
import flashinfer.experimental.cake.cake_rmsnorm_nondet_train as backend
from flashinfer.cake_rmsnorm_nondet_train import (
    CakeRMSNormNondetFunction,
    cake_rmsnorm_nondet,
    cake_rmsnorm_nondet_train_backward,
    cake_rmsnorm_nondet_train_backward_workspace,
    cake_rmsnorm_nondet_train_backward_workspace_bytes,
    cake_rmsnorm_nondet_train_forward,
    is_cake_rmsnorm_nondet_train_supported,
)
from flashinfer.experimental.cake.cake_rmsnorm_nondet_train import cake_jit as loader
from flashinfer.jit.env import has_flashinfer_jit_cache
from flashinfer.utils import get_compute_capability

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTHS = ((6144, 1e-5), (4096, 1e-5), (2048, 1e-6), (1536, 1e-6), (512, 1e-6))
HIDDEN_SIZES = tuple(hidden for hidden, _eps in WIDTHS)
EPS = {hidden: eps for hidden, eps in WIDTHS}
SEED = 8066
ENTRY_POINTS = (
    cake_rmsnorm_nondet_train_forward,
    cake_rmsnorm_nondet_train_backward,
    cake_rmsnorm_nondet_train_backward_workspace,
    cake_rmsnorm_nondet_train_backward_workspace_bytes,
    cake_rmsnorm_nondet,
    is_cake_rmsnorm_nondet_train_supported,
)


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and get_compute_capability(_device()) in SUPPORTED_CAPABILITIES
        and is_cake_rmsnorm_nondet_train_supported(_device())
    )


requires_device = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_nondet_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(scope="module")
def warmup_jit():
    if _supported() and not has_flashinfer_jit_cache():
        arch = loader.arch_for_capability(get_compute_capability(_device()))
        flashinfer.jit.build_jit_specs(loader.specs_for_arch(arch), verbose=False)
    yield


@pytest.fixture
def no_determinism():
    """Run with ``torch.use_deterministic_algorithms`` off and restore the caller's setting."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.fixture
def restore_determinism():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _randn(*shape: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        *shape, dtype=torch.float32, device=_device(), generator=generator
    ).to(torch.bfloat16)


def _skip_unless_hidden(hidden: int) -> None:
    if not is_cake_rmsnorm_nondet_train_supported(_device(), hidden):
        pytest.skip(f"hidden={hidden} has no exported route on this device")


def _package_warnings(caught) -> list[str]:
    """The warnings of this package among ``caught``."""
    return [
        str(item.message)
        for item in caught
        if "flashinfer.cake_rmsnorm_nondet_train." in str(item.message)
    ]


def _names(text: str, name: str) -> bool:
    """Whether ``text`` names ``name`` itself, not a longer name it begins
    (``cake_rmsnorm_det`` begins ``cake_rmsnorm_det_train_backward``)."""
    return re.search(rf"{re.escape(name)}\b", text) is not None


# --------------------------------------------------------------------------- the surface (no device)


def test_entry_points_are_reached_through_the_module_only() -> None:
    module = flashinfer.cake_rmsnorm_nondet_train
    assert module.cake_rmsnorm_nondet is cake_rmsnorm_nondet
    assert module.cake_rmsnorm_nondet_train_forward is cake_rmsnorm_nondet_train_forward
    assert (
        module.cake_rmsnorm_nondet_train_backward is cake_rmsnorm_nondet_train_backward
    )
    assert (
        module.cake_rmsnorm_nondet_train_backward_workspace_bytes
        is cake_rmsnorm_nondet_train_backward_workspace_bytes
    )
    assert module.CakeRMSNormNondetFunction is CakeRMSNormNondetFunction
    # the top level binds none of these names, only the det package's
    for name in (
        "cake_rmsnorm_nondet",
        "cake_rmsnorm_nondet_train_forward",
        "cake_rmsnorm_nondet_train_backward",
        "cake_rmsnorm_nondet_train_backward_workspace_bytes",
        "CakeRMSNormNondetFunction",
    ):
        assert not hasattr(flashinfer, name), name
    assert flashinfer.cake_rmsnorm_det is det.cake_rmsnorm_det
    assert flashinfer.CakeRMSNormDetFunction is det.CakeRMSNormDetFunction


@pytest.mark.parametrize("entry", ENTRY_POINTS, ids=lambda f: f.__name__)
def test_no_entry_point_takes_a_deterministic_keyword(entry) -> None:
    """The mode is part of the name only: no entry point (nor the autograd
    function's forward) has a ``deterministic`` parameter, and
    ``cake_rmsnorm_nondet`` keeps its call shape ``(x, w, eps, residual)``."""
    assert "deterministic" not in inspect.signature(entry).parameters
    forward = inspect.signature(CakeRMSNormNondetFunction.forward).parameters
    assert list(forward) == ["ctx", "x", "w", "eps", "residual"]
    assert list(inspect.signature(cake_rmsnorm_nondet).parameters) == [
        "x",
        "w",
        "eps",
        "residual",
    ]


def _cpu_tensors():
    x = torch.zeros(4, 512, dtype=torch.bfloat16)
    w = torch.ones(512, dtype=torch.bfloat16)
    rstd = torch.ones(4, dtype=torch.float32)
    return x, w, rstd


@pytest.mark.parametrize("mode", ["strict", "warn_only"])
@pytest.mark.parametrize("feature", ["plain", "nd", "w32"])
def test_only_the_backward_consults_deterministic_algorithms(
    feature: str, mode: str, restore_determinism
) -> None:
    """``torch.use_deterministic_algorithms(True)`` alerts only before work whose
    result can vary from run to run.  The forward and the autograd forward never
    alert: they run this package's path (here: CPU tensors, refused by this
    package's own CUDA check, no warning).  The backward always alerts:
    ``RuntimeError`` naming the det backward before any device query, or, with
    ``warn_only=True``, one ``UserWarning`` naming it, then this package's
    backward -- also for N-D inputs and FP32 weights (no det fallback, no other
    error)."""
    x, w, rstd = _cpu_tensors()
    if feature == "nd":
        x = x.view(2, 2, 512)
        rstd = rstd.view(2, 2)
    elif feature == "w32":
        w = w.float()
    torch.use_deterministic_algorithms(True, warn_only=mode == "warn_only")
    for run in (
        lambda: cake_rmsnorm_nondet_train_forward(x, w, 1e-6),
        lambda: cake_rmsnorm_nondet(x, w, 1e-6),
        lambda: CakeRMSNormNondetFunction.apply(x, w, 1e-6, None),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="CUDA"):
                run()
        assert _package_warnings(caught) == []
    det_name = "flashinfer.cake_rmsnorm_det_train.cake_rmsnorm_det_train_backward"
    if mode == "strict":
        with pytest.raises(RuntimeError) as info:
            cake_rmsnorm_nondet_train_backward(x, x, w, rstd)
        assert _names(str(info.value), det_name), str(info.value)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="CUDA"):
            cake_rmsnorm_nondet_train_backward(x, x, w, rstd)
    messages = _package_warnings(caught)
    assert len(messages) == 1, messages
    assert _names(messages[0], det_name), messages[0]
    assert "running this package's kernels" in messages[0]


@pytest.mark.parametrize("entry", ["cake_rmsnorm_nondet", "CakeRMSNormNondetFunction"])
@pytest.mark.parametrize("mode", ["strict", "warn_only"])
def test_autograd_backward_consults_deterministic_algorithms_when_it_runs(
    mode: str, entry: str, monkeypatch: pytest.MonkeyPatch, restore_determinism
) -> None:
    """The autograd forward does not consult
    ``torch.use_deterministic_algorithms``; its backward does when it runs:
    ``RuntimeError`` naming ``cake_rmsnorm_det`` before this package's backward,
    or, with ``warn_only=True``, one ``UserWarning`` naming it, then this
    package's backward.  This package's forward and backward are stubbed (no
    device needed)."""
    calls: list[tuple[int, ...]] = []

    def fake_forward(x, w, eps, *, residual, out, rstd, residual_out):
        return x.clone(), torch.ones(x.shape[0], dtype=torch.float32)

    def fake_backward(g, x, w, rstd, *, g_residual, workspace, dx, dw):
        calls.append(tuple(g.shape))
        return torch.zeros_like(x), torch.zeros(w.shape[0], dtype=torch.float32)

    monkeypatch.setattr(backend, "_forward", fake_forward)
    monkeypatch.setattr(backend, "_backward", fake_backward)
    x, w, _rstd = _cpu_tensors()
    xa = x.clone().requires_grad_(True)
    wa = w.clone().requires_grad_(True)
    torch.use_deterministic_algorithms(True, warn_only=mode == "warn_only")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if entry == "cake_rmsnorm_nondet":
            y = cake_rmsnorm_nondet(xa, wa, 1e-6)
        else:
            y = CakeRMSNormNondetFunction.apply(xa, wa, 1e-6, None)
    assert _package_warnings(caught) == []
    det_name = "flashinfer.cake_rmsnorm_det_train.cake_rmsnorm_det"
    if mode == "strict":
        with pytest.raises(RuntimeError) as info:
            y.backward(torch.ones_like(y))
        assert _names(str(info.value), det_name), str(info.value)
        assert calls == []
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y.backward(torch.ones_like(y))
    messages = _package_warnings(caught)
    assert len(messages) == 1, messages
    assert _names(messages[0], det_name), messages[0]
    assert "running this package's kernels" in messages[0]
    assert calls == [tuple(x.shape)]
    assert wa.grad is not None and wa.grad.dtype == w.dtype


@pytest.mark.parametrize("mode", ["strict", "warn_only"])
def test_workspace_helpers_and_support_query_ignore_deterministic_algorithms(
    mode: str, restore_determinism
) -> None:
    """The workspace helpers and the support query do not consult
    ``torch.use_deterministic_algorithms``: no warning, no ``RuntimeError``
    (here: CPU, so the helpers reach this package's CUDA check)."""
    torch.use_deterministic_algorithms(True, warn_only=mode == "warn_only")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert is_cake_rmsnorm_nondet_train_supported("cpu", 512) is False
        assert is_cake_rmsnorm_nondet_train_supported("cpu") is False
        with pytest.raises(ValueError, match="CUDA"):
            cake_rmsnorm_nondet_train_backward_workspace_bytes(4, 512, "cpu")
        with pytest.raises(ValueError, match="CUDA"):
            cake_rmsnorm_nondet_train_backward_workspace(4, 512, "cpu")
    assert _package_warnings(caught) == []


def test_backend_never_needs_the_det_package(
    monkeypatch: pytest.MonkeyPatch, no_determinism
) -> None:
    """The backend imports and runs on its own (a vendored copy has no det
    sibling): no import of the det package anywhere in its source, and with
    the det package unimportable every entry point behaves as before -- the
    calls reach this package's own checks and a ``deterministic`` keyword is
    still Python's ``TypeError``."""
    tree = ast.parse(inspect.getsource(backend))
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and "cake_rmsnorm_det_train" in ast.unparse(node)
    ]
    assert imports == []
    monkeypatch.delattr(cake_namespace, "cake_rmsnorm_det_train", raising=False)
    monkeypatch.setitem(
        sys.modules, "flashinfer.experimental.cake.cake_rmsnorm_det_train", None
    )
    x, w, rstd = _cpu_tensors()
    assert is_cake_rmsnorm_nondet_train_supported("cpu", 512) is False
    with pytest.raises(ValueError, match="CUDA"):
        cake_rmsnorm_nondet_train_forward(x, w, 1e-6)
    with pytest.raises(ValueError, match="CUDA"):
        cake_rmsnorm_nondet_train_backward(x, x, w, rstd)
    for value in (True, False):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            cake_rmsnorm_nondet_train_forward(x, w, 1e-6, deterministic=value)


# --------------------------------------------------------------------------- on the device


@requires_device
@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_every_call_runs_this_package(
    hidden: int, warmup_jit, no_determinism, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every call launches this package's modules, the same ones on a repeat
    with the same ``y``, ``rstd`` and ``dx`` bits (``dx`` is bitwise
    reproducible) and ``dw`` within the budget of the FP64 reference; the
    autograd wrapper returns that ``y`` and ``dx`` and the kernel's ``dw`` in
    the weight's dtype."""
    _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    launched: list[str] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        launched.append(name)
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    generator = torch.Generator(device=_device()).manual_seed(SEED + hidden)
    x = _randn(129, hidden, generator=generator)
    g = _randn(129, hidden, generator=generator)
    w = _randn(hidden, generator=generator)
    y, rstd = cake_rmsnorm_nondet_train_forward(x, w, eps)
    dx, dw = cake_rmsnorm_nondet_train_backward(g, x, w, rstd)
    first = list(launched)
    launched.clear()
    y_nd, rstd_nd = cake_rmsnorm_nondet_train_forward(x, w, eps)
    dx_nd, dw_nd = cake_rmsnorm_nondet_train_backward(g, x, w, rstd)
    assert first == launched and all(name in loader.MODULES for name in first)
    assert torch.equal(y, y_nd) and torch.equal(rstd, rstd_nd)
    assert torch.equal(dx, dx_nd)
    reference = _dw_reference(g, x, rstd)
    assert _rel(dw, reference) <= 1e-4 and _rel(dw_nd, reference) <= 1e-4
    xa = x.clone().requires_grad_(True)
    wa = w.clone().requires_grad_(True)
    launched.clear()
    ya = cake_rmsnorm_nondet(xa, wa, eps)
    assert torch.equal(ya, y)
    ya.backward(g)
    assert launched == first, launched
    assert torch.equal(xa.grad, dx)
    assert wa.grad.dtype == w.dtype
    # the kernel's FP32 dw rounded to BF16 (at most 2^-9 relative per element)
    assert _rel(wa.grad, reference) <= 2.0**-8, _rel(wa.grad, reference)


@requires_device
@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_nondet_calls_select_this_package(
    hidden: int, warmup_jit, no_determinism, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The nondet calls launch this package's modules (and no det module) --
    exactly the records the registry's windows select for the call: one
    forward, then one single backward kernel or a det pair's main kernel and
    dw reduce; the forward bits equal the det forward's where the two packages
    select the same program, and ``dx`` equals the det ``dx`` up to the BF16
    rounding of the same formula."""
    _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    launched: list[str] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        launched.append(name)
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    generator = torch.Generator(device=_device()).manual_seed(SEED + 7 * hidden)
    x = _randn(257, hidden, generator=generator)
    g = _randn(257, hidden, generator=generator)
    w = _randn(hidden, generator=generator)
    y, rstd = cake_rmsnorm_nondet_train_forward(x, w, eps)
    dx, dw = cake_rmsnorm_nondet_train_backward(g, x, w, rstd)
    assert launched and all(name in loader.MODULES for name in launched)
    arch = loader.device_arch(_device().index or 0)
    expected = [
        *loader.selected_modules(arch, hidden, "fwd", False, rows=257),
        *loader.selected_modules(arch, hidden, "bwd", False, rows=257),
    ]
    assert launched == expected, launched
    owners = [
        loader.MODULES[name]["workspace_rule"]["kind"]
        for name in launched
        if loader.MODULES[name]["workspace_rule"] is not None
    ]
    assert len(owners) == 1 and len(launched) == (
        2 if owners[0] in ("claim_sum_f64", "claim_part_f32") else 3
    ), launched
    y_det, rstd_det = det.cake_rmsnorm_det_train_forward(x, w, eps)
    torch.testing.assert_close(y.float(), y_det.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(rstd, rstd_det, atol=0, rtol=1e-6)
    dx_det, dw_det = det.cake_rmsnorm_det_train_backward(g, x, w, rstd)
    torch.testing.assert_close(dx.float(), dx_det.float(), atol=1e-2, rtol=1e-2)
    rel = (dw.double() - dw_det.double()).norm() / dw_det.double().norm().clamp_min(
        1e-30
    )
    assert float(rel) <= 1e-4, float(rel)


def _at_rest(ws: torch.Tensor) -> bool:
    """The workspace's documented rest state after every call: every counter and
    FP64 sum zero -- every byte before the det pair's partial rows
    (``cake_jit.PAIR_ROWS_OFFSET``), which are scratch."""
    torch.cuda.synchronize()
    return not bool(ws[: loader.PAIR_ROWS_OFFSET].any())


def _dw_reference(g, x, rstd) -> torch.Tensor:
    return (g.double() * x.double() * rstd.double().unsqueeze(-1)).sum(0)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


@requires_device
def test_nondeterministic_workspace_serves_every_call_up_to_its_rows(
    no_determinism,
) -> None:
    """The non-deterministic workspace of ``rows`` serves every backward call of
    its width of at most ``rows`` tokens without a host write and is at rest
    after each call; a width whose windows all run the single kernel needs the
    counters and its sums whatever ``rows``."""

    arch = loader.device_arch(_device().index or 0)
    for hidden in (512, 2048, 6144):
        if hidden not in loader.supported_hidden_sizes(arch):
            continue
        records = [
            loader.MODULES[name]
            for w_dtype in loader.weight_dtypes(arch, hidden, "bwd", False)
            for name in loader.route_modules(arch, hidden, "bwd", False, w_dtype)
            if loader.MODULES[name]["workspace_rule"] is not None
        ]
        if all(r["workspace_rule"]["kind"] == "claim_sum_f64" for r in records):
            sums = max(r["workspace_rule"]["sum_bytes"] for r in records)
            for rows in (0, 1, 12345, 65536):
                assert (
                    cake_rmsnorm_nondet_train_backward_workspace_bytes(
                        rows, hidden, _device()
                    )
                    == loader.SUMS_OFFSET + -(-sums // 256) * 256
                )
        generator = torch.Generator(device=_device()).manual_seed(SEED + hidden)
        for rows in (257, 4096):
            ws = cake_rmsnorm_nondet_train_backward_workspace(rows, hidden, _device())
            assert ws.dtype == torch.uint8 and not bool(ws.any())
            assert ws.numel() == cake_rmsnorm_nondet_train_backward_workspace_bytes(
                rows, hidden, _device()
            )
            for tokens in (1, rows // 2, rows):
                x = _randn(tokens, hidden, generator=generator)
                g = _randn(tokens, hidden, generator=generator)
                w = _randn(hidden, generator=generator)
                _y, rstd = cake_rmsnorm_nondet_train_forward(x, w, EPS[hidden])
                dx, dw = cake_rmsnorm_nondet_train_backward(g, x, w, rstd, workspace=ws)
                rel = _rel(dw, _dw_reference(g, x, rstd))
                assert rel <= 1e-4, (hidden, rows, tokens, rel)
                assert _at_rest(ws), (hidden, rows, tokens)


def _sequence(arch: str) -> list[tuple[int, bool, torch.dtype, int]]:
    """``(hidden, residual, w dtype, tokens)`` calls that change every layout
    parameter (R3 C1.4a: H6144 res -> H256 -> H2048 -> H6144 -> H6144 res, the
    FP32 weight where a route ships it), restricted to the device's routes."""
    wanted = [
        (6144, True, torch.bfloat16, 3000),
        (256, False, torch.bfloat16, 257),
        (2048, False, torch.bfloat16, 4096),
        (6144, False, torch.float32, 1),
        (6144, False, torch.bfloat16, 20000),
        (2048, False, torch.float32, 640),
        (6144, True, torch.bfloat16, 129),
    ]
    out = []
    for hidden, residual, w_dtype, tokens in wanted:
        token = "f32" if w_dtype == torch.float32 else "bf16"
        if hidden not in loader.supported_hidden_sizes(arch):
            continue
        if token not in loader.weight_dtypes(arch, hidden, "bwd", residual):
            continue
        out.append((hidden, residual, w_dtype, tokens))
    return out


def _call(hidden, residual, w_dtype, tokens, ws, generator):
    eps = EPS.get(hidden, 1e-6)
    x = _randn(tokens, hidden, generator=generator)
    g = _randn(tokens, hidden, generator=generator)
    w = (1.0 + 0.1 * _randn(hidden, generator=generator).float()).to(w_dtype)
    u = _randn(tokens, hidden, generator=generator) if residual else None
    gh = _randn(tokens, hidden, generator=generator) if residual else None
    outputs = cake_rmsnorm_nondet_train_forward(x, w, eps, residual=u)
    normalized = outputs[2] if residual else x
    rstd = outputs[1]
    dx, dw = cake_rmsnorm_nondet_train_backward(
        g, normalized, w, rstd, g_residual=gh, workspace=ws
    )
    return _rel(dw, _dw_reference(g, normalized, rstd))


@requires_device
def test_one_workspace_serves_a_sequence_of_layouts(no_determinism) -> None:
    """R3 C1.4a/c: ONE zeroed workspace, sized for the largest call, serves a
    sequence that changes the hidden size, residual flag, weight dtype and
    token count (so the kernel instance, its sums and its windows), twice over:
    ``dw`` matches the FP64 reference after every call and the workspace is at
    rest after every call."""
    arch = loader.device_arch(_device().index or 0)
    calls = _sequence(arch)
    assert len({(h, r, d) for h, r, d, _t in calls}) >= 3, calls
    nbytes = max(
        cake_rmsnorm_nondet_train_backward_workspace_bytes(
            tokens, hidden, _device(), residual=residual
        )
        for hidden, residual, _d, tokens in calls
    )
    ws = torch.zeros(nbytes, dtype=torch.uint8, device=_device())
    generator = torch.Generator(device=_device()).manual_seed(SEED + 41)
    for _round in range(2):
        for hidden, residual, w_dtype, tokens in calls:
            rel = _call(hidden, residual, w_dtype, tokens, ws, generator)
            assert rel <= 1e-4, (hidden, residual, w_dtype, tokens, rel)
            assert _at_rest(ws), (hidden, residual, w_dtype, tokens)


@requires_device
def test_a_stale_workspace_is_correct_after_the_fresh_form(no_determinism) -> None:
    """R3 C1.4b: a workspace holding stale bytes of another layout (nonzero
    counters and sums) is correct after its documented fresh form -- zeroed
    once (``torch.zeros`` / ``ws.zero_()``) -- and stays at rest."""
    arch = loader.device_arch(_device().index or 0)
    calls = _sequence(arch)
    nbytes = max(
        cake_rmsnorm_nondet_train_backward_workspace_bytes(
            tokens, hidden, _device(), residual=residual
        )
        for hidden, residual, _d, tokens in calls
    )
    ws = torch.empty(nbytes, dtype=torch.uint8, device=_device())
    generator = torch.Generator(device=_device()).manual_seed(SEED + 43)
    for hidden, residual, w_dtype, tokens in calls:
        ws.copy_(
            torch.randint(
                1,
                256,
                (nbytes,),
                dtype=torch.uint8,
                device=_device(),
                generator=generator,
            )
        )
        ws.zero_()  # the documented fresh form
        rel = _call(hidden, residual, w_dtype, tokens, ws, generator)
        assert rel <= 1e-4, (hidden, residual, w_dtype, tokens, rel)
        assert _at_rest(ws), (hidden, residual, w_dtype, tokens)


def _single_window(arch: str) -> tuple[int, bool, str, int] | None:
    """``(hidden, residual, w dtype token, tokens)`` inside one single-kernel
    window of ``arch``'s registry, or ``None``."""
    for route, names in loader.ROUTES.get(arch, {}).items():
        if not route.startswith("bwd"):
            continue
        for name in names:
            record = loader.MODULES[name]
            rule = record["workspace_rule"]
            if rule is None or rule["kind"] != "claim_sum_f64":
                continue
            window = record["select_rule"]
            above = 0 if window is None else int(window["above"])
            at_most = None if window is None else window["at_most"]
            tokens = above + 1 if at_most is None else (above + 1 + int(at_most)) // 2
            residual = "_residual" in route
            token = "f32" if "_w32" in route else "bf16"
            return int(record["hidden"]), residual, token, tokens
    return None


@requires_device
def test_a_single_kernel_window_launches_exactly_one_kernel(no_determinism) -> None:
    """R3 C2: a backward call in a single-kernel window with an explicit
    workspace and preallocated outputs launches exactly one GPU activity -- one
    RMSNorm backward kernel, no dw reduce, memset or copy -- counted by the
    profiler, independently of the host's module selection.  An architecture
    whose every backward route is listed in ``DET_PAIR_ROUTES`` ships no single
    kernel to count."""
    from flashinfer.experimental.cake.cake_rmsnorm_nondet_train import (
        cake_windows as windows,
    )
    from tests.cake.test_helpers.cake_profiler import cuda_kernel_names

    arch = loader.device_arch(_device().index or 0)
    found = _single_window(arch)
    if found is None:
        routes = [r for r in loader.ROUTES.get(arch, {}) if r.startswith("bwd")]
        if routes and set(routes) <= set(windows.DET_PAIR_ROUTES.get(arch, {})):
            pytest.skip(f"every backward route on {arch} is listed in DET_PAIR_ROUTES")
        pytest.fail(f"no single-kernel window is shipped on {arch}")
    hidden, residual, token, tokens = found
    w_dtype = torch.float32 if token == "f32" else torch.bfloat16
    generator = torch.Generator(device=_device()).manual_seed(SEED + 47)
    x = _randn(tokens, hidden, generator=generator)
    g = _randn(tokens, hidden, generator=generator)
    w = _randn(hidden, generator=generator).to(w_dtype)
    gh = _randn(tokens, hidden, generator=generator) if residual else None
    _y, rstd = cake_rmsnorm_nondet_train_forward(x, w, EPS.get(hidden, 1e-6))
    ws = cake_rmsnorm_nondet_train_backward_workspace(
        tokens, hidden, _device(), residual=residual
    )
    dx = torch.empty_like(x)
    dw = torch.empty(hidden, dtype=torch.float32, device=_device())

    def call():
        cake_rmsnorm_nondet_train_backward(
            g,
            x,
            w,
            rstd,
            g_residual=gh,
            workspace=ws,
            dx=dx,
            dw=dw,
        )

    call()
    torch.cuda.synchronize()
    # Every CUDA activity of the call, memsets and copies included, read
    # between sentinel kernels that prove the session kept the whole window (a
    # session can lose its device records; see tests.cake.test_helpers.cake_profiler).
    # When every session lost them, the workspace is still checked before the
    # launch count is skipped.
    lost = None
    try:
        names = cuda_kernel_names(call, exclude=())
    except pytest.skip.Exception as skipped:
        lost = skipped
    assert _at_rest(ws)
    if lost is not None:
        raise lost
    assert len(names) == 1, names
    assert "rmsnorm" in names[0] and "reduce" not in names[0], names


@requires_device
@pytest.mark.parametrize("heads", [24, 4])
@pytest.mark.parametrize("consumer", ["sum", "transposed"])
def test_nd_autograd_gradients(heads: int, consumer: str, no_determinism) -> None:
    """R3 C4 host-5: ``[T, heads, 256]`` through autograd with a ``.sum()``
    consumer (an expanded all-zero-stride gradient) and a transposed consumer
    (a non-collapsible gradient layout): the backward copies such gradients
    once and returns ``dx`` / ``dw`` matching the FP64 reference."""
    arch = loader.device_arch(_device().index or 0)
    hidden = 256
    if hidden not in loader.supported_hidden_sizes(arch):
        pytest.skip(f"hidden={hidden} has no exported route on {arch}")
    eps = 1e-6
    generator = torch.Generator(device=_device()).manual_seed(SEED + heads)
    tokens = 300
    x = _randn(tokens, heads, hidden, generator=generator).requires_grad_(True)
    w = (1.0 + 0.1 * _randn(hidden, generator=generator).float()).to(torch.bfloat16)
    w.requires_grad_(True)
    scale = _randn(heads, tokens, hidden, generator=generator).float()
    y = cake_rmsnorm_nondet(x, w, eps)
    if consumer == "sum":
        loss = y.float().sum()
        grad_y = torch.ones(
            tokens, heads, hidden, dtype=torch.float64, device=_device()
        )
    else:
        loss = (y.transpose(0, 1).float() * scale).sum()
        grad_y = scale.double().transpose(0, 1)
    loss.backward()
    xd = x.detach().double().requires_grad_(True)
    wd = w.detach().double().requires_grad_(True)
    rstd = (xd.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    ((xd * rstd * wd) * grad_y).sum().backward()
    assert x.grad.shape == x.shape and w.grad.shape == w.shape
    assert _rel(x.grad, xd.grad) <= 1e-2, _rel(x.grad, xd.grad)
    assert _rel(w.grad, wd.grad) <= 1e-2, _rel(w.grad, wd.grad)


@requires_device
@pytest.mark.parametrize("w_token", ["bf16", "f32"])
@pytest.mark.parametrize("hidden", [256, 512, 5120, 6144])
def test_warn_only_warns_at_the_backward_and_runs_this_package(
    hidden: int,
    w_token: str,
    warmup_jit,
    restore_determinism,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under ``torch.use_deterministic_algorithms(True, warn_only=True)`` the
    forward and the autograd forward run this package's modules without a
    warning; the backward and the autograd backward each issue one warning
    naming the det entry point and run this package's modules -- on widths the
    det package lacks too (256, 5120), with FP32 weights and a caller's
    workspace: ``y``, ``rstd`` and ``dx`` are the bits of the same calls with
    the setting off (``dx`` is bitwise reproducible), ``dw`` is within the
    budget of the FP64 reference and the workspace is at rest after the call."""
    arch = loader.device_arch(_device().index or 0)
    if hidden not in loader.supported_hidden_sizes(arch):
        pytest.skip(f"hidden={hidden} has no route on {arch}")
    if not all(
        w_token in loader.weight_dtypes(arch, hidden, kind, False)
        for kind in ("fwd", "bwd")
    ):
        pytest.skip(f"hidden={hidden} ships no {w_token} weight route on {arch}")
    eps = EPS.get(hidden, 1e-6)
    tokens = 257
    generator = torch.Generator(device=_device()).manual_seed(SEED + 3 * hidden)
    x = _randn(tokens, hidden, generator=generator)
    g = _randn(tokens, hidden, generator=generator)
    w = (1.0 + 0.1 * _randn(hidden, generator=generator).float()).to(
        torch.float32 if w_token == "f32" else torch.bfloat16
    )
    torch.use_deterministic_algorithms(False)
    y0, rstd0 = cake_rmsnorm_nondet_train_forward(x, w, eps)
    dx0, _dw0 = cake_rmsnorm_nondet_train_backward(g, x, w, rstd0)
    ws = cake_rmsnorm_nondet_train_backward_workspace(tokens, hidden, _device())
    launched: list[str] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        launched.append(name)
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)

    def warned(call, det_name: Optional[str]):
        """``call()``, asserting one warning naming ``det_name``, or none."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = call()
        messages = _package_warnings(caught)
        if det_name is None:
            assert messages == [], messages
        else:
            assert len(messages) == 1, messages
            assert _names(
                messages[0], f"flashinfer.cake_rmsnorm_det_train.{det_name}"
            ), messages[0]
        return result

    torch.use_deterministic_algorithms(True, warn_only=True)
    y, rstd = warned(lambda: cake_rmsnorm_nondet_train_forward(x, w, eps), None)
    dx, dw = warned(
        lambda: cake_rmsnorm_nondet_train_backward(g, x, w, rstd, workspace=ws),
        "cake_rmsnorm_det_train_backward",
    )
    expected = [
        name
        for kind in ("fwd", "bwd")
        for name in loader.selected_modules(
            arch, hidden, kind, False, rows=tokens, w_dtype=w_token
        )
    ]
    assert launched == expected, launched
    assert torch.equal(y, y0) and torch.equal(rstd, rstd0)
    assert torch.equal(dx, dx0)
    rel = _rel(dw, _dw_reference(g, x, rstd))
    assert rel <= 1e-4, rel
    assert _at_rest(ws)
    xa = x.clone().requires_grad_(True)
    wa = w.clone().requires_grad_(True)
    launched.clear()
    ya = warned(lambda: cake_rmsnorm_nondet(xa, wa, eps), None)
    warned(lambda: ya.backward(g), "cake_rmsnorm_det")
    assert launched == expected, launched
    assert torch.equal(ya, y0) and torch.equal(xa.grad, dx0)
