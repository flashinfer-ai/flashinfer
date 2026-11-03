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
package with the deterministic default and the opt-in non-deterministic weight
gradient: the ``flashinfer.cake_rmsnorm_train`` facade, its re-export from the
``flashinfer`` top level, the ``deterministic`` flag of every entry point and its
interplay with ``torch.use_deterministic_algorithms`` (the ``warn_only``
fallback included), the standalone import of the backend, the workspace
contract (one buffer serves any sequence of layouts and is at rest after every
call), the one-kernel launch of a single-kernel window and the N-D autograd
gradients.  The default call returns the bits of
``flashinfer.cake_rmsnorm_det_train``; ``deterministic=False`` runs this
package's modules.  The numerical contract of the non-deterministic kernels is
``test_cake_rmsnorm_train.py``'s.
"""

from __future__ import annotations

import ast
import inspect
import sys
import warnings

import pytest
import torch

import flashinfer
import flashinfer.cake_rmsnorm_det_train as det
import flashinfer.cake_rmsnorm_train_backend as backend
from flashinfer.cake_rmsnorm_train import (
    CakeRMSNormFunction,
    cake_rmsnorm,
    cake_rmsnorm_train_backward,
    cake_rmsnorm_train_backward_workspace,
    cake_rmsnorm_train_backward_workspace_bytes,
    cake_rmsnorm_train_forward,
    is_cake_rmsnorm_train_supported,
)
from flashinfer.cake_rmsnorm_train_backend import cake_jit as loader
from flashinfer.jit.env import has_flashinfer_jit_cache
from flashinfer.utils import get_compute_capability

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTHS = ((6144, 1e-5), (4096, 1e-5), (2048, 1e-6), (1536, 1e-6), (512, 1e-6))
HIDDEN_SIZES = tuple(hidden for hidden, _eps in WIDTHS)
EPS = {hidden: eps for hidden, eps in WIDTHS}
SEED = 8066
ENTRY_POINTS = (
    cake_rmsnorm_train_forward,
    cake_rmsnorm_train_backward,
    cake_rmsnorm_train_backward_workspace,
    cake_rmsnorm_train_backward_workspace_bytes,
    cake_rmsnorm,
    is_cake_rmsnorm_train_supported,
)


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and get_compute_capability(_device()) in SUPPORTED_CAPABILITIES
        and is_cake_rmsnorm_train_supported(_device(), deterministic=False)
    )


requires_device = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_train needs an SM100 / SM103 / SM107 device with exported routes",
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


def _randn(*shape: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        *shape, dtype=torch.float32, device=_device(), generator=generator
    ).to(torch.bfloat16)


def _skip_unless_hidden(hidden: int, deterministic: bool) -> None:
    if not is_cake_rmsnorm_train_supported(
        _device(), hidden, deterministic=deterministic
    ):
        pytest.skip(f"hidden={hidden} has no exported route on this device")


# --------------------------------------------------------------------------- the surface (no device)


def test_function_signature_is_exported() -> None:
    assert flashinfer.cake_rmsnorm is cake_rmsnorm
    assert flashinfer.cake_rmsnorm_train_forward is cake_rmsnorm_train_forward
    assert flashinfer.cake_rmsnorm_train_backward is cake_rmsnorm_train_backward
    assert (
        flashinfer.cake_rmsnorm_train_backward_workspace_bytes
        is cake_rmsnorm_train_backward_workspace_bytes
    )
    assert flashinfer.CakeRMSNormFunction is CakeRMSNormFunction
    # the det names stay the det package's objects
    assert flashinfer.cake_rmsnorm_det is det.cake_rmsnorm_det
    assert flashinfer.CakeRMSNormDetFunction is det.CakeRMSNormDetFunction


@pytest.mark.parametrize("entry", ENTRY_POINTS, ids=lambda f: f.__name__)
def test_every_entry_point_defaults_to_deterministic(entry) -> None:
    parameter = inspect.signature(entry).parameters["deterministic"]
    assert parameter.default is True
    assert parameter.kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    forward = inspect.signature(CakeRMSNormFunction.forward).parameters
    assert forward["deterministic"].default is True


def _cpu_tensors():
    x = torch.zeros(4, 512, dtype=torch.bfloat16)
    w = torch.ones(512, dtype=torch.bfloat16)
    rstd = torch.ones(4, dtype=torch.float32)
    return x, w, rstd


_DET_ENTRIES = (
    "cake_rmsnorm_det_train_forward",
    "cake_rmsnorm_det_train_backward",
    "cake_rmsnorm_det_train_backward_workspace",
    "cake_rmsnorm_det_train_backward_workspace_bytes",
)


def _stub_det(monkeypatch: pytest.MonkeyPatch, result) -> list[tuple[str, dict]]:
    """Replace the det package's entry points with recorders (no device needed)."""
    calls: list[tuple[str, dict]] = []
    for name in _DET_ENTRIES:
        monkeypatch.setattr(
            backend._det_package(),
            name,
            lambda *a, _name=name, **k: calls.append((_name, k)) or result,
        )
    return calls


@pytest.fixture
def restore_determinism():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize(
    "call",
    ["forward", "backward", "workspace", "workspace_bytes", "autograd"],
)
def test_nondeterministic_request_is_refused_under_deterministic_algorithms(
    call: str, monkeypatch: pytest.MonkeyPatch, restore_determinism
) -> None:
    """``deterministic=False`` while ``torch.use_deterministic_algorithms(True)``
    raises before any kernel or device query.  With ``warn_only=True`` it warns
    and falls back: the backward and the workspace helpers run the det package
    (stubbed here: no device needed); the forward and the autograd forward run
    this package's forward programs (here: CPU tensors, refused after the
    warning)."""
    x, w, rstd = _cpu_tensors()
    calls = _stub_det(monkeypatch, (x, rstd))
    run = {
        "forward": lambda: cake_rmsnorm_train_forward(x, w, 1e-6, deterministic=False),
        "backward": lambda: cake_rmsnorm_train_backward(
            x, x, w, rstd, deterministic=False
        ),
        "workspace": lambda: cake_rmsnorm_train_backward_workspace(
            4, 512, "cpu", deterministic=False
        ),
        "workspace_bytes": lambda: cake_rmsnorm_train_backward_workspace_bytes(
            4, 512, "cpu", deterministic=False
        ),
        "autograd": lambda: CakeRMSNormFunction.apply(x, w, 1e-6, None, False),
    }[call]
    torch.use_deterministic_algorithms(True)
    with pytest.raises(RuntimeError, match="deterministic=True"):
        run()
    assert calls == []
    torch.use_deterministic_algorithms(True, warn_only=True)
    if call in ("forward", "autograd"):
        with (
            pytest.warns(UserWarning, match="running this package's forward"),
            pytest.raises(ValueError, match="CUDA"),
        ):
            run()
        assert calls == []
    else:
        with pytest.warns(UserWarning, match="running the deterministic package"):
            run()
        assert len(calls) == 1 and calls[0][0].startswith("cake_rmsnorm_det_train_")


def test_deterministic_request_never_warns(
    monkeypatch: pytest.MonkeyPatch, restore_determinism
) -> None:
    x, w, rstd = _cpu_tensors()
    _stub_det(monkeypatch, (x, rstd))
    torch.use_deterministic_algorithms(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cake_rmsnorm_train_forward(x, w, 1e-6)
        cake_rmsnorm_train_forward(x, w, 1e-6, deterministic=True)


@pytest.mark.parametrize("value", [0, 1, None, "False", torch.tensor(False)])
def test_deterministic_accepts_bool_only(
    value, monkeypatch: pytest.MonkeyPatch
) -> None:
    x, w, rstd = _cpu_tensors()
    calls = _stub_det(monkeypatch, (x, rstd))
    for run in (
        lambda: cake_rmsnorm_train_forward(x, w, 1e-6, deterministic=value),
        lambda: cake_rmsnorm_train_backward(x, x, w, rstd, deterministic=value),
        lambda: cake_rmsnorm_train_backward_workspace_bytes(
            4, 512, "cpu", deterministic=value
        ),
        lambda: cake_rmsnorm_train_backward_workspace(
            4, 512, "cpu", deterministic=value
        ),
        lambda: is_cake_rmsnorm_train_supported("cpu", 512, deterministic=value),
        lambda: CakeRMSNormFunction.apply(x, w, 1e-6, None, value),
        lambda: cake_rmsnorm(x, w, 1e-6, deterministic=value),
    ):
        with pytest.raises(TypeError, match="deterministic must be a bool"):
            run()
    assert calls == []


def test_warn_only_fallback_never_hands_the_workspace_to_the_det_package(
    monkeypatch: pytest.MonkeyPatch, restore_determinism
) -> None:
    """The caller's workspace is laid out for this package's kernels: the
    ``warn_only`` fallback runs the det backward with a workspace of its own."""
    x, w, rstd = _cpu_tensors()
    calls = _stub_det(monkeypatch, (x, x))
    ws = torch.zeros(4096, dtype=torch.uint8)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning, match="workspace argument is not used"):
        cake_rmsnorm_train_backward(x, x, w, rstd, deterministic=False, workspace=ws)
    assert [name for name, _kwargs in calls] == ["cake_rmsnorm_det_train_backward"]
    assert calls[0][1]["workspace"] is None


@pytest.mark.parametrize("feature", ["nd", "w32", "nd_autograd", "w32_autograd"])
def test_warn_only_fallback_the_det_package_cannot_serve_raises_once(
    feature: str, monkeypatch: pytest.MonkeyPatch, restore_determinism
) -> None:
    """A ``warn_only`` fallback the det package cannot serve (an N-D input, an
    FP32 weight) raises one ``RuntimeError`` naming the feature and the
    ``deterministic=False`` opt-in, with no warning and no det call; the
    autograd function refuses in its forward, before any work."""
    x, w, rstd = _cpu_tensors()
    calls = _stub_det(monkeypatch, (x, x))
    if feature.startswith("nd"):
        x = x.view(2, 2, 512)
        rstd = rstd.view(2, 2)
        match = "more than two dimensions"
    else:
        w = w.float()
        match = "torch.float32 weights"
    run = (
        (lambda: CakeRMSNormFunction.apply(x, w, 1e-6, None, False))
        if feature.endswith("autograd")
        else (lambda: cake_rmsnorm_train_backward(x, x, w, rstd, deterministic=False))
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match=match) as info:
            run()
    assert "only deterministic=False serves this call" in str(info.value)
    assert [str(w.message) for w in caught] == [] and calls == []


def test_backend_imports_without_the_det_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backend imports on its own (a vendored copy has no det sibling): no
    module-level import of the det package; a deterministic call without it
    raises one clear ``RuntimeError``."""
    tree = ast.parse(inspect.getsource(backend))
    top_level = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and "cake_rmsnorm_det_train" in ast.unparse(node)
    ]
    assert top_level == []
    monkeypatch.delattr(flashinfer, "cake_rmsnorm_det_train_backend", raising=False)
    monkeypatch.setitem(sys.modules, "flashinfer.cake_rmsnorm_det_train_backend", None)
    x, w, _rstd = _cpu_tensors()
    with pytest.raises(RuntimeError, match="cannot be imported here"):
        cake_rmsnorm_train_forward(x, w, 1e-6)
    with pytest.raises(RuntimeError, match="cannot be imported here"):
        is_cake_rmsnorm_train_supported("cpu", 512)


# --------------------------------------------------------------------------- on the device


@requires_device
@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_default_returns_the_det_package_bits(
    hidden: int, warmup_jit, no_determinism
) -> None:
    """The default call of every entry point returns the det package's bits."""
    _skip_unless_hidden(hidden, True)
    eps = EPS[hidden]
    generator = torch.Generator(device=_device()).manual_seed(SEED + hidden)
    x = _randn(129, hidden, generator=generator)
    g = _randn(129, hidden, generator=generator)
    w = _randn(hidden, generator=generator)
    y, rstd = cake_rmsnorm_train_forward(x, w, eps)
    y_det, rstd_det = det.cake_rmsnorm_det_train_forward(x, w, eps)
    assert torch.equal(y, y_det) and torch.equal(rstd, rstd_det)
    dx, dw = cake_rmsnorm_train_backward(g, x, w, rstd)
    dx_det, dw_det = det.cake_rmsnorm_det_train_backward(g, x, w, rstd)
    assert torch.equal(dx, dx_det) and torch.equal(dw, dw_det)
    assert cake_rmsnorm_train_backward_workspace_bytes(
        129, hidden, _device()
    ) == det.cake_rmsnorm_det_train_backward_workspace_bytes(129, hidden, _device())
    xa = x.clone().requires_grad_(True)
    wa = w.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)
    wb = w.clone().requires_grad_(True)
    ya = cake_rmsnorm(xa, wa, eps)
    yb = det.cake_rmsnorm_det(xb, wb, eps)
    assert torch.equal(ya, yb)
    ya.backward(g)
    yb.backward(g)
    assert torch.equal(xa.grad, xb.grad) and torch.equal(wa.grad, wb.grad)


@requires_device
@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_nondeterministic_flag_selects_this_package(
    hidden: int, warmup_jit, no_determinism, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deterministic=False`` launches this package's modules (and no det
    module) -- exactly the records the registry's windows select for the call:
    one forward, then one single backward kernel or a det pair's main kernel and
    dw reduce; the forward bits equal the det forward's where the two packages
    select the same program, and ``dx`` equals the det ``dx`` up to the BF16
    rounding of the same formula."""
    _skip_unless_hidden(hidden, False)
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
    y, rstd = cake_rmsnorm_train_forward(x, w, eps, deterministic=False)
    dx, dw = cake_rmsnorm_train_backward(g, x, w, rstd, deterministic=False)
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
        2 if owners[0] == "claim_sum_f64" else 3
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
                    cake_rmsnorm_train_backward_workspace_bytes(
                        rows, hidden, _device(), deterministic=False
                    )
                    == loader.SUMS_OFFSET + -(-sums // 256) * 256
                )
        generator = torch.Generator(device=_device()).manual_seed(SEED + hidden)
        for rows in (257, 4096):
            ws = cake_rmsnorm_train_backward_workspace(
                rows, hidden, _device(), deterministic=False
            )
            assert ws.dtype == torch.uint8 and not bool(ws.any())
            assert ws.numel() == cake_rmsnorm_train_backward_workspace_bytes(
                rows, hidden, _device(), deterministic=False
            )
            for tokens in (1, rows // 2, rows):
                x = _randn(tokens, hidden, generator=generator)
                g = _randn(tokens, hidden, generator=generator)
                w = _randn(hidden, generator=generator)
                _y, rstd = cake_rmsnorm_train_forward(
                    x, w, EPS[hidden], deterministic=False
                )
                dx, dw = cake_rmsnorm_train_backward(
                    g, x, w, rstd, deterministic=False, workspace=ws
                )
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
    outputs = cake_rmsnorm_train_forward(x, w, eps, residual=u, deterministic=False)
    normalized = outputs[2] if residual else x
    rstd = outputs[1]
    dx, dw = cake_rmsnorm_train_backward(
        g, normalized, w, rstd, deterministic=False, g_residual=gh, workspace=ws
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
        cake_rmsnorm_train_backward_workspace_bytes(
            tokens, hidden, _device(), residual=residual, deterministic=False
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
        cake_rmsnorm_train_backward_workspace_bytes(
            tokens, hidden, _device(), residual=residual, deterministic=False
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
    from torch.profiler import ProfilerActivity, profile

    from flashinfer.cake_rmsnorm_train_backend import cake_windows as windows

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
    _y, rstd = cake_rmsnorm_train_forward(
        x, w, EPS.get(hidden, 1e-6), deterministic=False
    )
    ws = cake_rmsnorm_train_backward_workspace(
        tokens, hidden, _device(), residual=residual, deterministic=False
    )
    dx = torch.empty_like(x)
    dw = torch.empty(hidden, dtype=torch.float32, device=_device())

    def call():
        cake_rmsnorm_train_backward(
            g,
            x,
            w,
            rstd,
            deterministic=False,
            g_residual=gh,
            workspace=ws,
            dx=dx,
            dw=dw,
        )

    call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    activities = [
        event
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    names = [event.name for event in activities]
    assert len(names) == 1, names
    assert "rmsnorm" in names[0] and "reduce" not in names[0], names
    assert _at_rest(ws)


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
    y = cake_rmsnorm(x, w, eps, deterministic=False)
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
@pytest.mark.parametrize("hidden", [256, 5120])
def test_warn_only_fallback_on_a_width_the_det_package_lacks(
    hidden: int, restore_determinism
) -> None:
    """R3 C4 host-6 on the device: a width this package serves and the det
    package does not (on this device) raises one ``RuntimeError`` naming it
    under ``warn_only``, with no warning; the forward runs this package's
    programs."""
    arch = loader.device_arch(_device().index or 0)
    if hidden not in loader.supported_hidden_sizes(arch):
        pytest.skip(f"hidden={hidden} has no non-deterministic route on {arch}")
    if det.is_cake_rmsnorm_det_train_supported(_device(), hidden):
        pytest.skip(f"the det package serves hidden={hidden} on {arch}")
    generator = torch.Generator(device=_device()).manual_seed(SEED + hidden)
    x = _randn(64, hidden, generator=generator)
    g = _randn(64, hidden, generator=generator)
    w = _randn(hidden, generator=generator)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning, match="running this package's forward"):
        _y, rstd = cake_rmsnorm_train_forward(x, w, 1e-6, deterministic=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match=f"hidden={hidden}"):
            cake_rmsnorm_train_backward(g, x, w, rstd, deterministic=False)
    assert [str(item.message) for item in caught] == []
