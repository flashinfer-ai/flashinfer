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
point and its interplay with ``torch.use_deterministic_algorithms``.

The facade dispatches: ``deterministic=True`` (the default) runs the
deterministic package ``flashinfer.cake_rmsnorm_silu_gate_det_train`` (its bits,
its errors, its autograd node), ``deterministic=False`` the standalone backend
``flashinfer.cake_rmsnorm_silu_gate_train_backend``, whose entry points take
``deterministic=False`` only.  This module checks:

* the surface without a device: the exports, the keyword of every entry point,
  each entry point's dispatch (the two packages' entry points replaced by
  recorders), the refusal of ``deterministic=False`` under
  ``torch.use_deterministic_algorithms(True)`` (with ``warn_only=True``: a
  warning and the deterministic package), the alert of a ``False`` autograd
  node whose backward runs after the mode was enabled, the workspace reset of
  both modes, and the backend's standalone form (no import of another
  flashinfer package; a copy under another name imports with flashinfer
  blocked);
* on a device: the default is bitwise the deterministic package;
  ``deterministic=False`` launches only the backend's modules, and on the
  backend suite's accuracy matrix and paths (accumulation,
  ``compute_grad_weight=False``, a reset workspace with nonzero scratch reused
  across shapes, two streams, a side stream, CUDA Graph replays, autograd and
  non-reentrant checkpointing) its ``y``, ``dx`` and ``dz`` are the
  deterministic package's bits and its ``dw`` lies within the weight-gradient
  budget of the deterministic ``dw``, on every repeat.

The numerical contract of the non-deterministic kernels against the FP64
reference is ``test_cake_rmsnorm_silu_gate_train.py``'s; its data generators,
references and budgets are reused here.
"""

from __future__ import annotations

import ast
import functools
import inspect
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

import flashinfer
import flashinfer.cake_rmsnorm_silu_gate_det_train as deterministic_facade
import flashinfer.cake_rmsnorm_silu_gate_train as facade
import flashinfer.cake_rmsnorm_silu_gate_train_backend as backend
from flashinfer.cake_rmsnorm_silu_gate_det_train_backend import (
    cake_jit as deterministic_loader,
)
from flashinfer.cake_rmsnorm_silu_gate_train import (
    CakeRMSNormSiLUGateFunction,
    cake_rmsnorm_silu_gate,
    cake_rmsnorm_silu_gate_train_backward,
    cake_rmsnorm_silu_gate_train_backward_workspace,
    cake_rmsnorm_silu_gate_train_backward_workspace_bytes,
    cake_rmsnorm_silu_gate_train_forward,
    cake_rmsnorm_silu_gate_train_reset_workspace,
    is_cake_rmsnorm_silu_gate_train_supported,
)
from flashinfer.cake_rmsnorm_silu_gate_train_backend import cake_jit as loader
from tests.norm import test_cake_rmsnorm_silu_gate_train as suite

D = 128
EPS = 1e-6
ENTRY_POINTS = (
    cake_rmsnorm_silu_gate_train_forward,
    cake_rmsnorm_silu_gate_train_backward,
    cake_rmsnorm_silu_gate_train_backward_workspace,
    cake_rmsnorm_silu_gate_train_backward_workspace_bytes,
    cake_rmsnorm_silu_gate_train_reset_workspace,
    cake_rmsnorm_silu_gate,
    is_cake_rmsnorm_silu_gate_train_supported,
    backend.cake_rmsnorm_silu_gate_train_forward,
    backend.cake_rmsnorm_silu_gate_train_backward,
    backend.cake_rmsnorm_silu_gate_train_backward_workspace,
    backend.cake_rmsnorm_silu_gate_train_backward_workspace_bytes,
    backend.cake_rmsnorm_silu_gate_train_reset_workspace,
    backend.cake_rmsnorm_silu_gate,
    backend.is_cake_rmsnorm_silu_gate_train_supported,
)
# The deterministic package's entry point each facade call runs for
# ``deterministic=True``.
DETERMINISTIC_ENTRY_POINTS = {
    "forward": "cake_rmsnorm_silu_gate_det_train_forward",
    "backward": "cake_rmsnorm_silu_gate_det_train_backward",
    "workspace": "cake_rmsnorm_silu_gate_det_train_backward_workspace",
    "workspace_bytes": "cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes",
    "autograd": "cake_rmsnorm_silu_gate_det",
}
# The backend entry points the facade calls for ``deterministic=False``.
BACKEND_ENTRY_POINTS = (
    "cake_rmsnorm_silu_gate_train_forward",
    "cake_rmsnorm_silu_gate_train_backward",
    "cake_rmsnorm_silu_gate_train_backward_workspace",
    "cake_rmsnorm_silu_gate_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate",
)
BACKEND_DIR = Path(backend.__file__).parent

nd_forward = functools.partial(
    cake_rmsnorm_silu_gate_train_forward, deterministic=False
)
nd_backward = functools.partial(
    cake_rmsnorm_silu_gate_train_backward, deterministic=False
)
nd_workspace = functools.partial(
    cake_rmsnorm_silu_gate_train_backward_workspace, deterministic=False
)
nd_reset = functools.partial(
    cake_rmsnorm_silu_gate_train_reset_workspace, deterministic=False
)
nd_gated_rmsnorm = functools.partial(cake_rmsnorm_silu_gate, deterministic=False)
det_forward = deterministic_facade.cake_rmsnorm_silu_gate_det_train_forward
det_backward = deterministic_facade.cake_rmsnorm_silu_gate_det_train_backward


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
    # the facade's dispatching autograd function, not the backend's (False only)
    assert CakeRMSNormSiLUGateFunction is not backend.CakeRMSNormSiLUGateFunction
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


# facade entry point -> (deterministic package entry point, backend entry
# point, positional arguments, keyword arguments) of one call.
_X, _Z, _W, _G, _OUT, _WS = (object() for _ in range(6))
DISPATCH = {
    "is_cake_rmsnorm_silu_gate_train_supported": (
        "is_cake_rmsnorm_silu_gate_det_train_supported",
        "is_cake_rmsnorm_silu_gate_train_supported",
        ("cuda:0",),
        {},
    ),
    "cake_rmsnorm_silu_gate_train_forward": (
        "cake_rmsnorm_silu_gate_det_train_forward",
        "cake_rmsnorm_silu_gate_train_forward",
        (_X, _Z, _W, 1e-5),
        {"out": _OUT},
    ),
    "cake_rmsnorm_silu_gate_train_backward": (
        "cake_rmsnorm_silu_gate_det_train_backward",
        "cake_rmsnorm_silu_gate_train_backward",
        (_G, _X, _Z, _W, 1e-5),
        {
            "grad_x": None,
            "grad_z": _OUT,
            "grad_weight": None,
            "accumulate_grad_weight": False,
            "compute_grad_weight": True,
            "workspace": _WS,
        },
    ),
    "cake_rmsnorm_silu_gate_train_backward_workspace": (
        "cake_rmsnorm_silu_gate_det_train_backward_workspace",
        "cake_rmsnorm_silu_gate_train_backward_workspace",
        (77, "cuda:0"),
        {},
    ),
    "cake_rmsnorm_silu_gate_train_backward_workspace_bytes": (
        "cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes",
        "cake_rmsnorm_silu_gate_train_backward_workspace_bytes",
        (77, "cuda:0"),
        {},
    ),
    "cake_rmsnorm_silu_gate": (
        "cake_rmsnorm_silu_gate_det",
        "cake_rmsnorm_silu_gate",
        (_X, _Z, _W, 1e-5),
        {},
    ),
}


@pytest.mark.parametrize("name", sorted(DISPATCH))
def test_dispatch(
    name: str, monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """The default and ``deterministic=True`` call the deterministic package's
    entry point with the same arguments; ``deterministic=False`` calls the
    backend's with ``deterministic=False``; nothing else runs."""
    det_name, nd_name, args, kwargs = DISPATCH[name]
    calls: list = []
    monkeypatch.setattr(
        facade._deterministic,
        det_name,
        lambda *a, **k: calls.append(("det", a, k)) or "det",
    )
    monkeypatch.setattr(
        backend, nd_name, lambda *a, **k: calls.append(("nd", a, k)) or "nd"
    )
    entry = getattr(facade, name)
    assert entry(*args, **kwargs) == "det"
    assert entry(*args, **kwargs, deterministic=True) == "det"
    assert entry(*args, **kwargs, deterministic=False) == "nd"
    assert calls == [
        ("det", args, kwargs),
        ("det", args, kwargs),
        ("nd", args, {**kwargs, "deterministic": False}),
    ]


@pytest.mark.parametrize(
    "mode", [None, 0, 1, "False"], ids=["none", "zero", "one", "string"]
)
def test_a_mode_that_is_not_a_bool_is_a_type_error(
    mode, monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """``deterministic`` must be a ``bool``: ``None``, ``0``, ``1`` or a string
    selects neither package and raises ``TypeError`` before any call, whatever
    ``torch.use_deterministic_algorithms`` says (stubbed: no device needed)."""
    calls: list = []
    for det_name, nd_name, _, _ in DISPATCH.values():
        monkeypatch.setattr(
            facade._deterministic, det_name, lambda *a, **k: calls.append(a)
        )
        monkeypatch.setattr(backend, nd_name, lambda *a, **k: calls.append(a))
    monkeypatch.setattr(
        backend,
        "cake_rmsnorm_silu_gate_train_reset_workspace",
        lambda *a, **k: calls.append(a),
    )
    x, w = _cpu_tensors()
    runs = [
        *(
            functools.partial(getattr(facade, name), *args, **kwargs)
            for name, (_, _, args, kwargs) in DISPATCH.items()
        ),
        functools.partial(
            cake_rmsnorm_silu_gate_train_reset_workspace,
            torch.zeros(8, dtype=torch.float64),
        ),
        lambda deterministic: CakeRMSNormSiLUGateFunction.apply(
            x, x, w, EPS, deterministic
        ),
    ]
    for enabled, warn_only in ((False, False), (True, False), (True, True)):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        for run in runs:
            with pytest.raises(TypeError, match="deterministic must be a bool"):
                run(deterministic=mode)
    assert calls == []


def test_autograd_function_dispatches_both_directions(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """:class:`CakeRMSNormSiLUGateFunction` (also
    ``flashinfer.CakeRMSNormSiLUGateFunction``): the default and ``True`` run
    the det package's forward and backward with the arguments
    ``CakeRMSNormSiLUGateDetFunction`` passes; ``False`` runs the backend's
    with ``deterministic=False``.  Under
    ``torch.use_deterministic_algorithms(True)`` ``False`` is refused before
    any call; with ``warn_only=True`` it warns and runs the det package in both
    directions.  Kernels stubbed: no device needed."""
    calls: list = []

    def forward(tag):
        return lambda x, z, w, eps, **k: calls.append((tag, "fwd", k)) or x * 1

    def backward(tag):
        return lambda dy, x, z, w, eps, **k: (
            calls.append((tag, "bwd", k)) or (dy * 1, dy * 1, w * 1)
        )

    for module, prefix, tag in (
        (facade._deterministic, "cake_rmsnorm_silu_gate_det_train", "det"),
        (backend, "cake_rmsnorm_silu_gate_train", "nd"),
    ):
        monkeypatch.setattr(module, f"{prefix}_forward", forward(tag))
        monkeypatch.setattr(module, f"{prefix}_backward", backward(tag))

    def run(*flag):
        x, w = _cpu_tensors()
        leaves = [t.clone().requires_grad_(True) for t in (x, x, w)]
        y = CakeRMSNormSiLUGateFunction.apply(*leaves, EPS, *flag)
        assert y.grad_fn._forward_cls is CakeRMSNormSiLUGateFunction
        y.backward(torch.ones_like(y))

    det = [("det", "fwd", {}), ("det", "bwd", {"compute_grad_weight": True})]
    nd = [
        ("nd", "fwd", {"deterministic": False}),
        ("nd", "bwd", {"compute_grad_weight": True, "deterministic": False}),
    ]
    run()
    run(True)
    run(False)
    assert calls == det + det + nd
    calls.clear()
    torch.use_deterministic_algorithms(True)
    with pytest.raises(RuntimeError, match="deterministic=True"):
        run(False)
    assert calls == []
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning, match="running the deterministic kernels"):
        run(False)
    assert calls == det


@pytest.mark.parametrize(
    "call", ["forward", "backward", "workspace", "workspace_bytes", "autograd"]
)
def test_nondeterministic_request_is_refused_under_deterministic_algorithms(
    call: str, monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """``deterministic=False`` while ``torch.use_deterministic_algorithms(True)``
    raises before any kernel or device query; with ``warn_only=True`` it warns
    and runs the det package (stubbed here: no device needed).  The backend is
    never called."""
    x, w = _cpu_tensors()
    calls: list[str] = []
    for name in DETERMINISTIC_ENTRY_POINTS.values():
        monkeypatch.setattr(
            facade._deterministic,
            name,
            lambda *a, _name=name, **k: calls.append(_name) or x.clone(),
        )
    for name in BACKEND_ENTRY_POINTS:
        monkeypatch.setattr(
            backend, name, lambda *a, _name=name, **k: calls.append(_name)
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
    assert calls == [DETERMINISTIC_ENTRY_POINTS[call]]


def test_deterministic_request_never_warns(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    x, w = _cpu_tensors()
    monkeypatch.setattr(
        facade._deterministic,
        "cake_rmsnorm_silu_gate_det_train_forward",
        lambda *a, **k: x.clone(),
    )
    torch.use_deterministic_algorithms(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cake_rmsnorm_silu_gate_train_forward(x, x, w, EPS)
        cake_rmsnorm_silu_gate_train_forward(x, x, w, EPS, deterministic=True)


def test_autograd_backward_alerts_once_deterministic_algorithms_are_enabled(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """A ``deterministic=False`` node is the backend's
    :class:`CakeRMSNormSiLUGateFunction`.  When
    ``torch.use_deterministic_algorithms(True)`` is enabled between its forward
    and its backward, the backward raises PyTorch's alert before any launch;
    under ``warn_only=True`` it warns and runs the non-deterministic backward
    (the deterministic package is not called).  A node whose weight needs no
    gradient has no unordered sum and does not alert.  Kernels stubbed: no
    device needed."""
    calls: list[str] = []
    monkeypatch.setattr(
        backend, "_forward", lambda x, z, w, eps, *, out: calls.append("fwd") or x * 1
    )
    monkeypatch.setattr(
        backend, "_backward", lambda *a, **k: calls.append("bwd") or (None,) * 3
    )
    for name in DETERMINISTIC_ENTRY_POINTS.values():
        monkeypatch.setattr(
            facade._deterministic, name, lambda *a, _name=name, **k: calls.append(_name)
        )

    def node(weight_grad=True):
        x, w = _cpu_tensors()
        y = cake_rmsnorm_silu_gate(
            x.clone().requires_grad_(True),
            x.clone().requires_grad_(True),
            w.clone().requires_grad_(weight_grad),
            EPS,
            deterministic=False,
        )
        assert y.grad_fn._forward_cls is backend.CakeRMSNormSiLUGateFunction
        return y

    y = node()
    torch.use_deterministic_algorithms(True)
    with pytest.raises(
        RuntimeError, match="does not have a deterministic implementation"
    ):
        y.backward(torch.ones_like(y))
    assert calls == ["fwd"]
    torch.use_deterministic_algorithms(False)
    y = node()
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(
        UserWarning, match="does not have a deterministic implementation"
    ):
        y.backward(torch.ones_like(y))
    assert calls == ["fwd", "fwd", "bwd"]
    torch.use_deterministic_algorithms(False)
    y = node(weight_grad=False)
    torch.use_deterministic_algorithms(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y.backward(torch.ones_like(y))
    assert calls == ["fwd", "fwd", "bwd", "fwd", "bwd"]


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
        facade._deterministic,
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


def test_reset_workspace_of_both_modes(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """``deterministic=False`` resets a non-deterministic workspace through the
    backend; ``True`` (the default) accepts the deterministic package's FP32
    buffer, which holds no state between calls, and launches nothing, while an
    FP64 workspace is refused rather than silently skipped.  Neither form
    depends on ``torch.use_deterministic_algorithms``."""
    resets: list = []
    monkeypatch.setattr(
        backend,
        "cake_rmsnorm_silu_gate_train_reset_workspace",
        lambda workspace, **k: resets.append((workspace, k)),
    )
    fp32 = torch.ones(8, dtype=torch.float32)
    fp64 = torch.ones(8, dtype=torch.float64)
    for enabled, warn_only in ((False, False), (True, False), (True, True)):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert cake_rmsnorm_silu_gate_train_reset_workspace(fp32) is None
            assert (
                cake_rmsnorm_silu_gate_train_reset_workspace(fp32, deterministic=True)
                is None
            )
            assert torch.equal(fp32, torch.ones_like(fp32))
            with pytest.raises(ValueError, match="deterministic=False"):
                cake_rmsnorm_silu_gate_train_reset_workspace(fp64)
            with pytest.raises(TypeError):
                cake_rmsnorm_silu_gate_train_reset_workspace(None)
            cake_rmsnorm_silu_gate_train_reset_workspace(fp64, deterministic=False)
    assert len(resets) == 3
    for workspace, kwargs in resets:
        assert workspace is fp64 and kwargs == {"deterministic": False}


def test_backend_package_imports_no_other_flashinfer_package() -> None:
    """The backend package is standalone: no source file imports from a parent
    package or another flashinfer module, and every relative import names a
    module of the package.  The one exception is the shared loader
    ``cake_build.py``, whose JIT fallback imports ``flashinfer.jit`` inside a
    function when no prebuilt library exists."""
    sources = sorted(BACKEND_DIR.glob("*.py"))
    assert {p.name for p in sources} == {"__init__.py", "cake_build.py", "cake_jit.py"}
    jit_fallback = []
    for path in sources:
        text = path.read_text()
        assert "import_module(" not in text and "__import__(" not in text, path.name
        tree = ast.parse(text)
        in_function = {
            id(node)
            for function in ast.walk(tree)
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(function)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                assert node.level == 1, (path.name, node.lineno, node.level)
                names = (
                    [node.module.split(".")[0]]
                    if node.module
                    else [alias.name for alias in node.names]
                )
                for name in names:
                    assert (BACKEND_DIR / f"{name}.py").is_file(), (path.name, name)
                continue
            if isinstance(node, ast.ImportFrom):
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                if module == "flashinfer" or module.startswith("flashinfer."):
                    assert path.name == "cake_build.py" and id(node) in in_function, (
                        path.name,
                        node.lineno,
                        module,
                    )
                    assert module.startswith("flashinfer.jit."), (path.name, module)
                    jit_fallback.append(module)
    assert jit_fallback, "the cake_build JIT fallback import was not found"


_VENDORED_CHILD = r"""
import importlib, importlib.abc, json, sys


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "flashinfer" or name.startswith("flashinfer."):
            raise ImportError("flashinfer is blocked: " + name)
        return None


sys.meta_path.insert(0, Block())
sys.modules["flashinfer"] = None
sys.path.insert(0, sys.argv[1])
import torch

package = importlib.import_module(sys.argv[2])
importlib.import_module(sys.argv[2] + ".cake_jit")
importlib.import_module(sys.argv[2] + ".cake_build")
x = torch.zeros(2, 128, dtype=torch.bfloat16)
out = {
    "supported_cpu": package.is_cake_rmsnorm_silu_gate_train_supported(
        torch.device("cpu"), deterministic=False
    ),
    "all": sorted(package.__all__),
}
try:
    package.cake_rmsnorm_silu_gate_train_forward(x, x, x[0])
except ValueError as error:
    out["refusal"] = str(error)
try:
    package.cake_rmsnorm_silu_gate_train_forward(x, x, x[0], deterministic=False)
except ValueError as error:
    out["cpu"] = str(error)
out["flashinfer"] = sorted(
    name for name, module in sys.modules.items()
    if module is not None and (name == "flashinfer" or name.startswith("flashinfer."))
)
print(json.dumps(out))
"""


def test_vendored_copy_imports_with_flashinfer_blocked(tmp_path: Path) -> None:
    """The backend package copied under another top-level name (the vendored
    gate's first step, without its sources) imports, answers its support query
    and refuses ``deterministic=True`` in an interpreter where every
    ``flashinfer`` import raises; no flashinfer module is loaded."""
    neutral = "vendored_cake_rmsnorm_silu_gate_train_backend"
    shutil.copytree(
        BACKEND_DIR,
        tmp_path / neutral,
        ignore=shutil.ignore_patterns("aot", "__pycache__", "csrc", "build.json"),
    )
    run = subprocess.run(
        [sys.executable, "-c", _VENDORED_CHILD, str(tmp_path), neutral],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-8:])
    assert run.returncode == 0, tail
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["flashinfer"] == [], out["flashinfer"]
    assert out["supported_cpu"] is False
    assert out["all"] == sorted(backend.__all__)
    assert out["refusal"].startswith(
        "cake_rmsnorm_silu_gate_train_forward computes non-deterministically and "
        "takes deterministic=False"
    ), out["refusal"]
    assert "CUDA" in out["cpu"], out["cpu"]


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
    # the deterministic buffer needs no reset: the default reset launches nothing
    assert cake_rmsnorm_silu_gate_train_reset_workspace(ws) is None
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
    # autograd (the det package's node) and non-reentrant checkpointing
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    xb, zb, wb = (t.clone().requires_grad_(True) for t in (x, z, w))
    ya = cake_rmsnorm_silu_gate(xa, za, wa, EPS)
    assert (
        ya.grad_fn._forward_cls is deterministic_facade.CakeRMSNormSiLUGateDetFunction
    )
    yb = deterministic_facade.cake_rmsnorm_silu_gate_det(xb, zb, wb, EPS)
    assert torch.equal(ya, yb)
    ya.backward(dy)
    yb.backward(dy)
    for a, b in ((xa, xb), (za, zb), (wa, wb)):
        assert torch.equal(a.grad, b.grad)
    # the autograd function's default runs the det package in both directions
    xd, zd, wd = (t.clone().requires_grad_(True) for t in (x, z, w))
    yd = CakeRMSNormSiLUGateFunction.apply(xd, zd, wd, EPS)
    yd.backward(dy)
    assert torch.equal(yd, yb)
    for a, b in ((xd, xb), (zd, zb), (wd, wb)):
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
    """``deterministic=False`` launches the backend's forward and its one
    backward kernel (and no det module): ``y``, ``dx`` and ``dz`` are the det
    package's bits, ``dw`` within one BF16 rounding plus the FP32 accumulation
    budget of the det ``dw``.  The two workspace kinds do not substitute."""
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
    for name in launched:
        assert name not in deterministic_loader.MODULES, name
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
    # the deterministic package's FP32 partial buffer is not a workspace of the
    # non-deterministic backward
    deterministic_ws = cake_rmsnorm_silu_gate_train_backward_workspace(
        257 * 48, x.device
    )
    with pytest.raises(ValueError, match="FP64"):
        cake_rmsnorm_silu_gate_train_backward(
            dy, x, z, w, EPS, workspace=deterministic_ws, deterministic=False
        )


# --------------------------------------------------------------------------- against the deterministic package

# The backend suite's accuracy matrix: (T, H), the generator's keywords, eps and
# the weight-gradient dtype.
MATRIX = [
    *(
        pytest.param((T, H), {}, EPS, torch.bfloat16, id=f"shape-{T}x{H}")
        for T, H in suite.SHAPES
    ),
    *(
        pytest.param((65, 48), {"layout": v, "seed": 1}, EPS, torch.bfloat16, id=v)
        for v in suite.LAYOUTS
    ),
    *(
        pytest.param((37, 48), {"data": v, "seed": 2}, EPS, torch.bfloat16, id=v)
        for v in suite.EDGE_DATA
    ),
    *(
        pytest.param((19, 48), {"seed": 3}, eps, torch.bfloat16, id=f"eps-{eps}")
        for eps in suite.EPSILONS
    ),
    *(
        pytest.param((T, 48), {"seed": 4}, EPS, torch.bfloat16, id=f"issue-{T}")
        for T in suite.ISSUE_TOKENS
    ),
    *(
        pytest.param((T, H), {"seed": 5}, EPS, torch.float32, id=f"fp32-{T}x{H}")
        for T, H in suite.FP32_SHAPES
    ),
]


@requires_device
@pytest.mark.parametrize("shape,make_kwargs,eps,dw_dtype", MATRIX)
def test_nondeterministic_matches_the_deterministic_package(
    shape, make_kwargs, eps, dw_dtype, determinism_setting
) -> None:
    """``y`` and, on every one of ``suite.REPEATS`` backward launches, ``dx``
    and ``dz`` are the deterministic package's bits; ``dw`` lies within the
    weight-gradient budget of the deterministic ``dw``."""
    x, z, w, dy = suite.make(*shape, **make_kwargs)
    y = nd_forward(x, z, w, eps)
    assert torch.equal(y, det_forward(x, z, w, eps)), "y vs the deterministic package"
    mag = suite.magnitudes(dy, x, z, eps)

    def out():
        return (
            None
            if dw_dtype == torch.bfloat16
            else torch.empty(D, dtype=dw_dtype, device=x.device)
        )

    dx_d, dz_d, dw_d = det_backward(dy, x, z, w, eps, grad_weight=out())
    for _ in range(suite.REPEATS):
        dx, dz, dw = nd_backward(dy, x, z, w, eps, grad_weight=out())
        assert dw.dtype == dw_dtype
        assert torch.equal(dx, dx_d), "dx vs the deterministic package"
        assert torch.equal(dz, dz_d), "dz vs the deterministic package"
        suite.assert_dw_within(dw, dw_d, mag, what="dw vs the deterministic package")


@requires_device
def test_accumulation_matches_the_deterministic_package(determinism_setting) -> None:
    x, z, w, dy = suite.make(1031, 48, seed=6)
    x2, z2, _, dy2 = suite.make(1031, 48, seed=7)
    dw64 = suite.fp64_reference(dy, x, z, w, EPS)[3]
    dw64b = suite.fp64_reference(dy2, x2, z2, w, EPS)[3]
    mag = suite.magnitudes(dy, x, z, EPS) + suite.magnitudes(dy2, x2, z2, EPS)
    first = 0.25 + dw64
    want = first + dw64b
    # Two roundings into the accumulator's dtype, each within its relative unit.
    bound = (first.abs() + want.abs()) * suite.BF16_RTOL + mag * suite.DW_MAG_TOL
    for dtype in (torch.float32, torch.bfloat16):
        main = torch.full((D,), 0.25, dtype=dtype, device="cuda")
        main_reference = main.clone()
        for args in ((dy, x, z), (dy2, x2, z2)):
            nd_backward(*args, w, EPS, grad_weight=main, accumulate_grad_weight=True)
            det_backward(
                *args, w, EPS, grad_weight=main_reference, accumulate_grad_weight=True
            )
        assert ((main.double() - main_reference.double()).abs() <= 2 * bound).all(), (
            dtype
        )
    # autograd accumulation into .grad over two microbatches
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    nd_gated_rmsnorm(xa, za, wa, EPS).backward(dy)
    nd_gated_rmsnorm(xa, za, wa, EPS).backward(dy)
    dx_d, dz_d, _ = det_backward(dy, x, z, w, EPS)
    assert torch.equal(xa.grad, dx_d + dx_d)
    assert torch.equal(za.grad, dz_d + dz_d)


@requires_device
def test_compute_grad_weight_false_matches_the_deterministic_package(
    determinism_setting,
) -> None:
    x, z, w, dy = suite.make(257, 48, seed=12)
    dx_d, dz_d, _ = det_backward(dy, x, z, w, EPS, compute_grad_weight=False)
    ws = nd_workspace(257 * 48, x.device)
    for workspace in (ws, None):
        dx, dz, dw = nd_backward(
            dy, x, z, w, EPS, compute_grad_weight=False, workspace=workspace
        )
        assert dw is None
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
    # autograd with a weight that needs no gradient
    xa, za = (t.clone().requires_grad_(True) for t in (x, z))
    nd_gated_rmsnorm(xa, za, w, EPS).backward(dy)
    assert torch.equal(xa.grad, dx_d) and torch.equal(za.grad, dz_d)


@requires_device
def test_reset_workspace_then_reuse_across_shapes(determinism_setting) -> None:
    """A workspace with nonzero scratch (garbage FP64 accumulators, arrival
    ticket and claim counter) zeroed by
    ``cake_rmsnorm_silu_gate_train_reset_workspace(..., deterministic=False)``,
    then reused without a reset over the backend suite's sequence of shapes,
    weight-gradient dtypes and accumulation: ``dx`` and ``dz`` are the
    deterministic package's bits, ``dw`` lies within the budget of the
    deterministic ``dw`` and of the FP64 reference, and the workspace is zero
    after every call."""
    device = torch.device("cuda")
    ws = nd_workspace(1, device)
    g = torch.Generator(device="cuda").manual_seed(70)
    ws.copy_(
        torch.randn(ws.shape, dtype=torch.float64, device=device, generator=g) * 1e3
    )
    # a stale count in the low 32 bits of every element, the arrival ticket
    # included; the high 32 bits, the claim counter's among them, keep the
    # random values' nonzero bits
    ws.view(torch.int32)[0::2] = 3
    assert bool(ws.ne(0).all())
    nd_reset(ws)
    torch.cuda.synchronize()
    assert not bool(ws.any())
    cases = (
        (1031, 48, torch.bfloat16, False),
        (7, 48, torch.float32, False),
        (4097, 16, torch.bfloat16, False),
        (1, 1, torch.float32, True),
        (30193, 48, torch.bfloat16, False),
        (33, 7, torch.float32, True),
    )
    for i, (T, H, dtype, accumulate) in enumerate(cases):
        x, z, w, dy = suite.make(T, H, seed=20 + i)
        start = 0.5 if accumulate else 0.0

        def gw(start=start, dtype=dtype, accumulate=accumulate):
            if accumulate or dtype != torch.bfloat16:
                return torch.full((D,), start, dtype=dtype, device=device)
            return None

        dx, dz, dw = nd_backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=gw(),
            accumulate_grad_weight=accumulate,
            workspace=ws,
        )
        dx_d, dz_d, dw_d = det_backward(
            dy, x, z, w, EPS, grad_weight=gw(), accumulate_grad_weight=accumulate
        )
        torch.cuda.synchronize()
        assert not bool(ws.any()), (T, H, dtype)
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), (T, H)
        assert dw.dtype == dtype
        mag = suite.magnitudes(dy, x, z, EPS)
        want = suite.fp64_reference(dy, x, z, w, EPS)[3] + start
        suite.assert_dw_within(dw, want, mag, what=f"dw T={T} H={H} {dtype} vs FP64")
        suite.assert_dw_within(
            dw, dw_d, mag, what=f"dw T={T} H={H} {dtype} vs the deterministic package"
        )


@requires_device
def test_two_streams_match_the_deterministic_package(determinism_setting) -> None:
    cases = [suite.make(4097, 48, seed=30), suite.make(1031, 16, seed=31)]
    refs = [det_backward(dy, x, z, w, EPS) for x, z, w, dy in cases]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    workspaces = [nd_workspace(1, torch.device("cuda")) for _ in streams]
    outputs: list[list] = [[], []]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for _ in range(5):
        for i, stream in enumerate(streams):
            x, z, w, dy = cases[i]
            with torch.cuda.stream(stream):
                outputs[i].append(
                    nd_backward(dy, x, z, w, EPS, workspace=workspaces[i])
                )
    torch.cuda.synchronize()
    for i in range(2):
        dx_d, dz_d, _ = refs[i]
        for dx, dz, _ in outputs[i]:
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), i


@requires_device
def test_side_stream_matches_the_deterministic_package(determinism_setting) -> None:
    x, z, w, dy = suite.make(2049, 48, seed=9)
    want_y = det_forward(x, z, w, EPS)
    dx_d, dz_d, _ = det_backward(dy, x, z, w, EPS)
    ws = nd_workspace(x.shape[0] * x.shape[1], x.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = nd_forward(x, z, w, EPS)
        got = nd_backward(dy, x, z, w, EPS, workspace=ws)
        fresh = nd_backward(dy, x, z, w, EPS)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert torch.equal(y, want_y)
    for dx, dz, _ in (got, fresh):
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)


@requires_device
def test_cuda_graph_replay_matches_the_deterministic_package(
    determinism_setting,
) -> None:
    T, H = 300, 48
    x, z, w, dy = suite.make(T, H, seed=10)
    xs, zs, dys = x.clone(), z.clone(), dy.clone()
    y = torch.empty_like(x)
    dx, dz = torch.empty_like(x), torch.empty_like(x)
    dw = torch.empty(D, dtype=torch.bfloat16, device="cuda")
    ws = nd_workspace(T * H, x.device)

    def step():
        nd_forward(xs, zs, w, EPS, out=y)
        nd_backward(
            dys, xs, zs, w, EPS, grad_x=dx, grad_z=dz, grad_weight=dw, workspace=ws
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # loads the modules before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for k in range(3):
        xn, zn, _, dyn = suite.make(T, H, seed=40 + k)
        xs.copy_(xn)
        zs.copy_(zn)
        dys.copy_(dyn)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, det_forward(xn, zn, w, EPS)), k
        dx_d, dz_d, _ = det_backward(dyn, xn, zn, w, EPS)
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), k


@requires_device
def test_autograd_and_checkpoint_match_the_deterministic_package(
    determinism_setting,
) -> None:
    x, z, w, dy = suite.make(513, 48, seed=8)
    y_ref = det_forward(x, z, w, EPS)
    dx_d, dz_d, dw_d = det_backward(dy, x, z, w, EPS)
    mag = suite.magnitudes(dy, x, z, EPS)

    def check(y, xg, zg, wg, what):
        assert torch.equal(y, y_ref), what
        assert torch.equal(xg, dx_d) and torch.equal(zg, dz_d), what
        assert wg.dtype == torch.bfloat16, what
        suite.assert_dw_within(
            wg, dw_d, mag, what=f"{what} dw vs the deterministic package"
        )

    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    y = nd_gated_rmsnorm(xa, za, wa, EPS)
    assert y.grad_fn._forward_cls is backend.CakeRMSNormSiLUGateFunction
    y.backward(dy)
    check(y, xa.grad, za.grad, wa.grad, "autograd")
    xf, zf, wf = (t.clone().requires_grad_(True) for t in (x, z, w))
    yf = CakeRMSNormSiLUGateFunction.apply(xf, zf, wf, EPS, False)
    yf.backward(dy)
    check(yf, xf.grad, zf.grad, wf.grad, "autograd function")
    xc, zc, wc = (t.clone().requires_grad_(True) for t in (x, z, w))
    yc = torch.utils.checkpoint.checkpoint(
        nd_gated_rmsnorm, xc, zc, wc, EPS, use_reentrant=False
    )
    yc.backward(dy)
    check(yc, xc.grad, zc.grad, wc.grad, "checkpoint")


# --------------------------------------------------------------------------- the long modules' thresholds


def _spy_launches(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    launched: list[str] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        launched.append(name)
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    return launched


def _sm_count() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


# Forward inputs around and past 2400 rows per SM, where sm_100a and sm_103a
# switch to the long forward: (tokens, heads, layout); ``heads=None`` takes
# ``2400 * SMs + tokens`` rows as one flat [rows, 128] tensor.
LONG_FORWARD_CASES = [
    pytest.param(-1, None, "flat", id="threshold-1"),
    pytest.param(1, None, "flat", id="threshold+1"),
    pytest.param(8192, 48, "contiguous", id="8192x48"),
    pytest.param(8192, 48, "projection", id="8192x48-projection"),
    pytest.param(8192, 48, "head_major", id="8192x48-head_major"),
    pytest.param(32737, 48, "contiguous", id="32737x48"),
    pytest.param(32737, 32, "contiguous", id="32737x32"),
]


@requires_device
@pytest.mark.parametrize("tokens,heads,layout", LONG_FORWARD_CASES)
def test_forward_around_the_long_threshold_matches_the_deterministic_package(
    tokens, heads, layout, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``y`` is the deterministic package's bits on both sides of the long
    forward's threshold and past it, and the forward launches
    ``module_for_rows``' module: the long forward from ``2400 * SMs`` rows
    where the architecture routes one, else the forward."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    if heads is None:
        x, z, w, _ = suite.make(2400 * sms + tokens, 1, layout=layout, seed=21)
    else:
        x, z, w, _ = suite.make(tokens, heads, layout=layout, seed=21)
    rows = x.numel() // D
    launched = _spy_launches(monkeypatch)
    y = nd_forward(x, z, w, EPS)
    name = loader.module_for_rows(arch, "fwd", rows, sms)
    assert launched == [name], launched
    long = loader.ROUTES[arch].get("fwd_long")
    assert name == (
        long
        if long is not None and rows >= 2400 * sms
        else loader.route_module(arch, "fwd")
    )
    assert torch.equal(y, det_forward(x, z, w, EPS)), "y vs the deterministic package"


@requires_device
@pytest.mark.parametrize("offset", [-1, 1], ids=["threshold-1", "threshold+1"])
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_backward_around_the_long_threshold_matches_the_deterministic_package(
    offset, dw_dtype, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On an architecture that routes a long backward (sm_107a: from 1024 rows
    per SM), one row below and above the threshold: the module launched is
    ``module_for_rows``', ``dx`` and ``dz`` are the deterministic package's
    bits and ``dw`` lies within the budget of the deterministic ``dw`` on
    every repeat, and the workspace is zero after every call."""
    arch = loader.device_arch(torch.cuda.current_device())
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    long = loader.ROUTES[arch].get(f"{role}_long")
    if long is None:
        pytest.skip(f"{arch} routes no long backward")
    sms = _sm_count()
    rows = 1024 * sms + offset
    x, z, w, dy = suite.make(rows, 1, layout="flat", seed=22)
    mag = suite.magnitudes(dy, x, z, EPS)

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x.device)

    dx_d, dz_d, dw_d = det_backward(dy, x, z, w, EPS, grad_weight=out())
    ws = nd_workspace(rows, x.device)
    launched = _spy_launches(monkeypatch)
    for _ in range(suite.REPEATS):
        dx, dz, dw = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
        assert dw.dtype == dw_dtype
        assert torch.equal(dx, dx_d), "dx vs the deterministic package"
        assert torch.equal(dz, dz_d), "dz vs the deterministic package"
        suite.assert_dw_within(dw, dw_d, mag, what="dw vs the deterministic package")
        assert not bool(ws.any()), "the workspace is not zero"
    name = long if offset > 0 else loader.route_module(arch, role)
    assert launched == [name] * suite.REPEATS, launched
