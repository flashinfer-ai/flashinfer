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
  both modes, the backend's standalone form (no import of another
  flashinfer package; a copy under another name imports with flashinfer
  blocked), and the strides the backend passes for one-head and other inputs
  (a single head's stride as 128, so packed one-head rows pass the kernels'
  packed-rows test);
* on a device: the default is bitwise the deterministic package;
  ``deterministic=False`` launches only the backend's modules, and on the
  backend suite's accuracy matrix and paths (accumulation,
  ``compute_grad_weight=False``, a reset workspace with nonzero scratch reused
  across shapes, two streams, a side stream, CUDA Graph replays, autograd and
  non-reentrant checkpointing) its ``y``, ``dx`` and ``dz`` are the
  deterministic package's bits and its ``dw`` lies within the weight-gradient
  budget of the deterministic ``dw``, on every repeat; so are they for
  one-head inputs of several layouts past the start of each module that warms
  packed rows in L2, which launch with the packed-rows test passing exactly
  when their rows follow one another; and for weight gradients in the
  subnormal range (identical rows whose FP32 terms underflow) on the short,
  the mid and the long backward: ``dx`` and ``dz`` the deterministic bits, a
  BF16 ``dw`` the deterministic ``dw``'s bits, and every ``dw`` within the
  budget widened by the underflow allowance the package documents.

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
    arch = loader.device_arch(torch.cuda.current_device())
    routes = loader.ROUTES[arch]
    fwd = loader.module_for_rows(arch, "fwd", 257 * 48, _sm_count())
    assert launched == [fwd, routes["bwd_bf16"]], launched
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


# --------------------------------------------------------------------------- the short, long and mid modules' thresholds


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


# The rows per SM up to which each architecture runs the short forward, and
# from which each architecture runs the mid and the long forward.
SHORT_FORWARD_ROWS_PER_SM = {"sm_100a": 12, "sm_103a": 12, "sm_107a": 59}
MID_FORWARD_ROWS_PER_SM = {"sm_100a": 976, "sm_103a": 976, "sm_107a": 800}
LONG_FORWARD_ROWS_PER_SM = {"sm_100a": 2400, "sm_103a": 2400, "sm_107a": 1740}
# The rows per SM from which an architecture runs the extra-long forward.
XLONG_FORWARD_ROWS_PER_SM = {"sm_103a": 3000, "sm_107a": 1740}
# Forward inputs around and past the short, the mid and the long forward's
# thresholds: (tokens, heads, layout); ``heads=None`` takes ``threshold * SMs
# + tokens`` rows (the short threshold for ``flat-short``, the mid one for
# ``flat-mid``, the long one for ``flat``) as one flat [rows, 128] tensor.
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


@requires_device
@pytest.mark.parametrize("tokens,heads,layout", LONG_FORWARD_CASES)
def test_forward_around_the_long_threshold_matches_the_deterministic_package(
    tokens, heads, layout, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``y`` is the deterministic package's bits on both sides of the mid and
    the long forward's thresholds and past them, and the forward launches
    ``module_for_rows``' module: where the architecture routes them, the short
    forward up to ``SHORT_FORWARD_ROWS_PER_SM * SMs`` rows, the long forward
    from ``LONG_FORWARD_ROWS_PER_SM * SMs`` rows and the mid forward from
    ``MID_FORWARD_ROWS_PER_SM * SMs`` rows below that, else the forward."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    short_end = SHORT_FORWARD_ROWS_PER_SM[arch]
    mid_start = MID_FORWARD_ROWS_PER_SM[arch]
    long_start = LONG_FORWARD_ROWS_PER_SM[arch]
    xlong_start = XLONG_FORWARD_ROWS_PER_SM.get(arch)
    if layout == "flat-xlong" and xlong_start is None:
        pytest.skip(f"{arch} routes no fwd_xlong module")
    if heads is None:
        start = {
            "flat-short": short_end,
            "flat-mid": mid_start,
            "flat-xlong": xlong_start,
        }.get(layout, long_start)
        x, z, w, _ = suite.make(start * sms + tokens, 1, layout="flat", seed=21)
    else:
        x, z, w, _ = suite.make(tokens, heads, layout=layout, seed=21)
    rows = x.numel() // D
    launched = _spy_launches(monkeypatch)
    y = nd_forward(x, z, w, EPS)
    name = loader.module_for_rows(arch, "fwd", rows, sms)
    assert launched == [name], launched
    routes = loader.ROUTES[arch]
    want = loader.route_module(arch, "fwd")
    if "fwd_short" in routes and rows <= short_end * sms:
        want = routes["fwd_short"]
    elif "fwd_xlong" in routes and rows >= xlong_start * sms:
        want = routes["fwd_xlong"]
    elif "fwd_long" in routes and rows >= long_start * sms:
        want = routes["fwd_long"]
    elif "fwd_mid" in routes and rows >= mid_start * sms:
        want = routes["fwd_mid"]
    assert name == want, (arch, rows, sms)
    assert torch.equal(y, det_forward(x, z, w, EPS)), "y vs the deterministic package"


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
# Forward inputs the short forward runs where the architecture routes it, and
# those at its threshold on sm_100a and sm_103a, 12 rows per SM, on 148 SMs
# (37x48, 253x7, 55x32) and on 152 SMs (38x48, 260x7, 57x32), and on sm_107a,
# 59 rows per SM, on 212 SMs (260x48, 1786x7, 390x32) and one token past it
# (261x48, 1787x7, 391x32).
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


@requires_device
@pytest.mark.parametrize("tokens,heads", GATE_EDGE_CASES)
def test_forward_gate_edges_match_the_deterministic_package(
    tokens, heads, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With gate inputs at and past the packed gate's range in two of every
    three rows (a quarter of their elements, so the range check fails for some
    threads of a row and holds for others) and ordinary ones in the rest,
    ``y`` is the deterministic package's bits, NaN included, and the forward
    launches ``module_for_rows``' module."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    x, z, w, _ = suite.make(tokens, heads, seed=23)
    g = torch.Generator(device=z.device).manual_seed(24)
    values = torch.tensor(GATE_EDGE_VALUES, device=z.device).bfloat16()
    pick = torch.randint(
        0, len(GATE_EDGE_VALUES), z.shape, device=z.device, generator=g
    )
    edge = torch.rand(z.shape, device=z.device, generator=g) < 0.25
    rows = torch.arange(tokens * heads, device=z.device).view(tokens, heads, 1)
    edge &= rows % 3 != 0
    z.copy_(torch.where(edge, values[pick], z))
    launched = _spy_launches(monkeypatch)
    y = nd_forward(x, z, w, EPS)
    assert launched == [loader.module_for_rows(arch, "fwd", tokens * heads, sms)]
    want = det_forward(x, z, w, EPS)
    assert torch.equal(y.view(torch.int16), want.view(torch.int16)), (
        "y vs the deterministic package"
    )


# The rows per SM from which each architecture runs the long backward.
LONG_BACKWARD_ROWS_PER_SM = {"sm_100a": 1280, "sm_103a": 12000, "sm_107a": 1024}
# Backward inputs around and past the long backward's threshold: (tokens,
# heads, layout); ``heads=None`` takes ``threshold * SMs + tokens`` rows as one
# flat [rows, 128] tensor.
LONG_BACKWARD_CASES = [
    pytest.param(-1, None, "flat", id="threshold-1"),
    pytest.param(1, None, "flat", id="threshold+1"),
    pytest.param(1031, 48, "contiguous", id="1031x48"),
    pytest.param(2048, 48, "contiguous", id="2048x48"),
    pytest.param(4097, 48, "projection", id="4097x48-projection"),
    pytest.param(4097, 48, "head_major", id="4097x48-head_major"),
    pytest.param(32737, 48, "contiguous", id="32737x48"),
]


@requires_device
@pytest.mark.parametrize("tokens,heads,layout", LONG_BACKWARD_CASES)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_backward_around_the_long_threshold_matches_the_deterministic_package(
    tokens,
    heads,
    layout,
    dw_dtype,
    determinism_setting,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Around and past the long backward's threshold
    (``LONG_BACKWARD_ROWS_PER_SM``): the module launched is
    ``module_for_rows``' (the long backward from the threshold, the backward
    below it), ``dx`` and ``dz`` are the deterministic package's bits and
    ``dw`` lies within the budget of the deterministic ``dw`` on every repeat,
    and the workspace is zero after every call."""
    arch = loader.device_arch(torch.cuda.current_device())
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    long = loader.ROUTES[arch].get(f"{role}_long")
    if long is None:
        pytest.skip(f"{arch} routes no long backward")
    start = LONG_BACKWARD_ROWS_PER_SM[arch]
    assert loader.min_rows_per_sm(long, arch) == start
    sms = _sm_count()
    if heads is None:
        x, z, w, dy = suite.make(start * sms + tokens, 1, layout="flat", seed=22)
    else:
        x, z, w, dy = suite.make(tokens, heads, layout=layout, seed=22)
    rows = x.numel() // D
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
    name = long if rows >= start * sms else loader.route_module(arch, role)
    mid = loader.ROUTES[arch].get(f"{role}_mid")
    if (
        mid is not None
        and start * sms > rows >= loader.min_rows_per_sm(mid, arch) * sms
    ):
        name = mid
    assert loader.module_for_rows(arch, role, rows, sms) == name
    assert launched == [name] * suite.REPEATS, launched


# The rows per SM from which an architecture runs the mid backward.
MID_BACKWARD_ROWS_PER_SM = {"sm_103a": 700}
# Backward inputs around and in the mid backward's rows: (tokens, heads,
# layout); ``heads=None`` takes ``threshold * SMs + tokens`` rows as one flat
# [rows, 128] tensor.
MID_BACKWARD_CASES = [
    pytest.param(-1, None, "flat", id="threshold-1"),
    pytest.param(0, None, "flat", id="threshold"),
    pytest.param(1, None, "flat", id="threshold+1"),
    pytest.param(2701, 48, "contiguous", id="2701x48"),
    pytest.param(3091, 48, "projection", id="3091x48-projection"),
    pytest.param(3157, 48, "head_major", id="3157x48-head_major"),
    pytest.param(5300, 32, "contiguous", id="5300x32"),
    pytest.param(9600, 48, "contiguous", id="9600x48"),
    pytest.param(25000, 48, "contiguous", id="25000x48"),
]


@requires_device
@pytest.mark.parametrize("tokens,heads,layout", MID_BACKWARD_CASES)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_backward_around_the_mid_threshold_matches_the_deterministic_package(
    tokens,
    heads,
    layout,
    dw_dtype,
    determinism_setting,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Around and in the mid backward's rows (``MID_BACKWARD_ROWS_PER_SM``,
    on the architectures that route it): the module launched is
    ``module_for_rows``', ``dx`` and ``dz`` are the deterministic package's
    bits, ``dw`` lies within the budget of the deterministic ``dw`` on every
    repeat, and the workspace is zero after every call."""
    arch = loader.device_arch(torch.cuda.current_device())
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    mid = loader.ROUTES[arch].get(f"{role}_mid")
    if mid is None:
        pytest.skip(f"{arch} routes no mid backward")
    start = MID_BACKWARD_ROWS_PER_SM[arch]
    assert loader.min_rows_per_sm(mid, arch) == start
    sms = _sm_count()
    if heads is None:
        x, z, w, dy = suite.make(start * sms + tokens, 1, layout="flat", seed=24)
    else:
        x, z, w, dy = suite.make(tokens, heads, layout=layout, seed=24)
    rows = x.numel() // D
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
    name = loader.module_for_rows(arch, role, rows, sms)
    long = loader.ROUTES[arch].get(f"{role}_long")
    if long is None or rows < loader.min_rows_per_sm(long, arch) * sms:
        assert name == (mid if rows >= start * sms else loader.route_module(arch, role))
    assert launched == [name] * suite.REPEATS, launched


# The most rows of the short backward.
SHORT_BACKWARD_ROWS = 256
# Backward inputs around the short backward's 256 rows: (tokens, heads,
# layout); ``heads=None`` takes ``tokens`` rows as one flat [rows, 128] tensor.
SHORT_BACKWARD_CASES = [
    pytest.param(1, 48, "contiguous", id="1x48"),
    pytest.param(5, 48, "projection", id="5x48-projection"),
    pytest.param(13, 7, "head_major", id="13x7-head_major"),
    pytest.param(255, None, "flat", id="threshold-1"),
    pytest.param(256, None, "flat", id="threshold"),
    pytest.param(257, None, "flat", id="threshold+1"),
]


@requires_device
@pytest.mark.parametrize("tokens,heads,layout", SHORT_BACKWARD_CASES)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_backward_around_the_short_threshold_matches_the_deterministic_package(
    tokens,
    heads,
    layout,
    dw_dtype,
    determinism_setting,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Around the short backward's 256 rows: the module launched is
    ``module_for_rows``' (where the architecture routes it, the short backward
    up to ``SHORT_BACKWARD_ROWS`` rows, else the backward), ``dx`` and ``dz``
    are the deterministic package's bits, ``dw`` lies within the budget of the
    deterministic ``dw`` on every repeat (an accumulated one within the budget
    of the FP64 reference), and the workspace is zero after every call."""
    arch = loader.device_arch(torch.cuda.current_device())
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    sms = _sm_count()
    if heads is None:
        x, z, w, dy = suite.make(tokens, 1, layout="flat", seed=23)
    else:
        x, z, w, dy = suite.make(tokens, heads, layout=layout, seed=23)
    rows = x.numel() // D
    mag = suite.magnitudes(dy, x, z, EPS)

    def out(value=None):
        if value is None:
            return torch.empty(D, dtype=dw_dtype, device=x.device)
        return torch.full((D,), value, dtype=dw_dtype, device=x.device)

    dx_d, dz_d, dw_d = det_backward(dy, x, z, w, EPS, grad_weight=out())
    dw64 = suite.fp64_reference(dy, x, z, w, EPS)[3]
    ws = nd_workspace(rows, x.device)
    launched = _spy_launches(monkeypatch)
    for _ in range(suite.REPEATS):
        dx, dz, dw = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
        assert dw.dtype == dw_dtype
        assert torch.equal(dx, dx_d), "dx vs the deterministic package"
        assert torch.equal(dz, dz_d), "dz vs the deterministic package"
        suite.assert_dw_within(dw, dw_d, mag, what="dw vs the deterministic package")
        assert not bool(ws.any()), "the workspace is not zero"
    acc = nd_backward(
        dy, x, z, w, EPS, grad_weight=out(0.25), accumulate_grad_weight=True
    )[2]
    suite.assert_dw_within(acc, 0.25 + dw64, mag, what="accumulated dw vs FP64")
    short = loader.ROUTES[arch].get(f"{role}_short")
    want = loader.route_module(arch, role)
    if short is not None and rows <= SHORT_BACKWARD_ROWS:
        assert loader.MODULES[short]["grid_rule"]["max_rows"] == SHORT_BACKWARD_ROWS
        want = short
    assert loader.module_for_rows(arch, role, rows, sms) == want
    assert launched == [want] * (suite.REPEATS + 1), launched


# --------------------------------------------------------------------------- one head: packed rows


def _packed(strides: dict) -> bool:
    """The kernels' packed-rows test over the strides of one launch: every
    input's head stride is 128 and its token stride ``heads * 128``.  The mid
    and the extra-long forward and, on sm_100a and sm_103a, the long backward
    warm row blocks in L2 with a bulk prefetch only when it holds."""
    heads = int(strides["heads"])
    inputs = [k[: -len("_head")] for k in strides if k.endswith("_head")]
    return bool(inputs) and all(
        strides[f"{n}_head"] == D and strides[f"{n}_tok"] == heads * D for n in inputs
    )


def _spy_strides(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    """``(module, {"heads", "<input>_tok", "<input>_head"})`` of every launch."""
    launched: list[tuple[str, dict]] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        strides = {
            k: int(v)
            for k, v in values.items()
            if k == "heads" or k.endswith(("_tok", "_head"))
        }
        launched.append((name, strides))
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    return launched


# One-head layouts of a [rows, 128] input: (passes the packed-rows test, read
# in place).  ``three_d`` [rows, 1, 128], ``flat`` [rows, 128], ``transposed``
# the transpose of [1, rows, 128] (token stride 128, head stride rows * 128),
# ``first_of_two`` the first head of [rows, 2, 128] (token stride 256: rows
# that do not follow one another), ``misaligned`` [rows, 128] at a storage
# offset of 8 elements (copied once by the host).
ONE_HEAD_LAYOUTS = {
    "three_d": (True, True),
    "flat": (True, True),
    "transposed": (True, True),
    "first_of_two": (False, True),
    "misaligned": (True, False),
}


def _one_head(base: torch.Tensor, layout: str) -> torch.Tensor:
    """``base`` ([rows, 128]) as a one-head input of ``layout``."""
    rows = base.shape[0]
    if layout == "three_d":
        return base.clone().view(rows, 1, D)
    if layout == "flat":
        return base.clone()
    if layout == "transposed":
        return base.clone().view(1, rows, D).transpose(0, 1)
    if layout == "first_of_two":
        wide = torch.zeros(rows, 2, D, dtype=base.dtype, device=base.device)
        wide[:, 0] = base
        return wide[:, :1]
    if layout == "misaligned":
        buf = torch.zeros(rows * D + 8, dtype=base.dtype, device=base.device)
        out = buf[8:].view(rows, D)
        out.copy_(base)
        return out
    raise ValueError(layout)


@pytest.mark.parametrize("layout", list(ONE_HEAD_LAYOUTS))
def test_one_head_input_passes_the_packed_rows_test_when_its_rows_follow_one_another(
    layout,
) -> None:
    """Without a device: a one-head input's head stride is passed as 128 (it is
    never dereferenced), so the kernels see packed rows exactly when its token
    stride is 128 (or the host copied it), as for ``[T, H, 128]`` inputs."""
    base = torch.randn(7, D).bfloat16()
    t = _one_head(base, layout)
    view, tok, head = backend._kernel_view(t, "x", shape=t.shape, device=t.device)
    packed, in_place = ONE_HEAD_LAYOUTS[layout]
    assert head == D
    assert tok == (2 * D if layout == "first_of_two" else D)
    assert _packed({"heads": 1, "x_tok": tok, "x_head": head}) == packed
    assert (view.data_ptr() == t.data_ptr()) == in_place
    assert torch.equal(view.reshape(-1, D), base)


@pytest.mark.parametrize(
    "shape,strides,want",
    [
        pytest.param((1, 1, D), None, (0, D), id="1x1"),
        pytest.param((1, D), None, (0, D), id="1-flat"),
        pytest.param((D,), None, (0, D), id="one-row"),
        pytest.param((1, 2, D), None, (0, D), id="1x2"),
        pytest.param((5, 2, D), None, (2 * D, D), id="5x2"),
        pytest.param((5, 3, D), None, (3 * D, D), id="5x3"),
        pytest.param((5, 2, D), (4 * D, D, 1), (4 * D, D), id="5x2-projection"),
        pytest.param((5, 2, D), (D, 5 * D, 1), (D, 5 * D), id="5x2-head_major"),
    ],
)
def test_kernel_view_strides(shape, strides, want) -> None:
    """Without a device: a single token's stride is passed as 0, a single
    head's as 128, any other dimension's as it is."""
    if strides is None:
        t = torch.zeros(shape, dtype=torch.bfloat16)
    else:
        t = torch.empty_strided(shape, strides, dtype=torch.bfloat16)
    _, tok, head = backend._kernel_view(t, "x", shape=t.shape, device=t.device)
    assert (tok, head) == want


# The modules that warm packed rows in L2, by role: the one-head inputs take
# one row past each one's start.
ONE_HEAD_ROUTES = [
    pytest.param("fwd", "fwd_mid", id="fwd_mid"),
    pytest.param("fwd", "fwd_xlong", id="fwd_xlong"),
    pytest.param("bwd_bf16", "bwd_bf16_long", id="bwd_bf16_long"),
    pytest.param("bwd_f32", "bwd_f32_long", id="bwd_f32_long"),
]


@requires_device
@pytest.mark.parametrize("role,tier", ONE_HEAD_ROUTES)
def test_one_head_inputs_match_the_flat_rows_and_the_deterministic_package(
    role, tier, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One row past the start of the mid and the extra-long forward and of the
    long backward: a one-head input of every layout (``ONE_HEAD_LAYOUTS``)
    launches ``module_for_rows``' module with head stride 128, passes the
    packed-rows test exactly when its rows follow one another (or the host
    copied it), and returns the deterministic package's ``y``, ``dx`` and
    ``dz`` bits and a ``dw`` within the budget of the deterministic ``dw`` on
    every repeat, with the workspace zero after every call."""
    arch = loader.device_arch(torch.cuda.current_device())
    name = loader.ROUTES[arch].get(tier)
    if name is None:
        pytest.skip(f"{arch} routes no {tier} module")
    sms = _sm_count()
    rows = loader.min_rows_per_sm(name, arch) * sms + 1
    assert loader.module_for_rows(arch, role, rows, sms) == name
    x0, z0, w, dy0 = suite.make(rows, 1, layout="flat", seed=25)
    dw_dtype = torch.float32 if role == "bwd_f32" else torch.bfloat16

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x0.device)

    if role == "fwd":
        y_d = det_forward(x0, z0, w, EPS)
    else:
        dx_d, dz_d, dw_d = det_backward(dy0, x0, z0, w, EPS, grad_weight=out())
        mag = suite.magnitudes(dy0, x0, z0, EPS)
        ws = nd_workspace(rows, x0.device)
    launched = _spy_strides(monkeypatch)
    for layout, (packed, _) in ONE_HEAD_LAYOUTS.items():
        x, z, dy = (_one_head(t, layout) for t in (x0, z0, dy0))
        launched.clear()
        if role == "fwd":
            y = nd_forward(x, z, w, EPS)
            assert torch.equal(y.reshape(rows, D), y_d), (layout, "y")
            calls = 1
        else:
            for _ in range(suite.REPEATS):
                dx, dz, dw = nd_backward(
                    dy, x, z, w, EPS, grad_weight=out(), workspace=ws
                )
                assert torch.equal(dx.reshape(rows, D), dx_d), (layout, "dx")
                assert torch.equal(dz.reshape(rows, D), dz_d), (layout, "dz")
                suite.assert_dw_within(
                    dw, dw_d, mag, what=f"{layout}: dw vs the deterministic package"
                )
                assert not bool(ws.any()), (layout, "the workspace is not zero")
            calls = suite.REPEATS
        assert [n for n, _ in launched] == [name] * calls, (layout, launched)
        for _, strides in launched:
            assert strides["heads"] == 1, (layout, strides)
            heads = [v for k, v in strides.items() if k.endswith("_head")]
            assert heads and all(v == D for v in heads), (layout, strides)
            assert _packed(strides) == packed, (layout, strides)


# Identical rows whose weight-gradient terms ``dy * silu(z) * x * rstd`` are
# below FP32's smallest subnormal, 2^-149 (``x`` 1, ``z`` 2^-133, the smallest
# BF16 subnormal, so ``silu(z)`` is 2^-134 exactly; ``dy`` 2^-16 .. 2^-14 by
# column): each term rounds to 0 or 2^-149 in FP32, the same in every row, so
# the error grows with the rows.  Rows routed to the short, the mid and the long
# backward on every architecture.
SUBNORMAL_SHAPES = [
    pytest.param(16, 1, id="short-16x1"),
    pytest.param(4096, 1, id="mid-4096x1"),
    pytest.param(8192, 32, id="long-8192x32"),
]


def _underflowing_rows(T: int, H: int) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    col = torch.arange(D, device=device, dtype=torch.float64)
    x = torch.ones(T, H, D, device=device).bfloat16()
    z = torch.full((T, H, D), 2.0**-133, device=device).bfloat16()
    scale = torch.where(
        col < 64, 2.0**-16 * (1 + col / 64), 2.0**-15 * (1 + (col - 64) / 64)
    )
    dy = scale.float().bfloat16().view(1, 1, D).expand(T, H, D).contiguous()
    w = torch.ones(D, device=device).bfloat16()
    return x, z, w, dy


@requires_device
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
@pytest.mark.parametrize("T,H", SUBNORMAL_SHAPES)
def test_subnormal_weight_gradient_matches_the_deterministic_package(
    T, H, dw_dtype, determinism_setting
) -> None:
    """Weight gradients in the subnormal range, where the documented budget
    ``|dw64| * 2^-7 + sum|terms| * 2^-20`` (for FP32 normal gates, terms and
    sums) does not hold for either package: ``dx`` and ``dz`` are the
    deterministic package's bits, a BF16 ``dw`` the deterministic ``dw``'s
    bits, and every ``dw`` lies within that budget widened by the documented
    underflow allowance, ``(1 + |x * rstd|) * 2^-150`` per row plus half the
    dtype's smallest subnormal, on every repeat, with the workspace zero."""
    x, z, w, dy = _underflowing_rows(T, H)
    rows = T * H
    dx_d, dz_d, dw_d = det_backward(
        dy, x, z, w, EPS, grad_weight=torch.empty(D, dtype=dw_dtype, device=x.device)
    )
    xd, zd = x.double(), z.double()
    n = xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + EPS)
    terms = (dy.double() * zd * torch.sigmoid(zd) * n).reshape(rows, D)
    dw64 = terms.sum(0)
    smallest = 2.0**-133 if dw_dtype == torch.bfloat16 else 2.0**-149
    allowance = (1 + n.abs().max().item()) * rows * 2.0**-150 + smallest / 2
    budget = dw64.abs() * suite.BF16_RTOL + terms.abs().sum(0) * suite.DW_MAG_TOL
    assert bool(((dw_d.double() - dw64).abs() > budget).any()), (
        "the inputs no longer leave the normal-range budget"
    )
    ws = nd_workspace(rows, x.device)
    for _ in range(suite.REPEATS):
        dw = torch.empty(D, dtype=dw_dtype, device=x.device)
        dx, dz, _ = nd_backward(dy, x, z, w, EPS, grad_weight=dw, workspace=ws)
        assert torch.equal(dx, dx_d), "dx"
        assert torch.equal(dz, dz_d), "dz"
        if dw_dtype == torch.bfloat16:
            assert torch.equal(dw.view(torch.int16), dw_d.view(torch.int16)), "dw bits"
        excess = (dw.double() - dw64).abs() - budget - allowance
        assert not bool((excess > 0).any()), (
            f"dw beyond the floored budget by {excess.max().item():.3e}"
        )
        assert not bool(ws.any()), "the workspace is not zero"
