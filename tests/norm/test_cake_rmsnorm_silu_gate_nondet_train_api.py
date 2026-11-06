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
package with the non-deterministic weight gradient (``nondet``): the
``flashinfer.cake_rmsnorm_silu_gate_nondet_train`` facade, its re-export from
the ``flashinfer`` top level, and its interplay with
``torch.use_deterministic_algorithms``.

The mode is part of every name.  Every entry point of the facade and of the
standalone backend ``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train``
runs only the non-deterministic kernels; no entry point takes a
``deterministic`` keyword (passing one is a ``TypeError``), and no call runs
the deterministic package ``flashinfer.cake_rmsnorm_silu_gate_det_train``.
This module checks:

* the surface without a device: the exports (one autograd function, the
  backend's class), the facade's sources (no import of the deterministic
  package), the signature of every entry point (no ``deterministic``
  parameter), the ``TypeError`` for a ``deterministic`` keyword of any value
  from every facade and backend entry point under each
  ``torch.use_deterministic_algorithms`` setting (nothing launched), that the
  facade runs the backend with the same arguments (the backend replaced by
  recorders), the autograd function in both directions, the support query and
  the workspace reset, the backend's standalone form (no import of another
  flashinfer package; a copy under another name imports with flashinfer
  blocked), and the strides the backend passes for one-head and other inputs
  (a single head's stride as 128, so packed one-head rows pass the kernels'
  packed-rows test);
* on a device: the alert of an autograd node whose backward runs after
  ``torch.use_deterministic_algorithms(True)`` was enabled (exact texts; none
  for a node without a weight gradient or of at most 256 rows), the
  ``TypeError`` for a mode keyword on real tensors before any launch; the
  entry points launch only the backend's modules, each package's backward
  refuses the other's workspace, and on the backend suite's accuracy matrix
  and paths (accumulation, ``compute_grad_weight=False``, a reset workspace
  with nonzero scratch reused across shapes, two streams, a side stream, CUDA
  Graph replays, autograd and non-reentrant checkpointing) its ``y``, ``dx``
  and ``dz`` are the deterministic package's bits and its ``dw`` lies within
  the weight-gradient budget of the deterministic ``dw``, on every repeat; so
  are they for one-head inputs of several layouts past the start of each
  module that warms packed rows in L2, which launch with the packed-rows test
  passing exactly when their rows follow one another; and for weight
  gradients in the subnormal range (identical rows whose FP32 terms
  underflow) on the short, the mid and the long backward: ``dx`` and ``dz``
  the deterministic bits, a BF16 ``dw`` the deterministic ``dw``'s bits, and
  every ``dw`` within the budget widened by the underflow allowance the
  package documents.

The numerical contract of the non-deterministic kernels against the FP64
reference is ``test_cake_rmsnorm_silu_gate_nondet_train.py``'s; its data
generators, references and budgets are reused here.
"""

from __future__ import annotations

import ast
import inspect
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy
import pytest
import torch
import torch.utils.checkpoint

import flashinfer
import flashinfer.cake_rmsnorm_silu_gate_det_train as deterministic_facade
import flashinfer.cake_rmsnorm_silu_gate_nondet_train as facade
import flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train as backend
from flashinfer.experimental.cake.cake_rmsnorm_silu_gate_det_train import (
    cake_jit as deterministic_loader,
)
from flashinfer.cake_rmsnorm_silu_gate_nondet_train import (
    CakeRMSNormSiLUGateNondetFunction,
    cake_rmsnorm_silu_gate_nondet,
    cake_rmsnorm_silu_gate_nondet_train_backward,
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace,
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes,
    cake_rmsnorm_silu_gate_nondet_train_forward,
    cake_rmsnorm_silu_gate_nondet_train_reset_workspace,
    is_cake_rmsnorm_silu_gate_nondet_train_supported,
)
from flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train import (
    cake_jit as loader,
)
from tests.norm import test_cake_rmsnorm_silu_gate_nondet_train as suite

D = 128
EPS = 1e-6
DET = "flashinfer.cake_rmsnorm_silu_gate_det_train"
FACADE = "flashinfer.cake_rmsnorm_silu_gate_nondet_train"
BACKEND = "flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train"
# The entry points of the facade and of the backend package (the same names;
# the autograd function is the backend's class).
ENTRY_NAMES = (
    "cake_rmsnorm_silu_gate_nondet_train_forward",
    "cake_rmsnorm_silu_gate_nondet_train_backward",
    "cake_rmsnorm_silu_gate_nondet_train_backward_workspace",
    "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate_nondet_train_reset_workspace",
    "cake_rmsnorm_silu_gate_nondet",
    "is_cake_rmsnorm_silu_gate_nondet_train_supported",
)
# Values a caller might pass as ``deterministic``.
MODES = (True, False, None, 0, 1, "False", numpy.bool_(False))
MODE_IDS = ["true", "false", "none", "zero", "one", "string", "numpy_false"]
BACKEND_DIR = Path(backend.__file__).parent

nd_forward = cake_rmsnorm_silu_gate_nondet_train_forward
nd_backward = cake_rmsnorm_silu_gate_nondet_train_backward
nd_workspace = cake_rmsnorm_silu_gate_nondet_train_backward_workspace
nd_reset = cake_rmsnorm_silu_gate_nondet_train_reset_workspace
nd_gated_rmsnorm = cake_rmsnorm_silu_gate_nondet
det_forward = deterministic_facade.cake_rmsnorm_silu_gate_det_train_forward
det_backward = deterministic_facade.cake_rmsnorm_silu_gate_det_train_backward


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and is_cake_rmsnorm_silu_gate_nondet_train_supported(torch.device("cuda"))
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


def _stub_kernels(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace the backend's entry points and its forward and backward bodies
    by recorders (no device needed)."""
    calls: list = []
    for name in (*ENTRY_NAMES, "_forward", "_backward"):
        monkeypatch.setattr(
            backend, name, lambda *a, _name=name, **k: calls.append((_name, a, k))
        )
    monkeypatch.setattr(
        loader, "run_module", lambda name, *a, **k: calls.append(("run", name))
    )
    return calls


def _mode_calls(module, ws) -> dict:
    """One call of every entry point of ``module`` (the facade or the backend)
    on CPU operands, taking extra keywords: ``entry -> call``."""
    x, w = _cpu_tensors()
    return {
        "cake_rmsnorm_silu_gate_nondet_train_forward": lambda **k: module.cake_rmsnorm_silu_gate_nondet_train_forward(x, x, w, EPS, **k),
        "cake_rmsnorm_silu_gate_nondet_train_backward": lambda **k: module.cake_rmsnorm_silu_gate_nondet_train_backward(x, x, x, w, EPS, workspace=ws, **k),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace": lambda **k: module.cake_rmsnorm_silu_gate_nondet_train_backward_workspace(8, "cuda:0", **k),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes": lambda **k: module.cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(8, "cuda:0", **k),
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace": lambda **k: module.cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws, **k),
        "cake_rmsnorm_silu_gate_nondet": lambda **k: module.cake_rmsnorm_silu_gate_nondet(x, x, w, EPS, **k),
        "is_cake_rmsnorm_silu_gate_nondet_train_supported": lambda **k: module.is_cake_rmsnorm_silu_gate_nondet_train_supported("cuda:0", **k),
    }  # fmt: skip


# --------------------------------------------------------------------------- the surface (no device)


def test_top_level_exports_are_the_facade() -> None:
    assert flashinfer.cake_rmsnorm_silu_gate_nondet is cake_rmsnorm_silu_gate_nondet
    assert (
        flashinfer.cake_rmsnorm_silu_gate_nondet_train_forward
        is cake_rmsnorm_silu_gate_nondet_train_forward
    )
    assert (
        flashinfer.cake_rmsnorm_silu_gate_nondet_train_backward
        is cake_rmsnorm_silu_gate_nondet_train_backward
    )
    assert (
        flashinfer.cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes
        is cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes
    )
    # one non-deterministic autograd function: the backend's class, reached
    # through the facade and the top level
    assert (
        CakeRMSNormSiLUGateNondetFunction is backend.CakeRMSNormSiLUGateNondetFunction
    )
    assert (
        flashinfer.CakeRMSNormSiLUGateNondetFunction
        is CakeRMSNormSiLUGateNondetFunction
    )
    # the det names stay the det package's objects
    assert (
        flashinfer.cake_rmsnorm_silu_gate_det
        is deterministic_facade.cake_rmsnorm_silu_gate_det
    )
    assert (
        flashinfer.cake_rmsnorm_silu_gate_det_train_forward
        is deterministic_facade.cake_rmsnorm_silu_gate_det_train_forward
    )


def test_facade_imports_no_deterministic_package() -> None:
    """The facade never runs the deterministic package: its source imports
    nothing from it."""
    tree = ast.parse(Path(facade.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [alias.name for alias in node.names]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        assert not any("silu_gate_det" in name for name in names), (node.lineno, names)


@pytest.mark.parametrize("namespace", ["facade", "backend"])
def test_no_entry_point_takes_a_mode_keyword(namespace: str) -> None:
    """The mode is in the name (``nondet``): no entry point of the facade or
    the backend package has a ``deterministic`` parameter, and the autograd
    function takes no fifth input."""
    module = {"facade": facade, "backend": backend}[namespace]
    assert set(ENTRY_NAMES) | {"CakeRMSNormSiLUGateNondetFunction"} == set(
        module.__all__
    ) - {"ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES"}
    for name in ENTRY_NAMES:
        parameters = inspect.signature(getattr(module, name)).parameters
        assert "deterministic" not in parameters, (namespace, name)
    assert list(
        inspect.signature(CakeRMSNormSiLUGateNondetFunction.forward).parameters
    ) == ["ctx", "x", "z", "weight", "eps"]


@pytest.mark.parametrize("value", MODES, ids=MODE_IDS)
def test_a_mode_keyword_is_a_type_error(
    value, monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """``deterministic=<any value>`` is a ``TypeError`` from every facade and
    backend entry point before any call reaches a kernel, under each
    ``torch.use_deterministic_algorithms`` setting; so is a fifth input of the
    autograd function (the backend's bodies recorded: no device needed)."""
    x, w = _cpu_tensors()
    ws = torch.zeros(4 * D + 1, dtype=torch.float64)
    launched: list = []
    monkeypatch.setattr(
        loader, "run_module", lambda name, *a, **k: launched.append(name)
    )
    for enabled, warn_only in ((False, False), (True, False), (True, True)):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        for module in (facade, backend):
            for call in _mode_calls(module, ws).values():
                with pytest.raises(TypeError, match="deterministic"):
                    call(deterministic=value)
        with pytest.raises(TypeError):
            CakeRMSNormSiLUGateNondetFunction.apply(x, x, w, EPS, value)
    assert launched == []
    assert not bool(ws.any())


def test_facade_runs_the_backend_with_the_same_arguments(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """Each facade entry point calls the backend entry point of the same name
    with the same arguments and nothing else (stubbed)."""
    calls = _stub_kernels(monkeypatch)
    x, w = _cpu_tensors()
    ws = torch.zeros(4 * D + 1, dtype=torch.float64)
    cake_rmsnorm_silu_gate_nondet_train_forward(x, x, w, EPS, out=x)
    cake_rmsnorm_silu_gate_nondet_train_backward(
        x, x, x, w, EPS, workspace=ws, compute_grad_weight=False
    )
    cake_rmsnorm_silu_gate_nondet(x, x, w, EPS)
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace(7, "cuda:0")
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(7, "cuda:0")
    cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws)
    is_cake_rmsnorm_silu_gate_nondet_train_supported("cuda:0")
    names = [c[0] for c in calls]
    assert names == [
        "cake_rmsnorm_silu_gate_nondet_train_forward",
        "cake_rmsnorm_silu_gate_nondet_train_backward",
        "cake_rmsnorm_silu_gate_nondet",
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace",
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes",
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace",
        "is_cake_rmsnorm_silu_gate_nondet_train_supported",
    ], names
    assert calls[0][1:] == ((x, x, w, EPS), {"out": x})
    assert calls[1][1] == (x, x, x, w, EPS)
    assert calls[1][2] == {
        "grad_x": None,
        "grad_z": None,
        "grad_weight": None,
        "accumulate_grad_weight": False,
        "compute_grad_weight": False,
        "workspace": ws,
    }
    assert calls[2][1:] == ((x, x, w, EPS), {})
    assert calls[3][1:] == ((7, "cuda:0"), {})
    assert calls[4][1:] == ((7, "cuda:0"), {})
    assert calls[5][1:] == ((ws,), {})
    assert calls[6][1:] == (("cuda:0",), {})


def test_autograd_function_runs_the_backend_in_both_directions(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """:class:`CakeRMSNormSiLUGateNondetFunction` (the backend's class, also an
    attribute of the facade) runs the backend's forward and backward bodies;
    its backward names the autograd backward for
    the alert.  The forward does not alert under
    ``torch.use_deterministic_algorithms(True)`` (bodies stubbed: no device
    needed)."""
    calls: list = []
    monkeypatch.setattr(
        backend,
        "_forward",
        lambda x, z, w, eps, *, out: calls.append(("fwd", out)) or x * 1,
    )
    monkeypatch.setattr(
        backend,
        "_backward",
        lambda dy, x, z, w, eps, **k: calls.append(("bwd", k))
        or (dy * 1, dy * 1, w * 1),
    )
    x, w = _cpu_tensors()
    leaves = [t.clone().requires_grad_(True) for t in (x, x, w)]
    torch.use_deterministic_algorithms(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = CakeRMSNormSiLUGateNondetFunction.apply(*leaves, EPS)
    assert y.grad_fn._forward_cls is backend.CakeRMSNormSiLUGateNondetFunction
    y.backward(torch.ones_like(y))
    assert calls == [
        ("fwd", None),
        (
            "bwd",
            {
                "grad_x": None,
                "grad_z": None,
                "grad_weight": None,
                "accumulate_grad_weight": False,
                "compute_grad_weight": True,
                "workspace": None,
                "entry": "CakeRMSNormSiLUGateNondetFunction.backward",
            },
        ),
    ]


def test_support_query_ignores_torch_determinism(determinism_setting) -> None:
    torch.use_deterministic_algorithms(True)
    cpu = torch.device("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert is_cake_rmsnorm_silu_gate_nondet_train_supported(cpu) is False
        assert backend.is_cake_rmsnorm_silu_gate_nondet_train_supported(cpu) is False


def test_reset_workspace_passes_through(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """The facade's reset resets through the backend whatever
    ``torch.use_deterministic_algorithms`` says (a fill has no reduction); the
    backend refuses an FP32 buffer (checked on the device)."""
    resets: list = []
    monkeypatch.setattr(
        backend,
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace",
        lambda workspace, **k: resets.append((workspace, k)),
    )
    fp64 = torch.ones(8, dtype=torch.float64)
    for enabled, warn_only in ((False, False), (True, False), (True, True)):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cake_rmsnorm_silu_gate_nondet_train_reset_workspace(fp64)
    assert resets == [(fp64, {})] * 3


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
    "supported_cpu": package.is_cake_rmsnorm_silu_gate_nondet_train_supported(
        torch.device("cpu")
    ),
    "all": sorted(package.__all__),
}
try:
    package.cake_rmsnorm_silu_gate_nondet_train_forward(x, x, x[0], deterministic=False)
except TypeError as error:
    out["mode_keyword"] = str(error)
try:
    package.cake_rmsnorm_silu_gate_nondet_train_forward(x, x, x[0])
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
    gate's first step, without its sources) imports, answers its support query,
    takes no mode keyword and refuses a CPU tensor in an interpreter where every
    ``flashinfer`` import raises; no flashinfer module is loaded."""
    neutral = "vendored_cake_rmsnorm_silu_gate_nondet_train"
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
    assert "deterministic" in out["mode_keyword"], out["mode_keyword"]
    assert "CUDA" in out["cpu"], out["cpu"]


# --------------------------------------------------------------------------- on the device


@requires_device
def test_autograd_backward_alerts_once_deterministic_algorithms_are_enabled(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """A node of :func:`cake_rmsnorm_silu_gate_nondet` is the backend's
    :class:`CakeRMSNormSiLUGateNondetFunction`.  When
    ``torch.use_deterministic_algorithms(True)`` is enabled between its forward
    and its backward, the backward raises PyTorch's alert, naming the autograd
    backward and the deterministic autograd function, immediately before the
    order-varying launch; under ``warn_only=True`` it warns and runs the
    non-deterministic backward.  A node whose weight needs no gradient, and a
    node of at most 256 rows (the fixed-order short backward), does not alert.
    The launches are recorded and skipped (``run_module`` stubbed)."""
    launched: list[str] = []
    monkeypatch.setattr(
        loader, "run_module", lambda name, *a, **k: launched.append(name)
    )
    arch = loader.device_arch(torch.cuda.current_device())
    q = f"{BACKEND}.CakeRMSNormSiLUGateNondetFunction.backward"
    d = f"{DET}.CakeRMSNormSiLUGateDetFunction"

    def node(T=257, weight_grad=True):
        x, z, w, _ = _inputs(T, 48, 21)
        y = cake_rmsnorm_silu_gate_nondet(
            x.requires_grad_(True),
            z.requires_grad_(True),
            w.requires_grad_(weight_grad),
            EPS,
        )
        assert y.grad_fn._forward_cls is backend.CakeRMSNormSiLUGateNondetFunction
        return y

    def module(T=257):
        return loader.module_for_rows(arch, "bwd_bf16", T * 48, _sm_count())

    y = node()
    launched.clear()
    torch.use_deterministic_algorithms(True)
    with pytest.raises(RuntimeError) as info:
        y.backward(torch.ones_like(y))
    assert str(info.value) == (
        f"{q} does not have a deterministic implementation, but you set "
        f"'torch.use_deterministic_algorithms(True)'. You can call {d} instead, "
        "or use the 'warn_only=True' option, if that's acceptable for your "
        "application."
    )
    assert launched == []
    torch.use_deterministic_algorithms(False)
    y = node()
    launched.clear()
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning) as record:
        y.backward(torch.ones_like(y))
    assert [str(r.message) for r in record if BACKEND in str(r.message)] == [
        f"{q} does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True, warn_only=True)'; the "
        f"non-deterministic kernels run. {d} is the deterministic one."
    ]
    assert launched == [module()]
    for T, weight_grad in ((257, False), (5, True)):
        torch.use_deterministic_algorithms(False)
        y = node(T, weight_grad)
        launched.clear()
        torch.use_deterministic_algorithms(True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            y.backward(torch.ones_like(y))
        assert len(launched) == 1, (T, weight_grad, launched)


@requires_device
def test_a_mode_keyword_is_a_type_error_on_the_device(
    determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a device too, ``deterministic=<any value>`` is a ``TypeError`` from
    every facade entry point before any launch or change of the workspace; so
    is a fifth input of the autograd function.  The same calls without the
    keyword run."""
    launched = _spy_launches(monkeypatch)
    T, H = 129, 48
    x, z, w, dy = _inputs(T, H, 11)
    device = x.device
    ws = nd_workspace(T * H, device)
    runs = {
        "cake_rmsnorm_silu_gate_nondet_train_forward": lambda **m: (
            cake_rmsnorm_silu_gate_nondet_train_forward(x, z, w, EPS, **m)
        ),
        "cake_rmsnorm_silu_gate_nondet_train_backward": lambda **m: (
            cake_rmsnorm_silu_gate_nondet_train_backward(
                dy, x, z, w, EPS, workspace=ws, **m
            )
        ),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace": lambda **m: (
            cake_rmsnorm_silu_gate_nondet_train_backward_workspace(T * H, device, **m)
        ),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes": lambda **m: (
            cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(
                T * H, device, **m
            )
        ),
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace": lambda **m: (
            cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws, **m)
        ),
        "cake_rmsnorm_silu_gate_nondet": lambda **m: cake_rmsnorm_silu_gate_nondet(
            x, z, w, EPS, **m
        ),
        "is_cake_rmsnorm_silu_gate_nondet_train_supported": lambda **m: (
            is_cake_rmsnorm_silu_gate_nondet_train_supported(device, **m)
        ),
    }
    assert set(runs) == set(ENTRY_NAMES)
    for run in runs.values():
        for value in (True, False):
            with pytest.raises(TypeError, match="deterministic"):
                run(deterministic=value)
    for value in (True, False):
        with pytest.raises(TypeError):
            CakeRMSNormSiLUGateNondetFunction.apply(x, z, w, EPS, value)
    assert launched == []
    torch.cuda.synchronize()
    assert not bool(ws.any())
    for run in runs.values():
        run()
    assert launched, "the entry points launched nothing"


@requires_device
def test_nondet_entry_points_run_this_package(
    determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``nondet`` entry points launch the backend's forward and its one
    backward kernel (and no det module): ``y``, ``dx`` and ``dz`` are the det
    package's bits, ``dw`` within one BF16 rounding plus the FP32 accumulation
    budget of the det ``dw``.  The two workspace kinds do not substitute: each
    backward refuses the other package's buffer, naming both."""
    launched = _spy_launches(monkeypatch)
    x, z, w, dy = _inputs(257, 48, 12)
    y = cake_rmsnorm_silu_gate_nondet_train_forward(x, z, w, EPS)
    dx, dz, dw = cake_rmsnorm_silu_gate_nondet_train_backward(dy, x, z, w, EPS)
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
    # non-deterministic backward, nor the other way round
    launched.clear()
    deterministic_ws = (
        deterministic_facade.cake_rmsnorm_silu_gate_det_train_backward_workspace(
            257 * 48, x.device
        )
    )
    with pytest.raises(ValueError) as info:
        cake_rmsnorm_silu_gate_nondet_train_backward(
            dy, x, z, w, EPS, workspace=deterministic_ws
        )
    assert str(info.value) == (
        "workspace is FP32, a workspace of the deterministic mode "
        f"({DET}.cake_rmsnorm_silu_gate_det_train_backward_workspace); the "
        "non-deterministic backward needs its own zero-filled FP64 workspace "
        f"({BACKEND}.cake_rmsnorm_silu_gate_nondet_train_backward_workspace)"
    )
    with pytest.raises(ValueError, match="workspace is FP64, a workspace of the"):
        deterministic_facade.cake_rmsnorm_silu_gate_det_train_backward(
            dy, x, z, w, EPS, workspace=nd_workspace(257 * 48, x.device)
        )
    assert launched == []


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
    ``cake_rmsnorm_silu_gate_nondet_train_reset_workspace``,
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
    assert y.grad_fn._forward_cls is backend.CakeRMSNormSiLUGateNondetFunction
    y.backward(dy)
    check(y, xa.grad, za.grad, wa.grad, "autograd")
    xf, zf, wf = (t.clone().requires_grad_(True) for t in (x, z, w))
    yf = CakeRMSNormSiLUGateNondetFunction.apply(xf, zf, wf, EPS)
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
LONG_BACKWARD_ROWS_PER_SM = {"sm_100a": 12000, "sm_103a": 12000, "sm_107a": 3200}
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
MID_BACKWARD_ROWS_PER_SM = {"sm_100a": 800, "sm_103a": 700, "sm_107a": 1024}
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


@requires_device
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_mid_backward_with_infinite_grad_out_keeps_the_documented_classes(
    dw_dtype, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Infinite ``grad_out`` in rows of the mid backward that are not the last
    (``MID_BACKWARD_ROWS_PER_SM``, on the architectures that route it): the
    weight gradient of the columns holding them is not finite (``+-inf`` or
    ``NaN``, as the module documentation states for either package), every
    other column's lies within the budget of the deterministic ``dw``, and
    ``dx`` and ``dz`` have the deterministic package's ``NaN`` positions and
    its bits everywhere else, on every repeat, with the workspace zero after
    every call."""
    arch = loader.device_arch(torch.cuda.current_device())
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    mid = loader.ROUTES[arch].get(f"{role}_mid")
    if mid is None:
        pytest.skip(f"{arch} routes no mid backward")
    sms = _sm_count()
    rows = MID_BACKWARD_ROWS_PER_SM[arch] * sms + 4097
    assert loader.module_for_rows(arch, role, rows, sms) == mid
    x, z, w, dy = suite.make(rows, 1, layout="flat", seed=27)
    # (row, column, sign): +inf twice in column 3, -inf in column 40, +inf and
    # -inf in column 77; no infinite term in the last row
    for r, c, sign in (
        (0, 3, 1.0),
        (rows // 2, 3, 1.0),
        (17, 40, -1.0),
        (rows // 3, 77, 1.0),
        (rows // 3 + 5, 77, -1.0),
    ):
        dy[r, c] = sign * float("inf")
    cols = torch.zeros(D, dtype=torch.bool, device=x.device)
    cols[[3, 40, 77]] = True
    mag = suite.magnitudes(dy, x, z, EPS)

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x.device)

    dx_d, dz_d, dw_d = det_backward(dy, x, z, w, EPS, grad_weight=out())
    assert torch.equal(~torch.isfinite(dw_d), cols), "deterministic dw columns"
    ws = nd_workspace(rows, x.device)
    launched = _spy_launches(monkeypatch)
    for _ in range(suite.REPEATS):
        dx, dz, dw = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
        assert dw.dtype == dw_dtype
        assert torch.equal(~torch.isfinite(dw), cols), "non-finite dw columns"
        suite.assert_dw_within(
            dw[~cols], dw_d[~cols], mag[~cols], what="finite dw columns"
        )
        for got, want, what in ((dx, dx_d, "dx"), (dz, dz_d, "dz")):
            nan = torch.isnan(want)
            assert torch.equal(torch.isnan(got), nan), what
            assert torch.equal(got[~nan], want[~nan]), what
        assert not bool(ws.any()), "the workspace is not zero"
    assert launched == [mid] * suite.REPEATS, launched


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
    pytest.param("bwd_bf16", "bwd_bf16_mid", id="bwd_bf16_mid"),
    pytest.param("bwd_f32", "bwd_f32_mid", id="bwd_f32_mid"),
    pytest.param("bwd_bf16", "bwd_bf16_long", id="bwd_bf16_long"),
    pytest.param("bwd_f32", "bwd_f32_long", id="bwd_f32_long"),
]


@requires_device
@pytest.mark.parametrize("role,tier", ONE_HEAD_ROUTES)
def test_one_head_inputs_match_the_flat_rows_and_the_deterministic_package(
    role, tier, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One row past the start of the mid and the extra-long forward and of the
    mid and the long backward: a one-head input of every layout
    (``ONE_HEAD_LAYOUTS``)
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
