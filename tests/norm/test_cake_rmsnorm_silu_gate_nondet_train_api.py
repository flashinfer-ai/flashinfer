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

Every output of the package repeats bit for bit.  The backward writes ``dx``
and ``dz`` with the deterministic backward's row arithmetic, so they are the
deterministic package's bits on every path checked here.  ``y`` is not: the
forward computes ``rstd`` and the SiLU gate with approximate FP32 intrinsics.
Nor, near rounding boundaries, is ``dw``: the backward adds the deterministic
chunk partials of the weight gradient exactly, as integers.  So every output
is also compared with the deterministic package through an error budget
against the backend suite's FP64 reference (:func:`assert_error_within_det`):
on the same inputs, the relative L2 error of an output and its largest error
relative to the reference's largest magnitude are each at most
``DET_ERROR_RATIO`` (1.1) times the deterministic package's, and 0 where the
deterministic package's is 0.  The deterministic outputs come from that
package's public forward and backward only.  This module checks:

* the surface without a device: the exports (one autograd function, the
  backend's class), the facade's sources (no import of the deterministic
  package), the signature of every entry point (no ``deterministic``
  parameter), the ``TypeError`` for a ``deterministic`` keyword of any value
  from every facade and backend entry point under each
  ``torch.use_deterministic_algorithms`` setting (nothing launched), that the
  facade runs the backend with the same arguments (the backend replaced by
  recorders), the autograd function in both directions with no warning under
  ``torch.use_deterministic_algorithms(True)``, the support query and the
  workspace reset, the backend's standalone form (no import of another
  flashinfer package; a copy under another name imports with flashinfer
  blocked), and the strides the backend passes for one-head and other inputs
  (a single head's stride as 128, so packed one-head rows pass the kernels'
  packed-rows test);
* on a device: no call warns or raises under
  ``torch.use_deterministic_algorithms(True)``, strict or with
  ``warn_only=True`` (the forward, the backward of more than 256 rows into a
  BF16 and an FP32 weight gradient, autograd and non-reentrant
  checkpointing); the ``TypeError`` for a mode keyword on real tensors before
  any launch; the entry points launch only the backend's modules; the
  non-deterministic backward refuses an FP32 workspace (the deterministic
  package's) naming both modes' workspaces, also in a call with no rows with
  the error of a call with rows, and the deterministic backward refuses the
  ``torch.uint8`` workspace of this package with its own FP32 workspace
  requirement; on the backend suite's accuracy matrix (weight-gradient terms
  that cancel per column included) and paths (accumulation,
  ``compute_grad_weight=False``, a reset
  workspace with nonzero bytes reused across shapes, two streams, a side
  stream, CUDA Graph replays, autograd and non-reentrant checkpointing) every
  output repeats bit for bit (an accumulated ``dw`` from the same starting
  value too), ``dx`` and ``dz`` are the deterministic package's bits, ``y``,
  ``dx``, ``dz`` and ``dw`` lie within the error budget, and ``dw`` within
  the weight-gradient budget of the deterministic ``dw``; so do they around
  the ends of the forward's ``FORWARD_BANDS`` bands (packed rows) and
  ``STRIDED_FORWARD_BANDS`` bands (other layouts) and
  at the first and last rows of each backward program, each call launching
  ``module_for_rows``' module: the small-row backwards (one cluster each, in
  the rows of their ``BACKWARD_BANDS`` bands), the static backward (at most 8
  chunks of rows per SM) and the claimed backward (more chunks); gate inputs at and past the packed
  gate's range (around FP32 ``expf``'s overflow too), in the forward and the
  backward, and infinite upstream gradients give the FP64 reference's
  ``NaN``, ``+inf`` and ``-inf`` positions and finite elements within the
  error budget (``dx`` and ``dz`` have the deterministic package's ``NaN``
  positions and bits elsewhere); one-head inputs of several layouts one row past the start of
  each module that warms rows in L2 give the same bits for every layout,
  passing the packed-rows test exactly when their rows follow one another;
  and weight gradients in the subnormal range (identical rows whose FP32 terms
  underflow) lie within the weight-gradient budget widened by the underflow
  allowance the package documents (``dx`` and ``dz`` the deterministic
  package's bits, a BF16 ``dw`` the deterministic ``dw``'s bits).

The numerical contract of the non-deterministic kernels against the FP64
reference is ``test_cake_rmsnorm_silu_gate_nondet_train.py``'s; its data
generators, references and budgets are reused here.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import re
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
# The bound of an output's errors against the FP64 reference, as a multiple of
# the deterministic package's errors on the same inputs.
DET_ERROR_RATIO = 1.1
# The non-deterministic backward's refusal of an FP32 workspace (the
# deterministic package's), naming both modes' workspace dtypes and factories
# (the backend suite asserts the same text).
REFUSAL = (
    "workspace is FP32, a workspace of the deterministic mode "
    "(flashinfer.cake_rmsnorm_silu_gate_det_train."
    "cake_rmsnorm_silu_gate_det_train_backward_workspace); the "
    "non-deterministic backward needs its own zero-filled uint8 workspace "
    f"({backend.__name__}.cake_rmsnorm_silu_gate_nondet_train_backward_workspace)"
)

nd_forward = cake_rmsnorm_silu_gate_nondet_train_forward
nd_backward = cake_rmsnorm_silu_gate_nondet_train_backward
nd_workspace = cake_rmsnorm_silu_gate_nondet_train_backward_workspace
nd_workspace_bytes = cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes
nd_reset = cake_rmsnorm_silu_gate_nondet_train_reset_workspace
nd_gated_rmsnorm = cake_rmsnorm_silu_gate_nondet
# The deterministic package, through its public forward and backward only.
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


def fp64_errors(out, ref, mask=None) -> tuple[float, float]:
    """``(rel_l2, max_rel)`` of ``out`` against the FP64 reference ``ref``, in
    FP64, over the elements ``mask`` picks (all without one):
    ``||out - ref||_2 / ||ref||_2`` and ``max|out - ref| / max|ref|`` (a zero
    reference divides by 1)."""
    ref = ref.double().reshape(-1)
    out = out.double().reshape(-1)
    if mask is not None:
        mask = mask.reshape(-1)
        ref, out = ref[mask], out[mask]
    diff = out - ref
    l2 = ref.norm().item() or 1.0
    top = ref.abs().max().item() or 1.0
    return diff.norm().item() / l2, diff.abs().max().item() / top


def assert_error_within_det(what, got, det, ref, *, mask=None) -> None:
    """The error budget against the deterministic package: ``got``'s
    ``rel_l2`` and ``max_rel`` against the FP64 reference ``ref``
    (:func:`fp64_errors`) are each at most ``DET_ERROR_RATIO`` times those of
    ``det``, the deterministic package's output on the same inputs, so 0
    where ``det``'s is 0.  ``mask`` picks the elements compared."""
    errors = zip(
        ("rel_l2", "max_rel"),
        fp64_errors(got, ref, mask),
        fp64_errors(det, ref, mask),
        strict=True,
    )
    for metric, nondet_error, det_error in errors:
        ratio = nondet_error / det_error if det_error else float("inf")
        assert nondet_error <= DET_ERROR_RATIO * det_error, (
            f"{what}: {metric} vs FP64 {nondet_error:.6e}, the deterministic "
            f"package's {det_error:.6e}: ratio {ratio:.4f} > {DET_ERROR_RATIO}"
        )


def _bits(t: torch.Tensor) -> torch.Tensor:
    """``t``'s bits as integers (a ``NaN`` equals its own bits)."""
    ints = {2: torch.int16, 4: torch.int32, 8: torch.int64}
    return t.detach().view(ints[t.element_size()])


def assert_same_bits(what, got, want) -> None:
    """Run to run: ``got`` is ``want``'s bits."""
    assert got.dtype == want.dtype and got.shape == want.shape, (
        what,
        got.dtype,
        want.dtype,
        tuple(got.shape),
        tuple(want.shape),
    )
    assert torch.equal(_bits(got), _bits(want)), f"{what}: not the same bits"


def assert_same_classes(what, got, ref) -> None:
    """``got`` has the FP64 reference's classes: ``NaN`` exactly where it is
    ``NaN`` (any sign and payload), ``+inf`` and ``-inf`` exactly where it is."""
    got = got.detach().double().reshape(-1)
    ref = ref.double().reshape(-1)
    for name, cls in (
        ("NaN", torch.isnan),
        ("+inf", torch.isposinf),
        ("-inf", torch.isneginf),
    ):
        assert torch.equal(cls(got), cls(ref)), f"{what}: {name} positions vs FP64"


def assert_det_bits(what, got, det) -> None:
    """``got`` (``dx`` or ``dz``) is the deterministic package's ``det``
    bits."""
    assert torch.equal(got, det), f"{what} vs the deterministic package"


def assert_det_bits_beside_nans(what, got, det) -> None:
    """``got`` has ``det``'s ``NaN`` positions and its bits everywhere
    else."""
    nan = torch.isnan(det)
    assert torch.equal(torch.isnan(got), nan), f"{what}: NaN positions vs det"
    assert torch.equal(got[~nan], det[~nan]), f"{what} vs the deterministic package"


def _assert_refusal(error: BaseException) -> None:
    """The refusal of an FP32 workspace is :data:`REFUSAL`, naming both
    modes' workspaces."""
    assert str(error) == REFUSAL, str(error)


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
    ws = torch.zeros(64, dtype=torch.uint8)
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
    ws = torch.zeros(64, dtype=torch.uint8)
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
    neither direction warns under ``torch.use_deterministic_algorithms(True)``
    (bodies stubbed: no device needed)."""
    calls: list = []

    def forward(x, z, w, eps, *, out):
        calls.append(("fwd", out))
        return x * 1

    def backward(dy, x, z, w, eps, **kwargs):
        calls.append(("bwd", kwargs))
        return dy * 1, dy * 1, w * 1

    monkeypatch.setattr(backend, "_forward", forward)
    monkeypatch.setattr(backend, "_backward", backward)
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
    ws = torch.ones(8, dtype=torch.uint8)
    for enabled, warn_only in ((False, False), (True, False), (True, True)):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws)
    assert resets == [(ws, {})] * 3


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
@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_no_call_alerts_under_deterministic_algorithms(
    warn_only, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under ``torch.use_deterministic_algorithms(True)``, strict or with
    ``warn_only=True``, every call runs with no warning and no error: the
    forward and the backward into a BF16 and an FP32 weight gradient at the
    first rows of the static and of the claimed backward past the small-row
    bands (``_backward_tiers``), and the autograd backward, with the weight
    gradient, of :func:`cake_rmsnorm_silu_gate_nondet`, of
    :class:`CakeRMSNormSiLUGateNondetFunction` and of a non-reentrant
    checkpoint.  The functional calls launch ``module_for_rows``' modules."""
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    least = _backward_tiers(arch, "bwd_bf16", sms)["static"][1]
    calls = []
    for role, dtype in (("bwd_bf16", torch.bfloat16), ("bwd_f32", torch.float32)):
        tiers = _backward_tiers(arch, role, sms)
        for tier in ("static", "claimed"):
            name, first, _ = tiers[tier]
            rows = max(first, least)
            assert loader.module_for_rows(arch, role, rows, sms) == name, (role, tier)
            calls.append((rows, dtype, name))
    inputs = {rows: suite.make(rows, 1, layout="flat", seed=21) for rows, _, _ in calls}
    x, z, w, dy = inputs[least]
    leaves = [[t.clone().requires_grad_(True) for t in (x, z, w)] for _ in range(3)]
    launched = _spy_launches(monkeypatch)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for rows, dtype, _ in calls:
            xr, zr, wr, dyr = inputs[rows]
            nd_forward(xr, zr, wr, EPS)
            nd_backward(
                dyr,
                xr,
                zr,
                wr,
                EPS,
                grad_weight=torch.empty(D, dtype=dtype, device=xr.device),
            )
        nd_gated_rmsnorm(*leaves[0], EPS).backward(dy)
        CakeRMSNormSiLUGateNondetFunction.apply(*leaves[1], EPS).backward(dy)
        torch.utils.checkpoint.checkpoint(
            nd_gated_rmsnorm, *leaves[2], EPS, use_reentrant=False
        ).backward(dy)
        torch.cuda.synchronize()
    assert [str(c.message) for c in caught] == []
    for leaf_set in leaves:
        assert all(t.grad is not None for t in leaf_set)
    want = []
    for rows, _, name in calls:
        want += [loader.module_for_rows(arch, "fwd", rows, sms), name]
    assert launched[: len(want)] == want, launched


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
        "cake_rmsnorm_silu_gate_nondet_train_forward": lambda **m: cake_rmsnorm_silu_gate_nondet_train_forward(x, z, w, EPS, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward": lambda **m: cake_rmsnorm_silu_gate_nondet_train_backward(dy, x, z, w, EPS, workspace=ws, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace": lambda **m: cake_rmsnorm_silu_gate_nondet_train_backward_workspace(T * H, device, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes": lambda **m: cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(T * H, device, **m),
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace": lambda **m: cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws, **m),
        "cake_rmsnorm_silu_gate_nondet": lambda **m: cake_rmsnorm_silu_gate_nondet(x, z, w, EPS, **m),
        "is_cake_rmsnorm_silu_gate_nondet_train_supported": lambda **m: is_cake_rmsnorm_silu_gate_nondet_train_supported(device, **m),
    }  # fmt: skip
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
    """The ``nondet`` entry points launch the backend's forward and backward
    modules that ``module_for_rows`` picks: ``y``, ``dx``, ``dz`` and ``dw``
    repeat bit for bit and lie within the error budget of the deterministic
    package's outputs, ``dx`` and ``dz`` are its bits, ``dw`` also within one
    BF16 rounding plus the FP32
    accumulation budget of the deterministic ``dw``.  The two workspace kinds
    do not substitute, and a refusal launches nothing: this backward refuses
    an FP32 tensor (the deterministic package's workspace), naming both modes'
    workspaces, and the deterministic backward refuses this package's
    ``torch.uint8`` workspace (``ValueError``)."""
    launched = _spy_launches(monkeypatch)
    T, H = 257, 48
    x, z, w, dy = _inputs(T, H, 12)
    y = cake_rmsnorm_silu_gate_nondet_train_forward(x, z, w, EPS)
    dx, dz, dw = cake_rmsnorm_silu_gate_nondet_train_backward(dy, x, z, w, EPS)
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    assert launched == [
        loader.module_for_rows(arch, "fwd", T * H, sms),
        loader.module_for_rows(arch, "bwd_bf16", T * H, sms),
    ], launched
    assert_same_bits("y", nd_forward(x, z, w, EPS), y)
    again = nd_backward(dy, x, z, w, EPS)
    for what, got, want in zip(("dx", "dz", "dw"), again, (dx, dz, dw), strict=True):
        assert_same_bits(what, got, want)
    y64, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
    dx_d, dz_d, dw_d = det_backward(dy, x, z, w, EPS)
    assert_det_bits("dx", dx, dx_d)
    assert_det_bits("dz", dz, dz_d)
    outputs = zip(
        ("y", "dx", "dz", "dw"),
        (y, dx, dz, dw),
        (det_forward(x, z, w, EPS), dx_d, dz_d, dw_d),
        (y64, dx64, dz64, dw64),
        strict=True,
    )
    for what, got, det, ref in outputs:
        assert_error_within_det(what, got, det, ref)
    terms = (dy.double() * torch.nn.functional.silu(z.double())) * (
        x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + EPS)
    )
    mag = terms.abs().reshape(-1, D).sum(0)
    err = (dw.double() - dw_d.double()).abs()
    assert (err <= dw_d.double().abs() * 2.0**-7 + mag * 2.0**-20).all()
    # an FP32 buffer (the deterministic package's workspace) is not a
    # workspace of the non-deterministic backward, nor the other way round
    launched.clear()
    fp32_workspace = torch.zeros(T * H, dtype=torch.float32, device=x.device)
    with pytest.raises(ValueError) as info:
        cake_rmsnorm_silu_gate_nondet_train_backward(
            dy, x, z, w, EPS, workspace=fp32_workspace
        )
    _assert_refusal(info.value)
    # the deterministic backward's requirement names its own dtype (FP32)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"workspace must be a contiguous FP32 tensor on {x.device} with at least "
        ),
    ):
        det_backward(dy, x, z, w, EPS, workspace=nd_workspace(T * H, x.device))
    assert launched == []


@requires_device
@pytest.mark.parametrize("shape", [(0, D), (0, 48, D)], ids=["0x128", "0x48x128"])
def test_backward_with_no_rows_refuses_the_deterministic_workspace(
    shape: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call with no rows refuses an FP32 tensor (the deterministic package's
    workspace) with the ``ValueError`` of a call with rows, naming both modes'
    workspaces, before it returns (``grad_weight`` keeps its value).  With
    this package's ``torch.uint8`` workspace it returns zero-sized ``dx`` and
    ``dz`` and a zero ``dw``, or keeps ``grad_weight`` when accumulating;
    nothing is launched and the workspace stays zero."""
    launched = _spy_launches(monkeypatch)
    empty = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    x, z, w, dy = _inputs(65, 48, 14)
    fp32_workspace = torch.zeros(65 * 48, dtype=torch.float32, device=x.device)
    ws = nd_workspace(0, x.device)
    assert ws.dtype == torch.uint8
    assert ws.numel() == nd_workspace_bytes(0, x.device)
    for module in (facade, backend):
        backward = module.cake_rmsnorm_silu_gate_nondet_train_backward
        with pytest.raises(ValueError) as with_rows:
            backward(dy, x, z, w, EPS, workspace=fp32_workspace)
        _assert_refusal(with_rows.value)
        main = torch.full((D,), 1.5, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError) as no_rows:
            backward(
                empty,
                empty,
                empty,
                w,
                EPS,
                grad_weight=main,
                workspace=fp32_workspace,
            )
        assert str(no_rows.value) == str(with_rows.value), str(no_rows.value)
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
    torch.cuda.synchronize()
    assert launched == []
    assert not bool(ws.any())


# --------------------------------------------------------------------------- within the deterministic package's error

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
def test_accuracy_matrix_within_the_deterministic_error(
    shape, make_kwargs, eps, dw_dtype, determinism_setting
) -> None:
    """``y`` repeats bit for bit and, on every one of ``suite.REPEATS``
    backward launches, ``dx``, ``dz`` and ``dw`` are the first launch's bits
    and ``dx`` and ``dz`` the deterministic package's; each output lies within
    the error budget of the deterministic package's output against the FP64
    reference, and ``dw`` within the weight-gradient budget of the
    deterministic ``dw``."""
    x, z, w, dy = suite.make(*shape, **make_kwargs)
    y64, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, eps)
    y = nd_forward(x, z, w, eps)
    assert_same_bits("y", nd_forward(x, z, w, eps), y)
    assert_error_within_det("y", y, det_forward(x, z, w, eps), y64)
    mag = suite.magnitudes(dy, x, z, eps)

    def out():
        return (
            None
            if dw_dtype == torch.bfloat16
            else torch.empty(D, dtype=dw_dtype, device=x.device)
        )

    dets = det_backward(dy, x, z, w, eps, grad_weight=out())
    first = None
    for _ in range(suite.REPEATS):
        got = nd_backward(dy, x, z, w, eps, grad_weight=out())
        assert got[2].dtype == dw_dtype
        if first is None:
            first = got
        for what, g, f in zip(("dx", "dz", "dw"), got, first, strict=True):
            assert_same_bits(what, g, f)
        assert_det_bits("dx", got[0], dets[0])
        assert_det_bits("dz", got[1], dets[1])
    for what, got, det, ref in zip(
        ("dx", "dz", "dw"), first, dets, (dx64, dz64, dw64), strict=True
    ):
        assert_error_within_det(what, got, det, ref)
    suite.assert_dw_within(
        first[2], dets[2], mag, what="dw vs the deterministic package"
    )


@requires_device
def test_accumulation_within_the_deterministic_error(determinism_setting) -> None:
    """Two microbatches accumulated into an FP32 and a BF16 buffer: the same
    starting buffer gives the same bits, within two roundings of the
    deterministic accumulation and within the error budget of the
    deterministic package's against the FP64 sum.  Autograd accumulation into
    ``.grad`` over two microbatches gives twice the functional backward's
    bits, ``x.grad`` and ``z.grad`` twice the deterministic package's ``dx``
    and ``dz`` (bits) and within the error budget of those."""
    x, z, w, dy = suite.make(1031, 48, seed=6)
    x2, z2, _, dy2 = suite.make(1031, 48, seed=7)
    _, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
    dw64b = suite.fp64_reference(dy2, x2, z2, w, EPS)[3]
    mag = suite.magnitudes(dy, x, z, EPS) + suite.magnitudes(dy2, x2, z2, EPS)
    first = 0.25 + dw64
    want = first + dw64b
    # Two roundings into the accumulator's dtype, each within its relative unit.
    bound = (first.abs() + want.abs()) * suite.BF16_RTOL + mag * suite.DW_MAG_TOL
    for dtype in (torch.float32, torch.bfloat16):
        mains = []
        for _ in range(2):
            main = torch.full((D,), 0.25, dtype=dtype, device="cuda")
            for args in ((dy, x, z), (dy2, x2, z2)):
                nd_backward(
                    *args, w, EPS, grad_weight=main, accumulate_grad_weight=True
                )
            mains.append(main)
        assert_same_bits(f"{dtype} accumulator", mains[1], mains[0])
        main_reference = torch.full((D,), 0.25, dtype=dtype, device="cuda")
        for args in ((dy, x, z), (dy2, x2, z2)):
            det_backward(
                *args, w, EPS, grad_weight=main_reference, accumulate_grad_weight=True
            )
        assert (
            (mains[0].double() - main_reference.double()).abs() <= 2 * bound
        ).all(), dtype
        assert_error_within_det(f"{dtype} accumulator", mains[0], main_reference, want)
    # autograd accumulation into .grad over two microbatches
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    nd_gated_rmsnorm(xa, za, wa, EPS).backward(dy)
    nd_gated_rmsnorm(xa, za, wa, EPS).backward(dy)
    dx, dz, dw = nd_backward(dy, x, z, w, EPS)
    assert_same_bits("x.grad", xa.grad, dx + dx)
    assert_same_bits("z.grad", za.grad, dz + dz)
    assert_same_bits("w.grad", wa.grad, dw + dw)
    dx_d, dz_d, _ = det_backward(dy, x, z, w, EPS)
    assert_det_bits("x.grad", xa.grad, dx_d + dx_d)
    assert_det_bits("z.grad", za.grad, dz_d + dz_d)
    assert_error_within_det("x.grad", xa.grad, dx_d + dx_d, 2 * dx64)
    assert_error_within_det("z.grad", za.grad, dz_d + dz_d, 2 * dz64)


@requires_device
def test_compute_grad_weight_false_within_the_deterministic_error(
    determinism_setting,
) -> None:
    """Without the weight gradient (with this package's workspace or none,
    and through autograd with a weight that needs no gradient), ``dx`` and
    ``dz`` are the bits of a backward with the weight gradient and the
    deterministic package's bits, within its error budget."""
    x, z, w, dy = suite.make(257, 48, seed=12)
    _, dx64, dz64, _ = suite.fp64_reference(dy, x, z, w, EPS)
    dx_d, dz_d, _ = det_backward(dy, x, z, w, EPS, compute_grad_weight=False)
    want_dx, want_dz, _ = nd_backward(dy, x, z, w, EPS)
    ws = nd_workspace(257 * 48, x.device)
    for workspace in (ws, None):
        dx, dz, dw = nd_backward(
            dy, x, z, w, EPS, compute_grad_weight=False, workspace=workspace
        )
        assert dw is None
        assert_same_bits("dx", dx, want_dx)
        assert_same_bits("dz", dz, want_dz)
        assert_det_bits("dx", dx, dx_d)
        assert_det_bits("dz", dz, dz_d)
    assert_error_within_det("dx", want_dx, dx_d, dx64)
    assert_error_within_det("dz", want_dz, dz_d, dz64)
    # autograd with a weight that needs no gradient
    xa, za = (t.clone().requires_grad_(True) for t in (x, z))
    nd_gated_rmsnorm(xa, za, w, EPS).backward(dy)
    assert_same_bits("x.grad", xa.grad, want_dx)
    assert_same_bits("z.grad", za.grad, want_dz)
    assert_det_bits("x.grad", xa.grad, dx_d)
    assert_det_bits("z.grad", za.grad, dz_d)


@requires_device
def test_reset_workspace_then_reuse_across_shapes(determinism_setting) -> None:
    """A workspace with nonzero bytes in every buffer (the integer windows,
    the class words, the arrival ticket and the claim counter) zeroed by
    ``cake_rmsnorm_silu_gate_nondet_train_reset_workspace``, then reused
    without a reset over the backend suite's sequence of shapes,
    weight-gradient dtypes and accumulation: every output repeats bit for bit
    (an accumulated ``dw`` from the same starting value too), ``dx`` and
    ``dz`` are the deterministic package's bits, ``dx``, ``dz`` and ``dw`` lie
    within the error budget of the deterministic package's,
    ``dw`` within the budget of the deterministic ``dw`` and of the FP64
    reference, and the workspace is zero after every call."""
    device = torch.device("cuda")
    ws = nd_workspace(1, device)
    assert ws.dtype == torch.uint8
    assert ws.numel() == nd_workspace_bytes(1, device)
    assert nd_workspace_bytes(1, device) == nd_workspace_bytes(10**7, device)
    generator = torch.Generator(device="cuda").manual_seed(70)
    ws.copy_(
        torch.randint(
            1, 256, ws.shape, dtype=torch.uint8, device=device, generator=generator
        )
    )
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
        case = f"T={T} H={H} {dtype}"

        def gw(start=start, dtype=dtype, accumulate=accumulate):
            if accumulate or dtype != torch.bfloat16:
                return torch.full((D,), start, dtype=dtype, device=device)
            return None

        runs = [
            nd_backward(
                dy,
                x,
                z,
                w,
                EPS,
                grad_weight=gw(),
                accumulate_grad_weight=accumulate,
                workspace=ws,
            )
            for _ in range(2)
        ]
        torch.cuda.synchronize()
        assert not bool(ws.any()), case
        for what, first, again in zip(("dx", "dz", "dw"), *runs, strict=True):
            assert_same_bits(f"{what} {case}", again, first)
        dx, dz, dw = runs[0]
        assert dw.dtype == dtype
        dets = det_backward(
            dy, x, z, w, EPS, grad_weight=gw(), accumulate_grad_weight=accumulate
        )
        assert_det_bits(f"dx {case}", dx, dets[0])
        assert_det_bits(f"dz {case}", dz, dets[1])
        mag = suite.magnitudes(dy, x, z, EPS)
        _, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
        want = dw64 + start
        for what, got, det, ref in zip(
            ("dx", "dz", "dw"), (dx, dz, dw), dets, (dx64, dz64, want), strict=True
        ):
            assert_error_within_det(f"{what} {case}", got, det, ref)
        suite.assert_dw_within(dw, want, mag, what=f"dw {case} vs FP64")
        suite.assert_dw_within(
            dw, dets[2], mag, what=f"dw {case} vs the deterministic package"
        )


@requires_device
def test_two_streams_repeat_the_current_stream_bits(determinism_setting) -> None:
    """Backwards on two streams, each with its own workspace, five times over:
    every ``dx``, ``dz`` and ``dw`` is the bits of the same backward on the
    current stream, which lies within the error budget of the deterministic
    package's, and every ``dx`` and ``dz`` the deterministic package's bits."""
    cases = [suite.make(4097, 48, seed=30), suite.make(1031, 16, seed=31)]
    refs = [nd_backward(dy, x, z, w, EPS) for x, z, w, dy in cases]
    dets = [det_backward(dy, x, z, w, EPS) for x, z, w, dy in cases]
    for (x, z, w, dy), ref, det_out in zip(cases, refs, dets, strict=True):
        _, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
        for what, got, det, want in zip(
            ("dx", "dz", "dw"), ref, det_out, (dx64, dz64, dw64), strict=True
        ):
            assert_error_within_det(what, got, det, want)
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
        for got in outputs[i]:
            for what, g, want in zip(("dx", "dz", "dw"), got, refs[i], strict=True):
                assert_same_bits(f"stream {i}: {what}", g, want)
            assert_det_bits(f"stream {i}: dx", got[0], dets[i][0])
            assert_det_bits(f"stream {i}: dz", got[1], dets[i][1])


@requires_device
def test_side_stream_repeats_the_current_stream_bits(determinism_setting) -> None:
    """The forward and the backward (with this package's workspace and
    without one) on a side stream give the bits of the same calls on the
    current stream, which lie within the error budget of the deterministic
    package's; ``dx`` and ``dz`` are the deterministic package's bits."""
    x, z, w, dy = suite.make(2049, 48, seed=9)
    want_y = nd_forward(x, z, w, EPS)
    want = nd_backward(dy, x, z, w, EPS)
    dets = det_backward(dy, x, z, w, EPS)
    y64, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
    assert_error_within_det("y", want_y, det_forward(x, z, w, EPS), y64)
    for what, got, det, ref in zip(
        ("dx", "dz", "dw"), want, dets, (dx64, dz64, dw64), strict=True
    ):
        assert_error_within_det(what, got, det, ref)
    ws = nd_workspace(x.shape[0] * x.shape[1], x.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = nd_forward(x, z, w, EPS)
        got = nd_backward(dy, x, z, w, EPS, workspace=ws)
        fresh = nd_backward(dy, x, z, w, EPS)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert_same_bits("y", y, want_y)
    for outputs in (got, fresh):
        for what, g, w_ in zip(("dx", "dz", "dw"), outputs, want, strict=True):
            assert_same_bits(what, g, w_)
        assert_det_bits("dx", outputs[0], dets[0])
        assert_det_bits("dz", outputs[1], dets[1])


@requires_device
def test_cuda_graph_replay_repeats_the_eager_bits(determinism_setting) -> None:
    """Replays of a captured forward and backward (this package's workspace,
    a BF16 ``dw``) on new inputs give the bits of eager calls on those inputs,
    which lie within the error budget of the deterministic package's; ``dx``
    and ``dz`` are the deterministic package's bits."""
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
    names = ("y", "dx", "dz", "dw")
    for k in range(3):
        xn, zn, _, dyn = suite.make(T, H, seed=40 + k)
        xs.copy_(xn)
        zs.copy_(zn)
        dys.copy_(dyn)
        graph.replay()
        torch.cuda.synchronize()
        eager = (nd_forward(xn, zn, w, EPS), *nd_backward(dyn, xn, zn, w, EPS))
        for what, got, want in zip(names, (y, dx, dz, dw), eager, strict=True):
            assert_same_bits(f"replay {k}: {what}", got, want)
        dets = (det_forward(xn, zn, w, EPS), *det_backward(dyn, xn, zn, w, EPS))
        assert_det_bits(f"replay {k}: dx", dx, dets[1])
        assert_det_bits(f"replay {k}: dz", dz, dets[2])
        refs = suite.fp64_reference(dyn, xn, zn, w, EPS)
        for what, got, det, ref in zip(names, eager, dets, refs, strict=True):
            assert_error_within_det(f"replay {k}: {what}", got, det, ref)


@requires_device
def test_autograd_and_checkpoint_repeat_the_functional_bits(
    determinism_setting,
) -> None:
    """Autograd through :func:`cake_rmsnorm_silu_gate_nondet`, through
    :class:`CakeRMSNormSiLUGateNondetFunction` and through a non-reentrant
    checkpoint gives the functional entry points' bits (``y``, ``dx``, ``dz``
    and a BF16 ``dw``), which lie within the error budget of the deterministic
    package's outputs, ``dx`` and ``dz`` its bits, ``dw`` also within the
    weight-gradient budget of the deterministic ``dw``."""
    x, z, w, dy = suite.make(513, 48, seed=8)
    want = (nd_forward(x, z, w, EPS), *nd_backward(dy, x, z, w, EPS))
    dets = (det_forward(x, z, w, EPS), *det_backward(dy, x, z, w, EPS))
    names = ("y", "dx", "dz", "dw")
    for what, got, det, ref in zip(
        names, want, dets, suite.fp64_reference(dy, x, z, w, EPS), strict=True
    ):
        assert_error_within_det(what, got, det, ref)
    mag = suite.magnitudes(dy, x, z, EPS)
    suite.assert_dw_within(
        want[3], dets[3], mag, what="dw vs the deterministic package"
    )

    def check(y, xg, zg, wg, what):
        assert wg.dtype == torch.bfloat16, what
        for name, got, w_ in zip(names, (y, xg, zg, wg), want, strict=True):
            assert_same_bits(f"{what}: {name}", got, w_)
        assert_det_bits(f"{what}: dx", xg, dets[1])
        assert_det_bits(f"{what}: dz", zg, dets[2])

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


# --------------------------------------------------------------------------- the forward's bands and the backward's programs


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


def _start_rows(arch: str, name: str, sms: int, packed: bool = True) -> int:
    """The fewest rows the routed module ``name`` runs for on ``arch`` with
    ``sms`` SMs: for a forward module the first row of its first
    ``FORWARD_BANDS`` band (``STRIDED_FORWARD_BANDS`` where the rows are not
    ``packed``; one more than the previous band's ``last * SMs``), for the
    claimed backward the first row of more than ``more_chunks_than_per_sm *
    SMs`` chunks of one block (its grid rule)."""
    rule = loader.MODULES[name]["grid_rule"]
    if "more_chunks_than_per_sm" in rule:
        chunks = int(rule["more_chunks_than_per_sm"]) * sms
        return chunks * int(rule["rows_per_block"]) + 1
    role = loader.MODULES[name]["role"]
    first = 1
    table = loader.FORWARD_BANDS if packed else loader.STRIDED_FORWARD_BANDS
    for last, band in table[arch]:
        if band == role:
            return first
        assert last is not None, (arch, role)
        first = int(last) * sms + 1
    raise AssertionError(f"{arch} has no forward band of {role}")


def _backward_tiers(
    arch: str, role: str, sms: int
) -> dict[str, tuple[str, int, int | None]]:
    """``tier -> (module, first rows, last rows or None)`` of the backward
    ``role`` (``bwd_bf16`` or ``bwd_f32``) on ``arch`` with ``sms`` SMs, from
    the loader's routes, bands and grid rules: ``band<i>`` the module of the
    role's ``i``-th ``BACKWARD_BANDS`` band (a small-row backward) over its
    rows, the static backward from the next row while the rows have at most
    ``more_chunks_than_per_sm`` chunks per SM, and the claimed backward past
    that, with no last row."""
    routes = loader.ROUTES[arch]
    tiers: dict[str, tuple[str, int, int | None]] = {}
    first = 1
    for i, (last, band) in enumerate(loader.BACKWARD_BANDS.get(arch, {}).get(role, ())):
        tiers[f"band{i}"] = (routes[band], first, int(last))
        first = int(last) + 1
    claimed = routes[f"{role}_mid"]
    start = _start_rows(arch, claimed, sms)
    tiers["static"] = (routes[role], first, start - 1)
    tiers["claimed"] = (claimed, start, None)
    return tiers


# Forward inputs around and past the ends of the forward's bands: (tokens,
# heads, layout); ``heads=None`` takes ``last * SMs + tokens`` rows of one
# head, ``last`` the rows per SM of the architecture's ``FORWARD_BANDS`` band
# ``i`` of ``layout`` ``flat<i>`` (packed rows: one flat [rows, 128] tensor) or
# of its ``STRIDED_FORWARD_BANDS`` band ``i`` of ``layout`` ``strided<i>``
# (the projection layout, whose ``z`` rows are not packed), skipped where the
# architecture has no such band with an end.
FORWARD_CASES = [
    pytest.param(1, 48, "contiguous", id="1x48"),
    pytest.param(13, 7, "contiguous", id="13x7"),
    pytest.param(13, 48, "projection", id="13x48-projection"),
    pytest.param(21, 48, "head_major", id="21x48-head_major"),
    pytest.param(0, None, "flat0", id="band0-end"),
    pytest.param(1, None, "flat0", id="band0-end+1"),
    pytest.param(0, None, "flat1", id="band1-end"),
    pytest.param(1, None, "flat1", id="band1-end+1"),
    pytest.param(0, None, "strided0", id="strided-band0-end"),
    pytest.param(1, None, "strided0", id="strided-band0-end+1"),
    pytest.param(0, None, "strided1", id="strided-band1-end"),
    pytest.param(1, None, "strided1", id="strided-band1-end+1"),
    pytest.param(2048, 48, "contiguous", id="2048x48"),
    pytest.param(4097, 48, "contiguous", id="4097x48"),
    pytest.param(4097, 48, "projection", id="4097x48-projection"),
    pytest.param(4097, 48, "head_major", id="4097x48-head_major"),
    pytest.param(6000, 32, "contiguous", id="6000x32"),
    pytest.param(0, None, "flat2", id="band2-end"),
    pytest.param(1, None, "flat2", id="band2-end+1"),
    pytest.param(-1, None, "flat3", id="band3-end-1"),
    pytest.param(0, None, "flat3", id="band3-end"),
    pytest.param(1, None, "flat3", id="band3-end+1"),
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
@pytest.mark.parametrize("tokens,heads,layout", FORWARD_CASES)
def test_forward_around_its_bands_within_the_deterministic_error(
    tokens, heads, layout, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On both sides of the ends of the forward's bands and past them, the
    forward launches ``module_for_rows``' module: the module of the first
    ``FORWARD_BANDS`` band of the architecture (``STRIDED_FORWARD_BANDS``
    band where the rows are not packed) with ``rows <= last * SMs`` (the
    tiny, short, mid, long and extra-long forwards and the forward, each where
    the architecture's bands name it).  ``y`` repeats bit for bit and lies
    within the error budget of the deterministic package's ``y``."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    routes = loader.ROUTES[arch]
    if heads is None:
        kind = "strided" if layout.startswith("strided") else "flat"
        i = int(layout[len(kind) :])
        layout = "projection" if kind == "strided" else "flat"
        table = (
            loader.STRIDED_FORWARD_BANDS if kind == "strided" else loader.FORWARD_BANDS
        )
        if i >= len(table[arch]) or table[arch][i][0] is None:
            pytest.skip(f"{arch} has no {kind} forward band {i} with an end")
        tokens += int(table[arch][i][0]) * sms
        heads = 1
    x, z, w, dy = suite.make(tokens, heads, layout=layout, seed=21)
    rows = x.numel() // D
    packed = suite._packed(tokens, heads, layout)
    bands = (loader.FORWARD_BANDS if packed else loader.STRIDED_FORWARD_BANDS)[arch]
    launched = _spy_launches(monkeypatch)
    y = nd_forward(x, z, w, EPS)
    assert_same_bits("y", nd_forward(x, z, w, EPS), y)
    name = loader.module_for_rows(arch, "fwd", rows, sms, packed=packed)
    assert launched == [name] * 2, launched
    want = next(
        routes[band] for last, band in bands if last is None or rows <= last * sms
    )
    assert name == want, (arch, rows, sms)
    y64 = suite.fp64_reference(dy, x, z, w, EPS)[0]
    assert_error_within_det("y", y, det_forward(x, z, w, EPS), y64)


# Gate inputs at and past the edges of the packed gate's range: below -41,
# around FP32 expf's overflow (|z| from 87.5 to 89), subnormal and zero,
# infinite and NaN, beside saturated and ordinary ones.
GATE_EDGE_VALUES = [
    -1e4,
    -100.0,
    -89.0,
    -88.5,
    -88.0,
    -87.5,
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
    87.5,
    88.0,
    89.0,
    100.0,
    1e4,
    float("inf"),
    float("-inf"),
    float("nan"),
]
# Forward inputs (``x`` and ``z`` packed rows, or ``z`` the gate columns of a
# wider projection) the tiny or the short forward runs, and those at the end
# of the short forward's band on sm_100a and sm_103a, 12 rows per SM, on 148 SMs (37x48, 253x7, 55x32) and on
# 152 SMs (38x48, 260x7, 57x32), and on sm_107a, 59 rows per SM, on 212 SMs
# (260x48, 1786x7, 390x32), and one token past it (261x48, 1787x7, 391x32),
# with forward inputs at and past 48 rows per SM on 148 and 152 SMs (148x48,
# 1014x7, 222x32, 152x48, 1042x7, 228x32 and one token more), which the mid
# forward runs there.
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
    pytest.param(148, 48, id="148x48"),
    pytest.param(149, 48, id="149x48"),
    pytest.param(152, 48, id="152x48"),
    pytest.param(153, 48, id="153x48"),
    pytest.param(1014, 7, id="1014x7"),
    pytest.param(1015, 7, id="1015x7"),
    pytest.param(1042, 7, id="1042x7"),
    pytest.param(1043, 7, id="1043x7"),
    pytest.param(222, 32, id="222x32"),
    pytest.param(223, 32, id="223x32"),
    pytest.param(228, 32, id="228x32"),
    pytest.param(229, 32, id="229x32"),
    pytest.param(260, 48, id="260x48"),
    pytest.param(261, 48, id="261x48"),
    pytest.param(1786, 7, id="1786x7"),
    pytest.param(1787, 7, id="1787x7"),
    pytest.param(390, 32, id="390x32"),
    pytest.param(391, 32, id="391x32"),
]


def _gate_edge_inputs(tokens: int, heads: int, columns: int = D):
    """``(x, z, w, dy)`` of ``suite.make`` with gate inputs from
    ``GATE_EDGE_VALUES`` in a quarter of the elements of the first
    ``columns`` columns of two of every three rows (so the packed gate's
    range check fails for some threads of a row and holds for others) and
    ordinary ones in the rest."""
    x, z, w, dy = suite.make(tokens, heads, seed=23)
    g = torch.Generator(device=z.device).manual_seed(24)
    values = torch.tensor(GATE_EDGE_VALUES, device=z.device).bfloat16()
    pick = torch.randint(
        0, len(GATE_EDGE_VALUES), z.shape, device=z.device, generator=g
    )
    edge = torch.rand(z.shape, device=z.device, generator=g) < 0.25
    rows = torch.arange(tokens * heads, device=z.device).view(tokens, heads, 1)
    edge &= rows % 3 != 0
    edge &= torch.arange(D, device=z.device) < columns
    z.copy_(torch.where(edge, values[pick], z))
    return x, z, w, dy


@requires_device
@pytest.mark.parametrize("layout", ["contiguous", "projection"])
@pytest.mark.parametrize("tokens,heads", GATE_EDGE_CASES)
def test_forward_gate_edges_keep_the_fp64_classes(
    tokens, heads, layout, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With gate inputs at and past the packed gate's range in two of every
    three rows (a quarter of their elements, so the range check fails for some
    threads of a row and holds for others) and ordinary ones in the rest, the
    forward launches ``module_for_rows``' module and ``y`` repeats bit for bit,
    has the FP64 reference's ``NaN``, ``+inf`` and ``-inf`` positions, and its
    finite elements lie within the error budget of the deterministic
    package's."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    x, z, w, dy = _gate_edge_inputs(tokens, heads)
    if layout == "projection":
        wide = torch.zeros(tokens, 2 * heads * D + 16, dtype=z.dtype, device=z.device)
        view = wide[:, heads * D : 2 * heads * D].view(tokens, heads, D)
        view.copy_(z)
        z = view
    launched = _spy_launches(monkeypatch)
    y = nd_forward(x, z, w, EPS)
    assert_same_bits("y", nd_forward(x, z, w, EPS), y)
    packed = suite._packed(tokens, heads, layout)
    name = loader.module_for_rows(arch, "fwd", tokens * heads, sms, packed=packed)
    assert launched == [name] * 2, launched
    y64 = suite.fp64_reference(dy, x, z, w, EPS)[0]
    assert_same_classes("y", y, y64)
    assert_error_within_det(
        "finite y", y, det_forward(x, z, w, EPS), y64, mask=torch.isfinite(y64)
    )


# Backward inputs with the forward's gate edges (``_gate_edge_inputs``) in the
# first GATE_EDGE_BACKWARD_COLUMNS columns, so every output keeps finite
# elements (``dw`` in the other columns): 48 and 240 rows (the small-row
# band), 1008 rows (the static backward past it) and 6192 and 49488 rows (the
# static or the claimed backward, by the SM count).
GATE_EDGE_BACKWARD_COLUMNS = 96
GATE_EDGE_BACKWARD_CASES = [
    pytest.param(1, 48, id="1x48"),
    pytest.param(5, 48, id="5x48"),
    pytest.param(21, 48, id="21x48"),
    pytest.param(129, 48, id="129x48"),
    pytest.param(1031, 48, id="1031x48"),
]


@requires_device
@pytest.mark.parametrize("tokens,heads", GATE_EDGE_BACKWARD_CASES)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_backward_gate_edges_keep_the_fp64_classes(
    tokens, heads, dw_dtype, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the forward's gate edges in the first
    ``GATE_EDGE_BACKWARD_COLUMNS`` columns (``_gate_edge_inputs``) the
    backward launches ``module_for_rows``' module, every output repeats bit
    for bit, ``dx`` and ``dz`` have the deterministic package's ``NaN``
    positions and its bits elsewhere, ``dx``, ``dz`` and ``dw`` have the FP64
    reference's ``NaN``, ``+inf`` and ``-inf`` positions, their finite
    elements (each has some) lie within the error budget of the
    deterministic package's, and the workspace is zero after every call."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    x, z, w, dy = _gate_edge_inputs(tokens, heads, GATE_EDGE_BACKWARD_COLUMNS)
    rows = tokens * heads
    _, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x.device)

    dets = det_backward(dy, x, z, w, EPS, grad_weight=out())
    ws = nd_workspace(rows, x.device)
    launched = _spy_launches(monkeypatch)
    got = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
    assert not bool(ws.any()), "the workspace is not zero"
    again = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
    assert not bool(ws.any()), "the workspace is not zero"
    name = loader.module_for_rows(arch, role, rows, sms)
    assert launched == [name] * 2, launched
    assert got[2].dtype == dw_dtype
    for what, g, a in zip(("dx", "dz", "dw"), got, again, strict=True):
        assert_same_bits(what, a, g)
    assert_det_bits_beside_nans("dx", got[0], dets[0])
    assert_det_bits_beside_nans("dz", got[1], dets[1])
    for what, g, det, ref in zip(
        ("dx", "dz", "dw"), got, dets, (dx64, dz64, dw64), strict=True
    ):
        assert_same_classes(what, g, ref)
        finite = torch.isfinite(ref)
        assert bool(finite.any()), f"{what}: the FP64 reference has no finite element"
        assert_error_within_det(f"finite {what}", g, det, ref, mask=finite)


def _check_backward(x, z, w, dy, dw_dtype, name, monkeypatch) -> None:
    """``suite.REPEATS`` backward launches of module ``name`` into a
    ``dw_dtype`` weight gradient with this package's workspace, then two
    accumulating ones from the same starting value: every output is the
    first launch's bits (the accumulated ``dw`` too), ``dx`` and ``dz`` the
    deterministic package's bits, and the workspace is zero after every call;
    ``dx``, ``dz``, ``dw`` and the accumulated ``dw`` lie
    within the error budget of the deterministic package's, ``dw`` within the
    weight-gradient budget of the deterministic ``dw`` and the accumulated one
    within that of the FP64 reference."""
    rows = x.numel() // D
    mag = suite.magnitudes(dy, x, z, EPS)
    _, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)

    def out(value=None):
        if value is None:
            return torch.empty(D, dtype=dw_dtype, device=x.device)
        return torch.full((D,), value, dtype=dw_dtype, device=x.device)

    dets = det_backward(dy, x, z, w, EPS, grad_weight=out())
    acc_d = det_backward(
        dy, x, z, w, EPS, grad_weight=out(0.25), accumulate_grad_weight=True
    )[2]
    ws = nd_workspace(rows, x.device)
    launched = _spy_launches(monkeypatch)
    first = None
    for _ in range(suite.REPEATS):
        got = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
        assert got[2].dtype == dw_dtype
        if first is None:
            first = got
        for what, g, f in zip(("dx", "dz", "dw"), got, first, strict=True):
            assert_same_bits(what, g, f)
        assert_det_bits("dx", got[0], dets[0])
        assert_det_bits("dz", got[1], dets[1])
        assert not bool(ws.any()), "the workspace is not zero"
    for what, got, det, ref in zip(
        ("dx", "dz", "dw"), first, dets, (dx64, dz64, dw64), strict=True
    ):
        assert_error_within_det(what, got, det, ref)
    suite.assert_dw_within(
        first[2], dets[2], mag, what="dw vs the deterministic package"
    )
    acc = [
        nd_backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=out(0.25),
            accumulate_grad_weight=True,
            workspace=ws,
        )[2]
        for _ in range(2)
    ]
    assert not bool(ws.any()), "the workspace is not zero"
    assert_same_bits("accumulated dw", acc[1], acc[0])
    assert_error_within_det("accumulated dw", acc[0], acc_d, 0.25 + dw64)
    suite.assert_dw_within(acc[0], 0.25 + dw64, mag, what="accumulated dw vs FP64")
    assert launched == [name] * (suite.REPEATS + 2), launched


# The most small-row bands any architecture routes for a backward role
# (``loader.BACKWARD_BANDS``), so no tier below is a band no architecture has.
BACKWARD_BAND_COUNT = max(
    len(bands) for roles in loader.BACKWARD_BANDS.values() for bands in roles.values()
)
# The backward's small-row bands (``_backward_tiers``: ``band0`` ... up to
# ``BACKWARD_BAND_COUNT``), then its static and claimed backward.
BACKWARD_TIERS = [
    *(f"band{i}" for i in range(BACKWARD_BAND_COUNT)),
    "static",
    "claimed",
]
# The edges of the backward's programs on this device (``_backward_tiers``):
# (tier, edge).  ``chunks`` is the claimed backward's last rows of one-block
# chunks (``rows_per_block * chunk_limit``), ``chunks+1`` its first rows of
# two-block chunks.
BACKWARD_TIER_EDGES = [
    *(
        pytest.param(f"band{i}", edge, id=f"band{i}-{edge}")
        for i in range(BACKWARD_BAND_COUNT)
        for edge in ("first", "last")
    ),
    pytest.param("static", "first", id="static-first"),
    pytest.param("static", "last", id="static-last"),
    pytest.param("claimed", "first", id="claimed-first"),
    pytest.param("claimed", "chunks", id="claimed-one-block-chunks-last"),
    pytest.param("claimed", "chunks+1", id="claimed-two-block-chunks-first"),
]


@requires_device
@pytest.mark.parametrize("tier,edge", BACKWARD_TIER_EDGES)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_backward_tier_edges_within_the_deterministic_error(
    tier, edge, dw_dtype, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At the first and the last rows of each backward program on this
    device's SM count (``_backward_tiers``: the small-row backwards in their
    bands, the static backward while the rows
    have at most ``more_chunks_than_per_sm`` chunks per SM, the claimed
    backward past that) and where the claimed backward's chunks grow from one
    block to two (``backward_chunks``): ``module_for_rows`` runs the program
    from its first to its last rows only, and ``_check_backward`` holds on
    flat [rows, 128] inputs."""
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    tiers = _backward_tiers(arch, role, sms)
    if tier not in tiers:
        pytest.skip(f"{arch} routes no {tier} {role} backward")
    name, first, last = tiers[tier]
    if edge == "first":
        rows = first
    elif edge == "last":
        rows = last
    else:
        rule = loader.MODULES[name]["grid_rule"]
        blocks_per_chunk = 2 if edge == "chunks+1" else 1
        rows = int(rule["rows_per_block"]) * int(rule["chunk_limit"])
        rows += blocks_per_chunk - 1
        assert loader.backward_chunks(name, rows)[0] == blocks_per_chunk
    assert loader.module_for_rows(arch, role, rows, sms) == name, (rows, sms)
    if edge == "first" and first > 1:
        assert loader.module_for_rows(arch, role, rows - 1, sms) != name, rows
    if edge == "last":
        assert loader.module_for_rows(arch, role, rows + 1, sms) != name, rows
    x, z, w, dy = suite.make(rows, 1, layout="flat", seed=22)
    _check_backward(x, z, w, dy, dw_dtype, name, monkeypatch)


# Backward inputs of every program and layout: (tokens, heads, layout).
BACKWARD_SHAPES = [
    pytest.param(1, 48, "contiguous", id="1x48"),
    pytest.param(5, 48, "projection", id="5x48-projection"),
    pytest.param(13, 7, "head_major", id="13x7-head_major"),
    pytest.param(256, 1, "flat", id="256x1"),
    pytest.param(257, 1, "flat", id="257x1"),
    pytest.param(1031, 48, "contiguous", id="1031x48"),
    pytest.param(2048, 48, "contiguous", id="2048x48"),
    pytest.param(2701, 48, "contiguous", id="2701x48"),
    pytest.param(3091, 48, "projection", id="3091x48-projection"),
    pytest.param(3157, 48, "head_major", id="3157x48-head_major"),
    pytest.param(4097, 48, "projection", id="4097x48-projection"),
    pytest.param(4097, 48, "head_major", id="4097x48-head_major"),
    pytest.param(5300, 32, "contiguous", id="5300x32"),
    pytest.param(9600, 48, "contiguous", id="9600x48"),
    pytest.param(25000, 48, "contiguous", id="25000x48"),
    pytest.param(32737, 48, "contiguous", id="32737x48"),
]


@requires_device
@pytest.mark.parametrize("tokens,heads,layout", BACKWARD_SHAPES)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_backward_shapes_within_the_deterministic_error(
    tokens,
    heads,
    layout,
    dw_dtype,
    determinism_setting,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inputs of several shapes and layouts: ``module_for_rows`` runs one of
    the role's programs (``_backward_tiers``) and ``_check_backward``
    holds."""
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    x, z, w, dy = suite.make(tokens, heads, layout=layout, seed=24)
    rows = x.numel() // D
    name = loader.module_for_rows(arch, role, rows, sms)
    programs = {module for module, _, _ in _backward_tiers(arch, role, sms).values()}
    assert name in programs, (name, programs)
    _check_backward(x, z, w, dy, dw_dtype, name, monkeypatch)


@requires_device
@pytest.mark.parametrize("tier", BACKWARD_TIERS)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_backward_with_infinite_grad_out_keeps_the_fp64_classes(
    tier, dw_dtype, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Infinite ``grad_out`` in rows that are not the last, on each backward
    program (``_backward_tiers``: each small-row band's last rows, the static
    backward's last rows, 4097 rows past the claimed backward's first): the
    weight gradient of the columns holding them is not finite and has the
    FP64 reference's class (an infinity its sign, a ``NaN`` a ``NaN``), every
    other column lies within the error budget of the deterministic package's
    and within the weight-gradient budget of the deterministic ``dw``, and
    ``dx`` and ``dz`` have the FP64 reference's ``NaN``, ``+inf`` and ``-inf``
    positions, the deterministic package's ``NaN`` positions and its bits
    everywhere else, and their finite elements lie within the error budget,
    on every repeat the same bits, with the workspace zero after every
    call."""
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    tiers = _backward_tiers(arch, role, sms)
    if tier not in tiers:
        pytest.skip(f"{arch} routes no {tier} {role} backward")
    name, first, last = tiers[tier]
    rows = first + 4097 if last is None else last
    assert loader.module_for_rows(arch, role, rows, sms) == name
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
    _, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
    assert torch.equal(~torch.isfinite(dw64), cols), "FP64 dw columns"
    mag = suite.magnitudes(dy, x, z, EPS)

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x.device)

    dets = det_backward(dy, x, z, w, EPS, grad_weight=out())
    ws = nd_workspace(rows, x.device)
    launched = _spy_launches(monkeypatch)
    first_run = None
    for _ in range(suite.REPEATS):
        got = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
        assert got[2].dtype == dw_dtype
        if first_run is None:
            first_run = got
        for what, g, f in zip(("dx", "dz", "dw"), got, first_run, strict=True):
            assert_same_bits(what, g, f)
        assert_det_bits_beside_nans("dx", got[0], dets[0])
        assert_det_bits_beside_nans("dz", got[1], dets[1])
        assert not bool(ws.any()), "the workspace is not zero"
    for what, got, det, ref in zip(
        ("dx", "dz", "dw"), first_run, dets, (dx64, dz64, dw64), strict=True
    ):
        assert_same_classes(what, got, ref)
        assert_error_within_det(
            f"finite {what}", got, det, ref, mask=torch.isfinite(ref)
        )
    dw, dw_d = first_run[2], dets[2]
    suite.assert_dw_within(dw[~cols], dw_d[~cols], mag[~cols], what="finite dw columns")
    assert launched == [name] * suite.REPEATS, launched


def _block_rows(name: str) -> int:
    """The rows of one row block of the backward module ``name``: 8 for the
    exact tiny backward's teams and the static and the claimed backward."""
    assert name in loader.MODULES, name
    return 8


def _last_block_rows(block: int, first: int, last: int | None) -> dict[str, int]:
    """Two row counts of a program of ``block``-row blocks over ``first`` to
    ``last`` rows that are not a multiple of its block: ``rem1`` one row in
    the last block (the smallest such count from ``2 * block + 1`` rows and
    ``first``) and ``remmax`` one row short of a whole block (the largest up
    to ``last``, or 4096 rows past ``rem1`` for the claimed backward, which
    has no last row)."""
    lo = max(first, 2 * block + 1)
    low = lo + (1 - lo) % block
    high = low + 4096 + block - 2 if last is None else last - (last + 1) % block
    return {"rem1": low, "remmax": high}


@requires_device
@pytest.mark.parametrize("count", ["rem1", "remmax"])
@pytest.mark.parametrize("tier", BACKWARD_TIERS)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_backward_with_infinite_grad_out_in_the_last_row_keeps_the_fp64_classes(
    count, tier, dw_dtype, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Infinite ``grad_out`` in the last row, at row counts that are not a
    multiple of the program's row block (``_last_block_rows``), on each
    backward program (``_backward_tiers``): ``+inf`` in column 3, ``-inf`` in
    column 40, and ``+inf`` in column 77 beside ``-inf`` in the row before.
    The weight gradient of those columns has the FP64 reference's class
    (``+inf``, ``-inf`` and ``NaN``), every other column lies within the error
    budget of the deterministic package's and within the weight-gradient
    budget of the deterministic ``dw``, and ``dx`` and ``dz`` have the FP64
    reference's ``NaN``, ``+inf`` and ``-inf`` positions, the deterministic
    package's ``NaN`` positions and its bits everywhere else, on every repeat
    the same bits, with the workspace zero after every call."""
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    tiers = _backward_tiers(arch, role, sms)
    if tier not in tiers:
        pytest.skip(f"{arch} routes no {tier} {role} backward")
    name, first, last = tiers[tier]
    block = _block_rows(name)
    rows = _last_block_rows(block, first, last)[count]
    if not first <= rows <= (last if last is not None else rows):
        pytest.skip(f"{tier} ({first}-{last} rows) has no such row count")
    assert rows % block != 0, rows
    assert loader.module_for_rows(arch, role, rows, sms) == name, (rows, sms)
    x, z, w, dy = suite.make(rows, 1, layout="flat", seed=28)
    # (row, column, sign of the weight-gradient term): a term dy * silu(z) * x
    # * rstd has dy's sign times that of z * x, so dy takes the product
    for r, c, sign in (
        (rows - 1, 3, 1.0),
        (rows - 1, 40, -1.0),
        (rows - 1, 77, 1.0),
        (rows - 2, 77, -1.0),
    ):
        zx = float(z[r, c]) * float(x[r, c])
        assert zx != 0.0, (r, c)
        dy[r, c] = math.copysign(float("inf"), sign * zx)
    cols = torch.zeros(D, dtype=torch.bool, device=x.device)
    cols[[3, 40, 77]] = True
    _, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
    assert torch.equal(~torch.isfinite(dw64), cols), "FP64 dw columns"
    assert bool(torch.isposinf(dw64[3])), dw64[3]
    assert bool(torch.isneginf(dw64[40])), dw64[40]
    assert bool(torch.isnan(dw64[77])), dw64[77]
    mag = suite.magnitudes(dy, x, z, EPS)

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x.device)

    dets = det_backward(dy, x, z, w, EPS, grad_weight=out())
    ws = nd_workspace(rows, x.device)
    launched = _spy_launches(monkeypatch)
    first_run = None
    for _ in range(suite.REPEATS):
        got = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
        assert got[2].dtype == dw_dtype
        if first_run is None:
            first_run = got
        for what, g, f in zip(("dx", "dz", "dw"), got, first_run, strict=True):
            assert_same_bits(what, g, f)
        assert_det_bits_beside_nans("dx", got[0], dets[0])
        assert_det_bits_beside_nans("dz", got[1], dets[1])
        assert not bool(ws.any()), "the workspace is not zero"
    for what, got, det, ref in zip(
        ("dx", "dz", "dw"), first_run, dets, (dx64, dz64, dw64), strict=True
    ):
        assert_same_classes(f"{what} rows={rows}", got, ref)
        assert_error_within_det(
            f"finite {what}", got, det, ref, mask=torch.isfinite(ref)
        )
    dw, dw_d = first_run[2], dets[2]
    suite.assert_dw_within(dw[~cols], dw_d[~cols], mag[~cols], what="finite dw columns")
    assert launched == [name] * suite.REPEATS, launched


# --------------------------------------------------------------------------- one head: packed rows


def _packed(strides: dict) -> bool:
    """The kernels' packed-rows test over the strides of one launch: every
    input's head stride is 128 and its token stride ``heads * 128``.  The mid
    and the extra-long forward warm row blocks in L2 with a bulk prefetch only
    when it holds."""
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


# The modules that warm rows in L2, by role: the mid and the extra-long
# forward (row blocks of packed rows) and the claimed backward (the rows of
# the block two ahead).  The one-head inputs take one row past each one's
# start (``_start_rows``).
ONE_HEAD_ROUTES = [
    pytest.param("fwd", "fwd_mid", id="fwd_mid"),
    pytest.param("fwd", "fwd_xlong", id="fwd_xlong"),
    pytest.param("bwd_bf16", "bwd_bf16_mid", id="bwd_bf16_claimed"),
    pytest.param("bwd_f32", "bwd_f32_mid", id="bwd_f32_claimed"),
]


@requires_device
@pytest.mark.parametrize("role,tier", ONE_HEAD_ROUTES)
def test_one_head_inputs_give_the_same_bits_for_every_layout(
    role, tier, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One row past the start of the mid and the extra-long forward and of the
    claimed backward: a one-head input of every layout (``ONE_HEAD_LAYOUTS``)
    launches ``module_for_rows``' module with head stride 128, passes the
    packed-rows test exactly when its rows follow one another (or the host
    copied it), and returns the same ``y``, or ``dx``, ``dz`` and ``dw``, bits
    for every layout and repeat, with the workspace zero after every call,
    ``dx`` and ``dz`` the deterministic package's bits on the flat rows; those
    lie within the error budget of the deterministic package's on the flat
    rows, ``dw`` also within the weight-gradient budget of the deterministic
    ``dw``."""
    arch = loader.device_arch(torch.cuda.current_device())
    name = loader.ROUTES[arch].get(tier)
    if name is None:
        pytest.skip(f"{arch} routes no {tier} module")
    sms = _sm_count()
    rows = _start_rows(arch, name, sms) + 1
    assert loader.module_for_rows(arch, role, rows, sms) == name
    x0, z0, w, dy0 = suite.make(rows, 1, layout="flat", seed=25)
    dw_dtype = torch.float32 if role == "bwd_f32" else torch.bfloat16
    y64, dx64, dz64, dw64 = suite.fp64_reference(dy0, x0, z0, w, EPS)

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x0.device)

    if role == "fwd":
        names = ("y",)
        dets = (det_forward(x0, z0, w, EPS),)
        refs = (y64,)
        calls = 2
    else:
        names = ("dx", "dz", "dw")
        dets = det_backward(dy0, x0, z0, w, EPS, grad_weight=out())
        refs = (dx64, dz64, dw64)
        calls = suite.REPEATS
        ws = nd_workspace(rows, x0.device)
    launched = _spy_strides(monkeypatch)
    want = None
    for layout, (packed, _) in ONE_HEAD_LAYOUTS.items():
        x, z, dy = (_one_head(t, layout) for t in (x0, z0, dy0))
        launched.clear()
        for _ in range(calls):
            if role == "fwd":
                got = (nd_forward(x, z, w, EPS).reshape(rows, D),)
            else:
                dx, dz, dw = nd_backward(
                    dy, x, z, w, EPS, grad_weight=out(), workspace=ws
                )
                got = (dx.reshape(rows, D), dz.reshape(rows, D), dw)
                assert_det_bits(f"{layout}: dx", got[0], dets[0])
                assert_det_bits(f"{layout}: dz", got[1], dets[1])
                assert not bool(ws.any()), (layout, "the workspace is not zero")
            if want is None:
                want = got
            for what, g, w_ in zip(names, got, want, strict=True):
                assert_same_bits(f"{layout}: {what}", g, w_)
        assert [n for n, _ in launched] == [name] * calls, (layout, launched)
        for _, strides in launched:
            assert strides["heads"] == 1, (layout, strides)
            heads = [v for k, v in strides.items() if k.endswith("_head")]
            assert heads and all(v == D for v in heads), (layout, strides)
            assert _packed(strides) == packed, (layout, strides)
    for what, got, det, ref in zip(names, want, dets, refs, strict=True):
        assert_error_within_det(what, got, det, ref)
    if role != "fwd":
        mag = suite.magnitudes(dy0, x0, z0, EPS)
        suite.assert_dw_within(
            want[2], dets[2], mag, what="dw vs the deterministic package"
        )


# Identical rows whose weight-gradient terms ``dy * silu(z) * x * rstd`` are
# below FP32's smallest subnormal, 2^-149 (``x`` 1, ``z`` 2^-133, the smallest
# BF16 subnormal, so ``silu(z)`` is 2^-134 exactly; ``dy`` 2^-16 .. 2^-14 by
# column): each term rounds to 0 or 2^-149 in FP32, the same in every row, so
# the error grows with the rows.  Row counts of the short (a BF16 weight
# gradient), the static and the claimed backward.
SUBNORMAL_SHAPES = [
    pytest.param(16, 1, id="16x1"),
    pytest.param(4096, 1, id="4096x1"),
    pytest.param(8192, 32, id="8192x32"),
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
def test_subnormal_weight_gradient_within_the_floored_budget(
    T, H, dw_dtype, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weight gradients in the subnormal range, where the documented budget
    ``|dw64| * 2^-7 + sum|terms| * 2^-20`` (for FP32 normal gates, terms and
    sums) does not hold for either package: the backward launches
    ``module_for_rows``' module, every output repeats bit for bit, ``dx`` and
    ``dz`` are the deterministic package's bits and a BF16 ``dw`` the
    deterministic ``dw``'s bits, ``dx``, ``dz`` and ``dw`` lie within the error
    budget of the deterministic package's, and every ``dw`` lies within that
    budget widened by the
    documented underflow allowance, ``(1 + |x * rstd|) * 2^-150`` per row
    plus half the dtype's smallest subnormal, with the workspace zero."""
    x, z, w, dy = _underflowing_rows(T, H)
    rows = T * H
    arch = loader.device_arch(torch.cuda.current_device())
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    name = loader.module_for_rows(arch, role, rows, _sm_count())
    dets = det_backward(
        dy, x, z, w, EPS, grad_weight=torch.empty(D, dtype=dw_dtype, device=x.device)
    )
    _, dx64, dz64, _ = suite.fp64_reference(dy, x, z, w, EPS)
    xd, zd = x.double(), z.double()
    n = xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + EPS)
    terms = (dy.double() * zd * torch.sigmoid(zd) * n).reshape(rows, D)
    dw64 = terms.sum(0)
    smallest = 2.0**-133 if dw_dtype == torch.bfloat16 else 2.0**-149
    allowance = (1 + n.abs().max().item()) * rows * 2.0**-150 + smallest / 2
    budget = dw64.abs() * suite.BF16_RTOL + terms.abs().sum(0) * suite.DW_MAG_TOL
    assert bool(((dets[2].double() - dw64).abs() > budget).any()), (
        "the inputs stay within the normal-range budget"
    )
    ws = nd_workspace(rows, x.device)
    launched = _spy_launches(monkeypatch)
    first = None
    for _ in range(suite.REPEATS):
        dw = torch.empty(D, dtype=dw_dtype, device=x.device)
        dx, dz, _ = nd_backward(dy, x, z, w, EPS, grad_weight=dw, workspace=ws)
        if first is None:
            first = (dx, dz, dw)
        for what, g, f in zip(("dx", "dz", "dw"), (dx, dz, dw), first, strict=True):
            assert_same_bits(what, g, f)
        assert_det_bits("dx", dx, dets[0])
        assert_det_bits("dz", dz, dets[1])
        if dw_dtype == torch.bfloat16:
            assert_same_bits("dw vs the deterministic package", dw, dets[2])
        excess = (dw.double() - dw64).abs() - budget - allowance
        assert not bool((excess > 0).any()), (
            f"dw beyond the floored budget by {excess.max().item():.3e}"
        )
        assert not bool(ws.any()), "the workspace is not zero"
    for what, got, det, ref in zip(
        ("dx", "dz", "dw"), first, dets, (dx64, dz64, dw64), strict=True
    ):
        assert_error_within_det(what, got, det, ref)
    assert launched == [name] * suite.REPEATS, launched
