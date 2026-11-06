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

Tests of ``flashinfer.experimental.cake.cake_mhc_train``, the experimental module
of the Cake four-stream mHC training kernels' non-deterministic package (SM100 /
SM103 / SM107; FlashInfer issue #8068).

Every entry point but ``cake_mhc_train_reset_workspace`` and ``route_key`` takes
a keyword-only ``deterministic: bool = True`` and runs the non-deterministic
package ``flashinfer.experimental.cake.cake_mhc_train_backend`` only for an
explicit ``deterministic=False``: any other value, the default included, raises
``ValueError`` naming the deterministic package's entry point
(``flashinfer.cake_mhc_det_train.cake_mhc_det_train_*``), and nothing falls back
to that package.  With PyTorch's deterministic algorithms on, the backward
raises PyTorch's alert and the other entry points run unchanged.  This module
checks:

* the surface without a device: the exports (no head, no deterministic names),
  the signatures (the deterministic package's plus the keyword-only
  ``deterministic=True``), no import of this module by ``flashinfer``'s top
  level, no import of the deterministic package by this module and none of the
  former fallback's state, the ``ValueError`` of the default and of every value
  but ``False`` before anything else in every torch mode, each entry point's
  forwarding to the backend's (replaced by a recorder) with every argument
  unchanged, and the ``TypeError`` of ``cake_mhc_train_finish`` for any other
  handle (naming the deterministic finish);
* on a device, at every hidden size the device's architecture routes (4096,
  5120 and 7168 on SM100 / SM103, 4096 on SM107): ``deterministic=False``, the
  deterministic package's launches refused, gives the backend's workspace sizes
  and results (the forward entry points and the backward's ``dX`` / ``dscale`` /
  ``dbase`` bitwise the deterministic package's, ``dfn`` within FP32 rounding);
  under ``torch.use_deterministic_algorithms(True)``
  ``is_cake_mhc_train_supported``, the workspace helpers, the forward entry
  points and ``backward_prepare`` run as with the mode off, while
  ``backward_finish`` and the prepare node's backward raise PyTorch's alert
  before any launch (``warn_only``: one alert, then the backend's backward); a
  deterministic backward workspace serves the backward; a deterministic prepare
  handle is refused by ``cake_mhc_train_finish``; a hidden size runs exactly
  where the deterministic package runs it;
* training with ``deterministic=False`` composed with the model's head of the
  deterministic package (``flashinfer.cake_mhc_det_train.cake_mhc_det_train_head``,
  the stream mean of the last layer): the gradients reach every leaf through
  the head and equal the deterministic training and head's.

The numerical contract of the non-deterministic backward is
``test_cake_mhc_train.py``'s.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import re
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer.cake_mhc_det_train_backend as det_backend
import flashinfer.experimental.cake.cake_mhc_train as facade
import flashinfer.experimental.cake.cake_mhc_train_backend as backend
from flashinfer import cake_mhc_det_train as det_facade
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.cake_mhc_det_train_backend import cake_jit as det_loader
from flashinfer.experimental.cake.cake_mhc_train_backend import cake_jit as loader

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
# The hidden sizes of both packages; a device case of a size runs where the device's architecture routes it.
HIDDEN_SIZES = (4096, 5120, 7168)
REPO = Path(__file__).resolve().parents[2]
SEED = 2032
BF16 = torch.bfloat16
# The ticket region every workspace starts with (zero again after every completed launch).
TICKET_BYTES = 4 * loader.TICKET_WORDS
_OUTPUTS = ("dX", "dfn", "dscale", "dbase")
_EXACT = ("dX", "dscale", "dbase")
_LEAVES = ("X", "fn", "scale", "base")
# The ``dfn`` of the non-deterministic backward against the deterministic one: per-chunk partials, the same or another
# partition's, summed in another order (the per-element bound is test_cake_mhc_train.py's).
DFN_REL_L2_MAX = 1e-6
_ALERT = "does not have a deterministic implementation"
# facade entry point -> (the deterministic package's entry point its ValueError names, positional stand-ins: the mode
# check comes first, so no stand-in is read before it)
_ENTRIES = {
    "is_cake_mhc_train_supported": ("is_cake_mhc_det_train_supported", ("device", 5120)),
    "cake_mhc_train_forward_workspace_bytes": ("cake_mhc_det_train_forward_workspace_bytes", (8, "device")),
    "cake_mhc_train_backward_workspace_bytes": ("cake_mhc_det_train_backward_workspace_bytes", (8, "device")),
    "cake_mhc_train_forward_workspace": ("cake_mhc_det_train_forward_workspace", (8, "device")),
    "cake_mhc_train_backward_workspace": ("cake_mhc_det_train_backward_workspace", (8, "device")),
    "cake_mhc_train_forward_prepare": ("cake_mhc_det_train_forward_prepare", ("X", "fn", "scale", "base")),
    "cake_mhc_train_forward_stats": ("cake_mhc_det_train_forward_stats", ("X", "fn", "scale", "base")),
    "cake_mhc_train_forward_finish": ("cake_mhc_det_train_forward_finish", ("X", "B", "stats")),
    "cake_mhc_train_backward_prepare": ("cake_mhc_det_train_backward_prepare", ("dY", "B", "X", "stats")),
    "cake_mhc_train_backward_finish": (
        "cake_mhc_det_train_backward_finish",
        ("X", "dY", "dU", "B", "fn", "scale", "base", "stats", "grad_stats"),
    ),
    "cake_mhc_train_prepare": ("cake_mhc_det_train_prepare", ("X", "fn", "scale", "base")),
    "cake_mhc_train_finish": ("cake_mhc_det_train_finish", ("X", "B", "handle")),
}  # fmt: skip
# The names the facade re-exports from the backend unchanged.
_REEXPORTS = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeMHCTrainFinishFunction",
    "CakeMHCTrainPrepareFunction",
    "cake_mhc_train_reset_workspace",
    "route_key",
)
_WORKSPACE_HELPERS = (
    "cake_mhc_train_forward_workspace_bytes",
    "cake_mhc_train_backward_workspace_bytes",
    "cake_mhc_train_forward_workspace",
    "cake_mhc_train_backward_workspace",
)
# The ``deterministic`` keywords every entry point refuses: the default, ``True`` and values that are not bools.
_REFUSED_MODES = (
    {},
    {"deterministic": True},
    {"deterministic": None},
    {"deterministic": 0},
    {"deterministic": 1},
    {"deterministic": "False"},
)
# PyTorch's deterministic algorithms as ``(mode, warn_only)``: off, strict, warn_only.
_TORCH_MODES = ((False, False), (True, False), (True, True))
# The state of the former fallback to the deterministic package: the once-per-process warning, the mode switch and
# the companion backward workspaces.
_REMOVED = (
    "_DET_FALLBACK_WARNED",
    "_deterministic",
    "_COMPANIONS",
    "_forget_companions",
    "_fallback_backward_workspace",
)
_HANDLE_ERROR = (
    "handle must be the handle returned by cake_mhc_train_prepare (a cake_mhc_det_train_prepare handle goes to "
    "flashinfer.cake_mhc_det_train.cake_mhc_det_train_finish)"
)


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(_device()))


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and det_facade.is_cake_mhc_det_train_supported(_device())
        and facade.is_cake_mhc_train_supported(_device(), deterministic=False)
    )


requires_device = pytest.mark.skipif(
    not _supported(),
    reason="cake_mhc_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries of both packages that have no prebuilt file through the spec objects
    # the loaders build on first use; a prebuilt or cached library loads without compiling.
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in [*det_loader.specs_for_arch(arch), *loader.specs_for_arch(arch)]:
            spec.build_and_load()
    yield


@contextlib.contextmanager
def _torch_deterministic(mode: bool, *, warn_only: bool = False):
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


@pytest.fixture
def deterministic_algorithms():
    """``torch.use_deterministic_algorithms``; the process-wide flags are restored however the test ends."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield torch.use_deterministic_algorithms
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _shape(parameters) -> list:
    """Name, kind and default of each parameter (the annotations are strings under postponed evaluation)."""
    return [(p.name, p.kind, p.default) for p in parameters]


def _exactly(message: str) -> str:
    """A ``pytest.raises`` pattern that matches ``message`` and nothing else."""
    return f"^{re.escape(message)}$"


def _mode_error(name: str, value) -> str:
    """The ``ValueError`` message of the entry point ``name`` for ``deterministic=value`` (any value but ``False``)."""
    return (
        f"{name} computes non-deterministically (got deterministic={value!r}): pass deterministic=False for the "
        f"non-deterministic path, or call flashinfer.cake_mhc_det_train.{_ENTRIES[name][0]} for the deterministic "
        "one (no fallback)"
    )


def _refuse_deterministic_package(m) -> None:
    """Every entry point of the deterministic package (its facade's and its backend's) fails the test when called."""
    for det_name, _stand_ins in _ENTRIES.values():

        def refuse(*_args, _name=det_name, **_kwargs):
            raise AssertionError(f"the deterministic package's {_name} was called")

        for module in (det_facade, det_backend):
            m.setattr(module, det_name, refuse)


# ----------------------------------------------------------------------------------------------------------------
# The surface (no device)


@pytest.mark.cpu_only
def test_exports() -> None:
    """``__all__``: the re-exported backend objects and the facade's own entry points (``def``s of this module, not
    the backend's objects); no public head name and no deterministic name."""
    expected = set(_ENTRIES) | set(_REEXPORTS)
    assert set(facade.__all__) == expected, sorted(set(facade.__all__) ^ expected)
    for name in _REEXPORTS:
        assert getattr(facade, name) is getattr(backend, name), name
    for name in _ENTRIES:
        function = getattr(facade, name)
        assert inspect.unwrap(function).__module__ == facade.__name__, name
        assert function is not getattr(backend, name), name
    for name in (n for n in dir(facade) if not n.startswith("_")):
        assert "head" not in name.lower(), name
        assert "det_train" not in name and "DetTrain" not in name, name
    assert facade.SUPPORTED_COMPUTE_CAPABILITIES == SUPPORTED_CAPABILITIES


@pytest.mark.cpu_only
def test_signatures_mirror_the_deterministic_facade() -> None:
    """Every entry point, ``cake_mhc_train_finish`` included, has the parameters of the deterministic entry point its
    ``ValueError`` names (names, kinds and defaults, in order; an exported name of
    ``flashinfer.cake_mhc_det_train``) plus a keyword-only ``deterministic=True`` it documents.  The backend's entry
    points have the facade's parameters, but for ``is_cake_mhc_train_supported`` (no mode: the facade checks it).
    ``cake_mhc_train_reset_workspace`` and ``route_key`` take no ``deterministic``."""
    for name, (det_name, _stand_ins) in _ENTRIES.items():
        assert det_name in det_facade.__all__, det_name
        function = getattr(facade, name)
        parameters = _shape(inspect.signature(function).parameters.values())
        det_parameters = _shape(
            inspect.signature(getattr(det_facade, det_name)).parameters.values()
        )
        backend_parameters = _shape(
            inspect.signature(getattr(backend, name)).parameters.values()
        )
        assert parameters[:-1] == det_parameters, name
        assert parameters[-1] == (
            "deterministic",
            inspect.Parameter.KEYWORD_ONLY,
            True,
        ), name
        assert "deterministic" in (inspect.getdoc(function) or ""), name
        if name == "is_cake_mhc_train_supported":
            assert backend_parameters == parameters[:-1], name
        else:
            assert backend_parameters == parameters, name
    for name in ("cake_mhc_train_reset_workspace", "route_key"):
        assert (
            "deterministic" not in inspect.signature(getattr(facade, name)).parameters
        ), name


def _imports(path: Path) -> list[str]:
    """The modules the imports of ``path`` name, and each imported name qualified by its module (a relative module
    without its leading dots)."""
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.append(module)
            names.extend(
                f"{module}.{alias.name}" if module else alias.name
                for alias in node.names
            )
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return names


@pytest.mark.cpu_only
def test_top_level_does_not_import_the_facade() -> None:
    """``flashinfer/__init__.py`` does not import the experimental facade."""
    imported = _imports(REPO / "flashinfer" / "__init__.py")
    leaks = [
        m for m in imported if m.startswith("experimental") or "experimental.cake" in m
    ]
    assert not leaks, leaks


@pytest.mark.cpu_only
def test_facade_does_not_import_the_deterministic_package() -> None:
    """The facade's source imports nothing of the deterministic package (no imported module or name contains
    ``cake_mhc_det_train``) and defines and reads none of the former fallback's names (the once-per-process
    warning flag, the mode switch, the companion workspaces)."""
    path = Path(facade.__file__)
    det_imports = [m for m in _imports(path) if "cake_mhc_det_train" in m]
    assert not det_imports, det_imports
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Global):
            names.update(node.names)
    left = sorted(names & set(_REMOVED))
    assert not left, left


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "mode",
    _REFUSED_MODES,
    ids=lambda mode: repr(mode["deterministic"]) if mode else "default",
)
def test_every_mode_but_false_is_a_value_error(mode: dict, monkeypatch) -> None:
    """The default and every ``deterministic`` but ``False`` (``True``, ``None``, ``0``, ``1``, ``"False"``) make every
    entry point raise one ``ValueError`` that names the deterministic package's entry point, before anything else (no
    stand-in argument is read) and without calling the deterministic package, with PyTorch's deterministic algorithms
    off, strict and ``warn_only`` alike (no warning, no alert)."""
    value = mode.get("deterministic", True)
    _refuse_deterministic_package(monkeypatch)
    for torch_mode, warn_only in _TORCH_MODES:
        with (
            _torch_deterministic(torch_mode, warn_only=warn_only),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error")
            warnings.simplefilter("ignore", ExperimentalWarning)
            for name, (_det_name, args) in _ENTRIES.items():
                with pytest.raises(
                    ValueError, match=_exactly(_mode_error(name, value))
                ):
                    getattr(facade, name)(*args, **mode)


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", sorted(_ENTRIES))
def test_forwards_to_the_backend(name: str, monkeypatch) -> None:
    """With the backend's entry point replaced by a recorder and the deterministic package's refused, a
    ``deterministic=False`` call (stand-ins for every positional and keyword-only argument) calls the backend's entry
    point of the same name once with every argument unchanged (``deterministic=False`` included, but for
    ``is_cake_mhc_train_supported``: the backend's has no mode), with PyTorch's deterministic algorithms off, strict
    and ``warn_only`` alike (no warning, no alert)."""
    function = getattr(facade, name)
    target = inspect.signature(getattr(backend, name))
    _det_name, args = _ENTRIES[name]
    keywords = {
        p.name: f"<{p.name}>"
        for p in inspect.signature(function).parameters.values()
        if p.kind is inspect.Parameter.KEYWORD_ONLY and p.name != "deterministic"
    }
    mode = {} if name == "is_cake_mhc_train_supported" else {"deterministic": False}
    want = target.bind(*args, **keywords, **mode)
    want.apply_defaults()
    calls: list = []

    def record(*a, **k):
        calls.append(target.bind(*a, **k))
        return "nd"

    monkeypatch.setattr(backend, name, record)
    _refuse_deterministic_package(monkeypatch)
    for torch_mode, warn_only in _TORCH_MODES:
        calls.clear()
        with (
            _torch_deterministic(torch_mode, warn_only=warn_only),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error")
            warnings.simplefilter("ignore", ExperimentalWarning)
            assert function(*args, **keywords, deterministic=False) == "nd"
        assert len(calls) == 1, (torch_mode, warn_only)
        calls[0].apply_defaults()
        assert calls[0].arguments == want.arguments, (torch_mode, warn_only)


@pytest.mark.cpu_only
def test_finish_refuses_other_handles() -> None:
    """``cake_mhc_train_finish(..., deterministic=False)`` refuses every handle but its prepare's -- the deterministic
    package's handle included -- with one ``TypeError`` naming the deterministic finish; the mode is checked first
    (the default is the mode's ``ValueError`` whatever the handle)."""
    det_handle = det_backend._MHCTrainHandle(None, ())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        for handle in (det_handle, object(), None, (None, ())):
            with pytest.raises(TypeError, match=_exactly(_HANDLE_ERROR)):
                facade.cake_mhc_train_finish("X", "B", handle, deterministic=False)
            with pytest.raises(
                ValueError,
                match=_exactly(_mode_error("cake_mhc_train_finish", True)),
            ):
                facade.cake_mhc_train_finish("X", "B", handle)


# ----------------------------------------------------------------------------------------------------------------
# On a device


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _sm_count() -> int:
    return loader.sm_count(_device().index)


def _hidden_sizes() -> tuple[int, ...]:
    """The hidden sizes the device's architecture routes every kind of (the deterministic package's sizes there)."""
    arch = _arch()
    return tuple(loader.supported_hidden_sizes(arch)) if arch else ()


@pytest.fixture(params=HIDDEN_SIZES, ids=lambda h: f"H{h}")
def hidden(request) -> int:
    """The hidden size of a device case (skipped where the architecture has no route of it)."""
    if request.param not in _hidden_sizes():
        pytest.skip(
            f"hidden {request.param} has no exported route on {_arch()} "
            f"(exported hidden sizes: {_hidden_sizes()})"
        )
    return request.param


def _inputs(T: int, seed: int = SEED, hidden: int = 4096) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(
            *shape, generator=generator, device=_device(), dtype=torch.float32
        ).to(dtype)

    i = {
        "X": randn(T, 4, hidden, dtype=BF16),
        "fn": (randn(24, 4 * hidden) * 1e-2).contiguous(),
        "scale": 1.0 + 0.1 * randn(3),
        "base": 0.1 * randn(24),
        "B": randn(T, hidden, dtype=BF16),
        "dY": randn(T, 4, hidden, dtype=BF16),
        "dU": randn(T, hidden, dtype=BF16),
    }
    _U, i["stats"] = det_facade.cake_mhc_det_train_forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"]
    )
    _dB, i["gstats"] = det_facade.cake_mhc_det_train_backward_prepare(
        i["dY"], i["B"], i["X"], i["stats"]
    )
    return i


_ARG_KEYS = ("X", "dY", "dU", "B", "fn", "scale", "base", "stats", "gstats")


def _args(i: dict) -> tuple:
    return tuple(i[k] for k in _ARG_KEYS)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm())


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and torch.equal(
        a.detach().contiguous().reshape(-1).view(torch.uint8),
        b.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def _assert_backward(got, want, what: str) -> None:
    """``dX`` / ``dscale`` / ``dbase`` of a non-deterministic backward bitwise ``want``'s, ``dfn`` finite and within
    ``DFN_REL_L2_MAX`` of it."""
    for k, g, w in zip(_OUTPUTS, got, want, strict=True):
        if k in _EXACT:
            assert _same_bits(g, w), f"{k} {what}: bits differ"
    assert bool(torch.isfinite(got[1]).all()), f"dfn {what}"
    assert _rel_l2(got[1], want[1]) <= DFN_REL_L2_MAX, f"dfn {what}"


def _assert_grads(got: list, want: list, what: str) -> None:
    """The ``X`` / ``scale`` / ``base`` gradients of a non-deterministic backward bitwise ``want``'s, the ``fn``
    gradient finite and within ``DFN_REL_L2_MAX`` of it."""
    for name, g, w in zip(_LEAVES, got, want, strict=True):
        if name == "fn":
            assert bool(torch.isfinite(g).all()), f"grad fn {what}"
            assert _rel_l2(g, w) <= DFN_REL_L2_MAX, f"grad fn {what}"
        else:
            assert _same_bits(g, w), f"grad {name} {what}"


def _assert_ticket_zero(ws: torch.Tensor, what: str) -> None:
    torch.cuda.synchronize()
    nonzero = int(torch.count_nonzero(ws[:TICKET_BYTES]))
    assert nonzero == 0, f"{what}: {nonzero} non-zero bytes in the ticket region"


_FORWARD_ENTRIES = (
    "forward_prepare",
    "forward_stats",
    "forward_finish",
    "backward_prepare",
)


def _entries(module, prefix: str = "cake_mhc_train") -> dict:
    """The forward entry points and ``backward_prepare`` of ``module`` by short name."""
    return {k: getattr(module, f"{prefix}_{k}") for k in _FORWARD_ENTRIES}


def _forward_outputs(entries: dict, i: dict, **mode) -> dict:
    X, fn, scale, base = (i[k] for k in _LEAVES)
    U, stats = entries["forward_prepare"](X, fn, scale, base, **mode)
    dB, gstats = entries["backward_prepare"](i["dY"], i["B"], X, stats, **mode)
    return {
        "U": U.clone(),
        "stats": stats.clone(),
        "replay": entries["forward_stats"](X, fn, scale, base, **mode).clone(),
        "Y": entries["forward_finish"](X, i["B"], stats, **mode).clone(),
        "dB": dB.clone(),
        "gstats": gstats.clone(),
    }


def _nd_prepare(*args, **kwargs):
    return facade.cake_mhc_train_prepare(*args, deterministic=False, **kwargs)


def _nd_finish(*args, **kwargs):
    return facade.cake_mhc_train_finish(*args, deterministic=False, **kwargs)


def _graph(prepare, finish, i: dict, **kwargs) -> tuple:
    """``(leaves, handle, Y, loss)`` of ``Y = finish(X, U + B, prepare(X, fn, scale, base))`` and the loss
    ``sum(Y * dY)``, not yet differentiated; ``kwargs`` go to ``prepare``."""
    leaves = [i[k].clone().requires_grad_(True) for k in _LEAVES]
    U, handle = prepare(*leaves, **kwargs)
    Y = finish(leaves[0], (U.float() + i["B"].float()).to(BF16), handle)
    return leaves, handle, Y, (Y.float() * i["dY"].float()).sum()


def _autograd(prepare, finish, i: dict, **kwargs) -> tuple:
    """``(handle, Y, [X.grad, fn.grad, scale.grad, base.grad])`` of :func:`_graph` after its loss's backward."""
    leaves, handle, Y, loss = _graph(prepare, finish, i, **kwargs)
    loss.backward()
    return handle, Y.detach(), [t.grad for t in leaves]


def _refuse_launches(m, module) -> None:
    """Every launch of ``module`` (all go through its ``_launch_route`` / ``_run_route``) fails the test."""

    def refuse(*_args, **_kwargs):
        raise AssertionError(f"{module.__name__} launched a kernel")

    m.setattr(module, "_launch_route", refuse)
    m.setattr(module, "_run_route", refuse)


def _det_backward(i: dict) -> list[torch.Tensor]:
    return [t.clone() for t in det_facade.cake_mhc_det_train_backward_finish(*_args(i))]


def _references(i: dict) -> dict:
    """The deterministic package's outputs of ``i``: the forward entry points, the backward, the autograd ``Y`` and
    gradients."""
    _h, Y, grads = _autograd(
        det_facade.cake_mhc_det_train_prepare, det_facade.cake_mhc_det_train_finish, i
    )
    return {
        "forward": _forward_outputs(_entries(det_facade, "cake_mhc_det_train"), i),
        "backward": _det_backward(i),
        "Y": Y,
        "grads": grads,
    }


def _workspace_sizes(T: int, hidden: int) -> dict[str, int]:
    """The facade's forward and backward workspace bytes for ``T`` tokens at ``hidden``."""
    return {
        kind: getattr(facade, f"cake_mhc_train_{kind}_workspace_bytes")(
            T, _device(), hidden=hidden, deterministic=False
        )
        for kind in ("forward", "backward")
    }


@requires_device
@pytest.mark.parametrize("T", (7, 2049))
def test_false_is_the_nondeterministic_backend(
    T: int, hidden: int, monkeypatch
) -> None:
    """``deterministic=False``, with the deterministic package's launches refused: the backend's workspace sizes (the
    forward one the deterministic package's, the backward one smaller: the deterministic layout adds the per-chunk
    ``dfn`` partials) and zeroed buffers; the forward entry points bitwise the backend's and the deterministic
    package's; ``backward_finish`` (default and caller workspaces) with the backend's and the deterministic ``dX`` /
    ``dscale`` / ``dbase`` bits and a ``dfn`` within FP32 rounding of both; the autograd prepare makes the backend's
    handle (``Y`` and the ``X`` / ``scale`` / ``base`` gradients the deterministic bits, the ``fn`` gradient within
    FP32 rounding)."""
    i = _inputs(T, hidden=hidden)
    ref = _references(i)
    direct_forward = _forward_outputs(_entries(backend), i, deterministic=False)
    direct = [
        t.clone()
        for t in backend.cake_mhc_train_backward_finish(*_args(i), deterministic=False)
    ]
    dev = _device()
    nd_bytes = loader.workspace_bytes_upto("bwd", T, _sm_count(), _arch(), hidden)
    what = f"T={T} H={hidden}"
    with monkeypatch.context() as m:
        _refuse_launches(m, det_backend)
        sizes = _workspace_sizes(T, hidden)
        assert (
            sizes["backward"]
            == nd_bytes
            == backend.cake_mhc_train_backward_workspace_bytes(
                T, dev, hidden=hidden, deterministic=False
            )
        ), what
        assert nd_bytes < det_facade.cake_mhc_det_train_backward_workspace_bytes(
            T, dev, hidden=hidden
        ), what
        assert (
            sizes["forward"]
            == loader.workspace_bytes_upto("fwd", T, _sm_count(), _arch(), hidden)
            == det_facade.cake_mhc_det_train_forward_workspace_bytes(
                T, dev, hidden=hidden
            )
        ), what
        ws = facade.cake_mhc_train_backward_workspace(
            T, dev, hidden=hidden, deterministic=False
        )
        assert ws.dtype == torch.uint8 and ws.numel() == nd_bytes, what
        assert int(torch.count_nonzero(ws)) == 0, what
        got = _forward_outputs(_entries(facade), i, deterministic=False)
        for k, v in ref["forward"].items():
            assert _same_bits(got[k], v), f"{k} {what} vs deterministic"
            assert _same_bits(got[k], direct_forward[k]), f"{k} {what} vs the backend"
        for workspace in (None, ws):
            got = facade.cake_mhc_train_backward_finish(
                *_args(i), workspace=workspace, deterministic=False
            )
            label = f"{what} workspace={workspace is not None}"
            _assert_backward(got, ref["backward"], f"{label} vs deterministic")
            _assert_backward(got, direct, f"{label} vs the backend")
        handle, Y, grads = _autograd(_nd_prepare, _nd_finish, i)
    assert isinstance(handle, backend._MHCTrainHandle), what
    assert _same_bits(Y, ref["Y"]), f"Y {what}"
    _assert_grads(grads, ref["grads"], what)


@requires_device
def test_strict_mode_alerts_the_backward_only(
    hidden: int, deterministic_algorithms, monkeypatch
) -> None:
    """Under ``torch.use_deterministic_algorithms(True)``, with the deterministic package's launches refused (nothing
    falls back to it): ``is_cake_mhc_train_supported``, the workspace helpers, the forward entry points,
    ``backward_prepare`` and the prepare / finish forwards run without an alert or a warning (warnings are errors)
    and give what they give with the mode off; ``backward_finish`` raises PyTorch's alert (``RuntimeError``) before
    any launch (the backend's launches refused for that call), the caller's outputs and workspace keeping their
    bytes, and so does the backward of a prepare node, recorded with the mode on or off."""
    T = 129
    dev = _device()
    i = _inputs(T, hidden=hidden)
    want_supported = facade.is_cake_mhc_train_supported(
        dev, hidden, deterministic=False
    )
    want_sizes = _workspace_sizes(T, hidden)
    want = _forward_outputs(_entries(facade), i, deterministic=False)
    _leaves_off, _handle_off, want_Y, recorded_off = _graph(_nd_prepare, _nd_finish, i)
    ws = facade.cake_mhc_train_backward_workspace(
        T, dev, hidden=hidden, deterministic=False
    )
    ws[TICKET_BYTES:].fill_(0xAB)
    before = ws.clone()
    outputs = {
        "dX": torch.full((T, 4, hidden), 3.0, dtype=BF16, device=dev),
        "dfn": torch.full((24, 4 * hidden), 3.0, device=dev),
        "dscale": torch.full((3,), 3.0, device=dev),
        "dbase": torch.full((24,), 3.0, device=dev),
    }
    with monkeypatch.context() as m:
        _refuse_launches(m, det_backend)
        deterministic_algorithms(True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.simplefilter("ignore", ExperimentalWarning)
            supported = facade.is_cake_mhc_train_supported(
                dev, hidden, deterministic=False
            )
            sizes = _workspace_sizes(T, hidden)
            buffers = {
                kind: getattr(facade, f"cake_mhc_train_{kind}_workspace")(
                    T, dev, hidden=hidden, deterministic=False
                )
                for kind in ("forward", "backward")
            }
            got = _forward_outputs(_entries(facade), i, deterministic=False)
            _leaves_on, _handle_on, Y, recorded_on = _graph(_nd_prepare, _nd_finish, i)
        with (
            monkeypatch.context() as n,
            pytest.raises(
                RuntimeError,
                match="^" + re.escape(f"cake_mhc_train_backward_finish {_ALERT}"),
            ),
        ):
            _refuse_launches(n, backend)
            facade.cake_mhc_train_backward_finish(
                *_args(i), workspace=ws, deterministic=False, **outputs
            )
        for loss in (recorded_on, recorded_off):
            with pytest.raises(
                RuntimeError,
                match=re.escape(f"CakeMHCTrainPrepareFunction.backward {_ALERT}"),
            ):
                loss.backward()
        deterministic_algorithms(False)
    torch.cuda.synchronize()
    assert supported is True and want_supported is True
    assert sizes == want_sizes, (sizes, want_sizes)
    for kind, buffer in buffers.items():
        assert buffer.dtype == torch.uint8 and buffer.numel() == sizes[kind], kind
        assert int(torch.count_nonzero(buffer)) == 0, kind
    for k, v in want.items():
        assert _same_bits(got[k], v), f"{k} under the strict mode"
    assert _same_bits(Y, want_Y), "Y under the strict mode"
    for k, v in outputs.items():
        assert bool((v == 3.0).all()), f"{k}: a refused call wrote its output"
    assert torch.equal(ws, before), "a refused call wrote its workspace"


@requires_device
def test_warn_only_alerts_once_and_runs_the_backend(
    hidden: int, deterministic_algorithms, monkeypatch
) -> None:
    """Under ``torch.use_deterministic_algorithms(True, warn_only=True)``, with the deterministic package's launches
    refused, ``backward_finish`` gives exactly one alert (a ``UserWarning`` naming it) and the backend's backward:
    ``dX`` / ``dscale`` / ``dbase`` bitwise the deterministic package's, ``dfn`` within FP32 rounding of it."""
    T = 129
    i = _inputs(T, hidden=hidden)
    want = _det_backward(i)
    with monkeypatch.context() as m:
        _refuse_launches(m, det_backend)
        deterministic_algorithms(True, warn_only=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.simplefilter("ignore", ExperimentalWarning)
            got = [
                t.clone()
                for t in facade.cake_mhc_train_backward_finish(
                    *_args(i), deterministic=False
                )
            ]
        deterministic_algorithms(False)
    alerts = [w for w in caught if _ALERT in str(w.message)]
    assert len(alerts) == 1, [str(w.message) for w in caught]
    assert alerts[0].category is UserWarning
    assert str(alerts[0].message).startswith(
        f"cake_mhc_train_backward_finish {_ALERT}"
    ), str(alerts[0].message)
    _assert_backward(got, want, f"warn_only H={hidden}")


@requires_device
@pytest.mark.parametrize("T", (7, 2049))
def test_deterministic_backward_workspace_serves_the_backward(
    T: int, hidden: int, monkeypatch
) -> None:
    """A backward workspace of the deterministic package
    (``flashinfer.cake_mhc_det_train.cake_mhc_det_train_backward_workspace``, larger than the backend's: the same
    regions at the same offsets plus the per-chunk ``dfn`` partials) serves ``backward_finish`` and a prepare node's
    ``backward_workspace``, the deterministic package's launches refused: ``dX`` / ``dscale`` / ``dbase``, ``Y`` and
    the ``X`` / ``scale`` / ``base`` gradients the deterministic bits, ``dfn`` and the ``fn`` gradient within FP32
    rounding, the ticket region zero after every call."""
    i = _inputs(T, hidden=hidden)
    ref = _references(i)
    what = f"T={T} H={hidden} on a deterministic workspace"
    ws = det_facade.cake_mhc_det_train_backward_workspace(T, _device(), hidden=hidden)
    assert ws.numel() > _workspace_sizes(T, hidden)["backward"], what
    with monkeypatch.context() as m:
        _refuse_launches(m, det_backend)
        got = [
            t.clone()
            for t in facade.cake_mhc_train_backward_finish(
                *_args(i), workspace=ws, deterministic=False
            )
        ]
        _assert_ticket_zero(ws, f"backward_finish {what}")
        handle, Y, grads = _autograd(_nd_prepare, _nd_finish, i, backward_workspace=ws)
        _assert_ticket_zero(ws, f"prepare node {what}")
    _assert_backward(got, ref["backward"], what)
    assert isinstance(handle, backend._MHCTrainHandle), what
    assert _same_bits(Y, ref["Y"]), f"Y {what}"
    _assert_grads(grads, ref["grads"], what)


@requires_device
def test_finish_refuses_a_deterministic_handle() -> None:
    """``cake_mhc_train_finish`` given the handle of the deterministic package's prepare raises the mode's
    ``ValueError`` by default and the handle's ``TypeError`` (naming the deterministic finish) with
    ``deterministic=False``; neither call touches the handle, whose deterministic finish then gives the deterministic
    ``Y`` and gradients bit for bit."""
    i = _inputs(7, seed=SEED + 10)

    def finish(X, B, handle):
        with pytest.raises(
            ValueError, match=_exactly(_mode_error("cake_mhc_train_finish", True))
        ):
            facade.cake_mhc_train_finish(X, B, handle)
        with pytest.raises(TypeError, match=_exactly(_HANDLE_ERROR)):
            facade.cake_mhc_train_finish(X, B, handle, deterministic=False)
        return det_facade.cake_mhc_det_train_finish(X, B, handle)

    handle, Y, grads = _autograd(det_facade.cake_mhc_det_train_prepare, finish, i)
    _h, want_Y, want = _autograd(
        det_facade.cake_mhc_det_train_prepare, det_facade.cake_mhc_det_train_finish, i
    )
    assert isinstance(handle, det_backend._MHCTrainHandle)
    assert _same_bits(Y, want_Y)
    for name, g, w in zip(_LEAVES, grads, want, strict=True):
        assert _same_bits(g, w), f"grad {name}"


# ----------------------------------------------------------------------------------------------------------------
# Hidden sizes


@pytest.mark.cpu_only
def test_nondeterministic_workspace_helpers_reject_other_hidden_sizes() -> None:
    """``deterministic=False`` lists the hidden sizes 4096, 5120 and 7168 (``cake_jit.HIDDEN_SIZES``): its workspace
    helpers raise ``ValueError`` for another hidden size or a non-``int`` one before any device query, in the facade
    and in the backend (the sizes of the device's architecture are ``test_hidden_sizes_follow_the_device``'s)."""
    assert tuple(loader.HIDDEN_SIZES) == HIDDEN_SIZES
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        for name in _WORKSPACE_HELPERS:
            for module in (facade, backend):
                for hidden in (6144, 2048, 4097, 0, -4096, True, 4096.0, "4096"):
                    with pytest.raises(ValueError, match="hidden"):
                        getattr(module, name)(
                            1, torch.device("cuda"), hidden=hidden, deterministic=False
                        )


@requires_device
@pytest.mark.parametrize("hidden", HIDDEN_SIZES[1:], ids=lambda h: f"H{h}")
def test_hidden_sizes_follow_the_device(hidden: int) -> None:
    """``deterministic=False`` takes a hidden size exactly where the deterministic package does: on sm_100a / sm_103a
    5120 and 7168 size their workspaces (the forward one the deterministic package's, the backward one smaller than
    its, both other than the H 4096 sizes) and the entry points run on them; on sm_107a (hidden 4096 only) the
    workspace helpers and the entry points refuse them with ``ValueError``."""
    dev = _device()
    exported = det_facade.is_cake_mhc_det_train_supported(dev, hidden)
    assert exported == (hidden in _hidden_sizes()) == (_arch() != "sm_107a")
    assert (
        facade.is_cake_mhc_train_supported(dev, hidden, deterministic=False) == exported
    )
    assert backend.is_cake_mhc_train_supported(dev, hidden) == exported
    i = _inputs(7, hidden=hidden) if exported else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        if not exported:
            for name in _WORKSPACE_HELPERS:
                for module in (facade, backend):
                    with pytest.raises(ValueError):
                        getattr(module, name)(
                            300, dev, hidden=hidden, deterministic=False
                        )
            X = torch.zeros(7, 4, hidden, dtype=BF16, device=dev)
            fn = torch.zeros(24, 4 * hidden, device=dev)
            scale = torch.ones(3, device=dev)
            base = torch.zeros(24, device=dev)
            for module in (facade, backend):
                with pytest.raises(ValueError):
                    module.cake_mhc_train_forward_prepare(
                        X, fn, scale, base, deterministic=False
                    )
            return
        for kind in ("forward", "backward"):
            nbytes = getattr(facade, f"cake_mhc_train_{kind}_workspace_bytes")
            reference_nbytes = getattr(
                det_facade, f"cake_mhc_det_train_{kind}_workspace_bytes"
            )
            got = nbytes(300, dev, hidden=hidden, deterministic=False)
            want = reference_nbytes(300, dev, hidden=hidden)
            assert got != nbytes(300, dev, deterministic=False), kind
            assert got == want if kind == "forward" else got < want, (kind, got, want)
            ws = getattr(facade, f"cake_mhc_train_{kind}_workspace")(
                300, dev, hidden=hidden, deterministic=False
            )
            assert ws.numel() == got and int(torch.count_nonzero(ws)) == 0, kind
        got = facade.cake_mhc_train_backward_finish(*_args(i), deterministic=False)
        _assert_backward(got, _det_backward(i), f"H={hidden}")


# ----------------------------------------------------------------------------------------------------------------
# The model's head


def _head_connections(prepare, finish, i: dict, layers: list, **kwargs) -> tuple:
    """Two connections of ``prepare`` / ``finish`` (``layers``: their ``(fn, scale, base)``; the branch ``B = U +
    B_0``), then the model's head ``cake_mhc_det_train_head`` (the stream mean of the last connection's ``Y``) under
    the loss ``sum(head * dU)``: ``(head output, [X.grad, *(fn, scale, base).grad per connection])``."""
    X = i["X"].clone().requires_grad_(True)
    params = [[p.clone().requires_grad_(True) for p in layer] for layer in layers]
    x = X
    for fn, scale, base in params:
        U, handle = prepare(x, fn, scale, base, **kwargs)
        x = finish(x, (U.float() + i["B"].float()).to(BF16), handle)
    head = det_facade.cake_mhc_det_train_head(x)
    (head.float() * i["dU"].float()).sum().backward()
    return head.detach(), [X.grad, *[p.grad for layer in params for p in layer]]


@requires_device
@pytest.mark.parametrize("T", (7, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_nd_training_composes_with_the_stream_mean_head(
    T: int, save_stats: bool, hidden: int
) -> None:
    """Training with ``deterministic=False`` feeding the deterministic package's model head
    (``flashinfer.cake_mhc_det_train.cake_mhc_det_train_head``, ``mean_i X[:, i, :]`` of the last connection's
    four streams), at every hidden size the device routes: two connections, the head and a loss on its output.  The
    gradients reach every leaf through the head (finite, non-zero), the connections' handles are the backend's, and
    the head output and the ``X`` / ``scale`` / ``base`` gradients are bitwise those of the deterministic training
    and the same head; each ``fn`` gradient is within FP32 rounding of the deterministic one (rel-L2 <=
    ``DFN_REL_L2_MAX``)."""
    if not det_facade.is_cake_mhc_det_train_head_supported(_device(), hidden):
        pytest.skip(f"the head has no program of hidden {hidden} on {_arch()}")
    i = _inputs(T, seed=SEED + 40, hidden=hidden)
    second = _inputs(1, seed=SEED + 41, hidden=hidden)
    layers = [[j[k] for k in ("fn", "scale", "base")] for j in (i, second)]
    handles = []

    def nd_prepare(*args, **kwargs):
        U, handle = _nd_prepare(*args, **kwargs)
        handles.append(handle)
        return U, handle

    head, grads = _head_connections(
        nd_prepare, _nd_finish, i, layers, save_stats=save_stats
    )
    want_head, want = _head_connections(
        det_facade.cake_mhc_det_train_prepare,
        det_facade.cake_mhc_det_train_finish,
        i,
        layers,
        save_stats=save_stats,
    )
    what = f"T={T} H={hidden} save_stats={save_stats}"
    assert len(handles) == 2
    assert all(isinstance(h, backend._MHCTrainHandle) for h in handles), what
    assert head.shape == (T, hidden) and _same_bits(head, want_head), f"head {what}"
    names = ["X", *(f"{k}[{c}]" for c in range(2) for k in ("fn", "scale", "base"))]
    for name, g, w in zip(names, grads, want, strict=True):
        assert g is not None and bool(torch.isfinite(g).all()), f"grad {name} {what}"
        assert bool((g != 0).any()), f"grad {name} {what}: no gradient reached the leaf"
        if name.startswith("fn"):
            assert _rel_l2(g, w) <= DFN_REL_L2_MAX, f"grad {name} {what}"
        else:
            assert _same_bits(g, w), f"grad {name} {what}"
