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

Tests of the facade ``flashinfer.experimental.cake.cake_mhc_train`` of the Cake
four-stream mHC training kernels with a selectable numerical mode (SM100 /
SM103 / SM107; FlashInfer issue #8068).

Each entry point dispatches on ``deterministic``: ``True`` (the default) runs
the deterministic package ``flashinfer.cake_mhc_det_train_backend`` (bitwise
the ``cake_mhc_det_train_*`` names), ``False`` the non-deterministic backend
``flashinfer.experimental.cake.cake_mhc_train_backend``; with PyTorch's
deterministic algorithms on, ``False`` runs the deterministic package and warns
once per process, on a private companion backward workspace when the caller's
is too small for the deterministic layout.  ``cake_mhc_train_finish`` follows
its handle.  This module checks:

* the surface without a device: the exports (no head, no deterministic names),
  the signatures (the deterministic facade's plus a keyword-only
  ``deterministic=True``), the legacy ``flashinfer.cake_mhc_train`` names and
  the top-level names left deterministic, a non-bool mode refused with a
  ``TypeError`` before anything else, and each entry point's dispatch (the two
  packages' entry points replaced by recorders) in both torch modes, with the
  one-time warning;
* on a device: ``True`` is bitwise the deterministic package and launches
  nothing of the backend; ``False`` launches nothing of the deterministic
  package and gives the backend's results; under PyTorch's deterministic
  algorithms (strict and ``warn_only``) ``False`` gives the deterministic bits
  without an alert, the backend launching nothing;
* the companion workspace of that fallback: a caller backward workspace of the
  non-deterministic sizing is left untouched and the call runs on a private
  companion of the deterministic size of the token-count bucket (never the
  deterministic package's shared default buffer), reused by every later call it
  serves and released with the caller's workspace; a workspace that serves the
  deterministic layout, and any invalid one, passes through (the deterministic
  checks refuse the invalid ones); inside a CUDA graph capture a missing
  companion is a ``ValueError`` and, after one eager call, the captured
  fallback replays with the deterministic bits.

The numerical contract of the non-deterministic backward is
``test_cake_mhc_train.py``'s.
"""

from __future__ import annotations

import ast
import contextlib
import gc
import inspect
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer
import flashinfer.cake_mhc_det_train_backend as det_backend
import flashinfer.cake_mhc_train as legacy
import flashinfer.experimental.cake.cake_mhc_train as facade
import flashinfer.experimental.cake.cake_mhc_train_backend as backend
from flashinfer import cake_mhc_det_train as det_facade
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.cake_mhc_det_train_backend import cake_jit as det_loader
from flashinfer.experimental.cake.cake_mhc_train_backend import cake_jit as loader

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
REPO = Path(__file__).resolve().parents[2]
SEED = 2032
BF16 = torch.bfloat16
_OUTPUTS = ("dX", "dfn", "dscale", "dbase")
_EXACT = ("dX", "dscale", "dbase")
_LEAVES = ("X", "fn", "scale", "base")
# The ``dfn`` of the non-deterministic backward against the deterministic one: the same FP32 partial products summed
# in another order (the per-element bound is test_cake_mhc_train.py's).
DFN_REL_L2_MAX = 1e-6
_FALLBACK = "runs the deterministic package"
_ALERT = "does not have a deterministic implementation"
# facade entry point -> (deterministic package entry point, positional stand-ins)
_DISPATCH = {
    "is_cake_mhc_train_supported": ("is_cake_mhc_det_train_supported", ("device",)),
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


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(_device()))


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and facade.is_cake_mhc_train_supported(_device())
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


def _fallback_warnings(caught) -> list:
    return [w for w in caught if _FALLBACK in str(w.message)]


def _shape(parameters) -> list:
    """Name, kind and default of each parameter (the annotations are strings under postponed evaluation)."""
    return [(p.name, p.kind, p.default) for p in parameters]


# ----------------------------------------------------------------------------------------------------------------
# The surface (no device)


@pytest.mark.cpu_only
def test_exports() -> None:
    """``__all__``: the re-exported backend objects and the facade's own entry points (``def``s of this module, not
    the backend's objects); no public head name and no deterministic name."""
    expected = set(_DISPATCH) | set(_REEXPORTS) | {"cake_mhc_train_finish"}
    assert set(facade.__all__) == expected, sorted(set(facade.__all__) ^ expected)
    for name in _REEXPORTS:
        assert getattr(facade, name) is getattr(backend, name), name
    for name in (*_DISPATCH, "cake_mhc_train_finish"):
        function = getattr(facade, name)
        assert inspect.unwrap(function).__module__ == facade.__name__, name
        assert function is not getattr(backend, name), name
    for name in (n for n in dir(facade) if not n.startswith("_")):
        assert "head" not in name.lower(), name
        assert "det_train" not in name and "DetTrain" not in name, name
    assert facade.SUPPORTED_COMPUTE_CAPABILITIES == SUPPORTED_CAPABILITIES


def _det_name(name: str) -> str:
    return _DISPATCH[name][0] if name in _DISPATCH else "cake_mhc_det_train_finish"


@pytest.mark.cpu_only
def test_signatures_mirror_the_deterministic_facade() -> None:
    """Every mode entry point has the deterministic facade's parameters (names, kinds and defaults, in order) plus a
    keyword-only ``deterministic=True`` it documents; ``cake_mhc_train_finish`` has exactly the deterministic
    finish's parameters (the handle names the package).  The backend's entry points have the facade's parameters,
    but for ``is_cake_mhc_train_supported`` (no mode) and its finish (a ``deterministic`` keyword)."""
    for name in (*_DISPATCH, "cake_mhc_train_finish"):
        function = getattr(facade, name)
        parameters = _shape(inspect.signature(function).parameters.values())
        det_parameters = _shape(
            inspect.signature(getattr(det_facade, _det_name(name))).parameters.values()
        )
        backend_parameters = _shape(
            inspect.signature(getattr(backend, name)).parameters.values()
        )
        if name == "cake_mhc_train_finish":
            assert parameters == det_parameters, name
            assert backend_parameters[:-1] == parameters, name
            assert backend_parameters[-1][0] == "deterministic", name
            continue
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


def _top_level_imports(path: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
            if not node.module:
                names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return names


@pytest.mark.cpu_only
def test_legacy_and_top_level_names_stay_deterministic() -> None:
    """The legacy module ``flashinfer.cake_mhc_train`` and the top-level ``flashinfer.cake_mhc_train_*`` names are the
    deterministic objects (no ``deterministic`` keyword), never this facade's; ``flashinfer/__init__.py`` does not
    import the experimental facade."""
    top_level = set()
    for name in legacy.__all__:
        det_name = name.replace("cake_mhc_train", "cake_mhc_det_train").replace(
            "CakeMHCTrain", "CakeMHCDetTrain"
        )
        assert getattr(legacy, name) is getattr(det_facade, det_name), name
        if not any(part in name for part in ("cake_mhc_train", "CakeMHCTrain")):
            continue
        if hasattr(facade, name):
            assert getattr(legacy, name) is not getattr(facade, name), name
        if hasattr(flashinfer, name):
            assert getattr(flashinfer, name) is getattr(legacy, name), name
            top_level.add(name)
    assert {
        "cake_mhc_train_prepare",
        "cake_mhc_train_finish",
        "cake_mhc_train_backward_finish",
        "CakeMHCTrainPrepareFunction",
    } <= top_level, sorted(top_level)
    for name in _DISPATCH:
        if hasattr(legacy, name):
            parameters = inspect.signature(getattr(legacy, name)).parameters
            assert "deterministic" not in parameters, name
    imported = _top_level_imports(REPO / "flashinfer" / "__init__.py")
    leaks = [
        m for m in imported if m.startswith("experimental") or "experimental.cake" in m
    ]
    assert not leaks, leaks


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", (0, 1, None, "False", 1.0), ids=repr)
def test_non_bool_mode_is_a_type_error(value, monkeypatch) -> None:
    """A ``deterministic`` that is not a bool is a ``TypeError`` naming the entry point before any dispatch, warning
    or device check, also with PyTorch's deterministic algorithms on."""
    monkeypatch.setattr(facade, "_DET_FALLBACK_WARNED", False)
    for torch_mode in (False, True):
        with (
            _torch_deterministic(torch_mode),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            warnings.simplefilter("ignore", ExperimentalWarning)
            for name, (_det_entry, args) in _DISPATCH.items():
                with pytest.raises(
                    TypeError, match=f"{name}: deterministic must be a bool"
                ):
                    getattr(facade, name)(*args, deterministic=value)
        assert not _fallback_warnings(caught)
    assert facade._DET_FALLBACK_WARNED is False


def _recorders(monkeypatch, name: str, det_name: str) -> list:
    """Replace the backend's entry point ``name`` and the deterministic package's ``det_name`` with recorders that
    return ``"nd"`` / ``"det"``; returns the list of ``(tag, args, kwargs)`` calls."""
    calls: list = []

    def recorder(tag: str):
        def record(*args, **kwargs):
            calls.append((tag, args, kwargs))
            return tag

        return record

    monkeypatch.setattr(det_backend, det_name, recorder("det"))
    monkeypatch.setattr(backend, name, recorder("nd"))
    return calls


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", sorted(_DISPATCH))
def test_dispatch(name: str, monkeypatch) -> None:
    """With the two packages' entry points replaced by recorders: the default and ``True`` call the deterministic
    package's entry point without a mode keyword, ``False`` the backend's with ``deterministic=False`` (the same
    arguments otherwise; ``is_cake_mhc_train_supported`` without a mode), without a warning.  With PyTorch's
    deterministic algorithms on (strict and ``warn_only``) ``False`` calls the deterministic package's entry point
    and the process warns once (a ``UserWarning`` naming the entry point), later calls silently, and nothing alerts."""
    det_name, args = _DISPATCH[name]
    calls = _recorders(monkeypatch, name, det_name)
    function = getattr(facade, name)
    monkeypatch.setattr(facade, "_DET_FALLBACK_WARNED", False)
    with _torch_deterministic(False), warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", ExperimentalWarning)
        assert function(*args) == "det"
        assert function(*args, deterministic=True) == "det"
        assert function(*args, deterministic=False) == "nd"
    assert [tag for tag, _a, _k in calls] == ["det", "det", "nd"]
    (
        (_t0, det_args, det_kwargs),
        (_t1, true_args, true_kwargs),
        (_t2, nd_args, nd_kwargs),
    ) = calls
    assert det_args[: len(args)] == args and det_args == true_args == nd_args
    assert "deterministic" not in det_kwargs and det_kwargs == true_kwargs
    if name == "is_cake_mhc_train_supported":
        assert nd_kwargs == det_kwargs
    else:
        assert nd_kwargs == {**det_kwargs, "deterministic": False}
    assert facade._DET_FALLBACK_WARNED is False
    for warn_only in (False, True):
        monkeypatch.setattr(facade, "_DET_FALLBACK_WARNED", False)
        calls.clear()
        with (
            _torch_deterministic(True, warn_only=warn_only),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            warnings.simplefilter("ignore", ExperimentalWarning)
            for _ in range(3):
                assert function(*args, deterministic=False) == "det"
            assert function(*args) == "det"
        assert [tag for tag, _a, _k in calls] == ["det"] * 4
        assert all(kwargs == det_kwargs for _t, _a, kwargs in calls)
        fallback = _fallback_warnings(caught)
        assert len(fallback) == 1, [str(w.message) for w in caught]
        assert fallback[0].category is UserWarning
        assert str(fallback[0].message).startswith(f"{name}: deterministic=False")
        assert not [w for w in caught if _ALERT in str(w.message)]
        assert facade._DET_FALLBACK_WARNED is True


@pytest.mark.cpu_only
def test_finish_follows_the_handle(monkeypatch) -> None:
    """``cake_mhc_train_finish`` (no mode keyword) calls the finish of the package whose prepare made the handle, also
    with PyTorch's deterministic algorithms on; any other handle is a ``TypeError``."""
    calls = _recorders(
        monkeypatch, "cake_mhc_train_finish", "cake_mhc_det_train_finish"
    )
    nd_handle = backend._MHCTrainHandle(None, ())
    det_handle = det_backend._MHCTrainHandle(None, ())
    for torch_mode in (False, True):
        calls.clear()
        with _torch_deterministic(torch_mode):
            assert facade.cake_mhc_train_finish("X", "B", nd_handle) == "nd"
            assert facade.cake_mhc_train_finish("X", "B", det_handle) == "det"
            for other in (object(), None, (None, ())):
                with pytest.raises(TypeError, match="handle"):
                    facade.cake_mhc_train_finish("X", "B", other)
        assert calls == [
            ("nd", ("X", "B", nd_handle), {"deterministic": False}),
            ("det", ("X", "B", det_handle), {}),
        ]


# ----------------------------------------------------------------------------------------------------------------
# On a device


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _sm_count() -> int:
    return loader.sm_count(_device().index)


def _inputs(T: int, seed: int = SEED) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(
            *shape, generator=generator, device=_device(), dtype=torch.float32
        ).to(dtype)

    i = {
        "X": randn(T, 4, 4096, dtype=BF16),
        "fn": (randn(24, 16384) * 1e-2).contiguous(),
        "scale": 1.0 + 0.1 * randn(3),
        "base": 0.1 * randn(24),
        "B": randn(T, 4096, dtype=BF16),
        "dY": randn(T, 4, 4096, dtype=BF16),
        "dU": randn(T, 4096, dtype=BF16),
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


def _assert_all_bits(got, want, label: str) -> None:
    for k, g, w in zip(_OUTPUTS, got, want, strict=True):
        assert _same_bits(g, w), f"{k} {label}: bits differ"


_FORWARD_ENTRIES = (
    "forward_prepare",
    "forward_stats",
    "forward_finish",
    "backward_prepare",
)


def _facade_entries() -> dict:
    return {k: getattr(facade, f"cake_mhc_train_{k}") for k in _FORWARD_ENTRIES}


def _det_entries() -> dict:
    return {k: getattr(det_facade, f"cake_mhc_det_train_{k}") for k in _FORWARD_ENTRIES}


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


def _autograd(prepare, finish, i: dict, **kwargs) -> tuple:
    """``(handle, Y, [X.grad, fn.grad, scale.grad, base.grad])`` of ``finish(X, U + B, prepare(X, fn, scale, base))``
    under the loss ``sum(Y * dY)``; ``kwargs`` go to ``prepare``."""
    leaves = [i[k].clone().requires_grad_(True) for k in _LEAVES]
    U, handle = prepare(*leaves, **kwargs)
    Y = finish(leaves[0], (U.float() + i["B"].float()).to(BF16), handle)
    (Y.float() * i["dY"].float()).sum().backward()
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
    """The deterministic package's outputs of ``i``: the forward entry points, the backward, an accumulating
    backward onto carried values, the autograd ``Y`` and gradients."""
    g = torch.Generator(device=_device()).manual_seed(SEED + 4)
    carried = {
        k: torch.randn(shape, generator=g, device=_device()) * 1e-3
        for k, shape in (("dfn", (24, 16384)), ("dscale", (3,)), ("dbase", (24,)))
    }
    accumulated = {k: v.clone() for k, v in carried.items()}
    det_facade.cake_mhc_det_train_backward_finish(
        *_args(i), accumulate=True, **accumulated
    )
    _h, Y, grads = _autograd(
        det_facade.cake_mhc_det_train_prepare, det_facade.cake_mhc_det_train_finish, i
    )
    return {
        "forward": _forward_outputs(_det_entries(), i),
        "backward": _det_backward(i),
        "carried": carried,
        "accumulated": accumulated,
        "Y": Y,
        "grads": grads,
    }


@requires_device
@pytest.mark.parametrize("T", (7, 2049))
def test_true_is_the_deterministic_package(T: int, monkeypatch) -> None:
    """The default and ``deterministic=True``, with the backend's launches refused: the workspace sizes and buffers,
    the forward entry points, ``backward_finish`` (default and caller workspaces, ``accumulate``) and the autograd
    prepare / finish (a deterministic handle) are bitwise the deterministic package's."""
    i = _inputs(T)
    ref = _references(i)
    dev = _device()
    det_bytes = det_backend.cake_mhc_det_train_backward_workspace_bytes(T, dev)
    with monkeypatch.context() as m:
        _refuse_launches(m, backend)
        for mode in ({}, {"deterministic": True}):
            what = f"T={T} {mode}"
            assert (
                facade.cake_mhc_train_backward_workspace_bytes(T, dev, **mode)
                == det_bytes
            )
            assert facade.cake_mhc_train_forward_workspace_bytes(
                T, dev, **mode
            ) == det_backend.cake_mhc_det_train_forward_workspace_bytes(T, dev)
            ws = facade.cake_mhc_train_backward_workspace(T, dev, **mode)
            assert ws.dtype == torch.uint8 and ws.numel() == det_bytes
            assert int(torch.count_nonzero(ws)) == 0
            got = _forward_outputs(_facade_entries(), i, **mode)
            for k, v in ref["forward"].items():
                assert _same_bits(got[k], v), f"{k} {what}"
            for workspace in (None, ws):
                got = facade.cake_mhc_train_backward_finish(
                    *_args(i), workspace=workspace, **mode
                )
                _assert_all_bits(
                    got, ref["backward"], f"{what} workspace={workspace is not None}"
                )
            accumulated = {k: v.clone() for k, v in ref["carried"].items()}
            facade.cake_mhc_train_backward_finish(
                *_args(i), accumulate=True, **accumulated, **mode
            )
            for k, v in accumulated.items():
                assert _same_bits(v, ref["accumulated"][k]), f"{k} accumulate {what}"
            handle, Y, grads = _autograd(
                facade.cake_mhc_train_prepare, facade.cake_mhc_train_finish, i, **mode
            )
            assert isinstance(handle, det_backend._MHCTrainHandle), what
            assert not isinstance(handle, backend._MHCTrainHandle), what
            assert _same_bits(Y, ref["Y"]), f"Y {what}"
            for name, g, w in zip(_LEAVES, grads, ref["grads"], strict=True):
                assert _same_bits(g, w), f"grad {name} {what}"


@requires_device
@pytest.mark.parametrize("T", (7, 2049))
def test_false_is_the_nondeterministic_backend(T: int, monkeypatch) -> None:
    """``deterministic=False``, with the deterministic package's launches refused: the backend's workspace sizes
    (smaller than the deterministic one); the forward entry points bitwise the deterministic package's;
    ``backward_finish`` (default and caller workspaces) with the deterministic ``dX`` / ``dscale`` / ``dbase`` bits
    and a ``dfn`` within FP32 rounding of the deterministic one and of the backend's own; the autograd prepare makes
    the backend's handle and the finish follows it (``Y`` and the ``X`` / ``scale`` / ``base`` gradients the
    deterministic bits, the ``fn`` gradient within FP32 rounding)."""
    i = _inputs(T)
    ref = _references(i)
    direct = [
        t.clone()
        for t in backend.cake_mhc_train_backward_finish(*_args(i), deterministic=False)
    ]
    dev = _device()
    nd_bytes = loader.workspace_bytes_upto("bwd", T, _sm_count(), _arch())
    with monkeypatch.context() as m:
        _refuse_launches(m, det_backend)
        assert (
            facade.cake_mhc_train_backward_workspace_bytes(T, dev, deterministic=False)
            == nd_bytes
        )
        assert nd_bytes < det_backend.cake_mhc_det_train_backward_workspace_bytes(
            T, dev
        )
        assert facade.cake_mhc_train_forward_workspace_bytes(
            T, dev, deterministic=False
        ) == loader.workspace_bytes_upto("fwd", T, _sm_count(), _arch())
        ws = facade.cake_mhc_train_backward_workspace(T, dev, deterministic=False)
        assert ws.dtype == torch.uint8 and ws.numel() == nd_bytes
        got = _forward_outputs(_facade_entries(), i, deterministic=False)
        for k, v in ref["forward"].items():
            assert _same_bits(got[k], v), f"{k} T={T} deterministic=False"
        for workspace in (None, ws):
            got = facade.cake_mhc_train_backward_finish(
                *_args(i), workspace=workspace, deterministic=False
            )
            what = f"T={T} deterministic=False workspace={workspace is not None}"
            for n, k in enumerate(_OUTPUTS):
                if k in _EXACT:
                    assert _same_bits(got[n], ref["backward"][n]), (
                        f"{k} {what} vs deterministic"
                    )
                    assert _same_bits(got[n], direct[n]), f"{k} {what} vs the backend"
            assert torch.isfinite(got[1]).all(), what
            assert _rel_l2(got[1], ref["backward"][1]) <= DFN_REL_L2_MAX, what
            assert _rel_l2(got[1], direct[1]) <= DFN_REL_L2_MAX, what
        handle, Y, grads = _autograd(
            facade.cake_mhc_train_prepare,
            facade.cake_mhc_train_finish,
            i,
            deterministic=False,
        )
        assert isinstance(handle, backend._MHCTrainHandle)
        assert _same_bits(Y, ref["Y"])
        for index in (0, 2, 3):
            assert _same_bits(grads[index], ref["grads"][index]), _LEAVES[index]
        assert torch.isfinite(grads[1]).all()
        assert _rel_l2(grads[1], ref["grads"][1]) <= DFN_REL_L2_MAX


@requires_device
@pytest.mark.parametrize("warn_only", (False, True), ids=("strict", "warn_only"))
def test_torch_deterministic_mode_runs_the_deterministic_package(
    warn_only: bool, deterministic_algorithms, monkeypatch
) -> None:
    """Under ``torch.use_deterministic_algorithms(True[, warn_only=True])``, with the backend's launches refused:
    ``deterministic=False`` gives the deterministic package's workspace size and bits on every entry point (the
    forward entry points, ``backward_finish``, the autograd prepare with a deterministic handle and its backward)
    without an alert, and the process warns once."""
    T = 129
    i = _inputs(T)
    ref = _references(i)
    dev = _device()
    monkeypatch.setattr(facade, "_DET_FALLBACK_WARNED", False)
    with monkeypatch.context() as m:
        _refuse_launches(m, backend)
        deterministic_algorithms(True, warn_only=warn_only)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.simplefilter("ignore", ExperimentalWarning)
            assert facade.cake_mhc_train_backward_workspace_bytes(
                T, dev, deterministic=False
            ) == det_backend.cake_mhc_det_train_backward_workspace_bytes(T, dev)
            got = _forward_outputs(_facade_entries(), i, deterministic=False)
            for k, v in ref["forward"].items():
                assert _same_bits(got[k], v), k
            got = facade.cake_mhc_train_backward_finish(*_args(i), deterministic=False)
            _assert_all_bits(
                got, ref["backward"], "backward_finish under the deterministic mode"
            )
            handle, Y, grads = _autograd(
                facade.cake_mhc_train_prepare,
                facade.cake_mhc_train_finish,
                i,
                deterministic=False,
            )
        deterministic_algorithms(False)
    assert isinstance(handle, det_backend._MHCTrainHandle)
    assert _same_bits(Y, ref["Y"])
    for name, g, w in zip(_LEAVES, grads, ref["grads"], strict=True):
        assert _same_bits(g, w), f"grad {name}"
    fallback = _fallback_warnings(caught)
    assert len(fallback) == 1, [str(w.message) for w in caught]
    assert fallback[0].category is UserWarning
    assert not [
        w for w in caught if _ALERT in str(w.message) and "cake_mhc" in str(w.message)
    ]


# ----------------------------------------------------------------------------------------------------------------
# The companion workspace of the deterministic fallback


def _det_defaults() -> set[int]:
    return {ws.data_ptr() for ws in det_backend._DEFAULT_WORKSPACES.values()}


def _companions(workspace: torch.Tensor) -> list:
    entry = facade._COMPANIONS.get(id(workspace))
    if entry is None:
        return []
    assert entry[0]() is workspace
    return entry[1]


@requires_device
def test_fallback_runs_on_a_private_companion_workspace(
    deterministic_algorithms, monkeypatch
) -> None:
    """Under PyTorch's deterministic algorithms a ``deterministic=False`` ``backward_finish`` with a caller backward
    workspace of the non-deterministic sizing (too small for the deterministic layout) leaves that buffer untouched
    and runs on a companion: zero-initialized (its ticket region zero after every call), of the deterministic size of
    the token-count bucket, never one of the deterministic package's default buffers, reused by every later call it
    serves (also through the autograd prepare's ``backward_workspace``), a second one when the first is too small,
    and released with the caller's workspace.  A workspace of the deterministic sizing runs the deterministic
    backward in place (no companion); invalid workspaces and operands reach the deterministic checks unchanged and
    make no companion."""
    dev = _device()
    sm, arch = _sm_count(), _arch()
    monkeypatch.setattr(facade, "_COMPANIONS", {})
    monkeypatch.setattr(facade, "_DET_FALLBACK_WARNED", True)
    sequence = (
        129,
        200,
        2049,
        3000,
    )  # two token-count buckets: 129 / 200 share one, 2049 / 3000 another
    cases = {T: _inputs(T, seed=SEED + T) for T in sequence}
    want = {T: _det_backward(i) for T, i in cases.items()}
    ref_2049 = _references(cases[2049])
    # every caller workspace exists before the deterministic algorithms turn deterministic=False allocations
    # deterministic
    nd_ws = facade.cake_mhc_train_backward_workspace(
        max(sequence), dev, deterministic=False
    )
    nd_ws[256:].fill_(0xAB)
    before = nd_ws.clone()
    det_ws = facade.cake_mhc_train_backward_workspace(2049, dev)
    assert int(torch.count_nonzero(det_ws)) == 0
    fresh = facade.cake_mhc_train_backward_workspace(2049, dev, deterministic=False)
    deterministic_algorithms(True)
    for T in sequence:
        need = det_loader.workspace_bytes("bwd", T, sm, arch)
        # the premise: a valid deterministic=False workspace for T, too small for the deterministic layout
        assert loader.workspace_bytes("bwd", T, sm, arch) <= nd_ws.numel() < need, T
        existing = list(_companions(nd_ws))
        serving = next((c for c in existing if c.numel() >= need), None)
        got = facade.cake_mhc_train_backward_finish(
            *_args(cases[T]), workspace=nd_ws, deterministic=False
        )
        _assert_all_bits(got, want[T], f"T={T} on a companion")
        companions = _companions(nd_ws)
        assert all(a is b for a, b in zip(existing, companions, strict=False)), T
        if serving is not None:
            assert len(companions) == len(existing), (
                f"T={T}: a serving companion was not reused"
            )
            used = serving
        else:
            assert len(companions) == len(existing) + 1, T
            used = companions[-1]
            bucket = 1 << (T - 1).bit_length()
            assert (
                used.numel()
                == det_backend.cake_mhc_det_train_backward_workspace_bytes(bucket, dev)
            ), T
        assert used.dtype == torch.uint8 and used.device == dev and used.numel() >= need
        assert used.data_ptr() not in _det_defaults(), (
            "a companion is a deterministic default buffer"
        )
        assert used.data_ptr() != nd_ws.data_ptr()
        torch.cuda.synchronize()
        assert int(torch.count_nonzero(used[:256])) == 0, T
    companions = list(_companions(nd_ws))
    assert len(companions) == 2, [c.numel() for c in companions]
    assert torch.equal(nd_ws, before), "the fallback wrote the caller's workspace"
    # the autograd prepare's backward workspace takes the same companion
    handle, Y, grads = _autograd(
        facade.cake_mhc_train_prepare,
        facade.cake_mhc_train_finish,
        cases[2049],
        deterministic=False,
        backward_workspace=nd_ws,
    )
    assert isinstance(handle, det_backend._MHCTrainHandle)
    assert _same_bits(Y, ref_2049["Y"])
    for name, g, w in zip(_LEAVES, grads, ref_2049["grads"], strict=True):
        assert _same_bits(g, w), f"grad {name}"
    assert len(_companions(nd_ws)) == 2
    assert all(a is b for a, b in zip(companions, _companions(nd_ws), strict=True))
    assert torch.equal(nd_ws, before)
    # a workspace of the deterministic sizing runs the deterministic backward in place
    got = facade.cake_mhc_train_backward_finish(
        *_args(cases[2049]), workspace=det_ws, deterministic=False
    )
    _assert_all_bits(got, want[2049], "on a deterministic-sized workspace")
    assert id(det_ws) not in facade._COMPANIONS
    torch.cuda.synchronize()
    assert int(torch.count_nonzero(det_ws)) > 0, (
        "the deterministic backward did not run on the caller's workspace"
    )
    assert int(torch.count_nonzero(det_ws[:256])) == 0
    # invalid workspaces and operands pass through to the deterministic checks
    i = cases[2049]
    count = len(facade._COMPANIONS)
    nd_bytes = loader.workspace_bytes("bwd", 2049, sm, arch)
    shifted = torch.zeros(nd_bytes + 512, dtype=torch.uint8, device=dev)[
        1 : 1 + nd_bytes
    ]
    invalid = (
        (
            ValueError,
            "uint8",
            torch.zeros(nd_bytes // 4 + 1, dtype=torch.float32, device=dev),
        ),
        (ValueError, "holds", torch.zeros(256, dtype=torch.uint8, device=dev)),
        (ValueError, "uint8", torch.zeros(nd_bytes, dtype=torch.uint8)),
        (ValueError, "aligned", shifted),
        (TypeError, "workspace", [0] * 256),
    )
    for error, match, workspace in invalid:
        with pytest.raises(error, match=match):
            facade.cake_mhc_train_backward_finish(
                *_args(i), workspace=workspace, deterministic=False
            )
    with pytest.raises(ValueError, match="X"):
        facade.cake_mhc_train_backward_finish(
            i["X"][:, :3], *_args(i)[1:], workspace=fresh, deterministic=False
        )
    assert len(facade._COMPANIONS) == count and id(fresh) not in facade._COMPANIONS
    deterministic_algorithms(False)
    # released with the caller's workspace
    key = id(nd_ws)
    del nd_ws, companions, existing, serving, used, handle
    gc.collect()
    assert key not in facade._COMPANIONS


@requires_device
def test_fallback_under_cuda_graph_capture(
    deterministic_algorithms, monkeypatch
) -> None:
    """Under PyTorch's deterministic algorithms a ``deterministic=False`` ``backward_finish`` captured with a caller
    workspace of the non-deterministic sizing and no companion yet is a ``ValueError`` naming the capture (nothing is
    allocated during the capture); after an eager call made the companion, the capture succeeds, keeps that companion
    and its replays with new inputs copied in place give the deterministic bits, the caller's workspace untouched."""
    T = 129
    dev = _device()
    monkeypatch.setattr(facade, "_COMPANIONS", {})
    monkeypatch.setattr(facade, "_DET_FALLBACK_WARNED", True)
    static = _inputs(T, seed=SEED + 20)
    nan = float("nan")
    outs = {
        "dX": torch.full((T, 4, 4096), nan, dtype=BF16, device=dev),
        "dfn": torch.full((24, 16384), nan, device=dev),
        "dscale": torch.full((3,), nan, device=dev),
        "dbase": torch.full((24,), nan, device=dev),
    }
    nd_ws = facade.cake_mhc_train_backward_workspace(T, dev, deterministic=False)
    before = nd_ws.clone()
    _det_backward(
        static
    )  # the deterministic package's modules exist before the capture
    torch.cuda.synchronize()
    deterministic_algorithms(True)

    def step() -> None:
        facade.cake_mhc_train_backward_finish(
            *_args(static), workspace=nd_ws, deterministic=False, **outs
        )

    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(ValueError, match="CUDA graph capture"),
        torch.cuda.graph(graph),
    ):
        step()
    assert _companions(nd_ws) == []
    del graph
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    companions = list(_companions(nd_ws))
    assert len(companions) == 1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    torch.cuda.synchronize()
    assert len(_companions(nd_ws)) == 1 and _companions(nd_ws)[0] is companions[0]
    for r in range(3):
        new = _inputs(T, seed=SEED + 30 + r)
        reference = _det_backward(new)
        for k in _ARG_KEYS:
            static[k].copy_(new[k])
        for v in outs.values():
            v.fill_(nan)
        graph.replay()
        torch.cuda.synchronize()
        _assert_all_bits([outs[k] for k in _OUTPUTS], reference, f"replay {r}")
    deterministic_algorithms(False)
    assert torch.equal(nd_ws, before), (
        "the captured fallback wrote the caller's workspace"
    )
    del graph
