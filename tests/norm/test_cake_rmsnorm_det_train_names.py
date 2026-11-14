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

Names and modes of the two Cake fused BF16 RMSNorm training packages (no device).

The mode is part of the name, and only of the name (flashinfer issue #8928).
The deterministic package (the ``flashinfer.cake_rmsnorm_det_train`` facade and
its backend ``flashinfer.experimental.cake.cake_rmsnorm_det_train``) and the
non-deterministic package (the ``flashinfer.cake_rmsnorm_nondet_train`` facade
and its backend ``flashinfer.experimental.cake.cake_rmsnorm_nondet_train``)
export only constants (``ARCHES``, ``SUPPORTED_COMPUTE_CAPABILITIES``) and
names that carry their mode word (``det`` / ``nondet`` as a ``_``-separated
token, ``Det`` / ``Nondet`` in a CamelCase name); no name of one package
resolves on the other.  No entry point of either package takes
``deterministic``: passing it raises ``TypeError``.  The names from before the
rename fail at import: the former modules raise ``ModuleNotFoundError``, the
former names raise ``ImportError`` (checked in fresh interpreters).  The
``flashinfer`` top level serves the five det entry points and the module
``cake_rmsnorm_nondet_train`` as rows of its lazy-export table
(``flashinfer._LAZY_EXPORTS``), and no name of the non-deterministic package.
Nothing loads a backend early (fresh interpreters): ``import flashinfer``
loads no RMSNorm module, importing a facade loads no backend package, and the
first use of a facade's entry point or backend object loads that package's
backend only; each lazily resolved name is the backend's object.  The facade
entry points carry the backend's written docstrings.
Under ``torch.use_deterministic_algorithms(True)`` only the non-deterministic
backward alerts (``RuntimeError`` naming the det entry point; with
``warn_only=True`` one ``UserWarning``, then its own backward); its forward and
helpers and the whole det package never do.  The non-deterministic package is
self-contained (its own registry, submodules, star import and ``dir``).
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer
import flashinfer.cake_rmsnorm_det_train as det_facade
import flashinfer.cake_rmsnorm_nondet_train as facade
import flashinfer.experimental.cake.cake_rmsnorm_det_train as det_backend
import flashinfer.experimental.cake.cake_rmsnorm_nondet_train as backend

pytestmark = pytest.mark.cpu_only

FACADE = "flashinfer.cake_rmsnorm_nondet_train"
DET_FACADE = "flashinfer.cake_rmsnorm_det_train"
PACKAGE = "flashinfer.experimental.cake.cake_rmsnorm_nondet_train"
DET_PACKAGE = "flashinfer.experimental.cake.cake_rmsnorm_det_train"
SUBMODULES = ("cake_build", "cake_jit")

# (non-det name, det name) of the entry points and helpers of the same role.
PAIRS = (
    ("CakeRMSNormNondetFunction", "CakeRMSNormDetFunction"),
    ("cake_rmsnorm_nondet", "cake_rmsnorm_det"),
    ("cake_rmsnorm_nondet_train_backward", "cake_rmsnorm_det_train_backward"),
    (
        "cake_rmsnorm_nondet_train_backward_workspace",
        "cake_rmsnorm_det_train_backward_workspace",
    ),
    (
        "cake_rmsnorm_nondet_train_backward_workspace_bytes",
        "cake_rmsnorm_det_train_backward_workspace_bytes",
    ),
    ("cake_rmsnorm_nondet_train_forward", "cake_rmsnorm_det_train_forward"),
    ("cake_rmsnorm_nondet_train_route_key", "cake_rmsnorm_det_train_route_key"),
    ("is_cake_rmsnorm_nondet_train_supported", "is_cake_rmsnorm_det_train_supported"),
)
CONSTANTS = ("ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES")
NONDET_NAMES = tuple(nondet for nondet, _det in PAIRS)
DET_NAMES = (
    *(det for _nondet, det in PAIRS),
    "cake_rmsnorm_det_train_program_windows",
    "cake_rmsnorm_det_train_weight_dtypes",
)
# The ``__all__`` of the facade and of the backend package of each mode.
NONDET_ALL = sorted([*CONSTANTS, *NONDET_NAMES])
DET_ALL = sorted([*CONSTANTS, *DET_NAMES])
# The names the ``flashinfer`` top level exports: five det entry points.
TOP_LEVEL_DET = (
    "CakeRMSNormDetFunction",
    "cake_rmsnorm_det",
    "cake_rmsnorm_det_train_backward",
    "cake_rmsnorm_det_train_backward_workspace_bytes",
    "cake_rmsnorm_det_train_forward",
)
# Their rows of ``flashinfer._LAZY_EXPORTS`` (``name -> (module, attribute)``),
# with the modules of both facades.
TOP_LEVEL_ROWS = {
    **{name: (".cake_rmsnorm_det_train", name) for name in TOP_LEVEL_DET},
    "cake_rmsnorm_det_train": (".cake_rmsnorm_det_train", None),
    "cake_rmsnorm_nondet_train": (".cake_rmsnorm_nondet_train", None),
}
# The keys of another family that share the ``cake_rmsnorm`` prefix.
OTHER_FAMILY_PREFIXES = ("cake_rmsnorm_silu_gate", "CakeRMSNormSiLUGate")
# The RMSNorm modules: the two facades and the two backend packages (with
# their submodules).
RMSNORM_MODULES = (DET_FACADE, FACADE, DET_PACKAGE, PACKAGE)
# The modules the RMSNorm rows of the lazy-export table may name.
ROW_MODULES = (
    ".cake_rmsnorm_det_train",
    ".cake_rmsnorm_nondet_train",
    DET_FACADE,
    FACADE,
)
NAMESPACES = {
    "det_facade": (det_facade, "det"),
    "det_backend": (det_backend, "det"),
    "nondet_facade": (facade, "nondet"),
    "nondet_backend": (backend, "nondet"),
}
DET_NAMESPACES = ("det_backend", "det_facade")
NONDET_NAMESPACES = ("nondet_backend", "nondet_facade")
# The module constants keep their names (no mode word).
CONSTANT = re.compile(r"^[A-Z][A-Z0-9_]*$")

# The names from before the rename: they fail at import.
OLD_MODULES = (
    "flashinfer.cake_rmsnorm_train",
    "flashinfer.cake_rmsnorm_train_backend",
    "flashinfer.cake_rmsnorm_det_train_backend",
)
OLD_NONDET_NAMES = (
    "CakeRMSNormFunction",
    "cake_rmsnorm",
    "cake_rmsnorm_train_backward",
    "cake_rmsnorm_train_backward_workspace",
    "cake_rmsnorm_train_backward_workspace_bytes",
    "cake_rmsnorm_train_forward",
    "is_cake_rmsnorm_train_supported",
)
OLD_HELPERS = ("program_windows", "route_key", "weight_dtypes")
OLD_NAMES = (*OLD_NONDET_NAMES, *OLD_HELPERS)


def _carries(name: str, mode: str) -> bool:
    """Whether ``name`` carries the mode word ``mode`` (``det`` / ``nondet``): a
    ``_``-separated token, or the CamelCase part ``Det`` / ``Nondet``."""
    if mode in name.split("_"):
        return True
    return re.search(rf"{mode.capitalize()}(?=[A-Z0-9]|$)", name) is not None


def _cpu_tensors():
    x = torch.zeros(4, 512, dtype=torch.bfloat16)
    w = torch.ones(512, dtype=torch.bfloat16)
    rstd = torch.ones(4, dtype=torch.float32)
    return x, w, rstd


def _calls(namespace: str, **flag) -> dict:
    """Every callable of ``namespace``'s ``__all__``, called on CPU tensors with
    the keyword arguments ``flag`` (the autograd function with the values of
    ``flag`` as extra forward inputs)."""
    module, mode = NAMESPACES[namespace]
    stem = f"cake_rmsnorm_{mode}_train"
    x, w, rstd = _cpu_tensors()
    extra = tuple(flag.values())
    arch = module.ARCHES[0]
    calls = {
        f"{stem}_forward": functools.partial(
            getattr(module, f"{stem}_forward"), x, w, 1e-6, **flag
        ),
        f"{stem}_backward": functools.partial(
            getattr(module, f"{stem}_backward"), x, x, w, rstd, **flag
        ),
        f"{stem}_backward_workspace": functools.partial(
            getattr(module, f"{stem}_backward_workspace"), 4, 512, "cpu", **flag
        ),
        f"{stem}_backward_workspace_bytes": functools.partial(
            getattr(module, f"{stem}_backward_workspace_bytes"), 4, 512, "cpu", **flag
        ),
        f"{stem}_route_key": functools.partial(
            getattr(module, f"{stem}_route_key"), 512, "bwd", False, **flag
        ),
        f"is_{stem}_supported": functools.partial(
            getattr(module, f"is_{stem}_supported"), "cpu", 512, **flag
        ),
        f"cake_rmsnorm_{mode}": functools.partial(
            getattr(module, f"cake_rmsnorm_{mode}"), x, w, 1e-6, **flag
        ),
        f"CakeRMSNorm{mode.capitalize()}Function": functools.partial(
            getattr(module, f"CakeRMSNorm{mode.capitalize()}Function").apply,
            x,
            w,
            1e-6,
            None,
            *extra,
        ),
    }
    if mode == "det":
        hidden = det_backend.cake_jit.supported_hidden_sizes(arch)[0]
        calls[f"{stem}_program_windows"] = functools.partial(
            module.cake_rmsnorm_det_train_program_windows,
            arch,
            hidden,
            "fwd",
            False,
            **flag,
        )
        calls[f"{stem}_weight_dtypes"] = functools.partial(
            module.cake_rmsnorm_det_train_weight_dtypes,
            arch,
            hidden,
            "fwd",
            False,
            **flag,
        )
    return calls


def _names(text: str, name: str) -> bool:
    """Whether ``text`` names ``name`` itself, not a longer name it begins
    (``cake_rmsnorm_det`` begins ``cake_rmsnorm_det_train_backward``)."""
    return re.search(rf"{re.escape(name)}\b", text) is not None


@pytest.fixture
def no_determinism():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


# --------------------------------------------------------------------------- names


@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
def test_public_names_are_constants_or_carry_their_mode(namespace: str) -> None:
    """Every name of ``__all__`` is a module constant or carries the mode word
    of its package and not the other package's."""
    module, mode = NAMESPACES[namespace]
    other = "det" if mode == "nondet" else "nondet"
    assert sorted(module.__all__) == (DET_ALL if mode == "det" else NONDET_ALL)
    for name in module.__all__:
        assert hasattr(module, name), (namespace, name)
        if CONSTANT.match(name):
            continue
        assert _carries(name, mode), (namespace, name)
        assert not _carries(name, other), (namespace, name)


def test_mode_word_rule() -> None:
    """The mode-word check itself: ``Nondet`` is not ``Det`` and ``nondet`` is
    not ``det``."""
    assert _carries("CakeRMSNormDetFunction", "det")
    assert not _carries("CakeRMSNormDetFunction", "nondet")
    assert _carries("CakeRMSNormNondetFunction", "nondet")
    assert not _carries("CakeRMSNormNondetFunction", "det")
    assert _carries("cake_rmsnorm_nondet_train_forward", "nondet")
    assert not _carries("cake_rmsnorm_nondet_train_forward", "det")
    assert not _carries("cake_rmsnorm", "det")
    assert not _carries("route_key", "nondet")


@pytest.mark.parametrize("namespace", NONDET_NAMESPACES)
@pytest.mark.parametrize("name", DET_NAMES)
def test_nondet_package_resolves_no_det_name(namespace: str, name: str) -> None:
    module = NAMESPACES[namespace][0]
    assert not hasattr(module, name), f"{module.__name__}.{name} resolves"
    assert name not in dir(module), (namespace, name)
    if namespace.endswith("_backend"):
        assert "__getattr__" not in vars(module), namespace


@pytest.mark.parametrize("namespace", DET_NAMESPACES)
@pytest.mark.parametrize("name", NONDET_NAMES)
def test_det_package_resolves_no_nondet_name(namespace: str, name: str) -> None:
    module = NAMESPACES[namespace][0]
    assert not hasattr(module, name), f"{module.__name__}.{name} resolves"
    assert name not in dir(module), (namespace, name)
    if namespace.endswith("_backend"):
        assert "__getattr__" not in vars(module), namespace


@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
@pytest.mark.parametrize("name", OLD_NAMES)
def test_no_name_from_before_the_rename_resolves(namespace: str, name: str) -> None:
    module = NAMESPACES[namespace][0]
    assert not hasattr(module, name), f"{module.__name__}.{name} resolves"
    assert name not in dir(module), (namespace, name)


@pytest.mark.parametrize("namespace", NONDET_NAMESPACES)
def test_nondet_package_never_imports_the_det_package(namespace: str) -> None:
    """No import of the det package anywhere in the non-det facade or backend
    (module level or inside a function), and no import by name."""
    module = NAMESPACES[namespace][0]
    source = inspect.getsource(module)
    tree = ast.parse(source)
    imports = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not [line for line in imports if "cake_rmsnorm_det_train" in line], imports
    assert "import_module" not in source, namespace


# --------------------------------------------------------------------------- no ``deterministic`` keyword


def _entries(module):
    """``(name, callable)`` of every callable of ``module.__all__`` (the autograd
    function's ``forward`` for the class)."""
    for name in module.__all__:
        entry = getattr(module, name)
        if inspect.isclass(entry):
            yield name, entry.forward
        elif callable(entry):
            yield name, entry


@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
def test_no_entry_point_has_a_deterministic_parameter(namespace: str) -> None:
    module = NAMESPACES[namespace][0]
    entries = dict(_entries(module))
    assert set(entries) == set(module.__all__) - set(CONSTANTS), namespace
    for name, entry in entries.items():
        parameters = inspect.signature(entry).parameters
        assert "deterministic" not in parameters, (namespace, name)


def test_autograd_call_shapes() -> None:
    """Both autograd functionals take ``(x, w, eps, residual)`` and both autograd
    functions' forwards ``(ctx, x, w, eps, residual)``."""
    for module, mode in NAMESPACES.values():
        functional = getattr(module, f"cake_rmsnorm_{mode}")
        assert list(inspect.signature(functional).parameters) == [
            "x",
            "w",
            "eps",
            "residual",
        ]
        function = getattr(module, f"CakeRMSNorm{mode.capitalize()}Function")
        assert list(inspect.signature(function.forward).parameters) == [
            "ctx",
            "x",
            "w",
            "eps",
            "residual",
        ]


@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
@pytest.mark.parametrize("value", [True, False], ids=["True", "False"])
def test_the_deterministic_keyword_is_a_type_error(
    namespace: str, value: bool, no_determinism
) -> None:
    """Passing ``deterministic`` to any entry point of either package is
    Python's ``TypeError`` (an unexpected keyword argument; for the autograd
    function, an extra forward input), whatever its value."""
    module = NAMESPACES[namespace][0]
    calls = _calls(namespace, deterministic=value)
    assert set(calls) == set(module.__all__) - set(CONSTANTS), namespace
    for name, call in calls.items():
        if name.startswith("CakeRMSNorm"):
            expected = "positional argument"
        else:
            expected = "unexpected keyword argument 'deterministic'"
        with pytest.raises(TypeError, match=expected):
            call()


# --------------------------------------------------------------------------- torch.use_deterministic_algorithms


@pytest.mark.parametrize("namespace", NONDET_NAMESPACES)
@pytest.mark.parametrize("algorithms", ["on", "warn_only"])
def test_only_the_nondet_backward_alerts(
    namespace: str, algorithms: str, no_determinism
) -> None:
    """With ``torch.use_deterministic_algorithms(True)`` the non-det backward
    raises ``RuntimeError`` naming the det backward (``warn_only``: one
    ``UserWarning`` naming it, then this package's own CUDA check); the
    forward, the autograd forward, the workspace helpers, the route helper and
    the support query neither warn nor raise ``RuntimeError``."""
    torch.use_deterministic_algorithms(True, warn_only=algorithms == "warn_only")
    det_name = "flashinfer.cake_rmsnorm_det_train.cake_rmsnorm_det_train_backward"
    for name, call in _calls(namespace).items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if name == "cake_rmsnorm_nondet_train_backward" and algorithms == "on":
                with pytest.raises(RuntimeError) as info:
                    call()
                text = str(info.value)
                assert _names(text, det_name), text
                assert _names(text, f"{FACADE}.{name}"), text
                assert "deterministic=False" not in text, text
            elif name == "is_cake_rmsnorm_nondet_train_supported":
                assert call() is False
            elif name == "cake_rmsnorm_nondet_train_route_key":
                assert call() == "bwd_h512"
            else:
                with pytest.raises(ValueError, match="requires CUDA tensors"):
                    call()
        messages = [str(item.message) for item in caught]
        if name == "cake_rmsnorm_nondet_train_backward" and algorithms == "warn_only":
            assert len(messages) == 1, messages
            assert _names(messages[0], det_name), messages[0]
            assert "deterministic=False" not in messages[0], messages[0]
        else:
            assert messages == [], (name, messages)


@pytest.mark.parametrize("namespace", DET_NAMESPACES)
@pytest.mark.parametrize("algorithms", ["on", "warn_only"])
def test_det_package_ignores_deterministic_algorithms(
    namespace: str, algorithms: str, no_determinism
) -> None:
    """The det package does not consult ``torch.use_deterministic_algorithms``:
    its source never reads the setting, and with it on (strict or
    ``warn_only``) every det entry point refuses a CPU tensor with its own
    ``ValueError`` only (no ``RuntimeError``, no warning), the helpers answer
    and the support query answers ``False``."""
    module = NAMESPACES[namespace][0]
    tree = ast.parse(inspect.getsource(module))
    consulted = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and "deterministic_algorithms" in node.attr
    ]
    assert consulted == [], (namespace, consulted)
    torch.use_deterministic_algorithms(True, warn_only=algorithms == "warn_only")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name, call in _calls(namespace).items():
            if name == "is_cake_rmsnorm_det_train_supported":
                assert call() is False
            elif name == "cake_rmsnorm_det_train_route_key":
                assert call() == "bwd_h512"
            elif name in (
                "cake_rmsnorm_det_train_program_windows",
                "cake_rmsnorm_det_train_weight_dtypes",
            ):
                assert call(), name
            else:
                with pytest.raises(ValueError, match="requires CUDA tensors"):
                    call()
    assert [str(item.message) for item in caught] == []


# --------------------------------------------------------------------------- the flashinfer top level


@pytest.mark.parametrize("name", sorted({*NONDET_NAMES, *OLD_NAMES}))
def test_top_level_binds_no_nondet_name(name: str) -> None:
    assert not hasattr(flashinfer, name), name
    assert name not in dir(flashinfer), name


@pytest.mark.parametrize("name", TOP_LEVEL_DET)
def test_top_level_det_name_is_the_det_object(name: str) -> None:
    assert getattr(flashinfer, name) is getattr(det_facade, name), name


def test_top_level_modules() -> None:
    assert flashinfer.cake_rmsnorm_nondet_train is facade
    assert flashinfer.cake_rmsnorm_det_train is det_facade
    for old in OLD_MODULES:
        assert not hasattr(flashinfer, old.rpartition(".")[2]), old
        assert old not in sys.modules, old
    assert sys.modules[FACADE] is not sys.modules[DET_FACADE]
    assert facade.CakeRMSNormNondetFunction is backend.CakeRMSNormNondetFunction
    assert det_facade.CakeRMSNormDetFunction is det_backend.CakeRMSNormDetFunction
    assert not issubclass(
        backend.CakeRMSNormNondetFunction, det_backend.CakeRMSNormDetFunction
    )


def _rmsnorm_row(name: str, module_name: str) -> bool:
    """Whether a row of ``flashinfer._LAZY_EXPORTS`` is RMSNorm's: it names an
    RMSNorm facade, or its key has the ``cake_rmsnorm`` prefix of no other
    family."""
    if module_name in ROW_MODULES:
        return True
    prefixed = name.startswith(("cake_rmsnorm", "CakeRMSNorm"))
    return prefixed and not name.startswith(OTHER_FAMILY_PREFIXES)


def test_top_level_rows_are_the_lazy_exports() -> None:
    """The RMSNorm names of the top level are exactly the seven rows of
    ``flashinfer._LAZY_EXPORTS``; ``dir(flashinfer)`` lists them and each
    resolves to its facade's object (the two module rows to the facade
    modules)."""
    rows = flashinfer._LAZY_EXPORTS
    family = {n: row for n, row in rows.items() if _rmsnorm_row(n, row[0])}
    assert family == TOP_LEVEL_ROWS
    for name in family:
        assert _carries(name, "det") or _carries(name, "nondet"), name
    removed = {*OLD_NAMES, *(old.rpartition(".")[2] for old in OLD_MODULES)}
    assert not removed & set(rows), sorted(removed & set(rows))
    listed = dir(flashinfer)
    for name, (module_name, attribute) in TOP_LEVEL_ROWS.items():
        assert name in listed, name
        module = importlib.import_module(module_name, "flashinfer")
        expected = module if attribute is None else getattr(module, attribute)
        assert getattr(flashinfer, name) is expected, name
    assert flashinfer.cake_rmsnorm_det is det_facade.cake_rmsnorm_det
    assert flashinfer.cake_rmsnorm_det_train is sys.modules[DET_FACADE]
    assert flashinfer.cake_rmsnorm_nondet_train is sys.modules[FACADE]


# --------------------------------------------------------------------------- lazy facades


def _defs(mode: str) -> tuple[str, ...]:
    """The three entry points a facade defines itself (``def``s)."""
    stem = f"cake_rmsnorm_{mode}_train"
    return (f"cake_rmsnorm_{mode}", f"{stem}_forward", f"{stem}_backward")


def _lazy(mode: str) -> list[str]:
    """The names of a facade's ``__all__`` that resolve to the backend's objects."""
    return sorted(set(DET_ALL if mode == "det" else NONDET_ALL) - set(_defs(mode)))


@pytest.mark.parametrize("namespace", ["det_facade", "nondet_facade"])
def test_facade_resolves_backend_objects_on_first_access(namespace: str) -> None:
    """Every name of the facade's ``__all__`` that it does not define is the
    backend's object itself, cached in the facade on first access and listed
    by ``dir``; any other name is an ``AttributeError``."""
    module, mode = NAMESPACES[namespace]
    package = NAMESPACES[f"{mode}_backend"][0]
    for name in _lazy(mode):
        assert getattr(module, name) is getattr(package, name), name
        assert vars(module)[name] is getattr(package, name), name
        assert name in dir(module), name
    for name in _defs(mode):
        assert name in vars(module), name
        assert getattr(module, name) is not getattr(package, name), name
    with pytest.raises(AttributeError, match="has no attribute 'no_such_name'"):
        module.no_such_name  # noqa: B018


def _written_docstrings(module) -> dict[str, str]:
    """The docstring each top-level function of ``module`` has in its source
    (cleaned), before any text the module appends at import."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return {
        node.name: ast.get_docstring(node) or ""
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


@pytest.mark.parametrize("namespace", ["det_facade", "nondet_facade"])
def test_facade_docstrings_are_the_backend_written_docstrings(namespace: str) -> None:
    """A facade entry point's docstring is the backend function's written
    docstring (without the program-selection notes the backend appends from its
    registry at import); the forward's and the backward's then add one "Notes"
    section naming the backend function whose docstring lists the windows."""
    module, mode = NAMESPACES[namespace]
    package = DET_PACKAGE if mode == "det" else PACKAGE
    written = _written_docstrings(NAMESPACES[f"{mode}_backend"][0])
    autograd = f"cake_rmsnorm_{mode}"
    for name in _defs(mode):
        doc = inspect.getdoc(getattr(module, name)) or ""
        assert written[name] and doc.startswith(written[name]), name
        notes = doc[len(written[name]) :]
        if name == autograd:
            assert notes == "", (name, notes)
            continue
        assert notes.startswith("\n\nNotes\n-----\n"), (name, notes)
        assert f":func:`{package}.{name}`" in notes, (name, notes)
        assert notes.count("\n\n") == 1, (name, notes)


def _loaded_after(*steps: str) -> list[list[str]]:
    """Run ``steps`` in order in a fresh interpreter: the RMSNorm modules
    (``RMSNORM_MODULES`` and their submodules) loaded after each step."""
    prefixes = tuple(f"{name}." for name in RMSNORM_MODULES)
    condition = f"m in {RMSNORM_MODULES!r} or m.startswith({prefixes!r})"
    code = [
        "import json, sys",
        "def loaded():",
        f"    return sorted(m for m in sys.modules if {condition})",
        "snapshots = []",
    ]
    for step in steps:
        code += [step, "snapshots.append(loaded())"]
    code.append("print(json.dumps(snapshots))")
    r = subprocess.run(
        [sys.executable, "-c", "\n".join(code)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, (steps, r.stdout, r.stderr)
    return json.loads(r.stdout.strip().splitlines()[-1])


def _modules(mode: str) -> tuple[str, str]:
    """The names of ``mode``'s facade module and backend package."""
    return (DET_FACADE, DET_PACKAGE) if mode == "det" else (FACADE, PACKAGE)


def _owned(name: str, mode: str) -> bool:
    """Whether the module ``name`` is ``mode``'s facade or in its backend package."""
    facade_name, package = _modules(mode)
    return name in (facade_name, package) or name.startswith(f"{package}.")


def test_import_flashinfer_loads_no_rmsnorm_module() -> None:
    """``import flashinfer`` (and ``dir(flashinfer)``) loads neither facade nor
    backend: the RMSNorm names are bound on first access only."""
    after_import, after_dir = _loaded_after(
        "import flashinfer", "listed = dir(flashinfer)"
    )
    assert after_import == [] and after_dir == []


# The first use of a facade that loads its backend: a backend object, a
# constant, and a call of an entry point (CPU tensors: the backend refuses them
# with its own ``ValueError``).
FIRST_USES = {
    "autograd_class": "facade.CakeRMSNorm{Mode}Function",
    "constant": "facade.ARCHES",
    "entry_point_call": (
        "import torch\n"
        "try:\n"
        "    facade.cake_rmsnorm_{mode}_train_forward(torch.zeros(4, 512,"
        " dtype=torch.bfloat16), torch.ones(512, dtype=torch.bfloat16), 1e-6)\n"
        "except ValueError as e:\n"
        "    assert 'requires CUDA tensors' in str(e), e\n"
        "else:\n"
        "    raise AssertionError('the forward accepted CPU tensors')"
    ),
}


@pytest.mark.parametrize("use", sorted(FIRST_USES))
@pytest.mark.parametrize("mode", ["det", "nondet"])
def test_facade_loads_only_its_own_backend_on_first_use(mode: str, use: str) -> None:
    """Importing a facade loads no backend package (its lazily resolved names are
    absent from the module until accessed); the first use loads that facade's
    backend package and nothing of the other package."""
    facade_name, package = _modules(mode)
    unresolved = "sorted(set(facade.__all__) - set(vars(facade)))"
    on_import, checked, on_use = _loaded_after(
        f"import {facade_name} as facade",
        f"assert {unresolved} == {_lazy(mode)!r}, {unresolved}",
        FIRST_USES[use].format(mode=mode, Mode=mode.capitalize()),
    )
    assert on_import == checked == [facade_name]
    assert package in on_use, on_use
    assert [name for name in on_use if not _owned(name, mode)] == [], on_use


def test_top_level_rows_load_their_facade_only() -> None:
    """A top-level det entry point loads the det facade only, each module row
    its facade only, and a backend object of a row its own backend."""
    on_import, det_entry, nondet_module, det_class = _loaded_after(
        "import flashinfer",
        "flashinfer.cake_rmsnorm_det",
        "flashinfer.cake_rmsnorm_nondet_train",
        "flashinfer.CakeRMSNormDetFunction",
    )
    assert on_import == []
    assert det_entry == [DET_FACADE]
    assert nondet_module == sorted([DET_FACADE, FACADE])
    assert DET_PACKAGE in det_class, det_class
    assert [name for name in det_class if not _owned(name, "det")] == [FACADE]


def test_top_level_det_module_row_loads_the_det_facade_only() -> None:
    """After a bare ``import flashinfer``, ``flashinfer.cake_rmsnorm_det_train``
    is the det facade module: the row loads that facade and no backend."""
    on_import, det_module = _loaded_after(
        "import flashinfer",
        f"assert flashinfer.cake_rmsnorm_det_train is sys.modules['{DET_FACADE}']",
    )
    assert on_import == []
    assert det_module == [DET_FACADE]


# --------------------------------------------------------------------------- the non-deterministic package is self-contained


def test_nondet_facade_is_its_own_module() -> None:
    assert importlib.import_module(FACADE) is facade
    assert Path(facade.__file__).name == "cake_rmsnorm_nondet_train.py"
    assert sorted(facade.__all__) == NONDET_ALL
    for name in facade.__all__:
        assert getattr(facade, name) is getattr(backend, name) or name in (
            "cake_rmsnorm_nondet",
            "cake_rmsnorm_nondet_train_backward",
            "cake_rmsnorm_nondet_train_forward",
        ), name


@pytest.mark.parametrize("submodule", SUBMODULES)
def test_nondet_package_submodule_is_its_own_module(submodule: str) -> None:
    mine = importlib.import_module(f"{PACKAGE}.{submodule}")
    theirs = importlib.import_module(f"{DET_PACKAGE}.{submodule}")
    assert mine is not theirs, submodule
    assert sys.modules[f"{PACKAGE}.{submodule}"] is mine, submodule
    assert Path(mine.__file__).parent == Path(backend.__file__).parent, submodule


def test_nondet_package_is_self_contained() -> None:
    assert sorted(backend.__all__) == NONDET_ALL
    # its own registry: the forward programs of the det package renamed, the
    # backward records of its single kernels and of the det pairs it renders
    # (the kernel library names keep their internal prefix)
    modules = backend.cake_jit.MODULES
    assert modules and all(name.startswith("cake_rmsnorm_train_") for name in modules)
    kinds = ("claim_sum_f64", "chunk_partials", "claim_part_f32")
    assert all(
        record["workspace_rule"] is None or record["workspace_rule"]["kind"] in kinds
        for record in modules.values()
    )
    files = {
        p.name
        for p in Path(backend.__file__).parent.iterdir()
        if p.name != "__pycache__"
    }
    assert {
        "__init__.py",
        "cake_build.py",
        "cake_jit.py",
        "cake_windows.py",
        "build.json",
        "csrc",
    } <= files
    for package in (PACKAGE, DET_PACKAGE):
        manifest = importlib.import_module(f"{package}.cake_build").manifest()
        assert manifest["package"] == package, package


def test_nondet_package_star_import_binds_its_own_objects() -> None:
    names: dict = {}
    exec(f"from {PACKAGE} import *", names)
    del names["__builtins__"]
    assert sorted(names) == NONDET_ALL
    for name in NONDET_ALL:
        assert names[name] is getattr(backend, name), name


def test_nondet_package_dir_lists_its_own_names() -> None:
    listed = set(dir(backend))
    assert set(NONDET_ALL) | set(SUBMODULES) <= listed
    assert not set(DET_NAMES) & listed


# Import forms in a fresh interpreter: the submodule path first, the attributes
# of flashinfer, the names and modules from before the rename, and a star
# import of the package.
FRESH = {
    "submodule_first": (
        "import importlib\n"
        f"mine = importlib.import_module('{PACKAGE}.cake_jit')\n"
        f"theirs = importlib.import_module('{DET_PACKAGE}.cake_jit')\n"
        "assert mine is not theirs and mine.MODULES is not theirs.MODULES\n"
        f"assert importlib.import_module('{FACADE}') is not importlib.import_module('{DET_FACADE}')\n"
    ),
    "flashinfer_attributes": (
        "import sys\nimport flashinfer\n"
        f"assert flashinfer.{FACADE.rpartition('.')[2]} is sys.modules['{FACADE}']\n"
        f"assert not hasattr(flashinfer, '{PAIRS[0][0]}')\n"
        f"assert flashinfer.{DET_FACADE.rpartition('.')[2]} is sys.modules['{DET_FACADE}']\n"
        f"assert flashinfer.{PAIRS[0][1]} is sys.modules['{DET_FACADE}'].{PAIRS[0][1]}\n"
        f"assert not hasattr(sys.modules['{DET_FACADE}'], '{PAIRS[0][0]}')\n"
    ),
    "old_modules_fail": (
        "import importlib\n"
        f"for name in {OLD_MODULES!r}:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except ModuleNotFoundError as e:\n"
        "        assert e.name == name, (name, e.name)\n"
        "    else:\n"
        "        raise AssertionError(f'{name} still imports')\n"
    ),
    "old_names_fail": (
        f"for module, names in (('flashinfer', {OLD_NAMES!r}),"
        f" ('{FACADE}', {OLD_NAMES!r}), ('{DET_FACADE}', {OLD_NAMES!r})):\n"
        "    for name in names:\n"
        "        try:\n"
        "            exec(f'from {module} import {name}', {})\n"
        "        except ImportError as e:\n"
        "            assert f\"cannot import name '{name}'\" in str(e), e\n"
        "        else:\n"
        "            raise AssertionError(f'{module}.{name} still imports')\n"
        "from flashinfer import cake_rmsnorm_det\n"
        f"from {FACADE} import cake_rmsnorm_nondet\n"
        "assert cake_rmsnorm_nondet is not cake_rmsnorm_det\n"
    ),
    "package_star_import": (
        "import importlib\n"
        f"from {PACKAGE} import *\n"
        f"mine = importlib.import_module('{PACKAGE}')\n"
        "assert all(globals()[n] is getattr(mine, n) for n in mine.__all__)\n"
        "assert 'cake_rmsnorm_det_train_forward' not in globals()\n"
    ),
}


@pytest.mark.parametrize("form", sorted(FRESH))
def test_import_forms_in_a_fresh_interpreter(form: str) -> None:
    r = subprocess.run(
        [sys.executable, "-c", FRESH[form]], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, (form, r.stdout, r.stderr)
