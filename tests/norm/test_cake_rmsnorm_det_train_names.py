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

Names and the ``deterministic`` flag of the two Cake fused BF16 RMSNorm training
packages (no device).

The deterministic package (the ``flashinfer.cake_rmsnorm_det_train`` facade and its
backend package) defines only names with ``det``: no name from before the det
rename resolves on it.  The non-deterministic package (the
``flashinfer.cake_rmsnorm_train`` facade and its backend package) resolves no det
name and never imports the deterministic package.  No det entry point takes
``deterministic`` (passing it raises ``TypeError``).  Every non-det entry point
requires ``deterministic=False``: any other value, the default ``True`` included,
raises ``ValueError`` naming the det entry point of the same role and saying
"pass deterministic=False for the non-deterministic path", before any warning;
``False`` passes the check.  The deterministic package does
not consult ``torch.use_deterministic_algorithms``.  The ``flashinfer`` top level
binds each name to its own package's object, and the non-deterministic package is
self-contained (its own registry, submodules, star import and ``dir``; checked
in fresh interpreters too).
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import flashinfer
import flashinfer.cake_rmsnorm_det_train as det_facade
import flashinfer.cake_rmsnorm_det_train_backend as det_backend
import flashinfer.cake_rmsnorm_train as facade
import flashinfer.cake_rmsnorm_train_backend as backend

pytestmark = pytest.mark.cpu_only

FACADE = "flashinfer.cake_rmsnorm_train"
DET_FACADE = "flashinfer.cake_rmsnorm_det_train"
PACKAGE = "flashinfer.cake_rmsnorm_train_backend"
DET_PACKAGE = "flashinfer.cake_rmsnorm_det_train_backend"
SUBMODULES = ("cake_build", "cake_jit")

# (non-det name, det name) of every entry point of the two packages.
PAIRS = (
    ("CakeRMSNormFunction", "CakeRMSNormDetFunction"),
    ("cake_rmsnorm", "cake_rmsnorm_det"),
    ("cake_rmsnorm_train_backward", "cake_rmsnorm_det_train_backward"),
    (
        "cake_rmsnorm_train_backward_workspace",
        "cake_rmsnorm_det_train_backward_workspace",
    ),
    (
        "cake_rmsnorm_train_backward_workspace_bytes",
        "cake_rmsnorm_det_train_backward_workspace_bytes",
    ),
    ("cake_rmsnorm_train_forward", "cake_rmsnorm_det_train_forward"),
    ("is_cake_rmsnorm_train_supported", "is_cake_rmsnorm_det_train_supported"),
)
NONDET_NAMES = tuple(nondet for nondet, _det in PAIRS)
DET_NAMES = tuple(det for _nondet, det in PAIRS)
DET_OF = dict(PAIRS)
# The names the ``flashinfer`` top level exports from each package.
TOP_LEVEL_NONDET = (
    "CakeRMSNormFunction",
    "cake_rmsnorm",
    "cake_rmsnorm_train_backward",
    "cake_rmsnorm_train_backward_workspace_bytes",
    "cake_rmsnorm_train_forward",
)
TOP_LEVEL_DET = (
    "CakeRMSNormDetFunction",
    "cake_rmsnorm_det",
    "cake_rmsnorm_det_train_backward",
    "cake_rmsnorm_det_train_backward_workspace_bytes",
    "cake_rmsnorm_det_train_forward",
)
DET_NAMESPACES = {"facade": det_facade, "backend": det_backend}
NONDET_NAMESPACES = {"facade": facade, "backend": backend}
# The ``__all__`` of the non-deterministic facade and backend package.
NONDET_ALL = sorted(
    ["ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES", "route_key", *NONDET_NAMES]
)
PHRASE = "pass deterministic=False for the non-deterministic path"
# ``deterministic`` omitted: the entry point's default.
DEFAULT = object()
# Every value of ``deterministic`` a non-det entry point refuses: anything but the
# object ``False``.
NOT_FALSE = {
    "default": DEFAULT,
    "True": True,
    "None": None,
    "0": 0,
    "1": 1,
    "str_False": "False",
    "np_False": np.bool_(False),
    "np_True": np.bool_(True),
    "tensor_False": torch.tensor(False),
}


def _cpu_tensors():
    x = torch.zeros(4, 512, dtype=torch.bfloat16)
    w = torch.ones(512, dtype=torch.bfloat16)
    rstd = torch.ones(4, dtype=torch.float32)
    return x, w, rstd


def _nondet_calls(module, flag) -> dict:
    """Every entry point of the non-det ``module``, called on CPU tensors with
    ``deterministic=flag`` (omitted for ``DEFAULT``; the autograd function takes
    it as its last forward input).  The flag is checked before any device
    query."""
    x, w, rstd = _cpu_tensors()
    kw = {} if flag is DEFAULT else {"deterministic": flag}
    last = () if flag is DEFAULT else (flag,)
    return {
        "cake_rmsnorm_train_forward": functools.partial(
            module.cake_rmsnorm_train_forward, x, w, 1e-6, **kw
        ),
        "cake_rmsnorm_train_backward": functools.partial(
            module.cake_rmsnorm_train_backward, x, x, w, rstd, **kw
        ),
        "cake_rmsnorm_train_backward_workspace": functools.partial(
            module.cake_rmsnorm_train_backward_workspace, 4, 512, "cpu", **kw
        ),
        "cake_rmsnorm_train_backward_workspace_bytes": functools.partial(
            module.cake_rmsnorm_train_backward_workspace_bytes, 4, 512, "cpu", **kw
        ),
        "is_cake_rmsnorm_train_supported": functools.partial(
            module.is_cake_rmsnorm_train_supported, "cpu", 512, **kw
        ),
        "cake_rmsnorm": functools.partial(module.cake_rmsnorm, x, w, 1e-6, **kw),
        "CakeRMSNormFunction": functools.partial(
            module.CakeRMSNormFunction.apply, x, w, 1e-6, None, *last
        ),
    }


def _det_calls(module, **flag) -> dict:
    """Every entry point of the det package ``module``, called on CPU tensors
    with the keyword arguments ``flag`` (the autograd function with
    ``flag["deterministic"]`` as an extra forward input when given)."""
    x, w, rstd = _cpu_tensors()
    extra = tuple(flag.values())
    return {
        "cake_rmsnorm_det_train_forward": functools.partial(
            module.cake_rmsnorm_det_train_forward, x, w, 1e-6, **flag
        ),
        "cake_rmsnorm_det_train_backward": functools.partial(
            module.cake_rmsnorm_det_train_backward, x, x, w, rstd, **flag
        ),
        "cake_rmsnorm_det_train_backward_workspace": functools.partial(
            module.cake_rmsnorm_det_train_backward_workspace, 4, 512, "cpu", **flag
        ),
        "cake_rmsnorm_det_train_backward_workspace_bytes": functools.partial(
            module.cake_rmsnorm_det_train_backward_workspace_bytes,
            4,
            512,
            "cpu",
            **flag,
        ),
        "is_cake_rmsnorm_det_train_supported": functools.partial(
            module.is_cake_rmsnorm_det_train_supported, "cpu", 512, **flag
        ),
        "cake_rmsnorm_det": functools.partial(
            module.cake_rmsnorm_det, x, w, 1e-6, **flag
        ),
        "CakeRMSNormDetFunction": functools.partial(
            module.CakeRMSNormDetFunction.apply, x, w, 1e-6, None, *extra
        ),
    }


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


@pytest.mark.parametrize("namespace", sorted(DET_NAMESPACES))
@pytest.mark.parametrize("name", NONDET_NAMES)
def test_det_package_has_no_name_from_before_the_rename(
    namespace: str, name: str
) -> None:
    module = DET_NAMESPACES[namespace]
    assert not hasattr(module, name), f"{module.__name__}.{name} resolves"
    assert name not in dir(module), (namespace, name)
    assert name not in module.__all__, (namespace, name)


@pytest.mark.parametrize("namespace", sorted(DET_NAMESPACES))
def test_det_package_rmsnorm_names_carry_det(namespace: str) -> None:
    """Every public RMSNorm name of the det package carries ``det``; its det
    names are in ``__all__``."""
    module = DET_NAMESPACES[namespace]
    public = [n for n in dir(module) if not n.startswith("_")]
    rmsnorm = [n for n in public if "rmsnorm" in n.lower()]
    assert rmsnorm, namespace
    assert all("det" in n.lower() for n in rmsnorm), sorted(
        n for n in rmsnorm if "det" not in n.lower()
    )
    assert set(DET_NAMES) <= set(module.__all__), namespace


@pytest.mark.parametrize("namespace", sorted(NONDET_NAMESPACES))
@pytest.mark.parametrize("name", DET_NAMES)
def test_nondet_package_resolves_no_det_name(namespace: str, name: str) -> None:
    module = NONDET_NAMESPACES[namespace]
    assert not hasattr(module, name), f"{module.__name__}.{name} resolves"
    with pytest.raises(AttributeError):
        getattr(module, name)
    assert name not in dir(module), (namespace, name)
    assert "__getattr__" not in vars(module), namespace


@pytest.mark.parametrize("namespace", sorted(NONDET_NAMESPACES))
def test_nondet_package_never_imports_the_det_package(namespace: str) -> None:
    """No import of the det package anywhere in the non-det facade or backend
    (module level or inside a function), and no import by name."""
    module = NONDET_NAMESPACES[namespace]
    source = inspect.getsource(module)
    tree = ast.parse(source)
    imports = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not [line for line in imports if "cake_rmsnorm_det_train" in line], imports
    assert "import_module" not in source, namespace
    public = [n for n in dir(module) if not n.startswith("_") and "rmsnorm" in n]
    assert not [n for n in public if "det" in n.lower()], public


# --------------------------------------------------------------------------- the deterministic flag


def _det_signatures():
    """``(module, name, signature)`` of every det entry point (the autograd
    function's forward)."""
    for module in (det_facade, det_backend):
        for name in DET_NAMES:
            entry = getattr(module, name)
            if name == "CakeRMSNormDetFunction":
                entry = entry.forward
            yield module, name, inspect.signature(entry)


def test_det_entries_take_no_deterministic_parameter() -> None:
    for module, name, signature in _det_signatures():
        assert "deterministic" not in signature.parameters, (module, name)


def test_nondet_flag_defaults_to_true_and_is_keyword_only() -> None:
    """``deterministic`` defaults to ``True`` (which raises) on every non-det entry
    point: keyword-only, but on ``cake_rmsnorm`` (its call shape is kept) and the
    autograd function's forward (``apply`` takes positional inputs only), where it
    is the last parameter."""
    for name in NONDET_NAMES:
        for module in (facade, backend):
            entry = getattr(module, name)
            if name == "CakeRMSNormFunction":
                entry = entry.forward
            parameters = inspect.signature(entry).parameters
            parameter = parameters["deterministic"]
            assert parameter.default is True, (module, name)
            if name in ("cake_rmsnorm", "CakeRMSNormFunction"):
                assert list(parameters)[-1] == "deterministic", (module, name)
                assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            else:
                assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (module, name)


@pytest.mark.parametrize("namespace", sorted(DET_NAMESPACES))
@pytest.mark.parametrize("entry", DET_NAMES)
@pytest.mark.parametrize("value", [True, False], ids=["True", "False"])
def test_det_entry_rejects_the_deterministic_keyword(
    namespace: str, entry: str, value: bool, no_determinism
) -> None:
    """Passing ``deterministic`` to a det entry point is Python's ``TypeError``
    (an unexpected keyword argument; for the autograd function, an extra
    forward input), whatever its value."""
    call = _det_calls(DET_NAMESPACES[namespace], deterministic=value)[entry]
    if entry == "CakeRMSNormDetFunction":
        expected = "positional argument"
    else:
        expected = "unexpected keyword argument 'deterministic'"
    with pytest.raises(TypeError, match=expected):
        call()


@pytest.mark.parametrize("namespace", sorted(NONDET_NAMESPACES))
@pytest.mark.parametrize("entry", NONDET_NAMES)
@pytest.mark.parametrize("value", list(NOT_FALSE.values()), ids=list(NOT_FALSE))
@pytest.mark.parametrize("algorithms", ["off", "on", "warn_only"])
def test_nondet_entry_rejects_every_value_but_false(
    namespace: str, entry: str, value, algorithms: str, no_determinism
) -> None:
    """The default and every value of ``deterministic`` but the object ``False``
    raise ``ValueError`` naming the det entry point of the same role and saying
    how to request this package, before any device query and whatever
    ``torch.use_deterministic_algorithms`` says (no warning, no
    ``RuntimeError``)."""
    call = _nondet_calls(NONDET_NAMESPACES[namespace], value)[entry]
    if algorithms != "off":
        torch.use_deterministic_algorithms(True, warn_only=algorithms == "warn_only")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as info:
            call()
    assert [str(item.message) for item in caught] == []
    text = str(info.value)
    assert _names(text, f"flashinfer.cake_rmsnorm_det_train.{DET_OF[entry]}"), text
    assert _names(text, f"flashinfer.cake_rmsnorm_train.{entry}"), text
    assert PHRASE in text, text


@pytest.mark.parametrize("namespace", sorted(NONDET_NAMESPACES))
@pytest.mark.parametrize("entry", NONDET_NAMES)
def test_nondet_entry_runs_with_false(
    namespace: str, entry: str, no_determinism
) -> None:
    """``deterministic=False`` passes the flag check: the support query answers
    ``False`` for the CPU, every other entry point goes on to this package's
    CUDA check (here: CPU tensors)."""
    call = _nondet_calls(NONDET_NAMESPACES[namespace], False)[entry]
    if entry == "is_cake_rmsnorm_train_supported":
        assert call() is False
        return
    with pytest.raises(ValueError, match="requires CUDA tensors") as info:
        call()
    assert PHRASE not in str(info.value)


@pytest.mark.parametrize("namespace", sorted(DET_NAMESPACES))
@pytest.mark.parametrize("algorithms", ["on", "warn_only"])
def test_det_package_ignores_deterministic_algorithms(
    namespace: str, algorithms: str, no_determinism
) -> None:
    """The det package does not consult ``torch.use_deterministic_algorithms``:
    its source never reads the setting, and with it on (strict or
    ``warn_only``) every det entry point refuses a CPU tensor with its own
    ``ValueError`` only (no ``RuntimeError``, no warning) and the support query
    answers ``False``."""
    module = DET_NAMESPACES[namespace]
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
        for name, call in _det_calls(module).items():
            if name == "is_cake_rmsnorm_det_train_supported":
                continue
            with pytest.raises(ValueError, match="requires CUDA tensors"):
                call()
        assert module.is_cake_rmsnorm_det_train_supported("cpu") is False
        assert module.is_cake_rmsnorm_det_train_supported("cpu", 512) is False
    assert [str(item.message) for item in caught] == []


# --------------------------------------------------------------------------- the flashinfer top level


@pytest.mark.parametrize("name", TOP_LEVEL_NONDET)
def test_top_level_nondet_name_is_the_nondet_object(name: str) -> None:
    assert getattr(flashinfer, name) is getattr(facade, name), name
    assert getattr(flashinfer, name) is not getattr(flashinfer, DET_OF[name]), name


@pytest.mark.parametrize("name", TOP_LEVEL_DET)
def test_top_level_det_name_is_the_det_object(name: str) -> None:
    assert getattr(flashinfer, name) is getattr(det_facade, name), name


def test_top_level_modules() -> None:
    assert flashinfer.cake_rmsnorm_train is facade
    assert flashinfer.cake_rmsnorm_train_backend is backend
    assert sys.modules[FACADE] is not sys.modules[DET_FACADE]
    assert facade.CakeRMSNormFunction is backend.CakeRMSNormFunction
    assert det_facade.CakeRMSNormDetFunction is det_backend.CakeRMSNormDetFunction
    assert not issubclass(
        backend.CakeRMSNormFunction, det_backend.CakeRMSNormDetFunction
    )


# --------------------------------------------------------------------------- the non-deterministic package is self-contained


def test_nondet_facade_is_its_own_module() -> None:
    assert importlib.import_module(FACADE) is facade
    assert Path(facade.__file__).name == "cake_rmsnorm_train.py"
    assert sorted(facade.__all__) == NONDET_ALL
    for name in facade.__all__:
        assert getattr(facade, name) is getattr(backend, name) or name in (
            "cake_rmsnorm",
            "cake_rmsnorm_train_backward",
            "cake_rmsnorm_train_forward",
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
    modules = backend.cake_jit.MODULES
    assert modules and all(name.startswith("cake_rmsnorm_train_") for name in modules)
    assert all(
        record["workspace_rule"] is None
        or record["workspace_rule"]["kind"] in ("claim_sum_f64", "chunk_partials")
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


# Import forms in a fresh interpreter: the submodule path first, the attributes of
# flashinfer, and a star import of the package.
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
        f"assert flashinfer.{PACKAGE.rpartition('.')[2]} is sys.modules['{PACKAGE}']\n"
        f"assert flashinfer.{PAIRS[0][0]} is sys.modules['{FACADE}'].{PAIRS[0][0]}\n"
        f"assert flashinfer.{PAIRS[0][1]} is sys.modules['{DET_FACADE}'].{PAIRS[0][1]}\n"
        f"assert not hasattr(sys.modules['{DET_FACADE}'], '{PAIRS[0][0]}')\n"
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
