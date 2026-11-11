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

Public names of the Cake deterministic causal conv1d + SiLU training packages.

Every module, package and entry point of the deterministic packages carries ``det`` in its name, and every former
name is gone: the modules before the det rename (``flashinfer.cake_causal_conv1d_train``,
``flashinfer.cake_causal_conv1d_train_backend`` and its submodules) and the backend package's directory before the
move under ``flashinfer/experimental/cake/`` (``flashinfer.cake_causal_conv1d_det_train_backend`` and its submodules)
do not import, and no namespace (the ``flashinfer`` top level, ``flashinfer.cake_causal_conv1d_det_train``, its
backend package ``flashinfer.experimental.cake.cake_causal_conv1d_det_train``, the KCP facade
``flashinfer.cake_causal_conv1d_det_train_kcp`` and the KCP backend module) binds a former name.  Every public name
carries the mode word ``det`` in one of the allowed forms (UPPER_CASE constants exempt), no ``fwd`` / ``bwd`` token
in an entry point's name, and no entry point takes a ``deterministic`` keyword.  Importing ``flashinfer`` or a facade
loads no backend module until first use, and importing a single-rank facade or package loads no KCP host module
(checked in fresh interpreters).
"""

from __future__ import annotations

import importlib
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import flashinfer
import flashinfer.cake_causal_conv1d_det_train as facade
import flashinfer.cake_causal_conv1d_det_train_kcp as kcp_facade
import flashinfer.experimental.cake.cake_causal_conv1d_det_train as backend
from flashinfer.experimental.cake.cake_causal_conv1d_det_train import kcp as kcp_backend

pytestmark = pytest.mark.cpu_only

FACADE = "flashinfer.cake_causal_conv1d_det_train"
PACKAGE = "flashinfer.experimental.cake.cake_causal_conv1d_det_train"
KCP_FACADE = "flashinfer.cake_causal_conv1d_det_train_kcp"
SUBMODULES = ("cake_build", "cake_jit", "kcp", "kcp_jit")
REMOVED_MODULES = (
    # before the det rename
    "flashinfer.cake_causal_conv1d_train",
    "flashinfer.cake_causal_conv1d_train_backend",
    *(f"flashinfer.cake_causal_conv1d_train_backend.{m}" for m in SUBMODULES),
    # the det backend package before its move under flashinfer/experimental/cake/
    "flashinfer.cake_causal_conv1d_det_train_backend",
    *(f"flashinfer.cake_causal_conv1d_det_train_backend.{m}" for m in SUBMODULES),
)

# (removed name, det name) of the single-rank entry points: the facade and the backend package.
SINGLE_RANK = (
    ("CakeCausalConv1dFunction", "CakeCausalConv1dDetFunction"),
    ("cake_causal_conv1d", "cake_causal_conv1d_det"),
    ("cake_causal_conv1d_train_forward", "cake_causal_conv1d_det_train_forward"),
    ("cake_causal_conv1d_train_backward", "cake_causal_conv1d_det_train_backward"),
    (
        "cake_causal_conv1d_train_forward_workspace",
        "cake_causal_conv1d_det_train_forward_workspace",
    ),
    (
        "cake_causal_conv1d_train_forward_workspace_bytes",
        "cake_causal_conv1d_det_train_forward_workspace_bytes",
    ),
    (
        "cake_causal_conv1d_train_backward_workspace",
        "cake_causal_conv1d_det_train_backward_workspace",
    ),
    (
        "cake_causal_conv1d_train_backward_workspace_bytes",
        "cake_causal_conv1d_det_train_backward_workspace_bytes",
    ),
    (
        "is_cake_causal_conv1d_train_supported",
        "is_cake_causal_conv1d_det_train_supported",
    ),
)
# The single-rank names the ``flashinfer`` top level exports.
TOP_LEVEL = (
    ("CakeCausalConv1dFunction", "CakeCausalConv1dDetFunction"),
    ("cake_causal_conv1d", "cake_causal_conv1d_det"),
    ("cake_causal_conv1d_train_forward", "cake_causal_conv1d_det_train_forward"),
    ("cake_causal_conv1d_train_backward", "cake_causal_conv1d_det_train_backward"),
    (
        "cake_causal_conv1d_train_forward_workspace_bytes",
        "cake_causal_conv1d_det_train_forward_workspace_bytes",
    ),
    (
        "cake_causal_conv1d_train_backward_workspace_bytes",
        "cake_causal_conv1d_det_train_backward_workspace_bytes",
    ),
)
# (removed names, current name) of the KCP entry points and types: the KCP facade and the KCP backend module.  The
# removed names are the ones before the det rename, then the det names before the KCP names took the module prefix
# and the ``Kcp`` qualifier its place before the noun (FlashInfer #8928).
KCP_FACADE_NAMES = (
    (("cake_causal_conv1d_kcp",), "cake_causal_conv1d_det_kcp"),
    (
        ("cake_causal_conv1d_kcp_multi", "cake_causal_conv1d_det_multi_kcp"),
        "cake_causal_conv1d_det_kcp_multi",
    ),
    (
        ("KcpRecomputeHandle", "RecomputeHandleDetKcp"),
        "CakeCausalConv1dDetKcpRecomputeHandle",
    ),
    (
        ("CakeCausalConv1dKcpFunction", "CakeCausalConv1dDetFunctionKcp"),
        "CakeCausalConv1dDetKcpFunction",
    ),
    (
        ("CakeCausalConv1dKcpMultiFunction", "CakeCausalConv1dDetMultiFunctionKcp"),
        "CakeCausalConv1dDetKcpMultiFunction",
    ),
)
KCP_BACKEND_NAMES = (
    *KCP_FACADE_NAMES,
    (
        ("cake_causal_conv1d_kcp_forward", "cake_causal_conv1d_det_forward_kcp"),
        "cake_causal_conv1d_det_train_kcp_forward",
    ),
    (
        ("cake_causal_conv1d_kcp_backward", "cake_causal_conv1d_det_backward_kcp"),
        "cake_causal_conv1d_det_train_kcp_backward",
    ),
    (
        (
            "cake_causal_conv1d_kcp_multi_forward",
            "cake_causal_conv1d_det_multi_forward_kcp",
        ),
        "cake_causal_conv1d_det_train_kcp_multi_forward",
    ),
    (
        (
            "cake_causal_conv1d_kcp_multi_backward",
            "cake_causal_conv1d_det_multi_backward_kcp",
        ),
        "cake_causal_conv1d_det_train_kcp_multi_backward",
    ),
    (("graph_cache_clear",), "cake_causal_conv1d_det_train_kcp_graph_cache_clear"),
    (("host_cache_clear",), "cake_causal_conv1d_det_train_kcp_host_cache_clear"),
)
NAMESPACES = {"facade": facade, "backend": backend}
# Every public namespace of the family whose ``__all__`` the mode-word rule covers.
ALL_NAMESPACES = {
    "facade": facade,
    "kcp_facade": kcp_facade,
    "backend": backend,
    "kcp_backend": kcp_backend,
}
# The allowed forms of a public name (UPPER_CASE module constants are exempt): the autograd entries, the module
# prefix + verb (no ``fwd`` / ``bwd`` token), the support query, the classes.
NAME_FORMS = (
    re.compile(r"cake_causal_conv1d_det(_kcp(_multi)?)?"),
    re.compile(r"cake_causal_conv1d_det_train(_kcp)?_[a-z_]+"),
    re.compile(r"is_cake_causal_conv1d_det_train_supported"),
    re.compile(r"CakeCausalConv1dDet(Kcp)?(Multi)?[A-Z]\w*"),
)


@pytest.mark.parametrize("name", REMOVED_MODULES)
def test_removed_module_does_not_import(name: str) -> None:
    sys.modules.pop(name, None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)
    leaf = name.removeprefix("flashinfer.")
    if "." not in leaf:
        assert not hasattr(flashinfer, leaf), leaf


@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
@pytest.mark.parametrize("pair", SINGLE_RANK, ids=lambda p: p[0])
def test_single_rank_names(namespace: str, pair: tuple[str, str]) -> None:
    module = NAMESPACES[namespace]
    old, new = pair
    assert not hasattr(module, old), (namespace, old)
    assert old not in module.__all__, (namespace, old)
    assert new in module.__all__ and callable(getattr(module, new)), (namespace, new)


@pytest.mark.parametrize("pair", TOP_LEVEL, ids=lambda p: p[0])
def test_top_level_names(pair: tuple[str, str]) -> None:
    old, new = pair
    assert not hasattr(flashinfer, old), old
    assert getattr(flashinfer, new) is getattr(facade, new), new


@pytest.mark.parametrize("pair", KCP_FACADE_NAMES, ids=lambda p: p[1])
def test_kcp_names_not_on_the_single_rank_facade(
    pair: tuple[tuple[str, ...], str],
) -> None:
    olds, new = pair
    for module in (facade, backend, flashinfer):
        for name in (*olds, new):
            assert not hasattr(module, name), (module.__name__, name)
    assert new in kcp_facade.__all__, new


@pytest.mark.parametrize("pair", KCP_BACKEND_NAMES, ids=lambda p: p[1])
def test_kcp_backend_names(pair: tuple[tuple[str, ...], str]) -> None:
    olds, new = pair
    for name in olds:
        for module in (kcp_facade, kcp_backend):
            assert not hasattr(module, name), (module.__name__, name)
            assert name not in module.__all__, (module.__name__, name)
    assert new in kcp_backend.__all__ and hasattr(kcp_backend, new), new
    # the KCP launch grid is private
    assert not hasattr(kcp_backend, "launch_grid") and callable(
        kcp_backend._launch_grid
    )


def _tokens(name: str) -> list[str]:
    """The lower-case words of a name: split on ``_`` and at CamelCase boundaries (``CakeCausalConv1dDetKcpFunction``
    -> cake, causal, conv1d, det, kcp, function); ``nondet`` stays one word."""
    return [
        word.lower()
        for part in name.split("_")
        for word in re.findall(r"[A-Z]+(?![a-z0-9])|[A-Z]?[a-z0-9]+", part)
    ]


def test_tokens() -> None:
    assert _tokens("CakeCausalConv1dDetKcpFunction") == [
        "cake",
        "causal",
        "conv1d",
        "det",
        "kcp",
        "function",
    ]
    assert "det" not in _tokens("cake_causal_conv1d_nondet_kcp")
    assert "det" in _tokens("cake_causal_conv1d_det_train_kcp_forward")


@pytest.mark.parametrize("namespace", sorted(ALL_NAMESPACES))
def test_public_names_carry_the_mode_word(namespace: str) -> None:
    module = ALL_NAMESPACES[namespace]
    for name in module.__all__:
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            continue  # ALL_CAPS module constants are exempt
        assert "det" in _tokens(name), (namespace, name)
        assert any(form.fullmatch(name) for form in NAME_FORMS), (namespace, name)
        assert not {"fwd", "bwd"} & set(name.split("_")), (namespace, name)
        assert hasattr(module, name), (namespace, name)


def _signatures(obj: object) -> list[inspect.Signature]:
    """The signatures a caller binds: a function's; an autograd Function's ``forward``; another class's own
    constructor."""
    if inspect.isclass(obj):
        if issubclass(obj, torch.autograd.Function):
            return [inspect.signature(obj.forward)]
        return [inspect.signature(obj.__init__)] if "__init__" in vars(obj) else []
    return [inspect.signature(obj)]


@pytest.mark.parametrize("namespace", sorted(ALL_NAMESPACES))
def test_no_entry_point_takes_deterministic(namespace: str) -> None:
    module = ALL_NAMESPACES[namespace]
    checked = 0
    for name in module.__all__:
        obj = getattr(module, name)
        if not callable(obj):
            continue
        for sig in _signatures(obj):
            assert "deterministic" not in sig.parameters, (namespace, name)
            with pytest.raises(TypeError):
                sig.bind_partial(deterministic=False)
        checked += 1
    assert checked >= 3, namespace


def test_det_entry_points_are_the_decorated_defs() -> None:
    # One ``flashinfer_api`` wrapper per entry point, named as the det def.
    for _, new in SINGLE_RANK[1:4]:
        assert getattr(facade, new).__name__ == new, new
    assert backend.CakeCausalConv1dDetFunction.__name__ == "CakeCausalConv1dDetFunction"


# The KCP names, checked in a fresh interpreter after the single-rank imports (which must not load the KCP host
# module), then after importing the KCP facade and backend module.
_KCP_CHECK = """
import sys
from pathlib import Path

{imports}
import {package} as package

kcp_files = {{(Path(package.__file__).parent / f).resolve() for f in ("kcp.py", "kcp_jit.py")}}
loaded = sorted(
    name
    for name, module in list(sys.modules.items())
    if getattr(module, "__file__", None) and Path(module.__file__).resolve() in kcp_files
)
assert not loaded, loaded
print("KCP_NOT_LOADED")
import {kcp_facade} as kcp_facade
from {package} import kcp as kcp_backend
for olds, new in {facade_names!r}:
    for old in olds:
        assert not hasattr(kcp_facade, old), old
    assert new in kcp_facade.__all__ and hasattr(kcp_backend, new), new
for name in ("CakeCausalConv1dDetKcpRecomputeHandle", "CakeCausalConv1dDetKcpFunction", "CakeCausalConv1dDetKcpMultiFunction"):
    assert getattr(kcp_facade, name) is getattr(kcp_backend, name), name
for olds, new in {backend_names!r}:
    for old in olds:
        assert not hasattr(kcp_backend, old), old
        assert old not in kcp_backend.__all__, old
    assert new in kcp_backend.__all__, new
print("KCP_DET_NAMES")
"""

IMPORTS = {
    "flashinfer": "import flashinfer",
    "facade": f"import {FACADE}",
    "package": f"import {PACKAGE}",
    "package_names": f"from {PACKAGE} import cake_causal_conv1d_det_train_forward, cake_jit",
}


@pytest.mark.parametrize("form", sorted(IMPORTS))
def test_imports_do_not_load_kcp_and_kcp_names_carry_det(form: str) -> None:
    code = _KCP_CHECK.format(
        imports=IMPORTS[form],
        package=PACKAGE,
        kcp_facade=KCP_FACADE,
        facade_names=KCP_FACADE_NAMES,
        backend_names=KCP_BACKEND_NAMES,
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, (form, r.stdout, r.stderr)
    assert "KCP_NOT_LOADED" in r.stdout and "KCP_DET_NAMES" in r.stdout, (
        form,
        r.stdout,
        r.stderr,
    )


def test_old_package_directory_is_gone() -> None:
    root = Path(flashinfer.__file__).parent
    assert not (root / "cake_causal_conv1d_train_backend").exists()
    assert not (root / "cake_causal_conv1d_train.py").exists()
    assert not (root / "cake_causal_conv1d_det_train_backend").exists()
    assert Path(backend.__file__).parent == (
        root / "experimental" / "cake" / "cake_causal_conv1d_det_train"
    )


# ``import flashinfer`` and the facades load no module of the backend package (it sits under
# flashinfer/experimental/); listing a facade's names (``dir``) does not load it either.  The first access to a
# re-exported name loads it and returns the backend's object.  Checked in fresh interpreters.
_ISOLATION_CHECK = """
import sys

def loaded():
    return sorted(m for m in sys.modules if m == {package!r} or m.startswith({package!r} + "."))

{imports}
assert not loaded(), loaded()
print("BACKEND_NOT_LOADED")
{first_use}
assert loaded(), "the first use did not load the backend"
print("BACKEND_LOADED_ON_FIRST_USE")
"""

ISOLATION = {
    "flashinfer": (
        "import flashinfer\nnames = dir(flashinfer)\nassert 'CakeCausalConv1dDetFunction' in names",
        "import importlib\n"
        "b = importlib.import_module({package!r})\n"
        "assert flashinfer.CakeCausalConv1dDetFunction is b.CakeCausalConv1dDetFunction\n"
        "assert flashinfer.cake_causal_conv1d_det_train_forward_workspace_bytes"
        " is b.cake_causal_conv1d_det_train_forward_workspace_bytes",
    ),
    "facade": (
        f"import {FACADE} as f\nassert 'ARCHES' in dir(f)",
        "import importlib\nassert f.ARCHES is importlib.import_module({package!r}).ARCHES",
    ),
    "kcp_facade": (
        f"import {KCP_FACADE} as f\nassert 'CakeCausalConv1dDetKcpFunction' in dir(f)",
        "import importlib\n"
        "assert f.CakeCausalConv1dDetKcpFunction"
        " is importlib.import_module({package!r} + '.kcp').CakeCausalConv1dDetKcpFunction",
    ),
}


@pytest.mark.parametrize("form", sorted(ISOLATION))
def test_import_flashinfer_and_facades_load_no_backend(form: str) -> None:
    imports, first_use = ISOLATION[form]
    code = _ISOLATION_CHECK.format(
        package=PACKAGE, imports=imports, first_use=first_use.format(package=PACKAGE)
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, (form, r.stdout, r.stderr)
    assert "BACKEND_NOT_LOADED" in r.stdout, (form, r.stdout, r.stderr)
    assert "BACKEND_LOADED_ON_FIRST_USE" in r.stdout, (form, r.stdout, r.stderr)
