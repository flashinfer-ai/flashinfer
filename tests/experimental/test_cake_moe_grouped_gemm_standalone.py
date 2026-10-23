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

Standalone-import tests of the Cake MoE grouped GEMM package.

A copy of ``flashinfer/experimental/cake_moe_grouped_gemm`` must import and
reach its host-side checks on its own: under another top-level name, in a
``python -I`` subprocess with ``flashinfer`` blocked (the form of a vendoring
gate).  Inside FlashInfer the package must bind FlashInfer's own
experimental-API helpers, and the stand-ins it falls back to must match them
word for word.  CPU-only: no GPU, no generated program and no JIT build is
needed, so every test here runs on every lane.
"""

import ast
import inspect
import os
import shutil
import site
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

tvm_ffi = pytest.importorskip("tvm_ffi")  # cake_backend imports it at module level

import flashinfer.api_logging as api_logging
import flashinfer.experimental.cake_moe_grouped_gemm as pkg
from flashinfer.experimental.cake_moe_grouped_gemm import _experimental as ex
from flashinfer.experimental.cake_moe_grouped_gemm import cake_backend as cb

PACKAGE_DIR = Path(pkg.__file__).resolve().parent
VENDORED_NAME = "vendored_grouped_gemm"  # the copy's top-level name in the subprocess
API_NAMES = (
    "grouped_gemm_fwd",
    "grouped_gemm_dgrad",
    "grouped_gemm_wgrad",
    "cake_grouped_mm",
)


def _f32(*shape):
    return torch.zeros(*shape, dtype=torch.float32)


def _host_args(name):
    """float32 CPU operands for API ``name``.

    The bfloat16 check is the first host-side check of every entry point, so
    the call raises ``ValueError`` before any device access.
    """
    offs = torch.tensor([2, 4], dtype=torch.int32)
    rows, w_nk = _f32(4, 64), _f32(2, 256, 64)
    return {
        "grouped_gemm_fwd": (rows, w_nk, offs),
        "grouped_gemm_dgrad": (rows, _f32(2, 64, 256), offs),
        "grouped_gemm_wgrad": (_f32(4, 256), rows, offs),
        "cake_grouped_mm": (rows, w_nk, offs),
    }[name]


def _emit(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _recorded(fn, *args, **kwargs):
    """The one ExperimentalWarning (by category name) of ``fn(*args, **kwargs)``.

    The call goes through one helper frame, so a decorator warning
    (``stacklevel=2``) is attributed to ``_emit`` and a backend warning
    (``stacklevel=3``) to this function, whichever helper emitted it.
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _emit(fn, *args, **kwargs)
    hits = [w for w in rec if w.category.__name__ == "ExperimentalWarning"]
    assert len(hits) == 1, [str(w.message) for w in rec]
    return hits[0]


def _experimental_warnings_of(call):
    """ExperimentalWarnings (by category name) of a ``call()`` raising ValueError."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            call()
    return [w for w in rec if w.category.__name__ == "ExperimentalWarning"]


# ---------------------------------------------------------------------------
# A vendored copy, flashinfer blocked
# ---------------------------------------------------------------------------


def _dependency_paths():
    """Where ``torch`` and ``tvm_ffi`` live, plus the interpreter's site-packages.

    ``python -I`` drops ``PYTHONPATH`` and the user site, so the subprocess is
    told explicitly where the package's two dependencies are.
    """
    paths = list(site.getsitepackages())
    for module in (torch, tvm_ffi):
        paths.append(os.path.dirname(os.path.dirname(os.path.abspath(module.__file__))))
    return list(dict.fromkeys(paths))


_ISOLATED = """
import importlib
import importlib.abc
import os
import sys
import warnings

PKG_ROOT, PKG_NAME = sys.argv[1], sys.argv[2]
for p in sys.argv[3:]:
    if p not in sys.path:
        sys.path.append(p)


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "flashinfer" or name.startswith("flashinfer."):
            raise ImportError(f"flashinfer is blocked in this test: {name}")
        return None


sys.meta_path.insert(0, _Block())
sys.modules["flashinfer"] = None
sys.path.insert(0, PKG_ROOT)

import torch

assert not torch.cuda.is_available()
warnings.simplefilter("always")

pkg = importlib.import_module(PKG_NAME)
assert os.path.realpath(pkg.__file__).startswith(os.path.realpath(PKG_ROOT) + os.sep), pkg.__file__
ex = pkg._experimental
assert ex.STANDALONE is True
assert ex.__name__ == PKG_NAME + "._experimental"
assert issubclass(ex.ExperimentalWarning, UserWarning)
assert ex.ExperimentalWarning.__name__ == "ExperimentalWarning"
assert ex.ExperimentalWarning.__module__ == PKG_NAME + "._experimental"
for name in pkg.__all__:  # the lazy re-exports load cake_backend and cake_autograd
    getattr(pkg, name)
cb = importlib.import_module(PKG_NAME + ".cake_backend")


def f32(*shape):
    return torch.zeros(*shape, dtype=torch.float32)


offs = torch.tensor([2, 4], dtype=torch.int32)
calls = {
    "grouped_gemm_fwd": lambda: pkg.grouped_gemm_fwd(f32(4, 64), f32(2, 256, 64), offs),
    "grouped_gemm_dgrad": lambda: pkg.grouped_gemm_dgrad(f32(4, 64), f32(2, 64, 256), offs),
    "grouped_gemm_wgrad": lambda: pkg.grouped_gemm_wgrad(f32(4, 256), f32(4, 64), offs),
    "cake_grouped_mm": lambda: pkg.cake_grouped_mm(f32(4, 64), f32(2, 256, 64), offs),
}


def experimental_warnings(call):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("float32 operands must be rejected on the host")
    return [w for w in rec if w.category.__name__ == "ExperimentalWarning"]


for name, call in calls.items():
    fn = getattr(pkg, name)
    assert fn.is_experimental is True and fn.__wrapped__.is_experimental is True, name
    assert fn.experimental_feature == "cake_moe_grouped_gemm." + name, name
    assert fn.__doc__.startswith(".. warning::"), name
    [w] = experimental_warnings(call)
    assert w.category is ex.ExperimentalWarning, (name, w.category)
    assert "'cake_moe_grouped_gemm." + name + "' is an experimental FlashInfer API" in str(w.message), str(w.message)
    assert w.filename == "<string>", (name, w.filename)  # stacklevel=2: attributed to the caller
    assert experimental_warnings(call) == [], name  # second call: silent

m_indptr = torch.tensor([0, 2, 4], dtype=torch.int32)
for i in range(2):
    hits = experimental_warnings(
        lambda: cb.grouped_mm_bf16_cake(f32(4, 64), f32(2, 256, 64), m_indptr, out_dtype=torch.float32)
    )
    assert len(hits) == (1 if i == 0 else 0), (i, [str(w.message) for w in hits])
    if hits:
        assert hits[0].category is ex.ExperimentalWarning, hits[0].category
        assert "experimental backend 'cake' selected for 'grouped_mm_bf16'" in str(hits[0].message), str(hits[0].message)
        assert hits[0].filename == "<string>", hits[0].filename  # stacklevel=3: attributed to the stable API's caller

leaked = sorted(
    k for k in sys.modules if (k == "flashinfer" and sys.modules[k] is not None) or k.startswith("flashinfer.")
)
assert not leaked, leaked
print("OK")
"""


def test_vendored_copy_imports_and_warns_with_flashinfer_blocked(tmp_path):
    """A copy of the package under another top-level name imports in a ``python -I``
    subprocess with ``flashinfer`` blocked (meta-path finder plus
    ``sys.modules["flashinfer"] = None``), resolves every ``__all__`` name, binds
    the stand-ins, warns once per API and once per (API, backend) pair with the
    stand-in class, raises its host-side ``ValueError``s on CPU operands, and
    loads no ``flashinfer`` module."""
    root = tmp_path / "site"
    shutil.copytree(
        PACKAGE_DIR,
        root / VENDORED_NAME,
        ignore=shutil.ignore_patterns("__pycache__", "aot", "csrc"),
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, "-I", "-c", _ISOLATED, str(root), VENDORED_NAME]
        + _dependency_paths(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert proc.stdout.strip() == "OK", proc.stdout


# ---------------------------------------------------------------------------
# The package's import surface (source only)
# ---------------------------------------------------------------------------


def _exception_names(expr):
    """Exception class names of an ``except`` clause: a Name or a tuple of Names."""
    if isinstance(expr, ast.Tuple):
        return [e.id for e in expr.elts if isinstance(e, ast.Name)]
    return [expr.id] if isinstance(expr, ast.Name) else []


def _imports_flashinfer(node):
    if isinstance(node, ast.Import):
        return any(
            a.name == "flashinfer" or a.name.startswith("flashinfer.")
            for a in node.names
        )
    if isinstance(node, ast.ImportFrom) and not node.level and node.module:
        return node.module == "flashinfer" or node.module.startswith("flashinfer.")
    return False


def test_flashinfer_is_named_only_inside_two_importerror_guards():
    """Every ``flashinfer`` import of the package sits in the body of a ``try`` with
    an ``except ImportError`` handler, there are exactly two such guards (the
    helper binding in ``_experimental.py`` and the JIT fallback in
    ``cake_build.py``), and no import reaches outside the package
    (``from .. import``)."""
    guards = {}
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        guarded = {}  # import node -> line of the enclosing guarded try
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and any(
                "ImportError" in _exception_names(h.type) for h in node.handlers
            ):
                for stmt in node.body:
                    for sub in ast.walk(stmt):
                        guarded[sub] = node.lineno
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level < 2, (
                    f"{path.name}:{node.lineno}: import from outside the package"
                )
            if _imports_flashinfer(node):
                assert node in guarded, (
                    f"{path.name}:{node.lineno}: flashinfer import outside a "
                    "try/except ImportError"
                )
                guards.setdefault(path.name, set()).add(guarded[node])
    assert {name: len(lines) for name, lines in guards.items()} == {
        "_experimental.py": 1,
        "cake_build.py": 1,
    }, guards


# ---------------------------------------------------------------------------
# In-tree: FlashInfer's helpers are bound, and the stand-ins match them
# ---------------------------------------------------------------------------


def test_in_tree_binds_the_core_helpers(monkeypatch):
    """Inside FlashInfer the package's helpers are ``flashinfer.api_logging``'s own
    objects: the four APIs and the stable-API backend warn with FlashInfer's
    ``ExperimentalWarning`` (so ``pytest.warns`` on that class matches) and the
    backend pair is remembered in FlashInfer's set."""
    assert ex.STANDALONE is False
    assert ex.flashinfer_experimental_api is api_logging.flashinfer_experimental_api
    assert (
        ex.warn_experimental_backend_once is api_logging.warn_experimental_backend_once
    )
    # The stand-in class exists but nothing in-tree emits it.
    assert ex.ExperimentalWarning is not api_logging.ExperimentalWarning

    probe = ex.flashinfer_experimental_api(feature="cake_moe_grouped_gemm.probe")(
        lambda: None
    )
    with pytest.warns(api_logging.ExperimentalWarning, match="probe"):
        probe()

    for name in API_NAMES:
        fn = getattr(pkg, name)
        assert fn.is_experimental is True, name
        assert fn.__wrapped__.is_experimental is True, name
        assert fn.experimental_feature == f"cake_moe_grouped_gemm.{name}", name
        assert fn.__doc__.startswith(".. warning::"), name
        # A fresh wrapper of the real function: its once-only warning is
        # observable whatever ran earlier in this process.
        fresh = ex.flashinfer_experimental_api(feature=fn.experimental_feature)(
            fn.__wrapped__
        )
        [w] = _experimental_warnings_of(lambda: fresh(*_host_args(name)))
        assert w.category is api_logging.ExperimentalWarning, (name, w.category)
        assert fn.experimental_feature in str(w.message), str(w.message)
        assert _experimental_warnings_of(lambda: fresh(*_host_args(name))) == [], name

    monkeypatch.setattr(api_logging, "_WARNED_EXPERIMENTAL_BACKENDS", set())
    m_indptr = torch.tensor([0, 2, 4], dtype=torch.int32)
    backend_call = lambda: cb.grouped_mm_bf16_cake(
        _f32(4, 64), _f32(2, 256, 64), m_indptr, out_dtype=torch.float32
    )
    [w] = _experimental_warnings_of(backend_call)
    assert w.category is api_logging.ExperimentalWarning, w.category
    assert "experimental backend 'cake' selected for 'grouped_mm_bf16'" in str(
        w.message
    )
    assert ("grouped_mm_bf16", "cake") in api_logging._WARNED_EXPERIMENTAL_BACKENDS
    assert not ex._WARNED_BACKENDS
    assert _experimental_warnings_of(backend_call) == []


def test_stand_ins_match_the_core_helpers(monkeypatch):
    """The stand-ins reproduce FlashInfer's helpers: the same docstring banner,
    attributes, signature, warning text, category name and attribution
    (``stacklevel``), and the same once-only rule.  A wording change in
    ``flashinfer.api_logging`` fails here until the stand-ins follow."""

    def core_fn(x, *, y=1):
        """Throwaway."""
        return (x, y)

    def standin_fn(x, *, y=1):
        """Throwaway."""
        return (x, y)

    def raw(x):
        return x

    core = api_logging.flashinfer_experimental_api(feature="f")(core_fn)
    standin = ex._standalone_experimental_api(feature="f")(standin_fn)
    assert core.__doc__ == standin.__doc__
    assert core.experimental_feature == standin.experimental_feature == "f"
    assert core.is_experimental is True and standin.is_experimental is True
    assert core.__wrapped__.is_experimental is True
    assert standin.__wrapped__.is_experimental is True
    assert inspect.signature(core) == inspect.signature(standin)
    # Without feature= both fall back to the function's qualified name.
    assert (
        api_logging.flashinfer_experimental_api(raw).experimental_feature
        == ex._standalone_experimental_api(raw).experimental_feature
    )

    a, b = _recorded(core, 1, y=2), _recorded(standin, 1, y=2)
    assert str(a.message) == str(b.message)
    assert a.category.__name__ == b.category.__name__ == "ExperimentalWarning"
    assert a.category is api_logging.ExperimentalWarning
    assert b.category is ex.ExperimentalWarning
    assert (a.filename, a.lineno) == (b.filename, b.lineno)  # same stacklevel
    assert Path(a.filename).name == Path(__file__).name
    with warnings.catch_warnings():
        warnings.simplefilter("error", api_logging.ExperimentalWarning)
        warnings.simplefilter("error", ex.ExperimentalWarning)
        assert core(1) == standin(1) == (1, 1)  # second call: silent

    assert inspect.signature(
        ex._standalone_warn_backend_once, eval_str=True
    ) == inspect.signature(api_logging.warn_experimental_backend_once, eval_str=True)
    monkeypatch.setattr(api_logging, "_WARNED_EXPERIMENTAL_BACKENDS", set())
    monkeypatch.setattr(ex, "_WARNED_BACKENDS", set())
    for automatic in (False, True):
        backend = "auto_backend" if automatic else "backend"
        a = _recorded(
            api_logging.warn_experimental_backend_once,
            "api",
            backend,
            automatic=automatic,
        )
        b = _recorded(
            ex._standalone_warn_backend_once, "api", backend, automatic=automatic
        )
        assert str(a.message) == str(b.message)
        assert a.category.__name__ == b.category.__name__ == "ExperimentalWarning"
        assert a.category is api_logging.ExperimentalWarning
        assert b.category is ex.ExperimentalWarning
        assert (a.filename, a.lineno) == (b.filename, b.lineno)  # same stacklevel
        assert Path(a.filename).name == Path(__file__).name
    with warnings.catch_warnings():
        warnings.simplefilter("error", api_logging.ExperimentalWarning)
        warnings.simplefilter("error", ex.ExperimentalWarning)
        api_logging.warn_experimental_backend_once("api", "backend")
        ex._standalone_warn_backend_once("api", "backend")
    expected = {("api", "backend"), ("api", "auto_backend")}
    assert expected == api_logging._WARNED_EXPERIMENTAL_BACKENDS
    assert expected == ex._WARNED_BACKENDS
