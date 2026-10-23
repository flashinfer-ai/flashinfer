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
gate).  The package imports only itself, ``torch`` and ``tvm_ffi``; the
experimental-API marking and the once-only warnings belong to the core entry
points of ``flashinfer.grouped_mm`` (covered by
``test_cake_moe_grouped_gemm_api.py``), so a direct call of the package emits
no warning, and the only ``flashinfer`` import is the JIT fallback of
``cake_build.py``.  CPU-only: no GPU, no generated program and no JIT build is
needed, so every test here runs on every lane.
"""

import ast
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

import pytest
import torch

tvm_ffi = pytest.importorskip("tvm_ffi")  # cake_backend imports it at module level

import flashinfer.experimental.cake_moe_grouped_gemm as pkg

PACKAGE_DIR = Path(pkg.__file__).resolve().parent
VENDORED_NAME = "vendored_grouped_gemm"  # the copy's top-level name in the subprocess


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

pkg = importlib.import_module(PKG_NAME)
assert os.path.realpath(pkg.__file__).startswith(
    os.path.realpath(PKG_ROOT) + os.sep
), pkg.__file__
for name in pkg.__all__:  # the lazy re-exports load cake_backend and cake_autograd
    getattr(pkg, name)
cb = importlib.import_module(PKG_NAME + ".cake_backend")
for name in (
    "grouped_gemm_fwd",
    "grouped_gemm_dgrad",
    "grouped_gemm_wgrad",
    "cake_grouped_mm",
):
    # plain functions: the experimental marking belongs to the core entry points
    assert not hasattr(getattr(pkg, name), "is_experimental"), name


def f32(*shape):
    return torch.zeros(*shape, dtype=torch.float32)


offs = torch.tensor([2, 4], dtype=torch.int32)
m_indptr = torch.tensor([0, 2, 4], dtype=torch.int32)
# float32 operands fail the bfloat16 check of the four package functions; the
# two grouped_mm_bf16_cake calls fail their out_dtype / tactic checks first.
# Every call raises ValueError before any device access.
calls = [
    lambda: pkg.grouped_gemm_fwd(f32(4, 64), f32(2, 256, 64), offs),
    lambda: pkg.grouped_gemm_dgrad(f32(4, 64), f32(2, 64, 256), offs),
    lambda: pkg.grouped_gemm_wgrad(f32(4, 256), f32(4, 64), offs),
    lambda: pkg.cake_grouped_mm(f32(4, 64), f32(2, 256, 64), offs),
    lambda: cb.grouped_mm_bf16_cake(
        f32(4, 64), f32(2, 256, 64), m_indptr, out_dtype=torch.float32
    ),
    lambda: cb.grouped_mm_bf16_cake(f32(4, 64), f32(2, 256, 64), m_indptr, tactic=0),
]

with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    for _ in range(2):  # a second round: nothing here warns, not even once
        for call in calls:
            try:
                call()
            except ValueError:
                pass
            else:
                raise AssertionError("float32 operands must be rejected on the host")
assert not rec, [f"{w.category.__name__}: {w.message}" for w in rec]

leaked = sorted(
    k
    for k in sys.modules
    if (k == "flashinfer" and sys.modules[k] is not None)
    or k.startswith("flashinfer.")
)
assert not leaked, leaked
print("OK")
"""


def test_vendored_copy_imports_with_flashinfer_blocked(tmp_path):
    """A copy of the package under another top-level name imports in a ``python -I``
    subprocess with ``flashinfer`` blocked (meta-path finder plus
    ``sys.modules["flashinfer"] = None``), resolves every ``__all__`` name, raises
    the host-side ``ValueError`` of every function on CPU operands without emitting
    a warning of any kind, and loads no ``flashinfer`` module."""
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


def test_package_names_flashinfer_only_in_the_jit_fallback():
    """The package imports nothing from the rest of FlashInfer: no relative import
    climbs above the package (``from .. import``), every ``flashinfer`` import sits
    in the body of a ``try`` with an ``except ImportError`` handler, exactly one
    such guarded block exists -- the JIT fallback in ``cake_build.py`` -- and the
    package carries no ``_experimental.py`` of its own helpers."""
    assert not (PACKAGE_DIR / "_experimental.py").exists()
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
        "cake_build.py": 1
    }, guards
