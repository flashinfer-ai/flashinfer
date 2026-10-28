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

Standalone-import tests of the Cake chunked LM-head + loss package.

A copy of ``flashinfer/experimental/cake_lm_head_loss`` must import and reach
its host-side checks on its own: under another top-level name, in a
``python -I`` subprocess with ``flashinfer`` blocked (the form of a vendoring
gate, ``scripts/check_cake_build_manifests.py --vendored-gate``).  The package
imports only itself, ``torch`` and ``tvm_ffi``; the experimental marking and
the environment switches belong to the public entry points
``flashinfer.chunked_lm_head.chunked_lm_head_loss`` /
``chunked_lm_head_logprob`` (covered by ``test_cake_lm_head_loss_api.py``), so
a direct call of the package emits no warning, and the only ``flashinfer``
import is the JIT fallback of ``cake_build.py``.  Two tests, CPU-only: no GPU,
no generated program and no JIT build is needed, so both run on every lane.
"""

from __future__ import annotations

import ast
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

import pytest
import torch

tvm_ffi = pytest.importorskip("tvm_ffi")  # the package's load path imports it

import flashinfer.experimental.cake_lm_head_loss as pkg

PACKAGE_DIR = Path(pkg.__file__).resolve().parent
VENDORED_NAME = "vendored_lm_head_loss"  # the copy's top-level name in the subprocess


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
cb = importlib.import_module(PKG_NAME + ".cake_backend")
cake_jit = importlib.import_module(PKG_NAME + ".cake_jit")
cake_build = importlib.import_module(PKG_NAME + ".cake_build")
for name in cake_build.__all__:
    getattr(cake_build, name)

# The backend's public surface: its ``__all__`` when it defines one, else every
# public callable the module itself defines.  Resolving each name imports
# nothing further; the entry points of the package are among them.
public = getattr(cb, "__all__", None)
if public is None:
    public = sorted(
        name
        for name, value in vars(cb).items()
        if not name.startswith("_")
        and callable(value)
        and getattr(value, "__module__", None) == cb.__name__
    )
for name in public:
    getattr(cb, name)
assert {
    "prepare_lm_head_loss",
    "forward_loss",
    "backward_loss",
    "forward_logprob",
    "backward_logprob",
    "chunked_lm_head_loss",
    "chunked_lm_head_logprob",
}.issubset(public), sorted(public)
assert isinstance(cake_jit.MODULES, dict) and cake_jit.STAGES  # registry loaded


def bf16(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


def f32(*shape):
    return torch.zeros(*shape, dtype=torch.float32)


T, H, V = 4, 256, 256  # H and V multiples of 256: the geometry checks pass
labels = torch.zeros(T, dtype=torch.int64)
# Host-side rejections on CPU tensors, each raised by the shape / dtype / option
# validation before any device query, program bind or launch:
calls = [
    # the loss forward: X must be BF16 (validate_lm_head_inputs)
    lambda: cb.forward_loss(f32(T, H), bf16(V, H), labels, loss_div=1.0),
    # the log-probability forward: labels must be int64
    lambda: cb.forward_logprob(bf16(T, H), bf16(V, H), labels.to(torch.int32)),
    # the prepared runner: the cake backend takes CUDA tensors only
    lambda: cb.prepare_lm_head_loss(bf16(T, H), bf16(V, H), labels, loss_div=1.0),
    # the autograd entry: grad_weight_dtype must equal W.dtype
    lambda: cb.chunked_lm_head_loss(
        bf16(T, H), bf16(V, H), labels, loss_div=1.0, grad_weight_dtype=torch.float32
    ),
    # the loss backward: a compacted accumulator needs num_rows
    lambda: cb.backward_loss(f32(2, H), None, row_index=torch.tensor([0, 2])),
    # the output cast: the accumulator must be FP32
    lambda: cb.scale_cast(bf16(8), None, torch.float32),
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
                raise AssertionError("the host-side check must reject the call")
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
    ``sys.modules["flashinfer"] = None``), resolves the backend's public names and
    the loader's ``__all__``, raises the host-side ``ValueError`` of six entry
    points on CPU operands without emitting a warning of any kind, and loads no
    ``flashinfer`` module."""
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
    in the body of a ``try`` with an ``except ImportError`` handler, and exactly
    one such guarded block exists -- the JIT fallback in ``cake_build.py``."""
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
