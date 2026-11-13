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

A copy of ``flashinfer/experimental/cake/cake_dsa_sparse_attention_det_train`` must import and reach its host-side
checks on its own (flashinfer-ai/flashinfer#6553): under another top-level name, in a ``python -I`` subprocess with
``flashinfer`` blocked (the form of a vendoring gate).  The package imports only itself, ``torch`` and ``tvm_ffi``:
the API logging wrappers belong to the module ``flashinfer.cake_dsa_sparse_attention_det_train``, a direct call of
the package emits no warning, and the only ``flashinfer`` import is the JIT fallback of ``cake_build.py``.  The
backend carries no determinism alert and no warning machinery.  CPU-only: no GPU, no generated program and no JIT
build is needed.
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

import flashinfer.experimental.cake.cake_dsa_sparse_attention_det_train as pkg

pytestmark = pytest.mark.cpu_only

PACKAGE_DIR = Path(pkg.__file__).resolve().parent
VENDORED_NAME = (
    "vendored_dsa_sparse_attention_det"  # the copy's top-level name in the subprocess
)


def _dependency_paths():
    """Where ``torch`` and ``tvm_ffi`` live, plus the interpreter's site-packages (``python -I`` drops
    ``PYTHONPATH`` and the user site)."""
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
assert os.path.realpath(pkg.__file__).startswith(os.path.realpath(PKG_ROOT) + os.sep), pkg.__file__
cb = importlib.import_module(PKG_NAME + ".cake_backend")
jit = importlib.import_module(PKG_NAME + ".cake_jit")
launch = importlib.import_module(PKG_NAME + ".cake_launch")
for name in (
    "cake_dsa_sparse_attention_det",
    "cake_dsa_sparse_attention_det_varlen",
    "cake_dsa_sparse_attention_det_nope",
    "cake_dsa_sparse_attention_det_nope_varlen",
    "CakeDSASparseAttentionDetFunction",
    "CakeDSASparseAttentionDetNopeFunction",
    "CakeDSASparseAttentionDetNopeVarlenFunction",
):
    assert not hasattr(getattr(cb, name), "is_experimental"), name
for stage in jit.registered_stages(jit.PROGRAM) + jit.registered_stages(jit.NOPE_PROGRAM):
    assert stage in launch.LAUNCH and stage in launch.GRID, stage

T, S, K = 4, 8, 16
f32 = lambda *shape: torch.zeros(*shape, dtype=torch.float32)
bf16 = lambda *shape: torch.zeros(*shape, dtype=torch.bfloat16)
idx = torch.zeros(T, K, dtype=torch.int32)
calls = [
    # float32 queries fail the BF16 check before any device access
    lambda: cb.validate_dsa_train_inputs(f32(T, 64, 512), bf16(T, 64, 64), bf16(S, 512), bf16(S, 64), idx),
    lambda: cb.validate_dsa_nope_inputs(f32(T, 64, 512), bf16(S, 512), idx),
    lambda: cb._check_key_passes(0, S),
    lambda: cb._check_dkv_direct("yes"),
]

with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    for _ in range(2):
        for call in calls:
            try:
                call()
            except ValueError:
                pass
            else:
                raise AssertionError("the host-side check must raise ValueError")
    # pure host planning helpers run without a device
    assert cb.carve_fits(T, S, S)
    assert not cb.carve_fits(1, 4096, 4096)
    assert cb.accumulator_region_layout(S, S)["total"] > 0
assert not rec, [f"{w.category.__name__}: {w.message}" for w in rec]

leaked = sorted(
    k for k in sys.modules if (k == "flashinfer" and sys.modules[k] is not None) or k.startswith("flashinfer.")
)
assert not leaked, leaked
print("OK")
"""


def test_vendored_copy_imports_with_flashinfer_blocked(tmp_path):
    """A copy of the package under another top-level name imports in a ``python -I`` subprocess with ``flashinfer``
    blocked, resolves its entry points, launch tables and planning helpers, raises the host-side ``ValueError`` of
    its checks on CPU operands without emitting a warning, and loads no ``flashinfer`` module."""
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


def _exception_names(expr):
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
    """No relative import climbs above the package, every ``flashinfer`` import sits in a ``try`` with an ``except
    ImportError`` handler, and exactly one such guarded block exists: the JIT fallback in ``cake_build.py``."""
    guards = {}
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        guarded = {}
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
                    f"{path.name}:{node.lineno}: flashinfer import outside a try/except ImportError"
                )
                guards.setdefault(path.name, set()).add(guarded[node])
    assert {name: len(lines) for name, lines in guards.items()} == {
        "cake_build.py": 1
    }, guards


def test_backend_has_no_determinism_alert():
    """The backend registers no determinism alert: it imports neither ``warnings`` nor ``inspect``, calls no
    ``torch._C._alert_not_deterministic`` / ``torch.utils.deterministic`` hook and defines no ``_alert_*``
    helper."""
    tree = ast.parse((PACKAGE_DIR / "cake_backend.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not {a.name for a in node.names} & {"warnings", "inspect"}, (
                node.lineno
            )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert not node.name.startswith("_alert"), node.name
        if isinstance(node, ast.Attribute):
            assert "alert" not in node.attr.lower(), (node.lineno, node.attr)
