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
"""

"""Borrow-contract test for the SM12x MoE kernels.

MoE kernel classes invoke DenseGemmKernel helpers unbound, e.g.
``self._dense_cls._partition_fragment_SFA(self, ...)`` with ``self`` being
the MoE kernel. Every ``self.<attr>`` read inside such a borrowed method
(including transitive intra-class calls) must therefore be provided by the
borrowing class.

Borrower classes are discovered by scanning the fused_moe sources for the
borrow pattern, so new kernels are covered without editing this test. The
test is CPU-only; it catches regressions like adding a ``self.``-resolved
helper to a borrowed method without needing an SM12x GPU.
"""

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

import flashinfer.fused_moe as fused_moe
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import (
    DenseGemmKernel,
)

FUSED_MOE_DIR = Path(inspect.getfile(fused_moe)).parent


def _tree(obj):
    return ast.parse(textwrap.dedent(inspect.getsource(obj)))


def _self_deps(cls, entry):
    """All ``self.<attr>`` reads reachable from method ``entry``."""
    methods = {n.name: n for n in ast.walk(_tree(cls)) if isinstance(n, ast.FunctionDef)}
    deps, seen, stack = set(), set(), [entry]
    while stack:
        name = stack.pop()
        if name in seen or name not in methods:
            continue
        seen.add(name)
        for node in ast.walk(methods[name]):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                deps.add(node.attr)
                stack.append(node.attr)
    deps.discard(entry)  # the borrowed method itself stays on DenseGemmKernel
    return deps


def _provided(class_def):
    """Attributes a class offers: its methods and ``self.<attr>`` assignments."""
    out = set()
    for node in ast.walk(class_def):
        if isinstance(node, ast.FunctionDef):
            out.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            out.update(
                t.attr
                for t in targets
                if isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            )
    return out


def _is_borrow_call(node):
    """Match ``self._dense_cls.<name>(self, ...)``; return (name, target)."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "_dense_cls"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "self"
    ):
        return None
    return node.func.attr


def _find_borrowers():
    """All (file, class, borrowed methods, _dense_cls target) in fused_moe."""
    out = []
    for path in sorted(FUSED_MOE_DIR.rglob("*.py")):
        for cls in (n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ClassDef)):
            borrowed = {n for node in ast.walk(cls) if (n := _is_borrow_call(node))}
            if not borrowed:
                continue
            target = None
            for node in ast.walk(cls):
                if (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(t, ast.Attribute) and t.attr == "_dense_cls"
                        for t in node.targets
                    )
                    and isinstance(node.value, ast.Name)
                ):
                    target = node.value.id
            out.append((path.relative_to(FUSED_MOE_DIR), cls, borrowed, target))
    return out


CASES = _find_borrowers()


@pytest.mark.parametrize(
    "path,cls,borrowed,target", CASES, ids=[f"{p.name}:{c.name}" for p, c, _, _ in CASES]
)
def test_borrowed_dense_method_self_deps(path, cls, borrowed, target):
    assert target == "DenseGemmKernel", (
        f"{cls.name} borrows from {target}; extend this test to resolve it."
    )
    provided = _provided(cls)
    for name in borrowed:
        missing = _self_deps(DenseGemmKernel, name) - provided
        assert not missing, (
            f"{cls.name} borrows DenseGemmKernel.{name}() but does not provide "
            f"{sorted(missing)}; make the helper a module-level function or add "
            f"it to {cls.name}."
        )
