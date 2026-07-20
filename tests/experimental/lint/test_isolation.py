# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Isolation lint: the structural guarantees of flashinfer.experimental.

Two invariants, enforced statically (stdlib-only, no torch needed):

1. Nothing under ``flashinfer/experimental/`` imports core flashinfer.
   Experimental is self-sufficient (zero outbound imports), so core
   refactors can never break it.
2. Nothing in core ``flashinfer/`` imports (or names in a string literal)
   ``flashinfer.experimental``.  This single check also proves that aot.py,
   the jit-cache wheel, and the cubin wheel can never pick up experimental
   code: their source sets are derived from core imports.

Tests may import both worlds; only ``flashinfer/`` itself is scanned.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "flashinfer"
EXP = PKG / "experimental"

_ESCAPES_PACKAGE = "<relative-import-escapes-repo>"


def _iter_py(root: Path):
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _import_targets(path: Path):
    """Yield absolute dotted targets of every import, resolving relatives."""
    tree = ast.parse(path.read_text(), filename=str(path))
    pkg_parts = path.relative_to(REPO).parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                yield node.module or ""
                continue
            if node.level - 1 > len(pkg_parts):
                yield _ESCAPES_PACKAGE
                continue
            base = ".".join(pkg_parts[: len(pkg_parts) - (node.level - 1)])
            if node.module:
                yield f"{base}.{node.module}" if base else node.module
            else:
                yield base


def _fmt(violations: list[tuple[str, str]]) -> str:
    return "\n".join(f"  {file}: {detail}" for file, detail in violations)


def test_experimental_never_imports_core():
    violations = []
    for path in _iter_py(EXP):
        rel = str(path.relative_to(REPO))
        for target in _import_targets(path):
            if target == _ESCAPES_PACKAGE:
                violations.append((rel, "relative import escapes the repo root"))
            elif target.startswith("flashinfer") and not target.startswith(
                "flashinfer.experimental"
            ):
                violations.append((rel, f"imports core module {target!r}"))
    assert not violations, (
        "flashinfer/experimental must not import core flashinfer "
        "(zero-outbound-imports rule):\n" + _fmt(violations)
    )


def test_core_never_imports_experimental():
    literal = re.compile(r"""["']flashinfer\.experimental""")
    violations = []
    for path in _iter_py(PKG):
        if path == EXP or EXP in path.parents:
            continue
        rel = str(path.relative_to(REPO))
        for target in _import_targets(path):
            if target.startswith("flashinfer.experimental"):
                violations.append((rel, f"imports {target!r}"))
        if literal.search(path.read_text()):
            violations.append((rel, "references 'flashinfer.experimental' in a string"))
    assert not violations, (
        "core flashinfer must never import or reference flashinfer.experimental:\n"
        + _fmt(violations)
    )
