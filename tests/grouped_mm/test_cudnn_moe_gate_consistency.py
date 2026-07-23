"""Regression test for #4064.

``requires_cudnn_moe`` (tests/grouped_mm/conftest.py) must never let a test run
at a cuDNN backend version the runtime check in
``flashinfer.grouped_mm.cudnn._check_cudnn_version`` would still reject, or the
test isn't skipped and instead crashes on the RuntimeError.

Prior to this fix, ``requires_cudnn_moe`` hardcoded its own threshold (91800)
independently of ``flashinfer.grouped_mm.cudnn._CUDNN_MOE_MIN_VERSION`` (the
value actually passed to ``_check_cudnn_version`` by grouped_mm_bf16/grouped_mm
_fp8). PR #3797 raised the runtime constant to 92100 without touching the test
threshold, so on cuDNN 9.18.0-9.20.x the test ran (91800 <= version) and then
hit the runtime RuntimeError (version < 92100) instead of being skipped.

This check parses both source files directly with `ast` and never imports
`flashinfer`, so it needs no torch, no cuDNN install, and no GPU: it catches
the drift on any CI runner regardless of hardware (the actual RuntimeError
only reproduces on real cuDNN 9.19/9.20, which CI does not have -- see the PR
description for the executed differential proof).

The threshold resolver below is deliberately shape-agnostic: conftest.py may
wire `requires_cudnn_moe` to a `pytest.mark.skipif(...)` directly, or (as of
PR #4185) build it through a small local helper function
(`_requires_cudnn_moe(feature)`) that aliases `_CUDNN_MOE_MIN_VERSION` into a
local variable first. Either way, this walks local variable assignments to
trace the right-hand side of the `< ...` comparison back to either a literal
(compared against the runtime constant) or the imported
`_CUDNN_MOE_MIN_VERSION` name itself (provably in sync by construction). If a
future refactor changes the shape enough that neither can be traced, this
raises loudly rather than silently passing.
"""

import ast
from pathlib import Path

import pytest

_CONFTEST_PATH = Path(__file__).parent / "conftest.py"
_CORE_PATH = (
    Path(__file__).parent.parent.parent
    / "flashinfer"
    / "grouped_mm"
    / "cudnn"
    / "core.py"
)

# Sentinel returned by `_resolve_threshold` when the version comparison traces
# back to the `_CUDNN_MOE_MIN_VERSION` import itself rather than a literal:
# such a gate can't drift from the runtime constant because it *is* the
# runtime constant, so no numeric comparison is needed.
_WIRED_TO_RUNTIME_CONST = object()


def _runtime_min_version() -> int:
    tree = ast.parse(_CORE_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_CUDNN_MOE_MIN_VERSION"
            for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("could not find _CUDNN_MOE_MIN_VERSION in core.py")


def _simple_assignments(stmts, base_scope=None):
    """Build {name: literal-or-_WIRED_TO_RUNTIME_CONST} for `name = <const>` /
    `name = <other known name>` assignments in `stmts`, seeded from
    `base_scope` (e.g. module-level constants visible inside a function)."""
    scope = dict(base_scope or {})
    for stmt in stmts:
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            continue
        name, value = stmt.targets[0].id, stmt.value
        if isinstance(value, ast.Constant):
            scope[name] = value.value
        elif isinstance(value, ast.Name):
            if value.id == "_CUDNN_MOE_MIN_VERSION":
                scope[name] = _WIRED_TO_RUNTIME_CONST
            elif value.id in scope:
                scope[name] = scope[value.id]
    return scope


def _resolve_operand(node, scope):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "_CUDNN_MOE_MIN_VERSION":
            return _WIRED_TO_RUNTIME_CONST
        if node.id in scope:
            return scope[node.id]
    return None


def _find_lt_threshold(search_root, scope):
    """Walk `search_root` for a `... < X` comparison and resolve X via `scope`."""
    for cmp_node in ast.walk(search_root):
        if isinstance(cmp_node, ast.Compare) and any(
            isinstance(op, ast.Lt) for op in cmp_node.ops
        ):
            resolved = _resolve_operand(cmp_node.comparators[0], scope)
            if resolved is not None:
                return resolved
    return None


def _gate_threshold(mark_name: str, source: str):
    """Resolve the `< N` threshold `mark_name` (`requires_cudnn_moe` /
    `requires_cudnn_moe_block_scale`) gates on in conftest.py, following
    either a direct `pytest.mark.skipif(...)` assignment or an assignment
    built by calling a module-level helper function. Returns an int literal,
    or `_WIRED_TO_RUNTIME_CONST` if it traces straight back to the
    `_CUDNN_MOE_MIN_VERSION` import."""
    tree = ast.parse(source)
    module_scope = _simple_assignments(tree.body)
    funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == mark_name for t in node.targets)
        ):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in funcs
        ):
            fn = funcs[value.func.id]
            fn_scope = _simple_assignments(fn.body, base_scope=module_scope)
            threshold = _find_lt_threshold(fn, fn_scope)
        else:
            threshold = _find_lt_threshold(value, module_scope)
        if threshold is not None:
            return threshold
    raise AssertionError(f"could not find a `< N` threshold for {mark_name}")


@pytest.mark.parametrize(
    "mark_name", ["requires_cudnn_moe", "requires_cudnn_moe_block_scale"]
)
def test_cudnn_moe_gate_matches_runtime_min_version(mark_name):
    conftest_source = _CONFTEST_PATH.read_text()
    threshold = _gate_threshold(mark_name, conftest_source)
    min_version = _runtime_min_version()
    if threshold is _WIRED_TO_RUNTIME_CONST:
        # Gate is aliased straight to _CUDNN_MOE_MIN_VERSION: can't drift.
        return
    assert threshold == min_version, (
        f"{mark_name} gates on backend >= {threshold}, but the runtime check "
        f"(_check_cudnn_version) enforces >= {min_version}. Any version in "
        f"between would run the test and then crash instead of skipping "
        f"(#4064)."
    )
