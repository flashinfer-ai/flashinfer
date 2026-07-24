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


def _runtime_min_version() -> int:
    tree = ast.parse(_CORE_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_CUDNN_MOE_MIN_VERSION"
            for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("could not find _CUDNN_MOE_MIN_VERSION in core.py")


def _gate_threshold(mark_name: str, source: str) -> int:
    """Extract the literal or named `< N` threshold used inside conftest.py's
    `pytest.mark.skipif(...)` assignment for `mark_name`."""
    tree = ast.parse(source)
    module_consts = {
        t.id: ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == mark_name for t in node.targets
        ):
            for cmp_node in ast.walk(node.value):
                if isinstance(cmp_node, ast.Compare) and isinstance(
                    cmp_node.ops[0], ast.Lt
                ):
                    right = cmp_node.comparators[0]
                    if isinstance(right, ast.Constant):
                        return right.value
                    if isinstance(right, ast.Name) and right.id in module_consts:
                        return module_consts[right.id]
                    if isinstance(right, ast.Name):
                        # Imported from another module (e.g. _CUDNN_MOE_MIN_VERSION):
                        # resolve it against the runtime source of truth directly.
                        return _runtime_min_version()
    raise AssertionError(f"could not find a `< N` threshold for {mark_name}")


@pytest.mark.parametrize(
    "mark_name", ["requires_cudnn_moe", "requires_cudnn_moe_block_scale"]
)
def test_cudnn_moe_gate_matches_runtime_min_version(mark_name):
    conftest_source = _CONFTEST_PATH.read_text()
    threshold = _gate_threshold(mark_name, conftest_source)
    min_version = _runtime_min_version()
    assert threshold == min_version, (
        f"{mark_name} gates on backend >= {threshold}, but the runtime check "
        f"(_check_cudnn_version) enforces >= {min_version}. Any version in "
        f"between would run the test and then crash instead of skipping "
        f"(#4064)."
    )
