# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant MoE kernels' compile closures forward the DSL's keyword options, so the disk-cache path compiles the
exported object without the execution engine (``flashinfer.batch_invariant.jit._export_compile_fn``): every
``compile_fn`` of the MoE kernel modules takes ``**compile_options`` and hands them to every ``cute.compile`` call it
makes (a static read of the sources), and the two grouped-body closures are export-only compiles to the JIT module."""

import ast
import pathlib

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = (
    pathlib.Path(__file__).resolve().parents[3]
    / "flashinfer"
    / "fused_moe"
    / "batch_invariant"
)
#: The MoE kernel sources (the roots of the ``batch_invariant_moe`` and ``batch_invariant_moe_nvfp4_quant`` modules).
SOURCES = (
    "blocks/compile.py",
    "blocks/fp8bs_small.py",
    "blocks/quant_nvfp4.py",
    "route.py",
    "route_plan.py",
    "permute.py",
    "permute_nvfp4.py",
    "mx_permute.py",
    "combine.py",
)


def _closures(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "compile_fn":
            yield node


def _cute_compile_calls(fn: ast.FunctionDef):
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compile"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cute"
        ):
            yield node


@pytest.mark.parametrize("source", SOURCES)
def test_every_compile_closure_forwards_the_keyword_options(source):
    tree = ast.parse((ROOT / source).read_text())
    closures = list(_closures(tree))
    assert closures, source
    for fn in closures:
        assert fn.args.kwarg is not None and fn.args.kwarg.arg == "compile_options", (
            source,
            fn.lineno,
        )
        calls = list(_cute_compile_calls(fn))
        assert calls, (source, fn.lineno)
        for call in calls:
            forwarded = [
                kw
                for kw in call.keywords
                if kw.arg is None
                and isinstance(kw.value, ast.Name)
                and kw.value.id == "compile_options"
            ]
            assert forwarded, (source, call.lineno)
            assert any(kw.arg == "options" for kw in call.keywords), (
                source,
                call.lineno,
            )


def test_grouped_body_closures_are_export_only_compiles():
    """The two body closures (the grouped and the swapped record) are handed to the JIT module as export-only compiles:
    ``_export_compile_fn`` binds ``no_jit_engine=True`` to them."""
    import functools

    from flashinfer.batch_invariant import jit as J
    from flashinfer.fused_moe.batch_invariant.blocks import compile as C
    from flashinfer.gemm.batch_invariant.generator import Knobs

    grouped = C.compile_grouped_fn(
        Knobs((128, 128), (1, 2), False, True, "m", 1), C.as_chain("swiglu")
    )
    swapped = C.compile_swapped_fn(
        Knobs((128, 16), (1, 1), False, True, "m", 1, swap_ab=True), C.as_chain("plain")
    )
    for fn in (grouped, swapped):
        export = J._export_compile_fn(fn)
        assert isinstance(export, functools.partial) and export.func is fn
        assert export.keywords == {"no_jit_engine": True}
    assert J._export_compile_fn(lambda: None) is None
