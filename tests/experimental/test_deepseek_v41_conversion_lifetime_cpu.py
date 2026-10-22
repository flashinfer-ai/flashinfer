# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Conversion-probe ownership checks without importing Torch, CUDA or CuTe.

Run with --confcutdir=tests/experimental to exclude root CUDA collection probes.
"""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


_SOURCE = Path(__file__).with_name("test_deepseek_v41_decode.py")


def _conversion_fixture():
    tree = ast.parse(_SOURCE.read_text())
    fixture = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "conversion_kernels"
    )
    fixture.decorator_list = []
    namespace = {}
    exec(
        compile(ast.Module(body=[fixture], type_ignores=[]), str(_SOURCE), "exec"),
        namespace,
    )
    return contextmanager(namespace["conversion_kernels"])


@pytest.mark.parametrize("launched", [False, True])
def test_conversion_module_teardown(launched):
    unloaded = []
    module = SimpleNamespace(unload=lambda: unloaded.append("module"))
    compiled = SimpleNamespace(jit_module=None)
    with _conversion_fixture()() as kernels:
        kernels.append(compiled)
        # CuTe materializes jit_module on the first execution, not compilation.
        if launched:
            compiled.jit_module = module
        assert not unloaded
    assert unloaded == (["module"] if launched else [])


def test_conversion_module_teardown_after_failure():
    unloaded = []
    compiled = SimpleNamespace(
        jit_module=SimpleNamespace(unload=lambda: unloaded.append("module"))
    )
    with (
        pytest.raises(ValueError, match="probe failed"),
        _conversion_fixture()() as kernels,
    ):
        kernels.append(compiled)
        raise ValueError("probe failed")
    assert unloaded == ["module"]


def test_conversion_module_teardown_does_not_hide_errors():
    def unload():
        raise RuntimeError("unload failed")

    with (
        pytest.raises(RuntimeError, match="unload failed"),
        _conversion_fixture()() as kernels,
    ):
        kernels.append(SimpleNamespace(jit_module=SimpleNamespace(unload=unload)))


@pytest.mark.parametrize(
    "name",
    ["test_fp4_fp8_all_finite_codes"],
)
def test_conversion_probe_registers_owner_before_launch(name):
    tree = ast.parse(_SOURCE.read_text())
    probe = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    assert "conversion_kernels" in [arg.arg for arg in probe.args.args]
    compile_call = next(
        node
        for node in ast.walk(probe)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func) == "cute.compile"
    )
    register = next(
        node
        for node in ast.walk(probe)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "conversion_kernels.append"
    )
    launch = next(
        node
        for node in ast.walk(probe)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "compiled"
    )
    assert ast.unparse(register.args[0]) == "compiled"
    assert compile_call.lineno < register.lineno < launch.lineno
    # The common-unit probe compiles the real native producer. Each compiled
    # specialization must acquire its owner before its first execution.
    assert (
        sum(
            isinstance(node, ast.Call) and ast.unparse(node.func) == "cute.compile"
            for node in ast.walk(probe)
        )
        == 1
    )
