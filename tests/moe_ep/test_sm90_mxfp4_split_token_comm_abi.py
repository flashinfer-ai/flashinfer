# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Static CPU contracts for the split-K1 token-communication payload."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


_DISABLED_DISPATCH_FIELDS = (
    "carrier_row_table",
    "group_count",
    "group_rows",
    "token_rank_mask",
    "dispatch_done_counter",
)
_VENDOR_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel"
    / "src/moe_hopper_fp8/megamoe_kernel_fp8.py"
)
_TOKEN_COMM_SOURCE = _VENDOR_SOURCE.parents[1] / "src" / "token_comm.py"


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _assigned_tuple(node: ast.ClassDef, name: str) -> tuple[str, ...]:
    for statement in node.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
        ):
            value = ast.literal_eval(statement.value)
            assert isinstance(value, tuple)
            return value
    raise AssertionError(f"{node.name}.{name} is not a literal tuple")


@dataclass(frozen=True)
class _Operand:
    value: str


class _Ir:
    class Value:
        pass


def _load_argument_classes(tree: ast.Module):
    """Execute only the two data-only classes, with a tiny DSL serializer."""

    def extract_mlir_values(value):
        return [value.value] if isinstance(value, _Operand) else []

    def extract_mlir_attributes(value):
        return [f"attr:{value.value}"] if isinstance(value, _Operand) else []

    def new_from_mlir_values(prototype, values):
        if isinstance(prototype, _Operand):
            assert len(values) == 1
            return _Operand(values[0])
        assert not values
        return prototype

    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Tuple": Tuple,
        "ir": _Ir,
        "extract_mlir_values": extract_mlir_values,
        "extract_mlir_attributes": extract_mlir_attributes,
        "new_from_mlir_values": new_from_mlir_values,
    }
    selected = ast.Module(
        body=[
            _class_node(tree, "_SplitTokenCommArgs"),
            _class_node(tree, "SplitK1TokenCommArgs"),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(selected)
    exec(compile(selected, str(_VENDOR_SOURCE), "exec"), namespace)
    return namespace["SplitK1TokenCommArgs"]


def test_split_k1_disabled_dispatch_fields_are_static_none() -> None:
    tree = ast.parse(_VENDOR_SOURCE.read_text())
    split_node = _class_node(tree, "SplitK1TokenCommArgs")
    mlir_fields = _assigned_tuple(split_node, "_mlir_value_fields")
    const_fields = _assigned_tuple(split_node, "_const_fields")

    assert set(_DISABLED_DISPATCH_FIELDS).issubset(const_fields)
    assert set(_DISABLED_DISPATCH_FIELDS).isdisjoint(mlir_fields)

    constructors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SplitK1TokenCommArgs"
    ]
    assert len(constructors) == 1
    keywords = {kw.arg: kw.value for kw in constructors[0].keywords}
    for name in _DISABLED_DISPATCH_FIELDS:
        assert isinstance(keywords[name], ast.Constant)
        assert keywords[name].value is None

    argument_type = _load_argument_classes(tree)
    kwargs = {name: _Operand(name) for name in mlir_fields}
    kwargs.update({name: f"const:{name}" for name in const_fields})
    kwargs.update({name: None for name in _DISABLED_DISPATCH_FIELDS})
    payload = argument_type(**kwargs)

    serialized = payload.__extract_mlir_values__()
    assert serialized == list(mlir_fields)
    rebuilt = payload.__new_from_mlir_values__(
        [f"rebuilt:{name}" for name in mlir_fields]
    )
    assert rebuilt.__extract_mlir_values__() == [
        f"rebuilt:{name}" for name in mlir_fields
    ]
    for name in _DISABLED_DISPATCH_FIELDS:
        assert getattr(payload, name) is None
        assert getattr(rebuilt, name) is None


def test_split_k1_payload_covers_shared_dispatch_prefix() -> None:
    argument_tree = ast.parse(_VENDOR_SOURCE.read_text())
    split_node = _class_node(argument_tree, "SplitK1TokenCommArgs")
    payload_fields = set(
        _assigned_tuple(split_node, "_mlir_value_fields")
        + _assigned_tuple(split_node, "_const_fields")
    )

    comm_tree = ast.parse(_TOKEN_COMM_SOURCE.read_text())
    comm_class = _class_node(comm_tree, "TokenInPullTokenBackPush")
    dispatch_body = next(
        node
        for node in comm_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "dispatch_warp_body"
    )
    token_back_index = next(
        index
        for index, statement in enumerate(dispatch_body.body)
        if isinstance(statement, ast.If)
        and "self.enable_token_back" in ast.unparse(statement.test)
    )
    dispatch_prefix = ast.Module(
        body=dispatch_body.body[:token_back_index], type_ignores=[]
    )
    accessed_fields = {
        node.attr
        for node in ast.walk(dispatch_prefix)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "token_comm_args"
    }

    assert set(_DISABLED_DISPATCH_FIELDS).issubset(accessed_fields)
    assert accessed_fields.issubset(payload_fields), (
        "split K1 payload lacks fields accessed before the FC1 compile-time "
        f"token-back exclusion: {sorted(accessed_fields - payload_fields)}"
    )
