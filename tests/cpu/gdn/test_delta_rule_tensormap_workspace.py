"""CPU-only contract: GDN tensor-map workspace is sized by %nsmid, not SM count (#9484)."""

import ast
from pathlib import Path

import pytest

DSL = Path(__file__).resolve().parents[3] / "flashinfer/gdn_kernels/delta_rule_dsl"
WRAPPERS = [
    "delta_rule_sm90.py",
    "delta_rule_sm120.py",
    "delta_rule_cp_sm90.py",
    "delta_rule_cp_sm120.py",
]


def _workspace_size_value(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and getattr(node.targets[0], "id", None) == "workspace_size"
        ):
            return ast.unparse(node.value)
    raise AssertionError(f"no workspace_size assignment in {path.name}")


@pytest.mark.parametrize("name", WRAPPERS)
def test_workspace_sized_by_nsmid(name):
    value = _workspace_size_value(DSL / name)
    assert "get_device_nsmid" in value
    assert "get_device_sm_count" not in value
    assert "* 128" in value
