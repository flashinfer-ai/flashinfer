# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys

from tests.trace.template_registry import (
    _TRACE_REGISTRATION_MODULES,
    collect_registered_trace_templates,
    trace_registry_entry_key,
)


def _uses_trace_decorator(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return False
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        name = getattr(
            decorator.func,
            "id",
            getattr(decorator.func, "attr", None),
        )
        if name == "flashinfer_api" and any(
            keyword.arg == "trace" for keyword in decorator.keywords
        ):
            return True
    return False


def _registration_modules_from_source() -> set[str]:
    package_root = Path(__file__).parents[2] / "flashinfer"
    modules: set[str] = set()
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not any(_uses_trace_decorator(node) for node in ast.walk(tree)):
            continue
        relative = path.relative_to(package_root.parent).with_suffix("")
        parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        modules.add(".".join(parts))
    return modules


def test_registration_module_inventory_is_complete():
    assert set(_TRACE_REGISTRATION_MODULES) == _registration_modules_from_source()


def test_registered_template_discovery_is_import_order_independent():
    from flashinfer.api_logging import _TRACE_REGISTRY

    expected = collect_registered_trace_templates()
    original_registry = list(_TRACE_REGISTRY)
    try:
        _TRACE_REGISTRY.reverse()
        actual = collect_registered_trace_templates()
    finally:
        _TRACE_REGISTRY[:] = original_registry

    assert [trace_registry_entry_key(entry) for entry in actual] == [
        trace_registry_entry_key(entry) for entry in expected
    ]


def test_formerly_order_dependent_templates_are_discovered():
    labels = {label for _, _, label in collect_registered_trace_templates()}
    assert "concat_mla_k" in labels

    optional_labels = {
        "flashinfer.comm.allreduce": "allreduce_fusion",
        "flashinfer.comm.dcp_alltoall": "decode_cp_a2a_alltoall",
        "flashinfer.cute_dsl.attention.wrappers.batch_mla": ("cute_dsl_batch_mla_run"),
        "flashinfer.cute_dsl.attention.wrappers.batch_prefill": (
            "cute_dsl_batch_prefill_run"
        ),
    }
    for module_name, label in optional_labels.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            continue
        assert label in labels


def _collect_parametrized_nodeids(
    *,
    workspace: Path,
    preimport_concat: bool,
) -> set[str]:
    imports = "import flashinfer.concat_ops\n" if preimport_concat else ""
    script = (
        imports
        + "import pytest\n"
        + "raise SystemExit(pytest.main([\n"
        + "    '--collect-only', '-q', '--color=no',\n"
        + "    'tests/trace/test_fi_trace_template_consistency.py',\n"
        + "    'tests/trace/test_template_init.py',\n"
        + "]))\n"
    )
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env["FLASHINFER_WORKSPACE_BASE"] = str(workspace)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return {
        line
        for line in result.stdout.splitlines()
        if line.startswith("tests/trace/") and "::" in line
    }


def test_parametrized_nodeids_do_not_depend_on_prior_imports(tmp_path):
    isolated = _collect_parametrized_nodeids(
        workspace=tmp_path / "isolated",
        preimport_concat=False,
    )
    preimported = _collect_parametrized_nodeids(
        workspace=tmp_path / "preimported",
        preimport_concat=True,
    )

    assert isolated
    assert preimported == isolated
