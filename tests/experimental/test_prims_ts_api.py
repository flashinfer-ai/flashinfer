# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-only coverage of PrimTS experimental API annotations and tracing."""

import ast
from collections import Counter
import importlib
import inspect
from pathlib import Path

import pytest


_PRIMS_TS_ROOT = Path(__file__).parents[2] / "flashinfer/attention/prims_ts"


def _logged_apis() -> list[tuple[str, str, bool, str]]:
    apis = []
    for path in sorted(_PRIMS_TS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                target = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                if not isinstance(target, ast.Name) or target.id not in (
                    "flashinfer_api",
                    "flashinfer_experimental_api",
                ):
                    continue
                parent = parents[node]
                name = (
                    f"{parent.name}.{node.name}"
                    if isinstance(parent, ast.ClassDef)
                    else node.name
                )
                module = "flashinfer.attention.prims_ts." + ".".join(
                    path.relative_to(_PRIMS_TS_ROOT).with_suffix("").parts
                )
                traced = isinstance(decorator, ast.Call) and any(
                    kw.arg == "trace" for kw in decorator.keywords
                )
                apis.append((module, name, traced, target.id))
    return apis


_LOGGED_APIS = _logged_apis()

_EXPECTED_PRIMTS_TRACE_VARIANTS = {
    (
        "flashinfer.attention.prims_ts.block_sparse",
        "BlockSparseTSWrapper.run",
    ): 4,
    (
        "flashinfer.attention.prims_ts.block_sparse",
        "BlockSparsePagedTSWrapper.run",
    ): 2,
    (
        "flashinfer.attention.prims_ts.block_sparse",
        "block_sparse_attention",
    ): 4,
    (
        "flashinfer.attention.prims_ts.block_sparse",
        "block_sparse_attention_with_paged_kv_cache",
    ): 2,
    (
        "flashinfer.attention.prims_ts.decode",
        "batch_decode_with_paged_kv_cache",
    ): 12,
    (
        "flashinfer.attention.prims_ts.decode",
        "prims_ts_batch_decode_with_kv_cache",
    ): 12,
    (
        "flashinfer.attention.prims_ts.decode",
        "BatchDecodePagedTSWrapper.run",
    ): 24,
    (
        "flashinfer.attention.prims_ts.mla_decode",
        "batch_mla_decode_with_paged_kv_cache",
    ): 4,
    (
        "flashinfer.attention.prims_ts.mla_decode",
        "prims_ts_batch_mla_decode_with_kv_cache",
    ): 4,
    (
        "flashinfer.attention.prims_ts.mla_decode",
        "BatchMLADecodePagedTSWrapper.run",
    ): 4,
}


def test_prims_ts_experimental_trace_registry_coverage() -> None:
    pytest.importorskip("cutlass", minversion="4.7.0")
    for module_name in {api[0] for api in _LOGGED_APIS}:
        importlib.import_module(module_name)
    from flashinfer.api_logging import _TRACE_REGISTRY
    from tests.trace.test_fi_trace_template_consistency import (
        assert_fi_trace_complete,
        assert_template_axes_covered,
        assert_template_signature_consistency,
    )

    entries = [
        entry
        for entry in _TRACE_REGISTRY
        if entry[0].__module__.startswith("flashinfer.attention.prims_ts.")
    ]
    assert Counter(
        (func.__module__, func.__qualname__) for func, _, _ in entries
    ) == Counter(_EXPECTED_PRIMTS_TRACE_VARIANTS)
    for func, template, label in entries:
        assert func.is_experimental is True
        assert_template_signature_consistency(func, template, label=label)
        assert_template_axes_covered(template, label=label, func=func)
        assert_fi_trace_complete(func, template, label=label)


def test_prims_ts_logged_apis_use_experimental_decorator() -> None:
    assert _LOGGED_APIS
    assert all(api[3] == "flashinfer_experimental_api" for api in _LOGGED_APIS)


@pytest.mark.parametrize(
    "module_name,qualified_name,traced,decorator_name",
    _LOGGED_APIS,
    ids=[f"{api[0]}.{api[1]}" for api in _LOGGED_APIS],
)
def test_prims_ts_experimental_markers_preserve_signatures_and_traces(
    module_name: str, qualified_name: str, traced: bool, decorator_name: str
) -> None:
    pytest.importorskip("cutlass", minversion="4.7.0")
    api = importlib.import_module(module_name)
    for name in qualified_name.split("."):
        api = getattr(api, name)
    assert api.is_experimental is True
    assert api.__doc__.startswith(".. warning::")
    assert inspect.signature(api) == inspect.signature(inspect.unwrap(api))
    assert getattr(inspect.unwrap(api), "is_experimental", False)
    if traced:
        from flashinfer.api_logging import _TRACE_REGISTRY

        assert callable(api.fi_trace)
        assert any(func is inspect.unwrap(api) for func, _, _ in _TRACE_REGISTRY)
