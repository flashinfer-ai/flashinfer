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

"""Structural checks for trace solution modules."""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

from flashinfer.api_logging import _TRACE_REGISTRY
from flashinfer.trace.solutions import load_solutions


_TRACE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TRACE_DIR.parents[1]
_SOLUTIONS_DIR = _REPO_ROOT / "flashinfer" / "trace" / "solutions"

_EXPECTED_EXPLICIT_BACKEND_SOLUTIONS = {
    "gemm_bf16": {
        "cublaslt": "cublaslt",
        "cudnn": "cudnn",
        "cutlass": "cutlass",
        "tgv": "tgv",
        "tinygemm": "tinygemm",
    },
    "gemm_fp4": {
        "b12x": "b12x",
        "cute_dsl": "cute-dsl",
        "cudnn": "cudnn",
        "cutlass": "cutlass",
        "trtllm": "trtllm",
    },
    "gemm_mxfp8": {
        "cute_dsl": "cute-dsl",
        "cutlass": "cutlass",
        "trtllm": "trtllm",
    },
    "gqa_paged_decode": {
        "fa2": "fa2",
        "fa3": "fa3",
        "trtllm_gen": "trtllm-gen",
    },
    "gqa_paged_prefill": {
        "fa2": "fa2",
        "fa3": "fa3",
        "cudnn": "cudnn",
        "trtllm_gen": "trtllm-gen",
    },
    "gqa_ragged": {
        "fa2": "fa2",
        "fa3": "fa3",
        "cutlass": "cutlass",
    },
    "mla_paged_decode": {
        "fa2": "fa2",
        "fa3": "fa3",
    },
    "single_prefill": {
        "fa2": "fa2",
        "fa3": "fa3",
    },
    "segment_gemm_run": {
        "sm80": "sm80",
        "sm90": "sm90",
    },
}


def _definition_jsons():
    return sorted(path for path in _TRACE_DIR.glob("fi_trace_out*/*.json"))


def _json_by_definition():
    by_definition = {}
    for path in _definition_jsons():
        data = json.loads(path.read_text())
        by_definition.setdefault(data["definition"], []).append((path, data))
    return by_definition


def _api_name(func):
    return f"{getattr(func, '__module__', '')}.{getattr(func, '__qualname__', '')}"


def _resolve_api(api):
    parts = api.split(".")
    last_error = None
    for end in range(len(parts), 0, -1):
        module_name = ".".join(parts[:end])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            last_error = exc
            continue
        for attr in parts[end:]:
            obj = getattr(obj, attr)
        return obj
    if last_error is not None:
        raise last_error
    raise ValueError(f"Cannot resolve API {api!r}")


def _registered_template(definition, api):
    _resolve_api(api)
    for func, template, _label in _TRACE_REGISTRY:
        if (
            getattr(template, "definition", None) == definition
            and _api_name(func) == api
        ):
            return func, template
    raise ValueError(f"No trace template registered for {definition!r}, {api!r}")


def _expected_api_kwargs(definition, api):
    func, template = _registered_template(definition, api)
    signature = inspect.signature(func)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    mapping = {}
    tuple_parts = {}
    for json_key, descriptor in template.inputs.items():
        param = getattr(descriptor, "param", None) or json_key
        if param not in signature.parameters and not accepts_kwargs:
            continue
        tuple_idx = getattr(descriptor, "tuple_idx", None)
        if tuple_idx is None:
            mapping[param] = json_key
        else:
            tuple_parts.setdefault(param, []).append((tuple_idx, json_key))

    for param, parts in tuple_parts.items():
        mapping[param] = tuple(name for _idx, name in sorted(parts))
    return mapping


def test_trace_jsons_have_definition_field():
    for path in _definition_jsons():
        data = json.loads(path.read_text())
        definition = data.get("definition")
        assert isinstance(definition, str) and definition, f"{path} missing definition"
        assert data["name"] == definition or data["name"].startswith(
            f"{definition}_"
        ), f"{path} definition={definition!r} does not prefix name={data['name']!r}"


def test_solution_modules_cover_committed_definitions():
    for definition in _json_by_definition():
        package_dir = _SOLUTIONS_DIR / definition
        assert package_dir.is_dir(), f"Missing solution directory for {definition!r}"
        modules = [
            path
            for path in package_dir.glob("*.py")
            if path.stem != "__init__" and not path.name.startswith("_")
        ]
        assert modules, f"Missing solution module for {definition!r}"


def test_solution_modules_import_and_match_parent_definition():
    for package_dir in sorted(_SOLUTIONS_DIR.iterdir()):
        if not package_dir.is_dir() or package_dir.name.startswith("_"):
            continue
        for module_path in sorted(package_dir.glob("*.py")):
            if module_path.stem == "__init__" or module_path.name.startswith("_"):
                continue
            module_name = (
                f"flashinfer.trace.solutions.{package_dir.name}.{module_path.stem}"
            )
            module = importlib.import_module(module_name)
            assert module.definition == package_dir.name
            assert getattr(module, "backend", None) != "auto"
            assert callable(module.run)
            assert module.api_kwargs == _expected_api_kwargs(
                module.definition, module.api
            )


def test_solution_run_signatures_accept_definition_inputs():
    for definition, entries in _json_by_definition().items():
        modules = load_solutions(definition)
        assert modules, f"No importable solution modules for {definition!r}"
        for module in modules:
            signature = inspect.signature(module.run)
            parameters = list(signature.parameters.values())
            assert all(
                param.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                for param in parameters
            ), f"{module.__name__}.run should expose trace inputs explicitly"
            for path, data in entries:
                expected_inputs = tuple(data["inputs"])
                expected_outputs = tuple(data["outputs"])
                assert module.inputs == expected_inputs
                assert module.outputs == expected_outputs
                assert tuple(signature.parameters) == expected_inputs, (
                    f"{module.__name__}.run signature does not match inputs from {path}"
                )
                if getattr(module, "requires_setup", False):
                    setup_signature = inspect.signature(module.setup)
                    assert tuple(setup_signature.parameters) == expected_inputs, (
                        f"{module.__name__}.setup signature does not match inputs "
                        f"from {path}"
                    )


def test_backend_solution_modules_are_explicit():
    for definition, expected in _EXPECTED_EXPLICIT_BACKEND_SOLUTIONS.items():
        modules = {
            module.__name__.rsplit(".", 1)[-1]: getattr(module, "backend", None)
            for module in load_solutions(definition)
        }
        assert modules == expected


def test_solution_modules_do_not_use_generic_native_runner():
    assert not (_SOLUTIONS_DIR / "_native.py").exists()
    for module_path in _SOLUTIONS_DIR.glob("*/*.py"):
        assert "run_native_solution" not in module_path.read_text()


def test_stateful_solution_run_excludes_setup_work():
    for definition in (
        "gqa_paged_decode",
        "gqa_paged_prefill",
        "gqa_ragged",
        "mla_paged_decode",
        "segment_gemm_run",
    ):
        for module in load_solutions(definition):
            assert module.requires_setup is True
            run_source = inspect.getsource(module.run)
            assert ".plan(" not in run_source
            assert "workspace(" not in run_source
            assert "Wrapper(" not in run_source
            assert "torch.empty" not in run_source
            assert "torch.full" not in run_source
            setup_source = inspect.getsource(module.setup)
            assert "Wrapper(" in setup_source
