"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Names of the deterministic mHC training package: every public name of
``flashinfer.cake_mhc_det_train`` and of its backends carries ``det`` (the
generic constants and route helpers below are qualified by the module path),
the names without ``det`` that FlashInfer 0.7.3 used for the same objects are
gone (no alias module, no top-level alias), and no entry point takes a
``deterministic`` keyword.
"""

import importlib
import importlib.util
import inspect

import pytest

import flashinfer

REMOVED = (
    "flashinfer.cake_mhc_train",
    "flashinfer.cake_mhc_train_backend",
    "flashinfer.cake_mhc_train_head_backend",
)
DET = (
    "flashinfer.cake_mhc_det_train",
    "flashinfer.cake_mhc_det_train_backend",
    "flashinfer.cake_mhc_det_train_head_backend",
)
GENERIC = {
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "TENSOR_ALIGNMENT_BYTES",
    "WORKSPACE_ALIGNMENT_BYTES",
    "route_key",
    "route_modules",
}


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", REMOVED)
def test_names_without_det_are_gone(name: str) -> None:
    spec = importlib.util.find_spec(name)
    # Absent, or at most a namespace package left by a stale __pycache__ directory.
    assert spec is None or spec.origin is None, name
    assert not hasattr(flashinfer, name.rpartition(".")[2]), name


@pytest.mark.cpu_only
def test_top_level_has_no_mhc_training_name_without_det() -> None:
    leftovers = [
        n for n in dir(flashinfer) if "cake_mhc_train" in n or "CakeMHCTrain" in n
    ]
    assert not leftovers, leftovers


@pytest.mark.cpu_only
@pytest.mark.parametrize("module_name", DET)
def test_det_names_and_no_mode_keyword(module_name: str) -> None:
    module = importlib.import_module(module_name)
    without_det = [
        n
        for n in module.__all__
        if n not in GENERIC and "_det_" not in n and "Det" not in n
    ]
    assert not without_det, without_det
    for name in module.__all__:
        obj = getattr(module, name)
        if not callable(obj) or isinstance(obj, type):
            continue
        for parameter in inspect.signature(obj).parameters.values():
            assert parameter.name != "deterministic", name
            assert parameter.kind is not inspect.Parameter.VAR_KEYWORD, name
