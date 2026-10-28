# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental admission, constructor warning and hot-path contracts."""

import importlib.util
import builtins
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch


def test_benchmark_workspace_traverses_experimental_owners():
    benchmark = (
        Path(__file__).resolve().parents[3]
        / "benchmarks/bench_fp8_mqa_selective_logits.py"
    )
    spec = importlib.util.spec_from_file_location("workspace_benchmark", benchmark)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    owner_type = type(
        "Owner", (), {"__module__": "flashinfer.experimental.attn_scores"}
    )
    owner = owner_type()
    owner.workspace = torch.empty(8)
    owner.alias = owner.workspace[:4]
    owner.input = torch.empty(12)
    inventory = module._workspace_inventory(owner, (owner.input,))
    assert inventory == {
        "owned_reachable_tensor_bytes": 8 * owner.workspace.element_size()
    }


def test_backend_definition_import_preserves_original_exception(monkeypatch):
    from flashinfer.experimental.attn_scores import _wrapper_type

    _wrapper_type.cache_clear()
    original_import = builtins.__import__
    missing = ModuleNotFoundError("missing backend dependency")

    def import_definition(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "selective_logits"
            and level == 1
            and globals is not None
            and globals.get("__package__") == "flashinfer.experimental.attn_scores"
        ):
            raise missing
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_definition)
    with pytest.raises(ModuleNotFoundError) as caught:
        _wrapper_type()
    assert caught.value is missing


def test_wrapper_construction_warns_once_without_wrapping_run():
    """Use a fresh process so another test cannot consume the once warning."""
    program = r"""
        import inspect
        import sys
        import warnings
        import flashinfer
        from flashinfer import FP8PagedMQATopKWrapper as Public
        from flashinfer.api_logging import ExperimentalWarning
        from flashinfer.experimental import attn_scores

        assert Public is flashinfer.attn_scores.FP8PagedMQATopKWrapper
        assert Public.is_experimental and Public.__init__.is_experimental
        assert not any(name.startswith("flashinfer.experimental.attn_scores.kernels")
                       for name in sys.modules)

        class Backend(Public):
            def _initialize(self, **kwargs):
                self.options = kwargs
            def run(self, q, weights, kv_fused, *, out=None):
                return q

        attn_scores._wrapper_type = lambda: Backend
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ExperimentalWarning)
            first = Public(strategy="selective")
            second = Public(strategy="full")
        assert len([w for w in caught if issubclass(w.category, ExperimentalWarning)]) == 1
        assert isinstance(first, Public) and isinstance(second, Public)
        assert first.run.__func__ is Backend.run
        assert first.run(1, 2, 3) == 1
        assert tuple(inspect.signature(Public.plan).parameters) == (
            "self", "qo_indptr", "kv_indptr", "block_table", "num_query_rows")
        assert tuple(inspect.signature(Public.run).parameters) == (
            "self", "q", "weights", "kv_fused", "out")
        assert inspect.signature(Public.plan).parameters["qo_indptr"].annotation == "torch.Tensor"
        assert inspect.signature(Public.plan).return_annotation == "None"
        assert inspect.signature(Public.run).parameters["out"].annotation == "torch.Tensor | None"
        assert inspect.signature(Public.run).return_annotation == "torch.Tensor"
        constructor = inspect.signature(Public)
        assert tuple(constructor.parameters) == tuple(
            key for key in inspect.signature(Public.__init__).parameters if key != "self")
    """
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)], check=True, timeout=120
    )


@pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="CuTe DSL is required"
)
def test_first_definition_import_does_not_pin_constructor_inputs():
    """DSL import frames must not retain even rejected constructor arguments."""
    program = r"""
        import gc
        import weakref
        import torch
        from flashinfer import FP8PagedMQATopKWrapper

        gc.disable()
        try:
            layout = torch.empty(4, 128, 1, 132, dtype=torch.uint8)
            owner = weakref.ref(layout)
            try:
                FP8PagedMQATopKWrapper(strategy="invalid", kv_layout=layout)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid strategy was accepted")
            del layout
            assert owner() is None
        finally:
            gc.enable()
    """
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)], check=True, timeout=180
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("cutlass") is None
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="experimental paged TopK requires SM100 or SM103 and CuTe DSL",
)
@pytest.mark.parametrize("layout_source", ["explicit", "exemplar"])
def test_first_constructor_definition_import_releases_by_refcount(layout_source):
    """First-use DSL decorator frames must not keep the new wrapper alive."""
    program = r"""
        import gc
        import weakref
        import torch
        from flashinfer import FP8PagedMQATopKWrapper

        gc.disable()
        try:
            layout = torch.empty(4, 128, 1, 132, dtype=torch.uint8, device="cuda")
            layout_owner = weakref.ref(layout)
            options = ({"kv_layout": layout} if USE_EXEMPLAR else
                       {"num_pages": 4, "device": "cuda"})
            wrapper = FP8PagedMQATopKWrapper(
                strategy="selective", num_heads=8, top_k=512,
                max_context_length=512, max_chunk_size=1, max_batch_size=1,
                **options)
            owner = weakref.ref(wrapper)
            del options, layout
            assert layout_owner() is None
            del wrapper
            assert owner() is None
        finally:
            gc.enable()
    """
    program = program.replace("USE_EXEMPLAR", repr(layout_source == "exemplar"))
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)], check=True, timeout=180
    )
