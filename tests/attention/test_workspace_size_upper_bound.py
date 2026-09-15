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
"""

import pathlib

import pytest
import torch

from flashinfer import decode as decode_module
from flashinfer import prefill as prefill_module

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _bare_prefill_wrapper(backend="auto", is_cuda_graph_enabled=False):
    cls = prefill_module.BatchPrefillWithPagedKVCacheWrapper
    wrapper = cls.__new__(cls)
    wrapper._backend = backend
    wrapper._jit_module = None
    wrapper.device = torch.device("cpu")
    wrapper._use_cuda_graph = is_cuda_graph_enabled
    wrapper._float_workspace_buffer = torch.empty(1, dtype=torch.uint8)
    return wrapper


def _bare_decode_wrapper(backend="auto", use_tensor_cores=True):
    cls = decode_module.BatchDecodeWithPagedKVCacheWrapper
    wrapper = cls.__new__(cls)
    wrapper._backend = backend
    wrapper._jit_module = None
    wrapper.device = torch.device("cpu")
    wrapper._use_tensor_cores = use_tensor_cores
    wrapper._use_cuda_graph = False
    wrapper._float_workspace_buffer = torch.empty(1, dtype=torch.uint8)
    return wrapper


class _ModuleWithoutBound:
    """A backend module that has no upper-bound entry point, like fa3."""


class _ModuleWithBound:
    def __init__(self):
        self.args = None

    def workspace_size_upper_bound(self, *args):
        self.args = args
        return (1024, 64)


def test_prefill_bound_resolves_auto_with_the_head_dims_plan_uses(monkeypatch):
    """`auto` has to land on the same backend the plan will get.

    The selector takes the head dimensions into account, so a bound that
    resolved without them could describe a different scheduler than the one
    that ends up planning.
    """
    seen = {}

    def fake_determine(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "fa2"

    monkeypatch.setattr(prefill_module, "determine_attention_backend", fake_determine)
    bound_module = _ModuleWithBound()
    monkeypatch.setattr(
        prefill_module, "get_batch_prefill_module", lambda *a, **k: bound_module
    )

    wrapper = _bare_prefill_wrapper()
    wrapper.workspace_size_upper_bound(
        max_batch_size=4,
        max_total_num_rows=64,
        max_num_pages_per_request=8,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=192,
        page_size=16,
        head_dim_vo=128,
    )

    assert seen["kwargs"]["head_dim_qk"] == 192
    assert seen["kwargs"]["head_dim_vo"] == 128


def test_prefill_bound_and_workspace_size_resolve_auto_the_same_way(monkeypatch):
    """Sizing and bounding must not drift onto different backends."""
    calls = []

    def fake_determine(*args, **kwargs):
        calls.append(kwargs)
        return "fa2"

    monkeypatch.setattr(prefill_module, "determine_attention_backend", fake_determine)

    resolved = prefill_module._resolve_prefill_backend(
        "auto",
        torch.device("cpu"),
        "NONE",
        False,
        False,
        torch.float16,
        torch.float16,
        192,
        128,
    )
    assert resolved == "fa2"
    assert calls == [{"head_dim_qk": 192, "head_dim_vo": 128}]

    assert (
        prefill_module._resolve_prefill_backend(
            "fa3",
            torch.device("cpu"),
            "NONE",
            False,
            False,
            torch.float16,
            torch.float16,
            128,
            128,
        )
        == "fa3"
    )
    # An explicit backend is never re-resolved.
    assert len(calls) == 1


def test_prefill_bound_refuses_a_backend_without_an_entry_point(monkeypatch):
    """A backend with no bound falls closed rather than borrowing another's."""
    monkeypatch.setattr(
        prefill_module, "determine_attention_backend", lambda *a, **k: "fa3"
    )
    monkeypatch.setattr(
        prefill_module,
        "get_batch_prefill_module",
        lambda *a, **k: _ModuleWithoutBound(),
    )

    wrapper = _bare_prefill_wrapper()
    with pytest.raises(NotImplementedError):
        wrapper.workspace_size_upper_bound(
            max_batch_size=4,
            max_total_num_rows=64,
            max_num_pages_per_request=8,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=128,
            page_size=16,
        )


@pytest.mark.parametrize(
    ("q_data_type", "kv_data_type", "expected"),
    [
        pytest.param(torch.float16, torch.float16, "fa2", id="fp16-stays-fa2"),
        pytest.param(
            torch.float8_e4m3fn, torch.float8_e4m3fn, "fa3", id="fp8-asks-the-selector"
        ),
    ],
)
def test_decode_tensor_core_bound_resolves_auto_like_plan(
    monkeypatch, q_data_type, kv_data_type, expected
):
    """Tensor-core decode is planned as a prefill, and which one depends on dtype."""
    monkeypatch.setattr(
        decode_module, "determine_attention_backend", lambda *a, **k: "fa3"
    )
    assert (
        decode_module._resolve_decode_tensor_core_backend(
            "auto",
            torch.device("cpu"),
            "NONE",
            q_data_type,
            kv_data_type,
            128,
            1,
        )
        == expected
    )


def test_decode_tensor_core_bound_refuses_fa3_without_an_entry_point(monkeypatch):
    """An FP8 decode that resolves to fa3 must not be given the fa2 bound."""
    monkeypatch.setattr(
        decode_module, "determine_attention_backend", lambda *a, **k: "fa3"
    )
    monkeypatch.setattr(
        decode_module, "get_batch_prefill_module", lambda *a, **k: _ModuleWithoutBound()
    )

    wrapper = _bare_decode_wrapper()
    with pytest.raises(NotImplementedError):
        wrapper.workspace_size_upper_bound(
            max_batch_size=4,
            max_num_pages_per_request=8,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=16,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
        )


def test_decode_bound_forwards_the_split_settings(monkeypatch):
    """A fixed split bypasses the scheduler ceiling, so the bound must see it."""
    monkeypatch.setattr(
        decode_module, "determine_attention_backend", lambda *a, **k: "fa2"
    )
    bound_module = _ModuleWithBound()
    monkeypatch.setattr(
        decode_module, "get_batch_prefill_module", lambda *a, **k: bound_module
    )

    wrapper = _bare_decode_wrapper()
    wrapper.workspace_size_upper_bound(
        max_batch_size=4,
        max_num_pages_per_request=8,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        page_size=16,
        fixed_split_size=8,
        disable_split_kv=True,
    )

    # (buffer, batch, rows, pages, qo, kv, page_size, graph, qk, vo,
    #  fixed_split_size, disable_split_kv, colocated)
    assert bound_module.args[10] == 8
    assert bound_module.args[11] is True


def test_decode_bound_rejects_a_zero_q_len_per_req():
    wrapper = _bare_decode_wrapper()
    with pytest.raises(ValueError):
        wrapper.workspace_size_upper_bound(
            max_batch_size=4,
            max_num_pages_per_request=8,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=16,
            q_len_per_req=0,
        )


def test_decode_cuda_core_bound_refuses_the_split_settings():
    """Only the tensor-core path plans through a scheduler that takes them."""
    wrapper = _bare_decode_wrapper(use_tensor_cores=False)
    with pytest.raises(NotImplementedError):
        wrapper.workspace_size_upper_bound(
            max_batch_size=4,
            max_num_pages_per_request=8,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=16,
            fixed_split_size=8,
        )


def test_cta_tile_q_candidates_have_a_single_source():
    """The selector and the bound must iterate the same list.

    The bound covers a tile it cannot predict by evaluating every candidate,
    so a second copy of the list is a silent way to weaken it.
    """
    utils = (_REPO_ROOT / "include/flashinfer/utils.cuh").read_text()
    scheduler = (_REPO_ROOT / "include/flashinfer/attention/scheduler.cuh").read_text()

    assert utils.count("constexpr uint32_t kFA2CtaTileQCandidates[]") == 1
    assert "kFA2CtaTileQCandidates[]" not in scheduler
    assert "kFA2CtaTileQCandidates" in scheduler
    # The selector's result is checked against the list rather than assumed.
    assert "FA2CtaTileQIsCandidate(cta_tile_q)" in utils
