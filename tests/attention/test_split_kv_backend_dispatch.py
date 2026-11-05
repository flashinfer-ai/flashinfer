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

"""``fixed_split_size`` / ``disable_split_kv`` never reach a backend that drops them.

Only the fa2 planner takes the two split-KV controls. ``backend="auto"`` keeps
fa2 whenever either is set (also where it would take fa3 for FP8 decode or
Hopper prefill, or upgrade ragged prefill on Blackwell), an explicit ``fa3``
raises, and the values reach the fa2 planner intact. The CPU cases mock the
compiled module (issue #7981).
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import flashinfer
import flashinfer.decode as decode
import flashinfer.prefill as prefill
from flashinfer.utils import is_sm90a_supported

SPLIT_CONTROLS = [
    pytest.param({"disable_split_kv": True}, id="disable_split_kv"),
    pytest.param({"fixed_split_size": 64}, id="fixed_split_size"),
]
Q_DTYPE = torch.bfloat16
KV_DTYPE = torch.float8_e4m3fn  # makes decode ``auto`` consult the fa3 resolver


@pytest.fixture
def planner_calls(monkeypatch):
    # Pinned allocation needs a CUDA driver even for CPU tensors.
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    calls = []

    def plan(*args):
        calls.append(args)
        return []

    def workspace_size(*args):
        calls.append(args)
        return 0, 0

    module = SimpleNamespace(plan=plan, workspace_size=workspace_size)
    for mod in (decode, prefill):
        monkeypatch.setattr(mod, "get_batch_prefill_module", lambda *a, **k: module)
        monkeypatch.setattr(mod, "determine_attention_backend", lambda *a, **k: "fa3")
    monkeypatch.setattr(
        decode.BatchDecodeWithPagedKVCacheWrapper,
        "_auto_prefers_cudnn",
        lambda self, *a, **k: False,
    )
    monkeypatch.setattr(
        prefill, "_blackwell_ragged_auto_upgrade", lambda *a, **k: "cudnn"
    )
    return calls


def _decode_wrapper(backend):
    return decode.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(1024, dtype=torch.uint8),
        use_tensor_cores=True,
        backend=backend,
    )


def _decode_args():
    return (
        torch.tensor([0, 4], dtype=torch.int32),
        torch.arange(4, dtype=torch.int32),
        torch.tensor([16], dtype=torch.int32),
        8,
        2,
        128,
        16,
    )


def _decode_kwargs(**extra):
    return {"q_data_type": Q_DTYPE, "kv_data_type": KV_DTYPE, **extra}


def _paged_prefill_wrapper(backend):
    return prefill.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(1024, dtype=torch.uint8), "NHD", backend=backend
    )


def _paged_prefill_args():
    return (
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 4], dtype=torch.int32),
        torch.arange(4, dtype=torch.int32),
        torch.tensor([16], dtype=torch.int32),
        8,
        2,
        128,
        16,
    )


def _ragged_prefill_wrapper(backend):
    return prefill.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(1024, dtype=torch.uint8), "NHD", backend=backend
    )


def _ragged_prefill_args():
    return (
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 64], dtype=torch.int32),
        8,
        2,
        128,
    )


def _assert_fa2_split_args(args, first, control):
    assert args[first] == control.get("fixed_split_size", -1)
    assert args[first + 1] is control.get("disable_split_kv", False)


@pytest.mark.parametrize("control", SPLIT_CONTROLS)
@pytest.mark.parametrize("entrypoint", ["plan", "workspace_size"])
def test_decode_auto_keeps_fa2_with_split_control(planner_calls, entrypoint, control):
    wrapper = _decode_wrapper("auto")
    getattr(wrapper, entrypoint)(*_decode_args(), **_decode_kwargs(**control))
    assert len(planner_calls) == 1
    _assert_fa2_split_args(
        planner_calls[0], 14 if entrypoint == "workspace_size" else 16, control
    )
    if entrypoint == "plan":
        assert wrapper.resolved_backend == "fa2"


def test_decode_auto_takes_fa3_without_split_control(planner_calls):
    wrapper = _decode_wrapper("auto")
    wrapper.plan(*_decode_args(), **_decode_kwargs())
    assert wrapper.resolved_backend == "fa3"
    assert len(planner_calls[0]) == 16


@pytest.mark.parametrize("control", SPLIT_CONTROLS)
@pytest.mark.parametrize("entrypoint", ["plan", "workspace_size", "fast_decode_plan"])
def test_decode_explicit_fa3_rejects_split_control(planner_calls, entrypoint, control):
    wrapper = _decode_wrapper("fa3")
    with pytest.raises(ValueError, match="honored only by the fa2 backend"):
        if entrypoint == "fast_decode_plan":
            decode.fast_decode_plan(
                wrapper, *_decode_args(), **_decode_kwargs(**control)
            )
        else:
            getattr(wrapper, entrypoint)(*_decode_args(), **_decode_kwargs(**control))
    assert not planner_calls


@pytest.mark.parametrize("control", SPLIT_CONTROLS)
@pytest.mark.parametrize("entrypoint", ["plan", "workspace_size"])
def test_paged_prefill_auto_keeps_fa2_with_split_control(
    planner_calls, entrypoint, control
):
    wrapper = _paged_prefill_wrapper("auto")
    getattr(wrapper, entrypoint)(
        *_paged_prefill_args(), causal=True, q_data_type=Q_DTYPE, **control
    )
    assert len(planner_calls) == 1
    _assert_fa2_split_args(
        planner_calls[0], 14 if entrypoint == "workspace_size" else 16, control
    )
    if entrypoint == "plan":
        assert wrapper._backend == "fa2"


@pytest.mark.parametrize("control", SPLIT_CONTROLS)
@pytest.mark.parametrize("entrypoint", ["plan", "workspace_size"])
def test_paged_prefill_explicit_fa3_rejects_split_control(
    planner_calls, entrypoint, control
):
    wrapper = _paged_prefill_wrapper("fa3")
    with pytest.raises(ValueError, match="honored only by the fa2 backend"):
        getattr(wrapper, entrypoint)(
            *_paged_prefill_args(), causal=True, q_data_type=Q_DTYPE, **control
        )
    assert not planner_calls


@pytest.mark.parametrize("control", SPLIT_CONTROLS)
def test_ragged_prefill_auto_keeps_fa2_with_split_control(planner_calls, control):
    wrapper = _ragged_prefill_wrapper("auto")
    wrapper.plan(*_ragged_prefill_args(), causal=True, q_data_type=Q_DTYPE, **control)
    assert wrapper._backend == "fa2"
    assert len(planner_calls) == 1
    _assert_fa2_split_args(planner_calls[0], 16, control)


@pytest.mark.parametrize("control", SPLIT_CONTROLS)
def test_ragged_prefill_explicit_fa3_rejects_split_control(planner_calls, control):
    wrapper = _ragged_prefill_wrapper("fa3")
    with pytest.raises(ValueError, match="honored only by the fa2 backend"):
        wrapper.plan(
            *_ragged_prefill_args(), causal=True, q_data_type=Q_DTYPE, **control
        )
    assert not planner_calls


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda")),
    reason="auto resolves FP8-query decode to fa3 only on SM90",
)
@pytest.mark.parametrize("control", SPLIT_CONTROLS)
def test_sm90_fp8_query_auto_decode_does_not_drop_split_control(control):
    device = torch.device("cuda")
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    indptr = torch.tensor([0, 4], dtype=torch.int32, device=device)
    indices = torch.arange(4, dtype=torch.int32, device=device)
    last_page_len = torch.tensor([16], dtype=torch.int32, device=device)
    fp8 = torch.float8_e4m3fn

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=True, backend="auto"
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        8,
        2,
        128,
        16,
        q_data_type=fp8,
        kv_data_type=fp8,
        o_data_type=torch.bfloat16,
    )
    assert wrapper.resolved_backend == "fa3"

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=True, backend="auto"
    )
    # fa2 has no e4m3-query decode on SM90: fail rather than silently take fa3.
    with pytest.raises((ValueError, NotImplementedError)):
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            8,
            2,
            128,
            16,
            q_data_type=fp8,
            kv_data_type=fp8,
            o_data_type=torch.bfloat16,
            **control,
        )
    assert wrapper.resolved_backend != "fa3"
