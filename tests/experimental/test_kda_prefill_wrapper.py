# Copyright (c) 2026 by FlashInfer team.
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

"""Tests for the experimental ``RecurrentKDAPrefillWrapper`` (see #4936).

Numerics for the kernels the wrapper hands off to are covered by the stable
lane in ``tests/kda/test_recurrent_kda_prefill.py``; these cover the wrapper's
own contract -- that ``plan`` stages offsets without reading them on the host,
and that ``run`` forwards the planned buffers.
"""

import importlib

import pytest
import torch

from flashinfer.kda import RecurrentKDAPrefillWrapper

from tests.test_helpers.kda_prefill import cpu_route_tensors

kda_api = importlib.import_module("flashinfer.kda")


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


def test_prefill_wrapper_plan_builds_stable_device_metadata(cuda_device, monkeypatch):
    wrapper = RecurrentKDAPrefillWrapper(cuda_device)
    offsets = torch.tensor([0, 0, 7, 7, 12], device=cuda_device)
    original_to = torch.Tensor.to

    def reject_device_to_host(self, *args, **kwargs):
        if args and torch.device(args[0]).type == "cpu" and self.is_cuda:
            pytest.fail("wrapper plan must not read CUDA offsets on the host")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", reject_device_to_host)
    wrapper.plan(offsets)

    cu_seqlens_ptr = wrapper._cu_seqlens_buf.data_ptr()
    seq_order_ptr = wrapper._seq_order_buf.data_ptr()
    cu_chunks_ptr = wrapper._cu_chunks_buf.data_ptr()
    assert wrapper._cu_seqlens_buf.dtype == torch.int64
    assert wrapper._cu_seqlens_buf.tolist() == [0, 0, 7, 7, 12]
    assert wrapper._workspace._cute_dsl_generate_planned_metadata is True

    wrapper.plan(torch.tensor([0, 0, 2, 2, 12], device=cuda_device))
    assert wrapper._cu_seqlens_buf.data_ptr() == cu_seqlens_ptr
    assert wrapper._seq_order_buf.data_ptr() == seq_order_ptr
    assert wrapper._cu_chunks_buf.data_ptr() == cu_chunks_ptr

    with pytest.raises(ValueError, match="number of sequences is fixed"):
        wrapper.plan(torch.tensor([0, 2, 12], device=cuda_device))


def test_prefill_wrapper_run_forwards_planned_buffers(cuda_device, monkeypatch):
    wrapper = RecurrentKDAPrefillWrapper(cuda_device)
    wrapper.plan(torch.tensor([0, 1, 3], device=cuda_device))
    calls = []
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_api,
        "recurrent_kda",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )
    tensors = cpu_route_tensors(token_count=3)
    tensors = {
        key: value.to(cuda_device) if isinstance(value, torch.Tensor) else value
        for key, value in tensors.items()
    }

    assert wrapper.run(**tensors) is sentinel
    assert calls[0]["cu_seqlens"] is wrapper._cu_seqlens_buf
    assert calls[0]["seq_order"] is wrapper._seq_order_buf
    assert calls[0]["prefill_workspace"] is wrapper._workspace
    assert calls[0]["backend"] == "cute-dsl"
    assert wrapper._workspace._cute_dsl_cu_chunks is wrapper._cu_chunks_buf
    assert wrapper._workspace._cute_dsl_generate_planned_metadata is True
