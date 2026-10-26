# Copyright (c) 2026 by FlashInfer team.
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

import threading
from types import SimpleNamespace

import pytest
import torch

from flashinfer.kda_prefill import (
    _cached_packed_task_metadata,
    _generated_affine_state_indices_i64,
)


@pytest.fixture
def workspace():
    return SimpleNamespace(
        _packed_metadata_lock=threading.Lock(),
        _packed_metadata_tensor=None,
        _packed_metadata_signature=None,
        _packed_metadata=None,
        _generated_affine_state_indices_tensor=None,
        _generated_affine_state_indices_version=None,
        _generated_affine_state_indices_destination_ptr=None,
    )


def packed_metadata(workspace, offsets):
    return _cached_packed_task_metadata(
        workspace,
        offsets,
        total_tokens=6,
        num_heads=96,
        sm_count=148,
        build_persistent_plan=True,
    )


@pytest.mark.parametrize("inference", [False, True])
def test_packed_metadata_mutation(workspace, monkeypatch, inference):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with torch.inference_mode(inference):
        offsets = torch.tensor([0, 2, 6], dtype=torch.int64)
        first = packed_metadata(workspace, offsets)
        second = packed_metadata(workspace, offsets)
        assert (first is second) is not inference
        assert first[0] == (1, 0)
        assert first[3:] == ((0, 2, 6), (2, 4))

        offsets[1] = 5
        updated = packed_metadata(workspace, offsets)
        assert updated[0] == (0, 1)
        assert updated[3:] == ((0, 5, 6), (5, 1))

        offsets[1] = 0
        with pytest.raises(ValueError, match="strictly increasing"):
            packed_metadata(workspace, offsets)


@pytest.mark.parametrize("inference", [False, True])
def test_packed_metadata_requires_capture_warmup(workspace, monkeypatch, inference):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with torch.inference_mode(inference):
        offsets = torch.tensor([0, 2, 6], dtype=torch.int64)
        with pytest.raises(RuntimeError, match="not warmed"):
            packed_metadata(workspace, offsets)


@pytest.mark.parametrize("inference", [False, True])
def test_affine_state_indices_mutation(workspace, inference):
    with torch.inference_mode(inference):
        source = torch.tensor([2], dtype=torch.int32)
        plan = SimpleNamespace(state_indices_i64=torch.empty(1, dtype=torch.int64))

        def carrier():
            return _generated_affine_state_indices_i64(
                workspace=workspace, plan=plan, state_indices=source, capturing=False
            )

        assert carrier().tolist() == [2]
        plan.state_indices_i64.fill_(7)
        assert carrier().tolist() == ([2] if inference else [7])
        source.fill_(3)
        assert carrier().tolist() == [3]


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph test")
def test_inference_metadata_cuda_graph(workspace, inference):
    with torch.inference_mode(inference):
        offsets = torch.tensor([0, 2, 6], dtype=torch.int64, device="cuda")
        source = torch.tensor([2], dtype=torch.int32, device="cuda")
        plan = SimpleNamespace(
            state_indices_i64=torch.empty(1, dtype=torch.int64, device="cuda")
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            warmed = packed_metadata(workspace, offsets)
            _generated_affine_state_indices_i64(
                workspace=workspace, plan=plan, state_indices=source, capturing=False
            )
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            assert packed_metadata(workspace, offsets) is warmed
            _generated_affine_state_indices_i64(
                workspace=workspace, plan=plan, state_indices=source, capturing=True
            )
        source.fill_(4)
        graph.replay()
        torch.cuda.synchronize()
        assert plan.state_indices_i64.tolist() == [4]
