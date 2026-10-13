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

"""Host-side bounds checks for shared dispatch/combine storage."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flashinfer.comm import trtllm_moe_alltoall as a2a


@pytest.fixture
def comm():
    instance = a2a.MoeAlltoAll.__new__(a2a.MoeAlltoAll)
    instance.ep_size, instance.ep_rank = 4, 0
    instance.max_num_tokens = 32
    instance.workspace_size_per_rank = 128 + 4 * 32 * (256 + 64 + 256)
    instance._WORKSPACE = {}
    instance._METAINFO_INDEX = {"PAYLOAD_DATA_OFFSET_INDEX": 0}
    instance.metainfo = torch.tensor([128])
    instance._state = a2a._A2AState(phase="dispatched", dispatch_payload_end=128)
    instance.mnnvl_mem = SimpleNamespace(mapped=True)
    return instance


def test_combine_offset_is_shared_and_stable(comm):
    other = copy.copy(comm)
    offset = comm._reserve_combine_region(128, torch.bfloat16)
    for tokens in (1, 9, 32, 2):
        for quantized in (False, True):
            payloads = [
                torch.empty(
                    (0, 64 if quantized else 128),
                    dtype=torch.uint8 if quantized else torch.bfloat16,
                ),
                torch.empty((0, 8), dtype=torch.int32),
                torch.empty((0, 8)),
            ]
            if quantized:
                payloads.append(torch.empty((0, 8), dtype=torch.uint8))
            other._check_dispatch_region(payloads, tokens)
            assert other._reserve_combine_region(128, torch.bfloat16) == offset
    assert offset % 128 == 0
    assert offset + 4 * 32 * 128 * 2 <= comm.workspace_size_per_rank


@pytest.mark.parametrize("tokens", [0, -1, 33])
def test_dispatch_rejects_token_count_before_launch(comm, monkeypatch, tokens):
    native = Mock()
    monkeypatch.setattr(a2a, "moe_a2a_dispatch", native)
    comm._state.phase = "idle"
    with pytest.raises(ValueError, match="capacity"):
        comm.dispatch(torch.empty((0, 8)), [torch.empty((0, 128))], tokens)
    native.assert_not_called()


def test_dispatch_rejects_overlap_before_launch(comm, monkeypatch):
    comm._reserve_combine_region(128, torch.bfloat16)
    comm._state.phase = "idle"
    native = Mock()
    monkeypatch.setattr(a2a, "moe_a2a_dispatch", native)
    with pytest.raises(ValueError, match="workspace region"):
        comm.dispatch(torch.empty((0, 8)), [torch.empty((0, 256))], 32)
    native.assert_not_called()


@pytest.mark.parametrize("hidden,dtype", [(64, torch.bfloat16), (128, torch.float32)])
def test_rejects_combine_layout_change(comm, hidden, dtype):
    comm._reserve_combine_region(128, torch.bfloat16)
    with pytest.raises(ValueError, match="stable combine"):
        comm._reserve_combine_region(hidden, dtype)


@pytest.mark.parametrize("too_small", [False, True])
def test_failed_reservation_leaves_shared_state_unchanged(comm, too_small):
    if too_small:
        comm.workspace_size_per_rank = 256
    else:
        comm._state.dispatch_payload_end = comm.workspace_size_per_rank
    with pytest.raises(ValueError, match="too small|overlaps"):
        comm._reserve_combine_region(128, torch.bfloat16)
    assert not comm._WORKSPACE


def test_workspace_view_and_combine_use_reserved_offset(comm, monkeypatch):
    comm.workspace = torch.empty((4, comm.workspace_size_per_rank), dtype=torch.uint8)
    comm._backend = "trtllm"
    comm.top_k, comm.enable_rank_mask = 8, False
    comm._state.local_num_tokens = 1
    wrap = Mock(return_value=torch.empty((4, 9, 128), dtype=torch.bfloat16))
    native = Mock(return_value=torch.empty((1, 128), dtype=torch.bfloat16))
    monkeypatch.setattr(a2a, "moe_a2a_wrap_payload_tensor_in_workspace", wrap)
    monkeypatch.setattr(a2a, "moe_a2a_combine", native)
    payload = comm.get_combine_payload_tensor_in_workspace(9, 128, torch.bfloat16)
    offset = comm._WORKSPACE["combine_storage_offset"]
    assert wrap.call_args.args[2:4] == (offset, offset + 4 * 9 * 128 * 2)
    comm.combine(payload, 9, payload_in_workspace=True)
    assert native.call_args.args[8:10] == (offset, True)
    assert comm._state.phase == "idle"
