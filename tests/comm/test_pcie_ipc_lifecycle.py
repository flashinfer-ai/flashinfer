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

"""CPU tests for collective failure handling and workspace resource ownership."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.comm._pcie_ipc_lifecycle import (
    bind_stream,
    joint_check,
    release_workspace,
)


@pytest.mark.parametrize("require_identical", [False, True])
def test_remote_failure_precedes_capability_agreement(monkeypatch, require_identical):
    def gather(output, local, group):
        output[:] = [local, {"error": "peer init failed", "capability": False}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(ValueError, match="peer init failed"):
        joint_check(
            object(),
            2,
            {"error": None, "capability": True},
            "preparing",
            collective_name="test IPC",
            require_identical=require_identical,
        )


def test_capability_exchange_can_differ_but_layout_cannot(monkeypatch):
    def gather(output, local, group):
        output[:] = [local, {"error": None, "capability": False}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    args = (object(), 2, {"error": None, "capability": True}, "preparing")
    entries = joint_check(*args, collective_name="test IPC", require_identical=False)
    assert [entry["capability"] for entry in entries] == [True, False]
    with pytest.raises(ValueError, match="identical collective arguments"):
        joint_check(*args, collective_name="test IPC")


def test_graph_capture_preserves_eager_stream_binding(monkeypatch):
    original, other = object(), object()
    device = torch.device("cuda", 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: other)
    assert bind_stream(device, original, "all_gather") is original
    assert bind_stream(device, None, "all_gather") is None

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with pytest.raises(RuntimeError, match="one workspace per stream"):
        bind_stream(device, original, "all_gather")
    assert bind_stream(device, None, "all_gather") is other


def test_free_failure_does_not_repeat_protocol_disposal(monkeypatch):
    events = []
    workspace = SimpleNamespace(
        device=torch.device("cuda", 0), group=object(), _handle=17, _ipc_ptrs=[100, 200]
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))

    def dispose(handle):
        assert handle == 17
        events.append("dispose")

    def failed_free(ptrs, group):
        assert group is workspace.group
        events.append("failed_free")
        # Fail before changing any mappings. Partial unmap recovery belongs
        # to the IPC allocator and is not supplied by the lifecycle helper.
        raise RuntimeError("free failed")

    with pytest.raises(RuntimeError, match="free failed"):
        release_workspace(workspace, dispose=dispose, free=failed_free)
    assert workspace._handle is None
    assert workspace._ipc_ptrs == [100, 200]

    def finish_free(ptrs, group):
        assert ptrs == [100, 200]
        events.append("finish_free")

    release_workspace(workspace, dispose=dispose, free=finish_free)
    release_workspace(workspace, dispose=dispose, free=finish_free)
    assert workspace._ipc_ptrs is None
    assert events == ["sync", "dispose", "failed_free", "sync", "finish_free"]


def test_dispose_failure_keeps_handle_and_peer_mappings(monkeypatch):
    workspace = SimpleNamespace(
        device=torch.device("cuda", 0), group=object(), _handle=17, _ipc_ptrs=[100, 200]
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    def dispose(handle):
        raise RuntimeError("dispose failed")

    def unexpected_free(*args, **kwargs):
        pytest.fail("peer mappings must survive failed protocol disposal")

    with pytest.raises(RuntimeError, match="dispose failed"):
        release_workspace(workspace, dispose=dispose, free=unexpected_free)
    assert workspace._handle == 17
    assert workspace._ipc_ptrs == [100, 200]
