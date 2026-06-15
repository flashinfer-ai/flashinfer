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

import torch
import pytest

from flashinfer.comm import mnnvl as mnnvl_mod
from flashinfer.comm.allreduce import TRTLLMAllReduceFusionWorkspace
from flashinfer.comm.mnnvl import SymmDeviceMemory
from flashinfer.comm.trtllm_mnnvl_ar import MNNVLAllReduceFusionWorkspace


class _FakeComm:
    def __init__(self, size=2, rank=0, mapped_states=None, metadata=None):
        self.size = size
        self.rank = rank
        self.barriers = 0
        self.mapped_states = mapped_states
        self.metadata = metadata

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return self.rank

    def allgather(self, value):
        if isinstance(value, bool):
            if self.mapped_states is not None:
                return self.mapped_states
            return [value] * self.size
        if isinstance(value, dict):
            if self.metadata is not None:
                return self.metadata
            return [{**value, "group_rank": rank} for rank in range(self.size)]
        return [value] * self.size

    def barrier(self):
        self.barriers += 1


class _FakeCheckpointableHandle:
    def __init__(self):
        self.comm_backend = _FakeComm()
        self.calls = []

    def validate_graph_visible_addresses(self):
        self.calls.append(("validate",))

    def detach_physical_keep_va(self, *, synchronize=True, barrier=True):
        self.calls.append(("detach", synchronize, barrier))

    def remap_physical_same_va(
        self, *, comm_backend=None, synchronize=True, barrier=True, zero_local=True
    ):
        self.calls.append(("remap", comm_backend, synchronize, barrier, zero_local))

    def lamport_initialize(self, rank, dtype):
        self.calls.append(("lamport", rank, dtype))


def _make_trtllm_workspace():
    workspace = object.__new__(TRTLLMAllReduceFusionWorkspace)
    workspace.world_size = 2
    workspace.rank = 0
    workspace.ipc_handles = [[0x1000, 0x2000], [0x3000, 0x4000], [0x5000, 0x6000]]
    workspace.workspace_tensor = torch.tensor(
        [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000],
        dtype=torch.int64,
    )
    workspace.mem_handles = [_FakeCheckpointableHandle() for _ in range(3)]
    workspace.metadata = {
        "use_fp32_lamport": False,
        "lamport_comm_size": 1024,
    }
    workspace._graph_visible_addresses = workspace.get_graph_visible_addresses()
    workspace._destroyed = False
    return workspace


def test_trtllm_allreduce_checkpoint_hooks_delegate_to_handles():
    workspace = _make_trtllm_workspace()

    workspace.detach_physical_keep_va(synchronize=False, barrier=False)
    workspace.remap_physical_same_va(
        comm_backend="fresh-comm",
        synchronize=False,
        barrier=False,
        reset=False,
    )

    for handle in workspace.mem_handles:
        assert ("validate",) in handle.calls
        assert ("detach", False, False) in handle.calls
        assert ("remap", "fresh-comm", False, False, True) in handle.calls


def test_trtllm_allreduce_checkpoint_hooks_fail_closed_without_handle_support():
    workspace = _make_trtllm_workspace()
    workspace.mem_handles = [object()]

    try:
        workspace.detach_physical_keep_va()
    except RuntimeError as exc:
        assert "does not support" in str(exc)
    else:
        raise AssertionError("checkpoint pause should fail without handle support")


def test_mnnvl_allreduce_checkpoint_hooks_delegate_to_allocator_handle():
    workspace = object.__new__(MNNVLAllReduceFusionWorkspace)
    workspace.world_size = 2
    workspace.rank = 0
    workspace.tp_size = 2
    workspace.ptrs = [0x1000, 0x2000]
    workspace.uc_ptrs_dev = 0x3000
    workspace.uc_ptr_local = 0x1000
    workspace.mc_ptr = 0x4000
    workspace.buffer_size_bytes = 128
    workspace.buffer_flags = torch.tensor(
        [0, 2, workspace.buffer_size_bytes, 0, 0, 0, 0, 0, 0],
        dtype=torch.uint32,
    )
    workspace.workspace_size_bytes = 384
    workspace.handle = _FakeCheckpointableHandle()
    workspace.comm_backend = workspace.handle.comm_backend
    workspace._graph_visible_addresses = workspace.get_graph_visible_addresses()

    workspace.detach_physical_keep_va(synchronize=False, barrier=False)
    workspace.remap_physical_same_va(
        comm_backend="fresh-comm",
        synchronize=False,
        barrier=False,
        reset=False,
    )

    assert ("detach", False, False) in workspace.handle.calls
    assert ("remap", "fresh-comm", False, False, True) in workspace.handle.calls


def _make_symm_memory(comm, *, mapped=True):
    memory = object.__new__(SymmDeviceMemory)
    memory.device_idx = 0
    memory.group_size = 2
    memory.group_rank = 0
    memory.buf_size = 128
    memory.signal_pad_offset = 128
    memory.allocation_size = 0x1000
    memory.total_uc_size = 0x2000
    memory.uc_base_ptr = 0x1000
    memory.uc_ptrs = [0x1000, 0x2000]
    memory.uc_ptrs_dev = 0x3000
    memory.signal_pads = [0x1080, 0x2080]
    memory.signal_pads_dev = 0x4000
    memory.mc_ptr = 0x5000
    memory.mc_handle = "old_mc_handle" if mapped else 0
    memory.uc_handles = ["old_handle0", "old_handle1"] if mapped else [0, 0]
    memory.comm_backend = comm
    memory._mapped = mapped
    memory._exchanger = None
    memory._graph_visible_addresses = memory.get_graph_visible_addresses()
    memory._get_mem_access_desc = lambda: "access"
    return memory


def test_symm_device_memory_rejects_changed_comm_layout():
    memory = _make_symm_memory(_FakeComm(rank=1))

    with pytest.raises(RuntimeError, match="communicator"):
        memory.detach_physical_keep_va(synchronize=False, barrier=False)


def test_symm_device_memory_rejects_inconsistent_mapped_state():
    memory = _make_symm_memory(_FakeComm(mapped_states=[True, False]))

    with pytest.raises(RuntimeError, match="mapped state"):
        memory.detach_physical_keep_va(synchronize=False, barrier=False)


def test_symm_device_memory_rejects_inconsistent_allocation_metadata():
    metadata = [
        {
            "group_rank": 0,
            "group_size": 2,
            "buf_size": 128,
            "allocation_size": 0x1000,
            "signal_pad_offset": 128,
            "total_uc_size": 0x2000,
            "has_multicast": True,
        },
        {
            "group_rank": 1,
            "group_size": 2,
            "buf_size": 256,
            "allocation_size": 0x1000,
            "signal_pad_offset": 128,
            "total_uc_size": 0x2000,
            "has_multicast": True,
        },
    ]
    memory = _make_symm_memory(_FakeComm(mapped_states=[True, True], metadata=metadata))

    with pytest.raises(RuntimeError, match="metadata"):
        memory.detach_physical_keep_va(synchronize=False, barrier=False)


def test_symm_device_memory_remap_rolls_back_partial_mappings(monkeypatch):
    memory = _make_symm_memory(_FakeComm(mapped_states=[False, False]), mapped=False)
    fresh = type(
        "_FreshSymmMemory",
        (),
        {
            "allocation_size": 0x1000,
            "uc_ptrs": [0xA000, 0xB000],
            "uc_base_ptr": 0xA000,
            "total_uc_size": 0x2000,
            "uc_handles": ["new_handle0", "new_handle1"],
            "mc_ptr": 0xC000,
            "mc_handle": "new_mc_handle",
            "uc_ptrs_dev": 0,
            "_exchanger": None,
        },
    )()
    success = type("_Success", (), {"value": 0})()
    calls = []

    class _FakeCuda:
        def cuCtxSynchronize(self):
            calls.append(("sync",))
            return (success,)

        def cuMemUnmap(self, ptr, size):
            calls.append(("unmap", ptr, size))
            return (success,)

        def cuMemAddressFree(self, ptr, size):
            calls.append(("address_free", ptr, size))
            return (success,)

        def cuMemMap(self, ptr, size, offset, handle, flags):
            calls.append(("map", ptr, handle))
            if ptr == 0x2000:
                raise RuntimeError("map failed")
            return (success,)

        def cuMemSetAccess(self, ptr, size, desc, count):
            calls.append(("access", ptr, size))
            return (success,)

        def cuMemRelease(self, handle):
            calls.append(("release", handle))
            return (success,)

    monkeypatch.setattr(mnnvl_mod, "cuda", _FakeCuda())
    monkeypatch.setattr(mnnvl_mod, "SymmDeviceMemory", lambda **kwargs: fresh)

    with pytest.raises(RuntimeError, match="map failed"):
        memory.remap_physical_same_va(synchronize=False, barrier=False)

    assert ("map", 0x1000, "new_handle0") in calls
    assert ("unmap", 0x1000, 0x1000) in calls
    assert memory.uc_handles == [0, 0]
    assert not memory._mapped


def test_trtllm_allreduce_validates_workspace_tensor_storage():
    workspace = _make_trtllm_workspace()
    workspace.workspace_tensor = workspace.workspace_tensor.clone()

    with pytest.raises(RuntimeError, match="graph-visible"):
        workspace.validate_graph_visible_addresses()


def test_mnnvl_allreduce_validates_buffer_flags_storage():
    workspace = object.__new__(MNNVLAllReduceFusionWorkspace)
    workspace.world_size = 2
    workspace.rank = 0
    workspace.ptrs = [0x1000, 0x2000]
    workspace.uc_ptrs_dev = 0x3000
    workspace.uc_ptr_local = 0x1000
    workspace.mc_ptr = 0x4000
    workspace.buffer_size_bytes = 128
    workspace.workspace_size_bytes = 384
    workspace.buffer_flags = torch.tensor(
        [0, 2, workspace.buffer_size_bytes, 0, 0, 0, 0, 0, 0],
        dtype=torch.uint32,
    )
    workspace.handle = _FakeCheckpointableHandle()
    workspace._graph_visible_addresses = workspace.get_graph_visible_addresses()
    workspace.buffer_flags = workspace.buffer_flags.clone()

    with pytest.raises(RuntimeError, match="graph-visible"):
        workspace.validate_graph_visible_addresses()


def test_mnnvl_allreduce_reset_barrier_uses_fresh_comm(monkeypatch):
    workspace = object.__new__(MNNVLAllReduceFusionWorkspace)
    workspace.world_size = 2
    workspace.rank = 0
    workspace.ptrs = [0x1000, 0x2000]
    workspace.uc_ptrs_dev = 0x3000
    workspace.uc_ptr_local = 0x1000
    workspace.mc_ptr = 0x4000
    workspace.buffer_size_bytes = 128
    workspace.workspace_size_bytes = 384
    workspace.buffer_flags = torch.tensor(
        [0, 2, workspace.buffer_size_bytes, 0, 0, 0, 0, 0, 0],
        dtype=torch.uint32,
    )
    workspace.handle = _FakeCheckpointableHandle()
    old_comm = _FakeComm()
    fresh_comm = _FakeComm()
    workspace.comm_backend = old_comm
    workspace._graph_visible_addresses = workspace.get_graph_visible_addresses()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    workspace.remap_physical_same_va(
        comm_backend=fresh_comm,
        synchronize=False,
        barrier=True,
        reset=True,
    )

    assert workspace.comm_backend is fresh_comm
    assert old_comm.barriers == 0
    assert fresh_comm.barriers == 1


def test_mnnvl_allreduce_reset_respects_barrier_false(monkeypatch):
    workspace = object.__new__(MNNVLAllReduceFusionWorkspace)
    workspace.world_size = 2
    workspace.rank = 0
    workspace.ptrs = [0x1000, 0x2000]
    workspace.uc_ptrs_dev = 0x3000
    workspace.uc_ptr_local = 0x1000
    workspace.mc_ptr = 0x4000
    workspace.buffer_size_bytes = 128
    workspace.workspace_size_bytes = 384
    workspace.buffer_flags = torch.tensor(
        [0, 2, workspace.buffer_size_bytes, 0, 0, 0, 0, 0, 0],
        dtype=torch.uint32,
    )
    workspace.handle = _FakeCheckpointableHandle()
    fresh_comm = _FakeComm()
    workspace.comm_backend = _FakeComm()
    workspace._graph_visible_addresses = workspace.get_graph_visible_addresses()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    workspace.remap_physical_same_va(
        comm_backend=fresh_comm,
        synchronize=False,
        barrier=False,
        reset=True,
    )

    assert fresh_comm.barriers == 0
