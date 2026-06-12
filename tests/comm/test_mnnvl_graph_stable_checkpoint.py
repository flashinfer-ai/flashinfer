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

from types import SimpleNamespace

import pytest

from flashinfer.comm import mnnvl as mnnvl_mod
from flashinfer.comm import trtllm_moe_alltoall as moe_a2a_mod
from flashinfer.comm.mnnvl import MnnvlConfig, MnnvlMemory, _MnnvlAllocationRecord
from flashinfer.comm.trtllm_moe_alltoall import MoeAlltoAll


class _FakeComm:
    def __init__(self, size: int = 2, rank: int = 0, values=None):
        self.size = size
        self.rank = rank
        self.barriers = 0
        self.values = values

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return self.rank

    def allgather(self, value):
        if self.values is not None:
            return self.values
        return [value] * self.size

    def barrier(self):
        self.barriers += 1


class _FakeTensor:
    def __init__(
        self,
        data_ptr: int,
        element_size: int,
        size: tuple[int, int],
        stride: tuple[int, int],
    ):
        self._data_ptr = data_ptr
        self._element_size = element_size
        self._size = size
        self._stride = stride

    def data_ptr(self):
        return self._data_ptr

    def element_size(self):
        return self._element_size

    def size(self, dim: int):
        return self._size[dim]

    def stride(self, dim: int):
        return self._stride[dim]


@pytest.fixture
def fake_mnnvl_state():
    old_comm = MnnvlMemory.comm
    old_allocated_map = MnnvlMemory.allocated_map
    old_address_refcnt = MnnvlMemory.address_refcnt
    try:
        fake_comm = _FakeComm()
        MnnvlMemory.comm = fake_comm
        MnnvlMemory.allocated_map = {
            0x1000: _MnnvlAllocationRecord(
                mapping=object(),
                comm=fake_comm,
                comm_size=2,
                comm_rank=0,
                aligned_size=0x1000,
                mem_handles=["handle0", "handle1"],
                start_address=0x1000,
                rank_stride=0x4000,
                address_offset=0,
            )
        }
        MnnvlMemory.address_refcnt = {0x1000: 1}
        yield fake_comm
    finally:
        MnnvlMemory.comm = old_comm
        MnnvlMemory.allocated_map = old_allocated_map
        MnnvlMemory.address_refcnt = old_address_refcnt


def _make_memory():
    mem = object.__new__(MnnvlMemory)
    mem.ptr = 0x1000
    mem.segment_size = 0x400
    mem.rank_stride = 0x4000
    return mem


@pytest.mark.usefixtures("fake_mnnvl_state")
def test_graph_visible_address_validation():
    mem = _make_memory()
    expected = mem.get_graph_visible_addresses()
    tensor = _FakeTensor(
        data_ptr=0x1000,
        element_size=4,
        size=(2, 0x100),
        stride=(0x1000, 1),
    )

    mem.validate_graph_visible_addresses(expected, tensor)

    changed_expected = {**expected, "rank_stride": 0x8000}
    with pytest.raises(RuntimeError, match="rank_stride"):
        mem.validate_graph_visible_addresses(changed_expected, tensor)

    changed_tensor = _FakeTensor(
        data_ptr=0x2000,
        element_size=4,
        size=(2, 0x100),
        stride=(0x1000, 1),
    )
    with pytest.raises(RuntimeError, match="data_ptr"):
        mem.validate_graph_visible_addresses(expected, changed_tensor)

    changed_rank = {**expected, "comm_rank": 1}
    with pytest.raises(RuntimeError, match="comm_rank"):
        mem.validate_graph_visible_addresses(changed_rank, tensor)


def test_detach_and_remap_preserve_va_metadata(monkeypatch, fake_mnnvl_state):
    calls = []

    success = SimpleNamespace(value=0)

    def _record(name, *args):
        calls.append((name, *args))
        return (success,)

    fake_cuda = SimpleNamespace(
        cuCtxSynchronize=lambda: _record("sync"),
        cuMemUnmap=lambda ptr, size: _record("unmap", ptr, size),
        cuMemRelease=lambda handle: _record("release", handle),
    )
    monkeypatch.setattr(mnnvl_mod, "cuda", fake_cuda)

    MnnvlMemory.detach_mnnvl_memory_keep_va(0x1000)

    record = MnnvlMemory.allocated_map[0x1000]
    assert not record.mapped
    assert record.mem_handles == [None, None]
    assert ("unmap", 0x1000, 0x1000) in calls
    assert ("unmap", 0x5000, 0x1000) in calls
    assert not any(call[0] == "address_free" for call in calls)
    assert fake_mnnvl_state.barriers == 2

    remap_args = None

    def _remap(*args, **kwargs):
        nonlocal remap_args
        remap_args = (args, kwargs)
        return ["new_handle0", "new_handle1"]

    monkeypatch.setattr(MnnvlMemory, "_create_and_map_mnnvl_handles", _remap)
    MnnvlMemory.remap_mnnvl_memory_same_va(0x1000)

    record = MnnvlMemory.allocated_map[0x1000]
    assert record.mapped
    assert record.mem_handles == ["new_handle0", "new_handle1"]
    assert remap_args == (
        (record.mapping, 0x1000, 0x1000, 0x4000, 0),
        {"comm": fake_mnnvl_state, "zero_local": True},
    )


def test_remap_refreshes_comm_from_current_config(monkeypatch, fake_mnnvl_state):
    record = MnnvlMemory.allocated_map[0x1000]
    record.mapped = False
    record.mem_handles = [None, None]

    new_comm = _FakeComm(size=2, rank=0)
    config = MnnvlConfig(comm_backend=SimpleNamespace())

    def _refresh(mapping, refresh_config):
        assert mapping is record.mapping
        assert refresh_config is config
        return new_comm

    remap_args = None

    def _remap(*args, **kwargs):
        nonlocal remap_args
        remap_args = (args, kwargs)
        return ["new_handle0", "new_handle1"]

    success = SimpleNamespace(value=0)
    fake_cuda = SimpleNamespace(cuCtxSynchronize=lambda: (success,))
    monkeypatch.setattr(mnnvl_mod, "cuda", fake_cuda)
    monkeypatch.setattr(MnnvlMemory, "refresh_comm_from_config", _refresh)
    monkeypatch.setattr(MnnvlMemory, "_create_and_map_mnnvl_handles", _remap)

    MnnvlMemory.remap_mnnvl_memory_same_va(0x1000, config=config)

    assert record.comm is new_comm
    assert record.mapped
    assert record.mem_handles == ["new_handle0", "new_handle1"]
    assert remap_args == (
        (record.mapping, 0x1000, 0x1000, 0x4000, 0),
        {"comm": new_comm, "zero_local": True},
    )


def test_remap_rejects_changed_comm_before_mapping(monkeypatch, fake_mnnvl_state):
    record = MnnvlMemory.allocated_map[0x1000]
    record.mapped = False
    record.mem_handles = [None, None]
    config = MnnvlConfig(comm_backend=SimpleNamespace())

    def _refresh(mapping, refresh_config):
        return _FakeComm(size=3, rank=0)

    def _remap(*args, **kwargs):
        raise AssertionError("remap should not run with changed communicator")

    monkeypatch.setattr(MnnvlMemory, "refresh_comm_from_config", _refresh)
    monkeypatch.setattr(MnnvlMemory, "_create_and_map_mnnvl_handles", _remap)

    with pytest.raises(RuntimeError, match="does not match"):
        MnnvlMemory.remap_mnnvl_memory_same_va(0x1000, config=config)


def test_remap_config_requires_detached_allocation(fake_mnnvl_state):
    config = MnnvlConfig(comm_backend=SimpleNamespace())

    with pytest.raises(RuntimeError, match="still mapped"):
        MnnvlMemory.remap_mnnvl_memory_same_va(0x1000, config=config)


def test_posix_handle_exchange_closes_exported_and_pidfds(monkeypatch):
    fake_comm = _FakeComm(values=[10, 11])
    closed = []

    class _FakeCuda:
        class CUmemAllocationHandleType:
            CU_MEM_HANDLE_TYPE_FABRIC = "fabric"
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = "posix"

        def cuMemExportToShareableHandle(
            self, allocated_mem_handle, handle_type, flags
        ):
            posix_fd = (
                self.CUmemAllocationHandleType
                .CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            )
            assert handle_type == posix_fd
            return (SimpleNamespace(value=0), 7)

    syscall_results = iter([100, 200, 101, 201])

    class _FakeLibc:
        def syscall(self, number, *args):
            return next(syscall_results)

    allocation_prop = SimpleNamespace(
        requestedHandleTypes=(
            _FakeCuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
    )

    monkeypatch.setattr(mnnvl_mod, "cuda", _FakeCuda())
    monkeypatch.setattr(mnnvl_mod.ctypes, "CDLL", lambda *args, **kwargs: _FakeLibc())
    monkeypatch.setattr(mnnvl_mod.os, "close", lambda fd: closed.append(fd))

    remote_fds = MnnvlMemory._exchange_shareable_handles(
        fake_comm, allocation_prop, "handle"
    )

    assert remote_fds == [200, 201]
    assert closed == [100, 101, 7]
    assert fake_comm.barriers == 1


def test_create_and_map_rolls_back_unconsumed_fds_on_import_failure(monkeypatch):
    fake_comm = _FakeComm(size=3, rank=0)
    success = SimpleNamespace(value=0)
    calls = []
    closed = []

    class _FakeCuda:
        class CUmemAllocationHandleType:
            CU_MEM_HANDLE_TYPE_FABRIC = "fabric"
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = "posix"

        class CUmemAccess_flags:
            CU_MEM_ACCESS_FLAGS_PROT_READWRITE = "rw"

        class CUmemAccessDesc:
            pass

        def cuCtxGetDevice(self):
            return (success, 0)

        def cuMemCreate(self, size, allocation_prop, flags):
            calls.append(("create", size))
            return (success, "local_handle")

        def cuMemImportFromShareableHandle(self, fd, handle_type):
            calls.append(("import", fd, handle_type))
            raise RuntimeError("import failed")

        def cuMemMap(self, ptr, size, offset, handle, flags):
            calls.append(("map", ptr, handle))
            return (success,)

        def cuMemSetAccess(self, ptr, size, desc, count):
            calls.append(("access", ptr))
            return (success,)

        def cuMemUnmap(self, ptr, size):
            calls.append(("unmap", ptr, size))
            return (success,)

        def cuMemRelease(self, handle):
            calls.append(("release", handle))
            return (success,)

    monkeypatch.setattr(mnnvl_mod, "cuda", _FakeCuda())
    monkeypatch.setattr(MnnvlMemory, "dev_id", 0)
    monkeypatch.setattr(
        MnnvlMemory,
        "get_allocation_prop",
        lambda dev_id: SimpleNamespace(
            requestedHandleTypes="posix",
            location="device0",
        ),
    )
    monkeypatch.setattr(
        MnnvlMemory,
        "_exchange_shareable_handles",
        lambda *args: [10, 11, 12],
    )
    monkeypatch.setattr(mnnvl_mod.os, "close", lambda fd: closed.append(fd))

    with pytest.raises(RuntimeError, match="import failed"):
        MnnvlMemory._create_and_map_mnnvl_handles(
            object(),
            0x1000,
            0x1000,
            0x4000,
            0,
            comm=fake_comm,
        )

    assert closed == [10, 11, 12]
    assert ("unmap", 0x1000, 0x1000) in calls
    assert ("release", "local_handle") in calls


def test_create_and_map_rolls_back_partial_mappings_on_access_failure(monkeypatch):
    fake_comm = _FakeComm(size=2, rank=0)
    success = SimpleNamespace(value=0)
    calls = []
    closed = []

    class _FakeCuda:
        class CUmemAllocationHandleType:
            CU_MEM_HANDLE_TYPE_FABRIC = "fabric"
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = "posix"

        class CUmemAccess_flags:
            CU_MEM_ACCESS_FLAGS_PROT_READWRITE = "rw"

        class CUmemAccessDesc:
            pass

        def cuCtxGetDevice(self):
            return (success, 0)

        def cuMemCreate(self, size, allocation_prop, flags):
            calls.append(("create", size))
            return (success, "local_handle")

        def cuMemImportFromShareableHandle(self, fd, handle_type):
            calls.append(("import", fd, handle_type))
            return (success, f"remote_handle_{fd}")

        def cuMemMap(self, ptr, size, offset, handle, flags):
            calls.append(("map", ptr, handle))
            return (success,)

        def cuMemSetAccess(self, ptr, size, desc, count):
            calls.append(("access", ptr))
            if ptr == 0x5000:
                raise RuntimeError("set access failed")
            return (success,)

        def cuMemUnmap(self, ptr, size):
            calls.append(("unmap", ptr, size))
            return (success,)

        def cuMemRelease(self, handle):
            calls.append(("release", handle))
            return (success,)

    monkeypatch.setattr(mnnvl_mod, "cuda", _FakeCuda())
    monkeypatch.setattr(MnnvlMemory, "dev_id", 0)
    monkeypatch.setattr(
        MnnvlMemory,
        "get_allocation_prop",
        lambda dev_id: SimpleNamespace(
            requestedHandleTypes="posix",
            location="device0",
        ),
    )
    monkeypatch.setattr(
        MnnvlMemory,
        "_exchange_shareable_handles",
        lambda *args: [10, 11],
    )
    monkeypatch.setattr(mnnvl_mod.os, "close", lambda fd: closed.append(fd))

    with pytest.raises(RuntimeError, match="set access failed"):
        MnnvlMemory._create_and_map_mnnvl_handles(
            object(),
            0x1000,
            0x1000,
            0x4000,
            0,
            comm=fake_comm,
        )

    assert closed == [10, 11]
    assert ("unmap", 0x5000, 0x1000) in calls
    assert ("unmap", 0x1000, 0x1000) in calls
    assert ("release", "remote_handle_11") in calls
    assert ("release", "local_handle") in calls


def test_moe_alltoall_remap_uses_allocation_record_comm(monkeypatch):
    class _FakeMnnvlMemory:
        ptr = 0x1000
        mapping = object()

        def __init__(self):
            self.remap_kwargs = None

        def remap_physical_same_va(self, **kwargs):
            self.remap_kwargs = kwargs

        def validate_graph_visible_addresses(self, expected, tensor):
            assert expected == "graph"
            assert tensor == "workspace"

    record_comm = _FakeComm()
    fake_mnnvl_mem = _FakeMnnvlMemory()
    moe = object.__new__(MoeAlltoAll)
    moe.mnnvl_config = "config"
    moe.mnnvl_mem = fake_mnnvl_mem
    moe.workspace = "workspace"
    moe.metainfo = "metainfo"
    moe.ep_rank = 0
    moe.ep_size = 2
    moe.max_num_tokens = 8
    moe._WORKSPACE = {"graph_visible_addresses": "graph"}

    monkeypatch.setattr(
        MnnvlMemory,
        "allocated_map",
        {fake_mnnvl_mem.ptr: SimpleNamespace(comm=record_comm)},
    )
    monkeypatch.setattr(
        MnnvlMemory,
        "get_comm",
        lambda mapping: (_ for _ in ()).throw(AssertionError("used global comm")),
    )
    monkeypatch.setattr(
        moe_a2a_mod,
        "moe_a2a_initialize",
        lambda workspace, ep_rank, ep_size, max_num_tokens: "new_metainfo",
    )
    monkeypatch.setattr(moe_a2a_mod.torch, "equal", lambda lhs, rhs: True)

    moe.remap_physical_same_va(synchronize=False)

    assert fake_mnnvl_mem.remap_kwargs == {
        "config": "config",
        "synchronize": False,
        "barrier": True,
        "zero_local": False,
    }
    assert record_comm.barriers == 1
