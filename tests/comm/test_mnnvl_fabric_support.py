# Copyright (c) 2024 by FlashInfer team.
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
from unittest.mock import Mock, patch

import pytest

from flashinfer.comm import mnnvl


@pytest.mark.parametrize(
    "fabric_uuid,state,supported,expected",
    [
        ("00112233445566778899aabbccddeeff", 3, 1, True),
        ("112233445566778899aabbccddeeff00", 3, 1, True),
        ("00000000000000000000000000000001", 3, 1, True),
        ("00000000000000000000000000000000", 3, 1, False),
        ("00112233445566778899aabbccddeeff", 2, 1, False),
        ("00112233445566778899aabbccddeeff", 3, 0, False),
    ],
)
def test_is_mnnvl_fabric_supported(fabric_uuid, state, supported, expected):
    fabric_info = mnnvl.pynvml.c_nvmlGpuFabricInfoV_t()
    fabric_info.state = state
    fabric_info.clusterUuid[:] = bytes.fromhex(fabric_uuid)
    with (
        patch.object(
            mnnvl.cuda,
            "cuDeviceGetAttribute",
            return_value=(mnnvl.cuda.CUresult.CUDA_SUCCESS, supported),
        ),
        patch.object(mnnvl.pynvml, "nvmlInit") as init,
        patch.object(mnnvl.pynvml, "nvmlShutdown") as shutdown,
        patch.object(mnnvl.pynvml, "nvmlDeviceGetHandleByIndex"),
        patch.object(mnnvl.pynvml, "c_nvmlGpuFabricInfoV_t", return_value=fabric_info),
        patch.object(mnnvl.pynvml, "nvmlDeviceGetGpuFabricInfoV"),
    ):
        assert mnnvl.is_mnnvl_fabric_supported(0) is expected
        if supported:
            init.assert_called_once()
            shutdown.assert_called_once()
        else:
            init.assert_not_called()
            shutdown.assert_not_called()


@pytest.mark.parametrize("entry", ["exchanger", "memory"])
@pytest.mark.parametrize(
    "hosts,statuses,expected",
    [
        (["host-a", "host-a"], [0, 0], True),
        (["host-a", "host-b"], [0, 0], True),
        (["host-a", "host-a"], [800, 800], False),
        (["host-a", "host-a"], [0, 800], False),
        (["host-a", "host-a"], [801, 801], False),
        (["host-a", "host-b"], [0, 800], None),
        (["host-a", "host-b"], [801, 801], None),
    ],
)
def test_fabric_handle_policy(entry, hosts, statuses, expected, monkeypatch):
    monkeypatch.setattr(mnnvl.MnnvlMemory, "_fabric_supported", None)
    comm = Mock()
    comm.allgather.return_value = [
        (host, status, mnnvl.cuda.CUresult(status).name)
        for host, status in zip(hosts, statuses, strict=True)
    ]
    types = mnnvl.cuda.CUmemAllocationHandleType
    with patch.object(
        mnnvl, "_probe_fabric_handle", return_value=mnnvl.cuda.CUresult(statuses[0])
    ) as probe:
        if expected is None:
            with pytest.raises(RuntimeError, match="same IMEX channel"):
                if entry == "exchanger":
                    mnnvl.make_handle_exchanger(comm, 0, 2, 0)
                else:
                    mnnvl.MnnvlMemory._resolve_fabric_support(comm, 0)
            assert mnnvl.MnnvlMemory._fabric_supported is None
        else:
            if entry == "exchanger":
                handle_type = mnnvl.make_handle_exchanger(comm, 0, 2, 0).handle_type
            else:
                assert mnnvl.MnnvlMemory._resolve_fabric_support(comm, 0) is expected
                handle_type = mnnvl.MnnvlMemory.get_allocation_prop(
                    0
                ).requestedHandleTypes
            assert handle_type == (
                types.CU_MEM_HANDLE_TYPE_FABRIC
                if expected
                else types.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            )
        probe.assert_called_once_with(0)
    comm.allgather.assert_called_once()


@pytest.mark.parametrize("local_exception", [False, True])
def test_fabric_policy_reports_unexpected_errors_collectively(local_exception):
    comm = Mock()
    comm.allgather.return_value = [
        ("host-a", None if local_exception else 0, "driver failure"),
        ("host-a", 2, "CUDA_ERROR_OUT_OF_MEMORY"),
    ]
    with (
        patch.object(
            mnnvl,
            "_probe_fabric_handle",
            side_effect=RuntimeError("driver failure") if local_exception else None,
            return_value=mnnvl.cuda.CUresult.CUDA_SUCCESS,
        ),
        pytest.raises(RuntimeError, match="rank 1: CUDA_ERROR_OUT_OF_MEMORY"),
    ):
        mnnvl.make_handle_exchanger(comm, 0, 2, 0)
    comm.allgather.assert_called_once()


@pytest.mark.parametrize(
    "failure,error,released",
    [
        (None, 0, [456, 123]),
        ("cuDeviceGetAttribute", 801, []),
        ("cuMemGetAllocationGranularity", 801, []),
        ("cuMemCreate", 800, []),
        ("cuMemCreate", 2, []),
        ("cuMemExportToShareableHandle", 800, [123]),
        ("cuMemImportFromShareableHandle", 800, [123]),
    ],
)
def test_fabric_probe_releases_handles(failure, error, released, monkeypatch):
    success = mnnvl.cuda.CUresult.CUDA_SUCCESS
    calls = {
        "cuDeviceGetAttribute": (success, 1),
        "cuMemGetAllocationGranularity": (success, 65536),
        "cuMemCreate": (success, 123),
        "cuMemExportToShareableHandle": (success, mnnvl.cuda.CUmemFabricHandle()),
        "cuMemImportFromShareableHandle": (success, 456),
        "cuMemRelease": (success,),
    }
    mocks = {}
    for name, result in calls.items():
        if name == failure:
            result = (mnnvl.cuda.CUresult(error), *result[1:])
        mocks[name] = Mock(return_value=result)
        monkeypatch.setattr(mnnvl.cuda, name, mocks[name])
    assert mnnvl._probe_fabric_handle(3) == mnnvl.cuda.CUresult(error)
    assert [call.args[0] for call in mocks["cuMemRelease"].call_args_list] == released
    if mocks["cuMemCreate"].called:
        size, prop, flags = mocks["cuMemCreate"].call_args.args
        assert size == 65536
        assert prop.requestedHandleTypes == (
            mnnvl.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        )
        assert prop.location.id == 3
        assert flags == 0
    if mocks["cuMemImportFromShareableHandle"].called:
        mocks["cuMemImportFromShareableHandle"].assert_called_once_with(
            calls["cuMemExportToShareableHandle"][1].data,
            mnnvl.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
        )


def test_fabric_probe_skips_allocation_without_device_support():
    with (
        patch.object(
            mnnvl.cuda,
            "cuDeviceGetAttribute",
            return_value=(mnnvl.cuda.CUresult.CUDA_SUCCESS, 0),
        ),
        patch.object(mnnvl.cuda, "cuMemCreate") as create,
    ):
        assert mnnvl._probe_fabric_handle(0) == (
            mnnvl.cuda.CUresult.CUDA_ERROR_NOT_SUPPORTED
        )
        create.assert_not_called()
