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
import ctypes
from unittest.mock import patch

import pytest

from flashinfer.comm import mnnvl


@pytest.mark.cpu_only
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


class _CharUuidFabricInfo(ctypes.Structure):
    """nvidia-ml-py 12.x fabric info: ``clusterUuid`` is ``c_char * 16`` and
    attribute access UTF-8 decodes bytes, as ``_PrintableStructure`` does."""

    _fields_ = [
        ("version", ctypes.c_uint),
        ("clusterUuid", ctypes.c_char * 16),
        ("status", ctypes.c_uint),
        ("cliqueId", ctypes.c_uint),
        ("state", ctypes.c_ubyte),
        ("healthMask", ctypes.c_uint),
    ]

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        return value.decode() if isinstance(value, bytes) else value


def _char_uuid_fabric_info(fabric_uuid, state):
    fabric_info = _CharUuidFabricInfo()
    fabric_info.state = state
    field = _CharUuidFabricInfo.clusterUuid
    raw = bytes.fromhex(fabric_uuid)
    ctypes.memmove(ctypes.addressof(fabric_info) + field.offset, raw, len(raw))
    return fabric_info


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "fabric_uuid,expected",
    [
        ("e5a1d4c3b2a1908f7e6d5c4b3a291807", True),
        ("00e5a1d4c3b2a1908f7e6d5c4b3a2918", True),
        ("11220033445566778899aabbccddeeff", True),
        ("00000000000000000000000000000000", False),
    ],
)
def test_is_mnnvl_fabric_supported_with_char_uuid_field(fabric_uuid, expected):
    fabric_info = _char_uuid_fabric_info(fabric_uuid, state=3)
    with (
        patch.object(
            mnnvl.cuda,
            "cuDeviceGetAttribute",
            return_value=(mnnvl.cuda.CUresult.CUDA_SUCCESS, 1),
        ),
        patch.object(mnnvl.pynvml, "nvmlInit"),
        patch.object(mnnvl.pynvml, "nvmlShutdown"),
        patch.object(mnnvl.pynvml, "nvmlDeviceGetHandleByIndex"),
        patch.object(mnnvl.pynvml, "c_nvmlGpuFabricInfoV_t", return_value=fabric_info),
        patch.object(mnnvl.pynvml, "nvmlDeviceGetGpuFabricInfoV"),
    ):
        assert mnnvl.is_mnnvl_fabric_supported(0) is expected


@pytest.mark.cpu_only
def test_local_fabric_status_returns_raw_char_uuid():
    fabric_uuid = "e5a1d4c3b2a1908f7e6d5c4b3a291807"
    fabric_info = _char_uuid_fabric_info(fabric_uuid, state=3)
    with (
        patch.object(
            mnnvl.cuda,
            "cuDeviceGetAttribute",
            return_value=(mnnvl.cuda.CUresult.CUDA_SUCCESS, 1),
        ),
        patch.object(mnnvl.pynvml, "nvmlInit"),
        patch.object(mnnvl.pynvml, "nvmlShutdown"),
        patch.object(mnnvl.pynvml, "nvmlDeviceGetHandleByIndex"),
        patch.object(mnnvl.pynvml, "c_nvmlGpuFabricInfoV_t", return_value=fabric_info),
        patch.object(mnnvl.pynvml, "nvmlDeviceGetGpuFabricInfoV"),
    ):
        assert mnnvl._local_fabric_status(0) == (None, bytes.fromhex(fabric_uuid))
