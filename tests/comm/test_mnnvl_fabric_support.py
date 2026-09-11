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
from unittest.mock import patch

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
