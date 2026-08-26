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

import inspect

from flashinfer.fused_moe.backends.prims_ts.fp8_op import (
    _fake_prims_ts_fp8_per_tensor_scale_moe,
    prims_ts_fp8_per_tensor_scale_moe,
)
from flashinfer.fused_moe.core import trtllm_fp8_per_tensor_scale_moe


def test_prims_ts_fp8_positional_contract_matches_trtllm():
    trtllm_params = list(
        inspect.signature(trtllm_fp8_per_tensor_scale_moe).parameters.values()
    )
    prims_ts_params = list(
        inspect.signature(prims_ts_fp8_per_tensor_scale_moe).parameters.values()
    )

    assert [parameter.name for parameter in prims_ts_params[: len(trtllm_params)]] == [
        parameter.name for parameter in trtllm_params
    ]
    assert all(
        parameter.kind == expected.kind
        for parameter, expected in zip(prims_ts_params, trtllm_params)
    )

    backend_params = prims_ts_params[len(trtllm_params) :]
    assert [parameter.name for parameter in backend_params] == [
        "weight_layout",
        "fc1_per_channel_weight_scale",
        "fc2_per_channel_weight_scale",
    ]
    assert all(
        parameter.kind == inspect.Parameter.KEYWORD_ONLY
        for parameter in backend_params
    )
    assert inspect.signature(
        _fake_prims_ts_fp8_per_tensor_scale_moe
    ) == inspect.signature(prims_ts_fp8_per_tensor_scale_moe)
