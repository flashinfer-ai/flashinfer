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

import pytest
import torch
import tvm_ffi

from flashinfer.fused_moe.shared.inputs import unpack_trtllm_moe_output


def _ffi_results(tensors):
    return tvm_ffi.Array(
        [None if tensor is None else tvm_ffi.from_dlpack(tensor) for tensor in tensors]
    )


def _assert_views(actual, expected):
    assert len(actual) == len(expected)
    for view, tensor in zip(actual, expected, strict=True):
        assert view.data_ptr() == tensor.data_ptr()
        torch.testing.assert_close(view, tensor)


@pytest.mark.parametrize("return_activation", [False, True])
def test_finalized_output_uses_caller_buffer(return_activation):
    output = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    indices = torch.arange(2, dtype=torch.int32)
    activation = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    slots = [None, indices, activation] if return_activation else [None]
    intermediate = _ffi_results(slots)
    expected = [output, indices, activation] if return_activation else [output]

    for _ in range(2):
        result = unpack_trtllm_moe_output(
            intermediate,
            output,
            do_finalize=True,
            gemm1_lora_delta=activation if return_activation else None,
        )
        assert result[0] is output
        _assert_views(result, expected)


@pytest.mark.parametrize("return_activation", [False, True])
@pytest.mark.parametrize("weights_owner", ["caller", "launcher", "empty_placeholder"])
def test_unfinalized_output_preserves_optional_weight_slot(
    return_activation, weights_owner
):
    output = torch.empty(2, 0)
    gemm2 = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    weights = torch.tensor([[0.25], [0.75]])
    indices = torch.arange(2, dtype=torch.int32)
    activation = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    caller_weights = None
    if weights_owner == "caller":
        caller_weights = weights
    elif weights_owner == "empty_placeholder":
        caller_weights = torch.empty(0)
    slots = [gemm2, None if weights_owner == "caller" else weights, indices]
    expected = [gemm2, weights, indices]
    if return_activation:
        slots.append(activation)
        expected.append(activation)
    intermediate = _ffi_results(slots)

    for _ in range(2):
        result = unpack_trtllm_moe_output(
            intermediate,
            output,
            do_finalize=False,
            gemm1_lora_delta=activation if return_activation else None,
            expert_weights=caller_weights,
        )
        if weights_owner == "caller":
            assert result[1] is weights
        _assert_views(result, expected)
