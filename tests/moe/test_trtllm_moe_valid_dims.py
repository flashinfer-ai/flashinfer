"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest

from flashinfer.fused_moe.core import _infer_trtllm_moe_output_hidden_size


@pytest.mark.parametrize(
    ("hidden_size", "valid_hidden_size", "expected"),
    [
        (3072, None, 3072),
        (3072, 2880, 2944),
        (512, 64, 128),
    ],
)
def test_infer_trtllm_moe_output_hidden_size(hidden_size, valid_hidden_size, expected):
    assert (
        _infer_trtllm_moe_output_hidden_size(hidden_size, valid_hidden_size) == expected
    )


@pytest.mark.parametrize("valid_hidden_size", [0, -1, 129])
def test_infer_trtllm_moe_output_hidden_size_rejects_invalid(
    valid_hidden_size,
):
    with pytest.raises(ValueError):
        _infer_trtllm_moe_output_hidden_size(128, valid_hidden_size)
