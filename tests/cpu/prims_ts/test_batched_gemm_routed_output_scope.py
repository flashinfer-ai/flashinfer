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

"""Repeat-check scope for routed plain output with expert padding."""

import pytest
import torch

from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
    _active_route_output,
    _same_launch_output,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("is_swap_ab", [False, True])
def test_routed_repeat_scope_ignores_padding_but_detects_active_changes(is_swap_ab):
    active = torch.tensor([True, False, True, False])
    first = torch.tensor([[1.0, 2.0], [float("nan"), 0.0], [3.0, 4.0], [7.0, 8.0]])
    second = first.clone()
    second[~active] = torch.tensor([[9.0, float("nan")], [11.0, 12.0]])
    # The last row is declared capacity beyond the routed padded extent.
    second = torch.cat([second, torch.tensor([[13.0, 14.0]])])
    if is_swap_ab:
        first, second = first.T, second.T

    def observable(output):
        return _active_route_output(output, active, is_swap_ab=is_swap_ab)

    assert _same_launch_output(observable(first), observable(second))
    second[0, 0] += 1
    assert not _same_launch_output(observable(first), observable(second))
    second[0, 0] = float("nan")
    assert not _same_launch_output(observable(first), observable(second))
    # Selection must retain poison on real routes for the coverage check.
    assert torch.isnan(observable(second)[0, 0])


@pytest.mark.parametrize("is_swap_ab", [False, True])
def test_routed_repeat_scope_rejects_missing_token_storage(is_swap_ab):
    output = torch.zeros((3, 2))
    if is_swap_ab:
        output = output.T
    with pytest.raises(AssertionError, match="token extent"):
        _active_route_output(
            output, torch.tensor([True, False, True, False]), is_swap_ab=is_swap_ab
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("is_swap_ab", [False, True])
def test_routed_repeat_scope_compares_signed_zero_bits(dtype, is_swap_ab):
    first = torch.zeros((2, 2), dtype=dtype)
    second = first.clone()
    active = torch.tensor([True, False])
    if is_swap_ab:
        first, second = first.T, second.T

    def observable(output):
        return _active_route_output(output, active, is_swap_ab=is_swap_ab)

    second[1, 1] = -0.0
    assert _same_launch_output(observable(first), observable(second))
    second[0, 0] = -0.0
    assert not _same_launch_output(observable(first), observable(second))
