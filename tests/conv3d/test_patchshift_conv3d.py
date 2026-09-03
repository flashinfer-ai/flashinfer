"""
Copyright (c) 2026 by the PatchShift Conv3d contributors.

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
import torch
import torch.nn.functional as F

from flashinfer.conv3d import (
    pack_patchshift_conv3d_weight,
    patchshift_conv3d,
    prepare_patchshift_conv3d,
)


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="requires SM100a/B200")


def _reference(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.conv3d(input.permute(0, 4, 1, 2, 3), weight, padding=1).permute(
        0, 2, 3, 4, 1
    )


@pytest.mark.parametrize(
    "shape,out_channels",
    [
        ((1, 1, 8, 8, 8), 32),
        ((1, 1, 16, 30, 32), 64),
        ((1, 2, 17, 31, 64), 96),
        ((1, 4, 16, 30, 96), 128),
        ((1, 2, 17, 31, 128), 160),
    ],
)
def test_patchshift_conv3d_matches_torch(shape, out_channels):
    torch.manual_seed(0)
    input = torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 0.125
    weight = (
        torch.randn(
            out_channels, shape[-1], 3, 3, 3, dtype=torch.bfloat16, device="cuda"
        )
        * 0.0625
    )
    packed_weight = pack_patchshift_conv3d_weight(weight)
    workspace = prepare_patchshift_conv3d(input, packed_weight, out_channels)

    actual = patchshift_conv3d(input, packed_weight, workspace, out_channels)
    expected = _reference(input, weight)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_patchshift_conv3d_out_and_cuda_graph():
    torch.manual_seed(1)
    shape = (1, 2, 17, 31, 64)
    out_channels = 96
    input = torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 0.125
    weight = (
        torch.randn(
            out_channels, shape[-1], 3, 3, 3, dtype=torch.bfloat16, device="cuda"
        )
        * 0.0625
    )
    packed_weight = pack_patchshift_conv3d_weight(weight)
    workspace = prepare_patchshift_conv3d(input, packed_weight, out_channels)
    out = torch.empty((*shape[:-1], out_channels), dtype=input.dtype, device="cuda")

    patchshift_conv3d(input, packed_weight, workspace, out_channels, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = patchshift_conv3d(
            input, packed_weight, workspace, out_channels, out=out
        )
    graph.replay()
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, _reference(input, weight), rtol=2e-2, atol=2e-2)


def test_patchshift_conv3d_rejects_invalid_input_channels():
    input = torch.empty((1, 1, 2, 2, 7), dtype=torch.bfloat16, device="cuda")
    packed_weight = torch.empty(1, dtype=torch.bfloat16, device="cuda")
    workspace = torch.empty(1664, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="divisible by 8"):
        patchshift_conv3d(input, packed_weight, workspace, 32)


def test_patchshift_conv3d_rejects_input_output_alias():
    shape = (1, 1, 8, 8, 8)
    input = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((8, 8, 3, 3, 3), dtype=torch.bfloat16, device="cuda")
    packed_weight = pack_patchshift_conv3d_weight(weight)
    workspace = prepare_patchshift_conv3d(input, packed_weight, 8)

    with pytest.raises(ValueError, match="must not alias"):
        patchshift_conv3d(input, packed_weight, workspace, 8, out=input)
