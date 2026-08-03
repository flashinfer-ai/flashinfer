# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

from flashinfer import conv3d_nvfp4, prepare_nvfp4_conv3d_weight
from tests.test_helpers.conv import (
    SM120_CUDA13_SKIP_REASON,
    is_sm120_cuda13_supported,
)


pytestmark = [
    pytest.mark.solo,
    pytest.mark.skipif(
        os.environ.get("FLASHINFER_TEST_CONV3D_NVFP4_WORKLOAD") != "1",
        reason=(
            "full Wan2.2 decoder shape suite; set "
            "FLASHINFER_TEST_CONV3D_NVFP4_WORKLOAD=1 to run"
        ),
    ),
    pytest.mark.skipif(
        not is_sm120_cuda13_supported(),
        reason=SM120_CUDA13_SKIP_REASON,
    ),
]

_VALUE = 0.125
_PADDING = (0, 1, 1)

# (C, K, D, H, W, calls per decoder invocation)
_WAN22_DECODER_CASES = (
    (512, 512, 6, 176, 320, 100),
    (256, 256, 6, 352, 640, 100),
    (1024, 1024, 4, 88, 160, 120),
    (1024, 512, 6, 176, 320, 20),
    (512, 256, 6, 352, 640, 20),
    (1024, 1024, 3, 44, 80, 210),
    (1024, 1024, 3, 88, 160, 6),
    (512, 512, 3, 176, 320, 5),
    (256, 256, 3, 352, 640, 5),
    (1024, 512, 3, 176, 320, 1),
    (512, 256, 3, 352, 640, 1),
)


@pytest.mark.parametrize(
    ("input_channels", "output_channels", "depth", "height", "width", "calls"),
    _WAN22_DECODER_CASES,
    ids=[
        f"c{c}-k{k}-d{d}-h{h}-w{w}-calls{calls}"
        for c, k, d, h, w, calls in _WAN22_DECODER_CASES
    ],
)
def test_conv3d_nvfp4_wan22_decoder_shape(
    input_channels,
    output_channels,
    depth,
    height,
    width,
    calls,
):
    del calls
    input = torch.full(
        (1, input_channels, depth, height, width),
        _VALUE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.full(
        (output_channels, input_channels, 3, 3, 3),
        _VALUE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    bias = torch.linspace(
        -0.25,
        0.25,
        output_channels,
        device="cuda",
        dtype=torch.bfloat16,
    )
    packed_weight, weight_scale, weight_global_scale = prepare_nvfp4_conv3d_weight(
        weight
    )
    input_global_scale = torch.tensor(
        [448.0 * 6.0 / _VALUE],
        device="cuda",
        dtype=torch.float32,
    )

    output = conv3d_nvfp4(
        input,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        bias,
        padding=_PADDING,
    )
    torch.cuda.synchronize()

    expected_shape = (1, output_channels, depth - 2, height, width)
    assert tuple(output.shape) == expected_shape
    assert output.is_contiguous(memory_format=torch.channels_last_3d)
    assert torch.isfinite(output).all()

    output_channel = output_channels // 2
    center = output[
        0,
        output_channel,
        output.shape[2] // 2,
        output.shape[3] // 2,
        output.shape[4] // 2,
    ]
    corner = output[0, output_channel, 0, 0, 0]
    center_expected = (
        input_channels * 27 * _VALUE * _VALUE + bias[output_channel].float()
    ).to(torch.bfloat16)
    corner_expected = (
        input_channels * 12 * _VALUE * _VALUE + bias[output_channel].float()
    ).to(torch.bfloat16)
    assert torch.equal(center, center_expected)
    assert torch.equal(corner, corner_expected)
