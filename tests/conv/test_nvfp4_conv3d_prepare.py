# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from flashinfer import SfLayout, nvfp4_quantize, prepare_nvfp4_conv3d_weight


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0:2] != (12, 0)
    or not torch.version.cuda
    or int(torch.version.cuda.split(".")[0]) < 13,
    reason="SM120 NVFP4 Conv3d requires SM120 and CUDA 13+",
)


def test_prepare_nvfp4_conv3d_weight_matches_canonical_quantizer():
    torch.manual_seed(17)
    weight = torch.randn(
        (128, 128, 3, 3, 3),
        device="cuda",
        dtype=torch.bfloat16,
    )
    global_scale = torch.tensor([173.0], device="cuda", dtype=torch.float32)

    packed, scales, returned_scale = prepare_nvfp4_conv3d_weight(weight, global_scale)
    matrix = weight.permute(0, 2, 3, 4, 1).contiguous().reshape(128, -1)
    expected_packed, expected_scales = nvfp4_quantize(
        matrix,
        global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )

    assert packed.dtype == torch.uint8
    assert packed.shape == (128, 3, 3, 3, 64)
    assert torch.equal(packed.reshape_as(expected_packed), expected_packed)
    assert torch.equal(scales, expected_scales)
    assert returned_scale.data_ptr() == global_scale.data_ptr()


def test_prepare_nvfp4_conv3d_weight_derives_finite_scale():
    weight = torch.zeros(
        (128, 128, 3, 3, 3),
        device="cuda",
        dtype=torch.bfloat16,
    )
    _, _, global_scale = prepare_nvfp4_conv3d_weight(weight)
    assert global_scale.shape == (1,)
    assert global_scale.dtype == torch.float32
    assert torch.isfinite(global_scale).all()
    assert (global_scale > 0).all()


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((127, 128, 3, 3, 3), "output channels"),
        ((128, 127, 3, 3, 3), "input channels"),
        ((128, 128, 1, 3, 3), "3x3x3"),
    ],
)
def test_prepare_nvfp4_conv3d_weight_rejects_unsupported_shapes(shape, message):
    weight = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=message):
        prepare_nvfp4_conv3d_weight(weight)


def test_prepare_nvfp4_conv3d_weight_rejects_wrong_dtype():
    weight = torch.empty((128, 128, 3, 3, 3), device="cuda", dtype=torch.float16)
    with pytest.raises(TypeError, match="bfloat16"):
        prepare_nvfp4_conv3d_weight(weight)
