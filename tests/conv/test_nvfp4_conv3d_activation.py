# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from flashinfer import SfLayout, nvfp4_quantize
from flashinfer.conv.nvfp4 import _quantize_nvfp4_conv3d_activation


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0:2] != (12, 0)
    or not torch.version.cuda
    or int(torch.version.cuda.split(".")[0]) < 13,
    reason="SM120 NVFP4 Conv3d requires SM120 and CUDA 13+",
)


@pytest.mark.parametrize("padding", [(0, 0, 0), (0, 1, 1)])
@pytest.mark.parametrize("tile_variant", range(5))
def test_activation_producer_matches_linear_nvfp4_quantization(padding, tile_variant):
    torch.manual_seed(23)
    channels = 256
    input = torch.randn(
        (1, channels, 3, 7, 11),
        device="cuda",
        dtype=torch.bfloat16,
    )
    global_scale = torch.tensor([151.0], device="cuda", dtype=torch.float32)

    packed, scales = _quantize_nvfp4_conv3d_activation(
        input,
        global_scale,
        padding,
        tile_variant=tile_variant,
    )
    _, pad_height, pad_width = padding
    padded = F.pad(input, (pad_width, pad_width, pad_height, pad_height, 0, 0))
    matrix = padded.permute(0, 2, 3, 4, 1).contiguous().reshape(-1, channels)
    expected_packed, expected_scales = nvfp4_quantize(
        matrix,
        global_scale,
        sfLayout=SfLayout.layout_linear,
        do_shuffle=False,
    )

    assert torch.equal(packed.reshape_as(expected_packed), expected_packed)
    assert torch.equal(scales.reshape_as(expected_scales), expected_scales)


def test_activation_producer_uses_current_stream():
    input = torch.randn(
        (1, 128, 3, 5, 9),
        device="cuda",
        dtype=torch.bfloat16,
    )
    global_scale = torch.tensor([97.0], device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        packed, scales = _quantize_nvfp4_conv3d_activation(
            input,
            global_scale,
            (0, 1, 1),
        )
        event = torch.cuda.Event()
        event.record()
    event.synchronize()
    assert packed.count_nonzero() > 0
    assert scales.count_nonzero() > 0


def test_activation_producer_reuses_output_buffers():
    input = torch.randn(
        (1, 128, 3, 5, 9),
        device="cuda",
        dtype=torch.bfloat16,
    )
    global_scale = torch.tensor([97.0], device="cuda", dtype=torch.float32)
    packed_out = torch.empty(
        (1, 3, 7, 11, 64),
        device="cuda",
        dtype=torch.uint8,
    )
    scale_out = torch.empty(
        (1, 3, 7, 11, 8),
        device="cuda",
        dtype=torch.uint8,
    )
    packed, scales = _quantize_nvfp4_conv3d_activation(
        input,
        global_scale,
        (0, 1, 1),
        packed_out=packed_out,
        scale_out=scale_out,
    )
    assert packed.data_ptr() == packed_out.data_ptr()
    assert scales.data_ptr() == scale_out.data_ptr()
    assert packed.count_nonzero() > 0
    assert scales.count_nonzero() > 0


def test_activation_producer_validation():
    global_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    non_contiguous = torch.empty(
        (1, 3, 5, 7, 128),
        device="cuda",
        dtype=torch.bfloat16,
    ).permute(0, 4, 1, 2, 3)
    with pytest.raises(ValueError, match="contiguous"):
        _quantize_nvfp4_conv3d_activation(
            non_contiguous,
            global_scale,
            (0, 1, 1),
        )
    contiguous = non_contiguous.contiguous()
    with pytest.raises(ValueError, match="padding"):
        _quantize_nvfp4_conv3d_activation(
            contiguous,
            global_scale,
            (1, 1, 1),
        )
    wrong_packed_out = torch.empty(
        (1, 3, 5, 7, 64),
        device="cuda",
        dtype=torch.uint8,
    )
    with pytest.raises(ValueError, match="packed_out must have shape"):
        _quantize_nvfp4_conv3d_activation(
            contiguous,
            global_scale,
            (0, 1, 1),
            packed_out=wrong_packed_out,
        )
