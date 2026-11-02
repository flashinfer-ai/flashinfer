import math

import pytest
import torch

from flashinfer.quantization.fp4_quantization import (
    _e2m1_and_ufp8sf_scale_to_float_cpu,
    e2m1_and_ufp8sf_scale_to_float,
)
from flashinfer.utils import get_compute_capability


def _finite_e4m3(byte):
    # Decode arithmetically, independently of torch's float8 conversion.
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    magnitude = (
        math.ldexp(mantissa, -9)
        if exponent == 0
        else math.ldexp(8 + mantissa, exponent - 10)
    )
    return math.copysign(magnitude, -1.0 if byte & 128 else 1.0)


def _finite_pairs(global_scale):
    # E4M3FN has 254 finite encodings; 0x7f and 0xff are NaNs and excluded.
    finite = [byte for byte in range(256) if byte not in (0x7F, 0xFF)]
    packed = torch.arange(256, dtype=torch.uint8).repeat(254, 1)
    # Rotate scales across 16-element blocks so a row has distinct scales,
    # while every packed byte still sees every finite scale exactly once.
    scales = torch.tensor(
        [[finite[(row + block) % 254] for block in range(32)] for row in range(254)],
        dtype=torch.uint8,
    )
    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    expected = []
    for row in range(254):
        values = []
        for column in range(512):
            byte = column // 2
            code = byte & 15 if column % 2 == 0 else byte >> 4
            value = math.copysign(magnitudes[code & 7], -1.0 if code & 8 else 1.0)
            scale = _finite_e4m3(finite[(row + column // 16) % 254])
            values.append(value * scale * global_scale)
        expected.append(values)
    return packed, scales, torch.tensor(expected, dtype=torch.float32)


@pytest.mark.parametrize("global_scale", [None, 1.0, -2.0])
def test_nvfp4_dequantize_fallback_finite_pairs(global_scale):
    packed, scales, expected = _finite_pairs(
        1.0 if global_scale is None else global_scale
    )
    actual = _e2m1_and_ufp8sf_scale_to_float_cpu(
        packed,
        scales,
        None if global_scale is None else torch.tensor([global_scale]),
        sf_vec_size=16,
        ufp8_type=1,
        is_sf_swizzled_layout=False,
    )
    # Integer views distinguish +0 from -0; all finite products here are exact.
    torch.testing.assert_close(
        actual.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0
    )


@pytest.mark.parametrize("layout", ["linear", "128x4", "8x4"])
@pytest.mark.parametrize("global_scale", [1.0, -2.0])
def test_nvfp4_dequantize_native_finite_pairs(layout, global_scale):
    # This is a host dequantizer, but its module is built with CUDA and selected
    # using the GPU architecture. It does not execute a GPU dequantization kernel.
    if not torch.cuda.is_available():
        pytest.skip("Native host dequantizer requires a CUDA build environment")
    major, minor = get_compute_capability(torch.device("cuda:0"))
    if major * 10 + minor not in (90, 100, 103, 107, 110, 120, 121):
        pytest.skip("No native FP4 quantization module for this architecture")

    packed, scales, expected = _finite_pairs(global_scale)
    if layout != "linear":
        # Both layouts pad the 254 logical rows to 256. Poison padding with
        # NaNs so an accidental read cannot silently look like a zero scale.
        padded = torch.full((256, 32), 0x7F, dtype=torch.uint8)
        padded[:254] = scales
        if layout == "128x4":
            scales = padded.reshape(2, 4, 32, 8, 4).permute(0, 3, 2, 1, 4)
        else:
            scales = padded.reshape(32, 8, 8, 4).permute(0, 2, 1, 3)
        scales = scales.contiguous().flatten()

    actual = e2m1_and_ufp8sf_scale_to_float(
        packed,
        scales,
        torch.tensor([global_scale]),
        sf_vec_size=16,
        ufp8_type=1,
        is_sf_swizzled_layout=layout != "linear",
        is_sf_8x4_layout=layout == "8x4",
    )
    torch.testing.assert_close(
        actual.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0
    )
