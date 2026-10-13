# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Standalone quantizers are JIT-loaded without constructing a transport."""

import os

import pytest
import torch

from .reference import decode, quantize_fp8_reference, quantize_nvfp4_reference


@pytest.fixture
def quant_device():
    if not torch.cuda.is_available():
        pytest.skip("Requires an SM120 CUDA device")
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("Requires an SM120 CUDA device")
    return torch.device("cuda")


@pytest.mark.parametrize("hidden", (4096, 6144, 7168))
def test_exact_nvfp4_codes_and_roundtrip(quant_device, hidden):
    from flashinfer.comm import RdmaMoeAlltoAll

    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.bfloat16,
        device=quant_device,
    )
    x = values.repeat(hidden // 8)[None].contiguous()
    scales = torch.full((1, 1), 256.0, dtype=torch.float32, device=quant_device)
    actual_packed, actual_blocks = RdmaMoeAlltoAll.quantize_bf16_to_nvfp4(x, scales)
    expected_packed, expected_blocks = quantize_nvfp4_reference(x, scales)
    torch.testing.assert_close(actual_packed, expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(actual_blocks, expected_blocks, rtol=0, atol=0)
    restored = RdmaMoeAlltoAll.dequantize_nvfp4_to_bf16(
        actual_packed, scales, actual_blocks
    )
    torch.testing.assert_close(restored, x, rtol=0, atol=0)


@pytest.mark.parametrize("hidden", (4096, 6144, 7168))
def test_exact_fp8_codes_and_roundtrip(quant_device, hidden):
    from flashinfer.comm import RdmaMoeAlltoAll

    values = torch.tensor(
        [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 224.0, 448.0],
        dtype=torch.bfloat16,
        device=quant_device,
    )
    x = values.repeat(hidden // 8)[None].contiguous()
    packed, blocks = RdmaMoeAlltoAll.quantize_bf16_to_fp8(x)
    expected_packed, expected_blocks = quantize_fp8_reference(x)
    torch.testing.assert_close(packed, expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(blocks, expected_blocks, rtol=0, atol=0)
    restored = RdmaMoeAlltoAll.dequantize_fp8_to_bf16(packed, blocks)
    torch.testing.assert_close(
        restored, decode(packed, blocks, "fp8").to(torch.bfloat16), rtol=0, atol=0
    )


@pytest.mark.parametrize("hidden", (4096, 6144, 7168))
@pytest.mark.parametrize("rows", (0, 1, 3))
@pytest.mark.parametrize("global_scale", (1.0, 128.0, 256.0))
def test_nvfp4_scale_and_empty_rows_match_reference(
    quant_device, hidden, rows, global_scale
):
    from flashinfer.comm import RdmaMoeAlltoAll

    # Exact block scales and values away from E2M1 midpoints make the
    # independent reference insensitive to the device reciprocal approximation.
    values = torch.arange(rows * hidden, device=quant_device).reshape(rows, hidden)
    x = ((((values % 95) * 2 - 95).float()) / 32).to(torch.bfloat16)
    x.reshape(rows, hidden // 16, 16)[..., 0] = 3.0
    if rows:
        x[0].zero_()
    row_scales = torch.full(
        (rows, 1), global_scale, dtype=torch.float32, device=quant_device
    )
    expected, expected_blocks = quantize_nvfp4_reference(x, row_scales)
    packed, blocks = RdmaMoeAlltoAll.quantize_bf16_to_nvfp4(x, row_scales)
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    torch.testing.assert_close(blocks, expected_blocks, rtol=0, atol=0)


@pytest.mark.parametrize("precision", ("fp8", "nvfp4"))
def test_misaligned_payloads_are_rejected_before_launch(quant_device, precision):
    from flashinfer.comm import RdmaMoeAlltoAll

    hidden = 4096
    # Contiguous views can still start at an address unsuitable for int4 loads.
    x = torch.zeros(hidden + 1, dtype=torch.bfloat16, device=quant_device)[1:].view(
        1, hidden
    )
    global_scale = torch.full((1, 1), 256.0, device=quant_device)
    with pytest.raises(ValueError, match="16-byte aligned"):
        if precision == "fp8":
            RdmaMoeAlltoAll.quantize_bf16_to_fp8(x)
        else:
            RdmaMoeAlltoAll.quantize_bf16_to_nvfp4(x, global_scale)
    width = hidden if precision == "fp8" else hidden // 2
    packed = torch.zeros(width + 1, dtype=torch.uint8, device=quant_device)[1:].view(
        1, width
    )
    scales = torch.ones(
        (1, hidden // (128 if precision == "fp8" else 16)),
        dtype=torch.float32 if precision == "fp8" else torch.uint8,
        device=quant_device,
    )
    with pytest.raises(ValueError, match="16-byte aligned"):
        if precision == "fp8":
            RdmaMoeAlltoAll.dequantize_fp8_to_bf16(packed, scales)
        else:
            RdmaMoeAlltoAll.dequantize_nvfp4_to_bf16(packed, global_scale, scales)
