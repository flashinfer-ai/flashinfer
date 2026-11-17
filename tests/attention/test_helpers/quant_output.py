# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference checks for the PrimTS MXFP8 attention output.

The kernels quantize the ``[tokens, Hq * Dv]`` view of O with the trtllm-gen
``MxE4m3`` recipe: per 32-channel block, exponent byte
``max(floor(log2(amax)) + 127 - 8, 0)`` and E4M3 data
``satfinite(x * 2**(127 - exponent))``. Scales are 128x4 swizzled.
"""

from __future__ import annotations

import math

import torch

E4M3_MAX = 448.0


def mxfp8_reference_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D tensor; return ``(float8_e4m3fn data, uint8 exponents)``."""
    rows, cols = x.shape
    blocks = x.float().reshape(rows, cols // 32, 32)
    amax = blocks.abs().amax(-1)
    biased = (amax.view(torch.int32) >> 23) & 0xFF
    exponent = torch.clamp(biased - 8, min=0)
    inverse = ((254 - exponent) << 23).to(torch.int32).view(torch.float32)
    # torch's E4M3 cast does not saturate; the kernels' satfinite cast does.
    scaled = (blocks * inverse[..., None]).clamp(-E4M3_MAX, E4M3_MAX)
    return scaled.reshape(rows, cols).to(torch.float8_e4m3fn), exponent.to(torch.uint8)


def unswizzle_mxfp8_scales(scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Return the ``[rows, cols]`` exponents of a 128x4 swizzled scale buffer."""
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 3) // 4 * 4
    flat = scales.reshape(-1).view(torch.uint8)
    assert flat.numel() == padded_rows * padded_cols
    return (
        flat.view(padded_rows // 128, padded_cols // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(padded_rows, padded_cols)[:rows, :cols]
    )


def swizzled_scale_rows(rows: int, cols: int, device) -> torch.Tensor:
    """Token row of every swizzled scale byte, or -1 for padding."""
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 3) // 4 * 4
    row_ids = torch.full(
        (padded_rows, padded_cols), -1, dtype=torch.int64, device=device
    )
    row_ids[:rows, :cols] = torch.arange(rows, device=device)[:, None]
    return (
        row_ids.view(padded_rows // 128, 4, 32, padded_cols // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(-1)
    )


def written_scale_mask(rows: int, cols: int, device) -> torch.Tensor:
    """Flat mask of the swizzled bytes that hold real (unpadded) scales."""
    return swizzled_scale_rows(rows, cols, device) >= 0


def dequantize_mxfp8(data: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    """Dequantize ``[rows, cols]`` E4M3 data with ``[rows, cols / 32]`` exponents."""
    rows, cols = data.shape
    scale = torch.pow(2.0, exponents.float() - 127.0)
    return (data.float().reshape(rows, cols // 32, 32) * scale[..., None]).reshape(
        rows, cols
    )


def _as_rows(out: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    rows = math.prod(out.shape[:-2])
    cols = out.shape[-2] * out.shape[-1]
    return out.reshape(rows, cols), rows, cols


def assert_mxfp8_output(
    out: torch.Tensor,
    out_scale: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_relative_l2: float,
    min_scale_match: float = 0.95,
) -> None:
    """Check MXFP8 ``out``/``out_scale`` against an FP32 attention oracle.

    The scale of each block must be the exponent of the oracle's block maximum
    up to one binade (block maxima of the kernel and the oracle may straddle a
    power of two), and the dequantized output must stay within the MXFP8
    rounding of the oracle: E4M3 keeps 3 mantissa bits, and the recipe may
    saturate a block's largest value by up to an eighth of the block maximum.
    """
    assert out.dtype == torch.float8_e4m3fn
    data, rows, cols = _as_rows(out)
    exponents = unswizzle_mxfp8_scales(out_scale, rows, cols // 32)
    expected2d = expected.float().reshape(rows, cols)
    _, expected_exponents = mxfp8_reference_quantize(expected2d)
    difference = (exponents.int() - expected_exponents.int()).abs()
    assert int(difference.max()) <= 1
    assert float((difference == 0).float().mean()) >= min_scale_match
    actual = dequantize_mxfp8(data, exponents)
    assert torch.isfinite(actual).all()
    block_max = expected2d.abs().reshape(rows, cols // 32, 32).amax(-1).clamp_min(1e-30)
    block_error = (actual - expected2d).abs().reshape(rows, cols // 32, 32).amax(-1)
    assert float((block_error / block_max).max()) <= 0.125 + 4 * max_relative_l2
    relative_l2 = torch.linalg.vector_norm(
        actual - expected2d
    ) / torch.linalg.vector_norm(expected2d).clamp_min(1e-6)
    assert float(relative_l2) <= max_relative_l2


def assert_mxfp8_matches_quantized(
    out: torch.Tensor,
    out_scale: torch.Tensor,
    carrier_out: torch.Tensor,
    *,
    min_scale_match: float = 0.99,
) -> None:
    """Check MXFP8 output against the same plan's 16-bit output, re-quantized.

    Both runs share every stage before the final store, so the scales and
    codes agree up to the 16-bit rounding of the carrier output: a block's
    exponent may move by one binade only when its maximum rounds across a
    power of two, and same-scale codes differ by at most one E4M3 step.
    """
    data, rows, cols = _as_rows(out)
    reference_data, reference_exponents = mxfp8_reference_quantize(
        carrier_out.float().reshape(rows, cols)
    )
    exponents = unswizzle_mxfp8_scales(out_scale, rows, cols // 32)
    difference = (exponents.int() - reference_exponents.int()).abs()
    assert int(difference.max()) <= 1
    assert float((difference == 0).float().mean()) >= min_scale_match
    same_scale = (difference == 0).repeat_interleave(32, dim=1)
    # E4M3 bit patterns are monotonic in magnitude within one sign.
    code = data.view(torch.uint8).int()
    reference_code = reference_data.view(torch.uint8).int()
    signed = torch.where(code >= 128, 128 - code, code)
    reference_signed = torch.where(
        reference_code >= 128, 128 - reference_code, reference_code
    )
    step = (signed - reference_signed).abs()
    assert int(step[same_scale].max()) <= 1
