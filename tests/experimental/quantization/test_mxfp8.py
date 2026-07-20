# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""quantization.mxfp8: the standalone CuTe row quantizer writes the same
bits as the packaged block-FP8 input-quant path (same underlying kernel via
its opaque custom op), and dequantizes back close to the source.
"""

from __future__ import annotations

import torch

from flashinfer.experimental.sm12x.gemm import block_fp8_linear as bfl
from flashinfer.experimental.sm12x.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
)
from flashinfer.experimental.sm12x.quantization import mxfp8

from ..conftest import require_sm12x


def test_quantize_rows_matches_packaged_path_and_dequantizes() -> None:
    require_sm12x()
    torch.manual_seed(20260715)

    source = (
        torch.randn((5, 256), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()

    reference = bfl.quantize_input(source)
    values = torch.empty_like(reference.values)
    scale_rows = torch.empty_like(reference.scale_rows)
    # The kernel writes only live-row lanes of the swizzled MMA scale tensor;
    # the packaged path pre-fills the 128-row-tile padding with the neutral
    # e8m0 1.0 (biased exponent 127). Match that so the compare stays bitwise.
    scale_mma = torch.full_like(reference.scale_mma.view(torch.uint8), 127).view(
        torch.float8_e8m0fnu
    )

    mxfp8.quantize_rows(source, values, scale_rows, scale_mma)
    torch.cuda.synchronize()

    assert torch.equal(values.view(torch.uint8), reference.values.view(torch.uint8))
    assert torch.equal(
        scale_rows.view(torch.uint8), reference.scale_rows.view(torch.uint8)
    )
    assert torch.equal(
        scale_mma.view(torch.uint8), reference.scale_mma.view(torch.uint8)
    )

    deq = dequantize_mxfp8_rows_torch(values, scale_rows)
    assert bool(torch.isfinite(deq).all().item())
    assert (deq.float() - source.float()).abs().max().item() < 0.05
