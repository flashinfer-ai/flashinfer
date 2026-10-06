"""Prepared MXFP8 quantization preserves values and both scale storage forms."""
from __future__ import annotations

import pytest
import torch

from b12x._lib.intrinsics import (
    as_grouped_scale_view_mx,
    quantize_grouped_mxfp8_torch,
    swizzle_block_scale,
)
from b12x.preparation import PreparedCall, PreparationSession
from b12x.quantization import mxfp8
from ..conftest import require_b12x


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("scale_storage", (torch.float8_e8m0fnu, torch.uint8))
def test_quantize_rows_matches_oracle_and_dequantizes(dtype, scale_storage):
    require_b12x()
    torch.manual_seed(20260715)
    rows, columns = 5, 256
    source = (torch.randn((rows, columns), device="cuda", dtype=dtype) / 4).contiguous()
    expected_values, expected_mma = quantize_grouped_mxfp8_torch(
        source.unsqueeze(0), torch.tensor([rows], dtype=torch.int32, device=source.device),
    )
    padded_rows, padded_groups = 128, columns // 32
    expected_rows = expected_mma.view(torch.uint8).permute(5, 2, 1, 0, 4, 3).contiguous().view(
        1, padded_rows, padded_groups,
    )[0, :rows].contiguous()
    padded = torch.full((1, padded_rows, padded_groups), 127, dtype=torch.uint8, device=source.device)
    padded[:, :rows].copy_(expected_rows)
    expected_mma = as_grouped_scale_view_mx(swizzle_block_scale(padded), rows, columns)
    values = torch.empty((rows, columns), device=source.device, dtype=torch.float8_e4m3fn)
    scale_rows = torch.empty((rows, columns // 32), device=source.device, dtype=scale_storage)
    storage = torch.full((1, padded_rows, padded_groups), 127, device=source.device, dtype=torch.uint8)
    scale_mma = as_grouped_scale_view_mx(storage, rows, columns)
    if scale_storage == torch.uint8:
        scale_mma = scale_mma.view(torch.uint8)

    query = mxfp8.query_from_call(source, values, scale_rows, scale_mma)
    declaration = mxfp8.plan(query)
    request = declaration.request(
        name="rows",
        prepare_call=lambda state: PreparedCall(run=lambda: state.run(source, values, scale_rows, scale_mma)),
    )
    with PreparationSession(device=source.device) as session:
        session.prepare((request,))
        mxfp8.quantize_rows(source, values, scale_rows, scale_mma, plan=declaration)
        torch.testing.assert_close(values.view(torch.uint8), expected_values[:, :, 0], rtol=0, atol=0)
        torch.testing.assert_close(scale_rows.view(torch.uint8), expected_rows, rtol=0, atol=0)
        torch.testing.assert_close(scale_mma.view(torch.uint8), expected_mma.view(torch.uint8), rtol=0, atol=0)
        dequantized = values.float() * torch.exp2(scale_rows.view(torch.uint8).float() - 127).repeat_interleave(32, dim=1)
        assert (dequantized - source.float()).abs().max().item() < 0.05
