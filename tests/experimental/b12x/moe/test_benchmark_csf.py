"""Benchmark checkpoint encoding preserves every native scale byte."""

import pytest
import torch

from b12x._lib.quant._csf_prepare import expand_source_scales, validate_source
from benchmarks.experimental.b12x.moe_preparation import encode_csf_32x2_source
from ..conftest import require_b12x
from ..quantization.test_csf_encode import _scales


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("rows,columns", [(64, 3), (384, 18), (1152, 160)])
def test_block_checkpoint_roundtrip(quantization, rows, columns):
    device = require_b12x()
    logical = _scales(
        3, rows, columns, window=16 if quantization == "nvfp4" else 2, seed=71
    ).to(device)
    source = encode_csf_32x2_source(logical, quantization=quantization)
    assert source.encoding == "csf_32x2_v1"
    validate_source(source)
    restored = torch.full_like(logical, 0x7F)
    expand_source_scales(source, restored, swizzled=False)
    torch.testing.assert_close(restored, logical, rtol=0, atol=0)
