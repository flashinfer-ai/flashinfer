"""Online CSF encoding reconstructs native FP4 block-scale bytes exactly."""

import numpy as np
import pytest
import torch

from b12x._lib.quant._csf_prepare import expand_source_scales, validate_source
from b12x._lib.quant.csf_encode import encode_csf_source
from ..conftest import require_b12x


def _scales(experts, rows, columns, *, window, seed, exceptions=0.02):
    """Row-local windows with sparse exceptions, one dense row, and extreme bytes."""
    rng = np.random.default_rng(seed)
    logical = rng.integers(0, 257 - window, (experts, rows, 1)) + rng.integers(
        0, window, (experts, rows, columns)
    )
    outside = rng.random((experts, rows, columns)) < exceptions
    logical[outside] = rng.integers(0, 256, int(outside.sum()))
    logical[:, 0] = rng.integers(0, 256, (experts, columns))
    logical[:, 1] = 255
    logical[:, 2] = 0
    logical[:, 3, ::2] = 0
    logical[:, 3, 1::2] = 255
    return torch.from_numpy(logical.astype(np.uint8))


@pytest.mark.parametrize(
    "quantization,experts,rows,columns",
    [
        ("nvfp4", 3, 128, 16),
        ("nvfp4", 2, 256, 20),
        ("nvfp4", 2, 384, 160),
        ("nvfp4", 1, 2560, 40),
        ("mxfp4", 3, 384, 112),
        ("mxfp4", 2, 3584, 6),
        ("mxfp4", 2, 64, 3),
        ("mxfp4", 2, 1152, 18),
    ],
)
def test_encoded_scales_expand_to_the_native_bytes(
    quantization, experts, rows, columns
):
    device = require_b12x()
    window = 16 if quantization == "nvfp4" else 2
    logical = _scales(experts, rows, columns, window=window, seed=7 * rows + columns)
    logical = logical.to(device)
    source = encode_csf_source(logical, quantization=quantization)
    validate_source(source)
    assert (source.encoding, source.quantization) == ("csf", quantization)
    assert (source.num_experts, source.rows, source.columns) == (experts, rows, columns)
    output = torch.full_like(logical, 0x7F)
    expand_source_scales(source, output, swizzled=False)
    torch.testing.assert_close(output, logical, rtol=0, atol=0)


@pytest.mark.parametrize("quantization,columns", [("nvfp4", 160), ("mxfp4", 112)])
def test_windowed_scales_compress_and_choose_the_most_covering_base(
    quantization, columns
):
    device = require_b12x()
    window = 16 if quantization == "nvfp4" else 2
    logical = _scales(4, 256, columns, window=window, seed=11, exceptions=0).to(device)
    logical[:, 4] = 200
    logical[:, 4, :3] = 10
    source = encode_csf_source(logical, quantization=quantization)
    fixed, exceptions = source.components
    # Rows 0 and 3 are the only rows with bytes outside one window, apart from row 4's three.
    assert fixed.numel() + 4 * exceptions.numel() < logical.numel()
    assert 0 < exceptions.numel() <= 4 * (2 * columns + 3)
    # The lowest base whose window still covers row 4's repeated byte.
    assert torch.all(fixed[:, 0, 4] == 200 - (window - 1))
    output = torch.empty_like(logical)
    expand_source_scales(source, output, swizzled=False)
    torch.testing.assert_close(output, logical, rtol=0, atol=0)


def test_encoding_rejects_unsupported_inputs():
    device = require_b12x()
    scales = torch.zeros(2, 64, 8, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="nvfp4 or mxfp4"):
        encode_csf_source(scales, quantization="fp8")
    for bad in (scales.cpu(), scales.float(), scales[:, :, ::2], scales[0]):
        with pytest.raises(ValueError, match="contiguous CUDA uint8"):
            encode_csf_source(bad, quantization="mxfp4")
    with pytest.raises(ValueError, match="32-row blocks"):
        encode_csf_source(scales[:, :48].contiguous(), quantization="mxfp4")
