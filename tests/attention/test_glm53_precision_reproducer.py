# Copyright (c) 2026 Francesco Parisio
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the independent packed-cache oracle; no GPU result implied."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def repro():
    """Load the standalone oracle without importing FlashInfer's CUDA package."""
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/repro_glm53_sm120_precision.py"
    )
    spec = importlib.util.spec_from_file_location("glm53_precision_repro", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packing_stores_four_arbitrary_scales(repro):
    """Check E4M3 payload bytes and all four FP32 scales against known values."""
    gains = torch.tensor([0.37, 0.71, 1.13, 1.91])
    values = gains.repeat_interleave(128).reshape(1, 512)
    packed = repro.pack_cache(values)
    assert packed.shape == (1, 528)
    codes = packed[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = packed[:, 512:].contiguous().view(torch.float32)
    torch.testing.assert_close(codes, torch.full_like(codes, 448.0))
    torch.testing.assert_close(scales, gains.reshape(1, 4) / 448.0)


def test_oracle_reads_bytes_and_skips_masked_nan(repro):
    """Check hand-packed values, negative-index holes and length-masked NaN slots."""
    # Hand-assembled bytes avoid testing a writer against its own inverse.
    packed = torch.empty(3, 528, dtype=torch.uint8)
    packed[0].fill_(0x7F)
    for row, code in ((1, 1.0), (2, 2.0)):
        packed[row, :512] = (
            torch.full((512,), code).to(torch.float8_e4m3fn).view(torch.uint8)
        )
        packed[row, 512:] = torch.tensor([0.25, 0.5, 0.75, 1.25]).view(torch.uint8)
    q = torch.zeros(2, 16, 512, dtype=torch.bfloat16)
    indices = torch.tensor([[1, -1, 2, 0], [0, 0, 0, 0]], dtype=torch.int32)
    lengths = torch.tensor([3, 0], dtype=torch.int32)
    result = repro.reference(q, packed, indices, lengths)
    expected = (1.5 * torch.tensor([0.25, 0.5, 0.75, 1.25])).repeat_interleave(128)
    torch.testing.assert_close(result[0], expected.expand(16, -1))
    assert torch.equal(result[1], torch.zeros_like(result[1]))


@pytest.mark.parametrize("width", [2112, 2176])
def test_index_padding_keeps_poison_outside_length(repro, width):
    """Keep the poisoned slot outside each live prefix at both supported widths."""
    locations = torch.arange(4095, 0, -1)
    _, indices, lengths = repro.make_inputs(9, locations, width)
    assert lengths.tolist() == [1, 63, 64, 65, 2047, 2048, 2049, 2050, 2051]
    for row, length in enumerate(lengths.tolist()):
        assert torch.equal(indices[row, :length], locations[:length].int())
        assert indices[row, length] == 0
        assert (indices[row, length + 1 :] == -1).all()


def test_metrics_reject_nonfinite_and_measure_error(repro):
    """Verify exact errors for a known perturbation and abort on NaN output."""
    expected = torch.ones(1, 16, 512)
    result = expected + 0.125
    metrics = repro.error_metrics(result, expected)
    assert metrics["relative_rms"] == 0.125
    assert metrics["max_absolute"] == 0.125
    result[0, 0, 0] = float("nan")
    with pytest.raises(AssertionError, match="Non-finite"):
        repro.error_metrics(result, expected)
