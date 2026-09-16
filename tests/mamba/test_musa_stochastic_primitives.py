"""Bitwise device checks against independently specified scalar oracles."""

import os
import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl

if os.environ.get("FLASHINFER_MAMBA_TEST_DEVICE") != "musa":
    pytest.skip("requires MUSA", allow_module_level=True)

from flashinfer.mamba.musa_stochastic import philox4x32, cvt_rs_f16
from .test_philox_cpu_oracle import philox4x32_words, cvt_rs_f16_bits


@triton.jit
def _random_words(seed, offsets, output, count: tl.constexpr, rounds: tl.constexpr):
    i = tl.arange(0, 32)
    offset = tl.load(offsets + i, i < count, other=0)
    key = tl.load(seed)
    r0, r1, r2, r3 = philox4x32(key, offset, rounds)
    tl.store(output + i * 4, r0.to(tl.int64), i < count)
    tl.store(output + i * 4 + 1, r1.to(tl.int64), i < count)
    tl.store(output + i * 4 + 2, r2.to(tl.int64), i < count)
    tl.store(output + i * 4 + 3, r3.to(tl.int64), i < count)


@triton.jit
def _round_bits(bits, noise, output, count: tl.constexpr):
    i = tl.arange(0, 32)
    x = tl.load(bits + i, i < count, other=0).to(tl.uint32).to(tl.float32, bitcast=True)
    r = tl.load(noise + i, i < count, other=0).to(tl.uint32)
    result = cvt_rs_f16(x, r).to(tl.uint16, bitcast=True).to(tl.int64)
    tl.store(output + i, result, i < count)


@pytest.mark.parametrize("seed", [0, 42 + 2**40, -1])
@pytest.mark.parametrize("rounds", [5, 10])
def test_device_philox_words(seed, rounds):
    offsets = [0, 1, 2**32 + 1, 2**40 + 31, 2**63 - 1]
    keys = torch.tensor([seed], dtype=torch.int64, device="musa")
    counters = torch.tensor(offsets, dtype=torch.int64, device="musa")
    result = torch.empty(len(offsets), 4, dtype=torch.int64, device="musa")
    _random_words[(1,)](keys, counters, result, len(offsets), rounds)
    expected = torch.tensor(
        [philox4x32_words(seed, o, rounds) for o in offsets], dtype=torch.int64
    )
    assert torch.equal(result.cpu(), expected)


def test_device_software_conversion():
    values = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x38800000,
        0x387FFFFF,
        0xB87FFFFF,
        0x3F800000,
        0x3F801000,
        0xBF801000,
        0x477FF000,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
    ]
    for noise in [0, 1, 4095, 4096, 8191, 0xFFFFFFFF]:
        bits = torch.tensor(values, dtype=torch.int64, device="musa")
        noises = torch.full_like(bits, noise)
        result = torch.empty_like(bits)
        _round_bits[(1,)](bits, noises, result, len(values))
        expected = torch.tensor(
            [cvt_rs_f16_bits(b, noise) for b in values], dtype=torch.int64
        )
        assert torch.equal(result.cpu(), expected)
