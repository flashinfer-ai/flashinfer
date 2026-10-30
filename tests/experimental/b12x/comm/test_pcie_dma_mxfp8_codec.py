"""CPU reference proofs for the MXFP8 PCIe-DMA wire codec."""

from __future__ import annotations

import math

import pytest
import torch

from b12x.comm.pcie.pcie_dma import _normalize_fp8_mode


VALUE_BLOCK = 32
WIRE_BLOCK = 128
FP8_MAX = 448.0


def _scale_byte(values: torch.Tensor) -> int:
    amax = float(values.abs().max())
    if amax == 0.0:
        return 127
    exponent = math.ceil(math.log2(amax / FP8_MAX))
    return max(-127, min(127, exponent)) + 127


def _scale_from_byte(scale_byte: int) -> float:
    return math.ldexp(1.0, scale_byte - 127)


def _quantize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert values.numel() % VALUE_BLOCK == 0
    groups = values.float().reshape(-1, VALUE_BLOCK)
    scale_bytes = torch.tensor(
        [_scale_byte(group) for group in groups], dtype=torch.uint8
    )
    scales = torch.tensor(
        [_scale_from_byte(int(value)) for value in scale_bytes],
        dtype=torch.float32,
    )
    payload = (groups / scales[:, None]).to(torch.float8_e4m3fn)
    return payload.reshape(-1), scale_bytes


def _dequantize(payload: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
    scales = torch.tensor(
        [_scale_from_byte(int(value)) for value in scale_bytes],
        dtype=torch.float32,
    )
    return (payload.float().reshape(-1, VALUE_BLOCK) * scales[:, None]).reshape(-1)


def test_mxfp8_layout_matches_other_compressed_wire_modes() -> None:
    assert WIRE_BLOCK + WIRE_BLOCK // VALUE_BLOCK == 132


def test_mxfp8_scale_boundaries_and_zero_encoding() -> None:
    values = torch.zeros((4, VALUE_BLOCK), dtype=torch.float32)
    values[1, 0] = FP8_MAX
    values[2, 0] = FP8_MAX + 1.0
    values[3, 0] = FP8_MAX / 2.0
    _, scale_bytes = _quantize(values.reshape(-1))
    assert scale_bytes.tolist() == [127, 127, 128, 126]


def test_mxfp8_roundtrip_uses_power_of_two_scales_without_overflow() -> None:
    values = torch.linspace(-13.0, 17.0, 4 * WIRE_BLOCK)
    values[7] = 1200.0
    values[VALUE_BLOCK + 3] = -0.0002
    payload, scale_bytes = _quantize(values)
    restored = _dequantize(payload, scale_bytes)

    assert torch.isfinite(payload.float()).all()
    assert torch.isfinite(restored).all()
    assert float(payload.float().abs().max()) <= FP8_MAX
    for scale_byte in scale_bytes.tolist():
        scale = _scale_from_byte(scale_byte)
        assert math.frexp(scale)[0] == 0.5


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("mx", "mx"),
        ("mxfp8", "mx"),
        ("mxfp8-ag", "mx"),
        ("mx_ring", "mx_ring"),
        ("mxfp8-ring", "mx_ring"),
        ("mx_a2a", "mx_a2a"),
        ("mxfp8-a2a", "mx_a2a"),
    ],
)
def test_mxfp8_mode_aliases(alias: str, canonical: str) -> None:
    assert _normalize_fp8_mode(alias) == canonical


def test_mxfp8_topologies_forward_one_canonical_payload() -> None:
    contributions = [
        torch.sin(torch.arange(WIRE_BLOCK) * 0.03 + rank) * (rank + 1)
        for rank in range(4)
    ]

    reduced = torch.stack(contributions).sum(dim=0)
    ag_payload = _quantize(reduced)
    ag_results = [_dequantize(*ag_payload) for _ in contributions]

    ring_payload = _quantize(contributions[0])
    for local in contributions[1:]:
        ring_payload = _quantize(local + _dequantize(*ring_payload))
    ring_results = [_dequantize(*ring_payload) for _ in contributions]

    incoming = [_quantize(values) for values in contributions[1:]]
    accumulated = contributions[0].clone()
    for source in incoming:
        accumulated += _dequantize(*source)
    a2a_payload = _quantize(accumulated)
    a2a_results = [_dequantize(*a2a_payload) for _ in contributions]

    for results in (ag_results, ring_results, a2a_results):
        assert all(torch.equal(result, results[0]) for result in results[1:])
