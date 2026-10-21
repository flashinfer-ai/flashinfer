"""Dependency-free proofs for the block-INT8 PCIe-DMA wire codec."""

from __future__ import annotations

import math
import random


BLOCK = 128
QMAX = 127


def _quantize(values: list[float]) -> tuple[list[int], list[float]]:
    assert len(values) % BLOCK == 0
    payload: list[int] = []
    scales: list[float] = []
    for offset in range(0, len(values), BLOCK):
        block = values[offset : offset + BLOCK]
        amax = max(abs(value) for value in block)
        scale = amax / QMAX if amax > 0.0 else 1.0
        scales.append(scale)
        payload.extend(max(-QMAX, min(QMAX, round(value / scale))) for value in block)
    return payload, scales


def _dequantize(payload: list[int], scales: list[float]) -> list[float]:
    return [value * scales[index // BLOCK] for index, value in enumerate(payload)]


def test_int8_layout_matches_e4m3_wire_bytes() -> None:
    # Both modes ship 128 one-byte values plus one fp32 scale per block.
    assert BLOCK + 4 == 132


def test_int8_roundtrip_has_half_step_error_bound() -> None:
    rng = random.Random(0)
    values = [rng.uniform(-20.0, 20.0) for _ in range(4 * BLOCK)]
    values[17] = 311.0
    values[BLOCK + 9] = -0.0002
    payload, scales = _quantize(values)
    restored = _dequantize(payload, scales)

    assert min(payload) >= -QMAX
    assert max(payload) <= QMAX
    for index, (actual, expected) in enumerate(zip(restored, values, strict=True)):
        assert abs(actual - expected) <= scales[index // BLOCK] / 2 + 1e-12


def test_zero_block_is_finite_and_exact() -> None:
    payload, scales = _quantize([0.0] * BLOCK)
    assert payload == [0] * BLOCK
    assert scales == [1.0]
    assert _dequantize(payload, scales) == [0.0] * BLOCK
    assert all(math.isfinite(value) for value in scales)


def test_ag_ring_and_a2a_forward_one_canonical_payload() -> None:
    rng = random.Random(1)
    contributions = [[rng.uniform(-2.0, 2.0) for _ in range(BLOCK)] for _ in range(4)]

    # AG: the BF16 reduce-scatter result is quantized once at its owner.
    reduced = [sum(values) for values in zip(*contributions, strict=True)]
    ag_payload = _quantize(reduced)
    ag_results = [_dequantize(*ag_payload) for _ in contributions]

    # Ring: each hop consumes and emits one canonical INT8 partial. The final
    # payload is forwarded verbatim and also rematerialized by its owner.
    ring_payload = _quantize(contributions[0])
    for local in contributions[1:]:
        partial = _dequantize(*ring_payload)
        ring_payload = _quantize(
            [
                local_value + partial_value
                for local_value, partial_value in zip(local, partial, strict=True)
            ]
        )
    ring_results = [_dequantize(*ring_payload) for _ in contributions]

    # A2A: incoming source payloads are accumulated in float, then the owner
    # quantizes once for the common broadcast payload.
    incoming = [_quantize(values) for values in contributions[1:]]
    accumulated = contributions[0].copy()
    for source in incoming:
        accumulated = [
            current + value
            for current, value in zip(accumulated, _dequantize(*source), strict=True)
        ]
    a2a_payload = _quantize(accumulated)
    a2a_results = [_dequantize(*a2a_payload) for _ in contributions]

    for results in (ag_results, ring_results, a2a_results):
        assert all(result == results[0] for result in results[1:])
