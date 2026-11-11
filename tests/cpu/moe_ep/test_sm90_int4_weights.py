# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU oracles for INT4 payload permutation and exact FP8 scale tables."""

import importlib.util
from pathlib import Path
import struct
import sys

import pytest
import torch


pytestmark = pytest.mark.cpu_only
# Avoid importing the mutually exclusive SM90/SM100 kernel trees during CPU CI.
_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel"
    / "shim/int4_weights.py"
)
_SPEC = importlib.util.spec_from_file_location("_sm90_int4_weights_cpu", _SOURCE)
weights = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = weights
_SPEC.loader.exec_module(weights)


def _native_interleave_scalar(payload):
    """Independent scalar transcription of the native CUDA uint16 stores."""
    experts, rows, packed_k = payload.shape
    k = 2 * packed_k
    source = payload.contiguous().view(torch.uint8).reshape(experts, -1)
    out = torch.empty_like(source)
    for expert in range(experts):
        for block_id in range(rows // 2):
            row = (block_id // 8) * 16 + block_id % 8
            for partition in range(k // 64):
                for lane in range(16):
                    dst_row = row + ((lane % 8) // 4) * 8
                    src_col = partition * 16 + lane
                    dst_col = partition * 16 + (lane // 8) * 8 + (lane % 4) * 2
                    for half in range(2):
                        src = ((row + 8 * half) * k // 4 + src_col) * 2
                        dst = (dst_row * k // 4 + dst_col + half) * 2
                        out[expert, dst : dst + 2] = source[expert, src : src + 2]
    return out.reshape(experts, rows, packed_k)


def _fp8_after_bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    rounded = struct.unpack("<f", struct.pack("<I", bits))[0]
    magnitude = min(abs(rounded), 448)
    values = [
        (code & 7) / 512
        if code < 8
        else 2.0 ** ((code >> 3) - 7) * (1 + (code & 7) / 8)
        for code in range(127)
    ]
    code = min(range(127), key=lambda c: (abs(values[c] - magnitude), c % 2))
    return code | (0x80 if bits >> 31 else 0)


def _pack():
    generator = torch.Generator().manual_seed(12)

    def leg(rows, k, dtype):
        return (
            torch.randint(
                256, (3, rows, k // 2), dtype=torch.uint8, generator=generator
            ).view(dtype),
            (torch.rand(3, rows, k // 128, generator=generator) + 0.125).bfloat16(),
        )

    w13, s13 = leg(256, 256, torch.int8)
    w2, s2 = leg(256, 128, torch.uint8)
    return weights.Int4WeightPack(w13, s13, w2, s2)


def test_scale_tables_rounding_and_linear_row_layout():
    # Includes FP8 underflow, double-rounding, ties, and saturation.
    values = torch.tensor(
        [2**-20, 2**-12, 2**-9, 0.035888671875, 0.140625, 0.333984375, 1, 56, 60],
        dtype=torch.bfloat16,
    )
    scales = values[torch.arange(2 * 128 * 3) % len(values)].reshape(2, 128, 3)
    actual = weights.fold_int4_scales(scales)
    assert actual.dtype == torch.uint64
    assert actual.shape == (2, 2, 3, 8, 8)
    assert actual.is_contiguous()
    actual_bytes = actual.view(torch.uint8).reshape(*actual.shape, 8)
    expected_rows = {
        scale: torch.tensor(
            [_fp8_after_bf16(q * scale) for q in range(1, 9)], dtype=torch.uint8
        )
        for scale in values.tolist()
    }
    for expert in range(2):
        for m in range(128):
            for group in range(3):
                assert torch.equal(
                    actual_bytes[expert, m // 64, group, (m % 64) // 8, m % 8],
                    expected_rows[float(scales[expert, m, group])],
                )


def test_preprocessing_payload_gate_up_metadata_and_chunking():
    raw = _pack()
    kwargs = dict(hidden_size=256, intermediate_size=128)
    prepared = weights.preprocess_int4_mega_weights(raw, **kwargs, expert_chunk_size=1)
    one_chunk = weights.preprocess_int4_mega_weights(raw, **kwargs, expert_chunk_size=4)
    gate_rows = torch.tensor(
        [
            half * 128 + block * 8 + j
            for block in range(16)
            for half in range(2)
            for j in range(8)
        ]
    )
    code_map = torch.tensor([11, 0, 1, 2, 3, 8, 9, 10, 15, 14, 13, 12, 7, 6, 5, 4])
    for index, (payload, scales) in enumerate(
        ((raw.w13[:, gate_rows], raw.w13_scale[:, gate_rows]), (raw.w2, raw.w2_scale))
    ):
        bits = payload.to(torch.uint8).long()
        recoded = (code_map[bits & 15] | (code_map[bits >> 4] << 4)).to(torch.uint8)
        packed, tables, activation_scale, expert_scale = prepared[index]
        assert torch.equal(packed.transpose(1, 2), _native_interleave_scalar(recoded))
        assert packed.stride(1) == 1
        assert tables.dtype == torch.uint64 and tables.is_contiguous()
        table_bytes = tables.view(torch.uint8).reshape(*tables.shape, 8)
        for m in range(scales.shape[1]):
            expected = (
                (torch.arange(1, 9).float() * scales[:, m, :, None].float())
                .bfloat16()
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
            assert torch.equal(
                table_bytes[:, m // 64, :, (m % 64) // 8, m % 8], expected
            )
        assert torch.equal(activation_scale, torch.ones(1))
        assert torch.equal(expert_scale, torch.ones(3))
        for actual, other in zip(prepared[index], one_chunk[index], strict=True):
            assert torch.equal(actual, other)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_preprocessing_requires_positive_finite_scales(value):
    raw = _pack()
    raw.w13_scale[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite and positive"):
        weights.preprocess_int4_mega_weights(
            raw, hidden_size=256, intermediate_size=128
        )
