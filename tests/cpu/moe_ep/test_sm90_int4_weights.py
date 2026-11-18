# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU oracles for INT4 payload permutation and exact FP8 scale tables."""

import struct
import subprocess
import sys

import pytest
import torch

from flashinfer.moe_ep import MoEWeightPack, preprocess_sm90_pull_int4_mega_weights
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_int4_bf16_pull_cutedsl import (
    weights,
)


pytestmark = pytest.mark.cpu_only


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
    return MoEWeightPack(w13, w2, s13, s2)


def test_public_preprocessing_does_not_bootstrap_kernel_tree():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from tests.cpu.moe_ep.test_sm90_int4_weights import _pack; "
            "from flashinfer.moe_ep import preprocess_sm90_pull_int4_mega_weights; "
            "preprocess_sm90_pull_int4_mega_weights("
            "_pack(), hidden_size=256, intermediate_size=128); "
            "assert not any(n.startswith('flashinfer.moe_ep.kernel_src.sm90') "
            "for n in sys.modules); "
            "assert not any('pull_style_cutedsl_megakernel/src' in p "
            "for p in sys.path)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_scale_tables_rounding_and_paired_row_layout():
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
            [_fp8_after_bf16(q * scale) for q in (1, 2, 3, 4, -5, -6, -7, -8)],
            dtype=torch.uint8,
        )
        for scale in values.tolist()
    }
    for expert in range(2):
        for m in range(128):
            row = (m % 64) // 16 * 16 + (m % 8) * 2 + (m // 8) % 2
            for group in range(3):
                assert torch.equal(
                    actual_bytes[expert, m // 64, group, row // 8, row % 8],
                    expected_rows[float(scales[expert, m, group])],
                )


def test_preprocessing_payload_gate_up_metadata_and_chunking():
    raw = _pack()
    kwargs = dict(hidden_size=256, intermediate_size=128)
    prepared = preprocess_sm90_pull_int4_mega_weights(
        raw, **kwargs, expert_chunk_size=1
    )
    one_chunk = weights.preprocess_mega_weights(raw, **kwargs, expert_chunk_size=4)
    weights.validate_transformed_mega_weights(
        prepared, **kwargs, num_experts=3, world_size=1
    )
    with pytest.raises(ValueError, match="prequantized"):
        preprocess_sm90_pull_int4_mega_weights(MoEWeightPack(raw.w13, raw.w2), **kwargs)
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
                (
                    torch.tensor([1, 2, 3, 4, -5, -6, -7, -8]).float()
                    * scales[:, m, :, None].float()
                )
                .bfloat16()
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
            row = (m % 64) // 16 * 16 + (m % 8) * 2 + (m // 8) % 2
            assert torch.equal(table_bytes[:, m // 64, :, row // 8, row % 8], expected)
        assert torch.equal(activation_scale, torch.ones(1))
        assert torch.equal(expert_scale, torch.ones(3))
        for actual, other in zip(prepared[index], one_chunk[index], strict=True):
            assert torch.equal(actual, other)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_preprocessing_requires_positive_finite_scales(value):
    raw = _pack()
    raw.w13_scale[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite and positive"):
        preprocess_sm90_pull_int4_mega_weights(
            raw, hidden_size=256, intermediate_size=128
        )
