"""CPU contract tests for the versioned SM90 NVFP4 K-major layout."""

import json
import math
from dataclasses import replace

import pytest
import torch

import flashinfer.fused_moe.sm90_nvfp4_repack as repack_module
from flashinfer.fused_moe.nvfp4_checkpoint import (
    NVFP4Checkpoint,
    reference_dequantize_nvfp4,
)
from flashinfer.fused_moe.sm90_nvfp4_repack import (
    NVFP4_RS_LAYOUT_VERSION,
    NVFP4_SM90_K_ALIGNMENT,
    NVFP4_SM90_LAYOUT_VERSION,
    NVFP4V3Manifest,
    build_nvfp4_rs_weight_view,
    convert_nvfp4_rs_v2_to_v3,
    reference_dequantize_nvfp4_sm90_v3_promoted,
    repack_nvfp4_sm90_v3,
    repack_nvfp4_sm90_v3_selected,
    unpack_nvfp4_payload_v2,
    unpack_nvfp4_scales_v2,
    unpack_nvfp4_sm90_v3,
)
from tests.moe._nvfp4_w4a8_oracle import (
    simulate_w4a8,
    simulate_w4a8_operand_bytes,
)


def _checkpoint(
    *,
    experts: int = 2,
    physical_n: int = 67,
    physical_k: int = 64,
    logical_n: int = 65,
    logical_k: int = 47,
    scalar_alpha: bool = False,
) -> NVFP4Checkpoint:
    expert_axis = torch.arange(experts, dtype=torch.int64)[:, None, None]
    row_axis = torch.arange(physical_n, dtype=torch.int64)[None, :, None]
    packed_k_axis = torch.arange(physical_k // 2, dtype=torch.int64)[None, None, :]
    payload = (
        (expert_axis * 101 + row_axis * 17 + packed_k_axis * 37 + 11)
        .remainder(256)
        .to(torch.uint8)
    )
    scale_block_axis = torch.arange(physical_k // 16, dtype=torch.int64)[None, None, :]
    scale_bits = (
        (expert_axis * 43 + row_axis * 17 + scale_block_axis * 29)
        .remainder(126)
        .add(1)
        .to(torch.uint8)
    )
    scales = scale_bits.view(torch.float8_e4m3fn)
    if scalar_alpha:
        alpha = torch.tensor(0.25, dtype=torch.float32)
    else:
        alpha = torch.linspace(0.25, 0.75, experts, dtype=torch.float32)
    return NVFP4Checkpoint(
        packed_e2m1=payload,
        scale_e4m3_per16=scales,
        global_alpha=alpha,
        logical_shape=(experts, logical_n, logical_k),
        expert_mapping=tuple(range(10, 10 + experts)),
        source_format_version="modelopt.nvfp4.test",
    )


_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _checkpoint_code(
    checkpoint: NVFP4Checkpoint, expert: int, row: int, column: int
) -> int:
    _, logical_n, logical_k = checkpoint.logical_shape
    if row >= logical_n or column >= logical_k:
        return 0
    packed = int(checkpoint.packed_e2m1[expert, row, column // 2].item())
    return (packed >> (4 * (column % 2))) & 0x0F


def _checkpoint_scale(
    checkpoint: NVFP4Checkpoint, expert: int, row: int, scale_block: int
) -> torch.Tensor:
    _, logical_n, logical_k = checkpoint.logical_shape
    if row >= logical_n or scale_block >= math.ceil(logical_k / 16):
        return torch.zeros((), dtype=torch.float8_e4m3fn)
    return checkpoint.scale_e4m3_per16[expert, row, scale_block]


def _independent_promotion_row(
    checkpoint: NVFP4Checkpoint,
    *,
    expert: int,
    row: int,
    padded_k: int,
    group_size: int,
    residual_scheme: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the v3 promotion contract without using a repack inverse."""

    scale_blocks = padded_k // 16
    blocks_per_group = group_size // 16
    scales = torch.tensor(
        [
            float(_checkpoint_scale(checkpoint, expert, row, block).item())
            for block in range(scale_blocks)
        ],
        dtype=torch.float32,
    )
    block_max = torch.tensor(
        [
            max(
                _E2M1_MAGNITUDES[
                    _checkpoint_code(checkpoint, expert, row, column) & 0x07
                ]
                for column in range(block * 16, (block + 1) * 16)
            )
            for block in range(scale_blocks)
        ],
        dtype=torch.float32,
    )
    grouped_scales = scales.reshape(-1, blocks_per_group)
    grouped_max = block_max.reshape(-1, blocks_per_group)
    group_scale = (grouped_max * grouped_scales).amax(dim=-1) / 448.0
    group_scale = torch.where(
        group_scale > 0, group_scale, torch.ones_like(group_scale)
    )

    if residual_scheme == "generic":
        group_scale = group_scale * (1.0 + 2.0**-7)
        residual = (grouped_scales / group_scale[:, None]).to(torch.bfloat16)
    else:
        ratio = grouped_scales / group_scale[:, None]
        exponent = torch.where(
            ratio > 0,
            torch.round(torch.log2(ratio)),
            torch.zeros_like(ratio),
        ).clamp(-127, 127)
        residual_value = torch.where(
            ratio > 0,
            torch.pow(torch.full_like(exponent, 2.0), exponent),
            torch.zeros_like(exponent),
        )
        normalized_max = (grouped_max * residual_value).amax(dim=-1)
        shift = torch.where(
            normalized_max > 448.0,
            torch.ceil(torch.log2(normalized_max / 448.0)),
            torch.zeros_like(normalized_max),
        ).clamp_min(0)
        group_scale = group_scale * torch.pow(torch.full_like(shift, 2.0), shift)
        exponent = (exponent - shift[:, None]).clamp(-127, 127)
        residual = torch.where(
            ratio > 0,
            exponent,
            torch.full_like(exponent, -128),
        ).to(torch.int8)
    return group_scale, residual.reshape(-1)


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("residual_scheme", ["generic", "pow2"])
def test_v3_semantic_roundtrip_and_determinism(group_size, residual_scheme):
    checkpoint = _checkpoint()
    first = repack_nvfp4_sm90_v3(
        checkpoint, group_size=group_size, residual_scheme=residual_scheme
    )
    second = repack_nvfp4_sm90_v3(
        checkpoint, group_size=group_size, residual_scheme=residual_scheme
    )
    restored = unpack_nvfp4_sm90_v3(first)

    assert NVFP4_SM90_LAYOUT_VERSION == 3
    assert first.manifest.group_size == group_size
    assert first.manifest.residual_scheme == residual_scheme
    assert first.manifest.alpha_scope == "per_expert"
    assert first.manifest.to_dict() == second.manifest.to_dict()
    assert torch.equal(first.packed_e2m1, second.packed_e2m1)
    assert torch.equal(
        first.scale_e4m3_per16.view(torch.uint8),
        second.scale_e4m3_per16.view(torch.uint8),
    )
    assert torch.equal(first.promotion_group_scale, second.promotion_group_scale)
    assert torch.equal(first.promotion_residual, second.promotion_residual)
    assert first.promotion_residual.dtype == (
        torch.bfloat16 if residual_scheme == "generic" else torch.int8
    )
    torch.testing.assert_close(
        reference_dequantize_nvfp4(restored),
        reference_dequantize_nvfp4(checkpoint),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        reference_dequantize_nvfp4_sm90_v3_promoted(first),
        simulate_w4a8(checkpoint, group_size, residual_scheme),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("residual_scheme", ["generic", "pow2"])
def test_v3_operand_byte_oracle_consumes_physical_device_streams(
    group_size, residual_scheme
):
    view = repack_nvfp4_sm90_v3(
        _checkpoint(), group_size=group_size, residual_scheme=residual_scheme
    )
    operand_bytes = simulate_w4a8_operand_bytes(
        view.packed_e2m1,
        view.promotion_residual,
        residual_scheme=residual_scheme,
    )
    experts, padded_n, padded_k = view.manifest.padded_shape
    operand = (
        operand_bytes.permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(experts, padded_n, padded_k)
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    group_scale = (
        view.promotion_group_scale.permute(0, 2, 3, 1)
        .contiguous()
        .view(experts, padded_n, padded_k // group_size)
        .repeat_interleave(group_size, dim=-1)
    )
    reconstructed = operand * group_scale
    alpha = (
        view.global_alpha.expand(experts)
        if view.global_alpha.ndim == 0
        else view.global_alpha
    )
    reconstructed = reconstructed * alpha[:, None, None]
    _, logical_n, logical_k = view.manifest.logical_shape
    torch.testing.assert_close(
        reconstructed[:, :logical_n, :logical_k],
        reference_dequantize_nvfp4_sm90_v3_promoted(view),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("residual_scheme", ["generic", "pow2"])
def test_v3_operand_byte_oracle_pins_e2m1_subnormal_and_zero_bytes(
    residual_scheme,
):
    payload = torch.zeros((1, 1, 1, 64, 16), dtype=torch.uint8)
    codes = torch.zeros(32, dtype=torch.uint8)
    codes[:16] = torch.arange(16, dtype=torch.uint8)
    codes[16:18] = torch.tensor([7, 15], dtype=torch.uint8)
    payload[0, 0, 0, 0] = codes[0::2] | (codes[1::2] << 4)
    payload[0, 0, 0, 1, 0] = 7 | (15 << 4)

    if residual_scheme == "generic":
        residual = torch.zeros((1, 1, 1, 64, 2), dtype=torch.bfloat16)
        residual[0, 0, 0, 0] = torch.tensor([1.0, 2.0**-12], dtype=torch.bfloat16)
    else:
        residual = torch.full((1, 1, 1, 64, 2), -128, dtype=torch.int8)
        residual[0, 0, 0, 0] = torch.tensor([0, -12], dtype=torch.int8)

    actual = simulate_w4a8_operand_bytes(
        payload, residual, residual_scheme=residual_scheme
    )
    assert actual.dtype == torch.uint8
    assert actual.shape == (1, 1, 1, 64, 32)
    assert actual.is_contiguous()
    assert actual[0, 0, 0, 0, :16].tolist() == [
        0x00,
        0x30,
        0x38,
        0x3C,
        0x40,
        0x44,
        0x48,
        0x4C,
        0x80,
        0xB0,
        0xB8,
        0xBC,
        0xC0,
        0xC4,
        0xC8,
        0xCC,
    ]
    assert actual[0, 0, 0, 0, 16:18].tolist() == [0x01, 0x81]
    assert actual[0, 0, 0, 1, :2].tolist() == [0x00, 0x80]


@pytest.mark.parametrize(
    ("residual_scheme", "residual_dtype"),
    [("generic", torch.int8), ("pow2", torch.bfloat16)],
)
def test_v3_operand_byte_oracle_rejects_non_abi_residual_dtype(
    residual_scheme, residual_dtype
):
    payload = torch.zeros((1, 1, 1, 64, 16), dtype=torch.uint8)
    residual = torch.zeros((1, 1, 1, 64, 2), dtype=residual_dtype)
    with pytest.raises(ValueError, match="promotion_residual must have dtype"):
        simulate_w4a8_operand_bytes(payload, residual, residual_scheme=residual_scheme)


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("residual_scheme", ["generic", "pow2"])
def test_v3_physical_coordinates_follow_the_frozen_layout_contract(
    group_size, residual_scheme
):
    checkpoint = _checkpoint(
        physical_n=70,
        physical_k=96,
        logical_n=67,
        logical_k=77,
    )
    view = repack_nvfp4_sm90_v3(
        checkpoint,
        group_size=group_size,
        residual_scheme=residual_scheme,
    )
    experts, padded_n, padded_k = view.manifest.padded_shape
    n_tiles = padded_n // 64
    k_tiles = padded_k // 32
    scale_blocks = padded_k // 16

    assert n_tiles >= 2 and k_tiles >= 3
    assert padded_n > checkpoint.logical_shape[1]
    assert padded_k > checkpoint.logical_shape[2]
    assert view.packed_e2m1.shape == (experts, k_tiles, n_tiles, 64, 16)
    assert view.scale_e4m3_per16.shape == (
        experts,
        k_tiles,
        n_tiles,
        64,
        2,
    )
    assert view.promotion_residual.shape == (
        experts,
        k_tiles,
        n_tiles,
        64,
        2,
    )
    assert view.promotion_group_scale.shape == (
        experts,
        padded_k // group_size,
        n_tiles,
        64,
    )

    for expert in range(experts):
        for row in range(padded_n):
            for column in range(padded_k):
                packed = int(
                    view.packed_e2m1[
                        expert,
                        column // 32,
                        row // 64,
                        row % 64,
                        (column % 32) // 2,
                    ].item()
                )
                actual_code = (packed >> (4 * (column % 2))) & 0x0F
                assert actual_code == _checkpoint_code(checkpoint, expert, row, column)

            expected_group_scale, expected_residual = _independent_promotion_row(
                checkpoint,
                expert=expert,
                row=row,
                padded_k=padded_k,
                group_size=group_size,
                residual_scheme=residual_scheme,
            )
            for scale_block in range(scale_blocks):
                coordinate = (
                    expert,
                    scale_block // 2,
                    row // 64,
                    row % 64,
                    scale_block % 2,
                )
                actual_scale = view.scale_e4m3_per16[coordinate]
                expected_scale = _checkpoint_scale(checkpoint, expert, row, scale_block)
                assert torch.equal(
                    actual_scale.view(torch.uint8),
                    expected_scale.view(torch.uint8),
                )
                actual_residual = view.promotion_residual[coordinate]
                assert torch.equal(
                    actual_residual.reshape(1).view(torch.uint8),
                    expected_residual[scale_block].reshape(1).view(torch.uint8),
                )

            for group in range(padded_k // group_size):
                actual_group_scale = view.promotion_group_scale[
                    expert, group, row // 64, row % 64
                ]
                assert torch.equal(
                    actual_group_scale.reshape(1).view(torch.uint8),
                    expected_group_scale[group].reshape(1).view(torch.uint8),
                )


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("residual_scheme", ["generic", "pow2"])
def test_v3_sparse_sentinels_pin_every_physical_axis(group_size, residual_scheme):
    payload = torch.zeros((2, 70, 80), dtype=torch.uint8)
    payload_markers = (
        (0, 0, 0, 0xA3),
        (1, 0, 0, 0xB4),
        (0, 65, 0, 0xC5),
        (0, 17, 0, 0xD6),
        (0, 0, 16, 0xE7),
        (0, 0, 7, 0x91),
        (0, 0, 64, 0x72),
    )
    for expert, row, packed_k, value in payload_markers:
        payload[expert, row, packed_k] = value

    scale_bits = torch.zeros((2, 70, 10), dtype=torch.uint8)
    scale_markers = (
        (0, 0, 0, 0x11),
        (1, 0, 0, 0x22),
        (0, 65, 0, 0x33),
        (0, 17, 0, 0x44),
        (0, 0, 2, 0x55),
        (0, 0, 1, 0x66),
        (0, 0, 8, 0x77),
    )
    for expert, row, scale_block, value in scale_markers:
        scale_bits[expert, row, scale_block] = value

    checkpoint = NVFP4Checkpoint(
        payload,
        scale_bits.view(torch.float8_e4m3fn),
        torch.ones(2, dtype=torch.float32),
        (2, 67, 145),
        (10, 11),
        "modelopt.nvfp4.sentinel",
    )
    view = repack_nvfp4_sm90_v3(
        checkpoint,
        group_size=group_size,
        residual_scheme=residual_scheme,
    )

    for expert, row, packed_k, value in payload_markers:
        coordinate = (
            expert,
            packed_k // 16,
            row // 64,
            row % 64,
            packed_k % 16,
        )
        assert int(view.packed_e2m1[coordinate].item()) == value
        assert int((view.packed_e2m1 == value).sum().item()) == 1

    residual_zero = 0 if residual_scheme == "generic" else -128
    actual_residual_markers = set()
    for coordinate in torch.nonzero(
        view.promotion_residual != residual_zero, as_tuple=False
    ).tolist():
        actual_residual_markers.add(tuple(coordinate))

    expected_residual_markers = set()
    for expert, row, scale_block, value in scale_markers:
        coordinate = (
            expert,
            scale_block // 2,
            row // 64,
            row % 64,
            scale_block % 2,
        )
        expected_residual_markers.add(coordinate)
        actual_scale_bits = view.scale_e4m3_per16[coordinate].view(torch.uint8)
        assert int(actual_scale_bits.item()) == value
        assert int((view.scale_e4m3_per16.view(torch.uint8) == value).sum().item()) == 1
    assert actual_residual_markers == expected_residual_markers

    baseline = 1.0 + 2.0**-7 if residual_scheme == "generic" else 1.0
    actual_group_markers = {
        tuple(coordinate)
        for coordinate in torch.nonzero(
            view.promotion_group_scale != baseline, as_tuple=False
        ).tolist()
    }
    expected_group_markers = {
        (expert, (packed_k * 2) // group_size, row // 64, row % 64)
        for expert, row, packed_k, _ in payload_markers
    }
    assert actual_group_markers == expected_group_markers


@pytest.mark.parametrize("logical_k", [17, 33, 63])
def test_v3_tail_padding_is_canonical_zero(logical_k):
    checkpoint = _checkpoint(
        physical_n=66,
        physical_k=64,
        logical_n=65,
        logical_k=logical_k,
        scalar_alpha=True,
    )
    view = repack_nvfp4_sm90_v3(checkpoint, group_size=64, residual_scheme="pow2")
    restored = unpack_nvfp4_sm90_v3(view)
    _, padded_n, padded_k = restored.physical_shape
    assert view.manifest.alpha_scope == "per_tensor"
    assert not bool(restored.packed_e2m1[:, 65:, :].any())
    assert not bool(restored.packed_e2m1[:, :65, math.ceil(logical_k / 2) :].any())
    if logical_k % 2:
        tail = restored.packed_e2m1[:, :65, logical_k // 2]
        assert not bool(tail.bitwise_and(0xF0).any())
    scale_tail = math.ceil(logical_k / 16)
    assert not bool(
        restored.scale_e4m3_per16[:, :65, scale_tail:].view(torch.uint8).any()
    )
    assert padded_n % 64 == 0 and padded_k % NVFP4_SM90_K_ALIGNMENT == 0


@pytest.mark.parametrize("logical_k", [1, 31, 32, 33, 63, 64, 65, 127, 128, 129])
def test_v3_k128_padding_is_frozen_across_group_sizes(logical_k):
    physical_k = math.ceil(logical_k / 16) * 16
    checkpoint = _checkpoint(
        experts=1,
        physical_n=1,
        physical_k=physical_k,
        logical_n=1,
        logical_k=logical_k,
    )
    expected_k = math.ceil(logical_k / NVFP4_SM90_K_ALIGNMENT) * NVFP4_SM90_K_ALIGNMENT
    for group_size in (32, 64, 128):
        view = repack_nvfp4_sm90_v3(
            checkpoint, group_size=group_size, residual_scheme="generic"
        )
        assert view.manifest.padded_shape == (1, 64, expected_k)
        assert view.packed_e2m1.shape[1] == expected_k // 32
        assert view.promotion_group_scale.shape[1] == expected_k // group_size


@pytest.mark.parametrize("invalid_padded_k", [32, 64, 129, 256])
def test_v3_manifest_rejects_noncanonical_k128_padding(invalid_padded_k):
    view = repack_nvfp4_sm90_v3(
        _checkpoint(experts=1), group_size=32, residual_scheme="generic"
    )
    manifest = view.manifest.to_dict()
    manifest["padded_shape"] = [1, 128, invalid_padded_k]
    with pytest.raises(ValueError, match="minimal N64/K128 padding"):
        NVFP4V3Manifest.from_dict(manifest)


def test_v3_ignores_source_physical_padding():
    first = _checkpoint()
    payload = first.packed_e2m1.clone()
    scales = first.scale_e4m3_per16.clone()
    payload[:, 65:, :].fill_(0xFF)
    payload[:, :65, 24:].fill_(0xFF)
    payload[:, :65, 23].bitwise_or_(0xF0)
    scales[:, 65:, :].view(torch.uint8).fill_(0x7E)
    scales[:, :65, 3:].view(torch.uint8).fill_(0x7E)
    second = NVFP4Checkpoint(
        payload,
        scales,
        first.global_alpha,
        first.logical_shape,
        first.expert_mapping,
        first.source_format_version,
    )
    first_v3 = repack_nvfp4_sm90_v3(first, group_size=128, residual_scheme="generic")
    second_v3 = repack_nvfp4_sm90_v3(second, group_size=128, residual_scheme="generic")
    assert torch.equal(first_v3.packed_e2m1, second_v3.packed_e2m1)
    assert torch.equal(
        first_v3.scale_e4m3_per16.view(torch.uint8),
        second_v3.scale_e4m3_per16.view(torch.uint8),
    )
    assert first_v3.manifest.checksums == second_v3.manifest.checksums


def test_v3_empty_expert_axis_and_stable_manifest_json():
    checkpoint = NVFP4Checkpoint(
        torch.empty((0, 3, 16), dtype=torch.uint8),
        torch.empty((0, 3, 2), dtype=torch.float32).to(torch.float8_e4m3fn),
        torch.empty((0,), dtype=torch.float32),
        (0, 2, 17),
        (),
        "empty.test",
    )
    view = repack_nvfp4_sm90_v3(checkpoint, group_size=32, residual_scheme="pow2")
    restored = unpack_nvfp4_sm90_v3(view)
    manifest_json = json.dumps(
        view.manifest.to_dict(), sort_keys=True, separators=(",", ":")
    )
    assert json.loads(manifest_json) == view.manifest.to_dict()
    assert restored.logical_shape == (0, 2, 17)
    assert restored.expert_mapping == ()
    assert view.packed_e2m1.shape == (0, 4, 1, 64, 16)


def test_v3_frozen_manifest_and_checksums_are_byte_exact():
    view = repack_nvfp4_sm90_v3(
        _checkpoint(
            experts=1,
            physical_n=64,
            physical_k=32,
            logical_n=3,
            logical_k=17,
        ),
        group_size=32,
        residual_scheme="generic",
    )
    assert view.manifest.to_dict() == {
        "layout_version": 3,
        "source_format_version": "modelopt.nvfp4.test",
        "sm_target": "sm90a",
        "group_size": 32,
        "residual_scheme": "generic",
        "rounding_mode": "rne",
        "byte_order": "little",
        "global_layout": "kmajor_k32_n64_contiguous",
        "w13_layout": "gate_then_up",
        "logical_shape": [1, 3, 17],
        "padded_shape": [1, 64, 128],
        "nibble_order": "low_even_high_odd",
        "alpha_scope": "per_expert",
        "expert_mapping": [10],
        "checksums": {
            "payload_sha256": (
                "ea59d4df20a51c94f4a4e2d292182e85fe635a5f45630a7519e9cb3e9da87abf"
            ),
            "scale_sha256": (
                "36c76dd032fb1235a05bba43e8aa73da7d6a0783690d947e934ee87d8c4833c1"
            ),
            "group_scale_sha256": (
                "77be6a17a9e8d445c11d8e78253ad9449486a2cdfb592e7147c9bf425a78bace"
            ),
            "residual_sha256": (
                "938ed246d3597afa430ba834536e2638fb6ee270b1364b0733fb565c20723de2"
            ),
            "alpha_sha256": (
                "1c8de671610945ee97d172c59abe42c3e3b3c6098890ce163ef7bd77224ed695"
            ),
        },
    }


@pytest.mark.parametrize(
    ("value", "expected_raw"),
    (
        (448.0, 0x7E),
        (464.0, 0x7E),
        (
            torch.nextafter(
                torch.tensor(464.0, dtype=torch.float32),
                torch.tensor(float("inf"), dtype=torch.float32),
            ).item(),
            0x7E,
        ),
        (1000.0, 0x7E),
        (-0.0, 0x80),
    ),
)
def test_v3_promotion_e4m3_cast_clamps_finite_boundaries(value, expected_raw):
    actual = repack_module._cast_promoted_e4m3(
        torch.tensor((value,), dtype=torch.float32)
    )
    assert actual.view(torch.uint8).item() == expected_raw


@pytest.mark.parametrize("value", (float("inf"), float("-inf"), float("nan")))
def test_v3_promotion_e4m3_cast_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="promoted E4M3 inputs must be finite"):
        repack_module._cast_promoted_e4m3(torch.tensor((value,), dtype=torch.float32))


def test_v3_rejects_negative_zero_in_exact_scales():
    view = repack_nvfp4_sm90_v3(_checkpoint(), group_size=64, residual_scheme="generic")
    scales = view.scale_e4m3_per16.clone()
    scales.view(torch.uint8).reshape(-1)[0] = 0x80
    with pytest.raises(ValueError, match="scales must not contain negative zero"):
        replace(view, scale_e4m3_per16=scales)


@pytest.mark.parametrize("raw_nan", [0x7F, 0xFF])
def test_v3_rejects_e4m3_nan_encodings(raw_nan):
    view = repack_nvfp4_sm90_v3(_checkpoint(), group_size=64, residual_scheme="generic")
    scales = view.scale_e4m3_per16.clone()
    scales.view(torch.uint8).reshape(-1)[0] = raw_nan
    with pytest.raises(ValueError, match="scales must be finite"):
        replace(view, scale_e4m3_per16=scales)


def test_v3_rejects_negative_zero_in_generic_residuals():
    view = repack_nvfp4_sm90_v3(_checkpoint(), group_size=64, residual_scheme="generic")
    residual = view.promotion_residual.clone()
    residual.view(torch.int16).reshape(-1)[0] = -0x8000
    with pytest.raises(ValueError, match="residuals must not contain negative zero"):
        replace(view, promotion_residual=residual)


def test_v3_checksum_bypass_still_revalidates_mutable_values():
    view = repack_nvfp4_sm90_v3(_checkpoint(), group_size=64, residual_scheme="generic")
    view.scale_e4m3_per16.view(torch.uint8).reshape(-1)[0] = 0x7F
    with pytest.raises(ValueError, match="scales must be finite"):
        unpack_nvfp4_sm90_v3(view, verify_checksums=False)


def test_v3_rejects_non_little_endian_host(monkeypatch):
    view = repack_nvfp4_sm90_v3(_checkpoint(), group_size=64, residual_scheme="generic")
    monkeypatch.setattr(repack_module.sys, "byteorder", "big")
    with pytest.raises(RuntimeError, match="little-endian host"):
        view.validate_layout()


@pytest.mark.parametrize(
    "tensor_name", ["payload", "scale", "group_scale", "residual", "alpha"]
)
def test_v3_detects_tensor_tampering(tensor_name):
    view = repack_nvfp4_sm90_v3(_checkpoint(), group_size=64, residual_scheme="generic")
    if tensor_name == "payload":
        view.packed_e2m1.reshape(-1)[0].bitwise_xor_(1)
    elif tensor_name == "scale":
        view.scale_e4m3_per16.view(torch.uint8).reshape(-1)[0].bitwise_xor_(1)
    elif tensor_name == "group_scale":
        view.promotion_group_scale.reshape(-1)[0].add_(0.125)
    elif tensor_name == "residual":
        view.promotion_residual.view(torch.uint8).reshape(-1)[0].bitwise_xor_(1)
    else:
        view.global_alpha[0].add_(0.125)
    with pytest.raises(ValueError, match=f"checksum mismatch for {tensor_name}"):
        unpack_nvfp4_sm90_v3(view)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_format_version", "tampered.source"),
        ("expert_mapping", [101, 102]),
        ("logical_shape", [2, 66, 47]),
    ],
)
def test_v3_checksums_bind_manifest_metadata(field, value):
    view = repack_nvfp4_sm90_v3(_checkpoint(), group_size=64, residual_scheme="generic")
    manifest_dict = view.manifest.to_dict()
    manifest_dict[field] = value
    tampered_manifest = NVFP4V3Manifest.from_dict(manifest_dict)
    tampered_view = replace(view, manifest=tampered_manifest)
    with pytest.raises(ValueError, match="checksum mismatch"):
        unpack_nvfp4_sm90_v3(tampered_view)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("source_format_version", " modelopt.nvfp4.test ", ValueError),
        ("logical_shape", [2, 65.9, 47], TypeError),
        ("expert_mapping", [10, 11.9], TypeError),
        ("layout_version", 3.0, TypeError),
        ("group_size", 64.0, TypeError),
    ],
)
def test_v3_manifest_rejects_lossy_or_aliasing_metadata(field, value, error):
    view = repack_nvfp4_sm90_v3(_checkpoint(), group_size=64, residual_scheme="generic")
    manifest_dict = view.manifest.to_dict()
    manifest_dict[field] = value
    with pytest.raises(error):
        NVFP4V3Manifest.from_dict(manifest_dict)


def test_v3_residual_scheme_is_materialized_not_manifest_only():
    checkpoint = _checkpoint()
    generic = repack_nvfp4_sm90_v3(checkpoint, group_size=64, residual_scheme="generic")
    pow2 = repack_nvfp4_sm90_v3(checkpoint, group_size=64, residual_scheme="pow2")
    assert generic.promotion_residual.dtype == torch.bfloat16
    assert pow2.promotion_residual.dtype == torch.int8
    assert (
        generic.manifest.checksums.residual_sha256
        != pow2.manifest.checksums.residual_sha256
    )


def test_v3_streaming_promotion_and_hash_chunks_preserve_semantics(monkeypatch):
    checkpoint = _checkpoint()
    baseline = repack_nvfp4_sm90_v3(
        checkpoint, group_size=32, residual_scheme="generic"
    )
    monkeypatch.setattr(repack_module, "_NVFP4_SM90_PROMOTION_CHUNK_ELEMENTS", 64)
    monkeypatch.setattr(repack_module, "_NVFP4_SM90_HASH_CHUNK_BYTES", 17)
    view = repack_nvfp4_sm90_v3(checkpoint, group_size=32, residual_scheme="generic")
    view.verify_checksums()
    assert view.manifest.checksums == baseline.manifest.checksums
    assert torch.equal(view.promotion_group_scale, baseline.promotion_group_scale)
    assert torch.equal(view.promotion_residual, baseline.promotion_residual)
    torch.testing.assert_close(
        reference_dequantize_nvfp4_sm90_v3_promoted(view),
        simulate_w4a8(checkpoint, 32, "generic"),
        rtol=0,
        atol=0,
    )


def test_v3_selected_policy_materializes_homogeneous_buckets_and_fallback():
    checkpoint = _checkpoint(experts=3)
    selection = {
        "mode": "per_expert",
        "residual_scheme": "generic",
        "group_size": None,
        "experts": [
            {"expert_id": 10, "group_size": 32},
            {"expert_id": 11, "group_size": 128},
            {"expert_id": 12, "group_size": None, "fallback": "W4A16"},
        ],
    }
    bundle = repack_nvfp4_sm90_v3_selected(checkpoint, selection)
    bundle.verify_checksums()
    assert bundle.expert_mapping == (10, 11, 12)
    assert [bucket.manifest.group_size for bucket in bundle.promoted_buckets] == [
        32,
        128,
    ]
    assert [bucket.manifest.expert_mapping for bucket in bundle.promoted_buckets] == [
        (10,),
        (11,),
    ]
    for bucket in bundle.promoted_buckets:
        expert_id = bucket.manifest.expert_mapping[0]
        source_index = checkpoint.expert_mapping.index(expert_id)
        expected = simulate_w4a8(checkpoint, bucket.manifest.group_size, "generic")[
            source_index : source_index + 1
        ]
        torch.testing.assert_close(
            reference_dequantize_nvfp4_sm90_v3_promoted(bucket),
            expected,
            rtol=0,
            atol=0,
        )
    assert bundle.w4a16_fallback is not None
    assert bundle.w4a16_fallback.expert_mapping == (12,)
    torch.testing.assert_close(
        reference_dequantize_nvfp4(bundle.w4a16_fallback),
        reference_dequantize_nvfp4(checkpoint)[2:3],
        rtol=0,
        atol=0,
    )


def test_v3_selected_policy_requires_exact_expert_coverage():
    checkpoint = _checkpoint()
    selection = {
        "mode": "model",
        "residual_scheme": "pow2",
        "group_size": 64,
        "experts": [{"expert_id": 10, "group_size": 64}],
    }
    with pytest.raises(ValueError, match="cover checkpoint experts exactly"):
        repack_nvfp4_sm90_v3_selected(checkpoint, selection)


def test_v3_selected_policy_rejects_lossy_integer_metadata():
    checkpoint = _checkpoint()
    valid = {
        "mode": "model",
        "residual_scheme": "generic",
        "group_size": 32,
        "experts": [
            {"expert_id": 10, "group_size": 32},
            {"expert_id": 11, "group_size": 32},
        ],
    }

    lossy_expert = {
        **valid,
        "experts": [
            {"expert_id": 10.9, "group_size": 32},
            {"expert_id": 11, "group_size": 32},
        ],
    }
    with pytest.raises(TypeError, match="expert_id must be an integer"):
        repack_nvfp4_sm90_v3_selected(checkpoint, lossy_expert)

    lossy_expert_group = {
        **valid,
        "experts": [
            {"expert_id": 10, "group_size": 32.9},
            {"expert_id": 11, "group_size": 32},
        ],
    }
    with pytest.raises(TypeError, match="group_size must be an integer or None"):
        repack_nvfp4_sm90_v3_selected(checkpoint, lossy_expert_group)

    lossy_model_group = {**valid, "group_size": 32.0}
    with pytest.raises(TypeError, match="model selection group_size"):
        repack_nvfp4_sm90_v3_selected(checkpoint, lossy_model_group)

    bundle = repack_nvfp4_sm90_v3_selected(checkpoint, valid)
    with pytest.raises(TypeError, match="expert_mapping entries must be integers"):
        replace(bundle, expert_mapping=(10.9, 11))


def test_v3_manifest_rejects_version_and_field_mismatch():
    view = repack_nvfp4_sm90_v3(_checkpoint(), group_size=32, residual_scheme="pow2")
    wrong_version = view.manifest.to_dict()
    wrong_version["layout_version"] = 2
    with pytest.raises(ValueError, match="layout version mismatch: expected 3"):
        NVFP4V3Manifest.from_dict(wrong_version)
    for field in ("byte_order", "global_layout", "w13_layout"):
        missing_field = view.manifest.to_dict()
        del missing_field[field]
        with pytest.raises(ValueError, match="manifest fields differ"):
            NVFP4V3Manifest.from_dict(missing_field)

    wrong_byte_order = view.manifest.to_dict()
    wrong_byte_order["byte_order"] = "big"
    with pytest.raises(ValueError, match="byte_order must be 'little'"):
        NVFP4V3Manifest.from_dict(wrong_byte_order)

    wrong_global_layout = view.manifest.to_dict()
    wrong_global_layout["global_layout"] = "thread_major"
    with pytest.raises(ValueError, match="global_layout must be"):
        NVFP4V3Manifest.from_dict(wrong_global_layout)

    wrong_w13_layout = view.manifest.to_dict()
    wrong_w13_layout["w13_layout"] = "up_then_gate"
    with pytest.raises(ValueError, match="w13_layout must be 'gate_then_up'"):
        NVFP4V3Manifest.from_dict(wrong_w13_layout)


def test_explicit_v2_to_v3_conversion_and_version_gate():
    checkpoint = _checkpoint(
        physical_n=64,
        physical_k=32,
        logical_n=63,
        logical_k=31,
    )
    v2 = build_nvfp4_rs_weight_view(
        checkpoint.packed_e2m1,
        checkpoint.scale_e4m3_per16,
        checkpoint.global_alpha,
    )
    v3 = convert_nvfp4_rs_v2_to_v3(
        v2,
        source_layout_version=NVFP4_RS_LAYOUT_VERSION,
        logical_shape=checkpoint.logical_shape,
        expert_mapping=checkpoint.expert_mapping,
        source_format_version=checkpoint.source_format_version,
        alpha_scope="per_expert",
        group_size=64,
        residual_scheme="generic",
    )
    restored = unpack_nvfp4_sm90_v3(v3)
    torch.testing.assert_close(
        reference_dequantize_nvfp4(restored),
        reference_dequantize_nvfp4(checkpoint),
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="source layout version mismatch"):
        convert_nvfp4_rs_v2_to_v3(
            v2,
            source_layout_version=NVFP4_SM90_LAYOUT_VERSION,
            logical_shape=checkpoint.logical_shape,
            expert_mapping=checkpoint.expert_mapping,
            source_format_version=checkpoint.source_format_version,
            alpha_scope="per_expert",
            group_size=64,
            residual_scheme="generic",
        )


def test_v2_and_v3_consumers_reject_the_other_layout_view():
    checkpoint = _checkpoint(
        physical_n=64,
        physical_k=32,
        logical_n=63,
        logical_k=31,
    )
    v2 = build_nvfp4_rs_weight_view(
        checkpoint.packed_e2m1,
        checkpoint.scale_e4m3_per16,
        checkpoint.global_alpha,
    )
    v3 = repack_nvfp4_sm90_v3(
        checkpoint,
        group_size=32,
        residual_scheme="generic",
    )

    with pytest.raises(TypeError, match="view must be NVFP4SM90WeightViewV3"):
        unpack_nvfp4_sm90_v3(v2)
    with pytest.raises(ValueError, match="payload fragment shape is invalid"):
        unpack_nvfp4_payload_v2(v3.packed_e2m1)
    with pytest.raises(ValueError, match=r"scales must be E4M3 \[E,Nt,Kt,64\]"):
        unpack_nvfp4_scales_v2(v3.scale_e4m3_per16)
    with pytest.raises(TypeError, match="view must be NVFP4RSWeightView"):
        convert_nvfp4_rs_v2_to_v3(
            v3,
            source_layout_version=NVFP4_RS_LAYOUT_VERSION,
            logical_shape=checkpoint.logical_shape,
            expert_mapping=checkpoint.expert_mapping,
            source_format_version=checkpoint.source_format_version,
            alpha_scope="per_expert",
            group_size=32,
            residual_scheme="generic",
        )


def test_v2_to_v3_requires_explicit_alpha_scope_and_preserves_per_tensor():
    checkpoint = _checkpoint(
        experts=2,
        physical_n=64,
        physical_k=32,
        logical_n=63,
        logical_k=31,
        scalar_alpha=True,
    )
    v2 = build_nvfp4_rs_weight_view(
        checkpoint.packed_e2m1,
        checkpoint.scale_e4m3_per16,
        checkpoint.global_alpha,
    )
    v3 = convert_nvfp4_rs_v2_to_v3(
        v2,
        source_layout_version=NVFP4_RS_LAYOUT_VERSION,
        logical_shape=checkpoint.logical_shape,
        expert_mapping=checkpoint.expert_mapping,
        source_format_version=checkpoint.source_format_version,
        alpha_scope="per_tensor",
        group_size=32,
        residual_scheme="pow2",
    )
    assert v3.manifest.alpha_scope == "per_tensor"
    assert v3.global_alpha.ndim == 0
    assert v3.global_alpha.item() == checkpoint.global_alpha.item()
