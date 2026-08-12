"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pytest
import torch

from flashinfer.fused_moe.nvfp4_checkpoint import (
    NVFP4Checkpoint,
    reference_dequantize_nvfp4,
)
from flashinfer.fused_moe.sm90_nvfp4_repack import (
    NVFP4_RS_LAYOUT_VERSION,
    NVFP4_SM90_LAYOUT_VERSION,
    NVFP4V3Manifest,
    build_nvfp4_rs_weight_view,
    convert_nvfp4_rs_v2_to_v3,
    repack_nvfp4_payload_v2,
    repack_nvfp4_scales_v2,
    repack_nvfp4_sm90_v3,
    unpack_nvfp4_payload_v2,
    unpack_nvfp4_scales_v2,
    unpack_nvfp4_sm90_v3,
)
from tests.moe.nvfp4_repack_v2_spec import (
    LAYOUT_VERSION,
    fragment_element_coordinate,
    repack_payload,
    repack_scales,
    thread_scale_rows,
    unpack_payload,
    unpack_scales,
)


def _e4m3_bytes(shape):
    return (
        torch.arange(np.prod(shape), dtype=torch.int64)
        .remainder(255)
        .to(torch.uint8)
        .reshape(shape)
    )


def _unique_coordinate_codes(rows, logical_k):
    coordinate = np.arange(rows * logical_k, dtype=np.uint16).reshape(rows, logical_k)
    digits = np.stack(
        [((coordinate >> (4 * digit)) & 0xF).astype(np.uint8) for digit in range(3)]
    )
    return digits[:, :, 0::2] | (digits[:, :, 1::2] << 4)


def test_nvfp4_repack_v2_ptx_fragment_coordinates():
    seen = set()
    for thread in range(128):
        warp, lane = divmod(thread, 32)
        lane_row, lane_col = divmod(lane, 4)
        scale_rows = thread_scale_rows(thread)
        assert scale_rows == (16 * warp + lane_row, 16 * warp + lane_row + 8)
        for register in range(4):
            for element in range(2):
                row, k = fragment_element_coordinate(thread, register, element)
                assert row == 16 * warp + lane_row + 8 * (register % 2)
                assert k == 2 * lane_col + 8 * (register // 2) + element
                assert row == scale_rows[register % 2]
                seen.add((row, k))
    assert seen == {(row, k) for row in range(64) for k in range(16)}
    assert [
        tuple(fragment_element_coordinate(0, register, element) for element in range(2))
        for register in range(4)
    ] == [
        ((0, 0), (0, 1)),
        ((8, 0), (8, 1)),
        ((0, 8), (0, 9)),
        ((8, 8), (8, 9)),
    ]
    assert [
        tuple(
            fragment_element_coordinate(127, register, element) for element in range(2)
        )
        for register in range(4)
    ] == [
        ((55, 6), (55, 7)),
        ((63, 6), (63, 7)),
        ((55, 14), (55, 15)),
        ((63, 14), (63, 15)),
    ]


def test_nvfp4_repack_v2_exact_fragment_ownership():
    rows, logical_k = 128, 32
    payload = _unique_coordinate_codes(rows, logical_k)
    expected = repack_payload(payload)
    actual = repack_nvfp4_payload_v2(torch.from_numpy(payload.copy())).numpy()
    assert np.array_equal(actual, expected)
    decoded = np.zeros((*actual.shape[1:], 2), dtype=np.uint16)
    for digit in range(3):
        fragment = actual[digit].astype(np.uint16)
        decoded[..., 0] |= (fragment & 0xF) << (4 * digit)
        decoded[..., 1] |= (fragment >> 4) << (4 * digit)
    for n_tile in range(rows // 64):
        for k_tile in range(logical_k // 16):
            for thread in range(128):
                for register in range(4):
                    for element in range(2):
                        row, k = fragment_element_coordinate(thread, register, element)
                        expected_coordinate = (
                            (n_tile * 64 + row) * logical_k + k_tile * 16 + k
                        )
                        assert (
                            decoded[n_tile, k_tile, thread, register, element]
                            == expected_coordinate
                        )


@pytest.mark.parametrize("shape", [(1, 64, 8), (2, 128, 32)])
def test_nvfp4_repack_v2_payload_matches_numpy(shape):
    payload = (
        torch.arange(np.prod(shape), dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(shape)
    )
    expected = repack_payload(payload.numpy())
    actual = repack_nvfp4_payload_v2(payload)
    assert actual.shape == expected.shape
    assert np.array_equal(actual.numpy(), expected)
    assert torch.equal(unpack_nvfp4_payload_v2(actual), payload)
    assert np.array_equal(unpack_payload(expected), payload.numpy())


@pytest.mark.parametrize("shape", [(1, 64, 1), (2, 128, 4)])
def test_nvfp4_repack_v2_scales_match_numpy(shape):
    raw = _e4m3_bytes(shape)
    scales = raw.view(torch.float8_e4m3fn)
    expected = repack_scales(raw.numpy())
    actual = repack_nvfp4_scales_v2(scales)
    assert actual.shape == expected.shape
    assert np.array_equal(actual.view(torch.uint8).numpy(), expected)
    assert torch.equal(
        unpack_nvfp4_scales_v2(actual).view(torch.uint8),
        raw,
    )
    assert np.array_equal(unpack_scales(expected), raw.numpy())


def test_nvfp4_repack_v2_scale_rows_match_fragment_owners():
    shape = (2, 128, 3)
    raw = (
        ((np.arange(np.prod(shape), dtype=np.uint32) * 37 + 11) % 127)
        .astype(np.uint8)
        .reshape(shape)
    )
    scales = torch.from_numpy(raw.copy()).view(torch.float8_e4m3fn)
    expected = repack_scales(raw)
    actual = repack_nvfp4_scales_v2(scales).view(torch.uint8).numpy()
    assert np.array_equal(actual, expected)
    for expert in range(shape[0]):
        for n_tile in range(shape[1] // 64):
            for k_tile in range(shape[2]):
                for thread in range(128):
                    rows = thread_scale_rows(thread)
                    for register in range(4):
                        row, _ = fragment_element_coordinate(thread, register, 0)
                        scale_row = rows[register % 2]
                        assert row == scale_row
                        assert (
                            actual[expert, n_tile, k_tile, scale_row]
                            == raw[expert, n_tile * 64 + row, k_tile]
                        )


def test_nvfp4_repack_v2_view_contract():
    payload = torch.zeros(2, 64, 16, dtype=torch.uint8)
    scales = torch.ones(2, 64, 2).to(torch.float8_e4m3fn)
    alpha = torch.tensor([0.5, 1.0], dtype=torch.float32)
    view = build_nvfp4_rs_weight_view(payload, scales, alpha)
    assert NVFP4_RS_LAYOUT_VERSION == LAYOUT_VERSION == 2
    assert view.payload.shape == (2, 1, 2, 128, 4)
    assert view.scales.shape == (2, 1, 2, 64)
    assert view.alpha.data_ptr() == alpha.data_ptr()


def test_nvfp4_repack_v2_rejects_bad_shapes():
    with pytest.raises(ValueError, match="tile alignment"):
        repack_nvfp4_payload_v2(torch.zeros(1, 32, 8, dtype=torch.uint8))
    with pytest.raises(ValueError, match="tile alignment"):
        repack_payload(np.zeros((1, 32, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="fragment shape"):
        unpack_nvfp4_payload_v2(torch.zeros(1, 1, 1, 64, 8, dtype=torch.uint8))
    with pytest.raises(ValueError, match="fragment shape"):
        unpack_payload(np.zeros((1, 1, 1, 64, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="divisible by 64"):
        repack_nvfp4_scales_v2(torch.ones(1, 32, 1).to(torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="tile width"):
        unpack_nvfp4_scales_v2(torch.ones(1, 1, 1, 32).to(torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="tile width"):
        unpack_scales(np.zeros((1, 1, 1, 32), dtype=np.uint8))


def _nvfp4_v3_checkpoint() -> NVFP4Checkpoint:
    payload = (
        torch.arange(2 * 65 * 24, dtype=torch.int64)
        .mul(37)
        .remainder(256)
        .to(torch.uint8)
        .reshape(2, 65, 24)
    )
    scales = (
        torch.arange(2 * 65 * 3, dtype=torch.int64)
        .remainder(17)
        .to(torch.float32)
        .reshape(2, 65, 3)
        .to(torch.float8_e4m3fn)
    )
    return NVFP4Checkpoint(
        payload,
        scales,
        torch.tensor((0.5, 0.75), dtype=torch.float32),
        (2, 65, 47),
        (7, 11),
        "modelopt.nvfp4.test",
    )


@pytest.mark.parametrize("group_size", (32, 64, 128))
@pytest.mark.parametrize("residual_scheme", ("generic", "pow2"))
def test_nvfp4_repack_v3_roundtrip_and_manifest_contract(group_size, residual_scheme):
    checkpoint = _nvfp4_v3_checkpoint()
    view = repack_nvfp4_sm90_v3(
        checkpoint,
        group_size=group_size,
        residual_scheme=residual_scheme,
    )
    restored = unpack_nvfp4_sm90_v3(view)
    assert view.manifest.layout_version == NVFP4_SM90_LAYOUT_VERSION == 3
    assert set(view.manifest.to_dict()) == {
        "layout_version",
        "source_format_version",
        "sm_target",
        "group_size",
        "residual_scheme",
        "rounding_mode",
        "logical_shape",
        "padded_shape",
        "nibble_order",
        "byte_order",
        "global_layout",
        "w13_layout",
        "alpha_scope",
        "expert_mapping",
        "checksums",
    }
    torch.testing.assert_close(
        reference_dequantize_nvfp4(restored),
        reference_dequantize_nvfp4(checkpoint),
        rtol=0,
        atol=0,
    )


def test_nvfp4_repack_v3_manifest_tampering_is_rejected():
    view = repack_nvfp4_sm90_v3(
        _nvfp4_v3_checkpoint(),
        group_size=64,
        residual_scheme="generic",
    )
    lossy = view.manifest.to_dict()
    lossy["expert_mapping"] = [7, 11.9]
    with pytest.raises(TypeError, match="expert_mapping entries"):
        NVFP4V3Manifest.from_dict(lossy)

    rebound = view.manifest.to_dict()
    rebound["sm_target"] = "sm90"
    tampered_manifest = NVFP4V3Manifest.from_dict(rebound)
    tampered_view = type(view)(
        view.packed_e2m1,
        view.scale_e4m3_per16,
        view.promotion_group_scale,
        view.promotion_residual,
        view.global_alpha,
        tampered_manifest,
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        unpack_nvfp4_sm90_v3(tampered_view)


def test_nvfp4_repack_explicit_v2_to_v3_conversion():
    payload = torch.arange(64 * 16, dtype=torch.int64).remainder(256).to(torch.uint8)
    payload = payload.reshape(1, 64, 16)
    scales = torch.ones((1, 64, 2), dtype=torch.float32).to(torch.float8_e4m3fn)
    alpha = torch.tensor((0.5,), dtype=torch.float32)
    source = build_nvfp4_rs_weight_view(payload, scales, alpha)
    converted = convert_nvfp4_rs_v2_to_v3(
        source,
        source_layout_version=NVFP4_RS_LAYOUT_VERSION,
        logical_shape=(1, 63, 31),
        expert_mapping=(3,),
        source_format_version="modelopt.nvfp4.test",
        alpha_scope="per_expert",
        group_size=64,
        residual_scheme="pow2",
    )
    assert converted.manifest.layout_version == 3
    assert converted.manifest.source_format_version == "modelopt.nvfp4.test"
    with pytest.raises(ValueError, match="source layout version mismatch"):
        convert_nvfp4_rs_v2_to_v3(
            source,
            source_layout_version=NVFP4_SM90_LAYOUT_VERSION,
            logical_shape=(1, 63, 31),
            expert_mapping=(3,),
            source_format_version="modelopt.nvfp4.test",
            alpha_scope="per_expert",
            group_size=64,
            residual_scheme="pow2",
        )
