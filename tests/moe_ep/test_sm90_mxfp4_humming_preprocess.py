"""Permanent reference/golden tests for SM90 Humming MXFP4 preprocessing.

The oracle is owned by this target repository and has no donor-path import.
Host tests cover bit-level scale, payload, fold and physical-interleave
contracts.  Hopper-only tests additionally compare the production CUDA path.
"""

from __future__ import annotations

import hashlib

import pytest


torch = pytest.importorskip("torch")

from flashinfer.fused_moe.prepare import (  # noqa: E402
    _humming_mxfp4_w4a8_rewrite_lut_cpu,
    interleave_moe_scales_for_sm90_mixed_gemm,
    preprocess_moe_weights_for_sm90_mixed_gemm_humming,
)
from tests.moe_ep._sm90_mxfp4_humming_reference import (  # noqa: E402
    reference_fold_offsets,
    reference_interleave_weight,
    reference_payload_rewrite_lut,
    reference_preprocess,
    reference_scale_factorization,
    reference_unfold_offsets,
)


EDGE_SPANS = (0, 1, 11, 12, 14, 127, 255)


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    assert codes.shape[-1] % 2 == 0
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def _edge_case() -> tuple[torch.Tensor, torch.Tensor]:
    """Return seven experts whose exponent spans exercise every clamp edge."""

    experts = len(EDGE_SPANS)
    rows = 64
    logical_k = 128
    codes = torch.arange(experts * rows * logical_k, dtype=torch.int64)
    codes = codes.remainder(16).to(torch.uint8).reshape(experts, rows, logical_k)
    weight = _pack_codes(codes)

    elements_per_expert = rows * (logical_k // 32)
    raw_scale = torch.empty((experts, rows, logical_k // 32), dtype=torch.uint8)
    for expert, span in enumerate(EDGE_SPANS):
        values = torch.arange(elements_per_expert, dtype=torch.int64)
        values = values.remainder(span + 1).to(torch.uint8)
        raw_scale[expert] = values.reshape(rows, logical_k // 32)
        assert int(raw_scale[expert].amin()) == 0
        assert int(raw_scale[expert].amax()) == span
    return weight, raw_scale


@pytest.mark.parametrize("span", EDGE_SPANS)
def test_scale_clamp_edge_spans(span: int) -> None:
    raw_scale = torch.tensor([[[0, span]]], dtype=torch.uint8)
    offset, residual, delta = reference_scale_factorization(raw_scale)

    retained_span = min(span, 11)
    base = span - retained_span
    assert torch.equal(
        offset,
        torch.tensor([[[1, retained_span + 1]]], dtype=torch.uint8),
    )
    assert torch.equal(
        delta,
        torch.tensor([[[base, 0]]], dtype=torch.uint8),
    )
    assert torch.equal(
        residual,
        torch.tensor([2.0 ** (base - 128)], dtype=torch.float32),
    )
    assert 1 <= int(offset.amin()) <= int(offset.amax()) <= 12


def test_scale_clamp_and_residual_multi_expert_golden() -> None:
    raw_scale = torch.tensor(
        [
            [[128, 128, 128, 128]],
            [[114, 128, 120, 127]],
            [[0, 255, 244, 250]],
        ],
        dtype=torch.uint8,
    )
    offset, residual, delta = reference_scale_factorization(raw_scale)

    assert torch.equal(
        offset,
        torch.tensor(
            [
                [[1, 1, 1, 1]],
                [[1, 12, 4, 11]],
                [[1, 12, 1, 7]],
            ],
            dtype=torch.uint8,
        ),
    )
    assert torch.equal(
        delta,
        torch.tensor(
            [
                [[0, 0, 0, 0]],
                [[3, 0, 0, 0]],
                [[244, 0, 0, 0]],
            ],
            dtype=torch.uint8,
        ),
    )
    assert torch.equal(
        residual,
        torch.tensor([1.0, 2.0**-11, 2.0**116], dtype=torch.float32),
    )


def test_payload_rewrite_lut_all_codes_negative_zero_and_golden() -> None:
    target_lut = _humming_mxfp4_w4a8_rewrite_lut_cpu()
    reference_lut = reference_payload_rewrite_lut()
    assert target_lut.shape == (256, 16)
    assert torch.equal(target_lut, reference_lut)

    golden_rows = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15],
        1: [0, 1, 1, 2, 2, 3, 4, 5, 0, 9, 9, 10, 10, 11, 12, 13],
        2: [0, 0, 1, 1, 1, 2, 2, 3, 0, 8, 9, 9, 9, 10, 10, 11],
        4: [0, 0, 0, 0, 0, 0, 1, 1, 0, 8, 8, 8, 8, 8, 9, 9],
        5: [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8],
        # A 0..255 exponent span clamps the low block with delta=244.  The
        # golden preserves Humming's 32-bit exponent-field construction.
        244: [8, 8, 10, 11, 12, 13, 14, 15, 8, 0, 2, 3, 4, 5, 6, 7],
    }
    for delta, expected in golden_rows.items():
        assert torch.equal(target_lut[delta], torch.tensor(expected, dtype=torch.uint8))

    # E2M1 code 8 is negative zero and is canonicalized before any rewrite.
    assert int(target_lut[0, 8]) == 0
    assert int(target_lut[1, 8]) == 0


def test_offset_fold_unfold_shape_and_golden() -> None:
    offset = torch.arange(256, dtype=torch.int64).to(torch.uint8).reshape(1, 64, 4)
    folded = interleave_moe_scales_for_sm90_mixed_gemm(offset, group_size=32)
    expected = reference_fold_offsets(offset)

    assert tuple(folded.shape) == (1, 1, 1, 16, 16)
    assert folded.is_contiguous()
    assert torch.equal(folded, expected)
    assert torch.equal(reference_unfold_offsets(folded), offset)
    assert torch.equal(
        folded[0, 0, 0, 0],
        torch.tensor(
            [
                0,
                1,
                2,
                3,
                64,
                65,
                66,
                67,
                128,
                129,
                130,
                131,
                192,
                193,
                194,
                195,
            ],
            dtype=torch.uint8,
        ),
    )


def test_offset_fold_unfold_multiple_experts_and_blocks() -> None:
    values = torch.arange(2 * 128 * 8, dtype=torch.int64)
    offset = values.remainder(12).add(1).to(torch.uint8).reshape(2, 128, 8)
    actual = interleave_moe_scales_for_sm90_mixed_gemm(offset, group_size=32)
    expected = reference_fold_offsets(offset)

    assert tuple(actual.shape) == (2, 2, 2, 16, 16)
    assert torch.equal(actual, expected)
    assert torch.equal(reference_unfold_offsets(actual), offset)


def test_weight_physical_interleave_fixed_golden() -> None:
    values = torch.arange(16 * 32, dtype=torch.int64).mul(37).add(11)
    weight = values.remainder(256).to(torch.uint8).reshape(1, 16, 32)
    interleaved = reference_interleave_weight(weight)

    digest = hashlib.sha256(bytes(interleaved.flatten().tolist())).hexdigest()
    assert digest == "3977b26a594a789772e8c2d481e246633b8b388829e22c5504ace579ad407700"
    assert interleaved[0, 0].tolist() == [
        139,
        48,
        3,
        48,
        85,
        114,
        221,
        114,
        159,
        204,
        23,
        204,
        233,
        142,
        233,
        6,
        219,
        0,
        83,
        136,
        37,
        202,
        173,
        202,
        239,
        156,
        103,
        20,
        185,
        86,
        185,
        86,
    ]


@pytest.mark.arch_hopper
def test_target_logical_preprocess_matches_independent_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the target Humming preprocessing")

    weight, raw_scale = _edge_case()
    expected_weight, expected_offset, expected_residual = reference_preprocess(
        weight, raw_scale, interleave=False
    )
    actual_weight, actual_offset, actual_residual = (
        preprocess_moe_weights_for_sm90_mixed_gemm_humming(
            weight.cuda(), raw_scale.cuda(), interleave=False
        )
    )

    assert torch.equal(actual_weight.cpu(), expected_weight)
    assert torch.equal(actual_offset.cpu(), expected_offset)
    assert torch.equal(actual_residual.cpu(), expected_residual)


@pytest.mark.arch_hopper
def test_target_physical_interleave_matches_independent_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the target SM90 weight interleave")

    weight, raw_scale = _edge_case()
    expected_weight, expected_offset, expected_residual = reference_preprocess(
        weight, raw_scale, interleave=True
    )
    actual_weight, actual_offset, actual_residual = (
        preprocess_moe_weights_for_sm90_mixed_gemm_humming(
            weight.cuda(), raw_scale.cuda(), interleave=True
        )
    )

    assert tuple(actual_offset.shape) == (
        len(EDGE_SPANS),
        1,
        1,
        16,
        16,
    )
    assert torch.equal(actual_weight.cpu(), expected_weight)
    assert torch.equal(actual_offset.cpu(), expected_offset)
    assert torch.equal(actual_residual.cpu(), expected_residual)
