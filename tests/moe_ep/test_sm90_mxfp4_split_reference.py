"""CPU tests for the independent split handoff and routing oracle."""

from __future__ import annotations

import pytest
import torch

from tests.moe_ep._sm90_mxfp4_split_reference import (
    K64,
    dequantize_handoff_reference,
    make_routing_case,
    pack_route_metadata,
    quantize_handoff_reference,
    unpack_route_metadata,
    validate_route_indexed_handoff,
)


def test_k64_handoff_reference_preserves_exact_bytes_and_fp32_scales() -> None:
    values = torch.linspace(-17.0, 19.0, 3 * 128, dtype=torch.float32).reshape(3, 128)
    payload, scale = quantize_handoff_reference(values)
    repeat_payload, repeat_scale = quantize_handoff_reference(values)

    assert payload.dtype is torch.float8_e4m3fn
    assert scale.dtype is torch.float32
    assert payload.shape == values.shape
    assert scale.shape == (3, 2)
    assert torch.equal(payload.view(torch.uint8), repeat_payload.view(torch.uint8))
    torch.testing.assert_close(scale, repeat_scale, atol=0.0, rtol=0.0)
    dequant = dequantize_handoff_reference(payload, scale)
    assert torch.isfinite(dequant).all()
    error = dequant - values
    assert error.norm() / values.norm() < 3.0e-2
    assert error.abs().amax() / values.abs().amax() < 7.0e-2


def test_zero_handoff_uses_nonzero_epsilon_scale_and_zero_payload() -> None:
    payload, scale = quantize_handoff_reference(torch.zeros(2, 2 * K64))
    assert torch.count_nonzero(payload.view(torch.uint8)) == 0
    assert (scale > 0).all()
    assert torch.isfinite(scale).all()


def test_route_metadata_round_trip_uses_the_production_bit_fields() -> None:
    packed = torch.tensor([pack_route_metadata(3, 0x12345678, 5)], dtype=torch.int64)
    rank, token, topk = unpack_route_metadata(packed.view(torch.uint8).reshape(1, 8))
    assert (rank.item(), token.item(), topk.item()) == (3, 0x12345678, 5)


@pytest.mark.parametrize(
    "case", ["balanced", "skewed", "remote_heavy", "masked", "edge"]
)
def test_routing_cases_have_expected_shape_and_valid_masks(case: str) -> None:
    ids, weights = make_routing_case(
        case=case,
        rank=1,
        world_size=4,
        num_tokens=7,
        top_k=6,
        local_experts=4,
    )
    assert ids.shape == weights.shape == (7, 6)
    assert ((ids >= 0) | (ids == -1)).all()
    assert (weights[ids == -1] == 0).all()
    if case == "remote_heavy":
        assert ((ids // 4) != 1).to(torch.float32).mean() >= 0.75
    if case == "skewed":
        assert (ids == 0).to(torch.float32).mean() >= 0.75
    if case == "masked":
        assert (ids == -1).any()


def _physical_fixture():
    world_size, tokens, top_k, _local_experts, width = 2, 2, 1, 2, 64
    ids = torch.tensor([[[2], [3]], [[3], [2]]], dtype=torch.int64)
    values = (
        torch.arange(world_size * tokens * top_k * width, dtype=torch.float32)
        .reshape(world_size, tokens, top_k, width)
        .remainder(17)
        .sub(8)
    )
    route_q, route_scale = quantize_handoff_reference(values)
    pool_rows = 8
    actual_q = torch.zeros(pool_rows, width, dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    actual_scale = torch.zeros(pool_rows, 1, dtype=torch.float32)
    metadata = torch.zeros(pool_rows, 8, dtype=torch.uint8)
    physical_routes = {
        0: ((0, 0, 0), (1, 1, 0)),
        1: ((0, 1, 0), (1, 0, 0)),
    }
    for expert, routes in physical_routes.items():
        base = expert * 4
        for offset, (source_rank, source_token, source_topk) in enumerate(routes):
            row = base + offset
            packed = pack_route_metadata(source_rank, source_token, source_topk)
            metadata[row].copy_(
                torch.tensor([packed], dtype=torch.int64).view(torch.uint8)
            )
            actual_q[row].copy_(route_q[source_rank, source_token, source_topk])
            actual_scale[row].copy_(route_scale[source_rank, source_token, source_topk])
    return ids, route_q, route_scale, actual_q, actual_scale, metadata


def test_route_indexed_validator_checks_every_valid_physical_row() -> None:
    ids, route_q, route_scale, actual_q, actual_scale, metadata = _physical_fixture()
    result = validate_route_indexed_handoff(
        actual_payload=actual_q,
        actual_scale=actual_scale,
        actual_metadata=metadata,
        valid_counts=[2, 2],
        route_payload=route_q,
        route_scale=route_scale,
        global_topk_idx=ids,
        target_rank=1,
        local_experts=2,
        token_padding_block=4,
    )
    assert result.valid_rows == 4
    assert result.experts_with_routes == 2
    assert result.e4m3_values == 4 * 64
    assert result.fp32_scales == 4


@pytest.mark.parametrize("corruption", ["payload", "scale", "metadata"])
def test_route_indexed_validator_fails_closed_on_any_handoff_corruption(
    corruption: str,
) -> None:
    ids, route_q, route_scale, actual_q, actual_scale, metadata = _physical_fixture()
    if corruption == "payload":
        actual_q.view(torch.uint8)[0, 0] ^= 1
    elif corruption == "scale":
        actual_scale.view(torch.int32)[0, 0] ^= 1
    else:
        metadata[0].copy_(
            torch.tensor([pack_route_metadata(0, 1, 0)], dtype=torch.int64).view(
                torch.uint8
            )
        )
    with pytest.raises((AssertionError, RuntimeError)):
        validate_route_indexed_handoff(
            actual_payload=actual_q,
            actual_scale=actual_scale,
            actual_metadata=metadata,
            valid_counts=[2, 2],
            route_payload=route_q,
            route_scale=route_scale,
            global_topk_idx=ids,
            target_rank=1,
            local_experts=2,
            token_padding_block=4,
        )


def test_route_indexed_validator_rejects_duplicate_route_and_omission() -> None:
    ids, route_q, route_scale, actual_q, actual_scale, metadata = _physical_fixture()

    # Rows 0 and 1 belong to the same expert. Duplicate row 0 wholesale so
    # payload and scale remain self-consistent while row 1's route is omitted.
    metadata[1].copy_(metadata[0])
    actual_q[1].copy_(actual_q[0])
    actual_scale[1].copy_(actual_scale[0])

    with pytest.raises(AssertionError, match="duplicate handoff route metadata"):
        validate_route_indexed_handoff(
            actual_payload=actual_q,
            actual_scale=actual_scale,
            actual_metadata=metadata,
            valid_counts=[2, 2],
            route_payload=route_q,
            route_scale=route_scale,
            global_topk_idx=ids,
            target_rank=1,
            local_experts=2,
            token_padding_block=4,
        )
