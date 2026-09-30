from __future__ import annotations

import pytest
import torch

from b12x.moe.fused_moe.config import (
    ScaleGranularity,
    TrellisConfig,
    TrellisScaleFactorsConfig,
)
from b12x.moe.fused_moe.trellis import (
    _bundle_offsets,
    _coalesce_payloads,
    _effective_input_scales,
    _effective_intermediate_scales,
    _matrix_section_bytes,
    _projection_native,
)
from b12x.moe.fused_moe.weights import ScaleFactors


def _k3_config() -> dict[str, object]:
    return {
        "version": 2,
        "codebook": "lut_e4m3",
        "rate": {"granularity": "uniform"},
        "scale": {
            "input_scales": {"vectors": "per_layer", "gains": "per_layer"},
            "intermediate_scales": {
                "vectors": "per_layer",
                "gains": "per_expert",
            },
            "output_scales": {"vectors": "per_layer", "gains": "per_layer"},
        },
        "transform": {
            "projection": {"kind": "scaled_hadamard", "block_size": 128},
            "expert": {
                "kind": "intermediate_hadamard",
                "pre_block_size": 512,
                "post_block_size": 128,
                "sign_pattern_granularity": "per_expert",
            },
        },
    }


def _glm_config() -> dict[str, object]:
    return {
        "version": 2,
        "codebook": "mcg",
        "rate": {"granularity": "per_expert_projection"},
        "scale": {
            "input_scales": {"vectors": "per_layer", "gains": "none"},
            "intermediate_scales": {
                "vectors": "per_expert",
                "gains": "none",
            },
            "output_scales": {"vectors": "per_layer", "gains": "none"},
        },
        "transform": {
            "projection": {"kind": "scaled_hadamard", "block_size": 128},
            "expert": {"kind": "none"},
        },
    }


@pytest.mark.parametrize("value", [_k3_config(), _glm_config()])
def test_trellis_config_round_trips(value: dict[str, object]) -> None:
    config = TrellisConfig.from_dict(value)
    assert config.to_dict() == value


def test_quantization_config_wrapper_is_strict() -> None:
    value = {
        "quant_method": "b12x_trellis",
        "b12x_trellis": _glm_config(),
    }
    assert TrellisConfig.from_quantization_config(value).to_dict() == _glm_config()

    value["tensor_integrity"] = True
    with pytest.raises(ValueError, match="unknown fields: tensor_integrity"):
        TrellisConfig.from_quantization_config(value)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["rate"].update(structure="uniform"), "structure"),
        (lambda value: value.update(rates_gate="gate"), "rates_gate"),
        (lambda value: value.update(version=1), "expected 2"),
        (
            lambda value: value["scale"].update(scales="flattened"),
            "scales",
        ),
    ],
)
def test_trellis_config_rejects_legacy_or_unknown_fields(mutation, match: str) -> None:
    value = _glm_config()
    mutation(value)
    with pytest.raises(ValueError, match=match):
        TrellisConfig.from_dict(value)


def test_rate_tensor_shapes() -> None:
    value = _glm_config()
    config = TrellisConfig.from_dict(value)
    assert config.rate.tensor_shape(
        num_layers=75,
        num_experts=256,
        intermediate_size=2048,
    ) == (75, 256, 3)

    value["rate"] = {"granularity": "per_expert", "group_size": 256}
    config = TrellisConfig.from_dict(value)
    assert config.rate.tensor_shape(
        num_layers=75,
        num_experts=256,
        intermediate_size=2048,
    ) == (75, 256, 8)


def test_schema_keeps_codebook_rate_and_transform_orthogonal() -> None:
    value = _glm_config()
    value["codebook"] = "mcg"
    value["rate"] = {"granularity": "per_expert", "group_size": 256}
    value["transform"]["expert"] = {
        "kind": "intermediate_hadamard",
        "pre_block_size": 512,
        "post_block_size": 128,
        "sign_pattern_granularity": "per_expert",
    }

    assert TrellisConfig.from_dict(value).to_dict() == value


def test_projection_tier_payloads_share_one_flat_allocation() -> None:
    payloads = (
        torch.arange(12, dtype=torch.int32).reshape(2, 2, 3),
        torch.arange(20, dtype=torch.int32).reshape(1, 4, 5),
        torch.arange(42, dtype=torch.int32).reshape(3, 2, 7),
    )

    combined, views = _coalesce_payloads(payloads)

    assert tuple(combined.shape) == (74,)
    assert all(view.untyped_storage().data_ptr() == combined.data_ptr() for view in views)
    for actual, expected in zip(views, payloads, strict=True):
        assert tuple(actual.shape) == tuple(expected.shape)
        torch.testing.assert_close(actual, expected)


def test_projection_payload_supports_tp4_512_channel_extent() -> None:
    hidden_size = 128
    num_slots = 512 // 32
    bits = torch.tensor(((3, 4, 5), (4, 3, 5)), dtype=torch.int64)
    offsets = _bundle_offsets(bits, hidden_size)
    row_stride = offsets[-1][-1] + _matrix_section_bytes(
        hidden_size, int(bits[-1, -1])
    )
    codes = (
        torch.arange(num_slots * row_stride, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(num_slots, row_stride)
    )

    gate_k3 = _projection_native(
        codes,
        experts=[0],
        projection=0,
        bits=3,
        offsets=offsets,
        hidden_size=hidden_size,
        fc1=True,
    )
    down_k5 = _projection_native(
        codes,
        experts=[0, 1],
        projection=2,
        bits=5,
        offsets=offsets,
        hidden_size=hidden_size,
        fc1=False,
    )

    assert tuple(gate_k3.shape) == (1, 8, 32, 48)
    assert tuple(down_k5.shape) == (2, 32, 8, 80)


def test_selected_per_layer_scalar_gains_broadcast() -> None:
    declaration = TrellisScaleFactorsConfig(
        vectors=ScaleGranularity.PER_LAYER,
        gains=ScaleGranularity.PER_LAYER,
    )
    device = torch.device("cpu")
    gate, up = _effective_input_scales(
        ScaleFactors(
            vectors=torch.ones(8, dtype=torch.float16),
            gains=torch.tensor(2.0, dtype=torch.float16),
        ),
        declaration,
        num_experts=3,
        hidden_size=8,
        device=device,
    )
    intermediate = _effective_intermediate_scales(
        ScaleFactors(
            vectors=torch.ones((3, 16), dtype=torch.float16),
            gains=torch.tensor(0.5, dtype=torch.float16),
        ),
        declaration,
        num_experts=3,
        intermediate_size=16,
        device=device,
    )

    torch.testing.assert_close(gate, torch.full((1, 8), 2.0, dtype=torch.float16))
    torch.testing.assert_close(up, gate)
    torch.testing.assert_close(
        intermediate,
        torch.full((3, 3, 16), 0.5, dtype=torch.float16),
    )
