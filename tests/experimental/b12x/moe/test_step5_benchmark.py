"""Step-5 benchmark loading preserves stacked expert shards and calibration."""

import json

import pytest
import torch
from safetensors.torch import save_file

from benchmarks.experimental.b12x.benchmark_moe import (
    MODEL_PROFILES,
    build_model_spec,
    load_expert_weights,
)
from b12x.moe._shared.kernels.reference import unswizzle_block_scale

from ..conftest import require_b12x


@pytest.fixture
def checkpoint(tmp_path):
    e, h, n = 4, 128, 256
    text = dict(
        model_type="step5",
        hidden_size=h,
        moe_intermediate_size=n,
        moe_num_experts=e,
        moe_top_k=2,
        moe_layer_list=[3],
        moe_router_scaling_factor=3.0,
        norm_expert_weight=True,
    )
    quant = dict(quant_method="modelopt", quant_algo="NVFP4", group_size=16, ignore=[])
    (tmp_path / "config.json").write_text(
        json.dumps(dict(text_config=text, quantization_config=quant))
    )
    tensors = {}
    prefix = "model.layers.3.moe"
    for index, (projection, rows, columns) in enumerate(
        (("up_proj", n, h), ("gate_proj", n, h), ("down_proj", h, n))
    ):
        packed = torch.arange(e * rows * columns // 2).reshape(e, rows, columns // 2)
        tensors[f"{prefix}.{projection}.weight"] = (packed + index * 17).to(torch.uint8)
        scales = torch.arange(e * rows * columns // 16).reshape(e, rows, columns // 16)
        tensors[f"{prefix}.{projection}.weight_scale"] = (
            (scales % 31 + 1 + index).float().to(torch.float8_e4m3fn)
        )
        tensors[f"{prefix}.{projection}.input_scale"] = (
            torch.arange(1, e + 1).float() * 0.25
        )
        tensors[f"{prefix}.{projection}.weight_scale_2"] = (
            torch.arange(1, e + 1).float() * 0.125
        )
    tensors[f"{prefix}.gate.weight"] = torch.randn(e, h)
    tensors[f"{prefix}.router_bias"] = torch.randn(e)
    save_file(tensors, tmp_path / "experts.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "experts.safetensors" for key in tensors}})
    )
    return tmp_path, tensors


@pytest.mark.parametrize("rank", [0, 1])
def test_stacked_tp_weights_and_scales(checkpoint, rank):
    require_b12x()
    path, tensors = checkpoint
    profile = MODEL_PROFILES["step5-preview-nvfp4"]
    spec = build_model_spec(path, profile, tp_size_override=2, tp_rank=rank)
    loaded = load_expert_weights(path, spec, layer_idx=3, checkpoint_family="step5")
    interval = slice(rank * 128, (rank + 1) * 128)
    prefix = "model.layers.3.moe"
    expected13 = torch.cat(
        [
            tensors[f"{prefix}.{p}.weight"][:, interval]
            for p in ("up_proj", "gate_proj")
        ],
        dim=1,
    )
    torch.testing.assert_close(loaded.w13_weight.cpu(), expected13, rtol=0, atol=0)
    torch.testing.assert_close(
        loaded.w2_weight.cpu(),
        tensors[f"{prefix}.down_proj.weight"][:, :, rank * 64 : (rank + 1) * 64],
        rtol=0,
        atol=0,
    )
    for expert in range(4):
        expected_sf13 = torch.cat(
            [
                tensors[f"{prefix}.{p}.weight_scale"][expert, interval].float()
                for p in ("up_proj", "gate_proj")
            ],
            dim=0,
        )
        expected_sf2 = tensors[f"{prefix}.down_proj.weight_scale"][
            expert, :, rank * 8 : (rank + 1) * 8
        ].float()
        for actual, expected in (
            (loaded.w13_blockscale_swizzled[expert], expected_sf13),
            (loaded.w2_blockscale_swizzled[expert], expected_sf2),
        ):
            restored = unswizzle_block_scale(actual.view(torch.uint8), *expected.shape)
            torch.testing.assert_close(restored.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        loaded.w13_input_scale_per_expert.cpu(),
        tensors[f"{prefix}.up_proj.input_scale"],
        rtol=0,
        atol=0,
    )
    assert loaded.w13_layout == "w13"
    assert loaded.gate_score_func == "sigmoid" and loaded.gate_route_scale == 3.0


def test_rejects_incompatible_calibration_and_unquantized_layer(checkpoint):
    path, tensors = checkpoint
    spec = build_model_spec(
        path, MODEL_PROFILES["step5-preview-nvfp4"], tp_size_override=2
    )
    tensors["model.layers.3.moe.up_proj.weight_scale_2"] *= 2
    save_file(tensors, path / "experts.safetensors")
    with pytest.raises(ValueError, match="fusion would change calibration"):
        load_expert_weights(path, spec, layer_idx=3, checkpoint_family="step5")
    config = json.loads((path / "config.json").read_text())
    config["quantization_config"]["ignore"] = ["*layers.3.moe.experts*"]
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unquantized experts"):
        load_expert_weights(path, spec, layer_idx=3, checkpoint_family="step5")


def test_rejects_invalid_layer_and_tp_geometry(checkpoint):
    path, _ = checkpoint
    profile = MODEL_PROFILES["step5-preview-nvfp4"]
    with pytest.raises(ValueError, match="routed-expert layer"):
        build_model_spec(path, profile, layer_idx=0)
    for tp, rank in ((0, 0), (3, 0), (2, 2), (16, 0)):
        with pytest.raises(ValueError, match="valid TP rank"):
            build_model_spec(path, profile, tp_size_override=tp, tp_rank=rank)
