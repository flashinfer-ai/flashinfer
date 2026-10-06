"""Block codecs expose real, resource-valid fused-MoE launch choices."""
from dataclasses import replace
from itertools import product

import pytest

from b12x.preparation import DeviceIdentity, FrozenMapping
from b12x.moe.fused_moe._tuning import MoeDecodeConfig, TUNING
from .test_precision_choices import _nvfp4_query


DEVICE = DeviceIdentity("nvidia", (12, 1), 48, "NVIDIA GB10")


def query(codec="iq2_xxs", capacity=256):
    return replace(
        _nvfp4_query(), quant_mode="w4a16", quant_modes=("w4a16",),
        source_format=codec, activation="relu2", num_experts=256,
        hidden_size=1024, intermediate_size=2048, top_k=9,
        num_tokens=capacity, routed_rows=capacity * 9, route_num_experts=256,
        weight_layouts=(codec,), w4a16_weight_layout=codec,
        w4a16_scale_format=codec, w13_layout="w31", shared_input_scales=False,
    )


@pytest.mark.parametrize("codec", ("iq2_xs", "iq2_xxs", "q8_0"))
@pytest.mark.parametrize("capacity", (1, 8, 16, 256))
def test_candidates_expose_tiles_routes_and_pipeline(codec, capacity):
    q = query(codec, capacity)
    configs = [c for _, c in TUNING.eligible_plan(q, DEVICE).candidates]
    assert TUNING.configure(q, device=DEVICE).default in configs
    assert {c.w4a16_route_mode for c in configs} == (
        {"direct", "packed"} if capacity <= 8 else {"packed"}
    )
    explicit = [c for c in configs if c.w4a16_tile_config is not None]
    assert {c.w4a16_block_size_m for c in explicit} == {8, 16, 32, 48, 64}
    assert {c.w4a16_pipeline_stages for c in explicit} == {2, 3, 4, 5}
    tiles = ((128, 64), (64, 128), (128, 128), (64, 256))
    assert {c.w4a16_tile_config for c in explicit} == {
        a + b for a, b in product(tiles, repeat=2) if a[0] * a[1] == b[0] * b[1]
    }
    for c in configs:
        assert MoeDecodeConfig.from_config(FrozenMapping(c.to_dict())) == c
        if c.w4a16_route_mode == "direct":
            assert c.w4a16_block_size_m in (None, 8)


@pytest.mark.parametrize("codec", ("iq2_xs", "iq2_xxs", "q8_0"))
def test_every_candidate_constructs_resource_valid_kernel(codec, monkeypatch):
    import torch
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel

    # This is a metadata/resource check, never a GPU qualification or timing.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    q = query(codec)
    keys = set()
    for _, c in TUNING.eligible_plan(q, DEVICE).candidates:
        if c.w4a16_tile_config is None:
            continue
        k1, n1, k2, n2 = c.w4a16_tile_config
        kernel = W4A16FusedMoeKernel(
            size_m=256, hidden_size=1024, intermediate_size=2048,
            num_experts=256, top_k=9, activation="relu2",
            apply_router_weight_on_input=False, zero_fc2_output=False,
            fc1_tile_n=n1, fc1_tile_k=k1, fc2_tile_n=n2, fc2_tile_k=k2,
            moe_block_size=c.w4a16_block_size_m, max_m_blocks=1024,
            pipeline_stages=c.w4a16_pipeline_stages, weight_layout=codec,
            scale_format=codec, w13_layout="packed",
        )
        assert kernel.shared_words * 4 <= 101376
        assert kernel.fc1.stages == kernel.fc2.stages == c.w4a16_pipeline_stages
        assert kernel.__cache_key__ not in keys
        keys.add(kernel.__cache_key__)


@pytest.mark.parametrize("change", (
    {"w4a16_pipeline_stages": 1}, {"w4a16_pipeline_stages": True},
    {"w4a16_block_size_m": 7}, {"w4a16_tile_config": (128, 64, 128, 128)},
    {"w4a16_tile_config": (256, 64, 256, 64)},
    {"w4a16_pipeline_stages": None},
))
def test_override_cannot_bypass_launch_validation(change):
    q = query()
    c = MoeDecodeConfig(
        backend="w4a16", route_planner="internal", max_active_clusters=None,
        w4a16_route_mode="packed", w4a16_tile_config=(128, 64, 128, 64),
        w4a16_block_size_m=8, w4a16_pipeline_stages=3,
    )
    with pytest.raises(ValueError):
        TUNING.configure(q, device=DEVICE, override=replace(c, **change))


@pytest.mark.parametrize("capacity", (1, 256))
def test_route_block_declaration_is_respected(capacity):
    q = replace(query(capacity=capacity), w4a16_block_size_m=32)
    configs = [c for _, c in TUNING.eligible_plan(q, DEVICE).candidates]
    assert {c.w4a16_block_size_m for c in configs} == {None, 32}
    assert TUNING.configure(q, device=DEVICE, search=False).default.w4a16_route_mode == "packed"


def test_q8_large_tiles_reject_excess_shared_memory_before_compilation():
    c = MoeDecodeConfig(
        backend="w4a16", route_planner="internal", max_active_clusters=None,
        w4a16_route_mode="packed", w4a16_tile_config=(128, 128, 128, 128),
        w4a16_block_size_m=64, w4a16_pipeline_stages=5,
    )
    with pytest.raises(ValueError, match="shared-memory"):
        TUNING.configure(query("q8_0"), device=DEVICE, override=c)
