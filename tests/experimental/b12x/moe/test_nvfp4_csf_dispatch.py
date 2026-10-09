"""Scale reconstruction is a prepared choice without changing expert precision."""

from dataclasses import replace

import pytest

from b12x.moe.fused_moe._tuning import TUNING, MoeDecodeConfig
from b12x.preparation import FrozenMapping
from tests.experimental.b12x.preparation.test_precision_choices import (
    DEVICE,
    _nvfp4_query,
)


@pytest.mark.parametrize("capacity", [4, 32, 128])
def test_prepared_compressed_planes_race_both_scale_decoders(capacity):
    native = replace(_nvfp4_query(), num_tokens=capacity, routed_rows=capacity * 4)
    compressed = replace(native, nvfp4_inline_scales=True)
    native_configs = [c for _, c in TUNING.eligible_plan(native, DEVICE).candidates]
    csf_configs = [c for _, c in TUNING.eligible_plan(compressed, DEVICE).candidates]
    assert all(not c.nvfp4_inline_scales for c in native_configs)
    assert {c.nvfp4_inline_scales for c in csf_configs} == {False, True}
    assert set(native_configs) <= set(csf_configs)
    assert TUNING.encode_query(native) != TUNING.encode_query(compressed)
    for config in csf_configs:
        assert MoeDecodeConfig.from_config(FrozenMapping(config.to_dict())) == config
        if config.nvfp4_inline_scales:
            assert not config.nvfp4_materialize_intermediate
            assert config.backend == "dynamic"


@pytest.mark.parametrize(
    "changes",
    [
        {"nvfp4_inline_scales": False},
        {"hidden_size": 192},
        {"intermediate_size": 96},
        {"activation": "relu2"},
        {"controls": FrozenMapping({"dynamic_tile_mn": (16, 64)})},
    ],
)
def test_inline_scale_override_rejects_incompatible_storage_or_geometry(changes):
    query = replace(_nvfp4_query(), nvfp4_inline_scales=True)
    config = next(
        c
        for _, c in TUNING.eligible_plan(query, DEVICE).candidates
        if c.nvfp4_inline_scales
    )
    with pytest.raises(ValueError, match="prepared compressed planes"):
        TUNING.configure(replace(query, **changes), device=DEVICE, override=config)
