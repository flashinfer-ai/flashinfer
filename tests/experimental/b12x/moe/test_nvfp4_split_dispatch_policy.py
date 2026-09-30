"""NVFP4 materialization follows the captured declaration and explicit pins."""
from dataclasses import replace

import pytest

from b12x.moe.fused_moe._tuning import TUNING
from b12x.preparation import FrozenMapping
from tests.experimental.b12x.preparation.test_precision_choices import DEVICE, _nvfp4_query


@pytest.mark.parametrize("tile", (16, 64))
def test_split_rejects_other_source_tiles(tile):
    query = _nvfp4_query()
    split = next(config for _, config in TUNING.eligible_plan(query, DEVICE).candidates
                 if config.nvfp4_materialize_intermediate)
    with pytest.raises(ValueError, match="M128 contract"):
        TUNING.configure(query, device=DEVICE, override=replace(split, dynamic_tile_m=tile))


@pytest.mark.parametrize("enabled", (False, True))
def test_captured_materialization_default_and_explicit_pin(enabled):
    query = replace(_nvfp4_query(), controls=FrozenMapping({"dynamic_nvfp4_materialized": enabled}))
    default = TUNING.configure(query, device=DEVICE).default
    assert default.nvfp4_materialize_intermediate == enabled
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    assert {config.nvfp4_materialize_intermediate for config in configs} == {False, True}
    opposite = replace(default, nvfp4_materialize_intermediate=not enabled)
    assert TUNING.configure(query, device=DEVICE, override=opposite).default == opposite
