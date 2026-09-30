from __future__ import annotations

import pytest
import torch

from b12x.gemm import block_fp8_linear as bfl


@pytest.mark.parametrize("max_tokens,tile", [(1, (16, 64)), (8, (16, 128)), (9, (64, 64)), (129, (64, 128))])
def test_block_fp8_default_preparation_config_preserves_capacity_thresholds(max_tokens, tile) -> None:
    declaration = bfl.plan(bfl.Caps(
        device="cpu", max_tokens=max_tokens, in_features=128, out_features=256,
        output_dtype=torch.bfloat16,
    ))
    config = declaration.contract.default_config(declaration.query, None)
    assert (config.tile_m, config.tile_n) == tile


def test_block_fp8_declaration_keeps_exact_capacity_in_query() -> None:
    declaration = bfl.plan(bfl.Caps(device="cpu", max_tokens=129, in_features=128, out_features=256))
    assert declaration.query.max_tokens == 129
    assert declaration.query.output_dtype == "bfloat16"
