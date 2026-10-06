from __future__ import annotations

import torch

from b12x.gemm import block_fp8_linear as bfl


def test_block_fp8_preparation_contract_retains_real_compile_knobs() -> None:
    declaration = bfl.plan(bfl.Caps(device="cpu", max_tokens=8, in_features=128, out_features=256,
                                    output_dtype=torch.bfloat16))
    assert tuple(knob.name for knob in declaration.contract.knobs) == ("backend", "tile_m", "tile_n")
    candidates = tuple(declaration.contract.choices(declaration.query, None))
    assert {(config.tile_m, config.tile_n) for _, config in candidates} == {
        (tile_m, tile_n) for tile_m in (16, 32, 64, 128) for tile_n in (64, 128)
    }
