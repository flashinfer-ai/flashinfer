from __future__ import annotations

import torch

from b12x.gemm import block_fp8_linear as bfl
from b12x.gemm._shared.block_fp8 import _scratch_plan


def test_block_fp8_declaration_has_no_scratch_or_runtime_owner() -> None:
    caps = bfl.Caps(device="cpu", max_tokens=4, in_features=128, out_features=256)
    declaration = bfl.plan(caps)
    assert declaration.query.max_tokens == 4
    assert declaration.component_id == "gemm.block_fp8_linear"


def test_block_fp8_private_scratch_layout_models_quantizer_views() -> None:
    caps = bfl.Caps(device="cpu", max_tokens=4, in_features=128, out_features=256)
    scratch = _scratch_plan(caps, (16, 64))
    spec = scratch.scratch_specs()[0]
    assert spec.name == "block_fp8_linear.scratch"
    assert spec.dtype == torch.uint8
    assert spec.nbytes == spec.shape[0]
