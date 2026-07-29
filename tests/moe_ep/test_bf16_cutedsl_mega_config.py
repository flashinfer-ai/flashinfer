"""Host-only validation for the BF16 CuTeDSL MegaMoE integration."""

from __future__ import annotations

import pytest

from flashinfer.moe_ep.backends.mega.kernel.bf16_cutedsl.config import (
    Bf16CutedslMegaMoeConfig,
)
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.bf16 import MegaMoEBf16Config
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.autotune import (
    bf16_candidates,
)
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.tuner import (
    default_knobs,
    is_valid_bf16,
)


def test_bf16_knobs_only_expose_valid_fixed_geometry():
    knobs = default_knobs(256, dtype="bf16")
    assert knobs["mma_tiler_mnk"] == (256, 256, 64)
    assert knobs["cluster_shape_mnk"] == (2, 1, 1)
    assert is_valid_bf16(knobs)
    assert not is_valid_bf16({**knobs, "mma_tiler_mnk": (256, 256, 256)})
    assert bf16_candidates() == [knobs]


@pytest.mark.parametrize(
    ("hidden", "intermediate", "top_k", "message"),
    ((33, 64, 1, "hidden"), (32, 65, 1, "intermediate"), (32, 64, 33, "topk")),
)
def test_bf16_frontend_rejects_unsupported_shapes(
    hidden: int, intermediate: int, top_k: int, message: str
):
    with pytest.raises(ValueError, match=message):
        MegaMoEBf16Config(
            rank=0,
            world_size=1,
            num_tokens_per_rank=64,
            num_topk=top_k,
            num_total_experts=1,
            hidden=hidden,
            intermediate=intermediate,
        )


def test_bf16_backend_defaults_to_scale_free_contract():
    config = Bf16CutedslMegaMoeConfig(intermediate_size=64, top_k=1)
    assert config.kernel_name == "bf16_cutedsl"
    assert config.knobs is None


def test_bf16_backend_accepts_collective_autotune():
    assert (
        Bf16CutedslMegaMoeConfig(intermediate_size=64, top_k=1, knobs="auto").knobs
        == "auto"
    )
