"""Host-only validation for mixed MXFP8-weight/BF16-activation MegaMoE."""

from __future__ import annotations

import pytest

from flashinfer.moe_ep import Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
from flashinfer.moe_ep.backends.mega.kernel.sm100.common.bf16_config import (
    Sm100_Bf16_Cutedsl_MegaMoeConfigBase,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
    MegaMoEBf16Mxfp8Config,
)
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.autotune import (
    bf16_mxfp8_candidates,
)
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.bf16_mxfp8 import (
    MegaMoEBf16Mxfp8Frontend,
)


def _config(**kwargs):
    return MegaMoEBf16Mxfp8Config(
        rank=0,
        world_size=4,
        num_tokens_per_rank=256,
        num_topk=4,
        num_total_experts=32,
        hidden=1024,
        intermediate=1024,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("mma_tiler_mnk", "transform_buffer", "accumulator_overlap", "transform_k_tile"),
    [
        ((256, 128, 128), "tmem", False, 128),
        ((256, 256, 128), "smem", False, 128),
        ((256, 256, 128), "tmem", True, 64),
    ],
)
def test_mixed_config_accepts_supported_implementation(
    mma_tiler_mnk, transform_buffer, accumulator_overlap, transform_k_tile
):
    config = _config(
        mma_tiler_mnk=mma_tiler_mnk,
        transform_buffer=transform_buffer,
        accumulator_overlap=accumulator_overlap,
        transform_k_tile=transform_k_tile,
    )
    assert config.num_experts_per_rank == 8


def test_mixed_config_rejects_unsupported_implementation_and_token_back():
    with pytest.raises(ValueError, match="implementation tuple"):
        _config(mma_tiler_mnk=(256, 256, 128))
    with pytest.raises(ValueError, match="standalone"):
        _config(token_back_mode="standalone_warps")  # type: ignore[arg-type]


def test_mixed_backend_is_registered():
    backend = create_mega_kernel(
        Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=128,
            top_k=2,
        )
    )
    assert backend.kernel_name() == "sm100_bf16_mxfp8_bf16_cutedsl"


def test_mixed_config_inherits_bf16_options():
    config = Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        gate_up_clamp=1.5,
        in_kernel_fc2_reduce=True,
    )
    assert isinstance(config, Sm100_Bf16_Cutedsl_MegaMoeConfigBase)
    assert config.gate_up_clamp == 1.5
    assert config.in_kernel_fc2_reduce


def test_mixed_autotune_candidate_matches_default_config():
    config = _config()
    frontend = MegaMoEBf16Mxfp8Frontend(config)
    assert bf16_mxfp8_candidates() == [
        {
            "mma_tiler_mnk": (256, 128, 128),
            "transform_buffer": "tmem",
            "accumulator_overlap": False,
            "transform_k_tile": 128,
            "cluster_shape_mnk": (2, 1, 1),
            "flag_batch": 1,
            "epi_flag_batch": (1, 1),
            "token_back_mode": "epi_warps",
            "load_balance_mode": "static",
        }
    ]
    frontend.apply_knobs(bf16_mxfp8_candidates()[0])
    assert frontend.config == config
