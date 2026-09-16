"""Distributed BF16 CuTeDSL MegaMoE configuration coverage.

The actual fused launch is exercised by the Blackwell mega job. These checks
keep scheduling and token-back modes visible to the regular multi-rank target
without importing the kernel drop directly.
"""

from __future__ import annotations

import pytest

from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.bf16 import MegaMoEBf16Config


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("load_balance_mode", ("static", "atomic_counter"))
@pytest.mark.parametrize(
    "token_back_mode",
    ("epi_warps", "standalone_warps", "reuse_dispatch_warps"),
)
def test_bf16_multirank_modes_are_constructible(
    load_balance_mode: str, token_back_mode: str
):
    config = MegaMoEBf16Config(
        rank=0,
        world_size=4,
        num_tokens_per_rank=256,
        num_topk=8,
        num_total_experts=256,
        hidden=7168,
        intermediate=2048,
        load_balance_mode=load_balance_mode,  # type: ignore[arg-type]
        token_back_mode=token_back_mode,  # type: ignore[arg-type]
    )
    assert config.num_experts_per_rank == 64
