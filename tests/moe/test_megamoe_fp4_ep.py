"""CPU checks for EP-specific configuration and JIT cache isolation."""

import pytest

from flashinfer.jit.megamoe import get_megamoe_sm90_nvfp4_uri
from flashinfer.megamoe import select_fused_plan


@pytest.mark.parametrize("num_sms", [78, 132])
@pytest.mark.parametrize(
    "experts,hidden,intermediate,topk",
    [(256, 4096, 2048, 6), (384, 6144, 2048, 8), (384, 7168, 3072, 6)],
)
@pytest.mark.parametrize("tokens", [1, 24, 64, 256])
def test_ep4_fused_plan_uses_legal_generic_pipeline(
    num_sms, experts, hidden, intermediate, topk, tokens
):
    plan = select_fused_plan(
        num_sms, 4, experts, topk, hidden, intermediate, 384, tokens
    )
    assert not plan.rs_swap_ab
    assert not plan.push_dispatch
    assert not plan.no_clean_barrier
    assert (experts // 4) % plan.num_experts_per_wave == 0
    if plan.block_m == 8:
        assert plan.num_stages in (3, 4)
    else:
        assert plan.num_stages == 3


def test_nvfp4_jit_cache_distinguishes_ep_width():
    plan = select_fused_plan(132, 8, 256, 6, 4096, 2048, 384, 1)
    args = (plan, 132, 384, 256, 6, 4096, 2048, 10.0, True)
    ep4 = get_megamoe_sm90_nvfp4_uri(*args, num_ranks=4)
    ep8 = get_megamoe_sm90_nvfp4_uri(*args, num_ranks=8)
    assert ep4 != ep8
    assert "_r4_" in ep4 and "_r8_" in ep8


@pytest.mark.parametrize("ep", [1, 2, 3, 16])
def test_nvfp4_planner_rejects_unvalidated_ep_width(ep):
    with pytest.raises(ValueError, match="4 or 8 ranks"):
        select_fused_plan(132, ep, 256, 6, 4096, 2048, 384, 1)
