"""CPU checks for EP-specific configuration and JIT cache isolation."""

import pytest

from flashinfer.jit.megamoe import get_megamoe_sm90_nvfp4_uri
from flashinfer.megamoe import select_fused_plan
from flashinfer.megamoe.heuristics import SMEM_CAPACITY, _fused_smem_bytes


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_nvfp4_jit_cache_distinguishes_ep_width():
    plan = select_fused_plan(132, 8, 256, 6, 4096, 2048, 384, 1)
    args = (plan, 132, 384, 256, 6, 4096, 2048, 10.0, True)
    ep4 = get_megamoe_sm90_nvfp4_uri(*args, num_ranks=4)
    ep8 = get_megamoe_sm90_nvfp4_uri(*args, num_ranks=8)
    assert ep4 != ep8
    assert "_r4_" in ep4 and "_r8_" in ep8


@pytest.mark.cpu_only
@pytest.mark.parametrize("ep", [1, 2, 3, 16])
def test_nvfp4_planner_rejects_unvalidated_ep_width(ep):
    with pytest.raises(ValueError, match="4 or 8 ranks"):
        select_fused_plan(132, ep, 256, 6, 4096, 2048, 384, 1)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "experts,hidden,intermediate,topk,push_max,rotated_from",
    [
        (256, 4096, 2048, 6, 32, None),
        (384, 6144, 2048, 8, 32, None),
        (384, 7168, 3072, 6, 64, 33),
    ],
)
def test_h200_ep8_counter_policy(
    experts, hidden, intermediate, topk, push_max, rotated_from
):
    for tokens in (1, 8, 24, 32, 33, 48, 64, 65, 128, 256):
        plan = select_fused_plan(
            132, 8, experts, topk, hidden, intermediate, 4096, tokens
        )
        assert plan.push_dispatch == (tokens <= push_max), tokens
        assert plan.no_clean_barrier == (
            rotated_from is not None and rotated_from <= tokens <= push_max
        ), tokens


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_sms", [78, 132])
def test_counter_policy_limits(num_sms):
    # Unmeasured geometry (H7168 / E256 / K8, DeepSeek-V3): push up to M=32, no rotated pool.
    for tokens in (1, 16, 32, 33, 64):
        plan = select_fused_plan(num_sms, 8, 256, 8, 7168, 2048, 4096, tokens)
        assert plan.push_dispatch == (tokens <= 32), tokens
        assert not plan.no_clean_barrier
    # Forced counter dispatch is ignored outside the validated small-M plans.
    for tokens in (65, 128, 256):
        plan = select_fused_plan(
            num_sms,
            8,
            256,
            6,
            4096,
            2048,
            4096,
            tokens,
            force_push_dispatch=True,
            force_no_clean_barrier=True,
        )
        assert not plan.push_dispatch and not plan.no_clean_barrier, tokens


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_sms", [78, 132])
@pytest.mark.parametrize(
    "hidden,intermediate,experts,topk",
    [
        (7168, 2048, 256, 8),
        (7168, 2048, 384, 8),
        (8192, 2048, 256, 8),
        (4096, 2048, 384, 8),
        (512, 256, 256, 6),
        (5120, 1536, 160, 8),  # GLM-4.6 / 4.7
        (4096, 1408, 128, 8),  # GLM-4.5-Air
        (2048, 768, 128, 8),  # Qwen3-30B-A3B
        (4096, 1024, 512, 10),  # Qwen3.5-397B-A17B
        (4096, 1024, 8192, 8),  # expert table alone overflows the BM64 pipeline
        (7168, 2048, 2048, 8),
    ],
)
def test_untuned_geometry_plans_are_legal(num_sms, hidden, intermediate, experts, topk):
    # Shapes outside the tuned set get the generic schedule (no H20 RS-only depth), a pipeline
    # depth that fits shared memory, and push only where the push row copy supports the hidden size.
    for tokens in range(1, 257):
        plan = select_fused_plan(
            num_sms, 8, experts, topk, hidden, intermediate, 4096, tokens
        )
        assert not plan.rs_swap_ab
        legal = {8: (3, 4), 16: (3,), 24: (3,), 64: (3,)}[plan.block_m]
        assert plan.num_stages in legal, (tokens, plan)
        assert _fused_smem_bytes(plan, num_sms, experts, hidden) <= SMEM_CAPACITY
        assert not plan.push_dispatch or (hidden % 1024 == 0 and tokens <= 32), tokens


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "hidden,intermediate", [(4096, 2560), (4352, 2048), (4096, 1000)]
)
def test_unsupported_fused_shapes_raise(hidden, intermediate):
    with pytest.raises(ValueError):
        select_fused_plan(132, 8, 256, 6, hidden, intermediate, 4096, 1)


@pytest.mark.cpu_only
def test_push_needs_one_publishing_sm_per_local_expert():
    # Local expert e is published by SM e, so push needs experts per rank <= SMs.
    plan = select_fused_plan(78, 8, 1024, 8, 4096, 1024, 4096, 1)
    assert not plan.push_dispatch
    plan = select_fused_plan(132, 8, 1024, 8, 4096, 1024, 4096, 1)
    assert plan.push_dispatch
