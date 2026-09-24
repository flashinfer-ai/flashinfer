"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

from flashinfer.experimental.kimi_k3_fused_router import cake_backend
from flashinfer.experimental.kimi_k3_fused_router.cake_backend import (
    ARM_Q_CLUSTER,
    BLOCK_M_VALUES,
    NUM_EXPERTS,
    OWNER_CTAS,
    SHAPE_ROUTES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    SUPPORTED_NUM_TOKENS,
    TOP_K,
    KimiK3RoutePlan,
    allocate_kimi_k3_route_plan,
    launch_grid,
    max_route_blocks,
    persistent_grid_cap,
    route_arm,
    validate_kimi_k3_fused_router_inputs,
)
from flashinfer.fused_moe import (
    kimi_k3_fused_router,
    prepare_kimi_k3_fused_router,
)

WEIGHT_ATOL = 1e-6
WEIGHT_RTOL = 1e-5
SORTED_POISON = -1_234_567
EXPERT_POISON = -7_654_321
WORKSPACE_POISON = -91

# The 28 routed shapes: num_tokens in {1, 2, ..., 8192} x block_m in {8, 16}.
ROUTED_SHAPES = [(rows, bm) for bm in BLOCK_M_VALUES for rows in SUPPORTED_NUM_TOKENS]


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------


def make_inputs(num_tokens, device, seed=0):
    """Random gate logits with tiny expert/token tie-breakers and a small bias."""
    gen = torch.Generator(device=device).manual_seed(seed)
    expert = torch.arange(NUM_EXPERTS, dtype=torch.float32, device=device)
    token = torch.arange(num_tokens, dtype=torch.float32, device=device).reshape(-1, 1)
    logits = torch.randn((num_tokens, NUM_EXPERTS), generator=gen, device=device)
    logits = logits + expert.reshape(1, -1) * 1.0e-5 + token * 1.0e-7
    bias = torch.randn((NUM_EXPERTS,), generator=gen, device=device) * 0.05
    return logits.contiguous(), bias.contiguous()


def reference_route(logits, bias, block_m):
    """FP32 torch reference of the fused router and its aligned route plan."""
    logits = logits.float()
    scores = torch.sigmoid(logits)
    ranking = scores + bias.float().reshape(1, NUM_EXPERTS)
    ranked = torch.argsort(ranking, dim=-1, descending=True, stable=True)[:, :TOP_K]
    topk_ids = torch.sort(ranked, dim=-1).values.to(torch.int32)
    selected = torch.gather(scores, 1, topk_ids.to(torch.int64))
    total = torch.zeros(selected.shape[0], dtype=torch.float32, device=logits.device)
    for route in range(TOP_K):  # sequential FP32 sum in ascending-id order
        total = total + selected[:, route]
    norm = torch.where(total > 0, total, torch.ones_like(total))
    topk_weights = (selected / norm.reshape(-1, 1)).to(torch.float32)

    num_tokens = int(logits.shape[0])
    pair_count = num_tokens * TOP_K
    flat = topk_ids.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=NUM_EXPERTS).to(torch.int32)
    padded = ((counts + block_m - 1) // block_m) * block_m
    offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=logits.device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(padded, dim=0)
    extent = int(offsets[-1].item())
    sorted_token_ids = torch.full(
        (extent,), pair_count, dtype=torch.int32, device=logits.device
    )
    pair_ids = torch.arange(pair_count, dtype=torch.int32, device=logits.device)
    order = torch.argsort(flat, stable=True)
    ordered_experts = flat[order]
    unpadded = torch.empty_like(offsets)
    unpadded[0] = 0
    unpadded[1:] = torch.cumsum(counts, dim=0)
    rank_in_expert = torch.arange(
        pair_count, dtype=torch.int64, device=logits.device
    ) - torch.repeat_interleave(unpadded[:-1].to(torch.int64), counts.to(torch.int64))
    destinations = offsets[:-1].to(torch.int64)[ordered_experts] + rank_in_expert
    sorted_token_ids[destinations] = pair_ids[order]
    expert_ids = torch.repeat_interleave(
        torch.arange(NUM_EXPERTS, dtype=torch.int32, device=logits.device),
        (padded // block_m).to(torch.int64),
    )
    return KimiK3RoutePlan(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=torch.tensor(
            [extent], dtype=torch.int32, device=logits.device
        ),
        expert_counts=counts,
        expert_offsets=offsets,
        expert_scatter_offsets=counts.clone(),
    )


def poison_plan(plan):
    """Fill every output with a value the kernel must overwrite (or leave)."""
    plan.topk_weights.fill_(float("nan"))
    plan.topk_ids.fill_(-1)
    plan.sorted_token_ids.fill_(SORTED_POISON)
    plan.expert_ids.fill_(EXPERT_POISON)
    plan.num_tokens_post_padded.fill_(WORKSPACE_POISON)
    plan.expert_counts.fill_(WORKSPACE_POISON)
    plan.expert_offsets.fill_(WORKSPACE_POISON)
    plan.expert_scatter_offsets.fill_(WORKSPACE_POISON)


def check_route_plan(actual, expected, *, num_tokens, block_m):
    """Exact selection / plan comparison plus the grouped-pair invariants."""
    pair_count = num_tokens * TOP_K
    assert torch.equal(actual.topk_ids, expected.topk_ids)
    torch.testing.assert_close(
        actual.topk_weights, expected.topk_weights, atol=WEIGHT_ATOL, rtol=WEIGHT_RTOL
    )
    assert torch.equal(actual.num_tokens_post_padded, expected.num_tokens_post_padded)
    assert torch.equal(actual.expert_counts, expected.expert_counts)
    assert torch.equal(actual.expert_offsets, expected.expert_offsets)
    assert torch.equal(actual.expert_scatter_offsets, expected.expert_scatter_offsets)
    extent = int(expected.num_tokens_post_padded[0].item())
    assert extent % block_m == 0
    assert extent <= actual.sorted_token_ids.numel()
    blocks = extent // block_m
    assert torch.equal(actual.expert_ids[:blocks], expected.expert_ids)
    # Grouped-pair invariants (independent of the emission order within an expert).
    ids = actual.topk_ids.to(torch.int64)
    assert bool(((ids >= 0) & (ids < NUM_EXPERTS)).all())
    assert bool((ids[:, 1:] > ids[:, :-1]).all())
    counts = expected.expert_counts.to(torch.int64)
    offsets = expected.expert_offsets.to(torch.int64)
    padded = offsets[1:] - offsets[:-1]
    active = actual.sorted_token_ids[:extent].to(torch.int64)
    position_experts = torch.repeat_interleave(
        torch.arange(NUM_EXPERTS, dtype=torch.int64, device=ids.device), padded
    )
    starts = torch.repeat_interleave(offsets[:-1], padded)
    within = torch.arange(extent, dtype=torch.int64, device=ids.device) - starts
    real = within < torch.repeat_interleave(counts, padded)
    assert bool((active[~real] == pair_count).all()), "padding sentinel mismatch"
    real_pairs = active[real]
    assert real_pairs.numel() == pair_count
    assert torch.equal(
        torch.sort(real_pairs).values,
        torch.arange(pair_count, dtype=torch.int64, device=ids.device),
    ), "real pairs are not an exact permutation"
    flat_ids = expected.topk_ids.reshape(-1).to(torch.int64)
    assert torch.equal(flat_ids[real_pairs], position_experts[real]), (
        "pair grouped under the wrong expert"
    )
    # The generated program emits every expert segment in ascending pair order.
    assert torch.equal(actual.sorted_token_ids[:extent], expected.sorted_token_ids)
    # Capacity beyond the valid extent is left untouched.
    assert bool((actual.sorted_token_ids[extent:] == SORTED_POISON).all())
    assert bool((actual.expert_ids[blocks:] == EXPERT_POISON).all())


# ---------------------------------------------------------------------------
# Host-side contract (CPU)
# ---------------------------------------------------------------------------


def test_route_tables_cover_the_routed_shapes():
    for arch, table in SHAPE_ROUTES.items():
        assert sorted(table) == sorted(ROUTED_SHAPES), arch
        assert len(table) == 28
    assert {arm for arm in SHAPE_ROUTES["sm_100a"].values()} == {
        "A",
        "L",
        "M",
        "M4S",
        "Q",
        "G",
    }
    assert {arm for arm in SHAPE_ROUTES["sm_103a"].values()} == {
        "A",
        "L",
        "M",
        "Q",
        "G",
    }
    assert route_arm("sm_100a", 512, 8) == "M4S"
    assert route_arm("sm_103a", 512, 8) == "Q"
    for arch in SHAPE_ROUTES:
        assert route_arm(arch, 1, 16) == "A"
        assert route_arm(arch, 128, 8) == "L"
        assert route_arm(arch, 256, 16) == "M"
        assert route_arm(arch, 2048, 8) == "Q"
        assert route_arm(arch, 8192, 16) == "G"
    with pytest.raises(NotImplementedError, match="exactly num_tokens"):
        route_arm("sm_100a", 3, 8)
    with pytest.raises(NotImplementedError):
        route_arm("sm_103a", 16384, 16)


def test_persistent_grid_cap():
    assert persistent_grid_cap((10, 0), 148) == 444
    assert persistent_grid_cap((10, 3), 148) == 592
    assert persistent_grid_cap((10, 3), 160) == 640


@pytest.mark.parametrize(
    "compute_capability,sm_count", [((10, 0), 148), ((10, 3), 148)]
)
def test_launch_grid_rules(compute_capability, sm_count):
    cap = persistent_grid_cap(compute_capability, sm_count)
    kw = dict(compute_capability=compute_capability, sm_count=sm_count)
    assert launch_grid("A", 1, **kw) == 1
    # Two and four tokens launch eight CTAs; otherwise one CTA per token up to the cap.
    assert launch_grid("L", 2, **kw) == 8
    assert launch_grid("L", 4, **kw) == 8
    assert launch_grid("L", 8, **kw) == 8
    assert launch_grid("L", 128, **kw) == 128
    assert launch_grid("L", 512, **kw) == min(512, cap)
    assert launch_grid("M", 256, **kw) == 256
    assert launch_grid("M", 2048, **kw) == cap
    assert launch_grid("G", 4096, **kw) == cap
    assert launch_grid("G", 8192, **kw) == cap
    # Arm M4S: four CTAs per SM regardless of the architecture cap.
    assert launch_grid("M4S", 512, **kw) == 512
    assert launch_grid("M4S", 2048, **kw) == 4 * sm_count
    # Arm Q: bounded by whole co-resident clusters, then rounded down to clusters.
    assert launch_grid("Q", 1024, max_active_clusters=1000, **kw) == (cap // 4) * 4
    assert launch_grid("Q", 1024, max_active_clusters=100, **kw) == 400
    assert (
        launch_grid("Q", 512, max_active_clusters=200, **kw) == (min(512, cap) // 4) * 4
    )
    assert launch_grid("Q", 1024, max_active_clusters=37, **kw) == 37 * ARM_Q_CLUSTER
    with pytest.raises(RuntimeError, match="co-resident owner CTAs"):
        launch_grid(
            "Q", 1024, max_active_clusters=OWNER_CTAS // ARM_Q_CLUSTER - 1, **kw
        )
    with pytest.raises(RuntimeError, match="cluster capacity"):
        launch_grid("Q", 1024, **kw)
    with pytest.raises(RuntimeError):
        launch_grid("L", 1024, **kw)
    with pytest.raises(RuntimeError):
        launch_grid("M", 64, **kw)
    with pytest.raises(RuntimeError):
        launch_grid("M4S", 256, **kw)
    with pytest.raises(RuntimeError):
        launch_grid("A", 2, **kw)
    with pytest.raises(ValueError):
        launch_grid("Z", 8, **kw)


def test_max_route_blocks_and_plan_capacity():
    assert max_route_blocks(1, 8) == 16
    assert max_route_blocks(1, 16) == 16
    assert max_route_blocks(64, 8) == 896 + (1024 - 896) // 8
    assert max_route_blocks(8192, 16) == 896 + (131072 - 896) // 16
    plan = allocate_kimi_k3_route_plan(64, 8, torch.device("cpu"))
    assert (
        plan.topk_weights.shape == (64, TOP_K)
        and plan.topk_weights.dtype == torch.float32
    )
    assert plan.topk_ids.shape == (64, TOP_K) and plan.topk_ids.dtype == torch.int32
    assert plan.sorted_token_ids.numel() == max_route_blocks(64, 8) * 8
    assert plan.expert_ids.numel() == max_route_blocks(64, 8)
    assert plan.num_tokens_post_padded.shape == (1,)
    assert plan.expert_counts.shape == (NUM_EXPERTS,)
    assert plan.expert_offsets.shape == (NUM_EXPERTS + 1,)
    assert plan.expert_scatter_offsets.shape == (NUM_EXPERTS,)
    with pytest.raises(ValueError):
        allocate_kimi_k3_route_plan(64, 4, torch.device("cpu"))
    with pytest.raises(ValueError):
        allocate_kimi_k3_route_plan(0, 8, torch.device("cpu"))


def test_validate_inputs():
    logits = torch.zeros(16, NUM_EXPERTS)
    bias = torch.zeros(NUM_EXPERTS)
    assert validate_kimi_k3_fused_router_inputs(logits, bias, 16) == (16, 16)
    with pytest.raises(TypeError):
        validate_kimi_k3_fused_router_inputs(logits.half(), bias, 8)
    with pytest.raises(ValueError, match="896"):
        validate_kimi_k3_fused_router_inputs(torch.zeros(16, 512), bias, 8)
    with pytest.raises(ValueError, match="bias"):
        validate_kimi_k3_fused_router_inputs(logits, torch.zeros(NUM_EXPERTS + 1), 8)
    with pytest.raises(ValueError, match="contiguous"):
        validate_kimi_k3_fused_router_inputs(logits.t().contiguous().t(), bias, 8)
    with pytest.raises(ValueError, match="block_m"):
        validate_kimi_k3_fused_router_inputs(logits, bias, 32)
    with pytest.raises(TypeError):
        validate_kimi_k3_fused_router_inputs(logits, bias, 8.0)


def test_reference_plan_is_self_consistent():
    logits, bias = make_inputs(37, torch.device("cpu"), seed=3)
    plan = reference_route(logits, bias, 8)
    poisoned = KimiK3RoutePlan(
        topk_weights=plan.topk_weights.clone(),
        topk_ids=plan.topk_ids.clone(),
        sorted_token_ids=torch.cat(
            [plan.sorted_token_ids, torch.full((24,), SORTED_POISON, dtype=torch.int32)]
        ),
        expert_ids=torch.cat(
            [plan.expert_ids, torch.full((3,), EXPERT_POISON, dtype=torch.int32)]
        ),
        num_tokens_post_padded=plan.num_tokens_post_padded.clone(),
        expert_counts=plan.expert_counts.clone(),
        expert_offsets=plan.expert_offsets.clone(),
        expert_scatter_offsets=plan.expert_scatter_offsets.clone(),
    )
    check_route_plan(poisoned, plan, num_tokens=37, block_m=8)


# ---------------------------------------------------------------------------
# GPU correctness
# ---------------------------------------------------------------------------


def _gpu_arch():
    if not torch.cuda.is_available():
        return None
    return SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(0))


def _require_program(num_tokens, block_m):
    arch = _gpu_arch()
    if arch is None:
        pytest.skip("the Kimi-K3 fused router requires an SM100/SM103 GPU")
    device = torch.device("cuda", 0)
    if not cake_backend.generated_program_available(device, num_tokens, block_m):
        pytest.skip(
            f"no generated Kimi-K3 fused router program (arm "
            f"{route_arm(arch, num_tokens, block_m)}, block_m {block_m}) registered for {arch}"
        )
    return device


def _run_and_check(num_tokens, block_m, seed):
    device = _require_program(num_tokens, block_m)
    logits, bias = make_inputs(num_tokens, device, seed=seed)
    plan = allocate_kimi_k3_route_plan(num_tokens, block_m, device)
    poison_plan(plan)
    runner = prepare_kimi_k3_fused_router(logits, bias, block_m=block_m, plan=plan)
    assert runner.arm == route_arm(_gpu_arch(), num_tokens, block_m)
    out = runner()
    torch.cuda.synchronize()
    assert out is plan
    expected = reference_route(logits, bias, block_m)
    check_route_plan(plan, expected, num_tokens=num_tokens, block_m=block_m)
    return runner, logits, bias, plan


@pytest.mark.parametrize("num_tokens,block_m", ROUTED_SHAPES)
def test_fused_router_matches_reference(num_tokens, block_m):
    _run_and_check(num_tokens, block_m, seed=4568000 + num_tokens + block_m)


def test_allocating_api_matches_reference():
    device = _require_program(64, 8)
    logits, bias = make_inputs(64, device, seed=11)
    plan = kimi_k3_fused_router(logits, bias, block_m=8)
    torch.cuda.synchronize()
    expected = reference_route(logits, bias, 8)
    assert torch.equal(plan.topk_ids, expected.topk_ids)
    torch.testing.assert_close(
        plan.topk_weights, expected.topk_weights, atol=WEIGHT_ATOL, rtol=WEIGHT_RTOL
    )
    extent = int(expected.num_tokens_post_padded[0].item())
    assert torch.equal(plan.sorted_token_ids[:extent], expected.sorted_token_ids)
    assert torch.equal(plan.expert_ids[: extent // 8], expected.expert_ids)
    assert torch.equal(plan.expert_counts, expected.expert_counts)
    assert torch.equal(plan.expert_offsets, expected.expert_offsets)


@pytest.mark.parametrize("num_tokens,block_m", [(8, 16), (256, 8), (2048, 16)])
def test_graph_replay_follows_device_inputs(num_tokens, block_m):
    """Capture once, replay with new logits / bias written into the same buffers."""
    runner, logits, bias, plan = _run_and_check(num_tokens, block_m, seed=21)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner()
    torch.cuda.synchronize()
    for round_index in range(3):
        new_logits, new_bias = make_inputs(
            num_tokens, logits.device, seed=1000 + round_index
        )
        logits.copy_(new_logits)
        bias.copy_(new_bias)
        poison_plan(plan)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        expected = reference_route(logits, bias, block_m)
        check_route_plan(plan, expected, num_tokens=num_tokens, block_m=block_m)


def test_launch_makes_no_allocation():
    runner, _, _, _ = _run_and_check(1024, 8, seed=5)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] - before["allocation.all.allocated"] == 0


def test_prepare_rejects_bad_plan_and_shapes():
    device = _require_program(16, 8)
    logits, bias = make_inputs(16, device, seed=7)
    small = allocate_kimi_k3_route_plan(8, 8, device)
    with pytest.raises(ValueError, match="shape"):
        prepare_kimi_k3_fused_router(logits, bias, block_m=8, plan=small)
    plan = allocate_kimi_k3_route_plan(16, 8, device)
    aliased = plan._replace(expert_scatter_offsets=plan.expert_counts)
    with pytest.raises(ValueError, match="overlap"):
        prepare_kimi_k3_fused_router(logits, bias, block_m=8, plan=aliased)
    with pytest.raises(NotImplementedError, match="exactly num_tokens"):
        prepare_kimi_k3_fused_router(logits[:3].contiguous(), bias, block_m=8)
    with pytest.raises(ValueError, match="one CUDA device"):
        prepare_kimi_k3_fused_router(logits, bias.cpu(), block_m=8)
    with pytest.raises(ValueError, match="backend"):
        prepare_kimi_k3_fused_router(logits, bias, block_m=8, backend="triton")
