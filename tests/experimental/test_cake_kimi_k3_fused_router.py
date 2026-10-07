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

from flashinfer.experimental.kimi_k3_fused_router import cake_backend, cake_jit
from flashinfer.experimental.kimi_k3_fused_router.cake_backend import (
    ARM_GW_CTAS_PER_SM,
    ARM_L_FAMILY,
    ARM_L_MAX_TOKENS,
    ARM_L_MIN_GRID,
    ARM_LC_TOKENS,
    ARM_M_MAX_TOKENS,
    ARM_Q4S_CLUSTER,
    ARM_Q4S_CTAS_PER_SM,
    ARM_Q4S_FAMILY,
    BLOCK_M_VALUES,
    MEASURED_NUM_TOKENS,
    NUM_EXPERTS,
    OWNER_CTAS,
    SHAPE_ROUTES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    TOP_K,
    KimiK3RoutePlan,
    allocate_kimi_k3_route_plan,
    launch_grid,
    max_route_blocks,
    persistent_grid_cap,
    route_arm,
    route_cell,
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

# The 28 measured cells: num_tokens in {1, 2, ..., 8192} x block_m in {8, 16}.
MEASURED_SHAPES = [(rows, bm) for bm in BLOCK_M_VALUES for rows in MEASURED_NUM_TOKENS]
# Token counts between the measured cells, one or two per arm range (including
# the first count above every cell boundary up to the 8192-token limit).
ARBITRARY_NUM_TOKENS = (
    3,
    5,
    17,
    31,
    33,
    100,
    129,
    200,
    257,
    300,
    1000,
    1025,
    2047,
    2049,
    3000,
    4097,
    8000,
)
ARBITRARY_SHAPES = [
    (rows, bm) for bm in BLOCK_M_VALUES for rows in ARBITRARY_NUM_TOKENS
]


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


# Two ulps of a unit-scale FP32 score.  The kernel reproduces the SGLang
# router's ``__expf`` / ``__fdividef`` sigmoid, whose last bit can differ from
# ``torch.sigmoid`` (and between GPU generations).  Two candidates whose
# biased scores agree to within this tolerance may be ordered either way at
# the top-16 boundary; everything else must match exactly.
TIE_TOLERANCE = 2.0**-22


def reference_route(logits, bias, block_m, topk_ids=None):
    """FP32 torch reference of the fused router and its aligned route plan.

    ``topk_ids`` (ascending, validated by :func:`assert_selection_matches`)
    replaces the reference selection so that weights and the plan are derived
    from a tie-resolved selection.
    """
    logits = logits.float()
    scores = torch.sigmoid(logits)
    ranking = scores + bias.float().reshape(1, NUM_EXPERTS)
    if topk_ids is None:
        ranked = torch.argsort(ranking, dim=-1, descending=True, stable=True)[:, :TOP_K]
        topk_ids = torch.sort(ranked, dim=-1).values.to(torch.int32)
    else:
        topk_ids = topk_ids.to(torch.int32).clone()
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


def assert_selection_matches(actual_ids, logits, bias):
    """Every row selects the 16 largest biased scores up to ``TIE_TOLERANCE``.

    Rows with a clear gap between the 16th and 17th candidate must equal the
    torch reference exactly (the tolerance cannot admit a lower-ranked expert);
    only candidates tied to within two ulps may swap.  Returns the number of
    rows resolved through the tie tolerance.
    """
    ids = actual_ids.to(torch.int64)
    assert ids.shape == (logits.shape[0], TOP_K)
    assert bool(((ids >= 0) & (ids < NUM_EXPERTS)).all())
    assert bool((ids[:, 1:] > ids[:, :-1]).all()), "ids must be ascending and distinct"
    ranking = torch.sigmoid(logits.float()) + bias.float().reshape(1, NUM_EXPERTS)
    selected = torch.zeros_like(ranking, dtype=torch.bool).scatter_(1, ids, True)
    selected_min = torch.where(
        selected, ranking, torch.full_like(ranking, float("inf"))
    )
    unselected_max = torch.where(
        selected, torch.full_like(ranking, float("-inf")), ranking
    )
    slack = unselected_max.amax(dim=1) - selected_min.amin(dim=1)
    assert bool((slack <= TIE_TOLERANCE).all()), (
        f"rows {torch.nonzero(slack > TIE_TOLERANCE).flatten().tolist()[:8]} select an "
        f"expert ranked below the top-16 by more than {TIE_TOLERANCE:.3g}"
    )
    reference_ids = reference_route(logits, bias, 8).topk_ids.to(torch.int64)
    return int((ids != reference_ids).any(dim=1).sum().item())


def expected_plan(actual, logits, bias, block_m):
    """Reference plan for the kernel's (tie-validated) selection."""
    assert_selection_matches(actual.topk_ids, logits, bias)
    return reference_route(logits, bias, block_m, topk_ids=actual.topk_ids)


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


def test_route_tables_cover_the_measured_cells():
    for arch, table in SHAPE_ROUTES.items():
        assert sorted(table) == sorted(MEASURED_SHAPES), arch
        assert len(table) == 28
        assert set(table.values()) == {"L", "LC", "LP", "M", "Q4S", "Q4SP", "GW"}
    # The two tables differ in exactly two cells of the L / LP family.
    differing = {
        key
        for key in SHAPE_ROUTES["sm_100a"]
        if SHAPE_ROUTES["sm_100a"][key] != SHAPE_ROUTES["sm_103a"][key]
    }
    assert differing == {(64, 16), (128, 8)}
    assert route_arm("sm_100a", 64, 16) == "LP"
    assert route_arm("sm_103a", 64, 16) == "L"
    assert route_arm("sm_100a", 128, 8) == "L"
    assert route_arm("sm_103a", 128, 8) == "LP"
    for arch in SHAPE_ROUTES:
        # Arm LC serves exactly the LC row counts (one kernel per token count).
        assert set(ARM_LC_TOKENS) == {
            rows for (rows, _), arm in SHAPE_ROUTES[arch].items() if arm == "LC"
        }
        for rows in ARM_LC_TOKENS:
            for bm in BLOCK_M_VALUES:
                assert route_arm(arch, rows, bm) == "LC"
        assert {
            rows for (rows, _), arm in SHAPE_ROUTES[arch].items() if arm in ARM_L_FAMILY
        } == {32, 64, 128}
        assert route_arm(arch, 32, 8) == "LP"
        assert route_arm(arch, 32, 16) == "LP"
        assert route_arm(arch, 64, 8) == "LP"
        assert route_arm(arch, 128, 16) == "L"
        assert route_arm(arch, 256, 16) == "M"
        assert route_arm(arch, 512, 16) == "Q4S"
        assert route_arm(arch, 1024, 16) == "Q4S"
        assert route_arm(arch, 2048, 8) == "Q4SP"
        assert route_arm(arch, 2048, 16) == "Q4SP"
        assert route_arm(arch, 8192, 16) == "GW"
        # Every measured cell's arm admits its token count on that architecture.
        for (rows, _bm), arm in SHAPE_ROUTES[arch].items():
            cc = next(c for c, a in SUPPORTED_COMPUTE_CAPABILITIES.items() if a == arch)
            launch_grid(
                arm, rows, compute_capability=cc, sm_count=148, max_active_clusters=1000
            )
    with pytest.raises(NotImplementedError, match="registered"):
        route_arm("sm_90a", 64, 8)
    with pytest.raises(ValueError, match="block_m"):
        route_arm("sm_100a", 64, 4)


def test_route_cell_and_arbitrary_token_counts():
    # Measured counts are their own cell; LC counts stay exact.
    for rows in MEASURED_NUM_TOKENS:
        assert route_cell(rows) == rows
    # Counts below 32 that are not an LC row take the 32-token cell (LP).
    for rows in (3, 5, 6, 7, 9, 15, 17, 31):
        assert route_cell(rows) == 32
        for arch in SHAPE_ROUTES:
            assert route_arm(arch, rows, 8) == "LP"
    # Every other count takes the next measured cell up.
    assert route_cell(33) == 64
    assert route_cell(100) == 128
    assert route_cell(129) == 256
    assert route_cell(257) == 512
    assert route_cell(1025) == 2048
    assert route_cell(2049) == 4096
    assert route_cell(4097) == 8192
    assert route_cell(8000) == 8192
    # Above the largest measured cell no kernel serves; the route rejects.
    assert route_cell(8193) is None
    for arch in SHAPE_ROUTES:
        assert route_arm(arch, 4097, 8) == "GW"
        assert route_arm(arch, 8000, 16) == "GW"
        for rows in (8193, 10000, 1 << 20):
            with pytest.raises(ValueError, match="8192"):
                route_arm(arch, rows, 8)
        assert route_arm(arch, 100, 16) == "L"
        assert route_arm(arch, 200, 8) == "M"
        assert route_arm(arch, 300, 8) == "Q4S"
        assert route_arm(arch, 1000, 16) == "Q4S"
        assert route_arm(arch, 2047, 8) == "Q4SP"
        assert route_arm(arch, 3000, 16) == "GW"
    assert route_arm("sm_100a", 33, 16) == "LP"
    assert route_arm("sm_103a", 33, 16) == "L"
    assert route_arm("sm_100a", 100, 8) == "L"
    assert route_arm("sm_103a", 100, 8) == "LP"
    # The chosen arm admits every count of its range on its architecture.
    for arch, cc in (("sm_100a", (10, 0)), ("sm_103a", (10, 3))):
        for rows in (*ARBITRARY_NUM_TOKENS, *MEASURED_NUM_TOKENS):
            for bm in BLOCK_M_VALUES:
                arm = route_arm(arch, rows, bm)
                grid = launch_grid(
                    arm,
                    rows,
                    compute_capability=cc,
                    sm_count=148,
                    max_active_clusters=1000,
                )
                assert grid >= 1
    with pytest.raises(ValueError, match="positive"):
        route_arm("sm_100a", 0, 8)
    with pytest.raises(ValueError, match="8192"):
        route_arm("sm_100a", 8193, 8)


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
    # Arm LC: one cluster of num_tokens CTAs (a single CTA for one token, a
    # 16-CTA cluster for sixteen).
    for rows in ARM_LC_TOKENS:
        assert launch_grid("LC", rows, **kw) == rows
    # Arm L: at least the 128 plan owners, otherwise one CTA per token up to
    # the cap; admits at most 128 tokens.
    assert ARM_L_MAX_TOKENS == 128
    assert launch_grid("L", 1, **kw) == ARM_L_MIN_GRID
    assert launch_grid("L", 3, **kw) == ARM_L_MIN_GRID
    assert launch_grid("L", 16, **kw) == ARM_L_MIN_GRID
    assert launch_grid("L", 100, **kw) == ARM_L_MIN_GRID
    assert launch_grid("L", 128, **kw) == 128
    with pytest.raises(RuntimeError, match="at most"):
        launch_grid("L", 129, **kw)
    # Arm LP: the L rule (same kernel family, same admission guard).
    assert ARM_L_FAMILY == ("L", "LP")
    for rows in (3, 32, 64, 100, 128):
        assert launch_grid("LP", rows, **kw) == launch_grid("L", rows, **kw)
    with pytest.raises(RuntimeError, match="at most"):
        launch_grid("LP", 1024, **kw)
    # Arm M: 128 <= num_tokens <= the architecture's owner scratch bound.
    assert launch_grid("M", 128, **kw) == 128
    assert launch_grid("M", 200, **kw) == 200
    assert launch_grid("M", 256, **kw) == 256
    m_max = ARM_M_MAX_TOKENS[compute_capability]
    assert m_max == (256 if compute_capability == (10, 0) else 2048)
    assert launch_grid("M", m_max, **kw) == min(m_max, cap)
    with pytest.raises(RuntimeError, match="admits"):
        launch_grid("M", m_max + 1, **kw)
    with pytest.raises(RuntimeError, match="admits"):
        launch_grid("M", 64, **kw)
    # Arm GW: CTAs-per-SM launch bound of the architecture (four on both).
    gw_ctas = ARM_GW_CTAS_PER_SM[compute_capability] * sm_count
    assert launch_grid("GW", 300, **kw) == 300
    assert launch_grid("GW", 4096, **kw) == min(4096, gw_ctas)
    assert launch_grid("GW", 8192, **kw) == gw_ctas
    # Arm Q4S: four CTAs per SM regardless of the architecture cap, bounded by
    # whole co-resident clusters, then rounded down to clusters.
    q4s_ctas = ARM_Q4S_CTAS_PER_SM * sm_count
    assert launch_grid("Q4S", 512, max_active_clusters=1000, **kw) == 512
    assert launch_grid("Q4S", 300, max_active_clusters=1000, **kw) == 300
    assert (
        launch_grid("Q4S", 1024, max_active_clusters=1000, **kw) == (q4s_ctas // 4) * 4
    )
    assert (
        launch_grid("Q4S", 2048, max_active_clusters=1000, **kw) == (q4s_ctas // 4) * 4
    )
    assert launch_grid("Q4S", 1024, max_active_clusters=100, **kw) == 400
    assert (
        launch_grid("Q4S", 1024, max_active_clusters=37, **kw) == 37 * ARM_Q4S_CLUSTER
    )
    with pytest.raises(RuntimeError, match="co-resident owner CTAs"):
        launch_grid(
            "Q4S", 1024, max_active_clusters=OWNER_CTAS // ARM_Q4S_CLUSTER - 1, **kw
        )
    with pytest.raises(RuntimeError, match="cluster capacity"):
        launch_grid("Q4S", 1024, **kw)
    # Arm Q4SP: the Q4S rule (four CTAs per SM, whole co-resident clusters).
    assert ARM_Q4S_FAMILY == ("Q4S", "Q4SP")
    assert launch_grid("Q4SP", 2048, max_active_clusters=1000, **kw) == launch_grid(
        "Q4S", 2048, max_active_clusters=1000, **kw
    )
    assert (
        launch_grid("Q4SP", 2048, max_active_clusters=37, **kw) == 37 * ARM_Q4S_CLUSTER
    )
    with pytest.raises(RuntimeError, match="cluster capacity"):
        launch_grid("Q4SP", 2048, **kw)
    with pytest.raises(RuntimeError):
        launch_grid("Q4SP", 2049, max_active_clusters=1000, **kw)
    with pytest.raises(RuntimeError):
        launch_grid("Q4SP", 64, max_active_clusters=1000, **kw)
    with pytest.raises(RuntimeError):
        launch_grid("Q4S", 64, max_active_clusters=1000, **kw)
    with pytest.raises(RuntimeError, match="exactly num_tokens"):
        launch_grid("LC", 32, **kw)
    with pytest.raises(RuntimeError, match="launch bound"):
        launch_grid("GW", 4096, compute_capability=(9, 0), sm_count=132)
    with pytest.raises(ValueError):
        launch_grid("A", 1, **kw)


def test_registry_keys_are_dispatch_arms():
    for arch, table in cake_jit.ROUTES.items():
        assert arch in SHAPE_ROUTES
        for key, name in table.items():
            arm, _, rows = key.partition(":")
            assert arm in cake_jit.ARMS
            assert (arm == "LC") == bool(rows)
            if rows:
                assert int(rows) in ARM_LC_TOKENS
            record = cake_jit.MODULES[name]
            assert record["arm"] == arm
            assert record["num_tokens"] == (int(rows) if rows else None)
            assert arch in record["arches"]
            assert record["launch"]["cooperative"] == (arm != "LC")
            cluster = record["launch"]["cluster"][0]
            if arm == "LC":
                assert cluster == int(rows)
            elif arm in ARM_Q4S_FAMILY:
                assert cluster == ARM_Q4S_CLUSTER
            else:
                assert cluster == 1
            # The occupancy query ships exactly with the cluster-bounded programs.
            assert cake_jit.queries_occupancy(name) == (
                arm in ARM_Q4S_FAMILY or key == "LC:16"
            )
            assert ("kernel_declaration" in record) == cake_jit.queries_occupancy(name)
    with pytest.raises(ValueError, match="per num_tokens"):
        cake_jit.kernel_key("LC")
    assert cake_jit.kernel_key("LC", 4) == "LC:4"
    assert cake_jit.kernel_key("GW") == "GW"


def test_max_route_blocks_and_plan_capacity():
    assert max_route_blocks(1, 8) == 16
    assert max_route_blocks(1, 16) == 16
    assert max_route_blocks(64, 8) == 896 + (1024 - 896) // 8
    assert max_route_blocks(8192, 16) == 896 + (131072 - 896) // 16
    assert max_route_blocks(100, 8) == 896 + (1600 - 896) // 8
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
    assert validate_kimi_k3_fused_router_inputs(logits[:3], bias, 8) == (3, 8)
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


def test_selection_tie_tolerance():
    logits, bias = make_inputs(5, torch.device("cpu"), seed=9)
    plan = reference_route(logits, bias, 8)
    assert assert_selection_matches(plan.topk_ids, logits, bias) == 0
    ranking = torch.sigmoid(logits) + bias.reshape(1, NUM_EXPERTS)
    order = torch.argsort(ranking[0], descending=True)
    sixteenth, seventeenth = int(order[15]), int(order[16])
    # Move the 17th candidate onto the 16th's biased score: either order is valid.
    tied = logits.clone()
    target = (
        torch.sigmoid(logits[0, sixteenth]) + bias[sixteenth] - bias[seventeenth]
    ).double()
    tied[0, seventeenth] = torch.logit(target).float()
    tied_ranking = torch.sigmoid(tied) + bias.reshape(1, NUM_EXPERTS)
    assert (
        abs(float(tied_ranking[0, sixteenth] - tied_ranking[0, seventeenth]))
        <= TIE_TOLERANCE
    )
    reference_ids = reference_route(tied, bias, 8).topk_ids
    kept, dropped = (
        (sixteenth, seventeenth)
        if sixteenth in reference_ids[0].tolist()
        else (seventeenth, sixteenth)
    )
    swapped = reference_ids.clone()
    row = [
        dropped if expert == kept else expert for expert in reference_ids[0].tolist()
    ]
    swapped[0] = torch.tensor(sorted(row), dtype=torch.int32)
    assert assert_selection_matches(swapped, tied, bias) == 1
    assert torch.equal(
        reference_route(tied, bias, 8, topk_ids=swapped).topk_ids, swapped
    )
    # A clearly lower-ranked expert is never admitted.
    wrong = plan.topk_ids.clone()
    lower = int(torch.argsort(ranking[1], descending=True)[40])
    row = [
        lower if expert == int(plan.topk_ids[1, 0]) else expert
        for expert in plan.topk_ids[1].tolist()
    ]
    wrong[1] = torch.tensor(sorted(row), dtype=torch.int32)
    with pytest.raises(AssertionError):
        assert_selection_matches(wrong, logits, bias)


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
    expected = expected_plan(plan, logits, bias, block_m)
    check_route_plan(plan, expected, num_tokens=num_tokens, block_m=block_m)
    return runner, logits, bias, plan


@pytest.mark.parametrize("num_tokens,block_m", MEASURED_SHAPES)
def test_fused_router_matches_reference(num_tokens, block_m):
    _run_and_check(num_tokens, block_m, seed=4568000 + num_tokens + block_m)


@pytest.mark.parametrize("num_tokens,block_m", ARBITRARY_SHAPES)
def test_fused_router_serves_arbitrary_token_counts(num_tokens, block_m):
    """Token counts between the measured cells route to an admitting arm."""
    _run_and_check(num_tokens, block_m, seed=7310000 + num_tokens + block_m)


def test_allocating_api_matches_reference():
    device = _require_program(64, 8)
    logits, bias = make_inputs(64, device, seed=11)
    plan = kimi_k3_fused_router(logits, bias, block_m=8)
    torch.cuda.synchronize()
    expected = expected_plan(plan, logits, bias, 8)
    assert torch.equal(plan.topk_ids, expected.topk_ids)
    torch.testing.assert_close(
        plan.topk_weights, expected.topk_weights, atol=WEIGHT_ATOL, rtol=WEIGHT_RTOL
    )
    extent = int(expected.num_tokens_post_padded[0].item())
    assert torch.equal(plan.sorted_token_ids[:extent], expected.sorted_token_ids)
    assert torch.equal(plan.expert_ids[: extent // 8], expected.expert_ids)
    assert torch.equal(plan.expert_counts, expected.expert_counts)
    assert torch.equal(plan.expert_offsets, expected.expert_offsets)


@pytest.mark.parametrize(
    "num_tokens,block_m",
    [(1, 8), (8, 16), (100, 8), (256, 8), (2048, 16), (3000, 8), (4096, 8)],
)
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
        expected = expected_plan(plan, logits, bias, block_m)
        check_route_plan(plan, expected, num_tokens=num_tokens, block_m=block_m)


def test_launch_makes_no_allocation():
    runner, _, _, _ = _run_and_check(1024, 8, seed=5)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] - before["allocation.all.allocated"] == 0


def test_prepare_caches_device_facts_and_occupancy():
    """A second preparation of a cluster-bounded shape performs no device query."""
    device = _require_program(1024, 8)
    logits, bias = make_inputs(1024, device, seed=13)
    first = prepare_kimi_k3_fused_router(logits, bias, block_m=8)
    facts_before = cake_backend._device_facts.cache_info().hits
    clusters_before = cake_backend._max_active_clusters.cache_info().hits
    second = prepare_kimi_k3_fused_router(logits, bias, block_m=8, plan=first.plan)
    assert second.grid_x == first.grid_x and second.arm in ARM_Q4S_FAMILY
    assert cake_backend._device_facts.cache_info().hits == facts_before + 1
    assert cake_backend._max_active_clusters.cache_info().hits == clusters_before + 1


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
    with pytest.raises(ValueError, match="one CUDA device"):
        prepare_kimi_k3_fused_router(logits, bias.cpu(), block_m=8)
    with pytest.raises(ValueError, match="backend"):
        prepare_kimi_k3_fused_router(logits, bias, block_m=8, backend="triton")
