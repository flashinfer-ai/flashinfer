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
    ARM_GW_CLUSTER,
    ARM_GW_CTAS_PER_SM,
    ARM_L_FAMILY,
    ARM_L_MAX_TOKENS,
    ARM_L_MIN_GRID,
    ARM_LC1_NC,
    ARM_LC_TOKENS,
    ARM_LCG_CLUSTER,
    ARM_LCG_FAMILY,
    ARM_LCG_MAX_TOKENS,
    ARM_M_MAX_TOKENS,
    ARM_Q4S_CLUSTER,
    ARM_Q4S_CTAS_PER_SM,
    ARM_Q4S_FAMILY,
    BLOCK_M_VALUES,
    CAPACITY_PROGRAM_ARM,
    CAPACITY_ROUTES,
    CELL_LOW_ROUTES,
    MEASURED_NUM_TOKENS,
    NUM_EXPERTS,
    OWNER_CTAS,
    SHAPE_ROUTES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    TOP_K,
    CapacityRule,
    KimiK3RoutePlan,
    allocate_kimi_k3_route_plan,
    capacity_arm,
    launch_grid,
    max_route_blocks,
    persistent_grid_cap,
    route_arm,
    route_arm_candidates,
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
# Tensors the kernels access 16 bytes at a time; each must be 16-byte aligned.
ALIGNED_TENSORS = (
    "logits",
    "bias",
    "topk_ids",
    "expert_counts",
    "expert_offsets",
    "expert_scatter_offsets",
)

# The 28 measured cells: num_tokens in {1, 2, ..., 8192} x block_m in {8, 16}.
MEASURED_SHAPES = [(rows, bm) for bm in BLOCK_M_VALUES for rows in MEASURED_NUM_TOKENS]
# Token counts between the measured cells, one or two per arm range (including
# the first count above every cell boundary up to the 8192-token limit), both
# sides of the 32-token cell's split on SM100 at block_m 16 (28 / 29), both
# sides of the 512-token cell's split on SM103 (383 / 384), the first count
# above the 512-token cell (513), and both sides of the capacity
# rule's thresholds on a 148-SM B200 (C = 568) and a 152-SM GB300 (C = 592): C,
# the ends of the local-rank bands (832 / 833 and 1055 / 1056 on the B200,
# 1039 / 1040 on the GB300), the end of the 1024-token bands (1025), 2C (1136 /
# 1184) and 2C + W, the start of the wide arm (1200 on the B200, 1184 on the
# GB300 at block_m 8, 1248 at block_m 16).
ARBITRARY_NUM_TOKENS = (
    3,
    5,
    17,
    28,
    29,
    31,
    33,
    100,
    129,
    200,
    257,
    300,
    383,
    384,
    513,
    568,
    569,
    592,
    593,
    832,
    833,
    1000,
    1025,
    1039,
    1040,
    1055,
    1056,
    1135,
    1136,
    1183,
    1184,
    1199,
    1200,
    1247,
    1248,
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


def misaligned_view(tensor):
    """Contiguous view shaped like ``tensor`` whose start is not 16-byte aligned."""
    flat = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
    view = flat[1:].view(tensor.shape)
    assert view.is_contiguous() and view.data_ptr() % 16 != 0
    return view


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
        arms = {"L", "LC", "LC1_NC", "LCG_LW", "M", "Q4S", "Q4S_E16", "Q4SP", "GW"}
        # LPC serves sm_100a's 64-token cells, LP sm_103a's (64, 8) and (128, 8);
        # LCG serves sm_103a's (32, 8) cell (and sm_100a's counts up to 28 at
        # block_m 16).
        arms.update(("LP", "LCG") if arch == "sm_103a" else ("LPC",))
        assert set(table.values()) == arms
    # The two tables differ in exactly three cells of the L family and in the
    # 32-token cell at block_m 8 (LCG / LCG_LW).
    differing = {
        key
        for key in SHAPE_ROUTES["sm_100a"]
        if SHAPE_ROUTES["sm_100a"][key] != SHAPE_ROUTES["sm_103a"][key]
    }
    assert differing == {(32, 8), (64, 8), (64, 16), (128, 8)}
    assert ARM_LCG_FAMILY == ("LCG", "LCG_LW")
    assert CELL_LOW_ROUTES == {
        ("sm_100a", 32, 16): (28, "LCG"),
        ("sm_103a", 512, 8): (383, "Q4S"),
        ("sm_103a", 512, 16): (383, "Q4S"),
    }
    # One token: LC at block_m 8, LC1_NC (the same kernel with the read-only
    # loads) at block_m 16, on both architectures.
    assert ARM_LC1_NC == "LC1_NC"
    for arch in SHAPE_ROUTES:
        assert route_arm(arch, 1, 8) == "LC" and route_arm(arch, 1, 16) == "LC1_NC"
    assert route_arm("sm_100a", 32, 8) == "LCG_LW"
    assert route_arm("sm_100a", 32, 16) == "LCG_LW"
    assert route_arm("sm_100a", 29, 16) == "LCG_LW"
    assert route_arm("sm_100a", 28, 16) == "LCG"
    assert route_arm("sm_103a", 32, 8) == "LCG"
    assert route_arm("sm_103a", 32, 16) == "LCG_LW"
    assert route_arm("sm_103a", 28, 16) == "LCG_LW"
    for arch in SHAPE_ROUTES:
        assert route_arm(arch, 512, 8) == "Q4S_E16"
        assert route_arm(arch, 512, 16) == "Q4S_E16"
    assert route_arm("sm_100a", 64, 8) == "LPC"
    assert route_arm("sm_103a", 64, 8) == "LP"
    assert route_arm("sm_100a", 64, 16) == "LPC"
    assert route_arm("sm_103a", 64, 16) == "L"
    assert route_arm("sm_100a", 128, 8) == "L"
    assert route_arm("sm_103a", 128, 8) == "LP"
    for arch in SHAPE_ROUTES:
        # Arm LC serves exactly the LC row counts (one kernel per token count;
        # one token at block_m 16 takes LC1_NC).
        assert set(ARM_LC_TOKENS) == {
            rows for (rows, _), arm in SHAPE_ROUTES[arch].items() if arm == "LC"
        }
        assert {
            cell for cell, arm in SHAPE_ROUTES[arch].items() if arm == ARM_LC1_NC
        } == {(1, 16)}
        for rows in ARM_LC_TOKENS:
            for bm in BLOCK_M_VALUES:
                expected = ARM_LC1_NC if (rows, bm) == (1, 16) else "LC"
                assert route_arm(arch, rows, bm) == expected
        assert {
            rows for (rows, _), arm in SHAPE_ROUTES[arch].items() if arm in ARM_L_FAMILY
        } == {64, 128}
        # Arms LCG and LCG_LW (one 16-CTA cluster of two row groups) serve the
        # 32-token cell: each architecture runs both of them in it.
        for arm in ARM_LCG_FAMILY:
            cells = {rows for (rows, _), a in SHAPE_ROUTES[arch].items() if a == arm}
            assert cells <= {ARM_LCG_MAX_TOKENS}
        assert {
            route_arm(arch, rows, bm) for rows in (3, 32) for bm in BLOCK_M_VALUES
        } == set(ARM_LCG_FAMILY)
        assert route_arm(arch, 128, 16) == "L"
        assert route_arm(arch, 256, 16) == "M"
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
    # Counts below 32 that are not an LC row take the 32-token cell (LCG or
    # LCG_LW, by architecture and block_m; on sm_100a at block_m 16, LCG up to
    # 28 tokens and LCG_LW from 29).
    for rows in (3, 5, 6, 7, 9, 15, 17, 28, 29, 31):
        assert route_cell(rows) == 32
        for arch in SHAPE_ROUTES:
            for bm in BLOCK_M_VALUES:
                low = CELL_LOW_ROUTES.get((arch, 32, bm))
                expected = SHAPE_ROUTES[arch][(32, bm)]
                if low is not None and rows <= low[0]:
                    expected = low[1]
                assert route_arm(arch, rows, bm) == expected
        assert route_arm("sm_100a", rows, 8) == "LCG_LW"
        assert route_arm("sm_100a", rows, 16) == ("LCG" if rows <= 28 else "LCG_LW")
        assert route_arm("sm_103a", rows, 8) == "LCG"
        assert route_arm("sm_103a", rows, 16) == "LCG_LW"
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
        assert route_arm(arch, 384, 8) == "Q4S_E16"
        assert route_arm(arch, 513, 8) == "Q4S"
        assert route_arm(arch, 1000, 16) == "Q4S"
        assert route_arm(arch, 2047, 8) == "Q4SP"
        assert route_arm(arch, 3000, 16) == "GW"
    # The 512-token cells: Q4S_E16 on SM100; on SM103 Q4S up to 383 tokens.
    for bm in BLOCK_M_VALUES:
        for rows in (257, 300, 383):
            assert route_arm("sm_100a", rows, bm) == "Q4S_E16"
            assert route_arm("sm_103a", rows, bm) == "Q4S"
        for rows in (384, 512):
            assert route_arm("sm_103a", rows, bm) == "Q4S_E16"
    assert route_arm("sm_100a", 33, 8) == "LPC"
    assert route_arm("sm_103a", 33, 8) == "LP"
    assert route_arm("sm_100a", 33, 16) == "LPC"
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
        assert launch_grid("LC", rows, max_active_clusters=1, **kw) == rows
    # Arm LC1_NC: one CTA for exactly one token.
    assert launch_grid(ARM_LC1_NC, 1, **kw) == 1
    assert launch_grid(ARM_LC1_NC, 1, max_active_clusters=1, **kw) == 1
    for rows in (2, 3, 16):
        with pytest.raises(RuntimeError, match="exactly one token"):
            launch_grid(ARM_LC1_NC, rows, **kw)
    # Arms LCG and LCG_LW: one cluster of 16 CTAs for any count up to 32.
    assert ARM_LCG_CLUSTER == 16 and ARM_LCG_MAX_TOKENS == 32
    for arm in ARM_LCG_FAMILY:
        for rows in (1, 3, 17, 31, 32):
            assert launch_grid(arm, rows, **kw) == ARM_LCG_CLUSTER
            assert (
                launch_grid(arm, rows, max_active_clusters=1, **kw) == ARM_LCG_CLUSTER
            )
        with pytest.raises(RuntimeError, match="at most 32"):
            launch_grid(arm, 33, **kw)
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
    # Arms LP and LPC: the L rule (same kernel family, same admission guard).
    assert ARM_L_FAMILY == ("L", "LP", "LPC")
    for arm in ARM_L_FAMILY[1:]:
        for rows in (3, 32, 64, 100, 128):
            assert launch_grid(arm, rows, **kw) == launch_grid("L", rows, **kw)
        with pytest.raises(RuntimeError, match="at most"):
            launch_grid(arm, 1024, **kw)
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
    # Arm GW: whole CTA pairs under the CTAs-per-SM launch bound of the
    # architecture (four on both) and the co-resident pair capacity.
    assert ARM_GW_CLUSTER == 2
    gw_ctas = ARM_GW_CTAS_PER_SM[compute_capability] * sm_count
    gw = dict(max_active_clusters=1000, **kw)
    assert launch_grid("GW", 300, **gw) == 300
    assert launch_grid("GW", 301, **gw) == 302
    assert launch_grid("GW", 4096, **gw) == min(4096, gw_ctas)
    assert launch_grid("GW", 8192, **gw) == gw_ctas
    assert launch_grid("GW", 8192, max_active_clusters=100, **kw) == 200
    assert launch_grid("GW", 8192, max_active_clusters=1, **kw) == ARM_GW_CLUSTER
    for rows in (300, 301, 4096, 8192):
        for clusters in (1, 37, 100, 1000):
            grid = launch_grid("GW", rows, max_active_clusters=clusters, **kw)
            assert grid % ARM_GW_CLUSTER == 0
            assert grid <= min(gw_ctas, clusters * ARM_GW_CLUSTER)
    with pytest.raises(RuntimeError, match="cluster capacity"):
        launch_grid("GW", 4096, **kw)
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
    # Arms Q4S_E16, Q4SB, Q4SP, Q4SP_PG, Q4SP_LR and Q4SPW_LR: the Q4S rule
    # (four CTAs per SM, whole co-resident clusters).
    assert ARM_Q4S_FAMILY == (
        "Q4S",
        "Q4S_E16",
        "Q4SB",
        "Q4SP",
        "Q4SP_PG",
        "Q4SP_LR",
        "Q4SPW_LR",
    )
    for arm in ARM_Q4S_FAMILY[1:]:
        for rows in (300, 1024, 2048):
            assert launch_grid(
                arm, rows, max_active_clusters=1000, **kw
            ) == launch_grid("Q4S", rows, max_active_clusters=1000, **kw)
        assert (
            launch_grid(arm, 2048, max_active_clusters=37, **kw) == 37 * ARM_Q4S_CLUSTER
        )
        with pytest.raises(RuntimeError, match="cluster capacity"):
            launch_grid(arm, 2048, **kw)
    with pytest.raises(RuntimeError):
        launch_grid("Q4SP", 2049, max_active_clusters=1000, **kw)
    with pytest.raises(RuntimeError):
        launch_grid("Q4SP", 64, max_active_clusters=1000, **kw)
    with pytest.raises(RuntimeError):
        launch_grid("Q4S", 64, max_active_clusters=1000, **kw)
    with pytest.raises(RuntimeError, match="exactly num_tokens"):
        launch_grid("LC", 32, **kw)
    with pytest.raises(RuntimeError, match="launch bound"):
        launch_grid(
            "GW", 4096, compute_capability=(9, 0), sm_count=132, max_active_clusters=1
        )
    with pytest.raises(ValueError):
        launch_grid("A", 1, **kw)


@pytest.mark.parametrize(
    "compute_capability,sm_count,admits_m",
    [
        ((10, 0), 42, False),
        ((10, 0), 43, True),
        ((10, 3), 31, False),
        ((10, 3), 32, True),
    ],
)
def test_launch_grid_small_devices(compute_capability, sm_count, admits_m):
    cap = persistent_grid_cap(compute_capability, sm_count)
    kw = dict(compute_capability=compute_capability, sm_count=sm_count)
    # Arm M: CTA b < OWNER_CTAS alone publishes experts 7b to 7b + 6, so a
    # device whose persistent grid cap is below OWNER_CTAS cannot run it.
    assert (cap >= OWNER_CTAS) == admits_m
    if admits_m:
        assert launch_grid("M", 256, **kw) == min(256, cap)
    else:
        with pytest.raises(RuntimeError, match="co-resident owner CTAs"):
            launch_grid("M", 256, **kw)
    # Arms L / LP / LPC and GW stride their owners over the experts, so any grid
    # publishes the whole plan.
    for arm in ARM_L_FAMILY:
        assert launch_grid(arm, 128, **kw) == min(ARM_L_MIN_GRID, cap)
    gw_ctas = ARM_GW_CTAS_PER_SM[compute_capability] * sm_count
    assert launch_grid("GW", 4096, max_active_clusters=1000, **kw) == min(4096, gw_ctas)
    # A device holding one CTA pair per SM halves the GW grid.
    assert (
        launch_grid("GW", 4096, max_active_clusters=sm_count, **kw)
        == ARM_GW_CLUSTER * sm_count
    )


@pytest.mark.parametrize("compute_capability", [(10, 0), (10, 3)])
def test_launch_grid_refuses_missing_clusters(compute_capability):
    """A device (or MIG slice) that admits too few clusters is refused, not hung."""
    kw = dict(compute_capability=compute_capability, sm_count=148)
    # The one-cluster arms need the driver to admit one cluster.
    for arm, rows in (
        ("LC", 16),
        ("LC", 8),
        ("LC1_NC", 1),
        ("LCG", 32),
        ("LCG", 3),
        ("LCG_LW", 32),
        ("LCG_LW", 3),
    ):
        with pytest.raises(RuntimeError, match="co-resident clusters"):
            launch_grid(arm, rows, max_active_clusters=0, **kw)
    # Arm GW needs at least one co-resident CTA pair.
    with pytest.raises(RuntimeError, match="2-CTA clusters"):
        launch_grid("GW", 4096, max_active_clusters=0, **kw)
    assert launch_grid("GW", 4096, max_active_clusters=1, **kw) == ARM_GW_CLUSTER
    # The Q4S family needs the 128 plan-owner CTAs co-resident.
    for arm in ARM_Q4S_FAMILY:
        with pytest.raises(RuntimeError, match="co-resident owner CTAs"):
            launch_grid(
                arm, 2048, max_active_clusters=OWNER_CTAS // ARM_Q4S_CLUSTER - 1, **kw
            )
        assert (
            launch_grid(
                arm, 2048, max_active_clusters=OWNER_CTAS // ARM_Q4S_CLUSTER, **kw
            )
            == OWNER_CTAS
        )


def test_capacity_rule_routes_q4s_cells():
    """Q4S-family cells follow the batch against C, the device's Q4S-family cap."""
    assert CAPACITY_PROGRAM_ARM == "Q4SP"
    sm100 = ((832, "Q4SP_LR"), (1055, "Q4SP"), (2048, "Q4SP_LR"))
    sm100_bm8 = (
        (832, "Q4SP_LR"),
        (1024, "Q4SP_PG"),
        (1055, "Q4SP"),
        (2048, "Q4SP_LR"),
    )
    sm103 = ((1024, "Q4SB"), (1039, "Q4SP"), (2048, "Q4SP_LR"))
    assert {
        ("sm_100a", 8): CapacityRule(sm100_bm8, "Q4SPW_LR", 64),
        ("sm_100a", 16): CapacityRule(sm100, "Q4SPW_LR", 64),
        ("sm_103a", 8): CapacityRule(sm103, "Q4SPW_LR", 0),
        ("sm_103a", 16): CapacityRule(sm103, "Q4SPW_LR", 64),
    } == CAPACITY_ROUTES
    # GB300 (152 SMs): C = 592; Q4S up to 383 tokens, Q4S_E16 up to 512 and Q4S
    # up to C; above
    # C Q4SB up to 1024 tokens, Q4SP up to 1039 and Q4SP_LR from 1040; the wide
    # arm starts at 2C = 1184 at block_m 8 and at 2C + 64 = 1248 at block_m 16.
    gb300 = {
        256: "M",
        257: "Q4S",
        383: "Q4S",
        384: "Q4S_E16",
        512: "Q4S_E16",
        513: "Q4S",
        592: "Q4S",
        593: "Q4SB",
        1024: "Q4SB",
        1025: "Q4SP",
        1039: "Q4SP",
        1040: "Q4SP_LR",
        1183: "Q4SP_LR",
        1184: "Q4SPW_LR",
        2048: "Q4SPW_LR",
        2049: "GW",
        8192: "GW",
    }
    wide16 = {1184: "Q4SP_LR", 1247: "Q4SP_LR", 1248: "Q4SPW_LR"}
    for bm in BLOCK_M_VALUES:
        expected = {**gb300, **(wide16 if bm == 16 else {})}
        for rows, arm in expected.items():
            got = route_arm("sm_103a", rows, bm, cluster_capacity=592)
            assert got == arm, (bm, rows)
        # Without a capacity the cell's arm (the arm of num_tokens <= C) remains.
        assert route_arm("sm_103a", 1184, bm) == "Q4SP"
        assert route_arm("sm_103a", 593, bm) == "Q4S"
    # B200 (148 SMs): C = 568; Q4S_E16 up to 512 tokens and Q4S up to C; above
    # C Q4SP_LR up to 832 tokens, Q4SP up to 1055 (at block_m 8 Q4SP_PG up to
    # 1024 tokens) and Q4SP_LR from 1056; at both block_m values the wide arm
    # Q4SPW_LR starts at 2C + 64 = 1200.
    b200 = {
        256: "M",
        257: "Q4S_E16",
        512: "Q4S_E16",
        513: "Q4S",
        568: "Q4S",
        569: "Q4SP_LR",
        832: "Q4SP_LR",
        833: "Q4SP",
        1024: "Q4SP",
        1025: "Q4SP",
        1055: "Q4SP",
        1056: "Q4SP_LR",
        1135: "Q4SP_LR",
        1136: "Q4SP_LR",
        1199: "Q4SP_LR",
        1200: "Q4SPW_LR",
        2048: "Q4SPW_LR",
        2049: "GW",
        8192: "GW",
    }
    pass_guard8 = {833: "Q4SP_PG", 1000: "Q4SP_PG", 1024: "Q4SP_PG"}
    for bm in BLOCK_M_VALUES:
        expected = {**b200, **(pass_guard8 if bm == 8 else {})}
        for rows, arm in expected.items():
            got = route_arm("sm_100a", rows, bm, cluster_capacity=568)
            assert got == arm, (bm, rows)
        assert route_arm("sm_100a", 1200, bm) == "Q4SP"
        assert route_arm("sm_100a", 569, bm) == "Q4S"
        assert route_arm("sm_100a", 512, bm) == "Q4S_E16"
    # A small slice moves to the wide arm early; other arms never change.
    rule8 = CAPACITY_ROUTES[("sm_103a", 8)]
    rule16 = CAPACITY_ROUTES[("sm_103a", 16)]
    assert capacity_arm("Q4S", 128, 128, rule8) == "Q4S"
    assert capacity_arm("Q4S", 200, 128, rule8) == "Q4SB"
    assert capacity_arm("Q4S", 256, 128, rule8) == "Q4SPW_LR"
    assert capacity_arm("Q4S", 300, 128, rule16) == "Q4SB"
    assert capacity_arm("Q4S", 320, 128, rule16) == "Q4SPW_LR"
    sm100_rule8 = CAPACITY_ROUTES[("sm_100a", 8)]
    sm100_rule16 = CAPACITY_ROUTES[("sm_100a", 16)]
    assert capacity_arm("Q4S_E16", 300, 512, sm100_rule8) == "Q4S_E16"
    assert capacity_arm("Q4S_E16", 300, 128, sm100_rule8) == "Q4SP_LR"
    assert capacity_arm("Q4S_E16", 300, 128, sm100_rule16) == "Q4SP_LR"
    assert capacity_arm("Q4S_E16", 320, 128, sm100_rule16) == "Q4SPW_LR"
    assert capacity_arm("Q4S", 900, 600, sm100_rule8) == "Q4SP_PG"
    assert capacity_arm("Q4S", 900, 600, sm100_rule16) == "Q4SP"
    # Above the last band's bound the last band's arm remains.
    narrow = CapacityRule(((1050, "Q4SB"),), "Q4SPW_LR", 0)
    assert capacity_arm("Q4SP", 1100, 1024, narrow) == "Q4SB"
    for arm in ("L", "LP", "LPC", "LC", "LC1_NC", "LCG", "LCG_LW", "M", "GW"):
        assert capacity_arm(arm, 4096, 1, rule8) == arm
    with pytest.raises(RuntimeError, match="admits none"):
        capacity_arm("Q4S", 300, 0, rule8)
    assert rule8.arms() == ("Q4SB", "Q4SP", "Q4SP_LR", "Q4SPW_LR")
    assert sm100_rule8.arms() == ("Q4SP_LR", "Q4SP_PG", "Q4SP", "Q4SPW_LR")
    assert sm100_rule16.arms() == ("Q4SP_LR", "Q4SP", "Q4SPW_LR")
    # Availability asks for every arm a shape may take on some device.
    assert route_arm_candidates("sm_103a", 1024, 8) == (
        "Q4S",
        "Q4SB",
        "Q4SP",
        "Q4SP_LR",
        "Q4SPW_LR",
    )
    assert route_arm_candidates("sm_103a", 2048, 16) == (
        "Q4SP",
        "Q4SB",
        "Q4SP_LR",
        "Q4SPW_LR",
    )
    assert route_arm_candidates("sm_103a", 512, 8) == (
        "Q4S_E16",
        "Q4SB",
        "Q4SP",
        "Q4SP_LR",
        "Q4SPW_LR",
    )
    assert route_arm_candidates("sm_103a", 128, 8) == ("LP",)
    assert route_arm_candidates("sm_100a", 64, 8) == ("LPC",)
    assert route_arm_candidates("sm_100a", 1024, 8) == (
        "Q4S",
        "Q4SP_LR",
        "Q4SP_PG",
        "Q4SP",
        "Q4SPW_LR",
    )
    assert route_arm_candidates("sm_100a", 512, 8) == (
        "Q4S_E16",
        "Q4SP_LR",
        "Q4SP_PG",
        "Q4SP",
        "Q4SPW_LR",
    )
    assert route_arm_candidates("sm_100a", 1024, 16) == (
        "Q4S",
        "Q4SP_LR",
        "Q4SP",
        "Q4SPW_LR",
    )
    assert route_arm_candidates("sm_100a", 2048, 16) == ("Q4SP", "Q4SP_LR", "Q4SPW_LR")
    assert route_arm_candidates("sm_100a", 28, 16) == ("LCG",)


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
            assert record["launch"]["cooperative"] == (
                arm not in ("LC", ARM_LC1_NC) and arm not in ARM_LCG_FAMILY
            )
            cluster = record["launch"]["cluster"][0]
            if arm == "LC":
                assert cluster == int(rows)
            elif arm in ARM_LCG_FAMILY:
                assert cluster == ARM_LCG_CLUSTER
            elif arm in ARM_Q4S_FAMILY:
                assert cluster == ARM_Q4S_CLUSTER
            elif arm == "GW":
                assert cluster == ARM_GW_CLUSTER
            else:
                assert cluster == 1
            # The occupancy query ships exactly with the cluster-bounded programs.
            assert cake_jit.queries_occupancy(name) == (
                arm in (*ARM_Q4S_FAMILY, *ARM_LCG_FAMILY, "GW") or key == "LC:16"
            )
            assert ("kernel_declaration" in record) == cake_jit.queries_occupancy(name)
    # Every arm a route table, a capacity rule or a 32-token split names is
    # registered, and only those: the package ships no program that no cell
    # dispatches to.
    for arch, table in SHAPE_ROUTES.items():
        keys = {
            cake_jit.kernel_key(arm, rows if arm == "LC" else None)
            for (rows, _), arm in table.items()
        }
        for (rule_arch, _), rule in CAPACITY_ROUTES.items():
            if rule_arch == arch:
                keys.update(rule.arms())
        for (low_arch, _, _), (_, low_arm) in CELL_LOW_ROUTES.items():
            if low_arch == arch:
                keys.add(low_arm)
        assert keys == set(cake_jit.ROUTES[arch]), arch
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
    # A row slice stays 16-byte aligned (one row is 3584 bytes).
    assert validate_kimi_k3_fused_router_inputs(logits[5:], bias, 8) == (11, 8)
    with pytest.raises(ValueError, match="logits must be 16-byte aligned"):
        validate_kimi_k3_fused_router_inputs(misaligned_view(logits), bias, 8)
    with pytest.raises(ValueError, match="bias must be 16-byte aligned"):
        validate_kimi_k3_fused_router_inputs(logits, misaligned_view(bias), 8)


def test_plan_requires_vector_alignment():
    """Only the plan tensors the kernels access 16 bytes at a time need alignment."""
    device = torch.device("cpu")
    plan = allocate_kimi_k3_route_plan(16, 8, device)
    cake_backend._validate_plan(plan, num_tokens=16, block_m=8, device=device)
    for name in KimiK3RoutePlan._fields:
        shifted = plan._replace(**{name: misaligned_view(getattr(plan, name))})
        if name in ALIGNED_TENSORS:
            with pytest.raises(ValueError, match=f"plan.{name} must be 16-byte"):
                cake_backend._validate_plan(
                    shifted, num_tokens=16, block_m=8, device=device
                )
        else:
            cake_backend._validate_plan(
                shifted, num_tokens=16, block_m=8, device=device
            )


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
    capacity = cake_backend.route_capacity(device, block_m)
    assert runner.arm == route_arm(
        _gpu_arch(), num_tokens, block_m, cluster_capacity=capacity
    )
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


def test_route_capacity_follows_the_device():
    device = _require_program(1024, 8)
    for bm in BLOCK_M_VALUES:
        capacity = cake_backend.route_capacity(device, bm)
        if (_gpu_arch(), bm) in CAPACITY_ROUTES:
            # C is whole 4-CTA clusters and at least the plan owners.
            assert capacity is not None
            assert capacity % ARM_Q4S_CLUSTER == 0 and capacity >= OWNER_CTAS
        else:
            assert capacity is None


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


def test_prepare_rejects_misaligned_tensors():
    """Preparation rejects each misaligned vector-accessed tensor before a launch."""
    device = _require_program(1, 8)
    logits, bias = make_inputs(1, device, seed=17)
    plan = allocate_kimi_k3_route_plan(1, 8, device)
    runner = prepare_kimi_k3_fused_router(logits, bias, block_m=8, plan=plan)
    assert runner.plan is plan
    for name in ALIGNED_TENSORS:
        inputs = {"logits": logits, "bias": bias}
        bound = plan
        if name in inputs:
            inputs[name] = misaligned_view(inputs[name])
            message = f"{name} must be 16-byte aligned"
        else:
            bound = plan._replace(**{name: misaligned_view(getattr(plan, name))})
            message = f"plan.{name} must be 16-byte aligned"
        with pytest.raises(ValueError, match=message):
            prepare_kimi_k3_fused_router(**inputs, block_m=8, plan=bound)
